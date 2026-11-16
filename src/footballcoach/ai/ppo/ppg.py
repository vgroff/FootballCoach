"""PPOTrainer's PPG value refit (standalone and interleaved into PPO), its KL anchor, and the flip_y consistency refit."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from footballcoach.ai.action.distributions import MaskedCategorical, _von_mises_kl
from footballcoach.ai.obs.augment import augment_batch
from footballcoach.ai.ppo import consistency as _consistency
from footballcoach.ai.ppo.trainer_common import _load_state_dict_tolerant, log
from footballcoach.ai.ppo.trainer_common import _ai_types, _split_batch_releasing
from footballcoach.ai.ppo.trainer_common import _lr_warmup_scale


_AI_TYPE_NAMES = ("rules", "immobile", "neural")  # order of obs/schema.py's AI-type one-hot

def _episode_phase_fraction(track_ids: list, dones: np.ndarray, mc: np.ndarray) -> np.ndarray:
    """Per-row position within its episode as ``(step_in_episode + 1) /
    episode_length`` in (0, 1] (NaN for unfinished-tail rows). Rows of one
    track are chronological; a new episode starts after a ``done=1`` row, and
    ALSO where an unfinished tail (NaN ``mc``) is followed by a finite row --
    that is the seam between two concatenated per-env buffers, which never
    carries a done marker of its own."""
    n = len(dones)
    frac = np.full(n, np.nan, dtype=np.float64)
    tracks = np.asarray(track_ids, dtype=object)
    for t in set(track_ids):
        idx = np.nonzero(tracks == t)[0]
        if len(idx) == 0:
            continue
        d = dones[idx] > 0.5
        nan = np.isnan(mc[idx])
        is_start = np.zeros(len(idx), dtype=bool)
        is_start[0] = True
        is_start[1:] |= d[:-1]
        is_start[1:] |= nan[:-1] & ~nan[1:]
        ar = np.arange(len(idx))
        start_pos = np.maximum.accumulate(np.where(is_start, ar, 0))
        ep_id = np.cumsum(is_start) - 1
        length = np.bincount(ep_id)
        f = (ar - start_pos + 1) / length[ep_id]
        f[nan] = np.nan
        frac[idx] = f
    return frac

def _value_loss_breakdown(
    preds: dict, mc, outcomes: list, track_ids: list, dones, other_type, ret_std: float,
    n_worst_rows: int = 5, phase_frac=None,
) -> list[str]:
    """Human-readable breakdown of where the value error lives, for
    ``ppg_value_refit``'s opt-in ``bc.ppg_loss_diagnostics``. Pure function
    over per-row arrays (unit-tested without any network).

    ``preds``: ``{"current": (N,), "original": (N,) optional}`` value
    predictions; ``mc``: pure-MC returns with NaN on unfinished tails (only
    finite rows are scored, so every figure is against value-net-free
    targets); ``outcomes``/``track_ids``: per-row lists ('' outcome = unfinished);
    ``dones``: (N,); ``other_type``: (N,) int index into ``_AI_TYPE_NAMES``
    of the OTHER player as seen from the row's owner, or -1 if unknown (may be
    None). All errors are normalised by ``ret_std**2`` like every other loss in
    the refit logs. Returns log lines."""
    mc = np.asarray(mc, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    valid = np.isfinite(mc)
    var_norm = float(ret_std) ** 2
    P = {k: np.asarray(v, dtype=np.float64) for k, v in preds.items()}
    cur = P["current"]
    orig = P.get("original")
    n_valid = int(valid.sum())
    lines: list[str] = []
    if n_valid == 0:
        return ["  [ppg loss diag] no rows with finished-episode MC targets"]

    def _mse(p, m):
        return float(np.mean((p[m] - mc[m]) ** 2)) / var_norm

    tgt = mc[valid]
    tot_var = float(np.var(tgt))
    mse_cur = _mse(cur, valid)
    lines.append(
        f"  [ppg loss diag] {n_valid:,} rows with MC targets | target mean={tgt.mean():+.3f} std={tgt.std():.3f} | "
        f"current: pred mean={cur[valid].mean():+.3f} std={cur[valid].std():.3f} bias={cur[valid].mean() - tgt.mean():+.3f} "
        f"mse={mse_cur:.4f} R2={1.0 - mse_cur * var_norm / max(tot_var, 1e-12):.3f}"
        + (f" | original mse={_mse(orig, valid):.4f}" if orig is not None else "")
    )

    header = (
        f"    {'group':<34}{'n':>9}{'share':>7}{'tgt mean':>10}{'tgt std':>9}"
        f"{'cur pred':>10}{'bias':>8}{'cur mse':>9}{'SSE%':>7}" + (f"{'orig mse':>10}" if orig is not None else "")
    )
    sse_total = float(np.sum((cur[valid] - mc[valid]) ** 2))

    def _table(title: str, labels: np.ndarray, order=None) -> None:
        lines.append(f"  [ppg loss diag] by {title}:")
        lines.append(header)
        uniq = order if order is not None else sorted({str(x) for x in labels[valid]})
        for u in uniq:
            m = valid & (labels == u)
            k = int(m.sum())
            if k == 0:
                continue
            sse = float(np.sum((cur[m] - mc[m]) ** 2))
            row = (
                f"    {str(u):<34}{k:>9,}{100.0 * k / n_valid:>6.1f}%{mc[m].mean():>+10.3f}{mc[m].std():>9.3f}"
                f"{cur[m].mean():>+10.3f}{cur[m].mean() - mc[m].mean():>+8.3f}{_mse(cur, m):>9.4f}"
                f"{100.0 * sse / max(sse_total, 1e-12):>6.1f}%"
            )
            if orig is not None:
                row += f"{_mse(orig, m):>10.4f}"
            lines.append(row)

    trk = np.asarray(track_ids, dtype=object)
    # The outcome label is the TRAINEE's view of the episode, so the opponent
    # track's rows in a 'box_possession' episode are the losing side -- always
    # split the outcome by the row owner's track or the groups are meaningless.
    out_lab = np.asarray(
        [f"{o if o else '?'} | {t}" for o, t in zip(outcomes, track_ids)], dtype=object,
    )
    _table("episode outcome (the trainee's view) x row owner", out_lab)

    if other_type is not None:
        ot = np.asarray(other_type)
        names = np.asarray(
            [_AI_TYPE_NAMES[i] if 0 <= i < len(_AI_TYPE_NAMES) else "?" for i in ot], dtype=object,
        )
        combo = np.asarray([f"{t} vs {n}" for t, n in zip(trk, names)], dtype=object)
        _table("row owner (track) vs the other player's AI type", combo)
    else:
        _table("track (trainee / opponent)", trk)

    if phase_frac is None:
        phase_frac = _episode_phase_fraction(list(track_ids), dones, mc)
    pf = np.asarray(phase_frac, dtype=np.float64)
    bins = np.digitize(np.where(np.isfinite(pf), pf, 0.0), [0.25, 0.5, 0.75, 1.0000001, 9.0], right=True)
    names_q = ["0-25% of episode", "25-50% of episode", "50-75% of episode", "75-100% of episode"]
    plab = np.asarray([names_q[min(int(b), 3)] for b in bins], dtype=object)
    plab[~np.isfinite(pf)] = "?"
    _table("position within the episode", plab, order=names_q)

    # Error concentration.
    sq = (cur[valid] - tgt) ** 2
    srt = np.sort(sq)[::-1]
    csum = np.cumsum(srt) / max(float(srt.sum()), 1e-12)
    top1 = float(csum[max(int(0.01 * len(srt)) - 1, 0)])
    top10 = float(csum[max(int(0.10 * len(srt)) - 1, 0)])
    lines.append(
        f"  [ppg loss diag] error concentration: worst 1% of rows = {100 * top1:.0f}% of total squared error, "
        f"worst 10% = {100 * top10:.0f}%"
    )

    # Yardstick: how much variance would be left if the episode's outcome (and
    # the row owner and how far into the episode we are, in quartiles) were known?
    # Row returns are DISCOUNTED, so even within one outcome they vary with
    # time-to-go -- conditioning on the phase quartile removes most of that.
    grp = np.asarray([f"{a}|{b}" for a, b in zip(out_lab[valid], plab[valid])], dtype=object)
    within = 0.0
    for u in set(grp.tolist()):
        m = grp == u
        within += float(np.var(tgt[m])) * int(m.sum())
    within /= n_valid
    lines.append(
        f"  [ppg loss diag] target variance {tot_var:.3f}; with the outcome, row owner and episode quartile all "
        f"known the remaining variance would be {within:.3f} ({100 * within / max(tot_var, 1e-12):.1f}% of total); "
        f"current mse (raw) = {mse_cur * var_norm:.3f}. The gap is what predicting the outcome (which is not known "
        f"in advance) costs the value net"
    )

    # Reliability: predicted-value deciles vs realised return.
    edges = np.quantile(cur[valid], np.linspace(0, 1, 11))
    lines.append("  [ppg loss diag] calibration (deciles of predicted value -> mean realised MC return):")
    cells = []
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        m = valid & (cur >= lo) & ((cur <= hi) if i == 9 else (cur < hi))
        if m.any():
            cells.append(f"[{lo:+.2f},{hi:+.2f}]: pred {cur[m].mean():+.2f} -> real {mc[m].mean():+.2f}")
    lines.append("    " + "  ".join(cells))

    if n_worst_rows > 0:
        worst = np.argsort(np.where(valid, (cur - np.where(valid, mc, 0.0)) ** 2, -1.0))[::-1][:n_worst_rows]
        lines.append("  [ppg loss diag] worst rows (row, outcome, track, phase, target, pred):")
        for i in worst:
            lines.append(
                f"    row {int(i)}: {out_lab[i]}, {trk[i]}, phase={pf[i]:.2f}, target={mc[i]:+.2f}, pred={cur[i]:+.2f}"
            )
    return lines

def _trimmed_rmse(sq_err: torch.Tensor, trim_frac: float) -> float:
    """RMSE (in the same units as the errors, i.e. sqrt of a mean of squared
    errors) after dropping the ``trim_frac`` fraction of rows with the LARGEST
    squared error -- e.g. ``0.10`` = "RMSE excluding the worst 10%". A robust
    companion to plain RMSE for the value refit's train/val lines: a small
    tail of badly-mis-valued states (e.g. rare high-variance outcomes)
    dominates a plain RMSE and can hide whether the bulk of predictions is
    improving. One-sided (unlike ``_trimmed_mean``, which cuts both tails),
    since a squared error has no meaningful "too good" tail to drop.

    Implemented with a sort rather than ``torch.quantile`` on purpose: the
    latter raises on inputs past ~16M elements, and an augmented train batch
    can get large. ``trim_frac <= 0`` (or a keep-count reaching every row)
    degenerates to the plain RMSE; empty input returns ``nan``.
    """
    n = sq_err.numel()
    if n == 0:
        return float("nan")
    keep = n if trim_frac <= 0.0 else max(1, int(math.ceil(n * (1.0 - trim_frac))))
    if keep >= n:
        return math.sqrt(float(sq_err.mean()))
    kept = torch.sort(sq_err.reshape(-1)).values[:keep]
    return math.sqrt(float(kept.mean()))


class PPGMixin:
    """PPG value refit and consistency refit for PPOTrainer. Mixed into PPOTrainer; relies on its attributes."""

    def _ppg_update_episode_replay(
        self, this_cycle_seed_mse: list[tuple[int, float]], pending_before: dict[int, float],
    ) -> tuple[dict[int, float], Optional[list[int]]]:
        """``ppg_value_refit``'s own rollout-to-rollout episode-seed-replay
        bookkeeping -- modeled on ``_update_episode_replay`` (same
        report-then-reselect shape), but:

        - Call-scoped, not ``self.`` state: ``pending_before`` is passed in
          and the new value returned, rather than read/written on
          ``self._pending_replay_before`` -- PPG bursts are independent,
          self-contained events with no cross-call carry-over (PPO's own
          queue, by contrast, persists across the whole training loop).
        - Ranks by value-prediction MSE descending (worst-predicted), not
          mean(|advantage|) -- the value-fitting analog of PPO's ranking,
          since this refit's job is fitting the value function, not
          improving the policy.
        - Reuses ``self._episode_replay_top_fraction`` (the SAME config
          value PPO's own mechanism uses) rather than a separate knob.

        Args:
            this_cycle_seed_mse: ``[(seed, mean_squared_error), ...]`` for
                every completed episode in the cycle that just ran (see
                ``_collect_value_pretrain_rollout``'s
                ``episode_seed_mse``/``collect_episode_replay_stats``).
            pending_before: ``{seed: mse}`` queued at the end of the
                PREVIOUS cycle (empty dict if there was none, e.g. this is
                cycle 1 -- cycle 0 never queues anything here).

        Returns:
            ``(new_pending_before, next_cycle_replay_seeds)`` --
            ``next_cycle_replay_seeds`` is ``None`` when
            ``this_cycle_seed_mse`` was empty (nothing eligible to select,
            e.g. the batched worker path isn't active).
        """
        if pending_before:
            matched = [
                (seed, pending_before[seed], mse)
                for seed, mse in this_cycle_seed_mse
                if seed in pending_before
            ]
            if matched:
                before_mean = float(np.mean([b for _s, b, _a in matched]))
                after_mean = float(np.mean([a for _s, _b, a in matched]))
                log.info(
                    f"  [ppg episode replay] matched={len(matched)}/{len(pending_before)}  "
                    f"before_mean_mse={before_mean:.4f}  after_mean_mse={after_mean:.4f}  "
                    f"delta={after_mean - before_mean:+.4f}"
                )
            else:
                log.info(
                    f"  [ppg episode replay] 0/{len(pending_before)} queued seeds were "
                    f"replayed this cycle (not enough episode resets to reach them)"
                )
            pending_before = {}

        if not this_cycle_seed_mse:
            return pending_before, None

        n_top = max(1, math.ceil(self._episode_replay_top_fraction * len(this_cycle_seed_mse)))
        top_k = sorted(this_cycle_seed_mse, key=lambda t: t[1], reverse=True)[:n_top]
        pending_before = {seed: mse for seed, mse in top_k}
        log.info(
            f"  [ppg episode replay] queued {len(top_k)} seed(s) for next cycle "
            f"(top {self._episode_replay_top_fraction * 100:.1f}% of "
            f"{len(this_cycle_seed_mse)} episodes by value-MSE)"
        )
        return pending_before, list(pending_before.keys())

    def _load_snapshot_dict_from_checkpoint(self, path: Path) -> dict:
        """Load a checkpoint FILE (as already written by _save_checkpoint())
        into the same {"step", "decision", "execution", "value"} shape
        _eval_vs_neural_snapshot() expects as its opponent_snapshot -- the
        single source of truth for "what did the network look like at
        checkpoint N" is the checkpoint file itself, not a separate
        in-memory copy."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        from footballcoach.ai.ppo.checkpoint_io import pad_value_side_columns
        pad_value_side_columns(ckpt["execution_net"], self.execution_net, "snapshot execution_net")
        if self.separate_value_net and ckpt.get("value_net") is not None:
            pad_value_side_columns(ckpt["value_net"], self.value_net, "snapshot value_net")
        return {
            "step": ckpt.get("step", 0),
            "decision": ckpt["decision_net"],
            "execution": ckpt["execution_net"],
            "value": ckpt.get("value_net") if self.separate_value_net else None,
        }

    def _ppg_interleave_trigger_path(self) -> Optional[Path]:
        """File whose presence (checked once per rollout) manually fires an interleaved PPG
        burst. bc.ppg_interleave_trigger_file overrides; otherwise
        '<checkpoint_dir>/ppg_trigger' -- None if there's no checkpoint_dir to put it in
        (matches _maybe_run_neural_snapshot_eval's own checkpoint_dir requirement)."""
        if self._ppg_interleave_trigger_file:
            return Path(self._ppg_interleave_trigger_file)
        if self.checkpoint_dir is not None:
            return Path(self.checkpoint_dir) / "ppg_trigger"
        return None

    def _maybe_run_ppg_interleave(
        self, env, phase_id: Optional[int],
        close_main_pools_fn: Optional[Callable[[], None]] = None,
        respawn_main_pools_fn: Optional[Callable[[], None]] = None,
        replay_seeds: Optional[list[int]] = None,
    ) -> None:
        """Interleaved PPG: run a burst of ppg_value_refit() cycles mid-PPO, then resume
        normal rollout collection, instead of only via the standalone --ppg-refit-only entry
        point. Reuses ppg_value_refit() completely unchanged -- this only adds the trigger
        check. Off by default (bc.ppg_interleave_enabled); see ai/knowledge.md
        "Interleaved PPG".

        ``replay_seeds``: forwarded as ppg_value_refit's ``first_rollout_replay_seeds`` --
        the caller's CURRENT ``replay_seeds_for_next_cycle`` (episode-seed replay's queued
        top-|advantage| seeds, if that feature is enabled and this rollout produced any),
        so the burst's first cycle sees the same deliberately-hard episodes the main loop's
        own next rollout is about to use, instead of ordinary on-policy sampling. Purely
        additive -- this never consumes/mutates the caller's list, so the main loop's next
        rollout still replays the same seeds independently afterward. None (the default,
        and the only value the two callers without episode-seed replay wired up ever pass)
        = no replay, byte-identical to before this parameter existed.

        Two ways to fire, checked once per rollout (same cadence as
        _maybe_run_neural_snapshot_eval), not mutually exclusive:
          - Manual: create the file _ppg_interleave_trigger_path() points at (touch it, no
            content needed) at any time while the run is live. Consumed (deleted) the next
            time this is checked, so it never fires twice for one touch.
          - Cadence safety net: bc.ppg_interleave_cadence_rollouts > 0 fires automatically
            every N rollouts, counted since the last interleave (manual or cadence) fired.
        Runs the burst and saves an extra checkpoint immediately after, so the refit's effect
        is captured on disk right away rather than only at the next natural checkpoint.

        ``close_main_pools_fn`` / ``respawn_main_pools_fn`` (both optional, no-ops when the
        caller passes None -- the single-process train() path has no persistent pool to manage):
        called immediately before/after the burst to close and respawn the MAIN LOOP's own
        persistent rollout + eval worker pools around it, so at most ONE pool (the main loop's
        OR ppg_value_refit()'s own freshly-spawned one, never both) is alive at a time. Added
        2026-09-23 after a real crash: running the main loop's persistent pool (rollout workers
        + eval workers) concurrently with ppg_value_refit()'s own freshly-spawned batched pool
        exhausted this machine's RAM (confirmed: free memory jumped from a few GB to 25GB/32GB
        the moment the crashed process's tree died, and the crash traceback was a cascade of
        BrokenPipeError/EOFError from workers -- both rollout AND _eval_worker_main -- losing
        their pipe to a main process that had just died, consistent with an OOM kill). The
        earlier version of this comment called the double-pool footprint "accepted" -- it
        wasn't safe, and the fix (pay a respawn-time cost, never hold both pools at once) is
        worth it. See project_ppg_interleave_design memory for the original "easy version"
        design reasoning this replaces the risky half of.
        """
        if not self._ppg_interleave_enabled:
            return
        self._rollouts_since_ppg_interleave += 1
        trigger_path = self._ppg_interleave_trigger_path()
        triggered_by: Optional[str] = None
        if trigger_path is not None and trigger_path.exists():
            triggered_by = "manual"
        elif (
            self._ppg_interleave_cadence_rollouts > 0
            and self._rollouts_since_ppg_interleave >= self._ppg_interleave_cadence_rollouts
        ):
            triggered_by = "cadence"
        if triggered_by is None:
            return
        if triggered_by == "manual" and trigger_path is not None:
            try:
                trigger_path.unlink()
            except OSError:
                pass
        log.info(
            f"[ppg interleave] triggered ({triggered_by}): running "
            f"{self._ppg_interleave_num_rollouts} PPG refit cycle(s) mid-PPO "
            f"({self._ppg_interleave_n_steps:,} steps/cycle)..."
        )
        if close_main_pools_fn is not None:
            log.info("[ppg interleave] closing the main loop's persistent worker pools before the burst...")
            close_main_pools_fn()
        try:
            self.ppg_value_refit(
                env,
                n_steps=self._ppg_interleave_n_steps,
                phase_id=phase_id,
                num_rollouts=self._ppg_interleave_num_rollouts,
                first_rollout_replay_seeds=replay_seeds,
            )
        finally:
            if respawn_main_pools_fn is not None:
                log.info("[ppg interleave] respawning the main loop's persistent worker pools after the burst...")
                respawn_main_pools_fn()
        self._rollouts_since_ppg_interleave = 0
        if self.checkpoint_dir is not None:
            self._save_checkpoint(self._total_steps)
        log.info("[ppg interleave] done, resuming normal PPO rollout collection.")

    def ppg_value_refit(
        self,
        env,
        n_steps: int,
        phase_id: Optional[int] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        kl_coef: Optional[float] = None,
        batch_size: Optional[int] = None,
        num_rollouts: Optional[int] = None,
        first_rollout_replay_seeds: Optional[list[int]] = None,
    ) -> dict:
        """Re-fit the (shared-trunk) value function with FULL, unfrozen
        gradient flow through decision_net/execution_net's trunk, while an
        analytic per-head KL penalty (``_ppg_kl_penalty``) anchors the
        policy's output distributions to a snapshot taken right before this
        call, so the value function can track a reward/curriculum change (or
        simply be given more capacity to fit returns) without the value
        gradient damaging an already-decent policy.

        This is the middle ground ``pretrain_value()`` cannot offer in the
        shared-trunk case: ``value_pretrain_frozen_layers=-1`` (default)
        freezes the trunk so only ``execution_net.value_head`` can adapt (a
        weak fit); ``=0`` unfreezes everything with no protection at all
        (the value gradient reshapes the exact trunk features the policy
        heads read from). Inspired by the auxiliary phase of OpenAI's
        Phasic Policy Gradient (PPG) paper -- see ai/knowledge.md "PPG-style
        value refit" for the full design writeup -- but scoped here as a
        standalone/occasional refit (called once after pretraining, or
        on-demand against an existing checkpoint), NOT an automatic
        alternating cadence inside the main PPO rollout loop.

        Only meaningful when ``self.separate_value_net`` is False -- the
        separate-net critic already has its own unfrozen trunk with zero
        risk to the policy (no-ops with a warning otherwise, matching
        ``pretrain_value``'s identical branching rationale).

        Structurally mirrors ``pretrain_value()`` (rollout collection via
        the same ``_collect_value_pretrain_rollout``, 85/15 train/val episode
        split, per-epoch minibatch loop, early stop + best-val restore) with
        four real differences: no trunk freezing, a fresh dedicated
        optimizer over EVERY decision_net+execution_net param (never
        ``self.optimizer`` -- matches ``pretrain_value``'s own throwaway-
        optimizer convention rather than mixing this objective's gradient
        statistics into the live PPO Adam state), the added KL-anchor term,
        and GAE-bootstrapped returns instead of ``pretrain_value``'s pure
        Monte Carlo returns -- see ``_collect_value_pretrain_rollout``'s
        ``use_gae`` docstring paragraph for why: real PPO's own value loss
        targets GAE returns, and the actual point of this refit (matching
        PPG's own auxiliary phase) is to converge the value head toward the
        SAME quantity real training already uses, not a different-but-
        correlated one that still needs readjusting once training resumes.

        ``num_rollouts`` > 1 repeats the whole "collect a fresh rollout ->
        take a new anchor snapshot off the CURRENT (by-then already-shifted)
        weights -> fit epochs -> restore best-val" cycle that many times in
        one call, each cycle fully independent (own rollout, own train/val
        split, own anchor, own early-stop/best-val tracking) except for the
        optimizer: ``ppg_opt`` (and its Adam momentum) is built ONCE and
        reused across every cycle in this call, deliberately -- resetting
        Adam's moving averages every cycle would be pure waste, since
        cycles within one call are meant to behave like one continuous
        refit session, just periodically re-grounded against fresh on-
        policy data and a fresh anchor. This is intentionally NOT the same
        thing as the "automatic alternating cadence inside the main PPO
        loop" this method's docstring says is out of scope -- it never
        touches ``_train_batched_parallel``/the main rollout loop at all,
        it's purely an internal repeat-N-times knob on one standalone call.

        If ``self.checkpoint_dir`` is set (true for any trainer built the
        normal way, via ``train.py``), a checkpoint is saved to
        ``checkpoint_dir/checkpoint_pretrained.pt`` after EVERY cycle, not
        just once at the end -- a multi-rollout call can run a long time,
        so this bounds how much work an interrupt/crash partway through
        loses. No-op (no save) when ``checkpoint_dir`` is None.

        Args:
            env: forwarded to ``_collect_value_pretrain_rollout`` -- ignored
                when ``ppo.value_pretrain_n_processes > 1`` and ``phase_id``
                is given (same convention as ``pretrain_value``).
            n_steps: rollout steps to collect PER rollout cycle.
            phase_id: curriculum phase id, for parallel rollout collection.
            epochs/lr/kl_coef/batch_size: override the ``bc.ppg_*``/
                ``bc.value_pretrain_batch_size`` config defaults for a
                single call.
            num_rollouts: override ``bc.ppg_num_rollouts`` (default 1) --
                how many collect+fit+restore cycles to run in this call.
            first_rollout_replay_seeds: episode seeds to force-replay in ONLY
                the FIRST cycle's rollout collection (cycle 2+ within this
                same call, if any, sample normally) -- forwarded as-is to
                ``_collect_value_pretrain_rollout``'s own ``replay_seeds``
                (see its docstring for chunking/support details, notably
                that it needs the batched worker path). Intended use:
                ``_maybe_run_ppg_interleave`` passes the main PPO loop's own
                ``replay_seeds_for_next_cycle`` (the top-|advantage| episodes
                queued by episode-seed replay right before the interleave
                fired) so the refit's value-fitting sees the same
                deliberately-hard episodes real training was about to use,
                instead of ordinary on-policy sampling -- purely additive:
                the main loop's own next rollout still consumes the SAME
                seed list independently afterward (this never mutates or
                consumes it). None (default) = no replay, i.e. every cycle
                samples normally, unchanged from before this parameter
                existed. When ``ppg_episode_replay_enabled`` (a separate,
                independent config flag), cycles AFTER cycle 0 additionally
                run their own rollout-to-rollout replay loop -- each cycle
                selects its own top ``episode_replay_top_fraction`` of
                episodes by value-prediction MSE (worst-predicted) and
                force-replays them next cycle, with a
                ``[ppg episode replay] matched=.../before_mean_mse=.../
                after_mean_mse=.../delta=...`` report each time. See that
                flag's ``ai_config.json`` comment for the full mechanism.

        Returns:
            The rollout-stats dict from the LAST cycle's
            ``_collect_value_pretrain_rollout`` call (episode returns/
            outcomes), for parity with ``pretrain_value``'s return value --
            not an aggregate across all ``num_rollouts`` cycles.
        """
        if self.separate_value_net:
            log.warning(
                "ppg_value_refit is a no-op with separate_value_net=True "
                "(the critic already has its own unfrozen trunk -- nothing to protect)."
            )
            return {}

        epochs = epochs if epochs is not None else self._ppg_epochs
        lr = lr if lr is not None else self._ppg_lr
        kl_coef = kl_coef if kl_coef is not None else self._ppg_kl_coef
        _batch_size = batch_size if batch_size is not None else self._value_pretrain_batch_size
        num_rollouts = num_rollouts if num_rollouts is not None else self._ppg_num_rollouts
        log.info(
            f"PPG value refit: {num_rollouts} rollout(s), {n_steps} steps/rollout, "
            f"{epochs} epochs, lr={lr}, kl_coef={kl_coef}, batch_size={_batch_size}"
        )

        # Optimizer (+ its grad-clip param groups) built ONCE and reused
        # across every rollout cycle below -- see docstring's num_rollouts
        # paragraph for why Adam's momentum deliberately persists across
        # cycles within this one call, unlike the rollout/anchor/best-val
        # state, which is fully independent per cycle.
        ppg_params_all = list(self.decision_net.parameters()) + list(self.execution_net.parameters())
        ppg_opt = torch.optim.Adam(ppg_params_all, lr=lr, eps=1e-5)
        if self._ppg_seed_adam_variance and self.optimizer is not None:
            # One-directional Adam seeding (bc.ppg_seed_adam_variance, off by default -- see
            # its config comment). Copies each param's *live* PPO Adam variance estimate
            # (exp_avg_sq) and step count into this fresh optimizer, so PPG doesn't start
            # from a cold (all-zero) per-parameter step-size calibration -- the untested lead
            # hypothesis in [[project_kick_power_ppg_kl_investigation]] for why move_dir's
            # kappa historically dominates PPG-refit KL. Momentum (exp_avg) is explicitly
            # zeroed, NOT copied -- self.optimizer's momentum points toward PPO's own
            # clipped-surrogate objective, which has nothing to do with this refit's pure
            # value-loss + KL-anchor gradient, so carrying it over would bias PPG's first
            # steps in a direction unrelated to what it's actually optimizing.
            #
            # The step count must travel WITH exp_avg_sq: Adam's bias-correction divides by
            # (1 - beta2**step), which assumes exp_avg_sq grew from zero over `step` calls.
            # Seeding a mature exp_avg_sq at step=0 would make that correction factor tiny
            # (~0.001 at step=1), inflating the "corrected" variance ~1000x and producing
            # near-frozen effective updates -- copying self.optimizer's own (huge) step count
            # keeps the correction factor at ~1, so the borrowed variance estimate is used
            # as-is, which is the whole point.
            #
            # Strictly read-only: every value below is a fresh .clone() of a tensor read out
            # of self.optimizer.state, and self.optimizer itself is never written to -- this
            # cannot affect the live PPO optimizer no matter how PPG's refit goes.
            _seeded = _skipped = 0
            for _p in ppg_params_all:
                _live_state = self.optimizer.state.get(_p)
                if not _live_state or "exp_avg_sq" not in _live_state:
                    _skipped += 1  # e.g. frozen physics-encoder params, never in self.optimizer
                    continue
                _step_val = _live_state.get("step", 0)
                ppg_opt.state[_p]["exp_avg_sq"] = _live_state["exp_avg_sq"].detach().clone()
                ppg_opt.state[_p]["exp_avg"] = torch.zeros_like(_p)
                ppg_opt.state[_p]["step"] = _step_val.detach().clone() if torch.is_tensor(_step_val) else _step_val
                _seeded += 1
            log.info(
                f"  [ppg value refit] seeded Adam exp_avg_sq (variance) from live PPO optimizer for "
                f"{_seeded} param(s) ({_skipped} skipped -- not present in self.optimizer.state); "
                f"momentum (exp_avg) left at zero, read-only copy (live optimizer untouched)."
            )
        _non_direction_params = [p for p in ppg_params_all if id(p) not in self.direction_param_ids]
        _direction_params = [p for p in ppg_params_all if id(p) in self.direction_param_ids]

        # Parallel rollout workers (plain or batched, per
        # ppo.value_pretrain_batched_rollout), if applicable, are ALSO
        # spawned ONCE and reused across every cycle below -- see
        # _collect_value_pretrain_rollout's ``pool`` docstring paragraph.
        # Respawning num_rollouts times would pay full process-spawn overhead
        # on every cycle for no reason (weights still get re-synced each
        # cycle regardless, since the policy keeps moving). None when
        # parallel collection doesn't apply (single-process branch).
        _orig_nets = None
        if self._ppg_compare_to_original:
            _orig_nets = (
                copy.deepcopy(self.decision_net).eval(), copy.deepcopy(self.execution_net).eval(),
            )
            for _m in _orig_nets:
                for _p in _m.parameters():
                    _p.requires_grad_(False)
        _mc_kw = {"with_mc_returns": True} if (self._ppg_compare_to_original or self._ppg_loss_diagnostics) else {}
        if self._ppg_rollout_deterministic:
            _mc_kw["deterministic"] = True
            log.info("  [ppg value refit] rollouts are collected DETERMINISTICALLY (bc.ppg_rollout_deterministic)")
        _pool = self._spawn_value_pretrain_workers(phase_id)
        if _pool is not None and num_rollouts > 1:
            log.info(
                f"  [ppg value refit] spawned {_pool.n_processes} rollout process(es) "
                f"({'batched' if _pool.batched else 'plain'}), reused across all {num_rollouts} cycles"
            )

        # PPG's OWN rollout-to-rollout episode-seed replay (ppg_episode_replay_enabled),
        # modeled on but independent of PPO's self._pending_replay_before -- call-scoped
        # (local, not self.) since each ppg_value_refit call's cycles are a self-contained
        # burst with no carry-over to the next call. {seed: mse_when_queued}.
        _ppg_pending_replay_before: dict[int, float] = {}
        _ppg_next_cycle_replay_seeds: Optional[list[int]] = None

        # Rotating buffer of this CALL's own most recent cycle-end weights
        # (cycle number, decision_state_dict, execution_state_dict), most
        # recent last, capped at 2 -- feeds the rollout-1+ prev-checkpoint
        # value-loss diagnostic below for a standalone multi-rollout refit
        # (e.g. --ppg-refit-only num_rollouts>1), where there's no on-disk
        # checkpoint{N}.pt history to fall back on since this call only ever
        # writes one fixed checkpoint_pretrained.pt. A live single-rollout
        # PPO-interleaved call never populates this (num_rollouts=1, no
        # "previous cycle" within the call), so it falls through to the
        # on-disk checkpoint{N}.pt path entirely, unchanged from before.
        _cycle_end_snapshots: list[tuple[int, dict, dict]] = []

        try:
            _rollout_stats: dict = {}
            for _rollout_i in range(num_rollouts):
                if num_rollouts > 1:
                    log.info(f"=== PPG value refit: rollout {_rollout_i + 1}/{num_rollouts} ===")
                # _collect_value_pretrain_rollout() is shared with pretrain_value()
                # (same rollout-collection code either way, so its own progress/
                # summary lines below are still labeled "[value pretrain rollout]"
                # -- this line exists purely so that doesn't read as "oh, it's
                # running pretrain_value again" when it's actually the PPG
                # refit's own collection step) -- but use_gae=True here, UNLIKE
                # pretrain_value's own call: see _collect_value_pretrain_rollout's
                # use_gae docstring paragraph for why GAE (matching what real PPO
                # training itself targets) is the right choice specifically for
                # refitting an already-decent value function, as opposed to
                # pretrain_value's cold-start-from-BC scenario where MC returns
                # avoid a real circularity problem GAE would have there.
                log.info("Doing PPG value refit rollout collection now (reuses pretrain_value()'s rollout collector, hence the '[value pretrain rollout]' label below)...")
                # Cycle 0 uses the main PPO loop's one-shot queued seeds
                # (first_rollout_replay_seeds -- see this method's own docstring
                # paragraph). From cycle 1 onward, when ppg_episode_replay_enabled,
                # PPG runs its OWN version of the same idea using whatever the
                # PREVIOUS cycle queued (set at the bottom of this loop body).
                _cycle_replay_seeds = first_rollout_replay_seeds if _rollout_i == 0 else _ppg_next_cycle_replay_seeds
                if _cycle_replay_seeds:
                    log.info(f"  [ppg value refit] replaying {len(_cycle_replay_seeds)} queued episode-replay seed(s) in this cycle's rollout")
                batch, _rollout_stats = self._collect_value_pretrain_rollout(
                    env, n_steps, phase_id, use_gae=True, pool=_pool,
                    replay_seeds=_cycle_replay_seeds,
                    collect_episode_replay_stats=self._ppg_episode_replay_enabled, **_mc_kw,
                )

                # --- Episode-level 85/15 train/val split (overfit detection) ---
                # Identical to pretrain_value()'s own split -- see its comments.
                dones_arr = batch["dones"].numpy()
                episode_end_idxs = np.where(dones_arr > 0.5)[0]
                n_complete_eps = len(episode_end_idxs)
                n_val_eps = (
                    max(1, round(self._ppg_val_episode_fraction * n_complete_eps))
                    if (self._ppg_val_episode_fraction > 0.0 and n_complete_eps >= 2) else 0
                )
                n_train_eps = n_complete_eps - n_val_eps
                n_total = len(dones_arr)
                val_mask = np.zeros(n_total, dtype=bool)
                if n_val_eps > 0:
                    ep_starts = np.concatenate([[0], episode_end_idxs[:-1] + 1])
                    for _i in range(n_train_eps, n_complete_eps):
                        val_mask[ep_starts[_i]:episode_end_idxs[_i] + 1] = True
                train_mask = ~val_mask

                # ret_std source (all returns) -- taken BEFORE the split below
                # empties ``batch`` key by key.
                all_returns_t = batch["returns"].to(self.device)

                # Split WITHOUT keeping the full batch alive next to its two
                # slices (see _split_batch_releasing): peak ~1x, not ~2x.
                train_batch_raw, val_batch_raw = _split_batch_releasing(
                    batch, train_mask, val_mask if n_val_eps > 0 else None,
                )

                log.info(
                    f"  PPG refit split: {n_train_eps} train eps ({int(train_mask.sum())} steps)"
                    + (f"  |  {n_val_eps} val eps ({int(val_mask.sum())} steps)" if n_val_eps > 0 else "")
                )

                # Augment only the train portion, same as pretrain_value(). Unlike
                # PPO's own ratio-based objective, neither the value loss nor the
                # KL-anchor term here depend on any precomputed "old" log_prob --
                # the anchor is snapshotted AFTER augmentation (below), over the
                # exact same (possibly-augmented, larger) row set being trained on
                # -- so augment_batch()'s known old_log_prob approximation for
                # flip_y copies (see augment.py) simply never comes into play here.
                _n_base = len(train_batch_raw["returns"])  # identity copy = the first _n_base rows after augmentation
                if self.augment_n_slot_shuffles > 0:
                    train_batch = augment_batch(train_batch_raw, self.augment_n_slot_shuffles, self._aug_rng)
                    del train_batch_raw  # the un-augmented copy is dead weight from here on
                else:
                    train_batch = train_batch_raw

                # Frozen physics-encoder outputs depend only on the (now fixed)
                # input features, so compute them ONCE per row here instead of
                # re-running the encoders inside every anchor/baseline/epoch/val
                # forward pass (~2x epochs + 4 extra passes over the data per
                # cycle) -- the same trick _ppo_update uses across its n_epochs.
                # Rides along as ordinary obs/* entries -> picked up by the
                # generic obs slicing below; decision_net calls pass it through.
                train_batch.update(self._precompute_physics_full(train_batch, _batch_size))
                if val_batch_raw is not None:
                    val_batch_raw.update(self._precompute_physics_full(val_batch_raw, _batch_size))

                ret_std = all_returns_t.std().clamp(min=1.0)
                returns_t = train_batch["returns"].to(self.device)

                val_returns_t = None
                val_obs_dict = None
                val_actions_dict = None
                if val_batch_raw is not None:
                    val_returns_t = val_batch_raw["returns"].to(self.device)
                    # obs stays on CPU (moved per chunk) -- see _train_obs_full below.
                    val_obs_dict = {k.replace("obs/", ""): val_batch_raw[k]
                                    for k in val_batch_raw if k.startswith("obs/")}
                    val_actions_dict = {k.replace("action/", ""): val_batch_raw[k].to(self.device)
                                         for k in val_batch_raw if k.startswith("action/")}

                n = len(returns_t)

                # --- Anchor snapshot(s): ONE no_grad pass each over the whole
                # (post-augmentation) train batch AND the held-out val batch, under
                # the CURRENT weights, before any gradient step -- see
                # _ppg_snapshot_anchor's docstring. Everything from here on is
                # pulled back toward these snapshots by the KL term (train) or
                # measured against them (val, diagnostic only -- val is never
                # trained on). anchor_globals is shared by both: it's just the
                # handful of GLOBAL spread nn.Parameters at refit-start, not
                # per-row, so there's nothing train/val-specific about it.
                # Deliberately CPU-resident: the whole augmented train obs is
                # several GB (other_feat + the precomputed other_physics_full
                # alone are ~2.4k floats/row), and parking it on the GPU next
                # to the anchors + training activations can overflow VRAM into
                # (very slow) shared system memory. Every consumer moves one
                # chunk at a time, exactly like PPO's per-minibatch .to().
                _train_obs_full = {
                    k.replace("obs/", ""): v
                    for k, v in train_batch.items() if k.startswith("obs/")
                }
                anchor = self._ppg_snapshot_anchor(_train_obs_full, _batch_size)
                val_anchor = (
                    self._ppg_snapshot_anchor(val_obs_dict, _batch_size)
                    if val_obs_dict is not None else None
                )
                anchor_globals = {
                    "move_dir_log_kappa": self.execution_net.move_dir_log_kappa.detach().clone(),
                    "kick_dir_log_kappa": self.execution_net.kick_dir_log_kappa.detach().clone(),
                    "kick_dir_z_log_std": self.execution_net.kick_dir_z_log_std.detach().clone(),
                    "kick_power_log_std": self.execution_net.kick_power_log_std.detach().clone(),
                }
                if not self._kick_spin_frozen:
                    anchor_globals["kick_spin_log_std"] = self.execution_net.kick_spin_log_std.detach().clone()

                # NOTE: ppg_opt/_non_direction_params/_direction_params are built
                # ONCE, before the rollout loop (see above) -- deliberately NOT
                # rebuilt per cycle, so Adam's momentum persists across
                # num_rollouts cycles within this one call.

                def _eval_value_loss(
                    obs_source: dict, ret_source: torch.Tensor,
                ) -> tuple[float, float, float]:
                    """Returns (normalized_mse, rmse, rmse excluding the worst
                    ``ppg_rmse_trim_frac`` rows -- see ``_trimmed_rmse``)."""
                    total_sq = 0.0
                    n_rows = 0
                    sq_chunks: list[torch.Tensor] = []
                    with torch.no_grad():
                        for start in range(0, len(ret_source), _batch_size):
                            mb_ret = ret_source[start:start + _batch_size]
                            mb_obs = {k: v[start:start + _batch_size].to(self.device) for k, v in obs_source.items()}
                            sat, oat = _ai_types(mb_obs)
                            d_heads = self.decision_net(
                                mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"],
                                mb_obs["ball_feat"], mb_obs["global_feat"], sat, oat,
                                ball_physics_full=mb_obs.get("ball_physics_full"),
                                self_physics_full=mb_obs.get("self_physics_full"),
                                other_physics_full=mb_obs.get("other_physics_full"),
                            )
                            _value = self.execution_net(
                                mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"],
                                mb_obs["ball_feat"], mb_obs["global_feat"], d_heads, sat, oat,
                                value_only=True,
                            )
                            preds = _value.squeeze(-1)
                            _sq = (preds - mb_ret) ** 2
                            sq_chunks.append(_sq)
                            total_sq += float(_sq.sum())
                            n_rows += len(mb_ret)
                    mse = total_sq / max(n_rows, 1)
                    norm_mse = mse / float(ret_std ** 2)
                    _trim_rmse = _trimmed_rmse(torch.cat(sq_chunks) if sq_chunks else torch.empty(0), self._ppg_rmse_trim_frac)
                    return norm_mse, float(ret_std) * math.sqrt(norm_mse), _trim_rmse

                def _eval_value_and_kl(
                    obs_source: dict, actions_source: dict, ret_source: torch.Tensor, anchor_source: dict,
                ) ->tuple[float, float, float, float, dict[str, float]]:
                    """Same value-loss computation as _eval_value_loss, PLUS the KL
                    penalty against anchor_source -- used for the held-out val set
                    each epoch (never trained on; purely diagnostic) so the log can
                    show kl_val alongside kl_train instead of only measuring drift
                    on the data actually being optimized. Returns
                    (normalized_mse, rmse, trimmed_rmse, mean_kl, per_head_kl)."""
                    total_sq = 0.0
                    n_rows = 0
                    sq_chunks: list[torch.Tensor] = []
                    kl_sum = 0.0
                    per_head_sum: dict[str, float] = {}
                    with torch.no_grad():
                        for start in range(0, len(ret_source), _batch_size):
                            mb_ret = ret_source[start:start + _batch_size]
                            mb_obs = {k: v[start:start + _batch_size].to(self.device) for k, v in obs_source.items()}
                            mb_actions = {k: v[start:start + _batch_size] for k, v in actions_source.items()}
                            mb_anchor = {k: v[start:start + _batch_size] for k, v in anchor_source.items()}
                            sat, oat = _ai_types(mb_obs)
                            d_heads = self.decision_net(
                                mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"],
                                mb_obs["ball_feat"], mb_obs["global_feat"], sat, oat,
                                ball_physics_full=mb_obs.get("ball_physics_full"),
                                self_physics_full=mb_obs.get("self_physics_full"),
                                other_physics_full=mb_obs.get("other_physics_full"),
                            )
                            e_heads = self.execution_net(
                                mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"],
                                mb_obs["ball_feat"], mb_obs["global_feat"], d_heads, sat, oat,
                            )
                            preds = e_heads.value.squeeze(-1)
                            bsz = len(mb_ret)
                            _sq = (preds - mb_ret) ** 2
                            sq_chunks.append(_sq)
                            total_sq += float(_sq.sum())
                            n_rows += bsz
                            total_kl, per_head_kl = self._ppg_kl_penalty(
                                d_heads, e_heads, mb_anchor, anchor_globals, mb_actions, mb_obs["exists_mask"],
                            )
                            kl_sum += float(total_kl) * bsz
                            for k, v in per_head_kl.items():
                                per_head_sum[k] = per_head_sum.get(k, 0.0) + float(v) * bsz
                    mse = total_sq / max(n_rows, 1)
                    norm_mse = mse / float(ret_std ** 2)
                    rmse = float(ret_std) * math.sqrt(norm_mse)
                    _trim_rmse = _trimmed_rmse(torch.cat(sq_chunks) if sq_chunks else torch.empty(0), self._ppg_rmse_trim_frac)
                    mean_kl = kl_sum / max(n_rows, 1)
                    per_head_mean = {k: v / max(n_rows, 1) for k, v in per_head_sum.items()}
                    return norm_mse, rmse, _trim_rmse, mean_kl, per_head_mean

                _trim_tag = f"ex{int(round(self._ppg_rmse_trim_frac * 100))}"
                _baseline_train_loss, _baseline_train_rmse, _baseline_train_trim = _eval_value_loss(
                    _train_obs_full, returns_t,
                )
                if val_obs_dict is not None and val_returns_t is not None:
                    _baseline_val_loss, _baseline_val_rmse, _baseline_val_trim = _eval_value_loss(
                        val_obs_dict, val_returns_t,
                    )
                    log.info(
                        f"  PPG refit epoch   0/{epochs} (baseline): "
                        f"train={_baseline_train_loss:.4f} rmse={_baseline_train_rmse:.2f} rmse_{_trim_tag}={_baseline_train_trim:.2f}  "
                        f"val={_baseline_val_loss:.4f} val_rmse={_baseline_val_rmse:.2f} val_rmse_{_trim_tag}={_baseline_val_trim:.2f} "
                        f"(std={float(ret_std):.1f})"
                    )
                else:
                    log.info(
                        f"  PPG refit epoch   0/{epochs} (baseline): "
                        f"train_loss={_baseline_train_loss:.4f}  rmse={_baseline_train_rmse:.2f} rmse_{_trim_tag}={_baseline_train_trim:.2f} "
                        f"(returns std={float(ret_std):.1f})"
                    )

                if self._ppg_log_prev_checkpoint_value_loss:
                    # User-requested (2026-10-01): is the value fn actually improving
                    # checkpoint-over-checkpoint / cycle-over-cycle, not just within
                    # this refit's own epochs? Scores up to 2 REFERENCE snapshots'
                    # frozen value heads (unmodified, as they were at the time)
                    # against THIS cycle's freshly collected rollout -- data none of
                    # them ever trained on -- using the exact same GAE-return target
                    # _baseline_train_loss (the current live weights' own epoch-0
                    # reading) is scored against, so all numbers are comparable.
                    # Reference snapshots, most recent first: THIS call's own prior
                    # cycle-end weights (_cycle_end_snapshots, populated below after
                    # each cycle -- the only history that exists for a standalone
                    # multi-rollout --ppg-refit-only call, which never writes
                    # numbered checkpoint{N}.pt files) first, falling back to
                    # on-disk checkpoint{N}.pt for any remaining slot (up to 2 total)
                    # -- the only source for a live single-rollout PPO-interleaved
                    # call (num_rollouts=1, so _cycle_end_snapshots is always empty
                    # when this runs), and also fills early cycles of a standalone
                    # call before 2 in-memory snapshots exist yet. Silently no-ops
                    # (nothing logged) until at least one reference of either kind
                    # exists, same "not enough history yet" convention as
                    # neural_snapshot_lookbacks.
                    _ref_sources: list[tuple[str, dict, dict]] = [
                        (f"cycle{_cyc_n}", _dstate, _estate)
                        for _cyc_n, _dstate, _estate in reversed(_cycle_end_snapshots)
                    ]
                    if len(_ref_sources) < 2 and self.checkpoint_dir is not None:
                        _disk_n = self._checkpoint_count
                        while len(_ref_sources) < 2 and _disk_n >= 1:
                            _ckpt_path = self.checkpoint_dir / f"checkpoint{_disk_n}.pt"
                            if _ckpt_path.exists():
                                try:
                                    _snap = self._load_snapshot_dict_from_checkpoint(_ckpt_path)
                                    _ref_sources.append((f"ckpt{_disk_n}", _snap["decision"], _snap["execution"]))
                                except Exception as _e:
                                    log.warning(f"  [ppg prev-checkpoint value loss] failed to load {_ckpt_path}: {_e}")
                            _disk_n -= 1
                    _ref_sources = _ref_sources[:2]

                    if _ref_sources:
                        _prev_losses: dict[str, float] = {}
                        for _label, _dstate, _estate in _ref_sources:
                            # _load_state_dict_tolerant, not a plain strict load: decision_net's
                            # frozen physics-encoder submodules aren't part of a saved checkpoint's
                            # state dict (they're reloaded separately from their own fixed path at
                            # construction time, same reason the real best-val restore a few lines
                            # above uses this exact helper) -- a strict load_state_dict here throws
                            # "Missing key(s)" for every ball_physics_encoder.*/player_physics_encoder.*
                            # param, confirmed the hard way (silent process death: this was an
                            # uncaught RuntimeError whose traceback went to stderr, which the
                            # launcher wasn't capturing, so it looked like an unexplained crash).
                            _dnet = copy.deepcopy(self.decision_net)
                            _load_state_dict_tolerant(_dnet, _dstate, "ppg prev-checkpoint diagnostic (decision_net)")
                            _dnet.eval()
                            _enet = copy.deepcopy(self.execution_net)
                            _load_state_dict_tolerant(_enet, _estate, "ppg prev-checkpoint diagnostic (execution_net)")
                            _enet.eval()
                            for _p in list(_dnet.parameters()) + list(_enet.parameters()):
                                _p.requires_grad_(False)
                            _sq_sum, _cnt = 0.0, 0
                            with torch.no_grad():
                                for _s in range(0, len(returns_t), _batch_size):
                                    _tgt = returns_t[_s:_s + _batch_size]
                                    _o = {k: v[_s:_s + _batch_size].to(self.device) for k, v in _train_obs_full.items()}
                                    _sat, _oat = _ai_types(_o)
                                    _dh = _dnet(
                                        _o["self_feat"], _o["other_feat"], _o["exists_mask"],
                                        _o["ball_feat"], _o["global_feat"], _sat, _oat,
                                        ball_physics_full=_o.get("ball_physics_full"),
                                        self_physics_full=_o.get("self_physics_full"),
                                        other_physics_full=_o.get("other_physics_full"),
                                    )
                                    _v = _enet(
                                        _o["self_feat"], _o["other_feat"], _o["exists_mask"],
                                        _o["ball_feat"], _o["global_feat"], _dh, _sat, _oat,
                                        value_only=True,
                                    ).squeeze(-1)
                                    _sq_sum += float(((_v - _tgt) ** 2).sum())
                                    _cnt += len(_tgt)
                            _prev_losses[_label] = _sq_sum / max(_cnt, 1) / float(ret_std ** 2)
                            del _dnet, _enet
                        _parts = [f"current(pre)={_baseline_train_loss:.4f}"] + [
                            f"{_lbl}={_mse:.4f}" for _lbl, _mse in _prev_losses.items()
                        ]
                        log.info(
                            "  [ppg prev-checkpoint value loss] " + "  ".join(_parts)
                            + "  (same fresh rollout for all; lower=better)"
                        )

                if _orig_nets is not None and "mc_returns" in train_batch:
                    # Paired, noise-cancelling check of whether the refit is
                    # actually getting better: score the frozen ORIGINAL
                    # networks and the CURRENT ones (identical to the last
                    # cycle's end weights) on this rollout's rows -- fresh, not
                    # yet trained on -- against pure-MC returns. Not GAE
                    # returns: those contain the sampling (current) network's
                    # own value predictions, which would favour it by exactly
                    # its disagreement with any other network.
                    _mc_t = train_batch["mc_returns"].to(self.device)

                    def _mse_vs_mc(dec_net, exec_net) -> tuple[float, int]:
                        _sq_sum, _cnt = 0.0, 0
                        with torch.no_grad():
                            for _s in range(0, len(_mc_t), _batch_size):
                                _tgt = _mc_t[_s:_s + _batch_size]
                                _ok = torch.isfinite(_tgt)
                                if not bool(_ok.any()):
                                    continue
                                _o = {k: v[_s:_s + _batch_size].to(self.device) for k, v in _train_obs_full.items()}
                                _sat, _oat = _ai_types(_o)
                                _dh = dec_net(
                                    _o["self_feat"], _o["other_feat"], _o["exists_mask"],
                                    _o["ball_feat"], _o["global_feat"], _sat, _oat,
                                    ball_physics_full=_o.get("ball_physics_full"),
                                    self_physics_full=_o.get("self_physics_full"),
                                    other_physics_full=_o.get("other_physics_full"),
                                )
                                _v = exec_net(
                                    _o["self_feat"], _o["other_feat"], _o["exists_mask"],
                                    _o["ball_feat"], _o["global_feat"], _dh, _sat, _oat,
                                    value_only=True,
                                ).squeeze(-1)
                                _sq_sum += float((((_v - _tgt) ** 2)[_ok]).sum())
                                _cnt += int(_ok.sum())
                        return _sq_sum / max(_cnt, 1) / float(ret_std ** 2), _cnt

                    _orig_mse, _n_mc = _mse_vs_mc(*_orig_nets)
                    _cur_mse, _ = _mse_vs_mc(self.decision_net, self.execution_net)
                    log.info(
                        f"  [ppg compare vs pure-MC returns] n={_n_mc:,} fresh rows: "
                        f"original={_orig_mse:.4f}  current={_cur_mse:.4f}  "
                        f"delta(original-current)={_orig_mse - _cur_mse:+.4f} "
                        f"({(_orig_mse - _cur_mse) / max(_orig_mse, 1e-12) * 100:+.1f}%; positive = refit better)"
                    )
                    _mc_t = None

                _diag_cycle = _rollout_i + 1
                if (
                    self._ppg_loss_diagnostics and "mc_returns" in train_batch
                    and (_diag_cycle == 1 or _diag_cycle % self._ppg_diag_every_n_cycles == 0)
                ):
                    _nb = _n_base

                    def _predict_rows(dec_net, exec_net) -> torch.Tensor:
                        _out = []
                        with torch.no_grad():
                            for _s in range(0, _nb, _batch_size):
                                _e = min(_s + _batch_size, _nb)
                                _o = {k: v[_s:_e].to(self.device) for k, v in _train_obs_full.items()}
                                _sat, _oat = _ai_types(_o)
                                _dh = dec_net(
                                    _o["self_feat"], _o["other_feat"], _o["exists_mask"],
                                    _o["ball_feat"], _o["global_feat"], _sat, _oat,
                                    ball_physics_full=_o.get("ball_physics_full"),
                                    self_physics_full=_o.get("self_physics_full"),
                                    other_physics_full=_o.get("other_physics_full"),
                                )
                                _out.append(exec_net(
                                    _o["self_feat"], _o["other_feat"], _o["exists_mask"],
                                    _o["ball_feat"], _o["global_feat"], _dh, _sat, _oat,
                                    value_only=True,
                                ).squeeze(-1).float().cpu())
                        return torch.cat(_out)

                    _preds = {"current": _predict_rows(self.decision_net, self.execution_net)}
                    if _orig_nets is not None:
                        _preds["original"] = _predict_rows(*_orig_nets)
                    _mc_np = train_batch["mc_returns"][:_nb].numpy()
                    _dones_np = train_batch["dones"][:_nb].numpy()
                    _trk = train_batch["track_ids"][:_nb]
                    _other = None
                    if "obs/other_ai_type" in train_batch:
                        _oat_t = train_batch["obs/other_ai_type"][:_nb]
                        _ex_t = train_batch["obs/exists_mask"][:_nb]
                        _oh = (_oat_t * _ex_t.unsqueeze(-1)).sum(dim=1)
                        _other = torch.where(_oh.sum(dim=-1) > 0, _oh.argmax(dim=-1), torch.full_like(_oh.argmax(dim=-1), -1)).numpy()
                    _phase = _episode_phase_fraction(list(_trk), _dones_np, _mc_np)
                    _lines = _value_loss_breakdown(
                        {k: v.numpy() for k, v in _preds.items()}, _mc_np,
                        train_batch["step_outcomes"][:_nb], list(_trk), _dones_np, _other, float(ret_std),
                        phase_frac=_phase,
                    )
                    for _ln in _lines:
                        # every line carries the tag so a log filter / monitor can grab the whole block
                        log.info(_ln if "[ppg loss diag]" in _ln else "  [ppg loss diag]" + _ln)
                    if self.checkpoint_dir is not None:
                        _npz = self.checkpoint_dir / f"ppg_diag_cycle{_diag_cycle}.npz"
                        np.savez_compressed(
                            _npz, mc=_mc_np.astype(np.float32), gae_return=train_batch["returns"][:_nb].numpy(),
                            pred_current=_preds["current"].numpy(),
                            pred_original=(_preds["original"].numpy() if "original" in _preds else np.zeros(0, np.float32)),
                            phase=_phase.astype(np.float32), dones=_dones_np.astype(np.float32),
                            other_ai_type=(_other if _other is not None else np.zeros(0, np.int64)),
                            track=np.asarray(_trk, dtype=object).astype(str),
                            outcome=np.asarray(train_batch["step_outcomes"][:_nb], dtype=object).astype(str),
                            ret_std=np.float32(float(ret_std)),
                        )
                        log.info(f"  [ppg loss diag] per-row arrays saved to {_npz}")
                    del _preds, _mc_np, _dones_np, _trk, _other, _phase, _lines

                _best_val_loss = float("inf")
                _best_decision_state: Optional[dict] = None
                _best_execution_state: Optional[dict] = None
                _patience = 0
                _EARLY_STOP_PATIENCE = self._value_pretrain_early_stop_patience
                _EARLY_STOP_MIN_DELTA = self._value_pretrain_early_stop_min_delta
                mean_loss = float(_baseline_train_loss)
                epochs_done = 0
                _steps_this_cycle = 0

                for ep in range(epochs):
                    indices = torch.randperm(n)
                    ep_value_losses = []
                    ep_sq_errs: list[torch.Tensor] = []
                    ep_kl_totals = []
                    ep_head_kl_accum: dict[str, list[float]] = {}
                    for start in range(0, n, _batch_size):
                        mb_idx = indices[start:start + _batch_size]
                        mb_obs = {k.replace("obs/", ""): train_batch[k][mb_idx].to(self.device)
                                  for k in train_batch if k.startswith("obs/")}
                        mb_actions = {k.replace("action/", ""): train_batch[k][mb_idx].to(self.device)
                                      for k in train_batch if k.startswith("action/")}
                        mb_ret = returns_t[mb_idx]
                        # anchor's tensors live on self.device (built by
                        # _ppg_snapshot_anchor's forward passes); mb_idx itself is a
                        # plain CPU torch.randperm() slice (matching train_batch's
                        # own CPU-tensor indexing above) -- move it once for the
                        # anchor lookup so this works whether self.device is "cpu"
                        # (index device == tensor device already, a no-op move) or
                        # a real accelerator (where indexing a device tensor with a
                        # CPU index tensor would otherwise raise).
                        mb_idx_dev = mb_idx.to(self.device)
                        mb_anchor = {k: v[mb_idx_dev] for k, v in anchor.items()}

                        sf, of, em = mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"]
                        bf, gf = mb_obs["ball_feat"], mb_obs["global_feat"]
                        sat, oat = _ai_types(mb_obs)

                        d_heads = self.decision_net(
                            sf, of, em, bf, gf, sat, oat,
                            ball_physics_full=mb_obs.get("ball_physics_full"),
                            self_physics_full=mb_obs.get("self_physics_full"),
                            other_physics_full=mb_obs.get("other_physics_full"),
                        )
                        e_heads = self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat)
                        new_values = e_heads.value.squeeze(-1)
                        value_loss = F.mse_loss(new_values, mb_ret) / (ret_std ** 2)

                        total_kl, per_head_kl = self._ppg_kl_penalty(
                            d_heads, e_heads, mb_anchor, anchor_globals, mb_actions, em,
                        )
                        loss = value_loss + kl_coef * total_kl

                        ppg_opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(_non_direction_params, self.max_grad_norm)
                        if _direction_params:
                            nn.utils.clip_grad_norm_(_direction_params, self.direction_max_grad_norm)
                        if self._ppg_warmup_steps > 0:
                            _warm_scale = _lr_warmup_scale(_steps_this_cycle, self._ppg_warmup_steps)
                            for _g in ppg_opt.param_groups:
                                _g["lr"] = lr * _warm_scale
                        _steps_this_cycle += 1
                        ppg_opt.step()

                        ep_value_losses.append(value_loss.item())
                        # Pre-step training-forward errors (same convention as the
                        # plain train rmse from ep_value_losses above), kept per row
                        # so the trimmed rmse below can drop the worst tail.
                        ep_sq_errs.append(((new_values.detach() - mb_ret) ** 2))
                        ep_kl_totals.append(total_kl.item())
                        if self._ppg_minibatch_trace is not None:
                            self._ppg_minibatch_trace.append((ep, ep_value_losses[-1], ep_kl_totals[-1]))
                        for k, v in per_head_kl.items():
                            ep_head_kl_accum.setdefault(k, []).append(v.item())

                    mean_loss = float(np.mean(ep_value_losses))
                    mean_kl_train = float(np.mean(ep_kl_totals))
                    epochs_done = ep + 1
                    _train_rmse = float(ret_std) * math.sqrt(mean_loss)
                    _train_trim_rmse = _trimmed_rmse(torch.cat(ep_sq_errs), self._ppg_rmse_trim_frac)
                    _inactive = self._inactive_head_lp_keys()
                    _head_kl_train_str = "  ".join(
                        f"{k}={np.mean(v):+.4f}" for k, v in ep_head_kl_accum.items() if k not in _inactive
                    )

                    if val_obs_dict is not None and val_returns_t is not None and val_anchor is not None:
                        _vl, _val_rmse, _val_trim_rmse, mean_kl_val, _val_head_kl = _eval_value_and_kl(
                            val_obs_dict, val_actions_dict, val_returns_t, val_anchor,
                        )
                        log.info(
                            f"  PPG refit epoch {epochs_done}/{epochs}: "
                            f"train={mean_loss:.4f} rmse={_train_rmse:.2f} rmse_{_trim_tag}={_train_trim_rmse:.2f}  "
                            f"val={_vl:.4f} val_rmse={_val_rmse:.2f} val_rmse_{_trim_tag}={_val_trim_rmse:.2f} "
                            f"(std={float(ret_std):.1f})  "
                            f"kl_train={mean_kl_train:.4f} kl_val={mean_kl_val:.4f}"
                        )
                        log.info(f"    [ppg aux KL by head] (train) {_head_kl_train_str}")
                        _head_kl_val_str = "  ".join(
                            f"{k}={v:+.4f}" for k, v in _val_head_kl.items() if k not in _inactive
                        )
                        log.info(f"    [ppg aux KL by head] (val)   {_head_kl_val_str}")
                        if _vl < _best_val_loss - _EARLY_STOP_MIN_DELTA:
                            _best_val_loss = _vl
                            _patience = 0
                            _best_decision_state = copy.deepcopy(self.decision_net.state_dict())
                            _best_execution_state = copy.deepcopy(self.execution_net.state_dict())
                        else:
                            _patience += 1
                            if _patience >= _EARLY_STOP_PATIENCE:
                                log.info(
                                    f"  [ppg value refit] early stop at epoch {epochs_done} "
                                    f"(val stagnant for {_EARLY_STOP_PATIENCE} epochs, best={_best_val_loss:.4f})"
                                )
                                break
                    else:
                        log.info(
                            f"  PPG refit epoch {epochs_done}/{epochs}: "
                            f"train_loss={mean_loss:.4f}  rmse={_train_rmse:.2f} rmse_{_trim_tag}={_train_trim_rmse:.2f} "
                            f"(returns std={float(ret_std):.1f})  kl_train={mean_kl_train:.4f}"
                        )
                        log.info(f"    [ppg aux KL by head] (train) {_head_kl_train_str}")

                if _best_decision_state is not None:
                    # NOT a plain strict load_state_dict -- decision_net can carry
                    # optional submodules (e.g. the frozen physics-dynamics encoders)
                    # that may or may not be part of every checkpoint/config
                    # combination; _load_state_dict_tolerant is the same forgiving
                    # loader every other best-state restoration in this file already
                    # uses for exactly this reason (see pretrain_combined's Phase 0
                    # best-state handling).
                    _load_state_dict_tolerant(self.decision_net, _best_decision_state, "ppg_value_refit decision_net restore")
                    _load_state_dict_tolerant(self.execution_net, _best_execution_state, "ppg_value_refit execution_net restore")
                    log.info(f"  [ppg value refit] restored best-val weights (val_loss={_best_val_loss:.4f})")
                _cycle_label = f" (rollout {_rollout_i + 1}/{num_rollouts})" if num_rollouts > 1 else ""
                log.info(f"PPG value refit{_cycle_label} done ({epochs_done} epoch(s), final train_loss={mean_loss:.4f})")

                if self._ppg_log_prev_checkpoint_value_loss:
                    # Record THIS cycle's final weights (post best-val-restore) for
                    # the prev-checkpoint value-loss diagnostic above -- feeds the
                    # NEXT cycle's reading. Capped at 2 (oldest dropped) since that's
                    # all the diagnostic ever looks at. State dicts only (not full
                    # module deepcopies) to keep the memory cost modest.
                    _cycle_end_snapshots.append((
                        _rollout_i + 1,
                        copy.deepcopy(self.decision_net.state_dict()),
                        copy.deepcopy(self.execution_net.state_dict()),
                    ))
                    _cycle_end_snapshots = _cycle_end_snapshots[-2:]

                # Checkpoint after EVERY cycle, not just once at the end -- a
                # multi-rollout call can run a long time (each rollout alone can
                # take a while at real ppg_rollout_steps sizes), so this bounds
                # how much work a crash/interrupt partway through actually loses.
                # Same filename train.py itself saves to after this method
                # returns, so that final save is just a harmless idempotent
                # re-save of whatever the last cycle already wrote here.
                if self.checkpoint_dir is not None:
                    _ckpt_path = self.checkpoint_dir / "checkpoint_pretrained.pt"
                    self._save_checkpoint_to(_ckpt_path)
                    log.info(f"  [ppg value refit] checkpoint saved to {_ckpt_path}")

                # PPG's own rollout-to-rollout episode-seed replay (see
                # ppg_episode_replay_enabled's config comment): report this cycle's
                # before/after on whatever the PREVIOUS cycle queued, then select this
                # cycle's own worst-value-MSE episodes to queue for the NEXT cycle.
                # Skipped on the last cycle -- there is no next cycle within this call
                # to consume a fresh selection, so it would be dead work.
                if self._ppg_episode_replay_enabled and _rollout_i < num_rollouts - 1:
                    _ppg_pending_replay_before, _ppg_next_cycle_replay_seeds = self._ppg_update_episode_replay(
                        _rollout_stats.get("episode_seed_mse", []), _ppg_pending_replay_before,
                    )

                # Drop this cycle's big tensors NOW. Names bound inside a loop
                # body live until they're reassigned, so without this the whole
                # previous cycle (raw + augmented train batch, obs/anchor
                # snapshots, val copies -- easily several GB) would stay
                # resident throughout the NEXT rollout's collection, on top of
                # that rollout's own linear growth. Assigning None (rather
                # than ``del``) is safe whether or not a name was already
                # deleted/unset on this path.
                batch = train_batch = train_batch_raw = val_batch_raw = None
                returns_t = val_returns_t = all_returns_t = None
                _train_obs_full = val_obs_dict = val_actions_dict = None
                anchor = val_anchor = None
        finally:
            self._close_value_pretrain_workers(_pool)

        return _rollout_stats

    def consistency_refit(
        self,
        env,
        n_steps: int,
        phase_id: Optional[int] = None,
        teacher_checkpoint: Optional[Path] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        num_rollouts: Optional[int] = None,
        val_episode_fraction: Optional[float] = None,
        logit_bound: Optional[float] = None,
        kappa_deg: Optional[float] = None,
    ) -> dict:
        """flip_y consistency refit: make the policy give the same answer to a state and its y-mirror.

        Each cycle collects a fresh rollout with the CURRENT (student) policy, then for every state:
        the "primary" orientation is the one where the observer's own y is >= 0; a frozen teacher
        (``teacher_checkpoint`` -- the network BEFORE any --reset-bernoullis/kappa reset -- or a copy
        of the current nets if None) is evaluated on it; the student is trained (analytic per-head KL,
        the same ``_ppg_kl_penalty`` PPG uses) so that the primary state reproduces the teacher's
        output and the mirror state reproduces the y-mirrored output. Both orientations therefore
        share one target. Bernoulli targets are clamped to ``+-logit_bound`` so the trained heads end
        up bounded (default 6.907 = probabilities within [0.001, 0.999]).

        Frozen for the duration (restored afterwards): the direction/power/spin spread parameters
        (a trainable kappa would let the consistency KL be reduced by simply widening the
        distributions), the value head + its value-only helper modules (the shared trunk still
        moves, so value error is logged before/after), and the physics encoders (always frozen).
        ``kappa_deg`` > 0 first sets move_dir/kick_dir kappa to 1/radians(kappa_deg)**2.

        Cycle structure mirrors ``ppg_value_refit`` (persistent worker pool, one Adam across cycles,
        episode-level train/val split, best-val restore, checkpoint every cycle -- here to
        ``checkpoint_consistency.pt``). Held-out validation reports the quantity that actually
        matters, the student's own mirror consistency (angle error, hard-decision disagreement),
        alongside teacher fidelity, boundedness and value drift.
        """
        epochs = epochs if epochs is not None else self._consistency_epochs
        lr = lr if lr is not None else self._consistency_lr
        _bs = batch_size if batch_size is not None else self._value_pretrain_batch_size
        num_rollouts = num_rollouts if num_rollouts is not None else self._consistency_num_rollouts
        val_frac = val_episode_fraction if val_episode_fraction is not None else self._consistency_val_episode_fraction
        bound = logit_bound if logit_bound is not None else self._consistency_logit_bound
        kdeg = kappa_deg if kappa_deg is not None else self._consistency_kappa_deg

        if teacher_checkpoint is not None:
            from footballcoach.ai.ppo.ppo_trainer import PPOTrainer  # local: imported by PPOTrainer's module

            _t = PPOTrainer.from_config(
                device=self.device, inference_only=True, separate_value_net=self.separate_value_net,
            )
            _t.load_checkpoint(Path(teacher_checkpoint))
            teacher_nets = (_t.decision_net.eval(), _t.execution_net.eval())
        else:
            teacher_nets = (copy.deepcopy(self.decision_net).eval(), copy.deepcopy(self.execution_net).eval())
        for _m in teacher_nets:
            for _p in _m.parameters():
                _p.requires_grad_(False)

        en = self.execution_net
        _spread_params = [en.move_dir_log_kappa, en.kick_dir_log_kappa, en.kick_dir_z_log_std,
                          en.kick_power_log_std, en.kick_spin_log_std]
        if kdeg > 0:
            _kappa = 1.0 / math.radians(kdeg) ** 2
            with torch.no_grad():
                en.move_dir_log_kappa.fill_(math.log(_kappa))
                en.kick_dir_log_kappa.fill_(math.log(_kappa))
        _value_only_names = ("value_head", "value_ai_type_channel")
        _restore_grad: list[tuple[torch.nn.Parameter, bool]] = []
        for _name, _p in list(self.decision_net.named_parameters()) + list(self.execution_net.named_parameters()):
            if any(s in _name for s in _value_only_names) or any(_p is g for g in _spread_params):
                _restore_grad.append((_p, _p.requires_grad))
                _p.requires_grad_(False)
        trainable = [p for p in list(self.decision_net.parameters()) + list(self.execution_net.parameters()) if p.requires_grad]
        _dir_ids = getattr(self, "direction_param_ids", set())
        _non_dir = [p for p in trainable if id(p) not in _dir_ids]
        _dir = [p for p in trainable if id(p) in _dir_ids]
        opt = torch.optim.Adam(trainable, lr=lr, eps=1e-5)
        anchor_globals = {
            "move_dir_log_kappa": en.move_dir_log_kappa.detach().clone(),
            "kick_dir_log_kappa": en.kick_dir_log_kappa.detach().clone(),
            "kick_dir_z_log_std": en.kick_dir_z_log_std.detach().clone(),
            "kick_power_log_std": en.kick_power_log_std.detach().clone(),
        }
        if not self._kick_spin_frozen:
            anchor_globals["kick_spin_log_std"] = en.kick_spin_log_std.detach().clone()
        _kd = math.degrees(1.0 / math.sqrt(math.exp(en.move_dir_log_kappa.item())))
        log.info(
            f"Consistency refit: {num_rollouts} rollout(s) x {n_steps} steps, {epochs} epochs, lr={lr}, "
            f"batch_size={_bs}, val_episode_fraction={val_frac}, logit_bound={bound}, "
            f"move_dir kappa={math.exp(en.move_dir_log_kappa.item()):.2f} (~{_kd:.1f} deg std), "
            f"{len(trainable)} trainable tensors ({len(_restore_grad)} frozen: spread params + value-only modules)"
        )

        def _batch_obs(d: dict, idx=None, sl=None):
            src = (lambda v: v[idx]) if idx is not None else (lambda v: v[sl])
            return {k: src(v).to(self.device) for k, v in d.items()}

        def _forward(mb_obs):
            sat, oat = _ai_types(mb_obs)
            d_heads = self.decision_net(
                mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"],
                mb_obs["ball_feat"], mb_obs["global_feat"], sat, oat,
                ball_physics_full=mb_obs.get("ball_physics_full"),
                self_physics_full=mb_obs.get("self_physics_full"),
                other_physics_full=mb_obs.get("other_physics_full"),
            )
            e_heads = self.execution_net(
                mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"],
                mb_obs["ball_feat"], mb_obs["global_feat"], d_heads, sat, oat,
            )
            return d_heads, e_heads

        _ACT_KEYS = ("exec_move", "kick", "pass_", "tackle", "mark")

        def _build_set(raw: dict) -> dict:
            """2n rows: first n = primary orientation, last n = its y-mirror, with matching teacher anchors."""
            obs = {k: v for k, v in raw.items() if k.startswith("obs/")}
            n = len(obs["obs/self_feat"])
            primary, mirror = _consistency.orientation_pair(obs)
            both = {k: torch.cat([primary[k], mirror[k]], dim=0) for k in obs}
            both.update(self._precompute_physics_full(both, _bs))
            obs_dict = {k.replace("obs/", ""): v for k, v in both.items()}
            teacher_obs = {k: v[:n] for k, v in obs_dict.items() if not k.endswith("physics_full")}
            anc_p = _consistency.bound_bernoulli_anchor(
                self._ppg_snapshot_anchor(teacher_obs, _bs, nets=teacher_nets), bound,
            )
            anchor = _consistency.concat_anchors(anc_p, _consistency.mirror_anchor_y(anc_p))
            acts = {k: torch.cat([raw[f"action/{k}"]] * 2, dim=0) for k in _ACT_KEYS}
            return {"obs": obs_dict, "acts": acts, "anchor": anchor,
                    "returns": torch.cat([raw["returns"]] * 2, dim=0), "n": n}

        def _evaluate(ds: dict, ret_std: float) -> dict:
            n = ds["n"]
            kl_sum, head_sum, sq_sum = 0.0, {}, 0.0
            rows: dict[str, list[torch.Tensor]] = {}
            with torch.no_grad():
                for s in range(0, 2 * n, _bs):
                    sl = slice(s, min(s + _bs, 2 * n))
                    mb = _batch_obs(ds["obs"], sl=sl)
                    d_heads, e_heads = _forward(mb)
                    acts = {k: v[sl].to(self.device) for k, v in ds["acts"].items()}
                    anc = {k: v[sl] for k, v in ds["anchor"].items()}
                    total_kl, per_head = self._ppg_kl_penalty(d_heads, e_heads, anc, anchor_globals, acts, mb["exists_mask"])
                    bsz = sl.stop - sl.start
                    kl_sum += float(total_kl) * bsz
                    for k, v in per_head.items():
                        head_sum[k] = head_sum.get(k, 0.0) + float(v) * bsz
                    sq_sum += float(((e_heads.value.squeeze(-1) - ds["returns"][sl].to(self.device)) ** 2).sum())
                    for k, v in self._snapshot_row(d_heads, e_heads).items():
                        rows.setdefault(k, []).append(v.detach())
            student = {k: torch.cat(v, dim=0) for k, v in rows.items()}
            sp = {k: v[:n] for k, v in student.items()}
            sq = {k: v[n:] for k, v in student.items()}
            moving = ds["acts"]["exec_move"][:n].squeeze(-1).to(sp["move_direction"].device) > 0.5
            out = {
                "kl": kl_sum / (2 * n), "per_head": {k: v / (2 * n) for k, v in head_sum.items()},
                "value_norm_mse": sq_sum / (2 * n) / ret_std ** 2,
                "consistency": _consistency.flip_consistency_metrics(sp, sq, moving),
                "bounded_pct": _consistency.bounded_share_pct(sp, bound if bound > 0 else 6.907),
            }
            anc_p = {k: v[:n] for k, v in ds["anchor"].items()}
            agree = {}
            for key in ("exec_move_logit", "sprint_logit", "kick_logit", "tackle_attempt_logit"):
                agree[key.replace("_logit", "")] = float(((sp[key] > 0) == (anc_p[key] > 0)).float().mean() * 100)
            ang_s = torch.atan2(sp["move_direction"][:, 1], sp["move_direction"][:, 0])
            ang_t = torch.atan2(anc_p["move_direction"][:, 1], anc_p["move_direction"][:, 0])
            err = torch.rad2deg((torch.remainder(ang_s - ang_t + math.pi, 2 * math.pi) - math.pi).abs())[moving]
            out["teacher_agree_pct"] = agree
            out["teacher_move_dir_err_p50_deg"] = float(err.quantile(0.5)) if len(err) else float("nan")
            return out

        def _fmt(m: dict, tag: str) -> list[str]:
            c = m["consistency"]
            heads = ("exec_move", "sprint", "kick", "tackle_attempt")
            return [
                f"    [consistency {tag}] val_kl={m['kl']:.4f}  value_norm_mse={m['value_norm_mse']:.4f}",
                f"    [consistency {tag}] flip-consistency: move_dir err p50/p90={c['move_dir_err_p50_deg']:.1f}/{c['move_dir_err_p90_deg']:.1f} deg "
                f">90deg={c['move_dir_err_gt90_pct']:.1f}%  |  hard-decision disagree%: "
                + "  ".join(f"{h}={c.get(h + '_sign_disagree_pct', float('nan')):.1f}" for h in heads),
                f"    [consistency {tag}] teacher fidelity: move_dir err p50={m['teacher_move_dir_err_p50_deg']:.1f} deg  decision agree%: "
                + "  ".join(f"{h}={m['teacher_agree_pct'][h]:.1f}" for h in heads)
                + "  |  within-bound%: " + "  ".join(f"{k}={v:.1f}" for k, v in m["bounded_pct"].items()),
                "    [consistency " + tag + "] KL by head: " + "  ".join(f"{k}={v:+.4f}" for k, v in m["per_head"].items()
                                                                       if k not in self._inactive_head_lp_keys()),
            ]

        _pool = self._spawn_value_pretrain_workers(phase_id)
        _first_val: Optional[dict] = None
        _last_val: dict = {}
        try:
            for _ci in range(num_rollouts):
                log.info(f"=== Consistency refit: rollout {_ci + 1}/{num_rollouts} ===")
                batch, _stats = self._collect_value_pretrain_rollout(env, n_steps, phase_id, use_gae=True, pool=_pool)
                dones = batch["dones"].numpy()
                ends = np.where(dones > 0.5)[0]
                n_eps = len(ends)
                n_val_eps = max(1, round(val_frac * n_eps)) if (val_frac > 0.0 and n_eps >= 2) else 0
                n_train_eps = n_eps - n_val_eps
                val_mask = np.zeros(len(dones), dtype=bool)
                if n_val_eps > 0:
                    starts = np.concatenate([[0], ends[:-1] + 1])
                    for _i in range(n_train_eps, n_eps):
                        val_mask[starts[_i]:ends[_i] + 1] = True
                ret_std = float(batch["returns"].std().clamp(min=1.0))
                train_raw, val_raw = _split_batch_releasing(batch, ~val_mask, val_mask if n_val_eps > 0 else None)
                log.info(
                    f"  consistency split: {n_train_eps} train eps ({int((~val_mask).sum())} states)"
                    + (f"  |  {n_val_eps} val eps ({int(val_mask.sum())} states)" if n_val_eps > 0 else "")
                    + "  (each state is used in both orientations)"
                )
                train = _build_set(train_raw)
                val = _build_set(val_raw) if val_raw is not None else None
                train_raw = val_raw = None
                n2 = 2 * train["n"]

                if val is not None:
                    _m0 = _evaluate(val, ret_std)
                    if _first_val is None:
                        _first_val = _m0
                    for _ln in _fmt(_m0, "epoch 0 (baseline)"):
                        log.info(_ln)

                best_val, best_state, patience, steps = float("inf"), None, 0, 0
                for ep in range(epochs):
                    perm = torch.randperm(n2)
                    kl_list, head_acc = [], {}
                    for s in range(0, n2, _bs):
                        idx = perm[s:s + _bs]
                        mb = _batch_obs(train["obs"], idx=idx)
                        acts = {k: v[idx].to(self.device) for k, v in train["acts"].items()}
                        idx_dev = idx.to(self.device)
                        anc = {k: v[idx_dev] for k, v in train["anchor"].items()}
                        d_heads, e_heads = _forward(mb)
                        total_kl, per_head = self._ppg_kl_penalty(d_heads, e_heads, anc, anchor_globals, acts, mb["exists_mask"])
                        opt.zero_grad()
                        total_kl.backward()
                        nn.utils.clip_grad_norm_(_non_dir, self.max_grad_norm)
                        if _dir:
                            nn.utils.clip_grad_norm_(_dir, self.direction_max_grad_norm)
                        if self._consistency_warmup_steps > 0:
                            _w = _lr_warmup_scale(steps, self._consistency_warmup_steps)
                            for _g in opt.param_groups:
                                _g["lr"] = lr * _w
                        steps += 1
                        opt.step()
                        kl_list.append(total_kl.item())
                        for k, v in per_head.items():
                            head_acc.setdefault(k, []).append(v.item())
                    log.info(f"  consistency epoch {ep + 1}/{epochs}: train_kl={np.mean(kl_list):.4f}")
                    if val is not None:
                        _last_val = _evaluate(val, ret_std)
                        for _ln in _fmt(_last_val, f"epoch {ep + 1}/{epochs} val"):
                            log.info(_ln)
                        if _last_val["kl"] < best_val - self._value_pretrain_early_stop_min_delta:
                            best_val, patience = _last_val["kl"], 0
                            best_state = (copy.deepcopy(self.decision_net.state_dict()), copy.deepcopy(self.execution_net.state_dict()))
                        else:
                            patience += 1
                            if patience >= self._value_pretrain_early_stop_patience:
                                log.info(f"  [consistency] early stop at epoch {ep + 1} (val_kl stagnant, best={best_val:.4f})")
                                break
                if best_state is not None:
                    _load_state_dict_tolerant(self.decision_net, best_state[0], "consistency_refit decision_net restore")
                    _load_state_dict_tolerant(self.execution_net, best_state[1], "consistency_refit execution_net restore")
                    log.info(f"  [consistency] restored best-val weights (val_kl={best_val:.4f})")
                if self.checkpoint_dir is not None:
                    _ckpt = self.checkpoint_dir / "checkpoint_consistency.pt"
                    self._save_checkpoint_to(_ckpt)
                    log.info(f"  [consistency] checkpoint saved to {_ckpt}")
                train = val = None
        finally:
            self._close_value_pretrain_workers(_pool)
            for _p, _rg in _restore_grad:
                _p.requires_grad_(_rg)

        if _first_val is not None and _last_val:
            log.info("Consistency refit summary (val, first-cycle baseline -> last epoch):")
            for _ln in _fmt(_first_val, "baseline"):
                log.info(_ln)
            for _ln in _fmt(_last_val, "final"):
                log.info(_ln)
        return {"first_val": _first_val, "last_val": _last_val}

    @torch.no_grad()
    def _ppg_snapshot_anchor(
        self, obs: dict, batch_size: int, progress: Optional["ProgressReporter"] = None,
        nets: Optional[tuple] = None,
    ) -> dict[str, torch.Tensor]:
        """Snapshot the state-dependent policy outputs needed to compute a
        per-head KL penalty later, for ``ppg_value_refit()``. Called ONCE,
        under the CURRENT (about-to-be-refit) weights, before that method's
        epoch loop begins -- this is the "anchor" the policy gets pulled
        back toward for the rest of the refit.

        Only captures per-row, STATE-DEPENDENT outputs (Bernoulli/Categorical
        logits, and the raw mean vector for the continuous heads) -- the
        continuous heads' spread parameters (``move_dir_log_kappa`` etc.) are
        global ``nn.Parameter``s, not state-dependent, so ``ppg_value_refit``
        snapshots those separately as a handful of scalars, not per-row.

        Runs in minibatch-sized chunks (not one giant forward pass) to bound
        memory, mirroring every other full-batch pass in this file (e.g.
        ``pretrain_value``'s own baseline-loss eval).

        Args:
            obs: dict of ``obs/*``-stripped observation tensors (kept on CPU
                on purpose -- only one chunk at a time is moved to
                ``self.device``, see ``ppg_value_refit``), e.g.
                ``{"self_feat": ..., "other_feat": ...}``.
            batch_size: chunk size for the forward passes.
            progress: optional ``ProgressReporter`` (total = number of rows),
                updated after every chunk -- purely cosmetic, so the caller
                can show a bar for what is otherwise a silent, slow pass.

        Returns:
            dict of per-row tensors, same row order as ``obs``, keyed by the
            network's own attribute names (``shoot_logit``, ``pass_logit``,
            ..., ``pass_target_logits``, ..., ``move_direction``,
            ``kick_direction``, ``kick_power``, and ``kick_spin`` only when
            ``self._kick_spin_frozen`` is False).
        """
        n = len(obs["self_feat"])
        chunks: dict[str, list[torch.Tensor]] = {}
        # ``nets`` = (decision_net, execution_net) of some OTHER network (e.g. a frozen teacher);
        # its frozen physics encoders are re-run on the obs rather than fed the caller's
        # precomputed physics_full, so it never depends on the caller's encoder weights.
        _dec_net, _exec_net = nets if nets is not None else (self.decision_net, self.execution_net)
        for start in range(0, n, batch_size):
            mb_obs = {k: v[start:start + batch_size].to(self.device) for k, v in obs.items()}
            sat, oat = _ai_types(mb_obs)
            _phys = {} if nets is not None else {
                "ball_physics_full": mb_obs.get("ball_physics_full"),
                "self_physics_full": mb_obs.get("self_physics_full"),
                "other_physics_full": mb_obs.get("other_physics_full"),
            }
            d_heads = _dec_net(
                mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"],
                mb_obs["ball_feat"], mb_obs["global_feat"], sat, oat, **_phys,
            )
            e_heads = _exec_net(
                mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"],
                mb_obs["ball_feat"], mb_obs["global_feat"], d_heads, sat, oat,
            )
            row = self._snapshot_row(d_heads, e_heads)
            for k, v in row.items():
                chunks.setdefault(k, []).append(v.detach())
            if progress is not None:
                progress.update(min(start + batch_size, n))
        return {k: torch.cat(v, dim=0) for k, v in chunks.items()}

    def _snapshot_row(self, d_heads, e_heads) -> dict[str, torch.Tensor]:
        """The per-row state-dependent head outputs ``_ppg_kl_penalty`` needs (the "anchor" layout)."""
        row: dict[str, torch.Tensor] = {
            "shoot_logit": d_heads.shoot_logit,
            "pass_logit": d_heads.pass_logit,
            "move_logit": d_heads.move_logit,
            "tackle_logit": d_heads.tackle_logit,
            "get_possession_raw": d_heads.get_possession_raw,
            "mark_logit": d_heads.mark_logit,
            "hold_position_logit": d_heads.hold_position_logit,
            "pass_target_logits": d_heads.pass_target_logits,
            "tackle_target_logits": d_heads.tackle_target_logits,
            "mark_target_logits": d_heads.mark_target_logits,
            "exec_move_logit": e_heads.exec_move_logit,
            "sprint_logit": e_heads.sprint_logit,
            "kick_logit": e_heads.kick_logit,
            "tackle_attempt_logit": e_heads.tackle_attempt_logit,
            "move_direction": e_heads.move_direction,
            "kick_direction": e_heads.kick_direction,
            "kick_power": e_heads.kick_power,
        }
        if not self._kick_spin_frozen:
            row["kick_spin"] = e_heads.kick_spin
        return row

    def _ppg_kl_penalty(
        self, d_heads, e_heads, anchor: dict[str, torch.Tensor],
        anchor_globals: dict[str, torch.Tensor], mb_actions: dict, exists_mask: torch.Tensor,
        return_per_row: bool = False,
    ) -> tuple:
        """Analytic per-head KL(current policy || anchor snapshot), for
        ``ppg_value_refit()``'s KL-anchor loss term.

        Mirrors ``_recompute_log_prob``'s gating exactly (hard boolean mask
        for the target-categorical heads keyed on the parent's actually-
        sampled action; float exec_move/kick masks for their respective
        sub-heads) so the SAME rows/heads that would carry real log_prob
        signal during ordinary PPO training are the ones being KL-anchored
        here -- and heads currently in ``_inactive_head_lp_keys()``
        (curriculum-frozen decision heads, or permanently-frozen kick_spin)
        are skipped entirely, exactly like the ``[per-head KL]`` diagnostic
        already does.

        Every head except move_dir/kick_dir's azimuthal component has a
        ``torch.distributions.kl_divergence``-registered closed form
        (Bernoulli/Categorical/Normal) -- reusing the SAME
        ``_move_dir_head``/``_kick_dir_head``/``_kick_power_head``/
        ``_kick_spin_dist`` constructors already used for log_prob/entropy
        for BOTH the current and anchor distributions guarantees identical
        clamping/construction on both sides. move_dir/kick_dir's azimuthal
        part uses the hand-derived ``_von_mises_kl`` (torch has none
        registered for VonMises).

        Returns:
            (total_kl, per_head_kl): ``total_kl`` is the batch-mean summed-
            across-heads KL (for the loss). ``per_head_kl`` is a dict keyed
            by ``HEAD_LP_KEYS`` (the same 15 short names the ``[per-head
            KL]``/entropy diagnostics use) of that head's batch-mean KL, for
            a ``[ppg aux KL by head]`` log line -- target-categorical KL is
            folded into its parent's entry (``pass_target`` -> ``pass_``
            etc.), matching how ``_recompute_log_prob`` never surfaces the
            target categoricals as a separate scalar either.
        """
        device = self.device
        bsz = exists_mask.shape[0]
        zero = torch.zeros(bsz, device=device)
        inactive = self._inactive_head_lp_keys()

        # A logit past ~+-16.7 makes torch.sigmoid(logit) round to EXACTLY
        # 0.0/1.0 in float32 (1 - 2**-24 is indistinguishable from 1.0 at
        # that precision). torch's own Bernoulli-Bernoulli KL formula
        # deliberately returns +inf when the ANCHOR side has hit that exact
        # boundary while the current side hasn't (a real property of KL --
        # q assigning literally zero probability to something p doesn't is
        # genuinely infinite divergence) -- correct in isolation, but fatal
        # here: exec_move_mask/kick_mask then multiply that +inf by 0.0 on
        # non-firing rows (IEEE 0*inf = nan), poisoning this row's per-head
        # sum and then the whole minibatch's .mean(). A mature, confident
        # checkpoint (any head that's near-always/near-never true, e.g.
        # move/tackle_attempt after enough PPO training) hits this easily.
        # Clamped well inside the float32 saturation boundary on both sides
        # before computing KL -- this only affects the KL PENALTY term, not
        # the actual sampled action/log_prob/entropy used anywhere else.
        _BERNOULLI_LOGIT_CLAMP = 12.0

        def _bern_kl(new_logit, anchor_logit):
            new_c = new_logit.clamp(-_BERNOULLI_LOGIT_CLAMP, _BERNOULLI_LOGIT_CLAMP)
            anchor_c = anchor_logit.clamp(-_BERNOULLI_LOGIT_CLAMP, _BERNOULLI_LOGIT_CLAMP)
            return torch.distributions.kl_divergence(
                torch.distributions.Bernoulli(logits=new_c),
                torch.distributions.Bernoulli(logits=anchor_c),
            ).squeeze(-1)

        def _cat_kl(new_logits, anchor_logits):
            return torch.distributions.kl_divergence(
                MaskedCategorical(new_logits, exists_mask).dist,
                MaskedCategorical(anchor_logits, exists_mask).dist,
            )

        kl_shoot = zero if "shoot" in inactive else _bern_kl(d_heads.shoot_logit, anchor["shoot_logit"])
        kl_move = zero if "move" in inactive else _bern_kl(d_heads.move_logit, anchor["move_logit"])
        kl_gp = zero if "gp_extra" in inactive else _bern_kl(d_heads.get_possession_raw, anchor["get_possession_raw"])
        kl_hold = zero if "hold" in inactive else _bern_kl(d_heads.hold_position_logit, anchor["hold_position_logit"])

        # Parent Bernoulli KL + (gated) target-categorical KL folded into one
        # entry, exactly mirroring _recompute_log_prob's pass_mask/tackle_mask/
        # mark_mask boolean-row-subset convention for the categorical target
        # heads -- pass_/tackle/mark each have one, shoot/move/gp_extra/hold
        # don't (see the plain _bern_kl-only heads above).
        def _parent_plus_target_kl(short_key, parent_logit, anchor_parent_logit,
                                    action_key, target_logits, anchor_target_logits):
            if short_key in inactive:
                return zero
            kl = _bern_kl(parent_logit, anchor_parent_logit)
            mask = mb_actions[action_key].squeeze(-1) > 0.5
            if mask.any():
                cat_kl = _cat_kl(target_logits, anchor_target_logits)
                kl = kl.clone()
                kl[mask] = kl[mask] + cat_kl[mask]
            return kl

        kl_pass = _parent_plus_target_kl(
            "pass_", d_heads.pass_logit, anchor["pass_logit"], "pass_",
            d_heads.pass_target_logits, anchor["pass_target_logits"],
        )
        kl_tackle = _parent_plus_target_kl(
            "tackle", d_heads.tackle_logit, anchor["tackle_logit"], "tackle",
            d_heads.tackle_target_logits, anchor["tackle_target_logits"],
        )
        kl_mark = _parent_plus_target_kl(
            "mark", d_heads.mark_logit, anchor["mark_logit"], "mark",
            d_heads.mark_target_logits, anchor["mark_target_logits"],
        )

        exec_move_mask = (mb_actions["exec_move"].squeeze(-1) > 0.5).float()
        kick_mask = (mb_actions["kick"].squeeze(-1) > 0.5).float()

        kl_exec_move = zero if "exec_move" in inactive else _bern_kl(e_heads.exec_move_logit, anchor["exec_move_logit"])
        kl_sprint = zero if "sprint" in inactive else exec_move_mask * _bern_kl(e_heads.sprint_logit, anchor["sprint_logit"])
        kl_kick = zero if "kick" in inactive else _bern_kl(e_heads.kick_logit, anchor["kick_logit"])
        kl_tackle_attempt = (
            zero if "tackle_attempt" in inactive
            else _bern_kl(e_heads.tackle_attempt_logit, anchor["tackle_attempt_logit"])
        )

        log_kappa_move = self.execution_net.move_dir_log_kappa.to(device)
        log_kappa_kick = self.execution_net.kick_dir_log_kappa.to(device)
        log_std_z_kick = self.execution_net.kick_dir_z_log_std.to(device)
        log_std_power = self.execution_net.kick_power_log_std.to(device)

        if "move_dir" in inactive:
            kl_move_dir = zero
        else:
            cur_move = self._move_dir_head(e_heads.move_direction, log_kappa_move)
            anc_move = self._move_dir_head(anchor["move_direction"], anchor_globals["move_dir_log_kappa"])
            kl_move_dir = exec_move_mask * _von_mises_kl(
                cur_move.mean_angle, cur_move.kappa, anc_move.mean_angle, anc_move.kappa,
            )

        if "kick_dir" in inactive:
            kl_kick_dir = zero
        else:
            cur_kick = self._kick_dir_head(e_heads.kick_direction, log_kappa_kick, log_std_z_kick)
            anc_kick = self._kick_dir_head(
                anchor["kick_direction"], anchor_globals["kick_dir_log_kappa"], anchor_globals["kick_dir_z_log_std"],
            )
            kl_kick_dir_azimuth = _von_mises_kl(cur_kick.theta_mean, cur_kick.kappa, anc_kick.theta_mean, anc_kick.kappa)
            kl_kick_dir_z = torch.distributions.kl_divergence(
                torch.distributions.Normal(cur_kick.mean_z, cur_kick.std_z),
                torch.distributions.Normal(anc_kick.mean_z, anc_kick.std_z),
            )
            kl_kick_dir = kick_mask * (kl_kick_dir_azimuth + kl_kick_dir_z)

        if "kick_power" in inactive:
            kl_kick_power = zero
        else:
            cur_power = self._kick_power_head(e_heads.kick_power, log_std_power)
            anc_power = self._kick_power_head(anchor["kick_power"], anchor_globals["kick_power_log_std"])
            kl_kick_power = kick_mask * torch.distributions.kl_divergence(cur_power.dist, anc_power.dist).sum(dim=-1)

        # _inactive_head_lp_keys() already folds in _kick_spin_frozen (see its
        # own docstring), so checking "kick_spin" in inactive alone suffices.
        if "kick_spin" in inactive:
            kl_kick_spin = zero
        else:
            log_std_spin = self.execution_net.kick_spin_log_std.to(device)
            cur_spin = self._kick_spin_dist(e_heads.kick_spin, log_std_spin)
            anc_spin = self._kick_spin_dist(anchor["kick_spin"], anchor_globals["kick_spin_log_std"])
            kl_kick_spin = kick_mask * torch.distributions.kl_divergence(cur_spin, anc_spin).sum(dim=-1)

        per_head = {
            "shoot": kl_shoot, "pass_": kl_pass, "move": kl_move, "tackle": kl_tackle,
            "gp_extra": kl_gp, "mark": kl_mark, "hold": kl_hold,
            "exec_move": kl_exec_move, "sprint": kl_sprint, "kick": kl_kick,
            "tackle_attempt": kl_tackle_attempt, "move_dir": kl_move_dir,
            "kick_dir": kl_kick_dir, "kick_power": kl_kick_power, "kick_spin": kl_kick_spin,
        }
        total_per_row = sum(per_head.values())
        per_head_mean = {k: v.mean() for k, v in per_head.items()}
        if return_per_row:
            return total_per_row.mean(), per_head_mean, total_per_row.detach(), {k: v.detach() for k, v in per_head.items()}
        return total_per_row.mean(), per_head_mean

    def _flip_consistency_stats(self, batch: dict, pair: dict) -> Optional[dict]:
        """How consistent is the (pre-update) policy between each state and its y-mirror?

        ``pair`` comes from ``_recompute_old_log_probs_for_augmented_batch(..., pair_out=pair)``.
        Returns the analytic KL(policy(state) || y-mirror of policy(flipped state)) summed over
        heads (``_ppg_kl_penalty``, the same quantity ``consistency_refit`` trains down), its per-head
        breakdown, and the hard-decision / direction-angle disagreement stats. None if unavailable.
        """
        from types import SimpleNamespace

        id_rows, flip_rows = pair.get("id"), pair.get("flip")
        if not id_rows or not flip_rows:
            return None
        n_base = len(id_rows["move_direction"])
        en = self.execution_net
        globals_ = {
            "move_dir_log_kappa": en.move_dir_log_kappa.detach(), "kick_dir_log_kappa": en.kick_dir_log_kappa.detach(),
            "kick_dir_z_log_std": en.kick_dir_z_log_std.detach(), "kick_power_log_std": en.kick_power_log_std.detach(),
        }
        if not self._kick_spin_frozen:
            globals_["kick_spin_log_std"] = en.kick_spin_log_std.detach()
        mirrored = _consistency.mirror_anchor_y(flip_rows)
        kl_sum, head_sum = 0.0, {}
        with torch.no_grad():
            for lo in range(0, n_base, self.minibatch_size):
                hi = min(lo + self.minibatch_size, n_base)
                ns = SimpleNamespace(**{k: v[lo:hi].to(self.device) for k, v in id_rows.items()})
                anchor = {k: v[lo:hi].to(self.device) for k, v in mirrored.items()}
                acts = {k: batch[f"action/{k}"][lo:hi].to(self.device) for k in ("exec_move", "kick", "pass_", "tackle", "mark")}
                em = batch["obs/exists_mask"][lo:hi].to(self.device)
                total, per_head = self._ppg_kl_penalty(ns, ns, anchor, globals_, acts, em)
                kl_sum += float(total) * (hi - lo)
                for k, v in per_head.items():
                    head_sum[k] = head_sum.get(k, 0.0) + float(v) * (hi - lo)
        moving = batch["action/exec_move"][:n_base].squeeze(-1) > 0.5
        metrics = _consistency.flip_consistency_metrics(id_rows, flip_rows, moving)
        return {"kl": kl_sum / n_base, "per_head": {k: v / n_base for k, v in head_sum.items()}, "metrics": metrics, "n": n_base}

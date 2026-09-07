"""Production DAgger (Ross, Gordon & Bagnell 2011) phase for Phase 1 BC
pretraining.

Promotes the prototype in ``debug_policy_net.py`` (repo root) -- alternate
BC training with closed-loop policy rollouts, aggregating every visited
state labelled by what Phase1RulesAI would actually do from there -- into
``PPOTrainer.pretrain_combined()``. Directly targets the exposure-bias gap
plain BC has: near-zero training loss doesn't bound closed-loop rollout
error, because BC never sees "recover from a slightly-off state" examples.

Kept as a separate module (mirroring ``bc.py``/``rollout_buffer.py``, both
already outside ``ppo_trainer.py``) rather than private ``PPOTrainer``
methods, so ``ppo_trainer.py`` doesn't grow further. Functions take
``trainer`` as an explicit first parameter (same convention
``debug_policy_net.py``'s own prototype already uses) instead of being
methods, and this module must NOT import ``ppo_trainer`` at module level --
that would be circular, since ``ppo_trainer.py`` imports this module to call
it from ``pretrain_combined()``. Anything needed from ``ppo_trainer.py``
(``_ai_types``) is imported lazily inside function bodies instead.
"""
from __future__ import annotations

import logging
import random as _random
from typing import Callable

import torch

log = logging.getLogger("footballcoach.ai.ppo.dagger")


# Fields RolloutBuffer.add() appends to, one entry per row -- kept in sync
# with rollout_buffer.py's own field list by hand (that module has no public
# "give me every per-row field name" accessor). Used below to extend/
# subsample a buffer used purely as an in-memory (obs, bc_label) store (all
# other fields carry dummy placeholder values -- see collect_dagger_rollout)
# without duplicating RolloutBuffer's storage/stacking logic.
_ROLLOUT_BUFFER_FIELDS = [
    "obs", "actions", "log_probs", "values", "rewards", "dones",
    "bc_labels", "head_log_probs", "weights", "reward_comps",
    "step_outcomes", "track_ids",
]


def extend_buffer(dst, src) -> None:
    for f in _ROLLOUT_BUFFER_FIELDS:
        getattr(dst, f).extend(getattr(src, f))


def subsample_buffer(buf, max_size: int, rng: _random.Random) -> None:
    """In-place: if buf has more than max_size rows, keep a uniform random
    subset of exactly max_size rows -- eviction policy for the aggregated
    DAgger buffer once it grows past its cap. A plain uniform re-sample of
    the whole (already fully materialized, in-memory) buffer rather than a
    streaming reservoir sampler -- simpler, and equivalent since this only
    ever runs between DAgger iterations on the buffer's full current
    contents.
    """
    n = len(buf)
    if n <= max_size:
        return
    keep = sorted(rng.sample(range(n), max_size))
    for f in _ROLLOUT_BUFFER_FIELDS:
        vals = getattr(buf, f)
        setattr(buf, f, [vals[i] for i in keep])


def dagger_extra_tensors(buf, device):
    """Pack an aggregated-rows buffer into the (obs_dict, bc_labels) shape
    train_bc_epochs()'s extra_obs/extra_bc_labels params expect, via
    RolloutBuffer.as_tensors() -- the same stacking production PPO training
    uses to turn a rollout into network-ready tensors, reused here rather
    than re-implemented. Returns (None, None) for an empty buffer.
    """
    if len(buf) == 0:
        return None, None
    t = buf.as_tensors([0.0] * len(buf), [0.0] * len(buf))
    extra_obs = {k[len("obs/"):]: v.to(device) for k, v in t.items() if k.startswith("obs/")}
    extra_bc_labels = t["bc_labels"].to(device)
    return extra_obs, extra_bc_labels


def build_phase1_replay_env(trainer, seed: int, *, deterministic: bool = False):
    """Build + reset a phase-1 ScenarioEnv with trainer.decision_net/
    execution_net driving the trainee, seeded/opponent-matched exactly like
    the recorded episode this seed came from. Returns the reset env, ready
    to env.step().

    deterministic=False (default) matches how real PPO rollout collection
    actually drives the trainee -- train()'s own env.sample_action_fn =
    self._sample_action, called with no deterministic override, i.e.
    stochastic sampling from every head's distribution (see _sample_action's
    own default). DAgger's whole point is to aggregate (state, correct
    label) pairs for wherever the CURRENT policy actually goes when
    collecting real training data -- if this rollout instead always takes
    the mode/mean action, it only ever visits the narrow, most-confident
    trajectory, which is NOT the state distribution PPO's own stochastic
    rollout collection (exploration noise and all) produces. Aggregating
    corrections for the wrong distribution would leave exactly the
    states most exposed to exposure bias -- the ones a real, noisy rollout
    actually wanders into -- uncorrected, defeating much of the point of
    running DAgger in the first place. deterministic=True remains available
    for callers that genuinely want an exact, reproducible replay (e.g. a
    one-off debug/comparison script), just no longer the default here.
    """
    import functools

    from footballcoach.ai.curriculum.envs import build_env, bc_label_fn_for_phase_player
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.scripts.replay_episode import apply_phase1_opponent_roll

    trainer.decision_net.eval()
    trainer.execution_net.eval()

    env = build_env(PHASES_BY_ID[1])
    # ScenarioEnv's own built-in mechanism for "policy net plays": set this
    # BEFORE reset(), and reset() auto-assigns NeuralPlayerAI to the trainee
    # (scenario_env.py) -- no manual player.ai / apply_nn_action.py call needed.
    env.sample_action_fn = functools.partial(trainer._sample_action, deterministic=deterministic)
    # (player, match) -> BCLabel, computed INSIDE NeuralPlayerAI.act() at the
    # same instant as the observation -- see that method's own bc_label_fn
    # docstring and phase1_labels_for_player's "CRITICAL -- CALLER-SIDE
    # TIMING" section for why this can't be a separate call after env.step()
    # returns (collect_dagger_rollout() used to do exactly that; fixed
    # 2026-09).
    env.bc_label_fn = bc_label_fn_for_phase_player(1)
    env.reset(seed=seed)

    match = env._loop.match
    opponent = match.player_by_id("opponent")
    apply_phase1_opponent_roll(match, opponent, seed)
    return env


def dagger_episode_candidates(dataset) -> list[tuple[int, int, int]]:
    """[(start, end, seed), ...] over every full-dataset episode that has a
    recorded seed (episode_seed() returns non-None) -- episodes predating
    meta_episode_seeds are skipped.

    Expensive -- dataset.episode_row_ranges(dataset.valid_indices()) walks
    every row of valid_indices() in a plain Python loop (not vectorized) to
    find episode boundaries within that filtered pool, then episode_seed()
    is called once per episode found. For a multi-million-row dataset this
    is a genuinely slow pass (confirmed: several million valid rows -> a
    multi-second-plus pure-Python loop). MUST be called ONCE per DAgger
    phase (see run_dagger_phase), not once per seed needed -- it used to be
    embedded directly in pick_random_dagger_episode() and called once per
    seed (500/iteration in a real run), which meant this entire expensive,
    dataset-invariant pass was redone hundreds of times before the first
    rollout could even start, with no log output anywhere in that loop to
    show why -- confirmed live as several minutes of apparent "hang" per
    DAgger iteration, every iteration, all of it before collect_dagger_
    rollouts_parallel's own "spinning up worker process(es)" log line ever
    had a chance to print.
    """
    ranges = dataset.episode_row_ranges(dataset.valid_indices())
    candidates = []
    for start, end in ranges:
        seed = dataset.episode_seed(end)
        if seed is not None:
            candidates.append((start, end, seed))
    return candidates


def pick_random_dagger_episode(candidates: list[tuple[int, int, int]], rng: _random.Random):
    """Uniformly-random (start, end, seed) from a candidate list already
    built by dagger_episode_candidates() -- cheap, just an rng.choice().
    Returns None if `candidates` is empty (dataset has no seeded episode at
    all)."""
    if not candidates:
        return None
    return rng.choice(candidates)


def train_bc_epochs(
    trainer, dataset, train_idx, opt: torch.optim.Optimizer, *,
    max_epochs: int, batch_size: int, log_every: int = 1,
    extra_obs: dict | None = None, extra_bc_labels: "torch.Tensor | None" = None,
    max_extra_per_minibatch: int | None = None,
    target_loss_margin: float | None = None,
) -> tuple[float, float, int]:
    """BC-train trainer.decision_net/execution_net over `train_idx` (a full
    dataset train split, same shape as Phase 1's own _bc_train_idx) for up
    to max_epochs epochs.

    Mirrors PPOTrainer.pretrain_combined()'s Phase 1 per-minibatch step
    exactly -- same forward pass / label canonicalization / loss function,
    AND the same loss-shaping coefficients (direction_loss_weight,
    region_loss_weight, dec_weight/exec_weight, dec/exec_label_smoothing,
    pos_weight_kick/tackle_attempt) read from the SAME trainer.* attributes
    Phase 1 itself reads them from (set once at PPOTrainer.__init__ from
    ai_config.json, with pos_weight_kick/tackle_attempt auto-computed from
    the dataset at the very start of pretrain_combined() if unset in
    config -- see ppo_trainer.py:1771-1774) -- generalized from
    debug_policy_net.py's _bc_overfit_episode() (which trained on ONE
    episode's rows only) to train over a full dataset split instead, since
    DAgger's own training batch is not restricted to whichever episode a
    given iteration's rollout happens to use (see dagger_iterations'
    ai_config.json comment).

    extra_obs/extra_bc_labels: optional additional (already batched, already
    on `device`) rows -- used by run_dagger_phase() to mix in aggregated
    on-policy (state, rules-AI counterfactual label) pairs alongside the
    regular dataset rows. A bounded RANDOM SUBSET of up to
    max_extra_per_minibatch rows is (re-)sampled from this pool and
    concatenated onto EACH minibatch (a fresh sample every minibatch, with
    replacement across minibatches) -- NOT the whole pool appended to every
    minibatch. That distinction matters a lot once train_idx spans a
    full multi-million-row dataset with hundreds of minibatches per epoch
    and the aggregated buffer has grown large (DAgger's buffer_max_size can
    be in the hundreds of thousands): concatenating the ENTIRE pool onto
    every single minibatch would both massively over-represent aggregated
    rows relative to the dataset (they'd appear once per minibatch instead
    of once per epoch, like everything else) and make each minibatch's
    size scale with the aggregated buffer's size, unboundedly, as DAgger
    iterations progress -- this was a real bug that GPU-OOM'd partway
    through a production run once the aggregated buffer grew past a few
    thousand rows. max_extra_per_minibatch defaults to batch_size (so a
    minibatch is at most doubled). extra_bc_labels must be in the same raw/
    world-frame convention as dataset._labels (uncanonicalized) --
    canonicalize_bc_labels() below is applied AFTER concatenation,
    uniformly to both sources.

    target_loss_margin: None (default) runs a fixed epoch budget with no
    floor-convergence early stop -- appropriate for a full-dataset pass,
    where "converge to the analytic BCE floor" is neither expected nor
    desired the way it is for a small single-episode overfit. Passing a
    float preserves _bc_overfit_episode's original early-stop-on-floor
    behaviour (bc_loss - floor <= target_loss_margin), so this function can
    also serve debug_policy_net.py's own single-episode-overfit use case
    unchanged by passing train_idx=ep_valid and a real margin.

    Returns (final_mean_loss, floor, epochs_run).
    """
    import numpy as np

    from footballcoach.ai.ppo.bc import (
        bc_loss_from_tensor, compute_bc_loss_floor, compute_bc_loss_floor_components,
        direction_magnitude_reg,
    )
    from footballcoach.ai.obs.canonical import canonicalize_bc_labels, x_sign_of
    from footballcoach.ai.ppo.ppo_trainer import (
        EXEC_HEAD_MODULES, _ai_types, _binary_confusion_counts, _precision_recall_f1,
        _measure_exec_head_grad_norms_and_maybe_log_spike,
    )
    from footballcoach.ai.progress import ProgressReporter

    if len(train_idx) == 0:
        raise ValueError("train_idx has zero rows -- nothing to train on.")

    # Reuse the SAME cached values Phase 1 already established (config-set,
    # or auto-computed from the dataset once at the top of
    # pretrain_combined() if unset) -- NOT a fresh dataset.compute_pos_weights()
    # call, which could silently disagree with what Phase 1 actually trained
    # with if the two ever compute slightly different numbers.
    pos_weight_kick = trainer._bc_pos_weight_kick
    pos_weight_tackle_attempt = trainer._bc_pos_weight_tackle_attempt

    if max_extra_per_minibatch is None:
        max_extra_per_minibatch = batch_size
    extra_n = 0 if extra_obs is None else next(iter(extra_obs.values())).shape[0]

    device = trainer.device
    trainer.decision_net.train()
    trainer.execution_net.train()
    floor = None
    mean_loss = float("nan")
    epoch = 0
    all_params = list(trainer.decision_net.parameters()) + list(trainer.execution_net.parameters())
    for epoch in range(1, max_epochs + 1):
        losses = []
        grad_norms: list[float] = []
        # Per-execution-head grad norm + grad-norm-spike diagnostic -- same
        # instrumentation Phase 1's own BC epoch loop gets (ppo_trainer.py),
        # via the shared helper, so DAgger rounds get the same per-head
        # visibility instead of only the single aggregate grad_norm above.
        _head_grad_norm: dict[str, list[float]] = {name: [] for name, _ in EXEC_HEAD_MODULES}
        # Same per-epoch diagnostics Phase 1 logs (ppo_trainer.py's own BC
        # epoch loop) -- direction/kick-direction cosine similarity, mean
        # predicted probability per Bernoulli head, precision/recall/F1 for
        # the rare-positive kick/tackle heads, and the floor-adjusted
        # per-component loss breakdown. Accumulated the identical way, using
        # the SAME shared functions (_binary_confusion_counts,
        # _precision_recall_f1, compute_bc_loss_floor_components) Phase 1
        # uses, so numbers are directly comparable across both phases.
        dir_cosines: list[float] = []
        kick_dir_cosines: list[float] = []
        move_probs: list[float] = []
        sprint_probs: list[float] = []
        kick_probs: list[float] = []
        tackle_attempt_probs: list[float] = []
        _kick_tp = _kick_fp = _kick_fn = 0.0
        _tackle_tp = _tackle_fp = _tackle_fn = 0.0
        _bkdn_acc: dict[str, float] = {}
        _bkdn_floor_acc: dict[str, float] = {}
        _bkdn_n = 0
        _epoch_progress = ProgressReporter(
            len(train_idx), prefix=f"    [dagger bc] epoch {epoch}/{max_epochs}: ",
        )
        _rows_done = 0
        for _mb_i, (obs_dict, bc_labels) in enumerate(dataset.iterate_minibatches(
            batch_size, shuffle=True, device=device, valid_only=True, indices_override=train_idx,
        )):
            _rows_done += bc_labels.shape[0]
            if extra_n > 0:
                # Bounded, freshly-resampled slice of the aggregated buffer
                # per minibatch -- NOT the whole buffer -- see this
                # function's docstring for why (unbounded-per-minibatch
                # blowup as the buffer grows across DAgger iterations).
                k = min(max_extra_per_minibatch, extra_n)
                _extra_idx = torch.randint(0, extra_n, (k,), device=device)
                obs_dict = {k2: torch.cat([v, extra_obs[k2][_extra_idx]], dim=0) for k2, v in obs_dict.items()}
                bc_labels = torch.cat([bc_labels, extra_bc_labels[_extra_idx]], dim=0)
            sat, oat = _ai_types(obs_dict)
            d_heads = trainer.decision_net(
                obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
                obs_dict["ball_feat"], obs_dict["global_feat"], sat, oat,
            )
            e_heads = trainer.execution_net(
                obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
                obs_dict["ball_feat"], obs_dict["global_feat"], d_heads, sat, oat,
            )
            bc_labels = canonicalize_bc_labels(bc_labels, x_sign_of(obs_dict["self_feat"]))
            loss, _bkdn = bc_loss_from_tensor(
                bc_labels, d_heads, e_heads,
                direction_loss_weight=trainer._bc_dir_loss_w,
                direction_loss_mode=trainer._bc_dir_loss_mode,
                region_loss_weight=trainer._bc_region_loss_w,
                pos_weight_kick=pos_weight_kick,
                pos_weight_tackle_attempt=pos_weight_tackle_attempt,
                dec_weight=trainer._bc_dec_weight,
                exec_weight=trainer._bc_exec_weight,
                dec_label_smoothing=trainer._bc_dec_label_smoothing,
                exec_label_smoothing=trainer._bc_exec_label_smoothing,
                return_breakdown=True,
            )
            loss = loss + direction_magnitude_reg(e_heads, trainer._bc_dir_mag_reg_coef)
            if floor is None:
                # Always computed (not just when target_loss_margin is set)
                # so the per-epoch log line below can report the SAME
                # floor-adjusted loss (bc_adj = loss - floor) Phase 1's own
                # logging reports -- comparable numbers across both phases.
                floor = compute_bc_loss_floor(
                    bc_labels,
                    pos_weight_kick=pos_weight_kick,
                    pos_weight_tackle_attempt=pos_weight_tackle_attempt,
                    dec_weight=trainer._bc_dec_weight,
                    exec_weight=trainer._bc_exec_weight,
                    dec_label_smoothing=trainer._bc_dec_label_smoothing,
                    exec_label_smoothing=trainer._bc_exec_label_smoothing,
                )
            for _bk, _bv in _bkdn.items():
                _bkdn_acc[_bk] = _bkdn_acc.get(_bk, 0.0) + _bv
            _bkdn_floor = compute_bc_loss_floor_components(
                bc_labels,
                pos_weight_kick=pos_weight_kick,
                pos_weight_tackle_attempt=pos_weight_tackle_attempt,
                dec_label_smoothing=trainer._bc_dec_label_smoothing,
                exec_label_smoothing=trainer._bc_exec_label_smoothing,
                has_exec=True,
            )
            for _bk, _bv in _bkdn_floor.items():
                _bkdn_floor_acc[_bk] = _bkdn_floor_acc.get(_bk, 0.0) + _bv
            _bkdn_n += 1

            # Same dir_cos/kick_dir_cos/head-prob/precision-recall accumulation
            # Phase 1's own BC epoch loop does (ppo_trainer.py) -- identical
            # column indices (_I_VALID=14, move_direction=7:9,
            # kick_this_tick=12, kick_direction=18/19/24, tackle_attempt=13),
            # kept in sync by hand with that loop since bc_labels' column
            # layout isn't otherwise exposed as named constants at this
            # tensor-level (only bc.py's _I_* module constants, used at the
            # BCLabel-construction layer, not here).
            with torch.no_grad():
                valid_mask = bc_labels[:, 14] > 0.5  # _I_VALID
                has_dir = (bc_labels[:, 7].abs() + bc_labels[:, 8].abs()) > 1e-6
                sel = valid_mask & has_dir
                if sel.any():
                    pred_dir = e_heads.move_direction[sel]
                    tgt_dir = bc_labels[sel, 7:9]
                    eps = 1e-6
                    pred_n = pred_dir / (pred_dir.norm(dim=-1, keepdim=True) + eps)
                    cos_vals = (pred_n * tgt_dir).sum(dim=-1)
                    dir_cosines.append(cos_vals.mean().item())
                kicked_mask = valid_mask & (bc_labels[:, 12] > 0.5)  # _I_KICK_THIS_TICK
                has_kick_dir = (bc_labels[:, 18].abs() + bc_labels[:, 19].abs() + bc_labels[:, 24].abs()) > 1e-6
                ksel = kicked_mask & has_kick_dir
                if ksel.any():
                    pred_kdir = e_heads.kick_direction[ksel]
                    tgt_kdir = torch.stack([bc_labels[ksel, 18], bc_labels[ksel, 19], bc_labels[ksel, 24]], dim=-1)
                    eps = 1e-6
                    pred_kn = pred_kdir / (pred_kdir.norm(dim=-1, keepdim=True) + eps)
                    kcos_vals = (pred_kn * tgt_kdir).sum(dim=-1)
                    kick_dir_cosines.append(kcos_vals.mean().item())
                if valid_mask.any():
                    move_probs.append(torch.sigmoid(e_heads.exec_move_logit.squeeze(-1)[valid_mask]).mean().item())
                    sprint_probs.append(torch.sigmoid(e_heads.sprint_logit.squeeze(-1)[valid_mask]).mean().item())
                    kick_probs.append(torch.sigmoid(e_heads.kick_logit.squeeze(-1)[valid_mask]).mean().item())
                    tackle_attempt_probs.append(torch.sigmoid(e_heads.tackle_attempt_logit.squeeze(-1)[valid_mask]).mean().item())
                    _tp, _fp, _fn = _binary_confusion_counts(e_heads.kick_logit, bc_labels[:, 12], valid_mask)
                    _kick_tp += _tp; _kick_fp += _fp; _kick_fn += _fn
                    _tp, _fp, _fn = _binary_confusion_counts(e_heads.tackle_attempt_logit, bc_labels[:, 13], valid_mask)
                    _tackle_tp += _tp; _tackle_fp += _fp; _tackle_fn += _fn

            opt.zero_grad()
            loss.backward()
            _gn_vals = _measure_exec_head_grad_norms_and_maybe_log_spike(trainer, "DAgger", epoch, _mb_i)
            for _head_name, _ in EXEC_HEAD_MODULES:
                if _head_name in _gn_vals:
                    _head_grad_norm[_head_name].append(_gn_vals[_head_name])
            clip_limit = trainer._bc_max_grad_norm if trainer._bc_max_grad_norm is not None else float("inf")
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, clip_limit).item()
            opt.step()
            losses.append(loss.item())
            grad_norms.append(grad_norm)
            _epoch_progress.update(_rows_done, postfix=f"loss={loss.item():.4f}")
        _epoch_progress.update(_epoch_progress.total, postfix=f"loss={losses[-1]:.4f}")
        mean_loss = float(sum(losses) / len(losses))
        if epoch == 1 or epoch % log_every == 0 or epoch == max_epochs:
            mean_grad_norm = float(sum(grad_norms) / len(grad_norms))
            min_grad_norm = float(min(grad_norms))
            max_grad_norm = float(max(grad_norms))
            std_grad_norm = float(np.std(grad_norms))
            mean_cos = float(np.mean(dir_cosines)) if dir_cosines else float("nan")
            mean_kick_cos = float(np.mean(kick_dir_cosines)) if kick_dir_cosines else float("nan")
            mean_mv = float(np.mean(move_probs)) if move_probs else float("nan")
            mean_spr = float(np.mean(sprint_probs)) if sprint_probs else float("nan")
            mean_kk = float(np.mean(kick_probs)) if kick_probs else float("nan")
            mean_tk = float(np.mean(tackle_attempt_probs)) if tackle_attempt_probs else float("nan")
            kk_prec, kk_rec, kk_f1 = _precision_recall_f1(_kick_tp, _kick_fp, _kick_fn)
            tk_prec, tk_rec, tk_f1 = _precision_recall_f1(_tackle_tp, _tackle_fp, _tackle_fn)
            _kick_bkdn_keys = {"kick", "kick_direction", "kick_power", "kick_spin"}
            # Subtract each component's own analytic label-smoothing floor
            # before printing -- same rationale as Phase 1's identical block
            # (ppo_trainer.py): raw-magnitude differences between components
            # otherwise masquerade as real imitation-quality differences.
            bkdn_str = "  ".join(
                f"{k}={max(0.0, v / _bkdn_n - _bkdn_floor_acc.get(k, 0.0) / _bkdn_n):.5f}"
                if k in _kick_bkdn_keys else
                f"{k}={max(0.0, v / _bkdn_n - _bkdn_floor_acc.get(k, 0.0) / _bkdn_n):.3f}"
                for k, v in _bkdn_acc.items()
            ) if _bkdn_n else ""
            _lines = [
                f"    [dagger bc] epoch {epoch}/{max_epochs}",
                f"      loss       bc={mean_loss:.4f}  bc_adj={mean_loss - floor:.4f}"
                f"(floor={floor:.4f})  grad_norm={mean_grad_norm:.3f}",
                f"      heads      dir_cos={mean_cos:.3f}  kick_dir_cos={mean_kick_cos:.3f}",
                f"                 move_prob={mean_mv:.3f}  sprint_prob={mean_spr:.3f}  "
                f"kick_prob={mean_kk:.3f}  tackle_prob={mean_tk:.3f}",
                f"      pr/rec     kick:   p={kk_prec:.3f}  r={kk_rec:.3f}  f1={kk_f1:.3f}  "
                f"(tp={_kick_tp:.0f} fp={_kick_fp:.0f} fn={_kick_fn:.0f})",
                f"                 tackle: p={tk_prec:.3f}  r={tk_rec:.3f}  f1={tk_f1:.3f}  "
                f"(tp={_tackle_tp:.0f} fp={_tackle_fp:.0f} fn={_tackle_fn:.0f})",
            ]
            if bkdn_str:
                _bkdn_parts = bkdn_str.split("  ")
                _mid = (len(_bkdn_parts) + 1) // 2
                _lines.append(f"      breakdown (floor-adj)  {'  '.join(_bkdn_parts[:_mid])}")
                if _bkdn_parts[_mid:]:
                    _lines.append(f"                             {'  '.join(_bkdn_parts[_mid:])}")
            _lines.append(
                f"      [dagger grad norm total] mean={mean_grad_norm:.3f}  min={min_grad_norm:.3f}  "
                f"max={max_grad_norm:.3f}  std={std_grad_norm:.3f}  (n={len(grad_norms)})"
            )
            if any(_head_grad_norm.values()):
                _head_gn_str = "  ".join(
                    f"{name}={np.mean(vals):.3f}" for name, vals in _head_grad_norm.items() if vals
                )
                _lines.append(f"      [exec head grad norm] {_head_gn_str}")
            log.info("\n".join(_lines))
            for _v in _head_grad_norm.values():
                _v.clear()
        if target_loss_margin is not None and floor is not None and (mean_loss - floor) <= target_loss_margin:
            log.info(f"    [dagger bc] converged at epoch {epoch}: loss={mean_loss:.4f}  floor={floor:.4f}")
            return mean_loss, floor, epoch
    return mean_loss, (floor if floor is not None else 0.0), epoch


def collect_dagger_rollout(trainer, seed: int, max_steps: int):
    """Roll the CURRENT policy out closed-loop from `seed`, with real PPO-
    style stochastic action sampling (see build_phase1_replay_env's own
    deterministic=False default/rationale) and NO divergence stopping --
    the whole point of DAgger is to visit wherever the policy actually
    goes (including off the original recorded trajectory, and including
    wherever real exploration noise takes it) and get those states
    correctly labelled.

    Every visited state is labelled via phase1_labels_for_player(), called
    from INSIDE NeuralPlayerAI.act() at the same instant the observation is
    encoded (see build_phase1_replay_env's env.bc_label_fn wiring, and
    NeuralPlayerAI.bc_label_fn's own docstring) -- the SAME mechanism
    production's on-policy BC-aux-loss training uses (rollout_worker.py's
    env.bc_label_fn wiring, phase_id==1 case). Every execution-level field
    of that label (move_direction/sprint/exec_move AND kick_this_tick/
    kick_direction/kick_power_fraction/kick_spin/tackle_attempt) is a
    genuine Phase1RulesAI counterfactual -- "what would the rules AI do
    from this exact state" -- entirely independent of whatever the
    trainee's own NeuralPlayerAI actually did. See
    phase1_labels_for_player()'s docstring (bc.py) for the full mechanism
    and the two distinct historical bugs it fixes (kick/tackle echoing the
    acting AI instead of a genuine counterfactual; the counterfactual being
    evaluated one physics tick later than the observation it's paired
    with) -- collect_dagger_rollout() used to have both, until 2026-09.

    Returns (buffer, summary) -- buffer is a RolloutBuffer holding one row
    per decision step (obs + bc_label, all other fields dummy placeholders);
    summary is {"outcome": str|None, "n_steps": int, "total_reward": float,
    "n_kicks": int, "n_kicks_labelled": int}.

    n_kicks counts real, physics-executed kicks only (the trainee's own
    Player.kicked_this_tick flag -- set exactly the tick kick_direct()
    actually fires, per entities/player.py's own docstring). Read from
    StepInfo.trainee_kicks_this_step, which env.step() itself accumulates by
    scanning EVERY physics tick of its internal decision interval -- env.
    step() is not one physics tick, it's self._ticks_per_decision of them
    (~4-15 depending on config), and kicked_this_tick resets every physics
    tick (Match._process_orders), so a single post-hoc read of the flag
    after env.step() returns only ever sees whichever tick happened to be
    last, silently undercounting real kicks landing on any earlier tick in
    the interval (confirmed live 2026-09: real kicks missed entirely on
    seeds where the kick landed mid-interval). Fixed 2026-09 -- this used to
    read env._loop.match.player_by_id("trainee").kicked_this_tick directly
    here, once per env.step() call. This is what the trainee's OWN policy
    actually did.

    n_kicks_labelled counts decision steps where the collected BC label's
    own kick_this_tick was >= 0.5 -- a genuine, independent "would
    Phase1RulesAI have kicked from this exact state" count (see above), NOT
    an echo of n_kicks. The two counts diverging is expected and healthy --
    it's exactly the signal DAgger is meant to aggregate and correct
    ("the rules AI would have kicked here; the policy didn't" or vice
    versa). If you see n_kicks_labelled == n_kicks tick-for-tick across a
    whole rollout, that's a red flag worth checking (either the policy has
    converged very closely to the rules AI's own kick timing, or something
    upstream broke and it's silently echoing again).
    """
    from footballcoach.ai.ppo.bc import _I_KICK_THIS_TICK
    from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer

    env = build_phase1_replay_env(trainer, seed)
    buf = RolloutBuffer()
    total_reward = 0.0
    outcome = None
    n_steps = 0
    n_kicks = 0
    n_kicks_labelled = 0
    with torch.no_grad():
        for n_steps in range(1, max_steps + 1):
            _, reward, done, info = env.step()
            n_kicks += info.trainee_kicks_this_step
            tr = env.last_trainee_transition
            if tr is not None:
                total_reward += reward
                # Computed INSIDE NeuralPlayerAI.act() (see
                # build_phase1_replay_env's env.bc_label_fn wiring), at the
                # same instant as tr["obs"] -- already a numpy array.
                bc_label_arr = tr.get("bc_label")
                if bc_label_arr is not None and bc_label_arr[_I_KICK_THIS_TICK] >= 0.5:
                    n_kicks_labelled += 1
                buf.add(
                    obs=tr["obs"], action={}, log_prob=0.0, value=0.0,
                    reward=0.0, done=0.0, bc_label=bc_label_arr,
                )
            if done:
                outcome = info.trial_outcome
                break
    return buf, {
        "outcome": outcome, "n_steps": n_steps, "total_reward": total_reward,
        "n_kicks": n_kicks, "n_kicks_labelled": n_kicks_labelled,
    }


# Module-level, set once per worker process by _dagger_pool_worker_init --
# live nn.Module objects aren't picklable across a process boundary, so each
# worker builds its OWN inference-only PPOTrainer exactly once (at Pool
# startup, via the initializer= mechanism below) and every task that worker
# subsequently runs reuses it, rather than rebuilding per seed.
_worker_trainer = None


def _dagger_pool_worker_init(
    decision_state: dict, execution_state: dict, separate_value_net: bool, value_state: dict | None,
) -> None:
    """``Pool(initializer=...)`` target -- runs once when a worker process
    starts, before it's given any task. Mirrors ai/eval/seeded_eval.py's
    per-worker rebuild-from-state-dict pattern, but as a persistent
    initializer rather than something repeated inside every task, so
    ``collect_dagger_rollouts_parallel`` can dispatch one task PER SEED
    (via ``imap_unordered``) instead of pre-chunking seeds across workers --
    that's what lets the driver see (and progress-bar) each seed's
    completion as it happens instead of waiting for an entire chunk."""
    import torch as _torch
    # Each worker only ever runs ONE env forward-pass at a time (no internal
    # batching), so letting torch/BLAS use its default multi-threaded pool
    # means N worker processes each spin up a full thread pool and fight
    # over the same cores -- mirrors rollout_worker.py's/seeded_eval.py's
    # worker_torch_threads=1 default, same rationale.
    _torch.set_num_threads(1)
    global _worker_trainer
    from footballcoach.ai.ppo.ppo_trainer import rebuild_inference_trainer

    _worker_trainer = rebuild_inference_trainer(decision_state, execution_state, separate_value_net, value_state)


def _dagger_pool_worker_task(seed: int, max_replay_steps: int):
    """Pool task body -- one seed, using the trainer ``_dagger_pool_worker_
    init`` already built for this worker process. Returns
    ``(seed, (buffer, summary))``, same per-seed result shape
    ``collect_dagger_rollouts_parallel`` already returns for its serial
    path."""
    return seed, collect_dagger_rollout(_worker_trainer, seed, max_replay_steps)


def collect_dagger_rollouts_parallel(
    trainer, seeds: list[int], max_replay_steps: int, n_workers: int = 1,
) -> list:
    """Parallel variant of collect_dagger_rollout() -- rolls each of `seeds`
    out independently (its own starting spawn/opponent-type roll), spread
    across up to n_workers subprocesses. n_workers<=1 runs sequentially in
    the caller's (this) process with no subprocess overhead -- same
    fallback convention ai/eval/seeded_eval.py's run_seeded_evaluation_parallel()
    uses, so a caller doesn't need its own separate serial/parallel branch.

    Reports ONE overall progress bar over episode COUNT (not step count, and
    not one bar per episode/worker) -- with n_workers>1 dispatching one task
    per seed via imap_unordered (rather than pre-chunked per-worker batches,
    see _dagger_pool_worker_init's docstring), the driver sees each seed's
    completion as it actually happens and can update this bar live, instead
    of every worker's own per-episode bar (there wasn't a shared bar before)
    all landing on stdout in one unreadable burst whenever a chunk finished.
    Not step-accurate (a rollout can end anywhere from a few dozen to
    max_replay_steps ticks) -- doesn't need to be, this is a "how far into
    the episode batch are we" indicator, not an ETA guarantee.

    Returns a list of (seed, (buffer, summary)) tuples covering every seed
    in `seeds`, in arbitrary order (completion order, not necessarily
    matching `seeds`' own order) -- callers that need per-seed correspondence
    should keep their own seed -> result mapping via the returned seed value.
    """
    from footballcoach.ai.progress import ProgressReporter

    progress = ProgressReporter(len(seeds), prefix="  [dagger rollout] ")
    results: list = []
    if n_workers <= 1:
        for seed in seeds:
            results.append((seed, collect_dagger_rollout(trainer, seed, max_replay_steps)))
            progress.update(len(results))
        progress.finish(len(results), n_episodes=len(results))
        return results

    import functools
    import multiprocessing as mp

    decision_state, execution_state, value_state = trainer._cpu_state_dicts()
    separate_value_net = trainer.separate_value_net
    ctx = mp.get_context("spawn")
    n_procs = min(n_workers, len(seeds))
    # Nothing prints between here and this function's first progress.update()
    # call (that only fires once a worker's FIRST rollout actually completes)
    # -- and everything in between (spawning n_procs fresh interpreter
    # processes, each re-importing torch/numpy/footballcoach, then rebuilding
    # a full inference trainer including a fresh disk read of the frozen
    # physics-pretrain encoder checkpoints, see _dagger_pool_worker_init) can
    # legitimately take a while with nothing else to show for it in the
    # meantime. This line exists purely so that gap has a visible start time
    # instead of looking like a hang.
    log.info(f"  [dagger rollout] spinning up {n_procs} worker process(es)...")
    with ctx.Pool(
        processes=n_procs, initializer=_dagger_pool_worker_init,
        initargs=(decision_state, execution_state, separate_value_net, value_state),
    ) as pool:
        task = functools.partial(_dagger_pool_worker_task, max_replay_steps=max_replay_steps)
        for result in pool.imap_unordered(task, seeds):
            results.append(result)
            progress.update(len(results))
    progress.finish(len(results), n_episodes=len(results))
    return results


def run_dagger_phase(
    trainer, dataset, train_idx, opt: torch.optim.Optimizer, *,
    iterations: int, bc_epochs_per_iter: int, buffer_max_size: int,
    max_replay_steps: int, batch_size: int, episodes_per_iteration: int = 1,
    n_workers: int = 1, log_every: int = 1,
    checkpoint_fn: Callable[[str], None] | None = None,
) -> None:
    """DAgger outer loop: each iteration picks `episodes_per_iteration` NEW
    random episodes (used only to seed that many rollouts, each reproducing
    its own episode's exact recorded opponent-type roll -- NOT to restrict
    training rows), rolls the CURRENT policy out from each picked episode's
    seed and aggregates all the newly-visited states, THEN trains on the
    regular full-dataset split (`train_idx`) with the now-updated aggregated
    on-policy buffer mixed in. Rollout-then-train (not the other way
    around) so every iteration's training step -- including the first --
    actually sees on-policy correction data, and no rollout's data ever
    goes uncollected-into-training at the very end of the phase.

    Directly addresses the exposure-bias/covariate-shift gap plain BC has --
    BC never sees "recover from a slightly-off state" examples, so small
    per-step errors compound in closed-loop rollout even at low training
    loss. Aggregating the policy's OWN visited states (correctly labelled by
    Phase1RulesAI) directly teaches it to recover from exactly the kind of
    drift it produces. episodes_per_iteration > 1 collects that many
    independent rollouts (different seeds/opponent-type rolls) per round
    instead of just one, before the buffer is next trained on -- more
    diverse on-policy coverage per round, at roughly proportional extra
    rollout cost per iteration. n_workers > 1 collects that round's rollouts
    in parallel (see collect_dagger_rollouts_parallel) -- rollout collection
    is by far the slowest part of each iteration (real env physics ticks,
    not GPU-batched like the BC training step), so this is usually where
    parallelism matters most.

    checkpoint_fn: optional callback(phase_label: str), called after each
    iteration's training step (i.e. once weights have actually moved this
    round). None (default) = no checkpointing here -- the caller
    (pretrain_combined()) already saves once before DAgger starts (end of
    Phase 1) and once after the whole phase ends, so this is opt-in extra
    granularity for a phase that can run for a long time, so a crash/kill
    partway through doesn't lose every iteration's progress since the last
    save. Pass PPOTrainer.pretrain_combined()'s own `_save_pretrain_checkpoint`
    closure -- same one every other phase in that function already uses.
    """
    import time

    from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
    from footballcoach.ai.progress import ProgressReporter

    rng = trainer._aug_rng
    aggregated = RolloutBuffer()

    # Computed ONCE for the whole phase, not once per seed needed -- see
    # dagger_episode_candidates()'s own docstring for why this used to be
    # embedded in the per-seed picker instead (a genuinely slow, dataset-
    # size-scaling pass, redone hundreds of times per iteration for no
    # reason, entirely before any rollout or worker-pool log line could
    # print -- confirmed live as several minutes of apparently-hung silence
    # before the very first DAgger rollout of a run).
    log.info("[dagger] building episode candidate list from dataset (once for this phase)...")
    _t0 = time.monotonic()
    candidates = dagger_episode_candidates(dataset)
    log.info(f"[dagger] found {len(candidates)} seeded episode(s) in {time.monotonic() - _t0:.1f}s")
    if not candidates:
        log.warning("[dagger] no seeded episode available in dataset -- DAgger phase skipped entirely.")
        return

    # Coarse, iteration-granularity bar for overall DAgger-phase ETA (each
    # iteration itself contains its own finer BC-epoch/rollout-tick bars
    # below) -- live=False (milestone lines only) since this phase can run
    # for a long time and per-iteration granularity is too coarse for a
    # meaningfully-updating \r bar; the finer bars inside each iteration
    # already give live feedback.
    _iter_progress = ProgressReporter(iterations, prefix="[dagger phase] ", live=False)
    for it in range(1, iterations + 1):
        # Rollout-and-aggregate BEFORE training, not after: with the
        # opposite order, iteration 1's training step would run against an
        # empty aggregated buffer (nothing collected yet) -- DAgger
        # wouldn't actually influence training until iteration 2 -- and the
        # LAST iteration's rollout would be collected but never trained on
        # (the phase ends right after it). Collecting first means every
        # iteration's training step, including the first, sees on-policy
        # correction data from the CURRENT policy (Phase 1's fresh weights,
        # for iteration 1), and no rollout ever goes to waste.
        _iter_t0 = time.monotonic()
        log.info(f"[dagger] iteration {it}/{iterations}: starting")
        seeds = [pick_random_dagger_episode(candidates, rng)[2] for _ in range(episodes_per_iteration)]

        log.info(f"[dagger] iteration {it}/{iterations}: picked {len(seeds)} episode seed(s), starting rollout collection...")
        _rollout_t0 = time.monotonic()
        results = collect_dagger_rollouts_parallel(trainer, seeds, max_replay_steps, n_workers=n_workers)
        _rollout_elapsed = time.monotonic() - _rollout_t0
        for _seed, (new_buf, _summary) in results:
            extend_buffer(aggregated, new_buf)
        _pre_subsample_n = len(aggregated)
        subsample_buffer(aggregated, buffer_max_size, rng)
        if len(aggregated) < _pre_subsample_n:
            log.info(
                f"[dagger] iteration {it}/{iterations}: aggregated buffer exceeded cap "
                f"({_pre_subsample_n} > {buffer_max_size}), subsampled down to {len(aggregated)}"
            )

        import numpy as np

        from footballcoach.ai.ppo.ppo_trainer import _PHASE1_OUTCOME_LEGEND, outcome_breakdown

        summaries = [summary for _seed, (_buf, summary) in results]
        rewards = [s["total_reward"] for s in summaries]
        # kicks: <real NN kicks>/<BC-labelled kick_this_tick decision ticks>
        # summed across this round's rollouts. The second number is an
        # independent rules-AI counterfactual ("would Phase1RulesAI have
        # kicked from this exact state"), not an echo of the first -- see
        # collect_dagger_rollout()'s own docstring for the full rationale.
        total_kicks = sum(s["n_kicks"] for s in summaries)
        total_kicks_labelled = sum(s["n_kicks_labelled"] for s in summaries)
        log.info(
            f"[dagger] iteration {it}/{iterations}: {len(summaries)} rollout(s) in {_rollout_elapsed:.1f}s -- "
            f"outcomes[{_PHASE1_OUTCOME_LEGEND}]={outcome_breakdown([s['outcome'] for s in summaries])}  "
            f"reward mean={float(np.mean(rewards)):.2f} std={float(np.std(rewards)):.2f}  "
            f"kicks={total_kicks}/{total_kicks_labelled}  aggregated_buffer_size={len(aggregated)}"
        )

        extra_obs, extra_bc_labels = dagger_extra_tensors(aggregated, trainer.device)
        log.info(f"[dagger] iteration {it}/{iterations}: BC training starting ({len(aggregated)} aggregated rows so far)...")
        _bc_t0 = time.monotonic()
        train_bc_epochs(
            trainer, dataset, train_idx, opt,
            max_epochs=bc_epochs_per_iter, batch_size=batch_size, log_every=log_every,
            extra_obs=extra_obs, extra_bc_labels=extra_bc_labels,
        )
        _bc_elapsed = time.monotonic() - _bc_t0
        log.info(f"[dagger] iteration {it}/{iterations}: BC training took {_bc_elapsed:.1f}s")

        if checkpoint_fn is not None:
            _ckpt_t0 = time.monotonic()
            checkpoint_fn(f"DAgger iter {it}/{iterations}")
            log.info(f"[dagger] iteration {it}/{iterations}: checkpoint saved in {time.monotonic() - _ckpt_t0:.1f}s")
        log.info(
            f"[dagger] iteration {it}/{iterations}: done in {time.monotonic() - _iter_t0:.1f}s total "
            f"(rollout {_rollout_elapsed:.1f}s, BC training {_bc_elapsed:.1f}s)"
        )
        _iter_progress.update(it)
    _iter_progress.finish(iterations)

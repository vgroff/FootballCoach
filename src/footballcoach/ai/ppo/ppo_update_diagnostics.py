"""Logging-only diagnostics for PPOTrainer's PPO update (see ppo_update.py).

Nothing here changes training: every method reads the update's context and
accumulated statistics and writes log lines. Kept apart from ppo_update.py so
the update's own logic reads top to bottom without the reporting in between.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from footballcoach.ai.action.distributions import IndependentBernoulli, _von_mises_kl
from footballcoach.ai.ppo.rollout_buffer import HEAD_LP_KEYS
from footballcoach.ai.ppo.trainer_common import _ai_types, format_outcome_rmse_breakdown, log

if TYPE_CHECKING:
    from footballcoach.ai.ppo.ppo_update import _UpdateContext, _UpdateStats

# The detailed ratio/per-head block below is logged when the update's mean KL
# exceeds this (about 5x a typical target_kl).
KL_DIAG_THRESHOLD = 0.05
# torch.quantile() refuses inputs past 16,777,216 elements; the ratio
# percentiles are taken from a random subsample of at most this many.
_MAX_QUANTILE_ELEMENTS = 8_000_000
_PCTS = [0, 1, 10, 50, 90, 99, 100]


def _percentile_str(t: torch.Tensor, digits: int = 4) -> str:
    """'p0=... p1=... ... p100=...' over _PCTS, each to `digits` decimals."""
    return "  ".join(f"p{p}={float(t.quantile(p / 100.0)):.{digits}f}" for p in _PCTS)


def _angular_dmean_deg(dmean: float) -> float:
    """L2 shift of a unit-vector mean -> approximate angular shift in degrees
    (for unit vectors |u - v| = d means the angle is arccos(1 - d^2 / 2))."""
    cos_theta = max(-1.0, min(1.0, 1.0 - dmean ** 2 / 2.0))
    return math.degrees(math.acos(cos_theta))


def _angular_dlog_std_deg(dlog_kappa: float, ls_end: float) -> float:
    """Change in log_kappa -> change in angular std, in degrees (large-kappa von
    Mises approximation: variance ~= 1 / kappa)."""
    return abs(
        math.degrees(math.sqrt(1.0 / math.exp(ls_end)))
        - math.degrees(math.sqrt(1.0 / math.exp(ls_end - dlog_kappa)))
    )


def _episode_abs_adv_means(advantages: torch.Tensor, dones: torch.Tensor) -> Optional[np.ndarray]:
    """mean(|advantage|) of each complete episode (an episode is
    [previous done + 1, this done]; a trailing partial episode is dropped).
    None when the batch has no episode end."""
    raw_adv = advantages.detach().cpu().numpy()
    dones_np = dones.detach().cpu().numpy()
    ep_ends = np.where(dones_np > 0.5)[0]
    if len(ep_ends) == 0:
        return None
    ep_starts = np.concatenate([[0], ep_ends[:-1] + 1])
    return np.array([np.abs(raw_adv[s:e + 1]).mean() for s, e in zip(ep_starts, ep_ends)])


def _quantile_sample(ratios: torch.Tensor) -> torch.Tensor:
    """`ratios` itself, or an unbiased random subsample of _MAX_QUANTILE_ELEMENTS
    when it is too large for torch.quantile (consumes the torch RNG only then)."""
    if ratios.numel() > _MAX_QUANTILE_ELEMENTS:
        idx = torch.randint(0, ratios.numel(), (_MAX_QUANTILE_ELEMENTS,), device=ratios.device)
        return ratios.flatten()[idx]
    return ratios


def _trimmed_mean(x: torch.Tensor, lo_q: float, hi_q: float) -> float:
    """Mean of ``x`` after dropping values outside ``[quantile(lo_q),
    quantile(hi_q)]`` -- a robust companion to a plain mean, so a handful of
    extreme outliers (e.g. a rare mis-valued state producing a huge
    squared-error term, or a ratio-spike row) don't single-handedly
    dominate the reported number. Returns ``nan`` for an empty input.
    Shared core for ``_trimmed_mean_p10_p90``/``_trimmed_mean_p25_p75``
    below -- see those for the specific cuts actually logged."""
    if x.numel() == 0:
        return float("nan")
    lo = x.quantile(lo_q)
    hi = x.quantile(hi_q)
    mask = (x >= lo) & (x <= hi)
    trimmed = x[mask]
    return float(trimmed.mean()) if trimmed.numel() > 0 else float(x.mean())

def _trimmed_mean_p10_p90(x: torch.Tensor) -> float:
    """Mean of ``x`` after dropping the bottom/top 10% (i.e. keeping only
    values in [p10, p90]) -- same motivation as the policy_loss percentile
    lines elsewhere in this file, just collapsed to one robust scalar
    instead of a full percentile spread."""
    return _trimmed_mean(x, 0.10, 0.90)

def _trimmed_mean_p25_p75(x: torch.Tensor) -> float:
    """Mean of ``x`` after dropping the bottom AND top 25% (i.e. the
    interquartile mean, keeping only values in [p25, p75]) -- a much more
    aggressive, but symmetric/centered, cut than p10_p90's. Logged alongside
    p10_p90 (not instead of it) as a second, more heavily-trimmed data
    point: comparing the two shows whether a metric's mean is dominated by
    its tails generally (p25_p75 reads much smaller/flatter than p10_p90)
    or is fairly stable regardless of how aggressively both tails are cut
    (the two read close together)."""
    return _trimmed_mean(x, 0.25, 0.75)


class PPOUpdateDiagnosticsMixin:
    """Log lines for PPOUpdateMixin._ppo_update. Mixed into PPOTrainer through
    PPOUpdateMixin; relies on its attributes."""

    def _active_heads_str(self, values, fmt: str = "+.4f") -> str:
        """'key=value' for each HEAD_LP_KEYS entry, skipping masked/frozen/inactive
        heads (their ratio is forced to 1, so their numbers carry no signal)."""
        inactive = self._inactive_head_lp_keys()
        return "  ".join(f"{k}={v:{fmt}}" for k, v in zip(HEAD_LP_KEYS, values) if k not in inactive)

    def _ppo_log_rollout_debug(self, ctx: "_UpdateContext", progress: float) -> None:
        """DEBUG-level summary of the rollout the update is about to train on."""
        batch, n = ctx.batch, ctx.n
        old_log_probs, returns = ctx.old_log_probs, ctx.returns
        raw_adv = batch["advantages"]
        raw_vals = batch["values"]
        raw_rews = batch["rewards"]
        log.debug(
            f"[PPO UPDATE] step={self._total_steps:,}  n={n}  progress={progress:.3f}\n"
            f"  old_log_prob: mean={old_log_probs.mean():.3f}  std={old_log_probs.std():.3f}"
            f"  min={old_log_probs.min():.3f}  max={old_log_probs.max():.3f}\n"
            f"  returns:      mean={returns.mean():.3f}  std={returns.std():.3f}"
            f"  min={returns.min():.3f}  max={returns.max():.3f}\n"
            f"  advantages:   mean={raw_adv.mean():.3f}  std={raw_adv.std():.3f}"
            f"  min={raw_adv.min():.3f}  max={raw_adv.max():.3f}\n"
            f"  values(old):  mean={raw_vals.mean():.3f}  std={raw_vals.std():.3f}"
            f"  min={raw_vals.min():.3f}  max={raw_vals.max():.3f}\n"
            f"  rewards:      mean={raw_rews.mean():.4f}  std={raw_rews.std():.4f}"
            f"  nonzero={int((raw_rews != 0).sum())}/{n}"
        )

    def _ppo_log_baseline_block(self, tag: str, n_rows_label: str, eval_result: tuple) -> None:
        """One population's epoch-0 lines, from an _eval_val_episode_losses() result."""
        (
            _b_pol, _b_val, _b_kl, _b_per_sample_pol,
            _b_val_p90, _b_val_p75, _b_pol_p90, _b_pol_p75,
            _b_surr_mean, _b_surr_p90, _b_surr_p75, _b_head_pol,
        ) = eval_result
        log.info(
            f"  [{tag}policy] mean={_b_pol:.4f}  p10_p90={_b_pol_p90:.4f}  "
            f"p25_p75={_b_pol_p75:.4f}  kl={_b_kl:.4f}  ({n_rows_label})"
        )
        log.info(
            f"  [{tag}policy surrogate] mean={_b_surr_mean:.4f}  "
            f"p10_p90={_b_surr_p90:.4f}  p25_p75={_b_surr_p75:.4f}  "
            f"(n={len(_b_per_sample_pol):,})"
        )
        if len(_b_per_sample_pol) > 0:
            log.info(f"  [{tag}policy percentiles, n={len(_b_per_sample_pol):,}]  {_percentile_str(_b_per_sample_pol)}")
        if _b_head_pol is not None:
            log.info(f"  [{tag}policy heads] {self._active_heads_str(_b_head_pol.tolist())}")
        log.info(
            f"  [{tag}value] mean={_b_val:.4f}(x{self.vf_coef})={self.vf_coef * _b_val:.4f}  "
            f"p10_p90={_b_val_p90:.4f}(x{self.vf_coef})={self.vf_coef * _b_val_p90:.4f}  "
            f"p25_p75={_b_val_p75:.4f}(x{self.vf_coef})={self.vf_coef * _b_val_p75:.4f}"
        )

    def _ppo_log_epoch_zero_baseline(self, ctx: "_UpdateContext") -> None:
        """Pre-training baseline for the train rows (and the held-out rows, if any).

        Reuses _eval_val_episode_losses() for both populations, so the formulas
        are exactly those of the per-epoch lines. Like the real held-out lines,
        it does not pass the cached physics-encoder features through (irrelevant
        when the encoders are off). old_log_prob == new_log_prob exactly here, so
        policy loss, KL and surrogate are ~0 by construction; the value and
        per-head numbers are the useful before/after comparison."""
        log.info(f"  == epoch 0 (pre-training baseline, ratio=1 exactly) " + "=" * 32)
        self._ppo_log_baseline_block("", f"n={ctx.n} rows, pre-training", self._eval_val_episode_losses(ctx.batch, ctx.clip))
        if ctx.val_batch is not None:
            self._ppo_log_baseline_block(
                "held-out ", f"n={len(ctx.val_batch['log_probs'])} held-out steps, pre-training",
                self._eval_val_episode_losses(ctx.val_batch, ctx.clip),
            )
        log.info("  [grad clip] N/A -- no gradient step taken yet")

    def _ppo_log_first_minibatch_debug(
        self, d_heads, e_heads, mb_actions: dict, new_log_probs: torch.Tensor,
        mb_old_lp: torch.Tensor, new_values: torch.Tensor, mb_ret: torch.Tensor,
    ) -> None:
        """DEBUG-level per-head log-prob breakdown of the first minibatch."""
        with torch.no_grad():
            def _blp(logit, key):
                return IndependentBernoulli(logit).log_prob(mb_actions[key]).squeeze(-1).mean().item()
            lp_shoot    = _blp(d_heads.shoot_logit, "shoot")
            lp_pass     = _blp(d_heads.pass_logit, "pass_")
            lp_move     = _blp(d_heads.move_logit, "move")
            lp_tackle   = _blp(d_heads.tackle_logit, "tackle")
            lp_gp       = _blp(d_heads.get_possession_raw, "get_possession_extra")
            lp_mark     = _blp(d_heads.mark_logit, "mark")
            lp_hold     = _blp(d_heads.hold_position_logit, "hold_position")
            lp_exec_mv  = _blp(e_heads.exec_move_logit, "exec_move")
            lp_sprint   = _blp(e_heads.sprint_logit, "sprint")
            lp_kick     = _blp(e_heads.kick_logit, "kick")
            lp_tackle_a = _blp(e_heads.tackle_attempt_logit, "tackle_attempt")
            log_kappa_move = self.execution_net.move_dir_log_kappa.to(self.device)
            log_kappa_kick = self.execution_net.kick_dir_log_kappa.to(self.device)
            log_std_z_kick = self.execution_net.kick_dir_z_log_std.to(self.device)
            lp_movedir  = self._move_dir_head(e_heads.move_direction, log_kappa_move).log_prob(mb_actions["move_dir_raw"]).mean().item()
            lp_kickdir  = self._kick_dir_head(e_heads.kick_direction, log_kappa_kick, log_std_z_kick).log_prob(mb_actions["kick_dir_raw"]).mean().item()
            lp_new_mb   = new_log_probs.mean().item()
            lp_old_mb   = mb_old_lp.mean().item()
            ratio_mb    = torch.exp(new_log_probs - mb_old_lp)
            dval_mb     = d_heads.value.squeeze(-1).mean().item()
            eval_mb     = e_heads.value.squeeze(-1).mean().item()
            stored_raw   = mb_actions["move_dir_raw"]
            stored_norm  = stored_raw.norm(dim=-1)
            current_norm = e_heads.move_direction.norm(dim=-1)
            mean_vec     = e_heads.move_direction.mean(dim=0)
            angle_deg    = math.degrees(math.atan2(float(mean_vec[1]), float(mean_vec[0])))
        log.debug(
            f"  [DIAG e0 mb0]\n"
            f"    old_lp={lp_old_mb:.3f}  new_lp={lp_new_mb:.3f}  diff={lp_new_mb - lp_old_mb:.3f}\n"
            f"    ratio: mean={ratio_mb.mean():.4f}  std={ratio_mb.std():.4f}"
            f"  min={ratio_mb.min():.4f}  max={ratio_mb.max():.4f}\n"
            f"    new_values={new_values.mean():.3f}  ret(mb)={mb_ret.mean():.3f}"
            f"  [d_val={dval_mb:.3f} e_val={eval_mb:.3f}]\n"
            f"    shoot={lp_shoot:.3f} pass={lp_pass:.3f} move={lp_move:.3f} tackle={lp_tackle:.3f}\n"
            f"    gp={lp_gp:.3f} mark={lp_mark:.3f} hold={lp_hold:.3f}\n"
            f"    exec_mv={lp_exec_mv:.3f} sprint={lp_sprint:.3f} kick={lp_kick:.3f} t_attempt={lp_tackle_a:.3f}\n"
            f"    move_dir={lp_movedir:.3f} kick_dir={lp_kickdir:.3f}\n"
            f"    move_dir log_kappa={self.execution_net.move_dir_log_kappa.data.tolist()}\n"
            f"    [move_dir] cur_norm={current_norm.mean():.3f}  stored_norm={stored_norm.mean():.3f}"
            f"  angle={angle_deg:.1f}deg"
        )

    def _ppo_log_epoch_summary(self, ctx: "_UpdateContext", stats: "_UpdateStats", epoch_i: int,
                               epoch_slice_start: int) -> None:
        """Per-epoch lines: train-side policy/surrogate/percentiles/heads/value,
        then the held-out side (if any), then the epoch's grad-norm spread.

        "[policy]"/"[value]" are TRAIN-side numbers. They pool every minibatch of
        the epoch, so rows from early minibatches reflect staler weights than
        late ones: a summary over the epoch's trajectory, not one snapshot. The
        held-out lines are one clean snapshot, since held-out rows never train.
        Full-epoch means are reported because any single minibatch's pol=/kl= is
        effectively a random draw."""
        _epoch_pol = stats.policy_loss[epoch_slice_start:]
        _epoch_kl = stats.kl[epoch_slice_start:]
        _epoch_val = stats.value_loss[epoch_slice_start:]
        mean_pol_epoch = float(np.mean(_epoch_pol)) if _epoch_pol else 0.0
        mean_kl_epoch = float(np.mean(_epoch_kl)) if _epoch_kl else 0.0
        mean_val_epoch = float(np.mean(_epoch_val)) if _epoch_val else 0.0
        # Trimmed (p10-p90, p25-p75) companions to the means. "val" here is the
        # train-side value-head loss, not the held-out set.
        _epoch_pol_cat = torch.cat(stats.epoch_sample_policy_loss) if stats.epoch_sample_policy_loss else torch.zeros(0)
        _epoch_val_cat = torch.cat(stats.epoch_sample_value_loss) if stats.epoch_sample_value_loss else torch.zeros(0)
        mean_pol_epoch_trimmed = _trimmed_mean_p10_p90(_epoch_pol_cat) if stats.epoch_sample_policy_loss else float("nan")
        mean_pol_epoch_trimmed2 = _trimmed_mean_p25_p75(_epoch_pol_cat) if stats.epoch_sample_policy_loss else float("nan")
        mean_val_epoch_trimmed = _trimmed_mean_p10_p90(_epoch_val_cat) if stats.epoch_sample_value_loss else float("nan")
        mean_val_epoch_trimmed2 = _trimmed_mean_p25_p75(_epoch_val_cat) if stats.epoch_sample_value_loss else float("nan")
        # E[(r-1)*A], unclipped and without dual-clip.
        _epoch_surrogate_cat = torch.cat(stats.epoch_sample_surrogate) if stats.epoch_sample_surrogate else torch.zeros(0)
        mean_surrogate_epoch = float(_epoch_surrogate_cat.mean()) if stats.epoch_sample_surrogate else float("nan")
        mean_surrogate_epoch_trimmed = _trimmed_mean_p10_p90(_epoch_surrogate_cat) if stats.epoch_sample_surrogate else float("nan")
        mean_surrogate_epoch_trimmed2 = _trimmed_mean_p25_p75(_epoch_surrogate_cat) if stats.epoch_sample_surrogate else float("nan")

        log.info(f"  == epoch {epoch_i + 1}/{self.n_epochs} " + "=" * 32)
        log.info(
            f"  [policy] mean={mean_pol_epoch:.4f}  p10_p90={mean_pol_epoch_trimmed:.4f}  "
            f"p25_p75={mean_pol_epoch_trimmed2:.4f}  "
            f"kl={mean_kl_epoch:.4f}  (n={len(_epoch_kl)} minibatch(es), {stats.epoch_times[-1]:.0f}ms)"
        )
        if stats.epoch_sample_surrogate:
            # Equals -[policy]'s mean wherever clip and the dual-clip floor are
            # inactive (mean(A) ~= 0 after normalisation). Its value is in the rows
            # where clipping engages: [policy] floors those to protect the step,
            # which also hides how far the ratios really moved.
            log.info(
                f"  [policy surrogate] mean={mean_surrogate_epoch:.4f}  "
                f"p10_p90={mean_surrogate_epoch_trimmed:.4f}  "
                f"p25_p75={mean_surrogate_epoch_trimmed2:.4f}  "
                f"(n={len(_epoch_surrogate_cat):,})"
            )
        if stats.epoch_sample_policy_loss:
            # Distribution shape, not just the mean: a few huge samples can drag the
            # mean while the bulk sits near 0.
            log.info(f"  [policy percentiles, n={len(_epoch_pol_cat):,}]  {_percentile_str(_epoch_pol_cat)}")
        # This epoch's slice of the per-head counterfactual policy loss (each head's
        # ratio alone, others held at 1). Not a decomposition: it does not sum to
        # the [policy] mean.
        _epoch_head_pol = stats.head_policy_loss[epoch_slice_start:]
        if _epoch_head_pol:
            log.info(f"  [policy heads] {self._active_heads_str(torch.stack(_epoch_head_pol).mean(dim=0).tolist())}")
        log.info(
            f"  [value] mean={mean_val_epoch:.4f}(x{self.vf_coef})={self.vf_coef * mean_val_epoch:.4f}  "
            f"p10_p90={mean_val_epoch_trimmed:.4f}(x{self.vf_coef})="
            f"{self.vf_coef * mean_val_epoch_trimmed:.4f}  "
            f"p25_p75={mean_val_epoch_trimmed2:.4f}(x{self.vf_coef})="
            f"{self.vf_coef * mean_val_epoch_trimmed2:.4f}"
        )
        if ctx.val_batch is not None:
            self._ppo_log_held_out_epoch(ctx)
        # Pre-clip grad-norm spread for this epoch's minibatches (main and
        # direction are separate clip groups); std/min show whether a high mean is
        # a broad shift or one or two spikes.
        _epoch_gn_main = stats.grad_norm_main[epoch_slice_start:]
        _epoch_gn_dir = stats.grad_norm_dir[epoch_slice_start:] if stats.grad_norm_dir else []
        if _epoch_gn_main:
            _gn_main_arr = np.array(_epoch_gn_main)
            _gn_line = (
                f"  [grad norm] main: mean={_gn_main_arr.mean():.3f} std={_gn_main_arr.std():.3f} "
                f"min={_gn_main_arr.min():.3f} max={_gn_main_arr.max():.3f} (n={len(_epoch_gn_main)})"
            )
            if _epoch_gn_dir:
                _gn_dir_arr = np.array(_epoch_gn_dir)
                _gn_line += (
                    f"\n              direction: mean={_gn_dir_arr.mean():.3f} "
                    f"std={_gn_dir_arr.std():.3f} min={_gn_dir_arr.min():.3f} max={_gn_dir_arr.max():.3f}"
                )
            log.info(_gn_line)

    def _ppo_log_held_out_epoch(self, ctx: "_UpdateContext") -> None:
        """Held-out lines for one epoch.

        Held-out policy loss is NOT a generalisation diagnostic: at ratio = 1 the
        clipped surrogate collapses to -mean(normalised advantage), which is 0 by
        construction, so it reads ~0 for any untouched rows. Only [held-out value]
        (a real regression MSE) is an apples-to-apples comparison. The held-out
        surrogate line is a tripwire for the rare held-out row whose ratio drifted
        past the clip band purely through training on other rows."""
        val_batch = ctx.val_batch
        (
            _val_pol_loss, _val_val_loss, _val_kl, _val_per_sample_pol,
            _val_val_loss_trimmed, _val_val_loss_trimmed2,
            _val_pol_loss_trimmed, _val_pol_loss_trimmed2,
            _val_surrogate_mean, _val_surrogate_trimmed, _val_surrogate_trimmed2,
            _val_head_pol,
        ) = self._eval_val_episode_losses(val_batch, ctx.clip)
        log.info(
            f"  [held-out policy] mean={_val_pol_loss:.4f}  "
            f"p10_p90={_val_pol_loss_trimmed:.4f}  p25_p75={_val_pol_loss_trimmed2:.4f}  "
            f"kl={_val_kl:.4f}  "
            f"(n={len(val_batch['log_probs'])} held-out steps, never trained on)"
        )
        log.info(
            f"  [held-out policy surrogate] mean={_val_surrogate_mean:.4f}  "
            f"p10_p90={_val_surrogate_trimmed:.4f}  p25_p75={_val_surrogate_trimmed2:.4f}  "
            f"(n={len(_val_per_sample_pol)} held-out steps)"
        )
        if len(_val_per_sample_pol) > 0:
            log.info(
                f"  [held-out policy percentiles, n={len(_val_per_sample_pol):,}]  {_percentile_str(_val_per_sample_pol)}"
            )
        if _val_head_pol is not None:
            log.info(f"  [held-out policy heads] {self._active_heads_str(_val_head_pol.tolist())}")
        log.info(
            f"  [held-out value] mean={_val_val_loss:.4f}(x{self.vf_coef})="
            f"{self.vf_coef * _val_val_loss:.4f}  "
            f"p10_p90={_val_val_loss_trimmed:.4f}(x{self.vf_coef})="
            f"{self.vf_coef * _val_val_loss_trimmed:.4f}  "
            f"p25_p75={_val_val_loss_trimmed2:.4f}(x{self.vf_coef})="
            f"{self.vf_coef * _val_val_loss_trimmed2:.4f}"
        )

    def _ppo_kl_diag_head_log_probs(self, diag_obs: dict, diag_act: dict) -> tuple[dict, object]:
        """Per-head log-probs of the stored actions under the current policy, for
        the KL diagnostic. Returns ({head name: (rows,) tensor}, execution heads).

        Matches the real (masked) training log-prob: decision heads frozen for
        this curriculum phase and the permanently frozen kick_spin read 0, and
        sub-parameter heads are gated by their parent action. Direction heads are
        unweighted, as in the real importance ratio (the ent_*_weight knobs only
        scale the entropy bonus)."""
        with torch.no_grad():
            _sat_d, _oat_d = _ai_types(diag_obs)
            d_d = self.decision_net(
                diag_obs["self_feat"], diag_obs["other_feat"],
                diag_obs["exists_mask"], diag_obs["ball_feat"], diag_obs["global_feat"],
                _sat_d, _oat_d,
                ball_physics_full=diag_obs.get("ball_physics_full"),
                self_physics_full=diag_obs.get("self_physics_full"),
                other_physics_full=diag_obs.get("other_physics_full"),
            )
            e_d = self.execution_net(
                diag_obs["self_feat"], diag_obs["other_feat"],
                diag_obs["exists_mask"], diag_obs["ball_feat"], diag_obs["global_feat"], d_d,
                _sat_d, _oat_d,
            )

            def _blpv(logit, key):
                return IndependentBernoulli(logit).log_prob(diag_act[key]).squeeze(-1)
            lp = {
                "shoot": _blpv(d_d.shoot_logit, "shoot"),
                "pass_": _blpv(d_d.pass_logit, "pass_"),
                "move": _blpv(d_d.move_logit, "move"),
                "tackle": _blpv(d_d.tackle_logit, "tackle"),
                "gp_extra": _blpv(d_d.get_possession_raw, "get_possession_extra"),
                "mark": _blpv(d_d.mark_logit, "mark"),
                "hold": _blpv(d_d.hold_position_logit, "hold_position"),
                "sprint": _blpv(e_d.sprint_logit, "sprint"),
                "kick": _blpv(e_d.kick_logit, "kick"),
                "tackle_attempt": _blpv(e_d.tackle_attempt_logit, "tackle_attempt"),
            }
            for name, attr in (("shoot", "shoot_logit"), ("pass_", "pass_logit"), ("move", "move_logit"),
                               ("tackle", "tackle_logit"), ("gp_extra", "get_possession_raw"),
                               ("mark", "mark_logit"), ("hold", "hold_position_logit")):
                if attr in self._ppo_lp_masked_heads:
                    lp[name] = torch.zeros_like(lp[name])
            _lsm = self.execution_net.move_dir_log_kappa.to(self.device)
            _lsk = self.execution_net.kick_dir_log_kappa.to(self.device)
            _lskz = self.execution_net.kick_dir_z_log_std.to(self.device)
            _lspow = self.execution_net.kick_power_log_std.to(self.device)
            _lsspin = self.execution_net.kick_spin_log_std.to(self.device)
            lp["move_dir"] = self._move_dir_head(e_d.move_direction, _lsm).log_prob(diag_act["move_dir_raw"])
            lp["kick_dir"] = self._kick_dir_head(e_d.kick_direction, _lsk, _lskz).log_prob(diag_act["kick_dir_raw"])
            lp["kick_power"] = self._kick_power_head(e_d.kick_power, _lspow).log_prob(diag_act["kick_power_raw"])
            lp["kick_spin"] = (
                torch.zeros(diag_act["kick_spin_raw"].shape[0], device=self.device)
                if self._kick_spin_frozen else
                self._kick_spin_dist(e_d.kick_spin, _lsspin).log_prob(diag_act["kick_spin_raw"]).sum(dim=-1)
            )
            _exec_move_mask_d = (diag_act["exec_move"].squeeze(-1) > 0.5).float()
            _kick_mask_d = (diag_act["kick"].squeeze(-1) > 0.5).float()
            lp["sprint"] = lp["sprint"] * _exec_move_mask_d
            lp["move_dir"] = lp["move_dir"] * _exec_move_mask_d
            lp["kick_dir"] = lp["kick_dir"] * _kick_mask_d
            lp["kick_power"] = lp["kick_power"] * _kick_mask_d
            lp["kick_spin"] = lp["kick_spin"] * _kick_mask_d
        return lp, e_d

    def _ppo_log_kl_diagnostics(self, ctx: "_UpdateContext", stats: "_UpdateStats") -> None:
        """When the update's mean KL exceeds KL_DIAG_THRESHOLD: ratio percentiles,
        then a per-head breakdown of the first 256 rows under the FINAL weights
        (worst, top-2 and best samples). Not necessarily the sample behind the true
        max ratio; that one is the [ratio spike] line in the update summary."""
        mean_kl = float(np.mean(stats.kl)) if stats.kl else 0.0
        # Median alongside the mean: a few near-saturated Bernoulli samples can
        # dominate the mean through log-ratio blowup without any broad drift.
        median_kl = float(np.median(stats.kl)) if stats.kl else 0.0
        if not (mean_kl > KL_DIAG_THRESHOLD and stats.ratios):
            return
        batch, n = ctx.batch, ctx.n
        move_log_std = self.execution_net.move_dir_log_kappa.data.tolist()
        kick_log_std = self.execution_net.kick_dir_log_kappa.data.tolist()
        ratios_t = torch.cat(stats.ratios)
        ratios_max = ratios_t.max()  # true max, before any subsampling
        ratios_t = _quantile_sample(ratios_t)
        log.info(
            f"  [KL mean={mean_kl:.4f} median={median_kl:.4f} > {KL_DIAG_THRESHOLD}] ratio percentiles:"
            f"  p5={ratios_t.quantile(0.05):.3f}"
            f"  p25={ratios_t.quantile(0.25):.3f}"
            f"  p50={ratios_t.quantile(0.50):.3f}"
            f"  p75={ratios_t.quantile(0.75):.3f}"
            f"  p95={ratios_t.quantile(0.95):.3f}"
            f"  max={ratios_max:.3f}\n"
            f"  move_dir_log_kappa={move_log_std}  kick_dir_log_kappa={kick_log_std}"
        )
        diag_n = min(256, n)
        diag_obs = {k.replace("obs/", ""): batch[k][:diag_n].to(self.device)
                    for k in batch if k.startswith("obs/")}
        diag_act = {k.replace("action/", ""): batch[k][:diag_n].to(self.device)
                    for k in batch if k.startswith("action/")}
        diag_old_lp = ctx.old_log_probs[:diag_n].to(self.device)
        # batch["advantages"] is the raw (un-normalised) advantage; the update
        # normalises a separate copy.
        diag_adv = batch["advantages"][:diag_n]
        lp, e_d = self._ppo_kl_diag_head_log_probs(diag_obs, diag_act)
        _per_head_names = ["shoot", "pass_", "move", "tackle", "gp_extra", "mark", "hold",
                           "sprint", "kick", "tackle_attempt", "move_dir", "kick_dir",
                           "kick_power", "kick_spin"]
        diag_new_lp = (lp["shoot"] + lp["pass_"] + lp["move"] + lp["tackle"] + lp["gp_extra"] +
                       lp["mark"] + lp["hold"] + lp["sprint"] + lp["kick"] + lp["tackle_attempt"] +
                       lp["move_dir"] + lp["kick_dir"] + lp["kick_power"] + lp["kick_spin"])
        diag_ratio = torch.exp(diag_new_lp - diag_old_lp)
        worst_i = int(diag_ratio.argmax())
        stored_mv = diag_act["move_dir_raw"][worst_i]
        new_mv_mean = e_d.move_direction[worst_i]
        s_angle = math.degrees(math.atan2(float(stored_mv[1]), float(stored_mv[0])))
        n_angle = math.degrees(math.atan2(float(new_mv_mean[1]), float(new_mv_mean[0])))

        # Angular difference over all diag rows: "one freak sample" vs "the whole
        # batch's move_dir genuinely shifted".
        with torch.no_grad():
            _stored_angles_all = torch.atan2(diag_act["move_dir_raw"][:, 1], diag_act["move_dir_raw"][:, 0])
            _new_angles_all = torch.atan2(e_d.move_direction[:, 1], e_d.move_direction[:, 0])
            _ang_diff_all = (_stored_angles_all - _new_angles_all).abs() * (180.0 / math.pi)
            _ang_diff_all = torch.minimum(_ang_diff_all, 360.0 - _ang_diff_all)
            # (diag_n, 14), same terms as diag_new_lp, for per-sample head attribution.
            _per_head_new_lp = torch.stack([lp[k] for k in _per_head_names], dim=-1)

        _new_lp_heads_map = {k: float(lp[k].mean()) for k in _per_head_names}
        _head_lp_delta_str = ""
        _old_per_head = None
        if "head_log_probs" in batch:
            old_hlp = batch["head_log_probs"][:diag_n].mean(dim=0)
            for _ki, _k in enumerate(HEAD_LP_KEYS):
                _delta = _new_lp_heads_map.get(_k, 0.0) - float(old_hlp[_ki])
                if abs(_delta) > 0.05:
                    _head_lp_delta_str += f" {_k}:{_delta:+.2f}"
            # Per-sample old per-head lp in _per_head_names order: HEAD_LP_KEYS has
            # an extra exec_move column the 14-head stack does not.
            _old_full = batch["head_log_probs"][:diag_n].to(self.device)
            _exec_move_col = HEAD_LP_KEYS.index("exec_move")
            _keep_cols = [i for i in range(_old_full.shape[-1]) if i != _exec_move_col]
            _old_per_head = _old_full[:, _keep_cols]

        _topk = min(2, diag_n)
        _worst_idxs = diag_ratio.topk(_topk).indices.tolist()
        _worst_lines = [
            self._kl_diag_sample_line(batch, n, _wi, diag_ratio, diag_old_lp, diag_new_lp, diag_adv,
                                      _per_head_new_lp, _old_per_head, _per_head_names, e_d)
            for _wi in _worst_idxs
        ]
        _worst_delta_row = _per_head_new_lp[worst_i]
        if _old_per_head is not None:
            _worst_delta_row = _worst_delta_row - _old_per_head[worst_i]
        _worst_delta_str = "  ".join(
            f"{_n}:{float(_v):+.3f}" for _n, _v in zip(_per_head_names, _worst_delta_row)
            if abs(float(_v)) > 0.02
        )
        # Best sample: highest new log-prob (what the policy is most confident about).
        best_i = int(diag_new_lp.argmax())
        _best_mv = diag_act["move_dir_raw"][best_i]
        _best_mv_mean = e_d.move_direction[best_i]
        _best_s_angle = math.degrees(math.atan2(float(_best_mv[1]), float(_best_mv[0])))
        _best_n_angle = math.degrees(math.atan2(float(_best_mv_mean[1]), float(_best_mv_mean[0])))
        _best_per_head = _per_head_new_lp[best_i]
        _best_contribs_sorted = sorted(
            ((_per_head_names[_hi], float(_best_per_head[_hi])) for _hi in range(len(_per_head_names))),
            key=lambda kv: kv[1], reverse=True,
        )
        _best_contrib_str = "  ".join(f"{_n}:{_v:.3f}" for _n, _v in _best_contribs_sorted if abs(_v) > 0.02)
        _best_adv = float(diag_adv[best_i]) if best_i < len(diag_adv) else float("nan")
        lp_mvdir_d, lp_kkdir_d = lp["move_dir"], lp["kick_dir"]
        log.info(
            f"  [per-head new lp means, n={diag_n}]\n"
            f"    shoot={lp['shoot'].mean():.3f}  pass={lp['pass_'].mean():.3f}"
            f"  move={lp['move'].mean():.3f}  tackle={lp['tackle'].mean():.3f}"
            f"  gp={lp['gp_extra'].mean():.3f}  mark={lp['mark'].mean():.3f}  hold={lp['hold'].mean():.3f}\n"
            f"    sprint={lp['sprint'].mean():.3f}  kick={lp['kick'].mean():.3f}"
            f"  t_att={lp['tackle_attempt'].mean():.3f}\n"
            f"    move_dir={lp_mvdir_d.mean():.3f} (min={lp_mvdir_d.min():.3f} max={lp_mvdir_d.max():.3f})"
            f"  kick_dir={lp_kkdir_d.mean():.3f} (min={lp_kkdir_d.min():.3f} max={lp_kkdir_d.max():.3f})\n"
            + (f"  [head lp deltas (new-old, |d|>0.05)]{_head_lp_delta_str}\n" if _head_lp_delta_str else "")
            + f"  [worst sample] idx={worst_i}  ratio={diag_ratio[worst_i]:.3f}"
            f"  adv={float(diag_adv[worst_i]):+.3f}"
            f"  old_lp={diag_old_lp[worst_i]:.3f}  new_lp={diag_new_lp[worst_i]:.3f}\n"
            f"    stored move_dir={s_angle:.1f}°  new_mean={n_angle:.1f}°"
            f"  angular_diff={min(abs(s_angle-n_angle), 360-abs(s_angle-n_angle)):.1f}°\n"
            f"    [move_dir angular_diff distribution, n={diag_n}]"
            f"  p50={float(_ang_diff_all.quantile(0.50)):.1f}°"
            f"  p90={float(_ang_diff_all.quantile(0.90)):.1f}°"
            f"  p99={float(_ang_diff_all.quantile(0.99)):.1f}°"
            f"  max={float(_ang_diff_all.max()):.1f}°"
            f"  -- low p90/p99 with a high max = one outlier sample, not a broad shift\n"
            f"    [worst sample per-head delta, sorted by |delta|] {_worst_delta_str}\n"
            f"  [top-{_topk} highest-ratio samples]\n"
            + "\n".join(_worst_lines)
            + f"\n  [best sample (highest new_lp)] idx={best_i}  new_lp={diag_new_lp[best_i]:.3f}"
            f"  adv={_best_adv:+.3f}"
            f"  stored move_dir={_best_s_angle:.1f}°  new_mean={_best_n_angle:.1f}°\n"
            f"    per-head contributions: {_best_contrib_str}"
        )

    def _kl_diag_sample_line(self, batch: dict, n: int, _wi: int, diag_ratio, diag_old_lp, diag_new_lp,
                             diag_adv, _per_head_new_lp, _old_per_head, _per_head_names, e_d) -> str:
        """One highest-ratio sample's lines for the KL diagnostic."""
        _rcomp_list = batch.get("reward_comps_raw", [])
        _outcome_list = batch.get("step_outcomes", [])
        _old_lp_wi = float(diag_old_lp[_wi])
        _new_lp_wi = float(diag_new_lp[_wi])
        _adv_wi = float(diag_adv[_wi]) if _wi < len(diag_adv) else float("nan")
        _rew_wi = float(batch["rewards"][_wi]) if _wi < n else float("nan")
        _ret_wi = float(batch["returns"][_wi]) if _wi < n else float("nan")
        _val_wi = float(batch["values"][_wi]) if _wi < n else float("nan")
        _rc = _rcomp_list[_wi] if _wi < len(_rcomp_list) else {}
        _rcomp_str = "  ".join(f"{_k}={_v:+.3f}" for _k, _v in _rc.items() if abs(_v) > 0.001) or "n/a"
        _oc = _outcome_list[_wi] if _wi < len(_outcome_list) else ""
        _outcome_wi = f"terminal:{_oc}" if _oc else "mid-ep"
        _delta_row = _per_head_new_lp[_wi]
        if _old_per_head is not None:
            _delta_row = _delta_row - _old_per_head[_wi]
        _contribs = sorted(
            ((_per_head_names[_hi], float(_delta_row[_hi])) for _hi in range(len(_per_head_names))),
            key=lambda kv: abs(kv[1]), reverse=True,
        )
        _contrib_str = "  ".join(f"{_n}:{_v:+.3f}" for _n, _v in _contribs if abs(_v) > 0.02)
        # Saturation check: near-0/near-1 probabilities make the log-ratio highly
        # nonlinear. These are the CURRENT policy's probabilities, a proxy for the
        # sampling-time ones.
        _sat_bern_heads = [
            ("exec_move", e_d.exec_move_logit), ("sprint", e_d.sprint_logit),
            ("kick", e_d.kick_logit), ("tackle_attempt", e_d.tackle_attempt_logit),
        ]
        _sat_str = "  ".join(f"{_hn}_p_new={float(torch.sigmoid(_hl[_wi])):.4f}" for _hn, _hl in _sat_bern_heads)
        return (
            f"    idx={_wi:4d}  ratio={float(diag_ratio[_wi]):8.3f}  adv={_adv_wi:+.3f}"
            f"  lp: old={_old_lp_wi:.3f}  new={_new_lp_wi:.3f}\n"
            f"      rew={_rew_wi:+.4f}  ret={_ret_wi:+.4f}  val={_val_wi:+.4f}  outcome={_outcome_wi}\n"
            f"      rew_breakdown: {_rcomp_str}\n"
            f"      head_deltas: {_contrib_str}\n"
            f"      saturation: {_sat_str}"
        )

    def _ppo_log_update_summary(self, ctx: "_UpdateContext", stats: "_UpdateStats",
                                spread_start: tuple[float, float, float]) -> None:
        """End-of-update lines: advantage, per-episode |advantage|, ratio, ratio
        spike, per-head grad norm, direction-spread drift, per-head KL and
        counterfactual policy loss, grad clipping, direction param delta and the
        value error by outcome."""
        batch = ctx.batch
        if stats.adv_mean:
            log.info(
                f"  [advantage] mean={float(np.mean(stats.adv_mean)):.3f}"
                f"  std={float(np.mean(stats.adv_std)):.3f}"
                f"  min={float(np.min(stats.adv_min)):.3f}"
                f"  max={float(np.max(stats.adv_max)):.3f}"
            )
        # Per-episode mean |advantage| from the RAW advantages (the update
        # normalises a separate copy), as percentiles across episodes: tells a
        # rollout of uniformly moderate advantages from one with a few huge-swing
        # episodes, which the pooled per-row line above cannot.
        if "dones" in batch:
            _ep_abs_adv_means = _episode_abs_adv_means(batch["advantages"], batch["dones"])
            if _ep_abs_adv_means is not None:
                log.info(
                    f"  [advantage |.| by episode, n_episodes={len(_ep_abs_adv_means)}]  "
                    f"{_percentile_str(torch.from_numpy(_ep_abs_adv_means), digits=3)}"
                )
        if stats.ratios:
            _ratios_all = torch.cat(stats.ratios)
            _ratio_clip_frac = float(np.mean(stats.ratio_clipped_frac)) if stats.ratio_clipped_frac else 0.0
            log.info(
                f"  [ratio] mean={_ratios_all.mean():.4f}"
                f"  std={_ratios_all.std():.4f}"
                f"  min={_ratios_all.min():.4f}"
                f"  max={_ratios_all.max():.4f}"
                f"  clipped={_ratio_clip_frac * 100:.1f}%"
            )
        if stats.ratio_spike["ratio"] > float("-inf"):
            self._ppo_log_ratio_spike(batch, stats.ratio_spike)
        _head_grad_norm_str = "  ".join(
            f"{name}={np.mean(vals):.3f}" for name, vals in stats.head_grad_norm.items() if vals
        )
        if _head_grad_norm_str:
            log.info(f"  [exec head grad norm] {_head_grad_norm_str}")
        self._ppo_log_direction_spread_drift(stats, spread_start)
        _disc_shift_str = "  ".join(
            f"{name}={np.mean(vals):.4f}" for name, vals in stats.discrete_logit_shift.items() if vals
        )
        if _disc_shift_str:
            log.info(f"  [exec discrete Δlogit per opt step] {_disc_shift_str}")
        if stats.head_kl:
            log.info(f"  [per-head KL] {self._active_heads_str(torch.stack(stats.head_kl).mean(dim=0).tolist())}")
        if stats.head_policy_loss:
            # Counterfactual (each head's ratio alone), not a decomposition; sorted
            # by |value| to show which heads pull the objective away from 0.
            _inactive_heads = self._inactive_head_lp_keys()
            _mean_head_pol = torch.stack(stats.head_policy_loss).mean(dim=0)
            _head_pol_pairs = sorted(
                (kv for kv in zip(HEAD_LP_KEYS, _mean_head_pol.tolist()) if kv[0] not in _inactive_heads),
                key=lambda kv: -abs(kv[1]),
            )
            _head_pol_str = "  ".join(f"{k}={v:+.4f}" for k, v in _head_pol_pairs)
            log.info(
                f"  [per-head policy_loss (counterfactual, ratio-in-isolation, "
                f"sorted by |.|)] {_head_pol_str}"
            )
        self._ppo_log_grad_clip_summary(stats)
        _value_by_outcome = stats.value_by_outcome()
        if _value_by_outcome:
            log.info(f"  [value RMSE by outcome] {format_outcome_rmse_breakdown(_value_by_outcome)}")

    def _ppo_log_ratio_spike(self, batch: dict, spike: dict) -> None:
        """The single highest-ratio row seen during the update, captured at the
        weights that produced it."""
        _spike_row = spike["row"]
        _new_head_row = spike["per_head_new"].tolist()
        if "head_log_probs" in batch:
            _old_head_row = batch["head_log_probs"][_spike_row].tolist()
            _deltas = list(zip(HEAD_LP_KEYS, (nv - ov for nv, ov in zip(_new_head_row, _old_head_row))))
        else:
            _deltas = list(zip(HEAD_LP_KEYS, _new_head_row))
        _deltas.sort(key=lambda kv: abs(kv[1]), reverse=True)
        _spike_delta_str = "  ".join(f"{k}:{v:+.3f}" for k, v in _deltas if abs(v) > 0.02)
        _rcomp_list = batch.get("reward_comps_raw", [])
        _outcome_list = batch.get("step_outcomes", [])
        _rc = _rcomp_list[_spike_row] if _spike_row < len(_rcomp_list) else {}
        _rcomp_str = "  ".join(f"{k}={v:+.3f}" for k, v in _rc.items() if abs(v) > 0.001) or "n/a"
        _oc = _outcome_list[_spike_row] if _spike_row < len(_outcome_list) else ""
        _spike_outcome_str = f"terminal:{_oc}" if _oc else "mid-ep"
        log.info(
            f"  [ratio spike] max={spike['ratio']:.3f}"
            f"  epoch={spike['epoch'] + 1}/{self.n_epochs}  mb={spike['mb']}"
            f"  row={_spike_row}(global)  adv={spike['adv']:+.3f}"
            f"  old_lp={spike['old_lp']:.3f}  new_lp={spike['new_lp']:.3f}\n"
            f"    rew_breakdown: {_rcomp_str}  outcome={_spike_outcome_str}\n"
            f"    per-head Δ(new-old), sorted by |Δ|: {_spike_delta_str}"
        )

    def _ppo_log_direction_spread_drift(self, stats: "_UpdateStats",
                                        spread_start: tuple[float, float, float]) -> None:
        """Start -> end log_kappa / log_std of the direction heads, the analytic KL
        of that spread change alone, and the per-step mean/spread drift."""
        _move_ls_start, _kick_ls_start, _kickz_ls_start = spread_start
        _move_ls_end = float(self.execution_net.move_dir_log_kappa.mean().item())
        _kick_ls_end = float(self.execution_net.kick_dir_log_kappa.mean().item())
        _kickz_ls_end = float(self.execution_net.kick_dir_z_log_std.mean().item())
        log.info(
            f"  [exec continuous log_kappa] move_direction: start={_move_ls_start:.4f} end={_move_ls_end:.4f}"
            f"   kick_direction: start={_kick_ls_start:.4f} end={_kick_ls_end:.4f}"
            f"   kick_direction_z (log_std): start={_kickz_ls_start:.4f} end={_kickz_ls_end:.4f}"
        )
        # Analytic KL of kick_dir's azimuth vs elevation from the start->end
        # kappa/sigma_z alone (mean angle held fixed). Isolates how much kick_dir's
        # SHARPNESS moved; not the ratio-based per-head KL used for training.
        _kick_azimuth_kl = float(_von_mises_kl(
            torch.tensor(0.0), torch.tensor(math.exp(_kick_ls_end)),
            torch.tensor(0.0), torch.tensor(math.exp(_kick_ls_start)),
        ))
        _kick_z_kl = float(torch.distributions.kl_divergence(
            torch.distributions.Normal(torch.tensor(0.0), torch.tensor(math.exp(_kickz_ls_end))),
            torch.distributions.Normal(torch.tensor(0.0), torch.tensor(math.exp(_kickz_ls_start))),
        ))
        log.info(
            f"  [kick_dir analytic KL split] azimuth(kappa-only)={_kick_azimuth_kl:.4f}"
            f"  elevation(sigma_z-only)={_kick_z_kl:.4f}  (start->end snapshot, mean-angle/mean-z held fixed)"
        )
        _ls_end_by_name = {"move_direction": _move_ls_end, "kick_direction": _kick_ls_end}
        _cont_shift_parts = []
        for name, vals in stats.continuous_mean_shift.items():
            if not vals:
                continue
            n_steps = len(vals)
            mean_dmean = float(np.mean(vals))
            mean_dlog_std = float(np.mean(stats.continuous_log_std_shift[name]))
            epoch_dmean = float(np.sum(vals))   # cumulative shift over the whole update
            ls_end = _ls_end_by_name.get(name, 0.0)
            mean_deg = _angular_dmean_deg(mean_dmean)
            epoch_deg = _angular_dmean_deg(epoch_dmean / max(n_steps, 1)) * n_steps  # rough total
            mean_dstd_deg = _angular_dlog_std_deg(mean_dlog_std, ls_end)
            _cont_shift_parts.append(
                f"{name}("
                f"dmean={mean_dmean:.4f}≈{mean_deg:.2f}°/step  epoch≈{epoch_deg:.1f}°  "
                f"dlog_kappa={mean_dlog_std:.5f}  Δ(ang std)°={mean_dstd_deg:.3f}/step)"
            )
        # kick_dir_z_log_std is a plain Gaussian log_std, so it uses the direct
        # sigma -> degrees approximation (small z approximates an elevation angle).
        if stats.kickz_log_std_shift:
            mean_dlog_stdz = float(np.mean(stats.kickz_log_std_shift))
            mean_dstdz_deg = abs(
                math.degrees(math.exp(_kickz_ls_end))
                - math.degrees(math.exp(_kickz_ls_end - mean_dlog_stdz))
            )
            _cont_shift_parts.append(
                f"kick_direction_z(dlog_std={mean_dlog_stdz:.5f}  Δσ°≈{mean_dstdz_deg:.3f}/step)"
            )
        if _cont_shift_parts:
            log.info(f"  [exec continuous Δ per opt step] {'  '.join(_cont_shift_parts)}")

    def _ppo_log_grad_clip_summary(self, stats: "_UpdateStats") -> None:
        """How often each clip group's pre-clip norm exceeded its limit, and the
        direction group's actual applied parameter movement.

        When direction_max_grad_norm is unset the two groups share max_grad_norm,
        so both "limit=" values match and the split only attributes clipping
        activity. Adam's real step is m_hat / sqrt(v_hat), not a function of the
        clipped gradient, so the param delta can diverge from the clip numbers
        (notably right after --reset-optimizer)."""
        g = stats.grad_clip_summary()
        log.info(
            f"  [grad clip] main: {stats.clip_triggered_main}/{g['n_main']} steps clipped ({g['pct_main']:.0f}%)"
            f"  pre-clip norm mean={g['mean_main']:.3f} max={g['max_main']:.3f}  limit={self.max_grad_norm}"
            + (
                f"\n              direction: {stats.clip_triggered_dir}/{g['n_dir']} steps clipped ({g['pct_dir']:.0f}%)"
                f"  pre-clip norm mean={g['mean_dir']:.3f} max={g['max_dir']:.3f}"
                f"  limit={self.direction_max_grad_norm}"
                if g["n_dir"] else ""
            )
        )
        if stats.param_delta_dir:
            log.info(
                f"  [direction param delta] actual ||params_after - params_before|| per opt step:"
                f"  mean={float(np.mean(stats.param_delta_dir)):.4f}"
                f"  max={float(np.max(stats.param_delta_dir)):.4f}"
                f"  (n={len(stats.param_delta_dir)})"
            )


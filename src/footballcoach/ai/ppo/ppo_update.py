"""PPOTrainer's PPO update: the clipped/dual-clip objective, held-out validation episodes, action-opportunity masking and the NaN guard.

_ppo_update is an orchestrator over named stages (see its docstring and
ai/knowledge.md, "PPOTrainer module layout"); its logging-only lines live in
ppo_update_diagnostics.py. Check any change with
scripts/ppo_update_trace_snapshot.py (0 differences for a pure refactor)."""
from __future__ import annotations

import dataclasses
import logging
import math
import time
import warnings
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from footballcoach.ai.action.distributions import kick_power_from_raw
from footballcoach.ai.action.schema import DecisionHeadsRaw
from footballcoach.ai.obs.augment import augment_batch
from footballcoach.ai.obs.y_canonical import mirror_y_obs, y_flip_mask
from footballcoach.ai.obs.canonical import canonicalize_obs
from footballcoach.ai.ppo.rollout_buffer import HEAD_LP_KEYS
from footballcoach.ai.ppo.action_opportunity_stats import format_opportunity_block, has_opportunity_flags, opportunity_masks, opportunity_stats
from footballcoach.ai.ppo.trainer_common import log
from footballcoach.ai.ppo.trainer_common import EXEC_HEAD_MODULES, _DECISION_HEADS_FIELD_NAMES, _ai_types, _detach_decision_heads, _maybe_log_grad_norm_spike, value_mse_by_outcome
from footballcoach.ai.ppo.trainer_common import _lr_warmup_scale
from footballcoach.ai.ppo.ppo_update_diagnostics import (  # noqa: F401  (trimmed means re-exported)
    PPOUpdateDiagnosticsMixin, _trimmed_mean, _trimmed_mean_p10_p90, _trimmed_mean_p25_p75,
)


# exp() overflows float32 past ~88.7. Under dual-clip an overflowed (inf)
# ratio still poisons the gradient: dual-clip selects the finite floor, but
# autograd multiplies the unselected branch's zero gradient by d(exp)/dx = inf,
# giving 0 * inf = NaN. Capping the log-ratio at 80 (exp(80) ~= 5.5e34) keeps
# the ratio finite; a ratio that large is already on dual-clip's zero-gradient
# floor, so only the NaN changes. Below the cap the result is bit-identical.
_MAX_LOG_RATIO = 80.0

# _kick_agreement's fields, in order.
_AGREE_KEYS = ("label_kick_rate", "p_kick_label_kick", "p_kick_label_no_kick", "power_err", "kick_dir_err_deg")


def _kick_agreement(labels: torch.Tensor, e_heads, rows: torch.Tensor) -> list:
    """How close the policy is to its BC expert on the kick group, over the
    boolean ``rows`` of ``labels`` (no grad): the share of rows the expert
    labels "kick now"; the policy's mean P(kick) on those rows and on the
    others; on the "kick now" rows, the mean |power fraction error| and the mean
    angle (deg) between the policy's kick-direction mean and the label's. NaN
    for a field with no rows."""
    from footballcoach.ai.ppo.bc import (
        _I_KICK_DIR_X, _I_KICK_DIR_Y, _I_KICK_DIR_Z, _I_KICK_POWER, _I_KICK_THIS_TICK, _kick_power_physical,
    )
    nan = float("nan")
    with torch.no_grad():
        lab = labels[rows]
        if lab.shape[0] == 0:
            return [nan] * len(_AGREE_KEYS)
        kick = lab[:, _I_KICK_THIS_TICK] > 0.5
        p_kick = torch.sigmoid(e_heads.kick_logit[rows].reshape(-1))
        out = [float(kick.float().mean()),
               float(p_kick[kick].mean()) if kick.any() else nan,
               float(p_kick[~kick].mean()) if (~kick).any() else nan]
        if kick.any():
            pred_power = _kick_power_physical(e_heads.kick_power[rows].reshape(-1))[kick]
            out.append(float((pred_power - lab[kick, _I_KICK_POWER]).abs().mean()))
            target = torch.stack([lab[kick, _I_KICK_DIR_X], lab[kick, _I_KICK_DIR_Y], lab[kick, _I_KICK_DIR_Z]], dim=-1)
            has_dir = target.norm(dim=-1) > 1e-6
            if has_dir.any():
                pred = F.normalize(e_heads.kick_direction[rows][kick][has_dir], dim=-1)
                cos = (pred * F.normalize(target[has_dir], dim=-1)).sum(dim=-1).clamp(-1.0, 1.0)
                out.append(float(torch.rad2deg(torch.acos(cos)).mean()))
            else:
                out.append(nan)
        else:
            out += [nan, nan]
        return out

def _scale_decision_heads_grad(d_heads: DecisionHeadsRaw, coef: float) -> DecisionHeadsRaw:
    """Like _detach_decision_heads, but instead of cutting the graph, clone
    each field and register a backward hook that scales ITS OWN gradient by
    coef before continuing upstream into decision_net -- see
    share_value_grad_with_decision/decision_value_coef (__init__) and
    ai_trainer_knowledge.md "Separate value network" for why this needs a
    dedicated coefficient rather than reusing vf_coef.

    Cloning (not just hooking the original tensor) is load-bearing: d_heads
    is also fed to execution_net in the same forward pass to produce the
    policy's own e_heads, so hooking the original tensor would scale THAT
    gradient contribution too. A clone gets its own node in the graph --
    the hook fires only on the gradient arriving from whatever consumes the
    clone (self.value_net here), leaving the policy's own use of d_heads
    completely unaffected.

    coef == 1.0 is a no-op (returns d_heads itself, no clone needed) --
    callers that also need coef == 0.0 to behave identically to full
    detach should special-case that themselves (kept simple here since the
    grad-hook path still keeps a live, if inert, backward node in that
    case)."""
    if coef == 1.0:
        return d_heads

    def _hook(grad: torch.Tensor, c: float = coef) -> torch.Tensor:
        return grad * c

    fields = {}
    for name in _DECISION_HEADS_FIELD_NAMES:
        v = getattr(d_heads, name)
        if v is not None and v.requires_grad:
            v = v.clone()
            v.register_hook(_hook)
        fields[name] = v
    return dataclasses.replace(d_heads, **fields)

def _kick_power_frac_stats(batch: dict, floor: float = 0.0) -> dict:
    """n / mean / std of the executed kick power fraction (floor + (1-floor) *
    sigmoid of the stored raw draw) over the rows where a kick was sampled."""
    kick = batch["action/kick"].reshape(-1) > 0.5
    if not bool(kick.any()):
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    pf = kick_power_from_raw(batch["action/kick_power_raw"].reshape(-1)[kick].float(), floor)
    return {"n": int(pf.numel()), "mean": float(pf.mean()), "std": float(pf.std()) if pf.numel() > 1 else 0.0}


@dataclasses.dataclass
class _UpdateContext:
    """What one _ppo_update call fixes before its first gradient step."""
    batch: dict                       # train rows: augmented, physics features cached, exact old log-probs
    val_batch: Optional[dict]         # held-out rows, None when the split is off
    n: int
    clip: float
    ent_coef: float
    bc_coeff: float
    bc_kick_coeff: float
    bc_tackle_coeff: float
    has_bc: bool
    adv: torch.Tensor                 # normalised advantages
    old_log_probs: torch.Tensor       # whole-policy log pi_old, for diagnostics
    old_log_probs_ratio: torch.Tensor  # log pi_old used in the ratio (opportunity-masked when enabled)
    returns: torch.Tensor
    opp_mask: Optional[torch.Tensor]  # (rows, heads) action-opportunity mask, or None
    opp_grad_scale: Optional[torch.Tensor]
    warm_base_lrs: list               # this update's scheduled lrs, for the per-update warmup
    step_outcomes: list
    # Per-row BC coefficients (ai/ppo/bc_sources.py): (n_sources + 1, 3)
    # (other, kick, tackle) by row source; None when no per-phase / per-role BC
    # is configured, in which case the global coefficients above apply to
    # every row exactly as before.
    bc_coeff_table: Optional[torch.Tensor] = None


@dataclasses.dataclass
class _UpdateStats:
    """Everything _ppo_update accumulates for its log lines and returned metrics.

    The epoch_sample_* lists hold per-SAMPLE values and are reset every epoch,
    so the per-epoch lines can show distribution shape, not just the mean.
    head_policy_loss is a per-head COUNTERFACTUAL (each head's ratio alone,
    others held at 1), measured before the step like policy_loss; it does not
    sum to policy_loss. head_kl is measured after the step. ratio_spike is the
    single highest-ratio row over the whole update, captured at the weights
    that produced it. param_delta_dir is the applied ||after - before|| of the
    direction parameters, which can differ a lot from the clipped gradient
    norm because Adam's step is m_hat / sqrt(v_hat)."""
    policy_loss: list = dataclasses.field(default_factory=list)
    value_loss: list = dataclasses.field(default_factory=list)
    entropy: list = dataclasses.field(default_factory=list)
    kl: list = dataclasses.field(default_factory=list)
    bc_loss: list = dataclasses.field(default_factory=list)
    bc_tackle_loss: list = dataclasses.field(default_factory=list)   # BCE on tackle_attempt only
    tackle_prob: list = dataclasses.field(default_factory=list)      # mean sigmoid(tackle_attempt_logit) per mb
    kick_prob: list = dataclasses.field(default_factory=list)        # mean sigmoid(kick_logit) per mb
    ratios: list = dataclasses.field(default_factory=list)
    entropy_breakdown: dict = dataclasses.field(default_factory=dict)
    # BC source id -> [(other, kick, tackle) mean group loss over its valid
    # rows, per minibatch] (per-phase / per-role BC only).
    bc_by_source: dict = dataclasses.field(default_factory=dict)
    # BC source id -> per-minibatch kick agreement with the expert
    # (_kick_agreement: label kick rate, P(kick | label kick), P(kick | label
    # no kick), |power error|, kick-direction error deg), NaN where no rows.
    bc_agree_by_source: dict = dataclasses.field(default_factory=dict)
    # How hard BC pulls (_measure_bc_pull, first minibatch of each epoch):
    # [|g_bc| main, |g_rest| main, cos main, |g_bc| dir, |g_rest| dir, cos dir].
    bc_pull: list = dataclasses.field(default_factory=list)
    epoch_sample_policy_loss: list = dataclasses.field(default_factory=list)
    epoch_sample_value_loss: list = dataclasses.field(default_factory=list)
    epoch_sample_surrogate: list = dataclasses.field(default_factory=list)  # unclipped (r-1)*A per row
    ratio_spike: dict = dataclasses.field(default_factory=lambda: {"ratio": float("-inf")})
    mv_log_std_grad: list = dataclasses.field(default_factory=list)  # grad norm on move_dir_log_kappa
    adv_mean: list = dataclasses.field(default_factory=list)
    adv_std: list = dataclasses.field(default_factory=list)
    adv_min: list = dataclasses.field(default_factory=list)
    adv_max: list = dataclasses.field(default_factory=list)
    ratio_clipped_frac: list = dataclasses.field(default_factory=list)
    head_grad_norm: dict = dataclasses.field(default_factory=lambda: {name: [] for name, _ in EXEC_HEAD_MODULES})
    head_kl: list = dataclasses.field(default_factory=list)
    head_policy_loss: list = dataclasses.field(default_factory=list)
    continuous_mean_shift: dict = dataclasses.field(
        default_factory=lambda: {"move_direction": [], "kick_direction": []})
    continuous_log_std_shift: dict = dataclasses.field(
        default_factory=lambda: {"move_direction": [], "kick_direction": []})
    # kick_dir_z_log_std has no per-head KL/entropy slot of its own (it is summed
    # into kick_dir everywhere), so its drift gets its own tracker.
    kickz_log_std_shift: list = dataclasses.field(default_factory=list)
    discrete_logit_shift: dict = dataclasses.field(
        default_factory=lambda: {name: [] for name in ("exec_move", "sprint", "kick", "tackle_attempt")})
    grad_norm_main: list = dataclasses.field(default_factory=list)   # pre-clip, non-direction group
    grad_norm_dir: list = dataclasses.field(default_factory=list)    # pre-clip, direction group
    clip_triggered_main: int = 0
    clip_triggered_dir: int = 0
    param_delta_dir: list = dataclasses.field(default_factory=list)
    epoch_times: list = dataclasses.field(default_factory=list)
    warm_steps_done: int = 0
    first_minibatch_done: bool = False
    # Value error by outcome: squared-error sum, row count and ground-truth
    # moments per outcome string, over the main loop and the value-only
    # continuation.
    outc_sq_err_sum: dict = dataclasses.field(default_factory=dict)
    outc_n_sum: dict = dataclasses.field(default_factory=dict)
    outc_gt_sum: dict = dataclasses.field(default_factory=dict)
    outc_gt_sqsum: dict = dataclasses.field(default_factory=dict)

    def record_advantage(self, mb_adv: torch.Tensor) -> None:
        """Minibatch advantage mean/std/min/max, in one device sync."""
        mean, std, lo, hi = torch.stack([mb_adv.mean(), mb_adv.std(), mb_adv.min(), mb_adv.max()]).tolist()
        self.adv_mean.append(mean)
        self.adv_std.append(std)
        self.adv_min.append(lo)
        self.adv_max.append(hi)

    def accum_value_by_outcome(self, pred: torch.Tensor, target: torch.Tensor, idx: torch.Tensor,
                               step_outcomes: list) -> None:
        if not step_outcomes:
            return
        outcomes_mb = [step_outcomes[i] for i in idx.tolist()]
        for _name, (_mse, _n, _gt_mean, _gt_mean_sq) in value_mse_by_outcome(pred, target, outcomes_mb).items():
            self.outc_sq_err_sum[_name] = self.outc_sq_err_sum.get(_name, 0.0) + _mse * _n
            self.outc_n_sum[_name] = self.outc_n_sum.get(_name, 0) + _n
            self.outc_gt_sum[_name] = self.outc_gt_sum.get(_name, 0.0) + _gt_mean * _n
            self.outc_gt_sqsum[_name] = self.outc_gt_sqsum.get(_name, 0.0) + _gt_mean_sq * _n

    def value_by_outcome(self) -> dict:
        """{outcome: (mse, rows, gt mean, gt mean square)}; empty when nothing was accumulated."""
        if not self.outc_n_sum:
            return {}
        return {
            name: (
                self.outc_sq_err_sum[name] / max(self.outc_n_sum[name], 1),
                self.outc_n_sum[name],
                self.outc_gt_sum[name] / max(self.outc_n_sum[name], 1),
                self.outc_gt_sqsum[name] / max(self.outc_n_sum[name], 1),
            )
            for name in self.outc_sq_err_sum
        }

    def grad_clip_summary(self) -> dict:
        """Per clip group: step count, % of steps clipped, mean and max pre-clip norm."""
        n_main = len(self.grad_norm_main)
        n_dir = len(self.grad_norm_dir)
        return {
            "n_main": n_main,
            "n_dir": n_dir,
            "pct_main": (100.0 * self.clip_triggered_main / n_main) if n_main else 0.0,
            "pct_dir": (100.0 * self.clip_triggered_dir / n_dir) if n_dir else 0.0,
            "mean_main": float(np.mean(self.grad_norm_main)) if n_main else 0.0,
            "max_main": float(np.max(self.grad_norm_main)) if n_main else 0.0,
            "mean_dir": float(np.mean(self.grad_norm_dir)) if n_dir else 0.0,
            "max_dir": float(np.max(self.grad_norm_dir)) if n_dir else 0.0,
        }


def _obs_parts(mb_obs: dict) -> tuple:
    """(self_feat, other_feat, exists_mask, ball_feat, global_feat, self_ai_type,
    other_ai_type) of a minibatch obs dict, in the networks' argument order."""
    sat, oat = _ai_types(mb_obs)
    return (mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"],
            mb_obs["ball_feat"], mb_obs["global_feat"], sat, oat)


def _normalised_sample_weights(w: torch.Tensor) -> torch.Tensor:
    """Per-sample importance weights rescaled to sum to the minibatch size, so the
    loss scale does not depend on the weights' overall magnitude."""
    return w * (len(w) / w.sum().clamp(min=1e-8))


def _ppo_value_loss(new_values: torch.Tensor, mb_returns: torch.Tensor,
                    all_returns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean, per-sample) squared value error, divided by Var(returns) over the
    whole update (floored at 1) so it stays O(1) whatever the return scale and
    does not swamp the policy gradient."""
    ret_var = all_returns.var().clamp(min=1.0)
    per_sample = (new_values - mb_returns).pow(2) / ret_var
    return per_sample.mean(), per_sample


class PPOUpdateMixin(PPOUpdateDiagnosticsMixin):
    """The PPO update step for PPOTrainer. Mixed into PPOTrainer; relies on its attributes."""

    def _bc_heads_for_loss(self, d_heads: DecisionHeadsRaw) -> DecisionHeadsRaw:
        """Decision heads to hand to bc_loss_from_tensor() for the BC
        auxiliary loss computed during PPO's own update loop.

        Returns ``d_heads`` unchanged unless ``bc.bc_trains_frozen_heads`` is
        False AND some heads are currently frozen for this curriculum phase
        (self._ppo_lp_masked_heads, see set_frozen_heads) -- in that case,
        returns a shallow copy with exactly those heads' logits detached.
        The head's own weights are already frozen (requires_grad=False) so
        BC can never update them directly either way; what detaching stops
        is the BC loss backpropagating through the frozen head's forward
        pass into the *shared* decision_net trunk, matching how PPO's own
        policy-gradient (_recompute_log_prob) and entropy (_compute_entropy)
        terms are already masked for these heads.
        """
        masked = self._ppo_lp_masked_heads
        if self._bc_trains_frozen_heads or not masked:
            return d_heads
        return dataclasses.replace(
            d_heads,
            shoot_logit=d_heads.shoot_logit.detach() if "shoot_logit" in masked else d_heads.shoot_logit,
            pass_logit=d_heads.pass_logit.detach() if "pass_logit" in masked else d_heads.pass_logit,
            move_logit=d_heads.move_logit.detach() if "move_logit" in masked else d_heads.move_logit,
            tackle_logit=d_heads.tackle_logit.detach() if "tackle_logit" in masked else d_heads.tackle_logit,
            get_possession_raw=(
                d_heads.get_possession_raw.detach() if "get_possession_raw" in masked
                else d_heads.get_possession_raw
            ),
            mark_logit=d_heads.mark_logit.detach() if "mark_logit" in masked else d_heads.mark_logit,
            hold_position_logit=(
                d_heads.hold_position_logit.detach() if "hold_position_logit" in masked
                else d_heads.hold_position_logit
            ),
        )

    @torch.no_grad()
    def _precompute_physics_full(self, batch: dict, chunk_size: int) -> dict[str, torch.Tensor]:
        """Run the FROZEN physics encoders once over every row of ``batch``
        (a dict with ``obs/self_feat``, ``obs/other_feat``, ``obs/ball_feat``,
        ``obs/global_feat`` CPU tensors) and return their RAW canonical-frame
        outputs as CPU tensors keyed ``obs/{ball,self,other}_physics_full`` --
        ready to merge into the same batch so the generic ``obs/*`` slicing
        picks them up and every ``decision_net(...)`` call site can pass them
        through instead of re-running the encoders on identical rows every
        pass. Empty dict when neither encoder exists. Shared by
        ``_ppo_update`` and ``ppg_value_refit``."""
        dn = self.decision_net
        if dn.ball_physics_encoder is None and dn.player_physics_encoder is None:
            return {}
        n = len(batch["obs/self_feat"])
        ball_parts: list[torch.Tensor] = []
        self_parts: list[torch.Tensor] = []
        other_parts: list[torch.Tensor] = []
        for start in range(0, n, chunk_size):
            idx = torch.arange(start, min(start + chunk_size, n))
            _sf = batch["obs/self_feat"][idx].to(self.device)
            _of = batch["obs/other_feat"][idx].to(self.device)
            _bf = batch["obs/ball_feat"][idx].to(self.device)
            if self.y_canonical:
                # The wrapped networks mirror y-negative rows before their forward pass and pass these
                # cached tensors through untouched, so they must be computed from the mirrored rows.
                _sf, _of, _bf = mirror_y_obs(_sf, _of, _bf, y_flip_mask(_sf))
            sf_c, of_c, bf_c, _ = canonicalize_obs(_sf, _of, _bf)
            gf_c = batch["obs/global_feat"][idx].to(self.device)
            if dn.ball_physics_encoder is not None:
                ball_parts.append(dn.ball_physics_encoder(bf_c, gf_c).cpu())
            if dn.player_physics_encoder is not None:
                self_parts.append(dn.player_physics_encoder(sf_c, gf_c).cpu())
                other_parts.append(dn.player_physics_encoder(of_c, gf_c).cpu())
        out: dict[str, torch.Tensor] = {}
        if ball_parts:
            out["obs/ball_physics_full"] = torch.cat(ball_parts, dim=0)
        if self_parts:
            out["obs/self_physics_full"] = torch.cat(self_parts, dim=0)
            out["obs/other_physics_full"] = torch.cat(other_parts, dim=0)
        return out

    def _dump_nan_diagnostic(
        self, stage: str, epoch_i: int, mb_i: int, mb_idx: torch.Tensor,
        track_ids_mb: list[str], d_heads, e_heads,
        policy_loss: torch.Tensor, value_loss: torch.Tensor, entropy: torch.Tensor,
    ) -> None:
        import dataclasses
        import json
        import time

        secondary_mask = torch.tensor([t != "trainee" for t in track_ids_mb])
        n_secondary = int(secondary_mask.sum())
        n_total = len(track_ids_mb)

        def _field_report(heads, label):
            out = {}
            for f in dataclasses.fields(heads):
                v = getattr(heads, f.name)
                if v is None or not torch.is_tensor(v):
                    continue
                v = v.detach()
                finite = torch.isfinite(v)
                out[f.name] = {
                    "all_finite": bool(finite.all()),
                    "n_nonfinite": int((~finite).sum()),
                    "n_nonfinite_secondary": int((~finite).cpu().reshape(v.shape[0], -1).any(dim=1)[secondary_mask].sum()) if n_secondary else 0,
                    "min": float(v[finite].min()) if finite.any() else None,
                    "max": float(v[finite].max()) if finite.any() else None,
                }
            return out

        report = {
            "stage": stage,
            "epoch_i": epoch_i,
            "mb_i": mb_i,
            "n_rows_total": n_total,
            "n_rows_secondary": n_secondary,
            "secondary_fraction": n_secondary / max(n_total, 1),
            "unique_track_ids_in_minibatch": sorted(set(track_ids_mb)),
            "policy_loss_finite": bool(torch.isfinite(policy_loss)),
            "value_loss_finite": bool(torch.isfinite(value_loss)),
            "entropy_finite": bool(torch.isfinite(entropy)),
            "d_heads": _field_report(d_heads, "d_heads"),
            "e_heads": _field_report(e_heads, "e_heads"),
        }
        out_path = Path(self.checkpoint_dir or ".") / f"nan_diagnostic_{int(time.time())}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2)
        log.error(
            f"[NaN DIAGNOSTIC] stage={stage} epoch={epoch_i} mb={mb_i} "
            f"secondary_rows={n_secondary}/{n_total} ({report['secondary_fraction']:.1%}) "
            f"-- full report written to {out_path}"
        )
        from footballcoach.ai.ppo.ppo_trainer import PPOTrainer  # local: imported by PPOTrainer's module

        raise PPOTrainer._NaNReproFound(str(out_path))

    # -----------------------------------------------------------------------
    # Held-out validation episodes (diagnostic only, see __init__'s
    # val_episode_fraction comment)
    # -----------------------------------------------------------------------

    def _split_train_val_episodes(self, batch: dict) -> tuple[dict, Optional[dict]]:
        """Randomly hold out ``val_episode_fraction`` of complete episodes.

        Segmented per ``track_ids`` (same reasoning as
        ``RolloutBuffer.compute_gae``'s own per-track pass): a flat
        ``dones > 0.5`` scan over the concatenated batch would misidentify
        episode boundaries whenever a secondary neural opponent's rows are
        interleaved with the trainee's own (see ``track_ids``'s docstring in
        rollout_buffer.py) -- segmenting per track first guarantees every
        held-out "episode" is a real, single-track episode. Note a held-out
        episode's rows are NOT necessarily a contiguous flat-index range --
        two tracks interleave row-by-row (see track_ids's docstring), so an
        episode's own rows can have other tracks' rows sitting in between
        them; the exact index LIST for each episode is tracked and masked
        explicitly below rather than via a ``[start:end+1]`` slice, which
        would wrongly sweep in whatever other track's rows happen to fall
        inside that span.

        Returns ``(train_batch, val_batch)`` -- ``val_batch`` is ``None``
        when there are fewer than 2 complete episodes in this rollout (val
        split skipped, everything trains, matching every other
        opt-in-with-a-minimum-episode-count split in this file).
        """
        track_ids = batch.get("track_ids") or ["trainee"] * len(batch["dones"])
        dones_np = batch["dones"].numpy()
        n_total = len(dones_np)

        groups: dict[str, list[int]] = {}
        for i, t in enumerate(track_ids):
            groups.setdefault(t, []).append(i)

        episodes: list[list[int]] = []  # each entry: that episode's own flat indices
        for _track, idxs in groups.items():
            ep_start_pos = 0
            for pos, flat_i in enumerate(idxs):
                if dones_np[flat_i] > 0.5:
                    episodes.append(idxs[ep_start_pos:pos + 1])
                    ep_start_pos = pos + 1
            # Trailing partial episode (no terminal done yet) dropped --
            # same convention as every other episode-boundary scan here.

        n_complete = len(episodes)
        n_val_eps = max(1, round(self.val_episode_fraction * n_complete)) if n_complete >= 2 else 0
        if n_val_eps == 0:
            return batch, None

        val_episodes = self._val_split_rng.sample(episodes, n_val_eps)
        val_mask = np.zeros(n_total, dtype=bool)
        for ep_idxs in val_episodes:
            val_mask[ep_idxs] = True
        train_mask = ~val_mask

        _LIST_KEYS = {"reward_comps_raw", "step_outcomes", "track_ids"}

        def _sel(mask: np.ndarray) -> dict:
            idx = np.where(mask)[0]
            idx_list = idx.tolist()
            idx_t = torch.from_numpy(idx).long()
            return {
                k: ([v[i] for i in idx_list] if k in _LIST_KEYS else v[idx_t])
                for k, v in batch.items()
            }

        train_batch = _sel(train_mask)
        val_batch = _sel(val_mask)
        log.info(
            f"  [val split] {n_complete - n_val_eps} train episodes ({int(train_mask.sum())} steps)"
            f"  |  {n_val_eps} val episodes ({int(val_mask.sum())} steps, held out from training)"
        )
        return train_batch, val_batch

    def _loss_ratio(self, log_ratio: torch.Tensor) -> torch.Tensor:
        """PPO importance ratio for a loss term. With dual-clip on, the
        log-ratio is capped at _MAX_LOG_RATIO so an overflowing ratio can't turn
        the gradient into NaN. Off, it is plain exp(): vanilla PPO stays
        byte-identical, overflow included (the NaN guard then halts the run)."""
        if self.dual_clip_c <= 0:
            return torch.exp(log_ratio)
        return torch.exp(log_ratio.clamp(max=_MAX_LOG_RATIO))

    def _apply_dual_clip(self, min_surr: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
        """Dual-clip PPO's negative-advantage floor (see __init__'s
        dual_clip_c comment for the full rationale): ``max(min_surr,
        dual_clip_c * adv)`` wherever ``adv < 0``, unchanged everywhere
        else. ``self.dual_clip_c <= 0`` (default) is a true no-op -- returns
        ``min_surr`` completely unmodified, so every call site can call this
        unconditionally without its own gating and stays byte-identical to
        vanilla PPO when the feature is off.

        ``adv`` may be a lower-rank/broadcastable view of ``min_surr``
        (e.g. the per-head breakdown's ``(batch, 1)`` advantage against a
        ``(batch, 15)`` min_surr) -- ``torch.where``/multiplication both
        broadcast normally.
        """
        if self.dual_clip_c <= 0:
            return min_surr
        floor = self.dual_clip_c * adv
        return torch.where(adv < 0, torch.max(min_surr, floor), min_surr)

    def _eval_val_episode_losses(
        self, val_batch: dict, clip: float,
    ) -> tuple[float, float, float, torch.Tensor, float, float, float, float, float, float, float, Optional[torch.Tensor]]:
        """No-grad policy/value loss + approx KL on held-out episodes, under
        CURRENT weights.

        Same PPO-clipped-surrogate, normalised-MSE, and approximate-KL
        (``mean(old_log_prob - new_log_prob)``, same "k1" estimator and
        same ``clamp(min=-1e6)`` floor as ``approx_kl``/``kl_after_step`` in
        the main training minibatch loop, see ``_ppo_update``) formulas as
        training itself, just under ``torch.no_grad()`` and never followed
        by an optimizer step -- called once per epoch so the reported
        numbers track how the *policy being trained this call* moves on
        episodes it never saw, epoch by epoch. Advantages are normalised
        using THIS val batch's own mean/std (val rows never participate in
        the train batch's normalisation stats).

        Returns ``(policy_loss, value_loss, kl, per_sample_policy_loss,
        value_loss_p10_p90, value_loss_p25_p75, policy_loss_p10_p90,
        policy_loss_p25_p75, surrogate_mean, surrogate_p10_p90,
        surrogate_p25_p75, per_head_policy_loss)``.
        ``per_sample_policy_loss`` is every row's own (pre-mean) clipped-
        surrogate value, concatenated across minibatches in val_batch's own
        row order -- lets a caller look at the DISTRIBUTION (percentiles,
        outliers), not just the mean, the same way _ppo_update's per-epoch
        percentile line does. ``value_loss_p10_p90``/``policy_loss_p10_p90``
        (and their ``_p25_p75`` companions) are ``value_loss``/``policy_loss``
        recomputed as a TRIMMED mean (see ``_trimmed_mean_p10_p90``/
        ``_trimmed_mean_p25_p75``) over per-sample terms -- p10_p90 drops the
        most extreme 10% on each end, p25_p75 is the more aggressive,
        symmetric interquartile mean (drops 25% each end) -- logged
        alongside each other so a caller can see whether a mean is being
        dragged by its tails generally (the two read far apart) or is
        fairly stable regardless of trim aggressiveness (they read close
        together). Both are robust companions to the plain mean, which a
        single outlier row can otherwise dominate. See __init__'s
        val_episode_fraction comment / _ppo_update's log line for why
        ``policy_loss``/``kl`` here are NOT meaningful generalization
        diagnostics the way ``value_loss`` is: both collapse toward 0 for
        ANY row that received zero gradient steps (ratio~1), regardless of
        train/val identity -- ``policy_loss_p10_p90``/``policy_loss_p25_p75``
        inherit that same caveat (robust versions of a not-very-meaningful
        number, not a robust generalization diagnostic on their own).

        ``surrogate_mean``/``surrogate_p10_p90``/``surrogate_p25_p75`` are
        the mean/trimmed-means of the UNCLIPPED, no-dual-clip linearised
        surrogate ``(ratio - 1) * A`` per row -- mathematically
        ``policy_loss == -surrogate_mean`` exactly whenever the clip and
        dual-clip floor are both inactive for a row (``mean(A)~=0`` by
        normalisation), so this is NOT a different signal in the common
        case -- it's the same quantity with the clip/dual-clip machinery
        stripped out, so a row whose ratio actually drifted far enough to
        hit the clip band shows its true magnitude here instead of being
        floored. See "[held-out policy surrogate]"'s log-line comment in
        _ppo_update for the fuller rationale.

        ``per_head_policy_loss`` (shape ``(15,)``, ``HEAD_LP_KEYS`` order,
        ``None`` when ``val_batch`` lacks ``head_log_probs``) is the SAME
        per-head COUNTERFACTUAL breakdown as the train-side "[per-head
        policy_loss (counterfactual, ...)]" line (each head's own ratio in
        isolation, others held at ratio=1, through the same clipped-
        surrogate + dual-clip formula) -- a counterfactual, not a strict
        decomposition (sum of the 15 values != policy_loss above, since the
        real ratio is the PRODUCT of every head's own ratio).
        """
        n_val = len(val_batch["log_probs"])
        val_adv = val_batch["advantages"]
        val_adv = (val_adv - val_adv.mean()) / (val_adv.std() + 1e-8)
        val_old_lp = val_batch["log_probs"]
        _val_opp_mask = self._opp_head_mask_matrix(val_batch)
        if _val_opp_mask is not None:
            val_old_lp = self._mask_total_log_probs(val_old_lp, val_batch["head_log_probs"], _val_opp_mask)
        val_returns = val_batch["returns"]
        val_ret_var = val_returns.var().clamp(min=1.0)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl = 0.0
        total_n = 0
        _per_sample_chunks: list[torch.Tensor] = []
        _per_sample_value_chunks: list[torch.Tensor] = []
        # Unclipped, no-dual-clip linearised surrogate (r-1)*A for each row --
        # see "[held-out policy surrogate]"'s own log-line comment below for
        # why this is a deliberately clip-invariant companion to
        # val_policy_loss, not a replacement for it.
        _per_sample_surrogate_chunks: list[torch.Tensor] = []
        _head_policy_loss_sum: Optional[torch.Tensor] = None  # running (bs-weighted) sum, (15,)
        with torch.no_grad():
            for start in range(0, n_val, self.minibatch_size):
                idx = torch.arange(start, min(start + self.minibatch_size, n_val))
                mb_obs = {k.replace("obs/", ""): val_batch[k][idx].to(self.device)
                          for k in val_batch if k.startswith("obs/")}
                sf, of, em = mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"]
                bf, gf = mb_obs["ball_feat"], mb_obs["global_feat"]
                sat, oat = _ai_types(mb_obs)

                d_heads = self.decision_net(sf, of, em, bf, gf, sat, oat)
                e_heads = self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat)
                if self.separate_value_net:
                    d_heads_for_value = _detach_decision_heads(d_heads)
                    new_values = self.value_net(
                        sf, of, em, bf, gf, d_heads_for_value, sat, oat, value_only=True
                    ).squeeze(-1)
                else:
                    new_values = e_heads.value.squeeze(-1)

                mb_actions = {k.replace("action/", ""): val_batch[k][idx].to(self.device)
                              for k in val_batch if k.startswith("action/")}
                new_log_probs = self._recompute_log_prob(d_heads, e_heads, mb_actions, em)
                if _val_opp_mask is not None:
                    new_log_probs = self._mask_total_log_probs(
                        new_log_probs, self._per_head_new_log_probs(d_heads, e_heads, mb_actions, em),
                        _val_opp_mask[idx].to(self.device),
                    )

                mb_adv = val_adv[idx].to(self.device)
                mb_ret = val_returns[idx].to(self.device)
                mb_old_lp = val_old_lp[idx].to(self.device)
                ratio = self._loss_ratio(new_log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * mb_adv
                _mb_min_surr = self._apply_dual_clip(torch.min(surr1, surr2), mb_adv)
                _mb_per_sample_policy_loss = -_mb_min_surr
                mb_policy_loss = _mb_per_sample_policy_loss.mean()
                # (r-1)*A == surr1 - mb_adv -- reuses surr1 rather than
                # recomputing ratio*mb_adv, cheap either way.
                _per_sample_surrogate_chunks.append((surr1 - mb_adv).cpu())
                _mb_per_sample_value_loss = (new_values - mb_ret).pow(2) / val_ret_var
                mb_value_loss = _mb_per_sample_value_loss.mean()
                mb_kl = (mb_old_lp - new_log_probs.clamp(min=-1e6)).mean()
                _per_sample_chunks.append(_mb_per_sample_policy_loss.cpu())
                _per_sample_value_chunks.append(_mb_per_sample_value_loss.cpu())

                bs = len(idx)
                total_policy_loss += float(mb_policy_loss) * bs
                total_value_loss += float(mb_value_loss) * bs
                total_kl += float(mb_kl) * bs
                total_n += bs

                # Per-head counterfactual policy_loss breakdown -- same
                # formula/masking as the train-side all_head_policy_loss
                # computation (see its own comment in the main minibatch
                # loop below), just evaluated here under no_grad on the
                # held-out set instead.
                if "head_log_probs" in val_batch:
                    mb_old_head_lp = val_batch["head_log_probs"][idx].to(self.device)
                    mb_new_head_lp = self._per_head_new_log_probs(d_heads, e_heads, mb_actions, em)
                    head_ratio = self._loss_ratio(mb_new_head_lp - mb_old_head_lp)
                    head_adv = mb_adv.unsqueeze(-1)
                    head_surr1 = head_ratio * head_adv
                    head_surr2 = torch.clamp(head_ratio, 1.0 - clip, 1.0 + clip) * head_adv
                    head_min_surr = self._apply_dual_clip(torch.min(head_surr1, head_surr2), head_adv)
                    head_policy_loss_mb = -head_min_surr.mean(dim=0).cpu()  # (15,)
                    _head_policy_loss_sum = (
                        head_policy_loss_mb * bs if _head_policy_loss_sum is None
                        else _head_policy_loss_sum + head_policy_loss_mb * bs
                    )

        _per_sample_pol_cat = torch.cat(_per_sample_chunks) if _per_sample_chunks else torch.zeros(0)
        _per_sample_value_cat = (
            torch.cat(_per_sample_value_chunks) if _per_sample_value_chunks else torch.zeros(0)
        )
        _surrogate_cat = (
            torch.cat(_per_sample_surrogate_chunks) if _per_sample_surrogate_chunks else torch.zeros(0)
        )
        return (
            total_policy_loss / max(total_n, 1),
            total_value_loss / max(total_n, 1),
            total_kl / max(total_n, 1),
            _per_sample_pol_cat,
            _trimmed_mean_p10_p90(_per_sample_value_cat) if _per_sample_value_chunks else float("nan"),
            _trimmed_mean_p25_p75(_per_sample_value_cat) if _per_sample_value_chunks else float("nan"),
            _trimmed_mean_p10_p90(_per_sample_pol_cat) if _per_sample_chunks else float("nan"),
            _trimmed_mean_p25_p75(_per_sample_pol_cat) if _per_sample_chunks else float("nan"),
            float(_surrogate_cat.mean()) if _per_sample_surrogate_chunks else float("nan"),
            _trimmed_mean_p10_p90(_surrogate_cat) if _per_sample_surrogate_chunks else float("nan"),
            _trimmed_mean_p25_p75(_surrogate_cat) if _per_sample_surrogate_chunks else float("nan"),
            _head_policy_loss_sum / max(total_n, 1) if _head_policy_loss_sum is not None else None,
        )

    # -----------------------------------------------------------------------
    # PPO update
    # -----------------------------------------------------------------------

    def _recompute_old_log_probs_for_augmented_batch(
        self, batch: dict, pair_out: Optional[dict] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Properly recompute old_log_prob (+ per-head old_log_prob) for
        EVERY row of an augmented batch, replacing ``augment_batch()``'s
        cheap "tile the original row's log_prob onto every augmented copy"
        shortcut.

        Only valid to call here, BEFORE this update's first gradient step:
        the current weights are still exactly the rollout-collection policy
        (theta_old), so a forward pass right now gives the TRUE
        ``log pi_old(action | obs)`` for every row -- flip_y-augmented
        copies included, not just the identity copies (which already had
        this correct by construction, since nothing about them changed).

        Why this replaces the tiling instead of leaving it: the borrowed/
        tiled log_prob for a flip_y copy is generically WRONG unless the
        network already happens to be exactly flip-equivariant at that
        specific (obs, action) pair, and the resulting ratio-scaling error
        is a real, uncontrolled source of gradient-MAGNITUDE noise -- e.g.
        confirmed via the epoch-0 baseline's own "[policy percentiles]"
        line reading p100 in the 20s on the augmented train batch vs a
        same-rollout, never-augmented held-out batch's single digits, both
        measured at ratio=1 (no gradient step yet). NOT a deliberate "push
        toward equivariance" mechanism: the gradient DIRECTION contributed
        by an augmented row is ``d(new_log_prob)/dtheta``, identical
        regardless of which old_log_prob constant is subtracted from it
        (that constant doesn't depend on theta either way) -- only the
        magnitude (via the ratio it's exponentiated into) differs. Whatever
        genuinely teaches the network the y-flip symmetry comes from
        training on the correctly-flipped (obs, action, advantage) sample
        at all, with its own correctly-signed advantage -- not from which
        log_prob happens to sit in the ratio's denominator. See
        ai/knowledge.md "Augmentation log_prob approximation" for the full
        writeup.

        Chunked by minibatch_size under torch.no_grad(), mirroring
        pre_update_value_loss's own pass in _ppo_update (including passing
        through the physics-encoder full features, if present, the same
        way) -- call this AFTER that block populates
        obs/ball_physics_full etc. on ``batch``, not before.

        Returns ``(log_probs, head_log_probs)`` -- ``head_log_probs`` is
        ``None`` when ``"head_log_probs"`` isn't present in ``batch``
        (matches ``RolloutBuffer.add()``'s own optionality for that field).

        ``pair_out``: when a dict is passed, it is filled with ``"id"`` and ``"flip"`` -- CPU
        head-output snapshots (``_snapshot_row`` layout) of the identity block and the flip_y block
        (both with identity slot order) that this same pass already computed, for
        ``_flip_consistency_stats``. Frozen categorical target logits are replaced by a placeholder.
        """
        n = len(batch["log_probs"])
        has_head_lp = "head_log_probs" in batch
        lp_parts: list[torch.Tensor] = []
        head_lp_parts: list[torch.Tensor] = []
        _k = max(1, self.augment_n_slot_shuffles)
        _n_base = n // (2 * _k)
        _blocks = {"id": (0, _n_base), "flip": (_k * _n_base, _k * _n_base + _n_base)}
        _pair_parts: dict[str, list[dict]] = {"id": [], "flip": []}
        _inactive = self._inactive_head_lp_keys() if pair_out is not None else frozenset()
        with torch.no_grad():
            for start in range(0, n, self.minibatch_size):
                idx = torch.arange(start, min(start + self.minibatch_size, n))
                mb_obs = {k.replace("obs/", ""): batch[k][idx].to(self.device)
                          for k in batch if k.startswith("obs/")}
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
                mb_actions = {k.replace("action/", ""): batch[k][idx].to(self.device)
                              for k in batch if k.startswith("action/")}
                lp_parts.append(self._recompute_log_prob(d_heads, e_heads, mb_actions, em).cpu())
                if has_head_lp:
                    head_lp_parts.append(
                        self._per_head_new_log_probs(d_heads, e_heads, mb_actions, em).cpu()
                    )
                if pair_out is not None:
                    _row = None
                    for _name, (_lo, _hi) in _blocks.items():
                        _a, _b = max(start, _lo), min(start + len(idx), _hi)
                        if _a >= _b:
                            continue
                        if _row is None:
                            _row = self._snapshot_row(d_heads, e_heads)
                            for _short, _key in (("pass_", "pass_target_logits"), ("tackle", "tackle_target_logits"),
                                                 ("mark", "mark_target_logits")):
                                if _short in _inactive:
                                    _row[_key] = torch.zeros(len(idx), 1, device=self.device)
                        _pair_parts[_name].append(
                            {k: v[_a - start:_b - start].detach().cpu() for k, v in _row.items()}
                        )
        if pair_out is not None:
            for _name, _parts in _pair_parts.items():
                pair_out[_name] = {k: torch.cat([p[k] for p in _parts], dim=0) for k in _parts[0]} if _parts else {}
        log_probs = torch.cat(lp_parts, dim=0)
        head_log_probs = torch.cat(head_lp_parts, dim=0) if has_head_lp else None
        return log_probs, head_log_probs

    def _opp_head_mask_matrix(self, batch: dict) -> Optional[torch.Tensor]:
        """(n, len(HEAD_LP_KEYS)) float mask (1 = that head trains on that row) or None when
        ``ppo.mask_untriggered_actions`` is off. Fails loudly if masking is on but the rollout carries no flags."""
        if not self.mask_untriggered_actions:
            return None
        if not has_opportunity_flags(batch):
            raise ValueError(
                "ppo.mask_untriggered_actions is on but the rollout batch has no action/opp_* flags -- the "
                "rollout producer did not attach them (ScenarioEnv._with_opportunity_flags)."
            )
        _decision_parents = {"pass_logit", "tackle_logit", "mark_logit"}
        if not _decision_parents <= set(self._ppo_lp_masked_heads):
            raise NotImplementedError(
                "ppo.mask_untriggered_actions is only implemented for curriculum phases whose pass/tackle/mark "
                "decision heads are frozen (their target-categorical log-prob terms are outside the per-head "
                "log-prob columns the mask is applied to)."
            )
        masks = opportunity_masks(batch)
        m = torch.ones(len(batch["log_probs"]), len(HEAD_LP_KEYS))
        for head, key in self._OPP_HEAD_MASK_KEY.items():
            m[:, HEAD_LP_KEYS.index(head)] = masks[key].reshape(-1)
        return m

    def _opp_head_grad_scale(self, mask_matrix: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Per-head gradient multiplier 1/mean(mask) (plan D9, ``ppo.mask_rescale_by_opportunity_rate``),
        capped at 100; None when rescaling (or masking) is off. Value-preserving: see _mask_total_log_probs."""
        if mask_matrix is None or not self.mask_rescale_by_opportunity_rate:
            return None
        return (1.0 / mask_matrix.mean(dim=0).clamp(min=0.01)).to(self.device)

    @staticmethod
    def _mask_total_log_probs(
        total: torch.Tensor, cols: torch.Tensor, mask: torch.Tensor, grad_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Total log-prob with the masked-out heads' terms removed: ``sum_h mask_h * col_h`` plus whatever the
        total holds beyond the per-head columns (0 in the supported phases; detached when ``total`` carries a
        graph so masked heads receive EXACTLY zero gradient). ``cols`` = per-head log-probs (n, 15) of the same
        rows (``head_log_probs`` for the old policy, ``_per_head_new_log_probs`` for the current one).
        ``grad_scale`` multiplies each head's gradient without changing any value."""
        residual = total - cols.sum(dim=-1)
        if total.requires_grad:
            residual = residual.detach()
        c = cols
        if grad_scale is not None:
            c = cols.detach() + grad_scale * (cols - cols.detach())
        return (c * mask).sum(dim=-1) + residual

    def _log_action_opportunity(self, batch: dict) -> None:
        """[action opp] block (Phase 1 of agent_plans/masked_action_training_plan.md): how often a kick / armed
        tackle could actually take effect, and how the gates behave inside vs outside those rows."""
        st = opportunity_stats(batch)
        if st is None:
            log.warning("[action opp] enabled but the batch carries no action/opp_* flags (rollout producer did not attach them)")
            return
        for line in format_opportunity_block(st):
            log.info(line)

    def _ppo_update(self, batch: dict, progress: float) -> dict:
        """Run N epochs of minibatch PPO updates over the collected rollout.

        Returns dict of mean loss metrics for logging.

        Stages, in order: prepare the batch (held-out split, augmentation,
        cached physics features, exact old log-probs); fix the update's
        schedules, advantages and masks; measure the pre-update value loss;
        run the epochs, stopping early once the KL passes target_kl; after an
        early stop, the value-only continuation; the BC-only continuation;
        then the logging-only diagnostics and the returned metrics.
        """
        batch, val_batch, flip_consistency = self._ppo_prepare_batch(batch)
        ctx = self._ppo_update_context(batch, val_batch, progress)
        stats = _UpdateStats()
        if log.isEnabledFor(logging.DEBUG):
            self._ppo_log_rollout_debug(ctx, progress)
        # Direction-head spread at the very start of the update, so the summary
        # can report start -> end drift across the whole update.
        spread_start = self._direction_spread_means()
        pre_update_value_loss = self._ppo_pre_update_value_loss(ctx)
        if self.log_epoch_zero_baseline:
            self._ppo_log_epoch_zero_baseline(ctx)
        if self._ppo_run_epochs(ctx, stats):
            self._ppo_value_only_continuation(ctx, stats)
        self._ppo_bc_only_continuation(ctx, stats)
        self._ppo_log_kl_diagnostics(ctx, stats)
        self._ppo_log_update_summary(ctx, stats, spread_start)
        return self._ppo_update_metrics(ctx, stats, pre_update_value_loss, flip_consistency)

    # ----- setup -------------------------------------------------------------

    def _ppo_prepare_batch(self, batch: dict) -> tuple[dict, Optional[dict], Optional[dict]]:
        """Returns (train batch, held-out batch or None, flip-consistency stats or None).

        1. Hold out a random subset of complete episodes BEFORE augmentation, so
           held-out rows are never expanded or flipped and never enter a
           minibatch (see __init__'s val_episode_fraction comment).
        2. Augment with geometric flips and slot permutations (expands the batch
           by 2 x n_slot_shuffles; see augment.py).
        3. Cache the frozen physics encoders' output once as obs/*_physics_full
           entries. It depends only on the (now fixed) inputs, not on trainable
           weights, so it is identical across every pass over this batch;
           computed in canonical frame and chunked by minibatch_size.
        4. After augmentation, replace augment_batch()'s tiled old log-probs
           (exact only for the identity copies) with the true log pi_old of every
           row (see _recompute_old_log_probs_for_augmented_batch). Needs step 3.
        """
        val_batch: Optional[dict] = None
        if self.val_episode_fraction > 0.0:
            batch, val_batch = self._split_train_val_episodes(batch)
        if self.augment_n_slot_shuffles > 0:
            batch = augment_batch(batch, self.augment_n_slot_shuffles, self._aug_rng)
        if self.log_action_opportunity_stats:
            self._log_action_opportunity(batch)
        batch.update(self._precompute_physics_full(batch, self.minibatch_size))
        flip_consistency = None
        if self.augment_n_slot_shuffles > 0:
            _flip_pair: Optional[dict] = {} if self._log_flip_consistency else None
            batch["log_probs"], _recomputed_head_lp = self._recompute_old_log_probs_for_augmented_batch(
                batch, pair_out=_flip_pair,
            )
            if _recomputed_head_lp is not None:
                batch["head_log_probs"] = _recomputed_head_lp
            if _flip_pair is not None:
                flip_consistency = self._flip_consistency_stats(batch, _flip_pair)
        return batch, val_batch, flip_consistency

    def _ppo_update_context(self, batch: dict, val_batch: Optional[dict], progress: float) -> _UpdateContext:
        """Apply this update's scheduled learning rates and fix its coefficients,
        normalised advantages and action-opportunity masks.

        With ppo.mask_untriggered_actions, heads that could not take effect on a
        row carry no policy gradient there: the ratio is built from log-probs with
        those heads' terms removed on BOTH sides. The unmasked old log-probs stay
        for the whole-policy diagnostics."""
        clip = self.schedules.clip(progress)
        lr = self.schedules.lr(progress)
        value_lr = self.schedules.value_lr(progress)
        bc_coeff = self.schedules.bc(progress)
        bc_kick_coeff = self.schedules.bc_kick(progress)
        bc_tackle_coeff = self.schedules.bc_tackle(progress)
        ent_coef = self.schedules.ent(progress)
        for pg in self.optimizer.param_groups:
            pg["lr"] = value_lr if pg.get("name") == "value" else lr
        has_bc = "bc_labels" in batch and (bc_coeff > 0.0 or bc_kick_coeff > 0.0 or bc_tackle_coeff > 0.0)
        from footballcoach.ai.ppo import bc_sources as _bcs
        bc_coeff_table = None
        if _bcs.sources() and "bc_sources" in batch:
            bc_coeff_table = _bcs.coeff_table(progress, (bc_coeff, bc_kick_coeff, bc_tackle_coeff))
            _used = torch.unique(batch["bc_sources"])
            has_bc = "bc_labels" in batch and bool((bc_coeff_table[_used] > 0.0).any())
        adv = batch["advantages"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        old_log_probs = batch["log_probs"]
        opp_mask = self._opp_head_mask_matrix(batch)
        opp_grad_scale = self._opp_head_grad_scale(opp_mask)
        old_log_probs_ratio = (
            old_log_probs if opp_mask is None
            else self._mask_total_log_probs(old_log_probs, batch["head_log_probs"], opp_mask)
        )
        return _UpdateContext(
            batch=batch, val_batch=val_batch, n=len(batch["log_probs"]), clip=clip, ent_coef=ent_coef,
            bc_coeff=bc_coeff, bc_kick_coeff=bc_kick_coeff, bc_tackle_coeff=bc_tackle_coeff, has_bc=has_bc,
            adv=adv, old_log_probs=old_log_probs, old_log_probs_ratio=old_log_probs_ratio,
            returns=batch["returns"], opp_mask=opp_mask, opp_grad_scale=opp_grad_scale,
            warm_base_lrs=[pg["lr"] for pg in self.optimizer.param_groups],
            step_outcomes=batch.get("step_outcomes", []),
            bc_coeff_table=bc_coeff_table,
        )

    def _direction_spread_means(self) -> tuple[float, float, float]:
        """Mean move_dir_log_kappa, kick_dir_log_kappa and kick_dir_z_log_std."""
        net = self.execution_net
        return (float(net.move_dir_log_kappa.mean().item()),
                float(net.kick_dir_log_kappa.mean().item()),
                float(net.kick_dir_z_log_std.mean().item()))

    # ----- forward helpers ---------------------------------------------------

    def _minibatch_obs(self, batch: dict, idx: torch.Tensor) -> dict:
        """The batch's obs/* entries for rows idx, on the device, without the prefix."""
        return {k.replace("obs/", ""): batch[k][idx].to(self.device) for k in batch if k.startswith("obs/")}

    def _minibatch_actions(self, batch: dict, idx: torch.Tensor) -> dict:
        """The batch's action/* entries for rows idx, on the device, without the prefix."""
        return {k.replace("action/", ""): batch[k][idx].to(self.device) for k in batch if k.startswith("action/")}

    def _ppo_decision_forward(self, mb_obs: dict, parts: tuple):
        """decision_net on a minibatch, passing the cached physics features through."""
        sf, of, em, bf, gf, sat, oat = parts
        return self.decision_net(
            sf, of, em, bf, gf, sat, oat,
            ball_physics_full=mb_obs.get("ball_physics_full"),
            self_physics_full=mb_obs.get("self_physics_full"),
            other_physics_full=mb_obs.get("other_physics_full"),
        )

    def _value_only_forward(self, parts: tuple, d_heads) -> torch.Tensor:
        """Value-only pass: the dedicated value net when separate_value_net, else
        execution_net's own value head. Returns (rows, 1)."""
        sf, of, em, bf, gf, sat, oat = parts
        if self.separate_value_net:
            return self.value_net(sf, of, em, bf, gf, d_heads, sat, oat, value_only=True)
        return self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat, value_only=True)

    def _ppo_value_prediction(self, parts: tuple, d_heads, e_heads) -> torch.Tensor:
        """The training step's value estimate, (rows,).

        Without separate_value_net it is execution_net's value head from the
        forward pass already made. With it, the dedicated value net reads the
        decision heads DETACHED, so the value loss never reaches decision_net:
        the point of a separate critic is a trunk independent of the BC-primed
        policy trunk. share_value_grad_with_decision opts back into a controlled
        leak, scaled by decision_value_coef (decision_net only, never
        execution_net; see _scale_decision_heads_grad)."""
        if not self.separate_value_net:
            return e_heads.value.squeeze(-1)
        if self.share_value_grad_with_decision:
            d_heads_for_value = _scale_decision_heads_grad(d_heads, self.decision_value_coef)
        else:
            d_heads_for_value = _detach_decision_heads(d_heads)
        return self._value_only_forward(parts, d_heads_for_value).squeeze(-1)

    def _ppo_new_log_probs(self, d_heads, e_heads, mb_actions: dict, em: torch.Tensor,
                           mb_opp_mask: Optional[torch.Tensor], opp_grad_scale: Optional[torch.Tensor]) -> torch.Tensor:
        """Current log-probs of the stored actions, opportunity-masked when enabled."""
        new_log_probs = self._recompute_log_prob(d_heads, e_heads, mb_actions, em)
        if mb_opp_mask is not None:
            new_log_probs = self._mask_total_log_probs(
                new_log_probs, self._per_head_new_log_probs(d_heads, e_heads, mb_actions, em),
                mb_opp_mask, opp_grad_scale,
            )
        return new_log_probs

    # ----- losses ------------------------------------------------------------

    def _ppo_policy_loss(self, new_log_probs: torch.Tensor, old_log_probs: torch.Tensor, adv: torch.Tensor,
                         weights: torch.Tensor, clip: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Clipped (and, when enabled, dual-clipped) PPO surrogate.

        Returns (policy_loss, per-sample loss, per-sample (r - 1) * A). The last is
        unclipped and without dual-clip, weighted the same way: a clip-invariant
        companion for the logs. The loss uses the capped ratio (_loss_ratio)."""
        loss_ratio = self._loss_ratio(new_log_probs - old_log_probs)
        surr1 = loss_ratio * adv
        surr2 = torch.clamp(loss_ratio, 1.0 - clip, 1.0 + clip) * adv
        min_surr = self._apply_dual_clip(torch.min(surr1, surr2), adv)
        per_sample = -(min_surr * weights)
        return per_sample.mean(), per_sample, (surr1 - adv) * weights

    def _ppo_head_counterfactual_policy_loss(
        self, new_head_lp: torch.Tensor, old_head_lp: torch.Tensor, adv: torch.Tensor, weights: torch.Tensor,
        clip: float, mb_opp_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """(heads,) clipped surrogate loss if ONLY that head's probability had moved
        (every other head held at ratio 1).

        The real ratio is the product of every head's ratio, so these do not sum to
        the policy loss: they show which heads drive the objective and in which
        direction. Dual-clip is applied as in the real loss. With an opportunity
        mask, each head averages over the rows where it could take effect."""
        head_ratio = self._loss_ratio(new_head_lp - old_head_lp)
        head_adv = adv.unsqueeze(-1)
        head_w = weights.unsqueeze(-1)
        surr1 = head_ratio * head_adv
        surr2 = torch.clamp(head_ratio, 1.0 - clip, 1.0 + clip) * head_adv
        min_surr = self._apply_dual_clip(torch.min(surr1, surr2), head_adv)
        if mb_opp_mask is None:
            return -(min_surr * head_w).mean(dim=0)
        return -(min_surr * head_w * mb_opp_mask).sum(dim=0) / mb_opp_mask.sum(dim=0).clamp(min=1.0)

    @staticmethod
    def _opp_entropy_masks(mb_opp_mask: Optional[torch.Tensor],
                           opp_grad_scale: Optional[torch.Tensor]) -> Optional[dict]:
        """The kick/tackle_attempt opportunity masks (and gradient scales) for
        _compute_entropy, or None when masking is off."""
        if mb_opp_mask is None:
            return None
        _ik, _it = HEAD_LP_KEYS.index("kick"), HEAD_LP_KEYS.index("tackle_attempt")
        masks = {"kick": mb_opp_mask[:, _ik], "tackle_attempt": mb_opp_mask[:, _it]}
        if opp_grad_scale is not None:
            masks["kick_scale"] = float(opp_grad_scale[_ik])
            masks["tackle_attempt_scale"] = float(opp_grad_scale[_it])
        return masks

    def _dir_spread_reg(self) -> torch.Tensor:
        """L2 pull of move/kick direction log_kappa and kick_dir_z_log_std toward
        their targets, shape (1,); each term off when its coefficient is 0.

        Without it the entropy bonus is a one-directional force: it only deflates
        the log_kappas (entropy falls as kappa grows) and inflates the z log_std,
        and nothing in the PPO/BC losses pulls them back (the direction mean is
        unit-normalised). Independent of the clamp, which only caps the value fed
        to the distribution."""
        net = self.execution_net
        reg = torch.zeros(1, device=self.device)
        if self.move_dir_log_kappa_reg_coef > 0.0:
            reg = reg + self.move_dir_log_kappa_reg_coef * ((net.move_dir_log_kappa - self.move_dir_log_kappa_target) ** 2).mean()
        if self.kick_dir_log_kappa_reg_coef > 0.0:
            reg = reg + self.kick_dir_log_kappa_reg_coef * ((net.kick_dir_log_kappa - self.kick_dir_log_kappa_target) ** 2).mean()
        if self.kick_dir_z_log_std_reg_coef > 0.0:
            reg = reg + self.kick_dir_z_log_std_reg_coef * ((net.kick_dir_z_log_std - self.kick_dir_z_log_std_target) ** 2).mean()
        return reg

    def _bc_loss_kwargs(self) -> dict:
        """bc_loss_from_tensor's weighting arguments, shared by the in-epoch BC
        auxiliary loss and the BC-only continuation."""
        return dict(
            direction_loss_weight=self._bc_dir_loss_w,
            direction_loss_mode=self._bc_dir_loss_mode,
            region_loss_weight=self._bc_region_loss_w,
            pos_weight_kick=self._bc_pos_weight_kick,
            pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
            dec_weight=self._bc_dec_weight,
            exec_weight=self._bc_exec_weight,
            dec_label_smoothing=self._bc_dec_label_smoothing,
            exec_label_smoothing=self._bc_exec_label_smoothing,
        )

    def _ppo_bc_aux_loss(self, ctx: _UpdateContext, stats: _UpdateStats, mb_idx: torch.Tensor,
                         d_heads, e_heads) -> tuple[torch.Tensor, torch.Tensor]:
        """BC auxiliary loss for one minibatch: (unweighted total, for logging;
        weighted term to add to the loss).

        Split into a kick group (kick_this_tick / kick_direction / kick_power /
        kick_spin), a tackle group (tackle_attempt) and everything else, each
        scaled by its own annealed coefficient (TrainingSchedules.bc / bc_kick /
        bc_tackle)."""
        from footballcoach.ai.ppo.bc import bc_loss_from_tensor
        mb_bc = ctx.batch["bc_labels"][mb_idx].to(self.device)
        per_row = ctx.bc_coeff_table is not None
        bc_loss_val, _bkdn, _bsplit = bc_loss_from_tensor(
            mb_bc, self._bc_heads_for_loss(d_heads), e_heads,
            **self._bc_loss_kwargs(),
            return_breakdown=True,
            split_kick=True,
            split_tackle=True,
            return_group_rows=per_row,
        )
        stats.bc_tackle_loss.append(_bkdn.get("tackle_attempt", 0.0))
        if per_row:
            # Each row's (other, kick, tackle) group losses weighted by ITS
            # source's coefficients, averaged over the valid rows -- with every
            # row on source 0 this is the expression below (same means).
            src = ctx.batch["bc_sources"][mb_idx].to(self.device)
            coeffs = ctx.bc_coeff_table.to(self.device)[src]
            valid = _bsplit["valid"]
            rows = _bsplit["rows"]
            weighted = ((coeffs * rows).sum(dim=1) * valid).sum() / valid.sum().clamp(min=1)
            with torch.no_grad():
                for s in torch.unique(src[valid]).tolist():
                    if s:
                        m = valid & (src == s)
                        stats.bc_by_source.setdefault(int(s), []).append(rows[m].mean(dim=0).tolist())
                        stats.bc_agree_by_source.setdefault(int(s), []).append(_kick_agreement(mb_bc, e_heads, m))
            return bc_loss_val, weighted
        weighted = (
            ctx.bc_coeff * _bsplit["other_group_loss"]
            + ctx.bc_kick_coeff * _bsplit["kick_group_loss"]
            + ctx.bc_tackle_coeff * _bsplit["tackle_group_loss"]
        )
        return bc_loss_val, weighted

    @staticmethod
    def _bc_by_source_metrics(ctx: _UpdateContext, stats: _UpdateStats) -> dict:
        """{"phase2_1v2/attacker": {"coeff": [other, kick, tackle], "loss":
        [other, kick, tackle]}} for each per-phase / per-role BC source seen."""
        if ctx.bc_coeff_table is None:
            return {}
        from footballcoach.ai.ppo.bc_sources import describe
        out = {}
        for s, vals in sorted(stats.bc_by_source.items()):
            out[describe(s)] = {"coeff": ctx.bc_coeff_table[s].tolist(),
                                "loss": np.mean(np.asarray(vals), axis=0).tolist()}
            agree = stats.bc_agree_by_source.get(s)
            if agree:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns
                    out[describe(s)]["agree"] = dict(zip(_AGREE_KEYS, np.nanmean(np.asarray(agree), axis=0).tolist()))
        return out

    @staticmethod
    def _bc_pull_metrics(stats: _UpdateStats) -> dict:
        """Mean of _measure_bc_pull's samples over this update ({} if none)."""
        if not stats.bc_pull:
            return {}
        keys = ("bc_main", "rest_main", "cos_main", "bc_dir", "rest_dir", "cos_dir")
        return dict(zip(keys, np.mean(np.asarray(stats.bc_pull), axis=0).tolist()))

    def _measure_bc_pull(self, stats: _UpdateStats, rest_loss: torch.Tensor, bc_aux: torch.Tensor) -> None:
        """How hard the (coefficient-weighted) BC term pulls on the shared
        weights compared with everything else in the loss (PPO surrogate,
        value, entropy, regularisers): the gradient norm of each, and the cosine
        between them, over the non-direction ("main") and direction parameter
        groups the optimizer clips separately. Two extra backward passes on this
        minibatch's graph; measurement only, the step is unchanged."""
        main, direction = self._main_and_direction_params()
        groups = [[p for p in main if p.requires_grad], [p for p in direction if p.requires_grad]]
        params = groups[0] + groups[1]
        if not params:
            return
        g_bc = torch.autograd.grad(bc_aux, params, retain_graph=True, allow_unused=True)
        g_rest = torch.autograd.grad(rest_loss, params, retain_graph=True, allow_unused=True)
        out, i = [], 0
        for grp in groups:
            bb = rr = br = torch.zeros((), device=self.device)
            for _ in grp:
                b, r = g_bc[i], g_rest[i]
                i += 1
                if b is not None:
                    bb = bb + (b * b).sum()
                if r is not None:
                    rr = rr + (r * r).sum()
                if b is not None and r is not None:
                    br = br + (b * r).sum()
            nb, nr = bb.sqrt(), rr.sqrt()
            out += [nb, nr, br / (nb * nr).clamp(min=1e-12)]
        stats.bc_pull.append(torch.stack(out).tolist())

    def _ppo_pre_update_value_loss(self, ctx: _UpdateContext) -> float:
        """Value loss on the full batch with the weights as they are BEFORE this
        update trains on it: a held-out-ish generalisation number, since the
        critic has never trained on these exact transitions. Same MSE / Var(returns)
        formula as the in-training value loss and pretrain_value's val loss, so
        all three compare directly (see ai_trainer_knowledge.md section 8,
        "val=" / "val_pre=")."""
        returns, n = ctx.returns, ctx.n
        _ret_var_full = returns.var().clamp(min=1.0)
        with torch.no_grad():
            _pre_sq_err_sum = 0.0
            _pre_n = 0
            for _start in range(0, n, self.minibatch_size):
                _idx = torch.arange(_start, min(_start + self.minibatch_size, n))
                _pre_obs = self._minibatch_obs(ctx.batch, _idx)
                parts = _obs_parts(_pre_obs)
                _pd_heads = self._ppo_decision_forward(_pre_obs, parts)
                if self.separate_value_net:
                    _pd_heads = _detach_decision_heads(_pd_heads)
                _pred = self._value_only_forward(parts, _pd_heads).squeeze(-1)
                _pret = returns[_idx].to(self.device)
                _pre_sq_err_sum += F.mse_loss(_pred, _pret, reduction="sum").item()
                _pre_n += len(_idx)
            return (_pre_sq_err_sum / max(_pre_n, 1)) / float(_ret_var_full)

    # ----- main loop ---------------------------------------------------------

    def _ppo_run_epochs(self, ctx: _UpdateContext, stats: _UpdateStats) -> bool:
        """Up to n_epochs shuffled passes of minibatch steps, each followed by its
        epoch summary. Returns True when the KL early stop fired."""
        for epoch_i in range(self.n_epochs):
            epoch_start = time.perf_counter()
            epoch_slice_start = len(stats.policy_loss)
            stats.epoch_sample_policy_loss.clear()
            stats.epoch_sample_value_loss.clear()
            stats.epoch_sample_surrogate.clear()
            early_stopped = False
            indices = torch.randperm(ctx.n)
            for start in range(0, ctx.n, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]
                if len(mb_idx) == 0:
                    continue
                if self._ppo_minibatch_step(ctx, stats, epoch_i, start // self.minibatch_size, mb_idx):
                    early_stopped = True
                    break
            stats.epoch_times.append((time.perf_counter() - epoch_start) * 1000)
            self._ppo_log_epoch_summary(ctx, stats, epoch_i, epoch_slice_start)
            if early_stopped:
                return True
        return False

    def _ppo_minibatch_step(self, ctx: _UpdateContext, stats: _UpdateStats, epoch_i: int, mb_i: int,
                            mb_idx: torch.Tensor) -> bool:
        """One PPO gradient step on rows mb_idx. Returns True when the KL measured
        after the step exceeds target_kl (stop the update: this limits drift past
        the trust region to one step rather than the rest of the epochs)."""
        from footballcoach.ai.ppo.bc import direction_magnitude_reg

        mb_obs = self._minibatch_obs(ctx.batch, mb_idx)
        mb_adv = ctx.adv[mb_idx].to(self.device)
        stats.record_advantage(mb_adv)
        mb_ret = ctx.returns[mb_idx].to(self.device)
        mb_old_lp = ctx.old_log_probs_ratio[mb_idx].to(self.device)
        mb_w = _normalised_sample_weights(ctx.batch["sample_weights"][mb_idx].to(self.device))

        parts = _obs_parts(mb_obs)
        sf, of, em, bf, gf, sat, oat = parts
        d_heads = self._ppo_decision_forward(mb_obs, parts)
        e_heads = self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat)
        before = self._exec_drift_snapshot(e_heads)
        new_values = self._ppo_value_prediction(parts, d_heads, e_heads)
        mb_actions = self._minibatch_actions(ctx.batch, mb_idx)
        mb_opp_mask = None if ctx.opp_mask is None else ctx.opp_mask[mb_idx].to(self.device)
        new_log_probs = self._ppo_new_log_probs(d_heads, e_heads, mb_actions, em, mb_opp_mask, ctx.opp_grad_scale)

        if not stats.first_minibatch_done:
            stats.first_minibatch_done = True
            if log.isEnabledFor(logging.DEBUG):
                self._ppo_log_first_minibatch_debug(d_heads, e_heads, mb_actions, new_log_probs, mb_old_lp,
                                                    new_values, mb_ret)

        # Diagnostics keep the raw ratio (inf stays visible); the loss uses the
        # capped one (see _MAX_LOG_RATIO).
        ratio = torch.exp(new_log_probs - mb_old_lp)
        stats.ratios.append(ratio.detach().cpu())
        self._ppo_track_ratio_spike(stats, ratio, d_heads, e_heads, mb_actions, em, mb_idx, mb_old_lp,
                                    new_log_probs, mb_adv, epoch_i, mb_i)
        ratio_clipped_frac_t = ((ratio < 1.0 - ctx.clip) | (ratio > 1.0 + ctx.clip)).float().mean()
        policy_loss, per_sample_policy_loss, per_sample_surrogate = self._ppo_policy_loss(
            new_log_probs, mb_old_lp, mb_adv, mb_w, ctx.clip)
        stats.epoch_sample_policy_loss.append(per_sample_policy_loss.detach().cpu())
        stats.epoch_sample_surrogate.append(per_sample_surrogate.detach().cpu())
        if "head_log_probs" in ctx.batch:
            with torch.no_grad():
                head_policy_loss = self._ppo_head_counterfactual_policy_loss(
                    self._per_head_new_log_probs(d_heads, e_heads, mb_actions, em),
                    ctx.batch["head_log_probs"][mb_idx].to(self.device),
                    mb_adv, mb_w, ctx.clip, mb_opp_mask,
                )
            stats.head_policy_loss.append(head_policy_loss.cpu())

        value_loss, per_sample_value_loss = _ppo_value_loss(new_values, mb_ret, ctx.returns)
        stats.epoch_sample_value_loss.append(per_sample_value_loss.detach().cpu())
        stats.accum_value_by_outcome(new_values, mb_ret, mb_idx, ctx.step_outcomes)

        entropy, _ent_bkdn, _kick_ent_boost = self._compute_entropy(
            d_heads, e_heads, em, return_breakdown=True, return_kick_boost=True,
            opp_masks=self._opp_entropy_masks(mb_opp_mask, ctx.opp_grad_scale))
        for _ek, _ev in _ent_bkdn.items():
            stats.entropy_breakdown.setdefault(_ek, []).append(_ev)

        # No L2 penalty on the direction means: they are unit-normalised in
        # forward(). The magnitude regulariser on the RAW vectors is unconditional
        # (independent of BC): a raw vector collapsing toward 0 amplifies the
        # normalisation's gradient by ~1/||raw|| for PPO's own gradient too, and
        # must not vanish once BC has annealed to 0.
        total_loss = (policy_loss
                      + self.vf_coef * value_loss
                      - ctx.ent_coef * (entropy + _kick_ent_boost)
                      + self._dir_spread_reg()
                      + direction_magnitude_reg(e_heads, self._bc_dir_mag_reg_coef))
        bc_loss_val = torch.zeros(1, device=self.device)
        if ctx.has_bc:
            bc_loss_val, bc_aux = self._ppo_bc_aux_loss(ctx, stats, mb_idx, d_heads, e_heads)
            if mb_i == 0:
                self._measure_bc_pull(stats, total_loss, bc_aux)
            total_loss = total_loss + bc_aux
        with torch.no_grad():
            tackle_prob_t = torch.sigmoid(e_heads.tackle_attempt_logit).mean()
            kick_prob_t = torch.sigmoid(e_heads.kick_logit).mean()
        # One device sync for all four.
        _ratio_clip_f, _tackle_prob_f, _kick_prob_f, _bc_loss_f = torch.stack([
            ratio_clipped_frac_t, tackle_prob_t, kick_prob_t, bc_loss_val.detach().squeeze(),
        ]).tolist()
        stats.ratio_clipped_frac.append(_ratio_clip_f)
        stats.tackle_prob.append(_tackle_prob_f)
        stats.kick_prob.append(_kick_prob_f)
        stats.bc_loss.append(_bc_loss_f)

        # TEMPORARY diagnostic (see _dump_nan_diagnostic docstring).
        if not torch.isfinite(total_loss):
            self._dump_nan_diagnostic(
                "forward", epoch_i, mb_i, mb_idx, [ctx.batch["track_ids"][i] for i in mb_idx.tolist()],
                d_heads, e_heads, policy_loss, value_loss, entropy,
            )
        raw_grad_norm = self._ppo_backward_and_step(
            ctx, stats, total_loss, epoch_i, mb_i, mb_idx, d_heads, e_heads, policy_loss, value_loss, entropy)
        vals, per_head_kl_mb = self._ppo_after_step(
            ctx, stats, mb_obs, parts, mb_actions, mb_idx, mb_opp_mask, e_heads, before,
            new_log_probs, mb_old_lp, total_loss, policy_loss, value_loss, entropy)
        kl_after_step = vals["kl_after_step"]

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"  [e{epoch_i} mb{mb_i:02d}]"
                f"  grad={raw_grad_norm:.1f}"
                f"  total={vals['total_loss']:.3f}"
                f"  pol={vals['policy_loss']:.3f}"
                f"  val={vals['value_loss']:.3f}(x{self.vf_coef})={self.vf_coef * vals['value_loss']:.3f}"
                f"  kl={kl_after_step:.4f}"
                f"  mv_shift={vals['movedir_mean_shift']:.3f}"
                f"  mv_kl={vals['movedir_hyp_kl']:.4f}"
                f"  kk_shift={vals['kickdir_mean_shift']:.3f}"
            )
        stats.policy_loss.append(vals["policy_loss"])
        stats.value_loss.append(vals["value_loss"])
        stats.entropy.append(vals["entropy"])
        stats.kl.append(kl_after_step)

        if kl_after_step > self.target_kl:
            _stop_head_kl_str = (
                "  ".join(
                    f"{k}={v:+.4f}" for k, v in zip(HEAD_LP_KEYS, per_head_kl_mb.tolist())
                    if abs(v) > 0.001
                )
                if per_head_kl_mb is not None else "(head_log_probs unavailable)"
            )
            log.info(
                f"  [early stop e{epoch_i} mb{mb_i}]"
                f"  KL={kl_after_step:.5f} > target={self.target_kl}"
                f"  steps_this_update={len(stats.kl)}\n"
                f"    [per-head KL] {_stop_head_kl_str}"
            )
            return True
        return False

    def _exec_drift_snapshot(self, e_heads) -> dict:
        """Copies of the direction spreads and execution logits before the step,
        for the drift diagnostics."""
        net = self.execution_net
        return {
            "move_dir_log_kappa": net.move_dir_log_kappa.detach().clone(),
            "kick_dir_log_kappa": net.kick_dir_log_kappa.detach().clone(),
            "kick_dir_z_log_std": net.kick_dir_z_log_std.detach().clone(),
            "exec_move": e_heads.exec_move_logit.detach().clone(),
            "sprint": e_heads.sprint_logit.detach().clone(),
            "kick": e_heads.kick_logit.detach().clone(),
            "tackle_attempt": e_heads.tackle_attempt_logit.detach().clone(),
        }

    def _ppo_track_ratio_spike(self, stats: _UpdateStats, ratio: torch.Tensor, d_heads, e_heads, mb_actions: dict,
                               em: torch.Tensor, mb_idx: torch.Tensor, mb_old_lp: torch.Tensor,
                               new_log_probs: torch.Tensor, mb_adv: torch.Tensor, epoch_i: int, mb_i: int) -> None:
        """Keep the highest-ratio row of the whole update, with its per-head
        log-probs from the same (pre-step) weights that produced the ratio."""
        with torch.no_grad():
            _mb_max_ratio_t, _mb_max_local = ratio.max(dim=0)
            _mb_max_ratio_f = float(_mb_max_ratio_t)
            if _mb_max_ratio_f > stats.ratio_spike["ratio"]:
                _row = int(_mb_max_local)
                _per_head_row = self._per_head_new_log_probs(d_heads, e_heads, mb_actions, em)[_row].detach().cpu()
                stats.ratio_spike = {
                    "ratio": _mb_max_ratio_f,
                    "epoch": epoch_i,
                    "mb": mb_i,
                    "row": int(mb_idx[_row]),
                    "old_lp": float(mb_old_lp[_row]),
                    "new_lp": float(new_log_probs[_row]),
                    "adv": float(mb_adv[_row]),
                    "per_head_new": _per_head_row,
                }

    def _main_and_direction_params(self) -> tuple[list, list]:
        """(non-direction params of decision_net + execution_net, direction params).

        The direction heads (move/kick direction weights and their log_kappa /
        z log_std) are clipped as their own group with direction_max_grad_norm, so
        one sample's large direction gradient cannot shrink every other head's
        step, and a calm direction gradient cannot lend clip headroom to a large
        gradient elsewhere."""
        non_direction = [
            p for p in list(self.decision_net.parameters()) + list(self.execution_net.parameters())
            if id(p) not in self.direction_param_ids
        ]
        direction = [p for p in self.execution_net.parameters() if id(p) in self.direction_param_ids]
        return non_direction, direction

    def _ppo_backward_and_step(self, ctx: _UpdateContext, stats: _UpdateStats, total_loss: torch.Tensor,
                               epoch_i: int, mb_i: int, mb_idx: torch.Tensor, d_heads, e_heads,
                               policy_loss: torch.Tensor, value_loss: torch.Tensor,
                               entropy: torch.Tensor) -> float:
        """Backward, the separate value net's own clipped step, gradient-norm
        measurements, the two-group clip and the (warmed-up) optimizer step.
        Returns the raw pre-clip gradient norm of decision_net + execution_net.

        With separate_value_net the value loss's gradient normally reaches only
        value_net (its input heads are detached); with
        share_value_grad_with_decision a scaled slice also lands on decision_net
        and is stepped by self.optimizer like any other decision_net gradient.
        Every norm is collected as a tensor and synced once."""
        self.optimizer.zero_grad()
        if self.separate_value_net:
            self.value_net_optimizer.zero_grad()
        total_loss.backward()
        if self.separate_value_net:
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            self.value_net_optimizer.step()

        # max_norm=inf only measures.
        _gn_tensors: dict[str, torch.Tensor] = {}
        for _head_name, _attr in EXEC_HEAD_MODULES:
            _head_params = list(getattr(self.execution_net, _attr).parameters())
            if _head_params:
                _gn_tensors[f"head:{_head_name}"] = torch.nn.utils.clip_grad_norm_(_head_params, float("inf"))
        _mv_ls_grad = self.execution_net.move_dir_log_kappa.grad
        if _mv_ls_grad is not None:
            _gn_tensors["mv_log_std_grad"] = _mv_ls_grad.norm()
        _gn_tensors["raw"] = torch.nn.utils.clip_grad_norm_(
            list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
            float("inf"),
        )
        # TEMPORARY diagnostic (see _dump_nan_diagnostic docstring). The loss WAS
        # finite but the gradient is not: a backward-pass singularity, not
        # corrupted forward-pass data.
        if not torch.isfinite(_gn_tensors["raw"]):
            self._dump_nan_diagnostic(
                "backward", epoch_i, mb_i, mb_idx, [ctx.batch["track_ids"][i] for i in mb_idx.tolist()],
                d_heads, e_heads, policy_loss, value_loss, entropy,
            )
        _non_direction_params, _direction_params = self._main_and_direction_params()
        _gn_tensors["main"] = nn.utils.clip_grad_norm_(_non_direction_params, self.max_grad_norm)
        if _direction_params:
            _gn_tensors["dir"] = nn.utils.clip_grad_norm_(_direction_params, self.direction_max_grad_norm)
            _dir_params_before = [p.detach().clone() for p in _direction_params]

        # clip_grad_norm_ returns a CPU tensor when every parameter in its group
        # has .grad None (e.g. a head with no gradient this minibatch); .to() keeps
        # the stack below on one device.
        _gn_names = list(_gn_tensors.keys())
        _gn_vals = dict(zip(
            _gn_names,
            torch.stack([_gn_tensors[k].to(self.device) for k in _gn_names]).tolist(),
        ))
        for _head_name, _attr in EXEC_HEAD_MODULES:
            _gnk = f"head:{_head_name}"
            if _gnk in _gn_vals:
                stats.head_grad_norm[_head_name].append(_gn_vals[_gnk])
        if "mv_log_std_grad" in _gn_vals:
            stats.mv_log_std_grad.append(_gn_vals["mv_log_std_grad"])
        raw_grad_norm = _gn_vals["raw"]
        # Which parameter tensor drove an outlier step (shared with BC/DAgger);
        # reads .grad, so it must run before the step.
        _maybe_log_grad_norm_spike(self, "PPO", epoch_i, mb_i, raw_grad_norm)
        _gn_main = _gn_vals["main"]
        stats.grad_norm_main.append(_gn_main)
        if _gn_main > self.max_grad_norm:
            stats.clip_triggered_main += 1
        if _direction_params:
            _gn_dir = _gn_vals["dir"]
            stats.grad_norm_dir.append(_gn_dir)
            if _gn_dir > self.direction_max_grad_norm:
                stats.clip_triggered_dir += 1
        self._ppo_optimizer_step(ctx, stats)
        if _direction_params:
            _dir_delta_norm = torch.sqrt(sum(
                (p.detach() - p_before).pow(2).sum()
                for p, p_before in zip(_direction_params, _dir_params_before)
            )).item()
            stats.param_delta_dir.append(_dir_delta_norm)
        return raw_grad_norm

    def _ppo_optimizer_step(self, ctx: _UpdateContext, stats: _UpdateStats) -> None:
        """optimizer.step(), with the per-update linear LR warmup over the first
        lr_warmup_steps steps of EVERY update (restoring the scheduled lrs after)."""
        if self.lr_warmup_steps > 0 and stats.warm_steps_done < self.lr_warmup_steps:
            _ws = _lr_warmup_scale(stats.warm_steps_done, self.lr_warmup_steps)
            for _pg, _base in zip(self.optimizer.param_groups, ctx.warm_base_lrs):
                _pg["lr"] = _base * _ws
        stats.warm_steps_done += 1
        self.optimizer.step()
        if self.lr_warmup_steps > 0 and stats.warm_steps_done == self.lr_warmup_steps:
            for _pg, _base in zip(self.optimizer.param_groups, ctx.warm_base_lrs):
                _pg["lr"] = _base

    def _ppo_after_step(self, ctx: _UpdateContext, stats: _UpdateStats, mb_obs: dict, parts: tuple,
                        mb_actions: dict, mb_idx: torch.Tensor, mb_opp_mask: Optional[torch.Tensor], e_heads,
                        before: dict, new_log_probs: torch.Tensor, mb_old_lp: torch.Tensor,
                        total_loss: torch.Tensor, policy_loss: torch.Tensor, value_loss: torch.Tensor,
                        entropy: torch.Tensor) -> tuple[dict, Optional[torch.Tensor]]:
        """Re-run the minibatch with the stepped weights: KL after the step (the
        early-stop signal), per-head KL, and the direction/logit drift. Returns
        (scalar values, per-head KL or None), every scalar synced in one call."""
        sf, of, em, bf, gf, sat, oat = parts
        with torch.no_grad():
            d_after = self._ppo_decision_forward(mb_obs, parts)
            e_after = self.execution_net(sf, of, em, bf, gf, d_after, sat, oat)
            lp_after = self._recompute_log_prob(d_after, e_after, mb_actions, em)
            if mb_opp_mask is not None:
                lp_after = self._mask_total_log_probs(
                    lp_after, self._per_head_new_log_probs(d_after, e_after, mb_actions, em), mb_opp_mask,
                )
            # move_direction's own KL contribution (it is part of the ratio).
            _stored_raw_mb = mb_actions["move_dir_raw"]
            _log_std_move = self.execution_net.move_dir_log_kappa.to(self.device)
            _lp_movedir_before = self._move_dir_head(e_heads.move_direction, _log_std_move).log_prob(_stored_raw_mb)
            _lp_movedir_after = self._move_dir_head(e_after.move_direction, _log_std_move).log_prob(_stored_raw_mb)
            # Guarded on head_log_probs, which RolloutBuffer and augment_batch always
            # provide but a hand-built batch might not.
            per_head_kl_mb = None
            if "head_log_probs" in ctx.batch:
                mb_old_head_lp = ctx.batch["head_log_probs"][mb_idx].to(self.device)
                per_head_new_lp_after = self._per_head_new_log_probs(d_after, e_after, mb_actions, em)
                per_head_kl_mb = (
                    (mb_old_head_lp - per_head_new_lp_after).mean(dim=0) if mb_opp_mask is None
                    else ((mb_old_head_lp - per_head_new_lp_after) * mb_opp_mask).sum(dim=0)
                    / mb_opp_mask.sum(dim=0).clamp(min=1.0)
                )
                stats.head_kl.append(per_head_kl_mb.detach().cpu())
            net = self.execution_net
            _shift_tensors: dict[str, torch.Tensor] = {
                "movedir_mean_shift": (e_after.move_direction - e_heads.move_direction).norm(dim=-1).mean(),
                "kickdir_mean_shift": (e_after.kick_direction - e_heads.kick_direction).norm(dim=-1).mean(),
                "movedir_hyp_kl": (_lp_movedir_before - _lp_movedir_after).mean(),
                "log_std_shift_move": (net.move_dir_log_kappa.detach() - before["move_dir_log_kappa"]).abs().mean(),
                "log_std_shift_kick": (net.kick_dir_log_kappa.detach() - before["kick_dir_log_kappa"]).abs().mean(),
                "log_std_shift_kickz": (net.kick_dir_z_log_std.detach() - before["kick_dir_z_log_std"]).abs().mean(),
                "logit_shift_exec_move": (e_after.exec_move_logit - before["exec_move"]).abs().mean(),
                "logit_shift_sprint": (e_after.sprint_logit - before["sprint"]).abs().mean(),
                "logit_shift_kick": (e_after.kick_logit - before["kick"]).abs().mean(),
                "logit_shift_tackle_attempt": (e_after.tackle_attempt_logit - before["tackle_attempt"]).abs().mean(),
            }
        # Clamp to a finite floor first: a near-zero-probability row
        # (log_prob = -inf) would otherwise make the KL +inf.
        _shift_tensors["approx_kl"] = (mb_old_lp - new_log_probs.clamp(min=-1e6)).mean()
        _shift_tensors["kl_after_step"] = (mb_old_lp - lp_after.clamp(min=-1e6)).mean()
        _shift_tensors["total_loss"] = total_loss.detach().squeeze()
        _shift_tensors["policy_loss"] = policy_loss.detach()
        _shift_tensors["value_loss"] = value_loss.detach()
        _shift_tensors["entropy"] = entropy.detach().squeeze()
        _shift_names = list(_shift_tensors.keys())
        vals = dict(zip(_shift_names, torch.stack([_shift_tensors[k] for k in _shift_names]).tolist()))

        stats.continuous_mean_shift["move_direction"].append(vals["movedir_mean_shift"])
        stats.continuous_mean_shift["kick_direction"].append(vals["kickdir_mean_shift"])
        stats.continuous_log_std_shift["move_direction"].append(vals["log_std_shift_move"])
        stats.continuous_log_std_shift["kick_direction"].append(vals["log_std_shift_kick"])
        stats.kickz_log_std_shift.append(vals["log_std_shift_kickz"])
        stats.discrete_logit_shift["exec_move"].append(vals["logit_shift_exec_move"])
        stats.discrete_logit_shift["sprint"].append(vals["logit_shift_sprint"])
        stats.discrete_logit_shift["kick"].append(vals["logit_shift_kick"])
        stats.discrete_logit_shift["tackle_attempt"].append(vals["logit_shift_tackle_attempt"])
        return vals, per_head_kl_mb

    # ----- continuations -----------------------------------------------------

    def _ppo_value_only_continuation(self, ctx: _UpdateContext, stats: _UpdateStats) -> None:
        """After a KL early stop, keep training the value head alone for
        value_only_continuation_epochs.

        The early stop often cuts the combined loop after 1-3 minibatches, which
        starved the critic regardless of its own learning rate. Policy and value
        have independent parameter groups and gradients, so value-only steps
        (no policy loss or backward) carry no KL risk."""
        value_only_steps = 0
        for _ in range(self.value_only_continuation_epochs):
            indices = torch.randperm(ctx.n)
            for start in range(0, ctx.n, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]
                if len(mb_idx) == 0:
                    continue
                mb_obs = self._minibatch_obs(ctx.batch, mb_idx)
                mb_ret = ctx.returns[mb_idx].to(self.device)
                parts = _obs_parts(mb_obs)
                with torch.no_grad():
                    d_heads_vo = self._ppo_decision_forward(mb_obs, parts)
                new_values_vo = self._value_only_forward(parts, d_heads_vo).squeeze(-1)
                ret_var = ctx.returns.var().clamp(min=1.0)
                value_loss_vo = F.mse_loss(new_values_vo, mb_ret) / ret_var
                stats.accum_value_by_outcome(new_values_vo, mb_ret, mb_idx, ctx.step_outcomes)
                if self.separate_value_net:
                    self.value_net_optimizer.zero_grad()
                    (self.vf_coef * value_loss_vo).backward()
                    nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                    self.value_net_optimizer.step()
                else:
                    self.optimizer.zero_grad()
                    (self.vf_coef * value_loss_vo).backward()
                    nn.utils.clip_grad_norm_(
                        list(self.execution_net.value_head.parameters())
                        + list(self.execution_net.value_ai_type_channel.parameters()),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                stats.value_loss.append(value_loss_vo.item())
                value_only_steps += 1
        if value_only_steps:
            log.info(
                f"  [value-only continuation] {value_only_steps} extra minibatch step(s)"
                f"  after policy early-stop  final_val_loss={stats.value_loss[-1]:.4f}"
            )

    def _ppo_bc_only_continuation(self, ctx: _UpdateContext, stats: _UpdateStats) -> None:
        """BC-only steps for bc.bc_only_continuation_epochs after the main loop.

        Same rationale as the value-only continuation: the early stop was also
        truncating BC's intended gradient budget. BC updates the same parameters
        as the policy, so this runs after the value-only continuation and
        computes no ratio or KL, so it cannot trigger another early stop.

        The coefficient is bc.bc_only_continuation_coeff when set, else the
        annealed bc_coeff. A fixed value lets aux_coeff anneal to 0 (no BC pull on
        the main policy gradient) while this loop keeps nudging toward the demos."""
        from footballcoach.ai.ppo.bc import bc_loss_from_tensor, direction_magnitude_reg

        bc_only_coeff = (
            self.bc_only_continuation_coeff
            if self.bc_only_continuation_coeff is not None
            else ctx.bc_coeff
        )
        if not (ctx.has_bc and self.bc_only_continuation_epochs > 0 and bc_only_coeff > 0.0):
            return
        bc_only_steps = 0
        last_bc_only_loss = 0.0
        bc_clip_triggered_main = 0
        bc_clip_triggered_dir = 0
        bc_grad_norm_main: list[float] = []
        bc_grad_norm_dir: list[float] = []
        for _ in range(self.bc_only_continuation_epochs):
            indices = torch.randperm(ctx.n)
            for start in range(0, ctx.n, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]
                if len(mb_idx) == 0:
                    continue
                mb_obs = self._minibatch_obs(ctx.batch, mb_idx)
                mb_bc = ctx.batch["bc_labels"][mb_idx].to(self.device)
                if ctx.bc_coeff_table is not None:
                    # The continuation applies the GLOBAL coefficient: keep it to
                    # the rows the global schedules govern (source 0).
                    from footballcoach.ai.ppo.bc import _I_VALID
                    mb_bc = mb_bc.clone()
                    mb_bc[ctx.batch["bc_sources"][mb_idx].to(self.device) != 0, _I_VALID] = 0.0
                parts = _obs_parts(mb_obs)
                sf, of, em, bf, gf, sat, oat = parts
                d_heads_bo = self._ppo_decision_forward(mb_obs, parts)
                e_heads_bo = self.execution_net(sf, of, em, bf, gf, d_heads_bo, sat, oat)
                bc_loss_bo, _ = bc_loss_from_tensor(
                    mb_bc, self._bc_heads_for_loss(d_heads_bo), e_heads_bo,
                    **self._bc_loss_kwargs(),
                    return_breakdown=True,
                )
                if not bc_loss_bo.requires_grad:
                    # No valid BC rows in this minibatch (bc_loss_from_tensor returns
                    # a disconnected zero): nothing to backprop.
                    continue
                self.optimizer.zero_grad()
                (bc_only_coeff * bc_loss_bo + direction_magnitude_reg(e_heads_bo, self._bc_dir_mag_reg_coef)).backward()
                # Same direction-head clip isolation as the main loop.
                _non_direction_params_bo, _direction_params_bo = self._main_and_direction_params()
                _bc_gn_main = nn.utils.clip_grad_norm_(_non_direction_params_bo, self.max_grad_norm).item()
                bc_grad_norm_main.append(_bc_gn_main)
                if _bc_gn_main > self.max_grad_norm:
                    bc_clip_triggered_main += 1
                if _direction_params_bo:
                    _bc_gn_dir = nn.utils.clip_grad_norm_(_direction_params_bo, self.direction_max_grad_norm).item()
                    bc_grad_norm_dir.append(_bc_gn_dir)
                    if _bc_gn_dir > self.direction_max_grad_norm:
                        bc_clip_triggered_dir += 1
                self.optimizer.step()
                stats.bc_loss.append(bc_loss_bo.detach().item())
                last_bc_only_loss = bc_loss_bo.item()
                bc_only_steps += 1
        if bc_only_steps:
            _bc_n_main = len(bc_grad_norm_main)
            _bc_n_dir = len(bc_grad_norm_dir)
            _bc_clip_pct_main = (100.0 * bc_clip_triggered_main / _bc_n_main) if _bc_n_main else 0.0
            _bc_clip_pct_dir = (100.0 * bc_clip_triggered_dir / _bc_n_dir) if _bc_n_dir else 0.0
            log.info(
                f"  [bc-only continuation grad clip] main: {bc_clip_triggered_main}/{_bc_n_main}"
                f" steps clipped ({_bc_clip_pct_main:.0f}%)  mean_norm={np.mean(bc_grad_norm_main) if _bc_n_main else 0.0:.3f}"
                + (
                    f"  |  direction: {bc_clip_triggered_dir}/{_bc_n_dir} steps clipped"
                    f" ({_bc_clip_pct_dir:.0f}%)  mean_norm={np.mean(bc_grad_norm_dir) if _bc_n_dir else 0.0:.3f}"
                    if _bc_n_dir else ""
                )
            )
            log.info(
                f"  [bc-only continuation] {bc_only_steps} extra minibatch step(s)"
                f"  after policy early-stop  final_bc_loss={last_bc_only_loss:.4f}"
            )

    # ----- result ------------------------------------------------------------

    def _ppo_update_metrics(self, ctx: _UpdateContext, stats: _UpdateStats, pre_update_value_loss: float,
                            flip_consistency: Optional[dict]) -> dict:
        """The dict _ppo_update returns (consumed by the rollout summary)."""
        batch, returns = ctx.batch, ctx.returns
        g = stats.grad_clip_summary()

        # Per-head activation rates (0-100%) from the stored 0/1 actions.
        def _act(key: str) -> int:
            t = batch[f"action/{key}"]
            return round(float(t.mean()) * 100)

        head_act = {
            "mv":  _act("move"),
            "gp":  _act("get_possession_extra"),
            "emv": _act("exec_move"),
            "spr": _act("sprint"),
            "kck": _act("kick"),
            "tk":  _act("tackle_attempt"),
            "sh":  _act("shoot"),
            "hld": _act("hold_position"),
            "ta_p": float(np.mean(stats.tackle_prob)) if stats.tackle_prob else float('nan'),
            "kk_p": float(np.mean(stats.kick_prob)) if stats.kick_prob else float('nan'),
        }
        return {
            "policy_loss": float(np.mean(stats.policy_loss)),
            "value_loss": float(np.mean(stats.value_loss)),
            "pre_update_value_loss": pre_update_value_loss,
            "flip_consistency": flip_consistency,
            "entropy": float(np.mean(stats.entropy)),
            "entropy_breakdown": {k: float(np.mean(v)) for k, v in stats.entropy_breakdown.items()},
            "ent_coef": ctx.ent_coef,
            "approx_kl": float(np.mean(stats.kl)),
            "bc_loss": float(np.mean(stats.bc_loss)),
            "bc_tackle_loss": float(np.mean(stats.bc_tackle_loss)) if stats.bc_tackle_loss else 0.0,
            "bc_coeff": ctx.bc_coeff,
            "bc_kick_coeff": ctx.bc_kick_coeff,
            "bc_tackle_coeff": ctx.bc_tackle_coeff,
            "bc_by_source": self._bc_by_source_metrics(ctx, stats),
            "bc_pull": self._bc_pull_metrics(stats),
            "epoch_time_ms": float(np.mean(stats.epoch_times)) if stats.epoch_times else 0.0,
            "move_log_std": self.execution_net.move_dir_log_kappa.data.tolist(),
            "kick_log_std": self.execution_net.kick_dir_log_kappa.data.tolist(),
            "kick_z_log_std": self.execution_net.kick_dir_z_log_std.data.tolist(),
            "kick_power_log_std": self.execution_net.kick_power_log_std.data.tolist(),
            "mv_ls_grad": float(np.mean(stats.mv_log_std_grad)) if stats.mv_log_std_grad else 0.0,
            "head_act": head_act,
            "kick_power_frac": _kick_power_frac_stats(batch, self.kick_power_floor),
            "grad_clip_pct_main": g["pct_main"],
            "grad_clip_pct_dir": g["pct_dir"],
            "grad_clip_mean_norm_main": g["mean_main"],
            "grad_clip_mean_norm_dir": g["mean_dir"],
            "values_mean": float(batch["values"].mean()),
            "values_std": float(batch["values"].std()),
            "returns_mean": float(returns.mean()),
            "returns_std": float(returns.std()),
            "adv_mean": float(batch["advantages"].mean()),
            "adv_std": float(batch["advantages"].std()),
            "adv_abs_mean": float(batch["advantages"].abs().mean()),
            "mean_sq_td": float(((returns - batch["values"]) ** 2).mean()),
        }

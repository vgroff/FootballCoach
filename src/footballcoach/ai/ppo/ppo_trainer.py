"""PPO training loop - custom PyTorch implementation.

See ai_design_doc.md sections 3.1 and 9.6 for design rationale.

This is a single-player, single-environment PPO loop for the MVP experiments.
Multi-player / multi-env scaling is a future extension.

Key design choices:
- Custom loop (not stable-baselines3) - see design doc 3.1 for why.
- Shared actor/critic trunk (Option A from design doc 9.4).
- Multiple epochs over shuffled minibatches (standard PPO).
- Early stop per batch if approx_kl > target_kl (standard PPO safety valve).
- All hyperparameters from ai_config.json.

Usage:
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    from footballcoach.ai.env.scenario_env import ScenarioEnv
    from footballcoach.ui.scenarios import SCENARIOS

    env = ScenarioEnv(SCENARIOS[0], trainee_player_id="kicker", phase=1, max_episode_s=120.0)
    trainer = PPOTrainer.from_config()
    trainer.train(env, total_steps=500_000)
"""
from __future__ import annotations

import collections
import dataclasses
import logging
import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Rollout collection is CPU-inference-bound (profiled: ~88% of per-step
# wall time is _sample_action, of which only ~11% is actual matmul FLOPs --
# the rest is Python/PyTorch dispatch overhead on tiny batch-of-1 calls).
# torch.distributions' default validate_args=True re-checks every
# distribution's parameters/sample against its support on EVERY
# construction and every .log_prob()/.sample() call -- pure assertion
# overhead in this hot path, not something PPO relies on (log_prob/sample
# numerics are unaffected either way). Disabling it globally measured ~13%
# faster single-env rollout stepping (48.0 -> 55.4 steps/s) with zero
# behavior change. Set once at import time so it's in effect in the main
# process, every rollout_worker.py subprocess, and every eval_worker.py
# subprocess (all import this module for PPOTrainer/rebuild_inference_trainer).
torch.distributions.Distribution.set_default_validate_args(False)

from footballcoach.ai.action.distributions import IndependentBernoulli, _von_mises_kl, kick_power_from_raw
from footballcoach.ai.action.schema import DecisionHeadsRaw
from footballcoach.ai.eval.seeded_eval import default_eval_seeds
from footballcoach.ai.models.decision_network import DecisionNetwork
from footballcoach.ai.models.execution_network import ExecutionNetwork
from footballcoach.ai.obs.augment import augment_batch
from footballcoach.ai.obs.y_canonical import YCanonicalNetworkWrapper, mirror_y_obs, y_flip_mask
from footballcoach.ai.progress import ProgressReporter
from footballcoach.ai.obs.canonical import CanonicalNetworkWrapper, canonicalize_obs
from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer, HEAD_LP_KEYS
from footballcoach.ai.ppo.action_opportunity_stats import (
    format_opportunity_block, has_opportunity_flags, opportunity_masks, opportunity_stats,
)
from footballcoach.ai.ppo.schedules import TrainingSchedules
from footballcoach.ai.ppo.checkpoint_io import CheckpointIOMixin
from footballcoach.ai.ppo.rollout_logging import RolloutLoggingMixin
from footballcoach.ai.ppo.trainer_eval import TrainerEvalMixin
from footballcoach.ai.ppo.trainer_common import REWARD_COMP_LABELS, log
from footballcoach.ai.ppo.action_sampling import ActionSamplingMixin
from footballcoach.ai.ppo.pretraining import PretrainingMixin
from footballcoach.ai.ppo.trainer_common import EXEC_HEAD_MODULES, _DECISION_HEADS_FIELD_NAMES, _action_to_numpy, _ai_types, _detach_decision_heads, _episode_abs_adv_means, _maybe_log_grad_norm_spike, _merge_worker_batches, format_outcome_rmse_breakdown, value_mse_by_outcome
from footballcoach.ai.ppo.ppg import PPGMixin
from footballcoach.ai.ppo.trainer_common import _lr_warmup_scale





# exp() overflows float32 past ~88.7. Under dual-clip an overflowed (inf)
# ratio still poisons the gradient: dual-clip selects the finite floor, but
# autograd multiplies the unselected branch's zero gradient by d(exp)/dx = inf,
# giving 0 * inf = NaN. Capping the log-ratio at 80 (exp(80) ~= 5.5e34) keeps
# the ratio finite; a ratio that large is already on dual-clip's zero-gradient
# floor, so only the NaN changes. Below the cap the result is bit-identical.
_MAX_LOG_RATIO = 80.0










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



_GRAD_NORM_SPIKE_HISTORY_MAXLEN = 200  # rolling window size for the adaptive spike bar's running mean/std











# decision_net attribute names of the 7 Bernoulli heads that participate in
# PPO's log_prob/entropy computation (_recompute_log_prob/_compute_entropy)
# and so are eligible for PPOTrainer._ppo_lp_masked_heads masking when
# frozen. pass_target_logits/tackle_target_logits/mark_target_logits are
# also freezable via set_frozen_heads but deliberately excluded here -- they
# aren't part of HEAD_LP_KEYS/the log_prob stack at all (gated by their
# parent Bernoulli instead), so masking them would be a no-op.
_LP_HEAD_NAMES: tuple[str, ...] = (
    "shoot_logit", "pass_logit", "move_logit", "tackle_logit",
    "get_possession_raw", "mark_logit", "hold_position_logit",
)



def _slice_dataclass_row(obj, i: int):
    """Return a NEW instance of ``obj``'s dataclass type with every tensor
    field sliced to row ``i``, keeping the batch dim (shape ``(1, ...)``).

    Used by ``PPOTrainer._sample_action_batch`` to run
    ``_sample_action_from_heads``'s unchanged per-row logic on one row of an
    already-batched ``DecisionHeadsRaw``/``ExecutionHeadsRaw``. Reflective
    (uses ``dataclasses.fields()``) rather than hardcoding field names, so it
    can't silently miss a field if either schema gains/loses one later;
    ``None`` fields (e.g. the physics-encoder passthrough fields, absent
    when that integration is disabled) pass through as ``None`` unchanged.
    """
    kwargs = {}
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        kwargs[f.name] = None if val is None else val[i:i + 1]
    return type(obj)(**kwargs)




def _kick_power_frac_stats(batch: dict, floor: float = 0.0) -> dict:
    """n / mean / std of the executed kick power fraction (floor + (1-floor) *
    sigmoid of the stored raw draw) over the rows where a kick was sampled."""
    kick = batch["action/kick"].reshape(-1) > 0.5
    if not bool(kick.any()):
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    pf = kick_power_from_raw(batch["action/kick_power_raw"].reshape(-1)[kick].float(), floor)
    return {"n": int(pf.numel()), "mean": float(pf.mean()), "std": float(pf.std()) if pf.numel() > 1 else 0.0}














def _trimmed_mean(x: torch.Tensor, lo_q: float, hi_q: float) -> float:
    """Mean of ``x`` after dropping values outside ``[quantile(lo_q),
    quantile(hi_q)]`` -- a robust companion to a plain mean, so a handful of
    extreme outliers (e.g. a rare mis-valued state producing a huge
    squared-error term, or a ratio-spike row) don't single-handedly
    dominate the reported number. Returns ``nan`` for an empty input.
    Shared core for ``_trimmed_mean_p10_p90``/``_trimmed_mean_p10_p75``
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





































class PPOTrainer(PPGMixin, PretrainingMixin, ActionSamplingMixin, CheckpointIOMixin, RolloutLoggingMixin, TrainerEvalMixin):
    """PPO trainer for the two-network (decision + execution) player AI.

    Args:
        decision_net: The decision network.
        execution_net: The execution network.
        cfg: The full ai_config.json dict (or the relevant 'ppo' sub-dict).
        device: torch device.
        checkpoint_dir: Directory to save/load checkpoints.
    """

    def __init__(
        self,
        decision_net: DecisionNetwork,
        execution_net: ExecutionNetwork,
        cfg: dict,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None,
        inference_only: bool = False,
        separate_value_net: bool = False,
        share_value_grad_with_decision: bool = False,
        decision_value_coef: float = 0.5,
        y_canonical: Optional[bool] = None,
    ):
        # Opt-in y-canonical frame (ai/obs/y_canonical.py): None -> ppo.y_canonical from the config.
        self.y_canonical = bool(cfg["ppo"].get("y_canonical", False)) if y_canonical is None else bool(y_canonical)
        # Wrapped so every forward call automatically canonicalizes
        # self_feat/other_feat/ball_feat into the canonical AI frame (see
        # ai/obs/canonical.py "CanonicalNetworkWrapper") — existing call
        # sites throughout this file need NO changes; state_dict()/
        # load_state_dict() are transparently delegated so checkpoints stay
        # unaffected.
        self.decision_net = CanonicalNetworkWrapper(decision_net)
        self.execution_net = CanonicalNetworkWrapper(execution_net)
        if self.y_canonical:
            self.decision_net = YCanonicalNetworkWrapper(self.decision_net, "decision")
            self.execution_net = YCanonicalNetworkWrapper(self.execution_net, "execution")
        self.device = device or torch.device("cpu")
        # --- Permanent separate-trunk value network (see CLI --separate-value-net) ---
        # A fully independent ExecutionNetwork (same class/config, own weights, zero
        # sharing with self.execution_net) used as the ONLY critic for the entire
        # training run whenever this is enabled -- unlike
        # --experiment-separate-value-net (diagnostic-only, discarded after
        # pretrain_value()), this one is the real, permanent critic: its .value
        # output replaces execution_net.value everywhere (GAE bootstrap, PPO value
        # loss, pretrain_value()/pretrain_combined() value warm-up, _get_value(),
        # _sample_action()). It never receives BC gradients -- the whole point is a
        # critic trunk that never gets BC-primed, unlike the shared trunk which is
        # BC-pretrained before PPO. Persisted in checkpoints under "value_net"/
        # "value_net_optimizer" so it survives resume/--latest/--from-pretrained.
        self.separate_value_net = bool(separate_value_net)
        self.value_net: Optional[ExecutionNetwork] = None
        self.value_net_optimizer: Optional[torch.optim.Optimizer] = None
        if self.separate_value_net:
            # network.value_net_trunk_hidden (ai_config.json): optional override
            # for value_net's trunk_hidden, independent of the main policy
            # execution_net's trunk size. None/absent (default) = same size as
            # the main trunk (network.trunk_hidden).
            _value_trunk_override = cfg.get("network", {}).get("value_net_trunk_hidden")
            self.value_net = CanonicalNetworkWrapper(ExecutionNetwork.from_config(
                trunk_hidden_override=_value_trunk_override
            ))
            if self.y_canonical:
                self.value_net = YCanonicalNetworkWrapper(self.value_net, "execution")
        # --- Opt-in: let separate_value_net's critic gradient reach decision_net ---
        # Only meaningful when separate_value_net is on -- in the default
        # (non-separate) mode decision_net already receives value_loss's
        # gradient unconditionally via execution_net's shared trunk (nothing
        # detaches it there), so this flag would be a no-op. Scoped to the
        # main PPO update's per-minibatch loop ONLY (_ppo_update) -- that is
        # the one live call site where decision_net's forward isn't already
        # under torch.no_grad() (pretrain_value(), the value-only
        # continuation) or running through a genuinely separate, differently
        # -timed optimizer/backward (Phase 1's _use_separate_value_training
        # fallback), either of which would make "just stop detaching" silently
        # wrong rather than inert. See _scale_decision_heads_grad's docstring
        # for the actual gradient-scaling mechanism and
        # ai_trainer_knowledge.md "Separate value network" for the full
        # rationale. decision_value_coef is deliberately a SEPARATE knob from
        # vf_coef: vf_coef already scales value_loss's contribution to
        # value_net's own trunk (and, in non-separate mode, execution_net's),
        # and must keep doing so unchanged regardless of this feature --
        # decision_value_coef only scales the additional slice of that
        # gradient crossing into decision_net specifically.
        self.share_value_grad_with_decision = bool(share_value_grad_with_decision) and self.separate_value_net
        if share_value_grad_with_decision and not self.separate_value_net:
            log.warning(
                "share_value_grad_with_decision has no effect without separate_value_net=True "
                "(non-separate mode already lets value_loss's gradient reach decision_net via "
                "the shared trunk) -- ignoring."
            )
        self.decision_value_coef = float(decision_value_coef)
        self.checkpoint_dir = checkpoint_dir

        ppo_cfg = cfg["ppo"]
        curriculum_cfg = cfg.get("curriculum", {})
        bc_cfg = cfg.get("bc", {})
        self.schedules = TrainingSchedules(ppo_cfg, curriculum_cfg, bc_cfg)

        self.gamma = float(ppo_cfg.get("gamma", 0.99))
        self.lam = float(ppo_cfg.get("lam", 0.95))
        self.clip_range = float(ppo_cfg.get("clip_range", 0.2))
        self.vf_coef = float(ppo_cfg.get("vf_coef", 0.5))
        # ent_coef is now a per-progress schedule (self.schedules.ent(progress),
        # built from ppo.ent_coef_start/end/anneal_fraction, falling back to
        # the legacy constant ppo.ent_coef when those aren't set) rather than
        # a fixed float — see schedules.py and _ppo_update's `ent_coef` local.
        # Previous rollout's per-head entropy breakdown, for the "[per-head
        # entropy]" diagnostic's Δ-from-last-rollout column — see
        # _log_rollout_summary. None until the first rollout completes.
        self._prev_entropy_breakdown: Optional[dict] = None
        self.max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))
        # None (disabled) or a float k: whenever a minibatch's raw (pre-clip)
        # grad norm exceeds (running_mean + k*running_std) of a rolling
        # window of recent raw grad norms (pooled across PPO/BC/DAgger --
        # see _grad_norm_history below), dump a per-parameter-tensor
        # breakdown (top _GRAD_NORM_SPIKE_TOP_K by norm) for THAT minibatch
        # only -- see _maybe_log_grad_norm_spike(). Adaptive rather than a
        # fixed absolute number because "typical scale" varies a lot by
        # run/config (confirmed: a real run with mean~5 max~200 needs a very
        # different bar than a small smoke-test with mean~4.6 max~14.3) --
        # k=5 means "more than 5 standard deviations above this run's own
        # recent average," regardless of what that average actually is.
        # Diagnostic for "why is raw grad norm occasionally way above its
        # typical value" -- the existing [exec head grad norm]/main-vs-dir
        # summaries are aggregated over a whole rollout/epoch and only split
        # by execution-head, not by which specific parameter tensor (across
        # BOTH decision_net and execution_net, including encoders/trunk) is
        # actually driving an individual spike.
        _gn_spike_cfg = ppo_cfg.get("grad_norm_spike_k", 5.0)
        self.grad_norm_spike_k = float(_gn_spike_cfg) if _gn_spike_cfg is not None else None
        # Rolling window feeding the adaptive spike bar above -- shared
        # (same deque instance) across every call from _ppo_update, Phase
        # 1's BC loop, and DAgger's train_bc_epochs(), so "this run's recent
        # typical grad norm" reflects whichever of those actually ran most
        # recently, not three separate disconnected baselines.
        self._grad_norm_history: "collections.deque[float]" = collections.deque(
            maxlen=_GRAD_NORM_SPIKE_HISTORY_MAXLEN,
        )
        self.n_epochs = int(ppo_cfg.get("n_epochs", 4))
        # Linear LR warmup over the first N optimizer steps of EVERY PPO update
        # (same idea as bc.ppg_warmup_steps; see _ppo_update). 0 = off.
        self.lr_warmup_steps = max(0, int(ppo_cfg.get("lr_warmup_steps", 0)))
        _value_only_cont = ppo_cfg.get("value_only_continuation_epochs")
        self.value_only_continuation_epochs = int(_value_only_cont) if _value_only_cont is not None else self.n_epochs
        self.bc_only_continuation_epochs = int(bc_cfg.get("bc_only_continuation_epochs", 0))
        # Optional fixed coefficient for the BC-only continuation loop, decoupled
        # from the annealed aux_coeff (bc.aux_coeff_start/end/anneal_fraction).
        # None (default) = fall back to the annealed bc_coeff, matching prior
        # behaviour (see bc-only continuation comment below). Set a float in
        # ai_config.json's "bc.bc_only_continuation_coeff" to keep this
        # continuation loop active with a fixed weight even when annealing has
        # driven aux_coeff to 0.0 (e.g. to disable in-epoch BC pressure on the
        # main policy gradient while still nudging decision_net/execution_net
        # back toward demo behaviour with spare post-early-stop gradient budget).
        _bc_only_coeff_raw = bc_cfg.get("bc_only_continuation_coeff", None)
        self.bc_only_continuation_coeff = (
            float(_bc_only_coeff_raw) if _bc_only_coeff_raw is not None else None
        )
        self.minibatch_size = int(ppo_cfg.get("minibatch_size", 64))
        self.target_kl = float(ppo_cfg.get("target_kl", 0.02))
        # Dual-clip PPO (Ye et al. 2020, "Mastering Complex Control in MOBA
        # Games with Deep Reinforcement Learning" -- the OpenAI Five/Tencent
        # lineage): vanilla PPO's clip only bounds the surrogate objective
        # for POSITIVE advantage -- for NEGATIVE advantage, if ratio blows
        # up (the network drastically increases the probability of an
        # action its OWN advantage estimate says was bad), the unclipped
        # side (ratio*adv) keeps decreasing without bound as ratio grows,
        # and min() picks it the moment ratio exceeds (1+clip_range) --
        # giving an UNBOUNDED positive policy_loss contribution from that
        # one row. Confirmed live in this codebase: recurring "[ratio
        # spike]" log entries with negative adv imply per-sample policy_loss
        # in the hundreds to (rarely) tens of thousands. This adds a SECOND
        # floor, ONLY for adv<0: max(min(surr1,surr2), dual_clip_c*adv) --
        # bounds that row's worst case at dual_clip_c*|adv| without
        # touching the already-fine positive-advantage path at all. 0
        # (default, falsy) disables entirely -- byte-identical to vanilla
        # PPO. When enabled, dual_clip_c should be > 1 (paper/typical
        # practice: 2-3) so the floor sits below (more negative than, since
        # adv<0) the standard clip's own (1+clip_range)*adv bound -- picking
        # dual_clip_c <= 1+clip_range would instead tighten the ALREADY-
        # clipped region too, which is not the intent. See
        # _apply_dual_clip().
        self.dual_clip_c = float(ppo_cfg.get("dual_clip_c", 0.0) or 0.0)
        self.rollout_steps = int(ppo_cfg.get("rollout_steps", 2048))
        self.dir_l2_coef = float(ppo_cfg.get("dir_l2_coef", 0.01))
        self.move_dir_log_kappa_min = float(ppo_cfg.get("move_dir_log_kappa_min", ppo_cfg.get("dir_log_kappa_min", -2.0)))
        self.move_dir_log_kappa_max = float(ppo_cfg.get("move_dir_log_kappa_max", ppo_cfg.get("dir_log_kappa_max", 10.0)))
        self.move_dir_log_kappa_target = float(ppo_cfg.get("move_dir_log_kappa_target", ppo_cfg.get("dir_log_kappa_target", self.move_dir_log_kappa_max)))
        self.move_dir_log_kappa_reg_coef = float(ppo_cfg.get("move_dir_log_kappa_reg_coef", ppo_cfg.get("dir_log_kappa_reg_coef", 0.0)))
        self.kick_dir_log_kappa_min = float(ppo_cfg.get("kick_dir_log_kappa_min", ppo_cfg.get("dir_log_kappa_min", -2.0)))
        self.kick_dir_log_kappa_max = float(ppo_cfg.get("kick_dir_log_kappa_max", ppo_cfg.get("dir_log_kappa_max", 10.0)))
        self.kick_dir_log_kappa_target = float(ppo_cfg.get("kick_dir_log_kappa_target", ppo_cfg.get("dir_log_kappa_target", self.kick_dir_log_kappa_max)))
        self.kick_dir_log_kappa_reg_coef = float(ppo_cfg.get("kick_dir_log_kappa_reg_coef", ppo_cfg.get("dir_log_kappa_reg_coef", 0.0)))
        self.kick_dir_z_log_std_min = float(ppo_cfg.get("kick_dir_z_log_std_min", -5.0))
        self.kick_dir_z_log_std_max = float(ppo_cfg.get("kick_dir_z_log_std_max", 2.0))
        self.kick_dir_z_log_std_target = float(ppo_cfg.get("kick_dir_z_log_std_target", self.kick_dir_z_log_std_min))
        self.kick_dir_z_log_std_reg_coef = float(ppo_cfg.get("kick_dir_z_log_std_reg_coef", 0.0))
        self.ent_kick_power_weight = float(ppo_cfg.get("ent_kick_power_weight", 1.0))
        self.ent_kick_spin_weight = float(ppo_cfg.get("ent_kick_spin_weight", 1.0))
        self.ent_decision_weight = float(ppo_cfg.get("ent_decision_weight", 1.0))
        # Extra multiplier on the kick heads' entropy bonus in the PPO loss only
        # (kick gate + kick_dir + kick_power); 1.0 = today's behaviour exactly.
        self.ent_kick_weight = float(ppo_cfg.get("ent_kick_weight", 1.0))
        # Extra multiplier on the move_dir entropy bonus in the PPO loss only; 1.0 = today's behaviour exactly.
        # Only reaches move_dir_log_kappa (see _compute_entropy) -- slows the head's sharpening without touching the exec_move gate.
        self.ent_move_dir_weight = float(ppo_cfg.get("ent_move_dir_weight", 1.0))
        # Same idea as ent_move_dir_weight but for kick_power_log_std / kick_dir_log_kappa /
        # kick_dir_z_log_std specifically: the kick gate's contribution to their (E[kick]-weighted)
        # entropy bonus is detached, so these boosts only reach the named spread parameter, never
        # kick_logit. kick_dir's azimuth (kappa) and elevation (z) got their own independent weights
        # 2026-09-23 -- KickDirectionHead.entropy() sums the two, so a single combined weight here
        # (the old ent_kick_dir_only_weight) pushed BOTH kappa and z at once, not azimuth alone as
        # previously documented. See _compute_entropy. 1.0 = today's behaviour exactly for all three.
        self.ent_kick_power_only_weight = float(ppo_cfg.get("ent_kick_power_only_weight", 1.0))
        self.ent_kick_dir_azimuth_only_weight = float(ppo_cfg.get("ent_kick_dir_azimuth_only_weight", 1.0))
        self.ent_kick_dir_z_only_weight = float(ppo_cfg.get("ent_kick_dir_z_only_weight", 1.0))
        # Lower bound of the executed kick power fraction (0.0 = plain sigmoid).
        self.kick_power_floor = float(ppo_cfg.get("kick_power_floor", 0.0))
        # agent_plans/masked_action_training_plan.md: per-decision "could the action take effect" flags
        # (entities/action_opportunity.py). log_* = Phase-1 diagnostics only; mask_* = Phase-3 training masks
        # (kick / tackle gates on opportunity rows, kick_dir / kick_power on rows where a kick fired).
        self.mask_untriggered_actions = bool(ppo_cfg.get("mask_untriggered_actions", False))
        self.mask_rescale_by_opportunity_rate = bool(ppo_cfg.get("mask_rescale_by_opportunity_rate", False))
        self.log_action_opportunity_stats = bool(ppo_cfg.get("log_action_opportunity_stats", False)) or self.mask_untriggered_actions
        self.augment_n_slot_shuffles = int(ppo_cfg.get("augment_n_slot_shuffles", 0))
        # Log-only: KL between the identity and flip_y augmentation copies (each with its y-mirrored
        # output), from the pass _ppo_update already makes over the augmented batch. Needs augmentation.
        self._log_flip_consistency = bool(ppo_cfg.get("log_flip_consistency", False))
        if self.y_canonical and self.augment_n_slot_shuffles > 0:
            log.warning(
                "ppo.y_canonical is on but ppo.augment_n_slot_shuffles=%d: flip_y augmentation copies are exact "
                "duplicates of the originals under the y-canonical frame (wasted compute, double-counted rows). "
                "Set ppo.augment_n_slot_shuffles to 0.", self.augment_n_slot_shuffles,
            )
        self.rollout_eval_trials = int(ppo_cfg.get("rollout_eval_trials", 10))
        # Seeded eval config (see ai/eval/seeded_eval.py) -- rollout_eval_trials
        # above is now just the on/off gate (<=0 disables); the actual seed
        # list/repeat count come from ai_config.json's "eval" section so
        # pre-training eval and every rollout's eval use IDENTICAL scenarios.
        eval_cfg = cfg.get("eval", {})
        self._eval_seeds = default_eval_seeds(cfg)
        self._eval_repeats_per_seed = int(eval_cfg.get("eval_repeats_per_seed", 2))
        self._eval_n_parallel_workers = int(eval_cfg.get("eval_n_parallel_workers", 1))
        # Field-side fairness (see ai/eval/seeded_eval.py's swap_player_states):
        # deliberately a separate bool, NOT folded into eval_repeats_per_seed
        # -- repeats_per_seed always means "episodes per seed [per side]",
        # this just controls whether each seed also gets a second pass with
        # trainee/opponent roles swapped. Applies to every seeded eval in
        # this file (vs-rules, vs-immobile, rules-vs-rules baseline, and the
        # neural-snapshot comparisons below).
        self._eval_swap_sides = bool(eval_cfg.get("swap_sides", True))
        # Opt-in batched eval (see ai/eval/seeded_eval.py's
        # run_seeded_evaluation_batched): when True, every seeded eval in
        # this file (vs-rules, vs-immobile, rules-vs-rules baseline, and the
        # neural-snapshot comparisons) batches its episodes'
        # decision-network calls across eval_envs_per_process envs at a
        # time (per worker process), the same speedup
        # ppo.batched_rollout/batch_secondary_players gives real rollout
        # collection -- see that module's docstring. False (default) =
        # unchanged: each worker runs its episodes one at a time
        # (batch-of-1 network calls).
        self._eval_batched = bool(eval_cfg.get("batched_eval", False))
        self._eval_envs_per_process = int(eval_cfg.get("eval_envs_per_process", 8))
        # Set only by _train_parallel() (see ai/eval/eval_worker.py) for the
        # duration of that call -- a persistent eval worker pool that
        # _eval_vs_opponent_type() reuses instead of spinning up a fresh
        # multiprocessing.Pool on every periodic eval. None everywhere else
        # (single-process train(), evaluate.py CLI, standalone use).
        self._persistent_eval_workers = None
        # If the curriculum never actually trains against an immobile
        # opponent (ratio 0), the periodic eval-vs-immobile check is pure
        # sanity insurance, not a tracked metric -- cut its trial count down
        # (see _eval_vs_opponent_type) instead of spending the full
        # eval_seeds x eval_repeats_per_seed budget on it every rollout.
        self._phase1_opponent_immobile_ratio = float(
            cfg.get("curriculum", {}).get("phase1_opponent_immobile_ratio", 1.0)
        )
        # Rules-AI-vs-rules-AI baseline on the SAME eval seed set as the
        # periodic "[eval vs rules]" check -- computed once and cached (see
        # _compute_rules_vs_rules_baseline), since it never changes: same
        # seeds, same deterministic-ish rules-AI opponent logic.
        self._rules_vs_rules_baseline = None
        # --- Neural-snapshot self-eval (see _maybe_run_neural_snapshot_eval)
        # -- every_n<=0 disables the whole feature (default). No separate
        # in-memory snapshot ring: _save_checkpoint() already writes
        # checkpoint{N}.pt to disk every rollout regardless of this feature,
        # so "K rollouts back" is just checkpoint{current_count - K}.pt in
        # the same checkpoint_dir -- read fresh each time, nothing extra to
        # keep in memory. 'original' is always checkpoint1.pt.
        self._neural_snapshot_every_n = int(eval_cfg.get("neural_snapshot_eval_every_n_rollouts", 0))
        # Whole-number entries (1, 5, 20, ...) are a FIXED number of rollouts back, unchanged from
        # before. A non-integral entry (e.g. 0.5) is a FRACTION of the run's checkpoint count so far
        # -- see _resolve_snapshot_lookback -- so "0.5" always means "the checkpoint at the run's
        # current halfway point", a target that moves forward as the run progresses, unlike a fixed
        # lookback. Kept as whatever type json.load hands back (int or float); the distinction that
        # matters is made in _resolve_snapshot_lookback, not here.
        self._neural_snapshot_lookbacks: list[int | float] = list(
            eval_cfg.get("neural_snapshot_lookbacks", [1, 5, 20])
        )
        self._nn_snapshot_rollout_count = 0
        self.n_processes = int(ppo_cfg.get("n_processes", 1))
        self.worker_torch_threads = int(ppo_cfg.get("worker_torch_threads", 1))
        # Opt-in batched-rollout mode (see ai/ppo/batched_rollout_worker.py):
        # when True (and n_processes > 1), train() routes to
        # _train_batched_parallel() instead of _train_parallel() --
        # n_processes worker processes, each internally stepping
        # envs_per_process environments (total envs = n_processes *
        # envs_per_process) and batching their trainee decisions into one
        # network call per round. Default False/1 = today's unchanged
        # one-env-per-process behavior (envs_per_process ignored, implicitly
        # 1, so n_processes alone is both the process count and total envs).
        self.batched_rollout = bool(ppo_cfg.get("batched_rollout", False))
        self.envs_per_process = int(ppo_cfg.get("envs_per_process", 1))
        # Opt-in, only meaningful alongside batched_rollout=True: also
        # batches every secondary/opponent neural player's decision (e.g.
        # phase 1's self-play "neural remainder" opponent -- see
        # curriculum.phase1_opponent_neural_ratio, often the majority of
        # episodes) into its own one-call-per-round
        # BatchedEnvGroup.secondary_trainer batch, instead of each opponent
        # deciding unbatched one at a time. Passes the SAME trainer as the
        # primary (true self-play, identical weights, just batched for
        # speed) -- see BatchedEnvGroup's own docstring. False (default) =
        # unchanged: secondary players decide unbatched via their own
        # sample_action_fn fallback inside env.step().
        self.batch_secondary_players = bool(ppo_cfg.get("batch_secondary_players", False))
        # Batched-rollout only: bounds each worker's peak memory by having
        # it flush/send its results in chunks of roughly this many steps
        # instead of accumulating its full steps_per_worker budget (often
        # tens of thousands of rows) before one giant end-of-rollout
        # pickle+send -- see ai/ppo/batched_rollout_worker.py's "Chunked
        # streaming" docstring section. Fixes a real MemoryError observed in
        # production (many workers finishing near-simultaneously, each
        # trying to pickle its entire buffer at once). 0/falsy disables --
        # restores the old monolithic single-send-per-rollout behavior.
        self.batched_rollout_chunk_steps = int(ppo_cfg.get("batched_rollout_chunk_steps", 3000))
        # Episode-seed replay (batched-rollout only, see
        # ai/ppo/batched_rollout_worker.py's "Episode-seed replay" docstring
        # section and _train_batched_parallel()'s selection logic): when
        # enabled, the top episode_replay_top_fraction of each rollout's
        # episodes by mean(|advantage|) have their seeds re-queued for the
        # VERY NEXT rollout only (one-shot, not a persistent buffer) --
        # PLR-style re-exposure to whatever the policy found most
        # surprising/informative, plus a before/after mean-reward log line.
        self._episode_replay_enabled = bool(ppo_cfg.get("episode_replay_enabled", False))
        self._episode_replay_top_fraction = float(ppo_cfg.get("episode_replay_top_fraction", 0.05))
        # {seed: {"reward": float, "abs_adv": float}} for seeds queued at the
        # END of the previous rollout cycle -- the "before" side of the
        # before/after report, consumed (and replaced) once per cycle in
        # _train_batched_parallel().
        self._pending_replay_before: dict[int, dict] = {}
        # Held-out validation episodes (diagnostic only -- see
        # _split_train_val_episodes()/_eval_val_episode_losses() and
        # _ppo_update()'s per-epoch val_episode_policy_loss/
        # val_episode_value_loss log line). A random val_episode_fraction of
        # each rollout's COMPLETE episodes (split per track_id, so a
        # secondary neural opponent's interleaved rows can never straddle
        # the train/val boundary -- see RolloutBuffer.compute_gae's own
        # per-track segmentation note) are excluded from every minibatch
        # update this call, then re-evaluated (forward pass only, no
        # backward/optimizer step) under the CURRENT weights at the end of
        # every epoch. Purely informational -- unlike bc pretrain's/value
        # pretrain's val splits, nothing here ever early-stops or otherwise
        # changes what gets trained. 0.0 (default) disables entirely: no
        # split, no held-out rows, _ppo_update behaves exactly as before.
        self.val_episode_fraction = float(ppo_cfg.get("val_episode_fraction", 0.0))
        # Opt-in (default off -- costs one extra full no-grad pass over the
        # whole batch, and a second over val_batch when enabled): logs an
        # "epoch 0" block, in the EXACT same format as every real per-epoch
        # block below, evaluated under CURRENT weights before epoch 1's
        # first gradient step touches anything -- a clean ratio=1-exactly
        # baseline (policy_loss/KL/surrogate trivially ~0 by construction,
        # but value/entropy/percentiles are real and useful) to diff epoch
        # 1+ against. See _ppo_update's own comment at the call site.
        self.log_epoch_zero_baseline = bool(ppo_cfg.get("log_epoch_zero_baseline", False))
        # Dedicated RNG for choosing which episodes are held out each
        # update -- deliberately separate from self._aug_rng (batch
        # augmentation) so enabling/disabling one never shifts the other's
        # draw sequence.
        self._val_split_rng = random.Random()
        # Separate, deliberately DECOUPLED from n_processes above:
        # _collect_value_pretrain_rollout() (pretrain_value()'s own rollout
        # collection, also used by pretrain_combined()'s Phase 2/3 warm-up)
        # has no batched_rollout-aware path of its own -- it always spawns
        # this many plain one-env-per-process rollout_worker.py workers,
        # regardless of batched_rollout/envs_per_process. Sharing
        # n_processes directly would silently make this path spawn whatever
        # (likely much larger, batching-oriented) process count the main PPO
        # loop uses -- benchmarked this session: naive one-env-per-process
        # oversubscription well past physical core count makes throughput
        # WORSE, not better (25 procs measured 0.66x vs a 6-proc baseline),
        # so a separate, deliberately modest default here protects this
        # one-off warm-up stage from that regression.
        self.value_pretrain_n_processes = int(
            ppo_cfg.get("value_pretrain_n_processes", 9)
        )
        # Opt-in (default off, this repo's convention): use the batched
        # rollout workers (ai/ppo/batched_rollout_worker.py) for the value-
        # pretrain rollout instead of plain one-env-per-process
        # rollout_worker.py workers -- see _spawn_value_pretrain_workers().
        # Total envs = value_pretrain_n_processes * value_pretrain_envs_per_process.
        self.value_pretrain_batched_rollout = bool(
            ppo_cfg.get("value_pretrain_batched_rollout", False)
        )
        self.value_pretrain_envs_per_process = int(
            ppo_cfg.get("value_pretrain_envs_per_process", 12)
        )
        # Whole-episode chunk streaming for the batched value-pretrain workers
        # (rows collected between flushes; 0 = one big send at the end, the
        # old memory profile). Safe to leave on: flushes only ever send
        # COMPLETED episodes -- see batched_rollout_worker.py's "Chunked
        # streaming" docstring.
        self.value_pretrain_chunk_steps = int(
            ppo_cfg.get("value_pretrain_chunk_steps", 3000)
        )
        self._aug_rng = random.Random()
        self._bc_cfg = bc_cfg
        self._bc_dir_loss_w = float(bc_cfg.get("direction_loss_weight", 3.0))
        self._bc_dir_loss_mode = str(bc_cfg.get("direction_loss_mode", "cosine"))
        self._bc_dir_mag_reg_coef = float(bc_cfg.get("direction_mag_reg_coef", 0.0))
        self._bc_region_loss_w = float(bc_cfg.get("region_loss_weight", 1.0))
        self._bc_dec_label_smoothing = float(bc_cfg.get("dec_label_smoothing", 0.0))
        self._bc_exec_label_smoothing = float(bc_cfg.get("exec_label_smoothing", 0.0))
        self._bc_dec_weight = float(bc_cfg.get("bc_dec_weight", 1.0))
        self._bc_exec_weight = float(bc_cfg.get("bc_exec_weight", 1.0))
        # Whether the BC auxiliary loss (computed during PPO's own update
        # loop, not pretraining) still trains decision heads that are frozen
        # for this curriculum phase (self._ppo_lp_masked_heads, see
        # set_frozen_heads). True (default) = prior behaviour: BC keeps
        # pulling those heads -- and, since only the head's OWN weights are
        # frozen while the shared decision_net trunk is not, keeps pulling
        # the shared trunk too -- toward the demo labels even though PPO's
        # own policy gradient/entropy give them no signal this phase. Set
        # False to fully detach frozen heads from BC's loss as well, via
        # _bc_heads_for_loss().
        self._bc_trains_frozen_heads = bool(bc_cfg.get("bc_trains_frozen_heads", True))
        # pos_weight_*: None means "auto-compute from the training dataset at
        # load time" (see DemonstrationDataset.compute_pos_weights()). Set to
        # a float in config to override. Populated once pretrain_combined()
        # is given a dataset (see below); default 1.0 (no reweighting) until then.
        self._bc_pos_weight_kick_cfg = bc_cfg.get("pos_weight_kick")
        self._bc_pos_weight_tackle_attempt_cfg = bc_cfg.get("pos_weight_tackle_attempt")
        self._bc_pos_weight_kick = 1.0 if self._bc_pos_weight_kick_cfg is None else float(self._bc_pos_weight_kick_cfg)
        self._bc_pos_weight_tackle_attempt = (
            1.0 if self._bc_pos_weight_tackle_attempt_cfg is None else float(self._bc_pos_weight_tackle_attempt_cfg)
        )
        # Optional cap applied to the auto-computed (dataset-derived) pos_weight_*
        # ratios only — has no effect when pos_weight_kick/pos_weight_tackle_attempt
        # are set explicitly above. None = uncapped.
        _pos_weight_max_cfg = bc_cfg.get("pos_weight_max")
        self._bc_pos_weight_max = None if _pos_weight_max_cfg is None else float(_pos_weight_max_cfg)
        # Grad-norm clip for BC pretrain optimizers (bc_opt/demo_opt/repair_opt).
        # None (default) = fall back to ppo.max_grad_norm (prior behaviour).
        # Set bc.max_grad_norm to a float to use a BC-specific cap instead, or
        # explicitly null in config to disable clipping entirely during BC.
        _bc_max_grad_norm_cfg = bc_cfg.get("max_grad_norm", "__unset__")
        if _bc_max_grad_norm_cfg == "__unset__":
            self._bc_max_grad_norm: Optional[float] = self.max_grad_norm
        elif _bc_max_grad_norm_cfg is None:
            self._bc_max_grad_norm = None
        else:
            self._bc_max_grad_norm = float(_bc_max_grad_norm_cfg)
        self._downsample_trivial_enabled = bool(bc_cfg.get("downsample_trivial_enabled", False))
        self._downsample_trivial_frac_default = float(bc_cfg.get("downsample_trivial_frac_default", 0.5))
        self._downsample_trivial_frac_high_epoch = float(bc_cfg.get("downsample_trivial_frac_high_epoch", 0.65))
        self._downsample_trivial_epoch_threshold = int(bc_cfg.get("downsample_trivial_epoch_threshold", 5))
        self._downsample_trivial_cos_threshold = float(bc_cfg.get("downsample_trivial_cos_threshold", 0.98))
        self._downsample_trivial_exclude_radius_steps = int(bc_cfg.get("downsample_trivial_exclude_radius_steps", 5))
        self._secondary_weight = float(curriculum_cfg.get("secondary_weight", 1.0))
        self._value_pretrain_frozen_layers = int(bc_cfg.get("value_pretrain_frozen_layers", -1))
        self._value_pretrain_early_stop_patience = int(bc_cfg.get("value_pretrain_early_stop_patience", 5))
        self._value_pretrain_early_stop_min_delta = float(bc_cfg.get("value_pretrain_early_stop_min_delta", 1e-4))
        self._value_pretrain_weight_decay = float(bc_cfg.get("value_pretrain_weight_decay", 0.0))
        # Decoupled from ppo.minibatch_size (PPO's 12000 is tuned for the
        # clip/KL-early-stop dynamics of the policy update, not for a plain
        # supervised value-MSE regression). Used by pretrain_value() and
        # Phase 0's demo-return value warm-up in pretrain_combined() -- both
        # previously fell back silently to self.minibatch_size.
        self._value_pretrain_batch_size = int(bc_cfg.get("value_pretrain_batch_size", 512))
        # PPG-style KL-anchored value refit (see ppg_value_refit()'s own
        # docstring and ai/knowledge.md "PPG-style value refit"). A standalone
        # phase, not part of the ppo.* rollout-cycle loop -- keyed under "bc"
        # like every other pretrain-adjacent knob (value_pretrain_* above),
        # not "ppo".
        self._ppg_enabled = bool(bc_cfg.get("ppg_enabled", False))
        self._ppg_rollout_steps = int(bc_cfg.get("ppg_rollout_steps", 110000))
        self._ppg_epochs = int(bc_cfg.get("ppg_epochs", 6))
        self._ppg_lr = float(bc_cfg.get("ppg_lr", 1e-4))
        self._ppg_kl_coef = float(bc_cfg.get("ppg_kl_coef", 1.0))
        self._ppg_num_rollouts = int(bc_cfg.get("ppg_num_rollouts", 1))
        # Log-only: fraction of largest-squared-error rows excluded from the
        # extra "rmse_exNN" train/val figures in ppg_value_refit's lines (see
        # _trimmed_rmse). 0 disables the trimmed figure's trimming (== plain
        # rmse); never affects training or early stopping.
        self._ppg_rmse_trim_frac = min(max(float(bc_cfg.get("ppg_rmse_trim_frac", 0.10)), 0.0), 0.99)
        # Fraction of each ppg_value_refit rollout's complete episodes held out
        # as a val set (drives the val lines, early stopping and best-weights
        # restore). 0 = no val set: every episode trains, and with nothing to
        # early-stop or restore on, the refit just runs all its epochs.
        self._ppg_val_episode_fraction = min(max(float(bc_cfg.get("ppg_val_episode_fraction", 0.15)), 0.0), 0.9)
        # Linear LR warmup over the first N optimizer steps of EVERY refit
        # cycle (the anchor is re-snapshotted per cycle, so the KL starts at
        # exactly 0 each time and the value gradient overshoots before the KL
        # pull catches up). 0 = off (constant lr, the previous behavior).
        self._ppg_warmup_steps = max(0, int(bc_cfg.get("ppg_warmup_steps", 0)))
        # Diagnostic: keep a frozen copy of the networks as loaded at the START
        # of ppg_value_refit and, on every rollout's fresh data (before
        # training on it), score both it and the current weights against
        # pure-MC returns. The paired difference cancels the rollout-to-rollout
        # noise that swamps the raw baselines. Log-only; never affects training.
        self._ppg_compare_to_original = bool(bc_cfg.get("ppg_compare_to_original", False))
        # Diagnostic: per-cycle breakdown of where the value error lives
        # (outcome / opponent type / track / episode phase / calibration --
        # see _value_loss_breakdown), logged on fresh rows before training on
        # them; every ppg_diag_every_n_cycles-th cycle (and cycle 1) also dumps
        # the per-row arrays to <checkpoint_dir>/ppg_diag_cycle<k>.npz.
        self._ppg_loss_diagnostics = bool(bc_cfg.get("ppg_loss_diagnostics", False))
        self._ppg_diag_every_n_cycles = max(1, int(bc_cfg.get("ppg_diag_every_n_cycles", 5)))
        # Experiment: collect the refit's rollouts with the policy acting
        # deterministically (each head's mode/mean, no sampling), to test how
        # much of the value loss floor is the agent's own action-sampling
        # noise. Off for real use. Needs the batched collection path.
        self._ppg_rollout_deterministic = bool(bc_cfg.get("ppg_rollout_deterministic", False))
        # ppg_value_refit's OWN rollout-to-rollout episode-seed-replay loop (independent of,
        # but modeled on, the main PPO loop's episode_replay_enabled/_update_episode_replay).
        # See _comment_ppg_episode_replay_enabled in ai_config.json for the full mechanism.
        # Off by default: zero behavior/cost change to ppg_value_refit's existing cycles.
        self._ppg_episode_replay_enabled = bool(bc_cfg.get("ppg_episode_replay_enabled", False))
        # User-requested diagnostic (2026-10-01): is the value function actually
        # improving checkpoint-over-checkpoint, not just within one refit's own
        # epochs? Defaults True (unlike most opt-in bc.ppg_* flags) because it was
        # requested as a standing "always do this" behavior -- see ppg_value_refit's
        # rollout-1 block for the mechanism (evaluates the two most recently SAVED
        # checkpoint{N}.pt files' value heads, unmodified, on this call's freshly
        # collected rollout, alongside the current live weights' own epoch-0
        # baseline -- all three on IDENTICAL data no prior checkpoint trained on).
        self._ppg_log_prev_checkpoint_value_loss = bool(bc_cfg.get("ppg_log_prev_checkpoint_value_loss", True))
        # Experiment (untested): one-directional seeding of ppg_value_refit's fresh Adam
        # optimizer from the LIVE PPO self.optimizer's per-parameter variance estimate
        # (exp_avg_sq) -- see ppg_value_refit's "Adam seeding" comment for the mechanism
        # and [[project_kick_power_ppg_kl_investigation]] for the motivating finding (Adam's
        # own adaptive per-parameter step size was the untested lead hypothesis for why
        # move_dir's kappa dominates PPG-refit KL, and tripling ppg_kl_coef didn't fix it).
        # READ-ONLY from self.optimizer.state (cloned tensors) -- can never affect the live
        # PPO optimizer. Off by default: this is a new, unverified knob, not yet compared
        # against the existing cold-Adam behavior on a real run.
        self._ppg_seed_adam_variance = bool(bc_cfg.get("ppg_seed_adam_variance", False))
        # Interleaved PPG: run bursts of ppg_value_refit() mid-PPO instead of only via the
        # standalone --ppg-refit-only entry point. See _maybe_run_ppg_interleave() and
        # ai/knowledge.md "Interleaved PPG". Opt-in, off by default.
        self._ppg_interleave_enabled = bool(bc_cfg.get("ppg_interleave_enabled", False))
        self._ppg_interleave_trigger_file = bc_cfg.get("ppg_interleave_trigger_file") or None
        self._ppg_interleave_cadence_rollouts = max(0, int(bc_cfg.get("ppg_interleave_cadence_rollouts", 0)))
        self._ppg_interleave_num_rollouts = int(bc_cfg.get("ppg_interleave_num_rollouts", self._ppg_num_rollouts))
        self._ppg_interleave_n_steps = int(bc_cfg.get("ppg_interleave_n_steps", self._ppg_rollout_steps))
        self._rollouts_since_ppg_interleave = 0
        # flip_y consistency refit (consistency_refit(); CLI --consistency-refit-only) -- see
        # ai_config.json's _comment_consistency and ai/knowledge.md "flip_y consistency refit".
        self._consistency_epochs = int(bc_cfg.get("consistency_epochs", 5))
        self._consistency_lr = float(bc_cfg.get("consistency_lr", 1e-4))
        self._consistency_num_rollouts = int(bc_cfg.get("consistency_num_rollouts", 10))
        self._consistency_val_episode_fraction = float(bc_cfg.get("consistency_val_episode_fraction", 0.15))
        self._consistency_warmup_steps = max(0, int(bc_cfg.get("consistency_warmup_steps", 50)))
        self._consistency_logit_bound = float(bc_cfg.get("consistency_logit_bound", 6.907))
        self._consistency_kappa_deg = float(bc_cfg.get("consistency_kappa_deg", 22.0))
        # Diagnostic only: when set to a list, ppg_value_refit appends one
        # (epoch_idx, value_loss, total_kl) tuple per minibatch.
        self._ppg_minibatch_trace: Optional[list] = None
        self._bc_pretrain_early_stop_patience = int(bc_cfg.get("bc_pretrain_early_stop_patience", 0))
        self._bc_pretrain_early_stop_min_delta = float(bc_cfg.get("bc_pretrain_early_stop_min_delta", 1e-4))
        self._p0_early_stop_patience = int(bc_cfg.get("demo_pretrain_early_stop_patience", 0))
        self._p0_early_stop_min_delta = float(bc_cfg.get("demo_pretrain_early_stop_min_delta", 1e-4))
        self._demo_value_pretrain_epochs = int(bc_cfg.get("demo_value_pretrain_epochs", 10))
        self._demo_value_pretrain_lr = float(bc_cfg.get("demo_value_pretrain_lr", 4e-3))
        self._demo_value_pretrain_gamma = float(bc_cfg.get("demo_value_pretrain_gamma", 0.99))
        # Weight of value loss added to BC loss during BC epochs (0 = disabled)
        self._demo_value_bc_coef = float(bc_cfg.get("demo_value_bc_coef", 0.5))
        # Weight of value loss added to full BC pre-train loss in Phase 1 (both networks).
        # Falls back to demo_value_bc_coef for backward compatibility.
        self._bc_value_coef = float(bc_cfg.get("bc_value_coef", self._demo_value_bc_coef))
        # Weight of value loss added to decision-heads-only BC loss in Phase 0
        # (demo value pretrain). See pretrain_combined()'s Phase 0 block.
        self._phase0_value_coef = float(bc_cfg.get("phase0_value_coef", 1.0))
        # DAgger phase (pretrain_combined(), right after Phase 1 BC pretrain) --
        # see ai/ppo/dagger.py and ai_config.json's _comment_dagger.
        # dagger_iterations=0 (default) disables it entirely -- opt-in only.
        self._dagger_iterations = int(bc_cfg.get("dagger_iterations", 0))
        self._dagger_bc_epochs_per_iter = int(bc_cfg.get("dagger_bc_epochs_per_iter", 2))
        self._dagger_buffer_max_size = int(bc_cfg.get("dagger_buffer_max_size", 2000))
        self._dagger_max_replay_steps = int(bc_cfg.get("dagger_max_replay_steps", 500))
        self._dagger_episodes_per_iteration = int(bc_cfg.get("dagger_episodes_per_iteration", 1))
        self._dagger_n_workers = int(bc_cfg.get("dagger_n_workers", 1))
        # When True, pretrain_combined() trains ONLY the value head(s) --
        # decision_net and execution_net's policy heads are frozen (requires_grad
        # False) for the whole call, Phase 1's BC epoch loop and the BC repair
        # epochs are both skipped entirely (nothing to repair -- policy never
        # moves), and Phase 0 becomes the sole training loop, correct under both
        # separate_value_net=True/False since it already routes to the right
        # critic in each case. See _frozen_for_value_only_bc() and
        # pretrain_combined()'s docstring.
        self._bc_train_value_only = bool(bc_cfg.get("bc_train_value_only", False))

        lr = float(ppo_cfg.get("learning_rate", 3e-4))
        value_lr = float(ppo_cfg.get("value_learning_rate", lr))
        # Adam weight_decay (L2) applied ONLY to the value param group during PPO
        # (see ai_config.json's ppo.value_weight_decay comment). 0.0 (default) =
        # disabled, matching prior behaviour.
        value_weight_decay = float(ppo_cfg.get("value_weight_decay", 0.0))
        # Optional separate LR for the direction heads (move_direction, kick_direction,
        # move_dir_log_kappa, kick_dir_log_kappa, kick_dir_z_log_std). None (default) = share the "policy"
        # param group's LR, matching prior behaviour. Set ppo.direction_learning_rate
        # to give these params their own (typically smaller) step size, independent
        # of the rest of the policy — see also direction_max_grad_norm below, which
        # is the equivalent split for gradient-norm clipping.
        _dir_lr_raw = ppo_cfg.get("direction_learning_rate", None)
        direction_lr = float(_dir_lr_raw) if _dir_lr_raw is not None else lr
        # Optional separate grad-norm clip for the same direction-head params. None
        # (default) = share max_grad_norm, matching prior behaviour (all params
        # clipped together in one combined-norm group). Set ppo.direction_max_grad_norm
        # to isolate direction-head gradients into their own clip_grad_norm_() call so
        # a single outlier sample's large move_dir/kick_dir gradient can no longer
        # force a proportional shrink of every other head's gradient in the same step
        # (and vice versa) — see ai_trainer_knowledge.md "grad norm clipping" discussion.
        _dir_gn_raw = ppo_cfg.get("direction_max_grad_norm", None)
        self.direction_max_grad_norm = (
            float(_dir_gn_raw) if _dir_gn_raw is not None else self.max_grad_norm
        )
        if not inference_only:
            # Separate param group for the value head (+ its ai-type side channel)
            # with its own (typically higher) LR. During PPO, the policy LR is kept
            # very conservative (protects the BC-primed policy under 12x augmentation),
            # but the shared optimizer LR was starving the critic — PPO's per-minibatch
            # KL early-stop cuts gradient steps short based on the *policy's* KL, which
            # also cuts off the value head's updates for that rollout. A dedicated,
            # higher LR lets the value head keep learning at a reasonable pace even
            # when only 1-2 minibatches get through before early-stop fires.
            value_param_ids = set()
            value_params = []
            for name, p in execution_net.named_parameters():
                if name.startswith("value_head.") or name.startswith("value_ai_type_channel."):
                    value_params.append(p)
                    value_param_ids.add(id(p))
            # Separate param group for the direction heads (see direction_lr/
            # direction_max_grad_norm comments above). Named params only (not raw
            # nn.Parameter attributes like move_dir_log_kappa/kick_dir_log_kappa/
            # kick_dir_z_log_std, which named_parameters() also yields with their
            # attribute names). kick_dir_z_log_std is grouped with the rest of the
            # kick direction params (simplest default -- same LR/clip treatment as
            # kick_dir_log_kappa, not split into its own group).
            direction_param_ids = set()
            direction_params = []
            move_dir_param_ids = set()
            move_dir_params = []
            kick_dir_param_ids = set()
            kick_dir_params = []
            for name, p in execution_net.named_parameters():
                if id(p) in value_param_ids:
                    continue
                if name.startswith("move_direction.") or name == "move_dir_log_kappa":
                    move_dir_params.append(p)
                    move_dir_param_ids.add(id(p))
                elif name.startswith("kick_direction.") or name in ("kick_dir_log_kappa", "kick_dir_z_log_std"):
                    kick_dir_params.append(p)
                    kick_dir_param_ids.add(id(p))
            direction_param_ids = move_dir_param_ids | kick_dir_param_ids
            self.direction_param_ids = direction_param_ids
            # Frozen physics-dynamics encoders (ai/knowledge.md "Frozen
            # physics-dynamics encoders") -- owned only by decision_net, kept
            # requires_grad=False by their loaders. Excluded from the
            # optimizer entirely (not even a zero-LR param group), same
            # treatment decision_net.value_head gets under the "single value
            # head convention" -- no-op when the feature is disabled (the
            # submodules don't exist, so named_parameters() yields nothing
            # with these prefixes).
            physics_encoder_param_ids = {
                id(p) for name, p in decision_net.named_parameters()
                if name.startswith("ball_physics_encoder.") or name.startswith("player_physics_encoder.")
            }
            policy_params = [
                p for p in list(decision_net.parameters()) + list(execution_net.parameters())
                if id(p) not in value_param_ids and id(p) not in direction_param_ids
                and id(p) not in physics_encoder_param_ids
            ]
            # When separate_value_net is enabled, execution_net's own
            # value_head/value_ai_type_channel are dead weight (never used --
            # self.value_net is the sole critic below), so exclude them from
            # the main optimizer entirely rather than training an unused head.
            _kick_dir_lr_raw = ppo_cfg.get("kick_direction_learning_rate", None)
            kick_dir_lr = float(_kick_dir_lr_raw) if _kick_dir_lr_raw is not None else direction_lr
            if self.separate_value_net:
                self.optimizer = torch.optim.Adam(
                    [
                        {"params": policy_params, "lr": lr, "name": "policy"},
                        {"params": move_dir_params, "lr": direction_lr, "name": "move_direction"},
                        {"params": kick_dir_params, "lr": kick_dir_lr, "name": "kick_direction"},
                    ],
                    eps=1e-5,
                )
                self.value_net_optimizer = torch.optim.Adam(
                    self.value_net.parameters(), lr=value_lr, eps=1e-5,
                    weight_decay=value_weight_decay,
                )
            else:
                self.optimizer = torch.optim.Adam(
                    [
                        {"params": policy_params, "lr": lr, "name": "policy"},
                        {"params": value_params, "lr": value_lr, "name": "value", "weight_decay": value_weight_decay},
                        {"params": move_dir_params, "lr": direction_lr, "name": "move_direction"},
                        {"params": kick_dir_params, "lr": kick_dir_lr, "name": "kick_direction"},
                    ],
                    eps=1e-5,
                )
        else:
            self.optimizer = None  # type: ignore[assignment]  # not needed for inference
            self.direction_param_ids = set()

        self.decision_net.to(self.device)
        self.execution_net.to(self.device)
        if self.value_net is not None:
            self.value_net.to(self.device)

        self._total_steps = 0
        self._checkpoint_count = 0  # sequential counter for checkpoint{N}.pt naming
        self._log_file_handler: Optional[logging.FileHandler] = None  # see _rotate_log_file()
        if self.checkpoint_dir is not None:
            self._rotate_log_file()
        # _ppo_lp_masked_heads (which decision heads are excluded from the PPO
        # log_prob/entropy computation) is a computed @property below, derived
        # live from each head's actual requires_grad state -- nothing to
        # initialise here. See the property's own docstring for why.

        # --- kick_spin: permanently frozen (see agent_plans/spin_implementation_plan.md
        # section 0) ---
        # kick_spin still has ZERO physical effect (apply_nn_action.py hardcodes
        # Vector3.zero() for spin regardless of what this head outputs -- see
        # the plan doc's "Status quo"), so training it wastes gradient budget
        # for no benefit. Unlike decision_net.value_head above, this is NOT a
        # "permanently unused, kept for checkpoint compat" situation -- kick_spin
        # is meant to become live once real spin physics lands (see the plan
        # doc), so this flag (and the masking it gates in _compute_log_prob/
        # _recompute_log_prob/_per_head_new_log_probs/_compute_entropy, and the
        # deterministic-mode diagnostic block in _ppo_update) needs to be
        # explicitly reverted at that point -- see the plan doc's "How to
        # re-enable" checklist. Freezing requires_grad here alone is NOT
        # sufficient to make kick_spin's PPO contribution zero (its output
        # still drifts because the shared trunk keeps moving under other,
        # live heads' gradients) -- the masking flag below is the load-bearing
        # half, this freeze is the other (BC-gradient-blocking) half.
        self._kick_spin_frozen: bool = True
        for p in self.execution_net.kick_spin.parameters():
            p.requires_grad_(False)
        self.execution_net.kick_spin_log_std.requires_grad_(False)

        # --- Single value head convention ---
        # Commit to execution_net.value_head as the ONLY trained critic (or, when
        # separate_value_net is enabled, self.value_net.value_head instead -- see
        # above). decision_net.value_head is kept (checkpoint/state_dict compat, and
        # in case two-critic training is revisited later) but is permanently
        # frozen and excluded from every value loss. This avoids the
        # averaging-vs-independent-fit inconsistency that existed when both
        # heads were trained (Phase 0/1 fit them independently, pretrain_value()/
        # PPO only fit their average, letting the two heads silently diverge).
        for p in self.decision_net.value_head.parameters():
            p.requires_grad_(False)
        if self.separate_value_net:
            # execution_net's own value_head/value_ai_type_channel are unused
            # (self.value_net is the sole critic) -- freeze them too so any
            # stray forward/backward through execution_net never trains them.
            for p in self.execution_net.value_head.parameters():
                p.requires_grad_(False)
            for p in self.execution_net.value_ai_type_channel.parameters():
                p.requires_grad_(False)





    def set_frozen_heads(self, frozen_head_names: list[str]) -> None:
        """Freeze named decision-network heads so PPO gradients skip them.

        Sets ``requires_grad=False`` on the parameters of each named
        ``nn.Module`` attribute on ``decision_net``.  The Adam optimizer
        already holds references to these params; it simply won't step them
        because they receive no gradient.

        BC pre-training runs its own optimizer over all parameters and is
        unaffected — freezing only applies during the PPO update loop.

        ``_ppo_lp_masked_heads`` (which heads are excluded from the PPO
        importance-ratio log_prob and the entropy bonus) is a computed
        property derived from live ``requires_grad`` state, not tracked
        here — freezing a head via this method (or any other mechanism) is
        automatically reflected there. See that property's docstring.

        Args:
            frozen_head_names: Names of ``nn.Module`` attributes on
                ``decision_net``, e.g. ``["shoot_logit", "pass_logit"]``.
        """
        for name in frozen_head_names:
            module = getattr(self.decision_net, name, None)
            if module is None:
                log.warning(f"set_frozen_heads: '{name}' not found on decision_net — skipping")
                continue
            for p in module.parameters():
                p.requires_grad_(False)
            log.info(f"Frozen decision_net.{name}")
        masked_now = self._ppo_lp_masked_heads
        if masked_now:
            log.warning(
                "PPO log_prob masking ACTIVE — the following decision heads are excluded "
                "from the importance ratio (frozen for this curriculum phase, no reward "
                "signal): %s.  Their BC aux loss is %s.",
                ", ".join(sorted(masked_now)),
                "still computed normally" if self._bc_trains_frozen_heads
                else "ALSO detached (bc.bc_trains_frozen_heads=false)",
            )

    @property
    def _ppo_lp_masked_heads(self) -> frozenset[str]:
        """Which of the 7 decision heads in ``_LP_HEAD_NAMES`` are currently
        fully frozen (every parameter's ``requires_grad`` is False).

        Computed live from ``decision_net``'s actual parameter state rather
        than tracked as a separate flag — freezing a head (via
        ``set_frozen_heads`` or any other mechanism) is automatically
        reflected here, and unfreezing it later automatically un-masks it
        too, so this can never drift out of sync with what's actually
        frozen the way a manually-updated set could (e.g. a head frozen
        without going through ``set_frozen_heads``, or a curriculum config
        edited without remembering to keep some separate list in sync).
        """
        return frozenset(
            name for name in _LP_HEAD_NAMES
            if (module := getattr(self.decision_net, name, None)) is not None
            and all(not p.requires_grad for p in module.parameters())
        )


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

    # -----------------------------------------------------------------------
    # Main training entry point
    # -----------------------------------------------------------------------

    def train(self, env, total_steps: int, bc_label_fn=None, phase_id: Optional[int] = None) -> None:
        """Run PPO for ``total_steps`` decision-interval steps.

        Args:
            env: ScenarioEnv (or any env with reset()/step() returning
                 ObservationBatch, float, bool, info). Ignored (may be None)
                 when ``ppo.n_processes > 1`` -- each rollout worker
                 builds its own env from ``phase_id`` instead.
            total_steps: Total number of decision steps to train for.
            bc_label_fn: Optional callable ``(player, match) -> BCLabel``
                (e.g. ``curriculum.envs.bc_label_fn_for_phase_player()`` --
                NOT ``bc_label_fn_for_phase()``, which returns the OTHER,
                ``(env) -> BCLabel`` convention). Threaded into
                ``env.bc_label_fn`` before ``env.reset()`` so
                ``NeuralPlayerAI.act()`` computes it internally, at the same
                instant it encodes the observation -- see that method's own
                ``bc_label_fn`` docstring for why this must happen there and
                not via a separate call after ``env.step()`` returns. When
                provided, a BC supervision label is collected at each step
                and stored in the rollout buffer so it can be used as an
                auxiliary loss during the PPO update (weight controlled by
                ``bc.aux_coeff_start/end`` in ai_config.json).  Both the
                decision network's Bernoulli heads and the execution
                network's move_direction and sprint are supervised.
            phase_id: Curriculum phase id, required only when
                ``ppo.n_processes > 1`` (each rollout worker rebuilds its
                own env from this id via ``curriculum.envs.build_env``). See
                ai/ppo/rollout_worker.py.
        """
        if self.n_processes > 1:
            if phase_id is None:
                raise ValueError(
                    "ppo.n_processes > 1 requires train(phase_id=...) so "
                    "each rollout worker can rebuild its own environment."
                )
            from footballcoach.ai.curriculum.phases import PHASES_BY_ID
            max_episode_s = float(PHASES_BY_ID[phase_id].env_kwargs.get("max_episode_s", 120.0))
            if self.batched_rollout:
                self._train_batched_parallel(total_steps, phase_id, max_episode_s)
            else:
                self._train_parallel(total_steps, phase_id, max_episode_s)
            return

        # Inject the sampling function so ScenarioEnv assigns NeuralPlayerAI to
        # the trainee (and secondary players when not in rules-based mode).
        env.sample_action_fn = self._sample_action
        # Threaded straight into NeuralPlayerAI's own constructor (see
        # ScenarioEnv.reset()) so the label is computed INSIDE act(), at the
        # same instant as the observation -- see this method's own
        # bc_label_fn docstring above.
        env.bc_label_fn = bc_label_fn
        if self.checkpoint_dir is not None and hasattr(env, "match_log_dir"):
            env.match_log_dir = self.checkpoint_dir / "match_logs"

        obs = env.reset()
        buffer = RolloutBuffer()
        steps_this_rollout = 0
        rollout_progress = ProgressReporter(self.rollout_steps, prefix="  [rollout] ", live=True)
        episode_rewards: list[float] = []
        episode_reward_accum = 0.0
        secondary_episode_rewards: list[float] = []
        secondary_episode_reward_accum = 0.0
        episode_outcomes_vs_rules: list[str] = []
        episode_outcomes_vs_neural: list[str] = []
        episode_outcomes_vs_immobile: list[str] = []
        rollout_components: dict[str, float] = {}
        # Per-episode reward components for statistics (mean/std/min/max per type).
        episode_comp_accum: dict[str, float] = {}   # accumulates within current episode
        episode_comp_list: list[dict[str, float]] = []  # one entry per completed episode
        # Episode durations in sim-seconds (StepInfo.ticks_elapsed * env dt), for
        # the mean/std episode-length line alongside the other per-rollout logs.
        episode_durations_s: list[float] = []
        episode_kick_counts: list[int] = []
        episode_armed_kick_counts: list[int] = []
        episode_kick_accum = 0
        episode_armed_kick_accum = 0

        log.info(f"PPO training started: steps_so_far={self._total_steps:,}  target={self._total_steps + total_steps:,}  (+{total_steps:,} this run)")

        rollout_start = time.perf_counter()

        # progress is relative to THIS call's step budget (0.0 at the start of
        # this train() invocation, 1.0 once total_steps more decision-steps
        # have been collected) -- NOT self._total_steps / total_steps, which
        # would desync every schedule (bc.aux_coeff anneal in particular) on
        # any resumed run (--latest/--checkpoint/--pretrain-from-checkpoint)
        # since self._total_steps starts wherever the loaded checkpoint left
        # off instead of 0. See ai_trainer_knowledge.md "resume progress bug".
        _steps_at_call_start = self._total_steps
        target_steps = _steps_at_call_start + total_steps

        while self._total_steps < target_steps:
            progress = (self._total_steps - _steps_at_call_start) / total_steps

            # --- Collect one decision step ---
            # NeuralPlayerAI fires inside env.step() — no pre-sampling needed here.
            next_obs, reward, done, info = env.step()
            if info is not None:
                episode_kick_accum += info.trainee_kicks_this_step
                episode_armed_kick_accum += info.trainee_armed_kicks_this_step
            if done:
                episode_kick_counts.append(episode_kick_accum)
                episode_armed_kick_counts.append(episode_armed_kick_accum)
                episode_kick_accum = 0
                episode_armed_kick_accum = 0

            # Read transition data from the trainee's NeuralPlayerAI
            tr = env.last_trainee_transition
            if tr is None:
                # No neural decision this step (e.g. rules-based episode where
                # trainee is immobile); skip storing a transition.
                if done:
                    episode_rewards.append(episode_reward_accum)
                    episode_reward_accum = 0.0
                    obs = env.reset()
                else:
                    obs = next_obs
                continue

            action = tr["action"]
            log_prob = tr["log_prob"]
            value = tr["value"]
            raw_exec_samples = tr["raw_exec"]
            # Computed INSIDE NeuralPlayerAI.act() (see env.bc_label_fn
            # wiring above), at the same instant as tr["obs"] -- already a
            # numpy array (or None if bc_label_fn wasn't configured), no
            # further conversion needed here.
            bc_label_arr = tr.get("bc_label")

            # Store trainee transition
            buffer.add(
                obs=tr["obs"],
                action=_action_to_numpy(action, raw_exec_samples),
                log_prob=float(log_prob),
                value=float(value),
                reward=reward,
                done=1.0 if done else 0.0,
                bc_label=bc_label_arr,
                head_log_probs=tr.get("head_log_probs"),
                reward_comps=dict(getattr(env, "last_reward_components", {})),
                step_outcome=(info.trial_outcome or "") if (done and info is not None) else "",
            )

            # Store secondary player transitions (shared-weight training data)
            for sec in getattr(env, "last_secondary_results", []):
                buffer.add(
                    obs=sec["obs"],  # already numpy dict from NeuralPlayerAI
                    action=_action_to_numpy(sec["action"], sec["raw_exec"]),
                    log_prob=sec["log_prob"],
                    value=sec["value"],
                    reward=sec["reward"],
                    done=sec["done"],
                    bc_label=None,
                    # Secondary players use the same NeuralPlayerAI as the
                    # trainee (see rules_ai.py's last_transition, which sets
                    # "head_log_probs" unconditionally), so this is real
                    # data, not a placeholder -- omitting it here (as this
                    # call did until now) silently fell back to
                    # RolloutBuffer.add()'s all-zero default for every
                    # secondary/opponent row, corrupting every per-head KL/
                    # policy_loss diagnostic (which reads batch["head_log_probs"]
                    # as the "old" baseline) for the majority of rows in any
                    # curriculum phase with a nonzero phase1_opponent_neural_ratio
                    # -- the scalar total policy_loss/KL was never affected
                    # (it reads log_prob, which WAS always passed correctly
                    # above), only the per-head breakdown lines.
                    head_log_probs=sec.get("head_log_probs"),
                    weight=self._secondary_weight,
                    track_id=sec["player_id"],
                    # Shared env-level episode outcome (same for every
                    # player on the pitch that tick) -- see
                    # scenario_env.py's last_secondary_results["step_outcome"]
                    # and ai/knowledge.md "unknown outcome bucket" for why
                    # omitting this silently corrupted value_mse_by_outcome().
                    step_outcome=sec.get("step_outcome", ""),
                )
                secondary_episode_reward_accum += sec["reward"]
                if sec["done"]:
                    secondary_episode_rewards.append(secondary_episode_reward_accum)
                    secondary_episode_reward_accum = 0.0
                steps_this_rollout += 1
                self._total_steps += 1

            episode_reward_accum += reward
            for _k, _v in getattr(env, "last_reward_components", {}).items():
                rollout_components[_k] = rollout_components.get(_k, 0.0) + _v
                episode_comp_accum[_k] = episode_comp_accum.get(_k, 0.0) + _v
            self._total_steps += 1
            steps_this_rollout += 1
            rollout_progress.update(steps_this_rollout, postfix=f"eps={len(episode_rewards)}")

            if done:
                episode_rewards.append(episode_reward_accum)
                episode_reward_accum = 0.0
                if episode_comp_accum:
                    episode_comp_list.append(dict(episode_comp_accum))
                episode_comp_accum = {}
                if info is not None and info.trial_outcome is not None:
                    if info.is_rules_episode:
                        episode_outcomes_vs_rules.append(info.trial_outcome)
                    elif info.is_immobile_episode:
                        episode_outcomes_vs_immobile.append(info.trial_outcome)
                    else:
                        episode_outcomes_vs_neural.append(info.trial_outcome)
                if info is not None:
                    episode_durations_s.append(info.ticks_elapsed * env._dt_s)
                obs = env.reset()
            else:
                obs = next_obs

            # --- PPO update when rollout buffer is full ---
            if steps_this_rollout >= self.rollout_steps:
                rollout_time = time.perf_counter() - rollout_start
                steps_per_sec = self.rollout_steps / max(rollout_time, 1e-6)

                # Bootstrap value for last state (per track — see docstring)
                last_values = self._bootstrap_last_values(env, next_obs, buffer)

                advantages, returns = buffer.compute_gae(self.gamma, self.lam, last_values)
                batch = buffer.as_tensors(advantages, returns)

                # Per-component step-level stats (pre-augmentation batch).
                # For each reward component: at the steps where it fired,
                # what are the return/advantage/TD stats?  Uses the raw
                # (non-normalised) advantages and returns from GAE.
                _comp_step_stats: dict[str, dict] = {}
                _rcomp_raw = batch.get("reward_comps_raw", [])
                if _rcomp_raw:
                    _rets_np  = batch["returns"].numpy()
                    _advs_np  = batch["advantages"].numpy()
                    _vals_np  = batch["values"].numpy()
                    _td_np    = _rets_np - _vals_np
                    for _ck, _clabel in REWARD_COMP_LABELS:
                        _cvals = np.array([float(_d.get(_ck, 0.0)) for _d in _rcomp_raw])
                        _mask  = _cvals != 0.0
                        _td_m  = _td_np[_mask] if _mask.any() else np.array([])
                        _comp_step_stats[_ck] = {
                            "mean": float(_cvals.mean()),
                            "std":  float(_cvals.std()),
                            "count": int(_mask.sum()),
                            "mean_ret": float(_rets_np[_mask].mean()) if _mask.any() else float("nan"),
                            "std_ret":  float(_rets_np[_mask].std())  if _mask.any() else float("nan"),
                            "mean_gae": float(_advs_np[_mask].mean()) if _mask.any() else float("nan"),
                            "mean_sq_td": float((_td_m ** 2).mean()) if _mask.any() else float("nan"),
                            "mean_abs_td": float(np.abs(_td_m).mean()) if _mask.any() else float("nan"),
                            "p95_td": float(np.percentile(np.abs(_td_m), 95)) if _mask.any() else float("nan"),
                            "label": _clabel,
                        }

                update_start = time.perf_counter()
                metrics = self._ppo_update(batch, progress)
                update_time = time.perf_counter() - update_start

                self._log_rollout_summary(
                    metrics=metrics,
                    steps_per_sec=steps_per_sec,
                    episode_rewards=episode_rewards,
                    secondary_episode_rewards=secondary_episode_rewards,
                    episode_outcomes_vs_rules=episode_outcomes_vs_rules,
                    episode_outcomes_vs_immobile=episode_outcomes_vs_immobile,
                    episode_outcomes_vs_neural=episode_outcomes_vs_neural,
                    rollout_components=rollout_components,
                    episode_comp_list=episode_comp_list,
                    episode_durations_s=episode_durations_s,
                    comp_step_stats=_comp_step_stats,
                    n_reward_comp_steps=len(_rcomp_raw),
                    episode_kick_counts=episode_kick_counts,
                    episode_armed_kick_counts=episode_armed_kick_counts,
                )
                episode_rewards.clear()
                secondary_episode_rewards.clear()
                episode_outcomes_vs_rules.clear()
                episode_outcomes_vs_immobile.clear()
                episode_outcomes_vs_neural.clear()
                rollout_components.clear()
                episode_comp_list.clear()
                episode_durations_s.clear()
                episode_kick_counts.clear()
                episode_armed_kick_counts.clear()

                buffer.clear()
                steps_this_rollout = 0
                rollout_progress = ProgressReporter(self.rollout_steps, prefix="  [rollout] ", live=True)
                rollout_start = time.perf_counter()

                # Save checkpoint
                if self.checkpoint_dir is not None:
                    self._save_checkpoint(self._total_steps)

                # Quick periodic eval vs rules-based AI (always, regardless of training opponent)
                if self.rollout_eval_trials > 0:
                    self._eval_vs_rules(env.max_episode_s)
                self._maybe_run_neural_snapshot_eval(env.max_episode_s)
                self._maybe_run_ppg_interleave(env, phase_id)

        # Always save a final checkpoint so the result of the run is not lost
        # even if total_steps is not an exact multiple of rollout_steps.
        if self.checkpoint_dir is not None:
            self._save_checkpoint(self._total_steps)
            log.info("Final checkpoint saved.")

        log.info(f"Training complete. Total steps: {self._total_steps:,}")


    def _train_parallel(self, total_steps: int, phase_id: int, max_episode_s: float) -> None:
        """Multi-process rollout collection path (``ppo.n_processes > 1``).

        Each worker runs its own full env + local policy copy (see
        ai/ppo/rollout_worker.py) — no batched/centralized inference, no
        change to the single-process sampling code. Per-worker GAE is
        computed independently (each worker returns its own trailing
        bootstrap value) before batches are concatenated, since concatenating
        raw transitions across worker boundaries first would corrupt
        advantage estimates by treating unrelated episodes as one trajectory.
        """
        import multiprocessing
        import multiprocessing.connection

        from footballcoach.ai.ppo.rollout_worker import spawn_workers, close_workers

        n_workers = self.n_processes
        steps_per_worker = max(1, self.rollout_steps // n_workers)
        base_seed = random.randint(0, 2**31 - 1)
        log.info(
            f"PPO parallel training started: {n_workers} worker(s), "
            f"~{steps_per_worker} steps/worker/rollout, "
            f"steps_so_far={self._total_steps:,}  target={self._total_steps + total_steps:,}"
        )
        # Shared aggregate step counter across all workers, polled below to
        # render ONE live rollout-collection bar instead of leaving the
        # terminal silent for the whole blocking collection window (see
        # _collect_value_pretrain_rollout()'s identical pattern/reasoning).
        # Must be created with the SAME "spawn" context spawn_workers() uses
        # and passed at process-creation time.
        _ctx = multiprocessing.get_context("spawn")
        _progress_value = _ctx.Value("l", 0)
        workers = spawn_workers(
            phase_id, n_workers, base_seed, self.separate_value_net, self.worker_torch_threads,
            progress_value=_progress_value,
        )
        # Persistent eval worker pool (see ai/eval/eval_worker.py's module
        # docstring): spawned once here, alongside the rollout workers,
        # rather than _eval_vs_opponent_type() spinning a fresh
        # multiprocessing.Pool on every single periodic eval call -- that
        # used to mean every rollout cycle re-paid a full torch/
        # footballcoach re-import per eval worker for no reason. Stored on
        # self so _eval_vs_opponent_type() (called from deep inside the loop
        # below) can reach it without threading it through every call.
        eval_workers = None
        if self._eval_n_parallel_workers > 1:
            from footballcoach.ai.eval.eval_worker import spawn_eval_workers
            eval_workers = spawn_eval_workers(
                self._eval_n_parallel_workers, self.separate_value_net, self.worker_torch_threads,
            )
        self._persistent_eval_workers = eval_workers
        try:
            _steps_at_call_start = self._total_steps
            target_steps = _steps_at_call_start + total_steps

            while self._total_steps < target_steps:
                progress = (self._total_steps - _steps_at_call_start) / total_steps
                rollout_start = time.perf_counter()

                # Broadcast current weights before collecting (workers start
                # this rollout on the policy as of the end of the previous
                # PPO update -- standard PPO already tolerates this since the
                # importance ratio corrects for an "old" behaviour policy).
                dec_state = self.decision_net.state_dict()
                exec_state = self.execution_net.state_dict()
                val_state = self.value_net.state_dict() if self.value_net is not None else None
                for w in workers:
                    w.set_weights(dec_state, exec_state, val_state)

                for w in workers:
                    w.collect(steps_per_worker, progress)
                _agg_progress = ProgressReporter(
                    steps_per_worker * n_workers,
                    prefix=f"  [rollout] ({n_workers} workers): ", live=True,
                )
                _pending = {w.conn: w for w in workers}
                while _pending:
                    ready = multiprocessing.connection.wait(list(_pending.keys()), timeout=0.2)
                    _agg_progress.update(int(_progress_value.value))
                    for conn in ready:
                        _pending.pop(conn, None)
                results = [w.recv_result() for w in workers]

                worker_batches = []
                episode_rewards: list[float] = []
                secondary_episode_rewards: list[float] = []
                episode_outcomes_vs_rules: list[str] = []
                episode_outcomes_vs_neural: list[str] = []
                episode_outcomes_vs_immobile: list[str] = []
                episode_comp_list: list[dict[str, float]] = []
                episode_durations_s: list[float] = []
                episode_kick_counts: list[int] = []
                episode_armed_kick_counts: list[int] = []
                rollout_components: dict[str, float] = {}
                for r in results:
                    advantages, returns = r["buffer"].compute_gae(self.gamma, self.lam, r["last_value"])
                    worker_batches.append(r["buffer"].as_tensors(advantages, returns))
                    stats = r["stats"]
                    episode_rewards.extend(stats["episode_rewards"])
                    secondary_episode_rewards.extend(stats["secondary_episode_rewards"])
                    episode_outcomes_vs_rules.extend(stats["episode_outcomes_vs_rules"])
                    episode_outcomes_vs_neural.extend(stats["episode_outcomes_vs_neural"])
                    episode_outcomes_vs_immobile.extend(stats["episode_outcomes_vs_immobile"])
                    episode_comp_list.extend(stats["episode_comp_list"])
                    episode_durations_s.extend(stats["episode_durations_s"])
                    episode_kick_counts.extend(stats.get("episode_kick_counts", []))
                    episode_armed_kick_counts.extend(stats.get("episode_armed_kick_counts", []))
                    # rollout_components (rollout-total reward breakdown) is summed
                    # from each worker's completed-episode component dicts, since
                    # workers don't expose an in-progress per-step accumulator
                    # across the process boundary -- unlike the single-process
                    # path's rollout_components (accumulated every step, including
                    # partial/in-flight episodes), this only reflects EPISODES
                    # THAT COMPLETED within this rollout's collection window.
                    for ep in stats["episode_comp_list"]:
                        for _k, _v in ep.items():
                            rollout_components[_k] = rollout_components.get(_k, 0.0) + _v

                batch = _merge_worker_batches(worker_batches)
                n_collected = int(batch["rewards"].shape[0])
                self._total_steps += n_collected
                rollout_time = time.perf_counter() - rollout_start
                steps_per_sec = n_collected / max(rollout_time, 1e-6)

                # Per-component step-level stats -- identical computation to
                # the single-process path (see train()'s "_comp_step_stats").
                _comp_step_stats: dict[str, dict] = {}
                _rcomp_raw = batch.get("reward_comps_raw", [])
                if _rcomp_raw:
                    _rets_np = batch["returns"].numpy()
                    _advs_np = batch["advantages"].numpy()
                    _vals_np = batch["values"].numpy()
                    _td_np = _rets_np - _vals_np
                    for _ck, _clabel in REWARD_COMP_LABELS:
                        _cvals = np.array([float(_d.get(_ck, 0.0)) for _d in _rcomp_raw])
                        _mask = _cvals != 0.0
                        _td_m = _td_np[_mask] if _mask.any() else np.array([])
                        _comp_step_stats[_ck] = {
                            "mean": float(_cvals.mean()),
                            "std": float(_cvals.std()),
                            "count": int(_mask.sum()),
                            "mean_ret": float(_rets_np[_mask].mean()) if _mask.any() else float("nan"),
                            "std_ret": float(_rets_np[_mask].std()) if _mask.any() else float("nan"),
                            "mean_gae": float(_advs_np[_mask].mean()) if _mask.any() else float("nan"),
                            "mean_sq_td": float((_td_m ** 2).mean()) if _mask.any() else float("nan"),
                            "mean_abs_td": float(np.abs(_td_m).mean()) if _mask.any() else float("nan"),
                            "p95_td": float(np.percentile(np.abs(_td_m), 95)) if _mask.any() else float("nan"),
                            "label": _clabel,
                        }

                metrics = self._ppo_update(batch, progress)

                self._log_rollout_summary(
                    metrics=metrics,
                    steps_per_sec=steps_per_sec,
                    episode_rewards=episode_rewards,
                    secondary_episode_rewards=secondary_episode_rewards,
                    episode_outcomes_vs_rules=episode_outcomes_vs_rules,
                    episode_outcomes_vs_immobile=episode_outcomes_vs_immobile,
                    episode_outcomes_vs_neural=episode_outcomes_vs_neural,
                    rollout_components=rollout_components,
                    episode_comp_list=episode_comp_list,
                    episode_durations_s=episode_durations_s,
                    comp_step_stats=_comp_step_stats,
                    n_reward_comp_steps=len(_rcomp_raw),
                    episode_kick_counts=episode_kick_counts,
                    episode_armed_kick_counts=episode_armed_kick_counts,
                )

                if self.checkpoint_dir is not None:
                    self._save_checkpoint(self._total_steps)

                if self.rollout_eval_trials > 0:
                    self._eval_vs_rules(max_episode_s)
                self._maybe_run_neural_snapshot_eval(max_episode_s)

                def _close_main_pools() -> None:
                    nonlocal workers, eval_workers
                    close_workers(workers)
                    if eval_workers is not None:
                        from footballcoach.ai.eval.eval_worker import close_eval_workers
                        close_eval_workers(eval_workers)
                    eval_workers = None
                    self._persistent_eval_workers = None

                def _respawn_main_pools() -> None:
                    nonlocal workers, eval_workers
                    workers = spawn_workers(
                        phase_id, n_workers, base_seed, self.separate_value_net, self.worker_torch_threads,
                        progress_value=_progress_value,
                    )
                    if self._eval_n_parallel_workers > 1:
                        from footballcoach.ai.eval.eval_worker import spawn_eval_workers
                        eval_workers = spawn_eval_workers(
                            self._eval_n_parallel_workers, self.separate_value_net, self.worker_torch_threads,
                        )
                    self._persistent_eval_workers = eval_workers

                self._maybe_run_ppg_interleave(None, phase_id, _close_main_pools, _respawn_main_pools)
        finally:
            close_workers(workers)
            if eval_workers is not None:
                from footballcoach.ai.eval.eval_worker import close_eval_workers
                close_eval_workers(eval_workers)
            self._persistent_eval_workers = None

        if self.checkpoint_dir is not None:
            self._save_checkpoint(self._total_steps)
            log.info("Final checkpoint saved.")
        log.info(f"Training complete. Total steps: {self._total_steps:,}")

    def _update_episode_replay(
        self, all_episode_seed_reward_adv: list[tuple[int, float, float]],
    ) -> list[int]:
        """One rollout cycle's worth of episode-seed-replay bookkeeping --
        see ``_episode_replay_enabled``'s __init__ comment and
        ai/ppo/batched_rollout_worker.py's "Episode-seed replay" docstring
        section. Called once per cycle from ``_train_batched_parallel()``,
        right after this rollout's per-episode ``(seed, reward,
        mean_abs_advantage)`` triples are assembled.

        1. Before/after report: any seed in THIS rollout that was queued at
           the end of the PREVIOUS cycle (``self._pending_replay_before``)
           just gave us its "after" reward -- log the before-vs-after
           comparison, then clear it (one-shot: a seed not seen this
           rollout, because no env happened to reset while it was queued,
           is simply dropped, not carried forward another cycle).
        2. New selection: rank ALL of this rollout's episodes (fresh-drawn
           and any replays alike) by mean(|advantage|) descending, take the
           top ``episode_replay_top_fraction``, and stash them as the new
           "before" baseline for next cycle's report.

        Returns the list of seeds to request next cycle (empty if this
        rollout had no eligible episodes at all).
        """
        if self._pending_replay_before:
            _matched = [
                (seed, info["reward"], reward)
                for seed, reward, _adv in all_episode_seed_reward_adv
                if (info := self._pending_replay_before.get(seed)) is not None
            ]
            if _matched:
                _before_mean = float(np.mean([b for _s, b, _a in _matched]))
                _after_mean = float(np.mean([a for _s, _b, a in _matched]))
                log.info(
                    f"  [episode replay] matched={len(_matched)}/{len(self._pending_replay_before)}  "
                    f"before_mean_rew={_before_mean:.3f}  after_mean_rew={_after_mean:.3f}  "
                    f"delta={_after_mean - _before_mean:+.3f}"
                )
            else:
                log.info(
                    f"  [episode replay] 0/{len(self._pending_replay_before)} queued seeds were "
                    f"replayed this rollout (not enough episode resets to reach them)"
                )
            self._pending_replay_before = {}

        if not all_episode_seed_reward_adv:
            return []

        n_top = max(1, math.ceil(self._episode_replay_top_fraction * len(all_episode_seed_reward_adv)))
        top_k = sorted(all_episode_seed_reward_adv, key=lambda t: t[2], reverse=True)[:n_top]
        self._pending_replay_before = {seed: {"reward": rew, "abs_adv": adv} for seed, rew, adv in top_k}
        log.info(
            f"  [episode replay] queued {len(top_k)} seed(s) for next rollout "
            f"(top {self._episode_replay_top_fraction * 100:.1f}% of {len(all_episode_seed_reward_adv)} episodes by |adv|)"
        )
        return list(self._pending_replay_before.keys())


    def _train_batched_parallel(self, total_steps: int, phase_id: int, max_episode_s: float) -> None:
        """Batched multi-environment rollout collection path
        (``ppo.batched_rollout=True`` alongside ``ppo.n_processes > 1``).

        Opt-in alternative to ``_train_parallel()``: instead of
        ``n_processes`` worker PROCESSES each running ONE env with its
        own batch-of-1 network call, this runs ``n_processes`` worker
        processes, each internally running ``envs_per_process``
        environments (total envs = ``n_processes * envs_per_process``) and
        batching their trainee decisions into ONE network call per round --
        see
        ``ai/ppo/batched_rollout_worker.py``'s module docstring for the
        full design/measured-speedup rationale. Per-env GAE (never
        concatenate raw transitions across envs before bootstrap), the same
        discipline ``_train_parallel()`` uses per-worker.

        This method deliberately duplicates most of ``_train_parallel()``'s
        loop body rather than sharing it via a refactor, for now: this is a
        new, less-battle-tested collection path (opt-in, off by default,
        not yet validated by a real training run -- see the plan doc's
        verification section), and refactoring the PROVEN, currently-live
        ``_train_parallel()`` to share code with it would risk the working
        path for the sake of the new one. Worth unifying once this path is
        validated and its config knobs (``envs_per_process`` etc.) have
        settled -- flagged here rather than silently left as accepted debt.
        """
        import multiprocessing
        import multiprocessing.connection

        from footballcoach.ai.ppo.batched_rollout_worker import close_batched_workers, spawn_batched_workers

        envs_per_process = max(1, self.envs_per_process)
        n_processes = self.n_processes
        total_envs = n_processes * envs_per_process
        # PER-PROCESS budget (matches _train_parallel()'s steps_per_worker
        # convention) -- collect() aggregates across ALL envs_per_process
        # envs a process owns, so dividing by total_envs instead of
        # n_processes here previously made this envs_per_process times too
        # small, forcing envs_per_process-times more outer rounds (more
        # weight broadcasts + PPO updates per unit of data collected) than
        # intended -- confirmed via a real benchmark: 3 processes x 4 envs
        # needed 4 outer rounds instead of 1 to reach the same total_steps.
        steps_per_worker = max(1, self.rollout_steps // n_processes)
        base_seed = random.randint(0, 2**31 - 1)
        log.info(
            f"PPO batched-parallel training started: {n_processes} process(es) x "
            f"{envs_per_process} env(s) = {total_envs} total envs, "
            f"~{steps_per_worker} steps/worker/rollout, "
            f"steps_so_far={self._total_steps:,}  target={self._total_steps + total_steps:,}"
        )
        workers = spawn_batched_workers(
            phase_id, n_processes, envs_per_process, base_seed,
            self.separate_value_net, self.worker_torch_threads,
            batch_secondary_players=self.batch_secondary_players,
            chunk_steps=self.batched_rollout_chunk_steps or None,
        )
        # Persistent eval worker pool -- identical rationale/pattern to
        # _train_parallel()'s own (see that method's comment).
        eval_workers = None
        if self._eval_n_parallel_workers > 1:
            from footballcoach.ai.eval.eval_worker import spawn_eval_workers
            eval_workers = spawn_eval_workers(
                self._eval_n_parallel_workers, self.separate_value_net, self.worker_torch_threads,
            )
        self._persistent_eval_workers = eval_workers
        try:
            _steps_at_call_start = self._total_steps
            target_steps = _steps_at_call_start + total_steps
            # Episode-seed replay (see __init__'s _episode_replay_enabled
            # comment): seeds selected at the END of one cycle are consumed
            # at the START of the NEXT -- one-shot, not carried further if a
            # cycle doesn't happen to reset enough envs to use them all.
            replay_seeds_for_next_cycle: list[int] = []

            while self._total_steps < target_steps:
                progress = (self._total_steps - _steps_at_call_start) / total_steps
                rollout_start = time.perf_counter()

                # Broadcast current weights before collecting -- identical
                # rationale to _train_parallel() (standard PPO already
                # tolerates a slightly-stale behavior policy via the
                # importance ratio correction).
                dec_state = self.decision_net.state_dict()
                exec_state = self.execution_net.state_dict()
                val_state = self.value_net.state_dict() if self.value_net is not None else None
                for w in workers:
                    w.set_weights(dec_state, exec_state, val_state)

                if replay_seeds_for_next_cycle:
                    _replay_chunks = [
                        c for c in (replay_seeds_for_next_cycle[i::len(workers)] for i in range(len(workers))) if c
                    ]
                else:
                    _replay_chunks = []
                # Workers finalize their own results (GAE with THIS process's
                # gamma/lam -- passed explicitly so a worker's separately-read
                # ai_config.json can never drift from ours -- plus as_tensors)
                # and ship compact numpy arrays instead of per-row buffers:
                # see batched_rollout_worker.py's "Worker-side finalization".
                _returns_spec = {"mode": "gae", "gamma": self.gamma, "lam": self.lam}
                for _wi, w in enumerate(workers):
                    w.collect(
                        steps_per_worker,
                        replay_seeds=(_replay_chunks[_wi] if _wi < len(_replay_chunks) else None),
                        returns=_returns_spec,
                    )

                worker_batches = []
                episode_rewards: list[float] = []
                secondary_episode_rewards: list[float] = []
                episode_outcomes_vs_rules: list[str] = []
                episode_outcomes_vs_neural: list[str] = []
                episode_outcomes_vs_immobile: list[str] = []
                episode_comp_list: list[dict[str, float]] = []
                episode_durations_s: list[float] = []
                episode_kick_counts: list[int] = []
                episode_armed_kick_counts: list[int] = []
                rollout_components: dict[str, float] = {}
                # (seed, reward, mean_abs_advantage) for every completed
                # trainee episode this rollout with a recorded seed -- only
                # populated when episode_replay is enabled (see
                # _episode_abs_adv_means's docstring for why this is
                # computed per-r, using each buffer's OWN track_ids/dones,
                # rather than after _merge_worker_batches concatenates
                # everything together).
                all_episode_seed_reward_adv: list[tuple[int, float, float]] = []

                def _consume_result(r: dict) -> None:
                    _tensors, advantages, _r_track_ids, _r_dones = _decode_rollout_result(
                        r, self.gamma, self.lam, self._episode_replay_enabled,
                    )
                    worker_batches.append(_tensors)
                    stats = r["stats"]
                    episode_rewards.extend(stats["episode_rewards"])
                    secondary_episode_rewards.extend(stats["secondary_episode_rewards"])
                    episode_outcomes_vs_rules.extend(stats["episode_outcomes_vs_rules"])
                    episode_outcomes_vs_neural.extend(stats["episode_outcomes_vs_neural"])
                    episode_outcomes_vs_immobile.extend(stats["episode_outcomes_vs_immobile"])
                    episode_comp_list.extend(stats["episode_comp_list"])
                    episode_durations_s.extend(stats["episode_durations_s"])
                    episode_kick_counts.extend(stats.get("episode_kick_counts", []))
                    episode_armed_kick_counts.extend(stats.get("episode_armed_kick_counts", []))
                    for ep in stats["episode_comp_list"]:
                        for _k, _v in ep.items():
                            rollout_components[_k] = rollout_components.get(_k, 0.0) + _v
                    if self._episode_replay_enabled:
                        _abs_adv_means = _episode_abs_adv_means(
                            _r_track_ids, _r_dones, advantages,
                        )
                        for _seed, _rew, _adv in zip(
                            stats.get("episode_seeds", []), stats["episode_rewards"], _abs_adv_means,
                        ):
                            if _seed is not None:
                                all_episode_seed_reward_adv.append((_seed, _rew, _adv))

                # Interleaved receive: each worker sends zero-or-more
                # {"chunk": [...]} messages (see ai/ppo/batched_rollout_worker.py's
                # "Chunked streaming" docstring -- this is what bounds each
                # worker's peak memory instead of one giant end-of-rollout
                # pickle) followed by exactly one {"done": True}. A worker
                # still counts as "done" (for progress reporting) exactly
                # once, when its sentinel arrives, regardless of how many
                # chunks preceded it -- unchanged semantics from before
                # chunking existed. wait() reports a connection ready again
                # immediately if it still has buffered messages, so a
                # connection with several queued messages just costs a few
                # extra (cheap) polling iterations, never a lost message.
                _pending = {w.conn for w in workers}
                _agg_progress = ProgressReporter(
                    n_processes, prefix=f"  [batched rollout] ({total_envs} envs, {n_processes} proc): ", live=True,
                )
                _n_done = 0
                while _pending:
                    ready = multiprocessing.connection.wait(list(_pending), timeout=0.2)
                    for conn in ready:
                        msg = conn.recv()
                        for r in msg.get("chunk", []):
                            _consume_result(r)
                        if msg.get("done"):
                            _pending.discard(conn)
                            _n_done += 1
                            _agg_progress.update(_n_done)

                # release_inputs: drop each key from the per-chunk dicts as it's
                # concatenated (peak ~1x the rollout instead of ~2x -- the
                # rollout-end RAM spike). worker_batches is unused afterwards.
                batch = _merge_worker_batches(worker_batches, release_inputs=True)
                worker_batches.clear()
                n_collected = int(batch["rewards"].shape[0])

                if self._episode_replay_enabled:
                    replay_seeds_for_next_cycle = self._update_episode_replay(all_episode_seed_reward_adv)
                self._total_steps += n_collected
                rollout_time = time.perf_counter() - rollout_start
                steps_per_sec = n_collected / max(rollout_time, 1e-6)

                _comp_step_stats: dict[str, dict] = {}
                _rcomp_raw = batch.get("reward_comps_raw", [])
                if _rcomp_raw:
                    _rets_np = batch["returns"].numpy()
                    _advs_np = batch["advantages"].numpy()
                    _vals_np = batch["values"].numpy()
                    _td_np = _rets_np - _vals_np
                    for _ck, _clabel in REWARD_COMP_LABELS:
                        _cvals = np.array([float(_d.get(_ck, 0.0)) for _d in _rcomp_raw])
                        _mask = _cvals != 0.0
                        _td_m = _td_np[_mask] if _mask.any() else np.array([])
                        _comp_step_stats[_ck] = {
                            "mean": float(_cvals.mean()),
                            "std": float(_cvals.std()),
                            "count": int(_mask.sum()),
                            "mean_ret": float(_rets_np[_mask].mean()) if _mask.any() else float("nan"),
                            "std_ret": float(_rets_np[_mask].std()) if _mask.any() else float("nan"),
                            "mean_gae": float(_advs_np[_mask].mean()) if _mask.any() else float("nan"),
                            "mean_sq_td": float((_td_m ** 2).mean()) if _mask.any() else float("nan"),
                            "mean_abs_td": float(np.abs(_td_m).mean()) if _mask.any() else float("nan"),
                            "p95_td": float(np.percentile(np.abs(_td_m), 95)) if _mask.any() else float("nan"),
                            "label": _clabel,
                        }

                metrics = self._ppo_update(batch, progress)

                self._log_rollout_summary(
                    metrics=metrics,
                    steps_per_sec=steps_per_sec,
                    episode_rewards=episode_rewards,
                    secondary_episode_rewards=secondary_episode_rewards,
                    episode_outcomes_vs_rules=episode_outcomes_vs_rules,
                    episode_outcomes_vs_immobile=episode_outcomes_vs_immobile,
                    episode_outcomes_vs_neural=episode_outcomes_vs_neural,
                    rollout_components=rollout_components,
                    episode_comp_list=episode_comp_list,
                    episode_durations_s=episode_durations_s,
                    comp_step_stats=_comp_step_stats,
                    n_reward_comp_steps=len(_rcomp_raw),
                    episode_kick_counts=episode_kick_counts,
                    episode_armed_kick_counts=episode_armed_kick_counts,
                )

                if self.checkpoint_dir is not None:
                    self._save_checkpoint(self._total_steps)

                if self.rollout_eval_trials > 0:
                    self._eval_vs_rules(max_episode_s)
                self._maybe_run_neural_snapshot_eval(max_episode_s)

                def _close_main_pools() -> None:
                    nonlocal workers, eval_workers
                    close_batched_workers(workers)
                    if eval_workers is not None:
                        from footballcoach.ai.eval.eval_worker import close_eval_workers
                        close_eval_workers(eval_workers)
                    eval_workers = None
                    self._persistent_eval_workers = None

                def _respawn_main_pools() -> None:
                    nonlocal workers, eval_workers
                    workers = spawn_batched_workers(
                        phase_id, n_processes, envs_per_process, base_seed,
                        self.separate_value_net, self.worker_torch_threads,
                        batch_secondary_players=self.batch_secondary_players,
                        chunk_steps=self.batched_rollout_chunk_steps or None,
                    )
                    if self._eval_n_parallel_workers > 1:
                        from footballcoach.ai.eval.eval_worker import spawn_eval_workers
                        eval_workers = spawn_eval_workers(
                            self._eval_n_parallel_workers, self.separate_value_net, self.worker_torch_threads,
                        )
                    self._persistent_eval_workers = eval_workers

                self._maybe_run_ppg_interleave(
                    None, phase_id, _close_main_pools, _respawn_main_pools,
                    replay_seeds=replay_seeds_for_next_cycle,
                )
        finally:
            close_batched_workers(workers)
            if eval_workers is not None:
                from footballcoach.ai.eval.eval_worker import close_eval_workers
                close_eval_workers(eval_workers)
            self._persistent_eval_workers = None

        if self.checkpoint_dir is not None:
            self._save_checkpoint(self._total_steps)
            log.info("Final checkpoint saved.")
        log.info(f"Training complete. Total steps: {self._total_steps:,}")




    def _cpu_state_dicts(self) -> tuple[dict, dict, Optional[dict]]:
        """Snapshot (decision_state, execution_state, value_state) as plain
        CPU tensors, ready to hand to a spawn-pool worker subprocess (see
        rebuild_inference_trainer()) -- the single source of truth for this
        snapshot, used by every parallel-worker call site in this file
        (_eval_vs_opponent_type below, ai/ppo/dagger.py's parallel rollout
        collection).

        .cpu() each tensor before handing the state dict to spawn-pool
        workers -- those workers always rebuild a CPU-only trainer (see
        rebuild_inference_trainer()), but pickling CUDA tensors as-is makes
        torch try to CUDA-IPC-share them into the subprocess instead of
        copying, which is unreliable on Windows and surfaces as spurious
        "out of memory" / "device busy" errors. value_state is None when
        this trainer has no separate value_net (single-critic mode).
        """
        decision_state = {k: v.detach().cpu() for k, v in self.decision_net.state_dict().items()}
        execution_state = {k: v.detach().cpu() for k, v in self.execution_net.state_dict().items()}
        value_state = (
            {k: v.detach().cpu() for k, v in self.value_net.state_dict().items()}
            if self.value_net is not None else None
        )
        return decision_state, execution_state, value_state

    def _run_persistent_eval(
        self, workers, decision_state: dict, execution_state: dict, value_state: Optional[dict],
        seeds: list[int], repeats_per_seed: int, use_rules_ai: bool, max_episode_s: float,
        win_outcome: str = "box_possession", swap_sides: bool = False,
        batched: bool = False, envs_per_process: int = 8,
    ):
        """Dispatch one seeded eval across an already-running persistent
        eval worker pool (see ai/eval/eval_worker.py), mirroring
        run_seeded_evaluation_parallel()'s seed-chunking/merge but without
        spawning a fresh Pool -- the workers were spawned once by
        _train_parallel() and are reused every call. ``batched``/
        ``envs_per_process``: see eval.batched_eval -- threaded straight
        into each worker's "eval" IPC message (ai/eval/eval_worker.py's
        _eval_worker_main picks the batched vs unbatched path there)."""
        from footballcoach.ai.eval.seeded_eval import merge_eval_results

        for w in workers:
            w.set_weights(decision_state, execution_state, value_state)
        chunks = [c for c in (seeds[i::len(workers)] for i in range(len(workers))) if c]
        active = workers[:len(chunks)]
        for w, chunk in zip(active, chunks):
            w.eval(chunk, repeats_per_seed, use_rules_ai, max_episode_s, win_outcome, swap_sides,
                   batched=batched, envs_per_process=envs_per_process)
        results = [w.recv_result() for w in active]
        return merge_eval_results(results, repeats_per_seed)



















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












    # -----------------------------------------------------------------------
    # PERMANENT: NaN/Inf diagnostic (added 2026-09-09, investigating a real
    # self-play training crash -- root cause still open, see
    # ai_trainer_knowledge.md/training notes). Deliberately kept in
    # (not temporary instrumentation to strip out later): the user's explicit
    # preference is a LOUD, immediate, richly-diagnosed halt the instant this
    # fires again, over any kind of silent skip-and-continue recovery -- see
    # the deliberately-declined "skip a non-finite optimizer step" guard
    # discussed alongside this. Distinguishes forward-pass corruption (a
    # value already NaN/Inf BEFORE backward(), e.g. bad rollout data) from
    # backward-pass numerical instability (a perfectly finite forward value
    # whose GRADIENT is singular/NaN -- classic examples: atan2(0, 0), a
    # Bessel-function ratio i1e/i0e at kappa=0, sqrt(0)'s infinite
    # derivative -- all real possibilities given the VonMises/atan2-heavy
    # direction heads this codebase uses; confirmed for real at kappa->0,
    # see VonMisesDirectionHead -- not yet fixed, a known-open landmine).
    # Uses batch["track_ids"] (now threaded through
    # RolloutBuffer.as_tensors(), see its own comment) to identify whether
    # the offending minibatch's rows skew toward the secondary (self-play
    # opponent) track -- a real per-row identity, not the fragile
    # sample_weights != 1.0 proxy this used before track_ids was wired
    # through.
    # -----------------------------------------------------------------------
    class _NaNReproFound(Exception):
        """Raised by _dump_nan_diagnostic to halt training the instant a
        non-finite loss/gradient is caught -- deliberately fatal, not
        recoverable: the run stops immediately (before the epoch loop can
        grind through the remaining minibatches on garbage weights, and
        before a corrupted checkpoint can get saved), with the full
        diagnostic dump already written to disk by the time this propagates
        up and kills the process."""

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


    # HEAD_LP_KEYS column -> opportunity_masks() key for the heads that only carry policy-gradient signal where the
    # action could take effect (agent_plans/masked_action_training_plan.md). kick_spin is permanently frozen
    # (log-prob column always 0) but is listed with its parent params for completeness.
    _OPP_HEAD_MASK_KEY = {
        "kick": "kick", "tackle_attempt": "tackle_attempt",
        "kick_dir": "kick_dir", "kick_power": "kick_power", "kick_spin": "kick_dir",
    }

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
        """
        from footballcoach.ai.ppo.bc import bc_loss_from_tensor, direction_magnitude_reg

        # Hold out a random subset of complete episodes from training (see
        # __init__'s val_episode_fraction comment) BEFORE augmentation, so
        # held-out rows are never expanded/flipped and never enter any
        # minibatch below -- `batch` is reassigned to the train-only portion
        # for the rest of this function; `val_batch` (None when disabled or
        # too few episodes) is only touched by the per-epoch eval below.
        val_batch: Optional[dict] = None
        if self.val_episode_fraction > 0.0:
            batch, val_batch = self._split_train_val_episodes(batch)

        # Augment batch with geometric flips + slot permutations before any
        # gradient computation.  This expands the batch by 2 × n_slot_shuffles
        # (N_FLIP_VARIANTS=2: identity, flip_y -- see augment.py's module
        # docstring; flip_x is a fixed canonical-frame transform, not a
        # random augmentation choice here).
        if self.augment_n_slot_shuffles > 0:
            batch = augment_batch(batch, self.augment_n_slot_shuffles, self._aug_rng)

        n = len(batch["log_probs"])

        if self.log_action_opportunity_stats:
            self._log_action_opportunity(batch)

        # --- Precompute frozen physics-encoder features ONCE for this
        # rollout's batch (fixed by this point -- augmentation above, if
        # any, has already run and will not run again for this batch),
        # instead of letting every one of the ppo.n_epochs passes below
        # (plus the value-only/bc-only continuation passes and diagnostics
        # further down, all of which replay this exact same batch) re-run
        # the frozen ball_physics_encoder/player_physics_encoder from
        # scratch on the exact same rows. Their output depends only on
        # these (now-fixed) input features, not on the policy's own
        # changing weights, so it is provably identical across every pass
        # over this batch -- unlike the rest of decision_net's forward
        # pass, which genuinely must be recomputed every epoch since its
        # trainable weights change. Computed in CANONICAL frame via the
        # same canonicalize_obs() helper CanonicalNetworkWrapper itself
        # uses internally (not a re-derivation of that logic), chunked by
        # minibatch_size like the pre-update value-loss pass below to
        # bound peak memory. Cached as ordinary "obs/*" batch entries
        # (moved back to CPU, matching every other batch["obs/..."]
        # tensor's storage convention -- see the .to(self.device) calls at
        # every mb_obs/_pre_obs/diag_obs construction site below) so the
        # existing generic per-minibatch slicing loops pick them up for
        # free; only the self.decision_net(...) call sites themselves need
        # to pass them through via the ball_physics_full/self_physics_full/
        # other_physics_full kwargs (see DecisionNetwork.forward()). ---
        batch.update(self._precompute_physics_full(batch, self.minibatch_size))

        # Replace augment_batch()'s cheap tiled old_log_prob (correct only
        # for the identity-flip copies) with the TRUE log pi_old(action|obs)
        # for every row, flip_y-augmented copies included -- see
        # _recompute_old_log_probs_for_augmented_batch's own docstring for
        # the full rationale. Must run AFTER the physics-full precompute
        # just above (consumes those obs/*_physics_full fields) and is only
        # meaningful when augmentation actually ran -- an unaugmented batch's
        # log_probs are already exact.
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
            _flip_pair = None

        clip = self.schedules.clip(progress)
        lr = self.schedules.lr(progress)
        value_lr = self.schedules.value_lr(progress)
        bc_coeff = self.schedules.bc(progress)
        bc_kick_coeff = self.schedules.bc_kick(progress)
        bc_tackle_coeff = self.schedules.bc_tackle(progress)
        ent_coef = self.schedules.ent(progress)
        for pg in self.optimizer.param_groups:
            pg["lr"] = value_lr if pg.get("name") == "value" else lr
        # Per-update LR warmup: remember this update's scheduled lrs and
        # scale them for the first lr_warmup_steps optimizer steps below.
        _warm_base_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        _warm_steps_done = 0

        has_bc ="bc_labels" in batch and (bc_coeff > 0.0 or bc_kick_coeff > 0.0 or bc_tackle_coeff > 0.0)

        # Normalize advantages
        adv = batch["advantages"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        old_log_probs = batch["log_probs"]
        returns = batch["returns"]
        # ppo.mask_untriggered_actions: heads that could not take effect on a row carry no policy-gradient signal
        # there -- the ratio is built from log-probs with those heads' terms removed on BOTH sides. The unmasked
        # old_log_probs stays for the whole-policy diagnostics further down.
        _opp_mask = self._opp_head_mask_matrix(batch)
        _opp_grad_scale = self._opp_head_grad_scale(_opp_mask)
        old_log_probs_ratio = (
            old_log_probs if _opp_mask is None
            else self._mask_total_log_probs(old_log_probs, batch["head_log_probs"], _opp_mask)
        )

        all_policy_loss = []
        all_value_loss = []
        # Per-SAMPLE (not per-minibatch-mean) policy_loss, reset and
        # reported every epoch -- lets us look at the actual distribution's
        # shape (skew, outliers) rather than just pol_mean, which can hide
        # e.g. a small number of huge-magnitude samples dragging the mean
        # away from 0. Like pol_mean itself, this is a PROGRESSIVE pool
        # across that epoch's minibatches -- rows collected early in the
        # epoch reflect staler, less-trained weights than rows collected
        # near its end (the network is being updated minibatch-by-minibatch
        # throughout), so treat each epoch's percentiles as an average-ish
        # summary over that epoch's trajectory, not a single clean
        # snapshot -- unlike the val-side percentiles (_eval_val_episode_losses),
        # which really are one clean snapshot since val rows never train.
        _epoch_sample_policy_loss: list[torch.Tensor] = []
        # Same progressive-pool caveat as _epoch_sample_policy_loss above,
        # but for the value HEAD's per-sample squared-error terms -- reports
        # as val_mean_p10_p90 (a trimmed companion to val_mean, this
        # function's TRAIN-side value-head loss; not to be confused with
        # val_episode_value_loss_p10_p90, the HELD-OUT set's own trimmed
        # value loss -- "val" is unfortunately overloaded in this file
        # between "value head" and "held-out validation set").
        _epoch_sample_value_loss: list[torch.Tensor] = []
        # Unclipped, no-dual-clip linearised surrogate (r-1)*A per row -- see
        # "[policy surrogate]"'s own log-line comment below for the
        # rationale (a clip-invariant companion to pol_mean/policy_loss).
        _epoch_sample_surrogate: list[torch.Tensor] = []
        all_entropy = []
        all_entropy_breakdown: dict[str, list[float]] = {}
        all_kl = []
        # Per-outcome value-loss accumulator (see value_mse_by_outcome()) --
        # squared-error sum + row count per outcome string, across every
        # minibatch of this update (main loop + value-only continuation
        # below), for a single end-of-update breakdown log line.
        _outc_sq_err_sum: dict[str, float] = {}
        _outc_n_sum: dict[str, int] = {}
        _outc_gt_sum: dict[str, float] = {}
        _outc_gt_sqsum: dict[str, float] = {}
        _step_outcomes = batch.get("step_outcomes", [])

        def _accum_value_by_outcome(pred: torch.Tensor, target: torch.Tensor, idx: torch.Tensor) -> None:
            if not _step_outcomes:
                return
            outcomes_mb = [_step_outcomes[i] for i in idx.tolist()]
            for _name, (_mse, _n, _gt_mean, _gt_mean_sq) in value_mse_by_outcome(pred, target, outcomes_mb).items():
                _outc_sq_err_sum[_name] = _outc_sq_err_sum.get(_name, 0.0) + _mse * _n
                _outc_n_sum[_name] = _outc_n_sum.get(_name, 0) + _n
                _outc_gt_sum[_name] = _outc_gt_sum.get(_name, 0.0) + _gt_mean * _n
                _outc_gt_sqsum[_name] = _outc_gt_sqsum.get(_name, 0.0) + _gt_mean_sq * _n
        all_bc_loss = []
        all_bc_tackle_loss: list[float] = []   # BCE on tackle_attempt head only
        all_tackle_prob: list[float] = []       # mean sigmoid(tackle_attempt_logit) per mb
        all_kick_prob: list[float] = []         # mean sigmoid(kick_logit) per mb
        all_ratios: list[torch.Tensor] = []
        # Tracks the single highest-ratio sample seen across the WHOLE update
        # (every epoch/minibatch), captured at the exact (row, weight
        # snapshot) that produced it -- unlike the [worst sample]/[top-2
        # highest-ratio] diagnostic below, which only re-examines the first
        # 256 buffer rows under the FINAL post-update weights and is not
        # necessarily the same sample (or even the same epoch) that produced
        # the true max reported in the [ratio] percentile line.
        _ratio_spike: dict = {"ratio": float("-inf")}
        all_mv_log_std_grad: list[float] = []  # grad on move_dir_log_kappa after each backward
        # --- Extra diagnostics: advantage stats, ratio stats, per-execution-head
        # grad norm, continuous/discrete head drift, per-head KL. All aggregated
        # over the rollout and printed once at the end (see below, near the
        # existing [grad clip] log line). ---
        all_adv_mean: list[float] = []
        all_adv_std: list[float] = []
        all_adv_min: list[float] = []
        all_adv_max: list[float] = []
        all_ratio_clipped_frac: list[float] = []
        all_head_grad_norm: dict[str, list[float]] = {name: [] for name, _ in EXEC_HEAD_MODULES}
        all_head_kl: list[torch.Tensor] = []  # per-mb (13,) per-head KL, after step
        # Per-head COUNTERFACTUAL policy_loss: "if only this ONE head's
        # probability had shifted from the rollout-collection policy (every
        # other head held at ratio=1), what would the clipped surrogate
        # loss be" -- computed pre-step (matching the real, reported
        # `policy_loss` scalar, unlike all_head_kl above which is measured
        # AFTER the optimizer step). Since the real per-sample ratio is the
        # PRODUCT of every head's own ratio (log-probs sum -> ratio =
        # exp(sum) = product of exp(per-head delta)), this is NOT a strict
        # decomposition of the scalar policy_loss (sum of these 15 numbers
        # != actual policy_loss) -- it's a per-head counterfactual, useful
        # for comparing which heads are driving the objective in which
        # direction, not for reconstructing the total.
        all_head_policy_loss: list[torch.Tensor] = []
        all_continuous_mean_shift: dict[str, list[float]] = {"move_direction": [], "kick_direction": []}
        all_continuous_log_std_shift: dict[str, list[float]] = {"move_direction": [], "kick_direction": []}
        # kick_dir_z_log_std (elevation, plain Gaussian) has no per-head KL/
        # entropy slot of its own -- KickDirectionHead sums it into the same
        # combined "kick_dir" number as the azimuthal von Mises component
        # everywhere that's already a single scalar (entropy, KL, log_prob).
        # Its own VALUE/drift has no such existing home, so it gets a
        # dedicated tracker here instead of being silently invisible.
        all_kickz_log_std_shift: list[float] = []
        all_discrete_logit_shift: dict[str, list[float]] = {
            name: [] for name in ("exec_move", "sprint", "kick", "tackle_attempt")
        }
        # Grad-norm clipping diagnostics: pre-clip norm for each of the two isolated
        # groups (non-direction "main", direction) per minibatch, plus how many
        # minibatches actually exceeded their respective limit (i.e. were clipped).
        # Lets us see how often/hard each group is being clipped, rather than just
        # the old single combined-group raw_grad_norm debug value.
        all_grad_norm_main: list[float] = []
        all_grad_norm_dir: list[float] = []
        clip_triggered_main = 0
        clip_triggered_dir = 0
        # Actual applied parameter-delta norm for the direction group -- i.e.
        # ||params_after - params_before|| across move_direction/kick_direction/
        # move_dir_log_kappa/kick_dir_log_kappa/kick_dir_z_log_std, measured directly around
        # optimizer.step() (see below). Separate from all_grad_norm_dir (the
        # pre-clip GRADIENT norm): Adam's real step size is m_hat/sqrt(v_hat),
        # a RATIO, not a direct function of the clipped gradient -- especially
        # right after --reset-optimizer, when v is still unreliable (Adam's
        # ~1000-step characteristic warm-up at the default beta2=0.999), the
        # actual move can be much larger than the clip limit alone suggests.
        # This makes that gap directly visible instead of inferred.
        all_param_delta_dir: list[float] = []
        epoch_times = []
        KL_DIAG_THRESHOLD = 0.05  # ~5× target_kl; log detailed diagnostics above this

        # Debug-level rollout stats (hidden by default)
        if log.isEnabledFor(logging.DEBUG):
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

        _diag_done = False  # print per-head breakdown only once
        _early_stopped = False
        # Snapshot the continuous-head log_kappa at the very start of this update
        # ("before the optimiser step", rollout-wide) so the final summary can
        # report start -> end drift across the whole rollout, not just per-mb.
        _move_ls_start = float(self.execution_net.move_dir_log_kappa.mean().item())
        _kick_ls_start = float(self.execution_net.kick_dir_log_kappa.mean().item())
        _kickz_ls_start = float(self.execution_net.kick_dir_z_log_std.mean().item())

        # --- Pre-update value loss: a genuinely held-out-ish generalisation
        # diagnostic, computed on this rollout's FULL batch with the current
        # weights BEFORE any minibatch training this call touches them (this
        # rollout's transitions were collected under the policy/value from
        # the END of the previous update, so the critic has never been
        # trained on this exact data). Uses the same MSE/Var(returns)
        # formula as the in-sample `value_loss`/`val=` computed below (and
        # as pretrain_value()'s val loss), so it's directly comparable to
        # both -- unlike `all_value_loss` below, which averages losses
        # computed progressively DURING training on this same batch (and,
        # after early-stop, during the value-only continuation's extra
        # epochs on the same data), so it structurally reads better than a
        # true generalisation number regardless of any real overfitting on
        # top. See ai_trainer_knowledge.md section 8's "val=" / "val_pre="
        # note for the full rationale.
        _ret_var_full = returns.var().clamp(min=1.0)
        with torch.no_grad():
            _pre_sq_err_sum = 0.0
            _pre_n = 0
            for _start in range(0, n, self.minibatch_size):
                _idx = torch.arange(_start, min(_start + self.minibatch_size, n))
                _pre_obs = {k.replace("obs/", ""): batch[k][_idx].to(self.device)
                            for k in batch if k.startswith("obs/")}
                _psf, _pof, _pem = _pre_obs["self_feat"], _pre_obs["other_feat"], _pre_obs["exists_mask"]
                _pbf, _pgf = _pre_obs["ball_feat"], _pre_obs["global_feat"]
                _psat, _poat = _ai_types(_pre_obs)
                _pd_heads = self.decision_net(
                    _psf, _pof, _pem, _pbf, _pgf, _psat, _poat,
                    ball_physics_full=_pre_obs.get("ball_physics_full"),
                    self_physics_full=_pre_obs.get("self_physics_full"),
                    other_physics_full=_pre_obs.get("other_physics_full"),
                )
                if self.separate_value_net:
                    _pd_heads_v = _detach_decision_heads(_pd_heads)
                    _pvalue = self.value_net(_psf, _pof, _pem, _pbf, _pgf, _pd_heads_v, _psat, _poat, value_only=True)
                else:
                    _pvalue = self.execution_net(_psf, _pof, _pem, _pbf, _pgf, _pd_heads, _psat, _poat, value_only=True)
                _pred = _pvalue.squeeze(-1)
                _pret = returns[_idx].to(self.device)
                _pre_sq_err_sum += F.mse_loss(_pred, _pret, reduction="sum").item()
                _pre_n += len(_idx)
            pre_update_value_loss = (_pre_sq_err_sum / max(_pre_n, 1)) / float(_ret_var_full)

        if self.log_epoch_zero_baseline:
            # Reuses _eval_val_episode_losses() for BOTH populations (it
            # takes an arbitrary batch dict, not hardcoded to "the held-out
            # set") -- gives byte-identical formulas to the real per-epoch
            # lines below with no duplicated math, at the cost of the same
            # physics-encoder gap that function already has for the REAL
            # held-out lines (no ball_physics_full/self_physics_full/
            # other_physics_full passthrough) -- an existing limitation,
            # not a new one introduced here; irrelevant when physics
            # encoders are disabled. old_log_prob==new_log_prob EXACTLY for
            # every row here (epoch 1's first gradient step hasn't happened
            # yet), so policy_loss/KL/surrogate are trivially ~0 by
            # construction -- expected, not a bug -- while value/entropy-
            # adjacent percentiles/per-head numbers are real and useful as
            # a before/after-epoch-1 comparison point.
            def _log_baseline_block(tag: str, n_rows_label: str, eval_result: tuple) -> None:
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
                    _pcts = [0, 1, 10, 50, 90, 99, 100]
                    _pct_str = "  ".join(
                        f"p{p}={float(_b_per_sample_pol.quantile(p / 100.0)):.4f}" for p in _pcts
                    )
                    log.info(f"  [{tag}policy percentiles, n={len(_b_per_sample_pol):,}]  {_pct_str}")
                if _b_head_pol is not None:
                    _inactive_heads = self._inactive_head_lp_keys()
                    _head_str = "  ".join(
                        f"{k}={v:+.4f}" for k, v in zip(HEAD_LP_KEYS, _b_head_pol.tolist())
                        if k not in _inactive_heads
                    )
                    log.info(f"  [{tag}policy heads] {_head_str}")
                log.info(
                    f"  [{tag}value] mean={_b_val:.4f}(x{self.vf_coef})={self.vf_coef * _b_val:.4f}  "
                    f"p10_p90={_b_val_p90:.4f}(x{self.vf_coef})={self.vf_coef * _b_val_p90:.4f}  "
                    f"p25_p75={_b_val_p75:.4f}(x{self.vf_coef})={self.vf_coef * _b_val_p75:.4f}"
                )

            log.info(f"  == epoch 0 (pre-training baseline, ratio=1 exactly) " + "=" * 32)
            _log_baseline_block("", f"n={n} rows, pre-training", self._eval_val_episode_losses(batch, clip))
            if val_batch is not None:
                _log_baseline_block(
                    "held-out ", f"n={len(val_batch['log_probs'])} held-out steps, pre-training",
                    self._eval_val_episode_losses(val_batch, clip),
                )
            log.info("  [grad clip] N/A -- no gradient step taken yet")

        for epoch_i in range(self.n_epochs):
            epoch_start = time.perf_counter()
            _epoch_slice_start = len(all_policy_loss)
            _epoch_sample_policy_loss.clear()
            _epoch_sample_value_loss.clear()
            _epoch_sample_surrogate.clear()
            indices = torch.randperm(n)
            for start in range(0, n, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]
                if len(mb_idx) == 0:
                    continue

                mb_obs = {k.replace("obs/", ""): batch[k][mb_idx].to(self.device)
                          for k in batch if k.startswith("obs/")}
                mb_adv = adv[mb_idx].to(self.device)
                # Advantage stats, BEFORE the optimiser step (this minibatch's slice).
                # Single 4-element sync instead of 4 separate .item() calls (each forces one).
                _adv_stats = torch.stack([mb_adv.mean(), mb_adv.std(), mb_adv.min(), mb_adv.max()]).tolist()
                all_adv_mean.append(_adv_stats[0])
                all_adv_std.append(_adv_stats[1])
                all_adv_min.append(_adv_stats[2])
                all_adv_max.append(_adv_stats[3])
                mb_ret = returns[mb_idx].to(self.device)
                mb_old_lp = old_log_probs_ratio[mb_idx].to(self.device)
                # Normalised per-sample weights: sum to minibatch size so loss scale is stable
                mb_w_raw = batch["sample_weights"][mb_idx].to(self.device)
                mb_w = mb_w_raw * (len(mb_w_raw) / mb_w_raw.sum().clamp(min=1e-8))

                # Recompute log_probs and values with current policy
                sf = mb_obs["self_feat"]
                of = mb_obs["other_feat"]
                em = mb_obs["exists_mask"]
                bf = mb_obs["ball_feat"]
                gf = mb_obs["global_feat"]
                sat, oat = _ai_types(mb_obs)

                d_heads = self.decision_net(
                    sf, of, em, bf, gf, sat, oat,
                    ball_physics_full=mb_obs.get("ball_physics_full"),
                    self_physics_full=mb_obs.get("self_physics_full"),
                    other_physics_full=mb_obs.get("other_physics_full"),
                )
                e_heads = self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat)

                # Snapshot BEFORE the optimiser step, for the drift diagnostics
                # printed once at the end of _ppo_update (see EXEC_HEAD_MODULES).
                _log_std_move_before = self.execution_net.move_dir_log_kappa.detach().clone()
                _log_std_kick_before = self.execution_net.kick_dir_log_kappa.detach().clone()
                _log_std_kickz_before = self.execution_net.kick_dir_z_log_std.detach().clone()
                _exec_move_logit_before = e_heads.exec_move_logit.detach().clone()
                _sprint_logit_before = e_heads.sprint_logit.detach().clone()
                _kick_logit_before = e_heads.kick_logit.detach().clone()
                _tackle_attempt_logit_before = e_heads.tackle_attempt_logit.detach().clone()

                # Value estimate — single value head (execution_net only), OR the
                # dedicated self.value_net when separate_value_net is enabled. In
                # the latter case d_heads is normally detached before feeding
                # value_net so its (separately-optimised) value loss never
                # contributes gradient into decision_net -- the whole point of
                # separate_value_net is a critic trunk fully independent of the
                # (BC-primed) policy trunk. share_value_grad_with_decision opts
                # back into a controlled version of that leak (decision_net only,
                # never execution_net's trunk -- value_net has zero weight
                # sharing with execution_net either way), scaled by
                # decision_value_coef instead of full-strength -- see that flag's
                # __init__ comment and _scale_decision_heads_grad's docstring.
                if self.separate_value_net:
                    if self.share_value_grad_with_decision:
                        d_heads_for_value = _scale_decision_heads_grad(d_heads, self.decision_value_coef)
                    else:
                        d_heads_for_value = _detach_decision_heads(d_heads)
                    _value = self.value_net(sf, of, em, bf, gf, d_heads_for_value, sat, oat, value_only=True)
                    new_values = _value.squeeze(-1)
                else:
                    new_values = e_heads.value.squeeze(-1)

                # New log_probs (sample stored actions from batch)
                mb_actions = {k.replace("action/", ""): batch[k][mb_idx].to(self.device)
                              for k in batch if k.startswith("action/")}
                new_log_probs = self._recompute_log_prob(d_heads, e_heads, mb_actions, em)
                _mb_opp_mask = None if _opp_mask is None else _opp_mask[mb_idx].to(self.device)
                if _mb_opp_mask is not None:
                    new_log_probs = self._mask_total_log_probs(
                        new_log_probs, self._per_head_new_log_probs(d_heads, e_heads, mb_actions, em),
                        _mb_opp_mask, _opp_grad_scale,
                    )

                # Per-head log_prob breakdown (debug only, first mb of first epoch)
                if not _diag_done and log.isEnabledFor(logging.DEBUG):
                    _diag_done = True
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
                else:
                    _diag_done = True  # skip computation when not in DEBUG

                # PPO clipped objective (weighted by per-sample importance weights)
                ratio = torch.exp(new_log_probs - mb_old_lp)
                all_ratios.append(ratio.detach().cpu())
                with torch.no_grad():
                    _mb_max_ratio_t, _mb_max_local = ratio.max(dim=0)
                    _mb_max_ratio_f = float(_mb_max_ratio_t)
                    if _mb_max_ratio_f > _ratio_spike["ratio"]:
                        _row = int(_mb_max_local)
                        # Same (d_heads, e_heads) snapshot that produced this
                        # minibatch's ratio -- pre-step, matching new_log_probs.
                        _per_head_row = self._per_head_new_log_probs(
                            d_heads, e_heads, mb_actions, em
                        )[_row].detach().cpu()
                        _ratio_spike = {
                            "ratio": _mb_max_ratio_f,
                            "epoch": epoch_i,
                            "mb": start // self.minibatch_size,
                            "row": int(mb_idx[_row]),
                            "old_lp": float(mb_old_lp[_row]),
                            "new_lp": float(new_log_probs[_row]),
                            "adv": float(mb_adv[_row]),
                            "per_head_new": _per_head_row,
                        }
                # Kept as a tensor (not .item()'d) until the batched sync
                # below, alongside tackle_prob/kick_prob/bc_loss_val -- same
                # value, one CPU<->GPU round trip instead of four.
                ratio_clipped_frac_t = ((ratio < 1.0 - clip) | (ratio > 1.0 + clip)).float().mean()
                # The diagnostics above keep the raw ratio (inf stays visible);
                # the loss uses the capped one -- see _MAX_LOG_RATIO.
                loss_ratio = self._loss_ratio(new_log_probs - mb_old_lp)
                surr1 = loss_ratio * mb_adv
                surr2 = torch.clamp(loss_ratio, 1.0 - clip, 1.0 + clip) * mb_adv
                _min_surr = self._apply_dual_clip(torch.min(surr1, surr2), mb_adv)
                _per_sample_policy_loss = -(_min_surr * mb_w)
                policy_loss = _per_sample_policy_loss.mean()
                _epoch_sample_policy_loss.append(_per_sample_policy_loss.detach().cpu())
                # (r-1)*A, weighted the same way as _per_sample_policy_loss
                # (mb_w) for direct comparability -- see "[policy surrogate]"
                # below for why this is logged unclipped/no-dual-clip.
                _epoch_sample_surrogate.append(((surr1 - mb_adv) * mb_w).detach().cpu())

                # Per-head counterfactual policy_loss breakdown -- see
                # all_head_policy_loss's own comment above for why this is a
                # counterfactual (each head's ratio in isolation, others
                # held at 1) rather than a strict decomposition of the
                # scalar above. No extra network forward pass: d_heads/
                # e_heads are already computed; this just re-slices their
                # existing logits/distributions per head (same cost as the
                # ratio-spike diagnostic's per-head call a few lines up).
                # Dual-clip (if enabled) is applied here too, for the same
                # reason it's applied to the val-side diagnostic -- keeps
                # this breakdown consistent with whatever the real,
                # backpropagated policy_loss formula actually is.
                if "head_log_probs" in batch:
                    with torch.no_grad():
                        _mb_old_head_lp = batch["head_log_probs"][mb_idx].to(self.device)
                        _mb_new_head_lp = self._per_head_new_log_probs(d_heads, e_heads, mb_actions, em)
                        _head_ratio = self._loss_ratio(_mb_new_head_lp - _mb_old_head_lp)  # (batch, 15)
                        _head_adv = mb_adv.unsqueeze(-1)
                        _head_w = mb_w.unsqueeze(-1)
                        _head_surr1 = _head_ratio * _head_adv
                        _head_surr2 = torch.clamp(_head_ratio, 1.0 - clip, 1.0 + clip) * _head_adv
                        _head_min_surr = self._apply_dual_clip(
                            torch.min(_head_surr1, _head_surr2), _head_adv
                        )
                        _head_policy_loss = (
                            -(_head_min_surr * _head_w).mean(dim=0) if _mb_opp_mask is None
                            else -(_head_min_surr * _head_w * _mb_opp_mask).sum(dim=0) / _mb_opp_mask.sum(dim=0).clamp(min=1.0)
                        )
                    all_head_policy_loss.append(_head_policy_loss.cpu())

                # Value loss — normalise by return variance so it stays ~O(1)
                # regardless of how large/negative the returns are. This keeps
                # the value gradient from overwhelming the policy gradient.
                ret_var = returns.var().clamp(min=1.0)
                _per_sample_value_loss = (new_values - mb_ret).pow(2) / ret_var
                value_loss = _per_sample_value_loss.mean()
                _epoch_sample_value_loss.append(_per_sample_value_loss.detach().cpu())
                _accum_value_by_outcome(new_values, mb_ret, mb_idx)

                # Entropy bonus
                _opp_entropy_masks = None
                if _mb_opp_mask is not None:
                    _ik, _it = HEAD_LP_KEYS.index("kick"), HEAD_LP_KEYS.index("tackle_attempt")
                    _opp_entropy_masks = {"kick": _mb_opp_mask[:, _ik], "tackle_attempt": _mb_opp_mask[:, _it]}
                    if _opp_grad_scale is not None:
                        _opp_entropy_masks["kick_scale"] = float(_opp_grad_scale[_ik])
                        _opp_entropy_masks["tackle_attempt_scale"] = float(_opp_grad_scale[_it])
                entropy, _ent_bkdn, _kick_ent_boost = self._compute_entropy(
                    d_heads, e_heads, em, return_breakdown=True, return_kick_boost=True, opp_masks=_opp_entropy_masks)
                for _ek, _ev in _ent_bkdn.items():
                    all_entropy_breakdown.setdefault(_ek, []).append(_ev)

                # No dir_l2 penalty needed: direction means are unit-normalized in
                # forward() so their magnitude is always 1 — penalizing it is a no-op.

                # log_kappa/log_std restoring force: without this, the entropy bonus
                # is a one-directional force that only ever DEFLATES
                # move_dir_log_kappa/kick_dir_log_kappa (entropy is monotonic
                # DECREASING in log_kappa -- the inverse relationship of the old
                # log_std, where entropy increased with it) and inflates
                # kick_dir_z_log_std (still a plain log_std, same direction as
                # before), and nothing in the PPO/BC losses pulls either back the
                # other way (the mean is unit-normalized so dir_l2 above is a
                # no-op on these too). This term adds an explicit L2 pull toward
                # each target, independent of clamp (which only caps the value
                # fed into the distribution and zeroes the gradient once the raw
                # parameter drifts past the bound — see ai_trainer_knowledge.md /
                # VonMisesDirectionHead/KickDirectionHead for the clamp mechanism).
                # Coefficient 0.0 (default) fully disables this — opt-in.
                dir_log_std_reg = torch.zeros(1, device=self.device)
                _lsm_raw = self.execution_net.move_dir_log_kappa
                _lsk_raw = self.execution_net.kick_dir_log_kappa
                _lskz_raw = self.execution_net.kick_dir_z_log_std
                if self.move_dir_log_kappa_reg_coef > 0.0:
                    dir_log_std_reg = dir_log_std_reg + self.move_dir_log_kappa_reg_coef * ((_lsm_raw - self.move_dir_log_kappa_target) ** 2).mean()
                if self.kick_dir_log_kappa_reg_coef > 0.0:
                    dir_log_std_reg = dir_log_std_reg + self.kick_dir_log_kappa_reg_coef * ((_lsk_raw - self.kick_dir_log_kappa_target) ** 2).mean()
                if self.kick_dir_z_log_std_reg_coef > 0.0:
                    dir_log_std_reg = dir_log_std_reg + self.kick_dir_z_log_std_reg_coef * ((_lskz_raw - self.kick_dir_z_log_std_target) ** 2).mean()

                # Raw-vector magnitude regularizer for move_direction/
                # kick_direction (see direction_magnitude_reg()'s own
                # docstring in bc.py) -- deliberately UNCONDITIONAL here,
                # independent of has_bc/bc_coeff below: the failure mode it
                # guards against (||raw|| collapsing toward 0, amplifying
                # the L2-normalize op's gradient by ~1/||raw||) is exactly
                # as relevant to PPO's own policy-gradient term as to the
                # BC-aux loss, and must not disappear once BC annealing
                # reaches 0 (which is precisely when PPO's own gradient is
                # the only thing left training these heads).
                dir_mag_reg = direction_magnitude_reg(e_heads, self._bc_dir_mag_reg_coef)

                total_loss = (policy_loss
                              + self.vf_coef * value_loss
                              - ent_coef * (entropy + _kick_ent_boost)
                              + dir_log_std_reg
                              + dir_mag_reg)

                # BC auxiliary loss (decision + execution, annealed to 0).
                # Split into a "kick group" (kick_this_tick/kick_direction/
                # kick_power/kick_spin), a "tackle group" (tackle_attempt),
                # and "other" (everything else), each scaled by its own
                # annealed coefficient -- see
                # bc_loss_from_tensor(split_kick=True, split_tackle=True) in
                # bc.py and TrainingSchedules.bc/bc_kick/bc_tackle in
                # schedules.py. bc_loss_val (the unweighted combined total)
                # is kept only for logging.
                bc_loss_val = torch.zeros(1, device=self.device)
                if has_bc:
                    mb_bc = batch["bc_labels"][mb_idx].to(self.device)
                    bc_loss_val, _bkdn, _bsplit = bc_loss_from_tensor(
                        mb_bc, self._bc_heads_for_loss(d_heads), e_heads,
                        direction_loss_weight=self._bc_dir_loss_w,
                        direction_loss_mode=self._bc_dir_loss_mode,
                        region_loss_weight=self._bc_region_loss_w,
                        pos_weight_kick=self._bc_pos_weight_kick,
                        pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                        dec_weight=self._bc_dec_weight,
                        exec_weight=self._bc_exec_weight,
                        dec_label_smoothing=self._bc_dec_label_smoothing,
                        exec_label_smoothing=self._bc_exec_label_smoothing,
                        return_breakdown=True,
                        split_kick=True,
                        split_tackle=True,
                    )
                    total_loss = total_loss + (
                        bc_coeff * _bsplit["other_group_loss"]
                        + bc_kick_coeff * _bsplit["kick_group_loss"]
                        + bc_tackle_coeff * _bsplit["tackle_group_loss"]
                    )
                    # Track tackle_attempt BCE separately for diagnostics
                    all_bc_tackle_loss.append(_bkdn.get("tackle_attempt", 0.0))
                # Track mean tackle/kick activation probability (pre-sampling) for logging.
                with torch.no_grad():
                    tackle_prob_t = torch.sigmoid(e_heads.tackle_attempt_logit).mean()
                    kick_prob_t = torch.sigmoid(e_heads.kick_logit).mean()
                # Single combined sync for all four -- same values as four
                # separate .item() calls, one CPU<->GPU round trip instead.
                _ratio_clip_f, _tackle_prob_f, _kick_prob_f, _bc_loss_f = torch.stack([
                    ratio_clipped_frac_t, tackle_prob_t, kick_prob_t, bc_loss_val.detach().squeeze(),
                ]).tolist()
                all_ratio_clipped_frac.append(_ratio_clip_f)
                all_tackle_prob.append(_tackle_prob_f)
                all_kick_prob.append(_kick_prob_f)
                all_bc_loss.append(_bc_loss_f)

                # TEMPORARY diagnostic (see _dump_nan_diagnostic docstring).
                if not torch.isfinite(total_loss):
                    self._dump_nan_diagnostic(
                        "forward", epoch_i, start // self.minibatch_size, mb_idx,
                        [batch["track_ids"][i] for i in mb_idx.tolist()],
                        d_heads, e_heads, policy_loss, value_loss, entropy,
                    )

                self.optimizer.zero_grad()
                if self.separate_value_net:
                    self.value_net_optimizer.zero_grad()
                total_loss.backward()
                # d_heads_for_value was detached above (unless
                # share_value_grad_with_decision), so value_loss's gradient
                # (folded into total_loss) normally only reaches self.value_net's
                # params here -- decision_net/execution_net's policy heads never
                # see it. When share_value_grad_with_decision is on, a
                # decision_value_coef-scaled slice of it also lands on
                # decision_net's own .grad by this point (accumulated alongside
                # policy_loss's contribution from the same backward) -- picked up
                # normally by self.optimizer's step()/grad-clipping below, same as
                # any other decision_net gradient this minibatch.
                if self.separate_value_net:
                    nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                    self.value_net_optimizer.step()

                # Per-execution-head gradient norm, BEFORE the optimiser step
                # (measurement only — max_norm=inf never rescales, same trick as
                # raw_grad_norm below). Every norm below is collected as a
                # TENSOR here (clip_grad_norm_'s own clipping side effect, where
                # it has one, still happens at its normal call site and order --
                # only the .item() EXTRACTION is deferred) and synced together
                # in one combined call right after -- same values as one
                # .item() per tensor, one CPU<->GPU round trip instead of
                # ~(len(EXEC_HEAD_MODULES) + 3).
                _gn_tensors: dict[str, torch.Tensor] = {}
                for _head_name, _attr in EXEC_HEAD_MODULES:
                    _head_params = list(getattr(self.execution_net, _attr).parameters())
                    if _head_params:
                        _gn_tensors[f"head:{_head_name}"] = torch.nn.utils.clip_grad_norm_(_head_params, float("inf"))

                # Capture dir_log_kappa gradient before it is zeroed
                _mv_ls_grad = self.execution_net.move_dir_log_kappa.grad
                if _mv_ls_grad is not None:
                    _gn_tensors["mv_log_std_grad"] = _mv_ls_grad.norm()

                # Grad norm BEFORE clipping
                _gn_tensors["raw"] = torch.nn.utils.clip_grad_norm_(
                    list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                    float("inf"),  # don't clip yet, just measure
                )
                # TEMPORARY diagnostic (see _dump_nan_diagnostic docstring).
                # total_loss WAS finite (checked above) but the gradient
                # isn't -- points at a backward-pass numerical singularity,
                # not corrupted forward-pass data.
                if not torch.isfinite(_gn_tensors["raw"]):
                    self._dump_nan_diagnostic(
                        "backward", epoch_i, start // self.minibatch_size, mb_idx,
                        [batch["track_ids"][i] for i in mb_idx.tolist()],
                        d_heads, e_heads, policy_loss, value_loss, entropy,
                    )
                # Direction-head params (move_direction/kick_direction weights +
                # move_dir_log_kappa/kick_dir_log_kappa/kick_dir_z_log_std) are clipped in their own
                # isolated group via direction_max_grad_norm, so a single sample's
                # large direction gradient can no longer force a proportional
                # shrink of every other head's gradient in the same step (and a
                # calm direction gradient no longer "borrows" clip headroom from
                # a genuinely large gradient elsewhere). Falls back to sharing
                # max_grad_norm when direction_max_grad_norm is unset (None).
                _non_direction_params = [
                    p for p in list(self.decision_net.parameters()) + list(self.execution_net.parameters())
                    if id(p) not in self.direction_param_ids
                ]
                _direction_params = [
                    p for p in self.execution_net.parameters()
                    if id(p) in self.direction_param_ids
                ]
                _gn_tensors["main"] = nn.utils.clip_grad_norm_(_non_direction_params, self.max_grad_norm)
                if _direction_params:
                    _gn_tensors["dir"] = nn.utils.clip_grad_norm_(_direction_params, self.direction_max_grad_norm)
                    _dir_params_before = [p.detach().clone() for p in _direction_params]

                _gn_names = list(_gn_tensors.keys())
                # torch.nn.utils.clip_grad_norm_ returns a CPU tensor (not on
                # the parameters' own device) whenever every parameter in that
                # call's group has .grad is None -- e.g. a head that happened
                # to get zero gradient this exact minibatch (frozen from the
                # PPO ratio for this curriculum phase and BC aux already
                # annealed to ~0, or simply no rows touching that head this
                # step). torch.stack() below then fails cross-device against
                # the other (real, self.device-resident) norms. .to(self.device)
                # is a no-op when already on the right device, so this doesn't
                # change anything for the common case.
                _gn_vals = dict(zip(
                    _gn_names,
                    torch.stack([_gn_tensors[k].to(self.device) for k in _gn_names]).tolist(),
                ))

                for _head_name, _attr in EXEC_HEAD_MODULES:
                    _gnk = f"head:{_head_name}"
                    if _gnk in _gn_vals:
                        all_head_grad_norm[_head_name].append(_gn_vals[_gnk])
                if "mv_log_std_grad" in _gn_vals:
                    all_mv_log_std_grad.append(_gn_vals["mv_log_std_grad"])
                raw_grad_norm = _gn_vals["raw"]
                # Spike diagnostic: the [exec head grad norm]/main-vs-dir
                # summaries logged at the end of a rollout are aggregated
                # over every minibatch and only split by execution-head --
                # neither tells you which specific parameter TENSOR (across
                # BOTH decision_net and execution_net, including encoders/
                # trunk, not just the execution heads) drove an individual
                # outlier step. Shared with BC/DAgger's own minibatch loops
                # (see _maybe_log_grad_norm_spike's own docstring for the
                # adaptive running-mean/std bar and why it isn't a fixed
                # absolute number). Must read .grad here, before
                # self.optimizer.step() below (which doesn't zero grads
                # itself, but there's no reason to risk it).
                _maybe_log_grad_norm_spike(self, "PPO", epoch_i, start // self.minibatch_size, raw_grad_norm)
                _gn_main = _gn_vals["main"]
                all_grad_norm_main.append(_gn_main)
                if _gn_main > self.max_grad_norm:
                    clip_triggered_main += 1
                if _direction_params:
                    _gn_dir = _gn_vals["dir"]
                    all_grad_norm_dir.append(_gn_dir)
                    if _gn_dir > self.direction_max_grad_norm:
                        clip_triggered_dir += 1
                if self.lr_warmup_steps > 0 and _warm_steps_done < self.lr_warmup_steps:
                    _ws = _lr_warmup_scale(_warm_steps_done, self.lr_warmup_steps)
                    for _pg, _base in zip(self.optimizer.param_groups, _warm_base_lrs):
                        _pg["lr"] = _base * _ws
                _warm_steps_done += 1
                self.optimizer.step()
                if self.lr_warmup_steps > 0 and _warm_steps_done == self.lr_warmup_steps:
                    for _pg, _base in zip(self.optimizer.param_groups, _warm_base_lrs):
                        _pg["lr"] = _base
                if _direction_params:
                    _dir_delta_norm = torch.sqrt(sum(
                        (p.detach() - p_before).pow(2).sum()
                        for p, p_before in zip(_direction_params, _dir_params_before)
                    )).item()
                    all_param_delta_dir.append(_dir_delta_norm)

                # After step: measure KL and direction mean shift
                with torch.no_grad():
                    d_after = self.decision_net(
                        sf, of, em, bf, gf, sat, oat,
                        ball_physics_full=mb_obs.get("ball_physics_full"),
                        self_physics_full=mb_obs.get("self_physics_full"),
                        other_physics_full=mb_obs.get("other_physics_full"),
                    )
                    e_after = self.execution_net(sf, of, em, bf, gf, d_after, sat, oat)
                    lp_after = self._recompute_log_prob(d_after, e_after, mb_actions, em)
                    if _mb_opp_mask is not None:
                        lp_after = self._mask_total_log_probs(
                            lp_after, self._per_head_new_log_probs(d_after, e_after, mb_actions, em), _mb_opp_mask,
                        )
                    # Kept as tensors (not .item()'d) -- see the single
                    # combined sync below, gathering every scalar diagnostic
                    # in this whole "after step" section (roughly a dozen
                    # values) into one CPU<->GPU round trip instead of one
                    # per value.
                    movedir_mean_shift_t = (e_after.move_direction - e_heads.move_direction).norm(dim=-1).mean()
                    kickdir_mean_shift_t = (e_after.kick_direction - e_heads.kick_direction).norm(dim=-1).mean()
                    # Actual KL contribution from move_direction (now included in ratio).
                    _stored_raw_mb = mb_actions["move_dir_raw"]
                    _log_std_move  = self.execution_net.move_dir_log_kappa.to(self.device)
                    _lp_movedir_before = self._move_dir_head(e_heads.move_direction, _log_std_move).log_prob(_stored_raw_mb)
                    _lp_movedir_after  = self._move_dir_head(e_after.move_direction, _log_std_move).log_prob(_stored_raw_mb)
                    movedir_hyp_kl_t = (_lp_movedir_before - _lp_movedir_after).mean()

                    # --- AFTER the optimiser step: per-head KL, continuous head
                    # mean/log_std drift, discrete head logit drift. Guarded on
                    # "head_log_probs" presence -- always populated by
                    # RolloutBuffer.add()/as_tensors() and propagated through
                    # augment_batch(), but stay defensive against any future
                    # caller that builds a batch dict by hand without it.
                    per_head_kl_mb = None
                    if "head_log_probs" in batch:
                        mb_old_head_lp = batch["head_log_probs"][mb_idx].to(self.device)
                        per_head_new_lp_after = self._per_head_new_log_probs(d_after, e_after, mb_actions, em)
                        per_head_kl_mb = (
                            (mb_old_head_lp - per_head_new_lp_after).mean(dim=0) if _mb_opp_mask is None
                            else ((mb_old_head_lp - per_head_new_lp_after) * _mb_opp_mask).sum(dim=0)
                            / _mb_opp_mask.sum(dim=0).clamp(min=1.0)
                        )  # (15,)
                        all_head_kl.append(per_head_kl_mb.detach().cpu())

                    _shift_tensors: dict[str, torch.Tensor] = {
                        "movedir_mean_shift": movedir_mean_shift_t,
                        "kickdir_mean_shift": kickdir_mean_shift_t,
                        "movedir_hyp_kl": movedir_hyp_kl_t,
                        "log_std_shift_move": (self.execution_net.move_dir_log_kappa.detach() - _log_std_move_before).abs().mean(),
                        "log_std_shift_kick": (self.execution_net.kick_dir_log_kappa.detach() - _log_std_kick_before).abs().mean(),
                        "log_std_shift_kickz": (self.execution_net.kick_dir_z_log_std.detach() - _log_std_kickz_before).abs().mean(),
                        "logit_shift_exec_move": (e_after.exec_move_logit - _exec_move_logit_before).abs().mean(),
                        "logit_shift_sprint": (e_after.sprint_logit - _sprint_logit_before).abs().mean(),
                        "logit_shift_kick": (e_after.kick_logit - _kick_logit_before).abs().mean(),
                        "logit_shift_tackle_attempt": (e_after.tackle_attempt_logit - _tackle_attempt_logit_before).abs().mean(),
                    }

                # Clamp to finite floor before KL to avoid inf from near-zero-probability
                # samples in the current policy (log_prob = -inf → KL = +inf).
                _lp_clamped = new_log_probs.clamp(min=-1e6)
                _la_clamped = lp_after.clamp(min=-1e6)
                _shift_tensors["approx_kl"] = (mb_old_lp - _lp_clamped).mean()
                _shift_tensors["kl_after_step"] = (mb_old_lp - _la_clamped).mean()
                _shift_tensors["total_loss"] = total_loss.detach().squeeze()
                _shift_tensors["policy_loss"] = policy_loss.detach()
                _shift_tensors["value_loss"] = value_loss.detach()
                _shift_tensors["entropy"] = entropy.detach().squeeze()

                # Single combined sync for every scalar gathered above -- same
                # values as ~15 separate .item() calls, one CPU<->GPU round
                # trip instead.
                _shift_names = list(_shift_tensors.keys())
                _shift_vals = dict(zip(_shift_names, torch.stack([_shift_tensors[k] for k in _shift_names]).tolist()))

                movedir_mean_shift = _shift_vals["movedir_mean_shift"]
                kickdir_mean_shift = _shift_vals["kickdir_mean_shift"]
                movedir_hyp_kl = _shift_vals["movedir_hyp_kl"]
                all_continuous_mean_shift["move_direction"].append(movedir_mean_shift)
                all_continuous_mean_shift["kick_direction"].append(kickdir_mean_shift)
                all_continuous_log_std_shift["move_direction"].append(_shift_vals["log_std_shift_move"])
                all_continuous_log_std_shift["kick_direction"].append(_shift_vals["log_std_shift_kick"])
                all_kickz_log_std_shift.append(_shift_vals["log_std_shift_kickz"])
                all_discrete_logit_shift["exec_move"].append(_shift_vals["logit_shift_exec_move"])
                all_discrete_logit_shift["sprint"].append(_shift_vals["logit_shift_sprint"])
                all_discrete_logit_shift["kick"].append(_shift_vals["logit_shift_kick"])
                all_discrete_logit_shift["tackle_attempt"].append(_shift_vals["logit_shift_tackle_attempt"])
                approx_kl = _shift_vals["approx_kl"]
                kl_after_step = _shift_vals["kl_after_step"]
                _total_loss_f = _shift_vals["total_loss"]
                _policy_loss_f = _shift_vals["policy_loss"]
                _value_loss_f = _shift_vals["value_loss"]
                _entropy_f = _shift_vals["entropy"]

                mb_i = start // self.minibatch_size
                # f-strings inside log.debug(...) are eagerly evaluated
                # regardless of the active log level -- guard explicitly
                # (matching the OTHER debug block earlier in this loop) so
                # this doesn't force string formatting (harmless) AND every
                # value it references (already-batched above, so no extra
                # sync either way now) on every minibatch when DEBUG isn't
                # even enabled.
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(
                        f"  [e{epoch_i} mb{mb_i:02d}]"
                        f"  grad={raw_grad_norm:.1f}"
                        f"  total={_total_loss_f:.3f}"
                        f"  pol={_policy_loss_f:.3f}"
                        f"  val={_value_loss_f:.3f}(x{self.vf_coef})={self.vf_coef * _value_loss_f:.3f}"
                        f"  kl={kl_after_step:.4f}"
                        f"  mv_shift={movedir_mean_shift:.3f}"
                        f"  mv_kl={movedir_hyp_kl:.4f}"
                        f"  kk_shift={kickdir_mean_shift:.3f}"
                    )

                all_policy_loss.append(_policy_loss_f)
                all_value_loss.append(_value_loss_f)
                all_entropy.append(_entropy_f)
                all_kl.append(kl_after_step)

                # Early stop per-minibatch: limits drift to O(1) gradient step
                # past the trust region boundary rather than O(n_minibatches).
                if kl_after_step > self.target_kl:
                    mb_i_stop = start // self.minibatch_size
                    _stop_head_kl_str = (
                        "  ".join(
                            f"{k}={v:+.4f}" for k, v in zip(HEAD_LP_KEYS, per_head_kl_mb.tolist())
                            if abs(v) > 0.001
                        )
                        if per_head_kl_mb is not None else "(head_log_probs unavailable)"
                    )
                    log.info(
                        f"  [early stop e{epoch_i} mb{mb_i_stop}]"
                        f"  KL={kl_after_step:.5f} > target={self.target_kl}"
                        f"  steps_this_update={len(all_kl)}\n"
                        f"    [per-head KL] {_stop_head_kl_str}"
                    )
                    _early_stopped = True
                    break

            epoch_times.append((time.perf_counter() - epoch_start) * 1000)
            # Full-epoch means (every minibatch this epoch actually ran, not
            # a single sampled one) -- reading pol=/kl= off any one minibatch
            # is effectively a random draw, not a trend; two consecutive
            # single-minibatch values swinging between e.g. -0.017 and +0.026
            # (both close to zero) is easy to misread as the policy getting
            # worse or better within the update when it's just ordinary
            # sampling noise. This line answers "does this actually degrade
            # deeper into the update" directly, once per epoch.
            _epoch_pol = all_policy_loss[_epoch_slice_start:]
            _epoch_kl = all_kl[_epoch_slice_start:]
            _epoch_val = all_value_loss[_epoch_slice_start:]
            mean_pol_epoch = float(np.mean(_epoch_pol)) if _epoch_pol else 0.0
            mean_kl_epoch = float(np.mean(_epoch_kl)) if _epoch_kl else 0.0
            mean_val_epoch = float(np.mean(_epoch_val)) if _epoch_val else 0.0
            # Trimmed (p10-p90) companions to pol_mean/val_mean -- pol_mean's
            # uses the SAME _epoch_sample_policy_loss pool the percentile
            # line below already builds (no extra work); val_mean's is this
            # function's TRAIN-side value-HEAD loss (as opposed to
            # val_episode_value_loss_p10_p90, the HELD-OUT set's own trimmed
            # value loss; see _epoch_sample_value_loss's own comment for the
            # "val" = value-head vs val = held-out-set naming clash).
            _epoch_pol_cat_for_trim = (
                torch.cat(_epoch_sample_policy_loss) if _epoch_sample_policy_loss else torch.zeros(0)
            )
            _epoch_val_cat_for_trim = (
                torch.cat(_epoch_sample_value_loss) if _epoch_sample_value_loss else torch.zeros(0)
            )
            mean_pol_epoch_trimmed = (
                _trimmed_mean_p10_p90(_epoch_pol_cat_for_trim) if _epoch_sample_policy_loss else float("nan")
            )
            mean_pol_epoch_trimmed2 = (
                _trimmed_mean_p25_p75(_epoch_pol_cat_for_trim) if _epoch_sample_policy_loss else float("nan")
            )
            mean_val_epoch_trimmed = (
                _trimmed_mean_p10_p90(_epoch_val_cat_for_trim) if _epoch_sample_value_loss else float("nan")
            )
            mean_val_epoch_trimmed2 = (
                _trimmed_mean_p25_p75(_epoch_val_cat_for_trim) if _epoch_sample_value_loss else float("nan")
            )
            # E[(r-1)*A], unclipped/no-dual-clip -- see "[policy surrogate]"
            # log line below for the rationale. Same progressive-pool
            # caveat as _epoch_sample_policy_loss above (this epoch's pool
            # mixes rows shaped by progressively-more-trained weights).
            _epoch_surrogate_cat = (
                torch.cat(_epoch_sample_surrogate) if _epoch_sample_surrogate else torch.zeros(0)
            )
            mean_surrogate_epoch = float(_epoch_surrogate_cat.mean()) if _epoch_sample_surrogate else float("nan")
            mean_surrogate_epoch_trimmed = (
                _trimmed_mean_p10_p90(_epoch_surrogate_cat) if _epoch_sample_surrogate else float("nan")
            )
            mean_surrogate_epoch_trimmed2 = (
                _trimmed_mean_p25_p75(_epoch_surrogate_cat) if _epoch_sample_surrogate else float("nan")
            )
            # --- Per-epoch summary, one topic per line (train side, then
            # held-out side if val_episode_fraction is enabled) -- replaces
            # the old single mega-line-per-topic format. "[policy]"/"[value]"
            # are this function's TRAIN-side numbers; "[held-out policy]"/
            # "[held-out value]" are the HELD-OUT SET's (see
            # _epoch_sample_value_loss's own comment for the "val" = value-
            # head vs val = held-out-set naming clash this deliberately
            # avoids by not using "val" as a line tag at all). ---
            log.info(f"  == epoch {epoch_i + 1}/{self.n_epochs} " + "=" * 32)
            log.info(
                f"  [policy] mean={mean_pol_epoch:.4f}  p10_p90={mean_pol_epoch_trimmed:.4f}  "
                f"p25_p75={mean_pol_epoch_trimmed2:.4f}  "
                f"kl={mean_kl_epoch:.4f}  (n={len(_epoch_kl)} minibatch(es), {epoch_times[-1]:.0f}ms)"
            )
            if _epoch_sample_surrogate:
                # E[(r-1)*A] -- the UNCLIPPED, no-dual-clip linearised
                # surrogate. Mathematically, [policy]'s mean above ==
                # -mean_surrogate_epoch exactly whenever the clip/dual-clip
                # floor are inactive for a row (mean(A)~=0 by
                # normalisation) -- so on a stable step this line will read
                # close to the negative of [policy]'s mean, NOT a
                # different number. Its value is in the rows where clipping
                # DOES engage (far more common here, mid-optimization, than
                # on the held-out side): [policy] intentionally floors those
                # to protect the actual gradient step, which also hides how
                # far ratios really moved -- this line shows the true
                # unclipped magnitude, trimmed the same way as every other
                # per-sample line here since a single ratio-spike row is
                # just as capable of dominating an unclipped mean.
                log.info(
                    f"  [policy surrogate] mean={mean_surrogate_epoch:.4f}  "
                    f"p10_p90={mean_surrogate_epoch_trimmed:.4f}  "
                    f"p25_p75={mean_surrogate_epoch_trimmed2:.4f}  "
                    f"(n={len(_epoch_surrogate_cat):,})"
                )
            if _epoch_sample_policy_loss:
                # Distribution shape, not just the mean -- e.g. a few
                # huge-magnitude samples can drag the mean positive even
                # while the bulk of samples sit right around/below 0. Same
                # percentile list/style as the existing "[advantage |.| by
                # episode]" line above, and as "[held-out policy percentiles]"
                # below -- but see _epoch_sample_policy_loss's own comment
                # (declared near the top of this function) for why this one
                # is a progressive pool across the epoch's minibatches (rows
                # shaped by different, progressively-more-trained weights),
                # not one clean snapshot the way the held-out numbers are.
                _epoch_pol_t = _epoch_pol_cat_for_trim
                _pcts = [0, 1, 10, 50, 90, 99, 100]
                _epoch_pct_str = "  ".join(
                    f"p{p}={float(_epoch_pol_t.quantile(p / 100.0)):.4f}" for p in _pcts
                )
                log.info(f"  [policy percentiles, n={len(_epoch_pol_t):,}]  {_epoch_pct_str}")
            # Which heads are actually driving [policy] mean above -- the
            # SAME per-head COUNTERFACTUAL breakdown as the end-of-run
            # "[per-head policy_loss (counterfactual, ...)]" line (each
            # head's own ratio in isolation, others held at ratio=1), just
            # sliced to THIS epoch's minibatches only (same _epoch_slice_start
            # window as _epoch_pol/_epoch_kl above) instead of the whole
            # update. NOT a strict decomposition -- see all_head_policy_loss's
            # own comment for why the 15 values don't sum to mean_pol_epoch.
            _epoch_head_pol = all_head_policy_loss[_epoch_slice_start:]
            if _epoch_head_pol:
                _mean_epoch_head_pol = torch.stack(_epoch_head_pol).mean(dim=0)
                _inactive_heads = self._inactive_head_lp_keys()
                _epoch_head_pol_str = "  ".join(
                    f"{k}={v:+.4f}" for k, v in zip(HEAD_LP_KEYS, _mean_epoch_head_pol.tolist())
                    if k not in _inactive_heads
                )
                log.info(f"  [policy heads] {_epoch_head_pol_str}")
            log.info(
                f"  [value] mean={mean_val_epoch:.4f}(x{self.vf_coef})={self.vf_coef * mean_val_epoch:.4f}  "
                f"p10_p90={mean_val_epoch_trimmed:.4f}(x{self.vf_coef})="
                f"{self.vf_coef * mean_val_epoch_trimmed:.4f}  "
                f"p25_p75={mean_val_epoch_trimmed2:.4f}(x{self.vf_coef})="
                f"{self.vf_coef * mean_val_epoch_trimmed2:.4f}"
            )
            if val_batch is not None:
                # held-out policy loss is NOT a generalization diagnostic --
                # at ratio=1 (i.e. for any row that received zero gradient
                # steps this call), the clipped surrogate collapses to
                # -mean(adv_normalized), which is exactly 0 by construction
                # (advantages are normalized to zero mean) -- confirmed
                # empirically: a same-size sample of about-to-be-trained
                # TRAIN rows reads 0.000000 too, measured before any epoch
                # runs. So it reads near-zero for ANY untouched rows
                # regardless of split quality or true generalization, and
                # says nothing that [policy] above doesn't. Only [held-out
                # value] (real regression MSE, no ratio/zero-mean-advantage
                # artifact) is a meaningful apples-to-apples comparison.
                (
                    _val_pol_loss, _val_val_loss, _val_kl, _val_per_sample_pol,
                    _val_val_loss_trimmed, _val_val_loss_trimmed2,
                    _val_pol_loss_trimmed, _val_pol_loss_trimmed2,
                    _val_surrogate_mean, _val_surrogate_trimmed, _val_surrogate_trimmed2,
                    _val_head_pol,
                ) = self._eval_val_episode_losses(val_batch, clip)
                log.info(
                    f"  [held-out policy] mean={_val_pol_loss:.4f}  "
                    f"p10_p90={_val_pol_loss_trimmed:.4f}  p25_p75={_val_pol_loss_trimmed2:.4f}  "
                    f"kl={_val_kl:.4f}  "
                    f"(n={len(val_batch['log_probs'])} held-out steps, never trained on)"
                )
                # E[(r-1)*A] -- the UNCLIPPED, no-dual-clip linearised
                # surrogate, mathematically == -[held-out policy]'s own mean
                # whenever the clip/dual-clip floor are inactive for a row
                # (see _eval_val_episode_losses' docstring). NOT a different
                # signal in the common case -- held-out rows rarely drift far
                # from ratio=1 since nothing here ever gets a direct
                # gradient, so this and [held-out policy] will usually track
                # each other closely. Its value is narrower than that:
                # it's a tripwire for the rare held-out row whose ratio DID
                # drift past the clip band purely from the shared network
                # being trained on OTHER (train-side) rows -- exactly the
                # kind of real cross-row generalisation drift [held-out
                # policy] silently floors away. Trimmed the same way as
                # every other per-sample line here since it's unclipped and
                # therefore just as exposed to a single ratio-spike row.
                log.info(
                    f"  [held-out policy surrogate] mean={_val_surrogate_mean:.4f}  "
                    f"p10_p90={_val_surrogate_trimmed:.4f}  p25_p75={_val_surrogate_trimmed2:.4f}  "
                    f"(n={len(_val_per_sample_pol)} held-out steps)"
                )
                # Same distribution-shape percentiles as [policy percentiles]
                # above, but for the held-out set, every epoch (cheap:
                # val_batch is fixed, this is the same no-grad pass
                # _eval_val_episode_losses already runs to get the mean
                # above -- no extra forward pass).
                if len(_val_per_sample_pol) > 0:
                    _pcts = [0, 1, 10, 50, 90, 99, 100]
                    _val_pct_str = "  ".join(
                        f"p{p}={float(_val_per_sample_pol.quantile(p / 100.0)):.4f}" for p in _pcts
                    )
                    log.info(
                        f"  [held-out policy percentiles, n={len(_val_per_sample_pol):,}]  {_val_pct_str}"
                    )
                if _val_head_pol is not None:
                    _inactive_heads = self._inactive_head_lp_keys()
                    _val_head_pol_str = "  ".join(
                        f"{k}={v:+.4f}" for k, v in zip(HEAD_LP_KEYS, _val_head_pol.tolist())
                        if k not in _inactive_heads
                    )
                    log.info(f"  [held-out policy heads] {_val_head_pol_str}")
                log.info(
                    f"  [held-out value] mean={_val_val_loss:.4f}(x{self.vf_coef})="
                    f"{self.vf_coef * _val_val_loss:.4f}  "
                    f"p10_p90={_val_val_loss_trimmed:.4f}(x{self.vf_coef})="
                    f"{self.vf_coef * _val_val_loss_trimmed:.4f}  "
                    f"p25_p75={_val_val_loss_trimmed2:.4f}(x{self.vf_coef})="
                    f"{self.vf_coef * _val_val_loss_trimmed2:.4f}"
                )
            # Final line of the per-epoch block: pre-clip grad-norm spread for
            # this epoch's minibatches only (sliced the same way as
            # _epoch_pol/_epoch_kl/_epoch_val above, off the SAME
            # all_grad_norm_main/all_grad_norm_dir lists the end-of-call
            # "[grad clip]" summary uses over the whole rollout -- see that
            # line's own comment for why "main"/"direction" are reported as
            # separate isolated clip groups). mean/max alone (as in that
            # end-of-call line) hide whether a high mean is a broad shift or
            # one or two spikes -- std/min complete the picture per epoch.
            _epoch_gn_main = all_grad_norm_main[_epoch_slice_start:]
            _epoch_gn_dir = all_grad_norm_dir[_epoch_slice_start:] if all_grad_norm_dir else []
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
            if _early_stopped:
                break

        # --- Value-only continuation: the policy's KL early-stop above cuts the
        # combined policy+value loop short (often after just 1-3 minibatches), which
        # was starving the critic of gradient steps regardless of its own dedicated,
        # higher LR param group. Since policy and value are independent param groups
        # with independent gradients, keep training the value head alone (no policy
        # loss/backward, so no further KL risk) for the remaining epoch/minibatch
        # budget. This directly targets the "val stuck > 1.0 for the whole rollout"
        # symptom without touching the policy's trust region.
        if _early_stopped:
            value_only_steps = 0
            for _ in range(self.value_only_continuation_epochs):
                indices = torch.randperm(n)
                for start in range(0, n, self.minibatch_size):
                    mb_idx = indices[start:start + self.minibatch_size]
                    if len(mb_idx) == 0:
                        continue
                    mb_obs = {k.replace("obs/", ""): batch[k][mb_idx].to(self.device)
                              for k in batch if k.startswith("obs/")}
                    mb_ret = returns[mb_idx].to(self.device)

                    sf = mb_obs["self_feat"]
                    of = mb_obs["other_feat"]
                    em = mb_obs["exists_mask"]
                    bf = mb_obs["ball_feat"]
                    gf = mb_obs["global_feat"]
                    sat, oat = _ai_types(mb_obs)

                    with torch.no_grad():
                        d_heads_vo = self.decision_net(
                            sf, of, em, bf, gf, sat, oat,
                            ball_physics_full=mb_obs.get("ball_physics_full"),
                            self_physics_full=mb_obs.get("self_physics_full"),
                            other_physics_full=mb_obs.get("other_physics_full"),
                        )
                    if self.separate_value_net:
                        _value_vo = self.value_net(sf, of, em, bf, gf, d_heads_vo, sat, oat, value_only=True)
                    else:
                        _value_vo = self.execution_net(sf, of, em, bf, gf, d_heads_vo, sat, oat, value_only=True)
                    new_values_vo = _value_vo.squeeze(-1)

                    ret_var = returns.var().clamp(min=1.0)
                    value_loss_vo = F.mse_loss(new_values_vo, mb_ret) / ret_var
                    _accum_value_by_outcome(new_values_vo, mb_ret, mb_idx)

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

                    all_value_loss.append(value_loss_vo.item())
                    value_only_steps += 1
            if value_only_steps:
                log.info(
                    f"  [value-only continuation] {value_only_steps} extra minibatch step(s)"
                    f"  after policy early-stop  final_val_loss={all_value_loss[-1]:.4f}"
                )

        # --- BC-only continuation: same rationale as value-only continuation
        # above, but for the BC auxiliary loss (see bc.bc_only_continuation_epochs
        # in ai_config.json). Early-stop cuts the combined policy+value+BC loop
        # short, which was also silently truncating BC's intended annealed
        # gradient budget every rollout (not just the value head). BC updates
        # the SAME decision_net/execution_net params the policy uses (unlike
        # the value head's isolated param group), so this runs strictly after
        # the value-only continuation, using no policy forward/backward and
        # computing no ratio/KL — it cannot itself trigger further early-stops.
        #
        # Coefficient: uses bc.bc_only_continuation_coeff if set, otherwise
        # falls back to the same annealed bc_coeff used for the in-epoch BC aux
        # loss (prior behaviour). This lets you decouple the two - e.g. anneal
        # aux_coeff to 0.0 to stop BC from fighting the policy gradient during
        # the main epoch loop, while keeping a fixed bc_only_continuation_coeff
        # so this loop still nudges the network toward demo behaviour with
        # otherwise-unused post-early-stop gradient budget.
        bc_only_coeff = (
            self.bc_only_continuation_coeff
            if self.bc_only_continuation_coeff is not None
            else bc_coeff
        )
        if has_bc and self.bc_only_continuation_epochs > 0 and bc_only_coeff > 0.0:
            bc_only_steps = 0
            last_bc_only_loss = 0.0
            bc_clip_triggered_main = 0
            bc_clip_triggered_dir = 0
            bc_grad_norm_main: list[float] = []
            bc_grad_norm_dir: list[float] = []
            for _ in range(self.bc_only_continuation_epochs):
                indices = torch.randperm(n)
                for start in range(0, n, self.minibatch_size):
                    mb_idx = indices[start:start + self.minibatch_size]
                    if len(mb_idx) == 0:
                        continue
                    mb_obs = {k.replace("obs/", ""): batch[k][mb_idx].to(self.device)
                              for k in batch if k.startswith("obs/")}
                    mb_bc = batch["bc_labels"][mb_idx].to(self.device)

                    sf = mb_obs["self_feat"]
                    of = mb_obs["other_feat"]
                    em = mb_obs["exists_mask"]
                    bf = mb_obs["ball_feat"]
                    gf = mb_obs["global_feat"]
                    sat, oat = _ai_types(mb_obs)

                    d_heads_bo = self.decision_net(
                        sf, of, em, bf, gf, sat, oat,
                        ball_physics_full=mb_obs.get("ball_physics_full"),
                        self_physics_full=mb_obs.get("self_physics_full"),
                        other_physics_full=mb_obs.get("other_physics_full"),
                    )
                    e_heads_bo = self.execution_net(sf, of, em, bf, gf, d_heads_bo, sat, oat)

                    bc_loss_bo, _ = bc_loss_from_tensor(
                        mb_bc, self._bc_heads_for_loss(d_heads_bo), e_heads_bo,
                        direction_loss_weight=self._bc_dir_loss_w,
                        direction_loss_mode=self._bc_dir_loss_mode,
                        region_loss_weight=self._bc_region_loss_w,
                        pos_weight_kick=self._bc_pos_weight_kick,
                        pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                        dec_weight=self._bc_dec_weight,
                        exec_weight=self._bc_exec_weight,
                        dec_label_smoothing=self._bc_dec_label_smoothing,
                        exec_label_smoothing=self._bc_exec_label_smoothing,
                        return_breakdown=True,
                    )

                    if not bc_loss_bo.requires_grad:
                        # No valid BC rows in this minibatch (bc_loss_from_tensor's
                        # early-return path returns a disconnected zero tensor) —
                        # nothing to backprop, skip this minibatch rather than
                        # crashing on .backward() with no grad_fn.
                        continue

                    self.optimizer.zero_grad()
                    (bc_only_coeff * bc_loss_bo + direction_magnitude_reg(e_heads_bo, self._bc_dir_mag_reg_coef)).backward()
                    # Same direction-head isolation as the main policy loop above.
                    _non_direction_params_bo = [
                        p for p in list(self.decision_net.parameters()) + list(self.execution_net.parameters())
                        if id(p) not in self.direction_param_ids
                    ]
                    _direction_params_bo = [
                        p for p in self.execution_net.parameters()
                        if id(p) in self.direction_param_ids
                    ]
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

                    all_bc_loss.append(bc_loss_bo.detach().item())
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

        # --- KL diagnostics (fires whenever rollout KL exceeds threshold) ---
        mean_kl = float(np.mean(all_kl)) if all_kl else 0.0
        # Median KL alongside the mean: a handful of near-saturated Bernoulli
        # samples (e.g. rare kick/tackle_attempt at p≈0.01-0.05) can dominate the
        # plain mean via nonlinear log-ratio blowup without any real broad drift
        # — median is robust to that and shows whether "real" typical KL is low.
        median_kl = float(np.median(all_kl)) if all_kl else 0.0
        move_log_std = self.execution_net.move_dir_log_kappa.data.tolist()
        kick_log_std = self.execution_net.kick_dir_log_kappa.data.tolist()
        kick_z_log_std = self.execution_net.kick_dir_z_log_std.data.tolist()
        kick_power_log_std = self.execution_net.kick_power_log_std.data.tolist()
        mean_mv_ls_grad = float(np.mean(all_mv_log_std_grad)) if all_mv_log_std_grad else 0.0
        if mean_kl > KL_DIAG_THRESHOLD and all_ratios:
            ratios_t = torch.cat(all_ratios)
            ratios_max = ratios_t.max()  # true max, before any subsampling below
            # torch.quantile() hard-errors ("input tensor is too large") past
            # 16,777,216 elements -- with enough epochs/minibatches (e.g. a
            # KL-triggered run with n_epochs in the hundreds) all_ratios'
            # concatenation can exceed that easily. This is a diagnostic
            # printout, not a training-affecting computation, so an unbiased
            # random subsample is just as informative for p5/p25/p50/p75/p95
            # -- max above is already taken from the full tensor so we don't
            # lose the one statistic a subsample could plausibly miss.
            _MAX_QUANTILE_ELEMENTS = 8_000_000
            if ratios_t.numel() > _MAX_QUANTILE_ELEMENTS:
                _idx = torch.randint(
                    0, ratios_t.numel(), (_MAX_QUANTILE_ELEMENTS,), device=ratios_t.device,
                )
                ratios_t = ratios_t.flatten()[_idx]
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
            # Per-head new log_prob means on stored actions (first 256 transitions)
            diag_n = min(256, n)
            diag_obs = {k.replace("obs/", ""): batch[k][:diag_n].to(self.device)
                        for k in batch if k.startswith("obs/")}
            diag_act = {k.replace("action/", ""): batch[k][:diag_n].to(self.device)
                        for k in batch if k.startswith("action/")}
            diag_old_lp = old_log_probs[:diag_n].to(self.device)
            # Raw (pre-normalisation) advantages for the diag slice — used to
            # annotate worst/best samples. Note: adv = batch["advantages"] has
            # already been normalised above (mean 0, std 1); use the raw returns
            # indirectly via batch["advantages"] which IS the normalised version.
            diag_adv = batch["advantages"][:diag_n]
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
                lp_shoot_d  = _blpv(d_d.shoot_logit, "shoot")
                lp_pass_d   = _blpv(d_d.pass_logit, "pass_")
                lp_move_d   = _blpv(d_d.move_logit, "move")
                lp_tackle_d = _blpv(d_d.tackle_logit, "tackle")
                lp_gp_d     = _blpv(d_d.get_possession_raw, "get_possession_extra")
                lp_mark_d   = _blpv(d_d.mark_logit, "mark")
                lp_hold_d   = _blpv(d_d.hold_position_logit, "hold_position")
                lp_sprint_d = _blpv(e_d.sprint_logit, "sprint")
                lp_kick_d   = _blpv(e_d.kick_logit, "kick")
                lp_ta_d     = _blpv(e_d.tackle_attempt_logit, "tackle_attempt")
                # Zero out decision heads frozen for this curriculum phase so this
                # diagnostic matches the real (masked) ratio used for the actual
                # policy loss/KL/early-stop — previously always summed all 12 heads
                # unconditionally, which wrongly blamed frozen heads (e.g. gp_extra,
                # hold, shoot) for driving KL/ratio outliers they never contribute to.
                _masked_diag = self._ppo_lp_masked_heads
                if "shoot_logit" in _masked_diag:
                    lp_shoot_d = torch.zeros_like(lp_shoot_d)
                if "pass_logit" in _masked_diag:
                    lp_pass_d = torch.zeros_like(lp_pass_d)
                if "move_logit" in _masked_diag:
                    lp_move_d = torch.zeros_like(lp_move_d)
                if "tackle_logit" in _masked_diag:
                    lp_tackle_d = torch.zeros_like(lp_tackle_d)
                if "get_possession_raw" in _masked_diag:
                    lp_gp_d = torch.zeros_like(lp_gp_d)
                if "mark_logit" in _masked_diag:
                    lp_mark_d = torch.zeros_like(lp_mark_d)
                if "hold_position_logit" in _masked_diag:
                    lp_hold_d = torch.zeros_like(lp_hold_d)
                _lsm = self.execution_net.move_dir_log_kappa.to(self.device)
                _lsk = self.execution_net.kick_dir_log_kappa.to(self.device)
                _lskz = self.execution_net.kick_dir_z_log_std.to(self.device)
                _lspow = self.execution_net.kick_power_log_std.to(self.device)
                _lsspin = self.execution_net.kick_spin_log_std.to(self.device)
                lp_mvdir_d  = self._move_dir_head(e_d.move_direction, _lsm).log_prob(diag_act["move_dir_raw"])
                lp_kkdir_d  = self._kick_dir_head(e_d.kick_direction, _lsk, _lskz).log_prob(diag_act["kick_dir_raw"])
                lp_kkpow_d  = self._kick_power_head(e_d.kick_power, _lspow).log_prob(diag_act["kick_power_raw"])
                # kick_spin is permanently frozen (see agent_plans/spin_implementation_plan.md
                # section 0) -- masked to zero here too, matching the real training
                # log_prob functions, so this diagnostic doesn't show spurious
                # trunk-drift KL/ratio contributions from a head that never trains.
                lp_kkspin_d = (
                    torch.zeros(diag_act["kick_spin_raw"].shape[0], device=self.device)
                    if self._kick_spin_frozen else
                    self._kick_spin_dist(e_d.kick_spin, _lsspin).log_prob(diag_act["kick_spin_raw"]).sum(dim=-1)
                )
                # Gate sub-parameter heads by their parent action, matching
                # _compute_log_prob/_recompute_log_prob — otherwise kick_dir/sprint/
                # move_dir noise from never-taken actions (e.g. kick=0 the entire
                # rollout) pollutes diag_new_lp/diag_ratio and the printed means.
                _exec_move_mask_d = (diag_act["exec_move"].squeeze(-1) > 0.5).float()
                _kick_mask_d = (diag_act["kick"].squeeze(-1) > 0.5).float()
                lp_sprint_d = lp_sprint_d * _exec_move_mask_d
                lp_mvdir_d  = lp_mvdir_d * _exec_move_mask_d
                lp_kkdir_d  = lp_kkdir_d * _kick_mask_d
                lp_kkpow_d  = lp_kkpow_d * _kick_mask_d
                lp_kkspin_d = lp_kkspin_d * _kick_mask_d
                # NOTE: direction heads (move_dir/kick_dir) are NOT scaled by any
                # entropy-bonus weight in the REAL training log_prob (see
                # _compute_log_prob/_recompute_log_prob above — the entropy-only
                # weights, e.g. ent_move_dir_weight, only scale _compute_entropy's
                # bonus, never the importance ratio's log_prob).
                # lp_mvdir_w/lp_kkdir_w keep their names for the rest of this
                # block's plumbing but are now equal to lp_mvdir_d/lp_kkdir_d
                # unweighted — this mirrors the real training log_prob exactly,
                # same rationale as before (diag_new_lp/diag_ratio/worst_i and
                # the per-head delta table must match what actually drives
                # this rollout's KL/early-stop).
                lp_mvdir_w  = lp_mvdir_d
                lp_kkdir_w  = lp_kkdir_d
                diag_new_lp = (lp_shoot_d + lp_pass_d + lp_move_d + lp_tackle_d + lp_gp_d +
                               lp_mark_d + lp_hold_d + lp_sprint_d + lp_kick_d + lp_ta_d +
                               lp_mvdir_w + lp_kkdir_w + lp_kkpow_d + lp_kkspin_d)
                diag_ratio  = torch.exp(diag_new_lp - diag_old_lp)
                worst_i     = int(diag_ratio.argmax())
                stored_mv   = diag_act["move_dir_raw"][worst_i]
                new_mv_mean = e_d.move_direction[worst_i]
                s_angle     = math.degrees(math.atan2(float(stored_mv[1]),   float(stored_mv[0])))
                n_angle     = math.degrees(math.atan2(float(new_mv_mean[1]), float(new_mv_mean[0])))

                # Full-minibatch angular_diff distribution (not just the single
                # worst sample above) -- distinguishes "one freak outlier sample"
                # (expected even for a tiny mean shift, if move_dir_log_kappa is
                # high (narrow) enough) from "the whole batch's move_dir target
                # genuinely shifted a lot" (a different, more concerning
                # failure mode). Same diag_n-row slice as everything else here.
                _stored_angles_all = torch.atan2(diag_act["move_dir_raw"][:, 1], diag_act["move_dir_raw"][:, 0])
                _new_angles_all = torch.atan2(e_d.move_direction[:, 1], e_d.move_direction[:, 0])
                _ang_diff_all = (_stored_angles_all - _new_angles_all).abs() * (180.0 / math.pi)
                _ang_diff_all = torch.minimum(_ang_diff_all, 360.0 - _ang_diff_all)

                # Per-sample, per-head new-lp stack for KL attribution (which
                # head(s) drive each sample's ratio, not just the aggregate mean).
                # Uses the SAME (unweighted) direction terms as diag_new_lp
                # above so the per-head deltas sum to (approximately) the same
                # total used to compute diag_ratio/worst_i.
                _per_head_new_lp = torch.stack([
                    lp_shoot_d, lp_pass_d, lp_move_d, lp_tackle_d, lp_gp_d, lp_mark_d, lp_hold_d,
                    lp_sprint_d, lp_kick_d, lp_ta_d,
                    lp_mvdir_w, lp_kkdir_w, lp_kkpow_d, lp_kkspin_d,
                ], dim=-1)  # (diag_n, 14)
                _per_head_names = ["shoot", "pass_", "move", "tackle", "gp_extra", "mark", "hold",
                                    "sprint", "kick", "tackle_attempt", "move_dir", "kick_dir",
                                    "kick_power", "kick_spin"]
            # Per-head old vs new log_probs: read stored head_log_probs from buffer
            # and compare to what the current policy assigns.  The diff shows which
            # head is driving the KL.
            _new_lp_heads = [
                lp_shoot_d.mean(), lp_pass_d.mean(), lp_move_d.mean(), lp_tackle_d.mean(),
                lp_gp_d.mean(), lp_mark_d.mean(), lp_hold_d.mean(),
                lp_sprint_d.mean(), lp_kick_d.mean(), lp_ta_d.mean(),
                lp_mvdir_d.mean(), lp_kkdir_d.mean(), lp_kkpow_d.mean(), lp_kkspin_d.mean(),
            ]
            _new_lp_heads_map = dict(zip(
                ["shoot","pass_","move","tackle","gp_extra","mark","hold",
                 "sprint","kick","tackle_attempt","move_dir","kick_dir",
                 "kick_power","kick_spin"],
                [float(v) for v in _new_lp_heads]
            ))
            from footballcoach.ai.ppo.rollout_buffer import HEAD_LP_KEYS as _HLK
            _head_lp_delta_str = ""
            _old_per_head = None
            if "head_log_probs" in batch:
                old_hlp = batch["head_log_probs"][:diag_n].mean(dim=0)  # (13,)
                for _ki, _k in enumerate(_HLK):
                    _new_v = _new_lp_heads_map.get(_k, 0.0)
                    _old_v = float(old_hlp[_ki])
                    _delta = _new_v - _old_v
                    if abs(_delta) > 0.05:
                        _head_lp_delta_str += f" {_k}:{_delta:+.2f}"
                # Per-sample old per-head lp, aligned to _per_head_names order
                # (HEAD_LP_KEYS has an extra "exec_move" column that our 12-head
                # stack above doesn't include - drop it here to keep indices in sync).
                _old_full = batch["head_log_probs"][:diag_n].to(self.device)  # (diag_n, 13)
                _exec_move_col = _HLK.index("exec_move")
                _keep_cols = [i for i in range(_old_full.shape[-1]) if i != _exec_move_col]
                _old_per_head = _old_full[:, _keep_cols]  # (diag_n, 12)

            # Top-2 highest-ratio samples, each with full diagnostic fields.
            _topk = min(2, diag_n)
            _worst_idxs = diag_ratio.topk(_topk).indices.tolist()
            _rcomp_list = batch.get("reward_comps_raw", [])
            _outcome_list = batch.get("step_outcomes", [])
            _worst_lines = []
            for _wi in _worst_idxs:
                _old_lp_wi = float(diag_old_lp[_wi])
                _new_lp_wi = float(diag_new_lp[_wi])
                _adv_wi = float(diag_adv[_wi]) if _wi < len(diag_adv) else float("nan")
                _rew_wi = float(batch["rewards"][_wi]) if _wi < n else float("nan")
                _ret_wi = float(batch["returns"][_wi]) if _wi < n else float("nan")
                _val_wi = float(batch["values"][_wi]) if _wi < n else float("nan")
                _rc = _rcomp_list[_wi] if _wi < len(_rcomp_list) else {}
                _rcomp_str = (
                    "  ".join(f"{_k}={_v:+.3f}" for _k, _v in _rc.items() if abs(_v) > 0.001)
                    or "n/a"
                )
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
                # Saturation check: for each active (non-masked) Bernoulli head, the
                # old sampled probability — near-0/near-1 values make the log-ratio
                # highly nonlinear, so a tiny logit shift can produce a huge ratio
                # for this single sample without any real broad policy drift.
                _sat_bern_heads = [
                    ("exec_move", e_d.exec_move_logit), ("sprint", e_d.sprint_logit),
                    ("kick", e_d.kick_logit), ("tackle_attempt", e_d.tackle_attempt_logit),
                ]
                # NOTE: d_d/e_d were computed under the CURRENT policy snapshot,
                # so these are the "new" probabilities, not the ones at sampling
                # time — still useful as a proxy since ratio-outlier samples are
                # by definition ones where old/new probability sit on opposite
                # sides of a near-0/near-1 saturation region.
                _sat_str = "  ".join(
                    f"{_hn}_p_new={float(torch.sigmoid(_hl[_wi])):.4f}"
                    for _hn, _hl in _sat_bern_heads
                )
                _worst_lines.append(
                    f"    idx={_wi:4d}  ratio={float(diag_ratio[_wi]):8.3f}  adv={_adv_wi:+.3f}"
                    f"  lp: old={_old_lp_wi:.3f}  new={_new_lp_wi:.3f}\n"
                    f"      rew={_rew_wi:+.4f}  ret={_ret_wi:+.4f}  val={_val_wi:+.4f}  outcome={_outcome_wi}\n"
                    f"      rew_breakdown: {_rcomp_str}\n"
                    f"      head_deltas: {_contrib_str}\n"
                    f"      saturation: {_sat_str}"
                )

            _worst_delta_row = _per_head_new_lp[worst_i]
            if _old_per_head is not None:
                _worst_delta_row = _worst_delta_row - _old_per_head[worst_i]
            _worst_delta_str = "  ".join(
                f"{_n}:{float(_v):+.3f}" for _n, _v in zip(_per_head_names, _worst_delta_row)
                if abs(float(_v)) > 0.02
            )

            # Best sample: highest new log_prob (most "on-distribution" for
            # current policy). Shows what the policy is most confident about
            # and which heads drive that high probability.
            best_i = int(diag_new_lp.argmax())
            _best_mv  = diag_act["move_dir_raw"][best_i]
            _best_mv_mean = e_d.move_direction[best_i]
            _best_s_angle = math.degrees(math.atan2(float(_best_mv[1]),        float(_best_mv[0])))
            _best_n_angle = math.degrees(math.atan2(float(_best_mv_mean[1]),   float(_best_mv_mean[0])))
            _best_per_head = _per_head_new_lp[best_i]
            if _old_per_head is not None:
                _best_per_head_delta = _best_per_head - _old_per_head[best_i]
            else:
                _best_per_head_delta = _best_per_head
            _best_contribs_sorted = sorted(
                ((_per_head_names[_hi], float(_best_per_head[_hi])) for _hi in range(len(_per_head_names))),
                key=lambda kv: kv[1], reverse=True,
            )
            _best_contrib_str = "  ".join(
                f"{_n}:{_v:.3f}" for _n, _v in _best_contribs_sorted if abs(_v) > 0.02
            )
            _best_adv = float(diag_adv[best_i]) if best_i < len(diag_adv) else float("nan")

            log.info(
                f"  [per-head new lp means, n={diag_n}]\n"
                f"    shoot={lp_shoot_d.mean():.3f}  pass={lp_pass_d.mean():.3f}"
                f"  move={lp_move_d.mean():.3f}  tackle={lp_tackle_d.mean():.3f}"
                f"  gp={lp_gp_d.mean():.3f}  mark={lp_mark_d.mean():.3f}  hold={lp_hold_d.mean():.3f}\n"
                f"    sprint={lp_sprint_d.mean():.3f}  kick={lp_kick_d.mean():.3f}"
                f"  t_att={lp_ta_d.mean():.3f}\n"
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

        # --- New diagnostics block (advantage / ratio / per-head grad-norm /
        # continuous+discrete head drift / per-head KL) — same style as the
        # [grad clip] log line below, printed once per rollout. ---
        if all_adv_mean:
            log.info(
                f"  [advantage] mean={float(np.mean(all_adv_mean)):.3f}"
                f"  std={float(np.mean(all_adv_std)):.3f}"
                f"  min={float(np.min(all_adv_min)):.3f}"
                f"  max={float(np.max(all_adv_max)):.3f}"
            )
        # --- Per-episode |advantage| percentiles: unlike the [advantage] line
        # above (normalised advantage, pooled across every row from every
        # minibatch/epoch this rollout), this uses the RAW (pre-normalisation)
        # per-step advantages in batch["advantages"] -- untouched by the
        # `adv = (adv - adv.mean()) / (adv.std() + 1e-8)` reassignment above,
        # which rebinds the local name `adv` to a new tensor rather than
        # mutating batch["advantages"] in place -- segmented into episodes via
        # batch["dones"] (same convention as the value-pretrain 85/15 episode
        # split above: an episode is [prev_done_idx+1, this_done_idx]; any
        # trailing partial episode after the last done is dropped). For each
        # complete episode we take mean(|advantage|) across its rows, then
        # report percentiles of THAT per-episode number across all episodes in
        # the rollout -- i.e. "how big are advantage swings, episode by
        # episode", distinct from the pooled per-row view above which can't
        # tell a rollout with uniformly-moderate advantages apart from one
        # with a few huge-swing episodes among many quiet ones.
        if "dones" in batch:
            _raw_adv_np = batch["advantages"].detach().cpu().numpy()
            _dones_np = batch["dones"].detach().cpu().numpy()
            _ep_end_idxs = np.where(_dones_np > 0.5)[0]
            if len(_ep_end_idxs) > 0:
                _ep_starts = np.concatenate([[0], _ep_end_idxs[:-1] + 1])
                _ep_abs_adv_means = np.array([
                    np.abs(_raw_adv_np[s:e + 1]).mean()
                    for s, e in zip(_ep_starts, _ep_end_idxs)
                ])
                _ep_abs_adv_t = torch.from_numpy(_ep_abs_adv_means)
                _pcts = [0, 1, 10, 50, 90, 99, 100]
                _pct_str = "  ".join(
                    f"p{p}={float(_ep_abs_adv_t.quantile(p / 100.0)):.3f}" for p in _pcts
                )
                log.info(
                    f"  [advantage |.| by episode, n_episodes={len(_ep_abs_adv_means)}]  {_pct_str}"
                )
        if all_ratios:
            _ratios_all = torch.cat(all_ratios)
            _ratio_clip_frac = float(np.mean(all_ratio_clipped_frac)) if all_ratio_clipped_frac else 0.0
            log.info(
                f"  [ratio] mean={_ratios_all.mean():.4f}"
                f"  std={_ratios_all.std():.4f}"
                f"  min={_ratios_all.min():.4f}"
                f"  max={_ratios_all.max():.4f}"
                f"  clipped={_ratio_clip_frac * 100:.1f}%"
            )
        if _ratio_spike["ratio"] > float("-inf"):
            _spike_row = _ratio_spike["row"]
            _new_head_row = _ratio_spike["per_head_new"].tolist()
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
                f"  [ratio spike] max={_ratio_spike['ratio']:.3f}"
                f"  epoch={_ratio_spike['epoch'] + 1}/{self.n_epochs}  mb={_ratio_spike['mb']}"
                f"  row={_spike_row}(global)  adv={_ratio_spike['adv']:+.3f}"
                f"  old_lp={_ratio_spike['old_lp']:.3f}  new_lp={_ratio_spike['new_lp']:.3f}\n"
                f"    rew_breakdown: {_rcomp_str}  outcome={_spike_outcome_str}\n"
                f"    per-head Δ(new-old), sorted by |Δ|: {_spike_delta_str}"
            )
        _head_grad_norm_str = "  ".join(
            f"{name}={np.mean(vals):.3f}" for name, vals in all_head_grad_norm.items() if vals
        )
        if _head_grad_norm_str:
            log.info(f"  [exec head grad norm] {_head_grad_norm_str}")
        _move_ls_end = float(self.execution_net.move_dir_log_kappa.mean().item())
        _kick_ls_end = float(self.execution_net.kick_dir_log_kappa.mean().item())
        _kickz_ls_end = float(self.execution_net.kick_dir_z_log_std.mean().item())
        log.info(
            f"  [exec continuous log_kappa] move_direction: start={_move_ls_start:.4f} end={_move_ls_end:.4f}"
            f"   kick_direction: start={_kick_ls_start:.4f} end={_kick_ls_end:.4f}"
            f"   kick_direction_z (log_std): start={_kickz_ls_start:.4f} end={_kickz_ls_end:.4f}"
        )
        # Approximate, mean-angle/mean-z-held-fixed analytic KL split of kick_dir's azimuth vs
        # elevation components, from this rollout's start->end kappa/sigma_z snapshot alone (same
        # _von_mises_kl / Normal-kl_divergence machinery _ppg_kl_penalty uses for its own kick_dir
        # split, reused here as a cheap diagnostic -- NOT the same quantity as the ratio-based,
        # per-minibatch, opportunity-weighted "[per-head KL] ... kick_dir=" figure above: this one
        # isolates ONLY the concentration/spread change (ignoring real mean-angle/mean-z drift,
        # which the "[exec continuous Δ per opt step]" dmean/Δσ° figures below already cover), so
        # it is a cleaner "how much did kick_dir's SHARPNESS alone move this rollout" signal, not a
        # substitute for the real training-relevant combined KL.
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
        # Build per-step and per-epoch Δ strings, with angular interpretations.
        # dmean is the per-step L2 shift of the unit-vector mean; for unit vectors
        # |u-v|=d → angle θ = arccos(1 - d²/2). dlog_kappa is the mean absolute
        # per-step log_kappa change; angular effect uses the large-kappa von
        # Mises approximation (variance ~= 1/kappa), same as _kappa_deg() above.
        def _angular_dmean_deg(dmean: float) -> float:
            """L2 shift of unit-vector mean → approx angular shift in degrees."""
            cos_theta = max(-1.0, min(1.0, 1.0 - dmean ** 2 / 2.0))
            return math.degrees(math.acos(cos_theta))

        def _angular_dlog_std_deg(dlog_kappa: float, ls_end: float) -> float:
            """Change in log_kappa -> change in angular std, in degrees."""
            return abs(
                math.degrees(math.sqrt(1.0 / math.exp(ls_end)))
                - math.degrees(math.sqrt(1.0 / math.exp(ls_end - dlog_kappa)))
            )

        _ls_end_by_name = {"move_direction": _move_ls_end, "kick_direction": _kick_ls_end}
        _cont_shift_parts = []
        for name, vals in all_continuous_mean_shift.items():
            if not vals:
                continue
            n_steps = len(vals)
            mean_dmean = float(np.mean(vals))
            mean_dlog_std = float(np.mean(all_continuous_log_std_shift[name]))
            epoch_dmean = float(np.sum(vals))   # cumulative shift over whole rollout
            epoch_dlog_std = float(np.sum(all_continuous_log_std_shift[name]))
            ls_end = _ls_end_by_name.get(name, 0.0)
            mean_deg = _angular_dmean_deg(mean_dmean)
            epoch_deg = _angular_dmean_deg(epoch_dmean / max(n_steps, 1)) * n_steps  # rough epoch total
            mean_dstd_deg = _angular_dlog_std_deg(mean_dlog_std, ls_end)
            _cont_shift_parts.append(
                f"{name}("
                f"dmean={mean_dmean:.4f}≈{mean_deg:.2f}°/step  epoch≈{epoch_deg:.1f}°  "
                f"dlog_kappa={mean_dlog_std:.5f}  Δ(ang std)°={mean_dstd_deg:.3f}/step)"
            )
        # kick_dir_z_log_std is a plain Gaussian log_std (not a von Mises
        # log_kappa), so it uses the direct sigma->degrees approximation
        # (small-z regime: z itself approximates an elevation angle in
        # radians) rather than _angular_dlog_std_deg's inverse-sqrt-kappa
        # formula above. No corresponding "mean shift" entry -- kick_dir's
        # combined 3D mean-shift (kickdir_mean_shift above) already reflects
        # z's contribution as part of the whole vector's movement.
        if all_kickz_log_std_shift:
            mean_dlog_stdz = float(np.mean(all_kickz_log_std_shift))
            mean_dstdz_deg = abs(
                math.degrees(math.exp(_kickz_ls_end))
                - math.degrees(math.exp(_kickz_ls_end - mean_dlog_stdz))
            )
            _cont_shift_parts.append(
                f"kick_direction_z(dlog_std={mean_dlog_stdz:.5f}  Δσ°≈{mean_dstdz_deg:.3f}/step)"
            )
        if _cont_shift_parts:
            log.info(f"  [exec continuous \u0394 per opt step] {'  '.join(_cont_shift_parts)}")
        _disc_shift_str = "  ".join(
            f"{name}={np.mean(vals):.4f}" for name, vals in all_discrete_logit_shift.items() if vals
        )
        if _disc_shift_str:
            log.info(f"  [exec discrete \u0394logit per opt step] {_disc_shift_str}")
        # Masked/frozen/inactive heads (see _inactive_head_lp_keys) are
        # dropped from both lines below -- ratio=1 forces KL to an exact,
        # content-free 0.0000 and the counterfactual policy_loss to a
        # meaningless -mean(this update's advantages) residual for these
        # heads, never a real training signal, so listing them just buries
        # the heads that actually moved.
        _inactive_heads = self._inactive_head_lp_keys()
        if all_head_kl:
            _mean_head_kl = torch.stack(all_head_kl).mean(dim=0)
            _head_kl_str = "  ".join(
                f"{k}={v:+.4f}" for k, v in zip(HEAD_LP_KEYS, _mean_head_kl.tolist())
                if k not in _inactive_heads
            )
            log.info(f"  [per-head KL] {_head_kl_str}")
        if all_head_policy_loss:
            # See all_head_policy_loss's own comment (declared near the top
            # of this function) -- a per-head COUNTERFACTUAL (this head's
            # ratio alone, others held at 1), not a strict decomposition:
            # these 15 numbers do NOT sum to the scalar policy_loss= above.
            # Sorted by |value| (biggest driver first) rather than
            # HEAD_LP_KEYS order, since the whole point is spotting which
            # head(s) are pulling the objective away from 0.
            _mean_head_pol = torch.stack(all_head_policy_loss).mean(dim=0)
            _head_pol_pairs = sorted(
                (kv for kv in zip(HEAD_LP_KEYS, _mean_head_pol.tolist()) if kv[0] not in _inactive_heads),
                key=lambda kv: -abs(kv[1]),
            )
            _head_pol_str = "  ".join(f"{k}={v:+.4f}" for k, v in _head_pol_pairs)
            log.info(
                f"  [per-head policy_loss (counterfactual, ratio-in-isolation, "
                f"sorted by |.|)] {_head_pol_str}"
            )

        # Per-head mean activation rates from the buffer (0–100%). Zero-cost: just
        # averages the stored 0/1 action arrays — no extra forward pass needed.
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
            "ta_p": float(np.mean(all_tackle_prob)) if all_tackle_prob else float('nan'),
            "kk_p": float(np.mean(all_kick_prob)) if all_kick_prob else float('nan'),
        }

        # --- Grad-norm clipping summary (human-readable) ---
        # Reports, for each isolated param group, how often the pre-clip gradient
        # norm actually exceeded its limit (i.e. clipping fired) this rollout, and
        # the mean/max pre-clip norm seen. If direction_max_grad_norm/
        # direction_learning_rate are unset (None) in ai_config.json, the direction
        # group still exists but shares max_grad_norm with the main group — in that
        # case the two "limit=" values below will be identical, and the split just
        # tells you what fraction of clipping activity is attributable to the
        # direction heads specifically vs everything else.
        n_main = len(all_grad_norm_main)
        n_dir = len(all_grad_norm_dir)
        grad_clip_pct_main = (100.0 * clip_triggered_main / n_main) if n_main else 0.0
        grad_clip_pct_dir = (100.0 * clip_triggered_dir / n_dir) if n_dir else 0.0
        grad_clip_mean_main = float(np.mean(all_grad_norm_main)) if n_main else 0.0
        grad_clip_max_main = float(np.max(all_grad_norm_main)) if n_main else 0.0
        grad_clip_mean_dir = float(np.mean(all_grad_norm_dir)) if n_dir else 0.0
        grad_clip_max_dir = float(np.max(all_grad_norm_dir)) if n_dir else 0.0
        log.info(
            f"  [grad clip] main: {clip_triggered_main}/{n_main} steps clipped ({grad_clip_pct_main:.0f}%)"
            f"  pre-clip norm mean={grad_clip_mean_main:.3f} max={grad_clip_max_main:.3f}  limit={self.max_grad_norm}"
            + (
                f"\n              direction: {clip_triggered_dir}/{n_dir} steps clipped ({grad_clip_pct_dir:.0f}%)"
                f"  pre-clip norm mean={grad_clip_mean_dir:.3f} max={grad_clip_max_dir:.3f}"
                f"  limit={self.direction_max_grad_norm}"
                if n_dir else ""
            )
        )
        if all_param_delta_dir:
            # Actual applied movement vs. the clipped GRADIENT norm above --
            # Adam's real step is m_hat/sqrt(v_hat) (a ratio), not a direct
            # function of the clipped gradient, so these can diverge a lot
            # early after --reset-optimizer (v not yet warmed up). Compare
            # this line's mean/max directly against the "direction:" clip
            # line's pre-clip norm/limit just above.
            log.info(
                f"  [direction param delta] actual ||params_after - params_before|| per opt step:"
                f"  mean={float(np.mean(all_param_delta_dir)):.4f}"
                f"  max={float(np.max(all_param_delta_dir)):.4f}"
                f"  (n={len(all_param_delta_dir)})"
            )

        if _outc_n_sum:
            _value_by_outcome = {
                name: (
                    _outc_sq_err_sum[name] / max(_outc_n_sum[name], 1),
                    _outc_n_sum[name],
                    _outc_gt_sum[name] / max(_outc_n_sum[name], 1),
                    _outc_gt_sqsum[name] / max(_outc_n_sum[name], 1),
                )
                for name in _outc_sq_err_sum
            }
            log.info(f"  [value RMSE by outcome] {format_outcome_rmse_breakdown(_value_by_outcome)}")

        return {
            "policy_loss": float(np.mean(all_policy_loss)),
            "value_loss": float(np.mean(all_value_loss)),
            "pre_update_value_loss": pre_update_value_loss,
            "flip_consistency": flip_consistency,
            "entropy": float(np.mean(all_entropy)),
            "entropy_breakdown": {k: float(np.mean(v)) for k, v in all_entropy_breakdown.items()},
            "ent_coef": ent_coef,
            "approx_kl": float(np.mean(all_kl)),
            "bc_loss": float(np.mean(all_bc_loss)),
            "bc_tackle_loss": float(np.mean(all_bc_tackle_loss)) if all_bc_tackle_loss else 0.0,
            "bc_coeff": bc_coeff,
            "bc_kick_coeff": bc_kick_coeff,
            "bc_tackle_coeff": bc_tackle_coeff,
            "epoch_time_ms": float(np.mean(epoch_times)) if epoch_times else 0.0,
            "move_log_std": move_log_std,
            "kick_log_std": kick_log_std,
            "kick_z_log_std": kick_z_log_std,
            "kick_power_log_std": kick_power_log_std,
            "mv_ls_grad": mean_mv_ls_grad,
            "head_act": head_act,
            "kick_power_frac": _kick_power_frac_stats(batch, self.kick_power_floor),
            "grad_clip_pct_main": grad_clip_pct_main,
            "grad_clip_pct_dir": grad_clip_pct_dir,
            "grad_clip_mean_norm_main": grad_clip_mean_main,
            "grad_clip_mean_norm_dir": grad_clip_mean_dir,
            "values_mean": float(batch["values"].mean()),
            "values_std": float(batch["values"].std()),
            "returns_mean": float(returns.mean()),
            "returns_std": float(returns.std()),
            "adv_mean": float(batch["advantages"].mean()),
            "adv_std": float(batch["advantages"].std()),
            "adv_abs_mean": float(batch["advantages"].abs().mean()),
            "mean_sq_td": float(((returns - batch["values"]) ** 2).mean()),
        }
















def _decode_rollout_result(
    r: dict, gamma: float, lam: float, want_replay_inputs: bool,
) -> tuple[dict, Optional[list], list, Optional[list]]:
    """Decode ONE per-env rollout result from a batched worker into
    ``(tensors, advantages, track_ids, dones)`` for ``_train_batched_parallel``.

    - Worker-finalized result (``"batch"`` key -- what the main loop asks for
      via ``collect(returns=...)``): GAE/returns were already computed in the
      worker with THIS process's gamma/lam; the numpy wire batch is turned
      back into torch tensors (zero-copy).
    - Legacy result (``"buffer"``): GAE is computed here with
      ``r["last_value"]``, exactly the pre-worker-finalization behaviour.

    Both paths must produce identical numbers -- that equivalence is the
    correctness claim of worker-side finalization (see the tests).
    ``advantages``/``dones`` (Python lists) are only built when
    ``want_replay_inputs`` (episode-seed replay needs them for
    ``_episode_abs_adv_means``); otherwise they're None so the main loop
    doesn't pay for a conversion nobody reads. ``track_ids`` is always
    returned (cheap: already a list).
    """
    if "batch" in r:
        from footballcoach.ai.ppo.batched_rollout_worker import batch_from_wire

        tensors = batch_from_wire(r["batch"])
        advantages = tensors["advantages"].tolist() if want_replay_inputs else None
        dones = tensors["dones"].tolist() if want_replay_inputs else None
        return tensors, advantages, tensors["track_ids"], dones
    buf = r["buffer"]
    advantages, returns = buf.compute_gae(gamma, lam, r["last_value"])
    return buf.as_tensors(advantages, returns), advantages, buf.track_ids, buf.dones









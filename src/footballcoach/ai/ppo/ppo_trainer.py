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

    env = ScenarioEnv(SCENARIOS[0], trainee_player_id="kicker", phase=1)
    trainer = PPOTrainer.from_config()
    trainer.train(env, total_steps=500_000)
"""
from __future__ import annotations

import collections
import copy
import dataclasses
import logging
import math
import random
import shutil
import time
from dataclasses import dataclass, field
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

from footballcoach.ai.action.distributions import (
    IndependentBernoulli,
    KickDirectionHead,
    MaskedCategorical,
    SquashedNormalHead,
    VonMisesDirectionHead,
    _von_mises_kl,
)
from footballcoach.ai.action.gating import select_action
from footballcoach.ai.action.schema import DecisionAction, DecisionHeadsRaw, ExecutionAction
from footballcoach.ai.config import load_ai_config
from footballcoach.ai.eval.seeded_eval import (
    _capped_thread_env_for_pool_spawn,
    default_eval_seeds,
    run_seeded_evaluation,
    run_seeded_evaluation_batched,
    run_seeded_evaluation_parallel,
    run_seeded_evaluation_parallel_batched,
)
from footballcoach.ai.models.decision_network import DecisionNetwork, derive_get_possession_prob
from footballcoach.ai.models.execution_network import ExecutionNetwork, flatten_decision_heads
from footballcoach.ai.obs.augment import N_FLIP_VARIANTS, augment_batch, augment_obs_bc
from footballcoach.ai.progress import ProgressReporter
from footballcoach.ai.obs.canonical import (
    CanonicalNetworkWrapper,
    canonicalize_bc_labels,
    canonicalize_obs,
    mirror_x,
    x_sign_of,
)
from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer, HEAD_LP_KEYS
from footballcoach.ai.ppo.schedules import TrainingSchedules

log = logging.getLogger("footballcoach.ai.ppo")

# DecisionHeadsRaw's field set is fixed at class-definition time -- caching this
# once avoids a dataclasses.fields() reflection call every PPO/pretrain minibatch
# (see the detach-for-separate-value-net call sites below).
_DECISION_HEADS_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(DecisionHeadsRaw))


def _load_state_dict_tolerant(module: torch.nn.Module, ckpt_sd: dict, label: str) -> None:
    """Load ckpt_sd into module, skipping any param whose shape no longer
    matches (e.g. value_head grew an extra input column since the
    checkpoint was saved) instead of raising. Skipped params keep their
    fresh random init; everything else loads normally. Lets checkpoints
    saved under an older network architecture keep loading after additive
    changes to value_head/value_ai_type_channel sizing, at the cost of
    those specific params needing to be retrained (typically fine, since
    such changes are usually paired with --reset-value-weights anyway)."""
    own_sd = module.state_dict()
    compatible = {}
    skipped = []
    for k, v in ckpt_sd.items():
        if k in own_sd and own_sd[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)
    module.load_state_dict(compatible, strict=False)
    if skipped:
        log.warning(
            f"{label}: skipped {len(skipped)} shape-mismatched param(s) "
            f"(network architecture changed since checkpoint was saved), "
            f"keeping fresh init for: {skipped}"
        )


def _migrate_direction_log_std_to_kappa(state_dict: dict) -> dict:
    """Backward-compat shim for the move_dir_log_std/kick_dir_log_std ->
    move_dir_log_kappa/kick_dir_log_kappa/kick_dir_z_log_std rename (see
    ai_trainer_knowledge.md "Direction heads: von Mises"). Unlike
    _migrate_crossing_head_state_dict's shape-only migrations, this is a
    same-shape RENAME plus a numeric transform, since log_kappa is not the
    same quantity as log_std (kappa ~= 1/sigma^2, an inverted, approximate
    relationship, not an exact one).

    Guarded against double-application: a checkpoint already saved under the
    new names is left untouched (checked BEFORE looking for old keys, same
    ordering rationale as _migrate_crossing_head_state_dict's shape-match
    guard -- a checkpoint in the current format must never be reinterpreted
    as if it were the old one). Returns a NEW dict (does not mutate
    state_dict); a no-op if neither old key is present (checkpoint predates
    these params entirely, or already uses the new names).
    """
    old_move_key, new_move_key = "move_dir_log_std", "move_dir_log_kappa"
    old_kick_key, new_kick_kappa_key, new_kick_z_key = (
        "kick_dir_log_std", "kick_dir_log_kappa", "kick_dir_z_log_std",
    )
    if new_move_key in state_dict or new_kick_kappa_key in state_dict:
        return state_dict
    if old_move_key not in state_dict and old_kick_key not in state_dict:
        return state_dict
    state_dict = dict(state_dict)
    if old_move_key in state_dict:
        old_log_std = state_dict.pop(old_move_key)
        sigma = torch.exp(old_log_std)
        new_log_kappa = torch.log(1.0 / (sigma * sigma))
        state_dict[new_move_key] = new_log_kappa
        log.info(
            f"Migrated checkpoint's move_dir_log_std={float(old_log_std):.4f} "
            f"(sigma={float(sigma):.4f}) to move_dir_log_kappa={float(new_log_kappa):.4f} "
            f"via kappa=1/sigma^2 (small-angle approximation)."
        )
    if old_kick_key in state_dict:
        old_log_std = state_dict.pop(old_kick_key)
        sigma = torch.exp(old_log_std)
        new_log_kappa = torch.log(1.0 / (sigma * sigma))
        state_dict[new_kick_kappa_key] = new_log_kappa
        # No analog for the new elevation-only z component in the old
        # (single, isotropic-3D) parameterization -- seed it from the SAME
        # old value directly (same numeric scale, an approximate starting
        # guess, not an exact conversion) rather than a fresh random init,
        # so kick_dir's exploration magnitude doesn't jump discontinuously
        # on resume.
        state_dict[new_kick_z_key] = old_log_std.clone()
        log.info(
            f"Migrated checkpoint's kick_dir_log_std={float(old_log_std):.4f} "
            f"(sigma={float(sigma):.4f}) to kick_dir_log_kappa={float(new_log_kappa):.4f} "
            f"(via kappa=1/sigma^2) and seeded kick_dir_z_log_std={float(old_log_std):.4f} "
            f"from the same old value (approximate starting guess -- the old "
            f"parameterization had no separate elevation component)."
        )
    return state_dict


def _detach_decision_heads(d_heads: DecisionHeadsRaw) -> DecisionHeadsRaw:
    """Return a copy of d_heads with every field detached (see separate_value_net).

    ball_physics_full/self_physics_full/other_physics_full (see
    "Frozen physics-dynamics encoders" in ai/knowledge.md) are None when
    that feature is disabled (the default) -- skip detach() for those,
    same None-passthrough every other consumer of DecisionHeadsRaw already
    has to handle."""
    return dataclasses.replace(
        d_heads, **{
            name: (v.detach() if (v := getattr(d_heads, name)) is not None else None)
            for name in _DECISION_HEADS_FIELD_NAMES
        }
    )


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


# Reward component short-key → display label mapping (order = display order).
# Used by both the per-rollout log and the pre-training diagnostic in train.py.
REWARD_COMP_LABELS: list[tuple[str, str]] = [
    ("appr",  "approach"),
    ("retr",  "retreat"),
    ("appr_sq", "approach_speed"),
    ("hdg",   "heading"),
    ("poss",  "get_possession"),
    ("prog",  "progress"),
    ("lpos",  "lose_possession"),
    ("out",   "ball_out"),
    ("ill",   "illegal"),
    ("box",   "box_possession"),
    ("spd",   "speed_bonus"),
    ("lterm", "opponent_box"),
    ("tout",  "timeout"),
    ("prox",  "proximity_bonus"),
    ("step",  "step_penalty"),
    ("stam",  "stamina_penalty"),
    ("sprint", "sprint_penalty"),
    ("tack",  "tackle_armed_penalty"),
    ("tatt",  "tackle_attempted_bonus"),
    ("karm",  "kick_armed_penalty"),
    ("katt",  "kick_attempted_bonus"),
]

# Execution-network head name -> nn.Module attribute name, used for the
# per-head gradient-norm / logit-drift diagnostics in _ppo_update().
_GRAD_NORM_SPIKE_TOP_K = 15  # how many params to show in a grad-norm-spike breakdown, see PPOTrainer.grad_norm_spike_k
_GRAD_NORM_SPIKE_HISTORY_MAXLEN = 200  # rolling window size for the adaptive spike bar's running mean/std
_GRAD_NORM_SPIKE_MIN_SAMPLES = 20  # don't check for spikes until the running history has at least this many samples

EXEC_HEAD_MODULES: list[tuple[str, str]] = [
    ("move_direction", "move_direction"),
    ("exec_move", "exec_move_logit"),
    ("sprint", "sprint_logit"),
    ("kick", "kick_logit"),
    ("kick_direction", "kick_direction"),
    ("kick_power", "kick_power"),
    ("kick_spin", "kick_spin"),
    ("tackle_attempt", "tackle_attempt_logit"),
]


def _maybe_log_grad_norm_spike(
    trainer: "PPOTrainer", label: str, epoch_i: int, mb_i: int, raw_grad_norm: float,
) -> None:
    """Adaptive grad-norm-spike diagnostic, shared by PPO/BC/DAgger's
    minibatch loops (_ppo_update, pretrain_combined()'s Phase 1, dagger.py's
    train_bc_epochs()). Maintains a bounded rolling history of raw grad
    norms on the trainer (trainer._grad_norm_history, most-recent
    _GRAD_NORM_SPIKE_HISTORY_MAXLEN samples pooled across ALL THREE training
    loops -- deliberately not reset per-epoch/per-phase, since "is this
    minibatch's grad norm unusually large FOR THIS RUN" is the question, and
    a brand new run naturally starts the window empty again anyway).

    Logs a [grad norm SPIKE] DEBUG line (visible with train.py's --verbose
    flag) with a top-K per-parameter-tensor breakdown whenever raw_grad_norm
    exceeds (running_mean +
    trainer.grad_norm_spike_k * running_std) of that history -- adaptive to
    whatever scale a given run/config actually produces, instead of a fixed
    absolute number that needs re-tuning any time the typical scale changes
    (confirmed necessary: a real run with mean~5 max~200 needs a very
    different absolute bar than a small smoke-test with mean~4.6 max~14.3).
    Requires at least _GRAD_NORM_SPIKE_MIN_SAMPLES observations before
    checking at all -- an empty/tiny history has a meaningless mean/std and
    would otherwise flag the first few minibatches of every run as "spikes"
    against almost no baseline. The just-observed raw_grad_norm is appended
    to the history AFTER computing the bar, so it never inflates its own
    baseline.

    Must be called with .grad still populated on trainer.decision_net/
    execution_net (i.e. after backward(), before optimizer.step()/
    zero_grad()) -- the per-parameter breakdown reads those .grad tensors
    directly. `label` is a short tag (e.g. "PPO", "BC", "DAgger") included
    in the log line so it's clear which training loop an outlier fired from.
    """
    if trainer.grad_norm_spike_k is None:
        return
    history = trainer._grad_norm_history
    if len(history) >= _GRAD_NORM_SPIKE_MIN_SAMPLES:
        running_mean = float(np.mean(history))
        running_std = float(np.std(history))
        spike_bar = running_mean + trainer.grad_norm_spike_k * running_std
        if raw_grad_norm > spike_bar:
            from footballcoach.ai.obs.schema import BallFeatures, GlobalFeatures, PlayerFeatures
            # "First touch" Linear layers -- the ones whose in_features IS
            # the raw named feature schema (not a hidden/intermediate
            # width) -- keyed by parameter-name SUFFIX (matched via
            # str.endswith below) so both decision_net's and execution_net's
            # copies match regardless of module-path prefix.
            _first_touch_layer_schemas = {
                "entity_encoder.per_entity_mlp.0.weight": PlayerFeatures,
                "entity_encoder.ball_query_proj.weight": BallFeatures,
                "entity_encoder.global_query_proj.weight": GlobalFeatures,
                "self_mlp.0.weight": PlayerFeatures,
                "ball_mlp.0.weight": BallFeatures,
                "global_mlp.0.weight": GlobalFeatures,
            }
            _named_params = [
                (f"{_net_name}.{_n}", _n, _p)
                for _net_name, _net in (("decision_net", trainer.decision_net), ("execution_net", trainer.execution_net))
                for _n, _p in _net.named_parameters()
                if _p.grad is not None
            ]
            _named_grad_norms = [(_full, _p.grad.norm().item()) for _full, _n, _p in _named_params]
            _named_grad_norms.sort(key=lambda kv: kv[1], reverse=True)
            _top_spike = _named_grad_norms[:_GRAD_NORM_SPIKE_TOP_K]
            _spike_str = "  ".join(f"{name}={norm:.2f}" for name, norm in _top_spike)
            # DEBUG, not WARNING -- gated behind train.py's --verbose flag
            # (sets the "footballcoach.ai" parent logger to DEBUG, which
            # this "footballcoach.ai.ppo" child inherits). Was WARNING while
            # this diagnostic was actively hunting the crossing_head/
            # crosses_logit scale bug (see physics_encoders.py's _apply_
            # logit_sigmoid) -- now that that's fixed, remaining spikes are
            # the ordinary "categorical/context flags carry a lot of
            # gradient" pattern, not something worth surfacing by default.
            log.debug(
                f"[grad norm SPIKE][{label}] epoch={epoch_i} mb={mb_i} raw={raw_grad_norm:.2f} "
                f"(running mean={running_mean:.2f} std={running_std:.2f} "
                f"bar=mean+{trainer.grad_norm_spike_k:.1f}*std={spike_bar:.2f}, n={len(history)}) -- "
                f"top {len(_top_spike)} params: {_spike_str}"
            )
            # Per-input-COLUMN gradient norm for whichever "first touch"
            # Linear layers (the ones whose in_features are the raw named
            # feature schema itself, not a hidden/intermediate width) appear
            # among this spike's parameters -- a genuine causal decomposition
            # of the SAME weight.grad tensor already read above (weight.grad
            # has shape (out_features, in_features), and column i is BY
            # CONSTRUCTION exactly the gradient signal attributable to input
            # feature i: grad_W[j,i] = sum_batch grad_output[b,j]*input[b,i]).
            # Free -- no extra backward pass, no hooks, no per-sample loop,
            # just a different reduction on a tensor we already have. This is
            # the principled alternative to the raw-feature-range heuristic
            # below, which only measures coincidental correlation with
            # magnitude and can mislead (e.g. distance_m/ball_distance_m
            # routinely reach ~2.0 for two entities near opposite pitch
            # corners -- a mundane geometric fact, unrelated to gradient
            # magnitude, that would otherwise look like a smoking gun).
            _seen_param_ids: set[int] = set()
            for _full, _n, _p in _named_params:
                if id(_p) in _seen_param_ids:
                    continue  # shared tensor (e.g. share_entity_encoder) already reported once
                for _suffix, _schema in _first_touch_layer_schemas.items():
                    if _n.endswith(_suffix):
                        _seen_param_ids.add(id(_p))
                        _cols_str = _describe_weight_grad_columns(_p, _schema)
                        if _cols_str:
                            log.debug(f"  [grad norm SPIKE][{label}] {_full} per-input-column grad norm: {_cols_str}")
                        break
    history.append(raw_grad_norm)


_GRAD_COLUMN_TOP_K = 5  # how many named input columns to show per weight tensor in a grad-norm-spike breakdown


def _describe_weight_grad_columns(
    weight: "torch.Tensor", schema_dataclass: "type | None", top_k: int = _GRAD_COLUMN_TOP_K,
) -> str:
    """Per-INPUT-COLUMN gradient norm for a Linear layer's weight tensor
    (shape (out_features, in_features)). weight.grad[:, i] is, BY
    CONSTRUCTION, exactly the gradient signal attributable to input feature
    i (grad_W[j,i] = sum_batch grad_output[b,j] * input[b,i]) -- so this is
    a genuine causal decomposition of an already-computed gradient tensor,
    unlike _describe_feature_extremes' raw-magnitude heuristic above, which
    only measures coincidental correlation with input VALUE and can mislead
    (confirmed in practice: distance_m/ball_distance_m routinely reach ~2.0
    for two entities merely near opposite pitch corners -- a mundane
    geometric fact with no causal relationship to gradient magnitude, that
    still looked like a smoking gun under the magnitude-only heuristic).

    Free to compute -- reads weight.grad.norm(dim=0), no extra backward
    pass, no hooks, no per-sample loop, just a different reduction on a
    tensor already sitting in memory after backward().

    Only meaningful for a layer's FIRST touch of named raw features -- a
    schema_dataclass is expected to have exactly as many fields as
    weight.shape[1] (in_features); if it doesn't (e.g. physics-encoder
    features concatenated on, extending in_features beyond the named
    schema -- see DecisionNetwork's self_dim_eff/ball_dim_eff), falls back
    to anonymous "col_i" labels rather than mis-naming columns.
    """
    if weight.grad is None:
        return ""
    col_norms = weight.grad.norm(dim=0)
    names = None
    if schema_dataclass is not None:
        _candidate_names = [f.name for f in dataclasses.fields(schema_dataclass)]
        if len(_candidate_names) == col_norms.shape[0]:
            names = _candidate_names
    if names is None:
        names = [f"col_{i}" for i in range(col_norms.shape[0])]
    k = min(top_k, len(names))
    top_vals, top_idx = col_norms.topk(k)
    return "  ".join(f"{names[i]}={v:.3f}" for i, v in zip(top_idx.tolist(), top_vals.tolist()))


def _measure_exec_head_grad_norms_and_maybe_log_spike(
    trainer: "PPOTrainer", label: str, epoch_i: int, mb_i: int,
) -> dict[str, float]:
    """Measurement-only (max_norm=inf, never rescales) per-execution-head
    grad norm snapshot for BC/DAgger's minibatch loops (pretrain_combined()'s
    Phase 1, dagger.py's train_bc_epochs()) -- mirrors PPO's own per-head
    grad-norm diagnostic (_ppo_update's [exec head grad norm]) so BC/DAgger
    rounds get the same per-head visibility PPO already has, instead of only
    a single aggregate clip norm. Also runs the grad-norm-spike diagnostic
    (see _maybe_log_grad_norm_spike) against the same combined
    decision_net+execution_net raw norm PPO's own spike check uses, so an
    outlier minibatch during BC/DAgger gets the same per-parameter-tensor
    breakdown PPO gets.

    Must be called AFTER loss.backward() and BEFORE the caller's own real
    clip_grad_norm_()/optimizer.step() (grads must still reflect the
    ORIGINAL, un-clipped magnitudes for this measurement to mean anything --
    this function never itself clips; the caller's real cap is unaffected).

    Returns {head_name: grad_norm}, plus "raw" for the combined norm across
    both networks. `label` is a short tag (e.g. "BC", "DAgger") passed
    through to _maybe_log_grad_norm_spike.
    """
    _gn_tensors: dict[str, torch.Tensor] = {}
    for _head_name, _attr in EXEC_HEAD_MODULES:
        _head_params = list(getattr(trainer.execution_net, _attr).parameters())
        if _head_params:
            _gn_tensors[_head_name] = torch.nn.utils.clip_grad_norm_(_head_params, float("inf"))
    _gn_tensors["raw"] = torch.nn.utils.clip_grad_norm_(
        list(trainer.decision_net.parameters()) + list(trainer.execution_net.parameters()),
        float("inf"),
    )
    _gn_names = list(_gn_tensors.keys())
    _gn_vals = dict(zip(
        _gn_names,
        torch.stack([_gn_tensors[k].to(trainer.device) for k in _gn_names]).tolist(),
    ))

    _maybe_log_grad_norm_spike(trainer, label, epoch_i, mb_i, _gn_vals["raw"])
    return _gn_vals


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

# _LP_HEAD_NAMES (decision_net attribute names, as returned by
# PPOTrainer._ppo_lp_masked_heads) -> their short HEAD_LP_KEYS-style name
# (rollout_buffer.py) -- the per-head diagnostic lines (entropy breakdown,
# [per-head KL], [per-head policy_loss (counterfactual...)], [policy heads])
# all key on the short form, so masked-head filtering needs this mapping.
_LP_HEAD_NAME_TO_SHORT: dict[str, str] = {
    "shoot_logit": "shoot", "pass_logit": "pass_", "move_logit": "move",
    "tackle_logit": "tackle", "get_possession_raw": "gp_extra",
    "mark_logit": "mark", "hold_position_logit": "hold",
}


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


def rebuild_inference_trainer(
    decision_state: dict, execution_state: dict, separate_value_net: bool = False,
    value_state: Optional[dict] = None,
) -> "PPOTrainer":
    """Rebuild a fresh CPU, inference-only PPOTrainer from plain state dicts
    -- the single source of truth for "give a worker subprocess a usable
    trainer" across every parallel-worker path in this file (live
    nn.Module/optimizer objects aren't picklable across a process boundary,
    so every one of these workers rebuilds from scratch instead of
    inheriting a live trainer). Used by _eval_worker_factory below (parallel
    seeded eval) and ai/ppo/dagger.py's parallel rollout collection --
    previously duplicated inline in the former only; extracted here once a
    second caller needed the exact same rebuild logic.
    """
    trainer = PPOTrainer.from_config(
        device=torch.device("cpu"), inference_only=True, separate_value_net=separate_value_net,
    )
    # Tolerant, not module.load_state_dict(..., strict=...) directly: a
    # physics-encoder-enabled DecisionNetwork's state_dict() deliberately
    # excludes ball_physics_encoder.*/player_physics_encoder.* (see that
    # method's own docstring -- they're an external checkpoint artifact,
    # never part of the main PPO checkpoint), but a freshly-constructed
    # DecisionNetwork here still HAS those params (loaded fresh from their
    # own physics_pretrain checkpoint path by from_config() above) -- a
    # strict load would then fail on "missing keys" for something that was
    # never supposed to travel through decision_state at all. Mirrors
    # load_checkpoint()'s own tolerant load for the exact same reason.
    _load_state_dict_tolerant(trainer.decision_net, decision_state, "decision_net")
    _load_state_dict_tolerant(trainer.execution_net, execution_state, "execution_net")
    if value_state is not None and trainer.value_net is not None:
        _load_state_dict_tolerant(trainer.value_net, value_state, "value_net")
    return trainer


def _build_eval_env_factory(use_rules_ai: bool, max_episode_s: float):
    """Builds a ``seed -> ScenarioEnv`` factory for periodic phase-1 eval
    (rules or immobile opponent). Extracted out of ``_eval_worker_factory``
    below so ``ai/eval/eval_worker.py``'s PERSISTENT eval workers can reuse
    the exact same env-building logic without also paying for a trainer
    rebuild on every eval call -- their trainer is cached across many eval
    calls and only refreshed on an explicit ``set_weights`` message, unlike
    the throwaway-Pool path below which rebuilds fresh every single call."""
    from footballcoach.rules_ai import Phase1RulesAI
    from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition
    from footballcoach.ai.env.scenario_env import ScenarioEnv
    from footballcoach.ai.eval.seeded_eval import swap_player_states

    _label = "rules" if use_rules_ai else "immobile"

    def _eval_env_factory(seed: int, swap: bool = False) -> ScenarioEnv:
        def _eval_build(*_a, **_kw):
            _m = build_1v1_scenario(*_a, seed=seed, **_kw)
            if swap:
                swap_player_states(_m, "trainee", "opponent")
            if use_rules_ai:
                _m.player_by_id("opponent").ai = Phase1RulesAI()
            _m._opponent_use_rules_ai = use_rules_ai
            _m._opponent_is_immobile = not use_rules_ai
            return _m

        return ScenarioEnv(
            ScenarioDefinition(key=f"_eval_{_label}", label=f"eval_{_label}",
                               description=f"periodic {_label} eval", build=_eval_build),
            trainee_player_id="trainee",
            max_episode_s=max_episode_s,
        )

    return _eval_env_factory


def _eval_worker_factory(
    decision_state: dict, execution_state: dict, separate_value_net: bool,
    value_state: Optional[dict], use_rules_ai: bool, max_episode_s: float,
) -> tuple:
    """Module-level (picklable) zero-arg-after-partial factory for parallel
    seeded eval (ai/eval/seeded_eval.py's run_seeded_evaluation_parallel) --
    each subprocess rebuilds its own inference-only PPOTrainer from the
    passed state dicts via rebuild_inference_trainer(), mirroring
    ai/ppo/rollout_worker.py."""
    trainer = rebuild_inference_trainer(decision_state, execution_state, separate_value_net, value_state)
    return _build_eval_env_factory(use_rules_ai, max_episode_s), trainer._sample_action


def _eval_worker_factory_batched(
    decision_state: dict, execution_state: dict, separate_value_net: bool,
    value_state: Optional[dict], use_rules_ai: bool, max_episode_s: float,
) -> tuple:
    """Batched-eval counterpart to ``_eval_worker_factory``: returns
    ``(env_factory, trainer, secondary_trainer)`` -- the trainer OBJECT
    (not just its bound ``sample_action_fn``), since
    ``run_seeded_evaluation_batched`` needs ``_sample_action_batch()``.
    ``secondary_trainer`` is always ``None`` here: rules/immobile opponents
    have no neural decision to batch (see ``_secondary_neural_candidates``'s
    duck-typed skip of anything without ``is_due_for_decision()``)."""
    trainer = rebuild_inference_trainer(decision_state, execution_state, separate_value_net, value_state)
    return _build_eval_env_factory(use_rules_ai, max_episode_s), trainer, None


def _build_neural_vs_neural_env_factory(max_episode_s: float, opponent_sample_action_fn):
    """Builds a ``(seed, swap) -> ScenarioEnv`` factory for neural-vs-neural
    snapshot eval (see ``PPOTrainer._eval_vs_neural_snapshot``): the trainee
    uses whatever ``sample_action_fn`` the caller assigns onto the returned
    env (same convention as ``_build_eval_env_factory``), while the
    'opponent' secondary player is pinned to a DIFFERENT, fixed
    ``opponent_sample_action_fn`` (a frozen snapshot) via a small
    ``ScenarioEnv`` subclass -- ``reset()``'s normal
    ``maybe_assign_neural_opponent()`` (``ai/env/scenario_env.py``) always
    mirrors the trainee's OWN live weights onto secondary players (real
    self-play), so this overrides the opponent's ``.ai`` right after
    ``super().reset()`` assigns it, with an otherwise-identical
    ``NeuralPlayerAI`` just pointed at the snapshot instead."""
    from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition
    from footballcoach.ai.env.scenario_env import ScenarioEnv
    from footballcoach.ai.eval.seeded_eval import swap_player_states

    class _NeuralOpponentEnv(ScenarioEnv):
        def reset(self, *, seed: Optional[int] = None):
            obs = super().reset(seed=seed)
            from footballcoach.rules_ai import NeuralPlayerAI
            match = self._loop.match
            try:
                opponent = match.player_by_id("opponent")
            except KeyError:
                return obs
            opponent.ai = NeuralPlayerAI(
                opponent_sample_action_fn,
                decision_interval_ticks=self._ticks_per_decision,
                max_episode_s=self.max_episode_s,
                ema_smoothed=self._sec_ema["opponent"].smoothed,
                rng=self.rng,
            )
            return obs

    def _eval_env_factory(seed: int, swap: bool = False) -> ScenarioEnv:
        def _eval_build(*_a, **_kw):
            # opponent_rules_prob/opponent_immobile_prob=0.0: force
            # build_1v1_scenario's OWN internal opponent-type roll to land on
            # its "neural" branch (ai=None, no order touched at all) instead
            # of its default opponent_immobile_prob=1.0, which would assign a
            # JogOrder that re-asserts a fixed direction every tick
            # regardless of what player.ai does (see that function's own
            # opponent-roll comment, and ui/scenarios.py's
            # build_checkpoint_vs_checkpoint-style scenario for the exact
            # same fix already applied once before for the identical reason).
            # Without this, the opponent.ai reassignment below is completely
            # neutralized by the engine re-applying the leftover JogOrder on
            # top of it every tick -- confirmed via a real self-play sanity
            # check (checkpoint vs itself came back ~90% "trainee" win rate
            # instead of ~50%, i.e. the opponent was just jogging in a
            # straight line, never actually running any network).
            _m = build_1v1_scenario(
                *_a, seed=seed, opponent_rules_prob=0.0, opponent_immobile_prob=0.0, **_kw,
            )
            if swap:
                swap_player_states(_m, "trainee", "opponent")
            return _m

        return _NeuralOpponentEnv(
            ScenarioDefinition(key="_eval_neural_snapshot", label="eval_neural_snapshot",
                               description="neural-vs-neural snapshot eval", build=_eval_build),
            trainee_player_id="trainee",
            secondary_player_ids=["opponent"],
            max_episode_s=max_episode_s,
        )

    return _eval_env_factory


def _eval_worker_factory_neural_snapshot(
    trainee_decision_state: dict, trainee_execution_state: dict, trainee_value_state: Optional[dict],
    opponent_decision_state: dict, opponent_execution_state: dict, opponent_value_state: Optional[dict],
    separate_value_net: bool, max_episode_s: float,
) -> tuple:
    """Module-level (picklable) factory for parallel neural-vs-neural
    snapshot eval (``PPOTrainer._eval_vs_neural_snapshot``) -- mirrors
    ``_eval_worker_factory``, but rebuilds TWO inference-only PPOTrainers
    (the trainee's current live weights, and a frozen snapshot for the
    opponent) instead of one."""
    trainee_trainer = rebuild_inference_trainer(
        trainee_decision_state, trainee_execution_state, separate_value_net, trainee_value_state)
    opponent_trainer = rebuild_inference_trainer(
        opponent_decision_state, opponent_execution_state, separate_value_net, opponent_value_state)
    env_factory = _build_neural_vs_neural_env_factory(max_episode_s, opponent_trainer._sample_action)
    return env_factory, trainee_trainer._sample_action


def _eval_worker_factory_neural_snapshot_batched(
    trainee_decision_state: dict, trainee_execution_state: dict, trainee_value_state: Optional[dict],
    opponent_decision_state: dict, opponent_execution_state: dict, opponent_value_state: Optional[dict],
    separate_value_net: bool, max_episode_s: float,
) -> tuple:
    """Batched-eval counterpart to ``_eval_worker_factory_neural_snapshot``:
    returns ``(env_factory, trainee_trainer, opponent_trainer)`` instead of
    ``(env_factory, trainee_sample_action_fn)`` -- ``run_seeded_evaluation_batched``
    batches the opponent's decisions too (secondary_trainer=opponent_trainer)
    the same way ``BatchedEnvGroup.secondary_trainer`` does for training
    self-play, since neural-vs-neural eval's 'opponent' is exactly that kind
    of secondary neural player (see ``_build_neural_vs_neural_env_factory``'s
    ``_NeuralOpponentEnv``)."""
    trainee_trainer = rebuild_inference_trainer(
        trainee_decision_state, trainee_execution_state, separate_value_net, trainee_value_state)
    opponent_trainer = rebuild_inference_trainer(
        opponent_decision_state, opponent_execution_state, separate_value_net, opponent_value_state)
    env_factory = _build_neural_vs_neural_env_factory(max_episode_s, opponent_trainer._sample_action)
    return env_factory, trainee_trainer, opponent_trainer


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


def _ai_types(obs_dict: dict) -> tuple:
    """Extract (self_ai_type, other_ai_type) tensors from an obs dict, or
    (None, None) if absent — DecisionNetwork/ExecutionNetwork.forward()
    default to all-zero one-hots in that case. Centralised here so every
    ``decision_net(...)``/``execution_net(...)`` call site in this file uses
    the identical fallback behaviour. See ai/knowledge.md "Opponent-AI-type
    (value-only)".
    """
    return obs_dict.get("self_ai_type"), obs_dict.get("other_ai_type")



# All phase-1 StepInfo.trial_outcome values (see ScenarioEnv.step()) that
# outcome_breakdown() below always reports a percentage for, even when a
# given rollout/eval happens to have zero of them (0% rather than silently
# omitted) — win/loss stay first for backward-compat with old log-scrapers
# that split on "/".
_PHASE1_OUTCOME_KEYS: list[tuple[str, str]] = [
    ("box_possession", "win"),
    ("opponent_box_possession", "loss"),
    ("timeout", "tout"),
    ("miss", "miss"),
    ("invalid", "inval"),
]


def outcome_breakdown(outcomes: list[str]) -> str:
    """Format a list of StepInfo.trial_outcome strings as
    "win%/loss%/tout%/miss%/inval%[/other%]" -- a fuller breakdown than just
    win/loss so a swing in win% can be traced to (e.g.) more timeouts vs.
    more losses vs. more ball-out-of-play, instead of both being lumped into
    an invisible remainder. "inval" is a ball-out with no toucher at all
    (nobody's fault, e.g. a bad initial placement) as opposed to "miss"
    (the last toucher is blamed). 'other' only appears if some outcome value
    isn't one of the known keys above (e.g. a phase-2 "goal"/"dispossessed"
    leaking through, or an "unknown" from a missing info object).
    """
    n = len(outcomes)
    if n == 0:
        return "n/a"
    parts = [f"{outcomes.count(key) / n * 100:.1f}%" for key, _ in _PHASE1_OUTCOME_KEYS]
    known = sum(outcomes.count(key) for key, _ in _PHASE1_OUTCOME_KEYS)
    other = n - known
    if other > 0:
        parts.append(f"{other / n * 100:.0f}%")
    return "/".join(parts)


def format_outcomes_with_pct(outcomes: dict) -> str:
    """Format an outcome-count dict as ``{'key': N (P%), ...}`` so relative
    shares are visible in logs alongside the raw counts, without requiring
    mental division while reading them.
    """
    total = sum(outcomes.values())
    if total == 0:
        return "{}"
    parts = [f"'{k}': {v} ({v / total * 100:.1f}%)" for k, v in outcomes.items()]
    return "{" + ", ".join(parts) + "}"


def _episode_abs_adv_means(
    track_ids: list, dones, advantages,
) -> list[float]:
    """Per-episode mean(|advantage|), for episode-seed replay (see
    ``PPOTrainer._update_episode_replay``'s highest-mean-|advantage| seed
    selection). Segments on the TRAINEE's own ``done > 0.5`` events (one
    entry per completed trainee episode, in order) so the result lines up
    1:1 with ``stats["episode_rewards"]``/``stats["episode_seeds"]`` --
    those are populated the same way, only ever appended when the
    trainee's own ``done`` fires. A trailing incomplete episode (no
    terminal trainee ``done=1`` row yet) is silently dropped, matching
    those two stats lists' own behavior.

    Unlike the previous (``_trainee_episode_abs_adv_means``) version of
    this function, each episode's mean is now taken over EVERY row in that
    episode's span -- the trainee's own rows AND any interleaved
    secondary-track rows -- not just the trainee's. This is deliberate,
    not an oversight: whenever a secondary/opponent row exists in the
    buffer at all, it is -- today -- ALWAYS the current, live network
    playing itself. See ai/ppo/batched_rollout_worker.py's "Secondary
    neural players" docstring section: ``_batched_worker_main`` passes
    ``secondary_trainer=trainer``, the exact same weights as the trainee's
    own network, and ``_secondary_neural_candidates()`` excludes
    rules-based/immobile secondaries entirely (they never had a
    ``NeuralPlayerAI``/``last_transition`` to add to the buffer in the
    first place) -- so a rules/immobile-opponent episode simply never puts
    a non-"trainee" row in the buffer, and this function reduces to
    trainee-only for those episodes automatically, with no explicit
    filtering needed. A secondary row's own ``done`` is always set to the
    SAME shared, env-level ``done`` as the trainee's for that tick (see
    scenario_env.py's ``last_secondary_results`` construction -- one trial
    ends the same way for every player on the pitch at once), so every
    secondary row for an episode arrives strictly between that episode's
    previous trainee-``done`` row and its own trainee-``done`` row (the
    trainee row for a given tick is always added to the buffer BEFORE that
    tick's secondary rows -- see ``_batched_worker_main``/single-process
    ``_collect()``'s ``buffer.add()`` ordering) -- segmenting on trainee
    ``done`` events alone still captures every secondary row exactly once,
    in the right episode, with no separate secondary-side bookkeeping
    needed. Closing a segment is deliberately deferred until the NEXT
    trainee row is seen (not done immediately at the trainee ``done=1``
    row) specifically so that tick's own trailing secondary rows -- which
    arrive right after it, still done=1, before the next episode's first
    trainee row -- are pulled into the segment being closed, not the next
    one.

    CAUTION: if a frozen/older-checkpoint opponent is ever added (this has
    been discussed as a real future possibility), the "any secondary row
    implies current network" invariant this function leans on breaks --
    at that point this needs an explicit per-row "is this the live
    network" signal (not currently threaded through the buffer) before it
    can keep including secondary rows unconditionally. Don't assume this
    still holds without checking when that lands.

    ``dones``/``advantages`` accept plain lists OR anything indexable with
    ``[int] -> float`` (e.g. a ``RolloutBuffer.dones`` list alongside a
    freshly computed ``compute_gae()`` advantages list) -- both are always
    the SAME length as ``track_ids`` (one entry per buffer row).
    """
    means: list[float] = []
    seg: list[int] = []
    pending_close = False
    for i, t in enumerate(track_ids):
        if t == "trainee" and pending_close:
            means.append(sum(abs(advantages[j]) for j in seg) / len(seg))
            seg = []
            pending_close = False
        seg.append(i)
        if t == "trainee" and dones[i] > 0.5:
            pending_close = True
    if pending_close:
        means.append(sum(abs(advantages[j]) for j in seg) / len(seg))
    return means


# Slash-joined short labels matching outcome_breakdown()'s column order, for
# a one-time "vs[...]:" legend prefix on log lines instead of repeating the
# key names on every vs_rules/vs_immobile/vs_neural segment.
_PHASE1_OUTCOME_LEGEND = "/".join(label for _, label in _PHASE1_OUTCOME_KEYS)


def value_mse_by_outcome(
    pred: torch.Tensor, target: torch.Tensor, outcomes: list[str],
) -> dict[str, tuple[float, int, float, float]]:
    """Per-outcome (raw MSE, n_rows, target_mean, target_mean_sq) breakdown
    of a value-prediction batch, shared by every value-fitting call site
    (Phase 0 demo warm-up, PPO rollout value pre-training, and the PPO value
    loss itself) so "is the critic doing worse on losses/timeouts than wins"
    can be answered the same way everywhere instead of duplicating the
    group-by logic per caller. ``outcomes`` must be the same length as
    ``pred``/``target`` (e.g. a dataset's ``row_outcomes()`` for demo rows,
    or a rollout batch's ``step_outcomes`` list, with "" -> "unknown").
    Empty/missing outcome strings are grouped under "unknown".

    ``target_mean``/``target_mean_sq`` (mean of target, mean of target**2 --
    NOT variance/std directly) are the ground-truth return's own first two
    raw moments for this group, in the same "per-call mean, multiply by n to
    re-accumulate a sum across chunks" convention ``mse`` already uses --
    callers accumulating this across multiple calls (e.g.
    ``_accum_value_by_outcome`` below) should do the same
    ``value * n``-then-divide-by-total-n dance for these two fields that
    they already do for ``mse``, then derive
    ``std = sqrt(max(target_mean_sq - target_mean**2, 0))`` once at the
    final, fully-accumulated point -- NOT per chunk (mean/mean_sq combine
    linearly across chunks via weighted averaging, std does not).
    """
    if len(outcomes) == 0:
        return {}
    target_np = target.detach().float().cpu().numpy()
    sq_err = (pred.detach() - target).float().cpu().numpy() ** 2
    out: dict[str, tuple[float, int, float, float]] = {}
    labels = np.array([o if o else "unknown" for o in outcomes], dtype=object)
    for name in np.unique(labels):
        mask = labels == name
        n = int(mask.sum())
        out[str(name)] = (
            float(sq_err[mask].sum() / max(n, 1)),
            n,
            float(target_np[mask].mean()),
            float((target_np[mask] ** 2).mean()),
        )
    return out


def log_episode_reward_stats_by_outcome(
    episode_returns: list[float], episode_outcome_raw: list[str], log_prefix: str = "",
) -> None:
    """Log mean/std/min/max of per-episode TOTAL reward (undiscounted sum
    across the whole episode -- i.e. Monte Carlo return with gamma=1, not
    GAE/bootstrapped), grouped by outcome (mapped through
    ``_PHASE1_OUTCOME_KEYS``' short labels; "unknown" for anything else or
    missing). ``episode_returns``/``episode_outcome_raw`` must be the same
    length and in the same per-episode order -- index i of one must
    correspond to index i of the other. No-op if empty."""
    if not episode_returns:
        return
    _label_map = dict(_PHASE1_OUTCOME_KEYS)
    returns_arr = np.array(episode_returns, dtype=np.float64)
    labels = np.array([_label_map.get(o, "unknown") for o in episode_outcome_raw], dtype=object)
    log.info(f"  {log_prefix}episode total reward (gamma=1, {len(returns_arr)} episode(s)):")
    for name in sorted(set(labels)):
        vals = returns_arr[labels == name]
        log.info(
            f"    {name:<6} n={len(vals):>5}  mean={vals.mean():+.3f}  "
            f"std={vals.std():.3f}  min={vals.min():+.3f}  max={vals.max():+.3f}"
        )


def format_outcome_rmse_breakdown(by_outcome: dict[str, tuple[float, int, float, float]]) -> str:
    """One-line ``outcome=rmse(gt=mean±std, n)`` summary of
    ``value_mse_by_outcome()``'s output, for appending to an existing log
    line without a full extra multi-line block -- callers that want the
    fuller multi-line form (see debug_value_network.py) can iterate the
    dict themselves instead.

    Takes the sqrt of each outcome's mean squared error before formatting,
    so the displayed RMSE is in the same units as the value/return itself
    (reward-scale) rather than squared units -- easier to eyeball against
    the return std printed elsewhere on the same log line. The ground-truth
    ``gt=mean±std`` alongside it is what the critic is actually being
    scored against for that outcome group -- an RMSE of e.g. 0.66 reads very
    differently against a gt std of 0.45 (worse than just predicting the
    mean) than against a gt std of 2.0 (a solid fit).

    ``by_outcome`` values are ``(mse, n, target_mean, target_mean_sq)`` --
    the LAST two are target_mean_sq, not variance/std; std is derived here
    as ``sqrt(max(target_mean_sq - target_mean**2, 0))``, matching
    ``value_mse_by_outcome()``'s own docstring on why that derivation must
    happen only once, at the final fully-accumulated point.
    """
    if not by_outcome:
        return ""
    return "  ".join(
        f"{name}={math.sqrt(max(mse, 0.0)):.3f}"
        f"(gt={gt_mean:+.2f}±{math.sqrt(max(gt_mean_sq - gt_mean ** 2, 0.0)):.2f}, n={n})"
        for name, (mse, n, gt_mean, gt_mean_sq) in sorted(by_outcome.items())
    )


def _binary_confusion_counts(
    pred_logit: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Return (tp, fp, fn) counts for a Bernoulli head over valid rows.

    ``pred_logit``/``target`` are raw (pre-sigmoid) logits and 0/1 float
    labels respectively, shape (N,) or (N, 1) (squeezed internally).
    ``valid_mask`` selects which rows to include (e.g. BC's _I_VALID mask).
    Used to accumulate precision/recall/F1 diagnostics for rare-positive
    heads (kick_this_tick, tackle_attempt), where a flat/low mean predicted
    probability alone can't distinguish over-firing (low precision) from
    under-firing (low recall) -- see ai_trainer_knowledge.md BC diagnostics.
    """
    with torch.no_grad():
        p = torch.sigmoid(pred_logit.squeeze(-1))[valid_mask]
        t = target[valid_mask] > 0.5
        pred_pos = p > threshold
        tp = float((pred_pos & t).sum())
        fp = float((pred_pos & ~t).sum())
        fn = float((~pred_pos & t).sum())
    return tp, fp, fn


def _precision_recall_f1(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """Precision/recall/F1 from accumulated (tp, fp, fn) counts.

    NaN-safe: returns float('nan') for any ratio with a zero denominator (no
    positive predictions this epoch / no positive labels seen this epoch)
    rather than raising or silently returning 0.0, so it's visually distinct
    from a genuine 0.0 score in logs.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1 = (
        2 * precision * recall / (precision + recall)
        if (tp + fp) > 0 and (tp + fn) > 0 and (precision + recall) > 0
        else float('nan')
    )
    return precision, recall, f1


class PPOTrainer:
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
    ):
        # Wrapped so every forward call automatically canonicalizes
        # self_feat/other_feat/ball_feat into the canonical AI frame (see
        # ai/obs/canonical.py "CanonicalNetworkWrapper") — existing call
        # sites throughout this file need NO changes; state_dict()/
        # load_state_dict() are transparently delegated so checkpoints stay
        # unaffected.
        self.decision_net = CanonicalNetworkWrapper(decision_net)
        self.execution_net = CanonicalNetworkWrapper(execution_net)
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
        self.ent_dir_weight = float(ppo_cfg.get("ent_dir_weight", 1.0))
        self.ent_kick_power_weight = float(ppo_cfg.get("ent_kick_power_weight", 1.0))
        self.ent_kick_spin_weight = float(ppo_cfg.get("ent_kick_spin_weight", 1.0))
        self.ent_decision_weight = float(ppo_cfg.get("ent_decision_weight", 1.0))
        self.augment_n_slot_shuffles = int(ppo_cfg.get("augment_n_slot_shuffles", 0))
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
        self._neural_snapshot_lookbacks: list[int] = [
            int(k) for k in eval_cfg.get("neural_snapshot_lookbacks", [1, 5, 20])
        ]
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

    def _value_heads(self, sf, of, em, bf, gf, d_heads, sat, oat) -> "torch.Tensor":
        """Return THE critic value estimate (batch, 1) -- and ONLY that,
        every caller of this method must never need anything else.

        When ``separate_value_net`` is disabled (default), this is just
        ``self.execution_net(...)`` (the normal shared-trunk forward pass,
        already computed by the caller in nearly every call site — this
        method exists so call sites that only need value can go through
        one path). When enabled, forwards through the dedicated
        ``self.value_net`` instead — a fully independent ExecutionNetwork
        with its own trunk/encoders, never touched by BC losses. Both take
        identical inputs (same decision_heads too).

        Always calls with ``value_only=True`` (see ExecutionNetwork.forward's
        own docstring) -- this returns the raw value tensor directly, NOT an
        ExecutionHeadsRaw, so a caller that (wrongly) expected e.g.
        ``.move_direction`` off the result fails immediately and loudly
        rather than silently reading a stale/placeholder field.
        """
        if self.separate_value_net:
            return self.value_net(sf, of, em, bf, gf, d_heads, sat, oat, value_only=True)
        return self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat, value_only=True)

    # -----------------------------------------------------------------------
    # Curriculum helpers
    # -----------------------------------------------------------------------

    def _get_value_pretrain_freeze_params(self) -> list:
        """Return params to set requires_grad=False during value warm-up.

        Controlled by ``bc.value_pretrain_frozen_layers`` in ai_config.json:
          -1 / 3 : encoders + full trunk (default — matches prior behaviour)
          2       : encoders + first trunk Linear only
          1       : pre-trunk encoders only
          0       : nothing frozen (gradients flow through trunk freely)
        """
        n = self._value_pretrain_frozen_layers
        if n == 0:
            return []
        params: list[nn.Parameter] = []
        for net in (self.decision_net, self.execution_net):
            pre_trunk: list[nn.Module] = [
                net.entity_encoder, net.self_mlp, net.ball_mlp, net.global_mlp,
            ]
            if hasattr(net, "decision_mlp"):
                pre_trunk.append(net.decision_mlp)
            layer_groups = [
                pre_trunk,
                [net.trunk[0]],
                [net.trunk[2]],
            ]
            count = len(layer_groups) if n == -1 else min(n, len(layer_groups))
            for group in layer_groups[:count]:
                for m in group:
                    params.extend(m.parameters())
        return params

    def _freeze_for_value_only_bc(self) -> None:
        """Freeze decision_net entirely, and execution_net entirely EXCEPT
        its value head(s) -- pair with ``_unfreeze_after_value_only_bc()``.

        Used by ``pretrain_combined()`` when ``bc.bc_train_value_only`` is
        set: Phase 0's forward pass still needs decision_net (the value head
        reads decision-network context) and execution_net's trunk (the value
        head sits on top of it), but with both frozen here no gradient flows
        past the value head's own parameters, so Adam only ever sees nonzero
        grad on the value head. ``decision_net.value_head`` is already
        permanently frozen elsewhere (single value head convention) and
        stays that way regardless -- this does not touch it either way.

        Saves each param's prior ``requires_grad`` onto
        ``self._voe_saved_grad_state`` so ``_unfreeze_after_value_only_bc()``
        can restore it exactly, rather than assuming everything was
        previously trainable.
        """
        self._voe_saved_grad_state = [(p, p.requires_grad) for p in self.decision_net.parameters()]
        self._voe_saved_grad_state += [(p, p.requires_grad) for p in self.execution_net.parameters()]
        self.decision_net.requires_grad_(False)
        self.execution_net.requires_grad_(False)
        self.execution_net.value_head.requires_grad_(True)
        if hasattr(self.execution_net, "value_ai_type_channel"):
            self.execution_net.value_ai_type_channel.requires_grad_(True)

    def _unfreeze_after_value_only_bc(self) -> None:
        """Restore every param's ``requires_grad`` saved by
        ``_freeze_for_value_only_bc()``, so a value-only BC pass never
        leaks a frozen policy into what runs after it (Phase 2/3's rollout
        warm-up, then PPO)."""
        for p, orig in self._voe_saved_grad_state:
            p.requires_grad_(orig)
        self._voe_saved_grad_state = []

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

    def _inactive_head_lp_keys(self) -> frozenset[str]:
        """HEAD_LP_KEYS-style short names (rollout_buffer.py) of every head
        that's currently forced to ratio=1 / zero entropy everywhere --
        curriculum-frozen decision heads (_ppo_lp_masked_heads, translated
        via _LP_HEAD_NAME_TO_SHORT) plus kick_spin when permanently frozen
        (_kick_spin_frozen, a separate flag from the curriculum mask).
        These heads' per-head diagnostic values are either an exact 0.0000
        (entropy, KL -- both literally zero when old==new==0) or a
        content-free "-mean(this slice's advantages)" residual (the
        counterfactual policy_loss breakdown, since ratio=1 collapses the
        clipped surrogate to -adv regardless of the head) -- never a real
        gradient/training signal either way, so every per-head diagnostic
        line drops these entirely rather than clutter the output with
        guaranteed-uninformative entries."""
        keys = {_LP_HEAD_NAME_TO_SHORT[name] for name in self._ppo_lp_masked_heads}
        if self._kick_spin_frozen:
            keys.add("kick_spin")
        return frozenset(keys)

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
                )
                episode_rewards.clear()
                secondary_episode_rewards.clear()
                episode_outcomes_vs_rules.clear()
                episode_outcomes_vs_immobile.clear()
                episode_outcomes_vs_neural.clear()
                rollout_components.clear()
                episode_comp_list.clear()
                episode_durations_s.clear()

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

        # Always save a final checkpoint so the result of the run is not lost
        # even if total_steps is not an exact multiple of rollout_steps.
        if self.checkpoint_dir is not None:
            self._save_checkpoint(self._total_steps)
            log.info("Final checkpoint saved.")

        log.info(f"Training complete. Total steps: {self._total_steps:,}")

    def _log_rollout_summary(
        self,
        metrics: dict,
        steps_per_sec: float,
        episode_rewards: list[float],
        secondary_episode_rewards: list[float],
        episode_outcomes_vs_rules: list[str],
        episode_outcomes_vs_immobile: list[str],
        episode_outcomes_vs_neural: list[str],
        rollout_components: dict[str, float],
        episode_comp_list: list[dict[str, float]],
        episode_durations_s: list[float],
        comp_step_stats: dict[str, dict],
        n_reward_comp_steps: int,
    ) -> None:
        """Emit the full multi-line per-rollout summary (reward breakdown,
        per-episode reward stats, per-component GAE/TD stats, direction
        log_std, action-head probabilities, outcome breakdown).

        Shared by the single-process ``train()`` loop and ``_train_parallel()``
        so both paths produce IDENTICAL diagnostics -- the parallel path
        previously logged a stripped-down one-liner instead of this, which
        was a real regression (see ai_trainer_knowledge.md "Parallel rollout
        collection").
        """
        mean_ep_reward = (
            float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0
        )
        mean_opp_reward = (
            float(np.mean(secondary_episode_rewards[-20:])) if secondary_episode_rewards else float('nan')
        )
        _bc_tk = metrics.get("bc_tackle_loss", 0.0)
        _bc_kick_coeff = metrics.get("bc_kick_coeff", 0.0)
        _bc_tackle_coeff = metrics.get("bc_tackle_coeff", 0.0)
        bc_str = (
            f"  bc={metrics['bc_loss']:.4f}(x{metrics['bc_coeff']:.2f}"
            + (f"/kick x{_bc_kick_coeff:.2f}" if _bc_kick_coeff > 0.0 else "")
            + (f"/tackle x{_bc_tackle_coeff:.2f}" if _bc_tackle_coeff > 0.0 else "")
            + f")[tk={_bc_tk:.3f}]"
            if metrics.get("bc_coeff", 0.0) > 0.0 or _bc_kick_coeff > 0.0 or _bc_tackle_coeff > 0.0 else ""
        )
        # Phase-1 outcome breakdown over this rollout's episodes, split by
        # opponent type -- win/loss/timeout/miss(+other), see
        # outcome_breakdown() for the win%/loss%/tout%/miss% format.
        outcome_parts = []
        if episode_outcomes_vs_rules:
            outcome_parts.append(
                f"vs_rules({len(episode_outcomes_vs_rules)}): {outcome_breakdown(episode_outcomes_vs_rules)}"
            )
        if episode_outcomes_vs_immobile:
            outcome_parts.append(
                f"vs_immobile({len(episode_outcomes_vs_immobile)}): {outcome_breakdown(episode_outcomes_vs_immobile)}"
            )
        if episode_outcomes_vs_neural:
            outcome_parts.append(
                f"vs_neural({len(episode_outcomes_vs_neural)}): {outcome_breakdown(episode_outcomes_vs_neural)}"
            )
        outcome_str = (
            f"  vs[{_PHASE1_OUTCOME_LEGEND}]  " + "  ".join(outcome_parts)
        ) if outcome_parts else ""
        mv_ls = metrics.get('move_log_std', [])
        kk_ls = metrics.get('kick_log_std', [])
        kz_ls = metrics.get('kick_z_log_std', [])
        kp_ls = metrics.get('kick_power_log_std', [])
        mv_ls_grad = metrics.get('mv_ls_grad', 0.0)
        # These are actually log_kappa now (von Mises concentration), not
        # log_std -- see ai_trainer_knowledge.md "Direction heads: von Mises".
        # kappa=exp(log_kappa); for large kappa a von Mises approaches a
        # wrapped normal with variance ~= 1/kappa, so degrees(sqrt(1/kappa))
        # is the (approximate) angular std this actually controls -- the
        # NUMBER that actually controls how tightly direction samples
        # cluster around the predicted mean, same role log_std's sigma
        # played before, just via the inverse relationship (larger kappa =
        # narrower). Delta vs the previous rollout's value shows whether
        # log_kappa is actually moving at all (it was observed to sit frozen
        # at its init value across many rollouts when the policy LR is tiny
        # and early-stop cuts gradient steps short).
        def _kappa_deg(ls_pair):
            if not ls_pair:
                return None
            kappa = [math.exp(v) for v in ls_pair]
            deg = [math.degrees(math.sqrt(1.0 / k)) if k > 0 else float("inf") for k in kappa]
            return kappa, deg
        # kick_dir_z_log_std is a PLAIN Gaussian log_std (elevation component,
        # not a von Mises kappa) -- larger = wider, same convention the old
        # log_std parameters used, so sigma=exp(log_std) directly (no
        # kappa->degrees inversion needed here). See ai_trainer_knowledge.md
        # "Direction heads: von Mises".
        def _sigma(ls_pair):
            if not ls_pair:
                return None
            return [math.exp(v) for v in ls_pair]
        _mv_sig = _kappa_deg(mv_ls)
        _kk_sig = _kappa_deg(kk_ls)
        _kz_sig = _sigma(kz_ls)
        _kp_sig = _sigma(kp_ls)
        _prev_mv_ls = self._prev_move_log_std if hasattr(self, "_prev_move_log_std") else None
        _prev_kk_ls = self._prev_kick_log_std if hasattr(self, "_prev_kick_log_std") else None
        _prev_kz_ls = self._prev_kick_z_log_std if hasattr(self, "_prev_kick_z_log_std") else None
        _prev_kp_ls = self._prev_kick_power_log_std if hasattr(self, "_prev_kick_power_log_std") else None
        _mv_delta_str = ""
        if mv_ls and _prev_mv_ls:
            _d = [b - a for a, b in zip(_prev_mv_ls, mv_ls)]
            # Show Δlog_kappa and the resulting change in angular std (degrees)
            _mv_dstd_deg = [
                abs(math.degrees(math.sqrt(1.0 / math.exp(b))) - math.degrees(math.sqrt(1.0 / math.exp(a))))
                for a, b in zip(_prev_mv_ls, mv_ls)
            ]
            _mv_delta_str = (
                f"  d_move=[{','.join(f'{v:+.4f}' for v in _d)}]"
                f" (Δ(ang std)≈{','.join(f'{v:.3f}°' for v in _mv_dstd_deg)})"
            )
        _kk_delta_str = ""
        if kk_ls and _prev_kk_ls:
            _d = [b - a for a, b in zip(_prev_kk_ls, kk_ls)]
            _kk_dstd_deg = [
                abs(math.degrees(math.sqrt(1.0 / math.exp(b))) - math.degrees(math.sqrt(1.0 / math.exp(a))))
                for a, b in zip(_prev_kk_ls, kk_ls)
            ]
            _kk_delta_str = (
                f"  d_kick=[{','.join(f'{v:+.4f}' for v in _d)}]"
                f" (Δ(ang std)≈{','.join(f'{v:.3f}°' for v in _kk_dstd_deg)})"
            )
        _kz_delta_str = ""
        if kz_ls and _prev_kz_ls:
            _d = [b - a for a, b in zip(_prev_kz_ls, kz_ls)]
            # Same small-angle sigma->degrees approximation as the non-delta
            # kz_ls line above (accurate near-horizontal, z_mean~=0) -- kept
            # analogous to d_move/d_kick's Δ(ang std) rather than a raw Δσ.
            _kz_dstd_deg = [
                abs(math.degrees(math.exp(b)) - math.degrees(math.exp(a)))
                for a, b in zip(_prev_kz_ls, kz_ls)
            ]
            _kz_delta_str = (
                f"  d_kickz=[{','.join(f'{v:+.4f}' for v in _d)}]"
                f" (Δ(ang std)≈{','.join(f'{v:.3f}°' for v in _kz_dstd_deg)})"
            )
        _kp_delta_str = ""
        if kp_ls and _prev_kp_ls:
            _d = [b - a for a, b in zip(_prev_kp_ls, kp_ls)]
            # kick_power isn't an angle (it's a sigmoid-squashed [0,1] power
            # fraction) -- no ang-std-style geometric conversion applies here,
            # unlike kz_ls's z-component-of-a-direction-vector case. Report
            # the raw sigma delta directly.
            _kp_dsig = [abs(math.exp(b) - math.exp(a)) for a, b in zip(_prev_kp_ls, kp_ls)]
            _kp_delta_str = (
                f"  d_kickpower=[{','.join(f'{v:+.4f}' for v in _d)}]"
                f" (Δσ≈{','.join(f'{v:.4f}' for v in _kp_dsig)})"
            )
        self._prev_move_log_std = list(mv_ls) if mv_ls else None
        self._prev_kick_log_std = list(kk_ls) if kk_ls else None
        self._prev_kick_z_log_std = list(kz_ls) if kz_ls else None
        self._prev_kick_power_log_std = list(kp_ls) if kp_ls else None
        mv_ls_str = ""
        if mv_ls:
            mv_ls_str = f"  mv_ls=[{','.join(f'{v:.4f}' for v in mv_ls)}]"
            if _mv_sig:
                _kap, _deg = _mv_sig
                mv_ls_str += f" (\u03ba\u2248{','.join(f'{k:.2f}' for k in _kap)}, ang std\u2248{','.join(f'{d:.0f}\u00b0' for d in _deg)})"
            mv_ls_str += f" g={mv_ls_grad:.2e}" + _mv_delta_str
        if kk_ls:
            mv_ls_str += f"\n  kk_ls=[{','.join(f'{v:.4f}' for v in kk_ls)}]"
            if _kk_sig:
                _kap, _deg = _kk_sig
                mv_ls_str += f" (\u03ba\u2248{','.join(f'{k:.2f}' for k in _kap)}, ang std\u2248{','.join(f'{d:.0f}\u00b0' for d in _deg)})"
            mv_ls_str += _kk_delta_str
        if kz_ls:
            mv_ls_str += f"\n  kz_ls=[{','.join(f'{v:.4f}' for v in kz_ls)}]"
            if _kz_sig:
                # kick_dir_z_log_std's sigma is a std on the raw z-COMPONENT of
                # the kick direction unit vector, not an angle -- elevation
                # angle = arcsin(z), and d(arcsin)/dz = 1/sqrt(1-z^2) = 1 at
                # z=0, so treating sigma directly as a small-angle radian
                # estimate is only accurate for near-horizontal kicks (mean
                # elevation near 0); it under-reports the true angular spread
                # for kicks with a large mean elevation (|z| closer to 1),
                # same caveat the move/kick_dir kappa->degrees conversion
                # already carries for its own small-angle approximation.
                _kz_deg = [math.degrees(s) for s in _kz_sig]
                mv_ls_str += (
                    f" (\u03c3\u2248{','.join(f'{s:.4f}' for s in _kz_sig)}"
                    f", ang std\u2248{','.join(f'{d:.0f}\u00b0' for d in _kz_deg)})"
                )
            mv_ls_str += _kz_delta_str
        if kp_ls:
            mv_ls_str += f"\n  kp_ls=[{','.join(f'{v:.4f}' for v in kp_ls)}]"
            if _kp_sig:
                # kick_power is a plain Gaussian log_std (same convention as
                # kz_ls -- larger = wider), but unlike kz_ls it's not any
                # kind of direction/angle component (SquashedNormalHead
                # sigmoid-squashes it to a [0,1] power fraction) -- no
                # geometric ang-std interpretation applies, so just report
                # sigma directly.
                mv_ls_str += f" (σ≈{','.join(f'{s:.4f}' for s in _kp_sig)})"
            mv_ls_str += _kp_delta_str
        ha = metrics.get("head_act", {})
        _ta_p = ha.get('ta_p', float('nan'))
        _kk_p = ha.get('kk_p', float('nan'))
        _prob_str = (
            (f" tackle_prob={_ta_p:.4f}" if _ta_p == _ta_p else "")
            + (f" kick_prob={_kk_p:.4f}" if _kk_p == _kk_p else "")
        )
        act_str = (
            f"  act: move={ha.get('mv','?'):>3} get_poss={ha.get('gp','?'):>3}"
            f" exec_move={ha.get('emv','?'):>3} sprint={ha.get('spr','?'):>3}"
            f" kick={ha.get('kck','?'):>3} tackle={ha.get('tk','?'):>3}"
            f" shoot={ha.get('sh','?'):>3} hold={ha.get('hld','?'):>3}"
            + _prob_str
        ) if ha else ""
        opp_rew_str = f"/{mean_opp_reward:.2f}" if not (mean_opp_reward != mean_opp_reward) else ""
        comp_parts = [
            f"{label}={rollout_components[k]:+.2f}"
            for k, label in REWARD_COMP_LABELS
            if abs(rollout_components.get(k, 0.0)) > 0.01
        ]
        comp_str = ("  rew: " + "  ".join(comp_parts)) if comp_parts else ""

        # Per-episode reward statistics table (mean/std/min/max per type).
        # Collect all keys that appeared in any episode this rollout,
        # then emit a compact aligned table.
        _rew_stats_lines: list[str] = []
        _n_ep_for_stats = len(episode_comp_list)
        if episode_comp_list:
            _all_keys = [k for k, _ in REWARD_COMP_LABELS
                         if any(k in ep for ep in episode_comp_list)]
            if _all_keys:
                _col_w = 14  # display-label column width
                _hdr = f"  {'component':<{_col_w}}  {'mean':>8}  {'std':>7}  {'min':>8}  {'max':>8}"
                _sep = "  " + "-" * _col_w + "  " + "-" * 8 + "  " + "-" * 7 + "  " + "-" * 8 + "  " + "-" * 8
                _rew_stats_lines.append(_hdr)
                _rew_stats_lines.append(_sep)
                _lbl_map = {k: lbl for k, lbl in REWARD_COMP_LABELS}
                for _k in _all_keys:
                    _vals = [ep[_k] for ep in episode_comp_list if _k in ep]
                    if not _vals:
                        continue
                    _arr = np.array(_vals)
                    # Same "never fired / always exactly zero this rollout"
                    # skip as the "rew/step" table's own "_cs['mean'] == 0.0
                    # and _cs['std'] == 0.0" check -- mean==0 and std==0
                    # together mean every episode's value for this
                    # component was exactly 0.0, so there's nothing to show.
                    # Component identity/whether it's config-disabled can
                    # change between rollouts (a coefficient retuned in
                    # ai_config.json, a curriculum phase change), so this is
                    # a per-rollout runtime check, not a permanent removal
                    # from REWARD_COMP_LABELS -- a component that starts
                    # firing again just reappears on its own next rollout,
                    # nothing to keep in sync by hand.
                    if _arr.mean() == 0.0 and _arr.std() == 0.0:
                        continue
                    _lbl = _lbl_map.get(_k, _k)
                    _rew_stats_lines.append(
                        f"  {_lbl:<{_col_w}}  {_arr.mean():>+8.3f}  {_arr.std():>7.3f}"
                        f"  {_arr.min():>+8.3f}  {_arr.max():>+8.3f}"
                    )

        _val_diag_str = (
            f"V={metrics['values_mean']:.2f}\u00b1{metrics['values_std']:.2f}  "
            f"R={metrics['returns_mean']:.2f}\u00b1{metrics['returns_std']:.2f}  "
            f"adv={metrics['adv_mean']:.2f}\u00b1{metrics['adv_std']:.2f}"
            f"  |adv|={metrics.get('adv_abs_mean', float('nan')):.2f}"
        )
        # Per-head entropy breakdown + \u0394 from the previous rollout \u2014 shows
        # which heads are actually driving a rising aggregate `entropy=`
        # scalar (e.g. a few Bernoulli heads saturating toward p=0.5, or a
        # continuous direction head's log_std climbing toward its clamp)
        # instead of only the summed total. |adv| above is the companion
        # number for the "tug of war" read: entropy's pull on the loss is a
        # FIXED ent_coef every step, while the policy-gradient term's pull
        # scales with |advantage| \u2014 as the policy converges and advantages
        # shrink, entropy's constant pull increasingly wins by default with
        # nothing here to anneal it back down. See ai_trainer_knowledge.md
        # "Reading the training log" / entropy-runaway discussion.
        # Masked/frozen/inactive heads (see _inactive_head_lp_keys) are
        # explicitly zeroed inside _compute_entropy -- a real, exact 0.0000,
        # not a residual -- but showing a wall of guaranteed-zero entries
        # here just buries the heads that are actually live. Dropped
        # entirely rather than displayed as 0.0000.
        _inactive_heads = self._inactive_head_lp_keys()
        _ent_bkdn = {
            k: v for k, v in metrics.get("entropy_breakdown", {}).items()
            if k not in _inactive_heads
        }
        if _ent_bkdn:
            _prev = self._prev_entropy_breakdown or {}
            _ent_parts = [
                f"{k}={v:.4f}({'+' if v - _prev.get(k, v) >= 0 else ''}{v - _prev.get(k, v):.4f})"
                if k in _prev else f"{k}={v:.4f}"
                for k, v in _ent_bkdn.items()
            ]
            self._prev_entropy_breakdown = dict(_ent_bkdn)
        # Tabulated multi-line rollout summary (readability refactor only —
        # every field present in the old single-line format is still here,
        # just grouped/aligned, with full-word labels and long lines wrapped
        # to avoid overflow. See
        # agent_plans/bc_execution_label_boundary_and_followups.md Part 4.
        _lines = [
            "\u2500" * 70,
            f"[PPO] step={self._total_steps:,}  speed={steps_per_sec:.0f}/s  "
            f"reward={mean_ep_reward:.2f}{opp_rew_str}",
            f"  loss     policy={metrics['policy_loss']:.4f}  "
            f"value={metrics['value_loss']:.4f}(x{self.vf_coef})={self.vf_coef * metrics['value_loss']:.4f}"
            f"  val_pre={metrics['pre_update_value_loss']:.4f}",
            f"           ent_coef={metrics['ent_coef']:.4f}  entropy={metrics['entropy']:.4f}"
            f"  kl={metrics['approx_kl']:.4f}"
            + (f"  {bc_str.strip()}" if bc_str else ""),
            f"  value    {_val_diag_str}",
        ]
        if _ent_bkdn:
            _mid_e = (len(_ent_parts) + 1) // 2
            _lines.append(f"  entropy  {'  '.join(_ent_parts[:_mid_e])}")
            if _ent_parts[_mid_e:]:
                _lines.append(f"           {'  '.join(_ent_parts[_mid_e:])}")
        if mv_ls_str:
            _mv_ls_sublines = mv_ls_str.strip().split("\n")
            _lines.append(f"  moves    {_mv_ls_sublines[0].strip()}")
            for _sub in _mv_ls_sublines[1:]:
                _lines.append(f"           {_sub.strip()}")
        if act_str:
            _act_body = act_str.strip().lstrip('act:').strip()
            _act_parts = _act_body.split("  ")
            _mid = (len(_act_parts) + 1) // 2
            _lines.append(f"  heads    {'  '.join(_act_parts[:_mid])}")
            if _act_parts[_mid:]:
                _lines.append(f"           {'  '.join(_act_parts[_mid:])}")
        if outcome_str:
            _lines.append(f"  vs       {outcome_str.strip()}")
        if episode_durations_s:
            _dur_arr = np.array(episode_durations_s)
            _lines.append(
                f"  ep_len   {_dur_arr.mean():.1f}\u00b1{_dur_arr.std():.1f}s"
                f"  (n={len(episode_durations_s)}, min={_dur_arr.min():.1f}s, max={_dur_arr.max():.1f}s)"
            )
        if comp_str:
            _comp_body = comp_str.strip().lstrip('rew:').strip()
            _comp_parts = _comp_body.split("  ")
            _mid = (len(_comp_parts) + 1) // 2
            _lines.append(f"  reward   {'  '.join(_comp_parts[:_mid])}")
            if _comp_parts[_mid:]:
                _lines.append(f"           {'  '.join(_comp_parts[_mid:])}")
        if _rew_stats_lines:
            _lines.append(f"  rew/ep   (mean/std/min/max per episode, {_n_ep_for_stats} ep)")
            for _sl in _rew_stats_lines:
                _lines.append(_sl)
        if comp_step_stats:
            _cw = 14
            _lines.append(f"  rew/step (per-step stats, n={n_reward_comp_steps} steps; ret/gae/td at steps where component fired)")
            _lines.append(
                f"  {'component':<{_cw}}  {'count':>6}  {'mean':>8}  {'std':>7}"
                f"  {'mean_ret':>9}  {'std_ret':>8}  {'mean_gae':>9}"
                f"  {'mean_sq_td':>10}  {'mean|td|':>9}  {'p95|td|':>8}"
            )
            _lines.append("  " + "-" * _cw + "  " + "  ".join(["-"*6, "-"*8, "-"*7, "-"*9, "-"*8, "-"*9, "-"*10, "-"*9, "-"*8]))
            for _ck, _clabel in REWARD_COMP_LABELS:
                _cs = comp_step_stats.get(_ck)
                if _cs is None or (_cs["mean"] == 0.0 and _cs["std"] == 0.0):
                    continue
                def _fmt(v: float, w: int, prec: int = 3) -> str:
                    return f"{v:>+{w}.{prec}f}" if v == v else f"{'nan':>{w}}"
                def _fmtu(v: float, w: int, prec: int = 3) -> str:
                    return f"{v:>{w}.{prec}f}" if v == v else f"{'nan':>{w}}"
                _lines.append(
                    f"  {_clabel:<{_cw}}  {_cs['count']:>6d}  {_fmt(_cs['mean'],8)}  {_cs['std']:>7.3f}"
                    f"  {_fmt(_cs['mean_ret'],9)}  {_cs['std_ret']:>8.3f}  {_fmt(_cs['mean_gae'],9)}"
                    f"  {_fmtu(_cs['mean_sq_td'],10,4)}  {_fmtu(_cs['mean_abs_td'],9,3)}  {_fmtu(_cs['p95_td'],8,3)}"
                )
        _lines.append(
            f"  gae/td   mean_return={metrics['returns_mean']:+.3f}"
            f"  std_return={metrics['returns_std']:.3f}"
            f"  mean_gae={metrics['adv_mean']:+.3f}"
            f"  mean_sq_td={metrics['mean_sq_td']:.4f}"
        )
        _lines.append("\u2500" * 70)
        log.info("\n".join(_lines))

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
                )

                if self.checkpoint_dir is not None:
                    self._save_checkpoint(self._total_steps)

                if self.rollout_eval_trials > 0:
                    self._eval_vs_rules(max_episode_s)
                self._maybe_run_neural_snapshot_eval(max_episode_s)
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
                )

                if self.checkpoint_dir is not None:
                    self._save_checkpoint(self._total_steps)

                if self.rollout_eval_trials > 0:
                    self._eval_vs_rules(max_episode_s)
                self._maybe_run_neural_snapshot_eval(max_episode_s)
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

    def _eval_vs_rules(self, max_episode_s: float) -> None:
        """Quick periodic eval vs immobile AND rules-based AI, shared by
        both the single-process and parallel training loops. Uses a FIXED
        seed list (ai_config.json['eval']) via ai/eval/seeded_eval.py so
        pre-training eval and every rollout's eval see identical scenarios
        -- comparable numbers across the whole run instead of noise from a
        fresh random scenario draw each time. PPO training itself stays
        unseeded. vs-immobile runs first (cheaper baseline sanity check,
        e.g. catching the "runs in circles vs immobile" regression) then
        vs-rules."""
        self._eval_vs_immobile(max_episode_s)
        self._eval_vs_opponent_type(max_episode_s, use_rules_ai=True)

    def _eval_vs_immobile(self, max_episode_s: float) -> None:
        """Seeded eval vs a standing-still opponent -- see _eval_vs_rules()."""
        self._eval_vs_opponent_type(max_episode_s, use_rules_ai=False)

    def _compute_rules_vs_rules_baseline(self, max_episode_s: float):
        """Rules-AI-vs-rules-AI reference point on the EXACT SAME fixed eval
        seed set (self._eval_seeds/_eval_repeats_per_seed) as the periodic
        "[eval vs rules]" check, so a trained policy's numbers can be
        compared directly against "what does a rules-AI trainee itself score
        on these scenarios" -- e.g. beating this baseline's win rate is a
        much more meaningful bar than 0%/100%.

        Computed and cached ONCE (returns the cached result on every
        subsequent call) since it never changes: same seeds, same
        deterministic rules-AI decision logic on both sides (only residual
        match-physics RNG varies run to run, which repeats_per_seed already
        averages over the same way the main eval does). Deliberately
        sequential, not parallel like _eval_vs_opponent_type's main path --
        this only ever runs once per training process, so the extra
        worker-pool code path isn't worth it for a one-time cost.
        """
        if self._rules_vs_rules_baseline is not None:
            return self._rules_vs_rules_baseline
        from footballcoach.rules_ai import Phase1RulesAI
        from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition
        from footballcoach.ai.env.scenario_env import ScenarioEnv
        from footballcoach.ai.eval.seeded_eval import swap_player_states

        def _baseline_env_factory(seed: int, swap: bool = False) -> ScenarioEnv:
            def _build(*_a, **_kw):
                _m = build_1v1_scenario(*_a, seed=seed, **_kw)
                if swap:
                    swap_player_states(_m, "trainee", "opponent")
                for p in _m.players:
                    p.ai = Phase1RulesAI()
                _m._opponent_use_rules_ai = True
                _m._opponent_is_immobile = False
                return _m

            return ScenarioEnv(
                ScenarioDefinition(key="_eval_rules_vs_rules", label="eval_rules_vs_rules",
                                   description="rules-vs-rules eval baseline", build=_build),
                trainee_player_id="trainee",
                max_episode_s=max_episode_s,
            )

        try:
            # sample_action_fn=None: trainee's ai is already Phase1RulesAI()
            # from _build above, so ScenarioEnv.reset() never assigns a
            # NeuralPlayerAI (see its "if self.sample_action_fn is not None"
            # gate) -- both players stay pure rules-AI.
            self._rules_vs_rules_baseline = run_seeded_evaluation(
                _baseline_env_factory, None, self._eval_seeds, self._eval_repeats_per_seed,
                swap_sides=self._eval_swap_sides,
            )
        except Exception as _e:
            log.warning(f"  [eval baseline rules-vs-rules] failed: {_e}")
            self._rules_vs_rules_baseline = False  # sentinel: tried, failed, don't retry every rollout
        return self._rules_vs_rules_baseline

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

    def _eval_vs_opponent_type(self, max_episode_s: float, use_rules_ai: bool) -> None:
        _label = "rules" if use_rules_ai else "immobile"
        # If the curriculum never actually trains against an immobile
        # opponent, this check is pure sanity insurance (catching e.g. a
        # "runs in circles vs immobile" regression), not a tracked metric --
        # cut it down to 10 total episodes instead of the full
        # eval_seeds x eval_repeats_per_seed budget every rollout.
        if not use_rules_ai and self._phase1_opponent_immobile_ratio == 0:
            _seeds = self._eval_seeds[:10]
            _repeats = 1
        else:
            _seeds = self._eval_seeds
            _repeats = self._eval_repeats_per_seed
        try:
            _persistent_workers = getattr(self, "_persistent_eval_workers", None)
            if self._eval_n_parallel_workers > 1 and _persistent_workers:
                # Persistent-pool path (see ai/eval/eval_worker.py): workers
                # were already spawned once by _train_parallel() -- just
                # push current weights and dispatch, no process spawn here.
                _decision_state, _execution_state, _value_state = self._cpu_state_dicts()
                result = self._run_persistent_eval(
                    _persistent_workers, _decision_state, _execution_state, _value_state,
                    _seeds, _repeats, use_rules_ai, max_episode_s,
                    swap_sides=self._eval_swap_sides,
                    batched=self._eval_batched, envs_per_process=self._eval_envs_per_process,
                )
            elif self._eval_n_parallel_workers > 1:
                # Fallback parallel path (no persistent pool available --
                # e.g. evaluate.py CLI or the single-process train() loop):
                # each subprocess rebuilds its own trainer from these state
                # dicts (see _eval_worker_factory) -- must snapshot weights
                # now, not capture self._sample_action, since bound methods/
                # live nn.Modules aren't picklable.
                import functools
                _decision_state, _execution_state, _value_state = self._cpu_state_dicts()
                if self._eval_batched:
                    worker_factory = functools.partial(
                        _eval_worker_factory_batched, _decision_state, _execution_state,
                        self.separate_value_net, _value_state, use_rules_ai, max_episode_s,
                    )
                    result = run_seeded_evaluation_parallel_batched(
                        worker_factory, _seeds, _repeats,
                        n_workers=self._eval_n_parallel_workers,
                        swap_sides=self._eval_swap_sides,
                        envs_per_process=self._eval_envs_per_process,
                    )
                else:
                    worker_factory = functools.partial(
                        _eval_worker_factory, _decision_state, _execution_state,
                        self.separate_value_net, _value_state, use_rules_ai, max_episode_s,
                    )
                    result = run_seeded_evaluation_parallel(
                        worker_factory, _seeds, _repeats,
                        n_workers=self._eval_n_parallel_workers,
                        swap_sides=self._eval_swap_sides,
                    )
            else:
                # Reuse the exact same factory the persistent-pool/fallback-
                # parallel branches above use (_build_eval_env_factory) rather
                # than duplicating the build logic a third time -- keeps
                # swap_sides handling (swap_player_states) in one place.
                _eval_env_factory = _build_eval_env_factory(use_rules_ai, max_episode_s)
                if self._eval_batched:
                    result = run_seeded_evaluation_batched(
                        _eval_env_factory, self,
                        _seeds, _repeats, swap_sides=self._eval_swap_sides,
                        envs_per_process=self._eval_envs_per_process,
                    )
                else:
                    result = run_seeded_evaluation(
                        _eval_env_factory, self._sample_action,
                        _seeds, _repeats, swap_sides=self._eval_swap_sides,
                    )
            log.info(
                f"  [eval vs {_label}] step={self._total_steps:,}  "
                f"seeds={len(_seeds)}x{_repeats}  "
                f"win={result.win_rate_pct:.0f}%  "
                f"mean_rew={result.mean_reward:.3f}±{result.std_reward:.3f} "
                f"(sem={result.sem_reward:.3f})  "
                f"V={result.mean_value_pred:.3f}  gap={result.mean_value_pred - result.mean_reward:+.3f}  "
                f"outcomes={format_outcomes_with_pct(result.outcomes)}"
            )
            if use_rules_ai:
                _baseline = self._compute_rules_vs_rules_baseline(max_episode_s)
                if _baseline:
                    log.info(
                        f"  [eval baseline rules-vs-rules] seeds={len(self._eval_seeds)}x{self._eval_repeats_per_seed}  "
                        f"win={_baseline.win_rate_pct:.0f}%  "
                        f"mean_rew={_baseline.mean_reward:.3f}±{_baseline.std_reward:.3f} "
                        f"(sem={_baseline.sem_reward:.3f})  "
                        f"outcomes={format_outcomes_with_pct(_baseline.outcomes)}"
                    )
        except Exception as _e:
            log.warning(f"  [eval vs {_label}] failed: {_e}")

    def _load_snapshot_dict_from_checkpoint(self, path: Path) -> dict:
        """Load a checkpoint FILE (as already written by _save_checkpoint())
        into the same {"step", "decision", "execution", "value"} shape
        _eval_vs_neural_snapshot() expects as its opponent_snapshot -- the
        single source of truth for "what did the network look like at
        checkpoint N" is the checkpoint file itself, not a separate
        in-memory copy."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "step": ckpt.get("step", 0),
            "decision": ckpt["decision_net"],
            "execution": ckpt["execution_net"],
            "value": ckpt.get("value_net") if self.separate_value_net else None,
        }

    def _eval_vs_neural_snapshot(
        self, max_episode_s: float, label: str, opponent_snapshot: dict, pool=None,
    ) -> None:
        """One neural-vs-neural comparison: current live weights (trainee
        side) vs a frozen snapshot (opponent side), on the SAME fixed eval
        seed set as _eval_vs_rules() (self._eval_seeds/_eval_repeats_per_seed
        -- 'use the same number of episodes as the existing evals'), with the
        same eval.swap_sides fairness treatment. Unlike _eval_vs_opponent_type,
        this doesn't use the persistent eval worker pool (that pool's IPC
        protocol only supports ONE trainer's weights per call, see
        ai/eval/eval_worker.py) -- it uses a throwaway parallel pool via
        run_seeded_evaluation_parallel[_batched], sized by
        eval.eval_n_parallel_workers. Pass ``pool`` (see
        _maybe_run_neural_snapshot_eval, which spawns ONE pool per rollout
        cycle and reuses it across every lookback comparison) rather than
        letting each call spawn its own -- with
        eval.neural_snapshot_lookbacks holding several entries,
        respawning eval_n_parallel_workers fresh PROCESSES per lookback
        (on top of the already-running persistent rollout+eval pools) is
        real, confirmed oversubscription that can stall a live training run
        for many minutes with every process fighting for the same cores.
        Failure is caught and logged, never raised, matching
        _eval_vs_opponent_type -- a broken comparison here must not take
        down the training run."""
        try:
            import functools
            _decision_state, _execution_state, _value_state = self._cpu_state_dicts()
            if self._eval_batched:
                worker_factory = functools.partial(
                    _eval_worker_factory_neural_snapshot_batched,
                    _decision_state, _execution_state, _value_state,
                    opponent_snapshot["decision"], opponent_snapshot["execution"], opponent_snapshot["value"],
                    self.separate_value_net, max_episode_s,
                )
                result = run_seeded_evaluation_parallel_batched(
                    worker_factory, self._eval_seeds, self._eval_repeats_per_seed,
                    n_workers=self._eval_n_parallel_workers,
                    swap_sides=self._eval_swap_sides,
                    envs_per_process=self._eval_envs_per_process,
                    pool=pool,
                )
            else:
                worker_factory = functools.partial(
                    _eval_worker_factory_neural_snapshot,
                    _decision_state, _execution_state, _value_state,
                    opponent_snapshot["decision"], opponent_snapshot["execution"], opponent_snapshot["value"],
                    self.separate_value_net, max_episode_s,
                )
                result = run_seeded_evaluation_parallel(
                    worker_factory, self._eval_seeds, self._eval_repeats_per_seed,
                    n_workers=self._eval_n_parallel_workers,
                    swap_sides=self._eval_swap_sides,
                    pool=pool,
                )
            log.info(
                f"  [eval vs neural:{label}] step={self._total_steps:,}  "
                f"opp_step={opponent_snapshot['step']:,}  "
                f"seeds={len(self._eval_seeds)}x{self._eval_repeats_per_seed}  "
                f"win={result.win_rate_pct:.0f}%  "
                f"mean_rew={result.mean_reward:.3f}±{result.std_reward:.3f} "
                f"(sem={result.sem_reward:.3f})  "
                f"outcomes={format_outcomes_with_pct(result.outcomes)}"
            )
        except Exception as _e:
            log.warning(f"  [eval vs neural:{label}] failed: {_e}")

    def _maybe_run_neural_snapshot_eval(self, max_episode_s: float) -> None:
        """Every eval.neural_snapshot_eval_every_n_rollouts rollouts: compare
        the CURRENT live policy against the checkpoint saved
        eval.neural_snapshot_lookbacks[i] rollouts ago, for every entry in
        that list, plus 'original' (this run's checkpoint1.pt) -- using the
        checkpoint FILES _save_checkpoint() already writes every rollout
        regardless of this feature, rather than keeping a separate in-memory
        snapshot ring. Requires self.checkpoint_dir (no-op without one,
        since there's nothing on disk to compare against). Lookbacks with
        not-yet-enough history (checkpoint count - k < 1) are silently
        skipped, not logged as failures -- this is expected early in a run
        and self-resolves as more checkpoints accumulate.

        All comparisons this cycle share ONE throwaway worker pool (spawned
        once here, closed at the end) instead of each
        _eval_vs_neural_snapshot() call spawning its own -- with several
        lookbacks configured, respawning eval_n_parallel_workers fresh
        processes per comparison stacks on top of the already-running
        persistent rollout+eval pools and can genuinely stall a live run for
        minutes (confirmed live: processes starved of CPU before they could
        even finish importing). Reusing one pool cuts that spawn/teardown
        cost from N times per cycle to once.
        """
        if self._neural_snapshot_every_n <= 0 or self.checkpoint_dir is None:
            return
        self._nn_snapshot_rollout_count += 1
        if self._nn_snapshot_rollout_count % self._neural_snapshot_every_n != 0:
            return

        current_count = self._checkpoint_count  # _save_checkpoint() already ran this rollout
        comparisons: list[tuple[str, dict]] = []
        for k in self._neural_snapshot_lookbacks:
            target_count = current_count - k
            if target_count < 1:
                continue
            ckpt_path = self.checkpoint_dir / f"checkpoint{target_count}.pt"
            if not ckpt_path.exists():
                continue
            try:
                snapshot = self._load_snapshot_dict_from_checkpoint(ckpt_path)
            except Exception as _e:
                log.warning(f"  [eval vs neural:{k}_back] failed to load {ckpt_path}: {_e}")
                continue
            comparisons.append((f"{k}_back", snapshot))

        # checkpoint1.pt always exists by this point (current_count >= 1 is
        # guaranteed -- _save_checkpoint() already ran this rollout, see the
        # comment above) -- always attempt "original", even on rollout 1
        # itself, where current_count == 1 means this IS checkpoint1.pt: a
        # degenerate but still useful self-play comparison (current live
        # weights vs an identical copy of themselves), since both sides
        # still sample stochastically -- a ~50% win rate here is the same
        # "eval pipeline isn't systematically biased" sanity check as this
        # session's earlier manual self-play probes, now free every run.
        original_path = self.checkpoint_dir / "checkpoint1.pt"
        if original_path.exists():
            try:
                comparisons.append(("original", self._load_snapshot_dict_from_checkpoint(original_path)))
            except Exception as _e:
                log.warning(f"  [eval vs neural:original] failed to load {original_path}: {_e}")

        if not comparisons:
            return

        pool = None
        if self._eval_n_parallel_workers > 1:
            import multiprocessing as mp
            # See _capped_thread_env_for_pool_spawn's own docstring: without
            # this, each of these eval_n_parallel_workers fresh processes
            # defaults to one BLAS thread per logical core, on top of the
            # already-running persistent rollout+eval worker pools -- real,
            # confirmed to crash a live run outright with
            # "OMP: Error #111: Memory allocation failed.", not just slow it
            # down. The other throwaway seeded-eval pools (seeded_eval.py's
            # own run_seeded_evaluation_parallel[_batched] fallback path)
            # are capped the same way at their own Pool(...) construction.
            with _capped_thread_env_for_pool_spawn("1"):
                pool = mp.get_context("spawn").Pool(processes=self._eval_n_parallel_workers)
        try:
            for label, snapshot in comparisons:
                self._eval_vs_neural_snapshot(max_episode_s, label, snapshot, pool=pool)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    # -----------------------------------------------------------------------
    # Value pre-training
    # -----------------------------------------------------------------------

    def pretrain_combined(
        self,
        env,
        dataset,
        n_epochs: int,
        batch_size: int,
        bc_lr: float,
        value_lr: float,
        rollout_steps: int,
        value_epochs: int = 5,
        repair_lr: Optional[float] = None,
        experiment_separate_value_net: bool = False,
        phase_id: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        """Joint BC + value pre-training in a single pass.

        Collects one rollout to get GAE returns for the value loss, then for
        each epoch iterates over the BC dataset in minibatches.  Each iteration
        does two backward passes:

          1. BC loss (all parameters) — teaches the policy trunk to imitate the
             rules-based AI.
          2. Value loss (value heads only, trunk detached) — warm-starts the
             critic to predict actual returns, without corrupting the trunk.

        This replaces the two separate ``BCPretrainer.pretrain`` +
        ``pretrain_value`` calls and avoids the oscillation of online BC
        pre-training.

        Args:
            env: ScenarioEnv
            dataset: DemonstrationDataset
            n_epochs: epochs over the BC dataset
            batch_size: minibatch size for both losses
            bc_lr: learning rate for BC (all params)
            value_lr: learning rate for value heads only
            rollout_steps: steps to collect for value targets (≥ rollout_steps in config)
            checkpoint_dir: If given, ``checkpoint_dir/checkpoint_pretrained.pt``
                is saved (overwritten in place) after EACH phase completes
                (Phase 0, Phase 1, Phase 2/3, Phase 4) instead of only once
                at the very end (the caller's own post-call save, e.g.
                train.py's, still happens too -- this just means a crash or
                interrupt partway through no longer loses everything back to
                the start of pretraining; whatever the last COMPLETED phase
                produced is always on disk). ``None`` (default) = no
                intermediate saves, matching prior behaviour.
        """
        from footballcoach.ai.ppo.bc import (
            bc_loss_from_tensor, compute_bc_loss_floor, compute_bc_loss_floor_components,
            direction_magnitude_reg,
        )
        from footballcoach.ai.bc.dataset import DemonstrationDataset

        # pos_weight_*: auto-compute from this dataset if not overridden in config.
        if self._bc_pos_weight_kick_cfg is None or self._bc_pos_weight_tackle_attempt_cfg is None:
            _auto_weights = dataset.compute_pos_weights(max_weight=self._bc_pos_weight_max)
            if self._bc_pos_weight_kick_cfg is None:
                self._bc_pos_weight_kick = _auto_weights["kick"]
            if self._bc_pos_weight_tackle_attempt_cfg is None:
                self._bc_pos_weight_tackle_attempt = _auto_weights["tackle_attempt"]
            log.info(
                f"BC pos_weight (auto-computed from dataset): "
                f"kick={self._bc_pos_weight_kick:.2f}  "
                f"tackle_attempt={self._bc_pos_weight_tackle_attempt:.2f}"
            )

        log.info(
            f"Combined BC + value pre-training: {n_epochs} epoch(s), "
            f"batch_size={batch_size}, dataset={len(dataset):,} steps, "
            f"rollout_steps={rollout_steps}"
        )

        def _save_pretrain_checkpoint(phase_label: str) -> None:
            """Overwrite checkpoint_dir/checkpoint_pretrained.pt with the
            current weights -- see checkpoint_dir's own docstring above for
            why this runs after every phase instead of only once at the end."""
            if checkpoint_dir is None:
                return
            _path = Path(checkpoint_dir) / "checkpoint_pretrained.pt"
            self._save_checkpoint_to(_path)
            log.info(f"  Pre-trained checkpoint saved after {phase_label}: {_path}")

        bc_opt = torch.optim.Adam(
            list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
            lr=bc_lr, eps=1e-5,
        )
        # NOTE: the value-only optimizer used to be built here, but Phase 2/3's
        # value warm-up now delegates to pretrain_value(), which builds its own
        # internal value_opt over decision_net.value_head + execution_net.value_head
        # (identical param set) — see the pretrain_value() call further below.

        if self._bc_train_value_only:
            log.info(
                "bc_train_value_only=True -- decision_net + execution_net policy "
                "heads frozen for Phase 0; Phase 1's BC epoch loop and the BC "
                "repair epochs are both skipped entirely (nothing to repair -- "
                "the policy never moves during this call). Only the value "
                "head(s) train, via Phase 0 alone. bc_opt above is still built "
                "over the full param set but will only ever see gradient on "
                "the value head once frozen."
            )
            self._freeze_for_value_only_bc()

        # --- Phase 0: decision-network-only warm-up on demo data (before any BC epochs) ---
        # Combined decision-heads-only BC loss + value MSE loss, ONE backward pass.
        # Optimizer covers ALL decision_net parameters (encoders + trunk;
        # decision_net.value_head itself stays frozen — single value head
        # convention, see __init__) PLUS the one live critic's value_head
        # (execution_net.value_head normally, or self.value_net when
        # separate_value_net is enabled — see ai/knowledge.md "Phase 0"
        # note). The rest of execution_net (encoders/trunk/action heads) is
        # NOT trained here; it gets its BC training in Phase 1 below.
        # execution_net still needs a forward pass every minibatch (to
        # produce e_heads.value from d_heads) even when separate_value_net
        # is on and its own value_head is frozen/unused — the actual critic
        # forward goes through self._value_heads() instead. Uses stored
        # rewards/dones so no env interaction is needed. Skipped if the
        # dataset has no reward data or demo_value_pretrain_epochs=0.
        # decision_net's BC warm-up runs regardless of separate_value_net —
        # only WHICH network's value_head is trained alongside it changes;
        # self.value_net gets its own Adam param group (self.value_net_
        # optimizer, built in __init__) instead of the ad-hoc demo_opt used
        # for execution_net.value_head's params in the non-separate case.
        _demo_epochs = self._demo_value_pretrain_epochs
        if _demo_epochs > 0 and dataset.has_rewards:
            if self.separate_value_net:
                demo_opt = torch.optim.Adam(
                    list(self.decision_net.parameters()),
                    lr=self._demo_value_pretrain_lr, eps=1e-5,
                )
                # Fresh optimizer, NOT self.value_net_optimizer -- that one is
                # built at ppo.value_learning_rate (5e-5 by default), tuned
                # for gentle per-rollout updates against a slowly-shifting
                # on-policy return distribution during real PPO, not a bulk
                # offline regression pass over the full demo dataset. Reusing
                # it here made Phase 0 (and Phase 1's later separate-value
                # branch, before this fix) converge far slower than
                # demo_value_pretrain_lr actually allows, and dragged
                # whatever Adam momentum state this builds up straight into
                # real PPO training afterward. demo_value_pretrain_lr is the
                # same rate Phase 1's own separate-value branch now uses, for
                # the identical underlying task (demo-return regression).
                _value_opt = torch.optim.Adam(
                    self.value_net.parameters(), lr=self._demo_value_pretrain_lr, eps=1e-5,
                )
                _value_clip_params = list(self.value_net.parameters())
            else:
                demo_opt = torch.optim.Adam(
                    list(self.decision_net.parameters())
                    + list(self.execution_net.value_head.parameters()),
                    lr=self._demo_value_pretrain_lr, eps=1e-5,
                )
                _value_opt = None
                _value_clip_params = list(self.execution_net.value_head.parameters())
            demo_returns = dataset.compute_returns(gamma=self._demo_value_pretrain_gamma)
            ret_t_all = torch.from_numpy(demo_returns).to(self.device)
            ret_std = ret_t_all.std().clamp(min=1.0)

            _p0_train_idx, _p0_val_idx = dataset.split_train_val_indices(val_frac=0.15, valid_only=True)
            _p0_early_stop_enabled = self._p0_early_stop_patience > 0 and len(_p0_val_idx) > 0
            log.info(
                f"Phase 0 — decision-net warm-up (BC + "
                f"{'self.value_net' if self.separate_value_net else 'execution_net.value_head'} "
                f"MSE; single value head convention): {_demo_epochs} epoch(s), "
                f"gamma={self._demo_value_pretrain_gamma}, "
                f"returns mean={ret_t_all.mean():.2f}  std={ret_std:.2f}  "
                f"lr={self._demo_value_pretrain_lr}  "
                f"phase0_value_coef={self._phase0_value_coef}  "
                f"split: {len(_p0_train_idx):,} train / {len(_p0_val_idx):,} val rows"
            )

            # Requires ground-truth episode outcomes (see
            # DemonstrationDataset.has_episode_outcomes) -- falls back to an
            # all-"n/a" lookup for older recordings rather than raising, so
            # Phase 0 warm-up still runs without the outcome breakdown.
            if dataset.has_episode_outcomes:
                _p0_outcome_by_row = dataset.outcome_by_row()
            else:
                log.warning("Dataset has no ground-truth episode outcomes -- "
                            "Phase 0 val-loss-by-outcome breakdown will be skipped.")
                _p0_outcome_by_row = np.full(len(dataset), "n/a", dtype=object)

            def _eval_p0_val_loss() -> tuple[float, float, float, dict[str, tuple[float, int, float, float]]]:
                """Combined dec_bc + value MSE on the held-out val rows (no grad).
                Returns (combined, bc_adj, val_mse, val_mse_by_outcome) where
                bc_adj = bc - floor and val_mse_by_outcome is the per-outcome
                (raw value MSE, n_rows, gt_mean, gt_std) breakdown (see
                value_mse_by_outcome())."""
                self.decision_net.eval()
                _v_losses: list[float] = []
                _v_bc_losses: list[float] = []
                _v_mse_losses: list[float] = []
                _v_floors: list[float] = []
                _outc_sq_err: dict[str, float] = {}
                _outc_n: dict[str, int] = {}
                _outc_gt_sum: dict[str, float] = {}
                _outc_gt_sqsum: dict[str, float] = {}
                _pos = 0
                with torch.no_grad():
                    for _obs_v, _lbl_v, _ret_v in dataset.iterate_minibatches(
                        batch_size=batch_size, shuffle=False, device=self.device,
                        indices_override=_p0_val_idx, returns=demo_returns,
                    ):
                        # iterate_minibatches(shuffle=False) with indices_override yields
                        # contiguous batch_size-sized slices of _p0_val_idx in order, so
                        # this chunk (for the outcome lookup only) is reconstructable here
                        # without the generator itself yielding row indices.
                        _chunk = _p0_val_idx[_pos:_pos + len(_ret_v)]
                        _pos += len(_ret_v)
                        _sat_v, _oat_v = _ai_types(_obs_v)
                        _d_v = self.decision_net(
                            _obs_v["self_feat"], _obs_v["other_feat"],
                            _obs_v["exists_mask"], _obs_v["ball_feat"], _obs_v["global_feat"],
                            _sat_v, _oat_v,
                        )
                        _lbl_v_c = canonicalize_bc_labels(_lbl_v, x_sign_of(_obs_v["self_feat"]))
                        _bc_v, _ = bc_loss_from_tensor(
                            _lbl_v_c, _d_v, exec_heads=None,
                            direction_loss_weight=self._bc_dir_loss_w,
                            direction_loss_mode=self._bc_dir_loss_mode,
                                region_loss_weight=self._bc_region_loss_w,
                            dec_weight=self._bc_dec_weight,
                            dec_label_smoothing=self._bc_dec_label_smoothing,
                            return_breakdown=True,
                        )
                        _floor_v = compute_bc_loss_floor(
                            _lbl_v,
                            dec_weight=self._bc_dec_weight,
                            dec_label_smoothing=self._bc_dec_label_smoothing,
                            has_exec=False,
                        )
                        _value_v = self._value_heads(
                            _obs_v["self_feat"], _obs_v["other_feat"],
                            _obs_v["exists_mask"], _obs_v["ball_feat"], _obs_v["global_feat"],
                            _d_v, _sat_v, _oat_v,
                        )
                        _pred_v = _value_v.squeeze(-1)
                        _mse_v = F.mse_loss(_pred_v, _ret_v) / (ret_std ** 2)
                        _v_losses.append((_bc_v + self._phase0_value_coef * _mse_v).item())
                        _v_bc_losses.append(_bc_v.item())
                        _v_mse_losses.append(_mse_v.item())
                        _v_floors.append(_floor_v)
                        for _name, (_mse, _n, _gt_mean, _gt_mean_sq) in value_mse_by_outcome(
                            _pred_v, _ret_v, list(_p0_outcome_by_row[_chunk])
                        ).items():
                            _outc_sq_err[_name] = _outc_sq_err.get(_name, 0.0) + _mse * _n
                            _outc_n[_name] = _outc_n.get(_name, 0) + _n
                            _outc_gt_sum[_name] = _outc_gt_sum.get(_name, 0.0) + _gt_mean * _n
                            _outc_gt_sqsum[_name] = _outc_gt_sqsum.get(_name, 0.0) + _gt_mean_sq * _n
                self.decision_net.train()
                _combined = float(np.mean(_v_losses)) if _v_losses else float("nan")
                _bc_mean = float(np.mean(_v_bc_losses)) if _v_bc_losses else float("nan")
                _floor_mean = float(np.mean(_v_floors)) if _v_floors else 0.0
                _mse_mean = float(np.mean(_v_mse_losses)) if _v_mse_losses else float("nan")
                _by_outcome = {
                    name: (
                        _outc_sq_err[name] / max(_outc_n[name], 1),
                        _outc_n[name],
                        _outc_gt_sum[name] / max(_outc_n[name], 1),
                        _outc_gt_sqsum[name] / max(_outc_n[name], 1),
                    )
                    for name in _outc_sq_err
                }
                return _combined, _bc_mean - _floor_mean, _mse_mean, _by_outcome

            _p0_best_val_loss = float("inf")
            _p0_best_state: Optional[dict] = None
            _p0_patience_ctr = 0
            _p0_stopped_early = False

            for epoch in range(_demo_epochs):
                epoch_losses: list[float] = []
                epoch_bc_losses: list[float] = []
                epoch_val_losses: list[float] = []
                epoch_floors: list[float] = []
                if self._downsample_trivial_enabled:
                    _p0_ds_frac = (
                        self._downsample_trivial_frac_high_epoch
                        if epoch >= self._downsample_trivial_epoch_threshold
                        else self._downsample_trivial_frac_default
                    )
                else:
                    _p0_ds_frac = 0.0
                _p0_progress = ProgressReporter(
                    len(_p0_train_idx), prefix=f"  Phase 0 epoch {epoch + 1}/{_demo_epochs}: ",
                )
                _p0_rows_done = 0
                for obs_dict, bc_labels, ret_batch in dataset.iterate_minibatches(
                    batch_size=batch_size, shuffle=True, device=self.device,
                    indices_override=_p0_train_idx, returns=demo_returns,
                    downsample_trivial_frac=_p0_ds_frac,
                    downsample_trivial_cos_threshold=self._downsample_trivial_cos_threshold,
                    downsample_trivial_exclude_radius_steps=self._downsample_trivial_exclude_radius_steps,
                ):
                    _sat, _oat = _ai_types(obs_dict)
                    d_heads = self.decision_net(
                        obs_dict["self_feat"], obs_dict["other_feat"],
                        obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                        _sat, _oat,
                    )
                    bc_labels = canonicalize_bc_labels(bc_labels, x_sign_of(obs_dict["self_feat"]))
                    # decision_net.value_head is frozen (single value head
                    # convention — see __init__); Phase 0's decision-net BC
                    # loss is heads-only (no value term from decision_net).
                    dec_bc_loss, _ = bc_loss_from_tensor(
                        bc_labels, d_heads, exec_heads=None,
                        direction_loss_weight=self._bc_dir_loss_w,
                        direction_loss_mode=self._bc_dir_loss_mode,
                        region_loss_weight=self._bc_region_loss_w,
                        dec_weight=self._bc_dec_weight,
                        dec_label_smoothing=self._bc_dec_label_smoothing,
                        return_breakdown=True,
                    )
                    epoch_floors.append(compute_bc_loss_floor(
                        bc_labels,
                        dec_weight=self._bc_dec_weight,
                        dec_label_smoothing=self._bc_dec_label_smoothing,
                        has_exec=False,
                    ))
                    # The one live critic IS trained here against the same
                    # demo returns — self.value_net when separate_value_net
                    # is on (via _value_heads(), routed to value_opt below),
                    # otherwise execution_net.value_head (via demo_opt). No
                    # other execution-network output (move/kick/tackle/etc
                    # heads) is used or optimized in this phase.
                    _value = self._value_heads(
                        obs_dict["self_feat"], obs_dict["other_feat"],
                        obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                        d_heads, _sat, _oat,
                    )
                    val_loss = F.mse_loss(_value.squeeze(-1), ret_batch) / (ret_std ** 2)
                    combined = dec_bc_loss + self._phase0_value_coef * val_loss
                    demo_opt.zero_grad()
                    if _value_opt is not None:
                        _value_opt.zero_grad()
                    combined.backward()
                    if self._bc_max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(
                            list(self.decision_net.parameters()) + _value_clip_params,
                            self._bc_max_grad_norm,
                        )
                    demo_opt.step()
                    if _value_opt is not None:
                        _value_opt.step()
                    epoch_losses.append(combined.item())
                    epoch_bc_losses.append(dec_bc_loss.item())
                    epoch_val_losses.append(val_loss.item())
                    _p0_rows_done += len(ret_batch)
                    _p0_progress.update(_p0_rows_done, postfix=f"loss={np.mean(epoch_losses):.4f}")
                # Force the bar to its final 100% render (with the trailing
                # newline that unlocks) even though _p0_rows_done usually
                # falls short of len(_p0_train_idx) -- downsample_trivial_frac
                # (see iterate_minibatches() above) skips a chunk of "trivial"
                # rows every epoch, so the running count legitimately never
                # reaches the pre-downsample total on its own. Without this,
                # the live bar never renders "finished" and the next log.info()
                # call runs straight onto its unterminated line.
                _p0_progress.update(_p0_progress.total, postfix=f"loss={np.mean(epoch_losses):.4f}")
                _mean_bc = float(np.mean(epoch_bc_losses))
                _mean_floor = float(np.mean(epoch_floors)) if epoch_floors else 0.0
                log.info(
                    f"  Phase 0 epoch {epoch + 1}/{_demo_epochs}: "
                    f"loss={np.mean(epoch_losses):.4f}  "
                    f"dec_bc={_mean_bc:.4f}  bc_adj={_mean_bc - _mean_floor:.4f}"
                    f"(floor={_mean_floor:.4f})  "
                    f"val_mse={np.mean(epoch_val_losses):.4f}(x{self._phase0_value_coef})="
                    f"{self._phase0_value_coef * np.mean(epoch_val_losses):.4f}"
                )
                if len(_p0_val_idx) > 0:
                    _p0_vl, _p0_vl_bc_adj, _p0_vl_mse, _p0_vl_by_outc = _eval_p0_val_loss()
                    _p0_improved = _p0_vl < (_p0_best_val_loss - self._p0_early_stop_min_delta)
                    _val_core = (
                        f"    val  p0_val_loss={_p0_vl:.4f}  bc_adj={_p0_vl_bc_adj:.4f}  "
                        f"val_mse={_p0_vl_mse:.4f}  best={min(_p0_best_val_loss, _p0_vl):.4f}"
                    )
                    if _p0_early_stop_enabled:
                        log.info(
                            _val_core
                            + ("  (improved)" if _p0_improved else f"  (patience {_p0_patience_ctr + 1}/{self._p0_early_stop_patience})")
                        )
                    else:
                        log.info(_val_core)
                    if _p0_vl_by_outc:
                        log.info(f"    val_rmse by outcome: {format_outcome_rmse_breakdown(_p0_vl_by_outc)}")
                    if _p0_improved:
                        _p0_best_val_loss = _p0_vl
                        if _p0_early_stop_enabled:
                            _p0_best_state = {
                                "decision_net": copy.deepcopy(self.decision_net.state_dict()),
                                **(({"value_net": copy.deepcopy(self.value_net.state_dict())} if self.separate_value_net
                                    else {"exec_value_head": copy.deepcopy(self.execution_net.value_head.state_dict())})),
                            }
                            _p0_patience_ctr = 0
                    elif _p0_early_stop_enabled:
                        _p0_patience_ctr += 1
                        if _p0_patience_ctr >= self._p0_early_stop_patience:
                            log.info(
                                f"  [Phase 0] early stop at epoch {epoch + 1} "
                                f"(val stagnant for {self._p0_early_stop_patience} epochs, "
                                f"best={_p0_best_val_loss:.4f})"
                            )
                            _p0_stopped_early = True
                            break
            if _p0_stopped_early and _p0_best_state is not None:
                _load_state_dict_tolerant(self.decision_net, _p0_best_state["decision_net"], "decision_net")
                if self.separate_value_net:
                    self.value_net.load_state_dict(_p0_best_state["value_net"])
                else:
                    self.execution_net.value_head.load_state_dict(_p0_best_state["exec_value_head"])
                log.info(f"  [Phase 0] restored best-val weights (p0_val_loss={_p0_best_val_loss:.4f})")
            log.info(f"Phase 0 done (decision-net BC + critic value_head warm-up, {_demo_epochs} epoch(s))")
        elif _demo_epochs > 0 and not dataset.has_rewards:
            log.info(
                "Phase 0 skipped — dataset has no reward data "
                "(re-record demonstrations to enable demo value pretrain)"
            )

        if self._bc_train_value_only:
            self._unfreeze_after_value_only_bc()

        _save_pretrain_checkpoint("Phase 0")

        # --- Phase 1: BC epochs over the dataset ---
        # If dataset has reward data and bc_value_coef > 0, also train value
        # against demo returns every minibatch. Two different mechanisms
        # depending on separate_value_net:
        #   - not separate_value_net: value loss (against execution_net.
        #     value_head) is folded into total_loss and shares bc_opt's
        #     single backward/step with the BC loss -- see total_loss below.
        #   - separate_value_net: value loss (against the independent
        #     self.value_net) gets its OWN backward()/self.value_net_
        #     optimizer.step(), computed from a DETACHED d_heads, so it
        #     never contributes gradient into decision_net and never
        #     touches bc_opt/bc_losses/_eval_bc_val_loss's early-stop
        #     patience -- mirrors _ppo_update's identical detach-before-
        #     value_net pattern (see _detach_decision_heads' call sites),
        #     and fixes the gap where self.value_net previously only ever
        #     got demo-return training from Phase 0 (a separate, usually
        #     much shorter epoch budget), never from the bulk of Phase 1.
        _use_joint_val = (
            not self.separate_value_net
            and self._bc_value_coef > 0.0
            and dataset.has_rewards
        )
        _use_separate_value_training = (
            self.separate_value_net
            and self._bc_value_coef > 0.0
            and dataset.has_rewards
        )
        if _use_joint_val or _use_separate_value_training:
            _joint_returns = dataset.compute_returns(gamma=self._demo_value_pretrain_gamma)
            _joint_ret_std = float(np.std(_joint_returns).clip(1.0))
            log.info(
                f"Phase 1 BC epochs will include "
                f"{'joint' if _use_joint_val else 'separate (self.value_net, detached from decision_net)'} "
                f"value loss (coef={self._bc_value_coef}, gamma={self._demo_value_pretrain_gamma}, "
                f"returns std={_joint_ret_std:.2f})"
            )
        else:
            _joint_returns = None
            _joint_ret_std = 1.0
        # Fresh optimizer, NOT self.value_net_optimizer -- see Phase 0's
        # identical fix/rationale above (that one is tuned for gentle
        # per-rollout PPO updates, ~20x slower than demo_value_pretrain_lr,
        # and reusing it here also drags Phase 1's momentum state into real
        # PPO training afterward). Only actually built/used when
        # _use_separate_value_training is True.
        _sep_value_opt = (
            torch.optim.Adam(self.value_net.parameters(), lr=self._demo_value_pretrain_lr, eps=1e-5)
            if _use_separate_value_training else None
        )

        # Do BC first so the rollout is collected with the BC-warmed policy,
        # giving on-policy value targets instead of random-init targets.
        # bc_losses is declared here (not just inside the loop) so the
        # "BC pre-training done" log line below stays well-defined even when
        # n_epochs=0 (e.g. bc.bc_pretrain_epochs_from_ckpt=0 with
        # --latest-pretrain/--pretrain-from-checkpoint) -- the loop body then
        # never runs and bc_losses is simply empty.
        bc_losses: list[float] = []
        bc_floors: list[float] = []

        # --- Episode-level train/val split, reported every epoch (see
        # bc.bc_pretrain_early_stop_patience in ai_config.json for the
        # OPTIONAL early-stop behaviour layered on top). The split itself
        # (and the per-epoch "val bc_val_loss=" log line below) is always
        # computed whenever the dataset has >=2 episodes, regardless of
        # whether early stop is enabled (patience=0 = report-only, no
        # stopping/best-weight-restore -- matches prior "train blind"
        # behaviour for the actual training, just with visibility added). ---
        _bc_early_stop_enabled = self._bc_pretrain_early_stop_patience > 0
        _bc_train_idx, _bc_val_idx = dataset.split_train_val_indices(val_frac=0.15, valid_only=True)
        log.info(
            f"  BC pretrain split: {len(_bc_train_idx):,} train rows"
            + (f"  |  {len(_bc_val_idx):,} val rows" if len(_bc_val_idx) > 0 else "  (val split empty -- <2 episodes, val loss unavailable)")
        )
        _bc_best_val_loss = float("inf")
        _bc_best_state: Optional[dict] = None
        _bc_patience_ctr = 0
        _bc_stopped_early = False

        def _eval_bc_val_loss() -> float:
            """Mean BC loss (no grad) over the held-out val rows, current weights."""
            self.decision_net.eval()
            self.execution_net.eval()
            _losses: list[float] = []
            with torch.no_grad():
                for _obs, _labels in dataset.iterate_minibatches(
                    batch_size=batch_size, shuffle=False, device=self.device,
                    indices_override=_bc_val_idx,
                ):
                    _sat_v, _oat_v = _ai_types(_obs)
                    _d_v = self.decision_net(
                        _obs["self_feat"], _obs["other_feat"], _obs["exists_mask"],
                        _obs["ball_feat"], _obs["global_feat"], _sat_v, _oat_v,
                    )
                    _e_v = self.execution_net(
                        _obs["self_feat"], _obs["other_feat"], _obs["exists_mask"],
                        _obs["ball_feat"], _obs["global_feat"], _d_v, _sat_v, _oat_v,
                    )
                    _labels = canonicalize_bc_labels(_labels, x_sign_of(_obs["self_feat"]))
                    _losses.append(bc_loss_from_tensor(
                        _labels, _d_v, _e_v,
                        direction_loss_weight=self._bc_dir_loss_w,
                        direction_loss_mode=self._bc_dir_loss_mode,
                        region_loss_weight=self._bc_region_loss_w,
                        pos_weight_kick=self._bc_pos_weight_kick,
                        pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                        dec_weight=self._bc_dec_weight,
                        exec_weight=self._bc_exec_weight,
                        dec_label_smoothing=self._bc_dec_label_smoothing,
                        exec_label_smoothing=self._bc_exec_label_smoothing,
                    ).item())
            self.decision_net.train()
            self.execution_net.train()
            return float(np.mean(_losses)) if _losses else float("nan")

        def _eval_p1_value_val_loss() -> tuple[float, float]:
            """Mean value MSE (normalized) + RMSE (raw), no grad, over the
            held-out val rows -- mirrors _eval_bc_val_loss but for the value
            head/net (execution_net.value_head, or self.value_net when
            separate_value_net), so Phase 1's value loss has a genuine
            held-out counterpart. Only meaningful when _use_joint_val or
            _use_separate_value_training (returns are otherwise undefined)."""
            self.decision_net.eval()
            if _use_separate_value_training:
                self.value_net.eval()
            else:
                self.execution_net.eval()
            _losses: list[float] = []
            _raw_mses: list[float] = []
            with torch.no_grad():
                for _obs_v, _lbl_v, _ret_v in dataset.iterate_minibatches(
                    batch_size=batch_size, shuffle=False, device=self.device,
                    indices_override=_bc_val_idx, returns=_joint_returns,
                ):
                    _sat_v, _oat_v = _ai_types(_obs_v)
                    _d_v = self.decision_net(
                        _obs_v["self_feat"], _obs_v["other_feat"], _obs_v["exists_mask"],
                        _obs_v["ball_feat"], _obs_v["global_feat"], _sat_v, _oat_v,
                    )
                    if _use_separate_value_training:
                        _d_v_for_value = _detach_decision_heads(_d_v)
                        _value_v = self.value_net(
                            _obs_v["self_feat"], _obs_v["other_feat"], _obs_v["exists_mask"],
                            _obs_v["ball_feat"], _obs_v["global_feat"], _d_v_for_value, _sat_v, _oat_v,
                            value_only=True,
                        )
                    else:
                        _value_v = self.execution_net(
                            _obs_v["self_feat"], _obs_v["other_feat"], _obs_v["exists_mask"],
                            _obs_v["ball_feat"], _obs_v["global_feat"], _d_v, _sat_v, _oat_v,
                            value_only=True,
                        )
                    _v = _value_v.squeeze(-1)
                    _raw_mse = F.mse_loss(_v, _ret_v)
                    _raw_mses.append(_raw_mse.item())
                    _losses.append((_raw_mse / (_joint_ret_std ** 2)).item())
            self.decision_net.train()
            if _use_separate_value_training:
                self.value_net.train()
            else:
                self.execution_net.train()
            if not _losses:
                return float("nan"), float("nan")
            return float(np.mean(_losses)), float(np.sqrt(np.mean(_raw_mses)))

        _p1_n_epochs = 0 if self._bc_train_value_only else n_epochs
        if self._bc_train_value_only:
            log.info("  Phase 1 skipped (bc_train_value_only=True) -- Phase 0 already covers value training.")
        # Per-execution-head grad norm, accumulated across every epoch/
        # minibatch of Phase 1 -- same diagnostic PPO's _ppo_update already
        # gets (see _measure_exec_head_grad_norms_and_maybe_log_spike),
        # summarized once at the end of Phase 1 below. Also runs the
        # grad-norm-spike per-parameter breakdown inline, same as PPO.
        _bc_head_grad_norm: dict[str, list[float]] = {name: [] for name, _ in EXEC_HEAD_MODULES}
        # Total (combined decision_net+execution_net, pre-clip) grad norm
        # per minibatch -- Phase 1 never logged this at all before (only
        # DAgger's mean_grad_norm did, and only a mean, not the full spread).
        _bc_raw_grad_norm: list[float] = []
        for epoch in range(_p1_n_epochs):
            _epoch_t0 = time.monotonic()
            bc_losses = []
            bc_floors = []
            val_losses: list[float] = []
            val_raw_mse_losses: list[float] = []
            sep_val_losses: list[float] = []
            sep_val_raw_mse_losses: list[float] = []
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
            _bkdn_n: int = 0
            if self._downsample_trivial_enabled:
                _ds_frac = (
                    self._downsample_trivial_frac_high_epoch
                    if epoch >= self._downsample_trivial_epoch_threshold
                    else self._downsample_trivial_frac_default
                )
                _ds_stats = dataset.downsample_trivial_stats(
                    valid_only=True,
                    cos_threshold=self._downsample_trivial_cos_threshold,
                    exclude_radius_steps=self._downsample_trivial_exclude_radius_steps,
                    frac=_ds_frac,
                )
                log.info(
                    f"  Downsample trivial rows (epoch {epoch + 1}): "
                    f"{_ds_stats['n_trivial']:,}/{_ds_stats['n_total']:,} "
                    f"({_ds_stats['trivial_frac']:.1%}) rows classified trivial, "
                    f"excluding ~{_ds_stats['n_excluded_at_frac']:,} this epoch "
                    f"(frac={_ds_frac:.2f})"
                )
            else:
                _ds_frac = 0.0
            _p1_progress = ProgressReporter(
                len(_bc_train_idx), prefix=f"  Phase 1 epoch {epoch + 1}/{_p1_n_epochs}: ",
            )
            _p1_rows_done = 0
            for _p1_mb_i, mb in enumerate(dataset.iterate_minibatches(
                batch_size=batch_size, shuffle=True, device=self.device,
                valid_only=True, returns=_joint_returns,
                downsample_trivial_frac=_ds_frac,
                downsample_trivial_cos_threshold=self._downsample_trivial_cos_threshold,
                downsample_trivial_exclude_radius_steps=self._downsample_trivial_exclude_radius_steps,
                indices_override=_bc_train_idx,
            )):
                if _use_joint_val or _use_separate_value_training:
                    obs_dict, bc_labels, ret_batch = mb
                else:
                    obs_dict, bc_labels = mb
                    ret_batch = None
                _p1_rows_done += bc_labels.shape[0]  # pre-augmentation row count
                # Augment with geometric flips + slot permutations (ALWAYS applied).
                if self.augment_n_slot_shuffles > 0:
                    obs_dict, bc_labels = augment_obs_bc(
                        obs_dict, bc_labels, self.augment_n_slot_shuffles, self._aug_rng
                    )
                    if ret_batch is not None:
                        n_aug = N_FLIP_VARIANTS * max(1, self.augment_n_slot_shuffles)
                        ret_batch = ret_batch.repeat(n_aug)
                _sat, _oat = _ai_types(obs_dict)
                d_heads = self.decision_net(
                    obs_dict["self_feat"], obs_dict["other_feat"],
                    obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                    _sat, _oat,
                )
                e_heads = self.execution_net(
                    obs_dict["self_feat"], obs_dict["other_feat"],
                    obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                    d_heads, _sat, _oat,
                )
                bc_labels = canonicalize_bc_labels(bc_labels, x_sign_of(obs_dict["self_feat"]))
                bc_loss, bkdn = bc_loss_from_tensor(
                    bc_labels, d_heads, e_heads,
                    direction_loss_weight=self._bc_dir_loss_w,
                    region_loss_weight=self._bc_region_loss_w,
                    pos_weight_kick=self._bc_pos_weight_kick,
                    pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                    dec_weight=self._bc_dec_weight,
                    exec_weight=self._bc_exec_weight,
                    dec_label_smoothing=self._bc_dec_label_smoothing,
                    exec_label_smoothing=self._bc_exec_label_smoothing,
                    return_breakdown=True,
                )
                dir_mag_reg = direction_magnitude_reg(e_heads, self._bc_dir_mag_reg_coef)
                total_loss = bc_loss + dir_mag_reg
                if _use_joint_val and ret_batch is not None:
                    # Single value head: execution_net only (decision_net.value
                    # is frozen — see __init__ note). Folded into total_loss,
                    # sharing bc_opt's one backward/step with the BC loss --
                    # only reached when NOT separate_value_net (see
                    # _use_separate_value_training's own detached branch
                    # below for that case).
                    v_exc = e_heads.value.squeeze(-1)
                    val_loss = F.mse_loss(v_exc, ret_batch) / (_joint_ret_std ** 2)
                    total_loss = bc_loss + dir_mag_reg + self._bc_value_coef * val_loss
                    val_losses.append(val_loss.item())
                    # raw MSE for RMSE reporting (values already in raw space)
                    with torch.no_grad():
                        raw_mse = F.mse_loss(v_exc, ret_batch)
                        val_raw_mse_losses.append(raw_mse.item())
                bc_opt.zero_grad()
                total_loss.backward()
                _bc_gn_vals = _measure_exec_head_grad_norms_and_maybe_log_spike(self, "BC", epoch, _p1_mb_i)
                for _head_name, _ in EXEC_HEAD_MODULES:
                    if _head_name in _bc_gn_vals:
                        _bc_head_grad_norm[_head_name].append(_bc_gn_vals[_head_name])
                _bc_raw_grad_norm.append(_bc_gn_vals["raw"])
                if self._bc_max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                        self._bc_max_grad_norm,
                    )
                bc_opt.step()

                if _use_separate_value_training and ret_batch is not None:
                    # Own backward/optimizer step, entirely separate from
                    # bc_opt/total_loss above -- d_heads is detached first so
                    # this loss contributes ZERO gradient to decision_net
                    # (mirrors _ppo_update's identical pattern, see
                    # _detach_decision_heads' other call sites/docstring).
                    # Deliberately NOT scaled by self._bc_value_coef: that
                    # coefficient exists to balance a value term AGAINST a BC
                    # term sharing the same backward pass (the _use_joint_val
                    # branch above) -- here value_loss is the only term in
                    # its own graph, so scaling it would just be equivalent
                    # to rescaling _sep_value_opt's own LR, adding a second,
                    # redundant knob for the same effect. Does NOT feed
                    # bc_losses/_eval_bc_val_loss, so it has no effect on
                    # Phase 1's BC early-stop patience (that stays a pure
                    # BC-only metric). Uses _sep_value_opt (fresh, demo_value_
                    # pretrain_lr), NOT self.value_net_optimizer -- see that
                    # variable's own setup comment above for why.
                    d_heads_for_value = _detach_decision_heads(d_heads)
                    _value_sep = self.value_net(
                        obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
                        obs_dict["ball_feat"], obs_dict["global_feat"], d_heads_for_value, _sat, _oat,
                        value_only=True,
                    )
                    v_sep = _value_sep.squeeze(-1)
                    sep_val_loss = F.mse_loss(v_sep, ret_batch) / (_joint_ret_std ** 2)
                    _sep_value_opt.zero_grad()
                    sep_val_loss.backward()
                    if self._bc_max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(list(self.value_net.parameters()), self._bc_max_grad_norm)
                    _sep_value_opt.step()
                    sep_val_losses.append(sep_val_loss.item())
                    with torch.no_grad():
                        sep_raw_mse = F.mse_loss(v_sep, ret_batch)
                        sep_val_raw_mse_losses.append(sep_raw_mse.item())
                bc_losses.append(bc_loss.item())
                bc_floors.append(compute_bc_loss_floor(
                    bc_labels,
                    pos_weight_kick=self._bc_pos_weight_kick,
                    pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                    dec_weight=self._bc_dec_weight,
                    exec_weight=self._bc_exec_weight,
                    dec_label_smoothing=self._bc_dec_label_smoothing,
                    exec_label_smoothing=self._bc_exec_label_smoothing,
                    has_exec=True,
                ))
                for k, v in bkdn.items():
                    _bkdn_acc[k] = _bkdn_acc.get(k, 0.0) + v
                _bkdn_floor = compute_bc_loss_floor_components(
                    bc_labels,
                    pos_weight_kick=self._bc_pos_weight_kick,
                    pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                    dec_label_smoothing=self._bc_dec_label_smoothing,
                    exec_label_smoothing=self._bc_exec_label_smoothing,
                    has_exec=True,
                )
                for k, v in _bkdn_floor.items():
                    _bkdn_floor_acc[k] = _bkdn_floor_acc.get(k, 0.0) + v
                _bkdn_n += 1

                # Accumulate cosine similarity between predicted and label move directions.
                # Use _I_VALID (index 14) not -1 (_I_OPPONENT_AI_TYPE = 0.0 for rules
                # demos, which makes valid_mask always False and causes dir_cos=nan).
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
                    has_kick_dir = (bc_labels[:, 18].abs() + bc_labels[:, 19].abs() + bc_labels[:, 24].abs()) > 1e-6  # _I_KICK_DIR_X/Y/Z
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
                _p1_progress.update(_p1_rows_done, postfix=f"bc_loss={np.mean(bc_losses):.4f}")
            # Force the final 100% render -- see the identical comment on
            # _p0_progress above (same downsample_trivial_frac cause).
            _p1_progress.update(_p1_progress.total, postfix=f"bc_loss={np.mean(bc_losses):.4f}")

            mean_cos = float(np.mean(dir_cosines)) if dir_cosines else float('nan')
            mean_kick_cos = float(np.mean(kick_dir_cosines)) if kick_dir_cosines else float('nan')
            mean_mv = float(np.mean(move_probs)) if move_probs else float('nan')
            mean_spr = float(np.mean(sprint_probs)) if sprint_probs else float('nan')
            mean_kk = float(np.mean(kick_probs)) if kick_probs else float('nan')
            mean_tk = float(np.mean(tackle_attempt_probs)) if tackle_attempt_probs else float('nan')
            kk_prec, kk_rec, kk_f1 = _precision_recall_f1(_kick_tp, _kick_fp, _kick_fn)
            tk_prec, tk_rec, tk_f1 = _precision_recall_f1(_tackle_tp, _tackle_fp, _tackle_fn)
            _kick_bkdn_keys = {"kick", "kick_direction", "kick_power", "kick_spin"}
            # Subtract each component's own analytic label-smoothing floor (see
            # compute_bc_loss_floor_components) before printing, so raw-magnitude
            # differences between components (e.g. exec_bce summing 4 smoothed
            # heads vs. direction's true 0 floor) don't masquerade as real
            # imitation-quality differences. Clamp at 0 — mean-of-floors vs.
            # mean-of-raw over the same minibatches can differ by float noise
            # right at convergence, and a tiny negative "adjusted" value reads
            # as a bug rather than the near-zero residual it actually is.
            bkdn_str = "  ".join(
                f"{k}={max(0.0, v/_bkdn_n - _bkdn_floor_acc.get(k, 0.0)/_bkdn_n):.5f}"
                if k in _kick_bkdn_keys else
                f"{k}={max(0.0, v/_bkdn_n - _bkdn_floor_acc.get(k, 0.0)/_bkdn_n):.3f}"
                for k, v in _bkdn_acc.items()
            ) if _bkdn_n else ""
            if val_losses:
                # NOTE: despite the name, val_losses/val_rmse below are accumulated
                # over TRAINING minibatches (_bc_train_idx), not the held-out val
                # split -- they're the epoch's mean training-batch value loss. See
                # the genuine held-out "value_val_loss=" figure on the "val" line
                # further down (_eval_p1_value_val_loss) for an actual val metric.
                val_rmse = float(np.sqrt(np.mean(val_raw_mse_losses))) if val_raw_mse_losses else float('nan')
                _mean_val = np.mean(val_losses)
                val_str = (
                    f"  train_value_loss={_mean_val:.4f}"
                    f"(x{self._bc_value_coef})={_mean_val * self._bc_value_coef:.4f}"
                    f"  rmse={val_rmse:.2f} (returns std={_joint_ret_std:.1f})"
                )
            else:
                val_str = ""
            if sep_val_losses:
                # Separate, detached self.value_net training (see
                # _use_separate_value_training above) -- NOT scaled by
                # self._bc_value_coef (no combined loss to weight against),
                # so no "(x coef)=" term here unlike val_str above. Same
                # training-batch-not-held-out-val caveat as val_str above --
                # see "value_val_loss=" on the "val" line for the real metric.
                sep_val_rmse = float(np.sqrt(np.mean(sep_val_raw_mse_losses))) if sep_val_raw_mse_losses else float('nan')
                sep_val_str = (
                    f"    value_net  train_loss={np.mean(sep_val_losses):.4f}  "
                    f"rmse={sep_val_rmse:.2f} (returns std={_joint_ret_std:.1f}, detached from decision_net)"
                )
            else:
                sep_val_str = ""
            _epoch_elapsed = time.monotonic() - _epoch_t0
            # Tabulated multi-line epoch summary (readability refactor only — every
            # field from the old single-line format is preserved, just grouped, with
            # full-word labels and long lines wrapped to avoid overflow. See
            # agent_plans/bc_execution_label_boundary_and_followups.md Part 4.
            _mean_bc_loss = float(np.mean(bc_losses))
            _mean_bc_floor = float(np.mean(bc_floors)) if bc_floors else 0.0
            _bc_lines = [
                f"  BC epoch {epoch + 1}/{n_epochs}  ({_epoch_elapsed:.1f}s)",
                f"    loss       bc={_mean_bc_loss:.4f}  bc_adj={_mean_bc_loss - _mean_bc_floor:.4f}"
                f"(floor={_mean_bc_floor:.4f})" + (val_str.strip() and f"  {val_str.strip()}" or ""),
                f"    heads      dir_cos={mean_cos:.3f}  kick_dir_cos={mean_kick_cos:.3f}",
                f"               move_prob={mean_mv:.3f}  sprint_prob={mean_spr:.3f}  "
                f"kick_prob={mean_kk:.3f}  tackle_prob={mean_tk:.3f}",
                f"    pr/rec     kick:   p={kk_prec:.3f}  r={kk_rec:.3f}  f1={kk_f1:.3f}  "
                f"(tp={_kick_tp:.0f} fp={_kick_fp:.0f} fn={_kick_fn:.0f})",
                f"               tackle: p={tk_prec:.3f}  r={tk_rec:.3f}  f1={tk_f1:.3f}  "
                f"(tp={_tackle_tp:.0f} fp={_tackle_fp:.0f} fn={_tackle_fn:.0f})",
            ]
            if sep_val_str:
                _bc_lines.append(sep_val_str)
            if bkdn_str:
                _bkdn_parts = bkdn_str.split("  ")
                _mid = (len(_bkdn_parts) + 1) // 2
                _bc_lines.append(f"    breakdown (floor-adj)  {'  '.join(_bkdn_parts[:_mid])}")
                if _bkdn_parts[_mid:]:
                    _bc_lines.append(f"                           {'  '.join(_bkdn_parts[_mid:])}")
            log.info("\n".join(_bc_lines))
            if _bc_raw_grad_norm:
                log.info(
                    f"    [BC grad norm total] mean={np.mean(_bc_raw_grad_norm):.3f}  "
                    f"min={np.min(_bc_raw_grad_norm):.3f}  max={np.max(_bc_raw_grad_norm):.3f}  "
                    f"std={np.std(_bc_raw_grad_norm):.3f}  (n={len(_bc_raw_grad_norm)})"
                )
            _bc_raw_grad_norm.clear()
            if any(_bc_head_grad_norm.values()):
                _bc_head_gn_str = "  ".join(
                    f"{name}={np.mean(vals):.3f}" for name, vals in _bc_head_grad_norm.items() if vals
                )
                log.info(f"    [BC exec head grad norm] {_bc_head_gn_str}")
            for _v in _bc_head_grad_norm.values():
                _v.clear()
            dir_cosines.clear()
            kick_dir_cosines.clear()
            move_probs.clear()
            sprint_probs.clear()
            _bkdn_acc.clear()
            _bkdn_floor_acc.clear()
            _bkdn_n = 0

            # --- BC pretrain val loss, reported every epoch; early stop is opt-in
            # (see bc.bc_pretrain_early_stop_patience). Also reports a genuine
            # held-out value loss/RMSE (_eval_p1_value_val_loss) on the same line
            # when a value term is being trained this phase -- separate from the
            # "train_value_loss="/"value_net  train_loss=" figures printed above,
            # which are training-batch metrics, not a val-set metric, despite the
            # similar name. Does NOT feed BC's own early-stop bookkeeping below
            # (that stays a pure BC-only metric, see _use_separate_value_training's
            # own note above for why). ---
            if len(_bc_val_idx) > 0:
                _bc_val_loss = _eval_bc_val_loss()
                _improved = _bc_val_loss < (_bc_best_val_loss - self._bc_pretrain_early_stop_min_delta)
                _val_line = f"    val        bc_val_loss={_bc_val_loss:.4f}  best={min(_bc_best_val_loss, _bc_val_loss):.4f}"
                if _bc_early_stop_enabled:
                    _val_line += ("  (improved)" if _improved else f"  (patience {_bc_patience_ctr + 1}/{self._bc_pretrain_early_stop_patience})")
                elif _improved:
                    _val_line += "  (improved, checkpointed)"
                if _use_joint_val or _use_separate_value_training:
                    _val_value_loss, _val_value_rmse = _eval_p1_value_val_loss()
                    _val_line += f"  value_val_loss={_val_value_loss:.4f}  rmse={_val_value_rmse:.2f}"
                log.info(_val_line)
                # Best-val tracking/checkpointing is independent of whether
                # early STOPPING is enabled (patience=0 = report-only for
                # the stop decision, but "checkpoint the actual best" is
                # useful regardless) -- previously this whole block was
                # gated on _bc_early_stop_enabled, so with early stop
                # disabled (the default), a new best was never captured
                # in memory OR on disk at all, and a crash mid-Phase-1
                # lost all progress back to whatever Phase 0 left on disk.
                if _improved:
                    _bc_best_val_loss = _bc_val_loss
                    _bc_best_state = {
                        "decision_net": copy.deepcopy(self.decision_net.state_dict()),
                        "execution_net": copy.deepcopy(self.execution_net.state_dict()),
                    }
                    _save_pretrain_checkpoint(f"Phase 1 (new best bc_val_loss={_bc_best_val_loss:.4f}, epoch {epoch + 1})")
                    if _bc_early_stop_enabled:
                        _bc_patience_ctr = 0
                elif _bc_early_stop_enabled:
                    _bc_patience_ctr += 1
                    if _bc_patience_ctr >= self._bc_pretrain_early_stop_patience:
                        log.info(
                            f"  [BC pretrain] early stop at epoch {epoch + 1} "
                            f"(val stagnant for {self._bc_pretrain_early_stop_patience} epochs, "
                            f"best={_bc_best_val_loss:.4f})"
                        )
                        _bc_stopped_early = True
                        break
        # Restore the best-val weights before Phase 1's own final checkpoint
        # below, regardless of whether early stop actually triggered --
        # otherwise a full (non-early-stopped) run's final on-disk state is
        # just whatever the LAST epoch happened to leave (which can be worse
        # than an earlier epoch, e.g. from overfitting), silently overwriting
        # every "new best" checkpoint saved above with something worse.
        if _bc_best_state is not None:
            _load_state_dict_tolerant(self.decision_net, _bc_best_state["decision_net"], "decision_net")
            self.execution_net.load_state_dict(_bc_best_state["execution_net"])
            log.info(
                f"  [BC pretrain] restored best-val weights (bc_val_loss={_bc_best_val_loss:.4f})"
                + ("" if _bc_stopped_early else " before final Phase 1 checkpoint")
            )
        if bc_losses:
            log.info(f"BC pre-training done ({n_epochs} epoch(s), final bc_loss={np.mean(bc_losses):.4f})")
        else:
            log.info(f"BC pre-training done ({n_epochs} epoch(s) -- no BC epochs ran, dataset/pretrain skipped)")

        _save_pretrain_checkpoint("Phase 1")

        # --- Eval: BC-pretrained policy, before DAgger ---
        # Runs regardless of whether DAgger itself is enabled below, so a
        # Phase-1-only run still gets a "how good is the network right now"
        # readout right after BC pretraining, not just loss curves.
        log.info("Evaluating BC-pretrained policy (before DAgger)...")
        self._eval_vs_rules(env.max_episode_s)

        # --- DAgger (ai/ppo/dagger.py): alternate full-dataset BC training
        # with closed-loop rollout-and-aggregate, correcting exposure bias
        # in the BC-pretrained policy before value pre-training / Phase 4
        # repair run on top of it. Opt-in (bc.dagger_iterations, default 0). ---
        if self._dagger_iterations > 0:
            if phase_id != 1:
                log.warning(
                    f"bc.dagger_iterations={self._dagger_iterations} but phase_id={phase_id} "
                    f"(!= 1) -- DAgger's opponent-roll/phase1_labels() machinery is Phase-1-"
                    f"specific, skipping."
                )
            elif self._bc_train_value_only:
                log.info("DAgger skipped (bc_train_value_only=True) -- policy was never trained in Phase 1.")
            else:
                from footballcoach.ai.ppo.dagger import run_dagger_phase
                log.info(
                    f"DAgger: {self._dagger_iterations} iteration(s), "
                    f"{self._dagger_bc_epochs_per_iter} full-dataset BC epoch(s)/iteration, "
                    f"buffer cap {self._dagger_buffer_max_size}"
                )
                run_dagger_phase(
                    self, dataset, _bc_train_idx, bc_opt,
                    iterations=self._dagger_iterations,
                    bc_epochs_per_iter=self._dagger_bc_epochs_per_iter,
                    buffer_max_size=self._dagger_buffer_max_size,
                    max_replay_steps=self._dagger_max_replay_steps,
                    episodes_per_iteration=self._dagger_episodes_per_iteration,
                    n_workers=self._dagger_n_workers,
                    batch_size=batch_size,
                    checkpoint_fn=_save_pretrain_checkpoint,
                )
                _save_pretrain_checkpoint("DAgger")
                log.info("Evaluating policy after DAgger...")
                self._eval_vs_rules(env.max_episode_s)

        # --- Phase 2/3: collect on-policy rollout + value head warm-up ---
        # Delegates to pretrain_value(), which collects rollout_steps of
        # experience with the BC-warmed policy, computes GAE returns, applies
        # augmentation, and fits the value heads for value_epochs (with trunk
        # freezing retained — a DIFFERENT freezing decision than Phase 0 above,
        # see pretrain_value()'s docstring). This used to be duplicated inline
        # here; extracted so pretrain_value() remains the single source of
        # truth and is also usable standalone.
        self.pretrain_value(
            env,
            n_steps=rollout_steps,
            n_epochs=max(1, value_epochs),
            lr=value_lr,
            batch_size=batch_size,
            experiment_separate_value_net=experiment_separate_value_net,
            phase_id=phase_id,
        )

        # --- BC degradation check: re-evaluate BC loss over dataset after value warm-up ---
        # Meaningless (and bc_losses is empty, so the "before" mean would be
        # NaN over an empty list) when bc_train_value_only=True -- the policy
        # was never trained in this call, so it cannot have degraded.
        if self._bc_train_value_only:
            log.info("BC degradation check skipped (bc_train_value_only=True) -- policy was never trained.")
        else:
            self.decision_net.eval()
            self.execution_net.eval()
            post_bc_losses = []
            with torch.no_grad():
                for obs_dict, bc_labels in dataset.iterate_minibatches(
                    batch_size=batch_size, shuffle=False, device=self.device, valid_only=True
                ):
                    _sat, _oat = _ai_types(obs_dict)
                    d_check = self.decision_net(
                        obs_dict["self_feat"], obs_dict["other_feat"],
                        obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                        _sat, _oat,
                    )
                    e_check = self.execution_net(
                        obs_dict["self_feat"], obs_dict["other_feat"],
                        obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                        d_check, _sat, _oat,
                    )
                    bc_labels = canonicalize_bc_labels(bc_labels, x_sign_of(obs_dict["self_feat"]))
                    post_bc_losses.append(bc_loss_from_tensor(
                        bc_labels, d_check, e_check,
                        pos_weight_kick=self._bc_pos_weight_kick,
                        pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                        dec_weight=self._bc_dec_weight,
                        exec_weight=self._bc_exec_weight,
                        dec_label_smoothing=self._bc_dec_label_smoothing,
                        exec_label_smoothing=self._bc_exec_label_smoothing,
                    ).item())
            self.decision_net.train()
            self.execution_net.train()
            post_bc_loss = float(np.mean(post_bc_losses))
            bc_loss_before_value = float(np.mean(bc_losses))
            delta = post_bc_loss - bc_loss_before_value
            degraded = delta > 0.05
            log.info(
                f"BC check after value warm-up: bc_loss={post_bc_loss:.4f} "
                f"(before={bc_loss_before_value:.4f}, delta={delta:+.4f})"
                + ("  *** WARNING: significant BC degradation!" if degraded else "  OK")
            )

        _save_pretrain_checkpoint("Phase 2/3")

        # --- Phase 4: joint BC repair epoch (all params, bc_lr) ---
        # Runs the BC dataset once more with all parameters trainable to restore
        # any policy quality lost during value-only warm-up, while also keeping
        # the freshly-warmed value head in the training graph.
        repair_epochs = 0 if self._bc_train_value_only else int(self._bc_cfg.get("bc_repair_epochs", 1))
        if self._bc_train_value_only:
            log.info("  BC repair epochs skipped (bc_train_value_only=True) -- policy was never trained, nothing to repair.")
        if repair_epochs > 0:
            _repair_lr = repair_lr if repair_lr is not None else bc_lr
            repair_opt = torch.optim.Adam(
                list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                lr=_repair_lr, eps=1e-5,
            )
            for epoch in range(repair_epochs):
                repair_losses = []
                dir_cosines_r: list[float] = []
                kick_dir_cosines_r: list[float] = []
                move_probs_r: list[float] = []
                sprint_probs_r: list[float] = []
                kick_probs_r: list[float] = []
                tackle_attempt_probs_r: list[float] = []
                _kick_tp_r = _kick_fp_r = _kick_fn_r = 0.0
                _tackle_tp_r = _tackle_fp_r = _tackle_fn_r = 0.0
                _bkdn_r_acc: dict[str, float] = {}
                _bkdn_r_floor_acc: dict[str, float] = {}
                _bkdn_r_n: int = 0
                _p4_progress = ProgressReporter(
                    len(dataset.valid_indices()), prefix=f"  Phase 4 (repair) epoch {epoch + 1}/{repair_epochs}: ",
                )
                _p4_rows_done = 0
                for obs_dict, bc_labels in dataset.iterate_minibatches(
                    batch_size=batch_size, shuffle=True, device=self.device, valid_only=True
                ):
                    _p4_rows_done += bc_labels.shape[0]  # pre-augmentation row count
                    # Augment repair minibatch (ALWAYS applied).
                    if self.augment_n_slot_shuffles > 0:
                        obs_dict, bc_labels = augment_obs_bc(
                            obs_dict, bc_labels, self.augment_n_slot_shuffles, self._aug_rng
                        )
                    _sat, _oat = _ai_types(obs_dict)
                    d_r = self.decision_net(
                        obs_dict["self_feat"], obs_dict["other_feat"],
                        obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                        _sat, _oat,
                    )
                    e_r = self.execution_net(
                        obs_dict["self_feat"], obs_dict["other_feat"],
                        obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                        d_r, _sat, _oat,
                    )
                    bc_labels = canonicalize_bc_labels(bc_labels, x_sign_of(obs_dict["self_feat"]))
                    loss_r, bkdn_r = bc_loss_from_tensor(
                        bc_labels, d_r, e_r,
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
                    loss_r = loss_r + direction_magnitude_reg(e_r, self._bc_dir_mag_reg_coef)
                    repair_opt.zero_grad()
                    loss_r.backward()
                    if self._bc_max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(
                            list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                            self._bc_max_grad_norm,
                        )
                    repair_opt.step()
                    repair_losses.append(loss_r.item())
                    for k, v in bkdn_r.items():
                        _bkdn_r_acc[k] = _bkdn_r_acc.get(k, 0.0) + v
                    _bkdn_r_floor = compute_bc_loss_floor_components(
                        bc_labels,
                        pos_weight_kick=self._bc_pos_weight_kick,
                        pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                        dec_label_smoothing=self._bc_dec_label_smoothing,
                        exec_label_smoothing=self._bc_exec_label_smoothing,
                        has_exec=True,
                    )
                    for k, v in _bkdn_r_floor.items():
                        _bkdn_r_floor_acc[k] = _bkdn_r_floor_acc.get(k, 0.0) + v
                    _bkdn_r_n += 1

                    with torch.no_grad():
                        valid_mask = bc_labels[:, 14] > 0.5  # _I_VALID (not -1 = _I_OPPONENT_AI_TYPE)
                        has_dir = (bc_labels[:, 7].abs() + bc_labels[:, 8].abs()) > 1e-6
                        sel = valid_mask & has_dir
                        if sel.any():
                            pred_dir = e_r.move_direction[sel]
                            tgt_dir = bc_labels[sel, 7:9]
                            eps = 1e-6
                            pred_n = pred_dir / (pred_dir.norm(dim=-1, keepdim=True) + eps)
                            dir_cosines_r.append((pred_n * tgt_dir).sum(dim=-1).mean().item())
                        kicked_mask_r = valid_mask & (bc_labels[:, 12] > 0.5)  # _I_KICK_THIS_TICK
                        has_kick_dir_r = (bc_labels[:, 18].abs() + bc_labels[:, 19].abs() + bc_labels[:, 24].abs()) > 1e-6
                        ksel_r = kicked_mask_r & has_kick_dir_r
                        if ksel_r.any():
                            pred_kdir_r = e_r.kick_direction[ksel_r]
                            tgt_kdir_r = torch.stack([bc_labels[ksel_r, 18], bc_labels[ksel_r, 19], bc_labels[ksel_r, 24]], dim=-1)
                            eps = 1e-6
                            pred_kn_r = pred_kdir_r / (pred_kdir_r.norm(dim=-1, keepdim=True) + eps)
                            kick_dir_cosines_r.append((pred_kn_r * tgt_kdir_r).sum(dim=-1).mean().item())
                        if valid_mask.any():
                            move_probs_r.append(torch.sigmoid(e_r.exec_move_logit.squeeze(-1)[valid_mask]).mean().item())
                            sprint_probs_r.append(torch.sigmoid(e_r.sprint_logit.squeeze(-1)[valid_mask]).mean().item())
                            kick_probs_r.append(torch.sigmoid(e_r.kick_logit.squeeze(-1)[valid_mask]).mean().item())
                            tackle_attempt_probs_r.append(torch.sigmoid(e_r.tackle_attempt_logit.squeeze(-1)[valid_mask]).mean().item())
                            _tp, _fp, _fn = _binary_confusion_counts(e_r.kick_logit, bc_labels[:, 12], valid_mask)
                            _kick_tp_r += _tp; _kick_fp_r += _fp; _kick_fn_r += _fn
                            _tp, _fp, _fn = _binary_confusion_counts(e_r.tackle_attempt_logit, bc_labels[:, 13], valid_mask)
                            _tackle_tp_r += _tp; _tackle_fp_r += _fp; _tackle_fn_r += _fn
                    _p4_progress.update(_p4_rows_done, postfix=f"loss={np.mean(repair_losses):.4f}")
                # Force the final 100% render -- see the identical comment on
                # _p0_progress above. Phase 4 doesn't downsample today, but
                # this keeps all three phases' bars consistent/robust if that
                # ever changes.
                _p4_progress.update(_p4_progress.total, postfix=f"loss={np.mean(repair_losses):.4f}")

                mean_cos_r = float(np.mean(dir_cosines_r)) if dir_cosines_r else float('nan')
                mean_kick_cos_r = float(np.mean(kick_dir_cosines_r)) if kick_dir_cosines_r else float('nan')
                mean_mv_r = float(np.mean(move_probs_r)) if move_probs_r else float('nan')
                mean_spr_r = float(np.mean(sprint_probs_r)) if sprint_probs_r else float('nan')
                mean_kk_r = float(np.mean(kick_probs_r)) if kick_probs_r else float('nan')
                mean_tk_r = float(np.mean(tackle_attempt_probs_r)) if tackle_attempt_probs_r else float('nan')
                kk_prec_r, kk_rec_r, kk_f1_r = _precision_recall_f1(_kick_tp_r, _kick_fp_r, _kick_fn_r)
                tk_prec_r, tk_rec_r, tk_f1_r = _precision_recall_f1(_tackle_tp_r, _tackle_fp_r, _tackle_fn_r)
                # Also measure value loss on the stored rollout returns (no gradient)
                with torch.no_grad():
                    val_losses_r = []
                    for start_r in range(0, n_rollout, batch_size):
                        mb_obs_r = {k.replace("obs/", ""): rollout_batch[k][start_r:start_r+batch_size].to(self.device)
                                    for k in rollout_batch if k.startswith("obs/")}
                        mb_ret_r = returns_t[start_r:start_r+batch_size]
                        _sat_r, _oat_r = _ai_types(mb_obs_r)
                        d_vr = self.decision_net(
                            mb_obs_r["self_feat"], mb_obs_r["other_feat"],
                            mb_obs_r["exists_mask"], mb_obs_r["ball_feat"], mb_obs_r["global_feat"],
                            _sat_r, _oat_r,
                        )
                        _val_net_r = self.value_net if self.separate_value_net else self.execution_net
                        _value_vr = _val_net_r(
                            mb_obs_r["self_feat"], mb_obs_r["other_feat"],
                            mb_obs_r["exists_mask"], mb_obs_r["ball_feat"], mb_obs_r["global_feat"],
                            d_vr, _sat_r, _oat_r, value_only=True,
                        )
                        pred_vr = _value_vr.squeeze(-1)  # single critic (execution_net, or self.value_net)
                        val_losses_r.append(F.mse_loss(pred_vr, mb_ret_r).item() / (ret_std ** 2).item())
                _kick_bkdn_keys = {"kick", "kick_direction", "kick_power", "kick_spin"}
                # Floor-adjusted, same rationale as the main BC epoch loop above —
                # see compute_bc_loss_floor_components.
                bkdn_r_str = "  ".join(
                    f"{k}={max(0.0, v/_bkdn_r_n - _bkdn_r_floor_acc.get(k, 0.0)/_bkdn_r_n):.5f}"
                    if k in _kick_bkdn_keys else
                    f"{k}={max(0.0, v/_bkdn_r_n - _bkdn_r_floor_acc.get(k, 0.0)/_bkdn_r_n):.3f}"
                    for k, v in _bkdn_r_acc.items()
                ) if _bkdn_r_n else ""
                # Tabulated multi-line epoch summary (readability refactor only — every
                # field from the old single-line format is preserved, just grouped, with
                # full-word labels and long lines wrapped to avoid overflow. See
                # agent_plans/bc_execution_label_boundary_and_followups.md Part 4.
                _bc_r_lines = [
                    f"  BC repair epoch {epoch + 1}/{repair_epochs}",
                    f"    loss       bc={np.mean(repair_losses):.4f}  val={np.mean(val_losses_r):.4f}",
                    f"    heads      dir_cos={mean_cos_r:.3f}  kick_dir_cos={mean_kick_cos_r:.3f}",
                    f"               move_prob={mean_mv_r:.3f}  sprint_prob={mean_spr_r:.3f}  "
                    f"kick_prob={mean_kk_r:.3f}  tackle_prob={mean_tk_r:.3f}",
                    f"    pr/rec     kick:   p={kk_prec_r:.3f}  r={kk_rec_r:.3f}  f1={kk_f1_r:.3f}  "
                    f"(tp={_kick_tp_r:.0f} fp={_kick_fp_r:.0f} fn={_kick_fn_r:.0f})",
                    f"               tackle: p={tk_prec_r:.3f}  r={tk_rec_r:.3f}  f1={tk_f1_r:.3f}  "
                    f"(tp={_tackle_tp_r:.0f} fp={_tackle_fp_r:.0f} fn={_tackle_fn_r:.0f})",
                ]
                if bkdn_r_str:
                    _bkdn_r_parts = bkdn_r_str.split("  ")
                    _mid_r = (len(_bkdn_r_parts) + 1) // 2
                    _bc_r_lines.append(f"    breakdown (floor-adj)  {'  '.join(_bkdn_r_parts[:_mid_r])}")
                    if _bkdn_r_parts[_mid_r:]:
                        _bc_r_lines.append(f"                           {'  '.join(_bkdn_r_parts[_mid_r:])}")
                log.info("\n".join(_bc_r_lines))
            log.info(
                f"BC repair done ({repair_epochs} epoch(s), final bc_loss={np.mean(repair_losses):.4f}  "
                f"val_loss={np.mean(val_losses_r):.4f})\n"
                f"  final heads  dir_cos={mean_cos_r:.3f}  kick_dir_cos={mean_kick_cos_r:.3f}  "
                f"move_prob={mean_mv_r:.3f}  sprint_prob={mean_spr_r:.3f}\n"
                f"               kick_prob={mean_kk_r:.3f}  tackle_prob={mean_tk_r:.3f}"
            )

        _save_pretrain_checkpoint("Phase 4")

        log.info("Combined pre-training complete.")

    def _spawn_value_pretrain_workers(self, phase_id: Optional[int]) -> Optional["_ValuePretrainWorkers"]:
        """Spawn the parallel rollout workers for the value-pretrain
        rollout, or return None when parallel collection doesn't apply
        (``ppo.value_pretrain_n_processes <= 1`` or no ``phase_id`` -- the
        single-process branch of ``_collect_value_pretrain_rollout`` then
        steps ``env`` directly).

        ``ppo.value_pretrain_batched_rollout`` (default off) selects the
        worker kind: True = ``batched_rollout_worker.py`` workers, each
        owning ``ppo.value_pretrain_envs_per_process`` envs and batching all
        of their decisions into one network call per round (the same
        mechanism the main PPO loop uses; ``batch_secondary_players`` is
        reused from the main loop's setting so self-play opponents get
        batched too); False = today's plain one-env-per-process
        ``rollout_worker.py`` workers. Both share one ``ctx.Value`` step
        counter for a single aggregate progress bar. Batched workers stream
        WHOLE-EPISODE chunks every ``ppo.value_pretrain_chunk_steps`` rows
        (0 = one send at the end): a flush only ever sends completed
        episodes and keeps each env's unfinished tail, so MC returns stay
        exact and the train/val episode split stays clean while per-process
        memory stays bounded to one chunk -- see ai/knowledge.md "PPG-style
        value refit" and batched_rollout_worker.py's "Chunked streaming".

        Callers own the lifecycle: pair with ``_close_value_pretrain_workers``.
        """
        if not (self.value_pretrain_n_processes > 1 and phase_id is not None):
            return None
        import multiprocessing

        n_workers = self.value_pretrain_n_processes
        base_seed = random.randint(0, 2**31 - 1)
        # Must be created with the SAME "spawn" context the spawn functions
        # use, and passed at process-creation time (a raw ctx.Value can't be
        # sent to an already-running worker afterwards).
        ctx = multiprocessing.get_context("spawn")
        progress_value = ctx.Value("l", 0)
        if self.value_pretrain_batched_rollout:
            from footballcoach.ai.ppo.batched_rollout_worker import spawn_batched_workers

            envs_per_process = max(1, self.value_pretrain_envs_per_process)
            handles = spawn_batched_workers(
                phase_id, n_workers, envs_per_process, base_seed,
                self.separate_value_net, self.worker_torch_threads,
                batch_secondary_players=self.batch_secondary_players,
                chunk_steps=self.value_pretrain_chunk_steps or None, progress_value=progress_value,
            )
            return _ValuePretrainWorkers(handles, True, progress_value, n_workers, envs_per_process)
        from footballcoach.ai.ppo.rollout_worker import spawn_workers

        handles = spawn_workers(
            phase_id, n_workers, base_seed, self.separate_value_net, self.worker_torch_threads,
            progress_value=progress_value,
        )
        return _ValuePretrainWorkers(handles, False, progress_value, n_workers, 1)

    def _close_value_pretrain_workers(self, pool: Optional["_ValuePretrainWorkers"]) -> None:
        if pool is None:
            return
        if pool.batched:
            from footballcoach.ai.ppo.batched_rollout_worker import close_batched_workers

            close_batched_workers(pool.handles)
        else:
            from footballcoach.ai.ppo.rollout_worker import close_workers

            close_workers(pool.handles)

    def _collect_value_pretrain_rollout(
        self, env, n_steps: int, phase_id: Optional[int], use_gae: bool = False,
        pool: Optional["_ValuePretrainWorkers"] = None,
    ) -> tuple[dict, dict]:
        """Collect ``n_steps`` of on-policy experience for value warm-up.

        Returns ``(batch, stats)`` -- ``batch`` is the GAE-processed dict (same
        shape as ``RolloutBuffer.as_tensors()``); ``stats`` has
        ``episode_returns``/``outcomes_vs_rules``/``outcomes_vs_immobile``/
        ``outcomes_vs_neural`` for the caller's return value.

        Single-process when ``ppo.value_pretrain_n_processes == 1`` (uses
        ``env`` directly, exactly the previous inline behaviour). When
        ``ppo.value_pretrain_n_processes > 1`` and ``phase_id`` is given,
        spawns that many plain one-env-per-process ``rollout_worker.py``
        workers (its own process count, DECOUPLED from the main PPO loop's
        ``ppo.n_processes``/``ppo.batched_rollout`` -- this call has no
        batched-rollout-aware path of its own, see ``value_pretrain_n_processes``'s
        own config comment for why sharing ``n_processes`` directly would
        be a regression here) -- no weight sync needed since this is called
        once per pretraining stage, not per rollout; each worker's GAE is
        computed independently before merging, for the same reason as
        ``_train_parallel()`` (concatenating raw transitions across worker
        boundaries before GAE would corrupt advantage estimates).

        Args:
            use_gae: False (default, ``pretrain_value``'s behaviour, UNCHANGED)
                uses pure Monte Carlo discounted returns
                (``compute_mc_returns``) -- correct for a BC-fresh/untrained
                value net, where bootstrapping off its own (effectively
                random) predictions would fit targets that circularly depend
                on the very net being warm-started. True (``ppg_value_refit``)
                uses ``compute_gae`` instead: for an ALREADY-reasonably-
                trained value net (ppg_value_refit's actual use case -- a
                decent checkpoint, not a cold start), bootstrapping off its
                own predictions is just ordinary TD learning, exactly what
                real PPO training already does on every rollout -- and
                matters here because real PPO's OWN value loss targets GAE
                returns, not MC returns, so fitting the same quantity during
                a refit means the value head needs no further readjustment
                once real training resumes (the actual point of PPG's
                auxiliary phase: converge toward the SAME target, not a
                different-but-correlated one). Bootstrap handling lives in
                ``_finalize_value_pretrain_result``: parallel workers (both
                kinds) return a real ``last_value``, used as-is with NO
                truncation so a trailing partial episode's data is kept;
                the single-process branch has no ``last_value`` and falls
                back to truncate-then-GAE with a literal ``0.0`` bootstrap,
                which is provably never used (truncation leaves the buffer
                ending on a ``done=1`` row and GAE's recursion resets there).
            pool: an already-spawned worker pool (from
                ``_spawn_value_pretrain_workers``) to REUSE for this call
                instead of spawning a fresh one -- the worker main loops
                (both kinds) already serve repeated "collect" commands (only
                "close"/the pipe closing ends them), so nothing on the
                worker side needs to change. When given, this call does NOT
                spawn or close any process; the CALLER owns that lifecycle
                (``ppg_value_refit`` does this for ``num_rollouts`` > 1, so
                a fresh process pool isn't paid for every cycle). Weights
                are re-synced to the workers at the start of EVERY call
                regardless, which is what makes reuse correct while the
                policy keeps moving. ``None`` (default) = spawn a pool for
                this call alone and close it before returning -- exactly
                the original per-call behaviour (``pretrain_value``,
                ``pretrain_combined``).
        """
        _owns_pool = pool is None
        if _owns_pool:
            pool = self._spawn_value_pretrain_workers(phase_id)
        if pool is not None:
            import multiprocessing
            import multiprocessing.connection

            n_workers = pool.n_processes
            steps_per_worker = max(1, n_steps // n_workers)
            _kind = (
                f"batched: {n_workers} proc x {pool.envs_per_process} envs = "
                f"{n_workers * pool.envs_per_process} envs"
                if pool.batched else f"{n_workers} worker(s), 1 env each"
            )
            log.info(
                f"  [value pretrain rollout] parallel collection ({_kind}), "
                f"~{steps_per_worker} steps/process"
                + ("" if _owns_pool else " (reusing already-spawned workers)")
            )
            _wall_start = time.perf_counter()

            # Results are INGESTED AS THEY ARRIVE (each one is finalized,
            # its stats folded in, and the raw result dropped) rather than
            # collected into a list first -- the old "gather everything, then
            # convert everything, then merge" shape kept the raw results, the
            # converted per-worker tensors and the merged batch alive
            # together (the rollout-end RAM spike). Batched workers now also
            # finalize in the worker and stream whole-episode chunks, so what
            # arrives here is already compact and bounded per message.
            worker_batches: list[dict] = []
            episode_returns: list[float] = []
            episode_outcome_labels: list[str] = []
            outcomes_vs_rules: list[str] = []
            outcomes_vs_immobile: list[str] = []
            outcomes_vs_neural: list[str] = []
            episode_comp_list: list[dict[str, float]] = []
            episode_durations_s: list[float] = []
            n_dropped_total = 0
            _n_rows_total = 0

            def _ingest(r: dict) -> None:
                nonlocal n_dropped_total, _n_rows_total
                if "batch" in r:
                    # Worker-finalized (batched workers): numpy arrays on the wire
                    # (None = MC mode had no completed episode in this result).
                    from footballcoach.ai.ppo.batched_rollout_worker import batch_from_wire

                    tensors = batch_from_wire(r["batch"]) if r["batch"] is not None else None
                    n_dropped = r["n_dropped"]
                else:
                    # Legacy shape ({"buffer", ...}): plain workers, or a batched
                    # worker spoken to without a returns spec.
                    tensors, n_dropped = _finalize_value_pretrain_result(r, self.gamma, self.lam, use_gae)
                n_dropped_total += n_dropped
                _n_rows_total += (int(tensors["returns"].shape[0]) if tensors is not None else 0) + n_dropped
                if tensors is not None:
                    worker_batches.append(tensors)
                stats = r["stats"]
                episode_returns.extend(stats["episode_rewards"])
                episode_outcome_labels.extend(stats["episode_outcome_labels"])
                outcomes_vs_rules.extend(stats["episode_outcomes_vs_rules"])
                outcomes_vs_immobile.extend(stats["episode_outcomes_vs_immobile"])
                outcomes_vs_neural.extend(stats["episode_outcomes_vs_neural"])
                episode_comp_list.extend(stats["episode_comp_list"])
                episode_durations_s.extend(stats["episode_durations_s"])

            try:
                dec_state = self.decision_net.state_dict()
                exec_state = self.execution_net.state_dict()
                val_state = self.value_net.state_dict() if self.value_net is not None else None
                for w in pool.handles:
                    w.set_weights(dec_state, exec_state, val_state)
                # The PARENT resets the shared counter (workers never do --
                # a worker-side reset would race between workers sharing it).
                with pool.progress_value.get_lock():
                    pool.progress_value.value = 0
                # Batched workers finalize their own results with OUR gamma/lam
                # (explicit, so a worker's separately-read config can't drift).
                _returns_spec = {
                    "mode": "gae" if use_gae else "mc", "gamma": self.gamma, "lam": self.lam,
                }
                for w in pool.handles:
                    if pool.batched:
                        w.collect(steps_per_worker, returns=_returns_spec)
                    else:
                        w.collect(steps_per_worker, progress=0.0)
                _agg_progress = ProgressReporter(
                    steps_per_worker * n_workers,
                    prefix=f"  [value pretrain rollout] ({n_workers} workers): ", live=True,
                )
                _pending = {w.conn: w for w in pool.handles}
                while _pending:
                    ready = multiprocessing.connection.wait(list(_pending.keys()), timeout=0.2)
                    _agg_progress.update(int(pool.progress_value.value))
                    for conn in ready:
                        if pool.batched:
                            # Uniform batched protocol: zero-or-more
                            # {"chunk": [per-env results]} then {"done": True}.
                            msg = conn.recv()
                            for r in msg.get("chunk", []):
                                _ingest(r)
                            if msg.get("done"):
                                _pending.pop(conn, None)
                        else:
                            _ingest(_pending.pop(conn).recv_result())
            finally:
                if _owns_pool:
                    self._close_value_pretrain_workers(pool)
            _wall_elapsed = time.perf_counter() - _wall_start
            log.info(
                f"  [value pretrain rollout] parallel total: {_wall_elapsed:.1f}s wall  "
                f"({1000.0 * _wall_elapsed / max(_n_rows_total, 1):.2f} ms/step aggregate, "
                f"{_n_rows_total / max(_wall_elapsed, 1e-9):.1f} steps/s aggregate across "
                f"{n_workers} process(es)"
                + (", includes process spawn)" if _owns_pool else ")")
            )
            if n_dropped_total:
                log.info(
                    f"  [value pretrain rollout] dropped {n_dropped_total} trailing "
                    f"(incomplete-episode) step(s) across workers before "
                    f"{'GAE' if use_gae else 'MC'}-return fit"
                )
            if not worker_batches:
                raise RuntimeError(
                    "value-pretrain rollout produced no usable rows (no episode completed in "
                    f"{n_steps} steps across {n_workers} process(es)); raise n_steps or shorten episodes"
                )
            # release_inputs: peak ~1x the rollout during the merge, not ~2x.
            batch = _merge_worker_batches(worker_batches, release_inputs=True)
            worker_batches.clear()
        else:
            env.sample_action_fn = self._sample_action
            env.reset()
            buffer = RolloutBuffer()
            episode_returns = []
            episode_outcome_labels = []
            outcomes_vs_rules = []
            outcomes_vs_immobile = []
            outcomes_vs_neural = []
            episode_accum = 0.0
            next_obs = None
            episode_comp_accum: dict[str, float] = {}
            episode_comp_list = []
            episode_durations_s = []
            progress = ProgressReporter(n_steps, prefix="  [value pretrain rollout] ", live=True)

            for _step_i in range(n_steps):
                progress.update(_step_i + 1)
                next_obs, reward, done, info = env.step()
                tr = env.last_trainee_transition
                if tr is not None:
                    buffer.add(
                        obs=tr["obs"],
                        action=_action_to_numpy(tr["action"], tr["raw_exec"]),
                        log_prob=tr["log_prob"],
                        value=tr["value"],
                        reward=reward,
                        done=1.0 if done else 0.0,
                    )
                episode_accum += reward
                for _k, _v in getattr(env, "last_reward_components", {}).items():
                    episode_comp_accum[_k] = episode_comp_accum.get(_k, 0.0) + _v
                if done:
                    episode_returns.append(episode_accum)
                    episode_accum = 0.0
                    episode_outcome_labels.append(
                        info.trial_outcome if (info is not None and info.trial_outcome is not None) else "unknown"
                    )
                    if episode_comp_accum:
                        episode_comp_list.append(dict(episode_comp_accum))
                    episode_comp_accum = {}
                    if info is not None and info.trial_outcome is not None:
                        if info.is_rules_episode:
                            outcomes_vs_rules.append(info.trial_outcome)
                        elif info.is_immobile_episode:
                            outcomes_vs_immobile.append(info.trial_outcome)
                        else:
                            outcomes_vs_neural.append(info.trial_outcome)
                    if info is not None:
                        episode_durations_s.append(info.ticks_elapsed * env._dt_s)
                    env.reset()
                else:
                    obs = next_obs
            progress.finish(n_steps, n_episodes=len(episode_outcome_labels))

            # No "last_value" in this bare result dict -> the helper takes its
            # truncate-then-(MC | zero-bootstrap GAE) path, same as before.
            batch, n_dropped = _finalize_value_pretrain_result(
                {"buffer": buffer}, self.gamma, self.lam, use_gae,
            )
            if n_dropped:
                log.info(
                    f"  [value pretrain rollout] dropped {n_dropped} trailing "
                    f"(incomplete-episode) step(s) before {'GAE' if use_gae else 'MC'}-return fit"
                )

        log.info(
            f"  [value pretrain rollout] mean_return={np.mean(episode_returns) if episode_returns else float('nan'):.2f} "
            f"({len(episode_returns)} episode(s))  "
            f"vs[{_PHASE1_OUTCOME_LEGEND}]  "
            f"vs_rules({len(outcomes_vs_rules)}): {outcome_breakdown(outcomes_vs_rules)}  "
            f"vs_immobile({len(outcomes_vs_immobile)}): {outcome_breakdown(outcomes_vs_immobile)}  "
            f"vs_neural({len(outcomes_vs_neural)}): {outcome_breakdown(outcomes_vs_neural)}"
        )
        if episode_durations_s:
            _dur_arr = np.array(episode_durations_s)
            log.info(
                f"  [value pretrain rollout] ep_len {_dur_arr.mean():.1f}\u00b1{_dur_arr.std():.1f}s"
                f"  (n={len(episode_durations_s)}, min={_dur_arr.min():.1f}s, max={_dur_arr.max():.1f}s)"
            )
        log_episode_reward_stats_by_outcome(
            episode_returns, episode_outcome_labels, log_prefix="[value pretrain rollout] "
        )

        # Per-episode reward statistics table (mean/std/min/max per type),
        # identical formatting to the main PPO rollout loop's table.
        if episode_comp_list:
            _all_keys = [k for k, _ in REWARD_COMP_LABELS
                         if any(k in ep for ep in episode_comp_list)]
            if _all_keys:
                _col_w = 14
                _lbl_map = {k: lbl for k, lbl in REWARD_COMP_LABELS}
                _rew_stats_lines = [
                    f"  {'component':<{_col_w}}  {'mean':>8}  {'std':>7}  {'min':>8}  {'max':>8}",
                    "  " + "-" * _col_w + "  " + "-" * 8 + "  " + "-" * 7 + "  " + "-" * 8 + "  " + "-" * 8,
                ]
                for _k in _all_keys:
                    _vals = [ep[_k] for ep in episode_comp_list if _k in ep]
                    if not _vals:
                        continue
                    _arr = np.array(_vals)
                    # Same "never fired / always exactly zero this rollout"
                    # skip as the "rew/step" table's own "_cs['mean'] == 0.0
                    # and _cs['std'] == 0.0" check -- mean==0 and std==0
                    # together mean every episode's value for this
                    # component was exactly 0.0, so there's nothing to show.
                    # Component identity/whether it's config-disabled can
                    # change between rollouts (a coefficient retuned in
                    # ai_config.json, a curriculum phase change), so this is
                    # a per-rollout runtime check, not a permanent removal
                    # from REWARD_COMP_LABELS -- a component that starts
                    # firing again just reappears on its own next rollout,
                    # nothing to keep in sync by hand.
                    if _arr.mean() == 0.0 and _arr.std() == 0.0:
                        continue
                    _lbl = _lbl_map.get(_k, _k)
                    _rew_stats_lines.append(
                        f"  {_lbl:<{_col_w}}  {_arr.mean():>+8.3f}  {_arr.std():>7.3f}"
                        f"  {_arr.min():>+8.3f}  {_arr.max():>+8.3f}"
                    )
                log.info(
                    f"  [value pretrain rollout] rew/ep (mean/std/min/max per episode, "
                    f"{len(episode_comp_list)} ep)\n" + "\n".join(_rew_stats_lines)
                )

        return batch, {
            "episode_returns": episode_returns,
            "outcomes_vs_rules": outcomes_vs_rules,
            "outcomes_vs_immobile": outcomes_vs_immobile,
            "outcomes_vs_neural": outcomes_vs_neural,
        }

    def pretrain_value(
        self,
        env,
        n_steps: int,
        n_epochs: int,
        lr: float,
        batch_size: Optional[int] = None,
        experiment_separate_value_net: bool = False,
        phase_id: Optional[int] = None,
    ) -> dict:
        """Warm-start the value heads to predict actual returns before PPO starts.

        Collects n_steps of experience using the current (BC-warm-started) policy,
        computes Monte Carlo returns via GAE, then trains ONLY the value loss for
        n_epochs at the given lr. This prevents the enormous value gradient from
        destroying the policy on the very first PPO update.

        Trunk/encoder freezing (via ``_get_value_pretrain_freeze_params()``) is
        always applied here — this is a distinct stage from
        ``pretrain_combined()``'s Phase 0, which deliberately has freezing
        REMOVED (see ai/knowledge.md "Phase 0" note). Do not conflate the two.

        Also called internally by ``pretrain_combined()``'s Phase 2/3 (rollout
        collection + value warm-up), which used to duplicate this logic inline.

        Args:
            env: ScenarioEnv. Ignored (may be None) when
                ``ppo.value_pretrain_n_processes > 1`` and ``phase_id`` is
                given -- rollout collection spawns its own
                ``rollout_worker.py`` subprocess workers instead (a process
                count decoupled from the main PPO loop's ``n_processes``,
                see that config key's comment for why).
            n_steps: Steps to collect (should be >= rollout_steps, e.g. 4096)
            n_epochs: Epochs to fit the value network per collected rollout
            lr: Learning rate for value pre-training (higher than PPO lr, e.g. 1e-3)
            batch_size: Minibatch size. Defaults to ``self.minibatch_size``.
            experiment_separate_value_net: EXPERIMENTAL (see Idea2.md /
                ai_trainer_knowledge.md "separate value network" discussion) —
                when True, also constructs a second, completely independent
                ``ExecutionNetwork`` (same class + config, fresh random init, NOT
                sharing any weights with ``self.execution_net``) and trains it
                fully unfrozen (no ``_get_value_pretrain_freeze_params()``
                freezing) on the identical rollout data/returns as the main
                shared-trunk value head. It still reads ``decision_heads`` from
                the real (frozen) ``self.decision_net``, so the comparison
                isolates "separate execution-net trunk for the value head" as
                the only variable — same input information, same architecture,
                same data, only the trunk-sharing differs. Logs a side-by-side
                val_rmse comparison each epoch. Purely a read-only experiment:
                the second network is discarded when this method returns (no
                checkpoint save, no effect on the real value_head or PPO).
            phase_id: Curriculum phase id. Required to use parallel rollout
                collection (``ppo.value_pretrain_n_processes > 1``) -- each
                worker rebuilds its own env from this id, same as the main PPO
                training loop. Ignored when ``ppo.value_pretrain_n_processes
                == 1`` (uses ``env`` directly).

        Returns:
            dict with diagnostic stats from the rollout collection:
            ``{"episode_returns": list[float], "outcomes_vs_rules": list[str],
            "outcomes_vs_immobile": list[str], "outcomes_vs_neural": list[str]}``
        """
        # Decoupled from ppo.minibatch_size -- see self._value_pretrain_batch_size.
        _batch_size = batch_size if batch_size is not None else self._value_pretrain_batch_size
        log.info(f"Value pre-training: {n_steps} steps, {n_epochs} epochs, lr={lr}, batch_size={_batch_size}")
        if self.separate_value_net:
            # self.value_net is fully independent of the BC-primed policy trunk --
            # nothing to freeze/protect, train the whole thing (this is the whole
            # point: a critic that never saw a BC gradient, free to organise its
            # own features purely for value prediction from step one).
            value_opt = self.value_net_optimizer
        else:
            # Freeze trunk layers so BC-learned policy weights are not corrupted.
            _freeze_params = self._get_value_pretrain_freeze_params()
            for p in _freeze_params:
                p.requires_grad_(False)
            value_opt = torch.optim.Adam(
                list(self.execution_net.value_head.parameters()),
                lr=lr, eps=1e-5, weight_decay=self._value_pretrain_weight_decay,
            )

        # --- Experimental: separate-trunk value network (see docstring) ---
        _sep_net = None
        _sep_opt = None
        if experiment_separate_value_net:
            from footballcoach.ai.models.execution_network import ExecutionNetwork
            _sep_net = CanonicalNetworkWrapper(ExecutionNetwork.from_config()).to(self.device)
            # Fully unfrozen: every parameter of this fresh network trains,
            # unlike the main value_head-only optimizer above.
            _sep_opt = torch.optim.Adam(_sep_net.parameters(), lr=lr, eps=1e-5)
            log.info(
                "  [separate value net experiment] constructed a second, independent "
                "ExecutionNetwork (fresh init, fully unfrozen) for side-by-side "
                "value-loss comparison against the shared-trunk value_head above."
            )

        batch, _rollout_stats = self._collect_value_pretrain_rollout(env, n_steps, phase_id)

        # --- Episode-level 85/15 train/val split (overfit detection) ---
        # Split by complete episodes so no episode spans both sets.
        dones_arr = batch["dones"].numpy()
        episode_end_idxs = np.where(dones_arr > 0.5)[0]
        n_complete_eps = len(episode_end_idxs)
        n_val_eps = max(1, round(0.15 * n_complete_eps)) if n_complete_eps >= 2 else 0
        n_train_eps = n_complete_eps - n_val_eps
        n_total = len(dones_arr)
        val_mask = np.zeros(n_total, dtype=bool)
        if n_val_eps > 0:
            ep_starts = np.concatenate([[0], episode_end_idxs[:-1] + 1])
            for _i in range(n_train_eps, n_complete_eps):
                val_mask[ep_starts[_i]:episode_end_idxs[_i] + 1] = True
        train_mask = ~val_mask

        # ret_std from all returns for consistent normalisation scale --
        # taken BEFORE the split below empties ``batch`` key by key.
        all_returns_t = batch["returns"].to(self.device)

        # Split WITHOUT keeping the full batch alive next to its two slices
        # (see _split_batch_releasing): peak ~1x the rollout, not ~2x.
        train_batch_raw, val_batch_raw = _split_batch_releasing(
            batch, train_mask, val_mask if n_val_eps > 0 else None,
        )

        log.info(
            f"  Value pretrain split: {n_train_eps} train eps ({int(train_mask.sum())} steps)"
            + (f"  |  {n_val_eps} val eps ({int(val_mask.sum())} steps)" if n_val_eps > 0 else "")
        )

        # Augment only the train portion (geometric flips + slot permutations).
        if self.augment_n_slot_shuffles > 0:
            train_batch = augment_batch(train_batch_raw, self.augment_n_slot_shuffles, self._aug_rng)
            del train_batch_raw  # the un-augmented copy is dead weight from here on
        else:
            train_batch = train_batch_raw

        ret_std = all_returns_t.std().clamp(min=1.0)
        log.debug(f"  [value pretrain] returns: mean={all_returns_t.mean():.2f}  std={ret_std:.2f}"
                  f"  min={all_returns_t.min():.2f}  max={all_returns_t.max():.2f}")

        returns_t = train_batch["returns"].to(self.device)

        # Pre-load val tensors to device once.
        val_returns_t = None
        val_obs_dict = None
        val_outcomes: list[str] = []
        if val_batch_raw is not None:
            val_returns_t = val_batch_raw["returns"].to(self.device)
            val_obs_dict = {k.replace("obs/", ""): val_batch_raw[k].to(self.device)
                            for k in val_batch_raw if k.startswith("obs/")}
            val_outcomes = val_batch_raw.get("step_outcomes", [])

        n = len(returns_t)
        mean_loss = float("nan")
        epochs_done = 0
        _best_val_loss = float("inf")
        _best_val_state: Optional[dict] = None
        _patience = 0
        _EARLY_STOP_PATIENCE = self._value_pretrain_early_stop_patience
        _EARLY_STOP_MIN_DELTA = self._value_pretrain_early_stop_min_delta

        # --- Baseline: loss BEFORE any gradient step, train/val separately ---
        # Same normalized-MSE forward pass (and the same self.value_net vs.
        # self.execution_net routing) as the per-epoch train/val blocks below,
        # just under no_grad with no optimizer step -- so "epoch 1"'s
        # improvement can be read against a real starting point instead of
        # assumed. Minibatched the same way as the training loop purely to
        # bound peak memory on a large (post-augmentation) train_batch.
        def _eval_loss_over(obs_source: dict, ret_source: torch.Tensor) -> tuple[float, float]:
            total_sq = 0.0
            n_rows = 0
            with torch.no_grad():
                for start in range(0, len(ret_source), _batch_size):
                    mb_ret = ret_source[start:start + _batch_size]
                    mb_obs = {k: v[start:start + _batch_size] for k, v in obs_source.items()}
                    sat, oat = _ai_types(mb_obs)
                    d_heads = self.decision_net(
                        mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"],
                        mb_obs["ball_feat"], mb_obs["global_feat"], sat, oat,
                    )
                    _val_net = self.value_net if self.separate_value_net else self.execution_net
                    _value = _val_net(
                        mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"],
                        mb_obs["ball_feat"], mb_obs["global_feat"], d_heads, sat, oat,
                        value_only=True,
                    )
                    preds = _value.squeeze(-1)
                    total_sq += float(((preds - mb_ret) ** 2).sum())
                    n_rows += len(mb_ret)
            mse = total_sq / max(n_rows, 1)
            norm_mse = mse / float(ret_std ** 2)
            return norm_mse, float(ret_std) * math.sqrt(norm_mse)

        _baseline_train_obs = {
            k.replace("obs/", ""): v.to(self.device)
            for k, v in train_batch.items() if k.startswith("obs/")
        }
        _baseline_train_loss, _baseline_train_rmse = _eval_loss_over(_baseline_train_obs, returns_t)
        if val_obs_dict is not None and val_returns_t is not None:
            _baseline_val_loss, _baseline_val_rmse = _eval_loss_over(val_obs_dict, val_returns_t)
            log.info(
                f"  Value epoch   0/{n_epochs} (baseline, no training yet): "
                f"train={_baseline_train_loss:.4f} rmse={_baseline_train_rmse:.2f}  "
                f"val={_baseline_val_loss:.4f} val_rmse={_baseline_val_rmse:.2f} "
                f"(std={float(ret_std):.1f})"
            )
        else:
            log.info(
                f"  Value epoch   0/{n_epochs} (baseline, no training yet): "
                f"train_loss={_baseline_train_loss:.4f}  rmse={_baseline_train_rmse:.2f} "
                f"(returns std={float(ret_std):.1f})"
            )

        ep_losses_sep: list[float] = []  # populated only when experiment_separate_value_net
        for ep in range(n_epochs):
            indices = torch.randperm(n)
            ep_losses = []
            ep_losses_sep = []
            ep_pred_means: list[float] = []
            ep_ret_means: list[float] = []
            for start in range(0, n, _batch_size):
                mb_idx = indices[start:start + _batch_size]
                mb_obs = {k.replace("obs/", ""): train_batch[k][mb_idx].to(self.device)
                          for k in train_batch if k.startswith("obs/")}
                mb_ret = returns_t[mb_idx]

                sf = mb_obs["self_feat"]
                of = mb_obs["other_feat"]
                em = mb_obs["exists_mask"]
                bf = mb_obs["ball_feat"]
                gf = mb_obs["global_feat"]
                sat, oat = _ai_types(mb_obs)

                if self.separate_value_net:
                    # decision_net stays fully frozen/detached here -- self.value_net
                    # is completely independent, no need for its gradient to reach
                    # decision_net at all (unlike the shared-trunk path below, where
                    # decision_net params are included in the optimizer/clip so its
                    # BC-pretrained-but-still-nominally-trainable params get the
                    # gradient too under the old convention).
                    with torch.no_grad():
                        d_heads = self.decision_net(sf, of, em, bf, gf, sat, oat)
                    _value = self.value_net(sf, of, em, bf, gf, d_heads, sat, oat, value_only=True)
                    new_values = _value.squeeze(-1)

                    value_loss = F.mse_loss(new_values, mb_ret) / (ret_std ** 2)

                    value_opt.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                    value_opt.step()
                    ep_losses.append(value_loss.item())
                else:
                    d_heads = self.decision_net(sf, of, em, bf, gf, sat, oat)
                    _value = self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat, value_only=True)
                    new_values = _value.squeeze(-1)  # single value head (execution_net)

                    # Normalised MSE so the loss is O(1) regardless of return scale
                    value_loss = F.mse_loss(new_values, mb_ret) / (ret_std ** 2)

                    value_opt.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                        self.max_grad_norm,
                    )
                    value_opt.step()
                    ep_losses.append(value_loss.item())

                ep_pred_means.append(new_values.detach().mean().item())
                ep_ret_means.append(mb_ret.mean().item())

                # --- Experimental separate-trunk value net: identical mb_obs/mb_ret,
                # identical decision-head VALUES (from the same frozen decision_net) —
                # only the execution-net trunk/encoders/value_head differ (fresh,
                # unfrozen, zero weight sharing with self.execution_net). d_heads is
                # detached here so this experimental path is totally independent: no
                # gradient flows back into decision_net, and it does not try to reuse
                # the main value path's autograd graph (already freed by the
                # .backward() call above).
                if _sep_net is not None:
                    d_heads_detached = _detach_decision_heads(d_heads)
                    _value_sep2 = _sep_net(sf, of, em, bf, gf, d_heads_detached, sat, oat, value_only=True)
                    new_values_sep = _value_sep2.squeeze(-1)
                    value_loss_sep = F.mse_loss(new_values_sep, mb_ret) / (ret_std ** 2)

                    _sep_opt.zero_grad()
                    value_loss_sep.backward()
                    nn.utils.clip_grad_norm_(_sep_net.parameters(), self.max_grad_norm)
                    _sep_opt.step()
                    ep_losses_sep.append(value_loss_sep.item())

            mean_loss = float(np.mean(ep_losses))
            epochs_done = ep + 1
            _train_rmse = float(ret_std) * math.sqrt(mean_loss)
            _train_pred_mean = float(np.mean(ep_pred_means)) if ep_pred_means else float("nan")
            _train_ret_mean = float(np.mean(ep_ret_means)) if ep_ret_means else float("nan")
            _mean_loss_sep = float(np.mean(ep_losses_sep)) if ep_losses_sep else float("nan")
            _train_rmse_sep = float(ret_std) * math.sqrt(_mean_loss_sep) if ep_losses_sep else float("nan")

            _vl_sep = float("nan")
            _val_rmse_sep = float("nan")
            if val_obs_dict is not None and val_returns_t is not None:
                with torch.no_grad():
                    _sat_v, _oat_v = _ai_types(val_obs_dict)
                    d_v = self.decision_net(
                        val_obs_dict["self_feat"], val_obs_dict["other_feat"],
                        val_obs_dict["exists_mask"], val_obs_dict["ball_feat"],
                        val_obs_dict["global_feat"], _sat_v, _oat_v,
                    )
                    _val_net = self.value_net if self.separate_value_net else self.execution_net
                    _value_v = _val_net(
                        val_obs_dict["self_feat"], val_obs_dict["other_feat"],
                        val_obs_dict["exists_mask"], val_obs_dict["ball_feat"],
                        val_obs_dict["global_feat"], d_v, _sat_v, _oat_v,
                        value_only=True,
                    )
                    _val_preds = _value_v.squeeze(-1)
                    _vl = float(F.mse_loss(_val_preds, val_returns_t) / (ret_std ** 2))
                    _val_pred_mean = float(_val_preds.mean().item())
                    _val_ret_mean = float(val_returns_t.mean().item())
                    if _sep_net is not None:
                        _value_v_sep = _sep_net(
                            val_obs_dict["self_feat"], val_obs_dict["other_feat"],
                            val_obs_dict["exists_mask"], val_obs_dict["ball_feat"],
                            val_obs_dict["global_feat"], d_v, _sat_v, _oat_v,
                            value_only=True,
                        )
                        _vl_sep = float(F.mse_loss(
                            _value_v_sep.squeeze(-1), val_returns_t
                        ) / (ret_std ** 2))
                        _val_rmse_sep = float(ret_std) * math.sqrt(_vl_sep)
                _val_rmse = float(ret_std) * math.sqrt(_vl)
                log.info(
                    f"  Value epoch {epochs_done}/{n_epochs}: "
                    f"train={mean_loss:.4f} rmse={_train_rmse:.2f}  "
                    f"val={_vl:.4f} val_rmse={_val_rmse:.2f} "
                    f"(std={float(ret_std):.1f})"
                    f"\n    V(train)={_train_pred_mean:+.3f}  R(train)={_train_ret_mean:+.3f}"
                    f"  |  V(val)={_val_pred_mean:+.3f}  R(val)={_val_ret_mean:+.3f}"
                    + (
                        f"\n    [separate-trunk value net] train={_mean_loss_sep:.4f} rmse={_train_rmse_sep:.2f}  "
                        f"val={_vl_sep:.4f} val_rmse={_val_rmse_sep:.2f}"
                        f"  (shared-trunk val_rmse={_val_rmse:.2f} \u2014 "
                        f"{'separate is BETTER' if _val_rmse_sep < _val_rmse else 'shared is better or equal'})"
                        if _sep_net is not None else ""
                    )
                )
                if val_outcomes:
                    _val_by_outc = value_mse_by_outcome(_val_preds, val_returns_t, val_outcomes)
                    if _val_by_outc:
                        log.info(f"    val_rmse by outcome: {format_outcome_rmse_breakdown(_val_by_outc)}")
                if _vl < _best_val_loss - _EARLY_STOP_MIN_DELTA:
                    _best_val_loss = _vl
                    _patience = 0
                    if self.separate_value_net:
                        _best_val_state = copy.deepcopy(self.value_net.state_dict())
                    else:
                        _best_val_state = copy.deepcopy(self.execution_net.value_head.state_dict())
                else:
                    _patience += 1
                    if _patience >= _EARLY_STOP_PATIENCE:
                        log.info(
                            f"  [value pretrain] early stop at epoch {epochs_done} "
                            f"(val stagnant for {_EARLY_STOP_PATIENCE} epochs, best={_best_val_loss:.4f})"
                        )
                        break
            else:
                log.info(
                    f"  Value epoch {epochs_done}/{n_epochs}: "
                    f"train_loss={mean_loss:.4f}  rmse={_train_rmse:.2f} "
                    f"(returns std={float(ret_std):.1f})"
                    f"\n    V(train)={_train_pred_mean:+.3f}  R(train)={_train_ret_mean:+.3f}"
                    + (
                        f"\n    [separate-trunk value net] train_loss={_mean_loss_sep:.4f}  rmse={_train_rmse_sep:.2f}"
                        if _sep_net is not None else ""
                    )
                )
        if _best_val_state is not None:
            if self.separate_value_net:
                self.value_net.load_state_dict(_best_val_state)
            else:
                self.execution_net.value_head.load_state_dict(_best_val_state)
            log.info(f"  [value pretrain] restored best-val weights (val_loss={_best_val_loss:.4f})")
        log.info(f"Value pre-training done ({epochs_done} epoch(s), final train_loss={mean_loss:.4f})")
        if _sep_net is not None:
            log.info(
                f"  [separate value net experiment] final train_loss={_mean_loss_sep:.4f}"
                f"  (compare against shared-trunk final train_loss={mean_loss:.4f} above)"
            )
        if not self.separate_value_net:
            for p in _freeze_params:
                p.requires_grad_(True)

        return _rollout_stats

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
        _pool = self._spawn_value_pretrain_workers(phase_id)
        if _pool is not None and num_rollouts > 1:
            log.info(
                f"  [ppg value refit] spawned {_pool.n_processes} rollout process(es) "
                f"({'batched' if _pool.batched else 'plain'}), reused across all {num_rollouts} cycles"
            )

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
                batch, _rollout_stats = self._collect_value_pretrain_rollout(
                    env, n_steps, phase_id, use_gae=True, pool=_pool,
                )

                # --- Episode-level 85/15 train/val split (overfit detection) ---
                # Identical to pretrain_value()'s own split -- see its comments.
                dones_arr = batch["dones"].numpy()
                episode_end_idxs = np.where(dones_arr > 0.5)[0]
                n_complete_eps = len(episode_end_idxs)
                n_val_eps = max(1, round(0.15 * n_complete_eps)) if n_complete_eps >= 2 else 0
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
                anchor = self._ppg_snapshot_anchor(
                    _train_obs_full, _batch_size,
                    progress=ProgressReporter(
                        len(_train_obs_full["self_feat"]), prefix="  [ppg] anchor snapshot (train): ", live=True,
                    ),
                )
                val_anchor = (
                    self._ppg_snapshot_anchor(
                        val_obs_dict, _batch_size,
                        progress=ProgressReporter(
                            len(val_obs_dict["self_feat"]), prefix="  [ppg] anchor snapshot (val):   ", live=True,
                        ),
                    )
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
                    obs_source: dict, ret_source: torch.Tensor, progress_prefix: str = "",
                ) -> tuple[float, float, float]:
                    """Returns (normalized_mse, rmse, rmse excluding the worst
                    ``ppg_rmse_trim_frac`` rows -- see ``_trimmed_rmse``).
                    ``progress_prefix`` (optional) shows a per-chunk progress bar."""
                    total_sq = 0.0
                    n_rows = 0
                    sq_chunks: list[torch.Tensor] = []
                    _bar = ProgressReporter(len(ret_source), prefix=progress_prefix, live=True) if progress_prefix else None
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
                            if _bar is not None:
                                _bar.update(min(start + _batch_size, len(ret_source)))
                    mse = total_sq / max(n_rows, 1)
                    norm_mse = mse / float(ret_std ** 2)
                    _trim_rmse = _trimmed_rmse(torch.cat(sq_chunks) if sq_chunks else torch.empty(0), self._ppg_rmse_trim_frac)
                    return norm_mse, float(ret_std) * math.sqrt(norm_mse), _trim_rmse

                def _eval_value_and_kl(
                    obs_source: dict, actions_source: dict, ret_source: torch.Tensor, anchor_source: dict,
                    progress_prefix: str = "",
                ) -> tuple[float, float, float, float, dict[str, float]]:
                    """Same value-loss computation as _eval_value_loss, PLUS the KL
                    penalty against anchor_source -- used for the held-out val set
                    each epoch (never trained on; purely diagnostic) so the log can
                    show kl_val alongside kl_train instead of only measuring drift
                    on the data actually being optimized. Returns
                    (normalized_mse, rmse, trimmed_rmse, mean_kl, per_head_kl).
                    ``progress_prefix`` (optional) shows a per-chunk progress bar."""
                    total_sq = 0.0
                    n_rows = 0
                    sq_chunks: list[torch.Tensor] = []
                    kl_sum = 0.0
                    per_head_sum: dict[str, float] = {}
                    _bar = ProgressReporter(len(ret_source), prefix=progress_prefix, live=True) if progress_prefix else None
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
                            if _bar is not None:
                                _bar.update(min(start + _batch_size, len(ret_source)))
                    mse = total_sq / max(n_rows, 1)
                    norm_mse = mse / float(ret_std ** 2)
                    rmse = float(ret_std) * math.sqrt(norm_mse)
                    _trim_rmse = _trimmed_rmse(torch.cat(sq_chunks) if sq_chunks else torch.empty(0), self._ppg_rmse_trim_frac)
                    mean_kl = kl_sum / max(n_rows, 1)
                    per_head_mean = {k: v / max(n_rows, 1) for k, v in per_head_sum.items()}
                    return norm_mse, rmse, _trim_rmse, mean_kl, per_head_mean

                _trim_tag = f"ex{int(round(self._ppg_rmse_trim_frac * 100))}"
                _baseline_train_loss, _baseline_train_rmse, _baseline_train_trim = _eval_value_loss(
                    _train_obs_full, returns_t, progress_prefix="  [ppg] baseline eval (train): ",
                )
                if val_obs_dict is not None and val_returns_t is not None:
                    _baseline_val_loss, _baseline_val_rmse, _baseline_val_trim = _eval_value_loss(
                        val_obs_dict, val_returns_t, progress_prefix="  [ppg] baseline eval (val):   ",
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

                _best_val_loss = float("inf")
                _best_decision_state: Optional[dict] = None
                _best_execution_state: Optional[dict] = None
                _patience = 0
                _EARLY_STOP_PATIENCE = self._value_pretrain_early_stop_patience
                _EARLY_STOP_MIN_DELTA = self._value_pretrain_early_stop_min_delta
                mean_loss = float(_baseline_train_loss)
                epochs_done = 0

                for ep in range(epochs):
                    indices = torch.randperm(n)
                    ep_value_losses = []
                    ep_sq_errs: list[torch.Tensor] = []
                    ep_kl_totals = []
                    ep_head_kl_accum: dict[str, list[float]] = {}
                    # Live per-epoch training bar (rows/s; falls back to 10%
                    # milestone lines when stderr isn't a terminal). Its final
                    # update at start+batch >= n prints the closing newline, so
                    # the epoch's log line below never lands on the bar's row.
                    _ep_bar = ProgressReporter(n, prefix=f"  [ppg] epoch {ep + 1}/{epochs} train: ", live=True)
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
                        ppg_opt.step()

                        ep_value_losses.append(value_loss.item())
                        # Pre-step training-forward errors (same convention as the
                        # plain train rmse from ep_value_losses above), kept per row
                        # so the trimmed rmse below can drop the worst tail.
                        ep_sq_errs.append(((new_values.detach() - mb_ret) ** 2))
                        ep_kl_totals.append(total_kl.item())
                        for k, v in per_head_kl.items():
                            ep_head_kl_accum.setdefault(k, []).append(v.item())
                        _ep_bar.update(
                            min(start + _batch_size, n),
                            postfix=f"value_loss={np.mean(ep_value_losses):.4f} kl={np.mean(ep_kl_totals):.4f}",
                        )

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
                            progress_prefix=f"  [ppg] epoch {epochs_done}/{epochs} val:   ",
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

    # -----------------------------------------------------------------------
    # Policy sampling
    # -----------------------------------------------------------------------

    def _move_dir_head(self, raw_vec: torch.Tensor, log_kappa_param: torch.Tensor) -> "VonMisesDirectionHead":
        return VonMisesDirectionHead(raw_vec, log_kappa_param,
                             log_kappa_min=self.move_dir_log_kappa_min,
                             log_kappa_max=self.move_dir_log_kappa_max)

    def _kick_dir_head(self, raw_vec: torch.Tensor, log_kappa_param: torch.Tensor,
                        log_std_z_param: torch.Tensor) -> "KickDirectionHead":
        return KickDirectionHead(raw_vec, log_kappa_param, log_std_z_param,
                             log_kappa_min=self.kick_dir_log_kappa_min,
                             log_kappa_max=self.kick_dir_log_kappa_max,
                             log_std_z_min=self.kick_dir_z_log_std_min,
                             log_std_z_max=self.kick_dir_z_log_std_max)

    def _kick_power_head(self, raw_mean: torch.Tensor, log_std_param: torch.Tensor) -> "SquashedNormalHead":
        """kick_power: sigmoid-squashed scalar in [0,1] (power_fraction).

        Was previously applied fully deterministically (sigmoid(mean), no
        sampling/log_prob/entropy) -- see agent_plans/spin_implementation_plan.md
        section 6. SquashedNormalHead internally clamps its own log_std to
        (-5.0, 2.0), so no extra config bounds are needed here (unlike the
        VonMisesDirectionHead/KickDirectionHead heads above, which take
        externally-configured clamp bounds because their log_kappa/log_std
        also feed an explicit restoring-force regularizer this scoped fix
        does not add for kick_power/kick_spin).
        """
        return SquashedNormalHead(raw_mean, log_std_param, low=0.0, high=1.0, squash="sigmoid")

    def _kick_spin_dist(self, raw_mean: torch.Tensor, log_std_param: torch.Tensor) -> torch.distributions.Normal:
        """kick_spin: plain unsquashed 3D Normal, no L2-normalize.

        kick_spin has no bounded physical range wired up (it's still disabled
        at the apply_nn_action.py chokepoint -- real spin physics/clamping is
        deferred, see the plan doc). Neither VonMisesDirectionHead/
        KickDirectionHead (which force a unit-vector mean) nor SquashedNormalHead (which forces a squashed
        [low,high] range) fits an unbounded raw vector, so this uses
        torch.distributions.Normal directly -- same log_std clamp convention
        as SquashedNormalHead (-5.0, 2.0) for consistency.
        """
        std = torch.exp(log_std_param.clamp(-5.0, 2.0))
        return torch.distributions.Normal(raw_mean, std)

    def _per_head_new_log_probs(self, d_heads, e_heads, mb_actions: dict, exists_mask) -> torch.Tensor:
        """Per-sample, per-head log_prob under the CURRENT policy for stored
        actions, stacked in ``HEAD_LP_KEYS`` order (shape ``(batch, 15)``).

        Mirrors the gating in ``_recompute_log_prob`` (sprint/move_dir gated
        by exec_move; kick_dir/kick_power/kick_spin gated by kick) so summing
        this over heads
        matches the scalar log_prob used for the PPO ratio/KL. Used for the
        per-head KL diagnostic: ``batch["head_log_probs"][mb_idx] - this``,
        averaged over the batch dim, gives a per-head KL breakdown instead
        of just the scalar total.
        """
        def _b(logit, key):
            return IndependentBernoulli(logit).log_prob(mb_actions[key]).squeeze(-1)

        log_kappa_move = self.execution_net.move_dir_log_kappa.to(self.device)
        log_kappa_kick = self.execution_net.kick_dir_log_kappa.to(self.device)
        log_std_z_kick = self.execution_net.kick_dir_z_log_std.to(self.device)
        log_std_power = self.execution_net.kick_power_log_std.to(self.device)
        log_std_spin = self.execution_net.kick_spin_log_std.to(self.device)
        exec_move_mask = (mb_actions["exec_move"].squeeze(-1) > 0.5).float()
        kick_mask = (mb_actions["kick"].squeeze(-1) > 0.5).float()

        lp_move_dir = exec_move_mask * (
            self._move_dir_head(e_heads.move_direction, log_kappa_move).log_prob(
                mb_actions["move_dir_raw"]
            )
        )
        lp_kick_dir = kick_mask * (
            self._kick_dir_head(e_heads.kick_direction, log_kappa_kick, log_std_z_kick).log_prob(
                mb_actions["kick_dir_raw"]
            )
        )
        lp_kick_power = kick_mask * (
            self._kick_power_head(e_heads.kick_power, log_std_power).log_prob(
                mb_actions["kick_power_raw"]
            )
        )
        # kick_spin is permanently frozen (see agent_plans/spin_implementation_plan.md
        # section 0) -- masked to exactly zero here rather than computed and
        # discarded, since freezing requires_grad alone does NOT stop its
        # trunk-drift-induced log_prob contribution from drifting (see the
        # plan doc's section 0.2 for why).
        lp_kick_spin = torch.zeros(exists_mask.shape[0], device=self.device) if self._kick_spin_frozen else kick_mask * (
            self._kick_spin_dist(e_heads.kick_spin, log_std_spin).log_prob(
                mb_actions["kick_spin_raw"]
            ).sum(dim=-1)
        )
        lp_sprint = exec_move_mask * _b(e_heads.sprint_logit, "sprint")

        masked = self._ppo_lp_masked_heads
        _zero = torch.zeros(exists_mask.shape[0], device=self.device)
        return torch.stack([
            _zero if "shoot_logit" in masked else _b(d_heads.shoot_logit, "shoot"),
            _zero if "pass_logit" in masked else _b(d_heads.pass_logit, "pass_"),
            _zero if "move_logit" in masked else _b(d_heads.move_logit, "move"),
            _zero if "tackle_logit" in masked else _b(d_heads.tackle_logit, "tackle"),
            _zero if "get_possession_raw" in masked else _b(d_heads.get_possession_raw, "get_possession_extra"),
            _zero if "mark_logit" in masked else _b(d_heads.mark_logit, "mark"),
            _zero if "hold_position_logit" in masked else _b(d_heads.hold_position_logit, "hold_position"),
            _b(e_heads.exec_move_logit, "exec_move"),
            lp_sprint,
            _b(e_heads.kick_logit, "kick"),
            _b(e_heads.tackle_attempt_logit, "tackle_attempt"),
            lp_move_dir,
            lp_kick_dir,
            lp_kick_power,
            lp_kick_spin,
        ], dim=-1)

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
            sf_c, of_c, bf_c, _ = canonicalize_obs(
                batch["obs/self_feat"][idx].to(self.device),
                batch["obs/other_feat"][idx].to(self.device),
                batch["obs/ball_feat"][idx].to(self.device),
            )
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

    @torch.no_grad()
    def _ppg_snapshot_anchor(
        self, obs: dict, batch_size: int, progress: Optional["ProgressReporter"] = None,
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
        for start in range(0, n, batch_size):
            mb_obs = {k: v[start:start + batch_size].to(self.device) for k, v in obs.items()}
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
            for k, v in row.items():
                chunks.setdefault(k, []).append(v.detach())
            if progress is not None:
                progress.update(min(start + batch_size, n))
        return {k: torch.cat(v, dim=0) for k, v in chunks.items()}

    def _ppg_kl_penalty(
        self, d_heads, e_heads, anchor: dict[str, torch.Tensor],
        anchor_globals: dict[str, torch.Tensor], mb_actions: dict, exists_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        return total_per_row.mean(), per_head_mean

    @torch.no_grad()
    def _sample_action_networks(self, obs_dict_batch: dict) -> tuple:
        """Batched network-forward part of action sampling: decision_net +
        execution_net + value forward passes, run ONCE for however many rows
        ``obs_dict_batch`` holds (batch dim already present -- callers that
        want a single row pass a batch of 1). This is the expensive,
        batchable part -- profiled this session at ~88% of rollout wall
        time, of which only ~11% is actual matmul compute (the rest is
        Python/PyTorch dispatch overhead on tiny batch-of-1 calls); a direct
        benchmark measured 6 separate batch=1 CPU calls at 51.2s vs 1 batch=6
        CPU call at 10.5s for this exact 3-network combination (~4.88x).
        ALL per-row sampling/log_prob/decanonicalization logic stays in
        ``_sample_action_from_heads`` below, completely unchanged -- both the
        single-row path (``_sample_action``) and the batched path
        (``_sample_action_batch``) call that same method, so there is only
        ever one implementation of the sampling/log_prob math to keep
        correct, never a second parallel one that could silently drift.

        Returns ``(d_heads, e_heads, value_batch, x_sign_batch, em)`` -- all
        still batch-shaped ``(K, ...)``. ``em`` (exists_mask) is returned too
        since ``_sample_action_from_heads`` needs it again for the masked
        categorical target distributions (pass/tackle/mark target) -- it's
        part of the original observation, not something the network forward
        pass produces.
        """
        dev = self.device
        sf = obs_dict_batch["self_feat"].to(dev)
        of = obs_dict_batch["other_feat"].to(dev)
        em = obs_dict_batch["exists_mask"].to(dev)
        bf = obs_dict_batch["ball_feat"].to(dev)
        gf = obs_dict_batch["global_feat"].to(dev)
        sat = obs_dict_batch["self_ai_type"].to(dev) if "self_ai_type" in obs_dict_batch else None
        oat = obs_dict_batch["other_ai_type"].to(dev) if "other_ai_type" in obs_dict_batch else None

        # Canonical AI frame: mirror world-frame obs so self always attacks
        # +x (see ai/obs/canonical.py). Reused per-row below (via
        # _sample_action_from_heads) to decanonicalize move_direction/
        # kick_direction before they're returned to the caller.
        # (decision_net/execution_net wrap-canonicalize sf/of/bf automatically —
        # see CanonicalNetworkWrapper — so only x_sign itself is needed here.)
        # x_sign_of already returns a (K,) tensor for batched input -- see its
        # own docstring (ai/obs/canonical.py) -- no change needed here for
        # batch>1.
        x_sign_batch = x_sign_of(sf)

        d_heads = self.decision_net(sf, of, em, bf, gf, sat, oat)
        e_heads = self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat)

        # Single value head: execution_net only (decision_net.value_head is
        # frozen — see __init__ note), OR self.value_net when
        # separate_value_net is enabled (see _value_heads() docstring).
        if self.separate_value_net:
            with torch.no_grad():
                value_batch = self.value_net(sf, of, em, bf, gf, d_heads, sat, oat, value_only=True).squeeze(-1)
        else:
            value_batch = e_heads.value.squeeze(-1)

        return d_heads, e_heads, value_batch, x_sign_batch, em

    @torch.no_grad()
    def _sample_action_from_heads(
        self,
        d_heads,
        e_heads,
        value: float,
        x_sign: float,
        em,
        deterministic: bool = False,
        deterministic_decision: bool = False,
        deterministic_direction: bool = False,
    ) -> tuple:
        """Per-row sampling + log_prob + decanonicalization, given
        already-computed batch-of-1 network output for ONE row (``d_heads``/
        ``e_heads``/``value``/``x_sign``/``em`` -- see
        ``_sample_action_networks``, which also returns the batch-of-1
        ``exists_mask`` slice needed again here for the masked categorical
        target distributions). This is ``_sample_action``'s entire original
        body, byte-for-byte unchanged, minus the network forward passes
        themselves (now the caller's job) -- both ``_sample_action``
        (batch=1) and ``_sample_action_batch`` (batch=K, calling this once
        per row) route through this exact same code.

        Args:
            deterministic: if True, use each head's mode/mean instead of a
                stochastic sample for EVERY head (Bernoulli -> >=0.5,
                Categorical -> argmax, Normal/Direction heads -> mean) —
                shorthand for deterministic_decision=deterministic_direction=True.
            deterministic_decision: if True, only the discrete decision/execution
                heads (Bernoulli intents incl. exec_move/sprint/kick/tackle_attempt,
                and the pass/tackle/mark Categorical targets) use their mode;
                move/kick direction still sample.
            deterministic_direction: if True, only the continuous move_direction/
                kick_direction heads use their mean; discrete heads still sample.
            These are for evaluating a checkpoint without (all or part of) PPO
            exploration noise. Never used during rollout collection/training,
            only via load_for_inference() callers.

        Returns:
            (decision_action, log_prob, value, decision_probs,
             execution_physical, decision_physical, target_slots)
        """
        det_decision = deterministic or deterministic_decision
        det_direction = deterministic or deterministic_direction
        dev = self.device

        # Sample from each decision head
        shoot_dist = IndependentBernoulli(d_heads.shoot_logit)
        pass_dist = IndependentBernoulli(d_heads.pass_logit)
        move_dist = IndependentBernoulli(d_heads.move_logit)
        tackle_dist = IndependentBernoulli(d_heads.tackle_logit)
        gp_extra_dist = IndependentBernoulli(d_heads.get_possession_raw)
        mark_dist = IndependentBernoulli(d_heads.mark_logit)
        hold_dist = IndependentBernoulli(d_heads.hold_position_logit)
        shoot = shoot_dist.mode() if det_decision else shoot_dist.sample()
        pass_ = pass_dist.mode() if det_decision else pass_dist.sample()
        move = move_dist.mode() if det_decision else move_dist.sample()
        tackle = tackle_dist.mode() if det_decision else tackle_dist.sample()
        gp_extra = gp_extra_dist.mode() if det_decision else gp_extra_dist.sample()
        mark = mark_dist.mode() if det_decision else mark_dist.sample()
        hold = hold_dist.mode() if det_decision else hold_dist.sample()

        # Categorical targets (masked)
        pass_tgt_dist = MaskedCategorical(d_heads.pass_target_logits, em)
        tackle_tgt_dist = MaskedCategorical(d_heads.tackle_target_logits, em)
        mark_tgt_dist = MaskedCategorical(d_heads.mark_target_logits, em)
        pass_tgt = pass_tgt_dist.mode() if det_decision else pass_tgt_dist.sample()
        tackle_tgt = tackle_tgt_dist.mode() if det_decision else tackle_tgt_dist.sample()
        mark_tgt = mark_tgt_dist.mode() if det_decision else mark_tgt_dist.sample()

        # Continuous decision heads (pre-squash raw samples for PPO)
        mv_center_raw = d_heads.move_region_center  # (1, 2), no extra noise for now - use mean
        mv_size_raw = d_heads.move_region_size
        mv_speed_raw = d_heads.move_arrival_speed
        ad_raw = d_heads.attack_defence_raw

        # Decision probs for gating
        tackle_prob, gp_prob = derive_get_possession_prob(
            d_heads.tackle_logit, d_heads.get_possession_raw
        )
        decision_probs = {
            "shoot": float(torch.sigmoid(d_heads.shoot_logit)),
            "pass_": float(torch.sigmoid(d_heads.pass_logit)),
            "move": float(torch.sigmoid(d_heads.move_logit)),
            "tackle": float(tackle_prob),
            "get_possession": float(gp_prob),
            "mark": float(torch.sigmoid(d_heads.mark_logit)),
            "hold_position": float(torch.sigmoid(d_heads.hold_position_logit)),
        }
        target_slots = {
            "pass_": int(pass_tgt),
            "tackle": int(tackle_tgt),
            "mark": int(mark_tgt),
        }

        # Physical continuous outputs (after squashing)
        pitch_hl = 52.5  # standard half-length; TODO: get from obs if pitch varies
        pitch_hw = 34.0
        mv_center_phys = (
            torch.tanh(mv_center_raw) * torch.tensor([[pitch_hl, pitch_hw]], device=dev)
        )
        mv_size_phys = 1.0 + 3.0 * torch.sigmoid(mv_size_raw)  # [1, 4] m
        mv_speed_phys = float(torch.sigmoid(mv_speed_raw) * 9.5)  # [0, v_top]

        # Decanonicalize: move_region_center is a world-frame physical target.
        mv_center_world = mirror_x(mv_center_phys.squeeze(0), x_sign)

        decision_physical = {
            "move_region_center_m": mv_center_world.cpu().numpy(),
            "move_region_size_m": float(mv_size_phys),
            "move_arrival_speed_mps": mv_speed_phys,
        }

        # Sample execution heads
        exec_move_dist = IndependentBernoulli(e_heads.exec_move_logit)
        sprint_dist = IndependentBernoulli(e_heads.sprint_logit)
        kick_dist = IndependentBernoulli(e_heads.kick_logit)
        tackle_attempt_dist = IndependentBernoulli(e_heads.tackle_attempt_logit)
        exec_move = exec_move_dist.mode() if det_decision else exec_move_dist.sample()
        sprint = sprint_dist.mode() if det_decision else sprint_dist.sample()
        kick = kick_dist.mode() if det_decision else kick_dist.sample()
        tackle_attempt = tackle_attempt_dist.mode() if det_decision else tackle_attempt_dist.sample()

        # Direction heads: move_dir is a von Mises (azimuthal-only); kick_dir
        # is a von Mises (azimuthal) x plain Normal (elevation) composite --
        # see VonMisesDirectionHead/KickDirectionHead in ai/action/distributions.py.
        # We store the noisy raw sample (not the mean) so that log_prob ratios
        # during the PPO update are meaningful — new_mean vs stored sample.
        # In deterministic (direction) mode we use the (normalized) mean direction instead.
        eps = 1e-6
        log_kappa_move = self.execution_net.move_dir_log_kappa
        log_kappa_kick = self.execution_net.kick_dir_log_kappa
        log_std_z_kick = self.execution_net.kick_dir_z_log_std
        move_dir_head = self._move_dir_head(e_heads.move_direction, log_kappa_move)
        kick_dir_head = self._kick_dir_head(e_heads.kick_direction, log_kappa_kick, log_std_z_kick)
        if det_direction:
            move_dir_raw = move_dir_head.mode_physical()  # (1, 2)
            kick_dir_raw = kick_dir_head.mode_physical()   # (1, 3)
        else:
            move_dir_raw = move_dir_head.sample_raw()  # (1, 2)
            kick_dir_raw = kick_dir_head.sample_raw()   # (1, 3)
        move_dir_phys = (move_dir_raw / (move_dir_raw.norm(dim=-1, keepdim=True) + eps)).squeeze(0)
        kick_dir_phys = (kick_dir_raw / (kick_dir_raw.norm(dim=-1, keepdim=True) + eps)).squeeze(0)

        # kick_power/kick_spin: previously applied fully deterministically
        # (sigmoid(mean) / raw mean, no sampling/log_prob/entropy) -- see
        # agent_plans/spin_implementation_plan.md section 6. Now sampled the
        # same way as move_dir/kick_dir above (mode in deterministic-direction
        # mode, else a real stochastic sample), with the raw (pre-squash)
        # sample stored for PPO log_prob recomputation at update time.
        log_std_power = self.execution_net.kick_power_log_std
        log_std_spin = self.execution_net.kick_spin_log_std
        kick_power_head = self._kick_power_head(e_heads.kick_power, log_std_power)
        kick_spin_dist = self._kick_spin_dist(e_heads.kick_spin, log_std_spin)
        if det_direction:
            kick_power_raw = kick_power_head.dist.mean  # (1, 1), pre-squash
            kick_spin_raw = kick_spin_dist.mean.squeeze(0)  # (3,)
        else:
            kick_power_raw = kick_power_head.sample_raw()  # (1, 1), pre-squash
            kick_spin_raw = kick_spin_dist.rsample().squeeze(0)  # (3,)
        kick_power_phys = float(kick_power_head.to_physical(kick_power_raw))

        # Decanonicalize: these are world-frame physical directions from here on.
        move_dir_world = mirror_x(move_dir_phys, x_sign)
        kick_dir_world = mirror_x(kick_dir_phys, x_sign)

        execution_physical = {
            "exec_move": bool(exec_move.item() > 0.5),
            "move_direction": move_dir_world.cpu().numpy(),
            "sprint": bool(sprint.item() > 0.5),
            "kick_this_tick": bool(kick.item() > 0.5),
            "kick_direction": kick_dir_world.cpu().numpy(),
            "kick_power_fraction": kick_power_phys,
            "kick_spin": kick_spin_raw.cpu().numpy(),
            "tackle_attempt": bool(tackle_attempt.item() > 0.5),
        }

        # Combined log_prob
        # (value was already computed by _sample_action_networks and passed
        # in as a plain float -- see that method's docstring.)
        log_prob = self._compute_log_prob(d_heads, e_heads, {
            "shoot": shoot, "pass_": pass_, "move": move,
            "tackle": tackle, "gp_extra": gp_extra, "mark": mark, "hold": hold,
            "pass_tgt": pass_tgt, "tackle_tgt": tackle_tgt, "mark_tgt": mark_tgt,
            "exec_move": exec_move, "sprint": sprint, "kick": kick,
            "tackle_attempt": tackle_attempt,
            "move_dir_raw": move_dir_raw, "kick_dir_raw": kick_dir_raw,
            "kick_power_raw": kick_power_raw, "kick_spin_raw": kick_spin_raw.unsqueeze(0),
        }, em)

        # Per-head log_probs for DEBUG KL breakdown (stored alongside total in buffer)
        _lsm = self.execution_net.move_dir_log_kappa
        _lsk = self.execution_net.kick_dir_log_kappa
        _lskz = self.execution_net.kick_dir_z_log_std
        # Debug per-head log_probs: apply same masking as _compute_log_prob
        # so these values match what went into the stored total log_prob.
        # Frozen/masked decision heads (self._ppo_lp_masked_heads, set via
        # set_frozen_heads() for the current curriculum phase) are zeroed
        # here too -- _per_head_new_log_probs() (used at update time to
        # recompute the "after" side of the per-head KL diagnostic) already
        # zeroes these same heads, so without masking here the stored
        # "before" side was nonzero while "after" was zero, producing a
        # spurious per-head KL (= the raw sampled log_prob, not a real KL)
        # for every frozen head -- see ai_trainer_knowledge.md "per-head KL
        # masking" note.
        _exec_move_active = float(exec_move) > 0.5
        _kick_active = float(kick) > 0.5
        _masked = self._ppo_lp_masked_heads
        head_log_probs = np.array([
            0.0 if "shoot_logit" in _masked else float(IndependentBernoulli(d_heads.shoot_logit).log_prob(shoot).sum()),
            0.0 if "pass_logit" in _masked else float(IndependentBernoulli(d_heads.pass_logit).log_prob(pass_).sum()),
            0.0 if "move_logit" in _masked else float(IndependentBernoulli(d_heads.move_logit).log_prob(move).sum()),
            0.0 if "tackle_logit" in _masked else float(IndependentBernoulli(d_heads.tackle_logit).log_prob(tackle).sum()),
            0.0 if "get_possession_raw" in _masked else float(IndependentBernoulli(d_heads.get_possession_raw).log_prob(gp_extra).sum()),
            0.0 if "mark_logit" in _masked else float(IndependentBernoulli(d_heads.mark_logit).log_prob(mark).sum()),
            0.0 if "hold_position_logit" in _masked else float(IndependentBernoulli(d_heads.hold_position_logit).log_prob(hold).sum()),
            float(IndependentBernoulli(e_heads.exec_move_logit).log_prob(exec_move).sum()),
            # sprint: only when exec_move=True
            float(IndependentBernoulli(e_heads.sprint_logit).log_prob(sprint).sum()) if _exec_move_active else 0.0,
            float(IndependentBernoulli(e_heads.kick_logit).log_prob(kick).sum()),
            float(IndependentBernoulli(e_heads.tackle_attempt_logit).log_prob(tackle_attempt).sum()),
            # move_dir: only when exec_move=True
            float(self._move_dir_head(e_heads.move_direction, _lsm).log_prob(move_dir_raw)) if _exec_move_active else 0.0,
            # kick_dir: only when kick=True
            float(self._kick_dir_head(e_heads.kick_direction, _lsk, _lskz).log_prob(kick_dir_raw)) if _kick_active else 0.0,
            # kick_power: only when kick=True (same gating as kick_dir --
            # only ever used downstream inside the `if kick_this_tick:` block)
            float(kick_power_head.log_prob(kick_power_raw)) if _kick_active else 0.0,
            # kick_spin: permanently frozen (see spin_implementation_plan.md section 0) --
            # masked to exactly 0.0 rather than computed and discarded.
            0.0 if self._kick_spin_frozen else (
                float(kick_spin_dist.log_prob(kick_spin_raw.unsqueeze(0)).sum(dim=-1)) if _kick_active else 0.0
            ),
        ], dtype=np.float32)

        # Build DecisionAction for storage
        action = DecisionAction(
            shoot=float(shoot),
            pass_=float(pass_),
            move=float(move),
            tackle=float(tackle),
            get_possession_extra=float(gp_extra),
            mark=float(mark),
            hold_position=float(hold),
            pass_target=int(pass_tgt),
            tackle_target=int(tackle_tgt),
            mark_target=int(mark_tgt),
            move_region_center_raw=mv_center_raw.squeeze(0).cpu().numpy(),
            move_region_size_raw=float(mv_size_raw),
            move_arrival_speed_raw=float(mv_speed_raw),
            attack_defence_raw=float(ad_raw),
        )

        # Raw execution samples needed to recompute log_prob during PPO update
        raw_exec_samples = {
            "exec_move": np.array([float(exec_move)], dtype=np.float32),
            "sprint": np.array([float(sprint)], dtype=np.float32),
            "kick": np.array([float(kick)], dtype=np.float32),
            "tackle_attempt": np.array([float(tackle_attempt)], dtype=np.float32),
            "move_dir_raw": move_dir_raw.squeeze(0).cpu().numpy().astype(np.float32),
            "kick_dir_raw": kick_dir_raw.squeeze(0).cpu().numpy().astype(np.float32),
            "kick_power_raw": kick_power_raw.squeeze(0).cpu().numpy().astype(np.float32),
            "kick_spin_raw": kick_spin_raw.cpu().numpy().astype(np.float32),
        }

        return (
            action,
            float(log_prob),
            value,
            decision_probs,
            execution_physical,
            decision_physical,
            target_slots,
            raw_exec_samples,
            head_log_probs,
        )

    @torch.no_grad()
    def _sample_action_from_heads_batch(
        self,
        d_heads,
        e_heads,
        value_batch: torch.Tensor,
        x_sign_batch: torch.Tensor,
        em,
        deterministic: bool = False,
        deterministic_decision: bool = False,
        deterministic_direction: bool = False,
    ) -> list:
        """Vectorized tail: sampling + log_prob + decanonicalization for ALL
        K rows in ONE pass, replacing a Python loop of K individual
        ``_sample_action_from_heads`` calls (that loop -- still used by
        ``_sample_action_from_heads`` itself as the single-row reference,
        and by this method's own tests -- is ~53% of rollout decision time
        despite the network forward pass already being batched, since every
        distribution/log_prob/masking call it makes is batch=1). Returns a
        list of K tuples, each identical in shape/semantics to what
        ``_sample_action_from_heads`` returns for that row (see its
        docstring) -- this is a drop-in replacement for
        ``_sample_action_batch``'s inner loop, not a new/different sampling
        policy.

        Why this is safe to vectorize: every distribution class involved
        (IndependentBernoulli, MaskedCategorical, VonMisesDirectionHead,
        KickDirectionHead, SquashedNormalHead, torch.distributions.Normal)
        already operates on an arbitrary leading batch dimension -- the
        batch=1 restriction in ``_sample_action_from_heads`` came entirely
        from ITS OWN ``.squeeze(0)``/``float()``/``.item()`` calls, never
        from the underlying sampling machinery. Likewise ``mirror_x``/
        ``x_sign_of`` (ai/obs/canonical.py) already accept a batched
        x_sign. The one place with real per-row CONDITIONAL logic -- gating
        sprint/move_dir by exec_move and kick_dir/kick_power/kick_spin by
        kick, plus masking frozen heads (``self._ppo_lp_masked_heads``) --
        is deliberately NOT reimplemented here: this method builds an
        ``mb_actions`` dict from the freshly sampled actions and hands it to
        ``_recompute_log_prob``/``_per_head_new_log_probs``, the SAME
        already-vectorized functions the PPO update loop already calls every
        rollout to recompute log_prob for stored actions (verified
        line-for-line equivalent to ``_compute_log_prob``'s scalar gating
        used above by the single-row reference). The masking logic that was
        flagged as the highest-risk part of this vectorization therefore has
        exactly one implementation either way, never a second one that could
        silently drift -- this method's own new code is limited to sampling,
        decanonicalization, and packaging already-computed batched tensors
        into K per-row output tuples.
        """
        det_decision = deterministic or deterministic_decision
        det_direction = deterministic or deterministic_direction
        dev = self.device
        K = int(value_batch.shape[0])

        # Decision heads (Bernoulli)
        shoot_dist = IndependentBernoulli(d_heads.shoot_logit)
        pass_dist = IndependentBernoulli(d_heads.pass_logit)
        move_dist = IndependentBernoulli(d_heads.move_logit)
        tackle_dist = IndependentBernoulli(d_heads.tackle_logit)
        gp_extra_dist = IndependentBernoulli(d_heads.get_possession_raw)
        mark_dist = IndependentBernoulli(d_heads.mark_logit)
        hold_dist = IndependentBernoulli(d_heads.hold_position_logit)
        shoot = shoot_dist.mode() if det_decision else shoot_dist.sample()
        pass_ = pass_dist.mode() if det_decision else pass_dist.sample()
        move = move_dist.mode() if det_decision else move_dist.sample()
        tackle = tackle_dist.mode() if det_decision else tackle_dist.sample()
        gp_extra = gp_extra_dist.mode() if det_decision else gp_extra_dist.sample()
        mark = mark_dist.mode() if det_decision else mark_dist.sample()
        hold = hold_dist.mode() if det_decision else hold_dist.sample()

        # Categorical targets (masked)
        pass_tgt_dist = MaskedCategorical(d_heads.pass_target_logits, em)
        tackle_tgt_dist = MaskedCategorical(d_heads.tackle_target_logits, em)
        mark_tgt_dist = MaskedCategorical(d_heads.mark_target_logits, em)
        pass_tgt = pass_tgt_dist.mode() if det_decision else pass_tgt_dist.sample()
        tackle_tgt = tackle_tgt_dist.mode() if det_decision else tackle_tgt_dist.sample()
        mark_tgt = mark_tgt_dist.mode() if det_decision else mark_tgt_dist.sample()

        # Continuous decision heads (pre-squash raw samples for PPO)
        mv_center_raw = d_heads.move_region_center  # (K, 2)
        mv_size_raw = d_heads.move_region_size       # (K, 1)
        mv_speed_raw = d_heads.move_arrival_speed    # (K, 1)
        ad_raw = d_heads.attack_defence_raw          # (K, 1)

        tackle_prob, gp_prob = derive_get_possession_prob(
            d_heads.tackle_logit, d_heads.get_possession_raw
        )
        shoot_prob = torch.sigmoid(d_heads.shoot_logit)
        pass_prob = torch.sigmoid(d_heads.pass_logit)
        move_prob = torch.sigmoid(d_heads.move_logit)
        mark_prob = torch.sigmoid(d_heads.mark_logit)
        hold_prob = torch.sigmoid(d_heads.hold_position_logit)

        # Physical continuous outputs (after squashing)
        pitch_hl = 52.5  # standard half-length; TODO: get from obs if pitch varies
        pitch_hw = 34.0
        mv_center_phys = (
            torch.tanh(mv_center_raw) * torch.tensor([[pitch_hl, pitch_hw]], device=dev)
        )
        mv_size_phys = 1.0 + 3.0 * torch.sigmoid(mv_size_raw)  # (K, 1), [1, 4] m
        mv_speed_phys = torch.sigmoid(mv_speed_raw) * 9.5       # (K, 1), [0, v_top]

        # Decanonicalize: move_region_center is a world-frame physical target.
        mv_center_world = mirror_x(mv_center_phys, x_sign_batch)  # (K, 2)

        # Sample execution heads
        exec_move_dist = IndependentBernoulli(e_heads.exec_move_logit)
        sprint_dist = IndependentBernoulli(e_heads.sprint_logit)
        kick_dist = IndependentBernoulli(e_heads.kick_logit)
        tackle_attempt_dist = IndependentBernoulli(e_heads.tackle_attempt_logit)
        exec_move = exec_move_dist.mode() if det_decision else exec_move_dist.sample()
        sprint = sprint_dist.mode() if det_decision else sprint_dist.sample()
        kick = kick_dist.mode() if det_decision else kick_dist.sample()
        tackle_attempt = tackle_attempt_dist.mode() if det_decision else tackle_attempt_dist.sample()

        # Direction heads (see _sample_action_from_heads for the full
        # rationale comment -- unchanged here, just batched).
        eps = 1e-6
        log_kappa_move = self.execution_net.move_dir_log_kappa
        log_kappa_kick = self.execution_net.kick_dir_log_kappa
        log_std_z_kick = self.execution_net.kick_dir_z_log_std
        move_dir_head = self._move_dir_head(e_heads.move_direction, log_kappa_move)
        kick_dir_head = self._kick_dir_head(e_heads.kick_direction, log_kappa_kick, log_std_z_kick)
        if det_direction:
            move_dir_raw = move_dir_head.mode_physical()  # (K, 2)
            kick_dir_raw = kick_dir_head.mode_physical()   # (K, 3)
        else:
            move_dir_raw = move_dir_head.sample_raw()  # (K, 2)
            kick_dir_raw = kick_dir_head.sample_raw()   # (K, 3)
        move_dir_phys = move_dir_raw / (move_dir_raw.norm(dim=-1, keepdim=True) + eps)
        kick_dir_phys = kick_dir_raw / (kick_dir_raw.norm(dim=-1, keepdim=True) + eps)

        log_std_power = self.execution_net.kick_power_log_std
        log_std_spin = self.execution_net.kick_spin_log_std
        kick_power_head = self._kick_power_head(e_heads.kick_power, log_std_power)
        kick_spin_dist = self._kick_spin_dist(e_heads.kick_spin, log_std_spin)
        if det_direction:
            kick_power_raw = kick_power_head.dist.mean  # (K, 1), pre-squash
            kick_spin_raw = kick_spin_dist.mean          # (K, 3)
        else:
            kick_power_raw = kick_power_head.sample_raw()  # (K, 1), pre-squash
            kick_spin_raw = kick_spin_dist.rsample()        # (K, 3)
        kick_power_phys = kick_power_head.to_physical(kick_power_raw)  # (K, 1)

        # Decanonicalize: these are world-frame physical directions from here on.
        move_dir_world = mirror_x(move_dir_phys, x_sign_batch)  # (K, 2)
        kick_dir_world = mirror_x(kick_dir_phys, x_sign_batch)  # (K, 3)

        # Combined + per-head log_prob: reuse the update-time vectorized
        # masking (_recompute_log_prob/_per_head_new_log_probs) rather than
        # reimplementing exec_move/kick/frozen-head gating here -- see
        # docstring above.
        mb_actions = {
            "shoot": shoot, "pass_": pass_, "move": move, "tackle": tackle,
            "get_possession_extra": gp_extra, "mark": mark, "hold_position": hold,
            "pass_target": pass_tgt.unsqueeze(-1).float(),
            "tackle_target": tackle_tgt.unsqueeze(-1).float(),
            "mark_target": mark_tgt.unsqueeze(-1).float(),
            "exec_move": exec_move, "sprint": sprint, "kick": kick,
            "tackle_attempt": tackle_attempt,
            "move_dir_raw": move_dir_raw, "kick_dir_raw": kick_dir_raw,
            "kick_power_raw": kick_power_raw, "kick_spin_raw": kick_spin_raw,
        }
        log_prob_batch = self._recompute_log_prob(d_heads, e_heads, mb_actions, em)  # (K,)
        head_log_probs_batch = self._per_head_new_log_probs(d_heads, e_heads, mb_actions, em)  # (K, 15)

        # Unpack into K per-row tuples -- cheap indexing/dict-building only,
        # no distribution construction or network calls below this point.
        results = []
        for i in range(K):
            action = DecisionAction(
                shoot=float(shoot[i]),
                pass_=float(pass_[i]),
                move=float(move[i]),
                tackle=float(tackle[i]),
                get_possession_extra=float(gp_extra[i]),
                mark=float(mark[i]),
                hold_position=float(hold[i]),
                pass_target=int(pass_tgt[i]),
                tackle_target=int(tackle_tgt[i]),
                mark_target=int(mark_tgt[i]),
                move_region_center_raw=mv_center_raw[i].cpu().numpy(),
                move_region_size_raw=float(mv_size_raw[i]),
                move_arrival_speed_raw=float(mv_speed_raw[i]),
                attack_defence_raw=float(ad_raw[i]),
            )
            decision_probs = {
                "shoot": float(shoot_prob[i]),
                "pass_": float(pass_prob[i]),
                "move": float(move_prob[i]),
                "tackle": float(tackle_prob[i]),
                "get_possession": float(gp_prob[i]),
                "mark": float(mark_prob[i]),
                "hold_position": float(hold_prob[i]),
            }
            target_slots = {
                "pass_": int(pass_tgt[i]),
                "tackle": int(tackle_tgt[i]),
                "mark": int(mark_tgt[i]),
            }
            decision_physical = {
                "move_region_center_m": mv_center_world[i].cpu().numpy(),
                "move_region_size_m": float(mv_size_phys[i]),
                "move_arrival_speed_mps": float(mv_speed_phys[i]),
            }
            execution_physical = {
                "exec_move": bool(exec_move[i].item() > 0.5),
                "move_direction": move_dir_world[i].cpu().numpy(),
                "sprint": bool(sprint[i].item() > 0.5),
                "kick_this_tick": bool(kick[i].item() > 0.5),
                "kick_direction": kick_dir_world[i].cpu().numpy(),
                "kick_power_fraction": float(kick_power_phys[i]),
                "kick_spin": kick_spin_raw[i].cpu().numpy(),
                "tackle_attempt": bool(tackle_attempt[i].item() > 0.5),
            }
            raw_exec_samples = {
                "exec_move": np.array([float(exec_move[i])], dtype=np.float32),
                "sprint": np.array([float(sprint[i])], dtype=np.float32),
                "kick": np.array([float(kick[i])], dtype=np.float32),
                "tackle_attempt": np.array([float(tackle_attempt[i])], dtype=np.float32),
                "move_dir_raw": move_dir_raw[i].cpu().numpy().astype(np.float32),
                "kick_dir_raw": kick_dir_raw[i].cpu().numpy().astype(np.float32),
                "kick_power_raw": kick_power_raw[i].cpu().numpy().astype(np.float32),
                "kick_spin_raw": kick_spin_raw[i].cpu().numpy().astype(np.float32),
            }
            results.append((
                action,
                float(log_prob_batch[i]),
                float(value_batch[i]),
                decision_probs,
                execution_physical,
                decision_physical,
                target_slots,
                raw_exec_samples,
                head_log_probs_batch[i].cpu().numpy(),
            ))
        return results

    @torch.no_grad()
    def _sample_action(
        self,
        obs_dict: dict,
        deterministic: bool = False,
        deterministic_decision: bool = False,
        deterministic_direction: bool = False,
    ) -> tuple:
        """Forward pass + sample from all distributions, for ONE observation.

        Thin wrapper: builds a batch of 1 and routes through
        ``_sample_action_batch`` (see its docstring, and
        ``_sample_action_networks``/``_sample_action_from_heads``) so the
        single-row path can never diverge from the batched path -- it IS the
        batched path, called with batch_size=1.

        See ``_sample_action_from_heads`` for the deterministic*/return-value
        documentation (unchanged).
        """
        obs_batch = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
        return self._sample_action_batch(
            obs_batch,
            deterministic=deterministic,
            deterministic_decision=deterministic_decision,
            deterministic_direction=deterministic_direction,
        )[0]

    @torch.no_grad()
    def _sample_action_batch(
        self,
        obs_dict_batch: dict,
        deterministic: bool = False,
        deterministic_decision: bool = False,
        deterministic_direction: bool = False,
    ) -> list:
        """Batched variant of ``_sample_action``: ``obs_dict_batch`` already
        has a batch dimension (K rows, K>=1). Runs the 3 network forward
        passes ONCE (``_sample_action_networks``) and then the vectorized
        tail (``_sample_action_from_heads_batch``) ONCE, for all K rows
        together -- see that method's docstring for why this is safe.
        Returns a list of K tuples, each identical in shape/semantics to
        what a single ``_sample_action(...)`` call would return for that
        row's own observation (see ``_sample_action_from_heads``'s
        docstring for the exact tuple contents; ``_sample_action_from_heads_batch``
        returns the same shape, just computed vectorized).
        """
        d_heads_b, e_heads_b, value_b, x_sign_b, em_b = self._sample_action_networks(obs_dict_batch)
        return self._sample_action_from_heads_batch(
            d_heads_b, e_heads_b, value_b, x_sign_b, em_b,
            deterministic=deterministic,
            deterministic_decision=deterministic_decision,
            deterministic_direction=deterministic_direction,
        )

    def _get_value(self, obs_dict: dict) -> float:
        sf = obs_dict["self_feat"].to(self.device)
        of = obs_dict["other_feat"].to(self.device)
        em = obs_dict["exists_mask"].to(self.device)
        bf = obs_dict["ball_feat"].to(self.device)
        gf = obs_dict["global_feat"].to(self.device)
        sat = obs_dict["self_ai_type"].to(self.device) if "self_ai_type" in obs_dict else None
        oat = obs_dict["other_ai_type"].to(self.device) if "other_ai_type" in obs_dict else None
        d_heads = self.decision_net(sf, of, em, bf, gf, sat, oat)
        _value = self._value_heads(sf, of, em, bf, gf, d_heads, sat, oat)
        return float(_value.mean())  # single critic (execution_net, or self.value_net)

    def _bootstrap_last_values(self, env, next_obs, buffer: "RolloutBuffer") -> dict[str, float]:
        """Per-track bootstrap values for ``RolloutBuffer.compute_gae()``.

        The trainee's own next-state value comes from ``next_obs`` (already
        available at the caller's rollout-window boundary — the return value
        of the most recent ``env.step()``). For every OTHER track actually
        present in the buffer (i.e. a secondary player whose transitions
        were recorded this rollout — only possible when that player is
        neural-controlled), re-encode that player's own current observation
        via ``env._get_obs(player_id=...)`` and run it through the same
        critic. Without this, ``compute_gae`` would silently fall back to
        0.0 for that track's final-step bootstrap — see ``compute_gae``'s
        docstring for why that's only an approximation, not a crash.
        """
        with torch.no_grad():
            last_obs_dict = {
                k: v.unsqueeze(0).to(self.device) for k, v in next_obs.to_torch_dict().items()
            }
            last_values = {"trainee": self._get_value(last_obs_dict)}
            for track in set(buffer.track_ids):
                if track == "trainee" or track in last_values:
                    continue
                sec_obs = env._get_obs(player_id=track)
                sec_obs_dict = {
                    k: v.unsqueeze(0).to(self.device) for k, v in sec_obs.to_torch_dict().items()
                }
                last_values[track] = self._get_value(sec_obs_dict)
        return last_values

    def _compute_log_prob(self, d_heads, e_heads, samples: dict, exists_mask) -> torch.Tensor:
        """Compute combined log_prob across all action heads."""
        lp = torch.zeros(1, device=self.device)
        masked = self._ppo_lp_masked_heads

        # Bernoulli decision heads (skip heads masked for current phase)
        if "shoot_logit" not in masked:
            lp += IndependentBernoulli(d_heads.shoot_logit).log_prob(samples["shoot"]).sum()
        if "pass_logit" not in masked:
            lp += IndependentBernoulli(d_heads.pass_logit).log_prob(samples["pass_"]).sum()
        if "move_logit" not in masked:
            lp += IndependentBernoulli(d_heads.move_logit).log_prob(samples["move"]).sum()
        if "tackle_logit" not in masked:
            lp += IndependentBernoulli(d_heads.tackle_logit).log_prob(samples["tackle"]).sum()
        if "get_possession_raw" not in masked:
            lp += IndependentBernoulli(d_heads.get_possession_raw).log_prob(samples["gp_extra"]).sum()
        if "mark_logit" not in masked:
            lp += IndependentBernoulli(d_heads.mark_logit).log_prob(samples["mark"]).sum()
        if "hold_position_logit" not in masked:
            lp += IndependentBernoulli(d_heads.hold_position_logit).log_prob(samples["hold"]).sum()

        # Categorical target heads (gated by intent; also skip when parent is masked)
        if "pass_logit" not in masked and samples["pass_"] > 0.5:
            lp += MaskedCategorical(d_heads.pass_target_logits, exists_mask).log_prob(
                samples["pass_tgt"]
            )
        if "tackle_logit" not in masked and samples["tackle"] > 0.5:
            lp += MaskedCategorical(d_heads.tackle_target_logits, exists_mask).log_prob(
                samples["tackle_tgt"]
            )
        if "mark_logit" not in masked and samples["mark"] > 0.5:
            lp += MaskedCategorical(d_heads.mark_target_logits, exists_mask).log_prob(
                samples["mark_tgt"]
            )

        # Unconditional execution Bernoulli heads
        lp += IndependentBernoulli(e_heads.exec_move_logit).log_prob(samples["exec_move"]).sum()
        lp += IndependentBernoulli(e_heads.kick_logit).log_prob(samples["kick"]).sum()
        lp += IndependentBernoulli(e_heads.tackle_attempt_logit).log_prob(
            samples["tackle_attempt"]
        ).sum()

        # Sub-parameters gated by parent action — only contribute to log_prob
        # when the parent was actually taken.  Unconditional inclusion injects
        # large-variance noise from unused heads and inflates KL divergence.
        log_kappa_move = self.execution_net.move_dir_log_kappa
        log_kappa_kick = self.execution_net.kick_dir_log_kappa
        log_std_z_kick = self.execution_net.kick_dir_z_log_std
        log_std_power = self.execution_net.kick_power_log_std
        log_std_spin = self.execution_net.kick_spin_log_std
        # sprint + move_dir: only when exec_move=True (player was moving)
        if float(samples["exec_move"]) > 0.5:
            lp += IndependentBernoulli(e_heads.sprint_logit).log_prob(samples["sprint"]).sum()
            lp += self._move_dir_head(e_heads.move_direction, log_kappa_move).log_prob(
                samples["move_dir_raw"]
            )
        # kick_dir/kick_power: only when kick=True (a kick was taken) -- both
        # are only ever consumed downstream inside the engine's
        # `if kick_this_tick:` block.
        if float(samples["kick"]) > 0.5:
            lp += self._kick_dir_head(e_heads.kick_direction, log_kappa_kick, log_std_z_kick).log_prob(
                samples["kick_dir_raw"]
            )
            lp += self._kick_power_head(e_heads.kick_power, log_std_power).log_prob(
                samples["kick_power_raw"]
            )
        # kick_spin is permanently frozen (see agent_plans/spin_implementation_plan.md
        # section 0) -- deliberately excluded here, not just gated by kick=True,
        # since freezing requires_grad alone does not stop trunk-drift noise
        # from leaking into the ratio (see the plan doc's section 0.2).
        if not self._kick_spin_frozen and float(samples["kick"]) > 0.5:
            lp += self._kick_spin_dist(e_heads.kick_spin, log_std_spin).log_prob(
                samples["kick_spin_raw"]
            ).sum(dim=-1)

        return lp

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
        out_path = f"nan_diagnostic_{int(time.time())}.json"
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2)
        log.error(
            f"[NaN DIAGNOSTIC] stage={stage} epoch={epoch_i} mb={mb_i} "
            f"secondary_rows={n_secondary}/{n_total} ({report['secondary_fraction']:.1%}) "
            f"-- full report written to {out_path}"
        )
        raise PPOTrainer._NaNReproFound(out_path)

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

                mb_adv = val_adv[idx].to(self.device)
                mb_ret = val_returns[idx].to(self.device)
                mb_old_lp = val_old_lp[idx].to(self.device)
                ratio = torch.exp(new_log_probs - mb_old_lp)
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
                    head_ratio = torch.exp(mb_new_head_lp - mb_old_head_lp)
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
        self, batch: dict,
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
        """
        n = len(batch["log_probs"])
        has_head_lp = "head_log_probs" in batch
        lp_parts: list[torch.Tensor] = []
        head_lp_parts: list[torch.Tensor] = []
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
        log_probs = torch.cat(lp_parts, dim=0)
        head_log_probs = torch.cat(head_lp_parts, dim=0) if has_head_lp else None
        return log_probs, head_log_probs

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
        if self.augment_n_slot_shuffles > 0:
            batch["log_probs"], _recomputed_head_lp = self._recompute_old_log_probs_for_augmented_batch(batch)
            if _recomputed_head_lp is not None:
                batch["head_log_probs"] = _recomputed_head_lp

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

        # Normalize advantages
        adv = batch["advantages"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        old_log_probs = batch["log_probs"]
        returns = batch["returns"]

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
                mb_old_lp = old_log_probs[mb_idx].to(self.device)
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
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * mb_adv
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
                        _head_ratio = torch.exp(_mb_new_head_lp - _mb_old_head_lp)  # (batch, 15)
                        _head_adv = mb_adv.unsqueeze(-1)
                        _head_w = mb_w.unsqueeze(-1)
                        _head_surr1 = _head_ratio * _head_adv
                        _head_surr2 = torch.clamp(_head_ratio, 1.0 - clip, 1.0 + clip) * _head_adv
                        _head_min_surr = self._apply_dual_clip(
                            torch.min(_head_surr1, _head_surr2), _head_adv
                        )
                        _head_policy_loss = -(_head_min_surr * _head_w).mean(dim=0)
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
                entropy, _ent_bkdn = self._compute_entropy(d_heads, e_heads, em, return_breakdown=True)
                for _ek, _ev in _ent_bkdn.items():
                    all_entropy_breakdown.setdefault(_ek, []).append(_ev)

                # No dir_l2 penalty needed: direction means are unit-normalized in
                # forward() so their magnitude is always 1 — penalizing it is a no-op.

                # log_kappa/log_std restoring force: without this, ent_dir_weight *
                # entropy is a one-directional force that only ever DEFLATES
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
                              - ent_coef * entropy
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
                self.optimizer.step()
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
                        per_head_kl_mb = (mb_old_head_lp - per_head_new_lp_after).mean(dim=0)  # (13,)
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
                # NOTE: direction heads (move_dir/kick_dir) are NOT scaled by
                # ent_dir_weight in the REAL training log_prob (see
                # _compute_log_prob/_recompute_log_prob above — ent_dir_weight
                # now only scales the entropy bonus, see _compute_entropy).
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

    def _recompute_log_prob(self, d_heads, e_heads, mb_actions: dict, exists_mask) -> torch.Tensor:
        """Recompute log_probs for stored actions under the current policy."""
        lp = torch.zeros(exists_mask.shape[0], device=self.device)
        masked = self._ppo_lp_masked_heads

        def _b(logit, key):
            return IndependentBernoulli(logit).log_prob(mb_actions[key]).squeeze(-1)

        if "shoot_logit" not in masked:
            lp += _b(d_heads.shoot_logit, "shoot")
        if "pass_logit" not in masked:
            lp += _b(d_heads.pass_logit, "pass_")
        if "move_logit" not in masked:
            lp += _b(d_heads.move_logit, "move")
        if "tackle_logit" not in masked:
            lp += _b(d_heads.tackle_logit, "tackle")
        if "get_possession_raw" not in masked:
            lp += _b(d_heads.get_possession_raw, "get_possession_extra")
        if "mark_logit" not in masked:
            lp += _b(d_heads.mark_logit, "mark")
        if "hold_position_logit" not in masked:
            lp += _b(d_heads.hold_position_logit, "hold_position")
        lp += _b(e_heads.exec_move_logit, "exec_move")
        lp += _b(e_heads.kick_logit, "kick")
        lp += _b(e_heads.tackle_attempt_logit, "tackle_attempt")

        # Target categorical log_probs (gated by parent intent).
        # Automatically skipped when the parent Bernoulli is masked — if pass_logit
        # is masked then pass_ samples are excluded from the ratio and the categorical
        # target contribution would be spurious noise too.
        pass_mask = mb_actions["pass_"].squeeze(-1) > 0.5
        tackle_mask = mb_actions["tackle"].squeeze(-1) > 0.5
        mark_mask = mb_actions["mark"].squeeze(-1) > 0.5

        for parent_name, mask, logits, key in [
            ("pass_logit",   pass_mask,   d_heads.pass_target_logits,   "pass_target"),
            ("tackle_logit", tackle_mask, d_heads.tackle_target_logits, "tackle_target"),
            ("mark_logit",   mark_mask,   d_heads.mark_target_logits,   "mark_target"),
        ]:
            if parent_name in masked:
                continue
            if mask.any():
                cat_lp = MaskedCategorical(logits, exists_mask).log_prob(
                    mb_actions[key].long().squeeze(-1)
                )
                lp[mask] += cat_lp[mask]

        # Sub-parameters gated by parent action (vectorised float mask over minibatch).
        # sprint + move_dir only contribute when exec_move=True.
        # kick_dir only contributes when kick=True.
        # Without this gating, unused heads inject large-variance log_prob noise
        # that inflates KL and triggers spurious early stops every rollout.
        log_kappa_move = self.execution_net.move_dir_log_kappa.to(self.device)
        log_kappa_kick = self.execution_net.kick_dir_log_kappa.to(self.device)
        log_std_z_kick = self.execution_net.kick_dir_z_log_std.to(self.device)
        log_std_power = self.execution_net.kick_power_log_std.to(self.device)
        log_std_spin = self.execution_net.kick_spin_log_std.to(self.device)
        exec_move_mask = (mb_actions["exec_move"].squeeze(-1) > 0.5).float()
        kick_mask = (mb_actions["kick"].squeeze(-1) > 0.5).float()
        lp += exec_move_mask * _b(e_heads.sprint_logit, "sprint")
        lp += exec_move_mask * (
            self._move_dir_head(e_heads.move_direction, log_kappa_move).log_prob(
                mb_actions["move_dir_raw"]
            )
        )
        lp += kick_mask * (
            self._kick_dir_head(e_heads.kick_direction, log_kappa_kick, log_std_z_kick).log_prob(
                mb_actions["kick_dir_raw"]
            )
        )
        lp += kick_mask * (
            self._kick_power_head(e_heads.kick_power, log_std_power).log_prob(
                mb_actions["kick_power_raw"]
            )
        )
        # kick_spin is permanently frozen (see agent_plans/spin_implementation_plan.md
        # section 0) -- excluded here rather than just gated by kick_mask, see
        # the plan doc's section 0.2 for why requires_grad alone isn't enough.
        if not self._kick_spin_frozen:
            lp += kick_mask * (
                self._kick_spin_dist(e_heads.kick_spin, log_std_spin).log_prob(
                    mb_actions["kick_spin_raw"]
                ).sum(dim=-1)
            )

        return lp

    def _compute_entropy(self, d_heads, e_heads, exists_mask, return_breakdown: bool = False):
        """Entropy bonus, consistent with the masked log_prob.

        Restricted to exactly the heads in HEAD_LP_KEYS (rollout_buffer.py)
        -- the same 15 heads _recompute_log_prob/the per-head KL diagnostic
        track. pass_target/tackle_target/mark_target are NOT included: they
        never enter the PPO importance ratio at all (BC-aux-only, see
        set_frozen_heads' docstring), so an entropy bonus over them doesn't
        serve PPO's own exploration-regularization purpose. Any head
        currently in self._ppo_lp_masked_heads (frozen for this curriculum
        phase, no reward signal) is zeroed here too, matching
        _recompute_log_prob exactly -- otherwise the entropy bonus would
        still pull the shared decision_net trunk towards more uncertainty on
        a "frozen" head via that head's own (still-differentiable) forward
        pass, even though PPO's policy-gradient term for that head is zero.

        Sub-parameter heads (sprint, move_dir, kick_dir) are weighted by
        E[parent active] to match the masking in _recompute_log_prob where
        those terms are 0 when the parent is not taken.

        decision_net's 7 Bernoulli heads (shoot/pass/move/tackle/
        get_possession_raw/mark/hold_position) are additionally scaled by
        ent_decision_weight, separate from execution_net's exec_move/kick/
        tackle_attempt (always weight 1.0). See ai_config.json's
        ent_decision_weight comment.

        Args:
            return_breakdown: if True, also return a dict of each head's own
                (already E[parent active]-weighted, matching what's actually
                added into the total below) entropy contribution — used by
                the per-rollout "[per-head entropy]" diagnostic log line to
                show which heads are driving a rising aggregate entropy,
                rather than just the summed scalar. See ai/knowledge.md's
                "Per-head entropy diagnostics" note.
        """
        ent = torch.zeros(1, device=self.device)
        # Per-head breakdown tensors, NOT yet synced to CPU -- see the single
        # combined torch.stack(...).tolist() call below. Called every PPO
        # minibatch with return_breakdown=True, so 15 separate .item() calls
        # here (one CPU<->GPU round trip each) was a real, measurable per-
        # minibatch cost; batching into one round trip changes none of the
        # returned values, only how many syncs it takes to get them.
        _bkdn_tensors: dict[str, torch.Tensor] = {}
        masked = self._ppo_lp_masked_heads
        # Unconditional heads (no parent gate)
        for name, logit, mask_key in [
            ("shoot", d_heads.shoot_logit, "shoot_logit"),
            ("pass_", d_heads.pass_logit, "pass_logit"),
            ("move", d_heads.move_logit, "move_logit"),
            ("tackle", d_heads.tackle_logit, "tackle_logit"),
            ("gp_extra", d_heads.get_possession_raw, "get_possession_raw"),
            ("mark", d_heads.mark_logit, "mark_logit"),
            ("hold", d_heads.hold_position_logit, "hold_position_logit"),
            ("exec_move", e_heads.exec_move_logit, None),
            ("kick", e_heads.kick_logit, None),
            ("tackle_attempt", e_heads.tackle_attempt_logit, None),
        ]:
            if mask_key is not None and mask_key in masked:
                h = torch.zeros((), device=self.device)
            else:
                h = IndependentBernoulli(logit).entropy().mean()
                if mask_key is not None:
                    # mask_key is only set for decision_net's own heads (the
                    # 7 in _LP_HEAD_NAMES) -- exec_move/kick/tackle_attempt
                    # (mask_key=None) always keep weight 1.0 regardless of
                    # ent_decision_weight. See its ai_config.json comment.
                    h = self.ent_decision_weight * h
            ent += h
            _bkdn_tensors[name] = h
        # Sub-parameters: scale by E[parent active] to match masked log_prob.
        log_kappa_move = self.execution_net.move_dir_log_kappa
        log_kappa_kick = self.execution_net.kick_dir_log_kappa
        log_std_z_kick = self.execution_net.kick_dir_z_log_std
        log_std_power = self.execution_net.kick_power_log_std
        log_std_spin = self.execution_net.kick_spin_log_std
        p_exec_move = torch.sigmoid(e_heads.exec_move_logit).mean()
        p_kick = torch.sigmoid(e_heads.kick_logit).mean()
        h_sprint = p_exec_move * IndependentBernoulli(e_heads.sprint_logit).entropy().mean()
        h_move_dir = p_exec_move * self.ent_dir_weight * self._move_dir_head(e_heads.move_direction, log_kappa_move).entropy().mean()
        h_kick_dir = p_kick * self.ent_dir_weight * self._kick_dir_head(e_heads.kick_direction, log_kappa_kick, log_std_z_kick).entropy().mean()
        h_kick_power = p_kick * self.ent_kick_power_weight * self._kick_power_head(e_heads.kick_power, log_std_power).entropy().mean()
        # kick_spin is permanently frozen (see agent_plans/spin_implementation_plan.md
        # section 0) -- its entropy term is masked to exactly zero rather than
        # computed and discarded, same rationale as the log_prob masking above.
        h_kick_spin = (
            torch.zeros((), device=self.device) if self._kick_spin_frozen else
            p_kick * self.ent_kick_spin_weight * self._kick_spin_dist(e_heads.kick_spin, log_std_spin).entropy().sum(dim=-1).mean()
        )
        ent += h_sprint + h_move_dir + h_kick_dir + h_kick_power + h_kick_spin
        _bkdn_tensors["sprint"] = h_sprint
        _bkdn_tensors["move_dir"] = h_move_dir
        _bkdn_tensors["kick_dir"] = h_kick_dir
        _bkdn_tensors["kick_power"] = h_kick_power
        _bkdn_tensors["kick_spin"] = h_kick_spin
        if return_breakdown:
            _names = list(_bkdn_tensors.keys())
            _vals = torch.stack([_bkdn_tensors[n] for n in _names]).tolist()
            breakdown = dict(zip(_names, _vals))
            return ent, breakdown
        return ent

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def _rotate_log_file(self) -> None:
        """(Re)attach a FileHandler writing to
        ``checkpoint_dir/training_log{N}.txt`` where N is one past the
        current ``_checkpoint_count`` -- e.g. before checkpoint1 is saved,
        logs go to ``training_log1.txt``; once checkpoint1 is saved, the
        handler rotates so subsequent logs (up through checkpoint2) go to
        ``training_log2.txt``, mirroring the ``checkpoint{N}.pt`` numbering.
        Attaches to the ``"footballcoach"`` root logger so every module's
        logs (train.py, ppo_trainer.py, bc.py, dataset.py, ...) land in the
        same file, not just this module's own logger.
        """
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger("footballcoach")
        if self._log_file_handler is not None:
            root.removeHandler(self._log_file_handler)
            self._log_file_handler.close()
        log_path = self.checkpoint_dir / f"training_log{self._checkpoint_count + 1}.txt"
        # utf-8 explicitly -- many diagnostic log lines use Unicode symbols
        # (Delta, sigma, box-drawing separators, degree signs); without this,
        # FileHandler defaults to the system locale (cp1252 on Windows),
        # which can't encode them and drops those lines with a logging error.
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(handler)
        self._log_file_handler = handler
        log.info(f"Logging to {log_path}")

    def _save_checkpoint(self, step: int) -> None:
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_count += 1
        path = self.checkpoint_dir / f"checkpoint{self._checkpoint_count}.pt"
        ckpt = {
            "step": step,
            "checkpoint_count": self._checkpoint_count,
            "decision_net": self.decision_net.state_dict(),
            "execution_net": self.execution_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.value_net is not None:
            ckpt["value_net"] = self.value_net.state_dict()
            ckpt["value_net_optimizer"] = self.value_net_optimizer.state_dict()
        torch.save(ckpt, path)
        # Update latest.pt -- prefer a symlink (cheap, no duplicate disk
        # usage), but Windows requires admin/Developer Mode privileges to
        # create one. Fall back to a plain copy so checkpoint saving never
        # crashes an otherwise-healthy training run over this.
        latest = self.checkpoint_dir / "latest.pt"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        try:
            latest.symlink_to(path.name)
        except OSError:
            shutil.copy2(path, latest)
        log.info(f"Saved checkpoint: {path}")
        self._rotate_log_file()

    def _save_checkpoint_to(self, path: Path) -> None:
        """Save a checkpoint to an explicit path (used for pre-trained snapshot).

        self.optimizer is None for a trainer built with inference_only=True
        (e.g. a standalone script training decision_net/execution_net with
        its own separate optimizer, never touching PPO's) -- omit the
        "optimizer" key in that case rather than crashing; load_checkpoint()
        already tolerates its absence the same way ("if self.optimizer is
        not None and 'optimizer' in ckpt:"), this just makes the save side
        consistent with that.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "step": self._total_steps,
            "decision_net": self.decision_net.state_dict(),
            "execution_net": self.execution_net.state_dict(),
        }
        if self.optimizer is not None:
            ckpt["optimizer"] = self.optimizer.state_dict()
        if self.value_net is not None:
            ckpt["value_net"] = self.value_net.state_dict()
            if self.value_net_optimizer is not None:
                ckpt["value_net_optimizer"] = self.value_net_optimizer.state_dict()
        torch.save(ckpt, path)

    def load_checkpoint(self, path: Path, reset_optimizer: bool = False) -> int:
        """Load network weights (and, unless reset_optimizer, optimizer
        state) from a checkpoint saved by _save_checkpoint()/_save_checkpoint_to().

        Args:
            reset_optimizer: If True, skip restoring Adam's state (per-param
                running m/v moment estimates + step count) for both
                self.optimizer and self.value_net_optimizer -- network
                weights still load normally either way. Adam's own state is
                otherwise carried forward unchanged across a resume (same as
                self._total_steps), which is usually what you want for a
                true continuation, but not when you've since changed
                hyperparameters enough (e.g. max_grad_norm, learning rates,
                added/removed param groups) that Adam's old running
                averages -- tuned for the previous regime -- would fight the
                new one rather than help it. Left in place (not
                auto-reset), same param-group count or not, unless this is
                explicitly set.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # Tolerant load (not strict): decision_net.state_dict() deliberately
        # excludes ball_physics_encoder/player_physics_encoder keys (see
        # DecisionNetwork.state_dict()), so a checkpoint saved with this
        # feature enabled never has them either -- strict loading would
        # otherwise raise "missing keys" for a submodule that's already
        # correctly populated from its own physics_pretrain checkpoint path.
        _load_state_dict_tolerant(self.decision_net, ckpt["decision_net"], "decision_net")
        _execution_sd = _migrate_direction_log_std_to_kappa(ckpt["execution_net"])
        _load_state_dict_tolerant(self.execution_net, _execution_sd, "execution_net")
        if self.value_net is not None:
            if "value_net" in ckpt:
                # value_net reuses the whole ExecutionNetwork class (only its
                # .value output is ever read) so it carries the same stray
                # move_dir_log_std/kick_dir_log_std keys under an old
                # checkpoint -- migrate here too, purely for a clean load
                # (these params are never actually used by the critic).
                _value_sd = _migrate_direction_log_std_to_kappa(ckpt["value_net"])
                _load_state_dict_tolerant(self.value_net, _value_sd, "value_net")
                if not reset_optimizer and self.value_net_optimizer is not None and "value_net_optimizer" in ckpt:
                    self.value_net_optimizer.load_state_dict(ckpt["value_net_optimizer"])
            else:
                log.warning(
                    f"separate_value_net enabled but {path} has no 'value_net' key "
                    "(checkpoint predates this feature) -- value_net keeps its fresh "
                    "random init."
                )
        if reset_optimizer:
            log.info(f"--reset-optimizer: skipping optimizer state restore for {path} (network weights still loaded normally)")
        elif self.optimizer is not None and "optimizer" in ckpt:
            saved_n_groups = len(ckpt["optimizer"].get("param_groups", []))
            current_n_groups = len(self.optimizer.param_groups)
            if saved_n_groups == current_n_groups:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            else:
                # Optimizer shape changed (e.g. value-head param group added) since this
                # checkpoint was saved. Weights still load fine above; only the optimizer's
                # momentum/Adam state is skipped — harmless for --from-pretrained/--latest-
                # pretrain (a fresh PPO run builds its own optimizer state from step 0
                # anyway) but means a true PPO *resume* from an old-shape checkpoint will
                # restart Adam's running averages rather than continuing them exactly.
                log.warning(
                    f"Optimizer param group count changed ({saved_n_groups} -> "
                    f"{current_n_groups}); skipping optimizer state restore for {path} "
                    "(network weights still loaded normally)."
                )
        self._total_steps = ckpt["step"]
        log.info(f"Loaded checkpoint: {path} (step {self._total_steps})")
        return self._total_steps

    @classmethod
    def from_config(cls, **kwargs) -> "PPOTrainer":
        """Build a PPOTrainer with freshly-initialised networks from ai_config.json.

        Pass separate_value_net=True to enable the dedicated, fully independent
        critic network (see __init__ docstring / CLI --separate-value-net).
        """
        cfg = load_ai_config()
        decision_net = DecisionNetwork.from_config()
        net_cfg = cfg.get("network", {})
        if net_cfg.get("share_entity_encoder", False):
            execution_net = ExecutionNetwork.from_config(
                shared_entity_encoder=decision_net.entity_encoder,
                shared_ball_mlp=decision_net.ball_mlp,
                shared_global_mlp=decision_net.global_mlp,
            )
        else:
            execution_net = ExecutionNetwork.from_config()
        return cls(decision_net=decision_net, execution_net=execution_net, cfg=cfg, **kwargs)

    @classmethod
    def load_for_inference(cls, path: "Path | str") -> "PPOTrainer":
        """Load networks only — no optimizer created. Safe to call inside pygame/UI."""
        path = Path(path)
        cfg = load_ai_config()
        decision_net = DecisionNetwork.from_config()
        execution_net = ExecutionNetwork.from_config()
        # Auto-detect separate_value_net from the checkpoint itself so inference
        # (UI/evaluate.py) doesn't need to know which mode a given checkpoint was
        # trained with.
        _ckpt_peek = torch.load(path, map_location="cpu", weights_only=False)
        _separate_value_net = "value_net" in _ckpt_peek
        trainer = cls(
            decision_net=decision_net,
            execution_net=execution_net,
            cfg=cfg,
            inference_only=True,
            separate_value_net=_separate_value_net,
        )
        trainer.load_checkpoint(path)
        trainer.decision_net.eval()
        trainer.execution_net.eval()
        if trainer.value_net is not None:
            trainer.value_net.eval()
        return trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_numpy(action: DecisionAction, exec_samples: dict) -> dict[str, np.ndarray]:
    """Flatten a DecisionAction + raw execution samples to a numpy dict for the rollout buffer."""
    return {
        "shoot": np.array([action.shoot], dtype=np.float32),
        "pass_": np.array([action.pass_], dtype=np.float32),
        "move": np.array([action.move], dtype=np.float32),
        "tackle": np.array([action.tackle], dtype=np.float32),
        "get_possession_extra": np.array([action.get_possession_extra], dtype=np.float32),
        "mark": np.array([action.mark], dtype=np.float32),
        "hold_position": np.array([action.hold_position], dtype=np.float32),
        "pass_target": np.array([action.pass_target], dtype=np.float32),
        "tackle_target": np.array([action.tackle_target], dtype=np.float32),
        "mark_target": np.array([action.mark_target], dtype=np.float32),
        "exec_move": exec_samples["exec_move"],
        "sprint": exec_samples["sprint"],
        "kick": exec_samples["kick"],
        "tackle_attempt": exec_samples["tackle_attempt"],
        "move_dir_raw": exec_samples["move_dir_raw"],
        "kick_dir_raw": exec_samples["kick_dir_raw"],
        "kick_power_raw": exec_samples["kick_power_raw"],
        "kick_spin_raw": exec_samples["kick_spin_raw"],
        "move_region_center_raw": action.move_region_center_raw,
        "move_region_size_raw": np.array([action.move_region_size_raw], dtype=np.float32),
        "move_arrival_speed_raw": np.array([action.move_arrival_speed_raw], dtype=np.float32),
    }


def _merge_worker_batches(batches: list[dict], release_inputs: bool = False) -> dict:
    """Concatenate per-worker ``RolloutBuffer.as_tensors()`` dicts along dim 0.

    Each worker's batch must already have GAE applied independently (its
    ``advantages``/``returns`` were computed against its OWN trailing
    bootstrap value) -- concatenating raw transitions across worker
    boundaries BEFORE computing GAE would corrupt advantage estimates by
    treating unrelated episodes/workers as one continuous trajectory.

    ``release_inputs=True`` deletes each key from every input dict as soon as
    that key has been concatenated, so peak memory is ONE full copy of the
    data plus a single key's worth of extra, instead of the inputs AND the
    merged result alive together (~2x) -- the rollout-end RAM spike this
    exists to flatten. The input dicts are left EMPTY afterwards, so only
    pass it when the caller is done with them (the streaming rollout
    consumers are). Default False = inputs untouched, as before.
    """
    merged: dict = {}
    for key in list(batches[0].keys()):
        if key in ("reward_comps_raw", "step_outcomes", "track_ids"):
            merged[key] = [x for b in batches for x in b[key]]
        else:
            merged[key] = torch.cat([b[key] for b in batches], dim=0)
        if release_inputs:
            for b in batches:
                del b[key]
    return merged


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


def _split_batch_releasing(
    batch: dict, train_mask: np.ndarray, val_mask: Optional[np.ndarray],
) -> tuple[dict, Optional[dict]]:
    """Split a merged rollout batch into ``(train, val)`` row subsets (boolean
    masks over rows; ``val_mask=None`` -> ``val`` is None) while EMPTYING
    ``batch`` key by key as each key is sliced, so the full batch and its two
    slices are never all alive together (peak ~1 full copy plus one key,
    instead of ~2 full copies). ``batch`` is left empty afterwards -- the
    caller must have read anything it still needs from it first (e.g.
    ``batch["returns"]`` for a normalisation std) and must not use it again.
    The list-valued keys (``reward_comps_raw``/``step_outcomes``/
    ``track_ids``) are sliced as Python lists, exactly like the tensors."""
    list_keys = ("reward_comps_raw", "step_outcomes", "track_ids")
    train_idx = torch.from_numpy(np.where(train_mask)[0]).long()
    train_list = train_idx.tolist()
    val_idx = torch.from_numpy(np.where(val_mask)[0]).long() if val_mask is not None else None
    val_list = val_idx.tolist() if val_idx is not None else None
    train: dict = {}
    val: Optional[dict] = {} if val_idx is not None else None
    for key in list(batch.keys()):
        v = batch.pop(key)
        if key in list_keys:
            train[key] = [v[i] for i in train_list]
            if val is not None:
                val[key] = [v[i] for i in val_list]
        else:
            train[key] = v[train_idx]
            if val is not None:
                val[key] = v[val_idx]
        del v
    return train, val


@dataclass
class _ValuePretrainWorkers:
    """A live pool of value-pretrain rollout worker processes -- either
    plain one-env-per-process ``rollout_worker.py`` workers, or (when
    ``ppo.value_pretrain_batched_rollout`` is on) ``batched_rollout_worker.py``
    workers each owning ``envs_per_process`` envs. Built by
    ``PPOTrainer._spawn_value_pretrain_workers``; ``progress_value`` is the
    shared ``ctx.Value`` step counter the workers add to (always present).
    """
    handles: list
    batched: bool
    progress_value: object
    n_processes: int
    envs_per_process: int


def _finalize_value_pretrain_result(
    r: dict, gamma: float, lam: float, use_gae: bool, allow_empty: bool = False,
) -> tuple[Optional[dict], int]:
    """Turn ONE worker/env result dict (``{"buffer", "last_value"?, "stats"}``)
    from the value-pretrain rollout into an ``as_tensors()`` batch dict with
    its ``returns`` filled in. Returns ``(batch_dict, n_rows_dropped)``.

    ``allow_empty`` (MC mode only): when the buffer holds NO completed
    episode at all -- with per-env whole-episode chunking that is common (an
    env's final flush is often just its unfinished tail) -- MC returns for
    those rows would be truncated garbage and ``truncate_to_last_episode_end``
    (which never empties a buffer) would leave them in place. With
    ``allow_empty=True`` the whole buffer is dropped instead and
    ``(None, n_rows)`` is returned; callers must skip a None batch. Default
    False keeps the old keep-everything behaviour for the callers that can't
    tolerate an empty result (single-process collection, plain workers whose
    single per-worker buffer always spans many episodes).

    Never concatenate raw transitions across result boundaries before this
    (same per-worker discipline as ``_merge_worker_batches``'s docstring).

    - ``use_gae=False`` (``pretrain_value``): drop any trailing incomplete
      episode (``truncate_to_last_episode_end``), then pure Monte Carlo
      discounted returns -- unchanged from the original behaviour.
    - ``use_gae=True`` AND the result carries a ``"last_value"`` (both worker
      kinds compute one via ``PPOTrainer._bootstrap_last_values``): GAE with
      that REAL bootstrap value and NO truncation, so a trailing partial
      episode's data is kept (correctly bootstrapped) instead of thrown away.
    - ``use_gae=True`` but no ``"last_value"`` (the single-process
      collection branch, which builds a bare buffer): truncate, then GAE with
      a literal 0.0 bootstrap -- provably never used, since truncation leaves
      the buffer ending on a ``done=1`` row and GAE's backward recursion
      resets at every ``done=1``.
    """
    buf = r["buffer"]
    if use_gae and r.get("last_value") is not None:
        advantages, returns = buf.compute_gae(gamma, lam, r["last_value"])
        n_dropped = 0
    else:
        if allow_empty and not use_gae and buf.last_complete_episode_end() < 0:
            return None, len(buf)
        n_dropped = buf.truncate_to_last_episode_end()
        if use_gae:
            advantages, returns = buf.compute_gae(gamma, lam, 0.0)
        else:
            advantages = [0.0] * len(buf.rewards)
            returns = buf.compute_mc_returns(gamma)
    return buf.as_tensors(advantages, returns), n_dropped

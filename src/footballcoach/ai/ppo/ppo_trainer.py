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

import copy
import dataclasses
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from footballcoach.ai.action.distributions import (
    DirectionHead,
    IndependentBernoulli,
    MaskedCategorical,
    SquashedNormalHead,
)
from footballcoach.ai.action.gating import select_action
from footballcoach.ai.action.schema import DecisionAction, DecisionHeadsRaw, ExecutionAction
from footballcoach.ai.config import load_ai_config
from footballcoach.ai.eval.seeded_eval import (
    default_eval_seeds,
    run_seeded_evaluation,
    run_seeded_evaluation_parallel,
)
from footballcoach.ai.models.decision_network import DecisionNetwork, derive_get_possession_prob
from footballcoach.ai.models.execution_network import ExecutionNetwork, flatten_decision_heads
from footballcoach.ai.obs.augment import N_FLIP_VARIANTS, augment_batch, augment_obs_bc
from footballcoach.ai.obs.canonical import (
    CanonicalNetworkWrapper,
    canonicalize_bc_labels,
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


def _detach_decision_heads(d_heads: DecisionHeadsRaw) -> DecisionHeadsRaw:
    """Return a copy of d_heads with every field detached (see separate_value_net)."""
    return dataclasses.replace(
        d_heads, **{name: getattr(d_heads, name).detach() for name in _DECISION_HEADS_FIELD_NAMES}
    )


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
]

# Execution-network head name -> nn.Module attribute name, used for the
# per-head gradient-norm / logit-drift diagnostics in _ppo_update().
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


def _eval_worker_factory(
    decision_state: dict, execution_state: dict, separate_value_net: bool,
    value_state: Optional[dict], use_rules_ai: bool, max_episode_s: float,
) -> tuple:
    """Module-level (picklable) zero-arg-after-partial factory for parallel
    seeded eval (ai/eval/seeded_eval.py's run_seeded_evaluation_parallel) --
    each subprocess rebuilds its own inference-only PPOTrainer from the
    passed state dicts (live nn.Module/optimizer objects aren't picklable
    across the process boundary), mirroring ai/ppo/rollout_worker.py."""
    from footballcoach.rules_ai import Phase1RulesAI
    from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition
    from footballcoach.ai.env.scenario_env import ScenarioEnv

    trainer = PPOTrainer.from_config(
        device=torch.device("cpu"), inference_only=True, separate_value_net=separate_value_net,
    )
    trainer.decision_net.load_state_dict(decision_state)
    trainer.execution_net.load_state_dict(execution_state)
    if value_state is not None and trainer.value_net is not None:
        trainer.value_net.load_state_dict(value_state)

    _label = "rules" if use_rules_ai else "immobile"

    def _eval_env_factory(seed: int) -> ScenarioEnv:
        def _eval_build(*_a, **_kw):
            _m = build_1v1_scenario(*_a, seed=seed, **_kw)
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

    return _eval_env_factory, trainer._sample_action


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


# Slash-joined short labels matching outcome_breakdown()'s column order, for
# a one-time "vs[...]:" legend prefix on log lines instead of repeating the
# key names on every vs_rules/vs_immobile/vs_neural segment.
_PHASE1_OUTCOME_LEGEND = "/".join(label for _, label in _PHASE1_OUTCOME_KEYS)


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
        self.checkpoint_dir = checkpoint_dir

        ppo_cfg = cfg["ppo"]
        curriculum_cfg = cfg.get("curriculum", {})
        bc_cfg = cfg.get("bc", {})
        self.schedules = TrainingSchedules(ppo_cfg, curriculum_cfg, bc_cfg)

        self.gamma = float(ppo_cfg.get("gamma", 0.99))
        self.lam = float(ppo_cfg.get("lam", 0.95))
        self.clip_range = float(ppo_cfg.get("clip_range", 0.2))
        self.vf_coef = float(ppo_cfg.get("vf_coef", 0.5))
        self.ent_coef = float(ppo_cfg.get("ent_coef", 0.01))
        self.max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))
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
        self.rollout_steps = int(ppo_cfg.get("rollout_steps", 2048))
        self.dir_l2_coef = float(ppo_cfg.get("dir_l2_coef", 0.01))
        self.move_dir_log_std_min = float(ppo_cfg.get("move_dir_log_std_min", ppo_cfg.get("dir_log_std_min", -5.0)))
        self.move_dir_log_std_max = float(ppo_cfg.get("move_dir_log_std_max", ppo_cfg.get("dir_log_std_max", 2.0)))
        self.move_dir_log_std_target = float(ppo_cfg.get("move_dir_log_std_target", ppo_cfg.get("dir_log_std_target", self.move_dir_log_std_min)))
        self.move_dir_log_std_reg_coef = float(ppo_cfg.get("move_dir_log_std_reg_coef", ppo_cfg.get("dir_log_std_reg_coef", 0.0)))
        self.kick_dir_log_std_min = float(ppo_cfg.get("kick_dir_log_std_min", ppo_cfg.get("dir_log_std_min", -5.0)))
        self.kick_dir_log_std_max = float(ppo_cfg.get("kick_dir_log_std_max", ppo_cfg.get("dir_log_std_max", 2.0)))
        self.kick_dir_log_std_target = float(ppo_cfg.get("kick_dir_log_std_target", ppo_cfg.get("dir_log_std_target", self.kick_dir_log_std_min)))
        self.kick_dir_log_std_reg_coef = float(ppo_cfg.get("kick_dir_log_std_reg_coef", ppo_cfg.get("dir_log_std_reg_coef", 0.0)))
        self.ent_dir_weight = float(ppo_cfg.get("ent_dir_weight", 1.0))
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
        self.n_parallel_envs = int(ppo_cfg.get("n_parallel_envs", 1))
        self.worker_torch_threads = int(ppo_cfg.get("worker_torch_threads", 1))
        self._aug_rng = random.Random()
        self._bc_cfg = bc_cfg
        self._bc_dir_loss_w = float(bc_cfg.get("direction_loss_weight", 3.0))
        self._bc_region_loss_w = float(bc_cfg.get("region_loss_weight", 1.0))
        self._bc_dec_label_smoothing = float(bc_cfg.get("dec_label_smoothing", 0.0))
        self._bc_exec_label_smoothing = float(bc_cfg.get("exec_label_smoothing", 0.0))
        self._bc_dec_weight = float(bc_cfg.get("bc_dec_weight", 1.0))
        self._bc_exec_weight = float(bc_cfg.get("bc_exec_weight", 1.0))
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

        lr = float(ppo_cfg.get("learning_rate", 3e-4))
        value_lr = float(ppo_cfg.get("value_learning_rate", lr))
        # Adam weight_decay (L2) applied ONLY to the value param group during PPO
        # (see ai_config.json's ppo.value_weight_decay comment). 0.0 (default) =
        # disabled, matching prior behaviour.
        value_weight_decay = float(ppo_cfg.get("value_weight_decay", 0.0))
        # Optional separate LR for the direction heads (move_direction, kick_direction,
        # move_dir_log_std, kick_dir_log_std). None (default) = share the "policy"
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
            # nn.Parameter attributes like move_dir_log_std/kick_dir_log_std, which
            # named_parameters() also yields with their attribute names).
            direction_param_ids = set()
            direction_params = []
            move_dir_param_ids = set()
            move_dir_params = []
            kick_dir_param_ids = set()
            kick_dir_params = []
            for name, p in execution_net.named_parameters():
                if id(p) in value_param_ids:
                    continue
                if name.startswith("move_direction.") or name == "move_dir_log_std":
                    move_dir_params.append(p)
                    move_dir_param_ids.add(id(p))
                elif name.startswith("kick_direction.") or name == "kick_dir_log_std":
                    kick_dir_params.append(p)
                    kick_dir_param_ids.add(id(p))
            direction_param_ids = move_dir_param_ids | kick_dir_param_ids
            self.direction_param_ids = direction_param_ids
            policy_params = [
                p for p in list(decision_net.parameters()) + list(execution_net.parameters())
                if id(p) not in value_param_ids and id(p) not in direction_param_ids
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
        # Set of decision-head module names (matching ``decision_net`` attribute names,
        # same as the ``frozen_head_names`` argument to ``set_frozen_heads()``) that
        # are excluded from the PPO log_prob computation (both at sample time in
        # ``_compute_log_prob`` and at update time in ``_recompute_log_prob``).
        # Empty = no masking (default).  Populated by ``set_frozen_heads()`` so that
        # any head frozen for a curriculum phase is also silently dropped from the
        # importance ratio — they cancel exactly when frozen, but explicit masking
        # removes the noise when they are NOT frozen and avoids misleading KL
        # diagnostics from heads that carry no reward signal for the current phase.
        self._ppo_lp_masked_heads: frozenset[str] = frozenset()

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

    def _value_heads(self, sf, of, em, bf, gf, d_heads, sat, oat):
        """Return the ExecutionHeadsRaw whose .value is THE critic estimate.

        When ``separate_value_net`` is disabled (default), this is just
        ``self.execution_net(...)`` (the normal shared-trunk forward pass,
        already computed by the caller in nearly every call site — this
        method exists so call sites that only need value can go through
        one path). When enabled, forwards through the dedicated
        ``self.value_net`` instead — a fully independent ExecutionNetwork
        with its own trunk/encoders, never touched by BC losses. Both take
        identical inputs (same decision_heads too), so this slots in as a
        drop-in replacement for ``self.execution_net(...)`` wherever only
        the returned ``.value`` field is actually used downstream.
        """
        if self.separate_value_net:
            return self.value_net(sf, of, em, bf, gf, d_heads, sat, oat)
        return self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat)

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

    def set_frozen_heads(self, frozen_head_names: list[str]) -> None:
        """Freeze named decision-network heads so PPO gradients skip them.

        Sets ``requires_grad=False`` on the parameters of each named
        ``nn.Module`` attribute on ``decision_net``.  The Adam optimizer
        already holds references to these params; it simply won't step them
        because they receive no gradient.

        BC pre-training runs its own optimizer over all parameters and is
        unaffected — freezing only applies during the PPO update loop.

        Also updates ``_ppo_lp_masked_heads`` so these heads are excluded
        from the PPO importance-ratio log_prob (both at sample time and at
        update time).  Frozen heads cancel exactly in the ratio when their
        parameters are fixed, but explicit masking removes noisy log_prob
        contributions from heads that carry no reward signal for this phase
        and avoids misleading per-head KL diagnostics.

        Args:
            frozen_head_names: Names of ``nn.Module`` attributes on
                ``decision_net``, e.g. ``["shoot_logit", "pass_logit"]``.
        """
        # Decision-head Bernoulli names that participate in _compute_log_prob /
        # _recompute_log_prob.  Only these need log_prob masking; target
        # categoricals (pass_target_logits etc.) are already gated by their
        # parent Bernoulli so they are implicitly handled.
        _LP_HEADS = {
            "shoot_logit", "pass_logit", "move_logit", "tackle_logit",
            "get_possession_raw", "mark_logit", "hold_position_logit",
        }
        newly_masked = []
        for name in frozen_head_names:
            module = getattr(self.decision_net, name, None)
            if module is None:
                log.warning(f"set_frozen_heads: '{name}' not found on decision_net — skipping")
                continue
            for p in module.parameters():
                p.requires_grad_(False)
            log.info(f"Frozen decision_net.{name}")
            if name in _LP_HEADS:
                newly_masked.append(name)
        if newly_masked:
            self._ppo_lp_masked_heads = frozenset(newly_masked)
            log.warning(
                "PPO log_prob masking ACTIVE — the following decision heads are excluded "
                "from the importance ratio (frozen for this curriculum phase, no reward "
                "signal): %s.  Their BC aux loss is still computed normally.",
                ", ".join(sorted(newly_masked)),
            )

    # -----------------------------------------------------------------------
    # Main training entry point
    # -----------------------------------------------------------------------

    def train(self, env, total_steps: int, bc_label_fn=None, phase_id: Optional[int] = None) -> None:
        """Run PPO for ``total_steps`` decision-interval steps.

        Args:
            env: ScenarioEnv (or any env with reset()/step() returning
                 ObservationBatch, float, bool, info). Ignored (may be None)
                 when ``ppo.n_parallel_envs > 1`` -- each rollout worker
                 builds its own env from ``phase_id`` instead.
            total_steps: Total number of decision steps to train for.
            bc_label_fn: Optional callable ``(env) -> BCLabel``.  When
                provided, a BC supervision label is collected at each step
                and stored in the rollout buffer so it can be used as an
                auxiliary loss during the PPO update (weight controlled by
                ``bc.aux_coeff_start/end`` in ai_config.json).  Both the
                decision network's Bernoulli heads and the execution
                network's move_direction and sprint are supervised.
            phase_id: Curriculum phase id, required only when
                ``ppo.n_parallel_envs > 1`` (each rollout worker rebuilds its
                own env from this id via ``curriculum.envs.build_env``). See
                ai/ppo/rollout_worker.py.
        """
        if self.n_parallel_envs > 1:
            if phase_id is None:
                raise ValueError(
                    "ppo.n_parallel_envs > 1 requires train(phase_id=...) so "
                    "each rollout worker can rebuild its own environment."
                )
            from footballcoach.ai.curriculum.phases import PHASES_BY_ID
            max_episode_s = float(PHASES_BY_ID[phase_id].env_kwargs.get("max_episode_s", 120.0))
            self._train_parallel(total_steps, phase_id, max_episode_s)
            return

        from footballcoach.ai.ppo.bc import BCLabel
        # Inject the sampling function so ScenarioEnv assigns NeuralPlayerAI to
        # the trainee (and secondary players when not in rules-based mode).
        env.sample_action_fn = self._sample_action
        if self.checkpoint_dir is not None and hasattr(env, "match_log_dir"):
            env.match_log_dir = self.checkpoint_dir / "match_logs"

        obs = env.reset()
        buffer = RolloutBuffer()
        steps_this_rollout = 0
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

            # Collect BC label for this step
            bc_label_arr = None
            if bc_label_fn is not None:
                bc_label = bc_label_fn(env)
                bc_label_arr = bc_label.to_array()

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
                    weight=self._secondary_weight,
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

                # Bootstrap value for last state
                with torch.no_grad():
                    last_obs_dict = {k: v.unsqueeze(0).to(self.device)
                                     for k, v in next_obs.to_torch_dict().items()}
                    last_value = self._get_value(last_obs_dict)

                advantages, returns = buffer.compute_gae(self.gamma, self.lam, last_value)
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
                rollout_start = time.perf_counter()

                # Save checkpoint
                if self.checkpoint_dir is not None:
                    self._save_checkpoint(self._total_steps)

                # Quick periodic eval vs rules-based AI (always, regardless of training opponent)
                if self.rollout_eval_trials > 0:
                    self._eval_vs_rules(env.max_episode_s)

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
        bc_str = (
            f"  bc={metrics['bc_loss']:.4f}(x{metrics['bc_coeff']:.2f})[tk={_bc_tk:.3f}]"
            if metrics.get("bc_coeff", 0.0) > 0.0 else ""
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
        mv_ls_grad = metrics.get('mv_ls_grad', 0.0)
        # Effective sigma (exp(log_std)) in both raw units and approx degrees
        # of angular std, since log_std alone isn't very human-readable and
        # this is the number that actually controls how tightly direction
        # samples cluster around the predicted mean (see ai_trainer_knowledge.md
        # "Direction heads: log_std and KL"). Delta vs the previous rollout's
        # value shows whether log_std is actually moving at all (it was
        # observed to sit frozen at its init value across many rollouts when
        # the policy LR is tiny and early-stop cuts gradient steps short).
        def _sigma_deg(ls_pair):
            if not ls_pair:
                return None
            sig = [math.exp(v) for v in ls_pair]
            deg = [math.degrees(s) for s in sig]
            return sig, deg
        _mv_sig = _sigma_deg(mv_ls)
        _kk_sig = _sigma_deg(kk_ls)
        _prev_mv_ls = self._prev_move_log_std if hasattr(self, "_prev_move_log_std") else None
        _prev_kk_ls = self._prev_kick_log_std if hasattr(self, "_prev_kick_log_std") else None
        _mv_delta_str = ""
        if mv_ls and _prev_mv_ls:
            _d = [b - a for a, b in zip(_prev_mv_ls, mv_ls)]
            # Show Δlog_std and the resulting change in σ expressed in degrees
            _mv_dstd_deg = [
                abs(math.degrees(math.exp(b)) - math.degrees(math.exp(a)))
                for a, b in zip(_prev_mv_ls, mv_ls)
            ]
            _mv_delta_str = (
                f"  d_move=[{','.join(f'{v:+.4f}' for v in _d)}]"
                f" (Δσ≈{','.join(f'{v:.3f}°' for v in _mv_dstd_deg)})"
            )
        _kk_delta_str = ""
        if kk_ls and _prev_kk_ls:
            _d = [b - a for a, b in zip(_prev_kk_ls, kk_ls)]
            _kk_dstd_deg = [
                abs(math.degrees(math.exp(b)) - math.degrees(math.exp(a)))
                for a, b in zip(_prev_kk_ls, kk_ls)
            ]
            _kk_delta_str = (
                f"  d_kick=[{','.join(f'{v:+.4f}' for v in _d)}]"
                f" (Δσ≈{','.join(f'{v:.3f}°' for v in _kk_dstd_deg)})"
            )
        self._prev_move_log_std = list(mv_ls) if mv_ls else None
        self._prev_kick_log_std = list(kk_ls) if kk_ls else None
        mv_ls_str = ""
        if mv_ls:
            mv_ls_str = f"  mv_ls=[{','.join(f'{v:.4f}' for v in mv_ls)}]"
            if _mv_sig:
                _sig, _deg = _mv_sig
                mv_ls_str += f" (\u03c3\u2248{','.join(f'{s:.2f}' for s in _sig)}, \u2248{','.join(f'{d:.0f}\u00b0' for d in _deg)})"
            mv_ls_str += f" g={mv_ls_grad:.2e}" + _mv_delta_str
        if kk_ls:
            mv_ls_str += f"\n  kk_ls=[{','.join(f'{v:.4f}' for v in kk_ls)}]"
            if _kk_sig:
                _sig, _deg = _kk_sig
                mv_ls_str += f" (\u03c3\u2248{','.join(f'{s:.2f}' for s in _sig)}, \u2248{','.join(f'{d:.0f}\u00b0' for d in _deg)})"
            mv_ls_str += _kk_delta_str
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
                    _lbl = _lbl_map.get(_k, _k)
                    _rew_stats_lines.append(
                        f"  {_lbl:<{_col_w}}  {_arr.mean():>+8.3f}  {_arr.std():>7.3f}"
                        f"  {_arr.min():>+8.3f}  {_arr.max():>+8.3f}"
                    )

        _val_diag_str = (
            f"V={metrics['values_mean']:.2f}\u00b1{metrics['values_std']:.2f}  "
            f"R={metrics['returns_mean']:.2f}\u00b1{metrics['returns_std']:.2f}  "
            f"adv={metrics['adv_mean']:.2f}\u00b1{metrics['adv_std']:.2f}"
        )
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
            f"value={metrics['value_loss']:.4f}(x{self.vf_coef})={self.vf_coef * metrics['value_loss']:.4f}",
            f"           entropy={metrics['entropy']:.4f}  kl={metrics['approx_kl']:.4f}"
            + (f"  {bc_str.strip()}" if bc_str else ""),
            f"  value    {_val_diag_str}",
        ]
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
        """Multi-process rollout collection path (``ppo.n_parallel_envs > 1``).

        Each worker runs its own full env + local policy copy (see
        ai/ppo/rollout_worker.py) — no batched/centralized inference, no
        change to the single-process sampling code. Per-worker GAE is
        computed independently (each worker returns its own trailing
        bootstrap value) before batches are concatenated, since concatenating
        raw transitions across worker boundaries first would corrupt
        advantage estimates by treating unrelated episodes as one trajectory.
        """
        from footballcoach.ai.ppo.rollout_worker import spawn_workers, close_workers

        n_workers = self.n_parallel_envs
        steps_per_worker = max(1, self.rollout_steps // n_workers)
        base_seed = random.randint(0, 2**31 - 1)
        log.info(
            f"PPO parallel training started: {n_workers} worker(s), "
            f"~{steps_per_worker} steps/worker/rollout, "
            f"steps_so_far={self._total_steps:,}  target={self._total_steps + total_steps:,}"
        )
        workers = spawn_workers(phase_id, n_workers, base_seed, self.separate_value_net)
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
        finally:
            close_workers(workers)

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

    def _eval_vs_opponent_type(self, max_episode_s: float, use_rules_ai: bool) -> None:
        _label = "rules" if use_rules_ai else "immobile"
        try:
            if self._eval_n_parallel_workers > 1:
                # Parallel path: each subprocess rebuilds its own trainer
                # from these state dicts (see _eval_worker_factory) -- must
                # snapshot weights now, not capture self._sample_action,
                # since bound methods/live nn.Modules aren't picklable.
                import functools
                _decision_state = self.decision_net.state_dict()
                _execution_state = self.execution_net.state_dict()
                _value_state = self.value_net.state_dict() if self.value_net is not None else None
                worker_factory = functools.partial(
                    _eval_worker_factory, _decision_state, _execution_state,
                    self.separate_value_net, _value_state, use_rules_ai, max_episode_s,
                )
                result = run_seeded_evaluation_parallel(
                    worker_factory, self._eval_seeds, self._eval_repeats_per_seed,
                    n_workers=self._eval_n_parallel_workers,
                )
            else:
                from footballcoach.rules_ai import Phase1RulesAI
                from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition
                from footballcoach.ai.env.scenario_env import ScenarioEnv

                def _eval_env_factory(seed: int) -> ScenarioEnv:
                    def _eval_build(*_a, **_kw):
                        _m = build_1v1_scenario(*_a, seed=seed, **_kw)
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

                result = run_seeded_evaluation(
                    _eval_env_factory, self._sample_action,
                    self._eval_seeds, self._eval_repeats_per_seed,
                )
            log.info(
                f"  [eval vs {_label}] step={self._total_steps:,}  "
                f"seeds={len(self._eval_seeds)}x{self._eval_repeats_per_seed}  "
                f"win={result.win_rate_pct:.0f}%  "
                f"mean_rew={result.mean_reward:.3f}±{result.std_reward:.3f} "
                f"(sem={result.sem_reward:.3f})  "
                f"V={result.mean_value_pred:.3f}  gap={result.mean_value_pred - result.mean_reward:+.3f}  "
                f"outcomes={result.outcomes}"
            )
        except Exception as _e:
            log.warning(f"  [eval vs {_label}] failed: {_e}")
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
        """
        from footballcoach.ai.ppo.bc import bc_loss_from_tensor, compute_bc_loss_floor
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

        bc_opt = torch.optim.Adam(
            list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
            lr=bc_lr, eps=1e-5,
        )
        # NOTE: the value-only optimizer used to be built here, but Phase 2/3's
        # value warm-up now delegates to pretrain_value(), which builds its own
        # internal value_opt over decision_net.value_head + execution_net.value_head
        # (identical param set) — see the pretrain_value() call further below.

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
                _value_opt = self.value_net_optimizer
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

            def _eval_p0_val_loss() -> tuple[float, float, float]:
                """Combined dec_bc + value MSE on the held-out val rows (no grad).
                Returns (combined, bc_adj, val_mse) where bc_adj = bc - floor."""
                self.decision_net.eval()
                _v_losses: list[float] = []
                _v_bc_losses: list[float] = []
                _v_mse_losses: list[float] = []
                _v_floors: list[float] = []
                with torch.no_grad():
                    for _obs_v, _lbl_v, _ret_v in dataset.iterate_minibatches(
                        batch_size=batch_size, shuffle=False, device=self.device,
                        indices_override=_p0_val_idx, returns=demo_returns,
                    ):
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
                        _e_v = self._value_heads(
                            _obs_v["self_feat"], _obs_v["other_feat"],
                            _obs_v["exists_mask"], _obs_v["ball_feat"], _obs_v["global_feat"],
                            _d_v, _sat_v, _oat_v,
                        )
                        _mse_v = F.mse_loss(_e_v.value.squeeze(-1), _ret_v) / (ret_std ** 2)
                        _v_losses.append((_bc_v + self._phase0_value_coef * _mse_v).item())
                        _v_bc_losses.append(_bc_v.item())
                        _v_mse_losses.append(_mse_v.item())
                        _v_floors.append(_floor_v)
                self.decision_net.train()
                _combined = float(np.mean(_v_losses)) if _v_losses else float("nan")
                _bc_mean = float(np.mean(_v_bc_losses)) if _v_bc_losses else float("nan")
                _floor_mean = float(np.mean(_v_floors)) if _v_floors else 0.0
                _mse_mean = float(np.mean(_v_mse_losses)) if _v_mse_losses else float("nan")
                return _combined, _bc_mean - _floor_mean, _mse_mean

            _p0_best_val_loss = float("inf")
            _p0_best_state: Optional[dict] = None
            _p0_patience_ctr = 0
            _p0_stopped_early = False

            for epoch in range(_demo_epochs):
                epoch_losses: list[float] = []
                epoch_bc_losses: list[float] = []
                epoch_val_losses: list[float] = []
                epoch_floors: list[float] = []
                for obs_dict, bc_labels, ret_batch in dataset.iterate_minibatches(
                    batch_size=batch_size, shuffle=True, device=self.device,
                    indices_override=_p0_train_idx, returns=demo_returns,
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
                    e_heads = self._value_heads(
                        obs_dict["self_feat"], obs_dict["other_feat"],
                        obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                        d_heads, _sat, _oat,
                    )
                    val_loss = F.mse_loss(e_heads.value.squeeze(-1), ret_batch) / (ret_std ** 2)
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
                    _p0_vl, _p0_vl_bc_adj, _p0_vl_mse = _eval_p0_val_loss()
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
                self.decision_net.load_state_dict(_p0_best_state["decision_net"])
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

        # --- Phase 1: BC epochs over the dataset ---
        # If dataset has reward data and demo_value_bc_coef > 0, also add a
        # value loss term (MSE against demo returns) in the same backward pass.
        # Disabled when separate_value_net is enabled -- this term trains
        # execution_net.value_head, which is unused/frozen in that mode (see
        # Phase 0 comment above); self.value_net gets its warm-up purely from
        # Phase 0 (demo returns) and pretrain_value()'s MSE-only loop, never
        # mixed with Phase 1's BC gradients.
        _use_joint_val = (
            not self.separate_value_net
            and self._bc_value_coef > 0.0
            and dataset.has_rewards
        )
        if _use_joint_val:
            _joint_returns = dataset.compute_returns(gamma=self._demo_value_pretrain_gamma)
            _joint_ret_std = float(np.std(_joint_returns).clip(1.0))
            log.info(
                f"Phase 1 BC epochs will include joint value loss "
                f"(coef={self._bc_value_coef}, gamma={self._demo_value_pretrain_gamma}, "
                f"returns std={_joint_ret_std:.2f})"
            )
        else:
            _joint_returns = None
            _joint_ret_std = 1.0

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

        for epoch in range(n_epochs):
            _epoch_t0 = time.monotonic()
            bc_losses = []
            bc_floors = []
            val_losses: list[float] = []
            val_raw_mse_losses: list[float] = []
            dir_cosines: list[float] = []
            kick_dir_cosines: list[float] = []
            move_probs: list[float] = []
            sprint_probs: list[float] = []
            kick_probs: list[float] = []
            tackle_attempt_probs: list[float] = []
            _kick_tp = _kick_fp = _kick_fn = 0.0
            _tackle_tp = _tackle_fp = _tackle_fn = 0.0
            _bkdn_acc: dict[str, float] = {}
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
            for mb in dataset.iterate_minibatches(
                batch_size=batch_size, shuffle=True, device=self.device,
                valid_only=True, returns=_joint_returns,
                downsample_trivial_frac=_ds_frac,
                downsample_trivial_cos_threshold=self._downsample_trivial_cos_threshold,
                downsample_trivial_exclude_radius_steps=self._downsample_trivial_exclude_radius_steps,
                indices_override=_bc_train_idx,
            ):
                if _use_joint_val:
                    obs_dict, bc_labels, ret_batch = mb
                else:
                    obs_dict, bc_labels = mb
                    ret_batch = None
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
                total_loss = bc_loss
                if ret_batch is not None:
                    # Single value head: execution_net only (decision_net.value
                    # is frozen — see __init__ note). ret_batch is always None
                    # here when separate_value_net is enabled (_use_joint_val
                    # forces it off above), so this branch never touches
                    # self.value_net.
                    v_exc = e_heads.value.squeeze(-1)
                    val_loss = F.mse_loss(v_exc, ret_batch) / (_joint_ret_std ** 2)
                    total_loss = bc_loss + self._bc_value_coef * val_loss
                    val_losses.append(val_loss.item())
                    # raw MSE for RMSE reporting (values already in raw space)
                    with torch.no_grad():
                        raw_mse = F.mse_loss(v_exc, ret_batch)
                        val_raw_mse_losses.append(raw_mse.item())
                bc_opt.zero_grad()
                total_loss.backward()
                if self._bc_max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                        self._bc_max_grad_norm,
                    )
                bc_opt.step()
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

            mean_cos = float(np.mean(dir_cosines)) if dir_cosines else float('nan')
            mean_kick_cos = float(np.mean(kick_dir_cosines)) if kick_dir_cosines else float('nan')
            mean_mv = float(np.mean(move_probs)) if move_probs else float('nan')
            mean_spr = float(np.mean(sprint_probs)) if sprint_probs else float('nan')
            mean_kk = float(np.mean(kick_probs)) if kick_probs else float('nan')
            mean_tk = float(np.mean(tackle_attempt_probs)) if tackle_attempt_probs else float('nan')
            kk_prec, kk_rec, kk_f1 = _precision_recall_f1(_kick_tp, _kick_fp, _kick_fn)
            tk_prec, tk_rec, tk_f1 = _precision_recall_f1(_tackle_tp, _tackle_fp, _tackle_fn)
            _kick_bkdn_keys = {"kick", "kick_direction", "kick_power", "kick_spin"}
            bkdn_str = "  ".join(
                f"{k}={v/_bkdn_n:.5f}" if k in _kick_bkdn_keys else f"{k}={v/_bkdn_n:.3f}"
                for k, v in _bkdn_acc.items()
            ) if _bkdn_n else ""
            if val_losses:
                val_rmse = float(np.sqrt(np.mean(val_raw_mse_losses))) if val_raw_mse_losses else float('nan')
                _mean_val = np.mean(val_losses)
                val_str = (
                    f"  val_loss={_mean_val:.4f}"
                    f"(x{self._bc_value_coef})={_mean_val * self._bc_value_coef:.4f}"
                    f"  rmse={val_rmse:.2f} (returns std={_joint_ret_std:.1f})"
                )
            else:
                val_str = ""
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
            if bkdn_str:
                _bkdn_parts = bkdn_str.split("  ")
                _mid = (len(_bkdn_parts) + 1) // 2
                _bc_lines.append(f"    breakdown  {'  '.join(_bkdn_parts[:_mid])}")
                if _bkdn_parts[_mid:]:
                    _bc_lines.append(f"               {'  '.join(_bkdn_parts[_mid:])}")
            log.info("\n".join(_bc_lines))
            dir_cosines.clear()
            kick_dir_cosines.clear()
            move_probs.clear()
            sprint_probs.clear()
            _bkdn_acc.clear()
            _bkdn_n = 0

            # --- BC pretrain val loss, reported every epoch; early stop is opt-in
            # (see bc.bc_pretrain_early_stop_patience) ---
            if len(_bc_val_idx) > 0:
                _bc_val_loss = _eval_bc_val_loss()
                _improved = _bc_val_loss < (_bc_best_val_loss - self._bc_pretrain_early_stop_min_delta)
                if _bc_early_stop_enabled:
                    log.info(
                        f"    val        bc_val_loss={_bc_val_loss:.4f}  best={min(_bc_best_val_loss, _bc_val_loss):.4f}"
                        + ("  (improved)" if _improved else f"  (patience {_bc_patience_ctr + 1}/{self._bc_pretrain_early_stop_patience})")
                    )
                else:
                    log.info(f"    val        bc_val_loss={_bc_val_loss:.4f}")
                if _bc_early_stop_enabled and _improved:
                    _bc_best_val_loss = _bc_val_loss
                    _bc_best_state = {
                        "decision_net": copy.deepcopy(self.decision_net.state_dict()),
                        "execution_net": copy.deepcopy(self.execution_net.state_dict()),
                    }
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
        if _bc_stopped_early and _bc_best_state is not None:
            self.decision_net.load_state_dict(_bc_best_state["decision_net"])
            self.execution_net.load_state_dict(_bc_best_state["execution_net"])
            log.info(f"  [BC pretrain] restored best-val weights (bc_val_loss={_bc_best_val_loss:.4f})")
        if bc_losses:
            log.info(f"BC pre-training done ({n_epochs} epoch(s), final bc_loss={np.mean(bc_losses):.4f})")
        else:
            log.info(f"BC pre-training done ({n_epochs} epoch(s) -- no BC epochs ran, dataset/pretrain skipped)")

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

        # --- Phase 4: joint BC repair epoch (all params, bc_lr) ---
        # Runs the BC dataset once more with all parameters trainable to restore
        # any policy quality lost during value-only warm-up, while also keeping
        # the freshly-warmed value head in the training graph.
        repair_epochs = int(self._bc_cfg.get("bc_repair_epochs", 1))
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
                _bkdn_r_n: int = 0
                for obs_dict, bc_labels in dataset.iterate_minibatches(
                    batch_size=batch_size, shuffle=True, device=self.device, valid_only=True
                ):
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
                        region_loss_weight=self._bc_region_loss_w,
                        pos_weight_kick=self._bc_pos_weight_kick,
                        pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                        dec_weight=self._bc_dec_weight,
                        exec_weight=self._bc_exec_weight,
                        dec_label_smoothing=self._bc_dec_label_smoothing,
                        exec_label_smoothing=self._bc_exec_label_smoothing,
                        return_breakdown=True,
                    )
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
                        e_vr = _val_net_r(
                            mb_obs_r["self_feat"], mb_obs_r["other_feat"],
                            mb_obs_r["exists_mask"], mb_obs_r["ball_feat"], mb_obs_r["global_feat"],
                            d_vr, _sat_r, _oat_r,
                        )
                        pred_vr = e_vr.value.squeeze(-1)  # single critic (execution_net, or self.value_net)
                        val_losses_r.append(F.mse_loss(pred_vr, mb_ret_r).item() / (ret_std ** 2).item())
                _kick_bkdn_keys = {"kick", "kick_direction", "kick_power", "kick_spin"}
                bkdn_r_str = "  ".join(
                    f"{k}={v/_bkdn_r_n:.5f}" if k in _kick_bkdn_keys else f"{k}={v/_bkdn_r_n:.3f}"
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
                    _bc_r_lines.append(f"    breakdown  {'  '.join(_bkdn_r_parts[:_mid_r])}")
                    if _bkdn_r_parts[_mid_r:]:
                        _bc_r_lines.append(f"               {'  '.join(_bkdn_r_parts[_mid_r:])}")
                log.info("\n".join(_bc_r_lines))
            log.info(
                f"BC repair done ({repair_epochs} epoch(s), final bc_loss={np.mean(repair_losses):.4f}  "
                f"val_loss={np.mean(val_losses_r):.4f})\n"
                f"  final heads  dir_cos={mean_cos_r:.3f}  kick_dir_cos={mean_kick_cos_r:.3f}  "
                f"move_prob={mean_mv_r:.3f}  sprint_prob={mean_spr_r:.3f}\n"
                f"               kick_prob={mean_kk_r:.3f}  tackle_prob={mean_tk_r:.3f}"
            )

        log.info("Combined pre-training complete.")

    def _collect_value_pretrain_rollout(self, env, n_steps: int, phase_id: Optional[int]) -> tuple[dict, dict]:
        """Collect ``n_steps`` of on-policy experience for value warm-up.

        Returns ``(batch, stats)`` -- ``batch`` is the GAE-processed dict (same
        shape as ``RolloutBuffer.as_tensors()``); ``stats`` has
        ``episode_returns``/``outcomes_vs_rules``/``outcomes_vs_immobile``/
        ``outcomes_vs_neural`` for the caller's return value.

        Single-process when ``ppo.n_parallel_envs == 1`` (uses ``env`` directly,
        exactly the previous inline behaviour). When ``ppo.n_parallel_envs > 1``
        and ``phase_id`` is given, reuses the SAME subprocess workers as the
        main PPO loop (ai/ppo/rollout_worker.py) — no weight sync needed since
        this is called once per pretraining stage, not per rollout; each
        worker's GAE is computed independently before merging, for the same
        reason as ``_train_parallel()`` (concatenating raw transitions across
        worker boundaries before GAE would corrupt advantage estimates).
        """
        if self.n_parallel_envs > 1 and phase_id is not None:
            from footballcoach.ai.ppo.rollout_worker import spawn_workers, close_workers

            n_workers = self.n_parallel_envs
            steps_per_worker = max(1, n_steps // n_workers)
            base_seed = random.randint(0, 2**31 - 1)
            log.info(
                f"  [value pretrain rollout] parallel collection: {n_workers} worker(s), "
                f"~{steps_per_worker} steps/worker"
            )
            workers = spawn_workers(phase_id, n_workers, base_seed, self.separate_value_net)
            try:
                dec_state = self.decision_net.state_dict()
                exec_state = self.execution_net.state_dict()
                val_state = self.value_net.state_dict() if self.value_net is not None else None
                for w in workers:
                    w.set_weights(dec_state, exec_state, val_state)
                for w in workers:
                    w.collect(steps_per_worker, progress=0.0)
                results = [w.recv_result() for w in workers]
            finally:
                close_workers(workers)

            worker_batches = []
            episode_returns: list[float] = []
            outcomes_vs_rules: list[str] = []
            outcomes_vs_immobile: list[str] = []
            outcomes_vs_neural: list[str] = []
            episode_comp_list: list[dict[str, float]] = []
            episode_durations_s: list[float] = []
            n_dropped_total = 0
            for r in results:
                buf = r["buffer"]
                n_dropped_total += buf.truncate_to_last_episode_end()
                # MC returns, not GAE: the value net is untrained/stale here, so
                # bootstrapping off it (GAE) would fit targets that circularly
                # depend on the very net being warm-started (see
                # ai_trainer_knowledge.md "value pretrain MC vs GAE returns").
                advantages = [0.0] * len(buf.rewards)
                returns = buf.compute_mc_returns(self.gamma)
                worker_batches.append(buf.as_tensors(advantages, returns))
                stats = r["stats"]
                episode_returns.extend(stats["episode_rewards"])
                outcomes_vs_rules.extend(stats["episode_outcomes_vs_rules"])
                outcomes_vs_immobile.extend(stats["episode_outcomes_vs_immobile"])
                outcomes_vs_neural.extend(stats["episode_outcomes_vs_neural"])
                episode_comp_list.extend(stats["episode_comp_list"])
                episode_durations_s.extend(stats["episode_durations_s"])
            if n_dropped_total:
                log.info(
                    f"  [value pretrain rollout] dropped {n_dropped_total} trailing "
                    f"(incomplete-episode) step(s) across workers before MC-return fit"
                )
            batch = _merge_worker_batches(worker_batches)
        else:
            env.sample_action_fn = self._sample_action
            env.reset()
            buffer = RolloutBuffer()
            episode_returns = []
            outcomes_vs_rules = []
            outcomes_vs_immobile = []
            outcomes_vs_neural = []
            episode_accum = 0.0
            next_obs = None
            episode_comp_accum: dict[str, float] = {}
            episode_comp_list = []
            episode_durations_s = []

            for _ in range(n_steps):
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

            n_dropped = buffer.truncate_to_last_episode_end()
            if n_dropped:
                log.info(
                    f"  [value pretrain rollout] dropped {n_dropped} trailing "
                    f"(incomplete-episode) step(s) before MC-return fit"
                )
            # MC returns, not GAE: see parallel-path comment above.
            advantages = [0.0] * len(buffer.rewards)
            returns = buffer.compute_mc_returns(self.gamma)
            batch = buffer.as_tensors(advantages, returns)

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
            env: ScenarioEnv. Ignored (may be None) when ``ppo.n_parallel_envs > 1``
                and ``phase_id`` is given -- rollout collection uses the same
                subprocess workers as the main PPO loop instead (see
                ai/ppo/rollout_worker.py).
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
                collection (``ppo.n_parallel_envs > 1``) -- each worker rebuilds
                its own env from this id, same as the main PPO training loop.
                Ignored when ``ppo.n_parallel_envs == 1`` (uses ``env`` directly).

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

        _LIST_KEYS = {"reward_comps_raw", "step_outcomes"}

        def _sel(b: dict, mask: np.ndarray) -> dict:
            idx = torch.from_numpy(np.where(mask)[0]).long()
            idx_list = idx.tolist()
            return {
                k: ([v[i] for i in idx_list] if k in _LIST_KEYS else v[idx])
                for k, v in b.items()
            }

        train_batch_raw = _sel(batch, train_mask)
        val_batch_raw = _sel(batch, val_mask) if n_val_eps > 0 else None

        log.info(
            f"  Value pretrain split: {n_train_eps} train eps ({int(train_mask.sum())} steps)"
            + (f"  |  {n_val_eps} val eps ({int(val_mask.sum())} steps)" if n_val_eps > 0 else "")
        )

        # Augment only the train portion (geometric flips + slot permutations).
        if self.augment_n_slot_shuffles > 0:
            train_batch = augment_batch(train_batch_raw, self.augment_n_slot_shuffles, self._aug_rng)
        else:
            train_batch = train_batch_raw

        # ret_std from all returns for consistent normalisation scale.
        all_returns_t = batch["returns"].to(self.device)
        ret_std = all_returns_t.std().clamp(min=1.0)
        log.debug(f"  [value pretrain] returns: mean={all_returns_t.mean():.2f}  std={ret_std:.2f}"
                  f"  min={all_returns_t.min():.2f}  max={all_returns_t.max():.2f}")

        returns_t = train_batch["returns"].to(self.device)

        # Pre-load val tensors to device once.
        val_returns_t = None
        val_obs_dict = None
        if val_batch_raw is not None:
            val_returns_t = val_batch_raw["returns"].to(self.device)
            val_obs_dict = {k.replace("obs/", ""): val_batch_raw[k].to(self.device)
                            for k in val_batch_raw if k.startswith("obs/")}

        n = len(returns_t)
        mean_loss = float("nan")
        epochs_done = 0
        _best_val_loss = float("inf")
        _best_val_state: Optional[dict] = None
        _patience = 0
        _EARLY_STOP_PATIENCE = self._value_pretrain_early_stop_patience
        _EARLY_STOP_MIN_DELTA = self._value_pretrain_early_stop_min_delta
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
                    e_heads = self.value_net(sf, of, em, bf, gf, d_heads, sat, oat)
                    new_values = e_heads.value.squeeze(-1)

                    value_loss = F.mse_loss(new_values, mb_ret) / (ret_std ** 2)

                    value_opt.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                    value_opt.step()
                    ep_losses.append(value_loss.item())
                else:
                    d_heads = self.decision_net(sf, of, em, bf, gf, sat, oat)
                    e_heads = self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat)
                    new_values = e_heads.value.squeeze(-1)  # single value head (execution_net)

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
                    e_heads_sep = _sep_net(sf, of, em, bf, gf, d_heads_detached, sat, oat)
                    new_values_sep = e_heads_sep.value.squeeze(-1)
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
                    e_v = _val_net(
                        val_obs_dict["self_feat"], val_obs_dict["other_feat"],
                        val_obs_dict["exists_mask"], val_obs_dict["ball_feat"],
                        val_obs_dict["global_feat"], d_v, _sat_v, _oat_v,
                    )
                    _val_preds = e_v.value.squeeze(-1)
                    _vl = float(F.mse_loss(_val_preds, val_returns_t) / (ret_std ** 2))
                    _val_pred_mean = float(_val_preds.mean().item())
                    _val_ret_mean = float(val_returns_t.mean().item())
                    if _sep_net is not None:
                        e_v_sep = _sep_net(
                            val_obs_dict["self_feat"], val_obs_dict["other_feat"],
                            val_obs_dict["exists_mask"], val_obs_dict["ball_feat"],
                            val_obs_dict["global_feat"], d_v, _sat_v, _oat_v,
                        )
                        _vl_sep = float(F.mse_loss(
                            e_v_sep.value.squeeze(-1), val_returns_t
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

    # -----------------------------------------------------------------------
    # Policy sampling
    # -----------------------------------------------------------------------

    def _move_dir_head(self, raw_vec: torch.Tensor, log_std_param: torch.Tensor) -> "DirectionHead":
        return DirectionHead(raw_vec, log_std_param,
                             log_std_min=self.move_dir_log_std_min,
                             log_std_max=self.move_dir_log_std_max)

    def _kick_dir_head(self, raw_vec: torch.Tensor, log_std_param: torch.Tensor) -> "DirectionHead":
        return DirectionHead(raw_vec, log_std_param,
                             log_std_min=self.kick_dir_log_std_min,
                             log_std_max=self.kick_dir_log_std_max)

    def _per_head_new_log_probs(self, d_heads, e_heads, mb_actions: dict, exists_mask) -> torch.Tensor:
        """Per-sample, per-head log_prob under the CURRENT policy for stored
        actions, stacked in ``HEAD_LP_KEYS`` order (shape ``(batch, 13)``).

        Mirrors the gating in ``_recompute_log_prob`` (sprint/move_dir gated
        by exec_move; kick_dir gated by kick) so summing this over heads
        matches the scalar log_prob used for the PPO ratio/KL. Used for the
        per-head KL diagnostic: ``batch["head_log_probs"][mb_idx] - this``,
        averaged over the batch dim, gives a per-head KL breakdown instead
        of just the scalar total.
        """
        def _b(logit, key):
            return IndependentBernoulli(logit).log_prob(mb_actions[key]).squeeze(-1)

        log_std_move = self.execution_net.move_dir_log_std.to(self.device)
        log_std_kick = self.execution_net.kick_dir_log_std.to(self.device)
        exec_move_mask = (mb_actions["exec_move"].squeeze(-1) > 0.5).float()
        kick_mask = (mb_actions["kick"].squeeze(-1) > 0.5).float()

        lp_move_dir = exec_move_mask * (
            self.ent_dir_weight * self._move_dir_head(e_heads.move_direction, log_std_move).log_prob(
                mb_actions["move_dir_raw"]
            )
        )
        lp_kick_dir = kick_mask * (
            self.ent_dir_weight * self._kick_dir_head(e_heads.kick_direction, log_std_kick).log_prob(
                mb_actions["kick_dir_raw"]
            )
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
        ], dim=-1)

    @torch.no_grad()
    def _sample_action(
        self,
        obs_dict: dict,
        deterministic: bool = False,
        deterministic_decision: bool = False,
        deterministic_direction: bool = False,
    ) -> tuple:
        """Forward pass + sample from all distributions.

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
        sf = obs_dict["self_feat"].unsqueeze(0).to(dev)
        of = obs_dict["other_feat"].unsqueeze(0).to(dev)
        em = obs_dict["exists_mask"].unsqueeze(0).to(dev)
        bf = obs_dict["ball_feat"].unsqueeze(0).to(dev)
        gf = obs_dict["global_feat"].unsqueeze(0).to(dev)
        sat = obs_dict["self_ai_type"].unsqueeze(0).to(dev) if "self_ai_type" in obs_dict else None
        oat = obs_dict["other_ai_type"].unsqueeze(0).to(dev) if "other_ai_type" in obs_dict else None

        # Canonical AI frame: mirror world-frame obs so self always attacks
        # +x (see ai/obs/canonical.py). x_sign is reused below to decanonicalize
        # move_direction/kick_direction before they're returned to the caller.
        # (decision_net/execution_net wrap-canonicalize sf/of/bf automatically —
        # see CanonicalNetworkWrapper — so only x_sign itself is needed here.)
        x_sign = float(x_sign_of(sf).item())

        # Decision network forward
        d_heads = self.decision_net(sf, of, em, bf, gf, sat, oat)

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

        # Execution network forward
        e_heads = self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat)

        # Sample execution heads
        exec_move_dist = IndependentBernoulli(e_heads.exec_move_logit)
        sprint_dist = IndependentBernoulli(e_heads.sprint_logit)
        kick_dist = IndependentBernoulli(e_heads.kick_logit)
        tackle_attempt_dist = IndependentBernoulli(e_heads.tackle_attempt_logit)
        exec_move = exec_move_dist.mode() if det_decision else exec_move_dist.sample()
        sprint = sprint_dist.mode() if det_decision else sprint_dist.sample()
        kick = kick_dist.mode() if det_decision else kick_dist.sample()
        tackle_attempt = tackle_attempt_dist.mode() if det_decision else tackle_attempt_dist.sample()

        # Direction heads: sample from Normal(mean, std) per design doc 8.6.
        # We store the noisy raw sample (not the mean) so that log_prob ratios
        # during the PPO update are meaningful — new_mean vs stored sample.
        # In deterministic (direction) mode we use the (normalized) mean direction instead.
        eps = 1e-6
        log_std_move = self.execution_net.move_dir_log_std
        log_std_kick = self.execution_net.kick_dir_log_std
        move_dir_head = self._move_dir_head(e_heads.move_direction, log_std_move)
        kick_dir_head = self._kick_dir_head(e_heads.kick_direction, log_std_kick)
        if det_direction:
            move_dir_raw = move_dir_head.mode_physical()  # (1, 2)
            kick_dir_raw = kick_dir_head.mode_physical()   # (1, 3)
        else:
            move_dir_raw = move_dir_head.sample_raw()  # (1, 2)
            kick_dir_raw = kick_dir_head.sample_raw()   # (1, 3)
        move_dir_phys = (move_dir_raw / (move_dir_raw.norm(dim=-1, keepdim=True) + eps)).squeeze(0)
        kick_dir_phys = (kick_dir_raw / (kick_dir_raw.norm(dim=-1, keepdim=True) + eps)).squeeze(0)

        kick_power_phys = float(torch.sigmoid(e_heads.kick_power))
        kick_spin_raw = e_heads.kick_spin.squeeze(0)

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
        # Single value head: execution_net only (decision_net.value_head is
        # frozen — see __init__ note), OR self.value_net when
        # separate_value_net is enabled (see _value_heads() docstring).
        if self.separate_value_net:
            with torch.no_grad():
                value = float(self.value_net(sf, of, em, bf, gf, d_heads, sat, oat).value.mean())
        else:
            value = float(e_heads.value.mean())
        log_prob = self._compute_log_prob(d_heads, e_heads, {
            "shoot": shoot, "pass_": pass_, "move": move,
            "tackle": tackle, "gp_extra": gp_extra, "mark": mark, "hold": hold,
            "pass_tgt": pass_tgt, "tackle_tgt": tackle_tgt, "mark_tgt": mark_tgt,
            "exec_move": exec_move, "sprint": sprint, "kick": kick,
            "tackle_attempt": tackle_attempt,
            "move_dir_raw": move_dir_raw, "kick_dir_raw": kick_dir_raw,
            "kick_power_raw": e_heads.kick_power,
        }, em)

        # Per-head log_probs for DEBUG KL breakdown (stored alongside total in buffer)
        _lsm = self.execution_net.move_dir_log_std
        _lsk = self.execution_net.kick_dir_log_std
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
            float(self.ent_dir_weight * self._move_dir_head(e_heads.move_direction, _lsm).log_prob(move_dir_raw)) if _exec_move_active else 0.0,
            # kick_dir: only when kick=True
            float(self.ent_dir_weight * self._kick_dir_head(e_heads.kick_direction, _lsk).log_prob(kick_dir_raw)) if _kick_active else 0.0,
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

    def _get_value(self, obs_dict: dict) -> float:
        sf = obs_dict["self_feat"].to(self.device)
        of = obs_dict["other_feat"].to(self.device)
        em = obs_dict["exists_mask"].to(self.device)
        bf = obs_dict["ball_feat"].to(self.device)
        gf = obs_dict["global_feat"].to(self.device)
        sat = obs_dict["self_ai_type"].to(self.device) if "self_ai_type" in obs_dict else None
        oat = obs_dict["other_ai_type"].to(self.device) if "other_ai_type" in obs_dict else None
        d_heads = self.decision_net(sf, of, em, bf, gf, sat, oat)
        e_heads = self._value_heads(sf, of, em, bf, gf, d_heads, sat, oat)
        return float(e_heads.value.mean())  # single critic (execution_net, or self.value_net)

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
        log_std_move = self.execution_net.move_dir_log_std
        log_std_kick = self.execution_net.kick_dir_log_std
        # sprint + move_dir: only when exec_move=True (player was moving)
        if float(samples["exec_move"]) > 0.5:
            lp += IndependentBernoulli(e_heads.sprint_logit).log_prob(samples["sprint"]).sum()
            lp += self.ent_dir_weight * self._move_dir_head(e_heads.move_direction, log_std_move).log_prob(
                samples["move_dir_raw"]
            )
        # kick_dir: only when kick=True (a kick was taken)
        if float(samples["kick"]) > 0.5:
            lp += self.ent_dir_weight * self._kick_dir_head(e_heads.kick_direction, log_std_kick).log_prob(
                samples["kick_dir_raw"]
            )

        return lp

    # -----------------------------------------------------------------------
    # PPO update
    # -----------------------------------------------------------------------

    def _ppo_update(self, batch: dict, progress: float) -> dict:
        """Run N epochs of minibatch PPO updates over the collected rollout.

        Returns dict of mean loss metrics for logging.
        """
        from footballcoach.ai.ppo.bc import bc_loss_from_tensor

        # Augment batch with geometric flips + slot permutations before any
        # gradient computation.  This expands the batch by 4 × n_slot_shuffles.
        if self.augment_n_slot_shuffles > 0:
            batch = augment_batch(batch, self.augment_n_slot_shuffles, self._aug_rng)

        n = len(batch["log_probs"])
        clip = self.schedules.clip(progress)
        lr = self.schedules.lr(progress)
        value_lr = self.schedules.value_lr(progress)
        bc_coeff = self.schedules.bc(progress)
        for pg in self.optimizer.param_groups:
            pg["lr"] = value_lr if pg.get("name") == "value" else lr

        has_bc = "bc_labels" in batch and bc_coeff > 0.0

        # Normalize advantages
        adv = batch["advantages"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        old_log_probs = batch["log_probs"]
        returns = batch["returns"]

        all_policy_loss = []
        all_value_loss = []
        all_entropy = []
        all_kl = []
        all_bc_loss = []
        all_bc_tackle_loss: list[float] = []   # BCE on tackle_attempt head only
        all_tackle_prob: list[float] = []       # mean sigmoid(tackle_attempt_logit) per mb
        all_kick_prob: list[float] = []         # mean sigmoid(kick_logit) per mb
        all_ratios: list[torch.Tensor] = []
        all_mv_log_std_grad: list[float] = []  # grad on move_dir_log_std after each backward
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
        all_continuous_mean_shift: dict[str, list[float]] = {"move_direction": [], "kick_direction": []}
        all_continuous_log_std_shift: dict[str, list[float]] = {"move_direction": [], "kick_direction": []}
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
        # Snapshot the continuous-head log_std at the very start of this update
        # ("before the optimiser step", rollout-wide) so the final summary can
        # report start -> end drift across the whole rollout, not just per-mb.
        _move_ls_start = float(self.execution_net.move_dir_log_std.mean().item())
        _kick_ls_start = float(self.execution_net.kick_dir_log_std.mean().item())

        for epoch_i in range(self.n_epochs):
            epoch_start = time.perf_counter()
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

                d_heads = self.decision_net(sf, of, em, bf, gf, sat, oat)
                e_heads = self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat)

                # Snapshot BEFORE the optimiser step, for the drift diagnostics
                # printed once at the end of _ppo_update (see EXEC_HEAD_MODULES).
                _log_std_move_before = self.execution_net.move_dir_log_std.detach().clone()
                _log_std_kick_before = self.execution_net.kick_dir_log_std.detach().clone()
                _exec_move_logit_before = e_heads.exec_move_logit.detach().clone()
                _sprint_logit_before = e_heads.sprint_logit.detach().clone()
                _kick_logit_before = e_heads.kick_logit.detach().clone()
                _tackle_attempt_logit_before = e_heads.tackle_attempt_logit.detach().clone()

                # Value estimate — single value head (execution_net only), OR the
                # dedicated self.value_net when separate_value_net is enabled. In
                # the latter case d_heads is detached before feeding value_net so
                # its (separately-optimised) value loss never contributes gradient
                # into decision_net -- the whole point of separate_value_net is a
                # critic trunk fully independent of the (BC-primed) policy trunk.
                if self.separate_value_net:
                    d_heads_for_value = _detach_decision_heads(d_heads)
                    e_heads_value = self.value_net(sf, of, em, bf, gf, d_heads_for_value, sat, oat)
                    new_values = e_heads_value.value.squeeze(-1)
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
                        log_std_move = self.execution_net.move_dir_log_std.to(self.device)
                        log_std_kick = self.execution_net.kick_dir_log_std.to(self.device)
                        lp_movedir  = self._move_dir_head(e_heads.move_direction, log_std_move).log_prob(mb_actions["move_dir_raw"]).mean().item()
                        lp_kickdir  = self._kick_dir_head(e_heads.kick_direction, log_std_kick).log_prob(mb_actions["kick_dir_raw"]).mean().item()
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
                        f"    move_dir log_std={self.execution_net.move_dir_log_std.data.tolist()}\n"
                        f"    [move_dir] cur_norm={current_norm.mean():.3f}  stored_norm={stored_norm.mean():.3f}"
                        f"  angle={angle_deg:.1f}deg"
                    )
                else:
                    _diag_done = True  # skip computation when not in DEBUG

                # PPO clipped objective (weighted by per-sample importance weights)
                ratio = torch.exp(new_log_probs - mb_old_lp)
                all_ratios.append(ratio.detach().cpu())
                all_ratio_clipped_frac.append(
                    ((ratio < 1.0 - clip) | (ratio > 1.0 + clip)).float().mean().item()
                )
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * mb_adv
                policy_loss = -(torch.min(surr1, surr2) * mb_w).mean()

                # Value loss — normalise by return variance so it stays ~O(1)
                # regardless of how large/negative the returns are. This keeps
                # the value gradient from overwhelming the policy gradient.
                ret_var = returns.var().clamp(min=1.0)
                value_loss = F.mse_loss(new_values, mb_ret) / ret_var

                # Entropy bonus
                entropy = self._compute_entropy(d_heads, e_heads, em)

                # No dir_l2 penalty needed: direction means are unit-normalized in
                # forward() so their magnitude is always 1 — penalizing it is a no-op.

                # log_std restoring force: without this, ent_dir_weight * entropy
                # is a one-directional force that only ever inflates move_dir_log_std/
                # kick_dir_log_std (entropy is monotonic increasing in log_std), and
                # nothing in the PPO/BC losses pulls it back down (the mean is
                # unit-normalized so dir_l2 above is a no-op on log_std too). This
                # term adds an explicit L2 pull toward dir_log_std_target, independent
                # of clamp (which only caps the value fed into the distribution and
                # zeroes the gradient once the raw parameter drifts past the bound —
                # see ai_trainer_knowledge.md / DirectionHead for the clamp mechanism).
                # Coefficient 0.0 (default) fully disables this — opt-in.
                dir_log_std_reg = torch.zeros(1, device=self.device)
                _lsm_raw = self.execution_net.move_dir_log_std
                _lsk_raw = self.execution_net.kick_dir_log_std
                if self.move_dir_log_std_reg_coef > 0.0:
                    dir_log_std_reg = dir_log_std_reg + self.move_dir_log_std_reg_coef * ((_lsm_raw - self.move_dir_log_std_target) ** 2).mean()
                if self.kick_dir_log_std_reg_coef > 0.0:
                    dir_log_std_reg = dir_log_std_reg + self.kick_dir_log_std_reg_coef * ((_lsk_raw - self.kick_dir_log_std_target) ** 2).mean()

                total_loss = (policy_loss
                              + self.vf_coef * value_loss
                              - self.ent_coef * entropy
                              + dir_log_std_reg)

                # BC auxiliary loss (decision + execution, annealed to 0)
                bc_loss_val = torch.zeros(1, device=self.device)
                if has_bc:
                    mb_bc = batch["bc_labels"][mb_idx].to(self.device)
                    bc_loss_val, _bkdn = bc_loss_from_tensor(
                        mb_bc, d_heads, e_heads,
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
                    total_loss = total_loss + bc_coeff * bc_loss_val
                    # Track tackle_attempt BCE separately for diagnostics
                    all_bc_tackle_loss.append(_bkdn.get("tackle_attempt", 0.0))
                # Track mean tackle/kick activation probability (pre-sampling) for logging.
                with torch.no_grad():
                    all_tackle_prob.append(torch.sigmoid(e_heads.tackle_attempt_logit).mean().item())
                    all_kick_prob.append(torch.sigmoid(e_heads.kick_logit).mean().item())
                all_bc_loss.append(bc_loss_val.detach().item())

                self.optimizer.zero_grad()
                if self.separate_value_net:
                    self.value_net_optimizer.zero_grad()
                total_loss.backward()
                # d_heads_for_value was detached above, so value_loss's gradient
                # (folded into total_loss) only reaches self.value_net's params
                # here -- decision_net/execution_net's policy heads never see it.
                if self.separate_value_net:
                    nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                    self.value_net_optimizer.step()

                # Per-execution-head gradient norm, BEFORE the optimiser step
                # (measurement only — max_norm=inf never rescales, same trick as
                # raw_grad_norm below).
                for _head_name, _attr in EXEC_HEAD_MODULES:
                    _head_params = list(getattr(self.execution_net, _attr).parameters())
                    if _head_params:
                        all_head_grad_norm[_head_name].append(
                            torch.nn.utils.clip_grad_norm_(_head_params, float("inf")).item()
                        )

                # Capture dir_log_std gradient before it is zeroed
                _mv_ls_grad = self.execution_net.move_dir_log_std.grad
                if _mv_ls_grad is not None:
                    all_mv_log_std_grad.append(_mv_ls_grad.norm().item())

                # Grad norm BEFORE clipping
                raw_grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                    float("inf"),  # don't clip yet, just measure
                ).item()
                # Direction-head params (move_direction/kick_direction weights +
                # move_dir_log_std/kick_dir_log_std) are clipped in their own
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
                _gn_main = nn.utils.clip_grad_norm_(_non_direction_params, self.max_grad_norm).item()
                all_grad_norm_main.append(_gn_main)
                if _gn_main > self.max_grad_norm:
                    clip_triggered_main += 1
                if _direction_params:
                    _gn_dir = nn.utils.clip_grad_norm_(_direction_params, self.direction_max_grad_norm).item()
                    all_grad_norm_dir.append(_gn_dir)
                    if _gn_dir > self.direction_max_grad_norm:
                        clip_triggered_dir += 1
                self.optimizer.step()

                # After step: measure KL and direction mean shift
                with torch.no_grad():
                    d_after = self.decision_net(sf, of, em, bf, gf, sat, oat)
                    e_after = self.execution_net(sf, of, em, bf, gf, d_after, sat, oat)
                    lp_after = self._recompute_log_prob(d_after, e_after, mb_actions, em)
                    movedir_mean_shift = (e_after.move_direction - e_heads.move_direction).norm(dim=-1).mean().item()
                    kickdir_mean_shift = (e_after.kick_direction - e_heads.kick_direction).norm(dim=-1).mean().item()
                    # Actual KL contribution from move_direction (now included in ratio).
                    _stored_raw_mb = mb_actions["move_dir_raw"]
                    _log_std_move  = self.execution_net.move_dir_log_std.to(self.device)
                    _lp_movedir_before = self._move_dir_head(e_heads.move_direction, _log_std_move).log_prob(_stored_raw_mb)
                    _lp_movedir_after  = self._move_dir_head(e_after.move_direction, _log_std_move).log_prob(_stored_raw_mb)
                    movedir_hyp_kl = (_lp_movedir_before - _lp_movedir_after).mean().item()

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

                    all_continuous_mean_shift["move_direction"].append(movedir_mean_shift)
                    all_continuous_mean_shift["kick_direction"].append(kickdir_mean_shift)
                    all_continuous_log_std_shift["move_direction"].append(
                        (self.execution_net.move_dir_log_std.detach() - _log_std_move_before).abs().mean().item()
                    )
                    all_continuous_log_std_shift["kick_direction"].append(
                        (self.execution_net.kick_dir_log_std.detach() - _log_std_kick_before).abs().mean().item()
                    )
                    all_discrete_logit_shift["exec_move"].append(
                        (e_after.exec_move_logit - _exec_move_logit_before).abs().mean().item()
                    )
                    all_discrete_logit_shift["sprint"].append(
                        (e_after.sprint_logit - _sprint_logit_before).abs().mean().item()
                    )
                    all_discrete_logit_shift["kick"].append(
                        (e_after.kick_logit - _kick_logit_before).abs().mean().item()
                    )
                    all_discrete_logit_shift["tackle_attempt"].append(
                        (e_after.tackle_attempt_logit - _tackle_attempt_logit_before).abs().mean().item()
                    )

                # Clamp to finite floor before KL to avoid inf from near-zero-probability
                # samples in the current policy (log_prob = -inf → KL = +inf).
                _lp_clamped = new_log_probs.clamp(min=-1e6)
                _la_clamped = lp_after.clamp(min=-1e6)
                approx_kl = (mb_old_lp - _lp_clamped).mean().item()
                kl_after_step = (mb_old_lp - _la_clamped).mean().item()

                mb_i = start // self.minibatch_size
                log.debug(
                    f"  [e{epoch_i} mb{mb_i:02d}]"
                    f"  grad={raw_grad_norm:.1f}"
                    f"  total={total_loss.item():.3f}"
                    f"  pol={policy_loss.item():.3f}"
                    f"  val={value_loss.item():.3f}(x{self.vf_coef})={self.vf_coef * value_loss.item():.3f}"
                    f"  kl={kl_after_step:.4f}"
                    f"  mv_shift={movedir_mean_shift:.3f}"
                    f"  mv_kl={movedir_hyp_kl:.4f}"
                    f"  kk_shift={kickdir_mean_shift:.3f}"
                )

                all_policy_loss.append(policy_loss.item())
                all_value_loss.append(value_loss.item())
                all_entropy.append(entropy.item())
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
            mean_kl_epoch = float(np.mean(all_kl[-32:])) if all_kl else 0.0
            log.debug(f"  [epoch {epoch_i}] kl={mean_kl_epoch:.5f}  t={epoch_times[-1]:.0f}ms")
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
                        d_heads_vo = self.decision_net(sf, of, em, bf, gf, sat, oat)
                    if self.separate_value_net:
                        e_heads_vo = self.value_net(sf, of, em, bf, gf, d_heads_vo, sat, oat)
                    else:
                        e_heads_vo = self.execution_net(sf, of, em, bf, gf, d_heads_vo, sat, oat)
                    new_values_vo = e_heads_vo.value.squeeze(-1)

                    ret_var = returns.var().clamp(min=1.0)
                    value_loss_vo = F.mse_loss(new_values_vo, mb_ret) / ret_var

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

                    d_heads_bo = self.decision_net(sf, of, em, bf, gf, sat, oat)
                    e_heads_bo = self.execution_net(sf, of, em, bf, gf, d_heads_bo, sat, oat)

                    bc_loss_bo, _ = bc_loss_from_tensor(
                        mb_bc, d_heads_bo, e_heads_bo,
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

                    if not bc_loss_bo.requires_grad:
                        # No valid BC rows in this minibatch (bc_loss_from_tensor's
                        # early-return path returns a disconnected zero tensor) —
                        # nothing to backprop, skip this minibatch rather than
                        # crashing on .backward() with no grad_fn.
                        continue

                    self.optimizer.zero_grad()
                    (bc_only_coeff * bc_loss_bo).backward()
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
        move_log_std = self.execution_net.move_dir_log_std.data.tolist()
        kick_log_std = self.execution_net.kick_dir_log_std.data.tolist()
        mean_mv_ls_grad = float(np.mean(all_mv_log_std_grad)) if all_mv_log_std_grad else 0.0
        if mean_kl > KL_DIAG_THRESHOLD and all_ratios:
            ratios_t = torch.cat(all_ratios)
            log.info(
                f"  [KL mean={mean_kl:.4f} median={median_kl:.4f} > {KL_DIAG_THRESHOLD}] ratio percentiles:"
                f"  p5={ratios_t.quantile(0.05):.3f}"
                f"  p25={ratios_t.quantile(0.25):.3f}"
                f"  p50={ratios_t.quantile(0.50):.3f}"
                f"  p75={ratios_t.quantile(0.75):.3f}"
                f"  p95={ratios_t.quantile(0.95):.3f}"
                f"  max={ratios_t.max():.3f}\n"
                f"  move_dir_log_std={move_log_std}  kick_dir_log_std={kick_log_std}"
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
                _lsm = self.execution_net.move_dir_log_std.to(self.device)
                _lsk = self.execution_net.kick_dir_log_std.to(self.device)
                lp_mvdir_d  = self._move_dir_head(e_d.move_direction, _lsm).log_prob(diag_act["move_dir_raw"])
                lp_kkdir_d  = self._kick_dir_head(e_d.kick_direction, _lsk).log_prob(diag_act["kick_dir_raw"])
                # Gate sub-parameter heads by their parent action, matching
                # _compute_log_prob/_recompute_log_prob — otherwise kick_dir/sprint/
                # move_dir noise from never-taken actions (e.g. kick=0 the entire
                # rollout) pollutes diag_new_lp/diag_ratio and the printed means.
                _exec_move_mask_d = (diag_act["exec_move"].squeeze(-1) > 0.5).float()
                _kick_mask_d = (diag_act["kick"].squeeze(-1) > 0.5).float()
                lp_sprint_d = lp_sprint_d * _exec_move_mask_d
                lp_mvdir_d  = lp_mvdir_d * _exec_move_mask_d
                lp_kkdir_d  = lp_kkdir_d * _kick_mask_d
                # NOTE: direction heads (move_dir/kick_dir) are scaled by
                # ent_dir_weight in the REAL training log_prob (see
                # _compute_log_prob/_recompute_log_prob above) — apply the same
                # scaling here so diag_new_lp/diag_ratio (and therefore worst_i)
                # match what actually drove this rollout's KL/early-stop, and so
                # the per-head delta table's units are consistent with the
                # ratio used to pick the top-K worst samples. Previously this
                # used the unweighted lp_mvdir_d/lp_kkdir_d while the per-head
                # breakdown below applied the weight, which silently hid large
                # direction-driven ratios from the delta table (the weighted
                # value could be under the 0.02 display threshold while the
                # unweighted value used for diag_ratio was large).
                lp_mvdir_w  = self.ent_dir_weight * lp_mvdir_d
                lp_kkdir_w  = self.ent_dir_weight * lp_kkdir_d
                diag_new_lp = (lp_shoot_d + lp_pass_d + lp_move_d + lp_tackle_d + lp_gp_d +
                               lp_mark_d + lp_hold_d + lp_sprint_d + lp_kick_d + lp_ta_d +
                               lp_mvdir_w + lp_kkdir_w)
                diag_ratio  = torch.exp(diag_new_lp - diag_old_lp)
                worst_i     = int(diag_ratio.argmax())
                stored_mv   = diag_act["move_dir_raw"][worst_i]
                new_mv_mean = e_d.move_direction[worst_i]
                s_angle     = math.degrees(math.atan2(float(stored_mv[1]),   float(stored_mv[0])))
                n_angle     = math.degrees(math.atan2(float(new_mv_mean[1]), float(new_mv_mean[0])))

                # Per-sample, per-head new-lp stack for KL attribution (which
                # head(s) drive each sample's ratio, not just the aggregate mean).
                # Uses the SAME ent_dir_weight-scaled direction terms as diag_new_lp
                # above so the per-head deltas sum to (approximately) the same
                # total used to compute diag_ratio/worst_i.
                _per_head_new_lp = torch.stack([
                    lp_shoot_d, lp_pass_d, lp_move_d, lp_tackle_d, lp_gp_d, lp_mark_d, lp_hold_d,
                    lp_sprint_d, lp_kick_d, lp_ta_d,
                    lp_mvdir_w, lp_kkdir_w,
                ], dim=-1)  # (diag_n, 12)
                _per_head_names = ["shoot", "pass_", "move", "tackle", "gp_extra", "mark", "hold",
                                    "sprint", "kick", "tackle_attempt", "move_dir", "kick_dir"]
            # Per-head old vs new log_probs: read stored head_log_probs from buffer
            # and compare to what the current policy assigns.  The diff shows which
            # head is driving the KL.
            _new_lp_heads = [
                lp_shoot_d.mean(), lp_pass_d.mean(), lp_move_d.mean(), lp_tackle_d.mean(),
                lp_gp_d.mean(), lp_mark_d.mean(), lp_hold_d.mean(),
                lp_sprint_d.mean(), lp_kick_d.mean(), lp_ta_d.mean(),
                self.ent_dir_weight * lp_mvdir_d.mean(), self.ent_dir_weight * lp_kkdir_d.mean(),
            ]
            _new_lp_heads_map = dict(zip(
                ["shoot","pass_","move","tackle","gp_extra","mark","hold",
                 "sprint","kick","tackle_attempt","move_dir","kick_dir"],
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
        _head_grad_norm_str = "  ".join(
            f"{name}={np.mean(vals):.3f}" for name, vals in all_head_grad_norm.items() if vals
        )
        if _head_grad_norm_str:
            log.info(f"  [exec head grad norm] {_head_grad_norm_str}")
        _move_ls_end = float(self.execution_net.move_dir_log_std.mean().item())
        _kick_ls_end = float(self.execution_net.kick_dir_log_std.mean().item())
        log.info(
            f"  [exec continuous log_std] move_direction: start={_move_ls_start:.4f} end={_move_ls_end:.4f}"
            f"   kick_direction: start={_kick_ls_start:.4f} end={_kick_ls_end:.4f}"
        )
        # Build per-step and per-epoch Δ strings, with angular interpretations.
        # dmean is the per-step L2 shift of the unit-vector mean; for unit vectors
        # |u-v|=d → angle θ = arccos(1 - d²/2). dlog_std is the mean absolute
        # per-step log_std change; angular effect = dσ° = degrees(exp(ls)) change.
        def _angular_dmean_deg(dmean: float) -> float:
            """L2 shift of unit-vector mean → approx angular shift in degrees."""
            cos_theta = max(-1.0, min(1.0, 1.0 - dmean ** 2 / 2.0))
            return math.degrees(math.acos(cos_theta))

        def _angular_dlog_std_deg(dlog_std: float, ls_end: float) -> float:
            """Change in log_std → change in σ expressed in degrees."""
            return abs(math.degrees(math.exp(ls_end)) - math.degrees(math.exp(ls_end - dlog_std)))

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
                f"dlog_std={mean_dlog_std:.5f}  Δσ°={mean_dstd_deg:.3f}/step)"
            )
        if _cont_shift_parts:
            log.info(f"  [exec continuous \u0394 per opt step] {'  '.join(_cont_shift_parts)}")
        _disc_shift_str = "  ".join(
            f"{name}={np.mean(vals):.4f}" for name, vals in all_discrete_logit_shift.items() if vals
        )
        if _disc_shift_str:
            log.info(f"  [exec discrete \u0394logit per opt step] {_disc_shift_str}")
        if all_head_kl:
            _mean_head_kl = torch.stack(all_head_kl).mean(dim=0)
            _head_kl_str = "  ".join(
                f"{k}={v:+.4f}" for k, v in zip(HEAD_LP_KEYS, _mean_head_kl.tolist())
            )
            log.info(f"  [per-head KL] {_head_kl_str}")

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

        return {
            "policy_loss": float(np.mean(all_policy_loss)),
            "value_loss": float(np.mean(all_value_loss)),
            "entropy": float(np.mean(all_entropy)),
            "approx_kl": float(np.mean(all_kl)),
            "bc_loss": float(np.mean(all_bc_loss)),
            "bc_tackle_loss": float(np.mean(all_bc_tackle_loss)) if all_bc_tackle_loss else 0.0,
            "bc_coeff": bc_coeff,
            "epoch_time_ms": float(np.mean(epoch_times)) if epoch_times else 0.0,
            "move_log_std": move_log_std,
            "kick_log_std": kick_log_std,
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
        log_std_move = self.execution_net.move_dir_log_std.to(self.device)
        log_std_kick = self.execution_net.kick_dir_log_std.to(self.device)
        exec_move_mask = (mb_actions["exec_move"].squeeze(-1) > 0.5).float()
        kick_mask = (mb_actions["kick"].squeeze(-1) > 0.5).float()
        lp += exec_move_mask * _b(e_heads.sprint_logit, "sprint")
        lp += exec_move_mask * (
            self.ent_dir_weight * self._move_dir_head(e_heads.move_direction, log_std_move).log_prob(
                mb_actions["move_dir_raw"]
            )
        )
        lp += kick_mask * (
            self.ent_dir_weight * self._kick_dir_head(e_heads.kick_direction, log_std_kick).log_prob(
                mb_actions["kick_dir_raw"]
            )
        )

        return lp

    def _compute_entropy(self, d_heads, e_heads, exists_mask) -> torch.Tensor:
        """Entropy bonus, consistent with the masked log_prob.

        Sub-parameter heads (sprint, move_dir, kick_dir) are weighted by
        E[parent active] to match the masking in _recompute_log_prob where
        those terms are 0 when the parent is not taken.
        """
        ent = torch.zeros(1, device=self.device)
        # Unconditional heads (no parent gate)
        for logit in [
            d_heads.shoot_logit, d_heads.pass_logit, d_heads.move_logit,
            d_heads.tackle_logit, d_heads.get_possession_raw, d_heads.mark_logit,
            d_heads.hold_position_logit, e_heads.exec_move_logit,
            e_heads.kick_logit, e_heads.tackle_attempt_logit,
        ]:
            ent += IndependentBernoulli(logit).entropy().mean()
        for logits in [d_heads.pass_target_logits, d_heads.tackle_target_logits, d_heads.mark_target_logits]:
            ent += MaskedCategorical(logits, exists_mask).entropy().mean()
        # Sub-parameters: scale by E[parent active] to match masked log_prob.
        log_std_move = self.execution_net.move_dir_log_std
        log_std_kick = self.execution_net.kick_dir_log_std
        p_exec_move = torch.sigmoid(e_heads.exec_move_logit).mean()
        p_kick = torch.sigmoid(e_heads.kick_logit).mean()
        ent += p_exec_move * IndependentBernoulli(e_heads.sprint_logit).entropy().mean()
        ent += p_exec_move * self.ent_dir_weight * self._move_dir_head(e_heads.move_direction, log_std_move).entropy().mean()
        ent += p_kick * self.ent_dir_weight * self._kick_dir_head(e_heads.kick_direction, log_std_kick).entropy().mean()
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
        handler = logging.FileHandler(log_path)
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
        # Update latest.pt symlink
        latest = self.checkpoint_dir / "latest.pt"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(path.name)
        log.info(f"Saved checkpoint: {path}")
        self._rotate_log_file()

    def _save_checkpoint_to(self, path: Path) -> None:
        """Save a checkpoint to an explicit path (used for pre-trained snapshot)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "step": self._total_steps,
            "decision_net": self.decision_net.state_dict(),
            "execution_net": self.execution_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.value_net is not None:
            ckpt["value_net"] = self.value_net.state_dict()
            ckpt["value_net_optimizer"] = self.value_net_optimizer.state_dict()
        torch.save(ckpt, path)

    def load_checkpoint(self, path: Path) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.decision_net.load_state_dict(ckpt["decision_net"])
        self.execution_net.load_state_dict(ckpt["execution_net"])
        if self.value_net is not None:
            if "value_net" in ckpt:
                self.value_net.load_state_dict(ckpt["value_net"])
                if self.value_net_optimizer is not None and "value_net_optimizer" in ckpt:
                    self.value_net_optimizer.load_state_dict(ckpt["value_net_optimizer"])
            else:
                log.warning(
                    f"separate_value_net enabled but {path} has no 'value_net' key "
                    "(checkpoint predates this feature) -- value_net keeps its fresh "
                    "random init."
                )
        if self.optimizer is not None and "optimizer" in ckpt:
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
        "move_region_center_raw": action.move_region_center_raw,
        "move_region_size_raw": np.array([action.move_region_size_raw], dtype=np.float32),
        "move_arrival_speed_raw": np.array([action.move_arrival_speed_raw], dtype=np.float32),
    }


def _merge_worker_batches(batches: list[dict]) -> dict:
    """Concatenate per-worker ``RolloutBuffer.as_tensors()`` dicts along dim 0.

    Each worker's batch must already have GAE applied independently (its
    ``advantages``/``returns`` were computed against its OWN trailing
    bootstrap value) -- concatenating raw transitions across worker
    boundaries BEFORE computing GAE would corrupt advantage estimates by
    treating unrelated episodes/workers as one continuous trajectory.
    """
    merged: dict = {}
    for key in batches[0]:
        if key in ("reward_comps_raw", "step_outcomes"):
            merged[key] = [x for b in batches for x in b[key]]
        else:
            merged[key] = torch.cat([b[key] for b in batches], dim=0)
    return merged

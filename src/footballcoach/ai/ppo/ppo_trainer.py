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

from footballcoach.ai.eval.seeded_eval import default_eval_seeds
from footballcoach.ai.models.decision_network import DecisionNetwork
from footballcoach.ai.models.execution_network import ExecutionNetwork
from footballcoach.ai.obs.y_canonical import YCanonicalNetworkWrapper
from footballcoach.ai.progress import ProgressReporter
from footballcoach.ai.obs.canonical import CanonicalNetworkWrapper
from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
from footballcoach.ai.ppo.schedules import TrainingSchedules
from footballcoach.ai.ppo.checkpoint_io import CheckpointIOMixin
from footballcoach.ai.ppo.rollout_logging import RolloutLoggingMixin
from footballcoach.ai.ppo.trainer_eval import TrainerEvalMixin
from footballcoach.ai.ppo.trainer_common import REWARD_COMP_LABELS, log
from footballcoach.ai.ppo.action_sampling import ActionSamplingMixin
from footballcoach.ai.ppo.pretraining import PretrainingMixin
from footballcoach.ai.ppo.trainer_common import _action_to_numpy, _episode_abs_adv_means, _merge_worker_batches
from footballcoach.ai.ppo.ppg import PPGMixin
from footballcoach.ai.ppo.ppo_update import PPOUpdateMixin


















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

    Test helper (no production caller since the sampling tail was
    vectorized): tests/ai_unit/test_sample_action_from_heads_batch.py uses it
    to feed one row of an already-batched ``DecisionHeadsRaw``/
    ``ExecutionHeadsRaw`` to the per-row reference
    ``_sample_action_from_heads``. Reflective
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



























































class PPOTrainer(PPOUpdateMixin, PPGMixin, PretrainingMixin, ActionSamplingMixin, CheckpointIOMixin, RolloutLoggingMixin, TrainerEvalMixin):
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
        # {phase_id: weight} to train several curriculum phases in one run
        # (curriculum/mix.py; set by train.py from --phase-mix or
        # curriculum.training_phase_mix). None = only train()'s phase_id.
        # A mix of more than that one phase needs the batched rollout path.
        self.phase_mix: Optional[dict[int, float]] = None
        # Whether these weights may train on rows that carry task ids (always,
        # since 2026-11-16). True for fresh weights; load_checkpoint sets it from
        # the checkpoint, where only weights saved after init_new_phase_inputs
        # (ai/ppo/new_phase_inputs.py) or trained since carry it. Training
        # refuses False; see _check_task_id_inputs.
        self.task_id_inputs_ready: bool = True
        # Whether these weights expect the true time-remaining input (fresh
        # weights, trained since 2026-11-16, or remapped once by
        # ai/ppo/time_input_remap.py) rather than the old, wrong one. Set from
        # the checkpoint by load_checkpoint; training refuses False.
        self.true_time_input_ready: bool = True
        # (phase name, eval spec name) -> cached result of a rules-only role-phase eval.
        self._role_eval_cache: dict = {}
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



    # -----------------------------------------------------------------------
    # Main training entry point
    # -----------------------------------------------------------------------

    def _check_task_id_inputs(self) -> None:
        """Refuse to train weights on inputs they were not prepared for:
        weights trained on the old time-remaining input (see
        ai/ppo/time_input_remap.py), and weights trained without task ids in
        their rows (their task-id columns are not neutral, see
        ai/ppo/new_phase_inputs.py). Either would silently change how the
        checkpoint plays."""
        if not self.true_time_input_ready:
            raise RuntimeError(
                "these weights were trained on the old time-remaining input; remap them once first "
                "(train.py --remap-time-input or scripts/migrate_checkpoint_inputs.py, "
                "ai/ppo/time_input_remap.py)"
            )
        if not self.task_id_inputs_ready:
            raise RuntimeError(
                "these weights were trained without task ids in their rows; zero their task-id inputs once "
                "first (train.py --init-new-phase-inputs or scripts/migrate_checkpoint_inputs.py, "
                "ai/ppo/new_phase_inputs.py)"
            )

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
        self._check_task_id_inputs()
        _mixed = self.phase_mix is not None and set(self.phase_mix) != {phase_id}
        if _mixed and not (self.n_processes > 1 and self.batched_rollout):
            raise NotImplementedError(
                f"training a phase mix ({self.phase_mix}) needs the batched rollout path "
                "(ppo.n_processes > 1 and ppo.batched_rollout)"
            )
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
        # One phase per env slot for a phase mix (curriculum/mix.py); a single
        # phase is passed as before.
        if self.phase_mix is not None and set(self.phase_mix) != {phase_id}:
            from footballcoach.ai.curriculum.mix import assign_env_phases
            _env_phases = assign_env_phases(self.phase_mix, total_envs)
            log.info(
                "  phase mix " + "  ".join(
                    f"phase{p}={_env_phases.count(p)}/{total_envs} env slots" for p in sorted(set(_env_phases)))
            )
        else:
            _env_phases = phase_id
        workers = spawn_batched_workers(
            _env_phases, n_processes, envs_per_process, base_seed,
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
                episode_outcomes_by_tag: dict[str, list[str]] = {}
                episode_rewards_by_tag: dict[str, list[float]] = {}
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
                    for _tag, _outs in stats.get("episode_outcomes_by_tag", {}).items():
                        episode_outcomes_by_tag.setdefault(_tag, []).extend(_outs)
                    for _tag, _rews in stats.get("episode_rewards_by_tag", {}).items():
                        episode_rewards_by_tag.setdefault(_tag, []).extend(_rews)
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
                    episode_outcomes_by_tag=episode_outcomes_by_tag,
                    episode_rewards_by_tag=episode_rewards_by_tag,
                )

                if self.checkpoint_dir is not None:
                    self._save_checkpoint(self._total_steps)

                if self.rollout_eval_trials > 0:
                    self._eval_vs_rules(max_episode_s)
                    self._eval_role_phases()
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
                        _env_phases, n_processes, envs_per_process, base_seed,
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

                # The PPG refit collects in phase_id's envs only, so it gets
                # only that phase's replay seeds, untagged (curriculum/mix.py);
                # for a phase-1-only run this is the same list.
                from footballcoach.ai.curriculum.mix import untag_seed
                self._maybe_run_ppg_interleave(
                    None, phase_id, _close_main_pools, _respawn_main_pools,
                    replay_seeds=[untag_seed(_s)[0] for _s in replay_seeds_for_next_cycle
                                  if untag_seed(_s)[1] == phase_id],
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








    # HEAD_LP_KEYS column -> opportunity_masks() key for the heads that only carry policy-gradient signal where the
    # action could take effect (agent_plans/masked_action_training_plan.md). kick_spin is permanently frozen
    # (log-prob column always 0) but is listed with its parent params for completeness.
    _OPP_HEAD_MASK_KEY = {
        "kick": "kick", "tackle_attempt": "tackle_attempt",
        "kick_dir": "kick_dir", "kick_power": "kick_power", "kick_spin": "kick_dir",
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









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
from footballcoach.ai.action.schema import DecisionAction, ExecutionAction
from footballcoach.ai.config import load_ai_config
from footballcoach.ai.models.decision_network import DecisionNetwork, derive_get_possession_prob
from footballcoach.ai.models.execution_network import ExecutionNetwork, flatten_decision_heads
from footballcoach.ai.obs.augment import augment_batch, augment_obs_bc
from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
from footballcoach.ai.ppo.schedules import TrainingSchedules

log = logging.getLogger("footballcoach.ai.ppo")

# Reward component short-key → display label mapping (order = display order).
# Used by both the per-rollout log and the pre-training diagnostic in train.py.
REWARD_COMP_LABELS: list[tuple[str, str]] = [
    ("appr",  "approach"),
    ("retr",  "retreat"),
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
    ("stam",  "stamina_penalty"),
]

def _ai_types(obs_dict: dict) -> tuple:
    """Extract (self_ai_type, other_ai_type) tensors from an obs dict, or
    (None, None) if absent — DecisionNetwork/ExecutionNetwork.forward()
    default to all-zero one-hots in that case. Centralised here so every
    ``decision_net(...)``/``execution_net(...)`` call site in this file uses
    the identical fallback behaviour. See ai/knowledge.md "Opponent-AI-type
    (value-only)".
    """
    return obs_dict.get("self_ai_type"), obs_dict.get("other_ai_type")


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
    ):
        self.decision_net = decision_net
        self.execution_net = execution_net
        self.device = device or torch.device("cpu")
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
        self.minibatch_size = int(ppo_cfg.get("minibatch_size", 64))
        self.target_kl = float(ppo_cfg.get("target_kl", 0.02))
        self.rollout_steps = int(ppo_cfg.get("rollout_steps", 2048))
        self.dir_l2_coef = float(ppo_cfg.get("dir_l2_coef", 0.01))
        self.dir_log_std_min = float(ppo_cfg.get("dir_log_std_min", -5.0))
        self.dir_log_std_max = float(ppo_cfg.get("dir_log_std_max", 2.0))
        self.ent_dir_weight = float(ppo_cfg.get("ent_dir_weight", 1.0))
        self.augment_n_slot_shuffles = int(ppo_cfg.get("augment_n_slot_shuffles", 0))
        self.rollout_eval_trials = int(ppo_cfg.get("rollout_eval_trials", 10))
        self._aug_rng = random.Random()
        self._bc_cfg = bc_cfg
        self._bc_dir_loss_w = float(bc_cfg.get("direction_loss_weight", 3.0))
        self._bc_region_loss_w = float(bc_cfg.get("region_loss_weight", 1.0))
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
        self._downsample_trivial_enabled = bool(bc_cfg.get("downsample_trivial_enabled", False))
        self._downsample_trivial_frac_default = float(bc_cfg.get("downsample_trivial_frac_default", 0.5))
        self._downsample_trivial_frac_high_epoch = float(bc_cfg.get("downsample_trivial_frac_high_epoch", 0.65))
        self._downsample_trivial_epoch_threshold = int(bc_cfg.get("downsample_trivial_epoch_threshold", 5))
        self._downsample_trivial_cos_threshold = float(bc_cfg.get("downsample_trivial_cos_threshold", 0.98))
        self._downsample_trivial_exclude_radius_steps = int(bc_cfg.get("downsample_trivial_exclude_radius_steps", 5))
        self._secondary_weight = float(curriculum_cfg.get("secondary_weight", 1.0))
        self._value_pretrain_frozen_layers = int(bc_cfg.get("value_pretrain_frozen_layers", -1))
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
        if not inference_only:
            all_params = list(decision_net.parameters()) + list(execution_net.parameters())
            self.optimizer = torch.optim.Adam(all_params, lr=lr, eps=1e-5)
        else:
            self.optimizer = None  # type: ignore[assignment]  # not needed for inference

        self.decision_net.to(self.device)
        self.execution_net.to(self.device)

        self._total_steps = 0
        self._checkpoint_count = 0  # sequential counter for checkpoint{N}.pt naming

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

    # -----------------------------------------------------------------------
    # Main training entry point
    # -----------------------------------------------------------------------

    def train(self, env, total_steps: int, bc_label_fn=None) -> None:
        """Run PPO for ``total_steps`` decision-interval steps.

        Args:
            env: ScenarioEnv (or any env with reset()/step() returning
                 ObservationBatch, float, bool, info).
            total_steps: Total number of decision steps to train for.
            bc_label_fn: Optional callable ``(env) -> BCLabel``.  When
                provided, a BC supervision label is collected at each step
                and stored in the rollout buffer so it can be used as an
                auxiliary loss during the PPO update (weight controlled by
                ``bc.aux_coeff_start/end`` in ai_config.json).  Both the
                decision network's Bernoulli heads and the execution
                network's move_direction and sprint are supervised.
        """
        from footballcoach.ai.ppo.bc import BCLabel
        # Inject the sampling function so ScenarioEnv assigns NeuralPlayerAI to
        # the trainee (and secondary players when not in rules-based mode).
        env.sample_action_fn = self._sample_action

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

        log.info(f"PPO training started: steps_so_far={self._total_steps:,}  target={self._total_steps + total_steps:,}  (+{total_steps:,} this run)")

        rollout_start = time.perf_counter()

        while self._total_steps < total_steps:
            progress = self._total_steps / total_steps

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
            self._total_steps += 1
            steps_this_rollout += 1

            if done:
                episode_rewards.append(episode_reward_accum)
                episode_reward_accum = 0.0
                if info is not None and info.trial_outcome is not None:
                    if info.is_rules_episode:
                        episode_outcomes_vs_rules.append(info.trial_outcome)
                    elif info.is_immobile_episode:
                        episode_outcomes_vs_immobile.append(info.trial_outcome)
                    else:
                        episode_outcomes_vs_neural.append(info.trial_outcome)
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

                update_start = time.perf_counter()
                metrics = self._ppo_update(batch, progress)
                update_time = time.perf_counter() - update_start

                # Log one clean line per rollout
                mean_ep_reward = (
                    float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0
                )
                mean_opp_reward = (
                    float(np.mean(secondary_episode_rewards[-20:])) if secondary_episode_rewards else float('nan')
                )
                episode_rewards.clear()
                secondary_episode_rewards.clear()
                _bc_tk = metrics.get("bc_tackle_loss", 0.0)
                bc_str = (
                    f"  bc={metrics['bc_loss']:.4f}(x{metrics['bc_coeff']:.2f})[tk={_bc_tk:.3f}]"
                    if metrics.get("bc_coeff", 0.0) > 0.0 else ""
                )
                # Phase-1 outcome win-rate over this rollout's episodes, split by opponent type
                outcome_parts = []
                if episode_outcomes_vs_rules:
                    n = len(episode_outcomes_vs_rules)
                    tw = episode_outcomes_vs_rules.count("box_possession")
                    ow = episode_outcomes_vs_rules.count("opponent_box_possession")
                    outcome_parts.append(f"vs_rules({n}): {tw/n*100:.0f}%/{ow/n*100:.0f}%")
                    episode_outcomes_vs_rules.clear()
                if episode_outcomes_vs_immobile:
                    n = len(episode_outcomes_vs_immobile)
                    tw = episode_outcomes_vs_immobile.count("box_possession")
                    outcome_parts.append(f"vs_immobile({n}): {tw/n*100:.0f}%")
                    episode_outcomes_vs_immobile.clear()
                if episode_outcomes_vs_neural:
                    n = len(episode_outcomes_vs_neural)
                    tw = episode_outcomes_vs_neural.count("box_possession")
                    ow = episode_outcomes_vs_neural.count("opponent_box_possession")
                    outcome_parts.append(f"vs_neural({n}): {tw/n*100:.0f}%/{ow/n*100:.0f}%")
                    episode_outcomes_vs_neural.clear()
                outcome_str = ("  " + "  ".join(outcome_parts)) if outcome_parts else ""
                mv_ls = metrics.get('move_log_std', [])
                mv_ls_grad = metrics.get('mv_ls_grad', 0.0)
                mv_ls_str = (f"  mv_ls=[{','.join(f'{v:.4f}' for v in mv_ls)}] g={mv_ls_grad:.2e}") if mv_ls else ""
                ha = metrics.get("head_act", {})
                _ta_p = ha.get('ta_p', float('nan'))
                _kk_p = ha.get('kk_p', float('nan'))
                _prob_str = (
                    (f" ta_p={_ta_p:.2f}" if _ta_p == _ta_p else "")
                    + (f" kk_p={_kk_p:.2f}" if _kk_p == _kk_p else "")
                )
                act_str = (
                    f"  act: mv={ha.get('mv','?'):>3} gp={ha.get('gp','?'):>3}"
                    f" emv={ha.get('emv','?'):>3} spr={ha.get('spr','?'):>3}"
                    f" kck={ha.get('kck','?'):>3} tk={ha.get('tk','?'):>3}"
                    f" sh={ha.get('sh','?'):>3} hld={ha.get('hld','?'):>3}"
                    + _prob_str
                ) if ha else ""
                opp_rew_str = f"/{mean_opp_reward:.2f}" if not (mean_opp_reward != mean_opp_reward) else ""
                comp_parts = [
                    f"{label}={rollout_components[k]:+.2f}"
                    for k, label in REWARD_COMP_LABELS
                    if abs(rollout_components.get(k, 0.0)) > 0.01
                ]
                comp_str = ("  rew: " + "  ".join(comp_parts)) if comp_parts else ""
                rollout_components.clear()
                log.info(
                    f"step={self._total_steps:,} | "
                    f"rew={mean_ep_reward:.2f}{opp_rew_str} | "
                    f"pol={metrics['policy_loss']:.4f} "
                    f"val={metrics['value_loss']:.4f}(x{self.vf_coef})={self.vf_coef * metrics['value_loss']:.4f} "
                    f"ent={metrics['entropy']:.4f} "
                    f"kl={metrics['approx_kl']:.4f}"
                    f"{bc_str} | "
                    f"{steps_per_sec:.0f}sps"
                    f"{mv_ls_str}"
                    f"{act_str}"
                    f"{outcome_str}"
                    f"{comp_str}"
                )

                buffer.clear()
                steps_this_rollout = 0
                rollout_start = time.perf_counter()

                # Save checkpoint
                if self.checkpoint_dir is not None:
                    self._save_checkpoint(self._total_steps)

                # Quick periodic eval vs rules-based AI (always, regardless of training opponent)
                if self.rollout_eval_trials <= 0:
                    continue
                try:
                    from footballcoach.rules_ai import Phase1RulesAI
                    from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition
                    from footballcoach.ai.env.scenario_env import ScenarioEnv

                    def _eval_rules_build(*_a, **_kw):
                        _m = build_1v1_scenario(*_a, **_kw)
                        _m.player_by_id("opponent").ai = Phase1RulesAI()
                        _m._opponent_use_rules_ai = True
                        _m._opponent_is_immobile = False
                        return _m

                    _eval_env = ScenarioEnv(
                        ScenarioDefinition(key="_eval_rules", label="eval_rules",
                                           description="periodic rules eval", build=_eval_rules_build),
                        trainee_player_id="trainee",
                        max_episode_s=env.max_episode_s,
                    )
                    _eval_env.sample_action_fn = self._sample_action
                    _eval_n = self.rollout_eval_trials
                    _eval_wins = 0
                    _eval_outcomes: dict[str, int] = {}
                    for _ in range(_eval_n):
                        _eval_env.reset()
                        _eval_done = False
                        _eval_info = None
                        while not _eval_done:
                            _, _, _eval_done, _eval_info = _eval_env.step()
                        _oc = _eval_info.trial_outcome if _eval_info else "unknown"
                        _eval_outcomes[_oc] = _eval_outcomes.get(_oc, 0) + 1
                        if _oc == "box_possession":
                            _eval_wins += 1
                    log.info(
                        f"  [eval vs rules] step={self._total_steps:,}  "
                        f"win={_eval_wins}/{_eval_n} ({_eval_wins/_eval_n*100:.0f}%)  "
                        f"outcomes={_eval_outcomes}"
                    )
                except Exception as _e:
                    log.warning(f"  [eval vs rules] failed: {_e}")

        # Always save a final checkpoint so the result of the run is not lost
        # even if total_steps is not an exact multiple of rollout_steps.
        if self.checkpoint_dir is not None:
            self._save_checkpoint(self._total_steps)
            log.info("Final checkpoint saved.")

        log.info(f"Training complete. Total steps: {self._total_steps:,}")

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
        from footballcoach.ai.ppo.bc import bc_loss_from_tensor
        from footballcoach.ai.bc.dataset import DemonstrationDataset

        # pos_weight_*: auto-compute from this dataset if not overridden in config.
        if self._bc_pos_weight_kick_cfg is None or self._bc_pos_weight_tackle_attempt_cfg is None:
            _auto_weights = dataset.compute_pos_weights()
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
        # Combined decision-heads-only BC loss + value MSE loss, ONE backward pass,
        # over ALL decision_net parameters (encoders + trunk + value_head) — no
        # frozen layers here (unlike pretrain_value()'s standalone freezing; these
        # are two different call sites with two different freezing decisions, see
        # ai/knowledge.md "Phase 0" note). execution_net is NOT trained in this
        # phase; it gets its BC training in Phase 1 below. Uses stored
        # rewards/dones so no env interaction is needed. Skipped if the dataset
        # has no reward data or demo_value_pretrain_epochs=0.
        _demo_epochs = self._demo_value_pretrain_epochs
        if _demo_epochs > 0 and dataset.has_rewards:
            demo_opt = torch.optim.Adam(
                list(self.decision_net.parameters()),
                lr=self._demo_value_pretrain_lr, eps=1e-5,
            )
            demo_returns = dataset.compute_returns(gamma=self._demo_value_pretrain_gamma)
            ret_t_all = torch.from_numpy(demo_returns).to(self.device)
            ret_std = ret_t_all.std().clamp(min=1.0)
            log.info(
                f"Phase 0 — decision-net warm-up (BC + value, combined): {_demo_epochs} epoch(s), "
                f"gamma={self._demo_value_pretrain_gamma}, "
                f"returns mean={ret_t_all.mean():.2f}  std={ret_std:.2f}  "
                f"lr={self._demo_value_pretrain_lr}  phase0_value_coef={self._phase0_value_coef}"
            )
            for epoch in range(_demo_epochs):
                epoch_losses: list[float] = []
                epoch_bc_losses: list[float] = []
                epoch_val_losses: list[float] = []
                raw_mse_losses: list[float] = []
                for obs_dict, bc_labels, ret_batch in dataset.iterate_minibatches(
                    batch_size=batch_size, shuffle=True, device=self.device,
                    valid_only=True, returns=demo_returns,
                ):
                    _sat, _oat = _ai_types(obs_dict)
                    d_heads = self.decision_net(
                        obs_dict["self_feat"], obs_dict["other_feat"],
                        obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                        _sat, _oat,
                    )
                    v_dec = d_heads.value.squeeze(-1)
                    # NOTE: Phase 0 fits decision_net.value ALONE (execution_net
                    # is not run here), unlike Phase 2/3 (pretrain_value) which
                    # average decision_net.value and execution_net.value. This
                    # is an intentional, documented inconsistency — Phase 0 is
                    # scoped as a decision-network-only high-level warm-up.
                    # Value loss uses raw targets + variance normalisation,
                    # matching the value pretrain rollout and PPO loss convention
                    # so the network output scale is consistent throughout.
                    val_loss = F.mse_loss(v_dec, ret_batch) / (ret_std ** 2)
                    dec_bc_loss, _ = bc_loss_from_tensor(
                        bc_labels, d_heads, exec_heads=None,
                        direction_loss_weight=self._bc_dir_loss_w,
                        region_loss_weight=self._bc_region_loss_w,
                        dec_weight=self._bc_dec_weight,
                        return_breakdown=True,
                    )
                    combined = dec_bc_loss + self._phase0_value_coef * val_loss
                    raw_mse = F.mse_loss(v_dec, ret_batch)
                    demo_opt.zero_grad()
                    combined.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.decision_net.parameters()),
                        self.max_grad_norm,
                    )
                    demo_opt.step()
                    epoch_losses.append(combined.item())
                    epoch_bc_losses.append(dec_bc_loss.item())
                    epoch_val_losses.append(val_loss.item())
                    raw_mse_losses.append(raw_mse.item())
                log.info(
                    f"  Phase 0 epoch {epoch + 1}/{_demo_epochs}: "
                    f"loss={np.mean(epoch_losses):.4f}  "
                    f"dec_bc={np.mean(epoch_bc_losses):.4f}  "
                    f"val={np.mean(epoch_val_losses):.4f}"
                    f"(x{self._phase0_value_coef})={np.mean(epoch_val_losses)*self._phase0_value_coef:.4f}  "
                    f"rmse={np.sqrt(np.mean(raw_mse_losses)):.2f} "
                    f"(returns std={ret_std:.1f})"
                )
            log.info(f"Phase 0 done (decision-net warm-up, {_demo_epochs} epoch(s))")
        elif _demo_epochs > 0 and not dataset.has_rewards:
            log.info(
                "Phase 0 skipped — dataset has no reward data "
                "(re-record demonstrations to enable demo value pretrain)"
            )

        # --- Phase 1: BC epochs over the dataset ---
        # If dataset has reward data and demo_value_bc_coef > 0, also add a
        # value loss term (MSE against demo returns) in the same backward pass.
        _use_joint_val = (
            self._bc_value_coef > 0.0
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
        for epoch in range(n_epochs):
            _epoch_t0 = time.monotonic()
            bc_losses = []
            val_losses: list[float] = []
            val_raw_mse_losses: list[float] = []
            dir_cosines: list[float] = []
            move_probs: list[float] = []
            sprint_probs: list[float] = []
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
                        n_aug = 4 * max(1, self.augment_n_slot_shuffles)
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
                bc_loss, bkdn = bc_loss_from_tensor(
                    bc_labels, d_heads, e_heads,
                    direction_loss_weight=self._bc_dir_loss_w,
                    region_loss_weight=self._bc_region_loss_w,
                    pos_weight_kick=self._bc_pos_weight_kick,
                    pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                    dec_weight=self._bc_dec_weight,
                    exec_weight=self._bc_exec_weight,
                    return_breakdown=True,
                )
                total_loss = bc_loss
                if ret_batch is not None:
                    v_dec = d_heads.value.squeeze(-1)
                    v_exc = e_heads.value.squeeze(-1)
                    # Raw targets + variance normalisation — consistent with
                    # value pretrain rollout and PPO. Network outputs raw returns.
                    val_loss = 0.5 * (
                        F.mse_loss(v_dec, ret_batch)
                        + F.mse_loss(v_exc, ret_batch)
                    ) / (_joint_ret_std ** 2)
                    total_loss = bc_loss + self._bc_value_coef * val_loss
                    val_losses.append(val_loss.item())
                    # raw MSE for RMSE reporting (values already in raw space)
                    with torch.no_grad():
                        raw_mse = 0.5 * (
                            F.mse_loss(v_dec, ret_batch)
                            + F.mse_loss(v_exc, ret_batch)
                        )
                        val_raw_mse_losses.append(raw_mse.item())
                bc_opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                    self.max_grad_norm,
                )
                bc_opt.step()
                bc_losses.append(bc_loss.item())
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
                    if valid_mask.any():
                        move_probs.append(torch.sigmoid(e_heads.exec_move_logit.squeeze(-1)[valid_mask]).mean().item())
                        sprint_probs.append(torch.sigmoid(e_heads.sprint_logit.squeeze(-1)[valid_mask]).mean().item())

            mean_cos = float(np.mean(dir_cosines)) if dir_cosines else float('nan')
            mean_mv = float(np.mean(move_probs)) if move_probs else float('nan')
            mean_spr = float(np.mean(sprint_probs)) if sprint_probs else float('nan')
            bkdn_str = "  ".join(f"{k}={v/_bkdn_n:.3f}" for k, v in _bkdn_acc.items()) if _bkdn_n else ""
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
            log.info(
                f"  BC epoch {epoch + 1}/{n_epochs}: bc_loss={np.mean(bc_losses):.4f}"
                + val_str
                + f"  dir_cos={mean_cos:.3f}  mv_p={mean_mv:.3f}  spr_p={mean_spr:.3f}"
                + (f"  [{bkdn_str}]" if bkdn_str else "")
                + f"  ({_epoch_elapsed:.1f}s)"
            )
            dir_cosines.clear()
            move_probs.clear()
            sprint_probs.clear()
            _bkdn_acc.clear()
            _bkdn_n = 0
        log.info(f"BC pre-training done ({n_epochs} epoch(s), final bc_loss={np.mean(bc_losses):.4f})")

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
                post_bc_losses.append(bc_loss_from_tensor(
                    bc_labels, d_check, e_check,
                    pos_weight_kick=self._bc_pos_weight_kick,
                    pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                    dec_weight=self._bc_dec_weight,
                    exec_weight=self._bc_exec_weight,
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
                move_probs_r: list[float] = []
                sprint_probs_r: list[float] = []
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
                    loss_r, bkdn_r = bc_loss_from_tensor(
                        bc_labels, d_r, e_r,
                        direction_loss_weight=self._bc_dir_loss_w,
                        region_loss_weight=self._bc_region_loss_w,
                        pos_weight_kick=self._bc_pos_weight_kick,
                        pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                        dec_weight=self._bc_dec_weight,
                        exec_weight=self._bc_exec_weight,
                        return_breakdown=True,
                    )
                    repair_opt.zero_grad()
                    loss_r.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                        self.max_grad_norm,
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
                        if valid_mask.any():
                            move_probs_r.append(torch.sigmoid(e_r.exec_move_logit.squeeze(-1)[valid_mask]).mean().item())
                            sprint_probs_r.append(torch.sigmoid(e_r.sprint_logit.squeeze(-1)[valid_mask]).mean().item())

                mean_cos_r = float(np.mean(dir_cosines_r)) if dir_cosines_r else float('nan')
                mean_mv_r = float(np.mean(move_probs_r)) if move_probs_r else float('nan')
                mean_spr_r = float(np.mean(sprint_probs_r)) if sprint_probs_r else float('nan')
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
                        e_vr = self.execution_net(
                            mb_obs_r["self_feat"], mb_obs_r["other_feat"],
                            mb_obs_r["exists_mask"], mb_obs_r["ball_feat"], mb_obs_r["global_feat"],
                            d_vr, _sat_r, _oat_r,
                        )
                        pred_vr = ((d_vr.value + e_vr.value) / 2.0).squeeze(-1)
                        val_losses_r.append(F.mse_loss(pred_vr, mb_ret_r).item() / (ret_std ** 2).item())
                bkdn_r_str = "  ".join(f"{k}={v/_bkdn_r_n:.3f}" for k, v in _bkdn_r_acc.items()) if _bkdn_r_n else ""
                log.info(
                    f"  BC repair epoch {epoch + 1}/{repair_epochs}: bc_loss={np.mean(repair_losses):.4f}"
                    f"  dir_cos={mean_cos_r:.3f}  mv_p={mean_mv_r:.3f}  spr_p={mean_spr_r:.3f}"
                    f"  val_loss={np.mean(val_losses_r):.4f}"
                    + (f"  [{bkdn_r_str}]" if bkdn_r_str else "")
                )
            log.info(f"BC repair done ({repair_epochs} epoch(s), final bc_loss={np.mean(repair_losses):.4f}  dir_cos={mean_cos_r:.3f}  mv_p={mean_mv_r:.3f}  spr_p={mean_spr_r:.3f}  val_loss={np.mean(val_losses_r):.4f})")

        log.info("Combined pre-training complete.")

    def pretrain_value(
        self,
        env,
        n_steps: int,
        n_epochs: int,
        lr: float,
        batch_size: Optional[int] = None,
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
            env: ScenarioEnv
            n_steps: Steps to collect (should be >= rollout_steps, e.g. 4096)
            n_epochs: Epochs to fit the value network per collected rollout
            lr: Learning rate for value pre-training (higher than PPO lr, e.g. 1e-3)
            batch_size: Minibatch size. Defaults to ``self.minibatch_size``.

        Returns:
            dict with diagnostic stats from the rollout collection:
            ``{"episode_returns": list[float], "outcomes_vs_rules": list[str],
            "outcomes_vs_immobile": list[str], "outcomes_vs_neural": list[str]}``
        """
        _batch_size = batch_size if batch_size is not None else self.minibatch_size
        log.info(f"Value pre-training: {n_steps} steps, {n_epochs} epochs, lr={lr}")
        # Freeze trunk layers so BC-learned policy weights are not corrupted.
        _freeze_params = self._get_value_pretrain_freeze_params()
        for p in _freeze_params:
            p.requires_grad_(False)
        value_opt = torch.optim.Adam(
            list(self.decision_net.value_head.parameters())
            + list(self.execution_net.value_head.parameters()),
            lr=lr, eps=1e-5,
        )

        env.sample_action_fn = self._sample_action
        env.reset()
        buffer = RolloutBuffer()
        episode_returns: list[float] = []
        outcomes_vs_rules: list[str] = []
        outcomes_vs_immobile: list[str] = []
        outcomes_vs_neural: list[str] = []
        episode_accum = 0.0
        next_obs = None

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
            if done:
                episode_returns.append(episode_accum)
                episode_accum = 0.0
                if info is not None and info.trial_outcome is not None:
                    if info.is_rules_episode:
                        outcomes_vs_rules.append(info.trial_outcome)
                    elif info.is_immobile_episode:
                        outcomes_vs_immobile.append(info.trial_outcome)
                    else:
                        outcomes_vs_neural.append(info.trial_outcome)
                env.reset()

        def _win_frac(outcomes: list[str]) -> float:
            if not outcomes:
                return float('nan')
            return outcomes.count("box_possession") / len(outcomes)

        log.info(
            f"  [value pretrain rollout] mean_return={np.mean(episode_returns) if episode_returns else float('nan'):.2f} "
            f"({len(episode_returns)} episode(s))  "
            f"vs_rules({len(outcomes_vs_rules)}): {_win_frac(outcomes_vs_rules):.0%}  "
            f"vs_immobile({len(outcomes_vs_immobile)}): {_win_frac(outcomes_vs_immobile):.0%}  "
            f"vs_neural({len(outcomes_vs_neural)}): {_win_frac(outcomes_vs_neural):.0%}"
        )

        with torch.no_grad():
            last_obs_dict = {k: v.unsqueeze(0).to(self.device)
                             for k, v in (next_obs or env.reset()).to_torch_dict().items()}
            last_value = self._get_value(last_obs_dict)

        _, returns = buffer.compute_gae(self.gamma, self.lam, last_value)
        batch = buffer.as_tensors(_, returns)

        # --- Episode-level 85/15 train/val split (overfit detection) ---
        # Split by complete episodes so no episode spans both sets.
        dones_arr = np.array(buffer.dones)
        episode_end_idxs = np.where(dones_arr > 0.5)[0]
        n_complete_eps = len(episode_end_idxs)
        n_val_eps = max(1, round(0.15 * n_complete_eps)) if n_complete_eps >= 2 else 0
        n_train_eps = n_complete_eps - n_val_eps
        n_total = len(buffer.dones)
        val_mask = np.zeros(n_total, dtype=bool)
        if n_val_eps > 0:
            ep_starts = np.concatenate([[0], episode_end_idxs[:-1] + 1])
            for _i in range(n_train_eps, n_complete_eps):
                val_mask[ep_starts[_i]:episode_end_idxs[_i] + 1] = True
        train_mask = ~val_mask

        def _sel(b: dict, mask: np.ndarray) -> dict:
            idx = torch.from_numpy(np.where(mask)[0]).long()
            return {k: v[idx] for k, v in b.items()}

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
        _patience = 0
        _EARLY_STOP_PATIENCE = 5
        for ep in range(n_epochs):
            indices = torch.randperm(n)
            ep_losses = []
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

                d_heads = self.decision_net(sf, of, em, bf, gf, sat, oat)
                e_heads = self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat)
                new_values = ((d_heads.value + e_heads.value) / 2.0).squeeze(-1)

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

            mean_loss = float(np.mean(ep_losses))
            epochs_done = ep + 1
            _train_rmse = float(ret_std) * math.sqrt(mean_loss)

            if val_obs_dict is not None and val_returns_t is not None:
                with torch.no_grad():
                    _sat_v, _oat_v = _ai_types(val_obs_dict)
                    d_v = self.decision_net(
                        val_obs_dict["self_feat"], val_obs_dict["other_feat"],
                        val_obs_dict["exists_mask"], val_obs_dict["ball_feat"],
                        val_obs_dict["global_feat"], _sat_v, _oat_v,
                    )
                    e_v = self.execution_net(
                        val_obs_dict["self_feat"], val_obs_dict["other_feat"],
                        val_obs_dict["exists_mask"], val_obs_dict["ball_feat"],
                        val_obs_dict["global_feat"], d_v, _sat_v, _oat_v,
                    )
                    _vl = float(F.mse_loss(
                        ((d_v.value + e_v.value) / 2.0).squeeze(-1), val_returns_t
                    ) / (ret_std ** 2))
                _val_rmse = float(ret_std) * math.sqrt(_vl)
                log.info(
                    f"  Value epoch {epochs_done}/{n_epochs}: "
                    f"train={mean_loss:.4f} rmse={_train_rmse:.2f}  "
                    f"val={_vl:.4f} val_rmse={_val_rmse:.2f} "
                    f"(std={float(ret_std):.1f})"
                )
                if _vl < _best_val_loss - 1e-4:
                    _best_val_loss = _vl
                    _patience = 0
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
                )
        log.info(f"Value pre-training done ({epochs_done} epoch(s), final train_loss={mean_loss:.4f})")
        for p in _freeze_params:
            p.requires_grad_(True)

        return {
            "episode_returns": episode_returns,
            "outcomes_vs_rules": outcomes_vs_rules,
            "outcomes_vs_immobile": outcomes_vs_immobile,
            "outcomes_vs_neural": outcomes_vs_neural,
        }

    # -----------------------------------------------------------------------
    # Policy sampling
    # -----------------------------------------------------------------------

    def _dir_head(self, raw_vec: torch.Tensor, log_std_param: torch.Tensor) -> "DirectionHead":
        """Construct a DirectionHead with config-driven log_std clamp bounds."""
        return DirectionHead(raw_vec, log_std_param,
                             log_std_min=self.dir_log_std_min,
                             log_std_max=self.dir_log_std_max)

    @torch.no_grad()
    def _sample_action(self, obs_dict: dict) -> tuple:
        """Forward pass + sample from all distributions.

        Returns:
            (decision_action, log_prob, value, decision_probs,
             execution_physical, decision_physical, target_slots)
        """
        dev = self.device
        sf = obs_dict["self_feat"].unsqueeze(0).to(dev)
        of = obs_dict["other_feat"].unsqueeze(0).to(dev)
        em = obs_dict["exists_mask"].unsqueeze(0).to(dev)
        bf = obs_dict["ball_feat"].unsqueeze(0).to(dev)
        gf = obs_dict["global_feat"].unsqueeze(0).to(dev)
        sat = obs_dict["self_ai_type"].unsqueeze(0).to(dev) if "self_ai_type" in obs_dict else None
        oat = obs_dict["other_ai_type"].unsqueeze(0).to(dev) if "other_ai_type" in obs_dict else None

        # Decision network forward
        d_heads = self.decision_net(sf, of, em, bf, gf, sat, oat)

        # Sample from each decision head
        shoot = IndependentBernoulli(d_heads.shoot_logit).sample()
        pass_ = IndependentBernoulli(d_heads.pass_logit).sample()
        move = IndependentBernoulli(d_heads.move_logit).sample()
        tackle = IndependentBernoulli(d_heads.tackle_logit).sample()
        gp_extra = IndependentBernoulli(d_heads.get_possession_raw).sample()
        mark = IndependentBernoulli(d_heads.mark_logit).sample()
        hold = IndependentBernoulli(d_heads.hold_position_logit).sample()

        # Categorical targets (masked)
        pass_tgt = MaskedCategorical(d_heads.pass_target_logits, em).sample()
        tackle_tgt = MaskedCategorical(d_heads.tackle_target_logits, em).sample()
        mark_tgt = MaskedCategorical(d_heads.mark_target_logits, em).sample()

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

        decision_physical = {
            "move_region_center_m": mv_center_phys.squeeze(0).cpu().numpy(),
            "move_region_size_m": float(mv_size_phys),
            "move_arrival_speed_mps": mv_speed_phys,
        }

        # Execution network forward
        e_heads = self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat)

        # Sample execution heads
        exec_move = IndependentBernoulli(e_heads.exec_move_logit).sample()
        sprint = IndependentBernoulli(e_heads.sprint_logit).sample()
        kick = IndependentBernoulli(e_heads.kick_logit).sample()
        tackle_attempt = IndependentBernoulli(e_heads.tackle_attempt_logit).sample()

        # Direction heads: sample from Normal(mean, std) per design doc 8.6.
        # We store the noisy raw sample (not the mean) so that log_prob ratios
        # during the PPO update are meaningful — new_mean vs stored sample.
        eps = 1e-6
        log_std_move = self.execution_net.move_dir_log_std
        log_std_kick = self.execution_net.kick_dir_log_std
        move_dir_raw = self._dir_head(e_heads.move_direction, log_std_move).sample_raw()  # (1, 2)
        kick_dir_raw = self._dir_head(e_heads.kick_direction, log_std_kick).sample_raw()   # (1, 2)
        move_dir_phys = (move_dir_raw / (move_dir_raw.norm(dim=-1, keepdim=True) + eps)).squeeze(0)
        kick_dir_phys = (kick_dir_raw / (kick_dir_raw.norm(dim=-1, keepdim=True) + eps)).squeeze(0)

        kick_power_phys = float(torch.sigmoid(e_heads.kick_power))
        kick_spin_raw = e_heads.kick_spin.squeeze(0)

        execution_physical = {
            "exec_move": bool(exec_move.item() > 0.5),
            "move_direction": move_dir_phys.cpu().numpy(),
            "sprint": bool(sprint.item() > 0.5),
            "kick_this_tick": bool(kick.item() > 0.5),
            "kick_direction": kick_dir_phys.cpu().numpy(),
            "kick_power_fraction": kick_power_phys,
            "kick_spin": kick_spin_raw.cpu().numpy(),
            "tackle_attempt": bool(tackle_attempt.item() > 0.5),
        }

        # Combined log_prob
        value = float((d_heads.value + e_heads.value).mean())
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
        _exec_move_active = float(exec_move) > 0.5
        _kick_active = float(kick) > 0.5
        head_log_probs = np.array([
            float(IndependentBernoulli(d_heads.shoot_logit).log_prob(shoot).sum()),
            float(IndependentBernoulli(d_heads.pass_logit).log_prob(pass_).sum()),
            float(IndependentBernoulli(d_heads.move_logit).log_prob(move).sum()),
            float(IndependentBernoulli(d_heads.tackle_logit).log_prob(tackle).sum()),
            float(IndependentBernoulli(d_heads.get_possession_raw).log_prob(gp_extra).sum()),
            float(IndependentBernoulli(d_heads.mark_logit).log_prob(mark).sum()),
            float(IndependentBernoulli(d_heads.hold_position_logit).log_prob(hold).sum()),
            float(IndependentBernoulli(e_heads.exec_move_logit).log_prob(exec_move).sum()),
            # sprint: only when exec_move=True
            float(IndependentBernoulli(e_heads.sprint_logit).log_prob(sprint).sum()) if _exec_move_active else 0.0,
            float(IndependentBernoulli(e_heads.kick_logit).log_prob(kick).sum()),
            float(IndependentBernoulli(e_heads.tackle_attempt_logit).log_prob(tackle_attempt).sum()),
            # move_dir: only when exec_move=True
            float(self.ent_dir_weight * self._dir_head(e_heads.move_direction, _lsm).log_prob(move_dir_raw)) if _exec_move_active else 0.0,
            # kick_dir: only when kick=True
            float(self.ent_dir_weight * self._dir_head(e_heads.kick_direction, _lsk).log_prob(kick_dir_raw)) if _kick_active else 0.0,
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
        e_heads = self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat)
        return float((d_heads.value + e_heads.value).mean())

    def _compute_log_prob(self, d_heads, e_heads, samples: dict, exists_mask) -> torch.Tensor:
        """Compute combined log_prob across all action heads."""
        lp = torch.zeros(1, device=self.device)

        # Bernoulli decision heads
        lp += IndependentBernoulli(d_heads.shoot_logit).log_prob(samples["shoot"]).sum()
        lp += IndependentBernoulli(d_heads.pass_logit).log_prob(samples["pass_"]).sum()
        lp += IndependentBernoulli(d_heads.move_logit).log_prob(samples["move"]).sum()
        lp += IndependentBernoulli(d_heads.tackle_logit).log_prob(samples["tackle"]).sum()
        lp += IndependentBernoulli(d_heads.get_possession_raw).log_prob(samples["gp_extra"]).sum()
        lp += IndependentBernoulli(d_heads.mark_logit).log_prob(samples["mark"]).sum()
        lp += IndependentBernoulli(d_heads.hold_position_logit).log_prob(samples["hold"]).sum()

        # Categorical target heads (gated by intent)
        if samples["pass_"] > 0.5:
            lp += MaskedCategorical(d_heads.pass_target_logits, exists_mask).log_prob(
                samples["pass_tgt"]
            )
        if samples["tackle"] > 0.5:
            lp += MaskedCategorical(d_heads.tackle_target_logits, exists_mask).log_prob(
                samples["tackle_tgt"]
            )
        if samples["mark"] > 0.5:
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
            lp += self.ent_dir_weight * self._dir_head(e_heads.move_direction, log_std_move).log_prob(
                samples["move_dir_raw"]
            )
        # kick_dir: only when kick=True (a kick was taken)
        if float(samples["kick"]) > 0.5:
            lp += self.ent_dir_weight * self._dir_head(e_heads.kick_direction, log_std_kick).log_prob(
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
        bc_coeff = self.schedules.bc(progress)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

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
        epoch_times = []
        KL_DIAG_THRESHOLD = 0.1  # ~5× target_kl; log detailed diagnostics above this

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

                # Value estimate
                new_values = ((d_heads.value + e_heads.value) / 2.0).squeeze(-1)

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
                        lp_movedir  = self._dir_head(e_heads.move_direction, log_std_move).log_prob(mb_actions["move_dir_raw"]).mean().item()
                        lp_kickdir  = self._dir_head(e_heads.kick_direction, log_std_kick).log_prob(mb_actions["kick_dir_raw"]).mean().item()
                        lp_new_mb   = new_log_probs.mean().item()
                        lp_old_mb   = mb_old_lp.mean().item()
                        ratio_mb    = torch.exp(new_log_probs - mb_old_lp)
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
                        f"    new_values={new_values.mean():.3f}  ret(mb)={mb_ret.mean():.3f}\n"
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

                total_loss = (policy_loss
                              + self.vf_coef * value_loss
                              - self.ent_coef * entropy)

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
                total_loss.backward()

                # Capture dir_log_std gradient before it is zeroed
                _mv_ls_grad = self.execution_net.move_dir_log_std.grad
                if _mv_ls_grad is not None:
                    all_mv_log_std_grad.append(_mv_ls_grad.norm().item())

                # Grad norm BEFORE clipping
                raw_grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                    float("inf"),  # don't clip yet, just measure
                ).item()
                nn.utils.clip_grad_norm_(
                    list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                    self.max_grad_norm,
                )
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
                    _lp_movedir_before = self._dir_head(e_heads.move_direction, _log_std_move).log_prob(_stored_raw_mb)
                    _lp_movedir_after  = self._dir_head(e_after.move_direction, _log_std_move).log_prob(_stored_raw_mb)
                    movedir_hyp_kl = (_lp_movedir_before - _lp_movedir_after).mean().item()

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
                    log.info(
                        f"  [early stop e{epoch_i} mb{mb_i_stop}]"
                        f"  KL={kl_after_step:.5f} > target={self.target_kl}"
                        f"  steps_this_update={len(all_kl)}"
                    )
                    _early_stopped = True
                    break

            epoch_times.append((time.perf_counter() - epoch_start) * 1000)
            mean_kl_epoch = float(np.mean(all_kl[-32:])) if all_kl else 0.0
            log.debug(f"  [epoch {epoch_i}] kl={mean_kl_epoch:.5f}  t={epoch_times[-1]:.0f}ms")
            if _early_stopped:
                break

        # --- KL diagnostics (fires whenever rollout KL exceeds threshold) ---
        mean_kl = float(np.mean(all_kl)) if all_kl else 0.0
        move_log_std = self.execution_net.move_dir_log_std.data.tolist()
        kick_log_std = self.execution_net.kick_dir_log_std.data.tolist()
        mean_mv_ls_grad = float(np.mean(all_mv_log_std_grad)) if all_mv_log_std_grad else 0.0
        if mean_kl > KL_DIAG_THRESHOLD and all_ratios:
            ratios_t = torch.cat(all_ratios)
            log.info(
                f"  [KL={mean_kl:.4f} > {KL_DIAG_THRESHOLD}] ratio percentiles:"
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
                _lsm = self.execution_net.move_dir_log_std.to(self.device)
                _lsk = self.execution_net.kick_dir_log_std.to(self.device)
                lp_mvdir_d  = self._dir_head(e_d.move_direction, _lsm).log_prob(diag_act["move_dir_raw"])
                lp_kkdir_d  = self._dir_head(e_d.kick_direction, _lsk).log_prob(diag_act["kick_dir_raw"])
                diag_new_lp = (lp_shoot_d + lp_pass_d + lp_move_d + lp_tackle_d + lp_gp_d +
                               lp_mark_d + lp_hold_d + lp_sprint_d + lp_kick_d + lp_ta_d +
                               lp_mvdir_d + lp_kkdir_d)
                diag_ratio  = torch.exp(diag_new_lp - diag_old_lp)
                worst_i     = int(diag_ratio.argmax())
                stored_mv   = diag_act["move_dir_raw"][worst_i]
                new_mv_mean = e_d.move_direction[worst_i]
                s_angle     = math.degrees(math.atan2(float(stored_mv[1]),   float(stored_mv[0])))
                n_angle     = math.degrees(math.atan2(float(new_mv_mean[1]), float(new_mv_mean[0])))
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
            if "head_log_probs" in batch:
                old_hlp = batch["head_log_probs"][:diag_n].mean(dim=0)  # (13,)
                for _ki, _k in enumerate(_HLK):
                    _new_v = _new_lp_heads_map.get(_k, 0.0)
                    _old_v = float(old_hlp[_ki])
                    _delta = _new_v - _old_v
                    if abs(_delta) > 0.05:
                        _head_lp_delta_str += f" {_k}:{_delta:+.2f}"
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
                f"  old_lp={diag_old_lp[worst_i]:.3f}  new_lp={diag_new_lp[worst_i]:.3f}\n"
                f"    stored move_dir={s_angle:.1f}°  new_mean={n_angle:.1f}°"
                f"  angular_diff={min(abs(s_angle-n_angle), 360-abs(s_angle-n_angle)):.1f}°"
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
        }

    def _recompute_log_prob(self, d_heads, e_heads, mb_actions: dict, exists_mask) -> torch.Tensor:
        """Recompute log_probs for stored actions under the current policy."""
        lp = torch.zeros(exists_mask.shape[0], device=self.device)

        def _b(logit, key):
            return IndependentBernoulli(logit).log_prob(mb_actions[key]).squeeze(-1)

        lp += _b(d_heads.shoot_logit, "shoot")
        lp += _b(d_heads.pass_logit, "pass_")
        lp += _b(d_heads.move_logit, "move")
        lp += _b(d_heads.tackle_logit, "tackle")
        lp += _b(d_heads.get_possession_raw, "get_possession_extra")
        lp += _b(d_heads.mark_logit, "mark")
        lp += _b(d_heads.hold_position_logit, "hold_position")
        lp += _b(e_heads.exec_move_logit, "exec_move")
        lp += _b(e_heads.kick_logit, "kick")
        lp += _b(e_heads.tackle_attempt_logit, "tackle_attempt")

        # Target categorical log_probs (gated by parent intent)
        pass_mask = mb_actions["pass_"].squeeze(-1) > 0.5
        tackle_mask = mb_actions["tackle"].squeeze(-1) > 0.5
        mark_mask = mb_actions["mark"].squeeze(-1) > 0.5

        for mask, logits, key in [
            (pass_mask, d_heads.pass_target_logits, "pass_target"),
            (tackle_mask, d_heads.tackle_target_logits, "tackle_target"),
            (mark_mask, d_heads.mark_target_logits, "mark_target"),
        ]:
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
            self.ent_dir_weight * self._dir_head(e_heads.move_direction, log_std_move).log_prob(
                mb_actions["move_dir_raw"]
            )
        )
        lp += kick_mask * (
            self.ent_dir_weight * self._dir_head(e_heads.kick_direction, log_std_kick).log_prob(
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
        ent += p_exec_move * self.ent_dir_weight * self._dir_head(e_heads.move_direction, log_std_move).entropy().mean()
        ent += p_kick * self.ent_dir_weight * self._dir_head(e_heads.kick_direction, log_std_kick).entropy().mean()
        return ent

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def _save_checkpoint(self, step: int) -> None:
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_count += 1
        path = self.checkpoint_dir / f"checkpoint{self._checkpoint_count}.pt"
        torch.save({
            "step": step,
            "checkpoint_count": self._checkpoint_count,
            "decision_net": self.decision_net.state_dict(),
            "execution_net": self.execution_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        # Update latest.pt symlink
        latest = self.checkpoint_dir / "latest.pt"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(path.name)
        log.info(f"Saved checkpoint: {path}")

    def _save_checkpoint_to(self, path: Path) -> None:
        """Save a checkpoint to an explicit path (used for pre-trained snapshot)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": self._total_steps,
            "decision_net": self.decision_net.state_dict(),
            "execution_net": self.execution_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: Path) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.decision_net.load_state_dict(ckpt["decision_net"])
        self.execution_net.load_state_dict(ckpt["execution_net"])
        if self.optimizer is not None and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self._total_steps = ckpt["step"]
        log.info(f"Loaded checkpoint: {path} (step {self._total_steps})")
        return self._total_steps

    @classmethod
    def from_config(cls, **kwargs) -> "PPOTrainer":
        cfg = load_ai_config()
        decision_net = DecisionNetwork.from_config()
        execution_net = ExecutionNetwork.from_config()
        return cls(decision_net=decision_net, execution_net=execution_net, cfg=cfg, **kwargs)

    @classmethod
    def load_for_inference(cls, path: "Path | str") -> "PPOTrainer":
        """Load networks only — no optimizer created. Safe to call inside pygame/UI."""
        path = Path(path)
        cfg = load_ai_config()
        decision_net = DecisionNetwork.from_config()
        execution_net = ExecutionNetwork.from_config()
        trainer = cls(
            decision_net=decision_net,
            execution_net=execution_net,
            cfg=cfg,
            inference_only=True,
        )
        trainer.load_checkpoint(path)
        trainer.decision_net.eval()
        trainer.execution_net.eval()
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

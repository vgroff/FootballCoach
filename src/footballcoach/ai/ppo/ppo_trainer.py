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
from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
from footballcoach.ai.ppo.schedules import TrainingSchedules

log = logging.getLogger("footballcoach.ai.ppo")


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

        lr = float(ppo_cfg.get("learning_rate", 3e-4))
        all_params = list(decision_net.parameters()) + list(execution_net.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr, eps=1e-5)

        self.decision_net.to(self.device)
        self.execution_net.to(self.device)

        self._total_steps = 0

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
        obs = env.reset()
        buffer = RolloutBuffer()
        steps_this_rollout = 0
        episode_rewards: list[float] = []
        episode_reward_accum = 0.0

        log.info(f"PPO training started: total_steps={total_steps}")

        rollout_start = time.perf_counter()

        while self._total_steps < total_steps:
            progress = self._total_steps / total_steps

            # --- Collect one decision step ---
            obs_dict = {k: v for k, v in obs.to_torch_dict().items()}
            action, log_prob, value, decision_probs, execution_physical, decision_physical, target_slots, raw_exec_samples = (
                self._sample_action(obs_dict)
            )

            # Build the action dict for the env
            env_action = {
                "decision_probs": decision_probs,
                "execution_physical": execution_physical,
                "decision_physical": decision_physical,
                "target_slots": target_slots,
                "slot_player_ids": getattr(env, "_last_slot_player_ids", [None] * 21),
                "decision": action,
                "execution": ExecutionAction(),  # placeholder; gating uses execution_physical
            }

            next_obs, reward, done, info = env.step(env_action)

            # Collect BC label for this step (before resetting obs)
            bc_label_arr = None
            if bc_label_fn is not None:
                bc_label = bc_label_fn(env)
                bc_label_arr = bc_label.to_array()

            # Store transition (as numpy, not torch, for efficient storage)
            obs_numpy = {k: v.numpy() for k, v in obs_dict.items()}
            buffer.add(
                obs=obs_numpy,
                action=_action_to_numpy(action, raw_exec_samples),
                log_prob=float(log_prob),
                value=float(value),
                reward=reward,
                done=1.0 if done else 0.0,
                bc_label=bc_label_arr,
            )

            episode_reward_accum += reward
            self._total_steps += 1
            steps_this_rollout += 1

            if done:
                episode_rewards.append(episode_reward_accum)
                episode_reward_accum = 0.0
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

                # Log
                mean_ep_reward = (
                    float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0
                )
                bc_str = (
                    f" | bc_loss={metrics['bc_loss']:.4f}"
                    f" (coeff={metrics['bc_coeff']:.3f})"
                    if metrics.get("bc_coeff", 0.0) > 0.0 else ""
                )
                log.info(
                    f"step={self._total_steps:,} | "
                    f"ep_reward={mean_ep_reward:.2f} | "
                    f"policy_loss={metrics['policy_loss']:.4f} | "
                    f"value_loss={metrics['value_loss']:.4f} | "
                    f"entropy={metrics['entropy']:.4f} | "
                    f"approx_kl={metrics['approx_kl']:.4f}"
                    f"{bc_str} | "
                    f"rollout={rollout_time:.1f}s | "
                    f"update={update_time:.1f}s ({metrics['epoch_time_ms']:.0f}ms/epoch) | "
                    f"{steps_per_sec:.0f} steps/s"
                )

                buffer.clear()
                steps_this_rollout = 0
                rollout_start = time.perf_counter()

                # Save checkpoint
                if self.checkpoint_dir is not None:
                    self._save_checkpoint(self._total_steps)

        # Always save a final checkpoint so the result of the run is not lost
        # even if total_steps is not an exact multiple of rollout_steps.
        if self.checkpoint_dir is not None:
            self._save_checkpoint(self._total_steps)
            log.info("Final checkpoint saved.")

        log.info("Training complete.")

    # -----------------------------------------------------------------------
    # Value pre-training
    # -----------------------------------------------------------------------

    def pretrain_value(self, env, n_steps: int, n_epochs: int, lr: float) -> None:
        """Warm-start the value heads to predict actual returns before PPO starts.

        Collects n_steps of experience using the current (BC-warm-started) policy,
        computes Monte Carlo returns via GAE, then trains ONLY the value loss for
        n_epochs at the given lr. This prevents the enormous value gradient from
        destroying the policy on the very first PPO update.

        Args:
            env: ScenarioEnv
            n_steps: Steps to collect (should be >= rollout_steps, e.g. 4096)
            n_epochs: Epochs to fit the value network per collected rollout
            lr: Learning rate for value pre-training (higher than PPO lr, e.g. 1e-3)
        """
        log.info(f"Value pre-training: {n_steps} steps, {n_epochs} epochs, lr={lr}")
        value_opt = torch.optim.Adam(
            list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
            lr=lr, eps=1e-5,
        )

        obs = env.reset()
        buffer = RolloutBuffer()
        episode_rewards: list[float] = []
        episode_accum = 0.0

        for _ in range(n_steps):
            obs_dict = {k: v for k, v in obs.to_torch_dict().items()}
            result = self._sample_action(obs_dict)
            action, log_prob, value, decision_probs, exec_phys, dec_phys, target_slots, raw_exec = result

            env_action = {
                "decision_probs": decision_probs,
                "execution_physical": exec_phys,
                "decision_physical": dec_phys,
                "target_slots": target_slots,
                "slot_player_ids": getattr(env, "_last_slot_player_ids", [None] * 21),
                "decision": action,
                "execution": ExecutionAction(),
            }
            next_obs, reward, done, _ = env.step(env_action)

            buffer.add(
                obs={k: v.numpy() for k, v in obs_dict.items()},
                action=_action_to_numpy(action, raw_exec),
                log_prob=float(log_prob),
                value=float(value),
                reward=reward,
                done=1.0 if done else 0.0,
            )
            episode_accum += reward
            if done:
                episode_rewards.append(episode_accum)
                episode_accum = 0.0
                obs = env.reset()
            else:
                obs = next_obs

        with torch.no_grad():
            last_obs_dict = {k: v.unsqueeze(0).to(self.device)
                             for k, v in next_obs.to_torch_dict().items()}
            last_value = self._get_value(last_obs_dict)

        _, returns = buffer.compute_gae(self.gamma, self.lam, last_value)
        batch = buffer.as_tensors(_, returns)

        returns_t = batch["returns"].to(self.device)
        ret_mean = returns_t.mean()
        ret_std = returns_t.std().clamp(min=1.0)

        print(f"  [value pretrain] returns: mean={ret_mean:.2f}  std={ret_std:.2f}"
              f"  min={returns_t.min():.2f}  max={returns_t.max():.2f}")

        n = len(returns_t)
        for ep in range(n_epochs):
            indices = torch.randperm(n)
            ep_losses = []
            for start in range(0, n, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]
                mb_obs = {k.replace("obs/", ""): batch[k][mb_idx].to(self.device)
                          for k in batch if k.startswith("obs/")}
                mb_ret = returns_t[mb_idx]

                sf = mb_obs["self_feat"]
                of = mb_obs["other_feat"]
                em = mb_obs["exists_mask"]
                bf = mb_obs["ball_feat"]
                gf = mb_obs["global_feat"]

                d_heads = self.decision_net(sf, of, em, bf, gf)
                e_heads = self.execution_net(sf, of, em, bf, gf, d_heads)
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
            # Sample predicted value vs actual return for a spot-check
            with torch.no_grad():
                spot_obs = {k.replace("obs/", ""): batch[k][:64].to(self.device)
                            for k in batch if k.startswith("obs/")}
                d_s = self.decision_net(spot_obs["self_feat"], spot_obs["other_feat"],
                                       spot_obs["exists_mask"], spot_obs["ball_feat"],
                                       spot_obs["global_feat"])
                e_s = self.execution_net(spot_obs["self_feat"], spot_obs["other_feat"],
                                        spot_obs["exists_mask"], spot_obs["ball_feat"],
                                        spot_obs["global_feat"], d_s)
                pred_vals = ((d_s.value + e_s.value) / 2.0).squeeze(-1)
            print(f"  [value pretrain epoch {ep:2d}] loss={mean_loss:.4f}"
                  f"  pred_val mean={pred_vals.mean():.2f}"
                  f"  target mean={returns_t[:64].mean():.2f}")

        log.info("Value pre-training complete.")

    # -----------------------------------------------------------------------
    # Policy sampling
    # -----------------------------------------------------------------------

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

        # Decision network forward
        d_heads = self.decision_net(sf, of, em, bf, gf)

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
        e_heads = self.execution_net(sf, of, em, bf, gf, d_heads)

        # Sample execution heads
        sprint = IndependentBernoulli(e_heads.sprint_logit).sample()
        kick = IndependentBernoulli(e_heads.kick_logit).sample()
        tackle_attempt = IndependentBernoulli(e_heads.tackle_attempt_logit).sample()

        # Direction heads: sample from Normal(mean, std) per design doc 8.6.
        # We store the noisy raw sample (not the mean) so that log_prob ratios
        # during the PPO update are meaningful — new_mean vs stored sample.
        eps = 1e-6
        log_std_move = self.execution_net.move_dir_log_std
        log_std_kick = self.execution_net.kick_dir_log_std
        move_dir_raw = DirectionHead(e_heads.move_direction, log_std_move).sample_raw()  # (1, 2)
        kick_dir_raw = DirectionHead(e_heads.kick_direction, log_std_kick).sample_raw()   # (1, 2)
        move_dir_phys = (move_dir_raw / (move_dir_raw.norm(dim=-1, keepdim=True) + eps)).squeeze(0)
        kick_dir_phys = (kick_dir_raw / (kick_dir_raw.norm(dim=-1, keepdim=True) + eps)).squeeze(0)

        kick_power_phys = float(torch.sigmoid(e_heads.kick_power))
        kick_spin_raw = e_heads.kick_spin.squeeze(0)

        execution_physical = {
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
            "sprint": sprint, "kick": kick, "tackle_attempt": tackle_attempt,
            "move_dir_raw": move_dir_raw, "kick_dir_raw": kick_dir_raw,
            "kick_power_raw": e_heads.kick_power,
        }, em)

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
        )

    def _get_value(self, obs_dict: dict) -> float:
        sf = obs_dict["self_feat"].to(self.device)
        of = obs_dict["other_feat"].to(self.device)
        em = obs_dict["exists_mask"].to(self.device)
        bf = obs_dict["ball_feat"].to(self.device)
        gf = obs_dict["global_feat"].to(self.device)
        d_heads = self.decision_net(sf, of, em, bf, gf)
        e_heads = self.execution_net(sf, of, em, bf, gf, d_heads)
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

        # Execution Bernoulli heads
        lp += IndependentBernoulli(e_heads.sprint_logit).log_prob(samples["sprint"]).sum()
        lp += IndependentBernoulli(e_heads.kick_logit).log_prob(samples["kick"]).sum()
        lp += IndependentBernoulli(e_heads.tackle_attempt_logit).log_prob(
            samples["tackle_attempt"]
        ).sum()

        # Direction heads (move_direction, kick_direction) are intentionally
        # excluded from the PPO log_prob ratio. The BC loss and policy gradient
        # both move the direction head mean by large amounts each step, making
        # the Gaussian log_prob of stored samples under the new distribution
        # diverge catastrophically (mean_shift ~25-50 units with std~1 = KL ~2000).
        # Direction is trained implicitly via policy/value gradient and explicitly
        # via BC; it doesn't need to be in the ratio for correct PPO.

        return lp

    # -----------------------------------------------------------------------
    # PPO update
    # -----------------------------------------------------------------------

    def _ppo_update(self, batch: dict, progress: float) -> dict:
        """Run N epochs of minibatch PPO updates over the collected rollout.

        Returns dict of mean loss metrics for logging.
        """
        from footballcoach.ai.ppo.bc import bc_loss_from_tensor

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
        epoch_times = []

        # Print rollout-level stats once per update
        print(f"\n{'='*70}")
        print(f"[PPO UPDATE] step={self._total_steps:,}  n={n}  progress={progress:.3f}")
        print(f"  old_log_prob: mean={old_log_probs.mean():.3f}  std={old_log_probs.std():.3f}"
              f"  min={old_log_probs.min():.3f}  max={old_log_probs.max():.3f}")
        print(f"  returns:      mean={returns.mean():.3f}  std={returns.std():.3f}"
              f"  min={returns.min():.3f}  max={returns.max():.3f}")
        raw_adv = batch["advantages"]
        print(f"  advantages:   mean={raw_adv.mean():.3f}  std={raw_adv.std():.3f}"
              f"  min={raw_adv.min():.3f}  max={raw_adv.max():.3f}")
        raw_vals = batch["values"]
        print(f"  values(old):  mean={raw_vals.mean():.3f}  std={raw_vals.std():.3f}"
              f"  min={raw_vals.min():.3f}  max={raw_vals.max():.3f}")
        raw_rews = batch["rewards"]
        print(f"  rewards:      mean={raw_rews.mean():.4f}  std={raw_rews.std():.4f}"
              f"  min={raw_rews.min():.4f}  max={raw_rews.max():.4f}"
              f"  nonzero={int((raw_rews != 0).sum())}/{n}")
        print(f"{'='*70}")

        _diag_done = False  # print per-head breakdown only once

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

                # Recompute log_probs and values with current policy
                sf = mb_obs["self_feat"]
                of = mb_obs["other_feat"]
                em = mb_obs["exists_mask"]
                bf = mb_obs["ball_feat"]
                gf = mb_obs["global_feat"]

                d_heads = self.decision_net(sf, of, em, bf, gf)
                e_heads = self.execution_net(sf, of, em, bf, gf, d_heads)

                # Value estimate
                new_values = ((d_heads.value + e_heads.value) / 2.0).squeeze(-1)

                # New log_probs (sample stored actions from batch)
                mb_actions = {k.replace("action/", ""): batch[k][mb_idx].to(self.device)
                              for k in batch if k.startswith("action/")}
                new_log_probs = self._recompute_log_prob(d_heads, e_heads, mb_actions, em)

                # Per-head log_prob breakdown (first minibatch of first epoch only)
                if not _diag_done:
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
                        lp_sprint   = _blp(e_heads.sprint_logit, "sprint")
                        lp_kick     = _blp(e_heads.kick_logit, "kick")
                        lp_tackle_a = _blp(e_heads.tackle_attempt_logit, "tackle_attempt")
                        log_std_move = self.execution_net.move_dir_log_std.to(self.device)
                        log_std_kick = self.execution_net.kick_dir_log_std.to(self.device)
                        lp_movedir  = DirectionHead(e_heads.move_direction, log_std_move).log_prob(mb_actions["move_dir_raw"]).mean().item()
                        lp_kickdir  = DirectionHead(e_heads.kick_direction, log_std_kick).log_prob(mb_actions["kick_dir_raw"]).mean().item()
                        lp_new_mb   = new_log_probs.mean().item()
                        lp_old_mb   = mb_old_lp.mean().item()
                        ratio_mb    = torch.exp(new_log_probs - mb_old_lp)
                    print(f"  [DIAG epoch=0 mb=0]")
                    print(f"    old_lp(mb)={lp_old_mb:.3f}  new_lp(mb)={lp_new_mb:.3f}  diff={lp_new_mb - lp_old_mb:.3f}")
                    print(f"    ratio: mean={ratio_mb.mean():.4f}  std={ratio_mb.std():.4f}"
                          f"  min={ratio_mb.min():.4f}  max={ratio_mb.max():.4f}")
                    print(f"    new_values: mean={new_values.mean():.3f}  ret(mb): mean={mb_ret.mean():.3f}")
                    print(f"    per-head new_log_prob contributions (mean over mb):")
                    print(f"      shoot={lp_shoot:.3f}  pass={lp_pass:.3f}  move={lp_move:.3f}  tackle={lp_tackle:.3f}")
                    print(f"      gp_extra={lp_gp:.3f}  mark={lp_mark:.3f}  hold={lp_hold:.3f}")
                    print(f"      sprint={lp_sprint:.3f}  kick={lp_kick:.3f}  tackle_attempt={lp_tackle_a:.3f}")
                    print(f"      move_dir={lp_movedir:.3f}  kick_dir={lp_kickdir:.3f}")
                    print(f"      TOTAL={lp_shoot+lp_pass+lp_move+lp_tackle+lp_gp+lp_mark+lp_hold+lp_sprint+lp_kick+lp_tackle_a+lp_movedir:.3f}")
                    print(f"    move_dir log_std={self.execution_net.move_dir_log_std.data.tolist()}"
                          f"  kick_dir log_std={self.execution_net.kick_dir_log_std.data.tolist()}")

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss — normalise by return variance so it stays ~O(1)
                # regardless of how large/negative the returns are. This keeps
                # the value gradient from overwhelming the policy gradient.
                ret_var = returns.var().clamp(min=1.0)
                value_loss = F.mse_loss(new_values, mb_ret) / ret_var

                # Entropy bonus
                entropy = self._compute_entropy(d_heads, e_heads, em)

                # L2 penalty on raw direction outputs — keeps them from drifting
                # to large magnitudes, which causes huge mean-shifts between
                # minibatches and training instability. The vectors are normalised
                # to unit length before the engine anyway, so this doesn't harm output.
                dir_l2 = (e_heads.move_direction.pow(2).mean()
                          + e_heads.kick_direction.pow(2).mean())

                total_loss = (policy_loss
                              + self.vf_coef * value_loss
                              - self.ent_coef * entropy
                              + self.dir_l2_coef * dir_l2)

                # BC auxiliary loss (decision + execution, annealed to 0)
                bc_loss_val = torch.zeros(1, device=self.device)
                if has_bc:
                    mb_bc = batch["bc_labels"][mb_idx].to(self.device)
                    bc_loss_val = bc_loss_from_tensor(mb_bc, d_heads, e_heads)
                    total_loss = total_loss + bc_coeff * bc_loss_val
                all_bc_loss.append(bc_loss_val.detach().item())

                self.optimizer.zero_grad()
                total_loss.backward()

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
                    d_after = self.decision_net(sf, of, em, bf, gf)
                    e_after = self.execution_net(sf, of, em, bf, gf, d_after)
                    lp_after = self._recompute_log_prob(d_after, e_after, mb_actions, em)
                    movedir_mean_shift = (e_after.move_direction - e_heads.move_direction).norm(dim=-1).mean().item()
                    kickdir_mean_shift = (e_after.kick_direction - e_heads.kick_direction).norm(dim=-1).mean().item()

                approx_kl = (mb_old_lp - new_log_probs).mean().item()
                kl_after_step = (mb_old_lp - lp_after).mean().item()

                mb_i = start // self.minibatch_size
                print(f"  [e{epoch_i} mb{mb_i:02d}]"
                      f"  grad_norm={raw_grad_norm:.1f}"
                      f"  total={total_loss.item():.3f}"
                      f"  policy={policy_loss.item():.3f}"
                      f"  value={value_loss.item():.3f}"
                      f"  dir_l2={dir_l2.item():.3f}"
                      f"  kl_after={kl_after_step:.4f}"
                      f"  move_shift={movedir_mean_shift:.3f}"
                      f"  kick_shift={kickdir_mean_shift:.3f}")

                all_policy_loss.append(policy_loss.item())
                all_value_loss.append(value_loss.item())
                all_entropy.append(entropy.item())
                all_kl.append(kl_after_step)

            epoch_times.append((time.perf_counter() - epoch_start) * 1000)
            mean_kl_epoch = float(np.mean(all_kl[-32:])) if all_kl else 0.0
            print(f"  [epoch {epoch_i}] kl={mean_kl_epoch:.5f}  t={epoch_times[-1]:.0f}ms")
            # Early stop if KL too large
            if all_kl and np.mean(all_kl[-10:]) > self.target_kl:
                print(f"  [early stop] KL={np.mean(all_kl):.5f} > target={self.target_kl}")
                log.debug(f"Early stopping PPO epoch due to KL={np.mean(all_kl):.4f}")
                break

        return {
            "policy_loss": float(np.mean(all_policy_loss)),
            "value_loss": float(np.mean(all_value_loss)),
            "entropy": float(np.mean(all_entropy)),
            "approx_kl": float(np.mean(all_kl)),
            "bc_loss": float(np.mean(all_bc_loss)),
            "bc_coeff": bc_coeff,
            "epoch_time_ms": float(np.mean(epoch_times)) if epoch_times else 0.0,
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
        lp += _b(e_heads.sprint_logit, "sprint")
        lp += _b(e_heads.kick_logit, "kick")
        lp += _b(e_heads.tackle_attempt_logit, "tackle_attempt")

        # Target categorical log_probs (gated by intent)
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

        # Direction heads excluded — see comment in _compute_log_prob.

        return lp

    def _compute_entropy(self, d_heads, e_heads, exists_mask) -> torch.Tensor:
        ent = torch.zeros(1, device=self.device)
        for logit in [
            d_heads.shoot_logit, d_heads.pass_logit, d_heads.move_logit,
            d_heads.tackle_logit, d_heads.get_possession_raw, d_heads.mark_logit,
            d_heads.hold_position_logit, e_heads.sprint_logit, e_heads.kick_logit,
            e_heads.tackle_attempt_logit,
        ]:
            ent += IndependentBernoulli(logit).entropy().mean()
        for logits in [d_heads.pass_target_logits, d_heads.tackle_target_logits, d_heads.mark_target_logits]:
            ent += MaskedCategorical(logits, exists_mask).entropy().mean()
        return ent

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def _save_checkpoint(self, step: int) -> None:
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"checkpoint_{step:08d}.pt"
        torch.save({
            "step": step,
            "decision_net": self.decision_net.state_dict(),
            "execution_net": self.execution_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        log.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: Path) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.decision_net.load_state_dict(ckpt["decision_net"])
        self.execution_net.load_state_dict(ckpt["execution_net"])
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
        "sprint": exec_samples["sprint"],
        "kick": exec_samples["kick"],
        "tackle_attempt": exec_samples["tackle_attempt"],
        "move_dir_raw": exec_samples["move_dir_raw"],
        "kick_dir_raw": exec_samples["kick_dir_raw"],
        "move_region_center_raw": action.move_region_center_raw,
        "move_region_size_raw": np.array([action.move_region_size_raw], dtype=np.float32),
        "move_arrival_speed_raw": np.array([action.move_arrival_speed_raw], dtype=np.float32),
    }

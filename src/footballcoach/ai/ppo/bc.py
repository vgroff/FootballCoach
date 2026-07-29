"""Behavioural cloning (BC) support for rules-based AI bootstrapping.

Two modes, both configurable via ai_config.json['bc']:

1. **Pre-training** (before PPO starts):
   Run ``BCPretrainer.pretrain(env, n_steps)`` which rolls the rules-based AI
   in the environment and trains the network via pure supervised cross-entropy
   on the resulting (obs, label) pairs. This gives the network a decent
   starting point before PPO exploration begins.

2. **Auxiliary loss during PPO** (annealed to zero):
   At each PPO rollout-collection step, ``label_fn(env)`` is called to get a
   ``BCLabel`` for the current state. These labels are stored in the rollout
   buffer and later used in ``_ppo_update()`` as an auxiliary cross-entropy
   term weighted by ``bc_aux_coeff`` (linearly annealed to 0.0).

   This keeps the network from drifting too far from sensible behaviour early
   in PPO training while still letting the RL signal take over as it matures.

Design rule: BC labels do NOT go through the PPO importance ratio / clipping.
They are a separate, additive loss term. This means they can use actions taken
by the rules-based AI (which has no π_old) without corrupting PPO's math.

Flat tensor layout for stored BC labels (11 floats per step):
  [0]  shoot
  [1]  pass_
  [2]  move
  [3]  tackle
  [4]  get_possession_extra
  [5]  mark
  [6]  hold_position
  [7]  move_dir_x   (unit vector component; 0.0 if dir not applicable)
  [8]  move_dir_y
  [9]  sprint
  [10] valid         (1.0 = use this label, 0.0 = skip BC loss for this step)
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("footballcoach.ai.bc")

BC_LABEL_DIM = 11  # elements in the flat label vector (see module docstring)

# Indices into the flat label vector
_I_SHOOT     = 0
_I_PASS      = 1
_I_MOVE      = 2
_I_TACKLE    = 3
_I_GP_EXTRA  = 4
_I_MARK      = 5
_I_HOLD      = 6
_I_DIR_X     = 7
_I_DIR_Y     = 8
_I_SPRINT    = 9
_I_VALID     = 10


@dataclass
class BCLabel:
    """Rules-based supervision label for one decision step.

    All Bernoulli targets are floats in {0.0, 1.0}.
    ``move_direction`` is a (dx, dy) unit vector or ``None`` to skip the
    direction loss (e.g. when movement is not the selected action).
    ``valid=False`` skips BC loss entirely for this step.
    """
    shoot: float = 0.0
    pass_: float = 0.0
    move: float = 0.0
    tackle: float = 0.0
    get_possession_extra: float = 0.0  # raw gp head, not derived gp_prob
    mark: float = 0.0
    hold_position: float = 0.0
    move_direction: Optional[np.ndarray] = None  # shape (2,) unit vector
    sprint: float = 1.0
    valid: bool = True

    def to_array(self) -> np.ndarray:
        """Pack into a flat float32 array of length BC_LABEL_DIM."""
        arr = np.zeros(BC_LABEL_DIM, dtype=np.float32)
        arr[_I_SHOOT]    = self.shoot
        arr[_I_PASS]     = self.pass_
        arr[_I_MOVE]     = self.move
        arr[_I_TACKLE]   = self.tackle
        arr[_I_GP_EXTRA] = self.get_possession_extra
        arr[_I_MARK]     = self.mark
        arr[_I_HOLD]     = self.hold_position
        if self.move_direction is not None:
            arr[_I_DIR_X] = float(self.move_direction[0])
            arr[_I_DIR_Y] = float(self.move_direction[1])
        arr[_I_SPRINT]   = self.sprint
        arr[_I_VALID]    = 1.0 if self.valid else 0.0
        return arr

    @staticmethod
    def invalid() -> "BCLabel":
        """A label that contributes zero BC loss."""
        return BCLabel(valid=False)


# ---------------------------------------------------------------------------
# Rules-based label generators
# ---------------------------------------------------------------------------

def phase1_labels(env) -> BCLabel:
    """Generate rules-based BC labels for the Phase 1 trainee from the
    current match state.

    Rules:
    - Ball loose or opponent has it:
        get_possession=1 (gp_extra=1), move=0, tackle=0
        move_direction = unit vector toward ball
    - Trainee has ball:
        move=1, get_possession=0
        move_direction = unit vector toward opponent's goal (+x for Team.LEFT)
    - Always: sprint=1, all other heads=0
    """
    try:
        match = env.match
        trainee_id = env.trainee_player_id
        if match is None:
            return BCLabel.invalid()
        trainee = match.player_by_id(trainee_id)
        ball = match.ball
    except (KeyError, AttributeError):
        return BCLabel.invalid()

    tx, ty = trainee.position.x, trainee.position.y
    bx, by = ball.position.x, ball.position.y

    if ball.possessed_by == trainee_id:
        # Trainee has ball: move toward +x (LEFT team attacks +x)
        # Use opponent goal centre x which is at +half_length
        goal_x = match.pitch.half_length
        goal_y = 0.0
        dx = goal_x - tx
        dy = goal_y - ty
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return BCLabel.invalid()
        direction = np.array([dx / length, dy / length], dtype=np.float32)
        return BCLabel(
            move=1.0,
            sprint=1.0,
            move_direction=direction,
        )
    else:
        # No possession or opponent has it: get possession, move toward ball
        dx = bx - tx
        dy = by - ty
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return BCLabel.invalid()
        direction = np.array([dx / length, dy / length], dtype=np.float32)
        return BCLabel(
            get_possession_extra=1.0,
            sprint=1.0,
            move_direction=direction,
        )


# ---------------------------------------------------------------------------
# BC loss computation
# ---------------------------------------------------------------------------

def bc_loss_from_tensor(
    labels: torch.Tensor,
    decision_heads,
    exec_heads,
) -> torch.Tensor:
    """Compute BC loss for a minibatch, given packed label tensors.

    Args:
        labels: (N, BC_LABEL_DIM) float32 tensor from the rollout buffer.
        decision_heads: DecisionHeadsRaw from decision_net forward pass.
        exec_heads: ExecutionHeadsRaw from execution_net forward pass.

    Returns:
        Scalar BC loss (mean over valid steps).  Returns zero tensor if no
        valid steps in the batch.
    """
    valid = labels[:, _I_VALID] > 0.5  # (N,) bool
    if not valid.any():
        return torch.zeros(1, device=labels.device)

    loss = torch.zeros(labels.shape[0], device=labels.device)

    # --- Bernoulli decision heads (BCE from logits) ---
    def _bce(logit: torch.Tensor, col: int) -> torch.Tensor:
        target = labels[:, col]
        return F.binary_cross_entropy_with_logits(
            logit.squeeze(-1), target, reduction="none"
        )

    loss += _bce(decision_heads.shoot_logit,          _I_SHOOT)
    loss += _bce(decision_heads.pass_logit,            _I_PASS)
    loss += _bce(decision_heads.move_logit,            _I_MOVE)
    loss += _bce(decision_heads.tackle_logit,          _I_TACKLE)
    loss += _bce(decision_heads.get_possession_raw,    _I_GP_EXTRA)
    loss += _bce(decision_heads.mark_logit,            _I_MARK)
    loss += _bce(decision_heads.hold_position_logit,   _I_HOLD)

    # --- Execution: sprint Bernoulli ---
    loss += _bce(exec_heads.sprint_logit, _I_SPRINT)

    # --- Execution: move_direction cosine loss ---
    # Only where we have a valid direction (both components nonzero)
    has_dir = (labels[:, _I_DIR_X].abs() + labels[:, _I_DIR_Y].abs()) > 1e-6
    if has_dir.any():
        target_dir = labels[:, _I_DIR_X:_I_DIR_Y + 1]  # (N, 2)
        pred_dir = exec_heads.move_direction             # (N, 2) raw (pre-normalize)
        eps = 1e-6
        pred_norm = pred_dir / (pred_dir.norm(dim=-1, keepdim=True) + eps)
        # 1 - cosine similarity → 0 when perfectly aligned, 2 when opposite
        cos_loss = 1.0 - (pred_norm * target_dir).sum(dim=-1)
        loss += torch.where(has_dir, cos_loss, torch.zeros_like(cos_loss))

    # Mask to valid steps only and mean
    valid_loss = loss[valid]
    return valid_loss.mean() if len(valid_loss) > 0 else torch.zeros(1, device=labels.device)


# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------

class BCPretrainer:
    """Runs a short supervised pre-training phase before PPO begins.

    For each step:
      1. Encode the current match observation.
      2. Compute rules-based BC labels for the trainee.
      3. Forward pass both networks.
      4. Compute BC loss; backprop; optimizer step.
      5. Advance one decision interval in the env (using the label's implied
         action, NOT the network's sampled action) so the env state changes.

    The env is stepped with a dummy rules-based action derived from the label
    so the trainee actually moves toward the ball, producing a varied stream
    of states rather than replaying the same initial state forever.
    """

    def __init__(self, decision_net, execution_net, cfg: dict, device: torch.device):
        self.decision_net = decision_net
        self.execution_net = execution_net
        self.device = device
        bc_cfg = cfg.get("bc", {})
        lr = float(bc_cfg.get("pretrain_lr", 1e-3))
        all_params = (
            list(decision_net.parameters()) + list(execution_net.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=lr, eps=1e-5)

    def pretrain(
        self,
        env,
        n_steps: int,
        label_fn: Callable,
    ) -> None:
        """Run ``n_steps`` BC pre-training steps.

        Args:
            env: ScenarioEnv — must already have been reset externally.
            n_steps: Total gradient steps (= environment steps).
            label_fn: Callable(env) -> BCLabel — produces supervision labels.
        """
        if n_steps <= 0:
            return

        log.info(f"BC pre-training: {n_steps} steps")
        obs = env.reset()
        total_loss = 0.0
        valid_steps = 0

        for step in range(n_steps):
            label = label_fn(env)
            label_arr = torch.from_numpy(label.to_array()).unsqueeze(0).to(self.device)

            obs_dict = {
                k: v.unsqueeze(0).to(self.device)
                for k, v in obs.to_torch_dict().items()
            }

            self.optimizer.zero_grad()
            d_heads = self.decision_net(
                obs_dict["self_feat"], obs_dict["other_feat"],
                obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
            )
            e_heads = self.execution_net(
                obs_dict["self_feat"], obs_dict["other_feat"],
                obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                d_heads,
            )
            loss = bc_loss_from_tensor(label_arr, d_heads, e_heads)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            if label.valid:
                valid_steps += 1

            # Step the env with a dummy action derived from the label so the
            # state changes (trainee moves toward ball / goal).
            env_action = _label_to_env_action(label, env)
            next_obs, _reward, done, _info = env.step(env_action)
            obs = env.reset() if done else next_obs

            if (step + 1) % 200 == 0:
                mean_loss = total_loss / max(1, valid_steps)
                log.info(
                    f"BC pre-train step {step + 1}/{n_steps} | "
                    f"mean_bc_loss={mean_loss:.4f} (over {valid_steps} valid steps)"
                )
                total_loss = 0.0
                valid_steps = 0

        log.info("BC pre-training complete.")


def _label_to_env_action(label: BCLabel, env) -> dict:
    """Build a minimal env action dict from a BCLabel so the env can be
    stepped during pre-training.  The trainee's order is set by the env's
    apply_action_to_player via to_orders.py — we just need plausible
    decision_probs and execution_physical.
    """
    from footballcoach.ai.action.schema import ExecutionAction

    # Derive decision probs from label (0/1 → 0.1/0.9 to avoid log(0) in any
    # downstream callers; the env's gating uses a 0.5 threshold anyway).
    def _prob(v: float) -> float:
        return 0.9 if v > 0.5 else 0.1

    direction = label.move_direction
    if direction is None:
        direction = np.array([0.0, 0.0], dtype=np.float32)

    return {
        "decision_probs": {
            "shoot":          _prob(label.shoot),
            "pass_":          _prob(label.pass_),
            "move":           _prob(label.move),
            "tackle":         _prob(label.tackle),
            "get_possession": _prob(label.get_possession_extra),
            "mark":           _prob(label.mark),
            "hold_position":  _prob(label.hold_position),
        },
        "execution_physical": {
            "move_direction":    direction,
            "sprint":            label.sprint > 0.5,
            "kick_this_tick":    False,
            "kick_direction":    np.array([1.0, 0.0], dtype=np.float32),
            "kick_power_fraction": 0.0,
            "kick_spin":         np.zeros(3, dtype=np.float32),
            "tackle_attempt":    False,
        },
        "decision_physical": {
            "move_region_center_m":   np.zeros(2, dtype=np.float32),
            "move_region_size_m":     2.0,
            "move_arrival_speed_mps": 7.0,
        },
        "target_slots": {"pass_": 0, "tackle": 0, "mark": 0},
        "slot_player_ids": [None] * 21,
        "decision": None,
        "execution": ExecutionAction(),
    }

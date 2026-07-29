"""Action schemas: raw network output heads and sampled actions.

These dataclasses are the canonical representation of what the decision
network and execution network output, and what is stored in the rollout
buffer.  See ai_design_doc.md sections 2.4, 2.5, 8.2, 8.6.

Two-level structure:
  DecisionHeadsRaw  - raw logits/raw continuous values from the decision
                      network forward pass (pre-sigmoid, pre-softmax etc.).
  DecisionAction    - sampled values from the decision heads; these are what
                      PPO log_prob is computed against (NOT the derived
                      gated/winner-take-all values used by the engine).
  ExecutionHeadsRaw - raw output from the execution network forward pass.
  ExecutionAction   - sampled values from the execution heads.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Decision network
# ---------------------------------------------------------------------------

@dataclass
class DecisionHeadsRaw:
    """Raw (pre-sampling) output tensors from DecisionNetwork.forward().

    All tensors are PyTorch tensors shaped (batch, size) as returned by the
    network.  The *_logit fields are raw logits (use sigmoid for prob, or
    pass directly to IndependentBernoulli).  The *_target_logits are
    pre-mask raw logits for the masked categorical distributions.
    """
    shoot_logit: "torch.Tensor"          # (batch, 1)
    pass_logit: "torch.Tensor"           # (batch, 1)
    move_logit: "torch.Tensor"           # (batch, 1)
    tackle_logit: "torch.Tensor"         # (batch, 1)
    get_possession_raw: "torch.Tensor"   # (batch, 1)  - headroom above tackle_prob; NOT the final prob
    mark_logit: "torch.Tensor"           # (batch, 1)
    hold_position_logit: "torch.Tensor"  # (batch, 1)

    pass_target_logits: "torch.Tensor"   # (batch, MAX_OTHER_PLAYERS)
    tackle_target_logits: "torch.Tensor" # (batch, MAX_OTHER_PLAYERS)
    mark_target_logits: "torch.Tensor"   # (batch, MAX_OTHER_PLAYERS)

    move_region_center: "torch.Tensor"   # (batch, 2) raw, tanh-squashed to norm position in to_orders.py
    move_region_size: "torch.Tensor"     # (batch, 1) raw, sigmoid+scale to 1-4m in to_orders.py
    move_arrival_speed: "torch.Tensor"   # (batch, 1) raw, sigmoid+scale to 0-top_speed
    region_of_play_center: "torch.Tensor"  # (batch, 2)
    region_of_play_size: "torch.Tensor"    # (batch, 1) raw, sigmoid+scale to 15-40m
    attack_defence_raw: "torch.Tensor"     # (batch, 1) raw, sigmoid -> 0-1 (instantaneous target for EMA)
    latent_vector: "torch.Tensor"          # (batch, latent_dim) - free vector passed to execution net

    value: "torch.Tensor"                  # (batch, 1) - critic value estimate (shared trunk)


@dataclass
class DecisionAction:
    """One sampled action from the decision heads (stored in rollout buffer).

    All fields are scalars or small numpy arrays (unbatched, one decision step).
    These are the values PPO log_prob is computed against - they are the raw
    Bernoulli samples (0/1) and raw categorical indices, NOT the gated/derived
    values used to actually drive the engine this tick.

    ``get_possession_extra`` is the raw Bernoulli sample from
    ``get_possession_raw`` (the "headroom above tackle" term), NOT the
    derived effective get_possession_prob (which is a function of both
    tackle_intent and this value; see ai_design_doc.md 8.2.1).
    """
    # Bernoulli samples (0 or 1)
    shoot: float = 0.0
    pass_: float = 0.0
    move: float = 0.0
    tackle: float = 0.0
    get_possession_extra: float = 0.0
    mark: float = 0.0
    hold_position: float = 0.0

    # Categorical targets (integer slot index; only valid when the corresponding
    # intent is 1 - guard at call site before computing log_prob for these)
    pass_target: int = 0
    tackle_target: int = 0
    mark_target: int = 0

    # Continuous raw (pre-squash) samples - used for PPO log_prob
    move_region_center_raw: np.ndarray = None   # shape (2,), pre-tanh
    move_region_size_raw: float = 0.0
    move_arrival_speed_raw: float = 0.0
    region_of_play_center_raw: np.ndarray = None  # shape (2,)
    region_of_play_size_raw: float = 0.0
    attack_defence_raw: float = 0.0

    def __post_init__(self):
        if self.move_region_center_raw is None:
            self.move_region_center_raw = np.zeros(2, dtype=np.float32)
        if self.region_of_play_center_raw is None:
            self.region_of_play_center_raw = np.zeros(2, dtype=np.float32)


# ---------------------------------------------------------------------------
# Execution network
# ---------------------------------------------------------------------------

@dataclass
class ExecutionHeadsRaw:
    """Raw output tensors from ExecutionNetwork.forward().

    See ai_design_doc.md section 8.6 for the full output surface.
    """
    move_direction: "torch.Tensor"         # (batch, 2) raw 2-vector; L2-norm in gating.py
    sprint_logit: "torch.Tensor"           # (batch, 1) Bernoulli: sprint vs jog
    kick_logit: "torch.Tensor"             # (batch, 1) Bernoulli: kick this tick?
    kick_direction: "torch.Tensor"         # (batch, 2) raw 2-vector; L2-norm in gating.py
    kick_power: "torch.Tensor"             # (batch, 1) raw; sigmoid -> 0-1 power_fraction
    kick_spin: "torch.Tensor"              # (batch, 3) raw spin vector
    tackle_attempt_logit: "torch.Tensor"   # (batch, 1) Bernoulli

    value: "torch.Tensor"                  # (batch, 1) critic value estimate


@dataclass
class ExecutionAction:
    """Sampled action from the execution heads (stored in rollout buffer)."""
    move_direction_raw: np.ndarray = None  # shape (2,) pre-normalization raw 2-vector
    sprint: float = 0.0                    # 0 = jog, 1 = sprint
    kick: float = 0.0                      # 0 = no kick, 1 = kick this tick
    kick_direction_raw: np.ndarray = None  # shape (2,) pre-normalization
    kick_power_raw: float = 0.0            # pre-sigmoid raw
    kick_spin_raw: np.ndarray = None       # shape (3,) raw spin vector
    tackle_attempt: float = 0.0            # 0 = no, 1 = yes

    def __post_init__(self):
        if self.move_direction_raw is None:
            self.move_direction_raw = np.array([1.0, 0.0], dtype=np.float32)
        if self.kick_direction_raw is None:
            self.kick_direction_raw = np.array([1.0, 0.0], dtype=np.float32)
        if self.kick_spin_raw is None:
            self.kick_spin_raw = np.zeros(3, dtype=np.float32)


# ---------------------------------------------------------------------------
# Combined action (decision + execution, one step)
# ---------------------------------------------------------------------------

@dataclass
class PlayerAction:
    """The combined decision+execution action for one player for one decision
    interval.  Stored in the rollout buffer; log_prob is computed from both
    networks' raw outputs."""
    decision: DecisionAction = None
    execution: ExecutionAction = None

    def __post_init__(self):
        if self.decision is None:
            self.decision = DecisionAction()
        if self.execution is None:
            self.execution = ExecutionAction()

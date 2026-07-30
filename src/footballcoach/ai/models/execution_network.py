"""Execution network: observation + decision output -> low-level motor actions.

See ai_design_doc.md section 8.6 for full specification.

Structurally similar to the decision network (entity encoder + trunk), but:
  - Also takes the full DecisionHeadsRaw output as input (concatenated via
    a decision_mlp branch) - including heads not "selected" this tick.
  - Outputs the per-tick motor actions that drive the engine directly.

Outputs:
  move_direction   - unit vector (L2-normalized inside forward(); mean is always on unit circle)
  exec_move_logit  - Bernoulli: move vs standstill
  sprint_logit     - Bernoulli: sprint vs jog (only meaningful when exec_move=1)
  kick_logit       - Bernoulli: kick this tick?
  kick_direction   - unit vector (L2-normalized inside forward(); mean is always on unit circle)
  kick_power       - raw scalar; sigmoid -> [0, 1] power_fraction
  kick_spin        - raw 3D vector (physical units determined by to_orders.py)
  tackle_attempt_logit - Bernoulli
  value            - critic value estimate (shared trunk Option A)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from footballcoach.ai.action.schema import DecisionHeadsRaw, ExecutionHeadsRaw
from footballcoach.ai.models.entity_encoder import EntityEncoder
from footballcoach.ai.obs.schema import (
    BALL_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    PLAYER_FEATURE_DIM,
)


def _decision_output_dim(latent_dim: int) -> int:
    """Compute the flat dimension of DecisionHeadsRaw (for the decision_mlp).

    The decision output concatenated for the execution network includes:
      - 7 scalar Bernoulli logits
      - 3 categorical target logit vectors (MAX_OTHER_PLAYERS each)
      - move_region_center (2), move_region_size (1), move_arrival_speed (1)
      - region_of_play_center (2), region_of_play_size (1)
      - attack_defence_raw (1)
      - latent_vector (latent_dim)
    Value head is excluded (not useful as input to execution network).
    """
    from footballcoach.ai.obs.schema import MAX_OTHER_PLAYERS
    return (
        7 +                     # Bernoulli scalar logits
        3 * MAX_OTHER_PLAYERS + # pass/tackle/mark target logits
        2 + 1 + 1 +             # move_region_center, size, arrival_speed
        2 + 1 +                 # region_of_play_center, size
        1 +                     # attack_defence_raw
        latent_dim
    )


def flatten_decision_heads(heads: DecisionHeadsRaw) -> torch.Tensor:
    """Flatten all decision head tensors into a single vector per batch row.

    The value head is excluded (not input to execution network).
    """
    parts = [
        heads.shoot_logit,
        heads.pass_logit,
        heads.move_logit,
        heads.tackle_logit,
        heads.get_possession_raw,
        heads.mark_logit,
        heads.hold_position_logit,
        heads.pass_target_logits,
        heads.tackle_target_logits,
        heads.mark_target_logits,
        heads.move_region_center,
        heads.move_region_size,
        heads.move_arrival_speed,
        heads.region_of_play_center,
        heads.region_of_play_size,
        heads.attack_defence_raw,
        heads.latent_vector,
    ]
    return torch.cat(parts, dim=-1)


class ExecutionNetwork(nn.Module):
    """Execution network: produces per-tick motor actions.

    Args:
        self_dim: Dimension of the player feature vector.
        ball_dim: Dimension of the ball feature vector.
        global_dim: Dimension of the global feature vector.
        latent_dim: Latent vector dimension (must match DecisionNetwork).
        entity_embed_dim: Entity encoder embedding dimension.
        num_attention_heads: Number of attention heads.
        self_mlp_hidden: Hidden dim for self MLP branch.
        ball_mlp_hidden: Hidden dim for ball MLP branch.
        global_mlp_hidden: Hidden dim for global MLP branch.
        decision_mlp_hidden: Hidden dim for the decision-output MLP branch.
        trunk_hidden: Shared trunk hidden dimension.
    """

    def __init__(
        self,
        self_dim: int = PLAYER_FEATURE_DIM,
        ball_dim: int = BALL_FEATURE_DIM,
        global_dim: int = GLOBAL_FEATURE_DIM,
        latent_dim: int = 32,
        entity_embed_dim: int = 64,
        num_attention_heads: int = 4,
        self_mlp_hidden: int = 64,
        ball_mlp_hidden: int = 32,
        global_mlp_hidden: int = 32,
        decision_mlp_hidden: int = 64,
        trunk_hidden: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.entity_encoder = EntityEncoder(
            entity_feature_dim=self_dim,
            embed_dim=entity_embed_dim,
            num_heads=num_attention_heads,
        )
        self.self_mlp = nn.Sequential(
            nn.Linear(self_dim, self_mlp_hidden), nn.ReLU()
        )
        self.ball_mlp = nn.Sequential(
            nn.Linear(ball_dim, ball_mlp_hidden), nn.ReLU()
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, global_mlp_hidden), nn.ReLU()
        )
        dec_dim = _decision_output_dim(latent_dim)
        self.decision_mlp = nn.Sequential(
            nn.Linear(dec_dim, decision_mlp_hidden), nn.ReLU()
        )

        trunk_input_dim = (
            entity_embed_dim + self_mlp_hidden +
            ball_mlp_hidden + global_mlp_hidden +
            decision_mlp_hidden
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, trunk_hidden), nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(),
        )

        # Motor output heads
        self.move_direction = nn.Linear(trunk_hidden, 2)    # L2-normalized to unit vector in forward()
        self.exec_move_logit = nn.Linear(trunk_hidden, 1)   # Bernoulli: move vs standstill
        self.sprint_logit = nn.Linear(trunk_hidden, 1)       # Bernoulli: sprint vs jog
        self.kick_logit = nn.Linear(trunk_hidden, 1)          # Bernoulli
        self.kick_direction = nn.Linear(trunk_hidden, 2)       # L2-normalized to unit vector in forward()
        self.kick_power = nn.Linear(trunk_hidden, 1)            # raw; sigmoid -> [0,1]
        self.kick_spin = nn.Linear(trunk_hidden, 3)             # raw spin vector
        self.tackle_attempt_logit = nn.Linear(trunk_hidden, 1)  # Bernoulli

        # Critic value head (shared trunk, Option A)
        self.value_head = nn.Linear(trunk_hidden, 1)

        # Fixed (non-learnable) log_std for direction heads.
        # Direction heads are included in the PPO log_prob ratio. The mean is
        # constrained to the unit circle (|mean|=1), so max mean-shift is 2 and
        # KL contribution per step is bounded (~O(1) vs the previous ~2000 when
        # the raw vector could drift to magnitude 25-50). These buffers only
        # control rollout sampling noise via DirectionHead.sample_raw().
        self.move_dir_log_std = nn.Parameter(torch.zeros(2))
        self.kick_dir_log_std = nn.Parameter(torch.zeros(2))
        self.kick_power_log_std = nn.Parameter(torch.zeros(1))
        self.kick_spin_log_std = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        self_feat: torch.Tensor,        # (batch, self_dim)
        other_feat: torch.Tensor,       # (batch, MAX_OTHER_PLAYERS, self_dim)
        exists_mask: torch.Tensor,      # (batch, MAX_OTHER_PLAYERS)
        ball_feat: torch.Tensor,        # (batch, ball_dim)
        global_feat: torch.Tensor,      # (batch, global_dim)
        decision_heads: DecisionHeadsRaw,  # from DecisionNetwork.forward()
    ) -> ExecutionHeadsRaw:
        entity_ctx = self.entity_encoder(self_feat, other_feat, exists_mask)
        dec_flat = flatten_decision_heads(decision_heads)
        h = torch.cat([
            entity_ctx,
            self.self_mlp(self_feat),
            self.ball_mlp(ball_feat),
            self.global_mlp(global_feat),
            self.decision_mlp(dec_flat),
        ], dim=-1)
        h = self.trunk(h)

        eps = 1e-6
        raw_move = self.move_direction(h)
        raw_kick = self.kick_direction(h)

        return ExecutionHeadsRaw(
            move_direction=raw_move / (raw_move.norm(dim=-1, keepdim=True) + eps),
            exec_move_logit=self.exec_move_logit(h),
            sprint_logit=self.sprint_logit(h),
            kick_logit=self.kick_logit(h),
            kick_direction=raw_kick / (raw_kick.norm(dim=-1, keepdim=True) + eps),
            kick_power=self.kick_power(h),
            kick_spin=self.kick_spin(h),
            tackle_attempt_logit=self.tackle_attempt_logit(h),
            value=self.value_head(h),
        )

    @classmethod
    def from_config(cls) -> "ExecutionNetwork":
        from footballcoach.ai.config import load_ai_config
        cfg = load_ai_config()["network"]
        return cls(
            latent_dim=cfg["latent_dim"],
            entity_embed_dim=cfg["entity_embed_dim"],
            num_attention_heads=cfg["num_attention_heads"],
            self_mlp_hidden=cfg["self_mlp_hidden"],
            ball_mlp_hidden=cfg["ball_mlp_hidden"],
            global_mlp_hidden=cfg["global_mlp_hidden"],
            decision_mlp_hidden=cfg["decision_mlp_hidden"],
            trunk_hidden=cfg["trunk_hidden"],
        )

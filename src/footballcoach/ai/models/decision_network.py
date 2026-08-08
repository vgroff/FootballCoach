"""Decision network: observation -> all decision heads + value estimate.

See ai_design_doc.md sections 8.2 and 9.4 for architecture details.

Architecture:
  entity_encoder(self_feat, other_feat, exists_mask) -> context (embed_dim)
  self_mlp(self_feat)   -> (64,)
  ball_mlp(ball_feat)   -> (32,)
  global_mlp(global_feat) -> (32,)
  trunk([context; self; ball; global]) -> (trunk_hidden,)
  -> many heads (see DecisionHeadsRaw in action/schema.py)

Value function uses the shared trunk (Option A from design doc 9.4).

get_possession constraint (design doc 8.2.1):
  tackle_prob = sigmoid(tackle_logit)
  extra_prob = sigmoid(get_possession_raw)
  get_possession_prob = tackle_prob + extra_prob * (1 - tackle_prob)
This is guaranteed to be in [tackle_prob, 1.0].  PPO log_prob is computed
on tackle_logit and get_possession_raw as two independent Bernoullis.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from footballcoach.ai.action.schema import DecisionHeadsRaw
from footballcoach.ai.models.entity_encoder import EntityEncoder
from footballcoach.ai.models.value_side_channel import ValueAiTypeSideChannel
from footballcoach.ai.obs.schema import (
    AI_TYPE_ONE_HOT_DIM,
    BALL_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    MAX_OTHER_PLAYERS,
    PLAYER_FEATURE_DIM,
)


class DecisionNetwork(nn.Module):
    """Full decision network: observation -> DecisionHeadsRaw + value.

    Args:
        self_dim: Dimension of the self-player feature vector.
        ball_dim: Dimension of the ball feature vector.
        global_dim: Dimension of the global/match-context feature vector.
        entity_embed_dim: Entity encoder embedding dimension.
        num_attention_heads: Number of attention heads in EntityEncoder.
        self_mlp_hidden: Hidden dim for the self-specific MLP branch.
        ball_mlp_hidden: Hidden dim for the ball MLP branch.
        global_mlp_hidden: Hidden dim for the global MLP branch.
        trunk_hidden: Hidden dim for the shared trunk layers.
        latent_dim: Dimension of the latent vector head.
    """

    def __init__(
        self,
        self_dim: int = PLAYER_FEATURE_DIM,
        ball_dim: int = BALL_FEATURE_DIM,
        global_dim: int = GLOBAL_FEATURE_DIM,
        entity_embed_dim: int = 64,
        num_attention_heads: int = 4,
        self_mlp_hidden: int = 64,
        ball_mlp_hidden: int = 32,
        global_mlp_hidden: int = 32,
        trunk_hidden: int = 256,
        latent_dim: int = 32,
        value_extra_hidden: int = 16,
        value_hidden_dim: int = 0,
        inter_player_num_heads: int = 0,
    ):
        super().__init__()
        self.entity_encoder = EntityEncoder(
            entity_feature_dim=self_dim,
            embed_dim=entity_embed_dim,
            num_heads=num_attention_heads,
            ball_feat_dim=ball_dim,
            global_feat_dim=global_dim,
            inter_player_num_heads=inter_player_num_heads,
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

        trunk_input_dim = entity_embed_dim + self_mlp_hidden + ball_mlp_hidden + global_mlp_hidden
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, trunk_hidden), nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(),
        )

        # --- Independent Bernoulli heads (raw logits) ---
        # Each is a scalar logit; sigmoid gives the action probability.
        self.shoot_logit = nn.Linear(trunk_hidden, 1)
        self.pass_logit = nn.Linear(trunk_hidden, 1)
        self.move_logit = nn.Linear(trunk_hidden, 1)
        self.tackle_logit = nn.Linear(trunk_hidden, 1)
        # get_possession "headroom" above tackle_prob (see docstring + 8.2.1)
        self.get_possession_raw = nn.Linear(trunk_hidden, 1)
        self.mark_logit = nn.Linear(trunk_hidden, 1)
        self.hold_position_logit = nn.Linear(trunk_hidden, 1)

        # --- Masked categorical target heads ---
        self.pass_target_logits = nn.Linear(trunk_hidden, MAX_OTHER_PLAYERS)
        self.tackle_target_logits = nn.Linear(trunk_hidden, MAX_OTHER_PLAYERS)
        self.mark_target_logits = nn.Linear(trunk_hidden, MAX_OTHER_PLAYERS)

        # --- Continuous heads (raw outputs; squashing in to_orders.py) ---
        # move_region_center: 2D normalized offset; tanh-squashed later
        self.move_region_center = nn.Linear(trunk_hidden, 2)
        # move_region_size: sigmoid+scale to [1, 4] m
        self.move_region_size = nn.Linear(trunk_hidden, 1)
        # move_arrival_speed: sigmoid+scale to [0, v_top]
        self.move_arrival_speed = nn.Linear(trunk_hidden, 1)
        # region_of_play_center: 2D normalized; tanh-squashed later
        self.region_of_play_center = nn.Linear(trunk_hidden, 2)
        # region_of_play_size: sigmoid+scale to [15, 40] m
        self.region_of_play_size = nn.Linear(trunk_hidden, 1)
        # attack_defence: sigmoid -> [0, 1] instantaneous target for EMA
        self.attack_defence_raw = nn.Linear(trunk_hidden, 1)

        # Latent vector: passed through to execution network
        self.latent_vector = nn.Linear(trunk_hidden, latent_dim)

        # --- Value-only opponent-AI-type side channel ---
        # Bypasses the entity encoder/attention and every policy head entirely.
        # Feeds ONLY value_head. See ai/knowledge.md "Opponent-AI-type
        # (value-only)" - do not route this through the shared trunk `h` or
        # any policy head; that would let the policy condition on opponent
        # identity, which is the exact thing this design avoids.
        #
        # Uses a dedicated shared-per-slot-MLP + attention pool (own weights,
        # zero sharing with self.entity_encoder) instead of a flatten+Linear,
        # so it is exactly permutation-invariant like the main entity encoder
        # (a flatten+Linear gives each slot position its own weight block --
        # NOT permutation-invariant; see value_side_channel.py docstring).
        # Enriched with a DETACHED copy of the main entity encoder's per-slot
        # embeddings for richer per-opponent context than a bare one-hot type
        # flag; detaching is what keeps this value-only (no gradient back
        # into entity_encoder/policy from the value loss).
        self.value_ai_type_channel = ValueAiTypeSideChannel(
            ai_type_dim=AI_TYPE_ONE_HOT_DIM,
            entity_embed_dim=entity_embed_dim,
            hidden_dim=value_extra_hidden,
        )

        # Shared-trunk value head (critic, Option A) + value-only ai-type side channel.
        # value_hidden_dim=0 keeps the old single-linear-layer behaviour;
        # >0 inserts one ReLU hidden layer before the final scalar output.
        value_in_dim = trunk_hidden + value_extra_hidden
        if value_hidden_dim > 0:
            self.value_head = nn.Sequential(
                nn.Linear(value_in_dim, value_hidden_dim), nn.ReLU(),
                nn.Linear(value_hidden_dim, 1),
            )
        else:
            self.value_head = nn.Linear(value_in_dim, 1)

    def forward(
        self,
        self_feat: torch.Tensor,    # (batch, self_dim)
        other_feat: torch.Tensor,   # (batch, MAX_OTHER_PLAYERS, self_dim)
        exists_mask: torch.Tensor,  # (batch, MAX_OTHER_PLAYERS)
        ball_feat: torch.Tensor,    # (batch, ball_dim)
        global_feat: torch.Tensor,  # (batch, global_dim)
        self_ai_type: Optional[torch.Tensor] = None,   # (batch, AI_TYPE_ONE_HOT_DIM)
        other_ai_type: Optional[torch.Tensor] = None,  # (batch, MAX_OTHER_PLAYERS, AI_TYPE_ONE_HOT_DIM)
    ) -> DecisionHeadsRaw:
        entity_ctx, self_embed_raw, other_embed_raw = self.entity_encoder(
            self_feat, other_feat, exists_mask, return_embeds=True,
            ball_feat=ball_feat, global_feat=global_feat,
            # No extra_query_bias for decision network — ball+global already cover it.
        )
        h = torch.cat([
            entity_ctx,
            self.self_mlp(self_feat),
            self.ball_mlp(ball_feat),
            self.global_mlp(global_feat),
        ], dim=-1)
        h = self.trunk(h)

        # --- Value-only ai-type side channel (bypasses h/trunk entirely for
        # every head except value_head) ---
        batch_size = h.shape[0]
        if self_ai_type is None:
            self_ai_type = torch.zeros(batch_size, AI_TYPE_ONE_HOT_DIM, device=h.device, dtype=h.dtype)
        if other_ai_type is None:
            other_ai_type = torch.zeros(
                batch_size, MAX_OTHER_PLAYERS, AI_TYPE_ONE_HOT_DIM, device=h.device, dtype=h.dtype
            )
        value_extra = self.value_ai_type_channel(
            self_ai_type, other_ai_type, exists_mask,
            self_embed=self_embed_raw.detach(), other_embed=other_embed_raw.detach(),
        )
        value_input = torch.cat([h, value_extra], dim=-1)

        return DecisionHeadsRaw(
            shoot_logit=self.shoot_logit(h),
            pass_logit=self.pass_logit(h),
            move_logit=self.move_logit(h),
            tackle_logit=self.tackle_logit(h),
            get_possession_raw=self.get_possession_raw(h),
            mark_logit=self.mark_logit(h),
            hold_position_logit=self.hold_position_logit(h),
            pass_target_logits=self.pass_target_logits(h),
            tackle_target_logits=self.tackle_target_logits(h),
            mark_target_logits=self.mark_target_logits(h),
            move_region_center=self.move_region_center(h),
            move_region_size=self.move_region_size(h),
            move_arrival_speed=self.move_arrival_speed(h),
            region_of_play_center=self.region_of_play_center(h),
            region_of_play_size=self.region_of_play_size(h),
            attack_defence_raw=self.attack_defence_raw(h),
            latent_vector=self.latent_vector(h),
            value=self.value_head(value_input),
        )

    @classmethod
    def from_config(cls) -> "DecisionNetwork":
        """Build from ai_config.json (the canonical hyperparameter source)."""
        from footballcoach.ai.config import load_ai_config
        cfg = load_ai_config()["network"]
        return cls(
            entity_embed_dim=cfg["entity_embed_dim"],
            num_attention_heads=cfg["num_attention_heads"],
            self_mlp_hidden=cfg["self_mlp_hidden"],
            ball_mlp_hidden=cfg["ball_mlp_hidden"],
            global_mlp_hidden=cfg["global_mlp_hidden"],
            trunk_hidden=cfg["trunk_hidden"],
            latent_dim=cfg["latent_dim"],
            value_extra_hidden=cfg.get("value_extra_hidden", 16),
            value_hidden_dim=cfg.get("value_hidden_dim", 0),
            inter_player_num_heads=cfg.get("inter_player_attn_heads", 0),
        )


def derive_get_possession_prob(
    tackle_logit: torch.Tensor,
    get_possession_raw_logit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive the effective get-possession probability from the two logits.

    See ai_design_doc.md section 8.2.1.

    Returns:
        (tackle_prob, get_possession_prob) both in [0, 1].
        tackle_prob is just sigmoid(tackle_logit).
        get_possession_prob = tackle_prob + sigmoid(get_possession_raw_logit) * (1 - tackle_prob).
    """
    tackle_prob = torch.sigmoid(tackle_logit)
    extra = torch.sigmoid(get_possession_raw_logit)
    get_possession_prob = tackle_prob + extra * (1.0 - tackle_prob)
    return tackle_prob, get_possession_prob

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

from typing import Optional

import torch
import torch.nn as nn

from footballcoach.ai.action.schema import DecisionHeadsRaw, ExecutionHeadsRaw
from footballcoach.ai.models.entity_encoder import EntityEncoder
from footballcoach.ai.models.value_side_channel import ValueAiTypeSideChannel
from footballcoach.ai.obs.schema import (
    AI_TYPE_ONE_HOT_DIM,
    BALL_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    MAX_OTHER_PLAYERS,
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
        dir_log_std_init: float = -2.0,
        kick_dir_log_std_init: Optional[float] = None,
        value_extra_hidden: int = 16,
        value_hidden_dim: int = 0,
        value_dropout: float = 0.0,
        inter_player_num_heads: int = 0,
        shared_entity_encoder: Optional[nn.Module] = None,
        shared_ball_mlp: Optional[nn.Module] = None,
        shared_global_mlp: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        dec_dim = _decision_output_dim(latent_dim)

        # Use shared modules from DecisionNetwork when provided; own copies otherwise.
        if shared_entity_encoder is not None:
            self.entity_encoder = shared_entity_encoder
        else:
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
        self.ball_mlp = shared_ball_mlp if shared_ball_mlp is not None else nn.Sequential(
            nn.Linear(ball_dim, ball_mlp_hidden), nn.ReLU()
        )
        self.global_mlp = shared_global_mlp if shared_global_mlp is not None else nn.Sequential(
            nn.Linear(global_dim, global_mlp_hidden), nn.ReLU()
        )
        self.decision_mlp = nn.Sequential(
            nn.Linear(dec_dim, decision_mlp_hidden), nn.ReLU()
        )
        # Decision heads projected into the attention query — own weights even when sharing.
        self.decision_query_proj = nn.Linear(dec_dim, entity_embed_dim)

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
        self.kick_direction = nn.Linear(trunk_hidden, 3)       # L2-normalized to 3D unit vector in forward()
        self.kick_power = nn.Linear(trunk_hidden, 1)            # raw; sigmoid -> [0,1]
        self.kick_spin = nn.Linear(trunk_hidden, 3)             # raw spin vector
        self.tackle_attempt_logit = nn.Linear(trunk_hidden, 1)  # Bernoulli

        # --- Value-only opponent-AI-type side channel (mirrors DecisionNetwork's -
        # see ai/knowledge.md "Opponent-AI-type (value-only)"). Bypasses `h`/trunk
        # entirely for every head except value_head. Dedicated shared-per-slot-MLP
        # + attention pool (own weights, zero sharing with self.entity_encoder) --
        # exactly permutation-invariant, unlike the old flatten+Linear (see
        # value_side_channel.py docstring). Enriched with a DETACHED copy of the
        # main entity encoder's per-slot embeddings; detaching keeps this value-only. ---
        self.value_ai_type_channel = ValueAiTypeSideChannel(
            ai_type_dim=AI_TYPE_ONE_HOT_DIM,
            entity_embed_dim=entity_embed_dim,
            hidden_dim=value_extra_hidden,
        )

        # Critic value head (shared trunk, Option A) + value-only ai-type side channel.
        # value_hidden_dim=0 keeps the old single-linear-layer behaviour;
        # >0 inserts one ReLU hidden layer before the final scalar output.
        # value_dropout (network.value_dropout in ai_config.json, default 0.0
        # = disabled): applied ONLY inside value_head, never touching the
        # shared trunk `h` that also feeds the policy/motor heads -- see
        # ai_trainer_knowledge.md "value_head overfitting" discussion. Placed
        # right before each Linear (dropout on the incoming activations),
        # standard practice, and works whether or not value_hidden_dim>0.
        # nn.Dropout auto-disables under .eval() and is active whenever this
        # module is in .train() mode, including during PPO rollout action
        # sampling if the caller hasn't switched to eval() -- this only adds
        # noise to the *value* estimate (used for GAE bootstrapping), never to
        # the action distributions, so it is harmless there.
        value_in_dim = trunk_hidden + value_extra_hidden
        if value_hidden_dim > 0:
            self.value_head = nn.Sequential(
                nn.Dropout(value_dropout),
                nn.Linear(value_in_dim, value_hidden_dim), nn.ReLU(),
                nn.Dropout(value_dropout),
                nn.Linear(value_hidden_dim, 1),
            )
        else:
            self.value_head = nn.Sequential(
                nn.Dropout(value_dropout),
                nn.Linear(value_in_dim, 1),
            )

        # Learnable log_std for direction heads (move_dir, kick_dir).
        # Single scalar (isotropic) rather than one-per-axis: the direction
        # heads output a unit vector, so the x/y Gaussian error is really a
        # proxy for angular error, and there is no principled reason for the
        # x and y axes to have independently-tunable spreads. A single log_std
        # broadcasts against both dims in DirectionHead's Normal(...) — see
        # ai_trainer_knowledge.md "Direction heads: log_std and KL".
        # Initialized from config (dir_log_std_init). Lower = tighter sampling
        # (σ≈0.13 at -2.0, σ≈0.22 at -1.5, σ≈0.37 at -1.0).
        # KL spikes if too tight: a 4° mean shift at σ=0.13 → ratio~50×.
        # Clamped during forward() to [dir_log_std_min, dir_log_std_max].
        self.move_dir_log_std = nn.Parameter(torch.full((1,), dir_log_std_init))
        _kick_ls_init = kick_dir_log_std_init if kick_dir_log_std_init is not None else dir_log_std_init
        self.kick_dir_log_std = nn.Parameter(torch.full((1,), _kick_ls_init))
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
        self_ai_type: Optional[torch.Tensor] = None,   # (batch, AI_TYPE_ONE_HOT_DIM)
        other_ai_type: Optional[torch.Tensor] = None,  # (batch, MAX_OTHER_PLAYERS, AI_TYPE_ONE_HOT_DIM)
    ) -> ExecutionHeadsRaw:
        dec_flat = flatten_decision_heads(decision_heads)
        entity_ctx, self_embed_raw, other_embed_raw = self.entity_encoder(
            self_feat, other_feat, exists_mask, return_embeds=True,
            ball_feat=ball_feat, global_feat=global_feat,
            extra_query_bias=self.decision_query_proj(dec_flat),
        )
        h = torch.cat([
            entity_ctx,
            self.self_mlp(self_feat),
            self.ball_mlp(ball_feat),
            self.global_mlp(global_feat),
            self.decision_mlp(dec_flat),
        ], dim=-1)
        h = self.trunk(h)

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
            value=self.value_head(value_input),
        )

    @classmethod
    def from_config(
        cls,
        trunk_hidden_override: Optional[int] = None,
        shared_entity_encoder: Optional[nn.Module] = None,
        shared_ball_mlp: Optional[nn.Module] = None,
        shared_global_mlp: Optional[nn.Module] = None,
    ) -> "ExecutionNetwork":
        """Build from ai_config.json.

        trunk_hidden_override: overrides trunk_hidden (used for value_net sizing).
        shared_*: pass DecisionNetwork modules to share weights (see PPOTrainer.from_config).
        """
        from footballcoach.ai.config import load_ai_config
        full_cfg = load_ai_config()
        cfg = full_cfg["network"]
        ppo_cfg = full_cfg["ppo"]
        # exec_trunk_hidden: independent trunk capacity for execution network.
        # Falls back to trunk_hidden when absent or null.
        exec_trunk = cfg.get("exec_trunk_hidden") or cfg["trunk_hidden"]
        trunk = trunk_hidden_override if trunk_hidden_override is not None else exec_trunk
        return cls(
            latent_dim=cfg["latent_dim"],
            entity_embed_dim=cfg["entity_embed_dim"],
            num_attention_heads=cfg["num_attention_heads"],
            self_mlp_hidden=cfg["self_mlp_hidden"],
            ball_mlp_hidden=cfg["ball_mlp_hidden"],
            global_mlp_hidden=cfg["global_mlp_hidden"],
            decision_mlp_hidden=cfg["decision_mlp_hidden"],
            trunk_hidden=trunk,
            dir_log_std_init=ppo_cfg.get("dir_log_std_init", -2.0),
            kick_dir_log_std_init=ppo_cfg.get("kick_dir_log_std_init", None),
            value_extra_hidden=cfg.get("value_extra_hidden", 16),
            value_hidden_dim=cfg.get("value_hidden_dim", 0),
            value_dropout=cfg.get("value_dropout", 0.0),
            inter_player_num_heads=cfg.get("inter_player_attn_heads", 0),
            shared_entity_encoder=shared_entity_encoder,
            shared_ball_mlp=shared_ball_mlp,
            shared_global_mlp=shared_global_mlp,
        )

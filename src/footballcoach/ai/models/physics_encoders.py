"""Frozen ball/player physics-dynamics encoders, wired into the main
DecisionNetwork/ExecutionNetwork as an optional, opt-in feature.

See ai/knowledge.md's "Frozen physics-dynamics encoders" section and
agent_plans/ball_physics_pretrain_plan.md section 8 for the full design.

Both blocks below are owned EXCLUSIVELY by DecisionNetwork -- ExecutionNetwork
never constructs one, it only reads the already-computed output off the
DecisionHeadsRaw it receives (see decision_network.py/execution_network.py),
so each frozen encoder runs at most once per observation instead of once per
network.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from footballcoach.ai.physics_pretrain.live_encoder_features import (
    PhysicsPitchConstants,
    ball_live_to_physics_input,
    load_frozen_ball_encoder,
    load_frozen_player_encoder,
    load_physics_pitch_constants,
    player_obs_to_physics_input,
)


class BallPhysicsFeatureBlock(nn.Module):
    """Frozen BallDynamicsEncoder + its frozen auxiliary heads, combined into
    one (batch, output_dim) tensor per forward call -- latent concatenated
    with every auxiliary head's output, matching
    physics_value_net.py's ``ball_full = torch.cat([ball_latent, *ball_aux], dim=-1)``
    pattern. Not trainable: both the encoder and the aux heads are already
    frozen (requires_grad_(False)) by load_frozen_ball_encoder(), never given
    an optimizer param group, and excluded from PPOTrainer's saved
    checkpoint state_dict()."""

    def __init__(self, checkpoint_path: str):
        super().__init__()
        encoder, aux_heads, cfg = load_frozen_ball_encoder(checkpoint_path)
        self.encoder = encoder
        self.aux_heads = aux_heads
        self.output_dim = int(cfg["latent_dim"]) + sum(h.out_features for h in aux_heads.values())
        self.pitch_constants = load_physics_pitch_constants()

        from footballcoach.ai.config import load_ai_config
        obs_cfg = load_ai_config()["observation"]
        self._spin_norm = float(obs_cfg["ball_spin_nn_norm_rad_s"])
        self._height_norm = float(obs_cfg.get("height_norm_m", 3.0))
        self.eval()

    def forward(self, ball_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = ball_live_to_physics_input(
                ball_feat, global_feat, self.pitch_constants,
                live_ball_spin_nn_norm_rad_s=self._spin_norm, live_height_norm_m=self._height_norm,
            )
            latent = self.encoder(x)
            aux_outputs = [head(latent) for head in self.aux_heads.values()]
            return torch.cat([latent, *aux_outputs], dim=-1)


class PlayerPhysicsFeatureBlock(nn.Module):
    """Frozen PlayerDynamicsEncoder + its frozen auxiliary heads, combined
    exactly like BallPhysicsFeatureBlock above. forward() accepts either
    self_feat (batch, PLAYER_FEATURE_DIM) or other_feat
    (batch, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM) -- player_obs_to_physics_input
    and the frozen encoder/heads both broadcast over the extra leading dim
    for free, so the same block instance handles both call shapes."""

    def __init__(self, checkpoint_path: str):
        super().__init__()
        encoder, aux_heads, cfg = load_frozen_player_encoder(checkpoint_path)
        self.encoder = encoder
        self.aux_heads = aux_heads
        self.output_dim = int(cfg["latent_dim"]) + sum(h.out_features for h in aux_heads.values())
        self.pitch_constants = load_physics_pitch_constants()
        self.eval()

    def forward(self, player_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = player_obs_to_physics_input(player_feat, global_feat, self.pitch_constants)
            latent = self.encoder(x)
            aux_outputs = [head(latent) for head in self.aux_heads.values()]
            return torch.cat([latent, *aux_outputs], dim=-1)

"""Player-dynamics encoder/decoder network.

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

See agent_plans/player_physics_pretrain_plan.md. Directly mirrors
``ball_dynamics_net.py`` (agent_plans/ball_physics_pretrain_plan.md section
6) -- ``PlayerDynamicsEncoder``/``Decoder``/``Autoencoder`` are the ``Ball``
equivalents' exact structural analogues, sharing the identity-shortcut init
helpers via ``identity_shortcut.py``. Only ``PlayerDynamicsEncoder`` is ever
meant to be frozen/shipped as a live-network input (a *future*,
separately-approved change, out of scope for this pass -- see the ball
plan's §8/§11); the decoder heads exist purely to supervise the encoder
during offline pretraining and are discarded afterward.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from footballcoach.ai.physics_pretrain.identity_shortcut import (
    _init_identity_shortcut_decoder,
    _init_identity_shortcut_linear,
)
from footballcoach.ai.physics_pretrain.player_episode_gen import N_INPUT_FIELDS, N_TARGET_FIELDS_PER_HORIZON

# Raw input fields 0:7 (pos_x, pos_y, vel_x, vel_y, heading_sin, heading_cos,
# stamina) and target fields 0:7 use the exact same normalization/order (see
# player_episode_gen.py's _encode_input/_encode_target), so an identity
# mapping between them is meaningful -- this is what the identity-shortcut
# init below exploits, exactly mirroring the ball net's N_IDENTITY_SHORTCUT_
# FIELDS=9/Z_FIELD_INDEX pattern.
N_IDENTITY_SHORTCUT_FIELDS = 7

# Index of the current-stamina-fraction field within the 7 identity-shortcut
# fields above. Unlike the other 6 (position x/y, velocity x/y, heading
# sin/cos, each either sign), stamina is PROVABLY confined to [0, 1]
# (drain_stamina/regen_stamina both clamp) -- the player-net equivalent of
# the ball net's Z_FIELD_INDEX (pos_z). See
# ``identity_shortcut._init_identity_shortcut_decoder``'s docstring for what
# this buys.
STAMINA_FIELD_INDEX = 6


class PlayerDynamicsEncoder(nn.Module):
    """``input(N_INPUT_FIELDS) -> Linear(hidden) -> ReLU -> Linear(hidden) -> ReLU ->
    Linear(bottleneck) -> ReLU -> [concat raw input[0:7] if identity_shortcut] -> Linear(latent_dim)``.

    Structurally identical to ``BallDynamicsEncoder`` (see its docstring for
    the full rationale behind the bottleneck layer and the identity-shortcut
    concat) -- only the field counts differ (``N_IDENTITY_SHORTCUT_FIELDS``
    here is 7, not 9; no spin/height axis).
    """

    def __init__(
        self, input_dim: int = N_INPUT_FIELDS, hidden_dim: int = 64, latent_dim: int = 16, bottleneck_dim: int = 32,
        identity_shortcut: bool = False, identity_shortcut_noise_std: float = 0.0,
        concat_all_input_fields: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.identity_shortcut = identity_shortcut
        self.concat_all_input_fields = concat_all_input_fields
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim), nn.ReLU(),
        )
        if identity_shortcut:
            concat_dim = input_dim if concat_all_input_fields else N_IDENTITY_SHORTCUT_FIELDS
        else:
            concat_dim = 0
        self.out = nn.Linear(bottleneck_dim + concat_dim, latent_dim)
        if identity_shortcut:
            if latent_dim < N_IDENTITY_SHORTCUT_FIELDS:
                raise ValueError(
                    f"latent_dim ({latent_dim}) must be >= N_IDENTITY_SHORTCUT_FIELDS "
                    f"({N_IDENTITY_SHORTCUT_FIELDS}) to use identity_shortcut"
                )
            _init_identity_shortcut_linear(self.out, bottleneck_dim, N_IDENTITY_SHORTCUT_FIELDS, identity_shortcut_noise_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)
        if self.identity_shortcut:
            concat_slice = x if self.concat_all_input_fields else x[..., :N_IDENTITY_SHORTCUT_FIELDS]
            feat = torch.cat([feat, concat_slice], dim=-1)
        return self.out(feat)


class PlayerDynamicsDecoder(nn.Module):
    """One decoder shared across all horizons, conditioned on 3 horizon
    features concatenated onto the latent: ``latent_dim+3 -> hidden ->
    ReLU -> N_TARGET_FIELDS_PER_HORIZON``.

    Output layout: ``[pos_x, pos_y, vel_x, vel_y, heading_sin, heading_cos,
    stamina, out_of_bounds_logit, goal_scored_logit]`` -- first 7 are direct
    regression targets, last 2 are BCE logits. Structurally identical to
    ``BallDynamicsDecoder`` -- see its docstring for the full rationale
    behind the 3 horizon features (``t_norm``, ``t_norm^2``,
    ``log1p(horizon_s)``) and the identity-shortcut init.
    """

    def __init__(
        self, latent_dim: int = 16, horizons_s: list[float] | None = None, hidden_dim: int = 32,
        identity_shortcut: bool = False, identity_shortcut_noise_std: float = 0.0,
    ):
        super().__init__()
        horizons_s = horizons_s if horizons_s is not None else [0.2, 1.0, 3.0, 5.0, 10.0]
        self.n_horizons = len(horizons_s)
        self.latent_dim = latent_dim
        horizons_t = torch.tensor(horizons_s, dtype=torch.float32)
        self._t_norm_max = float(horizons_t.max())
        t_norm = horizons_t / horizons_t.max()
        self.register_buffer("t_norm", t_norm)
        self.register_buffer("t_norm_sq", t_norm ** 2)
        self.register_buffer("log_horizons", torch.log1p(horizons_t))
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, N_TARGET_FIELDS_PER_HORIZON),
        )
        if identity_shortcut:
            if latent_dim < N_IDENTITY_SHORTCUT_FIELDS:
                raise ValueError(
                    f"latent_dim ({latent_dim}) must be >= N_IDENTITY_SHORTCUT_FIELDS "
                    f"({N_IDENTITY_SHORTCUT_FIELDS}) to use identity_shortcut"
                )
            _init_identity_shortcut_decoder(
                self.net, N_IDENTITY_SHORTCUT_FIELDS, identity_shortcut_noise_std,
                nonneg_field_index=STAMINA_FIELD_INDEX,
            )

    def forward(self, latent: torch.Tensor) -> list[torch.Tensor]:
        batch = latent.shape[0]
        return [
            self.net(torch.cat([
                latent,
                self.t_norm[h].expand(batch, 1),
                self.t_norm_sq[h].expand(batch, 1),
                self.log_horizons[h].expand(batch, 1),
            ], dim=-1))
            for h in range(self.n_horizons)
        ]

    def forward_at(self, latent: torch.Tensor, horizon_s: float) -> torch.Tensor:
        """Query the decoder at an ARBITRARY horizon -- see
        ``BallDynamicsDecoder.forward_at``'s docstring, identical rationale."""
        batch = latent.shape[0]
        t_norm = horizon_s / self._t_norm_max
        feat = latent.new_tensor([t_norm, t_norm ** 2, math.log1p(horizon_s)])
        return self.net(torch.cat([latent, feat.expand(batch, 3)], dim=-1))


class PlayerDynamicsAutoencoder(nn.Module):
    """Thin training-only wrapper composing encoder + the shared decoder.

    Not itself a saved artifact -- only ``encoder.state_dict()`` becomes the
    real, permanent output, exactly mirroring ``BallDynamicsAutoencoder``.
    """

    def __init__(
        self,
        input_dim: int = N_INPUT_FIELDS,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        horizons_s: list[float] | None = None,
        decoder_hidden_dim: int = 32,
        encoder_bottleneck_dim: int = 32,
        identity_shortcut: bool = False,
        identity_shortcut_noise_std: float = 0.0,
        encoder_concat_all_input_fields: bool = False,
        decoder_identity_shortcut: bool | None = None,
    ):
        super().__init__()
        self.encoder = PlayerDynamicsEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, bottleneck_dim=encoder_bottleneck_dim,
            identity_shortcut=identity_shortcut, identity_shortcut_noise_std=identity_shortcut_noise_std,
            concat_all_input_fields=encoder_concat_all_input_fields,
        )
        decoder_identity_shortcut = identity_shortcut if decoder_identity_shortcut is None else decoder_identity_shortcut
        self.decoder = PlayerDynamicsDecoder(
            latent_dim=latent_dim, horizons_s=horizons_s, hidden_dim=decoder_hidden_dim,
            identity_shortcut=decoder_identity_shortcut, identity_shortcut_noise_std=identity_shortcut_noise_std,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        latent = self.encoder(x)
        return latent, self.decoder(latent)

    @classmethod
    def from_config(cls) -> "PlayerDynamicsAutoencoder":
        from footballcoach.ai.config import load_ai_config
        cfg = load_ai_config()["physics_pretrain"]["player"]
        return cls(
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            horizons_s=cfg["horizons_s"],
            decoder_hidden_dim=cfg.get("decoder_hidden_dim", 32),
            encoder_bottleneck_dim=cfg.get("encoder_bottleneck_dim", 32),
            identity_shortcut=cfg.get("identity_shortcut_enabled", False),
            identity_shortcut_noise_std=cfg.get("identity_shortcut_noise_std", 0.0),
            encoder_concat_all_input_fields=cfg.get("encoder_concat_all_input_fields", False),
            decoder_identity_shortcut=cfg.get("decoder_identity_shortcut_enabled"),
        )

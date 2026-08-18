"""Ball-dynamics encoder/decoder network.

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

See agent_plans/ball_physics_pretrain_plan.md section 6. Only
``BallDynamicsEncoder`` is ever frozen/shipped as a live-network input (a
*future*, separately-approved change -- see the plan's §8); the decoder
heads exist purely to supervise the encoder during offline pretraining and
are discarded afterward.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from footballcoach.ai.physics_pretrain.ball_episode_gen import N_INPUT_FIELDS, N_TARGET_FIELDS_PER_HORIZON


class BallDynamicsEncoder(nn.Module):
    """``input(14) -> Linear(hidden) -> ReLU -> Linear(hidden) -> ReLU -> Linear(latent_dim)``.

    No dropout/batchnorm: this network is meant to run frozen and
    deterministic at inference time (see §6.1's note on avoiding the
    ``value_dropout``-style ".eval() not called during rollout" caveat
    entirely rather than working around it).
    """

    def __init__(self, input_dim: int = N_INPUT_FIELDS, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BallDynamicsDecoder(nn.Module):
    """Five independent horizon heads, each ``latent_dim -> 32 -> ReLU -> 11``.

    Per-head output layout (§6.2): ``[pos_x, pos_y, height_m, vel_x, vel_y,
    vel_z, spin_x, spin_y, spin_z, out_of_bounds_logit, goal_scored_logit]``
    -- first 9 are direct regression targets, last 2 are BCE logits.
    """

    def __init__(self, latent_dim: int = 16, n_horizons: int = 5, head_hidden: int = 32):
        super().__init__()
        self.n_horizons = n_horizons
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, head_hidden), nn.ReLU(),
                nn.Linear(head_hidden, N_TARGET_FIELDS_PER_HORIZON),
            )
            for _ in range(n_horizons)
        ])

    def forward(self, latent: torch.Tensor) -> list[torch.Tensor]:
        return [head(latent) for head in self.heads]


class BallDynamicsAutoencoder(nn.Module):
    """Thin training-only wrapper composing encoder + all decoder heads.

    Not itself a saved artifact -- only ``encoder.state_dict()`` becomes the
    real, permanent output (§6.3, §7).
    """

    def __init__(
        self,
        input_dim: int = N_INPUT_FIELDS,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        n_horizons: int = 5,
        head_hidden: int = 32,
    ):
        super().__init__()
        self.encoder = BallDynamicsEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = BallDynamicsDecoder(latent_dim=latent_dim, n_horizons=n_horizons, head_hidden=head_hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        latent = self.encoder(x)
        return latent, self.decoder(latent)

    @classmethod
    def from_config(cls) -> "BallDynamicsAutoencoder":
        from footballcoach.ai.config import load_ai_config
        cfg = load_ai_config()["physics_pretrain"]["ball"]
        return cls(
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            n_horizons=len(cfg["horizons_s"]),
        )

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
    """``input(N_INPUT_FIELDS) -> Linear(hidden) -> LeakyReLU -> Linear(hidden) -> LeakyReLU ->
    Linear(bottleneck) -> LeakyReLU -> [concat raw input[0:7] if identity_shortcut] -> Linear(latent_dim)``.

    Structurally identical to ``BallDynamicsEncoder`` (see its docstring for
    the full rationale behind the bottleneck layer, the identity-shortcut
    concat, and ``leaky_relu_negative_slope``) -- only the field counts
    differ (``N_IDENTITY_SHORTCUT_FIELDS`` here is 7, not 9; no spin/height
    axis).
    """

    def __init__(
        self, input_dim: int = N_INPUT_FIELDS, hidden_dim: int = 64, latent_dim: int = 16, bottleneck_dim: int = 32,
        identity_shortcut: bool = False, identity_shortcut_noise_std: float = 0.0,
        concat_all_input_fields: bool = False, leaky_relu_negative_slope: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.identity_shortcut = identity_shortcut
        self.concat_all_input_fields = concat_all_input_fields
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
            nn.Linear(hidden_dim, bottleneck_dim), nn.LeakyReLU(negative_slope=leaky_relu_negative_slope),
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

    def has_horizon(self, horizon_s: float) -> bool:
        """Always True -- see BallDynamicsDecoder.has_horizon's identical
        rationale (interface parity with PlayerDynamicsLinearDecoder, which
        is NOT always True)."""
        return True


# pos_x, pos_y, vel_x, vel_y -- see PlayerDynamicsLinearDecoder.
N_LINEAR_DECODER_TARGET_FIELDS = 4

# See BallDynamicsLinearDecoder's identical constant/rationale.
_LINEAR_DECODER_HORIZON_ATOL = 1e-6


class PlayerDynamicsLinearDecoder(nn.Module):
    """Player analogue of ``BallDynamicsLinearDecoder`` -- see its docstring
    for the full rationale (an affine decoder can't represent ``position ~=
    velocity * t``, since that needs an episode-specific latent-derived
    quantity multiplied by a horizon-derived time feature; independent
    per-horizon linear heads sidestep this by not taking a time INPUT at
    all). ``n_horizons + 1`` (the configured ``horizons_s``, plus ``0.0``)
    independent ``Linear(latent_dim, 4)`` heads.

    Predicts position + velocity ONLY (``pos_x, pos_y, vel_x, vel_y``) --
    no heading, no stamina, no out_of_bounds/goal_scored logits.
    ``forward``/``forward_at`` still return the full ``N_TARGET_FIELDS_PER_
    HORIZON``-wide shape ``PlayerDynamicsDecoder`` does (zero-padding the
    missing heading/stamina/logit columns), so every downstream loss/
    breakdown/logging function works unchanged -- those constant-zero
    columns carry no gradient. Unlike the ball version, heading_mse/
    stamina_mse have no existing 0.0-disables weight of their own (they were
    always unconditionally summed into ``compute_loss``'s ``total``) --
    ``train_player_dynamics.py`` now threads ``heading_weight``/
    ``stamina_weight`` (default 1.0, unweighted, matching prior behaviour)
    through every loss function specifically so ``linear_decoder_enabled``
    can zero them out too, same convention as ``bce_weight``, instead of
    baking a meaningless constant offset into every reported train/val
    loss.

    ``0.0`` is always included as a registered horizon in addition to
    ``horizons_s``, for the t=0 autoencoding sanity check
    (``forward_at(latent, 0.0)``) -- not included in ``forward()``'s
    returned list.
    """

    def __init__(self, latent_dim: int, horizons_s: list[float] | None = None):
        super().__init__()
        horizons_s = horizons_s if horizons_s is not None else [0.2, 1.0, 3.0, 5.0, 10.0]
        self.n_horizons = len(horizons_s)
        self.latent_dim = latent_dim
        self._horizons = (0.0,) + tuple(horizons_s)
        self.net = nn.Linear(latent_dim, len(self._horizons) * N_LINEAR_DECODER_TARGET_FIELDS)

    def _heads_out(self, latent: torch.Tensor) -> torch.Tensor:
        batch = latent.shape[0]
        return self.net(latent).view(batch, len(self._horizons), N_LINEAR_DECODER_TARGET_FIELDS)

    def _pad(self, out4: torch.Tensor) -> torch.Tensor:
        pad = out4.new_zeros(out4.shape[0], N_TARGET_FIELDS_PER_HORIZON - N_LINEAR_DECODER_TARGET_FIELDS)
        return torch.cat([out4, pad], dim=-1)

    def forward(self, latent: torch.Tensor) -> list[torch.Tensor]:
        all_out = self._heads_out(latent)
        return [self._pad(all_out[:, i, :]) for i in range(1, len(self._horizons))]

    @property
    def unpadded_output_dim(self) -> int:
        """See BallDynamicsLinearDecoder.unpadded_output_dim's identical rationale."""
        return self.n_horizons * N_LINEAR_DECODER_TARGET_FIELDS

    def forward_all_unpadded(self, latent: torch.Tensor) -> torch.Tensor:
        """See BallDynamicsLinearDecoder.forward_all_unpadded's identical rationale
        (player analogue: pos_x/pos_y/vel_x/vel_y per horizon, no padding).
        Also mirrors its leading-dims-preserving shape handling -- needed
        here in particular, since PlayerPhysicsFeatureBlock's other_feat
        call passes a 3D (batch, MAX_OTHER_PLAYERS, latent_dim) latent."""
        leading_shape = latent.shape[:-1]
        all_out = self.net(latent).view(*leading_shape, len(self._horizons), N_LINEAR_DECODER_TARGET_FIELDS)
        return all_out[..., 1:, :].reshape(*leading_shape, -1)

    def has_horizon(self, horizon_s: float) -> bool:
        return any(abs(horizon_s - h) <= _LINEAR_DECODER_HORIZON_ATOL for h in self._horizons)

    def forward_at(self, latent: torch.Tensor, horizon_s: float) -> torch.Tensor:
        for i, h in enumerate(self._horizons):
            if abs(horizon_s - h) <= _LINEAR_DECODER_HORIZON_ATOL:
                return self._pad(self._heads_out(latent)[:, i, :])
        raise ValueError(
            f"PlayerDynamicsLinearDecoder has no head registered for horizon_s={horizon_s} "
            f"(registered: {self._horizons}) -- callers must check has_horizon() first."
        )


class PlayerDynamicsAutoencoder(nn.Module):
    """Thin training-only wrapper composing encoder + the shared decoder +
    four auxiliary linear heads reading the latent directly.

    Not itself a saved artifact -- only ``encoder.state_dict()`` becomes the
    real, permanent output, exactly mirroring ``BallDynamicsAutoencoder``.
    Every auxiliary head below is a single ``nn.Linear`` off the RAW latent
    (no hidden layer, no ReLU, no horizon conditioning) and is ALWAYS
    constructed -- they're a handful of parameters each, and whether they
    actually train is controlled purely by their config loss weight, never
    by a constructor flag (same convention as ``BallDynamicsAutoencoder``'s
    ``crossing_head``/``resting_head``). Being linear-off-the-latent is the
    point: whatever they predict, the latent is forced to encode LINEARLY.

    ``crossing_head``: ``Linear(latent_dim, 4)`` -> ``(pos_x, pos_y,
    crosses_logit, delta_t)`` of the first out_of_bounds/goal_scored
    crossing. ``crosses_logit`` is a BCE-with-logits classifier -- "does
    this row have a real crossing/already-crossed instance to report at
    all" (true whenever ``crossing_dt`` isn't the -1 'genuinely never'
    sentinel) -- and ``delta_t`` is a plain regression trained ONLY on rows
    where that classifier's target is true (no -1 sentinel mixed into the
    regression itself). Split out from a single combined delta_t regression
    (see ``train_player_dynamics._crossing_head_loss``'s docstring) because
    the old single-regression version forced MSE to blend two qualitatively
    different signals -- "will it cross at all" and "when, given it
    crosses" -- into one number, and for any input the network was even
    slightly unsure about, the MSE-optimal prediction was a weighted
    average of "-1" and "some real time", landing confidently on neither
    and inflating error on both. Splitting removes that blend entirely: the
    classifier only ever has to answer yes/no, and the regression only ever
    sees real, same-scale values (0 for an already-crossed row, a real
    positive delta_t for a genuine future crossing) with no -1 gap to
    average against. See ``PlayerDynamicsDataset``'s ``crossing_pos``/
    ``crossing_dt``/``crossing_mask`` and ``physics_pretrain.player.
    crossing_pos_loss_weight``/``crossing_crosses_loss_weight``/
    ``crossing_dt_loss_weight``.

    ``goal_dist_delta_head``: ``Linear(latent_dim, 2)`` -> the CHANGE in
    distance-to-the-closest-point-of-each-goal-mouth between t=0 and t=3.0s,
    one output per goal (index 0 = left goal at ``x = -half_length``, index
    1 = right goal at ``x = +half_length``). Negative = the player got
    closer to that goal. No ball equivalent. See
    ``train_player_dynamics._goal_dist_delta_targets`` and
    ``physics_pretrain.player.goal_dist_delta_loss_weight``.

    ``short_horizon_head_0_2s``/``short_horizon_head_1_0s``: two
    ``Linear(latent_dim, 4)`` diagnostic PROBES -> ``(pos_x, pos_y, vel_x,
    vel_y)`` at exactly 0.2s and 1.0s respectively. Deliberately duplicate
    what the shared, horizon-conditioned decoder already predicts at those
    horizons, with NO horizon/time features involved -- an extra, purely
    additive gradient path forcing the raw latent to encode short-horizon
    state linearly. Motivated by ``pos_rmse`` being anomalously bad at
    exactly those two shortest horizons relative to a trivial persistence
    baseline. The decoder's own per-horizon predictions (what the report,
    inspector and main loss all use) are completely unaffected by these.
    See ``physics_pretrain.player.short_horizon_probe_loss_weight``.
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
        linear_decoder: bool = False,
        leaky_relu_negative_slope: float = 0.0,
    ):
        super().__init__()
        self.encoder = PlayerDynamicsEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, bottleneck_dim=encoder_bottleneck_dim,
            identity_shortcut=identity_shortcut, identity_shortcut_noise_std=identity_shortcut_noise_std,
            concat_all_input_fields=encoder_concat_all_input_fields,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
        )
        self.linear_decoder = linear_decoder
        if linear_decoder:
            # decoder_hidden_dim/decoder_identity_shortcut/
            # identity_shortcut_noise_std don't apply -- see
            # PlayerDynamicsLinearDecoder's own docstring.
            self.decoder = PlayerDynamicsLinearDecoder(latent_dim=latent_dim, horizons_s=horizons_s)
        else:
            decoder_identity_shortcut = identity_shortcut if decoder_identity_shortcut is None else decoder_identity_shortcut
            self.decoder = PlayerDynamicsDecoder(
                latent_dim=latent_dim, horizons_s=horizons_s, hidden_dim=decoder_hidden_dim,
                identity_shortcut=decoder_identity_shortcut, identity_shortcut_noise_std=identity_shortcut_noise_std,
            )
        # Auxiliary linear heads off the raw latent -- see the class
        # docstring. Always present; trained iff their config weight != 0.
        self.crossing_head = nn.Linear(latent_dim, 4)
        self.goal_dist_delta_head = nn.Linear(latent_dim, 2)
        self.short_horizon_head_0_2s = nn.Linear(latent_dim, 4)
        self.short_horizon_head_1_0s = nn.Linear(latent_dim, 4)

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
            linear_decoder=cfg.get("linear_decoder_enabled", False),
            leaky_relu_negative_slope=cfg.get("encoder_leaky_relu_negative_slope", 0.0),
        )

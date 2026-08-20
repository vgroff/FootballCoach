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

import math

import torch
import torch.nn as nn

from footballcoach.ai.physics_pretrain.ball_episode_gen import N_INPUT_FIELDS, N_TARGET_FIELDS_PER_HORIZON
from footballcoach.ai.physics_pretrain.identity_shortcut import (
    _init_identity_shortcut_decoder,
    _init_identity_shortcut_linear,
)

# Raw input fields 0:9 (pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,spin_x,spin_y,spin_z)
# and target fields 0:9 use the exact same normalization/order (see
# ball_episode_gen.py's _encode_input/_encode_target), so an identity
# mapping between them is meaningful -- this is what the identity-shortcut
# init below exploits.
N_IDENTITY_SHORTCUT_FIELDS = 9

# Index of pos_z (height) within the 9 identity-shortcut fields above.
# Unlike the other 8 (position x/y, all 3 velocity components, all 3 spin
# components), which can each be either sign, height is PROVABLY never
# negative -- the ground floor is the ball's own radius (> 0), confirmed by
# scanning the generated dataset: zero negative z values anywhere, in
# inputs or targets. See `_init_identity_shortcut_decoder`'s docstring for
# what this buys.
Z_FIELD_INDEX = 2


class BallDynamicsEncoder(nn.Module):
    """``input(N_INPUT_FIELDS) -> Linear(hidden) -> ReLU -> Linear(hidden) -> ReLU ->
    Linear(bottleneck) -> ReLU -> [concat raw input[0:9] if identity_shortcut] -> Linear(latent_dim)``.

    The extra ``bottleneck`` layer compresses more gradually than going
    straight from ``hidden_dim`` to ``latent_dim`` in one step (e.g.
    196 -> 24 directly is an ~8x compression in a single Linear layer).

    ``identity_shortcut``: when enabled, concatenates the raw
    pos/vel/spin input fields (``x[0:N_IDENTITY_SHORTCUT_FIELDS]``) onto the
    bottleneck output right before the final Linear to latent, and hand-
    initializes that Linear so ``latent[0:9] ~= x[0:9]`` at init (see
    ``_init_identity_shortcut_linear``). Everything still routes through the
    latent -- the decoder never sees the raw input -- so this doesn't
    undermine the "what does the latent need to capture" probe the way a
    decoder-side skip around the latent would; it only gives the encoder's
    final projection a cheap, near-linear path to preserve those 9 values
    instead of relying on 3 ReLU layers to do it losslessly. Paired with
    ``BallDynamicsDecoder``'s own identity-shortcut init, this eases the
    encoder+decoder round trip (the post-training t=0 autoencoding sanity
    check) without eliminating it, provided ``identity_shortcut_noise_std``
    is nonzero.

    ``concat_all_input_fields`` (requires ``identity_shortcut``): widens
    that same concat from just the 9 identity fields to ALL of ``x``
    (restitution, pitch/goal dims, the 6 engineered features too), still
    ordered identity-fields-first so the hand-init above is unaffected.
    Motivated the same way as the identity fields: `self.out` is a single
    Linear with no nonlinearity of its own, so anything it needs exactly
    (e.g. a pitch-dim ratio, or a nonlinear engineered feature like
    speed_norm that a linear layer can't itself re-derive from raw
    components) either has to survive 3 compressing ReLU layers losslessly,
    or can just be handed to it directly for free.

    No dropout/batchnorm: this network is meant to run frozen and
    deterministic at inference time (see §6.1's note on avoiding the
    ``value_dropout``-style ".eval() not called during rollout" caveat
    entirely rather than working around it).
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
        # When true (and identity_shortcut is also on), concatenates ALL
        # raw input fields onto the bottleneck output before the final
        # latent Linear, not just the 9 identity-shortcut ones -- gives
        # that layer direct, uncompressed access to restitution/pitch-goal
        # dims/engineered features too, instead of relying on `trunk` to
        # preserve them losslessly through 3 ReLU layers. Cheap and safe
        # to add: `self.out` has no ReLU of its own, so the extra
        # concatenated columns (zero-initialized, same as everything but
        # the hand-set identity block -- see _init_identity_shortcut_linear)
        # aren't a dead fixed point the way the decoder's spare units were;
        # ordinary gradient reaches them from the very first backward pass.
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
            # shortcut_start_col=bottleneck_dim is correct regardless of
            # concat_all_input_fields: the identity fields are always the
            # first N_IDENTITY_SHORTCUT_FIELDS columns of `x` (and so of
            # the concatenated block), whether the concat is just those 9
            # fields or the full x -- see forward() below.
            _init_identity_shortcut_linear(self.out, bottleneck_dim, N_IDENTITY_SHORTCUT_FIELDS, identity_shortcut_noise_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)
        if self.identity_shortcut:
            concat_slice = x if self.concat_all_input_fields else x[..., :N_IDENTITY_SHORTCUT_FIELDS]
            feat = torch.cat([feat, concat_slice], dim=-1)
        return self.out(feat)


class BallDynamicsDecoder(nn.Module):
    """One decoder shared across all horizons, conditioned on 3 horizon
    features concatenated onto the latent: ``latent_dim+3 -> hidden ->
    ReLU -> 11``.

    Output layout (§6.2): ``[pos_x, pos_y, height_m, vel_x, vel_y, vel_z,
    spin_x, spin_y, spin_z, out_of_bounds_logit, goal_scored_logit]`` --
    first 9 are direct regression targets, last 2 are BCE logits.

    Only the DECODER sees the horizon -- the encoder never does, so the
    latent is pushed to represent general ball-dynamics state rather than
    horizon-specific features.

    The 3 horizon features are ``t_norm`` (``horizon_s / max(horizons_s)``,
    in ``(0, 1]``), ``t_norm^2``, and ``log1p(horizon_s)`` (``log(1 +
    horizon_s)``). Log-only was the original design, but most of the
    underlying kinematics is close to linear/quadratic in RAW time for
    short intervals (position ~ velocity*t, gravity's t^2 term, spin decay
    ~ linear-per-tick) -- log(t) forces the network to implicitly apply
    exp() to recover that, which a linear layer could otherwise represent
    exactly from raw/squared t. Feeding all 3 lets the network use
    whichever combination fits a given output best rather than committing
    to one hypothesis.

    ``log1p`` rather than plain ``log`` (which is undefined at t=0) for a
    reason beyond just avoiding ``-inf``: ``log1p(t) ≈ t`` for small t (its
    derivative at t=0 is exactly 1), so near t=0 it's ALSO close to linear
    -- it doesn't fight the near-linear-kinematics reasoning above the way
    an arbitrarily-shifted ``log(t+eps)`` still would for small eps, it's
    just a near-redundant near-linear term there. At large t it still
    flattens/compresses the way plain log(t) does, which is the actually
    useful part -- treating "7s vs 10s" as closer together than "0.2s vs
    0.5s", matching how little changes per extra second once the ball's
    likely already bounced/frozen. Same pattern already used elsewhere in
    this codebase for time_remaining normalization (``ai/obs/encoder.py``),
    for the same "spread out the low end, compress the high end" reason.
    Well-defined and unremarkable at t=0 (``log1p(0)=0``) with no epsilon
    to justify -- also what makes ``forward_at()``'s t=0 autoencoding
    sanity check (see train_ball_dynamics.py) possible without any
    special-casing.

    ``identity_shortcut``: when enabled, hand-initializes
    ``N_IDENTITY_SHORTCUT_FIELDS`` of the hidden layer (2 ReLU units per
    value) to pass the corresponding latent dims straight through to output
    fields 0:9, ignoring the horizon features -- see
    ``_init_identity_shortcut_decoder``. Combined with the encoder's own
    ``identity_shortcut``, this makes the whole round trip start close to
    identity at t=0 (and close to the persistence baseline at every other
    horizon), rather than starting from a random init that has to learn the
    copy-through behaviour from scratch. ``identity_shortcut_noise_std``
    keeps this imperfect at init rather than an exact match.
    """

    def __init__(
        self, latent_dim: int = 16, horizons_s: list[float] | None = None, hidden_dim: int = 32,
        identity_shortcut: bool = False, identity_shortcut_noise_std: float = 0.0,
    ):
        super().__init__()
        horizons_s = horizons_s if horizons_s is not None else [0.2, 0.5, 1.0, 2.0, 3.0]
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
                self.net, N_IDENTITY_SHORTCUT_FIELDS, identity_shortcut_noise_std, nonneg_field_index=Z_FIELD_INDEX,
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
        """Query the decoder at an ARBITRARY horizon, not necessarily one of
        the fixed horizons this instance was trained on (e.g. the t=0
        autoencoding sanity check in train_ball_dynamics.py, or any other
        interpolated/extrapolated horizon of interest at inference time).

        Uses the exact same 3-feature construction as ``forward()`` (same
        ``t_norm_max``, same ``log1p``), so results are directly comparable
        to the trained horizon grid's own features -- this is NOT a
        different code path with its own assumptions, just the same
        formula evaluated at a horizon that may not be in ``self.t_norm``.
        """
        batch = latent.shape[0]
        t_norm = horizon_s / self._t_norm_max
        feat = latent.new_tensor([t_norm, t_norm ** 2, math.log1p(horizon_s)])
        return self.net(torch.cat([latent, feat.expand(batch, 3)], dim=-1))


class BallDynamicsAutoencoder(nn.Module):
    """Thin training-only wrapper composing encoder + the shared decoder.

    Not itself a saved artifact -- only ``encoder.state_dict()`` becomes the
    real, permanent output (§6.3, §7).

    ``crossing_head``: a single ``Linear(latent_dim, 3)``, no hidden layer,
    reading directly off the latent (same input the decoder's per-horizon
    heads see, but not routed through the decoder itself) -- predicts the
    ball's out-of-bounds/goal-scored crossing position (``pos_x, pos_y``
    ONLY, not height -- same normalized units as target fields 0:2) and
    ``delta_t`` (seconds until that crossing, ``-1`` sentinel for "never
    happens within the simulated window" -- see ``BallDynamicsDataset.
    crossing_dt``), as one global per-episode prediction rather than one per
    horizon. Always present (cheap: 3*(latent_dim+1) params) so a
    checkpoint's shape doesn't depend on whether this head was ever trained;
    trained/not is entirely controlled by ``physics_pretrain.ball.
    crossing_loss_weight`` (0.0 = no gradient reaches it, same convention as
    ``bce_loss_weight``/``spin_loss_weight``) -- not a constructor flag, so
    there's nothing to keep in sync between "does the head exist" and "does
    it get supervision".

    ``resting_head``: same shape/idea as ``crossing_head`` (``Linear(latent_
    dim, 2)``, no hidden layer, off the latent) but predicts where the ball
    FINALLY comes to rest (``pos_x, pos_y`` only) rather than where it first
    crosses a boundary -- see ``BallDynamicsDataset.compute_resting_
    targets``. Controlled by ``physics_pretrain.ball.resting_loss_weight``,
    same convention as ``crossing_loss_weight``.
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
        self.encoder = BallDynamicsEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, bottleneck_dim=encoder_bottleneck_dim,
            identity_shortcut=identity_shortcut, identity_shortcut_noise_std=identity_shortcut_noise_std,
            concat_all_input_fields=encoder_concat_all_input_fields,
        )
        # `decoder_identity_shortcut` defaults to mirroring `identity_shortcut`
        # (the historical, single-flag behaviour) but can be set independently
        # -- e.g. keep the encoder's concat+hand-init (needed to reuse an
        # encoder checkpoint trained with it on) while giving the decoder a
        # plain random init with no hand-initialized persistence path and no
        # permanent gradient-masking protection on its dedicated units (see
        # `_init_identity_shortcut_decoder`'s docstring for what that
        # protection does), letting the decoder learn from scratch whatever
        # it actually finds useful instead of starting anchored to identity.
        decoder_identity_shortcut = identity_shortcut if decoder_identity_shortcut is None else decoder_identity_shortcut
        self.decoder = BallDynamicsDecoder(
            latent_dim=latent_dim, horizons_s=horizons_s, hidden_dim=decoder_hidden_dim,
            identity_shortcut=decoder_identity_shortcut, identity_shortcut_noise_std=identity_shortcut_noise_std,
        )
        self.crossing_head = nn.Linear(latent_dim, 3)
        self.resting_head = nn.Linear(latent_dim, 2)

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
            horizons_s=cfg["horizons_s"],
            decoder_hidden_dim=cfg.get("decoder_hidden_dim", 32),
            encoder_bottleneck_dim=cfg.get("encoder_bottleneck_dim", 32),
            identity_shortcut=cfg.get("identity_shortcut_enabled", False),
            identity_shortcut_noise_std=cfg.get("identity_shortcut_noise_std", 0.0),
            encoder_concat_all_input_fields=cfg.get("encoder_concat_all_input_fields", False),
            decoder_identity_shortcut=cfg.get("decoder_identity_shortcut_enabled"),
        )

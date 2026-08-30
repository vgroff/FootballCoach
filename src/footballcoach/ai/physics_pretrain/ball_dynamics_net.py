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
    """``input(N_INPUT_FIELDS) -> Linear(hidden) -> LeakyReLU -> Linear(hidden) -> LeakyReLU ->
    Linear(bottleneck) -> LeakyReLU -> [concat raw input[0:9] if identity_shortcut] -> Linear(latent_dim)``.

    ``leaky_relu_negative_slope`` (default ``0.0``, i.e. plain ``ReLU`` --
    ``nn.LeakyReLU(negative_slope=0.0)`` is mathematically identical to
    ``nn.ReLU()``) is the trunk's activation slope for negative
    preactivations. A unit whose preactivation is negative for every example
    in the dataset is a "dead" ReLU unit: it outputs exactly 0 always, and
    (since ``ReLU'(0) == 0`` by convention) gets exactly zero gradient
    forever, with no way to recover on its own -- see ``compute_latent_
    stats``'s dead-dim detection, which this directly addresses. A nonzero
    slope keeps a small but nonzero gradient flowing even when a unit's
    preactivation goes negative, so it can't get permanently stuck. Only
    applied to the TRUNK (this class) -- deliberately NOT threaded into
    ``BallDynamicsDecoder``'s own hidden ReLU, since ``_init_identity_
    shortcut_decoder`` relies on the EXACT identity ``ReLU(x) - ReLU(-x) ==
    x`` (a nonzero slope ``s`` would scale that to ``x*(1+s)`` instead,
    corrupting the hand-initialized reconstruction path).

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
        concat_all_input_fields: bool = False, leaky_relu_negative_slope: float = 0.0,
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

    def has_horizon(self, horizon_s: float) -> bool:
        """Always True -- this decoder is a continuous function of time
        (``forward_at`` above), so any horizon is queryable. Exists purely
        for interface parity with ``BallDynamicsLinearDecoder.has_horizon``
        (which is NOT always True), so callers (e.g. train_ball_dynamics.
        py's adjacent-pair-combo filtering) can check availability uniformly
        regardless of which decoder is in use."""
        return True


# pos_x, pos_y, pos_z, vel_x, vel_y, vel_z -- see BallDynamicsLinearDecoder.
N_LINEAR_DECODER_TARGET_FIELDS = 6

# Absolute tolerance (seconds) for matching a query horizon to one of
# BallDynamicsLinearDecoder's registered heads. horizons_s values and
# adjacent-pair deltas are both plain float subtraction/config literals, so
# exact matches are the common case, but this guards against harmless float
# noise (e.g. 3.5 - 2.0 vs 1.5 differing in the last bit).
_LINEAR_DECODER_HORIZON_ATOL = 1e-6


class BallDynamicsLinearDecoder(nn.Module):
    """Alternative to ``BallDynamicsDecoder``: instead of one shared
    nonlinear function of ``[latent; horizon features]``, this is
    ``n_horizons + 1`` (the configured ``horizons_s``, plus ``0.0``)
    INDEPENDENT ``Linear(latent_dim, 6)`` heads -- one dedicated weight
    matrix per registered horizon, no horizon/time INPUT at all.

    Motivation (see the physics-pretrain overfitting-diagnosis discussion in
    train_ball_dynamics.py's module docs / agent_plans): a shared decoder
    conditioned on a continuous time feature can only combine ``latent`` and
    ``horizon_feats`` through a genuine nonlinearity (a ReLU layer that sees
    both together) -- an AFFINE combination of the two can never represent
    "position = velocity * t" (a product of an episode-specific, latent-
    derived quantity and a horizon-derived one), since affine functions of
    ``[a; b]`` are always ``Ma + Nb + c``, never ``a*b``. Giving each
    horizon its OWN linear map sidesteps this entirely: the horizon-
    dependence lives in WHICH matrix is used, not in a runtime
    multiplication, so there's no "time" input to combine at all. This is
    also strictly more expressive per horizon than a shared narrow hidden
    layer (each head gets full, uncontested linear access to every latent
    dim), at the cost of losing the ability to query horizons that aren't
    exactly one of the registered ones -- no interpolation/extrapolation,
    unlike ``BallDynamicsDecoder.forward_at``.

    Predicts position + velocity ONLY (6 fields: ``pos_x, pos_y, pos_z,
    vel_x, vel_y, vel_z``) -- no spin, no out_of_bounds/goal_scored logits.
    ``forward``/``forward_at`` still return the SAME 11-wide-per-horizon
    shape ``BallDynamicsDecoder`` does (zero-padding the missing spin/logit
    columns) so every downstream loss/breakdown/logging function in
    train_ball_dynamics.py works completely unchanged -- those constant-zero
    columns carry no gradient (they're plain ``torch.zeros``, not connected
    to any parameter), so ``spin_loss_weight``/``bce_loss_weight`` should be
    left at 0.0 when this decoder is active (train_ball_dynamics.py logs a
    note and the logged spin_rmse/oob_bce/goal_bce numbers, if those weights
    are nonzero anyway, are meaningless diagnostics only -- not real
    training signal).

    ``0.0`` is ALWAYS included as a registered horizon (in addition to
    ``horizons_s``) specifically so the t=0 autoencoding sanity check
    (``forward_at(latent, 0.0)`` in train_ball_dynamics.py) has a head to
    query -- it is NOT included in ``forward()``'s returned list (which
    matches the dataset's own per-horizon targets, none of which is at
    t=0).
    """

    def __init__(self, latent_dim: int, horizons_s: list[float] | None = None):
        super().__init__()
        horizons_s = horizons_s if horizons_s is not None else [0.2, 0.5, 1.0, 2.0, 3.0]
        self.n_horizons = len(horizons_s)
        self.latent_dim = latent_dim
        # Index 0 is the extra t=0 head; indices [1:] mirror horizons_s's
        # own order, so forward()'s returned list (which must match dataset
        # target order) is simply self._heads_out(latent)[:, 1:, :].
        self._horizons = (0.0,) + tuple(horizons_s)
        self.net = nn.Linear(latent_dim, len(self._horizons) * N_LINEAR_DECODER_TARGET_FIELDS)

    def _heads_out(self, latent: torch.Tensor) -> torch.Tensor:
        batch = latent.shape[0]
        return self.net(latent).view(batch, len(self._horizons), N_LINEAR_DECODER_TARGET_FIELDS)

    def _pad(self, out6: torch.Tensor) -> torch.Tensor:
        """Zero-pads a ``(batch, 6)`` pos+vel prediction out to the full
        ``(batch, N_TARGET_FIELDS_PER_HORIZON)`` shape every loss/breakdown
        function in train_ball_dynamics.py expects -- see class docstring."""
        pad = out6.new_zeros(out6.shape[0], N_TARGET_FIELDS_PER_HORIZON - N_LINEAR_DECODER_TARGET_FIELDS)
        return torch.cat([out6, pad], dim=-1)

    def forward(self, latent: torch.Tensor) -> list[torch.Tensor]:
        all_out = self._heads_out(latent)
        return [self._pad(all_out[:, i, :]) for i in range(1, len(self._horizons))]

    def has_horizon(self, horizon_s: float) -> bool:
        return any(abs(horizon_s - h) <= _LINEAR_DECODER_HORIZON_ATOL for h in self._horizons)

    def forward_at(self, latent: torch.Tensor, horizon_s: float) -> torch.Tensor:
        for i, h in enumerate(self._horizons):
            if abs(horizon_s - h) <= _LINEAR_DECODER_HORIZON_ATOL:
                return self._pad(self._heads_out(latent)[:, i, :])
        raise ValueError(
            f"BallDynamicsLinearDecoder has no head registered for horizon_s={horizon_s} "
            f"(registered: {self._horizons}) -- callers must check has_horizon() first."
        )


class BallDynamicsAutoencoder(nn.Module):
    """Thin training-only wrapper composing encoder + the shared decoder.

    Not itself a saved artifact -- only ``encoder.state_dict()`` becomes the
    real, permanent output (§6.3, §7).

    ``crossing_head``: a single ``Linear(latent_dim, 4)``, no hidden layer,
    reading directly off the latent (same input the decoder's per-horizon
    heads see, but not routed through the decoder itself) -- predicts the
    ball's out-of-bounds/goal-scored crossing position (``pos_x, pos_y``
    ONLY, not height -- same normalized units as target fields 0:2),
    ``crosses_logit`` (BCE-with-logits classifier: "does this row have a
    real crossing/already-crossed instance to report at all", true whenever
    ``crossing_dt`` isn't the ``-1`` "genuinely never" sentinel), and
    ``delta_t`` (seconds until that crossing, regressed ONLY on rows where
    ``crosses_logit``'s target is true -- no ``-1`` sentinel mixed into the
    regression itself, matching ``PlayerDynamicsAutoencoder.crossing_head``'s
    identical split -- see ``train_ball_dynamics._crossing_head_loss``'s
    docstring for why the combined single-regression version was replaced).
    Always present (cheap: 4*(latent_dim+1) params) so a checkpoint's shape
    doesn't depend on whether this head was ever trained; trained/not is
    entirely controlled by ``physics_pretrain.ball.crossing_pos_loss_
    weight``/``crossing_crosses_loss_weight``/``crossing_dt_loss_weight``
    (all 0.0 = no gradient reaches that term, same convention as
    ``bce_loss_weight``/``spin_loss_weight``) -- not a constructor flag, so
    there's nothing to keep in sync between "does the head exist" and "does
    it get supervision".

    ``resting_head``: same shape/idea as ``crossing_head`` (``Linear(latent_
    dim, 2)``, no hidden layer, off the latent) but predicts where the ball
    FINALLY comes to rest (``pos_x, pos_y`` only) rather than where it first
    crosses a boundary -- see ``BallDynamicsDataset.compute_resting_
    targets``. Controlled by ``physics_pretrain.ball.resting_loss_weight``,
    same 0.0-disables convention as ``crossing_pos_loss_weight``/
    ``crossing_dt_loss_weight``.

    ``position_head``: same shape/idea again (``Linear(latent_dim, 2)``, no
    hidden layer, off the latent) but predicts the ball's CURRENT (t=0 of
    whichever pseudo-start the latent was encoded from) ``pos_x, pos_y`` --
    unlike ``crossing_head``/``resting_head`` there's no mask and no
    separate recorded target: the target is simply the model's own input
    fields 0:2, so this head is a cheap probe of how well the latent
    retains the position it was literally given, always fully defined.
    Controlled by ``physics_pretrain.ball.position_loss_weight``, same
    0.0-disables convention as the other two heads.

    ``event_head``: ``Linear(latent_dim, 2)``, no hidden layer, off the
    latent, NO horizon/time conditioning at all (unlike the shared
    decoder's own per-horizon oob/goal BCE heads, which take a horizon as
    input and predict the INSTANTANEOUS flag at that specific time) --
    predicts, as of t=0, whether the episode EVER goes out of bounds
    (logit 0) / EVER scores a goal (logit 1) across the whole recorded
    window, including "already true at t=0" -- see
    ``BallDynamicsDataset.compute_event_ever_masks``. Entirely separate
    ``nn.Linear`` parameters from the decoder's own oob/goal heads (own
    weights, own gradient, never shares a layer with them) despite
    predicting a related quantity -- this head asks "will/does it happen
    at all," the decoder's asks "is it true AT THIS TIME." Controlled by
    ``physics_pretrain.ball.event_loss_weight``, same 0.0-disables
    convention as the other auxiliary heads. Trained ONLY on t=0 (the
    "main" pass) -- unlike crossing_head/resting_head/position_head, it is
    NOT also applied at every recorded horizon as a pseudo-start, since
    "ever, from here to the end of the recorded window" is only a
    well-defined quantity relative to the ORIGINAL episode's own t=0.
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
        self.encoder = BallDynamicsEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, bottleneck_dim=encoder_bottleneck_dim,
            identity_shortcut=identity_shortcut, identity_shortcut_noise_std=identity_shortcut_noise_std,
            concat_all_input_fields=encoder_concat_all_input_fields,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
        )
        self.linear_decoder = linear_decoder
        if linear_decoder:
            # decoder_hidden_dim/decoder_identity_shortcut/
            # identity_shortcut_noise_std don't apply to this decoder -- see
            # BallDynamicsLinearDecoder's own docstring. Silently ignored
            # here; train_ball_dynamics.py logs a note if the config sets
            # decoder-identity-shortcut-related keys while this is active.
            self.decoder = BallDynamicsLinearDecoder(latent_dim=latent_dim, horizons_s=horizons_s)
        else:
            # `decoder_identity_shortcut` defaults to mirroring
            # `identity_shortcut` (the historical, single-flag behaviour)
            # but can be set independently -- e.g. keep the encoder's
            # concat+hand-init (needed to reuse an encoder checkpoint
            # trained with it on) while giving the decoder a plain random
            # init with no hand-initialized persistence path and no
            # permanent gradient-masking protection on its dedicated units
            # (see `_init_identity_shortcut_decoder`'s docstring for what
            # that protection does), letting the decoder learn from scratch
            # whatever it actually finds useful instead of starting
            # anchored to identity.
            decoder_identity_shortcut = identity_shortcut if decoder_identity_shortcut is None else decoder_identity_shortcut
            self.decoder = BallDynamicsDecoder(
                latent_dim=latent_dim, horizons_s=horizons_s, hidden_dim=decoder_hidden_dim,
                identity_shortcut=decoder_identity_shortcut, identity_shortcut_noise_std=identity_shortcut_noise_std,
            )
        self.crossing_head = nn.Linear(latent_dim, 4)
        self.resting_head = nn.Linear(latent_dim, 2)
        self.position_head = nn.Linear(latent_dim, 2)
        self.event_head = nn.Linear(latent_dim, 2)

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
            linear_decoder=cfg.get("linear_decoder_enabled", False),
            leaky_relu_negative_slope=cfg.get("encoder_leaky_relu_negative_slope", 0.0),
        )

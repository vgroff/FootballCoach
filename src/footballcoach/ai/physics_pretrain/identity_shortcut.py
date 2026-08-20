"""Shared identity-shortcut init helpers for physics-pretrain encoder/decoder
networks.

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

Extracted (pure refactor, no behaviour change) from ``ball_dynamics_net.py``
so ``player_dynamics_net.py`` can reuse the exact same init logic instead of
re-implementing it -- see agent_plans/player_physics_pretrain_plan.md and
agent_plans/ball_physics_pretrain_plan.md §6. Both functions were already
fully general in ``dim``/``noise_std``; the only ball-specific bit was the
hardcoded ``Z_FIELD_INDEX = 2`` module constant inside
``_init_identity_shortcut_decoder``, which is now the ``nonneg_field_index``
parameter below. Callers that have no such provably-non-negative field pass
``nonneg_field_index=None``, which falls back to the full bidirectional
2-unit-per-value init for every one of the ``dim`` fields (no freed spare
unit).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _init_identity_shortcut_linear(linear: nn.Linear, shortcut_start_col: int, dim: int, noise_std: float) -> None:
    """Hand-initializes ``linear``'s first ``dim`` output rows to copy input
    columns ``[shortcut_start_col : shortcut_start_col+dim)`` through
    (weight~1, all other weights/bias 0) -- i.e. those outputs start as a
    near-identity copy of that input slice. ``noise_std`` perturbs the
    identity block with Gaussian noise so the copy isn't exact, both to
    break init symmetry and so training doesn't start already sitting
    perfectly on the target (see the encoder/decoder classes' own docs)."""
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
        block = torch.eye(dim, dtype=linear.weight.dtype)
        if noise_std > 0:
            block = block + torch.randn_like(block) * noise_std
        linear.weight[:dim, shortcut_start_col:shortcut_start_col + dim] = block


def _init_identity_shortcut_decoder(
    decoder_net: nn.Sequential, dim: int, noise_std: float, nonneg_field_index: int | None = None,
) -> None:
    """Hand-initializes ``dim`` of the decoder's hidden units (2 per value,
    using the identity ``ReLU(x) - ReLU(-x) == x``, since a naive identity
    init would get clipped by the ReLU for negative inputs) to pass the
    first ``dim`` latent dims straight through to the first ``dim`` output
    fields, ignoring any extra conditioning features (e.g. horizon
    features) entirely. At init this makes the decoder predict persistence
    ("nothing changed") for every non-identity output, not just the
    zero-conditioning case -- a sensible prior given how informative the
    persistence baseline already is elsewhere in this pipeline. ``noise_std``
    perturbs the ±1 weights so this isn't an exact identity at init.

    ``nonneg_field_index`` (``None`` = no such field): the index (within
    ``[0, dim)``) of a field PROVABLY never negative (e.g. the ball's height
    field, or a player's stamina fraction) -- it gets only ONE dedicated
    unit -- a plain ``ReLU(x) == x`` for ``x >= 0`` is already exact, no
    negative-half unit needed. That field's would-be second unit
    (``free_idx``) is freed up as a genuine SPARE unit instead (same
    random-init/no-mask treatment as the other spare units below) rather
    than wasting it on a ``-1`` weight connection nothing in the real data
    ever needs. This also means that field's DEDICATED reconstruction path
    can itself never contribute a negative value (its one unit is a bare
    ReLU, clipped at 0) -- a soft architectural nudge against negative
    predictions for that field, on top of freeing capacity. Relies on the
    real premise holding (the field's target is never negative) -- feeding
    this path a genuinely negative synthetic target (as opposed to real
    data) produces real, unmaskable reconstruction-loss gradient the single
    ReLU unit can never resolve, which is a test-data-realism issue, not a
    reason to avoid using the freed capacity. When ``nonneg_field_index`` is
    ``None``, every one of the ``dim`` fields gets the full bidirectional
    2-unit treatment and there is no freed spare unit from this mechanism.

    The remaining "spare" hidden units (everything past the ``2*dim``
    dedicated ones, which drive any classification/auxiliary output this
    function doesn't otherwise touch) start zero-initialized here too (the
    whole matrix is zeroed before the dedicated entries are set). Left
    fully zero on BOTH layers, they're a genuine dead fixed point under
    gradient descent (weight=0 and bias=0 gives pre-activation exactly 0
    always, ReLU'(0)=0 by convention, so no gradient ever reaches either
    layer, regardless of loss or learning rate -- verified empirically on
    the ball version: after 50 training steps every example produced the
    IDENTICAL classification logit). So below, spare units get a small
    random re-init on the paths meant to be genuinely free: their read
    access to the non-identity latent dims + any extra conditioning
    features, their read access to the ``dim`` identity latent dims too
    (deliberately NOT zero and NOT frozen -- the classifier legitimately
    may be predictable from raw state, so it should be allowed to use that
    signal; it just starts small/quiet rather than a full-strength random
    connection, so the classifier's initially much larger loss gradient
    doesn't immediately force its way through into the encoder's identity
    weights the way it did before this fix), and their write access to the
    non-reconstruction outputs (second layer, rows ``[dim:)``). Their write
    access to the reconstruction outputs (second layer, rows ``[0:dim)``)
    stays at zero so the exact-reconstruction baseline at epoch 0 is
    preserved -- but is left unmasked/trainable, since the main training
    loop still needs spare units to learn real per-horizon dynamics deltas
    on top of the persistence prior at nonzero horizons.

    There's a SEPARATE, more dangerous leak this function also has to
    close: the DEDICATED units are very much alive (real latent signal
    flows through them), and ``second.weight``'s non-reconstruction rows
    start zero but were originally left unmasked -- gradient descent
    discovered it could cheaply route classification through the dedicated
    units' own (already-informative) activations, which pulled gradient
    back through THEIR weights too (now shared between their "home"
    reconstruction row and classification), corrupting the identity mapping
    directly (see agent_plans/ball_physics_pretrain_plan.md §12.6 for the
    real run where this was observed). Unlike the spare-units-reading-
    identity-latents case above, this one stays a HARD, permanent block:
    it's not about the classifier using state information, it's about a
    second, unrelated task getting to retune the specific hand-crafted
    weights that are supposed to guarantee an exact reconstruction -- there
    is no "soft" version of that which doesn't defeat the point of building
    this shortcut at all. A backward hook permanently zeroes the gradient
    for ``second.weight``'s ``[dim:, :2*dim]`` block (every
    non-reconstruction output row's connections FROM the dedicated units),
    so those entries can never leave zero, for the lifetime of training --
    EXCEPT the freed spare unit (see above), which is excluded from this
    mask since it's a genuine spare unit, not a real dedicated one.
    """
    first, second = decoder_net[0], decoder_net[2]
    hidden_dim = first.out_features
    if hidden_dim < 2 * dim:
        raise ValueError(
            f"decoder hidden_dim ({hidden_dim}) too small for identity-shortcut init of {dim} values "
            f"(needs >= {2 * dim})"
        )
    noise = (lambda: torch.randn(1).item() * noise_std) if noise_std > 0 else (lambda: 0.0)
    free_idx = (2 * nonneg_field_index + 1) if nonneg_field_index is not None else None
    with torch.no_grad():
        first.weight.zero_()
        first.bias.zero_()
        second.weight.zero_()
        second.bias.zero_()
        for i in range(dim):
            a, b = 2 * i, 2 * i + 1
            first.weight[a, i] = 1.0 + noise()
            second.weight[i, a] = 1.0 + noise()
            if i == nonneg_field_index:
                continue  # b (free_idx) is freed below instead of getting the usual negative-half unit
            first.weight[b, i] = -1.0 + noise()
            second.weight[i, b] = -1.0 + noise()

        # The spare units (everything past the 2*dim dedicated ones) were
        # just zeroed above along with everything else -- left that way,
        # BOTH their layers are a genuine fixed point under gradient
        # descent (weight=0, bias=0 -> pre-activation exactly 0 always ->
        # ReLU'(0)=0 by convention -> zero gradient reaches either layer,
        # forever, regardless of loss or learning rate). Re-random-init the
        # paths that are SUPPOSED to be free so they can actually learn:
        # spare units' read access to the non-identity latent dims + any
        # extra conditioning features (first layer), and their write access
        # to the non-reconstruction outputs (second layer, rows [dim:)).
        # Their write access to the reconstruction outputs (second layer,
        # rows [0:dim)) stays at zero so the exact-reconstruction baseline
        # at epoch 0 is preserved -- but is left UNMASKED/trainable, since
        # the main training loop still needs spare units to learn real
        # per-horizon dynamics deltas on top of the persistence prior.
        nn.init.kaiming_uniform_(first.weight[2 * dim:, dim:], a=math.sqrt(5))
        fan_in = first.in_features - dim
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        first.bias[2 * dim:].uniform_(-bound, bound)
        nn.init.kaiming_uniform_(second.weight[dim:, 2 * dim:], a=math.sqrt(5))

        if free_idx is not None:
            # The freed unit (free_idx, inside the [0, 2*dim) dedicated
            # range, not the [2*dim:) spare range above) gets the exact
            # same spare-unit treatment, just applied to its own single
            # row/column via a 1-wide SLICE rather than a bare integer
            # index -- indexing with a slice keeps this a view into
            # `first.weight`/`second.weight` so the in-place inits below
            # actually write back to the real parameter (a bare int index
            # combined with another slice would silently operate on a
            # detached copy instead).
            free = slice(free_idx, free_idx + 1)
            nn.init.kaiming_uniform_(first.weight[free, dim:], a=math.sqrt(5))
            first.bias[free].uniform_(-bound, bound)
            nn.init.kaiming_uniform_(second.weight[dim:, free], a=math.sqrt(5))

        # Spare units' read access to the identity latent dims themselves:
        # deliberately small, not zero, not masked. A full-strength random
        # connection here would immediately let the classifier's much
        # larger initial loss gradient force its way back into the
        # encoder's identity weights (the exact corruption this whole
        # function exists to prevent); a permanent hard mask (as used
        # below for the dedicated units) would instead forbid the
        # classifier from ever using state info that's legitimately
        # predictive, wasting the fact that it's already sitting right
        # there. Small-and-free is the middle ground: quiet at init, but
        # genuinely trainable, so gradient descent can grow this
        # connection gradually and deliberately if it's actually worth the
        # (now much gentler) tradeoff, rather than either being forced
        # into it immediately or never being allowed to use it at all.
        identity_bound = bound * 0.1
        first.weight[2 * dim:, :dim].uniform_(-identity_bound, identity_bound)
        if free_idx is not None:
            first.weight[free_idx:free_idx + 1, :dim].uniform_(-identity_bound, identity_bound)

    # The freed unit (if any) is excluded from the dedicated block: it's a
    # genuine spare unit now, so (like the other spare units) it stays free
    # to write to non-reconstruction outputs, not permanently masked the
    # way the REAL dedicated units are.
    dedicated_mask = torch.ones_like(second.weight)
    dedicated_mask[dim:, :2 * dim] = 0.0
    if free_idx is not None:
        dedicated_mask[dim:, free_idx] = 1.0
    second.weight.register_hook(lambda grad: grad * dedicated_mask)

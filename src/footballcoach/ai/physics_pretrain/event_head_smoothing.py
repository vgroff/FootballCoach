"""Label smoothing + analytic loss floor for physics_pretrain's BCE "event"
heads (out_of_bounds/goal_scored/crosses_logit/event_head).

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

Direct port of ``ai/ppo/bc.py``'s pre-existing label-smoothing/floor
mechanism (``bc_loss_from_tensor``'s internal ``_bce()`` helper and
``compute_bc_loss_floor_components()``'s ``_floor_bce()``) to
``physics_pretrain``'s two offline dynamics-pretraining pipelines
(``train_ball_dynamics.py``/``train_player_dynamics.py``) -- same math, same
conventions, just factored out into a small shared module since both
pipelines need it identically (bc.py's version stays private to that file
since it has only one caller).
"""
from __future__ import annotations

import math

import torch

# Matches F.binary_cross_entropy_with_logits' own epsilon-free convention --
# clamped here purely to keep log() finite at the y'=0/1 extremes (never hit
# in practice once smoothing>0, since y' is then bounded away from 0/1 by
# construction; only matters at smoothing==0, which short-circuits before
# reaching this clamp anyway).
_EPS = 1e-12


def smooth_target(target: torch.Tensor, smoothing: float) -> torch.Tensor:
    """``target*(1-smoothing) + 0.5*smoothing`` -- identical formula to
    ``ai/ppo/bc.py``'s ``_bce()``. ``smoothing<=0.0`` (default) returns
    ``target`` unchanged (no-op, byte-identical behaviour to before this
    module existed)."""
    if smoothing <= 0.0:
        return target
    return target * (1.0 - smoothing) + 0.5 * smoothing


def bce_label_smoothing_floor(target: torch.Tensor, smoothing: float, pos_weight: float = 1.0) -> torch.Tensor:
    """Per-row analytic minimum achievable BCE loss under label smoothing --
    the binary entropy ``H(y') = -y'*ln(y') - (1-y')*ln(1-y')`` of the
    smoothed target, scaled by ``pos_weight`` on hard-positive rows
    (``target > 0.5``, matching ``F.binary_cross_entropy_with_logits``'s own
    ``pos_weight`` semantics -- it multiplies the WHOLE per-sample loss on
    positive rows, not just the ``-log(p)`` term). Identical formula/
    rationale to ``ai/ppo/bc.py``'s ``compute_bc_loss_floor_components``'s
    ``_floor_bce()`` -- see that function's docstring for the full
    derivation and the "symmetric in the hard label, so cheap to compute"
    property (``H(0.5*smoothing) == H(1-0.5*smoothing)``, so this only
    varies row-to-row through ``pos_weight``, never through label balance
    itself when ``pos_weight==1.0``).

    Returns an all-zero tensor (shape/device/dtype matching ``target``) when
    ``smoothing<=0.0`` -- the label-smoothing-disabled case has a true zero
    floor, same convention as ``smooth_target``.
    """
    if smoothing <= 0.0:
        return torch.zeros_like(target)
    y_prime = smooth_target(target, smoothing)
    h = -(
        y_prime * torch.log(y_prime.clamp_min(_EPS))
        + (1.0 - y_prime) * torch.log((1.0 - y_prime).clamp_min(_EPS))
    )
    if pos_weight != 1.0:
        w = torch.where(target > 0.5, torch.full_like(target, pos_weight), torch.ones_like(target))
        h = h * w
    return h


def expected_bce_floor(smoothing: float, pos_weight: float = 1.0) -> float:
    """Closed-form EXPECTED analytic label-smoothing BCE floor, as a plain
    Python float -- the diagnostic-aggregation counterpart to
    ``bce_label_smoothing_floor`` above, usable with no target tensor at
    all. Returns 0.0 when ``smoothing<=0.0``.

    ``H(y')`` is symmetric in the hard label (``H(0.5*smoothing) ==
    H(1-0.5*smoothing)``, see ``bce_label_smoothing_floor``'s docstring),
    so the per-row floor is a single constant ``H0 = H(0.5*smoothing)``
    EXCEPT where ``pos_weight != 1.0`` scales positive rows only -- there,
    the batch MEAN floor is ``H0 * (frac_pos*pos_weight + frac_neg)``. This
    function estimates that mean using the dataset-level ``frac_pos``
    implied by ``pos_weight`` itself: every ``pos_weight`` in this codebase
    is computed as the dataset's inverse-class-frequency weight
    (``n_neg/n_pos``, see ``BallDynamicsDataset.compute_pos_weights()``/
    ``PlayerDynamicsDataset.compute_pos_weights()``), so
    ``frac_pos = 1/(1+pos_weight)`` recovers the SAME dataset-level label
    balance that produced it -- no need to re-derive it from an actual
    batch's target tensor (which would only add per-batch sampling noise
    to an otherwise-stable diagnostic number). Heads with no ``pos_weight``
    at all (``crosses_logit``, ``event_head``) simply call this with the
    default ``pos_weight=1.0``, giving the true constant ``H0`` directly.
    """
    if smoothing <= 0.0:
        return 0.0
    y_prime = 0.5 * smoothing
    eps = 1e-12
    h0 = -(
        y_prime * math.log(max(y_prime, eps))
        + (1.0 - y_prime) * math.log(max(1.0 - y_prime, eps))
    )
    if pos_weight == 1.0:
        return h0
    frac_pos = 1.0 / (1.0 + pos_weight)
    frac_neg = pos_weight / (1.0 + pos_weight)
    return h0 * (frac_pos * pos_weight + frac_neg)

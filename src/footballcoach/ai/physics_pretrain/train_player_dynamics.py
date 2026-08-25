"""CLI: train the player-dynamics encoder from a generated .npz dataset.

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

See agent_plans/player_physics_pretrain_plan.md. Mirrors
``train_ball_dynamics.py``'s structure and QOL layer (per
agent_plans/ball_physics_pretrain_plan.md §12.5's own checklist for this
follow-up): fine-grained per-quantity-group loss/classification reporting,
``pos_weight`` printed and cappable, interleaved same-epoch training passes
(main + adjacent-pair + t0-autoencode) from day one, per-phase best-val
tracking/early stopping, a ``midtrain_latest`` checkpoint on every new
best-val epoch (and a ``midtrain_latest_train`` checkpoint every epoch
unconditionally, independent of val -- a genuinely "latest" snapshot,
useful for overfitting sanity checks and for resuming a killed run without
losing more than the current epoch), and an auto-generated HTML report via
``report.py``.

One deliberate difference from the ball version's loss: ``goal_scored`` is
only physically meaningful when this episode's ``has_possession`` input
field is 1 (a non-possessing player can't "score" by walking into the goal
mouth) -- every place ``goal_scored`` BCE is computed is masked/weighted by
the batch's own ``has_possession`` column (input field 11), and
``compute_pos_weights``/``compute_confusion_counts`` restrict themselves to
``has_possession=1`` rows for that column too, so the reported metric and
the class-balance weighting both stay meaningful. See
``player_episode_gen.py``'s field-layout docs.

Usage::

    uv run python -m footballcoach.ai.physics_pretrain.train_player_dynamics \\
        --dataset physics_pretrain_data/player/ \\
        --output checkpoints/physics_pretrain/player_encoder.pt \\
        --epochs 50 --batch-size 1024

To generate the dataset first, see ``player_dataset.py``'s own ``__main__``
entry point.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from footballcoach.ai.physics_pretrain.player_dataset import GROUPS, PlayerDynamicsDataset
from footballcoach.ai.progress import ProgressReporter
from footballcoach.ai.physics_pretrain.player_dynamics_net import N_IDENTITY_SHORTCUT_FIELDS, PlayerDynamicsAutoencoder
from footballcoach.ai.physics_pretrain.player_episode_gen import N_TARGET_FIELDS_PER_HORIZON

log = logging.getLogger("footballcoach.ai.physics_pretrain.train_player_dynamics")

HAS_POSSESSION_INPUT_IDX = 11

_COMPONENTS = (
    "pos_rmse", "pos_dist", "vel_rmse", "vel_dist",
    "heading_rmse", "heading_dist", "stamina_rmse", "oob_bce", "goal_bce",
)


class LossBreakdown:
    """Per-horizon loss components, each a length-``n_horizons`` list of
    floats. Directly mirrors ``train_ball_dynamics.LossBreakdown`` (RMSE,
    not MSE; ``pos_dist``/``vel_dist`` are the literal mean Euclidean-
    distance metrics, distinct from the per-axis RMSE -- see that class's
    docstring for the Jensen's-inequality discussion, which applies
    identically here for pos/vel).

    ``heading_rmse``/``heading_dist`` are the player-specific addition, not
    present on the ball: ``heading_rmse`` is sqrt of the plain MSE on the
    ``(heading_sin, heading_cos)`` pair (what's actually optimized -- a
    circular quantity encoded as two Cartesian components has no
    discontinuity for MSE to trip over, unlike raw radians). ``heading_dist``
    is the CIRCULARITY-RESPECTING distance metric: the angle between the
    predicted and target heading vectors (``atan2(cross, dot)``, magnitude
    only), in radians -- the heading analogue of ``pos_dist``/``vel_dist``,
    reported purely for human interpretability (never backpropagated), and
    immune to the "0 vs 2*pi" wraparound problem a naive ``|pred_angle -
    target_angle|`` would have.
    """
    __slots__ = tuple(_COMPONENTS)

    def __init__(self):
        for c in _COMPONENTS:
            setattr(self, c, [])


def _heading_angular_dist(pred_sincos: torch.Tensor, target_sincos: torch.Tensor) -> torch.Tensor:
    """Mean angle (radians, always >= 0) between predicted and target
    heading unit vectors, via the standard ``atan2(cross, dot)`` between-
    vectors-angle formula -- correctly wraps around +-pi, unlike a naive
    difference of ``atan2``-recovered raw angles would need to be corrected
    for by hand."""
    pred_sin, pred_cos = pred_sincos[:, 0], pred_sincos[:, 1]
    target_sin, target_cos = target_sincos[:, 0], target_sincos[:, 1]
    dot = pred_sin * target_sin + pred_cos * target_cos
    cross = pred_sin * target_cos - pred_cos * target_sin
    return torch.atan2(cross, dot).abs().mean()


def _goal_bce_masked(
    goal_logit: torch.Tensor, goal_target: torch.Tensor, possession_mask: torch.Tensor,
    pos_weight: torch.Tensor | None, reduction: str = "mean",
) -> torch.Tensor:
    """``goal_scored`` BCE, restricted to ``has_possession=1`` rows -- see
    the module docstring. ``possession_mask`` is the batch's own input
    field 11 (0.0/1.0, held fixed per episode). When ``reduction="mean"``
    and no row in the batch has possession, returns 0 (a batch that happens
    to contain zero possessing rows contributes nothing, rather than NaN
    from a 0/0 division) -- negligible in practice at any real batch size
    given ``possession_start_frac``'s default, and never silently drops
    gradient information a possessing row would have provided."""
    per_row = F.binary_cross_entropy_with_logits(goal_logit, goal_target, pos_weight=pos_weight, reduction="none")
    if reduction == "none":
        return per_row * possession_mask
    denom = possession_mask.sum().clamp(min=1.0)
    return (per_row * possession_mask).sum() / denom


def compute_loss(
    pred_heads: list[torch.Tensor], target: torch.Tensor, input_x: torch.Tensor,
    pos_weight: torch.Tensor, bce_weight: float = 1.0, heading_weight: float = 1.0, stamina_weight: float = 1.0,
) -> tuple[torch.Tensor, LossBreakdown]:
    """Sum over horizons of (continuous MSE + event BCE), directly mirroring
    ``train_ball_dynamics.compute_loss`` -- see its docstring. ``pos_weight``:
    ``(n_horizons, 2)`` tensor from ``PlayerDynamicsDataset.compute_pos_
    weights()``. ``goal_scored``'s BCE is possession-masked -- see
    ``_goal_bce_masked``. ``heading_weight``/``stamina_weight`` (default 1.0,
    prior unweighted behaviour): same 0.0-disables convention as
    ``bce_weight`` -- exists so ``linear_decoder_enabled`` (see
    ``PlayerDynamicsLinearDecoder``, which zero-pads those two columns as a
    constant with no gradient) can exclude them from the trained objective
    instead of baking a meaningless constant offset into every reported
    train/val loss.
    """
    total = target.new_zeros(())
    breakdown = LossBreakdown()
    possession_mask = input_x[:, HAS_POSSESSION_INPUT_IDX]
    for h, head_out in enumerate(pred_heads):
        base = h * N_TARGET_FIELDS_PER_HORIZON
        target_h = target[:, base:base + N_TARGET_FIELDS_PER_HORIZON]
        pos_mse = F.mse_loss(head_out[:, 0:2], target_h[:, 0:2])
        pos_dist = torch.linalg.norm(head_out[:, 0:2] - target_h[:, 0:2], dim=-1).mean()
        vel_mse = F.mse_loss(head_out[:, 2:4], target_h[:, 2:4])
        vel_dist = torch.linalg.norm(head_out[:, 2:4] - target_h[:, 2:4], dim=-1).mean()
        heading_mse = F.mse_loss(head_out[:, 4:6], target_h[:, 4:6])
        heading_dist = _heading_angular_dist(head_out[:, 4:6], target_h[:, 4:6])
        stamina_mse = F.mse_loss(head_out[:, 6], target_h[:, 6])
        oob_bce = F.binary_cross_entropy_with_logits(head_out[:, 7], target_h[:, 7], pos_weight=pos_weight[h, 0])
        goal_bce = _goal_bce_masked(head_out[:, 8], target_h[:, 8], possession_mask, pos_weight[h, 1])
        total = (
            total + pos_mse + vel_mse + heading_weight * heading_mse + stamina_weight * stamina_mse
            + bce_weight * (oob_bce + goal_bce)
        )
        breakdown.pos_rmse.append(float(pos_mse.item()) ** 0.5)
        breakdown.pos_dist.append(float(pos_dist.item()))
        breakdown.vel_rmse.append(float(vel_mse.item()) ** 0.5)
        breakdown.vel_dist.append(float(vel_dist.item()))
        breakdown.heading_rmse.append(float(heading_mse.item()) ** 0.5)
        breakdown.heading_dist.append(float(heading_dist.item()))
        breakdown.stamina_rmse.append(float(stamina_mse.item()) ** 0.5)
        breakdown.oob_bce.append(float(oob_bce.item()))
        breakdown.goal_bce.append(float(goal_bce.item()))
    return total, breakdown


def compute_per_episode_loss(
    pred_heads: list[torch.Tensor], target: torch.Tensor, input_x: torch.Tensor,
    pos_weight: torch.Tensor, bce_weight: float = 1.0, heading_weight: float = 1.0, stamina_weight: float = 1.0,
) -> torch.Tensor:
    """Same components as ``compute_loss``'s ``total``, NOT reduced across
    the batch -- see ``train_ball_dynamics.compute_per_episode_loss``."""
    batch = target.shape[0]
    total = target.new_zeros(batch)
    possession_mask = input_x[:, HAS_POSSESSION_INPUT_IDX]
    for h, head_out in enumerate(pred_heads):
        base = h * N_TARGET_FIELDS_PER_HORIZON
        target_h = target[:, base:base + N_TARGET_FIELDS_PER_HORIZON]
        pos_sq = (head_out[:, 0:2] - target_h[:, 0:2]).pow(2).mean(dim=1)
        vel_sq = (head_out[:, 2:4] - target_h[:, 2:4]).pow(2).mean(dim=1)
        heading_sq = (head_out[:, 4:6] - target_h[:, 4:6]).pow(2).mean(dim=1)
        stamina_sq = (head_out[:, 6] - target_h[:, 6]).pow(2)
        oob_bce = F.binary_cross_entropy_with_logits(
            head_out[:, 7], target_h[:, 7], pos_weight=pos_weight[h, 0], reduction="none",
        )
        goal_bce = _goal_bce_masked(head_out[:, 8], target_h[:, 8], possession_mask, pos_weight[h, 1], reduction="none")
        total = (
            total + pos_sq + vel_sq + heading_weight * heading_sq + stamina_weight * stamina_sq
            + bce_weight * (oob_bce + goal_bce)
        )
    return total


def compute_confusion_counts(
    pred_heads: list[torch.Tensor], target: torch.Tensor, input_x: torch.Tensor,
) -> dict[str, list[tuple[int, int, int, int]]]:
    """Per-horizon ``(tp, fp, fn, tn)`` counts for ``out_of_bounds``/
    ``goal_scored`` -- see ``train_ball_dynamics.compute_confusion_counts``.
    ``goal_scored`` counts are restricted to ``has_possession=1`` rows (same
    masking as the loss -- see the module docstring)."""
    out: dict[str, list[tuple[int, int, int, int]]] = {"oob": [], "goal": []}
    possession_mask = input_x[:, HAS_POSSESSION_INPUT_IDX] > 0.5
    for h, head_out in enumerate(pred_heads):
        base = h * N_TARGET_FIELDS_PER_HORIZON
        target_h = target[:, base:base + N_TARGET_FIELDS_PER_HORIZON]
        pred_oob = head_out[:, 7] > 0
        actual_oob = target_h[:, 7] > 0.5
        out["oob"].append((
            int((pred_oob & actual_oob).sum().item()), int((pred_oob & ~actual_oob).sum().item()),
            int((~pred_oob & actual_oob).sum().item()), int((~pred_oob & ~actual_oob).sum().item()),
        ))
        pred_goal = (head_out[:, 8] > 0) & possession_mask
        actual_goal = (target_h[:, 8] > 0.5) & possession_mask
        eligible = possession_mask
        out["goal"].append((
            int((pred_goal & actual_goal).sum().item()), int((pred_goal & ~actual_goal & eligible).sum().item()),
            int((~pred_goal & actual_goal).sum().item()), int((~pred_goal & ~actual_goal & eligible).sum().item()),
        ))
    return out


def _single_target_loss_with_breakdown(
    pred: torch.Tensor, target: torch.Tensor, input_x: torch.Tensor,
    pos_weight_row: torch.Tensor | None = None, bce_weight: float = 1.0,
    heading_weight: float = 1.0, stamina_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Same components as one horizon's worth of ``compute_loss``, for a
    single ``(pred, target)`` pair -- shared by the adjacent-pair and
    autoencode-pretrain passes, see ``train_ball_dynamics._single_target_
    loss_with_breakdown``. ``heading_weight``/``stamina_weight``: see
    ``compute_loss``'s docstring."""
    possession_mask = input_x[:, HAS_POSSESSION_INPUT_IDX]
    pos_mse = F.mse_loss(pred[:, 0:2], target[:, 0:2])
    pos_dist = torch.linalg.norm(pred[:, 0:2] - target[:, 0:2], dim=-1).mean()
    vel_mse = F.mse_loss(pred[:, 2:4], target[:, 2:4])
    vel_dist = torch.linalg.norm(pred[:, 2:4] - target[:, 2:4], dim=-1).mean()
    heading_mse = F.mse_loss(pred[:, 4:6], target[:, 4:6])
    heading_dist = _heading_angular_dist(pred[:, 4:6], target[:, 4:6])
    stamina_mse = F.mse_loss(pred[:, 6], target[:, 6])
    oob_w = pos_weight_row[0] if pos_weight_row is not None else None
    goal_w = pos_weight_row[1] if pos_weight_row is not None else None
    oob_bce = F.binary_cross_entropy_with_logits(pred[:, 7], target[:, 7], pos_weight=oob_w)
    goal_bce = _goal_bce_masked(pred[:, 8], target[:, 8], possession_mask, goal_w)
    total = (
        pos_mse + vel_mse + heading_weight * heading_mse + stamina_weight * stamina_mse
        + bce_weight * (oob_bce + goal_bce)
    )
    breakdown = {
        "pos_rmse": float(pos_mse.item()) ** 0.5, "pos_dist": float(pos_dist.item()),
        "vel_rmse": float(vel_mse.item()) ** 0.5, "vel_dist": float(vel_dist.item()),
        "heading_rmse": float(heading_mse.item()) ** 0.5, "heading_dist": float(heading_dist.item()),
        "stamina_rmse": float(stamina_mse.item()) ** 0.5,
        "oob_bce": float(oob_bce.item()), "goal_bce": float(goal_bce.item()),
    }
    return total, breakdown


def _single_target_per_episode_loss(
    pred: torch.Tensor, target: torch.Tensor, input_x: torch.Tensor,
    pos_weight_row: torch.Tensor | None = None, bce_weight: float = 1.0,
    heading_weight: float = 1.0, stamina_weight: float = 1.0,
) -> torch.Tensor:
    possession_mask = input_x[:, HAS_POSSESSION_INPUT_IDX]
    pos_sq = (pred[:, 0:2] - target[:, 0:2]).pow(2).mean(dim=1)
    vel_sq = (pred[:, 2:4] - target[:, 2:4]).pow(2).mean(dim=1)
    heading_sq = (pred[:, 4:6] - target[:, 4:6]).pow(2).mean(dim=1)
    stamina_sq = (pred[:, 6] - target[:, 6]).pow(2)
    oob_w = pos_weight_row[0] if pos_weight_row is not None else None
    goal_w = pos_weight_row[1] if pos_weight_row is not None else None
    oob_bce = F.binary_cross_entropy_with_logits(pred[:, 7], target[:, 7], pos_weight=oob_w, reduction="none")
    goal_bce = _goal_bce_masked(pred[:, 8], target[:, 8], possession_mask, goal_w, reduction="none")
    return (
        pos_sq + vel_sq + heading_weight * heading_sq + stamina_weight * stamina_sq
        + bce_weight * (oob_bce + goal_bce)
    )


def _require_horizon_index(horizons_s, value: float, purpose: str) -> int:
    """Index of the horizon EXACTLY equal to ``value`` in ``horizons_s``, or
    a clear startup ``ValueError`` naming what needed it. Auxiliary heads
    that are pinned to one specific wall-clock horizon (see
    ``goal_dist_delta_head``/the short-horizon probes) must never guess a
    position in ``horizons_s`` -- that list is config-driven and reorderable,
    so a hardcoded index would silently start supervising the wrong horizon
    the moment someone edited it."""
    for i, h in enumerate(horizons_s):
        if float(h) == value:
            return i
    raise ValueError(
        f"physics_pretrain.player.horizons_s must contain {value} exactly -- {purpose} is defined at "
        f"that horizon and there is no sensible substitute. Got horizons_s={[float(h) for h in horizons_s]}. "
        "Either add that horizon or set the corresponding loss weight to 0.0 to disable the head."
    )


def _goal_mouth_distance_m(
    pos_x_m: np.ndarray, pos_y_m: np.ndarray, goal_line_x_m: np.ndarray, half_goal_width_m: np.ndarray,
) -> np.ndarray:
    """Vectorized 2D distance (metres) from a player at ``(pos_x_m,
    pos_y_m)`` to the CLOSEST POINT of a goal mouth -- the line segment
    ``x = goal_line_x_m``, ``y in [-half_goal_width_m, +half_goal_width_m]``
    (see ``Pitch.is_goal``: the left goal sits at ``x = -half_length``, the
    right at ``x = +half_length``, both spanning ``goal_width_m`` centred on
    ``y = 0``).

    Standard point-to-segment distance, specialized to an axis-aligned
    segment: the x-offset always counts; the y-offset only counts by however
    far the player is OUTSIDE the mouth's y-span (0 when level with the
    mouth, in which case the distance is just ``|pos_x - goal_line_x|``).
    Everything is elementwise, so each row may carry its own randomized
    pitch/goal dims.
    """
    dx = np.abs(pos_x_m - goal_line_x_m)
    dy = np.maximum(np.abs(pos_y_m) - half_goal_width_m, 0.0)
    return np.hypot(dx, dy)


def _goal_dist_delta_targets(
    ds: PlayerDynamicsDataset, params, horizon_idx_3s: int, pitch_half_diag_m: float,
) -> np.ndarray:
    """``(N, 2)`` float32 training targets for ``goal_dist_delta_head``:
    per goal, ``dist_to_closest_point_of_goal(pos_at_t=3.0s) -
    dist_to_closest_point_of_goal(pos_at_t=0)``. Column 0 = LEFT goal
    (``x = -half_length``), column 1 = RIGHT goal (``x = +half_length``).
    Negative = the player got closer to that goal over those 3 seconds.

    Computed at TRAIN time from data the dataset already carries (t=0
    position from ``ds.inputs`` fields 0/1, t=3.0s position from
    ``ds.targets``' horizon-3.0s block, per-episode pitch/goal dims from
    input fields 17/19) -- nothing new is recorded at dataset-generation
    time, so this needs no schema change and works on any player dataset.

    The distances themselves are computed in REAL METRES (the goal-mouth
    geometry isn't scale-invariant, and under
    ``normalize_kinematics_by_base_pitch=false`` pos_x and pos_y don't even
    share a divisor, so a "distance" in raw normalized units would be a
    meaningless anisotropic mix), then the resulting delta is divided by
    ``pitch_half_diag_m`` to land on the same normalized scale as every
    other quantity the network regresses. It stays a RAW (signed, absolute)
    delta -- deliberately NOT expressed as a fraction of the initial
    distance, which would blow up for a player who starts on the goal line.
    """
    inputs = ds.inputs
    n = len(inputs)
    # Actual pitch/goal geometry -- fields 17/19 are always a plain
    # ratio-to-base regardless of normalize_kinematics_by_base_pitch (see
    # player_episode_gen._encode_input).
    half_length_m = inputs[:, 17].astype(np.float64) * params.base_pitch_length_m / 2.0
    half_goal_width_m = inputs[:, 19].astype(np.float64) * params.base_goal_width_m / 2.0
    # Divisors position was ENCODED with -- same branch as
    # player_episode_gen._kinematics_divisors / _kinematics_denorm_scales
    # above, vectorized over rows.
    if params.normalize_kinematics_by_base_pitch:
        half_diag = math.hypot(params.base_pitch_length_m / 2, params.base_pitch_width_m / 2)
        div_x = np.full(n, half_diag, dtype=np.float64)
        div_y = np.full(n, half_diag, dtype=np.float64)
    else:
        div_x = half_length_m
        div_y = inputs[:, 18].astype(np.float64) * params.base_pitch_width_m / 2.0

    base = horizon_idx_3s * N_TARGET_FIELDS_PER_HORIZON
    pos0_x, pos0_y = inputs[:, 0] * div_x, inputs[:, 1] * div_y
    pos3_x, pos3_y = ds.targets[:, base] * div_x, ds.targets[:, base + 1] * div_y

    out = np.zeros((n, 2), dtype=np.float32)
    for col, side in enumerate((-1.0, 1.0)):  # 0 = left goal, 1 = right goal
        goal_line_x_m = side * half_length_m
        d0 = _goal_mouth_distance_m(pos0_x, pos0_y, goal_line_x_m, half_goal_width_m)
        d3 = _goal_mouth_distance_m(pos3_x, pos3_y, goal_line_x_m, half_goal_width_m)
        out[:, col] = (d3 - d0) / pitch_half_diag_m
    return out


def _crossing_head_loss(
    model: PlayerDynamicsAutoencoder, latent: torch.Tensor, pos_all: np.ndarray, dt_all: np.ndarray,
    mask_all: np.ndarray, row_idx: np.ndarray, device: str,
    pos_loss_weight: float = 1.0, crosses_loss_weight: float = 1.0, dt_loss_weight: float = 1.0,
    dt_norm_s: float = 1.0, trust_negatives: bool = True,
) -> tuple[torch.Tensor, float, float, float, float, float, float]:
    """``model.crossing_head``'s loss for one batch -- THREE separate terms:

    1. Position MSE (x/y, MASKED to rows where ``mask_all`` is True -- an
       episode with no crossing AHEAD of whichever pseudo-start these
       targets were built relative to has no meaningful crossing position
       to regress toward).
    2. ``crosses_logit`` BCE-with-logits: "does this row have a real
       crossing/already-crossed instance at all" -- target is ``dt_all !=
       -1.0`` (true for BOTH a genuine future crossing AND an already-
       crossed/already-out-of-bounds row, false only for the genuinely-
       never-happens case). UNMASKED -- every row has a well-defined yes/no
       answer to this, unlike position.
    3. ``dt`` regression, MASKED to the SAME ``dt_all != -1.0`` rows the
       classifier target uses (NOT ``mask_all`` -- deliberately wider,
       includes already-crossed rows too) -- trained on the real value only
       (0 for already-crossed, a real positive delta_t for a genuine future
       crossing), with the -1 sentinel excluded from the regression
       entirely rather than mixed in.

    This 3-way split (rather than one combined delta_t regression against
    the -1 sentinel directly) exists because the single-regression version
    forced MSE to blend two qualitatively different signals -- "will it
    cross at all" and "when, given it crosses" -- into one number. For any
    row the network was even slightly unsure about, the MSE-optimal single
    prediction was a weighted average of "-1" and "some real time", landing
    confidently on neither and inflating error on both sides at once
    (verified empirically: a real trained checkpoint predicted a mean of
    -0.227 on rows whose true target was the constant -1 -- nowhere near
    confidently -1, despite that being the easiest possible sub-case).
    Splitting removes the blend: the classifier only ever answers yes/no,
    and the regression only ever sees real, same-scale values with no -1
    gap to average against.

    ``dt_norm_s`` (default 1.0 -- no-op) divides ``dt_all`` by this before
    the network's raw ``dt`` output is compared to it (on the now-masked,
    sentinel-free regression), so the MODEL trains against a roughly-O(1)
    target instead of raw seconds -- same convention as every OTHER
    regression target in this pipeline. Callers pass ``max(horizons_s)``
    (see ``train()``'s ``crossing_dt_norm_s``). This matters even under
    Adam: for a parameter used by ONLY this one loss term (crossing_head's
    dt-output weight row, not shared with any other head), a constant
    loss-WEIGHT multiplier gets almost entirely cancelled by Adam's own
    per-parameter normalization (``m/sqrt(v) ~= (w*g)/sqrt((w*g)**2) ~=
    g/|g|``, independent of w) -- so `dt_loss_weight` alone can't fix the
    raw-seconds scale mismatch the way `pos_loss_weight` can for position
    (which stays properly normalized). What actually gates convergence
    speed is the ABSOLUTE MAGNITUDE the weight vector has to reach from its
    small random init (Adam's step size is roughly a fixed ~lr per
    iteration, not distance-proportional) -- normalizing the TARGET itself
    directly shrinks that required distance instead of just rescaling a
    gradient Adam mostly renormalizes away. ``dt_mae_mean`` (see Returns
    below) is converted back to real seconds for logging regardless of this
    normalization.

    ``pos_loss_weight``/``crosses_loss_weight``/``dt_loss_weight`` (all
    default 1.0, i.e. plain sum) scale the three terms BEFORE they're
    combined into the returned ``loss``. Callers apply these as
    `crossing_pos_loss_weight`/`crossing_crosses_loss_weight`/
    `crossing_dt_loss_weight` from config; there's no separate OUTER weight
    multiplying the whole head's loss (the weighting happens in here, once).

    ``trust_negatives`` (default True -- unchanged behaviour): whether a
    NEGATIVE ``crosses_target`` (i.e. ``dt_all == -1.0``, "never crosses")
    is trustworthy enough to train ``crosses_logit`` against. player_
    episode_gen.generate_episode only simulates each episode for
    ``max(horizons_s)`` seconds -- if no out-of-bounds/goal event happens
    in that fixed window, ``crossing_time_s`` (and therefore ``dt_all``)
    gets the ``-1.0``/inf "never" sentinel regardless of whether a real
    event would have happened just PAST the window. That's a RIGHT-
    CENSORED observation, not a verified "never": at t=0 it's a complete,
    uncensored fact (the full ``max(horizons_s)``-second window was
    observed), but at a horizon-pass pseudo-start ``h`` the REMAINING
    observed window is only ``max(horizons_s) - horizons_s[h]`` seconds --
    shorter, and shrinking as ``h`` grows -- while ``crossing_head`` has no
    input telling it which pseudo-start it's being asked from (it only
    sees the latent, i.e. current position/velocity/etc.). A player whose
    true exit happens just after the recording ends gets labelled
    "never" at EVERY pseudo-start, contradicting a possibly-correct
    physical extrapolation purely because the recording stopped first.
    POSITIVE labels (``future_valid``/``already_there_h``, see
    ``_build_horizon_bundle``) stay hard, unconditionally verified facts
    at any horizon and are never affected by this. Set False (only ever
    passed by horizon-pass call sites; t=0 stays at the default True) to
    exclude negative rows from the ``crosses_logit`` BCE entirely rather
    than training against a possibly-wrong censored label -- positive rows
    there are untouched. Does not affect ``pos_loss``/``dt_loss``: both
    are already masked to positive-only rows regardless of this flag (see
    above), so they were never exposed to this censoring issue.

    Near-verbatim port of ``train_ball_dynamics._crossing_head_loss``'s
    pos-masking convention, minus its height-exclusion note (a player has
    no height axis to exclude) -- ball hasn't been split into classifier+
    regression yet (see this function's own history for why player needed
    it first). ``pos_all``/``dt_all``/``mask_all`` are plain arrays (not
    tied to a specific dataset attribute) so this same function serves BOTH
    the main task's t=0 crossing target (``ds.crossing_pos``/
    ``crossing_dt``/``crossing_mask``, indexed by ``row_idx`` = original
    dataset row indices) AND the per-horizon pseudo-start generalization
    (see ``_build_horizon_bundle`` -- ``crossing_time - horizons_s[h]``,
    reindexed to whichever row order the caller's arrays use). Callers must
    ensure ``pos_all``/``dt_all``/``mask_all`` and ``row_idx`` share the
    same indexing convention; this function doesn't care which one it is.

    Returns ``(loss, pos_dist_mean, dt_mae_mean, pos_loss_val,
    crosses_loss_val, dt_loss_val, crosses_acc)`` -- ``pos_dist_mean``/
    ``dt_mae_mean``/``crosses_acc`` are plain floats (not backpropagated)
    for per-epoch reporting, UNWEIGHTED: mean Euclidean crossing-position
    error over the rows that actually crossed (0.0 if none did, matching
    the masked loss's own 0-numerator/1-denominator convention), mean
    absolute delta_t error over the ``dt_all != -1.0`` rows (matching what
    the regression itself now trains against), and classification accuracy
    of ``crosses_logit`` against its target. ``pos_loss_val``/
    ``crosses_loss_val``/``dt_loss_val`` are the three WEIGHTED sub-terms
    that sum to ``loss``, letting callers report how much of
    ``crossing_head``'s own loss comes from each part -- see
    ``_log_aux_diagnostics``'s per-head split line.
    """
    crossing_pred = model.crossing_head(latent)
    c_pos = torch.from_numpy(pos_all[row_idx]).to(device)
    c_dt_raw = torch.from_numpy(dt_all[row_idx]).to(device)
    c_mask = torch.from_numpy(mask_all[row_idx]).to(device)
    mask_f = c_mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    pos_err = crossing_pred[:, 0:2] - c_pos
    pos_loss = (pos_err.pow(2).sum(dim=-1) * mask_f).sum() / denom

    crosses_target = (c_dt_raw != -1.0).float()
    crosses_logit = crossing_pred[:, 2]
    if trust_negatives:
        crosses_loss = F.binary_cross_entropy_with_logits(crosses_logit, crosses_target)
    else:
        # See this function's own trust_negatives docstring -- a negative
        # label here is right-censored, not verified, so it's excluded
        # from the BCE entirely rather than trained against. Positive
        # rows (crosses_target == 1) are untouched.
        crosses_denom = crosses_target.sum().clamp_min(1.0)
        crosses_loss = (
            F.binary_cross_entropy_with_logits(crosses_logit, crosses_target, reduction="none") * crosses_target
        ).sum() / crosses_denom

    dt_mask_f = crosses_target
    dt_denom = dt_mask_f.sum().clamp_min(1.0)
    dt_pred = crossing_pred[:, 3]
    c_dt_norm = c_dt_raw / dt_norm_s
    dt_loss = ((dt_pred - c_dt_norm).pow(2) * dt_mask_f).sum() / dt_denom

    weighted_pos_loss = pos_loss_weight * pos_loss
    weighted_crosses_loss = crosses_loss_weight * crosses_loss
    weighted_dt_loss = dt_loss_weight * dt_loss
    loss = weighted_pos_loss + weighted_crosses_loss + weighted_dt_loss
    with torch.no_grad():
        pos_dist_mean = float((torch.linalg.norm(pos_err, dim=-1) * mask_f).sum().item() / float(denom.item()))
        # Converted back to real seconds (undoing dt_norm_s), masked to the
        # same dt_all != -1.0 rows the regression itself trains against.
        dt_mae_mean = float(((dt_pred - c_dt_norm).abs() * dt_mask_f).sum().item() / float(dt_denom.item())) * dt_norm_s
        # Masked to match whatever crosses_loss actually trained on above --
        # unmasked (all rows) when trust_negatives=True, positive-rows-only
        # when False, so a caller watching this metric sees accuracy over
        # the SAME population the gradient came from rather than a figure
        # diluted by rows that were deliberately excluded from training
        # (see trust_negatives's own docstring for why those are excluded).
        crosses_acc_mask = crosses_target if not trust_negatives else torch.ones_like(crosses_target)
        crosses_acc_denom = crosses_acc_mask.sum().clamp_min(1.0)
        crosses_correct = ((crosses_logit > 0) == (crosses_target > 0.5)).float()
        crosses_acc = float((crosses_correct * crosses_acc_mask).sum().item() / float(crosses_acc_denom.item()))
        pos_loss_val = float(weighted_pos_loss.item())
        crosses_loss_val = float(weighted_crosses_loss.item())
        dt_loss_val = float(weighted_dt_loss.item())
    return loss, pos_dist_mean, dt_mae_mean, pos_loss_val, crosses_loss_val, dt_loss_val, crosses_acc


def _migrate_crossing_head_state_dict(state_dict: dict, model: PlayerDynamicsAutoencoder) -> dict:
    """Backward-compat shim for checkpoints saved when ``crossing_head`` had
    3 outputs (``pos_x, pos_y, delta_t``) -- before ``delta_t`` was split
    into a separate ``crosses_logit`` classifier + ``dt`` regression (see
    ``_crossing_head_loss``'s docstring for why). Current layout is
    ``pos_x, pos_y, crosses_logit, delta_t``. Preserves the still-valid
    ``pos_x``/``pos_y`` rows from the old checkpoint; the two NEW rows
    (``crosses_logit``/``delta_t``) are left at the CURRENT model's own
    fresh init -- there's no sensible way to migrate the OLD single
    ``delta_t`` value into either new output alone, since it used to blend
    both signals into one number. Returns a NEW dict (doesn't mutate
    ``state_dict``); a no-op if shapes already match (nothing to migrate)
    or ``crossing_head.weight`` isn't present at all (checkpoint predates
    the head entirely -- an ordinary ``load_state_dict`` error surfaces
    normally in that case, same as any other genuinely-incompatible
    checkpoint). Direct port of ``train_ball_dynamics._migrate_crossing_
    head_state_dict``'s identical shim for ball's OWN earlier 4->3 output
    shrink (a different migration, same pattern).
    """
    key_w, key_b = "crossing_head.weight", "crossing_head.bias"
    if key_w not in state_dict:
        return state_dict
    old_out, new_out = state_dict[key_w].shape[0], model.crossing_head.weight.shape[0]
    if old_out == 3 and new_out == 4:
        state_dict = dict(state_dict)
        fresh_w = model.crossing_head.weight.detach().clone()
        fresh_b = model.crossing_head.bias.detach().clone()
        fresh_w[0:2] = state_dict[key_w][0:2]
        fresh_b[0:2] = state_dict[key_b][0:2]
        state_dict[key_w] = fresh_w
        state_dict[key_b] = fresh_b
        log.info(
            "Migrated checkpoint's crossing_head from 3 outputs (pos_x, pos_y, delta_t) to 4 (pos_x, pos_y, "
            "crosses_logit, delta_t) -- kept pos_x/pos_y, reset crosses_logit/delta_t to fresh init."
        )
    return state_dict


def _goal_dist_delta_head_loss(
    model: PlayerDynamicsAutoencoder, latent: torch.Tensor, targets_all: np.ndarray,
    row_idx: np.ndarray, device: str,
) -> tuple[torch.Tensor, float, float]:
    """``model.goal_dist_delta_head``'s loss for one batch -- plain UNMASKED
    MSE over both goals. Unlike the crossing head there's nothing to mask:
    every row has a well-defined position at t=0 and at t=3.0s, so every row
    has a well-defined distance delta to each goal.

    MAIN-ROWS-ONLY by design (never called from the "horizon" pass): this
    head is about one FIXED wall-clock horizon (t=3.0s from the episode's
    real start), and a mid-trajectory pseudo-start's own "3 seconds from
    here" is a different quantity entirely -- supervising both against the
    same head would train it on two incompatible targets.

    Returns ``(loss, mae_left, mae_right)`` -- the two diagnostics are the
    per-goal mean absolute error in the same normalized units the targets
    use (callers scale by ``pitch_half_diag_m`` to report metres).
    """
    pred = model.goal_dist_delta_head(latent)
    tgt = torch.from_numpy(targets_all[row_idx]).to(device)
    loss = F.mse_loss(pred, tgt)
    with torch.no_grad():
        mae = (pred - tgt).abs().mean(dim=0)
    return loss, float(mae[0].item()), float(mae[1].item())


def _short_horizon_probe_loss(
    model: PlayerDynamicsAutoencoder, latent: torch.Tensor,
    targets_0_2s: np.ndarray, targets_1_0s: np.ndarray, row_idx: np.ndarray, device: str,
) -> tuple[torch.Tensor, float, float]:
    """The two short-horizon PROBE heads' combined loss for one batch --
    plain UNMASKED MSE of each head's ``(pos_x, pos_y, vel_x, vel_y)``
    prediction against the recorded target block at exactly 0.2s / 1.0s,
    SUMMED across the two heads (matching how every other multi-component
    loss in this file combines its parts -- ``compute_loss`` sums pos/vel/
    heading/stamina rather than averaging them, so one shared config weight
    scales the pair as a unit).

    MAIN-ROWS-ONLY, same reasoning as ``_goal_dist_delta_head_loss``: each
    head is pinned to one literal horizon value, which a mid-trajectory
    pseudo-start doesn't share.

    Returns ``(loss, rmse_0_2s, rmse_1_0s)`` -- per-head RMSE in normalized
    units, for reporting only.
    """
    total = latent.new_zeros(())
    rmses: list[float] = []
    for head, targets_all in (
        (model.short_horizon_head_0_2s, targets_0_2s),
        (model.short_horizon_head_1_0s, targets_1_0s),
    ):
        pred = head(latent)
        tgt = torch.from_numpy(targets_all[row_idx]).to(device)
        mse = F.mse_loss(pred, tgt)
        total = total + mse
        rmses.append(float(mse.item()) ** 0.5)
    return total, rmses[0], rmses[1]


_AUX_METRIC_KEYS = (
    "crossing_loss", "crossing_pos_dist", "crossing_dt_mae", "crossing_crosses_acc",
    "crossing_pos_loss", "crossing_crosses_loss", "crossing_dt_loss",
    "goal_dist_delta_loss", "goal_dist_delta_mae_left", "goal_dist_delta_mae_right",
    "short_horizon_probe_loss", "short_horizon_rmse_0_2s", "short_horizon_rmse_1_0s",
)


class _AuxAccumulator:
    """Per-epoch accumulator for the three auxiliary latent heads'
    diagnostics, shared by the train and eval passes so the two can't drift
    apart in what they measure or how they average it.

    Every metric is a plain per-batch mean averaged over the epoch's batches
    (never merged into ``train_loss``/``val_loss``, same convention as
    pair/t0 -- see the ``backprop_loss`` comment in
    ``_run_interleaved_train_epoch``). ``summary()`` returns NaN for any head
    that contributed no batches (i.e. is disabled), which the log lines and
    ``.history.npz`` both carry through unchanged rather than pretending a
    disabled head scored 0.

    Note the crossing accumulators deliberately pool the main pass's t=0
    usage AND the horizon pass's per-pseudo-start usage into one number
    (matching train_ball_dynamics.py) -- the two are the same head learning
    the same task from different start points.
    """

    def __init__(self):
        self._values: dict[str, list[float]] = {k: [] for k in _AUX_METRIC_KEYS}

    def add_crossing(
        self, loss: torch.Tensor, pos_dist: float, dt_mae: float,
        pos_loss: float, crosses_loss: float, dt_loss: float, crosses_acc: float,
    ) -> None:
        self._values["crossing_loss"].append(float(loss.item()))
        self._values["crossing_pos_dist"].append(pos_dist)
        self._values["crossing_dt_mae"].append(dt_mae)
        self._values["crossing_pos_loss"].append(pos_loss)
        self._values["crossing_crosses_loss"].append(crosses_loss)
        self._values["crossing_dt_loss"].append(dt_loss)
        self._values["crossing_crosses_acc"].append(crosses_acc)

    def add_goal_dist(self, loss: torch.Tensor, mae_left: float, mae_right: float) -> None:
        self._values["goal_dist_delta_loss"].append(float(loss.item()))
        self._values["goal_dist_delta_mae_left"].append(mae_left)
        self._values["goal_dist_delta_mae_right"].append(mae_right)

    def add_probe(self, loss: torch.Tensor, rmse_0_2s: float, rmse_1_0s: float) -> None:
        self._values["short_horizon_probe_loss"].append(float(loss.item()))
        self._values["short_horizon_rmse_0_2s"].append(rmse_0_2s)
        self._values["short_horizon_rmse_1_0s"].append(rmse_1_0s)

    def summary(self) -> dict[str, float]:
        return {
            k: (float(np.mean(v)) if v else float("nan"))
            for k, v in self._values.items()
        }


def _summary_stats(values: list[float]) -> dict[str, float]:
    """``{mean, std, min, max}`` of ``values`` -- ``nan`` for all four if
    empty, ``std=0.0`` (not ``nan``) for a single value (a degenerate but
    well-defined population of one, matching ``np.std``'s own convention).
    Directly mirrors ``train_ball_dynamics._summary_stats`` -- shared shape
    for the per-epoch gradient-norm and train-loss-delta diagnostics below.
    """
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}


def _format_backprop_contrib(contrib: dict[str, float], total: float) -> str:
    """Directly mirrors ``train_ball_dynamics._format_backprop_contrib`` --
    see its docstring."""
    items = [(name, val) for name, val in contrib.items() if not math.isnan(val)]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    if not items or total == 0.0 or math.isnan(total):
        return "n/a"
    return "  ".join(f"{name}={val:.4f} ({100.0 * val / total:5.1f}%)" for name, val in items)


def _iterate_numpy_minibatches(
    inputs: np.ndarray, targets: np.ndarray, batch_size: int, device: str,
    rng: np.random.Generator | None = None,
):
    n = len(inputs)
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, batch_size):
        batch_idx = order[start:start + batch_size]
        if len(batch_idx) == 0:
            continue
        x = torch.from_numpy(inputs[batch_idx].astype(np.float32, copy=False)).to(device)
        y = torch.from_numpy(targets[batch_idx].astype(np.float32, copy=False)).to(device)
        yield x, y


def _interleaved_horizon_batches(
    data: list[tuple[np.ndarray, np.ndarray]], batch_size: int, rng: np.random.Generator,
) -> list[tuple[int, np.ndarray]]:
    """See ``train_ball_dynamics._interleaved_horizon_batches``'s docstring
    for the full rationale (avoiding a systematic per-horizon train-metric
    bias from always processing horizons in the same order within an
    epoch, and dropping the trailing under-batch_size remainder chunk per
    horizon unless it's that horizon's only chunk) -- identical logic
    here."""
    batches: list[tuple[int, np.ndarray]] = []
    for h_idx, (h_inputs, _h_targets) in enumerate(data):
        order = rng.permutation(len(h_inputs))
        h_chunks: list[np.ndarray] = []
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            if len(chunk) == batch_size or not h_chunks:
                h_chunks.append(chunk)
        batches.extend((h_idx, c) for c in h_chunks)
    shuffle_order = rng.permutation(len(batches))
    return [batches[i] for i in shuffle_order]


def _build_horizon_bundle(
    ds: PlayerDynamicsDataset, indices: np.ndarray, n_horizons: int, horizons_s: list[float],
    pair_enabled: bool, pair_max_skip: int, pair_min_start_speed_norm: float,
    has_crossing_data: bool, pair_delta_ok: Callable[[float], bool] | None = None,
) -> dict:
    """Precomputes, per recorded horizon ``h``, everything the shared
    "horizon" training step (see ``_run_interleaved_train_epoch``'s
    "horizon" branch) needs beyond the already-built ``autoencode_train_
    data[h]``/``autoencode_val_data[h]`` input/self-target pair (built
    separately via ``ds.build_autoencoding_data`` -- this function only adds
    what pair AND crossing-head training need ON TOP of that same shared row
    set). Directly mirrors ``train_ball_dynamics._build_horizon_bundle``,
    minus the resting pieces (no player equivalent of "where does the ball
    come to rest").

    ``pair_mask`` (boolean array, one entry per row in ``indices``, True =
    eligible for pair training at this horizon) is a MASK now, not a row
    filter, unlike ``build_adjacent_pair_data`` (which player's training
    loop used to call directly): t0 needs to share the exact same row set
    ``autoencode_train_data[h]`` already uses, so pair's own exclusion has
    to become a per-row mask applied only to ITS loss instead of shrinking
    the shared batch. Uses player's 9-field target layout (``out_of_bounds``
    at index 7, ``goal_scored`` at index 8, velocity at indices 2:4) --
    ``mask_h = (block_h[:, 7] < 0.5) & (block_h[:, 8] < 0.5) & (speed_h >=
    pair_min_start_speed_norm)``, i.e. exclude rows already out-of-
    bounds/scored OR too slow at the start horizon.

    ``pair_targets`` is a list of ``(skip, target_array)`` tuples for
    ``skip in range(1, pair_max_skip + 1)`` while ``h + skip < n_horizons``,
    each ``target_array`` being that later horizon's full target block in
    the SAME row order as ``indices``.

    ``crossing_dt``/``crossing_valid`` are the HORIZON-ADJUSTED crossing
    targets, exactly mirroring ball's: for a pseudo-start at horizon ``h``,
    ``delta_t = crossing_time - horizons_s[h]`` (the crossing POSITION needs
    no adjustment at all -- it's the same fixed (x, y) whichever horizon you
    predict it from, so callers pass ``ds.crossing_pos`` unchanged) when
    there's a genuine FUTURE crossing still ahead (masked=valid). Three
    distinct cases get three distinct treatments: a still-ahead crossing
    gets the real delta_t; horizon h's own recorded oob/goal flags reading
    TRUE right now (``already_there_h``, checked directly rather than
    inferred from ``crossing_time < horizons_s[h]`` -- physics isn't latched
    here either, a player who goes out of bounds keeps walking and can come
    back in, so a stale FIRST-ever ``crossing_time`` being in the past
    doesn't mean they're still out now) gets ``delta_t=0.0``, masked=invalid
    (nothing left to predict, but it's a different fact from "never
    happens" so it gets its own sentinel); neither applies (genuinely never
    crosses AND isn't currently out either -- including an episode excluded
    at generation time for starting out of bounds, see ``player_episode_gen.
    generate_episode``) falls back to ``delta_t=-1.0``, masked=invalid.

    Everything returned is indexed in the SAME row order as ``indices``
    itself (matching ``autoencode_train_data[h]``'s convention: position i
    corresponds to original dataset row ``indices[i]``), so one local
    ``row_idx`` indexes all of them.

    ``pair_delta_ok`` (default ``None`` = keep everything): see
    ``train_ball_dynamics._build_horizon_bundle``'s identical parameter --
    only ``PlayerDynamicsLinearDecoder.has_horizon`` currently passes a
    non-None predicate.

    Returns a dict with per-horizon (length ``n_horizons``) lists:
    ``pair_mask``, ``pair_targets``, ``crossing_dt``, ``crossing_valid``
    (the last two hold ``None`` entries when ``has_crossing_data`` is False)
    -- plus scalar ``n_pair_kept``/``n_pair_dropped`` counts.
    """
    n = len(indices)
    pair_mask: list[np.ndarray] = []
    pair_targets: list[list[tuple[int, np.ndarray]]] = []
    crossing_dt: list[np.ndarray | None] = []
    crossing_valid: list[np.ndarray | None] = []
    n_pair_kept = 0
    n_pair_dropped = 0

    crossing_times_here = ds.crossing_times[indices] if has_crossing_data else None
    for h in range(n_horizons):
        base_h = h * N_TARGET_FIELDS_PER_HORIZON
        block_h = ds.targets[indices, base_h:base_h + N_TARGET_FIELDS_PER_HORIZON]
        if pair_enabled and n_horizons > 1:
            speed_h = np.linalg.norm(block_h[:, 2:4], axis=1)
            mask_h = (block_h[:, 7] < 0.5) & (block_h[:, 8] < 0.5) & (speed_h >= pair_min_start_speed_norm)
            targets_h: list[tuple[int, np.ndarray]] = []
            for skip in range(1, pair_max_skip + 1):
                j = h + skip
                if j >= n_horizons:
                    break
                if pair_delta_ok is not None and not pair_delta_ok(horizons_s[j] - horizons_s[h]):
                    n_pair_dropped += 1
                    continue
                n_pair_kept += 1
                base_j = j * N_TARGET_FIELDS_PER_HORIZON
                targets_h.append((
                    skip, ds.targets[indices, base_j:base_j + N_TARGET_FIELDS_PER_HORIZON].astype(np.float32, copy=False),
                ))
        else:
            mask_h = np.zeros(n, dtype=bool)
            targets_h = []
        pair_mask.append(mask_h)
        pair_targets.append(targets_h)

        if has_crossing_data:
            # `already_there_h` reads horizon h's OWN recorded oob/goal
            # flags (block_h -- already computed above for pair's mask_h)
            # directly, rather than inferring "already crossed by h" from
            # comparing crossing_times_here against horizons_s[h]: physics
            # isn't latched (a player who goes oob keeps walking and can
            # come back in -- see this function's docstring), so a player
            # who crossed once and walked back in bounds by h is NOT
            # "already there" even though their recorded (FIRST-ever)
            # crossing_time is still < horizons_s[h] -- checking the flag
            # directly gets this right in every case. `future_valid` (a
            # real, still-ahead crossing to predict) is masked=True and
            # gets the real delta_t; `already_there_h` (masked=False,
            # nothing left to predict) gets delta_t=0.0 ("already there
            # right now"); neither applies (genuinely never crosses AND
            # isn't currently there either) falls back to -1.0 ("never
            # happens") -- three genuinely different facts, three distinct
            # values, matching train_ball_dynamics.py's identical fix (and
            # this function's own main-task equivalent -- see train()'s
            # already_oob_at_start handling).
            already_there_h = (block_h[:, 7] >= 0.5) | (block_h[:, 8] >= 0.5)
            adj_dt = crossing_times_here - horizons_s[h]
            future_valid = np.isfinite(crossing_times_here) & (adj_dt >= 0) & ~already_there_h
            crossing_dt.append(np.where(future_valid, adj_dt, np.where(already_there_h, 0.0, -1.0)).astype(np.float32))
            crossing_valid.append(future_valid)
        else:
            crossing_dt.append(None)
            crossing_valid.append(None)

    return {
        "pair_mask": pair_mask, "pair_targets": pair_targets,
        "crossing_dt": crossing_dt, "crossing_valid": crossing_valid,
        "n_pair_kept": n_pair_kept, "n_pair_dropped": n_pair_dropped,
    }


def compute_group_sq_err(
    pred_heads: list[torch.Tensor], target: torch.Tensor,
) -> dict[str, list[tuple[float, int]]]:
    """Per-horizon ``(sum of squared error, element count)`` for the 4
    continuous quantity groups (pos/vel/heading/stamina), the raw
    ingredients for an R^2 sanity check -- see
    ``train_ball_dynamics.compute_group_sq_err``. Uses the sin/cos MSE for
    ``heading`` (what's actually optimized), matching ``compute_loss``."""
    out: dict[str, list[tuple[float, int]]] = {g: [] for g in GROUPS}
    for h, head_out in enumerate(pred_heads):
        base = h * N_TARGET_FIELDS_PER_HORIZON
        target_h = target[:, base:base + N_TARGET_FIELDS_PER_HORIZON]
        for g, (lo, hi) in GROUPS.items():
            diff = head_out[:, lo:hi] - target_h[:, lo:hi]
            out[g].append((float((diff ** 2).sum().item()), diff.numel()))
    return out


def _r2_from_sq_err(sq_err: np.ndarray, n: np.ndarray, target_var: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        mse = np.where(n > 0, sq_err / np.maximum(n, 1), np.nan)
        return 1.0 - mse / target_var


def _pct_of_baseline_from_sq_err(sq_err: np.ndarray, n: np.ndarray, baseline_mse: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        mse = np.where(n > 0, sq_err / np.maximum(n, 1), np.nan)
        return 100.0 * np.sqrt(mse / baseline_mse)


def _safe_nanmean(arr: np.ndarray) -> float:
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _classification_metrics(counts: np.ndarray) -> dict[str, np.ndarray]:
    tp, fp, fn, tn = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    total = tp + fp + fn + tn
    with np.errstate(invalid="ignore", divide="ignore"):
        accuracy = np.where(total > 0, (tp + tn) / np.maximum(total, 1), np.nan)
        precision = np.where((tp + fp) > 0, tp / np.maximum(tp + fp, 1), np.nan)
        recall = np.where((tp + fn) > 0, tp / np.maximum(tp + fn, 1), np.nan)
    return {"accuracy": accuracy, "precision": precision, "recall": recall}


def _physics_config_hash() -> str:
    from footballcoach.config import load_physics_config, require_section
    movement = require_section(load_physics_config(), "movement")
    return hashlib.sha256(json.dumps(movement, sort_keys=True).encode()).hexdigest()[:16]


def _pitch_dims_m(input_row: np.ndarray, params) -> tuple[float, float]:
    return float(input_row[17]) * params.base_pitch_length_m, float(input_row[18]) * params.base_pitch_width_m


def _kinematics_denorm_scales(pitch_length_m: float, pitch_width_m: float, params) -> tuple[float, float, float]:
    if params.normalize_kinematics_by_base_pitch:
        half_length = params.base_pitch_length_m / 2
        half_width = params.base_pitch_width_m / 2
        half_diag = (half_length ** 2 + half_width ** 2) ** 0.5
        return half_diag, half_diag, half_diag
    half_length = pitch_length_m / 2
    half_width = pitch_width_m / 2
    half_diag = (half_length ** 2 + half_width ** 2) ** 0.5
    return half_length, half_width, half_diag


def describe_input_row(row: np.ndarray, params) -> str:
    from footballcoach.ai.physics_pretrain.player_episode_gen import PlayerEpisodeGenParams
    assert isinstance(params, PlayerEpisodeGenParams)
    pitch_length_m, pitch_width_m = _pitch_dims_m(row, params)
    half_length, half_width, half_diag = _kinematics_denorm_scales(pitch_length_m, pitch_width_m, params)
    pos = (float(row[0]) * half_length, float(row[1]) * half_width)
    vel = (float(row[2]) * half_diag, float(row[3]) * half_diag)
    heading = math.atan2(float(row[4]), float(row[5]))
    speed_mps = (vel[0] ** 2 + vel[1] ** 2) ** 0.5
    return (
        f"pos=({pos[0]:.2f}, {pos[1]:.2f})m  vel=({vel[0]:.2f}, {vel[1]:.2f})m/s  speed={speed_mps:.2f}m/s  "
        f"heading={heading:.2f}rad  stamina={float(row[6]):.2f}  "
        f"top_speed={float(row[7]):.2f}  accel={float(row[8]):.2f}  stamina_attr={float(row[9]):.2f}  "
        f"ball_control={float(row[10]):.2f}  has_possession={float(row[11]):.0f}  "
        f"pitch={pitch_length_m:.1f}x{pitch_width_m:.1f}m"
    )


def describe_target_row(
    row: np.ndarray, pitch_length_m: float, pitch_width_m: float, params, logits: bool = False,
) -> str:
    half_length, half_width, half_diag = _kinematics_denorm_scales(pitch_length_m, pitch_width_m, params)
    pos = (float(row[0]) * half_length, float(row[1]) * half_width)
    vel = (float(row[2]) * half_diag, float(row[3]) * half_diag)
    heading = math.atan2(float(row[4]), float(row[5]))
    if logits:
        oob_p = 1.0 / (1.0 + np.exp(-float(row[7])))
        goal_p = 1.0 / (1.0 + np.exp(-float(row[8])))
        oob_label, goal_label = f"p={oob_p:.3f}", f"p={goal_p:.3f}"
    else:
        oob_label, goal_label = f"{row[7]:.0f}", f"{row[8]:.0f}"
    return (
        f"pos=({pos[0]:.2f}, {pos[1]:.2f})m  vel=({vel[0]:.2f}, {vel[1]:.2f})m/s  "
        f"heading={heading:.2f}rad  stamina={float(row[6]):.2f}  "
        f"out_of_bounds={oob_label}  goal_scored={goal_label}"
    )


def train(
    dataset_dir: str,
    output_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float = 0.15,
    seed: int = 0,
    pos_weight_max: float | None = None,
    device: str = "cpu",
    open_browser: bool = False,
    init_checkpoint: str | None = None,
    reset_decoder_weights: bool = False,
    reset_optimizer_state: bool = False,
    max_episodes: int | None = None,
    linear_decoder: bool | None = None,
    isolate_crossing: bool = False,
) -> dict:
    from footballcoach.ai.config import load_ai_config
    # Shallow copy -- see train_ball_dynamics.py's identical fix: load_ai_
    # config() is @lru_cache'd (one shared dict for the whole process), so
    # mutating the returned dict in place (linear_decoder_enabled write-back
    # below) would leak into every OTHER unrelated load_ai_config() call for
    # the rest of the process's lifetime.
    cfg = dict(load_ai_config()["physics_pretrain"]["player"])
    linear_decoder = bool(cfg.get("linear_decoder_enabled", False)) if linear_decoder is None else bool(linear_decoder)
    # Write the RESOLVED value back into cfg -- see train_ball_dynamics.py's
    # identical fix for why (cfg is saved verbatim as config_snapshot below;
    # a --linear-decoder CLI override with no matching config-file edit
    # would otherwise save a snapshot that lies about the checkpoint's
    # actual decoder). Safe now that cfg is our own copy.
    cfg["linear_decoder_enabled"] = linear_decoder
    if isolate_crossing:
        # Diagnostic mode: zero out every OTHER latent-level objective so
        # crossing_head is the only thing left pulling on the shared
        # encoder -- tests whether it converges faster/better without
        # competing for encoder capacity against the main reconstruction
        # task and the other auxiliary heads. main_loss_weight=0 kills the
        # main per-horizon reconstruction task (the one loss term with no
        # pre-existing 0.0-disables knob of its own); the rest already have
        # one. adjacent_pair_training_enabled/autoencode_during_main_loop_
        # enabled are left alone deliberately -- crossing_head ALSO trains
        # from the horizon pass's pseudo-starts (see _build_horizon_bundle),
        # so disabling the horizon pass entirely would remove most of
        # crossing_head's own training data, not just the competing tasks.
        # crossing_pos_loss_weight/crossing_dt_loss_weight are left at
        # whatever the config/CLI already resolved -- this flag isolates
        # crossing from OTHER tasks, it doesn't retune crossing itself.
        cfg["main_loss_weight"] = 0.0
        cfg["bce_loss_weight"] = 0.0
        cfg["goal_dist_delta_loss_weight"] = 0.0
        cfg["short_horizon_probe_loss_weight"] = 0.0
        log.info(
            "isolate_crossing: main_loss_weight/bce_loss_weight/goal_dist_delta_loss_weight/"
            "short_horizon_probe_loss_weight all forced to 0.0 -- crossing_head is the only "
            "objective pulling on the shared encoder this run."
        )
    output_path = Path(output_path)

    ds = PlayerDynamicsDataset.from_directory(dataset_dir)
    if max_episodes is not None and max_episodes < len(ds):
        # See train_ball_dynamics.py's identical --max-episodes handling
        # for the full rationale (random subset, seeded off this run's own
        # --seed; typical use: testing whether the network can fit a small
        # subset near-perfectly as a capacity/optimization sanity check).
        full_n = len(ds)
        subset_idx = np.random.default_rng(seed).choice(full_n, size=max_episodes, replace=False)
        ds = ds.subset(subset_idx)
        log.info(f"--max-episodes: limited dataset from {full_n:,} to {len(ds):,} episodes (seed={seed})")
    train_idx, val_idx = ds.split_train_val(val_frac=val_frac, seed=seed)
    log.info(f"Dataset: {len(ds):,} episodes ({len(train_idx):,} train / {len(val_idx):,} val)")
    rng = np.random.default_rng(seed)

    n_horizons = len(cfg["horizons_s"])
    if pos_weight_max is None:
        pos_weight_max = cfg.get("pos_weight_max")
    pos_weight_np = ds.compute_pos_weights(n_horizons, indices=train_idx, max_weight=pos_weight_max)
    pos_weight = torch.from_numpy(pos_weight_np).to(device)

    log.info(f"pos_weight (max cap: {pos_weight_max if pos_weight_max is not None else 'uncapped'}):")
    for h, h_s in enumerate(cfg["horizons_s"]):
        log.info(f"    t={h_s:>4}s  out_of_bounds={pos_weight_np[h, 0]:.2f}  goal_scored={pos_weight_np[h, 1]:.2f}")

    from footballcoach.ai.physics_pretrain.player_episode_gen import PlayerEpisodeGenParams
    gen_params = PlayerEpisodeGenParams.from_config()
    pitch_half_diag_m = math.hypot(gen_params.base_pitch_length_m / 2, gen_params.base_pitch_width_m / 2)
    _RMSE_UNIT_SCALE = {
        "pos_rmse": (pitch_half_diag_m, "m"), "pos_dist": (pitch_half_diag_m, "m"),
        "vel_rmse": (pitch_half_diag_m, "m/s"), "vel_dist": (pitch_half_diag_m, "m/s"),
    }

    # ------------------------------------------------------------------
    # Auxiliary latent heads (see PlayerDynamicsAutoencoder's docstring).
    # Each has ONE config weight; 0.0 disables that head entirely (no
    # gradient, no diagnostics, and -- for the two horizon-pinned heads --
    # no horizons_s validation either, so an unusual horizon set stays
    # usable as long as you're not asking for the head that needs it).
    # ------------------------------------------------------------------
    # crossing_dt_norm_s: divides crossing_dt (raw seconds) by this before
    # it ever reaches the loss, so crossing_head's dt output trains against
    # a roughly-O(1) target like every other quantity in this pipeline,
    # instead of raw seconds spanning [-1, max(horizons_s)]. max(horizons_s)
    # is the natural choice -- it's exactly the largest real value dt can
    # take. See _crossing_head_loss's dt_norm_s docstring for why this
    # (not crossing_dt_loss_weight) is the real fix for the raw-unit scale
    # mismatch: under Adam, a constant loss-weight multiplier is almost
    # entirely cancelled by Adam's own per-parameter normalization for a
    # parameter that's only ever touched by this one loss term (crossing_
    # head's dt-output weight row isn't shared with any other head) -- what
    # actually gates convergence speed is the absolute weight-vector
    # magnitude needed to reach a raw-seconds-scale output from a small
    # random init, and normalizing the target is what actually shrinks that.
    crossing_dt_norm_s = float(max(cfg["horizons_s"]))
    # Split into separate position/delta_t weights (rather than one weight
    # on the combined pos_loss+dt_loss sum) -- kept as separate knobs even
    # though both terms now sit on the same roughly-O(1) scale (thanks to
    # crossing_dt_norm_s above), in case you want to deliberately weight one
    # more than the other for reasons other than compensating for a unit
    # mismatch. See _crossing_head_loss's docstring and crossing_dt_loss_
    # weight's config comment.
    crossing_pos_weight = float(cfg.get("crossing_pos_loss_weight", 0.0))
    # Weight on crossing_head's crosses_logit BCE term -- "does this row
    # have a real crossing/already-crossed instance at all," see
    # _crossing_head_loss's docstring for why this is a separate output
    # from delta_t now (used to be baked into a single delta_t regression
    # against a -1 sentinel, which forced MSE to blend the classification
    # and regression signals together).
    crossing_crosses_weight = float(cfg.get("crossing_crosses_loss_weight", 0.0))
    crossing_dt_weight = float(cfg.get("crossing_dt_loss_weight", 0.0))
    # A dataset built by hand (e.g. in a unit test) may carry no
    # crossings/crossing_times at all -- guard every crossing call site with
    # this rather than crashing, same convention as train_ball_dynamics.py.
    _any_crossing_weight = crossing_pos_weight != 0.0 or crossing_crosses_weight != 0.0 or crossing_dt_weight != 0.0
    has_crossing_data = ds.crossing_pos is not None and _any_crossing_weight
    if _any_crossing_weight and ds.crossing_pos is None:
        log.warning(
            "crossing_pos_loss_weight/crossing_crosses_loss_weight/crossing_dt_loss_weight != 0 but this dataset "
            "carries no crossings/crossing_times -- crossing_head will not be trained."
        )
    if has_crossing_data:
        # player_episode_gen.generate_episode already forces crossing_time
        # to math.inf for episodes that start already out of bounds --
        # baking "started already out of bounds" and "genuinely never
        # crosses" into the SAME sentinel at generation time. Recover the
        # distinction retroactively (see compute_already_out_of_bounds_at_
        # start_mask's docstring for why this needs no dataset
        # regeneration) so the two get different treatment: position stays
        # excluded either way (a near-trivial function of the raw t=0
        # input, not the "will an in-play player actually go out/score"
        # signal the head exists to predict), but delta_t gets its OWN
        # distinct sentinel -- 0.0 ("already there right now"), not -1.0
        # ("never happens"). Mirrors train_ball_dynamics.py's identical
        # already_oob_at_start handling, and _build_horizon_bundle's
        # already_there_h below (same convention at every later horizon).
        already_oob_at_start = ds.compute_already_out_of_bounds_at_start_mask(gen_params)
        ds.crossing_mask = ds.crossing_mask & ~already_oob_at_start
        ds.crossing_dt = np.where(already_oob_at_start, 0.0, ds.crossing_dt).astype(np.float32)

    goal_dist_delta_weight = float(cfg.get("goal_dist_delta_loss_weight", 0.0))
    goal_dist_delta_targets = None
    if goal_dist_delta_weight != 0.0:
        h_idx_3s = _require_horizon_index(cfg["horizons_s"], 3.0, "goal_dist_delta_head's t=3.0s target")
        goal_dist_delta_targets = _goal_dist_delta_targets(ds, gen_params, h_idx_3s, pitch_half_diag_m)

    short_horizon_probe_weight = float(cfg.get("short_horizon_probe_loss_weight", 0.0))
    probe_targets_0_2s = probe_targets_1_0s = None
    if short_horizon_probe_weight != 0.0:
        # Columns 0:4 (pos_x, pos_y, vel_x, vel_y) of each probe horizon's
        # own 9-wide recorded target block -- exactly what the probe heads
        # predict, in the units they're already stored in.
        h_idx_0_2s = _require_horizon_index(cfg["horizons_s"], 0.2, "short_horizon_head_0_2s' target")
        h_idx_1_0s = _require_horizon_index(cfg["horizons_s"], 1.0, "short_horizon_head_1_0s' target")
        b0, b1 = h_idx_0_2s * N_TARGET_FIELDS_PER_HORIZON, h_idx_1_0s * N_TARGET_FIELDS_PER_HORIZON
        probe_targets_0_2s = ds.targets[:, b0:b0 + 4].astype(np.float32, copy=False)
        probe_targets_1_0s = ds.targets[:, b1:b1 + 4].astype(np.float32, copy=False)

    def _log_component(prefix: str, c: str, values: np.ndarray) -> None:
        if c in _RMSE_UNIT_SCALE:
            scale, unit = _RMSE_UNIT_SCALE[c]
            real = values * scale
            log.info(f"    {prefix} {c:12s} by horizon ({unit}): {np.array2string(real, precision=4)}, mean: {real.mean():.4f} {unit}")
        else:
            log.info(f"    {prefix} {c:12s} by horizon: {np.array2string(values, precision=4)}, mean: {values.mean():.4f}")

    def _mean_breakdown_by_horizon(breakdowns_by_h: list[list[dict]]) -> dict[str, np.ndarray]:
        return {
            c: np.array([
                np.mean([b[c] for b in breakdowns_by_h[h]]) if breakdowns_by_h[h] else np.nan
                for h in range(n_horizons)
            ])
            for c in _COMPONENTS
        }

    target_var = ds.compute_group_variance(n_horizons, indices=train_idx)
    persistence_mse = ds.compute_persistence_baseline_mse(n_horizons, indices=train_idx)

    model = PlayerDynamicsAutoencoder(
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        horizons_s=cfg["horizons_s"],
        decoder_hidden_dim=cfg.get("decoder_hidden_dim", 32),
        encoder_bottleneck_dim=cfg.get("encoder_bottleneck_dim", 32),
        identity_shortcut=cfg.get("identity_shortcut_enabled", False),
        identity_shortcut_noise_std=cfg.get("identity_shortcut_noise_std", 0.0),
        encoder_concat_all_input_fields=cfg.get("encoder_concat_all_input_fields", False),
        decoder_identity_shortcut=cfg.get("decoder_identity_shortcut_enabled"),
        linear_decoder=linear_decoder,
        leaky_relu_negative_slope=cfg.get("encoder_leaky_relu_negative_slope", 0.0),
    ).to(device)
    if linear_decoder:
        _ignored = []
        if cfg.get("decoder_identity_shortcut_enabled") is not None or cfg.get("decoder_hidden_dim") is not None:
            _ignored.append("decoder_hidden_dim/decoder_identity_shortcut_enabled")
        ignored_note = f" (configured but has no effect here: {', '.join(_ignored)})" if _ignored else ""
        log.info(
            "linear_decoder_enabled: decoder is now one independent Linear(latent_dim, 4) head per registered "
            f"horizon (pos+vel only, no heading/stamina/oob/goal, no continuous-time interpolation){ignored_note}"
        )
    weight_decay = float(cfg.get("weight_decay", 0.0))
    # Main-loop optimizer choice ONLY -- autoencode_optimizer/decoder_only_
    # optimizer below always stay Adam, since they're early-phase warmup
    # (fast, robust initial descent is exactly what Adam is good at) rather
    # than the late-stage fine-convergence this switch is for. See
    # physics_pretrain.ball.optimizer_type's config comment for the full
    # rationale (identical here): "adam" (default) is fast/robust early on
    # but its per-parameter adaptive step size can produce disproportionately
    # large steps late in a run when a quiet parameter's gradient ticks up
    # even slightly; "sgd" (optionally with momentum via sgd_momentum) is
    # typically smoother/more monotonic once already near a good minimum,
    # at the cost of needing its own (usually larger) lr re-tuned rather
    # than reusing whatever worked for Adam. Meant for resuming a late-stage
    # run via --init-checkpoint, not as a from-scratch default.
    # crossing_head_lr (default None -- no-op, single param group, unchanged
    # behaviour): a SEPARATE, usually much bigger, learning rate for
    # crossing_head's own parameters. Exists because raising crossing_dt_
    # loss_weight turned out NOT to be the lever for making crossing_head's
    # dt output converge faster -- under Adam, a constant loss-weight
    # multiplier on a parameter used by only ONE loss term gets almost
    # entirely cancelled by Adam's own per-parameter normalization (``m/
    # sqrt(v) ~= (w*g)/sqrt((w*g)**2) ~= g/|g|``, independent of w) -- see
    # _crossing_head_loss's dt_norm_s docstring. Confirmed empirically: at
    # crossing_dt_loss_weight=100, crossing_head's dt-row weight norm moved
    # by <0.1% over ~10 minutes of training despite reporting 99.9% of the
    # model's total gradient. What Adam's step size DOES respond to is lr
    # (roughly a fixed ~lr-sized step per iteration, not proportional to
    # remaining distance) -- crossing_head's dt row needs a much bigger
    # weight-vector magnitude than everything else in the model (already
    # ~9 vs ~1-2 for other heads, having started from the same small random
    # init), so it needs proportionally more distance covered per step to
    # keep up, without disturbing the main task's already-converged lr.
    crossing_head_lr = cfg.get("crossing_head_lr")
    crossing_head_lr = float(crossing_head_lr) if crossing_head_lr is not None else None
    _crossing_head_param_ids = {id(p) for p in model.crossing_head.parameters()} if crossing_head_lr is not None else set()

    def _param_groups(base_lr: float) -> list:
        if crossing_head_lr is None:
            return list(model.parameters())
        other_params = [p for p in model.parameters() if id(p) not in _crossing_head_param_ids]
        return [
            {"params": other_params, "lr": base_lr},
            {"params": list(model.crossing_head.parameters()), "lr": crossing_head_lr},
        ]

    _lr_note = f", crossing_head_lr={crossing_head_lr:.2e}" if crossing_head_lr is not None else ""
    optimizer_type = str(cfg.get("optimizer_type", "adam")).lower()
    if optimizer_type == "sgd":
        sgd_momentum = float(cfg.get("sgd_momentum", 0.9))
        optimizer = torch.optim.SGD(_param_groups(lr), lr=lr, momentum=sgd_momentum, weight_decay=weight_decay)
        log.info(f"Main-loop optimizer: SGD (momentum={sgd_momentum}, lr={lr:.2e}{_lr_note}, weight_decay={weight_decay})")
    elif optimizer_type == "adam":
        # See train_ball_dynamics.py's _build_phase_optimizer/adam_beta2
        # comment for the full rationale -- beta2's effective EMA window is
        # ~1/(1-beta2) STEPS, and a large batch_size means far fewer steps
        # per epoch, so how much of one lr-schedule cycle that window
        # actually spans depends heavily on batch_size. Defaults match
        # PyTorch's own (0.9/0.999) when absent, no behaviour change.
        adam_beta1 = float(cfg.get("adam_beta1", 0.9))
        adam_beta2 = float(cfg.get("adam_beta2", 0.999))
        optimizer = torch.optim.Adam(_param_groups(lr), lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
        log.info(f"Main-loop optimizer: Adam (lr={lr:.2e}{_lr_note}, betas=({adam_beta1}, {adam_beta2}), weight_decay={weight_decay})")
    else:
        raise ValueError(f"Unknown physics_pretrain.player.optimizer_type: {optimizer_type!r} (expected 'adam' or 'sgd')")
    # Weight on the MAIN per-horizon reconstruction task (position/velocity/
    # heading/stamina, compute_loss's own return value) -- unlike every
    # other latent-level objective in this file, this one had no 0.0-
    # disables knob of its own until now, since it's normally always-on.
    # Exists for isolate_crossing (see train()'s docstring above) and for
    # anyone else who wants to isolate a different single auxiliary head the
    # same way -- 1.0 (default) is unchanged behaviour.
    main_loss_weight = float(cfg.get("main_loss_weight", 1.0))
    bce_weight = float(cfg.get("bce_loss_weight", 1.0))
    # PlayerDynamicsLinearDecoder only predicts pos+vel (see its docstring)
    # -- heading/stamina come back as constant zero, no gradient source, so
    # weighting them into the trained objective would just bake a
    # meaningless constant offset into every reported train/val loss.
    # bce_weight already gates the decoder's own oob/goal logits the same
    # way (also zero-padded in linear-decoder mode).
    heading_weight = 0.0 if linear_decoder else 1.0
    stamina_weight = 0.0 if linear_decoder else 1.0
    _LOG_COMPONENTS = tuple(
        c for c in _COMPONENTS
        if c != "pos_rmse"
        and not (c in ("oob_bce", "goal_bce") and bce_weight == 0.0)
    )

    if init_checkpoint:
        ckpt = torch.load(init_checkpoint, map_location=device)
        old_cfg = ckpt.get("config_snapshot")
        # If hidden_dim/encoder_bottleneck_dim/latent_dim/decoder_hidden_dim
        # in the CURRENT config don't match what this checkpoint was saved
        # with, a plain load_state_dict can't work at all -- those are real
        # shape mismatches on EXISTING keys, which strict=False doesn't
        # help with (it only tolerates keys missing/extra, never a keeping
        # key whose shape changed). Route through widen_player_checkpoint.
        # py's seam-preserving surgery instead of failing, so bumping those
        # config values and resuming from an old checkpoint just works --
        # see that module's docstring for why this exactly preserves old
        # training rather than approximating it. Mirrors train_ball_
        # dynamics.py's identical widen_needed check.
        # horizons_s changing (e.g. adding/removing recorded horizons and
        # regenerating the dataset) makes the DECODER side of a saved
        # checkpoint unusable, regardless of encoder dims -- see
        # train_ball_dynamics.py's identical check for the full rationale
        # (linear_decoder's single Linear width is directly tied to
        # len(horizons_s); the continuous decoder's t_norm/t_norm_sq/
        # log_horizons buffers are (len(horizons_s),) either way). Checked
        # `widen_needed` (hidden_dim/latent_dim/etc growing) and
        # `horizons_mismatch` are checked/handled INDEPENDENTLY -- either
        # can happen alone, or both at once (see the combined branch below).
        horizons_mismatch = old_cfg is not None and list(old_cfg.get("horizons_s", [])) != list(cfg.get("horizons_s", []))
        widen_needed = old_cfg is not None and any(
            old_cfg.get(k, 32) != cfg.get(k, 32)
            for k in ("hidden_dim", "encoder_bottleneck_dim", "latent_dim", "decoder_hidden_dim")
        )
        if horizons_mismatch and widen_needed:
            # BOTH changed at once: the decoder is unusable regardless (see
            # above), so there's no seam to preserve for it -- but the
            # encoder/aux heads still deserve real widening (not just a
            # shape-mismatch crash) since growing hidden_dim/latent_dim etc
            # has nothing to do with horizons_s. widen_model_'s
            # `widen_decoder=False` skips the decoder seam entirely, leaving
            # `model.decoder` at its own fresh (already-correct, new-
            # horizons-shaped) construction, and `_validate_widen_cfgs`'s
            # `check_decoder=False` skips every decoder-specific stability/
            # rejection check (horizons_s itself, linear_decoder_enabled)
            # since none of them apply when the decoder isn't being touched.
            from footballcoach.ai.physics_pretrain.widen_player_checkpoint import _build_model, _validate_widen_cfgs, widen_model_
            _validate_widen_cfgs(old_cfg, cfg, check_decoder=False)
            old_cfg = {**old_cfg, "encoder_leaky_relu_negative_slope": cfg.get("encoder_leaky_relu_negative_slope", 0.0)}
            old_model = _build_model(old_cfg).to(device)
            if "model_state_dict" in ckpt:
                old_model.load_state_dict(_migrate_crossing_head_state_dict(ckpt["model_state_dict"], old_model), strict=False)
            else:
                old_model.encoder.load_state_dict(ckpt["encoder_state_dict"])
            widen_model_(old_model, model, old_cfg, cfg, widen_decoder=False)
            dims = ", ".join(
                f"{k}: {old_cfg.get(k, 32)}->{cfg.get(k, 32)}"
                for k in ("hidden_dim", "encoder_bottleneck_dim", "latent_dim")
                if old_cfg.get(k, 32) != cfg.get(k, 32)
            )
            log.info(
                f"horizons_s changed ({old_cfg.get('horizons_s')} -> {cfg.get('horizons_s')}) AND encoder dims widened "
                f"({dims}) -- decoder weights are incompatible and were NOT restored (left at fresh init); encoder + "
                f"crossing_head/goal_dist_delta_head/short_horizon_head_0_2s/short_horizon_head_1_0s widened to the new "
                f"dims from {init_checkpoint} (phase={ckpt.get('phase', '?')})"
            )
        elif horizons_mismatch:
            # Only `decoder.*` genuinely depends on horizons_s (shape and/or
            # meaning). crossing_head/goal_dist_delta_head/short_horizon_
            # head_0_2s/short_horizon_head_1_0s are all plain Linear(
            # latent_dim, N) with N independent of horizons_s -- shape-
            # compatible and safe to keep, same as the encoder (their
            # REQUIRED horizon values -- 0.2s/1.0s/3.0s -- are validated by
            # _require_horizon_index at setup time above, before this block
            # even runs, so if we got this far those specific horizons still
            # exist in the new horizons_s). Filters out just the decoder's
            # own keys rather than falling back to a plain encoder-only
            # load, preserving strictly more of the checkpoint's training.
            source = ckpt.get("model_state_dict")
            if source is None:
                source = {f"encoder.{k}": v for k, v in ckpt["encoder_state_dict"].items()}
            filtered = {k: v for k, v in source.items() if not k.startswith("decoder.")}
            missing, unexpected = model.load_state_dict(_migrate_crossing_head_state_dict(filtered, model), strict=False)
            if missing:
                log.info(f"Checkpoint missing {len(missing)} param(s) not present when it was saved (left at fresh init): {missing}")
            if unexpected:
                log.info(f"Checkpoint had {len(unexpected)} unexpected param(s), ignored: {unexpected}")
            log.info(
                f"horizons_s changed ({old_cfg.get('horizons_s')} -> {cfg.get('horizons_s')}) -- decoder weights are "
                f"incompatible and were NOT restored (left at fresh init); everything else (encoder, crossing_head, "
                f"goal_dist_delta_head, short_horizon_head_0_2s, short_horizon_head_1_0s) loaded from {init_checkpoint} "
                f"(phase={ckpt.get('phase', '?')})"
            )
        elif widen_needed:
            from footballcoach.ai.physics_pretrain.widen_player_checkpoint import _build_model, _validate_widen_cfgs, widen_model_
            _validate_widen_cfgs(old_cfg, cfg)
            # encoder_leaky_relu_negative_slope isn't part of the saved
            # weights (LeakyReLU has no learnable params) -- old_model is
            # only a scratch scaffold for widen_model_'s seam surgery to
            # copy FROM, so build it with THIS run's slope (not whatever
            # old_cfg says) same as `model` above, rather than leaving it on
            # a stale value that gets thrown away anyway.
            old_cfg = {**old_cfg, "encoder_leaky_relu_negative_slope": cfg.get("encoder_leaky_relu_negative_slope", 0.0)}
            old_model = _build_model(old_cfg).to(device)
            if "model_state_dict" in ckpt:
                old_model.load_state_dict(_migrate_crossing_head_state_dict(ckpt["model_state_dict"], old_model), strict=False)
            else:
                old_model.encoder.load_state_dict(ckpt["encoder_state_dict"])
            widen_model_(old_model, model, old_cfg, cfg)
            dims = ", ".join(
                f"{k}: {old_cfg.get(k, 32)}->{cfg.get(k, 32)}"
                for k in ("hidden_dim", "encoder_bottleneck_dim", "latent_dim", "decoder_hidden_dim")
                if old_cfg.get(k, 32) != cfg.get(k, 32)
            )
            log.info(f"Widened checkpoint from {init_checkpoint} to current config dims ({dims}); resumed (phase={ckpt.get('phase', '?')})")
        elif "model_state_dict" in ckpt:
            # strict=False: tolerates a checkpoint saved before a param
            # existed at all (e.g. a future new head) -- those params are
            # simply missing from the old state_dict and left at their
            # fresh random init rather than raising, so everything else
            # that WAS restored still resumes. Logged explicitly so a
            # genuinely-missing/unexpected key isn't silently invisible --
            # see train_ball_dynamics.py's identical pattern.
            missing, unexpected = model.load_state_dict(
                _migrate_crossing_head_state_dict(ckpt["model_state_dict"], model), strict=False,
            )
            if missing:
                log.info(f"Checkpoint missing {len(missing)} param(s) not present when it was saved (left at fresh init): {missing}")
            if unexpected:
                log.info(f"Checkpoint had {len(unexpected)} unexpected param(s), ignored: {unexpected}")
            log.info(f"Resumed full model (encoder+decoder) from {init_checkpoint} (phase={ckpt.get('phase', '?')})")
            # Optimizer-state resume is Adam-only, and even then only the
            # per-parameter moment buffers (exp_avg/exp_avg_sq/step) --
            # NEVER the param_groups (lr/betas/weight_decay). See
            # train_ball_dynamics.py's identical restore for the full
            # rationale: SGD never resumes state at all (matches this
            # codebase's "SGD is a fresh late-stage comparison run"
            # convention, and sidesteps the Adam<->SGD state_dict
            # incompatibility that used to crash SGD.step() with
            # KeyError('momentum')); Adam restores ONLY `state`, keeping
            # this run's own configured lr/betas/weight_decay rather than
            # letting load_state_dict() silently overwrite them with
            # whatever the checkpoint's optimizer had at save time.
            ckpt_optimizer_type = ckpt.get("optimizer_type")
            if optimizer_type == "sgd":
                log.info("optimizer_type='sgd' -- optimizer always starts fresh on resume (never restores state from checkpoint)")
            elif reset_optimizer_state:
                # Skip restoring Adam's moment state entirely -- weights
                # still resume normally above, only exp_avg/exp_avg_sq/step
                # are left fresh. Exists because those moments carry over
                # across a --init-checkpoint resume even when this run's
                # loss weights (crossing_pos/crosses/dt_loss_weight etc.)
                # differ from what the checkpoint was saved under: exp_avg_
                # sq (v) is an EMA of squared gradients with an effective
                # memory of ~1/(1-beta2) steps (hundreds to 1000+), so after
                # editing a loss weight and resuming, v stays calibrated to
                # the OLD gradient scale until that many steps have passed
                # -- throttling Adam's effective step size (m_hat/sqrt(
                # v_hat), which is what actually gates how far a step
                # moves, not the raw gradient) far below what the CURRENT,
                # actually-low-noise gradient would support. Verified
                # directly on a real checkpoint: freshly-measured gradient
                # std (many batches, weights held fixed) was 20-50x SMALLER
                # than the checkpoint's own saved sqrt(v_hat) -- v was
                # stale from an earlier, noisier regime (e.g. right after
                # crossing_head's crosses_logit output was freshly reset,
                # or a prior loss-weight value), not tracking reality.
                log.info(
                    "--reset-optimizer-state: Adam moment state (exp_avg/exp_avg_sq/step) NOT restored from "
                    "checkpoint -- starts fresh so a changed loss weight isn't throttled by a stale noise "
                    "estimate (v) calibrated to the old regime"
                )
            elif "optimizer_state_dict" not in ckpt:
                log.info("Checkpoint has no optimizer_state_dict (older artifact) -- optimizer starts fresh")
            else:
                # Guard on the checkpoint's saved state actually being
                # Adam-shaped (has "exp_avg") before attempting a restore --
                # see train_ball_dynamics.py's identical check.
                ckpt_state = ckpt["optimizer_state_dict"].get("state", {})
                is_adam_shaped = bool(ckpt_state) and all("exp_avg" in s for s in ckpt_state.values())
                if not is_adam_shaped:
                    log.info(
                        f"Checkpoint's optimizer state isn't Adam-shaped (saved optimizer_type={ckpt_optimizer_type!r}) "
                        "-- optimizer starts fresh"
                    )
                else:
                    try:
                        this_run_hparams = [dict(g) for g in optimizer.param_groups]
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                        for group, keep in zip(optimizer.param_groups, this_run_hparams):
                            for key in ("lr", "betas", "weight_decay", "eps"):
                                if key in keep:
                                    group[key] = keep[key]
                            # See train_ball_dynamics.py's identical fixup:
                            # strip a leaked "initial_lr" (from the
                            # checkpoint's OWN earlier scheduler) so THIS
                            # run's scheduler (built after this restore
                            # block) anchors its peak to the just-restored
                            # lr above via its own setdefault, not the old
                            # run's peak.
                            if "initial_lr" not in keep:
                                group.pop("initial_lr", None)
                        log.info(
                            "Resumed Adam moment state (exp_avg/exp_avg_sq/step) from checkpoint -- "
                            "lr/betas/weight_decay kept at this run's configured values"
                        )
                    except (ValueError, RuntimeError, KeyError) as e:
                        log.warning(f"Could not restore optimizer state from {init_checkpoint} (starting fresh): {e}")
                    else:
                        # optimizer.load_state_dict only validates the
                        # NUMBER of param groups matches -- it does NOT
                        # check that each individual parameter's restored
                        # exp_avg/exp_avg_sq tensor shape matches that
                        # parameter's CURRENT shape. If a parameter's own
                        # shape changed since the checkpoint was saved (e.g.
                        # crossing_head migrating 3->4 outputs, see
                        # _migrate_crossing_head_state_dict) but the param
                        # GROUP count stayed the same, the mismatched
                        # restore silently "succeeds" here and only crashes
                        # later, inside Adam's own step() (`exp_avg.lerp_
                        # (grad, ...)`), the first time that parameter is
                        # actually updated. Scrub any such entries now so
                        # Adam just initializes fresh state for THAT
                        # parameter on its next step, without discarding
                        # the (still valid) restored state for everything
                        # else -- same targeted-pop pattern reset_decoder_
                        # weights uses below.
                        for group in optimizer.param_groups:
                            for p in group["params"]:
                                st = optimizer.state.get(p)
                                if st is not None and "exp_avg" in st and st["exp_avg"].shape != p.shape:
                                    log.info(f"Dropping stale Adam state for a param whose shape changed since the checkpoint was saved (was {tuple(st['exp_avg'].shape)}, now {tuple(p.shape)}) -- starts fresh for that param only")
                                    optimizer.state.pop(p, None)
        else:
            missing, unexpected = model.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
            if missing:
                log.info(f"Checkpoint missing {len(missing)} encoder param(s) not present when it was saved (left at fresh init): {missing}")
            if unexpected:
                log.info(f"Checkpoint had {len(unexpected)} unexpected encoder param(s), ignored: {unexpected}")
            log.info(f"Resumed encoder only from {init_checkpoint} (decoder left at fresh init)")

    if reset_decoder_weights:
        # Reinitializes every decoder-SIDE module (the shared per-horizon
        # decoder + crossing_head/goal_dist_delta_head/short_horizon_head_
        # 0_2s/short_horizon_head_1_0s -- everything except `model.encoder`)
        # back to a fresh random init, regardless of what --init-checkpoint
        # restored for them. See train_ball_dynamics.py's identical reset
        # for the full rationale (keep an already-good encoder, retrain
        # the decoder side from scratch) and why the optimizer's Adam
        # moment state for these params is also dropped afterward (stale
        # exp_avg/exp_avg_sq from the OLD decoder weights would otherwise
        # get misapplied to the freshly-random ones).
        from footballcoach.ai.physics_pretrain.widen_player_checkpoint import _build_model
        fresh_model = _build_model(cfg)
        decoder_side_modules = ("decoder", "crossing_head", "goal_dist_delta_head", "short_horizon_head_0_2s", "short_horizon_head_1_0s")
        for name in decoder_side_modules:
            getattr(model, name).load_state_dict(getattr(fresh_model, name).state_dict())
        for name in decoder_side_modules:
            for p in getattr(model, name).parameters():
                optimizer.state.pop(p, None)
        log.info(f"Reset decoder-side weights to fresh init ({', '.join(decoder_side_modules)}); encoder left as-is")

    # Latent-space diagnostic snapshot at "epoch 0" -- see
    # train_ball_dynamics.py's identical call site / latent_stats.
    # compute_latent_stats's docstring for what each figure means (dead
    # dims, off-diagonal correlation/redundancy, effective rank via the
    # covariance eigenspectrum, etc.). Pure logging, nothing here feeds
    # back into training.
    from footballcoach.ai.physics_pretrain.latent_stats import compute_latent_stats, format_latent_stats
    latent_stats_at_init = compute_latent_stats(model.encoder, ds.inputs[train_idx], device)
    log.info(format_latent_stats(latent_stats_at_init))

    normalization = {"pitch_half_diag_m": pitch_half_diag_m}

    def _save_phase_checkpoint(phase: str) -> None:
        path = output_path.with_suffix(f".{phase}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            # Main-loop optimizer's state (Adam's per-parameter m/v, or
            # SGD's momentum buffer) -- restored on resume above, so a later
            # `--init-checkpoint` run continues the optimizer's adaptive
            # state instead of rebuilding it cold. See train_ball_dynamics.
            # py's identical save for the full rationale.
            "optimizer_state_dict": optimizer.state_dict(),
            # Which optimizer class the state_dict above belongs to
            # ("adam"/"sgd") -- checked on resume so a state_dict saved
            # under one optimizer type is never fed into a differently-
            # shaped optimizer (see the init_checkpoint restore logic).
            "optimizer_type": optimizer_type,
            "config_snapshot": cfg,
            "normalization": normalization,
            "physics_config_hash": _physics_config_hash(),
            "phase": phase,
        }, path)
        log.info(f"Saved '{phase}' checkpoint to {path}")

    # ------------------------------------------------------------------
    # Derived-data construction (adjacent-pair + t=0 autoencode), shared by
    # every phase below -- pure reshaping of already-recorded data, no new
    # simulation. Built once here so the optional pretrain phases and the
    # main loop's t0 term don't each build their own copy.
    # ------------------------------------------------------------------
    adjacent_pair_enabled = bool(cfg.get("adjacent_pair_training_enabled", True))
    autoencode_during_main_loop_enabled = bool(cfg.get("autoencode_during_main_loop_enabled", True))
    autoencode_pretrain_epochs = int(cfg.get("autoencode_pretrain_epochs", 0))

    # `adjacent_pair_max_skip` (default 1, "adjacent" in the strict sense):
    # how many horizons ahead of each start horizon to ALSO predict, not
    # just the immediately-next one -- see physics_pretrain.player.
    # adjacent_pair_max_skip's config comment (mirrors physics_pretrain.
    # ball's identical key).
    #
    # `adjacent_pair_min_start_speed_mps` (default 0.0, no filtering):
    # excludes (via the mask built below, not a row drop) rows where the
    # player's speed at the START horizon is below this real-m/s threshold
    # -- a near-stationary player makes "predict approximately no movement"
    # a near-trivial target for pair training, same rationale as ball's
    # identical key. Converted to normalized units via pitch_half_diag_m.
    adjacent_pair_max_skip = int(cfg.get("adjacent_pair_max_skip", 1))
    adjacent_pair_min_start_speed_mps = float(cfg.get("adjacent_pair_min_start_speed_mps", 0.0))
    pair_min_start_speed_norm = (
        adjacent_pair_min_start_speed_mps / pitch_half_diag_m if adjacent_pair_min_start_speed_mps > 0.0 else 0.0
    )

    # Adjacent-pair training is UNIFIED with the t0 per-horizon task (see
    # `_build_horizon_bundle`'s and `_run_interleaved_train_epoch`'s
    # "horizon" branch docstrings): rather than deriving its own separately
    # -filtered row set the way `ds.build_adjacent_pair_data` used to, pair
    # training now rides the SAME unfiltered `autoencode_train_data[h]` rows
    # the t0 task uses, with its own resolved-state/speed exclusion applied
    # as a MASK on its loss instead of a row filter -- so a single shared
    # encoder call per horizon-batch can feed both t0 reconstruction and
    # every pair skip at once.
    horizon_pass_enabled = autoencode_during_main_loop_enabled or (adjacent_pair_enabled and n_horizons > 1)

    autoencode_train_data: list[tuple[np.ndarray, np.ndarray]] = []
    autoencode_val_data: list[tuple[np.ndarray, np.ndarray]] = []
    if autoencode_pretrain_epochs > 0 or horizon_pass_enabled:
        autoencode_train_data = [ds.build_autoencoding_data(h, indices=train_idx) for h in range(n_horizons)]
        autoencode_val_data = [ds.build_autoencoding_data(h, indices=val_idx) for h in range(n_horizons)] if len(val_idx) > 0 else []

    # See train_ball_dynamics.py's identical pair_delta_ok -- None (no
    # filtering) for the unrestricted, continuous-time PlayerDynamicsDecoder.
    pair_delta_ok = model.decoder.has_horizon if linear_decoder else None

    horizon_bundle_train: dict = {}
    horizon_bundle_val: dict = {}
    if horizon_pass_enabled:
        horizon_bundle_train = _build_horizon_bundle(
            ds, train_idx, n_horizons, cfg["horizons_s"],
            adjacent_pair_enabled, adjacent_pair_max_skip, pair_min_start_speed_norm, has_crossing_data,
            pair_delta_ok=pair_delta_ok,
        )
        if len(val_idx) > 0:
            horizon_bundle_val = _build_horizon_bundle(
                ds, val_idx, n_horizons, cfg["horizons_s"],
                adjacent_pair_enabled, adjacent_pair_max_skip, pair_min_start_speed_norm, has_crossing_data,
                pair_delta_ok=pair_delta_ok,
            )

    # crossing_head's per-horizon POSITION target is the same fixed (x, y)
    # at every pseudo-start (only delta_t shifts -- see
    # _build_horizon_bundle), pre-sliced into the horizon pass's own row
    # order once here rather than re-slicing ds.crossing_pos by absolute row
    # on every batch. Mirrors train_ball_dynamics.py's identical pair.
    crossing_pos_train = ds.crossing_pos[train_idx] if has_crossing_data else None
    crossing_pos_val = ds.crossing_pos[val_idx] if has_crossing_data and len(val_idx) > 0 else None

    # Row-count summary: how many training EXAMPLES each source contributes
    # this run, logged once up front (after every source above has been
    # built) so their very different contributions are visible at a glance.
    # "main" processes its own dedicated rows as its own batches. The
    # "horizon" pass (t0/pair, both riding the SAME per-(row, horizon)
    # shared encode -- see `_build_horizon_bundle`'s and
    # `_run_interleaved_train_epoch`'s "horizon" branch docstrings)
    # processes ONE set of rows per recorded horizon, with pair masking its
    # OWN eligible subset of that same shared set rather than adding
    # separate rows. crossing_head at t=0 (used only by "main") and at each
    # horizon (used only by the horizon pass) both ride latents those passes
    # already computed. goal_dist_delta_head/the short-horizon probes are
    # MAIN-ROWS-ONLY by design (see their loss functions' docstrings), so
    # they have no horizon-pass line at all. Mirrors train_ball_dynamics.py's
    # identical summary, minus resting_head (no equivalent here).
    n_autoencode_train = sum(len(inp) for inp, _ in autoencode_train_data) if autoencode_train_data else 0
    horizon_lines = ""
    if horizon_pass_enabled:
        n_pair_eligible = (
            int(sum(mask.sum() for mask in horizon_bundle_train["pair_mask"]))
            if adjacent_pair_enabled and n_horizons > 1 else 0
        )
        pair_drop_note = ""
        if pair_delta_ok is not None and adjacent_pair_enabled and n_horizons > 1:
            pair_drop_note = (
                f" ({horizon_bundle_train['n_pair_kept']} (start, skip) combos kept / "
                f"{horizon_bundle_train['n_pair_dropped']} dropped -- delta doesn't land on a registered "
                f"linear-decoder horizon)"
            )
        horizon_lines = (
            f"    autoencode/t0 (bottleneck recon): {n_autoencode_train:,} rows -- own batches "
            f"({len(train_idx):,} rows x {n_horizons} horizons)\n"
            f"    adjacent-pair (dynamics)        : {n_pair_eligible:,}/{n_autoencode_train:,} horizon-pass rows mask-eligible{pair_drop_note} -- "
            f"shares the horizon pass's own latent, no extra rows/batches\n"
        )
        if has_crossing_data:
            n_crossing_h_valid = int(sum(v.sum() for v in horizon_bundle_train["crossing_valid"]))
            horizon_lines += (
                f"    crossing_head (at each horizon) : {n_crossing_h_valid:,}/{n_autoencode_train:,} horizon-pass rows mask-eligible -- "
                f"shares the horizon pass's own latent, no extra rows/batches\n"
            )
    aux_lines = ""
    if has_crossing_data:
        n_crossing_valid = int(ds.crossing_mask[train_idx].sum())
        aux_lines += (
            f"    crossing_head (at t=0, in main) : {n_crossing_valid:,}/{len(train_idx):,} main rows masked-valid (position term only; "
            f"delta_t trains unmasked on the -1 sentinel) -- shares main's own latent\n"
        )
    if goal_dist_delta_weight != 0.0:
        aux_lines += f"    goal_dist_delta_head (main only): {len(train_idx):,} main rows, unmasked -- shares main's own latent\n"
    if short_horizon_probe_weight != 0.0:
        aux_lines += f"    short-horizon probes (main only): {len(train_idx):,} main rows x 2 heads, unmasked -- shares main's own latent\n"
    log.info(
        "Training row-count summary (train split):\n"
        f"    main (per-horizon heads)        : {len(train_idx):,} rows -- own batches\n"
        f"{horizon_lines}{aux_lines}".rstrip("\n")
    )

    # ------------------------------------------------------------------
    # PHASE: autoencode pretraining (optional, own best-val/LR, disabled by
    # default) -- see train_ball_dynamics.py's identical phase.
    # ------------------------------------------------------------------
    if autoencode_pretrain_epochs > 0:
        autoencode_lr = float(cfg.get("autoencode_lr", lr))
        autoencode_optimizer_type = str(cfg.get("autoencode_optimizer_type", "adam")).lower()
        if autoencode_optimizer_type == "sgd":
            autoencode_sgd_momentum = float(cfg.get("autoencode_sgd_momentum", 0.9))
            autoencode_optimizer = torch.optim.SGD(model.parameters(), lr=autoencode_lr, momentum=autoencode_sgd_momentum)
        elif autoencode_optimizer_type == "adam":
            autoencode_adam_beta1 = float(cfg.get("autoencode_adam_beta1", 0.9))
            autoencode_adam_beta2 = float(cfg.get("autoencode_adam_beta2", 0.999))
            autoencode_optimizer = torch.optim.Adam(
                model.parameters(), lr=autoencode_lr, betas=(autoencode_adam_beta1, autoencode_adam_beta2),
            )
        else:
            raise ValueError(
                f"Unknown physics_pretrain.player.autoencode_optimizer_type: {autoencode_optimizer_type!r} (expected 'adam' or 'sgd')"
            )
        log.info(f"Autoencode pretraining: {autoencode_pretrain_epochs} epoch(s), lr={autoencode_lr:.2e}, optimizer={autoencode_optimizer_type}")

        def _eval_autoencode_pass(data_list):
            losses, breakdowns_by_h = [], [[] for _ in range(n_horizons)]
            with torch.no_grad():
                for h_idx, (ae_inputs, ae_targets) in enumerate(data_list):
                    for x, y in _iterate_numpy_minibatches(ae_inputs, ae_targets, batch_size, device):
                        latent = model.encoder(x)
                        pred = model.decoder.forward_at(latent, 0.0)
                        loss, breakdown = _single_target_loss_with_breakdown(pred, y, x, pos_weight[h_idx], bce_weight, heading_weight, stamina_weight)
                        losses.append(float(loss.item()))
                        breakdowns_by_h[h_idx].append(breakdown)
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            return mean_loss, _mean_breakdown_by_horizon(breakdowns_by_h)

        best_val_loss_ae, best_state_ae = float("inf"), None
        for ae_epoch in range(autoencode_pretrain_epochs):
            model.train()
            train_losses_ae, train_breakdowns_by_h = [], [[] for _ in range(n_horizons)]
            ae_batches = list(_interleaved_horizon_batches(autoencode_train_data, batch_size, rng))
            ae_progress = ProgressReporter(
                total=len(ae_batches), prefix=f"  autoencode epoch {ae_epoch + 1}/{autoencode_pretrain_epochs} ", live=True,
            )
            for _ae_step_i, (h_idx, row_idx) in enumerate(ae_batches):
                ae_inputs, ae_targets = autoencode_train_data[h_idx]
                x = torch.from_numpy(ae_inputs[row_idx].astype(np.float32, copy=False)).to(device)
                y = torch.from_numpy(ae_targets[row_idx].astype(np.float32, copy=False)).to(device)
                latent = model.encoder(x)
                pred = model.decoder.forward_at(latent, 0.0)
                loss, breakdown = _single_target_loss_with_breakdown(pred, y, x, pos_weight[h_idx], bce_weight, heading_weight, stamina_weight)
                autoencode_optimizer.zero_grad()
                loss.backward()
                autoencode_optimizer.step()
                train_losses_ae.append(float(loss.item()))
                train_breakdowns_by_h[h_idx].append(breakdown)
                ae_progress.update(_ae_step_i + 1, postfix=f"loss={float(np.mean(train_losses_ae)):.5f}")
            mean_train_loss_ae = float(np.mean(train_losses_ae)) if train_losses_ae else float("nan")
            train_means_ae = _mean_breakdown_by_horizon(train_breakdowns_by_h)

            model.eval()
            mean_val_loss_ae, val_means_ae = (
                _eval_autoencode_pass(autoencode_val_data) if autoencode_val_data else (None, None)
            )
            val_line = f"  val_loss={mean_val_loss_ae:.4f}" if mean_val_loss_ae is not None else ""
            log.info(f"  autoencode pretrain epoch {ae_epoch + 1}/{autoencode_pretrain_epochs}: train_loss={mean_train_loss_ae:.4f}{val_line}")
            for c in _LOG_COMPONENTS:
                _log_component("    train", c, train_means_ae[c])
            if val_means_ae is not None:
                for c in _LOG_COMPONENTS:
                    _log_component("    val  ", c, val_means_ae[c])
                if mean_val_loss_ae < best_val_loss_ae:
                    best_val_loss_ae = mean_val_loss_ae
                    best_state_ae = copy.deepcopy(model.state_dict())
        if best_state_ae is not None:
            model.load_state_dict(best_state_ae)
            log.info(f"Autoencode pretraining: restored best-val weights (val_loss={best_val_loss_ae:.4f})")
        _save_phase_checkpoint("after_autoencode")

    # ------------------------------------------------------------------
    # Interleaved main + adjacent-pair + t0-autoencode passes (+ crossing/
    # goal-dist-delta/short-horizon-probe heads on "main" rows). Each
    # gradient step PAIRS one main batch with one horizon batch and sums
    # both losses into a SINGLE combined backward/step -- the standard
    # multi-task pattern for two objectives sharing one encoder+decoder,
    # mirroring train_ball_dynamics.py's identical pairing (see that
    # function's docstring for the full rationale: this is a strictly
    # stronger fix for cross-pass-type ordering bias than shuffling
    # alternating single-task steps was, since every step is symmetric).
    # The two streams' batch COUNTS rarely match (horizon typically has far
    # more rows than main, since every recorded horizon of every main
    # episode becomes its own horizon-pass row) -- the SHORTER stream is
    # CYCLED (repeated, wrapping around) so every batch of the LONGER
    # stream still gets a partner every step; some of the shorter stream's
    # rows are therefore reused multiple times per epoch, a deliberately
    # accepted tradeoff (see train_ball_dynamics.py's identical note).
    #
    # `main_batches` is built as ONE shuffled pass over `train_idx` (NOT
    # `n_horizons` separate passes -- an earlier version of this function
    # built `n_horizons` independent full permutations all tagged "main",
    # but the main branch below predicts every horizon at once and never
    # actually read that per-chunk `h_idx` tag; investigated against
    # agent_plans/player_physics_pretrain_plan.md and the ball-pretrain
    # plan doc, found no documented reason for the redundant n_horizons-way
    # oversampling of main data, and no equivalent in train_ball_dynamics.
    # py's single-pass main loop -- concluded it was an unintentional
    # copy-paste of horizon-batch-building's per-horizon loop structure,
    # not a deliberate design choice, and fixed to a single pass here).
    #
    # Takes `optimizer` as a parameter (not closed over) so BOTH the main
    # loop below AND decoder-only-pretrain above can share this exact same
    # loop body against their own separate optimizer -- mirrors
    # train_ball_dynamics.py's identical parameterization, which is what
    # lets decoder-only-pretrain there train crossing_head/resting_head too
    # instead of leaving them untouched for that whole phase.
    # ------------------------------------------------------------------
    def _run_interleaved_train_epoch(optimizer: torch.optim.Optimizer, epoch_label: str = "") -> dict:
        model.train()
        # Drops the trailing under-batch_size remainder chunk (if any) --
        # see _interleaved_horizon_batches' docstring for the rationale --
        # except when train_idx itself is smaller than batch_size, where
        # the one partial chunk is kept rather than training on zero main
        # rows every epoch.
        main_batches: list[np.ndarray] = []
        main_order = rng.permutation(len(train_idx))
        for start in range(0, len(main_order), batch_size):
            c = main_order[start:start + batch_size]
            if len(c) == batch_size or not main_batches:
                main_batches.append(train_idx[c])

        horizon_batches: list[tuple[int, np.ndarray]] = []
        if horizon_pass_enabled:
            horizon_batches = list(_interleaved_horizon_batches(autoencode_train_data, batch_size, rng))

        if main_batches and horizon_batches:
            n_pairs = max(len(main_batches), len(horizon_batches))
            main_seq = [main_batches[i % len(main_batches)] for i in range(n_pairs)]
            horizon_seq = [horizon_batches[i % len(horizon_batches)] for i in range(n_pairs)]
            pair_order = rng.permutation(n_pairs)
            pairs: list[tuple[np.ndarray, tuple[int, np.ndarray] | None]] = [
                (main_seq[i], horizon_seq[i]) for i in pair_order
            ]
        else:
            # Horizon pass disabled entirely this run -- main trains alone,
            # one gradient step per main batch, same as always.
            pairs = [(m, None) for m in main_batches]

        losses, breakdowns_by_h = [], [[] for _ in range(n_horizons)]
        # Tracks `backprop_loss` (main + crossing/goal_dist_delta/short_
        # horizon_probe, i.e. everything EXCEPT the horizon-pass's own t0/
        # pair/horizon-crossing terms -- same scope as the "backprop_loss
        # contribution by head" breakdown below) -- used ONLY for the live
        # progress-bar readout, kept separate from `losses` (main-only,
        # what `mean_loss` reports) so val_loss stays directly comparable to
        # train_loss. Mirrors train_ball_dynamics.py's identical
        # train_backprop_losses/running_loss split -- this one was
        # previously wired to `losses` (main-only) by mistake, making the
        # progress bar's live `loss=` silently understate the real
        # backpropped objective whenever crossing/goal_dist_delta/probe
        # contributed a meaningful share of it.
        backprop_losses: list[float] = []
        pair_losses, t0_losses, t0_breakdowns_by_h = [], [], [[] for _ in range(n_horizons)]
        oob_counts = np.zeros((n_horizons, 4), dtype=np.int64)
        goal_counts = np.zeros((n_horizons, 4), dtype=np.int64)
        sq_err = {g: np.zeros(n_horizons) for g in GROUPS}
        n_elem = {g: np.zeros(n_horizons, dtype=np.int64) for g in GROUPS}
        train_grad_norms: list[float] = []
        aux = _AuxAccumulator()
        # WEIGHTED per-step contributions of goal_dist_delta/short_horizon_
        # probe to backprop_loss -- unlike aux's own goal_dist_delta_loss/
        # short_horizon_probe_loss (which stay raw/unweighted for cross-run
        # comparability), these are scaled by their config weight, purely
        # for the backprop_loss-contribution-by-head diagnostic below (not
        # backpropagated themselves). main's own contribution is `losses`
        # itself (weight 1); crossing's is aux.summary()["crossing_loss"]
        # (already weighted -- see _crossing_head_loss).
        train_goal_dist_contrib: list[float] = []
        train_probe_contrib: list[float] = []

        progress = ProgressReporter(total=len(pairs), prefix=f"  {epoch_label} " if epoch_label else "  ", live=True)
        for _step_i, (row_idx, horizon_item) in enumerate(pairs):
            optimizer.zero_grad()

            # ---- main component (always present) ----
            x = torch.from_numpy(ds.inputs[row_idx].astype(np.float32, copy=False)).to(device)
            y = torch.from_numpy(ds.targets[row_idx].astype(np.float32, copy=False)).to(device)
            latent, pred_heads = model(x)
            loss, breakdown = compute_loss(pred_heads, y, x, pos_weight, bce_weight, heading_weight, stamina_weight)
            # `backprop_loss` folds in every auxiliary latent head's term
            # (crossing/goal-dist-delta/short-horizon-probe), same as
            # before pairing existed. `losses` below keeps reporting the
            # main per-horizon `loss` ALONE, same convention as pair/t0 and
            # as train_ball_dynamics.py: otherwise train_loss would carry
            # the aux terms while val_loss (computed independently)
            # doesn't, making the two look wildly divergent for reasons
            # unrelated to over/underfitting. The tensor actually
            # backpropped (`step_loss`) DOES include the paired horizon
            # component below -- only this reported scalar stays main-only.
            backprop_loss = main_loss_weight * loss
            if has_crossing_data:
                crossing_loss, c_pos_dist, c_dt_mae, c_pos_loss, c_crosses_loss, c_dt_loss, c_crosses_acc = _crossing_head_loss(
                    model, latent, ds.crossing_pos, ds.crossing_dt, ds.crossing_mask, row_idx, device,
                    pos_loss_weight=crossing_pos_weight, crosses_loss_weight=crossing_crosses_weight,
                    dt_loss_weight=crossing_dt_weight, dt_norm_s=crossing_dt_norm_s,
                )
                backprop_loss = backprop_loss + crossing_loss
                aux.add_crossing(crossing_loss, c_pos_dist, c_dt_mae, c_pos_loss, c_crosses_loss, c_dt_loss, c_crosses_acc)
            if goal_dist_delta_targets is not None:
                gdd_loss, gdd_mae_left, gdd_mae_right = _goal_dist_delta_head_loss(
                    model, latent, goal_dist_delta_targets, row_idx, device,
                )
                backprop_loss = backprop_loss + goal_dist_delta_weight * gdd_loss
                aux.add_goal_dist(gdd_loss, gdd_mae_left, gdd_mae_right)
                train_goal_dist_contrib.append(goal_dist_delta_weight * float(gdd_loss.item()))
            if probe_targets_0_2s is not None:
                probe_loss, probe_rmse_0_2, probe_rmse_1_0 = _short_horizon_probe_loss(
                    model, latent, probe_targets_0_2s, probe_targets_1_0s, row_idx, device,
                )
                backprop_loss = backprop_loss + short_horizon_probe_weight * probe_loss
                aux.add_probe(probe_loss, probe_rmse_0_2, probe_rmse_1_0)
                train_probe_contrib.append(short_horizon_probe_weight * float(probe_loss.item()))
            step_loss = backprop_loss

            # ---- horizon component -- ONE encoder pass for this recorded
            # horizon's shared pseudo-input, feeding whichever of {t0
            # reconstruction, every pair skip (masked)} are enabled, both
            # summed into `horizon_loss` -- see _build_horizon_bundle's
            # docstring for how the pair targets were derived, and why
            # pair's old row-filter had to become a mask so t0/pair can
            # share the exact same row set. Mirrors train_ball_dynamics.
            # py's "horizon" branch, minus the crossing/resting terms (no
            # equivalents here). Present whenever the horizon pass is
            # enabled at all this run -- cycling above guarantees every
            # pair gets one.
            if horizon_item is not None:
                h_idx, h_row_idx = horizon_item
                ae_inputs, ae_targets = autoencode_train_data[h_idx]
                x_h = torch.from_numpy(ae_inputs[h_row_idx].astype(np.float32, copy=False)).to(device)
                latent_h = model.encoder(x_h)
                horizon_loss = x_h.new_zeros(())

                if autoencode_during_main_loop_enabled:
                    y_h = torch.from_numpy(ae_targets[h_row_idx].astype(np.float32, copy=False)).to(device)
                    pred = model.decoder.forward_at(latent_h, 0.0)
                    t0_loss, t0_breakdown = _single_target_loss_with_breakdown(
                        pred, y_h, x_h, pos_weight[h_idx], bce_weight, heading_weight, stamina_weight,
                    )
                    # main_loss_weight here too (see isolate_crossing) --
                    # t0/pair are the horizon-pass's OWN version of the main
                    # reconstruction task, so they need the same gate as the
                    # main-task `loss` above for isolate_crossing to actually
                    # remove every competing objective, not just the t=0 one.
                    # t0_losses/pair_losses (reporting) stay unweighted.
                    horizon_loss = horizon_loss + main_loss_weight * t0_loss
                    t0_losses.append(float(t0_loss.item()))
                    t0_breakdowns_by_h[h_idx].append(t0_breakdown)

                pair_targets_h = horizon_bundle_train["pair_targets"][h_idx] if adjacent_pair_enabled else []
                if pair_targets_h:
                    mask_f = torch.from_numpy(horizon_bundle_train["pair_mask"][h_idx][h_row_idx]).to(device).float()
                    denom = mask_f.sum().clamp_min(1.0)
                    pair_loss = x_h.new_zeros(())
                    for skip, p_targets in pair_targets_h:
                        y_pair = torch.from_numpy(p_targets[h_row_idx]).to(device)
                        gap = float(cfg["horizons_s"][h_idx + skip] - cfg["horizons_s"][h_idx])
                        pred_pair = model.decoder.forward_at(latent_h, gap)
                        per_ex_loss = _single_target_per_episode_loss(
                            pred_pair, y_pair, x_h, pos_weight[h_idx + skip], bce_weight, heading_weight, stamina_weight,
                        )
                        pair_loss = pair_loss + (per_ex_loss * mask_f).sum() / denom
                    horizon_loss = horizon_loss + main_loss_weight * pair_loss
                    pair_losses.append(float(pair_loss.item()))

                if has_crossing_data:
                    # Horizon-generalized crossing: same head, same fixed
                    # (x, y) target, delta_t re-based on this pseudo-start
                    # (see _build_horizon_bundle). goal_dist_delta/the
                    # short-horizon probes deliberately have NO horizon-pass
                    # counterpart -- see their loss docstrings.
                    crossing_loss_h, c_pos_dist_h, c_dt_mae_h, c_pos_loss_h, c_crosses_loss_h, c_dt_loss_h, c_crosses_acc_h = _crossing_head_loss(
                        model, latent_h, crossing_pos_train, horizon_bundle_train["crossing_dt"][h_idx],
                        horizon_bundle_train["crossing_valid"][h_idx], h_row_idx, device,
                        pos_loss_weight=crossing_pos_weight, crosses_loss_weight=crossing_crosses_weight,
                        dt_loss_weight=crossing_dt_weight, dt_norm_s=crossing_dt_norm_s, trust_negatives=False,
                    )
                    horizon_loss = horizon_loss + crossing_loss_h
                    aux.add_crossing(crossing_loss_h, c_pos_dist_h, c_dt_mae_h, c_pos_loss_h, c_crosses_loss_h, c_dt_loss_h, c_crosses_acc_h)

                step_loss = step_loss + horizon_loss

            step_loss.backward()
            # Total gradient norm across every trainable param, BEFORE
            # optimizer.step() consumes it -- a direct read of how
            # large this step's raw update direction is, independent
            # of `lr`. `clip_grad_norm_` with max_norm=inf computes
            # (and returns) the norm without ever actually clipping/
            # rescaling anything -- the standard portable way to just
            # measure it. Now computed once per PAIRED step (covers both
            # components' combined gradient, not main's alone as before
            # pairing). Diagnostic only -- see train_ball_dynamics.py's
            # identical measurement for the full rationale.
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf")).item())
            train_grad_norms.append(grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
            backprop_losses.append(float(backprop_loss.item()))
            for h in range(n_horizons):
                breakdowns_by_h[h].append({c: getattr(breakdown, c)[h] for c in _COMPONENTS})
            counts = compute_confusion_counts(pred_heads, y, x)
            for h in range(n_horizons):
                oob_counts[h] += counts["oob"][h]
                goal_counts[h] += counts["goal"][h]
            sqerr = compute_group_sq_err(pred_heads, y)
            for g in GROUPS:
                for h in range(n_horizons):
                    s, cnt = sqerr[g][h]
                    sq_err[g][h] += s
                    n_elem[g][h] += cnt

            running_loss = float(np.mean(backprop_losses))
            progress.update(_step_i + 1, postfix=f"loss={running_loss:.5f}")

        aux_summary = aux.summary()
        return {
            "aux": aux_summary,
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
            "breakdowns": _mean_breakdown_by_horizon(breakdowns_by_h),
            "oob_counts": oob_counts, "goal_counts": goal_counts,
            "sq_err": sq_err, "n": n_elem,
            "mean_pair_loss": float(np.mean(pair_losses)) if pair_losses else float("nan"),
            "mean_t0_loss": float(np.mean(t0_losses)) if t0_losses else float("nan"),
            "t0_means": _mean_breakdown_by_horizon(t0_breakdowns_by_h) if any(t0_breakdowns_by_h) else None,
            "grad_norm_stats": _summary_stats(train_grad_norms),
            # Batch-to-batch CHANGE in the main task's own per-batch loss,
            # in the order those "main" gradient steps occurred -- see
            # train_ball_dynamics.py's identical diagnostic.
            "loss_delta_stats": _summary_stats(list(np.diff(losses))) if len(losses) > 1 else _summary_stats([]),
            # WEIGHTED mean contribution of each head to backprop_loss --
            # see train_ball_dynamics.py's identical diagnostic. main is
            # main_loss_weight * losses (usually weight 1, unless
            # main_loss_weight was overridden -- see isolate_crossing);
            # crossing is aux_summary's own crossing_loss (already weighted
            # -- see _crossing_head_loss).
            "backprop_contrib": {
                "main": main_loss_weight * float(np.mean(losses)) if losses else float("nan"),
                "crossing": aux_summary["crossing_loss"],
                "goal_dist_delta": float(np.mean(train_goal_dist_contrib)) if train_goal_dist_contrib else float("nan"),
                "short_horizon_probe": float(np.mean(train_probe_contrib)) if train_probe_contrib else float("nan"),
            },
        }

    def _run_eval_pass(indices: np.ndarray) -> dict:
        model.eval()
        losses, breakdowns_by_h = [], [[] for _ in range(n_horizons)]
        oob_counts = np.zeros((n_horizons, 4), dtype=np.int64)
        goal_counts = np.zeros((n_horizons, 4), dtype=np.int64)
        sq_err = {g: np.zeros(n_horizons) for g in GROUPS}
        n_elem = {g: np.zeros(n_horizons, dtype=np.int64) for g in GROUPS}
        aux = _AuxAccumulator()
        with torch.no_grad():
            # Batched by hand rather than via ds.iterate_minibatches (same
            # unshuffled contiguous chunks it would yield) purely so the
            # auxiliary heads can index their own per-row targets by the
            # batch's dataset row indices, which that generator doesn't
            # expose.
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                if len(batch_idx) == 0:
                    continue
                x = torch.from_numpy(ds.inputs[batch_idx].astype(np.float32, copy=False)).to(device)
                y = torch.from_numpy(ds.targets[batch_idx].astype(np.float32, copy=False)).to(device)
                latent, pred_heads = model(x)
                loss, breakdown = compute_loss(pred_heads, y, x, pos_weight, bce_weight, heading_weight, stamina_weight)
                losses.append(float(loss.item()))
                if has_crossing_data:
                    aux.add_crossing(*_crossing_head_loss(
                        model, latent, ds.crossing_pos, ds.crossing_dt, ds.crossing_mask, batch_idx, device,
                        pos_loss_weight=crossing_pos_weight, crosses_loss_weight=crossing_crosses_weight,
                        dt_loss_weight=crossing_dt_weight, dt_norm_s=crossing_dt_norm_s,
                    ))
                if goal_dist_delta_targets is not None:
                    aux.add_goal_dist(*_goal_dist_delta_head_loss(
                        model, latent, goal_dist_delta_targets, batch_idx, device,
                    ))
                if probe_targets_0_2s is not None:
                    aux.add_probe(*_short_horizon_probe_loss(
                        model, latent, probe_targets_0_2s, probe_targets_1_0s, batch_idx, device,
                    ))
                for h in range(n_horizons):
                    breakdowns_by_h[h].append({c: getattr(breakdown, c)[h] for c in _COMPONENTS})
                counts = compute_confusion_counts(pred_heads, y, x)
                for h in range(n_horizons):
                    oob_counts[h] += counts["oob"][h]
                    goal_counts[h] += counts["goal"][h]
                sqerr = compute_group_sq_err(pred_heads, y)
                for g in GROUPS:
                    for h in range(n_horizons):
                        s, cnt = sqerr[g][h]
                        sq_err[g][h] += s
                        n_elem[g][h] += cnt

            # Horizon-generalized pair/t0 -- one shared eval pass (no
            # gradient), same one-encode-many-losses structure as
            # `_run_interleaved_train_epoch`'s "horizon" branch, just
            # without any backward/step, and a plain sequential pass over
            # every horizon (no interleaving needed here -- eval has no
            # gradient updates for a fixed order to bias). Mirrors
            # train_ball_dynamics.py's `_eval_horizon_pass`.
            pair_losses: list[float] = []
            t0_losses: list[float] = []
            if horizon_pass_enabled:
                for h_idx, (ae_inputs, ae_targets) in enumerate(autoencode_val_data):
                    n_rows = len(ae_inputs)
                    for start in range(0, n_rows, batch_size):
                        row_idx = np.arange(start, min(start + batch_size, n_rows))
                        x = torch.from_numpy(ae_inputs[row_idx].astype(np.float32, copy=False)).to(device)
                        latent = model.encoder(x)

                        if autoencode_during_main_loop_enabled:
                            y = torch.from_numpy(ae_targets[row_idx].astype(np.float32, copy=False)).to(device)
                            pred = model.decoder.forward_at(latent, 0.0)
                            t0_loss = _single_target_loss_with_breakdown(pred, y, x, pos_weight[h_idx], bce_weight, heading_weight, stamina_weight)[0]
                            t0_losses.append(float(t0_loss.item()))

                        pair_targets_h = horizon_bundle_val["pair_targets"][h_idx] if adjacent_pair_enabled else []
                        if pair_targets_h:
                            mask_f = torch.from_numpy(horizon_bundle_val["pair_mask"][h_idx][row_idx]).to(device).float()
                            denom = mask_f.sum().clamp_min(1.0)
                            pair_loss = x.new_zeros(())
                            for skip, p_targets in pair_targets_h:
                                y_pair = torch.from_numpy(p_targets[row_idx]).to(device)
                                gap = float(cfg["horizons_s"][h_idx + skip] - cfg["horizons_s"][h_idx])
                                pred_pair = model.decoder.forward_at(latent, gap)
                                per_ex_loss = _single_target_per_episode_loss(
                                    pred_pair, y_pair, x, pos_weight[h_idx + skip], bce_weight, heading_weight, stamina_weight,
                                )
                                pair_loss = pair_loss + (per_ex_loss * mask_f).sum() / denom
                            pair_losses.append(float(pair_loss.item()))

                        if has_crossing_data:
                            aux.add_crossing(*_crossing_head_loss(
                                model, latent, crossing_pos_val, horizon_bundle_val["crossing_dt"][h_idx],
                                horizon_bundle_val["crossing_valid"][h_idx], row_idx, device,
                                pos_loss_weight=crossing_pos_weight, crosses_loss_weight=crossing_crosses_weight,
                        dt_loss_weight=crossing_dt_weight, dt_norm_s=crossing_dt_norm_s, trust_negatives=False,
                            ))

        return {
            "aux": aux.summary(),
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
            "breakdowns": _mean_breakdown_by_horizon(breakdowns_by_h),
            "oob_counts": oob_counts, "goal_counts": goal_counts,
            "sq_err": sq_err, "n": n_elem,
            "mean_pair_loss": float(np.mean(pair_losses)) if pair_losses else float("nan"),
            "mean_t0_loss": float(np.mean(t0_losses)) if t0_losses else float("nan"),
        }

    def _log_aux_diagnostics(train_aux: dict, val_aux: dict) -> None:
        """Shared by the main loop and decoder-only-pretrain -- one log
        line per ENABLED auxiliary latent head (a disabled head's metrics
        are all NaN and its line is skipped entirely rather than printing
        a row of nans every epoch). crossing_pos_dist and the goal-distance
        MAEs are in normalized position units, so both are also shown in
        metres via pitch_half_diag_m; crossing_dt_mae is already in
        seconds."""
        if has_crossing_data:
            log.info(
                f"    crossing_head: train loss={train_aux['crossing_loss']:.4f} "
                f"pos_dist={train_aux['crossing_pos_dist'] * pitch_half_diag_m:.3f}m "
                f"dt_mae={train_aux['crossing_dt_mae']:.3f}s crosses_acc={train_aux['crossing_crosses_acc']:.3f} | "
                f"val loss={val_aux['crossing_loss']:.4f} "
                f"pos_dist={val_aux['crossing_pos_dist'] * pitch_half_diag_m:.3f}m "
                f"dt_mae={val_aux['crossing_dt_mae']:.3f}s crosses_acc={val_aux['crossing_crosses_acc']:.3f}"
            )
            # Split of crossing_head's own loss between its three WEIGHTED
            # sub-terms (pos/crosses/dt sum to crossing_loss above) -- reuses
            # _format_backprop_contrib's same name=value(pct%) formatting
            # as the top-level "backprop_loss contribution by head" line,
            # one level down: which PART of crossing_head's own loss is
            # pos/crosses/dt, not which head contributes to the total.
            train_split = {
                "pos": train_aux["crossing_pos_loss"], "crosses": train_aux["crossing_crosses_loss"],
                "dt": train_aux["crossing_dt_loss"],
            }
            val_split = {
                "pos": val_aux["crossing_pos_loss"], "crosses": val_aux["crossing_crosses_loss"],
                "dt": val_aux["crossing_dt_loss"],
            }
            log.info(
                f"        pos/crosses/dt split: train {_format_backprop_contrib(train_split, train_aux['crossing_loss'])} | "
                f"val {_format_backprop_contrib(val_split, val_aux['crossing_loss'])}"
            )
        if goal_dist_delta_weight != 0.0:
            log.info(
                f"    goal_dist_delta_head: train loss={train_aux['goal_dist_delta_loss']:.5f} "
                f"mae=(left {train_aux['goal_dist_delta_mae_left'] * pitch_half_diag_m:.3f}m, "
                f"right {train_aux['goal_dist_delta_mae_right'] * pitch_half_diag_m:.3f}m) | "
                f"val loss={val_aux['goal_dist_delta_loss']:.5f} "
                f"mae=(left {val_aux['goal_dist_delta_mae_left'] * pitch_half_diag_m:.3f}m, "
                f"right {val_aux['goal_dist_delta_mae_right'] * pitch_half_diag_m:.3f}m)"
            )
        if short_horizon_probe_weight != 0.0:
            log.info(
                # RMSE left in NORMALIZED units here (unlike the two heads
                # above): each probe's 4 outputs mix position and velocity,
                # which don't share a real-world unit, so a single
                # metres-scaled number would be actively misleading.
                f"    short_horizon_probes: train loss={train_aux['short_horizon_probe_loss']:.5f} "
                f"rmse_norm=(0.2s {train_aux['short_horizon_rmse_0_2s']:.4f}, "
                f"1.0s {train_aux['short_horizon_rmse_1_0s']:.4f}) | "
                f"val loss={val_aux['short_horizon_probe_loss']:.5f} "
                f"rmse_norm=(0.2s {val_aux['short_horizon_rmse_0_2s']:.4f}, "
                f"1.0s {val_aux['short_horizon_rmse_1_0s']:.4f})"
            )

    # ------------------------------------------------------------------
    # PHASE: decoder-only pretraining (optional, own best-val/early-stop,
    # disabled by default) -- see train_ball_dynamics.py's identical phase.
    # ------------------------------------------------------------------
    decoder_only_pretrain_epochs = int(cfg.get("decoder_only_pretrain_epochs", 0))
    if decoder_only_pretrain_epochs > 0:
        decoder_only_lr = float(cfg.get("decoder_only_pretrain_lr", lr))
        freeze_latent = bool(cfg.get("decoder_only_pretrain_freeze_latent", True))
        do_early_stop_patience = int(cfg.get("decoder_only_early_stop_patience", 0))
        do_early_stop_min_delta = float(cfg.get("decoder_only_early_stop_min_delta", 1e-6))

        for p in model.encoder.trunk.parameters():
            p.requires_grad_(False)
        identity_mask_hook = None
        # The 4 auxiliary latent heads (crossing_head/goal_dist_delta_head/
        # short_horizon_head_0_2s/short_horizon_head_1_0s) are included here
        # regardless of freeze_latent -- they read the latent directly, same
        # as the decoder does, so there's nothing latent-frozen about
        # training THEM even when the encoder itself is fully frozen this
        # phase. Mirrors train_ball_dynamics.py's decoder_only_params, which
        # includes crossing_head/resting_head/position_head the same way.
        aux_head_params = (
            list(model.crossing_head.parameters()) + list(model.goal_dist_delta_head.parameters())
            + list(model.short_horizon_head_0_2s.parameters()) + list(model.short_horizon_head_1_0s.parameters())
        )
        if freeze_latent:
            for p in model.encoder.out.parameters():
                p.requires_grad_(False)
            decoder_only_params = list(model.decoder.parameters()) + aux_head_params
        else:
            n_id = N_IDENTITY_SHORTCUT_FIELDS
            id_mask = torch.ones_like(model.encoder.out.weight)
            id_mask[:n_id, :] = 0.0
            identity_mask_hook = model.encoder.out.weight.register_hook(lambda grad: grad * id_mask)
            decoder_only_params = list(model.decoder.parameters()) + list(model.encoder.out.parameters()) + aux_head_params
        decoder_only_optimizer_type = str(cfg.get("decoder_only_optimizer_type", "adam")).lower()
        if decoder_only_optimizer_type == "sgd":
            decoder_only_sgd_momentum = float(cfg.get("decoder_only_sgd_momentum", 0.9))
            decoder_only_optimizer = torch.optim.SGD(decoder_only_params, lr=decoder_only_lr, momentum=decoder_only_sgd_momentum)
        elif decoder_only_optimizer_type == "adam":
            decoder_only_adam_beta1 = float(cfg.get("decoder_only_adam_beta1", 0.9))
            decoder_only_adam_beta2 = float(cfg.get("decoder_only_adam_beta2", 0.999))
            decoder_only_optimizer = torch.optim.Adam(
                decoder_only_params, lr=decoder_only_lr, betas=(decoder_only_adam_beta1, decoder_only_adam_beta2),
            )
        else:
            raise ValueError(
                f"Unknown physics_pretrain.player.decoder_only_optimizer_type: {decoder_only_optimizer_type!r} (expected 'adam' or 'sgd')"
            )
        log.info(
            f"Decoder-only pretraining: {decoder_only_pretrain_epochs} epoch(s), lr={decoder_only_lr:.2e}, "
            f"optimizer={decoder_only_optimizer_type}, freeze_latent={freeze_latent}"
        )

        best_val_loss_do, best_state_do, patience_ctr_do = float("inf"), None, 0
        for do_epoch in range(decoder_only_pretrain_epochs):
            # Reuses the exact same interleaved-epoch/eval-pass functions
            # the main loop below uses (see their shared definition's
            # docstring) -- this is what lets crossing_head/goal_dist_
            # delta_head/the short-horizon probes actually train during
            # this phase instead of sitting untouched until the main loop,
            # mirroring train_ball_dynamics.py's identical reuse for
            # crossing_head/resting_head/position_head.
            train_res_do = _run_interleaved_train_epoch(
                decoder_only_optimizer, epoch_label=f"decoder-only epoch {do_epoch + 1}/{decoder_only_pretrain_epochs}",
            )
            val_res_do = _run_eval_pass(val_idx) if len(val_idx) > 0 else train_res_do
            mean_train_loss_do = train_res_do["mean_loss"]
            mean_val_loss_do = val_res_do["mean_loss"] if len(val_idx) > 0 else float("inf")
            raw_drop_do = best_val_loss_do - mean_val_loss_do
            improved_do = raw_drop_do > do_early_stop_min_delta
            do_verdict = (
                f"(improved by {raw_drop_do:.6f} > min_delta={do_early_stop_min_delta:.1e})" if improved_do
                else f"(patience {patience_ctr_do + 1}/{do_early_stop_patience}, raw_drop={raw_drop_do:.6f} <= min_delta={do_early_stop_min_delta:.1e})"
            )
            log.info(
                f"  decoder-only pretrain epoch {do_epoch + 1}/{decoder_only_pretrain_epochs}: "
                f"train_loss={mean_train_loss_do:.4f}  val_loss={mean_val_loss_do:.4f}  {do_verdict}"
            )
            _log_aux_diagnostics(train_res_do["aux"], val_res_do["aux"])

            if improved_do:
                best_val_loss_do = mean_val_loss_do
                best_state_do = copy.deepcopy(model.state_dict())
                patience_ctr_do = 0
            else:
                patience_ctr_do += 1
                if do_early_stop_patience > 0 and patience_ctr_do >= do_early_stop_patience:
                    log.info(f"  decoder-only pretrain: early stopping after {do_epoch + 1} epochs (patience={do_early_stop_patience})")
                    break
        if best_state_do is not None:
            model.load_state_dict(best_state_do)
            log.info(f"Decoder-only pretraining: restored best-val weights (val_loss={best_val_loss_do:.4f})")

        if identity_mask_hook is not None:
            identity_mask_hook.remove()
        for p in model.encoder.parameters():
            p.requires_grad_(True)
        _save_phase_checkpoint("after_decoder_pretrain")


    early_stop_patience = int(cfg.get("early_stop_patience", 0))
    early_stop_min_delta = float(cfg.get("early_stop_min_delta", 1e-4))
    early_stop_enabled = early_stop_patience > 0

    history: list[dict] = []
    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0
    n_epochs_run = 0
    prev_val_loss = float("nan")
    prev_val_backprop_loss = float("nan")
    prev_val_backprop_contrib: dict[str, float] | None = None
    def _log_val_baseline() -> None:
        """Logs an "epoch 0 (before training)" val-only baseline -- same
        metrics/format as a normal epoch's val portion (see the main loop
        below), just before any gradient step has run, so the very first
        real epoch's numbers have something to compare against instead of
        being the first data point on the chart. Reuses `_run_eval_pass`
        (the SAME function the main loop's own val evaluation calls), so
        the two can't drift apart in what they measure. Mirrors
        train_ball_dynamics.py's identical `_log_val_baseline`."""
        if len(val_idx) == 0:
            return
        val_res = _run_eval_pass(val_idx)
        val_aux = val_res["aux"]
        log.info(
            f"epoch 0/{epochs} (before training): val_loss={val_res['mean_loss']:.4f}  "
            f"pair_loss={val_res['mean_pair_loss']:.4f}  t0_loss={val_res['mean_t0_loss']:.4f}"
        )
        if has_crossing_data:
            log.info(
                f"    crossing_head: val loss={val_aux['crossing_loss']:.4f} "
                f"pos_dist={val_aux['crossing_pos_dist'] * pitch_half_diag_m:.3f}m "
                f"dt_mae={val_aux['crossing_dt_mae']:.3f}s crosses_acc={val_aux['crossing_crosses_acc']:.3f}"
            )
        if goal_dist_delta_weight != 0.0:
            log.info(
                f"    goal_dist_delta_head: val loss={val_aux['goal_dist_delta_loss']:.5f} "
                f"mae=(left {val_aux['goal_dist_delta_mae_left'] * pitch_half_diag_m:.3f}m, "
                f"right {val_aux['goal_dist_delta_mae_right'] * pitch_half_diag_m:.3f}m)"
            )
        if short_horizon_probe_weight != 0.0:
            log.info(
                f"    short_horizon_probes: val loss={val_aux['short_horizon_probe_loss']:.5f} "
                f"rmse_norm=(0.2s {val_aux['short_horizon_rmse_0_2s']:.4f}, "
                f"1.0s {val_aux['short_horizon_rmse_1_0s']:.4f})"
            )
        for c in _LOG_COMPONENTS:
            _log_component("    val  ", c, val_res["breakdowns"][c])
        val_pctd = {g: _pct_of_baseline_from_sq_err(val_res["sq_err"][g], val_res["n"][g], persistence_mse[g]) for g in GROUPS}
        for g in GROUPS:
            log.info(f"    val   {g:9s} %-of-persistence by horizon: {np.array2string(val_pctd[g], precision=1)}")
        if bce_weight != 0.0:
            val_oob_cls = _classification_metrics(val_res["oob_counts"])
            val_goal_cls = _classification_metrics(val_res["goal_counts"])
            for label, vm in (("oob", val_oob_cls), ("goal", val_goal_cls)):
                for key in ("accuracy", "precision", "recall"):
                    log.info(
                        f"    val   {label}_{key:9s} by horizon: {np.array2string(vm[key], precision=3)}, mean: {_safe_nanmean(vm[key]):.3f}"
                    )

    _log_val_baseline()

    for epoch in range(epochs):
        train_res = _run_interleaved_train_epoch(optimizer, epoch_label=f"epoch {epoch + 1}/{epochs}")
        val_res = _run_eval_pass(val_idx) if len(val_idx) > 0 else train_res
        n_epochs_run += 1

        # Always overwritten every epoch, regardless of whether this epoch
        # actually improved -- unlike `midtrain_latest` (val-gated, only
        # updates on a new best), this is meant as "what does the model
        # look like right now," for e.g. killing/resuming a run without
        # losing anything more than the current epoch's progress. Was
        # previously best-train-loss-gated (same convention as
        # midtrain_latest); changed since a genuinely "latest" checkpoint
        # is more useful once training plateaus/gets noisy and best-train-
        # loss stops updating for many epochs at a stretch. Mirrors
        # train_ball_dynamics.py's identical change.
        _save_phase_checkpoint("midtrain_latest_train")

        train_r2 = {g: _r2_from_sq_err(train_res["sq_err"][g], train_res["n"][g], target_var[g]) for g in GROUPS}
        val_r2 = {g: _r2_from_sq_err(val_res["sq_err"][g], val_res["n"][g], target_var[g]) for g in GROUPS}
        train_pctd = {g: _pct_of_baseline_from_sq_err(train_res["sq_err"][g], train_res["n"][g], persistence_mse[g]) for g in GROUPS}
        val_pctd = {g: _pct_of_baseline_from_sq_err(val_res["sq_err"][g], val_res["n"][g], persistence_mse[g]) for g in GROUPS}

        train_oob_cls = _classification_metrics(train_res["oob_counts"])
        train_goal_cls = _classification_metrics(train_res["goal_counts"])
        val_oob_cls = _classification_metrics(val_res["oob_counts"])
        val_goal_cls = _classification_metrics(val_res["goal_counts"])

        grad_norm_stats = train_res["grad_norm_stats"]
        loss_delta_stats = train_res["loss_delta_stats"]

        val_loss = val_res["mean_loss"]
        # raw_drop is the plain best_val_loss - val_loss difference, before
        # the min_delta bar is applied -- logged alongside the pass/fail
        # verdict so "why didn't patience reset" is answerable from this
        # line alone instead of needing to cross-reference the config.
        raw_drop = best_val_loss - val_loss
        improved = raw_drop > early_stop_min_delta
        val_line = f"  val_loss={val_loss:.4f}  best={min(best_val_loss, val_loss):.4f}"
        if early_stop_enabled:
            if improved:
                val_line += f"  (improved by {raw_drop:.6f} > min_delta={early_stop_min_delta:.1e})"
            else:
                val_line += (
                    f"  (patience {patience_ctr + 1}/{early_stop_patience}, "
                    f"raw_drop={raw_drop:.6f} <= min_delta={early_stop_min_delta:.1e})"
                )
        log.info(
            f"epoch {epoch + 1}/{epochs}: train_loss={train_res['mean_loss']:.4f}  "
            f"pair_loss={train_res['mean_pair_loss']:.4f}  t0_loss={train_res['mean_t0_loss']:.4f}{val_line}"
        )
        # Convergence diagnostics -- see train_ball_dynamics.py's identical
        # per-epoch lines/docstring for the full rationale. 6dp since at a
        # converged loss scale these deltas/norms are themselves small
        # enough that 4dp would round most of them to 0.0000.
        log.info(
            f"    grad_norm: mean={grad_norm_stats['mean']:.6f} std={grad_norm_stats['std']:.6f} "
            f"min={grad_norm_stats['min']:.6f} max={grad_norm_stats['max']:.6f}"
        )
        log.info(
            f"    train_loss_delta (batch-to-batch): mean={loss_delta_stats['mean']:.6f} std={loss_delta_stats['std']:.6f} "
            f"min={loss_delta_stats['min']:.6f} max={loss_delta_stats['max']:.6f}"
        )
        train_backprop_contrib = train_res["backprop_contrib"]
        mean_train_backprop_loss = sum(v for v in train_backprop_contrib.values() if not math.isnan(v))
        log.info(f"    backprop_loss contribution by head: {_format_backprop_contrib(train_backprop_contrib, mean_train_backprop_loss)}")
        train_aux, val_aux = train_res["aux"], val_res["aux"]
        _log_aux_diagnostics(train_aux, val_aux)
        val_loss_delta = float("nan")
        val_backprop_loss_delta = float("nan")
        mean_val_backprop_loss = float("nan")
        if len(val_idx) > 0:
            # Epoch-over-epoch change in val_loss (negative = improved) --
            # nan on the first epoch (no prior value yet). See
            # train_ball_dynamics.py's identical diagnostic.
            val_loss_delta = val_loss - prev_val_loss if not np.isnan(prev_val_loss) else float("nan")
            prev_val_loss = val_loss
            log.info(f"    val_loss_delta (epoch-over-epoch): {val_loss_delta:.6f}")
            # Same idea, but tracking the combined objective (val_loss +
            # crossing/goal_dist_delta/short_horizon_probe, mirroring
            # train_backprop_loss's own composition) instead of the bare
            # per-horizon val_loss -- val_loss alone can look flat/improving
            # while one of the auxiliary heads is actually degrading (or
            # vice versa), which this surfaces directly. crossing_loss is
            # already weighted (see _crossing_head_loss); goal_dist_delta/
            # short_horizon_probe aren't, so their config weights are
            # applied here the same way _run_interleaved_train_epoch does
            # for the train side. See train_ball_dynamics.py's identical
            # val_backprop_loss_delta diagnostic.
            val_backprop_contrib = {
                "main": main_loss_weight * val_loss,
                "crossing": val_aux["crossing_loss"],
                "goal_dist_delta": goal_dist_delta_weight * val_aux["goal_dist_delta_loss"],
                "short_horizon_probe": short_horizon_probe_weight * val_aux["short_horizon_probe_loss"],
            }
            mean_val_backprop_loss = sum(v for v in val_backprop_contrib.values() if not math.isnan(v))
            val_backprop_loss_delta = (
                mean_val_backprop_loss - prev_val_backprop_loss if not np.isnan(prev_val_backprop_loss) else float("nan")
            )
            prev_val_backprop_loss = mean_val_backprop_loss
            # Per-head breakdown of THAT delta -- which head's own val
            # contribution actually moved epoch-over-epoch, not just the
            # summed total (a flat/improving total can hide one head
            # regressing while another improves by a similar amount).
            # Reuses _format_backprop_contrib's existing signed-value/
            # near-zero-total handling (it already just divides by total
            # and sorts by magnitude, no assumption every value is
            # positive) -- same helper the absolute (non-delta)
            # backprop_loss-contribution-by-head line above already uses.
            # nan on the first epoch (no prior per-head snapshot yet), and
            # for any individual head that's nan in either epoch (a head
            # disabled all along stays nan; one that just got disabled/
            # enabled this epoch has no meaningful delta yet).
            val_backprop_contrib_delta: dict[str, float] = {}
            if prev_val_backprop_contrib is not None:
                for name, val in val_backprop_contrib.items():
                    prev_val = prev_val_backprop_contrib.get(name, float("nan"))
                    if not math.isnan(val) and not math.isnan(prev_val):
                        val_backprop_contrib_delta[name] = val - prev_val
            prev_val_backprop_contrib = dict(val_backprop_contrib)
            log.info(
                f"    val_backprop_loss_delta (epoch-over-epoch): {val_backprop_loss_delta:.6f}  "
                f"({_format_backprop_contrib(val_backprop_contrib_delta, val_backprop_loss_delta)})"
            )
        for c in _LOG_COMPONENTS:
            _log_component("    train", c, train_res["breakdowns"][c])
            _log_component("    val  ", c, val_res["breakdowns"][c])
        for g in GROUPS:
            log.info(f"    train {g:9s} %-of-persistence by horizon: {np.array2string(train_pctd[g], precision=1)}")
            log.info(f"    val   {g:9s} %-of-persistence by horizon: {np.array2string(val_pctd[g], precision=1)}")
        if bce_weight != 0.0:
            for label, tm, vm in (("oob", train_oob_cls, val_oob_cls), ("goal", train_goal_cls, val_goal_cls)):
                for key in ("accuracy", "precision", "recall"):
                    log.info(
                        f"    train {label}_{key:9s} by horizon: {np.array2string(tm[key], precision=3)}, mean: {_safe_nanmean(tm[key]):.3f}"
                    )
                    log.info(
                        f"    val   {label}_{key:9s} by horizon: {np.array2string(vm[key], precision=3)}, mean: {_safe_nanmean(vm[key]):.3f}"
                    )

        epoch_record = {"epoch": epoch, "train_loss": train_res["mean_loss"], "val_loss": val_loss, "lr": lr,
                         "train_pair_loss": train_res["mean_pair_loss"], "val_pair_loss": val_res["mean_pair_loss"],
                         "train_t0_loss": train_res["mean_t0_loss"], "val_t0_loss": val_res["mean_t0_loss"],
                         "grad_norm_mean": grad_norm_stats["mean"], "grad_norm_std": grad_norm_stats["std"],
                         "grad_norm_min": grad_norm_stats["min"], "grad_norm_max": grad_norm_stats["max"],
                         "train_loss_delta_mean": loss_delta_stats["mean"], "train_loss_delta_std": loss_delta_stats["std"],
                         "train_loss_delta_min": loss_delta_stats["min"], "train_loss_delta_max": loss_delta_stats["max"],
                         "val_loss_delta": val_loss_delta,
                         "val_backprop_loss": mean_val_backprop_loss, "val_backprop_loss_delta": val_backprop_loss_delta}
        for k in _AUX_METRIC_KEYS:
            epoch_record[f"train_{k}"] = train_aux[k]
            epoch_record[f"val_{k}"] = val_aux[k]
        for c in _COMPONENTS:
            epoch_record[f"train_{c}"] = train_res["breakdowns"][c]
            epoch_record[f"val_{c}"] = val_res["breakdowns"][c]
        for g in GROUPS:
            epoch_record[f"train_r2_{g}"] = train_r2[g]
            epoch_record[f"val_r2_{g}"] = val_r2[g]
            epoch_record[f"train_pctd_{g}"] = train_pctd[g]
            epoch_record[f"val_pctd_{g}"] = val_pctd[g]
        for label, tm, vm in (("oob", train_oob_cls, val_oob_cls), ("goal", train_goal_cls, val_goal_cls)):
            for key in ("accuracy", "precision", "recall"):
                epoch_record[f"train_{label}_{key}"] = tm[key]
                epoch_record[f"val_{label}_{key}"] = vm[key]
        history.append(epoch_record)

        if improved:
            best_val_loss = val_loss
            # Captured unconditionally (not gated behind early_stop_enabled)
            # -- matches this file's own autoencode/decoder-only phases,
            # which always track+restore best-val weights regardless of
            # whether their own early-stop patience is on. Leaving this
            # gated meant early_stop_patience=0 silently kept whatever the
            # LAST epoch happened to be instead of the best-val one, with
            # no cutoff to warn you epoch count and generalization had
            # diverged -- turning off the early CUTOFF shouldn't also turn
            # off restoring the best weights found along the way.
            best_state = copy.deepcopy(model.state_dict())
            if early_stop_enabled:
                patience_ctr = 0
            _save_phase_checkpoint("midtrain_latest")
        elif early_stop_enabled:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                log.info(f"Early stopping after {epoch + 1} epochs (patience={early_stop_patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info(f"Restored best-val weights (val_loss={best_val_loss:.4f})")
    _save_phase_checkpoint("after_training")

    # ------------------------------------------------------------------
    # Post-training diagnostics: median/worst val episode + a random
    # 10-example table (feeds the report's sample-predictions panel).
    # ------------------------------------------------------------------
    val_examples: list[dict] = []
    if len(val_idx) > 0:
        model.eval()
        with torch.no_grad():
            n_examples = min(10, len(val_idx))
            example_idx = rng.choice(val_idx, size=n_examples, replace=False)
            x = torch.from_numpy(ds.inputs[example_idx].astype(np.float32, copy=False)).to(device)
            y_all = ds.targets[example_idx]
            _, pred_heads = model(x)
            h_show = rng.integers(0, n_horizons)
            pred_row = pred_heads[h_show].cpu().numpy()
            for i, ep_idx in enumerate(example_idx):
                base = h_show * N_TARGET_FIELDS_PER_HORIZON
                pitch_length_m, pitch_width_m = _pitch_dims_m(ds.inputs[ep_idx], gen_params)
                val_examples.append({
                    "episode_idx": int(ep_idx),
                    "horizon_s": float(cfg["horizons_s"][h_show]),
                    "input": describe_input_row(ds.inputs[ep_idx], gen_params),
                    "pred": describe_target_row(pred_row[i], pitch_length_m, pitch_width_m, gen_params, logits=True),
                    "target": describe_target_row(y_all[i, base:base + N_TARGET_FIELDS_PER_HORIZON], pitch_length_m, pitch_width_m, gen_params),
                })

    artifact = {
        "encoder_state_dict": model.encoder.state_dict(),
        "config_snapshot": cfg,
        "normalization": normalization,
        "dataset_stats": {"n_episodes": len(ds), "n_train": len(train_idx), "n_val": len(val_idx)},
        "physics_config_hash": _physics_config_hash(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    log.info(f"Saved encoder checkpoint to {output_path}")

    history_arrays: dict = {
        "epoch": np.array([h["epoch"] for h in history]),
        "train_loss": np.array([h["train_loss"] for h in history]),
        "val_loss": np.array([h["val_loss"] for h in history]),
        "lr": np.array([h["lr"] for h in history]),
        "train_pair_loss": np.array([h["train_pair_loss"] for h in history]),
        "val_pair_loss": np.array([h["val_pair_loss"] for h in history]),
        "train_t0_loss": np.array([h["train_t0_loss"] for h in history]),
        "val_t0_loss": np.array([h["val_t0_loss"] for h in history]),
        "grad_norm_mean": np.array([h["grad_norm_mean"] for h in history]),
        "grad_norm_std": np.array([h["grad_norm_std"] for h in history]),
        "grad_norm_min": np.array([h["grad_norm_min"] for h in history]),
        "grad_norm_max": np.array([h["grad_norm_max"] for h in history]),
        "train_loss_delta_mean": np.array([h["train_loss_delta_mean"] for h in history]),
        "train_loss_delta_std": np.array([h["train_loss_delta_std"] for h in history]),
        "train_loss_delta_min": np.array([h["train_loss_delta_min"] for h in history]),
        "train_loss_delta_max": np.array([h["train_loss_delta_max"] for h in history]),
        "val_loss_delta": np.array([h["val_loss_delta"] for h in history]),
        "val_backprop_loss": np.array([h["val_backprop_loss"] for h in history]),
        "val_backprop_loss_delta": np.array([h["val_backprop_loss_delta"] for h in history]),
        "horizons_s": np.array(cfg["horizons_s"]),
        "pitch_half_diag_m": np.array(pitch_half_diag_m),
    }
    # Auxiliary latent-head metrics: one scalar per epoch each (unlike the
    # per-horizon _COMPONENTS below, which stack into (n_epochs, n_horizons)
    # arrays) -- these heads are not horizon-conditioned. NaN for a disabled
    # head, carried through as-is.
    for k in _AUX_METRIC_KEYS:
        for split in ("train", "val"):
            history_arrays[f"{split}_{k}"] = np.array([h[f"{split}_{k}"] for h in history])
    stacked_keys = [f"{split}_{c}" for c in _COMPONENTS for split in ("train", "val")]
    stacked_keys += [f"{split}_r2_{g}" for g in GROUPS for split in ("train", "val")]
    stacked_keys += [f"{split}_pctd_{g}" for g in GROUPS for split in ("train", "val")]
    stacked_keys += [f"{split}_{label}_{key}" for label in ("oob", "goal") for key in ("accuracy", "precision", "recall") for split in ("train", "val")]
    for key in stacked_keys:
        if history and key in history[0]:
            history_arrays[key] = np.stack([h[key] for h in history])

    panel_defs = [
        {"key": "pos_dist", "title": "Position error (mean distance)",
         "note": "mean ||pred_pos - target_pos|| in metres -- the literal \"how far off\" number, see LossBreakdown docstring.",
         "tickDigits": 2, "unitScale": pitch_half_diag_m, "unitLabel": "m", "unitDigits": 2},
        {"key": "vel_dist", "title": "Velocity error (mean distance)",
         "note": "mean ||pred_vel - target_vel|| in m/s.",
         "tickDigits": 3, "unitScale": pitch_half_diag_m, "unitLabel": "m/s", "unitDigits": 2},
        {"key": "heading_dist", "title": "Heading angular distance",
         "note": "angle (radians) between predicted and target heading unit vectors -- wraparound-safe, see LossBreakdown docstring.",
         "tickDigits": 3},
        {"key": "pos_rmse", "title": "Position RMSE", "note": "pos_x, pos_y -- right axis = metres.",
         "tickDigits": 2, "unitScale": pitch_half_diag_m, "unitLabel": "m", "unitDigits": 2},
        {"key": "vel_rmse", "title": "Velocity RMSE", "note": "vel_x, vel_y -- right axis = m/s.",
         "tickDigits": 3, "unitScale": pitch_half_diag_m, "unitLabel": "m/s", "unitDigits": 2},
        {"key": "heading_rmse", "title": "Heading RMSE (sin/cos)",
         "note": "RMSE on the (heading_sin, heading_cos) pair -- what's actually optimized.", "tickDigits": 3},
        {"key": "stamina_rmse", "title": "Stamina RMSE", "note": "current stamina fraction, in [0, 1].", "tickDigits": 3},
        {"key": "oob_bce", "title": "Out-of-bounds BCE", "note": "grey dashed = coin-flip baseline (ln 2 ~= 0.69)",
         "tickDigits": 2, "chanceLine": math.log(2)},
        {"key": "goal_bce", "title": "Goal-scored BCE (possession-gated)",
         "note": "grey dashed = coin-flip baseline (ln 2 ~= 0.69); only has_possession=1 rows contribute.",
         "tickDigits": 2, "chanceLine": math.log(2)},
        {"key": "oob_accuracy", "title": "Out-of-bounds accuracy", "note": "(tp+tn) / all -- threshold 0.5",
         "tickDigits": 2, "yRange": [0, 1]},
        {"key": "oob_precision", "title": "Out-of-bounds precision", "note": "tp / (tp+fp) -- threshold 0.5",
         "tickDigits": 2, "yRange": [0, 1]},
        {"key": "oob_recall", "title": "Out-of-bounds recall", "note": "tp / (tp+fn) -- threshold 0.5",
         "tickDigits": 2, "yRange": [0, 1]},
        {"key": "goal_accuracy", "title": "Goal-scored accuracy (possession-gated)", "note": "(tp+tn) / all",
         "tickDigits": 2, "yRange": [0, 1]},
        {"key": "goal_precision", "title": "Goal-scored precision (possession-gated)", "note": "tp / (tp+fp)",
         "tickDigits": 2, "yRange": [0, 1]},
        {"key": "goal_recall", "title": "Goal-scored recall (possession-gated)", "note": "tp / (tp+fn)",
         "tickDigits": 2, "yRange": [0, 1]},
        {"key": "r2_pos", "title": "Position R2", "note": "1 - MSE/Var(target) vs train-mean baseline",
         "tickDigits": 2, "yRange": [-0.2, 1], "chanceLine": 0},
        {"key": "r2_vel", "title": "Velocity R2", "note": "1 - MSE/Var(target) vs train-mean baseline",
         "tickDigits": 2, "yRange": [-0.2, 1], "chanceLine": 0},
        {"key": "r2_heading", "title": "Heading R2", "note": "1 - MSE/Var(target) vs train-mean baseline",
         "tickDigits": 2, "yRange": [-0.2, 1], "chanceLine": 0},
        {"key": "r2_stamina", "title": "Stamina R2", "note": "1 - MSE/Var(target) vs train-mean baseline",
         "tickDigits": 2, "yRange": [-0.2, 1], "chanceLine": 0},
        {"key": "pctd_pos", "title": "Position error (% of persistence)",
         "note": "100 x RMSE_model / RMSE(predict initial state) -- 100% = as bad as assuming nothing moved.",
         "tickDigits": 0, "chanceLine": 100},
        {"key": "pctd_vel", "title": "Velocity error (% of persistence)",
         "note": "100 x RMSE_model / RMSE(predict initial state).", "tickDigits": 0, "chanceLine": 100},
        {"key": "pctd_heading", "title": "Heading error (% of persistence)",
         "note": "100 x RMSE_model / RMSE(predict initial heading).", "tickDigits": 0, "chanceLine": 100},
        {"key": "pctd_stamina", "title": "Stamina error (% of persistence)",
         "note": "100 x RMSE_model / RMSE(predict initial stamina).", "tickDigits": 0, "chanceLine": 100},
    ]
    header_stat_defs = [
        {"label": "Position error", "key": "pos_dist", "unit": "m", "scaleKey": "pitch_half_diag_m"},
        {"label": "Velocity error", "key": "vel_dist", "unit": "m/s", "scaleKey": "pitch_half_diag_m"},
        {"label": "Heading error", "key": "heading_dist", "unit": "rad"},
    ]
    best_table_defs = [
        {"key": "pos_dist", "label": "Position error (mean distance)", "unit": "m", "scaleKey": "pitch_half_diag_m", "unitDigits": 2},
        {"key": "vel_dist", "label": "Velocity error (mean distance)", "unit": "m/s", "scaleKey": "pitch_half_diag_m", "unitDigits": 2},
        {"key": "pos_rmse", "label": "Position RMSE", "unit": "m", "scaleKey": "pitch_half_diag_m", "unitDigits": 2},
        {"key": "vel_rmse", "label": "Velocity RMSE", "unit": "m/s", "scaleKey": "pitch_half_diag_m", "unitDigits": 2},
        {"key": "heading_rmse", "label": "Heading RMSE (sin/cos)"},
        {"key": "heading_dist", "label": "Heading angular distance", "unit": "rad"},
        {"key": "stamina_rmse", "label": "Stamina RMSE"},
        {"key": "oob_bce", "label": "Out-of-bounds BCE"},
        {"key": "goal_bce", "label": "Goal-scored BCE (possession-gated)"},
        {"key": "oob_accuracy", "label": "Out-of-bounds accuracy"},
        {"key": "oob_precision", "label": "Out-of-bounds precision"},
        {"key": "oob_recall", "label": "Out-of-bounds recall"},
        {"key": "goal_accuracy", "label": "Goal-scored accuracy (possession-gated)"},
        {"key": "goal_precision", "label": "Goal-scored precision (possession-gated)"},
        {"key": "goal_recall", "label": "Goal-scored recall (possession-gated)"},
    ]

    from footballcoach.ai.physics_pretrain.report import open_in_browser, write_report
    report_path = write_report(
        history_arrays=history_arrays, dataset_stats=artifact["dataset_stats"],
        config_snapshot=cfg, normalization=normalization, val_examples=val_examples,
        output_path=output_path.with_suffix(".report.html"),
        panel_defs=panel_defs, title="Player Dynamics Training",
        header_stat_defs=header_stat_defs, best_table_defs=best_table_defs,
        config_namespace="physics_pretrain.player",
    )
    if open_browser:
        open_in_browser(report_path)

    artifact["history"] = history
    return artifact


def main() -> None:
    # Caps torch's own op-level thread pool -- otherwise it defaults to
    # using every available core for a single training process, which is
    # more than this workload benefits from and leaves nothing for the rest
    # of the machine while a long run is going. Same cap as train_ball_
    # dynamics.py.
    torch.set_num_threads(5)
    from footballcoach.ai.config import load_ai_config
    cfg = load_ai_config()["physics_pretrain"]["player"]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train the player-dynamics encoder.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--lr", type=float, default=cfg["lr"])
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pos-weight-max", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--open-report", dest="open_report", action="store_true", default=True)
    parser.add_argument("--no-open-report", dest="open_report", action="store_false")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument(
        "--reset-decoder-weights", action="store_true", default=False,
        help="After loading --init-checkpoint (if given), reinitialize every "
             "decoder-SIDE module (the shared per-horizon decoder + "
             "crossing_head/goal_dist_delta_head/short_horizon_head_0_2s/"
             "short_horizon_head_1_0s) to a fresh random init, leaving the "
             "encoder as-loaded. Also drops those parameters' Adam moment "
             "state so they're treated as genuinely fresh.",
    )
    parser.add_argument(
        "--reset-optimizer-state", action="store_true", default=False,
        help="After loading --init-checkpoint (if given), do NOT restore Adam's per-parameter moment state "
             "(exp_avg/exp_avg_sq/step) -- the optimizer starts fresh while model weights still resume "
             "normally. Use this whenever you've changed a loss weight (crossing_pos/crosses/dt_loss_weight "
             "etc.) since the checkpoint was saved: the restored exp_avg_sq (v) is an EMA of squared "
             "gradients with a ~1/(1-beta2) step memory (hundreds to 1000+ steps), so it stays calibrated to "
             "the OLD loss scale for a long time after a resume, artificially throttling Adam's effective "
             "step size far below what the current gradient would actually support.",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Randomly subsample the loaded dataset down to at most this many "
             "episodes before splitting train/val (seeded off --seed). Useful "
             "for testing whether the network can fit a small subset "
             "near-perfectly as a sanity check on capacity/optimization. "
             "Omit for the full dataset (default).",
    )
    parser.add_argument(
        "--linear-decoder", action="store_true", default=None,
        help="Override physics_pretrain.player.linear_decoder_enabled to true for this run: swaps the shared "
             "horizon-conditioned decoder for one independent Linear(latent_dim, 4) head per registered horizon "
             "(pos+vel only, no heading/stamina/oob/goal, no continuous-time interpolation -- see "
             "PlayerDynamicsLinearDecoder's docstring). Adjacent-pair combos whose delta doesn't land on a "
             "registered horizon are dropped (logged as kept/dropped). Omit to use whatever the config says "
             "(there is no CLI way to force it back off if the config has it on).",
    )
    parser.add_argument(
        "--isolate-crossing", action="store_true", default=False,
        help="Diagnostic mode: force main_loss_weight/bce_loss_weight/goal_dist_delta_loss_weight/"
             "short_horizon_probe_loss_weight to 0.0 for this run, so crossing_head is the ONLY objective "
             "pulling on the shared encoder (the horizon pass stays enabled -- crossing_head trains from its "
             "pseudo-starts too -- but its own t0/pair reconstruction terms are zeroed the same way). Tests "
             "whether crossing_head converges better without competing for encoder capacity against the other "
             "tasks. crossing_pos_loss_weight/crossing_dt_loss_weight are left as configured -- this isolates "
             "crossing from other tasks, it doesn't retune crossing itself.",
    )
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset, output_path=args.output, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, val_frac=args.val_frac, seed=args.seed, pos_weight_max=args.pos_weight_max,
        device=args.device, open_browser=args.open_report, init_checkpoint=args.init_checkpoint,
        reset_decoder_weights=args.reset_decoder_weights,
        reset_optimizer_state=args.reset_optimizer_state,
        max_episodes=args.max_episodes,
        linear_decoder=args.linear_decoder,
        isolate_crossing=args.isolate_crossing,
    )


if __name__ == "__main__":
    main()

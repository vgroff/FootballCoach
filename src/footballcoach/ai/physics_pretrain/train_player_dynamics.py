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
best-val epoch, and an auto-generated HTML report via ``report.py``.

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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from footballcoach.ai.physics_pretrain.player_dataset import GROUPS, PlayerDynamicsDataset
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
    pos_weight: torch.Tensor, bce_weight: float = 1.0,
) -> tuple[torch.Tensor, LossBreakdown]:
    """Sum over horizons of (continuous MSE + event BCE), directly mirroring
    ``train_ball_dynamics.compute_loss`` -- see its docstring. ``pos_weight``:
    ``(n_horizons, 2)`` tensor from ``PlayerDynamicsDataset.compute_pos_
    weights()``. ``goal_scored``'s BCE is possession-masked -- see
    ``_goal_bce_masked``.
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
        total = total + pos_mse + vel_mse + heading_mse + stamina_mse + bce_weight * (oob_bce + goal_bce)
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
    pos_weight: torch.Tensor, bce_weight: float = 1.0,
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
        total = total + pos_sq + vel_sq + heading_sq + stamina_sq + bce_weight * (oob_bce + goal_bce)
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
) -> tuple[torch.Tensor, dict[str, float]]:
    """Same components as one horizon's worth of ``compute_loss``, for a
    single ``(pred, target)`` pair -- shared by the adjacent-pair and
    autoencode-pretrain passes, see ``train_ball_dynamics._single_target_
    loss_with_breakdown``."""
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
    total = pos_mse + vel_mse + heading_mse + stamina_mse + bce_weight * (oob_bce + goal_bce)
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
    return pos_sq + vel_sq + heading_sq + stamina_sq + bce_weight * (oob_bce + goal_bce)


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
) -> tuple[torch.Tensor, float, float]:
    """``model.crossing_head``'s loss for one batch: position MSE (x/y,
    MASKED to rows where ``mask_all`` is True -- an episode with no crossing
    AHEAD of whichever pseudo-start these targets were built relative to, or
    one excluded for starting already out of bounds, has no meaningful
    crossing position to regress toward) plus delta_t MSE (UNMASKED, against
    ``dt_all``'s -1 sentinel for those same rows -- see
    ``PlayerDynamicsDataset``'s docstring for why delta_t is trained on the
    sentinel directly rather than also being masked).

    Near-verbatim port of ``train_ball_dynamics._crossing_head_loss``, minus
    its height-exclusion note (a player has no height axis to exclude).
    ``pos_all``/``dt_all``/``mask_all`` are plain arrays (not tied to a
    specific dataset attribute) so this same function serves BOTH the main
    task's t=0 crossing target (``ds.crossing_pos``/``crossing_dt``/
    ``crossing_mask``, indexed by ``row_idx`` = original dataset row indices)
    AND the per-horizon pseudo-start generalization (see
    ``_build_horizon_bundle`` -- ``crossing_time - horizons_s[h]``, reindexed
    to whichever row order the caller's arrays use). Callers must ensure
    ``pos_all``/``dt_all``/``mask_all`` and ``row_idx`` share the same
    indexing convention; this function doesn't care which one it is.

    Returns ``(loss, pos_dist_mean, dt_mae_mean)`` -- the last two are plain
    floats (not backpropagated) for per-epoch reporting: mean Euclidean
    crossing-position error over the rows that actually crossed (0.0 if none
    did, matching the masked loss's own 0-numerator/1-denominator
    convention), and mean absolute delta_t error over ALL rows (sentinel
    included, since that's what the loss itself trains against).
    """
    crossing_pred = model.crossing_head(latent)
    c_pos = torch.from_numpy(pos_all[row_idx]).to(device)
    c_dt = torch.from_numpy(dt_all[row_idx]).to(device)
    c_mask = torch.from_numpy(mask_all[row_idx]).to(device)
    mask_f = c_mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    pos_err = crossing_pred[:, 0:2] - c_pos
    pos_loss = (pos_err.pow(2).sum(dim=-1) * mask_f).sum() / denom
    dt_loss = F.mse_loss(crossing_pred[:, 2], c_dt)
    loss = pos_loss + dt_loss
    with torch.no_grad():
        pos_dist_mean = float((torch.linalg.norm(pos_err, dim=-1) * mask_f).sum().item() / float(denom.item()))
        dt_mae_mean = float((crossing_pred[:, 2] - c_dt).abs().mean().item())
    return loss, pos_dist_mean, dt_mae_mean


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
    "crossing_loss", "crossing_pos_dist", "crossing_dt_mae",
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

    def add_crossing(self, loss: torch.Tensor, pos_dist: float, dt_mae: float) -> None:
        self._values["crossing_loss"].append(float(loss.item()))
        self._values["crossing_pos_dist"].append(pos_dist)
        self._values["crossing_dt_mae"].append(dt_mae)

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
    epoch) -- identical logic here."""
    batches: list[tuple[int, np.ndarray]] = []
    for h_idx, (h_inputs, _h_targets) in enumerate(data):
        order = rng.permutation(len(h_inputs))
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            if len(chunk) > 0:
                batches.append((h_idx, chunk))
    shuffle_order = rng.permutation(len(batches))
    return [batches[i] for i in shuffle_order]


def _build_horizon_bundle(
    ds: PlayerDynamicsDataset, indices: np.ndarray, n_horizons: int, horizons_s: list[float],
    pair_enabled: bool, pair_max_skip: int, pair_min_start_speed_norm: float,
    has_crossing_data: bool,
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
    predict it from, so callers pass ``ds.crossing_pos`` unchanged).
    Physics isn't latched here either (a player who goes out of bounds keeps
    walking and can come back in), so ``crossing_time < horizons_s[h]`` is
    possible even when horizon h's own flags read false; that yields a
    negative delta_t, which isn't meaningful ("time until a FUTURE crossing"
    can't be negative) and is folded into the same -1 sentinel / invalid
    treatment as an episode that never crossed at all (or was excluded for
    starting out of bounds -- see ``player_episode_gen.generate_episode``).

    Everything returned is indexed in the SAME row order as ``indices``
    itself (matching ``autoencode_train_data[h]``'s convention: position i
    corresponds to original dataset row ``indices[i]``), so one local
    ``row_idx`` indexes all of them.

    Returns a dict with per-horizon (length ``n_horizons``) lists:
    ``pair_mask``, ``pair_targets``, ``crossing_dt``, ``crossing_valid``
    (the last two hold ``None`` entries when ``has_crossing_data`` is False).
    """
    n = len(indices)
    pair_mask: list[np.ndarray] = []
    pair_targets: list[list[tuple[int, np.ndarray]]] = []
    crossing_dt: list[np.ndarray | None] = []
    crossing_valid: list[np.ndarray | None] = []

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
            adj_dt = crossing_times_here - horizons_s[h]
            adj_valid = np.isfinite(crossing_times_here) & (adj_dt >= 0)
            crossing_dt.append(np.where(adj_valid, adj_dt, -1.0).astype(np.float32))
            crossing_valid.append(adj_valid)
        else:
            crossing_dt.append(None)
            crossing_valid.append(None)

    return {
        "pair_mask": pair_mask, "pair_targets": pair_targets,
        "crossing_dt": crossing_dt, "crossing_valid": crossing_valid,
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
) -> dict:
    from footballcoach.ai.config import load_ai_config
    cfg = load_ai_config()["physics_pretrain"]["player"]
    output_path = Path(output_path)

    ds = PlayerDynamicsDataset.from_directory(dataset_dir)
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
    crossing_weight = float(cfg.get("crossing_loss_weight", 0.0))
    # A dataset built by hand (e.g. in a unit test) may carry no
    # crossings/crossing_times at all -- guard every crossing call site with
    # this rather than crashing, same convention as train_ball_dynamics.py.
    has_crossing_data = ds.crossing_pos is not None and crossing_weight != 0.0
    if crossing_weight != 0.0 and ds.crossing_pos is None:
        log.warning("crossing_loss_weight != 0 but this dataset carries no crossings/crossing_times -- crossing_head will not be trained.")

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

    model = PlayerDynamicsAutoencoder.from_config().to(device)
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
    optimizer_type = str(cfg.get("optimizer_type", "adam")).lower()
    if optimizer_type == "sgd":
        sgd_momentum = float(cfg.get("sgd_momentum", 0.9))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=sgd_momentum, weight_decay=weight_decay)
        log.info(f"Main-loop optimizer: SGD (momentum={sgd_momentum}, lr={lr:.2e}, weight_decay={weight_decay})")
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown physics_pretrain.player.optimizer_type: {optimizer_type!r} (expected 'adam' or 'sgd')")
    bce_weight = float(cfg.get("bce_loss_weight", 1.0))
    _LOG_COMPONENTS = tuple(c for c in _COMPONENTS if not (c in ("oob_bce", "goal_bce") and bce_weight == 0.0))

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
        widen_needed = old_cfg is not None and any(
            old_cfg.get(k, 32) != cfg.get(k, 32)
            for k in ("hidden_dim", "encoder_bottleneck_dim", "latent_dim", "decoder_hidden_dim")
        )
        if widen_needed:
            from footballcoach.ai.physics_pretrain.widen_player_checkpoint import _build_model, _validate_widen_cfgs, widen_model_
            _validate_widen_cfgs(old_cfg, cfg)
            old_model = _build_model(old_cfg).to(device)
            if "model_state_dict" in ckpt:
                old_model.load_state_dict(ckpt["model_state_dict"], strict=False)
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
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if missing:
                log.info(f"Checkpoint missing {len(missing)} param(s) not present when it was saved (left at fresh init): {missing}")
            if unexpected:
                log.info(f"Checkpoint had {len(unexpected)} unexpected param(s), ignored: {unexpected}")
            log.info(f"Resumed full model (encoder+decoder) from {init_checkpoint} (phase={ckpt.get('phase', '?')})")
        else:
            missing, unexpected = model.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
            if missing:
                log.info(f"Checkpoint missing {len(missing)} encoder param(s) not present when it was saved (left at fresh init): {missing}")
            if unexpected:
                log.info(f"Checkpoint had {len(unexpected)} unexpected encoder param(s), ignored: {unexpected}")
            log.info(f"Resumed encoder only from {init_checkpoint} (decoder left at fresh init)")

    normalization = {"pitch_half_diag_m": pitch_half_diag_m}

    def _save_phase_checkpoint(phase: str) -> None:
        path = output_path.with_suffix(f".{phase}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
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

    horizon_bundle_train: dict = {}
    horizon_bundle_val: dict = {}
    if horizon_pass_enabled:
        horizon_bundle_train = _build_horizon_bundle(
            ds, train_idx, n_horizons, cfg["horizons_s"],
            adjacent_pair_enabled, adjacent_pair_max_skip, pair_min_start_speed_norm, has_crossing_data,
        )
        if len(val_idx) > 0:
            horizon_bundle_val = _build_horizon_bundle(
                ds, val_idx, n_horizons, cfg["horizons_s"],
                adjacent_pair_enabled, adjacent_pair_max_skip, pair_min_start_speed_norm, has_crossing_data,
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
        horizon_lines = (
            f"    autoencode/t0 (bottleneck recon): {n_autoencode_train:,} rows -- own batches "
            f"({len(train_idx):,} rows x {n_horizons} horizons)\n"
            f"    adjacent-pair (dynamics)        : {n_pair_eligible:,}/{n_autoencode_train:,} horizon-pass rows mask-eligible -- "
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
            autoencode_optimizer = torch.optim.Adam(model.parameters(), lr=autoencode_lr)
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
                        loss, breakdown = _single_target_loss_with_breakdown(pred, y, x, pos_weight[h_idx], bce_weight)
                        losses.append(float(loss.item()))
                        breakdowns_by_h[h_idx].append(breakdown)
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            return mean_loss, _mean_breakdown_by_horizon(breakdowns_by_h)

        best_val_loss_ae, best_state_ae = float("inf"), None
        for ae_epoch in range(autoencode_pretrain_epochs):
            model.train()
            train_losses_ae, train_breakdowns_by_h = [], [[] for _ in range(n_horizons)]
            for h_idx, row_idx in _interleaved_horizon_batches(autoencode_train_data, batch_size, rng):
                ae_inputs, ae_targets = autoencode_train_data[h_idx]
                x = torch.from_numpy(ae_inputs[row_idx].astype(np.float32, copy=False)).to(device)
                y = torch.from_numpy(ae_targets[row_idx].astype(np.float32, copy=False)).to(device)
                latent = model.encoder(x)
                pred = model.decoder.forward_at(latent, 0.0)
                loss, breakdown = _single_target_loss_with_breakdown(pred, y, x, pos_weight[h_idx], bce_weight)
                autoencode_optimizer.zero_grad()
                loss.backward()
                autoencode_optimizer.step()
                train_losses_ae.append(float(loss.item()))
                train_breakdowns_by_h[h_idx].append(breakdown)
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
    # goal-dist-delta/short-horizon-probe heads on "main" rows), built as
    # one combined/shuffled loop from day one (see agent_plans/
    # ball_physics_pretrain_plan.md §12.6's "interleave from the start"
    # lesson). Takes `optimizer` as a parameter (not closed over) so BOTH
    # the main loop below AND decoder-only-pretrain above can share this
    # exact same loop body against their own separate optimizer -- mirrors
    # train_ball_dynamics.py's identical parameterization, which is what
    # lets decoder-only-pretrain there train crossing_head/resting_head too
    # instead of leaving them untouched for that whole phase.
    # ------------------------------------------------------------------
    def _run_interleaved_train_epoch(optimizer: torch.optim.Optimizer) -> dict:
        model.train()
        chunks: list[tuple[str, int, np.ndarray]] = []
        for h_idx in range(n_horizons):
            order = rng.permutation(len(train_idx))
            for start in range(0, len(order), batch_size):
                c = order[start:start + batch_size]
                if len(c) > 0:
                    chunks.append(("main", h_idx, train_idx[c]))
        if horizon_pass_enabled:
            for h_idx, row_idx in _interleaved_horizon_batches(autoencode_train_data, batch_size, rng):
                chunks.append(("horizon", h_idx, row_idx))
        shuffle_order = rng.permutation(len(chunks))
        chunks = [chunks[i] for i in shuffle_order]

        losses, breakdowns_by_h = [], [[] for _ in range(n_horizons)]
        pair_losses, t0_losses, t0_breakdowns_by_h = [], [], [[] for _ in range(n_horizons)]
        oob_counts = np.zeros((n_horizons, 4), dtype=np.int64)
        goal_counts = np.zeros((n_horizons, 4), dtype=np.int64)
        sq_err = {g: np.zeros(n_horizons) for g in GROUPS}
        n_elem = {g: np.zeros(n_horizons, dtype=np.int64) for g in GROUPS}
        train_grad_norms: list[float] = []
        aux = _AuxAccumulator()

        for kind, h_idx, row_idx in chunks:
            if kind == "main":
                x = torch.from_numpy(ds.inputs[row_idx].astype(np.float32, copy=False)).to(device)
                y = torch.from_numpy(ds.targets[row_idx].astype(np.float32, copy=False)).to(device)
                latent, pred_heads = model(x)
                loss, breakdown = compute_loss(pred_heads, y, x, pos_weight, bce_weight)
                # `backprop_loss` (not `loss`) is what actually gets stepped
                # -- it folds in every auxiliary latent head's term for a
                # single combined backward pass (cheaper than separate steps,
                # and the encoder/latent is shared anyway). `losses` below
                # keeps reporting the main per-horizon `loss` ALONE, same
                # convention as pair/t0 and as train_ball_dynamics.py:
                # otherwise train_loss would carry the aux terms while
                # val_loss (computed independently) doesn't, making the two
                # look wildly divergent for reasons unrelated to
                # over/underfitting.
                backprop_loss = loss
                if has_crossing_data:
                    crossing_loss, c_pos_dist, c_dt_mae = _crossing_head_loss(
                        model, latent, ds.crossing_pos, ds.crossing_dt, ds.crossing_mask, row_idx, device,
                    )
                    backprop_loss = backprop_loss + crossing_weight * crossing_loss
                    aux.add_crossing(crossing_loss, c_pos_dist, c_dt_mae)
                if goal_dist_delta_targets is not None:
                    gdd_loss, gdd_mae_left, gdd_mae_right = _goal_dist_delta_head_loss(
                        model, latent, goal_dist_delta_targets, row_idx, device,
                    )
                    backprop_loss = backprop_loss + goal_dist_delta_weight * gdd_loss
                    aux.add_goal_dist(gdd_loss, gdd_mae_left, gdd_mae_right)
                if probe_targets_0_2s is not None:
                    probe_loss, probe_rmse_0_2, probe_rmse_1_0 = _short_horizon_probe_loss(
                        model, latent, probe_targets_0_2s, probe_targets_1_0s, row_idx, device,
                    )
                    backprop_loss = backprop_loss + short_horizon_probe_weight * probe_loss
                    aux.add_probe(probe_loss, probe_rmse_0_2, probe_rmse_1_0)
                optimizer.zero_grad()
                backprop_loss.backward()
                # Total gradient norm across every trainable param, BEFORE
                # optimizer.step() consumes it -- a direct read of how
                # large this step's raw update direction is, independent
                # of `lr`. `clip_grad_norm_` with max_norm=inf computes
                # (and returns) the norm without ever actually clipping/
                # rescaling anything -- the standard portable way to just
                # measure it. Diagnostic only -- see train_ball_dynamics.py's
                # identical measurement for the full rationale.
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf")).item())
                train_grad_norms.append(grad_norm)
                optimizer.step()
                losses.append(float(loss.item()))
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
            else:  # "horizon" -- ONE encoder pass for this recorded
                # horizon's shared pseudo-input, feeding whichever of {t0
                # reconstruction, every pair skip (masked)} are enabled,
                # both summed into ONE gradient step -- see
                # _build_horizon_bundle's docstring for how the pair
                # targets were derived, and why pair's old row-filter had
                # to become a mask so t0/pair can share the exact same row
                # set. Mirrors train_ball_dynamics.py's "horizon" branch,
                # minus the crossing/resting terms (no equivalents here).
                ae_inputs, ae_targets = autoencode_train_data[h_idx]
                x = torch.from_numpy(ae_inputs[row_idx].astype(np.float32, copy=False)).to(device)
                latent = model.encoder(x)
                combined_loss = x.new_zeros(())

                if autoencode_during_main_loop_enabled:
                    y = torch.from_numpy(ae_targets[row_idx].astype(np.float32, copy=False)).to(device)
                    pred = model.decoder.forward_at(latent, 0.0)
                    t0_loss, breakdown = _single_target_loss_with_breakdown(pred, y, x, pos_weight[h_idx], bce_weight)
                    combined_loss = combined_loss + t0_loss
                    t0_losses.append(float(t0_loss.item()))
                    t0_breakdowns_by_h[h_idx].append(breakdown)

                pair_targets_h = horizon_bundle_train["pair_targets"][h_idx] if adjacent_pair_enabled else []
                if pair_targets_h:
                    mask_f = torch.from_numpy(horizon_bundle_train["pair_mask"][h_idx][row_idx]).to(device).float()
                    denom = mask_f.sum().clamp_min(1.0)
                    pair_loss = x.new_zeros(())
                    for skip, p_targets in pair_targets_h:
                        y_pair = torch.from_numpy(p_targets[row_idx]).to(device)
                        gap = float(cfg["horizons_s"][h_idx + skip] - cfg["horizons_s"][h_idx])
                        pred_pair = model.decoder.forward_at(latent, gap)
                        per_ex_loss = _single_target_per_episode_loss(pred_pair, y_pair, x, pos_weight[h_idx + skip], bce_weight)
                        pair_loss = pair_loss + (per_ex_loss * mask_f).sum() / denom
                    combined_loss = combined_loss + pair_loss
                    pair_losses.append(float(pair_loss.item()))

                if has_crossing_data:
                    # Horizon-generalized crossing: same head, same fixed
                    # (x, y) target, delta_t re-based on this pseudo-start
                    # (see _build_horizon_bundle). goal_dist_delta/the
                    # short-horizon probes deliberately have NO horizon-pass
                    # counterpart -- see their loss docstrings.
                    crossing_loss_h, c_pos_dist_h, c_dt_mae_h = _crossing_head_loss(
                        model, latent, crossing_pos_train, horizon_bundle_train["crossing_dt"][h_idx],
                        horizon_bundle_train["crossing_valid"][h_idx], row_idx, device,
                    )
                    combined_loss = combined_loss + crossing_weight * crossing_loss_h
                    aux.add_crossing(crossing_loss_h, c_pos_dist_h, c_dt_mae_h)

                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

        return {
            "aux": aux.summary(),
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
                loss, breakdown = compute_loss(pred_heads, y, x, pos_weight, bce_weight)
                losses.append(float(loss.item()))
                if has_crossing_data:
                    aux.add_crossing(*_crossing_head_loss(
                        model, latent, ds.crossing_pos, ds.crossing_dt, ds.crossing_mask, batch_idx, device,
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
                            t0_loss = _single_target_loss_with_breakdown(pred, y, x, pos_weight[h_idx], bce_weight)[0]
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
                                per_ex_loss = _single_target_per_episode_loss(pred_pair, y_pair, x, pos_weight[h_idx + skip], bce_weight)
                                pair_loss = pair_loss + (per_ex_loss * mask_f).sum() / denom
                            pair_losses.append(float(pair_loss.item()))

                        if has_crossing_data:
                            aux.add_crossing(*_crossing_head_loss(
                                model, latent, crossing_pos_val, horizon_bundle_val["crossing_dt"][h_idx],
                                horizon_bundle_val["crossing_valid"][h_idx], row_idx, device,
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
                f"dt_mae={train_aux['crossing_dt_mae']:.3f}s | "
                f"val loss={val_aux['crossing_loss']:.4f} "
                f"pos_dist={val_aux['crossing_pos_dist'] * pitch_half_diag_m:.3f}m "
                f"dt_mae={val_aux['crossing_dt_mae']:.3f}s"
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
            decoder_only_optimizer = torch.optim.Adam(decoder_only_params, lr=decoder_only_lr)
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
            train_res_do = _run_interleaved_train_epoch(decoder_only_optimizer)
            val_res_do = _run_eval_pass(val_idx) if len(val_idx) > 0 else train_res_do
            mean_train_loss_do = train_res_do["mean_loss"]
            mean_val_loss_do = val_res_do["mean_loss"] if len(val_idx) > 0 else float("inf")
            log.info(f"  decoder-only pretrain epoch {do_epoch + 1}/{decoder_only_pretrain_epochs}: train_loss={mean_train_loss_do:.4f}  val_loss={mean_val_loss_do:.4f}")
            _log_aux_diagnostics(train_res_do["aux"], val_res_do["aux"])

            if mean_val_loss_do < best_val_loss_do - do_early_stop_min_delta:
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

    for epoch in range(epochs):
        train_res = _run_interleaved_train_epoch(optimizer)
        val_res = _run_eval_pass(val_idx) if len(val_idx) > 0 else train_res
        n_epochs_run += 1

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
        improved = val_loss < (best_val_loss - early_stop_min_delta)
        val_line = f"  val_loss={val_loss:.4f}  best={min(best_val_loss, val_loss):.4f}"
        if early_stop_enabled:
            val_line += "  (improved)" if improved else f"  (patience {patience_ctr + 1}/{early_stop_patience})"
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
        train_aux, val_aux = train_res["aux"], val_res["aux"]
        _log_aux_diagnostics(train_aux, val_aux)
        val_loss_delta = float("nan")
        if len(val_idx) > 0:
            # Epoch-over-epoch change in val_loss (negative = improved) --
            # nan on the first epoch (no prior value yet). See
            # train_ball_dynamics.py's identical diagnostic.
            val_loss_delta = val_loss - prev_val_loss if not np.isnan(prev_val_loss) else float("nan")
            prev_val_loss = val_loss
            log.info(f"    val_loss_delta (epoch-over-epoch): {val_loss_delta:.6f}")
        for c in _LOG_COMPONENTS:
            _log_component("    train", c, train_res["breakdowns"][c])
            _log_component("    val  ", c, val_res["breakdowns"][c])
        for g in GROUPS:
            log.info(f"    train {g:9s} R2 by horizon: {np.array2string(train_r2[g], precision=3)}, mean: {_safe_nanmean(train_r2[g]):.3f}")
            log.info(f"    val   {g:9s} R2 by horizon: {np.array2string(val_r2[g], precision=3)}, mean: {_safe_nanmean(val_r2[g]):.3f}")
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
                         "val_loss_delta": val_loss_delta}
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
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset, output_path=args.output, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, val_frac=args.val_frac, seed=args.seed, pos_weight_max=args.pos_weight_max,
        device=args.device, open_browser=args.open_report, init_checkpoint=args.init_checkpoint,
    )


if __name__ == "__main__":
    main()

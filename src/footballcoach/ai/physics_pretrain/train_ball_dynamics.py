"""CLI: train the ball-dynamics encoder from a generated .npz dataset.

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

See agent_plans/ball_physics_pretrain_plan.md section 7.

Usage::

    uv run python -m footballcoach.ai.physics_pretrain.train_ball_dynamics \\
        --dataset physics_pretrain_data/ball/ \\
        --output checkpoints/physics_pretrain/ball_encoder.pt \\
        --epochs 50 --batch-size 1024

To generate the dataset first, see ``ball_dataset.py``'s own ``__main__``
entry point (``python -m footballcoach.ai.physics_pretrain.ball_dataset
--help``).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from footballcoach.ai.physics_pretrain.ball_dataset import BallDynamicsDataset
from footballcoach.ai.physics_pretrain.ball_dynamics_net import N_IDENTITY_SHORTCUT_FIELDS, BallDynamicsAutoencoder
from footballcoach.ai.physics_pretrain.ball_episode_gen import BALL_SPIN_NORM_DIVISOR_RAD_S, N_TARGET_FIELDS_PER_HORIZON

log = logging.getLogger("footballcoach.ai.physics_pretrain.train_ball_dynamics")


class LossBreakdown:
    """Per-horizon loss components, each a length-``n_horizons`` list of floats.

    ``pos_rmse``/``vel_rmse``/``spin_rmse`` are reported SEPARATELY (not one
    blended "continuous MSE") because they're different physical quantities
    on different normalized scales (position vs velocity vs spin) -- a
    single combined number can look flat while one of the three is actually
    moving a lot and another is stuck, exactly the same masking problem as
    merging out_of_bounds/goal_scored BCE (see ``oob_bce``/``goal_bce``
    below). Reported as RMSE (sqrt of the underlying MSE), not raw MSE --
    RMSE is in the same (normalized) units as the quantity itself, so it's
    directly interpretable and directly convertible to real-world units
    (metres, m/s, rad/s) by multiplying by the relevant normalization scale,
    unlike squared-unit MSE. The actual optimized loss (``total`` returned
    by ``compute_loss``) still sums plain MSE terms -- only the REPORTED
    breakdown values are square-rooted, so this changes nothing about what
    gets backpropagated.

    ``pos_dist`` is a SEPARATE, purely-reporting-only metric alongside
    ``pos_rmse`` -- the mean per-sample Euclidean distance ``||pred_pos -
    target_pos||`` (mean AFTER the sqrt, per sample). NOTE this is NOT
    simply "smaller than pos_rmse": ``pos_rmse`` is sqrt of ``F.mse_loss``,
    which averages the squared error over the 3 position axes AND the
    batch together -- dividing by 3 axes makes it closer to a "typical
    per-AXIS error," not the RMS of the 3D vector distance itself. The true
    RMS of the 3D distance is ``sqrt(3) * pos_rmse``, and THAT is what
    Jensen's inequality bounds ``pos_dist`` against (``pos_dist <= sqrt(3)
    * pos_rmse``), not ``pos_rmse`` directly -- so ``pos_dist`` can
    legitimately come out larger than ``pos_rmse`` (by close to sqrt(3) in
    the low-variance case), even though it's still the more literal answer
    to "how far off is the model typically" than the per-axis pos_rmse is.
    Never contributes to the backpropagated loss, same convention as the
    rest of this class.

    ``vel_dist`` is the same idea as ``pos_dist`` but for velocity: mean
    per-sample speed-error magnitude ``||pred_vel - target_vel||``, vs.
    ``vel_rmse``'s per-axis RMS -- same ``vel_dist <= sqrt(3) * vel_rmse``
    relationship, same "never backpropagated" convention.
    """
    __slots__ = ("pos_rmse", "pos_dist", "vel_rmse", "vel_dist", "spin_rmse", "oob_bce", "goal_bce")

    def __init__(self):
        self.pos_rmse: list[float] = []
        self.pos_dist: list[float] = []
        self.vel_rmse: list[float] = []
        self.vel_dist: list[float] = []
        self.spin_rmse: list[float] = []
        self.oob_bce: list[float] = []
        self.goal_bce: list[float] = []


def compute_loss(
    pred_heads: list[torch.Tensor], target: torch.Tensor, pos_weight: torch.Tensor, bce_weight: float = 1.0,
    spin_weight: float = 1.0,
) -> tuple[torch.Tensor, LossBreakdown]:
    """Sum over horizons of (continuous MSE + event BCE), per §5.

    ``pos_weight``: ``(n_horizons, 2)`` tensor from
    ``BallDynamicsDataset.compute_pos_weights()``.

    ``bce_weight`` (default 1.0, no behaviour change): multiplies the
    oob/goal BCE terms' contribution to ``total`` (the backpropagated
    loss) ONLY -- ``breakdown.oob_bce``/``goal_bce`` always report the raw,
    UNweighted BCE value, so the reported metric stays comparable
    regardless of this weight. 0.0 fully disables the classification
    heads' gradient (pos/vel/spin regression trains exactly as if they
    didn't exist) while still computing/logging them for comparison --
    see ``physics_pretrain.ball.bce_loss_weight``.

    ``spin_weight`` (default 1.0, no behaviour change): same idea but for
    the spin MSE term -- ``breakdown.spin_rmse`` always reports the raw,
    UNweighted RMSE. 0.0 fully disables the spin head's gradient -- see
    ``physics_pretrain.ball.spin_loss_weight``.

    Returns ``(total, breakdown)`` where ``total`` is the plain-MSE+BCE sum
    actually backpropagated, and ``breakdown`` is a ``LossBreakdown`` with 5
    SEPARATE per-horizon REPORTING components (position RMSE, velocity
    RMSE, spin RMSE, out_of_bounds BCE, goal_scored BCE) -- never merged
    into fewer numbers, since each hides a different quantity's own
    learning progress (see ``LossBreakdown``'s docstring).
    """
    total = target.new_zeros(())
    breakdown = LossBreakdown()
    for h, head_out in enumerate(pred_heads):
        base = h * N_TARGET_FIELDS_PER_HORIZON
        target_h = target[:, base:base + N_TARGET_FIELDS_PER_HORIZON]
        pos_mse = F.mse_loss(head_out[:, 0:3], target_h[:, 0:3])
        pos_dist = torch.linalg.norm(head_out[:, 0:3] - target_h[:, 0:3], dim=-1).mean()
        vel_mse = F.mse_loss(head_out[:, 3:6], target_h[:, 3:6])
        vel_dist = torch.linalg.norm(head_out[:, 3:6] - target_h[:, 3:6], dim=-1).mean()
        spin_mse = F.mse_loss(head_out[:, 6:9], target_h[:, 6:9])
        oob_bce = F.binary_cross_entropy_with_logits(
            head_out[:, 9], target_h[:, 9], pos_weight=pos_weight[h, 0],
        )
        goal_bce = F.binary_cross_entropy_with_logits(
            head_out[:, 10], target_h[:, 10], pos_weight=pos_weight[h, 1],
        )
        total = total + pos_mse + vel_mse + spin_weight * spin_mse + bce_weight * (oob_bce + goal_bce)
        breakdown.pos_rmse.append(float(pos_mse.item()) ** 0.5)
        breakdown.pos_dist.append(float(pos_dist.item()))
        breakdown.vel_rmse.append(float(vel_mse.item()) ** 0.5)
        breakdown.vel_dist.append(float(vel_dist.item()))
        breakdown.spin_rmse.append(float(spin_mse.item()) ** 0.5)
        breakdown.oob_bce.append(float(oob_bce.item()))
        breakdown.goal_bce.append(float(goal_bce.item()))
    return total, breakdown


def compute_per_episode_loss(
    pred_heads: list[torch.Tensor], target: torch.Tensor, pos_weight: torch.Tensor, bce_weight: float = 1.0,
    spin_weight: float = 1.0,
) -> torch.Tensor:
    """Same 5 components as ``compute_loss``'s ``total`` (summed across all
    horizons), but NOT reduced across the batch -- returns one loss value
    per row, so individual episodes can be identified afterward (e.g. the
    median/worst-by-loss val episode, see ``train()``'s post-training
    diagnostic). Averaging this over the batch dim recovers the same value
    as ``compute_loss``'s ``total`` for that batch. ``bce_weight``/
    ``spin_weight``: see ``compute_loss``'s docstring.
    """
    batch = target.shape[0]
    total = target.new_zeros(batch)
    for h, head_out in enumerate(pred_heads):
        base = h * N_TARGET_FIELDS_PER_HORIZON
        target_h = target[:, base:base + N_TARGET_FIELDS_PER_HORIZON]
        pos_sq = (head_out[:, 0:3] - target_h[:, 0:3]).pow(2).mean(dim=1)
        vel_sq = (head_out[:, 3:6] - target_h[:, 3:6]).pow(2).mean(dim=1)
        spin_sq = (head_out[:, 6:9] - target_h[:, 6:9]).pow(2).mean(dim=1)
        oob_bce = F.binary_cross_entropy_with_logits(
            head_out[:, 9], target_h[:, 9], pos_weight=pos_weight[h, 0], reduction="none",
        )
        goal_bce = F.binary_cross_entropy_with_logits(
            head_out[:, 10], target_h[:, 10], pos_weight=pos_weight[h, 1], reduction="none",
        )
        total = total + pos_sq + vel_sq + spin_weight * spin_sq + bce_weight * (oob_bce + goal_bce)
    return total


def _crossing_head_loss(
    model: BallDynamicsAutoencoder, latent: torch.Tensor, pos_all: np.ndarray, dt_all: np.ndarray,
    mask_all: np.ndarray, row_idx: np.ndarray, device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    """``model.crossing_head``'s loss for one batch: position MSE (summed
    over x/y ONLY -- height is deliberately excluded, MASKED to rows where
    ``mask_all`` is True -- an episode with no crossing AHEAD of whichever
    pseudo-start these targets were built relative to has no meaningful
    crossing position to regress toward) plus delta_t MSE (UNMASKED, against
    ``dt_all``'s -1 sentinel for those same rows -- see ``BallDynamicsDataset``'s
    docstring for why delta_t is trained on the sentinel directly rather
    than also being masked).

    ``pos_all``/``dt_all``/``mask_all`` are plain arrays (not tied to a
    specific dataset attribute) so this same function serves BOTH the main
    task's t=0 crossing target (``ds.crossing_pos``/``crossing_dt``/
    ``crossing_mask``, indexed by ``row_idx=batch_idx`` -- original dataset
    row indices) AND the per-horizon pseudo-start generalization (see
    ``_horizon_pass``'s docstring -- ``crossing_time - horizons_s[h]``,
    reindexed to whichever row order the caller's arrays use). Callers must
    ensure ``pos_all``/``dt_all``/``mask_all`` and ``row_idx`` share the same
    indexing convention; this function doesn't care which one it is.

    Returns ``(loss, pos_dist_mean, dt_mae_mean)`` -- the last two are
    plain floats (not backpropagated) for per-epoch reporting: mean
    Euclidean crossing-position error (x/y only) over the rows that actually
    crossed (0.0 if none did, matching the masked loss's own 0-numerator/
    1-denominator convention), and mean absolute delta_t error over ALL rows
    (sentinel included, since that's what the loss itself trains against).
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


def _resting_head_loss(
    model: BallDynamicsAutoencoder, latent: torch.Tensor, resting_pos: np.ndarray, resting_mask: np.ndarray,
    batch_idx: np.ndarray, device: torch.device,
) -> tuple[torch.Tensor, float]:
    """``model.resting_head``'s loss for one batch -- masked position MSE
    (x/y only), same masked-mean shape as ``_crossing_head_loss``'s
    position term, but there's no unmasked delta_t counterpart here (no
    well-defined sentinel for "never comes to rest" the way -1 works for a
    scalar time -- see ``compute_resting_targets``'s docstring). Unlike
    ``_crossing_head_loss``, ``resting_pos``/``resting_mask`` are passed in
    directly (plain arrays, precomputed once in ``train()`` via
    ``ds.compute_resting_targets`` -- not attributes of ``ds`` itself, since
    they depend on the two threshold config values) rather than read off
    ``ds`` -- everything else about the call shape matches.

    Returns ``(loss, pos_dist_mean)`` -- the mean Euclidean resting-position
    error over rows with ``resting_mask`` True (0.0 if none this batch,
    matching the masked loss's own 0-numerator/1-denominator convention).
    """
    pred = model.resting_head(latent)
    r_pos = torch.from_numpy(resting_pos[batch_idx]).to(device)
    r_mask = torch.from_numpy(resting_mask[batch_idx]).to(device)
    mask_f = r_mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    err = pred - r_pos
    loss = (err.pow(2).sum(dim=-1) * mask_f).sum() / denom
    with torch.no_grad():
        pos_dist_mean = float((torch.linalg.norm(err, dim=-1) * mask_f).sum().item() / float(denom.item()))
    return loss, pos_dist_mean


def _position_head_loss(
    model: BallDynamicsAutoencoder, latent: torch.Tensor, x: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """``model.position_head``'s loss for one batch -- plain (unmasked) MSE
    against the CURRENT position, i.e. ``x[:, 0:2]``: the very fields the
    encoder was just given as input to produce ``latent``. Unlike
    ``_crossing_head_loss``/``_resting_head_loss`` there's no mask and no
    separate target array to pass in -- every row has a well-defined
    current position by construction, so the target is read directly off
    the same input tensor the caller already has on hand (works
    identically for the "main" t=0 batch and the "horizon" pseudo-start
    batches, since ``x`` is each pass's own current-state input either
    way).

    Returns ``(loss, pos_dist_mean)`` -- the mean Euclidean position error
    over the whole batch (never masked, unlike crossing/resting).
    """
    pred = model.position_head(latent)
    err = pred - x[:, 0:2]
    loss = err.pow(2).sum(dim=-1).mean()
    with torch.no_grad():
        pos_dist_mean = float(torch.linalg.norm(err, dim=-1).mean().item())
    return loss, pos_dist_mean


def _event_head_loss(
    model: BallDynamicsAutoencoder, latent: torch.Tensor, ever_oob: np.ndarray, ever_goal: np.ndarray,
    row_idx: np.ndarray, device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    """``model.event_head``'s loss for one batch -- plain (unmasked) BCE,
    summed, for two INDEPENDENT binary targets: whether the episode EVER
    goes out of bounds (logit 0) / EVER scores (logit 1) across the whole
    recorded window, as of t=0 -- see ``BallDynamicsDataset.
    compute_event_ever_masks``. No time/horizon conditioning (unlike the
    shared decoder's own per-horizon oob/goal BCE heads) and entirely
    separate parameters from them -- see ``BallDynamicsAutoencoder.
    event_head``'s docstring for why the two are kept apart despite
    predicting a related quantity. Unlike ``_crossing_head_loss``/
    ``_resting_head_loss`` there's no masking: "does this ever happen" is
    always a well-defined 0/1 target for every row.

    Returns ``(loss, oob_acc, goal_acc)`` -- the last two are plain floats
    (not backpropagated), the fraction of this batch's rows where the
    thresholded (>0) logit matches the target, for per-epoch reporting.
    """
    pred = model.event_head(latent)
    oob_t = torch.from_numpy(ever_oob[row_idx].astype(np.float32, copy=False)).to(device)
    goal_t = torch.from_numpy(ever_goal[row_idx].astype(np.float32, copy=False)).to(device)
    oob_loss = F.binary_cross_entropy_with_logits(pred[:, 0], oob_t)
    goal_loss = F.binary_cross_entropy_with_logits(pred[:, 1], goal_t)
    loss = oob_loss + goal_loss
    with torch.no_grad():
        oob_acc = float(((pred[:, 0] > 0) == (oob_t > 0.5)).float().mean().item())
        goal_acc = float(((pred[:, 1] > 0) == (goal_t > 0.5)).float().mean().item())
    return loss, oob_acc, goal_acc


def _build_horizon_bundle(
    ds: BallDynamicsDataset, indices: np.ndarray, n_horizons: int, horizons_s: list[float],
    pair_enabled: bool, pair_max_skip: int, pair_min_start_speed_norm: float,
    has_crossing_data: bool, resting_min_start_speed_norm: float, resting_speed_norm: float,
) -> dict:
    """Precomputes, per recorded horizon ``h``, everything the shared
    "horizon" training step (see ``_run_interleaved_train_epoch``'s
    "horizon" branch) needs beyond the already-built
    ``autoencode_train_data[h]``/``autoencode_val_data[h]`` input/self-target
    pair (built separately, via ``ds.build_autoencoding_data`` -- this
    function only adds what pair/crossing/resting need ON TOP of that same
    shared row set):

    - the adjacent-pair MASK and per-skip targets -- a MASK now, not a row
      filter, unlike the old ``build_adjacent_pair_data`` (which excluded
      already-resolved/too-slow rows outright): t0/crossing/resting all
      need to share the exact same row set autoencode_train_data[h] already
      uses, so pair's own exclusion has to become a per-row mask applied
      only to ITS loss instead of shrinking the shared batch.
    - the horizon-adjusted crossing delta_t/validity: for a pseudo-start at
      horizon h, ``delta_t = crossing_time - horizons_s[h]`` (the crossing
      POSITION needs no adjustment at all -- it's the same fixed (x, y)
      regardless of which horizon you're predicting from). Physics isn't
      latched (see ``ball_episode_gen.generate_episode``'s docstring -- a
      ball that goes oob can bounce back in bounds), so it's possible for
      ``crossing_time < horizons_s[h]`` even when horizon h's own oob/goal
      flags read false; that produces a negative delta_t, which isn't
      meaningful ("time until a FUTURE crossing" can't be negative) and is
      folded into the same -1 "no crossing" sentinel/mask treatment as a
      genuinely crossing-free episode, rather than fed to the loss as-is.
    - the horizon-adjusted resting-position target/validity, via
      ``ds.compute_resting_targets(..., start_horizon_idx=h)``.

    All of these are indexed in the SAME row order as ``indices`` itself
    (matching ``autoencode_train_data[h]``'s own convention -- position i
    corresponds to original dataset row ``indices[i]``), so a caller can
    index all of them with the same local ``row_idx`` already used for
    ``autoencode_train_data[h]``.

    Returns a dict with per-horizon (length ``n_horizons``) lists:
    ``pair_mask``, ``pair_targets`` (list of ``(skip, target_array)`` per
    horizon), ``crossing_dt``, ``crossing_valid`` (``None`` entries when
    ``has_crossing_data`` is False), ``resting_pos``, ``resting_mask``.
    """
    n = len(indices)
    pair_mask: list[np.ndarray] = []
    pair_targets: list[list[tuple[int, np.ndarray]]] = []
    crossing_dt: list[np.ndarray | None] = []
    crossing_valid: list[np.ndarray | None] = []
    resting_pos: list[np.ndarray] = []
    resting_mask: list[np.ndarray] = []

    crossing_times_here = ds.crossing_times[indices] if has_crossing_data else None
    for h in range(n_horizons):
        base_h = h * N_TARGET_FIELDS_PER_HORIZON
        block_h = ds.targets[indices, base_h:base_h + N_TARGET_FIELDS_PER_HORIZON]
        if pair_enabled and n_horizons > 1:
            speed_h = np.linalg.norm(block_h[:, 3:6], axis=1)
            mask_h = (block_h[:, 9] < 0.5) & (block_h[:, 10] < 0.5) & (speed_h >= pair_min_start_speed_norm)
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

        r_pos, r_mask = ds.compute_resting_targets(
            min_start_speed_norm=resting_min_start_speed_norm, rest_speed_norm=resting_speed_norm,
            start_horizon_idx=h,
        )
        resting_pos.append(r_pos[indices])
        resting_mask.append(r_mask[indices])

    return {
        "pair_mask": pair_mask, "pair_targets": pair_targets,
        "crossing_dt": crossing_dt, "crossing_valid": crossing_valid,
        "resting_pos": resting_pos, "resting_mask": resting_mask,
    }


def _migrate_crossing_head_state_dict(state_dict: dict, model: BallDynamicsAutoencoder) -> dict:
    """Backward-compat shim for checkpoints saved when ``crossing_head`` had
    4 outputs (``pos_x, pos_y, height, delta_t``) -- before height was
    dropped (see ``BallDynamicsAutoencoder``'s docstring, current layout
    ``pos_x, pos_y, delta_t``). Drops the old height row (index 2) from
    ``crossing_head.weight``/``.bias`` -- everything else (encoder, decoder)
    is untouched -- so an old checkpoint can still be resumed via
    ``--init-checkpoint`` instead of erroring on a plain shape mismatch.
    Returns a NEW dict (doesn't mutate ``state_dict``); a no-op if shapes
    already match (nothing to migrate) or ``crossing_head.weight`` isn't
    present at all (checkpoint predates the head entirely -- an ordinary
    ``load_state_dict`` error surfaces normally in that case, same as any
    other genuinely-incompatible checkpoint).
    """
    key_w, key_b = "crossing_head.weight", "crossing_head.bias"
    if key_w not in state_dict:
        return state_dict
    old_out, new_out = state_dict[key_w].shape[0], model.crossing_head.weight.shape[0]
    if old_out == 4 and new_out == 3:
        state_dict = dict(state_dict)
        keep_rows = [0, 1, 3]  # pos_x, pos_y, delta_t -- drops row 2 (height)
        state_dict[key_w] = state_dict[key_w][keep_rows]
        state_dict[key_b] = state_dict[key_b][keep_rows]
        log.info(
            "Migrated checkpoint's crossing_head from 4 outputs (pos_x, pos_y, height, delta_t) "
            "to 3 (pos_x, pos_y, delta_t) -- dropped the height row."
        )
    return state_dict


def _summary_stats(values: list[float]) -> dict[str, float]:
    """``{mean, std, min, max}`` of ``values`` -- ``nan`` for all four if
    empty, ``std=0.0`` (not ``nan``) for a single value (a degenerate but
    well-defined population of one, matching ``np.std``'s own convention).
    Shared by the per-epoch gradient-norm and train-loss-delta diagnostics
    below -- both are "here's the spread of some per-batch quantity within
    this epoch" in the same shape, so one helper computes both rather than
    duplicating the 4-way reduction twice.
    """
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}


def _pitch_dims_m(input_row: np.ndarray, params) -> tuple[float, float]:
    """Real-unit ``(pitch_length_m, pitch_width_m)`` for the episode
    ``input_row`` belongs to. Fields 10-11 are always a ratio-to-base
    pitch dims (unaffected by ``normalize_kinematics_by_base_pitch`` --
    see ``ball_episode_gen.py``'s ``_encode_input``), so this conversion
    is the same regardless of that flag.
    """
    return float(input_row[10]) * params.base_pitch_length_m, float(input_row[11]) * params.base_pitch_width_m


def _kinematics_denorm_scales(pitch_length_m: float, pitch_width_m: float, params) -> tuple[float, float, float, float, float]:
    """Returns ``(half_length, half_width, half_diag, spin_scale,
    height_scale)`` -- the scales that undo whatever ``ball_episode_gen.
    py``'s ``_kinematics_divisors`` applied when encoding pos/vel/spin.
    When ``normalize_kinematics_by_base_pitch`` is false (prior, still-
    default behaviour), uses the episode's own real pitch-derived scales
    (passed in via ``pitch_length_m``/``pitch_width_m``, from ``_pitch_
    dims_m``) for x/y, and ``params.height_norm_m`` for height (its own,
    much smaller, per-episode-invariant scale). When true, x/y/height ALL
    use the FIXED base pitch's ``half_diag`` instead -- one shared divisor
    across all 3 position axes, matching whichever convention the data was
    actually encoded with (see ``_kinematics_divisors``'s docstring).
    """
    if params.normalize_kinematics_by_base_pitch:
        half_length = params.base_pitch_length_m / 2
        half_width = params.base_pitch_width_m / 2
        half_diag = (half_length ** 2 + half_width ** 2) ** 0.5
        return half_diag, half_diag, half_diag, BALL_SPIN_NORM_DIVISOR_RAD_S, half_diag
    half_length = pitch_length_m / 2
    half_width = pitch_width_m / 2
    half_diag = (half_length ** 2 + half_width ** 2) ** 0.5
    return half_length, half_width, half_diag, BALL_SPIN_NORM_DIVISOR_RAD_S, params.height_norm_m


def describe_input_row(row: np.ndarray, params) -> str:
    """Denormalizes one raw ``inputs`` row (see ``ball_episode_gen._encode_
    input``) back into real-units initial conditions, for printing.

    Uses THIS episode's own pitch scale (fields 10-11) to recover pos/vel
    when ``normalize_kinematics_by_base_pitch`` is false (prior, still-
    default behaviour, exact for the episode in question), or the fixed
    base pitch dims when true, matching whichever convention the data was
    encoded with -- see ``_kinematics_denorm_scales``.
    """
    from footballcoach.ai.physics_pretrain.ball_episode_gen import BallEpisodeGenParams

    assert isinstance(params, BallEpisodeGenParams)
    pitch_length_m, pitch_width_m = _pitch_dims_m(row, params)
    half_length, half_width, half_diag, spin_scale, height_scale = _kinematics_denorm_scales(pitch_length_m, pitch_width_m, params)

    pos = (float(row[0]) * half_length, float(row[1]) * half_width, float(row[2]) * height_scale)
    vel = (float(row[3]) * half_diag, float(row[4]) * half_diag, float(row[5]) * half_diag)
    spin = (float(row[6]) * spin_scale, float(row[7]) * spin_scale, float(row[8]) * spin_scale)
    speed_mps = (vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2) ** 0.5
    spin_mag = (spin[0] ** 2 + spin[1] ** 2 + spin[2] ** 2) ** 0.5
    restitution = float(row[9])

    return (
        f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m  "
        f"vel=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})m/s  speed={speed_mps:.2f}m/s  "
        f"spin=({spin[0]:.2f}, {spin[1]:.2f}, {spin[2]:.2f})rad/s  spin_mag={spin_mag:.2f}rad/s  "
        f"restitution={restitution:.3f}  pitch={pitch_length_m:.1f}x{pitch_width_m:.1f}m"
    )


def describe_target_row(
    row: np.ndarray, pitch_length_m: float, pitch_width_m: float, params, logits: bool = False,
) -> str:
    """Denormalizes one 11-field target/prediction row (pos/vel/spin/oob/
    goal, same layout as ``ball_episode_gen._encode_target``) into real
    units for printing -- the target-row counterpart to ``describe_input_
    row`` above. Unlike an input row, a target row doesn't carry its own
    pitch scale (fields 10-11 there are oob/goal, not pitch dims), so the
    episode's pitch dims (from its INPUT row, via ``_pitch_dims_m``) must
    be passed in separately -- used or ignored by ``_kinematics_denorm_
    scales`` depending on ``normalize_kinematics_by_base_pitch``.

    ``logits=True`` for a raw model prediction row (fields 9/10 are
    unnormalized oob/goal logits, converted to a sigmoid probability here
    for readability); ``logits=False`` for an actual recorded target row
    (fields 9/10 are already 0/1 labels).
    """
    half_length, half_width, half_diag, spin_scale, height_scale = _kinematics_denorm_scales(pitch_length_m, pitch_width_m, params)

    pos = (float(row[0]) * half_length, float(row[1]) * half_width, float(row[2]) * height_scale)
    vel = (float(row[3]) * half_diag, float(row[4]) * half_diag, float(row[5]) * half_diag)
    spin = (float(row[6]) * spin_scale, float(row[7]) * spin_scale, float(row[8]) * spin_scale)
    if logits:
        oob_p = 1.0 / (1.0 + np.exp(-float(row[9])))
        goal_p = 1.0 / (1.0 + np.exp(-float(row[10])))
        oob_label, goal_label = f"p={oob_p:.3f}", f"p={goal_p:.3f}"
    else:
        oob_label, goal_label = f"{row[9]:.0f}", f"{row[10]:.0f}"

    return (
        f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m  "
        f"vel=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})m/s  "
        f"spin=({spin[0]:.2f}, {spin[1]:.2f}, {spin[2]:.2f})rad/s  "
        f"out_of_bounds={oob_label}  goal_scored={goal_label}"
    )


def compute_confusion_counts(
    pred_heads: list[torch.Tensor], target: torch.Tensor,
) -> dict[str, list[tuple[int, int, int, int]]]:
    """Per-horizon ``(tp, fp, fn, tn)`` counts for ``out_of_bounds``/
    ``goal_scored``, thresholding at ``logit > 0`` (equivalently
    ``sigmoid(logit) > 0.5``).

    Returns raw counts for THIS batch only -- callers must SUM (not
    average) these across a full epoch before computing accuracy/precision/
    recall. Averaging per-batch ratios instead would be wrong here: a small
    batch can easily contain zero predicted (or zero actual) positives for
    the rarer ``goal_scored`` flag, making that batch's precision/recall
    undefined, and naively skipping/zeroing those batches biases the epoch
    average. Summing the raw counts first and dividing once at the end
    sidesteps that entirely.
    """
    out: dict[str, list[tuple[int, int, int, int]]] = {"oob": [], "goal": []}
    for h, head_out in enumerate(pred_heads):
        base = h * N_TARGET_FIELDS_PER_HORIZON
        target_h = target[:, base:base + N_TARGET_FIELDS_PER_HORIZON]
        for key, col in (("oob", 9), ("goal", 10)):
            pred = head_out[:, col] > 0
            actual = target_h[:, col] > 0.5
            tp = int((pred & actual).sum().item())
            fp = int((pred & ~actual).sum().item())
            fn = int((~pred & actual).sum().item())
            tn = int((~pred & ~actual).sum().item())
            out[key].append((tp, fp, fn, tn))
    return out


_GROUPS = {"pos": (0, 3), "vel": (3, 6), "spin": (6, 9)}


def _single_target_loss_with_breakdown(
    pred: torch.Tensor, target: torch.Tensor, pos_weight_row: torch.Tensor | None = None, bce_weight: float = 1.0,
    spin_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Same 5-term loss as one horizon's worth of ``compute_loss`` (pos/vel/
    spin MSE + oob/goal BCE), for a single ``(pred, target)`` pair rather
    than a list of per-horizon heads -- shared by the adjacent-pair and
    autoencode-pretrain training passes in ``train()``, which each produce
    one prediction at a time via ``forward_at()`` rather than the fixed
    per-horizon head list ``compute_loss`` expects. Both callers pass the
    target horizon's own ``pos_weight[horizon_idx]`` row -- an EARLIER
    version of the autoencode-pretrain call site passed ``None``
    (unweighted BCE) as a "it's just a warmup phase" shortcut, but that let
    the rare `goal_scored` class collapse to a trivial always-predict-
    negative solution (visible as identical BCE across horizons in the
    log, since there was no pressure to actually discriminate the rare
    positive). ``pos_weight_row=None`` is still supported for callers that
    genuinely have no meaningful weighting available.

    Returns ``(total, breakdown)`` -- ``breakdown`` has the same component
    names as ``LossBreakdown`` (``pos_rmse``/``pos_dist``/``vel_rmse``/
    ``vel_dist``/``spin_rmse``/``oob_bce``/``goal_bce``, RMSE not raw MSE,
    same reporting convention as everywhere else in this file -- see
    ``LossBreakdown``'s docstring for ``pos_dist``/``vel_dist`` vs. their
    RMSE counterparts) so callers can log/aggregate it the same way as the
    main per-horizon training loop, not just a single blended scalar.
    """
    pos_mse = F.mse_loss(pred[:, 0:3], target[:, 0:3])
    pos_dist = torch.linalg.norm(pred[:, 0:3] - target[:, 0:3], dim=-1).mean()
    vel_mse = F.mse_loss(pred[:, 3:6], target[:, 3:6])
    vel_dist = torch.linalg.norm(pred[:, 3:6] - target[:, 3:6], dim=-1).mean()
    spin_mse = F.mse_loss(pred[:, 6:9], target[:, 6:9])
    if pos_weight_row is not None:
        oob_bce = F.binary_cross_entropy_with_logits(pred[:, 9], target[:, 9], pos_weight=pos_weight_row[0])
        goal_bce = F.binary_cross_entropy_with_logits(pred[:, 10], target[:, 10], pos_weight=pos_weight_row[1])
    else:
        oob_bce = F.binary_cross_entropy_with_logits(pred[:, 9], target[:, 9])
        goal_bce = F.binary_cross_entropy_with_logits(pred[:, 10], target[:, 10])
    total = pos_mse + vel_mse + spin_weight * spin_mse + bce_weight * (oob_bce + goal_bce)
    breakdown = {
        "pos_rmse": float(pos_mse.item()) ** 0.5,
        "pos_dist": float(pos_dist.item()),
        "vel_rmse": float(vel_mse.item()) ** 0.5,
        "vel_dist": float(vel_dist.item()),
        "spin_rmse": float(spin_mse.item()) ** 0.5,
        "oob_bce": float(oob_bce.item()),
        "goal_bce": float(goal_bce.item()),
    }
    return total, breakdown


def _single_target_loss(
    pred: torch.Tensor, target: torch.Tensor, pos_weight_row: torch.Tensor | None = None, bce_weight: float = 1.0,
    spin_weight: float = 1.0,
) -> torch.Tensor:
    """``_single_target_loss_with_breakdown`` without the breakdown, for
    callers (adjacent-pair training) that only need the scalar to
    backprop/accumulate."""
    total, _ = _single_target_loss_with_breakdown(pred, target, pos_weight_row, bce_weight, spin_weight)
    return total


def _single_target_per_episode_loss(
    pred: torch.Tensor, target: torch.Tensor, pos_weight_row: torch.Tensor | None = None, bce_weight: float = 1.0,
    spin_weight: float = 1.0,
) -> torch.Tensor:
    """Same 5-term loss as ``_single_target_loss``, but NOT reduced across
    the batch -- one loss value per row, mirroring ``compute_per_episode_
    loss``'s per-horizon-list version but for a single ``(pred, target)``
    pair (the autoencode-pretrain task's shape). Used to find the single
    worst-reconstructed (episode, horizon) example for the post-phase
    diagnostic below."""
    pos_sq = (pred[:, 0:3] - target[:, 0:3]).pow(2).mean(dim=1)
    vel_sq = (pred[:, 3:6] - target[:, 3:6]).pow(2).mean(dim=1)
    spin_sq = (pred[:, 6:9] - target[:, 6:9]).pow(2).mean(dim=1)
    if pos_weight_row is not None:
        oob_bce = F.binary_cross_entropy_with_logits(pred[:, 9], target[:, 9], pos_weight=pos_weight_row[0], reduction="none")
        goal_bce = F.binary_cross_entropy_with_logits(pred[:, 10], target[:, 10], pos_weight=pos_weight_row[1], reduction="none")
    else:
        oob_bce = F.binary_cross_entropy_with_logits(pred[:, 9], target[:, 9], reduction="none")
        goal_bce = F.binary_cross_entropy_with_logits(pred[:, 10], target[:, 10], reduction="none")
    return pos_sq + vel_sq + spin_weight * spin_sq + bce_weight * (oob_bce + goal_bce)


def _iterate_numpy_minibatches(
    inputs: np.ndarray, targets: np.ndarray, batch_size: int, device: str,
    rng: np.random.Generator | None = None,
):
    """Like ``BallDynamicsDataset.iterate_minibatches``, but for plain
    derived numpy arrays (not indices into the dataset) -- used for the
    adjacent-pair and autoencode-pretrain data, which are standalone
    arrays built by ``build_adjacent_pair_data``/``build_autoencoding_data``,
    not rows of the original dataset."""
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
    """Splits every horizon's data into minibatch-sized row-index chunks
    (shuffled within each horizon, same as ``_iterate_numpy_minibatches``
    with an ``rng``), then shuffles the ORDER these chunks get processed in
    across ALL horizons combined -- interleaving horizons throughout a
    training pass instead of training on horizon 0's data start-to-finish
    before horizon 1 even begins. Each returned chunk still contains rows
    from only ONE horizon, so callers can keep using a single per-horizon
    ``pos_weight[h_idx]`` row for the whole chunk -- only the ORDER chunks
    are visited in changes, not their contents.

    Matters because gradient updates happen continuously through a
    training pass (autoencode pretraining, decoder-only pretraining, and
    the main loop's t0 term all update the model after every minibatch):
    with a fixed horizon-major order, whichever horizon goes first is
    always measured against the least-updated model that epoch, and
    whichever goes last benefits from every update the earlier horizons
    already made -- a systematic (not noise-averaging-out) bias in
    per-horizon TRAIN metrics specifically, since it repeats identically
    every epoch rather than washing out. Confirmed in physics_runs.md: a
    pure forward-only "epoch 0 (before training)" baseline showed FLAT
    pos_rmse across horizons, but every subsequent trained epoch showed
    horizon 0 alone sitting 3-6x above its neighbors, purely from always
    being processed first. Eval-only passes (no optimizer) don't need
    this -- no gradient updates happen, so a fixed order can't bias
    anything there.
    """
    batches: list[tuple[int, np.ndarray]] = []
    for h_idx, (h_inputs, _h_targets) in enumerate(data):
        order = rng.permutation(len(h_inputs))
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            if len(chunk) > 0:
                batches.append((h_idx, chunk))
    shuffle_order = rng.permutation(len(batches))
    return [batches[i] for i in shuffle_order]


def compute_group_sq_err(
    pred_heads: list[torch.Tensor], target: torch.Tensor,
) -> dict[str, list[tuple[float, int]]]:
    """Per-horizon ``(sum of squared error, element count)`` for the three
    continuous quantity groups (pos/vel/spin) -- the raw ingredients for an
    R^2 sanity check ("is this actually better than always predicting the
    mean, or did it just collapse to it?").

    Returns raw sums for THIS batch only -- callers must SUM (not average)
    these across a full epoch before dividing into an MSE, same
    sum-then-divide rationale as ``compute_confusion_counts`` above
    (averaging per-batch MSE would be biased by a smaller trailing batch).
    """
    out: dict[str, list[tuple[float, int]]] = {g: [] for g in _GROUPS}
    for h, head_out in enumerate(pred_heads):
        base = h * N_TARGET_FIELDS_PER_HORIZON
        target_h = target[:, base:base + N_TARGET_FIELDS_PER_HORIZON]
        for g, (lo, hi) in _GROUPS.items():
            diff = head_out[:, lo:hi] - target_h[:, lo:hi]
            out[g].append((float((diff ** 2).sum().item()), diff.numel()))
    return out


def _r2_from_sq_err(sq_err: np.ndarray, n: np.ndarray, target_var: np.ndarray) -> np.ndarray:
    """``1 - MSE/Var(target)`` per horizon. ``target_var`` is always the
    TRAIN set's variance (even for val R^2) -- a fixed, not-peeked-at
    baseline, same convention as pos_weight/other stats being computed only
    from ``train_idx``. NaN where a horizon had zero elements this epoch
    (shouldn't happen in practice, guarded for consistency with the
    classification-metrics NaN convention)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mse = np.where(n > 0, sq_err / np.maximum(n, 1), np.nan)
        return 1.0 - mse / target_var


def _pct_of_baseline_from_sq_err(sq_err: np.ndarray, n: np.ndarray, baseline_mse: np.ndarray) -> np.ndarray:
    """``100 * sqrt(MSE_model / MSE_baseline)`` per horizon -- the model's
    typical (RMSE) error as a percentage of some baseline's typical error
    (e.g. the "persistence" or "ballistic" baselines below). More directly
    readable than the equivalent R^2 form (``1 - MSE_model/MSE_baseline``):
    R^2 compresses via squaring (a 0.9 there is only a ~68% RMSE reduction,
    not 90%), so this reports the same underlying ratio in the
    percentage-of-baseline-error form instead. 100% = exactly as good as
    the baseline; 0% = perfect; values > 100% are possible (worse than the
    baseline). Same train-only baseline convention as elsewhere -- NaN
    where a horizon had zero elements this epoch."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mse = np.where(n > 0, sq_err / np.maximum(n, 1), np.nan)
        return 100.0 * np.sqrt(mse / baseline_mse)


def _safe_nanmean(arr: np.ndarray) -> float:
    """``np.nanmean`` without the "Mean of empty slice" warning for an
    all-NaN row -- expected/harmless here (e.g. a rare event with zero
    positives anywhere in a small val split), not a bug to surface."""
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _classification_metrics(counts: np.ndarray) -> dict[str, np.ndarray]:
    """``counts``: ``(n_horizons, 4)`` array of summed ``(tp, fp, fn, tn)``.

    Returns ``{"accuracy": ..., "precision": ..., "recall": ...}``, each a
    length-``n_horizons`` array. ``nan`` where the denominator is zero
    (e.g. precision when nothing was predicted positive that epoch) rather
    than a misleading 0 -- "no predictions to be wrong about" is not the
    same claim as "0% precision".
    """
    tp, fp, fn, tn = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    total = tp + fp + fn + tn
    with np.errstate(invalid="ignore", divide="ignore"):
        accuracy = np.where(total > 0, (tp + tn) / np.maximum(total, 1), np.nan)
        precision = np.where((tp + fp) > 0, tp / np.maximum(tp + fp, 1), np.nan)
        recall = np.where((tp + fn) > 0, tp / np.maximum(tp + fn, 1), np.nan)
    return {"accuracy": accuracy, "precision": precision, "recall": recall}


def _build_phase_optimizer(
    params, lr: float, cfg: dict, type_key: str, momentum_key: str, log_label: str, weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Builds this phase's own optimizer from its own ``cfg[type_key]``/
    ``cfg[momentum_key]`` -- each of the main loop, autoencode-pretrain, and
    decoder-only-pretrain phases gets an independently-configurable choice
    (they used to all share the main loop's ``optimizer_type``/
    ``sgd_momentum`` for the SGD case, while autoencode/decoder-only were
    silently always Adam regardless of what the main loop was set to).
    Logs which one it picked either way, so a run's log makes clear which
    of the (now 3) independent choices actually took effect."""
    optimizer_type = str(cfg.get(type_key, "adam")).lower()
    if optimizer_type == "sgd":
        momentum = float(cfg.get(momentum_key, 0.9))
        log.info(f"{log_label} optimizer: SGD (momentum={momentum}, lr={lr:.2e}, weight_decay={weight_decay})")
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        log.info(f"{log_label} optimizer: Adam (lr={lr:.2e}, weight_decay={weight_decay})")
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown physics_pretrain.ball.{type_key}: {optimizer_type!r} (expected 'adam' or 'sgd')")


def _physics_config_hash() -> str:
    """Hash of physics.json's ball_physics section, saved alongside the
    checkpoint so a later load can warn if the real physics has drifted
    since this encoder was trained (§7, §8.4's staleness check)."""
    from footballcoach.config import load_physics_config, require_section
    section = require_section(load_physics_config(), "ball_physics")
    return hashlib.sha256(json.dumps(section, sort_keys=True).encode()).hexdigest()[:16]


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
    cfg = load_ai_config()["physics_pretrain"]["ball"]

    ds = BallDynamicsDataset.from_directory(dataset_dir)
    # freeze_semantics (default true): reconstructs the OLD freeze-on-event
    # targets (state frozen at the first out_of_bounds/goal_scored crossing,
    # held + latched for every later horizon) from the dataset's always-
    # continuous stored targets + per-episode crossings/crossing_times --
    # see BallDynamicsDataset.targets_with_freeze_semantics's docstring and
    # ball_episode_gen.generate_episode's. false = train against the raw,
    # always-continuous physics (ball never freezes; out_of_bounds/
    # goal_scored are per-horizon instantaneous, not latched) as actually
    # stored. Purely a TRAINING-time choice -- the same generated dataset
    # supports either, no regeneration needed to compare them.
    if bool(cfg.get("freeze_semantics", True)):
        ds.targets = ds.targets_with_freeze_semantics(list(cfg["horizons_s"]))
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

    # Real-unit scale factors for the RMSE log lines below (and re-derived
    # identically after training for the saved checkpoint/report -- kept
    # here too so per-epoch logging can use them without waiting for
    # training to finish). See the longer comment near the post-loop
    # `normalization` dict for the position-RMSE approximation caveat.
    from footballcoach.ai.physics_pretrain.ball_episode_gen import BallEpisodeGenParams
    import math as _math
    gen_params = BallEpisodeGenParams.from_config()
    # Always the BASE pitch's half-diagonal, regardless of
    # normalize_kinematics_by_base_pitch: an APPROXIMATE real-unit stand-in
    # when the data is encoded per-episode (false, default -- different
    # episodes actually used different scales), or an EXACT conversion
    # when encoded by-base-pitch (true -- every episode used this same
    # scale). height/spin were never per-episode-scaled either way.
    pitch_half_diag_m = _math.hypot(gen_params.base_pitch_length_m / 2, gen_params.base_pitch_width_m / 2)
    _RMSE_UNIT_SCALE = {
        "pos_rmse": (pitch_half_diag_m, "m"),
        "pos_dist": (pitch_half_diag_m, "m"),
        "vel_rmse": (pitch_half_diag_m, "m/s"),
        "vel_dist": (pitch_half_diag_m, "m/s"),
        "spin_rmse": (BALL_SPIN_NORM_DIVISOR_RAD_S, "rad/s"),
    }

    _COMPONENTS = ("pos_rmse", "pos_dist", "vel_rmse", "vel_dist", "spin_rmse", "oob_bce", "goal_bce")

    def _log_component(prefix: str, c: str, values: np.ndarray) -> None:
        if c in _RMSE_UNIT_SCALE:
            scale, unit = _RMSE_UNIT_SCALE[c]
            real = values * scale
            log.info(f"    {prefix} {c:9s} by horizon ({unit}): {np.array2string(real, precision=4)}, mean: {real.mean():.4f} {unit}")
        else:
            log.info(f"    {prefix} {c:9s} by horizon: {np.array2string(values, precision=4)}, mean: {values.mean():.4f}")

    def _mean_breakdown_by_horizon(breakdowns_by_h: list[list[dict]]) -> dict[str, np.ndarray]:
        """Same shape/averaging convention as `_mean_breakdown` (below) but
        for the dict-shaped breakdowns `_single_target_loss_with_breakdown`
        returns (single-prediction callers: autoencode pretraining, the
        main loop's/decoder-only pretraining's t=0 pass, adjacent-pair
        training) rather than the `LossBreakdown` dataclass `compute_loss`
        returns for the full per-horizon head list."""
        return {
            c: np.array([
                np.mean([b[c] for b in breakdowns_by_h[h]]) if breakdowns_by_h[h] else np.nan
                for h in range(n_horizons)
            ])
            for c in _COMPONENTS
        }

    # "Predict the mean" baseline variance per horizon/group, for the R^2
    # sanity check below (is the model actually better than always guessing
    # the train-set mean, or did it collapse to it?). Always from train_idx
    # only, including when used as the baseline for val R^2 -- see
    # `_r2_from_sq_err`'s docstring.
    target_var = ds.compute_group_variance(n_horizons, indices=train_idx)

    # "Predict the initial state" (persistence) baseline -- a much stronger
    # baseline than the train-set mean, especially at short horizons where
    # the ball genuinely hasn't moved much yet. Also doubles as an estimate
    # of the horizon's typical displacement, so 1 - model_mse/this
    # automatically normalizes error by how much the group actually tends
    # to change over that horizon (see `compute_persistence_baseline_mse`'s
    # docstring). Same train-only convention as target_var above.
    persistence_mse = ds.compute_persistence_baseline_mse(n_horizons, indices=train_idx)

    # "Straight-line physics" (ballistic) baseline -- constant velocity +
    # gravity only, no drag/Magnus/bounce. Much stronger than persistence
    # for pos/vel specifically: position and velocity are handed to the
    # encoder directly, and this extrapolation is close to exact at short
    # horizons, so beating it substantially there is a meaningful bar (see
    # `compute_ballistic_baseline_mse`'s docstring). Same train-only
    # convention as the other baselines.
    from footballcoach.engine.ball_physics import BallPhysicsParams
    gravity_mps2 = BallPhysicsParams.from_config().gravity_mps2
    ballistic_mse = ds.compute_ballistic_baseline_mse(
        list(cfg["horizons_s"]), gen_params, gravity_mps2, indices=train_idx,
    )

    model = BallDynamicsAutoencoder.from_config().to(device)
    # weight_decay (default 0.0) is deliberately applied ONLY to this main-
    # loop optimizer -- NOT autoencode_optimizer/decoder_only_optimizer
    # below. Uniform L2 decay pulls EVERY parameter toward 0 every step,
    # including the identity-shortcut's hand-set weights (encoder.out's
    # near-identity block, the decoder's dedicated identity units) --
    # unlike gradient-based corruption, decay isn't blocked by the
    # identity-shortcut's backward-hook masks (those only zero gradients
    # flowing from the LOSS, not the optimizer's separate decay term). The
    # main loop's own t=0 term (autoencode_during_main_loop_enabled) is
    # running every epoch specifically to keep pulling those weights back
    # toward exact, so a small decay value should just settle into a mild
    # equilibrium with it rather than eroding away -- watch `train t0`/
    # `val   t0` pos/vel/spin RMSE in the logs if trying a larger value, to
    # confirm it's actually holding. autoencode/decoder-only pretraining
    # deliberately get NO decay: those phases exist specifically to
    # ESTABLISH the good initial values before the main loop's ongoing t=0
    # reinforcement exists to counteract any erosion, so there's nothing
    # there yet to fight a constant pull toward 0.
    weight_decay = float(cfg.get("weight_decay", 0.0))
    # Main-loop optimizer choice -- independently configurable from
    # autoencode_optimizer/decoder_only_optimizer below (their own
    # autoencode_optimizer_type/decoder_only_optimizer_type), since they're
    # early-phase warmup (fast, robust initial descent is what Adam is good
    # at) vs. the late-stage fine-convergence this switch is originally for
    # -- but nothing stops trying SGD there too if that phase turns out to
    # have the same late-stage-noise symptom. "adam" (default, no behaviour
    # change): per-parameter adaptive step size (lr / sqrt(recent grad
    # variance)) -- fast and robust early on, but that same adaptivity means
    # a parameter with quiet recent gradients can get a disproportionately
    # LARGE effective step the moment a slightly bigger gradient shows up,
    # even at a tiny nominal `lr` -- a plausible source of the occasional
    # worse-epoch val_loss bumps seen late in a long, already-converging run
    # (see physics_runs.md). "sgd": plain (optionally momentum) SGD -- step
    # is just `lr * grad`, no per-parameter denominator to spike, typically
    # smoother/more monotonic once already near a good minimum, at the cost
    # of needing its own (usually different, often larger) `lr` re-tuned
    # rather than reusing whatever worked for Adam.
    optimizer = _build_phase_optimizer(
        model.parameters(), lr, cfg, "optimizer_type", "sgd_momentum", "Main-loop", weight_decay=weight_decay,
    )

    # Multiplies the oob/goal BCE terms' contribution to every backprop'd
    # loss in this run (main heads, adjacent-pair, autoencode-pretrain,
    # t0 term -- every call site below passes this through) -- see
    # compute_loss's docstring. Reported oob_bce/goal_bce metrics are
    # always the raw, UNweighted value regardless of this setting, so
    # comparing runs with different weights (including 0.0, fully
    # disabling the classification heads' gradient) stays apples-to-apples.
    bce_weight = float(cfg.get("bce_loss_weight", 1.0))

    # Same idea but for the spin MSE term -- see compute_loss's docstring.
    # Reported spin_rmse is always the raw, UNweighted value regardless of
    # this setting, so 0.0 (fully disabling the spin head's gradient) stays
    # comparable to other runs.
    spin_weight = float(cfg.get("spin_loss_weight", 1.0))

    # Weight on model.crossing_head's own loss (crossing position, masked to
    # episodes that actually went oob/scored within the simulated window,
    # plus unmasked delta_t against the ds.crossing_dt -1-sentinel target --
    # see BallDynamicsDataset's docstring). 0.0 fully disables its gradient,
    # same convention as bce_weight/spin_weight above. `has_crossing_data`
    # guards every crossing-head call site below: a dataset built without
    # crossings/crossing_times (e.g. a hand-built one in a test) simply skips
    # the head entirely rather than crashing on a None array.
    crossing_weight = float(cfg.get("crossing_loss_weight", 1.0))
    has_crossing_data = ds.crossing_pos is not None
    if has_crossing_data:
        # Episodes that started ALREADY out of bounds/in a goal mouth (see
        # compute_already_out_of_bounds_at_start_mask's docstring) get a
        # "crosses almost immediately" crossing_dt/crossing_pos recorded --
        # a near-trivial function of the raw t=0 input, not the "will an
        # in-play ball actually go out/score" signal the head exists to
        # predict. Excluded the same way "never crosses" already is: mask
        # dropped, delta_t forced to the -1 sentinel.
        already_oob_at_start = ds.compute_already_out_of_bounds_at_start_mask(gen_params)
        ds.crossing_mask = ds.crossing_mask & ~already_oob_at_start
        ds.crossing_dt = np.where(already_oob_at_start, -1.0, ds.crossing_dt).astype(np.float32)

    # Weight on model.resting_head's own loss (see BallDynamicsDataset.
    # compute_resting_targets / BallDynamicsAutoencoder.resting_head's
    # docstrings) -- same 0.0-disables convention as crossing_weight above.
    # Unlike crossing, resting targets are derivable from self.inputs/
    # self.targets alone (no extra recorded columns needed), so there's no
    # has_resting_data guard -- it's always computable, just possibly
    # entirely masked-out (resting_mask all False) if no episode in this
    # dataset ever comes to rest under rest_speed_norm, which the row-count
    # summary below will make visible either way.
    resting_weight = float(cfg.get("resting_loss_weight", 1.0))
    resting_min_start_speed_mps = float(cfg.get("resting_min_start_speed_mps", 1.5))
    resting_speed_threshold_mps = float(cfg.get("resting_speed_threshold_mps", 0.01))
    resting_pos, resting_mask = ds.compute_resting_targets(
        min_start_speed_norm=resting_min_start_speed_mps / pitch_half_diag_m,
        rest_speed_norm=resting_speed_threshold_mps / pitch_half_diag_m,
    )

    # Weight on model.position_head's own loss (current x/y position,
    # unmasked -- see BallDynamicsAutoencoder.position_head's docstring).
    # Same 0.0-disables convention as crossing_weight/resting_weight above.
    # Unlike either of those, there's no dataset-derived target/mask at
    # all: the target is always just the batch's own input fields 0:2.
    position_weight = float(cfg.get("position_loss_weight", 1.0))

    # Weight on model.event_head's own loss (two independent unmasked BCE
    # terms -- "does the episode EVER go out of bounds", "does it EVER
    # score" -- see BallDynamicsAutoencoder.event_head's docstring). Same
    # 0.0-disables convention. t=0 ONLY (no horizon-pass generalization --
    # see event_head's docstring for why), so ever_oob/ever_goal are
    # precomputed ONCE here over the full dataset rather than per-horizon
    # like crossing/resting.
    event_weight = float(cfg.get("event_loss_weight", 1.0))
    ever_oob, ever_goal = ds.compute_event_ever_masks(gen_params)

    # Per-epoch diagnostic printing skips a component entirely once its
    # weight is 0 -- it's not receiving gradient, so its (raw, unweighted)
    # value logged every epoch is noise, not signal. `_COMPONENTS` itself
    # stays unchanged since it's also used to compute the underlying means
    # dicts (e.g. `train_means`), which other code still reads by key.
    _LOG_COMPONENTS = tuple(
        c for c in _COMPONENTS
        if not (c == "spin_rmse" and spin_weight == 0.0)
        and not (c in ("oob_bce", "goal_bce") and bce_weight == 0.0)
    )

    # Resume from a phase checkpoint written by `_save_phase_checkpoint`
    # below (after_autoencode/after_decoder_pretrain/after_training), or
    # from an older encoder-only artifact (the final `torch.save` at the
    # bottom of this function) -- either way, weights are loaded BEFORE any
    # phase below runs, so e.g. loading an "after_autoencode" checkpoint and
    # leaving autoencode_pretrain_epochs=0 in the config skips straight to
    # decoder-only pretrain / the main loop on top of the restored weights,
    # while leaving autoencode_pretrain_epochs>0 continues pretraining
    # further first. An encoder-only artifact has no decoder weights to
    # restore, so the decoder is left at its fresh init in that case.
    if init_checkpoint:
        ckpt = torch.load(init_checkpoint, map_location=device)
        old_cfg = ckpt.get("config_snapshot")
        # If hidden_dim/encoder_bottleneck_dim/latent_dim/decoder_hidden_dim
        # in the CURRENT config don't match what this checkpoint was saved
        # with, a plain load_state_dict can't work at all -- those are real
        # shape mismatches on EXISTING keys, which strict=False doesn't
        # help with (it only tolerates keys missing/extra, never a keeping
        # key whose shape changed). Route through widen_ball_checkpoint.py's
        # seam-preserving surgery instead of failing, so bumping those config
        # values and resuming from an old checkpoint just works -- see that
        # module's docstring for why this exactly preserves old training
        # rather than approximating it.
        widen_needed = old_cfg is not None and any(
            old_cfg.get(k, 32) != cfg.get(k, 32)
            for k in ("hidden_dim", "encoder_bottleneck_dim", "latent_dim", "decoder_hidden_dim")
        )
        if widen_needed:
            from footballcoach.ai.physics_pretrain.widen_ball_checkpoint import _build_model, _validate_widen_cfgs, widen_model_
            _validate_widen_cfgs(old_cfg, cfg)
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
            # strict=False: tolerates a checkpoint saved before a NEW head
            # (e.g. resting_head) existed at all -- those params are simply
            # missing from the old state_dict rather than shape-mismatched
            # (that case is what _migrate_crossing_head_state_dict handles
            # instead), so they're left at their fresh random init and
            # start learning from scratch on top of everything else that
            # WAS restored. Logged explicitly so a genuinely-missing/
            # unexpected key isn't silently invisible.
            missing, unexpected = model.load_state_dict(
                _migrate_crossing_head_state_dict(ckpt["model_state_dict"], model), strict=False,
            )
            if missing:
                log.info(f"Checkpoint missing {len(missing)} param(s) not present when it was saved (left at fresh init): {missing}")
            if unexpected:
                log.info(f"Checkpoint had {len(unexpected)} unexpected param(s), ignored: {unexpected}")
            log.info(f"Resumed full model (encoder+decoder) from {init_checkpoint} (phase={ckpt.get('phase', '?')})")
        else:
            model.encoder.load_state_dict(ckpt["encoder_state_dict"])
            log.info(f"Resumed encoder only from {init_checkpoint} (decoder left at fresh init)")

    def _save_phase_checkpoint(phase: str) -> None:
        """Full model (encoder+decoder) checkpoint written either after a
        given phase completes or (phase="midtrain_latest") on every new
        best-val epoch of the main loop, so a later run can resume from
        exactly that point via `init_checkpoint` above -- distinct from the
        encoder-only artifact saved at `output_path` at the very end of
        training."""
        path = Path(output_path).with_suffix(f".{phase}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "config_snapshot": cfg,
            "normalization": {
                "pitch_half_diag_m": pitch_half_diag_m,
                "height_norm_m": gen_params.height_norm_m,
                "ball_spin_norm_max_rad_s": BALL_SPIN_NORM_DIVISOR_RAD_S,
            },
            "physics_config_hash": _physics_config_hash(),
            "phase": phase,
        }, path)
        log.info(f"Saved '{phase}' checkpoint to {path}")

    # Autoencode data (t=0 reconstruction): for EVERY recorded horizon (not
    # just t=0), treat its recorded state as a pseudo-initial-state and
    # reconstruct it via forward_at(latent, 0.0) -- see
    # `BallDynamicsDataset.build_autoencoding_data`'s docstring. Built once
    # here (pure reshaping of already-recorded data, no new simulation) if
    # EITHER the dedicated pretrain phase below OR the main-loop t=0 term
    # (see `autoencode_during_main_loop_enabled` near the main loop) needs
    # it, so the two don't each build their own copy.
    autoencode_pretrain_epochs = int(cfg.get("autoencode_pretrain_epochs", 0))
    autoencode_during_main_loop_enabled = bool(cfg.get("autoencode_during_main_loop_enabled", True))
    # Read early (rest of adjacent-pair config follows further below, once
    # pitch_half_diag_m etc. are available) -- needed here already since
    # adjacent-pair training now RIDES on this same per-horizon pseudo-start
    # data (see the "horizon" branch of _run_interleaved_train_epoch), so
    # autoencode_train_data/autoencode_val_data must be built whenever pair
    # training wants them too, not just when the t0 term itself is enabled.
    adjacent_pair_training_enabled = bool(cfg.get("adjacent_pair_training_enabled", True))
    autoencode_train_data: list[tuple[np.ndarray, np.ndarray]] = []
    autoencode_val_data: list[tuple[np.ndarray, np.ndarray]] = []
    if (
        autoencode_pretrain_epochs > 0 or autoencode_during_main_loop_enabled
        or (adjacent_pair_training_enabled and n_horizons > 1)
    ):
        autoencode_train_data = [ds.build_autoencoding_data(h, indices=train_idx) for h in range(n_horizons)]
        autoencode_val_data = [ds.build_autoencoding_data(h, indices=val_idx) for h in range(n_horizons)] if len(val_idx) > 0 else []

    # Autoencode PRETRAINING: an optional phase run BEFORE the main loop
    # that actively trains (not just uses) the t=0 round-trip, complements
    # (does not replace) the post-training t=0 diagnostic below, which
    # checks the FINAL model rather than actively training on this
    # objective. Flat LR, its OWN (`autoencode_lr`, separate from the main
    # loop's `lr`/cosine schedule) -- this phase is a much easier task
    # (copy-through, no real dynamics to learn) so the right step size for
    # it isn't necessarily the same as what suits the main loop. Uses its
    # own optimizer instance (`autoencode_optimizer`) rather than the main
    # `optimizer`, so their momentum/second-moment state don't mix -- its
    # type/momentum are independently configurable via autoencode_
    # optimizer_type/autoencode_sgd_momentum (default "adam", same as ever,
    # if unset). 0 autoencode_pretrain_epochs (default) = disabled, no
    # behaviour change.
    if autoencode_pretrain_epochs > 0:
        autoencode_lr = float(cfg.get("autoencode_lr", lr))
        autoencode_optimizer = _build_phase_optimizer(
            model.parameters(), autoencode_lr, cfg, "autoencode_optimizer_type", "autoencode_sgd_momentum", "Autoencode-pretrain",
        )
        log.info(
            f"Autoencode pretraining: {autoencode_pretrain_epochs} epoch(s) across all "
            f"{n_horizons} recorded horizons (t=0 reconstruction, {len(train_idx):,} train / {len(val_idx):,} val episodes each), "
            f"lr={autoencode_lr:.2e}"
        )

        def _eval_autoencode_pass(data_list: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, dict[str, np.ndarray]]:
            """Forward-only (no backward/step) pass over every horizon's
            autoencoding data, current model weights as-is -- used both for
            each epoch's val pass and the pre-training "epoch 0" baseline
            below (which reuses this on TRAIN data too, since that call
            site needs no gradient either)."""
            losses: list[float] = []
            breakdowns_by_h: list[list[dict]] = [[] for _ in range(n_horizons)]
            with torch.no_grad():
                for h_idx, (ae_inputs, ae_targets) in enumerate(data_list):
                    for x, y in _iterate_numpy_minibatches(ae_inputs, ae_targets, batch_size, device):
                        latent = model.encoder(x)
                        pred = model.decoder.forward_at(latent, 0.0)
                        loss, breakdown = _single_target_loss_with_breakdown(pred, y, pos_weight[h_idx], bce_weight, spin_weight)
                        losses.append(float(loss.item()))
                        breakdowns_by_h[h_idx].append(breakdown)
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            return mean_loss, _mean_breakdown_by_horizon(breakdowns_by_h)

        def _log_autoencode_epoch(
            label: str, mean_train_loss: float, train_means: dict[str, np.ndarray],
            mean_val_loss: float | None, val_means: dict[str, np.ndarray] | None,
        ) -> None:
            val_line = f"  val_loss={mean_val_loss:.4f}" if mean_val_loss is not None else ""
            log.info(f"  autoencode pretrain {label}: train_loss={mean_train_loss:.4f}{val_line}")
            for c in _LOG_COMPONENTS:
                _log_component("    train", c, train_means[c])
            if val_means is not None:
                for c in _LOG_COMPONENTS:
                    _log_component("    val  ", c, val_means[c])

        # Baseline: forward-only pass with the model exactly as constructed
        # (identity_shortcut init or plain random init, whichever's
        # configured), BEFORE any gradient step. Makes it possible to see
        # directly what the init alone buys (or doesn't) rather than only
        # ever seeing loss/RMSE after at least one epoch of training has
        # already run.
        model.eval()
        mean_train_loss0, train_means0 = _eval_autoencode_pass(autoencode_train_data)
        mean_val_loss0, val_means0 = (
            _eval_autoencode_pass(autoencode_val_data) if autoencode_val_data else (None, None)
        )
        _log_autoencode_epoch(
            f"epoch 0/{autoencode_pretrain_epochs} (before training)",
            mean_train_loss0, train_means0, mean_val_loss0, val_means0,
        )

        for ae_epoch in range(autoencode_pretrain_epochs):
            model.train()
            train_losses_ae: list[float] = []
            train_breakdowns_by_h: list[list[dict]] = [[] for _ in range(n_horizons)]
            for h_idx, row_idx in _interleaved_horizon_batches(autoencode_train_data, batch_size, rng):
                ae_inputs, ae_targets = autoencode_train_data[h_idx]
                x = torch.from_numpy(ae_inputs[row_idx].astype(np.float32, copy=False)).to(device)
                y = torch.from_numpy(ae_targets[row_idx].astype(np.float32, copy=False)).to(device)
                latent = model.encoder(x)
                pred = model.decoder.forward_at(latent, 0.0)
                loss, breakdown = _single_target_loss_with_breakdown(pred, y, pos_weight[h_idx], bce_weight, spin_weight)
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

            _log_autoencode_epoch(
                f"epoch {ae_epoch + 1}/{autoencode_pretrain_epochs}",
                mean_train_loss_ae, train_means_ae, mean_val_loss_ae, val_means_ae,
            )

        # Worst single (episode, horizon) autoencode reconstruction on val,
        # scanned across ALL horizons -- same "worst-by-loss" diagnostic
        # pattern as the main loop's post-training one (see
        # `compute_per_episode_loss` below), but for THIS phase's task
        # (t=0 reconstruction of a recorded horizon's own state) using
        # `_single_target_per_episode_loss`. Minibatched (not one giant
        # forward pass) since `autoencode_val_data` can be millions of rows
        # per horizon on a real dataset; `rng=None` keeps minibatches in
        # original order so the batch's `start` offset recovers the row's
        # index into that horizon's derived arrays.
        if autoencode_val_data:
            model.eval()
            worst_loss = -1.0
            worst_h = -1
            worst_row_idx = -1
            worst_pred_row: np.ndarray | None = None
            with torch.no_grad():
                for h_idx, (ae_inputs, ae_targets) in enumerate(autoencode_val_data):
                    start = 0
                    for x, y in _iterate_numpy_minibatches(ae_inputs, ae_targets, batch_size, device):
                        latent = model.encoder(x)
                        pred = model.decoder.forward_at(latent, 0.0)
                        per_ep = _single_target_per_episode_loss(pred, y, pos_weight[h_idx], bce_weight, spin_weight).cpu().numpy()
                        local_idx = int(np.argmax(per_ep))
                        if per_ep[local_idx] > worst_loss:
                            worst_loss = float(per_ep[local_idx])
                            worst_h = h_idx
                            worst_row_idx = start + local_idx
                            worst_pred_row = pred[local_idx].detach().cpu().numpy()
                        start += len(per_ep)
            if worst_h >= 0:
                worst_input_row = autoencode_val_data[worst_h][0][worst_row_idx]
                worst_target_row = autoencode_val_data[worst_h][1][worst_row_idx]
                pitch_length_m, pitch_width_m = _pitch_dims_m(worst_input_row, gen_params)
                log.info(
                    f"Worst autoencode val example (horizon={cfg['horizons_s'][worst_h]}s, loss={worst_loss:.4f}):"
                )
                log.info(f"    input : {describe_input_row(worst_input_row, gen_params)}")
                log.info(
                    f"    target: {describe_target_row(worst_target_row, pitch_length_m, pitch_width_m, gen_params, logits=False)}"
                )
                log.info(
                    f"    pred  : {describe_target_row(worst_pred_row, pitch_length_m, pitch_width_m, gen_params, logits=True)}"
                )

        _save_phase_checkpoint("after_autoencode")

    # Adjacent-horizon-pair training, UNIFIED with the t0/crossing/resting
    # per-horizon tasks (see `_build_horizon_bundle`'s and
    # `_run_interleaved_train_epoch`'s "horizon" branch docstrings): rather
    # than deriving its own separately-filtered row set the way
    # `build_adjacent_pair_data` used to, pair training now rides the SAME
    # unfiltered `autoencode_train_data[h]` rows every other per-horizon
    # task uses, with its own resolved-state/speed exclusion applied as a
    # MASK on its loss instead of a row filter -- so a single shared encoder
    # call per horizon-batch can feed t0 reconstruction, every pair skip,
    # crossing, and resting all at once. adjacent_pair_training_enabled=false
    # (default true) to disable without touching anything else.
    #
    # `adjacent_pair_max_skip` (default 1, "adjacent" in the strict sense):
    # how many horizons ahead of each start horizon to ALSO predict, not
    # just the immediately-next one. Kept well short of ALL (i, j) pairs
    # deliberately: going further doesn't test anything the main task +
    # adjacent pairs don't already cover indirectly (the shared decoder's
    # continuous horizon features already generalize across deltas -- see
    # BallDynamicsDecoder's docstring), while ALL-pairs would be a
    # combinatorially large row count for a large horizon list. A small
    # max_skip (2-3) widens delta coverage cheaply, and -- since every skip
    # from the same start horizon shares the one encoder pass -- costs only
    # one extra `forward_at()` decode per skip, not a full extra encoder
    # pass.
    #
    # `adjacent_pair_min_start_speed_mps` (default 0.0, no filtering):
    # excludes (via the mask, not a row drop) rows where the ball's speed
    # at the START horizon is below this real-m/s threshold -- a
    # near-stationary ball makes "predict approximately no further
    # movement" a near-trivial target, same rationale as excluding
    # already-resolved starts. Converted to normalized units via
    # `pitch_half_diag_m` -- the same "approximate real-unit stand-in"
    # scale used elsewhere in this function -- exact when
    # normalize_kinematics_by_base_pitch is true, an approximation
    # otherwise.
    adjacent_pair_max_skip = int(cfg.get("adjacent_pair_max_skip", 1))
    adjacent_pair_min_start_speed_mps = float(cfg.get("adjacent_pair_min_start_speed_mps", 0.0))
    min_start_speed_norm = adjacent_pair_min_start_speed_mps / pitch_half_diag_m if adjacent_pair_min_start_speed_mps > 0.0 else 0.0

    horizon_pass_enabled = autoencode_during_main_loop_enabled or (adjacent_pair_training_enabled and n_horizons > 1)
    resting_min_start_speed_norm = resting_min_start_speed_mps / pitch_half_diag_m
    resting_speed_norm = resting_speed_threshold_mps / pitch_half_diag_m
    # Crossing POSITION doesn't change with the pseudo-start horizon (it's
    # the same fixed (x, y) event regardless of where you're predicting
    # from -- see _build_horizon_bundle's docstring) -- sliced once here in
    # train_idx/val_idx's own row order so the "horizon" branch below can
    # index it with the same local row_idx it already uses for everything
    # else, instead of re-slicing ds.crossing_pos by absolute row on every
    # batch.
    crossing_pos_train = ds.crossing_pos[train_idx] if has_crossing_data else None
    crossing_pos_val = ds.crossing_pos[val_idx] if has_crossing_data and len(val_idx) > 0 else None

    horizon_bundle_train: dict = {}
    horizon_bundle_val: dict = {}
    if horizon_pass_enabled:
        horizon_bundle_train = _build_horizon_bundle(
            ds, train_idx, n_horizons, cfg["horizons_s"],
            adjacent_pair_training_enabled, adjacent_pair_max_skip, min_start_speed_norm,
            has_crossing_data, resting_min_start_speed_norm, resting_speed_norm,
        )
        if len(val_idx) > 0:
            horizon_bundle_val = _build_horizon_bundle(
                ds, val_idx, n_horizons, cfg["horizons_s"],
                adjacent_pair_training_enabled, adjacent_pair_max_skip, min_start_speed_norm,
                has_crossing_data, resting_min_start_speed_norm, resting_speed_norm,
            )
        if adjacent_pair_training_enabled and n_horizons > 1:
            n_pair_eligible = int(sum(mask.sum() for mask in horizon_bundle_train["pair_mask"]))
            n_pair_combos = sum(len(targets_h) for targets_h in horizon_bundle_train["pair_targets"])
            log.info(
                f"Adjacent-pair training enabled: {n_horizons - 1} start-horizon(s), max_skip={adjacent_pair_max_skip} "
                f"({n_pair_combos} (start, skip) combos total), min_start_speed={adjacent_pair_min_start_speed_mps:.2f}m/s, "
                f"{n_pair_eligible:,}/{len(train_idx) * (n_horizons - 1):,} (horizon, row) combos mask-eligible "
                f"(shares rows/batches with autoencode/t0 -- no separate rows of its own anymore)"
            )

    # Row-count summary: how many training EXAMPLES each source contributes
    # this run, logged once up front (after every source above has been
    # built) so their very different contributions are visible at a glance
    # instead of needing to be pieced together from several separate log
    # lines. "main" processes its own dedicated rows as its own batches.
    # The "horizon" pass (t0/pair/crossing/resting, all riding the SAME
    # per-(row, horizon) shared encode -- see _build_horizon_bundle's and
    # _run_interleaved_train_epoch's "horizon" branch docstrings) processes
    # ONE set of rows per recorded horizon, with pair/crossing/resting each
    # masking their OWN eligible subset of that same shared set rather than
    # adding separate rows. crossing_head/resting_head at t=0 (used only by
    # the "main" branch, a SEPARATE usage from their horizon-generalized
    # counterparts above) are a masked subset of the MAIN row count instead.
    n_autoencode_train = sum(len(inp) for inp, _ in autoencode_train_data) if autoencode_train_data else 0
    n_crossing_valid = int(ds.crossing_mask[train_idx].sum()) if has_crossing_data else 0
    n_resting_valid = int(resting_mask[train_idx].sum())
    horizon_lines = ""
    if horizon_pass_enabled:
        n_pair_eligible = (
            int(sum(mask.sum() for mask in horizon_bundle_train["pair_mask"]))
            if adjacent_pair_training_enabled and n_horizons > 1 else 0
        )
        n_crossing_h_valid = (
            int(sum(v.sum() for v in horizon_bundle_train["crossing_valid"])) if has_crossing_data else 0
        )
        n_resting_h_valid = int(sum(m.sum() for m in horizon_bundle_train["resting_mask"]))
        horizon_lines = (
            f"    autoencode/t0 (bottleneck recon): {n_autoencode_train:,} rows -- own batches "
            f"({len(train_idx):,} rows x {n_horizons} horizons)\n"
            f"    adjacent-pair (dynamics)        : {n_pair_eligible:,}/{n_autoencode_train:,} horizon-pass rows mask-eligible -- "
            f"shares the horizon pass's own latent, no extra rows/batches\n"
        )
        if has_crossing_data:
            horizon_lines += (
                f"    crossing_head (at each horizon) : {n_crossing_h_valid:,}/{n_autoencode_train:,} horizon-pass rows mask-eligible -- "
                f"shares the horizon pass's own latent, no extra rows/batches\n"
            )
        horizon_lines += (
            f"    resting_head (at each horizon)  : {n_resting_h_valid:,}/{n_autoencode_train:,} horizon-pass rows mask-eligible -- "
            f"shares the horizon pass's own latent, no extra rows/batches\n"
        )
    log.info(
        "Training row-count summary (train split):\n"
        f"    main (per-horizon heads)        : {len(train_idx):,} rows -- own batches\n"
        f"{horizon_lines}"
        f"    crossing_head (at t=0, in main) : {n_crossing_valid:,}/{len(train_idx):,} main rows masked-valid (position term only; "
        f"delta_t trains on all {len(train_idx):,}) -- NO extra rows/batches, shares main's own latent\n"
        f"    resting_head (at t=0, in main)  : {n_resting_valid:,}/{len(train_idx):,} main rows masked-valid -- "
        f"NO extra rows/batches, shares main's own latent"
    )

    # Cosine annealing with warm restarts (SGDR): LR decays smoothly to
    # eta_min over T_0 epochs, then jumps back up and repeats (each cycle
    # T_mult times longer than the last). Motivated by train loss
    # oscillating-but-occasionally-improving late in training -- a sign the
    # LR is too coarse for the local curvature near the minimum; decaying
    # within each cycle lets it actually settle instead of bouncing, while
    # the restarts still give it a chance to escape a bad local spot.
    # lr_cosine_restart_epochs = null/0 disables this entirely (flat LR,
    # prior behaviour).
    # lr_cosine_peak_decay < 1.0: PyTorch's CosineAnnealingWarmRestarts
    # always restarts back to the SAME peak LR every cycle (the vanilla
    # SGDR default) -- it has no built-in option to also decay the peak
    # across restarts. We do that manually below by scaling
    # `scheduler.base_lrs` (what the scheduler actually reads as eta_max
    # each cycle -- changing optimizer.param_groups[...]['lr'] directly
    # would NOT affect it) every time a restart is detected (T_cur wraps
    # back to 0). 1.0 = no decay (peak stays constant forever, prior
    # behaviour).
    lr_cosine_peak_decay = float(cfg.get("lr_cosine_peak_decay", 1.0))
    lr_cosine_eta_min_frac = float(cfg.get("lr_cosine_eta_min_frac", 0.0))
    lr_cosine_restart_epochs = cfg.get("lr_cosine_restart_epochs")
    scheduler = None
    if lr_cosine_restart_epochs:
        eta_min = lr * lr_cosine_eta_min_frac
        t_mult = int(cfg.get("lr_cosine_t_mult", 1))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=int(lr_cosine_restart_epochs), T_mult=t_mult, eta_min=eta_min,
        )
        log.info(
            f"LR schedule: cosine warm restarts, T_0={lr_cosine_restart_epochs} epochs, "
            f"T_mult={t_mult}, eta_min={eta_min:.2e}, peak_decay={lr_cosine_peak_decay}"
        )

    early_stop_patience = int(cfg.get("early_stop_patience", 5))
    early_stop_min_delta = float(cfg.get("early_stop_min_delta", 1e-4))
    early_stop_enabled = early_stop_patience > 0 and len(val_idx) > 0

    best_val_loss = float("inf")
    best_state: dict | None = None
    patience_ctr = 0
    stopped_early = False
    prev_val_loss = float("nan")

    # Full per-epoch history (every epoch, not just the log tail) -- saved
    # alongside the checkpoint below (as .history.npz) so it can be
    # inspected/plotted after the fact without re-parsing terminal output.
    history: list[dict] = []

    n_h = len(cfg["horizons_s"])
    _CLS_METRICS = ("accuracy", "precision", "recall")
    _CLS_KEYS = tuple(f"{event}_{m}" for event in ("oob", "goal") for m in _CLS_METRICS)
    _R2_KEYS = tuple(f"{g}_r2" for g in _GROUPS)
    _PCTD_KEYS = tuple(f"{g}_err_pct_disp" for g in _GROUPS)
    _PCTB_KEYS = tuple(f"{g}_err_pct_ballistic" for g in _GROUPS)

    # Per-epoch diagnostic printing (not the history/report data, which keeps
    # every key regardless) skips oob/goal classification metrics when
    # bce_weight is 0 and spin metrics when spin_weight is 0 -- same
    # reasoning as `_LOG_COMPONENTS` above: a head with no gradient has
    # nothing new to show epoch over epoch.
    _LOG_CLS_KEYS = _CLS_KEYS if bce_weight != 0.0 else ()
    _LOG_R2_KEYS = tuple(c for c in _R2_KEYS if not (c == "spin_r2" and spin_weight == 0.0))
    _LOG_PCTD_KEYS = tuple(c for c in _PCTD_KEYS if not (c == "spin_err_pct_disp" and spin_weight == 0.0))
    _LOG_PCTB_KEYS = tuple(c for c in _PCTB_KEYS if not (c == "spin_err_pct_ballistic" and spin_weight == 0.0))

    def _mean_breakdown(items: list[LossBreakdown]) -> dict[str, np.ndarray]:
        if not items:
            return {c: np.full(n_h, np.nan) for c in _COMPONENTS}
        return {c: np.mean([getattr(b, c) for b in items], axis=0) for c in _COMPONENTS}

    def _classification_from_counts(oob_counts: np.ndarray, goal_counts: np.ndarray) -> dict[str, np.ndarray]:
        oob_m = _classification_metrics(oob_counts)
        goal_m = _classification_metrics(goal_counts)
        return {
            **{f"oob_{m}": oob_m[m] for m in _CLS_METRICS},
            **{f"goal_{m}": goal_m[m] for m in _CLS_METRICS},
        }

    def _r2_from_group_sums(sq_err: dict[str, np.ndarray], n: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {f"{g}_r2": _r2_from_sq_err(sq_err[g], n[g], target_var[g]) for g in _GROUPS}

    def _pct_disp_from_group_sums(sq_err: dict[str, np.ndarray], n: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {f"{g}_err_pct_disp": _pct_of_baseline_from_sq_err(sq_err[g], n[g], persistence_mse[g]) for g in _GROUPS}

    def _pct_ballistic_from_group_sums(sq_err: dict[str, np.ndarray], n: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {f"{g}_err_pct_ballistic": _pct_of_baseline_from_sq_err(sq_err[g], n[g], ballistic_mse[g]) for g in _GROUPS}

    def _eval_horizon_pass(
        autoencode_data: list[tuple[np.ndarray, np.ndarray]], horizon_bundle: dict, crossing_pos_here: np.ndarray | None,
    ) -> dict:
        """Eval-only (no gradient) counterpart to `_run_interleaved_train_
        epoch`'s "horizon" branch -- same one-encode-many-losses structure
        (t0 reconstruction, every pair skip masked, crossing masked/
        horizon-adjusted, resting masked/horizon-adjusted), just without
        any backward/step, and a plain sequential pass over every horizon
        (no interleaving needed here -- eval has no gradient updates for a
        fixed order to bias, unlike the training side). Reused identically
        by the main loop's val pass and decoder-only-pretrain's val pass.

        t0's R² is a valid comparison against `target_var` (the train-mean
        baseline) here, since the t0 target is exactly the recorded state
        at that horizon -- same quantity `target_var` was computed over.
        Deliberately does NOT compute err_pct_disp/err_pct_ballistic for t0:
        those baselines assume the ORIGINAL episode's t=0 state as the
        prediction's starting point, but t0's "input" is horizon h's OWN
        recorded state reused as a pseudo-initial-state -- reusing those
        baselines here would silently compare against the wrong reference
        point.

        Returns a dict: ``mean_t0_loss``, ``t0_means``, ``t0_r2``, ``t0_cls``
        (t0's own diagnostics, unrelated to anything else); ``mean_pair_loss``
        (masked mean pair loss); and RAW per-batch lists --
        ``crossing_losses``/``crossing_pos_dists``/``crossing_dt_maes``,
        ``resting_losses``/``resting_pos_dists`` -- deliberately NOT
        pre-averaged here, since callers blend these into the SAME
        accumulator as the main task's own t=0 crossing/resting usage
        (matching the training side's convention of one combined
        `train_crossing_losses`/`train_resting_losses` list regardless of
        which pseudo-start produced each entry).
        """
        losses_t0: list[float] = []
        breakdowns_by_h: list[list[dict]] = [[] for _ in range(n_h)]
        sq_err = {g: np.zeros(n_h) for g in _GROUPS}
        n_count = {g: np.zeros(n_h, dtype=np.int64) for g in _GROUPS}
        oob_counts = np.zeros((n_h, 4), dtype=np.int64)
        goal_counts = np.zeros((n_h, 4), dtype=np.int64)
        pair_losses: list[float] = []
        crossing_losses: list[float] = []
        crossing_pos_dists: list[float] = []
        crossing_dt_maes: list[float] = []
        resting_losses: list[float] = []
        resting_pos_dists: list[float] = []
        position_losses: list[float] = []
        position_pos_dists: list[float] = []

        with torch.no_grad():
            for h_idx, (ae_inputs, ae_targets) in enumerate(autoencode_data):
                n_rows = len(ae_inputs)
                for start in range(0, n_rows, batch_size):
                    row_idx = np.arange(start, min(start + batch_size, n_rows))
                    x = torch.from_numpy(ae_inputs[row_idx].astype(np.float32, copy=False)).to(device)
                    latent = model.encoder(x)

                    if autoencode_during_main_loop_enabled:
                        y = torch.from_numpy(ae_targets[row_idx].astype(np.float32, copy=False)).to(device)
                        pw_row = pos_weight[h_idx]
                        pred = model.decoder.forward_at(latent, 0.0)
                        t0_loss, breakdown = _single_target_loss_with_breakdown(pred, y, pw_row, bce_weight, spin_weight)
                        losses_t0.append(float(t0_loss.item()))
                        breakdowns_by_h[h_idx].append(breakdown)
                        sq_err_counts = compute_group_sq_err([pred], y)
                        conf_counts = compute_confusion_counts([pred], y)
                        for g in _GROUPS:
                            se, cnt = sq_err_counts[g][0]
                            sq_err[g][h_idx] += se
                            n_count[g][h_idx] += cnt
                        oob_counts[h_idx] += np.array(conf_counts["oob"][0])
                        goal_counts[h_idx] += np.array(conf_counts["goal"][0])

                    pair_targets_h = horizon_bundle["pair_targets"][h_idx] if adjacent_pair_training_enabled else []
                    if pair_targets_h:
                        mask_f = torch.from_numpy(horizon_bundle["pair_mask"][h_idx][row_idx]).to(device).float()
                        denom = mask_f.sum().clamp_min(1.0)
                        pair_loss = x.new_zeros(())
                        for skip, p_targets in pair_targets_h:
                            y_pair = torch.from_numpy(p_targets[row_idx]).to(device)
                            delta = cfg["horizons_s"][h_idx + skip] - cfg["horizons_s"][h_idx]
                            pred_pair = model.decoder.forward_at(latent, delta)
                            pw_row = pos_weight[h_idx + skip]
                            per_ex_loss = _single_target_per_episode_loss(pred_pair, y_pair, pw_row, bce_weight, spin_weight)
                            pair_loss = pair_loss + (per_ex_loss * mask_f).sum() / denom
                        pair_losses.append(float(pair_loss.item()))

                    if has_crossing_data:
                        crossing_loss_h, pos_dist_mean_h, dt_mae_mean_h = _crossing_head_loss(
                            model, latent, crossing_pos_here, horizon_bundle["crossing_dt"][h_idx],
                            horizon_bundle["crossing_valid"][h_idx], row_idx, device,
                        )
                        crossing_losses.append(float(crossing_loss_h.item()))
                        crossing_pos_dists.append(pos_dist_mean_h)
                        crossing_dt_maes.append(dt_mae_mean_h)

                    resting_loss_h, resting_pos_dist_mean_h = _resting_head_loss(
                        model, latent, horizon_bundle["resting_pos"][h_idx], horizon_bundle["resting_mask"][h_idx],
                        row_idx, device,
                    )
                    resting_losses.append(float(resting_loss_h.item()))
                    resting_pos_dists.append(resting_pos_dist_mean_h)

                    position_loss_h, position_pos_dist_mean_h = _position_head_loss(model, latent, x)
                    position_losses.append(float(position_loss_h.item()))
                    position_pos_dists.append(position_pos_dist_mean_h)

        return {
            "mean_t0_loss": float(np.mean(losses_t0)) if losses_t0 else float("nan"),
            "t0_means": _mean_breakdown_by_horizon(breakdowns_by_h),
            "t0_r2": _r2_from_group_sums(sq_err, n_count),
            "t0_cls": _classification_from_counts(oob_counts, goal_counts),
            "mean_pair_loss": float(np.mean(pair_losses)) if pair_losses else float("nan"),
            "crossing_losses": crossing_losses,
            "crossing_pos_dists": crossing_pos_dists,
            "crossing_dt_maes": crossing_dt_maes,
            "resting_losses": resting_losses,
            "resting_pos_dists": resting_pos_dists,
            "position_losses": position_losses,
            "position_pos_dists": position_pos_dists,
        }

    def _log_t0_diagnostics(label: str, means: dict[str, np.ndarray], r2: dict[str, np.ndarray], cls: dict[str, np.ndarray] | None) -> None:
        # Only pos_rmse gets logged (r2/cls args still accepted/computed by
        # callers via _run_t0_pass -- unused here, kept for the loss/scalar
        # returns those calls also need -- but no longer printed; the
        # vel_rmse/spin_rmse/oob_bce/goal_bce/r2/classification breakdown
        # was too much per-epoch noise to scan).
        _log_component(f"{label} t0", "pos_rmse", means["pos_rmse"])

    def _run_interleaved_train_epoch(optimizer: torch.optim.Optimizer) -> dict:
        """Runs ONE epoch's worth of TRAINING gradient steps, drawn from
        every enabled source -- "main" (the original t=0 episode inputs,
        predicting every recorded horizon + crossing/resting at t=0) and
        "horizon" (one shared encode per recorded horizon's pseudo-start,
        feeding t0 reconstruction, every pair skip, and the horizon-
        generalized crossing/resting -- see `_build_horizon_bundle`'s and
        the "horizon" branch's own docstrings) -- in a single SHUFFLED
        order, rather than as separate sequential passes. Under a fixed-
        order structure, whichever pass ran LAST in the epoch was always
        trained against a model every OTHER pass had already updated that
        epoch, and whichever ran FIRST never benefited from the others'
        updates -- the exact same systematic (not noise-averaging-out) bias
        `_interleaved_horizon_batches` already fixes ACROSS HORIZONS within
        one pass, just one level up: across PASS-TYPE too. Metrics are
        still tracked and returned completely SEPARATELY per source
        (main/pair/t0/crossing/resting) -- only the ORDER gradient steps
        happen in is merged, not what they measure or how they're reported.

        Shared by both the main loop and decoder-only pretraining (pass in
        whichever `optimizer` that phase uses) -- their per-epoch training
        structure is otherwise identical, just over different trainable
        parameters/learning rates.

        Returns a dict with everything callers previously computed inline:
        ``mean_loss``, ``breakdowns`` (list[LossBreakdown], for
        ``_mean_breakdown``), ``oob_counts``/``goal_counts``, ``sq_err``/
        ``n`` (for R²/pct-baseline), ``mean_pair_loss``, ``mean_t0_loss``,
        ``t0_means`` (by horizon, for ``_log_t0_diagnostics``), ``t0_r2``,
        ``mean_crossing_loss``, ``crossing_pos_dist``, ``crossing_dt_mae``
        (the last two: mean over the epoch, NaN if ``has_crossing_data`` is
        False), ``mean_backprop_loss`` (the actual combined objective
        stepped each "main" iteration -- ``mean_loss`` plus the weighted
        crossing term when present; equal to ``mean_loss`` when
        ``has_crossing_data`` is False).
        """
        train_losses: list[float] = []
        train_breakdowns: list[LossBreakdown] = []
        train_oob_counts = np.zeros((n_h, 4), dtype=np.int64)
        train_goal_counts = np.zeros((n_h, 4), dtype=np.int64)
        train_crossing_losses: list[float] = []
        train_crossing_pos_dist: list[float] = []
        train_crossing_dt_mae: list[float] = []
        train_resting_losses: list[float] = []
        train_resting_pos_dist: list[float] = []
        train_position_losses: list[float] = []
        train_position_pos_dist: list[float] = []
        train_event_losses: list[float] = []
        train_event_oob_acc: list[float] = []
        train_event_goal_acc: list[float] = []
        train_backprop_losses: list[float] = []
        train_grad_norms: list[float] = []
        train_sq_err = {g: np.zeros(n_h) for g in _GROUPS}
        train_n = {g: np.zeros(n_h, dtype=np.int64) for g in _GROUPS}
        train_pair_losses: list[float] = []
        t0_losses: list[float] = []
        t0_breakdowns_by_h: list[list[dict]] = [[] for _ in range(n_h)]
        t0_sq_err = {g: np.zeros(n_h) for g in _GROUPS}
        t0_n_count = {g: np.zeros(n_h, dtype=np.int64) for g in _GROUPS}

        # Build every source's minibatch-sized row-index chunks up front
        # (main: shuffled chunks of `train_idx` itself; pair/t0: reuse
        # `_interleaved_horizon_batches`' own per-item shuffling), tag each
        # with which source/item it came from, then shuffle the COMBINED
        # list's order across all sources together.
        combined: list[tuple[str, int, np.ndarray]] = []
        main_order = rng.permutation(len(train_idx))
        for start in range(0, len(main_order), batch_size):
            chunk = main_order[start:start + batch_size]
            if len(chunk) > 0:
                combined.append(("main", 0, chunk))
        if horizon_pass_enabled:
            for h_idx, row_idx in _interleaved_horizon_batches(autoencode_train_data, batch_size, rng):
                combined.append(("horizon", h_idx, row_idx))
        shuffle_order = rng.permutation(len(combined))
        combined = [combined[i] for i in shuffle_order]

        for kind, item_idx, row_idx in combined:
            if kind == "main":
                batch_idx = train_idx[row_idx]
                x = torch.from_numpy(ds.inputs[batch_idx].astype(np.float32, copy=False)).to(device)
                y = torch.from_numpy(ds.targets[batch_idx].astype(np.float32, copy=False)).to(device)
                latent, heads = model(x)
                loss, breakdown = compute_loss(heads, y, pos_weight, bce_weight, spin_weight)
                # `backprop_loss` (not `loss`) is what actually gets stepped --
                # folds in the crossing-head term for a single combined
                # backward pass (cheaper than a separate step, and the
                # encoder/latent are shared anyway), but `train_losses` below
                # keeps reporting the main per-horizon `loss` ALONE, same
                # convention as pair/t0 (always separate metrics, never
                # merged into train_loss/val_loss) -- otherwise train_loss
                # would include the crossing term while val_loss doesn't
                # (val_loss is computed independently below, deliberately
                # NOT reusing this combined backprop value), making the two
                # look wildly different for a reason that has nothing to do
                # with over/underfitting.
                backprop_loss = loss
                if has_crossing_data:
                    crossing_loss, crossing_pos_dist_mean, crossing_dt_mae_mean = _crossing_head_loss(
                        model, latent, ds.crossing_pos, ds.crossing_dt, ds.crossing_mask, batch_idx, device,
                    )
                    backprop_loss = backprop_loss + crossing_weight * crossing_loss
                    train_crossing_losses.append(float(crossing_loss.item()))
                    train_crossing_pos_dist.append(crossing_pos_dist_mean)
                    train_crossing_dt_mae.append(crossing_dt_mae_mean)
                # resting_head reuses the SAME `latent` computed above for
                # the main heads/crossing_head -- it's the same input row,
                # so no extra encoder call needed (see resting_head's
                # docstring).
                resting_loss, resting_pos_dist_mean = _resting_head_loss(
                    model, latent, resting_pos, resting_mask, batch_idx, device,
                )
                backprop_loss = backprop_loss + resting_weight * resting_loss
                train_resting_losses.append(float(resting_loss.item()))
                train_resting_pos_dist.append(resting_pos_dist_mean)
                # position_head reuses the SAME `latent`/`x` as everything
                # else above -- no dataset lookup, no extra encoder call.
                position_loss, position_pos_dist_mean = _position_head_loss(model, latent, x)
                backprop_loss = backprop_loss + position_weight * position_loss
                train_position_losses.append(float(position_loss.item()))
                train_position_pos_dist.append(position_pos_dist_mean)
                # event_head is t=0 ONLY -- no horizon-branch counterpart,
                # see event_head's docstring.
                event_loss, event_oob_acc, event_goal_acc = _event_head_loss(
                    model, latent, ever_oob, ever_goal, batch_idx, device,
                )
                backprop_loss = backprop_loss + event_weight * event_loss
                train_event_losses.append(float(event_loss.item()))
                train_event_oob_acc.append(event_oob_acc)
                train_event_goal_acc.append(event_goal_acc)
                optimizer.zero_grad()
                backprop_loss.backward()
                # Total gradient norm across every trainable param, BEFORE
                # optimizer.step() consumes it -- a direct read of how large
                # this step's raw update direction is, independent of `lr`.
                # `clip_grad_norm_` with max_norm=inf computes (and returns)
                # the norm without ever actually clipping/rescaling
                # anything -- the standard portable way to just measure it.
                # Diagnostic only: distinguishes "the step size is too big
                # for the local curvature" (large, noisy grad norms -- try
                # a smaller lr/eta_min_frac) from "genuinely converged,
                # just slow" (small, stable grad norms -- more epochs is
                # the only lever left).
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf")).item())
                train_grad_norms.append(grad_norm)
                optimizer.step()
                train_losses.append(float(loss.item()))
                train_backprop_losses.append(float(backprop_loss.item()))
                train_breakdowns.append(breakdown)
                with torch.no_grad():
                    counts = compute_confusion_counts(heads, y)
                    sq_err_counts = compute_group_sq_err(heads, y)
                train_oob_counts += np.array(counts["oob"])
                train_goal_counts += np.array(counts["goal"])
                for g in _GROUPS:
                    for h, (sq_err, n) in enumerate(sq_err_counts[g]):
                        train_sq_err[g][h] += sq_err
                        train_n[g][h] += n
            else:  # "horizon" -- ONE encoder pass for this recorded
                # horizon's shared pseudo-input, feeding whichever of
                # {t0 reconstruction, every pair skip (masked), crossing
                # (masked/horizon-adjusted), resting (masked/horizon-
                # adjusted)} are enabled, all summed into ONE gradient step
                # -- see _build_horizon_bundle's docstring for how the
                # pair/crossing/resting targets were derived, and why pair's
                # old row-filter had to become a mask so every task here can
                # share the exact same row set.
                h_idx = item_idx
                ae_inputs, ae_targets = autoencode_train_data[h_idx]
                x = torch.from_numpy(ae_inputs[row_idx].astype(np.float32, copy=False)).to(device)
                latent = model.encoder(x)
                combined_loss = x.new_zeros(())

                if autoencode_during_main_loop_enabled:
                    y = torch.from_numpy(ae_targets[row_idx].astype(np.float32, copy=False)).to(device)
                    pw_row = pos_weight[h_idx]
                    pred = model.decoder.forward_at(latent, 0.0)
                    t0_loss, breakdown = _single_target_loss_with_breakdown(pred, y, pw_row, bce_weight, spin_weight)
                    combined_loss = combined_loss + t0_loss
                    t0_losses.append(float(t0_loss.item()))
                    t0_breakdowns_by_h[h_idx].append(breakdown)
                    with torch.no_grad():
                        sq_err_counts = compute_group_sq_err([pred], y)
                    for g in _GROUPS:
                        se, cnt = sq_err_counts[g][0]
                        t0_sq_err[g][h_idx] += se
                        t0_n_count[g][h_idx] += cnt

                pair_targets_h = horizon_bundle_train["pair_targets"][h_idx] if adjacent_pair_training_enabled else []
                if pair_targets_h:
                    mask_f = torch.from_numpy(horizon_bundle_train["pair_mask"][h_idx][row_idx]).to(device).float()
                    denom = mask_f.sum().clamp_min(1.0)
                    pair_loss = x.new_zeros(())
                    for skip, p_targets in pair_targets_h:
                        y_pair = torch.from_numpy(p_targets[row_idx]).to(device)
                        delta = cfg["horizons_s"][h_idx + skip] - cfg["horizons_s"][h_idx]
                        pred_pair = model.decoder.forward_at(latent, delta)
                        pw_row = pos_weight[h_idx + skip]
                        per_ex_loss = _single_target_per_episode_loss(pred_pair, y_pair, pw_row, bce_weight, spin_weight)
                        pair_loss = pair_loss + (per_ex_loss * mask_f).sum() / denom
                    combined_loss = combined_loss + pair_loss
                    train_pair_losses.append(float(pair_loss.item()))

                if has_crossing_data:
                    crossing_loss_h, pos_dist_mean_h, dt_mae_mean_h = _crossing_head_loss(
                        model, latent, crossing_pos_train, horizon_bundle_train["crossing_dt"][h_idx],
                        horizon_bundle_train["crossing_valid"][h_idx], row_idx, device,
                    )
                    combined_loss = combined_loss + crossing_weight * crossing_loss_h
                    train_crossing_losses.append(float(crossing_loss_h.item()))
                    train_crossing_pos_dist.append(pos_dist_mean_h)
                    train_crossing_dt_mae.append(dt_mae_mean_h)

                resting_loss_h, resting_pos_dist_mean_h = _resting_head_loss(
                    model, latent, horizon_bundle_train["resting_pos"][h_idx], horizon_bundle_train["resting_mask"][h_idx],
                    row_idx, device,
                )
                combined_loss = combined_loss + resting_weight * resting_loss_h
                train_resting_losses.append(float(resting_loss_h.item()))
                train_resting_pos_dist.append(resting_pos_dist_mean_h)

                position_loss_h, position_pos_dist_mean_h = _position_head_loss(model, latent, x)
                combined_loss = combined_loss + position_weight * position_loss_h
                train_position_losses.append(float(position_loss_h.item()))
                train_position_pos_dist.append(position_pos_dist_mean_h)

                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

        return {
            "mean_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "breakdowns": train_breakdowns,
            "oob_counts": train_oob_counts,
            "goal_counts": train_goal_counts,
            "sq_err": train_sq_err,
            "n": train_n,
            "mean_pair_loss": float(np.mean(train_pair_losses)) if train_pair_losses else float("nan"),
            "mean_t0_loss": float(np.mean(t0_losses)) if t0_losses else float("nan"),
            "t0_means": _mean_breakdown_by_horizon(t0_breakdowns_by_h),
            "t0_r2": _r2_from_group_sums(t0_sq_err, t0_n_count),
            "mean_crossing_loss": float(np.mean(train_crossing_losses)) if train_crossing_losses else float("nan"),
            "crossing_pos_dist": float(np.mean(train_crossing_pos_dist)) if train_crossing_pos_dist else float("nan"),
            "crossing_dt_mae": float(np.mean(train_crossing_dt_mae)) if train_crossing_dt_mae else float("nan"),
            "mean_resting_loss": float(np.mean(train_resting_losses)) if train_resting_losses else float("nan"),
            "resting_pos_dist": float(np.mean(train_resting_pos_dist)) if train_resting_pos_dist else float("nan"),
            "mean_position_loss": float(np.mean(train_position_losses)) if train_position_losses else float("nan"),
            "position_pos_dist": float(np.mean(train_position_pos_dist)) if train_position_pos_dist else float("nan"),
            "mean_event_loss": float(np.mean(train_event_losses)) if train_event_losses else float("nan"),
            "event_oob_acc": float(np.mean(train_event_oob_acc)) if train_event_oob_acc else float("nan"),
            "event_goal_acc": float(np.mean(train_event_goal_acc)) if train_event_goal_acc else float("nan"),
            "mean_backprop_loss": float(np.mean(train_backprop_losses)) if train_backprop_losses else float("nan"),
            "grad_norm_stats": _summary_stats(train_grad_norms),
            # Batch-to-batch CHANGE in the main task's own per-batch loss,
            # in the order those "main" gradient steps occurred (other
            # interleaved kinds don't break the sequence -- np.diff just
            # looks at consecutive entries of `train_losses` itself).
            # Dominated by batch-composition noise more than true
            # optimization progress at this loss scale, so read it as "how
            # noisy/bumpy is the trajectory," not "how fast is it improving"
            # -- see the gradient-norm stats above for the more direct
            # step-size-vs-curvature signal.
            "loss_delta_stats": _summary_stats(list(np.diff(train_losses))) if len(train_losses) > 1 else _summary_stats([]),
        }

    # Decoder-only pretraining: an optional phase, AFTER autoencode
    # pretraining and BEFORE the main joint loop, that trains ONLY the
    # decoder (encoder frozen exactly as autoencode pretraining left it) on
    # the REAL per-horizon dynamics task -- the same training the main loop
    # does (main per-horizon loss + adjacent-pair + t=0 term), just without
    # letting the decoder's initially-large, far-from-converged dynamics
    # loss drag the encoder's already-good identity mapping around. Same
    # failure mode already fixed for classification via
    # BallDynamicsDecoder's identity-shortcut masks, one level further out:
    # a freshly-initialized "head" (here, the decoder's dynamics
    # capability) generating a much bigger gradient than an already-good
    # "backbone" (the encoder) can safely absorb on epoch 1 of joint
    # training. The encoder was already trained to represent the initial
    # state accurately (autoencode pretraining + identity shortcut), which
    # is a reasonable starting point for a ROUGH dynamics guess even before
    # any decoder-only training happens. 0 (default) = disabled, no
    # behaviour change. No LR schedule/early-stopping/best-state tracking
    # here (same convention as autoencode pretraining) -- deliberately a
    # short warm-up, not a full training run in its own right. Full
    # diagnostic logging (matches a regular main-loop epoch exactly, reusing
    # the same helpers/format), but NOT persisted to `history`/
    # `.history.npz` (also matching autoencode pretraining's precedent,
    # and keeping that array's "epoch" indices unambiguous relative to
    # `epochs`).
    decoder_only_pretrain_epochs = int(cfg.get("decoder_only_pretrain_epochs", 0))
    if decoder_only_pretrain_epochs > 0:
        decoder_only_lr = float(cfg.get("decoder_only_pretrain_lr", lr))
        # `decoder_only_pretrain_freeze_latent` (default false, no behaviour
        # change): whether `encoder.out` (the latent-producing layer) stays
        # frozen for this phase too, or is left trainable like the rest of
        # this block assumes by default.
        #
        # false (default) -- only the encoder's TRUNK (the input->hidden->
        # hidden->bottleneck layers, everything before the final Linear-to-
        # latent) stays frozen here; `encoder.out` is left trainable, so the
        # latent's non-identity ("spare") dims can start shaping themselves
        # toward whatever the decoder finds useful for real dynamics
        # prediction a whole phase earlier, instead of sitting frozen/
        # untouched until the main loop. The trunk stays frozen because the
        # identity-shortcut concat bypasses it entirely (see
        # BallDynamicsEncoder's docstring) -- it has no protective role to
        # play here, it's just extra capacity that doesn't need touching
        # yet. `encoder.out`'s IDENTITY rows (producing latent[0:N_IDENTITY_
        # SHORTCUT_FIELDS] ~= raw pos/vel/spin input) still need the same
        # protection the decoder's own dedicated units already have: this
        # phase's decoder is at its freshest/noisiest (that's the whole
        # reason decoder-only pretraining exists), and unlike the main
        # loop (where the decoder has already calmed down by the time the
        # encoder is exposed to it again), letting THOSE specific rows
        # train here would reopen exactly the corruption failure mode
        # this session already found once (pos_rmse blowing up when a
        # second, unrelated task could cheaply route gradient through an
        # identity-preserving weight block) -- just one layer further
        # upstream than before. A permanent hard mask (register_hook,
        # zeroing gradient unconditionally) on just those rows, added
        # here and removed again once this phase ends, matches the "no
        # soft version of this" precedent from BallDynamicsDecoder's own
        # identity-shortcut masking -- scoped to ONLY this phase, since
        # autoencode pretraining and the main loop already train
        # `encoder.out` fully unmasked and that's working as intended.
        #
        # true -- the ENTIRE encoder (trunk AND `encoder.out`) is frozen for
        # this phase, matching this pipeline's ORIGINAL (pre-this-session)
        # behaviour: the decoder trains purely against whatever latent the
        # encoder already produces, with zero risk of the decoder's fresh/
        # noisy gradient touching the encoder at all -- no identity-row
        # masking needed either, since nothing in the encoder is trainable
        # to begin with. Trade-off: the latent's spare dims sit completely
        # untouched through this whole phase (same trade-off the false path
        # was specifically added to avoid -- see the post-launch-fix note in
        # the design doc), so this is for A/B-testing whether letting the
        # latent move here actually helps, not a strictly-better default.
        freeze_latent = bool(cfg.get("decoder_only_pretrain_freeze_latent", False))
        if freeze_latent:
            for p in model.encoder.parameters():
                p.requires_grad_(False)
        else:
            for p in model.encoder.trunk.parameters():
                p.requires_grad_(False)
        encoder_out_hook_handles = []
        if not freeze_latent and model.encoder.identity_shortcut:
            dim = N_IDENTITY_SHORTCUT_FIELDS
            weight_mask = torch.ones_like(model.encoder.out.weight)
            weight_mask[:dim, :] = 0.0
            bias_mask = torch.ones_like(model.encoder.out.bias)
            bias_mask[:dim] = 0.0
            encoder_out_hook_handles.append(model.encoder.out.weight.register_hook(lambda grad: grad * weight_mask))
            encoder_out_hook_handles.append(model.encoder.out.bias.register_hook(lambda grad: grad * bias_mask))
        decoder_only_params = (
            list(model.decoder.parameters()) + list(model.crossing_head.parameters()) + list(model.resting_head.parameters())
            + list(model.position_head.parameters()) + list(model.event_head.parameters())
        )
        if not freeze_latent:
            decoder_only_params += list(model.encoder.out.parameters())
        decoder_only_optimizer = _build_phase_optimizer(
            decoder_only_params, decoder_only_lr, cfg, "decoder_only_optimizer_type", "decoder_only_sgd_momentum", "Decoder-only-pretrain",
        )
        log.info(
            f"Decoder-only pretraining: {decoder_only_pretrain_epochs} epoch(s), "
            + (
                "entire encoder frozen (latent fixed), "
                if freeze_latent
                else "encoder TRUNK frozen (encoder.out latent layer trainable, identity rows gradient-masked), "
            )
            + f"lr={decoder_only_lr:.2e}"
        )
        # Best-val tracking + restore-at-the-end, same as the main loop's
        # early-stopping mechanism below -- this phase has no LR schedule
        # or early stopping of its own (deliberately, see the docstring
        # above), so without this it just hands off whatever the model
        # looks like after the LITERAL LAST epoch to the main loop, even
        # if that epoch happened to be a bad one (observed directly in a
        # real run: epoch 20/20's val_loss spiked to ~4.5x the typical
        # level, and the main loop's very first epoch showed a
        # correspondingly elevated loss consistent with starting from
        # that bad state instead of a better earlier one).
        # Early stopping for THIS phase specifically -- separate knobs from
        # the main loop's early_stop_patience/early_stop_min_delta above,
        # since decoder-only pretraining is typically a much shorter phase
        # (few-to-tens of epochs) with its own LR and dynamics, so the same
        # patience count means something different here. Best-val tracking
        # (best_val_loss_do/best_state_do) always happens regardless of
        # whether this is enabled (see the comment above) -- this only adds
        # an early BREAK out of the epoch loop once val has stopped
        # improving, so a short decoder_only_pretrain_epochs budget that's
        # mostly wasted after convergence doesn't have to be shortened by
        # hand. 0 (default) = disabled, no behaviour change (always runs
        # the full configured epoch count, same as before this was added).
        decoder_only_early_stop_patience = int(cfg.get("decoder_only_early_stop_patience", 0))
        decoder_only_early_stop_min_delta = float(cfg.get("decoder_only_early_stop_min_delta", 1e-4))
        decoder_only_early_stop_enabled = decoder_only_early_stop_patience > 0 and len(val_idx) > 0
        decoder_only_patience_ctr = 0

        best_val_loss_do = float("inf")
        best_state_do: dict | None = None
        for do_epoch in range(decoder_only_pretrain_epochs):
            model.train()
            epoch_result_do = _run_interleaved_train_epoch(decoder_only_optimizer)
            mean_train_loss_do = epoch_result_do["mean_loss"]
            mean_train_pair_loss_do = epoch_result_do["mean_pair_loss"]
            mean_train_t0_loss_do = epoch_result_do["mean_t0_loss"]
            train_t0_means_do = epoch_result_do["t0_means"]
            train_t0_r2_do = epoch_result_do["t0_r2"]
            mean_train_crossing_loss_do = epoch_result_do["mean_crossing_loss"]
            train_crossing_pos_dist_do = epoch_result_do["crossing_pos_dist"]
            train_crossing_dt_mae_do = epoch_result_do["crossing_dt_mae"]
            mean_train_resting_loss_do = epoch_result_do["mean_resting_loss"]
            train_resting_pos_dist_do = epoch_result_do["resting_pos_dist"]
            mean_train_position_loss_do = epoch_result_do["mean_position_loss"]
            train_position_pos_dist_do = epoch_result_do["position_pos_dist"]
            mean_train_event_loss_do = epoch_result_do["mean_event_loss"]
            train_event_oob_acc_do = epoch_result_do["event_oob_acc"]
            train_event_goal_acc_do = epoch_result_do["event_goal_acc"]
            mean_train_backprop_loss_do = epoch_result_do["mean_backprop_loss"]

            train_means_do = _mean_breakdown(epoch_result_do["breakdowns"])
            train_r2_do = _r2_from_group_sums(epoch_result_do["sq_err"], epoch_result_do["n"])
            train_pct_disp_do = _pct_disp_from_group_sums(epoch_result_do["sq_err"], epoch_result_do["n"])
            train_pct_ballistic_do = _pct_ballistic_from_group_sums(epoch_result_do["sq_err"], epoch_result_do["n"])

            val_line_do = ""
            mean_val_loss_do = float("nan")
            mean_val_pair_loss_do = float("nan")
            mean_val_t0_loss_do = float("nan")
            mean_val_crossing_loss_do = float("nan")
            val_crossing_pos_dist_do = float("nan")
            val_crossing_dt_mae_do = float("nan")
            mean_val_resting_loss_do = float("nan")
            val_resting_pos_dist_do = float("nan")
            mean_val_position_loss_do = float("nan")
            val_position_pos_dist_do = float("nan")
            mean_val_event_loss_do = float("nan")
            val_event_oob_acc_do = float("nan")
            val_event_goal_acc_do = float("nan")
            mean_val_backprop_loss_do = float("nan")
            val_means_do = _mean_breakdown([])
            val_cls_do = _classification_from_counts(np.zeros((n_h, 4), dtype=np.int64), np.zeros((n_h, 4), dtype=np.int64))
            val_r2_do = _r2_from_group_sums({g: np.zeros(n_h) for g in _GROUPS}, {g: np.zeros(n_h, dtype=np.int64) for g in _GROUPS})
            val_pct_disp_do = _pct_disp_from_group_sums({g: np.zeros(n_h) for g in _GROUPS}, {g: np.zeros(n_h, dtype=np.int64) for g in _GROUPS})
            val_pct_ballistic_do = _pct_ballistic_from_group_sums({g: np.zeros(n_h) for g in _GROUPS}, {g: np.zeros(n_h, dtype=np.int64) for g in _GROUPS})
            if len(val_idx) > 0:
                model.eval()
                val_losses_do: list[float] = []
                val_breakdowns_do: list[LossBreakdown] = []
                val_oob_counts_do = np.zeros((n_h, 4), dtype=np.int64)
                val_goal_counts_do = np.zeros((n_h, 4), dtype=np.int64)
                val_sq_err_do = {g: np.zeros(n_h) for g in _GROUPS}
                val_n_do = {g: np.zeros(n_h, dtype=np.int64) for g in _GROUPS}
                val_crossing_losses_do: list[float] = []
                val_crossing_pos_dists_do: list[float] = []
                val_crossing_dt_maes_do: list[float] = []
                val_resting_losses_do: list[float] = []
                val_resting_pos_dists_do: list[float] = []
                val_position_losses_do: list[float] = []
                val_position_pos_dists_do: list[float] = []
                val_event_losses_do: list[float] = []
                val_event_oob_accs_do: list[float] = []
                val_event_goal_accs_do: list[float] = []
                _val_pos_do = 0
                with torch.no_grad():
                    for x, y in ds.iterate_minibatches(batch_size, val_idx, shuffle=False, device=device):
                        latent, heads = model(x)
                        loss, breakdown = compute_loss(heads, y, pos_weight, bce_weight, spin_weight)
                        val_losses_do.append(float(loss.item()))
                        val_breakdowns_do.append(breakdown)
                        counts = compute_confusion_counts(heads, y)
                        val_oob_counts_do += np.array(counts["oob"])
                        val_goal_counts_do += np.array(counts["goal"])
                        sq_err_counts = compute_group_sq_err(heads, y)
                        for g in _GROUPS:
                            for h, (sq_err, n) in enumerate(sq_err_counts[g]):
                                val_sq_err_do[g][h] += sq_err
                                val_n_do[g][h] += n
                        batch_idx = val_idx[_val_pos_do:_val_pos_do + x.shape[0]]
                        if has_crossing_data:
                            crossing_loss, pos_dist_mean, dt_mae_mean = _crossing_head_loss(
                            model, latent, ds.crossing_pos, ds.crossing_dt, ds.crossing_mask, batch_idx, device,
                        )
                            val_crossing_losses_do.append(float(crossing_loss.item()))
                            val_crossing_pos_dists_do.append(pos_dist_mean)
                            val_crossing_dt_maes_do.append(dt_mae_mean)
                        resting_loss, resting_pos_dist_mean = _resting_head_loss(
                            model, latent, resting_pos, resting_mask, batch_idx, device,
                        )
                        val_resting_losses_do.append(float(resting_loss.item()))
                        val_resting_pos_dists_do.append(resting_pos_dist_mean)
                        position_loss, position_pos_dist_mean = _position_head_loss(model, latent, x)
                        val_position_losses_do.append(float(position_loss.item()))
                        val_position_pos_dists_do.append(position_pos_dist_mean)
                        event_loss, event_oob_acc, event_goal_acc = _event_head_loss(
                            model, latent, ever_oob, ever_goal, batch_idx, device,
                        )
                        val_event_losses_do.append(float(event_loss.item()))
                        val_event_oob_accs_do.append(event_oob_acc)
                        val_event_goal_accs_do.append(event_goal_acc)
                        _val_pos_do += x.shape[0]

                # Horizon-generalized pair/t0/crossing/resting -- one shared
                # eval pass (see _eval_horizon_pass's docstring); its
                # crossing/resting raw lists are BLENDED into the same
                # val_crossing_losses_do/val_resting_losses_do accumulators
                # the t=0 usage above just populated (matching the training
                # side's single shared accumulator convention), NOT reported
                # as separate metrics.
                horizon_result_do = (
                    _eval_horizon_pass(autoencode_val_data, horizon_bundle_val, crossing_pos_val)
                    if horizon_pass_enabled else None
                )
                if horizon_result_do is not None:
                    val_crossing_losses_do.extend(horizon_result_do["crossing_losses"])
                    val_crossing_pos_dists_do.extend(horizon_result_do["crossing_pos_dists"])
                    val_crossing_dt_maes_do.extend(horizon_result_do["crossing_dt_maes"])
                    val_resting_losses_do.extend(horizon_result_do["resting_losses"])
                    val_resting_pos_dists_do.extend(horizon_result_do["resting_pos_dists"])
                    val_position_losses_do.extend(horizon_result_do["position_losses"])
                    val_position_pos_dists_do.extend(horizon_result_do["position_pos_dists"])

                mean_val_loss_do = float(np.mean(val_losses_do))
                mean_val_crossing_loss_do = float(np.mean(val_crossing_losses_do)) if val_crossing_losses_do else float("nan")
                val_crossing_pos_dist_do = float(np.mean(val_crossing_pos_dists_do)) if val_crossing_pos_dists_do else float("nan")
                val_crossing_dt_mae_do = float(np.mean(val_crossing_dt_maes_do)) if val_crossing_dt_maes_do else float("nan")
                mean_val_resting_loss_do = float(np.mean(val_resting_losses_do)) if val_resting_losses_do else float("nan")
                val_resting_pos_dist_do = float(np.mean(val_resting_pos_dists_do)) if val_resting_pos_dists_do else float("nan")
                mean_val_position_loss_do = float(np.mean(val_position_losses_do)) if val_position_losses_do else float("nan")
                val_position_pos_dist_do = float(np.mean(val_position_pos_dists_do)) if val_position_pos_dists_do else float("nan")
                mean_val_event_loss_do = float(np.mean(val_event_losses_do)) if val_event_losses_do else float("nan")
                val_event_oob_acc_do = float(np.mean(val_event_oob_accs_do)) if val_event_oob_accs_do else float("nan")
                val_event_goal_acc_do = float(np.mean(val_event_goal_accs_do)) if val_event_goal_accs_do else float("nan")
                mean_val_backprop_loss_do = (
                    mean_val_loss_do + resting_weight * mean_val_resting_loss_do + position_weight * mean_val_position_loss_do
                    + event_weight * mean_val_event_loss_do
                )
                if has_crossing_data:
                    mean_val_backprop_loss_do += crossing_weight * mean_val_crossing_loss_do
                improved_do = mean_val_loss_do < (best_val_loss_do - decoder_only_early_stop_min_delta)
                # "best" only moves on a genuine (>=min_delta) improvement --
                # otherwise best_val_loss_do/the printed "best=" ratchets down
                # on every tiny (<min_delta) decrease while the patience
                # counter still climbs (since THAT'S gated by min_delta),
                # which looks like a contradiction ("best keeps improving but
                # patience says it's stagnant") even though both were
                # individually correct -- matches the main loop's identical
                # convention. best_state_do is only ever tracked/restored
                # when decoder_only_early_stop_enabled -- matches the main
                # loop, where turning patience off means no best-weight
                # restoration happens either, not just no early cutoff.
                if decoder_only_early_stop_enabled:
                    if improved_do:
                        best_val_loss_do = mean_val_loss_do
                        best_state_do = copy.deepcopy(model.state_dict())
                        decoder_only_patience_ctr = 0
                    else:
                        decoder_only_patience_ctr += 1
                elif mean_val_loss_do < best_val_loss_do:
                    best_val_loss_do = mean_val_loss_do

                val_t0_means_do = {c: np.full(n_h, np.nan) for c in _COMPONENTS}
                val_t0_r2_do: dict[str, np.ndarray] = {}
                val_t0_cls_do: dict[str, np.ndarray] = {}
                mean_val_pair_loss_do = float("nan")
                if horizon_result_do is not None:
                    mean_val_pair_loss_do = horizon_result_do["mean_pair_loss"]
                    if autoencode_during_main_loop_enabled:
                        mean_val_t0_loss_do = horizon_result_do["mean_t0_loss"]
                        val_t0_means_do = horizon_result_do["t0_means"]
                        val_t0_r2_do = horizon_result_do["t0_r2"]
                        val_t0_cls_do = horizon_result_do["t0_cls"]

                val_means_do = _mean_breakdown(val_breakdowns_do)
                val_cls_do = _classification_from_counts(val_oob_counts_do, val_goal_counts_do)
                val_r2_do = _r2_from_group_sums(val_sq_err_do, val_n_do)
                val_pct_disp_do = _pct_disp_from_group_sums(val_sq_err_do, val_n_do)
                val_pct_ballistic_do = _pct_ballistic_from_group_sums(val_sq_err_do, val_n_do)
                val_line_do = f"  val_loss={mean_val_loss_do:.4f}  best={best_val_loss_do:.4f}"
                if decoder_only_early_stop_enabled:
                    val_line_do += (
                        "  (improved)" if improved_do
                        else f"  (patience {decoder_only_patience_ctr}/{decoder_only_early_stop_patience})"
                    )

            pair_line_do = ""
            if adjacent_pair_training_enabled and n_horizons > 1:
                pair_line_do = f"  train_pair_loss={mean_train_pair_loss_do:.4f}"
                if len(val_idx) > 0:
                    pair_line_do += f"  val_pair_loss={mean_val_pair_loss_do:.4f}"
            t0_line_do = ""
            if autoencode_during_main_loop_enabled:
                t0_line_do = f"  train_t0_loss={mean_train_t0_loss_do:.4f}"
                if len(val_idx) > 0:
                    t0_line_do += f"  val_t0_loss={mean_val_t0_loss_do:.4f}"
            crossing_line_do = ""
            if has_crossing_data:
                crossing_line_do = (
                    f"  train_crossing_loss={mean_train_crossing_loss_do:.4f}"
                    f" (pos_dist={train_crossing_pos_dist_do:.4f}, dt_mae={train_crossing_dt_mae_do:.4f})"
                )
                if len(val_idx) > 0:
                    crossing_line_do += (
                        f"  val_crossing_loss={mean_val_crossing_loss_do:.4f}"
                        f" (pos_dist={val_crossing_pos_dist_do:.4f}, dt_mae={val_crossing_dt_mae_do:.4f})"
                    )
            crossing_line_do += (
                f"  train_resting_loss={mean_train_resting_loss_do:.4f} (pos_dist={train_resting_pos_dist_do:.4f})"
                f"  train_position_loss={mean_train_position_loss_do:.4f} (pos_dist={train_position_pos_dist_do:.4f})"
                f"  train_event_loss={mean_train_event_loss_do:.4f} (oob_acc={train_event_oob_acc_do:.4f}, goal_acc={train_event_goal_acc_do:.4f})"
                f"  train_backprop_loss={mean_train_backprop_loss_do:.4f}"
            )
            if len(val_idx) > 0:
                crossing_line_do += (
                    f"  val_resting_loss={mean_val_resting_loss_do:.4f} (pos_dist={val_resting_pos_dist_do:.4f})"
                    f"  val_position_loss={mean_val_position_loss_do:.4f} (pos_dist={val_position_pos_dist_do:.4f})"
                    f"  val_event_loss={mean_val_event_loss_do:.4f} (oob_acc={val_event_oob_acc_do:.4f}, goal_acc={val_event_goal_acc_do:.4f})"
                    f"  val_backprop_loss={mean_val_backprop_loss_do:.4f}"
                )
            log.info(
                f"decoder-only pretrain epoch {do_epoch + 1}/{decoder_only_pretrain_epochs}: "
                f"train_loss={mean_train_loss_do:.4f}{val_line_do}{pair_line_do}{t0_line_do}{crossing_line_do}"
            )
            for c in _LOG_COMPONENTS:
                _log_component("train", c, train_means_do[c])
            for c in _LOG_R2_KEYS:
                log.info(f"    train {c:9s} by horizon: {np.array2string(train_r2_do[c], precision=4)}, mean: {_safe_nanmean(train_r2_do[c]):.4f}")
            for c in _LOG_PCTD_KEYS:
                log.info(f"    train {c:16s} by horizon: {np.array2string(train_pct_disp_do[c], precision=4)}, mean: {_safe_nanmean(train_pct_disp_do[c]):.4f}")
            for c in _LOG_PCTB_KEYS:
                log.info(f"    train {c:20s} by horizon: {np.array2string(train_pct_ballistic_do[c], precision=4)}, mean: {_safe_nanmean(train_pct_ballistic_do[c]):.4f}")
            if autoencode_during_main_loop_enabled:
                _log_t0_diagnostics("train", train_t0_means_do, train_t0_r2_do, None)
            if len(val_idx) > 0:
                for c in _LOG_COMPONENTS:
                    _log_component("val  ", c, val_means_do[c])
                for c in _LOG_CLS_KEYS:
                    log.info(f"    val   {c:14s} by horizon: {np.array2string(val_cls_do[c], precision=4)}, mean: {_safe_nanmean(val_cls_do[c]):.4f}")
                for c in _LOG_R2_KEYS:
                    log.info(f"    val   {c:9s} by horizon: {np.array2string(val_r2_do[c], precision=4)}, mean: {_safe_nanmean(val_r2_do[c]):.4f}")
                for c in _LOG_PCTD_KEYS:
                    log.info(f"    val   {c:16s} by horizon: {np.array2string(val_pct_disp_do[c], precision=4)}, mean: {_safe_nanmean(val_pct_disp_do[c]):.4f}")
                for c in _LOG_PCTB_KEYS:
                    log.info(f"    val   {c:20s} by horizon: {np.array2string(val_pct_ballistic_do[c], precision=4)}, mean: {_safe_nanmean(val_pct_ballistic_do[c]):.4f}")
                if autoencode_during_main_loop_enabled:
                    _log_t0_diagnostics("val  ", val_t0_means_do, val_t0_r2_do, val_t0_cls_do)

            if decoder_only_early_stop_enabled and decoder_only_patience_ctr >= decoder_only_early_stop_patience:
                log.info(
                    f"Decoder-only pretrain early stop at epoch {do_epoch + 1}/{decoder_only_pretrain_epochs} "
                    f"(val stagnant for {decoder_only_early_stop_patience} epochs, best={best_val_loss_do:.4f})"
                )
                break

        if best_state_do is not None:
            model.load_state_dict(best_state_do)
            log.info(f"Restored best decoder-only-pretrain weights (val_loss={best_val_loss_do:.4f})")

        # Remove the identity-row mask -- it's scoped to THIS phase only
        # (see the long comment above); the main loop trains encoder.out
        # fully unmasked, same as it always has.
        for handle in encoder_out_hook_handles:
            handle.remove()
        for p in model.encoder.parameters():
            p.requires_grad_(True)

        _save_phase_checkpoint("after_decoder_pretrain")

    for epoch in range(epochs):
        model.train()
        epoch_result = _run_interleaved_train_epoch(optimizer)
        mean_train_loss = epoch_result["mean_loss"]
        mean_train_pair_loss = epoch_result["mean_pair_loss"]
        mean_train_t0_loss = epoch_result["mean_t0_loss"]
        train_t0_means = epoch_result["t0_means"]
        train_t0_r2 = epoch_result["t0_r2"]
        mean_train_crossing_loss = epoch_result["mean_crossing_loss"]
        train_crossing_pos_dist = epoch_result["crossing_pos_dist"]
        train_crossing_dt_mae = epoch_result["crossing_dt_mae"]
        mean_train_resting_loss = epoch_result["mean_resting_loss"]
        train_resting_pos_dist = epoch_result["resting_pos_dist"]
        mean_train_position_loss = epoch_result["mean_position_loss"]
        train_position_pos_dist = epoch_result["position_pos_dist"]
        mean_train_event_loss = epoch_result["mean_event_loss"]
        train_event_oob_acc = epoch_result["event_oob_acc"]
        train_event_goal_acc = epoch_result["event_goal_acc"]
        mean_train_backprop_loss = epoch_result["mean_backprop_loss"]
        grad_norm_stats = epoch_result["grad_norm_stats"]
        loss_delta_stats = epoch_result["loss_delta_stats"]

        train_means = _mean_breakdown(epoch_result["breakdowns"])
        train_cls = _classification_from_counts(epoch_result["oob_counts"], epoch_result["goal_counts"])
        train_r2 = _r2_from_group_sums(epoch_result["sq_err"], epoch_result["n"])
        train_pct_disp = _pct_disp_from_group_sums(epoch_result["sq_err"], epoch_result["n"])
        train_pct_ballistic = _pct_ballistic_from_group_sums(epoch_result["sq_err"], epoch_result["n"])

        val_line = ""
        val_loss = float("nan")
        mean_val_pair_loss = float("nan")
        mean_val_t0_loss = float("nan")
        mean_val_crossing_loss = float("nan")
        val_crossing_pos_dist = float("nan")
        val_crossing_dt_mae = float("nan")
        mean_val_resting_loss = float("nan")
        val_resting_pos_dist = float("nan")
        mean_val_position_loss = float("nan")
        val_position_pos_dist = float("nan")
        mean_val_event_loss = float("nan")
        val_event_oob_acc = float("nan")
        val_event_goal_acc = float("nan")
        mean_val_backprop_loss = float("nan")
        val_loss_delta = float("nan")
        val_means = _mean_breakdown([])
        val_cls = _classification_from_counts(np.zeros((n_h, 4), dtype=np.int64), np.zeros((n_h, 4), dtype=np.int64))
        val_r2 = _r2_from_group_sums({g: np.zeros(n_h) for g in _GROUPS}, {g: np.zeros(n_h, dtype=np.int64) for g in _GROUPS})
        val_pct_disp = _pct_disp_from_group_sums({g: np.zeros(n_h) for g in _GROUPS}, {g: np.zeros(n_h, dtype=np.int64) for g in _GROUPS})
        val_pct_ballistic = _pct_ballistic_from_group_sums({g: np.zeros(n_h) for g in _GROUPS}, {g: np.zeros(n_h, dtype=np.int64) for g in _GROUPS})
        if len(val_idx) > 0:
            model.eval()
            val_losses: list[float] = []
            val_breakdowns: list[LossBreakdown] = []
            val_oob_counts = np.zeros((n_h, 4), dtype=np.int64)
            val_goal_counts = np.zeros((n_h, 4), dtype=np.int64)
            val_sq_err = {g: np.zeros(n_h) for g in _GROUPS}
            val_n = {g: np.zeros(n_h, dtype=np.int64) for g in _GROUPS}
            val_crossing_losses: list[float] = []
            val_crossing_pos_dists: list[float] = []
            val_crossing_dt_maes: list[float] = []
            val_resting_losses: list[float] = []
            val_resting_pos_dists: list[float] = []
            val_position_losses: list[float] = []
            val_position_pos_dists: list[float] = []
            val_event_losses: list[float] = []
            val_event_oob_accs: list[float] = []
            val_event_goal_accs: list[float] = []
            _val_pos = 0
            with torch.no_grad():
                for x, y in ds.iterate_minibatches(batch_size, val_idx, shuffle=False, device=device):
                    latent, heads = model(x)
                    loss, breakdown = compute_loss(heads, y, pos_weight, bce_weight, spin_weight)
                    val_losses.append(float(loss.item()))
                    val_breakdowns.append(breakdown)
                    counts = compute_confusion_counts(heads, y)
                    val_oob_counts += np.array(counts["oob"])
                    val_goal_counts += np.array(counts["goal"])
                    sq_err_counts = compute_group_sq_err(heads, y)
                    for g in _GROUPS:
                        for h, (sq_err, n) in enumerate(sq_err_counts[g]):
                            val_sq_err[g][h] += sq_err
                            val_n[g][h] += n
                    # `shuffle=False` above guarantees `iterate_minibatches`
                    # yields batches in the SAME order as `val_idx` itself,
                    # so a running offset into `val_idx` recovers each
                    # batch's original row indices without `iterate_
                    # minibatches` needing to expose them itself.
                    batch_idx = val_idx[_val_pos:_val_pos + x.shape[0]]
                    if has_crossing_data:
                        crossing_loss, pos_dist_mean, dt_mae_mean = _crossing_head_loss(
                            model, latent, ds.crossing_pos, ds.crossing_dt, ds.crossing_mask, batch_idx, device,
                        )
                        val_crossing_losses.append(float(crossing_loss.item()))
                        val_crossing_pos_dists.append(pos_dist_mean)
                        val_crossing_dt_maes.append(dt_mae_mean)
                    resting_loss, resting_pos_dist_mean = _resting_head_loss(
                        model, latent, resting_pos, resting_mask, batch_idx, device,
                    )
                    val_resting_losses.append(float(resting_loss.item()))
                    val_resting_pos_dists.append(resting_pos_dist_mean)
                    position_loss, position_pos_dist_mean = _position_head_loss(model, latent, x)
                    val_position_losses.append(float(position_loss.item()))
                    val_position_pos_dists.append(position_pos_dist_mean)
                    event_loss, event_oob_acc, event_goal_acc = _event_head_loss(
                        model, latent, ever_oob, ever_goal, batch_idx, device,
                    )
                    val_event_losses.append(float(event_loss.item()))
                    val_event_oob_accs.append(event_oob_acc)
                    val_event_goal_accs.append(event_goal_acc)
                    _val_pos += x.shape[0]

            # Horizon-generalized pair/t0/crossing/resting -- one shared
            # eval pass (see _eval_horizon_pass's docstring); its
            # crossing/resting raw lists are BLENDED into the same
            # val_crossing_losses/val_resting_losses accumulators the t=0
            # usage above just populated (matching the training side's
            # single shared accumulator convention), NOT reported as
            # separate metrics.
            horizon_result = (
                _eval_horizon_pass(autoencode_val_data, horizon_bundle_val, crossing_pos_val)
                if horizon_pass_enabled else None
            )
            if horizon_result is not None:
                val_crossing_losses.extend(horizon_result["crossing_losses"])
                val_crossing_pos_dists.extend(horizon_result["crossing_pos_dists"])
                val_crossing_dt_maes.extend(horizon_result["crossing_dt_maes"])
                val_resting_losses.extend(horizon_result["resting_losses"])
                val_resting_pos_dists.extend(horizon_result["resting_pos_dists"])
                val_position_losses.extend(horizon_result["position_losses"])
                val_position_pos_dists.extend(horizon_result["position_pos_dists"])

            val_loss = float(np.mean(val_losses))
            mean_val_crossing_loss = float(np.mean(val_crossing_losses)) if val_crossing_losses else float("nan")
            val_crossing_pos_dist = float(np.mean(val_crossing_pos_dists)) if val_crossing_pos_dists else float("nan")
            val_crossing_dt_mae = float(np.mean(val_crossing_dt_maes)) if val_crossing_dt_maes else float("nan")
            mean_val_resting_loss = float(np.mean(val_resting_losses)) if val_resting_losses else float("nan")
            val_resting_pos_dist = float(np.mean(val_resting_pos_dists)) if val_resting_pos_dists else float("nan")
            mean_val_position_loss = float(np.mean(val_position_losses)) if val_position_losses else float("nan")
            val_position_pos_dist = float(np.mean(val_position_pos_dists)) if val_position_pos_dists else float("nan")
            mean_val_event_loss = float(np.mean(val_event_losses)) if val_event_losses else float("nan")
            val_event_oob_acc = float(np.mean(val_event_oob_accs)) if val_event_oob_accs else float("nan")
            val_event_goal_acc = float(np.mean(val_event_goal_accs)) if val_event_goal_accs else float("nan")
            # Same combined-objective value as `mean_backprop_loss` above,
            # computed post-hoc rather than accumulated per-batch (val has
            # no backward pass to piggyback the crossing/resting/position/
            # event terms onto) -- valid because every term is already a
            # batch-size-weighted mean over the SAME val batches, so summing
            # the means equals the mean of the sums.
            mean_val_backprop_loss = (
                val_loss + resting_weight * mean_val_resting_loss + position_weight * mean_val_position_loss
                + event_weight * mean_val_event_loss
            )
            if has_crossing_data:
                mean_val_backprop_loss += crossing_weight * mean_val_crossing_loss

            val_t0_means = {c: np.full(n_h, np.nan) for c in _COMPONENTS}
            val_t0_r2: dict[str, np.ndarray] = {}
            val_t0_cls: dict[str, np.ndarray] = {}
            mean_val_pair_loss = float("nan")
            if horizon_result is not None:
                mean_val_pair_loss = horizon_result["mean_pair_loss"]
                if autoencode_during_main_loop_enabled:
                    mean_val_t0_loss = horizon_result["mean_t0_loss"]
                    val_t0_means = horizon_result["t0_means"]
                    val_t0_r2 = horizon_result["t0_r2"]
                    val_t0_cls = horizon_result["t0_cls"]

            val_means = _mean_breakdown(val_breakdowns)
            val_cls = _classification_from_counts(val_oob_counts, val_goal_counts)
            val_r2 = _r2_from_group_sums(val_sq_err, val_n)
            val_pct_disp = _pct_disp_from_group_sums(val_sq_err, val_n)
            val_pct_ballistic = _pct_ballistic_from_group_sums(val_sq_err, val_n)
            improved = val_loss < (best_val_loss - early_stop_min_delta)
            val_line = f"  val_loss={val_loss:.4f}  best={min(best_val_loss, val_loss):.4f}"
            if early_stop_enabled:
                val_line += "  (improved)" if improved else f"  (patience {patience_ctr + 1}/{early_stop_patience})"
            if improved:
                best_val_loss = val_loss
                if early_stop_enabled:
                    best_state = copy.deepcopy(model.state_dict())
                    patience_ctr = 0
                _save_phase_checkpoint("midtrain_latest")
            elif early_stop_enabled:
                patience_ctr += 1

            # Epoch-over-epoch change in val_loss (negative = improved) --
            # a single, low-noise number (val_loss is already an average
            # over the whole val set, unlike the per-batch train diagnostics
            # below) for "how much did generalization actually move this
            # epoch," complementing best_val_loss's "how much better than
            # the best epoch ever" framing. nan on the first epoch (no prior
            # value yet).
            val_loss_delta = val_loss - prev_val_loss if not np.isnan(prev_val_loss) else float("nan")
            prev_val_loss = val_loss

        current_lr = optimizer.param_groups[0]["lr"]
        pair_line = ""
        if adjacent_pair_training_enabled and n_horizons > 1:
            pair_line = f"  train_pair_loss={mean_train_pair_loss:.4f}"
            if len(val_idx) > 0:
                pair_line += f"  val_pair_loss={mean_val_pair_loss:.4f}"
        t0_line = ""
        if autoencode_during_main_loop_enabled:
            t0_line = f"  train_t0_loss={mean_train_t0_loss:.4f}"
            if len(val_idx) > 0:
                t0_line += f"  val_t0_loss={mean_val_t0_loss:.4f}"
        crossing_line = ""
        if has_crossing_data:
            crossing_line = (
                f"  train_crossing_loss={mean_train_crossing_loss:.4f}"
                f" (pos_dist={train_crossing_pos_dist:.4f}, dt_mae={train_crossing_dt_mae:.4f})"
            )
            if len(val_idx) > 0:
                crossing_line += (
                    f"  val_crossing_loss={mean_val_crossing_loss:.4f}"
                    f" (pos_dist={val_crossing_pos_dist:.4f}, dt_mae={val_crossing_dt_mae:.4f})"
                )
        crossing_line += (
            f"  train_resting_loss={mean_train_resting_loss:.4f} (pos_dist={train_resting_pos_dist:.4f})"
            f"  train_position_loss={mean_train_position_loss:.4f} (pos_dist={train_position_pos_dist:.4f})"
            f"  train_event_loss={mean_train_event_loss:.4f} (oob_acc={train_event_oob_acc:.4f}, goal_acc={train_event_goal_acc:.4f})"
            f"  train_backprop_loss={mean_train_backprop_loss:.4f}"
        )
        if len(val_idx) > 0:
            crossing_line += (
                f"  val_resting_loss={mean_val_resting_loss:.4f} (pos_dist={val_resting_pos_dist:.4f})"
                f"  val_position_loss={mean_val_position_loss:.4f} (pos_dist={val_position_pos_dist:.4f})"
                f"  val_event_loss={mean_val_event_loss:.4f} (oob_acc={val_event_oob_acc:.4f}, goal_acc={val_event_goal_acc:.4f})"
                f"  val_backprop_loss={mean_val_backprop_loss:.4f}"
            )
        log.info(
            f"epoch {epoch + 1}/{epochs}: train_loss={mean_train_loss:.4f}{val_line}  lr={current_lr:.2e}{pair_line}{t0_line}{crossing_line}"
        )
        # Convergence diagnostics: gradient-norm stats (is the step size too
        # big for the local curvature -- large/noisy norms suggest trying a
        # smaller lr/eta_min_frac, small/stable norms suggest it's genuinely
        # converged and just slow) and train-loss-delta stats (how noisy/
        # bumpy the per-batch trajectory is -- mostly batch-composition
        # noise at a small loss scale, NOT a direct progress signal, see
        # `_run_interleaved_train_epoch`'s docstring for that distinction),
        # plus the single low-noise val_loss epoch-over-epoch delta. 6dp
        # since at a converged loss scale (~0.01-0.001) these deltas/norms
        # are themselves small enough that 4dp would round most of them to
        # 0.0000 and hide exactly the signal being looked for.
        log.info(
            f"    grad_norm: mean={grad_norm_stats['mean']:.6f} std={grad_norm_stats['std']:.6f} "
            f"min={grad_norm_stats['min']:.6f} max={grad_norm_stats['max']:.6f}"
        )
        log.info(
            f"    train_loss_delta (batch-to-batch): mean={loss_delta_stats['mean']:.6f} std={loss_delta_stats['std']:.6f} "
            f"min={loss_delta_stats['min']:.6f} max={loss_delta_stats['max']:.6f}"
        )
        if len(val_idx) > 0:
            log.info(f"    val_loss_delta (epoch-over-epoch): {val_loss_delta:.6f}")
        for c in _LOG_COMPONENTS:
            _log_component("train", c, train_means[c])
        # oob/goal accuracy/precision/recall are NOT logged for train (still
        # computed and saved to history/report) -- val alone is enough
        # signal per epoch and this was a lot of log-line noise.
        for c in _LOG_R2_KEYS:
            log.info(f"    train {c:9s} by horizon: {np.array2string(train_r2[c], precision=4)}, mean: {_safe_nanmean(train_r2[c]):.4f}")
        for c in _LOG_PCTD_KEYS:
            log.info(f"    train {c:16s} by horizon: {np.array2string(train_pct_disp[c], precision=4)}, mean: {_safe_nanmean(train_pct_disp[c]):.4f}")
        for c in _LOG_PCTB_KEYS:
            log.info(f"    train {c:20s} by horizon: {np.array2string(train_pct_ballistic[c], precision=4)}, mean: {_safe_nanmean(train_pct_ballistic[c]):.4f}")
        if autoencode_during_main_loop_enabled:
            _log_t0_diagnostics("train", train_t0_means, train_t0_r2, None)
        if len(val_idx) > 0:
            for c in _LOG_COMPONENTS:
                _log_component("val  ", c, val_means[c])
            for c in _LOG_CLS_KEYS:
                log.info(f"    val   {c:14s} by horizon: {np.array2string(val_cls[c], precision=4)}, mean: {_safe_nanmean(val_cls[c]):.4f}")
            for c in _LOG_R2_KEYS:
                log.info(f"    val   {c:9s} by horizon: {np.array2string(val_r2[c], precision=4)}, mean: {_safe_nanmean(val_r2[c]):.4f}")
            for c in _LOG_PCTD_KEYS:
                log.info(f"    val   {c:16s} by horizon: {np.array2string(val_pct_disp[c], precision=4)}, mean: {_safe_nanmean(val_pct_disp[c]):.4f}")
            if autoencode_during_main_loop_enabled:
                _log_t0_diagnostics("val  ", val_t0_means, val_t0_r2, val_t0_cls)
            for c in _LOG_PCTB_KEYS:
                log.info(f"    val   {c:20s} by horizon: {np.array2string(val_pct_ballistic[c], precision=4)}, mean: {_safe_nanmean(val_pct_ballistic[c]):.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": mean_train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "train_pair_loss": mean_train_pair_loss,
            "val_pair_loss": mean_val_pair_loss,
            "train_t0_loss": mean_train_t0_loss,
            "val_t0_loss": mean_val_t0_loss,
            "train_crossing_loss": mean_train_crossing_loss,
            "val_crossing_loss": mean_val_crossing_loss,
            "train_crossing_pos_dist": train_crossing_pos_dist,
            "val_crossing_pos_dist": val_crossing_pos_dist,
            "train_crossing_dt_mae": train_crossing_dt_mae,
            "val_crossing_dt_mae": val_crossing_dt_mae,
            "train_resting_loss": mean_train_resting_loss,
            "val_resting_loss": mean_val_resting_loss,
            "train_resting_pos_dist": train_resting_pos_dist,
            "val_resting_pos_dist": val_resting_pos_dist,
            "train_position_loss": mean_train_position_loss,
            "val_position_loss": mean_val_position_loss,
            "train_position_pos_dist": train_position_pos_dist,
            "val_position_pos_dist": val_position_pos_dist,
            "train_event_loss": mean_train_event_loss,
            "val_event_loss": mean_val_event_loss,
            "train_event_oob_acc": train_event_oob_acc,
            "val_event_oob_acc": val_event_oob_acc,
            "train_event_goal_acc": train_event_goal_acc,
            "val_event_goal_acc": val_event_goal_acc,
            "train_backprop_loss": mean_train_backprop_loss,
            "val_backprop_loss": mean_val_backprop_loss,
            "grad_norm_mean": grad_norm_stats["mean"],
            "grad_norm_std": grad_norm_stats["std"],
            "grad_norm_min": grad_norm_stats["min"],
            "grad_norm_max": grad_norm_stats["max"],
            "train_loss_delta_mean": loss_delta_stats["mean"],
            "train_loss_delta_std": loss_delta_stats["std"],
            "train_loss_delta_min": loss_delta_stats["min"],
            "train_loss_delta_max": loss_delta_stats["max"],
            "val_loss_delta": val_loss_delta,
            **{f"train_{c}": train_means[c] for c in _COMPONENTS},
            **{f"val_{c}": val_means[c] for c in _COMPONENTS},
            **{f"train_{c}": train_cls[c] for c in _CLS_KEYS},
            **{f"val_{c}": val_cls[c] for c in _CLS_KEYS},
            **{f"train_{c}": train_r2[c] for c in _R2_KEYS},
            **{f"val_{c}": val_r2[c] for c in _R2_KEYS},
            **{f"train_{c}": train_pct_disp[c] for c in _PCTD_KEYS},
            **{f"val_{c}": val_pct_disp[c] for c in _PCTD_KEYS},
            **{f"train_{c}": train_pct_ballistic[c] for c in _PCTB_KEYS},
            **{f"val_{c}": val_pct_ballistic[c] for c in _PCTB_KEYS},
        })

        if scheduler is not None:
            scheduler.step()
            if lr_cosine_peak_decay != 1.0 and scheduler.T_cur == 0:
                # scheduler.step() just wrote THIS epoch's LR into
                # optimizer.param_groups using the OLD (pre-decay) base_lrs
                # -- mutating base_lrs alone wouldn't take effect until the
                # NEXT step() call, leaving one full epoch at the stale
                # undecayed peak. At T_cur==0 the scheduler's own formula
                # collapses to exactly `base_lr` (cos(0)=1), so directly
                # overwrite param_groups' lr with the freshly decayed value
                # too, to actually apply it starting this epoch.
                # eta_min is a single scalar (not per-group) and, unlike
                # base_lrs, get_lr() reads it fresh from `self.eta_min`
                # every call rather than caching it -- so just reassigning
                # it here is enough, no separate "already applied to
                # param_groups" fixup needed the way base_lrs required.
                # Keeps the floor-to-peak ratio (lr_cosine_eta_min_frac)
                # constant across cycles instead of the floor staying fixed
                # while the peak shrinks toward (and eventually past) it.
                scheduler.base_lrs = [b * lr_cosine_peak_decay for b in scheduler.base_lrs]
                scheduler.eta_min = scheduler.base_lrs[0] * lr_cosine_eta_min_frac
                for pg, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                    pg["lr"] = base_lr
                log.info(
                    f"Cosine restart: peak LR decayed by {lr_cosine_peak_decay} -> {scheduler.base_lrs[0]:.2e}, "
                    f"eta_min -> {scheduler.eta_min:.2e}"
                )

        if early_stop_enabled and patience_ctr >= early_stop_patience:
            log.info(f"Early stop at epoch {epoch + 1} (val stagnant for {early_stop_patience} epochs, best={best_val_loss:.4f})")
            stopped_early = True
            break

    if stopped_early and best_state is not None:
        model.load_state_dict(best_state)
        log.info(f"Restored best-val weights (val_loss={best_val_loss:.4f})")

    _save_phase_checkpoint("after_training")

    # Median/worst-by-loss val episode diagnostic: helps distinguish "the
    # model is uniformly mediocre" from "a few pathological episodes are
    # dragging the average down" -- printing their (denormalized) initial
    # conditions lets you go look for a pattern (e.g. all near a touchline,
    # all high-spin) rather than guessing from aggregate metrics alone.
    if len(val_idx) > 0:
        model.eval()
        per_episode_losses = []
        with torch.no_grad():
            for x, y in ds.iterate_minibatches(batch_size, val_idx, shuffle=False, device=device):
                _, heads = model(x)
                per_episode_losses.append(compute_per_episode_loss(heads, y, pos_weight, bce_weight, spin_weight).cpu().numpy())
        per_episode_losses = np.concatenate(per_episode_losses)
        median_local = int(np.argmin(np.abs(per_episode_losses - np.median(per_episode_losses))))
        worst_local = int(np.argmax(per_episode_losses))
        for label, local_idx in (("Median", median_local), ("Worst (max loss)", worst_local)):
            episode_idx = val_idx[local_idx]
            loss_val = per_episode_losses[local_idx]
            conditions = describe_input_row(ds.inputs[episode_idx], gen_params)
            log.info(f"{label} val episode (idx={episode_idx}, loss={loss_val:.4f}): {conditions}")

            # Per-horizon breakdown for this one episode: ground-truth
            # target, model prediction, and loss at each horizon, so a
            # pathological episode's failure mode is visible (e.g. "fine
            # until it goes out of bounds at t=2s, then diverges") rather
            # than just the single summed-across-horizons loss above.
            input_row = ds.inputs[episode_idx]
            pitch_length_m, pitch_width_m = _pitch_dims_m(input_row, gen_params)
            x_single = torch.from_numpy(ds.inputs[episode_idx:episode_idx + 1].astype(np.float32, copy=False)).to(device)
            y_single = torch.from_numpy(ds.targets[episode_idx:episode_idx + 1].astype(np.float32, copy=False)).to(device)
            with torch.no_grad():
                _, heads_single = model(x_single)
                for h, head_out in enumerate(heads_single):
                    base = h * N_TARGET_FIELDS_PER_HORIZON
                    target_h = y_single[:, base:base + N_TARGET_FIELDS_PER_HORIZON]
                    h_loss = float(_single_target_per_episode_loss(head_out, target_h, pos_weight[h], bce_weight, spin_weight)[0].item())
                    gt_row = ds.targets[episode_idx, base:base + N_TARGET_FIELDS_PER_HORIZON]
                    pred_row = head_out[0].cpu().numpy()
                    log.info(f"    t={cfg['horizons_s'][h]:>4}s  loss={h_loss:.4f}")
                    log.info(f"        target: {describe_target_row(gt_row, pitch_length_m, pitch_width_m, gen_params, logits=False)}")
                    log.info(f"        pred  : {describe_target_row(pred_row, pitch_length_m, pitch_width_m, gen_params, logits=True)}")

    # Random-sample val-example table (10 (episode, horizon) pairs), saved
    # into the report's bottom panel: input/pred/target side by side so a
    # reader can eyeball a handful of TYPICAL predictions directly, not just
    # aggregate metrics -- complements (doesn't replace) the median/worst-
    # episode diagnostic above, which is deliberately NOT representative
    # (picked for being extreme, not typical). Reuses `rng` (seeded, same
    # generator driving the rest of this run) so the sample is reproducible
    # given the same --seed.
    val_examples: list[dict] = []
    if len(val_idx) > 0:
        model.eval()
        n_examples = min(10, len(val_idx))
        chosen_local = rng.choice(len(val_idx), size=n_examples, replace=False)
        with torch.no_grad():
            for local_idx in chosen_local:
                episode_idx = int(val_idx[local_idx])
                h = int(rng.integers(0, n_h))
                input_row = ds.inputs[episode_idx]
                pitch_length_m, pitch_width_m = _pitch_dims_m(input_row, gen_params)
                x_single = torch.from_numpy(input_row[None].astype(np.float32, copy=False)).to(device)
                _, heads_single = model(x_single)
                base = h * N_TARGET_FIELDS_PER_HORIZON
                pred_row = heads_single[h][0].cpu().numpy()
                target_row = ds.targets[episode_idx, base:base + N_TARGET_FIELDS_PER_HORIZON]
                val_examples.append({
                    "episode_idx": episode_idx,
                    "horizon_s": cfg["horizons_s"][h],
                    "input": describe_input_row(input_row, gen_params),
                    "pred": describe_target_row(pred_row, pitch_length_m, pitch_width_m, gen_params, logits=True),
                    "target": describe_target_row(target_row, pitch_length_m, pitch_width_m, gen_params, logits=False),
                })

    # t=0 autoencoding sanity check: query the decoder at a horizon it was
    # NEVER trained on (t=0, via `forward_at()`, not the fixed trained
    # grid), where the correct answer is exactly the input itself (nothing
    # has happened yet). Isolates "can the encoder+decoder round-trip
    # through the latent bottleneck at all" from "can it also predict real
    # dynamics" -- a bad result here specifically implicates the bottleneck
    # (latent_dim/encoder/decoder capacity), since there's no actual
    # dynamics-prediction difficulty at t=0 to blame it on instead.
    if len(val_idx) > 0:
        model.eval()
        t0_sq_err = {g: 0.0 for g in _GROUPS}
        t0_n = 0
        with torch.no_grad():
            for x, y in ds.iterate_minibatches(batch_size, val_idx, shuffle=False, device=device):
                latent = model.encoder(x)
                pred0 = model.decoder.forward_at(latent, 0.0)
                for g, (lo, hi) in _GROUPS.items():
                    t0_sq_err[g] += float(((pred0[:, lo:hi] - x[:, lo:hi]) ** 2).sum().item())
                t0_n += x.shape[0]
        log.info("t=0 autoencoding sanity check (val, vs. own input -- correct answer is exact):")
        _group_unit_scale = {"pos": (pitch_half_diag_m, "m"), "vel": (pitch_half_diag_m, "m/s"), "spin": (BALL_SPIN_NORM_DIVISOR_RAD_S, "rad/s")}
        for g in _GROUPS:
            rmse = (t0_sq_err[g] / (t0_n * 3)) ** 0.5
            scale, unit = _group_unit_scale[g]
            log.info(f"    {g:4s} RMSE: {rmse:.4f} ({rmse * scale:.4f} {unit})")

    # Normalization scale factors, saved so a later reader (e.g. the training
    # report artifact) can convert normalized RMSE back to approximate real
    # units (metres, m/s, rad/s) without re-deriving them. Position mixes
    # THREE different per-field scales (pos_x/half_length, pos_y/half_width,
    # height_m/height_norm_m) into one combined RMSE, so there is no single
    # exact metres conversion for it -- pitch_half_diag_m (the same scale
    # velocity uses) is saved as the best available APPROXIMATE stand-in,
    # since x/y dominate the combined error at typical pitch dimensions.
    # Every scale is the BASE (standard-pitch) value even though pitch size
    # is itself randomized per-episode (§4.2) -- an approximation, not exact,
    # since different episodes used different actual denominators.
    # (gen_params / pitch_half_diag_m already computed above, for the
    # per-epoch RMSE log lines.)
    normalization = {
        "pitch_half_diag_m": pitch_half_diag_m,
        "height_norm_m": gen_params.height_norm_m,
        "ball_spin_norm_max_rad_s": BALL_SPIN_NORM_DIVISOR_RAD_S,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "encoder_state_dict": model.encoder.state_dict(),
        "config_snapshot": cfg,
        "normalization": normalization,
        "dataset_stats": {
            "n_episodes": len(ds),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        },
        "physics_config_hash": _physics_config_hash(),
    }
    torch.save(artifact, output_path)
    log.info(f"Saved encoder checkpoint to {output_path}")

    history_path = output_path.with_suffix(".history.npz")
    history_arrays = {
        "epoch": np.array([h["epoch"] for h in history]),
        "train_loss": np.array([h["train_loss"] for h in history]),
        "val_loss": np.array([h["val_loss"] for h in history]),
        "lr": np.array([h["lr"] for h in history]),
        "train_pair_loss": np.array([h["train_pair_loss"] for h in history]),
        "val_pair_loss": np.array([h["val_pair_loss"] for h in history]),
        "train_t0_loss": np.array([h["train_t0_loss"] for h in history]),
        "val_t0_loss": np.array([h["val_t0_loss"] for h in history]),
        "horizons_s": np.array(cfg["horizons_s"]),
        "pitch_half_diag_m": np.array(pitch_half_diag_m),
        "height_norm_m": np.array(gen_params.height_norm_m),
        "ball_spin_norm_max_rad_s": np.array(BALL_SPIN_NORM_DIVISOR_RAD_S),
    }
    for c in (*_COMPONENTS, *_CLS_KEYS, *_R2_KEYS, *_PCTD_KEYS, *_PCTB_KEYS):
        history_arrays[f"train_{c}"] = np.stack([h[f"train_{c}"] for h in history])
        history_arrays[f"val_{c}"] = np.stack([h[f"val_{c}"] for h in history])
    np.savez_compressed(history_path, **history_arrays)
    log.info(f"Saved per-epoch history to {history_path}")

    from footballcoach.ai.physics_pretrain.report import open_in_browser, write_report
    report_path = write_report(
        history_arrays=history_arrays,
        dataset_stats=artifact["dataset_stats"],
        config_snapshot=cfg,
        normalization=normalization,
        val_examples=val_examples,
        output_path=output_path.with_suffix(".report.html"),
    )
    log.info(f"Saved training report to {report_path}")
    if open_browser:
        open_in_browser(report_path)

    artifact["history"] = history
    return artifact


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from footballcoach.ai.config import load_ai_config
    cfg = load_ai_config()["physics_pretrain"]["ball"]

    parser = argparse.ArgumentParser(description="Train the ball-dynamics encoder (physics pretraining).")
    parser.add_argument("--dataset", required=True, help="Directory of .npz shards (see ball_dataset.py's __main__).")
    parser.add_argument("--output", required=True, help="Output path for the frozen encoder checkpoint.")
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--lr", type=float, default=cfg["lr"])
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pos-weight-max", type=float, default=None,
        help="Cap the auto-computed inverse-frequency pos_weight. Defaults to "
             "physics_pretrain.ball.pos_weight_max from ai_config.json (null = uncapped) when omitted.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--open-report", dest="open_report", action="store_true", default=True,
        help="Open the training report in the default browser when training finishes (default).",
    )
    parser.add_argument(
        "--no-open-report", dest="open_report", action="store_false",
        help="Write the training report but don't open it automatically.",
    )
    parser.add_argument(
        "--init-checkpoint", default=None,
        help="Resume from a phase checkpoint (<output>.after_autoencode.pt / "
             ".after_decoder_pretrain.pt / .after_training.pt) or an older "
             "encoder-only artifact, loaded before any phase runs. Combine "
             "with the config's *_pretrain_epochs to control which phases "
             "still run on top of the restored weights (e.g. resume from "
             "after_autoencode with autoencode_pretrain_epochs=0 to skip "
             "straight to decoder-only pretrain / the main loop).",
    )
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        pos_weight_max=args.pos_weight_max,
        device=args.device,
        open_browser=args.open_report,
        init_checkpoint=args.init_checkpoint,
    )


if __name__ == "__main__":
    main()

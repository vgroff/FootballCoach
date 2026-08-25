#!/usr/bin/env python3
"""Interactively inspect a ball-dynamics pretrain checkpoint against a
dataset: picks a random episode, runs the model on it, and plots
ground-truth vs predicted trajectory/crossing/resting on a pitch. Click
"Next" (or press n / space / right arrow) to redraw with a new random row.

Needs a checkpoint that has the full model (encoder + decoder + crossing/
resting heads) -- e.g. one of ``*.midtrain_latest.pt``, ``*.after_training.
pt``, or any other ``_save_phase_checkpoint`` output from
``train_ball_dynamics.py``. The FINAL artifact saved at ``--output`` only
has ``encoder_state_dict`` (the decoder/heads are discarded, see
``ball_dynamics_net.py``'s module docstring) and can't be used here.

Closing the window runs a one-off error/input-correlation analysis: samples
``--error-samples`` (default 20,000) random episodes, predicts all of them
in one batch, and prints (to the TERMINAL, not a plot) which raw initial-
condition input variables correlate most strongly with high prediction
error -- see ``_correlate_errors_with_inputs``'s docstring.

Usage::

    uv run python scripts/inspect_ball_pretrain.py \\
        --checkpoint checkpoints/physics_pretrain/ball_encoder.midtrain_latest.pt \\
        --dataset physics_pretrain_data/ball/

    uv run python scripts/inspect_ball_pretrain.py ... --seed 0   # reproducible row sequence
    uv run python scripts/inspect_ball_pretrain.py ... --alpha 0.5   # more transparent markers/lines
"""
from __future__ import annotations

import argparse
import math
import queue
import sys
import textwrap
import threading
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from footballcoach.ai.physics_pretrain.ball_dataset import BallDynamicsDataset
from footballcoach.ai.physics_pretrain.ball_dynamics_net import BallDynamicsAutoencoder
from footballcoach.ai.physics_pretrain.ball_episode_gen import (
    BALL_SPIN_NORM_DIVISOR_RAD_S, BallEpisodeGenParams, N_TARGET_FIELDS_PER_HORIZON,
)
from footballcoach.ai.physics_pretrain.train_ball_dynamics import _migrate_crossing_head_state_dict, compute_per_episode_loss
from footballcoach.ai.physics_pretrain.latent_stats import compute_latent_stats, format_latent_stats

_C_PITCH = "#227832"
_C_LINE = "white"
_C_GT = "#4090e8"
_C_PRED = "#f5a623"
_C_BG = "#1a1a2e"
_GD = 2.45  # goal depth (visual only, not in physics config)
_DEFAULT_ALPHA = 0.75  # marker/line transparency -- see --alpha; low enough that overlapping horizon dots don't fully occlude each other


def _draw_pitch(ax, gen_params: BallEpisodeGenParams) -> None:
    hl, hw = gen_params.base_pitch_length_m / 2, gen_params.base_pitch_width_m / 2
    gw, gh = gen_params.base_goal_width_m, gen_params.base_goal_height_m
    box_l, box_w = 16.5, 40.32
    six_l, six_w = 5.5, 18.32
    lkw = dict(edgecolor=_C_LINE, facecolor="none", linewidth=1.3)

    ax.set_facecolor(_C_PITCH)
    ax.add_patch(mpatches.Rectangle((-hl, -hw), 2 * hl, 2 * hw, **lkw))
    ax.plot([0, 0], [-hw, hw], color=_C_LINE, lw=1.3)
    ax.add_patch(mpatches.Circle((0, 0), 9.15, **lkw))
    ax.plot(0, 0, "o", color=_C_LINE, ms=3, zorder=3)
    for sx in (-1, 1):
        gl = sx * hl
        ax.add_patch(mpatches.Rectangle((min(gl, gl - sx * box_l), -box_w / 2), box_l, box_w, **lkw))
        ax.add_patch(mpatches.Rectangle((min(gl, gl - sx * six_l), -six_w / 2), six_l, six_w, **lkw))
        ax.add_patch(mpatches.Rectangle(
            (min(gl, gl + sx * _GD), -gw / 2), _GD, gw,
            edgecolor=_C_LINE, facecolor="#ffffff1a", linewidth=1.6,
        ))
        _ = gh  # goal height has no top-down footprint; kept for reference only
    ax.set_xlim(-hl - _GD - 1.0, hl + _GD + 1.0)
    ax.set_ylim(-hw - 2, hw + 2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")


def _style_dark_ax(ax, title: str) -> None:
    ax.set_title(title, color="white", fontsize=10, pad=6)


def _draw_episode_boundary(ax, input_row: np.ndarray, gen_params: BallEpisodeGenParams) -> None:
    """Dashed overlay of THIS episode's own randomized pitch boundary
    (``pitch_scale_range`` -- see ``ball_episode_gen.py``'s ``_sample_pitch``)
    -- ``is_in_bounds``'s out-of-bounds check is against this boundary, NOT
    the fixed base outline ``_draw_pitch`` always draws, so a ball plotted
    inside the (bigger) base outline can still be legitimately out of
    bounds for its own (smaller) sampled pitch. Skipped when this episode's
    pitch is within 1cm of the base pitch (the overlay would just double
    the solid outline).
    """
    hl_ep = input_row[10] * gen_params.base_pitch_length_m / 2
    hw_ep = input_row[11] * gen_params.base_pitch_width_m / 2
    hl_base, hw_base = gen_params.base_pitch_length_m / 2, gen_params.base_pitch_width_m / 2
    if abs(hl_ep - hl_base) < 0.01 and abs(hw_ep - hw_base) < 0.01:
        return
    ax.add_patch(mpatches.Rectangle(
        (-hl_ep, -hw_ep), 2 * hl_ep, 2 * hw_ep,
        edgecolor="#ff5555", facecolor="none", linewidth=1.3, linestyle="--", zorder=3,
    ))
    ax.text(-hl_ep, hw_ep + 0.6, "this episode's own pitch boundary", color="#ff5555", fontsize=6.5, zorder=7)


def _episode_divisors(input_row: np.ndarray, gen_params: BallEpisodeGenParams, normalize_by_base: bool) -> tuple[float, float, float, float]:
    """(div_x, div_y, div_z, div_vel) to convert this episode's own
    normalized pos_x/pos_y/height/velocity back to metres (m/s for
    velocity) -- mirrors ball_episode_gen.py's ``_kinematics_divisors``, but
    reads the episode's own randomized pitch scale off its recorded input
    row (fields 10/11) instead of a live ``Pitch`` object, since that's all
    a saved dataset row carries. ``div_vel`` is always the (episode- or
    base-scale, per ``normalize_by_base``) half-diagonal, one shared
    divisor across vx/vy/vz -- same convention ``_kinematics_divisors``
    uses regardless of ``normalize_by_base``.
    """
    if normalize_by_base:
        half_diag = math.hypot(gen_params.base_pitch_length_m / 2, gen_params.base_pitch_width_m / 2)
        return half_diag, half_diag, half_diag, half_diag
    half_length = input_row[10] * gen_params.base_pitch_length_m / 2
    half_width = input_row[11] * gen_params.base_pitch_width_m / 2
    half_diag = math.hypot(half_length, half_width)
    return half_length, half_width, gen_params.height_norm_m, half_diag


def _fmt_xy(x: float, y: float) -> str:
    return f"({x:.1f}, {y:.1f})"


def _arrow(ax, p0: tuple[float, float], p1: tuple[float, float], color: str, alpha: float) -> None:
    ax.annotate(
        "", xy=p1, xytext=p0,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.65, alpha=alpha),
        zorder=4,
    )


def _correlate_errors_with_inputs(
    ds: BallDynamicsDataset, model: BallDynamicsAutoencoder, gen_params: BallEpisodeGenParams,
    normalize_by_base: bool, cfg: dict, n_samples: int, seed: int | None,
) -> None:
    """Samples ``min(n_samples, len(ds))`` random episodes, runs them
    through the model in one batch, computes two per-episode error
    measures, and prints (ranked by ``|Pearson r|``, most correlated
    first) which raw t=0 input variables -- in real physical units, not
    the normalized encoded ones -- correlate most strongly with each.

    The two error measures:

    - ``mean position error (m)``: average, across all recorded horizons,
      of the Euclidean (x/y only, real metres) gap between predicted and
      ground-truth position -- the same quantity the interactive
      "Trajectory" panel prints per-row, just averaged here instead of
      shown per-horizon. The most physically direct "how wrong is this
      prediction" signal.
    - ``combined per-episode training loss``: ``train_ball_dynamics.
      compute_per_episode_loss``, called with THIS run's actual ``cfg``
      ``bce_loss_weight``/``spin_loss_weight`` (NOT the function's own
      1.0/1.0 defaults) -- the exact per-row quantity the main training
      loop's batched loss is the MEAN of. Matching the real weights
      matters: at ``bce_loss_weight=0`` the oob/goal classification head
      never received gradient during training, so its logits are
      essentially arbitrary/uncalibrated noise -- using the default
      weight=1.0 here would silently mix a meaningless BCE term into
      "the loss", which can dominate the correlation ranking for reasons
      that have NOTHING to do with what the model actually learned (e.g.
      a systematically mislabeled-feeling BCE term on whichever rows
      happen to have the rarer label).

    Also runs the SAME correlation, TOP 5 only, for each of the 4
    auxiliary heads that read straight off the latent (see
    ``BallDynamicsAutoencoder``'s docstring for what each predicts):
    ``crossing_head`` (position error in metres, MASKED to rows that
    actually cross -- same masking convention as its own training loss;
    and delta_t absolute error in seconds, unmasked), ``resting_head``
    (position error in metres, masked to rows with a defined resting
    target), ``position_head`` (position error in metres, unmasked --
    always defined), ``event_head`` (summed oob+goal BCE, unmasked,
    computed with numerically-stable ``logaddexp`` -- always trained
    since it has no separate weight-gating in this analysis, unlike the
    decoder's own oob/goal BCE above).

    Input features are read straight off ``ds.inputs``' raw normalized
    fields and denormalized to real units the same way ``describe_input_
    row``/``_episode_divisors`` do (episode-own pitch scale when
    ``normalize_by_base`` is false, the fixed base pitch otherwise) --
    deliberately NOT the already-computed ``engineered features`` (input
    fields 14:19), which are themselves in normalized units and less
    directly interpretable in a printed correlation table. Also includes
    ``speed`` (overall 3D velocity magnitude) alongside the signed
    ``vel_x/y/z`` components, ``vertical_speed`` (``|vel_z|`` specifically
    -- distinct from ``speed``, which conflates horizontal and vertical
    motion, and from signed ``vel_z``, whose correlation can wash out
    toward zero even when vertical speed matters if fast-rising and
    fast-falling episodes drive similarly elevated error), ``dist_to_x/y_
    boundary`` (how close to the pitch edge the episode starts), and
    ``already_oob_or_goal_at_start`` (0/1, via
    ``compute_already_out_of_bounds_at_start_mask`` -- Pearson correlation
    against a 0/1 indicator is the point-biserial correlation, a standard
    and valid special case) since both are plausible error drivers not
    literally present as a single raw input field.

    NOTE on reading the printed ``r``/``slope`` values: Pearson's r is a
    unitless measure of LINEAR association strength (-1..1), NOT a
    regression slope -- "r=0.6 for speed" does NOT mean "0.6m of extra
    error per extra m/s of speed" (``r^2`` is the fraction of the error's
    variance linearly explained by that one feature, which is the closest
    r itself gets to a directly-interpretable number). The separately
    printed ``slope`` IS that "extra [error unit] per [feature unit]"
    figure -- the ordinary-least-squares slope of error regressed on the
    feature ALONE (``cov(feature, error) / var(feature)``, equivalently
    ``r * std(error) / std(feature)``) -- read its units as (whatever this
    section's error is measured in) per (whatever that feature's own
    listed unit is), e.g. "slope=+0.842/unit" under "speed (m/s)" in the
    "mean position error (m, ...)" section means +0.842m of error per
    extra m/s of speed. A univariate slope like this doesn't account for
    other correlated features (e.g. speed and vel_z moving together) --
    it's "how error trends with this feature alone," not an isolated
    causal effect.

    Pure diagnostic printout -- returns nothing, mutates nothing.
    """
    rng = np.random.default_rng(seed)
    n = len(ds)
    sample_size = min(n_samples, n)
    idx = rng.choice(n, size=sample_size, replace=False)

    inputs = ds.inputs[idx]
    targets = ds.targets[idx]
    n_horizons = len(cfg["horizons_s"])
    # Match the ACTUAL trained weights, not compute_per_episode_loss's own
    # 1.0/1.0 defaults -- see this function's docstring.
    bce_weight = float(cfg.get("bce_loss_weight", 1.0))
    spin_weight = float(cfg.get("spin_loss_weight", 1.0))

    with torch.no_grad():
        x = torch.from_numpy(inputs.astype(np.float32, copy=False))
        latent = model.encoder(x)
        decoder_outs = model.decoder(latent)
        crossing_pred = model.crossing_head(latent).numpy()
        resting_pred = model.resting_head(latent).numpy()
        position_pred = model.position_head(latent).numpy()
        event_pred = model.event_head(latent).numpy()

    if normalize_by_base:
        half_diag = math.hypot(gen_params.base_pitch_length_m / 2, gen_params.base_pitch_width_m / 2)
        half_length = np.full(sample_size, gen_params.base_pitch_length_m / 2)
        half_width = np.full(sample_size, gen_params.base_pitch_width_m / 2)
        height_div = np.full(sample_size, half_diag)
        vel_div = np.full(sample_size, half_diag)
    else:
        half_length = inputs[:, 10] * gen_params.base_pitch_length_m / 2
        half_width = inputs[:, 11] * gen_params.base_pitch_width_m / 2
        height_div = np.full(sample_size, gen_params.height_norm_m)
        vel_div = np.hypot(half_length, half_width)

    pos_err_m = np.zeros(sample_size)
    for h, out in enumerate(decoder_outs):
        out_np = out.numpy()
        base = h * N_TARGET_FIELDS_PER_HORIZON
        gt = targets[:, base:base + N_TARGET_FIELDS_PER_HORIZON]
        dx = (out_np[:, 0] - gt[:, 0]) * half_length
        dy = (out_np[:, 1] - gt[:, 1]) * half_width
        pos_err_m += np.hypot(dx, dy)
    pos_err_m /= n_horizons

    pos_weight = torch.from_numpy(ds.compute_pos_weights(n_horizons, indices=idx))
    target_t = torch.from_numpy(targets.astype(np.float32, copy=False))
    combined_loss = compute_per_episode_loss(
        list(decoder_outs), target_t, pos_weight, bce_weight=bce_weight, spin_weight=spin_weight,
    ).numpy()

    pos_x = inputs[:, 0] * half_length
    pos_y = inputs[:, 1] * half_width
    pos_z = inputs[:, 2] * height_div
    vel_x = inputs[:, 3] * vel_div
    vel_y = inputs[:, 4] * vel_div
    vel_z = inputs[:, 5] * vel_div
    speed = np.sqrt(vel_x ** 2 + vel_y ** 2 + vel_z ** 2)
    # |vel_z| specifically, distinct from the overall 3D `speed` above
    # (which conflates horizontal and vertical motion) and from the raw
    # SIGNED vel_z (whose correlation with error can wash out toward zero
    # even when vertical speed genuinely matters, if fast-rising and
    # fast-falling episodes both drive similarly elevated error -- a
    # signed linear correlation can't see a same-magnitude-either-sign
    # effect). Isolates vertical speed at whatever moment t=0 is, relevant
    # since ground-bounce dynamics depend specifically on vertical speed
    # at contact, not the horizontal components speed/vel_x/vel_y already
    # capture.
    vertical_speed = np.abs(vel_z)
    spin_x = inputs[:, 6] * BALL_SPIN_NORM_DIVISOR_RAD_S
    spin_y = inputs[:, 7] * BALL_SPIN_NORM_DIVISOR_RAD_S
    spin_z = inputs[:, 8] * BALL_SPIN_NORM_DIVISOR_RAD_S
    spin_mag = np.sqrt(spin_x ** 2 + spin_y ** 2 + spin_z ** 2)
    # Computed once, reused below both as its own feature AND to replicate
    # train_ball_dynamics.py's train()'s exact crossing_mask/crossing_dt
    # override for these rows (see the crossing_head block below) -- without
    # that override this analysis silently compares crossing_head against
    # ground truth for rows the model was NEVER trained to match on this
    # head (already-oob-at-start episodes are excluded from crossing_head's
    # position loss, and their crossing_dt target is trained toward the -1
    # sentinel, not the raw near-immediate crossing time) -- a mismatch
    # that shows up here as an inflated, misleading correlation with this
    # exact flag rather than reflecting anything about model quality.
    already_oob_bool = ds.compute_already_out_of_bounds_at_start_mask(gen_params, indices=idx)
    already_oob_or_goal = already_oob_bool.astype(np.float64)

    features = {
        "pos_x (m)": pos_x, "pos_y (m)": pos_y, "height/pos_z (m)": pos_z,
        "vel_x (m/s)": vel_x, "vel_y (m/s)": vel_y, "vel_z (m/s)": vel_z, "speed (m/s)": speed,
        "vertical_speed |vel_z| (m/s)": vertical_speed,
        "spin_x (rad/s)": spin_x, "spin_y (rad/s)": spin_y, "spin_z (rad/s)": spin_z, "spin_mag (rad/s)": spin_mag,
        "restitution": inputs[:, 9].astype(np.float64),
        "pitch_length (m)": inputs[:, 10] * gen_params.base_pitch_length_m,
        "pitch_width (m)": inputs[:, 11] * gen_params.base_pitch_width_m,
        "goal_width (m)": inputs[:, 12] * gen_params.base_goal_width_m,
        "goal_height (m)": inputs[:, 13] * gen_params.base_goal_height_m,
        "dist_to_x_boundary (m)": half_length - np.abs(pos_x),
        "dist_to_y_boundary (m)": half_width - np.abs(pos_y),
        "dist_to_center (m)": np.hypot(pos_x, pos_y),
        "already_oob_or_goal_at_start": already_oob_or_goal,
    }

    def _print_ranked(label: str, err: np.ndarray, mask: np.ndarray | None = None, top_k: int | None = None) -> None:
        if mask is not None:
            feats = {name: vals[mask] for name, vals in features.items()}
            err = err[mask]
            n_used = int(mask.sum())
        else:
            feats = features
            n_used = sample_size
        print(f"\n--- correlates with {label} (n={n_used:,}) ---")
        if n_used < 2:
            print("    (too few valid rows to correlate)")
            return
        rows = []
        for name, vals in feats.items():
            feat_std, err_std = np.std(vals), np.std(err)
            if feat_std < 1e-12 or err_std < 1e-12:
                continue  # a constant feature/error this sample has no defined correlation
            r = float(np.corrcoef(vals, err)[0, 1])
            # Effect size (regression slope), in the error metric's own
            # units per unit of the feature -- r alone is unitless
            # association STRENGTH, not magnitude (see this function's
            # docstring note): slope = r * std(err) / std(feature) is the
            # actual "how much extra error per unit of this feature" a
            # simple linear fit would report, for the SAME feature/error
            # pair r was computed from.
            slope = r * err_std / feat_std
            rows.append((name, r, slope))
        rows.sort(key=lambda t: -abs(t[1]))
        if top_k is not None:
            rows = rows[:top_k]
        for name, r, slope in rows:
            bar = "#" * int(round(abs(r) * 40))
            print(f"    {name:24s} r={r:+.3f}  slope={slope:+.4g}  {bar}")

    print(f"\n========== error / input correlation analysis ({sample_size:,} random episodes) ==========")
    _print_ranked("mean position error (m, averaged over horizons)", pos_err_m)
    _print_ranked(f"combined per-episode training loss (bce_weight={bce_weight:g}, spin_weight={spin_weight:g})", combined_loss)

    print("\n---------- auxiliary heads (own separate parameters, top 5 correlates each) ----------")
    if ds.crossing_pos is not None:
        # Same already-oob-at-start override train() applies in place
        # before training (see this function's docstring / the comment
        # above already_oob_bool) -- mask OUT those rows from the position
        # term, and override their crossing_dt target to the -1 sentinel,
        # so this analysis measures the model against the SAME targets it
        # was actually trained against, not the raw pre-override dataset.
        c_mask = ds.crossing_mask[idx] & ~already_oob_bool
        c_pos = ds.crossing_pos[idx]
        c_dt = np.where(already_oob_bool, -1.0, ds.crossing_dt[idx])
        crossing_pos_err_m = np.hypot(
            (crossing_pred[:, 0] - c_pos[:, 0]) * half_length, (crossing_pred[:, 1] - c_pos[:, 1]) * half_width,
        )
        crossing_dt_err_s = np.abs(crossing_pred[:, 2] - c_dt)
        _print_ranked("crossing_head position error (m, rows that actually cross only)", crossing_pos_err_m, mask=c_mask, top_k=5)
        _print_ranked("crossing_head delta_t error (s, all rows incl. -1 sentinel)", crossing_dt_err_s, top_k=5)

    resting_min_start_speed_mps = float(cfg.get("resting_min_start_speed_mps", 1.5))
    resting_speed_threshold_mps = float(cfg.get("resting_speed_threshold_mps", 0.01))
    pitch_half_diag_m = math.hypot(gen_params.base_pitch_length_m / 2, gen_params.base_pitch_width_m / 2)
    resting_pos_all, resting_mask_all = ds.compute_resting_targets(
        min_start_speed_norm=resting_min_start_speed_mps / pitch_half_diag_m,
        rest_speed_norm=resting_speed_threshold_mps / pitch_half_diag_m,
    )
    r_mask = resting_mask_all[idx]
    r_pos = resting_pos_all[idx]
    resting_pos_err_m = np.hypot(
        (resting_pred[:, 0] - r_pos[:, 0]) * half_length, (resting_pred[:, 1] - r_pos[:, 1]) * half_width,
    )
    _print_ranked("resting_head position error (m, rows with a defined resting target only)", resting_pos_err_m, mask=r_mask, top_k=5)

    position_err_m = np.hypot((position_pred[:, 0] - inputs[:, 0]) * half_length, (position_pred[:, 1] - inputs[:, 1]) * half_width)
    _print_ranked("position_head position error (m, current t=0 position)", position_err_m, top_k=5)

    ever_oob, ever_goal = ds.compute_event_ever_masks(gen_params, indices=idx)

    def _bce_with_logits(logit: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.logaddexp(0.0, logit) - target * logit

    event_bce = (
        _bce_with_logits(event_pred[:, 0], ever_oob.astype(np.float64))
        + _bce_with_logits(event_pred[:, 1], ever_goal.astype(np.float64))
    )
    _print_ranked("event_head BCE (ever-oob + ever-goal, unmasked)", event_bce, top_k=5)
    print()


class Inspector:
    """``_prefetch_loop`` runs on a background daemon thread, pulling random
    rows through the model (encoder + decoder + crossing/resting heads --
    the only part worth overlapping with UI time, prediction being the slow
    step relative to a plain array index) and pushing the results onto a
    small bounded queue. The main/UI thread never predicts -- it just pops
    the next-ready item and renders it, so clicking "Next" only pays for a
    redraw, not a forward pass, PROVIDED the queue has had time to refill
    since the last click (``maxsize=2`` keeps one row buffered ahead of
    whatever's currently on screen without prefetching arbitrarily far
    ahead of what the user will actually look at).
    """

    def __init__(
        self, ds: BallDynamicsDataset, model: BallDynamicsAutoencoder, cfg: dict,
        gen_params: BallEpisodeGenParams, normalize_by_base: bool, seed: int | None,
        alpha: float = _DEFAULT_ALPHA, error_samples: int = 20_000,
    ):
        self.ds = ds
        self.model = model
        self.cfg = cfg
        self.horizons_s = list(cfg["horizons_s"])
        self.gen_params = gen_params
        self.normalize_by_base = normalize_by_base
        self.alpha = alpha
        self.error_samples = error_samples
        self.rng = np.random.default_rng(seed)  # only ever touched by the prefetch thread -- single producer, no lock needed

        resting_min_start_speed_mps = float(cfg.get("resting_min_start_speed_mps", 1.5))
        resting_speed_threshold_mps = float(cfg.get("resting_speed_threshold_mps", 0.01))
        pitch_half_diag_m = math.hypot(gen_params.base_pitch_length_m / 2, gen_params.base_pitch_width_m / 2)
        self.resting_pos_all, self.resting_mask_all = ds.compute_resting_targets(
            min_start_speed_norm=resting_min_start_speed_mps / pitch_half_diag_m,
            rest_speed_norm=resting_speed_threshold_mps / pitch_half_diag_m,
        )

        self._queue: queue.Queue[dict] = queue.Queue(maxsize=2)
        self._prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._prefetch_thread.start()

        self.fig, (self.ax_traj, self.ax_cross, self.ax_rest) = plt.subplots(
            1, 3, figsize=(16, 8.5), facecolor=_C_BG,
        )
        self.fig.subplots_adjust(bottom=0.34, top=0.68, wspace=0.15)
        self.ax_table = self.fig.add_axes((0.06, 0.09, 0.88, 0.20))
        self.ax_table.set_facecolor(_C_BG)
        self.ax_table.axis("off")
        ax_btn = self.fig.add_axes((0.44, 0.01, 0.12, 0.045))
        self.btn = Button(ax_btn, "Next ▶", color="#333355", hovercolor="#4a4a7a")
        self.btn.label.set_color("white")
        self.btn.on_clicked(lambda _event: self.next_row())
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self.next_row()

    def _on_key(self, event) -> None:
        if event.key in ("n", " ", "right"):
            self.next_row()

    def _on_close(self, _event) -> None:
        # Runs synchronously as the window tears down -- fine here since
        # this only reads (never touches the plot), and the prefetch
        # thread's own model calls don't overlap with this one in any way
        # that shares mutable state (torch inference has none here).
        print(f"\nWindow closed -- running error/input correlation analysis on {self.error_samples:,} random episodes...")
        _correlate_errors_with_inputs(
            self.ds, self.model, self.gen_params, self.normalize_by_base, self.cfg,
            n_samples=self.error_samples, seed=None,
        )

    def _prefetch_loop(self) -> None:
        while True:
            idx = int(self.rng.integers(0, len(self.ds)))
            data = self._predict_row(idx)
            self._queue.put(data)  # blocks once 2 rows are buffered, pausing prediction until the UI consumes one

    def _predict_row(self, idx: int) -> dict:
        """Pure model-inference step, no matplotlib calls -- safe to run
        off the main thread (torch eval-mode inference has no shared
        mutable state here, and this class never predicts from the UI
        thread once running, so there's no cross-thread access to the same
        tensors to race on)."""
        ds, gen_params = self.ds, self.gen_params
        input_row = ds.inputs[idx]
        div_x, div_y, _div_z, div_vel = _episode_divisors(input_row, gen_params, self.normalize_by_base)
        with torch.no_grad():
            x = torch.from_numpy(input_row[None].astype(np.float32))
            latent = self.model.encoder(x)
            decoder_outs = [o[0].numpy() for o in self.model.decoder(latent)]
            crossing_pred = self.model.crossing_head(latent)[0].numpy()
            resting_pred = self.model.resting_head(latent)[0].numpy()
        return {
            "idx": idx, "input_row": input_row, "div_x": div_x, "div_y": div_y, "div_vel": div_vel,
            "decoder_outs": decoder_outs, "crossing_pred": crossing_pred, "resting_pred": resting_pred,
        }

    def next_row(self) -> None:
        self._render(self._queue.get())

    def _render(self, data: dict) -> None:
        idx = data["idx"]
        ds, gen_params = self.ds, self.gen_params
        input_row, div_x, div_y, div_vel = data["input_row"], data["div_x"], data["div_y"], data["div_vel"]
        decoder_outs, crossing_pred, resting_pred = data["decoder_outs"], data["crossing_pred"], data["resting_pred"]

        # ---- Trajectory ----
        ax = self.ax_traj
        ax.clear()
        _draw_pitch(ax, gen_params)
        _draw_episode_boundary(ax, input_row, gen_params)

        start_xy = (input_row[0] * div_x, input_row[1] * div_y)
        gt_pts = [start_xy]
        for h in range(len(self.horizons_s)):
            base = h * N_TARGET_FIELDS_PER_HORIZON
            block = ds.targets[idx, base:base + N_TARGET_FIELDS_PER_HORIZON]
            gt_pts.append((block[0] * div_x, block[1] * div_y))
        pred_pts = [start_xy]
        for h in range(len(self.horizons_s)):
            block = decoder_outs[h]
            pred_pts.append((block[0] * div_x, block[1] * div_y))

        # Euclidean (x/y-only, real metres) gap between gt/pred at each
        # horizon -- printed above the plot since eyeballing arrow-tip
        # distance on a shared pitch is imprecise at short horizons where
        # the gap is a few centimetres.
        pos_errs_m = [
            math.hypot(gt_pts[h + 1][0] - pred_pts[h + 1][0], gt_pts[h + 1][1] - pred_pts[h + 1][1])
            for h in range(len(self.horizons_s))
        ]
        entries = [f"{t:g}s: {e:.2f}m" for t, e in zip(self.horizons_s, pos_errs_m)]
        # Wrapped rather than one long line -- with the full 8-horizon config
        # a single line runs well past this axis' own width and bleeds into
        # the next panel's title (matplotlib doesn't clip axes titles).
        err_lines = textwrap.wrap("   ".join(entries), width=42)
        title = "\n".join([f"Trajectory by horizon  (row {idx})", "pos error (m):", *err_lines])
        ax.set_title(title, color="white", fontsize=8.5, pad=8, linespacing=1.4)

        for pts, color, label in ((gt_pts, _C_GT, "ground truth"), (pred_pts, _C_PRED, "predicted")):
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "o", color=color, ms=4.5, alpha=self.alpha, zorder=5, mec="white", mew=0.4, label=label)
            for i in range(len(pts) - 1):
                _arrow(ax, pts[i], pts[i + 1], color, self.alpha)
        ax.plot(*start_xy, "o", color="white", ms=4.5, alpha=0.3, zorder=6, mec="white", mew=0.4)
        for h, (gx, gy) in enumerate(gt_pts[1:]):
            ax.text(gx + 0.8, gy + 0.8, f"t={self.horizons_s[h]:g}s", color="#cccccc", fontsize=6, zorder=7)
        ax.legend(loc="upper left", fontsize=8, facecolor=_C_BG, edgecolor="none", labelcolor="white", framealpha=0.75)

        # ---- Crossing point ----
        ax = self.ax_cross
        ax.clear()
        _draw_pitch(ax, gen_params)
        _draw_episode_boundary(ax, input_row, gen_params)
        _style_dark_ax(ax, "Crossing point: gt vs pred")
        ax.plot(*start_xy, "o", color="white", ms=4.5, alpha=0.3, zorder=6, mec="white", mew=0.4)

        pred_dt = float(crossing_pred[2])
        if pred_dt >= 0:
            pred_cross_xy = (crossing_pred[0] * div_x, crossing_pred[1] * div_y)
            ax.plot(*pred_cross_xy, "^", color=_C_PRED, ms=10, alpha=self.alpha, zorder=6, mec="white", mew=0.5, label="predicted")
            ax.text(pred_cross_xy[0] + 0.8, pred_cross_xy[1] + 0.8, f"pred dt={pred_dt:.2f}s",
                    color=_C_PRED, fontsize=8, zorder=7)
        else:
            ax.text(0, gen_params.base_pitch_width_m / 2 + 3.0, f"predicted: never crosses (dt={pred_dt:.2f}s)",
                    color=_C_PRED, fontsize=8, ha="center")

        if ds.crossing_mask is not None and ds.crossing_mask[idx]:
            gt_cross_xy = (ds.crossing_pos[idx, 0] * div_x, ds.crossing_pos[idx, 1] * div_y)
            gt_dt = float(ds.crossing_dt[idx])
            ax.plot(*gt_cross_xy, "^", color=_C_GT, ms=10, alpha=self.alpha, zorder=6, mec="white", mew=0.5, label="ground truth")
            ax.text(gt_cross_xy[0] + 0.8, gt_cross_xy[1] - 1.6, f"gt dt={gt_dt:.2f}s",
                    color=_C_GT, fontsize=8, zorder=7)
        else:
            ax.text(0, -gen_params.base_pitch_width_m / 2 - 4.5, "ground truth: never crosses in this window",
                    color=_C_GT, fontsize=8, ha="center")
        ax.legend(loc="upper left", fontsize=8, facecolor=_C_BG, edgecolor="none", labelcolor="white", framealpha=0.75)

        # ---- Resting point ----
        ax = self.ax_rest
        ax.clear()
        _draw_pitch(ax, gen_params)
        _draw_episode_boundary(ax, input_row, gen_params)
        _style_dark_ax(ax, "Resting point: gt vs pred")
        ax.plot(*start_xy, "o", color="white", ms=4.5, alpha=0.3, zorder=6, mec="white", mew=0.4)

        pred_rest_xy = (resting_pred[0] * div_x, resting_pred[1] * div_y)
        ax.plot(*pred_rest_xy, "s", color=_C_PRED, ms=9, alpha=self.alpha, zorder=6, mec="white", mew=0.5, label="predicted")

        if self.resting_mask_all[idx]:
            gt_rest_xy = (self.resting_pos_all[idx, 0] * div_x, self.resting_pos_all[idx, 1] * div_y)
            ax.plot(*gt_rest_xy, "s", color=_C_GT, ms=9, alpha=self.alpha, zorder=6, mec="white", mew=0.5, label="ground truth")
        else:
            ax.text(0, -gen_params.base_pitch_width_m / 2 - 4.5, "ground truth: never comes to rest / too slow at start",
                    color=_C_GT, fontsize=8, ha="center")
        ax.legend(loc="upper left", fontsize=8, facecolor=_C_BG, edgecolor="none", labelcolor="white", framealpha=0.75)

        # ---- Position/velocity table ----
        ax = self.ax_table
        ax.clear()
        ax.axis("off")
        ax.set_facecolor(_C_BG)

        vx0, vy0 = input_row[3] * div_vel, input_row[4] * div_vel
        rows = [["0.0s (start)", _fmt_xy(*start_xy), "—", _fmt_xy(vx0, vy0), "—"]]
        for h, t in enumerate(self.horizons_s):
            base = h * N_TARGET_FIELDS_PER_HORIZON
            gt_block = ds.targets[idx, base:base + N_TARGET_FIELDS_PER_HORIZON]
            pred_block = decoder_outs[h]
            rows.append([
                f"{t:g}s",
                _fmt_xy(*gt_pts[h + 1]), _fmt_xy(*pred_pts[h + 1]),
                _fmt_xy(gt_block[3] * div_vel, gt_block[4] * div_vel),
                _fmt_xy(pred_block[3] * div_vel, pred_block[4] * div_vel),
            ])
        col_labels = ["horizon", "gt pos (m)", "pred pos (m)", "gt vel (m/s)", "pred vel (m/s)"]
        table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.35)
        for (r, _c), cell in table.get_celld().items():
            cell.set_edgecolor("#444466")
            cell.set_facecolor("#2a2a4a" if r == 0 else _C_BG)
            cell.get_text().set_color("white")
            if r == 0:
                cell.get_text().set_fontweight("bold")

        self.fig.suptitle(
            f"Ball-dynamics inspector  —  dataset row {idx}  ({len(self.ds):,} rows total)",
            color="white", fontsize=11,
        )
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path, help="Full-model checkpoint (has model_state_dict), not the encoder-only final artifact.")
    ap.add_argument("--dataset", required=True, type=Path, help="Directory of .npz shards (e.g. physics_pretrain_data/ball/).")
    ap.add_argument("--seed", type=int, default=None, help="Seed for the random row sequence (omit for a fresh sequence each run).")
    ap.add_argument(
        "--alpha", type=float, default=_DEFAULT_ALPHA,
        help=f"Marker/line transparency, 0 (invisible) to 1 (opaque) -- default {_DEFAULT_ALPHA}. "
             "Lower it further if overlapping horizon dots still occlude each other.",
    )
    ap.add_argument(
        "--error-samples", type=int, default=20_000,
        help="How many random episodes to sample for the on-close error/input correlation analysis (default 20,000).",
    )
    ap.add_argument(
        "--linear-decoder", action="store_true", default=None,
        help="Force-build BallDynamicsLinearDecoder regardless of the checkpoint's own "
             "config_snapshot['linear_decoder_enabled']. Needed for a checkpoint saved by a training process that "
             "was already running before linear-decoder support existed/was fixed here -- its config_snapshot is "
             "stale (still says false) even though the actual saved decoder.* weights are the linear-decoder "
             "shape, since that process never picked up the code change. Omit to trust config_snapshot as usual "
             "(correct for any checkpoint saved after a restart).",
    )
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" not in ckpt:
        sys.exit(
            f"{args.checkpoint} has no 'model_state_dict' (only an encoder). "
            "Pass a phase checkpoint instead, e.g. one ending in "
            "'.midtrain_latest.pt' or '.after_training.pt'."
        )
    cfg = ckpt["config_snapshot"]
    linear_decoder = cfg.get("linear_decoder_enabled", False) if args.linear_decoder is None else args.linear_decoder
    # encoder_leaky_relu_negative_slope isn't part of the saved weights
    # (LeakyReLU has no learnable params), so unlike hidden_dim/latent_dim/
    # etc. above (which MUST come from this checkpoint's own config_snapshot
    # to match its actual saved shapes), this always reflects the CURRENT
    # live config, even for a checkpoint saved before this setting existed
    # or trained under a different slope (plain ReLU included) -- see its
    # config comment in ai_config.json.
    from footballcoach.ai.config import load_ai_config
    leaky_relu_negative_slope = load_ai_config()["physics_pretrain"]["ball"].get("encoder_leaky_relu_negative_slope", 0.0)

    model = BallDynamicsAutoencoder(
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
        leaky_relu_negative_slope=leaky_relu_negative_slope,
    )
    missing, unexpected = model.load_state_dict(
        _migrate_crossing_head_state_dict(ckpt["model_state_dict"], model), strict=False,
    )
    if missing:
        print(f"Note: checkpoint missing {len(missing)} param(s), left at fresh init: {missing}")
    if unexpected:
        print(f"Note: checkpoint had {len(unexpected)} unexpected param(s), ignored: {unexpected}")
    model.eval()

    ds = BallDynamicsDataset.from_directory(args.dataset)
    gen_params = BallEpisodeGenParams.from_config()
    normalize_by_base = bool(cfg.get("normalize_kinematics_by_base_pitch", gen_params.normalize_kinematics_by_base_pitch))

    # Post-TRAINING capacity snapshot of EVERY stage of the encoder, not
    # just its final latent output -- train_ball_dynamics.py only ever logs
    # the final-latent version of this, once, at epoch 0 (before any
    # gradient step), which can't tell you how much of latent_dim the
    # FINAL, converged encoder actually uses, and says nothing at all about
    # hidden_dim/encoder_bottleneck_dim (internal activations that are
    # never otherwise inspected). compute_latent_stats works on any
    # nn.Module, so it's pointed here at increasingly-deep PREFIXES of
    # model.encoder.trunk (a plain nn.Sequential -- see BallDynamicsEncoder
    # -- [0:2]=Linear+ReLU #1, out width hidden_dim; [0:4]=...+Linear+ReLU
    # #2, still hidden_dim; [0:6]=full trunk, out width encoder_bottleneck_
    # dim) as well as the full encoder (out width latent_dim). A low
    # effective_rank_participation_ratio/n_components_for_95pct_variance
    # relative to that STAGE's own width -- or a nonzero n_dead_dims (ReLU
    # units that are exactly 0 for every row, a hard floor on wasted
    # capacity, only meaningful for the two hidden-layer stages since the
    # bottleneck/latent stages end in Linear, not ReLU) -- at any stage is
    # a concrete, checkpoint-specific signal that stage's own width could
    # shrink without losing accuracy, not just latent_dim.
    trunk_children = list(model.encoder.trunk.children())
    stages = [
        ("hidden layer 1 (post-ReLU)", torch.nn.Sequential(*trunk_children[0:2])),
        ("hidden layer 2 (post-ReLU)", torch.nn.Sequential(*trunk_children[0:4])),
        ("bottleneck (post-ReLU)", torch.nn.Sequential(*trunk_children[0:6])),
        ("final latent", model.encoder),
    ]
    for label, stage_module in stages:
        print(f"--- {label} ---")
        print(format_latent_stats(compute_latent_stats(stage_module, ds.inputs, device="cpu")))
        print()

    inspector = Inspector(
        ds, model, cfg, gen_params, normalize_by_base, args.seed, alpha=args.alpha, error_samples=args.error_samples,
    )
    inspector.show()


if __name__ == "__main__":
    main()

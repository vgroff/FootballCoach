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

Usage::

    uv run python scripts/inspect_ball_pretrain.py \\
        --checkpoint checkpoints/physics_pretrain/ball_encoder.midtrain_latest.pt \\
        --dataset physics_pretrain_data/ball/

    uv run python scripts/inspect_ball_pretrain.py ... --seed 0   # reproducible row sequence
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
from footballcoach.ai.physics_pretrain.ball_episode_gen import BallEpisodeGenParams, N_TARGET_FIELDS_PER_HORIZON
from footballcoach.ai.physics_pretrain.train_ball_dynamics import _migrate_crossing_head_state_dict

_C_PITCH = "#227832"
_C_LINE = "white"
_C_GT = "#4090e8"
_C_PRED = "#f5a623"
_C_BG = "#1a1a2e"
_GD = 2.45  # goal depth (visual only, not in physics config)


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


def _arrow(ax, p0: tuple[float, float], p1: tuple[float, float], color: str) -> None:
    ax.annotate(
        "", xy=p1, xytext=p0,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, alpha=0.85),
        zorder=4,
    )


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
    ):
        self.ds = ds
        self.model = model
        self.horizons_s = list(cfg["horizons_s"])
        self.gen_params = gen_params
        self.normalize_by_base = normalize_by_base
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

        self.next_row()

    def _on_key(self, event) -> None:
        if event.key in ("n", " ", "right"):
            self.next_row()

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
            ax.plot(xs, ys, "o", color=color, ms=6, zorder=5, mec="white", mew=0.5, label=label)
            for i in range(len(pts) - 1):
                _arrow(ax, pts[i], pts[i + 1], color)
        ax.plot(*start_xy, "*", color="white", ms=14, zorder=6, mec="black", mew=0.5)
        for h, (gx, gy) in enumerate(gt_pts[1:]):
            ax.text(gx + 0.8, gy + 0.8, f"t={self.horizons_s[h]:g}s", color="#cccccc", fontsize=6, zorder=7)
        ax.legend(loc="upper left", fontsize=8, facecolor=_C_BG, edgecolor="none", labelcolor="white", framealpha=0.75)

        # ---- Crossing point ----
        ax = self.ax_cross
        ax.clear()
        _draw_pitch(ax, gen_params)
        _draw_episode_boundary(ax, input_row, gen_params)
        _style_dark_ax(ax, "Crossing point: gt vs pred")
        ax.plot(*start_xy, "*", color="white", ms=12, zorder=6, mec="black", mew=0.5)

        pred_dt = float(crossing_pred[2])
        if pred_dt >= 0:
            pred_cross_xy = (crossing_pred[0] * div_x, crossing_pred[1] * div_y)
            ax.plot(*pred_cross_xy, "^", color=_C_PRED, ms=13, zorder=6, mec="white", mew=0.6, label="predicted")
            ax.text(pred_cross_xy[0] + 0.8, pred_cross_xy[1] + 0.8, f"pred dt={pred_dt:.2f}s",
                    color=_C_PRED, fontsize=8, zorder=7)
        else:
            ax.text(0, gen_params.base_pitch_width_m / 2 + 3.0, f"predicted: never crosses (dt={pred_dt:.2f}s)",
                    color=_C_PRED, fontsize=8, ha="center")

        if ds.crossing_mask is not None and ds.crossing_mask[idx]:
            gt_cross_xy = (ds.crossing_pos[idx, 0] * div_x, ds.crossing_pos[idx, 1] * div_y)
            gt_dt = float(ds.crossing_dt[idx])
            ax.plot(*gt_cross_xy, "^", color=_C_GT, ms=13, zorder=6, mec="white", mew=0.6, label="ground truth")
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
        ax.plot(*start_xy, "*", color="white", ms=12, zorder=6, mec="black", mew=0.5)

        pred_rest_xy = (resting_pred[0] * div_x, resting_pred[1] * div_y)
        ax.plot(*pred_rest_xy, "s", color=_C_PRED, ms=11, zorder=6, mec="white", mew=0.6, label="predicted")

        if self.resting_mask_all[idx]:
            gt_rest_xy = (self.resting_pos_all[idx, 0] * div_x, self.resting_pos_all[idx, 1] * div_y)
            ax.plot(*gt_rest_xy, "s", color=_C_GT, ms=11, zorder=6, mec="white", mew=0.6, label="ground truth")
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
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" not in ckpt:
        sys.exit(
            f"{args.checkpoint} has no 'model_state_dict' (only an encoder). "
            "Pass a phase checkpoint instead, e.g. one ending in "
            "'.midtrain_latest.pt' or '.after_training.pt'."
        )
    cfg = ckpt["config_snapshot"]

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

    inspector = Inspector(ds, model, cfg, gen_params, normalize_by_base, args.seed)
    inspector.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Interactively inspect a player-dynamics pretrain checkpoint against a
dataset: picks a random episode, runs the model on it, and plots
ground-truth vs predicted trajectory/heading/stamina on a pitch. Click
"Next" (or press n / space / right arrow) to redraw with a new random row.

Needs a checkpoint that has the full model (encoder + decoder) -- e.g. one
of ``*.midtrain_latest.pt``, ``*.after_training.pt``, or any other
``_save_phase_checkpoint`` output from ``train_player_dynamics.py``. The
FINAL artifact saved at ``--output`` only has ``encoder_state_dict`` (the
decoder is discarded, see ``player_dynamics_net.py``'s module docstring)
and can't be used here.

Unlike ``inspect_ball_pretrain.py``, there is no crossing/resting panel --
``PlayerDynamicsAutoencoder`` has no ``crossing_head``/``resting_head``
equivalent (players don't get "frozen" at an out-of-bounds/goal event, see
``player_episode_gen.py``'s module docstring). Instead this shows a
per-point heading indicator (a short tick mark, gt vs pred) on the
trajectory panel, and a dedicated stamina-over-horizon panel, since both
are genuinely new information the ball inspector never had to display.

Usage::

    uv run python scripts/inspect_player_pretrain.py \\
        --checkpoint checkpoints/physics_pretrain/player_encoder.midtrain_latest.pt \\
        --dataset physics_pretrain_data/player/

    uv run python scripts/inspect_player_pretrain.py ... --seed 0   # reproducible row sequence
"""
from __future__ import annotations

import argparse
import dataclasses
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

from footballcoach.ai.physics_pretrain.player_dataset import PlayerDynamicsDataset
from footballcoach.ai.physics_pretrain.player_dynamics_net import PlayerDynamicsAutoencoder
from footballcoach.ai.physics_pretrain.player_episode_gen import N_TARGET_FIELDS_PER_HORIZON, PlayerEpisodeGenParams
from footballcoach.ai.physics_pretrain.train_player_dynamics import _kinematics_denorm_scales, _pitch_dims_m

_C_PITCH = "#227832"
_C_LINE = "white"
_C_GT = "#4090e8"
_C_PRED = "#f5a623"
_C_BG = "#1a1a2e"
_GD = 2.45  # goal depth (visual only, not in physics config)
_HEADING_TICK_LEN_M = 2.2  # length of the gt/pred heading tick marks drawn on the trajectory panel


def _draw_pitch(ax, gen_params: PlayerEpisodeGenParams) -> None:
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


def _draw_episode_boundary(ax, input_row: np.ndarray, gen_params: PlayerEpisodeGenParams) -> None:
    """Dashed overlay of THIS episode's own randomized pitch boundary
    (``pitch_scale_range`` -- see ``player_episode_gen.py``'s
    ``_sample_pitch``) -- the out-of-bounds target is computed against this
    boundary, NOT the fixed base outline ``_draw_pitch`` always draws, so a
    player plotted inside the (bigger) base outline can still be
    legitimately out of bounds for its own (smaller) sampled pitch.

    Unlike ball_episode_gen (whose input row stores the episode's own
    half-length/half-width FRACTION-of-base directly), player's input row
    stores fields 17/18 as the episode's pitch length/width as a ratio to
    the BASE pitch's full length/width (see ``_encode_input``'s fields
    17-18) -- halved here to match ``_draw_pitch``'s half-extent convention.
    Skipped when this episode's pitch is within 1cm of the base pitch (the
    overlay would just double the solid outline).
    """
    hl_ep = input_row[17] * gen_params.base_pitch_length_m / 2
    hw_ep = input_row[18] * gen_params.base_pitch_width_m / 2
    hl_base, hw_base = gen_params.base_pitch_length_m / 2, gen_params.base_pitch_width_m / 2
    if abs(hl_ep - hl_base) < 0.01 and abs(hw_ep - hw_base) < 0.01:
        return
    ax.add_patch(mpatches.Rectangle(
        (-hl_ep, -hw_ep), 2 * hl_ep, 2 * hw_ep,
        edgecolor="#ff5555", facecolor="none", linewidth=1.3, linestyle="--", zorder=3,
    ))
    ax.text(-hl_ep, hw_ep + 0.6, "this episode's own pitch boundary", color="#ff5555", fontsize=6.5, zorder=7)


def _fmt_xy(x: float, y: float) -> str:
    return f"({x:.1f}, {y:.1f})"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _fmt_flag(label_or_logit: float, is_logit: bool) -> str:
    return f"p={_sigmoid(label_or_logit):.2f}" if is_logit else f"{label_or_logit:.0f}"


def _arrow(ax, p0: tuple[float, float], p1: tuple[float, float], color: str) -> None:
    ax.annotate(
        "", xy=p1, xytext=p0,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.65, alpha=0.85),
        zorder=4,
    )


def _heading_tick(ax, point: tuple[float, float], sin_val: float, cos_val: float, color: str) -> None:
    """Short fixed-length tick mark from ``point`` in the heading direction
    ``(cos_val, sin_val)`` -- distinguishable from the (longer, arrowhead-
    tipped) trajectory arrows by being short, plain-lined, and drawn with a
    small dot at the point end rather than an arrowhead at the far end."""
    x0, y0 = point
    x1, y1 = x0 + _HEADING_TICK_LEN_M * cos_val, y0 + _HEADING_TICK_LEN_M * sin_val
    ax.plot([x0, x1], [y0, y1], "-", color=color, lw=0.8, alpha=0.9, zorder=5, solid_capstyle="round")


class Inspector:
    """``_prefetch_loop`` runs on a background daemon thread, pulling random
    rows through the model (encoder + decoder -- the only part worth
    overlapping with UI time, prediction being the slow step relative to a
    plain array index) and pushing the results onto a small bounded queue.
    The main/UI thread never predicts -- it just pops the next-ready item
    and renders it, so clicking "Next" only pays for a redraw, not a
    forward pass, PROVIDED the queue has had time to refill since the last
    click (``maxsize=2`` keeps one row buffered ahead of whatever's
    currently on screen without prefetching arbitrarily far ahead of what
    the user will actually look at).
    """

    def __init__(
        self, ds: PlayerDynamicsDataset, model: PlayerDynamicsAutoencoder, cfg: dict,
        gen_params: PlayerEpisodeGenParams, seed: int | None,
    ):
        self.ds = ds
        self.model = model
        self.horizons_s = list(cfg["horizons_s"])
        self.gen_params = gen_params
        self.rng = np.random.default_rng(seed)  # only ever touched by the prefetch thread -- single producer, no lock needed

        self._queue: queue.Queue[dict] = queue.Queue(maxsize=2)
        self._prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._prefetch_thread.start()

        self.fig, (self.ax_traj, self.ax_stamina) = plt.subplots(
            1, 2, figsize=(15, 8.5), facecolor=_C_BG, gridspec_kw={"width_ratios": [2, 1]},
        )
        self.fig.subplots_adjust(bottom=0.34, top=0.68, wspace=0.2)
        self.ax_table = self.fig.add_axes((0.04, 0.06, 0.92, 0.22))
        self.ax_table.set_facecolor(_C_BG)
        self.ax_table.axis("off")
        ax_btn = self.fig.add_axes((0.44, 0.01, 0.12, 0.04))
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
        pitch_length_m, pitch_width_m = _pitch_dims_m(input_row, gen_params)
        div_x, div_y, div_vel = _kinematics_denorm_scales(pitch_length_m, pitch_width_m, gen_params)
        with torch.no_grad():
            x = torch.from_numpy(input_row[None].astype(np.float32))
            latent = self.model.encoder(x)
            decoder_outs = [o[0].numpy() for o in self.model.decoder(latent)]
        return {
            "idx": idx, "input_row": input_row, "div_x": div_x, "div_y": div_y, "div_vel": div_vel,
            "decoder_outs": decoder_outs,
        }

    def next_row(self) -> None:
        self._render(self._queue.get())

    def _render(self, data: dict) -> None:
        idx = data["idx"]
        ds, gen_params = self.ds, self.gen_params
        input_row, div_x, div_y, div_vel = data["input_row"], data["div_x"], data["div_y"], data["div_vel"]
        decoder_outs = data["decoder_outs"]
        has_possession = bool(input_row[11] > 0.5)

        # ---- Trajectory (+ heading ticks) ----
        ax = self.ax_traj
        ax.clear()
        _draw_pitch(ax, gen_params)
        _draw_episode_boundary(ax, input_row, gen_params)

        start_xy = (input_row[0] * div_x, input_row[1] * div_y)
        start_heading = (float(input_row[4]), float(input_row[5]))  # (sin, cos)
        gt_pts = [start_xy]
        gt_headings = [start_heading]
        for h in range(len(self.horizons_s)):
            base = h * N_TARGET_FIELDS_PER_HORIZON
            block = ds.targets[idx, base:base + N_TARGET_FIELDS_PER_HORIZON]
            gt_pts.append((block[0] * div_x, block[1] * div_y))
            gt_headings.append((float(block[4]), float(block[5])))
        pred_pts = [start_xy]
        pred_headings = [start_heading]
        for h in range(len(self.horizons_s)):
            block = decoder_outs[h]
            pred_pts.append((block[0] * div_x, block[1] * div_y))
            pred_headings.append((float(block[4]), float(block[5])))

        # Euclidean (real metres) gap between gt/pred position at each
        # horizon -- printed above the plot since eyeballing arrow-tip
        # distance on a shared pitch is imprecise at short horizons where
        # the gap is a few centimetres.
        pos_errs_m = [
            math.hypot(gt_pts[h + 1][0] - pred_pts[h + 1][0], gt_pts[h + 1][1] - pred_pts[h + 1][1])
            for h in range(len(self.horizons_s))
        ]
        entries = [f"{t:g}s: {e:.2f}m" for t, e in zip(self.horizons_s, pos_errs_m)]
        err_lines = textwrap.wrap("   ".join(entries), width=38)
        title = "\n".join([f"Trajectory by horizon  (row {idx})", "pos error (m):", *err_lines])
        ax.set_title(title, color="white", fontsize=8.5, pad=8, linespacing=1.4)

        for pts, color, label in ((gt_pts, _C_GT, "ground truth"), (pred_pts, _C_PRED, "predicted")):
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "o", color=color, ms=4.5, alpha=0.75, zorder=5, mec="white", mew=0.4, label=label)
            for i in range(len(pts) - 1):
                _arrow(ax, pts[i], pts[i + 1], color)
        for pt, (sin_v, cos_v), color in [
            *((p, h, _C_GT) for p, h in zip(gt_pts, gt_headings)),
            *((p, h, _C_PRED) for p, h in zip(pred_pts, pred_headings)),
        ]:
            _heading_tick(ax, pt, sin_v, cos_v, color)
        ax.plot(*start_xy, "o", color="white", ms=4.5, alpha=0.3, zorder=6, mec="white", mew=0.4)
        for h, (gx, gy) in enumerate(gt_pts[1:]):
            ax.text(gx + 0.8, gy + 0.8, f"t={self.horizons_s[h]:g}s", color="#cccccc", fontsize=6, zorder=7)
        ax.legend(loc="upper left", fontsize=8, facecolor=_C_BG, edgecolor="none", labelcolor="white", framealpha=0.75)

        # ---- Stamina over horizon ----
        ax = self.ax_stamina
        ax.clear()
        ax.set_facecolor(_C_PITCH)
        ts = [0.0, *self.horizons_s]
        gt_stamina = [float(input_row[6])]
        pred_stamina = [float(input_row[6])]
        for h in range(len(self.horizons_s)):
            base = h * N_TARGET_FIELDS_PER_HORIZON
            gt_stamina.append(float(ds.targets[idx, base + 6]))
            pred_stamina.append(float(decoder_outs[h][6]))
        ax.plot(ts, gt_stamina, "-o", color=_C_GT, ms=4.5, lw=1.6, label="ground truth", mec="white", mew=0.4)
        ax.plot(ts, pred_stamina, "-o", color=_C_PRED, ms=4.5, lw=1.6, label="predicted", mec="white", mew=0.4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("horizon (s)", color="white", fontsize=8)
        ax.set_ylabel("stamina", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#666688")
        ax.grid(True, color="#ffffff22", lw=0.6)
        _style_dark_ax(ax, "Stamina by horizon: gt vs pred")
        ax.legend(loc="best", fontsize=8, facecolor=_C_BG, edgecolor="none", labelcolor="white", framealpha=0.75)

        # ---- Position/velocity/heading/stamina/flags table ----
        ax = self.ax_table
        ax.clear()
        ax.axis("off")
        ax.set_facecolor(_C_BG)

        vx0, vy0 = input_row[2] * div_vel, input_row[3] * div_vel
        hdg0_deg = math.degrees(math.atan2(input_row[4], input_row[5]))
        rows = [[
            "0.0s (start)", _fmt_xy(*start_xy), "—", _fmt_xy(vx0, vy0), "—",
            f"{hdg0_deg:.0f}°", "—", f"{input_row[6]:.2f}", "—", "—", "—", "—", "—",
        ]]
        for h, t in enumerate(self.horizons_s):
            base = h * N_TARGET_FIELDS_PER_HORIZON
            gt_block = ds.targets[idx, base:base + N_TARGET_FIELDS_PER_HORIZON]
            pred_block = decoder_outs[h]
            gt_hdg_deg = math.degrees(math.atan2(gt_block[4], gt_block[5]))
            pred_hdg_deg = math.degrees(math.atan2(pred_block[4], pred_block[5]))
            rows.append([
                f"{t:g}s",
                _fmt_xy(*gt_pts[h + 1]), _fmt_xy(*pred_pts[h + 1]),
                _fmt_xy(gt_block[2] * div_vel, gt_block[3] * div_vel),
                _fmt_xy(pred_block[2] * div_vel, pred_block[3] * div_vel),
                f"{gt_hdg_deg:.0f}°", f"{pred_hdg_deg:.0f}°",
                f"{gt_block[6]:.2f}", f"{pred_block[6]:.2f}",
                _fmt_flag(gt_block[7], is_logit=False), _fmt_flag(pred_block[7], is_logit=True),
                _fmt_flag(gt_block[8], is_logit=False), _fmt_flag(pred_block[8], is_logit=True),
            ])
        col_labels = [
            "horizon", "gt pos (m)", "pred pos (m)", "gt vel (m/s)", "pred vel (m/s)",
            "gt hdg", "pred hdg", "gt stam", "pred stam", "gt oob", "pred oob", "gt goal", "pred goal",
        ]
        table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.35)
        for (r, _c), cell in table.get_celld().items():
            cell.set_edgecolor("#444466")
            cell.set_facecolor("#2a2a4a" if r == 0 else _C_BG)
            cell.get_text().set_color("white")
            if r == 0:
                cell.get_text().set_fontweight("bold")

        possession_note = (
            "" if has_possession else
            "   (has_possession=0 this row -- goal_scored is definitionally meaningless/always-0 here, per training's possession masking)"
        )
        self.fig.suptitle(
            f"Player-dynamics inspector  —  dataset row {idx}  ({len(self.ds):,} rows total){possession_note}",
            color="white", fontsize=10.5,
        )
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path, help="Full-model checkpoint (has model_state_dict), not the encoder-only final artifact.")
    ap.add_argument("--dataset", required=True, type=Path, help="Directory of .npz shards (e.g. physics_pretrain_data/player/).")
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
    )
    # strict=False: tolerates a checkpoint saved before a param existed at
    # all -- see train_player_dynamics.py's own --init-checkpoint resume
    # path for the identical pattern (no crossing-head-style migration
    # shim needed here, player has no such head).
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"Note: checkpoint missing {len(missing)} param(s), left at fresh init: {missing}")
    if unexpected:
        print(f"Note: checkpoint had {len(unexpected)} unexpected param(s), ignored: {unexpected}")
    model.eval()

    ds = PlayerDynamicsDataset.from_directory(args.dataset)
    gen_params = PlayerEpisodeGenParams.from_config()
    normalize_by_base = bool(cfg.get("normalize_kinematics_by_base_pitch", gen_params.normalize_kinematics_by_base_pitch))
    if normalize_by_base != gen_params.normalize_kinematics_by_base_pitch:
        # The checkpoint's own config snapshot disagrees with the CURRENT
        # live ai_config.json on this dataset-generation-time flag -- defer
        # to the checkpoint's snapshot (what the dataset it was trained
        # against actually used), matching inspect_ball_pretrain.py's
        # identical override.
        gen_params = dataclasses.replace(gen_params, normalize_kinematics_by_base_pitch=normalize_by_base)

    inspector = Inspector(ds, model, cfg, gen_params, args.seed)
    inspector.show()


if __name__ == "__main__":
    main()

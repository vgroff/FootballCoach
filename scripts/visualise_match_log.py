#!/usr/bin/env python3
"""Visualise a match log JSON produced by MatchLogger.

Usage:
    uv run python scripts/visualise_match_log.py episode.json
    uv run python scripts/visualise_match_log.py episode.json -o out.png [--show]
    uv run python scripts/visualise_match_log.py --dir match_logs/
    uv run python scripts/visualise_match_log.py episode.json --events kick goal tackle_attempt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------------
# Standard pitch constants (metres, origin at pitch centre)
# x = goal-to-goal axis, y = width axis
# ---------------------------------------------------------------------------
_HL = 52.5    # half-length
_HW = 34.0    # half-width
_GW = 7.32    # goal width
_GD = 2.45    # goal depth (past the goal line)
_BL = 16.5    # penalty-box length
_BW = 40.32   # penalty-box width
_SL = 5.5     # 6-yard-box length
_SW = 18.32   # 6-yard-box width
_PS = 11.0    # penalty-spot distance from goal line
_CR = 9.15    # centre circle / penalty arc radius

# Palette
_C_PITCH = "#227832"
_C_LINE  = "white"
_C_LEFT  = "#e84040"   # left-team possession
_C_RIGHT = "#4090e8"   # right-team possession
_C_LOOSE = "#c0c0c0"   # loose ball
_C_BG    = "#1a1a2e"   # figure background

# Event marker style: (matplotlib marker, face-colour, size)
_EV: dict[str, tuple[str, str, float]] = {
    "start":             ("o",  "white",   9),
    "kick":              ("^",  "#f5a623", 9),
    "tackle_attempt":    ("X",  "#ff5555", 9),
    "possession_change": ("D",  "#88ff44", 7),
    "goal":              ("*",  "#ffe000", 15),
    "ball_out":          ("s",  "#cc88ff", 8),
    "consistency":       (".",  "#888888", 5),
    "episode_end":       ("P",  "white",   10),
}


# ---------------------------------------------------------------------------
# Pitch geometry
# ---------------------------------------------------------------------------

def _draw_pitch(ax) -> None:
    """Draw standard pitch geometry on *ax* in world coordinates."""
    lkw = dict(edgecolor=_C_LINE, facecolor="none", linewidth=1.5)
    ax.set_facecolor(_C_PITCH)

    ax.add_patch(mpatches.Rectangle((-_HL, -_HW), 2*_HL, 2*_HW, **lkw))
    ax.plot([0, 0], [-_HW, _HW], color=_C_LINE, lw=1.5)
    ax.add_patch(mpatches.Circle((0, 0), _CR, **lkw))
    ax.plot(0, 0, "o", color=_C_LINE, ms=3, zorder=3)

    # Angle (deg) at which the penalty arc meets the box inner edge
    _arc_half = math.degrees(math.acos((_BL - _PS) / _CR))   # ≈ 53.1°

    for sx in (-1, 1):
        gl = sx * _HL   # goal-line x

        # Penalty box
        bx0 = min(gl, gl - sx * _BL)
        ax.add_patch(mpatches.Rectangle((bx0, -_BW / 2), _BL, _BW, **lkw))

        # 6-yard box
        sx0 = min(gl, gl - sx * _SL)
        ax.add_patch(mpatches.Rectangle((sx0, -_SW / 2), _SL, _SW, **lkw))

        # Penalty spot
        px = gl - sx * _PS
        ax.plot(px, 0, "o", color=_C_LINE, ms=3, zorder=3)

        # Penalty arc (only the part that lies outside the penalty box)
        t1, t2 = (
            (180 - _arc_half, 180 + _arc_half) if sx == 1
            else (-_arc_half, _arc_half)
        )
        ax.add_patch(mpatches.Arc(
            (px, 0), 2 * _CR, 2 * _CR, angle=0,
            theta1=t1, theta2=t2, color=_C_LINE, lw=1.5,
        ))

        # Goal net (semi-transparent rectangle extending past goal line)
        gx0 = min(gl, gl + sx * _GD)
        ax.add_patch(mpatches.Rectangle(
            (gx0, -_GW / 2), _GD, _GW,
            edgecolor=_C_LINE, facecolor="#ffffff1a", linewidth=2.0,
        ))

    ax.set_xlim(-_HL - _GD - 0.5, _HL + _GD + 0.5)
    ax.set_ylim(-_HW - 2, _HW + 2)
    # datalim: axes box fills the grid cell; y range expands slightly rather than leaving x whitespace
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _ball_colour(pid: Optional[str], team_map: dict[str, str]) -> str:
    team = team_map.get(pid) if pid else None
    if team == "left":
        return _C_LEFT
    if team == "right":
        return _C_RIGHT
    return _C_LOOSE


def _style_dark_ax(ax) -> None:
    ax.set_facecolor("#0d0d1a")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")
    ax.tick_params(colors="white", labelsize=7)


# ---------------------------------------------------------------------------
# Single-log visualisation
# ---------------------------------------------------------------------------

def visualise_single(
    log_path: Path,
    output: Optional[Path],
    show: bool,
    event_filter: Optional[set[str]],
) -> None:
    events = _load(log_path)
    if not events:
        print("Empty log.", file=sys.stderr)
        return

    start_ev = next((e for e in events if e["event"] == "start"), None)
    team_map: dict[str, str] = {}
    if start_ev and "player_positions" in start_ev:
        team_map = {
            pid: info["team"]
            for pid, info in start_ev["player_positions"].items()
        }

    # ---- Ball trail ----
    ball_xy:  list[tuple[float, float]] = []
    ball_z:   list[float] = []
    ball_t:   list[float] = []
    seg_col:  list[str]   = []   # possession colour for each outgoing segment
    possessor: Optional[str] = None

    for ev in events:
        bx, by, bz = ev["ball_pos"]
        ball_xy.append((float(bx), float(by)))
        ball_z.append(float(bz))
        ball_t.append(float(ev["time_s"]))
        etype = ev["event"]
        if etype in ("possession_change", "consistency"):
            possessor = ev.get("possessor_id")
        seg_col.append(_ball_colour(possessor, team_map))

    # ---- Player tracks (sparse, event-driven) ----
    player_tracks: dict[str, list[tuple[float, float, float]]] = {}
    if start_ev and "player_positions" in start_ev:
        for pid, info in start_ev["player_positions"].items():
            px, py, _ = info["pos"]
            player_tracks.setdefault(pid, []).append(
                (float(start_ev["time_s"]), float(px), float(py))
            )
    for ev in events:
        pid = ev.get("player_id")
        pp  = ev.get("player_pos")
        if pid and pp:
            px, py, _ = pp
            player_tracks.setdefault(pid, []).append(
                (float(ev["time_s"]), float(px), float(py))
            )

    # ---- Reward data ----
    rew_t: list[float] = []
    rew_cumul: list[dict[str, float]] = []
    for ev in events:
        rc = ev.get("reward_cumulative")
        if rc:
            rew_t.append(float(ev["time_s"]))
            rew_cumul.append(rc)

    duration = float(events[-1]["time_s"])
    outcome  = next(
        (e.get("outcome", "") for e in reversed(events) if e["event"] == "episode_end"), ""
    )
    final_cumul: dict[str, float] = rew_cumul[-1] if rew_cumul else {}
    total_r = sum(final_cumul.values()) if final_cumul else float("nan")

    # ---- Layout ----
    fig = plt.figure(figsize=(17, 11), facecolor=_C_BG, layout="constrained")
    fig.get_layout_engine().set(hspace=0.02, wspace=0.02, h_pad=0.02, w_pad=0.04)
    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[7, 1, 2.5],
        width_ratios=[3.5, 1],
    )
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0, 1], height_ratios=[5, 2], hspace=0.12,
    )
    ax_pitch    = fig.add_subplot(gs[0, 0])
    ax_legend   = fig.add_subplot(gs_right[0])
    ax_bar      = fig.add_subplot(gs_right[1])
    ax_timeline = fig.add_subplot(gs[1, :])
    ax_reward   = fig.add_subplot(gs[2, :])

    for ax in (ax_legend, ax_bar, ax_timeline, ax_reward):
        _style_dark_ax(ax)
    ax_legend.axis("off")
    ax_bar.axis("off")

    # ==================================================================
    # PITCH
    # ==================================================================
    _draw_pitch(ax_pitch)

    # Ball trail (coloured by possession)
    if len(ball_xy) >= 2:
        segs = [[ball_xy[i], ball_xy[i + 1]] for i in range(len(ball_xy) - 1)]
        lc = LineCollection(segs, colors=seg_col[:-1],
                            linewidths=1.6, alpha=0.85, zorder=4)
        ax_pitch.add_collection(lc)

        # Direction arrowheads every 5th long segment (keep clutter down)
        for i in range(0, len(ball_xy) - 1, 5):
            x0, y0 = ball_xy[i]
            x1, y1 = ball_xy[i + 1]
            d = math.hypot(x1 - x0, y1 - y0)
            if d > 3.0:
                mx = x0 + 0.70 * (x1 - x0)
                my = y0 + 0.70 * (y1 - y0)
                ux, uy = (x1 - x0) / d, (y1 - y0) / d
                ax_pitch.annotate(
                    "",
                    xy=(mx + ux * 0.5, my + uy * 0.5),
                    xytext=(mx, my),
                    arrowprops=dict(arrowstyle="->", color=seg_col[i], lw=1.0),
                    zorder=5,
                )

    # Ball dots; larger = higher (airborne)
    bxarr = np.array([p[0] for p in ball_xy])
    byarr = np.array([p[1] for p in ball_xy])
    bzarr = np.array(ball_z)
    ax_pitch.scatter(bxarr, byarr, s=18 + bzarr * 18,
                     c="white", zorder=6, alpha=0.75, linewidths=0)

    # Height labels for notably airborne ball (z > 0.5 m)
    for (bx, by), bz in zip(ball_xy, ball_z):
        if bz > 0.5:
            ax_pitch.text(bx + 0.6, by + 0.6, f"\u2191{bz:.1f}m",
                          color="white", fontsize=4.5, zorder=7, alpha=0.9)

    # Initial ball velocity arrow
    if start_ev and "ball_vel" in start_ev:
        vx, vy, _ = start_ev["ball_vel"]
        spd = math.hypot(float(vx), float(vy))
        if spd > 0.1:
            scale = min(8.0, spd * 0.5)
            bx0, by0 = float(start_ev["ball_pos"][0]), float(start_ev["ball_pos"][1])
            ax_pitch.annotate(
                "",
                xy=(bx0 + vx / spd * scale, by0 + vy / spd * scale),
                xytext=(bx0, by0),
                arrowprops=dict(arrowstyle="-|>", color="#ffe000", lw=2.2),
                zorder=8,
            )

    # Player starting positions
    if start_ev and "player_positions" in start_ev:
        for pid, info in start_ev["player_positions"].items():
            px, py, _ = info["pos"]
            col = _C_LEFT if info["team"] == "left" else _C_RIGHT
            ax_pitch.plot(float(px), float(py), "o", color=col, ms=11,
                          zorder=7, mec="white", mew=0.8)
            ax_pitch.text(float(px), float(py) + 1.5, pid,
                          color="white", fontsize=4.5,
                          ha="center", va="bottom", zorder=8)

    # Player movement tracks (dashed lines between known positions)
    for pid, track in player_tracks.items():
        if len(track) < 2:
            continue
        track.sort(key=lambda t: t[0])
        xs = [t[1] for t in track]
        ys = [t[2] for t in track]
        col = _C_LEFT if team_map.get(pid) == "left" else _C_RIGHT
        ax_pitch.plot(xs, ys, "--", color=col, lw=0.9, alpha=0.4, zorder=5)

    # Numbered event markers
    ev_rows: list[str] = []
    n = 1
    for ev in events:
        etype = ev["event"]
        if etype == "start":
            continue
        if event_filter and etype not in event_filter:
            continue
        mk, col, ms = _EV.get(etype, ("o", "white", 7))
        bx, by = float(ev["ball_pos"][0]), float(ev["ball_pos"][1])
        ax_pitch.plot(bx, by, mk, color=col, ms=ms, zorder=9,
                      mec="white", mew=0.4)
        ax_pitch.text(bx + 0.9, by + 0.9, str(n),
                      color="white", fontsize=4.5, fontweight="bold", zorder=10)
        pid_str = ev.get("player_id", "")
        ev_rows.append(f"[{n:>2}] {ev['time_s']:5.1f}s  {etype:<20}  {pid_str}")
        n += 1

    # ==================================================================
    # LEGEND (marker types + event log)
    # ==================================================================
    handles = []
    for etype, (mk, col, ms) in _EV.items():
        if etype == "start":
            continue
        handles.append(mlines.Line2D(
            [], [], marker=mk, linestyle="none",
            markerfacecolor=col, markeredgecolor="white",
            markeredgewidth=0.4, markersize=6, label=etype,
        ))
    handles += [
        mlines.Line2D([], [], color=_C_LEFT,  lw=2, label="left possession (trail)"),
        mlines.Line2D([], [], color=_C_RIGHT, lw=2, label="right possession (trail)"),
        mlines.Line2D([], [], color=_C_LOOSE, lw=2, label="loose ball (trail)"),
    ]
    ax_legend.legend(
        handles=handles, loc="upper left", fontsize=9,
        facecolor=_C_BG, edgecolor="none", labelcolor="white",
        framealpha=0.8, handlelength=1.4,
    )

    max_rows = 22
    ev_text = "\n".join(ev_rows[:max_rows])
    if len(ev_rows) > max_rows:
        ev_text += f"\n  ... +{len(ev_rows) - max_rows} more"
    ax_legend.text(
        0.02, 0.44, ev_text, transform=ax_legend.transAxes,
        color="#cccccc", fontsize=7.5, va="top", family="monospace",
    )

    # ==================================================================
    # REWARD BREAKDOWN BAR
    # ==================================================================
    if final_cumul:
        ax_bar.axis("on")
        items = sorted(final_cumul.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        keys = [k for k, _ in items]
        vals = [v for _, v in items]
        bar_colors = [_C_RIGHT if v >= 0 else "#ff5555" for v in vals]
        ax_bar.barh(range(len(keys)), vals, color=bar_colors, alpha=0.85)
        ax_bar.set_yticks(range(len(keys)))
        ax_bar.set_yticklabels(keys, fontsize=7.5, color="white")
        ax_bar.tick_params(axis="x", labelsize=7.0, colors="white")
        ax_bar.axvline(0, color="#666666", lw=0.8)
        ax_bar.set_title("Episode reward totals", color="white", fontsize=8.5, pad=2)
        ax_bar.set_facecolor("#0d0d1a")
        for sp in ax_bar.spines.values():
            sp.set_edgecolor("#444444")

    # ==================================================================
    # TIMELINE
    # ==================================================================
    ax_timeline.set_xlim(0, max(duration, 1.0))
    ax_timeline.set_ylim(-0.6, 0.6)
    ax_timeline.axhline(0, color="#555555", lw=1.5)
    ax_timeline.set_yticks([])
    ax_timeline.set_xlabel("time (s)", color="white", fontsize=7)
    ax_timeline.set_title("Event timeline", color="white", fontsize=7, pad=2)
    for ev in events:
        etype = ev["event"]
        if event_filter and etype not in event_filter and etype not in ("start", "episode_end"):
            continue
        mk, col, ms = _EV.get(etype, ("o", "white", 7))
        ax_timeline.plot(float(ev["time_s"]), 0, mk, color=col,
                         ms=ms * 0.65, zorder=5, mec="white", mew=0.3)

    # ==================================================================
    # CUMULATIVE REWARD
    # ==================================================================
    ax_reward.set_xlabel("time (s)", color="white", fontsize=7)
    ax_reward.set_ylabel("cumulative reward", color="white", fontsize=7)
    ax_reward.set_title("Reward accumulation over episode", color="white", fontsize=7, pad=2)
    if rew_t and rew_cumul:
        all_keys = sorted({k for rc in rew_cumul for k in rc})
        cmap = plt.cm.tab20
        for i, key in enumerate(all_keys):
            ys = [rc.get(key, 0.0) for rc in rew_cumul]
            ax_reward.plot(rew_t, ys, lw=1.1, alpha=0.85, label=key,
                           color=cmap(i / max(len(all_keys), 1)))
        totals = [sum(rc.values()) for rc in rew_cumul]
        ax_reward.plot(rew_t, totals, lw=2.0, color="white", label="TOTAL", zorder=5)
        ax_reward.axhline(0, color="#555555", lw=0.8)
        ax_reward.legend(
            loc="upper left", fontsize=5.5, facecolor=_C_BG,
            edgecolor="none", labelcolor="white", framealpha=0.7, ncol=4,
        )

    # ==================================================================
    # Figure title
    # ==================================================================
    total_r_str = f"{total_r:+.3f}" if not math.isnan(total_r) else "n/a"
    fig.suptitle(
        f"{log_path.name}  |  outcome: {outcome or 'n/a'}  "
        f"|  duration: {duration:.1f}s  |  total reward: {total_r_str}",
        color="white", fontsize=9,
    )

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved to {output}")
    if show or not output:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Directory heatmap
# ---------------------------------------------------------------------------

def visualise_dir(log_dir: Path, output: Optional[Path], show: bool) -> None:
    logs = sorted(log_dir.glob("*.json"))
    if not logs:
        sys.exit(f"No JSON files found in {log_dir}")

    all_xy: list[tuple[float, float]] = []
    outcome_counts: dict[str, int] = {}
    n = 0

    for p in logs:
        try:
            evs = _load(p)
        except Exception as exc:
            print(f"Skipping {p.name}: {exc}", file=sys.stderr)
            continue
        n += 1
        for ev in evs:
            bx, by, _ = ev["ball_pos"]
            all_xy.append((float(bx), float(by)))
        for ev in reversed(evs):
            if ev["event"] == "episode_end":
                oc = ev.get("outcome", "unknown")
                outcome_counts[oc] = outcome_counts.get(oc, 0) + 1
                break

    if not all_xy:
        sys.exit("No ball positions found.")

    fig, ax = plt.subplots(figsize=(13, 9), facecolor=_C_BG)
    fig.patch.set_facecolor(_C_BG)
    _draw_pitch(ax)

    xs = np.array([p[0] for p in all_xy])
    ys = np.array([p[1] for p in all_xy])
    h, _, _ = np.histogram2d(
        xs, ys,
        bins=[105, 68],
        range=[[-_HL, _HL], [-_HW, _HW]],
    )
    ax.imshow(
        h.T,
        extent=[-_HL, _HL, -_HW, _HW],
        origin="lower",
        cmap="hot",
        alpha=0.55,
        aspect="auto",
        interpolation="gaussian",
        zorder=3,
    )

    oc_str = "  ".join(f"{k}: {v}" for k, v in sorted(outcome_counts.items()))
    fig.suptitle(
        f"Ball position heatmap  —  {n} episodes from {log_dir.name}/\n"
        f"Outcomes: {oc_str or 'n/a'}",
        color="white", fontsize=10,
    )

    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved to {output}")
    if show or not output:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualise a MatchLogger JSON log file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python scripts/visualise_match_log.py episode.json\n"
            "  uv run python scripts/visualise_match_log.py episode.json -o out.png\n"
            "  uv run python scripts/visualise_match_log.py --dir match_logs/\n"
            "  uv run python scripts/visualise_match_log.py episode.json "
            "--events kick goal tackle_attempt\n"
        ),
    )
    ap.add_argument("log", nargs="?", type=Path,
                    help="Match log JSON file to visualise")
    ap.add_argument("--dir", type=Path, metavar="DIR",
                    help="Directory of logs: produce a ball-position heatmap instead")
    ap.add_argument("--output", "-o", type=Path, metavar="FILE",
                    help="Save figure to this file (PNG / SVG / PDF)")
    ap.add_argument("--show", action="store_true",
                    help="Open interactive window even when --output is set")
    ap.add_argument(
        "--events", nargs="+", metavar="TYPE",
        choices=list(_EV.keys()),
        help="Show only these event types on the pitch (timeline always shows all)",
    )
    args = ap.parse_args()

    if args.dir:
        visualise_dir(args.dir, args.output, args.show)
    elif args.log:
        visualise_single(
            args.log, args.output, args.show,
            set(args.events) if args.events else None,
        )
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

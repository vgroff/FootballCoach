"""Repulsion steering sandbox — run with: uv run python scripts/repulsion_sandbox.py

Simulates a few hand-crafted scenarios tick-by-tick and logs what the
repulsion steering does at every step.  Tweak PARAMS and SCENARIOS at the
top to explore different settings.
"""
from __future__ import annotations

import math
import sys

sys.path.insert(0, "src")

from footballcoach.entities.attributes import PlayerAttributes
from footballcoach.entities.player import Player, PlayerState, Team
from footballcoach.mathutils.vector3 import Vector3
from footballcoach.steering import RepulsionParams, compute_repulsion

# ─────────────────────────────────────────────────────────────────────────────
# TWEAK THESE
# ─────────────────────────────────────────────────────────────────────────────

PARAMS = RepulsionParams(
    radius_m=4.0,
    strength_base=3.5,
    ball_carrier_repulsion_mult=1.8,
    ball_carrier_speed_penalty_max=0.4,
    speed_penalty_scale=0.0847,
    alignment_dot_threshold=-0.7,
    min_orthogonal_adjust_mps=1.5,
)

DT = 0.1          # seconds per tick
TOP_SPEED = 8.0   # m/s — used to scale desired direction into velocity
N_TICKS = 20      # ticks to simulate per scenario

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

ATTRS = PlayerAttributes.average(0.7)


def make_player(pid: str, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> Player:
    p = Player(
        player_id=pid,
        team=Team.LEFT,
        attributes=ATTRS,
        position=Vector3(x, y, 0.0),
        velocity=Vector3(vx, vy, 0.0),
    )
    return p


def norm2(x: float, y: float) -> tuple[float, float]:
    """Normalise (x,y); returns (0,0) if near-zero."""
    length = math.hypot(x, y)
    if length < 1e-9:
        return 0.0, 0.0
    return x / length, y / length


def fmt_vec(x: float, y: float) -> str:
    return f"({x:+.3f}, {y:+.3f})"


def simulate(
    scenario_name: str,
    players: list[Player],
    desired_dirs: dict[str, tuple[float, float]],  # pid → (dx, dy) desired dir
    ball_carrier_id: str | None,
) -> None:
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name}")
    if ball_carrier_id:
        print(f"  ball carrier: {ball_carrier_id}")
    print(f"  params: radius={PARAMS.radius_m}m  strength={PARAMS.strength_base}"
          f"  bc_mult={PARAMS.ball_carrier_repulsion_mult}"
          f"  speed_pen_max={PARAMS.ball_carrier_speed_penalty_max}"
          f"  speed_pen_scale={PARAMS.speed_penalty_scale}")
    print(f"{'='*70}")

    # Mutable copy of positions/velocities as plain floats (avoid dataclass mutation)
    state: dict[str, list[float]] = {
        p.player_id: [p.position.x, p.position.y, p.velocity.x, p.velocity.y]
        for p in players
    }

    for tick in range(1, N_TICKS + 1):
        # Rebuild Player objects from current state
        live_players = [
            make_player(p.player_id, state[p.player_id][0], state[p.player_id][1],
                        state[p.player_id][2], state[p.player_id][3])
            for p in players
        ]

        print(f"\n  Tick {tick:02d}")

        for player in live_players:
            pid = player.player_id
            raw_dx, raw_dy = desired_dirs.get(pid, (0.0, 0.0))
            desired_dir = Vector3(raw_dx, raw_dy, 0.0)

            adj_dir, speed_mult = compute_repulsion(
                player=player,
                desired_dir=desired_dir,
                other_players=live_players,
                ball_carrier_id=ball_carrier_id,
                params=PARAMS,
            )

            # Effective velocity: speed_mult * TOP_SPEED * adj_dir_normalised
            adj_len = math.hypot(adj_dir.x, adj_dir.y)
            if adj_len > 1e-9:
                move_x = (adj_dir.x / adj_len) * TOP_SPEED * speed_mult
                move_y = (adj_dir.y / adj_len) * TOP_SPEED * speed_mult
            else:
                move_x, move_y = 0.0, 0.0

            # Compute repulsion vector just for logging (re-derive it as delta
            # between adjusted-dir and desired-dir)
            raw_len = math.hypot(raw_dx, raw_dy)
            if raw_len > 1e-9:
                raw_norm_x, raw_norm_y = raw_dx / raw_len, raw_dy / raw_len
            else:
                raw_norm_x, raw_norm_y = 0.0, 0.0

            if adj_len > 1e-9:
                adj_norm_x, adj_norm_y = adj_dir.x / adj_len, adj_dir.y / adj_len
            else:
                adj_norm_x, adj_norm_y = 0.0, 0.0

            dir_delta_x = adj_norm_x - raw_norm_x
            dir_delta_y = adj_norm_y - raw_norm_y
            angle_change_deg = math.degrees(
                math.atan2(raw_norm_x * adj_norm_y - raw_norm_y * adj_norm_x,
                           raw_norm_x * adj_norm_x + raw_norm_y * adj_norm_y)
            )

            # Nearest neighbour distance for context
            nearest_d = float("inf")
            for other in live_players:
                if other.player_id == pid:
                    continue
                d = math.hypot(player.position.x - other.position.x,
                               player.position.y - other.position.y)
                if d < nearest_d:
                    nearest_d = d

            is_bc = (pid == ball_carrier_id)
            tag = "[BC]" if is_bc else "    "

            print(
                f"    {tag} {pid:<8s}  pos={fmt_vec(player.position.x, player.position.y)}"
                f"  nearest={nearest_d:.2f}m"
                f"  raw_dir={fmt_vec(raw_norm_x, raw_norm_y)}"
                f"  adj_dir={fmt_vec(adj_norm_x, adj_norm_y)}"
                f"  Δangle={angle_change_deg:+.1f}°"
                f"  speed_mult={speed_mult:.3f}"
                f"  move={fmt_vec(move_x, move_y)}"
            )

            # Advance state: simple Euler integration
            new_x = player.position.x + move_x * DT
            new_y = player.position.y + move_y * DT
            state[pid] = [new_x, new_y, move_x, move_y]

    print()


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

# 1. Two non-carriers heading toward each other
simulate(
    scenario_name="1 — Two non-carriers on a head-on collision course",
    players=[
        make_player("A", x=0.0, y=0.0, vx=5.0, vy=0.0),
        make_player("B", x=6.0, y=0.0, vx=-5.0, vy=0.0),
    ],
    desired_dirs={
        "A": (1.0, 0.0),   # A wants to move right
        "B": (-1.0, 0.0),  # B wants to move left
    },
    ball_carrier_id=None,
)

# 2. Ball carrier moving toward a defender
simulate(
    scenario_name="2 — Ball carrier dribbling toward defender",
    players=[
        make_player("Carrier", x=0.0, y=0.0, vx=6.0, vy=0.0),
        make_player("Defender", x=5.0, y=0.5, vx=0.0, vy=0.0),
    ],
    desired_dirs={
        "Carrier":  (1.0, 0.0),
        "Defender": (0.0, 0.0),   # standing still
    },
    ball_carrier_id="Carrier",
)

# 3. Player approaching from the side — slight offset, should nudge orthogonally
simulate(
    scenario_name="3 — Orthogonal nudge: player nearly head-on, slight y-offset",
    players=[
        make_player("A", x=0.0, y=0.1, vx=5.0, vy=0.0),
        make_player("B", x=3.5, y=0.0, vx=0.0, vy=0.0),  # obstacle slightly below
    ],
    desired_dirs={
        "A": (1.0, 0.0),
        "B": (0.0, 0.0),
    },
    ball_carrier_id=None,
)

# 4. Cluster: one player surrounded by three opponents — expected strong multi-source repulsion
simulate(
    scenario_name="4 — Cluster: carrier in middle of three defenders",
    players=[
        make_player("Carrier", x=5.0, y=5.0, vx=4.0, vy=0.0),
        make_player("Def1",    x=6.5, y=5.0, vx=0.0, vy=0.0),
        make_player("Def2",    x=5.0, y=6.5, vx=0.0, vy=0.0),
        make_player("Def3",    x=5.0, y=3.5, vx=0.0, vy=0.0),
    ],
    desired_dirs={
        "Carrier": (1.0, 0.0),
        "Def1":    (-1.0, 0.0),
        "Def2":    (0.0, -1.0),
        "Def3":    (0.0, 1.0),
    },
    ball_carrier_id="Carrier",
)

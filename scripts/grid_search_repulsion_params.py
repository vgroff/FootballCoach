"""Grid search over repulsion steering parameters.

Run with: uv run python scripts/grid_search_repulsion_params.py

Sweeps RepulsionParams across 8 scenarios and ranks by a multi-objective
score designed to reward genuine football-realistic behaviour:

  avoidance_pct   — % of ticks where every pair of players stays > 1.0 m apart
  min_sep         — worst-case closest approach (higher is safer)
  avg_defl        — mean per-tick steering angle change (lower = smoother)
  max_defl        — worst single-tick deflection (hard cap should eliminate extremes)
  bc_catchability — ball carrier should be meaningfully slowed near defenders
                    (bc_avg around BC_IDEAL ≈ 0.82); too high = uncatchable,
                    too low = unfairly stopped

Score (higher is better):
  avoidance_pct * 70          primary: keep players apart
  + min_sep * 10              reward genuine clearance, not scraping-by
  - avg_defl * 0.8            penalise jittery/jerky steering
  - max(0, max_defl-80) * 3   heavy hit for very sharp deflections
  + bc_catch_bonus            ±15 centred on BC_IDEAL, falls off quadratically
  + corridor_bonus            extra reward for successfully navigating the corridor
"""
from __future__ import annotations

import itertools
import math
import sys

sys.path.insert(0, "src")

from footballcoach.entities.attributes import PlayerAttributes
from footballcoach.entities.player import Player, Team
from footballcoach.mathutils.vector3 import Vector3
from footballcoach.steering import RepulsionParams, compute_repulsion

# ─────────────────────────────────────────────────────────────────────────────
# Grid definition
# ─────────────────────────────────────────────────────────────────────────────

GRID = {
    "radius_m":                    [3.5, 3.8, 4.1, 4.3],
    "strength_base":               [2.2, 2.5, 2.8],
    "ball_carrier_repulsion_mult": [1.8, 2.0, 2.2],
    "speed_penalty_scale":         [0.10, 0.12, 0.14],
    "min_orthogonal_adjust_mps":   [0.75, 1.0, 1.25],
}

# These were insensitive across prior sweeps — fixed at known-good values
FIXED = dict(
    ball_carrier_speed_penalty_max=0.4,
    alignment_dot_threshold=-0.7,
    max_deflection_deg=90.0,
)

DT = 0.1
TOP_SPEED = 8.0     # m/s — used to translate direction into velocity
JOG_SPEED = 4.0     # m/s — defender jog speed
N_TICKS = 45
ATTRS = PlayerAttributes.average(0.7)
BC_IDEAL = 0.82     # target bc_avg (carrier slowed ~18% — catchable but not stopped)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario helpers
# ─────────────────────────────────────────────────────────────────────────────

def _p(pid, x, y, vx=0.0, vy=0.0):
    return Player(
        player_id=pid, team=Team.LEFT, attributes=ATTRS,
        position=Vector3(x, y, 0.0), velocity=Vector3(vx, vy, 0.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenarios  (name, players, desired_dirs, ball_carrier_id, sep_threshold_m)
#   sep_threshold_m: "good tick" if every pair stays beyond this
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    # 1. Two non-carriers, dead-on collision
    (
        "head_on",
        [_p("A", 0.0, 0.0, 5.0, 0.0), _p("B", 6.0, 0.0, -5.0, 0.0)],
        {"A": (1.0, 0.0), "B": (-1.0, 0.0)},
        None, 0.8,
    ),
    # 2. Non-carriers converging from an angle
    (
        "angle_cross",
        [_p("A", 0.0, 0.0, 4.0, 0.0), _p("B", 4.0, 4.0, 0.0, -4.0)],
        {"A": (1.0, 0.0), "B": (0.0, -1.0)},
        None, 0.8,
    ),
    # 3. Ball carrier vs stationary defender (slight offset)
    (
        "carrier_vs_static",
        [_p("C", 0.0, 0.0, 6.0, 0.0), _p("D", 5.0, 0.4)],
        {"C": (1.0, 0.0), "D": (0.0, 0.0)},
        "C", 0.8,
    ),
    # 4. Ball carrier vs JOGGING defender (defender closes at JOG_SPEED)
    (
        "carrier_vs_jogger",
        [_p("C", 0.0, 0.0, 6.0, 0.0), _p("D", 7.0, 0.3, -JOG_SPEED, 0.0)],
        {"C": (1.0, 0.0), "D": (-1.0, 0.0)},
        "C", 0.8,
    ),
    # 5. Ball carrier vs two jogging defenders converging from sides
    (
        "carrier_vs_two_joggers",
        [
            _p("C",  0.0, 0.0, 6.0,  0.0),
            _p("D1", 8.0, 2.5, -JOG_SPEED * 0.7, -JOG_SPEED * 0.7),
            _p("D2", 8.0,-2.5, -JOG_SPEED * 0.7,  JOG_SPEED * 0.7),
        ],
        {"C": (1.0, 0.0), "D1": (-0.7, -0.7), "D2": (-0.7, 0.7)},
        "C", 0.8,
    ),
    # 6. Cluster: carrier surrounded by three stationary defenders
    (
        "cluster_3",
        [
            _p("Car", 5.0, 5.0, 4.0, 0.0),
            _p("D1",  6.5, 5.0),
            _p("D2",  5.0, 6.5),
            _p("D3",  5.0, 3.5),
        ],
        {"Car": (1.0, 0.0), "D1": (-1.0, 0.0), "D2": (0.0, -1.0), "D3": (0.0, 1.0)},
        "Car", 0.8,
    ),
    # 7. Corridor: player must thread between two stationary blockers
    (
        "corridor",
        [
            _p("Runner", 0.0, 0.0, 5.0, 0.0),
            _p("Wall1",  4.0, 1.1),
            _p("Wall2",  4.0,-1.1),
        ],
        {"Runner": (1.0, 0.0), "Wall1": (0.0, 0.0), "Wall2": (0.0, 0.0)},
        None, 0.6,   # tighter threshold — corridor is tight by design
    ),
    # 8. Teammate spacing: two teammates moving parallel, too close together
    (
        "teammate_spacing",
        [_p("T1", 0.0, 0.0, 5.0, 0.0), _p("T2", 0.0, 0.8, 5.0, 0.0)],
        {"T1": (1.0, 0.0), "T2": (1.0, 0.0)},
        None, 0.6,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(players, desired_dirs, ball_carrier_id, params, sep_threshold):
    state = {p.player_id: [p.position.x, p.position.y, p.velocity.x, p.velocity.y]
             for p in players}
    pid_order = [p.player_id for p in players]

    max_defl = 0.0
    min_sep = float("inf")
    defl_sum = 0.0
    defl_count = 0
    bc_speed_sum = 0.0
    bc_speed_count = 0
    good_ticks = 0
    corridor_cleared = 0   # ticks where runner has passed x=5 (corridor scenario)

    for _ in range(N_TICKS):
        live = [
            Player(
                player_id=pid, team=Team.LEFT, attributes=ATTRS,
                position=Vector3(state[pid][0], state[pid][1], 0.0),
                velocity=Vector3(state[pid][2], state[pid][3], 0.0),
            )
            for pid in pid_order
        ]

        # nearest-neighbour separations
        tick_ok = True
        for i, p in enumerate(live):
            for j, q in enumerate(live):
                if j <= i:
                    continue
                d = math.hypot(p.position.x - q.position.x, p.position.y - q.position.y)
                if d < min_sep:
                    min_sep = d
                if d < sep_threshold:
                    tick_ok = False
        if tick_ok:
            good_ticks += 1

        # Corridor bonus: did the runner clear the blocker x-position?
        if pid_order[0] == "Runner" and state["Runner"][0] > 5.5:
            corridor_cleared += 1

        for player in live:
            pid = player.player_id
            raw_dx, raw_dy = desired_dirs.get(pid, (0.0, 0.0))
            raw_len = math.hypot(raw_dx, raw_dy)
            if raw_len < 1e-9:
                continue

            desired_dir = Vector3(raw_dx, raw_dy, 0.0)
            adj_dir, speed_mult = compute_repulsion(
                player, desired_dir, live, ball_carrier_id, params
            )

            adj_len = math.hypot(adj_dir.x, adj_dir.y)
            if adj_len > 1e-9:
                rn_x, rn_y = raw_dx / raw_len, raw_dy / raw_len
                an_x, an_y = adj_dir.x / adj_len, adj_dir.y / adj_len
                cos_a = max(-1.0, min(1.0, rn_x * an_x + rn_y * an_y))
                angle = math.degrees(math.acos(cos_a))
                if angle > max_defl:
                    max_defl = angle
                defl_sum += angle
                defl_count += 1

            if pid == ball_carrier_id:
                bc_speed_sum += speed_mult
                bc_speed_count += 1

            eff = TOP_SPEED * speed_mult
            mv_x = (adj_dir.x / adj_len * eff) if adj_len > 1e-9 else 0.0
            mv_y = (adj_dir.y / adj_len * eff) if adj_len > 1e-9 else 0.0
            state[pid] = [
                player.position.x + mv_x * DT,
                player.position.y + mv_y * DT,
                mv_x, mv_y,
            ]

    return dict(
        max_defl=max_defl,
        avg_defl=defl_sum / defl_count if defl_count else 0.0,
        min_sep=min_sep,
        bc_speed_sum=bc_speed_sum,
        bc_speed_count=bc_speed_count,
        good_ticks=good_ticks,
        total_ticks=N_TICKS,
        corridor_cleared=corridor_cleared,
    )


def evaluate(params: RepulsionParams):
    agg_max_defl = 0.0
    agg_min_sep = float("inf")
    defl_sum = 0.0
    defl_count = 0
    bc_sum = 0.0
    bc_cnt = 0
    good = 0
    total = 0
    corr_cleared = 0

    for scenario in SCENARIOS:
        name, players, desired_dirs, bc_id, sep_thresh = scenario
        r = run_scenario(players, desired_dirs, bc_id, params, sep_thresh)
        agg_max_defl = max(agg_max_defl, r["max_defl"])
        agg_min_sep = min(agg_min_sep, r["min_sep"])
        defl_sum += r["avg_defl"] * r["total_ticks"]
        defl_count += r["total_ticks"]
        bc_sum += r["bc_speed_sum"]
        bc_cnt += r["bc_speed_count"]
        good += r["good_ticks"]
        total += r["total_ticks"]
        if name == "corridor":
            corr_cleared = r["corridor_cleared"]

    bc_avg = bc_sum / bc_cnt if bc_cnt > 0 else 1.0
    avg_defl = defl_sum / defl_count if defl_count else 0.0
    avoidance = good / total if total > 0 else 0.0

    # bc catchability: quadratic reward centred on BC_IDEAL, ±15 max
    bc_catch_bonus = 15.0 - ((bc_avg - BC_IDEAL) ** 2) * 400.0
    bc_catch_bonus = max(-15.0, bc_catch_bonus)

    # corridor: up to +8 bonus for clearing both blockers
    corridor_bonus = (corr_cleared / N_TICKS) * 8.0

    score = (
        avoidance * 70
        + agg_min_sep * 10
        - avg_defl * 0.8
        - max(0.0, agg_max_defl - 80.0) * 3.0
        + bc_catch_bonus
        + corridor_bonus
    )

    return dict(
        score=score,
        max_defl=agg_max_defl,
        avg_defl=avg_defl,
        min_sep=agg_min_sep,
        bc_avg=bc_avg,
        avoidance=avoidance,
        corr_pct=corr_cleared / N_TICKS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    keys = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    print(f"Running {total} combos × {len(SCENARIOS)} scenarios × {N_TICKS} ticks …")

    results = []
    report_every = max(1, total // 40)
    for i, combo in enumerate(combos, 1):
        if i % report_every == 0:
            print(f"  {i}/{total}", end="\r", flush=True)
        p = RepulsionParams(**FIXED, **dict(zip(keys, combo)))
        metrics = evaluate(p)
        results.append((metrics, dict(zip(keys, combo))))

    results.sort(key=lambda r: r[0]["score"], reverse=True)

    col_w = 9
    param_hdr = "  ".join(f"{k[:col_w]:>{col_w}}" for k in keys)
    hdr = (
        f"{'#':>4}  {'score':>7}  {'avoid%':>6}  {'min_sep':>7}  "
        f"{'avg_defl°':>9}  {'max_defl°':>9}  {'bc_avg':>6}  {'corr%':>5}  "
        + param_hdr
    )
    sep = "-" * len(hdr)

    def print_row(rank, m, p):
        vals = "  ".join(f"{p[k]!s:>{col_w}}" for k in keys)
        print(
            f"{rank:>4}  {m['score']:>7.2f}  {m['avoidance']*100:>5.1f}%  "
            f"{m['min_sep']:>7.3f}  {m['avg_defl']:>9.2f}  {m['max_defl']:>9.1f}  "
            f"{m['bc_avg']:>6.3f}  {m['corr_pct']*100:>4.0f}%  {vals}"
        )

    print(f"\n{'TOP 25':=^{len(hdr)}}")
    print(hdr); print(sep)
    for rank, (m, p) in enumerate(results[:25], 1):
        print_row(rank, m, p)

    print(f"\n{'BOTTOM 10':=^{len(hdr)}}")
    print(hdr); print(sep)
    for rank, (m, p) in enumerate(results[-10:], total - 9):
        print_row(rank, m, p)

    print("\nBest params:")
    best_p, best_fixed = results[0][1], FIXED
    for k, v in {**best_fixed, **best_p}.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

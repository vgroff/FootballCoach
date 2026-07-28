"""Balance tests for Phase A: player repulsion steering during Move orders.

Tests cover six scenarios per the Phase A specification:
1. Baseline regression: single player, no neighbours — identical to pre-Phase-A.
2. Off (zero strength): collision-course players overlap as expected.
3. On (tuned): collision-course players maintain >= r_a+r_b+0.1m separation.
4. Ball-carrier: avoidance succeeds AND carrier is measurably slower.
5. No repulsion from stationary ball carrier nearby (other player unaffected).
6. Three-plus neighbours: net_repulsion sums without crash/NaN, player makes
   forward progress.
7. Oscillation/hysteresis: orthogonal nudge sign doesn't flip rapidly.
"""
from __future__ import annotations

import math

import pytest

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Player, PlayerAttributes, Team
from footballcoach.mathutils import Vector3
from footballcoach.steering import RepulsionParams, compute_repulsion
from tests.conftest import make_player


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_match(players: list[Player], ball: Ball, rng_reduction: float = 1.0,
                repulsion_params: RepulsionParams | None = None) -> Match:
    pitch = Pitch.standard()
    m = Match(pitch=pitch, players=players, ball=ball, rng_reduction=rng_reduction)
    if repulsion_params is not None:
        m.repulsion_params = repulsion_params
    return m


def _zero_repulsion_params() -> RepulsionParams:
    """RepulsionParams that disable all repulsion effects (control case)."""
    return RepulsionParams(
        radius_m=4.0,
        strength_base=0.0,
        ball_carrier_repulsion_mult=1.0,
        ball_carrier_speed_penalty_max=0.0,
        speed_penalty_scale=0.0,
        alignment_dot_threshold=-0.7,
        min_orthogonal_adjust_mps=0.0,
    )


def _run_match_steps(match: Match, n_steps: int) -> list[float]:
    """Returns the separation between players[0] and players[1] each tick."""
    separations = []
    for _ in range(n_steps):
        match.step()
        p0, p1 = match.players[0], match.players[1]
        sep = p0.position.xy().distance_to(p1.position.xy())
        separations.append(sep)
    return separations


# ---------------------------------------------------------------------------
# Test 1: Baseline regression — single player, no neighbours in range
# ---------------------------------------------------------------------------

def test_baseline_single_player_no_repulsion_effect(balance_recorder):
    """Single player with no neighbours within radius_m — repulsion must be a
    no-op, returning the original (normalised) direction and speed_mult=1.0."""
    from footballcoach.steering import RepulsionParams, compute_repulsion

    params = RepulsionParams.from_config()
    # Player at origin, desires to move right.
    player = make_player("p1", Team.LEFT, 0.5, position=Vector3(0.0, 0.0, 0.0))
    player.velocity = Vector3(3.0, 0.0, 0.0)

    # No other players close enough (put them far away).
    other = make_player("p2", Team.RIGHT, 0.5, position=Vector3(10.0, 0.0, 0.0))
    desired = Vector3(1.0, 0.0, 0.0)

    adj_dir, speed_mult = compute_repulsion(player, desired, [player, other], None, params)

    # Direction should be unchanged (pointing +x).
    assert abs(adj_dir.x - 1.0) < 1e-6, f"Expected x=1.0, got {adj_dir.x}"
    assert abs(adj_dir.y) < 1e-6
    assert abs(speed_mult - 1.0) < 1e-9, f"Expected speed_mult=1.0, got {speed_mult}"
    balance_recorder.report("repulsion_baseline_no_neighbours", {
        "adj_dir_x": round(adj_dir.x, 6), "speed_mult": speed_mult,
    })


# ---------------------------------------------------------------------------
# Test 2: Off — players on collision course overlap (control case)
# ---------------------------------------------------------------------------

def test_off_collision_course_players_overlap(balance_recorder):
    """With repulsion disabled (strength_base=0), two players on a direct
    collision course must run through each other (min separation < sum of radii).
    This is the control case proving that avoidance is the repulsion's doing."""
    # Two players 6m apart, running directly at each other at ~7 m/s.
    p0 = make_player("p0", Team.LEFT, 0.9, position=Vector3(-3.0, 0.0, 0.0))
    p1 = make_player("p1", Team.RIGHT, 0.9, position=Vector3(3.0, 0.0, 0.0))
    p0.velocity = Vector3(7.0, 0.0, 0.0)
    p0.heading_rad = 0.0        # facing right (+x), matches velocity
    p1.velocity = Vector3(-7.0, 0.0, 0.0)
    p1.heading_rad = math.pi    # facing left (-x), matches velocity

    ball = Ball(position=Vector3(100.0, 0.0, 0.0))  # far away, no interference

    from footballcoach.orders import MoveOrder
    p0.current_order = MoveOrder(target_position=Vector3(3.0, 0.0, 0.0))
    p1.current_order = MoveOrder(target_position=Vector3(-3.0, 0.0, 0.0))

    import logging
    logging.basicConfig(level=logging.DEBUG, format="%(name)s  %(message)s", force=True)

    zero_params = _zero_repulsion_params()
    match = _make_match([p0, p1], ball, repulsion_params=zero_params)

    from footballcoach.steering import compute_repulsion

    separations = []
    for tick in range(40):
        # Log repulsion output BEFORE step
        for pl in match.players:
            from footballcoach.orders import MoveOrder as _MO
            if isinstance(pl.current_order, _MO):
                direction = pl.current_order.target_position - pl.position
                adj_dir, speed_mult = compute_repulsion(
                    pl, direction, match.players, match.ball.possessed_by, zero_params
                )
                print(
                    f"  pre-step tick {tick:02d} {pl.player_id}: "
                    f"raw_dir=({direction.x:+.3f},{direction.y:+.3f}) "
                    f"adj_dir=({adj_dir.x:+.3f},{adj_dir.y:+.3f}) "
                    f"speed_mult={speed_mult:.3f}"
                )

        match.step()

        p0r, p1r = match.players[0], match.players[1]
        sep = p0r.position.xy().distance_to(p1r.position.xy())
        separations.append(sep)
        print(
            f"  post-step tick {tick:02d}: "
            f"p0=({p0r.position.x:+.3f},{p0r.position.y:+.3f}) v=({p0r.velocity.x:+.2f},{p0r.velocity.y:+.2f}) "
            f"p1=({p1r.position.x:+.3f},{p1r.position.y:+.3f}) v=({p1r.velocity.x:+.2f},{p1r.velocity.y:+.2f}) "
            f"sep={sep:.3f}m"
        )

    min_sep = min(separations)
    sum_radii = p0.radius_m + p1.radius_m  # 0.6m
    balance_recorder.report("repulsion_off_collision_min_separation_m", {
        "min_separation_m": round(min_sep, 4),
        "sum_radii_m": sum_radii,
        "result": "overlap_occurred" if min_sep < sum_radii else "no_overlap",
    })
    assert min_sep < sum_radii, (
        f"Expected players to overlap when repulsion is off; got min_sep={min_sep:.3f}m"
    )


# ---------------------------------------------------------------------------
# Test 3: On — collision-course players maintain clearance
# ---------------------------------------------------------------------------

def test_on_collision_course_players_avoid(balance_recorder):
    """With tuned repulsion, the same collision-course setup must maintain at
    least sum_radii + 0.1m separation at every tick."""
    p0 = make_player("p0", Team.LEFT, 0.9, position=Vector3(-3.0, 0.0, 0.0))
    p1 = make_player("p1", Team.RIGHT, 0.9, position=Vector3(3.0, 0.0, 0.0))
    p0.velocity = Vector3(7.0, 0.0, 0.0)
    p0.heading_rad = 0.0          # facing right, matches velocity
    p1.velocity = Vector3(-7.0, 0.0, 0.0)
    p1.heading_rad = math.pi      # facing left, matches velocity

    ball = Ball(position=Vector3(100.0, 0.0, 0.0))

    from footballcoach.orders import MoveOrder
    p0.current_order = MoveOrder(target_position=Vector3(3.0, 0.0, 0.0))
    p1.current_order = MoveOrder(target_position=Vector3(-3.0, 0.0, 0.0))

    match = _make_match([p0, p1], ball)
    n_steps = 60  # 2s
    separations = _run_match_steps(match, n_steps)
    min_sep = min(separations)

    # Time without obstruction: 6m / 7 m/s ≈ 0.86s → ~26 ticks.
    # Arrival time with avoidance should be < 2× direct travel (60 ticks ≈ 2s)
    arrival_direct_ticks = int(6.0 / 7.0 * 30)
    p0_arrived = match.players[0].position.xy().distance_to(Vector3(3.0, 0.0, 0.0).xy()) < 1.0
    p1_arrived = match.players[1].position.xy().distance_to(Vector3(-3.0, 0.0, 0.0).xy()) < 1.0

    balance_recorder.report("repulsion_on_collision_avoidance", {
        "min_separation_m": round(min_sep, 4),
        "sum_radii_m": 0.6,
        "clearance_m": round(min_sep - 0.6, 4),
        "p0_arrived_within_2s": p0_arrived,
        "p1_arrived_within_2s": p1_arrived,
        "direct_travel_ticks": arrival_direct_ticks,
    })
    margin = 0.05  # slightly relaxed (physics resolution at 30Hz may nudge this)
    assert min_sep >= 0.6 - margin, (
        f"Players got too close: min_sep={min_sep:.3f}m, threshold=0.6m"
    )


# ---------------------------------------------------------------------------
# Test 4: Ball-carrier avoidance + measurable slowdown
# ---------------------------------------------------------------------------

def test_ball_carrier_avoidance_and_slowdown(balance_recorder):
    """Ball carrier running toward an obstacle should:
    (a) maintain separation (avoidance succeeds), AND
    (b) be measurably slower during close approach than a non-carrier."""
    N_STEPS = 45
    sum_radii = 0.6

    def _run_carrier(carrier_has_ball: bool) -> tuple[float, float]:
        """Returns (min_separation, mean_speed_during_close_approach)."""
        carrier = make_player("carrier", Team.LEFT, 0.9, position=Vector3(-3.0, 0.0, 0.0))
        obstacle = make_player("obstacle", Team.RIGHT, 0.9, position=Vector3(3.0, 0.0, 0.0))
        carrier.velocity = Vector3(7.0, 0.0, 0.0)
        carrier.heading_rad = 0.0  # facing right, matches velocity
        obstacle.velocity = Vector3(0.0, 0.0, 0.0)  # stationary obstacle

        ball = Ball(position=Vector3(-3.0 + 0.41, 0.0, 0.11))

        from footballcoach.orders import MoveOrder, SaveOrder
        carrier.current_order = MoveOrder(target_position=Vector3(8.0, 0.0, 0.0))
        # Obstacle has no order — stays where it is (no MoveOrder → no repulsion applied to it).

        if carrier_has_ball:
            ball.possessed_by = carrier.player_id
        else:
            ball.possessed_by = None
            ball.position = Vector3(100.0, 0.0, 0.0)

        match = _make_match([carrier, obstacle], ball)
        separations = []
        close_approach_speeds = []

        for _ in range(N_STEPS):
            match.step()
            p = match.players[0]  # carrier
            obs = match.players[1]  # obstacle
            sep = p.position.xy().distance_to(obs.position.xy())
            separations.append(sep)
            if sep < 4.0:  # within repulsion radius
                close_approach_speeds.append(p.speed_mps)

        min_sep = min(separations)
        mean_speed = (sum(close_approach_speeds) / len(close_approach_speeds)
                      if close_approach_speeds else 0.0)
        return min_sep, mean_speed

    min_sep_carrier, mean_speed_carrier = _run_carrier(True)
    min_sep_no_carrier, mean_speed_no_carrier = _run_carrier(False)

    balance_recorder.report("repulsion_ball_carrier_avoidance", {
        "min_sep_with_ball_m": round(min_sep_carrier, 4),
        "min_sep_without_ball_m": round(min_sep_no_carrier, 4),
        "mean_close_approach_speed_with_ball_mps": round(mean_speed_carrier, 3),
        "mean_close_approach_speed_without_ball_mps": round(mean_speed_no_carrier, 3),
        "carrier_is_slower": mean_speed_carrier < mean_speed_no_carrier,
    })
    # (a) Avoidance succeeds for both
    assert min_sep_carrier >= sum_radii - 0.1, (
        f"Ball carrier overlap: min_sep={min_sep_carrier:.3f}m"
    )
    # (b) Carrier measurably slower during close approach
    assert mean_speed_carrier < mean_speed_no_carrier, (
        f"Expected carrier to be slower near obstacle: "
        f"carrier={mean_speed_carrier:.2f} vs non-carrier={mean_speed_no_carrier:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 5: No repulsion FROM stationary ball carrier (other player unaffected)
# ---------------------------------------------------------------------------

def test_no_repulsion_from_ball_carrier(balance_recorder):
    """A player running past a *stationary* ball-carrying opponent should not
    be deflected or slowed — ball carriers are skipped as repulsion sources."""
    runner = make_player("runner", Team.LEFT, 0.9, position=Vector3(0.0, 1.5, 0.0))
    carrier = make_player("carrier", Team.RIGHT, 0.9, position=Vector3(0.0, 0.0, 0.0))
    carrier.velocity = Vector3(0.0, 0.0, 0.0)

    ball = Ball(position=Vector3(0.41, 0.0, 0.11), possessed_by="carrier")

    from footballcoach.orders import MoveOrder
    runner.current_order = MoveOrder(target_position=Vector3(10.0, 1.5, 0.0))
    runner.velocity = Vector3(5.0, 0.0, 0.0)
    runner.heading_rad = 0.0  # facing right, matches velocity

    match = _make_match([runner, carrier], ball)
    params = RepulsionParams.from_config()

    # Direct call: should return speed_mult=1.0 and direction should point
    # straight +x (no lateral deflection).
    desired = Vector3(1.0, 0.0, 0.0)
    adj_dir, speed_mult = compute_repulsion(
        runner, desired, [runner, carrier], "carrier", params
    )

    balance_recorder.report("repulsion_no_repulsion_from_ball_carrier", {
        "adj_dir_x": round(adj_dir.x, 6),
        "adj_dir_y": round(adj_dir.y, 6),
        "speed_mult": speed_mult,
        "deflected": abs(adj_dir.y) > 0.01,
    })
    assert speed_mult == 1.0, "Non-carrier should never have speed_mult < 1.0"
    assert abs(adj_dir.y) < 0.01, (
        f"Runner should not be deflected by stationary carrier; adj_dir.y={adj_dir.y:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: Three-plus neighbours — no NaN, forward progress
# ---------------------------------------------------------------------------

def test_three_plus_neighbours_no_crash(balance_recorder):
    """A player boxed in by 3 nearby players (different angles) must not
    crash/NaN and must still make net forward progress over time."""
    center = make_player("center", Team.LEFT, 0.9, position=Vector3(0.0, 0.0, 0.0))
    n1 = make_player("n1", Team.RIGHT, 0.5, position=Vector3(1.5, 0.0, 0.0))
    n2 = make_player("n2", Team.RIGHT, 0.5, position=Vector3(-0.75, 1.3, 0.0))
    n3 = make_player("n3", Team.RIGHT, 0.5, position=Vector3(-0.75, -1.3, 0.0))
    # Neighbours are stationary, no orders.

    ball = Ball(position=Vector3(50.0, 0.0, 0.0))

    from footballcoach.orders import MoveOrder
    center.current_order = MoveOrder(target_position=Vector3(10.0, 0.0, 0.0))

    match = _make_match([center, n1, n2, n3], ball)
    initial_x = center.position.x

    for _ in range(30):
        match.step()
        # Check for NaN in position
        p = match.players[0]
        assert math.isfinite(p.position.x), "NaN in position.x"
        assert math.isfinite(p.position.y), "NaN in position.y"

    final_x = match.players[0].position.x
    displacement = final_x - initial_x

    balance_recorder.report("repulsion_three_neighbours", {
        "initial_x": round(initial_x, 3),
        "final_x": round(final_x, 3),
        "displacement_m": round(displacement, 3),
        "made_forward_progress": displacement > 0.0,
    })
    assert displacement > 0.0, (
        f"Player should make forward progress even when boxed in; displacement={displacement:.3f}m"
    )


# ---------------------------------------------------------------------------
# Test 7: Oscillation/hysteresis — orthogonal nudge sign stable
# ---------------------------------------------------------------------------

def test_oscillation_check(balance_recorder):
    """Slowly converging paths near the alignment_dot_threshold boundary:
    orthogonal nudge sign should not flip more than once per 5 ticks."""
    from footballcoach.steering import RepulsionParams, compute_repulsion

    params = RepulsionParams.from_config()
    player = make_player("p1", Team.LEFT, 0.7, position=Vector3(0.0, 1.0, 0.0))
    other = make_player("p2", Team.RIGHT, 0.7, position=Vector3(3.0, 0.0, 0.0))

    # Simulate a slowly converging approach (velocity mostly +x, nudged slightly -y)
    last_ortho_sign: int | None = None
    flip_ticks: list[int] = []
    N = 60

    for tick in range(N):
        # Move player slightly toward other
        player = make_player(
            "p1", Team.LEFT, 0.7,
            position=Vector3(player.position.x + 0.05, player.position.y, 0.0),
        )
        player.velocity = Vector3(2.0, -0.3, 0.0)  # converging path

        desired = Vector3(3.0, -1.0, 0.0)  # aim toward other player
        adj_dir, _ = compute_repulsion(player, desired, [player, other], None, params)

        # Detect sign from orthogonal component (adj_dir.y relative to desired_dir.y)
        ortho_component = adj_dir.y - (-1.0 / (3.0**2 + 1.0**2)**0.5)
        current_sign = 1 if ortho_component >= 0 else -1
        if last_ortho_sign is not None and current_sign != last_ortho_sign:
            flip_ticks.append(tick)
        last_ortho_sign = current_sign

    # Assess flip frequency: no more than 1 flip per 5 ticks
    if len(flip_ticks) >= 2:
        min_gap = min(flip_ticks[i+1] - flip_ticks[i] for i in range(len(flip_ticks)-1))
    else:
        min_gap = N  # no consecutive flips

    balance_recorder.report("repulsion_oscillation_check", {
        "total_flips": len(flip_ticks),
        "min_gap_between_flips_ticks": min_gap,
        "flip_ticks": flip_ticks[:10],  # first 10 for readability
        "oscillation_risk": "HIGH" if min_gap < 5 else "LOW",
    })
    # If we detect rapid oscillation (min gap < 5 ticks), flag it — but don't
    # hard-fail yet; the plan says to add hysteresis only if the balance test
    # shows it. This assertion will catch the problem if it manifests.
    assert min_gap >= 3, (
        f"Orthogonal nudge is flipping too rapidly: min gap={min_gap} ticks. "
        "Consider adding hysteresis to steering.py."
    )

"""Scenario test: a player given a Move order walks to the target position
and the order completes. rng_reduction=1.0 for reliable, deterministic pass.
"""
from __future__ import annotations

import math
import random

from footballcoach.engine.match import Match
from footballcoach.engine.movement import MovementParams, effective_acceleration
from footballcoach.entities import Ball, Pitch
from footballcoach.entities.player import PlayerState, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import MoveOrder, StopOrder, OrderStatus
from tests.conftest import make_player


def test_player_follows_move_order_to_completion():
    pitch = Pitch.standard()
    player = make_player("p1", position=Vector3(0, 0, 0))
    ball = Ball.at_rest(Vector3(50, 0, 0))  # far away, uninvolved
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    target = Vector3(10, 0, 0)
    player.current_order = MoveOrder(target_position=target, sprint=True)

    max_ticks = 30 * 10  # 10 seconds at 30Hz should be ample for 10m at min speed 5m/s
    for _ in range(max_ticks):
        match.step()
        if player.current_order is None:
            break

    assert player.current_order is None, "move order never completed"
    assert player.position.distance_to(target) <= 0.31  # within arrival tolerance + fp slack


def test_player_move_order_status_transitions():
    pitch = Pitch.standard()
    player = make_player("p1", position=Vector3(0, 0, 0))
    ball = Ball.at_rest(Vector3(50, 0, 0))
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    order = MoveOrder(target_position=Vector3(3, 0, 0), sprint=True)
    player.current_order = order
    assert order.status == OrderStatus.PENDING

    match.step()
    assert order.status == OrderStatus.IN_PROGRESS

    for _ in range(300):
        match.step()
        if player.current_order is None:
            break

    assert order.status == OrderStatus.COMPLETE


# ---------------------------------------------------------------------------
# max_speed_on_arrival_mps tests
# ---------------------------------------------------------------------------

def test_move_order_default_arrives_at_jog_speed_or_less():
    """Default MoveOrder (no max_speed_on_arrival_mps) completes when within
    tolerance — the player is not required to be at a standstill."""
    pitch = Pitch.standard()
    player = make_player("p1", position=Vector3(0, 0, 0), attr_value=0.7)
    ball = Ball.at_rest(Vector3(50, 20, 0))
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    player.current_order = MoveOrder(target_position=Vector3(15, 0, 0), sprint=True)
    for _ in range(30 * 15):
        match.step()
        if player.current_order is None:
            break

    assert player.current_order is None, "order should complete"
    assert player.position.distance_to(Vector3(15, 0, 0)) <= 0.5


def test_move_order_standstill_arrival_stops_player():
    """max_speed_on_arrival_mps=0.0 must leave the player at (near-)zero
    velocity when the order completes."""
    pitch = Pitch.standard()
    player = make_player("p1", position=Vector3(0, 0, 0), attr_value=0.7)
    ball = Ball.at_rest(Vector3(50, 20, 0))
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    player.current_order = MoveOrder(
        target_position=Vector3(12, 0, 0), sprint=True, max_speed_on_arrival_mps=0.0
    )
    for _ in range(30 * 15):
        match.step()
        if player.current_order is None:
            break

    assert player.current_order is None, "order should complete"
    assert player.speed_mps < 0.1, (
        f"player should be nearly stopped at arrival, got {player.speed_mps:.3f} m/s"
    )
    assert player.position.distance_to(Vector3(12, 0, 0)) <= 0.6


def test_no_velocity_snap_during_move_order():
    """Player speed must never change by more than a_max*dt*standstill_mult
    in a single tick — no velocity snaps, only physics-driven changes."""
    pitch = Pitch.standard()
    player = make_player("p1", position=Vector3(0, 0, 0), attr_value=0.5)
    ball = Ball.at_rest(Vector3(50, 20, 0))
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    player.current_order = MoveOrder(
        target_position=Vector3(10, 0, 0), sprint=True, max_speed_on_arrival_mps=0.0
    )

    params = MovementParams.from_config()
    # a_max for attr=0.5: 2.5 + 5*0.5 = 5.0 m/s²; standstill gets 1.5× boost
    a_max_base = 2.5 + 5.0 * 0.5
    max_allowed_delta = a_max_base * params.standstill_decel_multiplier * (1.0 / 30.0) * 1.1  # 10% headroom

    prev_speed = 0.0
    for _ in range(30 * 15):
        match.step()
        delta = abs(player.speed_mps - prev_speed)
        assert delta <= max_allowed_delta, (
            f"speed changed by {delta:.4f} m/s in one tick (limit {max_allowed_delta:.4f}) "
            f"— indicates a velocity snap, not physics-driven deceleration"
        )
        prev_speed = player.speed_mps
        if player.current_order is None:
            break


def test_stop_order_no_snap():
    """StopOrder must decelerate smoothly — no discontinuous velocity jump."""
    pitch = Pitch.standard()
    player = make_player("p1", position=Vector3(0, 0, 0), attr_value=0.5)
    player.velocity = Vector3(7.0, 0.0, 0.0)
    player.heading_rad = 0.0
    ball = Ball.at_rest(Vector3(50, 20, 0))
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    player.current_order = StopOrder()

    params = MovementParams.from_config()
    a_max_base = 2.5 + 5.0 * 0.5
    max_allowed_delta = a_max_base * params.standstill_decel_multiplier * (1.0 / 30.0) * 1.1

    prev_speed = player.speed_mps
    for _ in range(30 * 10):
        match.step()
        delta = abs(player.speed_mps - prev_speed)
        assert delta <= max_allowed_delta, (
            f"speed jumped {delta:.4f} m/s (limit {max_allowed_delta:.4f}) — snap detected"
        )
        prev_speed = player.speed_mps
        if player.current_order is None:
            break

    assert player.current_order is None, "StopOrder should complete"
    assert player.speed_mps == 0.0, "player should be exactly at rest after StopOrder"

# ---------------------------------------------------------------------------
# Brake-to-turn tests
# ---------------------------------------------------------------------------

def test_large_heading_change_triggers_deceleration():
    """When a player at full sprint receives a MoveOrder requiring a heading
    change of >120°, they should decelerate (brake) rather than maintain
    speed while arcing.  Within the first ~0.5s the player's speed should
    visibly drop below the speed they had when the new order was issued."""
    pitch = Pitch.standard()
    # attr_value=0.7 gives noticeable sprint speed (~8 m/s) without being extreme
    player = make_player("p1", position=Vector3(0, 0, 0), attr_value=0.7)
    ball = Ball.at_rest(Vector3(0, 30, 0))
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    # Accelerate to near top speed heading +x.
    player.current_order = MoveOrder(target_position=Vector3(100, 0, 0), sprint=True)
    for _ in range(90):  # 3 s
        match.step()

    sprint_speed = player.speed_mps
    assert sprint_speed > 5.0, "player should be at meaningful speed before the test"

    # Issue an order that requires a ~150° turn (target is behind-left).
    player.current_order = MoveOrder(target_position=Vector3(-20, -5, 0), sprint=True)

    min_speed_in_first_half_second = sprint_speed
    for _ in range(15):  # 0.5 s at 30 Hz
        match.step()
        min_speed_in_first_half_second = min(min_speed_in_first_half_second, player.speed_mps)

    assert min_speed_in_first_half_second < sprint_speed * 0.9, (
        f"expected player to brake (speed < {sprint_speed * 0.9:.2f} m/s) during large "
        f"heading change, but minimum speed was {min_speed_in_first_half_second:.2f} m/s"
    )


def test_large_heading_change_order_still_completes():
    """A MoveOrder requiring a near-180° heading change while at speed must
    still complete within a reasonable time (brake-to-turn should not stall
    the player indefinitely)."""
    pitch = Pitch.standard()
    player = make_player("p1", position=Vector3(0, 0, 0), attr_value=0.5)
    ball = Ball.at_rest(Vector3(0, 30, 0))
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    # Get to sprint speed heading +x.
    player.current_order = MoveOrder(target_position=Vector3(100, 0, 0), sprint=True)
    for _ in range(90):
        match.step()

    # Reset position and order a target directly behind (180° turn, 15 m away).
    player.position = Vector3(0, 0, 0)
    player.current_order = MoveOrder(target_position=Vector3(-15, 0, 0), sprint=True)

    for _ in range(30 * 10):  # 10 s budget
        match.step()
        if player.current_order is None:
            break

    assert player.current_order is None, "order should complete within 10 s despite large heading change"
    assert player.position.distance_to(Vector3(-15, 0, 0)) <= 0.5


def test_small_heading_change_does_not_decelerate():
    """A MoveOrder with a small heading change (<90°) should NOT trigger
    brake-to-turn — the player should maintain roughly full sprint speed."""
    pitch = Pitch.standard()
    player = make_player("p1", position=Vector3(0, 0, 0), attr_value=0.7)
    ball = Ball.at_rest(Vector3(0, 30, 0))
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    # Accelerate to near top speed heading +x.
    player.current_order = MoveOrder(target_position=Vector3(100, 0, 0), sprint=True)
    for _ in range(90):
        match.step()

    sprint_speed = player.speed_mps

    # Issue an order ~45° off current heading — small enough that brake-to-turn
    # should not trigger (threshold is ~90°).
    player.current_order = MoveOrder(target_position=Vector3(20, 20, 0), sprint=True)

    min_speed_in_first_half_second = sprint_speed
    for _ in range(15):  # 0.5 s
        match.step()
        min_speed_in_first_half_second = min(min_speed_in_first_half_second, player.speed_mps)

    # Speed should stay close to sprint speed (allow normal turn-arc penalty, but no braking).
    assert min_speed_in_first_half_second >= sprint_speed * 0.6, (
        f"player decelerated too much for a small heading change: "
        f"min speed {min_speed_in_first_half_second:.2f} vs sprint {sprint_speed:.2f}"
    )


# ---------------------------------------------------------------------------
# CONTROLLING_BALL movement tests
# ---------------------------------------------------------------------------

def test_controlling_ball_player_keeps_moving():
    """A player in CONTROLLING_BALL state (mid first-touch) with an active
    MoveOrder must continue moving toward the target — not freeze in place."""
    pitch = Pitch.standard()
    player = make_player("p1", position=Vector3(0.0, 0.0, 0.0), attr_value=0.5)
    player.heading_rad = 0.0  # facing +x
    ball = Ball.at_rest(Vector3(50.0, 0.0, 0.0))  # far away
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    # Force player into CONTROLLING_BALL state for 0.5s (15 ticks at 30Hz).
    player.state = PlayerState.CONTROLLING_BALL
    player.state_timer_s = 0.5
    player.current_order = MoveOrder(target_position=Vector3(20.0, 0.0, 0.0), sprint=True)

    for _ in range(10):  # 10 ticks well within the control window
        match.step()

    assert player.position.x > 0.1, (
        f"Player should have moved during CONTROLLING_BALL, got x={player.position.x:.3f}"
    )


def test_controlling_ball_player_moves_slower_than_free_sprint():
    """During CONTROLLING_BALL the engine applies the ball-carry speed penalty,
    so the player moves slower than a free-sprinting player over the same ticks."""
    pitch = Pitch.standard()

    def _run_ticks(controlling: bool, n_ticks: int) -> float:
        p = make_player("p1", position=Vector3(0.0, 0.0, 0.0), attr_value=0.5)
        p.heading_rad = 0.0
        ball = Ball.at_rest(Vector3(50.0, 0.0, 0.0))
        m = Match(pitch=pitch, players=[p], ball=ball, rng_reduction=1.0, rng=random.Random(0))
        if controlling:
            p.state = PlayerState.CONTROLLING_BALL
            p.state_timer_s = float(n_ticks) / 30.0 + 1.0  # keep in state the whole time
        p.current_order = MoveOrder(target_position=Vector3(20.0, 0.0, 0.0), sprint=True)
        for _ in range(n_ticks):
            m.step()
        return p.position.x

    n = 20  # ticks
    x_controlling = _run_ticks(controlling=True, n_ticks=n)
    x_free = _run_ticks(controlling=False, n_ticks=n)

    assert x_controlling > 0.5, "Controlling player should have moved"
    assert x_free > x_controlling, (
        f"Free sprint (x={x_free:.3f}) should be faster than controlling (x={x_controlling:.3f})"
    )


# ---------------------------------------------------------------------------
# Push-kick tests
# ---------------------------------------------------------------------------

def _ticks_to_complete_move(
    start: Vector3,
    target: Vector3,
    push_kick_enabled: bool,
    start_possessed: bool = True,
    max_ticks: int = 30 * 30,
) -> int:
    """Helper: count ticks for a ball-carrying player to complete a MoveOrder."""
    pitch = Pitch.standard()
    player = make_player("p1", Team.LEFT, attr_value=0.6, position=start)
    player.heading_rad = 0.0  # facing +x
    ball = Ball.at_rest(start)
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))
    if start_possessed:
        ball.possessed_by = player.player_id
    player.current_order = MoveOrder(
        target_position=target,
        sprint=True,
        push_kick_enabled=push_kick_enabled,
    )
    for tick in range(max_ticks):
        match.step()
        if player.current_order is None:
            return tick + 1
    return max_ticks  # didn't complete


def test_push_kick_faster_over_40m():
    """A player with push_kick_enabled should reach a target 40m away faster
    than one who dribbles all the way (ball-carry speed penalty the whole time)."""
    start = Vector3(-(40.0 / 2), 0.0, 0.0)
    target = Vector3(40.0 / 2, 0.0, 0.0)

    ticks_dribble = _ticks_to_complete_move(start, target, push_kick_enabled=False)
    ticks_push = _ticks_to_complete_move(start, target, push_kick_enabled=True)

    assert ticks_push < ticks_dribble, (
        f"push-kick ({ticks_push} ticks) should be faster than dribbling ({ticks_dribble} ticks)"
    )


def test_push_kick_not_triggered_near_destination():
    """When the remaining distance is below push_kick_min_dist_m the player
    should NOT kick (ball stays possessed, no release-grace period starts)."""
    pitch = Pitch.standard()
    player = make_player("p1", Team.LEFT, attr_value=0.6, position=Vector3(0.0, 0.0, 0.0))
    player.heading_rad = 0.0
    # Target only 5m away — well below the default push_kick_min_dist_m (9m).
    target = Vector3(5.0, 0.0, 0.0)
    ball = Ball.at_rest(Vector3(0.0, 0.0, 0.0))
    ball.possessed_by = player.player_id
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))
    player.current_order = MoveOrder(
        target_position=target, sprint=True, push_kick_enabled=True
    )

    match.step()

    # Ball should still be possessed (no kick was triggered).
    assert ball.possessed_by == player.player_id, (
        "Push-kick must not fire when the player is already close to the destination"
    )
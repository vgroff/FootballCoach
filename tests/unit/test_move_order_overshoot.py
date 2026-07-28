"""Unit tests for MoveOrder overshoot timeout behaviour.

Spec:
- `reached_target` is set to True the first tick the player is within
  `arrival_tolerance_m` of the target.
- Before the player has reached the target, no overshoot timer starts.
- Once `reached_target` is True and the player drifts outside
  `arrival_tolerance_m`, `_overshoot_timer_s` starts counting down from
  `overshoot_timeout_s` (default 0.5 s).
- When the timer reaches zero the order completes (current_order = None).
- While the timer is counting down the player decelerates (STANDSTILL mode).
"""
from __future__ import annotations

import random

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import MoveOrder, OrderStatus
from tests.conftest import make_player

DT = 1.0 / 30.0  # physics tick size


def _make_match(player_pos: Vector3, target: Vector3, sprint: bool = True) -> tuple[Match, object]:
    pitch = Pitch.standard()
    player = make_player("p", Team.LEFT, attr_value=0.8, position=player_pos)
    ball = Ball.at_rest(Vector3(0, 20, 0))  # far away
    match = Match(pitch=pitch, players=[player], ball=ball,
                  rng_reduction=1.0, rng=random.Random(0))
    order = MoveOrder(target_position=target, sprint=sprint)
    player.current_order = order
    return match, player


# ---------------------------------------------------------------------------
# reached_target flag
# ---------------------------------------------------------------------------

def test_reached_target_not_set_when_far():
    """Player starting 5 m away must NOT have reached_target after 1 tick."""
    match, player = _make_match(Vector3(-5, 0, 0), Vector3(0, 0, 0))
    order = player.current_order
    match.step()
    assert not order.reached_target


def test_reached_target_set_when_within_tolerance():
    """Player starting inside arrival_tolerance_m must have reached_target = True
    after one tick."""
    match, player = _make_match(Vector3(0.1, 0, 0), Vector3(0, 0, 0))
    order = player.current_order
    match.step()
    # Order may complete on this tick, but reached_target must have been set.
    assert order.reached_target or player.current_order is None


def test_normal_completion_within_tolerance():
    """Player starting inside tolerance must complete the order normally (no
    overshoot timer needed)."""
    match, player = _make_match(Vector3(0.1, 0, 0), Vector3(0, 0, 0))
    match.step()
    assert player.current_order is None


# ---------------------------------------------------------------------------
# Overshoot timer
# ---------------------------------------------------------------------------

def test_overshoot_timer_does_not_start_before_reaching_target():
    """Player still approaching from far away: no overshoot timer."""
    match, player = _make_match(Vector3(-5, 0, 0), Vector3(0, 0, 0))
    order = player.current_order
    for _ in range(5):
        match.step()
        if not order.reached_target:
            assert order._overshoot_timer_s is None


def test_overshoot_timer_starts_after_crossing_target():
    """Simulate overshoot by pre-setting reached_target=True and placing the
    player outside tolerance. The timer must start on the next tick."""
    match, player = _make_match(Vector3(0.5, 0, 0), Vector3(0, 0, 0))
    order = player.current_order
    # Manually mark as having reached target; player is 0.5m away (> 0.3m tolerance)
    order.reached_target = True
    # Give player some rightward velocity so they're "past" the target
    player.velocity = Vector3(3.0, 0, 0)
    player.heading_rad = 0.0

    match.step()

    assert order._overshoot_timer_s is not None, (
        "Overshoot timer should have started after reaching_target=True and dist > tolerance"
    )
    assert order._overshoot_timer_s < order.overshoot_timeout_s, (
        "Timer should have decremented by at least one dt"
    )


def test_overshoot_timer_completes_after_timeout():
    """After overshoot_timeout_s seconds of countdown the order must complete."""
    match, player = _make_match(Vector3(0.5, 0, 0), Vector3(0, 0, 0))
    order = player.current_order
    order.reached_target = True
    player.velocity = Vector3(3.0, 0, 0)
    player.heading_rad = 0.0

    # Run for slightly more than 0.5 s (default overshoot_timeout_s)
    ticks_needed = int(order.overshoot_timeout_s / DT) + 5
    for _ in range(ticks_needed):
        if player.current_order is None:
            break
        match.step()

    assert player.current_order is None, (
        f"Order should have completed after overshoot timeout "
        f"({order.overshoot_timeout_s}s); timer was {order._overshoot_timer_s}"
    )


def test_overshoot_player_decelerates_during_countdown():
    """While the overshoot timer counts down the player must be braking
    (speed decreases each tick)."""
    match, player = _make_match(Vector3(0.5, 0, 0), Vector3(0, 0, 0))
    order = player.current_order
    order.reached_target = True
    player.velocity = Vector3(5.0, 0, 0)
    player.heading_rad = 0.0

    speeds = []
    for _ in range(10):
        match.step()
        if player.current_order is None:
            break
        speeds.append(player.speed_mps)

    if len(speeds) >= 2:
        assert speeds[-1] < speeds[0], (
            f"Player should decelerate during overshoot timeout; "
            f"speed went {speeds[0]:.2f} → {speeds[-1]:.2f}"
        )


def test_custom_overshoot_timeout_respected():
    """A MoveOrder with a custom overshoot_timeout_s=0.1 completes much sooner
    than the default 0.5 s."""
    pitch = Pitch.standard()
    player = make_player("p", Team.LEFT, attr_value=0.8, position=Vector3(0.5, 0, 0))
    ball = Ball.at_rest(Vector3(0, 20, 0))
    match = Match(pitch=pitch, players=[player], ball=ball,
                  rng_reduction=1.0, rng=random.Random(0))
    order = MoveOrder(target_position=Vector3(0, 0, 0), sprint=True, overshoot_timeout_s=0.1)
    order.reached_target = True
    player.current_order = order
    player.velocity = Vector3(3.0, 0, 0)
    player.heading_rad = 0.0

    # 0.1 s = 3 ticks; run 10 to be sure
    for _ in range(10):
        if player.current_order is None:
            break
        match.step()

    assert player.current_order is None, "Custom short timeout should complete quickly"

"""Scenario test: a player given a Move order walks to the target position
and the order completes. rng_reduction=1.0 for reliable, deterministic pass.
"""
from __future__ import annotations

import random

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch
from footballcoach.mathutils import Vector3
from footballcoach.orders import MoveOrder, OrderStatus
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

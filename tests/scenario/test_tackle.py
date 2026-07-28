"""Scenario test: a player can tackle another player. rng_reduction=1.0 so
the outcome is deterministic based purely on attribute comparison.
"""
from __future__ import annotations

import random

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import ChaseTackleOrder
from tests.conftest import make_player


def test_tackle_wins_ball_from_carrier():
    pitch = Pitch.standard()
    tackler = make_player("tackler", Team.LEFT, position=Vector3(0, 0, 0), tackling=0.9)
    carrier = make_player("carrier", Team.RIGHT, position=Vector3(0.5, 0, 0), dribbling=0.1)

    ball = Ball.at_rest(Vector3(0.5, 0, 0))
    ball.possessed_by = carrier.player_id

    match = Match(pitch=pitch, players=[tackler, carrier], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    tackler.current_order = ChaseTackleOrder(target_player_id=carrier.player_id)
    match.step()

    assert ball.possessed_by == tackler.player_id
    assert carrier.state.name == "INACTIVE_TACKLED"

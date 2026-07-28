"""Scenario test: a player can kick the ball into the goal from the penalty
spot (no goalkeeper) and the goal is recorded correctly. rng_reduction=1.0
for a reliable, deterministic pass (dead-centre placed kick should always
score with zero error).
"""
from __future__ import annotations

import random

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import KickOrder
from tests.conftest import make_player


def test_penalty_scores_and_goal_recorded():
    pitch = Pitch.standard()
    penalty_spot = pitch.penalty_spot(left=False)  # attacking the right goal
    kicker = make_player("p1", team=Team.LEFT, position=penalty_spot, kick_precision=0.9, kick_power=0.9)

    ball = Ball.at_rest(penalty_spot)
    ball.possessed_by = kicker.player_id

    match = Match(pitch=pitch, players=[kicker], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    # Aim dead centre of the goal at a modest height.
    aim_point = pitch.right_goal_centre.with_z(1.1)
    kicker.current_order = KickOrder(aim_point=aim_point, power_fraction=0.6, spin=Vector3.zero())

    scored = False
    for _ in range(300):
        match.step()
        if match.scoreboard.left_goals > 0:
            scored = True
            break

    assert scored, "penalty from the spot with zero error should score"
    assert match.scoreboard.left_goals == 1
    assert match.scoreboard.right_goals == 0

"""Scenario tests: the high-level actions.py functions (move_to/shoot/
pass_to/tackle/save), run end-to-end through Match at rng_reduction=1.0 for
deterministic pass/fail.
"""
from __future__ import annotations

import random

from footballcoach import actions
from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.mathutils import Vector3
from tests.conftest import make_player


def test_move_to_reaches_target():
    pitch = Pitch.standard()
    player = make_player("p1", position=Vector3(0, 0, 0))
    ball = Ball.at_rest(Vector3(50, 20, 0))
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    target = Vector3(10, 5, 0)
    actions.move_to(player, target)

    for _ in range(300):
        match.step()
        if player.current_order is None:
            break

    assert player.position.distance_to(target) <= 0.31


def test_shoot_scores_from_close_range_no_keeper():
    pitch = Pitch.standard()
    position = Vector3(pitch.half_length - 5, 0, 0)
    kicker = make_player("k", Team.LEFT, position=position, kick_precision=0.9, kick_power=0.9)
    ball = Ball.at_rest(position)
    ball.possessed_by = kicker.player_id
    match = Match(pitch=pitch, players=[kicker], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    actions.shoot(kicker, pitch)

    scored = False
    for _ in range(200):
        match.step()
        if match.scoreboard.left_goals > 0:
            scored = True
            break
    assert scored


def test_pass_to_reaches_teammate():
    pitch = Pitch.standard()
    passer = make_player("passer", Team.LEFT, position=Vector3(0, 0, 0), kick_precision=0.8)
    receiver = make_player("receiver", Team.LEFT, position=Vector3(15, 0, 0))
    ball = Ball.at_rest(passer.position)
    ball.possessed_by = passer.player_id
    match = Match(pitch=pitch, players=[passer, receiver], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    actions.pass_to(passer, receiver.position)

    received = False
    for _ in range(300):
        match.step()
        if ball.possessed_by == receiver.player_id:
            received = True
            break
    assert received


def test_tackle_chases_and_wins_ball():
    pitch = Pitch.standard()
    defender = make_player("d", Team.LEFT, position=Vector3(0, 0, 0), tackling=0.9, top_speed=0.8, acceleration=0.8)
    attacker = make_player("a", Team.RIGHT, position=Vector3(10, 0, 0), dribbling=0.1)
    ball = Ball.at_rest(attacker.position)
    ball.possessed_by = attacker.player_id
    match = Match(pitch=pitch, players=[defender, attacker], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    actions.tackle(defender, attacker)

    won_ball = False
    for _ in range(300):
        match.step()
        if ball.possessed_by == defender.player_id:
            won_ball = True
            break
    assert won_ball


def test_save_intercepts_shot_on_target():
    pitch = Pitch.standard()
    gk = make_player(
        "gk", Team.LEFT, position=pitch.left_goal_centre, is_goalkeeper=True,
        top_speed=0.9, acceleration=0.9, ball_control=0.9,
    )
    shooter = make_player(
        "shooter", Team.RIGHT, position=Vector3(-25, 0, 0), kick_precision=0.9, kick_power=0.7,
    )
    ball = Ball.at_rest(shooter.position)
    ball.possessed_by = shooter.player_id
    match = Match(pitch=pitch, players=[gk, shooter], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    actions.save(gk)
    actions.shoot(shooter, pitch)

    saved = False
    scored = False
    for _ in range(300):
        match.step()
        if ball.possessed_by == gk.player_id:
            saved = True
            break
        if match.scoreboard.right_goals > 0:
            scored = True
            break

    # A shot aimed dead centre from directly in front, with an elite,
    # already-well-positioned keeper and zero RNG error, should be saved.
    assert saved
    assert not scored

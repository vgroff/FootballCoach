"""Unit tests for Match.goal_linger_s — the engine-level delay between a
goal being scored and the ball being reset to centre.

The linger gives the UI time to show the ball in the net before resetting.
With goal_linger_s=0.0 (the headless/test default) the existing behaviour
(immediate reset) must be preserved as a regression check.
"""
from __future__ import annotations

import random

import pytest

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, PlayerAttributes, Team
from footballcoach.entities.player import Player
from footballcoach.mathutils import Vector3


def _make_goal_match(goal_linger_s: float = 0.0) -> Match:
    """Minimal match with a ball already inside the right goal mouth.

    The right goal (positive x side) is attacked by Team.LEFT, so a ball
    crossing the right goal line scores a goal for Team.LEFT → left_goals.
    """
    pitch = Pitch.standard()
    attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    player = Player.create("p", Team.LEFT, attrs, position=Vector3(0, 0, 0))
    # Place ball inside the right goal: x > half_length, |y| < half_goal_width.
    ball = Ball.at_rest(Vector3(pitch.half_length + 0.5, 0.0, 0.1))
    ball.possessed_by = None
    return Match(
        pitch=pitch, players=[player], ball=ball,
        rng_reduction=0.3, rng=random.Random(0),
        goal_linger_s=goal_linger_s,
    )


def test_no_linger_resets_immediately():
    """With goal_linger_s=0.0 a goal immediately resets the ball to centre
    on the same step (existing behaviour, regression check)."""
    match = _make_goal_match(goal_linger_s=0.0)
    assert match.scoreboard.left_goals == 0  # right goal → left team scores
    match.step()
    # Goal should have been scored and ball reset.
    assert match.scoreboard.left_goals == 1
    # Ball must be back near the centre (reset_after_goal puts it at (0,0,0)).
    assert abs(match.ball.position.x) < 1.0


def test_linger_delays_reset():
    """With goal_linger_s > 0 the ball must remain in the net for
    approximately that many sim-seconds before being reset to centre."""
    linger_s = 0.5  # 0.5 s → ~15 ticks at 30 Hz
    match = _make_goal_match(goal_linger_s=linger_s)
    ball_start_x = match.ball.position.x  # inside the goal

    # Step once — goal detected, linger started, ball stays in net.
    match.step()
    # Right goal → left team scores (Team.LEFT attacks positive-x side).
    assert match.scoreboard.left_goals == 1, "goal must be scored on the detection tick"
    assert match.ball.position.x == pytest.approx(ball_start_x, abs=2.0), (
        "ball must not reset to centre while linger is active"
    )
    assert match._goal_linger_remaining_s > 0.0, "linger countdown must have started"

    # Keep stepping until the linger expires.
    dt = match.dt_s
    expected_ticks = round(linger_s / dt)
    for _ in range(expected_ticks + 10):
        match.step()
        if match._goal_linger_remaining_s <= 0.0:
            break

    # After linger, ball must be reset.
    assert match._goal_linger_remaining_s == pytest.approx(0.0, abs=dt)
    assert abs(match.ball.position.x) < 1.0, (
        "ball must be reset to centre after the linger expires"
    )


def test_no_double_goal_during_linger():
    """No additional goal must be recorded while the linger countdown is
    active, even though the ball stays 'in the net' position."""
    linger_s = 0.5
    match = _make_goal_match(goal_linger_s=linger_s)
    match.step()  # goal detected → linger started
    goals_after_detection = match.scoreboard.left_goals  # right goal → left team
    assert goals_after_detection == 1

    dt = match.dt_s
    for _ in range(round(linger_s / dt)):
        match.step()

    # Must still be exactly 1 goal — no phantom re-detection.
    assert match.scoreboard.left_goals == 1


def test_linger_countdown_decreases_each_tick():
    """_goal_linger_remaining_s must decrease by dt each step during linger."""
    linger_s = 0.3
    match = _make_goal_match(goal_linger_s=linger_s)
    match.step()  # triggers linger
    remaining_before = match._goal_linger_remaining_s
    match.step()
    remaining_after = match._goal_linger_remaining_s
    assert remaining_after == pytest.approx(remaining_before - match.dt_s, abs=1e-9)

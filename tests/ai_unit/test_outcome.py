"""Regression tests for detect_trial_outcome()'s ball-out-of-bounds branch.

Covers two fixes:
1. The "ball comes to rest unpossessed" branch (a second, undocumented way
   to reach "miss") was removed -- a loose ball that just stops rolling on
   the pitch is no longer a trial-ending event by itself.
2. Ball out of bounds is still detected purely from ball position, with no
   dependency on possession history -- attribution of who caused it lives
   in ScenarioEnv (see test_scenario_env_reward_wiring.py-style tests),
   not in detect_trial_outcome() itself.
"""
from __future__ import annotations

from footballcoach.ai.env.outcome import detect_trial_outcome
from footballcoach.mathutils import Vector3


def test_ball_out_of_bounds_on_x_returns_miss(duel_match):
    match = duel_match
    match.ball.position = Vector3(match.pitch.half_length + 2.0, 0.0, 0.0)
    match.ball.possessed_by = None
    outcome, half_linger = detect_trial_outcome(
        match,
        initial_scoreboard=(0, 0),
        initial_carrier_id="p1",
        ball_released=True,
        box_possession_terminal=True,
        trial_tick=10,
        timeout_ticks=500,
    )
    assert outcome == "miss"
    assert half_linger is True


def test_ball_out_of_bounds_on_y_returns_miss(duel_match):
    match = duel_match
    match.ball.position = Vector3(0.0, match.pitch.half_width + 1.0, 0.0)
    match.ball.possessed_by = None
    outcome, half_linger = detect_trial_outcome(
        match,
        initial_scoreboard=(0, 0),
        initial_carrier_id="p1",
        ball_released=True,
        box_possession_terminal=True,
        trial_tick=10,
        timeout_ticks=500,
    )
    assert outcome == "miss"
    assert half_linger is True


def _detect(match):
    return detect_trial_outcome(
        match,
        initial_scoreboard=(0, 0),
        initial_carrier_id="p1",
        ball_released=True,
        box_possession_terminal=False,
        trial_tick=10,
        timeout_ticks=500,
    )[0]


def test_ball_is_in_play_until_the_whole_ball_is_over_the_line(duel_match):
    """Law 9: out only once the WHOLE ball has crossed the WHOLE line, i.e. the
    centre is more than one ball radius past it -- on the line or overlapping it
    is still in play, on either axis."""
    match = duel_match
    r = match.ball.radius_m
    for pos in (
        Vector3(0.0, match.pitch.half_width, 0.0),
        Vector3(0.0, match.pitch.half_width + 0.5 * r, 0.0),
        Vector3(0.0, -(match.pitch.half_width + 0.99 * r), 0.0),
        Vector3(match.pitch.half_length + 0.5 * r, 20.0, 0.0),
        Vector3(-(match.pitch.half_length + 0.99 * r), 20.0, 0.0),
    ):
        match.ball.position = pos
        match.ball.possessed_by = None
        assert _detect(match) is None, pos


def test_ball_is_out_as_soon_as_the_whole_ball_is_over_the_line(duel_match):
    """The old rule waited until the centre was 0.5 m (touchline) / 1.0 m
    (goal line) past the line; a ball 0.3 m over the touchline must now be out."""
    match = duel_match
    r = match.ball.radius_m
    for pos in (
        Vector3(0.0, match.pitch.half_width + r + 0.01, 0.0),
        Vector3(0.0, -(match.pitch.half_width + 0.3), 0.0),
        Vector3(match.pitch.half_length + r + 0.01, 20.0, 0.0),
        Vector3(-(match.pitch.half_length + 0.3), 20.0, 0.0),
        Vector3(0.0, match.pitch.half_width + r + 0.01, 2.5),  # in the air: height is irrelevant
    ):
        match.ball.position = pos
        match.ball.possessed_by = None
        assert _detect(match) == "miss", pos


def test_a_scored_goal_beats_out_of_bounds_even_when_the_ball_is_far_past_the_line(duel_match):
    """A hard shot moves ~0.7 m per tick, so on the tick Match scores it the ball
    centre can already be more than a ball radius past the goal line. That must
    still be a goal, not a miss."""
    match = duel_match
    match.ball.position = Vector3(match.pitch.half_length + 0.7, 0.0, 1.0)
    match.ball.possessed_by = None
    match.scoreboard.left_goals = 1
    assert _detect(match) == "goal"


def test_loose_ball_at_rest_on_pitch_is_not_an_outcome(duel_match):
    """A loose, stationary ball still on the pitch must NOT end the trial --
    the old "comes to rest unpossessed" branch is gone. Only timeout should
    eventually end an episode like this."""
    match = duel_match
    match.ball.position = Vector3(5.0, 0.0, 0.0)  # well inside the pitch
    match.ball.velocity = Vector3(0.0, 0.0, 0.0)
    match.ball.possessed_by = None
    outcome, _ = detect_trial_outcome(
        match,
        initial_scoreboard=(0, 0),
        initial_carrier_id="p1",
        ball_released=True,
        box_possession_terminal=True,
        trial_tick=10,
        timeout_ticks=500,
    )
    assert outcome is None


def test_loose_ball_at_rest_eventually_times_out(duel_match):
    match = duel_match
    match.ball.position = Vector3(5.0, 0.0, 0.0)
    match.ball.velocity = Vector3(0.0, 0.0, 0.0)
    match.ball.possessed_by = None
    outcome, _ = detect_trial_outcome(
        match,
        initial_scoreboard=(0, 0),
        initial_carrier_id="p1",
        ball_released=True,
        box_possession_terminal=True,
        trial_tick=500,
        timeout_ticks=500,
    )
    assert outcome == "timeout"

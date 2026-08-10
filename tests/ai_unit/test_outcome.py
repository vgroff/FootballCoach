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

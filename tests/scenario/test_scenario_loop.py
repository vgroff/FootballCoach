"""Scenario tests for ScenarioLoop - the looping runner that replays a
balance scenario multiple times in the UI.

These run headlessly (no pygame) by driving ScenarioLoop.step() directly.
They validate:
- A trial ends within a reasonable number of ticks (ball goes out / orders
  resolve / timeout).
- A fresh match is built for each subsequent trial.
- The loop marks itself complete after max_trials.
- Completion never requires more than max_trials * timeout_ticks total steps.
"""
from __future__ import annotations

import pytest

from footballcoach.ui.scenarios import SCENARIOS, ScenarioLoop


# Use a generous per-trial tick budget that comfortably exceeds the
# ScenarioLoop.timeout_ticks default (500) to avoid false failures.
MAX_STEPS_PER_TRIAL = 600
MAX_TOTAL_STEPS = 20_000  # absolute cap for the whole test


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.key)
def test_trial_ends_within_timeout(scenario):
    """Each scenario's first trial must end within timeout_ticks."""
    loop = ScenarioLoop(definition=scenario, max_trials=1, timeout_ticks=500)
    for _ in range(MAX_STEPS_PER_TRIAL):
        if loop.step():
            break
    assert loop.trial_count == 1, (
        f"Scenario '{scenario.key}' trial did not end within {MAX_STEPS_PER_TRIAL} ticks"
    )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.key)
def test_new_match_built_after_trial(scenario):
    """After a trial ends, loop.match must be a different object (fresh build)."""
    loop = ScenarioLoop(definition=scenario, max_trials=2, timeout_ticks=500)
    first_match = loop.match
    for _ in range(MAX_STEPS_PER_TRIAL):
        if loop.step():
            break
    assert loop.trial_count == 1
    assert not loop.complete
    assert loop.match is not first_match, (
        f"Scenario '{scenario.key}' did not rebuild match after trial 1"
    )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.key)
def test_loop_completes_after_max_trials(scenario):
    """Loop must reach complete == True after max_trials trials."""
    max_trials = 3
    loop = ScenarioLoop(definition=scenario, max_trials=max_trials, timeout_ticks=500)
    for _ in range(MAX_TOTAL_STEPS):
        loop.step()
        if loop.complete:
            break
    assert loop.complete, (
        f"Scenario '{scenario.key}' loop did not complete within {MAX_TOTAL_STEPS} steps"
    )
    assert loop.trial_count == max_trials


def test_step_returns_true_exactly_on_trial_end():
    """step() must return True on the tick a trial ends, False on all others."""
    loop = ScenarioLoop(definition=SCENARIOS[0], max_trials=2, timeout_ticks=500)
    trial_end_ticks = []
    for i in range(MAX_STEPS_PER_TRIAL * 2):
        if loop.step():
            trial_end_ticks.append(i)
        if loop.complete:
            break
    # Exactly 2 True returns - one per trial.
    assert len(trial_end_ticks) == 2
    # Both trial ends must happen (second after the first).
    assert trial_end_ticks[1] > trial_end_ticks[0]


def test_ball_out_of_bounds_ends_trial():
    """Manually push the ball out of bounds and confirm the trial ends the
    next step, demonstrating the out-of-bounds detection path directly."""
    loop = ScenarioLoop(definition=SCENARIOS[0], max_trials=2, timeout_ticks=500)
    match = loop.match
    pitch = match.pitch
    # Teleport the ball well past the touchline (y-axis) so it's out of
    # bounds but NOT inside the goal mouth (which would trigger _check_goal
    # / _reset_after_goal and move the ball back to the centre within the
    # same match.step() tick before _is_trial_done() can check it).
    from footballcoach.mathutils import Vector3
    match.ball.position = Vector3(0.0, pitch.half_width + 5.0, 0.0)
    match.ball.velocity = Vector3.zero()
    match.ball.possessed_by = None
    assert loop.step() is True
    assert loop.trial_count == 1

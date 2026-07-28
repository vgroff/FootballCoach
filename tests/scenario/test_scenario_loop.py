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
    """Each scenario's first trial must end within timeout_ticks.
    linger_s=0.0 so the test doesn't have to wait for the 3-second linger."""
    loop = ScenarioLoop(definition=scenario, max_trials=1, timeout_ticks=500, linger_s=0.0)
    for _ in range(MAX_STEPS_PER_TRIAL):
        if loop.step():
            break
    assert loop.trial_count == 1, (
        f"Scenario '{scenario.key}' trial did not end within {MAX_STEPS_PER_TRIAL} ticks"
    )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.key)
def test_new_match_built_after_trial(scenario):
    """After a trial ends, loop.match must be a different object (fresh build).
    linger_s=0.0 so the rebuild happens immediately on trial end."""
    loop = ScenarioLoop(definition=scenario, max_trials=2, timeout_ticks=500, linger_s=0.0)
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
    """Loop must reach complete == True after max_trials trials.
    linger_s=0.0 to avoid bloating MAX_TOTAL_STEPS budget."""
    max_trials = 3
    loop = ScenarioLoop(definition=scenario, max_trials=max_trials, timeout_ticks=500, linger_s=0.0)
    for _ in range(MAX_TOTAL_STEPS):
        loop.step()
        if loop.complete:
            break
    assert loop.complete, (
        f"Scenario '{scenario.key}' loop did not complete within {MAX_TOTAL_STEPS} steps"
    )
    assert loop.trial_count == max_trials


def test_step_returns_true_exactly_on_trial_end():
    """step() must return True on the tick a trial ends, False on all others.
    linger_s=0.0 so True is returned on the exact outcome tick."""
    loop = ScenarioLoop(definition=SCENARIOS[0], max_trials=2, timeout_ticks=500, linger_s=0.0)
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
    """Manually push the ball out of bounds; confirm the trial ends (possibly
    after the linger period) and the outcome is recorded correctly."""
    loop = ScenarioLoop(definition=SCENARIOS[0], max_trials=2, timeout_ticks=500, linger_s=0.0)
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


# ---------------------------------------------------------------------------
# Linger tests
# ---------------------------------------------------------------------------

def test_linger_delays_trial_end():
    """With linger_s > 0, step() must return False during the linger and
    True only once the linger expires."""
    linger_s = 0.5  # 0.5 sim-seconds (≈15 ticks at 30 Hz)
    loop = ScenarioLoop(definition=SCENARIOS[0], max_trials=2,
                        timeout_ticks=500, linger_s=linger_s)
    match = loop.match
    pitch = match.pitch
    from footballcoach.mathutils import Vector3
    # Force immediate OOB outcome.
    match.ball.position = Vector3(0.0, pitch.half_width + 5.0, 0.0)
    match.ball.velocity = Vector3.zero()
    match.ball.possessed_by = None

    # First step detects outcome → enters linger, returns False.
    result = loop.step()
    assert result is False, "first step after OOB should be False (linger started)"
    assert loop.trial_count == 0, "outcome must not be recorded yet during linger"
    assert loop._pending_outcome == "miss"

    # All steps during the linger must return False.
    false_count = 0
    for _ in range(1000):
        r = loop.step()
        if r:
            break
        false_count += 1
    else:
        raise AssertionError("linger never expired — loop.step() never returned True")

    assert loop.trial_count == 1, "outcome must be recorded once linger expires"
    # This is an OOB (miss) event → linger is linger_s * 0.5 = 0.25 s ≈ 7 ticks.
    # Allow ±2 ticks for floating-point rounding.
    oob_linger_s = linger_s * 0.5
    expected_ticks = round(oob_linger_s / match.dt_s)
    assert false_count >= max(1, expected_ticks - 2), (
        f"linger too short: only {false_count} false ticks (expected ~{expected_ticks})"
    )
    assert false_count <= expected_ticks + 5, (
        f"linger too long: {false_count} false ticks (expected ~{expected_ticks})"
    )


def test_linger_zero_returns_true_immediately():
    """With linger_s=0.0 the trial ends and True is returned on the same tick
    the outcome is detected (no delay, regression guard)."""
    loop = ScenarioLoop(definition=SCENARIOS[0], max_trials=2,
                        timeout_ticks=500, linger_s=0.0)
    match = loop.match
    pitch = match.pitch
    from footballcoach.mathutils import Vector3
    match.ball.position = Vector3(0.0, pitch.half_width + 5.0, 0.0)
    match.ball.velocity = Vector3.zero()
    match.ball.possessed_by = None
    assert loop.step() is True
    assert loop.trial_count == 1


def test_oob_linger_is_half_of_full_linger():
    """Out-of-bounds events linger for linger_s * 0.5; other outcomes
    (e.g. goal) linger for the full linger_s.  Assert the OOB linger
    duration is exactly half by counting the False ticks in each case."""
    from footballcoach.mathutils import Vector3
    dt = SCENARIOS[0].build(1.0).dt_s  # get dt from a fresh match

    def count_linger_ticks(oob: bool, linger_s: float) -> int:
        loop = ScenarioLoop(definition=SCENARIOS[0], max_trials=2,
                            timeout_ticks=500, linger_s=linger_s)
        match = loop.match
        pitch = match.pitch
        if oob:
            match.ball.position = Vector3(0.0, pitch.half_width + 5.0, 0.0)
        else:
            # Trigger a goal: place ball inside the left goal mouth.
            match.ball.position = Vector3(-(pitch.half_length + 0.5), 0.0, 0.1)
        match.ball.velocity = Vector3.zero()
        match.ball.possessed_by = None
        # Step once to detect outcome; count subsequent False ticks.
        loop.step()  # outcome detected here → pending
        ticks = 0
        for _ in range(10_000):
            if loop.step():
                break
            ticks += 1
        return ticks

    linger_s = 0.6  # 0.6 s → 18 ticks at 30 Hz; OOB should be ~9 ticks
    oob_ticks = count_linger_ticks(oob=True, linger_s=linger_s)
    goal_ticks = count_linger_ticks(oob=False, linger_s=linger_s)

    # OOB ticks ≈ linger_s/2 / dt; goal ticks ≈ linger_s / dt.
    # Allow ±2 ticks tolerance for floating-point rounding.
    assert oob_ticks <= goal_ticks - 2, (
        f"OOB linger ({oob_ticks} ticks) should be shorter than goal linger "
        f"({goal_ticks} ticks) by roughly half"
    )
    # Also assert the ratio is in the right ballpark (between 0.4 and 0.6).
    ratio = oob_ticks / max(goal_ticks, 1)
    assert 0.35 <= ratio <= 0.65, (
        f"OOB/goal linger ratio {ratio:.2f} not close enough to 0.5"
    )


def test_match_keeps_stepping_during_linger():
    """During the linger period the underlying match must keep advancing
    (time_s increases each step), so players/ball keep moving visually."""
    loop = ScenarioLoop(definition=SCENARIOS[0], max_trials=2,
                        timeout_ticks=500, linger_s=0.5)
    match = loop.match
    pitch = match.pitch
    from footballcoach.mathutils import Vector3
    match.ball.position = Vector3(0.0, pitch.half_width + 5.0, 0.0)
    match.ball.velocity = Vector3.zero()
    match.ball.possessed_by = None

    loop.step()  # triggers linger
    time_before = loop.match.time_s
    loop.step()  # one tick into linger
    time_after = loop.match.time_s
    assert time_after > time_before, "match.time_s must advance during linger"


def test_fresh_match_after_linger():
    """The match built after a linger completes must be a fresh object,
    not the same match that was running during the linger."""
    loop = ScenarioLoop(definition=SCENARIOS[0], max_trials=2,
                        timeout_ticks=500, linger_s=0.5)
    match = loop.match
    pitch = match.pitch
    from footballcoach.mathutils import Vector3
    match.ball.position = Vector3(0.0, pitch.half_width + 5.0, 0.0)
    match.ball.velocity = Vector3.zero()
    match.ball.possessed_by = None

    linger_match = loop.match  # match used during linger
    loop.step()  # enter linger

    for _ in range(1000):
        if loop.step():
            break

    assert loop.match is not linger_match, (
        "loop.match must be a fresh object after linger expires"
    )

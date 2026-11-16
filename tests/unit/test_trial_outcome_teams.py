"""detect_trial_outcome: only the defence winning the ball ends a trial, decided
by team (keeper = "saved", outfield = "dispossessed"); a teammate of the initial
carrier receiving it does not, unless the scenario opts in ("received")."""
from __future__ import annotations

import pytest

from footballcoach.ai.env.outcome import RAW_OUTCOME_KEYS, detect_trial_outcome
from footballcoach.mathutils import Vector3
from footballcoach.ui.scenarios import SCENARIOS, build_2v2_scenario


def _detect(m, **kw):
    return detect_trial_outcome(
        m, initial_scoreboard=(m.scoreboard.left_goals, m.scoreboard.right_goals),
        initial_carrier_id="atk_a", ball_released=True, box_possession_terminal=False,
        trial_tick=1, timeout_ticks=10_000, **kw)[0]


@pytest.fixture
def m():
    m = build_2v2_scenario()
    m.ball.position = Vector3(0.0, 0.0, m.ball.radius_m)    # well inside the pitch
    return m


def _give(m, pid):
    m.ball.possessed_by = pid


def test_teammate_reception_does_not_end_the_trial(m):
    _give(m, "atk_b")
    assert _detect(m) is None


def test_drills_can_end_on_reception(m):
    _give(m, "atk_b")
    assert _detect(m, teammate_reception_ends_trial=True) == "received"


@pytest.mark.parametrize("holder,expected", [("keeper", "saved"), ("defender", "dispossessed")])
def test_the_defence_winning_the_ball_ends_it(m, holder, expected):
    _give(m, holder)
    assert _detect(m) == expected
    assert _detect(m, teammate_reception_ends_trial=True) == expected


def test_a_loss_after_a_pass_is_still_caught(m):
    _give(m, "atk_b")
    assert _detect(m) is None
    _give(m, "defender")
    assert _detect(m) == "dispossessed"


def test_a_holder_not_in_the_match_raises(m):
    _give(m, "nobody")
    with pytest.raises(KeyError):
        _detect(m)


def test_received_is_a_known_outcome_and_only_the_pass_drill_opts_in():
    assert "received" in RAW_OUTCOME_KEYS
    assert {d.key for d in SCENARIOS if d.teammate_reception_ends_trial} == {"pass"}

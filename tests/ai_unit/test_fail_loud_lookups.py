"""Player lookups that used to swallow KeyError/AttributeError and return a made-up
value (ScenarioEnv's reward distance helpers returned 1.0 / 0.0, the BC label
helpers returned an invalid label) now raise. A counting probe over the env trace
and the AI test suite on 2026-11-15 found no caller that ever hit those fallbacks,
so this only changes what happens on a real bug: an error instead of a quietly
wrong reward or label."""
from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest

from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.bc import phase1_labels, phase1_labels_from_teacher
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario


@pytest.fixture
def env() -> ScenarioEnv:
    defn = ScenarioDefinition(key="lookup_test", label="lookup", description="lookup",
                              build=functools.partial(build_1v1_scenario))
    e = ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=1,
                    secondary_player_ids=[], max_episode_s=5.0)
    e.reset(seed=3)
    return e


def test_ball_dist_for_player_is_the_2d_distance(env):
    m = env._loop.match
    tr = m.player_by_id("trainee")
    m.ball.position = Vector3(tr.position.x + 3.0, tr.position.y + 4.0, m.ball.radius_m)
    assert env._ball_dist_for_player("trainee", m) == pytest.approx(5.0)


def test_ball_dist_to_opponent_box_is_zero_inside_and_positive_outside(env):
    m = env._loop.match
    tr = m.player_by_id("trainee")
    attack_x = m.pitch.half_length - 5.0 if tr.team == Team.LEFT else -(m.pitch.half_length - 5.0)
    m.ball.position = Vector3(attack_x, 0.0, m.ball.radius_m)
    assert env._ball_dist_to_opponent_box("trainee", m) == 0.0
    m.ball.position = Vector3(0.0, 0.0, m.ball.radius_m)
    expected = m.pitch.half_length - m.pitch.box_length_m
    assert env._ball_dist_to_opponent_box("trainee", m) == pytest.approx(expected)


@pytest.mark.parametrize("helper", ["_ball_dist_for_player", "_ball_dist_to_opponent_box"])
def test_distance_helpers_raise_on_unknown_player(env, helper):
    with pytest.raises(KeyError):
        getattr(env, helper)("nobody", env._loop.match)


def test_phase1_labels_invalid_without_a_match():
    label = phase1_labels(SimpleNamespace(match=None, trainee_player_id="trainee"))
    assert not label.valid


def test_phase1_labels_raises_on_unknown_player(env):
    with pytest.raises(KeyError):
        phase1_labels(env, "nobody")


def test_phase1_labels_from_teacher_raises_on_unknown_player(env):
    with pytest.raises(KeyError):
        phase1_labels_from_teacher(env, teacher_trainer=None, player_id="nobody")

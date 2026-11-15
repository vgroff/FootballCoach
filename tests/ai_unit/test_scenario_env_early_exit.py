"""ScenarioEnv.step()'s early exit: once the trainee is holding the ball inside the
opponent box, step() stops ticking (further ticks would only add spurious negative
box-distance progress).

The exit reads possession as of the end of the PREVIOUS tick, carried across step()
calls in ``_trainee_had_possession_last_step`` and cleared by reset(). These tests
pin that timing exactly: it was the one live output of a per-tick possession
state machine removed on 2026-11-15, and they pass unchanged on the code before
and after that removal.
"""
from __future__ import annotations

import functools

import pytest

from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3
from footballcoach.rules_ai import StopWhenIdleAI
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario


def _env(seed: int = 5) -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="early_exit_test", label="early exit", description="early exit",
        build=functools.partial(build_1v1_scenario, opponent_rules_prob=0.0, opponent_immobile_prob=1.0),
    )
    env = ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=1,
                      secondary_player_ids=[], max_episode_s=30.0)
    env.reset(seed=seed)
    return env


def _hold_ball(env: ScenarioEnv, *, in_opponent_box: bool) -> None:
    """Stationary trainee holding a stationary ball, either well inside the opponent
    box or at the centre spot; the opponent is parked far away and stationary."""
    m = env._loop.match
    tr = m.player_by_id("trainee")
    op = m.player_by_id("opponent")
    attack_sign = -1.0 if tr.team == Team.RIGHT else 1.0   # the box step() checks is left=(team == RIGHT)
    x = attack_sign * (m.pitch.half_length - 5.0) if in_opponent_box else 0.0
    for p in (tr, op):
        p.ai = StopWhenIdleAI()
        p.current_order = None
        p.velocity = Vector3.zero()
    tr.position = Vector3(x, 0.0, 0.0)
    op.position = Vector3(-attack_sign * 40.0, 25.0, 0.0)
    m.ball.position = Vector3(x, 0.0, m.ball.radius_m)
    m.ball.velocity = Vector3.zero()
    m._set_possession("trainee")
    m._sync_possessed_ball()
    assert m.pitch.is_in_box(m.ball.position, left=(tr.team == Team.RIGHT)) == in_opponent_box


@pytest.mark.parametrize("seed", [5, 6])
def test_first_step_after_reset_exits_on_the_second_tick(seed):
    # Tick 1 sees the flag reset() cleared (False); tick 2 sees tick 1's possession.
    env = _env(seed)
    _hold_ball(env, in_opponent_box=True)
    assert env._ticks_per_decision > 2, "test needs a decision interval longer than the exit point"
    _, _, _, info = env.step()
    assert env._loop.match.ball.possessed_by == "trainee"
    assert info.ticks_elapsed == 2


def test_no_early_exit_outside_the_box():
    env = _env()
    _hold_ball(env, in_opponent_box=False)
    _, _, done, info = env.step()
    assert not done
    assert info.ticks_elapsed == env._ticks_per_decision


def test_possession_carried_over_from_the_previous_step_exits_on_the_first_tick():
    # Step 1 ends with the trainee holding the ball (outside the box). Teleporting the
    # held ball into the box before step 2: tick 1 already sees the carried-over flag.
    env = _env()
    _hold_ball(env, in_opponent_box=False)
    env.step()
    _hold_ball(env, in_opponent_box=True)
    before = env._episode_ticks
    _, _, _, info = env.step()
    assert info.ticks_elapsed - before == 1


def test_reset_clears_the_carried_over_flag():
    env = _env()
    _hold_ball(env, in_opponent_box=False)
    env.step()
    env.reset(seed=5)
    _hold_ball(env, in_opponent_box=True)
    _, _, _, info = env.step()
    assert info.ticks_elapsed == 2

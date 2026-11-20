"""SaveOrder options (orders.json goalkeeper_ai.save_*): the position snap can
be switched off (no teleport onto the save point), and a settle radius stops
the keeper, facing the ball, once he is close enough to collect it. Defaults
are the old keeper (snap on, no settle)."""
from __future__ import annotations

import pytest

from footballcoach.ai.curriculum.envs import build_role_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.engine.goalkeeping import keeper_save_target
from footballcoach.engine.movement import SpeedMode
from footballcoach.mathutils import Vector3
from footballcoach.orders import SaveOrder


def _shot_match():
    env = build_role_env(PHASES_BY_ID[2], controller_weights={r: {"rules": 1} for r in ("attacker", "defender", "goalkeeper")})
    env.scenario_kwargs = {**env.scenario_kwargs, "no_keeper_prob": 0.0}
    env.reset(seed=5)
    m = env.match
    gk = m.player_by_id("keeper")
    goal_x = m.pitch.half_length if gk.team.name == "RIGHT" else -m.pitch.half_length
    sign = 1.0 if goal_x > 0 else -1.0
    m.ball.possessed_by = None
    m.ball.position = Vector3(goal_x - sign * 18.0, 1.0, 0.3)
    m.ball.velocity = Vector3(sign * 18.0, 0.2, 0.0)
    return m, gk


def _near_target(m, gk, dist):
    t = keeper_save_target(gk, m)
    gk.position = Vector3(t.x, t.y + dist, gk.position.z)
    gk.velocity = Vector3.zero()
    return t


def test_default_keeps_the_snap():
    m, gk = _shot_match()
    t = _near_target(m, gk, 0.1)
    SaveOrder().execute(gk, m, 0.04)
    assert gk.position.xy().distance_to(keeper_save_target(gk, m).xy()) == pytest.approx(0.0, abs=1e-6)
    assert gk.desired_speed_mode is SpeedMode.STANDSTILL
    assert t is not None


def test_no_snap_never_sets_the_position():
    m, gk = _shot_match()
    _near_target(m, gk, 0.1)
    before = (gk.position.x, gk.position.y)
    SaveOrder(snap=False).execute(gk, m, 0.04)
    assert (gk.position.x, gk.position.y) == before


def test_settle_radius_stops_facing_the_ball():
    m, gk = _shot_match()
    _near_target(m, gk, 0.4)
    SaveOrder(snap=False, settle_radius_m=0.5).execute(gk, m, 0.04)
    assert gk.desired_speed_mode is SpeedMode.STANDSTILL
    to_ball = m.ball.position - gk.position
    d = gk.desired_direction
    assert d.x * to_ball.x + d.y * to_ball.y > 0.99 * to_ball.length_xy()


def test_outside_the_settle_radius_it_still_moves():
    m, gk = _shot_match()
    _near_target(m, gk, 2.0)
    SaveOrder(snap=False, settle_radius_m=0.5).execute(gk, m, 0.04)
    assert gk.desired_speed_mode is not SpeedMode.STANDSTILL

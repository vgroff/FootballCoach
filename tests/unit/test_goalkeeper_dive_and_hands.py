"""Keeper nerfs/buffs added 2026-11-19 (all off by default in physics.json):
gated dive (movement.goalkeeper_dive_gating), the keeper tackler's longer
post-tackle inactivity (tackling.goalkeeper_tackle_cooldown_multiplier) and
the keeper's larger pickup radius in his own box
(ball_pickup.goalkeeper_pickup_radius_multiplier)."""
from __future__ import annotations

import dataclasses

import pytest

from footballcoach.engine.movement import MovementParams, SpeedMode, effective_top_speed
from footballcoach.engine.tackling import TackleResult, TacklingParams, apply_tackle_result
from footballcoach.entities.player import PlayerState
from footballcoach.mathutils import Vector3
from footballcoach.ui.scenarios import build_1v2_scenario

DT = 1.0 / 30.0


def _match(**movement):
    m = build_1v2_scenario(seed=1)
    m.movement_params = dataclasses.replace(m.movement_params, **movement)
    return m


def _gated(**kw):
    return _match(goalkeeper_dive_gating=True, goalkeeper_dive_duration_s=1.5, goalkeeper_dive_recovery_s=1.5, **kw)


def _run(m, gk, mode, seconds):
    out = []
    for _ in range(round(seconds / DT)):
        out.append(m._gate_goalkeeper_dive(gk, mode, DT))
    return out


def test_a_dive_lasts_its_window_then_forces_a_jog_for_the_recovery():
    m = _gated(); gk = m.player_by_id("keeper")
    dive = _run(m, gk, SpeedMode.SPRINT, 1.5)
    assert all(mode is SpeedMode.SPRINT and boost for mode, boost in dive)
    assert gk.gk_dive_remaining_s == 0.0 and gk.gk_dive_recovery_s == pytest.approx(1.5)
    recovering = _run(m, gk, SpeedMode.SPRINT, 1.5)          # asks to sprint the whole time
    assert all(mode is SpeedMode.JOG and not boost for mode, boost in recovering)
    assert gk.gk_dive_recovery_s == pytest.approx(0.0, abs=1e-9)   # the countdown kept running
    mode, boost = m._gate_goalkeeper_dive(gk, SpeedMode.SPRINT, DT)
    assert mode is SpeedMode.SPRINT and boost                  # can dive again


def test_stopping_early_ends_the_dive_and_starts_the_recovery():
    m = _gated(); gk = m.player_by_id("keeper")
    _run(m, gk, SpeedMode.SPRINT, 0.5)
    assert gk.gk_dive_remaining_s == pytest.approx(1.0, abs=DT)
    mode, boost = m._gate_goalkeeper_dive(gk, SpeedMode.JOG, DT)
    assert mode is SpeedMode.JOG and not boost
    assert gk.gk_dive_remaining_s == 0.0 and gk.gk_dive_recovery_s == pytest.approx(1.5)


def test_no_boost_outside_a_dive_and_no_gating_by_default():
    m = _gated(); gk = m.player_by_id("keeper")
    assert m._gate_goalkeeper_dive(gk, SpeedMode.JOG, DT) == (SpeedMode.JOG, False)
    assert MovementParams.from_config().goalkeeper_dive_gating is False


def test_the_unboosted_keeper_moves_like_an_outfielder():
    m = _gated(); gk = m.player_by_id("keeper"); mp = m.movement_params
    a = gk.attributes
    boosted = effective_top_speed(mp, a.top_speed, gk.stamina, False, a.ball_control, True)
    plain = effective_top_speed(mp, a.top_speed, gk.stamina, False, a.ball_control, False)
    assert boosted == pytest.approx(plain * mp.goalkeeper_speed_multiplier)
    from footballcoach.engine.movement import step_player_towards
    gk.velocity = Vector3(0, 0, 0)
    for _ in range(300):   # 10 s of sprinting without the boost
        step_player_towards(gk, Vector3(-1, 0, 0), SpeedMode.SPRINT, DT, mp, False, goalkeeper_boost=False)
    assert gk.velocity.length_xy() == pytest.approx(plain, rel=0.02)


def test_a_keeper_tackler_sits_out_longer_the_tacklee_does_not():
    m = build_1v2_scenario(seed=2)
    gk, att = m.player_by_id("keeper"), m.player_by_id("attacker")
    params = dataclasses.replace(TacklingParams.from_config(), goalkeeper_tackle_cooldown_multiplier=2.0)
    apply_tackle_result(TackleResult(tackler_won=True, tackler_speed_mult=1.0, tacklee_speed_mult=1.0), gk, att, params)
    assert gk.state is PlayerState.INACTIVE_TACKLED and att.state is PlayerState.INACTIVE_TACKLED
    assert gk.state_timer_s == pytest.approx(2 * params.tackle_cooldown_s)
    assert att.state_timer_s == pytest.approx(params.tackle_cooldown_s)
    defender = m.player_by_id("defender")
    apply_tackle_result(TackleResult(tackler_won=True, tackler_speed_mult=1.0, tacklee_speed_mult=1.0), defender, att, params)
    assert defender.state_timer_s == pytest.approx(params.tackle_cooldown_s)


def test_the_keeper_gets_a_larger_pickup_radius_only_in_his_own_box():
    m = build_1v2_scenario(seed=3)
    m.ball_pickup_params = dataclasses.replace(m.ball_pickup_params, goalkeeper_pickup_radius_multiplier=1.5)
    base = m.ball_pickup_params.pickup_radius_m
    gk, dfn = m.player_by_id("keeper"), m.player_by_id("defender")
    assert m._pickup_params_for(gk).pickup_radius_m == pytest.approx(1.5 * base)   # parked on his line
    assert m._pickup_params_for(dfn).pickup_radius_m == pytest.approx(base)
    gk.position = Vector3(0.0, 0.0, 0.0)                                              # halfway line
    assert m._pickup_params_for(gk).pickup_radius_m == pytest.approx(base)

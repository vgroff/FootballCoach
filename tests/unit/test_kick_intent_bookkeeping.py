"""Every kick an order makes goes through the same bookkeeping
(Player._finish_kick): shots and passes now count as kicks, record
last_kick_*, fire on_kick once, and end a first touch -- as push-kicks and
KickOrder always did. (Before 2026-11-16 ShootOrder / PassOrder called the
engine's kick_ball / pass_ball directly and skipped all of it.)"""
from __future__ import annotations

import pytest

from footballcoach.entities.player import PlayerState
from footballcoach.mathutils import Vector3
from footballcoach.orders import PassOrder, ShootOrder
from footballcoach.ui.scenarios import build_2v2_scenario


def _setup():
    m = build_2v2_scenario()
    a = m.player_by_id("atk_a")
    for p in m.players:
        p.ai = None
        p.current_order = None
    a.position = Vector3(20.0, 0.0, 0.0)
    m.player_by_id("defender").position = Vector3(-30.0, 25.0, 0.0)
    m.player_by_id("keeper").position = Vector3(m.pitch.half_length - 0.5, 15.0, 0.0)
    m.ball.position = Vector3(20.3, 0.0, m.ball.radius_m)
    m._set_possession("atk_a")
    m._sync_possessed_ball()
    a.state, a.state_timer_s = PlayerState.CONTROLLING_BALL, 0.3
    return m, a


@pytest.mark.parametrize("make_order", [
    lambda m: ShootOrder(aim_point=m.pitch.right_goal_centre + Vector3(0, 2.0, 1.0), power_fraction=0.9,
                         compensate_for_run=False, chance_of_pausing=0.0),
    lambda m: PassOrder(target_position=m.player_by_id("atk_b").position),
])
def test_a_shot_or_pass_does_the_kick_bookkeeping(make_order):
    m, a = _setup()
    fired = []
    a.on_kick = lambda p: fired.append(p.player_id)
    a.current_order = make_order(m)
    before = a.kick_count
    m._process_orders(m.dt_s)
    assert m.ball.possessed_by is None and m.ball.velocity.length() > 1.0     # it really kicked
    assert a.kicked_this_tick and a.kick_count == before + 1
    assert a.last_kick_direction is not None and a.last_kick_power_fraction > 0.0
    assert fired == ["atk_a"]                                                 # on_kick exactly once
    assert a.state is PlayerState.ACTIVE and a.state_timer_s == 0.0           # first touch ended
    assert a.pending_kick is None


def test_a_chase_tackle_counts_like_an_armed_tackle():
    from footballcoach.orders import ChaseTackleOrder
    m = build_2v2_scenario()
    for p in m.players:
        p.ai, p.current_order = None, None
    a, d = m.player_by_id("atk_b"), m.player_by_id("defender")
    a.position, d.position = Vector3(10.0, 0.0, 0.0), Vector3(10.5, 0.0, 0.0)
    m.ball.position = Vector3(10.3, 0.0, m.ball.radius_m)
    m._set_possession("atk_b")
    m._sync_possessed_ball()
    tackles = []
    d.on_tackle = lambda *x: tackles.append(x)
    d.current_order = ChaseTackleOrder(target_player_id="atk_b")
    before = d.tackle_fire_count
    m._process_orders(m.dt_s)
    assert tackles and d.tackle_fire_count == before + 1 and d.pending_tackle is None

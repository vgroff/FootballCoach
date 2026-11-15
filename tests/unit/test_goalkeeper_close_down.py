"""StagedGoalkeeperAI closing down a ball carrier (orders.json goalkeeper_ai):
strict entry, committed stay with hysteresis, immediate return to goalkeeping
on a shot or when the defence wins the ball, and a cooldown before it can
commit again."""
from __future__ import annotations

import pytest

from footballcoach.config import load_orders_config
from footballcoach.mathutils import Vector3
from footballcoach.orders import GetPossessionOrder, MoveOrder, SaveOrder
from footballcoach.rules_ai import StagedGoalkeeperAI
from footballcoach.ui.scenarios import build_1v2_scenario

CFG = load_orders_config()["goalkeeper_ai"]
ENTRY, EXIT, COOLDOWN = CFG["close_down_entry_dist_m"], CFG["close_down_exit_dist_m"], CFG["close_down_cooldown_s"]


@pytest.fixture
def setup():
    m = build_1v2_scenario(seed=2)
    gk_player = m.player_by_id("keeper")
    gk = StagedGoalkeeperAI()
    gk_player.position = m.pitch.right_goal_centre
    gk_player.current_order = None
    # Defender well upfield (beaten) unless a test moves it goal-side.
    m.player_by_id("defender").position = Vector3(10.0, 20.0, 0.0)
    return m, gk, gk_player


def _carrier_at(m, dist_m, y=0.0, holder="attacker"):
    gc = m.pitch.right_goal_centre
    p = m.player_by_id(holder)
    p.position = Vector3(gc.x - dist_m, y, 0.0)
    m.ball.position = Vector3(gc.x - dist_m + 0.3, y, m.ball.radius_m)
    m.ball.velocity = Vector3.zero()
    m.ball.possessed_by = holder


def _decide(m, gk, gk_player, dt_s=0.0):
    m.time_s += dt_s
    gk.decide(gk_player, m, 0)
    return gk_player.current_order


def test_commits_when_the_last_line_and_the_carrier_is_close(setup):
    m, gk, p = setup
    _carrier_at(m, ENTRY - 1.0)
    assert isinstance(_decide(m, gk, p), GetPossessionOrder)


def test_no_entry_beyond_the_entry_distance(setup):
    m, gk, p = setup
    _carrier_at(m, ENTRY + 1.0)
    assert not isinstance(_decide(m, gk, p), GetPossessionOrder)


def test_no_entry_while_a_defender_is_goal_side(setup):
    m, gk, p = setup
    _carrier_at(m, ENTRY - 1.0)
    gc = m.pitch.right_goal_centre
    m.player_by_id("defender").position = Vector3(gc.x - 2.0, 1.0, 0.0)
    assert not isinstance(_decide(m, gk, p), GetPossessionOrder)


def test_no_entry_for_its_own_teammate_on_the_ball(setup):
    m, gk, p = setup
    _carrier_at(m, ENTRY - 1.0, holder="defender")
    assert not isinstance(_decide(m, gk, p), GetPossessionOrder)


def test_stays_committed_between_entry_and_exit_distance_and_ignores_a_recovering_defender(setup):
    m, gk, p = setup
    _carrier_at(m, ENTRY - 1.0)
    _decide(m, gk, p)
    _carrier_at(m, (ENTRY + EXIT) / 2)
    gc = m.pitch.right_goal_centre
    m.player_by_id("defender").position = Vector3(gc.x - 1.0, 2.0, 0.0)
    assert isinstance(_decide(m, gk, p, dt_s=0.1), GetPossessionOrder)


def test_returns_to_goal_once_the_carrier_is_beyond_the_exit_distance(setup):
    m, gk, p = setup
    _carrier_at(m, ENTRY - 1.0)
    _decide(m, gk, p)
    gc = m.pitch.right_goal_centre
    p.position = Vector3(gc.x - 4.0, 0.5, 0.0)     # it left its line while committed
    _carrier_at(m, EXIT + 1.0)
    order = _decide(m, gk, p, dt_s=0.1)
    assert isinstance(order, MoveOrder) and order.target_position == gc


def test_a_shot_switches_straight_to_save_duty(setup):
    m, gk, p = setup
    _carrier_at(m, ENTRY - 1.0)
    _decide(m, gk, p)
    m.ball.possessed_by = None
    m.ball.velocity = Vector3(15.0, 0.0, 0.0)   # loose, heading into the right goal
    assert isinstance(_decide(m, gk, p, dt_s=0.04), SaveOrder)


def test_loose_ball_not_heading_for_goal_is_still_chased_inside_the_exit_distance(setup):
    m, gk, p = setup
    _carrier_at(m, ENTRY - 1.0)
    _decide(m, gk, p)
    m.ball.possessed_by = None
    m.ball.velocity = Vector3(-2.0, 3.0, 0.0)    # knocked loose, away from goal
    assert isinstance(_decide(m, gk, p, dt_s=0.04), GetPossessionOrder)


def test_defence_winning_the_ball_ends_the_commitment(setup):
    m, gk, p = setup
    _carrier_at(m, ENTRY - 1.0)
    _decide(m, gk, p)
    m.ball.possessed_by = "defender"
    assert not isinstance(_decide(m, gk, p, dt_s=0.04), GetPossessionOrder)


def test_cooldown_blocks_an_immediate_recommit(setup):
    m, gk, p = setup
    _carrier_at(m, ENTRY - 1.0)
    _decide(m, gk, p)
    _carrier_at(m, EXIT + 1.0)
    _decide(m, gk, p, dt_s=0.1)                  # leaves
    _carrier_at(m, ENTRY - 1.0)
    assert not isinstance(_decide(m, gk, p, dt_s=COOLDOWN / 2), GetPossessionOrder)
    assert isinstance(_decide(m, gk, p, dt_s=COOLDOWN), GetPossessionOrder)


def test_switch_off_restores_the_old_keeper(setup, monkeypatch):
    m, gk, p = setup
    monkeypatch.setitem(load_orders_config()["goalkeeper_ai"], "close_down_carrier_in_box", False)
    _carrier_at(m, 1.0)
    assert not isinstance(_decide(m, gk, p), GetPossessionOrder)

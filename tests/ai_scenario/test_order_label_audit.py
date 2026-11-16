"""Audit: for every order class a rules AI can issue, the BC label
(ai/bc/order_labels.order_label)

1. leaves the real match untouched (every attribute of every player and the
   ball, the match rng, clock and game log), and
2. matches what the order then really does when it executes that tick: the
   movement intent, whether it kicks or arms a kick, the kick's direction,
   power and spin (direction exact: the random kick error is switched off
   with rng_reduction = 1), and whether it arms a tackle;
3. is labelled with the order's decision head (order_labels.DECISION_HEAD_OF_ORDER).

Each case builds a 2v2 state in which the order really does its thing, and
checks that it did (e.g. the push-kick case must kick)."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from footballcoach.ai.bc.order_labels import decision_head_of, order_label
from footballcoach.ai.ppo.bc import _I_GP_EXTRA, _I_HOLD, _I_MARK, _I_MOVE, _I_PASS, _I_SHOOT, _I_TACKLE
from footballcoach.engine.kicking import max_kick_speed_mps, running_power_multiplier
from footballcoach.engine.movement import SpeedMode, effective_top_speed
from footballcoach.mathutils import Vector3
from footballcoach.orders import (
    ChaseTackleOrder, GetPossessionOrder, JogOrder, KickOrder, MarkOrder, MoveOrder, PassOrder, SaveOrder,
    ShootOrder, StopOrder,
)
from footballcoach.ui.scenarios import build_2v2_scenario

_HEAD_COL = {"move": _I_MOVE, "get_possession_extra": _I_GP_EXTRA, "shoot": _I_SHOOT, "pass_": _I_PASS,
             "tackle": _I_TACKLE, "mark": _I_MARK, "hold_position": _I_HOLD}


def _match(ball_holder="atk_a"):
    m = build_2v2_scenario()
    m.rng_reduction = 1.0                    # no random kick error: directions compare exactly
    for p in m.players:
        p.ai, p.current_order = None, None
        p.velocity = Vector3.zero()
    place = {"atk_a": (0.0, 0.0), "atk_b": (15.0, 12.0), "defender": (-30.0, 25.0),
             "keeper": (m.pitch.half_length - 1.0, 0.0)}
    for pid, (x, y) in place.items():
        m.player_by_id(pid).position = Vector3(x, y, 0.0)
        m.player_by_id(pid).heading_rad = 0.0
    if ball_holder is not None:
        h = m.player_by_id(ball_holder)
        m.ball.position = Vector3(h.position.x + 0.3, h.position.y, m.ball.radius_m)
        m._set_possession(ball_holder)
        m._sync_possessed_ball()
    return m


def _carry(m, pid, x, y, vx=6.0):
    p = m.player_by_id(pid)
    p.position, p.velocity, p.heading_rad = Vector3(x, y, 0.0), Vector3(vx, 0.0, 0.0), 0.0
    m.ball.position = Vector3(x + 0.3, y, m.ball.radius_m)
    m._set_possession(pid)
    m._sync_possessed_ball()


def case_move():
    m = _match(ball_holder="atk_b")
    return m, "atk_a", MoveOrder(target_position=Vector3(10.0, 5.0, 0.0)), {}


def case_move_push_kick():
    m = _match(ball_holder=None)
    _carry(m, "atk_a", -45.0, 0.0)
    m.player_by_id("defender").position = Vector3(45.0, -30.0, 0.0)   # nobody can reach the target first
    return m, "atk_a", MoveOrder(target_position=Vector3(-10.0, 0.0, 0.0), push_kick_enabled=True), {"kicks": True}


def case_get_possession_loose():
    m = _match(ball_holder=None)
    m.ball.position, m.ball.velocity = Vector3(5.0, 3.0, m.ball.radius_m), Vector3.zero()
    m.ball.possessed_by = None
    return m, "atk_a", GetPossessionOrder(), {}


def case_get_possession_tackle():
    m = _match(ball_holder=None)
    _carry(m, "atk_b", 10.0, 0.0, vx=2.0)
    d = m.player_by_id("defender")
    d.position, d.heading_rad = Vector3(11.0, 0.0, 0.0), math.pi
    return m, "defender", GetPossessionOrder(), {}


def case_get_possession_armed():
    # Running onto a loose ball with a push-kick target: arms the first touch.
    m = _match(ball_holder=None)
    a = m.player_by_id("atk_a")
    a.position, a.velocity, a.heading_rad = Vector3(-45.0, 0.0, 0.0), Vector3(6.0, 0.0, 0.0), 0.0
    m.player_by_id("defender").position = Vector3(45.0, -30.0, 0.0)
    m.ball.position, m.ball.velocity = Vector3(-44.0, 0.0, m.ball.radius_m), Vector3.zero()
    m.ball.possessed_by = None
    return m, "atk_a", GetPossessionOrder(push_kick_enabled=True, target_position=Vector3(-10.0, 0.0, 0.0)),         {"arms": True}


def case_shoot():
    m = _match(ball_holder=None)
    _carry(m, "atk_a", 35.0, 0.0)
    aim = m.pitch.right_goal_centre + Vector3(0.0, 2.0, 1.0)
    return m, "atk_a", ShootOrder(aim_point=aim, power_fraction=0.9, compensate_for_run=False,
                                  chance_of_pausing=0.0), {"kicks": True}


def case_shoot_paused():
    m = _match(ball_holder=None)
    _carry(m, "atk_a", 35.0, 0.0)
    m.player_by_id("defender").position = Vector3(40.0, 0.3, 0.0)
    aim = m.pitch.right_goal_centre
    return m, "atk_a", ShootOrder(aim_point=aim, power_fraction=0.9, chance_of_pausing=1.0), {"becomes": MoveOrder}


def case_pass():
    m = _match()
    return m, "atk_a", PassOrder(target_position=m.player_by_id("atk_b").position), {"kicks": True}


def case_kick():
    m = _match()
    return m, "atk_a", KickOrder(aim_point=Vector3(30.0, 8.0, 0.5), power_fraction=0.7, spin=Vector3.zero()), \
        {"kicks": True}


def case_chase_tackle():
    m = _match(ball_holder=None)
    _carry(m, "atk_b", 10.0, 0.0, vx=2.0)
    d = m.player_by_id("defender")
    d.position, d.heading_rad = Vector3(11.0, 0.0, 0.0), math.pi
    return m, "defender", ChaseTackleOrder(target_player_id="atk_b"), {}


def case_chase_tackle_contact():
    # Touching: the order tackles. Before tackle intents its counterfactual
    # resolved a REAL tackle on atk_b (its velocity / state changed).
    m = _match(ball_holder=None)
    _carry(m, "atk_b", 10.0, 0.0, vx=2.0)
    d = m.player_by_id("defender")
    d.position, d.heading_rad = Vector3(10.5, 0.0, 0.0), math.pi
    return m, "defender", ChaseTackleOrder(target_player_id="atk_b"), {"tackles": True}


def case_mark():
    m = _match(ball_holder="atk_b")
    return m, "defender", MarkOrder(target_player_id="atk_a"), {}


def case_stop():
    m = _match(ball_holder="atk_b")
    m.player_by_id("atk_a").velocity = Vector3(4.0, 0.0, 0.0)
    return m, "atk_a", StopOrder(), {}


def case_save():
    m = _match(ball_holder=None)
    m.ball.position = Vector3(m.pitch.half_length - 12.0, 1.0, m.ball.radius_m)
    m.ball.velocity = Vector3(20.0, 0.5, 0.0)
    m.ball.possessed_by = None
    return m, "keeper", SaveOrder(auto_sprint=True), {}


def case_jog():
    m = _match(ball_holder="atk_b")
    return m, "atk_a", JogOrder(direction=Vector3(1.0, 0.0, 0.0)), {}


CASES = {name[5:]: fn for name, fn in globals().items() if name.startswith("case_")}


def _world(m):
    snap = {p.player_id: {k: repr(v) for k, v in vars(p).items() if not callable(v) and k != "ai"} for p in m.players}
    snap["ball"] = {k: repr(v) for k, v in vars(m.ball).items()}
    snap["match"] = (m.rng.getstate(), m.time_s, m.tick_index)
    return snap


@pytest.mark.parametrize("name", sorted(CASES))
def test_order_label_is_invisible_and_matches_the_real_execution(name):
    m, pid, order, expect = CASES[name]()
    player = m.player_by_id(pid)
    real_order = copy.deepcopy(order)           # the label run advances its own order's state
    log = []
    m.log_callback = lambda *a: log.append(a)

    before = _world(m)
    label, cf = order_label(player, m, order)
    assert _world(m) == before, "the label changed the match"
    assert log == [], "the label wrote to the game log"

    labelled_as = cf["order_after"] if cf["order_after"] is not None else order
    if "becomes" in expect:
        assert isinstance(labelled_as, expect["becomes"])
    arr = label.to_array()
    head = decision_head_of(labelled_as)
    assert head is not None and arr[_HEAD_COL[head]] == 1.0
    assert sum(arr[c] for c in _HEAD_COL.values()) == 1.0     # exactly one decision head

    # --- now run the same order for real, this tick ---
    player.current_order = real_order
    tackles = []
    player.on_tackle = lambda *a: tackles.append(a)
    m._process_orders(m.dt_s)

    want_dir = player.desired_direction
    if label.move_direction is None:
        assert want_dir.length_xy() < 1e-6 or player.desired_speed_mode is SpeedMode.STANDSTILL
    else:
        assert np.allclose(label.move_direction, [want_dir.x, want_dir.y], atol=1e-6)
    assert label.sprint == (1.0 if player.desired_speed_mode is SpeedMode.SPRINT else 0.0)
    assert label.exec_move == (0.0 if player.desired_speed_mode is SpeedMode.STANDSTILL else 1.0)
    tackled = bool(player.tackle_armed or tackles)
    assert label.tackle_attempt == (1.0 if tackled else 0.0)
    if expect.get("tackles"):
        assert tackles, "the case should make the order tackle"

    kicked = player.kicked_this_tick
    armed = player.kick_armed and player.kick_armed_direction is not None
    assert label.kick_this_tick == (1.0 if (kicked or armed) else 0.0)
    if expect.get("kicks"):
        assert kicked, "the case should make the order kick"
    if expect.get("arms"):
        assert armed and not kicked, "the case should make the order arm a kick"
    if kicked:
        v = m.ball.velocity
        d = v * (1.0 / v.length())
        assert np.allclose(label.kick_direction, [d.x, d.y, d.z], atol=1e-6)
        top = effective_top_speed(m.movement_params, player.attributes.top_speed, player.stamina,
                                  has_ball=True, ball_control_attr=player.attributes.ball_control)
        run_mult = running_power_multiplier(m.kicking_params.running_power_coefficient, player.velocity,
                                            Vector3(*label.kick_direction), top)
        expected_speed = max_kick_speed_mps(m.kicking_params, player.attributes.kick_power) * \
            label.kick_power_fraction * run_mult
        assert v.length() == pytest.approx(expected_speed, rel=1e-6)
        assert label.kick_power_fraction == pytest.approx(player.last_kick_power_fraction, rel=1e-6)
    elif armed:
        ad = player.kick_armed_direction
        assert np.allclose(label.kick_direction, [ad.x, ad.y, ad.z], atol=1e-6)
        assert label.kick_power_fraction == pytest.approx(player.kick_armed_power_fraction)


def test_phase1_labels_equal_the_general_labeller_on_phase1_orders():
    """phase1_labels_for_player asks Phase1RulesAI for its order and labels it;
    order_label on that same order must give the identical label vector."""
    import random

    import torch

    from footballcoach.ai.curriculum.envs import build_env
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.ppo.bc import phase1_labels_for_player
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    from footballcoach.rules_ai import Phase1RulesAI

    torch.manual_seed(0)
    trainer = PPOTrainer.from_config(device=torch.device("cpu"))
    env = build_env(PHASES_BY_ID[1])
    env.sample_action_fn = trainer._sample_action
    compared = 0
    for seed in range(4):
        env.rng = random.Random(seed)
        env.reset(seed=seed)
        done = False
        while not done:
            m = env._loop.match
            p = m.player_by_id("trainee")
            a = phase1_labels_for_player(p, m).to_array()
            saved = p.current_order
            p.current_order = None
            Phase1RulesAI(decision_interval_ticks=1).act(p, m, trial_tick=0)
            order, p.current_order = p.current_order, saved
            b = order_label(p, m, order)[0].to_array()
            assert np.array_equal(a, b), (seed, type(order).__name__)
            compared += 1
            _, _, done, _ = env.step()
    assert compared > 100

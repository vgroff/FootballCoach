"""ai/bc/role_experts.RulesExpert: the rules AI's memory (shadow order and the
AI's own fields) follows the episode as the NEURAL player actually plays it,
and labelling never changes the match. The neural player's decisions are
scripted by setting its last gating's SelectedAction."""
from __future__ import annotations

import math

import pytest

from footballcoach.ai.action.gating import SelectedAction
from footballcoach.ai.bc.role_experts import RulesExpert
from footballcoach.config import load_orders_config
from footballcoach.mathutils import Vector3
from footballcoach.orders import GetPossessionOrder, MoveOrder
from footballcoach.ui.scenarios import build_1v2_scenario

GK = load_orders_config()["goalkeeper_ai"]


class _Gating:
    def __init__(self, selected):
        self.selected = selected


class _NeuralStub:
    """Stands in for NeuralPlayerAI: only its last decision is read."""
    def __init__(self):
        self._last_gating = None

    def decided(self, selected):
        self._last_gating = _Gating(selected)


def _world(m):
    snap = {p.player_id: {k: repr(v) for k, v in vars(p).items() if not callable(v) and k != "ai"} for p in m.players}
    snap["ball"] = {k: repr(v) for k, v in vars(m.ball).items()}
    snap["rng"] = m.rng.getstate()
    return snap


def _label(expert, player, m):
    before = _world(m)
    lab = expert.label(player, m)
    assert _world(m) == before, "labelling changed the match"
    return lab


# --------------------------------------------------------------------- keeper

@pytest.fixture
def keeper_world():
    m = build_1v2_scenario(seed=2)
    k = m.player_by_id("keeper")
    rules_ai, opening = k.ai, k.current_order          # what the scenario built
    neural = _NeuralStub()
    k.ai, k.current_order = neural, None               # the neural keeper replaces it
    k.position, k.velocity = m.pitch.right_goal_centre, Vector3.zero()
    m.player_by_id("defender").position = Vector3(10.0, 20.0, 0.0)   # beaten
    expert = RulesExpert(rules_ai, opening, k)
    return m, k, neural, expert


def _carrier_at(m, dist_m):
    gc = m.pitch.right_goal_centre
    a = m.player_by_id("attacker")
    a.position, a.velocity = Vector3(gc.x - dist_m, 0.0, 0.0), Vector3.zero()
    m.ball.position = Vector3(gc.x - dist_m + 0.3, 0.0, m.ball.radius_m)
    m.ball.velocity = Vector3.zero()
    m.ball.possessed_by = "attacker"


def test_keeper_not_left_committed_when_the_neural_keeper_does_not_go(keeper_world):
    m, k, neural, expert = keeper_world
    entry, exit_ = GK["close_down_entry_dist_m"], GK["close_down_exit_dist_m"]
    _carrier_at(m, entry - 2.0)
    neural.decided(SelectedAction.MOVE)
    assert _label(expert, k, m).get_possession_extra == 1.0          # the expert would go
    assert expert.core._closing_down
    # The neural keeper did not go, and the carrier is now too far to START
    # closing down but close enough that a COMMITTED keeper would keep going.
    _carrier_at(m, (entry + exit_) / 2)
    m.time_s += 0.25
    lab = _label(expert, k, m)
    assert lab.get_possession_extra == 0.0                            # not left committed
    assert not expert.core._closing_down
    assert expert.core._close_down_ended_s == float("-inf")           # and no cooldown: it never went
    _carrier_at(m, entry - 2.0)
    m.time_s += 0.25
    assert _label(expert, k, m).get_possession_extra == 1.0           # close again: go now


def test_keeper_cooldown_starts_when_the_neural_keeper_stops(keeper_world):
    m, k, neural, expert = keeper_world
    _carrier_at(m, GK["close_down_entry_dist_m"] - 2.0)
    neural.decided(SelectedAction.GET_POSSESSION)
    for _ in range(3):                                                # the neural keeper closes down
        assert _label(expert, k, m).get_possession_extra == 1.0
        m.time_s += 0.25
    neural.decided(SelectedAction.MOVE)                               # ... and stops
    stopped_at = m.time_s
    lab = _label(expert, k, m)
    assert expert.core._close_down_ended_s == stopped_at
    assert lab.get_possession_extra == 0.0                            # cooling down: no recommit
    m.time_s += GK["close_down_cooldown_s"] / 2
    assert _label(expert, k, m).get_possession_extra == 0.0
    m.time_s += GK["close_down_cooldown_s"]
    assert _label(expert, k, m).get_possession_extra == 1.0           # cooldown over


def test_keeper_counts_as_committed_when_the_neural_keeper_goes_on_its_own(keeper_world):
    m, k, neural, expert = keeper_world
    entry, exit_ = GK["close_down_entry_dist_m"], GK["close_down_exit_dist_m"]
    _carrier_at(m, (entry + exit_) / 2)                               # too far to START, close enough to KEEP
    neural.decided(SelectedAction.MOVE)
    assert _label(expert, k, m).get_possession_extra == 0.0           # the expert would not go
    m.time_s += 0.25
    neural.decided(SelectedAction.GET_POSSESSION)                     # the neural keeper goes anyway
    assert _label(expert, k, m).get_possession_extra == 1.0           # committed now: keep going
    assert expert.core._closing_down


# ------------------------------------------------------------------- attacker

@pytest.fixture
def attacker_world():
    m = build_1v2_scenario(seed=3)
    a = m.player_by_id("attacker")
    rules_ai, opening = a.ai, a.current_order
    a.ai, a.current_order = _NeuralStub(), None
    expert = RulesExpert(rules_ai, opening, a)
    return m, a, expert, opening


def test_attacker_runs_then_shoots_and_keeps_asking_until_it_really_kicks(attacker_world):
    m, a, expert, opening = attacker_world
    assert isinstance(opening, MoveOrder)
    lab = _label(expert, a, m)
    assert lab.move == 1.0 and lab.kick_this_tick == 0.0               # the run
    # Carry the ball past the run's target (nearer the goal): the rules
    # attacker shoots from here.
    gc = m.pitch.right_goal_centre
    a.position = Vector3(gc.x - 12.0, 0.0, 0.0)
    a.heading_rad, a.velocity = 0.0, Vector3(5.0, 0.0, 0.0)
    m.ball.position = Vector3(a.position.x + 0.3, 0.0, m.ball.radius_m)
    m.ball.velocity = a.velocity
    m.time_s += 0.25
    lab = _label(expert, a, m)
    assert lab.shoot == 1.0 and lab.kick_this_tick == 1.0
    # The neural attacker did not shoot: the next decision still asks for it.
    m.time_s += 0.25
    lab = _label(expert, a, m)
    assert lab.shoot == 1.0 and lab.kick_this_tick == 1.0
    # The neural attacker kicks for real: the plan is over.
    a.kick_count += 1
    m.ball.possessed_by = None
    m.ball.velocity = Vector3(20.0, 0.0, 0.0)
    m.time_s += 0.25
    lab = _label(expert, a, m)
    assert lab.shoot == 0.0 and lab.kick_this_tick == 0.0


def test_attacker_run_ends_after_a_neural_kick_that_loses_the_ball(attacker_world):
    m, a, expert, opening = attacker_world
    assert _label(expert, a, m).move == 1.0                            # on its run
    a.kick_count += 1                                                  # the neural attacker kicks it away
    m.ball.possessed_by = None
    m.ball.position = Vector3(a.position.x + 6.0, a.position.y, m.ball.radius_m)
    m.ball.velocity = Vector3(15.0, 0.0, 0.0)
    m.time_s += 0.25
    lab = _label(expert, a, m)
    assert lab.move == 0.0 and lab.hold_position == 1.0               # the run is over, not resumed
    assert not isinstance(expert.shadow, MoveOrder)


def test_attacker_run_goes_on_after_a_recovered_dribble_touch(attacker_world):
    m, a, expert, opening = attacker_world
    assert _label(expert, a, m).move == 1.0
    a.kick_count += 1                                                  # a touch it has recovered
    m.time_s += 0.25
    lab = _label(expert, a, m)
    assert lab.move == 1.0                                             # still on the same run
    assert isinstance(expert.shadow, MoveOrder)


def test_an_unsupported_rules_ai_is_refused():
    from footballcoach.rules_ai import SprintWaypointAI
    m = build_1v2_scenario(seed=1)
    with pytest.raises(ValueError, match="no BC expert support"):
        RulesExpert(SprintWaypointAI([Vector3(0, 0, 0)]), None, m.player_by_id("attacker"))

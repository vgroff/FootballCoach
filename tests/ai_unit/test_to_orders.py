"""Unit tests for action/to_orders.py - GatingResult -> engine orders.

Tests both the legal-action path (correct order type assigned) and the
illegal-action detection path (flag set, reason recorded).

IMPORTANT: these tests use real Match objects (from conftest.py) and verify
the engine state, not just return values.  If the precondition checks in
to_orders.py drift out of sync with the engine's own guards, these tests
will catch it.
"""
import random

import pytest

from footballcoach.ai.action.gating import GatingResult, SelectedAction
from footballcoach.ai.action.to_orders import apply_action_to_player
from footballcoach.entities.player import PlayerState
from footballcoach.mathutils import Vector3
from footballcoach.orders import (
    ChaseTackleOrder,
    GetPossessionOrder,
    KickOrder,
    MarkOrder,
    MoveOrder,
    OrderStatus,
    PassOrder,
)

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOVE_DIR = np.array([1.0, 0.0])
_KICK_DIR = np.array([1.0, 0.0])
_KICK_SPIN = np.zeros(3)


def _gating(selected: SelectedAction, target_slot: int | None = None,
             move_dir=None, sprint=True, kick=False,
             kick_dir=None, kick_power=0.8, kick_spin=None,
             tackle_attempt=False) -> GatingResult:
    return GatingResult(
        selected=selected,
        target_slot=target_slot,
        move_direction=move_dir if move_dir is not None else _MOVE_DIR,
        sprint=sprint,
        kick_this_tick=kick,
        kick_direction=kick_dir if kick_dir is not None else _KICK_DIR,
        kick_power_fraction=kick_power,
        kick_spin=kick_spin if kick_spin is not None else _KICK_SPIN,
        tackle_attempt=tackle_attempt,
    )


def _apply(gating, match, player_id, slot_player_ids=None, decision_physical=None):
    player = match.player_by_id(player_id)
    return apply_action_to_player(
        gating=gating,
        player=player,
        match=match,
        slot_player_ids=slot_player_ids or [None] * 21,
        decision_physical=decision_physical or {},
    )


# ---------------------------------------------------------------------------
# SHOOT
# ---------------------------------------------------------------------------

class TestShoot:
    def test_shoot_with_possession_legal(self, duel_match):
        """p1 has the ball -> shoot is legal -> KickOrder assigned."""
        result = _apply(_gating(SelectedAction.SHOOT), duel_match, "p1")
        assert not result.illegal_action
        assert isinstance(duel_match.player_by_id("p1").current_order, KickOrder)

    def test_shoot_without_possession_illegal(self, duel_match):
        """p2 does NOT have the ball -> shoot is illegal."""
        result = _apply(_gating(SelectedAction.SHOOT), duel_match, "p2")
        assert result.illegal_action
        assert "possession" in result.illegal_reason

    def test_shoot_without_possession_no_order(self, duel_match):
        """Illegal shoot must not assign a KickOrder."""
        p2 = duel_match.player_by_id("p2")
        p2.current_order = None
        _apply(_gating(SelectedAction.SHOOT), duel_match, "p2")
        assert not isinstance(p2.current_order, KickOrder)


# ---------------------------------------------------------------------------
# PASS
# ---------------------------------------------------------------------------

class TestPass:
    def _slot_ids_with_p2(self, match, slot=3):
        ids = [None] * 21
        ids[slot] = "p2"
        return ids

    def test_pass_with_possession_legal(self, duel_match):
        slot_ids = self._slot_ids_with_p2(duel_match)
        result = _apply(
            _gating(SelectedAction.PASS, target_slot=3),
            duel_match, "p1",
            slot_player_ids=slot_ids,
        )
        assert not result.illegal_action
        assert isinstance(duel_match.player_by_id("p1").current_order, PassOrder)

    def test_pass_without_possession_illegal(self, duel_match):
        slot_ids = self._slot_ids_with_p2(duel_match)
        result = _apply(
            _gating(SelectedAction.PASS, target_slot=3),
            duel_match, "p2",
            slot_player_ids=slot_ids,
        )
        assert result.illegal_action

    def test_pass_no_valid_target_illegal(self, duel_match):
        """target_slot=5 but slot 5 is None -> illegal."""
        result = _apply(
            _gating(SelectedAction.PASS, target_slot=5),
            duel_match, "p1",
            slot_player_ids=[None] * 21,
        )
        assert result.illegal_action
        assert "target" in result.illegal_reason


# ---------------------------------------------------------------------------
# TACKLE
# ---------------------------------------------------------------------------

class TestTackle:
    def _slot_ids_with_opponent(self, slot=2):
        ids = [None] * 21
        ids[slot] = "p2"
        return ids

    def test_tackle_active_player_legal(self, duel_match):
        slot_ids = self._slot_ids_with_opponent()
        result = _apply(
            _gating(SelectedAction.TACKLE, target_slot=2),
            duel_match, "p1",
            slot_player_ids=slot_ids,
        )
        assert not result.illegal_action
        assert isinstance(duel_match.player_by_id("p1").current_order, ChaseTackleOrder)

    def test_tackle_while_inactive_illegal(self, duel_match):
        duel_match.player_by_id("p1").state = PlayerState.INACTIVE_TACKLED
        slot_ids = self._slot_ids_with_opponent()
        result = _apply(
            _gating(SelectedAction.TACKLE, target_slot=2),
            duel_match, "p1",
            slot_player_ids=slot_ids,
        )
        assert result.illegal_action
        assert "inactive" in result.illegal_reason

    def test_tackle_own_teammate_illegal(self, standard_pitch):
        """Tackle targeting own teammate -> illegal."""
        import random as _r
        from footballcoach.engine.match import Match
        from footballcoach.entities.player import Player, Team
        from footballcoach.entities.attributes import PlayerAttributes
        from footballcoach.entities.ball import Ball

        attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        p1 = Player.create("p1", Team.LEFT, attrs, position=Vector3(0, 0, 0))
        p2 = Player.create("p2", Team.LEFT, attrs, position=Vector3(5, 0, 0))  # same team!
        ball = Ball.at_rest(Vector3(0, 0, 0))
        match = Match(pitch=standard_pitch, players=[p1, p2], ball=ball,
                      rng_reduction=1.0, rng=_r.Random(0))
        slot_ids = [None] * 21
        slot_ids[0] = "p2"
        result = _apply(
            _gating(SelectedAction.TACKLE, target_slot=0),
            match, "p1",
            slot_player_ids=slot_ids,
        )
        assert result.illegal_action
        assert "teammate" in result.illegal_reason

    def test_tackle_no_valid_target_illegal(self, duel_match):
        result = _apply(
            _gating(SelectedAction.TACKLE, target_slot=10),
            duel_match, "p1",
            slot_player_ids=[None] * 21,
        )
        assert result.illegal_action


# ---------------------------------------------------------------------------
# GET_POSSESSION
# ---------------------------------------------------------------------------

class TestGetPossession:
    def test_active_player_legal(self, duel_match):
        result = _apply(_gating(SelectedAction.GET_POSSESSION), duel_match, "p2")
        assert not result.illegal_action
        assert isinstance(duel_match.player_by_id("p2").current_order, GetPossessionOrder)

    def test_inactive_player_illegal(self, duel_match):
        duel_match.player_by_id("p2").state = PlayerState.INACTIVE_TACKLED
        result = _apply(_gating(SelectedAction.GET_POSSESSION), duel_match, "p2")
        assert result.illegal_action


# ---------------------------------------------------------------------------
# MARK
# ---------------------------------------------------------------------------

class TestMark:
    def test_mark_opponent_legal(self, duel_match):
        slot_ids = [None] * 21
        slot_ids[0] = "p2"
        result = _apply(
            _gating(SelectedAction.MARK, target_slot=0),
            duel_match, "p1",
            slot_player_ids=slot_ids,
        )
        assert not result.illegal_action
        assert isinstance(duel_match.player_by_id("p1").current_order, MarkOrder)

    def test_mark_own_team_illegal(self, standard_pitch):
        """Cannot mark own teammate."""
        import random as _r
        from footballcoach.engine.match import Match
        from footballcoach.entities.player import Player, Team
        from footballcoach.entities.attributes import PlayerAttributes
        from footballcoach.entities.ball import Ball

        attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        p1 = Player.create("p1", Team.LEFT, attrs, position=Vector3(0, 0, 0))
        p2 = Player.create("p2", Team.LEFT, attrs, position=Vector3(5, 0, 0))
        ball = Ball.at_rest(Vector3(0, 0, 0))
        match = Match(pitch=standard_pitch, players=[p1, p2], ball=ball,
                      rng_reduction=1.0, rng=_r.Random(0))
        slot_ids = [None] * 21
        slot_ids[0] = "p2"
        result = _apply(
            _gating(SelectedAction.MARK, target_slot=0),
            match, "p1",
            slot_player_ids=slot_ids,
        )
        assert result.illegal_action
        assert "teammate" in result.illegal_reason


# ---------------------------------------------------------------------------
# MOVE / HOLD_POSITION
# ---------------------------------------------------------------------------

class TestMove:
    def test_move_assigns_move_order(self, solo_match):
        result = _apply(
            _gating(SelectedAction.MOVE),
            solo_match, "p1",
            decision_physical={"move_region_center_m": np.array([10.0, 5.0])},
        )
        assert not result.illegal_action
        order = solo_match.player_by_id("p1").current_order
        assert isinstance(order, MoveOrder)

    def test_move_target_matches_decision_physical(self, solo_match):
        center = np.array([15.0, -3.0])
        _apply(
            _gating(SelectedAction.MOVE),
            solo_match, "p1",
            decision_physical={"move_region_center_m": center},
        )
        order = solo_match.player_by_id("p1").current_order
        assert order.target_position.x == pytest.approx(15.0, abs=0.1)
        assert order.target_position.y == pytest.approx(-3.0, abs=0.1)

    def test_hold_position_assigns_move_order(self, solo_match):
        result = _apply(
            _gating(SelectedAction.HOLD_POSITION),
            solo_match, "p1",
            decision_physical={"move_region_center_m": np.array([0.0, 0.0])},
        )
        assert not result.illegal_action
        assert isinstance(solo_match.player_by_id("p1").current_order, MoveOrder)


# ---------------------------------------------------------------------------
# NONE (no action selected)
# ---------------------------------------------------------------------------

class TestNone:
    def test_none_with_no_order_assigns_move(self, solo_match):
        """When NONE and no active order, execution direction gives a MoveOrder."""
        solo_match.player_by_id("p1").current_order = None
        result = _apply(
            _gating(SelectedAction.NONE, move_dir=np.array([1.0, 0.0])),
            solo_match, "p1",
        )
        assert not result.illegal_action
        assert isinstance(solo_match.player_by_id("p1").current_order, MoveOrder)

    def test_none_with_active_order_leaves_it(self, solo_match):
        """When NONE and an in-progress order exists, it must not be overwritten."""
        existing = MoveOrder(target_position=Vector3(20, 0, 0))
        existing.status = OrderStatus.IN_PROGRESS
        solo_match.player_by_id("p1").current_order = existing
        _apply(_gating(SelectedAction.NONE), solo_match, "p1")
        assert solo_match.player_by_id("p1").current_order is existing

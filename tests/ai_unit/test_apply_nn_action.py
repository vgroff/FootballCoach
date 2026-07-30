"""Unit tests for action/apply_nn_action.py - neural network execution outputs -> direct player state.

The neural network NEVER issues Orders. It sets player.desired_direction,
player.desired_speed_mode, and calls player.kick_direct() / player.tackle_direct().

These tests verify:
  - Movement sets desired_direction and desired_speed_mode directly (no Orders).
  - Kick calls player.kick_direct() when player has possession; illegal otherwise.
  - Tackle calls player.tackle_direct() with correct precondition checks.
  - Decision head selections (SHOOT/PASS/MOVE/etc.) do NOT cause Orders.
"""
import random

import pytest

from footballcoach.ai.action.gating import GatingResult, SelectedAction
from footballcoach.ai.action.apply_nn_action import apply_action_to_player
from footballcoach.engine.movement import SpeedMode
from footballcoach.entities.player import PlayerState
from footballcoach.mathutils import Vector3

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOVE_DIR = np.array([1.0, 0.0])
_KICK_DIR = np.array([1.0, 0.0])
_KICK_SPIN = np.zeros(3)


def _gating(selected: SelectedAction, target_slot: int | None = None,
             move_dir=None, exec_move=True, sprint=True, kick=False,
             kick_dir=None, kick_power=0.8, kick_spin=None,
             tackle_attempt=False) -> GatingResult:
    return GatingResult(
        selected=selected,
        target_slot=target_slot,
        exec_move=exec_move,
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
# KICK (kick_this_tick output from execution network)
# ---------------------------------------------------------------------------

class TestKick:
    def test_kick_with_possession_legal(self, duel_match):
        """kick_this_tick=True + has ball -> kick_direct() fires, no illegal."""
        result = _apply(_gating(SelectedAction.NONE, kick=True), duel_match, "p1")
        assert not result.illegal_action

    def test_kick_without_possession_illegal(self, duel_match):
        """kick_this_tick=True but no possession -> illegal."""
        result = _apply(_gating(SelectedAction.NONE, kick=True), duel_match, "p2")
        assert result.illegal_action
        assert "possession" in result.illegal_reason

    def test_no_kick_no_order_set(self, duel_match):
        """kick_this_tick=False -> current_order not touched by kick path."""
        p1 = duel_match.player_by_id("p1")
        p1.current_order = None
        _apply(_gating(SelectedAction.NONE, kick=False), duel_match, "p1")
        assert p1.current_order is None  # movement sets desired_* not current_order


# PASS is a decision-context input — the neural network does not issue PassOrders.
# No pass tests needed here; passing will be trained via BC labels.


# ---------------------------------------------------------------------------
# TACKLE (tackle_attempt output from execution network)
# ---------------------------------------------------------------------------

class TestTackle:
    def _slot_ids_with_opponent(self, slot=2):
        ids = [None] * 21
        ids[slot] = "p2"
        return ids

    def test_tackle_active_player_legal(self, duel_match):
        """tackle_attempt=True + valid target -> no illegal (out-of-range is fine)."""
        slot_ids = self._slot_ids_with_opponent()
        result = _apply(
            _gating(SelectedAction.NONE, tackle_attempt=True, target_slot=2),
            duel_match, "p1",
            slot_player_ids=slot_ids,
        )
        assert not result.illegal_action

    def test_tackle_while_inactive_illegal(self, duel_match):
        duel_match.player_by_id("p1").state = PlayerState.INACTIVE_TACKLED
        slot_ids = self._slot_ids_with_opponent()
        result = _apply(
            _gating(SelectedAction.NONE, tackle_attempt=True, target_slot=2),
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
            _gating(SelectedAction.NONE, tackle_attempt=True, target_slot=0),
            match, "p1",
            slot_player_ids=slot_ids,
        )
        assert result.illegal_action
        assert "teammate" in result.illegal_reason

    def test_tackle_no_valid_target_illegal(self, duel_match):
        result = _apply(
            _gating(SelectedAction.NONE, tackle_attempt=True, target_slot=10),
            duel_match, "p1",
            slot_player_ids=[None] * 21,
        )
        assert result.illegal_action


# GET_POSSESSION and MARK are decision-context inputs — the neural network
# does not issue GetPossessionOrder or MarkOrder. No tests needed here.


# ---------------------------------------------------------------------------
# MOVEMENT (desired_direction / desired_speed_mode — no Orders)
# ---------------------------------------------------------------------------

class TestMove:
    def test_move_dir_sets_desired_direction(self, solo_match):
        """move_direction sets player.desired_direction directly, no MoveOrder."""
        player = solo_match.player_by_id("p1")
        move_dir = np.array([0.0, 1.0])
        _apply(_gating(SelectedAction.MOVE, move_dir=move_dir, sprint=False), solo_match, "p1")
        assert player.desired_direction.y == pytest.approx(1.0, abs=0.01)
        assert player.desired_direction.x == pytest.approx(0.0, abs=0.01)
        assert player.desired_speed_mode == SpeedMode.JOG
        assert player.current_order is None  # NO Order issued

    def test_sprint_sets_sprint_mode(self, solo_match):
        player = solo_match.player_by_id("p1")
        _apply(_gating(SelectedAction.MOVE, sprint=True), solo_match, "p1")
        assert player.desired_speed_mode == SpeedMode.SPRINT

    def test_zero_direction_sets_standstill(self, solo_match):
        """exec_move=False -> STANDSTILL regardless of move_direction."""
        player = solo_match.player_by_id("p1")
        _apply(_gating(SelectedAction.NONE, exec_move=False, move_dir=np.array([1.0, 0.0])), solo_match, "p1")
        assert player.desired_speed_mode == SpeedMode.STANDSTILL
        assert player.current_order is None

    def test_none_action_with_direction_still_moves(self, solo_match):
        """NONE (all decision heads < 0.5) with exec_move=True still moves the player."""
        player = solo_match.player_by_id("p1")
        _apply(_gating(SelectedAction.NONE, exec_move=True, move_dir=np.array([1.0, 0.0]), sprint=True), solo_match, "p1")
        assert player.desired_speed_mode == SpeedMode.SPRINT

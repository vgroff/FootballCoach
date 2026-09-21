"""Unit tests for action/apply_nn_action.py - neural network execution outputs -> direct player state.

The neural network NEVER issues Orders. It sets player.desired_direction,
player.desired_speed_mode, and sets player.tackle_armed.

These tests verify:
  - Movement sets desired_direction and desired_speed_mode directly (no Orders).
  - Kick calls player.kick_direct() when player has possession; illegal otherwise.
  - Tackle sets player.tackle_armed when an opposing carrier exists; illegal otherwise.
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


def _apply(gating, match, player_id, slot_player_ids=None, decision_physical=None, **kwargs):
    player = match.player_by_id(player_id)
    return apply_action_to_player(
        gating=gating,
        player=player,
        match=match,
        slot_player_ids=slot_player_ids or [None] * 21,
        decision_physical=decision_physical or {},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# KICK (kick_this_tick output from execution network)
# ---------------------------------------------------------------------------

class TestKick:
    def test_kick_with_possession_legal(self, duel_match):
        """kick_this_tick=True + has ball -> kick_direct() fires, no illegal."""
        result = _apply(_gating(SelectedAction.NONE, kick=True), duel_match, "p1")
        assert not result.illegal_action

    def test_kick_without_possession_is_silent_noop(self, duel_match):
        """kick_this_tick=True but no possession -> silent no-op, not illegal."""
        p2 = duel_match.player_by_id("p2")
        result = _apply(_gating(SelectedAction.NONE, kick=True), duel_match, "p2")
        assert not result.illegal_action
        assert not p2.kicked_this_tick

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
    def test_tackle_active_player_legal(self, duel_match):
        """tackle_attempt=True when opponent has ball -> arms tackle, no illegal."""
        # duel_match: p1 (LEFT) has ball, p2 (RIGHT) does not.
        # p2 arming against p1 (opposing carrier) is legal.
        p2 = duel_match.player_by_id("p2")
        result = _apply(
            _gating(SelectedAction.NONE, tackle_attempt=True),
            duel_match, "p2",
        )
        assert not result.illegal_action
        assert p2.tackle_armed

    def test_tackle_while_inactive_illegal(self, duel_match):
        duel_match.player_by_id("p2").state = PlayerState.INACTIVE_TACKLED
        result = _apply(
            _gating(SelectedAction.NONE, tackle_attempt=True),
            duel_match, "p2",
        )
        assert result.illegal_action
        assert "inactive" in result.illegal_reason

    def test_tackle_no_carrier_illegal(self, standard_pitch):
        """tackle_attempt=True when no opposing carrier exists -> illegal."""
        import random as _r
        from footballcoach.engine.match import Match
        from footballcoach.entities.player import Player, Team
        from footballcoach.entities.attributes import PlayerAttributes
        from footballcoach.entities.ball import Ball

        attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        p1 = Player.create("p1", Team.LEFT, attrs, position=Vector3(0, 0, 0))
        p2 = Player.create("p2", Team.RIGHT, attrs, position=Vector3(5, 0, 0))
        ball = Ball.at_rest(Vector3(20, 0, 0))  # loose — no carrier
        match = Match(pitch=standard_pitch, players=[p1, p2], ball=ball,
                      rng_reduction=1.0, rng=_r.Random(0))
        result = _apply(
            _gating(SelectedAction.NONE, tackle_attempt=True),
            match, "p1", arm_tackle_without_carrier=False,
        )
        assert result.illegal_action
        assert "carrier" in result.illegal_reason
        assert not match.player_by_id("p1").tackle_armed

    def test_tackle_own_team_carrier_illegal(self, standard_pitch):
        """tackle_attempt=True when only a same-team carrier exists -> illegal."""
        import random as _r
        from footballcoach.engine.match import Match
        from footballcoach.entities.player import Player, Team
        from footballcoach.entities.attributes import PlayerAttributes
        from footballcoach.entities.ball import Ball

        attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        p1 = Player.create("p1", Team.LEFT, attrs, position=Vector3(0, 0, 0))
        p2 = Player.create("p2", Team.LEFT, attrs, position=Vector3(5, 0, 0))  # same team carries
        ball = Ball.at_rest(Vector3(5, 0, 0))
        ball.possessed_by = "p2"
        match = Match(pitch=standard_pitch, players=[p1, p2], ball=ball,
                      rng_reduction=1.0, rng=_r.Random(0))
        result = _apply(
            _gating(SelectedAction.NONE, tackle_attempt=True),
            match, "p1", arm_tackle_without_carrier=False,
        )
        assert result.illegal_action
        assert "carrier" in result.illegal_reason
        assert not match.player_by_id("p1").tackle_armed


def _loose_ball_match(standard_pitch):
    import random as _r
    from footballcoach.engine.match import Match
    from footballcoach.entities.player import Player, Team
    from footballcoach.entities.attributes import PlayerAttributes
    from footballcoach.entities.ball import Ball

    attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    p1 = Player.create("p1", Team.LEFT, attrs, position=Vector3(0, 0, 0))
    p2 = Player.create("p2", Team.RIGHT, attrs, position=Vector3(5, 0, 0))
    return Match(pitch=standard_pitch, players=[p1, p2], ball=Ball.at_rest(Vector3(20, 0, 0)),
                 rng_reduction=1.0, rng=_r.Random(0))


class TestTackleArmWithoutCarrier:
    """action.arm_tackle_without_carrier: every sampled attempt arms (and is thus charged)."""

    def test_arms_when_ball_is_loose(self, standard_pitch):
        match = _loose_ball_match(standard_pitch)
        result = _apply(_gating(SelectedAction.NONE, tackle_attempt=True), match, "p1",
                        arm_tackle_without_carrier=True)
        assert not result.illegal_action
        assert match.player_by_id("p1").tackle_armed

    def test_arms_while_holding_the_ball(self, duel_match):
        # p1 carries the ball; arming while possessing is what tackle_armed_while_possessing_multiplier prices.
        result = _apply(_gating(SelectedAction.NONE, tackle_attempt=True), duel_match, "p1",
                        arm_tackle_without_carrier=True)
        assert not result.illegal_action
        assert duel_match.player_by_id("p1").tackle_armed

    def test_arms_when_own_team_carries(self, standard_pitch):
        import random as _r
        from footballcoach.engine.match import Match
        from footballcoach.entities.player import Player, Team
        from footballcoach.entities.attributes import PlayerAttributes
        from footballcoach.entities.ball import Ball

        attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        p1 = Player.create("p1", Team.LEFT, attrs, position=Vector3(0, 0, 0))
        p2 = Player.create("p2", Team.LEFT, attrs, position=Vector3(5, 0, 0))
        ball = Ball.at_rest(Vector3(5, 0, 0))
        ball.possessed_by = "p2"
        match = Match(pitch=standard_pitch, players=[p1, p2], ball=ball, rng_reduction=1.0, rng=_r.Random(0))
        result = _apply(_gating(SelectedAction.NONE, tackle_attempt=True), match, "p1",
                        arm_tackle_without_carrier=True)
        assert not result.illegal_action
        assert match.player_by_id("p1").tackle_armed

    def test_inactive_player_still_cannot_arm(self, duel_match):
        duel_match.player_by_id("p2").state = PlayerState.INACTIVE_TACKLED
        result = _apply(_gating(SelectedAction.NONE, tackle_attempt=True), duel_match, "p2",
                        arm_tackle_without_carrier=True)
        assert result.illegal_action and "inactive" in result.illegal_reason
        assert not duel_match.player_by_id("p2").tackle_armed

    def test_no_attempt_no_arm(self, standard_pitch):
        match = _loose_ball_match(standard_pitch)
        _apply(_gating(SelectedAction.NONE, tackle_attempt=False), match, "p1", arm_tackle_without_carrier=True)
        assert not match.player_by_id("p1").tackle_armed

    def test_pre_armed_tackle_resolves_once_the_opponent_gets_the_ball(self, standard_pitch):
        """Arm while the ball is loose, then the opponent takes it: the (re-applied) armed tackle resolves on contact."""
        import random as _r
        from footballcoach.engine.match import Match
        from footballcoach.entities.player import Player, Team
        from footballcoach.entities.attributes import PlayerAttributes
        from footballcoach.entities.ball import Ball

        attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        p1 = Player.create("p1", Team.LEFT, attrs, position=Vector3(0, 0, 0))
        p2 = Player.create("p2", Team.RIGHT, attrs, position=Vector3(0.3, 0, 0))
        match = Match(pitch=standard_pitch, players=[p1, p2], ball=Ball.at_rest(Vector3(30, 0, 0)),
                      rng_reduction=1.0, rng=_r.Random(0))
        p1 = match.player_by_id("p1")
        gating = _gating(SelectedAction.NONE, tackle_attempt=True)
        _apply(gating, match, "p1", arm_tackle_without_carrier=True)
        assert p1.tackle_armed and match.ball_carrier() is None
        match._check_armed_tackles()  # nobody carries -> nothing to resolve, no error
        match.ball.possessed_by = "p2"
        fired = []
        p1.on_tackle = lambda pl: fired.append(pl.player_id)
        _apply(gating, match, "p1", arm_tackle_without_carrier=True)
        match._check_armed_tackles()
        assert fired == ["p1"]


class TestTackleConfigDefault:
    def test_flag_reads_config_when_not_passed(self, standard_pitch, monkeypatch):
        import footballcoach.ai.action.apply_nn_action as ana
        match = _loose_ball_match(standard_pitch)
        monkeypatch.setattr(ana, "action_flag", lambda name: True)
        _apply(_gating(SelectedAction.NONE, tackle_attempt=True), match, "p1")
        assert match.player_by_id("p1").tackle_armed
        match2 = _loose_ball_match(standard_pitch)
        monkeypatch.setattr(ana, "action_flag", lambda name: False)
        result = _apply(_gating(SelectedAction.NONE, tackle_attempt=True), match2, "p1")
        assert result.illegal_action and not match2.player_by_id("p1").tackle_armed

    def test_config_declares_both_flags_as_booleans(self):
        from footballcoach.ai.action.apply_nn_action import action_flag
        from footballcoach.ai.config import load_ai_config
        section = load_ai_config()["action"]
        for key in ("arm_tackle_without_carrier", "kick_one_shot"):
            assert isinstance(section[key], bool) and f"_comment_{key}" in section
            assert action_flag(key) is section[key]
        assert action_flag("nonexistent_flag") is False


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


class TestKickPowerValidation:
    """A kick decision must carry a real power; the old silent 0.85 fallback is gone."""

    @pytest.mark.parametrize("bad_power", [0.0, -0.1, float("nan")])
    def test_non_positive_or_nan_power_raises(self, duel_match, bad_power):
        with pytest.raises(ValueError, match="kick_power_fraction"):
            _apply(_gating(SelectedAction.NONE, kick=True, kick_power=bad_power), duel_match, "p1")

    def test_a_small_positive_power_is_used_as_is(self, duel_match):
        p1 = duel_match.player_by_id("p1")
        _apply(_gating(SelectedAction.NONE, kick=True, kick_power=0.03), duel_match, "p1")
        assert p1.kicked_this_tick and p1.last_kick_power_fraction == pytest.approx(0.03, abs=1e-6)

    def test_zero_power_without_a_kick_decision_is_fine(self, duel_match):
        _apply(_gating(SelectedAction.NONE, kick=False, kick_power=0.0), duel_match, "p1")

    def test_armed_kick_keeps_the_given_power(self, duel_match):
        p2 = duel_match.player_by_id("p2")
        _apply(_gating(SelectedAction.NONE, kick=True, kick_power=0.04), duel_match, "p2")
        assert p2.kick_armed and p2.kick_armed_power_fraction == pytest.approx(0.04)

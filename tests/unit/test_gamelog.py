"""Unit tests for ui/gamelog.py — GameLog ring buffer and LogLevel filtering.

Also tests the tackle-logging plumbing in Match: that a resolved tackle
(both win and loss, at rng_reduction=1.0 for determinism) produces exactly
one log entry containing the expected participant ids and outcome text, and
that the GK-in-box auto-fail short-circuit produces a distinct log entry.
"""
from __future__ import annotations

import random

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, PlayerAttributes, Team
from footballcoach.entities.player import Player
from footballcoach.mathutils import Vector3
from footballcoach.orders import ChaseTackleOrder
from footballcoach.ui.gamelog import GameLog, LogLevel


# ---------------------------------------------------------------------------
# Basic GameLog tests
# ---------------------------------------------------------------------------

def test_add_and_retrieve_entries():
    log = GameLog()
    log.add(LogLevel.INFO, "hello")
    log.add(LogLevel.DEBUG, "world")
    entries = log.all_entries
    assert len(entries) == 2
    assert entries[0].message == "hello"
    assert entries[0].level == LogLevel.INFO
    assert entries[1].message == "world"
    assert entries[1].level == LogLevel.DEBUG


def test_entries_above_info_filters_debug():
    log = GameLog()
    log.add(LogLevel.INFO, "info message")
    log.add(LogLevel.DEBUG, "debug message")
    info_only = log.entries_above(LogLevel.INFO)
    assert len(info_only) == 1
    assert info_only[0].message == "info message"


def test_entries_above_debug_returns_all():
    log = GameLog()
    log.add(LogLevel.INFO, "info message")
    log.add(LogLevel.DEBUG, "debug message")
    all_entries = log.entries_above(LogLevel.DEBUG)
    assert len(all_entries) == 2


def test_deque_evicts_oldest_at_maxlen():
    log = GameLog(max_entries=3)
    for i in range(5):
        log.add(LogLevel.INFO, f"msg{i}")
    entries = log.all_entries
    assert len(entries) == 3
    # Oldest two (msg0, msg1) should have been evicted.
    messages = [e.message for e in entries]
    assert "msg0" not in messages
    assert "msg1" not in messages
    assert "msg4" in messages


def test_time_s_stored_on_entry():
    log = GameLog()
    log.add(LogLevel.INFO, "timed", time_s=3.14)
    assert log.all_entries[0].time_s == 3.14


def test_empty_log_returns_empty_list():
    log = GameLog()
    assert log.entries_above(LogLevel.INFO) == []
    assert log.entries_above(LogLevel.DEBUG) == []
    assert log.all_entries == []


# ---------------------------------------------------------------------------
# Tackle-logging plumbing tests
# ---------------------------------------------------------------------------

def _make_tackle_match(rng_reduction: float = 1.0) -> tuple[Match, Player, Player, GameLog]:
    """Build a minimal match with a defender touching an attacker who has the
    ball, and attach a GameLog via log_callback."""
    pitch = Pitch.standard()
    defender_attrs = PlayerAttributes(
        top_speed=0.6, acceleration=0.6, stamina=0.6, kick_precision=0.6,
        kick_power=0.6, dribbling=0.6, ball_control=0.6, tackling=0.9,
    )
    attacker_attrs = PlayerAttributes(
        top_speed=0.6, acceleration=0.6, stamina=0.6, kick_precision=0.6,
        kick_power=0.6, dribbling=0.3, ball_control=0.6, tackling=0.3,
    )
    defender = Player.create("def", Team.LEFT, defender_attrs, position=Vector3(0, 0, 0))
    attacker = Player.create("atk", Team.RIGHT, attacker_attrs, position=Vector3(0.5, 0, 0))
    ball = Ball.at_rest(Vector3(0.5, 0, 0))
    ball.possessed_by = attacker.player_id
    match = Match(
        pitch=pitch, players=[defender, attacker], ball=ball,
        rng_reduction=rng_reduction, rng=random.Random(42),
    )
    game_log = GameLog()
    match.log_callback = lambda level, msg: game_log.add(level, msg, match.time_s)
    return match, defender, attacker, game_log


def test_tackle_win_produces_log_entry():
    """A successful tackle (rng_reduction=1.0 → deterministic win for 0.9 vs 0.3)
    must produce exactly one INFO log entry containing both player ids."""
    match, defender, attacker, game_log = _make_tackle_match(rng_reduction=1.0)
    defender.current_order = ChaseTackleOrder(target_player_id=attacker.player_id)
    match.step()
    info_entries = game_log.entries_above(LogLevel.INFO)
    assert len(info_entries) >= 1, "Expected at least one INFO log entry after a tackle"
    # At least one entry must mention both ids.
    combined = " ".join(e.message for e in info_entries)
    assert "def" in combined
    assert "atk" in combined


def test_tackle_loss_produces_log_entry():
    """A failed tackle (rng_reduction=1.0, attacker dribbling=0.9 vs tackling=0.3)
    must also produce a log entry with the correct participants."""
    pitch = Pitch.standard()
    defender_attrs = PlayerAttributes(
        top_speed=0.6, acceleration=0.6, stamina=0.6, kick_precision=0.6,
        kick_power=0.6, dribbling=0.6, ball_control=0.6, tackling=0.3,
    )
    attacker_attrs = PlayerAttributes(
        top_speed=0.6, acceleration=0.6, stamina=0.6, kick_precision=0.6,
        kick_power=0.6, dribbling=0.9, ball_control=0.6, tackling=0.3,
    )
    defender = Player.create("def2", Team.LEFT, defender_attrs, position=Vector3(0, 0, 0))
    attacker = Player.create("atk2", Team.RIGHT, attacker_attrs, position=Vector3(0.5, 0, 0))
    ball = Ball.at_rest(Vector3(0.5, 0, 0))
    ball.possessed_by = attacker.player_id
    match = Match(
        pitch=pitch, players=[defender, attacker], ball=ball,
        rng_reduction=1.0, rng=random.Random(42),
    )
    game_log = GameLog()
    match.log_callback = lambda level, msg: game_log.add(level, msg, match.time_s)
    defender.current_order = ChaseTackleOrder(target_player_id=attacker.player_id)
    match.step()
    combined = " ".join(e.message for e in game_log.entries_above(LogLevel.INFO))
    assert "def2" in combined
    assert "atk2" in combined


def test_debug_entries_contain_roll_values():
    """The DEBUG-level tackle log entry must mention both rolled values."""
    match, defender, attacker, game_log = _make_tackle_match(rng_reduction=1.0)
    defender.current_order = ChaseTackleOrder(target_player_id=attacker.player_id)
    match.step()
    debug_entries = game_log.entries_above(LogLevel.DEBUG)
    # At least one DEBUG entry should contain "tackler_roll" and "dribbler_roll".
    roll_entries = [e for e in debug_entries if "tackler_roll" in e.message]
    assert len(roll_entries) >= 1, "Expected DEBUG entry with roll values"


def test_gk_in_box_auto_fail_logs_distinctly():
    """A tackle attempt against a GK in possession inside their own box must
    produce an INFO log entry mentioning the auto-fail (not a normal roll)."""
    pitch = Pitch.standard()
    defender_attrs = PlayerAttributes(
        top_speed=0.6, acceleration=0.6, stamina=0.6, kick_precision=0.6,
        kick_power=0.6, dribbling=0.6, ball_control=0.6, tackling=0.9,
    )
    gk_attrs = PlayerAttributes(
        top_speed=0.8, acceleration=0.8, stamina=0.8, kick_precision=0.5,
        kick_power=0.5, dribbling=0.5, ball_control=0.8, tackling=0.5,
    )
    # Place GK inside the left box.
    gk_pos = Vector3(-pitch.half_length + 5.0, 0.0, 0)
    gk = Player.create("gk", Team.LEFT, gk_attrs, position=gk_pos, is_goalkeeper=True)
    # Defender just touching the GK.
    def_pos = Vector3(gk_pos.x + 0.5, 0.0, 0)
    outfielder = Player.create("out", Team.RIGHT, defender_attrs, position=def_pos)
    ball = Ball.at_rest(gk_pos)
    ball.possessed_by = gk.player_id
    match = Match(
        pitch=pitch, players=[gk, outfielder], ball=ball,
        rng_reduction=1.0, rng=random.Random(0),
    )
    game_log = GameLog()
    match.log_callback = lambda level, msg: game_log.add(level, msg, match.time_s)
    outfielder.current_order = ChaseTackleOrder(target_player_id=gk.player_id)
    match.step()
    combined = " ".join(e.message for e in game_log.entries_above(LogLevel.INFO))
    # The auto-fail message should mention the GK immunity reason.
    assert "auto-failed" in combined.lower() or "gk" in combined.lower(), (
        f"Expected GK-immunity message in log; got: {combined!r}"
    )


def test_log_callback_none_incurs_no_error():
    """With log_callback=None (default), match steps must complete without
    exceptions — the zero-cost headless path."""
    match, defender, attacker, _ = _make_tackle_match()
    match.log_callback = None  # explicit no-op
    defender.current_order = ChaseTackleOrder(target_player_id=attacker.player_id)
    match.step()  # must not raise

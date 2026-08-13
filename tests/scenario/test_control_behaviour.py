"""Tests for first-touch control behaviour introduced alongside the
'ball-glued-to-controlling-player' rework:

- Ball is possessed immediately on contact (not after the timer expires).
- Player's speed is snapped down by control_speed_multiplier on contact.
- Player can be tackled while controlling a ground ball (below waist height).
- Player is immune to both regular and head-on tackles while controlling an
  aerial ball (above control_tackle_immune_height_m = waist height).
"""
from __future__ import annotations

import random

import pytest

from footballcoach.engine.match import Match
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import PlayerState, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import GetPossessionOrder

from tests.conftest import make_player


def _make_match(carrier, receiver, ball, *, rng_reduction=1.0):
    return Match(
        pitch=Pitch.standard(),
        players=[carrier, receiver],
        ball=ball,
        rng_reduction=rng_reduction,
        rng=random.Random(0),
    )


# ---------------------------------------------------------------------------
# Immediate possession on contact
# ---------------------------------------------------------------------------

def test_possession_granted_immediately_on_control_start():
    """The moment a player begins CONTROLLING_BALL the ball is possessed by
    them — not after the timer expires."""
    receiver = make_player("rcv", Team.LEFT, attr_value=0.5, position=Vector3(0, 0, 0))
    ball = Ball(position=Vector3(0.3, 0, 0.11), velocity=Vector3(0, 0, 0), spin=Vector3.zero())

    match = Match(
        pitch=Pitch.standard(),
        players=[receiver],
        ball=ball,
        rng_reduction=1.0,
        rng=random.Random(0),
    )

    match.step()  # pickup radius check fires → CONTROLLING_BALL + possession

    assert receiver.state == PlayerState.CONTROLLING_BALL
    assert match.ball.possessed_by == receiver.player_id


# ---------------------------------------------------------------------------
# Speed penalty on contact
# ---------------------------------------------------------------------------

def test_speed_reduced_on_control_start():
    """A sprinting player's speed is multiplied by control_speed_multiplier
    the instant they start controlling the ball."""
    receiver = make_player("rcv", Team.LEFT, attr_value=0.5, position=Vector3(0, 0, 0))
    receiver.velocity = Vector3(8.0, 0, 0)  # fast sprint

    ball = Ball(position=Vector3(0.3, 0, 0.11), velocity=Vector3(0, 0, 0), spin=Vector3.zero())

    match = Match(
        pitch=Pitch.standard(),
        players=[receiver],
        ball=ball,
        rng_reduction=1.0,
        rng=random.Random(0),
    )

    pre_speed = receiver.velocity.length_xy()
    match.step()

    assert receiver.state == PlayerState.CONTROLLING_BALL
    expected_max = pre_speed * match.movement_params.control_speed_multiplier
    actual = receiver.velocity.length_xy()
    assert actual == pytest.approx(expected_max, rel=1e-5)


# ---------------------------------------------------------------------------
# Ground ball — can be tackled
# ---------------------------------------------------------------------------

def test_controlling_ground_ball_can_be_tackled():
    """A player controlling a ground ball (z below waist height) is tackleable —
    tackle should resolve and possession should transfer on a win.

    We pre-set CONTROLLING_BALL state directly to avoid control-timer expiry
    timing issues (very easy balls complete in < 1 tick).
    """
    receiver = make_player("rcv", Team.LEFT, attr_value=0.5, position=Vector3(0, 0, 0))
    # Tackler touching-close; high tackling guarantees win at rng_reduction=1.
    tackler = make_player("tkl", Team.RIGHT, attr_value=0.0, tackling=1.0,
                          position=Vector3(0.59, 0, 0))

    # Ball at ground level — well below waist height (0.95 m).
    ball = Ball(position=Vector3(0.3, 0, 0.11), velocity=Vector3(0, 0, 0), spin=Vector3.zero())

    match = _make_match(receiver, tackler, ball)

    # Pre-set receiver into CONTROLLING_BALL with a long timer so it won't
    # expire this tick, and grant possession immediately (the new behaviour).
    receiver.state = PlayerState.CONTROLLING_BALL
    receiver.state_timer_s = 0.5
    match._set_possession(receiver.player_id)

    tackler.current_order = GetPossessionOrder()

    match.step()

    assert match.ball.possessed_by == tackler.player_id


# ---------------------------------------------------------------------------
# Aerial ball — immune to regular tackle
# ---------------------------------------------------------------------------

def test_controlling_aerial_ball_immune_to_regular_tackle():
    """A player controlling a high ball (above waist height) cannot be tackled
    via a GetPossessionOrder/ChaseTackle approach."""
    # Receiver nudged from (0,0,0) to (0.05,0,0) -- clearly closer to the ball
    # (0.25m) than the tackler is (0.29m) so pickup contention is unambiguous
    # (closest ACTIVE candidate wins ties, see engine/knowledge.md's ball-
    # pickup contention note). The original (0,0,0)/0.59 placement was a
    # near-tie (0.30m vs 0.29m) that previously happened to resolve to the
    # receiver only via players-list iteration order. Ball/tackler positions
    # are left untouched so the tackler's tick-1 GetPossessionOrder chase
    # target and tick-2 touching-range tackle attempt are unaffected.
    receiver = make_player("rcv", Team.LEFT, attr_value=0.5, position=Vector3(0.05, 0, 0))
    tackler = make_player("tkl", Team.RIGHT, attr_value=0.0, tackling=1.0,
                          position=Vector3(0.59, 0, 0))

    # Ball well above waist height (0.95 m) — chest height.
    waist_h = 0.95
    ball = Ball(
        position=Vector3(0.3, 0, waist_h + 0.3),
        velocity=Vector3(0, 0, 0),
        spin=Vector3.zero(),
    )

    match = _make_match(receiver, tackler, ball)
    tackler.current_order = GetPossessionOrder()

    # Tick 1: receiver starts controlling, takes immediate possession.
    match.step()
    assert receiver.state == PlayerState.CONTROLLING_BALL
    assert match.ball.possessed_by == receiver.player_id

    # Tick 2: tackle attempt is blocked — possession stays with receiver.
    match.step()
    assert match.ball.possessed_by == receiver.player_id


# ---------------------------------------------------------------------------
# Aerial ball — immune to head-on auto-tackle
# ---------------------------------------------------------------------------

def test_controlling_aerial_ball_immune_to_head_on_tackle():
    """The auto-tackle (head-on overlap) is also blocked while the controlling
    player's ball is above the immune height threshold."""
    receiver = make_player("rcv", Team.LEFT, attr_value=0.5, position=Vector3(0, 0, 0))
    # Place tackler so they are in serious overlap (well inside auto-tackle range).
    overlap_r = (receiver.radius_m + 0.3) * 1.1  # inside factor*combined_radius
    tackler = make_player("tkl", Team.RIGHT, attr_value=0.0, tackling=1.0,
                          position=Vector3(overlap_r, 0, 0))
    # Both moving toward each other to ensure closing speed threshold is met.
    receiver.velocity = Vector3(-1.0, 0, 0)
    tackler.velocity = Vector3(-1.0, 0, 0)

    waist_h = 0.95
    ball = Ball(
        position=Vector3(0.3, 0, waist_h + 0.3),
        velocity=Vector3(0, 0, 0),
        spin=Vector3.zero(),
    )

    # Give receiver possession first (simulating they just started control).
    match = _make_match(receiver, tackler, ball)
    receiver.state = PlayerState.CONTROLLING_BALL
    receiver.state_timer_s = 0.3
    match._set_possession(receiver.player_id)

    match.step()

    # Possession must remain with receiver — head-on auto-tackle was blocked.
    assert match.ball.possessed_by == receiver.player_id

"""First-touch kick: kicking while in CONTROLLING_BALL applies firsttime
difficulty automatically and immediately clears the player back to ACTIVE.
"""
from __future__ import annotations

import random

import pytest

from footballcoach.engine.match import Match
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import PlayerState, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import KickOrder, MoveOrder
from tests.conftest import make_player


def _match(players, ball, *, rng_reduction=1.0):
    return Match(
        pitch=Pitch.standard(), players=players, ball=ball,
        rng_reduction=rng_reduction, rng=random.Random(0),
    )


# ---------------------------------------------------------------------------
# Basic: kick during CONTROLLING_BALL clears state
# ---------------------------------------------------------------------------

def test_kick_during_controlling_ball_clears_state():
    """KickOrder fired while CONTROLLING_BALL must leave the player ACTIVE."""
    player = make_player("p1", Team.LEFT, position=Vector3(0, 0, 0))
    ball = Ball(position=Vector3(0.3, 0, 0.05), velocity=Vector3(0, 0, 0), spin=Vector3.zero())
    match = _match([player], ball)

    match.step()  # ball pickup -> CONTROLLING_BALL
    assert player.state == PlayerState.CONTROLLING_BALL

    player.current_order = KickOrder(
        aim_point=Vector3(20, 0, 0), power_fraction=0.5, spin=Vector3.zero()
    )
    match.step()

    assert player.kicked_this_tick
    assert player.state == PlayerState.ACTIVE, "kick must clear CONTROLLING_BALL immediately"
    assert match.ball.possessed_by is None


def test_normal_kick_does_not_touch_state():
    """Kicking after full control (ACTIVE state) must not change state."""
    player = make_player("p1", Team.LEFT, position=Vector3(0, 0, 0))
    ball = Ball(position=Vector3(0, 0, 0), velocity=Vector3(0, 0, 0), spin=Vector3.zero())
    ball.possessed_by = player.player_id
    match = _match([player], ball)

    player.current_order = KickOrder(
        aim_point=Vector3(20, 0, 0), power_fraction=0.5, spin=Vector3.zero()
    )
    match.step()

    assert player.kicked_this_tick
    assert player.state == PlayerState.ACTIVE  # unchanged


# ---------------------------------------------------------------------------
# firsttime_difficulty is stored at pickup
# ---------------------------------------------------------------------------

def test_firsttime_difficulty_stored_for_fast_aerial_ball():
    """A fast chest-height ball must produce firsttime_difficulty > 0."""
    player = make_player("p1", Team.LEFT, position=Vector3(0, 0, 0))
    ball = Ball(position=Vector3(0.3, 0, 0.9), velocity=Vector3(-8, 0, 0), spin=Vector3.zero())
    match = _match([player], ball)
    match.step()

    assert player.state == PlayerState.CONTROLLING_BALL
    assert player.firsttime_difficulty > 0.0


def test_firsttime_difficulty_near_zero_for_stationary_ground_ball():
    """A stationary ball at floor level must produce firsttime_difficulty ≈ 0."""
    player = make_player("p1", Team.LEFT, position=Vector3(0, 0, 0))
    ball = Ball(position=Vector3(0.3, 0, 0.05), velocity=Vector3(0, 0, 0), spin=Vector3.zero())
    match = _match([player], ball)
    match.step()

    assert player.state == PlayerState.CONTROLLING_BALL
    assert player.firsttime_difficulty < 0.1


def test_firsttime_difficulty_larger_for_harder_ball():
    """A fast aerial ball must produce higher difficulty than a slow ground ball."""
    def _difficulty_for(ball_pos, ball_vel):
        player = make_player("p1", Team.LEFT, position=Vector3(0, 0, 0))
        ball = Ball(position=ball_pos, velocity=ball_vel, spin=Vector3.zero())
        match = _match([player], ball)
        match.step()
        return player.firsttime_difficulty

    easy = _difficulty_for(Vector3(0.3, 0, 0.05), Vector3(0, 0, 0))
    hard = _difficulty_for(Vector3(0.3, 0, 0.9), Vector3(-8, 0, 0))
    assert hard > easy


# ---------------------------------------------------------------------------
# Push-kick via MoveOrder fires as first-touch and clears state
# ---------------------------------------------------------------------------

def test_push_kick_during_controlling_ball_clears_state():
    """MoveOrder push-kick fired while CONTROLLING_BALL must clear the state."""
    player = make_player("p1", Team.LEFT, position=Vector3(-10, 0, 0))
    player.heading_rad = 0.0  # facing +x (toward target)
    player.velocity = Vector3(5, 0, 0)  # already moving rightward

    ball = Ball(position=Vector3(-9.7, 0, 0.05), velocity=Vector3(0, 0, 0), spin=Vector3.zero())
    match = _match([player], ball)

    match.step()  # pickup -> CONTROLLING_BALL
    assert player.state == PlayerState.CONTROLLING_BALL

    player.current_order = MoveOrder(
        target_position=Vector3(30, 0, 0), sprint=True, push_kick_enabled=True
    )
    match.step()

    assert player.kicked_this_tick, "push-kick must fire on the CONTROLLING_BALL step"
    assert player.state == PlayerState.ACTIVE, "push-kick must clear CONTROLLING_BALL"
    assert match.ball.possessed_by is None

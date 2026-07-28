"""Unit tests for ShootOrder.chance_of_pausing — the blocker-detection
feature that replaces a shot with a 2 m advance MoveOrder when an
opposition player lies on the shot line.

All tests use rng_reduction=1.0 (fully deterministic) so outcomes are
reproducible without needing many trials.
"""
from __future__ import annotations

import math
import random

import pytest

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import MoveOrder, ShootOrder
from tests.conftest import make_player


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_match(shooter_pos: Vector3, aim_point: Vector3,
                 extra_players: list = (),
                 rng_reduction: float = 1.0,
                 rng_seed: int = 0) -> tuple[Match, object]:
    """Build a minimal match with a ball-carrying shooter.
    Returns (match, shooter_player).
    """
    pitch = Pitch.standard()
    shooter = make_player("shooter", Team.LEFT, position=shooter_pos, kick_precision=0.8, kick_power=0.8)
    ball = Ball.at_rest(shooter_pos)
    ball.possessed_by = shooter.player_id

    players = [shooter, *extra_players]
    match = Match(pitch=pitch, players=players, ball=ball,
                  rng_reduction=rng_reduction, rng=random.Random(rng_seed))
    return match, shooter


# ---------------------------------------------------------------------------
# No blocker: ShootOrder fires the ball immediately
# ---------------------------------------------------------------------------

def test_no_blocker_shoots_immediately():
    """When no opposition player is on the shot line, ShootOrder kicks immediately."""
    pitch = Pitch.standard()
    aim = pitch.right_goal_centre.with_z(1.0)
    shooter_pos = Vector3(0.0, 0.0, 0.0)

    match, shooter = _setup_match(shooter_pos, aim)
    shooter.current_order = ShootOrder(aim_point=aim, power_fraction=0.85, chance_of_pausing=0.8)

    match.step()

    # Ball should be in motion (no longer at rest at shooter's feet)
    assert match.ball.possessed_by is None, "Ball was not released — shot did not fire"
    assert match.ball.velocity.length() > 0.5, "Ball velocity too low — shot did not fire"
    # Order consumed
    assert shooter.current_order is None


# ---------------------------------------------------------------------------
# Blocker on line + chance_of_pausing=1.0: always pauses, issues MoveOrder
# ---------------------------------------------------------------------------

def test_blocker_on_line_pauses_shot():
    """Opposition player directly on the shot line → ShootOrder becomes MoveOrder."""
    pitch = Pitch.standard()
    aim = pitch.right_goal_centre.with_z(1.0)
    shooter_pos = Vector3(0.0, 0.0, 0.0)

    # Place blocker exactly halfway between shooter and goal, dead on the line.
    blocker_pos = Vector3(pitch.half_length / 2.0, 0.0, 0.0)
    blocker = make_player("blocker", Team.RIGHT, position=blocker_pos)

    match, shooter = _setup_match(shooter_pos, aim, extra_players=[blocker])
    shooter.current_order = ShootOrder(aim_point=aim, power_fraction=0.85, chance_of_pausing=1.0)

    match.step()

    # Ball must still be with the shooter
    assert match.ball.possessed_by == shooter.player_id, (
        "Shooter lost possession when they should have paused"
    )
    # ShootOrder should have been replaced by a MoveOrder
    assert isinstance(shooter.current_order, MoveOrder), (
        f"Expected MoveOrder after pause, got {type(shooter.current_order)}"
    )


# ---------------------------------------------------------------------------
# MoveOrder target is ~2 m toward the aim point
# ---------------------------------------------------------------------------

def test_pause_move_target_is_2m_toward_goal():
    """After pausing, the MoveOrder target must be ~2 m from the shooter
    in the direction of the aim point."""
    pitch = Pitch.standard()
    aim = pitch.right_goal_centre.with_z(1.0)
    shooter_pos = Vector3(0.0, 0.0, 0.0)

    blocker_pos = Vector3(pitch.half_length / 2.0, 0.0, 0.0)
    blocker = make_player("blocker", Team.RIGHT, position=blocker_pos)

    match, shooter = _setup_match(shooter_pos, aim, extra_players=[blocker])
    shooter.current_order = ShootOrder(aim_point=aim, power_fraction=0.85, chance_of_pausing=1.0)
    match.step()

    assert isinstance(shooter.current_order, MoveOrder)
    target = shooter.current_order.target_position
    dist = math.hypot(target.x - shooter_pos.x, target.y - shooter_pos.y)
    assert abs(dist - 2.0) < 0.1, f"Advance distance should be ~2 m, got {dist:.3f} m"

    # Direction must point from shooter toward aim point (+x here)
    dx = target.x - shooter_pos.x
    assert dx > 1.5, "MoveOrder target should advance toward the goal (+x direction)"


# ---------------------------------------------------------------------------
# chance_of_pausing=0.0: never pauses even with a blocker
# ---------------------------------------------------------------------------

def test_chance_of_pausing_zero_never_pauses():
    """chance_of_pausing=0.0 disables the blocker check; shot fires regardless."""
    pitch = Pitch.standard()
    aim = pitch.right_goal_centre.with_z(1.0)
    shooter_pos = Vector3(0.0, 0.0, 0.0)

    blocker_pos = Vector3(pitch.half_length / 2.0, 0.0, 0.0)
    blocker = make_player("blocker", Team.RIGHT, position=blocker_pos)

    match, shooter = _setup_match(shooter_pos, aim, extra_players=[blocker])
    shooter.current_order = ShootOrder(aim_point=aim, power_fraction=0.85, chance_of_pausing=0.0)
    match.step()

    assert match.ball.possessed_by is None, "Shot should have fired with chance_of_pausing=0.0"


# ---------------------------------------------------------------------------
# Blocker behind shooter (outside line segment) does not trigger pause
# ---------------------------------------------------------------------------

def test_blocker_behind_shooter_does_not_pause():
    """Opposition player behind the shooter is not on the shot segment → shot fires."""
    pitch = Pitch.standard()
    aim = pitch.right_goal_centre.with_z(1.0)
    shooter_pos = Vector3(0.0, 0.0, 0.0)

    # Behind the shooter (in the -x direction, away from goal)
    behind_pos = Vector3(-20.0, 0.0, 0.0)
    blocker = make_player("blocker", Team.RIGHT, position=behind_pos)

    match, shooter = _setup_match(shooter_pos, aim, extra_players=[blocker])
    shooter.current_order = ShootOrder(aim_point=aim, power_fraction=0.85, chance_of_pausing=1.0)
    match.step()

    assert match.ball.possessed_by is None, (
        "Shot should fire; blocker is behind the shooter, not on the shot line"
    )


# ---------------------------------------------------------------------------
# Blocker far to the side (outside threshold) does not trigger pause
# ---------------------------------------------------------------------------

def test_blocker_far_side_does_not_pause():
    """Opposition player more than 1 m perpendicular from the shot line → shot fires."""
    pitch = Pitch.standard()
    aim = pitch.right_goal_centre.with_z(1.0)
    shooter_pos = Vector3(0.0, 0.0, 0.0)

    # 5 m to the side on the halfway line
    side_pos = Vector3(pitch.half_length / 2.0, 5.0, 0.0)
    blocker = make_player("blocker", Team.RIGHT, position=side_pos)

    match, shooter = _setup_match(shooter_pos, aim, extra_players=[blocker])
    shooter.current_order = ShootOrder(aim_point=aim, power_fraction=0.85, chance_of_pausing=1.0)
    match.step()

    assert match.ball.possessed_by is None, (
        "Shot should fire; blocker is 5 m to the side of the shot line"
    )


# ---------------------------------------------------------------------------
# Inactive (tackled) blocker is ignored
# ---------------------------------------------------------------------------

def test_inactive_blocker_ignored():
    """An INACTIVE_TACKLED opposition player on the shot line does not trigger a pause."""
    from footballcoach.entities.player import PlayerState

    pitch = Pitch.standard()
    aim = pitch.right_goal_centre.with_z(1.0)
    shooter_pos = Vector3(0.0, 0.0, 0.0)

    blocker_pos = Vector3(pitch.half_length / 2.0, 0.0, 0.0)
    blocker = make_player("blocker", Team.RIGHT, position=blocker_pos)
    blocker.state = PlayerState.INACTIVE_TACKLED
    blocker.state_timer_s = 2.0

    match, shooter = _setup_match(shooter_pos, aim, extra_players=[blocker])
    shooter.current_order = ShootOrder(aim_point=aim, power_fraction=0.85, chance_of_pausing=1.0)
    match.step()

    assert match.ball.possessed_by is None, (
        "Inactive blocker should be ignored; shot should fire"
    )


# ---------------------------------------------------------------------------
# Teammate on the shot line does not trigger pause (only opposition matters)
# ---------------------------------------------------------------------------

def test_teammate_on_line_does_not_pause():
    """A teammate standing on the shot line does not count as a blocker."""
    pitch = Pitch.standard()
    aim = pitch.right_goal_centre.with_z(1.0)
    shooter_pos = Vector3(0.0, 0.0, 0.0)

    teammate_pos = Vector3(pitch.half_length / 2.0, 0.0, 0.0)
    teammate = make_player("teammate", Team.LEFT, position=teammate_pos)  # same team as shooter

    match, shooter = _setup_match(shooter_pos, aim, extra_players=[teammate])
    shooter.current_order = ShootOrder(aim_point=aim, power_fraction=0.85, chance_of_pausing=1.0)
    match.step()

    assert match.ball.possessed_by is None, (
        "Teammate on the shot line should not block; shot should fire"
    )


# ---------------------------------------------------------------------------
# Pause MoveOrder uses sprint=True (same as normal MoveOrder)
# ---------------------------------------------------------------------------

def test_pause_move_order_is_sprint():
    """The MoveOrder issued after a pause must use sprint=True."""
    pitch = Pitch.standard()
    aim = pitch.right_goal_centre.with_z(1.0)
    shooter_pos = Vector3(0.0, 0.0, 0.0)

    blocker_pos = Vector3(pitch.half_length / 2.0, 0.0, 0.0)
    blocker = make_player("blocker", Team.RIGHT, position=blocker_pos)

    match, shooter = _setup_match(shooter_pos, aim, extra_players=[blocker])
    shooter.current_order = ShootOrder(aim_point=aim, power_fraction=0.85, chance_of_pausing=1.0)
    match.step()

    assert isinstance(shooter.current_order, MoveOrder)
    assert shooter.current_order.sprint is True, "Pause MoveOrder must use sprint=True"

"""resolve_ball_block_by_inactive_players: the ray-circle block on inactive players."""
from __future__ import annotations

import pytest

from footballcoach.engine.collision import resolve_ball_block_by_inactive_players
from footballcoach.entities.attributes import PlayerAttributes
from footballcoach.entities.ball import Ball
from footballcoach.entities.player import Player, PlayerState, Team
from footballcoach.mathutils import Vector3

RESTITUTION = 0.35


def _player(x: float = 0.0, y: float = 0.0, inactive: bool = True, player_id: str = "p") -> Player:
    attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    player = Player.create(player_id, Team.LEFT, attrs, position=Vector3(x, y, 0))
    if inactive:
        player.state = PlayerState.INACTIVE_TACKLED
    return player


def _ball(start_end_velocity: tuple[Vector3, Vector3, Vector3]) -> tuple[Ball, Vector3]:
    start, end, velocity = start_end_velocity
    ball = Ball.at_rest(end)
    ball.velocity = velocity
    return ball, start


def _shot(y: float = 0.0, from_x: float = -2.0, to_x: float = 2.0, z: float = 0.5, vz: float = 2.0):
    return _ball((Vector3(from_x, y, z), Vector3(to_x, y, z), Vector3(10.0, 0.0, vz)))


def test_ball_crossing_in_stops_at_the_entry_point_and_deflects():
    player = _player()
    ball, previous = _shot()
    resolve_ball_block_by_inactive_players(ball, [player], previous, RESTITUTION)
    assert ball.position.x == pytest.approx(-player.radius_m)
    assert ball.position.y == pytest.approx(0.0) and ball.position.z == 0.5
    assert ball.velocity.x == pytest.approx(-10.0 * RESTITUTION)
    assert ball.velocity.z == pytest.approx(2.0 * RESTITUTION)


def test_ball_starting_inside_the_cylinder_is_not_blocked():
    player = _player()
    ball, previous = _shot(from_x=0.1)
    position_before = ball.position
    resolve_ball_block_by_inactive_players(ball, [player], previous, RESTITUTION)
    assert ball.position == position_before and ball.velocity.x == 10.0


def test_active_player_does_not_block():
    ball, previous = _shot()
    resolve_ball_block_by_inactive_players(ball, [_player(inactive=False)], previous, RESTITUTION)
    assert ball.position.x == 2.0 and ball.velocity.x == 10.0


def test_possessed_ball_is_ignored():
    ball, previous = _shot()
    ball.possessed_by = "someone"
    resolve_ball_block_by_inactive_players(ball, [_player()], previous, RESTITUTION)
    assert ball.position.x == 2.0


def test_stationary_ball_is_ignored():
    ball = Ball.at_rest(Vector3(-2.0, 0.0, 0.0))
    previous = Vector3(-2.0, 0.0, 0.0)
    resolve_ball_block_by_inactive_players(ball, [_player()], previous, RESTITUTION)
    assert ball.position.x == -2.0


def test_ball_passing_wide_of_the_circle_is_not_blocked():
    player = _player()
    ball, previous = _shot(y=player.radius_m + 0.01)
    resolve_ball_block_by_inactive_players(ball, [player], previous, RESTITUTION)
    assert ball.position.x == 2.0 and ball.velocity.x == 10.0


def test_exact_tangent_counts_as_a_hit_at_the_tangent_point():
    """discriminant == 0: the path touches the circle at one point directly above the centre."""
    player = _player()
    ball, previous = _shot(y=player.radius_m)
    resolve_ball_block_by_inactive_players(ball, [player], previous, RESTITUTION)
    assert ball.position.x == pytest.approx(0.0) and ball.position.y == pytest.approx(player.radius_m)
    assert ball.velocity.x == pytest.approx(10.0 * RESTITUTION)
    assert ball.velocity.y == pytest.approx(0.0)


def test_grazing_hit_just_inside_the_tangent_deflects_mostly_forward():
    player = _player()
    ball, previous = _shot(y=player.radius_m - 1e-4)
    resolve_ball_block_by_inactive_players(ball, [player], previous, RESTITUTION)
    assert ball.position.x < 0.0 and ball.velocity.x > 0.0


def test_ball_stopping_short_of_the_circle_is_not_blocked():
    player = _player()
    ball, previous = _shot(to_x=-player.radius_m - 0.05)
    resolve_ball_block_by_inactive_players(ball, [player], previous, RESTITUTION)
    assert ball.position.x == pytest.approx(-player.radius_m - 0.05)


def test_ball_moving_away_is_not_blocked():
    player = _player()
    ball, previous = _ball((Vector3(1.0, 0, 0.5), Vector3(2.0, 0, 0.5), Vector3(10.0, 0, 0)))
    resolve_ball_block_by_inactive_players(ball, [player], previous, RESTITUTION)
    assert ball.position.x == 2.0


def test_ball_stopping_just_short_is_blocked_on_the_next_tick():
    player = _player()
    just_outside = -player.radius_m - 1e-6
    ball, previous = _shot(to_x=just_outside)
    resolve_ball_block_by_inactive_players(ball, [player], previous, RESTITUTION)
    assert ball.velocity.x == 10.0

    ball.position = Vector3(1.0, 0.0, 0.5)
    resolve_ball_block_by_inactive_players(ball, [player], Vector3(just_outside, 0.0, 0.5), RESTITUTION)
    assert ball.velocity.x == pytest.approx(-10.0 * RESTITUTION)


def test_only_the_first_listed_inactive_player_blocks():
    near, far = _player(x=0.0, player_id="near"), _player(x=1.0, player_id="far")
    ball, previous = _shot()
    resolve_ball_block_by_inactive_players(ball, [far, near], previous, RESTITUTION)
    assert ball.position.x == pytest.approx(1.0 - far.radius_m)

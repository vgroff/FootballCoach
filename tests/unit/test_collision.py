from __future__ import annotations

import math

from footballcoach.engine.collision import (
    CollisionParams,
    _damp_overlap_velocity,
    are_touching,
    resolve_all_overlaps,
    resolve_player_overlap,
)
from footballcoach.entities.player import PlayerState
from footballcoach.mathutils import Vector3
from tests.conftest import make_player


# --------------------------------------------------------------------------
# Phase D: velocity damping tests
# --------------------------------------------------------------------------

def _make_collision_params(retention: float = 0.5, floor: float = 0.3) -> CollisionParams:
    return CollisionParams(
        collision_velocity_retention=retention,
        collision_damping_min_closing_speed_mps=floor,
    )


def test_head_on_closing_velocity_reduced_by_half():
    """Two players closing at 3.0 m/s (per worked example): first tick should
    reduce closing speed to ~1.5 m/s (50% retention)."""
    params = _make_collision_params()
    p1 = make_player("p1", position=Vector3(0, 0, 0))
    p2 = make_player("p2", position=Vector3(0.3, 0, 0))  # overlapping
    p1.velocity = Vector3(1.5, 0, 0)   # toward p2
    p2.velocity = Vector3(-1.5, 0, 0)  # toward p1  (total closing = 3.0 m/s)

    _damp_overlap_velocity(p1, p2, params)

    # After damping, p1's velocity in the +x direction (toward p2) should be halved
    assert abs(p1.velocity.x - 0.75) < 1e-6
    # p2's velocity in the -x direction (toward p1) should be halved
    assert abs(p2.velocity.x - (-0.75)) < 1e-6


def test_closing_speed_below_floor_not_damped():
    """Closing speed of 0.25 m/s is below the 0.3 floor — no damping."""
    params = _make_collision_params()
    p1 = make_player("p1", position=Vector3(0, 0, 0))
    p2 = make_player("p2", position=Vector3(0.3, 0, 0))
    p1.velocity = Vector3(0.125, 0, 0)
    p2.velocity = Vector3(-0.125, 0, 0)

    _damp_overlap_velocity(p1, p2, params)

    assert abs(p1.velocity.x - 0.125) < 1e-9
    assert abs(p2.velocity.x - (-0.125)) < 1e-9


def test_closing_speed_just_above_floor_is_damped():
    """A player's individual closing component of 0.35 m/s (above the 0.3
    floor) should be damped; p2 stationary so only p1 gets damped."""
    params = _make_collision_params()
    p1 = make_player("p1", position=Vector3(0, 0, 0))
    p2 = make_player("p2", position=Vector3(0.3, 0, 0))  # overlapping in x
    p1.velocity = Vector3(0.35, 0, 0)   # p1 closing toward p2 at 0.35 > 0.3 floor
    p2.velocity = Vector3.zero()        # p2 stationary

    _damp_overlap_velocity(p1, p2, params)

    # p1's closing component should be halved: 0.35 * 0.5 = 0.175
    assert abs(p1.velocity.x - 0.175) < 1e-6
    # p2 stays zero (closing_b = 0 < floor)
    assert abs(p2.velocity.x) < 1e-9


def test_tangential_motion_unaffected():
    """Two players moving side-by-side (zero closing component): velocities unchanged."""
    params = _make_collision_params()
    p1 = make_player("p1", position=Vector3(0, 0, 0))
    p2 = make_player("p2", position=Vector3(0.3, 0, 0))  # overlapping in x
    # Both moving purely in +y (tangential to the x-axis collision normal)
    p1.velocity = Vector3(0, 5.0, 0)
    p2.velocity = Vector3(0, 5.0, 0)

    _damp_overlap_velocity(p1, p2, params)

    assert abs(p1.velocity.y - 5.0) < 1e-9
    assert abs(p2.velocity.y - 5.0) < 1e-9
    assert abs(p1.velocity.x) < 1e-9
    assert abs(p2.velocity.x) < 1e-9


def test_no_damping_when_not_overlapping():
    """No overlap → damping must not apply even at high closing speed."""
    params = _make_collision_params()
    p1 = make_player("p1", position=Vector3(0, 0, 0))
    p2 = make_player("p2", position=Vector3(5.0, 0, 0))  # far apart
    p1.velocity = Vector3(10.0, 0, 0)
    p2.velocity = Vector3(-10.0, 0, 0)

    _damp_overlap_velocity(p1, p2, params)

    assert abs(p1.velocity.x - 10.0) < 1e-9
    assert abs(p2.velocity.x - (-10.0)) < 1e-9


def test_inactive_player_still_gets_velocity_damped():
    """Inactive (just-tackled) players are excluded from position push-apart
    but MUST still have their closing velocity damped — prevents gliding at
    full speed into an opponent after a tackle."""
    params = _make_collision_params()
    p1 = make_player("p1", position=Vector3(0, 0, 0))
    p2 = make_player("p2", position=Vector3(0.3, 0, 0))
    p1.velocity = Vector3(1.5, 0, 0)
    p2.velocity = Vector3(-1.5, 0, 0)
    # Mark p2 as inactive
    p2.state = PlayerState.INACTIVE_TACKLED
    p2.state_timer_s = 0.5

    # Position push-apart is skipped for inactive pairs in resolve_all_overlaps,
    # but velocity damping (_damp_overlap_velocity) should still apply.
    _damp_overlap_velocity(p1, p2, params)

    assert abs(p1.velocity.x - 0.75) < 1e-6  # halved
    assert abs(p2.velocity.x - (-0.75)) < 1e-6  # halved


def test_resolve_all_overlaps_position_pushes_active_pairs_only():
    """Position push-apart is skipped for inactive pairs, but velocity
    damping is applied for all pairs via resolve_all_overlaps."""
    params = _make_collision_params()
    p1 = make_player("p1", position=Vector3(0, 0, 0))
    p2 = make_player("p2", position=Vector3(0.3, 0, 0))  # overlapping
    p2.state = PlayerState.INACTIVE_TACKLED
    p2.state_timer_s = 0.5
    p1.velocity = Vector3(3.0, 0, 0)  # rushing into inactive p2
    p2.velocity = Vector3.zero()

    initial_p1_pos = p1.position
    initial_p2_pos = p2.position

    resolve_all_overlaps([p1, p2], collision_params=params)

    # Position: inactive pair is NOT pushed apart
    assert p1.position == initial_p1_pos
    assert p2.position == initial_p2_pos

    # Velocity: p1's closing speed (3.0 > 0.3 floor) should be damped
    assert p1.velocity.x < 3.0


def test_multi_tick_compounding_self_limits():
    """Running the damping for multiple ticks verifies the floor prevents
    unbounded speed reduction, matching the worked example (self-limits ~4 ticks)."""
    params = _make_collision_params(retention=0.5, floor=0.3)
    p1 = make_player("p1", position=Vector3(0, 0, 0))
    p2 = make_player("p2", position=Vector3(0.3, 0, 0))
    p1.velocity = Vector3(3.0, 0, 0)  # initial closing: 3.0 m/s
    p2.velocity = Vector3.zero()

    speed_history = []
    for _ in range(8):
        _damp_overlap_velocity(p1, p2, params)
        speed_history.append(p1.velocity.x)

    # Speed should monotonically decrease or plateau at/below the floor
    for i in range(1, len(speed_history)):
        assert speed_history[i] <= speed_history[i - 1] + 1e-9

    # Speed must not reach zero — floor stops it
    assert speed_history[-1] >= 0.0
    # After enough ticks the speed should be well below the initial 3.0 m/s
    assert speed_history[-1] < 1.0


def test_no_overlap_no_change():
    p1 = make_player("p1", position=Vector3(0, 0, 0))
    p2 = make_player("p2", position=Vector3(10, 0, 0))
    original_p1, original_p2 = p1.position, p2.position
    resolve_player_overlap(p1, p2)
    assert p1.position == original_p1
    assert p2.position == original_p2


def test_overlap_is_resolved_to_min_distance():
    p1 = make_player("p1", position=Vector3(0, 0, 0))
    p2 = make_player("p2", position=Vector3(0.3, 0, 0))  # well within combined radius 0.6
    resolve_player_overlap(p1, p2)
    distance = p1.position.distance_to(p2.position)
    assert abs(distance - (p1.radius_m + p2.radius_m)) < 1e-6


def test_faster_player_pushes_slower_player_more():
    p1 = make_player("p1", position=Vector3(0, 0, 0))
    p1.velocity = Vector3(5.0, 0, 0)  # charging into p2
    p2 = make_player("p2", position=Vector3(0.3, 0, 0))  # stationary
    resolve_player_overlap(p1, p2)
    # p2 (stationary, being charged into) should move further than p1.
    p1_moved = abs(p1.position.x - 0.0)
    p2_moved = abs(p2.position.x - 0.3)
    assert p2_moved > p1_moved


def test_resolve_all_overlaps_handles_chain():
    players = [
        make_player("p1", position=Vector3(0, 0, 0)),
        make_player("p2", position=Vector3(0.3, 0, 0)),
        make_player("p3", position=Vector3(0.6, 0, 0)),
    ]
    resolve_all_overlaps(players, iterations=20)
    d12 = players[0].position.distance_to(players[1].position)
    d23 = players[1].position.distance_to(players[2].position)
    min_dist = players[0].radius_m + players[1].radius_m
    assert d12 >= min_dist - 1e-3
    assert d23 >= min_dist - 1e-3


def test_are_touching():
    p1 = make_player("p1", position=Vector3(0, 0, 0))
    p2 = make_player("p2", position=Vector3(0.55, 0, 0))  # within radius sum (0.6) + tolerance
    p3 = make_player("p3", position=Vector3(5.0, 0, 0))
    assert are_touching(p1, p2)
    assert not are_touching(p1, p3)

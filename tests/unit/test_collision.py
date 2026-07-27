from __future__ import annotations

from footballcoach.engine.collision import are_touching, resolve_all_overlaps, resolve_player_overlap
from footballcoach.mathutils import Vector3
from tests.conftest import make_player


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

from __future__ import annotations

from footballcoach.entities.pitch import Pitch
from footballcoach.mathutils import Vector3


def test_standard_pitch_dimensions():
    pitch = Pitch.standard()
    assert pitch.length_m == 105.0
    assert pitch.width_m == 68.0


def test_penalty_spot_distance():
    pitch = Pitch.standard()
    left_spot = pitch.penalty_spot(left=True)
    right_spot = pitch.penalty_spot(left=False)
    assert left_spot.x == -pitch.half_length + pitch.penalty_spot_distance_m
    assert right_spot.x == pitch.half_length - pitch.penalty_spot_distance_m


def test_is_in_box():
    pitch = Pitch.standard()
    inside = Vector3(-pitch.half_length + 5, 0, 0)
    outside = Vector3(0, 0, 0)
    assert pitch.is_in_box(inside, left=True)
    assert not pitch.is_in_box(outside, left=True)


def test_is_in_bounds():
    pitch = Pitch.standard()
    assert pitch.is_in_bounds(Vector3(0, 0, 0))
    assert not pitch.is_in_bounds(Vector3(pitch.half_length + 1, 0, 0))

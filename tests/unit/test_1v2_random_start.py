"""build_1v2_scenario(random_initial_velocity=True): every player starts moving
in a random direction (heading = that direction) at up to its own top speed,
from its own seeded stream (positions unchanged for a seed)."""
from __future__ import annotations

import math

from footballcoach.ui.scenarios import build_1v2_scenario


def test_default_keeps_the_old_start():
    m = build_1v2_scenario(seed=3)
    assert m.player_by_id("defender").velocity.length() == 0.0
    assert m.player_by_id("keeper").velocity.length() == 0.0


def test_random_velocity_moves_everyone_along_their_heading():
    moving = 0
    for s in range(30):
        m = build_1v2_scenario(seed=s, random_initial_velocity=True)
        for p in m.players:
            v = p.velocity
            if v.length() > 1e-6:
                moving += 1
                assert math.isclose(math.atan2(v.y, v.x), p.heading_rad, abs_tol=1e-9)
    assert moving > 80          # 90 players, speeds uniform from 0


def test_random_velocity_keeps_the_seed_positions():
    for s in range(20):
        a = build_1v2_scenario(seed=s)
        b = build_1v2_scenario(seed=s, random_initial_velocity=True)
        for p in a.players:
            assert b.player_by_id(p.player_id).position == p.position


def test_random_velocity_is_seeded():
    a = build_1v2_scenario(seed=7, random_initial_velocity=True)
    b = build_1v2_scenario(seed=7, random_initial_velocity=True)
    assert [p.velocity for p in a.players] == [p.velocity for p in b.players]


def test_keeper_off_line_jitter_moves_only_the_keeper_both_ways():
    pitch_x = build_1v2_scenario(seed=0).player_by_id("keeper").position.x   # on the line
    xs = []
    for s in range(60):
        a = build_1v2_scenario(seed=s)
        b = build_1v2_scenario(seed=s, gk_start_x_jitter_m=0.5)
        ka, kb = a.player_by_id("keeper"), b.player_by_id("keeper")
        assert kb.position.y == ka.position.y                        # sideways draw unchanged
        assert abs(kb.position.x - ka.position.x) <= 0.5
        for pid in ("attacker", "defender"):
            assert b.player_by_id(pid).position == a.player_by_id(pid).position
        xs.append(kb.position.x - pitch_x)
    assert min(xs) < -0.1 and max(xs) > 0.1                           # in front of AND behind the line

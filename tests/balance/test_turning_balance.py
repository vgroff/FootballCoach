"""Balance test: the turning/direction-change mechanic.

Scenario (per user's approved adjustment to the original spec, since turning
at rest is "free"): two identical players start at the SAME point, both
already moving at top speed in the +x direction. At t=0, player A is ordered
to reverse (target -x) while player B keeps moving in +x. We compare how far
A ends up from B's position by the time A has actually turned around and is
moving purely in -x - this stresses the turn-rate-limited-speed model (a
sharp reversal should cost real time/distance, not be instantaneous).
"""
from __future__ import annotations

import math
import random

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch
from footballcoach.mathutils import Vector3
from footballcoach.orders import MoveOrder
from tests.conftest import make_player

RNG_REDUCTION = 0.3


def _run_reversal_scenario(acceleration_attr: float) -> dict:
    pitch = Pitch.standard()
    start = Vector3(0, 0, 0)

    player_a = make_player("a", position=start, top_speed=0.6, acceleration=acceleration_attr, stamina=1.0)
    player_b = make_player("b", position=start, top_speed=0.6, acceleration=acceleration_attr, stamina=1.0)
    player_a.heading_rad = 0.0
    player_b.heading_rad = 0.0
    player_a.velocity = Vector3(player_a.velocity.length() or 6.0, 0, 0)

    # Get both players up to top speed heading +x first (not timed).
    ball = Ball.at_rest(Vector3(0, 20, 0))
    match = Match(pitch=pitch, players=[player_a, player_b], ball=ball, rng_reduction=RNG_REDUCTION, rng=random.Random(0))
    for p in (player_a, player_b):
        p.current_order = MoveOrder(target_position=p.position + Vector3(100, 0, 0), sprint=True)
    for _ in range(90):  # 3s to reach top speed
        match.step()

    # Reset positions to a common reference point but keep velocities (both
    # now at their shared top speed, heading +x).
    reference_velocity = player_a.velocity
    player_a.position = Vector3(0, 0, 0)
    player_b.position = Vector3(0, 0, 0)

    # Now: player A reverses direction, player B continues.
    player_a.current_order = MoveOrder(target_position=Vector3(-100, 0, 0), sprint=True)
    player_b.current_order = MoveOrder(target_position=Vector3(100, 0, 0), sprint=True)

    ticks_to_reverse = None
    for i in range(300):
        match.step()
        if ticks_to_reverse is None and player_a.velocity.x < 0:
            ticks_to_reverse = i + 1

    time_to_reverse_s = ticks_to_reverse * match.dt_s if ticks_to_reverse else None
    separation_m = player_a.position.distance_to(player_b.position)
    elapsed_s = 300 * match.dt_s

    return {
        "reference_speed_mps": round(reference_velocity.length(), 2),
        "time_to_start_reversing_s": round(time_to_reverse_s, 3) if time_to_reverse_s else None,
        f"separation_after_{elapsed_s:.0f}s_m": round(separation_m, 2),
    }


def test_reversal_costs_meaningful_time_and_distance(balance_recorder):
    stats = _run_reversal_scenario(acceleration_attr=0.5)
    balance_recorder.report("turning_reversal_scenario_accel_0.5", stats)
    # The turn/reversal should not be instantaneous (turn-rate-limited), so
    # some non-trivial time passes before the player is actually moving
    # backwards, and a meaningful separation builds up vs. the player who
    # kept going straight.
    assert stats["time_to_start_reversing_s"] is not None
    assert stats["time_to_start_reversing_s"] > 0.05
    assert stats["separation_after_10s_m"] > 5.0


def test_higher_acceleration_attribute_turns_around_faster(balance_recorder):
    low_accel = _run_reversal_scenario(acceleration_attr=0.1)
    high_accel = _run_reversal_scenario(acceleration_attr=0.9)
    balance_recorder.report(
        "turning_reversal_accel_comparison",
        {"low_accel_attr_0.1": low_accel, "high_accel_attr_0.9": high_accel},
    )
    assert high_accel["time_to_start_reversing_s"] < low_accel["time_to_start_reversing_s"]

"""Balance tests: sprint speed/time sanity checks for top/bottom attribute
players, with and without the ball. rng_reduction is irrelevant here since
movement itself is deterministic (no RNG in movement.py), but we still run
via the Match engine at the default 0.3 for consistency with real usage.

Targets are derived from real-world sprint intuition rather than a number
the user specified directly: a top real athlete covers ~100m in under 11s
(~9.1 m/s average incl. acceleration), our top_speed=1.0 attr gives a
theoretical max cruising speed of 9.5 m/s (5.0 + 4.5), which is a very fast
but plausible football sprint speed. A bottom-tier attr=0.0 gives 5.0 m/s,
a brisk jog, still "competent" per the design brief.
"""
from __future__ import annotations

import random

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import MoveOrder
from tests.conftest import make_player

RNG_REDUCTION = 0.3
SPRINT_DISTANCE_M = 100.0


def _time_to_run(top_speed_attr: float, acceleration_attr: float, has_ball: bool, ball_control_attr: float = 0.5) -> float:
    pitch = Pitch.standard()
    player = make_player(
        "p1", position=Vector3(-pitch.half_length + 1, 0, 0),
        top_speed=top_speed_attr, acceleration=acceleration_attr,
        ball_control=ball_control_attr, stamina=1.0,
    )
    if has_ball:
        ball = Ball.at_rest(player.position)
        ball.possessed_by = player.player_id
    else:
        # Keep the ball well away from the player so it isn't auto-picked-up
        # mid-run (which would otherwise stall the "no ball" baseline with a
        # spurious control-time delay).
        ball = Ball.at_rest(Vector3(player.position.x, 20.0, 0.0))
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=RNG_REDUCTION, rng=random.Random(0))

    target = player.position + Vector3(SPRINT_DISTANCE_M, 0, 0)
    player.current_order = MoveOrder(target_position=target, sprint=True, arrival_tolerance_m=0.3)

    max_ticks = 30 * 60  # 60s cap
    for i in range(max_ticks):
        match.step()
        if player.current_order is None:
            return (i + 1) * match.dt_s
    return float("inf")  # did not arrive - flagged by test assertion


def test_top_attribute_player_100m_sprint_time(balance_recorder):
    t = _time_to_run(top_speed_attr=1.0, acceleration_attr=1.0, has_ball=False)
    balance_recorder.report("sprint_100m_top_attrs_no_ball_seconds", {"time_s": round(t, 2)})
    # 100m at up to 9.5 m/s cruising speed, with acceleration ramp-up, should
    # land comfortably in an elite-sprinter-ish range once you allow for
    # accel time; wide band since this is a sanity check not a hard target.
    assert 10.0 < t < 18.0


def test_bottom_attribute_player_100m_sprint_time(balance_recorder):
    t = _time_to_run(top_speed_attr=0.0, acceleration_attr=0.0, has_ball=False)
    balance_recorder.report("sprint_100m_bottom_attrs_no_ball_seconds", {"time_s": round(t, 2)})
    assert 18.0 < t < 35.0


def test_top_attribute_player_is_faster_than_bottom(balance_recorder):
    t_top = _time_to_run(top_speed_attr=1.0, acceleration_attr=1.0, has_ball=False)
    t_bottom = _time_to_run(top_speed_attr=0.0, acceleration_attr=0.0, has_ball=False)
    balance_recorder.report(
        "sprint_100m_top_vs_bottom_seconds",
        {"top_attrs_s": round(t_top, 2), "bottom_attrs_s": round(t_bottom, 2)},
    )
    assert t_top < t_bottom


def test_dribbling_with_ball_is_slower_than_without(balance_recorder):
    t_no_ball = _time_to_run(top_speed_attr=0.6, acceleration_attr=0.6, has_ball=False)
    t_with_ball = _time_to_run(top_speed_attr=0.6, acceleration_attr=0.6, has_ball=True, ball_control_attr=0.6)
    balance_recorder.report(
        "sprint_100m_with_vs_without_ball_seconds",
        {"no_ball_s": round(t_no_ball, 2), "with_ball_s": round(t_with_ball, 2)},
    )
    assert t_with_ball > t_no_ball


def test_ball_control_reduces_but_never_eliminates_dribble_penalty(balance_recorder):
    t_low_control = _time_to_run(top_speed_attr=0.6, acceleration_attr=0.6, has_ball=True, ball_control_attr=0.0)
    t_high_control = _time_to_run(top_speed_attr=0.6, acceleration_attr=0.6, has_ball=True, ball_control_attr=1.0)
    t_no_ball = _time_to_run(top_speed_attr=0.6, acceleration_attr=0.6, has_ball=False)
    balance_recorder.report(
        "sprint_100m_ball_control_effect_seconds",
        {
            "no_ball_s": round(t_no_ball, 2),
            "with_ball_low_control_s": round(t_low_control, 2),
            "with_ball_high_control_s": round(t_high_control, 2),
        },
    )
    assert t_high_control < t_low_control
    assert t_high_control > t_no_ball  # never fully eliminated, per design spec

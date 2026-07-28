"""Balance tests for Phase C: GK jumping (jump height penalty on control time).

Validates that:
1. GK control time at ground-level vs above-head-height balls differs measurably.
2. GK has advantage over outfield above head height.
3. Early intercept reduces average GK travel distance on close slow shots.
"""
from __future__ import annotations

import random

from footballcoach.engine.goalkeeping import GoalkeepingParams, early_intercept_target, save_target_position
from footballcoach.engine.possession import ControlTimeParams, control_time_s
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3

N_TRIALS = 2000


def test_gk_control_time_ground_vs_above_head(balance_recorder):
    """GK's control time at head-height-plus balls is measurably longer than
    at ground-level balls, confirming the jump zone penalty activates."""
    params = ControlTimeParams.from_config()

    heights = [0.0, 0.5, 1.0, 1.5, 1.8, 2.0, 2.2]
    results = {}
    for h in heights:
        t = control_time_s(
            params, ball_height_m=h, relative_speed_mps=0.0,
            player_speed_mps=0.0, ball_control_attr=0.7, is_goalkeeper_in_box=True,
        )
        results[f"gk_h={h:.1f}m"] = round(t, 4)

    balance_recorder.report("gk_control_time_by_height", results)

    t_ground = control_time_s(params, 0.0, 0.0, 0.0, 0.7, is_goalkeeper_in_box=True)
    t_jump = control_time_s(params, 2.0, 0.0, 0.0, 0.7, is_goalkeeper_in_box=True)
    assert t_jump > t_ground * 1.2, (
        f"Expected jump time ({t_jump:.4f}s) to be at least 20% longer than "
        f"ground time ({t_ground:.4f}s)"
    )


def test_outfield_control_time_ground_vs_above_head(balance_recorder):
    """Outfield player control time above head height is steeper than GK at same height."""
    params = ControlTimeParams.from_config()

    heights = [0.0, 0.5, 1.0, 1.5, 1.8, 1.9, 2.0]
    gk_results = {}
    of_results = {}
    for h in heights:
        t_gk = control_time_s(params, h, 0.0, 0.0, 0.7, is_goalkeeper_in_box=True)
        t_of = control_time_s(params, h, 0.0, 0.0, 0.7, is_goalkeeper_in_box=False)
        gk_results[f"h={h:.1f}m"] = round(t_gk, 4)
        of_results[f"h={h:.1f}m"] = round(t_of, 4)

    balance_recorder.report("gk_vs_outfield_control_time_by_height", {
        "gk": gk_results,
        "outfield": of_results,
    })

    # At 2.0m (within both GK's 2.2m and outfield's 2.0m reach), GK must be faster
    t_gk_2 = control_time_s(params, 2.0, 0.0, 0.0, 0.7, is_goalkeeper_in_box=True)
    t_of_2 = control_time_s(params, 2.0, 0.0, 0.0, 0.7, is_goalkeeper_in_box=False)
    assert t_gk_2 < t_of_2, (
        f"GK should be faster than outfield at 2.0m jump height: "
        f"GK={t_gk_2:.4f}s, outfield={t_of_2:.4f}s"
    )


def test_early_intercept_reduces_travel_distance_on_close_shots(balance_recorder):
    """Over a batch of random close shots (ball within 10m of GK), early intercept
    should on average result in a shorter GK travel distance than the default
    goal-line target, confirming it actually helps the GK get to the ball faster."""
    pitch = Pitch.standard()
    params = GoalkeepingParams.from_config()
    rng = random.Random(42)

    n = 200
    intercept_dists = []
    goal_line_dists = []

    for _ in range(n):
        # Random GK position near own goal
        gk_pos = Vector3(
            -pitch.half_length + rng.uniform(0.5, 2.0),
            rng.uniform(-2.0, 2.0),
            0,
        )
        # Ball within early_intercept_max_distance_m (10m), heading toward goal
        ball_distance = rng.uniform(2.0, 9.0)
        ball_pos = gk_pos + Vector3(ball_distance, rng.uniform(-1.0, 1.0), rng.uniform(0.2, 1.5))
        ball_vel = Vector3(-rng.uniform(5.0, 15.0), rng.uniform(-1.0, 1.0), rng.uniform(-0.5, 0.5))
        gk_top_speed = 7.0

        intercept = early_intercept_target(
            gk_position=gk_pos,
            gk_effective_top_speed_mps=gk_top_speed,
            ball_position=ball_pos,
            ball_velocity=ball_vel,
            pitch=pitch,
            team=Team.LEFT,
            gravity_mps2=9.81,
            params=params,
        )
        goal_line_tgt = save_target_position(pitch, Team.LEFT, ball_pos, ball_vel, 9.81, params)

        goal_line_dist = gk_pos.xy().distance_to(goal_line_tgt.xy())
        goal_line_dists.append(goal_line_dist)

        if intercept is not None:
            intercept_dist = gk_pos.xy().distance_to(intercept.xy())
            intercept_dists.append(intercept_dist)

    n_intercepted = len(intercept_dists)
    avg_intercept_dist = sum(intercept_dists) / max(n_intercepted, 1)
    avg_goal_line_dist = sum(goal_line_dists[:n_intercepted]) / max(n_intercepted, 1)

    balance_recorder.report("early_intercept_travel_distance", {
        "n_trials": n,
        "n_with_early_intercept": n_intercepted,
        "avg_intercept_target_dist_m": round(avg_intercept_dist, 3),
        "avg_goal_line_dist_m": round(avg_goal_line_dist, 3),
    })

    # When early intercept fires, it must result in strictly shorter travel
    if n_intercepted > 0:
        assert avg_intercept_dist < avg_goal_line_dist, (
            f"Early intercept ({avg_intercept_dist:.3f}m) should be closer than "
            f"goal-line target ({avg_goal_line_dist:.3f}m)"
        )

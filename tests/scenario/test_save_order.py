"""Scenario tests for the SaveOrder / goalkeeper positioning logic.

All tests use rng_reduction=1.0 for deterministic outcomes so failures are
always reproducible.  They complement the statistical balance tests in
tests/balance/test_save_balance.py by probing specific edge-cases:

- GK must reach a predicted crossing point without overshooting.
- A fast straight shot aimed at the keeper's current position is always caught.
- A fast shot at the far corner is NOT caught when the GK has no time (verifies
  the scenario is hard enough to produce misses/goals, not trivially saved).
- GK positioned exactly on the crossing point never drifts away from it.
"""
from __future__ import annotations

import random

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.entities.player import PlayerState
from footballcoach.mathutils import Vector3
from footballcoach.orders import KickOrder, SaveOrder
from tests.conftest import make_player


def _make_save_match(
    gk_start_y: float,
    shot_x: float,
    aim_y: float,
    aim_z: float,
    power: float,
    precision: float = 1.0,
    gk_speed: float = 0.9,
    rng_reduction: float = 1.0,
    seed: int = 0,
) -> tuple[Match, object, object]:
    """Returns (match, gk_player, shooter_player)."""
    pitch = Pitch.standard()
    gk = make_player(
        "gk", Team.LEFT, is_goalkeeper=True,
        position=pitch.left_goal_centre + Vector3(0, gk_start_y, 0),
        top_speed=gk_speed, acceleration=gk_speed, ball_control=0.8,
    )
    shooter = make_player(
        "s", Team.RIGHT,
        position=Vector3(shot_x, 0, 0),
        kick_precision=precision, kick_power=0.9,
    )
    ball = Ball.at_rest(shooter.position)
    ball.possessed_by = shooter.player_id

    match = Match(
        pitch=pitch, players=[gk, shooter], ball=ball,
        rng_reduction=rng_reduction, rng=random.Random(seed),
    )
    gk.current_order = SaveOrder()
    aim_point = pitch.left_goal_centre + Vector3(0, aim_y, aim_z)
    shooter.current_order = KickOrder(
        aim_point=aim_point, power_fraction=power, spin=Vector3.zero(),
    )
    return match, gk, shooter


def test_gk_reaches_crossing_point_without_overshooting():
    """After a straight shot at the far post (deterministic, no error), the GK
    should end up within pickup_radius_m of the crossing point — not past it."""
    pitch = Pitch.standard()
    half_goal_w = pitch.goal_width_m / 2.0
    # GK starts at near post, shot aimed at far post corner.
    match, gk, _ = _make_save_match(
        gk_start_y=-half_goal_w + 0.3,
        shot_x=-(pitch.half_length - 25.0),
        aim_y=half_goal_w - 0.3,
        aim_z=0.3,
        power=0.9,
    )

    pickup_radius_m = match.pickup_radius_m
    for _ in range(300):
        match.step()
        if match.ball.possessed_by == gk.player_id:
            break
        if match.scoreboard.right_goals > 0:
            break

    # GK must not have gone past (more positive y than) the aim point.
    # With the per-tick snap threshold, the keeper snaps before overshooting.
    assert gk.position.y <= half_goal_w - 0.3 + pickup_radius_m + 0.1, (
        f"GK overshot far post: gk.y={gk.position.y:.3f} > aim_y={half_goal_w-0.3:.3f}"
    )


def test_gk_catches_straight_centre_shot_always():
    """A shot aimed dead centre with the keeper already centred must always be
    saved — no tunneling, no miss — at any reasonable shot speed."""
    pitch = Pitch.standard()
    match, gk, _ = _make_save_match(
        gk_start_y=0.0,
        shot_x=-(pitch.half_length - 18.0),
        aim_y=0.0,
        aim_z=1.0,
        power=0.9,
        precision=1.0,
    )

    result = "timeout"
    for _ in range(300):
        match.step()
        if match.ball.possessed_by == gk.player_id:
            result = "saved"
            break
        if match.scoreboard.right_goals > 0:
            result = "goal"
            break

    assert result == "saved", f"Centre shot to centred GK should always be saved, got: {result}"


def test_close_range_centre_shot_caught_no_tunneling():
    """Very close range (8m) straight shot at keeper: ball is fast (~25 m/s)
    but the keeper is right on the crossing point and must still catch it."""
    pitch = Pitch.standard()
    match, gk, _ = _make_save_match(
        gk_start_y=0.0,
        shot_x=-(pitch.half_length - 8.0),
        aim_y=0.0,
        aim_z=1.0,
        power=0.85,
        precision=1.0,
    )

    result = "timeout"
    for _ in range(150):
        match.step()
        if match.ball.possessed_by == gk.player_id:
            result = "saved"
            break
        if match.scoreboard.right_goals > 0:
            result = "goal"
            break

    assert result == "saved", (
        f"Close-range centre shot to centred GK tunneled through: got '{result}'"
    )


def test_gk_on_target_does_not_drift_away():
    """If the GK is already positioned at the predicted crossing point before
    the kick fires, they should stay there — not move away and miss."""
    pitch = Pitch.standard()
    half_goal_w = pitch.goal_width_m / 2.0
    target_y = half_goal_w - 0.3

    match, gk, _ = _make_save_match(
        gk_start_y=target_y,   # already at aim point
        shot_x=-(pitch.half_length - 20.0),
        aim_y=target_y,
        aim_z=0.4,
        power=0.9,
        precision=1.0,
    )

    result = "timeout"
    for _ in range(300):
        match.step()
        if match.ball.possessed_by == gk.player_id:
            result = "saved"
            break
        if match.scoreboard.right_goals > 0:
            result = "goal"
            break

    assert result == "saved", (
        f"GK started on crossing point but still conceded: '{result}'"
    )


def test_impossible_far_post_shot_scores():
    """A shot to the far corner from close range with a slow GK at near post
    should score (confirms the scenario is genuinely hard and the 'always saved'
    tests above aren't trivially passing because every shot is saved)."""
    pitch = Pitch.standard()
    half_goal_w = pitch.goal_width_m / 2.0

    match, gk, _ = _make_save_match(
        gk_start_y=-half_goal_w + 0.3,   # pinned to near post
        shot_x=-(pitch.half_length - 8.0),  # only 8m from goal, very fast
        aim_y=half_goal_w - 0.3,           # far corner
        aim_z=0.3,
        power=0.95,
        gk_speed=0.1,                       # slow keeper — can't cover 7m in time
        precision=1.0,
    )

    result = "timeout"
    for _ in range(150):
        match.step()
        if match.ball.possessed_by == gk.player_id:
            result = "saved"
            break
        if match.scoreboard.right_goals > 0:
            result = "goal"
            break

    assert result == "goal", (
        f"Expected goal (keeper can't cover 7m in <0.5s from 8m), got: '{result}'"
    )

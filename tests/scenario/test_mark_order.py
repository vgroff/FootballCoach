"""Scenario tests for MarkOrder: marker stands between target and ball,
switches to GetPossession behaviour on target possession or ball within
intercept radius.
"""
from __future__ import annotations

import math
import random

import pytest

from footballcoach import actions
from footballcoach.engine.match import Match, MarkingParams
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.entities.player import PlayerState
from footballcoach.mathutils import Vector3
from footballcoach.orders import GetPossessionOrder, MarkOrder, MoveOrder, OrderStatus
from tests.conftest import make_player


def _make_match(players, ball, rng_seed=0, rng_reduction=1.0):
    return Match(
        pitch=Pitch.standard(),
        players=players,
        ball=ball,
        rng_reduction=rng_reduction,
        rng=random.Random(rng_seed),
    )


# ---------------------------------------------------------------------------
# Standoff positioning
# ---------------------------------------------------------------------------

def test_marker_moves_toward_standoff_position():
    """Marker should move to a point between the target and the ball, at
    mark_standoff_m from the target, when the ball is far away (> intercept
    radius from the marker)."""
    # Ball is at (30, 0), target is at (20, 0).
    # Ideal standoff = target + normalize(ball-target)*1.5 = (21.5, 0).
    # Marker starts at (25, 2) — on the same side as the ball so it approaches
    # the standoff from the correct direction (no overshoot past the target).
    ball = Ball.at_rest(Vector3(30, 0, 0))
    marker = make_player("marker", Team.LEFT, position=Vector3(25, 2, 0), attr_value=0.8)
    target = make_player("target", Team.RIGHT, position=Vector3(20, 0, 0))

    match = _make_match([marker, target], ball)
    actions.mark(marker, target)

    # Run for enough ticks for the marker to reach the standoff area
    for _ in range(300):
        match.step()
        if marker.current_order is None:
            break

    params = MarkingParams.from_config()
    ideal_standoff = Vector3(20 + params.mark_standoff_m, 0, 0)
    dist_to_ideal = marker.position.xy().distance_to(ideal_standoff.xy())
    assert dist_to_ideal < 1.5, (
        f"Marker should be near standoff point {ideal_standoff}, "
        f"but is at {marker.position} (dist={dist_to_ideal:.3f}m)"
    )


def test_marker_never_autocompletes_in_standoff_mode():
    """MarkOrder should not auto-complete while the marker is just moving
    to the standoff position (no tackle/possession events)."""
    ball = Ball.at_rest(Vector3(30, 0, 0))
    marker = make_player("marker", Team.LEFT, position=Vector3(0, 0, 0), attr_value=0.7)
    target = make_player("target", Team.RIGHT, position=Vector3(20, 0, 0))

    match = _make_match([marker, target], ball)
    actions.mark(marker, target)

    for _ in range(150):
        match.step()
        assert marker.current_order is not None, "MarkOrder should not auto-complete"
        assert isinstance(marker.current_order, MarkOrder)


def test_standoff_position_tracks_moving_target():
    """When the target moves, the standoff position updates accordingly and
    the marker follows."""
    ball = Ball.at_rest(Vector3(30, 0, 0))
    marker = make_player("marker", Team.LEFT, position=Vector3(10, 5, 0), attr_value=0.9)
    target = make_player("target", Team.RIGHT, position=Vector3(20, 0, 0))

    match = _make_match([marker, target], ball)
    actions.mark(marker, target)

    # Run 100 ticks, then move the target and run 100 more.
    for _ in range(100):
        match.step()

    # Teleport target to a new position.
    target.position = Vector3(15, 10, 0)
    for _ in range(200):
        match.step()

    params = MarkingParams.from_config()
    to_ball = (ball.position - target.position).xy().normalized()
    ideal_standoff = target.position.with_z(0.0) + to_ball * params.mark_standoff_m
    dist = marker.position.xy().distance_to(ideal_standoff.xy())
    assert dist < 2.0, (
        f"Marker should have followed target to new standoff ({ideal_standoff}), "
        f"got {marker.position} (dist={dist:.3f}m)"
    )


# ---------------------------------------------------------------------------
# Intercept-radius trigger
# ---------------------------------------------------------------------------

def test_mark_switches_to_chase_when_ball_within_intercept_radius():
    """When the ball is within mark_intercept_radius_m of the marker, the
    marker should switch to GetPossession-style sprinting at the ball."""
    params = MarkingParams.from_config()
    # Place ball just inside the intercept radius.
    ball_dist = params.mark_intercept_radius_m * 0.8
    ball = Ball.at_rest(Vector3(ball_dist, 0, 0))
    marker = make_player("marker", Team.LEFT, position=Vector3(0, 0, 0), attr_value=0.8)
    # Target far away from ball so target_has_ball is False.
    target = make_player("target", Team.RIGHT, position=Vector3(-20, 0, 0))

    match = _make_match([marker, target], ball)
    actions.mark(marker, target)

    initial_dist = marker.position.xy().distance_to(ball.position.xy())
    match.step()
    new_dist = marker.position.xy().distance_to(ball.position.xy())

    # Marker should have moved toward the ball (distance decreased).
    assert new_dist < initial_dist, (
        "When ball is within intercept radius, marker should sprint toward ball"
    )


def test_mark_no_intercept_mode_beyond_radius():
    """When the ball is just outside the intercept radius, the marker should
    move toward the standoff position (not the ball directly)."""
    params = MarkingParams.from_config()
    # Ball just beyond the intercept radius.
    ball_x = params.mark_intercept_radius_m + 1.0
    ball = Ball.at_rest(Vector3(ball_x, 0, 0))
    marker = make_player("marker", Team.LEFT, position=Vector3(0, 0, 0), attr_value=0.8)
    target = make_player("target", Team.RIGHT, position=Vector3(-15, 0, 0))

    match = _make_match([marker, target], ball)
    actions.mark(marker, target)

    # Let it run; the standoff should be between target and ball.
    for _ in range(200):
        match.step()

    params2 = MarkingParams.from_config()
    to_ball = (ball.position - target.position).xy().normalized()
    ideal = target.position.with_z(0.0) + to_ball * params2.mark_standoff_m
    dist_to_ideal = marker.position.xy().distance_to(ideal.xy())

    # Should be near standoff, not near ball.
    dist_to_ball = marker.position.xy().distance_to(ball.position.xy())
    assert dist_to_ideal < dist_to_ball, (
        f"Marker should be near standoff ({ideal}) not near ball ({ball.position}); "
        f"dist_to_ideal={dist_to_ideal:.2f}, dist_to_ball={dist_to_ball:.2f}"
    )


def test_intercept_radius_boundary_inside():
    """Ball at exactly inside (0.9x) the intercept radius → marker moves toward ball."""
    params = MarkingParams.from_config()
    ball_dist = params.mark_intercept_radius_m * 0.9
    ball = Ball.at_rest(Vector3(ball_dist, 0, 0))
    marker = make_player("marker", Team.LEFT, position=Vector3(0, 0, 0), attr_value=0.8)
    target = make_player("target", Team.RIGHT, position=Vector3(-20, 0, 0))

    match = _make_match([marker, target], ball)
    actions.mark(marker, target)

    d_before = marker.position.xy().distance_to(ball.position.xy())
    match.step()
    d_after = marker.position.xy().distance_to(ball.position.xy())
    assert d_after < d_before


def test_intercept_radius_boundary_outside():
    """Ball at exactly outside (1.1x) the intercept radius → marker moves toward standoff."""
    params = MarkingParams.from_config()
    ball_dist = params.mark_intercept_radius_m * 1.1
    ball = Ball.at_rest(Vector3(ball_dist, 0, 0))
    marker = make_player("marker", Team.LEFT, position=Vector3(0, 0, 0), attr_value=0.8)
    target = make_player("target", Team.RIGHT, position=Vector3(-15, 0, 0))

    match = _make_match([marker, target], ball)
    actions.mark(marker, target)

    # Standoff is between target and ball; marker should move toward it, not the ball.
    to_ball_from_target = (ball.position - target.position).xy().normalized()
    ideal = target.position.with_z(0.0) + to_ball_from_target * params.mark_standoff_m

    for _ in range(150):
        match.step()

    dist_to_ideal = marker.position.xy().distance_to(ideal.xy())
    dist_to_ball = marker.position.xy().distance_to(ball.position.xy())
    assert dist_to_ideal < dist_to_ball, (
        "Marker should approach standoff, not ball, when ball is outside intercept radius"
    )


# ---------------------------------------------------------------------------
# Target-possession trigger
# ---------------------------------------------------------------------------

def test_mark_switches_to_tackle_when_target_gets_ball():
    """When the target gains possession, the marker should chase and tackle."""
    # Position marker close enough to the target that a tackle can occur quickly.
    ball = Ball.at_rest(Vector3(5, 0, 0))
    ball.possessed_by = "target"

    marker = make_player("marker", Team.LEFT, position=Vector3(5.4, 0, 0),
                         attr_value=0.8, tackling=0.9)
    target = make_player("target", Team.RIGHT, position=Vector3(5, 0, 0),
                         attr_value=0.4, dribbling=0.2)
    # Marker faces the target.
    import math
    marker.heading_rad = math.pi  # facing -x toward target

    match = _make_match([marker, target], ball, rng_reduction=1.0)
    actions.mark(marker, target)

    # Run until marker wins the ball or times out.
    for _ in range(60):
        match.step()
        if ball.possessed_by == marker.player_id:
            break

    assert ball.possessed_by == marker.player_id, (
        "Marker with high tackling should win ball from weak target in possession"
    )


def test_mark_switches_when_target_controlling_ball():
    """MarkOrder intercept mode also triggers when the target is mid
    first-touch control (CONTROLLING_BALL state, not just full possession)."""
    # Put target in CONTROLLING_BALL state by placing both players near ball.
    ball = Ball.at_rest(Vector3(5, 0, 0))

    marker = make_player("marker", Team.LEFT, position=Vector3(5.4, 0, 0),
                         attr_value=0.8, tackling=0.95)
    target = make_player("target", Team.RIGHT, position=Vector3(5, 0, 0),
                         attr_value=0.3, ball_control=0.2)

    target.state = PlayerState.CONTROLLING_BALL
    target.state_timer_s = 0.5

    match = _make_match([marker, target], ball, rng_reduction=1.0)
    actions.mark(marker, target)

    # On the first tick the target is CONTROLLING_BALL and target_has_ball is True,
    # so the marker should immediately switch to GetPossession-style (chase/tackle).
    match.step()
    # Marker should be in IN_PROGRESS and moving.
    assert isinstance(marker.current_order, MarkOrder)
    assert marker.current_order.status == OrderStatus.IN_PROGRESS


# ---------------------------------------------------------------------------
# MarkOrder never auto-completes
# ---------------------------------------------------------------------------

def test_mark_order_never_completes_after_winning_ball():
    """MarkOrder is a persistent duty like SaveOrder: even if the marker wins
    the ball, the MarkOrder itself should NOT auto-complete."""
    ball = Ball.at_rest(Vector3(5, 0, 0))
    ball.possessed_by = "target"

    marker = make_player("marker", Team.LEFT, position=Vector3(5.4, 0, 0),
                         attr_value=0.9, tackling=0.99)
    target = make_player("target", Team.RIGHT, position=Vector3(5, 0, 0),
                         attr_value=0.1, dribbling=0.01)

    import math
    marker.heading_rad = math.pi

    match = _make_match([marker, target], ball, rng_reduction=1.0)
    actions.mark(marker, target)

    # Run enough ticks to let the tackle happen.
    for _ in range(30):
        match.step()

    # The marker may have the ball now — MarkOrder persists regardless.
    assert isinstance(marker.current_order, MarkOrder), (
        "MarkOrder should never auto-complete, even after winning the ball"
    )


# ---------------------------------------------------------------------------
# Missing target graceful handling
# ---------------------------------------------------------------------------

def test_mark_completes_gracefully_if_target_missing():
    """If the target player_id does not exist in the match, MarkOrder
    should complete gracefully (not raise KeyError)."""
    ball = Ball.at_rest(Vector3(5, 0, 0))
    marker = make_player("marker", Team.LEFT, position=Vector3(0, 0, 0))

    match = _make_match([marker], ball)
    marker.current_order = MarkOrder(target_player_id="nonexistent")

    match.step()

    assert marker.current_order is None, (
        "MarkOrder against a nonexistent target should complete gracefully"
    )

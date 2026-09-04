"""Regression test for: a player who already HAS the ball runs it straight
out of bounds while executing a MoveOrder, with zero boundary awareness on
that path.

Real gap this reproduces
-------------------------
``Match._run_get_possession_behaviour``'s boundary-aware braking (see
``test_ball_out_overrun.py``) only ever runs while CHASING a loose ball.
Once a player actually has the ball, movement goes through ``MoveOrder``
(``steering.compute_repulsion`` via ``_compute_movement_intent``) instead --
a completely separate code path that, before this fix, had no concept of
the pitch boundary at all. Confirmed as a real traced episode: a player
picked up the ball 2m from the corner and sprinted dead straight through
the touchline a second later, despite plenty of room to have curved inward.

Fix (implemented): ``steering.compute_repulsion`` now also repels a ball
carrier away from the pitch boundary (``orders.json["repulsion"].
boundary_radius_m`` / ``boundary_strength_base``), reusing the same
blend/speed-penalty machinery already used for player-vs-player repulsion --
a steering correction (bend + slow down), not a kick.
"""
from __future__ import annotations

import random

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import MoveOrder
from footballcoach.rules_ai import StopWhenIdleAI
from tests.conftest import make_player


def _run_carry_toward_out_of_bounds_target(*, boundary_radius_m: float) -> Match:
    """Player already has the ball, 1m inside the sideline, running a
    MoveOrder whose straight-line target sits WELL past the boundary (y =
    half_width + 10) -- an unmodified straight-line path crosses the line
    almost immediately. Returns the match after 3s so the caller can check
    where the ball ended up."""
    pitch = Pitch.standard()
    start_y = pitch.half_width - 1.0
    start = Vector3(-10.0, start_y, 0.0)

    player = make_player("p1", Team.LEFT, attr_value=0.9, position=start)
    player.heading_rad = 0.0  # facing +x
    ball = Ball.at_rest(start)
    ball.possessed_by = "p1"

    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))
    # Aim well past the boundary in y -- an unsteered straight line drifts
    # out almost immediately from a 1m-inside-the-line start.
    player.current_order = MoveOrder(
        target_position=Vector3(10.0, pitch.half_width + 10.0, 0.0), sprint=True,
    )
    # Isolate the mechanism under test: only boundary_radius_m/
    # boundary_strength_base vary between the two calls in the test below --
    # every other repulsion knob stays at its real configured value.
    match.repulsion_params = match.repulsion_params.__class__(
        **{**match.repulsion_params.__dict__, "boundary_radius_m": boundary_radius_m,
           "boundary_strength_base": 1.2 if boundary_radius_m > 0.0 else 0.0},
    )

    for _ in range(30 * 3):
        match.step()
    return match


def test_ball_carrier_steered_away_from_boundary_stays_in_bounds():
    """With boundary repulsion active, a carrier running toward a target
    well past the sideline should be steered back inward before crossing
    it -- the ball (glued to the carrier) must stay in play."""
    pitch = Pitch.standard()
    match = _run_carry_toward_out_of_bounds_target(boundary_radius_m=4.0)
    assert abs(match.ball.position.y) <= pitch.half_width, (
        f"ball carrier ran the ball out over the sideline (y={match.ball.position.y:.2f}, "
        f"boundary={pitch.half_width:.2f}) despite boundary repulsion being active"
    )


def test_same_scenario_without_boundary_repulsion_does_go_out():
    """Sanity check on the test itself: with boundary_radius_m=0 (the
    mechanism disabled), the exact same run-toward-an-out-of-bounds-target
    scenario DOES carry the ball out -- proves the test geometry is a real
    positive case, not something any other existing mechanic already
    prevents."""
    pitch = Pitch.standard()
    match = _run_carry_toward_out_of_bounds_target(boundary_radius_m=0.0)
    assert abs(match.ball.position.y) > pitch.half_width, (
        "expected the ball to go out of bounds with boundary repulsion disabled -- "
        "if it didn't, this test's geometry no longer isolates the fix"
    )


def test_carrier_far_from_boundary_is_unaffected():
    """Nowhere near any boundary: boundary repulsion must be a no-op --
    the carrier should reach essentially the same position as without it."""
    pitch = Pitch.standard()
    start = Vector3(-20.0, 0.0, 0.0)  # pitch centre in y, far from either sideline
    player = make_player("p1", Team.LEFT, attr_value=0.9, position=start)
    player.heading_rad = 0.0
    # Idle-fallback AI: once the bare MoveOrder below completes (arrives +
    # brakes to jog speed) there is nothing left to reissue movement intent
    # each tick, so this stands the player still afterwards rather than
    # leaving them with no intent at all.
    player.ai = StopWhenIdleAI()
    ball = Ball.at_rest(start)
    ball.possessed_by = "p1"
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))
    target = Vector3(0.0, 0.0, 0.0)
    player.current_order = MoveOrder(target_position=target, sprint=True)

    # Track the CLOSEST approach rather than final position -- the MoveOrder
    # completes on arrival and the idle-fallback AI then holds the player at
    # the target, so final position is a meaningful-but-redundant check;
    # closest approach is what actually exercises the sprint-in behaviour.
    min_dist = float("inf")
    for _ in range(30 * 6):
        match.step()
        min_dist = min(min_dist, player.position.distance_to(target))

    assert min_dist < 1.0, (
        f"carrier never got within 1m of a mid-pitch target essentially unobstructed "
        f"(closest approach {min_dist:.2f}m) -- boundary repulsion should never engage "
        f"this far from any boundary"
    )

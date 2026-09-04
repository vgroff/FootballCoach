"""Regression tests for: a player chasing a nearly-stationary ball near the
pitch boundary carries it out of bounds themselves, purely because they never
decelerate on approach.

Real bug this reproduces
-------------------------
``GetPossessionOrder``'s loose-ball chase (``Match._run_get_possession_
behaviour``, the "ball is loose" branch) used to compute movement intent via
``_compute_movement_intent(..., arrival_dist=None, ...)`` -- passing
``arrival_dist=None`` meant NO braking-on-arrival curve was ever applied; the
player sprinted at full intent all the way up to the moment they entered
pickup range (``possession.can_pick_up_ball``'s ``pickup_radius_m``), with
zero anticipatory slowdown based on proximity to the ball, let alone
proximity to the pitch's own boundary.

Ball possession is glued to the carrying player's position every tick
(``Match._sync_possessed_ball``), and nothing in ``Match._apply_movement``
clamps a player's position to stay inside the pitch. So a player sprinting
toward a ball sitting right next to the boundary would pick it up while
still at full sprint speed and keep going straight over the line with it.
Confirmed as a real, live example in recorded training data (see
conversation) -- not just a constructed edge case.

Fix (implemented): the loose-ball chase now computes the fastest speed the
player could be moving at the ball and still brake to a stop before crossing
the NEAREST pitch boundary (kinematic v^2=2*a*d, same formula
``braking_speed_mode`` already uses elsewhere), and feeds that in as
``arrival_speed`` via the same braking machinery every other order uses. Far
from any boundary this naturally computes a safe speed at or above sprint
speed, so ordinary mid-pitch chases are unaffected.

This is a real, large reduction, not a perfect guarantee -- see the note in
match.py about discretisation slack. The tests below use the REAL
``Phase1RulesAI`` (not a bare ``GetPossessionOrder`` with no controlling AI)
so that after pickup the player is actually issued a new order each tick,
matching real gameplay -- a bare order with nothing to reissue it after
completion just coasts forever with zero deceleration once movement intent
stops being set, which is a test-harness artefact, not the real bug.
"""
from __future__ import annotations

import random

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import GetPossessionOrder
from footballcoach.rules_ai import Phase1RulesAI, StopWhenIdleAI
from tests.conftest import make_player


def test_player_chasing_slow_ball_near_sideline_does_not_carry_it_out():
    """A rules-AI player sprinting to collect an almost-stationary ball
    0.5m inside the sideline must not overrun the line with it -- they
    should slow down/stop in time to keep the ball in play, exactly as a
    real player closing on a ball right next to the touchline would."""
    pitch = Pitch.standard()

    ball_y = pitch.half_width - 0.5
    ball = Ball.at_rest(Vector3(0.0, ball_y, 0.0))
    ball.velocity = Vector3(0.0, 0.05, 0.0)  # crawling toward the line, not driving it out itself

    # 15m back is enough distance to reach full sprint speed well before
    # arrival -- exactly the "running in from distance" case described.
    player = make_player("p1", Team.LEFT, attr_value=0.9, position=Vector3(0.0, ball_y - 15.0, 0.0))
    player.heading_rad = 1.5708  # facing +y, straight at the ball
    player.ai = Phase1RulesAI()
    # A second player is required by Phase1RulesAI's opponent lookups. It never
    # gets its own order/AI decision here, so give it the idle-fallback AI to
    # satisfy the movement-intent invariant while it just stands there.
    opponent = make_player("opp", Team.RIGHT, attr_value=0.5, position=Vector3(30.0, 0.0, 0.0))
    opponent.ai = StopWhenIdleAI()

    match = Match(pitch=pitch, players=[player, opponent], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    max_ticks = 30 * 15
    got_possession = False
    for _ in range(max_ticks):
        match.step()
        if match.ball.possessed_by == "p1":
            got_possession = True
        # Real-time boundary check every tick, not just at the end -- the
        # ball must never leave the pitch, at any point in the chase.
        assert abs(match.ball.position.y) <= pitch.half_width + 0.5, (
            f"ball carried out of bounds over the sideline (y={match.ball.position.y:.2f}, "
            f"boundary={pitch.half_width:.2f}) by the player's own sprint momentum after "
            f"picking up an almost-stationary ball"
        )

    assert got_possession, "test setup didn't reach a pickup at all"


def test_player_chasing_slow_ball_near_goal_line_does_not_carry_it_out():
    """Same bug, other boundary: a rules-AI player sprinting to collect an
    almost-stationary ball 0.5m inside the goal line (away from the goal
    mouth, pure out-of-bounds -- not a shot) must not carry it over the
    line."""
    pitch = Pitch.standard()

    ball_x = pitch.half_length - 0.5
    ball = Ball.at_rest(Vector3(ball_x, 20.0, 0.0))
    ball.velocity = Vector3(0.05, 0.0, 0.0)

    player = make_player("p1", Team.LEFT, attr_value=0.9, position=Vector3(ball_x - 15.0, 20.0, 0.0))
    player.heading_rad = 0.0  # facing +x, straight at the ball
    player.ai = Phase1RulesAI()
    opponent = make_player("opp", Team.RIGHT, attr_value=0.5, position=Vector3(0.0, 0.0, 0.0))
    opponent.ai = StopWhenIdleAI()

    match = Match(pitch=pitch, players=[player, opponent], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    max_ticks = 30 * 15
    got_possession = False
    for _ in range(max_ticks):
        match.step()
        if match.ball.possessed_by == "p1":
            got_possession = True
        assert abs(match.ball.position.x) <= pitch.half_length + 1.0, (
            f"ball carried out of bounds over the goal line (x={match.ball.position.x:.2f}, "
            f"boundary={pitch.half_length:.2f}) by the player's own sprint momentum after "
            f"picking up an almost-stationary ball"
        )

    assert got_possession, "test setup didn't reach a pickup at all"


def test_hopeless_chase_is_abandoned_instead_of_carrying_ball_out():
    """When the ball is close enough to the boundary that the raw (pre-
    clamp) safety margin is deep in the negative -- past orders.json's
    boundary_braking.abandon_margin_m -- and nobody has touched the ball yet
    this episode (own_team_touched_last is False), the player should give up
    the chase entirely rather than closing in at the near-zero safe speed
    the kinematic formula would otherwise compute. Touching a ball this
    close to the line still reliably carries it out anyway (see match.py's
    own comment for the real traced episode this was modelled on) -- an
    untouched exit scores "invalid" (0 reward), strictly better than a
    penalised "ball_out" (-4) for a touch that couldn't have saved it."""
    pitch = Pitch.standard()

    # Only 0.05m inside the line -- raw_margin = 0.05 - pickup_radius(0.55)
    # - brake_buffer(0.6) = -1.1, past the default abandon_margin_m=-1.0
    # (well past the -0.65 raw_margin of the OTHER test above, which DOES
    # still recover cleanly -- this case is deliberately further gone).
    ball_y = pitch.half_width - 0.05
    ball = Ball.at_rest(Vector3(0.0, ball_y, 0.0))
    ball.velocity = Vector3(0.0, 0.05, 0.0)

    start_pos = Vector3(0.0, ball_y - 15.0, 0.0)
    player = make_player("p1", Team.LEFT, attr_value=0.9, position=start_pos)
    player.heading_rad = 1.5708  # facing +y, straight at the ball
    player.ai = Phase1RulesAI()
    opponent = make_player("opp", Team.RIGHT, attr_value=0.5, position=Vector3(30.0, 0.0, 0.0))
    opponent.ai = StopWhenIdleAI()

    match = Match(pitch=pitch, players=[player, opponent], ball=ball, rng_reduction=1.0, rng=random.Random(0))
    assert match.ball.last_touched_by_player_id is None  # nobody's touched it yet -- abandon logic applies

    max_ticks = 30 * 10
    for _ in range(max_ticks):
        match.step()
        assert match.ball.possessed_by != "p1", (
            "player should never gain possession of a hopeless boundary-hugging ball -- "
            "the chase should have been abandoned instead of ending in a carry-out"
        )

    # Held back rather than closing the full 15m in -- didn't rush a chase
    # it had already given up on.
    assert player.position.distance_to(start_pos) < 5.0, (
        f"player should have held position instead of sprinting toward a hopeless ball "
        f"(moved {player.position.distance_to(start_pos):.2f}m from start)"
    )


def test_own_team_touched_last_overrides_abandon_and_still_goes_for_it():
    """Same hopeless boundary geometry as test_hopeless_chase_is_abandoned_
    instead_of_carrying_ball_out, but the player's own team touched the ball
    last -- own_team_touched_last must take priority over the abandon check
    (it's checked first in match.py), so the player still sprints in at full
    risk rather than holding back. The untouched-exit ("invalid") floor is
    already gone for this team either way once they've touched it, so there
    is nothing left to protect by giving up."""
    pitch = Pitch.standard()

    ball_y = pitch.half_width - 0.05
    ball = Ball.at_rest(Vector3(0.0, ball_y, 0.0))
    ball.velocity = Vector3(0.0, 0.05, 0.0)
    ball.last_touched_by_player_id = "p1"  # our own team touched it last

    start_pos = Vector3(0.0, ball_y - 15.0, 0.0)
    player = make_player("p1", Team.LEFT, attr_value=0.9, position=start_pos)
    player.heading_rad = 1.5708
    player.ai = Phase1RulesAI()
    opponent = make_player("opp", Team.RIGHT, attr_value=0.5, position=Vector3(30.0, 0.0, 0.0))
    opponent.ai = StopWhenIdleAI()

    match = Match(pitch=pitch, players=[player, opponent], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    for _ in range(30):  # 1s -- enough to see real sprint-from-rest movement
        match.step()

    assert player.position.distance_to(start_pos) > 2.0, (
        "player should still chase aggressively when their own team touched the ball last, "
        "even in an otherwise-hopeless boundary geometry -- own_team_touched_last must override "
        "the abandon check, not be overridden by it"
    )


# ---------------------------------------------------------------------------
# Guard against overcorrecting: the boundary-aware braking above must only
# kick in when the boundary is actually a real constraint. A player chasing
# a ball with plenty of room -- either far from any boundary, or near one
# but with enough distance to comfortably brake -- should still close in and
# collect it at essentially full speed, not needlessly hang back just
# because SOME boundary check now exists on this code path.
# ---------------------------------------------------------------------------

# Reference (pre-fix and post-fix, open-field) pickup speed for a player at
# attr_value=0.9 chasing a stationary ball 20m away in a straight line: ~5.7
# m/s, i.e. essentially full sprint. Used as the bar both tests below must
# clear -- a regression that quietly made chases overcautious would show up
# as a pickup speed well below this, even though the ball still gets
# collected (so a bare "did we get possession" check wouldn't catch it).
_EXPECTED_FULL_SPRINT_PICKUP_SPEED_MPS = 5.0
_EXPECTED_MAX_TICKS = 30 * 6  # generous bound for a 15-20m sprint at 30Hz


def test_player_still_sprints_to_open_field_ball_at_full_speed():
    """Nowhere near any boundary: the fix must be a no-op here -- the
    player should close in and pick up the ball at essentially full sprint
    speed, the same as before the boundary-aware braking was added."""
    pitch = Pitch.standard()
    ball = Ball.at_rest(Vector3(0.0, 0.0, 0.0))  # pitch centre -- maximally far from every boundary
    player = make_player("p1", Team.LEFT, attr_value=0.9, position=Vector3(-20.0, 0.0, 0.0))
    player.heading_rad = 0.0
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))
    player.current_order = GetPossessionOrder(sprint=True)

    for tick in range(_EXPECTED_MAX_TICKS):
        match.step()
        if match.ball.possessed_by == "p1":
            assert player.speed_mps >= _EXPECTED_FULL_SPRINT_PICKUP_SPEED_MPS, (
                f"player picked up an open-field ball at only {player.speed_mps:.2f} m/s -- "
                f"the boundary-aware braking fix should never engage this far from any "
                f"boundary (unnecessary caution mid-pitch, not the bug it's meant to fix)"
            )
            return
    raise AssertionError("player never reached the open-field ball at all")


def test_player_still_sprints_to_near_boundary_ball_with_room_to_spare():
    """Near a boundary, but with 10m of clearance -- comfortably enough
    room to brake if it were ever needed -- the player should still close
    the distance and collect the ball at essentially full sprint speed,
    not hang back out of unnecessary caution."""
    pitch = Pitch.standard()
    ball = Ball.at_rest(Vector3(0.0, pitch.half_width - 10.0, 0.0))
    player = make_player("p1", Team.LEFT, attr_value=0.9, position=Vector3(0.0, pitch.half_width - 25.0, 0.0))
    player.heading_rad = 1.5708
    match = Match(pitch=pitch, players=[player], ball=ball, rng_reduction=1.0, rng=random.Random(0))
    player.current_order = GetPossessionOrder(sprint=True)

    for tick in range(_EXPECTED_MAX_TICKS):
        match.step()
        if match.ball.possessed_by == "p1":
            assert player.speed_mps >= _EXPECTED_FULL_SPRINT_PICKUP_SPEED_MPS, (
                f"player picked up a ball 10m from the boundary at only {player.speed_mps:.2f} "
                f"m/s -- with that much clearance the boundary-aware braking should not have "
                f"engaged at all"
            )
            return
    raise AssertionError("player never reached the ball at all")

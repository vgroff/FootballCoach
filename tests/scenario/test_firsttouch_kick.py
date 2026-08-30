"""First-touch kick: kicking while in CONTROLLING_BALL applies firsttime
difficulty automatically and immediately clears the player back to ACTIVE.
"""
from __future__ import annotations

import random

import pytest

from footballcoach.engine.match import Match
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import PlayerState, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import GetPossessionOrder, KickOrder, MoveOrder
from tests.conftest import make_player


def _match(players, ball, *, rng_reduction=1.0):
    return Match(
        pitch=Pitch.standard(), players=players, ball=ball,
        rng_reduction=rng_reduction, rng=random.Random(0),
    )


# ---------------------------------------------------------------------------
# Basic: kick during CONTROLLING_BALL clears state
# ---------------------------------------------------------------------------

def test_kick_during_controlling_ball_clears_state():
    """KickOrder fired while CONTROLLING_BALL must leave the player ACTIVE."""
    player = make_player("p1", Team.LEFT, position=Vector3(0, 0, 0))
    ball = Ball(position=Vector3(0.3, 0, 0.05), velocity=Vector3(0, 0, 0), spin=Vector3.zero())
    match = _match([player], ball)

    match.step()  # ball pickup -> CONTROLLING_BALL
    assert player.state == PlayerState.CONTROLLING_BALL

    player.current_order = KickOrder(
        aim_point=Vector3(20, 0, 0), power_fraction=0.5, spin=Vector3.zero()
    )
    match.step()

    assert player.kicked_this_tick
    assert player.state == PlayerState.ACTIVE, "kick must clear CONTROLLING_BALL immediately"
    assert match.ball.possessed_by is None


def test_normal_kick_does_not_touch_state():
    """Kicking after full control (ACTIVE state) must not change state."""
    player = make_player("p1", Team.LEFT, position=Vector3(0, 0, 0))
    ball = Ball(position=Vector3(0, 0, 0), velocity=Vector3(0, 0, 0), spin=Vector3.zero())
    ball.possessed_by = player.player_id
    match = _match([player], ball)

    player.current_order = KickOrder(
        aim_point=Vector3(20, 0, 0), power_fraction=0.5, spin=Vector3.zero()
    )
    match.step()

    assert player.kicked_this_tick
    assert player.state == PlayerState.ACTIVE  # unchanged


# ---------------------------------------------------------------------------
# firsttime_difficulty is stored at pickup
# ---------------------------------------------------------------------------

def test_firsttime_difficulty_stored_for_fast_aerial_ball():
    """A fast chest-height ball must produce firsttime_difficulty > 0."""
    player = make_player("p1", Team.LEFT, position=Vector3(0, 0, 0))
    ball = Ball(position=Vector3(0.3, 0, 0.9), velocity=Vector3(-8, 0, 0), spin=Vector3.zero())
    match = _match([player], ball)
    match.step()

    assert player.state == PlayerState.CONTROLLING_BALL
    assert player.firsttime_difficulty > 0.0


def test_firsttime_difficulty_near_zero_for_stationary_ground_ball():
    """A stationary ball at floor level must produce firsttime_difficulty ≈ 0."""
    player = make_player("p1", Team.LEFT, position=Vector3(0, 0, 0))
    ball = Ball(position=Vector3(0.3, 0, 0.05), velocity=Vector3(0, 0, 0), spin=Vector3.zero())
    match = _match([player], ball)
    match.step()

    assert player.state == PlayerState.CONTROLLING_BALL
    assert player.firsttime_difficulty < 0.1


def test_firsttime_difficulty_larger_for_harder_ball():
    """A fast aerial ball must produce higher difficulty than a slow ground ball."""
    def _difficulty_for(ball_pos, ball_vel):
        player = make_player("p1", Team.LEFT, position=Vector3(0, 0, 0))
        ball = Ball(position=ball_pos, velocity=ball_vel, spin=Vector3.zero())
        match = _match([player], ball)
        match.step()
        return player.firsttime_difficulty

    easy = _difficulty_for(Vector3(0.3, 0, 0.05), Vector3(0, 0, 0))
    hard = _difficulty_for(Vector3(0.3, 0, 0.9), Vector3(-8, 0, 0))
    assert hard > easy


# ---------------------------------------------------------------------------
# Push-kick via MoveOrder fires as first-touch and clears state
# ---------------------------------------------------------------------------

def test_push_kick_during_controlling_ball_clears_state():
    """MoveOrder push-kick fired while CONTROLLING_BALL must clear the state."""
    player = make_player("p1", Team.LEFT, position=Vector3(-10, 0, 0))
    player.heading_rad = 0.0  # facing +x (toward target)
    player.velocity = Vector3(5, 0, 0)  # already moving rightward

    ball = Ball(position=Vector3(-9.7, 0, 0.05), velocity=Vector3(0, 0, 0), spin=Vector3.zero())
    match = _match([player], ball)

    match.step()  # pickup -> CONTROLLING_BALL
    assert player.state == PlayerState.CONTROLLING_BALL

    player.current_order = MoveOrder(
        target_position=Vector3(30, 0, 0), sprint=True, push_kick_enabled=True
    )
    match.step()

    assert player.kicked_this_tick, "push-kick must fire on the CONTROLLING_BALL step"
    assert player.state == PlayerState.ACTIVE, "push-kick must clear CONTROLLING_BALL"
    assert match.ball.possessed_by is None


# ---------------------------------------------------------------------------
# Armed kick (pre-possession, via GetPossessionOrder/Match._update_loose_
# ball_pickup's kick_armed branch): a true one-touch redirect -- the whole
# point is that CONTROLLING_BALL is never entered at all, not just cleared
# quickly, so there's no control_speed_multiplier slowdown even for an
# instant. Unlike the MoveOrder case above (ball already in hand, so a
# CONTROLLING_BALL step happens first and the kick clears it), an armed
# touch never sets CONTROLLING_BALL in the first place -- see
# Match._update_loose_ball_pickup's kick_armed branch, which calls
# Player.kick_with_direction() without ever touching player.state.
# ---------------------------------------------------------------------------

def _1v1_get_possession_match(
    target: Vector3, opponent_position: Vector3 = Vector3(0, -34, 0),
) -> tuple[Match, "Player", Ball]:
    """A player 0.3m from a loose ball, already sprinting, with a
    GetPossessionOrder(push_kick_enabled=True) toward target -- close enough
    to arm+fire on the very first tick, far enough from target (>= push_kick's
    configured min_dist_m) that the gate doesn't immediately close.
    opponent_position defaults to just off the own touchline (realistic
    in-game placement, harmless for a near target); for a FAR target, the
    clearance check's opponent-ETA margin scales with self_eta too -- an
    opponent merely 34m to the side can look like a "threat" to a
    60-90m-away target purely because self_eta is already large by then, so
    callers with a far target should pass something unambiguously far
    instead (see test_armed_kick_pickup_skips_controlling_ball_across_two_
    consecutive_touches)."""
    player = make_player("p1", Team.LEFT, attr_value=0.6, position=Vector3(0, 0, 0))
    player.heading_rad = 0.0
    player.velocity = Vector3(7.0, 0, 0)  # already sprinting
    opponent = make_player("opp", Team.RIGHT, attr_value=0.6, position=opponent_position)
    ball = Ball.at_rest(Vector3(0.3, 0, 0))
    match = Match(
        pitch=Pitch.standard(), players=[player, opponent], ball=ball,
        rng_reduction=1.0, rng=random.Random(0),
    )
    player.current_order = GetPossessionOrder(
        sprint=True, target_position=target, push_kick_enabled=True,
    )
    return match, player, ball


def test_armed_kick_pickup_never_enters_controlling_ball():
    """A player with an armed kick who reaches the ball must redirect it
    immediately -- never passing through CONTROLLING_BALL, and never losing
    speed the way a normal (unarmed) first-touch pickup would."""
    match, player, ball = _1v1_get_possession_match(target=Vector3(30, 0, 0))

    match.step()

    assert player.kicked_this_tick, "armed kick must fire once the ball is reachable"
    assert player.state != PlayerState.CONTROLLING_BALL, (
        "armed kick must skip CONTROLLING_BALL entirely, not just clear it"
    )
    assert ball.possessed_by is None, "ball must be released again, not held"
    assert player.speed_mps > 5.0, (
        f"player speed ({player.speed_mps:.2f} m/s) must not crash toward zero -- "
        "an armed touch has no control-time delay"
    )


def test_armed_kick_pickup_skips_controlling_ball_across_two_consecutive_touches():
    """The property above must hold repeatedly, not just for a single touch
    -- back-to-back armed redirects (chasing a ball you just kicked, and
    redirecting it again the instant you reach it) must never slow down in
    between, which is the entire reason armed kicks exist. Drives this
    through the real order machinery (GetPossessionOrder), not hand-set
    kick_armed fields, so it also proves the arming is correctly re-issued
    each time -- exactly as Phase1RulesAI does after each touch completes
    its (self-completing) GetPossessionOrder."""
    # Deliberately far -- must stay well out of reach of a single kick
    # regardless of how push_kick's speed_factor/min_dist_m get tuned, so
    # this test verifies the no-slowdown invariant rather than being
    # coupled to a specific config value (that's test_push_kick_box_to_box_
    # realistic_kick_count's job). Opponent placed unambiguously far away
    # (not just off to the side) -- see _1v1_get_possession_match's
    # docstring for why a merely-off-to-the-side opponent can look like a
    # false "threat" once self_eta to a far target is already large.
    target = Vector3(80, 0, 0)
    match, player, ball = _1v1_get_possession_match(
        target=target, opponent_position=Vector3(-500, 0, 0),
    )

    touches = 0
    states_seen: set = set()
    for _ in range(150):
        match.step()
        states_seen.add(player.state)
        if player.kicked_this_tick:
            touches += 1
            if touches >= 2:
                break
        if player.current_order is None:
            # GetPossessionOrder self-completes the tick after possession is
            # (momentarily) gained -- reissue it and put the ball back in
            # reach, mirroring the AI re-arming for the next touch.
            player.current_order = GetPossessionOrder(
                sprint=True, target_position=target, push_kick_enabled=True,
            )
            player.velocity = Vector3(7.0, 0, 0)
            ball.position = player.position + Vector3(0.3, 0, 0)
            ball.velocity = Vector3.zero()

    assert touches >= 2, "test setup did not produce two armed touches"
    assert PlayerState.CONTROLLING_BALL not in states_seen, (
        "no touch across the whole sequence should ever enter CONTROLLING_BALL"
    )
    assert player.speed_mps > 5.0, (
        f"player speed ({player.speed_mps:.2f} m/s) must stay high across both touches"
    )


def test_get_possession_order_push_kick_fires_without_controlling_ball_slowdown():
    """End-to-end through the real order machinery (not the low-level
    kick_armed fields directly): GetPossessionOrder with push_kick_enabled
    and a far-enough target_position must arm and fire a redirect kick while
    still chasing a loose ball it has never possessed -- and CONTROLLING_BALL
    must never appear at any point during the approach or the kick itself."""
    player = make_player("p1", Team.LEFT, attr_value=0.6, position=Vector3(0, 0, 0))
    player.heading_rad = 0.0
    opponent = make_player("opp", Team.RIGHT, attr_value=0.6, position=Vector3(0, -34, 0))
    ball = Ball.at_rest(Vector3(2, 0, 0))
    match = Match(
        pitch=Pitch.standard(), players=[player, opponent], ball=ball,
        rng_reduction=1.0, rng=random.Random(0),
    )

    target = Vector3(30, 0, 0)  # >= push_kick min_dist_m (24m default)
    player.current_order = GetPossessionOrder(
        sprint=True, target_position=target, push_kick_enabled=True,
    )

    states_before_first_kick = set()
    kicked = False
    for _ in range(30 * 10):
        match.step()
        if player.kicked_this_tick:
            kicked = True
            break
        states_before_first_kick.add(player.state)

    assert kicked, "push-kick never fired -- test setup didn't reach the arm/fire gates"
    assert PlayerState.CONTROLLING_BALL not in states_before_first_kick, (
        "approaching the ball for an armed redirect must not enter CONTROLLING_BALL"
    )
    assert player.state != PlayerState.CONTROLLING_BALL, (
        "the kick itself must not leave the player in CONTROLLING_BALL"
    )
    assert match.ball.possessed_by is None, "ball must be released again after the redirect"

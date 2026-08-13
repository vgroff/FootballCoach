"""Unit tests for possession.can_pick_up_ball() -- the closing-velocity +
deadzone + swept-segment pickup eligibility rule that replaced the old
time-based release-grace mechanic.

Covers, independently of the full Match tick loop:
- radius gating (in/out of range)
- closing vs receding relative velocity
- the near-zero relative-speed deadzone (stationary/rolling ball pickable
  without needing to be actively closing)
- the swept-segment tunneling exception, and its critical exclusion for a
  player whose OWN kick starts inside the radius (must never be treated as
  a tunneling event -- see engine/knowledge.md and the regression this
  guards: a kicker could otherwise instantly re-pick-up their own kick).
"""
from __future__ import annotations

from footballcoach.engine.possession import BallPickupParams, can_pick_up_ball
from footballcoach.entities.ball import Ball
from footballcoach.mathutils import Vector3
from tests.conftest import make_player

_PARAMS = BallPickupParams(pickup_radius_m=0.4, closing_speed_deadzone_mps=0.3)


def _ball(position: Vector3, velocity: Vector3 = Vector3.zero()) -> Ball:
    b = Ball.at_rest(position)
    b.velocity = velocity
    return b


class TestRadiusGating:
    def test_within_radius_stationary_ball_is_pickable(self):
        player = make_player(position=Vector3(0, 0, 0))
        ball = _ball(Vector3(0.2, 0, 0))
        assert can_pick_up_ball(player, ball, _PARAMS)

    def test_outside_radius_stationary_ball_not_pickable(self):
        player = make_player(position=Vector3(0, 0, 0))
        ball = _ball(Vector3(1.0, 0, 0))
        assert not can_pick_up_ball(player, ball, _PARAMS)

    def test_exactly_at_radius_boundary_is_pickable(self):
        player = make_player(position=Vector3(0, 0, 0))
        ball = _ball(Vector3(_PARAMS.pickup_radius_m, 0, 0))
        assert can_pick_up_ball(player, ball, _PARAMS)

    def test_just_outside_radius_boundary_not_pickable(self):
        player = make_player(position=Vector3(0, 0, 0))
        ball = _ball(Vector3(_PARAMS.pickup_radius_m + 0.01, 0, 0))
        assert not can_pick_up_ball(player, ball, _PARAMS)


class TestClosingVelocity:
    def test_ball_moving_toward_player_is_pickable(self):
        """Ball within radius, moving toward the player faster than the
        deadzone -- must be pickable (closing)."""
        player = make_player(position=Vector3(0, 0, 0))
        ball = _ball(Vector3(0.3, 0, 0), velocity=Vector3(-2.0, 0, 0))
        assert can_pick_up_ball(player, ball, _PARAMS)

    def test_ball_moving_away_from_player_not_pickable(self):
        """Ball within radius, moving AWAY from the player faster than the
        deadzone -- must NOT be pickable (receding). This is the core fix:
        replaces the old identity-based release-grace exemption."""
        player = make_player(position=Vector3(0, 0, 0))
        ball = _ball(Vector3(0.3, 0, 0), velocity=Vector3(2.0, 0, 0))
        assert not can_pick_up_ball(player, ball, _PARAMS)

    def test_player_chasing_faster_than_ball_still_closes(self):
        """Ball moving away slowly, player moving toward it faster --
        relative velocity closes even though the ball's raw velocity is
        away from the player."""
        player = make_player(position=Vector3(0, 0, 0))
        player.velocity = Vector3(5.0, 0, 0)  # chasing hard
        ball = _ball(Vector3(0.3, 0, 0), velocity=Vector3(1.0, 0, 0))  # ball drifting away slowly
        assert can_pick_up_ball(player, ball, _PARAMS)

    def test_perpendicular_motion_at_the_tangent_is_not_closing(self):
        """Ball moving purely tangentially (perpendicular to the line to the
        player) neither closes nor recedes -- treated as not closing (must
        rely on the deadzone or fail); this test uses a speed well above the
        deadzone so it must fail."""
        player = make_player(position=Vector3(0, 0, 0))
        ball = _ball(Vector3(0.3, 0, 0), velocity=Vector3(0, 5.0, 0))
        assert not can_pick_up_ball(player, ball, _PARAMS)


class TestDeadzone:
    def test_ball_at_rest_next_to_stationary_player_is_pickable(self):
        player = make_player(position=Vector3(0, 0, 0))
        ball = _ball(Vector3(0.3, 0, 0), velocity=Vector3.zero())
        assert can_pick_up_ball(player, ball, _PARAMS)

    def test_relative_speed_just_below_deadzone_is_pickable_even_receding(self):
        player = make_player(position=Vector3(0, 0, 0))
        ball = _ball(Vector3(0.3, 0, 0), velocity=Vector3(_PARAMS.closing_speed_deadzone_mps - 0.05, 0, 0))
        assert can_pick_up_ball(player, ball, _PARAMS)

    def test_relative_speed_just_above_deadzone_and_receding_not_pickable(self):
        player = make_player(position=Vector3(0, 0, 0))
        ball = _ball(Vector3(0.3, 0, 0), velocity=Vector3(_PARAMS.closing_speed_deadzone_mps + 0.05, 0, 0))
        assert not can_pick_up_ball(player, ball, _PARAMS)

    def test_deadzone_uses_relative_not_absolute_speed(self):
        """Both ball and player moving fast in the SAME direction at nearly
        the same speed -- large absolute speeds, tiny relative speed -- must
        be pickable via the deadzone."""
        player = make_player(position=Vector3(0, 0, 0))
        player.velocity = Vector3(8.0, 0, 0)
        ball = _ball(Vector3(0.3, 0, 0), velocity=Vector3(8.1, 0, 0))
        assert can_pick_up_ball(player, ball, _PARAMS)


class TestSweptTunneling:
    def test_fast_ball_tunneling_past_player_within_one_tick_is_pickable(self):
        """Ball starts far away (outside radius), ends up far away on the
        OTHER side (outside radius, receding), but its straight-line path
        this tick passed directly through the player -- must be pickable."""
        player = make_player(position=Vector3(0, 0, 0))
        pre_tick = Vector3(-5.0, 0, 0)
        ball = _ball(Vector3(5.0, 0, 0), velocity=Vector3(300.0, 0, 0))  # receding fast at the endpoint
        assert can_pick_up_ball(player, ball, _PARAMS, ball_pre_tick_position=pre_tick)

    def test_fast_ball_passing_just_outside_radius_not_pickable(self):
        """Same as above but the straight-line path passes just outside the
        pickup radius (offset in y) -- must NOT be pickable."""
        player = make_player(position=Vector3(0, 0, 0))
        pre_tick = Vector3(-5.0, _PARAMS.pickup_radius_m + 0.1, 0)
        ball = _ball(Vector3(5.0, _PARAMS.pickup_radius_m + 0.1, 0), velocity=Vector3(300.0, 0, 0))
        assert not can_pick_up_ball(player, ball, _PARAMS, ball_pre_tick_position=pre_tick)

    def test_own_kick_starting_inside_radius_is_never_treated_as_tunneling(self):
        """CRITICAL regression test: a player who just kicked the ball --
        its pre-tick position is their own position (inside the radius) --
        must NOT be eligible to re-pick it up via the swept-tunneling
        exception, even though the ball ends up outside the radius and
        receding. This was the actual bug found: the swept check must only
        apply when the ball started the tick OUTSIDE the radius."""
        player = make_player(position=Vector3(0, 0, 0))
        pre_tick = Vector3(0.0, 0, 0)  # ball started at the kicker's own feet
        ball = _ball(Vector3(0.6, 0, 0), velocity=Vector3(18.0, 0, 0))  # kicked away fast, now outside radius
        assert not can_pick_up_ball(player, ball, _PARAMS, ball_pre_tick_position=pre_tick)

    def test_own_kick_ending_up_still_inside_radius_uses_normal_closing_check(self):
        """A weak kick that doesn't clear the radius within one tick must
        still be judged by the ordinary closing-velocity rule (not the
        swept exception, since it never started outside radius) -- and
        since it's moving away from the kicker, it must be rejected."""
        player = make_player(position=Vector3(0, 0, 0))
        pre_tick = Vector3(0.0, 0, 0)
        ball = _ball(Vector3(0.1, 0, 0), velocity=Vector3(1.5, 0, 0))  # still inside radius, receding
        assert not can_pick_up_ball(player, ball, _PARAMS, ball_pre_tick_position=pre_tick)

    def test_no_pre_tick_position_falls_back_to_endpoint_only_check(self):
        """ball_pre_tick_position=None (default) must behave exactly like
        the pure endpoint closing-velocity check, with no swept exception."""
        player = make_player(position=Vector3(0, 0, 0))
        ball = _ball(Vector3(0.3, 0, 0), velocity=Vector3(2.0, 0, 0))
        assert not can_pick_up_ball(player, ball, _PARAMS)
        assert not can_pick_up_ball(player, ball, _PARAMS, ball_pre_tick_position=None)

    def test_stationary_ball_pre_tick_equals_post_tick_no_crash(self):
        """Degenerate zero-length swept segment (ball didn't move this
        tick, e.g. resting) must not error and must fall back sanely to the
        endpoint check."""
        player = make_player(position=Vector3(0, 0, 0))
        pre_tick = Vector3(0.3, 0, 0)
        ball = _ball(Vector3(0.3, 0, 0), velocity=Vector3.zero())
        assert can_pick_up_ball(player, ball, _PARAMS, ball_pre_tick_position=pre_tick)

    def test_ball_exactly_coincident_with_player_is_pickable(self):
        player = make_player(position=Vector3(0, 0, 0))
        ball = _ball(Vector3(0, 0, 0), velocity=Vector3(5.0, 0, 0))
        assert can_pick_up_ball(player, ball, _PARAMS)


class TestMovingPlayer:
    def test_moving_player_and_ball_both_receding_from_each_other_not_pickable(self):
        player = make_player(position=Vector3(0, 0, 0))
        player.velocity = Vector3(-3.0, 0, 0)  # retreating from the ball
        ball = _ball(Vector3(0.3, 0, 0), velocity=Vector3(3.0, 0, 0))  # ball also moving away
        assert not can_pick_up_ball(player, ball, _PARAMS)

    def test_player_running_alongside_ball_converging_slightly_is_pickable(self):
        """Player and ball both moving in the same general direction, but
        the player's path converges slightly onto the ball's -- relative
        velocity should show closing."""
        player = make_player(position=Vector3(0, -0.35, 0))
        player.velocity = Vector3(5.0, 0.5, 0)  # angled slightly toward the ball's lane
        ball = _ball(Vector3(0.0, 0.0, 0), velocity=Vector3(5.0, 0.0, 0))
        assert can_pick_up_ball(player, ball, _PARAMS)


class TestPlayerSweptTunneling:
    """Symmetric counterpart to TestSweptTunneling: a fast PLAYER sweeping
    past a slow/stationary ball within one tick, exercised via the new
    `player_pre_tick_position` parameter. See engine/knowledge.md."""

    def test_fast_player_grazing_stationary_ball_is_pickable(self):
        """Player sweeps in a straight line from (-1, 0.35) to (1, 0.35)
        (constant y=0.35, inside the 0.4m radius) past a stationary ball at
        the origin. Both the tick-start and tick-end positions are outside
        the radius (distance ~1.06m), but the swept path grazes past at
        0.35m -- must be pickable via the player-side tunneling exception.

        The player is ALSO moving fast (+x) at the tick's end, i.e. already
        past and receding from the ball at that instant (closing_component
        < 0 at the endpoint) -- this specifically exercises that the grant
        does not depend on the endpoint velocity direction, only on the
        geometric sweep, mirroring the ball-side "regardless of endpoint
        closing direction" rule.
        """
        player = make_player(position=Vector3(1.0, 0.35, 0))
        player.velocity = Vector3(20.0, 0, 0)  # sprinting hard, already past+receding
        player_pre = Vector3(-1.0, 0.35, 0)
        ball = _ball(Vector3(0, 0, 0))  # stationary
        assert can_pick_up_ball(player, ball, _PARAMS, player_pre_tick_position=player_pre)

    def test_fast_player_passing_just_outside_radius_not_pickable(self):
        """Same sweep but offset to y=0.45 (just outside the 0.4m radius) --
        must NOT be pickable."""
        player = make_player(position=Vector3(1.0, 0.45, 0))
        player.velocity = Vector3(20.0, 0, 0)
        player_pre = Vector3(-1.0, 0.45, 0)
        ball = _ball(Vector3(0, 0, 0))
        assert not can_pick_up_ball(player, ball, _PARAMS, player_pre_tick_position=player_pre)

    def test_player_already_within_radius_at_tick_start_not_treated_as_tunneling(self):
        """CRITICAL regression test, symmetric to
        test_own_kick_starting_inside_radius_is_never_treated_as_tunneling:
        a player who was ALREADY standing next to the ball at the start of
        the tick (pre-tick separation <= radius) must not get a free pass
        via the swept exception just because the ball is now receding fast
        -- must fall through to the ordinary closing-velocity check and be
        rejected."""
        player = make_player(position=Vector3(0, 0, 0))
        player_pre = Vector3(0.05, 0, 0)  # started well within radius of the ball
        ball = _ball(Vector3(0.3, 0, 0), velocity=Vector3(2.0, 0, 0))  # receding fast
        assert not can_pick_up_ball(player, ball, _PARAMS, player_pre_tick_position=player_pre)

    def test_player_pre_tick_position_at_exactly_radius_boundary_does_not_trigger_sweep(self):
        """Pre-tick separation exactly equal to the radius (not strictly
        greater) must NOT count as 'started outside' -- boundary must match
        the existing `distance <= radius` convention used everywhere else."""
        player = make_player(position=Vector3(2.0, 0, 0))
        player_pre = Vector3(-_PARAMS.pickup_radius_m, 0, 0)  # exactly radius_m from a ball at origin
        ball = _ball(Vector3(0, 0, 0), velocity=Vector3(5.0, 0, 0))  # receding fast, endpoint far away
        # started_outside_radius should be False (separation == radius, not >),
        # so no swept grant; ordinary check (out of radius, not deadzone) rejects it.
        assert not can_pick_up_ball(player, ball, _PARAMS, player_pre_tick_position=player_pre)

    def test_stationary_player_pre_tick_equals_post_tick_no_crash(self):
        """Degenerate zero-length player-swept segment (player didn't move
        this tick) must not error and must fall back sanely to the ordinary
        endpoint check -- symmetric to the ball-side degenerate test."""
        player = make_player(position=Vector3(0.3, 0, 0))
        player_pre = Vector3(0.3, 0, 0)
        ball = _ball(Vector3(0, 0, 0), velocity=Vector3.zero())
        assert can_pick_up_ball(player, ball, _PARAMS, player_pre_tick_position=player_pre)

    def test_omitting_player_pre_tick_position_matches_ball_only_behaviour_exactly(self):
        """Passing only ball_pre_tick_position (player_pre_tick_position
        omitted) must reproduce the original ball-only-tunneling results
        bit-for-bit -- the historical default path must be unaffected by
        the new parameter's existence."""
        player = make_player(position=Vector3(0, 0, 0))
        pre_tick = Vector3(-5.0, 0, 0)
        ball = _ball(Vector3(5.0, 0, 0), velocity=Vector3(300.0, 0, 0))
        assert can_pick_up_ball(player, ball, _PARAMS, ball_pre_tick_position=pre_tick)
        assert can_pick_up_ball(
            player, ball, _PARAMS, ball_pre_tick_position=pre_tick, player_pre_tick_position=None,
        )

    def test_player_side_sweep_is_the_exact_mirror_of_ball_side_sweep(self):
        """Construct the mirror image of
        test_fast_ball_tunneling_past_player_within_one_tick_is_pickable:
        instead of a stationary player and a ball sweeping from -5 to +5,
        use a stationary ball and a player sweeping from -5 to +5 (with the
        same fast receding endpoint velocity) -- must be pickable, proving
        the two code paths are genuinely symmetric and not just
        superficially similar."""
        ball = _ball(Vector3(0, 0, 0))  # stationary at origin, at what was the player's spot
        player = make_player(position=Vector3(5.0, 0, 0))  # ends where the ball used to end
        player.velocity = Vector3(300.0, 0, 0)  # receding fast at the endpoint, like the ball was
        player_pre = Vector3(-5.0, 0, 0)  # started where the ball used to start
        assert can_pick_up_ball(player, ball, _PARAMS, player_pre_tick_position=player_pre)


class TestBothEntitiesMovingSweep:
    """Neither ball-only nor player-only sweep alone can catch a close pass
    where BOTH entities move during the tick and neither one's individual
    endpoint-vs-fixed-point check would register a close approach -- only
    the relative-motion sweep (see possession._swept_min_separation) does.
    """

    def test_crossing_paths_both_moving_is_pickable(self):
        """Ball travels along y=0 from x=-1 to x=1; player travels along
        x=0.2 from y=-1 to y=1 over the SAME tick (synchronized linear
        interpolation). Their paths cross near the origin around the
        segment's midpoint, well within the 0.4m radius, even though every
        individual endpoint pairing is over 1m apart."""
        ball_pre = Vector3(-1.0, 0.0, 0)
        ball = _ball(Vector3(1.0, 0.0, 0))
        player_pre = Vector3(0.2, -1.0, 0)
        player = make_player(position=Vector3(0.2, 1.0, 0))
        assert can_pick_up_ball(
            player, ball, _PARAMS,
            ball_pre_tick_position=ball_pre, player_pre_tick_position=player_pre,
        )

    def test_both_moving_but_paths_stay_far_apart_not_pickable(self):
        """Same shape of motion, but the player's line (x=5) never comes
        close to the ball's line (y=0, x in [-1, 1]) -- must not be
        pickable."""
        ball_pre = Vector3(-1.0, 0.0, 0)
        ball = _ball(Vector3(1.0, 0.0, 0))
        player_pre = Vector3(5.0, -1.0, 0)
        player = make_player(position=Vector3(5.0, 1.0, 0))
        assert not can_pick_up_ball(
            player, ball, _PARAMS,
            ball_pre_tick_position=ball_pre, player_pre_tick_position=player_pre,
        )

    def test_both_moving_endpoints_within_radius_still_pickable_via_ordinary_check(self):
        """Sanity check: if both pre-tick positions are given but the
        TICK-END separation already satisfies the ordinary within_radius
        check, the result must still be pickable regardless of the sweep
        (the sweep is an additional exception, not a replacement)."""
        ball_pre = Vector3(-1.0, 0.0, 0)
        ball = _ball(Vector3(0.1, 0.0, 0), velocity=Vector3.zero())
        player_pre = Vector3(5.0, -1.0, 0)  # started far away, unrelated to the ball's path
        player = make_player(position=Vector3(0.0, 0.0, 0))  # ends up right next to the ball
        assert can_pick_up_ball(
            player, ball, _PARAMS,
            ball_pre_tick_position=ball_pre, player_pre_tick_position=player_pre,
        )

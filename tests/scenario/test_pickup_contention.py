"""Engine-level tests for `Match._update_loose_ball_pickup`'s multi-player
contention resolution and its wiring of `player_pre_tick_position` through
`Match.step()`.

Companion to `tests/unit/test_ball_pickup.py`, which unit-tests
`possession.can_pick_up_ball()` (including the player-side swept-tunneling
exception) in isolation. These tests exercise the Match-level candidate
collection + resolution logic in `engine/match.py::_update_loose_ball_pickup`
directly (most tests here call it with a hand-crafted pre-tick-positions
dict, for determinism independent of movement-physics tuning), plus one
full `Match.step()` end-to-end test proving the pre-tick position snapshot
is captured before movement, not after.

See engine/knowledge.md.
"""
from __future__ import annotations

import random

import pytest

from footballcoach.engine.match import Match
from footballcoach.engine.possession import BallPickupParams
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import PlayerState, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import MoveOrder

from tests.conftest import make_player


def _make_match(*players, ball, rng_reduction=1.0) -> Match:
    # pickup_radius_m/closing_speed_deadzone_mps pinned explicitly rather than
    # read from the live physics.json config -- several tests below use
    # boundary-hugging distances (e.g. 0.35m, 0.39m) chosen relative to a
    # specific radius value, so letting them silently track a config change
    # would invalidate the numbers without the test actually testing
    # anything different (this already broke one test in this file once,
    # see TestStepWiringCapturesPreMovementPosition's docstring).
    return Match(
        pitch=Pitch.standard(),
        players=list(players),
        ball=ball,
        rng_reduction=rng_reduction,
        rng=random.Random(0),
        ball_pickup_params=BallPickupParams(pickup_radius_m=0.4, closing_speed_deadzone_mps=1.0),
    )


class TestContentionResolution:
    def test_closest_active_player_wins_regardless_of_list_order(self):
        """Two players both within pickup radius of a stationary ball --
        the CLOSER one must win, independent of their order in
        match.players (previously an arbitrary list-order tie-break)."""
        near = make_player("near", Team.LEFT, position=Vector3(0.1, 0, 0))
        far = make_player("far", Team.RIGHT, position=Vector3(0.35, 0, 0))
        ball = Ball.at_rest(Vector3(0, 0, 0))

        for players in ([near, far], [far, near]):
            near.state = PlayerState.ACTIVE
            far.state = PlayerState.ACTIVE
            m = _make_match(*players, ball=Ball.at_rest(Vector3(0, 0, 0)))
            m._update_loose_ball_pickup(
                m.dt_s, m.ball.position,
                {"near": Vector3(0.1, 0, 0), "far": Vector3(0.35, 0, 0)},
            )
            assert m.ball.possessed_by == "near", (
                f"closest player must win regardless of list order {[p.player_id for p in players]}"
            )

    def test_within_radius_candidate_beats_farther_swept_only_candidate(self):
        """Player A is sitting right on the ball (within_radius, 0.05m).
        Player B is only eligible via the swept-tunneling exception and
        ends the tick 0.35m away (still inside radius here, but the point
        is A is unambiguously closer) -- A must win."""
        a = make_player("a", Team.LEFT, position=Vector3(0.05, 0, 0))
        b = make_player("b", Team.RIGHT, position=Vector3(0.35, 0, 0))
        ball = Ball.at_rest(Vector3(0, 0, 0))
        m = _make_match(a, b, ball=ball)
        # B swept in from far away this tick (started outside radius, grazed close).
        m._update_loose_ball_pickup(
            m.dt_s, m.ball.position,
            {"a": Vector3(0.05, 0, 0), "b": Vector3(-2.0, 0.35, 0)},
        )
        assert m.ball.possessed_by == "a"

    def test_swept_only_candidate_wins_when_sole_eligible_player(self):
        """A player who is NOT currently within the plain pickup radius,
        but swept through it this tick, must still be granted possession
        when they are the only eligible candidate."""
        player = make_player("p", Team.LEFT, position=Vector3(1.0, 0.35, 0))
        player.velocity = Vector3(20.0, 0, 0)
        ball = Ball.at_rest(Vector3(0, 0, 0))
        m = _make_match(player, ball=ball)
        m._update_loose_ball_pickup(
            m.dt_s, m.ball.position, {"p": Vector3(-1.0, 0.35, 0)},
        )
        assert m.ball.possessed_by == "p"
        assert player.state == PlayerState.CONTROLLING_BALL

    def test_inactive_and_controlling_players_are_never_candidates(self):
        """A player who is closer to the ball than the eligible candidate,
        but is INACTIVE_TACKLED or already CONTROLLING_BALL (state !=
        ACTIVE), must never win contention even though they'd win on pure
        distance."""
        blocked = make_player("blocked", Team.LEFT, position=Vector3(0.01, 0, 0))
        blocked.state = PlayerState.INACTIVE_TACKLED
        eligible = make_player("eligible", Team.RIGHT, position=Vector3(0.39, 0, 0))
        ball = Ball.at_rest(Vector3(0, 0, 0))
        m = _make_match(blocked, eligible, ball=ball)
        m._update_loose_ball_pickup(
            m.dt_s, m.ball.position,
            {"blocked": Vector3(0.01, 0, 0), "eligible": Vector3(0.39, 0, 0)},
        )
        assert m.ball.possessed_by == "eligible"

    def test_no_eligible_candidates_leaves_ball_loose(self):
        far1 = make_player("far1", Team.LEFT, position=Vector3(5.0, 0, 0))
        far2 = make_player("far2", Team.RIGHT, position=Vector3(-5.0, 0, 0))
        ball = Ball.at_rest(Vector3(0, 0, 0))
        m = _make_match(far1, far2, ball=ball)
        m._update_loose_ball_pickup(
            m.dt_s, m.ball.position,
            {"far1": Vector3(5.0, 0, 0), "far2": Vector3(-5.0, 0, 0)},
        )
        assert m.ball.possessed_by is None

    def test_already_possessed_ball_short_circuits_before_any_contention(self):
        """If the ball is already possessed (e.g. by a third player mid-
        control elsewhere in the tick), _update_loose_ball_pickup must
        return immediately without touching any candidate's state."""
        a = make_player("a", Team.LEFT, position=Vector3(0.05, 0, 0))
        ball = Ball.at_rest(Vector3(0, 0, 0))
        m = _make_match(a, ball=ball)
        m.ball.possessed_by = "someone_else"  # simulate a possession granted elsewhere this tick
        m._update_loose_ball_pickup(m.dt_s, m.ball.position, {"a": Vector3(0.05, 0, 0)})
        assert m.ball.possessed_by == "someone_else"
        assert a.state == PlayerState.ACTIVE


class TestStepWiringCapturesPreMovementPosition:
    def test_sprinting_player_grazes_stationary_ball_via_real_step(self):
        """End-to-end: a real Match.step() call must capture each player's
        PRE-movement position for the swept check, not the post-movement
        one -- otherwise the swept check would degenerate to a zero-length
        segment (player.position vs itself) and never fire.

        Player starts just before a stationary ball, offset in y by less
        than the pickup radius, already at high sprint speed heading
        straight past it in +x with no turning required (so `_apply_movement`
        keeps the speed ~constant this tick and the swept check has a real,
        non-degenerate segment to test against).

        `pickup_radius_m` is pinned explicitly (not read from the live
        physics.json config) since this test's geometry is deliberately
        tuned to straddle a specific radius value -- letting it silently
        track a config change would invalidate the numbers without the test
        actually testing anything different (this happened once already:
        physics.json's live pickup_radius_m moved from 0.4 to 0.55 and this
        test's endpoint distance, ~0.406m, went from "just outside" to
        "comfortably inside," which made it exercise the ORDINARY
        receding-velocity rejection path instead of the swept-tunneling
        exception this test exists to cover).
        """
        ball = Ball.at_rest(Vector3(0.0, 0.0, 0.0))
        pickup_params = BallPickupParams(pickup_radius_m=0.4, closing_speed_deadzone_mps=1.0)
        # y=0.37 (just inside the 0.4m radius) with x=-0.1685 -> pre-tick
        # distance to ball ~0.407m (just OUTSIDE radius). At ~10.1 m/s this
        # player covers ~0.337m in one 1/30s tick, landing at x=+0.168,
        # distance ~0.406m (also just OUTSIDE radius) -- straddling the
        # 0.4m-radius chord at y=0.37 (half-width ~0.152m) with margin to
        # spare on both sides, so this isn't a knife-edge numeric fluke.
        player = make_player("p", Team.LEFT, position=Vector3(-0.1685, 0.37, 0), attr_value=1.0)
        player.velocity = Vector3(10.0, 0.0, 0.0)
        player.heading_rad = 0.0  # already facing +x -- no turning penalty this tick
        player.move_to(Vector3(1000.0, 0.37, 0.0), sprint=True)

        m = Match(
            pitch=Pitch.standard(), players=[player], ball=ball, rng=random.Random(0),
            ball_pickup_params=pickup_params,
        )
        m.step()

        assert m.ball.possessed_by == "p", (
            "a fast player whose straight-line path this tick grazed the "
            "stationary ball must pick it up even though neither their "
            "pre- nor post-tick position alone was within pickup_radius_m"
        )
        assert player.state == PlayerState.CONTROLLING_BALL

    def test_slow_player_does_not_falsely_trigger_sweep_from_standing_still(self):
        """Control: a player who starts and ends the tick outside the
        pickup radius, moving too slowly to sweep through it, must NOT
        pick up the ball -- guards against a degenerate always-True sweep
        bug (e.g. accidentally comparing a point to itself)."""
        ball = Ball.at_rest(Vector3(0.0, 0.0, 0.0))
        player = make_player("p", Team.LEFT, position=Vector3(-2.0, 5.0, 0), attr_value=0.0)
        m = Match(pitch=Pitch.standard(), players=[player], ball=ball, rng=random.Random(0))
        m.step()
        assert m.ball.possessed_by is None
        assert player.state == PlayerState.ACTIVE

"""Tests for the reusable scenario AI primitives:
``BallCarrierAttackerAI``, ``StagedGoalkeeperAI``.

These run headlessly (no pygame) at rng_reduction=1.0 (deterministic).
"""
from __future__ import annotations

import math
import random

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import MoveOrder, SaveOrder, ShootOrder
from footballcoach.rules_ai import BallCarrierAttackerAI, StagedGoalkeeperAI
from footballcoach.ui.scenarios import (
    build_1v2_scenario,
    build_2v2_scenario,
)
from tests.conftest import make_player


# ---------------------------------------------------------------------------
# BallCarrierAttackerAI
# ---------------------------------------------------------------------------

class TestBallCarrierAttackerAI:
    """Tests for BallCarrierAttackerAI."""

    def _make_match(self, player_pos: Vector3, aim: Vector3) -> tuple[Match, object]:
        pitch = Pitch.standard()
        player = make_player("atk", Team.LEFT, position=player_pos)
        ball = Ball.at_rest(player_pos)
        ball.possessed_by = player.player_id
        match = Match(pitch=pitch, players=[player], ball=ball,
                      rng_reduction=1.0, rng=random.Random(0))
        return match, player

    def test_issues_shoot_order_when_no_order(self):
        """When the attacker has the ball and no current order, AI issues ShootOrder."""
        pitch = Pitch.standard()
        aim = pitch.right_goal_centre.with_z(1.0)
        match, player = self._make_match(Vector3(0.0, 0.0, 0.0), aim)
        ai = BallCarrierAttackerAI(aim, power_fraction=0.9)

        assert player.current_order is None
        ai.act(player, match, 0)
        assert isinstance(player.current_order, ShootOrder), (
            "AI should issue ShootOrder when ball carrier has no order"
        )
        assert player.current_order.aim_point == aim
        assert player.current_order.power_fraction == 0.9

    def test_does_not_interfere_when_move_order_progressing(self):
        """While the move target distance is decreasing (progress), no ShootOrder is issued."""
        pitch = Pitch.standard()
        aim = pitch.right_goal_centre.with_z(1.0)
        target = Vector3(10.0, 0.0, 0.0)
        match, player = self._make_match(Vector3(0.0, 0.0, 0.0), aim)
        player.current_order = MoveOrder(target_position=target, sprint=True)
        ai = BallCarrierAttackerAI(aim, power_fraction=0.9)

        # First tick: records dist, doesn't shoot
        ai.act(player, match, 0)
        assert isinstance(player.current_order, MoveOrder), (
            "AI should not switch to ShootOrder while distance to target is unchanged"
        )

    def test_switches_to_shoot_when_distance_increases(self):
        """When distance to MoveOrder target starts increasing, AI switches to ShootOrder."""
        pitch = Pitch.standard()
        aim = pitch.right_goal_centre.with_z(1.0)
        target = Vector3(10.0, 0.0, 0.0)
        match, player = self._make_match(Vector3(9.0, 0.0, 0.0), aim)
        player.current_order = MoveOrder(target_position=target, sprint=True)
        ai = BallCarrierAttackerAI(aim, power_fraction=0.9)

        # First tick: dist = 1.0, recorded
        ai.act(player, match, 0)
        assert isinstance(player.current_order, MoveOrder)

        # Move the player away from the target (simulating repulsion)
        player.position = Vector3(7.0, 0.0, 0.0)  # now dist = 3.0 > 1.0
        ai.act(player, match, 1)
        assert isinstance(player.current_order, ShootOrder), (
            "AI should switch to ShootOrder when distance to target increases"
        )

    def test_noop_when_player_does_not_have_ball(self):
        """AI does nothing if the player does not have the ball."""
        pitch = Pitch.standard()
        aim = pitch.right_goal_centre.with_z(1.0)
        match, player = self._make_match(Vector3(0.0, 0.0, 0.0), aim)
        match.ball.possessed_by = None  # nobody has the ball
        ai = BallCarrierAttackerAI(aim)

        ai.act(player, match, 0)
        assert player.current_order is None, "AI should not act when player lacks the ball"

    def test_resets_prev_dist_when_ball_lost(self):
        """After the attacker loses the ball, _prev_dist_to_target is reset so
        a new possession cycle starts fresh."""
        pitch = Pitch.standard()
        aim = pitch.right_goal_centre.with_z(1.0)
        target = Vector3(10.0, 0.0, 0.0)
        match, player = self._make_match(Vector3(9.0, 0.0, 0.0), aim)
        player.current_order = MoveOrder(target_position=target, sprint=True)
        ai = BallCarrierAttackerAI(aim)

        ai.act(player, match, 0)  # prev_dist recorded
        match.ball.possessed_by = None  # ball lost
        ai.act(player, match, 1)  # should reset
        assert ai._prev_dist_to_target is None


# ---------------------------------------------------------------------------
# StagedGoalkeeperAI
# ---------------------------------------------------------------------------

class TestStagedGoalkeeperAI:
    """Tests for StagedGoalkeeperAI."""

    def test_does_not_act_while_gk_has_move_order(self):
        """AI does nothing while the GK still has a pending MoveOrder."""
        pitch = Pitch.standard()
        gk = make_player("gk", Team.LEFT, position=pitch.left_goal_centre, is_goalkeeper=True)
        ball = Ball.at_rest(Vector3(0.0, 0.0, 0.0))
        match = Match(pitch=pitch, players=[gk], ball=ball,
                      rng_reduction=1.0, rng=random.Random(0))
        gk.current_order = MoveOrder(target_position=pitch.left_goal_centre, sprint=False,
                                     max_speed_on_arrival_mps=0.0)
        ai = StagedGoalkeeperAI()

        ai.act(gk, match, 0)
        assert isinstance(gk.current_order, MoveOrder), (
            "AI should not issue SaveOrder while a MoveOrder is still active"
        )

    def test_issues_save_order_when_move_completes_and_ball_aimed_at_goal(self):
        """Once the GK's MoveOrder completes (order→None) and the loose ball is
        aimed at the GK's goal, AI issues SaveOrder."""
        pitch = Pitch.standard()
        gk = make_player("gk", Team.LEFT, position=pitch.left_goal_centre, is_goalkeeper=True)
        ball = Ball.at_rest(Vector3(0.0, 0.0, 0.0))
        ball.velocity = Vector3(-20.0, 0.0, 0.0)  # aimed at gk's (LEFT) goal
        match = Match(pitch=pitch, players=[gk], ball=ball,
                      rng_reduction=1.0, rng=random.Random(0))
        gk.current_order = None  # MoveOrder already completed
        ai = StagedGoalkeeperAI()

        ai.act(gk, match, 0)
        assert isinstance(gk.current_order, SaveOrder), (
            "AI should issue SaveOrder once the GK has no active order and the ball threatens goal"
        )

    def test_issues_move_order_to_goal_centre_when_move_completes_and_ball_not_threatening(self):
        """Once the GK's MoveOrder completes and the ball is not threatening,
        AI re-issues a goal-centre MoveOrder rather than SaveOrder -- as long
        as it isn't already parked there (see the parked/heading test below
        for that case)."""
        pitch = Pitch.standard()
        gk = make_player("gk", Team.LEFT,
                          position=pitch.left_goal_centre + Vector3(0.0, 5.0, 0.0),
                          is_goalkeeper=True)
        ball = Ball.at_rest(Vector3(0.0, 0.0, 0.0))
        match = Match(pitch=pitch, players=[gk], ball=ball,
                      rng_reduction=1.0, rng=random.Random(0))
        gk.current_order = None  # MoveOrder already completed
        ai = StagedGoalkeeperAI()

        ai.act(gk, match, 0)
        assert isinstance(gk.current_order, MoveOrder), (
            "AI should jog back to goal centre when the ball is not aimed at goal"
        )

    def test_faces_outfield_when_parked_at_goal_centre(self):
        """Once stationary at goal centre with no active order, the GK faces
        outfield (away from its own goal) rather than keeping whatever
        heading it last had -- covers both the 'ball loose, not threatening'
        and 'ball possessed by someone else' cases."""
        pitch = Pitch.standard()
        ai = StagedGoalkeeperAI()

        # Team.LEFT defends the left goal -> outfield heading is 0.0 (+x).
        gk = make_player("gk", Team.LEFT, position=pitch.left_goal_centre, is_goalkeeper=True)
        gk.heading_rad = 3.14  # wrong heading, as if left over from an earlier dive
        ball = Ball.at_rest(Vector3(0.0, 0.0, 0.0))
        match = Match(pitch=pitch, players=[gk], ball=ball,
                      rng_reduction=1.0, rng=random.Random(0))
        gk.current_order = None
        ai.act(gk, match, 0)
        assert gk.current_order is None
        assert gk.heading_rad == 0.0

        # Same, but the ball is possessed by someone else (the common case
        # during an attacker's build-up run) -- must also fix heading here,
        # not just in the "ball loose" branch.
        gk2 = make_player("gk2", Team.RIGHT, position=pitch.right_goal_centre, is_goalkeeper=True)
        gk2.heading_rad = 0.0  # wrong heading for a RIGHT-defending GK (faces own net)
        other = make_player("attacker", Team.LEFT, position=Vector3(0.0, 0.0, 0.0))
        ball2 = Ball.at_rest(other.position)
        ball2.possessed_by = other.player_id
        match2 = Match(pitch=pitch, players=[gk2, other], ball=ball2,
                       rng_reduction=1.0, rng=random.Random(0))
        gk2.current_order = None
        ai.act(gk2, match2, 0)
        assert gk2.current_order is None
        assert gk2.heading_rad == math.pi

    def test_does_not_override_save_order(self):
        """AI does not replace an existing SaveOrder with another SaveOrder."""
        pitch = Pitch.standard()
        gk = make_player("gk", Team.LEFT, position=pitch.left_goal_centre, is_goalkeeper=True)
        ball = Ball.at_rest(Vector3(0.0, 0.0, 0.0))
        match = Match(pitch=pitch, players=[gk], ball=ball,
                      rng_reduction=1.0, rng=random.Random(0))
        existing_save = SaveOrder()
        gk.current_order = existing_save
        ai = StagedGoalkeeperAI()

        ai.act(gk, match, 0)
        assert gk.current_order is existing_save, (
            "AI should not replace an active SaveOrder"
        )

    def test_noop_when_gk_has_ball(self):
        """AI does not issue SaveOrder when the GK currently has the ball."""
        pitch = Pitch.standard()
        gk = make_player("gk", Team.LEFT, position=pitch.left_goal_centre, is_goalkeeper=True)
        ball = Ball.at_rest(pitch.left_goal_centre)
        ball.possessed_by = gk.player_id
        match = Match(pitch=pitch, players=[gk], ball=ball,
                      rng_reduction=1.0, rng=random.Random(0))
        gk.current_order = None
        ai = StagedGoalkeeperAI()

        ai.act(gk, match, 0)
        assert gk.current_order is None, (
            "AI should not issue SaveOrder when the GK has possession"
        )


# ---------------------------------------------------------------------------
# Integration: 1v2 scenario completes within timeout
# ---------------------------------------------------------------------------

def test_1v2_scenario_completes_within_timeout():
    """The 1v2 scenario must end in a goal, save, or dispossession within 600 ticks."""
    from footballcoach.ui.scenarios import ScenarioLoop, SCENARIOS

    definition = next(s for s in SCENARIOS if s.key == "1v2")
    loop = ScenarioLoop(definition=definition, max_trials=1, timeout_ticks=600, linger_s=0.0)

    for _ in range(700):
        if loop.step():
            break

    assert loop.trial_count == 1, "1v2 scenario trial did not complete within 700 ticks"


# ---------------------------------------------------------------------------
# Integration: 2v2 scenario completes within timeout
# ---------------------------------------------------------------------------

def test_2v2_scenario_completes_within_timeout():
    """The 2v2 scenario must end in a goal, save, or dispossession within 600 ticks."""
    from footballcoach.ui.scenarios import ScenarioLoop, SCENARIOS

    definition = next(s for s in SCENARIOS if s.key == "2v2")
    loop = ScenarioLoop(definition=definition, max_trials=1, timeout_ticks=600, linger_s=0.0)

    for _ in range(700):
        if loop.step():
            break

    assert loop.trial_count == 1, "2v2 scenario trial did not complete within 700 ticks"

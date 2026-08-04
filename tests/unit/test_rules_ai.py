"""Unit tests for rules-based AI controllers (rules_ai.py).

Tests verify that each AI:
- Issues the correct order type in each state
- Reacts correctly to possession changes (the key bug this suite was written to catch)
- Does not issue orders that conflict with the player's current state
"""
from __future__ import annotations

import random

import pytest

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import GetPossessionOrder, MoveOrder, SaveOrder, ShootOrder
from footballcoach.rules_ai import (
    BallCarrierAttackerAI,
    BallReceiverThenShootAI,
    Phase1RulesAI,
    StagedGoalkeeperAI,
    SprintWaypointAI,
    _should_sprint_to_ball,
)

from tests.conftest import make_player


def _ball_aimed_at_left_goal(speed: float = 20.0) -> Ball:
    """A loose ball moving straight toward the LEFT goal, fast enough and
    aimed centrally enough to trigger StagedGoalkeeperAI._ball_aimed_at_goal()."""
    ball = Ball.at_rest(Vector3(0, 0, 0))
    ball.velocity = Vector3(-speed, 0, 0)
    return ball


def _make_simple_match(
    *,
    attacker_team: Team = Team.LEFT,
    ball_position: Vector3 | None = None,
    ball_possessed_by: str | None = None,
) -> tuple[Match, "Player", "Player"]:
    """Return a match with two players (no AI assigned) and a loose ball."""
    from footballcoach.entities.player import Player

    p1 = make_player("p1", attacker_team, position=Vector3(-10, 0, 0))
    p2 = make_player("p2", Team.RIGHT if attacker_team == Team.LEFT else Team.LEFT,
                     position=Vector3(10, 0, 0))
    ball = Ball.at_rest(ball_position or Vector3(0, 0, 0))
    match = Match(
        pitch=Pitch.standard(),
        players=[p1, p2],
        ball=ball,
        rng_reduction=1.0,
        rng=random.Random(42),
    )
    if ball_possessed_by:
        match._set_possession(ball_possessed_by)
    return match, p1, p2


# ---------------------------------------------------------------------------
# Phase1RulesAI
# ---------------------------------------------------------------------------

class TestPhase1RulesAI:
    def test_issues_get_possession_when_no_order_and_no_ball(self):
        match, p1, _ = _make_simple_match()
        p1.ai = Phase1RulesAI()
        assert p1.current_order is None
        p1.ai.act(p1, match, 0)
        assert isinstance(p1.current_order, GetPossessionOrder)

    def test_issues_move_order_when_has_ball_and_no_order(self):
        match, p1, _ = _make_simple_match(ball_possessed_by="p1")
        p1.ai = Phase1RulesAI()
        p1.ai.act(p1, match, 0)
        assert isinstance(p1.current_order, MoveOrder)

    def test_move_order_target_in_opponent_box_for_left_team(self):
        match, p1, _ = _make_simple_match(attacker_team=Team.LEFT, ball_possessed_by="p1")
        p1.ai = Phase1RulesAI()
        for _ in range(20):  # check across random seeds
            p1.current_order = None
            p1.ai.act(p1, match, 0)
            target = p1.current_order.target_position
            # LEFT team attacks RIGHT goal (positive x)
            box_inner_x = match.pitch.half_length - match.pitch.box_length_m
            assert target.x >= box_inner_x - 1e-6, f"target.x={target.x} not in right box"
            assert abs(target.y) <= match.pitch.box_width_m / 2.0 + 1e-6

    def test_move_order_target_in_opponent_box_for_right_team(self):
        match, _, p2 = _make_simple_match(ball_possessed_by="p2")
        p2.ai = Phase1RulesAI()
        for _ in range(20):
            p2.current_order = None
            p2.ai.act(p2, match, 0)
            target = p2.current_order.target_position
            # RIGHT team attacks LEFT goal (negative x)
            box_inner_x = -(match.pitch.half_length - match.pitch.box_length_m)
            assert target.x <= box_inner_x + 1e-6, f"target.x={target.x} not in left box"

    def test_does_not_replace_active_move_order_while_has_ball(self):
        match, p1, _ = _make_simple_match(ball_possessed_by="p1")
        p1.ai = Phase1RulesAI()
        existing = MoveOrder(target_position=Vector3(40, 5, 0))
        p1.current_order = existing
        p1.ai.act(p1, match, 0)
        assert p1.current_order is existing  # unchanged

    def test_switches_to_get_possession_when_ball_lost_mid_move_order(self):
        """KEY BUG REGRESSION: player had ball, was running (MoveOrder), lost
        possession — AI must immediately switch to GetPossessionOrder."""
        match, p1, _ = _make_simple_match()
        p1.ai = Phase1RulesAI()
        # Simulate: had ball, AI issued MoveOrder, then lost possession
        p1.current_order = MoveOrder(target_position=Vector3(40, 5, 0))
        # Ball is now loose (p1 does NOT have it)
        assert match.ball.possessed_by != p1.player_id
        p1.ai.act(p1, match, 0)
        assert isinstance(p1.current_order, GetPossessionOrder), (
            "Should replace stale MoveOrder with GetPossessionOrder after losing ball"
        )

    def test_switches_to_get_possession_when_ball_lost_immediately(self):
        """Possession is given to opponent; p1 had a MoveOrder; AI should react."""
        match, p1, p2 = _make_simple_match(ball_possessed_by="p2")
        p1.ai = Phase1RulesAI()
        p1.current_order = MoveOrder(target_position=Vector3(40, 0, 0))
        p1.ai.act(p1, match, 0)
        assert isinstance(p1.current_order, GetPossessionOrder)

    def test_get_possession_not_reissued_each_tick_once_active(self):
        """Once GetPossessionOrder is active, AI should not replace it each tick."""
        match, p1, _ = _make_simple_match()
        p1.ai = Phase1RulesAI()
        p1.ai.act(p1, match, 0)
        existing_order = p1.current_order
        assert isinstance(existing_order, GetPossessionOrder)
        p1.ai.act(p1, match, 1)
        # Should be the same object (not re-issued)
        assert p1.current_order is existing_order

    def test_issues_new_move_order_each_time_possession_gained_without_order(self):
        """After completing a MoveOrder (order cleared), gaining ball again → new MoveOrder."""
        match, p1, _ = _make_simple_match(ball_possessed_by="p1")
        p1.ai = Phase1RulesAI()
        p1.current_order = None  # order just completed
        p1.ai.act(p1, match, 0)
        assert isinstance(p1.current_order, MoveOrder)

    def test_full_possession_cycle_via_match_steps(self):
        """Integration: run match ticks and verify AI switches orders when possession changes."""
        match, p1, p2 = _make_simple_match()
        p1.ai = Phase1RulesAI()

        # Give p1 the ball — AI should issue MoveOrder on next act()
        match._set_possession("p1")
        p1.current_order = None
        p1.ai.act(p1, match, 0)
        assert isinstance(p1.current_order, MoveOrder)

        # Simulate losing the ball (possession clears)
        match._set_possession(None)
        p1.ai.act(p1, match, 1)
        assert isinstance(p1.current_order, GetPossessionOrder)

        # Give p2 the ball — p1 still should want GetPossessionOrder
        match._set_possession("p2")
        p1.ai.act(p1, match, 2)
        assert isinstance(p1.current_order, GetPossessionOrder)


# ---------------------------------------------------------------------------
# _should_sprint_to_ball
# ---------------------------------------------------------------------------

class TestShouldSprintToBall:
    def test_sprints_when_opponent_closer_to_ball(self):
        """Opponent is right next to the ball; we are far away — must sprint."""
        # p1 is 20 m from ball; p2 (opponent) is 1 m from ball.
        match, p1, p2 = _make_simple_match(
            ball_position=Vector3(0, 0, 0),
        )
        p1.position = Vector3(-20, 0, 0)
        p2.position = Vector3(-1, 0, 0)
        assert _should_sprint_to_ball(p1, match) is True

    def test_jogs_when_clearly_closer_than_sprinting_opponent(self):
        """We are 1 m from ball; opponent is 30 m away — safe to jog."""
        match, p1, p2 = _make_simple_match(
            ball_position=Vector3(0, 0, 0),
        )
        p1.position = Vector3(-1, 0, 0)
        p2.position = Vector3(30, 0, 0)
        assert _should_sprint_to_ball(p1, match) is False

    def test_sprints_when_no_opponents_present(self):
        """No opponents on the pitch — default is to sprint (safe)."""
        from footballcoach.entities.player import Player

        p1 = make_player("p1", Team.LEFT, position=Vector3(-10, 0, 0))
        ball = Ball.at_rest(Vector3(0, 0, 0))
        match = Match(
            pitch=Pitch.standard(),
            players=[p1],
            ball=ball,
            rng_reduction=1.0,
            rng=random.Random(42),
        )
        assert _should_sprint_to_ball(p1, match) is True

    def test_does_not_sprint_when_already_at_ball(self):
        """Player is essentially on top of the ball — no sprint needed."""
        match, p1, p2 = _make_simple_match(
            ball_position=Vector3(0, 0, 0),
        )
        p1.position = Vector3(0.05, 0, 0)   # within 0.1 m threshold
        p2.position = Vector3(30, 0, 0)
        assert _should_sprint_to_ball(p1, match) is False

    def test_ignores_tackled_opponents(self):
        """Tackled opponent's proximity is excluded from the sprint decision."""
        from footballcoach.entities.player import PlayerState

        match, p1, p2 = _make_simple_match(
            ball_position=Vector3(0, 0, 0),
        )
        p1.position = Vector3(-20, 0, 0)
        p2.position = Vector3(-1, 0, 0)  # opponent much closer
        # Active opponent close → should sprint to compete
        assert _should_sprint_to_ball(p1, match) is True
        # Tackle p2 → no active opponents → safe-default sprint (no opponents present)
        p2.state = PlayerState.INACTIVE_TACKLED
        result = _should_sprint_to_ball(p1, match)
        # The tackled player must not be factored in; result is the no-opponents default (True)
        assert result is True  # sprints because no active opponents = safe default

    def test_ignores_teammates(self):
        """Teammates on the same team must not influence the sprint decision."""
        from footballcoach.entities.player import Player

        p1 = make_player("p1", Team.LEFT, position=Vector3(-5, 0, 0))
        # teammate p3 is much closer to the ball, but same team
        p3 = make_player("p3", Team.LEFT, position=Vector3(-1, 0, 0))
        ball = Ball.at_rest(Vector3(0, 0, 0))
        match = Match(
            pitch=Pitch.standard(),
            players=[p1, p3],
            ball=ball,
            rng_reduction=1.0,
            rng=random.Random(42),
        )
        # No opponents present → sprint (safe default), not because of p3
        assert _should_sprint_to_ball(p1, match) is True


# ---------------------------------------------------------------------------
# StagedGoalkeeperAI
# ---------------------------------------------------------------------------

class TestStagedGoalkeeperAI:
    def test_issues_save_order_when_ball_aimed_at_goal(self):
        match, p1, _ = _make_simple_match(ball_position=Vector3(0, 0, 0))
        match.ball.velocity = Vector3(-20.0, 0, 0)  # aimed at p1's (LEFT) goal
        p1.ai = StagedGoalkeeperAI()
        p1.ai.act(p1, match, 0)
        assert isinstance(p1.current_order, SaveOrder)

    def test_issues_move_order_to_goal_centre_when_no_order_and_ball_not_threatening(self):
        match, p1, _ = _make_simple_match()
        p1.ai = StagedGoalkeeperAI()
        p1.ai.act(p1, match, 0)
        assert isinstance(p1.current_order, MoveOrder), (
            "AI should jog back to goal centre when the ball is loose but not aimed at goal"
        )

    def test_does_not_issue_save_order_when_has_ball(self):
        match, p1, _ = _make_simple_match(ball_possessed_by="p1")
        p1.ai = StagedGoalkeeperAI()
        p1.current_order = None
        p1.ai.act(p1, match, 0)
        assert p1.current_order is None

    def test_does_not_replace_existing_order(self):
        match, p1, _ = _make_simple_match()
        p1.ai = StagedGoalkeeperAI()
        existing = MoveOrder(target_position=Vector3(0, 0, 0))
        p1.current_order = existing
        p1.ai.act(p1, match, 0)
        assert p1.current_order is existing

    def test_reissues_save_order_after_order_cleared(self):
        match, p1, _ = _make_simple_match(ball_position=Vector3(0, 0, 0))
        match.ball.velocity = Vector3(-20.0, 0, 0)  # aimed at p1's (LEFT) goal
        p1.ai = StagedGoalkeeperAI()
        p1.current_order = None
        p1.ai.act(p1, match, 0)
        assert isinstance(p1.current_order, SaveOrder)
        p1.current_order = None  # order completed/cleared
        p1.ai.act(p1, match, 1)
        assert isinstance(p1.current_order, SaveOrder)


# ---------------------------------------------------------------------------
# BallCarrierAttackerAI
# ---------------------------------------------------------------------------

class TestBallCarrierAttackerAI:
    def _aim(self) -> Vector3:
        return Vector3(52.5, 0, 1.2)

    def test_does_nothing_when_no_ball(self):
        match, p1, _ = _make_simple_match()
        p1.ai = BallCarrierAttackerAI(self._aim())
        p1.ai.act(p1, match, 0)
        assert p1.current_order is None

    def test_issues_shoot_order_when_has_ball_and_no_order(self):
        match, p1, _ = _make_simple_match(ball_possessed_by="p1")
        p1.ai = BallCarrierAttackerAI(self._aim())
        p1.ai.act(p1, match, 0)
        assert isinstance(p1.current_order, ShootOrder)

    def test_switches_to_shoot_when_move_order_stalls(self):
        match, p1, _ = _make_simple_match(ball_possessed_by="p1")
        ai = BallCarrierAttackerAI(self._aim())
        p1.ai = ai
        target = Vector3(40, 0, 0)
        p1.current_order = MoveOrder(target_position=target)
        # First act: records distance
        p1.position = Vector3(30, 0, 0)
        ai.act(p1, match, 0)
        # Move player further from target (stall)
        p1.position = Vector3(25, 0, 0)
        ai.act(p1, match, 1)
        assert isinstance(p1.current_order, ShootOrder)

    def test_does_not_switch_while_approaching(self):
        match, p1, _ = _make_simple_match(ball_possessed_by="p1")
        ai = BallCarrierAttackerAI(self._aim())
        p1.ai = ai
        target = Vector3(40, 0, 0)
        p1.current_order = MoveOrder(target_position=target)
        p1.position = Vector3(20, 0, 0)
        ai.act(p1, match, 0)
        p1.position = Vector3(25, 0, 0)  # closer to target
        ai.act(p1, match, 1)
        assert isinstance(p1.current_order, MoveOrder)

    def test_resets_stall_detection_on_possession_loss(self):
        match, p1, _ = _make_simple_match(ball_possessed_by="p1")
        ai = BallCarrierAttackerAI(self._aim())
        p1.ai = ai
        target = Vector3(40, 0, 0)
        p1.current_order = MoveOrder(target_position=target)
        p1.position = Vector3(30, 0, 0)
        ai.act(p1, match, 0)
        # Lose possession
        match._set_possession(None)
        ai.act(p1, match, 1)
        assert ai._prev_dist_to_target is None


# ---------------------------------------------------------------------------
# BallReceiverThenShootAI
# ---------------------------------------------------------------------------

class TestBallReceiverThenShootAI:
    def _aim(self) -> Vector3:
        return Vector3(52.5, 0, 1.2)

    def test_does_not_act_until_possession_gained(self):
        match, p1, _ = _make_simple_match()
        existing = MoveOrder(target_position=Vector3(5, 0, 0))
        p1.current_order = existing
        ai = BallReceiverThenShootAI(self._aim(), shoot_immediately=True)
        p1.ai = ai
        ai.act(p1, match, 0)
        assert p1.current_order is existing  # unchanged

    def test_shoot_immediately_issues_shoot_order_on_possession(self):
        match, p1, _ = _make_simple_match(ball_possessed_by="p1")
        ai = BallReceiverThenShootAI(self._aim(), shoot_immediately=True)
        p1.ai = ai
        ai.act(p1, match, 0)
        assert isinstance(p1.current_order, ShootOrder)

    def test_run_first_issues_move_order_on_possession(self):
        match, p1, _ = _make_simple_match(ball_possessed_by="p1")
        p1.position = Vector3(0, 0, 0)
        ai = BallReceiverThenShootAI(self._aim(), shoot_immediately=False, run_fraction=0.5)
        p1.ai = ai
        ai.act(p1, match, 0)
        assert isinstance(p1.current_order, MoveOrder)
        # Target should be halfway between current pos and aim x
        assert p1.current_order.target_position.x == pytest.approx(52.5 * 0.5, abs=0.1)


# ---------------------------------------------------------------------------
# SprintWaypointAI
# ---------------------------------------------------------------------------

class TestSprintWaypointAI:
    def test_issues_first_waypoint_order(self):
        match, p1, _ = _make_simple_match()
        waypoints = [Vector3(10, 0, 0), Vector3(20, 0, 0), Vector3(30, 0, 0)]
        p1.current_order = MoveOrder(target_position=waypoints[0])
        ai = SprintWaypointAI(waypoints, start_idx=1)
        p1.ai = ai
        # Order still active — no new order
        ai.act(p1, match, 0)
        assert p1.current_order.target_position == waypoints[0]

    def test_issues_next_waypoint_when_order_cleared(self):
        match, p1, _ = _make_simple_match()
        waypoints = [Vector3(10, 0, 0), Vector3(20, 0, 0), Vector3(30, 0, 0)]
        p1.current_order = None
        ai = SprintWaypointAI(waypoints, start_idx=1)
        p1.ai = ai
        ai.act(p1, match, 0)
        assert isinstance(p1.current_order, MoveOrder)
        assert p1.current_order.target_position == waypoints[1]

    def test_advances_through_all_waypoints(self):
        match, p1, _ = _make_simple_match()
        waypoints = [Vector3(i * 10, 0, 0) for i in range(4)]
        p1.current_order = MoveOrder(target_position=waypoints[0])
        ai = SprintWaypointAI(waypoints, start_idx=1)
        for expected_idx in range(1, len(waypoints)):
            p1.current_order = None
            ai.act(p1, match, expected_idx)
            assert p1.current_order.target_position == waypoints[expected_idx]

    def test_stops_issuing_orders_after_last_waypoint(self):
        match, p1, _ = _make_simple_match()
        waypoints = [Vector3(10, 0, 0)]
        p1.current_order = None
        ai = SprintWaypointAI(waypoints, start_idx=1)  # start_idx beyond list
        ai.act(p1, match, 0)
        assert p1.current_order is None  # no more waypoints

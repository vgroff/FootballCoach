"""Regression tests for bugs observed in the UI scenarios."""
from __future__ import annotations

import random

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.entities.player import PlayerState
from footballcoach.mathutils import Vector3
from footballcoach.orders import ChaseTackleOrder, MoveOrder, SaveOrder
from tests.conftest import make_player


# ---------------------------------------------------------------------------
# Bug 1: Tackle scenario — defender stays still
# ChaseTackleOrder must move the defender toward the attacker each tick,
# not silently complete because they aren't touching yet.
# ---------------------------------------------------------------------------

def test_chase_tackle_defender_moves_toward_attacker():
    """Defender with ChaseTackleOrder must advance toward carrier each tick."""
    pitch = Pitch.standard()
    carrier = make_player("carrier", Team.RIGHT, position=Vector3(0, 0, 0), dribbling=0.5)
    defender = make_player("defender", Team.LEFT, position=Vector3(-8, 0, 0), tackling=0.8)

    ball = Ball.at_rest(Vector3(0, 0, 0))
    ball.possessed_by = carrier.player_id

    # Give carrier a move order so it stays active
    carrier.current_order = MoveOrder(target_position=Vector3(20, 0, 0), sprint=False)

    match = Match(pitch=pitch, players=[carrier, defender], ball=ball,
                  rng_reduction=1.0, rng=random.Random(0))
    defender.current_order = ChaseTackleOrder(target_player_id=carrier.player_id)

    start_x = defender.position.x
    match.step()

    assert defender.current_order is not None, "ChaseTackleOrder completed immediately (defender never chased)"
    assert defender.position.x > start_x, (
        f"Defender didn't move: started at x={start_x}, now at x={defender.position.x}"
    )


def test_chase_tackle_eventually_contacts_and_resolves():
    """Defender placed very close to carrier must contact and resolve the tackle."""
    pitch = Pitch.standard()
    carrier = make_player("carrier", Team.RIGHT, position=Vector3(0, 0, 0), dribbling=0.1)
    # Place defender within touching distance already
    defender = make_player("defender", Team.LEFT, position=Vector3(0.5, 0, 0), tackling=0.9)

    ball = Ball.at_rest(Vector3(0, 0, 0))
    ball.possessed_by = carrier.player_id

    match = Match(pitch=pitch, players=[carrier, defender], ball=ball,
                  rng_reduction=1.0, rng=random.Random(0))
    defender.current_order = ChaseTackleOrder(target_player_id=carrier.player_id)

    match.step()

    # With rng_reduction=1.0 and tackling=0.9 vs dribbling=0.1, tackler wins
    assert defender.current_order is None, "ChaseTackleOrder didn't resolve on contact"
    assert ball.possessed_by == defender.player_id, "Defender didn't win the ball"


# ---------------------------------------------------------------------------
# Bug 2: Sprint scenario stops after 1-2 waypoints
# The ScenarioLoop._trial_outcome fires "other" the tick a MoveOrder
# completes (current_order becomes None) before the on_tick hook can
# issue the next waypoint. Test that a player mid-run through sequential
# waypoints doesn't have its trial ended prematurely.
# ---------------------------------------------------------------------------

def test_move_order_completion_with_no_order_does_not_immediately_end():
    """A player that just completed a MoveOrder (current_order=None, ball still)
    should not be considered 'done' on the very same tick — the next waypoint
    should be issueable on the following tick."""
    from footballcoach.ui.scenarios import ScenarioLoop, ScenarioDefinition
    from footballcoach.rules_ai import SprintWaypointAI

    pitch = Pitch.standard()

    def build_two_waypoint_sprint(rng_reduction=0.3):
        rng = random.Random(42)
        player = make_player("runner", Team.LEFT, attr_value=0.8,
                             position=Vector3(0, 0, 0))
        ball = Ball.at_rest(Vector3(0, 20, 0))
        from footballcoach.config import load_gameplay_config
        ui_cfg = load_gameplay_config().get("ui", {})
        m = Match(pitch=pitch, players=[player], ball=ball,
                  rng_reduction=rng_reduction, rng=rng,
                  goal_linger_s=ui_cfg.get("goal_linger_s", 3.0))
        waypoints = [Vector3(3, 0, 0), Vector3(6, 0, 0)]
        player.current_order = MoveOrder(target_position=waypoints[0], sprint=True)
        player.ai = SprintWaypointAI(waypoints, start_idx=1)
        return m

    defn = ScenarioDefinition(
        key="test_sprint", label="test", description="",
        build=build_two_waypoint_sprint,
        on_tick=None,
    )
    loop = ScenarioLoop(definition=defn, max_trials=0, timeout_ticks=300)

    # Run up to 300 ticks — we expect the runner to reach the second waypoint
    # WITHOUT the trial ending prematurely after the first one.
    reached_second_waypoint = False
    for _ in range(300):
        player = loop.match.player_by_id("runner")
        if player.current_order is not None:
            order = player.current_order
            if isinstance(order, MoveOrder) and order.target_position.x > 4:
                reached_second_waypoint = True
                break
        loop.step()

    assert reached_second_waypoint, (
        "Sprint scenario never issued the second waypoint — trial ended prematurely "
        "when MoveOrder completed and current_order briefly became None"
    )


# ---------------------------------------------------------------------------
# Bug 3: Pass scenario — ball morphing through receiver
# A slowly rolling ball arriving within pickup_radius_m of a stationary
# receiver must trigger CONTROLLING_BALL state, not fly through them.
# ---------------------------------------------------------------------------

def test_slow_ball_picked_up_by_stationary_receiver():
    """A ball rolling at 2 m/s directly toward a stationary player must be
    picked up (CONTROLLING_BALL) once it enters the pickup radius.
    Ball starts 0.3m away so it crosses the 0.4m pickup radius within 1 tick.
    This tests the core pickup mechanic — the 'morphing through' bug would
    manifest as the ball passing through without triggering CONTROLLING_BALL."""
    pitch = Pitch.standard()
    receiver = make_player("recv", Team.LEFT, position=Vector3(0, 0, 0), ball_control=0.8)
    # Start ball just inside pickup radius (0.4m) — triggers pickup immediately
    ball = Ball.at_rest(Vector3(-0.3, 0, 0))
    ball.velocity = Vector3(2.0, 0, 0)  # rolling toward receiver

    match = Match(pitch=pitch, players=[receiver], ball=ball,
                  rng_reduction=1.0, rng=random.Random(0))

    match.step()

    assert (receiver.state == PlayerState.CONTROLLING_BALL or
            ball.possessed_by == receiver.player_id), (
        f"Receiver never acquired ball: state={receiver.state}, "
        f"possessed_by={ball.possessed_by}, ball_pos={ball.position}"
    )


def test_ball_approaching_from_5m_eventually_picked_up():
    """A ball rolling at 4 m/s from 5m away must be picked up by a stationary
    receiver — it must not fly past them. This is the 'morphing through' bug
    scenario from the passing drills."""
    pitch = Pitch.standard()
    receiver = make_player("recv", Team.LEFT, position=Vector3(0, 0, 0), ball_control=0.8)
    ball = Ball.at_rest(Vector3(-5, 0, 0))
    ball.velocity = Vector3(4.0, 0, 0)  # fast enough to cross 5m quickly

    match = Match(pitch=pitch, players=[receiver], ball=ball,
                  rng_reduction=1.0, rng=random.Random(0))

    # Run up to 5 sim-seconds (150 ticks)
    for _ in range(150):
        match.step()
        if receiver.state == PlayerState.CONTROLLING_BALL or ball.possessed_by == receiver.player_id:
            break

    assert (receiver.state == PlayerState.CONTROLLING_BALL or
            ball.possessed_by == receiver.player_id), (
        f"Ball morphed through receiver: state={receiver.state}, "
        f"possessed_by={ball.possessed_by}, final ball_pos={ball.position}"
    )


# ---------------------------------------------------------------------------
# Bug 4: 1v2 SaveOrder — keeper should stop running to other goal.
# A SaveOrder for a LEFT team GK defending the LEFT goal should move toward
# the LEFT goal line, not the RIGHT goal line.
# ---------------------------------------------------------------------------

def test_save_order_left_team_gk_moves_toward_left_goal():
    """LEFT-team GK with SaveOrder should move toward the LEFT goal (-x)."""
    pitch = Pitch.standard()
    ball = Ball.at_rest(Vector3(0, 0, 0))
    gk = make_player("gk", Team.LEFT, position=Vector3(0, 0, 0), is_goalkeeper=True)

    match = Match(pitch=pitch, players=[gk], ball=ball,
                  rng_reduction=1.0, rng=random.Random(0))
    gk.current_order = SaveOrder()

    initial_x = gk.position.x
    for _ in range(10):
        match.step()

    assert gk.position.x <= initial_x + 0.1, (
        f"LEFT-team GK moved rightward (away from own goal): {initial_x} -> {gk.position.x}"
    )


def test_save_order_right_team_gk_moves_toward_right_goal():
    """RIGHT-team GK with SaveOrder should move toward the RIGHT goal (+x).
    This is the correct setup for the 1v2 scenario (attacker attacks +x)."""
    pitch = Pitch.standard()
    ball = Ball.at_rest(Vector3(0, 0, 0))
    # GK starts at centre, defends right goal (+x)
    gk = make_player("gk", Team.RIGHT, position=Vector3(0, 0, 0), is_goalkeeper=True)

    match = Match(pitch=pitch, players=[gk], ball=ball,
                  rng_reduction=1.0, rng=random.Random(0))
    gk.current_order = SaveOrder()

    initial_x = gk.position.x
    for _ in range(10):
        match.step()

    assert gk.position.x >= initial_x, (
        f"RIGHT-team GK moved leftward (away from own goal): {initial_x} -> {gk.position.x}"
    )


def test_1v2_gk_team_is_right():
    """In the 1v2 scenario the GK must be Team.RIGHT (defends right goal at +x).
    The old bug was Team.LEFT which sent the GK to the opposite end of the pitch."""
    from footballcoach.ui.scenarios import build_1v2_scenario
    match = build_1v2_scenario(rng_reduction=1.0)
    gk = match.player_by_id("keeper")
    assert gk.team == Team.RIGHT, (
        f"1v2 GK is Team.{gk.team.name} but should be Team.RIGHT"
    )


# ---------------------------------------------------------------------------
# Bug 5: 1v2 attacker circles instead of shooting
# The OneVTwoController issues a ShootOrder only when:
#   (a) the attacker has the ball, AND
#   (b) current_order is None (MoveOrder has completed).
# If the MoveOrder never completes (e.g. repulsion from the defender keeps
# deflecting the attacker away from the target) the ShootOrder is never
# issued and the attacker loops indefinitely.
# ---------------------------------------------------------------------------

def test_1v2_move_order_completes_with_stationary_obstacle():
    """A MoveOrder must complete even with a stationary player directly in the
    path. The geometry mirrors the visual repulsion scenario: 40 m total run,
    obstacle at the midpoint, giving the attacker 20 m of runway after bypass
    to shed lateral momentum and converge on the target."""
    pitch = Pitch.standard()

    attacker_start = Vector3(-20.0, 0.0, 0.0)
    move_target = Vector3(20.0, 0.0, 0.0)
    # Obstacle dead on the path at the midpoint — 20 m from the target,
    # so the attacker has plenty of room to straighten up after passing.
    obstacle_pos = Vector3(0.0, 0.0, 0.0)

    attacker = make_player("attacker", Team.LEFT, attr_value=0.9, position=attacker_start)
    obstacle = make_player("obstacle", Team.RIGHT, attr_value=0.6, position=obstacle_pos)

    ball = Ball.at_rest(attacker_start)
    ball.possessed_by = attacker.player_id

    match = Match(pitch=pitch, players=[attacker, obstacle], ball=ball,
                  rng_reduction=1.0, rng=random.Random(42))

    attacker.current_order = MoveOrder(target_position=move_target, sprint=True)
    # obstacle has no order — stays still, providing a fixed repulsion source

    MAX_TICKS = 400
    move_completed = False
    for _ in range(MAX_TICKS):
        match.step()
        if attacker.current_order is None:
            move_completed = True
            break

    assert move_completed, (
        f"Attacker MoveOrder did not complete in {MAX_TICKS} ticks with stationary obstacle on path. "
        f"Final order: {attacker.current_order}, "
        f"attacker pos: {attacker.position}, move target: {move_target}"
    )


def test_1v2_controller_issues_shoot_after_move():
    """After the attacker's MoveOrder finishes, BallCarrierAttackerAI (via
    player.ai) must issue a ShootOrder within a handful of ticks."""
    from footballcoach.orders import GetPossessionOrder, ShootOrder
    from footballcoach.rules_ai import BallCarrierAttackerAI, Phase1RulesAI, StagedGoalkeeperAI

    pitch = Pitch.standard()
    goal_centre = pitch.right_goal_centre

    attacker_start = Vector3(pitch.half_length - 25.0, 5.0, 0.0)
    move_target = Vector3(
        attacker_start.x + (goal_centre.x - attacker_start.x) * 0.3,
        attacker_start.y + (goal_centre.y - attacker_start.y) * 0.3,
        0.0,
    )
    aim_point = goal_centre + Vector3(0, 0, 1.1)

    attacker = make_player("attacker", Team.LEFT, attr_value=0.9, position=attacker_start)
    defender = make_player("defender", Team.RIGHT, attr_value=0.6,
                           position=Vector3(pitch.half_length - 12.0, 3.0, 0.0))
    gk = make_player("keeper", Team.RIGHT, attr_value=0.6, position=goal_centre, is_goalkeeper=True)

    ball = Ball.at_rest(attacker_start)
    ball.possessed_by = attacker.player_id

    match = Match(pitch=pitch, players=[attacker, defender, gk], ball=ball,
                  rng_reduction=1.0, rng=random.Random(42))

    attacker.current_order = MoveOrder(target_position=move_target, sprint=True)
    defender.current_order = GetPossessionOrder()
    attacker.ai = BallCarrierAttackerAI(aim_point, power_fraction=0.9)
    defender.ai = Phase1RulesAI()
    gk.ai = StagedGoalkeeperAI()

    MAX_TICKS = 400
    shoot_issued = False
    for tick in range(MAX_TICKS):
        # ai.act() runs inside match.step() → ShootOrder issued → immediately
        # consumed within the same step().  Capture via a flag on the order.
        # Workaround: patch the AI's act to track firing.
        _orig_act = attacker.ai.act
        def _tracked_act(player, m, t, _orig=_orig_act):
            _orig(player, m, t)
            nonlocal shoot_issued
            if isinstance(player.current_order, ShootOrder):
                shoot_issued = True
        attacker.ai.act = _tracked_act

        match.step()
        if shoot_issued:
            break
        if match.ball.possessed_by != attacker.player_id:
            break

    assert shoot_issued, (
        f"ShootOrder was never issued after MoveOrder in {MAX_TICKS} ticks. "
        f"Final order: {attacker.current_order}, "
        f"possessed_by: {match.ball.possessed_by}"
    )

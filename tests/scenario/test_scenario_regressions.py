"""Regression tests for three scenario bugs.

1. GetPossession in pass scenario: receiver should move TOWARD the ball,
   not away from it.
2. Sprint scenario ScenarioLoop: must not terminate between waypoints when
   current_order briefly becomes None.
3. 1v2 defender: GetPossessionOrder must close distance to the carrier
   each tick and eventually make contact.
"""
from __future__ import annotations

import random

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.entities.player import PlayerState
from footballcoach.mathutils import Vector3
from footballcoach.orders import GetPossessionOrder, MoveOrder
from footballcoach.ui.scenarios import (
    ScenarioDefinition,
    ScenarioLoop,
    SprintController,
    _sprint_on_tick,
    _PASS_SCENARIO_GET_POSSESSION_RADIUS_M,
    _pass_on_tick,
)
from tests.conftest import make_player


# ---------------------------------------------------------------------------
# Bug 1: pass scenario GetPossession — receiver must move TOWARD the ball
# ---------------------------------------------------------------------------

def test_pass_scenario_receiver_moves_toward_ball_on_get_possession():
    """When _pass_on_tick fires GetPossessionOrder, the receiver must close
    the distance to the ball each tick, not increase it."""
    pitch = Pitch.standard()
    receiver = make_player("receiver", Team.LEFT, attr_value=0.8,
                           position=Vector3(5, 0, 0))
    # Ball rolling slowly toward receiver
    ball = Ball.at_rest(Vector3(3, 0, 0))
    ball.velocity = Vector3(0.5, 0, 0)

    match = Match(pitch=pitch, players=[receiver], ball=ball,
                  rng_reduction=1.0, rng=random.Random(0))

    # Manually fire the on_tick that triggers GetPossessionOrder
    _pass_on_tick(match, 0)
    assert isinstance(receiver.current_order, GetPossessionOrder), \
        "Ball is within radius, receiver should have GetPossessionOrder"

    initial_dist = receiver.position.xy().distance_to(ball.position.xy())
    match.step()
    final_dist = receiver.position.xy().distance_to(ball.position.xy())

    assert final_dist < initial_dist, (
        f"Receiver moved AWAY from ball: dist {initial_dist:.2f} -> {final_dist:.2f}. "
        "GetPossession is sending them in the wrong direction."
    )


def test_pass_on_tick_does_not_trigger_outside_radius():
    """_pass_on_tick must not give a GetPossessionOrder if ball is beyond the radius."""
    pitch = Pitch.standard()
    receiver = make_player("receiver", Team.LEFT, attr_value=0.8,
                           position=Vector3(20, 0, 0))
    ball = Ball.at_rest(Vector3(0, 0, 0))  # 20m away, well beyond radius

    match = Match(pitch=pitch, players=[receiver], ball=ball,
                  rng_reduction=1.0, rng=random.Random(0))
    _pass_on_tick(match, 0)

    assert not isinstance(receiver.current_order, GetPossessionOrder), \
        "Should not trigger GetPossession when ball is far away"


def test_pass_scenario_radius_is_six_metres():
    """The activation radius should be 6 m (not 4 m)."""
    assert _PASS_SCENARIO_GET_POSSESSION_RADIUS_M == 6.0, (
        f"Radius is {_PASS_SCENARIO_GET_POSSESSION_RADIUS_M} m, expected 6.0 m"
    )


# ---------------------------------------------------------------------------
# Bug 2: Sprint ScenarioLoop must not terminate between waypoints
# ---------------------------------------------------------------------------

def _build_multi_waypoint_sprint(rng_reduction: float = 1.0) -> Match:
    """Three close waypoints so the runner reaches each quickly."""
    from footballcoach.config import load_physics_config
    from footballcoach.entities import PlayerAttributes
    pitch = Pitch.standard()
    player = make_player("runner", Team.LEFT, attr_value=0.8,
                         position=Vector3(-5, 0, 0))
    ball = Ball.at_rest(Vector3(0, 20, 0))
    ui_cfg = load_physics_config().get("ui", {})
    m = Match(pitch=pitch, players=[player], ball=ball,
              rng_reduction=rng_reduction, rng=random.Random(0),
              goal_linger_s=ui_cfg.get("goal_linger_s", 3.0))
    waypoints = [Vector3(0, 0, 0), Vector3(5, 0, 0), Vector3(10, 0, 0)]
    ctrl = SprintController(player.player_id, waypoints)
    player.current_order = MoveOrder(target_position=waypoints[0], sprint=True)
    ctrl._next_idx = 1
    m._sprint_controller = ctrl  # type: ignore[attr-defined]
    return m


def test_sprint_scenario_reaches_all_three_waypoints():
    """ScenarioLoop must NOT terminate after the first waypoint completes.
    The runner must reach all three waypoints before the trial ends."""
    defn = ScenarioDefinition(
        key="test_sprint3", label="test", description="",
        build=_build_multi_waypoint_sprint,
        on_tick=_sprint_on_tick,
    )
    loop = ScenarioLoop(definition=defn, max_trials=0, timeout_ticks=600)

    max_x_seen = -999.0
    for _ in range(600):
        player = loop.match.player_by_id("runner")
        max_x_seen = max(max_x_seen, player.position.x)
        trial_ended = loop.step()
        if trial_ended:
            break

    assert max_x_seen >= 9.0, (
        f"Runner never reached the third waypoint (x=10). "
        f"Max x seen: {max_x_seen:.2f}. "
        f"Trial ended prematurely — ScenarioLoop is terminating between waypoints."
    )


def test_sprint_scenario_first_trial_runs_long_enough():
    """The first trial must last at least 30 ticks. If it ends sooner the
    ScenarioLoop is terminating prematurely (e.g. between waypoints when
    current_order is transiently None)."""
    defn = ScenarioDefinition(
        key="test_sprint_transient", label="test", description="",
        build=_build_multi_waypoint_sprint,
        on_tick=_sprint_on_tick,
    )
    loop = ScenarioLoop(definition=defn, max_trials=1, timeout_ticks=600)

    ticks_until_end = 0
    for i in range(600):
        if loop.step():
            ticks_until_end = i + 1
            break

    assert ticks_until_end >= 30, (
        f"First trial ended after only {ticks_until_end} ticks — "
        "ScenarioLoop is terminating prematurely between waypoints."
    )


# ---------------------------------------------------------------------------
# Bug 3: 1v2 defender — GetPossessionOrder must close on carrier
# ---------------------------------------------------------------------------

def test_get_possession_closes_distance_to_carrier_each_tick():
    """A player with GetPossessionOrder must move closer to the carrier
    every tick until contact."""
    pitch = Pitch.standard()
    carrier = make_player("carrier", Team.LEFT, attr_value=0.5,
                          position=Vector3(0, 0, 0))
    # Carrier moves straight away from defender
    carrier.current_order = MoveOrder(target_position=Vector3(20, 0, 0), sprint=True)
    carrier.heading_rad = 0.0

    defender = make_player("defender", Team.RIGHT, attr_value=0.8,
                           position=Vector3(-8, 0, 0))
    defender.heading_rad = 0.0

    ball = Ball.at_rest(Vector3(0, 0, 0))
    ball.possessed_by = carrier.player_id

    match = Match(pitch=pitch, players=[carrier, defender], ball=ball,
                  rng_reduction=1.0, rng=random.Random(0))
    defender.current_order = GetPossessionOrder()

    dists = []
    for _ in range(30):
        dists.append(defender.position.xy().distance_to(carrier.position.xy()))
        match.step()
        if defender.current_order is None:
            break  # completed (tackle happened)

    # The overall trend must be closing: last recorded dist < first recorded dist
    assert dists[-1] < dists[0], (
        f"Defender with GetPossessionOrder did not close on carrier. "
        f"Distance went {dists[0]:.2f} -> {dists[-1]:.2f}"
    )


def test_get_possession_eventually_makes_contact_with_slow_carrier():
    """A stationary carrier: GetPossessionOrder must make contact and the
    order must complete (tackle attempted) within a reasonable number of ticks."""
    pitch = Pitch.standard()
    carrier = make_player("carrier", Team.LEFT, attr_value=0.2,
                          position=Vector3(0, 0, 0))
    defender = make_player("defender", Team.RIGHT, attr_value=0.9,
                           position=Vector3(-6, 0, 0))
    defender.heading_rad = 0.0

    ball = Ball.at_rest(Vector3(0, 0, 0))
    ball.possessed_by = carrier.player_id

    match = Match(pitch=pitch, players=[carrier, defender], ball=ball,
                  rng_reduction=1.0, rng=random.Random(0))
    defender.current_order = GetPossessionOrder()

    for _ in range(120):  # 4 sim-seconds — more than enough to cross 6 m
        match.step()
        if defender.current_order is None:
            break

    assert defender.current_order is None, (
        f"Defender never completed GetPossessionOrder against stationary carrier "
        f"after 4 s. Final dist: "
        f"{defender.position.xy().distance_to(carrier.position.xy()):.2f} m"
    )

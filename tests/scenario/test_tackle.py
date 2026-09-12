"""Scenario test: a player can tackle another player. rng_reduction=1.0 so
the outcome is deterministic based purely on attribute comparison.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Player, Team
from footballcoach.mathutils import Vector3
from footballcoach.engine.collision import are_touching
from footballcoach.engine.movement import MovementParams, SpeedMode, effective_top_speed
from footballcoach.orders import ChaseTackleOrder, GetPossessionOrder, OrderStatus
from footballcoach.rules_ai import Phase1RulesAI, StopWhenIdleAI
from tests.conftest import make_player


@dataclass
class _StraightSprintOrder:
    """Minimal order: sprint in a fixed direction forever, no braking and no
    opponent-avoidance repulsion (unlike MoveOrder, which steers around
    other players). Used to force a genuine head-on collision in tests
    that need one, rather than the normal AI behaviour of curving around
    an opponent to avoid contact."""
    direction: Vector3
    status: OrderStatus = OrderStatus.PENDING
    on_complete: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def execute(self, player: Player, match: Match, dt: float) -> bool:
        player.desired_direction = self.direction
        player.desired_speed_mode = SpeedMode.SPRINT
        return False


def test_tackle_wins_ball_from_carrier():
    pitch = Pitch.standard()
    tackler = make_player("tackler", Team.LEFT, position=Vector3(0, 0, 0), tackling=0.9)
    carrier = make_player("carrier", Team.RIGHT, position=Vector3(0.5, 0, 0), dribbling=0.1)
    carrier.ai = StopWhenIdleAI()  # stationary ball-holder; not under test

    ball = Ball.at_rest(Vector3(0.5, 0, 0))
    ball.possessed_by = carrier.player_id

    match = Match(pitch=pitch, players=[tackler, carrier], ball=ball, rng_reduction=1.0, rng=random.Random(0))

    tackler.current_order = ChaseTackleOrder(target_player_id=carrier.player_id)
    match.step()

    assert ball.possessed_by == tackler.player_id
    assert carrier.state.name == "INACTIVE_TACKLED"


def test_phase1_rules_ai_approach_fires_explicit_tackle():
    """Phase1RulesAI sprinting from 3 m must fire on_tackle (the BC-recordable
    callback) when it contacts the carrier — via tackle_armed + _check_armed_tackles,
    not the closing-speed-only auto-tackle path."""
    # Use training dt; at 30Hz the chaser ramps up over many ticks and the
    # first contact happens at sprint speed, which also satisfies auto-tackle.
    dt = 0.06
    pitch = Pitch.standard()
    chaser = make_player("chaser", Team.LEFT, position=Vector3(-3.0, 0.0, 0.0),
                         tackling=0.9, dribbling=0.1)
    carrier = make_player("carrier", Team.RIGHT, position=Vector3(0.0, 0.0, 0.0),
                          dribbling=0.1)
    carrier.ai = StopWhenIdleAI()  # stationary ball-holder; not under test

    ball = Ball.at_rest(carrier.position)
    ball.possessed_by = carrier.player_id

    match = Match(pitch=pitch, players=[chaser, carrier], ball=ball,
                  rng_reduction=1.0, rng=random.Random(42), dt_s=dt)

    chaser.ai = Phase1RulesAI()

    on_tackle_fired = []
    chaser.on_tackle = lambda p: on_tackle_fired.append(True)

    assert not are_touching(chaser, carrier), "Pre-condition: chaser must start out of contact range"

    for _ in range(200):
        match.step()
        if ball.possessed_by == chaser.player_id:
            break

    assert ball.possessed_by == chaser.player_id, "Chaser never won possession in 200 ticks"
    assert on_tackle_fired, (
        "Possession changed but on_tackle was never called — "
        "tackle went through the auto-tackle path instead of GetPossession (not recorded by BC)"
    )


def test_armed_tackle_fires_before_autotackle_on_sprint_into_range():
    """When a GetPossessionOrder player sprints from just outside touching range
    into contact, _check_armed_tackles must fire on_tackle before
    _check_head_on_tackles (the auto-tackle path) can resolve it.

    Physically, sprinting across the touching threshold always satisfies the
    auto-tackle closing-speed condition too.  The test verifies the armed path
    wins by asserting on_tackle fires — auto-tackle never calls on_tackle, so
    if it had resolved first, this assertion would fail."""
    # Use training dt (~17Hz) so a single tick moves ~0.46m — enough to cross
    # the touching threshold in one step (necessary to reproduce the skip-over bug).
    dt = 0.06
    pitch = Pitch.standard()
    carrier = make_player("carrier", Team.RIGHT, position=Vector3(0.0, 0.0, 0.0),
                          dribbling=0.1)
    carrier.ai = StopWhenIdleAI()  # stationary ball-holder; not under test
    chaser = make_player("chaser", Team.LEFT, tackling=0.9, dribbling=0.1)

    params = MovementParams.from_config()
    sprint_mps = effective_top_speed(params, 0.5, 1.0, has_ball=False)
    touching_threshold = chaser.radius_m + carrier.radius_m + 0.05
    # Place chaser half a tick's distance outside touching range at full sprint speed.
    start_dist = touching_threshold + sprint_mps * dt * 0.5
    chaser.position = Vector3(-start_dist, 0.0, 0.0)
    # Pre-set velocity so the first tick travels at full sprint (no ramp-up delay).
    chaser.velocity = Vector3(sprint_mps, 0.0, 0.0)

    ball = Ball.at_rest(carrier.position)
    ball.possessed_by = carrier.player_id

    match = Match(pitch=pitch, players=[chaser, carrier], ball=ball,
                  rng_reduction=1.0, rng=random.Random(42), dt_s=dt)

    chaser.current_order = GetPossessionOrder(sprint=True)

    on_tackle_fired = []
    chaser.on_tackle = lambda p: on_tackle_fired.append(True)

    assert not are_touching(chaser, carrier), "Pre-condition: must start outside touching range"

    match.step()

    assert on_tackle_fired, (
        "on_tackle not called — auto-tackle resolved the contact instead of the "
        "armed path (tackle_armed + _check_armed_tackles)"
    )
    assert ball.possessed_by == chaser.player_id, "Armed tackle fired but chaser didn't win"


def test_elite_dribbler_pushes_past_beaten_defender_fast_and_unslowed():
    """An elite dribbler (1.0) running full pelt, with a slight lateral
    deviation, into a hopeless tackler (0.0) should: resolve exactly one
    tackle attempt (win it), then push straight through the now-inactive
    defender almost immediately and at near-unchanged speed — not get
    stuck gliding against them.

    This exercises collision.py's rule that BOTH position push-apart and
    velocity damping are skipped for pairs where either player is inactive
    (see resolve_all_overlaps' docstring): before that fix, damping still
    applied to inactive pairs, so the dribbler would keep bleeding speed
    for as long as the two remained overlapping.

    No explicit tackle order is given to the defender — contact alone
    triggers the collision-based auto-tackle path (_check_head_on_tackles).
    """
    pitch = Pitch.standard()
    attacker = make_player("attacker", Team.LEFT, position=Vector3(-10.0, 0.15, 0.0), dribbling=1.0)
    defender = make_player("defender", Team.RIGHT, position=Vector3(0.0, 0.0, 0.0), tackling=0.0)
    defender.ai = StopWhenIdleAI()  # stationary; not under test

    ball = Ball.at_rest(attacker.position)
    ball.possessed_by = attacker.player_id

    match = Match(pitch=pitch, players=[attacker, defender], ball=ball,
                  rng_reduction=1.0, rng=random.Random(0))

    # Straight sprint at the defender: MoveOrder would steer around them via
    # repulsion (realistic AI behaviour, but it would avoid the collision
    # this test needs to force).
    attacker.current_order = _StraightSprintOrder(direction=Vector3(1.0, 0.0, 0.0))

    tackle_results: list[bool] = []

    def on_defender_auto_result(player: Player, tackler_won: bool, is_tackler: bool) -> None:
        if is_tackler:
            tackle_results.append(tackler_won)

    defender.on_auto_tackle_result = on_defender_auto_result

    min_clear_distance = attacker.radius_m + defender.radius_m
    contact_start_s: float | None = None
    speed_at_contact_start: float | None = None
    clear_s: float | None = None

    for _ in range(10 * 30):  # 10s of headroom at the default 30Hz tick rate
        match.step()

        if contact_start_s is None and are_touching(attacker, defender):
            contact_start_s = match.time_s
            speed_at_contact_start = attacker.speed_mps

        if contact_start_s is not None and clear_s is None:
            dist = attacker.position.xy().distance_to(defender.position.xy())
            if dist >= min_clear_distance:
                clear_s = match.time_s
                break

    assert len(tackle_results) == 1, f"expected exactly one tackle attempt, got {len(tackle_results)}"
    assert tackle_results[0] is False, "defender (0 tackling) should lose to attacker (1.0 dribbling)"
    assert ball.possessed_by == attacker.player_id

    assert contact_start_s is not None, "attacker never made contact with the defender"
    assert clear_s is not None, "attacker never cleared the defender within the simulated window"

    time_to_clear = clear_s - contact_start_s
    assert time_to_clear < 0.2, f"took {time_to_clear:.3f}s to get past the defender, expected < 0.2s"

    speed_change = abs(attacker.speed_mps - speed_at_contact_start)
    assert speed_change < 0.5, (
        f"attacker's speed changed by {speed_change:.3f} m/s while passing the defender, expected < 0.5 m/s"
    )

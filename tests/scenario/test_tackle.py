"""Scenario test: a player can tackle another player. rng_reduction=1.0 so
the outcome is deterministic based purely on attribute comparison.
"""
from __future__ import annotations

import random

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.mathutils import Vector3
from footballcoach.engine.collision import are_touching
from footballcoach.engine.movement import MovementParams, effective_top_speed
from footballcoach.orders import ChaseTackleOrder, GetPossessionOrder
from footballcoach.rules_ai import Phase1RulesAI
from tests.conftest import make_player


def test_tackle_wins_ball_from_carrier():
    pitch = Pitch.standard()
    tackler = make_player("tackler", Team.LEFT, position=Vector3(0, 0, 0), tackling=0.9)
    carrier = make_player("carrier", Team.RIGHT, position=Vector3(0.5, 0, 0), dribbling=0.1)

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

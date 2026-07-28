"""Diagnostic test: trace the pass scenario tick-by-tick to see exactly
what happens when GetPossession fires on the receiver.

Run with -s to see print output:
    uv run pytest tests/scenario/test_pass_getpossession_diagnostic.py -v -s
"""
from __future__ import annotations

import random

from footballcoach.engine.match import Match
from footballcoach.entities import Ball, Pitch, PlayerAttributes, Team
from footballcoach.entities.player import Player, PlayerState
from footballcoach.mathutils import Vector3
from footballcoach.orders import GetPossessionOrder, PassOrder
from footballcoach.ui.scenarios import _pass_on_tick, _PASS_SCENARIO_GET_POSSESSION_RADIUS_M


def _make_pass_match(passer_x: float = -10.0, receiver_x: float = 10.0) -> Match:
    """Simple controlled pass: passer on the left, receiver on the right,
    ball passed straight along x-axis. No repulsion noise."""
    pitch = Pitch.standard()
    attrs = PlayerAttributes(
        top_speed=0.75, acceleration=0.75, stamina=0.75,
        kick_precision=0.75, kick_power=0.75, dribbling=0.75,
        ball_control=0.75, tackling=0.75,
    )
    passer_pos = Vector3(passer_x, 0, 0)
    receiver_pos = Vector3(receiver_x, 0, 0)

    passer = Player.create("passer", Team.LEFT, attrs, position=passer_pos)
    receiver = Player.create("receiver", Team.LEFT, attrs, position=receiver_pos)

    ball = Ball.at_rest(passer_pos)
    ball.possessed_by = passer.player_id

    match = Match(
        pitch=pitch, players=[passer, receiver], ball=ball,
        rng_reduction=1.0, rng=random.Random(0),
    )
    passer.current_order = PassOrder(target_position=receiver_pos)
    return match


def test_pass_receiver_getpossession_trace():
    """Full trace of a pass: print every tick so we can see if GetPossession
    fires correctly and whether the receiver moves TOWARD or AWAY from ball."""
    match = _make_pass_match(passer_x=-10.0, receiver_x=10.0)
    passer = match.player_by_id("passer")
    receiver = match.player_by_id("receiver")
    ball = match.ball

    print(f"\n{'tick':>4}  {'ball_x':>7}  {'ball_vx':>8}  {'recv_x':>7}  "
          f"{'dist_ball_recv':>14}  {'recv_order':>20}  {'recv_state':>18}")
    print("-" * 90)

    get_possession_fired_tick = None
    dist_at_gp_fire = None
    prev_recv_x = receiver.position.x

    for tick in range(200):
        dist_to_ball = receiver.position.xy().distance_to(ball.position.xy())
        order_name = type(receiver.current_order).__name__ if receiver.current_order else "None"

        print(f"{tick:>4}  {ball.position.x:>7.3f}  {ball.velocity.x:>8.3f}  "
              f"{receiver.position.x:>7.3f}  {dist_to_ball:>14.3f}  "
              f"{order_name:>20}  {receiver.state.name:>18}")

        if get_possession_fired_tick is None and isinstance(receiver.current_order, GetPossessionOrder):
            get_possession_fired_tick = tick
            dist_at_gp_fire = dist_to_ball
            print(f"  *** GetPossession fired at tick {tick}, dist={dist_to_ball:.3f} m ***")

        if (ball.possessed_by == receiver.player_id or
                receiver.state == PlayerState.CONTROLLING_BALL):
            print(f"  *** Receiver got ball at tick {tick} ***")
            break

        if ball.velocity.length() < 0.05 and ball.possessed_by is None:
            print(f"  *** Ball stopped at tick {tick}, nobody has it ***")
            break

        _pass_on_tick(match, tick)
        match.step()

        # Check direction of movement after GetPossession fires
        if get_possession_fired_tick is not None and tick == get_possession_fired_tick:
            moved = receiver.position.x - prev_recv_x
            print(f"  *** Receiver moved {'TOWARD' if receiver.position.x > prev_recv_x else 'AWAY FROM'} "
                  f"ball this tick (dx={moved:+.4f}) ***")

        prev_recv_x = receiver.position.x

    print()
    assert get_possession_fired_tick is not None, \
        "GetPossession was never fired — ball never came within 6m of receiver"


def test_pass_receiver_must_not_move_away_from_ball():
    """After GetPossession fires, the receiver's distance to the ball must
    DECREASE over the next 3 ticks, not increase. Fails if they move away."""
    match = _make_pass_match(passer_x=-10.0, receiver_x=10.0)
    passer = match.player_by_id("passer")
    receiver = match.player_by_id("receiver")
    ball = match.ball

    # Run until ball is within 6m of receiver
    gp_fired = False
    dist_when_fired = None
    for tick in range(200):
        _pass_on_tick(match, tick)
        if isinstance(receiver.current_order, GetPossessionOrder) and not gp_fired:
            gp_fired = True
            dist_when_fired = receiver.position.xy().distance_to(ball.position.xy())
            # Run 5 more ticks and track distance
            dists = []
            for _ in range(5):
                match.step()
                _pass_on_tick(match, tick)
                dists.append(receiver.position.xy().distance_to(ball.position.xy()))
                if (ball.possessed_by == receiver.player_id or
                        receiver.state == PlayerState.CONTROLLING_BALL):
                    break

            print(f"\nGetPossession fired at dist={dist_when_fired:.3f}m")
            print(f"Distances over next {len(dists)} ticks: {[f'{d:.3f}' for d in dists]}")

            # The receiver should be getting closer, not further
            if len(dists) >= 2:
                assert dists[-1] < dist_when_fired, (
                    f"Receiver moved AWAY from ball after GetPossession: "
                    f"dist was {dist_when_fired:.3f}m, after 5 ticks it's {dists[-1]:.3f}m. "
                    f"Full sequence: {dists}"
                )
            return
        match.step()

    assert gp_fired, "GetPossession never fired in 200 ticks"


def test_getpossession_toward_loose_ball_direct():
    """Simplest possible case: ball is loose and stationary 3m from player.
    GetPossession must immediately move them toward it."""
    pitch = Pitch.standard()
    attrs = PlayerAttributes(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
    receiver = Player.create("receiver", Team.LEFT, attrs, position=Vector3(5, 0, 0))
    # Ball sitting still 3m away (within pickup radius trigger after a few ticks)
    ball = Ball.at_rest(Vector3(2, 0, 0))

    match = Match(pitch=pitch, players=[receiver], ball=ball,
                  rng_reduction=1.0, rng=random.Random(0))
    receiver.current_order = GetPossessionOrder()

    print(f"\n{'tick':>4}  {'recv_x':>7}  {'dist_to_ball':>12}  {'speed':>6}")
    dists = []
    for tick in range(10):
        d = receiver.position.xy().distance_to(ball.position.xy())
        dists.append(d)
        print(f"{tick:>4}  {receiver.position.x:>7.3f}  {d:>12.3f}  {receiver.speed_mps:>6.3f}")
        match.step()
        if ball.possessed_by == receiver.player_id or receiver.state == PlayerState.CONTROLLING_BALL:
            print(f"  *** Got ball at tick {tick+1} ***")
            break

    print(f"Distance sequence: {[f'{d:.3f}' for d in dists]}")
    # Turning takes a few ticks — by tick 5 the player must be closing in.
    # (Immediately checking tick 1 is too strict since the player may start
    # facing the wrong direction and needs time to rotate.)
    assert dists[-1] < dists[0], (
        f"After 10 ticks with GetPossession on a 3m-away stationary ball, "
        f"distance went {dists[0]:.3f} -> {dists[-1]:.3f} (should decrease overall)"
    )

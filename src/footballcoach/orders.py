"""Player orders: Move / Kick / Tackle. A player holds a current order and
follows it to completion (per the design spec: while paused, the user issues
orders which players then execute until done, before returning to whatever
they were doing before - e.g. an autonomous policy in later milestones).

This module defines the order data types and a small state machine helper.
The actual per-tick execution logic lives in engine/match.py, which reads
`player.current_order` and dispatches to movement/kicking/tackling as
appropriate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

from footballcoach.mathutils import Vector3


class OrderStatus(Enum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETE = auto()


@dataclass
class MoveOrder:
    target_position: Vector3
    sprint: bool = True
    arrival_tolerance_m: float = 0.3
    # Controls how fast the player is moving when the order completes.
    # None  -> resolved to jog speed at execution time (smooth, natural stop).
    # 0.0   -> full standstill (the order does not complete until speed < 0.05
    #          m/s AND within the (slightly widened) distance tolerance).
    # >0    -> any explicit speed target in m/s.
    max_speed_on_arrival_mps: float | None = None
    # If the player overshoots (crosses the target point), this countdown
    # starts. When it reaches 0 the order completes (player brakes to stop).
    # None = no overshoot detected yet.
    overshoot_timeout_s: float = 0.5
    # Set to True the first tick the player is within arrival_tolerance_m.
    # Once True, if the player drifts back outside that radius the overshoot
    # countdown starts.
    reached_target: bool = False
    _overshoot_timer_s: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    on_complete: Callable[[], None] | None = field(default=None, repr=False, compare=False)


@dataclass
class KickOrder:
    aim_point: Vector3  # absolute world position the kicker intends to hit
    power_fraction: float  # in [0, 1]; or >1 when compensate_for_run=False and caller wants old raw behaviour
    spin: Vector3
    compensate_for_run: bool = True  # if True, match.py pre-divides by run_mult so the ball
                                     # leaves at the intended speed regardless of run direction
    status: OrderStatus = OrderStatus.PENDING
    on_complete: Callable[[], None] | None = field(default=None, repr=False, compare=False)


@dataclass
class PassOrder:
    """A grounded pass to a target position. The engine auto-computes pace
    from distance (if `power_fraction` is left as None). Error model is the
    same unified formula as KickOrder - lower power naturally produces a
    more accurate kick.

    If ``target_player_id`` is set, the pass is "led": the match engine
    estimates where that player will be when the ball arrives (based on
    their current velocity) and aims at the predicted position rather than
    their current position. ``target_position`` must still be set to the
    player's current position (used as a fallback and for distance
    estimation); ``actions.pass_to`` handles this automatically when a
    Player is passed instead of a Vector3.
    """
    target_position: Vector3
    power_fraction: float | None = None  # None = auto-computed from distance
    target_player_id: str | None = None  # set for leading passes
    status: OrderStatus = OrderStatus.PENDING
    on_complete: Callable[[], None] | None = field(default=None, repr=False, compare=False)


@dataclass
class ChaseTackleOrder:
    """"Tackle" high-level action: run straight at an opposing player and,
    once in range, attempt a tackle. Unlike `TackleOrder` (which only acts
    if the two players are already touching, and always completes in a
    single tick regardless of outcome), this order persists tick-to-tick,
    chasing the target's current position in a straight line until contact
    is made, then resolves exactly one tackle attempt before completing.
    """
    target_player_id: str
    status: OrderStatus = OrderStatus.PENDING
    on_complete: Callable[[], None] | None = field(default=None, repr=False, compare=False)


@dataclass
class SaveOrder:
    """Goalkeeper-only "Save" action: continuously predicts where an
    in-flight ball will cross this keeper's own goal line and moves there
    (see engine/goalkeeping.py). Deliberately does not auto-complete like
    the other orders - a real goalkeeper is always "on duty", holding a
    sensible default position (goal centre) when no shot is incoming and
    reacting the instant one is. Issue it once; it stays in effect until
    replaced by another order.
    """
    status: OrderStatus = OrderStatus.PENDING
    on_complete: Callable[[], None] | None = field(default=None, repr=False, compare=False)


@dataclass
class StopOrder:
    """Decelerates the player to a complete stop using their normal braking
    capability, then completes. Useful for explicitly halting a player who is
    mid-sprint without snapping their velocity to zero instantly."""
    status: OrderStatus = OrderStatus.PENDING
    on_complete: Callable[[], None] | None = field(default=None, repr=False, compare=False)


@dataclass
class GetPossessionOrder:
    """Runs straight at the ball and acquires it.

    - If the ball is loose, the player chases it at the given speed (sprint
      by default); pickup happens automatically via the normal control-time
      model once they're close enough.
    - If another player has the ball, the player chases that carrier and
      attempts one tackle on contact (exactly like ChaseTackleOrder),
      then completes regardless of the tackle outcome.
    - Completes immediately if this player already possesses the ball.
    """
    sprint: bool = True  # True = sprint to ball, False = jog
    status: OrderStatus = OrderStatus.PENDING
    on_complete: Callable[[], None] | None = field(default=None, repr=False, compare=False)


@dataclass
class MarkOrder:
    """Mark a specific opposition player: continuously position the marker
    between that player and the ball, and switch to GetPossession-style
    chase / tackle logic when:

    - The target gains ball possession (or is mid first-touch control), OR
    - The ball comes within ``mark_intercept_radius_m`` of the marker.

    Never auto-completes — holds indefinitely until replaced by another order
    (same persistent-duty model as ``SaveOrder``). The standoff distance and
    intercept radius are configured in ``physics.json["marking"]``.
    """
    target_player_id: str
    status: OrderStatus = OrderStatus.PENDING
    on_complete: Callable[[], None] | None = field(default=None, repr=False, compare=False)


@dataclass
class ShootOrder:
    """Shoot at goal by aiming at a specific 3-D point (e.g. a corner of the
    goal frame).  The player must have possession; if they do not the order
    completes immediately as a no-op.

    Mechanically identical to KickOrder - both call ``kick_ball`` with the
    same error model.  The semantic distinction is:

    - KickOrder: freeform kick; direction and power come from the UI drag
      gesture or explicit scenario setup (used by balance-test fixtures,
      penalty scenarios, etc.).
    - ShootOrder: deliberate shot on goal; the player (or the user via the
      ``K`` key in the UI) picks a target *inside the goal frame* and the
      engine fires at that point at the requested power.

    ``chance_of_pausing`` (default 0.8): if any opposition player lies on the
    shooter's line to the aim point, the engine does a random check with this
    probability. On success the shot is replaced by a 2 m MoveOrder in the
    aim direction (processed identically to a normal MoveOrder, including
    repulsion), giving the shooter a chance to clear the blocker before
    shooting again on the next cycle.  Set to 0.0 to disable the check.
    """
    aim_point: Vector3          # absolute world position to aim at
    power_fraction: float       # in [0, 1]; or >1 when compensate_for_run=False
    compensate_for_run: bool = True  # same semantics as KickOrder.compensate_for_run
    chance_of_pausing: float = 0.8  # probability of pausing when a blocker is detected
    status: OrderStatus = OrderStatus.PENDING
    on_complete: Callable[[], None] | None = field(default=None, repr=False, compare=False)


Order = MoveOrder | KickOrder | ShootOrder | PassOrder | ChaseTackleOrder | SaveOrder | StopOrder | GetPossessionOrder | MarkOrder

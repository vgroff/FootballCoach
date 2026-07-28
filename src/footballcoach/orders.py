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

from dataclasses import dataclass
from enum import Enum, auto

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
    status: OrderStatus = OrderStatus.PENDING


@dataclass
class KickOrder:
    aim_point: Vector3  # absolute world position the kicker intends to hit
    power_fraction: float  # in [0, 1]
    spin: Vector3
    status: OrderStatus = OrderStatus.PENDING


@dataclass
class TackleOrder:
    target_player_id: str
    status: OrderStatus = OrderStatus.PENDING


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


@dataclass
class StopOrder:
    """Decelerates the player to a complete stop using their normal braking
    capability, then completes. Useful for explicitly halting a player who is
    mid-sprint without snapping their velocity to zero instantly."""
    status: OrderStatus = OrderStatus.PENDING


@dataclass
class GetPossessionOrder:
    """Runs straight at the ball and acquires it.

    - If the ball is loose, the player sprints to it; pickup happens
      automatically via the normal control-time model once they're close
      enough.
    - If another player has the ball, the player chases that carrier and
      attempts one tackle on contact (exactly like the old ChaseTackleOrder),
      then completes regardless of the tackle outcome.
    - Completes immediately if this player already possesses the ball.
    """
    status: OrderStatus = OrderStatus.PENDING


Order = MoveOrder | KickOrder | TackleOrder | PassOrder | ChaseTackleOrder | SaveOrder | StopOrder | GetPossessionOrder

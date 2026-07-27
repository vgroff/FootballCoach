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
    """A grounded pass to a target position. Distinct from KickOrder: the
    engine computes an appropriate pace for the distance automatically (if
    `power_fraction` is left as None) and applies a dedicated passing
    accuracy model (see engine/kicking.py's `pass_ball`), which is more
    forgiving than the general shot-error model KickOrder uses - passing
    along the ground to a spot is a different technical skill than curling
    a shot into a corner, even though both are driven by `kick_precision`.
    """
    target_position: Vector3
    power_fraction: float | None = None  # None = auto-computed from distance
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


Order = MoveOrder | KickOrder | TackleOrder | PassOrder | ChaseTackleOrder | SaveOrder

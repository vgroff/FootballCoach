"""The player entity: a cylinder in the sim, with position, velocity, heading,
stamina, attributes, and current order/state machine hooks."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from footballcoach.config import load_physics_config
from footballcoach.entities.attributes import PlayerAttributes
from footballcoach.mathutils import Vector3


class Team(Enum):
    LEFT = auto()
    RIGHT = auto()


class PlayerState(Enum):
    """High-level state machine for a player, used to gate actions."""
    ACTIVE = auto()          # normal play
    INACTIVE_TACKLED = auto()  # just been tackled, reduced capability
    CONTROLLING_BALL = auto()  # mid first-touch control-time delay


@dataclass
class Player:
    player_id: str
    team: Team
    attributes: PlayerAttributes
    position: Vector3 = field(default_factory=Vector3.zero)
    velocity: Vector3 = field(default_factory=Vector3.zero)
    heading_rad: float = 0.0
    is_goalkeeper: bool = False

    stamina: float = 1.0  # current stamina fraction, drains/regens over time
    state: PlayerState = PlayerState.ACTIVE
    state_timer_s: float = 0.0  # remaining time in current transient state

    # Order handling (move/kick/tackle) is layered on via orders.py; this
    # holds a reference to the current order object (typed loosely here to
    # avoid a circular import - see orders.py for the concrete types).
    current_order: object | None = None

    radius_m: float = 0.3
    height_m: float = 1.8

    @staticmethod
    def create(
        player_id: str,
        team: Team,
        attributes: PlayerAttributes,
        position: Vector3 | None = None,
        is_goalkeeper: bool = False,
    ) -> "Player":
        cfg = load_physics_config()["player"]
        return Player(
            player_id=player_id,
            team=team,
            attributes=attributes,
            position=position or Vector3.zero(),
            is_goalkeeper=is_goalkeeper,
            radius_m=cfg["radius_m"],
            height_m=cfg["height_m"],
        )

    @property
    def speed_mps(self) -> float:
        return self.velocity.length_xy()

    def is_available_to_tackle(self) -> bool:
        return self.state != PlayerState.INACTIVE_TACKLED

    @property
    def is_inactive(self) -> bool:
        """True while the player is temporarily out of active play (just
        tackled/dispossessed, or having just failed a tackle attempt - see
        engine/tackling.py). Inactive players don't participate in
        player-player collision (others can run straight through them, see
        engine/collision.py's `resolve_player_overlap`), though they can
        still block a shot struck from outside their cylinder that flies
        through it (see `engine/collision.py`'s
        `resolve_ball_block_by_inactive_players`)."""
        return self.state == PlayerState.INACTIVE_TACKLED

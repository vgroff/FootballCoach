"""The player entity: a cylinder in the sim, with position, velocity, heading,
stamina, attributes, and current order/state machine hooks."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from footballcoach.config import load_physics_config
from footballcoach.entities.attributes import PlayerAttributes
from footballcoach.mathutils import Vector3

if TYPE_CHECKING:
    from footballcoach.engine.match import Match


class PlayerAI:
    """Base class for all player AI controllers.

    Subclass and override ``act(player, match, trial_tick)`` to implement
    per-player decision logic.  ``Match.step()`` calls ``player.ai.act(...)``
    once per physics tick for every player that has an AI assigned.

    The default implementation is a no-op (stationary / order-driven player).
    """

    def act(self, player: "Player", match: "Match", trial_tick: int) -> None:  # noqa: ARG002
        pass


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

    # Optional AI controller. If set, Match.step() calls ai.act(player, match, tick)
    # once per physics tick before processing orders.  None = purely order-driven.
    ai: PlayerAI | None = field(default=None, repr=False)

    # Optional event callbacks — called by the engine at the exact moment the
    # action executes (not when the order is set).  Signature: (player) -> None.
    # Useful for recording, logging, UI effects, stats, etc.
    on_kick: object | None = field(default=None, repr=False)     # fired when a kick/shoot/pass lands
    on_tackle: object | None = field(default=None, repr=False)   # fired when a tackle attempt executes
    on_possession_gained: object | None = field(default=None, repr=False)  # fired by match._set_possession() when this player gains the ball

    # Display hint: set by the engine when an action fires (kick, tackle,
    # first-touch control, GK save).  The UI layer polls this each frame,
    # records the icon with a wall-clock expiry, then clears it.  Engine
    # logic must not depend on this field.
    action_icon: str | None = field(default=None, repr=False, compare=False)

    # Movement intent — set by execute() / AI each tick, consumed by match._apply_movement().
    # Orders/AI MUST NOT call step_player_towards directly; they set these two fields and the
    # engine applies them via step_player_towards.  desired_speed_mode=None means no movement.
    desired_direction: Vector3 = field(default_factory=Vector3.zero, repr=False, compare=False)
    desired_speed_mode: object | None = field(default=None, repr=False, compare=False)  # SpeedMode at runtime

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

    # ------------------------------------------------------------------
    # Atomic action methods — the only correct way to assign Orders.
    # PlayerAI subclasses, to_orders.py, and tests all call these.
    # Never set player.current_order to an Order dataclass directly
    # from outside this class.
    # ------------------------------------------------------------------

    def kick(self, aim_point: Vector3, power_fraction: float, spin: Vector3) -> None:
        """Issue a KickOrder. Used by the rules-based AI and human input only."""
        from footballcoach.orders import KickOrder
        self.current_order = KickOrder(aim_point=aim_point, power_fraction=power_fraction, spin=spin)

    def kick_direct(self, match: "Match", aim_point: Vector3, power_fraction: float, spin: Vector3) -> None:
        """Execute a kick immediately — THE NEURAL NETWORK CALLS THIS, NO ORDER IS ISSUED.

        KickOrder.execute() also delegates here so all kick physics live in one place.
        Only has effect if this player currently has possession.
        """
        from footballcoach.engine.kicking import kick_ball, compensate_power_for_run_mult, running_power_multiplier
        from footballcoach.engine.movement import effective_top_speed
        if match.ball.possessed_by != self.player_id:
            return
        top_speed = effective_top_speed(
            match.movement_params, self.attributes.top_speed, self.stamina,
            has_ball=True, ball_control_attr=self.attributes.ball_control,
        )
        run_mult = running_power_multiplier(
            match.kicking_params.running_power_coefficient, self.velocity,
            aim_point - self.position, top_speed,
        )
        kick_ball(
            match.ball,
            self.position,
            aim_point,
            compensate_power_for_run_mult(power_fraction, run_mult),
            self.attributes.kick_precision,
            self.attributes.kick_power,
            spin,
            match.rng_reduction,
            match.rng,
            match.kicking_params,
            kicker_velocity=self.velocity,
            kicker_top_speed_mps=top_speed,
        )
        match._start_release_grace(self.player_id)
        match._log_debug(f"{self.player_id} kicked (direct)  power={power_fraction:.2f}")
        if self.on_kick is not None:
            self.on_kick(self)

    def tackle_direct(self, match: "Match", target_player_id: str) -> bool:
        """Attempt an immediate tackle if in contact range — THE NEURAL NETWORK CALLS THIS, NO ORDER IS ISSUED.

        Returns True if contact was made (tackle resolved), False if out of range.
        """
        from footballcoach.engine.collision import are_touching
        from footballcoach.engine.tackling import apply_tackle_result, attempt_tackle, tackle_angle_modifier
        try:
            target = match.player_by_id(target_player_id)
        except KeyError:
            return False
        if not are_touching(self, target):
            return False
        if not target.is_available_to_tackle():
            return True
        if self.on_tackle is not None:
            self.on_tackle(self)
        if match._gk_immune_from_tackle(target):
            match._apply_gk_immune_penalty(self)
            match._log_info(f"{self.player_id} direct-tackle on {target.player_id} auto-failed [GK immune]")
        else:
            result = attempt_tackle(
                self.attributes.tackling,
                match._effective_dribbling(target),
                match.rng_reduction,
                match.rng,
                match.tackling_params,
                is_goalkeeper_tackle=self.is_goalkeeper,
                angle_modifier=tackle_angle_modifier(
                    target.heading_rad, target.position, self.position, match.tackling_params
                ),
                gk_outside_box=match._gk_outside_own_box(self),
            )
            match._log_tackle_result(self.player_id, target.player_id, result)
            if result.tackler_won and match._target_has_or_controls_ball(target):
                match._set_possession(self.player_id)
            apply_tackle_result(result, self, target, match.tackling_params)
        return True

    def kick_direct(self, match: "Match", aim_point: Vector3, power_fraction: float, spin: Vector3) -> None:
        """Execute kick physics immediately WITHOUT issuing a KickOrder.

        THE NEURAL NETWORK CALLS THIS DIRECTLY — no Order is created.
        KickOrder.execute() also delegates here so all kick logic lives in one place.
        Only has effect if this player currently has possession.
        """
        from footballcoach.engine.kicking import kick_ball, compensate_power_for_run_mult, running_power_multiplier
        from footballcoach.engine.movement import effective_top_speed
        if match.ball.possessed_by != self.player_id:
            return
        top_speed = effective_top_speed(
            match.movement_params, self.attributes.top_speed, self.stamina,
            has_ball=True, ball_control_attr=self.attributes.ball_control,
        )
        run_mult = running_power_multiplier(
            match.kicking_params.running_power_coefficient, self.velocity,
            aim_point - self.position, top_speed,
        )
        kick_ball(
            match.ball,
            self.position,
            aim_point,
            compensate_power_for_run_mult(power_fraction, run_mult),
            self.attributes.kick_precision,
            self.attributes.kick_power,
            spin,
            match.rng_reduction,
            match.rng,
            match.kicking_params,
            kicker_velocity=self.velocity,
            kicker_top_speed_mps=top_speed,
        )
        match._start_release_grace(self.player_id)
        match._log_debug(f"{self.player_id} kicked (direct)  power={power_fraction:.2f}")
        if self.on_kick is not None:
            self.on_kick(self)

    def tackle_direct(self, match: "Match", target_player_id: str) -> bool:
        """Attempt an immediate tackle against target if in contact range.

        THE NEURAL NETWORK CALLS THIS DIRECTLY — no Order is created.
        Returns True if contact was made and a tackle attempt was resolved,
        False if out of range (network should keep moving toward target).
        """
        from footballcoach.engine.collision import are_touching
        from footballcoach.engine.tackling import apply_tackle_result, attempt_tackle, tackle_angle_modifier
        try:
            target = match.player_by_id(target_player_id)
        except KeyError:
            return False
        if not are_touching(self, target):
            return False
        if not target.is_available_to_tackle():
            return True
        if self.on_tackle is not None:
            self.on_tackle(self)
        if match._gk_immune_from_tackle(target):
            match._apply_gk_immune_penalty(self)
            match._log_info(f"{self.player_id} direct-tackle on {target.player_id} auto-failed [GK in own box]")
        else:
            result = attempt_tackle(
                self.attributes.tackling,
                match._effective_dribbling(target),
                match.rng_reduction,
                match.rng,
                match.tackling_params,
                is_goalkeeper_tackle=self.is_goalkeeper,
                angle_modifier=tackle_angle_modifier(
                    target.heading_rad, target.position, self.position, match.tackling_params
                ),
                gk_outside_box=match._gk_outside_own_box(self),
            )
            match._log_tackle_result(self.player_id, target.player_id, result)
            if result.tackler_won and match._target_has_or_controls_ball(target):
                match._set_possession(self.player_id)
            apply_tackle_result(result, self, target, match.tackling_params)
        return True

    def pass_ball(
        self,
        target_position: Vector3,
        target_player_id: str | None = None,
        power_fraction: float | None = None,
    ) -> None:
        """Issue a PassOrder toward a position or a specific teammate."""
        from footballcoach.orders import PassOrder
        self.current_order = PassOrder(
            target_position=target_position,
            power_fraction=power_fraction,
            target_player_id=target_player_id,
        )

    def get_possession(self) -> None:
        """Chase the ball / dispossess the carrier."""
        from footballcoach.orders import GetPossessionOrder
        self.current_order = GetPossessionOrder()

    def tackle_player(self, target_player_id: str) -> None:
        """Chase and tackle a specific opposing player."""
        from footballcoach.orders import ChaseTackleOrder
        self.current_order = ChaseTackleOrder(target_player_id=target_player_id)

    def mark_player(self, target_player_id: str) -> None:
        """Mark a specific opposing player."""
        from footballcoach.orders import MarkOrder
        self.current_order = MarkOrder(target_player_id=target_player_id)

    def stop(self) -> None:
        """Decelerate to a standstill."""
        from footballcoach.orders import StopOrder
        self.current_order = StopOrder()

    def save_goal(self) -> None:
        """Goalkeeper only: track incoming shot and move to intercept."""
        from footballcoach.orders import SaveOrder
        self.current_order = SaveOrder()

    def move_to(
        self,
        target_position: Vector3,
        sprint: bool = True,
        max_speed_on_arrival_mps: float | None = None,
    ) -> None:
        """Move toward a target position.

        Used by the rules-based AI and BC label generation.
        The neural execution network drives movement via move_direction
        (a far-target MoveOrder constructed in to_orders.py), not by
        calling this method with a strategic region centre.
        """
        from footballcoach.orders import MoveOrder
        self.current_order = MoveOrder(
            target_position=target_position,
            sprint=sprint,
            max_speed_on_arrival_mps=max_speed_on_arrival_mps,
        )

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

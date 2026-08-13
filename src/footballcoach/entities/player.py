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
    on_tackle: object | None = field(default=None, repr=False)   # fired when a tackle attempt executes (before the skill roll)
    # Fired right after the skill roll resolves, on BOTH the tackler and the
    # tacklee, with the boolean tackler_won result AND whether THIS player was
    # the tackler (was_tackler) or the tacklee for this specific attempt.
    # Signature: (player, tackler_won, was_tackler) -> None. was_tackler lets
    # callers pair this with on_tackle (which only fires on the tackler, i.e.
    # counts ATTEMPTS initiated) without double-counting a tacklee's outcome
    # as if they had attempted a tackle themselves.
    # Distinct from on_tackle (which fires unconditionally on attempt, before
    # the outcome is known) — lets callers (e.g. BC recording) tally success/fail.
    on_tackle_result: object | None = field(default=None, repr=False)
    # Analogous to on_tackle_result but for the AUTO-TACKLE (collision)
    # fallback path only (_check_head_on_tackles) -- fires on both the
    # tackler and tacklee with the same (player, tackler_won, was_tackler)
    # signature. on_tackle/on_tackle_result never fire for auto-tackles (see
    # test_tackle.py's auto-tackle-vs-armed-path tests), but possession still
    # transfers exactly the same way, so callers who want to also see/count
    # THOSE events (e.g. BC recording stats) use this separate callback
    # instead of conflating them with intentional/armed tackle attempts.
    on_auto_tackle_result: object | None = field(default=None, repr=False)
    on_possession_gained: object | None = field(default=None, repr=False)  # fired by match._set_possession() when this player gains the ball

    # Unconditional per-tick kick flag — set True inside kick_direct() every
    # time it actually executes kick physics, regardless of on_kick being set
    # and regardless of which Order (or no Order, e.g. MoveOrder's push-kick)
    # triggered it. Match._process_orders() resets this to False for every
    # player at the start of each tick, so any code that runs after order
    # processing (e.g. BC label derivation) can check "did this player kick
    # THIS tick" without caring about order-type bookkeeping.
    kicked_this_tick: bool = field(default=False, repr=False, compare=False)
    # Set by GetPossessionOrder / neural AI; engine fires on_tackle on next contact.
    tackle_armed: bool = field(default=False, repr=False, compare=False)
    # Set by rules AI / neural AI during approach; engine fires kick_direct on first-touch.
    # Analogous to tackle_armed. Reset each tick in _process_orders.
    kick_armed: bool = field(default=False, repr=False, compare=False)
    kick_armed_aim_point: "Vector3 | None" = field(default=None, repr=False, compare=False)
    kick_armed_power_fraction: float = field(default=0.85, repr=False, compare=False)
    kick_armed_spin: "Vector3 | None" = field(default=None, repr=False, compare=False)
    # Set by _update_loose_ball_pickup when CONTROLLING_BALL begins; read by kick_direct/
    # kick_with_direction to inflate first-touch error automatically. 0.0 when not controlling.
    firsttime_difficulty: float = field(default=0.0, repr=False, compare=False)

    # Unconditional per-tick kick output capture — set inside kick_direct() every
    # time it actually executes kick physics, regardless of on_kick being set and
    # regardless of which Order (or no Order, e.g. MoveOrder's push-kick)
    # triggered it. Mirrors kicked_this_tick's rationale (see its docstring) but
    # captures the actual kick VECTOR, not just the boolean fact of kicking, so
    # BC label derivation can supervise kick_direction/kick_power/kick_spin for
    # ANY AI that kicks (rules-based, human, future neural variants) with no
    # per-AI wiring. All three are None/unset when the player did not kick this
    # tick (reset alongside kicked_this_tick in Match._process_orders()).
    last_kick_direction: Vector3 | None = field(default=None, repr=False, compare=False)  # 3D unit vector of actual ball velocity
    last_kick_power_fraction: float | None = field(default=None, repr=False, compare=False)
    last_kick_spin: Vector3 | None = field(default=None, repr=False, compare=False)

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
    # Rules-based PlayerAI subclasses, HybridPlayerAI's order-override
    # channel, and tests all call these. The neural network's execution
    # path (ai/action/apply_nn_action.py) does NOT call these -- it sets
    # desired_direction/desired_speed_mode/tackle_armed directly and never
    # issues an Order at all.
    # Never set player.current_order to an Order dataclass directly
    # from outside this class.
    # ------------------------------------------------------------------

    def kick(self, aim_point: Vector3, power_fraction: float, spin: Vector3) -> None:
        """Issue a KickOrder. Used by the rules-based AI and human input only."""
        from footballcoach.orders import KickOrder
        self.current_order = KickOrder(aim_point=aim_point, power_fraction=power_fraction, spin=spin)

    def kick_direct(self, match: "Match", aim_point: Vector3, power_fraction: float, spin: Vector3, compensate_for_run: bool = True) -> None:
        """Execute kick physics immediately WITHOUT issuing a KickOrder.

        THE NEURAL NETWORK CALLS THIS DIRECTLY — no Order is created.
        KickOrder.execute() also delegates here so all kick logic lives in one place.
        Only has effect if this player currently has possession.

        compensate_for_run=True (default, used by neural net): pre-divides
        power_fraction by run_mult so the ball leaves at the intended speed
        regardless of running direction.
        compensate_for_run=False: power_fraction is used raw, so sprinting in
        line with the kick adds the full running boost to ball speed.
        """
        from footballcoach.engine.kicking import kick_ball, compensate_power_for_run_mult, firsttime_difficulty_multiplier, running_power_multiplier
        from footballcoach.engine.movement import effective_top_speed
        if match.ball.possessed_by != self.player_id:
            return
        is_first_touch = self.state == PlayerState.CONTROLLING_BALL
        diff_mult = (
            firsttime_difficulty_multiplier(match.kicking_params, self.attributes.kick_precision, self.firsttime_difficulty)
            if is_first_touch else 1.0
        )
        top_speed = effective_top_speed(
            match.movement_params, self.attributes.top_speed, self.stamina,
            has_ball=True, ball_control_attr=self.attributes.ball_control,
        )
        run_mult = running_power_multiplier(
            match.kicking_params.running_power_coefficient, self.velocity,
            aim_point - self.position, top_speed,
        )
        adjusted_power = (
            compensate_power_for_run_mult(power_fraction, run_mult)
            if compensate_for_run
            else power_fraction
        )
        kick_ball(
            match.ball,
            self.position,
            aim_point,
            adjusted_power,
            self.attributes.kick_precision,
            self.attributes.kick_power,
            spin,
            match.rng_reduction,
            match.rng,
            match.kicking_params,
            difficulty_multiplier=diff_mult,
            kicker_velocity=self.velocity,
            kicker_top_speed_mps=top_speed,
        )
        if is_first_touch:
            self.state = PlayerState.ACTIVE
            self.state_timer_s = 0.0
        match._log_debug(f"{self.player_id} kicked (direct)  power={power_fraction:.2f}")
        self.kicked_this_tick = True
        _vel = match.ball.velocity
        _vel_len = _vel.length()
        self.last_kick_direction = (_vel * (1.0 / _vel_len)) if _vel_len > 1e-6 else None
        self.last_kick_power_fraction = float(adjusted_power)  # what was actually passed to kick_ball
        self.last_kick_spin = spin
        if self.on_kick is not None:
            self.on_kick(self)

    def kick_with_direction(self, match: "Match", direction_3d: "Vector3", power_fraction: float, spin: "Vector3") -> None:
        """Execute a kick with an explicit 3D unit direction vector (no ballistic solve). Neural network only."""
        from footballcoach.engine.kicking import kick_ball_from_direction, firsttime_difficulty_multiplier
        from footballcoach.engine.movement import effective_top_speed
        if match.ball.possessed_by != self.player_id:
            return
        is_first_touch = self.state == PlayerState.CONTROLLING_BALL
        diff_mult = (
            firsttime_difficulty_multiplier(match.kicking_params, self.attributes.kick_precision, self.firsttime_difficulty)
            if is_first_touch else 1.0
        )
        top_speed = effective_top_speed(
            match.movement_params, self.attributes.top_speed, self.stamina,
            has_ball=True, ball_control_attr=self.attributes.ball_control,
        )
        kick_ball_from_direction(
            match.ball,
            self.position,
            direction_3d,
            power_fraction,
            self.attributes.kick_precision,
            self.attributes.kick_power,
            spin,
            match.rng_reduction,
            match.rng,
            match.kicking_params,
            difficulty_multiplier=diff_mult,
            kicker_velocity=self.velocity,
            kicker_top_speed_mps=top_speed,
        )
        if is_first_touch:
            self.state = PlayerState.ACTIVE
            self.state_timer_s = 0.0
        match._log_debug(f"{self.player_id} kicked (direct 3D)  power={power_fraction:.2f}")
        self.kicked_this_tick = True
        _vel = match.ball.velocity
        _vel_len = _vel.length()
        self.last_kick_direction = (_vel * (1.0 / _vel_len)) if _vel_len > 1e-6 else None
        self.last_kick_power_fraction = float(power_fraction)
        self.last_kick_spin = spin
        if self.on_kick is not None:
            self.on_kick(self)

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
        The neural execution network drives movement via move_direction,
        applied by setting desired_direction/desired_speed_mode directly in
        ai/action/apply_nn_action.py -- no MoveOrder or call to this method
        is involved.
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

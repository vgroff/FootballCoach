"""Player orders: Move / Kick / Tackle. A player holds a current order and
follows it to completion (per the design spec: while paused, the user issues
orders which players then execute until done, before returning to whatever
they were doing before - e.g. an autonomous policy in later milestones).

Each order exposes an ``execute(player, match, dt) -> bool`` method.  When
``execute`` returns ``True`` the order is complete; ``_process_orders`` in
match.py calls ``_complete_order`` and clears ``player.current_order``.

**READ-ONLY contract for ``match`` inside execute():**
``match`` is passed for READ access only — positions, the ball, params,
helper queries like ``match.player_by_id()``, ``match.ball_carrier()``.
NEVER write ``match.ball.*`` directly or mutate any player's velocity/state
from inside execute().  All physics side-effects (velocity scaling from
tackles, stamina drain, ball possession) must go through the engine's own
functions: ``apply_tackle_result()``, ``match._set_possession()``, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable

from footballcoach.mathutils import Vector3

if TYPE_CHECKING:
    from footballcoach.engine.match import Match
    from footballcoach.entities.player import Player


class OrderStatus(Enum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETE = auto()


# ---------------------------------------------------------------------------
# Shared movement-intent helper
# ---------------------------------------------------------------------------

# Brake-to-turn / close-proximity constants (all orders share these values).
_BRAKE_THRESH_RAD: float = 1.57    # ~90° heading change → brake first
_BRAKE_MIN_SPEED: float = 2.0      # don't bother below this speed
_CLOSE_PROX_M: float = 5.0         # close-proximity lateral-overshoot guard
_CLOSE_PROX_COS: float = 0.01      # cos-sim threshold for lateral overshoot


def _compute_movement_intent(
    player: "Player",
    target_direction: "Vector3",
    match: "Match",
    *,
    sprint: bool = True,
    arrival_dist: float | None = None,
    arrival_speed: float | None = None,
    use_repulsion: bool = True,
    use_brake_to_turn: bool = True,
) -> "tuple[Vector3, object]":
    """Compute ``(adjusted_direction, speed_mode)`` for a movement intent tick.

    All movement orders call this rather than duplicating braking / turning /
    repulsion logic.

    Parameters
    ----------
    target_direction:
        Raw direction vector toward the destination (un-normalised; zero = stop).
    arrival_dist:
        Current distance to target in metres.  When provided, the braking
        curve (``braking_speed_mode``) and close-proximity checks are applied.
        ``None`` means no braking — sprint/jog the whole way (e.g. chasing).
    arrival_speed:
        Desired speed on arrival (m/s).  ``None`` → resolved to jog speed.
    use_repulsion:
        Apply player-repulsion steering.  ``False`` for SaveOrder, chase paths.
    use_brake_to_turn:
        Apply the brake-to-turn heuristic (decelerate when heading change >90°).
    """
    from footballcoach.engine.movement import (
        SpeedMode, angle_diff, braking_speed_mode,
        effective_acceleration, effective_top_speed,
    )
    from footballcoach.steering import compute_repulsion

    has_ball = match.ball.possessed_by == player.player_id
    jog_speed = effective_top_speed(
        match.movement_params, player.attributes.top_speed, player.stamina,
        has_ball, player.attributes.ball_control, player.is_goalkeeper,
    ) * 0.5

    # ── Speed mode ──────────────────────────────────────────────────────────
    if arrival_dist is not None:
        a_max = effective_acceleration(
            match.movement_params, player.attributes.acceleration,
            player.stamina, player.is_goalkeeper,
        )
        eff_arrival = arrival_speed if arrival_speed is not None else jog_speed
        speed_mode = braking_speed_mode(
            arrival_dist, player.speed_mps, eff_arrival, a_max,
            match.movement_params.standstill_decel_multiplier, jog_speed, sprint,
        )
    else:
        speed_mode = SpeedMode.SPRINT if sprint else SpeedMode.JOG

    # ── Repulsion ────────────────────────────────────────────────────────────
    if use_repulsion and target_direction.length_xy() > 1e-9:
        adj_dir, speed_mult = compute_repulsion(
            player, target_direction, match.players,
            match.ball.possessed_by, match.repulsion_params,
        )
        if speed_mode is SpeedMode.SPRINT and speed_mult < 0.75:
            speed_mode = SpeedMode.JOG
    else:
        adj_dir = target_direction
        speed_mult = 1.0  # noqa: F841

    # ── Close-proximity lateral-overshoot brake ──────────────────────────────
    if (arrival_dist is not None and arrival_dist <= _CLOSE_PROX_M
            and player.speed_mps > _BRAKE_MIN_SPEED):
        vel_xy = player.velocity.xy()
        if vel_xy.length() > 1e-9 and adj_dir.length() > 1e-9:
            if vel_xy.normalized().dot(adj_dir.normalized()) < _CLOSE_PROX_COS:
                speed_mode = SpeedMode.STANDSTILL

    # ── Brake-to-turn ────────────────────────────────────────────────────────
    if (use_brake_to_turn and speed_mode is not SpeedMode.STANDSTILL
            and adj_dir.length() > 1e-9 and player.speed_mps > _BRAKE_MIN_SPEED):
        desired_heading = adj_dir.angle_xy()
        heading_error = abs(angle_diff(player.heading_rad, desired_heading))
        if heading_error > _BRAKE_THRESH_RAD:
            speed_mode = SpeedMode.STANDSTILL

    return adj_dir, speed_mode


def _gk_should_sprint(
    player: "Player",
    match: "Match",
    dist_to_save: float,
    gk_top_speed: float,
) -> bool:
    """Return True if the GK should sprint to the save point this tick.

    Sprints when the ball is heading toward goal and the GK travel time is within
    2× the ball arrival time (i.e. a real save attempt is needed).  Jogs when the
    ball is moving away, loose near the centre, or the GK easily has time to walk.
    """
    from footballcoach.entities.player import Team

    ball_vel = match.ball.velocity
    ball_speed = ball_vel.length()
    if match.ball.possessed_by is not None or ball_speed < 0.5:
        return False  # ball not in flight

    # Velocity component directly toward the GK's own goal line.
    if player.team == Team.LEFT:
        vel_toward_goal = -ball_vel.x  # negative x = toward left goal
    else:
        vel_toward_goal = ball_vel.x   # positive x = toward right goal

    if vel_toward_goal <= 0.3:
        return False  # ball moving away from or sideways to goal

    # Estimate ball arrival time at goal line (xy only).
    if player.team == Team.LEFT:
        dist_ball_to_goal = abs(-match.pitch.half_length - match.ball.position.x)
    else:
        dist_ball_to_goal = abs(match.pitch.half_length - match.ball.position.x)

    t_ball = dist_ball_to_goal / vel_toward_goal
    t_gk = dist_to_save / gk_top_speed if gk_top_speed > 0.1 else float("inf")
    return t_ball < t_gk * 2.0  # sprint if GK would otherwise be beaten


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

    def execute(self, player: "Player", match: "Match", dt: float) -> bool:
        """Execute one tick of movement toward target_position.  Returns True when arrived."""
        from footballcoach.engine.movement import SpeedMode, effective_top_speed

        self.status = OrderStatus.IN_PROGRESS
        has_ball = match.ball.possessed_by == player.player_id
        direction = self.target_position - player.position
        dist = direction.length_xy()

        if dist <= self.arrival_tolerance_m:
            self.reached_target = True

        jog_speed = effective_top_speed(
            match.movement_params, player.attributes.top_speed, player.stamina,
            has_ball, player.attributes.ball_control, player.is_goalkeeper,
        ) * 0.5
        arrival_speed = (
            self.max_speed_on_arrival_mps
            if self.max_speed_on_arrival_mps is not None
            else jog_speed
        )
        effective_tolerance = (
            self.arrival_tolerance_m * 1.5
            if arrival_speed < 0.1
            else self.arrival_tolerance_m
        )
        speed_ok = (
            self.max_speed_on_arrival_mps is None
            or player.speed_mps <= arrival_speed + 0.05
        )
        if dist <= effective_tolerance and speed_ok:
            return True
        elif self.reached_target and dist > self.arrival_tolerance_m:
            if self._overshoot_timer_s is None:
                self._overshoot_timer_s = self.overshoot_timeout_s
            self._overshoot_timer_s -= dt
            if self._overshoot_timer_s <= 0.0:
                return True
            else:
                player.desired_direction = Vector3.zero()
                player.desired_speed_mode = SpeedMode.STANDSTILL
        else:
            adj_dir, speed_mode = _compute_movement_intent(
                player, direction, match,
                sprint=self.sprint, arrival_dist=dist, arrival_speed=arrival_speed,
                use_repulsion=True, use_brake_to_turn=True,
            )
            player.desired_direction = adj_dir
            player.desired_speed_mode = speed_mode
        return False


@dataclass
class KickOrder:
    aim_point: Vector3  # absolute world position the kicker intends to hit
    power_fraction: float  # in [0, 1]; or >1 when compensate_for_run=False and caller wants old raw behaviour
    spin: Vector3
    compensate_for_run: bool = True  # if True, match.py pre-divides by run_mult so the ball
                                     # leaves at the intended speed regardless of run direction
    status: OrderStatus = OrderStatus.PENDING
    on_complete: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def execute(self, player: "Player", match: "Match", dt: float) -> bool:
        """Kick the ball this tick if the player has possession.  Always completes in one tick."""
        from footballcoach.engine.kicking import kick_ball, compensate_power_for_run_mult, running_power_multiplier
        from footballcoach.engine.movement import effective_top_speed

        if match.ball.possessed_by == player.player_id:
            top_speed = effective_top_speed(
                match.movement_params, player.attributes.top_speed, player.stamina,
                has_ball=True, ball_control_attr=player.attributes.ball_control,
            )
            run_mult = running_power_multiplier(
                match.kicking_params.running_power_coefficient, player.velocity,
                self.aim_point - player.position, top_speed,
            )
            kick_ball(
                match.ball,
                player.position,
                self.aim_point,
                compensate_power_for_run_mult(self.power_fraction, run_mult) if self.compensate_for_run else self.power_fraction,
                player.attributes.kick_precision,
                player.attributes.kick_power,
                self.spin,
                match.rng_reduction,
                match.rng,
                match.kicking_params,
                kicker_velocity=player.velocity,
                kicker_top_speed_mps=top_speed,
            )
            match._start_release_grace(player.player_id)
            match._log_debug(f"{player.player_id} kicked  power={self.power_fraction:.2f}")
            if player.on_kick is not None:
                player.on_kick(player)
        return True


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

    def execute(self, player: "Player", match: "Match", dt: float) -> bool:
        """Execute the pass this tick if the player has possession.  Always completes in one tick."""
        from footballcoach.engine.kicking import compensate_power_for_run_mult, pass_ball, running_power_multiplier
        from footballcoach.engine.movement import effective_top_speed

        if match.ball.possessed_by == player.player_id:
            pass_target = match._leading_pass_target(player, self)
            top_speed = effective_top_speed(
                match.movement_params, player.attributes.top_speed, player.stamina,
                has_ball=True, ball_control_attr=player.attributes.ball_control,
            )
            run_mult = running_power_multiplier(
                match.kicking_params.running_power_coefficient, player.velocity,
                pass_target - player.position, top_speed,
            )
            compensated = (
                compensate_power_for_run_mult(self.power_fraction, run_mult)
                if self.power_fraction is not None else None
            )
            pass_ball(
                match.ball,
                player.position,
                pass_target,
                player.attributes.kick_precision,
                match.rng_reduction,
                match.rng,
                match.passing_params,
                gravity_mps2=match.ball_physics_params.gravity_mps2,
                rolling_friction_coefficient=match.ball_physics_params.rolling_friction_coefficient,
                power_fraction=compensated,
                running_power_coefficient=match.kicking_params.running_power_coefficient,
                kicker_velocity=player.velocity,
                kicker_top_speed_mps=top_speed,
                kick_power_attr=player.attributes.kick_power,
                kicking_params=match.kicking_params,
            )
            match._start_release_grace(player.player_id)
            match._log_debug(f"{player.player_id} passed to {pass_target}")
            if player.on_kick is not None:
                player.on_kick(player)
        return True


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

    def execute(self, player: "Player", match: "Match", dt: float) -> bool:
        """Chase target and attempt one tackle on contact.  Returns True once contact is resolved."""
        from footballcoach.engine.collision import are_touching
        from footballcoach.engine.tackling import apply_tackle_result, attempt_tackle, tackle_angle_modifier

        self.status = OrderStatus.IN_PROGRESS
        target = match.player_by_id(self.target_player_id)
        if are_touching(player, target):
            if target.is_available_to_tackle():
                if player.on_tackle is not None:
                    player.on_tackle(player)
                if match._gk_immune_from_tackle(target):
                    match._apply_gk_immune_penalty(player)
                    match._log_info(f"{player.player_id} chase-tackle on {target.player_id} auto-failed [GK in own box]")
                else:
                    result = attempt_tackle(
                        player.attributes.tackling,
                        match._effective_dribbling(target),
                        match.rng_reduction,
                        match.rng,
                        match.tackling_params,
                        is_goalkeeper_tackle=player.is_goalkeeper,
                        angle_modifier=tackle_angle_modifier(
                            target.heading_rad, target.position, player.position, match.tackling_params
                        ),
                        gk_outside_box=match._gk_outside_own_box(player),
                    )
                    match._log_tackle_result(player.player_id, target.player_id, result)
                    if result.tackler_won and match._target_has_or_controls_ball(target):
                        match._set_possession(player.player_id)
                    apply_tackle_result(result, player, target, match.tackling_params)
            return True
        else:
            adj_dir, speed_mode = _compute_movement_intent(
                player, target.position - player.position, match,
                sprint=True, arrival_dist=None,
                use_repulsion=False, use_brake_to_turn=True,
            )
            player.desired_direction = adj_dir
            player.desired_speed_mode = speed_mode
            return False


@dataclass
class SaveOrder:
    """Goalkeeper-only "Save" action: continuously predicts where an
    in-flight ball will cross this keeper's own goal line and moves there
    (see engine/goalkeeping.py). Deliberately does not auto-complete like
    the other orders - a real goalkeeper is always "on duty", holding a
    sensible default position (goal centre) when no shot is incoming and
    reacting the instant one is. Issue it once; it stays in effect until
    replaced by another order.

    ``auto_sprint`` (default ``True``): each tick the order measures whether the
    ball is heading toward goal and whether there is enough time to jog or whether
    sprinting is needed to beat the ball.  Sprint is used when the estimated ball
    arrival time is within 2× the GK travel time.  Set to ``False`` and use
    ``sprint=False`` for Phase-1 training repositioning where no live shot is
    expected and jogging looks more natural.
    """
    auto_sprint: bool = True
    sprint: bool = True  # fallback when auto_sprint=False
    status: OrderStatus = OrderStatus.PENDING
    on_complete: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def execute(self, player: "Player", match: "Match", dt: float) -> bool:
        """Move keeper toward save position each tick.  Never returns True (persistent duty)."""
        from footballcoach.engine.goalkeeping import early_intercept_target, save_target_position
        from footballcoach.engine.movement import SpeedMode, effective_top_speed
        from footballcoach.entities.player import Team

        self.status = OrderStatus.IN_PROGRESS
        has_ball = match.ball.possessed_by == player.player_id

        if not player.is_goalkeeper:
            return True  # silently no-op for outfield
        if match._goal_linger_remaining_s > 0.0:
            return True  # goal just scored — cancel
        if has_ball:
            player.desired_direction = Vector3.zero()
            player.desired_speed_mode = SpeedMode.STANDSTILL
            return False  # never auto-completes

        gk_top_speed = effective_top_speed(
            match.movement_params, player.attributes.top_speed,
            player.stamina, has_ball=False, is_goalkeeper=True,
        )
        intercept = early_intercept_target(
            gk_position=player.position,
            gk_effective_top_speed_mps=gk_top_speed,
            ball_position=match.ball.position,
            ball_velocity=match.ball.velocity,
            pitch=match.pitch,
            team=player.team,
            gravity_mps2=match.ball_physics_params.gravity_mps2,
            params=match.goalkeeping_params,
        )
        target_position = intercept if intercept is not None else save_target_position(
            match.pitch,
            player.team,
            match.ball.position,
            match.ball.velocity,
            match.ball_physics_params.gravity_mps2,
            match.goalkeeping_params,
        )
        direction = target_position - player.position
        dist_to_save = direction.length_xy()
        snap_threshold = max(0.15, gk_top_speed * dt)
        if dist_to_save < snap_threshold:
            # GK is within one tick's reach — snap position then brake.
            player.position = target_position.with_z(player.position.z)
            player.desired_direction = Vector3.zero()
            player.desired_speed_mode = SpeedMode.STANDSTILL
        else:
            # Decide sprint vs jog.
            if self.auto_sprint:
                use_sprint = _gk_should_sprint(player, match, dist_to_save, gk_top_speed)
            else:
                use_sprint = self.sprint
            adj_dir, speed_mode = _compute_movement_intent(
                player, direction, match,
                sprint=use_sprint, arrival_dist=dist_to_save, arrival_speed=None,
                use_repulsion=False, use_brake_to_turn=True,
            )
            player.desired_direction = adj_dir
            player.desired_speed_mode = speed_mode
        return False  # never auto-completes


@dataclass
class StopOrder:
    """Decelerates the player to a complete stop using their normal braking
    capability, then completes. Useful for explicitly halting a player who is
    mid-sprint without snapping their velocity to zero instantly."""
    status: OrderStatus = OrderStatus.PENDING
    on_complete: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def execute(self, player: "Player", match: "Match", dt: float) -> bool:
        """Brake to standstill; complete when speed reaches zero."""
        from footballcoach.engine.movement import SpeedMode

        self.status = OrderStatus.IN_PROGRESS
        player.desired_direction = Vector3.zero()
        player.desired_speed_mode = SpeedMode.STANDSTILL
        return player.speed_mps == 0.0


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
    _possession_gained: bool = field(default=False, init=False, repr=False, compare=False)
    _callback_registered: bool = field(default=False, init=False, repr=False, compare=False)

    def execute(self, player: "Player", match: "Match", dt: float) -> bool:
        """Chase the ball / carrier; complete when this player gains possession."""
        self.status = OrderStatus.IN_PROGRESS

        # Register possession callback once so we don't poll ball.possessed_by.
        if not self._callback_registered:
            _order = self  # capture for closure

            def _on_possession(p: "Player") -> None:
                _order._possession_gained = True

            player.on_possession_gained = _on_possession
            self._callback_registered = True

        # Already have the ball (or callback just fired).
        if match.ball.possessed_by == player.player_id or self._possession_gained:
            player.on_possession_gained = None
            return True

        done = match._run_get_possession_behaviour(player, dt)
        if done:
            player.on_possession_gained = None
            return True
        return False


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

    def execute(self, player: "Player", match: "Match", dt: float) -> bool:
        """Mark target player; persistent (never auto-completes)."""
        from footballcoach.entities.player import PlayerState

        self.status = OrderStatus.IN_PROGRESS
        try:
            mark_target = match.player_by_id(self.target_player_id)
        except KeyError:
            return True  # target gone
        target_has_ball = (
            match.ball.possessed_by == mark_target.player_id
            or mark_target.state == PlayerState.CONTROLLING_BALL
        )
        ball_dist_m = player.position.xy().distance_to(match.ball.position.xy())
        if target_has_ball or ball_dist_m <= match.marking_params.mark_intercept_radius_m:
            match._run_get_possession_behaviour(player, dt)
            return False  # never auto-completes
        to_ball = (match.ball.position - mark_target.position).xy()
        toward_ball = to_ball.normalized() if to_ball.length() > 1e-6 else Vector3.zero()
        mark_pos = mark_target.position.with_z(0.0) + toward_ball * match.marking_params.mark_standoff_m
        direction = mark_pos - player.position
        adj_dir, speed_mode = _compute_movement_intent(
            player, direction, match,
            sprint=True, arrival_dist=direction.length_xy(), arrival_speed=0.0,
            use_repulsion=False, use_brake_to_turn=True,
        )
        player.desired_direction = adj_dir
        player.desired_speed_mode = speed_mode
        return False  # never auto-completes


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

    def execute(self, player: "Player", match: "Match", dt: float) -> bool:
        """Shoot at aim_point if player has possession; may pause when a blocker is on the line."""
        from footballcoach.engine.kicking import compensate_power_for_run_mult, has_blocker_on_shot_line, kick_ball, running_power_multiplier
        from footballcoach.engine.movement import effective_top_speed

        _SHOT_BLOCKER_M = 1.0
        _SHOT_PAUSE_M = 2.0

        if match.ball.possessed_by != player.player_id:
            return True  # no-op: lost ball

        opposition = [p for p in match.players if p.team != player.team]
        if (self.chance_of_pausing > 0.0
                and has_blocker_on_shot_line(player.position, self.aim_point, opposition, _SHOT_BLOCKER_M)
                and match.rng.random() < self.chance_of_pausing):
            aim_dir = (self.aim_point - player.position).xy()
            aim_len = aim_dir.length_xy()
            if aim_len > 1e-9:
                step_dir = aim_dir / aim_len
                raw_target = player.position.xy() + step_dir * _SHOT_PAUSE_M
                clamped_target = Vector3(
                    max(-match.pitch.half_length + 0.5, min(match.pitch.half_length - 0.5, raw_target.x)),
                    max(-match.pitch.half_width + 0.5, min(match.pitch.half_width - 0.5, raw_target.y)),
                    0.0,
                )
                player.current_order = MoveOrder(target_position=clamped_target, sprint=True)
                match._log_debug(
                    f"{player.player_id} shoot paused (blocker) → advancing to "
                    f"({clamped_target.x:.1f},{clamped_target.y:.1f})"
                )
                return False  # new MoveOrder installed; caller must NOT call _complete_order

        top_speed = effective_top_speed(
            match.movement_params, player.attributes.top_speed, player.stamina,
            has_ball=True, ball_control_attr=player.attributes.ball_control,
        )
        run_mult = running_power_multiplier(
            match.kicking_params.running_power_coefficient, player.velocity,
            self.aim_point - player.position, top_speed,
        )
        kick_ball(
            match.ball,
            player.position,
            self.aim_point,
            compensate_power_for_run_mult(self.power_fraction, run_mult) if self.compensate_for_run else self.power_fraction,
            player.attributes.kick_precision,
            player.attributes.kick_power,
            Vector3.zero(),
            match.rng_reduction,
            match.rng,
            match.kicking_params,
            kicker_velocity=player.velocity,
            kicker_top_speed_mps=top_speed,
        )
        match._start_release_grace(player.player_id)
        match._log_info(f"{player.player_id} shot at goal  power={self.power_fraction:.2f}")
        if player.on_kick is not None:
            player.on_kick(player)
        return True


Order = MoveOrder | KickOrder | ShootOrder | PassOrder | ChaseTackleOrder | SaveOrder | StopOrder | GetPossessionOrder | MarkOrder

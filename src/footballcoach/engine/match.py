"""Match: the top-level engine tying together movement, ball physics,
collision, kicking, tackling, possession, offside, and scoring into a single
steppable simulation. This is the main entry point other code (tests, a
future UI, a future RL training loop) should use.

See engine/knowledge.md for the overall tick order and design rationale.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from footballcoach.ui.gamelog import LogLevel

from footballcoach.config import load_physics_config
from footballcoach.engine.ball_physics import BallPhysicsParams, step_ball
from footballcoach.engine.collision import (
    CollisionParams,
    are_touching,
    resolve_all_overlaps,
    resolve_ball_block_by_inactive_players,
)
from footballcoach.engine.goalkeeping import GoalkeepingParams, early_intercept_target, save_target_position
from footballcoach.engine.kicking import KickingParams, PassingParams, kick_ball, pass_ball, pass_speed_mps, compensate_power_for_run_mult, running_power_multiplier
from footballcoach.engine.movement import (
    MovementParams,
    SpeedMode,
    effective_acceleration,
    effective_top_speed,
    regen_stamina,
    step_player_towards,
)
from footballcoach.engine.possession import ControlTimeParams, control_time_s
from footballcoach.engine.scoring import Scoreboard, check_goal
from footballcoach.engine.tackling import TacklingParams, attempt_tackle, tackle_angle_modifier
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Player, PlayerState, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import (
    ChaseTackleOrder,
    GetPossessionOrder,
    KickOrder,
    MarkOrder,
    MoveOrder,
    OrderStatus,
    PassOrder,
    SaveOrder,
    ShootOrder,
    StopOrder,
    TackleOrder,
)
from footballcoach.steering import RepulsionParams, compute_repulsion

import logging
log = logging.getLogger("footballcoach.match")


@dataclass(frozen=True)
class MarkingParams:
    """Config for the ``MarkOrder`` AI behaviour — loaded from
    ``physics.json["marking"]``."""
    mark_intercept_radius_m: float = 4.0
    mark_standoff_m: float = 1.5

    @staticmethod
    def from_config() -> "MarkingParams":
        d = load_physics_config().get("marking", {})
        return MarkingParams(
            mark_intercept_radius_m=d.get("mark_intercept_radius_m", 4.0),
            mark_standoff_m=d.get("mark_standoff_m", 1.5),
        )


@dataclass
class Match:
    pitch: Pitch
    players: list[Player]
    ball: Ball
    rng_reduction: float = 0.3
    dt_s: float = 1.0 / 30.0
    offside_enabled: bool = True
    scoreboard: Scoreboard = field(default_factory=Scoreboard)
    rng: random.Random = field(default_factory=random.Random)

    movement_params: MovementParams = field(default_factory=MovementParams.from_config)
    ball_physics_params: BallPhysicsParams = field(default_factory=BallPhysicsParams.from_config)
    kicking_params: KickingParams = field(default_factory=KickingParams.from_config)
    passing_params: PassingParams = field(default_factory=PassingParams.from_config)
    control_time_params: ControlTimeParams = field(default_factory=ControlTimeParams.from_config)
    tackling_params: TacklingParams = field(default_factory=TacklingParams.from_config)
    goalkeeping_params: GoalkeepingParams = field(default_factory=GoalkeepingParams.from_config)
    repulsion_params: RepulsionParams = field(default_factory=RepulsionParams.from_config)
    collision_params: CollisionParams = field(default_factory=CollisionParams.from_config)
    marking_params: MarkingParams = field(default_factory=MarkingParams.from_config)

    paused: bool = False
    time_s: float = 0.0

    # Pickup radius: how close a loose ball must be to a player before that
    # player begins the first-touch control-time countdown.
    pickup_radius_m: float = 0.4

    # How long (seconds) the player who just released the ball (kick/pass)
    # is excluded from re-picking it up. Needed because a slow pass/kick
    # (a few m/s) doesn't necessarily clear the passer's own pickup radius
    # within a single physics tick - see engine/knowledge.md.
    release_grace_duration_s: float = 0.3

    # Sim-seconds to keep the ball in the net after a goal before resetting
    # to centre.  0.0 = immediate reset (tests / headless use).  Set to
    # physics.json["ui"]["goal_linger_s"] when creating a match for the UI.
    goal_linger_s: float = 0.0
    _goal_linger_remaining_s: float = field(default=0.0, init=False, repr=False)

    # Optional callback invoked with (level, message) for UI game-log display.
    # None by default so tests / headless use incur zero cost.
    log_callback: Callable[["LogLevel", str], None] | None = field(default=None, repr=False)

    def player_by_id(self, player_id: str) -> Player:
        for p in self.players:
            if p.player_id == player_id:
                return p
        raise KeyError(f"no player with id {player_id}")

    def ball_carrier(self) -> Player | None:
        if self.ball.possessed_by is None:
            return None
        return self.player_by_id(self.ball.possessed_by)

    def step(self) -> None:
        if self.paused:
            return
        dt = self.dt_s

        self._update_state_timers(dt)
        self._process_orders(dt)
        self._sync_possessed_ball()

        # Advance a loose ball's free-flight physics *before* checking for
        # pickup, so a ball that was just kicked this tick has already moved
        # away from the kicker's feet before we check whether anyone is close
        # enough to start controlling it. Without this ordering, a player who
        # just kicked the ball would immediately re-acquire it at distance 0.
        #
        # A ball currently mid-control-time (some player is
        # CONTROLLING_BALL) is NOT advanced here: the instant a player makes
        # contact, the ball is frozen (see _update_loose_ball_pickup) rather
        # than continuing to fly at full speed for the whole control-time
        # window. Without this, a fast shot could sail straight through a
        # goalkeeper who is technically "catching" it and still cross the
        # goal line before the control-time timer completes.
        if self.ball.possessed_by is None and not self._any_player_controlling_ball():
            pre_flight_position = self.ball.position
            step_ball(self.ball, dt, self.ball_physics_params)
            resolve_ball_block_by_inactive_players(
                self.ball, self.players, pre_flight_position, self.ball_physics_params.block_restitution
            )

        self._update_loose_ball_pickup(dt)

        self._check_head_on_tackles()
        resolve_all_overlaps(self.players, collision_params=self.collision_params)

        # Goal linger: if a goal was recently scored, count down before
        # resetting the ball.  Skip further goal detection during the linger.
        if self._goal_linger_remaining_s > 0.0:
            self._goal_linger_remaining_s -= dt
            if self._goal_linger_remaining_s <= 0.0:
                self._goal_linger_remaining_s = 0.0
                self._reset_after_goal()
        else:
            self._check_goal()

        if self.ball.release_grace_s > 0.0:
            self.ball.release_grace_s = max(0.0, self.ball.release_grace_s - dt)
            if self.ball.release_grace_s == 0.0:
                self.ball.last_released_by = None

        self.time_s += dt

    # -- logging helpers (zero cost when log_callback is None) ---------------

    def _log_info(self, msg: str) -> None:
        if self.log_callback is not None:
            from footballcoach.ui.gamelog import LogLevel  # local import: only paid when UI is live
            self.log_callback(LogLevel.INFO, msg)

    def _log_debug(self, msg: str) -> None:
        if self.log_callback is not None:
            from footballcoach.ui.gamelog import LogLevel
            self.log_callback(LogLevel.DEBUG, msg)

    # -------------------------------------------------------------------------

    def _start_release_grace(self, player_id: str) -> None:
        """Marks `player_id` as temporarily unable to re-pick-up the ball
        they just released (kicked/passed), for `release_grace_duration_s`.
        See the `release_grace_duration_s` field docstring for why this is
        needed."""
        self.ball.last_released_by = player_id
        self.ball.release_grace_s = self.release_grace_duration_s

    def _update_state_timers(self, dt: float) -> None:
        for player in self.players:
            if player.state == PlayerState.ACTIVE:
                player.stamina = regen_stamina(
                    self.movement_params, player.stamina, player.attributes.stamina, dt * 0.3
                )
                continue
            player.state_timer_s -= dt
            if player.state_timer_s <= 0.0:
                if player.state == PlayerState.CONTROLLING_BALL:
                    self._complete_control(player)
                player.state = PlayerState.ACTIVE
                player.state_timer_s = 0.0

    def _process_orders(self, dt: float) -> None:
        for player in self.players:
            if player.state != PlayerState.ACTIVE:
                continue
            order = player.current_order
            if order is None:
                continue

            has_ball = self.ball.possessed_by == player.player_id

            if isinstance(order, MoveOrder):
                order.status = OrderStatus.IN_PROGRESS
                direction = order.target_position - player.position
                dist = direction.length_xy()

                # Resolve arrival speed: None means jog speed (natural coast-in).
                jog_speed = effective_top_speed(
                    self.movement_params, player.attributes.top_speed, player.stamina,
                    has_ball, player.attributes.ball_control, player.is_goalkeeper,
                ) * 0.5
                arrival_speed = (
                    order.max_speed_on_arrival_mps
                    if order.max_speed_on_arrival_mps is not None
                    else jog_speed
                )
                # Widen tolerance for standstill arrivals to give the braking
                # logic a comfortable window to decelerate into.
                effective_tolerance = (
                    order.arrival_tolerance_m * 1.5
                    if arrival_speed < 0.1
                    else order.arrival_tolerance_m
                )

                # Speed check only enforced when caller explicitly requested a
                # max arrival speed — the default (None) just means "get there".
                speed_ok = (
                    order.max_speed_on_arrival_mps is None
                    or player.speed_mps <= arrival_speed + 0.05
                )
                if dist <= effective_tolerance and speed_ok:
                    order.status = OrderStatus.COMPLETE
                    player.current_order = None
                else:
                    a_max = effective_acceleration(
                        self.movement_params, player.attributes.acceleration,
                        player.stamina, player.is_goalkeeper,
                    )
                    speed_mode = _braking_speed_mode(
                        dist, player.speed_mps, arrival_speed, a_max,
                        self.movement_params.standstill_decel_multiplier, jog_speed, order.sprint,
                    )
                    # Apply repulsion steering: adjust direction and compute speed
                    # multiplier. This is an AI/order-layer decision (steering.py);
                    # step_player_towards itself stays a pure kinematics function.
                    adj_dir, speed_mult = compute_repulsion(
                        player, direction, self.players,
                        self.ball.possessed_by, self.repulsion_params,
                    )
                    # Repulsion may downgrade SPRINT→JOG, but never overrides
                    # STANDSTILL (braking takes priority over avoidance steering).
                    if speed_mode is SpeedMode.SPRINT and speed_mult < 0.75:
                        speed_mode = SpeedMode.JOG
                    log.debug(
                        "[move] t=%.3f  pid=%s  pos=(%.3f,%.3f)  "
                        "raw_dir=(%.3f,%.3f)  adj_dir=(%.3f,%.3f)  "
                        "speed_mult=%.3f  mode=%s  has_ball=%s",
                        self.time_s, player.player_id,
                        player.position.x, player.position.y,
                        direction.x, direction.y,
                        adj_dir.x, adj_dir.y,
                        speed_mult, speed_mode.name, has_ball,
                    )
                    step_player_towards(
                        player, adj_dir, speed_mode, dt, self.movement_params, has_ball
                    )
                    player.stamina = _drain_if_sprinting(
                        self.movement_params, player, speed_mode is SpeedMode.SPRINT, dt
                    )

            elif isinstance(order, KickOrder):
                if has_ball:
                    _kick_top_speed = effective_top_speed(
                        self.movement_params, player.attributes.top_speed, player.stamina,
                        has_ball=True, ball_control_attr=player.attributes.ball_control,
                    )
                    _kick_run_mult = running_power_multiplier(
                        self.kicking_params.running_power_coefficient, player.velocity,
                        order.aim_point - player.position, _kick_top_speed,
                    )
                    kick_ball(
                        self.ball,
                        player.position,
                        order.aim_point,
                        compensate_power_for_run_mult(order.power_fraction, _kick_run_mult) if order.compensate_for_run else order.power_fraction,
                        player.attributes.kick_precision,
                        player.attributes.kick_power,
                        order.spin,
                        self.rng_reduction,
                        self.rng,
                        self.kicking_params,
                        kicker_velocity=player.velocity,
                        kicker_top_speed_mps=_kick_top_speed,
                    )
                    self._start_release_grace(player.player_id)
                    self._log_debug(f"{player.player_id} kicked  power={order.power_fraction:.2f}")
                order.status = OrderStatus.COMPLETE
                player.current_order = None

            elif isinstance(order, ShootOrder):
                if has_ball:
                    _shoot_top_speed = effective_top_speed(
                        self.movement_params, player.attributes.top_speed, player.stamina,
                        has_ball=True, ball_control_attr=player.attributes.ball_control,
                    )
                    _shoot_run_mult = running_power_multiplier(
                        self.kicking_params.running_power_coefficient, player.velocity,
                        order.aim_point - player.position, _shoot_top_speed,
                    )
                    kick_ball(
                        self.ball,
                        player.position,
                        order.aim_point,
                        compensate_power_for_run_mult(order.power_fraction, _shoot_run_mult) if order.compensate_for_run else order.power_fraction,
                        player.attributes.kick_precision,
                        player.attributes.kick_power,
                        Vector3.zero(),
                        self.rng_reduction,
                        self.rng,
                        self.kicking_params,
                        kicker_velocity=player.velocity,
                        kicker_top_speed_mps=_shoot_top_speed,
                    )
                    self._start_release_grace(player.player_id)
                    self._log_info(f"{player.player_id} shot at goal  power={order.power_fraction:.2f}")
                order.status = OrderStatus.COMPLETE
                player.current_order = None

            elif isinstance(order, TackleOrder):
                target = self.player_by_id(order.target_player_id)
                if are_touching(player, target) and target.is_available_to_tackle():
                    if self._gk_immune_from_tackle(target):
                        # Phase B: GK in own box with ball is untackleable —
                        # auto-fail, tackler penalised for the lunge, GK
                        # state is completely unchanged.
                        player.state = PlayerState.INACTIVE_TACKLED
                        player.state_timer_s = self.tackling_params.tackler_miss_inactive_duration_s
                        self._log_info(f"{player.player_id} tackle on {target.player_id} auto-failed [GK in own box]")
                    else:
                        result = attempt_tackle(
                            player.attributes.tackling,
                            self._effective_dribbling(target),  # Phase B: CONTROLLING_BALL penalty
                            self.rng_reduction,
                            self.rng,
                            self.tackling_params,
                            is_goalkeeper_tackle=player.is_goalkeeper,
                            angle_modifier=tackle_angle_modifier(
                                target.heading_rad, target.position, player.position, self.tackling_params
                            ),
                            gk_outside_box=self._gk_outside_own_box(player),  # Phase B: GK penalty
                        )
                        self._log_tackle_result(player.player_id, target.player_id, result)
                        if result.tackler_won and self._target_has_or_controls_ball(target):
                            self.ball.possessed_by = player.player_id
                        elif not result.tackler_won:
                            target.velocity = target.velocity * result.dribble_speed_multiplier
                        target.state = PlayerState.INACTIVE_TACKLED
                        target.state_timer_s = self.tackling_params.inactive_duration_s
                        if not result.tackler_won:
                            # A failed (lunging/sliding) tackle also leaves the
                            # tackler briefly unable to react - shorter than the
                            # dispossessed player's penalty, since they weren't
                            # actually beaten on the ball, just off-balance.
                            player.state = PlayerState.INACTIVE_TACKLED
                            player.state_timer_s = self.tackling_params.tackler_miss_inactive_duration_s
                order.status = OrderStatus.COMPLETE
                player.current_order = None

            elif isinstance(order, PassOrder):
                if has_ball:
                    pass_target = self._leading_pass_target(player, order)
                    _pass_top_speed = effective_top_speed(
                        self.movement_params, player.attributes.top_speed, player.stamina,
                        has_ball=True, ball_control_attr=player.attributes.ball_control,
                    )
                    _pass_run_mult = running_power_multiplier(
                        self.kicking_params.running_power_coefficient, player.velocity,
                        pass_target - player.position, _pass_top_speed,
                    )
                    # Compensate explicit power_fraction only; auto-pace (None) is
                    # distance-based and already targets the right arrival speed.
                    _compensated_pass_power = (
                        compensate_power_for_run_mult(order.power_fraction, _pass_run_mult)
                        if order.power_fraction is not None else None
                    )
                    pass_ball(
                        self.ball,
                        player.position,
                        pass_target,
                        player.attributes.kick_precision,
                        self.rng_reduction,
                        self.rng,
                        self.passing_params,
                        gravity_mps2=self.ball_physics_params.gravity_mps2,
                        rolling_friction_coefficient=self.ball_physics_params.rolling_friction_coefficient,
                        power_fraction=_compensated_pass_power,
                        running_power_coefficient=self.kicking_params.running_power_coefficient,
                        kicker_velocity=player.velocity,
                        kicker_top_speed_mps=_pass_top_speed,
                        kick_power_attr=player.attributes.kick_power,
                        kicking_params=self.kicking_params,
                    )
                    self._start_release_grace(player.player_id)
                    self._log_debug(f"{player.player_id} passed to {pass_target}")
                order.status = OrderStatus.COMPLETE
                player.current_order = None

            elif isinstance(order, ChaseTackleOrder):
                order.status = OrderStatus.IN_PROGRESS
                target = self.player_by_id(order.target_player_id)
                if are_touching(player, target):
                    if target.is_available_to_tackle():
                        if self._gk_immune_from_tackle(target):
                            # Phase B: GK in own box with ball is untackleable.
                            player.state = PlayerState.INACTIVE_TACKLED
                            player.state_timer_s = self.tackling_params.tackler_miss_inactive_duration_s
                            self._log_info(f"{player.player_id} chase-tackle on {target.player_id} auto-failed [GK in own box]")
                        else:
                            result = attempt_tackle(
                                player.attributes.tackling,
                                self._effective_dribbling(target),  # Phase B: CONTROLLING_BALL penalty
                                self.rng_reduction,
                                self.rng,
                                self.tackling_params,
                                is_goalkeeper_tackle=player.is_goalkeeper,
                                angle_modifier=tackle_angle_modifier(
                                    target.heading_rad, target.position, player.position, self.tackling_params
                                ),
                                gk_outside_box=self._gk_outside_own_box(player),  # Phase B: GK penalty
                            )
                            self._log_tackle_result(player.player_id, target.player_id, result)
                            if result.tackler_won and self._target_has_or_controls_ball(target):
                                self.ball.possessed_by = player.player_id
                            elif not result.tackler_won:
                                target.velocity = target.velocity * result.dribble_speed_multiplier
                            target.state = PlayerState.INACTIVE_TACKLED
                            target.state_timer_s = self.tackling_params.inactive_duration_s
                            if not result.tackler_won:
                                player.state = PlayerState.INACTIVE_TACKLED
                                player.state_timer_s = self.tackling_params.tackler_miss_inactive_duration_s
                    order.status = OrderStatus.COMPLETE
                    player.current_order = None
                else:
                    direction = target.position - player.position
                    step_player_towards(player, direction, SpeedMode.SPRINT, dt, self.movement_params, has_ball)
                    player.stamina = _drain_if_sprinting(self.movement_params, player, True, dt)

            elif isinstance(order, GetPossessionOrder):
                order.status = OrderStatus.IN_PROGRESS
                if self.ball.possessed_by == player.player_id:
                    # Already have the ball.
                    order.status = OrderStatus.COMPLETE
                    player.current_order = None
                elif self._run_get_possession_behaviour(player, dt):
                    order.status = OrderStatus.COMPLETE
                    player.current_order = None

            elif isinstance(order, MarkOrder):
                order.status = OrderStatus.IN_PROGRESS
                try:
                    mark_target = self.player_by_id(order.target_player_id)
                except KeyError:
                    # Target no longer in the match.
                    order.status = OrderStatus.COMPLETE
                    player.current_order = None
                    continue
                target_has_ball = (
                    self.ball.possessed_by == mark_target.player_id
                    or mark_target.state == PlayerState.CONTROLLING_BALL
                )
                ball_dist_m = player.position.xy().distance_to(self.ball.position.xy())
                if target_has_ball or ball_dist_m <= self.marking_params.mark_intercept_radius_m:
                    # Chase / tackle the carrier or sprint to the loose ball.
                    self._run_get_possession_behaviour(player, dt)
                    # MarkOrder never auto-completes — marker stays assigned
                    # indefinitely (same model as SaveOrder).
                else:
                    # Standoff: approach the point between target and ball,
                    # decelerating to a standstill there.  _braking_speed_mode
                    # handles the SPRINT→JOG→STANDSTILL transition automatically;
                    # the physics-level snap in step_player_towards prevents drift.
                    to_ball = (self.ball.position - mark_target.position).xy()
                    toward_ball = to_ball.normalized() if to_ball.length() > 1e-6 else Vector3.zero()
                    mark_pos = mark_target.position.with_z(0.0) + toward_ball * self.marking_params.mark_standoff_m
                    direction = mark_pos - player.position
                    mark_a_max = effective_acceleration(
                        self.movement_params, player.attributes.acceleration,
                        player.stamina, player.is_goalkeeper,
                    )
                    mark_jog_speed = effective_top_speed(
                        self.movement_params, player.attributes.top_speed, player.stamina,
                        has_ball, player.attributes.ball_control, player.is_goalkeeper,
                    ) * 0.5
                    mark_speed_mode = _braking_speed_mode(
                        direction.length_xy(), player.speed_mps, 0.0, mark_a_max,
                        self.movement_params.standstill_decel_multiplier, mark_jog_speed, True,
                    )
                    step_player_towards(player, direction, mark_speed_mode, dt, self.movement_params, has_ball)
                    player.stamina = _drain_if_sprinting(
                        self.movement_params, player, mark_speed_mode is SpeedMode.SPRINT, dt
                    )

            elif isinstance(order, StopOrder):
                order.status = OrderStatus.IN_PROGRESS
                step_player_towards(player, Vector3.zero(), SpeedMode.STANDSTILL, dt, self.movement_params, has_ball)
                # Completion is driven by the physics-level snap inside
                # step_player_towards: once speed drops below
                # _STOP_SNAP_THRESHOLD_MPS (0.02 m/s), new_speed is zeroed,
                # so speed_mps == 0.0 exactly next tick.
                if player.speed_mps == 0.0:
                    order.status = OrderStatus.COMPLETE
                    player.current_order = None

            elif isinstance(order, SaveOrder):
                order.status = OrderStatus.IN_PROGRESS
                if not player.is_goalkeeper:
                    # SaveOrder is goalkeeper-only; silently no-op for an
                    # outfield player rather than raising, since orders are
                    # not otherwise role-restricted.
                    order.status = OrderStatus.COMPLETE
                    player.current_order = None
                else:
                    gk_top_speed_early = effective_top_speed(
                        self.movement_params, player.attributes.top_speed,
                        player.stamina, has_ball=False, is_goalkeeper=True,
                    )
                    intercept = early_intercept_target(
                        gk_position=player.position,
                        gk_effective_top_speed_mps=gk_top_speed_early,
                        ball_position=self.ball.position,
                        ball_velocity=self.ball.velocity,
                        pitch=self.pitch,
                        team=player.team,
                        gravity_mps2=self.ball_physics_params.gravity_mps2,
                        params=self.goalkeeping_params,
                    )
                    target_position = intercept if intercept is not None else save_target_position(
                        self.pitch,
                        player.team,
                        self.ball.position,
                        self.ball.velocity,
                        self.ball_physics_params.gravity_mps2,
                        self.goalkeeping_params,
                    )
                    direction = target_position - player.position
                    # Snap threshold: the larger of 0.15m (minimum grab
                    # radius) and the distance the keeper can cover in one
                    # tick at their current effective top speed.  Without
                    # the per-tick term, a fast keeper (especially with the
                    # goalkeeper_speed_multiplier boost) can travel further
                    # than the fixed threshold in a single tick and overshoot
                    # the target entirely, ending up on the wrong side.
                    # (gk_top_speed_early was already computed above for the
                    # early-intercept distance gate — reuse it.)
                    snap_threshold = max(0.15, gk_top_speed_early * dt)
                    if direction.length_xy() < snap_threshold:
                        # GK position teleport to the save point — this is a
                        # deliberate physics shortcut (the GK has already
                        # "arrived" within one tick's reach), so snapping both
                        # position and velocity is intentional and consistent.
                        player.position = target_position.with_z(player.position.z)
                        step_player_towards(player, Vector3.zero(), SpeedMode.STANDSTILL, dt, self.movement_params, has_ball)
                    else:
                        step_player_towards(player, direction, SpeedMode.SPRINT, dt, self.movement_params, has_ball)
                        player.stamina = _drain_if_sprinting(self.movement_params, player, True, dt)
                    # Never auto-completes - a goalkeeper is always "on
                    # duty" reacting to the ball; see orders.SaveOrder.

    def _run_get_possession_behaviour(self, player: Player, dt: float) -> bool:
        """Runs one tick of 'acquire the ball' behaviour, shared by
        ``GetPossessionOrder`` and ``MarkOrder``'s intercept/tackle fallback.

        - If another player has the ball: chase them and, on contact, attempt
          a tackle.  Returns ``True`` once the tackle has been resolved.
        - If the ball is loose: sprint to intercept.  Returns ``True`` when
          already within pickup radius (``_update_loose_ball_pickup`` handles
          the rest next tick).
        - Returns ``False`` while still en-route (chasing / sprinting).
        """
        has_ball = self.ball.possessed_by == player.player_id
        carrier = self.ball_carrier()

        if carrier is not None and carrier.player_id != player.player_id:
            # Someone else has the ball — chase and tackle.
            if are_touching(player, carrier):
                if carrier.is_available_to_tackle():
                    if self._gk_immune_from_tackle(carrier):
                        # Phase B: GK in own box with ball is untackleable.
                        player.state = PlayerState.INACTIVE_TACKLED
                        player.state_timer_s = self.tackling_params.tackler_miss_inactive_duration_s
                    else:
                        result = attempt_tackle(
                            player.attributes.tackling,
                            self._effective_dribbling(carrier),  # Phase B: CONTROLLING_BALL penalty
                            self.rng_reduction,
                            self.rng,
                            self.tackling_params,
                            is_goalkeeper_tackle=player.is_goalkeeper,
                            angle_modifier=tackle_angle_modifier(
                                carrier.heading_rad, carrier.position, player.position,
                                self.tackling_params,
                            ),
                            gk_outside_box=self._gk_outside_own_box(player),  # Phase B: GK penalty
                        )
                        self._log_tackle_result(player.player_id, carrier.player_id, result)
                        if result.tackler_won and self._target_has_or_controls_ball(carrier):
                            self.ball.possessed_by = player.player_id
                        else:
                            carrier.velocity = carrier.velocity * result.dribble_speed_multiplier
                        carrier.state = PlayerState.INACTIVE_TACKLED
                        carrier.state_timer_s = self.tackling_params.inactive_duration_s
                        if not result.tackler_won:
                            player.state = PlayerState.INACTIVE_TACKLED
                            player.state_timer_s = self.tackling_params.tackler_miss_inactive_duration_s
                return True  # tackle attempted (or target not available) — terminal
            else:
                intercept = self._intercept_target(player, carrier.position, carrier.velocity)
                direction = intercept - player.position
                step_player_towards(player, direction, SpeedMode.SPRINT, dt, self.movement_params, has_ball)
                player.stamina = _drain_if_sprinting(self.movement_params, player, True, dt)
                return False
        else:
            # Ball is loose — sprint to intercept; pickup via _update_loose_ball_pickup.
            intercept = self._intercept_target(player, self.ball.position, self.ball.velocity)
            direction = intercept - player.position
            if direction.length_xy() <= self.pickup_radius_m:
                return True  # at the ball — pickup will complete next tick
            step_player_towards(player, direction, SpeedMode.SPRINT, dt, self.movement_params, has_ball)
            player.stamina = _drain_if_sprinting(self.movement_params, player, True, dt)
            return False

    def _sync_possessed_ball(self) -> None:
        carrier = self.ball_carrier()
        if carrier is None:
            return
        offset = Vector3.from_angle_xy(carrier.heading_rad, carrier.radius_m + self.ball.radius_m)
        self.ball.position = carrier.position + offset
        self.ball.position = self.ball.position.with_z(self.ball.radius_m)
        self.ball.velocity = carrier.velocity

    def _update_loose_ball_pickup(self, dt: float) -> None:
        if self.ball.possessed_by is not None:
            return
        for player in self.players:
            if player.state != PlayerState.ACTIVE:
                continue
            if self.ball.release_grace_s > 0.0 and self.ball.last_released_by == player.player_id:
                continue  # can't re-pick-up the ball they just kicked/passed
            distance = player.position.xy().distance_to(self.ball.position.xy())
            if distance > self.pickup_radius_m:
                continue

            relative_speed = (self.ball.velocity - player.velocity).length()
            t_control = control_time_s(
                self.control_time_params,
                self.ball.height_m,
                relative_speed,
                player.speed_mps,
                player.attributes.ball_control,
                is_goalkeeper_in_box=player.is_goalkeeper and self.pitch.is_in_either_box(player.position),
            )
            noise_sigma = t_control * self.control_time_params.noise_sigma_fraction * (1.0 - self.rng_reduction)
            t_control = max(0.01, t_control + self.rng.gauss(0.0, noise_sigma))

            player.state = PlayerState.CONTROLLING_BALL
            player.state_timer_s = t_control
            # Freeze the ball in place the instant contact is made (keeping
            # whatever height it was caught/received at) - see the note in
            # step() on why a ball mid-control-time must not keep flying.
            self.ball.velocity = Vector3.zero()
            return  # only one player starts controlling per tick (first come)

    def _any_player_controlling_ball(self) -> bool:
        return any(p.state == PlayerState.CONTROLLING_BALL for p in self.players)

    def _leading_pass_target(self, passer: Player, order: "PassOrder") -> Vector3:
        """If the PassOrder has a ``target_player_id``, leads the pass to
        where that player will be when the ball arrives.

        The estimate: t_arrive ≈ distance / ball_speed (from auto-pace model),
        capped at 2 s. If the target is standing still (speed < 0.3 m/s) or
        doesn't exist, falls back to ``order.target_position``.
        """
        from footballcoach.orders import PassOrder  # local import to avoid circular
        if order.target_player_id is None:
            return order.target_position
        try:
            target = self.player_by_id(order.target_player_id)
        except KeyError:
            return order.target_position

        if target.speed_mps < 0.3:
            return order.target_position  # stationary - no lead needed

        dist = passer.position.xy().distance_to(order.target_position.xy())
        ball_speed = pass_speed_mps(
            self.passing_params, dist,
            self.ball_physics_params.gravity_mps2,
            self.ball_physics_params.rolling_friction_coefficient,
        )
        t_arrive = min(dist / max(ball_speed, 1.0), 2.0)
        predicted = order.target_position + target.velocity.xy() * t_arrive
        return predicted.with_z(0.0)

    def _intercept_target(self, player: Player, target_pos: Vector3, target_vel: Vector3) -> Vector3:
        """Estimates where a moving target (ball or carrier) will be when
        this player arrives at its current position, using the player's
        current speed as a rough time estimate.

        The prediction is intentionally simple - one linear step:
            t_estimate = min(dist / max(my_speed, 1.0), 3.0)
            intercept = target_pos + target_vel_xy * t_estimate

        This is accurate enough for a sprint interception run and avoids
        the player always running to where the ball/opponent currently is
        (which causes them to perpetually chase from behind). Capping at
        3 s prevents overshooting when the player is nearly stationary.
        Only the xy component of target_vel is used (height is irrelevant
        for ground-level interception runs).
        """
        dist = player.position.xy().distance_to(target_pos.xy())
        my_speed = max(player.speed_mps, 1.0)
        t_estimate = min(dist / my_speed, 3.0)
        predicted_xy = target_pos + target_vel.xy() * t_estimate
        return predicted_xy.with_z(target_pos.z)

    def _check_head_on_tackles(self) -> None:
        """Automatically triggers a tackle when two players from opposite teams
        are overlapping, one has the ball, and both are charging directly
        towards each other (head-on collision). This replaces the normal
        push-apart resolution for such a pair, since two players sprinting
        at each other frontally should not just bounce off - the player
        without the ball is attempting to win it.

        Only one such tackle is resolved per tick (the first qualifying pair
        found). The resulting tackle uses the same skill-check, inactivity
        penalties, and dribble-speed-penalty model as regular tackles.
        """
        carrier = self.ball_carrier()
        if carrier is None or carrier.is_inactive:
            return

        min_charge = self.tackling_params.head_on_min_charge_speed_mps

        for other in self.players:
            if other.player_id == carrier.player_id:
                continue
            if other.is_inactive or other.team == carrier.team:
                continue
            if not are_touching(carrier, other):
                continue

            # Direction from carrier to other player.
            delta = other.position.xy() - carrier.position.xy()
            dist = delta.length()
            if dist < 1e-9:
                continue
            direction = delta / dist  # unit vector carrier -> other

            # Both must be charging towards each other along this axis.
            carrier_speed_towards = carrier.velocity.xy().dot(direction)
            other_speed_towards = -(other.velocity.xy().dot(direction))

            if carrier_speed_towards < min_charge or other_speed_towards < min_charge:
                continue

            # Head-on: resolve as a tackle (other player is the tackler).
            if self._gk_immune_from_tackle(carrier):
                # Phase B: GK in own box with ball is untackleable —
                # the onrushing player is penalised; carrier is untouched.
                other.state = PlayerState.INACTIVE_TACKLED
                other.state_timer_s = self.tackling_params.tackler_miss_inactive_duration_s
                self._log_info(f"{other.player_id} head-on tackle on {carrier.player_id} auto-failed [GK in own box]")
                return
            result = attempt_tackle(
                other.attributes.tackling,
                carrier.attributes.dribbling,
                self.rng_reduction,
                self.rng,
                self.tackling_params,
                is_goalkeeper_tackle=other.is_goalkeeper,
                angle_modifier=tackle_angle_modifier(
                    carrier.heading_rad, carrier.position, other.position, self.tackling_params
                ),
                gk_outside_box=self._gk_outside_own_box(other),  # Phase B: GK penalty
            )
            self._log_tackle_result(other.player_id, carrier.player_id, result, "head-on")
            if result.tackler_won:
                self.ball.possessed_by = other.player_id
            else:
                carrier.velocity = carrier.velocity * result.dribble_speed_multiplier

            carrier.state = PlayerState.INACTIVE_TACKLED
            carrier.state_timer_s = self.tackling_params.inactive_duration_s
            if not result.tackler_won:
                other.state = PlayerState.INACTIVE_TACKLED
                other.state_timer_s = self.tackling_params.tackler_miss_inactive_duration_s
            return  # only one head-on tackle per tick

    def _complete_control(self, player: Player) -> None:
        self.ball.possessed_by = player.player_id
        self.ball.velocity = Vector3.zero()
        self._log_info(f"{player.player_id} completed first touch and took possession")

    def _log_tackle_result(
        self,
        tackler_id: str,
        dribbler_id: str,
        result: "TackleResult",  # type: ignore[name-defined]  # noqa: F821
        modifier_notes: str = "",
    ) -> None:
        """Emit tackle outcome to log_callback (if set)."""
        if self.log_callback is None:
            return
        outcome = "tackled" if result.tackler_won else "failed tackle on"
        self._log_info(f"{tackler_id} {outcome} {dribbler_id}")
        notes = f"  tackler_roll={result.tackler_roll:.3f}  dribbler_roll={result.dribbler_roll:.3f}"
        if modifier_notes:
            notes += f"  [{modifier_notes}]"
        self._log_debug(notes)

    # ── Phase B helpers ────────────────────────────────────────────────────

    def _defends_left(self, player: Player) -> bool:
        """True if this player's team defends the left goal (Team.LEFT)."""
        return player.team == Team.LEFT

    def _gk_immune_from_tackle(self, gk_candidate: Player) -> bool:
        """True when a goalkeeper is in possession inside their own penalty
        box — in that case all tackle attempts auto-fail (GK is legally
        handling the ball in a protected area)."""
        return (
            gk_candidate.is_goalkeeper
            and self.ball.possessed_by == gk_candidate.player_id
            and self.pitch.is_in_box(
                gk_candidate.position, left=self._defends_left(gk_candidate)
            )
        )

    def _gk_outside_own_box(self, tackler: Player) -> bool:
        """True when the tackler is a goalkeeper making a tackle from outside
        their own penalty box (triggers the -40% tackle boost penalty)."""
        return (
            tackler.is_goalkeeper
            and not self.pitch.is_in_box(
                tackler.position, left=self._defends_left(tackler)
            )
        )

    def _effective_dribbling(self, target: Player) -> float:
        """Returns the target's dribbling attribute, with up to 25% penalty
        if they are mid first-touch control (CONTROLLING_BALL state).

        penalty_frac = min(1, state_timer_s / control_time_penalty_reference_s)
        dribbling_eff = dribbling_attr * (1 - 0.25 * penalty_frac)

        A player who just picked the ball up (near max timer) has the full
        25% penalty; one about to finish control (near 0 s remaining) has
        essentially no penalty.
        """
        if target.state != PlayerState.CONTROLLING_BALL:
            return target.attributes.dribbling
        ref_s = self.tackling_params.control_time_penalty_reference_s
        penalty_frac = min(1.0, target.state_timer_s / ref_s)
        return target.attributes.dribbling * (1.0 - 0.25 * penalty_frac)

    def _target_has_or_controls_ball(self, target: Player) -> bool:
        """True if the target has possession OR is mid first-touch control
        (ball frozen at their feet). Used to decide ball transfer on a
        tackle win against a CONTROLLING_BALL player."""
        return (
            self.ball.possessed_by == target.player_id
            or target.state == PlayerState.CONTROLLING_BALL
        )

    def _check_goal(self) -> None:
        side = check_goal(self.ball, self.pitch)
        if side is not None:
            self.scoreboard.score_for(side)
            self._log_info(f"GOAL for {'LEFT' if side == 'left' else 'RIGHT'} — score {self.scoreboard.left_goals}:{self.scoreboard.right_goals}")
            if self.goal_linger_s > 0.0:
                self._goal_linger_remaining_s = self.goal_linger_s
            else:
                self._reset_after_goal()

    def _reset_after_goal(self) -> None:
        self.ball.position = Vector3.zero()
        self.ball.velocity = Vector3.zero()
        self.ball.spin = Vector3.zero()
        self.ball.possessed_by = None


def _drain_if_sprinting(params: MovementParams, player: Player, sprinting: bool, dt: float) -> float:
    from footballcoach.engine.movement import drain_stamina

    if not sprinting:
        return player.stamina
    return drain_stamina(params, player.stamina, player.attributes.stamina, 1.0, dt)


def _braking_speed_mode(
    dist: float,
    current_speed: float,
    arrival_speed: float,
    a_max: float,
    standstill_decel_mult: float,
    jog_speed: float,
    sprint_requested: bool,
) -> SpeedMode:
    """Picks the ``SpeedMode`` for this tick so the player arrives at
    ``arrival_speed`` (m/s) within ``dist`` metres without overshooting.

    Logic:
    - If ``arrival_speed >= jog_speed``, no braking is needed: return
      SPRINT or JOG per ``sprint_requested``.
    - If ``arrival_speed ≈ 0`` (< 0.1):
        * Always STANDSTILL within 0.5 m (close-range guard that prevents
          re-acceleration oscillation when braking_dist ≈ 0 at low speed).
        * Switch to STANDSTILL earlier when ``dist <= v²/(2·a_eff)``
          (deceleration physics: the distance needed to reach 0 from the
          current speed under the boosted standstill deceleration).
    - Otherwise switch to JOG within the corresponding braking distance.
    """
    if arrival_speed >= jog_speed - 0.05:
        return SpeedMode.SPRINT if sprint_requested else SpeedMode.JOG

    if arrival_speed < 0.1:
        # Always decelerate within 0.5 m to avoid low-speed oscillation.
        if dist <= 0.5:
            return SpeedMode.STANDSTILL
        a_eff = a_max * standstill_decel_mult
        braking_dist = (current_speed ** 2) / (2.0 * a_eff) if current_speed > 0.0 else 0.0
        if dist <= braking_dist:
            return SpeedMode.STANDSTILL
    else:
        # Mid-range arrival speed: decelerate to JOG within braking distance.
        v_sq_diff = max(0.0, current_speed ** 2 - arrival_speed ** 2)
        braking_dist = v_sq_diff / (2.0 * a_max) if a_max > 0.0 else 0.0
        if dist <= braking_dist:
            return SpeedMode.JOG

    return SpeedMode.SPRINT if sprint_requested else SpeedMode.JOG

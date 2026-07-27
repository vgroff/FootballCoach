"""Match: the top-level engine tying together movement, ball physics,
collision, kicking, tackling, possession, offside, and scoring into a single
steppable simulation. This is the main entry point other code (tests, a
future UI, a future RL training loop) should use.

See engine/knowledge.md for the overall tick order and design rationale.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from footballcoach.config import load_physics_config
from footballcoach.engine.ball_physics import BallPhysicsParams, step_ball
from footballcoach.engine.collision import (
    are_touching,
    resolve_all_overlaps,
    resolve_ball_block_by_inactive_players,
)
from footballcoach.engine.goalkeeping import GoalkeepingParams, save_target_position
from footballcoach.engine.kicking import KickingParams, PassingParams, kick_ball, pass_ball
from footballcoach.engine.movement import (
    MovementParams,
    effective_top_speed,
    regen_stamina,
    step_player_towards,
)
from footballcoach.engine.possession import ControlTimeParams, control_time_s
from footballcoach.engine.scoring import Scoreboard, check_goal
from footballcoach.engine.tackling import TacklingParams, attempt_tackle
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Player, PlayerState
from footballcoach.mathutils import Vector3
from footballcoach.orders import (
    ChaseTackleOrder,
    KickOrder,
    MoveOrder,
    OrderStatus,
    PassOrder,
    SaveOrder,
    TackleOrder,
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

        resolve_all_overlaps(self.players)
        self._check_goal()

        if self.ball.release_grace_s > 0.0:
            self.ball.release_grace_s = max(0.0, self.ball.release_grace_s - dt)
            if self.ball.release_grace_s == 0.0:
                self.ball.last_released_by = None

        self.time_s += dt

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
                if direction.length_xy() <= order.arrival_tolerance_m:
                    player.velocity = Vector3.zero()
                    order.status = OrderStatus.COMPLETE
                    player.current_order = None
                else:
                    step_player_towards(
                        player, direction, order.sprint, dt, self.movement_params, has_ball
                    )
                    player.stamina = _drain_if_sprinting(
                        self.movement_params, player, order.sprint, dt
                    )

            elif isinstance(order, KickOrder):
                if has_ball:
                    kick_ball(
                        self.ball,
                        player.position,
                        order.aim_point,
                        order.power_fraction,
                        player.attributes.kick_precision,
                        player.attributes.kick_power,
                        order.spin,
                        self.rng_reduction,
                        self.rng,
                        self.kicking_params,
                        kicker_velocity=player.velocity,
                        kicker_top_speed_mps=effective_top_speed(
                            self.movement_params, player.attributes.top_speed, player.stamina,
                            has_ball=True, ball_control_attr=player.attributes.ball_control,
                        ),
                    )
                    self._start_release_grace(player.player_id)
                order.status = OrderStatus.COMPLETE
                player.current_order = None

            elif isinstance(order, TackleOrder):
                target = self.player_by_id(order.target_player_id)
                if are_touching(player, target) and target.is_available_to_tackle():
                    success = attempt_tackle(
                        player.attributes.tackling,
                        target.attributes.dribbling,
                        self.rng_reduction,
                        self.rng,
                        self.tackling_params,
                    )
                    if success and self.ball.possessed_by == target.player_id:
                        self.ball.possessed_by = player.player_id
                    target.state = PlayerState.INACTIVE_TACKLED
                    target.state_timer_s = self.tackling_params.inactive_duration_s
                    if not success:
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
                    pass_ball(
                        self.ball,
                        player.position,
                        order.target_position,
                        player.attributes.kick_precision,
                        self.rng_reduction,
                        self.rng,
                        self.passing_params,
                        gravity_mps2=self.ball_physics_params.gravity_mps2,
                        rolling_friction_coefficient=self.ball_physics_params.rolling_friction_coefficient,
                        power_fraction=order.power_fraction,
                        running_power_coefficient=self.kicking_params.running_power_coefficient,
                        kicker_velocity=player.velocity,
                        kicker_top_speed_mps=effective_top_speed(
                            self.movement_params, player.attributes.top_speed, player.stamina,
                            has_ball=True, ball_control_attr=player.attributes.ball_control,
                        ),
                    )
                    self._start_release_grace(player.player_id)
                order.status = OrderStatus.COMPLETE
                player.current_order = None

            elif isinstance(order, ChaseTackleOrder):
                order.status = OrderStatus.IN_PROGRESS
                target = self.player_by_id(order.target_player_id)
                if are_touching(player, target):
                    if target.is_available_to_tackle():
                        success = attempt_tackle(
                            player.attributes.tackling,
                            target.attributes.dribbling,
                            self.rng_reduction,
                            self.rng,
                            self.tackling_params,
                        )
                        if success and self.ball.possessed_by == target.player_id:
                            self.ball.possessed_by = player.player_id
                        target.state = PlayerState.INACTIVE_TACKLED
                        target.state_timer_s = self.tackling_params.inactive_duration_s
                        if not success:
                            player.state = PlayerState.INACTIVE_TACKLED
                            player.state_timer_s = self.tackling_params.tackler_miss_inactive_duration_s
                    order.status = OrderStatus.COMPLETE
                    player.current_order = None
                else:
                    direction = target.position - player.position
                    step_player_towards(player, direction, True, dt, self.movement_params, has_ball)
                    player.stamina = _drain_if_sprinting(self.movement_params, player, True, dt)

            elif isinstance(order, SaveOrder):
                order.status = OrderStatus.IN_PROGRESS
                if not player.is_goalkeeper:
                    # SaveOrder is goalkeeper-only; silently no-op for an
                    # outfield player rather than raising, since orders are
                    # not otherwise role-restricted.
                    order.status = OrderStatus.COMPLETE
                    player.current_order = None
                else:
                    target_position = save_target_position(
                        self.pitch,
                        player.team,
                        self.ball.position,
                        self.ball.velocity,
                        self.ball_physics_params.gravity_mps2,
                        self.goalkeeping_params,
                    )
                    direction = target_position - player.position
                    if direction.length_xy() < 0.15:
                        # Snap to the target (not just freeze velocity):
                        # without this, a keeper "arriving" anywhere within
                        # the 0.15m tolerance ring stays frozen at that
                        # residual offset for the rest of the shot's
                        # flight - usually harmless, but a keeper boosted by
                        # goalkeeper_accel_multiplier reaches (and freezes
                        # at) that ring well before the ball arrives, and
                        # the leftover gap could be just enough to sit
                        # outside pickup_radius_m, turning a correctly-read
                        # save into a miss (see
                        # tests/balance/test_save_balance.py's fast-vs-slow
                        # regression).
                        player.position = target_position.with_z(player.position.z)
                        player.velocity = Vector3.zero()
                    else:
                        step_player_towards(player, direction, True, dt, self.movement_params, has_ball)
                        player.stamina = _drain_if_sprinting(self.movement_params, player, True, dt)
                    # Never auto-completes - a goalkeeper is always "on
                    # duty" reacting to the ball; see orders.SaveOrder.

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

    def _complete_control(self, player: Player) -> None:
        self.ball.possessed_by = player.player_id
        self.ball.velocity = Vector3.zero()

    def _check_goal(self) -> None:
        side = check_goal(self.ball, self.pitch)
        if side is not None:
            self.scoreboard.score_for(side)
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

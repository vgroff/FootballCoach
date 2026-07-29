"""Match: the top-level engine tying together movement, ball physics,
collision, kicking, tackling, possession, offside, and scoring into a single
steppable simulation. This is the main entry point other code (tests, a
future UI, a future RL training loop) should use.

See engine/knowledge.md for the overall tick order and design rationale.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from footballcoach.ui.gamelog import LogLevel

from footballcoach.config import load_physics_config
from footballcoach.engine.ball_physics import BallPhysicsParams, step_ball, resolve_goal_boundary
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
    angle_diff,
    effective_acceleration,
    effective_top_speed,
    regen_stamina,
    step_player_towards,
)
from footballcoach.engine.possession import ControlTimeParams, control_time_s
from footballcoach.engine.scoring import Scoreboard, check_goal
from footballcoach.engine.tackling import TacklingParams, apply_tackle_result, attempt_tackle, tackle_angle_modifier
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
)
from footballcoach.steering import RepulsionParams, compute_repulsion

import logging
log = logging.getLogger("footballcoach.match")

# Brake-to-turn thresholds (order/AI layer only — no physics changes).
# If a MoveOrder requires a heading change larger than this while the player
# is moving above the minimum speed, downgrade to STANDSTILL for that tick so
# the player decelerates quickly (standstill_decel_multiplier applies) and
# pivot rate rises as speed drops (ω_max = a_lat / speed).
_BRAKE_TO_TURN_THRESHOLD_RAD: float = 1.57   # ~90° — prevents orbiting around targets
_BRAKE_TO_TURN_MIN_SPEED_MPS: float = 2.0    # don't bother below this; turn rate already high
# When within this radius of the target and the velocity vector is not pointing
# at it (cosine similarity < threshold), brake to a stop then jog in normally.
# Catches lateral overshoots / orbiting without stopping on every normal approach.
_CLOSE_PROXIMITY_BRAKE_M: float = 5.0
_CLOSE_PROXIMITY_COS_THRESHOLD: float = 0.01

# ShootOrder blocker detection: perpendicular-distance threshold from the shot
# line within which an opposition player is considered to be blocking the shot.
_SHOT_BLOCKER_THRESHOLD_M: float = 1.0
# How far the shooter moves toward the goal when pausing due to a blocker.
_SHOT_PAUSE_ADVANCE_M: float = 2.0


def _has_blocker_on_shot_line(
    shooter: "Player",
    aim_point: "Vector3",
    opposition: "list[Player]",
    threshold_m: float = _SHOT_BLOCKER_THRESHOLD_M,
) -> bool:
    """Return ``True`` if any active opposition player lies within *threshold_m*
    of the line segment from *shooter.position* to *aim_point* (XY plane only)
    and is between the shooter and the aim point (not behind either end).

    Inactive (tackled) players are excluded — they cannot intercept a shot.
    """
    sx, sy = shooter.position.x, shooter.position.y
    ax, ay = aim_point.x, aim_point.y
    dx, dy = ax - sx, ay - sy
    line_len_sq = dx * dx + dy * dy
    if line_len_sq < 1e-12:
        return False
    for opp in opposition:
        if opp.state == PlayerState.INACTIVE_TACKLED:
            continue
        ox, oy = opp.position.x, opp.position.y
        # Scalar projection onto the shot line (0 = shooter, 1 = aim point).
        t = ((ox - sx) * dx + (oy - sy) * dy) / line_len_sq
        if t <= 0.0 or t >= 1.0:
            continue  # not between shooter and aim point
        proj_x = sx + t * dx
        proj_y = sy + t * dy
        if math.hypot(ox - proj_x, oy - proj_y) < threshold_m:
            return True
    return False


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
        self._apply_movement(dt)
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
            resolve_goal_boundary(self.ball, self.pitch, self.ball_physics_params)

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

    def _complete_order(self, order) -> None:
        """Transition *order* to COMPLETE and fire its on_complete callback if set."""
        order.status = OrderStatus.COMPLETE
        if order.on_complete is not None:
            order.on_complete()

    def _apply_gk_immune_penalty(self, player: Player) -> None:
        """Apply the auto-fail penalty to a tackler who charged into a GK protected in their own box.
        The GK is untouched; only the tackler is penalised.
        """
        player.velocity = player.velocity * self.tackling_params.tackle_attempt_tackler_speed_mult
        player.state = PlayerState.INACTIVE_TACKLED
        player.state_timer_s = self.tackling_params.tackle_cooldown_s

    def _apply_movement(self, dt: float) -> None:
        """Apply deferred movement intent set by orders/AI during _process_orders.

        Orders and AI set ``player.desired_direction`` and ``player.desired_speed_mode`` each tick.
        This is the ONLY place ``step_player_towards`` and stamina drain are called for locomotion.
        Players with ``desired_speed_mode=None`` had no movement intent this tick — their velocity
        is left unchanged (physics inertia coasts them forward).
        ``desired_speed_mode`` is cleared to ``None`` after application so each tick is independent.
        """
        for player in self.players:
            if player.desired_speed_mode is None:
                continue
            has_ball = self.ball.possessed_by == player.player_id
            speed_mode = player.desired_speed_mode
            step_player_towards(player, player.desired_direction, speed_mode, dt, self.movement_params, has_ball)
            player.stamina = _drain_if_sprinting(self.movement_params, player, speed_mode is SpeedMode.SPRINT, dt)
            player.desired_speed_mode = None  # consumed; reset for next tick

    def _set_possession(self, player_id: str | None) -> None:
        """Single write-path for ball possession.

        Every place that changes ``ball.possessed_by`` must go through here so
        that ``on_possession_gained`` callbacks fire reliably.  Direct writes to
        ``self.ball.possessed_by`` outside this method are forbidden.
        """
        old = self.ball.possessed_by
        self.ball.possessed_by = player_id
        if player_id is not None and player_id != old:
            p = self.player_by_id(player_id)
            if p.on_possession_gained is not None:
                p.on_possession_gained(p)

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
            if player.state == PlayerState.CONTROLLING_BALL:
                continue
            if player.ai is not None:
                player.ai.act(player, self, 0)
            order = player.current_order
            if order is None:
                continue
            if order.execute(player, self, dt):
                self._complete_order(order)
                player.current_order = None

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

        order = player.current_order
        sprint = getattr(order, 'sprint', True) if order is not None else True
        speed_mode = SpeedMode.SPRINT if sprint else SpeedMode.JOG

        if carrier is not None and carrier.player_id != player.player_id:
            # Someone else has the ball — chase and tackle.
            if are_touching(player, carrier):
                if carrier.is_available_to_tackle():
                    if self._gk_immune_from_tackle(carrier):
                        # Phase B: GK in own box with ball is untackleable.
                        player.velocity = player.velocity * self.tackling_params.tackle_attempt_tackler_speed_mult
                        player.state = PlayerState.INACTIVE_TACKLED
                        player.state_timer_s = self.tackling_params.tackle_cooldown_s
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
                            self._set_possession(player.player_id)
                        apply_tackle_result(result, player, carrier, self.tackling_params)
                return True  # tackle attempted (or target not available) — terminal
            else:
                intercept = self._intercept_target(player, carrier.position, carrier.velocity)
                direction = intercept - player.position
                step_player_towards(player, direction, speed_mode, dt, self.movement_params, has_ball)
                player.stamina = _drain_if_sprinting(self.movement_params, player, sprint, dt)
                return False
        else:
            # Ball is loose — run to intercept; pickup via _update_loose_ball_pickup.
            intercept = self._intercept_target(player, self.ball.position, self.ball.velocity)
            direction = intercept - player.position
            if direction.length_xy() <= self.pickup_radius_m:
                return True  # at the ball — pickup will complete next tick
            step_player_towards(player, direction, speed_mode, dt, self.movement_params, has_ball)
            player.stamina = _drain_if_sprinting(self.movement_params, player, sprint, dt)
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
        """Returns the world position this player should sprint toward in order
        to intercept the target (ball or carrier) in the shortest possible time.

        Solves the quadratic: find t >= 0 such that the player (sprinting at
        their effective top speed v_p) can reach the point target_pos + v_b*t.

            |d + v_b*t|^2 = (v_p * t)^2   where d = target_pos - player_pos

        Expanding:
            (|v_b|^2 - v_p^2) * t^2 + 2*(d . v_b)*t + |d|^2 = 0

        If the discriminant is negative the player cannot catch the target
        at their current top speed (target escaping); fall back to the
        current target position (run toward where it is now).

        If ball speed is near-zero, the quadratic degenerates cleanly to
        t = |d| / v_p (simple sprint time), which is exactly right.

        Only xy components are used — height is irrelevant for interception runs.
        """
        import math as _math
        d = (target_pos - player.position).xy()
        vb = target_vel.xy()

        v_p = effective_top_speed(
            self.movement_params,
            player.attributes.top_speed,
            player.stamina,
            has_ball=False,
            is_goalkeeper=player.is_goalkeeper,
        )

        vb_sq = vb.dot(vb)
        vp_sq = v_p * v_p
        d_dot_vb = d.dot(vb)
        d_sq = d.dot(d)

        a = vb_sq - vp_sq
        b = 2.0 * d_dot_vb
        c = d_sq

        if abs(a) < 1e-6:
            # Ball and player at same speed: linear solution t = -c/b
            if abs(b) > 1e-6:
                t = -c / b
            else:
                t = 0.0
        else:
            discriminant = b * b - 4.0 * a * c
            if discriminant < 0.0:
                # Player cannot catch the target — run to current position.
                return target_pos.with_z(target_pos.z)
            sqrt_disc = _math.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)
            # Pick smallest non-negative root.
            t = None
            for candidate in (t1, t2):
                if candidate >= 0.0:
                    if t is None or candidate < t:
                        t = candidate
            if t is None:
                # Both roots negative — target already past us; go to current pos.
                return target_pos.with_z(target_pos.z)

        t = max(0.0, t)
        predicted_xy = target_pos + vb * t
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
                other.velocity = other.velocity * self.tackling_params.tackle_attempt_tackler_speed_mult
                other.state = PlayerState.INACTIVE_TACKLED
                other.state_timer_s = self.tackling_params.tackle_cooldown_s
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
                self._set_possession(other.player_id)
            apply_tackle_result(result, other, carrier, self.tackling_params)
            return  # only one head-on tackle per tick

    def _complete_control(self, player: Player) -> None:
        self._set_possession(player.player_id)
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
        self._set_possession(None)


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

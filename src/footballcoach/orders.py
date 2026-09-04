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

import math

from footballcoach.config import load_orders_config, require_section
from footballcoach.mathutils import Vector3

if TYPE_CHECKING:
    from footballcoach.engine.match import Match
    from footballcoach.entities.player import Player


class OrderStatus(Enum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETE = auto()


# ---------------------------------------------------------------------------
# Order-layer movement parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderLayerParams:
    """Brake-to-turn and close-proximity parameters for the order layer.

    These govern how orders translate high-level intent into SpeedMode
    requests. The neural network direct-drive path bypasses all of these.
    Loaded from config/orders.json["movement"].
    """
    brake_turn_angle_rad: float
    brake_min_speed_mps: float
    close_prox_radius_m: float
    close_prox_cos_threshold: float
    sprint_to_ball_clearance_margin: float
    intercept_overshoot_frac: float

    @staticmethod
    def from_config() -> "OrderLayerParams":
        d = require_section(load_orders_config(), "movement", "orders.json")
        return OrderLayerParams(
            brake_turn_angle_rad=math.radians(d.get("brake_turn_angle_deg", 75.0)),
            brake_min_speed_mps=d.get("brake_min_speed_mps", 2.0),
            close_prox_radius_m=d.get("close_prox_radius_m", 6.0),
            close_prox_cos_threshold=d.get("close_prox_cos_threshold", 0.3),
            sprint_to_ball_clearance_margin=d.get("sprint_to_ball_clearance_margin", 1.0),
            intercept_overshoot_frac=d.get("intercept_overshoot_frac", 0.0),
        )


def braking_speed_mode(
    dist: float,
    current_speed: float,
    arrival_speed: float,
    a_max: float,
    standstill_decel_mult: float,
    jog_speed: float,
    sprint_requested: bool,
):
    """Picks the SpeedMode for this tick so the player arrives at
    ``arrival_speed`` (m/s) within ``dist`` metres without overshooting.

    Order-layer logic (not engine physics): the neural network's direct-drive
    path bypasses this entirely and must learn braking behaviour itself.

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
    from footballcoach.engine.movement import SpeedMode

    if arrival_speed >= jog_speed - 0.05:
        return SpeedMode.SPRINT if sprint_requested else SpeedMode.JOG

    if arrival_speed < 0.1:
        if dist <= 0.5:
            return SpeedMode.STANDSTILL
        a_eff = a_max * standstill_decel_mult
        braking_dist = (current_speed ** 2) / (2.0 * a_eff) if current_speed > 0.0 else 0.0
        if dist <= braking_dist:
            return SpeedMode.STANDSTILL
    else:
        v_sq_diff = max(0.0, current_speed ** 2 - arrival_speed ** 2)
        braking_dist = v_sq_diff / (2.0 * a_max) if a_max > 0.0 else 0.0
        if dist <= braking_dist:
            return SpeedMode.JOG

    return SpeedMode.SPRINT if sprint_requested else SpeedMode.JOG


# ---------------------------------------------------------------------------
# Shared movement-intent helper
# ---------------------------------------------------------------------------


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
        SpeedMode, angle_diff,
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
            match.ball.possessed_by, match.repulsion_params, match.pitch,
        )
        if speed_mode is SpeedMode.SPRINT and speed_mult < 0.75:
            speed_mode = SpeedMode.JOG
    else:
        adj_dir = target_direction
        speed_mult = 1.0  # noqa: F841

    # ── Close-proximity lateral-overshoot brake ──────────────────────────────
    op = match.order_params
    if (arrival_dist is not None and arrival_dist <= op.close_prox_radius_m
            and player.speed_mps > op.brake_min_speed_mps):
        vel_xy = player.velocity.xy()
        if vel_xy.length() > 1e-9 and adj_dir.length() > 1e-9:
            if vel_xy.normalized().dot(adj_dir.normalized()) < op.close_prox_cos_threshold:
                speed_mode = SpeedMode.STANDSTILL

    # ── Brake-to-turn ────────────────────────────────────────────────────────
    if (use_brake_to_turn and speed_mode is not SpeedMode.STANDSTILL
            and adj_dir.length() > 1e-9 and player.speed_mps > op.brake_min_speed_mps):
        desired_heading = adj_dir.angle_xy()
        heading_error = abs(angle_diff(player.heading_rad, desired_heading))
        if heading_error > op.brake_turn_angle_rad:
            speed_mode = SpeedMode.STANDSTILL

    # Player.desired_direction enforces unit-xy-or-zero (see its own
    # setter's docstring) -- adj_dir up to this point has been carried
    # around as a raw, un-normalized direction (this function's own
    # target_direction param is explicitly documented as such), used only
    # for magnitude-invariant checks above (.length()>eps, .normalized(),
    # .angle_xy()), so normalizing only here, right before every caller
    # assigns it to player.desired_direction, is a pure no-op for actual
    # movement behaviour (step_player_towards() re-normalizes internally
    # regardless) but fixes the live desired_dir_x/y observation feature,
    # which is not magnitude-invariant to its consumers.
    adj_dir = adj_dir.xy().normalized()
    return adj_dir, speed_mode


def _continue_current_motion(
    player: "Player", match: "Match", *, sprint: bool = True,
) -> "tuple[Vector3, object]":
    """Movement intent for a tick where an order just completed an
    instantaneous action (kick/pass/shot/gained-possession) but there's no
    reason to stop -- continue in the direction the player is already
    facing/moving, at the given pace.

    Used instead of leaving ``desired_direction``/``desired_speed_mode``
    unset, which ``Match._apply_movement`` would otherwise interpret as pure
    inertial coasting (no acceleration/turning/stamina drain) -- a state the
    neural decode path (``apply_action_to_player``) can never produce, since
    it always emits a concrete SPRINT/JOG or an active STANDSTILL brake.
    """
    from footballcoach.engine.movement import SpeedMode

    velocity_xy = player.velocity.xy()
    direction = (
        velocity_xy if velocity_xy.length() > 1e-6
        else Vector3.from_angle_xy(player.heading_rad)
    )
    speed_mode = SpeedMode.SPRINT if sprint else SpeedMode.JOG
    # See _compute_movement_intent's identical normalize-before-return
    # comment -- direction here is the player's raw velocity (magnitude =
    # current speed_mps), not a unit vector; Player.desired_direction's
    # setter now enforces unit-xy-or-zero.
    direction = direction.xy().normalized()
    return direction, speed_mode


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
    from footballcoach.engine.movement import effective_acceleration, sprint_eta
    gk_accel = effective_acceleration(
        match.movement_params, player.attributes.acceleration, player.stamina,
        is_goalkeeper=True,
    )
    t_gk = sprint_eta(dist_to_save, player.speed_mps, gk_top_speed, gk_accel)
    return t_ball < t_gk * 2.0  # sprint if GK would otherwise be beaten


def _push_kick_params() -> dict:
    """Return the push_kick section from orders.json (cached by load_orders_config)."""
    return require_section(load_orders_config(), "push_kick", "orders.json")


def _push_kick_power_fraction(player: "Player", match: "Match", speed_factor: float) -> float:
    """Shared push-kick power calc, used both for an immediate kick (ball
    already in hand) and an ARMED kick (computed ahead of time for whenever
    contact happens) -- must stay identical between the two so an armed kick
    behaves exactly like a normal push-kick would have.

    Reference speed is the average of the player's CURRENT speed
    (player.speed_mps) and their theoretical max sprint speed (already
    stamina-adjusted). Using max speed alone overstates how fast the player
    will actually be moving by the time they reach the ball, especially
    early in an approach.

    ``spinup_speed_boost`` (orders.json's push_kick section, 2026-09-03)
    compensates for ball_physics.py's ground-contact spin-up cost: a
    push-kick is launched spin-free (see _try_push_kick's docstring for why
    it doesn't just impart matching spin instead), so it now pays a real,
    roughly-1/(1+ball_inertia_shell_factor) transition cost before it's
    genuinely rolling -- kicking harder up front compensates for that loss
    empirically, rather than avoiding it.
    """
    from footballcoach.engine.kicking import max_kick_speed_mps
    from footballcoach.engine.movement import effective_top_speed

    max_sprint_speed = effective_top_speed(
        match.movement_params, player.attributes.top_speed, player.stamina,
        has_ball=False,
    )
    reference_speed = (player.speed_mps + max_sprint_speed) / 2.0
    max_kick = max_kick_speed_mps(match.kicking_params, player.attributes.kick_power)
    spinup_speed_boost = _push_kick_params().get("spinup_speed_boost", 1.0)
    return min(1.0, reference_speed * speed_factor * spinup_speed_boost / max(max_kick, 0.1))


def _push_kick_is_clear(
    player: "Player",
    match: "Match",
    landing_point: "Vector3",
    dist_m: float,
    clearance_margin: float,
) -> bool:
    """Return True if no opponent would beat the carrier to landing_point.

    landing_point/dist_m are an ESTIMATE, not an exact prediction -- push-
    kicks are flat, direction-only kicks (see _try_push_kick), so there's no
    config-specified travel distance to check against any more. The order's
    real target_position (what the caller passes in) is the most reasonable
    proxy for "how far this kick might carry the ball" available without
    modelling ground-friction deceleration explicitly.
    """
    from footballcoach.engine.movement import (
        effective_acceleration, effective_top_speed, sprint_eta,
    )
    from footballcoach.entities.player import PlayerState

    self_v_top = max(
        effective_top_speed(
            match.movement_params, player.attributes.top_speed,
            player.stamina, has_ball=False,
        ),
        0.1,
    )
    self_accel = effective_acceleration(
        match.movement_params, player.attributes.acceleration, player.stamina,
    )
    self_eta = sprint_eta(dist_m, player.speed_mps, self_v_top, self_accel)

    for other in match.players:
        if other.player_id == player.player_id:
            continue
        if other.team == player.team:
            continue  # only opponents can steal the ball
        if other.state == PlayerState.INACTIVE_TACKLED:
            continue
        dist_other = (landing_point - other.position).length_xy()
        if dist_other > dist_m + 10.0:  # rough radius pre-filter
            continue
        their_v_top = max(
            effective_top_speed(
                match.movement_params, other.attributes.top_speed,
                other.stamina, has_ball=False,
            ),
            0.1,
        )
        their_accel = effective_acceleration(
            match.movement_params, other.attributes.acceleration, other.stamina,
        )
        # Opponents start from standstill (v0=0) — conservative: assumes they
        # react instantly and run the optimal line toward the landing spot.
        their_eta = sprint_eta(dist_other, 0.0, their_v_top, their_accel)
        if their_eta < self_eta * clearance_margin:
            return False  # opponent beats us there — don't kick
    return True


def _try_push_kick(
    player: "Player",
    match: "Match",
    target_position: "Vector3",
    push_kick_min_dist_m: float | None,
) -> tuple["Vector3", float] | None:
    """Check whether a push-kick toward target_position should fire right
    now -- shared by MoveOrder (ball already in hand, or chasing our own
    just-kicked loose ball) and GetPossessionOrder (chasing a loose ball
    we've never had, arming a first-touch redirect). Returns (direction_3d,
    power_fraction) if every gate passes (far enough from target_position,
    heading roughly toward it, no opponent would get there first), else
    None.

    power_fraction is returned ALREADY run-compensated (see
    compensate_power_for_run_mult) using the player's CURRENT velocity --
    callers must fire it via kick_with_direction(..., compensate_for_run=
    False) (the default), NOT re-compensate a second time. This is
    deliberate: for the has-ball/immediate-fire caller, arm time and fire
    time are the same instant, so this is exact. For the armed-for-later
    caller (GetPossessionOrder, or MoveOrder chasing its own just-kicked
    ball), fire happens whenever the ball is actually reachable -- usually
    the very same tick or the next one, so using arm-time velocity as the
    compensation estimate is a good approximation, not a stale one.
    Compensating fresh at fire time INSTEAD of at arm time (an earlier
    version of this did that) makes the recorded "armed intent" and the
    eventual "actual kick" carry two DIFFERENT power values for the exact
    same touch -- a real discontinuity for BC/replay consumers (bc.py's
    phase1_labels() reads kick_armed_power_fraction on approach ticks and
    last_kick_power_fraction on the fire tick; those must already agree,
    not just converge once the kick fires). Baking compensation in here,
    once, at the point the decision is made, removes that cliff entirely.

    Flat kick only -- no ballistic aim point / travel-distance parameter
    (see orders.json's push_kick section, and Player.kick_with_direction):
    direction alone plus speed_factor fully determines the kick; how far it
    actually rolls is real ground physics, not a config number. The
    min_dist_m gate is what stops repeated push-kicks from overshooting
    target_position as the player closes in -- there's no separate cap on
    the kick's own travel distance.

    Launched spin-free (2026-09-03: reverted an earlier attempt to give
    push-kicks matching rolling spin -- that broke the NN-replay-equivalence
    contract, since the neural network's own kick path is hardcoded
    spin-free and can't reproduce it; see agent_plans/physics_update.md
    section 8 for the full history). Compensated instead via
    `_push_kick_power_fraction`'s `spinup_speed_boost`.
    """
    from footballcoach.engine.movement import angle_diff, effective_top_speed
    from footballcoach.engine.kicking import running_power_multiplier, compensate_power_for_run_mult

    direction = target_position - player.position
    dist = direction.length_xy()
    if dist < 1e-6:
        return None
    pk = _push_kick_params()
    pk_min = push_kick_min_dist_m if push_kick_min_dist_m is not None else pk["min_dist_m"]
    if dist < pk_min:
        return None
    push_dir = direction.xy().normalized()
    kick_heading = push_dir.angle_xy()
    max_heading_err = math.radians(pk["max_heading_error_deg"])
    if abs(angle_diff(player.heading_rad, kick_heading)) > max_heading_err:
        return None
    if not _push_kick_is_clear(player, match, target_position, dist, pk["clearance_margin"]):
        return None
    direction_3d = Vector3(push_dir.x, push_dir.y, 0.0)
    power_fraction = _push_kick_power_fraction(player, match, pk["speed_factor"])
    top_speed = effective_top_speed(
        match.movement_params, player.attributes.top_speed, player.stamina,
        has_ball=True, ball_control_attr=player.attributes.ball_control,
    )
    run_mult = running_power_multiplier(
        match.kicking_params.running_power_coefficient, player.velocity, direction_3d, top_speed,
    )
    adjusted_power = compensate_power_for_run_mult(power_fraction, run_mult)
    return direction_3d, adjusted_power


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
    # Push-kick: if True and the player has the ball, kick it flat (no
    # ballistic loft -- see _try_push_kick) toward target_position and sprint
    # to it rather than dribbling. Only activates when the remaining distance
    # to target is >= push_kick_min_dist_m (don't kick near the destination --
    # this is also what stops repeated touches overshooting target_position,
    # since there's no separate travel-distance cap on the kick itself).
    # A clearance check prevents kicking if any opponent would reach
    # target_position before the carrier (within the clearance_margin ETA
    # window). Defaults loaded from orders.json["push_kick"] at first use.
    push_kick_enabled: bool = False
    push_kick_min_dist_m: float | None = None   # None → from config

    def execute(self, player: "Player", match: "Match", dt: float) -> bool:
        """Execute one tick of movement toward target_position.  Returns True when arrived."""
        from footballcoach.engine.movement import SpeedMode, effective_top_speed

        self.status = OrderStatus.IN_PROGRESS
        has_ball = match.ball.possessed_by == player.player_id
        direction = self.target_position - player.position
        dist = direction.length_xy()

        # Push-kick: when the player has the ball and is far enough from the
        # destination, kick ahead (flat, no ballistic loft -- see
        # _try_push_kick / Player.kick_with_direction) and sprint free
        # rather than dribbling.
        if has_ball and self.push_kick_enabled and direction.length_xy() > 1e-6:
            result = _try_push_kick(player, match, self.target_position, self.push_kick_min_dist_m)
            if result is not None:
                direction_3d, power_fraction = result
                # Deterministic -- no rng draw here (used to jitter
                # dist/speed by up to +/-15% via match.rng.uniform, an
                # order-layer draw the NN-replay shadow never reproduces; a
                # real source of replay divergence whenever it landed on the
                # same tick as another rng-consuming physics event like a
                # tackle roll -- see test_rules_ai_nn_replay_equivalence.py's
                # rng_state_before_kick_noise docstring for the general
                # problem class).
                # power_fraction is already run-compensated by _try_push_kick
                # -- compensate_for_run=False (the default) here, or this
                # would compensate a second time.
                player.kick_with_direction(match, direction_3d, power_fraction, Vector3.zero())
                match._log_debug(
                    f"{player.player_id} push-kick dir=({direction_3d.x:.2f},{direction_3d.y:.2f})"
                    f" power={power_fraction:.2f}"
                )
                # Immediately set movement intent: sprint free (no ball this tick).
                adj_dir, speed_mode = _compute_movement_intent(
                    player, direction, match,
                    sprint=True, arrival_dist=dist, arrival_speed=None,
                    use_repulsion=True, use_brake_to_turn=True,
                )
                player.desired_direction = adj_dir
                player.desired_speed_mode = speed_mode
                return False

        # Arm the NEXT touch while chasing our own just-kicked (now loose)
        # ball, so re-catching it redirects immediately via
        # Match._update_loose_ball_pickup's kick_armed branch -- true
        # one-touch, no CONTROLLING_BALL delay, no control_speed_multiplier
        # slowdown. Without this, arming was rules-AI-only (rules_ai.py's
        # since-removed _arm_box_kick) and a bare MoveOrder(push_kick_enabled
        # =True) -- no AI wrapper at all -- fell through to the slow
        # control-then-kick pickup on every touch after the first, silently
        # losing most of push-kick's speed advantage (confirmed:
        # test_push_kick_faster_over_40m regressed to SLOWER than plain
        # dribbling once the first touch stopped being artificially fast).
        # Shares _try_push_kick's gating with the has_ball branch above so an
        # armed touch produces exactly the kick a normal push-kick would
        # have taken here.
        if not has_ball and self.push_kick_enabled and direction.length_xy() > 1e-6:
            result = _try_push_kick(player, match, self.target_position, self.push_kick_min_dist_m)
            if result is not None:
                direction_3d, power_fraction = result
                player.kick_armed = True
                player.kick_armed_direction = direction_3d
                # Already run-compensated by _try_push_kick (using arm-time
                # velocity) -- fired later via kick_with_direction's default
                # compensate_for_run=False, so this value is used as-is, not
                # compensated a second time.
                player.kick_armed_power_fraction = power_fraction
                player.kick_armed_spin = Vector3.zero()

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
            # Arrived. Don't just stop dead -- continue at the requested
            # arrival pace (STANDSTILL if max_speed_on_arrival_mps=0.0 was
            # explicitly asked for; JOG/SPRINT otherwise) in the same
            # heading, via the same braking-curve computation the
            # in-progress branch below uses. Previously this returned
            # without setting intent at all, which the engine read as
            # inertial coasting -- unrepresentable by the NN decode path.
            adj_dir, speed_mode = _compute_movement_intent(
                player, direction, match,
                sprint=self.sprint, arrival_dist=dist, arrival_speed=arrival_speed,
                use_repulsion=True, use_brake_to_turn=True,
            )
            player.desired_direction = adj_dir
            player.desired_speed_mode = speed_mode
            return True
        elif self.reached_target and dist > self.arrival_tolerance_m:
            if self._overshoot_timer_s is None:
                self._overshoot_timer_s = self.overshoot_timeout_s
            self._overshoot_timer_s -= dt
            player.desired_direction = Vector3.zero()
            player.desired_speed_mode = SpeedMode.STANDSTILL
            if self._overshoot_timer_s <= 0.0:
                return True
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
        player.kick_direct(match, self.aim_point, self.power_fraction, self.spin, self.compensate_for_run)
        # Follow through -- keep moving rather than freeze after striking the ball.
        adj_dir, speed_mode = _continue_current_motion(player, match, sprint=True)
        player.desired_direction = adj_dir
        player.desired_speed_mode = speed_mode
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
    power_multiplier: float = 1.0  # scales auto-computed or explicit power_fraction
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
                power_multiplier=self.power_multiplier,
            )
            match._log_debug(f"{player.player_id} passed to {pass_target}")
            if player.on_kick is not None:
                player.on_kick(player)
        # Follow through (or, if the ball was already lost, just keep moving
        # rather than freeze) -- see _continue_current_motion.
        adj_dir, speed_mode = _continue_current_motion(player, match, sprint=True)
        player.desired_direction = adj_dir
        player.desired_speed_mode = speed_mode
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

        self.status = OrderStatus.IN_PROGRESS
        target = match.player_by_id(self.target_player_id)
        # Keep tracking the target's motion whether or not contact is made
        # this tick -- while touching/duelling, freezing mid-tackle is no
        # more correct than it is while still closing the distance.
        adj_dir, speed_mode = _compute_movement_intent(
            player, target.position - player.position, match,
            sprint=True, arrival_dist=None,
            use_repulsion=False, use_brake_to_turn=True,
        )
        player.desired_direction = adj_dir
        player.desired_speed_mode = speed_mode
        if are_touching(player, target):
            from footballcoach.engine.collision import can_tackle
            if can_tackle(player, target):
                match._attempt_tackle_contact(player, target)
            return True
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
            # SaveOrder should only ever be assigned to a goalkeeper by the
            # AI order-selection logic; reaching this means that invariant
            # was violated elsewhere, not a real match state -- fail loudly
            # rather than silently no-op'ing.
            raise AssertionError(
                f"SaveOrder assigned to non-goalkeeper player {player.player_id!r}"
            )
        if match._goal_linger_remaining_s > 0.0:
            # Goal just scored -- the save is moot and the ball's about to
            # reset; stand down rather than continuing toward a save point
            # that no longer means anything.
            player.desired_direction = Vector3.zero()
            player.desired_speed_mode = SpeedMode.STANDSTILL
            return True  # cancel
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
class JogOrder:
    """Jogs indefinitely in a fixed direction; never completes.

    Gives an otherwise order-less player continuous, physically real
    locomotion (proper accel/turn-rate/heading via step_player_towards)
    instead of coasting forever on stale initial velocity -- see
    Match._apply_movement's "no order this tick" branch, which advances
    position from velocity with no deceleration or heading update at all
    when desired_speed_mode is never set. Used by e.g. phase1_scenario's
    "immobile" opponent so it moves like a real (if simple) player rather
    than a frictionless drifting object.

    Deliberately NOT driven by player.ai -- assign this directly to
    player.current_order and leave player.ai as whatever it already was
    (None for the immobile-opponent case). Phase1RulesAI.decide() (and any
    other call site) uses `opponent.ai is None` as its "can this opponent
    ever move/contest the ball" signal; giving this opponent a real AI
    instead would silently break that check.
    """
    direction: Vector3
    status: OrderStatus = OrderStatus.PENDING
    on_complete: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def execute(self, player: "Player", match: "Match", dt: float) -> bool:
        from footballcoach.engine.movement import SpeedMode

        self.status = OrderStatus.IN_PROGRESS
        dir_xy = self.direction.xy()
        player.desired_direction = dir_xy.normalized() if dir_xy.length() > 1e-9 else Vector3.zero()
        player.desired_speed_mode = SpeedMode.JOG
        return False


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

    Optional push-kick: if push_kick_enabled and target_position are both
    set, and the ball is currently loose (never carried), a first-touch
    redirect kick toward target_position is armed once close enough --
    shares _try_push_kick's gating with MoveOrder, so the caller supplies
    the SAME target it would use for a MoveOrder box-run and gets identical
    kick behaviour whether the ball is already in hand or not. Skips
    CONTROLLING_BALL entirely on pickup (see Match._update_loose_ball_pickup's
    kick_armed branch) -- a true one-touch, no control-time delay, no
    control_speed_multiplier slowdown.
    """
    sprint: bool = True  # True = sprint to ball, False = jog
    target_position: Vector3 | None = None
    push_kick_enabled: bool = False
    push_kick_min_dist_m: float | None = None   # None → from config
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
                # An armed-kick pickup fires this callback from INSIDE
                # Match._update_loose_ball_pickup's kick_armed branch, which
                # calls _set_possession() (triggering this) and THEN
                # immediately releases the ball again via kick_with_direction
                # -- p.kick_armed is still True at this exact point (it's
                # only reset at the TOP of the FOLLOWING tick's
                # _process_orders, not right after use), so it reliably
                # distinguishes "this is a fleeting touch about to be
                # redirected" from "the player actually stopped to hold the
                # ball". Only the latter should complete this order.
                #
                # Without this check: this order self-completes the tick
                # after ANY momentary possession, including an armed
                # redirect. Phase1RulesAI.act() only reissues a fresh
                # GetPossessionOrder once it observes current_order is None
                # -- one tick AFTER this order already completed -- so on
                # that gap tick nothing re-arms the kick even though the
                # ball's still loose (often still right at the player's
                # feet if push_kick's speed_factor keeps it close). If the
                # ball happens to still be in pickup range that gap tick,
                # it falls through to a normal (slow, CONTROLLING_BALL)
                # pickup instead -- defeating the entire point of arming a
                # kick. Confirmed via direct trace: at speed_factor close to
                # 1.0 (ball barely outruns the player) this fires on
                # essentially every repossession attempt.
                if not p.kick_armed:
                    _order._possession_gained = True

            player.on_possession_gained = _on_possession
            self._callback_registered = True

        # Already have the ball (or callback just fired).
        if match.ball.possessed_by == player.player_id or self._possession_gained:
            player.on_possession_gained = None
            # Just won the ball mid-chase -- carry momentum into the attack
            # rather than freezing.
            adj_dir, speed_mode = _continue_current_motion(player, match, sprint=self.sprint)
            player.desired_direction = adj_dir
            player.desired_speed_mode = speed_mode
            return True

        # Arm a first-touch redirect toward target_position -- only while the
        # ball is loose (never carried by anyone). Chasing a carrier to
        # tackle them is a different behaviour entirely (see
        # _run_get_possession_behaviour below) and shouldn't arm a kick.
        if self.push_kick_enabled and self.target_position is not None and match.ball.possessed_by is None:
            result = _try_push_kick(player, match, self.target_position, self.push_kick_min_dist_m)
            if result is not None:
                direction_3d, power_fraction = result
                player.kick_armed = True
                player.kick_armed_direction = direction_3d
                # Already run-compensated by _try_push_kick (using arm-time
                # velocity) -- fired later via kick_with_direction's default
                # compensate_for_run=False, so this value is used as-is, not
                # compensated a second time.
                player.kick_armed_power_fraction = power_fraction
                player.kick_armed_spin = Vector3.zero()

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
            # Players are never removed mid-match in current scenarios, so a
            # mark target that no longer exists means an order-assignment
            # bug elsewhere -- fail loudly rather than silently cancelling.
            raise AssertionError(
                f"MarkOrder target {self.target_player_id!r} no longer exists in the match"
            ) from None
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
        from footballcoach.engine.kicking import (
            SHOT_BLOCKER_THRESHOLD_M,
            SHOT_PAUSE_ADVANCE_M,
            compensate_power_for_run_mult,
            has_blocker_on_shot_line,
            kick_ball,
            running_power_multiplier,
        )
        from footballcoach.engine.movement import SpeedMode, effective_top_speed

        if match.ball.possessed_by != player.player_id:
            # Lost the ball before shooting -- real transition, nothing to
            # react to; stand down explicitly rather than coasting.
            player.desired_direction = Vector3.zero()
            player.desired_speed_mode = SpeedMode.STANDSTILL
            return True  # no-op: lost ball

        opposition = [p for p in match.players if p.team != player.team]
        if (self.chance_of_pausing > 0.0
                and has_blocker_on_shot_line(player.position, self.aim_point, opposition, SHOT_BLOCKER_THRESHOLD_M)
                and match.rng.random() < self.chance_of_pausing):
            aim_dir = (self.aim_point - player.position).xy()
            aim_len = aim_dir.length_xy()
            if aim_len > 1e-9:
                step_dir = aim_dir / aim_len
                raw_target = player.position.xy() + step_dir * SHOT_PAUSE_ADVANCE_M
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
                # Run the newly-installed MoveOrder immediately this tick
                # instead of leaving intent unset for one tick while it
                # waits for the next _process_orders pass.
                if player.current_order.execute(player, match, dt):
                    match._complete_order(player.current_order)
                    player.current_order = None
                return False  # ShootOrder itself is superseded; caller must NOT touch player.current_order for it

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
        match._log_info(f"{player.player_id} shot at goal  power={self.power_fraction:.2f}")
        if player.on_kick is not None:
            player.on_kick(player)
        # Follow through -- keep moving rather than freeze after the shot.
        adj_dir, speed_mode = _continue_current_motion(player, match, sprint=True)
        player.desired_direction = adj_dir
        player.desired_speed_mode = speed_mode
        return True


Order = MoveOrder | KickOrder | ShootOrder | PassOrder | ChaseTackleOrder | SaveOrder | StopOrder | GetPossessionOrder | MarkOrder

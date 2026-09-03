"""Ball physics: gravity, drag, Magnus effect, ground bounce, and rolling
friction. See engine/knowledge.md for the full derivation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from footballcoach.config import load_physics_config, require_section
from footballcoach.entities.ball import Ball
from footballcoach.mathutils import Vector3

# Pitch imported lazily inside resolve_goal_boundary to avoid a circular
# dependency at module load time (entities.pitch → config → nothing that
# imports ball_physics, but keeping it lazy is defensive).
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from footballcoach.entities.pitch import Pitch


@dataclass(frozen=True)
class BallPhysicsParams:
    gravity_mps2: float
    air_density_kgpm3: float
    ball_radius_m: float
    ball_mass_kg: float
    drag_coefficient: float
    magnus_coefficient: float
    bounce_restitution_vertical: float
    # Tangential (horizontal) bounce behaviour is no longer a flat retention
    # constant -- it's derived per-bounce from a friction/spin-coupling
    # impulse model (see step_ball's ground-collision block and
    # engine/knowledge.md). ground_friction_coefficient is the Coulomb
    # friction coefficient for ball-ground contact -- used both for a
    # bounce's instantaneous impact (_resolve_bounce_friction) and for
    # ordinary ground contact spinning a ball up toward true
    # rolling-without-slipping (_resolve_ground_friction); physically the
    # same ball-on-grass friction in both cases. ball_inertia_shell_factor
    # is k in I = k*m*r^2 -- a ball that fully "grips" (bounce or ground
    # contact alike) retains exactly 1/(1+k) of its pre-slip horizontal
    # speed. 2/3 (0.667) is the idealized thin shell value (retention 0.6);
    # set to 0.5 (retention 0.667) as a deliberate softening, still short
    # of 2/5 = 0.4 (a solid sphere, which the ball isn't) -- see
    # physics.json's _comment_bounce_friction.
    ground_friction_coefficient: float
    ball_inertia_shell_factor: float
    # Applies only to the vertical-axis (spin.z) component on a bounce now --
    # the horizontal-axis (topspin/backspin/sidespin) components are instead
    # recomputed by the friction/inertia coupling above, which has its own
    # implicit "retention" behaviour that depends on incident spin.
    bounce_spin_retention: float
    rolling_friction_coefficient: float
    spin_decay_per_s: float
    block_restitution: float
    goal_net_restitution: float = 0.15
    just_bounced_display_duration_s: float = 0.3
    bounce_threshold_mps: float = 0.5  # min incoming vertical speed to count as a real bounce vs. rest/rolling

    @staticmethod
    def from_config() -> "BallPhysicsParams":
        cfg = load_physics_config()
        world = require_section(cfg, "world")
        ball = require_section(cfg, "ball")
        bp = require_section(cfg, "ball_physics")
        return BallPhysicsParams(
            gravity_mps2=world["gravity_mps2"],
            air_density_kgpm3=world["air_density_kgpm3"],
            ball_radius_m=ball["radius_m"],
            ball_mass_kg=ball["mass_kg"],
            drag_coefficient=bp["drag_coefficient"],
            magnus_coefficient=bp["magnus_coefficient"],
            bounce_restitution_vertical=bp["bounce_restitution_vertical"],
            ground_friction_coefficient=bp.get("ground_friction_coefficient", bp.get("bounce_friction_coefficient", 0.4)),
            ball_inertia_shell_factor=bp.get("ball_inertia_shell_factor", 0.5),
            bounce_spin_retention=bp["bounce_spin_retention"],
            rolling_friction_coefficient=bp["rolling_friction_coefficient"],
            spin_decay_per_s=bp["spin_decay_per_s"],
            block_restitution=bp["block_restitution"],
            goal_net_restitution=bp.get("goal_net_restitution", 0.15),
            just_bounced_display_duration_s=bp.get("just_bounced_display_duration_s", 0.3),
            bounce_threshold_mps=bp.get("bounce_threshold_mps", 0.5),
        )


def _cross_section_area(radius_m: float) -> float:
    return math.pi * radius_m * radius_m


def _apply_ground_friction_impulse(
    velocity: Vector3,
    spin: Vector3,
    normal_impulse: float,
    params: BallPhysicsParams,
) -> tuple[Vector3, Vector3]:
    """Shared Coulomb-friction slip/grip physics for ball-ground contact,
    coupling horizontal velocity and (horizontal-axis) spin change through
    the ball's moment of inertia. Used by both a bounce's instantaneous
    impact (_resolve_bounce_friction, normal_impulse = m*|v_z,in|*(1+e_v))
    and one tick's continuous rolling contact (_resolve_ground_friction,
    normal_impulse = m*g*dt_s) -- physically the same ball-on-grass
    friction coefficient in both cases, just delivered as an impulsive hit
    in one case and spread continuously over time in the other.

    Physics (standard rigid-sphere friction-contact model, e.g. Cross's
    football-bounce work -- see engine/knowledge.md): the ball's contact
    point (directly below its centre) has a horizontal velocity relative to
    the ground of ``slip = v_xy - r * (spin x z_hat)``, i.e.
    ``(vx - r*spin.y, vy + r*spin.x)`` (spin.x/spin.y are the horizontal-
    axis, topspin/backspin/sidespin components -- see
    ui/kick_trajectory.py's convention; spin.z, rotation about the vertical
    axis, doesn't couple to ground contact in this planar model).

    A Coulomb friction impulse opposes this slip, capped at
    ``mu * normal_impulse``. If that cap is enough to fully cancel the slip
    (the ball "grips" -- confirmed as the common real-world case for
    bounces by Cross's force-plate measurements), only the smaller impulse
    needed to reach zero slip is applied; otherwise the capped impulse
    applies throughout. Either way, the same impulse simultaneously reduces
    horizontal speed and changes spin (a ball with strong topspin needs
    little/no friction impulse to already be near-rolling, so it loses
    little horizontal speed; a ball with no spin or backspin needs a much
    larger impulse, losing considerably more speed -- reproducing, as an
    emergent result of the physical model rather than a hand-tuned
    constant, the qualitative finding "topspin -> small change in
    horizontal speed, no spin/backspin -> large reduction" from the
    literature this model is based on). Only x/y components of velocity and
    spin are changed -- callers set z themselves.
    """
    r = params.ball_radius_m
    m = params.ball_mass_kg
    k = params.ball_inertia_shell_factor  # I = k * m * r^2

    slip_x = velocity.x - r * spin.y
    slip_y = velocity.y + r * spin.x
    slip_speed = math.hypot(slip_x, slip_y)

    if slip_speed < 1e-9:
        # Already rolling without slipping -- no friction impulse needed or
        # possible (e.g. the extreme "topspin, minimal loss" bounce case).
        return velocity, spin

    sx, sy = slip_x / slip_speed, slip_y / slip_speed

    max_friction_impulse = params.ground_friction_coefficient * normal_impulse

    # Reduced-mass relationship between tangential impulse and slip
    # reduction for a sphere: d|slip|/dJ_f = (1/m) * (1 + 1/k) (derived from
    # combining the linear (1/m) and rotational (r^2/I = 1/(k*m)) responses
    # to the same contact-point impulse). Solving for the impulse that
    # drives slip to exactly zero:
    slip_reduction_per_unit_impulse = (1.0 / m) * (1.0 + 1.0 / k)
    friction_impulse_to_grip = slip_speed / slip_reduction_per_unit_impulse

    friction_impulse = min(max_friction_impulse, friction_impulse_to_grip)

    delta_v_h = friction_impulse / m
    new_vx = velocity.x - delta_v_h * sx
    new_vy = velocity.y - delta_v_h * sy

    # Angular impulse from the same friction force, applied at the contact
    # point (moment arm r): delta_omega = r * J_f / I, directed along
    # z_hat x s_hat = (-sy, sx, 0) (see docstring derivation in
    # engine/knowledge.md for the full cross-product working).
    moment_of_inertia = k * m * r * r
    delta_omega = r * friction_impulse / moment_of_inertia
    new_spin_x = spin.x - delta_omega * sy
    new_spin_y = spin.y + delta_omega * sx

    return Vector3(new_vx, new_vy, velocity.z), Vector3(new_spin_x, new_spin_y, spin.z)


def _resolve_bounce_friction(
    velocity: Vector3,
    spin: Vector3,
    outgoing_vz: float,
    params: BallPhysicsParams,
) -> tuple[Vector3, Vector3]:
    """Applies a single Coulomb-friction impulse at the ground-contact point
    of a real bounce (see _apply_ground_friction_impulse), replacing the old
    flat bounce_restitution_horizontal/bounce_spin_retention constants,
    which treated every bounce identically regardless of incident spin.
    """
    # Normal impulse delivered by the ground this bounce (from the vertical
    # velocity change already computed by the caller via
    # bounce_restitution_vertical): J_n = m * |v_z,in| * (1 + e_v).
    incoming_vz_mag = -velocity.z  # velocity.z is negative (moving down) here
    normal_impulse = params.ball_mass_kg * incoming_vz_mag * (1.0 + params.bounce_restitution_vertical)

    new_velocity, new_spin = _apply_ground_friction_impulse(velocity, spin, normal_impulse, params)
    # spin.z (vertical-axis spin) doesn't couple to this planar model --
    # keeps its own flat decay on a bounce, unlike ordinary ground contact.
    new_spin_z = spin.z * params.bounce_spin_retention
    return new_velocity.with_z(outgoing_vz), Vector3(new_spin.x, new_spin.y, new_spin_z)


def _resolve_ground_friction(
    velocity: Vector3,
    spin: Vector3,
    dt_s: float,
    params: BallPhysicsParams,
) -> tuple[Vector3, Vector3]:
    """Applies one tick's worth of Coulomb ground-contact friction to a
    grounded ball that isn't yet rolling without slipping (spin doesn't
    match v = r*omega at the contact point) -- e.g. a ball launched or
    kicked with less spin than true rolling requires, or none at all.

    A real ball transitioning from sliding to rolling pays a real physical
    cost: friction has to spin it up before it's truly rolling, over and
    above the much gentler ongoing rolling resistance
    (rolling_friction_coefficient) that applies once it is. This is exactly
    the classic "sliding-to-rolling" mechanics problem (the same result
    behind the textbook "5/7 v0" answer for a solid sphere): a ball that
    fully grips ends up retaining exactly 1/(1+k) of its pre-slip speed,
    regardless of the friction coefficient (which only controls how long
    the transition takes, not how much speed is lost reaching it). Uses the
    same _apply_ground_friction_impulse physics as a bounce, just with a
    continuous normal impulse (m*g*dt_s) instead of an instantaneous one.

    Once slip reaches zero (true rolling), this is a no-op and the flat
    rolling_friction_coefficient deceleration (applied separately by the
    caller, step_ball) is the only remaining horizontal decelerant --
    matching real rolling resistance, which is a much gentler (mostly
    elastic-deformation) effect than the kinetic sliding friction that
    dominates before the ball is truly rolling.
    """
    normal_impulse = params.ball_mass_kg * params.gravity_mps2 * dt_s
    return _apply_ground_friction_impulse(velocity, spin, normal_impulse, params)


def step_ball(ball: Ball, dt_s: float, params: BallPhysicsParams | None = None) -> None:
    """Advances the ball's position/velocity/spin by one physics tick.

    Only applies free-flight physics (gravity, drag, Magnus, ground
    bounce/roll). Does not handle possession "stuck to player" logic or
    kicks - those are handled by possession.py / kicking.py which set
    ball.velocity directly before/after calling this.
    """
    params = params or BallPhysicsParams.from_config()

    if ball.possessed_by is not None:
        return  # ball is glued to a player; possession.py handles that motion.

    area = _cross_section_area(params.ball_radius_m)

    # Gravity.
    gravity_force = Vector3(0.0, 0.0, -params.gravity_mps2 * params.ball_mass_kg)

    # Aerodynamic drag: F = -0.5 * rho * Cd * A * |v| * v
    speed = ball.velocity.length()
    if speed > 1e-9:
        drag_mag = 0.5 * params.air_density_kgpm3 * params.drag_coefficient * area * speed
        drag_force = ball.velocity.normalized() * (-drag_mag * speed)
    else:
        drag_force = Vector3.zero()

    # Magnus effect: F = rho * A * r * C_L * (omega x v)
    magnus_force = ball.spin.cross(ball.velocity) * (
        params.air_density_kgpm3 * area * params.ball_radius_m * params.magnus_coefficient
    )

    total_force = gravity_force + drag_force + magnus_force
    acceleration = total_force / params.ball_mass_kg

    new_velocity = ball.velocity + acceleration * dt_s
    new_position = ball.position + ball.velocity * dt_s + acceleration * (0.5 * dt_s * dt_s)

    # Spin decays exponentially over time (air resistance on rotation).
    spin_decay_factor = max(0.0, 1.0 - params.spin_decay_per_s * dt_s)
    new_spin = ball.spin * spin_decay_factor

    # Ground collision / bounce.
    #
    # IMPORTANT: only treat this as a genuine "bounce" if the ball was
    # actually AIRBORNE at the start of this tick. A resting/rolling ball on
    # the ground has gravity pull its *next* position slightly below
    # ball_radius_m every single tick, giving new_velocity.z a small negative
    # value purely from that one tick's gravity integration - not a real
    # bounce. This value scales with dt_s (it's ~g*dt_s), so a purely
    # velocity-threshold-based check (BOUNCE_THRESHOLD_MPS alone) is only
    # safe below dt_s ~= threshold/(g*e_v) -- comfortably true at the UI's
    # 30Hz tick, but only a ~12% margin at training's 16.7Hz (sim_dt_s=0.06)
    # tick, and actively broken above ~14.7Hz. Gating on the ball's
    # was_grounded_before_tick state (set BEFORE this tick's integration,
    # from last tick's resolved position) removes that dt-dependence
    # entirely: an already-grounded ball can never misfire as a "bounce" no
    # matter how coarse dt_s is, since the classification no longer depends
    # on a single tick's gravity-integration artifact at all. A ball that
    # WAS airborne this tick and is only now reaching the ground carries
    # real, multi-tick-accumulated fall velocity, so BOUNCE_THRESHOLD_MPS
    # remains meaningful (and dt-independent) for that case. See
    # engine/knowledge.md's "ground contact" note -- an earlier version of
    # this code (before BOUNCE_THRESHOLD_MPS existed at all) misread every
    # resting tick as a bounce, and a coarse-enough dt_s could silently
    # reintroduce exactly that bug through the velocity-threshold check
    # alone; the was_grounded_before_tick gate closes that regardless of
    # dt_s.
    was_grounded_before_tick = ball.is_grounded()
    real_bounce_this_tick = False
    if new_position.z <= params.ball_radius_m:
        new_position = new_position.with_z(params.ball_radius_m)
        if not was_grounded_before_tick and new_velocity.z < -params.bounce_threshold_mps:
            outgoing_vz = -new_velocity.z * params.bounce_restitution_vertical
            if outgoing_vz < params.bounce_threshold_mps:
                # The bounce would produce a smaller upward vz than the threshold.
                # Continuing to bounce would create a perpetual micro-bounce loop
                # (the ball never settles because restitution keeps it airborne by
                # a tiny amount each tick). Treat as grounded instead: zero vz and
                # let rolling friction take over.
                new_velocity = new_velocity.with_z(0.0)
            else:
                new_velocity, new_spin = _resolve_bounce_friction(
                    new_velocity, new_spin, outgoing_vz, params,
                )
                real_bounce_this_tick = True
        else:
            # Resting/rolling contact, not a real bounce: kill the (small,
            # spurious) vertical velocity, then spend one tick's worth of
            # ground friction spinning the ball up toward true
            # rolling-without-slipping if it isn't there yet (see
            # _resolve_ground_friction). Once it is, this is a no-op and
            # rolling friction (below) is solely responsible for
            # decelerating a grounded ball.
            new_velocity = new_velocity.with_z(0.0)
            new_velocity, new_spin = _resolve_ground_friction(new_velocity, new_spin, dt_s, params)

        # Rolling friction (the gentle, ongoing decay) only applies to a
        # ball that's ALREADY rolling without slipping -- if there's still
        # meaningful slip at the contact point (either the resting/rolling
        # branch's kinetic friction was capped and hasn't fully corrected
        # it yet, or this was a "slide" bounce that never gripped), kinetic
        # ground friction is the dominant resistance and already accounted
        # for it above; applying rolling friction on top would double-count
        # it. Checked here directly (not via a per-branch flag) so it's
        # driven by the actual resulting slip state, from either branch.
        r = params.ball_radius_m
        remaining_slip = math.hypot(
            new_velocity.x - r * new_spin.y, new_velocity.y + r * new_spin.x
        )
        horiz_speed = new_velocity.length_xy()
        if remaining_slip < 1e-3 and horiz_speed > 1e-9:
            friction_decel = params.rolling_friction_coefficient * params.gravity_mps2
            reduced_speed = max(0.0, horiz_speed - friction_decel * dt_s)
            scale = reduced_speed / horiz_speed
            new_velocity = Vector3(new_velocity.x * scale, new_velocity.y * scale, new_velocity.z)
            # Keep spin in lockstep with translation (both scaled the same
            # way) so v = r*omega stays true as the ball slows -- otherwise
            # the next tick would reopen slip and re-trigger the much
            # stronger kinetic ground friction instead of this gentle decay,
            # every single tick, for the rest of the ball's roll.
            new_spin = Vector3(new_spin.x * scale, new_spin.y * scale, new_spin.z)

    ball.position = new_position
    ball.velocity = new_velocity
    ball.spin = new_spin

    # Update the visual "just bounced" timer: set on a real bounce, decay each tick.
    if real_bounce_this_tick:
        ball.just_bounced_timer_s = params.just_bounced_display_duration_s
    elif ball.just_bounced_timer_s > 0.0:
        ball.just_bounced_timer_s = max(0.0, ball.just_bounced_timer_s - dt_s)

def _reflect_axis(ball: Ball, axis: str, boundary: float, net_e: float, moving_towards_boundary: bool) -> None:
    """Clamp `ball.position[axis]` to `boundary` and negate+damp the matching
    velocity component (scaling the other two components by `net_e` too),
    but only if the ball is actually moving towards the boundary.

    Shared by `resolve_goal_boundary`'s back-wall/side-post/crossbar checks,
    which otherwise repeat this "clamp position, negate velocity component,
    apply restitution" pattern three times.
    """
    if not moving_towards_boundary:
        return
    pos = {"x": ball.position.x, "y": ball.position.y, "z": ball.position.z}
    vel = {"x": ball.velocity.x, "y": ball.velocity.y, "z": ball.velocity.z}
    pos[axis] = boundary
    vel[axis] = -vel[axis] * net_e
    for other in ("x", "y", "z"):
        if other != axis:
            vel[other] *= net_e
    ball.position = Vector3(pos["x"], pos["y"], pos["z"])
    ball.velocity = Vector3(vel["x"], vel["y"], vel["z"])


def resolve_goal_boundary(ball: Ball, pitch: "Pitch", params: BallPhysicsParams) -> None:
    """Bounces the ball off the interior surfaces of whichever goal it has
    entered (back wall, side posts, crossbar).  Ground collisions inside the
    goal are already handled by step_ball's normal ground-contact logic.

    Call this after step_ball on every tick for loose balls.  Does nothing if
    the ball is possessed or has not passed the goal line.
    """
    if ball.possessed_by is not None:
        return

    r = params.ball_radius_m
    half_goal_w = pitch.goal_width_m / 2.0

    in_left = ball.position.x < -pitch.half_length
    in_right = ball.position.x > pitch.half_length
    if not (in_left or in_right):
        return

    # Only apply to balls within the goal mouth (with a small margin for the
    # ball radius).
    if abs(ball.position.y) > half_goal_w + r or ball.position.z > pitch.goal_height_m + r:
        return

    net_e = params.goal_net_restitution

    if in_left:
        back_wall_x = -(pitch.half_length + pitch.goal_depth_m)
        _reflect_axis(ball, "x", back_wall_x + r, net_e, ball.position.x <= back_wall_x + r and ball.velocity.x < 0.0)
    else:
        back_wall_x = pitch.half_length + pitch.goal_depth_m
        _reflect_axis(ball, "x", back_wall_x - r, net_e, ball.position.x >= back_wall_x - r and ball.velocity.x > 0.0)

    # Side posts (y): clamp and reflect.
    _reflect_axis(ball, "y", half_goal_w - r, net_e, ball.position.y > half_goal_w - r and ball.velocity.y > 0.0)
    _reflect_axis(ball, "y", -(half_goal_w - r), net_e, ball.position.y < -(half_goal_w - r) and ball.velocity.y < 0.0)

    # Crossbar (z): clamp and reflect.
    _reflect_axis(ball, "z", pitch.goal_height_m - r, net_e, ball.position.z > pitch.goal_height_m - r and ball.velocity.z > 0.0)
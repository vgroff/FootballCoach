"""Interception math: "where should a chaser aim to catch a moving target
in the shortest possible time" problems.

Both `Match`'s ball/carrier-chasing logic and `goalkeeping.py`'s shot-saving
logic solve variants of this problem; this module holds the general
constant-velocity quadratic solve (`intercept_target`) so it isn't buried as
a private method inside the (already large) `Match` class. `goalkeeping.py`
keeps its own gravity-aware `predict_goal_line_crossing`/
`early_intercept_target` since those solve a meaningfully different problem
(where a projectile crosses a fixed plane under gravity, not "shortest time
to catch a linearly-moving target").
"""
from __future__ import annotations

import math

from footballcoach.mathutils import Vector3


def _solve_min_time(d, vt, vc_sq: float) -> float | None:
    """Constant-velocity quadratic core: smallest non-negative t with
    |d + vt*t|^2 == vc_sq * t^2, or None if unreachable / already past.

    Has a real numerical singularity at |vt|^2 == vc_sq (a = 0 in the
    quadratic below): as the target's speed crosses the chaser's own speed,
    `t` blows up towards +/-infinity right at the crossing, and stays
    wildly (if not infinitely) large for a real neighbourhood around it --
    confirmed on a real traced chase: a = -0.377 (barely past the crossing)
    solved t=130s (the whole chase was 5s) and a predicted position 800+
    metres off the pitch, easing back to sane values only once |a| grew
    past roughly 2. This is inherent to holding the target's speed
    constant -- see `target_decel_mps2` on `intercept_target` for the fix.
    """
    vt_sq = vt.dot(vt)
    d_dot_vt = d.dot(vt)
    d_sq = d.dot(d)

    a = vt_sq - vc_sq
    b = 2.0 * d_dot_vt
    c = d_sq

    if abs(a) < 1e-6:
        return (-c / b) if abs(b) > 1e-6 else 0.0

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    t = None
    for candidate in (t1, t2):
        if candidate >= 0.0 and (t is None or candidate < t):
            t = candidate
    return t


def _decel_distance_along(speed: float, decel: float, t: float) -> float:
    """Distance a decelerating target (initial speed `speed`, constant
    deceleration `decel`, direction held fixed) has covered by time `t`.
    Clamped at the point it comes to a stop -- never reverses. This bound
    is exactly what keeps the decel-aware solve below from blowing up the
    way the constant-velocity one does: no matter how large a solved `t`
    comes out, the distance it maps to can never exceed the target's own
    real, finite stopping distance."""
    if decel <= 0.0:
        return speed * t
    t_stop = speed / decel
    if t >= t_stop:
        return (speed * speed) / (2.0 * decel)
    return speed * t - 0.5 * decel * t * t


def _solve_min_time_decel(d, direction, speed: float, decel: float, vc_sq: float) -> float | None:
    """Smallest non-negative t with |d + direction*s(t)|^2 == vc_sq*t^2,
    where s(t) is the decelerating target's real (quadratic-in-t, then
    clamped) distance along its fixed direction -- see
    `_decel_distance_along`. Quartic in t once expanded, so solved
    numerically (bisection) rather than in closed form.

    g(t) := |d + direction*s(t)|^2 - vc_sq*t^2 starts at g(0) = |d|^2 (>= 0
    whenever chaser and target aren't already colocated) and, because s(t)
    is bounded (the target eventually stops) while vc*t grows without
    bound, g(t) -> -inf as t grows -- so a downward root crossing always
    exists once the chaser is fast enough to ever reach the target's
    eventual stopping point at all; bisection finds it directly.
    """
    d_sq = d.dot(d)
    if d_sq < 1e-9:
        return 0.0

    def g(t: float) -> float:
        s = _decel_distance_along(speed, decel, t)
        rel_x = d.x + direction.x * s
        rel_y = d.y + direction.y * s
        return rel_x * rel_x + rel_y * rel_y - vc_sq * t * t

    t_hi = max(speed / max(decel, 1e-6), 1.0)
    tries = 0
    while g(t_hi) > 0.0 and tries < 60:
        t_hi *= 1.5
        tries += 1
    if g(t_hi) > 0.0:
        return None  # chaser can never reach even the target's eventual stopping point

    t_lo = 0.0
    for _ in range(50):
        t_mid = 0.5 * (t_lo + t_hi)
        if g(t_mid) > 0.0:
            t_lo = t_mid
        else:
            t_hi = t_mid
    return t_hi


def intercept_target(
    chaser_position: Vector3,
    chaser_speed_mps: float,
    target_position: Vector3,
    target_velocity: Vector3,
    target_decel_mps2: float = 0.0,
) -> Vector3:
    """Returns the world position a chaser moving at a constant
    `chaser_speed_mps` should run toward to intercept `target_position`
    (moving at `target_velocity`) in the shortest possible time.

    Base case (`target_decel_mps2=0.0`, exact prior behaviour): solves the
    quadratic for t >= 0 such that the chaser can reach the point
    target_position + target_velocity*t in exactly t seconds:

        |d + v_t*t|^2 = (v_c * t)^2   where d = target_position - chaser_position
        (|v_t|^2 - v_c^2) * t^2 + 2*(d . v_t)*t + |d|^2 = 0

    If the discriminant is negative the chaser cannot catch the target at
    their current speed (target escaping); falls back to the current target
    position. If the target's speed is near-zero, this degenerates cleanly
    to t = |d| / v_c (simple sprint time).

    Only xy components are used — height is irrelevant for interception runs;
    the returned Vector3's z matches `target_position.z`.

    `target_decel_mps2` (default 0.0): the target's own speed drop-off,
    direction held fixed — for a ROLLING BALL, its real ground-friction
    deceleration (see `Match.boundary_braking_params`'s sibling,
    `orders.json["interception"].assumed_ball_decel_mps2`; NOT 0, which is
    never physically true for a real ball). Two real, CONFIRMED problems
    the base (0.0) solve has without this:

    1. It assumes the ball holds its CURRENT speed for the whole chase.
       Since a real ball only ever slows down, the base solve is biased
       toward an early, closer "catch it right now" meeting point rather
       than a later one further along the ball's path that a real
       (slowing) ball leaves reachable for just as long.
    2. Far worse in practice: `_solve_min_time`'s quadratic has a genuine
       numerical singularity whenever the target's speed crosses the
       chaser's own speed (see that function's own docstring) — confirmed
       on a real traced chase, where it produced a predicted intercept
       point over 800 metres off the pitch for a real, multi-tick window,
       during which the chaser was aiming at that point instead of
       anywhere near the actual ball. This is what actually broke the
       traced case, not just the directional bias in point 1.

    Implementation: direction is held fixed (real ball trajectories change
    speed, not heading, between touches) but speed follows the target's
    REAL decelerating profile — s(t) = speed*t - 0.5*decel*t^2, clamped
    once the target would stop — not a single shrunk-but-still-constant
    velocity (that alternative was tried and rejected: it still pulls the
    solved point EARLIER/CLOSER, the opposite of what's needed, since a
    uniformly slower target is uniformly easier to catch sooner). Critically,
    this ALSO fixes problem 2 above: because s(t) is bounded (clamped at the
    target's real stopping distance) rather than growing without bound like
    v_t*t, the same near-zero-`a` condition that blows up the base solve's
    TIME can no longer blow up the resulting POSITION — confirmed: the same
    800m-off-pitch window computes a smooth, sub-10m-off-pitch trajectory
    with this enabled, instead. The quadratic-in-t distance profile makes
    the full equation quartic in t, so this branch solves it numerically
    (`_solve_min_time_decel`) instead of algebraically.
    """
    d = (target_position - chaser_position).xy()
    vt = target_velocity.xy()
    vc_sq = chaser_speed_mps * chaser_speed_mps

    if target_decel_mps2 > 0.0:
        speed = vt.length()
        if speed > 1e-6:
            direction = vt * (1.0 / speed)
            t = _solve_min_time_decel(d, direction, speed, target_decel_mps2, vc_sq)
            if t is not None:
                s = _decel_distance_along(speed, target_decel_mps2, t)
                predicted_xy = target_position + direction * s
                return predicted_xy.with_z(target_position.z)
            # Falls through to the constant-velocity solve below if the
            # numerical solve couldn't find a root (chaser can never reach
            # even the target's eventual stopping point) -- never worse
            # than the pre-existing behaviour.

    t = _solve_min_time(d, vt, vc_sq)
    if t is None:
        return target_position.with_z(target_position.z)
    t = max(0.0, t)
    predicted_xy = target_position + vt * t
    return predicted_xy.with_z(target_position.z)

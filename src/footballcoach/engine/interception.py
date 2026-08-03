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


def intercept_target(
    chaser_position: Vector3,
    chaser_speed_mps: float,
    target_position: Vector3,
    target_velocity: Vector3,
) -> Vector3:
    """Returns the world position a chaser moving at a constant
    `chaser_speed_mps` should run toward to intercept `target_position`
    (moving at constant `target_velocity`) in the shortest possible time.

    Solves the quadratic: find t >= 0 such that the chaser can reach the
    point target_position + target_velocity*t in exactly t seconds:

        |d + v_t*t|^2 = (v_c * t)^2   where d = target_position - chaser_position

    Expanding:
        (|v_t|^2 - v_c^2) * t^2 + 2*(d . v_t)*t + |d|^2 = 0

    If the discriminant is negative the chaser cannot catch the target at
    their current speed (target escaping); falls back to the current target
    position (run toward where it is now).

    If the target's speed is near-zero, the quadratic degenerates cleanly to
    t = |d| / v_c (simple sprint time), which is exactly right.

    Only xy components are used — height is irrelevant for interception runs;
    the returned Vector3's z matches `target_position.z`.
    """
    d = (target_position - chaser_position).xy()
    vt = target_velocity.xy()

    vt_sq = vt.dot(vt)
    vc_sq = chaser_speed_mps * chaser_speed_mps
    d_dot_vt = d.dot(vt)
    d_sq = d.dot(d)

    a = vt_sq - vc_sq
    b = 2.0 * d_dot_vt
    c = d_sq

    if abs(a) < 1e-6:
        # Target and chaser at same speed: linear solution t = -c/b
        t = -c / b if abs(b) > 1e-6 else 0.0
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            # Chaser cannot catch the target — run to current position.
            return target_position.with_z(target_position.z)
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        # Pick smallest non-negative root.
        t = None
        for candidate in (t1, t2):
            if candidate >= 0.0 and (t is None or candidate < t):
                t = candidate
        if t is None:
            # Both roots negative — target already past us; go to current pos.
            return target_position.with_z(target_position.z)

    t = max(0.0, t)
    predicted_xy = target_position + vt * t
    return predicted_xy.with_z(target_position.z)

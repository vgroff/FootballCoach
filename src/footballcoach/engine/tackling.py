"""Tackling: an RNG skill-check between tackler.tackling and
ball_carrier.dribbling. See engine/knowledge.md for the analytical
derivation of win probabilities used to validate this against the balance
targets.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from footballcoach.config import load_physics_config
from footballcoach.mathutils import Vector3
from footballcoach.mathutils.rng import skill_roll


@dataclass(frozen=True)
class TacklingParams:
    tackler_boost: float
    goalkeeper_tackle_boost: float
    goalkeeper_outside_box_tackle_penalty: float
    control_time_penalty_reference_s: float
    inactive_duration_s: float
    inactive_speed_penalty: float
    tackler_miss_inactive_duration_s: float
    dribble_beaten_speed_threshold: float
    dribble_beaten_max_penalty: float
    head_on_min_charge_speed_mps: float
    angle_modifier_frontal: float
    angle_modifier_side: float
    angle_modifier_behind: float

    @staticmethod
    def from_config() -> "TacklingParams":
        d = load_physics_config()["tackling"]
        return TacklingParams(
            tackler_boost=d["tackler_boost"],
            goalkeeper_tackle_boost=d["goalkeeper_tackle_boost"],
            goalkeeper_outside_box_tackle_penalty=d["goalkeeper_outside_box_tackle_penalty"],
            control_time_penalty_reference_s=d["control_time_penalty_reference_s"],
            inactive_duration_s=d["inactive_duration_s"],
            inactive_speed_penalty=d["inactive_speed_penalty"],
            tackler_miss_inactive_duration_s=d["tackler_miss_inactive_duration_s"],
            dribble_beaten_speed_threshold=d["dribble_beaten_speed_threshold"],
            dribble_beaten_max_penalty=d["dribble_beaten_max_penalty"],
            head_on_min_charge_speed_mps=d["head_on_min_charge_speed_mps"],
            angle_modifier_frontal=d["angle_modifier_frontal"],
            angle_modifier_side=d["angle_modifier_side"],
            angle_modifier_behind=d["angle_modifier_behind"],
        )


def tackle_angle_modifier(
    dribbler_heading_rad: float,
    dribbler_pos: Vector3,
    tackler_pos: Vector3,
    params: TacklingParams,
) -> float:
    """Returns an additive modifier for the tackler's effective boost, based on
    which direction the tackler approaches relative to the dribbler's heading.

    The angle is between the dribbler's facing direction and the vector from
    the dribbler to the tackler (i.e. which direction the challenge comes from):

    - 0° (tackler directly in front of dribbler): ``angle_modifier_frontal`` (+0.10)
    - 90° (side-on): ``angle_modifier_side`` (-0.05)
    - 180° (tackle from behind): ``angle_modifier_behind`` (-0.65)

    Interpolation is piecewise linear in cos(angle):
    - cos ∈ [0, 1]: lerp between side and frontal
    - cos ∈ [-1, 0]: lerp between side and behind
    """
    dribbler_dir = Vector3.from_angle_xy(dribbler_heading_rad, 1.0)
    d_to_t = (tackler_pos - dribbler_pos).xy()
    d_to_t_len = d_to_t.length()
    if d_to_t_len < 1e-9:
        return 0.0
    d_to_t_normalized = d_to_t / d_to_t_len
    cos_angle = float(dribbler_dir.dot(d_to_t_normalized))

    frontal = params.angle_modifier_frontal
    side = params.angle_modifier_side
    behind = params.angle_modifier_behind

    if cos_angle >= 0.0:
        # Frontal half: lerp from side (cos=0) to frontal (cos=1)
        return side + (frontal - side) * cos_angle
    else:
        # Behind half: lerp from side (cos=0) to behind (cos=-1)
        return side + (side - behind) * cos_angle


@dataclass(frozen=True)
class TackleResult:
    """Result of a tackle attempt.

    ``tackler_won`` is True when the tackler wins the ball.
    ``dribble_speed_multiplier`` is applied to the dribbler's velocity when
    the dribbler wins but was only narrowly beaten: at 0 margin it is
    ``1 - dribble_beaten_max_penalty`` (≈0.20); above
    ``dribble_beaten_speed_threshold`` margin it reaches 1.0 (no slowdown).
    Always 1.0 when the tackler wins (irrelevant in that case).
    ``tackler_roll`` / ``dribbler_roll``: the actual skill-roll values drawn
    during the contest, exposed so callers (e.g. the game log) can show the
    exact numbers without re-deriving them from ``skill_roll`` internals.
    """
    tackler_won: bool
    dribble_speed_multiplier: float
    tackler_roll: float = 0.0
    dribbler_roll: float = 0.0


def attempt_tackle(
    tackling_attr: float,
    dribbling_attr: float,
    rng_reduction: float,
    rng: random.Random | None = None,
    params: TacklingParams | None = None,
    is_goalkeeper_tackle: bool = False,
    angle_modifier: float = 0.0,
    gk_outside_box: bool = False,
) -> TackleResult:
    """Returns a TackleResult describing who won and how badly the dribbler
    was affected if they managed to keep the ball.

    tackler_roll = (rng_reduction + (1-rng_reduction)*U) * effective_boost * tackling_attr
    dribbler_roll = (rng_reduction + (1-rng_reduction)*U) * dribbling_attr

    The base boost is ``tackler_boost`` (1.25, +25%) for outfield players, or
    ``goalkeeper_tackle_boost`` (2.0, +100%) for goalkeepers.

    ``angle_modifier`` is an additive modifier to the boost, computed by
    ``tackle_angle_modifier`` from the dribbler's heading and the approach
    direction. A frontal tackle (+0.10) makes the tackle slightly easier; a
    tackle from directly behind (-0.65) makes it much harder.

    ``gk_outside_box`` — when ``True`` and ``is_goalkeeper_tackle`` is also
    ``True``, the GK's ``effective_boost`` is multiplied by
    ``(1 - goalkeeper_outside_box_tackle_penalty)`` (default −40%).  This
    reduces a roaming keeper to roughly an outfield-tackler level, per the
    design requirement that a GK leaving their box should not retain their
    full inside-box advantage.  The caller (match.py) is responsible for
    determining whether the GK is outside their own box and passing this flag.

    When the dribbler wins, the margin determines how much they're slowed:
    - Margin < ``dribble_beaten_speed_threshold`` (35%) relative to the
      tackler's roll: speed multiplier scales linearly from
      ``1 - dribble_beaten_max_penalty`` (0.20) at zero margin up to 1.0
      at the threshold.
    - Margin >= threshold: no slowdown (speed_multiplier = 1.0).
    """
    params = params or TacklingParams.from_config()
    r = rng or random

    boost = params.goalkeeper_tackle_boost if is_goalkeeper_tackle else params.tackler_boost
    if is_goalkeeper_tackle and gk_outside_box:
        boost *= (1.0 - params.goalkeeper_outside_box_tackle_penalty)
    effective_boost = boost * (1.0 + angle_modifier)
    t_roll = skill_roll(tackling_attr * effective_boost, rng_reduction, r)
    d_roll = skill_roll(dribbling_attr, rng_reduction, r)

    if t_roll >= d_roll:
        return TackleResult(tackler_won=True, dribble_speed_multiplier=1.0, tackler_roll=t_roll, dribbler_roll=d_roll)

    # Dribbler won - compute how much they're slowed by the near-miss.
    relative_margin = (d_roll - t_roll) / max(t_roll, 1e-9)
    threshold = params.dribble_beaten_speed_threshold
    max_penalty = params.dribble_beaten_max_penalty
    # Linear interpolation: 0 margin -> (1 - max_penalty), threshold -> 1.0
    speed_mult = (1.0 - max_penalty) + max_penalty * min(1.0, relative_margin / threshold)
    return TackleResult(tackler_won=False, dribble_speed_multiplier=speed_mult, tackler_roll=t_roll, dribbler_roll=d_roll)

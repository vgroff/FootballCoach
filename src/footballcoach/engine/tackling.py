"""Tackling: an RNG skill-check between tackler.tackling and
ball_carrier.dribbling. See engine/knowledge.md for the analytical
derivation of win probabilities used to validate this against the balance
targets.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from footballcoach.config import load_physics_config, require_section
from footballcoach.entities.player import Player, PlayerState
from footballcoach.mathutils import Vector3
from footballcoach.mathutils.interp import piecewise_lerp3
from footballcoach.mathutils.rng import skill_roll


@dataclass(frozen=True)
class TacklingParams:
    tackler_boost: float
    goalkeeper_tackle_boost: float
    goalkeeper_outside_box_tackle_penalty: float
    control_time_penalty_reference_s: float
    tackle_cooldown_s: float
    tackle_attempt_tackler_speed_mult: float
    tackle_attempt_tacklee_speed_mult: float
    loser_speed_penalty_scale: float
    loser_speed_penalty_max: float
    auto_tackle_overlap_factor: float
    auto_tackle_min_closing_mps: float
    angle_modifier_frontal: float
    angle_modifier_side: float
    angle_modifier_behind: float
    dribble_beaten_speed_threshold: float
    dribble_beaten_max_penalty: float

    @staticmethod
    def from_config() -> "TacklingParams":
        d = require_section(load_physics_config(), "tackling")
        return TacklingParams(
            tackler_boost=d["tackler_boost"],
            goalkeeper_tackle_boost=d["goalkeeper_tackle_boost"],
            goalkeeper_outside_box_tackle_penalty=d["goalkeeper_outside_box_tackle_penalty"],
            control_time_penalty_reference_s=d["control_time_penalty_reference_s"],
            tackle_cooldown_s=d["tackle_cooldown_s"],
            tackle_attempt_tackler_speed_mult=d["tackle_attempt_tackler_speed_mult"],
            tackle_attempt_tacklee_speed_mult=d["tackle_attempt_tacklee_speed_mult"],
            loser_speed_penalty_scale=d["loser_speed_penalty_scale"],
            loser_speed_penalty_max=d["loser_speed_penalty_max"],
            auto_tackle_overlap_factor=d.get("auto_tackle_overlap_factor", 1.3),
            auto_tackle_min_closing_mps=d.get("auto_tackle_min_closing_mps", 0.2),
            angle_modifier_frontal=d["angle_modifier_frontal"],
            angle_modifier_side=d["angle_modifier_side"],
            angle_modifier_behind=d["angle_modifier_behind"],
            dribble_beaten_speed_threshold=d["dribble_beaten_speed_threshold"],
            dribble_beaten_max_penalty=d["dribble_beaten_max_penalty"],
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

    return piecewise_lerp3(
        cos_angle,
        x_low=-1.0, x_mid=0.0, x_high=1.0,
        y_low=params.angle_modifier_behind,
        y_mid=params.angle_modifier_side,
        y_high=params.angle_modifier_frontal,
    )


@dataclass(frozen=True)
class TackleResult:
    """Result of a tackle attempt.

    ``tackler_won`` is True when the tackler wins the ball.

    Speed multipliers are always applied to both players:
    - Base contact reductions (``tackle_attempt_tackler/tacklee_speed_mult``)
      are applied to both regardless of outcome.
    - The *loser* additionally gets a penalty scaled by the roll difference:
      ``penalty = min(|t_roll - d_roll| * loser_speed_penalty_scale, loser_speed_penalty_max)``
      Their final speed mult = base × (1 - penalty).

    ``tackler_roll`` / ``dribbler_roll``: the actual skill-roll values drawn
    during the contest, exposed so callers (e.g. the game log) can show the
    exact numbers without re-deriving them from ``skill_roll`` internals.
    """
    tackler_won: bool
    tackler_speed_mult: float
    tacklee_speed_mult: float
    tackler_roll: float = 0.0
    dribbler_roll: float = 0.0

def apply_tackle_result(
    result: "TackleResult",
    tackler: "Player",
    tacklee: "Player",
    params: TacklingParams,
) -> None:
    """Apply the physical consequences of a resolved tackle to both players.

    This is the ONLY place velocity and state are written for tackle outcomes.
    Possession transfer is NOT handled here — the caller must call
    ``match._set_possession()`` separately so the possession callback fires.
    """
    from footballcoach.entities.player import Player, PlayerState  # local to avoid circular  # noqa: F401
    tackler.velocity = tackler.velocity * result.tackler_speed_mult
    tacklee.velocity = tacklee.velocity * result.tacklee_speed_mult
    tackler.state = PlayerState.INACTIVE_TACKLED
    tackler.state_timer_s = params.tackle_cooldown_s
    tacklee.state = PlayerState.INACTIVE_TACKLED
    tacklee.state_timer_s = params.tackle_cooldown_s


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

    # Base contact speed reductions applied to both players always.
    base_tackler = params.tackle_attempt_tackler_speed_mult
    base_tacklee = params.tackle_attempt_tacklee_speed_mult

    # Loser's additional penalty, proportional to roll difference.
    diff = abs(t_roll - d_roll)
    loser_extra_penalty = min(diff * params.loser_speed_penalty_scale, params.loser_speed_penalty_max)
    loser_extra_mult = 1.0 - loser_extra_penalty

    if t_roll >= d_roll:
        # Tackler wins: tacklee is the loser and gets extra slow-down.
        return TackleResult(
            tackler_won=True,
            tackler_speed_mult=base_tackler,
            tacklee_speed_mult=base_tacklee * loser_extra_mult,
            tackler_roll=t_roll,
            dribbler_roll=d_roll,
        )
    else:
        # Dribbler wins: tackler is the loser; dribbler speed depends on
        # how convincingly they won (relative margin vs tackler's roll).
        # Large margin (>= threshold) → full speed retained (mult=1.0).
        # Near-zero margin → slowed by up to dribble_beaten_max_penalty.
        threshold = params.dribble_beaten_speed_threshold
        max_penalty = params.dribble_beaten_max_penalty
        relative_margin = (d_roll - t_roll) / t_roll if t_roll > 1e-9 else 1.0
        if relative_margin >= threshold:
            dribbler_mult = 1.0
        else:
            dribbler_mult = 1.0 - max_penalty * (1.0 - relative_margin / threshold)
        return TackleResult(
            tackler_won=False,
            tackler_speed_mult=base_tackler * loser_extra_mult,
            tacklee_speed_mult=dribbler_mult,
            tackler_roll=t_roll,
            dribbler_roll=d_roll,
        )

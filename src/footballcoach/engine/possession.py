"""Ball possession and first-touch control-time model.

See engine/knowledge.md for the full derivation of the control-time formula
and its constants (control points chosen to feel intuitive: a rolling ball
at the feet is a near-instant touch, a fast chest-high ball is a real
challenge, especially for low ball-control players).
"""
from __future__ import annotations

from dataclasses import dataclass

from footballcoach.config import load_physics_config, require_section
from footballcoach.entities.player import Player


@dataclass(frozen=True)
class ControlTimeParams:
    t_base_s: float
    t_scale_s: float
    height_factor_knee: float
    height_factor_waist: float
    height_factor_head: float
    height_factor_above_head_extra: float
    height_factor_max: float
    k1_relative_velocity_s_per_mps: float
    k2_own_velocity_s_per_mps: float
    ball_control_alpha: float
    noise_sigma_fraction: float
    knee_height_m: float
    waist_height_m: float
    player_height_m: float
    gk_t_base_s: float
    gk_height_factor_scale: float
    gk_ball_control_alpha: float
    # Jump zone penalties (above player_height_m = 1.8m)
    # GK: flatter penalty, higher ceiling (gloves + specialised technique)
    gk_jump_scale_at_max_reach: float  # height_factor_scale at gk_max_reach_height_m
    gk_max_reach_height_m: float       # max height GK can credibly intercept
    # Outfield: steeper penalty, lower ceiling
    outfield_jump_scale_at_max_reach: float  # applied as multiplier on (height_factor-1) at max reach
    outfield_max_reach_height_m: float       # max height outfield can credibly jump to
    control_tackle_immune_height_m: float    # ball must be below this height for tackles to land on a CONTROLLING_BALL player

    @staticmethod
    def from_config() -> "ControlTimeParams":
        cfg = load_physics_config()
        d = require_section(cfg, "control_time")
        player_cfg = require_section(cfg, "player")
        gk = require_section(d, "goalkeeper", file_name="physics.json:control_time")
        return ControlTimeParams(
            t_base_s=d["t_base_s"],
            t_scale_s=d["t_scale_s"],
            height_factor_knee=d["height_factor_knee"],
            height_factor_waist=d["height_factor_waist"],
            height_factor_head=d["height_factor_head"],
            height_factor_above_head_extra=d["height_factor_above_head_extra"],
            height_factor_max=d["height_factor_max"],
            k1_relative_velocity_s_per_mps=d["k1_relative_velocity_s_per_mps"],
            k2_own_velocity_s_per_mps=d["k2_own_velocity_s_per_mps"],
            ball_control_alpha=d["ball_control_alpha"],
            noise_sigma_fraction=d["noise_sigma_fraction"],
            knee_height_m=player_cfg["knee_height_m"],
            waist_height_m=player_cfg["waist_height_m"],
            player_height_m=player_cfg["height_m"],
            gk_t_base_s=gk["t_base_s"],
            gk_height_factor_scale=gk["height_factor_scale"],
            gk_ball_control_alpha=gk["ball_control_alpha"],
            gk_jump_scale_at_max_reach=gk.get("jump_scale_at_max_reach", 1.2),
            gk_max_reach_height_m=gk.get("max_reach_height_m", 2.2),
            outfield_jump_scale_at_max_reach=d.get("outfield_jump_scale_at_max_reach", 2.0),
            outfield_max_reach_height_m=d.get("outfield_max_reach_height_m", 2.0),
            control_tackle_immune_height_m=d.get("control_tackle_immune_height_m", 0.95),
        )


def height_difficulty_factor(params: ControlTimeParams, ball_height_m: float) -> float:
    """Piecewise convex ramp: f=1.0 at/below knee height, rising to
    height_factor_waist at waist height, height_factor_head at head height,
    and increasing (capped) beyond that. Modeling choice: it's quadratic
    within each band so difficulty ramps up gently near the breakpoint and
    steeply as the ball gets awkwardly high (matches intuition that a
    shin-high ball is barely harder than a rolling one, but a shoulder-high
    ball is a real technical challenge).
    """
    knee = params.knee_height_m
    waist = params.waist_height_m
    head = params.player_height_m

    if ball_height_m <= knee:
        return params.height_factor_knee
    if ball_height_m <= waist:
        t = (ball_height_m - knee) / (waist - knee)
        return params.height_factor_knee + (params.height_factor_waist - params.height_factor_knee) * (t * t)
    if ball_height_m <= head:
        t = (ball_height_m - waist) / (head - waist)
        return params.height_factor_waist + (params.height_factor_head - params.height_factor_waist) * (t * t)

    extra = params.height_factor_above_head_extra * ((ball_height_m - head) / head)
    return min(params.height_factor_max, params.height_factor_head + extra)


def compute_difficulty(
    params: ControlTimeParams,
    ball_height_m: float,
    relative_speed_mps: float,
    player_speed_mps: float,
) -> float:
    """Returns a unitless "difficulty" score (0 = trivial rolling ball at
    rest at the feet, growing with height and velocity) used both for
    control-time and for inflating first-time shot error."""
    height_factor = height_difficulty_factor(params, ball_height_m)
    return (
        (height_factor - 1.0)
        + params.k1_relative_velocity_s_per_mps * relative_speed_mps
        + params.k2_own_velocity_s_per_mps * player_speed_mps
    )


def _jump_zone_scale(
    ball_height_m: float,
    player_height_m: float,
    max_reach_height_m: float,
    scale_at_player_height: float,
    scale_at_max_reach: float,
) -> float:
    """Interpolates the height-factor scale for balls above ``player_height_m``
    (the "jump zone"): ``scale_at_player_height`` at/below head height, rising
    linearly to ``scale_at_max_reach`` at ``max_reach_height_m`` and beyond.

    Shared by both the goalkeeper and outfield branches of `control_time_s`,
    which apply the same interpolation shape with different endpoints.
    """
    if ball_height_m <= player_height_m:
        return scale_at_player_height
    jump_range = max(max_reach_height_m - player_height_m, 1e-6)
    jump_frac = min(1.0, (ball_height_m - player_height_m) / jump_range)
    return scale_at_player_height + (scale_at_max_reach - scale_at_player_height) * jump_frac


def control_time_s(
    params: ControlTimeParams,
    ball_height_m: float,
    relative_speed_mps: float,
    player_speed_mps: float,
    ball_control_attr: float,
    is_goalkeeper_in_box: bool = False,
) -> float:
    """Returns the (deterministic, pre-RNG) time in seconds for a player to
    bring the ball under control on a first touch.

    t_control = t_base + t_scale * (1 - alpha*ball_control) * difficulty

    Goalkeepers in their own box use a flatter height penalty and a lower
    base time (catching is much easier/faster than an outfielder's touch),
    per the design spec.
    """
    if is_goalkeeper_in_box:
        height_factor = height_difficulty_factor(params, ball_height_m)
        # Below head height: existing flat-scale (gk_height_factor_scale).
        # Above head height (jump zone): interpolate the scale from
        # gk_height_factor_scale toward gk_jump_scale_at_max_reach as the
        # ball rises from player_height_m to gk_max_reach_height_m.
        # Beyond gk_max_reach_height_m the scale stays at max (ball is
        # unreachable — height_factor_max cap still applies).
        effective_scale = _jump_zone_scale(
            ball_height_m, params.player_height_m, params.gk_max_reach_height_m,
            params.gk_height_factor_scale, params.gk_jump_scale_at_max_reach,
        )
        scaled_height_factor = 1.0 + (height_factor - 1.0) * effective_scale
        difficulty = (
            (scaled_height_factor - 1.0)
            + params.k1_relative_velocity_s_per_mps * relative_speed_mps
            + params.k2_own_velocity_s_per_mps * player_speed_mps
        )
        alpha = params.gk_ball_control_alpha
        t_base = params.gk_t_base_s
    else:
        # Below head height: existing difficulty curve, unmodified (pure regression safety).
        # Above head height (outfield jump zone): apply extra difficulty scaling
        # on the height_factor term, interpolating from 1.0 at player_height_m
        # toward outfield_jump_scale_at_max_reach at outfield_max_reach_height_m.
        if ball_height_m > params.player_height_m:
            height_factor = height_difficulty_factor(params, ball_height_m)
            effective_scale = _jump_zone_scale(
                ball_height_m, params.player_height_m, params.outfield_max_reach_height_m,
                1.0, params.outfield_jump_scale_at_max_reach,
            )
            scaled_height_term = (height_factor - 1.0) * effective_scale
            difficulty = (
                scaled_height_term
                + params.k1_relative_velocity_s_per_mps * relative_speed_mps
                + params.k2_own_velocity_s_per_mps * player_speed_mps
            )
        else:
            difficulty = compute_difficulty(params, ball_height_m, relative_speed_mps, player_speed_mps)
        alpha = params.ball_control_alpha
        t_base = params.t_base_s

    extra = (1.0 - alpha * ball_control_attr) * difficulty
    return t_base + params.t_scale_s * extra

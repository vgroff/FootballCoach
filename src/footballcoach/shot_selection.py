"""AI-layer shot placement: pick which corner of the goal to aim a shot at.

Deliberately NOT under ``engine/`` — like ``steering.py``, this is an
AI-layer decision (what a shooter *chooses* to do) built on top of engine
physics/config, not a physics mechanic itself. The engine stays unaware of
this module; only rules-based AI (``rules_ai.py``) and scenario builders
call into it.

The idea: aim for one of the two "corners" of the goal (near either post,
inset by ``corner_tolerance_m``), at a random height capped at
``height_tolerance_frac`` of the goal height. Which corner is chosen is not
random -- it's whichever one the ball is more likely to reach before the
goalkeeper can get there, using the *same* interception-plane physics
(``engine/goalkeeping.py``'s ``goal_frame_margin_m``) and the *same*
accel-aware ETA model (``engine/movement.py``'s ``sprint_eta``, already used
by the goalkeeper's own sprint-vs-jog decision in ``orders.py``) that the
goalkeeper's real SaveOrder logic uses to react to a live shot. This keeps
the shot-planner's model of "will this be saved" consistent with what the
engine will actually do once the shot is taken, rather than an independently
tuned (and likely miscalibrated) guess.

See ``ai/config/ai_config.json``... actually see ``orders.json["shot_selection"]``
for the tunable corner/height tolerances.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from footballcoach.config import load_orders_config, require_section
from footballcoach.engine.ball_physics import BallPhysicsParams
from footballcoach.engine.goalkeeping import GoalkeepingParams, own_goal_x
from footballcoach.engine.kicking import KickingParams, max_kick_speed_mps, solve_launch_pitch_rad
from footballcoach.engine.movement import MovementParams, effective_acceleration, effective_top_speed, sprint_eta
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Player, Team
from footballcoach.mathutils import Vector3


@dataclass(frozen=True)
class ShotSelectionParams:
    """Config for smart shot placement — loaded from orders.json."""
    corner_tolerance_m: float
    height_tolerance_frac: float

    @staticmethod
    def from_config() -> "ShotSelectionParams":
        d = require_section(load_orders_config(), "shot_selection", "orders.json")
        return ShotSelectionParams(
            corner_tolerance_m=d["corner_tolerance_m"],
            height_tolerance_frac=d["height_tolerance_frac"],
        )


def _corner_ball_flight_time_s(
    shooter_position: Vector3,
    aim_point: Vector3,
    shot_speed_mps: float,
    gravity_mps2: float,
) -> float:
    """Time for a shot struck at ``shot_speed_mps`` from ``shooter_position``
    to reach ``aim_point``, using the same ballistic solve (``kick_ball``'s
    ``solve_launch_pitch_rad``) the engine uses to actually aim the kick —
    so this estimate matches what will really happen, not an independent
    approximation."""
    horizontal_distance = shooter_position.xy().distance_to(aim_point.xy())
    height_diff = aim_point.z - shooter_position.z
    pitch_rad = solve_launch_pitch_rad(horizontal_distance, height_diff, shot_speed_mps, gravity_mps2)
    horizontal_speed = shot_speed_mps * math.cos(pitch_rad)
    return horizontal_distance / max(horizontal_speed, 1e-3)


def _gk_eta_s(gk: Player, target_xy: Vector3, movement_params: MovementParams) -> float:
    """Accel-aware time for `gk` to reach `target_xy` (ground position only —
    jumping for height is a separate control-time concern handled elsewhere,
    not a movement-speed one), via the same ``sprint_eta`` model the
    goalkeeper's own sprint-vs-jog decision uses (see ``orders.py``'s
    ``_gk_should_sprint``)."""
    dist = gk.position.xy().distance_to(target_xy.xy())
    top_speed = effective_top_speed(
        movement_params, gk.attributes.top_speed, gk.stamina, has_ball=False, is_goalkeeper=True,
    )
    accel = effective_acceleration(movement_params, gk.attributes.acceleration, gk.stamina, is_goalkeeper=True)
    return sprint_eta(dist, gk.speed_mps, top_speed, accel)


def choose_shot_target(
    shooter: Player,
    power_fraction: float,
    gk: Player | None,
    pitch: Pitch,
    rng: random.Random,
    *,
    gravity_mps2: float | None = None,
    kicking_params: KickingParams | None = None,
    goalkeeping_params: GoalkeepingParams | None = None,
    movement_params: MovementParams | None = None,
    params: ShotSelectionParams | None = None,
) -> Vector3:
    """Pick a smart aim point for `shooter` to shoot at, at the moment of
    shooting (call this right before constructing the ``ShootOrder``/
    ``KickOrder``, not at scenario-build time, so it sees live shooter/GK
    positions).

    Candidates are the two goal corners (inset by
    ``params.corner_tolerance_m``), at a single random height shared by both
    candidates (capped at ``params.height_tolerance_frac`` of the goal
    height) — so the choice between corners is purely about which side is
    more open, not confounded by also varying height. For each corner,
    estimates the ball's flight time (via the real ballistic solve, at the
    shot speed this shooter will actually produce with `power_fraction`) and
    the goalkeeper's accel-aware ETA to the same point, evaluated at the
    *interception plane* the goalkeeper's own SaveOrder logic actually
    targets (``goal_frame_margin_m`` in front of the true goal line) — not
    the literal goal line — so this model of "will it be saved" matches what
    a real save attempt does. Returns whichever corner gives the ball the
    larger arrival-time margin over the goalkeeper.

    If `gk` is ``None`` (no goalkeeper on the pitch), falls back to a
    uniformly random corner — there's no race to evaluate.
    """
    params = params or ShotSelectionParams.from_config()
    kicking_params = kicking_params or KickingParams.from_config()
    goalkeeping_params = goalkeeping_params or GoalkeepingParams.from_config()
    movement_params = movement_params or MovementParams.from_config()
    gravity_mps2 = gravity_mps2 if gravity_mps2 is not None else BallPhysicsParams.from_config().gravity_mps2

    defending_team = Team.RIGHT if shooter.team == Team.LEFT else Team.LEFT
    goal_x = own_goal_x(pitch, defending_team)
    sign = 1.0 if defending_team == Team.LEFT else -1.0  # direction from goal line INTO the pitch
    target_plane_x = goal_x + sign * goalkeeping_params.goal_frame_margin_m

    half_goal_w = pitch.goal_width_m / 2.0
    corner_y = max(0.0, half_goal_w - params.corner_tolerance_m)
    height = rng.uniform(0.0, params.height_tolerance_frac * pitch.goal_height_m)

    candidates = [
        Vector3(target_plane_x, corner_y, height),
        Vector3(target_plane_x, -corner_y, height),
    ]

    if gk is None:
        return rng.choice(candidates)

    shot_speed = power_fraction * max_kick_speed_mps(kicking_params, shooter.attributes.kick_power)

    best_candidate = candidates[0]
    best_margin = -math.inf
    for candidate in candidates:
        ball_time = _corner_ball_flight_time_s(shooter.position, candidate, shot_speed, gravity_mps2)
        gk_time = _gk_eta_s(gk, candidate, movement_params)
        # Larger margin = better for the shooter: the GK takes longer to
        # reach this corner than the ball takes to arrive there.
        margin = gk_time - ball_time
        if margin > best_margin:
            best_margin = margin
            best_candidate = candidate

    return best_candidate

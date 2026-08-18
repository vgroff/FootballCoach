"""Random ball-episode generation for physics pretraining.

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

See agent_plans/ball_physics_pretrain_plan.md section 4 for the full spec.
One "episode" = one random ball state, stepped forward under the real
``step_ball()``/``resolve_goal_boundary()`` physics for up to
``max(horizons_s)`` seconds, recorded at 5 fixed horizons with an
out-of-bounds/goal freeze rule (section 4.3). No ``Match``/player/AI
machinery is needed -- see section 4.1.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

import numpy as np

from footballcoach.config import load_physics_config, require_section
from footballcoach.engine.ball_physics import BallPhysicsParams, resolve_goal_boundary, step_ball
from footballcoach.engine.scoring import check_goal
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.mathutils import Vector3

N_INPUT_FIELDS = 14
N_TARGET_FIELDS_PER_HORIZON = 11


@dataclass(frozen=True)
class BallEpisodeGenParams:
    """Randomization ranges + normalization constants for one episode draw.

    All values sourced from ``ai_config.json["physics_pretrain"]["ball"]``/
    ``["observation"]`` and ``physics.json["pitch"/"ball_physics"]`` via
    ``from_config()`` -- never hardcoded, see the plan's section 4.2.
    """
    horizons_s: tuple[float, ...]
    pitch_scale_range: tuple[float, float]
    restitution_scale_range: tuple[float, float]
    ball_speed_max_mps: float
    out_of_bounds_start_frac: float
    spin_active_frac: float
    sim_dt_s: float
    base_pitch_length_m: float
    base_pitch_width_m: float
    base_goal_width_m: float
    base_goal_height_m: float
    base_restitution: float
    ball_spin_norm_max_rad_s: float
    height_norm_m: float

    @staticmethod
    def from_config() -> "BallEpisodeGenParams":
        from footballcoach.ai.config import load_ai_config
        ai_cfg = load_ai_config()
        pp_cfg = ai_cfg["physics_pretrain"]["ball"]
        obs_cfg = ai_cfg["observation"]
        phys_cfg = load_physics_config()
        pitch_cfg = require_section(phys_cfg, "pitch")
        ball_physics_cfg = require_section(phys_cfg, "ball_physics")
        return BallEpisodeGenParams(
            horizons_s=tuple(float(h) for h in pp_cfg["horizons_s"]),
            pitch_scale_range=tuple(float(s) for s in pp_cfg["pitch_scale_range"]),
            restitution_scale_range=tuple(float(s) for s in pp_cfg["restitution_scale_range"]),
            ball_speed_max_mps=float(pp_cfg["ball_speed_max_mps"]),
            out_of_bounds_start_frac=float(pp_cfg["out_of_bounds_start_frac"]),
            spin_active_frac=float(pp_cfg.get("spin_active_frac", 0.35)),
            sim_dt_s=float(obs_cfg["sim_dt_s"]),
            base_pitch_length_m=float(pitch_cfg["length_m"]),
            base_pitch_width_m=float(pitch_cfg["width_m"]),
            base_goal_width_m=float(pitch_cfg["goal_width_m"]),
            base_goal_height_m=float(pitch_cfg["goal_height_m"]),
            base_restitution=float(ball_physics_cfg["bounce_restitution_vertical"]),
            ball_spin_norm_max_rad_s=float(obs_cfg["ball_spin_norm_max_rad_s"]),
            height_norm_m=float(obs_cfg.get("height_norm_m", 3.0)),
        )


def _sample_pitch(rng: random.Random, params: BallEpisodeGenParams) -> Pitch:
    base = Pitch.standard()
    lo, hi = params.pitch_scale_range
    return Pitch(
        length_m=params.base_pitch_length_m * rng.uniform(lo, hi),
        width_m=params.base_pitch_width_m * rng.uniform(lo, hi),
        goal_width_m=params.base_goal_width_m * rng.uniform(lo, hi),
        goal_height_m=params.base_goal_height_m * rng.uniform(lo, hi),
        # Box/six-yard/penalty-spot/centre-circle geometry is irrelevant to
        # ball flight and the out-of-bounds/goal checks (§3.1 explicitly
        # excludes box dims) -- kept at standard values, never read below.
        goal_depth_m=base.goal_depth_m,
        box_length_m=base.box_length_m,
        box_width_m=base.box_width_m,
        six_yard_length_m=base.six_yard_length_m,
        six_yard_width_m=base.six_yard_width_m,
        penalty_spot_distance_m=base.penalty_spot_distance_m,
        centre_circle_radius_m=base.centre_circle_radius_m,
    )


def _sample_restitution(rng: random.Random, params: BallEpisodeGenParams) -> float:
    lo, hi = params.restitution_scale_range
    r = params.base_restitution * rng.uniform(lo, hi)
    return max(0.2, min(0.95, r))  # same clamp as ui/scenarios.py's episode restitution randomization


def _sample_position_in_play(rng: random.Random, pitch: Pitch) -> Vector3:
    margin = 1.0
    x = rng.uniform(-pitch.half_length + margin, pitch.half_length - margin)
    y = rng.uniform(-pitch.half_width + margin, pitch.half_width - margin)
    z = rng.uniform(0.0, 3.0)  # include some already-airborne (post-kick) starts, not just grounded
    return Vector3(x, y, z)


def _sample_position_already_special(rng: random.Random, pitch: Pitch) -> Vector3:
    """A position that is already out of bounds or already inside a goal
    mouth at t=0 (§4.2's "~10-20% of episodes start already out of pitch
    bounds or already past the goal line")."""
    half_goal_w = pitch.goal_width_m / 2.0
    side = 1.0 if rng.random() < 0.5 else -1.0
    if rng.random() < 0.5:
        # Already inside a goal mouth (goal_scored from tick 1).
        x = side * (pitch.half_length + rng.uniform(0.1, max(0.2, pitch.goal_depth_m * 0.8)))
        y = rng.uniform(-half_goal_w * 0.8, half_goal_w * 0.8)
        z = rng.uniform(0.0, pitch.goal_height_m * 0.8)
    elif rng.random() < 0.5:
        # Out via the x boundary, outside the goal mouth width (out_of_bounds, not a goal).
        x = side * (pitch.half_length + rng.uniform(0.2, 5.0))
        y_side = 1.0 if rng.random() < 0.5 else -1.0
        y = y_side * rng.uniform(half_goal_w + 0.5, pitch.half_width + 3.0)
        z = rng.uniform(0.0, 3.0)
    else:
        # Out via the y (side) boundary.
        x = rng.uniform(-pitch.half_length, pitch.half_length)
        y = side * (pitch.half_width + rng.uniform(0.2, 5.0))
        z = rng.uniform(0.0, 3.0)
    return Vector3(x, y, z)


def _sample_velocity(rng: random.Random, speed_max_mps: float) -> Vector3:
    # Uniform random direction on the unit sphere (Gaussian-then-normalize),
    # magnitude uniform in [0, speed_max_mps] (§4.2).
    v = np.array([rng.gauss(0.0, 1.0) for _ in range(3)])
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        v = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    v = v / norm * rng.uniform(0.0, speed_max_mps)
    return Vector3(float(v[0]), float(v[1]), float(v[2]))


def _sample_spin(rng: random.Random, spin_max_rad_s: float, active_frac: float) -> Vector3:
    """Zero spin most of the time (real play is usually low/no-spin);
    ``active_frac`` of episodes get a random-axis, random-magnitude spin
    instead -- see ``physics_pretrain.ball.spin_active_frac``."""
    if rng.random() >= active_frac:
        return Vector3.zero()
    v = np.array([rng.gauss(0.0, 1.0) for _ in range(3)])
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        v = np.array([0.0, 0.0, 1.0])
        norm = 1.0
    v = v / norm * rng.uniform(0.0, spin_max_rad_s)
    return Vector3(float(v[0]), float(v[1]), float(v[2]))


def _encode_input(
    ball: Ball, pitch: Pitch, restitution: float, params: BallEpisodeGenParams,
) -> np.ndarray:
    half_diag = math.hypot(pitch.half_length, pitch.half_width)
    row = np.zeros(N_INPUT_FIELDS, dtype=np.float32)
    row[0] = ball.position.x / pitch.half_length
    row[1] = ball.position.y / pitch.half_width
    row[2] = ball.position.z / params.height_norm_m
    row[3] = ball.velocity.x / half_diag
    row[4] = ball.velocity.y / half_diag
    row[5] = ball.velocity.z / half_diag
    row[6] = ball.spin.x / params.ball_spin_norm_max_rad_s
    row[7] = ball.spin.y / params.ball_spin_norm_max_rad_s
    row[8] = ball.spin.z / params.ball_spin_norm_max_rad_s
    row[9] = restitution
    row[10] = pitch.length_m / params.base_pitch_length_m
    row[11] = pitch.width_m / params.base_pitch_width_m
    row[12] = pitch.goal_width_m / params.base_goal_width_m
    row[13] = pitch.goal_height_m / params.base_goal_height_m
    return row


def _encode_target(
    position: Vector3, velocity: Vector3, spin: Vector3,
    out_of_bounds: bool, goal_scored: bool,
    pitch: Pitch, params: BallEpisodeGenParams,
) -> np.ndarray:
    half_diag = math.hypot(pitch.half_length, pitch.half_width)
    row = np.zeros(N_TARGET_FIELDS_PER_HORIZON, dtype=np.float32)
    row[0] = position.x / pitch.half_length
    row[1] = position.y / pitch.half_width
    row[2] = position.z / params.height_norm_m
    row[3] = velocity.x / half_diag
    row[4] = velocity.y / half_diag
    row[5] = velocity.z / half_diag
    row[6] = spin.x / params.ball_spin_norm_max_rad_s
    row[7] = spin.y / params.ball_spin_norm_max_rad_s
    row[8] = spin.z / params.ball_spin_norm_max_rad_s
    row[9] = 1.0 if out_of_bounds else 0.0
    row[10] = 1.0 if goal_scored else 0.0
    return row


def generate_episode(
    rng: random.Random, params: BallEpisodeGenParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draws one random ball episode and returns ``(input[14], target[55])``.

    Target is 5 horizons (``params.horizons_s``, in that order) x 11 fields,
    flattened -- see §3.2. Freeze-on-event semantics per §4.3: the instant
    the ball goes out of bounds or a goal is scored, its state is frozen and
    held for every later-recorded horizon, with both flags latched true.
    """
    params = params or BallEpisodeGenParams.from_config()
    pitch = _sample_pitch(rng, params)
    restitution = _sample_restitution(rng, params)
    phys_params = replace(BallPhysicsParams.from_config(), bounce_restitution_vertical=restitution)

    start_special = rng.random() < params.out_of_bounds_start_frac
    position = (
        _sample_position_already_special(rng, pitch) if start_special
        else _sample_position_in_play(rng, pitch)
    )
    ball = Ball(
        position=position,
        velocity=_sample_velocity(rng, params.ball_speed_max_mps),
        spin=_sample_spin(rng, params.ball_spin_norm_max_rad_s, params.spin_active_frac),
    )

    input_row = _encode_input(ball, pitch, restitution, params)

    dt = params.sim_dt_s
    horizon_ticks = [max(1, round(h / dt)) for h in params.horizons_s]
    max_tick = max(horizon_ticks)
    horizon_tick_set = set(horizon_ticks)

    frozen: tuple[Vector3, Vector3, Vector3] | None = None
    out_of_bounds = False
    goal_scored = False
    recorded: dict[int, tuple[Vector3, Vector3, Vector3, bool, bool]] = {}

    for tick in range(1, max_tick + 1):
        if frozen is None:
            step_ball(ball, dt, phys_params)
            resolve_goal_boundary(ball, pitch, phys_params)
            if not pitch.is_in_bounds(ball.position):
                out_of_bounds = True
            if check_goal(ball, pitch) is not None:
                goal_scored = True
            if out_of_bounds or goal_scored:
                frozen = (ball.position, ball.velocity, ball.spin)
        if tick in horizon_tick_set:
            pos, vel, spn = frozen if frozen is not None else (ball.position, ball.velocity, ball.spin)
            recorded[tick] = (pos, vel, spn, out_of_bounds, goal_scored)

    target_row = np.concatenate([
        _encode_target(*recorded[t], pitch, params) for t in horizon_ticks
    ])
    return input_row, target_row


def generate_shard(
    n_episodes: int, seed: int, params: BallEpisodeGenParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates ``n_episodes`` independent episodes with a seeded RNG.

    Returns ``(inputs, targets)`` of shape ``(n_episodes, 14)`` /
    ``(n_episodes, 55)``. Each shard's episodes are fully independent draws
    (no shared state), so this is safe to call from a separate worker
    process per shard -- see ball_dataset.py's ``generate_dataset()``.
    """
    params = params or BallEpisodeGenParams.from_config()
    rng = random.Random(seed)
    inputs = np.empty((n_episodes, N_INPUT_FIELDS), dtype=np.float32)
    targets = np.empty((n_episodes, len(params.horizons_s) * N_TARGET_FIELDS_PER_HORIZON), dtype=np.float32)
    for i in range(n_episodes):
        inputs[i], targets[i] = generate_episode(rng, params)
    return inputs, targets

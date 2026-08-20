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

N_INPUT_FIELDS = 20
N_TARGET_FIELDS_PER_HORIZON = 11

# Fixed divisor for encoding/decoding spin (``_kinematics_divisors``'s
# ``div_spin``) -- DELIBERATELY hardcoded, NOT sourced from
# ``ball_spin_norm_max_rad_s`` (a `BallEpisodeGenParams` field, see below)
# even though the two used to be the same number. Reasoning: `params.
# ball_spin_norm_max_rad_s` is also `ai/obs/encoder.py`'s live-match spin
# normalization scale (same `ai_config.json["observation"]` key, shared
# between systems) AND `_sample_spin`'s sampling-range upper bound -- both
# of which are legitimate to want to change independently of "what number
# do already-generated dataset shards divide spin by". If this dataset's
# own normalization divisor tracked that config value, bumping
# `ball_spin_norm_max_rad_s` to widen the sampling range (or to retune the
# live encoder) would silently re-scale every NEWLY generated shard's spin
# encoding relative to already-generated ones still on disk -- and
# `ball_dataset.generate_dataset` APPENDS to an existing output directory
# rather than overwriting it, so that mismatch would go undetected and
# quietly corrupt training on any dataset grown across the config change.
# Fixing this divisor means the sampling range CAN be widened freely (a
# sampled spin above this constant just normalizes to a value >1.0 --
# already the norm for position under `normalize_kinematics_by_base_pitch`,
# see `_kinematics_divisors`'s own docstring, so an unbounded-magnitude
# normalized field is an established convention here, not a new one).
BALL_SPIN_NORM_DIVISOR_RAD_S = 30.0


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
    # Sampling-range upper bound for `_sample_spin` only, and (shared config
    # key) `ai/obs/encoder.py`'s live-match spin normalization scale. NOT
    # this dataset's own spin normalization divisor -- see
    # `BALL_SPIN_NORM_DIVISOR_RAD_S` above for why those were deliberately
    # split apart.
    ball_spin_norm_max_rad_s: float
    height_norm_m: float
    normalize_kinematics_by_base_pitch: bool
    goal_net_collisions_enabled: bool
    ball_radius_m: float

    @staticmethod
    def from_config() -> "BallEpisodeGenParams":
        from footballcoach.ai.config import load_ai_config
        ai_cfg = load_ai_config()
        pp_cfg = ai_cfg["physics_pretrain"]["ball"]
        obs_cfg = ai_cfg["observation"]
        phys_cfg = load_physics_config()
        pitch_cfg = require_section(phys_cfg, "pitch")
        ball_physics_cfg = require_section(phys_cfg, "ball_physics")
        ball_cfg = require_section(phys_cfg, "ball")
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
            normalize_kinematics_by_base_pitch=bool(pp_cfg.get("normalize_kinematics_by_base_pitch", False)),
            goal_net_collisions_enabled=bool(pp_cfg.get("goal_net_collisions_enabled", True)),
            ball_radius_m=float(ball_cfg["radius_m"]),
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


def _sample_position_in_play(rng: random.Random, pitch: Pitch, radius: float) -> Vector3:
    margin = 1.0
    x = rng.uniform(-pitch.half_length + margin, pitch.half_length - margin)
    y = rng.uniform(-pitch.half_width + margin, pitch.half_width - margin)
    # Floored at the ball's own radius, not 0 -- z is the ball's CENTER, and
    # a resting ball's center can never be below its radius (see step_ball's
    # ground clamp) -- sampling from 0 could start an episode with the
    # ball's center literally below/inside the ground, which step_ball then
    # resolves by snapping straight up to `radius` on the very first tick
    # (and treating it as a "bounce" if the sampled velocity had it moving
    # down fast enough), an artificial discontinuity that doesn't correspond
    # to any real starting condition.
    z = rng.uniform(radius, 3.0)  # include some already-airborne (post-kick) starts, not just grounded
    return Vector3(x, y, z)


def _sample_position_already_special(rng: random.Random, pitch: Pitch, radius: float) -> Vector3:
    """A position that is already out of bounds or already inside a goal
    mouth at t=0 (§4.2's "~10-20% of episodes start already out of pitch
    bounds or already past the goal line"). ``radius``: see
    ``_sample_position_in_play``'s docstring -- z is floored at the ball's
    own radius here too, for the same reason."""
    half_goal_w = pitch.goal_width_m / 2.0
    side = 1.0 if rng.random() < 0.5 else -1.0
    if rng.random() < 0.5:
        # Already inside a goal mouth (goal_scored from tick 1).
        x = side * (pitch.half_length + rng.uniform(0.1, max(0.2, pitch.goal_depth_m * 0.8)))
        y = rng.uniform(-half_goal_w * 0.8, half_goal_w * 0.8)
        z = rng.uniform(radius, pitch.goal_height_m * 0.8)
    elif rng.random() < 0.5:
        # Out via the x boundary, outside the goal mouth width (out_of_bounds, not a goal).
        x = side * (pitch.half_length + rng.uniform(0.2, 5.0))
        y_side = 1.0 if rng.random() < 0.5 else -1.0
        y = y_side * rng.uniform(half_goal_w + 0.5, pitch.half_width + 3.0)
        z = rng.uniform(radius, 3.0)
    else:
        # Out via the y (side) boundary.
        x = rng.uniform(-pitch.half_length, pitch.half_length)
        y = side * (pitch.half_width + rng.uniform(0.2, 5.0))
        z = rng.uniform(radius, 3.0)
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


def compute_engineered_features(pos_vel_spin: np.ndarray) -> np.ndarray:
    """Vectorized computation of input fields 14-19 (speed_norm,
    speed_norm_sq, spin_norm, magnus_cross_x/y/z) from an already-normalized
    ``(N, 9)`` pos/vel/spin block (fields 0-8's layout -- position isn't
    actually used, only vel[3:6]/spin[6:9], but the full 9-wide block is
    accepted since that's the natural slice both callers already have).

    Shared by ``_encode_input`` (single-row, freshly-SAMPLED state) and
    ``ball_dataset.py``'s adjacent-horizon-pair reconstruction (batched,
    RECORDED state at a horizon reused as a new pseudo-initial-state) --
    kept as one function so the two paths can't drift apart. See
    ``_encode_input``'s inline comments for the physics rationale behind
    each feature (drag ∝ speed^2, no spin^2 term exists in the real
    physics, Magnus force ∝ spin x velocity).
    """
    vx, vy, vz = pos_vel_spin[..., 3], pos_vel_spin[..., 4], pos_vel_spin[..., 5]
    sx, sy, sz = pos_vel_spin[..., 6], pos_vel_spin[..., 7], pos_vel_spin[..., 8]
    speed_norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    out = np.zeros(pos_vel_spin.shape[:-1] + (6,), dtype=pos_vel_spin.dtype)
    out[..., 0] = speed_norm
    out[..., 1] = speed_norm ** 2
    out[..., 2] = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
    out[..., 3] = sy * vz - sz * vy  # spin x velocity, x component
    out[..., 4] = sz * vx - sx * vz  # y component
    out[..., 5] = sx * vy - sy * vx  # z component
    return out


def _kinematics_divisors(pitch: Pitch, params: BallEpisodeGenParams) -> tuple[float, float, float, float, float]:
    """Returns ``(div_x, div_y, div_z, div_vel, div_spin)`` -- the divisors
    used to encode position/velocity/spin in ``_encode_input``/``_encode_
    target``, chosen per ``params.normalize_kinematics_by_base_pitch`` (see
    its config comment in ai_config.json). ``div_spin`` is always the fixed
    ``BALL_SPIN_NORM_DIVISOR_RAD_S`` module constant -- spin was never
    per-episode-scaled to begin with, and this divisor is deliberately NOT
    ``params.ball_spin_norm_max_rad_s`` (see that constant's docstring for
    why). ``div_vel`` is always ``half_diag``, one shared
    divisor across x/y/z (unaffected by this flag either way -- only WHICH
    pitch's half_diag it is changes).

    When false (prior, still-default behaviour): ``div_x``/``div_y``
    use the EPISODE's own randomized pitch (``pitch.half_length``/
    ``half_width``), ``div_z`` is the fixed ``height_norm_m`` constant --
    position keeps 3 DIFFERENT per-axis divisors, same as it always has.

    When true: ``div_x``/``div_y``/``div_z`` are ALL the FIXED base pitch's
    ``half_diag`` -- one shared divisor across all 3 position axes, matching
    velocity's convention (rather than height keeping its own much smaller
    ``height_norm_m`` scale, which would make height errors ~20x smaller in
    normalized space than under the per-axis scheme and barely register in
    pos_mse/pos_rmse).
    """
    if params.normalize_kinematics_by_base_pitch:
        half_length = params.base_pitch_length_m / 2
        half_width = params.base_pitch_width_m / 2
        half_diag = math.hypot(half_length, half_width)
        return half_diag, half_diag, half_diag, half_diag, BALL_SPIN_NORM_DIVISOR_RAD_S
    half_length = pitch.half_length
    half_width = pitch.half_width
    half_diag = math.hypot(half_length, half_width)
    return half_length, half_width, params.height_norm_m, half_diag, BALL_SPIN_NORM_DIVISOR_RAD_S


def _encode_input(
    ball: Ball, pitch: Pitch, restitution: float, params: BallEpisodeGenParams,
) -> np.ndarray:
    div_x, div_y, div_z, div_vel, div_spin = _kinematics_divisors(pitch, params)
    row = np.zeros(N_INPUT_FIELDS, dtype=np.float32)
    row[0] = ball.position.x / div_x
    row[1] = ball.position.y / div_y
    row[2] = ball.position.z / div_z
    row[3] = ball.velocity.x / div_vel
    row[4] = ball.velocity.y / div_vel
    row[5] = ball.velocity.z / div_vel
    row[6] = ball.spin.x / div_spin
    row[7] = ball.spin.y / div_spin
    row[8] = ball.spin.z / div_spin
    row[9] = restitution
    # Pitch/goal dims are always a ratio-to-base, regardless of
    # normalize_kinematics_by_base_pitch -- this is a separate, always-
    # exact, always-available (never needs bottleneck recovery) feature
    # describing how THIS episode's pitch compares to the standard one,
    # not part of the pos/vel/spin "kinematics" that flag controls.
    row[10] = pitch.length_m / params.base_pitch_length_m
    row[11] = pitch.width_m / params.base_pitch_width_m
    row[12] = pitch.goal_width_m / params.base_goal_width_m
    row[13] = pitch.goal_height_m / params.base_goal_height_m
    # Engineered features: air drag is proportional to speed^2, and only
    # the raw vx/vy/vz components were available above -- computing that
    # magnitude is a nonlinear (sqrt-of-sum-of-squares, then square)
    # combination across 3 inputs that a small net would otherwise have to
    # learn indirectly. Handing it over directly is free (deterministic
    # function of row[3:6], no new information) and removes that burden.
    # Both raw speed and speed^2 are included since they're each useful for
    # different things (speed^2 maps directly onto drag magnitude; plain
    # speed is still relevant elsewhere, e.g. bounce/restitution behaviour).
    # spin_norm: magnitude only, no squared counterpart -- checked the real
    # physics model (ball_physics.py) and nothing there is proportional to
    # spin^2. Spin decay is a plain exponential (linear in spin) and the
    # Magnus force below is linear in spin too, so spin^2 wouldn't map onto
    # anything real the way speed^2 maps onto drag.
    # Magnus term: the real physics model computes force ∝ spin x velocity
    # (ball_physics.py's magnus_force, a cross product -- bilinear in spin
    # and velocity, considerably harder for a small net to learn from the 6
    # separate raw components than a squared magnitude is). Feeding the
    # cross product's 3 components directly (not just its magnitude) keeps
    # the DIRECTION information, which is what determines which way the
    # ball actually curves.
    row[14:20] = compute_engineered_features(row[0:9])
    return row


def _encode_target(
    position: Vector3, velocity: Vector3, spin: Vector3,
    out_of_bounds: bool, goal_scored: bool,
    pitch: Pitch, params: BallEpisodeGenParams,
) -> np.ndarray:
    div_x, div_y, div_z, div_vel, div_spin = _kinematics_divisors(pitch, params)
    row = np.zeros(N_TARGET_FIELDS_PER_HORIZON, dtype=np.float32)
    row[0] = position.x / div_x
    row[1] = position.y / div_y
    row[2] = position.z / div_z
    row[3] = velocity.x / div_vel
    row[4] = velocity.y / div_vel
    row[5] = velocity.z / div_vel
    row[6] = spin.x / div_spin
    row[7] = spin.y / div_spin
    row[8] = spin.z / div_spin
    row[9] = 1.0 if out_of_bounds else 0.0
    row[10] = 1.0 if goal_scored else 0.0
    return row


def generate_episode(
    rng: random.Random, params: BallEpisodeGenParams | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Draws one random ball episode and returns ``(input[14], target[55],
    crossing[11], crossing_time_s)``.

    Target is 5 horizons (``params.horizons_s``, in that order) x 11
    fields, flattened -- see §3.2. Physics NEVER freezes/stops early: the
    ball is simulated with the SAME regular physics (gravity/drag/Magnus/
    ground bounce via ``step_ball``, and goal-net/post/crossbar bounce
    physics via ``resolve_goal_boundary``) for the FULL horizon regardless
    of out-of-bounds/goal-scored status -- a ball that clips back onto the
    pitch, or bounces back out of the goal mouth off the net/frame, keeps
    being simulated exactly like that, not artificially held in place.

    ``out_of_bounds``/``goal_scored`` at each recorded horizon are the
    INSTANTANEOUS state at that exact moment (NOT latched/sticky) -- if
    the ball is out of bounds at t=1s but a real bounce brings it back
    onto the pitch by t=2s, horizon t=2s's ``out_of_bounds`` is 0 again.
    This is a deliberate change from the old "freeze and latch forever"
    semantics: since physics keeps running and can genuinely reverse
    these conditions, the flags now just describe "is this true right
    now," per horizon, like every other recorded field.

    ``crossing`` is a separate 11-field row (same layout as one
    per-horizon target block) capturing the FULL state (pos/vel/spin) at
    the moment out_of_bounds/goal_scored FIRST became true (or the final
    simulated state, with both flags 0, if neither ever happened within
    ``max(horizons_s)``); ``crossing_time_s`` is the simulated time that
    happened at (``math.inf`` if it never happened). Together, these are
    what a later, TRAINING-time flag can use to reconstruct the OLD
    freeze-on-event semantics from this same, always-continuous dataset
    (substitute ``crossing`` into any horizon at or after ``crossing_
    time_s``) -- without needing to regenerate the dataset under a
    different convention just to compare the two. The time is needed
    alongside the state because, unlike the old semantics, the per-horizon
    flags are no longer latched -- "horizon h's own flag is set" no longer
    implies "a crossing happened by horizon h", since it can legitimately
    un-set again (see above).

    Simulates chronologically to each horizon's EXACT time (regular
    ``sim_dt_s``-sized steps, then a final partial step of whatever
    remainder is left), not the nearest whole physics tick -- so the
    recorded state is the real physics state at that literal
    ``horizons_s`` value, never off by up to half a tick the way naively
    rounding ``horizon_s / dt`` to the nearest integer tick would be
    (worth up to ~10% of the horizon itself at the shortest ones).
    """
    params = params or BallEpisodeGenParams.from_config()
    pitch = _sample_pitch(rng, params)
    restitution = _sample_restitution(rng, params)
    phys_params = replace(BallPhysicsParams.from_config(), bounce_restitution_vertical=restitution)

    start_special = rng.random() < params.out_of_bounds_start_frac
    position = (
        _sample_position_already_special(rng, pitch, params.ball_radius_m) if start_special
        else _sample_position_in_play(rng, pitch, params.ball_radius_m)
    )
    ball = Ball(
        position=position,
        velocity=_sample_velocity(rng, params.ball_speed_max_mps),
        spin=_sample_spin(rng, params.ball_spin_norm_max_rad_s, params.spin_active_frac),
    )

    input_row = _encode_input(ball, pitch, restitution, params)

    dt = params.sim_dt_s
    recorded: list[tuple[Vector3, Vector3, Vector3, bool, bool] | None] = [None] * len(params.horizons_s)
    crossing_state: tuple[Vector3, Vector3, Vector3, bool, bool] | None = None
    crossing_time = math.inf
    sim_time = 0.0
    oob_now = False
    goal_now = False

    def _step_and_check(dt_step: float) -> None:
        nonlocal sim_time, oob_now, goal_now, crossing_state, crossing_time
        step_ball(ball, dt_step, phys_params)
        if params.goal_net_collisions_enabled:
            resolve_goal_boundary(ball, pitch, phys_params)
        sim_time += dt_step
        oob_now = not pitch.is_in_bounds(ball.position)
        goal_now = check_goal(ball, pitch) is not None
        if crossing_state is None and (oob_now or goal_now):
            crossing_state = (ball.position, ball.velocity, ball.spin, oob_now, goal_now)
            crossing_time = sim_time

    for i in sorted(range(len(params.horizons_s)), key=lambda i: params.horizons_s[i]):
        target_time = params.horizons_s[i]
        # Full-dt steps until one more would overshoot the target time...
        while sim_time + dt < target_time - 1e-9:
            _step_and_check(dt)
        # ...then one final partial step of exactly the remainder, so
        # sim_time lands EXACTLY on target_time (not the nearest dt).
        remainder = target_time - sim_time
        if remainder > 1e-9:
            _step_and_check(remainder)
            sim_time = target_time
        recorded[i] = (ball.position, ball.velocity, ball.spin, oob_now, goal_now)

    target_row = np.concatenate([_encode_target(*recorded[i], pitch, params) for i in range(len(params.horizons_s))])
    final_state = crossing_state if crossing_state is not None else (ball.position, ball.velocity, ball.spin, False, False)
    crossing_row = _encode_target(*final_state, pitch, params)
    return input_row, target_row, crossing_row, crossing_time


def generate_shard(
    n_episodes: int, seed: int, params: BallEpisodeGenParams | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates ``n_episodes`` independent episodes with a seeded RNG.

    Returns ``(inputs, targets, crossings, crossing_times)`` of shape
    ``(n_episodes, 20)`` / ``(n_episodes, n_horizons*11)`` /
    ``(n_episodes, 11)`` / ``(n_episodes,)`` -- see ``generate_episode``'s
    docstring for what ``crossings``/``crossing_times`` hold. Each shard's
    episodes are fully independent draws (no shared state), so this is
    safe to call from a separate worker process per shard -- see
    ball_dataset.py's ``generate_dataset()``.
    """
    params = params or BallEpisodeGenParams.from_config()
    rng = random.Random(seed)
    inputs = np.empty((n_episodes, N_INPUT_FIELDS), dtype=np.float32)
    targets = np.empty((n_episodes, len(params.horizons_s) * N_TARGET_FIELDS_PER_HORIZON), dtype=np.float32)
    crossings = np.empty((n_episodes, N_TARGET_FIELDS_PER_HORIZON), dtype=np.float32)
    crossing_times = np.empty((n_episodes,), dtype=np.float64)
    for i in range(n_episodes):
        inputs[i], targets[i], crossings[i], crossing_times[i] = generate_episode(rng, params)
    return inputs, targets, crossings, crossing_times

"""Random player-episode generation for physics pretraining.

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

See agent_plans/player_physics_pretrain_plan.md for the full spec -- this
mirrors ``ball_episode_gen.py`` (see agent_plans/ball_physics_pretrain_plan.md
section 4) as closely as the domain allows. Key differences from the ball
version:

- A player needs an ongoing "intent" (``desired_direction``/``speed_mode``)
  to move at all, unlike the ball. Kept as close to the ball's
  one-random-draw-then-deterministic-physics shape as possible: intent is
  drawn ONCE at t=0 (independently of the player's own starting
  velocity/heading -- a player can be moving one way while WANTING to go
  another, e.g. mid-turn) and held FIXED for the entire episode, never
  resampled at any horizon boundary -- see ``_sample_intent``/
  ``generate_episode``. Earlier drafts of this pipeline resampled intent at
  each horizon boundary (piecewise-constant per segment); that made every
  horizon beyond the first genuinely unpredictable from the t=0 input alone
  (the target depended on intent draws the network never saw), which isn't
  a task a self-supervised encoder can actually learn to solve -- fixing
  intent for the whole episode makes the target a deterministic function of
  the input again, exactly like the ball task.
- No freeze-on-event rule: ``out_of_bounds``/``goal_scored`` at each horizon
  are simply the INSTANTANEOUS state at that exact moment (matching the
  ball pipeline's CURRENT, non-latched convention -- see
  agent_plans/ball_physics_pretrain_plan.md's ``freeze_semantics`` config
  key, which now defaults to false). Physics is never frozen/stopped early.
  A ``crossings``/``crossing_times`` side-channel IS recorded alongside the
  per-horizon targets (mirroring ``ball_episode_gen``'s), but purely to
  supervise ``PlayerDynamicsAutoencoder.crossing_head`` -- NOT to reconstruct
  freeze semantics, which this pipeline has never had.
- No spin/height (z) axis -- players are grounded (z=0 always), and there's
  no equivalent of ball spin.

One "episode" = one random ``Player`` state + one random initial intent,
stepped forward under the real ``step_player_towards()``/``drain_if_
sprinting()``/``regen_stamina()`` per-tick update (the same update order
``engine/match.py``'s ``_apply_movement``/``_update_state_timers`` use for a
live player) for ``max(horizons_s)`` seconds, recorded at each horizon. No
``Match``/AI/order machinery is needed -- ``step_player_towards`` only
touches a ``Player`` + ``MovementParams``, and ``Pitch.is_in_bounds``/
``is_goal`` only touch a plain ``Vector3``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from footballcoach.config import load_physics_config, require_section
from footballcoach.engine.movement import (
    MovementParams,
    SpeedMode,
    angle_diff,
    drain_if_sprinting,
    effective_top_speed,
    regen_stamina,
    step_player_towards,
)
from footballcoach.entities.attributes import PlayerAttributes
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Player, Team
from footballcoach.mathutils import Vector3

# See the field-layout table in agent_plans/player_physics_pretrain_plan.md:
# 7 identity-shortcut fields (pos_x, pos_y, vel_x, vel_y, heading_sin,
# heading_cos, stamina) + 14 additional context fields (top_speed,
# acceleration, stamina_attr, ball_control, has_possession, desired_
# direction_sin/cos, speed_mode one-hot x3, pitch_length/width/goal_width/
# goal_height_norm) + 3 engineered features (speed_norm, heading_desired_
# diff_sin/cos) = 24.
N_INPUT_FIELDS = 24
# 7 identity fields (same order/normalization as the input's identity block)
# + 2 classification fields (out_of_bounds, goal_scored) = 9.
N_TARGET_FIELDS_PER_HORIZON = 9

# Raw input fields 0:7 (pos_x, pos_y, vel_x, vel_y, heading_sin, heading_cos,
# stamina) and target fields 0:7 use the exact same normalization/order (see
# _encode_input/_encode_target below), so an identity mapping between them
# is meaningful -- this is what the identity-shortcut init exploits (see
# player_dynamics_net.py).
N_IDENTITY_SHORTCUT_FIELDS = 7

# Index of the current-stamina-fraction field within the 7 identity-shortcut
# fields above. Unlike the other 6 (position x/y, velocity x/y, heading
# sin/cos, each of which can be either sign), stamina is PROVABLY confined to
# [0, 1] (drain_stamina/regen_stamina both clamp via max(0.0, min(1.0, ...)))
# -- see player_dynamics_net.py / identity_shortcut.py's ``nonneg_field_
# index`` for what this buys (the ball pipeline's exact same trick, applied
# to its own provably-non-negative field, pos_z).
STAMINA_FIELD_INDEX = 6


@dataclass(frozen=True)
class PlayerEpisodeGenParams:
    """Randomization ranges + normalization constants for one episode draw.

    All values sourced from ``ai_config.json["physics_pretrain"]["player"]``/
    ``["observation"]`` and ``physics.json["pitch"/"movement"]`` via
    ``from_config()`` -- never hardcoded, mirroring
    ``BallEpisodeGenParams``'s exact convention.
    """
    horizons_s: tuple[float, ...]
    pitch_scale_range: tuple[float, float]
    out_of_bounds_start_frac: float
    possession_start_frac: float
    speed_mode_weights: tuple[float, float, float]  # (standstill, jog, sprint)
    sim_dt_s: float
    base_pitch_length_m: float
    base_pitch_width_m: float
    base_goal_width_m: float
    base_goal_height_m: float
    normalize_kinematics_by_base_pitch: bool

    @staticmethod
    def from_config() -> "PlayerEpisodeGenParams":
        from footballcoach.ai.config import load_ai_config
        ai_cfg = load_ai_config()
        pp_cfg = ai_cfg["physics_pretrain"]["player"]
        obs_cfg = ai_cfg["observation"]
        phys_cfg = load_physics_config()
        pitch_cfg = require_section(phys_cfg, "pitch")
        return PlayerEpisodeGenParams(
            horizons_s=tuple(float(h) for h in pp_cfg["horizons_s"]),
            pitch_scale_range=tuple(float(s) for s in pp_cfg["pitch_scale_range"]),
            out_of_bounds_start_frac=float(pp_cfg["out_of_bounds_start_frac"]),
            possession_start_frac=float(pp_cfg["possession_start_frac"]),
            speed_mode_weights=tuple(float(w) for w in pp_cfg["speed_mode_weights"]),
            sim_dt_s=float(obs_cfg["sim_dt_s"]),
            base_pitch_length_m=float(pitch_cfg["length_m"]),
            base_pitch_width_m=float(pitch_cfg["width_m"]),
            base_goal_width_m=float(pitch_cfg["goal_width_m"]),
            base_goal_height_m=float(pitch_cfg["goal_height_m"]),
            normalize_kinematics_by_base_pitch=bool(pp_cfg.get("normalize_kinematics_by_base_pitch", False)),
        )


def _sample_pitch(rng: random.Random, params: PlayerEpisodeGenParams) -> Pitch:
    base = Pitch.standard()
    lo, hi = params.pitch_scale_range
    return Pitch(
        length_m=params.base_pitch_length_m * rng.uniform(lo, hi),
        width_m=params.base_pitch_width_m * rng.uniform(lo, hi),
        goal_width_m=params.base_goal_width_m * rng.uniform(lo, hi),
        goal_height_m=params.base_goal_height_m * rng.uniform(lo, hi),
        # Box/six-yard/penalty-spot/centre-circle geometry is irrelevant to
        # player movement or the out-of-bounds/goal checks -- kept at
        # standard values, never read below (same as ball_episode_gen.py's
        # _sample_pitch).
        goal_depth_m=base.goal_depth_m,
        box_length_m=base.box_length_m,
        box_width_m=base.box_width_m,
        six_yard_length_m=base.six_yard_length_m,
        six_yard_width_m=base.six_yard_width_m,
        penalty_spot_distance_m=base.penalty_spot_distance_m,
        centre_circle_radius_m=base.centre_circle_radius_m,
    )


def _sample_position_in_play(rng: random.Random, pitch: Pitch) -> Vector3:
    margin = 1.0
    x = rng.uniform(-pitch.half_length + margin, pitch.half_length - margin)
    y = rng.uniform(-pitch.half_width + margin, pitch.half_width - margin)
    return Vector3(x, y, 0.0)  # players are grounded -- z is always 0


def _sample_position_already_special(rng: random.Random, pitch: Pitch) -> Vector3:
    """A position that is already out of bounds at t=0 (mirrors
    ``ball_episode_gen._sample_position_already_special``, simplified for 2D
    since there's no airborne/height case for a grounded player). Includes
    an "already past the goal line, within the goal mouth" case as a cheap
    adaptation of the ball's "already inside a goal mouth" case -- gives the
    dataset some ``goal_scored=1``-eligible (i.e. has_possession=1, position
    past the goal line within goal-mouth width) starting states without a
    dedicated resampling loop."""
    half_goal_w = pitch.goal_width_m / 2.0
    side = 1.0 if rng.random() < 0.5 else -1.0
    if rng.random() < 0.5:
        # Past the goal line, within the goal mouth width.
        x = side * (pitch.half_length + rng.uniform(0.1, 2.0))
        y = rng.uniform(-half_goal_w * 0.8, half_goal_w * 0.8)
    elif rng.random() < 0.5:
        # Out via the x boundary, outside the goal mouth width.
        x = side * (pitch.half_length + rng.uniform(0.2, 5.0))
        y_side = 1.0 if rng.random() < 0.5 else -1.0
        y = y_side * rng.uniform(half_goal_w + 0.5, pitch.half_width + 3.0)
    else:
        # Out via the y (side) boundary.
        x = rng.uniform(-pitch.half_length, pitch.half_length)
        y = side * (pitch.half_width + rng.uniform(0.2, 5.0))
    return Vector3(x, y, 0.0)


def _sample_speed_mode(rng: random.Random, weights: tuple[float, float, float]) -> "SpeedMode":
    """Weighted categorical draw over ``(STANDSTILL, JOG, SPRINT)`` -- see
    ``physics_pretrain.player.speed_mode_weights``."""
    w_standstill, w_jog, w_sprint = weights
    total = w_standstill + w_jog + w_sprint
    r = rng.uniform(0.0, total)
    if r < w_standstill:
        return SpeedMode.STANDSTILL
    if r < w_standstill + w_jog:
        return SpeedMode.JOG
    return SpeedMode.SPRINT


def _sample_intent(rng: random.Random, params: PlayerEpisodeGenParams) -> tuple[float, "SpeedMode"]:
    """Draws this episode's ONE (direction_rad, speed_mode) intent -- a
    fresh uniform direction and a fresh categorical speed_mode, drawn once
    and held fixed for the whole episode (see the module docstring's
    "Key differences from the ball version" for why: resampling this
    mid-episode would make later horizons depend on information the t=0
    input never carries)."""
    direction_rad = rng.uniform(-math.pi, math.pi)
    speed_mode = _sample_speed_mode(rng, params.speed_mode_weights)
    return direction_rad, speed_mode


def compute_engineered_features(identity_block: np.ndarray, desired_direction_sincos: np.ndarray) -> np.ndarray:
    """Vectorized computation of input fields 21-23 (speed_norm,
    heading_desired_diff_sin, heading_desired_diff_cos) from an
    already-normalized ``(..., 7)`` identity block (fields 0-6's layout:
    pos_x, pos_y, vel_x, vel_y, heading_sin, heading_cos, stamina -- position
    and stamina aren't actually used, only vel[2:4]/heading[4:6], but the
    full 7-wide block is accepted since that's the natural slice callers
    already have, mirroring ``ball_episode_gen.compute_engineered_features``'s
    "accept the natural slice" convention) and an ``(..., 2)`` desired-
    direction sin/cos block.

    Shared by ``_encode_input`` (single-row, freshly-SAMPLED state) and
    ``player_dataset.py``'s adjacent-horizon-pair/autoencode derived-data
    reconstruction (batched, RECORDED state at a horizon reused as a new
    pseudo-initial-state) -- kept as one function so the two paths can't
    drift apart, exactly mirroring the ball version's rationale.

    ``speed_norm``: this IS the ``current_speed``/``speed_mps`` term
    ``step_player_towards`` uses directly in ``turn_fraction``'s denominator
    (``max(current_speed, 0.5)``) and elsewhere -- a nonlinear (sqrt-of-sum-
    of-squares) combination of vel_x/vel_y a small net would otherwise have
    to learn indirectly.

    ``heading_desired_diff_sin/cos``: sin/cos of ``angle_diff(heading_rad,
    desired_direction_angle)`` (see ``engine/movement.py``'s ``angle_diff``)
    -- this angle difference directly drives ``turn_fraction = min(abs(
    heading_diff)/pi, 1)``, which then multiplies ``desired_speed`` every
    tick (``turn_speed_penalty``). Computed via the angle-SUBTRACTION-of-
    sines-and-cosines identity (``sin(b-a) = sin(b)cos(a) - cos(b)sin(a)``,
    ``cos(b-a) = cos(b)cos(a) + sin(b)sin(a)``) directly from the sin/cos
    representations already at hand, rather than round-tripping through
    ``atan2``/raw angles -- exact, vectorizable, and avoids re-deriving an
    angle from a representation that was chosen specifically to avoid
    needing one.
    """
    vx, vy = identity_block[..., 2], identity_block[..., 3]
    heading_sin, heading_cos = identity_block[..., 4], identity_block[..., 5]
    dd_sin, dd_cos = desired_direction_sincos[..., 0], desired_direction_sincos[..., 1]
    speed_norm = np.sqrt(vx ** 2 + vy ** 2)
    diff_sin = dd_sin * heading_cos - dd_cos * heading_sin
    diff_cos = dd_cos * heading_cos + dd_sin * heading_sin
    out = np.zeros(identity_block.shape[:-1] + (3,), dtype=identity_block.dtype)
    out[..., 0] = speed_norm
    out[..., 1] = diff_sin
    out[..., 2] = diff_cos
    return out


def _kinematics_divisors(pitch: Pitch, params: PlayerEpisodeGenParams) -> tuple[float, float, float]:
    """Returns ``(div_x, div_y, div_vel)`` -- the divisors used to encode
    position/velocity in ``_encode_input``/``_encode_target``, chosen per
    ``params.normalize_kinematics_by_base_pitch``. Directly mirrors
    ``ball_episode_gen._kinematics_divisors``, simplified: no height (z)
    axis to special-case, so there's no third position divisor to reconcile
    the way the ball's height/half_diag scale mismatch required -- x/y
    always use ``div_x``/``div_y`` and velocity always uses ``div_vel``
    (one shared divisor across vel_x/vel_y, unaffected by this flag either
    way -- only WHICH pitch's half_diag it is changes).

    When false (default): ``div_x``/``div_y`` use the EPISODE's own
    randomized pitch (``pitch.half_length``/``half_width``) -- position
    keeps 2 DIFFERENT per-axis divisors. When true: ``div_x``/``div_y`` are
    BOTH the FIXED base pitch's ``half_diag`` -- one shared divisor across
    both position axes, matching velocity's convention.
    """
    if params.normalize_kinematics_by_base_pitch:
        half_length = params.base_pitch_length_m / 2
        half_width = params.base_pitch_width_m / 2
        half_diag = math.hypot(half_length, half_width)
        return half_diag, half_diag, half_diag
    half_length = pitch.half_length
    half_width = pitch.half_width
    half_diag = math.hypot(half_length, half_width)
    return half_length, half_width, half_diag


def _encode_input(
    player: Player, pitch: Pitch, params: PlayerEpisodeGenParams,
    has_possession: bool, desired_direction_rad: float, desired_speed_mode: "SpeedMode",
) -> np.ndarray:
    div_x, div_y, div_vel = _kinematics_divisors(pitch, params)
    attrs = player.attributes
    row = np.zeros(N_INPUT_FIELDS, dtype=np.float32)
    row[0] = player.position.x / div_x
    row[1] = player.position.y / div_y
    row[2] = player.velocity.x / div_vel
    row[3] = player.velocity.y / div_vel
    row[4] = math.sin(player.heading_rad)
    row[5] = math.cos(player.heading_rad)
    row[6] = player.stamina
    row[7] = attrs.top_speed
    row[8] = attrs.acceleration
    row[9] = attrs.stamina
    row[10] = attrs.ball_control
    row[11] = 1.0 if has_possession else 0.0
    row[12] = math.sin(desired_direction_rad)
    row[13] = math.cos(desired_direction_rad)
    row[14] = 1.0 if desired_speed_mode is SpeedMode.STANDSTILL else 0.0
    row[15] = 1.0 if desired_speed_mode is SpeedMode.JOG else 0.0
    row[16] = 1.0 if desired_speed_mode is SpeedMode.SPRINT else 0.0
    # Pitch/goal dims are always a ratio-to-base, regardless of
    # normalize_kinematics_by_base_pitch -- same rationale as
    # ball_episode_gen._encode_input's fields 10-13.
    row[17] = pitch.length_m / params.base_pitch_length_m
    row[18] = pitch.width_m / params.base_pitch_width_m
    row[19] = pitch.goal_width_m / params.base_goal_width_m
    row[20] = pitch.goal_height_m / params.base_goal_height_m
    row[21:24] = compute_engineered_features(row[0:7], row[12:14])
    return row


def _encode_target(
    position: Vector3, velocity: Vector3, heading_rad: float, stamina: float,
    out_of_bounds: bool, goal_scored: bool,
    pitch: Pitch, params: PlayerEpisodeGenParams,
) -> np.ndarray:
    div_x, div_y, div_vel = _kinematics_divisors(pitch, params)
    row = np.zeros(N_TARGET_FIELDS_PER_HORIZON, dtype=np.float32)
    row[0] = position.x / div_x
    row[1] = position.y / div_y
    row[2] = velocity.x / div_vel
    row[3] = velocity.y / div_vel
    row[4] = math.sin(heading_rad)
    row[5] = math.cos(heading_rad)
    row[6] = stamina
    row[7] = 1.0 if out_of_bounds else 0.0
    row[8] = 1.0 if goal_scored else 0.0
    return row


def generate_episode(
    rng: random.Random, params: PlayerEpisodeGenParams | None = None, movement_params: MovementParams | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Draws one random player episode and returns ``(input[24],
    target[n_horizons*9], crossing[9], crossing_time_s)`` -- see the module
    docstring and agent_plans/player_physics_pretrain_plan.md for the full
    spec.

    Physics NEVER freezes/stops early -- the player is simulated with the
    same regular ``step_player_towards``/stamina-drain/stamina-regen update
    for the FULL horizon regardless of out-of-bounds/goal-scored status,
    matching the ball pipeline's CURRENT (non-latched) convention.
    ``out_of_bounds``/``goal_scored`` at each horizon are the INSTANTANEOUS
    state at that moment. ``goal_scored`` is always ``0.0`` when this
    episode's ``has_possession`` is ``False`` (a non-possessing player
    cannot "score" by walking into the goal mouth under the dribble-carry-
    glued assumption -- see the plan doc).

    Simulates chronologically to each horizon's EXACT time (regular
    ``sim_dt_s``-sized steps, then a final partial step of whatever
    remainder is left), copying ``ball_episode_gen.generate_episode``'s
    technique precisely to avoid up-to-half-a-tick rounding error.

    Intent (``desired_direction``/``speed_mode``) is drawn once at t=0 and
    held FIXED for the whole episode -- see ``_sample_intent``.

    ``crossing`` is a separate 9-field row (same layout/normalization as one
    per-horizon target block, i.e. ``_encode_target``'s output) capturing the
    FULL state at the moment ``out_of_bounds``/``goal_scored`` FIRST became
    true, and ``crossing_time_s`` is the simulated time that happened at.
    When no crossing is recorded, ``crossing`` is the FINAL simulated state
    with both flags 0 and ``crossing_time_s`` is ``math.inf`` -- a sentinel
    the dataset turns into ``crossing_mask=False`` (see
    ``PlayerDynamicsDataset``). Together they supervise
    ``PlayerDynamicsAutoencoder.crossing_head``; unlike the ball pipeline's
    identically-shaped side-channel, they are NOT used to reconstruct
    freeze-on-event semantics (this pipeline has none -- see the module
    docstring).

    Two deliberate rules about WHEN a crossing may be recorded, both
    mirroring/extending ``ball_episode_gen.generate_episode``:

    - The oob/goal check runs only from the FIRST physics step onward, never
      against the raw un-stepped t=0 sampled state (``_step_and_check`` is
      only ever called from inside the per-horizon stepping loop).
    - Episodes that START already out of bounds (``_sample_position_already_
      special``, i.e. ``out_of_bounds_start_frac`` of them) are forced to
      ``crossing_time_s = math.inf`` -- "no crossing" -- regardless of what
      the first-tick-onward tracking would otherwise detect. For such an
      episode ``out_of_bounds`` is essentially already true before any
      physics runs, so the "crossing" the tracker would record is a
      degenerate near-t=0 point rather than the genuinely interesting "this
      in-play player will leave the pitch at (x, y) in dt seconds"
      prediction the head exists to learn. Reusing the existing ``inf``
      sentinel means these rows fall out of ``crossing_mask`` for free,
      exactly like episodes that legitimately never cross -- no extra mask
      field needed. ``crossing``'s own CONTENT is unaffected by this
      override (it's the final simulated state, both flags 0, same as any
      other no-crossing episode).
    """
    params = params or PlayerEpisodeGenParams.from_config()
    movement_params = movement_params or MovementParams.from_config()
    pitch = _sample_pitch(rng, params)

    start_special = rng.random() < params.out_of_bounds_start_frac
    position = (
        _sample_position_already_special(rng, pitch) if start_special
        else _sample_position_in_play(rng, pitch)
    )
    has_possession = rng.random() < params.possession_start_frac

    attrs = PlayerAttributes(
        top_speed=rng.uniform(0.0, 1.0),
        acceleration=rng.uniform(0.0, 1.0),
        stamina=rng.uniform(0.0, 1.0),
        kick_precision=0.5, kick_power=0.5, dribbling=0.5, tackling=0.5,  # not used by movement physics
        ball_control=rng.uniform(0.0, 1.0),
    )
    stamina0 = rng.uniform(0.0, 1.0)

    # Physical-plausibility invariant (see the plan doc's initial-condition
    # sampling section): velocity must always equal speed*(cos(heading),
    # sin(heading)) at t=0, matching what step_player_towards enforces on
    # EVERY subsequent tick -- so heading and speed are sampled together and
    # velocity is DERIVED from them, never sampled independently.
    heading0 = rng.uniform(-math.pi, math.pi)
    v_top0 = effective_top_speed(
        movement_params, attrs.top_speed, stamina0, has_possession, attrs.ball_control, is_goalkeeper=False,
    )
    speed0 = rng.uniform(0.0, v_top0)

    player = Player(
        player_id="physics_pretrain",
        team=Team.LEFT,
        attributes=attrs,
        position=position,
        velocity=Vector3.from_angle_xy(heading0, speed0),
        heading_rad=heading0,
        is_goalkeeper=False,
        stamina=stamina0,
    )

    direction_rad, speed_mode = _sample_intent(rng, params)
    input_row = _encode_input(player, pitch, params, has_possession, direction_rad, speed_mode)

    dt = params.sim_dt_s
    sim_time = 0.0
    # See the docstring: an episode that is ALREADY out of bounds before any
    # physics runs is excluded from the crossing head's supervision entirely
    # (forced to the math.inf "no crossing" sentinel below), so its
    # degenerate near-t=0 "crossing" never becomes a training target.
    started_out_of_bounds = not pitch.is_in_bounds(player.position)
    crossing_state: tuple[Vector3, Vector3, float, float, bool, bool] | None = None
    crossing_time = math.inf

    def _step_and_check(dt_step: float) -> None:
        nonlocal sim_time, crossing_state, crossing_time
        target_direction = Vector3.from_angle_xy(direction_rad, 1.0)
        step_player_towards(player, target_direction, speed_mode, dt_step, movement_params, has_possession)
        # Mirrors engine/match.py's _apply_movement/_update_state_timers
        # per-tick order exactly: drain (SPRINT only, effort=1.0) AND
        # SEPARATELY regen at a passive 0.3x rate, unconditionally, every
        # tick (yes, even while draining -- this is what the real engine
        # does, see the plan doc's domain-grounding section).
        player.stamina = drain_if_sprinting(movement_params, player, speed_mode is SpeedMode.SPRINT, dt_step)
        player.stamina = regen_stamina(movement_params, player.stamina, attrs.stamina, dt_step * 0.3)
        sim_time += dt_step
        # Checked only from the first physics step onward, never against the
        # raw un-stepped t=0 sampled state -- copied deliberately from
        # ball_episode_gen._step_and_check (see the docstring).
        oob_now = not pitch.is_in_bounds(player.position)
        goal_now = has_possession and pitch.is_goal(player.position) is not None
        if crossing_state is None and (oob_now or goal_now):
            crossing_state = (
                player.position, player.velocity, player.heading_rad, player.stamina, oob_now, goal_now,
            )
            crossing_time = sim_time

    horizons_sorted = sorted(range(len(params.horizons_s)), key=lambda i: params.horizons_s[i])
    recorded: list[np.ndarray | None] = [None] * len(params.horizons_s)
    for i in horizons_sorted:
        target_time = params.horizons_s[i]
        while sim_time + dt < target_time - 1e-9:
            _step_and_check(dt)
        remainder = target_time - sim_time
        if remainder > 1e-9:
            _step_and_check(remainder)
            sim_time = target_time
        out_of_bounds = not pitch.is_in_bounds(player.position)
        goal_scored = has_possession and pitch.is_goal(player.position) is not None
        recorded[i] = _encode_target(
            player.position, player.velocity, player.heading_rad, player.stamina,
            out_of_bounds, goal_scored, pitch, params,
        )

    target_row = np.concatenate(recorded)
    if started_out_of_bounds:
        crossing_state, crossing_time = None, math.inf
    final_state = (
        crossing_state if crossing_state is not None
        else (player.position, player.velocity, player.heading_rad, player.stamina, False, False)
    )
    crossing_row = _encode_target(*final_state, pitch, params)
    return input_row, target_row, crossing_row, crossing_time


def generate_shard(
    n_episodes: int, seed: int, params: PlayerEpisodeGenParams | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates ``n_episodes`` independent episodes with a seeded RNG.

    Returns ``(inputs, targets, crossings, crossing_times)`` of shape
    ``(n_episodes, 24)`` / ``(n_episodes, n_horizons*9)`` /
    ``(n_episodes, 9)`` / ``(n_episodes,)`` -- see ``generate_episode``'s
    docstring for what ``crossings``/``crossing_times`` hold.
    ``crossing_times`` is float64 so it can carry the ``math.inf`` "never
    crossed (or excluded)" sentinel. Each shard's episodes are fully
    independent draws (no shared state), so this is safe to call from a
    separate worker process per shard -- see player_dataset.py's
    ``generate_dataset()``.
    """
    params = params or PlayerEpisodeGenParams.from_config()
    movement_params = MovementParams.from_config()
    rng = random.Random(seed)
    inputs = np.empty((n_episodes, N_INPUT_FIELDS), dtype=np.float32)
    targets = np.empty((n_episodes, len(params.horizons_s) * N_TARGET_FIELDS_PER_HORIZON), dtype=np.float32)
    crossings = np.empty((n_episodes, N_TARGET_FIELDS_PER_HORIZON), dtype=np.float32)
    crossing_times = np.empty((n_episodes,), dtype=np.float64)
    for i in range(n_episodes):
        inputs[i], targets[i], crossings[i], crossing_times[i] = generate_episode(rng, params, movement_params)
    return inputs, targets, crossings, crossing_times

"""Real-accuracy check for the frozen ball/player physics-dynamics
encoders' linear-decoder horizon predictions (see ai/models/
physics_encoders.py's BallPhysicsFeatureBlock/PlayerPhysicsFeatureBlock),
using the ACTUAL PRODUCTION inference path -- not anything from
ai/physics_pretrain/ball_episode_gen.py or the training scripts themselves.

Unlike test_physics_encoder_wiring.py (which builds tiny synthetic
checkpoints on the fly to test wiring/shapes/gradient-isolation, portable
across CI), this file needs a REAL trained checkpoint to say anything
meaningful about accuracy -- checkpoints/physics_pretrain/ is gitignored
(not portable), so every test here SKIPS (not fails) when the checkpoint
it needs isn't present on disk. Treat this as a local/opt-in regression
check: run it after retraining either encoder to confirm live inference
still tracks real physics, not as a CI gate.

Scenario shape matches what the checkpoints were actually trained on
(ball_episode_gen.py/player_episode_gen.py's own generate_episode()
docstrings), but reimplements neither -- everything here (Match/Player/
Ball, step_player_towards, step_ball, encode_observation,
BallPhysicsFeatureBlock/PlayerPhysicsFeatureBlock) is real production code:

  - Ball: generate_episode() simulates continuous, UNTOUCHED physics for
    the whole horizon (no further player contact) -- so this test places
    the ball loose, far from both players, and lets real Match.step()
    physics run it forward with nobody able to reach it. (An earlier,
    cruder version of this check queried mid-dribble states from a real
    Phase1RulesAI match and saw errors up to ~8-28m by the 5-10s horizons
    -- turned out to be entirely a test artifact: the ball was getting
    re-kicked repeatedly within the prediction window, a task the model
    was never trained for. Confirmed by filtering to genuinely untouched
    windows, which recovered accuracy in line with training-time numbers.)
  - Player: generate_episode()'s own docstring: "Intent (desired_direction/
    speed_mode) is drawn once at t=0 and held FIXED for the whole
    episode." So this test gives the trainee no AI/order at all and
    manually re-asserts one fixed direction/speed_mode every physics tick
    (mirroring that scenario shape, not its code), letting real
    Match._apply_movement()/step_player_towards do the actual moving.

Tolerances below are calibrated from real measurements against these two
checkpoints (see this file's own history/PR for the raw numbers) with
roughly 1.5-2.5x margin -- loose enough not to be flaky across the handful
of seeds each test runs, tight enough to catch a real regression (the
desired_direction-not-unit bug this suite was written to catch a repeat of
inflated horizon error by 1-2 orders of magnitude, not a factor of 2).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from footballcoach.ai.models.physics_encoders import BallPhysicsFeatureBlock, PlayerPhysicsFeatureBlock
from footballcoach.ai.obs.encoder import encode_observation
from footballcoach.ai.physics_pretrain.live_encoder_features import load_physics_pitch_constants
from footballcoach.engine.movement import SpeedMode
from footballcoach.mathutils import Vector3
from footballcoach.ui.scenarios import build_1v1_scenario

BALL_CKPT = Path("checkpoints/physics_pretrain/ball_encoder_70.midtrain_latest.pt")
PLAYER_CKPT = Path("checkpoints/physics_pretrain/player_encoder_45.midtrain_latest.pt")
TRAINEE_ID = "trainee"
SIM_DT_S = 0.06
N_BALL_FIELDS = 6
N_PLAYER_FIELDS = 4
N_SEEDS = 8

# Per-horizon mean-error tolerances (metres), see module docstring.
BALL_TOLERANCE_M = {
    0.2: 0.5, 0.5: 0.6, 1.0: 0.8, 2.0: 1.2,
    3.5: 1.8, 5.0: 2.2, 7.0: 2.5, 10.0: 3.0,
}
PLAYER_TOLERANCE_M = {
    0.2: 1.0, 0.5: 2.5, 0.8: 5.0, 1.0: 7.0,
    2.0: 15.0, 3.0: 25.0, 5.0: 40.0, 7.0: 55.0, 10.0: 70.0,
}

_needs_ball_ckpt = pytest.mark.skipif(
    not BALL_CKPT.exists(), reason=f"no trained ball checkpoint at {BALL_CKPT} (gitignored, local-only)",
)
_needs_player_ckpt = pytest.mark.skipif(
    not PLAYER_CKPT.exists(), reason=f"no trained player checkpoint at {PLAYER_CKPT} (gitignored, local-only)",
)


def _base_half_diag() -> float:
    pitch = load_physics_pitch_constants()
    return math.hypot(pitch.base_pitch_length_m / 2.0, pitch.base_pitch_width_m / 2.0)


def _run_ball_case(block: BallPhysicsFeatureBlock, seed: int, start_z: float, start_vel: Vector3, horizons_s):
    """Ball loose, far from both (AI-less) players, real Match.step()
    physics for max(horizons_s). Returns {horizon_s: error_m}."""
    match = build_1v1_scenario(seed=seed, opponent_immobile_prob=1.0, sim_dt_s=SIM_DT_S)
    match.player_by_id(TRAINEE_ID).ai = None
    match.player_by_id("opponent").ai = None
    match.player_by_id(TRAINEE_ID).position = Vector3(-50.0, -30.0, 0.0)
    match.player_by_id("opponent").position = Vector3(-50.0, 30.0, 0.0)
    match.ball.position = Vector3(0.0, 0.0, start_z)
    match.ball.velocity = start_vel
    match.ball.spin = Vector3.zero()

    obs = encode_observation(match=match, player_id=TRAINEE_ID, time_remaining_s=60.0, phase=1)
    td = obs.to_torch_dict()
    with torch.no_grad():
        out = block(td["ball_feat"].unsqueeze(0), td["global_feat"].unsqueeze(0))
    pred = out[:, -block.linear_decoder.unpadded_output_dim:]
    base_half_diag = _base_half_diag()

    n_ticks = int(max(horizons_s) / SIM_DT_S) + 2
    traj = [(match.ball.position.x, match.ball.position.y)]
    for _ in range(n_ticks):
        match.step()
        traj.append((match.ball.position.x, match.ball.position.y))
        assert match.ball.possessed_by is None, "ball must stay loose/untouched for this check"

    errs = {}
    for i, h in enumerate(horizons_s):
        future_tick = round(h / SIM_DT_S)
        p = pred[0, i * N_BALL_FIELDS: i * N_BALL_FIELDS + 2] * base_half_diag
        real_x, real_y = traj[future_tick]
        errs[h] = math.hypot(float(p[0]) - real_x, float(p[1]) - real_y)
    return errs


def _run_player_case(block: PlayerPhysicsFeatureBlock, seed: int, direction_deg: float, speed_mode, horizons_s):
    """Trainee with no AI/order, one FIXED direction/speed_mode re-asserted
    every physics tick, real Match.step() physics for max(horizons_s).
    Returns {horizon_s: error_m}."""
    match = build_1v1_scenario(seed=seed, opponent_immobile_prob=1.0, sim_dt_s=SIM_DT_S)
    trainee = match.player_by_id(TRAINEE_ID)
    trainee.ai = None
    trainee.current_order = None
    match.player_by_id("opponent").ai = None
    match.player_by_id("opponent").position = Vector3(-50.0, 30.0, 0.0)
    trainee.position = Vector3(0.0, 0.0, 0.0)
    trainee.velocity = Vector3.zero()
    rad = math.radians(direction_deg)
    fixed_dir = Vector3(math.cos(rad), math.sin(rad), 0.0)
    trainee.heading_rad = rad
    trainee.desired_direction = fixed_dir
    trainee.desired_speed_mode = speed_mode
    trainee.last_desired_speed_mode = speed_mode

    obs = encode_observation(match=match, player_id=TRAINEE_ID, time_remaining_s=60.0, phase=1)
    td = obs.to_torch_dict()
    with torch.no_grad():
        out = block(td["self_feat"].unsqueeze(0), td["global_feat"].unsqueeze(0))
    pred = out[:, -block.linear_decoder.unpadded_output_dim:]
    base_half_diag = _base_half_diag()

    n_ticks = int(max(horizons_s) / SIM_DT_S) + 2
    traj = [(trainee.position.x, trainee.position.y)]
    for _ in range(n_ticks):
        # desired_speed_mode is cleared to None every tick once consumed
        # (Player.desired_speed_mode's own docstring) -- re-assert every
        # tick to hold the SAME fixed intent for the whole episode, exactly
        # matching player_episode_gen.py's documented scenario shape.
        trainee.desired_direction = fixed_dir
        trainee.desired_speed_mode = speed_mode
        match.step()
        traj.append((trainee.position.x, trainee.position.y))

    errs = {}
    for i, h in enumerate(horizons_s):
        future_tick = round(h / SIM_DT_S)
        p = pred[0, i * N_PLAYER_FIELDS: i * N_PLAYER_FIELDS + 2] * base_half_diag
        real_x, real_y = traj[future_tick]
        errs[h] = math.hypot(float(p[0]) - real_x, float(p[1]) - real_y)
    return errs


@_needs_ball_ckpt
def test_ball_horizon_predictions_track_real_untouched_physics():
    block = BallPhysicsFeatureBlock(str(BALL_CKPT))
    assert block.linear_decoder is not None, "checkpoint must have linear_decoder_enabled=true for this check"
    horizons_s = list(block.linear_decoder._horizons[1:])

    agg = {h: [] for h in horizons_s}
    for seed in range(1, N_SEEDS + 1):
        z = 0.11 if seed % 2 == 0 else 1.5  # mix grounded / airborne starts
        vel = Vector3((seed * 1.37) % 10 - 5, (seed * 2.11) % 10 - 5, 2.0 if z > 0.5 else 0.0)
        errs = _run_ball_case(block, seed, z, vel, horizons_s)
        for h, e in errs.items():
            agg[h].append(e)

    failures = []
    for h in horizons_s:
        mean_err = sum(agg[h]) / len(agg[h])
        tol = BALL_TOLERANCE_M[h]
        if mean_err > tol:
            failures.append(f"h={h}s: mean_err={mean_err:.3f}m > tolerance={tol}m")
    assert not failures, "Ball horizon prediction accuracy regressed:\n" + "\n".join(failures)


@_needs_player_ckpt
def test_player_horizon_predictions_track_real_fixed_intent_physics():
    block = PlayerPhysicsFeatureBlock(str(PLAYER_CKPT))
    assert block.linear_decoder is not None, "checkpoint must have linear_decoder_enabled=true for this check"
    horizons_s = list(block.linear_decoder._horizons[1:])

    agg = {h: [] for h in horizons_s}
    for seed in range(1, N_SEEDS + 1):
        direction_deg = (seed * 37) % 360
        speed_mode = [SpeedMode.JOG, SpeedMode.SPRINT, SpeedMode.STANDSTILL][seed % 3]
        errs = _run_player_case(block, seed, direction_deg, speed_mode, horizons_s)
        for h, e in errs.items():
            agg[h].append(e)

    failures = []
    for h in horizons_s:
        mean_err = sum(agg[h]) / len(agg[h])
        tol = PLAYER_TOLERANCE_M[h]
        if mean_err > tol:
            failures.append(f"h={h}s: mean_err={mean_err:.3f}m > tolerance={tol}m")
    assert not failures, "Player horizon prediction accuracy regressed:\n" + "\n".join(failures)

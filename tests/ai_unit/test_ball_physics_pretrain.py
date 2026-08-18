"""Unit tests for the ball-dynamics physics pretraining pipeline.

See agent_plans/ball_physics_pretrain_plan.md section 10. This suite only
covers the standalone pipeline (episode generation, dataset, network,
training loop) -- there is no live-network integration to test yet (see the
plan's section 8, not implemented).
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from footballcoach.ai.physics_pretrain.ball_episode_gen import (
    N_INPUT_FIELDS,
    N_TARGET_FIELDS_PER_HORIZON,
    BallEpisodeGenParams,
    generate_episode,
    generate_shard,
)
from footballcoach.ai.physics_pretrain.ball_dataset import BallDynamicsDataset, generate_dataset
from footballcoach.ai.physics_pretrain.ball_dynamics_net import (
    BallDynamicsAutoencoder,
    BallDynamicsDecoder,
    BallDynamicsEncoder,
)
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.mathutils import Vector3


@pytest.fixture(scope="module")
def gen_params() -> BallEpisodeGenParams:
    return BallEpisodeGenParams.from_config()


# ---------------------------------------------------------------------------
# ball_episode_gen
# ---------------------------------------------------------------------------

def test_generate_episode_shapes(gen_params):
    rng = random.Random(0)
    input_row, target_row = generate_episode(rng, gen_params)
    assert input_row.shape == (N_INPUT_FIELDS,)
    assert target_row.shape == (len(gen_params.horizons_s) * N_TARGET_FIELDS_PER_HORIZON,)
    assert np.isfinite(input_row).all()
    assert np.isfinite(target_row).all()


def test_generate_episode_deterministic_given_seed(gen_params):
    in1, tgt1 = generate_episode(random.Random(42), gen_params)
    in2, tgt2 = generate_episode(random.Random(42), gen_params)
    np.testing.assert_array_equal(in1, in2)
    np.testing.assert_array_equal(tgt1, tgt2)


def test_generate_episode_different_seeds_differ(gen_params):
    in1, _ = generate_episode(random.Random(1), gen_params)
    in2, _ = generate_episode(random.Random(2), gen_params)
    assert not np.array_equal(in1, in2)


def test_freeze_on_out_of_bounds():
    """A ball already well outside pitch bounds, at rest, must freeze at its
    own (unchanging) state and latch out_of_bounds=1 at every horizon, with
    goal_scored=0 (it's out via the side, not in the goal mouth)."""
    params = BallEpisodeGenParams.from_config()
    pitch = Pitch.standard()
    # Far outside the pitch on the y axis, well clear of any goal mouth.
    ball = Ball(
        position=Vector3(0.0, pitch.half_width + 20.0, 0.0),
        velocity=Vector3.zero(),
        spin=Vector3.zero(),
    )
    assert not pitch.is_in_bounds(ball.position)

    from dataclasses import replace as _replace
    from footballcoach.engine.ball_physics import BallPhysicsParams, resolve_goal_boundary, step_ball
    from footballcoach.engine.scoring import check_goal
    from footballcoach.ai.physics_pretrain.ball_episode_gen import _encode_input, _encode_target

    phys_params = BallPhysicsParams.from_config()
    input_row = _encode_input(ball, pitch, phys_params.bounce_restitution_vertical, params)

    dt = params.sim_dt_s
    horizon_ticks = [max(1, round(h / dt)) for h in params.horizons_s]
    max_tick = max(horizon_ticks)
    frozen = None
    out_of_bounds = False
    goal_scored = False
    recorded = {}
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
        if tick in set(horizon_ticks):
            recorded[tick] = frozen if frozen is not None else (ball.position, ball.velocity, ball.spin)
            recorded[tick] = (*recorded[tick], out_of_bounds, goal_scored)

    assert frozen is not None
    frozen_pos = frozen[0]
    for t in horizon_ticks:
        pos, vel, spin, oob, goal = recorded[t]
        assert oob is True
        assert goal is False
        assert pos.x == pytest.approx(frozen_pos.x)
        assert pos.y == pytest.approx(frozen_pos.y)
        assert pos.z == pytest.approx(frozen_pos.z)


def test_freeze_on_goal():
    """A ball starting inside the goal mouth must latch goal_scored=1 (and
    out_of_bounds=1, since inside-the-goal is also outside pitch bounds by
    Pitch.is_in_bounds's plain x/y-range check) at every horizon."""
    params = BallEpisodeGenParams.from_config()
    pitch = Pitch.standard()
    ball = Ball(
        position=Vector3(pitch.half_length + 0.5, 0.0, 0.5),
        velocity=Vector3.zero(),
        spin=Vector3.zero(),
    )
    assert pitch.is_goal(ball.position) == "right"

    from footballcoach.engine.ball_physics import BallPhysicsParams, resolve_goal_boundary, step_ball
    from footballcoach.engine.scoring import check_goal

    phys_params = BallPhysicsParams.from_config()
    dt = params.sim_dt_s
    step_ball(ball, dt, phys_params)
    resolve_goal_boundary(ball, pitch, phys_params)
    assert check_goal(ball, pitch) is not None


def test_generate_shard_distribution_sanity(gen_params):
    """Over a reasonably large sample, both event flags should fire a
    nonzero, non-saturated fraction of the time (§10's "distribution
    sanity" check)."""
    inputs, targets = generate_shard(400, seed=123, params=gen_params)
    assert inputs.shape == (400, N_INPUT_FIELDS)
    n_horizons = len(gen_params.horizons_s)
    assert targets.shape == (400, n_horizons * N_TARGET_FIELDS_PER_HORIZON)

    last_horizon_base = (n_horizons - 1) * N_TARGET_FIELDS_PER_HORIZON
    oob = targets[:, last_horizon_base + 9]
    goal = targets[:, last_horizon_base + 10]
    assert 0.0 < oob.mean() < 1.0
    assert 0.0 <= goal.mean() < 1.0  # goal is rarer; only assert non-saturation


# ---------------------------------------------------------------------------
# BallDynamicsEncoder / Decoder / Autoencoder
# ---------------------------------------------------------------------------

def test_encoder_decoder_shapes_no_nan():
    n_horizons = 5
    model = BallDynamicsAutoencoder(input_dim=N_INPUT_FIELDS, hidden_dim=32, latent_dim=8, n_horizons=n_horizons)
    x = torch.randn(6, N_INPUT_FIELDS)
    latent, heads = model(x)
    assert latent.shape == (6, 8)
    assert len(heads) == n_horizons
    for h in heads:
        assert h.shape == (6, N_TARGET_FIELDS_PER_HORIZON)
        assert torch.isfinite(h).all()
    assert torch.isfinite(latent).all()


def test_encoder_standalone_shape():
    enc = BallDynamicsEncoder(input_dim=N_INPUT_FIELDS, hidden_dim=16, latent_dim=4)
    out = enc(torch.randn(3, N_INPUT_FIELDS))
    assert out.shape == (3, 4)


def test_from_config_matches_ai_config():
    from footballcoach.ai.config import load_ai_config
    cfg = load_ai_config()["physics_pretrain"]["ball"]
    model = BallDynamicsAutoencoder.from_config()
    assert model.encoder.latent_dim == cfg["latent_dim"]
    assert model.decoder.n_horizons == len(cfg["horizons_s"])


# ---------------------------------------------------------------------------
# Loss function (hand-computed reference)
# ---------------------------------------------------------------------------

def test_compute_loss_hand_computed():
    from footballcoach.ai.physics_pretrain.train_ball_dynamics import compute_loss

    n_horizons = 1
    head_out = torch.zeros(1, N_TARGET_FIELDS_PER_HORIZON)
    head_out[0, 9] = 0.0   # out_of_bounds logit=0 -> p=0.5
    head_out[0, 10] = 0.0  # goal_scored logit=0 -> p=0.5
    target = torch.zeros(1, N_TARGET_FIELDS_PER_HORIZON)
    target[0, :9] = 1.0    # continuous target all 1.0, pred all 0.0 -> MSE = 1.0
    target[0, 9] = 1.0     # positive out_of_bounds label
    target[0, 10] = 0.0    # negative goal_scored label

    pos_weight = torch.ones(n_horizons, 2)
    total, breakdown = compute_loss([head_out], target, pos_weight)

    expected_component_mse = 1.0  # mean((1-0)^2) over each 3-field group, all identical
    # BCE with logits=0 for both a positive and a negative label, pos_weight=1:
    # -[y*log(sigmoid(0)) + (1-y)*log(1-sigmoid(0))] = log(2) for each, evaluated separately.
    expected_bce = float(np.log(2.0))

    assert breakdown.pos_mse[0] == pytest.approx(expected_component_mse, abs=1e-5)
    assert breakdown.vel_mse[0] == pytest.approx(expected_component_mse, abs=1e-5)
    assert breakdown.spin_mse[0] == pytest.approx(expected_component_mse, abs=1e-5)
    assert breakdown.oob_bce[0] == pytest.approx(expected_bce, abs=1e-5)
    assert breakdown.goal_bce[0] == pytest.approx(expected_bce, abs=1e-5)
    assert float(total.item()) == pytest.approx(3 * expected_component_mse + 2 * expected_bce, abs=1e-5)


# ---------------------------------------------------------------------------
# Dataset generation, I/O, and pos_weight computation
# ---------------------------------------------------------------------------

def test_generate_dataset_roundtrip(tmp_path):
    paths = generate_dataset(n_episodes=50, output_dir=tmp_path, seed=7, shard_size=20, n_workers=1)
    assert len(paths) == 3  # 20 + 20 + 10
    for p in paths:
        assert p.exists()

    ds = BallDynamicsDataset.from_directory(tmp_path)
    assert len(ds) == 50
    assert ds.inputs.shape == (50, N_INPUT_FIELDS)


def test_dataset_split_train_val(tmp_path):
    generate_dataset(n_episodes=40, output_dir=tmp_path, seed=1, shard_size=40, n_workers=1)
    ds = BallDynamicsDataset.from_directory(tmp_path)
    train_idx, val_idx = ds.split_train_val(val_frac=0.25, seed=0)
    assert len(train_idx) + len(val_idx) == 40
    assert len(val_idx) == 10
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0


def test_compute_pos_weights_shape(tmp_path):
    generate_dataset(n_episodes=60, output_dir=tmp_path, seed=2, shard_size=60, n_workers=1)
    ds = BallDynamicsDataset.from_directory(tmp_path)
    n_horizons = 5
    weights = ds.compute_pos_weights(n_horizons)
    assert weights.shape == (n_horizons, 2)
    assert (weights >= 0.0).all()


def test_iterate_minibatches_covers_all_rows(tmp_path):
    generate_dataset(n_episodes=25, output_dir=tmp_path, seed=3, shard_size=25, n_workers=1)
    ds = BallDynamicsDataset.from_directory(tmp_path)
    idx = np.arange(len(ds))
    seen = 0
    for x, y in ds.iterate_minibatches(batch_size=7, indices=idx, shuffle=False):
        assert x.shape[1] == N_INPUT_FIELDS
        seen += x.shape[0]
    assert seen == 25


# ---------------------------------------------------------------------------
# Training script smoke test
# ---------------------------------------------------------------------------

def test_train_smoke(tmp_path):
    from footballcoach.ai.physics_pretrain.train_ball_dynamics import train

    dataset_dir = tmp_path / "data"
    generate_dataset(n_episodes=80, output_dir=dataset_dir, seed=5, shard_size=80, n_workers=1)

    output_path = tmp_path / "ball_encoder.pt"
    artifact = train(
        dataset_dir=str(dataset_dir),
        output_path=str(output_path),
        epochs=3,
        batch_size=16,
        lr=1e-2,
        val_frac=0.2,
        seed=0,
    )
    assert output_path.exists()
    assert "encoder_state_dict" in artifact
    assert "physics_config_hash" in artifact

    # Round-trip: load the saved encoder and confirm weights match what was returned.
    from footballcoach.ai.physics_pretrain.ball_dynamics_net import BallDynamicsEncoder
    from footballcoach.ai.config import load_ai_config

    cfg = load_ai_config()["physics_pretrain"]["ball"]
    loaded = torch.load(output_path, weights_only=True)
    encoder = BallDynamicsEncoder(hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"])
    encoder.load_state_dict(loaded["encoder_state_dict"])
    for p1, p2 in zip(encoder.state_dict().values(), artifact["encoder_state_dict"].values()):
        torch.testing.assert_close(p1, p2)

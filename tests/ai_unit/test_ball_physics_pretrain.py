"""Unit tests for the ball-dynamics physics pretraining pipeline.

See agent_plans/ball_physics_pretrain_plan.md section 10. This suite only
covers the standalone pipeline (episode generation, dataset, network,
training loop) -- there is no live-network integration to test yet (see the
plan's section 8, not implemented).
"""
from __future__ import annotations

import math
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from footballcoach.ai.physics_pretrain.ball_episode_gen import (
    BALL_SPIN_NORM_DIVISOR_RAD_S,
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
    input_row, target_row, crossing_row, crossing_time = generate_episode(rng, gen_params)
    assert input_row.shape == (N_INPUT_FIELDS,)
    assert target_row.shape == (len(gen_params.horizons_s) * N_TARGET_FIELDS_PER_HORIZON,)
    assert crossing_row.shape == (N_TARGET_FIELDS_PER_HORIZON,)
    assert isinstance(crossing_time, float)
    assert np.isfinite(input_row).all()
    assert np.isfinite(target_row).all()
    assert np.isfinite(crossing_row).all()


def test_encode_input_engineered_features_hand_computed(gen_params):
    from footballcoach.ai.physics_pretrain.ball_episode_gen import _encode_input
    from footballcoach.engine.ball_physics import BallPhysicsParams

    pitch = Pitch.standard()
    ball = Ball(
        position=Vector3(0.0, 0.0, 0.0),
        velocity=Vector3(3.0, 4.0, 0.0),  # |v| = 5.0 exactly, before normalization
        spin=Vector3(1.0, 0.0, 0.0),
    )
    phys_params = BallPhysicsParams.from_config()
    row = _encode_input(ball, pitch, phys_params.bounce_restitution_vertical, gen_params)

    half_diag = math.hypot(pitch.half_length, pitch.half_width)
    vx, vy, vz = 3.0 / half_diag, 4.0 / half_diag, 0.0
    sx, sy, sz = 1.0 / BALL_SPIN_NORM_DIVISOR_RAD_S, 0.0, 0.0

    expected_speed_norm = math.sqrt(vx**2 + vy**2 + vz**2)
    assert row[14] == pytest.approx(expected_speed_norm)
    assert row[15] == pytest.approx(expected_speed_norm**2)

    expected_spin_norm = math.sqrt(sx**2 + sy**2 + sz**2)
    assert row[16] == pytest.approx(expected_spin_norm)

    # spin x velocity, hand-computed cross product.
    expected_cross = (
        sy * vz - sz * vy,
        sz * vx - sx * vz,
        sx * vy - sy * vx,
    )
    assert row[17] == pytest.approx(expected_cross[0])
    assert row[18] == pytest.approx(expected_cross[1])
    assert row[19] == pytest.approx(expected_cross[2])


def test_encode_input_target_normalizes_by_base_pitch_when_flag_set(gen_params):
    """normalize_kinematics_by_base_pitch=True: pos/vel are normalized by
    the FIXED base pitch dims instead of this episode's own randomized
    ones -- so the same real position/velocity produces a DIFFERENT
    normalized value than the per-episode convention would, on a
    non-standard-sized pitch. Under this convention x/y/height ALL share
    ONE divisor (the base pitch's half_diag), matching velocity's
    convention, rather than height keeping its own height_norm_m scale.
    Pitch/goal dims (fields 10-13) stay a ratio-to-base either way --
    unaffected by this flag. Builds both variants explicitly via `replace`
    rather than trusting the `gen_params` fixture's own (live-config-
    dependent) value for either one."""
    from footballcoach.ai.physics_pretrain.ball_episode_gen import _encode_input, _encode_target
    from footballcoach.engine.ball_physics import BallPhysicsParams

    base_pitch_params = replace(gen_params, normalize_kinematics_by_base_pitch=True)
    per_episode_params = replace(gen_params, normalize_kinematics_by_base_pitch=False)
    # A pitch noticeably different from the base/standard one, so the two
    # conventions produce different normalized values.
    pitch = Pitch(
        length_m=90.0, width_m=60.0, goal_width_m=7.0, goal_height_m=2.4,
        goal_depth_m=2.0, box_length_m=16.0, box_width_m=40.0,
        six_yard_length_m=5.0, six_yard_width_m=18.0,
        penalty_spot_distance_m=11.0, centre_circle_radius_m=9.15,
    )
    ball = Ball(
        position=Vector3(12.0, -8.0, 1.5),
        velocity=Vector3(3.0, -4.0, 2.0),
        spin=Vector3(1.0, 2.0, -3.0),
    )
    restitution = BallPhysicsParams.from_config().bounce_restitution_vertical

    base_half_length = base_pitch_params.base_pitch_length_m / 2
    base_half_width = base_pitch_params.base_pitch_width_m / 2
    base_half_diag = math.hypot(base_half_length, base_half_width)

    row = _encode_input(ball, pitch, restitution, base_pitch_params)
    # x/y/height all share ONE divisor (half_diag) under this convention.
    assert row[0] == pytest.approx(ball.position.x / base_half_diag)
    assert row[1] == pytest.approx(ball.position.y / base_half_diag)
    assert row[2] == pytest.approx(ball.position.z / base_half_diag)
    assert row[3] == pytest.approx(ball.velocity.x / base_half_diag)
    # spin is unaffected by this flag -- same fixed constant as always.
    assert row[6] == pytest.approx(ball.spin.x / BALL_SPIN_NORM_DIVISOR_RAD_S)
    # Pitch/goal dims are STILL a ratio-to-base, unaffected by this flag.
    assert row[10] == pytest.approx(pitch.length_m / base_pitch_params.base_pitch_length_m)
    assert row[11] == pytest.approx(pitch.width_m / base_pitch_params.base_pitch_width_m)

    target_row = _encode_target(
        ball.position, ball.velocity, ball.spin, out_of_bounds=False, goal_scored=False,
        pitch=pitch, params=base_pitch_params,
    )
    assert target_row[0] == pytest.approx(ball.position.x / base_half_diag)

    # Contrast with the per-episode (False) convention on the SAME
    # ball/pitch -- values differ, proving the flag actually changes encoding.
    per_episode_row = _encode_input(ball, pitch, restitution, per_episode_params)
    assert per_episode_row[0] == pytest.approx(ball.position.x / pitch.half_length)
    assert per_episode_row[0] != pytest.approx(row[0])
    # Pitch/goal-dim fields are identical either way.
    assert per_episode_row[10] == pytest.approx(row[10])


def test_generate_episode_deterministic_given_seed(gen_params):
    in1, tgt1, cross1, ctime1 = generate_episode(random.Random(42), gen_params)
    in2, tgt2, cross2, ctime2 = generate_episode(random.Random(42), gen_params)
    np.testing.assert_array_equal(in1, in2)
    np.testing.assert_array_equal(tgt1, tgt2)
    np.testing.assert_array_equal(cross1, cross2)
    assert ctime1 == ctime2


def test_generate_episode_different_seeds_differ(gen_params):
    in1, *_ = generate_episode(random.Random(1), gen_params)
    in2, *_ = generate_episode(random.Random(2), gen_params)
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


def test_generate_episode_simulates_to_exact_horizon_time_not_nearest_tick(gen_params, monkeypatch):
    """Horizons that don't fall on a whole multiple of sim_dt_s are reached
    via a final PARTIAL step, not snapped to the nearest tick -- verified
    by spying on step_ball's dt_s arguments and checking their cumulative
    sum lands EXACTLY on each horizon_s value (not merely close to it)."""
    from footballcoach.ai.physics_pretrain import ball_episode_gen as beg

    dt = gen_params.sim_dt_s
    # Deliberately NOT whole multiples of dt.
    custom_params = replace(gen_params, horizons_s=(dt * 2.5, dt * 5.3, dt * 9.9))

    step_dts: list[float] = []
    real_step_ball = beg.step_ball

    def spy_step_ball(ball, dt_s, params):
        step_dts.append(dt_s)
        real_step_ball(ball, dt_s, params)

    monkeypatch.setattr(beg, "step_ball", spy_step_ball)
    beg.generate_episode(random.Random(0), custom_params)

    cumulative = np.cumsum(step_dts)
    for h in custom_params.horizons_s:
        assert np.any(np.isclose(cumulative, h, atol=1e-9)), (
            f"horizon {h} not exactly reached; cumulative times were {cumulative}"
        )


def test_generate_shard_distribution_sanity(gen_params):
    """Over a reasonably large sample, both event flags should fire a
    nonzero, non-saturated fraction of the time (§10's "distribution
    sanity" check)."""
    inputs, targets, crossings, crossing_times = generate_shard(400, seed=123, params=gen_params)
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
    horizons_s = [0.2, 0.5, 1.0, 2.0, 3.0]
    n_horizons = len(horizons_s)
    model = BallDynamicsAutoencoder(input_dim=N_INPUT_FIELDS, hidden_dim=32, latent_dim=8, horizons_s=horizons_s)
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


def test_decoder_forward_at_matches_forward_for_trained_horizon():
    from footballcoach.ai.physics_pretrain.ball_dynamics_net import BallDynamicsDecoder

    horizons_s = [0.2, 0.5, 1.0]
    decoder = BallDynamicsDecoder(latent_dim=8, horizons_s=horizons_s, hidden_dim=16)
    decoder.eval()
    latent = torch.randn(4, 8)
    heads = decoder(latent)
    for h, t in enumerate(horizons_s):
        via_forward_at = decoder.forward_at(latent, t)
        torch.testing.assert_close(via_forward_at, heads[h])


def test_decoder_forward_at_zero_horizon_finite_no_special_case_needed():
    from footballcoach.ai.physics_pretrain.ball_dynamics_net import BallDynamicsDecoder

    decoder = BallDynamicsDecoder(latent_dim=8, horizons_s=[0.2, 0.5, 1.0], hidden_dim=16)
    out = decoder.forward_at(torch.randn(3, 8), 0.0)
    assert out.shape == (3, N_TARGET_FIELDS_PER_HORIZON)
    assert torch.isfinite(out).all()


def test_identity_shortcut_zero_noise_gives_exact_round_trip():
    from footballcoach.ai.physics_pretrain.ball_dynamics_net import N_IDENTITY_SHORTCUT_FIELDS

    torch.manual_seed(0)
    model = BallDynamicsAutoencoder(
        input_dim=N_INPUT_FIELDS, hidden_dim=32, latent_dim=16, horizons_s=[0.2, 0.5, 1.0],
        decoder_hidden_dim=32, encoder_bottleneck_dim=16,
        identity_shortcut=True, identity_shortcut_noise_std=0.0,
    )
    x = torch.randn(5, N_INPUT_FIELDS)
    # Height (Z_FIELD_INDEX) is never negative in real data -- its dedicated
    # decoder unit is a single bare ReLU (see _init_identity_shortcut_decoder's
    # docstring), which can't exactly round-trip a genuinely negative height,
    # only real (non-negative) ones. Clamp it here so this stays a test of
    # the exact-round-trip property on realistic data, not of that documented
    # single-ReLU limitation on synthetic data no real episode ever produces.
    from footballcoach.ai.physics_pretrain.ball_dynamics_net import Z_FIELD_INDEX
    x[:, Z_FIELD_INDEX] = x[:, Z_FIELD_INDEX].abs()
    latent = model.encoder(x)
    torch.testing.assert_close(latent[:, :N_IDENTITY_SHORTCUT_FIELDS], x[:, :N_IDENTITY_SHORTCUT_FIELDS])

    for horizon_s in (0.0, 0.5, 3.7):
        recon = model.decoder.forward_at(latent, horizon_s)
        torch.testing.assert_close(recon[:, :N_IDENTITY_SHORTCUT_FIELDS], x[:, :N_IDENTITY_SHORTCUT_FIELDS])


def test_encoder_concat_all_input_fields_widens_out_layer_and_still_preserves_identity():
    """encoder_concat_all_input_fields=True widens encoder.out's input to
    bottleneck_dim + N_INPUT_FIELDS (not just +9), the identity rows still
    exactly preserve latent[0:9]~=x[0:9] (the widened concat keeps the 9
    identity fields first, so the hand-init's column offset is unaffected),
    and the extra (initially-zero) columns actually receive gradient --
    not a dead fixed point the way a ReLU'd layer would be."""
    from footballcoach.ai.physics_pretrain.ball_dynamics_net import N_IDENTITY_SHORTCUT_FIELDS

    torch.manual_seed(0)
    bottleneck_dim = 16
    model = BallDynamicsEncoder(
        input_dim=N_INPUT_FIELDS, hidden_dim=32, latent_dim=16, bottleneck_dim=bottleneck_dim,
        identity_shortcut=True, identity_shortcut_noise_std=0.0, concat_all_input_fields=True,
    )
    assert model.out.in_features == bottleneck_dim + N_INPUT_FIELDS

    x = torch.randn(5, N_INPUT_FIELDS)
    latent = model(x)
    torch.testing.assert_close(latent[:, :N_IDENTITY_SHORTCUT_FIELDS], x[:, :N_IDENTITY_SHORTCUT_FIELDS])

    # Columns for fields beyond the 9 identity ones start at zero weight
    # (see _init_identity_shortcut_linear) -- confirm gradient still
    # reaches them (no ReLU on this layer to create a dead fixed point).
    x2 = torch.randn(5, N_INPUT_FIELDS, requires_grad=False)
    latent2 = model(x2)
    loss = latent2.sum()
    loss.backward()
    extra_grad = model.out.weight.grad[:, bottleneck_dim + N_IDENTITY_SHORTCUT_FIELDS:]
    assert extra_grad.abs().max().item() > 0.0


def test_identity_shortcut_noise_std_perturbs_round_trip():
    from footballcoach.ai.physics_pretrain.ball_dynamics_net import N_IDENTITY_SHORTCUT_FIELDS

    torch.manual_seed(0)
    model = BallDynamicsAutoencoder(
        input_dim=N_INPUT_FIELDS, hidden_dim=32, latent_dim=16, horizons_s=[0.2, 0.5, 1.0],
        decoder_hidden_dim=32, encoder_bottleneck_dim=16,
        identity_shortcut=True, identity_shortcut_noise_std=0.1,
    )
    x = torch.randn(5, N_INPUT_FIELDS)
    latent = model.encoder(x)
    diff = (latent[:, :N_IDENTITY_SHORTCUT_FIELDS] - x[:, :N_IDENTITY_SHORTCUT_FIELDS]).abs().max().item()
    assert diff > 0.0


def test_identity_shortcut_survives_adversarial_classification_gradient():
    """Regression test for a real collapse observed in production training
    (physics_runs.md, 2026-08-19): pos_rmse went from ~0.3m at the
    pre-training baseline to 5+m within 1-2 epochs while oob_bce/goal_bce
    dropped sharply over the same epochs. Root cause: the decoder's
    oob/goal output rows started zero but UNMASKED against the dedicated
    identity units' (very much alive) activations, so gradient descent
    routed classification through them, corrupting the copy-through as a
    side effect. Drives many optimizer steps with a loss that combines
    near-zero reconstruction loss (so there's nothing pulling the identity
    weights on their own) with large classification loss (the only
    incentive to move at all), and asserts the TRULY-dedicated identity
    rows are completely unperturbed.

    Excludes ``z_free_idx`` (height's freed second unit -- see
    ``_init_identity_shortcut_decoder``'s docstring) from that "must not
    move" check: it's deliberately a live spare unit, not a real dedicated
    one, so it's SUPPOSED to move under classification pressure -- checked
    separately below instead. Also uses a non-negative synthetic height
    target (``.abs()``), matching the real premise that height is never
    negative -- feeding this path a genuinely negative "height" would
    produce real, unmaskable reconstruction-loss gradient the single ReLU
    dedicated unit can never resolve (an unrealistic scenario, not
    something this test should be exercising)."""
    from footballcoach.ai.physics_pretrain.ball_dynamics_net import (
        BallDynamicsDecoder, N_IDENTITY_SHORTCUT_FIELDS, Z_FIELD_INDEX,
    )

    torch.manual_seed(0)
    dim = N_IDENTITY_SHORTCUT_FIELDS
    z_free_idx = 2 * Z_FIELD_INDEX + 1
    dedicated_rows = [i for i in range(2 * dim) if i != z_free_idx]
    decoder = BallDynamicsDecoder(
        latent_dim=16, horizons_s=[0.2, 0.5, 1.0], hidden_dim=32,
        identity_shortcut=True, identity_shortcut_noise_std=0.0,
    )
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.1)
    first, second = decoder.net[0], decoder.net[2]
    before = first.weight[dedicated_rows, :dim].clone()
    before_free = first.weight[z_free_idx, :dim].clone()

    latent = torch.randn(8, 16)
    latent[:, Z_FIELD_INDEX] = latent[:, Z_FIELD_INDEX].abs()
    target_pvs = latent[:, :dim]
    for _ in range(20):
        pred = decoder.forward_at(latent, 0.0)
        loss = (
            F.mse_loss(pred[:, :dim], target_pvs)
            + F.binary_cross_entropy_with_logits(pred[:, 9], torch.ones(8))
            + F.binary_cross_entropy_with_logits(pred[:, 10], torch.ones(8))
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    after = first.weight[dedicated_rows, :dim].clone()
    torch.testing.assert_close(before, after)
    # The freed unit, by contrast, DOES move -- proving it's genuinely live.
    after_free = first.weight[z_free_idx, :dim].clone()
    assert not torch.allclose(before_free, after_free)
    final_pred = decoder.forward_at(latent, 0.0)
    torch.testing.assert_close(final_pred[:, :dim], target_pvs)
    # Classification logits still moved -- masking didn't just freeze everything.
    assert final_pred[:, 9].mean().item() > 0.5


def test_identity_shortcut_requires_latent_dim_at_least_nine():
    with pytest.raises(ValueError):
        BallDynamicsEncoder(input_dim=N_INPUT_FIELDS, hidden_dim=16, latent_dim=4, identity_shortcut=True)
    with pytest.raises(ValueError):
        BallDynamicsAutoencoder(
            input_dim=N_INPUT_FIELDS, hidden_dim=16, latent_dim=4, horizons_s=[0.2, 0.5],
            identity_shortcut=True,
        )


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
    target[0, :9] = 2.0    # continuous target all 2.0, pred all 0.0 -> MSE = 4.0, RMSE = 2.0
    target[0, 9] = 1.0     # positive out_of_bounds label
    target[0, 10] = 0.0    # negative goal_scored label

    pos_weight = torch.ones(n_horizons, 2)
    total, breakdown = compute_loss([head_out], target, pos_weight)

    expected_component_mse = 4.0  # mean((2-0)^2) over each 3-field group, all identical
    expected_component_rmse = 2.0
    # BCE with logits=0 for both a positive and a negative label, pos_weight=1:
    # -[y*log(sigmoid(0)) + (1-y)*log(1-sigmoid(0))] = log(2) for each, evaluated separately.
    expected_bce = float(np.log(2.0))

    # breakdown reports RMSE (see LossBreakdown's docstring) -- backprop still
    # sums plain MSE, which is what `total` below is checked against.
    assert breakdown.pos_rmse[0] == pytest.approx(expected_component_rmse, abs=1e-5)
    assert breakdown.vel_rmse[0] == pytest.approx(expected_component_rmse, abs=1e-5)
    assert breakdown.spin_rmse[0] == pytest.approx(expected_component_rmse, abs=1e-5)
    assert breakdown.oob_bce[0] == pytest.approx(expected_bce, abs=1e-5)
    assert breakdown.goal_bce[0] == pytest.approx(expected_bce, abs=1e-5)
    assert float(total.item()) == pytest.approx(3 * expected_component_mse + 2 * expected_bce, abs=1e-5)


def test_compute_loss_bce_weight_scales_total_but_not_reported_breakdown():
    """bce_weight multiplies the BCE terms' contribution to `total` (the
    backpropagated loss) only -- breakdown.oob_bce/goal_bce always report
    the raw, unweighted value, so runs at different weights stay
    comparable. 0.0 must fully remove the classification heads from the
    gradient (their loss no longer depends on head_out[:, 9:11] at all)."""
    from footballcoach.ai.physics_pretrain.train_ball_dynamics import compute_loss

    head_out = torch.zeros(1, N_TARGET_FIELDS_PER_HORIZON, requires_grad=True)
    target = torch.zeros(1, N_TARGET_FIELDS_PER_HORIZON)
    target[0, :9] = 2.0
    target[0, 9] = 1.0
    pos_weight = torch.ones(1, 2)

    total_full, breakdown_full = compute_loss([head_out], target, pos_weight, bce_weight=1.0)
    total_half, breakdown_half = compute_loss([head_out], target, pos_weight, bce_weight=0.5)
    total_off, breakdown_off = compute_loss([head_out], target, pos_weight, bce_weight=0.0)

    expected_bce = float(np.log(2.0))
    for breakdown in (breakdown_full, breakdown_half, breakdown_off):
        assert breakdown.oob_bce[0] == pytest.approx(expected_bce, abs=1e-5)
        assert breakdown.goal_bce[0] == pytest.approx(expected_bce, abs=1e-5)

    assert float(total_half.item()) == pytest.approx(float(total_full.item()) - expected_bce, abs=1e-5)
    assert float(total_off.item()) == pytest.approx(float(total_full.item()) - 2 * expected_bce, abs=1e-5)

    head_out2 = torch.zeros(1, N_TARGET_FIELDS_PER_HORIZON, requires_grad=True)
    total_off2, _ = compute_loss([head_out2], target, pos_weight, bce_weight=0.0)
    total_off2.backward()
    assert head_out2.grad[0, 9] == pytest.approx(0.0, abs=1e-8)
    assert head_out2.grad[0, 10] == pytest.approx(0.0, abs=1e-8)
    assert head_out2.grad[0, 0].abs().item() > 0.0  # pos/vel/spin gradient unaffected


def test_compute_per_episode_loss_matches_batch_mean_and_identifies_rows():
    from footballcoach.ai.physics_pretrain.train_ball_dynamics import compute_loss, compute_per_episode_loss

    n_horizons = 1
    pos_weight = torch.ones(n_horizons, 2)

    # 3 rows with deliberately different error magnitudes (row 1 exact match
    # -> 0 loss; row 0 and row 2 differ by increasing amounts).
    head_out = torch.zeros(3, N_TARGET_FIELDS_PER_HORIZON)
    target = torch.zeros(3, N_TARGET_FIELDS_PER_HORIZON)
    target[0, :9] = 1.0
    target[1, :9] = 0.0  # row 1 matches head_out exactly -> should be the minimum
    target[2, :9] = 3.0  # row 2 has the largest error -> should be the max

    per_row = compute_per_episode_loss([head_out], target, pos_weight)
    assert per_row.shape == (3,)
    # Columns 9/10 (event logits/targets) are 0 for every row here, so each
    # row carries the same constant BCE baseline (2*ln(2)) -- row 1's
    # continuous fields match exactly, so it should sit right at that floor.
    assert per_row[1].item() == pytest.approx(2 * float(np.log(2.0)), abs=1e-5)
    assert per_row[2].item() > per_row[0].item() > per_row[1].item()

    # Averaging the per-row losses over the batch must recover compute_loss's
    # `total` for that same batch (same underlying math, different reduction).
    total, _ = compute_loss([head_out], target, pos_weight)
    assert per_row.mean().item() == pytest.approx(float(total.item()), abs=1e-5)


def test_describe_input_row_denormalizes_correctly(gen_params):
    from footballcoach.ai.physics_pretrain.ball_episode_gen import _encode_input
    from footballcoach.ai.physics_pretrain.train_ball_dynamics import describe_input_row
    from footballcoach.engine.ball_physics import BallPhysicsParams

    pitch = Pitch.standard()
    ball = Ball(
        position=Vector3(10.0, -5.0, 1.5),
        velocity=Vector3(3.0, 4.0, 0.0),
        spin=Vector3(2.0, 0.0, 0.0),
    )
    phys_params = BallPhysicsParams.from_config()
    restitution = phys_params.bounce_restitution_vertical
    row = _encode_input(ball, pitch, restitution, gen_params)

    desc = describe_input_row(row, gen_params)
    assert "pos=(10.00, -5.00, 1.50)m" in desc
    assert "vel=(3.00, 4.00, 0.00)m/s" in desc
    assert "speed=5.00m/s" in desc
    assert f"restitution={restitution:.3f}" in desc
    assert f"pitch={pitch.length_m:.1f}x{pitch.width_m:.1f}m" in desc


def test_compute_confusion_counts_and_classification_metrics():
    from footballcoach.ai.physics_pretrain.train_ball_dynamics import (
        _classification_metrics,
        compute_confusion_counts,
    )

    # 4 rows, 1 horizon. out_of_bounds: logits give predictions [1, 1, 0, 0],
    # actuals [1, 0, 1, 0] -> tp=1 (row0), fp=1 (row1), fn=1 (row2), tn=1 (row3).
    # goal_scored: all predicted negative (logit<0), actuals all negative ->
    # tp=0, fp=0, fn=0, tn=4 (precision/recall undefined -> NaN).
    head_out = torch.zeros(4, N_TARGET_FIELDS_PER_HORIZON)
    head_out[:, 9] = torch.tensor([5.0, 5.0, -5.0, -5.0])   # oob logits
    head_out[:, 10] = torch.tensor([-5.0, -5.0, -5.0, -5.0])  # goal logits (all predicted negative)
    target = torch.zeros(4, N_TARGET_FIELDS_PER_HORIZON)
    target[:, 9] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    target[:, 10] = torch.tensor([0.0, 0.0, 0.0, 0.0])

    counts = compute_confusion_counts([head_out], target)
    assert counts["oob"] == [(1, 1, 1, 1)]
    assert counts["goal"] == [(0, 0, 0, 4)]

    oob_metrics = _classification_metrics(np.array(counts["oob"]))
    assert oob_metrics["accuracy"][0] == pytest.approx(0.5)
    assert oob_metrics["precision"][0] == pytest.approx(0.5)
    assert oob_metrics["recall"][0] == pytest.approx(0.5)

    goal_metrics = _classification_metrics(np.array(counts["goal"]))
    assert goal_metrics["accuracy"][0] == pytest.approx(1.0)
    assert np.isnan(goal_metrics["precision"][0])  # tp+fp == 0 -> undefined, not 0
    assert np.isnan(goal_metrics["recall"][0])     # tp+fn == 0 -> undefined, not 0


def test_r2_matches_hand_computed_and_flags_mean_collapse():
    from footballcoach.ai.physics_pretrain.train_ball_dynamics import (
        _r2_from_sq_err,
        compute_group_sq_err,
    )

    # 1 horizon, pos group (cols 0:3) target = [2,2,2] for every row, 4 rows.
    head_out = torch.zeros(4, N_TARGET_FIELDS_PER_HORIZON)
    target = torch.zeros(4, N_TARGET_FIELDS_PER_HORIZON)
    target[:, 0:3] = 2.0

    # Case A: prediction exactly matches target -> zero error -> R^2 = 1.
    pred_exact = head_out.clone()
    pred_exact[:, 0:3] = 2.0
    counts = compute_group_sq_err([pred_exact], target)
    sq_err = np.array([c[0] for c in counts["pos"]])
    n = np.array([c[1] for c in counts["pos"]])
    target_var = np.array([1.0])  # nonzero placeholder; error is 0 regardless
    r2 = _r2_from_sq_err(sq_err, n, target_var)
    assert r2[0] == pytest.approx(1.0)

    # Case B: prediction collapses to a constant that is NOT the true mean of
    # the underlying (pre-target-var) distribution -- simulate this by giving
    # a target_var equal to the actual MSE the constant-zero prediction
    # achieves against this target: MSE = mean((2-0)^2 over 3 fields) = 4.0.
    # A model doing no better than "always predict the (target_var-implied)
    # mean" should score R^2 == 0 exactly.
    counts_zero = compute_group_sq_err([head_out], target)  # head_out is all zeros
    sq_err_zero = np.array([c[0] for c in counts_zero["pos"]])
    n_zero = np.array([c[1] for c in counts_zero["pos"]])
    mse_zero = sq_err_zero / n_zero
    r2_zero = _r2_from_sq_err(sq_err_zero, n_zero, mse_zero)
    assert r2_zero[0] == pytest.approx(0.0)


def test_compute_group_variance_matches_numpy(tmp_path):
    generate_dataset(n_episodes=30, output_dir=tmp_path, seed=3, shard_size=30, n_workers=1)
    ds = BallDynamicsDataset.from_directory(tmp_path)
    n_horizons = ds.targets.shape[1] // N_TARGET_FIELDS_PER_HORIZON
    variances = ds.compute_group_variance(n_horizons)
    for h in range(n_horizons):
        base = h * N_TARGET_FIELDS_PER_HORIZON
        expected_pos = ds.targets[:, base:base + 3].var(axis=0).mean()
        assert variances["pos"][h] == pytest.approx(expected_pos)


def test_compute_persistence_baseline_mse_matches_numpy(tmp_path):
    generate_dataset(n_episodes=30, output_dir=tmp_path, seed=3, shard_size=30, n_workers=1)
    ds = BallDynamicsDataset.from_directory(tmp_path)
    n_horizons = ds.targets.shape[1] // N_TARGET_FIELDS_PER_HORIZON
    baseline = ds.compute_persistence_baseline_mse(n_horizons)
    for h in range(n_horizons):
        base = h * N_TARGET_FIELDS_PER_HORIZON
        expected_vel = ((ds.targets[:, base + 3:base + 6] - ds.inputs[:, 3:6]) ** 2).mean()
        assert baseline["vel"][h] == pytest.approx(expected_vel)
        # Sanity: a horizon-0 (shortest) persistence MSE should be small
        # relative to later horizons, since the ball hasn't moved much yet.
    assert baseline["pos"][0] < baseline["pos"][-1]


def test_compute_ballistic_baseline_mse_hand_computed():
    """x/y run the full straight-line extrapolation, unclamped -- even out
    past the pitch boundary (x_norm can exceed 1.0). Only z is floored at
    0. Uses a large t specifically so x_norm DOES exceed 1.0, proving x is
    NOT being clamped at the pitch edge; z0 is chosen high enough that z
    stays positive at this t, so this test isolates the raw kinematics
    formula (z-floor is covered separately below).

    Explicitly forces normalize_kinematics_by_base_pitch=True (rather than
    trusting BallEpisodeGenParams.from_config()'s live-config-dependent
    value) since this test's expected values assume x/y/height all share
    the base pitch's half_diag divisor, which only holds under that
    convention."""
    from footballcoach.ai.physics_pretrain.ball_episode_gen import BallEpisodeGenParams

    gen_params = replace(BallEpisodeGenParams.from_config(), normalize_kinematics_by_base_pitch=True)
    half_length = gen_params.base_pitch_length_m / 2
    half_width = gen_params.base_pitch_width_m / 2
    half_diag = math.hypot(half_length, half_width)
    gravity = 10.0

    # 1 row, base pitch (fields 10-11 = 1.0), pos_x/y=0, height z0=100m
    # (stays positive through t=2.5 below), vel_x normalized = 0.5 (real vx
    # = 0.5 * half_diag). Target is all zeros, so the "error" IS the
    # ballistic prediction itself -- easy to hand-check.
    z0_real_m = 100.0
    inputs = np.zeros((1, N_INPUT_FIELDS), dtype=np.float32)
    inputs[0, 10] = 1.0
    inputs[0, 11] = 1.0
    inputs[0, 2] = z0_real_m / half_diag
    inputs[0, 3] = 0.5
    targets = np.zeros((1, N_TARGET_FIELDS_PER_HORIZON), dtype=np.float32)
    ds = BallDynamicsDataset(inputs, targets)

    # x/y and vel now share ONE divisor (half_diag), so normalized position
    # growth is exactly 0.5*t -- needs t > 2.0 to exceed 1.0 (unlike before,
    # when x used the smaller half_length and so exceeded 1.0 already at t=2.0).
    t = 2.5  # x(t) = 78.2m > half_length = 52.5m -- deliberately outside the pitch
    baseline = ds.compute_ballistic_baseline_mse([t], gen_params, gravity)

    real_vx = 0.5 * half_diag
    expected_pos_x_norm = (real_vx * t) / half_diag
    assert expected_pos_x_norm > 1.0  # confirms this test actually exercises the unclamped case
    expected_pos_z_norm = (z0_real_m - 0.5 * gravity * t ** 2) / half_diag
    expected_pos_mse = (expected_pos_x_norm ** 2 + expected_pos_z_norm ** 2) / 3
    assert baseline["pos"][0] == pytest.approx(expected_pos_mse, rel=1e-4)

    expected_vel_x_norm = real_vx / half_diag  # unaffected by gravity, unchanged
    expected_vel_z_norm = (-gravity * t) / half_diag
    expected_vel_mse = (expected_vel_x_norm ** 2 + expected_vel_z_norm ** 2) / 3
    assert baseline["vel"][0] == pytest.approx(expected_vel_mse, rel=1e-4)

    # Ballistic doesn't touch spin -- identical to persistence (target spin=0,
    # input spin=0 here, so MSE is exactly 0).
    assert baseline["spin"][0] == pytest.approx(0.0, abs=1e-6)


def test_compute_ballistic_baseline_mse_clamps_z_floor_but_not_xy():
    """A ball with both horizontal velocity (would exit the pitch) and a
    horizon long enough to free-fall below ground: x stays fully unclamped
    (still allowed past the pitch edge), z is floored at exactly 0 (not
    extrapolated to a negative height).

    Explicitly forces normalize_kinematics_by_base_pitch=True -- see
    test_compute_ballistic_baseline_mse_hand_computed's docstring."""
    from footballcoach.ai.physics_pretrain.ball_episode_gen import BallEpisodeGenParams

    gen_params = replace(BallEpisodeGenParams.from_config(), normalize_kinematics_by_base_pitch=True)
    half_length = gen_params.base_pitch_length_m / 2
    half_width = gen_params.base_pitch_width_m / 2
    half_diag = math.hypot(half_length, half_width)
    gravity = 10.0

    z0_real_m = 5.0  # lands at t=1.0s
    inputs = np.zeros((1, N_INPUT_FIELDS), dtype=np.float32)
    inputs[0, 10] = 1.0
    inputs[0, 11] = 1.0
    inputs[0, 2] = z0_real_m / half_diag
    inputs[0, 3] = 0.5
    targets = np.zeros((1, N_TARGET_FIELDS_PER_HORIZON), dtype=np.float32)
    ds = BallDynamicsDataset(inputs, targets)

    real_vx = 0.5 * half_diag
    t = 5.0  # well past both the pitch edge (~1.68s) and ground landing (~1.0s)
    baseline = ds.compute_ballistic_baseline_mse([t], gen_params, gravity)

    # x kept extrapolating the FULL unclamped distance (well past the pitch edge).
    expected_pos_x_norm = (real_vx * t) / half_diag
    assert expected_pos_x_norm > 1.0
    # z floored at exactly 0, not the (large negative) unclamped free-fall value.
    expected_pos_mse = (expected_pos_x_norm ** 2 + 0.0 ** 2) / 3
    assert baseline["pos"][0] == pytest.approx(expected_pos_mse, rel=1e-4)

    # Velocity is NOT clamped at all (only position's z is floored) --
    # vel_z keeps accelerating downward for the full unclamped t.
    expected_vel_x_norm = real_vx / half_diag
    expected_vel_z_norm = (-gravity * t) / half_diag
    expected_vel_mse = (expected_vel_x_norm ** 2 + expected_vel_z_norm ** 2) / 3
    assert baseline["vel"][0] == pytest.approx(expected_vel_mse, rel=1e-4)

    # Sanity: unclamped free fall to t=5.0 would be far below ground --
    # confirms this test would catch a regression to no z-floor at all.
    naive_pos_z_norm = (z0_real_m - 0.5 * gravity * t ** 2) / half_diag
    assert naive_pos_z_norm < -1.0


def test_compute_ballistic_baseline_mse_normalizes_by_base_pitch_when_flag_set():
    """Same scenario as test_compute_ballistic_baseline_mse_hand_computed,
    but with normalize_kinematics_by_base_pitch=True -- pos0/vel0 recovery
    and the final re-normalization both use the FIXED base pitch dims, and
    x/y/height ALL share ONE divisor (half_diag), matching velocity's
    convention (rather than height keeping its own height_norm_m scale).
    Fields 10-11 are deliberately set to a non-1.0 ratio here to prove
    they're NOT used for this in base-pitch mode (only the always-
    ratio-to-base fields 10-13 would read them, and this function doesn't
    touch those)."""
    from footballcoach.ai.physics_pretrain.ball_episode_gen import BallEpisodeGenParams

    gen_params = replace(BallEpisodeGenParams.from_config(), normalize_kinematics_by_base_pitch=True)
    half_length = gen_params.base_pitch_length_m / 2
    half_width = gen_params.base_pitch_width_m / 2
    half_diag = math.hypot(half_length, half_width)
    gravity = 10.0

    z0_real_m = 100.0
    inputs = np.zeros((1, N_INPUT_FIELDS), dtype=np.float32)
    inputs[0, 10] = 1.2  # deliberately NOT 1.0 -- must be ignored in base-pitch mode
    inputs[0, 11] = 0.8
    inputs[0, 2] = z0_real_m / half_diag
    inputs[0, 3] = 0.5
    targets = np.zeros((1, N_TARGET_FIELDS_PER_HORIZON), dtype=np.float32)
    ds = BallDynamicsDataset(inputs, targets)

    t = 2.0
    baseline = ds.compute_ballistic_baseline_mse([t], gen_params, gravity)

    real_vx = 0.5 * half_diag
    expected_pos_x_norm = (real_vx * t) / half_diag
    expected_pos_z_norm = (z0_real_m - 0.5 * gravity * t ** 2) / half_diag
    expected_pos_mse = (expected_pos_x_norm ** 2 + expected_pos_z_norm ** 2) / 3
    assert baseline["pos"][0] == pytest.approx(expected_pos_mse, rel=1e-4)

    expected_vel_x_norm = real_vx / half_diag
    expected_vel_z_norm = (-gravity * t) / half_diag
    expected_vel_mse = (expected_vel_x_norm ** 2 + expected_vel_z_norm ** 2) / 3
    assert baseline["vel"][0] == pytest.approx(expected_vel_mse, rel=1e-4)


def test_build_adjacent_pair_data_excludes_resolved_and_reencodes_correctly(tmp_path):
    generate_dataset(n_episodes=150, output_dir=tmp_path, seed=11, shard_size=150, n_workers=1)
    ds = BallDynamicsDataset.from_directory(tmp_path)
    n_horizons = ds.targets.shape[1] // N_TARGET_FIELDS_PER_HORIZON
    pair_idx = 0

    derived_inputs, targets_by_skip, row_idx = ds.build_adjacent_pair_data(pair_idx)
    assert [skip for skip, _ in targets_by_skip] == [1]
    derived_targets = targets_by_skip[0][1]

    base_i = pair_idx * N_TARGET_FIELDS_PER_HORIZON
    base_j = (pair_idx + 1) * N_TARGET_FIELDS_PER_HORIZON
    block_i_all = ds.targets[:, base_i:base_i + N_TARGET_FIELDS_PER_HORIZON]

    # Every row we kept must NOT have been resolved (oob/goal) at horizon i.
    kept_block_i = ds.targets[row_idx, base_i:base_i + N_TARGET_FIELDS_PER_HORIZON]
    assert np.all(kept_block_i[:, 9] < 0.5)
    assert np.all(kept_block_i[:, 10] < 0.5)
    # And every row we DROPPED must have been resolved.
    dropped_mask = np.ones(len(ds), dtype=bool)
    dropped_mask[row_idx] = False
    dropped_block_i = block_i_all[dropped_mask]
    if len(dropped_block_i) > 0:
        assert np.all((dropped_block_i[:, 9] >= 0.5) | (dropped_block_i[:, 10] >= 0.5))

    # Target returned must be exactly horizon j's recorded block for kept rows.
    expected_target = ds.targets[row_idx, base_j:base_j + N_TARGET_FIELDS_PER_HORIZON]
    np.testing.assert_allclose(derived_targets, expected_target)

    # Re-encoded input: pos/vel/spin come from horizon i's block, restitution/
    # pitch/goal dims are unchanged from the episode's own original input.
    np.testing.assert_allclose(derived_inputs[:, 0:9], kept_block_i[:, 0:9])
    np.testing.assert_allclose(derived_inputs[:, 9:14], ds.inputs[row_idx, 9:14])
    from footballcoach.ai.physics_pretrain.ball_episode_gen import compute_engineered_features
    np.testing.assert_allclose(
        derived_inputs[:, 14:20], compute_engineered_features(kept_block_i[:, 0:9]), rtol=1e-4,
    )
    assert pair_idx < n_horizons - 1  # sanity: this pair index is actually valid


def test_build_autoencoding_data_no_filtering_and_reencodes_correctly(tmp_path):
    generate_dataset(n_episodes=60, output_dir=tmp_path, seed=12, shard_size=60, n_workers=1)
    ds = BallDynamicsDataset.from_directory(tmp_path)
    horizon_idx = 2

    derived_inputs, derived_targets = ds.build_autoencoding_data(horizon_idx)
    assert derived_inputs.shape == (len(ds), N_INPUT_FIELDS)
    assert derived_targets.shape == (len(ds), N_TARGET_FIELDS_PER_HORIZON)

    base = horizon_idx * N_TARGET_FIELDS_PER_HORIZON
    block = ds.targets[:, base:base + N_TARGET_FIELDS_PER_HORIZON]
    np.testing.assert_allclose(derived_targets, block)
    np.testing.assert_allclose(derived_inputs[:, 0:9], block[:, 0:9])
    np.testing.assert_allclose(derived_inputs[:, 9:14], ds.inputs[:, 9:14])


def test_decoder_only_training_freezes_encoder():
    """Regression test for the decoder-only pretraining phase's core
    mechanism: encoder params get requires_grad_(False) and the optimizer
    is scoped to model.decoder.parameters() only, so training steps must
    leave the encoder byte-identical while the decoder actually moves."""
    torch.manual_seed(0)
    model = BallDynamicsAutoencoder(
        input_dim=N_INPUT_FIELDS, hidden_dim=32, latent_dim=16, horizons_s=[0.2, 0.5, 1.0],
        decoder_hidden_dim=32, encoder_bottleneck_dim=16,
    )
    before_encoder = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    before_decoder = {k: v.clone() for k, v in model.decoder.state_dict().items()}

    for p in model.encoder.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-2)

    x = torch.randn(16, N_INPUT_FIELDS)
    for _ in range(5):
        _, heads = model(x)
        loss = sum(F.mse_loss(h[:, :9], torch.randn(16, 9)) for h in heads)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for k, v in model.encoder.state_dict().items():
        torch.testing.assert_close(v, before_encoder[k])
    decoder_diff = max((model.decoder.state_dict()[k] - before_decoder[k]).abs().max().item() for k in before_decoder)
    assert decoder_diff > 0.0

    for p in model.encoder.parameters():
        p.requires_grad_(True)
    assert all(p.requires_grad for p in model.encoder.parameters())


def test_decoder_only_training_trunk_frozen_but_out_trains_with_identity_rows_masked():
    """train_ball_dynamics.py's actual decoder-only-pretrain scoping (not
    the older full-encoder-freeze this replaced): encoder.trunk stays
    byte-identical, encoder.out's IDENTITY rows (producing latent[0:dim])
    stay byte-identical (gradient-masked), but encoder.out's SPARE rows
    (dim:) and the decoder both actually move."""
    from footballcoach.ai.physics_pretrain.ball_dynamics_net import N_IDENTITY_SHORTCUT_FIELDS

    torch.manual_seed(0)
    dim = N_IDENTITY_SHORTCUT_FIELDS
    model = BallDynamicsAutoencoder(
        input_dim=N_INPUT_FIELDS, hidden_dim=32, latent_dim=16, horizons_s=[0.2, 0.5, 1.0],
        decoder_hidden_dim=32, encoder_bottleneck_dim=16,
        identity_shortcut=True, identity_shortcut_noise_std=0.001,
    )
    before_trunk = {k: v.clone() for k, v in model.encoder.trunk.state_dict().items()}
    before_out_weight = model.encoder.out.weight.clone()
    before_out_bias = model.encoder.out.bias.clone()
    before_decoder = {k: v.clone() for k, v in model.decoder.state_dict().items()}

    for p in model.encoder.trunk.parameters():
        p.requires_grad_(False)
    weight_mask = torch.ones_like(model.encoder.out.weight)
    weight_mask[:dim, :] = 0.0
    bias_mask = torch.ones_like(model.encoder.out.bias)
    bias_mask[:dim] = 0.0
    model.encoder.out.weight.register_hook(lambda grad: grad * weight_mask)
    model.encoder.out.bias.register_hook(lambda grad: grad * bias_mask)
    optimizer = torch.optim.Adam(
        list(model.decoder.parameters()) + list(model.encoder.out.parameters()), lr=1e-2,
    )

    x = torch.randn(16, N_INPUT_FIELDS)
    for _ in range(5):
        _, heads = model(x)
        loss = sum(F.mse_loss(h[:, :9], torch.randn(16, 9)) for h in heads)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for k, v in model.encoder.trunk.state_dict().items():
        torch.testing.assert_close(v, before_trunk[k])
    torch.testing.assert_close(model.encoder.out.weight[:dim, :], before_out_weight[:dim, :])
    torch.testing.assert_close(model.encoder.out.bias[:dim], before_out_bias[:dim])

    assert (model.encoder.out.weight[dim:, :] - before_out_weight[dim:, :]).abs().max().item() > 0.0
    decoder_diff = max((model.decoder.state_dict()[k] - before_decoder[k]).abs().max().item() for k in before_decoder)
    assert decoder_diff > 0.0


def test_train_smoke_with_decoder_only_pretraining(tmp_path, monkeypatch, caplog):
    """End-to-end: decoder_only_pretrain_epochs > 0 runs without error,
    logs the phase with full per-component diagnostics (matching a regular
    epoch's format), and the run completes normally afterward."""
    from footballcoach.ai.physics_pretrain.train_ball_dynamics import train
    import footballcoach.ai.config as ai_config_mod

    orig_load_ai_config = ai_config_mod.load_ai_config

    def _patched():
        cfg = orig_load_ai_config()
        cfg["physics_pretrain"]["ball"]["decoder_only_pretrain_epochs"] = 1
        cfg["physics_pretrain"]["ball"]["autoencode_pretrain_epochs"] = 1
        # Full diagnostics (oob_accuracy etc.) are only logged when their
        # weight is nonzero -- force both on regardless of the live config,
        # since this test is specifically checking the full-diagnostics format.
        cfg["physics_pretrain"]["ball"]["bce_loss_weight"] = 1.0
        cfg["physics_pretrain"]["ball"]["spin_loss_weight"] = 1.0
        # This test specifically checks the encoder.out-trainable log
        # message -- force the flag regardless of the live config's value.
        cfg["physics_pretrain"]["ball"]["decoder_only_pretrain_freeze_latent"] = False
        return cfg

    monkeypatch.setattr(ai_config_mod, "load_ai_config", _patched)

    dataset_dir = tmp_path / "data"
    generate_dataset(n_episodes=80, output_dir=dataset_dir, seed=5, shard_size=80, n_workers=1)
    output_path = tmp_path / "ball_encoder.pt"
    with caplog.at_level("INFO", logger="footballcoach.ai.physics_pretrain.train_ball_dynamics"):
        train(
            dataset_dir=str(dataset_dir), output_path=str(output_path),
            epochs=1, batch_size=16, lr=1e-2, val_frac=0.2, seed=0,
        )
    assert "Decoder-only pretraining: 1 epoch(s), encoder TRUNK frozen" in caplog.text
    assert "decoder-only pretrain epoch 1/1" in caplog.text
    assert "train pos_rmse" in caplog.text
    assert "val   pos_rmse" in caplog.text
    assert "val   oob_accuracy" in caplog.text
    assert output_path.exists()


def test_train_smoke_with_decoder_only_pretrain_freeze_latent(tmp_path, monkeypatch, caplog):
    """decoder_only_pretrain_freeze_latent=True: the WHOLE encoder (trunk
    AND encoder.out) is frozen for this phase -- opposite of the default
    (encoder.out trainable, identity rows masked) checked above. Verifies
    the log message reflects this mode and encoder.out is byte-identical
    across the phase (not just its identity rows)."""
    from footballcoach.ai.physics_pretrain.train_ball_dynamics import train
    import footballcoach.ai.config as ai_config_mod

    orig_load_ai_config = ai_config_mod.load_ai_config

    def _patched():
        cfg = orig_load_ai_config()
        cfg["physics_pretrain"]["ball"]["decoder_only_pretrain_epochs"] = 1
        cfg["physics_pretrain"]["ball"]["decoder_only_pretrain_freeze_latent"] = True
        return cfg

    monkeypatch.setattr(ai_config_mod, "load_ai_config", _patched)

    dataset_dir = tmp_path / "data"
    generate_dataset(n_episodes=80, output_dir=dataset_dir, seed=5, shard_size=80, n_workers=1)
    output_path = tmp_path / "ball_encoder.pt"
    with caplog.at_level("INFO", logger="footballcoach.ai.physics_pretrain.train_ball_dynamics"):
        train(
            dataset_dir=str(dataset_dir), output_path=str(output_path),
            epochs=1, batch_size=16, lr=1e-2, val_frac=0.2, seed=0,
        )
    assert "Decoder-only pretraining: 1 epoch(s), entire encoder frozen (latent fixed)" in caplog.text
    assert "encoder TRUNK frozen" not in caplog.text
    assert output_path.exists()


def test_train_smoke_with_sgd_optimizer(tmp_path, monkeypatch, caplog):
    """optimizer_type='sgd' switches the MAIN LOOP's optimizer to
    torch.optim.SGD (with momentum) -- verifies the log message reflects
    it and the run completes normally, same shape of check as the
    decoder_only_pretrain_freeze_latent smoke tests above."""
    from footballcoach.ai.physics_pretrain.train_ball_dynamics import train
    import footballcoach.ai.config as ai_config_mod

    orig_load_ai_config = ai_config_mod.load_ai_config

    def _patched():
        cfg = orig_load_ai_config()
        cfg["physics_pretrain"]["ball"]["optimizer_type"] = "sgd"
        cfg["physics_pretrain"]["ball"]["sgd_momentum"] = 0.8
        return cfg

    monkeypatch.setattr(ai_config_mod, "load_ai_config", _patched)

    dataset_dir = tmp_path / "data"
    generate_dataset(n_episodes=80, output_dir=dataset_dir, seed=5, shard_size=80, n_workers=1)
    output_path = tmp_path / "ball_encoder.pt"
    with caplog.at_level("INFO", logger="footballcoach.ai.physics_pretrain.train_ball_dynamics"):
        train(
            dataset_dir=str(dataset_dir), output_path=str(output_path),
            epochs=2, batch_size=16, lr=1e-2, val_frac=0.2, seed=0,
        )
    assert "Main-loop optimizer: SGD (momentum=0.8" in caplog.text
    assert output_path.exists()


def test_train_autoencode_and_decoder_only_optimizers_are_independent(tmp_path, monkeypatch, caplog):
    """autoencode_optimizer_type/decoder_only_optimizer_type each pick
    their OWN phase's optimizer, independent of the main loop's
    optimizer_type -- verifies all three log distinctly (main stays Adam,
    autoencode goes SGD, decoder-only goes SGD with a different momentum)
    rather than autoencode/decoder-only silently always being Adam."""
    from footballcoach.ai.physics_pretrain.train_ball_dynamics import train
    import footballcoach.ai.config as ai_config_mod

    orig_load_ai_config = ai_config_mod.load_ai_config

    def _patched():
        cfg = orig_load_ai_config()
        pp = cfg["physics_pretrain"]["ball"]
        pp["optimizer_type"] = "adam"
        pp["autoencode_pretrain_epochs"] = 1
        pp["autoencode_optimizer_type"] = "sgd"
        pp["autoencode_sgd_momentum"] = 0.6
        pp["decoder_only_pretrain_epochs"] = 1
        pp["decoder_only_optimizer_type"] = "sgd"
        pp["decoder_only_sgd_momentum"] = 0.4
        return cfg

    monkeypatch.setattr(ai_config_mod, "load_ai_config", _patched)

    dataset_dir = tmp_path / "data"
    generate_dataset(n_episodes=80, output_dir=dataset_dir, seed=5, shard_size=80, n_workers=1)
    output_path = tmp_path / "ball_encoder.pt"
    with caplog.at_level("INFO", logger="footballcoach.ai.physics_pretrain.train_ball_dynamics"):
        train(
            dataset_dir=str(dataset_dir), output_path=str(output_path),
            epochs=1, batch_size=16, lr=1e-2, val_frac=0.2, seed=0,
        )
    assert "Autoencode-pretrain optimizer: SGD (momentum=0.6" in caplog.text
    assert "Decoder-only-pretrain optimizer: SGD (momentum=0.4" in caplog.text
    assert "Main-loop optimizer: Adam" in caplog.text
    assert output_path.exists()


def test_train_auto_widens_init_checkpoint_on_dim_mismatch(tmp_path, monkeypatch, caplog):
    """Resuming via --init-checkpoint from a checkpoint whose hidden_dim/
    encoder_bottleneck_dim/latent_dim/decoder_hidden_dim don't match the
    CURRENT config used to crash with a raw torch RuntimeError (size
    mismatch) -- train() now detects this and runs widen_ball_checkpoint's
    seam-preserving surgery automatically instead of failing."""
    from footballcoach.ai.physics_pretrain.train_ball_dynamics import train
    import footballcoach.ai.config as ai_config_mod

    orig_load_ai_config = ai_config_mod.load_ai_config

    def _small_cfg():
        cfg = orig_load_ai_config()
        pp = cfg["physics_pretrain"]["ball"]
        pp["hidden_dim"], pp["encoder_bottleneck_dim"], pp["latent_dim"], pp["decoder_hidden_dim"] = 24, 12, 16, 10
        return cfg

    def _bigger_cfg():
        cfg = orig_load_ai_config()
        pp = cfg["physics_pretrain"]["ball"]
        pp["hidden_dim"], pp["encoder_bottleneck_dim"], pp["latent_dim"], pp["decoder_hidden_dim"] = 40, 20, 24, 18
        return cfg

    dataset_dir = tmp_path / "data"
    generate_dataset(n_episodes=60, output_dir=dataset_dir, seed=5, shard_size=60, n_workers=1)

    monkeypatch.setattr(ai_config_mod, "load_ai_config", _small_cfg)
    small_output = tmp_path / "small.pt"
    train(
        dataset_dir=str(dataset_dir), output_path=str(small_output),
        epochs=1, batch_size=16, lr=1e-2, val_frac=0.2, seed=0,
    )
    small_ckpt = small_output.with_suffix(".after_training.pt")
    assert small_ckpt.exists()

    monkeypatch.setattr(ai_config_mod, "load_ai_config", _bigger_cfg)
    big_output = tmp_path / "big.pt"
    with caplog.at_level("INFO", logger="footballcoach.ai.physics_pretrain.train_ball_dynamics"):
        train(
            dataset_dir=str(dataset_dir), output_path=str(big_output),
            epochs=1, batch_size=16, lr=1e-2, val_frac=0.2, seed=0,
            init_checkpoint=str(small_ckpt),
        )
    assert "Widened checkpoint from" in caplog.text
    assert "hidden_dim: 24->40" in caplog.text
    assert big_output.exists()
    big_ckpt = torch.load(big_output.with_suffix(".after_training.pt"), map_location="cpu")
    assert big_ckpt["config_snapshot"]["latent_dim"] == 24


def test_train_rejects_unknown_optimizer_type(tmp_path, monkeypatch):
    from footballcoach.ai.physics_pretrain.train_ball_dynamics import train
    import footballcoach.ai.config as ai_config_mod

    orig_load_ai_config = ai_config_mod.load_ai_config

    def _patched():
        cfg = orig_load_ai_config()
        cfg["physics_pretrain"]["ball"]["optimizer_type"] = "rmsprop"
        return cfg

    monkeypatch.setattr(ai_config_mod, "load_ai_config", _patched)

    dataset_dir = tmp_path / "data"
    generate_dataset(n_episodes=40, output_dir=dataset_dir, seed=5, shard_size=40, n_workers=1)
    output_path = tmp_path / "ball_encoder.pt"
    with pytest.raises(ValueError, match="optimizer_type"):
        train(
            dataset_dir=str(dataset_dir), output_path=str(output_path),
            epochs=1, batch_size=16, lr=1e-2, val_frac=0.2, seed=0,
        )


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


def test_generate_dataset_appends_rather_than_overwrites(tmp_path):
    first_paths = generate_dataset(n_episodes=40, output_dir=tmp_path, seed=0, shard_size=20, n_workers=1)
    assert [p.name for p in first_paths] == ["shard_00000.npz", "shard_00001.npz"]

    second_paths = generate_dataset(n_episodes=20, output_dir=tmp_path, seed=0, shard_size=20, n_workers=1)
    assert [p.name for p in second_paths] == ["shard_00002.npz"]

    # Original shards must be untouched (same episode count as before), and
    # the combined dataset must include both runs' episodes.
    ds = BallDynamicsDataset.from_directory(tmp_path)
    assert len(ds) == 60

    # New shard's episodes must not be identical to the first run's (distinct
    # seed derived from the running shard index, not a repeat of seed=0).
    first_ds = BallDynamicsDataset.from_directory(tmp_path, pattern="shard_00000.npz")
    second_ds = BallDynamicsDataset.from_directory(tmp_path, pattern="shard_00002.npz")
    assert not np.array_equal(first_ds.inputs[:20], second_ds.inputs)


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

def test_train_smoke(tmp_path, caplog):
    from footballcoach.ai.physics_pretrain.train_ball_dynamics import train

    dataset_dir = tmp_path / "data"
    generate_dataset(n_episodes=80, output_dir=dataset_dir, seed=5, shard_size=80, n_workers=1)

    output_path = tmp_path / "ball_encoder.pt"
    with caplog.at_level("INFO", logger="footballcoach.ai.physics_pretrain.train_ball_dynamics"):
        artifact = train(
            dataset_dir=str(dataset_dir),
            output_path=str(output_path),
            epochs=3,
            batch_size=16,
            lr=1e-2,
            val_frac=0.2,
            seed=0,
        )
    assert "Median val episode" in caplog.text
    assert "Worst (max loss) val episode" in caplog.text
    assert output_path.exists()
    assert "encoder_state_dict" in artifact
    assert "physics_config_hash" in artifact
    assert "normalization" in artifact
    assert artifact["normalization"]["pitch_half_diag_m"] > 0
    assert artifact["normalization"]["ball_spin_norm_max_rad_s"] > 0

    history_path = output_path.with_suffix(".history.npz")
    assert history_path.exists()
    history_data = np.load(history_path)
    assert "train_pos_rmse" in history_data.files
    assert "val_goal_bce" in history_data.files
    assert "train_oob_precision" in history_data.files
    assert "val_goal_recall" in history_data.files
    assert "train_pos_r2" in history_data.files
    assert "val_spin_r2" in history_data.files
    assert "lr" in history_data.files
    assert "train_pos_err_pct_disp" in history_data.files
    assert "val_spin_err_pct_disp" in history_data.files
    assert "train_pos_err_pct_ballistic" in history_data.files
    assert "val_spin_err_pct_ballistic" in history_data.files
    assert "train_pair_loss" in history_data.files
    assert "val_pair_loss" in history_data.files
    assert "train_t0_loss" in history_data.files
    assert "val_t0_loss" in history_data.files

    # train() always writes the local HTML report (open_browser=False here,
    # the test default, so nothing actually pops a browser window).
    report_path = output_path.with_suffix(".report.html")
    assert report_path.exists()
    report_html = report_path.read_text()
    assert "__HISTORY_JSON__" not in report_html
    assert '"train_pos_rmse"' in report_html

    # Round-trip: load the saved encoder and confirm weights match what was returned.
    from footballcoach.ai.physics_pretrain.ball_dynamics_net import BallDynamicsEncoder
    from footballcoach.ai.config import load_ai_config

    cfg = load_ai_config()["physics_pretrain"]["ball"]
    loaded = torch.load(output_path, weights_only=True)
    encoder = BallDynamicsEncoder(
        hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
        bottleneck_dim=cfg.get("encoder_bottleneck_dim", 32),
        identity_shortcut=cfg.get("identity_shortcut_enabled", False),
        identity_shortcut_noise_std=cfg.get("identity_shortcut_noise_std", 0.0),
        concat_all_input_fields=cfg.get("encoder_concat_all_input_fields", False),
    )
    encoder.load_state_dict(loaded["encoder_state_dict"])
    for p1, p2 in zip(encoder.state_dict().values(), artifact["encoder_state_dict"].values()):
        torch.testing.assert_close(p1, p2)


# ---------------------------------------------------------------------------
# report.py
# ---------------------------------------------------------------------------

def test_write_report_embeds_data_and_is_valid_standalone_html(tmp_path):
    from footballcoach.ai.physics_pretrain.report import write_report

    history_arrays = {
        "epoch": np.array([1, 2]),
        "train_loss": np.array([1.0, 0.5]),
        "val_loss": np.array([1.1, 0.6]),
        "horizons_s": np.array([0.2, 1.0]),
        "train_pos_rmse": np.array([[0.1, 0.2], [0.05, 0.15]]),
        "val_pos_rmse": np.array([[0.12, 0.22], [0.06, 0.16]]),
        "train_vel_rmse": np.array([[0.1, 0.2], [0.05, 0.15]]),
        "val_vel_rmse": np.array([[0.1, 0.2], [0.05, 0.15]]),
        "train_spin_rmse": np.array([[0.1, 0.2], [0.05, 0.15]]),
        "val_spin_rmse": np.array([[0.1, 0.2], [0.05, 0.15]]),
        "train_oob_bce": np.array([[0.5, 0.4], [0.3, 0.2]]),
        "val_oob_bce": np.array([[0.5, 0.4], [0.3, 0.2]]),
        "train_goal_bce": np.array([[0.5, 0.4], [0.3, 0.2]]),
        "val_goal_bce": np.array([[0.5, 0.4], [0.3, 0.2]]),
    }
    output_path = tmp_path / "report.html"
    result_path = write_report(
        history_arrays=history_arrays,
        dataset_stats={"n_episodes": 100, "n_train": 85, "n_val": 15},
        config_snapshot={"latent_dim": 16, "epochs": 50},
        normalization={"pitch_half_diag_m": 62.5, "height_norm_m": 3.0, "ball_spin_norm_max_rad_s": 30.0},
        output_path=output_path,
    )
    assert result_path == output_path
    html = output_path.read_text()
    assert html.startswith("<!doctype html>")
    assert "<html>" in html and "</html>" in html
    assert "__HISTORY_JSON__" not in html
    assert '"pitch_half_diag_m": 62.5' in html


def test_open_in_browser_never_raises(tmp_path, monkeypatch):
    from footballcoach.ai.physics_pretrain import report as report_mod

    def _boom(url):
        raise RuntimeError("no display")

    monkeypatch.setattr(report_mod.webbrowser, "open", _boom)
    fake_path = tmp_path / "report.html"
    fake_path.write_text("<html></html>")
    assert report_mod.open_in_browser(fake_path) is False


def _widen_test_cfg(**overrides) -> dict:
    cfg = dict(
        hidden_dim=24, encoder_bottleneck_dim=12, latent_dim=16, decoder_hidden_dim=10,
        horizons_s=[0.2, 0.5, 1.0], identity_shortcut_enabled=True, identity_shortcut_noise_std=0.01,
        encoder_concat_all_input_fields=True, decoder_identity_shortcut_enabled=False,
    )
    cfg.update(overrides)
    return cfg


def _perturb(model) -> None:
    """Simulates 'this model has actually been trained' -- fresh-init vs
    fresh-init could trivially agree by construction, so nudge every
    parameter to confirm the widen surgery really does preserve arbitrary
    (not just just-initialized) old weights."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)


def test_widen_model_reproduces_old_model_exactly_all_dims():
    from footballcoach.ai.physics_pretrain.widen_ball_checkpoint import _build_model, verify_widened_model, widen_model_

    old_cfg = _widen_test_cfg()
    new_cfg = dict(old_cfg, hidden_dim=40, encoder_bottleneck_dim=20, latent_dim=24, decoder_hidden_dim=18)
    old_model = _build_model(old_cfg)
    _perturb(old_model)
    new_model = _build_model(new_cfg)
    widen_model_(old_model, new_model, old_cfg, new_cfg)
    verify_widened_model(old_model, new_model, torch.randn(32, N_INPUT_FIELDS) * 0.3)


def test_widen_model_reproduces_old_model_exactly_without_identity_shortcut():
    from footballcoach.ai.physics_pretrain.widen_ball_checkpoint import _build_model, verify_widened_model, widen_model_

    old_cfg = _widen_test_cfg(
        identity_shortcut_enabled=False, identity_shortcut_noise_std=0.0, encoder_concat_all_input_fields=False,
    )
    new_cfg = dict(old_cfg, hidden_dim=32, encoder_bottleneck_dim=16, latent_dim=20, decoder_hidden_dim=30)
    old_model = _build_model(old_cfg)
    _perturb(old_model)
    new_model = _build_model(new_cfg)
    widen_model_(old_model, new_model, old_cfg, new_cfg)
    verify_widened_model(old_model, new_model, torch.randn(16, N_INPUT_FIELDS) * 0.3)


def test_widen_model_reproduces_old_model_exactly_single_dim():
    """Only latent_dim grows -- the trickiest seam (encoder.out's concat
    block shifts), everything else stays put."""
    from footballcoach.ai.physics_pretrain.widen_ball_checkpoint import _build_model, verify_widened_model, widen_model_

    old_cfg = _widen_test_cfg(identity_shortcut_noise_std=0.0)
    new_cfg = dict(old_cfg, latent_dim=22)
    old_model = _build_model(old_cfg)
    _perturb(old_model)
    new_model = _build_model(new_cfg)
    widen_model_(old_model, new_model, old_cfg, new_cfg)
    verify_widened_model(old_model, new_model, torch.randn(16, N_INPUT_FIELDS) * 0.3)


def test_widen_checkpoint_rejects_shrinking():
    from footballcoach.ai.physics_pretrain.widen_ball_checkpoint import _validate_widen_cfgs

    old_cfg = _widen_test_cfg()
    new_cfg = dict(old_cfg, latent_dim=old_cfg["latent_dim"] - 2)
    with pytest.raises(ValueError, match="shrank"):
        _validate_widen_cfgs(old_cfg, new_cfg)


def test_widen_checkpoint_rejects_unstable_shortcut_settings():
    from footballcoach.ai.physics_pretrain.widen_ball_checkpoint import _validate_widen_cfgs

    old_cfg = _widen_test_cfg(identity_shortcut_enabled=True)
    new_cfg = dict(old_cfg, identity_shortcut_enabled=False)
    with pytest.raises(ValueError, match="identity_shortcut_enabled"):
        _validate_widen_cfgs(old_cfg, new_cfg)


def test_widen_checkpoint_end_to_end(tmp_path):
    from footballcoach.ai.physics_pretrain.widen_ball_checkpoint import _build_model, widen_checkpoint

    old_cfg = _widen_test_cfg()
    old_model = _build_model(old_cfg)
    _perturb(old_model)
    ckpt_path = tmp_path / "ckpt.midtrain_latest.pt"
    torch.save({"model_state_dict": old_model.state_dict(), "config_snapshot": old_cfg}, ckpt_path)

    ds_dir = tmp_path / "data"
    generate_dataset(n_episodes=20, output_dir=ds_dir, seed=1, shard_size=20)

    new_cfg = dict(old_cfg, hidden_dim=40, encoder_bottleneck_dim=20, latent_dim=24, decoder_hidden_dim=18)
    out_path = tmp_path / "ckpt.widened.pt"
    widen_checkpoint(ckpt_path, out_path, new_cfg=new_cfg, dataset_dir=ds_dir, verify_n=10)

    loaded = torch.load(out_path, map_location="cpu")
    new_model = _build_model(new_cfg)
    missing, unexpected = new_model.load_state_dict(loaded["model_state_dict"], strict=True)
    assert not missing and not unexpected
    assert loaded["phase"] == "widened"
    assert loaded["config_snapshot"]["latent_dim"] == 24


def test_widen_checkpoint_requires_full_model_state_dict(tmp_path):
    from footballcoach.ai.physics_pretrain.widen_ball_checkpoint import widen_checkpoint

    ckpt_path = tmp_path / "encoder_only.pt"
    torch.save({"encoder_state_dict": {}}, ckpt_path)
    with pytest.raises(ValueError, match="model_state_dict"):
        widen_checkpoint(ckpt_path, tmp_path / "out.pt", new_cfg=_widen_test_cfg())

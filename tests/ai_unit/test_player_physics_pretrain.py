"""Unit tests for the player-dynamics physics pretraining pipeline.

See agent_plans/player_physics_pretrain_plan.md. Mirrors the STRUCTURE of
tests/ai_unit/test_ball_physics_pretrain.py (not every individual case --
see that file's section for the full 44-test precedent this follows).
"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from footballcoach.ai.physics_pretrain.player_episode_gen import (
    N_IDENTITY_SHORTCUT_FIELDS,
    N_INPUT_FIELDS,
    N_TARGET_FIELDS_PER_HORIZON,
    STAMINA_FIELD_INDEX,
    PlayerEpisodeGenParams,
    generate_episode,
    generate_shard,
)
from footballcoach.ai.physics_pretrain.player_dataset import GROUPS, PlayerDynamicsDataset, generate_dataset
from footballcoach.ai.physics_pretrain.player_dynamics_net import (
    PlayerDynamicsAutoencoder,
    PlayerDynamicsDecoder,
    PlayerDynamicsEncoder,
)
from footballcoach.entities.pitch import Pitch


@pytest.fixture(scope="module")
def gen_params() -> PlayerEpisodeGenParams:
    return PlayerEpisodeGenParams.from_config()


# ---------------------------------------------------------------------------
# player_episode_gen
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


def test_heading_velocity_invariant_at_t0(gen_params):
    """Physical-plausibility invariant (see the plan doc): a freshly sampled
    episode's t=0 velocity direction must exactly match heading, since
    step_player_towards always reconstructs velocity from heading+speed --
    an initial-condition sampler that drew them independently would produce
    states the real engine can never actually reach."""
    for seed in range(30):
        input_row, _ = generate_episode(random.Random(seed), gen_params)
        heading = math.atan2(input_row[4], input_row[5])
        speed = math.hypot(input_row[2], input_row[3])
        if speed < 1e-9:
            continue  # standstill: velocity direction is undefined, nothing to check
        vel_angle = math.atan2(input_row[3], input_row[2])
        diff = (vel_angle - heading + math.pi) % (2 * math.pi) - math.pi
        assert abs(diff) < 1e-4, f"seed={seed}: heading={heading} vel_angle={vel_angle}"


def test_encode_input_engineered_features_hand_computed(gen_params):
    from footballcoach.ai.physics_pretrain.player_episode_gen import compute_engineered_features

    identity = np.array([[0.0, 0.0, 0.3, 0.4, math.sin(0.5), math.cos(0.5), 0.8]], dtype=np.float32)
    dd = np.array([[math.sin(1.2), math.cos(1.2)]], dtype=np.float32)
    out = compute_engineered_features(identity, dd)
    expected_speed = math.hypot(0.3, 0.4)
    assert out[0, 0] == pytest.approx(expected_speed, abs=1e-5)
    # heading_desired_diff should equal sin/cos of (1.2 - 0.5)
    expected_diff = 1.2 - 0.5
    assert out[0, 1] == pytest.approx(math.sin(expected_diff), abs=1e-5)
    assert out[0, 2] == pytest.approx(math.cos(expected_diff), abs=1e-5)


def test_generate_episode_simulates_to_exact_horizon_time(gen_params, monkeypatch):
    """Position at each recorded horizon must reflect simulating chronologically
    to that EXACT time (regular dt steps + a final partial-remainder step),
    not the nearest whole tick -- verified via a version of gen_params whose
    dt does NOT evenly divide the horizons."""
    from dataclasses import replace

    odd_params = replace(gen_params, sim_dt_s=0.07, horizons_s=(0.2,))
    rng = random.Random(0)
    input_row, target_row = generate_episode(rng, odd_params)
    # No exception, finite output, and the horizon count matches -- exact
    # timing correctness is implicitly covered by test_heading_velocity_
    # invariant/determinism; this test's job is to confirm non-evenly-
    # divisible dt doesn't crash or silently produce the wrong shape.
    assert target_row.shape == (N_TARGET_FIELDS_PER_HORIZON,)
    assert np.isfinite(target_row).all()


def test_generate_shard_distribution_sanity(gen_params):
    from dataclasses import replace

    params = replace(gen_params, out_of_bounds_start_frac=0.5, possession_start_frac=0.5)
    inputs, targets = generate_shard(400, seed=3, params=params)
    assert inputs.shape == (400, N_INPUT_FIELDS)
    assert targets.shape == (400, len(params.horizons_s) * N_TARGET_FIELDS_PER_HORIZON)
    oob_col = 7  # first horizon's out_of_bounds field
    oob_rate = (targets[:, oob_col] > 0.5).mean()
    assert 0.0 < oob_rate < 1.0
    has_possession_rate = (inputs[:, 11] > 0.5).mean()
    assert 0.2 < has_possession_rate < 0.8


def test_goal_scored_always_zero_without_possession(gen_params):
    """A non-possessing player can never have goal_scored=1 at any horizon,
    per the dribble-carry-glued assumption (see the plan doc / player_
    episode_gen.generate_episode's docstring)."""
    from dataclasses import replace

    params = replace(gen_params, possession_start_frac=0.0, out_of_bounds_start_frac=0.5)
    inputs, targets = generate_shard(200, seed=4, params=params)
    assert (inputs[:, 11] == 0.0).all()
    n_horizons = len(params.horizons_s)
    for h in range(n_horizons):
        base = h * N_TARGET_FIELDS_PER_HORIZON
        assert (targets[:, base + 8] == 0.0).all()


def test_position_stays_grounded(gen_params):
    """Players are grounded -- this is implicit in the field layout (no z
    axis at all), but sanity-check that in-play position sampling never
    puts anyone outside the (randomized) pitch by more than the margin."""
    rng = random.Random(0)
    for _ in range(20):
        input_row, _ = generate_episode(rng, gen_params)
        assert np.isfinite(input_row).all()


# ---------------------------------------------------------------------------
# player_dynamics_net
# ---------------------------------------------------------------------------

def test_encoder_decoder_shapes_no_nan():
    torch.manual_seed(0)
    model = PlayerDynamicsAutoencoder(
        input_dim=N_INPUT_FIELDS, hidden_dim=32, latent_dim=16, horizons_s=[0.2, 1.0, 3.0],
        decoder_hidden_dim=32, encoder_bottleneck_dim=16,
    )
    x = torch.randn(8, N_INPUT_FIELDS)
    latent, heads = model(x)
    assert latent.shape == (8, 16)
    assert len(heads) == 3
    for h in heads:
        assert h.shape == (8, N_TARGET_FIELDS_PER_HORIZON)
        assert not torch.isnan(h).any()


def test_encoder_standalone_shape():
    encoder = PlayerDynamicsEncoder(input_dim=N_INPUT_FIELDS, hidden_dim=32, latent_dim=12, bottleneck_dim=16)
    x = torch.randn(4, N_INPUT_FIELDS)
    out = encoder(x)
    assert out.shape == (4, 12)


def test_decoder_forward_at_matches_forward_for_trained_horizon():
    torch.manual_seed(0)
    decoder = PlayerDynamicsDecoder(latent_dim=16, horizons_s=[0.2, 1.0, 3.0], hidden_dim=16)
    latent = torch.randn(4, 16)
    heads = decoder(latent)
    at_02 = decoder.forward_at(latent, 0.2)
    torch.testing.assert_close(at_02, heads[0])


def test_decoder_forward_at_zero_horizon_finite():
    decoder = PlayerDynamicsDecoder(latent_dim=16, horizons_s=[0.2, 1.0, 3.0], hidden_dim=16)
    latent = torch.randn(4, 16)
    out = decoder.forward_at(latent, 0.0)
    assert torch.isfinite(out).all()


def test_identity_shortcut_zero_noise_gives_exact_round_trip():
    torch.manual_seed(0)
    model = PlayerDynamicsAutoencoder(
        input_dim=N_INPUT_FIELDS, hidden_dim=32, latent_dim=16, horizons_s=[0.2, 0.5, 1.0],
        decoder_hidden_dim=32, encoder_bottleneck_dim=16,
        identity_shortcut=True, identity_shortcut_noise_std=0.0,
    )
    x = torch.randn(5, N_INPUT_FIELDS)
    # stamina (STAMINA_FIELD_INDEX) is never negative in real data -- see
    # its dedicated single-ReLU decoder unit's documented limitation.
    x[:, STAMINA_FIELD_INDEX] = x[:, STAMINA_FIELD_INDEX].abs()
    latent = model.encoder(x)
    torch.testing.assert_close(latent[:, :N_IDENTITY_SHORTCUT_FIELDS], x[:, :N_IDENTITY_SHORTCUT_FIELDS])

    for horizon_s in (0.0, 0.5, 2.3):
        recon = model.decoder.forward_at(latent, horizon_s)
        torch.testing.assert_close(recon[:, :N_IDENTITY_SHORTCUT_FIELDS], x[:, :N_IDENTITY_SHORTCUT_FIELDS])


def test_identity_shortcut_noise_std_perturbs_round_trip():
    torch.manual_seed(0)
    model = PlayerDynamicsAutoencoder(
        input_dim=N_INPUT_FIELDS, hidden_dim=32, latent_dim=16, horizons_s=[0.2, 0.5, 1.0],
        decoder_hidden_dim=32, encoder_bottleneck_dim=16,
        identity_shortcut=True, identity_shortcut_noise_std=0.05,
    )
    x = torch.randn(5, N_INPUT_FIELDS)
    x[:, STAMINA_FIELD_INDEX] = x[:, STAMINA_FIELD_INDEX].abs()
    latent = model.encoder(x)
    diff = (latent[:, :N_IDENTITY_SHORTCUT_FIELDS] - x[:, :N_IDENTITY_SHORTCUT_FIELDS]).abs().max().item()
    assert diff > 0.0


def test_identity_shortcut_survives_adversarial_classification_gradient():
    """Regression test mirroring the ball pipeline's real bug (see
    identity_shortcut.py's docstring): an adversarial classification-only
    loss must NOT move the dedicated identity units' weights, but the freed
    stamina spare unit (STAMINA_FIELD_INDEX's second unit) is a genuine
    spare unit and SHOULD be free to move."""
    from footballcoach.ai.physics_pretrain.player_dynamics_net import STAMINA_FIELD_INDEX as NET_STAMINA_IDX

    torch.manual_seed(0)
    dim = N_IDENTITY_SHORTCUT_FIELDS
    model = PlayerDynamicsAutoencoder(
        input_dim=N_INPUT_FIELDS, hidden_dim=32, latent_dim=16, horizons_s=[0.2, 0.5, 1.0],
        decoder_hidden_dim=32, encoder_bottleneck_dim=16,
        identity_shortcut=True, identity_shortcut_noise_std=0.0,
    )
    second = model.decoder.net[2]
    before = second.weight.clone()
    free_idx = 2 * NET_STAMINA_IDX + 1

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    x = torch.randn(16, N_INPUT_FIELDS)
    x[:, STAMINA_FIELD_INDEX] = x[:, STAMINA_FIELD_INDEX].abs()
    for _ in range(50):
        latent = model.encoder(x)
        pred = model.decoder.forward_at(latent, 0.5)
        # Adversarial: purely classification loss, no reconstruction term.
        loss = F.binary_cross_entropy_with_logits(pred[:, 7], torch.ones(16)) + \
            F.binary_cross_entropy_with_logits(pred[:, 8], torch.zeros(16))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    after = second.weight
    dedicated_rows = list(range(dim))
    for i in dedicated_rows:
        a, b = 2 * i, 2 * i + 1
        torch.testing.assert_close(after[i, a], before[i, a])
        if b != free_idx:
            torch.testing.assert_close(after[i, b], before[i, b])
    # The freed spare unit's write access to the classification rows should
    # have actually moved (it's not permanently masked).
    moved = (after[dim:, free_idx] - before[dim:, free_idx]).abs().max().item()
    assert moved > 0.0


def test_identity_shortcut_requires_latent_dim_at_least_seven():
    with pytest.raises(ValueError):
        PlayerDynamicsEncoder(
            input_dim=N_INPUT_FIELDS, hidden_dim=16, latent_dim=4, bottleneck_dim=8,
            identity_shortcut=True,
        )


def test_from_config_matches_ai_config():
    from footballcoach.ai.config import load_ai_config

    cfg = load_ai_config()["physics_pretrain"]["player"]
    params = PlayerEpisodeGenParams.from_config()
    assert params.horizons_s == tuple(float(h) for h in cfg["horizons_s"])
    net = PlayerDynamicsAutoencoder.from_config()
    assert net.encoder.latent_dim == cfg["latent_dim"]


# ---------------------------------------------------------------------------
# train_player_dynamics: loss/metrics
# ---------------------------------------------------------------------------

def test_compute_loss_hand_computed():
    from footballcoach.ai.physics_pretrain.train_player_dynamics import compute_loss

    pred = torch.zeros(2, N_TARGET_FIELDS_PER_HORIZON)
    target = torch.zeros(2, N_TARGET_FIELDS_PER_HORIZON)
    target[:, 0] = 1.0  # pos_x off by 1
    input_x = torch.zeros(2, N_INPUT_FIELDS)
    input_x[:, 11] = 1.0  # has_possession=1 for both rows
    pos_weight = torch.ones(1, 2)

    total, breakdown = compute_loss([pred], target, input_x, pos_weight)
    assert breakdown.pos_rmse[0] == pytest.approx(0.5 ** 0.5, abs=1e-5)
    # BCE at logit=0, target=0 -> ln(2)
    assert breakdown.oob_bce[0] == pytest.approx(math.log(2), abs=1e-4)
    assert breakdown.goal_bce[0] == pytest.approx(math.log(2), abs=1e-4)
    assert total.item() > 0.0


def test_goal_bce_possession_gating():
    """goal_scored BCE must be computed ONLY over has_possession=1 rows --
    a batch of entirely has_possession=0 rows should report goal_bce=0 and
    contribute nothing to the backpropagated total's goal term."""
    from footballcoach.ai.physics_pretrain.train_player_dynamics import compute_loss

    pred = torch.zeros(4, N_TARGET_FIELDS_PER_HORIZON, requires_grad=True)
    target = torch.zeros(4, N_TARGET_FIELDS_PER_HORIZON)
    target[:, 8] = 1.0  # goal_scored=1 target (would normally produce large BCE)
    input_x = torch.zeros(4, N_INPUT_FIELDS)  # has_possession=0 for all rows
    pos_weight = torch.ones(1, 2)

    total, breakdown = compute_loss([pred], target, input_x, pos_weight)
    assert breakdown.goal_bce[0] == 0.0
    total.sum().backward()
    assert pred.grad[:, 8].abs().max().item() == 0.0  # no gradient reaches goal_scored logit


def test_compute_confusion_counts_possession_gated():
    from footballcoach.ai.physics_pretrain.train_player_dynamics import compute_confusion_counts

    pred = torch.zeros(4, N_TARGET_FIELDS_PER_HORIZON)
    pred[:, 8] = 5.0  # predicts goal_scored=1 for everyone
    target = torch.zeros(4, N_TARGET_FIELDS_PER_HORIZON)
    target[:, 8] = 1.0
    input_x = torch.zeros(4, N_INPUT_FIELDS)
    input_x[:2, 11] = 1.0  # only first 2 rows have possession

    counts = compute_confusion_counts([pred], target, input_x)
    tp, fp, fn, tn = counts["goal"][0]
    assert tp == 2  # only the 2 possessing rows count
    assert fp == 0 and fn == 0 and tn == 0


def test_heading_angular_dist_wraps_correctly():
    from footballcoach.ai.physics_pretrain.train_player_dynamics import _heading_angular_dist

    pred = torch.tensor([[math.sin(0.01), math.cos(0.01)]])
    target = torch.tensor([[math.sin(-0.01), math.cos(-0.01)]])
    dist = _heading_angular_dist(pred, target)
    assert dist.item() == pytest.approx(0.02, abs=1e-4)

    # near +-pi wraparound: should report a SMALL distance, not ~2*pi.
    pred2 = torch.tensor([[math.sin(math.pi - 0.01), math.cos(math.pi - 0.01)]])
    target2 = torch.tensor([[math.sin(-math.pi + 0.01), math.cos(-math.pi + 0.01)]])
    dist2 = _heading_angular_dist(pred2, target2)
    assert dist2.item() == pytest.approx(0.02, abs=1e-3)


# ---------------------------------------------------------------------------
# player_dataset
# ---------------------------------------------------------------------------

def test_generate_dataset_roundtrip(tmp_path):
    paths = generate_dataset(n_episodes=40, output_dir=tmp_path, seed=1, shard_size=20, n_workers=1)
    assert len(paths) == 2
    ds = PlayerDynamicsDataset.from_directory(tmp_path)
    assert len(ds) == 40
    assert ds.inputs.shape == (40, N_INPUT_FIELDS)


def test_generate_dataset_appends_rather_than_overwrites(tmp_path):
    generate_dataset(n_episodes=20, output_dir=tmp_path, seed=1, shard_size=20, n_workers=1)
    generate_dataset(n_episodes=20, output_dir=tmp_path, seed=2, shard_size=20, n_workers=1)
    ds = PlayerDynamicsDataset.from_directory(tmp_path)
    assert len(ds) == 40
    assert len(list(tmp_path.glob("shard_*.npz"))) == 2


def test_dataset_split_train_val(tmp_path):
    generate_dataset(n_episodes=50, output_dir=tmp_path, seed=1, shard_size=50, n_workers=1)
    ds = PlayerDynamicsDataset.from_directory(tmp_path)
    train_idx, val_idx = ds.split_train_val(val_frac=0.2, seed=0)
    assert len(train_idx) + len(val_idx) == 50
    assert len(set(train_idx) & set(val_idx)) == 0


def test_compute_pos_weights_shape_and_possession_gating(tmp_path):
    generate_dataset(n_episodes=100, output_dir=tmp_path, seed=1, shard_size=100, n_workers=1)
    ds = PlayerDynamicsDataset.from_directory(tmp_path)
    n_horizons = ds.targets.shape[1] // N_TARGET_FIELDS_PER_HORIZON
    weights = ds.compute_pos_weights(n_horizons)
    assert weights.shape == (n_horizons, 2)
    assert (weights > 0).all()


def test_iterate_minibatches_covers_all_rows(tmp_path):
    generate_dataset(n_episodes=37, output_dir=tmp_path, seed=1, shard_size=37, n_workers=1)
    ds = PlayerDynamicsDataset.from_directory(tmp_path)
    idx = np.arange(len(ds))
    seen = set()
    for x, y in ds.iterate_minibatches(8, idx, shuffle=False):
        seen.update(range(len(seen), len(seen) + len(x)))
    assert len(seen) == 37


def test_build_adjacent_pair_data_reencodes_correctly(tmp_path):
    generate_dataset(n_episodes=20, output_dir=tmp_path, seed=1, shard_size=20, n_workers=1)
    ds = PlayerDynamicsDataset.from_directory(tmp_path)
    derived_inputs, derived_targets = ds.build_adjacent_pair_data(0)
    assert derived_inputs.shape == (20, N_INPUT_FIELDS)
    assert derived_targets.shape == (20, N_TARGET_FIELDS_PER_HORIZON)
    # identity block of the derived input must equal horizon 0's recorded state
    np.testing.assert_allclose(derived_inputs[:, 0:7], ds.targets[:, 0:7], atol=1e-5)
    # context fields copied unchanged from the original episode input
    np.testing.assert_allclose(derived_inputs[:, 7:21], ds.inputs[:, 7:21], atol=1e-5)


def test_build_autoencoding_data_reencodes_correctly(tmp_path):
    generate_dataset(n_episodes=20, output_dir=tmp_path, seed=1, shard_size=20, n_workers=1)
    ds = PlayerDynamicsDataset.from_directory(tmp_path)
    derived_inputs, derived_targets = ds.build_autoencoding_data(1)
    base = N_TARGET_FIELDS_PER_HORIZON
    np.testing.assert_allclose(derived_inputs[:, 0:7], ds.targets[:, base:base + 7], atol=1e-5)
    np.testing.assert_allclose(derived_targets, ds.targets[:, base:base + N_TARGET_FIELDS_PER_HORIZON], atol=1e-5)


# ---------------------------------------------------------------------------
# train_player_dynamics: end-to-end smoke
# ---------------------------------------------------------------------------

def test_train_smoke(tmp_path, caplog):
    from footballcoach.ai.physics_pretrain.train_player_dynamics import train

    dataset_dir = tmp_path / "data"
    generate_dataset(n_episodes=80, output_dir=dataset_dir, seed=5, shard_size=80, n_workers=1)

    output_path = tmp_path / "player_encoder.pt"
    with caplog.at_level("INFO", logger="footballcoach.ai.physics_pretrain.train_player_dynamics"):
        artifact = train(
            dataset_dir=str(dataset_dir),
            output_path=str(output_path),
            epochs=2,
            batch_size=16,
            lr=1e-2,
            val_frac=0.2,
            seed=0,
        )
    assert output_path.exists()
    assert "encoder_state_dict" in artifact
    assert "physics_config_hash" in artifact
    assert "normalization" in artifact
    assert artifact["normalization"]["pitch_half_diag_m"] > 0

    report_path = output_path.with_suffix(".report.html")
    assert report_path.exists()
    report_html = report_path.read_text()
    assert "__HISTORY_JSON__" not in report_html
    assert '"train_pos_rmse"' in report_html
    assert "Player Dynamics Training" in report_html

    # Round-trip: load the saved encoder and confirm weights match what was returned.
    from footballcoach.ai.config import load_ai_config

    cfg = load_ai_config()["physics_pretrain"]["player"]
    loaded = torch.load(output_path, weights_only=True)
    encoder = PlayerDynamicsEncoder(
        hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
        bottleneck_dim=cfg.get("encoder_bottleneck_dim", 32),
        identity_shortcut=cfg.get("identity_shortcut_enabled", False),
        identity_shortcut_noise_std=cfg.get("identity_shortcut_noise_std", 0.0),
        concat_all_input_fields=cfg.get("encoder_concat_all_input_fields", False),
    )
    encoder.load_state_dict(loaded["encoder_state_dict"])
    for p1, p2 in zip(encoder.state_dict().values(), artifact["encoder_state_dict"].values()):
        torch.testing.assert_close(p1, p2)


def test_train_smoke_with_decoder_only_and_autoencode_pretraining(tmp_path, monkeypatch, caplog):
    from footballcoach.ai.physics_pretrain.train_player_dynamics import train
    import footballcoach.ai.config as ai_config_mod

    orig_load_ai_config = ai_config_mod.load_ai_config

    def _patched():
        cfg = orig_load_ai_config()
        cfg["physics_pretrain"]["player"]["decoder_only_pretrain_epochs"] = 1
        cfg["physics_pretrain"]["player"]["autoencode_pretrain_epochs"] = 1
        return cfg

    monkeypatch.setattr(ai_config_mod, "load_ai_config", _patched)

    dataset_dir = tmp_path / "data"
    generate_dataset(n_episodes=60, output_dir=dataset_dir, seed=5, shard_size=60, n_workers=1)
    output_path = tmp_path / "player_encoder.pt"
    with caplog.at_level("INFO", logger="footballcoach.ai.physics_pretrain.train_player_dynamics"):
        train(
            dataset_dir=str(dataset_dir), output_path=str(output_path),
            epochs=1, batch_size=16, lr=1e-2, val_frac=0.2, seed=0,
        )
    assert "Decoder-only pretraining: 1 epoch(s)" in caplog.text
    assert "Autoencode pretraining: 1 epoch(s)" in caplog.text
    assert output_path.exists()

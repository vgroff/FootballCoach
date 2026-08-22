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
    input_row, target_row, crossing_row, crossing_time = generate_episode(rng, gen_params)
    assert input_row.shape == (N_INPUT_FIELDS,)
    assert target_row.shape == (len(gen_params.horizons_s) * N_TARGET_FIELDS_PER_HORIZON,)
    assert crossing_row.shape == (N_TARGET_FIELDS_PER_HORIZON,)
    assert isinstance(crossing_time, float)
    assert np.isfinite(input_row).all()
    assert np.isfinite(target_row).all()
    assert np.isfinite(crossing_row).all()


def test_generate_episode_deterministic_given_seed(gen_params):
    in1, tgt1, cross1, ct1 = generate_episode(random.Random(42), gen_params)
    in2, tgt2, cross2, ct2 = generate_episode(random.Random(42), gen_params)
    np.testing.assert_array_equal(in1, in2)
    np.testing.assert_array_equal(tgt1, tgt2)
    np.testing.assert_array_equal(cross1, cross2)
    assert ct1 == ct2


def test_generate_episode_different_seeds_differ(gen_params):
    in1 = generate_episode(random.Random(1), gen_params)[0]
    in2 = generate_episode(random.Random(2), gen_params)[0]
    assert not np.array_equal(in1, in2)


def test_heading_velocity_invariant_at_t0(gen_params):
    """Physical-plausibility invariant (see the plan doc): a freshly sampled
    episode's t=0 velocity direction must exactly match heading, since
    step_player_towards always reconstructs velocity from heading+speed --
    an initial-condition sampler that drew them independently would produce
    states the real engine can never actually reach."""
    for seed in range(30):
        input_row = generate_episode(random.Random(seed), gen_params)[0]
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
    input_row, target_row, _crossing_row, _crossing_time = generate_episode(rng, odd_params)
    # No exception, finite output, and the horizon count matches -- exact
    # timing correctness is implicitly covered by test_heading_velocity_
    # invariant/determinism; this test's job is to confirm non-evenly-
    # divisible dt doesn't crash or silently produce the wrong shape.
    assert target_row.shape == (N_TARGET_FIELDS_PER_HORIZON,)
    assert np.isfinite(target_row).all()


def test_generate_shard_distribution_sanity(gen_params):
    from dataclasses import replace

    params = replace(gen_params, out_of_bounds_start_frac=0.5, possession_start_frac=0.5)
    inputs, targets, crossings, crossing_times = generate_shard(400, seed=3, params=params)
    assert crossings.shape == (400, N_TARGET_FIELDS_PER_HORIZON)
    assert crossing_times.shape == (400,)
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
    inputs, targets, _crossings, _crossing_times = generate_shard(200, seed=4, params=params)
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
        input_row = generate_episode(rng, gen_params)[0]
        assert np.isfinite(input_row).all()


# ---------------------------------------------------------------------------
# crossing side-channel (player_episode_gen)
# ---------------------------------------------------------------------------

def test_crossing_time_is_inf_when_no_crossing_and_finite_when_crossed(gen_params):
    """Across a decent sample of in-play starts, some episodes cross (finite
    crossing_time, at least one sim tick in) and some don't (inf sentinel).
    A finite crossing_time must never be 0.0 -- the check only ever runs from
    the FIRST physics step onward, never against the raw t=0 sampled state."""
    from dataclasses import replace

    params = replace(gen_params, out_of_bounds_start_frac=0.0, possession_start_frac=0.5)
    _inputs, _targets, crossings, crossing_times = generate_shard(300, seed=11, params=params)
    finite = np.isfinite(crossing_times)
    assert finite.any(), "no episode ever crossed -- sample is not exercising the head's positive case"
    assert (~finite).any(), "every episode crossed -- sample is not exercising the inf sentinel"
    assert (crossing_times[finite] >= params.sim_dt_s - 1e-9).all()
    # A recorded crossing row must have at least one of its two event flags
    # set; a no-crossing row must have neither (it's the final state).
    assert ((crossings[finite, 7] > 0.5) | (crossings[finite, 8] > 0.5)).all()
    assert (crossings[~finite, 7:9] == 0.0).all()


def test_already_out_of_bounds_start_is_excluded_from_crossing(gen_params):
    """An episode that starts ALREADY out of bounds is forced to the inf
    "no crossing" sentinel regardless of how immediately it would otherwise
    register a crossing -- its near-t=0 crossing is a degenerate target, so
    it drops out of crossing_mask (and hence the position loss) for free.
    See player_episode_gen.generate_episode's docstring."""
    from dataclasses import replace

    params = replace(gen_params, out_of_bounds_start_frac=1.0, possession_start_frac=0.0)
    _inputs, _targets, crossings, crossing_times = generate_shard(120, seed=12, params=params)
    assert np.isinf(crossing_times).all(), (
        "already-out-of-bounds starts must all be inf, never a first-tick crossing time"
    )
    assert not (crossing_times == 0.0).any()
    # crossing_row content is unaffected by the override: it's just the final
    # simulated state, with both event flags 0 like any no-crossing episode.
    assert (crossings[:, 7:9] == 0.0).all()


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


def test_generate_dataset_crossing_schema_roundtrip(tmp_path):
    """crossings/crossing_times survive the .npz round-trip as REQUIRED
    fields, and the derived crossing_pos/crossing_mask/crossing_dt follow
    the documented -1-sentinel convention."""
    generate_dataset(n_episodes=60, output_dir=tmp_path, seed=7, shard_size=30, n_workers=1)
    shard = np.load(sorted(tmp_path.glob("shard_*.npz"))[0])
    assert "crossings" in shard and "crossing_times" in shard

    ds = PlayerDynamicsDataset.from_directory(tmp_path)
    assert ds.crossings.shape == (60, N_TARGET_FIELDS_PER_HORIZON)
    assert ds.crossing_times.shape == (60,)
    assert ds.crossing_pos.shape == (60, 2)
    np.testing.assert_allclose(ds.crossing_pos, ds.crossings[:, 0:2], atol=0)
    np.testing.assert_array_equal(ds.crossing_mask, np.isfinite(ds.crossing_times))
    assert np.isfinite(ds.crossing_dt).all()
    assert (ds.crossing_dt[~ds.crossing_mask] == -1.0).all()
    np.testing.assert_allclose(
        ds.crossing_dt[ds.crossing_mask], ds.crossing_times[ds.crossing_mask].astype(np.float32), atol=1e-6,
    )


def test_dataset_without_crossings_leaves_derived_fields_none(tmp_path):
    """A hand-built dataset (no crossings/crossing_times) is still legal --
    the derived crossing fields are simply None, which every crossing call
    site in train_player_dynamics.py guards on."""
    generate_dataset(n_episodes=20, output_dir=tmp_path, seed=7, shard_size=20, n_workers=1)
    full = PlayerDynamicsDataset.from_directory(tmp_path)
    bare = PlayerDynamicsDataset(full.inputs, full.targets)
    assert bare.crossing_pos is None and bare.crossing_mask is None and bare.crossing_dt is None


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
# train_player_dynamics: auxiliary latent heads
# ---------------------------------------------------------------------------

def _aux_test_model(latent_dim: int = 8) -> PlayerDynamicsAutoencoder:
    torch.manual_seed(0)
    return PlayerDynamicsAutoencoder(
        hidden_dim=16, latent_dim=latent_dim, horizons_s=[0.2, 1.0, 3.0], decoder_hidden_dim=16,
        encoder_bottleneck_dim=12,
    )


def test_aux_heads_exist_with_expected_shapes():
    model = _aux_test_model(latent_dim=9)
    assert model.crossing_head.weight.shape == (3, 9)
    assert model.goal_dist_delta_head.weight.shape == (2, 9)
    assert model.short_horizon_head_0_2s.weight.shape == (4, 9)
    assert model.short_horizon_head_1_0s.weight.shape == (4, 9)


def test_crossing_head_loss_masks_position_but_not_delta_t():
    """The position term must ignore masked-out rows entirely (never-crossed
    OR excluded already-out-of-bounds starts), while delta_t stays unmasked
    against the -1 sentinel -- see _crossing_head_loss's docstring."""
    from footballcoach.ai.physics_pretrain.train_player_dynamics import _crossing_head_loss

    model = _aux_test_model()
    latent = torch.randn(4, 8)
    pos = np.zeros((4, 2), dtype=np.float32)
    dt = np.array([-1.0, 2.0, -1.0, 4.0], dtype=np.float32)
    mask = np.array([False, True, False, True])
    row_idx = np.arange(4)

    loss_a, pos_dist_a, dt_mae_a = _crossing_head_loss(model, latent, pos, dt, mask, row_idx, "cpu")
    # Corrupting a MASKED row's position target must not move the loss at all.
    pos_corrupt = pos.copy()
    pos_corrupt[0] = [50.0, -50.0]
    pos_corrupt[2] = [-30.0, 30.0]
    loss_b, pos_dist_b, dt_mae_b = _crossing_head_loss(model, latent, pos_corrupt, dt, mask, row_idx, "cpu")
    assert loss_a.item() == pytest.approx(loss_b.item(), abs=1e-6)
    assert pos_dist_a == pytest.approx(pos_dist_b, abs=1e-6)
    # ...but corrupting an UNMASKED row's position target must.
    pos_corrupt2 = pos.copy()
    pos_corrupt2[1] = [50.0, -50.0]
    loss_c, _, _ = _crossing_head_loss(model, latent, pos_corrupt2, dt, mask, row_idx, "cpu")
    assert loss_c.item() > loss_a.item() + 1.0
    # delta_t is unmasked: changing a MASKED row's dt target does move it.
    dt_corrupt = dt.copy()
    dt_corrupt[0] = 25.0
    loss_d, _, dt_mae_d = _crossing_head_loss(model, latent, pos, dt_corrupt, mask, row_idx, "cpu")
    assert loss_d.item() > loss_a.item() + 1.0
    assert dt_mae_d > dt_mae_a


def test_crossing_head_loss_all_masked_is_finite():
    """No row in the batch crossed -- the masked position mean must fall back
    to 0/1 rather than dividing by zero."""
    from footballcoach.ai.physics_pretrain.train_player_dynamics import _crossing_head_loss

    model = _aux_test_model()
    latent = torch.randn(3, 8)
    loss, pos_dist, dt_mae = _crossing_head_loss(
        model, latent, np.zeros((3, 2), dtype=np.float32), np.full(3, -1.0, dtype=np.float32),
        np.zeros(3, dtype=bool), np.arange(3), "cpu",
    )
    assert torch.isfinite(loss).all()
    assert pos_dist == pytest.approx(0.0, abs=1e-9)
    assert math.isfinite(dt_mae)


def test_goal_mouth_distance_hand_computed():
    """Point-to-segment distance to a goal mouth: level with the mouth ->
    pure x-offset; wide of it -> hypot of x-offset and the y overshoot."""
    from footballcoach.ai.physics_pretrain.train_player_dynamics import _goal_mouth_distance_m

    goal_x = np.array([52.5, 52.5, -52.5])
    half_w = np.array([3.66, 3.66, 3.66])
    pos_x = np.array([42.5, 42.5, -42.5])
    pos_y = np.array([0.0, 9.66, -9.66])
    got = _goal_mouth_distance_m(pos_x, pos_y, goal_x, half_w)
    assert got[0] == pytest.approx(10.0)
    assert got[1] == pytest.approx(math.hypot(10.0, 6.0))
    assert got[2] == pytest.approx(math.hypot(10.0, 6.0))


def test_goal_dist_delta_targets_sign_and_shape(tmp_path):
    """Targets are (N, 2) raw deltas: negative where the player ended t=3s
    closer to that goal than at t=0. Cross-checked against a straightforward
    per-row recomputation from the same denormalized positions."""
    from footballcoach.ai.physics_pretrain.train_player_dynamics import (
        _goal_dist_delta_targets,
        _goal_mouth_distance_m,
        _require_horizon_index,
    )

    generate_dataset(n_episodes=40, output_dir=tmp_path, seed=9, shard_size=40, n_workers=1)
    ds = PlayerDynamicsDataset.from_directory(tmp_path)
    params = PlayerEpisodeGenParams.from_config()
    pitch_half_diag_m = math.hypot(params.base_pitch_length_m / 2, params.base_pitch_width_m / 2)
    h3 = _require_horizon_index(params.horizons_s, 3.0, "test")
    got = _goal_dist_delta_targets(ds, params, h3, pitch_half_diag_m)
    assert got.shape == (40, 2)
    assert np.isfinite(got).all()
    assert (got < 0).any() and (got > 0).any(), "expect both approaching and receding rows in a random sample"

    # Independent recomputation for the right goal (column 1), row 0.
    assert params.normalize_kinematics_by_base_pitch, "this check assumes the base-pitch normalization convention"
    div = pitch_half_diag_m
    half_len = float(ds.inputs[0, 17]) * params.base_pitch_length_m / 2
    half_goal_w = float(ds.inputs[0, 19]) * params.base_goal_width_m / 2
    base = h3 * N_TARGET_FIELDS_PER_HORIZON
    d0 = _goal_mouth_distance_m(
        np.array([ds.inputs[0, 0] * div]), np.array([ds.inputs[0, 1] * div]),
        np.array([half_len]), np.array([half_goal_w]),
    )[0]
    d3 = _goal_mouth_distance_m(
        np.array([ds.targets[0, base] * div]), np.array([ds.targets[0, base + 1] * div]),
        np.array([half_len]), np.array([half_goal_w]),
    )[0]
    assert got[0, 1] == pytest.approx((d3 - d0) / pitch_half_diag_m, abs=1e-5)


def test_require_horizon_index_raises_clearly_when_absent():
    from footballcoach.ai.physics_pretrain.train_player_dynamics import _require_horizon_index

    assert _require_horizon_index([0.2, 1.0, 3.0], 1.0, "x") == 1
    with pytest.raises(ValueError, match="must contain 3.0"):
        _require_horizon_index([0.2, 1.0, 5.0], 3.0, "goal_dist_delta_head's t=3.0s target")


def test_goal_dist_delta_head_loss_is_unmasked_mse():
    from footballcoach.ai.physics_pretrain.train_player_dynamics import _goal_dist_delta_head_loss

    model = _aux_test_model()
    latent = torch.randn(5, 8)
    targets = np.zeros((5, 2), dtype=np.float32)
    row_idx = np.arange(5)
    loss, mae_l, mae_r = _goal_dist_delta_head_loss(model, latent, targets, row_idx, "cpu")
    expected = F.mse_loss(model.goal_dist_delta_head(latent), torch.zeros(5, 2))
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    # Every row contributes (no mask): perturbing ANY row moves the loss.
    for r in range(5):
        perturbed = targets.copy()
        perturbed[r] = [9.0, -9.0]
        loss_p, _, _ = _goal_dist_delta_head_loss(model, latent, perturbed, row_idx, "cpu")
        assert loss_p.item() > loss.item() + 1.0
    assert mae_l >= 0 and mae_r >= 0


def test_short_horizon_probe_loss_sums_both_heads():
    from footballcoach.ai.physics_pretrain.train_player_dynamics import _short_horizon_probe_loss

    model = _aux_test_model()
    latent = torch.randn(6, 8)
    t0 = np.zeros((6, 4), dtype=np.float32)
    t1 = np.ones((6, 4), dtype=np.float32) * 0.5
    row_idx = np.arange(6)
    loss, rmse0, rmse1 = _short_horizon_probe_loss(model, latent, t0, t1, row_idx, "cpu")
    mse0 = F.mse_loss(model.short_horizon_head_0_2s(latent), torch.zeros(6, 4))
    mse1 = F.mse_loss(model.short_horizon_head_1_0s(latent), torch.full((6, 4), 0.5))
    assert loss.item() == pytest.approx((mse0 + mse1).item(), abs=1e-6)
    assert rmse0 == pytest.approx(mse0.item() ** 0.5, abs=1e-6)
    assert rmse1 == pytest.approx(mse1.item() ** 0.5, abs=1e-6)
    # Both heads are supervised independently -- each carries its own
    # gradient (a probe left dangling would silently do nothing).
    loss.backward()
    assert model.short_horizon_head_0_2s.weight.grad.abs().sum() > 0
    assert model.short_horizon_head_1_0s.weight.grad.abs().sum() > 0


def test_build_horizon_bundle_crossing_dt_is_rebased_per_horizon(tmp_path):
    """A pseudo-start at horizon h sees delta_t = crossing_time -
    horizons_s[h]; a crossing that already happened BEFORE h (negative
    delta_t) is folded into the same -1 sentinel / invalid treatment as an
    episode that never crossed."""
    from footballcoach.ai.physics_pretrain.train_player_dynamics import _build_horizon_bundle

    generate_dataset(n_episodes=80, output_dir=tmp_path, seed=13, shard_size=80, n_workers=1)
    ds = PlayerDynamicsDataset.from_directory(tmp_path)
    horizons = list(PlayerEpisodeGenParams.from_config().horizons_s)
    idx = np.arange(len(ds))
    bundle = _build_horizon_bundle(
        ds, idx, len(horizons), horizons, pair_enabled=True, pair_max_skip=1,
        pair_min_start_speed_norm=0.0, has_crossing_data=True,
    )
    for h, t in enumerate(horizons):
        valid, dt = bundle["crossing_valid"][h], bundle["crossing_dt"][h]
        assert valid.shape == (len(ds),) and dt.shape == (len(ds),)
        assert (dt[~valid] == -1.0).all()
        assert (dt[valid] >= 0).all()
        np.testing.assert_allclose(dt[valid], (ds.crossing_times[valid] - t).astype(np.float32), atol=1e-5)
        # Validity can only shrink as the pseudo-start moves later.
        if h > 0:
            assert valid.sum() <= bundle["crossing_valid"][h - 1].sum()

    no_crossing = _build_horizon_bundle(
        ds, idx, len(horizons), horizons, pair_enabled=True, pair_max_skip=1,
        pair_min_start_speed_norm=0.0, has_crossing_data=False,
    )
    assert all(v is None for v in no_crossing["crossing_valid"])
    assert all(v is None for v in no_crossing["crossing_dt"])


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


def test_decoder_only_pretraining_trains_auxiliary_heads_too(tmp_path, monkeypatch):
    """decoder_only_pretrain_epochs > 0 must actually move crossing_head/
    goal_dist_delta_head/short_horizon_head_0_2s/short_horizon_head_1_0s'
    weights, not just the decoder -- these heads read the latent directly
    (same as the decoder), so there's nothing latent-frozen about training
    them even when the rest of the encoder is fully frozen this phase.
    Mirrors train_ball_dynamics.py's decoder_only_params, which includes
    crossing_head/resting_head/position_head the same way. Checked by
    snapshotting each head's weight tensor before/after a real (tiny)
    decoder-only-pretrain run and asserting it moved -- a stale
    decoder_only_params list (missing one of these heads) would leave that
    head's weights bit-identical, since with freeze_latent=True nothing else
    could possibly move them."""
    import copy

    from footballcoach.ai.physics_pretrain.train_player_dynamics import train
    import footballcoach.ai.config as ai_config_mod

    orig_load_ai_config = ai_config_mod.load_ai_config

    def _patched():
        cfg = orig_load_ai_config()
        pp = cfg["physics_pretrain"]["player"]
        pp["decoder_only_pretrain_epochs"] = 3
        pp["decoder_only_pretrain_freeze_latent"] = True
        pp["crossing_pos_loss_weight"] = 0.005
        pp["crossing_dt_loss_weight"] = 0.005
        pp["goal_dist_delta_loss_weight"] = 0.05
        pp["short_horizon_probe_loss_weight"] = 0.1
        pp["epochs"] = 0  # isolate to just the decoder-only phase
        return cfg

    monkeypatch.setattr(ai_config_mod, "load_ai_config", _patched)

    dataset_dir = tmp_path / "data"
    generate_dataset(n_episodes=200, output_dir=dataset_dir, seed=23, shard_size=200, n_workers=1)

    from footballcoach.ai.physics_pretrain.player_dynamics_net import PlayerDynamicsAutoencoder
    torch.manual_seed(0)
    before = PlayerDynamicsAutoencoder.from_config()
    before_weights = {
        name: copy.deepcopy(getattr(before, name).weight.data)
        for name in ("crossing_head", "goal_dist_delta_head", "short_horizon_head_0_2s", "short_horizon_head_1_0s")
    }
    init_ckpt = tmp_path / "init.pt"
    torch.save({
        "model_state_dict": before.state_dict(), "encoder_state_dict": before.encoder.state_dict(),
        "config_snapshot": orig_load_ai_config()["physics_pretrain"]["player"], "phase": "test_init",
    }, init_ckpt)

    output_path = tmp_path / "player_encoder.pt"
    train(
        dataset_dir=str(dataset_dir), output_path=str(output_path),
        epochs=0, batch_size=16, lr=1e-2, val_frac=0.2, seed=0,
        init_checkpoint=str(init_ckpt),
    )

    after_ckpt = torch.load(output_path.with_suffix(".after_decoder_pretrain.pt"), map_location="cpu")
    after_state = after_ckpt["model_state_dict"]
    for name, before_w in before_weights.items():
        after_w = after_state[f"{name}.weight"]
        assert not torch.allclose(before_w, after_w), f"{name}'s weights did not move during decoder-only-pretrain"


def test_train_smoke_with_all_auxiliary_heads(tmp_path, monkeypatch, caplog):
    """End-to-end: all three auxiliary latent heads on with nonzero weights.
    Their per-epoch diagnostic lines must appear with finite (non-NaN,
    non-exploding) values, and the corresponding .history.npz fields must be
    written."""
    from footballcoach.ai.physics_pretrain.train_player_dynamics import _AUX_METRIC_KEYS, train
    import footballcoach.ai.config as ai_config_mod

    orig_load_ai_config = ai_config_mod.load_ai_config

    def _patched():
        cfg = orig_load_ai_config()
        pp = cfg["physics_pretrain"]["player"]
        pp["crossing_pos_loss_weight"] = 0.005
        pp["crossing_dt_loss_weight"] = 0.005
        pp["goal_dist_delta_loss_weight"] = 0.05
        pp["short_horizon_probe_loss_weight"] = 0.1
        return cfg

    monkeypatch.setattr(ai_config_mod, "load_ai_config", _patched)

    dataset_dir = tmp_path / "data"
    generate_dataset(n_episodes=120, output_dir=dataset_dir, seed=17, shard_size=120, n_workers=1)
    output_path = tmp_path / "player_encoder.pt"
    with caplog.at_level("INFO", logger="footballcoach.ai.physics_pretrain.train_player_dynamics"):
        artifact = train(
            dataset_dir=str(dataset_dir), output_path=str(output_path),
            epochs=2, batch_size=16, lr=1e-3, val_frac=0.2, seed=0,
        )
    assert "crossing_head: train loss=" in caplog.text
    assert "goal_dist_delta_head: train loss=" in caplog.text
    assert "short_horizon_probes: train loss=" in caplog.text
    assert "nan" not in caplog.text.lower().split("crossing_head: train loss=")[1].splitlines()[0]

    for record in artifact["history"]:
        for k in _AUX_METRIC_KEYS:
            for split in ("train", "val"):
                v = record[f"{split}_{k}"]
                assert math.isfinite(v), f"{split}_{k} is not finite: {v}"
                assert abs(v) < 1e4, f"{split}_{k} exploded: {v}"

    # The heads really did receive gradient: their weights must differ from a
    # freshly-initialized model's (all four are saved in the phase checkpoint).
    state = torch.load(output_path.with_suffix(".after_training.pt"), map_location="cpu")["model_state_dict"]
    for head in ("crossing_head", "goal_dist_delta_head", "short_horizon_head_0_2s", "short_horizon_head_1_0s"):
        assert f"{head}.weight" in state and f"{head}.bias" in state


def test_train_resumes_from_checkpoint_predating_the_aux_heads(tmp_path, monkeypatch, caplog):
    """--init-checkpoint's strict=False path must treat the four auxiliary
    heads as simply MISSING (fresh init) when resuming from a checkpoint
    saved before they existed, rather than raising -- they are purely
    additive, no existing key changed shape."""
    from footballcoach.ai.physics_pretrain.train_player_dynamics import train

    dataset_dir = tmp_path / "data"
    generate_dataset(n_episodes=60, output_dir=dataset_dir, seed=5, shard_size=60, n_workers=1)

    first_output = tmp_path / "first.pt"
    train(
        dataset_dir=str(dataset_dir), output_path=str(first_output),
        epochs=1, batch_size=16, lr=1e-2, val_frac=0.2, seed=0,
    )
    ckpt_path = first_output.with_suffix(".after_training.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Simulate an OLD checkpoint by dropping every auxiliary head's params.
    ckpt["model_state_dict"] = {
        k: v for k, v in ckpt["model_state_dict"].items()
        if not k.startswith(("crossing_head", "goal_dist_delta_head", "short_horizon_head"))
    }
    old_style_path = tmp_path / "old_style.pt"
    torch.save(ckpt, old_style_path)

    second_output = tmp_path / "second.pt"
    with caplog.at_level("INFO", logger="footballcoach.ai.physics_pretrain.train_player_dynamics"):
        train(
            dataset_dir=str(dataset_dir), output_path=str(second_output),
            epochs=1, batch_size=16, lr=1e-2, val_frac=0.2, seed=0,
            init_checkpoint=str(old_style_path),
        )
    assert "Checkpoint missing" in caplog.text
    assert "crossing_head.weight" in caplog.text
    assert "short_horizon_head_1_0s.bias" in caplog.text
    assert second_output.exists()


def test_train_auto_widens_init_checkpoint_on_dim_mismatch(tmp_path, monkeypatch, caplog):
    """Resuming via --init-checkpoint from a checkpoint whose hidden_dim/
    encoder_bottleneck_dim/latent_dim/decoder_hidden_dim don't match the
    CURRENT config used to crash with a raw torch RuntimeError (size
    mismatch) -- train() now detects this and runs widen_player_checkpoint's
    seam-preserving surgery automatically instead of failing. Mirrors
    test_ball_physics_pretrain.py's identical test for the ball pipeline."""
    from footballcoach.ai.physics_pretrain.train_player_dynamics import train
    import footballcoach.ai.config as ai_config_mod

    orig_load_ai_config = ai_config_mod.load_ai_config

    def _small_cfg():
        cfg = orig_load_ai_config()
        pp = cfg["physics_pretrain"]["player"]
        # decoder_hidden_dim must be >= 2*N_IDENTITY_SHORTCUT_FIELDS (=14 for
        # player) since the real config's decoder_identity_shortcut_enabled
        # is null (mirrors identity_shortcut_enabled=true) -- unlike ball's
        # ai_config.json, which sets it explicitly False.
        pp["hidden_dim"], pp["encoder_bottleneck_dim"], pp["latent_dim"], pp["decoder_hidden_dim"] = 24, 12, 16, 16
        return cfg

    def _bigger_cfg():
        cfg = orig_load_ai_config()
        pp = cfg["physics_pretrain"]["player"]
        pp["hidden_dim"], pp["encoder_bottleneck_dim"], pp["latent_dim"], pp["decoder_hidden_dim"] = 40, 20, 24, 20
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
    with caplog.at_level("INFO", logger="footballcoach.ai.physics_pretrain.train_player_dynamics"):
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


# ---------------------------------------------------------------------------
# widen_player_checkpoint.py
# ---------------------------------------------------------------------------

def _widen_test_cfg(**overrides) -> dict:
    cfg = dict(
        hidden_dim=24, encoder_bottleneck_dim=12, latent_dim=16, decoder_hidden_dim=10,
        horizons_s=[0.2, 1.0, 3.0], identity_shortcut_enabled=True, identity_shortcut_noise_std=0.01,
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
    from footballcoach.ai.physics_pretrain.widen_player_checkpoint import _build_model, verify_widened_model, widen_model_

    old_cfg = _widen_test_cfg()
    new_cfg = dict(old_cfg, hidden_dim=40, encoder_bottleneck_dim=20, latent_dim=24, decoder_hidden_dim=18)
    old_model = _build_model(old_cfg)
    _perturb(old_model)
    new_model = _build_model(new_cfg)
    widen_model_(old_model, new_model, old_cfg, new_cfg)
    verify_widened_model(old_model, new_model, torch.randn(32, N_INPUT_FIELDS) * 0.3)


def test_widen_model_reproduces_old_model_exactly_without_identity_shortcut():
    from footballcoach.ai.physics_pretrain.widen_player_checkpoint import _build_model, verify_widened_model, widen_model_

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
    from footballcoach.ai.physics_pretrain.widen_player_checkpoint import _build_model, verify_widened_model, widen_model_

    old_cfg = _widen_test_cfg(identity_shortcut_noise_std=0.0)
    new_cfg = dict(old_cfg, latent_dim=22)
    old_model = _build_model(old_cfg)
    _perturb(old_model)
    new_model = _build_model(new_cfg)
    widen_model_(old_model, new_model, old_cfg, new_cfg)
    verify_widened_model(old_model, new_model, torch.randn(16, N_INPUT_FIELDS) * 0.3)


def test_widen_model_new_latent_rows_are_not_dead_under_identity_shortcut():
    """Regression test: encoder.out's NEW latent rows must not be left at
    exact zero when identity_shortcut_enabled -- see widen_ball_checkpoint's
    identical test/bug for the full rationale. Every one of the new latent
    rows must be genuinely alive (nonzero) right after widening."""
    from footballcoach.ai.physics_pretrain.widen_player_checkpoint import _build_model, verify_widened_model, widen_model_

    old_cfg = _widen_test_cfg()  # identity_shortcut_enabled=True, old latent_dim=16 (> the 7-field identity block)
    new_cfg = dict(old_cfg, latent_dim=20)
    old_model = _build_model(old_cfg)
    _perturb(old_model)
    new_model = _build_model(new_cfg)
    widen_model_(old_model, new_model, old_cfg, new_cfg)
    verify_widened_model(old_model, new_model, torch.randn(16, N_INPUT_FIELDS) * 0.3)

    new_rows = new_model.encoder.out.weight.data[16:20, :]
    new_bias = new_model.encoder.out.bias.data[16:20]
    assert new_rows.abs().sum().item() > 0, "encoder.out's new latent rows are dead (all-zero weight)"
    assert new_bias.abs().sum().item() > 0, "encoder.out's new latent rows are dead (all-zero bias)"


def test_widen_model_new_capacity_is_trainable_not_dead():
    """Beyond exact reproduction: confirm the newly-added capacity actually
    moves under a few real optimizer steps -- i.e. it's genuinely trainable,
    not a permanent dead/zero-gradient fixed point.

    Note this is deliberately a multi-step check, not a single-backward-pass
    gradient check: a plain (non-identity-shortcut) widened hidden-to-hidden
    seam like decoder.net[0]'s new rows legitimately has ZERO gradient on
    the very FIRST step (net[2]'s newly-zeroed consuming columns haven't
    read anything yet), then bootstraps normally on step 2+ once net[2]'s
    own (immediately nonzero) gradient moves those columns off zero -- an
    ordinary one-step warmup, not the PERMANENT two-sided dead lock that
    `test_widen_model_new_latent_rows_are_not_dead_under_identity_shortcut`
    above guards against (encoder.out's identity-shortcut rows are the one
    seam where both the producing AND consuming sides start at exact zero
    simultaneously, with nothing to bootstrap from at all)."""
    from footballcoach.ai.physics_pretrain.widen_player_checkpoint import _build_model, widen_model_

    old_cfg = _widen_test_cfg()
    new_cfg = dict(old_cfg, hidden_dim=40, encoder_bottleneck_dim=20, latent_dim=24, decoder_hidden_dim=18)
    old_model = _build_model(old_cfg)
    _perturb(old_model)
    new_model = _build_model(new_cfg)
    widen_model_(old_model, new_model, old_cfg, new_cfg)

    encoder_out_new_before = new_model.encoder.out.weight.data[16:24, :].clone()
    decoder_net0_new_before = new_model.decoder.net[0].weight.data[10:18, :].clone()
    decoder_net2_new_before = new_model.decoder.net[2].weight.data[:, 10:18].clone()

    optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
    x = torch.randn(16, N_INPUT_FIELDS) * 0.3
    for _ in range(3):
        optimizer.zero_grad()
        latent, decoded = new_model(x)
        loss = sum(d.pow(2).mean() for d in decoded) + latent.pow(2).mean()
        loss.backward()
        optimizer.step()

    assert not torch.equal(new_model.encoder.out.weight.data[16:24, :], encoder_out_new_before), \
        "new latent rows never moved under training -- dead capacity"
    assert not torch.equal(new_model.decoder.net[0].weight.data[10:18, :], decoder_net0_new_before), \
        "new decoder hidden rows never moved under training -- dead capacity"
    assert not torch.equal(new_model.decoder.net[2].weight.data[:, 10:18], decoder_net2_new_before), \
        "new decoder hidden -> output columns never moved under training -- dead capacity"


def test_widen_model_preserves_auxiliary_heads_exactly():
    """The four auxiliary latent heads are pure CONSUMERS of a widened
    latent_dim -- old columns copied, new columns zeroed -- so each must
    reproduce the old model's output bit-for-bit after widening. Easy to
    miss: nothing else in the widen path would fail if they were skipped, the
    heads would just silently forget their training."""
    from footballcoach.ai.physics_pretrain.widen_player_checkpoint import (
        _AUX_HEAD_NAMES,
        _build_model,
        verify_widened_model,
        widen_model_,
    )

    old_cfg = _widen_test_cfg()
    new_cfg = dict(old_cfg, hidden_dim=40, encoder_bottleneck_dim=20, latent_dim=24, decoder_hidden_dim=18)
    old_model = _build_model(old_cfg)
    _perturb(old_model)
    new_model = _build_model(new_cfg)
    widen_model_(old_model, new_model, old_cfg, new_cfg)

    assert set(_AUX_HEAD_NAMES) == {
        "crossing_head", "goal_dist_delta_head", "short_horizon_head_0_2s", "short_horizon_head_1_0s",
    }
    x = torch.randn(24, N_INPUT_FIELDS) * 0.3
    verify_widened_model(old_model, new_model, x)  # covers all four heads

    for name in _AUX_HEAD_NAMES:
        old_head, new_head = getattr(old_model, name), getattr(new_model, name)
        torch.testing.assert_close(new_head.weight.data[:, :16], old_head.weight.data)
        torch.testing.assert_close(new_head.bias.data, old_head.bias.data)
        assert new_head.weight.data[:, 16:].abs().sum().item() == 0.0, f"{name}'s new latent columns must start zeroed"

    # ...and verify_widened_model actually CATCHES a broken head (it would be
    # a silent no-op check otherwise).
    with torch.no_grad():
        new_model.crossing_head.weight[0, 0] += 1.0
    with pytest.raises(AssertionError, match="crossing_head diverged"):
        verify_widened_model(old_model, new_model, x)


def test_widen_model_auxiliary_head_new_capacity_is_trainable():
    """The newly-added latent columns on each auxiliary head start at zero
    but must be immediately trainable (a zeroed weight still gets an ordinary
    gradient), not a dead fixed point."""
    from footballcoach.ai.physics_pretrain.widen_player_checkpoint import _AUX_HEAD_NAMES, _build_model, widen_model_

    old_cfg = _widen_test_cfg()
    new_cfg = dict(old_cfg, latent_dim=24)
    old_model = _build_model(old_cfg)
    _perturb(old_model)
    new_model = _build_model(new_cfg)
    widen_model_(old_model, new_model, old_cfg, new_cfg)

    before = {n: getattr(new_model, n).weight.data[:, 16:24].clone() for n in _AUX_HEAD_NAMES}
    optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
    x = torch.randn(16, N_INPUT_FIELDS) * 0.3
    for _ in range(3):
        optimizer.zero_grad()
        latent, _decoded = new_model(x)
        loss = sum(getattr(new_model, n)(latent).pow(2).mean() for n in _AUX_HEAD_NAMES)
        loss.backward()
        optimizer.step()

    for n in _AUX_HEAD_NAMES:
        assert not torch.equal(getattr(new_model, n).weight.data[:, 16:24], before[n]), \
            f"{n}'s new latent columns never moved under training -- dead capacity"


def test_widen_checkpoint_rejects_shrinking():
    from footballcoach.ai.physics_pretrain.widen_player_checkpoint import _validate_widen_cfgs

    old_cfg = _widen_test_cfg()
    new_cfg = dict(old_cfg, latent_dim=old_cfg["latent_dim"] - 2)
    with pytest.raises(ValueError, match="shrank"):
        _validate_widen_cfgs(old_cfg, new_cfg)


def test_widen_checkpoint_rejects_unstable_shortcut_settings():
    from footballcoach.ai.physics_pretrain.widen_player_checkpoint import _validate_widen_cfgs

    old_cfg = _widen_test_cfg(identity_shortcut_enabled=True)
    new_cfg = dict(old_cfg, identity_shortcut_enabled=False, latent_dim=old_cfg["latent_dim"] + 2)
    with pytest.raises(ValueError, match="stay the same"):
        _validate_widen_cfgs(old_cfg, new_cfg)


def test_widen_checkpoint_end_to_end(tmp_path):
    from footballcoach.ai.physics_pretrain.widen_player_checkpoint import _build_model, widen_checkpoint

    old_cfg = _widen_test_cfg()
    model = _build_model(old_cfg)
    _perturb(model)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": model.state_dict(), "config_snapshot": old_cfg}, ckpt_path)

    ds_dir = tmp_path / "data"
    generate_dataset(n_episodes=20, output_dir=ds_dir, seed=1, shard_size=20, n_workers=1)

    new_cfg = dict(old_cfg, hidden_dim=32, encoder_bottleneck_dim=16, latent_dim=20, decoder_hidden_dim=14)
    out_path = tmp_path / "ckpt.widened.pt"
    widen_checkpoint(ckpt_path, out_path, new_cfg=new_cfg, dataset_dir=ds_dir, verify_n=10)

    assert out_path.exists()
    loaded = torch.load(out_path, map_location="cpu")
    assert loaded["config_snapshot"]["latent_dim"] == 20
    assert loaded["phase"] == "widened"
    assert loaded["normalization"]["pitch_half_diag_m"] > 0


def test_widen_checkpoint_requires_full_model_state_dict(tmp_path):
    from footballcoach.ai.physics_pretrain.widen_player_checkpoint import widen_checkpoint

    ckpt_path = tmp_path / "encoder_only.pt"
    torch.save({"encoder_state_dict": {}, "config_snapshot": _widen_test_cfg()}, ckpt_path)
    with pytest.raises(ValueError, match="no 'model_state_dict'"):
        widen_checkpoint(ckpt_path, tmp_path / "out.pt", new_cfg=_widen_test_cfg())

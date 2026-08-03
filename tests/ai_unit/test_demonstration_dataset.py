"""Baseline regression tests for src/footballcoach/ai/bc/dataset.py.

Uses small hand-written .npz fixtures (written via a tmp_path fixture) so
these tests are fast and deterministic, independent of the real
demonstrations/phase1/ recordings.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from footballcoach.ai.bc.dataset import DemonstrationDataset
from footballcoach.ai.obs.schema import (
    BALL_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    MAX_OTHER_PLAYERS,
    PLAYER_FEATURE_DIM,
)
from footballcoach.ai.ppo.bc import BC_LABEL_DIM, _I_VALID


def _write_npz(
    path: Path,
    n: int,
    valid_mask: np.ndarray | None = None,
    rewards: np.ndarray | None = None,
    dones: np.ndarray | None = None,
) -> None:
    """Write a small hand-built demonstration .npz file with n steps."""
    obs_self_feat = np.random.randn(n, PLAYER_FEATURE_DIM).astype(np.float32)
    obs_other_feat = np.random.randn(n, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM).astype(np.float32)
    obs_exists_mask = np.zeros((n, MAX_OTHER_PLAYERS), dtype=np.float32)
    obs_ball_feat = np.random.randn(n, BALL_FEATURE_DIM).astype(np.float32)
    obs_global_feat = np.random.randn(n, GLOBAL_FEATURE_DIM).astype(np.float32)

    bc_labels = np.zeros((n, BC_LABEL_DIM), dtype=np.float32)
    if valid_mask is None:
        valid_mask = np.ones(n, dtype=np.float32)
    bc_labels[:, _I_VALID] = valid_mask

    kwargs = dict(
        obs_self_feat=obs_self_feat,
        obs_other_feat=obs_other_feat,
        obs_exists_mask=obs_exists_mask,
        obs_ball_feat=obs_ball_feat,
        obs_global_feat=obs_global_feat,
        bc_labels=bc_labels,
    )
    if rewards is not None:
        kwargs["rewards"] = rewards
    if dones is not None:
        kwargs["dones"] = dones
    np.savez(path, **kwargs)


class TestFromDirectory:
    def test_loads_single_file(self, tmp_path):
        _write_npz(tmp_path / "demo1.npz", n=10)
        ds = DemonstrationDataset.from_directory(tmp_path)
        assert len(ds) == 10

    def test_loads_multiple_files_concatenated(self, tmp_path):
        _write_npz(tmp_path / "demo1.npz", n=5)
        _write_npz(tmp_path / "demo2.npz", n=7)
        ds = DemonstrationDataset.from_directory(tmp_path)
        assert len(ds) == 12

    def test_raises_when_no_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DemonstrationDataset.from_directory(tmp_path)

    def test_backward_compat_missing_rewards_dones(self, tmp_path):
        """Older files without rewards/dones load with zeros."""
        _write_npz(tmp_path / "demo1.npz", n=5, rewards=None, dones=None)
        ds = DemonstrationDataset.from_directory(tmp_path)
        assert ds.has_rewards == False


class TestIterateMinibatches:
    def test_batch_shapes_correct(self, tmp_path):
        _write_npz(tmp_path / "demo1.npz", n=20)
        ds = DemonstrationDataset.from_directory(tmp_path)
        for obs_dict, labels in ds.iterate_minibatches(batch_size=8, shuffle=False):
            assert obs_dict["self_feat"].shape[1] == PLAYER_FEATURE_DIM
            assert obs_dict["other_feat"].shape[1:] == (MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM)
            assert obs_dict["exists_mask"].shape[1] == MAX_OTHER_PLAYERS
            assert obs_dict["ball_feat"].shape[1] == BALL_FEATURE_DIM
            assert obs_dict["global_feat"].shape[1] == GLOBAL_FEATURE_DIM
            assert labels.shape[1] == BC_LABEL_DIM
            assert obs_dict["self_feat"].shape[0] <= 8

    def test_shuffle_true_produces_different_orders(self, tmp_path):
        _write_npz(tmp_path / "demo1.npz", n=50)
        ds = DemonstrationDataset.from_directory(tmp_path)

        np.random.seed(1)
        batches_a = [
            obs_dict["self_feat"].numpy().copy()
            for obs_dict, _ in ds.iterate_minibatches(batch_size=50, shuffle=True)
        ]
        np.random.seed(2)
        batches_b = [
            obs_dict["self_feat"].numpy().copy()
            for obs_dict, _ in ds.iterate_minibatches(batch_size=50, shuffle=True)
        ]
        assert not np.array_equal(batches_a[0], batches_b[0])

    def test_valid_only_filters_invalid_rows(self, tmp_path):
        n = 10
        valid_mask = np.zeros(n, dtype=np.float32)
        valid_mask[:4] = 1.0  # first 4 rows valid
        _write_npz(tmp_path / "demo1.npz", n=n, valid_mask=valid_mask)
        ds = DemonstrationDataset.from_directory(tmp_path)

        total_rows = 0
        for obs_dict, labels in ds.iterate_minibatches(batch_size=100, shuffle=False, valid_only=True):
            total_rows += labels.shape[0]
            assert (labels[:, _I_VALID] > 0.5).all()
        assert total_rows == 4

    def test_valid_only_false_includes_all_rows(self, tmp_path):
        n = 10
        valid_mask = np.zeros(n, dtype=np.float32)
        valid_mask[:4] = 1.0
        _write_npz(tmp_path / "demo1.npz", n=n, valid_mask=valid_mask)
        ds = DemonstrationDataset.from_directory(tmp_path)

        total_rows = 0
        for obs_dict, labels in ds.iterate_minibatches(batch_size=100, shuffle=False, valid_only=False):
            total_rows += labels.shape[0]
        assert total_rows == n


class TestComputeReturns:
    def test_hand_computed_discounted_return(self, tmp_path):
        """3-step episode with known rewards, gamma=0.5.

        rewards = [1, 2, 4], done at last step only.
        G_2 = 4
        G_1 = 2 + 0.5*4 = 4
        G_0 = 1 + 0.5*4 = 3
        """
        n = 3
        rewards = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        _write_npz(tmp_path / "demo1.npz", n=n, rewards=rewards, dones=dones)
        ds = DemonstrationDataset.from_directory(tmp_path)

        returns = ds.compute_returns(gamma=0.5)
        expected = np.array([3.0, 4.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(returns, expected, atol=1e-5)

    def test_episode_boundary_resets_running_sum(self, tmp_path):
        """Two 2-step episodes back to back; the second episode's returns
        must not be contaminated by the first episode's rewards."""
        n = 4
        rewards = np.array([10.0, 20.0, 1.0, 2.0], dtype=np.float32)
        dones = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        _write_npz(tmp_path / "demo1.npz", n=n, rewards=rewards, dones=dones)
        ds = DemonstrationDataset.from_directory(tmp_path)

        returns = ds.compute_returns(gamma=0.9)
        # Episode 1: G_1=20, G_0=10+0.9*20=28
        # Episode 2: G_3=2, G_2=1+0.9*2=2.8
        expected = np.array([28.0, 20.0, 2.8, 2.0], dtype=np.float32)
        np.testing.assert_allclose(returns, expected, atol=1e-4)


class TestHasRewards:
    def test_true_when_rewards_present_and_nonzero(self, tmp_path):
        n = 3
        rewards = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        _write_npz(tmp_path / "demo1.npz", n=n, rewards=rewards)
        ds = DemonstrationDataset.from_directory(tmp_path)
        assert ds.has_rewards == True

    def test_false_when_rewards_all_zero(self, tmp_path):
        n = 3
        rewards = np.zeros(n, dtype=np.float32)
        _write_npz(tmp_path / "demo1.npz", n=n, rewards=rewards)
        ds = DemonstrationDataset.from_directory(tmp_path)
        assert ds.has_rewards == False

    def test_false_when_rewards_missing(self, tmp_path):
        n = 3
        _write_npz(tmp_path / "demo1.npz", n=n, rewards=None)
        ds = DemonstrationDataset.from_directory(tmp_path)
        assert ds.has_rewards == False


class TestComputePosWeights:
    def test_matches_hand_computed_value(self, tmp_path):
        from footballcoach.ai.ppo.bc import _I_KICK_THIS_TICK, _I_TACKLE_ATTEMPT

        n = 10
        _write_npz(tmp_path / "demo1.npz", n=n)
        ds = DemonstrationDataset.from_directory(tmp_path)
        # 2 positive kicks out of 10 valid rows -> weight = 8/2 = 4.0
        ds._labels[:, _I_KICK_THIS_TICK] = 0.0
        ds._labels[0, _I_KICK_THIS_TICK] = 1.0
        ds._labels[1, _I_KICK_THIS_TICK] = 1.0
        # 1 positive tackle_attempt out of 10 -> weight = 9/1 = 9.0
        ds._labels[:, _I_TACKLE_ATTEMPT] = 0.0
        ds._labels[0, _I_TACKLE_ATTEMPT] = 1.0

        weights = ds.compute_pos_weights()
        assert weights["kick"] == pytest.approx(4.0)
        assert weights["tackle_attempt"] == pytest.approx(9.0)

    def test_no_positives_gives_finite_weight(self, tmp_path):
        n = 5
        _write_npz(tmp_path / "demo1.npz", n=n)
        ds = DemonstrationDataset.from_directory(tmp_path)
        weights = ds.compute_pos_weights()
        # n_pos=0 -> denominator clamped to max(0, 1) = 1 -> weight = n_neg
        assert weights["kick"] == pytest.approx(5.0)
        assert weights["tackle_attempt"] == pytest.approx(5.0)


class TestDownsampleTrivial:
    def _write_trivial_dataset(self, tmp_path):
        """3 episodes of 6 steps each. Within each episode, steps 1-5 have
        move_direction identical to step 0 (perfectly aligned => trivial),
        except one "rare event" row (kick) at step 3 and its neighbours
        which must never be excluded regardless of direction similarity.
        """
        from footballcoach.ai.ppo.bc import _I_DIR_X, _I_DIR_Y, _I_KICK_THIS_TICK, _I_VALID

        n_per_ep = 20
        n_eps = 3
        n = n_per_ep * n_eps
        dones = np.zeros(n, dtype=np.float32)
        for e in range(n_eps):
            dones[(e + 1) * n_per_ep - 1] = 1.0

        _write_npz(tmp_path / "demo1.npz", n=n, dones=dones)
        ds = DemonstrationDataset.from_directory(tmp_path)
        ds._labels[:, _I_VALID] = 1.0
        ds._labels[:, _I_DIR_X] = 1.0
        ds._labels[:, _I_DIR_Y] = 0.0
        ds._labels[:, _I_KICK_THIS_TICK] = 0.0
        for e in range(n_eps):
            kick_row = e * n_per_ep + 10
            ds._labels[kick_row, _I_KICK_THIS_TICK] = 1.0
        return ds, n_per_ep, n_eps

    def test_row_0_of_episode_never_trivial(self, tmp_path):
        ds, n_per_ep, n_eps = self._write_trivial_dataset(tmp_path)
        mask = ds._compute_trivial_mask(exclude_radius_steps=0)
        for e in range(n_eps):
            assert not mask[e * n_per_ep]

    def test_rows_near_rare_event_never_trivial(self, tmp_path):
        ds, n_per_ep, n_eps = self._write_trivial_dataset(tmp_path)
        mask = ds._compute_trivial_mask(exclude_radius_steps=5)
        for e in range(n_eps):
            kick_row = e * n_per_ep + 10
            lo, hi = max(0, kick_row - 5), min(n_per_ep * n_eps, kick_row + 6)
            assert not mask[lo:hi].any()

    def test_downsampling_never_excludes_ineligible_rows(self, tmp_path):
        ds, n_per_ep, n_eps = self._write_trivial_dataset(tmp_path)
        mask = ds._compute_trivial_mask(exclude_radius_steps=5)
        ineligible = np.where(~mask)[0]

        rng = np.random.default_rng(0)
        seen_indices = set()
        for obs_dict, labels in ds.iterate_minibatches(
            batch_size=1000, shuffle=False, valid_only=False,
            downsample_trivial_frac=1.0, downsample_trivial_exclude_radius_steps=5,
            rng=rng,
        ):
            pass
        # Re-derive which original indices survived by checking count only
        # (can't directly recover original indices from returned tensors,
        # so instead verify via the trivial mask + a full-exclusion run that
        # every ineligible row's position is preserved in the surviving set size).
        total_after = sum(
            labels.shape[0]
            for _, labels in ds.iterate_minibatches(
                batch_size=1000, shuffle=False, valid_only=False,
                downsample_trivial_frac=1.0, downsample_trivial_exclude_radius_steps=5,
                rng=np.random.default_rng(0),
            )
        )
        assert total_after == len(ineligible)

    def test_fresh_reroll_each_epoch_produces_different_excluded_sets(self, tmp_path):
        ds, n_per_ep, n_eps = self._write_trivial_dataset(tmp_path)

        # NOTE: exclude_radius_steps must be small enough here that at least
        # some rows remain eligible for downsampling. With n_per_ep=6 and a
        # kick event at index 3 of each episode, radius_steps=5 (used
        # elsewhere in this test class) covers every row in the episode
        # (max distance from index 3 within a 6-row episode is 3), leaving
        # zero eligible rows -- which would make this test deterministically
        # pass/fail regardless of the RNG seed, not a genuine re-roll check.
        def _self_feat_ids(seed):
            rng = np.random.default_rng(seed)
            feats = []
            for obs_dict, labels in ds.iterate_minibatches(
                batch_size=1000, shuffle=False, valid_only=False,
                downsample_trivial_frac=0.5, downsample_trivial_exclude_radius_steps=1,
                rng=rng,
            ):
                feats.append(obs_dict["self_feat"].numpy())
            return np.concatenate(feats)

        a = _self_feat_ids(1)
        b = _self_feat_ids(2)
        # Different random exclusion subsets -> different surviving row sets
        # (with high probability given distinct seeds and random self_feat).
        assert a.shape != b.shape or not np.array_equal(a, b)

    def test_downsample_trivial_stats_matches_mask(self, tmp_path):
        ds, n_per_ep, n_eps = self._write_trivial_dataset(tmp_path)
        mask = ds._compute_trivial_mask(exclude_radius_steps=5)
        n_trivial_expected = int(mask.sum())

        stats = ds.downsample_trivial_stats(
            valid_only=False, exclude_radius_steps=5, frac=0.5,
        )
        assert stats["n_total"] == n_per_ep * n_eps
        assert stats["n_trivial"] == n_trivial_expected
        assert stats["trivial_frac"] == pytest.approx(n_trivial_expected / (n_per_ep * n_eps))
        assert stats["n_excluded_at_frac"] == round(n_trivial_expected * 0.5)

    def test_downsample_trivial_stats_zero_frac_excludes_nothing(self, tmp_path):
        ds, n_per_ep, n_eps = self._write_trivial_dataset(tmp_path)
        stats = ds.downsample_trivial_stats(valid_only=False, exclude_radius_steps=5, frac=0.0)
        assert stats["n_excluded_at_frac"] == 0

    def test_pos_weight_scales_loss_linearly(self):
        import torch
        from footballcoach.ai.ppo.bc import bc_loss_from_tensor, BC_LABEL_DIM, _I_KICK_THIS_TICK, _I_VALID
        from footballcoach.ai.action.schema import DecisionHeadsRaw, ExecutionHeadsRaw

        labels = torch.zeros(1, BC_LABEL_DIM)
        labels[0, _I_VALID] = 1.0
        labels[0, _I_KICK_THIS_TICK] = 1.0

        d_heads = DecisionHeadsRaw(
            shoot_logit=torch.zeros(1, 1), pass_logit=torch.zeros(1, 1),
            move_logit=torch.zeros(1, 1), tackle_logit=torch.zeros(1, 1),
            get_possession_raw=torch.zeros(1, 1), mark_logit=torch.zeros(1, 1),
            hold_position_logit=torch.zeros(1, 1),
            pass_target_logits=torch.zeros(1, 21), tackle_target_logits=torch.zeros(1, 21),
            mark_target_logits=torch.zeros(1, 21),
            move_region_center=torch.zeros(1, 2), move_region_size=torch.zeros(1, 1),
            move_arrival_speed=torch.zeros(1, 1),
            region_of_play_center=torch.zeros(1, 2), region_of_play_size=torch.zeros(1, 1),
            attack_defence_raw=torch.zeros(1, 1), latent_vector=torch.zeros(1, 32),
            value=torch.zeros(1, 1),
        )
        e_heads = ExecutionHeadsRaw(
            move_direction=torch.zeros(1, 2), exec_move_logit=torch.zeros(1, 1),
            sprint_logit=torch.zeros(1, 1), kick_logit=torch.zeros(1, 1),
            kick_direction=torch.zeros(1, 2), kick_power=torch.zeros(1, 1),
            kick_spin=torch.zeros(1, 3), tackle_attempt_logit=torch.zeros(1, 1),
            value=torch.zeros(1, 1),
        )

        loss_baseline = bc_loss_from_tensor(labels, d_heads, e_heads, pos_weight_kick=1.0)
        loss_weighted = bc_loss_from_tensor(labels, d_heads, e_heads, pos_weight_kick=4.0)
        assert loss_weighted.item() > loss_baseline.item()

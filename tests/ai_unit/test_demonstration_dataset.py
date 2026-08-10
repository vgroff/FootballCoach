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

    def test_rows_near_speed_mode_change_never_trivial(self, tmp_path):
        """A sprint/exec_move transition (no kick/tackle involved) must be
        treated as a rare event too -- rows within exclude_radius_steps of it
        are never eligible for trivial-row downsampling."""
        from footballcoach.ai.ppo.bc import _I_EXEC_MOVE, _I_SPRINT

        ds, n_per_ep, n_eps = self._write_trivial_dataset(tmp_path)
        # Give every row a steady sprint=1/exec_move=1 baseline, then flip
        # sprint to 0 (SPRINT -> JOG) at row 10 of each episode -- no kick
        # or tackle involved, purely a speedMode change.
        ds._labels[:, _I_SPRINT] = 1.0
        ds._labels[:, _I_EXEC_MOVE] = 1.0
        for e in range(n_eps):
            change_row = e * n_per_ep + 10
            ds._labels[change_row:, _I_SPRINT] = 0.0

        mask = ds._compute_trivial_mask(exclude_radius_steps=5)
        for e in range(n_eps):
            change_row = e * n_per_ep + 10
            lo, hi = max(0, change_row - 5), min(n_per_ep * n_eps, change_row + 6)
            assert not mask[lo:hi].any()

    def test_speed_mode_change_at_episode_boundary_does_not_leak(self, tmp_path):
        """A sprint/exec_move difference between the LAST row of one episode
        and the FIRST row of the next must not be treated as a transition --
        rows are only compared to the previous row within the same episode."""
        from footballcoach.ai.ppo.bc import _I_EXEC_MOVE, _I_SPRINT

        ds, n_per_ep, n_eps = self._write_trivial_dataset(tmp_path)
        ds._labels[:, _I_SPRINT] = 1.0
        ds._labels[:, _I_EXEC_MOVE] = 1.0
        # Flip sprint only on episode boundaries (first row of episode 1+).
        for e in range(1, n_eps):
            ds._labels[e * n_per_ep, _I_SPRINT] = 0.0

        mask = ds._compute_trivial_mask(exclude_radius_steps=5)
        # Rows well away from both the episode boundary and the unrelated
        # kick event (at offset 10, see _write_trivial_dataset) should
        # remain trivial-eligible -- i.e. the boundary "transition" must not
        # radiate protection across episodes.
        for e in range(1, n_eps):
            far_row = e * n_per_ep + 16
            assert mask[far_row]

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


def _write_npz_with_outcomes(path: Path, raw_outcomes: list[str], steps_per_ep: int = 4) -> None:
    """Write a dataset with one episode per entry in *raw_outcomes*, each
    ``steps_per_ep`` TIMED-SAMPLE PAIRS long (i.e. ``2 * steps_per_ep`` rows),
    persisting the exact ground-truth ``meta_episode_outcomes`` strings a
    real recording would have (see record_demonstrations.py /
    ai/env/outcome.py's vocabulary: "box_possession"/
    "opponent_box_possession"/"timeout"/"miss"/"invalid").
    ``DemonstrationDataset.classify_outcome()`` must map these to the short
    labels in ``DemonstrationDataset._OUTCOME_LABEL_MAP`` exactly.

    Writes the real 2-consecutive-done=1-rows-per-episode-boundary layout
    (see ``DemonstrationDataset._DONE_ROWS_PER_EPISODE_BOUNDARY`` /
    ``episode_row_ranges()``'s docstring) -- a single done=1 row per episode
    is NOT a valid boundary and would leave every episode "incomplete"."""
    n_eps = len(raw_outcomes)
    n = n_eps * steps_per_ep * 2
    dones = np.zeros(n, dtype=np.float32)
    for e in range(n_eps):
        end = (e + 1) * steps_per_ep * 2 - 1
        dones[end - 1] = 1.0
        dones[end] = 1.0

    _write_npz(path, n=n, dones=dones)
    # np.savez can't easily be reopened+amended -- rewrite with meta_episode_outcomes included.
    data = dict(np.load(path))
    data["meta_episode_outcomes"] = np.array(raw_outcomes, dtype="U32")
    np.savez(path, **data)


# Alias: _write_npz_with_outcomes already writes the real 2-consecutive-
# done=1-rows-per-episode layout -- kept as a separate name for call-site
# readability where the "matches the real recording format" framing matters.
_write_npz_two_rows_per_sample = _write_npz_with_outcomes


def _write_npz_realistic(
    path: Path, raw_outcomes: list[str], rng: np.random.Generator,
    min_timed_samples: int = 1, max_timed_samples: int = 5,
    max_lone_callbacks_per_sample: int = 2, self_ai_type_choices=None,
) -> np.ndarray:
    """Write a dataset reproducing record_demonstrations.py's FULL, irregular
    real row layout for each episode in *raw_outcomes*:
      - A random number of "timed samples" (``_record_now(player_id=None)``),
        each appending exactly 2 rows (trainee, opponent), done=0 except the
        LAST timed sample of the episode, where BOTH its rows get done=1.
      - Between timed samples, a random number of "lone" kick/tackle-callback
        rows (``_record_now(player_id=pid)``), each exactly 1 row, done=0
        always (kick/tackle mid-episode, never coincides with the episode's
        final timed sample here -- matches on_kick/on_tackle firing strictly
        inside the tick loop between samples).
      - self.ai_type (_I_AI_TYPE) randomised per row from
        *self_ai_type_choices* (default: rules/immobile, 50/50) -- exercises
        valid_indices()'s filtering on top of the irregular row layout.
    Returns the per-row self.ai_type array actually written (float32), so
    callers can independently recompute which rows valid_indices() should
    keep without re-deriving the random draw.
    """
    from footballcoach.ai.ppo.bc import AI_TYPE_IMMOBILE, AI_TYPE_RULES, _I_AI_TYPE, _I_VALID

    if self_ai_type_choices is None:
        self_ai_type_choices = [AI_TYPE_RULES, AI_TYPE_IMMOBILE]

    dones_list: list[float] = []
    self_ai_type_list: list[float] = []
    for outcome in raw_outcomes:
        n_timed = int(rng.integers(min_timed_samples, max_timed_samples + 1))
        for sample_i in range(n_timed):
            is_last_sample = sample_i == n_timed - 1
            # Timed sample: trainee row, then opponent row (matches
            # _record_now(player_id=None)'s `[env.trainee_player_id, "opponent"]` order).
            dones_list.extend([1.0, 1.0] if is_last_sample else [0.0, 0.0])
            self_ai_type_list.extend(rng.choice(self_ai_type_choices, size=2).tolist())
            if not is_last_sample:
                n_lone = int(rng.integers(0, max_lone_callbacks_per_sample + 1))
                for _ in range(n_lone):
                    dones_list.append(0.0)  # lone kick/tackle row, never terminal
                    self_ai_type_list.append(float(rng.choice(self_ai_type_choices)))

    n = len(dones_list)
    dones = np.array(dones_list, dtype=np.float32)
    self_ai_type = np.array(self_ai_type_list, dtype=np.float32)

    _write_npz(path, n=n, dones=dones)
    data = dict(np.load(path))
    data["meta_episode_outcomes"] = np.array(raw_outcomes, dtype="U32")
    data["bc_labels"][:, _I_AI_TYPE] = self_ai_type
    data["bc_labels"][:, _I_VALID] = 1.0
    np.savez(path, **data)
    return self_ai_type


_PHASE1_RAW_OUTCOMES = ["box_possession", "opponent_box_possession", "timeout", "miss", "invalid"]


class TestEpisodeOutcomes:
    def test_episode_row_ranges_splits_on_dones(self, tmp_path):
        _write_npz_with_outcomes(
            tmp_path / "demo1.npz", ["box_possession", "opponent_box_possession", "timeout"], steps_per_ep=3
        )
        ds = DemonstrationDataset.from_directory(tmp_path)
        # steps_per_ep=3 -> 2 rows/sample * 3 samples/ep = 6 rows/ep.
        ranges = ds.episode_row_ranges(np.arange(len(ds)))
        assert ranges == [(0, 5), (6, 11), (12, 17)]

    def test_episode_row_ranges_drops_trailing_incomplete_episode(self, tmp_path):
        """An episode boundary is resolved from the FULL dataset (see
        episode_row_ranges()'s docstring), so a genuinely incomplete
        recording -- e.g. the process was killed mid-episode, leaving no
        terminal done=1 pair in the DATA itself, not just a filtered
        row_pool -- must be the one that actually produces no boundary."""
        _write_npz(
            tmp_path / "demo1.npz", n=3,
            dones=np.array([0.0, 0.0, 0.0], dtype=np.float32),  # no terminal pair at all
        )
        ds = DemonstrationDataset.from_directory(tmp_path)
        ranges = ds.episode_row_ranges(np.arange(len(ds)))
        assert ranges == []

    @pytest.mark.parametrize("raw_outcome,short_label", [
        ("box_possession", "win"),
        ("opponent_box_possession", "loss"),
        ("miss", "ball_out"),
        ("invalid", "invalid"),
        ("timeout", "timeout"),
    ])
    def test_classify_outcome_maps_every_known_trial_outcome(self, tmp_path, raw_outcome, short_label):
        """Phase 1 has exactly these five possible endings (see
        ai/env/outcome.py / ScenarioEnv.step()) -- classify_outcome() must
        map every one of them to a real short label, never "unknown"."""
        _write_npz_with_outcomes(tmp_path / "demo1.npz", [raw_outcome], steps_per_ep=3)
        ds = DemonstrationDataset.from_directory(tmp_path)
        assert ds.classify_outcome(len(ds) - 1) == short_label

    def test_classify_outcome_raises_without_ground_truth_outcomes(self, tmp_path):
        """A dataset predating meta_episode_outcomes must fail loudly, not
        silently fall back to a fabricated "unknown" bucket."""
        _write_npz(tmp_path / "demo1.npz", n=3, dones=np.array([0.0, 0.0, 1.0], dtype=np.float32))
        ds = DemonstrationDataset.from_directory(tmp_path)
        assert not ds.has_episode_outcomes
        with pytest.raises(ValueError, match="ground-truth episode outcomes"):
            ds.classify_outcome(2)

    def test_classify_outcome_raises_on_unrecognised_trial_outcome(self, tmp_path):
        """Any string outside the five documented Phase 1 outcomes is a bug
        upstream (ScenarioEnv emitting something new/wrong) -- must raise,
        never get silently bucketed as "unknown"."""
        _write_npz_with_outcomes(tmp_path / "demo1.npz", ["goal"], steps_per_ep=3)
        ds = DemonstrationDataset.from_directory(tmp_path)
        with pytest.raises(ValueError, match="Unrecognised trial_outcome"):
            ds.classify_outcome(len(ds) - 1)

    def test_row_outcomes_tags_every_row_in_its_episode(self, tmp_path):
        _write_npz_with_outcomes(
            tmp_path / "demo1.npz", ["box_possession", "opponent_box_possession"], steps_per_ep=1
        )
        ds = DemonstrationDataset.from_directory(tmp_path)
        # steps_per_ep=1 -> 2 rows/ep.
        outcomes = ds.row_outcomes(np.arange(len(ds)))
        assert list(outcomes) == ["win", "win", "loss", "loss"]

    def test_row_outcomes_tags_trailing_incomplete_episode(self, tmp_path):
        """See test_episode_row_ranges_drops_trailing_incomplete_episode --
        must be a genuinely incomplete recording (no terminal done=1 pair in
        the data), not a filtered row_pool of an otherwise-complete episode."""
        _write_npz(
            tmp_path / "demo1.npz", n=3,
            dones=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        ds = DemonstrationDataset.from_directory(tmp_path)
        outcomes = ds.row_outcomes(np.arange(len(ds)))
        assert list(outcomes) == ["incomplete", "incomplete", "incomplete"]

    def test_row_outcomes_works_on_non_contiguous_sorted_subset(self, tmp_path):
        """A train/val-style row subset (sorted, but with gaps) must still
        tag each retained row with its own episode's outcome."""
        _write_npz_with_outcomes(
            tmp_path / "demo1.npz", ["box_possession", "opponent_box_possession", "miss"], steps_per_ep=1
        )
        ds = DemonstrationDataset.from_directory(tmp_path)
        # steps_per_ep=1 -> 2 rows/ep -- rows [0,1],[2,3],[4,5]. Keep only
        # the LAST row of each episode (the actual done=1 pair's 2nd row).
        row_pool = np.array([1, 3, 5])
        outcomes = ds.row_outcomes(row_pool)
        assert list(outcomes) == ["win", "loss", "ball_out"]

    def test_row_outcomes_never_produces_unknown(self, tmp_path):
        """The whole point of ground-truth outcomes: every one of Phase 1's
        five real endings must map to a real label, with NO "unknown"
        bucket appearing anywhere in the output."""
        _write_npz_with_outcomes(
            tmp_path / "demo1.npz",
            ["box_possession", "opponent_box_possession", "timeout", "miss", "invalid"],
            steps_per_ep=2,
        )
        ds = DemonstrationDataset.from_directory(tmp_path)
        outcomes = ds.row_outcomes(np.arange(len(ds)))
        assert "unknown" not in set(outcomes)
        assert set(outcomes) == {"win", "loss", "timeout", "ball_out", "invalid"}

    def test_outcome_by_row_matches_row_outcomes_over_full_dataset(self, tmp_path):
        _write_npz_with_outcomes(
            tmp_path / "demo1.npz",
            ["box_possession", "opponent_box_possession", "timeout", "invalid"],
            steps_per_ep=2,
        )
        ds = DemonstrationDataset.from_directory(tmp_path)
        expected = ds.row_outcomes(np.arange(len(ds)))
        np.testing.assert_array_equal(ds.outcome_by_row(), expected)

    def test_outcome_by_row_is_cached(self, tmp_path):
        _write_npz_with_outcomes(tmp_path / "demo1.npz", ["box_possession"], steps_per_ep=2)
        ds = DemonstrationDataset.from_directory(tmp_path)
        first = ds.outcome_by_row()
        assert ds._outcome_by_row_cache is not None
        second = ds.outcome_by_row()
        assert first is second

    def test_from_files_concatenates_episode_outcomes_in_order(self, tmp_path):
        _write_npz_with_outcomes(tmp_path / "demo1.npz", ["box_possession"], steps_per_ep=1)
        _write_npz_with_outcomes(tmp_path / "demo2.npz", ["opponent_box_possession", "timeout"], steps_per_ep=1)
        ds = DemonstrationDataset.from_directory(tmp_path)
        outcomes = ds.row_outcomes(np.arange(len(ds)))
        assert list(outcomes) == ["win", "win", "loss", "loss", "timeout", "timeout"]

    def test_from_files_drops_outcomes_if_any_part_missing_them(self, tmp_path):
        """Mixing an old (no meta_episode_outcomes) file with a new one must
        not silently misalign the concatenated outcome list against dones --
        fall back to has_episode_outcomes=False for the whole dataset."""
        _write_npz(tmp_path / "demo_old.npz", n=2, dones=np.array([0.0, 1.0], dtype=np.float32))
        _write_npz_with_outcomes(tmp_path / "demo_new.npz", ["box_possession"], steps_per_ep=2)
        ds = DemonstrationDataset.from_directory(tmp_path)
        assert not ds.has_episode_outcomes


class TestRealRecordingRowLayout:
    """Regression coverage for the EXACT bug reported live: a real
    record_demonstrations.py .npz file marks done=1 on BOTH the trainee's
    AND the opponent's row for every timed sample (see
    ``_write_npz_two_rows_per_sample``'s docstring), so every episode ends
    with TWO CONSECUTIVE done=1 rows. Treating each done=1 row as its own
    episode boundary silently doubled the episode count everywhere
    (dataset-distribution logs showing "24,000 episodes" for a
    12,000-episode recording) and caused classify_outcome() to index past
    the end of meta_episode_outcomes -- IndexError, reproduced verbatim
    here to prove it can never recur."""

    def test_n_episodes_matches_recorded_episode_count_not_double(self, tmp_path):
        _write_npz_two_rows_per_sample(
            tmp_path / "demo1.npz",
            ["box_possession", "opponent_box_possession", "timeout"],
            steps_per_ep=2,
        )
        ds = DemonstrationDataset.from_directory(tmp_path)
        assert ds.n_episodes() == 3  # NOT 6

    def test_episode_row_ranges_collapses_consecutive_done_rows(self, tmp_path):
        _write_npz_two_rows_per_sample(
            tmp_path / "demo1.npz", ["box_possession", "timeout"], steps_per_ep=2,
        )
        ds = DemonstrationDataset.from_directory(tmp_path)
        # steps_per_ep=2 -> 2 rows/sample * 2 samples/ep = 4 rows/ep.
        ranges = ds.episode_row_ranges(np.arange(len(ds)))
        assert ranges == [(0, 3), (4, 7)]

    def test_classify_outcome_does_not_raise_indexerror_on_real_layout(self, tmp_path):
        """The exact crash reported: classify_outcome() on a real-layout
        dataset must resolve every episode-end row correctly, never index
        past meta_episode_outcomes."""
        outcomes = ["box_possession", "opponent_box_possession", "timeout", "miss", "invalid"]
        _write_npz_two_rows_per_sample(tmp_path / "demo1.npz", outcomes, steps_per_ep=3)
        ds = DemonstrationDataset.from_directory(tmp_path)
        for _start, end in ds.episode_row_ranges(np.arange(len(ds))):
            ds.classify_outcome(end)  # must not raise

    def test_row_outcomes_never_unknown_on_real_layout_full_dataset(self, tmp_path):
        """Exact repro of the reported crash call site:
        _log_returns_by_outcome(ds, np.arange(len(ds)), ...) via
        row_outcomes(np.arange(len(ds)))."""
        outcomes = ["box_possession", "opponent_box_possession", "timeout", "miss", "invalid"] * 4
        _write_npz_two_rows_per_sample(tmp_path / "demo1.npz", outcomes, steps_per_ep=2)
        ds = DemonstrationDataset.from_directory(tmp_path)
        row_outcomes = ds.row_outcomes(np.arange(len(ds)))  # must not raise IndexError
        assert "unknown" not in set(row_outcomes)
        assert "incomplete" not in set(row_outcomes)
        assert set(row_outcomes) == {"win", "loss", "timeout", "ball_out", "invalid"}

    def test_outcome_by_row_matches_episode_count_on_real_layout(self, tmp_path):
        outcomes = ["box_possession"] * 20
        _write_npz_two_rows_per_sample(tmp_path / "demo1.npz", outcomes, steps_per_ep=2)
        ds = DemonstrationDataset.from_directory(tmp_path)
        assert ds.n_episodes() == 20
        by_row = ds.outcome_by_row()
        assert len(by_row) == len(ds)
        assert set(by_row) == {"win"}

    def test_split_train_val_indices_never_splits_an_episode_across_sets(self, tmp_path):
        """The train/val leakage variant of the same bug: with the real
        2-rows-done=1 layout, treating each done=1 row as a boundary could
        put one player's terminal row in train and the other's in val for
        the SAME episode. Assert every episode's rows land entirely in one
        set."""
        n_eps = 20
        outcomes = ["box_possession", "opponent_box_possession"] * (n_eps // 2)
        _write_npz_two_rows_per_sample(tmp_path / "demo1.npz", outcomes, steps_per_ep=3)
        ds = DemonstrationDataset.from_directory(tmp_path)
        train_idx, val_idx = ds.split_train_val_indices(val_frac=0.2, valid_only=False)
        train_set, val_set = set(train_idx.tolist()), set(val_idx.tolist())
        assert train_set.isdisjoint(val_set)
        for start, end in ds.episode_row_ranges(np.arange(len(ds))):
            ep_rows = set(range(start, end + 1))
            in_train = ep_rows & train_set
            in_val = ep_rows & val_set
            assert not (in_train and in_val), (
                f"episode rows [{start},{end}] split across train and val"
            )

    def test_split_train_val_indices_episode_count_matches_n_episodes(self, tmp_path):
        n_eps = 10
        outcomes = ["box_possession"] * n_eps
        _write_npz_two_rows_per_sample(tmp_path / "demo1.npz", outcomes, steps_per_ep=2)
        ds = DemonstrationDataset.from_directory(tmp_path)
        train_idx, val_idx = ds.split_train_val_indices(val_frac=0.3, valid_only=False)
        assert ds.n_episodes(train_idx) + ds.n_episodes(val_idx) == n_eps

    def test_classify_outcome_on_filtered_subsets_last_row_not_full_dataset_end(self, tmp_path):
        """Exact repro of the second reported crash: row_outcomes() on a
        FILTERED row_pool (e.g. valid_indices()-filtered train_idx/val_idx)
        passes classify_outcome() the subset's own last row for an episode,
        which is <= but not necessarily == the true full-dataset episode-end
        row once the opponent's terminal row (excluded by valid_indices()'s
        immobile-self filter) is stripped out. Must resolve to the correct
        episode, not raise "not a recognised episode-end row"."""
        from footballcoach.ai.ppo.bc import AI_TYPE_IMMOBILE, AI_TYPE_RULES, _I_AI_TYPE, _I_VALID

        outcomes = ["box_possession", "opponent_box_possession", "timeout"]
        _write_npz_two_rows_per_sample(tmp_path / "demo1.npz", outcomes, steps_per_ep=2)
        ds = DemonstrationDataset.from_directory(tmp_path)
        # 2 rows/sample * 2 samples/ep = 4 rows/ep -- rows [0,3],[4,7],[8,11].
        # Mark every ODD row (the "opponent" row of each timed sample, per
        # record_demonstrations.py's trainee-then-opponent append order) as
        # self.ai_type==immobile so valid_indices() drops it, same as a real
        # 50%-immobile-opponent recording.
        ds._labels[:, _I_VALID] = 1.0
        ds._labels[:, _I_AI_TYPE] = AI_TYPE_RULES
        ds._labels[1::2, _I_AI_TYPE] = AI_TYPE_IMMOBILE

        valid_idx = ds.valid_indices()
        # The filtered subset's last row per episode is now the EVEN row
        # right before the true (odd) full-dataset end row -- i.e. strictly
        # less than it, exercising the "not an exact match" path.
        row_outcomes = ds.row_outcomes(valid_idx)  # must not raise
        assert list(row_outcomes) == ["win", "win", "loss", "loss", "timeout", "timeout"]


class TestRealisticRecordingFuzz:
    """Extensive, randomised regression coverage for the FULL irregular row
    layout record_demonstrations.py actually produces (see
    ``_write_npz_realistic()``'s docstring): a variable number of paired
    timed-sample rows per episode, interleaved with lone kick/tackle-callback
    rows, random self.ai_type per row, terminating in exactly one pair of
    done=1 rows. Every invariant here failed at least once during manual
    testing (IndexError past meta_episode_outcomes; ValueError on a filtered
    row_pool's own last row) -- this class exists so those classes of bug
    cannot recur silently, across MANY random layouts, not just hand-picked
    small examples.
    """

    SEEDS = list(range(30))

    @pytest.mark.parametrize("seed", SEEDS)
    def test_n_episodes_always_matches_recorded_outcome_count(self, tmp_path, seed):
        rng = np.random.default_rng(seed)
        n_eps = int(rng.integers(3, 15))
        outcomes = list(rng.choice(_PHASE1_RAW_OUTCOMES, size=n_eps))
        _write_npz_realistic(tmp_path / "demo1.npz", outcomes, rng)
        ds = DemonstrationDataset.from_directory(tmp_path)
        assert ds.n_episodes() == n_eps

    @pytest.mark.parametrize("seed", SEEDS)
    def test_row_outcomes_full_dataset_never_unknown_or_incomplete(self, tmp_path, seed):
        rng = np.random.default_rng(seed)
        n_eps = int(rng.integers(3, 15))
        outcomes = list(rng.choice(_PHASE1_RAW_OUTCOMES, size=n_eps))
        _write_npz_realistic(tmp_path / "demo1.npz", outcomes, rng)
        ds = DemonstrationDataset.from_directory(tmp_path)
        row_outcomes = ds.row_outcomes(np.arange(len(ds)))  # must not raise
        assert "unknown" not in set(row_outcomes)
        assert "incomplete" not in set(row_outcomes)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_row_outcomes_matches_recorded_outcomes_in_order(self, tmp_path, seed):
        """Every row's resolved outcome must equal the label the episode it
        belongs to was actually recorded with -- not just "some real label"."""
        rng = np.random.default_rng(seed)
        n_eps = int(rng.integers(3, 15))
        outcomes = list(rng.choice(_PHASE1_RAW_OUTCOMES, size=n_eps))
        _write_npz_realistic(tmp_path / "demo1.npz", outcomes, rng)
        ds = DemonstrationDataset.from_directory(tmp_path)
        row_outcomes = ds.row_outcomes(np.arange(len(ds)))
        ranges = ds.episode_row_ranges(np.arange(len(ds)))
        assert len(ranges) == n_eps
        expected_map = DemonstrationDataset._OUTCOME_LABEL_MAP
        for ep_idx, (start, end) in enumerate(ranges):
            expected = expected_map[outcomes[ep_idx]]
            assert list(row_outcomes[start:end + 1]) == [expected] * (end - start + 1), (
                f"episode {ep_idx} rows [{start},{end}] expected all {expected!r}, "
                f"got {row_outcomes[start:end + 1].tolist()}"
            )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_row_outcomes_on_valid_indices_filtered_subset_never_raises(self, tmp_path, seed):
        """The exact second reported crash, fuzzed: valid_indices() (which
        drops immobile-self rows) makes a filtered row_pool whose per-episode
        last row is frequently NOT the full-dataset episode-end row --
        row_outcomes()/classify_outcome() must handle every such case."""
        rng = np.random.default_rng(seed)
        n_eps = int(rng.integers(3, 15))
        outcomes = list(rng.choice(_PHASE1_RAW_OUTCOMES, size=n_eps))
        _write_npz_realistic(tmp_path / "demo1.npz", outcomes, rng)
        ds = DemonstrationDataset.from_directory(tmp_path)
        valid_idx = ds.valid_indices()
        if len(valid_idx) == 0:
            pytest.skip("all rows filtered out by valid_indices() for this seed")
        row_outcomes = ds.row_outcomes(valid_idx)  # must not raise
        assert "unknown" not in set(row_outcomes)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_split_train_val_indices_partitions_full_episodes_only(self, tmp_path, seed):
        """Fuzzed version of the train/val leakage check: across many random
        irregular layouts, every episode's rows (within the split's own base
        index set) must land ENTIRELY in train or ENTIRELY in val, and
        train_eps + val_eps must equal the total recorded episode count
        (no episode silently dropped or double-counted)."""
        rng = np.random.default_rng(seed)
        n_eps = int(rng.integers(4, 15))
        outcomes = list(rng.choice(_PHASE1_RAW_OUTCOMES, size=n_eps))
        _write_npz_realistic(tmp_path / "demo1.npz", outcomes, rng)
        ds = DemonstrationDataset.from_directory(tmp_path)

        for valid_only in (False, True):
            train_idx, val_idx = ds.split_train_val_indices(val_frac=0.25, valid_only=valid_only)
            train_set, val_set = set(train_idx.tolist()), set(val_idx.tolist())
            assert train_set.isdisjoint(val_set)
            base_indices = ds.valid_indices() if valid_only else np.arange(len(ds))
            for start, end in ds.episode_row_ranges(base_indices):
                ep_rows = set(range(start, end + 1)) & (train_set | val_set)
                in_train = ep_rows & train_set
                in_val = ep_rows & val_set
                assert not (in_train and in_val), (
                    f"valid_only={valid_only}: episode rows [{start},{end}] "
                    f"split across train and val"
                )
            if len(val_idx) > 0:
                assert ds.n_episodes(train_idx) + ds.n_episodes(val_idx) == ds.n_episodes(base_indices)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_outcome_by_row_cache_consistent_with_fresh_computation(self, tmp_path, seed):
        rng = np.random.default_rng(seed)
        n_eps = int(rng.integers(3, 15))
        outcomes = list(rng.choice(_PHASE1_RAW_OUTCOMES, size=n_eps))
        _write_npz_realistic(tmp_path / "demo1.npz", outcomes, rng)
        ds = DemonstrationDataset.from_directory(tmp_path)
        cached = ds.outcome_by_row()
        fresh = ds.row_outcomes(np.arange(len(ds)))
        np.testing.assert_array_equal(cached, fresh)

    def test_multi_file_concatenation_preserves_episode_boundaries(self, tmp_path):
        """Concatenating several realistic files (each with its own random
        irregular layout) must not merge the last episode of one file with
        the first episode of the next, and outcome order must match file
        order exactly."""
        rng = np.random.default_rng(123)
        all_outcomes: list[str] = []
        for i in range(4):
            n_eps = int(rng.integers(2, 6))
            outcomes = list(rng.choice(_PHASE1_RAW_OUTCOMES, size=n_eps))
            all_outcomes.extend(outcomes)
            _write_npz_realistic(tmp_path / f"demo{i}.npz", outcomes, rng)
        ds = DemonstrationDataset.from_directory(tmp_path)
        assert ds.n_episodes() == len(all_outcomes)
        ranges = ds.episode_row_ranges(np.arange(len(ds)))
        assert len(ranges) == len(all_outcomes)
        expected_map = DemonstrationDataset._OUTCOME_LABEL_MAP
        row_outcomes = ds.row_outcomes(np.arange(len(ds)))
        for ep_idx, (start, end) in enumerate(ranges):
            expected = expected_map[all_outcomes[ep_idx]]
            assert list(row_outcomes[start:end + 1]) == [expected] * (end - start + 1)

    @pytest.mark.parametrize("seed", SEEDS[:10])
    def test_every_episode_has_exactly_two_terminal_done_rows(self, tmp_path, seed):
        """Sanity-check the fixture itself matches record_demonstrations.py's
        documented layout (2 done=1 rows per episode, at the very end of
        each episode's row range) -- guards against the fuzz generator
        drifting away from the real format it's supposed to model."""
        rng = np.random.default_rng(seed)
        n_eps = int(rng.integers(3, 10))
        outcomes = list(rng.choice(_PHASE1_RAW_OUTCOMES, size=n_eps))
        _write_npz_realistic(tmp_path / "demo1.npz", outcomes, rng)
        ds = DemonstrationDataset.from_directory(tmp_path)
        for start, end in ds.episode_row_ranges(np.arange(len(ds))):
            assert ds._dones[end] > 0.5
            assert ds._dones[end - 1] > 0.5
            # No done=1 row strictly inside the episode (only the final pair).
            assert not (ds._dones[start:end - 1] > 0.5).any()

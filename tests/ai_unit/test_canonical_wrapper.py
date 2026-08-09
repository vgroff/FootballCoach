"""Aggressive tests for CanonicalNetworkWrapper (ai/obs/canonical.py).

Covers: transparent delegation (state_dict/parameters/train/eval/to),
automatic input canonicalization on every forward call, equivariance of
real DecisionNetwork/ExecutionNetwork outputs under a LEFT vs mirrored-RIGHT
observation pair, checkpoint round-trip compatibility (unprefixed keys),
PPOTrainer end-to-end wiring (_sample_action / _get_value), and the
N_FLIP_VARIANTS regression that broke augment_obs_bc()'s return-tiling.
"""
import copy

import numpy as np
import pytest
import torch

from footballcoach.ai.models.decision_network import DecisionNetwork
from footballcoach.ai.models.execution_network import ExecutionNetwork
from footballcoach.ai.obs.augment import N_FLIP_VARIANTS, augment_obs_bc
from footballcoach.ai.obs.canonical import CanonicalNetworkWrapper, X_SIGN_FIELD_IDX
from footballcoach.ai.obs.schema import (
    BALL_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    MAX_OTHER_PLAYERS,
    PLAYER_FEATURE_DIM,
)


def _rand_batch(n=4, n_slots=MAX_OTHER_PLAYERS):
    torch.manual_seed(0)
    sf = torch.randn(n, PLAYER_FEATURE_DIM)
    of = torch.randn(n, n_slots, PLAYER_FEATURE_DIM)
    em = torch.ones(n, n_slots)
    bf = torch.randn(n, BALL_FEATURE_DIM)
    gf = torch.randn(n, GLOBAL_FEATURE_DIM)
    return sf, of, em, bf, gf


class TestTransparentDelegation:
    def test_state_dict_keys_unprefixed(self):
        net = DecisionNetwork.from_config()
        wrapped = CanonicalNetworkWrapper(net)
        raw_keys = set(net.state_dict().keys())
        wrapped_keys = set(wrapped.state_dict().keys())
        assert raw_keys == wrapped_keys

    def test_load_state_dict_roundtrip(self):
        net = DecisionNetwork.from_config()
        wrapped = CanonicalNetworkWrapper(net)
        sd = copy.deepcopy(wrapped.state_dict())
        # Mutate a param, then reload — should restore exactly.
        with torch.no_grad():
            for p in wrapped.parameters():
                p.add_(1.0)
        wrapped.load_state_dict(sd)
        for (k, v) in wrapped.state_dict().items():
            assert torch.allclose(v, sd[k])

    def test_parameters_match_wrapped(self):
        net = DecisionNetwork.from_config()
        wrapped = CanonicalNetworkWrapper(net)
        raw_n = sum(p.numel() for p in net.parameters())
        wrapped_n = sum(p.numel() for p in wrapped.parameters())
        assert raw_n == wrapped_n

    def test_attribute_passthrough(self):
        net = ExecutionNetwork.from_config()
        wrapped = CanonicalNetworkWrapper(net)
        assert wrapped.move_dir_log_std is net.move_dir_log_std
        assert wrapped.value_head is net.value_head

    def test_train_eval_mode_propagates(self):
        net = DecisionNetwork.from_config()
        wrapped = CanonicalNetworkWrapper(net)
        wrapped.eval()
        assert not net.training
        wrapped.train()
        assert net.training

    def test_to_device_noop_on_cpu(self):
        net = DecisionNetwork.from_config()
        wrapped = CanonicalNetworkWrapper(net).to(torch.device("cpu"))
        sf, of, em, bf, gf = _rand_batch()
        out = wrapped(sf, of, em, bf, gf)
        assert out.value.shape == (4, 1)

    def test_deepcopy_state_dict_independent(self):
        # copy.deepcopy(wrapped.state_dict()) used by BC-pretrain early-stop
        # "best weights" snapshotting — must not alias the live tensors.
        net = DecisionNetwork.from_config()
        wrapped = CanonicalNetworkWrapper(net)
        snap = copy.deepcopy(wrapped.state_dict())
        with torch.no_grad():
            for p in wrapped.parameters():
                p.add_(5.0)
        for k, v in snap.items():
            assert not torch.allclose(v, wrapped.state_dict()[k])


class TestForwardCanonicalization:
    def test_output_shapes_unaffected(self):
        net = DecisionNetwork.from_config()
        wrapped = CanonicalNetworkWrapper(net)
        sf, of, em, bf, gf = _rand_batch()
        raw_out = net(sf, of, em, bf, gf)
        wrapped_out = wrapped(sf, of, em, bf, gf)
        assert wrapped_out.value.shape == raw_out.value.shape
        assert wrapped_out.move_region_center.shape == raw_out.move_region_center.shape

    def test_left_team_forward_matches_raw_net(self):
        """attacking_direction=+1 (Team.LEFT) is a no-op mirror, so the
        wrapper's output must be IDENTICAL to calling the raw net directly."""
        net = DecisionNetwork.from_config()
        net.eval()
        wrapped = CanonicalNetworkWrapper(net)
        wrapped.eval()
        sf, of, em, bf, gf = _rand_batch()
        sf[:, X_SIGN_FIELD_IDX] = 1.0
        of[:, :, X_SIGN_FIELD_IDX] = 1.0
        with torch.no_grad():
            raw_out = net(sf, of, em, bf, gf)
            wrapped_out = wrapped(sf, of, em, bf, gf)
        assert torch.allclose(wrapped_out.value, raw_out.value)
        assert torch.allclose(wrapped_out.shoot_logit, raw_out.shoot_logit)

    def test_right_team_forward_differs_from_raw_net(self):
        """attacking_direction=-1 (Team.RIGHT) must trigger a real mirror —
        the wrapper's output should generally NOT equal a naive raw-net call
        on the same (unmirrored) input, since the network sees different
        numbers in each case."""
        net = DecisionNetwork.from_config()
        net.eval()
        wrapped = CanonicalNetworkWrapper(net)
        wrapped.eval()
        sf, of, em, bf, gf = _rand_batch()
        sf[:, X_SIGN_FIELD_IDX] = -1.0
        of[:, :, X_SIGN_FIELD_IDX] = -1.0
        with torch.no_grad():
            raw_out = net(sf, of, em, bf, gf)
            wrapped_out = wrapped(sf, of, em, bf, gf)
        assert not torch.allclose(wrapped_out.value, raw_out.value)

    def test_execution_network_ball_feat_position_is_index_1(self):
        """Regression guard: CanonicalNetworkWrapper.forward() hardcodes
        ball_feat as rest[1] (position 3 overall: self,other,exists,ball,...).
        If ExecutionNetwork's signature ever changes order, this test fails
        loudly instead of silently canonicalizing the wrong tensor."""
        import inspect
        sig = inspect.signature(ExecutionNetwork.forward)
        params = list(sig.parameters.keys())
        assert params[:5] == ["self", "self_feat", "other_feat", "exists_mask", "ball_feat"]

    def test_decision_network_ball_feat_position_is_index_1(self):
        import inspect
        sig = inspect.signature(DecisionNetwork.forward)
        params = list(sig.parameters.keys())
        assert params[:5] == ["self", "self_feat", "other_feat", "exists_mask", "ball_feat"]


class TestEquivarianceAcrossTeams:
    """The core correctness property: a LEFT-team player at world position P
    and a RIGHT-team player at the x-mirrored world position P' (with all
    other x-signed quantities correspondingly mirrored) must produce
    IDENTICAL canonical-frame network outputs, since after canonicalization
    both observations should be numerically identical."""

    def test_manually_mirrored_input_gives_identical_output(self):
        net = DecisionNetwork.from_config()
        net.eval()
        wrapped = CanonicalNetworkWrapper(net)
        wrapped.eval()

        from footballcoach.ai.obs.augment import BALL_FLIP_X_IDX, PLAYER_FLIP_X_IDX

        sf_left, of_left, em, bf_left, gf = _rand_batch(n=2)
        sf_left[:, X_SIGN_FIELD_IDX] = 1.0
        of_left[:, :, X_SIGN_FIELD_IDX] = 1.0

        sf_right = sf_left.clone()
        of_right = of_left.clone()
        bf_right = bf_left.clone()
        sf_right[:, PLAYER_FLIP_X_IDX] *= -1.0
        of_right[:, :, PLAYER_FLIP_X_IDX] *= -1.0
        bf_right[:, BALL_FLIP_X_IDX] *= -1.0
        # attacking_direction (one of PLAYER_FLIP_X_IDX) is now -1.0 -> Team.RIGHT

        with torch.no_grad():
            out_left = wrapped(sf_left, of_left, em, bf_left, gf)
            out_right = wrapped(sf_right, of_right, em, bf_right, gf)

        assert torch.allclose(out_left.value, out_right.value, atol=1e-5)
        assert torch.allclose(out_left.shoot_logit, out_right.shoot_logit, atol=1e-5)
        assert torch.allclose(out_left.move_region_center, out_right.move_region_center, atol=1e-5)


class TestNFlipVariantsRegression:
    """N_FLIP_VARIANTS must equal len(_FLIP_VARIANTS) (2, since flip_x was
    removed) — a stale hardcoded 4 here previously broke augment_obs_bc()'s
    ret_batch.repeat(n_aug) tiling with a batch-size mismatch (see
    ppo_trainer.py pretrain_combined() Phase 1)."""

    def test_value_is_two(self):
        assert N_FLIP_VARIANTS == 2

    def test_augment_obs_bc_output_batch_matches_n_flip_variants(self):
        n = 3
        obs_dict = {
            "self_feat": torch.randn(n, PLAYER_FEATURE_DIM),
            "other_feat": torch.randn(n, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM),
            "exists_mask": torch.ones(n, MAX_OTHER_PLAYERS),
            "ball_feat": torch.randn(n, BALL_FEATURE_DIM),
            "global_feat": torch.randn(n, GLOBAL_FEATURE_DIM),
        }
        from footballcoach.ai.ppo.bc import BC_LABEL_DIM
        bc_labels = torch.zeros(n, BC_LABEL_DIM)
        import random
        aug_obs, aug_labels = augment_obs_bc(obs_dict, bc_labels, n_slot_shuffles=1, rng=random.Random(0))
        expected_n = n * N_FLIP_VARIANTS * 1
        assert aug_obs["self_feat"].shape[0] == expected_n
        assert aug_labels.shape[0] == expected_n

    def test_ret_batch_repeat_matches_augmented_batch_size(self):
        """Directly reproduces the exact failure mode from
        pretrain_combined(): ret_batch.repeat(N_FLIP_VARIANTS * n_slot_shuffles)
        must equal the augmented obs/labels batch size for F.mse_loss to not
        raise a broadcast RuntimeError."""
        n = 5
        obs_dict = {
            "self_feat": torch.randn(n, PLAYER_FEATURE_DIM),
            "other_feat": torch.randn(n, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM),
            "exists_mask": torch.ones(n, MAX_OTHER_PLAYERS),
            "ball_feat": torch.randn(n, BALL_FEATURE_DIM),
            "global_feat": torch.randn(n, GLOBAL_FEATURE_DIM),
        }
        from footballcoach.ai.ppo.bc import BC_LABEL_DIM
        bc_labels = torch.zeros(n, BC_LABEL_DIM)
        ret_batch = torch.randn(n)
        import random
        aug_obs, _ = augment_obs_bc(obs_dict, bc_labels, n_slot_shuffles=2, rng=random.Random(0))
        n_aug = N_FLIP_VARIANTS * max(1, 2)
        ret_batch_tiled = ret_batch.repeat(n_aug)
        assert aug_obs["self_feat"].shape[0] == ret_batch_tiled.shape[0]


class TestPPOTrainerIntegration:
    """End-to-end: PPOTrainer's networks are actually wrapped, and
    _sample_action/_get_value produce sane, finite outputs for both teams."""

    @pytest.fixture
    def trainer(self):
        from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
        return PPOTrainer.from_config()

    def test_networks_are_wrapped(self, trainer):
        assert isinstance(trainer.decision_net, CanonicalNetworkWrapper)
        assert isinstance(trainer.execution_net, CanonicalNetworkWrapper)

    def test_checkpoint_save_load_roundtrip(self, trainer, tmp_path):
        from pathlib import Path
        ckpt_path = tmp_path / "ckpt.pt"
        trainer._save_checkpoint_to(Path(ckpt_path))
        from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
        loaded = PPOTrainer.load_for_inference(str(ckpt_path))
        # Compare state dicts key-for-key.
        orig_sd = trainer.decision_net.state_dict()
        loaded_sd = loaded.decision_net.state_dict()
        assert set(orig_sd.keys()) == set(loaded_sd.keys())
        for k in orig_sd:
            assert torch.allclose(orig_sd[k], loaded_sd[k])

    def test_sample_action_finite_for_both_teams(self, trainer):
        from footballcoach.ai.obs.schema import ObservationBatch
        rng_state = torch.get_rng_state()
        for x_sign in (1.0, -1.0):
            self_feat = np.zeros(PLAYER_FEATURE_DIM, dtype=np.float32)
            self_feat[X_SIGN_FIELD_IDX] = x_sign
            other_feat = np.zeros((MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM), dtype=np.float32)
            exists_mask = np.zeros(MAX_OTHER_PLAYERS, dtype=np.float32)
            exists_mask[0] = 1.0  # at least one real other-player slot (avoid the
            other_feat[0, X_SIGN_FIELD_IDX] = x_sign  # unrelated fully-padded-row NaN edge case)
            obs = ObservationBatch(
                self_feat=self_feat,
                other_feat=other_feat,
                exists_mask=exists_mask,
                ball_feat=np.zeros(BALL_FEATURE_DIM, dtype=np.float32),
                global_feat=np.zeros(GLOBAL_FEATURE_DIM, dtype=np.float32),
            )
            obs_dict = obs.to_torch_dict()
            result = trainer._sample_action(obs_dict, deterministic=True)
            action, log_prob, value = result[0], result[1], result[2]
            assert np.isfinite(log_prob)
            assert np.isfinite(value)
            move_dir = result[4]["move_direction"]
            assert np.all(np.isfinite(move_dir))
        torch.set_rng_state(rng_state)

    def test_get_value_finite_for_both_teams(self, trainer):
        for x_sign in (1.0, -1.0):
            self_feat = torch.zeros(1, PLAYER_FEATURE_DIM)
            self_feat[0, X_SIGN_FIELD_IDX] = x_sign
            other_feat = torch.zeros(1, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM)
            other_feat[0, 0, X_SIGN_FIELD_IDX] = x_sign
            exists_mask = torch.zeros(1, MAX_OTHER_PLAYERS)
            exists_mask[0, 0] = 1.0  # avoid unrelated fully-padded-row NaN edge case
            obs_dict = {
                "self_feat": self_feat,
                "other_feat": other_feat,
                "exists_mask": exists_mask,
                "ball_feat": torch.zeros(1, BALL_FEATURE_DIM),
                "global_feat": torch.zeros(1, GLOBAL_FEATURE_DIM),
            }
            val = trainer._get_value(obs_dict)
            assert np.isfinite(val)

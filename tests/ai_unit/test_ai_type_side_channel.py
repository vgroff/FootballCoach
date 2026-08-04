"""Tests for the value-only opponent-AI-type side channel (W6).

See ai/knowledge.md "Opponent-AI-type (value-only)" for the design rationale:
``self_ai_type``/``other_ai_type`` are flat one-hot side channels that feed
ONLY ``value_head`` in both DecisionNetwork and ExecutionNetwork, bypassing
the entity encoder/attention and every policy head entirely.

These tests guard the three properties that matter most:
  1. Gradient isolation: policy heads must never receive gradient through
     self_ai_type/other_ai_type; value_head must.
  2. Permutation consistency: other_ai_type must be permuted in lockstep
     with other_feat/exists_mask (same slot shuffle), never desynced.
  3. Forward-pass shapes: value_head's input dimension accounts for the
     extra side-channel MLP output.
"""
import torch
from torch import nn

from footballcoach.ai.obs.schema import (
    AI_TYPE_ONE_HOT_DIM,
    BALL_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    MAX_OTHER_PLAYERS,
    PLAYER_FEATURE_DIM,
)
from footballcoach.ai.models.decision_network import DecisionNetwork
from footballcoach.ai.models.execution_network import ExecutionNetwork


def _make_obs(batch_size: int = 2):
    return {
        "self_feat":   torch.randn(batch_size, PLAYER_FEATURE_DIM),
        "other_feat":  torch.randn(batch_size, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM),
        "exists_mask": torch.ones(batch_size, MAX_OTHER_PLAYERS),
        "ball_feat":   torch.randn(batch_size, BALL_FEATURE_DIM),
        "global_feat": torch.randn(batch_size, GLOBAL_FEATURE_DIM),
    }


class TestDecisionNetworkGradientIsolation:
    def test_policy_heads_never_see_ai_type_gradient(self):
        net = DecisionNetwork.from_config()
        obs = _make_obs(4)
        self_ai_type = torch.zeros(4, AI_TYPE_ONE_HOT_DIM, requires_grad=True)
        other_ai_type = torch.zeros(
            4, MAX_OTHER_PLAYERS, AI_TYPE_ONE_HOT_DIM, requires_grad=True
        )

        heads = net(
            obs["self_feat"], obs["other_feat"], obs["exists_mask"],
            obs["ball_feat"], obs["global_feat"], self_ai_type, other_ai_type,
        )
        # Pick a pure policy head, unrelated to value.
        heads.shoot_logit.sum().backward(retain_graph=True)
        assert self_ai_type.grad is None or bool((self_ai_type.grad == 0).all())
        assert other_ai_type.grad is None or bool((other_ai_type.grad == 0).all())

    def test_value_head_does_see_ai_type_gradient(self):
        net = DecisionNetwork.from_config()
        obs = _make_obs(4)
        self_ai_type = torch.zeros(4, AI_TYPE_ONE_HOT_DIM, requires_grad=True)
        other_ai_type = torch.zeros(
            4, MAX_OTHER_PLAYERS, AI_TYPE_ONE_HOT_DIM, requires_grad=True
        )

        heads = net(
            obs["self_feat"], obs["other_feat"], obs["exists_mask"],
            obs["ball_feat"], obs["global_feat"], self_ai_type, other_ai_type,
        )
        heads.value.sum().backward()
        assert self_ai_type.grad is not None
        assert not bool((self_ai_type.grad == 0).all())
        assert other_ai_type.grad is not None
        assert not bool((other_ai_type.grad == 0).all())


class TestExecutionNetworkGradientIsolation:
    def test_motor_heads_never_see_ai_type_gradient(self):
        d_net = DecisionNetwork.from_config()
        e_net = ExecutionNetwork.from_config()
        obs = _make_obs(4)
        self_ai_type = torch.zeros(4, AI_TYPE_ONE_HOT_DIM, requires_grad=True)
        other_ai_type = torch.zeros(
            4, MAX_OTHER_PLAYERS, AI_TYPE_ONE_HOT_DIM, requires_grad=True
        )
        d_heads = d_net(
            obs["self_feat"], obs["other_feat"], obs["exists_mask"],
            obs["ball_feat"], obs["global_feat"], self_ai_type, other_ai_type,
        )
        e_heads = e_net(
            obs["self_feat"], obs["other_feat"], obs["exists_mask"],
            obs["ball_feat"], obs["global_feat"], d_heads, self_ai_type, other_ai_type,
        )
        e_heads.kick_logit.sum().backward(retain_graph=True)
        assert self_ai_type.grad is None or bool((self_ai_type.grad == 0).all())
        assert other_ai_type.grad is None or bool((other_ai_type.grad == 0).all())

    def test_value_head_does_see_ai_type_gradient(self):
        d_net = DecisionNetwork.from_config()
        e_net = ExecutionNetwork.from_config()
        obs = _make_obs(4)
        self_ai_type = torch.zeros(4, AI_TYPE_ONE_HOT_DIM, requires_grad=True)
        other_ai_type = torch.zeros(
            4, MAX_OTHER_PLAYERS, AI_TYPE_ONE_HOT_DIM, requires_grad=True
        )
        d_heads = d_net(
            obs["self_feat"], obs["other_feat"], obs["exists_mask"],
            obs["ball_feat"], obs["global_feat"], self_ai_type, other_ai_type,
        )
        e_heads = e_net(
            obs["self_feat"], obs["other_feat"], obs["exists_mask"],
            obs["ball_feat"], obs["global_feat"], d_heads, self_ai_type, other_ai_type,
        )
        e_heads.value.sum().backward()
        assert not bool((self_ai_type.grad == 0).all())
        assert not bool((other_ai_type.grad == 0).all())


class TestForwardShapes:
    def test_value_head_input_dim_includes_side_channel(self):
        net = DecisionNetwork.from_config()
        # Side channel is now ValueAiTypeSideChannel (shared per-slot MLP +
        # dedicated attention pool), not a flatten+Linear -- see
        # ai/models/value_side_channel.py. Its per-slot MLP input dim is
        # ai_type_dim + entity_embed_dim (detached entity embeddings are
        # concatenated in), and its output dim is hidden_dim regardless of
        # MAX_OTHER_PLAYERS (permutation-invariant pooling, not flattening).
        channel = net.value_ai_type_channel
        expected_in_dim = AI_TYPE_ONE_HOT_DIM + channel.entity_embed_dim
        assert channel.per_slot_mlp[0].in_features == expected_in_dim
        hidden_dim = channel.per_slot_mlp[0].out_features
        expected_value_in = net.trunk[-2].out_features + hidden_dim
        # value_head is a bare nn.Linear when config's value_hidden_dim=0, or
        # an nn.Sequential(Linear, ReLU, Linear) when >0 (see
        # DecisionNetwork.__init__ / ai_config.json["network"]["value_hidden_dim"]).
        # Grab whichever layer actually has in_features so this test doesn't
        # depend on which mode the current config uses.
        first_value_layer = (
            net.value_head[0] if isinstance(net.value_head, nn.Sequential) else net.value_head
        )
        assert first_value_layer.in_features == expected_value_in

    def test_forward_runs_without_ai_type_args(self):
        """Omitting self_ai_type/other_ai_type must default to all-zero, not error."""
        net = DecisionNetwork.from_config()
        obs = _make_obs(2)
        heads = net(
            obs["self_feat"], obs["other_feat"], obs["exists_mask"],
            obs["ball_feat"], obs["global_feat"],
        )
        assert torch.isfinite(heads.value).all()

    def test_execution_forward_runs_without_ai_type_args(self):
        d_net = DecisionNetwork.from_config()
        e_net = ExecutionNetwork.from_config()
        obs = _make_obs(2)
        d_heads = d_net(
            obs["self_feat"], obs["other_feat"], obs["exists_mask"],
            obs["ball_feat"], obs["global_feat"],
        )
        e_heads = e_net(
            obs["self_feat"], obs["other_feat"], obs["exists_mask"],
            obs["ball_feat"], obs["global_feat"], d_heads,
        )
        assert torch.isfinite(e_heads.value).all()


class TestEncoderPermutationConsistency:
    def test_other_ai_type_tracks_same_player_as_is_own_team_across_shuffles(self):
        """other_ai_type[slot] must always describe the same real player as
        other_feat[slot]'s is_own_team/pos_x, regardless of the random slot
        shuffle - i.e. it can never desync from the shuffle.
        """
        import random
        from footballcoach.ai.obs.encoder import encode_observation
        from footballcoach.ui.scenarios import build_1v1_scenario
        from footballcoach.rules_ai import Phase1RulesAI
        from footballcoach.ai.obs.schema import PlayerFeatures
        from dataclasses import fields

        match = build_1v1_scenario()
        trainee = match.player_by_id("trainee")
        opponent = match.player_by_id("opponent")
        opponent.ai = Phase1RulesAI()

        is_own_team_idx = [f.name for f in fields(PlayerFeatures)].index("is_own_team")

        # Run the encoder twice with different shuffle seeds; whichever slot
        # the opponent lands in, its ai_type one-hot must show is_rules=1.0
        for seed in (1, 2, 3, 4, 5):
            obs = encode_observation(
                match=match, player_id=trainee.player_id, time_remaining_s=100.0,
                rng=random.Random(seed),
            )
            real_slots = obs.exists_mask.nonzero()[0]
            assert len(real_slots) == 1  # 1v1 - exactly one other player
            slot = real_slots[0]
            # is_own_team should be 0.0 (opponent is not on trainee's team)
            assert obs.other_feat[slot, is_own_team_idx] == 0.0
            # ai_type one-hot at that same slot must show is_rules=1.0 (index 0)
            assert obs.other_ai_type[slot, 0] == 1.0
            assert obs.other_ai_type[slot, 1] == 0.0
            assert obs.other_ai_type[slot, 2] == 0.0
            # All other (padded) slots must stay all-zero
            for other_slot in range(MAX_OTHER_PLAYERS):
                if other_slot != slot:
                    assert (obs.other_ai_type[other_slot] == 0.0).all()

    def test_self_ai_type_reflects_self_player_ai(self):
        import random
        from footballcoach.ai.obs.encoder import encode_observation
        from footballcoach.ui.scenarios import build_1v1_scenario
        from footballcoach.rules_ai import Phase1RulesAI

        match = build_1v1_scenario()
        trainee = match.player_by_id("trainee")
        trainee.ai = Phase1RulesAI()

        obs = encode_observation(
            match=match, player_id=trainee.player_id, time_remaining_s=100.0,
            rng=random.Random(0),
        )
        assert obs.self_ai_type[0] == 1.0  # is_rules
        assert obs.self_ai_type[1] == 0.0
        assert obs.self_ai_type[2] == 0.0

    def test_immobile_player_ai_type_is_immobile(self):
        import random
        from footballcoach.ai.obs.encoder import encode_observation
        from footballcoach.ui.scenarios import build_1v1_scenario

        match = build_1v1_scenario()
        trainee = match.player_by_id("trainee")
        opponent = match.player_by_id("opponent")
        opponent.ai = None  # immobile

        obs = encode_observation(
            match=match, player_id=trainee.player_id, time_remaining_s=100.0,
            rng=random.Random(0),
        )
        real_slots = obs.exists_mask.nonzero()[0]
        slot = real_slots[0]
        assert obs.other_ai_type[slot, 1] == 1.0  # is_immobile


class TestAugmentationPermutation:
    def test_other_ai_type_permuted_alongside_other_feat_under_slot_shuffle(self):
        from footballcoach.ai.obs.augment import augment_obs_bc
        from footballcoach.ai.ppo.bc import BC_LABEL_DIM
        import random as pyrandom

        n = 2
        obs_dict = _make_obs(n)
        # Exactly one real "other" slot (index 0) per row - everything else padded.
        obs_dict["exists_mask"] = torch.zeros(n, MAX_OTHER_PLAYERS)
        obs_dict["exists_mask"][:, 0] = 1.0
        obs_dict["self_ai_type"] = torch.tensor([[1.0, 0.0, 0.0]] * n)
        other_ai_type = torch.zeros(n, MAX_OTHER_PLAYERS, AI_TYPE_ONE_HOT_DIM)
        other_ai_type[:, 0] = torch.tensor([0.0, 1.0, 0.0])  # immobile at slot 0
        obs_dict["other_ai_type"] = other_ai_type
        bc_labels = torch.zeros(n, BC_LABEL_DIM)
        bc_labels[:, -1] = 1.0  # valid

        aug_obs, _ = augment_obs_bc(
            obs_dict, bc_labels, n_slot_shuffles=3, rng=pyrandom.Random(0),
        )
        # For every augmented row, wherever exists_mask==1, other_ai_type at
        # that slot must be the immobile one-hot; every other slot all-zero.
        for i in range(aug_obs["exists_mask"].shape[0]):
            real_slot = int(aug_obs["exists_mask"][i].nonzero()[0, 0])
            assert torch.allclose(
                aug_obs["other_ai_type"][i, real_slot],
                torch.tensor([0.0, 1.0, 0.0]),
            )
            for s in range(MAX_OTHER_PLAYERS):
                if s != real_slot:
                    assert torch.allclose(
                        aug_obs["other_ai_type"][i, s], torch.zeros(AI_TYPE_ONE_HOT_DIM)
                    )
        # self_ai_type must be unchanged (pass-through, no geometric flip)
        assert torch.allclose(aug_obs["self_ai_type"], torch.tensor([[1.0, 0.0, 0.0]] * aug_obs["self_ai_type"].shape[0]))


class TestValueSideChannelPermutationInvariance:
    """Regression test for the bug where the ai-type side channel's old
    flatten+Linear implementation was NOT permutation-invariant (each slot
    position had its own weight block), unlike the main entity encoder.
    See ai/models/value_side_channel.py docstring.
    """

    def test_value_output_unchanged_when_real_player_moved_to_different_slot(self):
        torch.manual_seed(0)
        net = DecisionNetwork.from_config()
        net.eval()

        batch_size = 1
        self_feat = torch.randn(batch_size, PLAYER_FEATURE_DIM)
        ball_feat = torch.randn(batch_size, BALL_FEATURE_DIM)
        global_feat = torch.randn(batch_size, GLOBAL_FEATURE_DIM)
        self_ai_type = torch.zeros(batch_size, AI_TYPE_ONE_HOT_DIM)
        self_ai_type[:, 2] = 1.0  # self is neural

        real_player_feat = torch.randn(PLAYER_FEATURE_DIM)
        real_player_ai_type = torch.tensor([1.0, 0.0, 0.0])  # is_rules

        def _build(slot: int):
            other_feat = torch.zeros(batch_size, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM)
            exists_mask = torch.zeros(batch_size, MAX_OTHER_PLAYERS)
            other_ai_type = torch.zeros(batch_size, MAX_OTHER_PLAYERS, AI_TYPE_ONE_HOT_DIM)
            other_feat[0, slot] = real_player_feat
            exists_mask[0, slot] = 1.0
            other_ai_type[0, slot] = real_player_ai_type
            return other_feat, exists_mask, other_ai_type

        with torch.no_grad():
            other_feat_a, exists_mask_a, other_ai_type_a = _build(slot=0)
            heads_a = net(
                self_feat, other_feat_a, exists_mask_a, ball_feat, global_feat,
                self_ai_type, other_ai_type_a,
            )

            other_feat_b, exists_mask_b, other_ai_type_b = _build(slot=17)
            heads_b = net(
                self_feat, other_feat_b, exists_mask_b, ball_feat, global_feat,
                self_ai_type, other_ai_type_b,
            )

        # Moving the one real player from slot 0 to slot 17 (all else identical
        # padding) must not change the value estimate at all.
        assert torch.allclose(heads_a.value, heads_b.value, atol=1e-5)

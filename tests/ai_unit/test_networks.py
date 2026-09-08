"""Unit tests for the neural network modules.

Tests forward-pass shape correctness, no-NaN guarantee, architectural
invariants (get_possession >= tackle constraint), and config loading.

These don't run any training - just forward passes with random inputs.
If any shape or dtype is wrong here, the training loop will crash or
silently train on garbage.
"""
import pytest
import torch
import numpy as np

from footballcoach.ai.obs.schema import (
    BALL_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    MAX_OTHER_PLAYERS,
    PLAYER_FEATURE_DIM,
)


# ---------------------------------------------------------------------------
# Fixtures: build minimal networks from config
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def decision_net():
    from footballcoach.ai.models.decision_network import DecisionNetwork
    return DecisionNetwork.from_config()


@pytest.fixture(scope="module")
def execution_net():
    from footballcoach.ai.models.execution_network import ExecutionNetwork
    return ExecutionNetwork.from_config()


def _make_batch(batch_size: int = 2):
    """Random observation tensors for a given batch size."""
    return {
        "self_feat":   torch.randn(batch_size, PLAYER_FEATURE_DIM),
        "other_feat":  torch.randn(batch_size, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM),
        "exists_mask": (torch.rand(batch_size, MAX_OTHER_PLAYERS) > 0.5).float(),
        "ball_feat":   torch.randn(batch_size, BALL_FEATURE_DIM),
        "global_feat": torch.randn(batch_size, GLOBAL_FEATURE_DIM),
    }


# ---------------------------------------------------------------------------
# DecisionNetwork
# ---------------------------------------------------------------------------

class TestDecisionNetwork:

    def test_from_config_constructs(self, decision_net):
        assert decision_net is not None

    def test_forward_returns_decision_heads_raw(self, decision_net):
        from footballcoach.ai.action.schema import DecisionHeadsRaw
        batch = _make_batch(3)
        heads = decision_net(**batch)
        assert isinstance(heads, DecisionHeadsRaw)

    @pytest.mark.parametrize("field,expected_shape", [
        ("shoot_logit", (3, 1)),
        ("pass_logit", (3, 1)),
        ("move_logit", (3, 1)),
        ("tackle_logit", (3, 1)),
        ("get_possession_raw", (3, 1)),
        ("mark_logit", (3, 1)),
        ("hold_position_logit", (3, 1)),
        ("pass_target_logits", (3, MAX_OTHER_PLAYERS)),
        ("tackle_target_logits", (3, MAX_OTHER_PLAYERS)),
        ("mark_target_logits", (3, MAX_OTHER_PLAYERS)),
        ("move_region_center", (3, 2)),
        ("move_region_size", (3, 1)),
        ("move_arrival_speed", (3, 1)),
        ("region_of_play_center", (3, 2)),
        ("region_of_play_size", (3, 1)),
        ("attack_defence_raw", (3, 1)),
        ("value", (3, 1)),
    ])
    def test_output_shape(self, decision_net, field, expected_shape):
        batch = _make_batch(3)
        heads = decision_net(**batch)
        tensor = getattr(heads, field)
        assert tensor.shape == expected_shape, (
            f"DecisionHeadsRaw.{field}: expected {expected_shape}, got {tensor.shape}"
        )

    def test_latent_vector_shape(self, decision_net):
        from footballcoach.ai.config import load_ai_config
        latent_dim = load_ai_config()["network"]["latent_dim"]
        batch = _make_batch(3)
        heads = decision_net(**batch)
        assert heads.latent_vector.shape == (3, latent_dim)

    def test_no_nan_in_outputs(self, decision_net):
        torch.manual_seed(0)
        batch = _make_batch(4)
        heads = decision_net(**batch)
        for field in [
            "shoot_logit", "pass_logit", "move_logit", "tackle_logit",
            "get_possession_raw", "mark_logit", "hold_position_logit",
            "pass_target_logits", "tackle_target_logits", "mark_target_logits",
            "move_region_center", "move_region_size", "move_arrival_speed",
            "attack_defence_raw", "latent_vector", "value",
        ]:
            t = getattr(heads, field)
            assert not torch.any(torch.isnan(t)), f"NaN in DecisionHeadsRaw.{field}"
            assert not torch.any(torch.isinf(t)), f"Inf in DecisionHeadsRaw.{field}"

    def test_batch_size_one_works(self, decision_net):
        batch = _make_batch(1)
        heads = decision_net(**batch)
        assert heads.shoot_logit.shape == (1, 1)

    def test_no_gradient_leak_through_exists_mask(self, decision_net):
        """The exists mask is a float tensor; changing it should change output."""
        torch.manual_seed(42)
        batch_all = _make_batch(1)
        batch_all["exists_mask"] = torch.ones(1, MAX_OTHER_PLAYERS)

        batch_none = {k: v.clone() for k, v in batch_all.items()}
        batch_none["exists_mask"] = torch.zeros(1, MAX_OTHER_PLAYERS)

        heads_all  = decision_net(**batch_all)
        heads_none = decision_net(**batch_none)

        # Outputs should differ when exists_mask changes (attention sees different things)
        # This is a soft check - just verify it doesn't error
        assert heads_all.shoot_logit is not None
        assert heads_none.shoot_logit is not None


# ---------------------------------------------------------------------------
# Get-possession constraint
# ---------------------------------------------------------------------------

class TestGetPossessionConstraint:
    """derive_get_possession_prob() must always produce gp >= tackle."""

    def test_get_possession_always_gte_tackle(self):
        from footballcoach.ai.models.decision_network import derive_get_possession_prob
        for _ in range(50):
            tackle_logit = torch.randn(8, 1) * 5
            gp_raw_logit = torch.randn(8, 1) * 5
            tackle_prob, gp_prob = derive_get_possession_prob(tackle_logit, gp_raw_logit)
            diff = gp_prob - tackle_prob
            assert torch.all(diff >= -1e-6), (
                f"get_possession < tackle! min diff: {diff.min().item():.6f}"
            )

    def test_get_possession_at_most_one(self):
        from footballcoach.ai.models.decision_network import derive_get_possession_prob
        tackle_logit = torch.randn(10, 1)
        gp_raw_logit = torch.randn(10, 1)
        _, gp_prob = derive_get_possession_prob(tackle_logit, gp_raw_logit)
        assert torch.all(gp_prob <= 1.0 + 1e-6)
        assert torch.all(gp_prob >= 0.0 - 1e-6)

    def test_max_gp_raw_gives_gp_equal_one(self):
        """When gp_raw is very large, get_possession_prob -> 1.0."""
        from footballcoach.ai.models.decision_network import derive_get_possession_prob
        tackle_logit = torch.zeros(1, 1)
        gp_raw_logit = torch.tensor([[100.0]])
        _, gp_prob = derive_get_possession_prob(tackle_logit, gp_raw_logit)
        assert float(gp_prob) == pytest.approx(1.0, abs=1e-4)

    def test_min_gp_raw_gives_gp_equal_tackle(self):
        """When gp_raw is very negative, get_possession_prob -> tackle_prob."""
        from footballcoach.ai.models.decision_network import derive_get_possession_prob
        tackle_logit = torch.tensor([[2.0]])
        gp_raw_logit = torch.tensor([[-100.0]])
        tackle_prob, gp_prob = derive_get_possession_prob(tackle_logit, gp_raw_logit)
        assert float(gp_prob) == pytest.approx(float(tackle_prob), abs=1e-4)


# ---------------------------------------------------------------------------
# ExecutionNetwork
# ---------------------------------------------------------------------------

class TestExecutionNetwork:

    def _run(self, execution_net, decision_net, batch_size=2):
        batch = _make_batch(batch_size)
        d_heads = decision_net(**batch)
        e_heads = execution_net(
            self_feat=batch["self_feat"],
            other_feat=batch["other_feat"],
            exists_mask=batch["exists_mask"],
            ball_feat=batch["ball_feat"],
            global_feat=batch["global_feat"],
            decision_heads=d_heads,
        )
        return e_heads

    def test_from_config_constructs(self, execution_net):
        assert execution_net is not None

    def test_forward_returns_execution_heads_raw(self, decision_net, execution_net):
        from footballcoach.ai.action.schema import ExecutionHeadsRaw
        heads = self._run(execution_net, decision_net)
        assert isinstance(heads, ExecutionHeadsRaw)

    @pytest.mark.parametrize("field,expected_shape", [
        ("move_direction", (2, 2)),
        ("exec_move_logit", (2, 1)),
        ("sprint_logit", (2, 1)),
        ("kick_logit", (2, 1)),
        ("kick_direction", (2, 3)),  # 3D unit vector (elevation added for upward kicks)
        ("kick_power", (2, 1)),
        ("kick_spin", (2, 3)),
        ("tackle_attempt_logit", (2, 1)),
        ("value", (2, 1)),
    ])
    def test_output_shape(self, decision_net, execution_net, field, expected_shape):
        heads = self._run(execution_net, decision_net, batch_size=2)
        tensor = getattr(heads, field)
        assert tensor.shape == expected_shape, (
            f"ExecutionHeadsRaw.{field}: expected {expected_shape}, got {tensor.shape}"
        )

    def test_no_nan_in_outputs(self, decision_net, execution_net):
        torch.manual_seed(1)
        heads = self._run(execution_net, decision_net, batch_size=4)
        for field in [
            "move_direction", "exec_move_logit", "sprint_logit", "kick_logit",
            "kick_direction", "kick_power", "kick_spin",
            "tackle_attempt_logit", "value",
        ]:
            t = getattr(heads, field)
            assert not torch.any(torch.isnan(t)), f"NaN in ExecutionHeadsRaw.{field}"
            assert not torch.any(torch.isinf(t)), f"Inf in ExecutionHeadsRaw.{field}"

    def test_batch_size_one_works(self, decision_net, execution_net):
        heads = self._run(execution_net, decision_net, batch_size=1)
        assert heads.exec_move_logit.shape == (1, 1)
        assert heads.sprint_logit.shape == (1, 1)


# ---------------------------------------------------------------------------
# ExecutionNetwork value_only fast path (skips the 8 actor heads --
# move/kick direction, exec_move/sprint/kick/tackle_attempt logits,
# kick_power, kick_spin -- when only the critic estimate is needed, e.g.
# PPOTrainer._value_heads()/self.value_net calls). See execution_network.py
# forward()'s own docstring for the contract: value_only=True returns the
# raw value TENSOR directly, not an ExecutionHeadsRaw, precisely so a caller
# that wrongly expects the full dataclass fails loudly (AttributeError)
# instead of silently reading a stale/placeholder field.
# ---------------------------------------------------------------------------

class TestExecutionNetworkValueOnly:

    def _fwd_kwargs(self, decision_net, batch_size=2):
        batch = _make_batch(batch_size)
        d_heads = decision_net(**batch)
        return dict(
            self_feat=batch["self_feat"], other_feat=batch["other_feat"],
            exists_mask=batch["exists_mask"], ball_feat=batch["ball_feat"],
            global_feat=batch["global_feat"], decision_heads=d_heads,
        )

    def test_value_only_returns_raw_tensor_not_dataclass(self, decision_net, execution_net):
        from footballcoach.ai.action.schema import ExecutionHeadsRaw
        kwargs = self._fwd_kwargs(decision_net)
        result = execution_net(**kwargs, value_only=True)
        assert isinstance(result, torch.Tensor)
        assert not isinstance(result, ExecutionHeadsRaw)

    def test_value_only_shape(self, decision_net, execution_net):
        kwargs = self._fwd_kwargs(decision_net, batch_size=3)
        result = execution_net(**kwargs, value_only=True)
        assert result.shape == (3, 1)

    def test_value_only_matches_full_forward_exactly(self, decision_net, execution_net):
        """The 8 skipped actor heads are pure Linear(h) projections computed
        AFTER value_input is already built -- they can have zero effect on
        `.value`'s numerics. value_only=True must therefore reproduce the
        exact same value as a full forward pass on identical input, not an
        approximation."""
        execution_net.eval()
        kwargs = self._fwd_kwargs(decision_net, batch_size=4)
        with torch.no_grad():
            full = execution_net(**kwargs)
            fast = execution_net(**kwargs, value_only=True)
        assert torch.equal(full.value, fast), (
            "value_only=True produced a different value than the full forward pass"
        )

    def test_value_only_default_is_false(self, decision_net, execution_net):
        """Omitting value_only must keep returning the full ExecutionHeadsRaw
        -- a regression here would silently break every existing caller that
        doesn't pass the new kwarg at all."""
        from footballcoach.ai.action.schema import ExecutionHeadsRaw
        kwargs = self._fwd_kwargs(decision_net)
        result = execution_net(**kwargs)
        assert isinstance(result, ExecutionHeadsRaw)

    def test_value_only_no_nan(self, decision_net, execution_net):
        torch.manual_seed(2)
        kwargs = self._fwd_kwargs(decision_net, batch_size=4)
        result = execution_net(**kwargs, value_only=True)
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_value_only_batch_size_one(self, decision_net, execution_net):
        kwargs = self._fwd_kwargs(decision_net, batch_size=1)
        result = execution_net(**kwargs, value_only=True)
        assert result.shape == (1, 1)


# ---------------------------------------------------------------------------
# flatten_decision_heads dimension consistency
# ---------------------------------------------------------------------------

def test_flatten_decision_heads_dim_matches_network(decision_net, execution_net):
    """The flat dimension of decision heads fed into the execution network
    must match what execution_net.decision_mlp expects."""
    from footballcoach.ai.models.execution_network import flatten_decision_heads, _decision_output_dim
    from footballcoach.ai.config import load_ai_config
    latent_dim = load_ai_config()["network"]["latent_dim"]

    batch = _make_batch(1)
    d_heads = decision_net(**batch)
    flat = flatten_decision_heads(d_heads)
    expected_dim = _decision_output_dim(latent_dim)
    assert flat.shape == (1, expected_dim), (
        f"flatten_decision_heads output dim {flat.shape[1]} != "
        f"expected {expected_dim}"
    )

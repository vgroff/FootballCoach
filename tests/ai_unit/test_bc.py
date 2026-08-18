"""Baseline regression tests for src/footballcoach/ai/ppo/bc.py.

These tests capture *current* behaviour of BCLabel packing, bc_loss_from_tensor,
and phase1_labels() before any of the class-balancing / demo-recording
workstreams change them.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from footballcoach.ai.action.schema import DecisionHeadsRaw, ExecutionHeadsRaw
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.bc import (
    BC_LABEL_DIM,
    BCLabel,
    _I_DIR_X,
    _I_DIR_Y,
    _I_EXEC_MOVE,
    _I_GP_EXTRA,
    _I_HOLD,
    _I_KICK_THIS_TICK,
    _I_MARK,
    _I_MOVE,
    _I_PASS,
    _I_REGION_X,
    _I_REGION_Y,
    _I_SHOOT,
    _I_SPRINT,
    _I_TACKLE,
    _I_TACKLE_ATTEMPT,
    _I_VALID,
    _I_KICK_DIR_X,
    _I_KICK_DIR_Y,
    _I_KICK_POWER,
    _I_KICK_SPIN_X,
    _I_KICK_SPIN_Y,
    _I_KICK_SPIN_Z,
    bc_loss_from_tensor,
    phase1_labels,
)
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario


# ---------------------------------------------------------------------------
# BCLabel.to_array() / BCLabel.invalid()
# ---------------------------------------------------------------------------

class TestBCLabelArray:
    def test_to_array_round_trip(self):
        direction = np.array([0.6, 0.8], dtype=np.float32)
        region = np.array([10.0, -5.0], dtype=np.float32)
        label = BCLabel(
            shoot=1.0,
            pass_=0.0,
            move=1.0,
            tackle=0.0,
            get_possession_extra=1.0,
            mark=0.0,
            hold_position=0.0,
            move_direction=direction,
            sprint=1.0,
            move_region_center_m=region,
            kick_this_tick=1.0,
            tackle_attempt=0.0,
            exec_move=1.0,
            valid=True,
        )
        arr = label.to_array()
        assert arr.shape == (BC_LABEL_DIM,)
        assert arr.dtype == np.float32
        assert arr[_I_SHOOT] == pytest.approx(1.0)
        assert arr[_I_PASS] == pytest.approx(0.0)
        assert arr[_I_MOVE] == pytest.approx(1.0)
        assert arr[_I_TACKLE] == pytest.approx(0.0)
        assert arr[_I_GP_EXTRA] == pytest.approx(1.0)
        assert arr[_I_MARK] == pytest.approx(0.0)
        assert arr[_I_HOLD] == pytest.approx(0.0)
        assert arr[_I_DIR_X] == pytest.approx(0.6)
        assert arr[_I_DIR_Y] == pytest.approx(0.8)
        assert arr[_I_SPRINT] == pytest.approx(1.0)
        assert arr[_I_REGION_X] == pytest.approx(10.0)
        assert arr[_I_REGION_Y] == pytest.approx(-5.0)
        assert arr[_I_KICK_THIS_TICK] == pytest.approx(1.0)
        assert arr[_I_TACKLE_ATTEMPT] == pytest.approx(0.0)
        assert arr[_I_VALID] == pytest.approx(1.0)
        assert arr[_I_EXEC_MOVE] == pytest.approx(1.0)

    def test_to_array_none_direction_and_region_are_zero(self):
        label = BCLabel(move_direction=None, move_region_center_m=None)
        arr = label.to_array()
        assert arr[_I_DIR_X] == pytest.approx(0.0)
        assert arr[_I_DIR_Y] == pytest.approx(0.0)
        assert arr[_I_REGION_X] == pytest.approx(0.0)
        assert arr[_I_REGION_Y] == pytest.approx(0.0)

    def test_invalid_label_has_valid_flag_zero(self):
        label = BCLabel.invalid()
        arr = label.to_array()
        assert arr[_I_VALID] == pytest.approx(0.0)
        assert label.valid is False

    def test_to_array_kick_fields_round_trip(self):
        kick_direction = np.array([0.0, 1.0], dtype=np.float32)
        kick_spin = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        label = BCLabel(
            kick_this_tick=1.0,
            kick_direction=kick_direction,
            kick_power_fraction=0.75,
            kick_spin=kick_spin,
            valid=True,
        )
        arr = label.to_array()
        assert arr[_I_KICK_DIR_X] == pytest.approx(0.0)
        assert arr[_I_KICK_DIR_Y] == pytest.approx(1.0)
        assert arr[_I_KICK_POWER] == pytest.approx(0.75)
        assert arr[_I_KICK_SPIN_X] == pytest.approx(1.0)
        assert arr[_I_KICK_SPIN_Y] == pytest.approx(2.0)
        assert arr[_I_KICK_SPIN_Z] == pytest.approx(3.0)

    def test_to_array_kick_fields_default_zero(self):
        label = BCLabel()
        arr = label.to_array()
        assert arr[_I_KICK_DIR_X] == pytest.approx(0.0)
        assert arr[_I_KICK_DIR_Y] == pytest.approx(0.0)
        assert arr[_I_KICK_POWER] == pytest.approx(0.0)
        assert arr[_I_KICK_SPIN_X] == pytest.approx(0.0)
        assert arr[_I_KICK_SPIN_Y] == pytest.approx(0.0)
        assert arr[_I_KICK_SPIN_Z] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# bc_loss_from_tensor
# ---------------------------------------------------------------------------

def _zeros_heads(n: int) -> tuple[DecisionHeadsRaw, ExecutionHeadsRaw]:
    """Build DecisionHeadsRaw/ExecutionHeadsRaw with all-zero logits (sigmoid=0.5)."""
    z1 = torch.zeros(n, 1)
    z2 = torch.zeros(n, 2)
    z_slots = torch.zeros(n, 21)
    d_heads = DecisionHeadsRaw(
        shoot_logit=z1.clone(),
        pass_logit=z1.clone(),
        move_logit=z1.clone(),
        tackle_logit=z1.clone(),
        get_possession_raw=z1.clone(),
        mark_logit=z1.clone(),
        hold_position_logit=z1.clone(),
        pass_target_logits=z_slots.clone(),
        tackle_target_logits=z_slots.clone(),
        mark_target_logits=z_slots.clone(),
        move_region_center=z2.clone(),
        move_region_size=z1.clone(),
        move_arrival_speed=z1.clone(),
        region_of_play_center=z2.clone(),
        region_of_play_size=z1.clone(),
        attack_defence_raw=z1.clone(),
        latent_vector=torch.zeros(n, 32),
        value=z1.clone(),
    )
    e_heads = ExecutionHeadsRaw(
        move_direction=z2.clone(),
        exec_move_logit=z1.clone(),
        sprint_logit=z1.clone(),
        kick_logit=z1.clone(),
        kick_direction=z2.clone(),
        kick_power=z1.clone(),
        kick_spin=torch.zeros(n, 3),
        tackle_attempt_logit=z1.clone(),
        value=z1.clone(),
    )
    return d_heads, e_heads


def _make_labels(rows: list[BCLabel]) -> torch.Tensor:
    return torch.from_numpy(np.stack([r.to_array() for r in rows]))


class TestBCLossFromTensor:
    def test_all_zero_logits_all_valid_rows_gives_bce_ln2(self):
        """All logits 0 -> sigmoid 0.5 -> BCE = -ln(0.5) = ln(2) per Bernoulli
        target regardless of target value.  With no direction/region targets
        set, only the Bernoulli BCE terms should contribute."""
        n = 4
        labels = _make_labels([BCLabel(valid=True) for _ in range(n)])
        d_heads, e_heads = _zeros_heads(n)
        loss = bc_loss_from_tensor(labels, d_heads, e_heads)
        # 7 decision Bernoullis + 4 exec Bernoullis (move, sprint, kick, tackle_attempt)
        # each contribute ln(2) -> 11 * ln(2) per row, direction/region are 0
        # since has_dir/has_region are False for the all-default label.
        expected = 11 * math.log(2.0)
        assert loss.item() == pytest.approx(expected, abs=1e-4)

    def test_invalid_row_excluded_from_mean(self):
        """A batch with one invalid garbage row should give the exact same
        loss as the same batch with that row removed."""
        n = 4
        valid_rows = [BCLabel(shoot=1.0, valid=True) for _ in range(n)]
        garbage = BCLabel(shoot=1.0, pass_=1.0, move=1.0, tackle=1.0, valid=False)

        labels_with_invalid = _make_labels(valid_rows + [garbage])
        labels_without = _make_labels(valid_rows)

        d1, e1 = _zeros_heads(n + 1)
        d2, e2 = _zeros_heads(n)

        loss_with = bc_loss_from_tensor(labels_with_invalid, d1, e1)
        loss_without = bc_loss_from_tensor(labels_without, d2, e2)
        assert loss_with.item() == pytest.approx(loss_without.item(), abs=1e-5)

    def test_direction_loss_zero_when_aligned(self):
        n = 1
        direction = np.array([1.0, 0.0], dtype=np.float32)
        labels = _make_labels([BCLabel(move_direction=direction, valid=True)])
        d_heads, e_heads = _zeros_heads(n)
        # Predicted direction perfectly aligned with target.
        e_heads.move_direction = torch.tensor([[1.0, 0.0]])
        _, breakdown = bc_loss_from_tensor(labels, d_heads, e_heads, return_breakdown=True)
        assert breakdown["direction"] == pytest.approx(0.0, abs=1e-4)

    def test_direction_loss_max_when_opposite(self):
        n = 1
        direction = np.array([1.0, 0.0], dtype=np.float32)
        labels = _make_labels([BCLabel(move_direction=direction, valid=True)])
        d_heads, e_heads = _zeros_heads(n)
        e_heads.move_direction = torch.tensor([[-1.0, 0.0]])
        direction_loss_weight = 3.0
        _, breakdown = bc_loss_from_tensor(
            labels, d_heads, e_heads, direction_loss_weight=direction_loss_weight,
            return_breakdown=True,
        )
        assert breakdown["direction"] == pytest.approx(direction_loss_weight * 2.0, abs=1e-4)

    def test_region_loss_zero_for_perfect_match(self):
        n = 1
        region = np.array([0.0, 0.0], dtype=np.float32)  # tanh(0)=0 matches target 0
        labels = _make_labels([BCLabel(move_region_center_m=region, valid=True)])
        d_heads, e_heads = _zeros_heads(n)
        # move_region_center raw logit 0 -> tanh(0) = 0, matches target norm 0.
        _, breakdown = bc_loss_from_tensor(labels, d_heads, e_heads, return_breakdown=True)
        assert breakdown["region"] == pytest.approx(0.0, abs=1e-4)

    def test_region_loss_nonzero_scales_with_weight(self):
        n = 1
        region = np.array([52.5, 0.0], dtype=np.float32)  # normalized target = 1.0
        labels = _make_labels([BCLabel(move_region_center_m=region, valid=True)])
        d_heads, e_heads = _zeros_heads(n)  # predicted tanh(0) = 0 -> mse = 1.0
        loss_w1, bd1 = bc_loss_from_tensor(
            labels, d_heads, e_heads, region_loss_weight=1.0, return_breakdown=True
        )
        loss_w2, bd2 = bc_loss_from_tensor(
            labels, d_heads, e_heads, region_loss_weight=2.0, return_breakdown=True
        )
        assert bd1["region"] > 0.0
        assert bd2["region"] == pytest.approx(2.0 * bd1["region"], rel=1e-4)

    def test_no_valid_rows_returns_zero(self):
        n = 3
        labels = _make_labels([BCLabel.invalid() for _ in range(n)])
        d_heads, e_heads = _zeros_heads(n)
        loss = bc_loss_from_tensor(labels, d_heads, e_heads)
        assert loss.item() == pytest.approx(0.0, abs=1e-8)

    def test_kick_direction_loss_zero_when_aligned(self):
        n = 1
        kick_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        labels = _make_labels([BCLabel(kick_this_tick=1.0, kick_direction=kick_direction, valid=True)])
        d_heads, e_heads = _zeros_heads(n)
        e_heads.kick_direction = torch.tensor([[1.0, 0.0, 0.0]])
        _, breakdown = bc_loss_from_tensor(labels, d_heads, e_heads, return_breakdown=True)
        assert breakdown["kick_direction"] == pytest.approx(0.0, abs=1e-4)

    def test_kick_direction_loss_max_when_opposite(self):
        n = 1
        kick_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        labels = _make_labels([BCLabel(kick_this_tick=1.0, kick_direction=kick_direction, valid=True)])
        d_heads, e_heads = _zeros_heads(n)
        e_heads.kick_direction = torch.tensor([[-1.0, 0.0, 0.0]])
        direction_loss_weight = 3.0
        _, breakdown = bc_loss_from_tensor(
            labels, d_heads, e_heads, direction_loss_weight=direction_loss_weight,
            return_breakdown=True,
        )
        assert breakdown["kick_direction"] == pytest.approx(direction_loss_weight * 2.0, abs=1e-4)

    def test_kick_direction_loss_gated_on_kick_this_tick(self):
        """kick_direction present but kick_this_tick=0 -> no loss contribution."""
        n = 1
        kick_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        labels = _make_labels([BCLabel(kick_this_tick=0.0, kick_direction=kick_direction, valid=True)])
        d_heads, e_heads = _zeros_heads(n)
        e_heads.kick_direction = torch.tensor([[-1.0, 0.0, 0.0]])
        _, breakdown = bc_loss_from_tensor(labels, d_heads, e_heads, return_breakdown=True)
        assert breakdown["kick_direction"] == pytest.approx(0.0, abs=1e-4)

    def test_kick_power_loss_matches_mse(self):
        n = 1
        labels = _make_labels([BCLabel(kick_this_tick=1.0, kick_power_fraction=0.8, valid=True)])
        d_heads, e_heads = _zeros_heads(n)  # sigmoid(0) = 0.5
        _, breakdown = bc_loss_from_tensor(labels, d_heads, e_heads, return_breakdown=True)
        assert breakdown["kick_power"] == pytest.approx((0.5 - 0.8) ** 2, abs=1e-4)

    def test_kick_power_loss_gated_on_kick_this_tick(self):
        n = 1
        labels = _make_labels([BCLabel(kick_this_tick=0.0, kick_power_fraction=0.8, valid=True)])
        d_heads, e_heads = _zeros_heads(n)
        _, breakdown = bc_loss_from_tensor(labels, d_heads, e_heads, return_breakdown=True)
        assert breakdown["kick_power"] == pytest.approx(0.0, abs=1e-4)

    def test_kick_spin_loss_zero_for_matching_spin(self):
        n = 1
        kick_spin = np.array([5.0, -5.0, 0.0], dtype=np.float32)
        labels = _make_labels([BCLabel(kick_this_tick=1.0, kick_spin=kick_spin, valid=True)])
        d_heads, e_heads = _zeros_heads(n)
        e_heads.kick_spin = torch.tensor([[5.0, -5.0, 0.0]])
        _, breakdown = bc_loss_from_tensor(labels, d_heads, e_heads, return_breakdown=True)
        assert breakdown["kick_spin"] == pytest.approx(0.0, abs=1e-4)

    def test_kick_breakdown_key_present_in_zero_dict(self):
        n = 3
        labels = _make_labels([BCLabel.invalid() for _ in range(n)])
        d_heads, e_heads = _zeros_heads(n)
        _, breakdown = bc_loss_from_tensor(labels, d_heads, e_heads, return_breakdown=True)
        assert breakdown["kick"] == pytest.approx(0.0)
        assert breakdown["kick_direction"] == pytest.approx(0.0)
        assert breakdown["kick_power"] == pytest.approx(0.0)
        assert breakdown["kick_spin"] == pytest.approx(0.0)


class TestBCLossFromTensorSplitKick:
    """split_kick=True: kick_group_loss + other_group_loss must reconstruct
    the combined `total`, and only kick_group_loss should react to kick-only
    prediction errors (gradient isolation between the two groups)."""

    def test_split_sums_to_total_all_valid(self):
        n = 4
        labels = _make_labels([
            BCLabel(shoot=1.0, kick_this_tick=1.0, kick_power_fraction=0.8, valid=True)
            for _ in range(n)
        ])
        d_heads, e_heads = _zeros_heads(n)
        total, breakdown, split = bc_loss_from_tensor(
            labels, d_heads, e_heads, return_breakdown=True, split_kick=True,
        )
        recombined = split["kick_group_loss"] + split["other_group_loss"]
        assert recombined.item() == pytest.approx(total.item(), abs=1e-5)

    def test_kick_only_error_isolated_to_kick_group(self):
        """A kick_power prediction error should move kick_group_loss but
        leave other_group_loss untouched (row has no other label deviation)."""
        n = 1
        labels = _make_labels([BCLabel(kick_this_tick=1.0, kick_power_fraction=0.8, valid=True)])
        d_heads, e_heads = _zeros_heads(n)
        _, _, split_baseline = bc_loss_from_tensor(
            labels, d_heads, e_heads, return_breakdown=True, split_kick=True,
        )
        e_heads.kick_power = torch.tensor([[-5.0]])  # sigmoid(-5)=0.007, far from target 0.8
        _, _, split_shifted = bc_loss_from_tensor(
            labels, d_heads, e_heads, return_breakdown=True, split_kick=True,
        )
        assert split_shifted["kick_group_loss"].item() > split_baseline["kick_group_loss"].item()
        assert split_shifted["other_group_loss"].item() == pytest.approx(
            split_baseline["other_group_loss"].item(), abs=1e-5
        )

    def test_split_kick_gradient_flows_only_to_kick_heads(self):
        """kick_group_loss.backward() must produce a nonzero gradient on
        kick_power but leave shoot_logit's gradient at None/zero -- proving
        the split tensors are still attached to autograd (unlike
        return_breakdown's detached floats)."""
        n = 1
        labels = _make_labels([BCLabel(shoot=1.0, kick_this_tick=1.0, kick_power_fraction=0.8, valid=True)])
        d_heads, e_heads = _zeros_heads(n)
        d_heads.shoot_logit.requires_grad_(True)
        e_heads.kick_power.requires_grad_(True)
        _, _, split = bc_loss_from_tensor(
            labels, d_heads, e_heads, return_breakdown=True, split_kick=True,
        )
        split["kick_group_loss"].backward()
        assert e_heads.kick_power.grad is not None
        assert e_heads.kick_power.grad.abs().item() > 0.0
        assert d_heads.shoot_logit.grad is None or d_heads.shoot_logit.grad.abs().item() == 0.0

    def test_split_kick_backward_compatible_without_return_breakdown(self):
        n = 2
        labels = _make_labels([BCLabel(kick_this_tick=1.0, valid=True) for _ in range(n)])
        d_heads, e_heads = _zeros_heads(n)
        total, split = bc_loss_from_tensor(labels, d_heads, e_heads, split_kick=True)
        assert set(split.keys()) == {"kick_group_loss", "other_group_loss"}
        assert isinstance(total, torch.Tensor)


class TestBCLossFromTensorSplitTackle:
    """split_tackle=True: tackle_group_loss + other_group_loss must
    reconstruct `total`, and only tackle_group_loss should react to
    tackle_attempt-only prediction errors (gradient isolation)."""

    def test_split_sums_to_total_all_valid(self):
        n = 4
        labels = _make_labels([
            BCLabel(shoot=1.0, tackle_attempt=1.0, valid=True) for _ in range(n)
        ])
        d_heads, e_heads = _zeros_heads(n)
        total, breakdown, split = bc_loss_from_tensor(
            labels, d_heads, e_heads, return_breakdown=True, split_tackle=True,
        )
        recombined = split["tackle_group_loss"] + split["other_group_loss"]
        assert recombined.item() == pytest.approx(total.item(), abs=1e-5)

    def test_tackle_only_error_isolated_to_tackle_group(self):
        n = 1
        labels = _make_labels([BCLabel(tackle_attempt=1.0, valid=True)])
        d_heads, e_heads = _zeros_heads(n)
        _, _, split_baseline = bc_loss_from_tensor(
            labels, d_heads, e_heads, return_breakdown=True, split_tackle=True,
        )
        e_heads.tackle_attempt_logit = torch.tensor([[-5.0]])  # sigmoid(-5) far from target 1.0
        _, _, split_shifted = bc_loss_from_tensor(
            labels, d_heads, e_heads, return_breakdown=True, split_tackle=True,
        )
        assert split_shifted["tackle_group_loss"].item() > split_baseline["tackle_group_loss"].item()
        assert split_shifted["other_group_loss"].item() == pytest.approx(
            split_baseline["other_group_loss"].item(), abs=1e-5
        )

    def test_split_tackle_gradient_flows_only_to_tackle_head(self):
        n = 1
        labels = _make_labels([BCLabel(shoot=1.0, tackle_attempt=1.0, valid=True)])
        d_heads, e_heads = _zeros_heads(n)
        d_heads.shoot_logit.requires_grad_(True)
        e_heads.tackle_attempt_logit.requires_grad_(True)
        _, _, split = bc_loss_from_tensor(
            labels, d_heads, e_heads, return_breakdown=True, split_tackle=True,
        )
        split["tackle_group_loss"].backward()
        assert e_heads.tackle_attempt_logit.grad is not None
        assert e_heads.tackle_attempt_logit.grad.abs().item() > 0.0
        assert d_heads.shoot_logit.grad is None or d_heads.shoot_logit.grad.abs().item() == 0.0

    def test_split_tackle_backward_compatible_without_return_breakdown(self):
        n = 2
        labels = _make_labels([BCLabel(tackle_attempt=1.0, valid=True) for _ in range(n)])
        d_heads, e_heads = _zeros_heads(n)
        total, split = bc_loss_from_tensor(labels, d_heads, e_heads, split_tackle=True)
        assert set(split.keys()) == {"tackle_group_loss", "other_group_loss"}
        assert isinstance(total, torch.Tensor)

    def test_split_kick_and_split_tackle_together(self):
        """When both flags are set, kick_group_loss + tackle_group_loss +
        other_group_loss must reconstruct total, and other_group_loss must
        exclude BOTH groups (regression check for the shared other_group_per_row
        subtraction in bc_loss_from_tensor)."""
        n = 1
        labels = _make_labels([BCLabel(
            shoot=1.0, kick_this_tick=1.0, kick_power_fraction=0.8,
            tackle_attempt=1.0, valid=True,
        )])
        d_heads, e_heads = _zeros_heads(n)
        total, breakdown, split = bc_loss_from_tensor(
            labels, d_heads, e_heads, return_breakdown=True,
            split_kick=True, split_tackle=True,
        )
        assert set(split.keys()) == {"kick_group_loss", "tackle_group_loss", "other_group_loss"}
        recombined = split["kick_group_loss"] + split["tackle_group_loss"] + split["other_group_loss"]
        assert recombined.item() == pytest.approx(total.item(), abs=1e-5)


class TestBCLossFromTensorDecisionOnly:
    """exec_heads=None (W7 Phase 0 path): loss must equal the decision-only
    subset of the full-signature call, with all exec-dependent terms
    (exec_move/sprint/kick/tackle_attempt BCE, move_direction cosine) zeroed."""

    def test_matches_full_signature_with_exec_terms_zeroed(self):
        n = 4
        direction = np.array([1.0, 0.0], dtype=np.float32)
        rows = [BCLabel(shoot=1.0, move_direction=direction, valid=True) for _ in range(n)]
        labels = _make_labels(rows)
        d_heads, e_heads = _zeros_heads(n)
        e_heads.move_direction = torch.tensor([[1.0, 0.0]] * n)

        loss_full, bkdn_full = bc_loss_from_tensor(labels, d_heads, e_heads, return_breakdown=True)
        loss_dec, bkdn_dec = bc_loss_from_tensor(labels, d_heads, exec_heads=None, return_breakdown=True)

        # Decision-only loss should equal full loss minus the exec-dependent terms.
        expected_dec_only = loss_full.item() - bkdn_full["exec_bce"] - bkdn_full["direction"]
        assert loss_dec.item() == pytest.approx(expected_dec_only, abs=1e-4)
        # Decision term itself is unaffected by exec_heads being absent.
        assert bkdn_dec["decision"] == pytest.approx(bkdn_full["decision"], abs=1e-5)
        # All exec-dependent breakdown entries are exactly zero.
        assert bkdn_dec["exec_bce"] == pytest.approx(0.0, abs=1e-8)
        assert bkdn_dec["sprint"] == pytest.approx(0.0, abs=1e-8)
        assert bkdn_dec["move"] == pytest.approx(0.0, abs=1e-8)
        assert bkdn_dec["tackle_attempt"] == pytest.approx(0.0, abs=1e-8)
        assert bkdn_dec["direction"] == pytest.approx(0.0, abs=1e-8)

    def test_no_valid_rows_returns_zero_without_exec_heads(self):
        n = 3
        labels = _make_labels([BCLabel.invalid() for _ in range(n)])
        d_heads, _ = _zeros_heads(n)
        loss = bc_loss_from_tensor(labels, d_heads, exec_heads=None)
        assert loss.item() == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# phase1_labels
# ---------------------------------------------------------------------------

def _make_env(max_episode_s: float = 30.0) -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="test_bc_1v1",
        label="Test: 1v1",
        description="1v1 scenario for BC label tests",
        build=build_1v1_scenario,
    )
    return ScenarioEnv(
        definition=defn,
        trainee_player_id="trainee",
        phase=1,
        max_episode_s=max_episode_s,
    )


class TestPhase1Labels:
    def test_move_label_when_chasing_ball_far_away(self):
        """When the trainee has no possession and is far from the ball,
        Phase1RulesAI issues a GetPossessionOrder (not MoveOrder) since it
        always chases the ball when it doesn't have possession -- verify
        get_possession_extra fires and a direction is produced.

        NOTE: move_direction is derived by actually running the decided
        order's execute() and reading back player.desired_direction (see
        ai/knowledge.md "Orders vs execution-network labels boundary") --
        it reflects real movement logic (braking/repulsion/turn-limiting),
        NOT a hand-derived straight-line vector toward the ball. So we only
        assert it's a valid unit vector here, not that it exactly matches
        raw target geometry."""
        env = _make_env()
        env.reset()
        match = env.match
        player = match.player_by_id(env.trainee_player_id)
        # Ensure trainee doesn't have the ball.
        match.ball.possessed_by = None

        label = phase1_labels(env, env.trainee_player_id)
        assert label.valid
        assert label.get_possession_extra == pytest.approx(1.0)
        if label.move_direction is not None:
            norm = float(np.linalg.norm(label.move_direction))
            assert norm == pytest.approx(1.0, abs=1e-3)

    def test_move_label_when_carrying_ball(self):
        """When the trainee has possession, Phase1RulesAI issues a MoveOrder
        toward the opponent box -- verify move==1.0 and direction points at
        the MoveOrder's target."""
        env = _make_env()
        env.reset()
        match = env.match
        player = match.player_by_id(env.trainee_player_id)
        match.ball.possessed_by = player.player_id

        label = phase1_labels(env, env.trainee_player_id)
        assert label.valid
        assert label.move == pytest.approx(1.0)
        assert label.move_direction is not None

    def test_valid_when_player_already_at_target(self):
        """Degenerate zero-length raw target vector (player exactly at the
        ball's position while chasing) must NOT discard the whole frame.

        Prior (buggy) behaviour hand-derived move_direction from
        `target - position` and returned BCLabel.invalid() whenever that
        vector was ~zero-length, silently dropping kick/tackle supervision
        on exactly the ticks where a push-kick/tackle is most likely to have
        just fired. The fix runs the decided order's execute() and reads
        back player.desired_direction/desired_speed_mode instead, so the
        label stays valid even when the raw target vector is degenerate --
        move_direction may legitimately be None if execute() resolves to a
        STANDSTILL/no-op for this tick, but the frame itself is not
        discarded. See ai/knowledge.md "Orders vs execution-network labels
        boundary"."""
        env = _make_env()
        env.reset()
        match = env.match
        player = match.player_by_id(env.trainee_player_id)
        match.ball.possessed_by = None
        # Move the ball exactly onto the player so the raw target vector is zero.
        from footballcoach.mathutils import Vector3
        match.ball.position = Vector3(player.position.x, player.position.y, 0.0)

        label = phase1_labels(env, env.trainee_player_id)
        assert label.valid
        assert label.get_possession_extra == pytest.approx(1.0)

    def test_invalid_for_missing_match(self):
        """If env.match is None, phase1_labels must return an invalid label
        rather than raising."""

        class _FakeEnv:
            match = None
            trainee_player_id = "trainee"

        label = phase1_labels(_FakeEnv(), "trainee")
        assert not label.valid

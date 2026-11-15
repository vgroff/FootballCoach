"""Unit tests for the pieces _ppo_update is built from (ppo_update.py and
ppo_update_diagnostics.py), each against a hand-written reference.

End-to-end equivalence of the whole update with its pre-restructuring version
is checked by scripts/ppo_update_trace_snapshot.py, not here.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import footballcoach.ai.ppo.ppo_update_diagnostics as diag
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
from footballcoach.ai.ppo.ppo_update import (
    _UpdateStats, _normalised_sample_weights, _obs_parts, _ppo_value_loss,
)
from footballcoach.ai.ppo.rollout_buffer import HEAD_LP_KEYS
from footballcoach.ai.ppo.trainer_common import value_mse_by_outcome


@pytest.fixture(scope="module")
def trainer() -> PPOTrainer:
    return PPOTrainer.from_config(device=torch.device("cpu"))


@pytest.fixture
def dual_clip(trainer):
    saved = trainer.dual_clip_c
    yield lambda c: setattr(trainer, "dual_clip_c", c)
    trainer.dual_clip_c = saved


# ----- policy loss -----------------------------------------------------------

def _reference_policy_loss(new_lp, old_lp, adv, w, clip, dual_c):
    out, surr = [], []
    for nl, ol, a, wi in zip(new_lp.tolist(), old_lp.tolist(), adv.tolist(), w.tolist()):
        r = math.exp(nl - ol)
        m = min(r * a, min(max(r, 1 - clip), 1 + clip) * a)
        if dual_c > 0 and a < 0:
            m = max(m, dual_c * a)
        out.append(-m * wi)
        surr.append((r * a - a) * wi)
    return out, surr


@pytest.mark.parametrize("dual_c", [0.0, 3.0])
def test_policy_loss_matches_the_clipped_surrogate(trainer, dual_clip, dual_c):
    dual_clip(dual_c)
    old = torch.zeros(6)
    # ratios: inside the band, above it, below it, and a huge one with a negative advantage
    new = torch.log(torch.tensor([1.05, 1.6, 0.5, 1.6, 0.5, 40.0]))
    adv = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -2.0])
    w = torch.tensor([1.0, 0.5, 2.0, 1.0, 1.5, 1.0])
    loss, per_sample, surrogate = trainer._ppo_policy_loss(new, old, adv, w, clip=0.2)
    ref, ref_surr = _reference_policy_loss(new, old, adv, w, 0.2, dual_c)
    assert per_sample.tolist() == pytest.approx(ref, rel=1e-6)
    assert float(loss) == pytest.approx(float(np.mean(ref)), rel=1e-6)
    assert surrogate.tolist() == pytest.approx(ref_surr, rel=1e-6)


def test_dual_clip_floor_actually_engages(trainer, dual_clip):
    # The 40x ratio row with A=-2: plain clip gives -min(-80, -2.4) = 80; the
    # dual-clip floor 3*A = -6 caps it at 6.
    new, old = torch.log(torch.tensor([40.0])), torch.zeros(1)
    adv, w = torch.tensor([-2.0]), torch.ones(1)
    dual_clip(0.0)
    assert float(trainer._ppo_policy_loss(new, old, adv, w, 0.2)[0]) == pytest.approx(80.0)
    dual_clip(3.0)
    assert float(trainer._ppo_policy_loss(new, old, adv, w, 0.2)[0]) == pytest.approx(6.0)


# ----- per-head counterfactual policy loss --------------------------------------

def test_head_counterfactual_is_per_head_surrogate_averaged_over_rows(trainer, dual_clip):
    dual_clip(0.0)
    old = torch.zeros(2, 3)
    new = torch.log(torch.tensor([[1.1, 1.0, 2.0], [0.9, 1.0, 0.5]]))
    adv, w = torch.tensor([1.0, -1.0]), torch.tensor([1.0, 2.0])
    got = trainer._ppo_head_counterfactual_policy_loss(new, old, adv, w, 0.2, None)
    for h in range(3):
        ref, _ = _reference_policy_loss(new[:, h], old[:, h], adv, w, 0.2, 0.0)
        assert float(got[h]) == pytest.approx(float(np.mean(ref)), rel=1e-6)


def test_head_counterfactual_with_a_mask_averages_only_where_the_head_could_act(trainer, dual_clip):
    dual_clip(0.0)
    old = torch.zeros(3, 2)
    new = torch.log(torch.tensor([[1.1, 1.3], [0.9, 1.0], [1.5, 0.7]]))
    adv, w = torch.tensor([1.0, -1.0, 0.5]), torch.ones(3)
    mask = torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    got = trainer._ppo_head_counterfactual_policy_loss(new, old, adv, w, 0.2, mask)
    ref, _ = _reference_policy_loss(new[[0, 2], 0], old[[0, 2], 0], adv[[0, 2]], w[[0, 2]], 0.2, 0.0)
    assert float(got[0]) == pytest.approx(float(np.mean(ref)), rel=1e-6)
    assert float(got[1]) == 0.0   # never had the opportunity: 0, not NaN


def test_head_counterfactual_equals_the_policy_loss_when_only_that_head_moved(trainer, dual_clip):
    dual_clip(0.0)
    old_heads = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))
    new_heads = old_heads.clone()
    new_heads[:, 2] += torch.tensor([0.3, -0.4, 0.1, 0.6, -0.05])
    adv = torch.tensor([1.0, -0.5, 2.0, -1.0, 0.3])
    w = torch.ones(5)
    head = trainer._ppo_head_counterfactual_policy_loss(new_heads, old_heads, adv, w, 0.2, None)
    total = trainer._ppo_policy_loss(new_heads.sum(-1), old_heads.sum(-1), adv, w, 0.2)[0]
    assert float(head[2]) == pytest.approx(float(total), rel=1e-5)


# ----- value loss, weights, masks, regulariser ----------------------------------

def test_value_loss_divides_by_return_variance_floored_at_one():
    all_ret = torch.tensor([0.0, 4.0, 8.0])           # var = 16
    loss, per = _ppo_value_loss(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 4.0]), all_ret)
    assert per.tolist() == pytest.approx([1.0 / 16, 4.0 / 16])
    assert float(loss) == pytest.approx(2.5 / 16)
    _, per_small = _ppo_value_loss(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([0.0, 0.1]))
    assert per_small.tolist() == pytest.approx([1.0])  # variance < 1 floored to 1


def test_sample_weights_rescale_to_the_minibatch_size():
    w = _normalised_sample_weights(torch.tensor([1.0, 3.0]))
    assert w.tolist() == pytest.approx([0.5, 1.5])
    assert _normalised_sample_weights(torch.zeros(3)).tolist() == [0.0, 0.0, 0.0]


def test_opp_entropy_masks_pick_the_kick_and_tackle_attempt_columns(trainer):
    assert trainer._opp_entropy_masks(None, None) is None
    n_heads = len(HEAD_LP_KEYS)
    mask = torch.arange(2 * n_heads, dtype=torch.float32).reshape(2, n_heads)
    scale = torch.arange(n_heads, dtype=torch.float32) / 10
    got = trainer._opp_entropy_masks(mask, scale)
    ik, it = HEAD_LP_KEYS.index("kick"), HEAD_LP_KEYS.index("tackle_attempt")
    assert torch.equal(got["kick"], mask[:, ik]) and torch.equal(got["tackle_attempt"], mask[:, it])
    assert got["kick_scale"] == pytest.approx(ik / 10) and got["tackle_attempt_scale"] == pytest.approx(it / 10)
    assert set(trainer._opp_entropy_masks(mask, None)) == {"kick", "tackle_attempt"}


def test_dir_spread_reg_is_zero_when_off_and_the_l2_pull_when_on(trainer):
    names = ["move_dir_log_kappa", "kick_dir_log_kappa", "kick_dir_z_log_std"]
    saved = {n: (getattr(trainer, f"{n}_reg_coef"), getattr(trainer, f"{n}_target")) for n in names}
    try:
        for n in names:
            setattr(trainer, f"{n}_reg_coef", 0.0)
        reg = trainer._dir_spread_reg()
        assert reg.shape == (1,) and float(reg) == 0.0
        expected = 0.0
        for i, n in enumerate(names):
            coef, target = 0.1 * (i + 1), 0.5 * i
            setattr(trainer, f"{n}_reg_coef", coef)
            setattr(trainer, f"{n}_target", target)
            expected += coef * float(((getattr(trainer.execution_net, n).detach() - target) ** 2).mean())
        assert float(trainer._dir_spread_reg().detach()) == pytest.approx(expected, rel=1e-6)
    finally:
        for n, (coef, target) in saved.items():
            setattr(trainer, f"{n}_reg_coef", coef)
            setattr(trainer, f"{n}_target", target)


# ----- parameter groups and optimizer step --------------------------------------

def test_main_and_direction_params_partition_both_networks(trainer):
    main, direction = trainer._main_and_direction_params()
    all_ids = {id(p) for p in list(trainer.decision_net.parameters()) + list(trainer.execution_net.parameters())}
    main_ids, dir_ids = {id(p) for p in main}, {id(p) for p in direction}
    assert not (main_ids & dir_ids)
    assert main_ids | dir_ids == all_ids
    assert dir_ids == set(trainer.direction_param_ids) & all_ids


def test_optimizer_step_warms_up_then_restores_the_scheduled_lrs():
    t = PPOTrainer.from_config(device=torch.device("cpu"))
    t.lr_warmup_steps = 3
    base = [0.01 * (i + 1) for i in range(len(t.optimizer.param_groups))]
    for pg, b in zip(t.optimizer.param_groups, base):
        pg["lr"] = b
    seen = []
    t.optimizer.step = lambda: seen.append([pg["lr"] for pg in t.optimizer.param_groups])
    ctx, stats = SimpleNamespace(warm_base_lrs=base), _UpdateStats()
    for _ in range(4):
        t._ppo_optimizer_step(ctx, stats)
    for k, scale in enumerate([1 / 3, 2 / 3, 1.0, 1.0]):
        assert seen[k] == pytest.approx([b * scale for b in base])
    assert [pg["lr"] for pg in t.optimizer.param_groups] == base
    assert stats.warm_steps_done == 4


# ----- minibatch slicing --------------------------------------------------------

def test_minibatch_obs_and_actions_strip_the_prefix_and_select_rows(trainer):
    batch = {"obs/self_feat": torch.arange(6.0).reshape(3, 2), "action/kick": torch.tensor([0.0, 1.0, 0.0]),
             "returns": torch.zeros(3)}
    idx = torch.tensor([2, 0])
    assert list(trainer._minibatch_obs(batch, idx)) == ["self_feat"]
    assert trainer._minibatch_obs(batch, idx)["self_feat"].tolist() == [[4.0, 5.0], [0.0, 1.0]]
    assert trainer._minibatch_actions(batch, idx)["kick"].tolist() == [0.0, 0.0]


def test_obs_parts_are_in_network_argument_order():
    obs = {k: torch.full((1, 1), float(i)) for i, k in
           enumerate(["self_feat", "other_feat", "exists_mask", "ball_feat", "global_feat"])}
    parts = _obs_parts(obs)
    assert [float(p) for p in parts[:5]] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert len(parts) == 7


# ----- accumulated statistics ---------------------------------------------------

def test_value_by_outcome_accumulated_over_minibatches_equals_one_pass():
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    target = torch.tensor([0.0, 2.5, 3.0, 1.0, 6.0])
    outcomes = ["goal", "timeout", "goal", "goal", "timeout"]
    stats = _UpdateStats()
    for idx in (torch.tensor([0, 1]), torch.tensor([2, 3, 4])):
        stats.accum_value_by_outcome(pred[idx], target[idx], idx, outcomes)
    once = value_mse_by_outcome(pred, target, outcomes)
    got = stats.value_by_outcome()
    assert set(got) == set(once)
    for name, (mse, n, gt_mean, gt_sq) in once.items():
        assert got[name] == pytest.approx((mse, n, gt_mean, gt_sq))


def test_value_by_outcome_is_empty_without_outcomes():
    stats = _UpdateStats()
    stats.accum_value_by_outcome(torch.ones(2), torch.zeros(2), torch.tensor([0, 1]), [])
    assert stats.value_by_outcome() == {}


def test_grad_clip_summary_and_advantage_record():
    stats = _UpdateStats(grad_norm_main=[1.0, 3.0, 5.0], clip_triggered_main=2)
    g = stats.grad_clip_summary()
    assert (g["n_main"], g["pct_main"], g["mean_main"], g["max_main"]) == (3, pytest.approx(200 / 3), 3.0, 5.0)
    assert (g["n_dir"], g["pct_dir"], g["mean_dir"], g["max_dir"]) == (0, 0.0, 0.0, 0.0)
    stats.record_advantage(torch.tensor([-1.0, 0.0, 4.0]))
    assert (stats.adv_mean[0], stats.adv_min[0], stats.adv_max[0]) == (pytest.approx(1.0), -1.0, 4.0)
    assert stats.adv_std[0] == pytest.approx(float(torch.tensor([-1.0, 0.0, 4.0]).std()))


def test_update_stats_instances_do_not_share_containers():
    a, b = _UpdateStats(), _UpdateStats()
    a.head_grad_norm[next(iter(a.head_grad_norm))].append(1.0)
    a.ratio_spike["ratio"] = 5.0
    assert all(not v for v in b.head_grad_norm.values()) and b.ratio_spike["ratio"] == float("-inf")


# ----- diagnostics helpers ------------------------------------------------------

def test_angular_helpers():
    assert diag._angular_dmean_deg(0.0) == 0.0
    assert diag._angular_dmean_deg(math.sqrt(2.0)) == pytest.approx(90.0)
    assert diag._angular_dmean_deg(5.0) == pytest.approx(180.0)   # clamped, no domain error
    assert diag._angular_dlog_std_deg(0.0, 2.0) == 0.0
    # log_kappa 2 -> 3: std goes from sqrt(1/e^2) to sqrt(1/e^3) radians.
    expected = math.degrees(math.exp(-1.0)) - math.degrees(math.exp(-1.5))
    assert diag._angular_dlog_std_deg(1.0, 3.0) == pytest.approx(expected)


def test_episode_abs_adv_means_drop_the_trailing_partial_episode():
    adv = torch.tensor([1.0, -3.0, 2.0, 2.0, -4.0, 9.0])
    dones = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    assert diag._episode_abs_adv_means(adv, dones).tolist() == pytest.approx([2.0, 8.0 / 3])
    assert diag._episode_abs_adv_means(adv, torch.zeros(6)) is None


def test_quantile_sample_only_subsamples_past_the_cap(monkeypatch):
    t = torch.arange(10.0)
    assert diag._quantile_sample(t) is t
    monkeypatch.setattr(diag, "_MAX_QUANTILE_ELEMENTS", 4)
    s = diag._quantile_sample(t)
    assert s.numel() == 4 and set(s.tolist()) <= set(t.tolist())


def test_percentile_str_format():
    s = diag._percentile_str(torch.arange(101.0), digits=1)
    assert s == "p0=0.0  p1=1.0  p10=10.0  p50=50.0  p90=90.0  p99=99.0  p100=100.0"


def test_trimmed_means_drop_the_tails():
    x = torch.tensor([-100.0] + [1.0] * 8 + [100.0])
    assert diag._trimmed_mean_p10_p90(x) == pytest.approx(1.0)
    assert math.isnan(diag._trimmed_mean_p25_p75(torch.zeros(0)))

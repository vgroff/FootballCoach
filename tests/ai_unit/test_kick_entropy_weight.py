"""ppo.ent_kick_weight: extra entropy-bonus multiplier on the kick gate + kick_dir + kick_power, applied only where the PPO loss is built.

The logged entropy and the per-head breakdown must stay UNBOOSTED (so trend lines are comparable across settings), the boost must be
exactly (w - 1) x the three kick terms, and it must only carry gradient into the kick heads.
"""
import pytest
import torch

from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _kick_power_frac_stats


@pytest.fixture(scope="module")
def trainer():
    return PPOTrainer.from_config(device=torch.device("cpu"), inference_only=True, separate_value_net=False)


@pytest.fixture(autouse=True)
def _isolate_move_dir_weight(trainer):
    # ent_move_dir_weight/ent_kick_power_only_weight/ent_kick_dir_azimuth_only_weight/
    # ent_kick_dir_z_only_weight all come from the live config (non-1.0 since runs 305/307/308) and
    # would otherwise leak extra boost terms into every "kick boost" assertion here, which only
    # cares about ent_kick_weight.
    old = (trainer.ent_move_dir_weight, trainer.ent_kick_power_only_weight,
           trainer.ent_kick_dir_azimuth_only_weight, trainer.ent_kick_dir_z_only_weight)
    trainer.ent_move_dir_weight = 1.0
    trainer.ent_kick_power_only_weight = 1.0
    trainer.ent_kick_dir_azimuth_only_weight = 1.0
    trainer.ent_kick_dir_z_only_weight = 1.0
    yield
    (trainer.ent_move_dir_weight, trainer.ent_kick_power_only_weight,
     trainer.ent_kick_dir_azimuth_only_weight, trainer.ent_kick_dir_z_only_weight) = old


def _heads(trainer, n=16, seed=0):
    torch.manual_seed(seed)
    sf, of, bf, gf = torch.randn(n, 39), torch.randn(n, 21, 39), torch.randn(n, 12), torch.randn(n, 31)
    ex, sat, oat = torch.ones(n, 21), torch.randn(n, 3), torch.randn(n, 21, 3)
    d = trainer.decision_net(sf, of, ex, bf, gf, sat, oat)
    e = trainer.execution_net(sf, of, ex, bf, gf, d, sat, oat)
    return d, e, ex


def test_default_weight_is_a_noop(trainer):
    trainer.ent_kick_weight = 1.0
    d, e, ex = _heads(trainer)
    _, _, boost = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    assert float(boost.detach()) == pytest.approx(0.0, abs=1e-9)


def test_boost_is_w_minus_one_times_the_three_kick_terms(trainer):
    d, e, ex = _heads(trainer)
    trainer.ent_kick_weight = 1.0
    ent1, bk1 = trainer._compute_entropy(d, e, ex, return_breakdown=True)
    trainer.ent_kick_weight = 10.0
    try:
        ent10, bk10, boost = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    finally:
        trainer.ent_kick_weight = 1.0
    assert float(boost.detach()) == pytest.approx(9.0 * (bk10["kick"] + bk10["kick_dir"] + bk10["kick_power"]), rel=1e-5)
    assert float(boost.detach()) > 0.0
    assert float(ent10.detach()) == pytest.approx(float(ent1.detach()), rel=1e-6)          # logged entropy is unboosted
    assert bk10 == pytest.approx(bk1)                                    # so is every per-head entry


def test_boost_only_reaches_the_kick_heads(trainer):
    d, e, ex = _heads(trainer)
    trainer.ent_kick_weight = 10.0
    try:
        _, _, boost = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    finally:
        trainer.ent_kick_weight = 1.0
    trainer.execution_net.zero_grad()
    boost.backward()
    en = trainer.execution_net
    # Entropy of the Bernoulli gate depends on its logit; entropy of the dir/power heads depends only on their SPREAD parameters
    # (von Mises kappa, z sigma, power sigma), not on their means -- so the boost widens spreads and raises p(kick).
    assert en.kick_logit.weight.grad is not None and float(en.kick_logit.weight.grad.abs().sum()) > 0.0
    for name in ("kick_dir_log_kappa", "kick_dir_z_log_std", "kick_power_log_std"):
        g = getattr(en, name).grad
        assert g is not None and float(g.abs().sum()) > 0.0, name
    for name in ("kick_direction", "kick_power", "exec_move_logit", "sprint_logit", "tackle_attempt_logit", "move_direction"):
        g = getattr(en, name).weight.grad
        assert g is None or float(g.abs().sum()) == 0.0, name
    assert en.move_dir_log_kappa.grad is None or float(en.move_dir_log_kappa.grad.abs().sum()) == 0.0


def test_call_signatures_stay_backward_compatible(trainer):
    d, e, ex = _heads(trainer)
    assert torch.is_tensor(trainer._compute_entropy(d, e, ex))
    out = trainer._compute_entropy(d, e, ex, return_breakdown=True)
    assert isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict)


def test_kick_power_frac_stats():
    raw = torch.tensor([[-1.0], [0.0], [2.0], [5.0]])
    batch = {"action/kick": torch.tensor([[1.0], [0.0], [1.0], [1.0]]), "action/kick_power_raw": raw}
    s = _kick_power_frac_stats(batch)
    pf = torch.sigmoid(torch.tensor([-1.0, 2.0, 5.0]))
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(float(pf.mean()), rel=1e-6)
    assert s["std"] == pytest.approx(float(pf.std()), rel=1e-6)


def test_kick_power_frac_stats_no_kicks_and_single_kick():
    empty = _kick_power_frac_stats({"action/kick": torch.zeros(5, 1), "action/kick_power_raw": torch.zeros(5, 1)})
    assert empty["n"] == 0 and empty["mean"] != empty["mean"]            # NaN
    one = _kick_power_frac_stats({"action/kick": torch.tensor([[1.0], [0.0]]), "action/kick_power_raw": torch.tensor([[0.0], [3.0]])})
    assert one["n"] == 1 and one["mean"] == pytest.approx(0.5) and one["std"] == 0.0

"""ppo.ent_move_dir_weight: extra entropy-bonus multiplier on the move_dir head only, applied only where the PPO loss is built.

The logged entropy and per-head breakdown must stay UNBOOSTED, the boost must be exactly (w - 1) x ent_dir_weight x P(exec_move) x the raw
von Mises entropy, and -- because the von Mises entropy depends on kappa alone -- it must only carry gradient into move_dir_log_kappa (in
particular NOT into the exec_move gate, which the ordinary move_dir entropy term also pulls on through its P(exec_move) factor).
"""
import pytest
import torch

from footballcoach.ai.ppo.ppo_trainer import PPOTrainer


@pytest.fixture(scope="module")
def trainer():
    return PPOTrainer.from_config(device=torch.device("cpu"), inference_only=True, separate_value_net=False)


def _heads(trainer, n=16, seed=0):
    torch.manual_seed(seed)
    sf, of, bf, gf = torch.randn(n, 39), torch.randn(n, 21, 39), torch.randn(n, 12), torch.randn(n, 31)
    ex, sat, oat = torch.ones(n, 21), torch.randn(n, 3), torch.randn(n, 21, 3)
    d = trainer.decision_net(sf, of, ex, bf, gf, sat, oat)
    e = trainer.execution_net(sf, of, ex, bf, gf, d, sat, oat)
    return d, e, ex


@pytest.fixture(autouse=True)
def _restore(trainer):
    old_k, old_m = trainer.ent_kick_weight, trainer.ent_move_dir_weight
    yield
    trainer.ent_kick_weight, trainer.ent_move_dir_weight = old_k, old_m


def test_default_weight_is_a_noop(trainer):
    trainer.ent_kick_weight, trainer.ent_move_dir_weight = 1.0, 1.0
    d, e, ex = _heads(trainer)
    _, _, boost = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    assert float(boost.detach()) == pytest.approx(0.0, abs=1e-9)


def test_boost_is_w_minus_one_times_the_move_dir_term(trainer):
    d, e, ex = _heads(trainer)
    trainer.ent_kick_weight, trainer.ent_move_dir_weight = 1.0, 1.0
    ent1, bk1 = trainer._compute_entropy(d, e, ex, return_breakdown=True)
    trainer.ent_move_dir_weight = 6.0
    ent6, bk6, boost = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    # bk["move_dir"] is already P(exec_move) x ent_dir_weight x raw entropy, so the boost is (w - 1) x that.
    assert float(boost.detach()) == pytest.approx(5.0 * bk6["move_dir"], rel=1e-5)
    assert float(ent6.detach()) == pytest.approx(float(ent1.detach()), rel=1e-6)      # logged entropy is unboosted
    assert bk6 == pytest.approx(bk1)                                    # so is every per-head entry


def test_boost_only_reaches_move_dir_log_kappa(trainer):
    d, e, ex = _heads(trainer)
    trainer.ent_kick_weight, trainer.ent_move_dir_weight = 1.0, 6.0
    _, _, boost = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    trainer.execution_net.zero_grad()
    trainer.decision_net.zero_grad()
    boost.backward()
    en = trainer.execution_net
    g = en.move_dir_log_kappa.grad
    assert g is not None and float(g.abs().sum()) > 0.0
    # boost = (w - 1) x P(exec_move) x ent_dir_weight x H(kappa) and H decreases as kappa grows, so d boost / d log kappa < 0: the trainer
    # subtracts ent_coef x boost from the loss, i.e. maximises it, which pulls log kappa DOWN (slower sharpening).
    assert float(g) < 0.0
    for name in ("exec_move_logit", "sprint_logit", "kick_logit", "tackle_attempt_logit", "move_direction", "kick_direction", "kick_power"):
        gg = getattr(en, name).weight.grad
        assert gg is None or float(gg.abs().sum()) == 0.0, name
    for name in ("kick_dir_log_kappa", "kick_dir_z_log_std", "kick_power_log_std"):
        gg = getattr(en, name).grad
        assert gg is None or float(gg.abs().sum()) == 0.0, name


def test_boost_scales_linearly_with_the_weight(trainer):
    d, e, ex = _heads(trainer)
    trainer.ent_kick_weight = 1.0
    trainer.ent_move_dir_weight = 3.0
    _, _, b3 = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    trainer.ent_move_dir_weight = 11.0
    _, _, b11 = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    assert float(b11.detach()) == pytest.approx(5.0 * float(b3.detach()), rel=1e-5)          # (11-1) / (3-1)


def test_kick_boost_is_unaffected_and_the_two_add(trainer):
    d, e, ex = _heads(trainer)
    trainer.ent_kick_weight, trainer.ent_move_dir_weight = 4.0, 1.0
    _, _, kick_only = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    trainer.ent_kick_weight, trainer.ent_move_dir_weight = 1.0, 6.0
    _, _, move_only = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    trainer.ent_kick_weight, trainer.ent_move_dir_weight = 4.0, 6.0
    _, _, both = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    assert float(both.detach()) == pytest.approx(float(kick_only.detach()) + float(move_only.detach()), rel=1e-5)


def test_call_signatures_stay_backward_compatible(trainer):
    d, e, ex = _heads(trainer)
    assert torch.is_tensor(trainer._compute_entropy(d, e, ex))
    out = trainer._compute_entropy(d, e, ex, return_breakdown=True)
    assert isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict)

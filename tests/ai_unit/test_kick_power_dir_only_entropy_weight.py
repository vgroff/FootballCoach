"""ppo.ent_kick_power_only_weight / ppo.ent_kick_dir_only_weight: extra entropy-bonus multipliers on
kick_power / kick_dir specifically, applied only where the PPO loss is built.

Unlike ppo.ent_kick_weight (which deliberately ALSO pushes p(kick) itself, to stop the kick gate
collapsing early), these detach the kick-gate's contribution to the (E[kick]-weighted) entropy term,
so the boost reaches only the spread parameter -- kick_power_log_std, or kick_dir_log_kappa /
kick_dir_z_log_std -- never kick_logit. Same pattern as ppo.ent_move_dir_weight, tested the same way:
logged entropy/breakdown must stay unboosted, and the boost's gradient must not reach the gate.
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


def _opp_masks(n=16, seed=1):
    torch.manual_seed(seed)
    opp_kick = (torch.rand(n) > 0.5).float()
    return {"kick": opp_kick, "tackle_attempt": torch.zeros(n)}


@pytest.fixture(autouse=True)
def _restore(trainer):
    old = (trainer.ent_kick_weight, trainer.ent_move_dir_weight, trainer.ent_kick_power_only_weight, trainer.ent_kick_dir_only_weight)
    yield
    (trainer.ent_kick_weight, trainer.ent_move_dir_weight, trainer.ent_kick_power_only_weight, trainer.ent_kick_dir_only_weight) = old


def _set(trainer, kick=1.0, move_dir=1.0, kick_power_only=1.0, kick_dir_only=1.0):
    trainer.ent_kick_weight = kick
    trainer.ent_move_dir_weight = move_dir
    trainer.ent_kick_power_only_weight = kick_power_only
    trainer.ent_kick_dir_only_weight = kick_dir_only


@pytest.mark.parametrize("opp_masks", [None, "masked"], ids=["unmasked", "masked"])
def test_default_weight_is_a_noop(trainer, opp_masks):
    _set(trainer)
    d, e, ex = _heads(trainer)
    om = _opp_masks() if opp_masks == "masked" else None
    _, _, boost = trainer._compute_entropy(d, e, ex, opp_masks=om, return_breakdown=True, return_kick_boost=True)
    assert float(boost.detach()) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("opp_masks", [None, "masked"], ids=["unmasked", "masked"])
def test_kick_power_only_boost_leaves_logged_entropy_and_breakdown_unboosted(trainer, opp_masks):
    d, e, ex = _heads(trainer)
    om = _opp_masks() if opp_masks == "masked" else None
    _set(trainer)
    ent1, bk1 = trainer._compute_entropy(d, e, ex, opp_masks=om, return_breakdown=True)
    _set(trainer, kick_power_only=7.0)
    ent7, bk7, boost = trainer._compute_entropy(d, e, ex, opp_masks=om, return_breakdown=True, return_kick_boost=True)
    assert float(ent7.detach()) == pytest.approx(float(ent1.detach()), rel=1e-6)
    assert bk7 == pytest.approx(bk1)
    assert float(boost.detach()) != 0.0


@pytest.mark.parametrize("opp_masks", [None, "masked"], ids=["unmasked", "masked"])
def test_kick_power_only_boost_only_reaches_kick_power_log_std(trainer, opp_masks):
    d, e, ex = _heads(trainer)
    om = _opp_masks() if opp_masks == "masked" else None
    _set(trainer, kick_power_only=7.0)
    _, _, boost = trainer._compute_entropy(d, e, ex, opp_masks=om, return_breakdown=True, return_kick_boost=True)
    trainer.execution_net.zero_grad()
    trainer.decision_net.zero_grad()
    boost.backward()
    en = trainer.execution_net
    g = en.kick_power_log_std.grad
    assert g is not None and float(g.abs().sum()) > 0.0
    # Unlike move_dir/kick_dir's log_kappa (a CONCENTRATION parameter -- higher kappa = narrower =
    # LOWER entropy, so its boost gradient is negative), kick_power_log_std is a plain Gaussian
    # scale: entropy = log_std + const, strictly INCREASING in log_std. So d(boost)/d(log_std) > 0
    # here -- the trainer's loss subtracts ent_coef*boost, so gradient descent on the loss still
    # raises log_std (widens sigma, slower sharpening), just via a positive raw boost-gradient.
    assert float(g) > 0.0
    for name in ("kick_logit", "exec_move_logit", "sprint_logit", "tackle_attempt_logit", "move_direction", "kick_direction", "kick_power"):
        gg = getattr(en, name).weight.grad
        assert gg is None or float(gg.abs().sum()) == 0.0, name
    for name in ("move_dir_log_kappa", "kick_dir_log_kappa", "kick_dir_z_log_std"):
        gg = getattr(en, name).grad
        assert gg is None or float(gg.abs().sum()) == 0.0, name


@pytest.mark.parametrize("opp_masks", [None, "masked"], ids=["unmasked", "masked"])
def test_kick_dir_only_boost_only_reaches_kick_dir_spread_params(trainer, opp_masks):
    d, e, ex = _heads(trainer)
    om = _opp_masks() if opp_masks == "masked" else None
    _set(trainer, kick_dir_only=7.0)
    _, _, boost = trainer._compute_entropy(d, e, ex, opp_masks=om, return_breakdown=True, return_kick_boost=True)
    trainer.execution_net.zero_grad()
    trainer.decision_net.zero_grad()
    boost.backward()
    en = trainer.execution_net
    g_kappa = en.kick_dir_log_kappa.grad
    assert g_kappa is not None and float(g_kappa.abs().sum()) > 0.0
    # kick_dir_log_kappa is a concentration parameter like move_dir's (entropy falls as it rises).
    assert float(g_kappa) < 0.0
    g_zstd = en.kick_dir_z_log_std.grad
    assert g_zstd is not None and float(g_zstd.abs().sum()) > 0.0
    # kick_dir_z_log_std is a plain Gaussian scale like kick_power's (entropy rises with it).
    assert float(g_zstd) > 0.0
    for name in ("kick_logit", "exec_move_logit", "sprint_logit", "tackle_attempt_logit", "move_direction", "kick_direction", "kick_power"):
        gg = getattr(en, name).weight.grad
        assert gg is None or float(gg.abs().sum()) == 0.0, name
    for name in ("move_dir_log_kappa", "kick_power_log_std"):
        gg = getattr(en, name).grad
        assert gg is None or float(gg.abs().sum()) == 0.0, name


def test_boost_scales_linearly_with_each_weight(trainer):
    d, e, ex = _heads(trainer)
    _set(trainer, kick_power_only=3.0)
    _, _, b3 = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    _set(trainer, kick_power_only=11.0)
    _, _, b11 = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    assert float(b11.detach()) == pytest.approx(5.0 * float(b3.detach()), rel=1e-5)  # (11-1)/(3-1)

    _set(trainer, kick_dir_only=4.0)
    _, _, c4 = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    _set(trainer, kick_dir_only=13.0)
    _, _, c13 = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    assert float(c13.detach()) == pytest.approx(4.0 * float(c4.detach()), rel=1e-5)  # (13-1)/(4-1)


def test_all_four_kick_move_boosts_add(trainer):
    d, e, ex = _heads(trainer)
    _set(trainer, kick=4.0)
    _, _, kick_only = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    _set(trainer, move_dir=6.0)
    _, _, move_only = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    _set(trainer, kick_power_only=5.0)
    _, _, power_only = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    _set(trainer, kick_dir_only=8.0)
    _, _, dir_only = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    _set(trainer, kick=4.0, move_dir=6.0, kick_power_only=5.0, kick_dir_only=8.0)
    _, _, all_four = trainer._compute_entropy(d, e, ex, return_breakdown=True, return_kick_boost=True)
    assert float(all_four.detach()) == pytest.approx(
        float(kick_only.detach()) + float(move_only.detach()) + float(power_only.detach()) + float(dir_only.detach()), rel=1e-5
    )


def test_call_signatures_stay_backward_compatible(trainer):
    _set(trainer)
    d, e, ex = _heads(trainer)
    assert torch.is_tensor(trainer._compute_entropy(d, e, ex))
    out = trainer._compute_entropy(d, e, ex, return_breakdown=True)
    assert isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict)

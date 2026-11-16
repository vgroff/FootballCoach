"""bc.kick_power_loss = "bce": the soft-target BCE for the kick_power head
(bc.kick_power_bce) and its use in bc_loss_from_tensor."""
from __future__ import annotations

import pytest
import torch

from footballcoach.ai.action.distributions import configured_kick_power_floor, kick_power_from_raw
from footballcoach.ai.ppo.bc import BC_LABEL_DIM, _I_KICK_POWER, _I_KICK_THIS_TICK, _I_VALID, bc_loss_from_tensor, kick_power_bce


def _y(power):
    f = configured_kick_power_floor()
    return (power - f) / (1.0 - f)


def test_gradient_on_raw_is_the_power_error():
    raw = torch.tensor([-2.8, 0.0, 1.5], requires_grad=True)
    target = torch.tensor([0.9, 0.2, 0.5])
    kick_power_bce(raw, target).sum().backward()
    assert raw.grad.tolist() == pytest.approx((torch.sigmoid(raw.detach()) - _y(target)).tolist(), abs=1e-6)


def test_zero_at_the_match_and_positive_elsewhere():
    target = torch.tensor([0.2, 0.5, 0.9])
    y = _y(target)
    matched = torch.log(y / (1 - y))                       # sigmoid(raw) == y
    assert kick_power_bce(matched, target).abs().max() < 1e-6
    assert (kick_power_bce(matched + 0.5, target) > 0).all()
    # Same physical power out of the head as the label asks for:
    assert kick_power_from_raw(matched, configured_kick_power_floor()).tolist() == pytest.approx(target.tolist(), abs=1e-6)


def test_finite_at_the_ends():
    raw = torch.tensor([-3.0, 3.0])
    out = kick_power_bce(raw, torch.tensor([configured_kick_power_floor(), 1.0]))
    assert torch.isfinite(out).all() and (out > 0).all()


@pytest.fixture(scope="module")
def heads():
    """Real decision / execution heads for 2 rows (a trainer forward pass on a
    phase-2 observation); tests swap in their own kick_power output."""
    import random
    from footballcoach.ai.curriculum.envs import build_role_env
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.obs.encoder import encode_observation
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    from footballcoach.ai.ppo.trainer_common import _ai_types
    torch.manual_seed(0)
    t = PPOTrainer.from_config(device=torch.device("cpu"))
    env = build_role_env(PHASES_BY_ID[2], controller_weights={"attacker": {"neural": 1}, "defender": {"rules": 1},
                                                             "goalkeeper": {"rules": 1}})
    env.sample_action_fn = t._sample_action
    env.reset(seed=0)
    obs = encode_observation(match=env.match, player_id="attacker", time_remaining_s=10.0,
                             attack_defence_smoothed=0.5, rng=random.Random(0), phase=2).to_torch_dict()
    b = {k: v.unsqueeze(0).repeat(2, *([1] * v.dim())) for k, v in obs.items()}
    sat, oat = _ai_types(b)
    with torch.no_grad():
        d = t.decision_net(b["self_feat"], b["other_feat"], b["exists_mask"], b["ball_feat"], b["global_feat"], sat, oat)
        e = t.execution_net(b["self_feat"], b["other_feat"], b["exists_mask"], b["ball_feat"], b["global_feat"], d, sat, oat)
    return t._bc_heads_for_loss(d), e


def _kick_labels(n, power=0.9):
    labels = torch.zeros(n, BC_LABEL_DIM)
    labels[:, _I_VALID] = 1.0
    labels[:, _I_KICK_THIS_TICK] = 1.0
    labels[:, _I_KICK_POWER] = power
    return labels


@pytest.mark.parametrize("mode", ["mse", "bce"])
def test_the_loss_form_reaches_the_kick_power_breakdown(heads, mode):
    import dataclasses
    d, e = heads
    raw = torch.tensor([-2.8, 0.5])
    e = dataclasses.replace(e, kick_power=raw.reshape(2, 1))
    labels = _kick_labels(2)
    _, bkdn = bc_loss_from_tensor(labels, d, e, return_breakdown=True, kick_power_loss=mode)
    want = (kick_power_bce(raw, labels[:, _I_KICK_POWER]) if mode == "bce"
            else (kick_power_from_raw(raw, configured_kick_power_floor()) - 0.9) ** 2).mean()
    assert bkdn["kick_power"] == pytest.approx(float(want), rel=1e-5)


def test_unknown_mode_is_refused(heads):
    d, e = heads
    with pytest.raises(ValueError, match="kick_power_loss"):
        bc_loss_from_tensor(_kick_labels(2), d, e, kick_power_loss="huber")


def test_trainer_reads_the_config(monkeypatch):
    from footballcoach.ai.config import load_ai_config
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    bc_cfg = load_ai_config()["bc"]
    monkeypatch.delitem(bc_cfg, "kick_power_loss", raising=False)
    t = PPOTrainer.from_config(device=torch.device("cpu"))
    assert t._bc_kick_power_loss == "mse" and t._bc_loss_kwargs()["kick_power_loss"] == "mse"
    monkeypatch.setitem(bc_cfg, "kick_power_loss", "bce")
    assert PPOTrainer.from_config(device=torch.device("cpu"))._bc_loss_kwargs()["kick_power_loss"] == "bce"

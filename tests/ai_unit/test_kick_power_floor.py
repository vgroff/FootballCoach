"""ppo.kick_power_floor: executed kick power = floor + (1 - floor) * sigmoid(raw). PPO maths (log_prob/entropy) is on the raw draw and must not change."""
import math

import pytest
import torch

from footballcoach.ai.action.distributions import configured_kick_power_floor, kick_power_from_raw
from footballcoach.ai.ppo import bc
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _kick_power_frac_stats


@pytest.fixture(scope="module")
def trainer():
    return PPOTrainer.from_config(device=torch.device("cpu"), inference_only=True, separate_value_net=False)


class TestMapping:
    def test_floor_zero_is_the_plain_sigmoid(self):
        raw = torch.tensor([-8.0, -1.0, 0.0, 2.0, 9.0])
        assert torch.allclose(kick_power_from_raw(raw, 0.0), torch.sigmoid(raw))

    def test_floor_bounds_and_midpoint(self):
        out = kick_power_from_raw(torch.tensor([-1000.0, 0.0, 1000.0]), 0.03)
        assert float(out[0]) == pytest.approx(0.03) and float(out[1]) == pytest.approx(0.03 + 0.97 * 0.5) and float(out[2]) == pytest.approx(1.0)

    def test_never_below_floor_for_any_raw(self):
        raw = torch.linspace(-200, 200, 4001)
        assert float(kick_power_from_raw(raw, 0.03).min()) >= 0.03 - 1e-7

    def test_matches_the_head_mapping_used_in_play(self, trainer):
        old = trainer.kick_power_floor
        trainer.kick_power_floor = 0.03
        try:
            head = trainer._kick_power_head(torch.zeros(4, 1), torch.zeros(1))
            raw = torch.tensor([[-30.0], [-1.0], [0.5], [12.0]])
            assert torch.allclose(head.to_physical(raw), kick_power_from_raw(raw, 0.03))
        finally:
            trainer.kick_power_floor = old


class TestPpoMathUnchanged:
    def test_log_prob_entropy_and_mode_raw_do_not_depend_on_the_floor(self, trainer):
        old = trainer.kick_power_floor
        mean, log_std, raw = torch.randn(8, 1), torch.zeros(1), torch.randn(8, 1)
        try:
            trainer.kick_power_floor = 0.0
            h0 = trainer._kick_power_head(mean, log_std)
            lp0, ent0 = h0.log_prob(raw), h0.entropy()
            trainer.kick_power_floor = 0.03
            h1 = trainer._kick_power_head(mean, log_std)
            assert torch.equal(h1.log_prob(raw), lp0) and torch.equal(h1.entropy(), ent0)
            assert float(h1.to_physical(raw).min()) >= 0.03 - 1e-7 and not torch.equal(h1.to_physical(raw), h0.to_physical(raw))
        finally:
            trainer.kick_power_floor = old

    def test_trainer_reads_the_floor_from_config(self, trainer):
        assert trainer.kick_power_floor == pytest.approx(configured_kick_power_floor())


class TestStatsAndBc:
    def test_frac_stats_use_the_floor(self):
        batch = {"action/kick": torch.tensor([[1.0], [1.0], [0.0]]), "action/kick_power_raw": torch.tensor([[-1000.0], [1000.0], [0.0]])}
        s = _kick_power_frac_stats(batch, floor=0.03)
        assert s["n"] == 2 and s["mean"] == pytest.approx((0.03 + 1.0) / 2)
        s0 = _kick_power_frac_stats(batch)
        assert s0["mean"] == pytest.approx(0.5)

    def test_bc_targets_are_in_floored_units(self, monkeypatch):
        monkeypatch.setattr(bc, "configured_kick_power_floor", lambda: 0.05)
        raw = torch.tensor([-1000.0, 0.0])
        assert torch.allclose(bc._kick_power_physical(raw), torch.tensor([0.05, 0.05 + 0.95 * 0.5]))
        monkeypatch.setattr(bc, "configured_kick_power_floor", lambda: 0.0)
        assert torch.allclose(bc._kick_power_physical(raw), torch.sigmoid(raw))


def test_config_floor_is_a_sane_fraction():
    f = configured_kick_power_floor()
    assert isinstance(f, float) and 0.0 <= f < 1.0 and not math.isnan(f)

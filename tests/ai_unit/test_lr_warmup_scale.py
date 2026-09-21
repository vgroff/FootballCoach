"""``_lr_warmup_scale`` (ai/ppo/ppo_trainer.py): the linear LR-warmup multiplier
shared by ``ppg_value_refit`` (bc.ppg_warmup_steps) and ``_ppo_update``
(ppo.lr_warmup_steps)."""
import pytest

from footballcoach.ai.ppo.ppo_trainer import _lr_warmup_scale


class TestLRWarmupScale:
    @pytest.mark.parametrize("step", [0, 1, 10, 10_000])
    def test_zero_or_negative_warmup_is_off(self, step):
        assert _lr_warmup_scale(step, 0) == 1.0
        assert _lr_warmup_scale(step, -5) == 1.0

    def test_first_step_runs_at_one_over_n(self):
        assert _lr_warmup_scale(0, 50) == pytest.approx(1 / 50)

    def test_reaches_full_lr_exactly_on_the_nth_step(self):
        assert _lr_warmup_scale(48, 50) == pytest.approx(49 / 50)
        assert _lr_warmup_scale(49, 50) == pytest.approx(1.0)

    def test_stays_at_one_afterwards(self):
        assert _lr_warmup_scale(50, 50) == 1.0
        assert _lr_warmup_scale(10_000, 50) == 1.0

    def test_monotonic_non_decreasing(self):
        scales = [_lr_warmup_scale(i, 50) for i in range(120)]
        assert all(b >= a for a, b in zip(scales, scales[1:]))
        assert 0.0 < scales[0] < scales[-1] == 1.0

    def test_single_step_warmup_is_a_noop(self):
        assert _lr_warmup_scale(0, 1) == 1.0

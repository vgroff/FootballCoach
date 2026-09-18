"""Coverage for ``_trimmed_rmse`` (ai/ppo/ppo_trainer.py) -- the one-sided
"RMSE excluding the worst N% of rows" used for ppg_value_refit's extra
train/val log figures."""
from __future__ import annotations

import math

import pytest
import torch

from footballcoach.ai.ppo.ppo_trainer import _trimmed_rmse


def _plain_rmse(sq: torch.Tensor) -> float:
    return math.sqrt(float(sq.mean()))


class TestTrimmedRmse:
    def test_drops_exactly_the_worst_fraction(self):
        # 10 rows: nine perfect-ish (sq err 1.0) and one huge outlier.
        sq = torch.tensor([1.0] * 9 + [10_000.0])
        assert _trimmed_rmse(sq, 0.10) == pytest.approx(1.0)
        assert _plain_rmse(sq) > 30.0  # sanity: the outlier dominates the plain figure

    def test_order_of_input_does_not_matter(self):
        sq = torch.tensor([4.0, 1.0, 9.0, 16.0, 25.0, 1.0, 4.0, 9.0, 100.0, 0.0])
        shuffled = sq[torch.randperm(len(sq))]
        assert _trimmed_rmse(sq, 0.2) == pytest.approx(_trimmed_rmse(shuffled, 0.2))

    def test_keep_count_rounds_up(self):
        # 7 rows, trim 10% -> keep ceil(6.3) = 7 -> nothing dropped.
        sq = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
        assert _trimmed_rmse(sq, 0.10) == pytest.approx(_plain_rmse(sq))
        # trim 30% -> keep ceil(4.9) = 5 -> the two largest (100 and one 1.0) dropped.
        assert _trimmed_rmse(sq, 0.30) == pytest.approx(1.0)

    def test_zero_or_negative_trim_is_plain_rmse(self):
        sq = torch.tensor([1.0, 4.0, 9.0, 100.0])
        assert _trimmed_rmse(sq, 0.0) == pytest.approx(_plain_rmse(sq))
        assert _trimmed_rmse(sq, -0.5) == pytest.approx(_plain_rmse(sq))

    def test_never_drops_every_row(self):
        sq = torch.tensor([2.0, 8.0])
        # even an extreme trim keeps at least one row (the smallest error).
        assert _trimmed_rmse(sq, 0.99) == pytest.approx(math.sqrt(2.0))

    def test_empty_input_is_nan(self):
        assert math.isnan(_trimmed_rmse(torch.empty(0), 0.10))

    def test_never_larger_than_plain_rmse(self):
        torch.manual_seed(0)
        sq = torch.rand(1000) ** 3 * 50
        for frac in (0.01, 0.1, 0.25, 0.5):
            assert _trimmed_rmse(sq, frac) <= _plain_rmse(sq) + 1e-9

    def test_accepts_multidimensional_input(self):
        sq = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 10_000.0]])
        # 6 elements, trim 20% -> keep ceil(4.8) = 5 -> only the outlier dropped.
        assert _trimmed_rmse(sq, 0.2) == pytest.approx(1.0)

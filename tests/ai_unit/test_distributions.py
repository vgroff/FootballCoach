"""Unit tests for action/distributions.py.

Covers IndependentBernoulli, MaskedCategorical, SquashedNormalHead, and
DirectionHead.  These are the lowest-level building blocks of PPO log_prob
computation - any bug here silently corrupts the training signal across
every single update step.

Key invariants tested:
- Masked slots get EXACTLY zero probability (not just small).
- Probabilities of unmasked slots sum exactly to 1.
- Squashed outputs always stay within declared physical bounds.
- Unit-vector outputs are exactly unit length (to numerical precision).
- No NaN or Inf anywhere.
- log_prob values are finite for valid (sampled from the same distribution) inputs.
"""
import math

import numpy as np
import pytest
import torch

from footballcoach.ai.action.distributions import (
    DirectionHead,
    IndependentBernoulli,
    MaskedCategorical,
    SquashedNormalHead,
)


# ---------------------------------------------------------------------------
# IndependentBernoulli
# ---------------------------------------------------------------------------

class TestIndependentBernoulli:

    def test_high_positive_logit_gives_prob_near_one(self):
        ib = IndependentBernoulli(torch.tensor([[10.0]]))
        assert float(ib.prob()) > 0.99

    def test_high_negative_logit_gives_prob_near_zero(self):
        ib = IndependentBernoulli(torch.tensor([[-10.0]]))
        assert float(ib.prob()) < 0.01

    def test_zero_logit_gives_prob_half(self):
        ib = IndependentBernoulli(torch.tensor([[0.0]]))
        assert float(ib.prob()) == pytest.approx(0.5, abs=1e-4)

    def test_sample_returns_zero_or_one(self):
        ib = IndependentBernoulli(torch.zeros(1, 5))
        s = ib.sample()
        assert s.shape == (1, 5)
        assert torch.all((s == 0) | (s == 1))

    def test_log_prob_action_one(self):
        """log_prob(1) = log(sigmoid(logit))."""
        logit = torch.tensor([[2.0]])
        ib = IndependentBernoulli(logit)
        lp = ib.log_prob(torch.ones_like(logit))
        expected = math.log(torch.sigmoid(logit).item())
        assert float(lp) == pytest.approx(expected, abs=1e-5)

    def test_log_prob_action_zero(self):
        """log_prob(0) = log(1 - sigmoid(logit))."""
        logit = torch.tensor([[2.0]])
        ib = IndependentBernoulli(logit)
        lp = ib.log_prob(torch.zeros_like(logit))
        expected = math.log(1.0 - torch.sigmoid(logit).item())
        assert float(lp) == pytest.approx(expected, abs=1e-5)

    def test_log_prob_finite(self):
        logit = torch.randn(4, 1)
        ib = IndependentBernoulli(logit)
        s = ib.sample()
        lp = ib.log_prob(s)
        assert torch.all(torch.isfinite(lp))

    def test_entropy_non_negative(self):
        for logit_val in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            ib = IndependentBernoulli(torch.tensor([[logit_val]]))
            assert float(ib.entropy()) >= 0.0

    def test_entropy_maximised_at_zero_logit(self):
        """H is maximised at p=0.5 (logit=0)."""
        ib_zero = IndependentBernoulli(torch.tensor([[0.0]]))
        ib_biased = IndependentBernoulli(torch.tensor([[3.0]]))
        assert float(ib_zero.entropy()) > float(ib_biased.entropy())

    def test_mode_above_half(self):
        ib = IndependentBernoulli(torch.tensor([[2.0]]))
        assert float(ib.mode()) == pytest.approx(1.0)

    def test_mode_below_half(self):
        ib = IndependentBernoulli(torch.tensor([[-2.0]]))
        assert float(ib.mode()) == pytest.approx(0.0)

    def test_batch_log_prob_no_nan(self):
        logits = torch.randn(8, 7)  # batch of 8, 7 heads
        ib = IndependentBernoulli(logits)
        s = ib.sample()
        lp = ib.log_prob(s)
        assert not torch.any(torch.isnan(lp))


# ---------------------------------------------------------------------------
# MaskedCategorical
# ---------------------------------------------------------------------------

class TestMaskedCategorical:

    def _make(self, n_real: int, total: int = 5):
        """Create a MaskedCategorical with n_real real slots out of total."""
        logits = torch.randn(1, total)
        mask = torch.zeros(1, total)
        mask[0, :n_real] = 1.0
        return MaskedCategorical(logits, mask), mask

    def test_masked_slots_have_exactly_zero_prob(self):
        mc, mask = self._make(n_real=2, total=5)
        probs = mc.probs()
        # Slots 2, 3, 4 are masked -> exactly 0.0
        for i in range(2, 5):
            assert float(probs[0, i]) == pytest.approx(0.0, abs=1e-7), (
                f"Slot {i} should have zero prob, got {float(probs[0, i])}"
            )

    def test_unmasked_probs_sum_to_one(self):
        mc, _ = self._make(n_real=3, total=5)
        probs = mc.probs()
        assert float(probs[0, :3].sum()) == pytest.approx(1.0, abs=1e-5)

    def test_all_probs_sum_to_one(self):
        mc, _ = self._make(n_real=3, total=5)
        assert float(mc.probs().sum()) == pytest.approx(1.0, abs=1e-5)

    def test_sample_always_unmasked(self):
        """sample() must never pick a masked slot."""
        logits = torch.tensor([[100.0, 100.0, -1000.0, -1000.0, -1000.0]])
        mask   = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        mc = MaskedCategorical(logits, mask)
        for _ in range(100):
            s = mc.sample()
            assert int(s) in {0, 1}, f"Expected slot 0 or 1, got {int(s)}"

    def test_log_prob_valid_action_is_finite(self):
        mc, _ = self._make(n_real=3, total=5)
        s = mc.sample()
        lp = mc.log_prob(s)
        assert torch.isfinite(lp)

    def test_single_unmasked_slot_zero_entropy(self):
        """With exactly one valid slot, entropy must be 0 (forced choice)."""
        logits = torch.randn(1, 5)
        mask   = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        mc = MaskedCategorical(logits, mask)
        assert float(mc.entropy()) == pytest.approx(0.0, abs=1e-5)

    def test_single_unmasked_slot_prob_is_one(self):
        logits = torch.randn(1, 5)
        mask   = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        mc = MaskedCategorical(logits, mask)
        probs = mc.probs()
        assert float(probs[0, 2]) == pytest.approx(1.0, abs=1e-5)

    def test_multiple_real_players_all_compete(self):
        """All real slots should have non-zero probability."""
        logits = torch.zeros(1, 5)  # equal logits -> equal probs
        mask   = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        mc = MaskedCategorical(logits, mask)
        probs = mc.probs()
        for i in range(3):
            assert float(probs[0, i]) == pytest.approx(1/3, abs=1e-4)

    def test_mode_is_highest_prob_unmasked(self):
        # Put massive logit on slot 1 (which is unmasked)
        logits = torch.tensor([[0.0, 100.0, 0.0, 0.0, 0.0]])
        mask   = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        mc = MaskedCategorical(logits, mask)
        assert int(mc.mode()) == 1

    def test_mode_ignores_masked_large_logit(self):
        # Massive logit on slot 3 (masked) -> should not win
        logits = torch.tensor([[0.0, 1.0, 0.0, 100.0, 0.0]])
        mask   = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        mc = MaskedCategorical(logits, mask)
        assert int(mc.mode()) == 1  # slot 1 has the largest unmasked logit

    def test_batch_log_prob_no_nan(self):
        batch = 4
        total = 21
        logits = torch.randn(batch, total)
        mask   = torch.zeros(batch, total)
        for i in range(batch):
            mask[i, :3] = 1.0  # 3 real players per row
        mc = MaskedCategorical(logits, mask)
        s = mc.sample()
        lp = mc.log_prob(s)
        assert not torch.any(torch.isnan(lp))

    def test_masked_logit_high_doesnt_win(self):
        """Even with the highest logit, a masked slot gets 0 probability."""
        logits = torch.tensor([[0.0, 0.0, 99999.0, 0.0, 0.0]])
        mask   = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]])  # slot 2 masked
        mc = MaskedCategorical(logits, mask)
        probs = mc.probs()
        assert float(probs[0, 2]) == pytest.approx(0.0, abs=1e-7)
        # All other slots compete normally
        assert float(probs[0, :].sum()) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# SquashedNormalHead
# ---------------------------------------------------------------------------

class TestSquashedNormalHead:

    def test_sigmoid_squash_within_bounds(self):
        mean = torch.randn(10, 1)
        log_std = torch.zeros(10, 1)
        low, high = 1.0, 4.0
        head = SquashedNormalHead(mean, log_std, low=low, high=high, squash="sigmoid")
        for _ in range(20):
            raw = head.sample_raw()
            phys = head.to_physical(raw)
            assert torch.all(phys >= low - 1e-5)
            assert torch.all(phys <= high + 1e-5)

    def test_tanh_squash_within_bounds(self):
        mean = torch.randn(10, 1)
        log_std = torch.zeros(10, 1)
        low, high = -5.0, 5.0
        head = SquashedNormalHead(mean, log_std, low=low, high=high, squash="tanh")
        for _ in range(20):
            raw = head.sample_raw()
            phys = head.to_physical(raw)
            assert torch.all(phys >= low - 1e-5)
            assert torch.all(phys <= high + 1e-5)

    def test_log_prob_finite(self):
        mean = torch.randn(4, 2)
        log_std = torch.zeros(4, 2)
        head = SquashedNormalHead(mean, log_std, low=0.0, high=1.0)
        raw = head.sample_raw()
        lp = head.log_prob(raw)
        assert torch.all(torch.isfinite(lp))

    def test_log_prob_no_nan(self):
        mean = torch.zeros(8, 2)
        log_std = torch.zeros(8, 2)
        head = SquashedNormalHead(mean, log_std, low=0.0, high=1.0)
        raw = head.sample_raw()
        lp = head.log_prob(raw)
        assert not torch.any(torch.isnan(lp))

    def test_mode_physical_within_bounds(self):
        for _ in range(10):
            mean = torch.randn(1, 1) * 5
            log_std = torch.zeros(1, 1)
            head = SquashedNormalHead(mean, log_std, low=1.0, high=4.0)
            phys = head.mode_physical()
            assert float(phys) >= 1.0 - 1e-5
            assert float(phys) <= 4.0 + 1e-5

    def test_entropy_positive(self):
        head = SquashedNormalHead(torch.zeros(1, 1), torch.zeros(1, 1), 0.0, 1.0)
        assert float(head.entropy()) > 0.0

    def test_large_log_std_clamped(self):
        """log_std is clamped to [-5, 2] to avoid numerical explosion."""
        head = SquashedNormalHead(
            torch.zeros(1, 1),
            torch.tensor([[999.0]]),  # way outside clamp range
            low=0.0, high=1.0,
        )
        raw = head.sample_raw()
        lp = head.log_prob(raw)
        assert torch.isfinite(lp)

    def test_small_log_std_clamped(self):
        head = SquashedNormalHead(
            torch.zeros(1, 1),
            torch.tensor([[-999.0]]),  # way outside clamp range
            low=0.0, high=1.0,
        )
        raw = head.sample_raw()
        lp = head.log_prob(raw)
        assert torch.isfinite(lp)


# ---------------------------------------------------------------------------
# DirectionHead
# ---------------------------------------------------------------------------

class TestDirectionHead:

    def test_to_physical_is_unit_vector(self):
        raw = torch.randn(4, 2)
        head = DirectionHead(raw, torch.zeros(2))
        for _ in range(20):
            sampled = head.sample_raw()
            phys = head.to_physical(sampled)
            norm = phys.norm(dim=-1)
            for n in norm:
                # to_physical uses eps=1e-6 guard; for small-norm samples the
                # deviation from 1.0 is eps/norm, which can reach ~2e-5.
                assert float(n) == pytest.approx(1.0, abs=1e-4)

    def test_mode_physical_is_unit_vector(self):
        for _ in range(10):
            raw = torch.randn(1, 2)
            head = DirectionHead(raw, torch.zeros(2))
            phys = head.mode_physical()
            norm = float(phys.norm())
            assert norm == pytest.approx(1.0, abs=1e-4)

    def test_log_prob_finite(self):
        raw = torch.randn(4, 2)
        head = DirectionHead(raw, torch.zeros(2))
        sampled = head.sample_raw()
        lp = head.log_prob(sampled)
        assert torch.all(torch.isfinite(lp))

    def test_log_prob_no_nan(self):
        raw = torch.randn(8, 2)
        head = DirectionHead(raw, torch.zeros(2))
        sampled = head.sample_raw()
        lp = head.log_prob(sampled)
        assert not torch.any(torch.isnan(lp))

    def test_entropy_positive(self):
        raw = torch.zeros(1, 2)
        head = DirectionHead(raw, torch.zeros(2))
        assert float(head.entropy()) > 0.0

    def test_near_zero_vector_does_not_explode(self):
        """L2-norm with eps guard: near-zero raw vector should still give a
        valid unit vector (not NaN or Inf)."""
        raw = torch.tensor([[1e-9, 1e-9]])
        head = DirectionHead(raw, torch.zeros(2))
        phys = head.to_physical(raw)
        assert torch.all(torch.isfinite(phys))
        # Should be approximately unit (epsilon prevents div-by-zero)
        assert float(phys.norm()) <= 1.0 + 1e-3

"""Coverage for ``_von_mises_kl`` (ai/action/distributions.py) -- the
hand-derived closed-form KL(vM(mu1,kappa1) || vM(mu2,kappa2)), needed
because ``torch.distributions`` has no ``kl_divergence`` registered for
``VonMises``. Written carefully per the user's explicit "this is genuinely
new math, be careful" request when this was added for
``PPOTrainer.ppg_value_refit()``'s KL-anchor penalty (move_dir/kick_dir).

Checked here, independent of any trainer/network machinery:
1. KL(p, p) == 0 for varied (mu, kappa).
2. Non-negativity under randomized parameter sampling (a real KL divergence
   between two proper distributions can never be negative).
3. Cross-check against a Monte-Carlo mean(log p1(x) - log p2(x)) estimate,
   for a few concrete parameter sets, within MC tolerance -- this validates
   the formula against ground truth, not just internal consistency.
4. Finite/no-NaN at very large and very small kappa (the numerically
   unstable regime _von_mises_entropy's own i0e/i1e trick exists to avoid).
5. Asymmetry: KL(p||q) != KL(q||p) in general -- a sanity check that the
   formula isn't accidentally symmetric (which would indicate a bug, since
   von Mises KL genuinely is asymmetric).
"""
from __future__ import annotations

import math

import pytest
import torch

from footballcoach.ai.action.distributions import _von_mises_kl


class TestVonMisesKLIdentity:
    @pytest.mark.parametrize("mu", [0.0, 1.3, -2.7, math.pi - 0.01, -math.pi + 0.01])
    @pytest.mark.parametrize("kappa", [0.01, 0.5, 1.0, 5.0, 50.0, 500.0])
    def test_kl_of_identical_distributions_is_zero(self, mu, kappa):
        mu_t = torch.tensor(mu)
        kappa_t = torch.tensor(kappa)
        kl = _von_mises_kl(mu_t, kappa_t, mu_t, kappa_t)
        assert kl.item() == pytest.approx(0.0, abs=1e-4)


class TestVonMisesKLNonNegativity:
    def test_kl_is_never_negative_random_params(self):
        torch.manual_seed(0)
        n = 500
        mu1 = (torch.rand(n) * 2 - 1) * math.pi
        mu2 = (torch.rand(n) * 2 - 1) * math.pi
        # log-uniform over a wide realistic kappa range
        kappa1 = torch.exp(torch.rand(n) * 12 - 2)  # ~[0.13, 160000]
        kappa2 = torch.exp(torch.rand(n) * 12 - 2)
        kl = _von_mises_kl(mu1, kappa1, mu2, kappa2)
        assert torch.isfinite(kl).all()
        assert (kl >= -1e-5).all(), f"found negative KL: min={kl.min().item()}"


class TestVonMisesKLMonteCarloCrossCheck:
    @pytest.mark.parametrize(
        "mu1,kappa1,mu2,kappa2",
        [
            (0.0, 3.0, 0.0, 3.0),      # identical
            (0.0, 3.0, 1.0, 3.0),      # same kappa, mean shifted
            (0.0, 3.0, 0.0, 8.0),      # same mean, kappa increased (narrower)
            (0.0, 8.0, 0.0, 3.0),      # same mean, kappa decreased (wider)
            (0.5, 2.0, -0.5, 6.0),     # both differ
        ],
    )
    def test_matches_monte_carlo_estimate(self, mu1, kappa1, mu2, kappa2):
        torch.manual_seed(42)
        mu1_t, kappa1_t = torch.tensor(mu1), torch.tensor(kappa1)
        mu2_t, kappa2_t = torch.tensor(mu2), torch.tensor(kappa2)
        closed_form = _von_mises_kl(mu1_t, kappa1_t, mu2_t, kappa2_t).item()

        p = torch.distributions.VonMises(mu1_t, kappa1_t)
        q = torch.distributions.VonMises(mu2_t, kappa2_t)
        samples = p.sample((200_000,))
        mc_estimate = (p.log_prob(samples) - q.log_prob(samples)).mean().item()

        # Monte Carlo estimate of a KL this size has real variance -- a
        # generous absolute+relative tolerance, not exact equality.
        assert closed_form == pytest.approx(mc_estimate, abs=0.02, rel=0.05)

    def test_zero_kl_case_matches_monte_carlo_near_zero(self):
        torch.manual_seed(7)
        mu_t, kappa_t = torch.tensor(1.0), torch.tensor(4.0)
        closed_form = _von_mises_kl(mu_t, kappa_t, mu_t, kappa_t).item()
        p = torch.distributions.VonMises(mu_t, kappa_t)
        samples = p.sample((200_000,))
        mc_estimate = (p.log_prob(samples) - p.log_prob(samples)).mean().item()
        assert closed_form == pytest.approx(0.0, abs=1e-4)
        assert mc_estimate == 0.0


class TestVonMisesKLNumericalStability:
    def test_finite_at_very_large_kappa(self):
        mu1, mu2 = torch.tensor(0.0), torch.tensor(0.3)
        for kappa_val in (1e3, 1e4, 1e6, 1e8):
            kl = _von_mises_kl(mu1, torch.tensor(kappa_val), mu2, torch.tensor(kappa_val))
            assert torch.isfinite(kl).all(), f"non-finite at kappa={kappa_val}"

    def test_finite_at_very_small_kappa(self):
        mu1, mu2 = torch.tensor(0.0), torch.tensor(1.5)
        for kappa_val in (1e-6, 1e-4, 1e-2):
            kl = _von_mises_kl(mu1, torch.tensor(kappa_val), mu2, torch.tensor(kappa_val))
            assert torch.isfinite(kl).all(), f"non-finite at kappa={kappa_val}"

    def test_broadcasts_global_kappa_against_per_row_mu(self):
        # Mirrors real usage: mu is per-row (batch,), kappa is a global
        # scalar-shaped nn.Parameter, e.g. shape (1,).
        mu1 = torch.tensor([0.0, 1.0, -1.0, 2.5])
        mu2 = torch.tensor([0.1, 0.9, -1.2, 2.4])
        kappa1 = torch.tensor([3.0])
        kappa2 = torch.tensor([5.0])
        kl = _von_mises_kl(mu1, kappa1, mu2, kappa2)
        assert kl.shape == (4,)
        assert torch.isfinite(kl).all()
        assert (kl >= -1e-5).all()


class TestVonMisesKLAsymmetry:
    def test_kl_is_not_symmetric_in_general(self):
        mu1, kappa1 = torch.tensor(0.0), torch.tensor(2.0)
        mu2, kappa2 = torch.tensor(0.4), torch.tensor(9.0)
        kl_pq = _von_mises_kl(mu1, kappa1, mu2, kappa2).item()
        kl_qp = _von_mises_kl(mu2, kappa2, mu1, kappa1).item()
        assert kl_pq != pytest.approx(kl_qp, rel=1e-3)

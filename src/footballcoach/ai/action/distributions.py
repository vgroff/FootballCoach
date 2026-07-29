"""Action distribution wrappers for PPO.

Provides a consistent sample()/log_prob()/entropy() interface across the
three kinds of action heads used in this project:
  - IndependentBernoulli: sigmoid action-probability heads (shoot, pass, etc.)
  - MaskedCategorical: softmax over a fixed-size slot set with -inf masking
  - SquashedNormalHead: continuous heads (move target, kick power, etc.)

See ai_design_doc.md sections 8.3-8.5 for full rationale.

All classes operate on PyTorch tensors and require torch to be importable.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Independent Bernoulli (sigmoid heads)
# ---------------------------------------------------------------------------

class IndependentBernoulli:
    """Thin wrapper around torch.distributions.Bernoulli.

    Built from raw logits (numerically more stable than sigmoid-then-prob).
    Each element of the input tensor is an independent Bernoulli variable.
    ``log_prob`` sums over all elements (treats the vector as a single joint
    factorized action).
    """

    def __init__(self, logits: torch.Tensor):
        self.dist = torch.distributions.Bernoulli(logits=logits)

    def sample(self) -> torch.Tensor:
        return self.dist.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Elementwise log_prob; does NOT sum here so callers can gate/mask
        individual terms before summing (e.g. only include target-head
        log_prob when the intent was 1)."""
        return self.dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()

    def prob(self) -> torch.Tensor:
        return torch.sigmoid(self.dist.logits)

    def mode(self) -> torch.Tensor:
        return (self.prob() >= 0.5).float()


# ---------------------------------------------------------------------------
# Masked Categorical (pass/tackle/mark target heads)
# ---------------------------------------------------------------------------

class MaskedCategorical:
    """Categorical distribution over MAX_OTHER_PLAYERS slots, masking out
    non-existent players via -inf before softmax.

    See ai_design_doc.md sections 2.4 and 8.3.  ``exists_mask`` is 1.0 for
    a real player slot, 0.0 for a padded/absent slot.

    IMPORTANT: if all slots for a given batch row are masked (exists_mask all
    zeros), do NOT call sample()/log_prob() for that row - torch.distributions
    .Categorical will produce NaN (softmax over all -inf is undefined).  The
    caller must guard for this case (e.g. skip the target log_prob term when
    there are zero valid targets).
    """

    def __init__(self, logits: torch.Tensor, exists_mask: torch.Tensor):
        # exists_mask: (batch, MAX_OTHER_PLAYERS) or (MAX_OTHER_PLAYERS,)
        masked_logits = logits.masked_fill(exists_mask < 0.5, float("-inf"))
        self.dist = torch.distributions.Categorical(logits=masked_logits)

    def sample(self) -> torch.Tensor:
        return self.dist.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()

    def mode(self) -> torch.Tensor:
        return self.dist.probs.argmax(dim=-1)

    def probs(self) -> torch.Tensor:
        return self.dist.probs


# ---------------------------------------------------------------------------
# Squashed Normal (continuous heads: move target, region, kick power, etc.)
# ---------------------------------------------------------------------------

class SquashedNormalHead:
    """Gaussian (Normal) policy head with post-sample squashing.

    The network outputs a raw (unbounded) mean; log_std is a separate
    parameter (global scalar nn.Parameter, NOT state-dependent, as the
    simplest stable starting point - see ai_design_doc.md section 8.5).

    PPO log_prob is computed on the *pre-squash* raw sample (the squashing
    Jacobian correction is intentionally omitted as a known minor
    approximation per section 8.5 - common in practice).  The ``to_physical``
    method applies sigmoid/tanh + linear rescaling to map to the physical
    output range for actually driving the engine.

    Args:
        mean: network output mean tensor (..., dim)
        log_std: tensor or nn.Parameter (..., dim) or scalar
        low: physical output lower bound (for sigmoid-squashed heads)
        high: physical output upper bound
        squash: 'sigmoid' or 'tanh' (tanh maps to [-1,1] before rescaling)
    """

    def __init__(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        low: float,
        high: float,
        squash: str = "sigmoid",
    ):
        assert squash in ("sigmoid", "tanh"), f"Unknown squash mode: {squash}"
        self.low = low
        self.high = high
        self.squash = squash
        # Clamp log_std for numerical stability (from CleanRL convention)
        std = torch.exp(log_std.clamp(-5.0, 2.0))
        self.dist = torch.distributions.Normal(mean, std)

    def sample_raw(self) -> torch.Tensor:
        """Sample from the unsquashed Gaussian (used for PPO log_prob)."""
        return self.dist.rsample()  # reparameterized for potential use in DDPG-style updates

    def log_prob(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Log prob of the pre-squash sample (summed over the last dimension)."""
        return self.dist.log_prob(raw_action).sum(dim=-1)

    def to_physical(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Apply squashing + linear scale to get physical-range values."""
        if self.squash == "sigmoid":
            squashed = torch.sigmoid(raw_action)
        else:  # tanh
            squashed = (torch.tanh(raw_action) + 1.0) / 2.0  # maps tanh(-1,1) to (0,1)
        return self.low + squashed * (self.high - self.low)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy().sum(dim=-1)

    def mode_physical(self) -> torch.Tensor:
        return self.to_physical(self.dist.mean)


# ---------------------------------------------------------------------------
# Direction head (unit-vector output, used for move_direction / kick_direction)
# ---------------------------------------------------------------------------

class DirectionHead:
    """2D unit-vector output with isotropic 2D Gaussian PPO log_prob.

    The network outputs a raw 2D vector; we L2-normalize it for the physical
    direction (avoids angle-wraparound discontinuity) and treat each
    component as an independent Normal for PPO purposes (same convention as
    SquashedNormalHead but without squash, just normalize-to-unit).

    See ai_design_doc.md section 8.6.
    """

    def __init__(self, raw_vector: torch.Tensor, log_std: torch.Tensor,
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        """
        Args:
            raw_vector: (..., 2) raw 2D vector from the network.
            log_std: (..., 2) or scalar log standard deviation.
            log_std_min/max: clamp bounds (configurable; defaults match original hardcoded values).
        """
        std = torch.exp(log_std.clamp(log_std_min, log_std_max))
        self.dist = torch.distributions.Normal(raw_vector, std)
        self._raw = raw_vector

    def sample_raw(self) -> torch.Tensor:
        """Sample from the 2D Gaussian (pre-normalization)."""
        return self.dist.rsample()

    def log_prob(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Sum log_prob over the 2 components."""
        return self.dist.log_prob(raw_action).sum(dim=-1)

    def to_physical(self, raw_action: torch.Tensor) -> torch.Tensor:
        """L2-normalize the raw vector to a unit direction."""
        eps = 1e-6
        return raw_action / (raw_action.norm(dim=-1, keepdim=True) + eps)

    def mode_physical(self) -> torch.Tensor:
        return self.to_physical(self._raw)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy().sum(dim=-1)

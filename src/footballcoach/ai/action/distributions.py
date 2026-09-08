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
        invalid = exists_mask < 0.5
        # Guard the "zero valid targets" edge case (e.g. UI training mode's
        # 1-player-only match, no other players at all): softmax over an
        # all -inf row is NaN. Leave those rows fully unmasked instead -- the
        # resulting sample is meaningless but finite; callers must not act on
        # a target slot when there were zero real other players to begin with
        # (see class docstring / apply_nn_action.py's slot_player_ids=None
        # guard, which already makes such a slot resolve to no-op).
        fully_masked_rows = invalid.all(dim=-1, keepdim=True)
        invalid = invalid & ~fully_masked_rows
        masked_logits = logits.masked_fill(invalid, float("-inf"))
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
# Direction heads (unit-vector output, used for move_direction / kick_direction)
# ---------------------------------------------------------------------------

def _von_mises_entropy(kappa: torch.Tensor) -> torch.Tensor:
    """Closed-form entropy of a von Mises distribution with concentration kappa.

    H(kappa) = log(2*pi*I0(kappa)) - kappa*I1(kappa)/I0(kappa).

    torch.distributions.VonMises has no .entropy() (confirmed directly against
    the installed torch build: 'entropy' not in VonMises.__dict__). Computed
    via the exponentially-scaled Bessel functions i0e/i1e = I0/I1 * exp(-kappa)
    (torch.special) rather than raw i0/i1, since I0(kappa) itself overflows
    float32 for kappa beyond roughly 90 -- i0e/i1e stay well-conditioned across
    the whole realistic kappa range and the exp(-kappa) factor cancels exactly
    in both the log(I0) term (log(i0e)+kappa) and the I1/I0 ratio (i1e/i0e).
    """
    i0e = torch.special.i0e(kappa)
    i1e = torch.special.i1e(kappa)
    log_two_pi = 1.8378770664093453  # log(2*pi)
    return log_two_pi + torch.log(i0e) + kappa - kappa * (i1e / i0e)


class VonMisesDirectionHead:
    """2D unit-vector output with a true von Mises (circular) PPO log_prob.

    Replaces the older isotropic-Gaussian-on-chordal-distance approximation
    (see DirectionHead below, kept only for any external callers/tests still
    referencing it directly) with the actual circular distribution for a 2D
    direction: mean angle from the network's raw output, concentration kappa
    in place of a standard deviation (larger kappa = narrower/more confident,
    the INVERSE relationship of log_std -- see ai_trainer_knowledge.md).

    atan2 is scale-invariant, so no explicit L2-normalize of raw_vector is
    needed to get the mean angle (unlike DirectionHead's normalize-then-Normal
    approach) -- the caller already passes an L2-normalized vector regardless
    (ExecutionNetwork.forward()'s own output), so this is purely a simplification,
    not a behavior difference.
    """

    def __init__(self, raw_vector: torch.Tensor, log_kappa: torch.Tensor,
                 log_kappa_min: float = -2.0, log_kappa_max: float = 10.0):
        """
        Args:
            raw_vector: (..., 2) vector from the network (need not be unit norm).
            log_kappa: (..., 1) or scalar log concentration.
            log_kappa_min/max: clamp bounds for log_kappa.
        """
        self._theta_mean = torch.atan2(raw_vector[..., 1], raw_vector[..., 0])
        self._kappa = torch.exp(log_kappa.clamp(log_kappa_min, log_kappa_max))
        self.dist = torch.distributions.VonMises(self._theta_mean, self._kappa)

    def sample_raw(self) -> torch.Tensor:
        """Sample an angle and return it as a unit (cos, sin) vector."""
        theta = self.dist.sample()
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    def log_prob(self, raw_action: torch.Tensor) -> torch.Tensor:
        """log_prob of a stored (x, y) unit-vector action, angle-only."""
        theta = torch.atan2(raw_action[..., 1], raw_action[..., 0])
        return self.dist.log_prob(theta)

    def to_physical(self, raw_action: torch.Tensor) -> torch.Tensor:
        """L2-normalize the raw vector to a unit direction (safety net)."""
        eps = 1e-6
        return raw_action / (raw_action.norm(dim=-1, keepdim=True) + eps)

    def mode_physical(self) -> torch.Tensor:
        return torch.stack([torch.cos(self._theta_mean), torch.sin(self._theta_mean)], dim=-1)

    def entropy(self) -> torch.Tensor:
        return _von_mises_entropy(self._kappa)


class KickDirectionHead:
    """3D unit-vector output for kick_direction: azimuthal angle (von Mises)
    x elevation z (plain unconstrained Normal), NOT a full spherical (von
    Mises-Fisher) distribution -- deliberately, since VMF has no PyTorch
    built-in and needs a nontrivial custom rejection sampler.

    z has no bounded range and needs no squashing: the reconstruction
    v = (cos(theta), sin(theta), z) / ||v|| is a valid unit vector for ANY
    real z (z=0 -> pure in-plane kick, z -> +-inf -> approaches vertical).
    This is not a true rotationally-symmetric spherical distribution (it
    privileges the z axis), which is an accepted, deliberate approximation --
    physically appropriate for a kick, where elevation is already a distinct,
    asymmetric axis from the two in-plane directions. log_prob is scored
    directly in (theta, z) space with no embedding-Jacobian correction, the
    same class of approximation SquashedNormalHead already makes for its own
    squash correction ("intentionally omitted... common in practice").
    """

    def __init__(self, raw_vector: torch.Tensor, log_kappa: torch.Tensor,
                 log_std_z: torch.Tensor,
                 log_kappa_min: float = -2.0, log_kappa_max: float = 10.0,
                 log_std_z_min: float = -5.0, log_std_z_max: float = 2.0):
        """
        Args:
            raw_vector: (..., 3) vector from the network.
            log_kappa: (..., 1) or scalar log concentration for the azimuthal component.
            log_std_z: (..., 1) or scalar log std for the elevation component.
            log_kappa_min/max, log_std_z_min/max: clamp bounds.
        """
        raw_xy = raw_vector[..., :2]
        raw_z = raw_vector[..., 2]
        self._theta_mean = torch.atan2(raw_xy[..., 1], raw_xy[..., 0])
        self._kappa = torch.exp(log_kappa.clamp(log_kappa_min, log_kappa_max))
        self._mean_z = raw_z
        self._std_z = torch.exp(log_std_z.clamp(log_std_z_min, log_std_z_max))
        self.dist_xy = torch.distributions.VonMises(self._theta_mean, self._kappa)
        self.dist_z = torch.distributions.Normal(self._mean_z, self._std_z)

    @staticmethod
    def _reconstruct(theta: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        v = torch.stack([torch.cos(theta), torch.sin(theta), z], dim=-1)
        return v / (v.norm(dim=-1, keepdim=True) + eps)

    def sample_raw(self) -> torch.Tensor:
        theta = self.dist_xy.sample()
        z = self.dist_z.sample()
        return self._reconstruct(theta, z)

    def log_prob(self, raw_action: torch.Tensor) -> torch.Tensor:
        """log_prob of a stored (x, y, z) unit-vector action.

        Inverts the reconstruction: theta = atan2(y, x) (unaffected by z,
        the sqrt(1+z^2) scale factor cancels out of atan2); z is recovered
        via z = a_z / sqrt(1 - a_z^2), eps-guarded against the pole
        singularity at a_z = +-1 (a measure-zero case in practice).
        """
        eps = 1e-6
        a_x, a_y, a_z = raw_action[..., 0], raw_action[..., 1], raw_action[..., 2]
        theta = torch.atan2(a_y, a_x)
        z = a_z / torch.sqrt((1.0 - a_z * a_z).clamp(min=eps))
        return self.dist_xy.log_prob(theta) + self.dist_z.log_prob(z)

    def to_physical(self, raw_action: torch.Tensor) -> torch.Tensor:
        """L2-normalize the raw vector to a unit direction (safety net)."""
        eps = 1e-6
        return raw_action / (raw_action.norm(dim=-1, keepdim=True) + eps)

    def mode_physical(self) -> torch.Tensor:
        return self._reconstruct(self._theta_mean, self._mean_z)

    def entropy(self) -> torch.Tensor:
        return _von_mises_entropy(self._kappa) + self.dist_z.entropy()


class DirectionHead:
    """2D unit-vector output with isotropic 2D Gaussian PPO log_prob.

    Superseded by VonMisesDirectionHead/KickDirectionHead above for
    move_direction/kick_direction (see ai_trainer_knowledge.md) -- kept here
    unused by ppo_trainer.py itself in case any external caller/test still
    wants the older chordal-distance approximation.

    The network outputs a raw 2D vector; we L2-normalize it for the physical
    direction (avoids angle-wraparound discontinuity) and treat each
    component as an independent Normal for PPO purposes (same convention as
    SquashedNormalHead but without squash, just normalize-to-unit).
    """

    def __init__(self, raw_vector: torch.Tensor, log_std: torch.Tensor,
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        """
        Args:
            raw_vector: (..., 2) raw 2D vector from the network.
            log_std: (..., 2) or scalar log standard deviation.
            log_std_min/max: clamp bounds (configurable; defaults match original hardcoded values).
        """
        eps = 1e-6
        # Normalize to unit vector so the Gaussian mean stays on the unit circle.
        # Prevents raw_vector magnitude drift from inflating KL between rollout
        # collection and PPO update (the "stored_norm >> 1" bug).
        normalized = raw_vector / (raw_vector.norm(dim=-1, keepdim=True) + eps)
        std = torch.exp(log_std.clamp(log_std_min, log_std_max))
        self.dist = torch.distributions.Normal(normalized, std)
        self._raw = normalized

    def sample_raw(self) -> torch.Tensor:
        """Sample from the 2D Gaussian, normalized so log_prob is angle-only."""
        raw = self.dist.rsample()
        eps = 1e-6
        return raw / (raw.norm(dim=-1, keepdim=True) + eps)

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

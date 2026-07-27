"""RNG-reduction helpers.

`rng_reduction` is a global game option in [0, 1] (default 0.3) that dials
down randomness for players who want a more skill-deterministic game:

- Skill-check rolls: roll = (rng_reduction + (1 - rng_reduction) * U(0,1)) * skill
  At rng_reduction=1.0 the roll degenerates to exactly `skill` (no randomness).
  At rng_reduction=0.0 it's a plain uniform roll scaled by skill.
- Gaussian errors: sigma_effective = sigma * (1 - rng_reduction)
  At rng_reduction=1.0, sigma becomes 0 (perfectly deterministic aim/kicks).
"""
from __future__ import annotations

import random


def skill_roll(skill: float, rng_reduction: float, rng: random.Random | None = None) -> float:
    """Returns a randomized roll of `skill`, dampened by `rng_reduction`."""
    r = rng or random
    u = r.random()
    return (rng_reduction + (1.0 - rng_reduction) * u) * skill


def reduced_sigma(sigma: float, rng_reduction: float) -> float:
    """Returns the effective Gaussian sigma after applying rng_reduction."""
    return sigma * (1.0 - rng_reduction)

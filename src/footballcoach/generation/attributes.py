"""Generates PlayerAttributes from a correlated multivariate Gaussian.

See knowledge.md in this package for the full explanation of the approach.
"""
from __future__ import annotations

import random

import numpy as np

from footballcoach.config import load_attributes_config
from footballcoach.entities.attributes import PlayerAttributes


def _build_correlation_matrix(order: list[str], correlations: dict[str, float]) -> np.ndarray:
    n = len(order)
    index = {name: i for i, name in enumerate(order)}
    corr = np.eye(n)
    for pair_key, rho in correlations.items():
        if pair_key.startswith("_"):
            continue
        a, b = pair_key.split(":")
        i, j = index[a], index[b]
        corr[i, j] = rho
        corr[j, i] = rho
    return corr


def generate_attributes(
    tier: str = "generic",
    rng: random.Random | np.random.Generator | None = None,
) -> PlayerAttributes:
    """Generates a single player's attributes for the given tier.

    tier: one of the presets in attributes.json["tiers"] (e.g.
    "premier_league", "league_three", "generic").
    """
    cfg = load_attributes_config()
    order = cfg["attribute_order"]
    tier_cfg = cfg["tiers"].get(tier, cfg["default"])
    mean = tier_cfg["mean"]
    sigma = tier_cfg["sigma"]

    corr = _build_correlation_matrix(order, cfg["correlations"])
    cov = corr * (sigma * sigma)

    if isinstance(rng, np.random.Generator):
        generator = rng
    elif isinstance(rng, random.Random):
        # Derive a numpy Generator seeded from the Random instance for
        # reproducibility when a `random.Random` is passed. Note: `rng.randint`
        # mutates the shared `random.Random` stream on every call, so this only
        # gives "same rng -> same attributes" reproducibility for a single call
        # right after a fresh seed - calling this twice with the same `rng`
        # instance will NOT produce the same attributes the second time.
        generator = np.random.default_rng(rng.randint(0, 2**32 - 1))
    else:
        generator = np.random.default_rng()

    sample = generator.multivariate_normal(mean=[mean] * len(order), cov=cov)
    clipped = np.clip(sample, 0.0, 1.0)

    values = {name: float(clipped[i]) for i, name in enumerate(order)}
    return PlayerAttributes(**values)


def generate_squad(
    size: int,
    tier: str = "generic",
    rng: random.Random | np.random.Generator | None = None,
) -> list[PlayerAttributes]:
    """Generates `size` independent players' attributes for the given tier."""
    return [generate_attributes(tier=tier, rng=rng) for _ in range(size)]

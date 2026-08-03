"""Small shared interpolation helpers.

``piecewise_lerp3`` factors out the "3-point piecewise-linear-in-cosine"
shape used by both ``engine/tackling.py`` (angle modifier based on tackle
approach direction) and ``engine/kicking.py`` (precision penalty based on
running direction) — both take a cosine-similarity value and lerp between
three named breakpoints.
"""
from __future__ import annotations


def piecewise_lerp3(
    x: float,
    x_low: float,
    x_mid: float,
    x_high: float,
    y_low: float,
    y_mid: float,
    y_high: float,
) -> float:
    """Piecewise-linear interpolation through three points
    ``(x_low, y_low)``, ``(x_mid, y_mid)``, ``(x_high, y_high)``.

    ``x_low <= x_mid <= x_high`` is assumed. Values of ``x`` outside
    ``[x_low, x_high]`` are clamped to ``y_low``/``y_high`` respectively.
    """
    if x >= x_high:
        return y_high
    if x <= x_low:
        return y_low
    if x >= x_mid:
        t = (x - x_mid) / (x_high - x_mid)
        return y_mid + (y_high - y_mid) * t
    t = (x - x_low) / (x_mid - x_low)
    return y_low + (y_mid - y_low) * t

"""Tackling: an RNG skill-check between tackler.tackling and
ball_carrier.dribbling. See engine/knowledge.md for the analytical
derivation of win probabilities used to validate this against the balance
targets.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from footballcoach.config import load_physics_config
from footballcoach.mathutils.rng import skill_roll


@dataclass(frozen=True)
class TacklingParams:
    tackler_boost: float
    inactive_duration_s: float
    inactive_speed_penalty: float

    @staticmethod
    def from_config() -> "TacklingParams":
        d = load_physics_config()["tackling"]
        return TacklingParams(
            tackler_boost=d["tackler_boost"],
            inactive_duration_s=d["inactive_duration_s"],
            inactive_speed_penalty=d["inactive_speed_penalty"],
        )


def attempt_tackle(
    tackling_attr: float,
    dribbling_attr: float,
    rng_reduction: float,
    rng: random.Random | None = None,
    params: TacklingParams | None = None,
) -> bool:
    """Returns True if the tackle succeeds (tackler wins the ball).

    tackler_roll = (rng_reduction + (1-rng_reduction)*U) * tackler_boost * tackling_attr
    dribbler_roll = (rng_reduction + (1-rng_reduction)*U) * dribbling_attr

    The tackler_boost (default 1.2) favours the defender, per the design
    spec, and can push the roll above 1.0.
    """
    params = params or TacklingParams.from_config()
    r = rng or random

    tackler_roll = skill_roll(tackling_attr * params.tackler_boost, rng_reduction, r)
    dribbler_roll = skill_roll(dribbling_attr, rng_reduction, r)

    return tackler_roll > dribbler_roll

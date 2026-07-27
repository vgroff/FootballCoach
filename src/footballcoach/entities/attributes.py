"""PlayerAttributes: the 8 core skills, each in [0, 1]."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlayerAttributes:
    top_speed: float
    acceleration: float
    stamina: float
    kick_precision: float
    kick_power: float
    dribbling: float
    ball_control: float
    tackling: float

    def __post_init__(self) -> None:
        for name in (
            "top_speed", "acceleration", "stamina", "kick_precision",
            "kick_power", "dribbling", "ball_control", "tackling",
        ):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"attribute {name}={value} out of range [0, 1]")

    @staticmethod
    def average(value: float = 0.5) -> "PlayerAttributes":
        return PlayerAttributes(value, value, value, value, value, value, value, value)

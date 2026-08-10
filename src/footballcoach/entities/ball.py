"""The ball entity: position, velocity, spin, and possession state."""
from __future__ import annotations

from dataclasses import dataclass

from footballcoach.config import load_physics_config
from footballcoach.mathutils import Vector3


@dataclass
class Ball:
    position: Vector3 = None  # type: ignore[assignment]
    velocity: Vector3 = None  # type: ignore[assignment]
    spin: Vector3 = None  # type: ignore[assignment]
    radius_m: float = 0.11
    mass_kg: float = 0.43

    # Player id currently in possession (ball "stuck" to them), or None if loose.
    possessed_by: str | None = None

    # Countdown timer set to just_bounced_display_duration_s whenever the
    # ball makes a genuine bounce (incoming vz exceeds BOUNCE_THRESHOLD_MPS).
    # Decayed by dt each tick in ball_physics.step_ball. Used by the renderer
    # to show a visual "just bounced" indicator. Zero when not recently bounced.
    just_bounced_timer_s: float = 0.0

    def __post_init__(self) -> None:
        if self.position is None:
            self.position = Vector3.zero()
        if self.velocity is None:
            self.velocity = Vector3.zero()
        if self.spin is None:
            self.spin = Vector3.zero()

    @staticmethod
    def at_rest(position: Vector3 | None = None) -> "Ball":
        cfg = load_physics_config()["ball"]
        return Ball(
            position=position or Vector3.zero(),
            velocity=Vector3.zero(),
            spin=Vector3.zero(),
            radius_m=cfg["radius_m"],
            mass_kg=cfg["mass_kg"],
        )

    @property
    def is_loose(self) -> bool:
        return self.possessed_by is None

    @property
    def height_m(self) -> float:
        return self.position.z

    def is_grounded(self, epsilon: float = 1e-6) -> bool:
        return self.position.z <= self.radius_m + epsilon

"""The pitch: dimensions and derived geometry (goal mouths, box, etc.)."""
from __future__ import annotations

from dataclasses import dataclass, field

from footballcoach.config import PitchConfig
from footballcoach.mathutils import Vector3


@dataclass
class Pitch:
    """Represents pitch geometry, centred at the origin.

    x runs from -length/2 (left goal line) to +length/2 (right goal line).
    y runs from -width/2 to +width/2. The left goal is centred at
    (-length/2, 0) and the right goal at (+length/2, 0).
    """

    length_m: float
    width_m: float
    goal_width_m: float
    goal_height_m: float
    goal_depth_m: float
    box_length_m: float
    box_width_m: float
    six_yard_length_m: float
    six_yard_width_m: float
    penalty_spot_distance_m: float
    centre_circle_radius_m: float

    @staticmethod
    def standard() -> "Pitch":
        cfg = PitchConfig.from_config()
        return Pitch(
            length_m=cfg.length_m,
            width_m=cfg.width_m,
            goal_width_m=cfg.goal_width_m,
            goal_height_m=cfg.goal_height_m,
            goal_depth_m=cfg.goal_depth_m,
            box_length_m=cfg.box_length_m,
            box_width_m=cfg.box_width_m,
            six_yard_length_m=cfg.six_yard_length_m,
            six_yard_width_m=cfg.six_yard_width_m,
            penalty_spot_distance_m=cfg.penalty_spot_distance_m,
            centre_circle_radius_m=cfg.centre_circle_radius_m,
        )

    @property
    def half_length(self) -> float:
        return self.length_m / 2.0

    @property
    def half_width(self) -> float:
        return self.width_m / 2.0

    @property
    def left_goal_centre(self) -> Vector3:
        return Vector3(-self.half_length, 0.0, 0.0)

    @property
    def right_goal_centre(self) -> Vector3:
        return Vector3(self.half_length, 0.0, 0.0)

    def penalty_spot(self, *, left: bool) -> Vector3:
        x = -self.half_length + self.penalty_spot_distance_m if left else \
            self.half_length - self.penalty_spot_distance_m
        return Vector3(x, 0.0, 0.0)

    def is_in_box(self, position: Vector3, *, left: bool) -> bool:
        half_box_w = self.box_width_m / 2.0
        if left:
            in_x = -self.half_length <= position.x <= -self.half_length + self.box_length_m
        else:
            in_x = self.half_length - self.box_length_m <= position.x <= self.half_length
        in_y = -half_box_w <= position.y <= half_box_w
        return in_x and in_y

    def is_in_either_box(self, position: Vector3) -> bool:
        return self.is_in_box(position, left=True) or self.is_in_box(position, left=False)

    def is_goal(self, position: Vector3) -> str | None:
        """Returns 'left' or 'right' if position is within that goal mouth, else None."""
        half_goal_w = self.goal_width_m / 2.0
        if abs(position.y) > half_goal_w or position.z > self.goal_height_m:
            return None
        if position.x <= -self.half_length:
            return "left"
        if position.x >= self.half_length:
            return "right"
        return None

    def is_in_bounds(self, position: Vector3) -> bool:
        return (
            -self.half_length <= position.x <= self.half_length
            and -self.half_width <= position.y <= self.half_width
        )

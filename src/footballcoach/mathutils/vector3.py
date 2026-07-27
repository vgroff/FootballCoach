"""A tiny, fast 3D vector wrapper around numpy for the football engine.

Convention: x = pitch length axis (goal-to-goal), y = pitch width axis,
z = height above ground. All units are metres / metres-per-second /
metres-per-second-squared as appropriate. This keeps every physics
computation in SI units per the project's design goals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Vector3:
    x: float
    y: float
    z: float = 0.0

    @staticmethod
    def zero() -> "Vector3":
        return Vector3(0.0, 0.0, 0.0)

    @staticmethod
    def from_array(arr: np.ndarray) -> "Vector3":
        return Vector3(float(arr[0]), float(arr[1]), float(arr[2]))

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Vector3":
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vector3":
        return Vector3(-self.x, -self.y, -self.z)

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def length_xy(self) -> float:
        """Magnitude ignoring the height (z) component - useful for ground-plane speed."""
        return math.hypot(self.x, self.y)

    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> "Vector3":
        length = self.length()
        if length < 1e-12:
            return Vector3.zero()
        return self / length

    def with_z(self, z: float) -> "Vector3":
        return Vector3(self.x, self.y, z)

    def xy(self) -> "Vector3":
        return Vector3(self.x, self.y, 0.0)

    def angle_xy(self) -> float:
        """Heading angle (radians) of the xy-projection, measured from +x axis."""
        return math.atan2(self.y, self.x)

    @staticmethod
    def from_angle_xy(angle_rad: float, magnitude: float = 1.0) -> "Vector3":
        return Vector3(math.cos(angle_rad) * magnitude, math.sin(angle_rad) * magnitude, 0.0)

    def distance_to(self, other: "Vector3") -> float:
        return (self - other).length()

"""Maps between world coordinates (metres, engine convention: x = pitch
length axis, y = pitch width axis, origin at pitch centre) and screen pixel
coordinates for a top-down 2D view.

The window is auto-sized to fit the full pitch plus a margin, at a fixed
pixels-per-metre scale, per the project's UI requirements.
"""
from __future__ import annotations

from dataclasses import dataclass

from footballcoach.entities.pitch import Pitch


@dataclass(frozen=True)
class Camera:
    pitch: Pitch
    pixels_per_metre: float
    margin_px: int

    @property
    def screen_width(self) -> int:
        return int(self.pitch.length_m * self.pixels_per_metre) + 2 * self.margin_px

    @property
    def screen_height(self) -> int:
        return int(self.pitch.width_m * self.pixels_per_metre) + 2 * self.margin_px

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """World (x=length axis, y=width axis, origin at pitch centre) to
        screen pixels (origin top-left, y grows downward)."""
        screen_x = self.margin_px + (x + self.pitch.half_length) * self.pixels_per_metre
        screen_y = self.margin_px + (self.pitch.half_width - y) * self.pixels_per_metre
        return int(screen_x), int(screen_y)

    def screen_to_world(self, screen_x: float, screen_y: float) -> tuple[float, float]:
        """Inverse of world_to_screen - used to translate mouse clicks into
        world positions."""
        x = (screen_x - self.margin_px) / self.pixels_per_metre - self.pitch.half_length
        y = self.pitch.half_width - (screen_y - self.margin_px) / self.pixels_per_metre
        return x, y

    def scale_length(self, length_m: float) -> int:
        """Converts a world-space length (e.g. a radius) to pixels."""
        return max(1, int(length_m * self.pixels_per_metre))

    @staticmethod
    def fit_to_pitch(pitch: Pitch, pixels_per_metre: float = 9.0, margin_px: int = 40) -> "Camera":
        return Camera(pitch=pitch, pixels_per_metre=pixels_per_metre, margin_px=margin_px)

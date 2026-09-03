"""Maps between world coordinates (metres, engine convention: x = pitch
length axis, y = pitch width axis, origin at pitch centre) and screen pixel
coordinates for a top-down 2D view.

The window is resizable (see ui/app.py's pygame.RESIZABLE handling):
pixels_per_metre is DERIVED from the actual window size via resize()
-- fit the whole pitch (plus margin) into whatever space is available,
preserving aspect ratio, and centre it in any leftover space on the other
axis -- rather than the window size being a fixed function of a configured
zoom level. The config's camera.pixels_per_metre is only the STARTING
window size's basis (see App.__init__); after that, resizing the window
(including OS-level maximise) is what changes zoom, not the other way
around.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from footballcoach.entities.pitch import Pitch


@dataclass
class Camera:
    pitch: Pitch
    pixels_per_metre: float
    margin_px: int
    # Actual window/surface size and the pixel offset of the pitch's
    # top-left corner within it -- both recomputed by resize() whenever the
    # window changes size. Defaults here match the pre-resize behaviour
    # (pitch pinned at (margin_px, margin_px), window exactly pitch+margin)
    # for any caller that never calls resize() at all (e.g. a headless/test
    # Camera built directly via the dataclass constructor).
    screen_width: int = 0
    screen_height: int = 0
    offset_x: float = field(default=0.0)
    offset_y: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.screen_width == 0 and self.screen_height == 0:
            self.screen_width = int(self.pitch.length_m * self.pixels_per_metre) + 2 * self.margin_px
            self.screen_height = int(self.pitch.width_m * self.pixels_per_metre) + 2 * self.margin_px
        if self.offset_x == 0.0 and self.offset_y == 0.0:
            self.offset_x = float(self.margin_px)
            self.offset_y = float(self.margin_px)

    def resize(self, window_w: int, window_h: int) -> None:
        """Recompute pixels_per_metre/offset_x/offset_y to fit the pitch
        (plus margin_px on every side) into a window_w x window_h window,
        preserving aspect ratio (the whole pitch stays visible -- "contain",
        not "cover") and centring it in whatever space is left over on
        the non-constraining axis. Call this whenever the actual window
        size changes (initial creation, drag-resize, OS maximise/restore)."""
        avail_w = max(1.0, window_w - 2 * self.margin_px)
        avail_h = max(1.0, window_h - 2 * self.margin_px)
        self.pixels_per_metre = max(
            1.0, min(avail_w / self.pitch.length_m, avail_h / self.pitch.width_m)
        )
        self.screen_width = window_w
        self.screen_height = window_h
        pitch_px_w = self.pitch.length_m * self.pixels_per_metre
        pitch_px_h = self.pitch.width_m * self.pixels_per_metre
        self.offset_x = (window_w - pitch_px_w) / 2.0
        self.offset_y = (window_h - pitch_px_h) / 2.0

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """World (x=length axis, y=width axis, origin at pitch centre) to
        screen pixels (origin top-left, y grows downward)."""
        screen_x = self.offset_x + (x + self.pitch.half_length) * self.pixels_per_metre
        screen_y = self.offset_y + (self.pitch.half_width - y) * self.pixels_per_metre
        return int(screen_x), int(screen_y)

    def screen_to_world(self, screen_x: float, screen_y: float) -> tuple[float, float]:
        """Inverse of world_to_screen - used to translate mouse clicks into
        world positions."""
        x = (screen_x - self.offset_x) / self.pixels_per_metre - self.pitch.half_length
        y = self.pitch.half_width - (screen_y - self.offset_y) / self.pixels_per_metre
        return x, y

    def scale_length(self, length_m: float) -> int:
        """Converts a world-space length (e.g. a radius) to pixels."""
        return max(1, int(length_m * self.pixels_per_metre))

    @staticmethod
    def fit_to_pitch(pitch: Pitch, pixels_per_metre: float = 9.0, margin_px: int = 40) -> "Camera":
        return Camera(pitch=pitch, pixels_per_metre=pixels_per_metre, margin_px=margin_px)

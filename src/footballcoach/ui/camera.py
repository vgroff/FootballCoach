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
    # Ball-follow zoom (App's "[-] zoom [+]" control, near the sim-speed
    # control). `zoomed=False` (zoom_factor <= 1.0) is the normal "whole
    # pitch visible" mode above; above 1.0, `pixels_per_metre` is boosted by
    # `zoom_factor` and centring switches from "fit the pitch" to "follow a
    # world point" via `follow()`, called once per frame with the ball's
    # position. Set via `set_zoom_level()`, not directly.
    zoomed: bool = False
    zoom_factor: float = 1.0
    _base_ppm: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.screen_width == 0 and self.screen_height == 0:
            self.screen_width = int(self.pitch.length_m * self.pixels_per_metre) + 2 * self.margin_px
            self.screen_height = int(self.pitch.width_m * self.pixels_per_metre) + 2 * self.margin_px
        if self.offset_x == 0.0 and self.offset_y == 0.0:
            self.offset_x = float(self.margin_px)
            self.offset_y = float(self.margin_px)
        self._base_ppm = self.pixels_per_metre

    def resize(self, window_w: int, window_h: int) -> None:
        """Recompute pixels_per_metre/offset_x/offset_y to fit the pitch
        (plus margin_px on every side) into a window_w x window_h window,
        preserving aspect ratio (the whole pitch stays visible -- "contain",
        not "cover") and centring it in whatever space is left over on
        the non-constraining axis. Call this whenever the actual window
        size changes (initial creation, drag-resize, OS maximise/restore).

        When `zoomed`, the fit-to-pitch ppm is still recomputed (as
        `_base_ppm`, the basis `zoom_factor` scales from) so the zoom level
        adapts consistently to window size, but the *offset* is left alone
        here -- it's about to be overwritten by the next `follow()` call
        rather than centred on the pitch."""
        avail_w = max(1.0, window_w - 2 * self.margin_px)
        avail_h = max(1.0, window_h - 2 * self.margin_px)
        self._base_ppm = max(
            1.0, min(avail_w / self.pitch.length_m, avail_h / self.pitch.width_m)
        )
        self.screen_width = window_w
        self.screen_height = window_h
        if self.zoomed:
            self.pixels_per_metre = self._base_ppm * self.zoom_factor
            return
        self.pixels_per_metre = self._base_ppm
        pitch_px_w = self.pitch.length_m * self.pixels_per_metre
        pitch_px_h = self.pitch.width_m * self.pixels_per_metre
        self.offset_x = (window_w - pitch_px_w) / 2.0
        self.offset_y = (window_h - pitch_px_h) / 2.0

    @property
    def zoom_scale(self) -> float:
        """Current `pixels_per_metre` as a multiple of the fit-to-pitch
        baseline (`_base_ppm`) -- 1.0 normally, `zoom_factor` while zoomed.
        Renderer uses this to scale the players'/ball's on-screen *minimum*
        radius floors along with everything else: those floors are sized in
        flat pixels for the normal whole-pitch view, so left un-scaled they
        stop growing under zoom as soon as an entity's true-to-scale size
        overtakes the floor -- which happens far sooner for the ball
        (much smaller than a player) than for players, making the ball look
        disproportionately tiny once zoomed in even though it visually grew
        the least of anything on screen."""
        return self.pixels_per_metre / self._base_ppm if self._base_ppm > 0 else 1.0

    def set_zoom_level(self, level: float) -> None:
        """Sets the ball-follow zoom multiplier (a continuous slider value,
        not just on/off). `level <= 1.0` restores the normal fit-to-pitch
        view immediately (`zoomed` becomes False); `level > 1.0` engages
        ball-follow zoom at that multiplier and takes full effect once
        `follow()` is next called (App does so every frame while zoomed,
        before drawing)."""
        self.zoom_factor = level
        self.zoomed = level > 1.0
        if self.zoomed:
            self.pixels_per_metre = self._base_ppm * self.zoom_factor
        else:
            self.pixels_per_metre = self._base_ppm
            pitch_px_w = self.pitch.length_m * self.pixels_per_metre
            pitch_px_h = self.pitch.width_m * self.pixels_per_metre
            self.offset_x = (self.screen_width - pitch_px_w) / 2.0
            self.offset_y = (self.screen_height - pitch_px_h) / 2.0

    def follow(self, world_x: float, world_y: float) -> None:
        """Recentres the (zoomed) view so `(world_x, world_y)` maps to the
        middle of the screen -- called once per frame with the ball's
        position while `zoomed` is True. No-op effect if called while not
        zoomed (harmless, just wasted work) since `set_zoom_level(1.0)`
        already re-centred on the pitch."""
        self.offset_x = self.screen_width / 2.0 - (world_x + self.pitch.half_length) * self.pixels_per_metre
        self.offset_y = self.screen_height / 2.0 - (self.pitch.half_width - world_y) * self.pixels_per_metre

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

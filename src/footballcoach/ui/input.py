"""Translates mouse/keyboard input into Move/Kick/Pass/Tackle/Save orders on
the currently selected player.

Interaction scheme:
- Click a player -> select them. Click a different same-team player to
  switch selection (no deselection on same-player click).
- Click an opposing-team player while one of your players is selected ->
  GetPossessionOrder (chase the ball carrier).
- Click empty ground while a player is selected -> MoveOrder to that point
  (default OrderMode.MOVE).

Multi-phase kick UI (replaces the old click-drag kick):
  Phase 1 - AIM_XY: click the already-selected player who has the ball to
    enter kick-aim mode.  Moving the mouse sets the XY direction (world-space
    vector from player to mouse) and power (mouse distance, capped at
    MAX_KICK_DRAG_M).  A live trajectory + 1-sigma error cone are drawn.
    Click anywhere to commit XY and advance to Phase 2.
  Phase 2 - AIM_Z: mouse distance from the player controls elevation angle
    (close = high loft, far = flat).  The trajectory is redrawn with height
    colour-coding (black=ascending/safe, blue=descending/safe, red=above
    goal height).  Click anywhere to commit elevation and advance to Phase 3.
  Phase 3 - SPIN: mouse angle around the player controls spin axis; distance
    from player controls spin magnitude.  Trajectory updates to show the
    Magnus-effect curve.  Click anywhere to fire the kick.
  Esc cancels the kick UI at any phase.

- Press `P` -> enter OrderMode.PASS for one order.
- Press `K` -> enter OrderMode.SHOOT for one order.
- Press `S` -> SaveOrder on the selected goalkeeper.
- Press `X` -> StopOrder on the selected player.
- See ui/app.py help overlay (H) for the full user-facing list.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

from footballcoach.engine.match import Match
from footballcoach.entities.player import Player, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import GetPossessionOrder, KickOrder, MoveOrder, PassOrder, SaveOrder, ShootOrder, StopOrder
from footballcoach.ui.camera import Camera

CLICK_DRAG_THRESHOLD_PX = 6
MAX_KICK_DRAG_M = 15.0
GROUND_AIM_HEIGHT_M = 0.3
SELECT_TOLERANCE_PX = 6


class OrderMode(Enum):
    """Which order a ground/player click issues to the selected player.

    MOVE is the persistent default.  PASS and SHOOT are transient one-shot
    modes that revert to MOVE after a single click."""
    MOVE = auto()
    PASS = auto()
    SHOOT = auto()


class KickPhase(Enum):
    """Phases of the multi-phase kick UI."""
    AIM_XY = auto()   # mouse pos -> aim direction + power
    AIM_Z = auto()    # mouse distance from player -> elevation angle
    SPIN = auto()     # mouse angle/distance from player -> spin vector


@dataclass
class KickUIState:
    """Accumulates parameters across all three kick UI phases.

    Fields are updated live from mouse motion; the renderer reads them
    each frame to draw the trajectory preview.
    """
    phase: KickPhase
    player_id: str

    # AIM_XY (live in phase 1, committed for phases 2+)
    aim_dir_x: float = 1.0    # world-space unit vector components
    aim_dir_y: float = 0.0
    power_fraction: float = 0.5
    aim_distance_m: float = 10.0  # projected aim distance along aim_dir

    # AIM_Z (live in phase 2, committed for phase 3)
    elevation_angle_rad: float = 0.0

    # SPIN (live in phase 3)
    spin: Vector3 = field(default_factory=Vector3.zero)


@dataclass
class DragState:
    """Tracks an in-progress mouse drag (used only for non-kick drags now)."""
    active: bool = False
    start_screen: tuple[int, int] = (0, 0)
    current_screen: tuple[int, int] = (0, 0)


@dataclass
class MatchInputController:
    match: Match
    camera: Camera
    selected_player_id: str | None = None
    drag: DragState = field(default_factory=DragState)
    order_mode: OrderMode = OrderMode.MOVE
    # Active multi-phase kick UI state, or None when not kicking.
    _kick_ui: KickUIState | None = field(default=None, repr=False)
    # Last known mouse position in world space (updated every motion event).
    _last_mouse_world: tuple[float, float] = field(default=(0.0, 0.0), repr=False)
    # Called when a human-issued order completes: (player_id, order_name).
    on_order_complete: Callable[[str, str], None] | None = field(default=None, repr=False, compare=False)
    # Called when any new order is issued (e.g. to clear a pause notification).
    on_new_order: Callable[[], None] | None = field(default=None, repr=False, compare=False)
    # Called when a kick is queued from the kick UI (to pre-pause before execution).
    on_kick_issued: Callable[[], None] | None = field(default=None, repr=False, compare=False)
    # Called when a human issues any order: (player_id, order_name, is_debug).
    # is_debug=True for MoveOrder (too common for INFO level).
    on_order_issued: Callable[[str, str, bool], None] | None = field(default=None, repr=False, compare=False)

    # -- accessors -----------------------------------------------------------

    def selected_player(self) -> Player | None:
        if self.selected_player_id is None:
            return None
        for p in self.match.players:
            if p.player_id == self.selected_player_id:
                return p
        return None

    def kick_ui_state(self) -> KickUIState | None:
        """Returns the active kick UI state for the renderer, or None."""
        return self._kick_ui

    def _player_has_ball(self, player: Player) -> bool:
        return self.match.ball.possessed_by == player.player_id

    # -- order helpers -------------------------------------------------------

    def _issue_order(self, player: Player, order, order_name: str) -> None:
        """Assign *order* to *player* and attach the on_complete callback."""
        if self.on_order_complete is not None:
            pid = player.player_id
            cb = self.on_order_complete
            name = order_name
            order.on_complete = lambda: cb(pid, name)
        if self.on_new_order is not None:
            self.on_new_order()
        player.current_order = order
        if self.on_order_issued is not None:
            self.on_order_issued(player.player_id, order_name, isinstance(order, MoveOrder))

    def _player_at_screen_pos(self, screen_pos: tuple[int, int]) -> Player | None:
        best: Player | None = None
        best_dist = float("inf")
        for player in self.match.players:
            px, py = self.camera.world_to_screen(player.position.x, player.position.y)
            radius_px = self.camera.scale_length(player.radius_m) + SELECT_TOLERANCE_PX
            dist = math.hypot(screen_pos[0] - px, screen_pos[1] - py)
            if dist <= radius_px and dist < best_dist:
                best = player
                best_dist = dist
        return best

    # -- mouse events --------------------------------------------------------

    def handle_mouse_down(self, screen_pos: tuple[int, int]) -> None:
        # Kick UI intercepts all mouse events — advance on mouse_up only.
        if self._kick_ui is not None:
            return
        self.drag = DragState(active=True, start_screen=screen_pos, current_screen=screen_pos)

    def handle_mouse_motion(self, screen_pos: tuple[int, int], shift_held: bool) -> None:
        world_x, world_y = self.camera.screen_to_world(*screen_pos)
        self._last_mouse_world = (world_x, world_y)

        if self._kick_ui is not None:
            self._update_kick_ui_from_mouse(world_x, world_y)
            return

        if self.drag.active:
            self.drag.current_screen = screen_pos

    def handle_mouse_up(self, screen_pos: tuple[int, int]) -> None:
        world_x, world_y = self.camera.screen_to_world(*screen_pos)
        self._last_mouse_world = (world_x, world_y)

        if self._kick_ui is not None:
            self._advance_kick_ui()
            return

        if not self.drag.active:
            return

        drag_distance_px = math.hypot(
            screen_pos[0] - self.drag.start_screen[0],
            screen_pos[1] - self.drag.start_screen[1],
        )
        is_drag = drag_distance_px > CLICK_DRAG_THRESHOLD_PX

        if not is_drag:
            self._handle_click(screen_pos)

        self.drag = DragState()

    # -- kick UI lifecycle ---------------------------------------------------

    def _enter_kick_ui(self, player: Player) -> None:
        """Begin Phase 1 (AIM_XY) for *player*.

        Immediately cancels the player's current order (stops them moving) and
        pauses the game, so all three aim phases happen while frozen.
        """
        # Clear any existing pause state, then set a StopOrder so the player
        # decelerates on the next tick instead of continuing their move.
        if self.on_new_order is not None:
            self.on_new_order()
        player.current_order = StopOrder()
        self._kick_ui = KickUIState(phase=KickPhase.AIM_XY, player_id=player.player_id)
        # Seed aim direction from current mouse position (already set).
        self._update_kick_ui_from_mouse(*self._last_mouse_world)
        if self.on_kick_ui_entered is not None:
            self.on_kick_ui_entered()

    def cancel_kick_ui(self) -> None:
        """Cancel the kick UI at any phase (called by Esc). Resumes play."""
        self._kick_ui = None
        if self.on_new_order is not None:
            self.on_new_order()

    def regress_kick_ui(self) -> None:
        """Go back one phase (right-click). Cancels (and resumes) from Phase 1."""
        if self._kick_ui is None:
            return
        if self._kick_ui.phase == KickPhase.AIM_XY:
            self._kick_ui = None
            if self.on_new_order is not None:
                self.on_new_order()
        elif self._kick_ui.phase == KickPhase.AIM_Z:
            self._kick_ui.phase = KickPhase.AIM_XY
        elif self._kick_ui.phase == KickPhase.SPIN:
            self._kick_ui.phase = KickPhase.AIM_Z

    def _update_kick_ui_from_mouse(self, world_x: float, world_y: float) -> None:
        """Recalculate kick parameters from the current mouse world position."""
        if self._kick_ui is None:
            return
        player = self.selected_player()
        if player is None:
            self._kick_ui = None
            return

        # Guard: abort if the player has lost the ball mid-phase.
        if not self._player_has_ball(player):
            self._kick_ui = None
            return

        dx = world_x - player.position.x
        dy = world_y - player.position.y
        dist_m = math.hypot(dx, dy)

        phase = self._kick_ui.phase

        if phase == KickPhase.AIM_XY:
            if dist_m > 1e-6:
                self._kick_ui.aim_dir_x = dx / dist_m
                self._kick_ui.aim_dir_y = dy / dist_m
            self._kick_ui.power_fraction = min(1.0, dist_m / MAX_KICK_DRAG_M)
            self._kick_ui.aim_distance_m = min(dist_m * 2.0, 60.0)

        elif phase == KickPhase.AIM_Z:
            # Mouse close to player = max loft; far away = flat.
            from footballcoach.config import load_graphics_config
            cfg = load_graphics_config()["kick_ui"]
            max_loft_rad = math.radians(cfg["max_loft_angle_deg"])
            t = 1.0 - min(1.0, dist_m / MAX_KICK_DRAG_M)
            self._kick_ui.elevation_angle_rad = max_loft_rad * t

        elif phase == KickPhase.SPIN:
            from footballcoach.config import load_graphics_config
            from footballcoach.ui.kick_trajectory import spin_from_mouse
            cfg = load_graphics_config()["kick_ui"]
            self._kick_ui.spin = spin_from_mouse(
                self._kick_ui.aim_dir_x,
                self._kick_ui.aim_dir_y,
                dx,
                dy,
                cfg["max_spin_magnitude_rads"],
            )

    def _advance_kick_ui(self) -> None:
        """Advance to the next kick phase, or fire the kick if in SPIN."""
        if self._kick_ui is None:
            return

        phase = self._kick_ui.phase
        if phase == KickPhase.AIM_XY:
            self._kick_ui.phase = KickPhase.AIM_Z
        elif phase == KickPhase.AIM_Z:
            self._kick_ui.phase = KickPhase.SPIN
        elif phase == KickPhase.SPIN:
            self._fire_kick()

    def _fire_kick(self) -> None:
        """Construct and issue the KickOrder from committed kick UI state."""
        if self._kick_ui is None:
            return
        player = self.selected_player()
        if player is None or not self._player_has_ball(player):
            self._kick_ui = None
            return

        ku = self._kick_ui
        # Reconstruct aim_point: project along aim_dir at aim_distance_m with
        # z derived from elevation angle so that solve_launch_pitch_rad yields
        # the intended launch angle.
        aim_z = GROUND_AIM_HEIGHT_M + math.tan(ku.elevation_angle_rad) * ku.aim_distance_m
        aim_point = Vector3(
            player.position.x + ku.aim_dir_x * ku.aim_distance_m,
            player.position.y + ku.aim_dir_y * ku.aim_distance_m,
            aim_z,
        )
        # Game is already paused (from _enter_kick_ui). Set the kick order
        # directly (no on_complete — one Space executes, no second pause).
        # Update the notification so the user knows to press Space.
        player.current_order = KickOrder(aim_point=aim_point, power_fraction=ku.power_fraction, spin=ku.spin)
        if self.on_kick_issued is not None:
            self.on_kick_issued()
        self._kick_ui = None

    # -- click handler -------------------------------------------------------

    def _handle_click(self, screen_pos: tuple[int, int]) -> None:
        clicked_player = self._player_at_screen_pos(screen_pos)
        selected = self.selected_player()

        if clicked_player is not None:
            if selected is None:
                self.selected_player_id = clicked_player.player_id
                return
            if clicked_player.player_id == selected.player_id:
                # Same player: enter kick UI if they have the ball, else no-op.
                if self._player_has_ball(selected):
                    self._enter_kick_ui(selected)
                return
            if clicked_player.team == selected.team:
                if self.order_mode == OrderMode.PASS:
                    self._issue_transient_order(selected, PassOrder(target_position=clicked_player.position), "Pass")
                elif self.order_mode == OrderMode.SHOOT:
                    self._issue_transient_order(selected, ShootOrder(aim_point=clicked_player.position, power_fraction=1.0), "Shoot")
                else:
                    self.selected_player_id = clicked_player.player_id
                return
            # Opposing player - chase and get possession.
            self._issue_order(selected, GetPossessionOrder(), "Get Possession")
            return

        # Empty ground.
        if selected is not None:
            world_x, world_y = self.camera.screen_to_world(*screen_pos)
            if self.order_mode == OrderMode.PASS:
                target = Vector3(world_x, world_y, 0.0)
                self._issue_transient_order(selected, PassOrder(target_position=target), "Pass")
            elif self.order_mode == OrderMode.SHOOT:
                target = Vector3(world_x, world_y, 1.0)
                self._issue_transient_order(selected, ShootOrder(aim_point=target, power_fraction=1.0), "Shoot")
            else:
                target = Vector3(world_x, world_y, 0.0)
                self._issue_order(selected, MoveOrder(target_position=target, sprint=True), "Move")

    def _issue_transient_order(self, player: Player, order, label: str) -> None:
        """Issues an order that applies to exactly one click while in a
        transient order mode (PASS/SHOOT), then reverts to MOVE.

        Centralizes the "auto-revert to MOVE after one click" behaviour so
        a future click-handling branch that issues a PASS/SHOOT order can't
        forget the reset and leave the controller stuck in that mode.
        """
        self._issue_order(player, order, label)
        self.order_mode = OrderMode.MOVE

    # -- transient order modes -----------------------------------------------

    def try_enter_kick_ui(self) -> bool:
        """Enter kick UI Phase 1 if the selected player has the ball.

        Returns True if kick UI was entered, False otherwise (caller can
        fall back to another action, e.g. enter_shoot_mode).
        """
        selected = self.selected_player()
        if selected is not None and self._player_has_ball(selected):
            self._enter_kick_ui(selected)
            return True
        return False

    def enter_pass_mode(self) -> None:
        self.order_mode = OrderMode.PASS

    def enter_shoot_mode(self) -> None:
        self.order_mode = OrderMode.SHOOT

    def cancel_order_mode(self) -> None:
        self.order_mode = OrderMode.MOVE

    def cancel_pass_mode(self) -> None:
        self.order_mode = OrderMode.MOVE

    def issue_save_order(self) -> None:
        """Issues a SaveOrder to the selected player if they are a goalkeeper."""
        selected = self.selected_player()
        if selected is not None and selected.is_goalkeeper:
            self._issue_order(selected, SaveOrder(), "Save")

    def issue_stop_order(self) -> None:
        """Decelerates the selected player to a standstill."""
        selected = self.selected_player()
        if selected is not None:
            self._issue_order(selected, StopOrder(), "Stop")

    def drag_indicator(self) -> tuple[tuple[float, float], tuple[int, int]] | None:
        """Legacy method — always returns None (kick drag replaced by kick UI)."""
        return None

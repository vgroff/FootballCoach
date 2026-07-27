"""Translates mouse input into Move/Kick/Tackle orders on the currently
selected player.

Interaction scheme (per project design):
- Click a player -> select them (click the same selected player again to
  deselect; click a same-team player to switch selection).
- Click an opposing-team player while a player is selected -> issue a
  TackleOrder from the selected player at the clicked player.
- Click empty ground while a player is selected -> issue a MoveOrder to that
  point.
- Click-and-drag starting ON the selected player (only meaningful if they
  currently have the ball) -> issue a KickOrder: drag direction sets the aim
  direction, drag length sets power (capped), release fires the kick. Hold
  Shift while dragging to loft the kick higher (chip/lob) instead of a low
  driven kick.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from footballcoach.engine.match import Match
from footballcoach.entities.player import Player, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import KickOrder, MoveOrder, TackleOrder
from footballcoach.ui.camera import Camera

CLICK_DRAG_THRESHOLD_PX = 6
MAX_KICK_DRAG_M = 15.0
GROUND_AIM_HEIGHT_M = 0.3
LOFTED_AIM_HEIGHT_M = 2.0
SELECT_TOLERANCE_PX = 6


@dataclass
class DragState:
    active: bool = False
    on_player_id: str | None = None
    start_screen: tuple[int, int] = (0, 0)
    current_screen: tuple[int, int] = (0, 0)
    lofted: bool = False


@dataclass
class MatchInputController:
    match: Match
    camera: Camera
    selected_player_id: str | None = None
    drag: DragState = field(default_factory=DragState)

    def selected_player(self) -> Player | None:
        if self.selected_player_id is None:
            return None
        for p in self.match.players:
            if p.player_id == self.selected_player_id:
                return p
        return None

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

    def handle_mouse_down(self, screen_pos: tuple[int, int]) -> None:
        clicked_player = self._player_at_screen_pos(screen_pos)
        selected = self.selected_player()

        self.drag = DragState(active=True, start_screen=screen_pos, current_screen=screen_pos)
        if selected is not None and clicked_player is not None and clicked_player.player_id == selected.player_id:
            self.drag.on_player_id = selected.player_id

    def handle_mouse_motion(self, screen_pos: tuple[int, int], shift_held: bool) -> None:
        if self.drag.active:
            self.drag.current_screen = screen_pos
            self.drag.lofted = shift_held

    def handle_mouse_up(self, screen_pos: tuple[int, int]) -> None:
        if not self.drag.active:
            return

        drag_distance_px = math.hypot(
            screen_pos[0] - self.drag.start_screen[0], screen_pos[1] - self.drag.start_screen[1]
        )
        is_drag = drag_distance_px > CLICK_DRAG_THRESHOLD_PX

        if is_drag and self.drag.on_player_id is not None:
            self._finish_kick_drag(screen_pos)
        else:
            self._handle_click(screen_pos)

        self.drag = DragState()

    def _handle_click(self, screen_pos: tuple[int, int]) -> None:
        clicked_player = self._player_at_screen_pos(screen_pos)
        selected = self.selected_player()

        if clicked_player is not None:
            if selected is None:
                self.selected_player_id = clicked_player.player_id
                return
            if clicked_player.player_id == selected.player_id:
                self.selected_player_id = None  # deselect
                return
            if clicked_player.team == selected.team:
                self.selected_player_id = clicked_player.player_id  # switch selection
                return
            # Opposing player - issue a tackle order.
            selected.current_order = TackleOrder(target_player_id=clicked_player.player_id)
            return

        # Empty ground - issue a move order to the selected player, if any.
        if selected is not None:
            world_x, world_y = self.camera.screen_to_world(*screen_pos)
            target = Vector3(world_x, world_y, 0.0)
            selected.current_order = MoveOrder(target_position=target, sprint=True)

    def _finish_kick_drag(self, release_screen: tuple[int, int]) -> None:
        player = self.selected_player()
        if player is None:
            return

        start_x, start_y = self.camera.world_to_screen(player.position.x, player.position.y)
        dx_px = release_screen[0] - start_x
        dy_px = release_screen[1] - start_y

        # Screen y grows downward; world y grows "upward" (toward +y), so
        # flip dy when converting to world-space direction.
        drag_length_px = math.hypot(dx_px, dy_px)
        drag_length_m = drag_length_px / self.camera.pixels_per_metre
        if drag_length_m < 1e-6:
            return

        direction_x = dx_px / drag_length_px
        direction_y = -dy_px / drag_length_px

        power_fraction = min(1.0, drag_length_m / MAX_KICK_DRAG_M)
        aim_distance_m = min(drag_length_m * 2.0, 60.0)  # project the aim point out along the drag direction
        aim_height = LOFTED_AIM_HEIGHT_M if self.drag.lofted else GROUND_AIM_HEIGHT_M

        aim_point = Vector3(
            player.position.x + direction_x * aim_distance_m,
            player.position.y + direction_y * aim_distance_m,
            aim_height,
        )

        player.current_order = KickOrder(aim_point=aim_point, power_fraction=power_fraction, spin=Vector3.zero())

    def drag_indicator(self) -> tuple[tuple[float, float], tuple[int, int]] | None:
        """Returns (player_world_xy, current_mouse_screen_pos) if a kick drag
        is in progress, for the renderer to draw an aim line."""
        if not self.drag.active or self.drag.on_player_id is None:
            return None
        player = self.selected_player()
        if player is None:
            return None
        return (player.position.x, player.position.y), self.drag.current_screen

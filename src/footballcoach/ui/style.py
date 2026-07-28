"""Colour palette and small drawing constants for the pygame renderer."""
from __future__ import annotations

PITCH_GREEN = (34, 120, 50)
PITCH_LINE_WHITE = (235, 235, 235)
BALL_COLOUR = (255, 255, 255)
BALL_OUTLINE = (40, 40, 40)

TEAM_LEFT_COLOUR = (60, 110, 220)
TEAM_RIGHT_COLOUR = (220, 70, 70)
SELECTED_OUTLINE = (255, 220, 40)
GOALKEEPER_COLOUR = (235, 140, 30)  # orange fill, overrides team colour
POSSESSION_OUTLINE = (255, 255, 255)  # white outline on the player in possession
INACTIVE_ALPHA = 110  # 0-255; inactive players are drawn translucent

HUD_BG = (15, 15, 20)
HUD_TEXT = (235, 235, 235)
HUD_ACCENT = (120, 200, 255)

DRAG_KICK_LINE = (255, 255, 255)
DRAG_TACKLE_LINE = (255, 90, 90)

HOTKEY_BAR_BG = (20, 20, 28)
HOTKEY_ENABLED = (210, 210, 215)
HOTKEY_DISABLED = (80, 80, 92)   # dim but still readable
HOTKEY_ACTIVE = (120, 200, 255)  # accent: current mode is active (same as HUD_ACCENT)

FONT_NAME = None  # None = pygame default font
HUD_FONT_SIZE = 18
TITLE_FONT_SIZE = 32

# A true-to-scale player (0.3m) or ball (0.11m) renders as only a couple of
# pixels at typical zoom levels - these floors keep them visible without
# affecting their (physically accurate) world position.
MIN_PLAYER_RADIUS_PX = 8
MIN_BALL_RADIUS_PX = 4

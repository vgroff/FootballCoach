import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import sys
sys.path.insert(0, "src")

import pygame
from footballcoach.ui import player_sprites as ps

pygame.init()
params = ps.PlayerSpriteParams.from_config()

shirt_colors = [(200, 30, 30), (30, 60, 200)]
sides = [1, -1]
levels = [None] + list(ps.STRIDE_LEVELS)  # None = standing

cell = ps._BASE_SIZE + 20
cols = len(sides) * len(levels)
rows = len(shirt_colors)
sheet = pygame.Surface((cell * cols, cell * rows), pygame.SRCALPHA)
sheet.fill((40, 40, 40, 255))

font = pygame.font.SysFont("arial", 12)

for r, shirt in enumerate(shirt_colors):
    sprite_set = ps.PlayerSpriteSet(params, shirt)
    c = 0
    for side in sides:
        for level in levels:
            surf = sprite_set.get(side, level)
            x, y = c * cell, r * cell
            sheet.blit(surf, (x + 10, y + 10))
            label = f"s{side} {level if level is not None else 'stand'}"
            txt = font.render(label, True, (255, 255, 0))
            sheet.blit(txt, (x + 4, y + cell - 16))
            c += 1

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_sprite_sheet.png")
pygame.image.save(sheet, out_path)
print("saved", out_path)

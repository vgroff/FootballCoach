"""Goal detection and score tracking."""
from __future__ import annotations

from dataclasses import dataclass, field

from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Team


@dataclass
class Scoreboard:
    left_goals: int = 0
    right_goals: int = 0

    def score_for(self, side: str) -> None:
        if side == "left":
            self.right_goals += 1  # ball entered LEFT goal => RIGHT team scored
        elif side == "right":
            self.left_goals += 1  # ball entered RIGHT goal => LEFT team scored
        else:
            raise ValueError(f"unknown goal side: {side}")

    def goals_for(self, team: Team) -> int:
        return self.left_goals if team == Team.LEFT else self.right_goals


def check_goal(ball: Ball, pitch: Pitch) -> str | None:
    """Returns 'left' or 'right' if the ball has crossed into that goal
    mouth, else None. Caller is responsible for calling Scoreboard.score_for
    and resetting the ball/players for kickoff."""
    return pitch.is_goal(ball.position)

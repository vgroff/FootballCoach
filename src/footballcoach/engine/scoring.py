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

    def record_goal_against(self, team: Team) -> None:
        if team == Team.LEFT:
            self.right_goals += 1
        else:
            self.left_goals += 1

    def goals_for(self, team: Team) -> int:
        return self.left_goals if team == Team.LEFT else self.right_goals


def check_goal(ball: Ball, pitch: Pitch) -> Team | None:
    """The team defending the goal mouth the ball has crossed into, else None.
    Caller is responsible for calling Scoreboard.record_goal_against and
    resetting the ball/players for kickoff."""
    return pitch.is_goal(ball.position)

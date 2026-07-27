from __future__ import annotations

from footballcoach.engine.scoring import Scoreboard, check_goal
from footballcoach.entities.ball import Ball
from footballcoach.entities.pitch import Pitch
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3


def test_check_goal_detects_left_and_right():
    pitch = Pitch.standard()
    left_ball = Ball.at_rest(Vector3(-pitch.half_length - 0.1, 0, 1.0))
    right_ball = Ball.at_rest(Vector3(pitch.half_length + 0.1, 0, 1.0))
    mid_ball = Ball.at_rest(Vector3(0, 0, 1.0))

    assert check_goal(left_ball, pitch) == "left"
    assert check_goal(right_ball, pitch) == "right"
    assert check_goal(mid_ball, pitch) is None


def test_check_goal_none_if_above_crossbar_or_wide():
    pitch = Pitch.standard()
    too_high = Ball.at_rest(Vector3(-pitch.half_length - 0.1, 0, pitch.goal_height_m + 1.0))
    too_wide = Ball.at_rest(Vector3(-pitch.half_length - 0.1, pitch.goal_width_m, 1.0))
    assert check_goal(too_high, pitch) is None
    assert check_goal(too_wide, pitch) is None


def test_scoreboard_scores_correct_team():
    board = Scoreboard()
    board.score_for("left")  # ball entered LEFT goal => RIGHT team scores
    assert board.goals_for(Team.RIGHT) == 1
    assert board.goals_for(Team.LEFT) == 0

    board.score_for("right")
    assert board.goals_for(Team.LEFT) == 1
    assert board.goals_for(Team.RIGHT) == 1

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

    assert check_goal(left_ball, pitch) is Team.LEFT
    assert check_goal(right_ball, pitch) is Team.RIGHT
    assert check_goal(mid_ball, pitch) is None


def test_check_goal_none_if_above_crossbar_or_wide():
    pitch = Pitch.standard()
    too_high = Ball.at_rest(Vector3(-pitch.half_length - 0.1, 0, pitch.goal_height_m + 1.0))
    too_wide = Ball.at_rest(Vector3(-pitch.half_length - 0.1, pitch.goal_width_m, 1.0))
    assert check_goal(too_high, pitch) is None
    assert check_goal(too_wide, pitch) is None


def test_scoreboard_scores_correct_team():
    board = Scoreboard()
    board.record_goal_against(Team.LEFT)
    assert board.goals_for(Team.RIGHT) == 1
    assert board.goals_for(Team.LEFT) == 0

    board.record_goal_against(Team.RIGHT)
    assert board.goals_for(Team.LEFT) == 1
    assert board.goals_for(Team.RIGHT) == 1


def test_team_opponent_is_the_other_team():
    assert Team.LEFT.opponent is Team.RIGHT
    assert Team.RIGHT.opponent is Team.LEFT


def _goal_match(ball_x: float, **kwargs):
    import random

    from footballcoach.engine.match import Match
    from footballcoach.entities.attributes import PlayerAttributes
    from footballcoach.entities.player import Player

    pitch = Pitch.standard()
    attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    players = [Player.create("p1", Team.LEFT, attrs), Player.create("p2", Team.RIGHT, attrs, position=Vector3(5, 0, 0))]
    ball = Ball.at_rest(Vector3(ball_x * (pitch.half_length + 0.1), 0, 1.0))
    return Match(pitch=pitch, players=players, ball=ball, rng_reduction=1.0, rng=random.Random(0), **kwargs)


def test_goal_message_names_the_team_that_scored():
    """A ball in the left goal is a RIGHT goal; the log used to say LEFT."""
    messages = []
    match = _goal_match(-1.0, log_callback=lambda level, message, detail: messages.append(message))
    match._check_goal()
    assert match.scoreboard.goals_for(Team.RIGHT) == 1 and match.scoreboard.goals_for(Team.LEFT) == 0
    assert any(m.startswith("GOAL for RIGHT") for m in messages), messages


def test_goal_message_in_the_right_goal_names_left():
    messages = []
    match = _goal_match(1.0, log_callback=lambda level, message, detail: messages.append(message))
    match._check_goal()
    assert match.scoreboard.goals_for(Team.LEFT) == 1
    assert any(m.startswith("GOAL for LEFT") for m in messages), messages


def test_match_logger_records_the_scoring_side():
    from footballcoach.engine.match_logger import MatchLogger

    logger = MatchLogger()
    match = _goal_match(-1.0, match_logger=logger)
    match._check_goal()
    goal = next(e for e in logger.events if e.event == "goal")
    assert goal.player_id == "right"

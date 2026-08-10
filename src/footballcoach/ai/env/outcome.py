"""Shared trial-end detection for ScenarioLoop (UI) and ScenarioEnv (training).

Both consumers need to answer the same question every tick -- "has this
trial ended, and why?" -- from the same Match state. Keeping the detection
logic in one place means the UI and training code can never silently drift
apart on what counts as a goal/miss/save/dispossession/box-possession, and
outcome labels shown in the UI always match the ones used for training
rewards/termination.

Outcome vocabulary (no catch-all "other" bucket -- every outcome is named
for what actually happened):
    "miss"          - ball left the pitch (out of bounds). ScenarioEnv
                      (phase 1 only) further splits this into "miss" (some
                      player last touched/kicked the ball -- that player is
                      penalised) vs "invalid" (nobody touched it at all this
                      episode -- nobody's fault, no penalty) -- see
                      ScenarioEnv.step(). This module only ever returns the
                      undifferentiated "miss".
    "goal"          - scoreboard changed.
    "saved"         - initial carrier's opponent goalkeeper took the ball.
    "dispossessed"  - initial carrier's opponent outfield player took the ball.
    "box_possession" - a player dribbled the ball into the opponent's box.
    "course_complete" - a player's AI reports it finished its course (e.g.
                      SprintWaypointAI reaching its final waypoint).
    "timeout"       - trial_tick reached timeout_ticks with no other outcome.
"""
from __future__ import annotations

from footballcoach.engine.match import Match
from footballcoach.entities.player import Team


def detect_trial_outcome(
    match: Match,
    *,
    initial_scoreboard: tuple[int, int],
    initial_carrier_id: str | None,
    ball_released: bool,
    box_possession_terminal: bool,
    trial_tick: int,
    timeout_ticks: int,
) -> tuple[str | None, bool]:
    """Returns (outcome_key, is_half_linger) if the trial is over, else (None, False).

    ``is_half_linger`` tells the caller whether this outcome should use a
    shorter linger (out-of-bounds events) or the full linger (everything
    else) -- callers decide the actual seconds themselves.
    """
    pitch = match.pitch
    ball = match.ball
    scoreboard = match.scoreboard

    if abs(ball.position.x) > pitch.half_length + 1.0:
        match.notify_ball_out()
        return "miss", True
    if abs(ball.position.y) > pitch.half_width + 0.5:
        match.notify_ball_out()
        return "miss", True

    if (scoreboard.left_goals, scoreboard.right_goals) != initial_scoreboard:
        return "goal", False

    if ball_released:
        if ball.possessed_by is not None and ball.possessed_by != initial_carrier_id:
            try:
                repossessor = match.player_by_id(ball.possessed_by)
                initial_carrier = (
                    match.player_by_id(initial_carrier_id) if initial_carrier_id else None
                )
                if initial_carrier is not None and repossessor.team != initial_carrier.team:
                    if repossessor.is_goalkeeper:
                        return "saved", False
                    return "dispossessed", False
            except KeyError:
                pass
            return "saved", False

    # Box possession: any player dribbled the ball into the opponent's box.
    # Team.LEFT attacks +x so their opponent box is the right box (left=False).
    # Skipped when box_possession_terminal is False (e.g. 1v2, where the
    # attacker is meant to enter the box and play to a natural end).
    if box_possession_terminal:
        for player in match.players:
            if ball.possessed_by == player.player_id:
                in_opp_box = pitch.is_in_box(
                    ball.position,
                    left=(player.team == Team.RIGHT),  # opponent box for LEFT; mirrored for RIGHT
                )
                if in_opp_box:
                    return "box_possession", False

    # Course completion: any player whose AI exposes course_complete() (e.g.
    # SprintWaypointAI) that reports having finished its waypoint course.
    for player in match.players:
        is_complete = getattr(player.ai, "course_complete", None)
        if is_complete is not None and is_complete(player):
            return "course_complete", False

    if trial_tick >= timeout_ticks:
        return "timeout", False

    return None, False

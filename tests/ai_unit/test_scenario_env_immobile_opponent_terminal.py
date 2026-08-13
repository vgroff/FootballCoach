"""Regression test: an immobile phase-1 opponent must never be able to
trigger the "opponent reached trainee's box with possession" terminal
condition, regardless of how it ever ended up positioned there (spawn
coincidence, collision push-apart, or anything else).

This directly gates ScenarioEnv's own terminal-condition check
(opponent_box_terminal in step()) on match._opponent_is_immobile, rather
than only trying to prevent the geometric circumstances that could lead to
it (see ui/scenarios.py's build_1v1_scenario spawn re-roll, which is a
belt-and-braces companion to this, not a substitute for it -- the spawn
re-roll alone was found insufficient in practice, see engine/knowledge.md).

Companion to tests/scenario/test_1v1_immobile_opponent_placement.py, which
covers the spawn-time re-roll specifically.
"""
from __future__ import annotations

from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario


def _make_env() -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="immobile_terminal_1v1",
        label="Immobile-opponent terminal-condition test: 1v1",
        description="Regression test for the opponent_box_possession-vs-immobile guard",
        build=build_1v1_scenario,
    )
    return ScenarioEnv(
        definition=defn,
        trainee_player_id="trainee",
        phase=1,
        max_episode_s=30.0,
    )


def _force_opponent_in_trainee_box_with_possession(env: ScenarioEnv) -> None:
    match = env._loop.match
    trainee = match.player_by_id("trainee")
    opponent = match.player_by_id("opponent")

    # Place the ball (and the opponent, so possession is stable and doesn't
    # immediately relocate the ball via _sync_possessed_ball) squarely
    # inside the TRAINEE's own defending box.
    box_pos = match.pitch.penalty_spot(left=(trainee.team == Team.LEFT))
    opponent.position = box_pos
    opponent.velocity = Vector3.zero()
    match.ball.position = box_pos
    match.ball.velocity = Vector3.zero()
    match._set_possession(opponent.player_id)

    # Trainee parked well away, stationary, no order/AI -- nothing should
    # move this tick regardless of outcome.
    trainee.position = Vector3(0.0, 20.0, 0.0)
    trainee.velocity = Vector3.zero()
    trainee.current_order = None


class TestImmobileOpponentCannotTriggerLossTerminal:
    def test_immobile_opponent_in_box_with_possession_does_not_end_episode(self):
        env = _make_env()
        env.reset()
        match = env._loop.match
        match._opponent_is_immobile = True
        match._opponent_use_rules_ai = False
        _force_opponent_in_trainee_box_with_possession(env)

        _obs, _reward, done, info = env.step()

        assert done is False, (
            "an immobile opponent possessing the ball inside the trainee's "
            "own box must NOT end the episode -- it never chases or holds "
            "a defensive line, so this can only be a coincidence, not a "
            "real result the trainee should be penalised for."
        )
        assert info.trial_outcome != "opponent_box_possession"

    def test_mobile_opponent_in_box_with_possession_still_ends_episode(self):
        """Control: the guard must be specific to the immobile case -- a
        rules-based or neural opponent legitimately achieving this through
        real play must still end the episode as a trainee loss, same as
        before."""
        env = _make_env()
        env.reset()
        match = env._loop.match
        match._opponent_is_immobile = False
        match._opponent_use_rules_ai = True
        _force_opponent_in_trainee_box_with_possession(env)

        _obs, _reward, done, info = env.step()

        assert done is True
        assert info.trial_outcome == "opponent_box_possession"

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

Also covers the sibling bug found in ``sec_box_terminal`` (a secondary
player's OWN reward-terminal check, used when recording demonstrations with
``always_compute_secondary_reward=True``): unlike box_terminal/
opponent_box_terminal above, sec_box_terminal does NOT feed into the real
episode-ending ``done`` flag, so an ungated version let the one-time
box_possession_terminal/speed_bonus reward re-fire every tick for as long as
an immobile secondary player happened to be sitting where the ball settled
(confirmed: 64 consecutive ticks, and 23,757 corrupted trainee-perspective
rows, in real recorded data) instead of firing once. Both bugs are now
closed by the single shared ``ScenarioEnv._can_score_box_terminal()`` gate
(``player.ai is not None``) -- see its docstring.
"""
from __future__ import annotations

from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3
from footballcoach.rules_ai import Phase1RulesAI
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
        before.

        The guard is keyed on whether the opponent has a real controller
        (``player.ai is not None`` -- see ScenarioEnv._can_score_box_terminal()),
        not just the ``_opponent_is_immobile`` bookkeeping flag, so this
        must actually install a controller to be a faithful "mobile
        opponent" control case."""
        env = _make_env()
        env.reset()
        match = env._loop.match
        match.player_by_id("opponent").ai = Phase1RulesAI()
        match._opponent_is_immobile = False
        match._opponent_use_rules_ai = True
        _force_opponent_in_trainee_box_with_possession(env)

        _obs, _reward, done, info = env.step()

        assert done is True
        assert info.trial_outcome == "opponent_box_possession"


def _make_env_with_secondary_reward() -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="sec_box_terminal_1v1",
        label="sec_box_terminal regression test: 1v1",
        description="Regression test for the sec_box_terminal-vs-immobile guard",
        build=build_1v1_scenario,
    )
    env = ScenarioEnv(
        definition=defn,
        trainee_player_id="trainee",
        phase=1,
        max_episode_s=30.0,
        secondary_player_ids=["opponent"],
    )
    env.always_compute_secondary_reward = True
    return env


class TestSecBoxTerminalCannotFireRepeatedlyForImmobileSecondary:
    """``sec_box_terminal`` computes a secondary player's OWN reward-terminal
    condition and, unlike box_terminal/opponent_box_terminal, is never
    gated by the real episode-ending ``done`` -- so an immobile secondary
    player left sitting in the scoring box (a physics fluke, not real play)
    could collect the one-time terminal bonus on every tick instead of
    once. Re-force the same fluke position before EACH step to simulate the
    ball staying parked there for several ticks in a row, as it did in the
    real recorded episode that surfaced this bug."""

    def test_immobile_secondary_never_gets_box_terminal_bonus(self):
        env = _make_env_with_secondary_reward()
        env.reset()
        match = env._loop.match
        match.player_by_id("opponent").ai = None
        match._opponent_is_immobile = True
        match._opponent_use_rules_ai = False

        for _ in range(5):
            _force_opponent_in_trainee_box_with_possession(env)
            _obs, _reward, done, _info = env.step()
            assert done is False
            [sec] = [s for s in env.last_secondary_results if s["player_id"] == "opponent"]
            sec_comps = sec["reward_components"]
            assert sec_comps.get("box", 0.0) == 0.0, (
                "an immobile secondary player must never collect the "
                "one-time box_possession_terminal bonus -- it can only be "
                "sitting in the box by physics coincidence, and unlike "
                "box_terminal/opponent_box_terminal this doesn't end the "
                "episode, so an ungated check would let it re-fire every "
                "tick instead of never."
            )
            assert sec_comps.get("spd", 0.0) == 0.0

    def test_mobile_secondary_gets_box_terminal_once_then_episode_ends(self):
        """Control: a real (rules-based) secondary player legitimately
        reaching the box must still be rewarded for it -- and because that
        also satisfies opponent_box_terminal, the episode ends immediately,
        so there is no repeated-firing window to worry about for a real
        opponent."""
        env = _make_env_with_secondary_reward()
        env.reset()
        match = env._loop.match
        match.player_by_id("opponent").ai = Phase1RulesAI()
        match._opponent_is_immobile = False
        match._opponent_use_rules_ai = True
        _force_opponent_in_trainee_box_with_possession(env)

        _obs, _reward, done, _info = env.step()

        assert done is True
        [sec] = [s for s in env.last_secondary_results if s["player_id"] == "opponent"]
        assert sec["reward_components"].get("box", 0.0) > 0.0


class TestRewardComponentsAreNotMergedAcrossPlayers:
    """Regression test: env.last_reward_components (the trainee's own
    per-component reward breakdown) and each secondary player's own
    last_secondary_results[i]["reward_components"] must never bleed into
    each other.

    Before this was split apart, scenario_env.py additively merged every
    secondary player's own components INTO env.last_reward_components, and
    record_demonstrations.py wrote that single merged dict onto every
    player's row for the tick -- so a trainee "win" row also carried the
    LOSING opponent's own loss_terminal ("lterm"/"opponent_box") penalty,
    e.g. box=+2.0 (trainee's real win bonus) alongside lterm=-2.5 (the
    opponent's own loss, not the trainee's) on the SAME row. This is what a
    real recorded demonstration directory looked like before the fix (see
    ai/knowledge.md)."""

    def test_trainee_win_row_does_not_carry_opponents_loss_penalty(self):
        env = _make_env_with_secondary_reward()
        env.reset()
        match = env._loop.match
        trainee = match.player_by_id("trainee")
        opponent = match.player_by_id("opponent")
        # box_terminal (the trainee's OWN win condition) is gated by
        # _can_score_box_terminal() same as everything else, so the trainee
        # needs a real controller too -- in every actual usage path
        # (live PPO rollouts, rules-AI matches) it always has one by the
        # time step() runs; a bare test harness must set it explicitly.
        trainee.ai = Phase1RulesAI()
        opponent.ai = Phase1RulesAI()
        match._opponent_is_immobile = False
        match._opponent_use_rules_ai = True

        # Place the ball (and the trainee, so possession is stable) squarely
        # inside the OPPONENT's box -- mirror image of
        # _force_opponent_in_trainee_box_with_possession above.
        box_pos = match.pitch.penalty_spot(left=(trainee.team == Team.RIGHT))
        trainee.position = box_pos
        trainee.velocity = Vector3.zero()
        match.ball.position = box_pos
        match.ball.velocity = Vector3.zero()
        match._set_possession(trainee.player_id)
        opponent.position = Vector3(0.0, -20.0, 0.0)
        opponent.velocity = Vector3.zero()
        opponent.current_order = None

        _obs, _reward, done, info = env.step()

        assert done is True
        assert info.trial_outcome == "box_possession"
        assert env.last_reward_components.get("box", 0.0) > 0.0, (
            "trainee's own row must carry its own win bonus"
        )
        assert env.last_reward_components.get("lterm", 0.0) == 0.0, (
            "trainee's own row must NOT carry the opponent's own loss "
            "penalty just because they happen to share a tick -- these are "
            "two different players' rewards, not one env-level total."
        )
        [sec] = [s for s in env.last_secondary_results if s["player_id"] == "opponent"]
        sec_comps = sec["reward_components"]
        assert sec_comps.get("lterm", 0.0) < 0.0, (
            "the losing opponent's own row must carry ITS OWN loss penalty"
        )
        assert sec_comps.get("box", 0.0) == 0.0, (
            "the losing opponent's own row must NOT carry the trainee's "
            "own win bonus"
        )

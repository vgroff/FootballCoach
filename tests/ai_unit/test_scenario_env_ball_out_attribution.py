"""Regression tests: ScenarioEnv must attribute the ball_out_penalty / "miss"
outcome to whichever player LAST touched the ball (possession or kick) this
episode, and to that player ONLY -- not to every player who touched the ball
at any point earlier in the episode. A ball-out with NO toucher at all this
episode is "invalid" (nobody's fault, no penalty for anyone) -- distinct from
"miss" (a known last toucher, that player is penalised).

Before the fix, ScenarioEnv tracked "did this player ever touch the ball
this episode" per player, independently for the trainee and each secondary
player, with no reset and no notion of *last* toucher -- so the wrong player
(or both) could be penalised when the ball went out.
"""
from __future__ import annotations

from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario


def _make_env(**scenario_kwargs) -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="ball_out_attribution_1v1",
        label="Ball-out attribution test: 1v1",
        description="Regression test for last-toucher ball_out attribution",
        build=build_1v1_scenario,
    )
    return ScenarioEnv(
        definition=defn,
        trainee_player_id="trainee",
        phase=1,
        max_episode_s=30.0,
        trainee_team=Team.LEFT,
        opponent_immobile_prob=1.0,
        **scenario_kwargs,
    )


def _send_ball_out(match, toucher_id: str) -> None:
    """Simulate `toucher_id` kicking the ball out of the pitch this tick.

    The ball must be LOOSE (not `possessed_by`) here -- a possessed ball is
    snapped back onto its carrier every tick by `_sync_possessed_ball()`,
    which would silently undo a manually-set out-of-bounds position.
    `kicked_this_tick` is the same flag ScenarioLoop._track_ball_toucher()
    reads to attribute the touch, mirroring what `Player.kick_direct()` sets
    in real play.

    match.step() is monkeypatched to a no-op: the real one resets EVERY
    player's kicked_this_tick=False at the top of _process_orders() (see
    Player.kicked_this_tick's per-tick reset in match.py), which would wipe
    this synthetic flag before ScenarioLoop._track_ball_toucher() -- now
    correctly called AFTER match.step(), not before, see the toucher-
    tracking dedup fix -- ever observes it. detect_trial_outcome() only
    reads match.ball.position (never requires match.step() to have run),
    so the no-op is safe for outcome detection too.
    """
    match.ball.possessed_by = None
    match.ball.position = Vector3(match.pitch.half_length + 5.0, 0.0, 0.0)
    match.ball.velocity = Vector3(5.0, 0.0, 0.0)
    match.player_by_id(toucher_id).kicked_this_tick = True
    match.step = lambda: None


class TestBallOutAttribution:
    def test_trainee_touches_ball_out_gets_penalised(self):
        env = _make_env()
        env.reset()
        match = env._loop.match
        _send_ball_out(match, "trainee")

        _, reward, done, info = env.step()

        assert done is True
        assert info.trial_outcome == "miss"
        assert env.last_reward_components["out"] < 0.0

    def test_opponent_touches_ball_out_trainee_not_penalised(self):
        env = _make_env()
        env.reset()
        match = env._loop.match
        _send_ball_out(match, "opponent")

        _, reward, done, info = env.step()

        assert done is True
        assert info.trial_outcome == "miss"
        assert env.last_reward_components["out"] == 0.0

    def test_earlier_trainee_touch_does_not_carry_over_to_opponents_miss(self):
        """The trainee touched the ball earlier in the episode, but the
        OPPONENT is the one who actually put it out this step -- the
        trainee must not be penalised just because it touched the ball at
        some earlier point."""
        env = _make_env()
        env.reset()
        match = env._loop.match
        trainee = match.player_by_id("trainee")

        # Trainee possesses the ball for a step, safely away from any line.
        match.ball.possessed_by = "trainee"
        match.ball.position = Vector3(trainee.position.x, trainee.position.y, 0.0)
        match.ball.velocity = Vector3(0.0, 0.0, 0.0)
        env.step()
        # Mid-episode (trial not over yet): the LIVE tracker lives on
        # ScenarioLoop now (single source of truth, see the toucher-tracking
        # dedup fix) -- env._loop.last_completed_trial_toucher_id only
        # updates when a trial actually ENDS, so it isn't the right thing to
        # check here.
        assert env._loop._last_ball_toucher_id == "trainee"

        # Now the opponent takes it and kicks it out.
        _send_ball_out(match, "opponent")
        _, reward, done, info = env.step()

        assert done is True
        assert info.trial_outcome == "miss"
        assert env.last_reward_components["out"] == 0.0

    def test_no_toucher_ball_out_is_invalid_not_miss(self):
        """Ball goes out with nobody having touched it this episode --
        outcome must be "invalid" (unintentional, nobody's fault), not
        "miss" (which implies a known, penalised last toucher), and nobody
        is penalised."""
        env = _make_env()
        env.reset()
        match = env._loop.match
        match.ball.possessed_by = None
        match.ball.position = Vector3(match.pitch.half_length + 5.0, 0.0, 0.0)
        match.ball.velocity = Vector3(5.0, 0.0, 0.0)

        _, reward, done, info = env.step()

        assert done is True
        assert info.trial_outcome == "invalid"
        assert env.last_reward_components["out"] == 0.0

    def test_only_five_outcome_buckets_exist(self):
        """Phase-1 outcomes must be exactly one of win/loss/miss/invalid/
        timeout (internally: box_possession/opponent_box_possession/miss/
        invalid/timeout). No other outcome label may ever be produced."""
        allowed = {"box_possession", "opponent_box_possession", "miss", "invalid", "timeout"}
        env = _make_env()
        env.reset()
        match = env._loop.match
        _send_ball_out(match, "trainee")
        _, _, done, info = env.step()
        assert done is True
        assert info.trial_outcome in allowed

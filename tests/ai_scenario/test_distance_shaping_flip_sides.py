"""Coverage for the 2026-10-14 re-introduced distance-shaping terms ("appr",
"prog" in reward.py) specifically for the "does it work properly for
flipping sides" requirement.

test_reward.py's TestApprAndProgShaping already proves phase1_reward()'s own
arithmetic is a correct, telescoping, single-coefficient potential (given
scalar inputs). This file is one level up: does ScenarioEnv correctly
RESOLVE those scalar inputs per player_id (never hardcoded to one team/
side), and does the single shared call site
(_compute_phase1_reward_for_player) treat the trainee and a secondary/
opponent player identically given matching inputs. A bug in either of those
-- not the formula itself -- is what would actually break "flipping sides"
(e.g. eval.swap_sides / seeded_eval.swap_player_states).
"""
from __future__ import annotations

import math

import pytest

from footballcoach.ai.curriculum.envs import build_env
from footballcoach.ai.curriculum.phases import CurriculumPhase
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3


def _env():
    phase = CurriculumPhase(
        name="p1", phase_id=1, scenario_key="phase1_1v1",
        env_kwargs={"max_episode_s": 15.0},
    )
    env = build_env(phase)
    env.always_compute_secondary_reward = True
    env.reset(seed=12345)
    return env


class TestBallDistanceResolvedPerPlayer:
    """_ball_dist_for_player / _ball_dist_to_opponent_box must depend on the
    GIVEN player_id's own position/team, never on which one happens to be
    'trainee'."""

    def test_ball_dist_for_player_uses_that_players_own_position(self):
        env = _env()
        match = env._loop.match
        trainee = match.player_by_id("trainee")
        opponent = match.player_by_id("opponent")
        trainee.position = Vector3(10.0, 0.0, 0.0)
        opponent.position = Vector3(-30.0, 5.0, 0.0)
        match.ball.position = Vector3(0.0, 0.0, 0.0)

        d_trainee = env._ball_dist_for_player("trainee", match)
        d_opponent = env._ball_dist_for_player("opponent", match)
        assert d_trainee == pytest.approx(10.0, rel=1e-6)
        assert d_opponent == pytest.approx(math.hypot(30.0, 5.0), rel=1e-6)

    def test_ball_dist_to_opponent_box_depends_on_each_players_own_team(self):
        """Swapping which team two players are on must swap WHICH box each
        one's distance is measured against -- confirms the box lookup is
        keyed off the player's own CURRENT team, not a fixed side or a
        stale per-player_id cache (_box_bounds_cache is keyed by Team, not
        player_id, so it stays correct across a team swap without needing
        to be cleared -- this test demonstrates that directly rather than
        just asserting it from reading the code)."""
        env = _env()
        match = env._loop.match
        trainee = match.player_by_id("trainee")
        opponent = match.player_by_id("opponent")
        # Ball near the RIGHT-side box.
        match.ball.position = Vector3(match.pitch.half_length - 5.0, 0.0, 0.0)

        trainee.team = Team.LEFT    # attacks the RIGHT box -> close to its target
        opponent.team = Team.RIGHT  # attacks the LEFT box -> far from its target
        d_trainee_before = env._ball_dist_to_opponent_box("trainee", match)
        d_opponent_before = env._ball_dist_to_opponent_box("opponent", match)
        assert d_trainee_before < d_opponent_before

        # Swap teams (mirrors what seeded_eval.swap_player_states does) --
        # each player_id's own box-distance must swap accordingly, with NO
        # cache-clearing needed in between.
        trainee.team, opponent.team = opponent.team, trainee.team
        d_trainee_after = env._ball_dist_to_opponent_box("trainee", match)
        d_opponent_after = env._ball_dist_to_opponent_box("opponent", match)
        assert d_trainee_after == pytest.approx(d_opponent_before, rel=1e-6)
        assert d_opponent_after == pytest.approx(d_trainee_before, rel=1e-6)


class TestComputeRewardForPlayerIsRoleAgnostic:
    """_compute_phase1_reward_for_player() -- the SINGLE shared call site for
    the trainee AND every secondary player -- must return IDENTICAL output
    given identical inputs, regardless of which concrete player_obj is
    passed. The only per-player_obj-derived input it reads itself is
    last_desired_speed_mode (for 'sprint'); confirm that doesn't leak any
    role-specific behaviour into 'appr'/'prog'."""

    def test_identical_inputs_give_identical_reward_for_either_player(self):
        env = _env()
        env._reward_cfg["phase1"] = {
            **env._reward_cfg["phase1"],
            "ball_approach_bonus": 0.3,
            "ball_progress_scale": 0.4,
        }
        match = env._loop.match
        trainee = match.player_by_id("trainee")
        opponent = match.player_by_id("opponent")

        kwargs = dict(
            ball_went_out_after_touch=False,
            reached_opponent_box_with_possession=False,
            opponent_reached_trainee_box=False,
            timed_out=False,
            player_ball_approach_delta_m=5.0,
            ball_progress_toward_goal_m=3.0,
        )
        reward_trainee, comps_trainee = env._compute_phase1_reward_for_player(player_obj=trainee, **kwargs)
        reward_opponent, comps_opponent = env._compute_phase1_reward_for_player(player_obj=opponent, **kwargs)

        assert reward_trainee == pytest.approx(reward_opponent)
        assert comps_trainee == pytest.approx(comps_opponent)
        # A genuine, nonzero sanity check -- not just "both zero".
        assert comps_trainee["appr"] == pytest.approx(0.3 * 5.0)
        assert comps_trainee["prog"] == pytest.approx(0.4 * 3.0)


class TestStepEndToEndWithSwappedSides:
    """Smoke-level end-to-end check: a real swap_player_states() + a few
    real env.step() calls must run cleanly and keep producing 'appr'/'prog'
    in both the trainee's and the secondary's own component breakdown --
    i.e. the real production swap-sides path (eval.swap_sides,
    seeded_eval.swap_player_states) actually reaches these terms, not just
    the lower-level pieces tested in isolation above."""

    def test_swapped_env_runs_and_both_roles_get_appr_components(self):
        import dataclasses
        from footballcoach.ai.eval.seeded_eval import swap_player_states
        from footballcoach.rules_ai import Phase1RulesAI

        phase = CurriculumPhase(
            name="p1", phase_id=1, scenario_key="phase1_1v1",
            env_kwargs={"max_episode_s": 15.0},
        )
        env = build_env(phase)
        env.always_compute_secondary_reward = True
        env._reward_cfg["phase1"] = {
            **env._reward_cfg["phase1"],
            "ball_approach_bonus": 0.3,
            "ball_progress_scale": 0.4,
        }
        _orig_build = env.definition.build

        def _swapped_build(*a, **kw):
            m = _orig_build(*a, **kw)
            swap_player_states(m, "trainee", "opponent")
            return m

        env.definition = dataclasses.replace(env.definition, build=_swapped_build)
        env.reset(seed=999)
        # No sample_action_fn is wired up in this test (no real network
        # needed) -- drive both players with the rules AI directly, same
        # mechanism record_demonstrations.py uses for a driver-less trainee.
        env._loop.match.player_by_id("trainee").ai = Phase1RulesAI()
        env._loop.match.player_by_id("opponent").ai = Phase1RulesAI()

        saw_trainee_appr = False
        saw_opponent_appr = False
        for _ in range(10):
            _obs, _reward, done, _info = env.step()
            assert "appr" in env.last_reward_components
            assert "prog" in env.last_reward_components
            if env.last_reward_components["appr"] != 0.0:
                saw_trainee_appr = True
            for sec in env.last_secondary_results:
                if sec["player_id"] == "opponent":
                    assert "appr" in sec["reward_components"]
                    if sec["reward_components"]["appr"] != 0.0:
                        saw_opponent_appr = True
            if done:
                break
        assert saw_trainee_appr
        assert saw_opponent_appr


def _attacking_aim_point(match, player) -> Vector3:
    """A point far down whichever direction *player*'s team attacks --
    Team.LEFT attacks +x (pitch.half_length end), Team.RIGHT attacks -x --
    see ScenarioEnv._ball_dist_to_opponent_box's own rx_min/rx_max convention."""
    x = match.pitch.half_length if player.team == Team.LEFT else -match.pitch.half_length
    return Vector3(x, 0.0, 0.0)


class TestRealKickGatingBehavior:
    """2026-10-15 fix validation: drive an ACTUAL kick through the real
    physics (Player.kick_direct, the same method the engine/AI use, not a
    hand-computed scalar delta) and confirm the live tick-loop gating in
    ScenarioEnv.step() behaves as designed: the kicker's own 'appr' is
    frozen (no self-inflicted penalty for the mere act of kicking) while
    'prog' DOES credit the ground covered during the kick's flight. The
    pure-function arithmetic is already proven in test_reward.py; this is
    the one level up that actually exercises _last_toucher_prev /
    trainee_appr_accum / trainee_prog_accum together, which no other test
    does with a REAL kick."""

    def _kicking_env(self, *, approach_coef: float, progress_coef: float):
        from footballcoach.rules_ai import Phase1RulesAI

        phase = CurriculumPhase(
            name="p1", phase_id=1, scenario_key="phase1_1v1",
            env_kwargs={"max_episode_s": 15.0},
        )
        env = build_env(phase)
        env._reward_cfg["phase1"] = {
            **env._reward_cfg["phase1"],
            "ball_approach_bonus": approach_coef,
            "ball_progress_scale": progress_coef,
        }
        env.reset(seed=4242)
        match = env._loop.match
        trainee = match.player_by_id(env.trainee_player_id)
        opponent = match.player_by_id("opponent")
        trainee.ai = Phase1RulesAI()
        opponent.ai = Phase1RulesAI()
        # Pull the opponent to the FAR end from wherever the trainee is
        # about to kick (its own defensive end, opposite the attacking
        # direction) so it has no chance of touching the ball during this
        # one step() -- isolating the trainee's own kick as the only thing
        # in play.
        attack_x = match.pitch.half_length if trainee.team == Team.LEFT else -match.pitch.half_length
        opponent.position = Vector3(-attack_x * 0.9, 0.0, 0.0)
        return env, match, trainee

    def test_own_kick_does_not_penalize_approach(self):
        """Kicking the ball forward necessarily separates the kicker from
        it -- confirm 'appr' does NOT register that as a penalty (it's
        frozen while last_touched_by_player_id is still the kicker, which a
        kick never changes -- see Ball.last_touched_by_player_id's own
        docstring)."""
        env, match, trainee = self._kicking_env(approach_coef=1.0, progress_coef=0.0)
        trainee.position = Vector3(0.0, 0.0, 0.0)
        match.ball.position = Vector3(0.0, 0.0, 0.0)
        match.ball.velocity = Vector3.zero()
        match.ball.possessed_by = trainee.player_id
        match.ball.last_touched_by_player_id = trainee.player_id

        aim = _attacking_aim_point(match, trainee)
        assert trainee.can_kick(match)
        trainee.kick_direct(match, aim_point=aim, power_fraction=1.0, spin=Vector3.zero())
        # Sanity: the kick actually fired and released the ball, without
        # touching last_touched_by_player_id (the property this whole fix
        # depends on).
        assert match.ball.possessed_by is None
        assert match.ball.last_touched_by_player_id == trainee.player_id

        env.step()
        assert env.last_reward_components["appr"] == pytest.approx(0.0, abs=1e-9), (
            "the trainee's own kick separated them from the ball, but 'appr' is gated on "
            "last_touched_by_player_id != self -- it must NOT charge a penalty for that "
            "self-inflicted (and, while in control, irrelevant) separation"
        )

    def test_own_kick_toward_box_credits_progress(self):
        """The ground covered while the ball is in flight from the
        trainee's OWN kick must count as 'prog' -- gated on
        last_touched_by_player_id == self, which survives the kick (unlike
        live possession, which a kick clears immediately)."""
        env, match, trainee = self._kicking_env(approach_coef=0.0, progress_coef=1.0)
        # Start at the pitch center so there's real box-distance left to
        # close, and aim the kick squarely at the trainee's attacking box.
        trainee.position = Vector3(0.0, 0.0, 0.0)
        match.ball.position = Vector3(0.0, 0.0, 0.0)
        match.ball.velocity = Vector3.zero()
        match.ball.possessed_by = trainee.player_id
        match.ball.last_touched_by_player_id = trainee.player_id

        box_dist_before = env._ball_dist_to_opponent_box(trainee.player_id, match)
        aim = _attacking_aim_point(match, trainee)
        assert trainee.can_kick(match)
        trainee.kick_direct(match, aim_point=aim, power_fraction=1.0, spin=Vector3.zero())
        assert match.ball.possessed_by is None
        assert match.ball.last_touched_by_player_id == trainee.player_id

        env.step()
        box_dist_after = env._ball_dist_to_opponent_box(trainee.player_id, match)
        assert box_dist_after < box_dist_before, "test setup didn't actually send the ball toward the box"
        assert env.last_reward_components["prog"] > 0.0, (
            "the kick's flight closed real box-distance while last_touched_by_player_id was "
            "still the trainee -- 'prog' must credit it, not silently exclude it the way "
            "gating on LIVE possession (the pre-2026-10-15 design) would have"
        )


class TestNonTouchingPlayerApprIsBoundedResidualNoise:
    """Known, ACCEPTED residual: a player who never touches the ball at all
    has last_touched_by_player_id != self for the WHOLE episode, so their
    'appr' gate never freezes -- they earn/pay for ball movement entirely
    driven by someone else (e.g. an immobile opponent reacting to nothing
    while the trainee carries the ball around). This is NOT the
    self-inflicted-kick-penalty bug the 2026-10-15 gating fixes (that one
    is specific to the mover's OWN kick); it's the same class of raw-
    distance-potential ball-motion contamination reward.py's own 'appr_sq'
    docstring already flagged and accepted as a tradeoff for using a linear
    (telescoping) potential at all. Documented here with a direct bound so
    a future reader doesn't mistake it for an oversight: it's real, it's
    accepted, and it's bounded by the coefficient -- not by magic."""

    def test_immobile_non_touching_player_accrues_bounded_appr_noise(self):
        from footballcoach.rules_ai import Phase1RulesAI, StopWhenIdleAI

        phase = CurriculumPhase(
            name="p1", phase_id=1, scenario_key="phase1_1v1",
            env_kwargs={"max_episode_s": 15.0},
        )
        env = build_env(phase)
        env.always_compute_secondary_reward = True
        coef = 0.3
        env._reward_cfg["phase1"] = {
            **env._reward_cfg["phase1"],
            "ball_approach_bonus": coef,
            "ball_progress_scale": 0.0,
        }
        env.reset(seed=777)
        match = env._loop.match
        trainee = match.player_by_id(env.trainee_player_id)
        trainee.ai = Phase1RulesAI()
        # Opponent genuinely never touches the ball: StopWhenIdleAI just
        # holds position (never chases/kicks), so last_touched_by_player_id
        # can never become "opponent" -- their 'appr' gate is open for the
        # entire run below.
        opponent = match.player_by_id("opponent")
        opponent.ai = StopWhenIdleAI()

        total_opponent_appr = 0.0
        max_abs_single_step = 0.0
        for _ in range(15):
            _obs, _reward, done, _info = env.step()
            for sec in env.last_secondary_results:
                if sec["player_id"] == "opponent":
                    v = sec["reward_components"]["appr"]
                    total_opponent_appr += v
                    max_abs_single_step = max(max_abs_single_step, abs(v))
            if done:
                break

        # It's real (not all zero) -- confirms the residual actually exists
        # rather than asserting a property that's vacuously true.
        assert total_opponent_appr != 0.0
        # And it's bounded by the coefficient times a plausible per-step
        # distance change (sprint top speed is well under 12 m/s; one
        # decision interval is ~0.25s) -- not some unbounded blow-up.
        assert max_abs_single_step < coef * 5.0, (
            f"a single step's 'appr' contribution for a player who never touched the ball "
            f"was {max_abs_single_step}, far larger than coef * a plausible per-step ball "
            f"displacement -- this would indicate something other than ordinary ball-motion "
            f"noise is going on"
        )

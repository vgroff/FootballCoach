"""Reward-attribution correctness for LIVE self-play -- both the trainee and
the secondary opponent controlled by a real ``NeuralPlayerAI``, exactly the
production code path (``ai_config.json``'s ``phase1_opponent_neural_ratio``),
never ``record_demonstrations.py``'s ``always_compute_secondary_reward=True``
escape hatch, which no PPO training path ever sets.

``ScenarioEnv.step()`` computes each player's reward independently via the
SAME ``_compute_phase1_reward_for_player()``, called once per player, with
the episode's two terminal booleans deliberately CROSS-WIRED for the
secondary player's own call (see ``scenario_env.py`` around the
``last_secondary_results`` loop)::

    reached_opponent_box_with_possession=sec_box_terminal,  # did THIS player score
    opponent_reached_trainee_box=box_terminal,  # from sec's POV, trainee winning = sec losing

This is the single most safety-critical line for self-play reward
correctness: swap which boolean feeds which player, or get the direction
backwards, and both players see contradictory or duplicated signals for the
exact same physical event -- corrupting self-play silently, with no crash
and no NaN, exactly the kind of bug that would look like "generally quite
stagnant" training rather than an obvious failure.

Existing coverage
(``test_scenario_env_immobile_opponent_terminal.py::TestRewardComponentsAreNotMergedAcrossPlayers``)
already proves components don't get ADDITIVELY MERGED across players, but
only for ONE direction (trainee wins) and with a ``Phase1RulesAI`` opponent
plus the ``always_compute_secondary_reward=True`` escape hatch -- never with
a genuine ``NeuralPlayerAI``-vs-``NeuralPlayerAI`` matchup (the actual
self-play case), and never for the opponent-wins direction, which is a
SEPARATE code path (a different branch of the same cross-wiring) that could
be wrong independently of the trainee-wins direction. This file:

1. Exercises BOTH directions (trainee scores / opponent scores) with two
   real ``NeuralPlayerAI`` instances.
2. Checks a natural (unforced) tick doesn't spuriously fire either
   terminal component -- a regression guard against the cross-wiring
   firing on a null event.
3. Checks the attribution survives into the actual ``RolloutBuffer`` via
   ``track_id`` (the ``weight=secondary_weight`` production pattern from
   ``ppo_trainer.py``'s single-process ``train()`` loop) and into
   ``compute_gae()``'s returns -- the thing PPO's per-track GAE
   segmentation and the policy gradient itself actually consume, not just
   the env-level reward dict.
"""
from __future__ import annotations

import functools
import math

import pytest

from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _action_to_numpy
from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
from footballcoach.entities.player import Team
from footballcoach.mathutils import Vector3
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario

_MAX_FORCE_ATTEMPTS = 40


def _make_self_play_env(trainer: PPOTrainer) -> ScenarioEnv:
    """1v1 env forced to a fully-neural secondary opponent every episode
    (mirrors test_secondary_opponent_gae.py's own fixture) -- the exact
    scenario shape real self-play training uses, not a rules-based
    stand-in.

    Uses deterministic=True sampling (mean action, no exploration noise) so
    that _force_scoring_episode()'s retries are driven by varying the
    network's WEIGHTS (a fresh PPOTrainer per attempt), not by re-rolling
    sampling noise on top of them too -- see that function's own docstring
    for why a live network's action can't just be forced into "hold the
    ball" by removing sampling noise alone.
    """
    defn = ScenarioDefinition(
        key="reward_attribution_self_play_test",
        label="Self-play reward attribution test",
        description="1v1 with a fully neural secondary opponent",
        build=functools.partial(
            build_1v1_scenario, opponent_rules_prob=0.0, opponent_immobile_prob=0.0,
        ),
    )
    env = ScenarioEnv(
        definition=defn, trainee_player_id="trainee", phase=1,
        secondary_player_ids=["opponent"], max_episode_s=30.0,
    )
    env.sample_action_fn = functools.partial(trainer._sample_action, deterministic=True)
    env.reset()
    assert type(env.match.player_by_id("trainee").ai).__name__ == "NeuralPlayerAI"
    assert type(env.match.player_by_id("opponent").ai).__name__ == "NeuralPlayerAI", (
        "fixture didn't wire a real NeuralPlayerAI onto the secondary "
        "player -- opponent_rules_prob/opponent_immobile_prob=0.0 should "
        "force this every episode; every test in this file is vacuous "
        "otherwise (it would silently fall back to the immobile/rules-AI "
        "path, which has its own, different, already-tested code)"
    )
    return env


def _force_scoring_position(env: ScenarioEnv, *, scorer_id: str, victim_id: str) -> None:
    """Place the ball, in ``scorer_id``'s possession, squarely inside
    ``victim_id``'s own defending box -- generalizes
    test_scenario_env_immobile_opponent_terminal.py's own forcing helpers to
    work in EITHER direction, since both players here are real
    NeuralPlayerAI (not one real controller + one stubbed-out bystander).
    Must be called on tick 1 of a fresh episode (right after env.reset(),
    before the first env.step()) so both players' decision counters are
    "due" and the forced possession/position is what the terminal check
    sees at the top of that tick's loop, before either player's own action
    this tick has a chance to move anything.
    """
    match = env.match
    scorer = match.player_by_id(scorer_id)
    victim = match.player_by_id(victim_id)

    box_pos = match.pitch.penalty_spot(left=(victim.team == Team.LEFT))
    scorer.position = box_pos
    scorer.velocity = Vector3.zero()
    match.ball.position = box_pos
    match.ball.velocity = Vector3.zero()
    match._set_possession(scorer.player_id)

    victim.position = Vector3(0.0, -30.0 if victim.team == Team.LEFT else 30.0, 0.0)
    victim.velocity = Vector3.zero()


def _force_scoring_episode(*, scorer_id: str, victim_id: str, expected_outcome: str):
    """Repeatedly builds a FRESH trainer + episode, forces scorer_id into
    victim_id's own box with possession, and steps once -- retrying (a new
    random weight init each attempt) until the expected terminal outcome is
    actually reached on that tick. Returns (trainer, env, reward, info).

    A single attempt is NOT reliable enough to build a test on: the scorer
    here is a real, live NeuralPlayerAI, and even with deterministic=True
    (mean action, no sampling noise) its action for "ball at my feet, deep
    in the box" depends on the network's (untrained) weights -- for a large
    minority of random inits the mean action there is a SHOT, which
    releases the ball at high speed (observed: 2-3m of ball travel within
    one 1/30s tick) before detect_phase1_box_terminal() gets to see it
    still possessed. That's not a WRONG outcome, just no outcome at all on
    this tick (done=False, exactly like any other ordinary non-terminal
    step) -- confirmed empirically: a fresh trainer's mean action reached
    the intended terminal on the very first forced tick only ~11/20 times.
    Critically, retrying with the SAME trainer (fresh episode, same
    weights) does NOT help -- deterministic=True means the mean action for
    a near-identical observation (ball at feet, deep in the box, opponent
    far away) is nearly IDENTICAL every retry, so a trainer whose weights
    say "shoot" here says it every time. Varying the WEIGHTS (a fresh
    PPOTrainer.from_config() per attempt) is what actually changes the
    outcome. This keeps every test in this file exercising REAL engine
    mechanics end to end (no monkeypatching detect_phase1_box_terminal, no
    hand-built action tuples), at the cost of a handful of extra, cheap
    (~0.1-0.2s each) attempts -- P(40 consecutive failures) is astronomically
    small at the observed ~50% per-attempt success rate.
    """
    for _attempt in range(_MAX_FORCE_ATTEMPTS):
        trainer = PPOTrainer.from_config()
        env = _make_self_play_env(trainer)
        _force_scoring_position(env, scorer_id=scorer_id, victim_id=victim_id)
        _obs, reward, done, info = env.step()
        if done and info.trial_outcome == expected_outcome:
            return trainer, env, reward, info
    raise AssertionError(
        f"never reached outcome={expected_outcome!r} (scorer_id={scorer_id!r}) "
        f"after {_MAX_FORCE_ATTEMPTS} attempts -- the forcing mechanism "
        "itself may be broken (not just unlucky), investigate before "
        "raising _MAX_FORCE_ATTEMPTS further"
    )


def _secondary_row(env: ScenarioEnv, player_id: str) -> dict:
    [row] = [r for r in env.last_secondary_results if r["player_id"] == player_id]
    return row


class TestLiveSelfPlayScoringAttribution:
    def test_trainee_scores_against_neural_opponent(self):
        _trainer, env, reward, info = _force_scoring_episode(
            scorer_id="trainee", victim_id="opponent", expected_outcome="box_possession",
        )

        assert env.last_reward_components.get("box", 0.0) > 0.0, (
            "trainee's own row must carry its own win bonus"
        )
        assert env.last_reward_components.get("lterm", 0.0) == 0.0, (
            "trainee's own row must NOT carry a loss penalty -- it won"
        )
        assert reward > 0.5, (
            f"trainee's total reward={reward} should be dominated by the "
            "+3.0 box_possession_terminal bonus"
        )

        sec = _secondary_row(env, "opponent")
        assert sec["reward_components"].get("lterm", 0.0) < 0.0, (
            "the losing secondary opponent's own row must carry ITS OWN "
            "loss penalty (opponent_reached_trainee_box must be wired to "
            "box_terminal for the secondary's call, not left False)"
        )
        assert sec["reward_components"].get("box", 0.0) == 0.0, (
            "the losing secondary opponent's own row must NOT carry the "
            "trainee's win bonus -- that would mean sec_box_terminal (the "
            "secondary's OWN scoring flag) was wrongly set True here"
        )
        assert sec["reward"] < -0.5, (
            f"opponent's total reward={sec['reward']} should be dominated "
            "by the -2.5 loss_terminal penalty"
        )
        assert sec["done"] == 1.0, (
            "the secondary player's own row must also be marked done -- "
            "the episode ended for both tracks on the same physical tick"
        )

    def test_opponent_scores_against_neural_trainee(self):
        """Mirror of the above, in the OTHER direction -- a separate branch
        of the same cross-wiring that could be broken independently of the
        trainee-wins case (e.g. sec_box_terminal or box_terminal computed
        correctly but passed to the wrong parameter name)."""
        _trainer, env, reward, info = _force_scoring_episode(
            scorer_id="opponent", victim_id="trainee", expected_outcome="opponent_box_possession",
        )

        assert env.last_reward_components.get("box", 0.0) == 0.0, (
            "trainee's own row must NOT carry a win bonus -- it lost"
        )
        assert env.last_reward_components.get("lterm", 0.0) < 0.0, (
            "trainee's own row must carry ITS OWN loss penalty"
        )
        assert reward < -0.5, (
            f"trainee's total reward={reward} should be dominated by the "
            "-2.5 loss_terminal penalty"
        )

        sec = _secondary_row(env, "opponent")
        assert sec["reward_components"].get("box", 0.0) > 0.0, (
            "the winning secondary opponent's own row must carry ITS OWN "
            "win bonus (sec_box_terminal must be wired to "
            "reached_opponent_box_with_possession for the secondary's "
            "call)"
        )
        assert sec["reward_components"].get("lterm", 0.0) == 0.0, (
            "the winning secondary opponent's own row must NOT carry the "
            "trainee's own loss penalty -- it won"
        )
        assert sec["reward"] > 0.5, (
            f"opponent's total reward={sec['reward']} should be dominated "
            "by the +3.0 box_possession_terminal bonus"
        )
        assert sec["done"] == 1.0

    def test_neither_scores_on_an_ordinary_tick(self):
        """Regression guard against the cross-wiring firing on a NULL
        event: an ordinary kickoff tick (no forcing at all) must credit
        neither player with box/lterm on either side. Also the natural
        control for the two tests above -- proves the +/- signals seen
        there come from the forced scoring event, not from box/lterm firing
        unconditionally every tick regardless of what actually happened."""
        trainer = PPOTrainer.from_config()
        env = _make_self_play_env(trainer)

        _obs, _reward, done, _info = env.step()

        assert done is False, "a fresh kickoff tick must not immediately end the episode"
        assert env.last_reward_components.get("box", 0.0) == 0.0
        assert env.last_reward_components.get("lterm", 0.0) == 0.0

        sec = _secondary_row(env, "opponent")
        assert sec["reward_components"].get("box", 0.0) == 0.0
        assert sec["reward_components"].get("lterm", 0.0) == 0.0


class TestScoringAttributionSurvivesIntoRolloutBuffer:
    """Same two forced-scoring scenarios as above, but pushed all the way
    through RolloutBuffer.add() (using the SAME weight=secondary_weight,
    track_id=player_id pattern as ppo_trainer.py's real train() loop, line
    ~1500) and compute_gae() -- proving the correct signs actually reach
    the data structure PPO's per-track GAE segmentation and the policy
    gradient consume, not just env.last_reward_components/
    last_secondary_results in isolation."""

    def _collect_buffer(self, *, scorer_id: str, victim_id: str, expected_outcome: str) -> RolloutBuffer:
        trainer, env, reward, _info = _force_scoring_episode(
            scorer_id=scorer_id, victim_id=victim_id, expected_outcome=expected_outcome,
        )

        buffer = RolloutBuffer()
        tr = env.last_trainee_transition
        assert tr is not None, "trainee must be due for a decision on tick 1 of a fresh episode"
        buffer.add(
            obs=tr["obs"], action=_action_to_numpy(tr["action"], tr["raw_exec"]),
            log_prob=tr["log_prob"], value=tr["value"], reward=reward,
            done=1.0,
            reward_comps=dict(env.last_reward_components),
        )
        for sec in env.last_secondary_results:
            buffer.add(
                obs=sec["obs"], action=_action_to_numpy(sec["action"], sec["raw_exec"]),
                log_prob=sec["log_prob"], value=sec["value"], reward=sec["reward"],
                done=sec["done"], weight=trainer._secondary_weight, track_id=sec["player_id"],
                reward_comps=dict(sec["reward_components"]),
            )
        assert set(buffer.track_ids) == {"trainee", "opponent"}
        return buffer

    def test_trainee_win_buffer_rewards_and_returns_have_correct_sign(self):
        buffer = self._collect_buffer(
            scorer_id="trainee", victim_id="opponent", expected_outcome="box_possession",
        )

        by_track = dict(zip(buffer.track_ids, buffer.rewards))
        assert by_track["trainee"] > 0.5
        assert by_track["opponent"] < -0.5

        advantages, returns = buffer.compute_gae(gamma=0.99, lam=0.95, last_value=0.0)
        returns_by_track = dict(zip(buffer.track_ids, returns))
        # Both rows are the terminal (done=1) step of their own track, so
        # no bootstrap applies -- return must equal the raw reward exactly.
        assert returns_by_track["trainee"] == pytest.approx(by_track["trainee"])
        assert returns_by_track["opponent"] == pytest.approx(by_track["opponent"])
        assert returns_by_track["trainee"] > 0.0, (
            "a value-gradient step on the winning track's return must push "
            "the policy TOWARD what it just did"
        )
        assert returns_by_track["opponent"] < 0.0, (
            "a value-gradient step on the losing track's return must push "
            "the policy AWAY from what it just did -- if this were "
            "positive, self-play would be reinforcing losing behaviour"
        )
        for a in advantages:
            assert math.isfinite(a)

    def test_opponent_win_buffer_rewards_and_returns_have_correct_sign(self):
        buffer = self._collect_buffer(
            scorer_id="opponent", victim_id="trainee", expected_outcome="opponent_box_possession",
        )

        by_track = dict(zip(buffer.track_ids, buffer.rewards))
        assert by_track["trainee"] < -0.5
        assert by_track["opponent"] > 0.5

        advantages, returns = buffer.compute_gae(gamma=0.99, lam=0.95, last_value=0.0)
        returns_by_track = dict(zip(buffer.track_ids, returns))
        assert returns_by_track["trainee"] == pytest.approx(by_track["trainee"])
        assert returns_by_track["opponent"] == pytest.approx(by_track["opponent"])
        assert returns_by_track["trainee"] < 0.0
        assert returns_by_track["opponent"] > 0.0
        for a in advantages:
            assert math.isfinite(a)

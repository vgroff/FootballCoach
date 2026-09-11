"""End-to-end coverage for PPO training with a NEURAL secondary opponent.

Until now, ``phase1_opponent_neural_ratio`` has always been 0 in
ai_config.json (see ai/knowledge.md and agent_plans/ -- the opponent audit),
so ``env.last_secondary_results`` was always empty and RolloutBuffer never
actually contained more than one chronological track. That meant a real
bug in ``RolloutBuffer.compute_gae()`` -- a flat backward scan that let the
trainee's and the secondary opponent's interleaved rows leak into each
other's "next step" bootstrap -- was never exercised by any test. See
tests/ai_unit/test_gae.py's ``TestGAEMultiTrack`` for the isolated
buffer-level regression tests for that bug; this file instead exercises the
REAL ``ScenarioEnv`` + ``PPOTrainer`` pipeline with a genuinely
neural-controlled secondary player, end to end, so a future regression in
how ppo_trainer.py/rollout_worker.py wire ``track_id``/per-track bootstrap
values together (not just the buffer math in isolation) would be caught.

Deliberately does NOT touch ai_config.json on disk (this repo's test suite
runs under pytest-xdist -- see ``bringing up nodes`` in CI output -- so
mutating the shared config file mid-run risks a concurrent test process
picking up the wrong curriculum ratios). The neural-opponent env is built by
constructing ``ScenarioEnv``/``build_1v1_scenario`` directly with explicit
kwargs instead of going through ``curriculum/envs.py::build_env()``, which
is what makes this test config-file-independent. This also means only the
single-process ``PPOTrainer.train()`` path is covered here -- the parallel
worker path (``rollout_worker.py``) shares the exact same
``PPOTrainer._bootstrap_last_values()`` helper and the same
``RolloutBuffer.add(track_id=...)`` call pattern (see the diff), so it is
not independently subprocess-tested with a neural opponent for the same
config-file-mutation reason ``test_parallel_rollout.py`` avoids it too.
"""
from __future__ import annotations

import math

import torch

from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _action_to_numpy
from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario

_ROLLOUT_STEPS = 20


def _make_env_with_neural_opponent() -> ScenarioEnv:
    """1v1 env where BOTH the trainee and the secondary player are neural.

    ``opponent_rules_prob=0.0, opponent_immobile_prob=0.0`` makes
    ``build_1v1_scenario`` roll into the neural remainder every episode
    (see ``ui/scenarios.py``'s per-episode roll), and
    ``secondary_player_ids=["opponent"]`` is what makes ``ScenarioEnv``
    assign a ``NeuralPlayerAI`` to that player and start populating
    ``env.last_secondary_results`` each tick -- mirrors
    ``curriculum/envs.py::_build_phase1_env()`` exactly, just with the
    ratios hardcoded here instead of read from ai_config.json.
    """
    import functools

    defn = ScenarioDefinition(
        key="secondary_opponent_gae_test",
        label="Secondary opponent GAE test",
        description="1v1 with a fully neural secondary opponent",
        build=functools.partial(
            build_1v1_scenario, opponent_rules_prob=0.0, opponent_immobile_prob=0.0,
        ),
    )
    return ScenarioEnv(
        definition=defn,
        trainee_player_id="trainee",
        phase=1,
        secondary_player_ids=["opponent"],
        max_episode_s=30.0,
    )


def _collect_rollout_with_secondary(
    env: ScenarioEnv, trainer: PPOTrainer, n: int,
) -> tuple[RolloutBuffer, object]:
    """Same shape as test_smoke.py's ``_collect_rollout``, but also drains
    ``env.last_secondary_results`` with ``track_id=sec["player_id"]`` --
    i.e. exactly what ``PPOTrainer.train()``'s inner loop does."""
    buffer = RolloutBuffer()
    env.sample_action_fn = trainer._sample_action
    env.reset()
    next_obs = None

    for _ in range(n):
        next_obs, reward, done, _info = env.step()
        tr = env.last_trainee_transition
        if tr is not None:
            buffer.add(
                obs=tr["obs"],
                action=_action_to_numpy(tr["action"], tr["raw_exec"]),
                log_prob=tr["log_prob"],
                value=tr["value"],
                reward=reward,
                done=1.0 if done else 0.0,
            )
        for sec in getattr(env, "last_secondary_results", []):
            buffer.add(
                obs=sec["obs"],
                action=_action_to_numpy(sec["action"], sec["raw_exec"]),
                log_prob=sec["log_prob"],
                value=sec["value"],
                reward=sec["reward"],
                done=sec["done"],
                track_id=sec["player_id"],
            )
        if done:
            env.reset()

    return buffer, next_obs


def test_neural_secondary_opponent_produces_transitions():
    """Sanity check on the env fixture itself: with both rules/immobile
    ratios at 0.0, the secondary player must actually be neural-controlled
    and produce transitions -- otherwise every other test in this file is
    vacuously exercising the immobile-opponent (single-track) case."""
    env = _make_env_with_neural_opponent()
    trainer = PPOTrainer.from_config()
    env.sample_action_fn = trainer._sample_action
    env.reset()

    saw_secondary = False
    for _ in range(_ROLLOUT_STEPS):
        env.step()
        for sec in getattr(env, "last_secondary_results", []):
            saw_secondary = True
            assert sec["player_id"] == "opponent"
            assert math.isfinite(sec["reward"])

    assert saw_secondary, (
        "env.last_secondary_results was empty for the whole rollout -- the "
        "neural-opponent fixture isn't actually producing secondary "
        "transitions, so this test file isn't testing what it claims to"
    )


def test_rollout_buffer_has_both_tracks():
    env = _make_env_with_neural_opponent()
    trainer = PPOTrainer.from_config()
    buffer, _next_obs = _collect_rollout_with_secondary(env, trainer, _ROLLOUT_STEPS)

    assert set(buffer.track_ids) == {"trainee", "opponent"}
    assert len(buffer.track_ids) == len(buffer.rewards)


def test_bootstrap_last_values_covers_every_active_track():
    env = _make_env_with_neural_opponent()
    trainer = PPOTrainer.from_config()
    buffer, next_obs = _collect_rollout_with_secondary(env, trainer, _ROLLOUT_STEPS)

    last_values = trainer._bootstrap_last_values(env, next_obs, buffer)

    assert "trainee" in last_values
    assert "opponent" in last_values
    for track, v in last_values.items():
        assert math.isfinite(v), f"last_values[{track!r}]={v} is not finite"


def test_ppo_update_with_neural_opponent_produces_finite_losses():
    """Full pipeline: collect a rollout with BOTH tracks present, bootstrap
    per-track values, compute (now correctly segmented) GAE, run one PPO
    update. Would have silently trained on corrupted advantages before this
    fix -- can't assert "wrong number" directly in an integration test, but
    this at minimum guards against a crash/NaN/shape-mismatch regression in
    how ppo_trainer.py wires track_id + per-track bootstrapping together."""
    env = _make_env_with_neural_opponent()
    trainer = PPOTrainer.from_config()
    buffer, next_obs = _collect_rollout_with_secondary(env, trainer, _ROLLOUT_STEPS)
    assert set(buffer.track_ids) == {"trainee", "opponent"}, (
        "fixture didn't produce both tracks this run -- rerun or increase "
        "_ROLLOUT_STEPS; the assertions below would be vacuous otherwise"
    )

    last_values = trainer._bootstrap_last_values(env, next_obs, buffer)
    advantages, returns = buffer.compute_gae(trainer.gamma, trainer.lam, last_values)
    assert len(advantages) == len(buffer)
    for a, r in zip(advantages, returns):
        assert math.isfinite(a)
        assert math.isfinite(r)

    batch = buffer.as_tensors(advantages, returns)
    metrics = trainer._ppo_update(batch, progress=0.0)
    for key, val in metrics.items():
        if not isinstance(val, (int, float)):
            continue
        assert math.isfinite(val), f"PPO metrics['{key}']={val} is not finite"


def test_train_single_process_with_neural_opponent_end_to_end():
    """Calls the REAL ``PPOTrainer.train()`` (not a hand-rolled loop) with a
    neural secondary opponent, forcing the single-process path
    (n_processes=1) so it stays fast and doesn't touch ai_config.json.
    This is the strongest regression guard for this fix: it exercises the
    actual shipped train() code, including the track_id=sec["player_id"]
    buffer.add() call and the _bootstrap_last_values() call, rather than a
    test-local reimplementation of that loop.

    ``checkpoint_dir`` is left at its default (None) -- checkpoint
    save/load is unrelated to this fix and already covered by
    test_separate_value_net.py/test_parallel_rollout.py (which currently
    fail on this Windows dev machine for an unrelated, pre-existing reason:
    ``latest.symlink_to()`` requires a privilege this environment doesn't
    grant -- see ai_trainer_knowledge.md). Setting a real checkpoint_dir
    here would just fail the same way for the same unrelated reason.
    """
    env = _make_env_with_neural_opponent()
    trainer = PPOTrainer.from_config()
    trainer.n_processes = 1  # force single-process path regardless of config
    trainer.rollout_steps = _ROLLOUT_STEPS

    trainer.train(env, total_steps=_ROLLOUT_STEPS)

    assert trainer._total_steps >= _ROLLOUT_STEPS

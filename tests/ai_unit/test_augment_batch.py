"""Direct unit coverage for ai/obs/augment.py's augment_batch() -- previously
had ZERO dedicated tests despite running on every real PPO update
(ai_config.json's augment_n_slot_shuffles=1 by default, applied
unconditionally inside _ppo_update() whenever > 0). Focused specifically on
the list-typed fields (reward_comps_raw, step_outcomes, track_ids) -- the
part most likely to silently break, since augment_batch() builds its output
dict key-by-key rather than copying unknown keys through, so a new
RolloutBuffer.as_tensors() field (like track_ids, added to support self-play
diagnostics) is silently DROPPED unless augment_batch() is explicitly
updated to carry it. This uses a REAL small self-play rollout (both tracks
present) rather than hand-built synthetic tensors, so shapes/dtypes are
guaranteed to match what _ppo_update() actually feeds this function.
"""
from __future__ import annotations

import functools
import math

import numpy as np
import torch

from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.obs.augment import augment_batch
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _action_to_numpy
from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario

_ROLLOUT_STEPS = 24


def _make_self_play_batch() -> dict:
    """Collect a small real rollout with a neural secondary opponent (both
    tracks present) and return its as_tensors() batch dict -- exactly the
    shape augment_batch() receives from _ppo_update()."""
    trainer = PPOTrainer.from_config(device=torch.device("cpu"))
    det_fn = functools.partial(trainer._sample_action, deterministic=True)

    defn = ScenarioDefinition(
        key="augment_batch_test", label="augment_batch test",
        description="1v1 with a neural secondary opponent, for augment_batch coverage",
        build=functools.partial(build_1v1_scenario, opponent_rules_prob=0.0, opponent_immobile_prob=0.0),
    )
    env = ScenarioEnv(
        definition=defn, trainee_player_id="trainee", phase=1,
        secondary_player_ids=["opponent"], max_episode_s=30.0,
    )
    env.sample_action_fn = det_fn
    env.reset()

    buffer = RolloutBuffer()
    collected = 0
    while collected < _ROLLOUT_STEPS:
        next_obs, reward, done, _info = env.step()
        tr = env.last_trainee_transition
        if tr is not None:
            buffer.add(
                obs=tr["obs"], action=_action_to_numpy(tr["action"], tr["raw_exec"]),
                log_prob=tr["log_prob"], value=tr["value"], reward=reward,
                done=1.0 if done else 0.0,
                reward_comps=dict(getattr(env, "last_reward_components", {})),
                step_outcome="",
            )
            collected += 1
        for sec in getattr(env, "last_secondary_results", []):
            buffer.add(
                obs=sec["obs"], action=_action_to_numpy(sec["action"], sec["raw_exec"]),
                log_prob=sec["log_prob"], value=sec["value"], reward=sec["reward"], done=sec["done"],
                weight=trainer._secondary_weight, track_id=sec["player_id"],
            )
            collected += 1
        if done:
            env.reset()

    assert set(buffer.track_ids) == {"trainee", "opponent"}, (
        "fixture didn't produce both tracks -- widen _ROLLOUT_STEPS"
    )
    advantages = [0.0] * len(buffer)
    returns = [0.0] * len(buffer)
    return buffer.as_tensors(advantages, returns)


class TestAugmentBatchTrackIds:
    def test_track_ids_present_and_correctly_sized(self):
        batch = _make_self_play_batch()
        n_original = len(batch["track_ids"])

        augmented = augment_batch(batch, n_slot_shuffles=2, rng=__import__("random").Random(0))

        expected_multiplier = 2 * 2  # N_FLIP_VARIANTS(2) x n_slot_shuffles(2)
        assert "track_ids" in augmented, (
            "augment_batch() silently dropped track_ids -- it builds its "
            "output dict key-by-key rather than copying unknown keys "
            "through, so a new RolloutBuffer.as_tensors() field must be "
            "explicitly wired into augment_batch() or it vanishes the "
            "moment augmentation runs"
        )
        assert len(augmented["track_ids"]) == n_original * expected_multiplier
        # Every other tensor field must have the SAME row count -- if
        # track_ids silently desynced in length, indexing augmented["track_ids"][i]
        # against augmented["rewards"][i] downstream would silently pair the
        # wrong row's track with the wrong row's data.
        assert augmented["rewards"].shape[0] == len(augmented["track_ids"])

    def test_track_ids_correctly_tiled_not_scrambled(self):
        """Each of the expected_multiplier augmented copies must reproduce
        the ORIGINAL track_id sequence verbatim (track identity is not
        geometric/slot content, so it must pass through unchanged in every
        copy, in the same relative order) -- not just "the right total
        count of each value" (which a bug could satisfy by accident, e.g.
        by using the wrong tiling axis)."""
        batch = _make_self_play_batch()
        original = list(batch["track_ids"])
        n = len(original)

        augmented = augment_batch(batch, n_slot_shuffles=2, rng=__import__("random").Random(0))
        expected_multiplier = 4
        aug_ids = augmented["track_ids"]
        assert len(aug_ids) == n * expected_multiplier

        for copy_i in range(expected_multiplier):
            chunk = aug_ids[copy_i * n:(copy_i + 1) * n]
            assert chunk == original, (
                f"augmented copy {copy_i}'s track_id sequence doesn't match "
                f"the original -- expected {original}, got {chunk}"
            )

    def test_finite_and_shape_consistent_with_n_slot_shuffles_1(self):
        """The actual production default (ai_config.json's
        augment_n_slot_shuffles=1) -- smoke-checks the whole function stays
        well-formed and finite at that specific multiplier, not just the
        n_slot_shuffles=2 value the other tests use."""
        batch = _make_self_play_batch()
        n = len(batch["track_ids"])

        augmented = augment_batch(batch, n_slot_shuffles=1, rng=__import__("random").Random(1))
        assert len(augmented["track_ids"]) == n * 2  # 2 flip variants x 1 shuffle
        assert set(augmented["track_ids"]) == {"trainee", "opponent"}
        assert math.isfinite(float(augmented["rewards"].sum()))
        assert bool(torch.isfinite(augmented["obs/self_feat"]).all())

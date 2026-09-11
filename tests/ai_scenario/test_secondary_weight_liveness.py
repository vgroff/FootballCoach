"""Does the secondary/opponent player in self-play actually train, or could
it silently be coasting on a stale/frozen reference to old weights?

The architecture is sound on inspection: ``env.sample_action_fn`` is a bound
method on the SAME ``PPOTrainer`` instance for both the trainee and every
secondary player, and worker processes refresh weights via
``_load_state_dict_tolerant`` loading directly into ``trainer.decision_net``/
``.execution_net``/``.value_net`` IN PLACE (``batched_rollout_worker.py``'s
``set_weights`` message handler) -- the same objects ``env.sample_action_fn``
already closes over. But until now nothing asserted this BEHAVIORALLY:
``grep -r set_weights tests/`` matches nothing. If a future refactor ever
captured a module reference before a reload instead of reading the live
attribute, self-play would silently start training against a permanently
frozen opponent -- no crash, no error, and (per the self-play audit this
session) no eval signal to catch it either, since periodic eval never plays
self-play at all.

TestSingleProcessLiveness verifies the core mechanism in-process, via
_load_state_dict_tolerant directly (the exact function the real
multi-process IPC path also calls to apply set_weights) -- fast,
deterministic. See the note at the bottom of this file for why a real
spawned-subprocess version (closing the gap for real production training,
which always runs multi-process) was attempted but not shipped.
"""
from __future__ import annotations

import functools

import pytest
import torch

from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _load_state_dict_tolerant
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario

_SEED = 777


def _make_self_play_env(trainer: PPOTrainer) -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="weight_liveness_test", label="Weight liveness test",
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
    return env


def _secondary_value_at_seed(trainer: PPOTrainer, seed: int) -> float:
    """The secondary player's OWN critic value for the first decision of a
    fully deterministic, fixed-seed episode -- reproducible given the same
    trainer weights, so any difference across two calls is attributable
    ONLY to a weight change in between."""
    env = _make_self_play_env(trainer)
    env.reset(seed=seed)
    env.step()
    [sec] = [r for r in env.last_secondary_results if r["player_id"] == "opponent"]
    return float(sec["value"])


class TestSingleProcessLiveness:
    def test_reloading_weights_changes_secondary_value_and_matches_source_trainer(self):
        trainer_a = PPOTrainer.from_config()
        trainer_b = PPOTrainer.from_config()

        value_a_before = _secondary_value_at_seed(trainer_a, _SEED)
        value_b = _secondary_value_at_seed(trainer_b, _SEED)
        assert value_a_before != pytest.approx(value_b, abs=1e-6), (
            "fixture check: trainer_a and trainer_b must actually produce "
            "different output for the identical seeded observation, or "
            "this test can't distinguish 'reload worked' from 'reload was "
            "a no-op' -- two independently random-initialized trainers "
            "producing the exact same value is vanishingly unlikely; if "
            "this fires, something is wrong with the fixture, not the "
            "weight-liveness property being tested"
        )

        # Mirrors EXACTLY what batched_rollout_worker.py's set_weights
        # message handler does: load the OTHER trainer's state dicts into
        # this trainer's nets in place.
        _load_state_dict_tolerant(trainer_a.decision_net, trainer_b.decision_net.state_dict(), "decision_net")
        _load_state_dict_tolerant(trainer_a.execution_net, trainer_b.execution_net.state_dict(), "execution_net")
        if trainer_a.value_net is not None and trainer_b.value_net is not None:
            _load_state_dict_tolerant(trainer_a.value_net, trainer_b.value_net.state_dict(), "value_net")

        value_a_after = _secondary_value_at_seed(trainer_a, _SEED)

        assert value_a_after != pytest.approx(value_a_before, abs=1e-6), (
            "trainer_a's secondary-player sampling didn't change after its "
            "weights were reloaded in place -- looks frozen/stale, exactly "
            "the failure mode that would make self-play train against a "
            "permanently-fixed opponent with no error or crash"
        )
        assert value_a_after == pytest.approx(value_b, abs=1e-4), (
            f"trainer_a's post-reload secondary value ({value_a_after}) "
            f"should now MATCH trainer_b's own value for the identical "
            f"observation ({value_b}) -- it merely being DIFFERENT from "
            "before isn't enough proof; it must specifically reflect the "
            "weights that were just loaded, not some other source of "
            "variation"
        )

# A real spawned-subprocess version of this test (spawn_batched_workers +
# the actual set_weights Pipe command, poisoning the worker's value_net
# with all-zero weights and checking secondary-player values reload to
# ~0.0) was attempted and deliberately NOT shipped here. Two runs gave two
# different, inconclusive results -- one where post-reload values stayed
# close to their pre-reload values instead of going to ~0.0, and one that
# hung entirely -- and mid-investigation it turned out this machine has a
# REAL, LIVE PPO training run active (phase 1, --device cuda,
# --total-steps 12000000, with several of its own rollout-worker
# subprocesses), which had been competing for CPU with every subprocess
# test in this session, this one included. That's a very plausible
# explanation for a hang and for unreliable timing, but it does NOT rule
# out a real bug in the subprocess set_weights path -- and shipping a test
# that might assert something false, or might just be flaky under
# contention, is worse than not shipping one. TestSingleProcessLiveness
# above still directly verifies the core mechanism (_load_state_dict_tolerant
# writing into the SAME nn.Module objects env.sample_action_fn already
# closes over) that the real IPC path also relies on. The subprocess-level
# version is real, valuable follow-up work -- re-run in isolation (no
# concurrent training) before trusting either the earlier failure or a
# clean pass.

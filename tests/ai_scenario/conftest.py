"""Shared test-session tuning for tests/ai_scenario.

ai_config.json's live default (``eval.eval_n_parallel_workers: 6``) is tuned
for real training throughput, not test speed: PPOTrainer._eval_vs_rules()
(called internally by pretrain_combined()/train()) spawns a fresh N-process
pool via run_seeded_evaluation_parallel() every time it's invoked, and
pretrain_combined() calls it twice per run (each doing two sub-evals,
vs-immobile and vs-rules) -- up to four pool spawn/teardown cycles per test.
With pytest-xdist's own default -n 5 outer parallelism on top, that fans out
to 5 x 6 = 30 concurrent subprocesses fighting over far fewer real cores,
which thrashes rather than helps and was confirmed (via direct process-tree
inspection) to make this directory ~20x slower per-test than every other
test directory in the suite.

These are correctness smoke/unit tests, not throughput benchmarks -- they
don't need real multi-worker eval parallelism to do their job, just SOME
n_workers>1 coverage of the parallel-eval code path itself. Capping (not
disabling) it at 2 here keeps that coverage while cutting the fan-out
multiplier by 3x relative to the live default.
"""
from __future__ import annotations

import pytest

from footballcoach.ai.config import load_ai_config

_TEST_EVAL_N_PARALLEL_WORKERS = 2

# Real production sizes (ai_config.json's live "network" section) exist to
# give the trained policy real capacity -- these tests check WIRING
# (freezing, routing, checkpoint round-trip, optimizer membership,
# non-finite guards), never learned behaviour, so they don't need anywhere
# near this much capacity. Shrinking cuts both network-construction cost AND
# every forward/backward pass's cost, on top of the eval-parallelism fix
# above. Kept internally consistent (entity_embed_dim divisible by both
# attn-head counts) and deliberately UNEQUAL across dims (16 vs 8 vs 12 ...)
# so a wiring bug that accidentally swaps two dims still surfaces as a shape
# mismatch instead of silently working. `None` (network_test.get(...) or
# real default) preserves any key not listed here.
_TEST_NETWORK_OVERRIDES = {
    "entity_embed_dim": 16,
    "num_attention_heads": 4,
    "inter_player_attn_heads": 4,
    "self_mlp_hidden": 12,
    "ball_mlp_hidden": 8,
    "global_mlp_hidden": 8,
    "decision_mlp_hidden": 16,
    "trunk_hidden": 24,
    "exec_trunk_hidden": 16,
    "value_net_trunk_hidden": 20,
    "latent_dim": 12,
    "value_extra_hidden": 8,
    "value_hidden_dim": 8,
    # Skip loading the real pretrained physics-encoder checkpoints entirely
    # -- disk I/O + building those frozen submodules for every fresh
    # PPOTrainer.from_config()/DecisionNetwork.from_config() call in these
    # tests, for a feature none of them are testing.
    "ball_physics_encoder_checkpoint": None,
    "player_physics_encoder_checkpoint": None,
}


@pytest.fixture(autouse=True)
def _cap_eval_parallelism_and_shrink_networks():
    # load_ai_config() is functools.lru_cache(maxsize=1) with no args, so
    # every caller anywhere in the process -- regardless of which module's
    # own `from footballcoach.ai.config import load_ai_config` import they
    # used -- shares this exact same dict object. Mutating it in place (and
    # restoring after) reaches every PPOTrainer/DecisionNetwork/
    # ExecutionNetwork construction path in this directory's tests without
    # needing to monkeypatch per-module import bindings individually.
    cfg = load_ai_config()

    eval_cfg = cfg.setdefault("eval", {})
    had_eval_key = "eval_n_parallel_workers" in eval_cfg
    original_eval = eval_cfg.get("eval_n_parallel_workers")
    eval_cfg["eval_n_parallel_workers"] = _TEST_EVAL_N_PARALLEL_WORKERS

    net_cfg = cfg.setdefault("network", {})
    originals_net = {k: net_cfg.get(k, "__absent__") for k in _TEST_NETWORK_OVERRIDES}
    net_cfg.update(_TEST_NETWORK_OVERRIDES)

    yield

    if had_eval_key:
        eval_cfg["eval_n_parallel_workers"] = original_eval
    else:
        eval_cfg.pop("eval_n_parallel_workers", None)
    for k, v in originals_net.items():
        if v == "__absent__":
            net_cfg.pop(k, None)
        else:
            net_cfg[k] = v


@pytest.fixture(scope="session", autouse=True)
def _disable_mask_untriggered_actions_for_generic_trainers():
    """ppo.mask_untriggered_actions (live config: true) is only implemented
    for curriculum phases that freeze pass/tackle/mark (PPOTrainer.
    _opp_head_mask_matrix() raises NotImplementedError otherwise -- see
    ai_trainer_knowledge.md / the masked-action-training plan). The generic
    PPOTrainer.from_config() fixtures all over this directory build a trainer
    with every head trainable (no curriculum phase applied), so picking up
    the live value here makes any of them that reach _ppo_update() trip that
    guard -- nothing to do with what they're each actually testing.

    SESSION-scoped (not folded into _cap_eval_parallelism_and_shrink_networks
    above, which is function-scoped): several test modules in this directory
    (e.g. test_val_episode_split.py) build their shared PPOTrainer in a
    scope="module" fixture, and pytest sets up broader-scope fixtures BEFORE
    narrower-scope ones -- a function-scoped override here would run AFTER
    that module-scoped trainer already baked in the live config value at
    construction time, too late to matter. Session scope guarantees this
    runs before any fixture in this directory, module-scoped or not.

    Dedicated coverage for the masking feature itself
    (test_masked_action_training.py) sets ``trainer.mask_untriggered_actions``
    directly on its own trainer instances AFTER freezing the right heads,
    bypassing this config value entirely, so it's unaffected by the override
    below.
    """
    cfg = load_ai_config()
    ppo_cfg = cfg.setdefault("ppo", {})
    had_mask_key = "mask_untriggered_actions" in ppo_cfg
    original_mask = ppo_cfg.get("mask_untriggered_actions")
    ppo_cfg["mask_untriggered_actions"] = False

    yield

    if had_mask_key:
        ppo_cfg["mask_untriggered_actions"] = original_mask
    else:
        ppo_cfg.pop("mask_untriggered_actions", None)

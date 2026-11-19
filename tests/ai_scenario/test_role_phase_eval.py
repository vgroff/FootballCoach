"""Per-phase seeded evals for role-based phases (ai/eval/role_phase_eval.py and
PPOTrainer._eval_role_phases)."""
from __future__ import annotations

import logging

import pytest
import torch

import footballcoach.ai.eval.role_phase_eval as rpe
from footballcoach.ai.config import load_ai_config
from footballcoach.ai.curriculum.envs import build_role_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID

PHASE2 = PHASES_BY_ID[2]
SPECS = {s["name"]: s for s in PHASE2.eval_specs}
TABLE = load_ai_config()["reward"]["phase2"]["terminal"]


@pytest.fixture(scope="module")
def trainer():
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    torch.manual_seed(0)
    t = PPOTrainer.from_config(device=torch.device("cpu"))
    t.set_frozen_heads(PHASES_BY_ID[1].frozen_heads)
    return t


NEURAL_ATTACKER = {"attacker": {"rules": 0, "neural": 1}, "defender": {"rules": 1, "neural": 0},
                   "goalkeeper": {"rules": 1, "neural": 0}}


@pytest.fixture
def controllers(monkeypatch):
    cfg = load_ai_config()["curriculum"]["phase2"]

    def _set(value):
        monkeypatch.setitem(cfg, "controllers", value)
    return _set


def test_the_eval_plan_is_the_phase1_budget_on_the_phase_seed_range():
    cfg = load_ai_config()
    seeds, repeats = rpe.eval_plan(PHASE2)
    assert seeds[0] == cfg["curriculum"]["phase2"]["eval_seed_offset"]
    assert len(seeds) == cfg["eval"]["eval_n_seeds"] and repeats == cfg["eval"]["eval_repeats_per_seed"]


def test_only_specs_of_trained_roles_are_evaluated(controllers):
    controllers(NEURAL_ATTACKER)
    assert [s["name"] for s in rpe.specs_to_evaluate(PHASE2)] == [
        "rules_baseline", "attacker_vs_rules", "rules_baseline_no_keeper", "attacker_vs_rules_no_keeper"]
    controllers({**NEURAL_ATTACKER, "defender": {"rules": 1, "neural": 2}})
    assert [s["name"] for s in rpe.specs_to_evaluate(PHASE2)] == list(SPECS)
    controllers({r: {"rules": 1, "neural": 0} for r in ("attacker", "defender", "goalkeeper")})
    assert [s["name"] for s in rpe.specs_to_evaluate(PHASE2)] == ["rules_baseline", "rules_baseline_no_keeper"]


def test_explicit_seeds_with_repeats():
    res = rpe.run_role_phase_eval(PHASE2, SPECS["rules_baseline"], None, seeds=[5000, 5001], repeats_per_seed=3)
    assert len(res.outcomes) == 6


def test_rules_baseline_matches_the_training_env_on_the_same_seeds():
    seeds = list(range(5000, 5012))
    res = rpe.run_role_phase_eval(PHASE2, SPECS["rules_baseline"], sample_action_fn=None, seeds=seeds)
    env = build_role_env(PHASE2, controller_weights={r: {"rules": 1} for r in ("attacker", "defender", "goalkeeper")})
    expected = []
    for s in seeds:
        env.reset(seed=s)
        done = False
        while not done:
            _, _, done, info = env.step()
        expected.append(info.trial_outcome)
    assert res.outcomes == expected
    assert res.rewards == [TABLE["attacker"][o] for o in expected]
    assert res.episode_tag == "phase2_1v2[attacker=rules,defender=rules,goalkeeper=rules]"


@pytest.mark.parametrize("spec,role", [("attacker_vs_rules", "attacker"), ("defender_vs_rules", "defender")])
def test_neural_spec_reports_the_neural_roles_reward(trainer, spec, role):
    res = rpe.run_role_phase_eval(PHASE2, SPECS[spec], trainer._sample_action, seeds=[7, 8, 9])
    assert len(res.outcomes) == 3
    assert res.rewards == [TABLE[role][o] for o in res.outcomes]
    assert f"{role}=neural" in res.episode_tag
    assert res.summary().startswith(res.episode_tag + "(3): goal=")


def test_a_spec_must_pin_every_role():
    with pytest.raises(ValueError, match="pin exactly"):
        rpe.run_role_phase_eval(PHASE2, {"name": "x", "controllers": {"attacker": "rules"}}, None, seeds=[1])


def test_trainer_runs_every_spec_and_caches_the_rules_baseline(trainer, monkeypatch, caplog, controllers):
    controllers({**NEURAL_ATTACKER, "defender": {"rules": 1, "neural": 1}})
    calls = []
    real = rpe.run_role_phase_eval

    def _counting(phase, spec, fn, seeds=None, repeats_per_seed=None):
        calls.append(spec["name"])
        return real(phase, spec, fn, seeds=[11, 12])
    monkeypatch.setattr(rpe, "run_role_phase_eval", _counting)
    trainer.phase_mix = {1: 3.0, 2: 1.0}
    trainer._role_eval_cache = {}
    try:
        with caplog.at_level(logging.INFO, logger="footballcoach"):
            trainer._eval_role_phases()
            trainer._eval_role_phases()
    finally:
        trainer.phase_mix = None
    # Rules-only specs run once (cached); the neural ones every time.
    assert calls == ["rules_baseline", "attacker_vs_rules", "defender_vs_rules",
                     "rules_baseline_no_keeper", "attacker_vs_rules_no_keeper",
                     "attacker_vs_rules", "defender_vs_rules", "attacker_vs_rules_no_keeper"]
    for name in SPECS:
        assert caplog.text.count(f"[eval phase2_1v2/{name}]") == 2


def _dispatch(trainer, monkeypatch, phase_mix, train_phase):
    calls = []
    monkeypatch.setattr(trainer, "rollout_eval_trials", 1)
    monkeypatch.setattr(trainer, "phase_mix", phase_mix)
    monkeypatch.setattr(trainer, "_train_phase_id", train_phase)
    monkeypatch.setattr(trainer, "_eval_vs_rules", lambda s: calls.append("phase1_vs_rules"))
    monkeypatch.setattr(trainer, "_maybe_run_neural_snapshot_eval", lambda s: calls.append("phase1_snapshots"))
    monkeypatch.setattr(trainer, "_eval_role_phases", lambda: calls.append(("role", tuple(trainer._trained_phase_ids()))))
    trainer._run_rollout_evals(15.0)
    return calls


def test_rollout_evals_cover_exactly_the_trained_phases(trainer, monkeypatch):
    assert _dispatch(trainer, monkeypatch, None, 1) == ["phase1_vs_rules", "phase1_snapshots", ("role", (1,))]
    assert _dispatch(trainer, monkeypatch, None, 2) == [("role", (2,))]
    assert _dispatch(trainer, monkeypatch, {2: 1.0}, 2) == [("role", (2,))]
    assert _dispatch(trainer, monkeypatch, {1: 3.0, 2: 1.0}, 1) == ["phase1_vs_rules", "phase1_snapshots", ("role", (1, 2))]
    assert _dispatch(trainer, monkeypatch, None, None) == ["phase1_vs_rules", "phase1_snapshots", ("role", ())]


def test_phase1_evals_use_phase1_episode_length(trainer, monkeypatch):
    """A --phase 2 run with phase 1 in the mix passes phase 2's episode length
    to the rollout evals; the phase-1 evals must still use phase 1's."""
    p1 = float(PHASES_BY_ID[1].env_kwargs["max_episode_s"])
    seen = []
    monkeypatch.setattr(trainer, "rollout_eval_trials", 1)
    monkeypatch.setattr(trainer, "phase_mix", {1: 0.8, 2: 0.2})
    monkeypatch.setattr(trainer, "_train_phase_id", 2)
    monkeypatch.setattr(trainer, "_eval_vs_rules", lambda s: seen.append(("rules", s)))
    monkeypatch.setattr(trainer, "_maybe_run_neural_snapshot_eval", lambda s: seen.append(("snapshots", s)))
    monkeypatch.setattr(trainer, "_eval_role_phases", lambda: None)
    trainer._run_rollout_evals(15.0)
    assert seen == [("rules", p1), ("snapshots", p1)]
    seen.clear()
    monkeypatch.setattr(trainer, "phase_mix", None)
    monkeypatch.setattr(trainer, "_train_phase_id", 1)
    trainer._run_rollout_evals(33.0)                        # a phase-1 run: its own value, unchanged
    assert seen == [("rules", 33.0), ("snapshots", 33.0)]


def test_a_phase2_only_run_without_a_mix_runs_the_role_evals(trainer, monkeypatch, caplog, controllers):
    controllers(NEURAL_ATTACKER)
    real = rpe.run_role_phase_eval
    monkeypatch.setattr(rpe, "run_role_phase_eval",
                        lambda phase, spec, fn, seeds=None, repeats_per_seed=None: real(phase, spec, fn, seeds=[11]))
    monkeypatch.setattr(trainer, "phase_mix", None)
    monkeypatch.setattr(trainer, "_train_phase_id", 2)
    trainer._role_eval_cache = {}
    with caplog.at_level(logging.INFO, logger="footballcoach"):
        trainer._eval_role_phases()
    assert "[eval phase2_1v2/attacker_vs_rules]" in caplog.text
    assert "[eval phase2_1v2/defender_vs_rules]" not in caplog.text


def test_a_phase1_only_run_runs_no_role_evals(trainer, monkeypatch, caplog):
    monkeypatch.setattr(trainer, "phase_mix", None)
    monkeypatch.setattr(trainer, "_train_phase_id", 1)
    with caplog.at_level(logging.INFO, logger="footballcoach"):
        trainer._eval_role_phases()
    assert "[eval phase2" not in caplog.text


def test_the_persistent_pool_gives_the_in_process_result(trainer):
    """Rules-only episodes are deterministic per seed, so the pool must
    reproduce the in-process run exactly (seeds dealt round-robin, merged
    back per worker); a neural spec returns every episode."""
    from footballcoach.ai.eval.eval_worker import close_eval_workers, spawn_eval_workers
    seeds = list(range(5000, 5007))
    # Spawned eval workers load the run config from disk (the session
    # fixtures patch only this process), so pin the run-only start settings
    # in the specs themselves: both sides then build the same scenario.
    pin = {"no_keeper_prob": 0.0, "attacker_start_min_m": 18.0, "attacker_start_max_m": 32.0,
           "random_initial_velocity": False}
    rules_spec = {**SPECS["rules_baseline"], "scenario_kwargs": pin}
    neural_spec = {**SPECS["attacker_vs_rules"], "scenario_kwargs": pin}
    local = rpe.run_role_phase_eval(PHASE2, rules_spec, None, seeds=seeds, repeats_per_seed=2)
    workers = spawn_eval_workers(2)
    try:
        pooled = rpe.run_role_phase_eval_on_workers(workers, PHASE2, rules_spec, seeds=seeds,
                                                    repeats_per_seed=2)
        d, e, v = trainer._cpu_state_dicts()
        for w in workers:
            w.set_weights(d, e, v)
        neural = rpe.run_role_phase_eval_on_workers(workers, PHASE2, neural_spec, seeds=seeds[:3],
                                                    repeats_per_seed=2)
    finally:
        close_eval_workers(workers)
    order = seeds[0::2] + seeds[1::2]                     # worker 0's seeds, then worker 1's
    by_seed = {s: local.outcomes[2 * i:2 * i + 2] for i, s in enumerate(seeds)}
    assert pooled.outcomes == [o for s in order for o in by_seed[s]]
    assert sorted(pooled.rewards) == sorted(local.rewards)
    assert len(neural.outcomes) == 6 and "attacker=neural" in neural.episode_tag

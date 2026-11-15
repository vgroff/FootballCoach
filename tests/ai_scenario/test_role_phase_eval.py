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


def test_eval_seeds_come_from_the_phase_config():
    cfg = load_ai_config()["curriculum"]["phase2"]
    seeds = rpe.eval_seeds(PHASE2)
    assert seeds[0] == cfg["eval_seed_offset"] and len(seeds) == cfg["eval_episodes"]


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


def test_trainer_runs_every_spec_and_caches_the_rules_baseline(trainer, monkeypatch, caplog):
    calls = []
    real = rpe.run_role_phase_eval

    def _counting(phase, spec, fn, seeds=None):
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
    assert calls == ["rules_baseline", "attacker_vs_rules", "defender_vs_rules", "attacker_vs_rules", "defender_vs_rules"]
    for name in SPECS:
        assert caplog.text.count(f"[eval phase2_1v2/{name}]") == 2


def test_no_phase_mix_means_no_role_evals(trainer, caplog):
    trainer.phase_mix = None
    with caplog.at_level(logging.INFO, logger="footballcoach"):
        trainer._eval_role_phases()
    assert "[eval phase2" not in caplog.text

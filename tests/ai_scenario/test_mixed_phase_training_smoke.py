"""End-to-end plumbing smoke test for mixed-phase training: one real
PPOTrainer.train() cycle through the batched multi-process path with a 1:1
phase-1/phase-2 mix. Spawned workers read ai_config.json from disk, so this
points FOOTBALLCOACH_AI_CONFIG_PATH at a temporary copy (phase 2 attacker and
defender neural, tiny rollout) taken from this process's live config dict, and
applies the same overrides to that dict with monkeypatch for the main process.

It must NOT clear the config cache: tests/ai_scenario/conftest.py mutates the
cached dict for the whole session (e.g. switching action masking off), and a
cache clear silently drops those mutations for every later test in this
pytest worker (it did: five unrelated tests failed in the full run).

A wiring check, not training: one ~1k-step rollout, one PPO epoch, no
checkpoint written.
"""
from __future__ import annotations

import json
import logging

import pytest
import torch

from footballcoach.ai.config import load_ai_config


_OVERRIDES = {
    ("curriculum", "phase2", "controllers"): {
        "attacker": {"neural": 1}, "defender": {"neural": 1}, "goalkeeper": {"rules": 1},
    },
    ("network", "ball_physics_encoder_checkpoint"): None,
    ("network", "player_physics_encoder_checkpoint"): None,
    **{("ppo", k): v for k, v in {
        "n_processes": 2, "envs_per_process": 4, "batched_rollout": True, "rollout_steps": 1200,
        "n_epochs": 1, "minibatch_size": 512, "rollout_eval_trials": 0, "worker_torch_threads": 1,
    }.items()},
}


@pytest.fixture
def mixed_config(tmp_path, monkeypatch):
    live = load_ai_config()
    for path, value in _OVERRIDES.items():
        node = live
        for key in path[:-1]:
            node = node[key]
        monkeypatch.setitem(node, path[-1], value)
    cfg_path = tmp_path / "ai_config.json"
    cfg_path.write_text(json.dumps(live))
    monkeypatch.setenv("FOOTBALLCOACH_AI_CONFIG_PATH", str(cfg_path))
    yield live


def test_one_mixed_cycle_collects_both_phases_and_logs_them(mixed_config, caplog):
    from footballcoach.ai.curriculum.mix import validate_training_mix
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

    assert load_ai_config()["ppo"]["n_processes"] == 2   # the temporary config is live here
    mix = {1: 1.0, 2: 1.0}
    validate_training_mix(mix)
    torch.manual_seed(0)
    trainer = PPOTrainer.from_config(device=torch.device("cpu"))
    trainer.set_frozen_heads(PHASES_BY_ID[1].frozen_heads)
    trainer.phase_mix = mix
    trainer._ppg_interleave_enabled = False
    trainer._eval_n_parallel_workers = 1
    trainer.checkpoint_dir = None
    with caplog.at_level(logging.INFO, logger="footballcoach"):
        trainer.train(None, total_steps=500, phase_id=1)
    text = caplog.text
    assert "phase mix phase1=4/8 env slots  phase2=4/8 env slots" in text
    assert "phase2_1v2[attacker=neural,defender=neural,goalkeeper=rules](" in text
    assert trainer._total_steps >= 500


_PPG_OVERRIDES = {
    **{("ppo", k): v for k, v in {
        "value_pretrain_n_processes": 2, "value_pretrain_batched_rollout": True,
        "value_pretrain_envs_per_process": 4, "value_pretrain_chunk_steps": 0,
    }.items()},
    **{("bc", k): v for k, v in {
        "ppg_epochs": 1, "ppg_episode_replay_enabled": True, "ppg_loss_diagnostics": False,
        "ppg_compare_to_original": False, "ppg_val_episode_fraction": 0.0,
    }.items()},
}


def test_ppg_refit_collects_both_phases_and_replays_per_phase(mixed_config, monkeypatch, tmp_path, caplog):
    """The PPG value refit over a phase mix: its worker pool runs both phases,
    logs the role-based phase's outcomes, and its second cycle replays seeds
    selected per phase."""
    import json as _json
    import os
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

    for path, value in _PPG_OVERRIDES.items():
        monkeypatch.setitem(mixed_config[path[0]], path[1], value)
    (tmp_path / "ai_config.json").write_text(_json.dumps(mixed_config))
    assert os.environ["FOOTBALLCOACH_AI_CONFIG_PATH"] == str(tmp_path / "ai_config.json")
    torch.manual_seed(0)
    trainer = PPOTrainer.from_config(device=torch.device("cpu"))
    trainer.set_frozen_heads(PHASES_BY_ID[1].frozen_heads)
    trainer.phase_mix = {1: 1.0, 2: 1.0}
    trainer.checkpoint_dir = None
    from footballcoach.ai.curriculum.mix import untag_seed
    queued = []
    real = trainer._ppg_update_episode_replay

    def _capture(seed_mse, pending):
        out = real(seed_mse, pending)
        queued.append((sorted({untag_seed(s)[1] for s, _ in seed_mse}), out[1]))
        return out
    trainer._ppg_update_episode_replay = _capture
    with caplog.at_level(logging.INFO, logger="footballcoach"):
        trainer.ppg_value_refit(None, n_steps=8000, phase_id=1, num_rollouts=2)
    text = caplog.text
    assert "[value pretrain rollout] phase mix phase1=4/8 env slots  phase2=4/8 env slots" in text
    assert "[value pretrain rollout] phase2_1v2[attacker=neural,defender=neural,goalkeeper=rules](" in text
    assert len(queued) == 1                      # selected after cycle 1, for cycle 2
    phases_seen, next_seeds = queued[0]
    assert phases_seen == [1, 2]                 # both phases' episodes were ranked
    assert sorted({untag_seed(s)[1] for s in next_seeds}) == [1, 2]   # and both got replay slots


def test_ppg_over_a_mix_refuses_the_single_process_path(mixed_config, monkeypatch):
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    monkeypatch.setitem(mixed_config["ppo"], "value_pretrain_n_processes", 1)
    trainer = PPOTrainer.from_config(device=torch.device("cpu"))
    trainer.phase_mix = {1: 1.0, 2: 1.0}
    with pytest.raises(NotImplementedError, match="phase mix"):
        trainer._collect_value_pretrain_rollout(None, 100, 1)


def test_one_mixed_cycle_with_attacker_kick_bc(mixed_config, monkeypatch, tmp_path, caplog):
    """Phase-2 attacker kick BC through the real multi-process training path:
    labels made by the attacker's rules expert in the workers, the source id
    stored per row, the per-row loss applied and logged."""
    import json as _json
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

    monkeypatch.setitem(mixed_config["curriculum"]["phase2"], "bc", {
        "attacker": {"kick": {"coeff_start": 0.5, "coeff_end": 0.5, "anneal_fraction": 1.0}}})
    (tmp_path / "ai_config.json").write_text(_json.dumps(mixed_config))
    torch.manual_seed(0)
    trainer = PPOTrainer.from_config(device=torch.device("cpu"))
    trainer.set_frozen_heads(PHASES_BY_ID[1].frozen_heads)
    trainer.phase_mix = {1: 1.0, 2: 1.0}
    trainer._ppg_interleave_enabled = False
    trainer._eval_n_parallel_workers = 1
    trainer.checkpoint_dir = None
    with caplog.at_level(logging.INFO, logger="footballcoach"):
        trainer.train(None, total_steps=500, phase_id=1)
    assert "bc[phase2_1v2/attacker]=kick_gate " in caplog.text and "/kick_power " in caplog.text
    assert "bc agree [phase2_1v2/attacker] expert kick-now=" in caplog.text
    assert "bc pull  main |g_bc|=" in caplog.text

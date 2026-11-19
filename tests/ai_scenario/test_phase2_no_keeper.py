"""Phase 2 without the goalkeeper (build_1v2_scenario's no_keeper_prob): the
role env runs the episode with the goalkeeper role left out, the seed keeps
everything else identical, and the eval specs pin the keeper in or out."""
from __future__ import annotations

import pytest

from footballcoach.ai.config import load_ai_config
from footballcoach.ai.curriculum.envs import build_role_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.eval import role_phase_eval as rpe
from footballcoach.ui.scenarios import build_1v2_scenario

PHASE2 = PHASES_BY_ID[2]
RULES = {r: {"rules": 1} for r in ("attacker", "defender", "goalkeeper")}


def _ids(match) -> set[str]:
    return {p.player_id for p in match.players}


def test_builder_default_always_has_the_keeper():
    assert all("keeper" in _ids(build_1v2_scenario(seed=s)) for s in range(50))


def test_builder_drops_the_keeper_at_about_the_configured_rate():
    dropped = sum("keeper" not in _ids(build_1v2_scenario(seed=s, no_keeper_prob=0.2)) for s in range(500))
    assert 70 <= dropped <= 130   # 100 expected; binomial sd ~9


def test_dropping_the_keeper_keeps_every_other_draw_of_the_seed():
    kw = dict(attribute_tiers={"attacker": ["premier_league"], "defender": ["generic"]}, stamina_range=[0.5, 1.0])
    for s in range(40):
        with_gk = build_1v2_scenario(seed=s, **kw)
        maybe = build_1v2_scenario(seed=s, no_keeper_prob=0.5, **kw)
        for p in maybe.players:
            q = with_gk.player_by_id(p.player_id)
            assert (p.position, p.stamina, p.attributes) == (q.position, q.stamina, q.attributes)


def _env_with(monkeypatch, prob: float):
    cfg = load_ai_config()["curriculum"]["phase2"]
    monkeypatch.setitem(cfg, "scenario_kwargs", {"no_keeper_prob": prob})
    return build_role_env(PHASE2, controller_weights=RULES)


def test_role_env_runs_an_episode_without_the_keeper(monkeypatch):
    env = _env_with(monkeypatch, 1.0)
    env.reset(seed=3)
    assert "keeper" not in _ids(env.match)
    assert env.episode_roles == {"attacker": "attacker", "defender": "defender"}
    assert set(env.episode_controllers) == {"attacker", "defender"}
    assert env.episode_tag == "phase2_1v2[attacker=rules,defender=rules]"
    done = False
    while not done:
        _, _, done, info = env.step()
    assert info.trial_outcome in env.outcome_keys
    assert set(env.episode_rewards_by_player(info.trial_outcome)) == {"attacker", "defender"}


def test_role_env_with_the_keeper_is_unchanged(monkeypatch):
    env = _env_with(monkeypatch, 0.0)
    env.reset(seed=3)
    assert env.episode_roles == dict(PHASE2.roles)
    assert env.episode_tag == "phase2_1v2[attacker=rules,defender=rules,goalkeeper=rules]"


def test_a_missing_role_that_is_not_optional_is_still_an_error(monkeypatch):
    env = _env_with(monkeypatch, 1.0)
    env.optional_roles = frozenset()
    with pytest.raises(ValueError, match="no player for roles"):
        env.reset(seed=3)


def test_eval_specs_pin_the_keeper_in_or_out(monkeypatch):
    cfg = load_ai_config()["curriculum"]["phase2"]
    monkeypatch.setitem(cfg, "scenario_kwargs", {"no_keeper_prob": 0.5})
    specs = {s["name"]: s for s in PHASE2.eval_specs}
    with_gk = rpe.run_role_phase_eval(PHASE2, specs["rules_baseline"], None, seeds=range(6))
    without = rpe.run_role_phase_eval(PHASE2, specs["rules_baseline_no_keeper"], None, seeds=range(6))
    assert with_gk.episode_tag.endswith("goalkeeper=rules]")
    assert without.episode_tag == "phase2_1v2[attacker=rules,defender=rules]"


def test_a_neural_keeper_trainee_survives_trials_without_a_keeper(monkeypatch):
    """Regression (2026-11-19 keeper pretrain crash): with the keeper as the
    episode's trainee, the end-of-episode observation was encoded against the
    NEXT trial's match, which (unseeded episodes draw the keeper per trial) may
    have no keeper -> KeyError. It now uses the episode's own final tick."""
    import random
    import torch
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    cfg = load_ai_config()["curriculum"]["phase2"]
    monkeypatch.setitem(cfg, "scenario_kwargs", {"no_keeper_prob": 0.5})
    trainer = PPOTrainer.from_config(device=torch.device("cpu"))
    weights = {"attacker": {"rules": 1}, "defender": {"rules": 1}, "goalkeeper": {"neural": 1}}
    env = build_role_env(PHASE2, controller_weights=weights)
    env.sample_action_fn = trainer._sample_action
    keeper_trainee = 0
    random.seed(0)
    for seed in range(30):
        # UNSEEDED resets, as the rollout workers' fresh episodes: each trial
        # then draws the keeper afresh, so the next trial (already built when
        # an episode ends) can lack him. Seeded ones rebuild the same trial.
        env.rng = random.Random(seed)
        env.reset(seed=None)
        keeper_trainee += env.trainee_player_id == "keeper"
        done = False
        while not done:
            obs, _, done, _ = env.step()
        assert obs is not None
    assert keeper_trainee > 0

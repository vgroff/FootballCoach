"""curriculum.phase2.scenario_kwargs reaches the phase-2 role env's scenario
build (e.g. per-episode attribute / stamina variety)."""
from __future__ import annotations

from footballcoach.ai.config import load_ai_config
from footballcoach.ai.curriculum.envs import build_role_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.entities.attributes import PlayerAttributes

RULES = {r: {"rules": 1} for r in ("attacker", "defender", "goalkeeper")}


def test_phase2_config_scenario_kwargs_reach_the_build(monkeypatch):
    cfg = load_ai_config()["curriculum"]["phase2"]
    monkeypatch.setitem(cfg, "scenario_kwargs", {
        "attribute_tiers": {"attacker": ["premier_league"], "defender": ["generic"], "keeper": ["generic"]},
        "stamina_range": [0.5, 0.9]})
    env = build_role_env(PHASES_BY_ID[2], controller_weights=RULES)
    env.reset(seed=7)
    players = {p.player_id: p for p in env.match.players}
    assert players["attacker"].attributes != PlayerAttributes.average(0.9)
    assert all(0.5 <= p.stamina <= 0.9 for p in players.values())


def test_without_it_the_matchup_is_fixed(monkeypatch):
    cfg = load_ai_config()["curriculum"]["phase2"]
    monkeypatch.delitem(cfg, "scenario_kwargs", raising=False)
    env = build_role_env(PHASES_BY_ID[2], controller_weights=RULES)
    env.reset(seed=7)
    assert {p.player_id: p for p in env.match.players}["attacker"].attributes == PlayerAttributes.average(0.9)

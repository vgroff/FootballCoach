"""build_1v2_scenario's optional per-trial attribute / stamina variety:
off by default (the fixed matchup), seeded, varies across seeds, and never
moves a position drawn for the same seed."""
from __future__ import annotations

import pytest

from footballcoach.entities.attributes import PlayerAttributes
from footballcoach.ui.scenarios import build_1v2_scenario

TIERS = {"attacker": ["premier_league", "boosted"], "defender": ["generic", "championship", "league_one"],
         "keeper": ["generic", "championship", "league_one"]}


def _players(m):
    return {p.player_id: p for p in m.players}


def test_default_is_the_fixed_matchup():
    p = _players(build_1v2_scenario(seed=3))
    assert p["attacker"].attributes == PlayerAttributes.average(0.9)
    assert p["defender"].attributes == PlayerAttributes.average(0.55)
    assert p["keeper"].attributes == PlayerAttributes.average(0.55)


def test_seeded_varied_and_positions_unchanged():
    fixed = _players(build_1v2_scenario(seed=11))
    a = _players(build_1v2_scenario(seed=11, attribute_tiers=TIERS, stamina_range=[0.5, 1.0]))
    b = _players(build_1v2_scenario(seed=11, attribute_tiers=TIERS, stamina_range=[0.5, 1.0]))
    c = _players(build_1v2_scenario(seed=12, attribute_tiers=TIERS, stamina_range=[0.5, 1.0]))
    for pid in ("attacker", "defender", "keeper"):
        assert a[pid].attributes == b[pid].attributes and a[pid].stamina == b[pid].stamina      # seeded
        assert a[pid].position == fixed[pid].position                                          # positions kept
        assert a[pid].attributes != PlayerAttributes.average(0.9 if pid == "attacker" else 0.55)
        assert 0.5 <= a[pid].stamina <= 1.0
    assert any(a[pid].attributes != c[pid].attributes for pid in a)                             # varies by seed


def test_a_role_left_out_keeps_its_fixed_skill():
    p = _players(build_1v2_scenario(seed=5, attribute_tiers={"defender": ["generic"]}))
    assert p["attacker"].attributes == PlayerAttributes.average(0.9)
    assert p["defender"].attributes != PlayerAttributes.average(0.55)

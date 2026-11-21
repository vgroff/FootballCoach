"""The Phase 2 UI scenario (ui/scenarios.py phase2_neural_attacker): a
separate entry from the rules-only 1v2, with all three roles (attacker,
defender, goalkeeper) driven by one picked checkpoint with task id 2, as the
all_neural eval plays it; per-role rules overrides, and a no-keeper option."""
from __future__ import annotations

import pytest

from footballcoach.ui import scenarios as s

DEF = next(d for d in s.SCENARIOS if d.key == "phase2_neural_attacker")
ALL_RULES = {"attacker_rules": True, "defender_rules": True, "keeper_rules": True}


def test_it_is_a_separate_entry_and_the_rules_1v2_is_unchanged():
    plain = next(d for d in s.SCENARIOS if d.key == "1v2")
    assert plain.build is s.build_1v2_scenario
    assert [p.name for p in plain.params] == [
        "attacker_skill", "defender_skill", "gk_skill", "attacker_start_min_m",
        "attacker_start_max_m", "move_fraction_min", "move_fraction_max"]
    assert DEF.build is not s.build_1v2_scenario
    assert {p.name for p in DEF.params} == {
        "checkpoint", "attacker_rules", "defender_rules", "keeper_rules",
        "no_keeper_prob", "player_variety", "decision_interval_ms"}


def test_rules_overrides_with_and_without_the_keeper():
    with_gk = DEF.build(0.3, **ALL_RULES, no_keeper_prob=0.0, seed=1)
    without = DEF.build(0.3, **ALL_RULES, no_keeper_prob=1.0, seed=1)
    assert {p.player_id for p in with_gk.players} == {"attacker", "defender", "keeper"}
    assert {p.player_id for p in without.players} == {"attacker", "defender"}
    assert with_gk._ai_fully_wired and without._ai_fully_wired


class _Trainer:
    def _sample_action(self, *a, **k):
        raise AssertionError("not called at build time")


@pytest.fixture
def picked(monkeypatch):
    label = DEF.params[0].default
    if label == "(none)":
        pytest.skip("no phase-2 checkpoints on this machine")
    monkeypatch.setattr(s, "_load_trainer", lambda path: (_Trainer(), None))
    return label


def test_a_picked_checkpoint_drives_every_role_with_task_id_2(picked):
    from footballcoach.rules_ai import NeuralPlayerAI
    m = DEF.build(0.3, checkpoint=picked, no_keeper_prob=0.0, seed=2)
    for pid in ("attacker", "defender", "keeper"):
        player = m.player_by_id(pid)
        assert isinstance(player.ai, NeuralPlayerAI) and player.ai.task_phase == 2, pid
        assert player.current_order is None, pid


def test_a_rules_override_keeps_only_that_role_on_its_rules_ai(picked):
    from footballcoach.rules_ai import NeuralPlayerAI
    m = DEF.build(0.3, checkpoint=picked, defender_rules=True, keeper_rules=True,
                  no_keeper_prob=0.0, seed=2)
    assert isinstance(m.player_by_id("attacker").ai, NeuralPlayerAI)
    assert not isinstance(m.player_by_id("defender").ai, NeuralPlayerAI)
    assert not isinstance(m.player_by_id("keeper").ai, NeuralPlayerAI)


def test_no_keeper_with_every_role_neural(picked):
    from footballcoach.rules_ai import NeuralPlayerAI
    m = DEF.build(0.3, checkpoint=picked, no_keeper_prob=1.0, seed=3)
    assert m.player_by_id_or_none("keeper") is None
    assert isinstance(m.player_by_id("defender").ai, NeuralPlayerAI)

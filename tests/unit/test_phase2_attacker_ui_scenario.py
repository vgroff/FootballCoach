"""The Phase 2 UI scenario (ui/scenarios.py phase2_neural_attacker): a
separate entry from the rules-only 1v2, with the attacker driven by a picked
checkpoint, task id 2, and a no-keeper option."""
from __future__ import annotations

from footballcoach.ui import scenarios as s

DEF = next(d for d in s.SCENARIOS if d.key == "phase2_neural_attacker")


def test_it_is_a_separate_entry_and_the_rules_1v2_is_unchanged():
    plain = next(d for d in s.SCENARIOS if d.key == "1v2")
    assert plain.build is s.build_1v2_scenario
    assert [p.name for p in plain.params] == [
        "attacker_skill", "defender_skill", "gk_skill", "attacker_start_min_m",
        "attacker_start_max_m", "move_fraction_min", "move_fraction_max"]
    assert DEF.build is not s.build_1v2_scenario
    assert {p.name for p in DEF.params} == {
        "attacker_checkpoint", "attacker_rules", "no_keeper_prob", "player_variety", "decision_interval_ms"}


def test_rules_override_with_and_without_the_keeper():
    with_gk = DEF.build(0.3, attacker_rules=True, no_keeper_prob=0.0, seed=1)
    without = DEF.build(0.3, attacker_rules=True, no_keeper_prob=1.0, seed=1)
    assert {p.player_id for p in with_gk.players} == {"attacker", "defender", "keeper"}
    assert {p.player_id for p in without.players} == {"attacker", "defender"}
    assert with_gk._ai_fully_wired and without._ai_fully_wired


def test_a_picked_checkpoint_drives_the_attacker_with_task_id_2(monkeypatch):
    from footballcoach.rules_ai import NeuralPlayerAI

    class _Trainer:
        def _sample_action(self, *a, **k):
            raise AssertionError("not called at build time")
    label = DEF.params[0].default
    monkeypatch.setattr(s, "_load_trainer", lambda path: (_Trainer(), None))
    m = DEF.build(0.3, attacker_checkpoint=label, seed=2)
    if label == "(none)":   # no phase-2 checkpoints on this machine: rules attacker
        assert not isinstance(m.player_by_id("attacker").ai, NeuralPlayerAI)
        return
    ai = m.player_by_id("attacker").ai
    assert isinstance(ai, NeuralPlayerAI) and ai.task_phase == 2
    assert m.player_by_id("attacker").current_order is None

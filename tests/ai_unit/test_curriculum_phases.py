"""PHASES_BY_ID: train.py looks phases up here, so a gap or a clash silently picks the wrong one."""
from __future__ import annotations

from footballcoach.ai.curriculum.phases import ALL_PHASES, PHASES_BY_ID, CurriculumPhase


def test_ids_are_unique_and_run_from_one_without_gaps():
    ids = [phase.phase_id for phase in ALL_PHASES]
    assert ids == list(range(1, len(ALL_PHASES) + 1))


def test_every_phase_is_reachable_by_its_own_id():
    assert len(PHASES_BY_ID) == len(ALL_PHASES)
    for phase in ALL_PHASES:
        assert PHASES_BY_ID[phase.phase_id] is phase


def test_every_module_level_phase_is_registered():
    import footballcoach.ai.curriculum.phases as phases
    declared = [v for k, v in vars(phases).items() if k.startswith("PHASE_") and isinstance(v, CurriculumPhase)]
    assert {id(p) for p in declared} == {id(p) for p in ALL_PHASES}


def test_phase_names_are_unique():
    names = [phase.name for phase in ALL_PHASES]
    assert len(set(names)) == len(names)


def test_frozen_heads_are_real_decision_heads():
    from dataclasses import fields

    from footballcoach.ai.action.schema import DecisionHeadsRaw
    real = {f.name for f in fields(DecisionHeadsRaw)}
    known = real | {name.removesuffix("_logit") for name in real}
    for phase in ALL_PHASES:
        assert set(phase.frozen_heads) <= known, phase.name

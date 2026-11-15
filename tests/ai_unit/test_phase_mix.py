"""curriculum/mix.py: phase-mix parsing and validation, env-slot assignment and
phase-tagged episode seeds."""
from __future__ import annotations

import pytest

import footballcoach.ai.curriculum.mix as mix
from footballcoach.ai.curriculum.phases import PHASES_BY_ID


def test_parse_accepts_strings_and_dicts():
    assert mix.parse_phase_mix("1:3,2:1") == {1: 3.0, 2: 1.0}
    assert mix.parse_phase_mix("2") == {2: 1.0}
    assert mix.parse_phase_mix({"2": 1, "1": 0.5}) == {1: 0.5, 2: 1.0}
    assert mix.parse_phase_mix("1:1,2:0") == {1: 1.0}


@pytest.mark.parametrize("spec,match", [("1:1,1:2", "twice"), ("1:-1", "negative"), ("1:0", "no phase"), ("", "no phase")])
def test_parse_rejects_bad_specs(spec, match):
    with pytest.raises(ValueError, match=match):
        mix.parse_phase_mix(spec)


def test_validate_rejects_unknown_and_unimplemented_phases():
    with pytest.raises(ValueError, match="unknown phase"):
        mix.validate_training_mix({1: 1, 99: 1})
    with pytest.raises(ValueError, match="not implemented"):
        mix.validate_training_mix({1: 1, 3: 1})


def test_validate_rejects_a_role_phase_that_cannot_roll_neural(monkeypatch):
    import copy
    from footballcoach.ai.config import load_ai_config
    cfg = copy.deepcopy(load_ai_config())
    cfg["curriculum"]["phase2"]["controllers"] = {r: {"rules": 1} for r in ("attacker", "defender", "goalkeeper")}
    monkeypatch.setattr("footballcoach.ai.config.load_ai_config", lambda: cfg)
    with pytest.raises(ValueError, match="no role can roll 'neural'"):
        mix.validate_training_mix({1: 1, 2: 1})
    cfg["curriculum"]["phase2"]["controllers"]["defender"] = {"rules": 1, "neural": 1}
    assert [p.phase_id for p in mix.validate_training_mix({1: 1, 2: 1})] == [1, 2]


def test_validate_rejects_differing_frozen_heads(monkeypatch):
    import copy
    from footballcoach.ai.config import load_ai_config
    cfg = copy.deepcopy(load_ai_config())
    cfg["curriculum"]["phase2"]["controllers"]["attacker"] = {"neural": 1}
    monkeypatch.setattr("footballcoach.ai.config.load_ai_config", lambda: cfg)
    monkeypatch.setattr(PHASES_BY_ID[2], "frozen_heads", ["shoot_logit"])
    with pytest.raises(ValueError, match="share frozen_heads"):
        mix.validate_training_mix({1: 1, 2: 1})


def test_phase_2_shares_phase_1_frozen_heads():
    assert sorted(PHASES_BY_ID[2].frozen_heads) == sorted(PHASES_BY_ID[1].frozen_heads)


@pytest.mark.parametrize("weights,n", [({1: 3, 2: 1}, 544), ({1: 1, 2: 1}, 7), ({1: 1}, 5), ({1: 2, 2: 1, 3: 1}, 10)])
def test_assign_env_phases_matches_the_shares_and_interleaves(weights, n):
    slots = mix.assign_env_phases(weights, n)
    assert len(slots) == n
    total = sum(weights.values())
    for pid, w in weights.items():
        assert abs(slots.count(pid) - n * w / total) < 1.0
    # Every contiguous block (one worker's envs) is close to the overall share.
    if len(weights) > 1 and n >= 32:
        block = slots[:32]
        for pid, w in weights.items():
            assert abs(block.count(pid) - 32 * w / total) <= 1.0
    assert mix.assign_env_phases(weights, n) == slots


def test_seed_tags_roundtrip_and_phase_one_is_the_plain_seed():
    for seed in (0, 12345, mix.PHASE_SEED_STRIDE - 1):
        assert mix.tag_seed(seed, 1) == seed
        for pid in (1, 2, 5):
            assert mix.untag_seed(mix.tag_seed(seed, pid)) == (seed, pid)
    with pytest.raises(ValueError):
        mix.tag_seed(mix.PHASE_SEED_STRIDE, 1)

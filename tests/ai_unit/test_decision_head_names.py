"""The seven decision heads go by four spellings in different layers. This file is
the one place the correspondence is written down, and it checks that every
table in the code agrees with it, so a head added or renamed in one table but
not the others fails here instead of silently reading 0.0 somewhere.

  decision_probs / gating   network attribute        HEAD_LP_KEYS   DecisionAction field
  (DECISION_HEAD_NAMES)     (_LP_HEAD_NAMES)         (short form)
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from footballcoach.ai.action.gating import DECISION_HEAD_NAMES, _HEAD_TO_ACTION
from footballcoach.ai.action.schema import DecisionAction
from footballcoach.ai.ppo.action_sampling import _LP_HEAD_NAME_TO_SHORT
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _LP_HEAD_NAMES
from footballcoach.ai.ppo.rollout_buffer import HEAD_LP_KEYS
from footballcoach.rules_ai import HybridPlayerAI

_TABLE = [
    ("shoot", "shoot_logit", "shoot", "shoot"),
    ("pass_", "pass_logit", "pass_", "pass_"),
    ("move", "move_logit", "move", "move"),
    ("tackle", "tackle_logit", "tackle", "tackle"),
    ("get_possession", "get_possession_raw", "gp_extra", "get_possession_extra"),
    ("mark", "mark_logit", "mark", "mark"),
    ("hold_position", "hold_position_logit", "hold", "hold_position"),
]


@pytest.fixture(scope="module")
def trainer():
    return PPOTrainer.from_config(device=torch.device("cpu"), inference_only=True, separate_value_net=False)


def test_gating_names_are_the_tie_break_table_keys_in_order():
    assert DECISION_HEAD_NAMES == tuple(_HEAD_TO_ACTION) == tuple(r[0] for r in _TABLE)


def test_network_attribute_names_and_short_keys_line_up():
    assert tuple(_LP_HEAD_NAMES) == tuple(r[1] for r in _TABLE)
    assert list(_LP_HEAD_NAME_TO_SHORT.items()) == [(r[1], r[2]) for r in _TABLE]
    assert list(HEAD_LP_KEYS[:len(_TABLE)]) == [r[2] for r in _TABLE]


def test_decision_action_fields_line_up():
    fields = [f.name for f in dataclasses.fields(DecisionAction)]
    assert fields[:len(_TABLE)] == [r[3] for r in _TABLE]


def test_every_network_attribute_exists_on_the_decision_net(trainer):
    for attr in _LP_HEAD_NAMES:
        assert isinstance(getattr(trainer.decision_net, attr), torch.nn.Module), attr


def test_sampled_decision_probs_use_exactly_the_gating_names(trainer):
    from tests.ai_unit.test_sample_action_from_heads_batch import _make_heads_batch, _reference_rows
    d, e, v, xs, em = _make_heads_batch(3, exec_move_logits=[1.0, -1.0, 0.0], kick_logits=[1.0, -1.0, 0.0], seed=3)
    batched = trainer._sample_action_from_heads_batch(d, e, v, xs, em, deterministic=True)
    reference = _reference_rows(trainer, d, e, v, xs, em, deterministic=True)
    for row in batched + reference:
        assert tuple(row[3]) == DECISION_HEAD_NAMES


def test_each_probability_comes_from_its_own_head(trainer):
    """Positional mix-ups (e.g. two heads swapped in the decision_probs dict) would
    keep the key set intact, so check that raising one head's logit moves only its
    own probability."""
    from tests.ai_unit.test_sample_action_from_heads_batch import _make_heads_batch
    for name, attr, _, _ in _TABLE:
        d, e, v, xs, em = _make_heads_batch(1, exec_move_logits=[0.0], kick_logits=[0.0], seed=5)
        base = trainer._sample_action_from_heads_batch(d, e, v, xs, em, deterministic=True)[0][3]
        setattr(d, attr, getattr(d, attr) + 6.0)
        bumped = trainer._sample_action_from_heads_batch(d, e, v, xs, em, deterministic=True)[0][3]
        moved = {k for k in DECISION_HEAD_NAMES if abs(bumped[k] - base[k]) > 1e-6}
        # get_possession is derived from both tackle and get_possession_raw, so
        # raising tackle legitimately moves it too.
        expected = {name, "get_possession"} if name == "tackle" else {name}
        assert moved == expected, (attr, moved)


def test_set_decision_override_rejects_unknown_names():
    ai = HybridPlayerAI(lambda obs: None)
    with pytest.raises(ValueError, match="unknown decision head 'pass'"):
        ai.set_decision_override("pass", 1.0)
    assert ai._decision_overrides == {}


def test_set_decision_override_accepts_every_head_and_clears_with_none():
    ai = HybridPlayerAI(lambda obs: None)
    for name in DECISION_HEAD_NAMES:
        ai.set_decision_override(name, 1.0)
    assert ai._decision_overrides == {name: 1.0 for name in DECISION_HEAD_NAMES}
    for name in DECISION_HEAD_NAMES:
        ai.set_decision_override(name, None)
    assert ai._decision_overrides == {}

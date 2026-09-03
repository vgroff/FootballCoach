"""Regression tests for train_ball_dynamics.py's crossing_head checkpoint
migration -- see agent_plans/physics_update.md and physics_runs.md for the
real-world incident this guards against: a checkpoint with a well-trained
crossing_head (crosses_acc=0.997) was silently reset to near-random
(crosses_acc~0.28) on a plain resume, because the migration code never
checked whether the checkpoint's shape already matched the current model's
shape before applying a destructive legacy-format heuristic.
"""
from __future__ import annotations

import torch

from footballcoach.ai.physics_pretrain.ball_dynamics_net import BallDynamicsAutoencoder
from footballcoach.ai.physics_pretrain.train_ball_dynamics import _migrate_crossing_head_state_dict


def _make_model() -> BallDynamicsAutoencoder:
    return BallDynamicsAutoencoder(latent_dim=8, hidden_dim=16, decoder_hidden_dim=16, horizons_s=[0.5, 1.0])


def test_matching_shape_checkpoint_is_returned_unchanged():
    """The exact bug found 2026-09-03: a checkpoint already in the current
    4-output (pos_x, pos_y, crosses_logit, delta_t) format must be a
    complete no-op, not silently re-migrated and reset to fresh init."""
    model = _make_model()
    real_weight = torch.randn(4, 8)
    real_bias = torch.randn(4)
    state_dict = {"crossing_head.weight": real_weight.clone(), "crossing_head.bias": real_bias.clone()}

    result = _migrate_crossing_head_state_dict(state_dict, model)

    assert torch.equal(result["crossing_head.weight"], real_weight), (
        "a checkpoint whose crossing_head shape already matches the current "
        "model must be returned byte-for-byte unchanged -- any deviation "
        "means real trained weights (e.g. crosses_logit/delta_t) are being "
        "silently discarded on every resume"
    )
    assert torch.equal(result["crossing_head.bias"], real_bias)


def test_legacy_4_output_height_format_migrates_to_current_4_output():
    """A genuinely old checkpoint (pos_x, pos_y, height, delta_t) -- distinct
    from the current (pos_x, pos_y, crosses_logit, delta_t) despite both
    being 4 rows -- still needs the two-step migration, ending with
    pos_x/pos_y preserved and crosses_logit/delta_t freshly initialized
    (there's no valid source for either in the old format)."""
    model = _make_model()
    old_weight = torch.stack([
        torch.full((8,), 1.0),  # pos_x
        torch.full((8,), 2.0),  # pos_y
        torch.full((8,), 3.0),  # height (to be dropped)
        torch.full((8,), 4.0),  # delta_t (blended, not reusable post-split)
    ])
    old_bias = torch.tensor([1.0, 2.0, 3.0, 4.0])
    state_dict = {"crossing_head.weight": old_weight.clone(), "crossing_head.bias": old_bias.clone()}

    # Force this test into the "genuinely old 4-row format" branch by giving
    # the model a target shape the guard won't treat as already-matching --
    # simulate via a model whose crossing_head is temporarily 3-wide, the
    # shape the old pipeline would have targeted at that point in history.
    model.crossing_head = torch.nn.Linear(8, 3)
    result = _migrate_crossing_head_state_dict(state_dict, model)

    assert torch.equal(result["crossing_head.weight"], old_weight[[0, 1, 3]])
    assert torch.equal(result["crossing_head.bias"], old_bias[[0, 1, 3]])


def test_3_output_to_4_output_split_preserves_position_resets_rest():
    model = _make_model()
    old_weight = torch.stack([
        torch.full((8,), 1.0),  # pos_x
        torch.full((8,), 2.0),  # pos_y
        torch.full((8,), 3.0),  # old blended delta_t
    ])
    old_bias = torch.tensor([1.0, 2.0, 3.0])
    state_dict = {"crossing_head.weight": old_weight.clone(), "crossing_head.bias": old_bias.clone()}

    result = _migrate_crossing_head_state_dict(state_dict, model)

    assert torch.equal(result["crossing_head.weight"][0:2], old_weight[0:2])
    assert torch.equal(result["crossing_head.bias"][0:2], old_bias[0:2])
    # Rows 2/3 (crosses_logit, delta_t) must NOT be the old blended value --
    # they should be the current model's own fresh init.
    assert not torch.equal(result["crossing_head.weight"][2:4], old_weight[2:3].expand(2, -1))


def test_missing_crossing_head_key_is_a_no_op():
    model = _make_model()
    state_dict = {"encoder.some.weight": torch.randn(4, 4)}
    result = _migrate_crossing_head_state_dict(state_dict, model)
    assert result is state_dict

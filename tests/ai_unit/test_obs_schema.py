"""Unit tests for obs/schema.py - feature vector shapes and dtypes.

These tests don't touch torch or build a Match; they just verify the
dataclass structure and dimension constants match across the whole
codebase (no silent shape mismatch between schema and network input layers).
"""
import math

import numpy as np
import pytest

from footballcoach.ai.obs.schema import (
    BALL_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    MAX_OTHER_PLAYERS,
    PLAYER_FEATURE_DIM,
    BallFeatures,
    GlobalFeatures,
    ObservationBatch,
    PlayerFeatures,
)


# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------

def test_player_feature_dim_is_39():
    """PLAYER_FEATURE_DIM must stay 39 (change this test IFF you change the schema).

    Fields: rel_dx, rel_dy, distance_m, ball_rel_dx, ball_rel_dy,
    ball_distance_m, ball_vel_rel_x, ball_vel_rel_y, ball_closing_speed,
    velocity_x, velocity_y, speed_mps, stamina,
    top_speed, acceleration, kick_power, kick_precision, dribbling,
    ball_control, tackling, stamina_attr, is_own_team, is_self, has_possession,
    is_inactive_tackled, is_controlling_ball, is_goalkeeper, attacking_direction,
    exists, is_immobile, pos_x, pos_y, heading_sin, heading_cos, desired_dir_x,
    desired_dir_y, desired_speed_standstill, desired_speed_jog, desired_speed_sprint.
    """
    assert PLAYER_FEATURE_DIM == 39


def test_ball_feature_dim_is_12():
    assert BALL_FEATURE_DIM == 12


def test_global_feature_dim_is_31():
    """11 original fields + MAX_TASK_IDS (20) task_id_N one-hot fields."""
    assert GLOBAL_FEATURE_DIM == 31


def test_max_other_players_is_21():
    assert MAX_OTHER_PLAYERS == 21


# ---------------------------------------------------------------------------
# to_array() output
# ---------------------------------------------------------------------------

def test_player_features_to_array_shape():
    feat = PlayerFeatures()
    arr = feat.to_array()
    assert arr.shape == (PLAYER_FEATURE_DIM,)


def test_player_features_to_array_dtype():
    arr = PlayerFeatures().to_array()
    assert arr.dtype == np.float32


def test_ball_features_to_array_shape():
    arr = BallFeatures().to_array()
    assert arr.shape == (BALL_FEATURE_DIM,)


def test_ball_features_to_array_dtype():
    arr = BallFeatures().to_array()
    assert arr.dtype == np.float32


def test_global_features_to_array_shape():
    arr = GlobalFeatures().to_array()
    assert arr.shape == (GLOBAL_FEATURE_DIM,)


def test_global_features_to_array_dtype():
    arr = GlobalFeatures().to_array()
    assert arr.dtype == np.float32


def test_global_features_task_id_default_all_zero():
    arr = GlobalFeatures().to_array()
    # Last MAX_TASK_IDS entries are the task_id one-hot block.
    assert np.all(arr[-20:] == 0.0)


def test_global_features_task_id_one_hot():
    feat = GlobalFeatures(task_id_2=1.0)
    arr = feat.to_array()
    task_block = arr[-20:]
    assert task_block[2] == 1.0
    assert task_block.sum() == 1.0


def test_player_features_defaults_no_nan():
    arr = PlayerFeatures().to_array()
    assert not np.any(np.isnan(arr))


def test_ball_features_defaults_no_nan():
    arr = BallFeatures().to_array()
    assert not np.any(np.isnan(arr))


def test_global_features_defaults_no_nan():
    arr = GlobalFeatures().to_array()
    assert not np.any(np.isnan(arr))


# ---------------------------------------------------------------------------
# Field ordering - ensure the to_array() order matches expectation
# (if field order ever changes, these catch it before the network silently
# trains on the wrong feature in the wrong position)
# ---------------------------------------------------------------------------

def test_player_features_first_three_are_position():
    """First three fields are rel_dx, rel_dy, distance_m."""
    feat = PlayerFeatures(rel_dx=1.0, rel_dy=2.0, distance_m=3.0)
    arr = feat.to_array()
    assert arr[0] == pytest.approx(1.0)
    assert arr[1] == pytest.approx(2.0)
    assert arr[2] == pytest.approx(3.0)


def test_player_features_pos_at_fixed_indices():
    """pos_x/pos_y stay at indices 30/31 -- ai/physics_pretrain/live_encoder_features.py
    hardcodes these as PF_POS_X/PF_POS_Y, so newer fields must be appended AFTER
    pos_y, never inserted earlier, to avoid silently breaking those offsets."""
    import dataclasses
    names = [f.name for f in dataclasses.fields(PlayerFeatures)]
    assert names[30] == "pos_x"
    assert names[31] == "pos_y"
    feat = PlayerFeatures(pos_x=0.5, pos_y=-0.3)
    arr = feat.to_array()
    assert arr[30] == pytest.approx(0.5)
    assert arr[31] == pytest.approx(-0.3)


def test_player_features_heading_and_intent_are_last_seven():
    """heading_sin/cos + desired_dir_x/y + speed-mode one-hot are appended
    after pos_y, in this exact order (see the field-ordering constraint in
    ai/knowledge.md's 'Heading and previous-decision movement intent' note)."""
    import dataclasses
    names = [f.name for f in dataclasses.fields(PlayerFeatures)]
    assert names[-7:] == [
        "heading_sin", "heading_cos",
        "desired_dir_x", "desired_dir_y",
        "desired_speed_standstill", "desired_speed_jog", "desired_speed_sprint",
    ]
    feat = PlayerFeatures(
        heading_sin=0.1, heading_cos=0.2, desired_dir_x=0.3, desired_dir_y=0.4,
        desired_speed_standstill=0.5, desired_speed_jog=0.6, desired_speed_sprint=0.7,
    )
    arr = feat.to_array()
    assert arr[-7:] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])


def test_ball_features_is_possessed_at_fixed_indices():
    """is_possessed/is_loose stay at indices 9/10 -- last_touch_team_direction
    was appended AFTER them (index 11), not inserted earlier."""
    feat = BallFeatures(is_possessed=1.0, is_loose=0.0)
    arr = feat.to_array()
    assert arr[9] == pytest.approx(1.0)
    assert arr[10] == pytest.approx(0.0)


def test_ball_features_last_touch_team_direction_is_last():
    import dataclasses
    names = [f.name for f in dataclasses.fields(BallFeatures)]
    assert names[-1] == "last_touch_team_direction"
    feat = BallFeatures(last_touch_team_direction=-1.0)
    arr = feat.to_array()
    assert arr[-1] == pytest.approx(-1.0)


def test_global_features_attack_defence_before_task_id_block():
    """attack_defence_smoothed is the last field before the task_id one-hot block."""
    feat = GlobalFeatures(attack_defence_smoothed=0.7)
    arr = feat.to_array()
    assert arr[-21] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# ObservationBatch.to_torch_dict()
# ---------------------------------------------------------------------------

def test_observation_batch_to_torch_dict_keys():
    obs = ObservationBatch(
        self_feat=np.zeros(PLAYER_FEATURE_DIM, dtype=np.float32),
        other_feat=np.zeros((MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM), dtype=np.float32),
        exists_mask=np.zeros(MAX_OTHER_PLAYERS, dtype=np.float32),
        ball_feat=np.zeros(BALL_FEATURE_DIM, dtype=np.float32),
        global_feat=np.zeros(GLOBAL_FEATURE_DIM, dtype=np.float32),
    )
    d = obs.to_torch_dict()
    assert set(d.keys()) == {
        "self_feat", "other_feat", "exists_mask", "ball_feat", "global_feat",
        "self_ai_type", "other_ai_type",
    }


def test_observation_batch_to_torch_dict_shapes():
    obs = ObservationBatch(
        self_feat=np.zeros(PLAYER_FEATURE_DIM, dtype=np.float32),
        other_feat=np.zeros((MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM), dtype=np.float32),
        exists_mask=np.zeros(MAX_OTHER_PLAYERS, dtype=np.float32),
        ball_feat=np.zeros(BALL_FEATURE_DIM, dtype=np.float32),
        global_feat=np.zeros(GLOBAL_FEATURE_DIM, dtype=np.float32),
    )
    d = obs.to_torch_dict()
    import torch
    assert d["self_feat"].shape == (PLAYER_FEATURE_DIM,)
    assert d["other_feat"].shape == (MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM)
    assert d["exists_mask"].shape == (MAX_OTHER_PLAYERS,)
    assert d["ball_feat"].shape == (BALL_FEATURE_DIM,)
    assert d["global_feat"].shape == (GLOBAL_FEATURE_DIM,)

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

def test_player_feature_dim_is_27():
    """PLAYER_FEATURE_DIM must stay 27 (change this test IFF you change the schema).

    Fields: rel_dx, rel_dy, distance_m, velocity_x, velocity_y, speed_mps,
    heading_sin, heading_cos, stamina, top_speed, acceleration, kick_power,
    kick_precision, dribbling, ball_control, tackling, stamina_attr,
    is_own_team, is_self, has_possession, is_inactive_tackled,
    is_controlling_ball, is_goalkeeper, attacking_direction, exists,
    pos_x, pos_y.
    """
    assert PLAYER_FEATURE_DIM == 27


def test_ball_feature_dim_is_12():
    assert BALL_FEATURE_DIM == 12


def test_global_feature_dim_is_11():
    assert GLOBAL_FEATURE_DIM == 11


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


def test_player_features_pos_are_last_two():
    """pos_x and pos_y are the last two fields (indices 25, 26)."""
    import dataclasses
    names = [f.name for f in dataclasses.fields(PlayerFeatures)]
    assert names[-2] == "pos_x"
    assert names[-1] == "pos_y"
    feat = PlayerFeatures(pos_x=0.5, pos_y=-0.3)
    arr = feat.to_array()
    assert arr[-2] == pytest.approx(0.5)
    assert arr[-1] == pytest.approx(-0.3)


def test_ball_features_is_possessed_second_to_last():
    feat = BallFeatures(is_possessed=1.0, is_loose=0.0)
    arr = feat.to_array()
    assert arr[-2] == pytest.approx(1.0)
    assert arr[-1] == pytest.approx(0.0)


def test_global_features_attack_defence_is_last():
    feat = GlobalFeatures(attack_defence_smoothed=0.7)
    arr = feat.to_array()
    assert arr[-1] == pytest.approx(0.7)


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
    assert set(d.keys()) == {"self_feat", "other_feat", "exists_mask", "ball_feat", "global_feat"}


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

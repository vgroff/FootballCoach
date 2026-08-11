"""Unit tests for obs/encoder.py - Match state -> ObservationBatch.

These are the most important correctness tests in the AI test suite:
if the observation encoding is wrong the network trains on the wrong
information and no amount of PPO hyperparameter tuning will fix it.

Coverage:
- Array shapes match schema constants
- Self slot: rel_dx=0, rel_dy=0, distance=0, is_self=1, exists=1
- Padded slots: all zeros, exists=0
- Random slot shuffling: permutation invariance
- Content invariance: same features regardless of which slot a player lands in
- Position normalization: boundary -> ≈±1; origin relative to observer
- Velocity normalization: feature ~ [0, 1] range at top speed
- Flags: team, possession, state, goalkeeper, attacking direction
- Ball features: correct relative position, possessed flag
- Global features: score_diff from team perspective, log1p time normalization
- No NaN or Inf anywhere
"""
import math
import random

import numpy as np
import pytest

from footballcoach.ai.obs.encoder import encode_observation, MAX_OTHER_PLAYERS
from footballcoach.ai.obs.schema import (
    BALL_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    PLAYER_FEATURE_DIM,
    PlayerFeatures,
)
from footballcoach.entities.player import PlayerState, Team
from footballcoach.mathutils import Vector3


# ---------------------------------------------------------------------------
# Helper: field index within PlayerFeatures.to_array()
# ---------------------------------------------------------------------------

from dataclasses import fields as _fields

_PF_FIELDS = [f.name for f in _fields(PlayerFeatures)]


def _pf_idx(name: str) -> int:
    return _PF_FIELDS.index(name)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestShapes:
    def test_self_feat_shape(self, solo_match):
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert obs.self_feat.shape == (PLAYER_FEATURE_DIM,)

    def test_other_feat_shape(self, solo_match):
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert obs.other_feat.shape == (MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM)

    def test_exists_mask_shape(self, solo_match):
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert obs.exists_mask.shape == (MAX_OTHER_PLAYERS,)

    def test_ball_feat_shape(self, solo_match):
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert obs.ball_feat.shape == (BALL_FEATURE_DIM,)

    def test_global_feat_shape(self, solo_match):
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert obs.global_feat.shape == (GLOBAL_FEATURE_DIM,)

    def test_dtypes_all_float32(self, solo_match):
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert obs.self_feat.dtype == np.float32
        assert obs.other_feat.dtype == np.float32
        assert obs.exists_mask.dtype == np.float32
        assert obs.ball_feat.dtype == np.float32
        assert obs.global_feat.dtype == np.float32


# ---------------------------------------------------------------------------
# Self slot correctness
# ---------------------------------------------------------------------------

class TestSelfSlot:
    def _obs(self, match, pid="p1"):
        return encode_observation(match, pid, time_remaining_s=60.0, rng=random.Random(0))

    def test_rel_dx_is_zero(self, solo_match):
        obs = self._obs(solo_match)
        assert obs.self_feat[_pf_idx("rel_dx")] == pytest.approx(0.0, abs=1e-6)

    def test_rel_dy_is_zero(self, solo_match):
        obs = self._obs(solo_match)
        assert obs.self_feat[_pf_idx("rel_dy")] == pytest.approx(0.0, abs=1e-6)

    def test_distance_is_zero(self, solo_match):
        obs = self._obs(solo_match)
        assert obs.self_feat[_pf_idx("distance_m")] == pytest.approx(0.0, abs=1e-6)

    def test_is_self_flag_one(self, solo_match):
        obs = self._obs(solo_match)
        assert obs.self_feat[_pf_idx("is_self")] == pytest.approx(1.0, abs=1e-6)

    def test_exists_flag_one(self, solo_match):
        obs = self._obs(solo_match)
        assert obs.self_feat[_pf_idx("exists")] == pytest.approx(1.0, abs=1e-6)

    def test_is_own_team_is_one_for_self(self, solo_match):
        obs = self._obs(solo_match)
        assert obs.self_feat[_pf_idx("is_own_team")] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Padded slots
# ---------------------------------------------------------------------------

class TestPaddedSlots:
    def test_solo_match_all_other_slots_zero(self, solo_match):
        """Solo match has no other players -> all 21 slots should be zero-filled."""
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert np.all(obs.other_feat == 0.0)

    def test_solo_match_exists_mask_all_zero(self, solo_match):
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert np.all(obs.exists_mask == 0.0)

    def test_duel_exactly_one_slot_filled(self, duel_match):
        obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert obs.exists_mask.sum() == pytest.approx(1.0, abs=1e-6)

    def test_duel_unfilled_slots_all_zero(self, duel_match):
        obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        for i in range(MAX_OTHER_PLAYERS):
            if obs.exists_mask[i] < 0.5:
                assert np.all(obs.other_feat[i] == 0.0), (
                    f"Padded slot {i} has non-zero values: {obs.other_feat[i]}"
                )

    def test_padded_slot_exists_exactly_zero(self, duel_match):
        """exists field in each padded slot must be exactly 0, not just small."""
        obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        exists_idx = _pf_idx("exists")
        for i in range(MAX_OTHER_PLAYERS):
            if obs.exists_mask[i] < 0.5:
                assert obs.other_feat[i, exists_idx] == pytest.approx(0.0, abs=1e-9)

    def test_real_slot_exists_one(self, duel_match):
        obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        exists_idx = _pf_idx("exists")
        filled_slots = [i for i in range(MAX_OTHER_PLAYERS) if obs.exists_mask[i] > 0.5]
        assert len(filled_slots) == 1
        assert obs.other_feat[filled_slots[0], exists_idx] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Random slot shuffling (permutation invariance)
# ---------------------------------------------------------------------------

class TestSlotShuffling:
    def _filled_slot(self, obs) -> int:
        """Return the index of the one real slot."""
        for i in range(MAX_OTHER_PLAYERS):
            if obs.exists_mask[i] > 0.5:
                return i
        raise AssertionError("No filled slot found")

    def test_different_seeds_put_player_in_different_slots(self, duel_match):
        """Same match, different RNG seeds -> player lands in different slots
        often enough (we run 20 seeds and check we see at least 2 distinct slots)."""
        seen_slots = set()
        for seed in range(20):
            obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                     rng=random.Random(seed))
            seen_slots.add(self._filled_slot(obs))
        assert len(seen_slots) > 1, (
            "Slot shuffling not working: same slot used for all seeds"
        )

    def test_content_invariant_across_slots(self, duel_match):
        """The feature *content* of the real player must be the same regardless
        of which slot they land in (only the slot index changes)."""
        feature_sets = []
        for seed in range(10):
            obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                     rng=random.Random(seed))
            slot = self._filled_slot(obs)
            feature_sets.append(obs.other_feat[slot].copy())

        # All feature vectors should be identical (same player, same match state)
        reference = feature_sets[0]
        for i, feat in enumerate(feature_sets[1:], 1):
            np.testing.assert_array_almost_equal(
                feat, reference, decimal=5,
                err_msg=f"Feature content changed between seed 0 and seed {i}"
            )

    def test_slots_never_exceed_range(self, duel_match):
        for seed in range(20):
            obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                     rng=random.Random(seed))
            slot = self._filled_slot(obs)
            assert 0 <= slot < MAX_OTHER_PLAYERS


# ---------------------------------------------------------------------------
# Position and normalization
# ---------------------------------------------------------------------------

class TestPositionNormalization:
    def test_other_player_relative_position_correct(self, duel_match):
        """p2 is at x=10, p1 at x=0. rel_dx should be 10 / half_diag (isotropic norm)."""
        obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        half_len = duel_match.pitch.length_m / 2.0
        half_wid = duel_match.pitch.width_m / 2.0
        half_diag = math.hypot(half_len, half_wid)
        expected_rel_dx = 10.0 / half_diag

        slot = next(i for i in range(MAX_OTHER_PLAYERS) if obs.exists_mask[i] > 0.5)
        assert obs.other_feat[slot, _pf_idx("rel_dx")] == pytest.approx(expected_rel_dx, rel=1e-4)

    def test_player_at_pitch_boundary_gives_rel_dx_matching_half_diag(self, standard_pitch):
        """A player standing exactly at the right goal line should have rel_dx == half_length / half_diag."""
        import random as _random
        from footballcoach.engine.match import Match
        from footballcoach.entities.player import Player, Team
        from footballcoach.entities.attributes import PlayerAttributes
        from footballcoach.entities.ball import Ball

        attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        observer = Player.create("obs", Team.LEFT, attrs, position=Vector3(0, 0, 0))
        boundary = Player.create("bnd", Team.RIGHT, attrs,
                                  position=Vector3(standard_pitch.half_length, 0, 0))
        ball = Ball.at_rest(Vector3(0, 0, 0))
        match = Match(pitch=standard_pitch, players=[observer, boundary], ball=ball,
                      rng_reduction=1.0, rng=_random.Random(0))

        obs = encode_observation(match, "obs", time_remaining_s=60.0, rng=_random.Random(0))
        half_len = standard_pitch.length_m / 2.0
        half_wid = standard_pitch.width_m / 2.0
        half_diag = math.hypot(half_len, half_wid)
        expected_rel_dx = half_len / half_diag
        slot = next(i for i in range(MAX_OTHER_PLAYERS) if obs.exists_mask[i] > 0.5)
        assert obs.other_feat[slot, _pf_idx("rel_dx")] == pytest.approx(expected_rel_dx, rel=1e-4)

    def test_distance_metric_is_non_negative(self, duel_match):
        obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        slot = next(i for i in range(MAX_OTHER_PLAYERS) if obs.exists_mask[i] > 0.5)
        assert obs.other_feat[slot, _pf_idx("distance_m")] >= 0.0


# ---------------------------------------------------------------------------
# Flag correctness
# ---------------------------------------------------------------------------

class TestFlags:
    def _get_other(self, obs):
        slot = next(i for i in range(MAX_OTHER_PLAYERS) if obs.exists_mask[i] > 0.5)
        return obs.other_feat[slot]

    def test_opponent_is_not_own_team(self, duel_match):
        obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        feat = self._get_other(obs)
        assert feat[_pf_idx("is_own_team")] == pytest.approx(0.0, abs=1e-6)

    def test_teammate_is_own_team(self, standard_pitch):
        """Two players on the same team: other player should be is_own_team=1."""
        import random as _r
        from footballcoach.engine.match import Match
        from footballcoach.entities.player import Player, Team
        from footballcoach.entities.attributes import PlayerAttributes
        from footballcoach.entities.ball import Ball
        attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        p1 = Player.create("p1", Team.LEFT, attrs, position=Vector3(0, 0, 0))
        p2 = Player.create("p2", Team.LEFT, attrs, position=Vector3(5, 0, 0))  # same team
        ball = Ball.at_rest(Vector3(0, 0, 0))
        match = Match(pitch=standard_pitch, players=[p1, p2], ball=ball,
                      rng_reduction=1.0, rng=_r.Random(0))
        obs = encode_observation(match, "p1", time_remaining_s=60.0, rng=_r.Random(0))
        slot = next(i for i in range(MAX_OTHER_PLAYERS) if obs.exists_mask[i] > 0.5)
        assert obs.other_feat[slot, _pf_idx("is_own_team")] == pytest.approx(1.0, abs=1e-6)

    def test_ball_carrier_has_possession_flag(self, duel_match):
        """p1 has the ball -> has_possession=1 in self_feat."""
        obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert obs.self_feat[_pf_idx("has_possession")] == pytest.approx(1.0, abs=1e-6)

    def test_non_carrier_no_possession_flag(self, duel_match):
        """p2 does not have the ball -> has_possession=0 in their slot."""
        obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        feat = self._get_other(obs)
        assert feat[_pf_idx("has_possession")] == pytest.approx(0.0, abs=1e-6)

    def test_left_team_attacking_direction_positive(self, duel_match):
        """Team.LEFT attacks +x -> attacking_direction=+1."""
        obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert obs.self_feat[_pf_idx("attacking_direction")] == pytest.approx(1.0, abs=1e-6)

    def test_right_team_attacking_direction_negative(self, duel_match):
        """Team.RIGHT attacks -x -> other player attacking_direction=-1."""
        obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        feat = self._get_other(obs)
        assert feat[_pf_idx("attacking_direction")] == pytest.approx(-1.0, abs=1e-6)

    def test_goalkeeper_flag(self, gk_match):
        """GK player should have is_goalkeeper=1 in their features (as seen by
        the attacker)."""
        obs = encode_observation(gk_match, "att", time_remaining_s=60.0,
                                 rng=random.Random(0))
        slot = next(i for i in range(MAX_OTHER_PLAYERS) if obs.exists_mask[i] > 0.5)
        assert obs.other_feat[slot, _pf_idx("is_goalkeeper")] == pytest.approx(1.0, abs=1e-6)

    def test_non_gk_no_goalkeeper_flag(self, duel_match):
        obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        feat = self._get_other(obs)
        assert feat[_pf_idx("is_goalkeeper")] == pytest.approx(0.0, abs=1e-6)

    def test_inactive_tackled_state_flag(self, solo_match):
        """Set player state to INACTIVE_TACKLED and check the flag."""
        solo_match.players[0].state = PlayerState.INACTIVE_TACKLED
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert obs.self_feat[_pf_idx("is_inactive_tackled")] == pytest.approx(1.0, abs=1e-6)
        assert obs.self_feat[_pf_idx("is_controlling_ball")] == pytest.approx(0.0, abs=1e-6)

    def test_controlling_ball_state_flag(self, solo_match):
        solo_match.players[0].state = PlayerState.CONTROLLING_BALL
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert obs.self_feat[_pf_idx("is_controlling_ball")] == pytest.approx(1.0, abs=1e-6)
        assert obs.self_feat[_pf_idx("is_inactive_tackled")] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Ball features
# ---------------------------------------------------------------------------

class TestBallFeatures:
    def test_ball_at_observer_has_zero_dist(self, standard_pitch):
        import random as _r
        from footballcoach.engine.match import Match
        from footballcoach.entities.player import Player, Team
        from footballcoach.entities.attributes import PlayerAttributes
        from footballcoach.entities.ball import Ball
        attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        p = Player.create("p1", Team.LEFT, attrs, position=Vector3(5, 3, 0))
        ball = Ball.at_rest(Vector3(5, 3, 0))  # ball exactly on player
        match = Match(pitch=standard_pitch, players=[p], ball=ball,
                      rng_reduction=1.0, rng=_r.Random(0))
        obs = encode_observation(match, "p1", time_remaining_s=60.0, rng=_r.Random(0))
        assert obs.ball_feat[2] == pytest.approx(0.0, abs=1e-5)  # distance_m

    def test_ball_possessed_flag(self, duel_match):
        obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        # ball.possessed_by = "p1" in duel_match
        assert obs.ball_feat[-2] == pytest.approx(1.0, abs=1e-6)  # is_possessed
        assert obs.ball_feat[-1] == pytest.approx(0.0, abs=1e-6)  # is_loose

    def test_ball_loose_flag(self, solo_match):
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        # solo_match ball is loose (not possessed)
        assert obs.ball_feat[-2] == pytest.approx(0.0, abs=1e-6)  # is_possessed
        assert obs.ball_feat[-1] == pytest.approx(1.0, abs=1e-6)  # is_loose

    def test_is_possessed_plus_is_loose_equals_one(self, duel_match):
        obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert (obs.ball_feat[-2] + obs.ball_feat[-1]) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Global features
# ---------------------------------------------------------------------------

class TestGlobalFeatures:
    def test_score_diff_zero_at_start(self, solo_match):
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert obs.global_feat[0] == pytest.approx(0.0, abs=1e-6)

    def test_score_diff_team_perspective_left_winning(self, duel_match):
        """Left team scores one: LEFT observer should see +1, RIGHT should see -1."""
        duel_match.scoreboard.left_goals = 1
        obs_left = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                      rng=random.Random(0))
        obs_right = encode_observation(duel_match, "p2", time_remaining_s=60.0,
                                       rng=random.Random(0))
        assert obs_left.global_feat[0] == pytest.approx(1.0, abs=1e-6)
        assert obs_right.global_feat[0] == pytest.approx(-1.0, abs=1e-6)

    def test_time_remaining_zero_gives_zero_norm(self, solo_match):
        obs = encode_observation(solo_match, "p1", time_remaining_s=0.0,
                                 rng=random.Random(0))
        assert obs.global_feat[1] == pytest.approx(0.0, abs=1e-6)

    def test_time_remaining_max_gives_one_norm(self, solo_match):
        from footballcoach.ai.config import load_ai_config
        max_t = load_ai_config()["observation"]["time_remaining_norm_max_s"]
        obs = encode_observation(solo_match, "p1", time_remaining_s=max_t,
                                 rng=random.Random(0))
        assert obs.global_feat[1] == pytest.approx(1.0, rel=1e-4)

    def test_time_normalization_monotonic(self, solo_match):
        """Shorter time remaining -> smaller normalized value."""
        obs_long = encode_observation(solo_match, "p1", time_remaining_s=3600.0,
                                      rng=random.Random(0))
        obs_short = encode_observation(solo_match, "p1", time_remaining_s=10.0,
                                       rng=random.Random(0))
        obs_urgent = encode_observation(solo_match, "p1", time_remaining_s=5.0,
                                        rng=random.Random(0))
        assert obs_long.global_feat[1] > obs_short.global_feat[1]
        assert obs_short.global_feat[1] > obs_urgent.global_feat[1]

    def test_urgent_endgame_distinguishable_from_normal(self, solo_match):
        """1s left vs 120s left must be clearly distinguishable (>0.05 difference)
        despite log1p normalization over a 7200s range."""
        obs_120s = encode_observation(solo_match, "p1", time_remaining_s=120.0,
                                      rng=random.Random(0))
        obs_1s   = encode_observation(solo_match, "p1", time_remaining_s=1.0,
                                      rng=random.Random(0))
        diff = obs_120s.global_feat[1] - obs_1s.global_feat[1]
        assert diff > 0.05, (
            f"Urgent vs normal time not distinguishable: diff={diff:.4f}"
        )

    def test_attack_defence_smoothed_passed_through(self, solo_match):
        """attack_defence_smoothed (last field) should match what we passed in."""
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 attack_defence_smoothed=0.8,
                                 rng=random.Random(0))
        assert obs.global_feat[-21] == pytest.approx(0.8, abs=1e-6)


# ---------------------------------------------------------------------------
# pos_x / pos_y negation under geometric flip augmentation
# ---------------------------------------------------------------------------
#
# flip_x is DELIBERATELY EXCLUDED from _FLIP_VARIANTS (see obs/augment.py
# module docstring "Canonical AI frame") — it's now a fixed, permanent
# transform applied by ai/obs/canonical.py rather than a random training-time
# augmentation. So pos_x negation is tested against canonical.py directly
# below; only flip_y remains a real _FLIP_VARIANTS entry to test here.

class TestFlipAugmentationPosXY:
    def _build_obs_dict(self, duel_match):
        import torch
        obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        return {
            "self_feat":   torch.from_numpy(obs.self_feat).unsqueeze(0),
            "other_feat":  torch.from_numpy(obs.other_feat).unsqueeze(0),
            "exists_mask": torch.from_numpy(obs.exists_mask).unsqueeze(0),
            "ball_feat":   torch.from_numpy(obs.ball_feat).unsqueeze(0),
            "global_feat": torch.from_numpy(obs.global_feat).unsqueeze(0),
        }, obs

    def test_pos_y_negated_under_flip_y_pos_x_unchanged(self, duel_match):
        from footballcoach.ai.obs.augment import augment_obs_bc, _FLIP_VARIANTS
        from footballcoach.ai.ppo.bc import BC_LABEL_DIM
        import torch

        obs_dict, _ = self._build_obs_dict(duel_match)
        bc_labels = torch.zeros((1, BC_LABEL_DIM))

        aug_obs, _aug_labels = augment_obs_bc(
            obs_dict, bc_labels, n_slot_shuffles=1, rng=random.Random(0)
        )

        pos_x_idx = _pf_idx("pos_x")
        pos_y_idx = _pf_idx("pos_y")

        original_self = obs_dict["self_feat"][0]
        # _FLIP_VARIANTS = [identity, flip_y] (flip_x removed — see above),
        # one slot-shuffle each -> rows 0,1 of the augmented batch.
        assert _FLIP_VARIANTS[0] == (False, False)
        assert _FLIP_VARIANTS[1] == (False, True)
        flip_y_self = aug_obs["self_feat"][1]

        assert flip_y_self[pos_y_idx] == pytest.approx(-original_self[pos_y_idx].item(), abs=1e-6)
        assert flip_y_self[pos_x_idx] == pytest.approx(original_self[pos_x_idx].item(), abs=1e-6)

    def test_pos_x_negated_under_canonical_frame_for_right_team(self, duel_match):
        """flip_x is now the CANONICAL AI FRAME transform (ai/obs/canonical.py),
        applied per-observer-team rather than randomly — verify it still
        negates pos_x (and leaves pos_y untouched) for a Team.RIGHT observer.
        """
        from footballcoach.ai.obs.canonical import canonicalize_obs
        import torch

        obs_dict, obs = self._build_obs_dict(duel_match)
        pos_x_idx = _pf_idx("pos_x")
        pos_y_idx = _pf_idx("pos_y")

        original_self = obs_dict["self_feat"][0]
        sf_c, _, _, x_sign = canonicalize_obs(
            obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["ball_feat"],
            x_sign=torch.tensor([-1.0]),  # force as if observer were Team.RIGHT
        )
        canon_self = sf_c[0]
        assert canon_self[pos_x_idx] == pytest.approx(-original_self[pos_x_idx].item(), abs=1e-6)
        assert canon_self[pos_y_idx] == pytest.approx(original_self[pos_y_idx].item(), abs=1e-6)

    def test_pitch_dimensions_in_global(self, solo_match):
        pitch = solo_match.pitch
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        # Indices 2,3 are pitch_length_norm, pitch_width_norm (normalised by standard dims)
        assert obs.global_feat[2] == pytest.approx(pitch.length_m / 105.0, rel=1e-4)
        assert obs.global_feat[3] == pytest.approx(pitch.width_m / 68.0, rel=1e-4)


# ---------------------------------------------------------------------------
# No NaN / Inf anywhere
# ---------------------------------------------------------------------------

class TestNoNaN:
    def test_solo_match_no_nan(self, solo_match):
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert not np.any(np.isnan(obs.self_feat))
        assert not np.any(np.isnan(obs.other_feat))
        assert not np.any(np.isnan(obs.ball_feat))
        assert not np.any(np.isnan(obs.global_feat))

    def test_duel_match_no_nan(self, duel_match):
        obs = encode_observation(duel_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        for arr in [obs.self_feat, obs.other_feat, obs.ball_feat, obs.global_feat]:
            assert not np.any(np.isnan(arr))
            assert not np.any(np.isinf(arr))


# ---------------------------------------------------------------------------
# Task-id one-hot (W3 scaffolding)
# ---------------------------------------------------------------------------

class TestTaskId:
    def test_phase_none_is_all_zero(self, solo_match):
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0))
        assert np.all(obs.global_feat[-20:] == 0.0)

    def test_phase_1_sets_index_0(self, solo_match):
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0), phase=1)
        task_block = obs.global_feat[-20:]
        assert task_block[0] == 1.0
        assert task_block.sum() == 1.0

    def test_phase_2_sets_index_1(self, solo_match):
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0), phase=2)
        task_block = obs.global_feat[-20:]
        assert task_block[1] == 1.0
        assert task_block.sum() == 1.0

    def test_out_of_range_phase_is_all_zero(self, solo_match):
        obs = encode_observation(solo_match, "p1", time_remaining_s=60.0,
                                 rng=random.Random(0), phase=999)
        assert np.all(obs.global_feat[-20:] == 0.0)

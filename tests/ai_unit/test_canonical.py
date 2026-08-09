"""Tests for the canonical AI-frame wrapper (ai/obs/canonical.py).

Covers: sign extraction, mirror invariants (idempotent double-mirror,
sign correctness per team), unbatched + batched shapes, mirror_x on
numpy and torch, round-trip (canonicalize -> decanonicalize == identity),
and integration with real encode_observation() output for both teams.
"""
import numpy as np
import pytest
import torch

from footballcoach.ai.obs.augment import BALL_FLIP_X_IDX, PLAYER_FLIP_X_IDX
from footballcoach.ai.obs.canonical import (
    X_SIGN_FIELD_IDX,
    canonicalize_obs,
    mirror_x,
    team_x_sign,
    x_sign_of,
)
from footballcoach.ai.obs.encoder import encode_observation
from footballcoach.ai.obs.schema import BallFeatures, PlayerFeatures
from footballcoach.entities.player import Team


def _make_self_feat(attacking_direction: float) -> torch.Tensor:
    arr = np.zeros(len(PlayerFeatures.__dataclass_fields__), dtype=np.float32)
    arr[X_SIGN_FIELD_IDX] = attacking_direction
    return torch.from_numpy(arr)


class TestTeamXSign:
    def test_left_is_positive(self):
        assert team_x_sign(Team.LEFT) == 1.0

    def test_right_is_negative(self):
        assert team_x_sign(Team.RIGHT) == -1.0


class TestXSignOf:
    def test_unbatched_left(self):
        sf = _make_self_feat(1.0)
        assert x_sign_of(sf).item() == 1.0

    def test_unbatched_right(self):
        sf = _make_self_feat(-1.0)
        assert x_sign_of(sf).item() == -1.0

    def test_batched(self):
        sf = torch.stack([_make_self_feat(1.0), _make_self_feat(-1.0), _make_self_feat(1.0)])
        signs = x_sign_of(sf)
        assert signs.shape == (3,)
        assert torch.allclose(signs, torch.tensor([1.0, -1.0, 1.0]))

    def test_matches_encoder_field_index(self):
        # X_SIGN_FIELD_IDX must point at attacking_direction, not drift if
        # schema fields are reordered.
        from dataclasses import fields
        names = [f.name for f in fields(PlayerFeatures)]
        assert names[X_SIGN_FIELD_IDX] == "attacking_direction"


class TestMirrorColumnsViaCanonicalizeObs:
    def _dummy_batch(self, x_sign_val: float, n_slots: int = 3):
        # attacking_direction IS one of PLAYER_FLIP_X_IDX, so it must be set
        # LAST (after the marker-value loop) or the marker overwrites it.
        pd = len(PlayerFeatures.__dataclass_fields__)
        bd = len(BallFeatures.__dataclass_fields__)
        sf = torch.zeros(pd)
        for i in PLAYER_FLIP_X_IDX:
            sf[i] = 3.0  # arbitrary nonzero marker value
        sf[X_SIGN_FIELD_IDX] = x_sign_val
        of = torch.zeros(n_slots, pd)
        for s in range(n_slots):
            for i in PLAYER_FLIP_X_IDX:
                of[s, i] = 5.0 + s
        bf = torch.zeros(bd)
        for i in BALL_FLIP_X_IDX:
            bf[i] = 7.0
        return sf, of, bf

    def test_left_team_unchanged(self):
        sf, of, bf = self._dummy_batch(x_sign_val=1.0)
        sf_c, of_c, bf_c, x_sign = canonicalize_obs(sf, of, bf)
        assert x_sign == 1.0
        for i in PLAYER_FLIP_X_IDX:
            if i == X_SIGN_FIELD_IDX:
                continue  # attacking_direction itself carries the sign, not the 3.0 marker
            assert sf_c[i].item() == pytest.approx(3.0)
        for i in BALL_FLIP_X_IDX:
            assert bf_c[i].item() == pytest.approx(7.0)

    def test_right_team_negated_on_flip_x_fields_only(self):
        sf, of, bf = self._dummy_batch(x_sign_val=-1.0)
        sf_c, of_c, bf_c, x_sign = canonicalize_obs(sf, of, bf)
        assert x_sign == -1.0
        for i in PLAYER_FLIP_X_IDX:
            if i == X_SIGN_FIELD_IDX:
                continue
            assert sf_c[i].item() == pytest.approx(-3.0)
        for s in range(of.shape[0]):
            for i in PLAYER_FLIP_X_IDX:
                if i == X_SIGN_FIELD_IDX:
                    continue
                assert of_c[s, i].item() == pytest.approx(-(5.0 + s))
        for i in BALL_FLIP_X_IDX:
            assert bf_c[i].item() == pytest.approx(-7.0)

    def test_non_flip_fields_untouched(self):
        sf, of, bf = self._dummy_batch(x_sign_val=-1.0)
        # mark a non-flip-x field (rel_dy) with a sentinel and confirm it survives
        from dataclasses import fields
        rel_dy_idx = [f.name for f in fields(PlayerFeatures)].index("rel_dy")
        sf[rel_dy_idx] = 42.0
        sf_c, _, _, _ = canonicalize_obs(sf, of, bf)
        assert sf_c[rel_dy_idx].item() == pytest.approx(42.0)

    def test_does_not_mutate_input_in_place(self):
        sf, of, bf = self._dummy_batch(x_sign_val=-1.0)
        sf_orig = sf.clone()
        canonicalize_obs(sf, of, bf)
        assert torch.equal(sf, sf_orig)

    def test_batched_mixed_teams(self):
        pd = len(PlayerFeatures.__dataclass_fields__)
        bd = len(BallFeatures.__dataclass_fields__)
        sf0 = torch.zeros(pd)
        sf1 = torch.zeros(pd)
        for i in PLAYER_FLIP_X_IDX:
            sf0[i] = 10.0
            sf1[i] = 10.0
        sf0[X_SIGN_FIELD_IDX] = 1.0
        sf1[X_SIGN_FIELD_IDX] = -1.0
        sf = torch.stack([sf0, sf1])
        of = torch.zeros(2, 1, pd)
        bf = torch.zeros(2, bd)
        sf_c, of_c, bf_c, x_sign = canonicalize_obs(sf, of, bf)
        assert torch.allclose(x_sign, torch.tensor([1.0, -1.0]))
        for i in PLAYER_FLIP_X_IDX:
            if i == X_SIGN_FIELD_IDX:
                continue
            assert sf_c[0, i].item() == pytest.approx(10.0)
            assert sf_c[1, i].item() == pytest.approx(-10.0)

    def test_explicit_x_sign_overrides_field(self):
        sf, of, bf = self._dummy_batch(x_sign_val=1.0)  # field says LEFT
        sf_c, _, _, x_sign = canonicalize_obs(sf, of, bf, x_sign=-1.0)  # force RIGHT
        assert x_sign == -1.0
        for i in PLAYER_FLIP_X_IDX:
            if i == X_SIGN_FIELD_IDX:
                continue
            assert sf_c[i].item() == pytest.approx(-3.0)


class TestMirrorX:
    def test_numpy_2d_vector(self):
        v = np.array([1.0, 2.0], dtype=np.float32)
        out = mirror_x(v, -1.0)
        assert out[0] == pytest.approx(-1.0)
        assert out[1] == pytest.approx(2.0)

    def test_numpy_3d_vector(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = mirror_x(v, -1.0)
        np.testing.assert_allclose(out, [-1.0, 2.0, 3.0])

    def test_torch_vector(self):
        v = torch.tensor([1.0, 2.0])
        out = mirror_x(v, -1.0)
        assert torch.allclose(out, torch.tensor([-1.0, 2.0]))

    def test_left_team_identity(self):
        v = np.array([1.0, 2.0], dtype=np.float32)
        out = mirror_x(v, 1.0)
        np.testing.assert_allclose(out, v)

    def test_none_passthrough(self):
        assert mirror_x(None, -1.0) is None

    def test_does_not_mutate_input(self):
        v = np.array([1.0, 2.0], dtype=np.float32)
        v_orig = v.copy()
        mirror_x(v, -1.0)
        np.testing.assert_allclose(v, v_orig)

    def test_batched_numpy(self):
        v = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        out = mirror_x(v, np.array([1.0, -1.0]))
        np.testing.assert_allclose(out, [[1.0, 2.0], [-3.0, 4.0]])

    def test_is_involution_numpy(self):
        # mirror(mirror(v)) == v for any single sign application
        v = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        once = mirror_x(v, -1.0)
        twice = mirror_x(once, -1.0)
        np.testing.assert_allclose(twice, v)

    def test_is_involution_torch(self):
        v = torch.tensor([1.0, -2.0, 3.0])
        once = mirror_x(v, -1.0)
        twice = mirror_x(once, -1.0)
        assert torch.allclose(twice, v)


class TestCanonicalizeIsInvolution:
    def test_double_mirror_recovers_original(self):
        pd = len(PlayerFeatures.__dataclass_fields__)
        bd = len(BallFeatures.__dataclass_fields__)
        sf = torch.randn(pd)
        sf[X_SIGN_FIELD_IDX] = -1.0
        of = torch.randn(4, pd)
        bf = torch.randn(bd)
        sf1, of1, bf1, xs1 = canonicalize_obs(sf, of, bf)
        # Applying the same x_sign again (a reflection is its own inverse)
        # must reconstruct the original raw values exactly.
        from footballcoach.ai.obs.canonical import _mirror_columns
        sf2 = _mirror_columns(sf1, PLAYER_FLIP_X_IDX, xs1)
        of2 = _mirror_columns(of1, PLAYER_FLIP_X_IDX, xs1)
        bf2 = _mirror_columns(bf1, BALL_FLIP_X_IDX, xs1)
        assert torch.allclose(sf2, sf)
        assert torch.allclose(of2, of)
        assert torch.allclose(bf2, bf)


@pytest.fixture
def simple_match():
    import random as _random
    from footballcoach.engine.match import Match
    from footballcoach.entities.attributes import PlayerAttributes
    from footballcoach.entities.ball import Ball
    from footballcoach.entities.pitch import Pitch
    from footballcoach.entities.player import Player
    from footballcoach.mathutils import Vector3

    pitch = Pitch.standard()
    attrs = PlayerAttributes(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    left = Player.create("L1", Team.LEFT, attrs, position=Vector3(-10.0, 5.0, 0.0))
    right = Player.create("R1", Team.RIGHT, attrs, position=Vector3(10.0, -5.0, 0.0))
    ball = Ball.at_rest(Vector3(0.0, 0.0, 0.0))
    match = Match(pitch=pitch, players=[left, right], ball=ball,
                  rng_reduction=1.0, rng=_random.Random(0))
    return match


class TestIntegrationWithEncoder:
    """Confirm canonicalization of REAL encode_observation() output produces
    the geometrically-mirrored relationship between a LEFT and RIGHT
    observer looking at the same (mirrored) scene."""

    def test_right_observer_self_feat_gets_mirrored(self, simple_match):
        import random
        obs = encode_observation(simple_match, "R1", time_remaining_s=60.0, rng=random.Random(0))
        sf = torch.from_numpy(obs.self_feat)
        assert x_sign_of(sf).item() == -1.0
        sf_c, _, _, x_sign = canonicalize_obs(sf, torch.from_numpy(obs.other_feat), torch.from_numpy(obs.ball_feat))
        assert x_sign == -1.0
        from dataclasses import fields
        pos_x_idx = [f.name for f in fields(PlayerFeatures)].index("pos_x")
        # world pos_x for R1 is +10/52.5 (positive); canonical frame must flip it negative
        assert obs.self_feat[pos_x_idx] > 0
        assert sf_c[pos_x_idx].item() < 0

    def test_left_observer_self_feat_unchanged(self, simple_match):
        import random
        obs = encode_observation(simple_match, "L1", time_remaining_s=60.0, rng=random.Random(0))
        sf = torch.from_numpy(obs.self_feat)
        sf_c, _, _, x_sign = canonicalize_obs(sf, torch.from_numpy(obs.other_feat), torch.from_numpy(obs.ball_feat))
        assert x_sign == 1.0
        assert torch.allclose(sf_c, sf)

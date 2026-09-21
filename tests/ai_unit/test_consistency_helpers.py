"""Unit tests for ai/ppo/consistency.py -- the pure helpers behind PPOTrainer.consistency_refit."""
from __future__ import annotations

import torch

from footballcoach.ai.obs.augment import augment_batch
from footballcoach.ai.obs.schema import BALL_FEATURE_DIM, GLOBAL_FEATURE_DIM, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM
from footballcoach.ai.ppo import consistency as C


def _obs(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "obs/self_feat": torch.randn(n, PLAYER_FEATURE_DIM, generator=g),
        "obs/other_feat": torch.randn(n, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM, generator=g),
        "obs/ball_feat": torch.randn(n, BALL_FEATURE_DIM, generator=g),
        "obs/global_feat": torch.randn(n, GLOBAL_FEATURE_DIM, generator=g),
        "obs/exists_mask": torch.ones(n, MAX_OTHER_PLAYERS),
    }


def _batch_for_augment(obs):
    n = len(obs["obs/self_feat"])
    return {**obs, "log_probs": torch.zeros(n), "values": torch.zeros(n), "rewards": torch.zeros(n),
            "advantages": torch.zeros(n), "returns": torch.zeros(n), "dones": torch.zeros(n),
            "sample_weights": torch.ones(n)}


def test_flip_y_obs_is_an_involution():
    obs = _obs()
    twice = C.flip_y_obs(C.flip_y_obs(obs))
    for k in ("obs/self_feat", "obs/other_feat", "obs/ball_feat"):
        assert torch.equal(twice[k], obs[k])


def test_flip_y_obs_matches_augment_batch_flip_y_block():
    obs = _obs()
    aug = augment_batch(_batch_for_augment(obs), 1, __import__("random").Random(0))
    n = len(obs["obs/self_feat"])
    flipped = C.flip_y_obs(obs)
    for k in ("obs/self_feat", "obs/other_feat", "obs/ball_feat"):
        assert torch.equal(aug[k][n:], flipped[k])
        assert torch.equal(aug[k][:n], obs[k])


def test_flip_y_obs_only_touches_masked_rows_and_leaves_non_geometric_keys():
    obs = _obs()
    mask = torch.zeros(len(obs["obs/self_feat"]), dtype=torch.bool)
    mask[::2] = True
    out = C.flip_y_obs(obs, mask)
    assert torch.equal(out["obs/self_feat"][~mask], obs["obs/self_feat"][~mask])
    assert not torch.equal(out["obs/self_feat"][mask], obs["obs/self_feat"][mask])
    assert out["obs/global_feat"] is obs["obs/global_feat"]
    assert out["obs/exists_mask"] is obs["obs/exists_mask"]


def test_orientation_pair_primary_has_observer_at_nonnegative_y_and_mirror_at_nonpositive_y():
    obs = _obs(64)
    primary, mirror = C.orientation_pair(obs)
    assert bool((primary["obs/self_feat"][:, C.POS_Y_IDX] >= 0).all())
    assert bool((mirror["obs/self_feat"][:, C.POS_Y_IDX] <= 0).all())
    for k in ("obs/self_feat", "obs/other_feat", "obs/ball_feat"):
        assert torch.equal(C.flip_y_obs(primary)[k], mirror[k])


def test_orientation_pair_maps_a_state_and_its_mirror_to_the_same_primary():
    obs = _obs(32)
    p1, _ = C.orientation_pair(obs)
    p2, _ = C.orientation_pair(C.flip_y_obs(obs))
    keep = obs["obs/self_feat"][:, C.POS_Y_IDX] != 0
    for k in ("obs/self_feat", "obs/other_feat", "obs/ball_feat"):
        assert torch.equal(p1[k][keep], p2[k][keep])


def _anchor(n=8):
    g = torch.Generator().manual_seed(1)
    a = {k: torch.randn(n, 1, generator=g) * 30 for k in C.BERNOULLI_KEYS}
    a["move_direction"] = torch.randn(n, 2, generator=g)
    a["kick_direction"] = torch.randn(n, 3, generator=g)
    a["kick_power"] = torch.randn(n, 1, generator=g)
    a["pass_target_logits"] = torch.randn(n, MAX_OTHER_PLAYERS, generator=g)
    return a


def test_mirror_anchor_y_negates_only_y_components_and_is_an_involution():
    a = _anchor()
    m = C.mirror_anchor_y(a)
    assert torch.equal(m["move_direction"][:, 0], a["move_direction"][:, 0])
    assert torch.equal(m["move_direction"][:, 1], -a["move_direction"][:, 1])
    assert torch.equal(m["kick_direction"][:, [0, 2]], a["kick_direction"][:, [0, 2]])
    assert torch.equal(m["kick_direction"][:, 1], -a["kick_direction"][:, 1])
    for k in (*C.BERNOULLI_KEYS, "kick_power", "pass_target_logits"):
        assert torch.equal(m[k], a[k])
    mm = C.mirror_anchor_y(m)
    for k in a:
        assert torch.equal(mm[k], a[k])


def test_mirror_anchor_y_treats_kick_spin_as_a_pseudovector():
    a = {"kick_spin": torch.tensor([[1.0, 2.0, 3.0]])}
    assert torch.equal(C.mirror_anchor_y(a)["kick_spin"], torch.tensor([[-1.0, 2.0, -3.0]]))


def test_bound_bernoulli_anchor_clamps_bernoulli_logits_only():
    a = _anchor()
    b = C.bound_bernoulli_anchor(a, 6.9)
    for k in C.BERNOULLI_KEYS:
        assert float(b[k].abs().max()) <= 6.9 + 1e-6
        assert torch.equal(torch.sign(b[k]), torch.sign(a[k]))
    assert torch.equal(b["pass_target_logits"], a["pass_target_logits"])
    assert torch.equal(b["move_direction"], a["move_direction"])
    assert torch.equal(C.bound_bernoulli_anchor(a, 0.0)["exec_move_logit"], a["exec_move_logit"])


def test_flip_consistency_metrics_are_zero_for_an_exactly_mirrored_pair_and_positive_otherwise():
    p = _anchor(200)
    q = C.mirror_anchor_y(p)                      # Q's output is exactly the mirror of P's
    m = C.flip_consistency_metrics(p, q)
    assert m["move_dir_err_p90_deg"] < 1e-3
    assert m["exec_move_sign_disagree_pct"] == 0.0
    assert m["exec_move_abs_logit_diff_p99"] == 0.0
    bad = dict(q)
    bad["exec_move_logit"] = -q["exec_move_logit"]
    bad["move_direction"] = torch.stack([q["move_direction"][:, 0], q["move_direction"][:, 1] + 3.0], dim=-1)
    m2 = C.flip_consistency_metrics(p, bad)
    assert m2["exec_move_sign_disagree_pct"] > 90.0
    assert m2["move_dir_err_p50_deg"] > 1.0


def test_flip_consistency_metrics_moving_mask_restricts_direction_stats():
    p = _anchor(50)
    q = C.mirror_anchor_y(p)
    q["move_direction"] = torch.stack([-q["move_direction"][:, 0], q["move_direction"][:, 1]], dim=-1)   # wrong on every row
    moving = torch.zeros(50, dtype=torch.bool)
    assert C.flip_consistency_metrics(p, q, moving)["move_dir_err_p50_deg"] == C.flip_consistency_metrics(p, q)["move_dir_err_p50_deg"]
    moving[:10] = True
    assert C.flip_consistency_metrics(p, q, moving)["move_dir_err_p50_deg"] > 0


def test_bounded_share_pct():
    h = {"exec_move_logit": torch.tensor([[1.0], [8.0], [-3.0], [-9.0]])}
    assert C.bounded_share_pct(h, 6.9) == {"exec_move": 50.0}

"""Coverage for the RAM-flattening helpers in ai/ppo/ppo_trainer.py:
``_merge_worker_batches(release_inputs=...)`` and ``_split_batch_releasing``.

Both must produce EXACTLY what the copying versions did -- the only
behavioural difference is that inputs are emptied key by key as they are
consumed (so the full data and its copy are never both alive).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from footballcoach.ai.ppo.ppo_trainer import _merge_worker_batches, _split_batch_releasing


def _make_batch(n: int, offset: float = 0.0) -> dict:
    base = torch.arange(n, dtype=torch.float32) + offset
    return {
        "obs/self_feat": torch.stack([base, base * 2], dim=1),
        "returns": base.clone(),
        "dones": (base % 3 == 0).float(),
        "reward_comps_raw": [{"r": float(x)} for x in base],
        "step_outcomes": [("win" if x % 3 == 0 else "") for x in base],
        "track_ids": ["trainee" if x % 2 == 0 else "opponent" for x in base],
    }


def _assert_batches_equal(a: dict, b: dict) -> None:
    assert set(a.keys()) == set(b.keys())
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


class TestMergeWorkerBatches:
    def test_release_inputs_gives_identical_result_and_empties_inputs(self):
        batches = [_make_batch(5, 0.0), _make_batch(3, 100.0), _make_batch(4, 200.0)]
        reference = _merge_worker_batches(copy.deepcopy(batches))  # default: copying behaviour

        merged = _merge_worker_batches(batches, release_inputs=True)

        _assert_batches_equal(merged, reference)
        assert all(len(b) == 0 for b in batches), "every input dict should have been emptied"

    def test_default_leaves_inputs_untouched(self):
        batches = [_make_batch(2), _make_batch(2, 10.0)]
        snapshot = copy.deepcopy(batches)
        _merge_worker_batches(batches)
        for b, s in zip(batches, snapshot):
            _assert_batches_equal(b, s)

    def test_row_order_is_input_order(self):
        merged = _merge_worker_batches([_make_batch(2, 0.0), _make_batch(2, 50.0)], release_inputs=True)
        assert merged["returns"].tolist() == [0.0, 1.0, 50.0, 51.0]
        assert len(merged["track_ids"]) == 4


class TestSplitBatchReleasing:
    def test_matches_plain_mask_slicing_for_tensors_and_lists(self):
        batch = _make_batch(10)
        reference = copy.deepcopy(batch)
        val_mask = np.zeros(10, dtype=bool)
        val_mask[[2, 3, 7]] = True
        train_mask = ~val_mask

        train, val = _split_batch_releasing(batch, train_mask, val_mask)

        t_idx = torch.from_numpy(np.where(train_mask)[0]).long()
        v_idx = torch.from_numpy(np.where(val_mask)[0]).long()
        for k, v in reference.items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(train[k], v[t_idx]), k
                assert torch.equal(val[k], v[v_idx]), k
            else:
                assert train[k] == [v[i] for i in t_idx.tolist()], k
                assert val[k] == [v[i] for i in v_idx.tolist()], k

    def test_empties_the_source_batch(self):
        batch = _make_batch(6)
        mask = np.array([True, True, False, True, False, True])
        _split_batch_releasing(batch, mask, ~mask)
        assert len(batch) == 0

    def test_no_val_mask_gives_none_val_and_full_train_selection(self):
        batch = _make_batch(4)
        reference = copy.deepcopy(batch)
        train, val = _split_batch_releasing(batch, np.ones(4, dtype=bool), None)
        assert val is None
        _assert_batches_equal(train, reference)

    def test_split_partitions_every_row_exactly_once(self):
        batch = _make_batch(20)
        val_mask = np.zeros(20, dtype=bool)
        val_mask[10:] = True
        train, val = _split_batch_releasing(batch, ~val_mask, val_mask)
        both = sorted(train["returns"].tolist() + val["returns"].tolist())
        assert both == [float(i) for i in range(20)]

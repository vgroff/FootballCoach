"""Offline demonstration dataset for BC pre-training.

Demonstrations are recorded by running a scenario with rules-based AI on both
sides and collecting (observation, bc_label) pairs at each decision step.

File format: one ``.npz`` file per recording session, containing:
  - ``obs_self_feat``:   (N, PLAYER_FEATURE_DIM) float32
  - ``obs_other_feat``:  (N, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM) float32
  - ``obs_exists_mask``: (N, MAX_OTHER_PLAYERS) float32
  - ``obs_ball_feat``:   (N, BALL_FEATURE_DIM) float32
  - ``obs_global_feat``: (N, GLOBAL_FEATURE_DIM) float32
  - ``bc_labels``:       (N, BC_LABEL_DIM) float32
  - ``meta_phase``:      scalar int — phase ID this was recorded for
  - ``meta_scenario``:   bytes — scenario key string

Multiple .npz files can be loaded together as one logical dataset and sampled
with replacement (uniform over all steps across all files).

Usage::

    from footballcoach.ai.bc.dataset import DemonstrationDataset

    ds = DemonstrationDataset.from_directory("demonstrations/phase1/")
    for obs_dict, labels in ds.iterate_minibatches(batch_size=64, shuffle=True):
        ...  # obs_dict keys match ObservationBatch.to_torch_dict()
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Generator

import numpy as np
import torch

from footballcoach.ai.ppo.bc import BC_LABEL_DIM

log = logging.getLogger("footballcoach.ai.bc.dataset")


class DemonstrationDataset:
    """In-memory dataset loaded from one or more .npz demonstration files."""

    def __init__(
        self,
        obs_self_feat: np.ndarray,
        obs_other_feat: np.ndarray,
        obs_exists_mask: np.ndarray,
        obs_ball_feat: np.ndarray,
        obs_global_feat: np.ndarray,
        bc_labels: np.ndarray,
    ):
        n = len(obs_self_feat)
        assert all(len(x) == n for x in [
            obs_other_feat, obs_exists_mask, obs_ball_feat, obs_global_feat, bc_labels
        ]), "All arrays must have the same first dimension"
        self._self_feat = obs_self_feat
        self._other_feat = obs_other_feat
        self._exists_mask = obs_exists_mask
        self._ball_feat = obs_ball_feat
        self._global_feat = obs_global_feat
        self._labels = bc_labels
        self._n = n

    def __len__(self) -> int:
        return self._n

    @classmethod
    def from_file(cls, path: str | Path) -> "DemonstrationDataset":
        """Load a single .npz file."""
        data = np.load(path)
        return cls(
            obs_self_feat=data["obs_self_feat"],
            obs_other_feat=data["obs_other_feat"],
            obs_exists_mask=data["obs_exists_mask"],
            obs_ball_feat=data["obs_ball_feat"],
            obs_global_feat=data["obs_global_feat"],
            bc_labels=data["bc_labels"],
        )

    @classmethod
    def from_files(cls, paths: list[str | Path]) -> "DemonstrationDataset":
        """Load and concatenate multiple .npz files."""
        parts = [cls.from_file(p) for p in paths]
        return cls(
            obs_self_feat=np.concatenate([p._self_feat for p in parts]),
            obs_other_feat=np.concatenate([p._other_feat for p in parts]),
            obs_exists_mask=np.concatenate([p._exists_mask for p in parts]),
            obs_ball_feat=np.concatenate([p._ball_feat for p in parts]),
            obs_global_feat=np.concatenate([p._global_feat for p in parts]),
            bc_labels=np.concatenate([p._labels for p in parts]),
        )

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        pattern: str = "*.npz",
    ) -> "DemonstrationDataset":
        """Load all .npz files matching *pattern* from *directory*."""
        directory = Path(directory)
        paths = sorted(directory.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No .npz files found in {directory}")
        log.info(f"Loading {len(paths)} demonstration file(s) from {directory}")
        ds = cls.from_files(paths)
        log.info(f"Dataset: {len(ds):,} steps loaded")
        return ds

    # Only steps where bc_labels[:, -1] == 1.0 (valid flag) are useful.
    def valid_indices(self) -> np.ndarray:
        return np.where(self._labels[:, -1] > 0.5)[0]

    def iterate_minibatches(
        self,
        batch_size: int,
        shuffle: bool = True,
        device: torch.device | None = None,
        valid_only: bool = True,
    ) -> Generator[tuple[dict, torch.Tensor], None, None]:
        """Yield (obs_dict, bc_labels) minibatches.

        Args:
            batch_size: Number of steps per batch.
            shuffle: Whether to shuffle indices each epoch.
            device: torch device to move tensors to (default: CPU).
            valid_only: If True, only yield steps where bc_label.valid==1.
        """
        indices = self.valid_indices() if valid_only else np.arange(self._n)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            if len(idx) == 0:
                continue
            obs_dict = {
                "self_feat":   _to_tensor(self._self_feat[idx],   device),
                "other_feat":  _to_tensor(self._other_feat[idx],  device),
                "exists_mask": _to_tensor(self._exists_mask[idx], device),
                "ball_feat":   _to_tensor(self._ball_feat[idx],   device),
                "global_feat": _to_tensor(self._global_feat[idx], device),
            }
            labels = _to_tensor(self._labels[idx], device)
            yield obs_dict, labels

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device | None = None,
        valid_only: bool = True,
    ) -> tuple[dict, torch.Tensor]:
        """Sample a random batch (with replacement)."""
        pool = self.valid_indices() if valid_only else np.arange(self._n)
        idx = np.random.choice(pool, size=min(batch_size, len(pool)), replace=True)
        obs_dict = {
            "self_feat":   _to_tensor(self._self_feat[idx],   device),
            "other_feat":  _to_tensor(self._other_feat[idx],  device),
            "exists_mask": _to_tensor(self._exists_mask[idx], device),
            "ball_feat":   _to_tensor(self._ball_feat[idx],   device),
            "global_feat": _to_tensor(self._global_feat[idx], device),
        }
        labels = _to_tensor(self._labels[idx], device)
        return obs_dict, labels


def _to_tensor(arr: np.ndarray, device: torch.device | None) -> torch.Tensor:
    t = torch.from_numpy(arr.astype(np.float32))
    return t.to(device) if device is not None else t

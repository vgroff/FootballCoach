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
  - ``rewards``:         (N,) float32  — per-step reward received (0 for mid-episode callback samples)
  - ``dones``:           (N,) float32  — 1.0 at episode boundaries, else 0.0
  - ``meta_phase``:      scalar int — phase ID this was recorded for
  - ``meta_scenario``:   bytes — scenario key string

Older files without ``rewards``/``dones`` are loaded with zeros for backward compatibility.

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
        rewards: np.ndarray | None = None,
        dones: np.ndarray | None = None,
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
        # rewards/dones are optional — older files won't have them
        self._rewards = rewards if rewards is not None else np.zeros(n, dtype=np.float32)
        self._dones   = dones   if dones   is not None else np.zeros(n, dtype=np.float32)
        self._n = n

    @property
    def has_rewards(self) -> bool:
        """True if this dataset was recorded with reward/done data."""
        return self._rewards.any()

    def __len__(self) -> int:
        return self._n

    @classmethod
    def from_file(cls, path: str | Path) -> "DemonstrationDataset":
        """Load a single .npz file."""
        data = np.load(path)
        n = len(data["obs_self_feat"])
        return cls(
            obs_self_feat=data["obs_self_feat"],
            obs_other_feat=data["obs_other_feat"],
            obs_exists_mask=data["obs_exists_mask"],
            obs_ball_feat=data["obs_ball_feat"],
            obs_global_feat=data["obs_global_feat"],
            bc_labels=data["bc_labels"],
            rewards=data["rewards"] if "rewards" in data else np.zeros(n, dtype=np.float32),
            dones=data["dones"]     if "dones"   in data else np.zeros(n, dtype=np.float32),
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
            rewards=np.concatenate([p._rewards for p in parts]),
            dones=np.concatenate([p._dones for p in parts]),
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

    def compute_returns(self, gamma: float = 0.99) -> np.ndarray:
        """Compute discounted returns G_t = r_t + gamma*r_{t+1} + ... per step.

        Episode boundaries are determined by ``dones``.  Steps at the end of
        an episode (done=1) bootstrap to 0.  The return array has the same
        length as the dataset.

        Args:
            gamma: Discount factor (default matches PPO config).

        Returns:
            float32 array of shape (N,) with per-step discounted returns.
        """
        rewards = self._rewards
        dones   = self._dones
        n = self._n
        returns = np.zeros(n, dtype=np.float32)
        running = 0.0
        # Scan backward: at a done boundary reset running sum
        for i in range(n - 1, -1, -1):
            if dones[i] > 0.5:
                running = 0.0
            running = rewards[i] + gamma * running
            returns[i] = running
        return returns

    def iterate_minibatches_with_returns(
        self,
        batch_size: int,
        returns: np.ndarray,
        shuffle: bool = True,
        device: torch.device | None = None,
    ):
        """Yield (obs_dict, returns_batch) minibatches for value pre-training.

        Args:
            batch_size: Steps per batch.
            returns: Pre-computed return array from ``compute_returns()``.
            shuffle: Shuffle indices each epoch.
            device: torch device.
        """
        indices = np.arange(self._n)
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
            ret_batch = _to_tensor(returns[idx], device)
            yield obs_dict, ret_batch

    def iterate_minibatches(
        self,
        batch_size: int,
        shuffle: bool = True,
        device: torch.device | None = None,
        valid_only: bool = True,
        returns: np.ndarray | None = None,
    ) -> Generator:
        """Yield (obs_dict, bc_labels) or (obs_dict, bc_labels, returns) minibatches.

        Args:
            batch_size: Number of steps per batch.
            shuffle: Whether to shuffle indices each epoch.
            device: torch device to move tensors to (default: CPU).
            valid_only: If True, only yield steps where bc_label.valid==1.
            returns: Optional pre-computed return array from ``compute_returns()``.
                When provided, yields 3-tuples ``(obs_dict, labels, returns_batch)``
                instead of 2-tuples so callers can add a value loss in the same pass.
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
            if returns is not None:
                yield obs_dict, labels, _to_tensor(returns[idx], device)
            else:
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

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

from footballcoach.ai.obs.schema import AI_TYPE_ONE_HOT_DIM
from footballcoach.ai.ppo.bc import (
    AI_TYPE_IMMOBILE,
    AI_TYPE_NEURAL,
    AI_TYPE_RULES,
    BC_LABEL_DIM,
    _I_AI_TYPE,
    _I_DIR_X,
    _I_DIR_Y,
    _I_EXEC_MOVE,
    _I_KICK_THIS_TICK,
    _I_OPPONENT_AI_TYPE,
    _I_SPRINT,
    _I_TACKLE_ATTEMPT,
    _I_VALID,
)

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
        reward_components: np.ndarray | None = None,
        reward_component_keys: list[str] | None = None,
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
        # reward_components/keys are optional -- older files predate per-
        # component persistence (see record_demonstrations.py). Falls back to
        # an empty (n, 0) array + empty key list rather than zeros-with-
        # unknown-width, since there's no width to infer without the keys.
        from footballcoach.ai.ppo.ppo_trainer import REWARD_COMP_LABELS
        self._reward_component_keys = reward_component_keys if reward_component_keys is not None else [k for k, _ in REWARD_COMP_LABELS]
        self._reward_components = (
            reward_components if reward_components is not None
            else np.zeros((n, len(self._reward_component_keys)), dtype=np.float32)
        )
        self._n = n
        self._trivial_mask_cache: np.ndarray | None = None

    @property
    def has_rewards(self) -> bool:
        """True if this dataset was recorded with reward/done data."""
        return self._rewards.any()

    @property
    def has_reward_components(self) -> bool:
        """True if this dataset was recorded with per-component reward data."""
        return self._reward_components.any()

    def __len__(self) -> int:
        return self._n

    @classmethod
    def from_file(cls, path: str | Path) -> "DemonstrationDataset":
        """Load a single .npz file."""
        data = np.load(path)
        n = len(data["obs_self_feat"])
        bc_labels = data["bc_labels"]
        # Schema guard: bc_labels' last dim must match the current BC_LABEL_DIM
        # (currently 24, after kick_direction/kick_power/kick_spin fields were
        # appended at indices 18-23, on top of the W6 opponent_ai_type field at
        # index 17). Older recordings (16-, 17-, or 18-wide) are no longer
        # loadable — they must be re-recorded via record_demonstrations.py.
        if bc_labels.shape[-1] != BC_LABEL_DIM:
            raise ValueError(
                f"{path}: bc_labels has width {bc_labels.shape[-1]}, expected "
                f"BC_LABEL_DIM={BC_LABEL_DIM}. This .npz was recorded with an "
                f"older BC label schema (e.g. missing kick_direction/kick_power/"
                f"kick_spin fields added for full execution-head BC coverage). "
                f"Re-record demonstrations: "
                f"uv run python -m footballcoach.ai.scripts.record_demonstrations "
                f"--phase 1 --n-episodes <N> --output <dir>"
            )
        _rc_keys = (
            [str(k) for k in data["meta_reward_component_keys"]]
            if "meta_reward_component_keys" in data else None
        )
        return cls(
            obs_self_feat=data["obs_self_feat"],
            obs_other_feat=data["obs_other_feat"],
            obs_exists_mask=data["obs_exists_mask"],
            obs_ball_feat=data["obs_ball_feat"],
            obs_global_feat=data["obs_global_feat"],
            bc_labels=bc_labels,
            rewards=data["rewards"] if "rewards" in data else np.zeros(n, dtype=np.float32),
            dones=data["dones"]     if "dones"   in data else np.zeros(n, dtype=np.float32),
            reward_components=data["reward_components"] if "reward_components" in data else None,
            reward_component_keys=_rc_keys,
        )

    @classmethod
    def from_files(cls, paths: list[str | Path]) -> "DemonstrationDataset":
        """Load and concatenate multiple .npz files."""
        parts = [cls.from_file(p) for p in paths]
        # reward_component_keys must agree across files (fixed by REWARD_COMP_LABELS
        # order at recording time) -- use the first part's keys/columns; parts
        # recorded before this feature existed already fell back to the same
        # default key list in __init__, so concatenation stays column-aligned.
        return cls(
            obs_self_feat=np.concatenate([p._self_feat for p in parts]),
            obs_other_feat=np.concatenate([p._other_feat for p in parts]),
            obs_exists_mask=np.concatenate([p._exists_mask for p in parts]),
            obs_ball_feat=np.concatenate([p._ball_feat for p in parts]),
            obs_global_feat=np.concatenate([p._global_feat for p in parts]),
            bc_labels=np.concatenate([p._labels for p in parts]),
            rewards=np.concatenate([p._rewards for p in parts]),
            dones=np.concatenate([p._dones for p in parts]),
            reward_components=np.concatenate([p._reward_components for p in parts]),
            reward_component_keys=parts[0]._reward_component_keys,
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

    # Only steps where bc_labels[:, _I_VALID] == 1.0 and SELF is not immobile
    # are used for BC training -- this must key off ai_type (self), not
    # opponent_ai_type. A row where self is rules/neural is a legitimate BC
    # example of "how to play" regardless of what the opponent is doing
    # (including standing still) -- excluding those too (as an earlier,
    # buggy version of this method did via opponent_ai_type) silently threw
    # away every rules-AI demonstration row from rules-vs-immobile episodes,
    # not just the immobile player's own (correctly excluded) rows.
    def valid_indices(self) -> np.ndarray:
        valid = self._labels[:, _I_VALID] > 0.5
        self_not_immobile = self._labels[:, _I_AI_TYPE] < (AI_TYPE_IMMOBILE - 0.5)
        return np.where(valid & self_not_immobile)[0]

    def split_train_val_indices(
        self, val_frac: float = 0.15, valid_only: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Episode-level train/val split of this dataset's row indices.

        Splits by COMPLETE episodes (using ``_dones``) so no single episode's
        rows span both sets — mirrors the identical split already used by
        ``PPOTrainer.pretrain_value()`` for its own train/val rollout split.
        The LAST ``val_frac`` fraction of complete episodes (in dataset
        order) become the val set; the rest are train.

        Args:
            val_frac: Fraction of complete episodes to hold out for
                validation. 0.0 disables the split (val indices will be empty).
            valid_only: If True, restrict both splits to ``valid_indices()``
                (BC-valid, non-immobile-self rows) before splitting —
                matches the row set BC training actually iterates over.

        Returns:
            (train_indices, val_indices): two disjoint index arrays into this
            dataset's rows (NOT episode ids). val_indices is empty if there
            are fewer than 2 complete episodes or val_frac <= 0.
        """
        base_indices = self.valid_indices() if valid_only else np.arange(self._n)
        if val_frac <= 0.0 or len(base_indices) == 0:
            return base_indices, np.array([], dtype=base_indices.dtype)

        dones_subset = self._dones[base_indices]
        episode_end_positions = np.where(dones_subset > 0.5)[0]
        n_complete_eps = len(episode_end_positions)
        if n_complete_eps < 2:
            return base_indices, np.array([], dtype=base_indices.dtype)

        n_val_eps = max(1, round(val_frac * n_complete_eps))
        n_train_eps = n_complete_eps - n_val_eps
        ep_starts = np.concatenate([[0], episode_end_positions[:-1] + 1])

        val_mask = np.zeros(len(base_indices), dtype=bool)
        for _i in range(n_train_eps, n_complete_eps):
            val_mask[ep_starts[_i]:episode_end_positions[_i] + 1] = True
        train_mask = ~val_mask

        return base_indices[train_mask], base_indices[val_mask]

    def compute_pos_weights(self, max_weight: float | None = None) -> dict[str, float]:
        """Inverse-frequency weights for rare Bernoulli BC targets.

        weight = n_negative / max(n_positive, 1), matching
        ``torch.nn.functional.binary_cross_entropy_with_logits``'s
        ``pos_weight`` argument semantics (weight applied to the positive
        class term). Computed once over all valid rows in this dataset.

        Args:
            max_weight: optional cap on the raw inverse-frequency ratio.
                ``None`` (default) = uncapped. Extreme class imbalance (e.g.
                tackle_attempt at ~1:130) can push the auto-computed weight
                high enough that a single positive example dominates the BC
                gradient and drives the Bernoulli logit toward saturation —
                see ai_config.json ``bc.pos_weight_max``.
        """
        valid = self.valid_indices()
        labels = self._labels[valid]
        out: dict[str, float] = {}
        for name, col in [
            ("kick", _I_KICK_THIS_TICK),
            ("tackle_attempt", _I_TACKLE_ATTEMPT),
        ]:
            n_pos = float((labels[:, col] > 0.5).sum())
            n_neg = float(len(labels)) - n_pos
            weight = n_neg / max(n_pos, 1.0)
            if max_weight is not None:
                weight = min(weight, max_weight)
            out[name] = weight
        return out

    def _compute_trivial_mask(
        self,
        cos_threshold: float = 0.98,
        exclude_radius_steps: int = 5,
    ) -> np.ndarray:
        """Static (epoch-independent) mask of rows eligible for downsampling.

        A row is "trivial" if its ``move_direction`` label is closely aligned
        (cosine similarity above ``cos_threshold``) with the immediately
        preceding row's ``move_direction`` *within the same episode* — i.e.
        "moving the same way as before, nothing interesting changed".

        Row 0 of every episode (no valid "previous" row within the episode)
        is never eligible. Rows within ``exclude_radius_steps`` of a
        ``kick_this_tick=1``, ``tackle_attempt=1``, or a ``sprint``/
        ``exec_move`` speedMode change from the preceding row (same episode)
        are also never eligible — a speedMode transition (e.g. SPRINT to JOG
        braking into a target) is itself a non-trivial event that a fixed
        0.3s sample interval can easily miss on either side of, so run-ups
        immediately around it are protected the same way as kicks/tackles.

        Computed once at load time; the *exclusion subset* drawn from this
        mask is what gets freshly re-rolled every epoch (see
        ``iterate_minibatches``).
        """
        n = self._n
        trivial = np.zeros(n, dtype=bool)
        if n == 0:
            return trivial

        dir_x = self._labels[:, _I_DIR_X]
        dir_y = self._labels[:, _I_DIR_Y]
        norm = np.sqrt(dir_x ** 2 + dir_y ** 2)
        has_dir = norm > 1e-6

        # Episode-boundary-aware "previous row" lookup: row i's previous row
        # is i-1, unless dones[i-1] marks i-1 as an episode's final step.
        prev_is_same_episode = np.ones(n, dtype=bool)
        prev_is_same_episode[0] = False
        if n > 1:
            prev_is_same_episode[1:] = self._dones[:-1] <= 0.5

        cos_sim = np.zeros(n, dtype=np.float64)
        eligible = has_dir & prev_is_same_episode
        eligible[1:] &= has_dir[:-1]
        idx = np.where(eligible)[0]
        if len(idx) > 0:
            cur = np.stack([dir_x[idx], dir_y[idx]], axis=-1)
            prev = np.stack([dir_x[idx - 1], dir_y[idx - 1]], axis=-1)
            cur_n = cur / np.maximum(np.linalg.norm(cur, axis=-1, keepdims=True), 1e-6)
            prev_n = prev / np.maximum(np.linalg.norm(prev, axis=-1, keepdims=True), 1e-6)
            cos_sim[idx] = (cur_n * prev_n).sum(axis=-1)
        trivial[idx] = cos_sim[idx] > cos_threshold

        # SpeedMode transition (sprint or exec_move differs from the previous
        # row, same episode) counts as a rare event too -- see docstring.
        sprint = self._labels[:, _I_SPRINT]
        exec_move = self._labels[:, _I_EXEC_MOVE]
        speed_mode_changed = np.zeros(n, dtype=bool)
        if n > 1:
            changed = (sprint[1:] != sprint[:-1]) | (exec_move[1:] != exec_move[:-1])
            speed_mode_changed[1:] = changed & prev_is_same_episode[1:]

        # Exclude rows within exclude_radius_steps of a rare event, same episode.
        rare_event = (
            (self._labels[:, _I_KICK_THIS_TICK] > 0.5)
            | (self._labels[:, _I_TACKLE_ATTEMPT] > 0.5)
            | speed_mode_changed
        )
        if exclude_radius_steps > 0 and rare_event.any():
            near_rare = np.zeros(n, dtype=bool)
            rare_idx = np.where(rare_event)[0]
            # Episode ids via cumulative-sum of dones (shifted so boundary row
            # keeps the episode it belongs to).
            episode_id = np.concatenate([[0], np.cumsum(self._dones[:-1] > 0.5)]) if n > 1 else np.zeros(1)
            for ri in rare_idx:
                lo = max(0, ri - exclude_radius_steps)
                hi = min(n, ri + exclude_radius_steps + 1)
                same_ep = episode_id[lo:hi] == episode_id[ri]
                near_rare[lo:hi] |= same_ep
            trivial &= ~near_rare

        return trivial

    def downsample_trivial_stats(
        self,
        valid_only: bool = True,
        cos_threshold: float = 0.98,
        exclude_radius_steps: int = 5,
        frac: float = 0.0,
    ) -> dict:
        """Report how many rows qualify as "trivial" and how many would be
        excluded at a given ``frac``, for logging/diagnostics.

        Uses (and populates) the same cached ``_trivial_mask_cache`` as
        ``iterate_minibatches(downsample_trivial_frac=...)``, so the reported
        counts match what training actually sees.

        Returns:
            dict with ``n_total``, ``n_trivial``, ``trivial_frac`` (fraction
            of ``n_total`` classified trivial), ``n_excluded_at_frac``
            (expected row count excluded this epoch at the given ``frac``).
        """
        indices = self.valid_indices() if valid_only else np.arange(self._n)
        if self._trivial_mask_cache is None:
            self._trivial_mask_cache = self._compute_trivial_mask(
                cos_threshold=cos_threshold,
                exclude_radius_steps=exclude_radius_steps,
            )
        n_total = len(indices)
        n_trivial = int(self._trivial_mask_cache[indices].sum())
        return {
            "n_total": n_total,
            "n_trivial": n_trivial,
            "trivial_frac": n_trivial / n_total if n_total > 0 else 0.0,
            "n_excluded_at_frac": int(round(n_trivial * frac)),
        }

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

    def compute_component_returns(self, gamma: float = 0.99) -> dict[str, np.ndarray]:
        """Per-component discounted MC return, mirroring ``compute_returns()``
        but computed independently for each reward component column so the
        components sum exactly to ``compute_returns()``'s total return
        (each component uses the SAME gamma/episode boundaries).

        Requires the dataset to have been recorded with per-component reward
        data (``has_reward_components``); returns an all-zero array per key
        otherwise (same "silently zero, don't crash" convention as
        ``has_rewards``/``compute_returns()`` on reward-less datasets).

        Returns:
            dict mapping component short-key (see REWARD_COMP_LABELS) to a
            float32 array of shape (N,), one MC return-contribution series
            per component.
        """
        dones = self._dones
        n = self._n
        out: dict[str, np.ndarray] = {}
        for col, key in enumerate(self._reward_component_keys):
            comp_rewards = self._reward_components[:, col]
            comp_returns = np.zeros(n, dtype=np.float32)
            running = 0.0
            for i in range(n - 1, -1, -1):
                if dones[i] > 0.5:
                    running = 0.0
                running = comp_rewards[i] + gamma * running
                comp_returns[i] = running
            out[key] = comp_returns
        return out

    def iterate_minibatches_with_returns(
        self,
        batch_size: int,
        returns: np.ndarray,
        shuffle: bool = True,
        device: torch.device | None = None,
        valid_only: bool = False,
    ):
        """Yield (obs_dict, returns_batch) minibatches for value pre-training.

        Args:
            batch_size: Steps per batch.
            returns: Pre-computed return array from ``compute_returns()``.
            shuffle: Shuffle indices each epoch.
            device: torch device.
            valid_only: If True, only yield steps where bc_label.valid==1.
                Default False (preserves prior behaviour: value-target
                fitting benefits from the full return distribution,
                including "boring"/invalid-BC-label states — those are
                still informative for the critic). Deliberately NOT
                combined with trivial-row downsampling here: downsampling
                is specifically about reducing redundant *BC* signal, not
                value-target signal.
        """
        indices = self.valid_indices() if valid_only else np.arange(self._n)
        # Shuffle BLOCK order, not row order: indices is sorted, so shuffling
        # rows BEFORE gathering would make every gather below (self._self_feat[idx],
        # etc.) touch random, far-apart rows -- shuffling which contiguous block
        # goes in which minibatch slot preserves random batch composition across
        # epochs while keeping each gather a near-contiguous slice.
        block_starts = list(range(0, len(indices), batch_size))
        if shuffle:
            np.random.shuffle(block_starts)
        for start in block_starts:
            idx = indices[start:start + batch_size]
            if len(idx) == 0:
                continue
            self_ai_type, other_ai_type = _build_ai_type_arrays(
                self._labels[idx], self._exists_mask[idx]
            )
            self_feat_t   = _to_tensor(self._self_feat[idx],   device)
            other_feat_t  = _to_tensor(self._other_feat[idx],  device)
            exists_mask_t = _to_tensor(self._exists_mask[idx], device)
            ball_feat_t   = _to_tensor(self._ball_feat[idx],   device)
            global_feat_t = _to_tensor(self._global_feat[idx], device)
            self_ai_type_t  = _to_tensor(self_ai_type,  device)
            other_ai_type_t = _to_tensor(other_ai_type, device)
            ret_batch_t = _to_tensor(returns[idx], device)
            if shuffle:
                # Row order WITHIN the block is shuffled AFTER gathering (cheap,
                # tensor-only permutation) so a dataset small enough to fit in one
                # block (n <= batch_size) still randomizes row order every epoch --
                # block-order shuffling alone is a no-op with only 1 block.
                perm = torch.randperm(len(idx))
                self_feat_t, other_feat_t, exists_mask_t, ball_feat_t, global_feat_t = (
                    self_feat_t[perm], other_feat_t[perm], exists_mask_t[perm],
                    ball_feat_t[perm], global_feat_t[perm],
                )
                self_ai_type_t, other_ai_type_t = self_ai_type_t[perm], other_ai_type_t[perm]
                ret_batch_t = ret_batch_t[perm]
            obs_dict = {
                "self_feat":   self_feat_t,
                "other_feat":  other_feat_t,
                "exists_mask": exists_mask_t,
                "ball_feat":   ball_feat_t,
                "global_feat": global_feat_t,
                "self_ai_type":  self_ai_type_t,
                "other_ai_type": other_ai_type_t,
            }
            yield obs_dict, ret_batch_t

    def iterate_minibatches(
        self,
        batch_size: int,
        shuffle: bool = True,
        device: torch.device | None = None,
        valid_only: bool = True,
        returns: np.ndarray | None = None,
        downsample_trivial_frac: float = 0.0,
        downsample_trivial_cos_threshold: float = 0.98,
        downsample_trivial_exclude_radius_steps: int = 5,
        rng: np.random.Generator | None = None,
        indices_override: np.ndarray | None = None,
    ) -> Generator:
        """Yield (obs_dict, bc_labels) or (obs_dict, bc_labels, returns) minibatches.

        Args:
            batch_size: Number of steps per batch.
            shuffle: Whether to shuffle indices each epoch.
            device: torch device to move tensors to (default: CPU).
            valid_only: If True, only yield steps where bc_label.valid==1.
                Ignored when ``indices_override`` is given.
            returns: Optional pre-computed return array from ``compute_returns()``.
                When provided, yields 3-tuples ``(obs_dict, labels, returns_batch)``
                instead of 2-tuples so callers can add a value loss in the same pass.
            downsample_trivial_frac: Fraction of "trivial" movement rows
                (see ``_compute_trivial_mask()``) to exclude from THIS call.
                0.0 (default) disables downsampling entirely. A fresh random
                subset is drawn each call (i.e. each epoch), not a fixed mask —
                non-trivial rows are never excluded regardless of this value.
            downsample_trivial_cos_threshold: Cosine-similarity threshold for
                the trivial-row classification (see ``_compute_trivial_mask()``).
            downsample_trivial_exclude_radius_steps: Rows within this many
                steps of a kick/tackle event (same episode) are never
                eligible for downsampling.
            rng: Optional ``numpy.random.Generator`` for reproducible
                per-epoch re-rolling of the excluded subset. Defaults to a
                fresh, unseeded generator each call.
            indices_override: When given, iterate over exactly this row-index
                array instead of ``valid_indices()``/``arange(n)`` -- used by
                callers that pre-computed a train/val split (see
                ``split_train_val_indices()``) so the val subset is never
                re-derived from the full dataset. Downsampling still applies
                on top of this subset when requested.
        """
        indices = (
            indices_override if indices_override is not None
            else (self.valid_indices() if valid_only else np.arange(self._n))
        )

        if downsample_trivial_frac > 0.0:
            if self._trivial_mask_cache is None:
                self._trivial_mask_cache = self._compute_trivial_mask(
                    cos_threshold=downsample_trivial_cos_threshold,
                    exclude_radius_steps=downsample_trivial_exclude_radius_steps,
                )
            trivial_in_indices = self._trivial_mask_cache[indices]
            trivial_positions = np.where(trivial_in_indices)[0]
            if len(trivial_positions) > 0:
                _rng = rng if rng is not None else np.random.default_rng()
                n_exclude = int(round(len(trivial_positions) * downsample_trivial_frac))
                if n_exclude > 0:
                    excluded = _rng.choice(trivial_positions, size=n_exclude, replace=False)
                    keep_mask = np.ones(len(indices), dtype=bool)
                    keep_mask[excluded] = False
                    indices = indices[keep_mask]

        # Shuffle BLOCK order, not row order -- see iterate_minibatches_with_returns()
        # above for the cache-locality rationale (indices may already be a
        # filtered/gapped-but-sorted subset here due to downsampling above).
        block_starts = list(range(0, len(indices), batch_size))
        if shuffle:
            np.random.shuffle(block_starts)

        for start in block_starts:
            idx = indices[start:start + batch_size]
            if len(idx) == 0:
                continue
            self_ai_type, other_ai_type = _build_ai_type_arrays(
                self._labels[idx], self._exists_mask[idx]
            )
            self_feat_t   = _to_tensor(self._self_feat[idx],   device)
            other_feat_t  = _to_tensor(self._other_feat[idx],  device)
            exists_mask_t = _to_tensor(self._exists_mask[idx], device)
            ball_feat_t   = _to_tensor(self._ball_feat[idx],   device)
            global_feat_t = _to_tensor(self._global_feat[idx], device)
            self_ai_type_t  = _to_tensor(self_ai_type,  device)
            other_ai_type_t = _to_tensor(other_ai_type, device)
            labels_t = _to_tensor(self._labels[idx], device)
            returns_t = _to_tensor(returns[idx], device) if returns is not None else None
            if shuffle:
                # Row order WITHIN the block, shuffled AFTER gathering -- see
                # iterate_minibatches_with_returns() above for why this is needed
                # even with block-order shuffling (single-block datasets otherwise
                # never randomize row order at all).
                perm = torch.randperm(len(idx))
                self_feat_t, other_feat_t, exists_mask_t, ball_feat_t, global_feat_t = (
                    self_feat_t[perm], other_feat_t[perm], exists_mask_t[perm],
                    ball_feat_t[perm], global_feat_t[perm],
                )
                self_ai_type_t, other_ai_type_t = self_ai_type_t[perm], other_ai_type_t[perm]
                labels_t = labels_t[perm]
                if returns_t is not None:
                    returns_t = returns_t[perm]
            obs_dict = {
                "self_feat":   self_feat_t,
                "other_feat":  other_feat_t,
                "exists_mask": exists_mask_t,
                "ball_feat":   ball_feat_t,
                "global_feat": global_feat_t,
                "self_ai_type":  self_ai_type_t,
                "other_ai_type": other_ai_type_t,
            }
            if returns_t is not None:
                yield obs_dict, labels_t, returns_t
            else:
                yield obs_dict, labels_t

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device | None = None,
        valid_only: bool = True,
    ) -> tuple[dict, torch.Tensor]:
        """Sample a random batch (with replacement)."""
        pool = self.valid_indices() if valid_only else np.arange(self._n)
        idx = np.random.choice(pool, size=min(batch_size, len(pool)), replace=True)
        self_ai_type, other_ai_type = _build_ai_type_arrays(
            self._labels[idx], self._exists_mask[idx]
        )
        obs_dict = {
            "self_feat":   _to_tensor(self._self_feat[idx],   device),
            "other_feat":  _to_tensor(self._other_feat[idx],  device),
            "exists_mask": _to_tensor(self._exists_mask[idx], device),
            "ball_feat":   _to_tensor(self._ball_feat[idx],   device),
            "global_feat": _to_tensor(self._global_feat[idx], device),
            "self_ai_type":  _to_tensor(self_ai_type,  device),
            "other_ai_type": _to_tensor(other_ai_type, device),
        }
        labels = _to_tensor(self._labels[idx], device)
        return obs_dict, labels


def _to_tensor(arr: np.ndarray, device: torch.device | None) -> torch.Tensor:
    t = torch.from_numpy(arr.astype(np.float32))
    return t.to(device) if device is not None else t


def _ai_type_code_to_one_hot(codes: np.ndarray) -> np.ndarray:
    """Map AI_TYPE_RULES/IMMOBILE/NEURAL float codes -> (N, AI_TYPE_ONE_HOT_DIM) one-hot.

    Column order matches ai/obs/encoder.py::_ai_type_one_hot: [is_rules, is_immobile, is_neural].
    """
    out = np.zeros((len(codes), AI_TYPE_ONE_HOT_DIM), dtype=np.float32)
    out[codes == AI_TYPE_RULES, 0] = 1.0
    out[codes == AI_TYPE_IMMOBILE, 1] = 1.0
    out[codes == AI_TYPE_NEURAL, 2] = 1.0
    return out


def _build_ai_type_arrays(
    labels: np.ndarray, exists_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build (self_ai_type, other_ai_type) arrays for a batch of BC rows.

    Demo recordings are strictly 1v1 (Phase 1) so ``exists_mask`` has exactly
    one real "other player" slot per row - the opponent. ``bc_labels``'
    ``ai_type``/``opponent_ai_type`` columns (recorded directly at demo time,
    see ai/ppo/bc.py) give the correct one-hot for self and that single
    opponent slot respectively; every other (padded) slot stays all-zero.
    This mirrors ai/obs/encoder.py's ``_ai_type_one_hot`` for live rollouts.
    """
    n, n_slots = exists_mask.shape
    self_ai_type = _ai_type_code_to_one_hot(labels[:, _I_AI_TYPE])
    opp_one_hot = _ai_type_code_to_one_hot(labels[:, _I_OPPONENT_AI_TYPE])
    other_ai_type = np.zeros((n, n_slots, AI_TYPE_ONE_HOT_DIM), dtype=np.float32)
    # Slot index of the (single) real opponent per row; rows with no real
    # opponent slot (shouldn't happen for valid 1v1 demo rows) are left zero.
    has_opponent = exists_mask.any(axis=1)
    opp_slot = np.argmax(exists_mask, axis=1)
    other_ai_type[np.where(has_opponent)[0], opp_slot[has_opponent]] = opp_one_hot[has_opponent]
    return self_ai_type, other_ai_type

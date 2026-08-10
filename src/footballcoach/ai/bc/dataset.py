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
        episode_outcomes: np.ndarray | None = None,
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
        # Ground-truth per-EPISODE outcome strings (ScenarioEnv's
        # info.trial_outcome, verbatim -- "box_possession"/
        # "opponent_box_possession"/"timeout"/"miss"/"invalid", see
        # ai/env/outcome.py's outcome vocabulary), one entry per complete
        # episode in dataset order. None/missing (older recordings predating
        # record_demonstrations.py's meta_episode_outcomes) for a dataset
        # with dones -> classify_outcome() cannot know the true outcome and
        # must say so explicitly (never silently guess from reward
        # components again -- see classify_outcome()'s docstring).
        self._episode_outcomes = (
            np.asarray(episode_outcomes, dtype=object) if episode_outcomes is not None else None
        )
        self._n = n
        self._trivial_mask_cache: np.ndarray | None = None
        self._outcome_by_row_cache: np.ndarray | None = None
        self._episode_end_rows_cache: np.ndarray | None = None
        self._episode_ranges_cache: list[tuple[int, int]] | None = None

    @property
    def has_rewards(self) -> bool:
        """True if this dataset was recorded with reward/done data."""
        return self._rewards.any()

    @property
    def has_reward_components(self) -> bool:
        """True if this dataset was recorded with per-component reward data."""
        return self._reward_components.any()

    @property
    def has_episode_outcomes(self) -> bool:
        """True if this dataset has ground-truth per-episode outcome strings
        (``meta_episode_outcomes``, see ``classify_outcome()``) -- required
        for ``classify_outcome()``/``row_outcomes()``/``outcome_by_row()``.
        False for recordings made before this field existed; re-record to
        enable outcome-split diagnostics."""
        return self._episode_outcomes is not None

    # Number of consecutive done=1 rows that mark ONE real episode boundary
    # (see episode_row_ranges()'s docstring) -- record_demonstrations.py's
    # _record_now(player_id=None) always appends exactly trainee+opponent
    # (2 rows) for the terminal timed sample, both backfilled done=1.
    _DONE_ROWS_PER_EPISODE_BOUNDARY = 2

    def _full_dataset_episode_row_ranges(self) -> list[tuple[int, int]]:
        """Cached [(start, end_inclusive), ...] row-index ranges for each
        complete episode over the WHOLE dataset (unfiltered), computed
        directly from ``self._dones``.

        Episodes are delimited by ``dones==1``, but record_demonstrations.py's
        ``_record_now(player_id=None)`` appends BOTH the trainee's AND the
        opponent's row for every timed sample, and backfills ``done=1`` onto
        BOTH rows at episode end (see its "player_id=None -> records BOTH
        trainee and opponent -> appends 2 rows" comment) -- so a single real
        episode boundary is exactly ``_DONE_ROWS_PER_EPISODE_BOUNDARY`` (2)
        CONSECUTIVE ``done=1`` rows, not one, and NOT an arbitrarily long run
        either: if one episode's terminal pair is immediately followed by
        the very next episode's terminal pair (e.g. two consecutive
        single-timed-sample episodes with no intervening non-done row),
        treating "however many done=1 rows in a row" as ONE boundary would
        wrongly merge two real episodes into one. Every 2 consecutive
        done=1 rows are therefore closed off as their own episode
        immediately, regardless of what follows.

        MUST be computed over the FULL, unfiltered dataset -- not an
        arbitrary row_pool -- because a filtered pool (e.g.
        ``valid_indices()``, which can drop ONE of an episode's two
        terminal done=1 rows, such as the immobile opponent's own row) may
        never observe 2 consecutive done=1 rows for a real episode at all,
        silently losing that boundary. ``episode_row_ranges()`` intersects
        this full-dataset truth with any given row_pool instead of
        re-deriving boundaries from the (possibly filtered) pool itself.
        """
        if self._episode_ranges_cache is None:
            ranges: list[tuple[int, int]] = []
            start = 0
            done_streak = 0
            for idx in range(self._n):
                is_done = self._dones[idx] > 0.5
                if is_done:
                    done_streak += 1
                    if done_streak == self._DONE_ROWS_PER_EPISODE_BOUNDARY:
                        ranges.append((start, idx))
                        start = idx + 1
                        done_streak = 0
                else:
                    done_streak = 0
            self._episode_ranges_cache = ranges
        return self._episode_ranges_cache

    def episode_row_ranges(self, row_pool: np.ndarray) -> list[tuple[int, int]]:
        """Return [(start, end_inclusive), ...] row-index ranges, ONE PER
        FULL-DATASET EPISODE that has at least one row present in
        *row_pool* (assumed contiguous-in-dataset-order and sorted
        increasing, e.g. a train/val split from ``split_train_val_indices()``
        or ``np.arange(len(self))``) -- ``start``/``end`` are *row_pool*'s
        own first/last row belonging to that episode, which may be a proper
        subset of the full-dataset episode's rows if *row_pool* is filtered
        (e.g. ``valid_indices()``). Episode boundaries themselves are always
        resolved against the FULL dataset first (see
        ``_full_dataset_episode_row_ranges()``'s docstring for why), so a
        filtered pool can never miss or merge an episode boundary.

        A trailing partial episode (row_pool contains rows from an episode
        whose full-dataset boundary hasn't been reached yet, i.e. an
        incomplete recording) is dropped, matching prior behaviour.
        """
        if len(row_pool) == 0:
            return []
        full_ranges = self._full_dataset_episode_row_ranges()
        full_ends = np.array([end for _s, end in full_ranges], dtype=np.int64)
        # Which full-dataset episode does each row_pool row belong to?
        ep_of_row = np.searchsorted(full_ends, row_pool, side="left")
        ranges: list[tuple[int, int]] = []
        n_eps = len(full_ranges)
        pos = 0
        while pos < len(row_pool):
            ep_idx = int(ep_of_row[pos])
            if ep_idx >= n_eps:
                break  # trailing rows past the last complete episode -- drop.
            run_end = pos
            while run_end + 1 < len(row_pool) and ep_of_row[run_end + 1] == ep_idx:
                run_end += 1
            ranges.append((int(row_pool[pos]), int(row_pool[run_end])))
            pos = run_end + 1
        return ranges

    def n_episodes(self, row_pool: np.ndarray | None = None) -> int:
        """Number of COMPLETE episodes with at least one row present in
        *row_pool* (default: the whole dataset), correctly resolving
        episode boundaries against the FULL dataset even when *row_pool* is
        filtered (see ``episode_row_ranges()``'s docstring) -- use this
        instead of ``int(ds._dones.sum())``/``int(ds._dones[idx].sum())``,
        which silently double the true episode count whenever 2+ players
        share a timed sample (every real Phase-1 recording)."""
        pool = row_pool if row_pool is not None else np.arange(self._n)
        return len(self.episode_row_ranges(pool))

    def _episode_end_rows(self) -> np.ndarray:
        """Cached, sorted array of the LAST row index of each real episode
        over the WHOLE dataset (see ``_full_dataset_episode_row_ranges()``).
        Index ``i`` here is episode ``i``'s end row, matching
        ``self._episode_outcomes[i]``'s order 1:1."""
        if self._episode_end_rows_cache is None:
            ranges = self._full_dataset_episode_row_ranges()
            self._episode_end_rows_cache = np.array([end for _start, end in ranges], dtype=np.int64)
        return self._episode_end_rows_cache

    # Maps ScenarioEnv's ground-truth info.trial_outcome strings (see
    # ai/env/outcome.py's outcome vocabulary and ScenarioEnv.step()'s
    # "invalid" split) to the short labels used by debug_value_network.py /
    # value_mse_by_outcome() breakdowns. Phase 1 has exactly these five
    # possible endings -- ANY other/missing value is a genuine bug upstream
    # (see classify_outcome()), never a legitimate fifth outcome.
    _OUTCOME_LABEL_MAP: dict[str, str] = {
        "box_possession": "win",
        "opponent_box_possession": "loss",
        "timeout": "timeout",
        "miss": "ball_out",
        "invalid": "invalid",
    }

    def classify_outcome(self, end_row: int) -> str:
        """Classify the outcome of the episode CONTAINING *end_row* using the
        GROUND-TRUTH ``info.trial_outcome`` string persisted by
        record_demonstrations.py (see ``meta_episode_outcomes`` /
        ``self._episode_outcomes``) -- NOT inferred from reward components.

        *end_row* need not be an exact full-dataset episode-end row: callers
        commonly pass the LAST row of an episode within a FILTERED row_pool
        (e.g. ``row_outcomes(train_idx)`` after ``valid_indices()`` dropped
        some rows from that episode, such as the immobile opponent's own
        rows) -- that filtered last row is always <= the true full-dataset
        end row for the same episode and > the previous episode's end row,
        so mapping to "the first full-dataset episode-end row >= end_row"
        always resolves to the correct episode regardless of filtering.

        Phase 1 has exactly five possible endings (see ai/env/outcome.py /
        ScenarioEnv.step()): a player wins (box_possession), a player loses
        (opponent_box_possession), the ball goes out with a toucher (miss ->
        "ball_out"), the ball goes out with NO toucher (invalid -- nobody's
        fault, e.g. bad spawn), or timeout. There is no sixth case, and no
        legitimate "unknown" -- reward-component-based inference was wrong
        because "invalid" episodes fire NO per-player reward component at
        all (nothing to infer from), which is exactly why this method no
        longer infers anything.

        Raises ``ValueError`` if this dataset predates ``meta_episode_outcomes``
        (re-record demonstrations), if *end_row* falls after the last known
        episode boundary, or if the recorded string isn't one of the five
        known outcomes above -- all are bugs to fix at the source, not
        something to paper over with a silent "unknown" bucket.
        """
        if self._episode_outcomes is None:
            raise ValueError(
                "This dataset has no ground-truth episode outcomes "
                "(meta_episode_outcomes) -- re-record demonstrations with "
                "the current record_demonstrations.py to enable outcome "
                "classification."
            )
        # episode index = index of the first full-dataset episode-end row
        # >= end_row (see docstring above for why this correctly handles a
        # filtered row_pool's own last row, not just an exact full-dataset
        # match) -- NOT dones[:end_row+1].sum(), which double-counts every
        # episode when 2+ players share one timed sample and both get
        # done=1 on the same episode (the exact bug that caused an
        # IndexError past the end of self._episode_outcomes).
        end_rows = self._episode_end_rows()
        pos = int(np.searchsorted(end_rows, end_row, side="left"))
        if pos >= len(end_rows):
            raise ValueError(
                f"row {end_row} is past the last known episode boundary "
                f"(last end row: {end_rows[-1] if len(end_rows) else 'n/a'}) -- "
                f"classify_outcome() must be called with a row belonging to "
                f"a complete episode."
            )
        ep_idx = pos
        if ep_idx >= len(self._episode_outcomes):
            raise ValueError(
                f"episode {ep_idx} (row {end_row}) has no matching entry in "
                f"meta_episode_outcomes (length {len(self._episode_outcomes)}) -- "
                f"the dataset's dones/episode-boundary count disagrees with "
                f"the number of episodes actually recorded; re-record with a "
                f"consistent record_demonstrations.py version."
            )
        raw_outcome = str(self._episode_outcomes[ep_idx])
        try:
            return self._OUTCOME_LABEL_MAP[raw_outcome]
        except KeyError:
            raise ValueError(
                f"Unrecognised trial_outcome {raw_outcome!r} for episode {ep_idx} "
                f"(row {end_row}) -- Phase 1 only ever produces "
                f"{sorted(self._OUTCOME_LABEL_MAP)}. This means ScenarioEnv "
                f"produced an outcome outside its documented vocabulary; fix "
                f"the source (ai/env/outcome.py / ai/env/scenario_env.py), "
                f"do not silently bucket this as \"unknown\"."
            ) from None

    def row_outcomes(self, row_pool: np.ndarray) -> np.ndarray:
        """Return a dtype=object array, same length/order as *row_pool*,
        tagging every row with the outcome (see ``classify_outcome()``) of
        the complete episode it belongs to. Rows belonging to a trailing
        incomplete episode (no terminal ``done`` row in row_pool) are
        tagged "incomplete". ``row_pool`` must be sorted increasing (true
        for ``split_train_val_indices()``'s train/val arrays and
        ``np.arange(len(self))``)."""
        out = np.full(len(row_pool), "incomplete", dtype=object)
        for start, end in self.episode_row_ranges(row_pool):
            outcome = self.classify_outcome(end)
            lo = int(np.searchsorted(row_pool, start))
            hi = int(np.searchsorted(row_pool, end, side="right"))
            out[lo:hi] = outcome
        return out

    def outcome_by_row(self) -> np.ndarray:
        """Cached ``row_outcomes(np.arange(len(self)))`` over the WHOLE
        dataset -- callers indexing an arbitrary (possibly unsorted/
        non-contiguous) row-index chunk (e.g. a training minibatch) should
        index directly into this array rather than re-deriving outcomes
        from just their chunk (which may not span whole episodes)."""
        if self._outcome_by_row_cache is None:
            self._outcome_by_row_cache = self.row_outcomes(np.arange(self._n))
        return self._outcome_by_row_cache

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
            episode_outcomes=(
                [str(o) for o in data["meta_episode_outcomes"]]
                if "meta_episode_outcomes" in data else None
            ),
        )

    @classmethod
    def from_files(cls, paths: list[str | Path]) -> "DemonstrationDataset":
        """Load and concatenate multiple .npz files."""
        parts = [cls.from_file(p) for p in paths]
        # reward_component_keys must agree across files (fixed by REWARD_COMP_LABELS
        # order at recording time) -- use the first part's keys/columns; parts
        # recorded before this feature existed already fell back to the same
        # default key list in __init__, so concatenation stays column-aligned.
        # episode_outcomes: concatenate per-part lists in file order, matching
        # each part's dones=1 row order -- but ONLY if every part has them; a
        # mix of old (no meta_episode_outcomes) and new files would produce
        # an outcome list misaligned with the concatenated dones, which is
        # worse than just falling back to None for the whole dataset.
        _all_have_outcomes = all(p._episode_outcomes is not None for p in parts)
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
            episode_outcomes=(
                np.concatenate([p._episode_outcomes for p in parts]) if _all_have_outcomes else None
            ),
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

        # Episode boundaries resolved against the FULL dataset (see
        # episode_row_ranges()'s docstring) -- NOT re-derived from
        # base_indices' own (possibly filtered) dones, which can miss a
        # boundary entirely if valid_indices() drops one of an episode's
        # two terminal done=1 rows (e.g. the immobile opponent's own row),
        # or merge two adjacent single-sample episodes if it doesn't.
        # Splitting on a missed/merged boundary is a real train/val leakage
        # bug, not just an off-by-one count.
        ranges = self.episode_row_ranges(base_indices)
        n_complete_eps = len(ranges)
        if n_complete_eps < 2:
            return base_indices, np.array([], dtype=base_indices.dtype)

        n_val_eps = max(1, round(val_frac * n_complete_eps))
        n_train_eps = n_complete_eps - n_val_eps

        val_row_set: set[int] = set()
        for start, end in ranges[n_train_eps:]:
            lo = int(np.searchsorted(base_indices, start))
            hi = int(np.searchsorted(base_indices, end, side="right"))
            val_row_set.update(range(lo, hi))
        val_mask = np.zeros(len(base_indices), dtype=bool)
        if val_row_set:
            val_mask[np.fromiter(val_row_set, dtype=np.int64)] = True
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

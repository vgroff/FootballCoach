"""Dataset generation, I/O, and minibatch iteration for player-dynamics
pretraining.

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

Directly mirrors ``ball_dataset.py`` (agent_plans/ball_physics_pretrain_plan.md
section 4.4) -- append-by-default/randomly-seeded-by-default shard
generation, ``ProgressReporter``-driven progress, the same ``.npz`` shard
convention, including the ``crossings``/``crossing_times`` side-channel that
supervises ``PlayerDynamicsAutoencoder.crossing_head``. Unlike the ball
version, that side-channel is NOT also used to reconstruct freeze-on-event
semantics (the player pipeline has no freeze rule at all -- see
player_episode_gen.py's module docstring), so there is no
``targets_with_freeze_semantics`` equivalent here.

``crossings``/``crossing_times`` are REQUIRED fields of every ``.npz``
shard -- there is deliberately no optional/backward-compat path for shards
generated before they existed. This pipeline's established convention on a
dataset schema change is "delete and regenerate", not "tolerate both
shapes"; a silently-half-supervised head would be far more expensive to
notice than a loud ``KeyError`` on a stale shard.
"""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import secrets
from pathlib import Path
from typing import Generator

import numpy as np
import torch

from footballcoach.ai.physics_pretrain.player_episode_gen import (
    N_INPUT_FIELDS,
    N_TARGET_FIELDS_PER_HORIZON,
    PlayerEpisodeGenParams,
    compute_engineered_features,
    generate_shard,
)
from footballcoach.ai.progress import ProgressReporter

log = logging.getLogger("footballcoach.ai.physics_pretrain.player_dataset")

# (start, end) column slices within one N_TARGET_FIELDS_PER_HORIZON-wide
# identity block (fields 0:7, same layout in inputs and per-horizon
# targets), grouped by physical quantity for per-group variance/baseline/
# loss reporting -- the player-net equivalent of ball_dataset.py's
# pos/vel/spin groups (mirroring player_dynamics_net's identity-field
# layout, not the ball's).
GROUPS = {"pos": (0, 2), "vel": (2, 4), "heading": (4, 6), "stamina": (6, 7)}


def _generate_shard_worker(args: tuple[int, int, int]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local_idx, n_episodes, seed = args
    inputs, targets, crossings, crossing_times = generate_shard(n_episodes, seed, PlayerEpisodeGenParams.from_config())
    return local_idx, inputs, targets, crossings, crossing_times


def generate_dataset(
    n_episodes: int,
    output_dir: str | Path,
    seed: int | None = None,
    shard_size: int = 10_000,
    n_workers: int = 1,
) -> list[Path]:
    """Generates ``n_episodes`` total episodes, split into shards of
    ``shard_size``, written as ``.npz`` files under ``output_dir``.

    Directly mirrors ``ball_dataset.generate_dataset``'s behaviour and
    docstring: ``seed=None`` (default) draws and logs a fresh random base
    seed each call; repeated calls against the same ``output_dir`` APPEND
    (continuing shard numbering/seeding) rather than overwriting; shards are
    written to disk as soon as they finish (``pool.imap_unordered``) with
    live progress via ``ai/progress.py``'s ``ProgressReporter``.
    """
    if seed is None:
        seed = secrets.randbelow(2**31)
        log.info(f"No --seed given -- using random base seed {seed} (pass --seed {seed} to reproduce this run)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("shard_*.npz"))
    existing_indices = [int(p.stem.removeprefix("shard_")) for p in existing]
    start_idx = (max(existing_indices) + 1) if existing_indices else 0
    if existing:
        log.info(f"Found {len(existing)} existing shard(s) in {output_dir} -- appending from shard_{start_idx:05d}")

    n_shards = math.ceil(n_episodes / shard_size)
    shard_specs: list[tuple[int, int, int]] = []
    remaining = n_episodes
    for i in range(n_shards):
        n = min(shard_size, remaining)
        shard_specs.append((i, n, seed + start_idx + i))
        remaining -= n

    log.info(f"Generating {n_episodes:,} player-dynamics episodes across {n_shards} shard(s), n_workers={n_workers}")

    progress = ProgressReporter(total=n_episodes, prefix="  player episodes: ", live=True)
    paths_by_idx: dict[int, Path] = {}
    completed_episodes = 0

    def _write_shard(
        local_idx: int, inputs: np.ndarray, targets: np.ndarray, crossings: np.ndarray, crossing_times: np.ndarray,
    ) -> None:
        nonlocal completed_episodes
        path = output_dir / f"shard_{start_idx + local_idx:05d}.npz"
        np.savez_compressed(path, inputs=inputs, targets=targets, crossings=crossings, crossing_times=crossing_times)
        paths_by_idx[local_idx] = path
        completed_episodes += len(inputs)
        progress.update(completed_episodes)

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            for local_idx, inputs, targets, crossings, crossing_times in pool.imap_unordered(_generate_shard_worker, shard_specs):
                _write_shard(local_idx, inputs, targets, crossings, crossing_times)
    else:
        for spec in shard_specs:
            local_idx, inputs, targets, crossings, crossing_times = _generate_shard_worker(spec)
            _write_shard(local_idx, inputs, targets, crossings, crossing_times)

    progress.finish(completed_episodes)
    paths = [paths_by_idx[i] for i in range(n_shards)]
    for path in paths:
        log.info(f"  wrote {path}")
    return paths


class PlayerDynamicsDataset:
    """In-memory dataset loaded from one or more ``.npz`` shards.

    ``inputs``: ``(N, 24)`` float32, ``targets``: ``(N, n_horizons*9)``
    float32 -- see player_episode_gen.py's ``N_INPUT_FIELDS``/``N_TARGET_
    FIELDS_PER_HORIZON``. Directly mirrors ``BallDynamicsDataset``, minus
    ``targets_with_freeze_semantics`` (no freeze-on-event rule here -- see
    the module docstring).

    ``crossings``/``crossing_times``: ``(N, 9)``/``(N,)``, the full state at
    the moment out_of_bounds/goal_scored FIRST became true and when -- always
    present on a dataset loaded via ``from_directory`` (a required ``.npz``
    field), and optional here only so callers that build a dataset BY HAND
    (e.g. a unit test that doesn't care about the crossing head) can pass
    ``None`` -- same convention as ``BallDynamicsDataset``.

    ``crossing_pos``/``crossing_dt``/``crossing_mask`` (derived from the two
    above, ``None`` iff those are ``None``): the ready-to-train targets for
    ``PlayerDynamicsAutoencoder.crossing_head``. ``crossing_pos`` is
    ``crossings[:, 0:2]`` (the normalized pos_x/pos_y at the crossing -- a
    player has no height axis, so unlike the ball's there's nothing to
    exclude here); ``crossing_mask`` is ``True`` where ``crossing_times`` is
    finite (this episode actually crossed within the simulated window AND
    wasn't excluded for starting already out of bounds -- see
    ``player_episode_gen.generate_episode``'s docstring) and ``False`` where
    it's ``inf``; ``crossing_dt`` is ``crossing_times`` with those ``inf``
    entries replaced by ``-1.0``. Callers mask the POSITION loss by
    ``crossing_mask`` (a never-crossed/excluded episode has no meaningful
    crossing position to regress toward) but train the DELTA_T loss unmasked
    against the ``-1`` sentinel, so the head itself learns to predict "no
    crossing" rather than relying on a separate classifier -- identical
    convention to ``BallDynamicsDataset``'s.
    """

    def __init__(
        self, inputs: np.ndarray, targets: np.ndarray,
        crossings: np.ndarray | None = None, crossing_times: np.ndarray | None = None,
    ):
        if len(inputs) != len(targets):
            raise ValueError(f"inputs/targets length mismatch: {len(inputs)} vs {len(targets)}")
        if crossings is not None and len(crossings) != len(inputs):
            raise ValueError(f"inputs/crossings length mismatch: {len(inputs)} vs {len(crossings)}")
        if crossing_times is not None and len(crossing_times) != len(inputs):
            raise ValueError(f"inputs/crossing_times length mismatch: {len(inputs)} vs {len(crossing_times)}")
        self.inputs = inputs
        self.targets = targets
        self.crossings = crossings
        self.crossing_times = crossing_times
        if crossings is not None and crossing_times is not None:
            self.crossing_pos = crossings[:, 0:2].astype(np.float32, copy=False)
            self.crossing_mask = np.isfinite(crossing_times)
            self.crossing_dt = np.where(self.crossing_mask, crossing_times, -1.0).astype(np.float32)
        else:
            self.crossing_pos = None
            self.crossing_mask = None
            self.crossing_dt = None

    def __len__(self) -> int:
        return len(self.inputs)

    @classmethod
    def from_directory(cls, directory: str | Path, pattern: str = "*.npz") -> "PlayerDynamicsDataset":
        paths = sorted(Path(directory).glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No .npz shards found in {directory}")
        inputs_parts, targets_parts, crossings_parts, crossing_times_parts = [], [], [], []
        for p in paths:
            data = np.load(p)
            inputs_parts.append(data["inputs"])
            targets_parts.append(data["targets"])
            # Required fields -- a stale shard predating them raises KeyError
            # here on purpose. See the module docstring.
            crossings_parts.append(data["crossings"])
            crossing_times_parts.append(data["crossing_times"])
        log.info(f"Loaded {len(paths)} shard(s) from {directory}")
        return cls(
            np.concatenate(inputs_parts), np.concatenate(targets_parts),
            np.concatenate(crossings_parts), np.concatenate(crossing_times_parts),
        )

    def split_train_val(self, val_frac: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Random train/val row-index split -- see
        ``BallDynamicsDataset.split_train_val``'s docstring (same "episode-
        level" naming convention degenerating to a plain row-level split,
        since each row already IS one complete episode)."""
        n = len(self)
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        if val_frac <= 0.0 or n < 2:
            return idx, np.array([], dtype=idx.dtype)
        n_val = max(1, int(round(n * val_frac)))
        return idx[n_val:], idx[:n_val]

    def compute_pos_weights(
        self, n_horizons: int, indices: np.ndarray | None = None, max_weight: float | None = None,
    ) -> np.ndarray:
        """Inverse-frequency ``pos_weight`` for the two event-flag columns
        (``out_of_bounds``, ``goal_scored``) of each horizon -- see
        ``BallDynamicsDataset.compute_pos_weights``'s docstring, identical
        semantics/convention. Returns an ``(n_horizons, 2)`` float32 array.

        ``goal_scored``'s weight is computed ONLY over ``has_possession=1``
        rows (input column 11): a non-possessing row's ``goal_scored`` is
        always a physically meaningless 0 (see player_episode_gen.
        generate_episode's docstring), so including those rows here would
        inflate the inverse-frequency weight past what the meaningful
        subset actually needs -- consistent with train_player_dynamics.py's
        loss/confusion-count possession-gating.
        """
        idx = indices if indices is not None else np.arange(len(self))
        targets = self.targets[idx]
        has_possession = self.inputs[idx, 11] > 0.5
        out = np.ones((n_horizons, 2), dtype=np.float32)
        for h in range(n_horizons):
            base = h * N_TARGET_FIELDS_PER_HORIZON
            oob_labels = targets[:, base + 7]
            goal_labels = targets[has_possession, base + 8]
            for j, labels in enumerate((oob_labels, goal_labels)):
                n_pos = float((labels > 0.5).sum())
                n_neg = float(len(labels)) - n_pos
                weight = n_neg / max(n_pos, 1.0)
                if max_weight is not None:
                    weight = min(weight, max_weight)
                out[h, j] = weight
        return out

    def compute_group_variance(
        self, n_horizons: int, indices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Per-horizon target variance for the four continuous quantity
        groups (``pos``, ``vel``, ``heading``, ``stamina`` -- see
        ``GROUPS``), used as the "predict the mean" R^2 baseline -- see
        ``BallDynamicsDataset.compute_group_variance``'s docstring, same
        mean-of-per-field-variances convention."""
        idx = indices if indices is not None else np.arange(len(self))
        targets = self.targets[idx]
        out = {g: np.zeros(n_horizons, dtype=np.float64) for g in GROUPS}
        for h in range(n_horizons):
            base = h * N_TARGET_FIELDS_PER_HORIZON
            for g, (lo, hi) in GROUPS.items():
                cols = targets[:, base + lo:base + hi]
                out[g][h] = cols.var(axis=0).mean()
        return out

    def compute_persistence_baseline_mse(
        self, n_horizons: int, indices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Per-horizon MSE of the "persistence" baseline (predict the
        group's own initial value, unchanged) -- see
        ``BallDynamicsDataset.compute_persistence_baseline_mse``'s
        docstring, same rationale (a much stronger baseline than the
        train-mean at short horizons; doubles as an estimate of how much
        the group actually tends to change over that horizon)."""
        idx = indices if indices is not None else np.arange(len(self))
        inputs = self.inputs[idx]
        targets = self.targets[idx]
        out = {g: np.zeros(n_horizons, dtype=np.float64) for g in GROUPS}
        for h in range(n_horizons):
            base = h * N_TARGET_FIELDS_PER_HORIZON
            for g, (lo, hi) in GROUPS.items():
                diff = targets[:, base + lo:base + hi] - inputs[:, lo:hi]
                out[g][h] = (diff ** 2).mean()
        return out

    def build_adjacent_pair_data(
        self, pair_idx: int, indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Derives extra (input, target) training pairs from data already
        fully recorded: pretend horizon ``pair_idx``'s recorded state IS the
        initial state, and predict horizon ``pair_idx+1``'s recorded state
        from it -- see ``BallDynamicsDataset.build_adjacent_pair_data``'s
        docstring for the full motivation (exposes the encoder to more
        realistic "mid-trajectory" input states).

        UNLIKE the ball version, there is no "already resolved, trivial to
        predict" filter here -- the player pipeline has no freeze-on-event
        rule (a player who's out of bounds keeps moving under normal
        physics, out_of_bounds/goal_scored can legitimately change again at
        the next horizon), so every row is an equally legitimate training
        pair, not just a subset.

        The non-identity context fields (attributes, has_possession,
        desired_direction, speed_mode, pitch/goal dims -- input columns
        7:21) are copied UNCHANGED from the ORIGINAL episode's t=0 input
        row. This is now EXACT, not an approximation: intent
        (desired_direction/speed_mode) is drawn once per episode and held
        FIXED for its entire duration (see player_episode_gen.
        generate_episode's docstring), so whatever it was at t=0 is still
        exactly what it is at horizon ``pair_idx`` too -- there is nothing
        to lose by copying it forward.
        """
        idx = indices if indices is not None else np.arange(len(self))
        base_i = pair_idx * N_TARGET_FIELDS_PER_HORIZON
        base_j = (pair_idx + 1) * N_TARGET_FIELDS_PER_HORIZON
        block_i = self.targets[idx, base_i:base_i + N_TARGET_FIELDS_PER_HORIZON]
        block_j = self.targets[idx, base_j:base_j + N_TARGET_FIELDS_PER_HORIZON]
        orig_inputs = self.inputs[idx]

        derived_inputs = np.zeros((len(idx), N_INPUT_FIELDS), dtype=np.float32)
        derived_inputs[:, 0:7] = block_i[:, 0:7]
        derived_inputs[:, 7:21] = orig_inputs[:, 7:21]
        derived_inputs[:, 21:24] = compute_engineered_features(block_i[:, 0:7], orig_inputs[:, 12:14])
        return derived_inputs, block_j.astype(np.float32, copy=False)

    def build_autoencoding_data(
        self, horizon_idx: int, indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Derives (input, target) pairs for pure autoencoding: treat
        horizon ``horizon_idx``'s recorded state as a pseudo-initial-state
        and ask the decoder to reconstruct THAT SAME state (queried at t=0
        via ``forward_at()``) -- see
        ``BallDynamicsDataset.build_autoencoding_data``'s docstring, same
        "no resolved-state filter needed" rationale (doubly true here since
        there's no freeze rule at all)."""
        idx = indices if indices is not None else np.arange(len(self))
        base = horizon_idx * N_TARGET_FIELDS_PER_HORIZON
        block = self.targets[idx, base:base + N_TARGET_FIELDS_PER_HORIZON]
        orig_inputs = self.inputs[idx]

        derived_inputs = np.zeros((len(idx), N_INPUT_FIELDS), dtype=np.float32)
        derived_inputs[:, 0:7] = block[:, 0:7]
        derived_inputs[:, 7:21] = orig_inputs[:, 7:21]
        derived_inputs[:, 21:24] = compute_engineered_features(block[:, 0:7], orig_inputs[:, 12:14])
        return derived_inputs, block.astype(np.float32, copy=False)

    def iterate_minibatches(
        self,
        batch_size: int,
        indices: np.ndarray,
        shuffle: bool = True,
        device: torch.device | None = None,
        rng: np.random.Generator | None = None,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        idx = indices.copy()
        if shuffle:
            (rng if rng is not None else np.random.default_rng()).shuffle(idx)
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start + batch_size]
            if len(batch_idx) == 0:
                continue
            x = torch.from_numpy(self.inputs[batch_idx].astype(np.float32, copy=False))
            y = torch.from_numpy(self.targets[batch_idx].astype(np.float32, copy=False))
            if device is not None:
                x, y = x.to(device), y.to(device)
            yield x, y


def _main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from footballcoach.ai.config import load_ai_config
    cfg = load_ai_config()["physics_pretrain"]["player"]

    parser = argparse.ArgumentParser(description="Generate the player-dynamics pretraining dataset.")
    parser.add_argument("--output", required=True, help="Output directory for .npz shards.")
    parser.add_argument("--n-episodes", type=int, default=cfg["n_episodes"])
    parser.add_argument("--shard-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=None, help="Omit for a fresh random seed each run (recommended).")
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    generate_dataset(
        n_episodes=args.n_episodes,
        output_dir=args.output,
        seed=args.seed,
        shard_size=args.shard_size,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    _main()

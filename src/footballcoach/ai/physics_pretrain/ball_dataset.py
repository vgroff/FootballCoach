"""Dataset generation, I/O, and minibatch iteration for ball-dynamics
pretraining.

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

See agent_plans/ball_physics_pretrain_plan.md section 4.4. Deliberately a
much simpler class than ``ai/bc/dataset.py``'s ``DemonstrationDataset`` --
there is no episode/BC-label/reward structure to carry here, each row IS one
complete episode already (one random draw, simulated once, §4.2), so no
per-episode boundary bookkeeping is needed either.
"""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
from pathlib import Path
from typing import Generator

import numpy as np
import torch

from footballcoach.ai.physics_pretrain.ball_episode_gen import (
    N_TARGET_FIELDS_PER_HORIZON,
    BallEpisodeGenParams,
    generate_shard,
)

log = logging.getLogger("footballcoach.ai.physics_pretrain.ball_dataset")


def _generate_shard_worker(args: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    n_episodes, seed = args
    # Re-read config fresh in each worker process (mirrors rollout_worker.py's
    # pattern of not pickling config across the process boundary).
    return generate_shard(n_episodes, seed, BallEpisodeGenParams.from_config())


def generate_dataset(
    n_episodes: int,
    output_dir: str | Path,
    seed: int = 0,
    shard_size: int = 10_000,
    n_workers: int = 1,
) -> list[Path]:
    """Generates ``n_episodes`` total episodes, split into shards of
    ``shard_size``, written as ``.npz`` files under ``output_dir``.

    Parallelized across ``n_workers`` plain OS processes (``multiprocessing.
    Pool``) -- unlike ``ppo/rollout_worker.py``'s persistent Pipe-based
    workers, episodes are fully independent single-shot draws with no shared
    state to synchronize, so a simple ``Pool.map`` over shard specs is
    sufficient (see the plan's §4.4: "parallelize the same way
    rollout_worker.py already does ... rather than inventing a new
    parallelism pattern" -- the shared idea being "plain multiprocessing,
    each worker owns an independent chunk of work", not the Pipe protocol
    itself, which exists there to handle long-lived per-tick RL rollouts).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_shards = math.ceil(n_episodes / shard_size)
    shard_specs: list[tuple[int, int]] = []
    remaining = n_episodes
    for i in range(n_shards):
        n = min(shard_size, remaining)
        shard_specs.append((n, seed + i))
        remaining -= n

    log.info(f"Generating {n_episodes:,} ball-dynamics episodes across {n_shards} shard(s), n_workers={n_workers}")

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            results = pool.map(_generate_shard_worker, shard_specs)
    else:
        results = [_generate_shard_worker(spec) for spec in shard_specs]

    paths: list[Path] = []
    for i, (inputs, targets) in enumerate(results):
        path = output_dir / f"shard_{i:05d}.npz"
        np.savez_compressed(path, inputs=inputs, targets=targets)
        paths.append(path)
        log.info(f"  wrote {path} ({len(inputs):,} episodes)")
    return paths


class BallDynamicsDataset:
    """In-memory dataset loaded from one or more ``.npz`` shards.

    ``inputs``: ``(N, 14)`` float32, ``targets``: ``(N, 5*11)`` float32 --
    see ball_episode_gen.py's ``N_INPUT_FIELDS``/``N_TARGET_FIELDS_PER_HORIZON``.
    """

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        if len(inputs) != len(targets):
            raise ValueError(f"inputs/targets length mismatch: {len(inputs)} vs {len(targets)}")
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    @classmethod
    def from_directory(cls, directory: str | Path, pattern: str = "*.npz") -> "BallDynamicsDataset":
        paths = sorted(Path(directory).glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No .npz shards found in {directory}")
        inputs_parts, targets_parts = [], []
        for p in paths:
            data = np.load(p)
            inputs_parts.append(data["inputs"])
            targets_parts.append(data["targets"])
        log.info(f"Loaded {len(paths)} shard(s) from {directory}")
        return cls(np.concatenate(inputs_parts), np.concatenate(targets_parts))

    def split_train_val(self, val_frac: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Random train/val row-index split.

        Named to match ``DemonstrationDataset.split_train_val_indices()``'s
        "episode-level" convention even though here it degenerates to a
        plain row-level split -- each row already IS one complete episode
        (§4.4), unlike the BC dataset's many-rows-per-episode structure, so
        there is no episode-boundary bookkeeping to do.
        """
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
        (``out_of_bounds``, ``goal_scored``) of each horizon, matching
        ``DemonstrationDataset.compute_pos_weights()``'s
        ``n_negative / max(n_positive, 1)`` convention (same semantics as
        ``binary_cross_entropy_with_logits``'s ``pos_weight`` arg).

        Returns an ``(n_horizons, 2)`` float32 array: column 0 =
        ``out_of_bounds`` weight, column 1 = ``goal_scored`` weight.
        """
        idx = indices if indices is not None else np.arange(len(self))
        targets = self.targets[idx]
        out = np.ones((n_horizons, 2), dtype=np.float32)
        for h in range(n_horizons):
            base = h * N_TARGET_FIELDS_PER_HORIZON
            for j, col in enumerate((base + 9, base + 10)):
                labels = targets[:, col]
                n_pos = float((labels > 0.5).sum())
                n_neg = float(len(labels)) - n_pos
                weight = n_neg / max(n_pos, 1.0)
                if max_weight is not None:
                    weight = min(weight, max_weight)
                out[h, j] = weight
        return out

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
    cfg = load_ai_config()["physics_pretrain"]["ball"]

    parser = argparse.ArgumentParser(description="Generate the ball-dynamics pretraining dataset (§4).")
    parser.add_argument("--output", required=True, help="Output directory for .npz shards.")
    parser.add_argument("--n-episodes", type=int, default=cfg["n_episodes"])
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
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

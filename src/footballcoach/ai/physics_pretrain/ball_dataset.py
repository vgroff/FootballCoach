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
import secrets
from pathlib import Path
from typing import Generator

import numpy as np
import torch

from footballcoach.ai.physics_pretrain.ball_episode_gen import (
    N_INPUT_FIELDS,
    N_TARGET_FIELDS_PER_HORIZON,
    BallEpisodeGenParams,
    compute_engineered_features,
    generate_shard,
)
from footballcoach.ai.progress import ProgressReporter

log = logging.getLogger("footballcoach.ai.physics_pretrain.ball_dataset")


def _generate_shard_worker(args: tuple[int, int, int]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local_idx, n_episodes, seed = args
    # Re-read config fresh in each worker process (mirrors rollout_worker.py's
    # pattern of not pickling config across the process boundary).
    inputs, targets, crossings, crossing_times = generate_shard(n_episodes, seed, BallEpisodeGenParams.from_config())
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

    ``seed``: ``None`` (default) draws a fresh random base seed each call
    (logged, so a specific run can still be reproduced after the fact by
    passing that value back in) -- so simply omitting ``--seed`` on repeated
    calls against the same ``output_dir`` gives genuinely new episodes each
    time, not a silently-deterministic repeat. Pass an explicit int only
    when you want reproducibility (e.g. a regression test).

    Parallelized across ``n_workers`` plain OS processes (``multiprocessing.
    Pool``) -- unlike ``ppo/rollout_worker.py``'s persistent Pipe-based
    workers, episodes are fully independent single-shot draws with no shared
    state to synchronize, so a simple ``Pool.map`` over shard specs is
    sufficient (see the plan's §4.4: "parallelize the same way
    rollout_worker.py already does ... rather than inventing a new
    parallelism pattern" -- the shared idea being "plain multiprocessing,
    each worker owns an independent chunk of work", not the Pipe protocol
    itself, which exists there to handle long-lived per-tick RL rollouts).

    ADDS to any existing shards already in ``output_dir`` rather than
    overwriting them -- new shard files continue numbering (and seeding)
    from whatever's already there, so calling this repeatedly against the
    same directory grows the dataset instead of clobbering it. To start
    fresh, delete ``output_dir`` yourself first.
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
        # Seed keyed off the GLOBAL shard index (not the local loop index),
        # so repeated calls against the same output_dir keep drawing fresh
        # episodes instead of re-generating (and overwriting) the same ones.
        shard_specs.append((i, n, seed + start_idx + i))
        remaining -= n

    log.info(f"Generating {n_episodes:,} ball-dynamics episodes across {n_shards} shard(s), n_workers={n_workers}")

    # Write each shard to disk as soon as IT finishes (not after every shard
    # is done) and report progress incrementally -- generating the full
    # target (e.g. 200k episodes) can take minutes, and previously nothing
    # printed until the very last shard completed.
    progress = ProgressReporter(total=n_episodes, prefix="  ball episodes: ", live=True)
    paths_by_idx: dict[int, Path] = {}
    completed_episodes = 0

    def _write_shard(local_idx: int, inputs: np.ndarray, targets: np.ndarray, crossings: np.ndarray, crossing_times: np.ndarray) -> None:
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


class BallDynamicsDataset:
    """In-memory dataset loaded from one or more ``.npz`` shards.

    ``inputs``: ``(N, 20)`` float32, ``targets``: ``(N, 5*11)`` float32 --
    see ball_episode_gen.py's ``N_INPUT_FIELDS``/``N_TARGET_FIELDS_PER_HORIZON``.

    ``crossings``/``crossing_times`` (optional -- ``None`` for callers that
    don't need them, e.g. tests constructing a dataset by hand): ``(N, 11)``/
    ``(N,)``, the full state at the moment out_of_bounds/goal_scored FIRST
    became true and when, recorded alongside the always-continuous
    ``targets`` -- see ``ball_episode_gen.generate_episode``'s docstring and
    ``targets_with_freeze_semantics`` below.

    ``crossing_pos``/``crossing_dt``/``crossing_mask`` (derived from
    ``crossings``/``crossing_times`` above, ``None`` iff those are ``None``):
    the ready-to-train targets for the crossing-prediction head (see
    ``BallDynamicsAutoencoder.crossing_head``) -- ``crossing_pos`` is just
    ``crossings[:, 0:2]`` (the normalized pos_x/pos_y at the crossing --
    height is deliberately excluded, the head only predicts the 2D pitch
    location);
    ``crossing_mask`` is ``True`` where ``crossing_times`` is finite (episode
    actually went oob/scored within the simulated window) and ``False``
    where it's ``inf`` (never happened); ``crossing_dt`` is ``crossing_times``
    with those ``inf`` entries replaced by ``-1.0`` (a plain float array,
    since ``inf`` isn't representable as a well-behaved regression target and
    a network can't be trained to predict it) -- callers mask the POSITION
    loss by ``crossing_mask`` (a never-happened episode has no meaningful
    crossing position) but train the DELTA_T loss unmasked against the ``-1``
    sentinel, so the head itself learns to predict "no crossing" rather than
    relying on a separately-trained classifier.
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
    def from_directory(cls, directory: str | Path, pattern: str = "*.npz") -> "BallDynamicsDataset":
        paths = sorted(Path(directory).glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No .npz shards found in {directory}")
        inputs_parts, targets_parts, crossings_parts, crossing_times_parts = [], [], [], []
        for p in paths:
            data = np.load(p)
            inputs_parts.append(data["inputs"])
            targets_parts.append(data["targets"])
            crossings_parts.append(data["crossings"])
            crossing_times_parts.append(data["crossing_times"])
        log.info(f"Loaded {len(paths)} shard(s) from {directory}")
        return cls(
            np.concatenate(inputs_parts), np.concatenate(targets_parts),
            np.concatenate(crossings_parts), np.concatenate(crossing_times_parts),
        )

    def targets_with_freeze_semantics(self, horizons_s: list[float]) -> np.ndarray:
        """Reconstructs the OLD freeze-on-event semantics (state frozen at
        the first out_of_bounds/goal_scored crossing, held for every later
        horizon, both flags latched true from that point on) from this
        dataset's always-continuous ``targets`` plus the per-episode
        ``crossings``/``crossing_times`` recorded alongside it. Returns a
        NEW array -- does NOT mutate ``self.targets`` -- so callers can
        choose which convention to train against without needing two
        separately-generated datasets (see ``physics_pretrain.ball.
        freeze_semantics`` in ai_config.json).

        For each horizon whose nominal time is at or after that episode's
        ``crossing_time``, substitutes the full ``crossings`` row (state +
        flags, all latched from that point per the old semantics) in place
        of the continuously-simulated one. Horizons before the crossing are
        left untouched (their own, already-0, flags and in-flight state).
        """
        if self.crossings is None or self.crossing_times is None:
            raise ValueError("This dataset has no crossings/crossing_times (built without them) -- can't reconstruct freeze semantics.")
        out = self.targets.copy()
        for h, t in enumerate(horizons_s):
            base = h * N_TARGET_FIELDS_PER_HORIZON
            mask = self.crossing_times <= t + 1e-9
            out[mask, base:base + N_TARGET_FIELDS_PER_HORIZON] = self.crossings[mask]
        return out

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

    def compute_group_variance(
        self, n_horizons: int, indices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Per-horizon target variance for the three continuous quantity
        groups (``pos``: pos_x/pos_y/height_m, ``vel``, ``spin``), used as
        the "predict the mean" baseline for R^2 sanity-checking
        ``train_ball_dynamics.py``'s MSE (see ``compute_group_sq_err``
        there). Each group's variance is the MEAN of its 3 fields' own
        variances (not the pooled variance of all 3 concatenated), matching
        how ``F.mse_loss(..., reduction='mean')`` already averages equally
        over fields when computing the group's actual MSE -- so ``1 -
        mse/this`` is a like-for-like comparison.
        """
        idx = indices if indices is not None else np.arange(len(self))
        targets = self.targets[idx]
        groups = {"pos": (0, 3), "vel": (3, 6), "spin": (6, 9)}
        out = {g: np.zeros(n_horizons, dtype=np.float64) for g in groups}
        for h in range(n_horizons):
            base = h * N_TARGET_FIELDS_PER_HORIZON
            for g, (lo, hi) in groups.items():
                cols = targets[:, base + lo:base + hi]
                out[g][h] = cols.var(axis=0).mean()
        return out

    def compute_persistence_baseline_mse(
        self, n_horizons: int, indices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Per-horizon MSE of the "persistence" baseline -- predicting that
        the group's value at this horizon equals its OWN initial value
        (``self.inputs``'s fields 0-8, same column ranges/units as the
        target's fields 0-8 for that group, since both are normalized by
        the same episode's own pitch scale). A much stronger baseline than
        ``compute_group_variance``'s "predict the train-set mean": at a
        short horizon the ball genuinely hasn't moved much, so persistence
        is a tough baseline to beat, and beating it means the model learned
        real short-term dynamics rather than just echoing the input. This
        MSE also doubles as an estimate of "how much did this group
        actually change over the horizon" -- 1 - model_mse/this
        automatically normalizes error by the horizon's typical
        displacement (1m off is a big miss at t=0.2s where the ball barely
        moved, a small one at t=8s where it's moved much further).
        """
        idx = indices if indices is not None else np.arange(len(self))
        inputs = self.inputs[idx]
        targets = self.targets[idx]
        groups = {"pos": (0, 3), "vel": (3, 6), "spin": (6, 9)}
        out = {g: np.zeros(n_horizons, dtype=np.float64) for g in groups}
        for h in range(n_horizons):
            base = h * N_TARGET_FIELDS_PER_HORIZON
            for g, (lo, hi) in groups.items():
                diff = targets[:, base + lo:base + hi] - inputs[:, lo:hi]
                out[g][h] = (diff ** 2).mean()
        return out

    def compute_ballistic_baseline_mse(
        self, horizons_s: list[float], gen_params: BallEpisodeGenParams, gravity_mps2: float,
        indices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Per-horizon MSE of a "straight-line physics" baseline: extrapolate
        the initial position/velocity forward using ONLY constant velocity
        and gravity -- no drag, no Magnus/spin effects, no bounce.
        ``pos(t) = pos0 + v0*t + 0.5*g*t^2`` (z only), ``vel(t) = v0 + g*t``
        (z only), ``spin(t) = spin0`` (unchanged -- nothing in this
        baseline acts on spin, so it's identical to
        ``compute_persistence_baseline_mse``'s spin baseline).

        x/y run the full straight-line extrapolation for the whole
        horizon, unclamped -- no pitch-boundary freeze. z IS clamped, but
        only as a simple floor: ``z_pred = max(0, pos0.z + vz*t -
        0.5*g*t^2)``, computed independently at each horizon from the full
        (unclamped) ``t`` -- not a "freeze forever after first ground
        contact" state carried between horizons, just "the ball can't be
        BELOW the ground at this instant" applied to each horizon's own
        prediction. So a trajectory that would arc back above ``z=0``
        later (e.g. thrown up again) is still allowed to be positive at a
        later horizon even if an earlier horizon clamped it to 0.

        A much stronger baseline than persistence for pos/vel specifically:
        position and velocity are handed to the encoder directly as raw
        inputs, and this is close to exact for short horizons (the
        drag/Magnus/bounce corrections this baseline ignores are small over
        e.g. 0.2s) -- so beating this baseline substantially, especially at
        short horizons, is a much more meaningful bar to clear than beating
        "assume nothing moved."

        Does the extrapolation in REAL units and converts back to the
        normalized units ``self.inputs``/``self.targets`` are stored in,
        since the math (velocity*time, 0.5*g*t^2) isn't scale-invariant
        the way persistence's "just compare the same normalized column"
        trick is. Recovers real units using either each episode's OWN
        randomized pitch scale (input fields 10-11, same as
        ``describe_input_row``) or the fixed base pitch dims, matching
        whichever ``gen_params.normalize_kinematics_by_base_pitch``
        convention the data was actually encoded with.
        """
        idx = indices if indices is not None else np.arange(len(self))
        inputs = self.inputs[idx]
        targets = self.targets[idx]

        if gen_params.normalize_kinematics_by_base_pitch:
            base_half_length = gen_params.base_pitch_length_m / 2
            base_half_width = gen_params.base_pitch_width_m / 2
            half_diag = np.hypot(base_half_length, base_half_width)
            # x/y/height all share ONE divisor (half_diag) under this
            # convention -- see ball_episode_gen.py's _kinematics_divisors.
            half_length = half_width = height_norm_m = half_diag
        else:
            half_length = inputs[:, 10] * gen_params.base_pitch_length_m / 2
            half_width = inputs[:, 11] * gen_params.base_pitch_width_m / 2
            half_diag = np.hypot(half_length, half_width)
            height_norm_m = gen_params.height_norm_m

        pos0 = np.stack([
            inputs[:, 0] * half_length,
            inputs[:, 1] * half_width,
            inputs[:, 2] * height_norm_m,
        ], axis=1)
        vel0 = np.stack([
            inputs[:, 3] * half_diag,
            inputs[:, 4] * half_diag,
            inputs[:, 5] * half_diag,
        ], axis=1)

        out = {g: np.zeros(len(horizons_s), dtype=np.float64) for g in ("pos", "vel", "spin")}
        for h, t in enumerate(horizons_s):
            pos_pred = pos0 + vel0 * t
            pos_pred[:, 2] -= 0.5 * gravity_mps2 * t ** 2
            pos_pred[:, 2] = np.maximum(pos_pred[:, 2], 0.0)
            vel_pred = vel0.copy()
            vel_pred[:, 2] -= gravity_mps2 * t

            pos_pred_norm = np.stack([
                pos_pred[:, 0] / half_length,
                pos_pred[:, 1] / half_width,
                pos_pred[:, 2] / height_norm_m,
            ], axis=1)
            vel_pred_norm = np.stack([
                vel_pred[:, 0] / half_diag,
                vel_pred[:, 1] / half_diag,
                vel_pred[:, 2] / half_diag,
            ], axis=1)

            base = h * N_TARGET_FIELDS_PER_HORIZON
            out["pos"][h] = ((targets[:, base:base + 3] - pos_pred_norm) ** 2).mean()
            out["vel"][h] = ((targets[:, base + 3:base + 6] - vel_pred_norm) ** 2).mean()
            out["spin"][h] = ((targets[:, base + 6:base + 9] - inputs[:, 6:9]) ** 2).mean()
        return out

    def build_adjacent_pair_data(
        self, pair_idx: int, max_skip: int = 1, indices: np.ndarray | None = None,
        min_start_speed_norm: float = 0.0,
    ) -> tuple[np.ndarray, list[tuple[int, np.ndarray]], np.ndarray]:
        """Derives extra (input, target) training pairs from data already
        fully recorded, no new physics simulation needed: pretend horizon
        ``pair_idx``'s recorded state IS the initial state, and predict
        horizon(s) ``pair_idx+skip`` (for ``skip`` in ``1..max_skip``,
        clipped to however many horizons actually exist past ``pair_idx``)
        from it -- delta = the fixed, statically-known gap between those two
        horizons, same for every row, so a caller can decode each skip with
        its own ``forward_at()`` call (see agent_plans/ball_physics_
        pretrain_plan.md §12.6).

        ``max_skip`` (default 1, "adjacent" in the strict sense the name
        implies): how many horizons AHEAD of ``pair_idx`` to also predict --
        1 predicts only the immediately-next recorded horizon; 2 additionally
        predicts one horizon further ahead; etc. See
        ``physics_pretrain.ball.adjacent_pair_max_skip``'s config comment
        for the reasoning (short-but-not-strictly-adjacent hops from the
        same mid-trajectory pseudo-starts, without the full combinatorial
        cost of ALL ``(i, j)`` pairs).

        Crucially, the pseudo-input (and its validity) only depends on
        ``pair_idx``'s OWN recorded state -- not on which horizon it's being
        asked to predict -- so it's built exactly ONCE here regardless of
        ``max_skip``, and returned alongside a target for EVERY skip sharing
        that same input. This lets a caller encode that input through the
        encoder just ONCE and decode it at every skip's delta via multiple
        ``forward_at()`` calls, summing their losses into a single gradient
        step -- the same "one encoder pass, several decoder queries, summed
        loss" pattern the main per-horizon task already uses across its own
        horizons (see train_ball_dynamics.py's "pair" branch), rather than
        wastefully re-encoding the identical input once per skip.

        Motivation beyond "more data": the encoder currently only ever
        learns to represent states drawn from the original uniform-random
        t=0 sampling. A live-deployed encoder (§8) would see whatever state
        the ball happens to be in mid-match, which looks much more like "a
        state partway through a trajectory" than "freshly uniform-random
        sampled" -- training on recorded mid-trajectory states as
        pseudo-initial-states exposes the encoder to a more realistic input
        distribution, not just a larger one.

        Rows where the episode was ALREADY resolved (out_of_bounds OR
        goal_scored) by horizon ``pair_idx`` are EXCLUDED -- predicting
        "stays frozen" from an already-frozen state is trivially correct
        and would dilute the more informative signal with easy examples
        (same rationale as ``bc.downsample_trivial`` elsewhere in this
        codebase).

        ``min_start_speed_norm`` (default 0.0, no filtering): additionally
        excludes rows where the ball's speed (normalized units, i.e.
        ``||vel_x, vel_y, vel_z||`` from ``targets``' own normalization, NOT
        real m/s) at horizon ``pair_idx`` is below this threshold -- a
        near-stationary ball (already resting/rolling to a near-stop) makes
        for a similarly near-trivial "predict approximately no further
        movement" pair example, same rationale as the resolved-state
        exclusion above. 0.0 keeps every row (matches prior behaviour
        exactly). Callers expose this in real m/s via
        ``physics_pretrain.ball.adjacent_pair_min_start_speed_mps``,
        converting to normalized units themselves (this method stays
        physics-params-agnostic, same as everywhere else in this class).

        Returns ``(derived_inputs, targets_by_skip, row_indices)`` --
        ``derived_inputs``/``row_indices`` cover only the valid (not-yet-
        resolved, not-too-slow) subset of ``indices``; ``targets_by_skip``
        is a list of ``(skip, derived_targets)`` pairs, one per skip that
        had a horizon available to predict, each ``derived_targets`` aligned
        row-for-row with ``derived_inputs``/``row_indices``.
        """
        idx = indices if indices is not None else np.arange(len(self))
        base_i = pair_idx * N_TARGET_FIELDS_PER_HORIZON
        block_i = self.targets[idx, base_i:base_i + N_TARGET_FIELDS_PER_HORIZON]

        valid = (block_i[:, 9] < 0.5) & (block_i[:, 10] < 0.5)
        if min_start_speed_norm > 0.0:
            speed_norm = np.linalg.norm(block_i[:, 3:6], axis=1)
            valid = valid & (speed_norm >= min_start_speed_norm)
        valid_idx = idx[valid]
        block_i = block_i[valid]
        orig_inputs = self.inputs[valid_idx]

        derived_inputs = np.zeros((len(valid_idx), N_INPUT_FIELDS), dtype=np.float32)
        derived_inputs[:, 0:9] = block_i[:, 0:9]
        derived_inputs[:, 9:14] = orig_inputs[:, 9:14]  # restitution/pitch/goal dims -- episode-level constants, unchanged
        derived_inputs[:, 14:20] = compute_engineered_features(block_i[:, 0:9])

        n_horizons_total = self.targets.shape[1] // N_TARGET_FIELDS_PER_HORIZON
        targets_by_skip: list[tuple[int, np.ndarray]] = []
        for skip in range(1, max_skip + 1):
            j = pair_idx + skip
            if j >= n_horizons_total:
                break
            base_j = j * N_TARGET_FIELDS_PER_HORIZON
            block_j = self.targets[valid_idx, base_j:base_j + N_TARGET_FIELDS_PER_HORIZON]
            targets_by_skip.append((skip, block_j.astype(np.float32, copy=False)))
        return derived_inputs, targets_by_skip, valid_idx

    def build_autoencoding_data(
        self, horizon_idx: int, indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Derives (input, target) pairs for pure autoencoding: treat horizon
        ``horizon_idx``'s recorded state as a pseudo-initial-state and ask
        the decoder to reconstruct THAT SAME state (queried at t=0 via
        ``forward_at()``, see ``BallDynamicsDecoder``'s ``forward_at``) --
        no "next horizon" involved at all, unlike ``build_adjacent_pair_
        data()``. Used for an optional autoencoding pretraining phase (see
        ``physics_pretrain.ball.autoencode_pretrain_epochs``) that primes
        the encoder+decoder to round-trip through the latent bottleneck
        before asking them to also learn dynamics.

        Unlike ``build_adjacent_pair_data()``, there's no resolved-state
        filter here -- reconstructing an already-frozen state through the
        bottleneck is just as meaningful a capacity test as any other
        state, there's no "trivial by definition" case to dilute (that
        concern was specific to DYNAMICS prediction, i.e. predicting no
        further change, not to autoencoding).

        Returns ``(derived_inputs, derived_targets)`` for ALL of
        ``indices`` (no filtering, so no need to also return row indices).
        """
        idx = indices if indices is not None else np.arange(len(self))
        base = horizon_idx * N_TARGET_FIELDS_PER_HORIZON
        block = self.targets[idx, base:base + N_TARGET_FIELDS_PER_HORIZON]
        orig_inputs = self.inputs[idx]

        derived_inputs = np.zeros((len(idx), N_INPUT_FIELDS), dtype=np.float32)
        derived_inputs[:, 0:9] = block[:, 0:9]
        derived_inputs[:, 9:14] = orig_inputs[:, 9:14]
        derived_inputs[:, 14:20] = compute_engineered_features(block[:, 0:9])
        return derived_inputs, block.astype(np.float32, copy=False)

    def compute_resting_targets(
        self, min_start_speed_norm: float, rest_speed_norm: float, start_horizon_idx: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Derives ``(resting_pos, resting_mask)`` for the resting-position
        head (see ``BallDynamicsAutoencoder.resting_head``) -- entirely from
        ``self.inputs``/``self.targets``, no extra recorded columns needed
        (unlike ``crossing_pos``/``crossing_dt``, which need the separate
        ``crossings``/``crossing_times`` arrays), so this works on any
        dataset regardless of when it was generated.

        ``start_horizon_idx`` (default ``None``): which pseudo-start's speed
        gates eligibility and which horizons count as "later, at rest".
        ``None`` (the original t=0 case) uses ``self.inputs``' own speed and
        considers ALL recorded horizons as candidates. An int ``h`` instead
        uses recorded horizon ``h``'s own speed (from ``self.targets``) and
        only considers horizons ``>= h`` as candidates -- generalizing the
        SAME target definition to any recorded horizon standing in as a
        pseudo-initial-state (see ``build_adjacent_pair_data``'s docstring
        for the general idea), so the resting-position task isn't limited to
        the episode's true t=0.

        For the chosen start: only eligible if its speed (``||vel_x, vel_y,
        vel_z||``, normalized units) exceeds ``min_start_speed_norm`` -- a
        ball that's already slow makes "predict where it comes to rest"
        close to a trivial "same place it started" answer, same rationale as
        ``build_adjacent_pair_data``'s own start-speed filter. Among the
        candidate horizons, finds every one whose speed drops below
        ``rest_speed_norm`` (i.e. has genuinely come to rest, not just
        slowed down) and, if at least one exists, takes the LATEST such
        horizon's ``(pos_x, pos_y)`` (no height -- same convention as
        ``crossing_head``) as the target -- the latest rather than the
        first, since the ball can still be settling (bouncing, rolling)
        at an earlier low-speed instant and the last recorded one is the
        best available estimate of where it actually ends up. Episodes
        with no candidate horizon below ``rest_speed_norm`` are EXCLUDED
        entirely (``resting_mask=False``) -- unlike ``crossing_dt``, there's
        no sentinel value to train against here, since "we don't know
        where/if it stops" isn't a well-defined (x, y) to regress toward
        the way "-1" is well-defined for a scalar time.

        Returns ``(resting_pos, resting_mask)``, both length ``len(self)``
        (the FULL dataset, not filtered by any ``indices`` -- callers slice
        by their own ``batch_idx``, same convention as ``crossing_pos``/
        ``crossing_mask``). ``resting_pos`` is garbage (zeros) wherever
        ``resting_mask`` is False -- callers must mask, never read it
        unmasked.
        """
        n = len(self)
        n_horizons = self.targets.shape[1] // N_TARGET_FIELDS_PER_HORIZON
        reshaped = self.targets.reshape(n, n_horizons, N_TARGET_FIELDS_PER_HORIZON)
        if start_horizon_idx is None:
            start_speed = np.linalg.norm(self.inputs[:, 3:6], axis=1)
            candidate_start = 0
        else:
            start_speed = np.linalg.norm(reshaped[:, start_horizon_idx, 3:6], axis=1)
            candidate_start = start_horizon_idx
        eligible = start_speed > min_start_speed_norm

        horizon_speed = np.linalg.norm(reshaped[:, candidate_start:, 3:6], axis=2)
        slow_mask = horizon_speed < rest_speed_norm
        has_slow = slow_mask.any(axis=1)
        # Index of the LAST True along axis 1 (latest resting horizon,
        # relative to `candidate_start`): reverse the axis, take the first
        # True via argmax (argmax stops at the first max -- here the first
        # True, since bool argmax treats True > False), then mirror the
        # index back and shift by candidate_start to get the absolute
        # horizon index.
        last_slow_rel = (horizon_speed.shape[1] - 1) - np.argmax(slow_mask[:, ::-1], axis=1)
        last_slow_idx = candidate_start + last_slow_rel

        resting_mask = eligible & has_slow
        resting_pos = reshaped[np.arange(n), last_slow_idx, 0:2].astype(np.float32, copy=False)
        return resting_pos, resting_mask

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

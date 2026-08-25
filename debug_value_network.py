"""Sanity-check a fresh, separate value network's ability to fit MC returns
from a demo dataset alone, with a train/val split.

If a value network can't get val loss well below 1.0 (normalized MSE) on
this data, it points to a bug upstream of PPO -- reward calculation,
observation encoding, or the returns themselves -- not to a PPO-specific
issue.

Usage:
    uv run python debug_value_network.py --data demonstrations/phase_1_debug/ \
        --epochs 40 --gamma 0.995 --lr 1e-3
"""
from __future__ import annotations

# Must run BEFORE numpy (or torch, which imports it) is first imported ANYWHERE
# in this process -- numpy's underlying BLAS (OpenBLAS here) reads these env
# vars at import/first-use to size its OWN internal thread pool, entirely
# independent of torch.set_num_threads(). Left unset, each process's numpy
# calls (the physics engine and observation encoder are numpy-heavy, not
# torch) default to one BLAS thread PER LOGICAL CORE.
#
# This has to live here, above the `import numpy as np`/`import torch` lines
# below, not inside _rollout_worker_entry() where it would be more obviously
# scoped to --n-parallel-envs workers: ProcessPoolExecutor's spawn bootstrap
# re-executes this ENTIRE script as __main__ in every child process before
# any function body runs (see multiprocessing.spawn's prepare()/
# _fixup_main_from_path -- the same mechanism that reconstructs sys.modules
# ["__main__"] for any spawned worker), so by the time _rollout_worker_entry
# actually starts executing, this module's own top-level `import numpy as np`
# has ALREADY run in that child and OpenBLAS has already picked its default
# thread count -- setting the env var at that point is too late to change it.
# Setting it here, before those imports, fixes it for the main process AND
# every spawned rollout worker at once. os.environ.setdefault() (not a flat
# assignment) so an operator's own OMP_NUM_THREADS etc. from the shell is
# still respected if already set.
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("footballcoach.ai.debug_value")


def _log_dataset_distribution(
    ds, valid_idx: np.ndarray, returns: np.ndarray, trainee_valid_idx: np.ndarray | None = None,
) -> None:
    """Print self/opponent ai_type mix, kick/tackle rates, and reward/done
    counts so a bad split or a degenerate reward signal is visible before
    burning time on training.

    ``valid_idx`` (both trainee's AND a non-immobile secondary player's own
    rows -- the actual TRAINING population) is used for kick/tackle rate.
    ``trainee_valid_idx`` (trainee's own rows only; defaults to ``valid_idx``
    if not given) is used for return percentiles -- returns are inherently
    tied to episode outcome, which is trainee-perspective (see main()'s
    ``trainee_valid_idx`` comment), so mixing in a losing secondary player's
    returns there would skew the percentiles same as it did the by-outcome
    breakdowns."""
    from footballcoach.ai.ppo.bc import (
        AI_TYPE_IMMOBILE, AI_TYPE_NEURAL, AI_TYPE_RULES,
        _I_AI_TYPE, _I_KICK_THIS_TICK, _I_OPPONENT_AI_TYPE, _I_TACKLE_ATTEMPT,
    )

    labels = ds._labels
    n = len(labels)
    n_eps = ds.n_episodes()
    log.info(f"--- Dataset distribution ({n:,} rows, {n_eps} episodes) ---")

    def _pct_by_type(col: int, name: str) -> None:
        for label, code in (("rules", AI_TYPE_RULES), ("immobile", AI_TYPE_IMMOBILE), ("neural", AI_TYPE_NEURAL)):
            frac = float((labels[:, col] == code).mean())
            log.info(f"  {name} == {label}: {100.0 * frac:.1f}%")

    _pct_by_type(_I_AI_TYPE, "self.ai_type")
    _pct_by_type(_I_OPPONENT_AI_TYPE, "opponent.ai_type")

    if len(valid_idx) > 0:
        v_labels = labels[valid_idx]
        kick_rate = float((v_labels[:, _I_KICK_THIS_TICK] > 0.5).mean())
        tackle_rate = float((v_labels[:, _I_TACKLE_ATTEMPT] > 0.5).mean())
        log.info(f"  valid rows: kick_this_tick rate={100.0 * kick_rate:.2f}%  "
                 f"tackle_attempt rate={100.0 * tackle_rate:.2f}%")

    n_done = int(ds._dones.sum())
    n_zero_reward = int((ds._rewards == 0.0).sum())
    log.info(f"  dones=1 rows: {n_done:,}  |  zero-reward rows: {n_zero_reward:,} "
              f"({100.0 * n_zero_reward / max(n, 1):.1f}%)")
    _ret_pool = trainee_valid_idx if trainee_valid_idx is not None else valid_idx
    _returns_valid = returns[_ret_pool] if len(_ret_pool) > 0 else returns
    log.info(f"  return percentiles (trainee's own valid rows): "
              f"p10={np.percentile(_returns_valid, 10):.2f}  p50={np.percentile(_returns_valid, 50):.2f}  "
              f"p90={np.percentile(_returns_valid, 90):.2f}")


def _log_reward_component_breakdown(ds, row_pool: np.ndarray, outcome_filter: str | None = None) -> None:
    """Print the same per-episode reward-component mean/std/min/max table
    ppo_trainer.py prints each rollout, but summed from the LOADED dataset's
    own per-row reward_components (ds._reward_components) over every complete
    episode in it -- describes the actual data being trained on, not a
    freshly-simulated, differently-configured opponent mix.

    ``row_pool``: restricts BOTH the episode boundaries AND which rows are
    summed within each episode to this set (callers pass ``ds.valid_indices()``
    to exclude immobile players' own rows -- self.ai_type==immobile rows
    carry no action supervision and, for a secondary/opponent player, can
    carry reward-component values that are only meaningful when that
    player's own AI is actually acting; see scenario_env.py's
    ``sec_box_terminal`` immobile-gate). Only using ``row_pool`` to pick
    episode START/END bounds (via ``ds.episode_row_ranges()``) would NOT be
    enough on its own -- interior rows between those bounds still belong to
    the full dataset's contiguous layout and would be silently re-included
    by a raw ``[start:end+1]`` slice, so this masks every interior row
    against ``row_pool`` as well.

    ``outcome_filter``: when given (e.g. "ball_out"), restricts the table to
    only episodes whose ``ds.classify_outcome()`` matches -- lets you check
    whether a component that should be terminal-exclusive (e.g. "box"/"spd",
    which should only ever fire on a "win" episode) is unexpectedly nonzero
    on a DIFFERENT outcome, which would point to a same-tick outcome/reward
    mismatch bug rather than an actual reward-design issue."""
    from footballcoach.ai.ppo.ppo_trainer import REWARD_COMP_LABELS

    if not ds.has_reward_components:
        log.info("--- No per-component reward data in this dataset "
                  "(re-record with the updated record_demonstrations.py) ---")
        return

    row_pool = np.sort(row_pool)
    pool_mask = np.zeros(len(ds), dtype=bool)
    pool_mask[row_pool] = True

    ranges = ds.episode_row_ranges(row_pool)
    if outcome_filter is not None:
        ranges = [(s, e) for s, e in ranges if ds.classify_outcome(e) == outcome_filter]
    episode_comp_list: list[dict[str, float]] = []
    for start, end in ranges:
        sub_mask = pool_mask[start:end + 1]
        ep_comp = {
            key: float(ds._reward_components[start:end + 1, col][sub_mask].sum())
            for col, key in enumerate(ds._reward_component_keys)
        }
        episode_comp_list.append(ep_comp)

    _label = f"outcome={outcome_filter}" if outcome_filter is not None else "all episodes"
    log.info(f"--- Reward component breakdown ({_label}, {len(ranges)} episode(s)) ---")
    if not ranges:
        return
    all_keys = [k for k, _ in REWARD_COMP_LABELS if any(k in ep for ep in episode_comp_list)]
    col_w = 14
    log.info(f"  {'component':<{col_w}}  {'mean':>8}  {'std':>7}  {'min':>8}  {'max':>8}")
    lbl_map = dict(REWARD_COMP_LABELS)
    for k in all_keys:
        vals = [ep[k] for ep in episode_comp_list if k in ep]
        if not vals:
            continue
        arr = np.array(vals)
        log.info(
            f"  {lbl_map.get(k, k):<{col_w}}  {arr.mean():>+8.3f}  {arr.std():>7.3f}"
            f"  {arr.min():>+8.3f}  {arr.max():>+8.3f}"
        )


def _iterate_over(ds, idx, returns, batch_size, shuffle):
    """Minibatch iterator over an arbitrary pre-split row-index array (train
    or val) -- the dataset's own iterators only support valid_indices()/
    arange(n), not a caller-supplied index subset.

    ``idx`` (train_idx/val_idx from split_train_val_indices()) is already
    sorted increasing. Row-level np.random.shuffle() would scatter each
    minibatch's gather (ds._self_feat[chunk], etc.) across random, far-apart
    dataset rows every epoch -- poor cache locality for no benefit, since
    randomizing which *rows* land in the same minibatch doesn't require
    randomizing memory access pattern. Shuffling block ORDER instead keeps
    each individual gather a near-contiguous slice of idx (and therefore of
    the underlying arrays) while still varying batch composition/order
    across epochs."""
    from footballcoach.ai.bc.dataset import _build_ai_type_arrays, _to_tensor

    if shuffle:
        block_starts = list(range(0, len(idx), batch_size))
        np.random.shuffle(block_starts)
    else:
        block_starts = range(0, len(idx), batch_size)
    for start in block_starts:
        chunk = idx[start:start + batch_size]
        if len(chunk) == 0:
            continue
        self_ai_type, other_ai_type = _build_ai_type_arrays(
            ds._labels[chunk], ds._exists_mask[chunk]
        )
        obs_dict = {
            "self_feat":   _to_tensor(ds._self_feat[chunk], None),
            "other_feat":  _to_tensor(ds._other_feat[chunk], None),
            "exists_mask": _to_tensor(ds._exists_mask[chunk], None),
            "ball_feat":   _to_tensor(ds._ball_feat[chunk], None),
            "global_feat": _to_tensor(ds._global_feat[chunk], None),
            "self_ai_type":  _to_tensor(self_ai_type, None),
            "other_ai_type": _to_tensor(other_ai_type, None),
        }
        ret_batch = _to_tensor(returns[chunk], None)
        yield obs_dict, ret_batch



def _log_component_correlation(ds, val_idx: np.ndarray, gamma: float) -> None:
    """Correlate each reward component's per-episode MC-return contribution
    against the value network's episode-level residual (return - predicted)
    is done by the caller (see main()) -- this only needs each component's
    per-episode total, reported by mean/std here for a first look at scale.
    Requires the dataset to carry per-component reward data."""
    if not ds.has_reward_components:
        log.info("--- No per-component reward data in this dataset "
                  "(re-record with the updated record_demonstrations.py) ---")
        return
    from footballcoach.ai.ppo.ppo_trainer import REWARD_COMP_LABELS
    comp_returns = ds.compute_component_returns(gamma=gamma)
    lbl_map = dict(REWARD_COMP_LABELS)
    log.info("--- Per-component MC-return magnitude (val rows) ---")
    for k, arr in comp_returns.items():
        vals = arr[val_idx]
        if not np.any(vals):
            continue
        log.info(f"  {lbl_map.get(k, k):<16}  mean={vals.mean():+.4f}  std={vals.std():.4f}")


def _episode_residual_correlation(
    ds, val_idx: np.ndarray, gamma: float, residual_by_row: np.ndarray,
) -> None:
    """Per reward component, correlate its per-episode MC-return total
    against the value network's per-episode mean residual (return -
    predicted), across all complete val episodes. Low |correlation| for a
    component that has non-trivial variance means the network isn't
    picking up on that component's contribution at all -- a candidate
    "hardest to predict" signal."""
    if not ds.has_reward_components:
        return
    from footballcoach.ai.ppo.ppo_trainer import REWARD_COMP_LABELS
    comp_returns = ds.compute_component_returns(gamma=gamma)
    ranges = ds.episode_row_ranges(val_idx)
    if len(ranges) < 3:
        log.info("--- Not enough complete val episodes for component-residual correlation ---")
        return
    lbl_map = dict(REWARD_COMP_LABELS)
    ep_residual = np.array([residual_by_row[s] for s, _e in ranges])  # return/residual is constant per-episode at row s (row 0 return == episode return)
    log.info(f"--- Reward-component vs. value-residual correlation ({len(ranges)} val episodes) ---")
    log.info(f"  {'component':<16}  {'corr':>7}  {'comp_std':>9}")
    rows = []
    for k, arr in comp_returns.items():
        ep_comp = np.array([arr[s] for s, _e in ranges])
        if ep_comp.std() < 1e-8:
            continue
        corr = float(np.corrcoef(ep_comp, ep_residual)[0, 1])
        rows.append((k, corr, ep_comp.std()))
    rows.sort(key=lambda r: abs(r[1]))
    for k, corr, std in rows:
        log.info(f"  {lbl_map.get(k, k):<16}  {corr:>+7.3f}  {std:>9.4f}")
    if rows:
        log.info("  (components near the top -- low |corr| despite real variance -- "
                  "are the ones the value net's errors track least; read alongside "
                  "the per-component MC-return magnitude above.)")


def _episode_rows_to_match_log(ds, start: int, end: int) -> list[dict]:
    """Convert dataset rows [start, end] (one complete episode, both players
    interleaved as consecutive rows per timed sample -- see
    record_demonstrations.py's _record_now(player_id=None)) into an event
    list SHAPED LIKE engine/match_logger.py's MatchEvent records (same field
    names/types: "time_s", 3-tuple "ball_pos"/"player_pos", "player_id",
    "possessor_id", a "start" event with "player_positions"/"ball_vel",
    etc.) so scripts/visualise_match_log.py can render it directly, exactly
    like a real MatchLogger.save() dump.

    Only a compact subset of rows becomes events -- possession changes (both
    self AND opponent gaining/losing, with the ball position at that
    moment), kicks, tackle attempts, and episode end (annotated with the
    full reward-component breakdown, since that's the only row where "why
    did this episode score what it scored" actually matters). Deliberately
    does NOT log a row for every nonzero per-step reward (appr/hdg/prog etc.
    fire almost every tick) -- this must stay skimmable, not a full
    tick-by-tick dump. Uses only what's already encoded in obs_self_feat/
    obs_other_feat/obs_ball_feat/bc_labels/reward_components -- no engine
    replay needed.

    "time_s" is NOT wall-clock time (real per-row timestamps aren't retained
    in the dataset) -- it's just the row's position within the episode, so
    the timeline/x-axis is evenly spaced by decision tick rather than by
    real seconds.
    """
    from footballcoach.ai.ppo.bc import (
        _I_AI_TYPE, _I_KICK_THIS_TICK, _I_OPPONENT_AI_TYPE, _I_TACKLE_ATTEMPT,
        AI_TYPE_IMMOBILE, AI_TYPE_NEURAL, AI_TYPE_RULES,
    )

    _AI_TYPE_NAME = {AI_TYPE_RULES: "rules", AI_TYPE_IMMOBILE: "immobile", AI_TYPE_NEURAL: "neural"}
    # PlayerFeatures column indices (see schema.py field order).
    _VEL_X, _VEL_Y, _HAS_POSS, _POS_X, _POS_Y = 9, 10, 23, 30, 31
    # BallFeatures column indices (height_m is already in real metres, not
    # normalized -- see schema.py's BallFeatures.height_m).
    _BALL_POS_X, _BALL_POS_Y, _BALL_POS_Z, _BALL_VEL_X, _BALL_VEL_Y = 0, 1, 2, 3, 4
    # Pitch half-diagonal -- ai/obs/encoder.py normalizes BOTH position
    # AND velocity x/y by this SAME constant for players and the ball alike
    # (encoder.py: "pos_x=ball.position.x / half_diag", "pos_y=... /
    # half_diag" -- NOT separate 52.5/34.0 per-axis divisors, despite what
    # schema.py's docstring implies ("normalized by standard half-dimensions
    # (52.5m x 34.0m)"). Using 52.5/34.0 here under-scales y by ~1.84x and x
    # by ~1.19x, silently producing positions that look well within bounds
    # when the real position is actually near/at the boundary -- this is
    # exactly what made early debugging of ball-out episodes so confusing.
    import math
    _HALF_DIAG = math.hypot(52.5, 34.0)

    def _pos3(feat, pos_x_i, pos_y_i) -> tuple[float, float, float]:
        return (
            round(float(feat[pos_x_i]) * _HALF_DIAG, 2),
            round(float(feat[pos_y_i]) * _HALF_DIAG, 2),
            0.0,
        )

    def _ball_pos3(ball_feat) -> tuple[float, float, float]:
        return (
            round(float(ball_feat[_BALL_POS_X]) * _HALF_DIAG, 2),
            round(float(ball_feat[_BALL_POS_Y]) * _HALF_DIAG, 2),
            round(float(ball_feat[_BALL_POS_Z]), 2),
        )

    def _vel3(feat, vel_x_i, vel_y_i) -> tuple[float, float, float]:
        return (
            round(float(feat[vel_x_i]) * _HALF_DIAG, 2),
            round(float(feat[vel_y_i]) * _HALF_DIAG, 2),
            0.0,
        )

    def _reward_breakdown(start_row: int, end_row: int) -> dict[str, float]:
        """Sum reward components over [start_row, end_row] (the WHOLE
        episode), not just end_row alone. Most components (poss/prog/appr
        etc.) are earned mid-episode on whichever row the underlying event
        happened, not necessarily the terminal row -- e.g. get_possession
        commonly fires several rows before a later ball_out, and can even
        fire on the SAME row as the terminal ball_out reward (both within
        one un-recorded decision interval's physics substeps). Reading only
        end_row silently drops every non-terminal component, which looked
        exactly like "the trainee never got possession" even on episodes
        where it clearly did (verified against live rollout data)."""
        if not ds.has_reward_components:
            return {}
        from footballcoach.ai.ppo.ppo_trainer import REWARD_COMP_LABELS
        lbl_map = dict(REWARD_COMP_LABELS)
        comp_sum = ds._reward_components[start_row:end_row + 1].sum(axis=0)
        return {
            lbl_map.get(k, k): round(float(comp_sum[i]), 4)
            for i, k in enumerate(ds._reward_component_keys)
            if abs(comp_sum[i]) > 1e-9
        }

    self_ai_type = _AI_TYPE_NAME.get(float(ds._labels[start][_I_AI_TYPE]), "unknown")
    opp_ai_type = _AI_TYPE_NAME.get(float(ds._labels[start][_I_OPPONENT_AI_TYPE]), "unknown")

    def _opp_feat(row: int):
        _exists_row = ds._exists_mask[row]
        _opp_slot = int(_exists_row.argmax()) if _exists_row.any() else 0
        return ds._other_feat[row, _opp_slot]

    start_self_feat = ds._self_feat[start]
    start_ball_feat = ds._ball_feat[start]
    start_opp_feat = _opp_feat(start)
    events: list[dict] = [{
        "time_s": 0.0, "event": "start",
        "ball_pos": _ball_pos3(start_ball_feat),
        "ball_vel": _vel3(start_ball_feat, _BALL_VEL_X, _BALL_VEL_Y),
        "player_positions": {
            "self": {"pos": _pos3(start_self_feat, _POS_X, _POS_Y), "team": "left",
                     "ai_type": self_ai_type},
            "opponent": {"pos": _pos3(start_opp_feat, _POS_X, _POS_Y), "team": "right",
                         "ai_type": opp_ai_type},
        },
    }]

    last_possessor: str | None = None
    for row in range(start, end + 1):
        t = float(row - start)
        self_feat = ds._self_feat[row]
        ball_feat = ds._ball_feat[row]
        ball_pos = _ball_pos3(ball_feat)
        pos = _pos3(self_feat, _POS_X, _POS_Y)
        # The (single, in phase 1) opponent occupies a RANDOMIZED slot in
        # other_feat each row (see ai/obs/encoder.py's rng.sample slot
        # shuffle) -- find it via exists_mask rather than assuming slot 0.
        # has_possession there signals the OPPONENT gaining/losing the ball,
        # which self_feat's has_possession alone can't distinguish from a
        # loose ball.
        other_feat = _opp_feat(row)
        self_has_poss = self_feat[_HAS_POSS] > 0.5
        opp_has_poss = other_feat[_HAS_POSS] > 0.5
        possessor = "self" if self_has_poss else "opponent" if opp_has_poss else None
        if possessor != last_possessor and (possessor is not None or last_possessor is not None):
            events.append({
                "time_s": t, "event": "possession_change",
                "possessor_id": possessor, "ball_pos": ball_pos,
                "player_id": possessor,
                "player_pos": pos if possessor == "self" else
                    _pos3(other_feat, _POS_X, _POS_Y) if possessor == "opponent" else None,
            })
            last_possessor = possessor
        label = ds._labels[row]
        if label[_I_KICK_THIS_TICK] > 0.5:
            events.append({
                "time_s": t, "event": "kick", "player_id": "self",
                "player_pos": pos, "ball_pos": ball_pos,
            })
        if label[_I_TACKLE_ATTEMPT] > 0.5:
            events.append({
                "time_s": t, "event": "tackle_attempt", "player_id": "self",
                "player_pos": pos, "ball_pos": ball_pos,
            })

    end_ball_feat = ds._ball_feat[end]
    end_self_feat = ds._self_feat[end]
    end_opp_feat = _opp_feat(end)
    end_breakdown = _reward_breakdown(start, end)
    end_t = float(end - start)
    # Neither real MatchLogger events ("consistency"/"episode_end") nor this
    # reconstruction's own possession_change/kick/tackle_attempt events
    # necessarily carry a position for BOTH players on the episode's last
    # row -- without one, the visualiser's player-track lines never extend
    # past their "start" position, making both players look frozen even
    # though they moved the whole episode. Unlike the real MatchLogger, we
    # DO have both players' positions on every row here, so add one
    # "consistency" event per player (the established quiet-snapshot event
    # type -- small muted marker, not a "real" event) right before
    # episode_end so the visualiser's generic player_id/player_pos track
    # picks up each player's final position.
    events.append({
        "time_s": end_t, "event": "consistency", "player_id": "self",
        "player_pos": _pos3(end_self_feat, _POS_X, _POS_Y),
        "ball_pos": _ball_pos3(end_ball_feat),
    })
    events.append({
        "time_s": end_t, "event": "consistency", "player_id": "opponent",
        "player_pos": _pos3(end_opp_feat, _POS_X, _POS_Y),
        "ball_pos": _ball_pos3(end_ball_feat),
    })
    events.append({
        "time_s": end_t, "event": "episode_end",
        "ball_pos": _ball_pos3(end_ball_feat),
        "reward_total": round(float(ds._rewards[start:end + 1].sum()), 4),
        "reward_components": end_breakdown,
        "reward_cumulative": end_breakdown,
        "outcome": ds.classify_outcome(end) if ds.has_episode_outcomes else None,
    })
    return events


def _save_worst_episode_match_log(ds, val_idx: np.ndarray, residual_by_row: np.ndarray, out_path: str) -> None:
    """For EACH outcome present among complete val episodes, find the one
    episode with the largest |residual| (return - predicted value, evaluated
    at the episode's first row) and save a synthetic match log (see
    _episode_rows_to_match_log) for it as JSON, so a single bad outcome
    bucket can't hide/dominate the others the way a single global "worst
    episode" would. `out_path` (e.g. "results/debug_value_worst_episode.json")
    is used as a naming template -- one file per outcome is written as
    "<stem>_<outcome><suffix>" alongside it (e.g.
    "results/debug_value_worst_episode_ball_out.json"). `outcome` is the
    only top-level field not already in the last event (episode_end) --
    everything else (AI matchup, final positions/velocities, reward
    breakdown) lives there, not duplicated at the top, to keep each file
    skimmable."""
    import json

    ranges = ds.episode_row_ranges(val_idx)
    if not ranges:
        log.info("--- No complete val episodes to extract worst-case match logs from ---")
        return

    by_outcome: dict[str, list[tuple[int, int]]] = {}
    for start, end in ranges:
        outcome = ds.classify_outcome(end) if ds.has_episode_outcomes else "unknown"
        by_outcome.setdefault(outcome, []).append((start, end))

    out_path = Path(out_path)
    for outcome in sorted(by_outcome):
        outcome_ranges = by_outcome[outcome]
        worst_start, worst_end = max(outcome_ranges, key=lambda r: abs(residual_by_row[r[0]]))
        worst_residual = float(residual_by_row[worst_start])
        events = _episode_rows_to_match_log(ds, worst_start, worst_end)
        # Metadata folded into the episode_end event as extra keys (ignored
        # by scripts/visualise_match_log.py, which only reads specific known
        # fields) rather than wrapped in a top-level dict -- the file itself
        # must be a bare event list to match real MatchLogger.save() dumps,
        # so this is directly viewable with:
        #   uv run python scripts/visualise_match_log.py <this file>
        events[-1]["residual"] = worst_residual
        events[-1]["row_range"] = [int(worst_start), int(worst_end)]
        events[-1]["n_episodes_this_outcome"] = len(outcome_ranges)
        outcome_path = out_path.with_name(f"{out_path.stem}_{outcome}{out_path.suffix}")
        with open(outcome_path, "w") as f:
            json.dump(events, f, indent=2)
        log.info(f"--- Worst val episode for outcome={outcome} ({len(outcome_ranges)} "
                  f"episode(s)): rows [{worst_start}, {worst_end}], "
                  f"residual={worst_residual:+.3f} -- saved match log to {outcome_path} ---")


def _iterate_over_with_indices(ds, idx, returns, batch_size):
    """Like _iterate_over(shuffle=False) but also yields the row-index chunk,
    so callers can scatter per-row outputs (e.g. residuals) back into a
    dataset-sized array."""
    from footballcoach.ai.bc.dataset import _build_ai_type_arrays, _to_tensor

    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        if len(chunk) == 0:
            continue
        self_ai_type, other_ai_type = _build_ai_type_arrays(
            ds._labels[chunk], ds._exists_mask[chunk]
        )
        obs_dict = {
            "self_feat":   _to_tensor(ds._self_feat[chunk], None),
            "other_feat":  _to_tensor(ds._other_feat[chunk], None),
            "exists_mask": _to_tensor(ds._exists_mask[chunk], None),
            "ball_feat":   _to_tensor(ds._ball_feat[chunk], None),
            "global_feat": _to_tensor(ds._global_feat[chunk], None),
            "self_ai_type":  _to_tensor(self_ai_type, None),
            "other_ai_type": _to_tensor(other_ai_type, None),
        }
        ret_batch = _to_tensor(returns[chunk], None)
        yield obs_dict, ret_batch, chunk


def _iterate_synthetic_batches(obs_list: list, returns_arr: np.ndarray, batch_size: int, shuffle: bool = False):
    """Like _iterate_over_with_indices but sourced from the synthetic
    obs_list/returns_arr (see _generate_synthetic_ball_out_set) instead of
    the dataset arrays -- lets those rows be folded into real training
    batches rather than only scored after the fact. Also yields the chunk
    index array so callers can tag each row with its real (ball_out/win)
    outcome."""
    from footballcoach.ai.bc.dataset import _to_tensor

    n = len(obs_list)
    order = np.arange(n)
    if shuffle:
        np.random.shuffle(order)
    self_feat = np.stack([o.self_feat for o in obs_list])
    other_feat = np.stack([o.other_feat for o in obs_list])
    exists_mask = np.stack([o.exists_mask for o in obs_list])
    ball_feat = np.stack([o.ball_feat for o in obs_list])
    global_feat = np.stack([o.global_feat for o in obs_list])
    self_ai_type = np.stack([o.self_ai_type for o in obs_list])
    other_ai_type = np.stack([o.other_ai_type for o in obs_list])
    for start in range(0, n, batch_size):
        chunk = order[start:start + batch_size]
        if len(chunk) == 0:
            continue
        obs_dict = {
            "self_feat":     _to_tensor(self_feat[chunk], None),
            "other_feat":    _to_tensor(other_feat[chunk], None),
            "exists_mask":   _to_tensor(exists_mask[chunk], None),
            "ball_feat":     _to_tensor(ball_feat[chunk], None),
            "global_feat":   _to_tensor(global_feat[chunk], None),
            "self_ai_type":  _to_tensor(self_ai_type[chunk], None),
            "other_ai_type": _to_tensor(other_ai_type[chunk], None),
        }
        ret_batch = _to_tensor(returns_arr[chunk], None)
        yield obs_dict, ret_batch, chunk


def _log_returns_by_outcome(ds, row_pool: np.ndarray, returns: np.ndarray, label: str) -> None:
    """Print MC-return mean/std/min/max split by episode outcome (win/loss/
    ball_out/invalid/timeout/incomplete, see DemonstrationDataset.
    classify_outcome()) over *row_pool* -- lets a "returns look fine
    overall" summary be checked for a single outcome bucket dominating or
    masking a badly-behaved one."""
    if not ds.has_episode_outcomes:
        log.info(f"--- Returns by outcome ({label}): no ground-truth episode "
                  "outcomes (re-record with the updated record_demonstrations.py) ---")
        return
    row_pool = np.sort(row_pool)
    outcomes = ds.row_outcomes(row_pool)
    log.info(f"--- MC returns by outcome ({label}, {len(row_pool):,} rows) ---")
    for outcome in sorted(set(outcomes)):
        mask = outcomes == outcome
        vals = returns[row_pool[mask]]
        log.info(
            f"  {outcome:<12} n={int(mask.sum()):>7,}  "
            f"mean={vals.mean():+.3f}  std={vals.std():.3f}  "
            f"min={vals.min():+.3f}  max={vals.max():+.3f}"
        )


def _log_episode_total_reward_by_outcome(ds, row_pool: np.ndarray, label: str) -> None:
    """Print mean/std/min/max of per-EPISODE total reward (undiscounted sum
    across the whole episode -- i.e. Monte Carlo return with gamma=1, NOT
    the possibly-discounted per-row ``returns`` array used elsewhere in this
    script), split by episode outcome, over complete episodes in *row_pool*.
    One number per episode, unlike ``_log_returns_by_outcome()`` above (which
    reports per-ROW discounted-return stats, i.e. many numbers per episode)."""
    if not ds.has_episode_outcomes:
        log.info(f"--- Episode total reward by outcome ({label}): no ground-truth "
                  "episode outcomes (re-record with the updated record_demonstrations.py) ---")
        return
    row_pool = np.sort(row_pool)
    ranges = ds.episode_row_ranges(row_pool)
    if not ranges:
        log.info(f"--- Episode total reward by outcome ({label}): no complete episodes ---")
        return
    # gamma=1 MC return evaluated at each episode's first row equals the plain
    # undiscounted sum of every reward in that episode (the backward MC
    # recursion in compute_returns() resets to 0 at each done boundary) --
    # computed over the WHOLE dataset (not row_pool, which may be a filtered
    # subset) so no reward within the episode is missed.
    mc1_returns = ds.compute_returns(gamma=1.0)
    ep_totals = np.array([mc1_returns[s] for s, _e in ranges])
    ep_outcomes = np.array([ds.classify_outcome(e) for _s, e in ranges])
    log.info(f"--- Episode total reward by outcome ({label}, {len(ranges)} episode(s)) ---")
    for outcome in sorted(set(ep_outcomes)):
        vals = ep_totals[ep_outcomes == outcome]
        log.info(
            f"  {outcome:<12} n={len(vals):>6,}  "
            f"mean={vals.mean():+.3f}  std={vals.std():.3f}  "
            f"min={vals.min():+.3f}  max={vals.max():+.3f}"
        )


def _lr_feature_matrix(top_speed: np.ndarray, ball_dist: np.ndarray, ball_feat: np.ndarray,
                       attacks_pos_x: np.ndarray, time_rem: np.ndarray) -> np.ndarray:
    """Shared 4-feature matrix (top_speed, ball_dist, ball_to_box, time_rem)
    used by BOTH the real-dataset linear regression fit and the synthetic
    ball-out set's evaluation (see _lr_features_for_obs_list) -- one
    definition so the two stay comparable."""
    half_len_m, half_wid_m = 52.5, 34.0
    half_diag_m = math.hypot(half_len_m, half_wid_m)
    box_length_m, half_box_w_m = 16.5, 40.32 / 2.0
    bx_m = ball_feat[:, 0] * half_len_m
    by_m = ball_feat[:, 1] * half_wid_m
    rx_min_pos_m = half_len_m - box_length_m  # +x box's near edge
    rx_max_neg_m = -half_len_m + box_length_m  # -x box's near edge
    dx_m = np.where(
        attacks_pos_x,
        np.maximum(rx_min_pos_m - bx_m, 0.0),
        np.maximum(bx_m - rx_max_neg_m, 0.0),
    )
    dy_m = np.maximum(np.abs(by_m) - half_box_w_m, 0.0)
    ball_to_box = np.hypot(dx_m, dy_m) / half_diag_m
    return np.stack([top_speed, ball_dist, ball_to_box, time_rem], axis=1).astype(np.float64)


def _lr_features_for_obs_list(obs_list: list) -> np.ndarray:
    """Same 4-feature matrix as _lr_feature_matrix, computed from a list of
    ObservationBatch (the synthetic ball-out set) instead of dataset arrays."""
    from dataclasses import fields as _dc_fields
    from footballcoach.ai.config import load_ai_config
    from footballcoach.ai.obs.schema import PlayerFeatures, GlobalFeatures

    _self_field_names = [f.name for f in _dc_fields(PlayerFeatures)]
    _global_field_names = [f.name for f in _dc_fields(GlobalFeatures)]
    i_top_speed = _self_field_names.index("top_speed")
    i_ball_dist = _self_field_names.index("ball_distance_m")
    i_attack_dir = _self_field_names.index("attacking_direction")
    i_time_rem = _global_field_names.index("time_remaining_norm")
    _cfg = load_ai_config()
    time_norm_max_s = float(_cfg.get("observation", {}).get("time_remaining_norm_max_s", 7200.0))
    max_episode_s = float(_cfg.get("curriculum", {}).get("phase1_max_episode_s", 60.0))

    self_feat = np.stack([o.self_feat for o in obs_list])
    ball_feat = np.stack([o.ball_feat for o in obs_list])
    global_feat = np.stack([o.global_feat for o in obs_list])
    top_speed = self_feat[:, i_top_speed]
    ball_dist = self_feat[:, i_ball_dist]
    attacks_pos_x = self_feat[:, i_attack_dir] > 0.0
    time_remaining_s = np.expm1(global_feat[:, i_time_rem] * math.log1p(time_norm_max_s))
    time_rem = np.clip(time_remaining_s / max_episode_s, 0.0, 1.0)
    return _lr_feature_matrix(top_speed, ball_dist, ball_feat, attacks_pos_x, time_rem)


def _run_linear_regression(
    ds, train_idx: np.ndarray, val_idx: np.ndarray, returns: np.ndarray,
    outcome_weight_by_row: np.ndarray | None = None,
    synthetic_obs_list: list | None = None, synthetic_returns: np.ndarray | None = None,
    synthetic_outcomes: np.ndarray | None = None,
) -> None:
    """Fit a plain closed-form linear regression (normal equations, no
    external dependency) predicting the MC return from just 4 hand-picked
    features -- top_speed attribute ("sprint"), player distance to ball,
    ball distance to opponent box, and time remaining -- once over ALL
    rows in train_idx/val_idx, once restricted to rows belonging to "win"
    episodes only. Purely diagnostic: if a 4-feature linear model gets
    anywhere close to the neural value net's val MSE, that's a strong
    signal the value net's job here is mostly "learn this simple
    relationship" rather than something intrinsically hard.

    If ``outcome_weight_by_row`` is given (see --outcome-reweight), an extra
    weighted-least-squares fit is run alongside the plain OLS fit -- rows are
    scaled by sqrt(weight) before the same normal-equations solve, which is
    algebraically equivalent to minimizing the weighted sum of squared
    residuals. Val MSE is always reported unweighted for comparability."""
    from dataclasses import fields as _dc_fields
    from footballcoach.ai.config import load_ai_config
    from footballcoach.ai.obs.schema import PlayerFeatures, GlobalFeatures

    _self_field_names = [f.name for f in _dc_fields(PlayerFeatures)]
    _global_field_names = [f.name for f in _dc_fields(GlobalFeatures)]
    _I_TOP_SPEED = _self_field_names.index("top_speed")
    _I_BALL_DIST = _self_field_names.index("ball_distance_m")
    _I_ATTACK_DIR = _self_field_names.index("attacking_direction")
    _I_TIME_REM = _global_field_names.index("time_remaining_norm")

    # time_remaining_norm is log1p-compressed (see ai/obs/encoder.py's
    # _encode_global_features()) -- invert it back to raw seconds, then
    # re-normalise as a plain LINEAR fraction of the phase's episode cap
    # (not log-compressed) so this regression sees time on a scale where
    # equal deltas mean equal time, as requested.
    _cfg = load_ai_config()
    _time_norm_max_s = float(_cfg.get("observation", {}).get("time_remaining_norm_max_s", 7200.0))
    _max_episode_s = float(_cfg.get("curriculum", {}).get("phase1_max_episode_s", 60.0))

    def _features(idx: np.ndarray) -> np.ndarray:
        top_speed = ds._self_feat[idx, _I_TOP_SPEED]
        ball_dist = ds._self_feat[idx, _I_BALL_DIST]
        time_remaining_s = np.expm1(ds._global_feat[idx, _I_TIME_REM] * math.log1p(_time_norm_max_s))
        time_rem = np.clip(time_remaining_s / _max_episode_s, 0.0, 1.0)
        # ball-dist-to-opponent-box isn't in the observation schema (see
        # scenario_env.py's _ball_dist_to_opponent_box()) -- recompute it
        # directly from ball pos_x/pos_y (BallFeatures) using the same
        # standard-pitch box geometry used by phase1_reward()'s "prox" term.
        # ball_feat is stored in WORLD/engine frame (canonicalization only
        # happens at the network forward boundary, see ai/obs/canonical.py),
        # and the trainee's attacking side is randomised per episode -- must
        # pick the box on the side self_feat.attacking_direction says this
        # player attacks, not always the +x box (an earlier version of this
        # function hardcoded +x, silently computing distance to the WRONG
        # box for ~half the rows and washing out ball_to_box's coefficient).
        #
        # dx/dy are computed in REAL METERS (pos_x*52.5, pos_y*34.0 -- undoing
        # the two per-axis normalisers), THEN combined via hypot() and divided
        # by the single isotropic pitch half-diagonal -- mirroring how every
        # other distance field in the observation schema (distance_m,
        # ball_distance_m, see ai/obs/encoder.py) is built, so this is a true
        # normalised Euclidean distance, not an anisotropic mix of two
        # differently-scaled per-axis normalisers (dx/52.5 vs dy/34.0 summed
        # under one hypot(), as an earlier version of this function did).
        attacks_pos_x = ds._self_feat[idx, _I_ATTACK_DIR] > 0.0
        return _lr_feature_matrix(top_speed, ball_dist, ds._ball_feat[idx], attacks_pos_x, time_rem)

    def _fit_and_eval(idx_train: np.ndarray, idx_val: np.ndarray, label: str, weighted: bool = False) -> None:
        if len(idx_train) < 5 or len(idx_val) == 0:
            log.info(f"  [{label}] not enough rows to fit (train={len(idx_train)}, val={len(idx_val)})")
            return
        X_train = _features(idx_train)
        y_train = returns[idx_train].astype(np.float64)
        X_val = _features(idx_val)
        y_val = returns[idx_val].astype(np.float64)
        X_train_b = np.hstack([X_train, np.ones((len(X_train), 1))])
        if weighted:
            assert outcome_weight_by_row is not None
            # WLS via normal equations: scaling both X and y rows by sqrt(w)
            # turns min sum(w*(Xb-y)^2) into an ordinary least-squares problem.
            sqrt_w = np.sqrt(outcome_weight_by_row[idx_train].astype(np.float64))
            coef, _, _, _ = np.linalg.lstsq(X_train_b * sqrt_w[:, None], y_train * sqrt_w, rcond=None)
        else:
            coef, _, _, _ = np.linalg.lstsq(X_train_b, y_train, rcond=None)
        X_val_b = np.hstack([X_val, np.ones((len(X_val), 1))])
        pred_train = X_train_b @ coef
        pred_val = X_val_b @ coef
        mse_train = float(np.mean((pred_train - y_train) ** 2))
        mse_val = float(np.mean((pred_val - y_val) ** 2))
        ret_var = max(float(y_train.var()), 1e-6)
        # RMSE is in the same units as the MC return itself (MSE is squared units,
        # which is why a return range of [-4, 5] can show MSE values like 40).
        log.info(
            f"  [{label}] n_train={len(idx_train):,}  n_val={len(idx_val):,}  "
            f"coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)="
            f"{np.round(coef, 3).tolist()}\n"
            f"    train_rmse={math.sqrt(mse_train):.4f} (norm={math.sqrt(mse_train / ret_var):.4f})  "
            f"val_rmse={math.sqrt(mse_val):.4f} (norm={math.sqrt(mse_val / ret_var):.4f})"
        )

    log.info("--- Linear regression baseline (top_speed, ball_dist, ball_to_box, time_remaining) ---")
    _fit_and_eval(train_idx, val_idx, "all outcomes")
    if outcome_weight_by_row is not None:
        _fit_and_eval(train_idx, val_idx, "all outcomes, outcome-reweighted", weighted=True)

    if ds.has_episode_outcomes:
        train_outcomes = ds.row_outcomes(np.sort(train_idx))
        val_outcomes = ds.row_outcomes(np.sort(val_idx))
        train_win_idx = np.sort(train_idx)[train_outcomes == "win"]
        val_win_idx = np.sort(val_idx)[val_outcomes == "win"]
        _fit_and_eval(train_win_idx, val_win_idx, "win outcomes only")
    else:
        log.info("  [win outcomes only] skipped -- no ground-truth episode outcomes")

    if synthetic_obs_list:
        # Re-fit on ALL real rows (unweighted) so we have a coef vector to
        # evaluate against the synthetic set below.
        X_train_b = np.hstack([_features(train_idx), np.ones((len(train_idx), 1))])
        coef, _, _, _ = np.linalg.lstsq(X_train_b, returns[train_idx].astype(np.float64), rcond=None)
        ret_var = max(float(returns[train_idx].var()), 1e-6)
        all_feats = _lr_features_for_obs_list(synthetic_obs_list)
        # Reported once for all rows, then again per outcome (ball_out vs.
        # win) so a good "all rows" RMSE can't hide one outcome dominating
        # or masking the other.
        for label, mask in [
            ("all", np.ones(len(synthetic_obs_list), dtype=bool)),
            *([(o, synthetic_outcomes == o) for o in np.unique(synthetic_outcomes)]
              if synthetic_outcomes is not None else []),
        ]:
            if not mask.any():
                continue
            X_syn_b = np.hstack([all_feats[mask], np.ones((int(mask.sum()), 1))])
            pred_syn = X_syn_b @ coef
            syn_ret = synthetic_returns[mask]
            mse_syn = float(np.mean((pred_syn - syn_ret) ** 2))
            log.info(
                f"  [synthetic {label}, n={int(mask.sum())}] using coef fit on real data: "
                f"return mean={syn_ret.mean():+.3f} std={syn_ret.std():.3f}  "
                f"rmse={math.sqrt(mse_syn):.4f} (norm={math.sqrt(mse_syn / ret_var):.4f})"
            )


def _synthetic_ball_out_episode(env, rng: np.random.Generator, t_max_s: float = 0.7):
    """Build one real episode via the actual ScenarioEnv: trainee starts in
    possession, already moving fast toward a boundary (so it has t_max_s
    seconds or less before it would leave the pitch), then hands control to
    the SAME Phase1RulesAI used for real training/BC demonstrations -- not a
    hand-rolled AI. Phase1RulesAI will try to redirect toward the opponent
    box as usual, but starting this close/fast, it may not have time to turn
    before the ball exits. Runs env.step() for real (real engine, real
    reward function, real terminal-outcome classification).

    Returns (obs_list, returns_to_go, outcome) where obs_list holds the
    ObservationBatch at EVERY tick of the episode (starting from the forced
    initial state), returns_to_go[i] is the real discounted reward from
    obs_list[i] onward to termination (so every tick becomes a usable
    training row, not just the first), and outcome is env's
    StepInfo.trial_outcome (mapped like DemonstrationDataset.
    classify_outcome(): "miss" -> "ball_out").

    Only the trainee's OWN attacking box is excluded as a spawn/exit target
    (per _log line: exiting there would already be a "win", not ball_out) --
    the trainee's own defending box is a valid exit, e.g. an own-half
    sideline/goal-line exit while carrying the ball.
    """
    from footballcoach.ai.config import load_ai_config
    from footballcoach.entities.player import Team
    from footballcoach.rules_ai import Phase1RulesAI

    env.reset()
    match = env._loop.match
    player = match.player_by_id(env.trainee_player_id)
    pitch = match.pitch

    # Exit boundary: either sideline (y = +-half_width) or the OWN goal-line
    # (x = -+half_length for the side NOT attacked by this player), chosen
    # uniformly. The attacked (opponent) goal-line is excluded outright --
    # carrying the ball across it is scoring context, not ball_out.
    half_len, half_wid = pitch.half_length, pitch.half_width
    attacks_pos_x = player.team == Team.LEFT
    boundary = rng.choice(["sideline_pos_y", "sideline_neg_y", "own_goal_line"])
    if boundary == "own_goal_line":
        exit_x = -half_len if attacks_pos_x else half_len
        exit_y = rng.uniform(-half_wid * 0.9, half_wid * 0.9)
        out_dir = np.array([-1.0 if attacks_pos_x else 1.0, 0.0])
    else:
        exit_y = half_wid if boundary == "sideline_pos_y" else -half_wid
        exit_x = rng.uniform(-half_len * 0.9, half_len * 0.9)
        out_dir = np.array([0.0, 1.0 if boundary == "sideline_pos_y" else -1.0])

    from footballcoach.engine.movement import effective_top_speed
    t = rng.uniform(0.1, t_max_s)
    # player.attributes.top_speed is a normalized [0,1] SKILL rating, not a
    # real m/s value -- effective_top_speed() converts it (plus stamina and
    # the ball-carry speed penalty, has_ball=True since this player already
    # has possession) into the actual real-world sprint speed in m/s.
    top_speed = effective_top_speed(
        match.movement_params, player.attributes.top_speed, player.stamina,
        has_ball=True, ball_control_attr=player.attributes.ball_control,
    )
    speed = rng.uniform(0.75, 1.0) * top_speed
    # Small lateral jitter so the exit isn't perfectly axis-aligned.
    lateral = np.array([-out_dir[1], out_dir[0]]) * rng.uniform(-0.1, 0.1) * speed
    velocity_xy = out_dir * speed + lateral
    start_xy = np.array([exit_x, exit_y]) - velocity_xy * t

    from footballcoach.mathutils import Vector3
    player.position = Vector3(float(start_xy[0]), float(start_xy[1]), 0.0)
    player.velocity = Vector3(float(velocity_xy[0]), float(velocity_xy[1]), 0.0)
    player.heading_rad = math.atan2(float(velocity_xy[1]), float(velocity_xy[0]))
    player.current_order = None  # let Phase1RulesAI issue a fresh order from this state
    player.ai = Phase1RulesAI()
    match._set_possession(player.player_id)
    match._sync_possessed_ball()  # ball position/velocity now follow the carrier


    # Re-encode the observation AFTER the forced override so it reflects the
    # actual starting state the value net will be scored against.
    obs_list = [env._get_obs()]

    gamma = float(load_ai_config().get("ppo", {}).get("gamma", 0.995))
    rewards = []
    outcome = "unknown"
    done = False
    while not done:
        obs, reward, done, info = env.step()
        rewards.append(reward)
        if not done:
            obs_list.append(obs)
        if done:
            outcome = info.trial_outcome or "unknown"
    if outcome == "miss":
        outcome = "ball_out"
    elif outcome == "goal":
        outcome = "win"

    # returns_to_go[i] = discounted sum of rewards[i:] -- computed via a
    # standard backward pass so every tick's row gets its own correct
    # return, not just the episode's first tick.
    returns_to_go = np.zeros(len(obs_list), dtype=np.float64)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        if i < len(obs_list):
            returns_to_go[i] = running
    return obs_list, returns_to_go, outcome


def _generate_synthetic_ball_out_set(
    n_episodes: int, t_max_s: float = 0.7, seed: int = 0,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Drive N real forced-ball-out-attempt episodes through the ACTUAL
    ScenarioEnv/engine (see _synthetic_ball_out_episode), and logs what
    fraction of attempts landed in each outcome bucket (a forced
    near-boundary trajectory can still end as e.g. "win" if possession is
    challenged, or "timeout" if geometry/physics didn't carry it out in
    time). Keeps rows from BOTH the intended "ball_out" episodes and the
    "win" ones that occur as a side effect -- the latter make a useful
    counter-example set (same forced-fast-carry setup, opposite outcome) for
    checking the value net/LR baseline isn't just learning "fast + near
    boundary -> bad" without regard to which way the carry actually resolved.
    Generated ONCE, before training, so the same fixed sets can be scored
    against both the linear regression baseline and the trained value net.

    Returns (obs_list, returns_arr, outcome_arr) -- every tick of every
    ball_out/win episode is kept (not just each episode's first row);
    outcome_arr holds each row's ball_out/win outcome; empty
    list/arrays if none resolved as either."""
    from footballcoach.ai.curriculum.envs import build_env
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID

    env = build_env(PHASES_BY_ID[1])
    _make_scenario_builds_deterministic(env, seed)
    rng = np.random.default_rng(seed)

    obs_list = []
    return_list = []
    outcome_list = []
    outcome_counts: dict[str, int] = {}
    for _ in range(n_episodes):
        ep_obs_list, ep_returns_to_go, outcome = _synthetic_ball_out_episode(env, rng, t_max_s=t_max_s)
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        if outcome not in ("ball_out", "win"):
            continue
        obs_list.extend(ep_obs_list)
        return_list.extend(ep_returns_to_go.tolist())
        outcome_list.extend([outcome] * len(ep_obs_list))

    n_total = sum(outcome_counts.values())
    log.info(f"--- Synthetic ball-out set: {n_total} attempted episodes, "
             f"t_max_s={t_max_s} ---")
    for name, count in sorted(outcome_counts.items(), key=lambda kv: -kv[1]):
        log.info(f"  outcome={name:<12} n={count:>5}  ({100.0 * count / n_total:.1f}%)")
    log.info(f"  -> {len(obs_list):,} rows kept across ball_out/win episodes ---")

    if not obs_list:
        log.warning("No episodes resolved as ball_out or win -- nothing to score.")
        return [], np.array([], dtype=np.float64), np.array([], dtype=object)
    return obs_list, np.array(return_list, dtype=np.float64), np.array(outcome_list, dtype=object)


def _score_synthetic_ball_out_with_net(
    decision_net, value_net, obs_list: list, returns_arr: np.ndarray,
    outcomes: np.ndarray | None = None,
) -> None:
    """Score the TRAINED value_net against the fixed synthetic set's real
    discounted returns (see _generate_synthetic_ball_out_set), broken down
    per outcome (ball_out vs. win) when *outcomes* is given."""
    if not obs_list:
        return
    self_feat = np.stack([o.self_feat for o in obs_list])
    other_feat = np.stack([o.other_feat for o in obs_list])
    exists_mask = np.stack([o.exists_mask for o in obs_list])
    ball_feat = np.stack([o.ball_feat for o in obs_list])
    global_feat = np.stack([o.global_feat for o in obs_list])
    self_ai_type = np.stack([o.self_ai_type for o in obs_list])
    other_ai_type = np.stack([o.other_ai_type for o in obs_list])

    value_net.eval()
    with torch.no_grad():
        d_heads = decision_net(
            torch.from_numpy(self_feat), torch.from_numpy(other_feat),
            torch.from_numpy(exists_mask), torch.from_numpy(ball_feat),
            torch.from_numpy(global_feat),
        )
        e_heads = value_net(
            torch.from_numpy(self_feat), torch.from_numpy(other_feat),
            torch.from_numpy(exists_mask), torch.from_numpy(ball_feat),
            torch.from_numpy(global_feat), d_heads,
            torch.from_numpy(self_ai_type), torch.from_numpy(other_ai_type),
        )
        pred = e_heads.value.squeeze(-1).numpy()

    ret_std = max(float(returns_arr.std()), 1e-6)
    for label, mask in [
        ("all", np.ones(len(returns_arr), dtype=bool)),
        *([(o, outcomes == o) for o in np.unique(outcomes)] if outcomes is not None else []),
    ]:
        if not mask.any():
            continue
        sq_err = (pred[mask] - returns_arr[mask]) ** 2
        rmse = float(np.sqrt(sq_err.mean()))
        log.info(
            f"  {label} (n={int(mask.sum())}): return mean={returns_arr[mask].mean():+.3f} "
            f"std={returns_arr[mask].std():.3f}  pred mean={pred[mask].mean():+.3f} std={pred[mask].std():.3f}  "
            f"rmse={rmse:.4f} (norm={rmse / ret_std:.4f})"
        )


def _synthetic_timeout_episode(env, rng: np.random.Generator, timeout_s_min: float = 2.0, timeout_s_max: float = 12.0):
    """Run one REGULAR phase-1 episode (normal env.reset() -- no forced
    start state, same opponent-type sampling as real training) except
    env.max_episode_s is overridden for this one episode to a value sampled
    uniformly from [timeout_s_min, timeout_s_max], i.e. a much shorter fuse
    than the real curriculum's ~240s. Gives many more `timeout` outcomes per
    wall-clock second than waiting for the real (rare) long-episode timeouts
    in demonstrations -- everything else (opponent AI, reward function,
    terminal-outcome classification) is the real thing.

    Returns (obs_list, returns_to_go, outcome) -- same shape/semantics as
    _synthetic_ball_out_episode."""
    from footballcoach.ai.config import load_ai_config

    timeout_s = rng.uniform(timeout_s_min, timeout_s_max)
    orig_max_episode_s = env.max_episode_s
    env.max_episode_s = float(timeout_s)
    try:
        obs_list = [env.reset()]
        gamma = float(load_ai_config().get("ppo", {}).get("gamma", 0.995))
        rewards = []
        outcome = "unknown"
        done = False
        while not done:
            obs, reward, done, info = env.step()
            rewards.append(reward)
            if not done:
                obs_list.append(obs)
            if done:
                outcome = info.trial_outcome or "unknown"
    finally:
        env.max_episode_s = orig_max_episode_s
    if outcome == "miss":
        outcome = "ball_out"
    elif outcome == "goal":
        outcome = "win"

    returns_to_go = np.zeros(len(obs_list), dtype=np.float64)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        if i < len(obs_list):
            returns_to_go[i] = running
    return obs_list, returns_to_go, outcome


def _generate_synthetic_timeout_set(
    n_episodes: int, timeout_s_min: float = 2.0, timeout_s_max: float = 12.0, seed: int = 0,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Drive N regular phase-1 episodes with a short randomized
    env.max_episode_s override (see _synthetic_timeout_episode), and log
    the outcome mix. Keeps rows from every episode regardless of outcome
    (timeout is the target case, but win/loss/ball_out attempts that
    resolve before the short fuse fires are kept too, as counter-examples).

    Returns (obs_list, returns_arr, outcome_arr) -- every tick of every
    episode is kept; empty list/arrays if n_episodes == 0."""
    from footballcoach.ai.curriculum.envs import build_env
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID

    env = build_env(PHASES_BY_ID[1])
    _make_scenario_builds_deterministic(env, seed)
    rng = np.random.default_rng(seed)

    obs_list = []
    return_list = []
    outcome_list = []
    outcome_counts: dict[str, int] = {}
    for _ in range(n_episodes):
        ep_obs_list, ep_returns_to_go, outcome = _synthetic_timeout_episode(
            env, rng, timeout_s_min=timeout_s_min, timeout_s_max=timeout_s_max,
        )
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        obs_list.extend(ep_obs_list)
        return_list.extend(ep_returns_to_go.tolist())
        outcome_list.extend([outcome] * len(ep_obs_list))

    n_total = sum(outcome_counts.values())
    log.info(f"--- Synthetic timeout set: {n_total} episodes, "
             f"max_episode_s~U[{timeout_s_min}, {timeout_s_max}] ---")
    for name, count in sorted(outcome_counts.items(), key=lambda kv: -kv[1]):
        log.info(f"  outcome={name:<12} n={count:>5}  ({100.0 * count / n_total:.1f}%)")
    log.info(f"  -> {len(obs_list):,} rows kept across all episodes ---")

    if not obs_list:
        return [], np.array([], dtype=np.float64), np.array([], dtype=object)
    return obs_list, np.array(return_list, dtype=np.float64), np.array(outcome_list, dtype=object)


# ---------------------------------------------------------------------------
# --checkpoint rollout mode: build a DemonstrationDataset-shaped dataset from
# live on-policy rollout instead of pre-recorded rules-AI demonstrations, so
# every diagnostic above (dataset distribution, reward-component breakdown,
# outcome-split returns, linear-regression baseline, worst-episode match log,
# ...) runs completely unmodified against it.
# ---------------------------------------------------------------------------

def _reset_value_head(net) -> None:
    """Reinitialise just ``net.value_head`` and its value-only
    ``value_ai_type_channel`` side-channel in place, discarding whatever the
    loaded checkpoint had there. Used by --reset-value-weights: keeps the
    checkpoint's trunk/encoder features but fits a fresh critic on top of
    them instead of continuing from a possibly-stale/differently-configured
    one. ``net`` may be a plain ``ExecutionNetwork`` or a
    ``CanonicalNetworkWrapper`` around one -- both transparently forward
    ``.value_head``/``.value_ai_type_channel`` attribute access."""
    from footballcoach.ai.models.execution_network import ExecutionNetwork

    fresh = ExecutionNetwork.from_config()
    net.value_head.load_state_dict(fresh.value_head.state_dict())
    net.value_ai_type_channel.load_state_dict(fresh.value_ai_type_channel.state_dict())


def _reset_dir_log_std(net) -> None:
    """Reinitialise ``net.move_dir_log_std``/``net.kick_dir_log_std`` (the
    learned direction-head exploration std -- see ai_trainer_knowledge.md
    "Direction heads: log_std and KL") back to their ``ai_config.json``
    ``ppo.dir_log_std_init``/``ppo.kick_dir_log_std_init`` values, in place.

    Unlike ``_reset_value_head()``, this affects ACTION SAMPLING, not just
    the critic -- a checkpoint whose direction std has collapsed toward
    near-deterministic sampling (small std) would otherwise bias
    --checkpoint rollout collection away from the exploration noise real
    training actually used, which matters if you're trying to reproduce
    "what did on-policy collection look like during training" rather than
    "what does this checkpoint do at (near-)greedy inference". Used by
    --reset-dir-log-std. ``net`` may be a plain ``ExecutionNetwork`` or a
    ``CanonicalNetworkWrapper`` around one -- both transparently forward
    ``.move_dir_log_std``/``.kick_dir_log_std`` attribute access. Must be
    called BEFORE rollout collection (unlike the value-head reset, which
    only matters for the later fitting stage) since it changes how actions
    are actually sampled."""
    from footballcoach.ai.models.execution_network import ExecutionNetwork

    fresh = ExecutionNetwork.from_config()
    net.move_dir_log_std.data.copy_(fresh.move_dir_log_std.data)
    net.kick_dir_log_std.data.copy_(fresh.kick_dir_log_std.data)


def _rollout_label_fn(env):
    """Like ``bc.phase1_labels(env)``, but with ``ai_type``/``opponent_ai_type``
    corrected for a NEURAL trainee -- ``phase1_labels()`` only ever
    distinguishes "rules" vs "not rules" (its docstring: no neural
    demo-recording mode existed when it was written), so it would otherwise
    tag a neural-controlled player as AI_TYPE_IMMOBILE, which silently drops
    every row from ``DemonstrationDataset.valid_indices()`` (excludes
    self.ai_type==IMMOBILE). Self is always the checkpoint's neural policy in
    rollout mode; the opponent may genuinely be rules/immobile/neural
    (shared-weight self-play) -- read the real per-episode flags the env
    already tracks (``match._opponent_use_rules_ai``/``_opponent_is_immobile``,
    the same attributes ``StepInfo.is_rules_episode``/``is_immobile_episode``
    are sourced from) rather than trusting ``phase1_labels()``'s isinstance
    check, which doesn't know about ``NeuralPlayerAI``. Everything else
    (decision heads, kick/tackle execution labels, move direction/region) is
    unaffected -- those already come from real physical player state
    (``Player.kicked_this_tick`` etc.), not from the ai_type fields."""
    from footballcoach.ai.ppo.bc import (
        AI_TYPE_IMMOBILE, AI_TYPE_NEURAL, AI_TYPE_RULES, phase1_labels,
    )

    label = phase1_labels(env)
    label.ai_type = AI_TYPE_NEURAL
    match = env.match
    opp_rules = getattr(match, "_opponent_use_rules_ai", False)
    opp_immobile = getattr(match, "_opponent_is_immobile", False)
    label.opponent_ai_type = (
        AI_TYPE_RULES if opp_rules else AI_TYPE_IMMOBILE if opp_immobile else AI_TYPE_NEURAL
    )
    return label


def _make_scenario_builds_deterministic(env, base_seed: int) -> None:
    """Wrap ``env.definition.build`` so every scenario reset (episode start)
    over this env's lifetime is fully deterministic for a given
    ``base_seed`` -- each call gets ``base_seed + call_index``, so rerunning
    with the same ``--seed`` reproduces the identical sequence of episodes/
    rewards.

    Without this, ``build_1v1_scenario()`` (see its own ``seed`` kwarg,
    normally only ever threaded through by ``ai/eval/seeded_eval.py`` for
    evaluation) draws from a fresh, UNSEEDED ``random.Random()`` every
    episode -- ``torch.manual_seed()``/``np.random.seed()`` at the top of
    ``main()`` never reach it, since it's an entirely separate RNG stream.
    That's why every run of this script produced different placements/
    rewards despite a fixed ``--seed``: only incidental torch ops (action
    sampling noise) were actually seeded, not the scenario itself.

    ``ScenarioEnv.reset()`` rebuilds a fresh ``ScenarioLoop`` (and therefore
    calls ``definition.build(...)`` again) on every call, so wrapping the
    ``build`` callable once, right after constructing ``env``, is enough to
    cover every episode for that env's whole lifetime -- no per-reset
    plumbing needed. ``ScenarioDefinition`` is a plain (non-frozen)
    dataclass, so replacing its ``.build`` attribute in place is safe."""
    _orig_build = env.definition.build
    _next_seed = [base_seed]

    def _seeded_build(*args, **kwargs):
        kwargs.setdefault("seed", _next_seed[0])
        _next_seed[0] += 1
        return _orig_build(*args, **kwargs)

    env.definition.build = _seeded_build


def _collect_rollout_arrays(
    trainer, env, n_steps: int, progress_prefix: str = "", progress_live: bool = True,
    shared_progress_counter=None, progress_milestone_pct: int = 10,
) -> dict:
    """Drive *env* with *trainer*'s (already checkpoint-loaded) policy for
    *n_steps* decision steps, collecting ONLY the trainee's own transitions
    (no secondary/self-play-opponent rows -- keeps one row == one player
    perspective, matching every other array in this dict) into a
    ``RolloutBuffer``, then repack it into the same flat numpy arrays
    ``DemonstrationDataset``'s constructor expects.

    ``progress_prefix``/``progress_live`` are forwarded to
    ``ai/progress.py``'s ``ProgressReporter`` -- ``progress_live=True`` (the
    single-process default) renders an in-place bar. When
    ``shared_progress_counter`` is given (a ``multiprocessing.Value``, see
    ``_rollout_worker_entry``), this call SKIPS its own per-step rendering
    (``progress_prefix``/``progress_live`` are then unused) and instead just
    increments the shared counter every step -- the caller (main process)
    is expected to poll that counter and render ONE aggregate bar across all
    workers, rather than every worker independently printing its own
    ticking output onto a terminal they all share. Either way,
    ``progress.finish()`` still prints this call's own one-line summary at
    the end (total time / ms-per-step / s-per-episode) -- that part is
    cheap, self-contained (one line, newline-terminated), and useful
    per-worker diagnostic detail, so it's kept even in shared-counter mode.

    Returns a dict with keys ``self_feat``/``other_feat``/``exists_mask``/
    ``ball_feat``/``global_feat``/``bc_labels``/``rewards``/``dones``/
    ``reward_components``/``reward_component_keys``/``episode_outcomes``/
    ``n_dropped``.
    """
    from footballcoach.ai.progress import ProgressReporter
    from footballcoach.ai.ppo.ppo_trainer import REWARD_COMP_LABELS, _action_to_numpy
    from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer

    _comp_key_order = [k for k, _ in REWARD_COMP_LABELS]

    env.sample_action_fn = trainer._sample_action
    env.reset()
    buffer = RolloutBuffer()
    episode_outcomes: list[str] = []
    progress = ProgressReporter(
        n_steps, prefix=progress_prefix, live=progress_live, milestone_pct=progress_milestone_pct,
    )

    collected = 0
    while collected < n_steps:
        next_obs, reward, done, info = env.step()
        tr = env.last_trainee_transition
        if tr is None:
            if done:
                env.reset()
            continue
        buffer.add(
            obs=tr["obs"],
            action=_action_to_numpy(tr["action"], tr["raw_exec"]),
            log_prob=float(tr["log_prob"]),
            value=float(tr["value"]),
            reward=reward,
            done=1.0 if done else 0.0,
            bc_label=_rollout_label_fn(env).to_array(),
            reward_comps=dict(getattr(env, "last_reward_components", {})),
            step_outcome=(info.trial_outcome or "") if (done and info is not None) else "",
        )
        collected += 1
        if shared_progress_counter is not None:
            # A Manager Value proxy (not a raw ctx.Value -- see
            # _build_rollout_dataset's docstring comment for why) has no
            # get_lock(); += is a GET+SET pair of separate RPCs to the
            # manager process, so a rare lost increment under a race is
            # possible in principle. Harmless here -- this only feeds a
            # cosmetic aggregate progress bar, not anything correctness-
            # sensitive.
            shared_progress_counter.value += 1
        else:
            progress.update(collected)
        if done:
            if info is not None and info.trial_outcome is not None:
                episode_outcomes.append(info.trial_outcome)
            env.reset()
    progress.finish(collected, n_episodes=len(episode_outcomes))

    n_dropped = buffer.truncate_to_last_episode_end()
    if not buffer.obs:
        return {
            "self_feat": np.zeros((0,), dtype=np.float32), "n_dropped": n_dropped,
            "episode_outcomes": [], "reward_component_keys": _comp_key_order,
        }

    reward_components = np.array(
        [[rc.get(k, 0.0) for k in _comp_key_order] for rc in buffer.reward_comps],
        dtype=np.float32,
    )
    return {
        "self_feat":   np.stack([o["self_feat"] for o in buffer.obs]),
        "other_feat":  np.stack([o["other_feat"] for o in buffer.obs]),
        "exists_mask": np.stack([o["exists_mask"] for o in buffer.obs]),
        "ball_feat":   np.stack([o["ball_feat"] for o in buffer.obs]),
        "global_feat": np.stack([o["global_feat"] for o in buffer.obs]),
        "bc_labels":   np.stack(buffer.bc_labels).astype(np.float32),
        "rewards":     np.array(buffer.rewards, dtype=np.float32),
        "dones":       np.array(buffer.dones, dtype=np.float32),
        "reward_components": reward_components,
        "reward_component_keys": _comp_key_order,
        "episode_outcomes": episode_outcomes,
        "n_dropped": n_dropped,
    }


def _rollout_worker_entry(
    checkpoint_path: str, phase_id: int, n_steps: int, seed: int, separate_value_net: bool,
    worker_torch_threads: int, worker_idx: int, shared_progress_counter=None,
    reset_dir_log_std: bool = False,
) -> dict:
    """Picklable top-level entry point for one parallel rollout-collection
    worker (see --n-parallel-envs). Each worker independently loads the
    checkpoint and builds its own ScenarioEnv -- no shared memory/batched
    inference, same rationale as ``ai/ppo/rollout_worker.py``'s PPO training
    workers (physics stepping is pure-Python/CPU-bound and serialized by the
    GIL within one process; real OS processes are needed for parallel
    speedup).

    ``worker_torch_threads`` mirrors ``ai/ppo/rollout_worker.py``'s
    ``_worker_main`` (``ppo.worker_torch_threads`` in ai_config.json, see
    --worker-torch-threads): without capping this, each worker process
    defaults to torch's full-core intra-op thread pool, so N workers
    oversubscribe the machine N-fold instead of getting clean per-worker
    parallelism -- this is the single most load-bearing knob for whether
    --n-parallel-envs actually speeds anything up. The other, BLAS-side half
    of that same fix (OMP_NUM_THREADS etc.) lives at the very top of this
    FILE, not here -- see the comment above the `import numpy as np` line
    for why it has to be that early to actually take effect for spawned
    workers."""
    import random

    import torch

    from footballcoach.ai.curriculum.envs import build_env
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

    torch.set_num_threads(max(1, worker_torch_threads))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    trainer = PPOTrainer.from_config(
        device=torch.device("cpu"), inference_only=True, separate_value_net=separate_value_net,
    )
    trainer.load_checkpoint(Path(checkpoint_path))
    if reset_dir_log_std:
        _reset_dir_log_std(trainer.execution_net)
    env = build_env(PHASES_BY_ID[phase_id])
    _make_scenario_builds_deterministic(env, seed)
    # Per-worker subdir: each worker's own episode-index counter restarts at
    # 0, so a shared dir would risk filename collisions across workers.
    env.match_log_dir = Path("results/match_logs") / f"worker_{worker_idx}"
    return _collect_rollout_arrays(
        trainer, env, n_steps,
        progress_prefix=f"[worker {worker_idx}] ", progress_live=False,
        shared_progress_counter=shared_progress_counter,
    )


def _build_rollout_dataset(trainer, checkpoint_path: str, phase_id: int, rollout_steps: int,
                            n_parallel_envs: int, seed: int, separate_value_net: bool,
                            worker_torch_threads: int, reset_dir_log_std: bool = False,
                            progress_milestone_pct: int = 10):
    """Collect on-policy rollout data with the CHECKPOINT's policy (not a
    rules-based demo dataset) and pack it into a real ``DemonstrationDataset``
    so every existing diagnostic in this script runs unmodified against it.

    ``trainer`` (already checkpoint-loaded in the main process, with
    --reset-dir-log-std already applied by the caller if requested) is
    reused directly for single-process collection; the parallel path
    ignores it and has each worker load its own copy from
    ``checkpoint_path`` instead (see ``_rollout_worker_entry``), applying
    ``reset_dir_log_std`` independently in each one. ``worker_torch_threads``
    only applies to the parallel path -- single-process collection already
    runs under whatever ``torch.set_num_threads()`` the main process set at
    startup."""
    from footballcoach.ai.bc.dataset import DemonstrationDataset

    if n_parallel_envs > 1:
        import concurrent.futures
        import multiprocessing

        steps_per_worker = max(1, rollout_steps // n_parallel_envs)
        log.info(f"Collecting rollout: {n_parallel_envs} parallel worker(s), "
                 f"~{steps_per_worker} steps/worker, worker_torch_threads={worker_torch_threads}")
        # mp_context="spawn" (not this platform's "fork" default): each
        # worker must be a genuinely fresh interpreter that hasn't imported
        # numpy/torch yet, so _rollout_worker_entry's BLAS-thread env vars
        # (see its docstring) actually take effect -- a forked child would
        # inherit the parent's already-initialized BLAS thread pool instead.
        from footballcoach.ai.progress import ProgressReporter

        _wall_start = time.monotonic()
        _ctx = multiprocessing.get_context("spawn")
        # Shared across all workers (not one per worker): each worker just
        # increments this by 1 per step under its lock, so the main process
        # can render ONE aggregate live bar across all of them instead of
        # every worker printing its own ticking output onto a terminal they
        # all share (see _collect_rollout_arrays' shared_progress_counter
        # docstring). Each worker still prints its own one-line FINAL
        # summary (total time / ms-per-step / s-per-episode) once it's done.
        #
        # A plain ctx.Value can only be shared with a process at CREATION
        # time (via Process(args=...) inheritance) -- ProcessPoolExecutor's
        # workers are long-lived and receive each task's arguments later,
        # through an internal Queue, which can't pickle a raw Synchronized
        # object at all ("Synchronized objects should only be shared between
        # processes through inheritance"). A SyncManager's Value is a proxy
        # object (talks to a separate manager server process over a real
        # connection) and CAN be pickled through that queue -- the per-step
        # increment now costs one small IPC round-trip instead of a direct
        # memory write, but that's negligible next to a full env.step().
        _manager = _ctx.Manager()
        _progress_counter = _manager.Value("l", 0)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_parallel_envs, mp_context=_ctx,
        ) as pool:
            futures = [
                pool.submit(_rollout_worker_entry, checkpoint_path, phase_id,
                            steps_per_worker, seed + i, separate_value_net, worker_torch_threads, i,
                            _progress_counter, reset_dir_log_std)
                for i in range(n_parallel_envs)
            ]
            _agg_progress = ProgressReporter(
                steps_per_worker * n_parallel_envs,
                prefix=f"Rollout ({n_parallel_envs} workers): ", live=True,
                milestone_pct=progress_milestone_pct,
            )
            while True:
                _, not_done = concurrent.futures.wait(futures, timeout=0.2)
                _agg_progress.update(int(_progress_counter.value))
                if not not_done:
                    break
            results = [f.result() for f in futures]
        _wall_elapsed = time.monotonic() - _wall_start
        results = [r for r in results if len(r["self_feat"]) > 0]
        if not results:
            raise SystemExit("Rollout collected 0 usable steps across all workers -- "
                              "try more --rollout-steps.")
        _n_steps_total = sum(len(r["self_feat"]) for r in results)
        log.info(
            f"  [parallel rollout] total: {_wall_elapsed:.1f}s wall  "
            f"({1000.0 * _wall_elapsed / max(_n_steps_total, 1):.2f} ms/step aggregate, "
            f"{_n_steps_total / max(_wall_elapsed, 1e-9):.1f} steps/s aggregate across "
            f"{n_parallel_envs} worker(s))"
        )
        n_dropped_total = sum(r["n_dropped"] for r in results)
        if n_dropped_total:
            log.info(f"Dropped {n_dropped_total} trailing (incomplete-episode) "
                      f"step(s) across workers")
        merged = {
            "self_feat":   np.concatenate([r["self_feat"] for r in results]),
            "other_feat":  np.concatenate([r["other_feat"] for r in results]),
            "exists_mask": np.concatenate([r["exists_mask"] for r in results]),
            "ball_feat":   np.concatenate([r["ball_feat"] for r in results]),
            "global_feat": np.concatenate([r["global_feat"] for r in results]),
            "bc_labels":   np.concatenate([r["bc_labels"] for r in results]),
            "rewards":     np.concatenate([r["rewards"] for r in results]),
            "dones":       np.concatenate([r["dones"] for r in results]),
            "reward_components": np.concatenate([r["reward_components"] for r in results]),
            "reward_component_keys": results[0]["reward_component_keys"],
            "episode_outcomes": [o for r in results for o in r["episode_outcomes"]],
        }
    else:
        from footballcoach.ai.curriculum.envs import build_env
        from footballcoach.ai.curriculum.phases import PHASES_BY_ID

        env = build_env(PHASES_BY_ID[phase_id])
        _make_scenario_builds_deterministic(env, seed)
        env.match_log_dir = Path("results/match_logs")
        log.info(f"Collecting rollout: single process, {rollout_steps} steps")
        merged = _collect_rollout_arrays(
            trainer, env, rollout_steps, progress_prefix="Rollout: ",
            progress_milestone_pct=progress_milestone_pct,
        )
        if len(merged["self_feat"]) == 0:
            raise SystemExit("Rollout collected 0 usable steps -- try more --rollout-steps.")
        if merged["n_dropped"]:
            log.info(f"Dropped {merged['n_dropped']} trailing (incomplete-episode) step(s)")

    ds = DemonstrationDataset(
        obs_self_feat=merged["self_feat"],
        obs_other_feat=merged["other_feat"],
        obs_exists_mask=merged["exists_mask"],
        obs_ball_feat=merged["ball_feat"],
        obs_global_feat=merged["global_feat"],
        bc_labels=merged["bc_labels"],
        rewards=merged["rewards"],
        dones=merged["dones"],
        reward_components=merged["reward_components"],
        reward_component_keys=merged["reward_component_keys"],
        episode_outcomes=merged["episode_outcomes"],
    )
    # Rollout rows are one-per-real-step, this player's own perspective only
    # (no paired secondary-player row) -- unlike .npz demo recordings, which
    # always append trainee+opponent together and mark BOTH done=1 at an
    # episode boundary (see DemonstrationDataset._DONE_ROWS_PER_EPISODE_BOUNDARY's
    # docstring). Override per-instance so episode-boundary detection expects
    # exactly ONE done=1 row per boundary here, not two.
    ds._DONE_ROWS_PER_EPISODE_BOUNDARY = 1
    log.info(f"Rollout dataset: {len(ds):,} steps, {ds.n_episodes()} complete episode(s)")
    return ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None,
                        help="Directory of .npz demonstration files (default: "
                             "demonstrations/phase_1_debug/ unless --checkpoint is given). "
                             "Mutually exclusive with --checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a trained checkpoint_*.pt. When given, skips the demo "
                             "dataset entirely and instead collects fresh on-policy rollout "
                             "data by driving the ACTUAL loaded decision_net/execution_net "
                             "through the real ScenarioEnv (phase 1) -- i.e. a live "
                             "value-pretrain-style run, not a fit against pre-recorded rules-AI "
                             "demonstrations. decision_net/execution_net (and value_net, if the "
                             "checkpoint used --separate-value-net) are loaded from the "
                             "checkpoint rather than freshly initialised; the same freezing "
                             "convention as PPOTrainer.pretrain_value() is used for fitting "
                             "(see --reset-value-weights/--reset-separate-value-net below). "
                             "Mutually exclusive with --data.")
    parser.add_argument("--reset-value-weights", action="store_true", default=False,
                        help="Only with --checkpoint. Reinitialise just value_head (and its "
                             "value_ai_type_channel side-channel) after loading the checkpoint, "
                             "before fitting -- keeps the loaded trunk/encoder features but "
                             "discards whatever critic weights the checkpoint had, so fitting "
                             "starts from a fresh value head instead of continuing from a "
                             "possibly-stale one.")
    parser.add_argument("--reset-separate-value-net", action="store_true", default=False,
                        help="Only with --checkpoint, and only meaningful if the checkpoint was "
                             "trained with --separate-value-net (warns and no-ops otherwise). "
                             "Ignores the checkpoint's saved value_net weights entirely and "
                             "constructs a fresh one instead (--trunk-hidden/--value-hidden-dim/"
                             "--entity-embed-dim capacity overrides are allowed again in this "
                             "case) -- still uses the checkpoint's decision_net for context.")
    parser.add_argument("--reset-dir-log-std", action="store_true", default=False,
                        help="Only with --checkpoint. Reinitialise move_dir_log_std/"
                             "kick_dir_log_std (the learned direction-head exploration std) back "
                             "to ai_config.json's ppo.dir_log_std_init/kick_dir_log_std_init "
                             "before COLLECTING the rollout (not just before fitting, unlike "
                             "--reset-value-weights) -- unlike the value-head reset, this changes "
                             "action sampling itself. Useful if a checkpoint's direction std has "
                             "collapsed toward near-deterministic sampling, which would otherwise "
                             "bias --checkpoint rollout collection away from the exploration "
                             "noise real training actually used. Applied independently in every "
                             "parallel worker too (each worker loads its own copy of the "
                             "checkpoint).")
    parser.add_argument("--rollout-steps", type=int, default=4096,
                        help="Only with --checkpoint. Number of on-policy decision steps to "
                             "collect before fitting (mirrors ppo.rollout_steps/"
                             "bc.value_pretrain_steps). Default 4096.")
    parser.add_argument("--n-parallel-envs", type=int, default=1,
                        help="Only with --checkpoint. Number of subprocess workers used to "
                             "collect the rollout in parallel (mirrors ppo.n_parallel_envs). "
                             "Each worker independently loads the checkpoint and runs its own "
                             "ScenarioEnv -- no shared memory/batched inference. Default 1 "
                             "(single process).")
    parser.add_argument("--worker-torch-threads", type=int, default=None,
                        help="Only with --checkpoint and --n-parallel-envs > 1. torch intra-op "
                             "thread cap applied inside EACH parallel rollout worker (mirrors "
                             "ppo.worker_torch_threads / ai/ppo/rollout_worker.py's "
                             "_worker_main). Default: ai_config.json's ppo.worker_torch_threads "
                             "(1 if unset there). Without a cap, every worker defaults to "
                             "torch's full-core thread pool, so N workers oversubscribe the "
                             "machine N-fold instead of parallelising cleanly -- this is usually "
                             "the first thing to check if --n-parallel-envs isn't actually "
                             "speeding up collection.")
    parser.add_argument("--progress-milestone-pct", type=int, default=10,
                        help="Percentage granularity of rollout progress lines when NOT writing "
                             "to a live terminal (auto-detected -- e.g. when piped to `tee`, "
                             "redirected to a file, or running non-interactively). Has no effect "
                             "on an interactive terminal, which always gets the live in-place "
                             "\\r-updating bar regardless of this value. Default 10 (a line every "
                             "10%%). Raise (e.g. 25 or 50) to shrink log file size further.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Adam weight_decay for the fresh value network's optimizer (default: 0.0, disabled).")
    parser.add_argument("--patience", type=int, default=2,
                        help="Early-stop patience: consecutive non-improving epochs (val "
                             "normalized MSE) allowed before stopping early (default: 2). "
                             "Set 0 to disable early stopping and always run all --epochs.")
    parser.add_argument("--batch-size", type=int, default=1200)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-immobile-self", action="store_true", default=False,
                        help="Include rows where self.ai_type==immobile (default: excluded, "
                             "matching DemonstrationDataset.valid_indices()'s convention used "
                             "by every real BC/value training path -- the immobile bot never "
                             "acts, so its own rows are not useful value-fitting examples).")
    parser.add_argument("--worst-episode-log-path", type=str,
                        default="results/debug_value_worst_episode.json",
                        help="Naming template for the synthetic match logs of the "
                             "highest-residual val episode PER OUTCOME (one file each, "
                             "e.g. results/debug_value_worst_episode_ball_out.json).")
    parser.add_argument("--outcome-reweight", action="store_true", default=False,
                        help="Weight the TRAIN loss per-row by inverse outcome frequency "
                             "(computed over train_idx only), so rare outcomes (loss/timeout/"
                             "ball_out/invalid) contribute as much total gradient as 'win' rows. "
                             "Val loss stays unweighted so it remains a fair, comparable metric. "
                             "Diagnostic only -- with very few rare-outcome rows this mostly "
                             "measures overfitting risk, it does not manufacture real signal.")
    parser.add_argument("--outcome-reweight-max", type=float, default=20.0,
                        help="Cap on the normalized per-row outcome weight (default 20x the "
                             "mean-weight row). Uncapped inverse-frequency weights blow up when "
                             "an outcome has only a handful of rows (e.g. 5 'loss' rows in "
                             "146k) -- a single such row can then get weight in the thousands "
                             "and dominate a minibatch's gradient/WLS solution outright, which "
                             "looks like (and is) training collapse, not a computation bug. "
                             "Only takes effect with --outcome-reweight.")
    parser.add_argument("--synthetic-ball-out-episodes", type=int, default=0,
                        help="If > 0, after training run this many real ScenarioEnv episodes "
                             "with the trainee forced into a straight-line carry off the pitch "
                             "(see _synthetic_ball_out_episode), then score the TRAINED value "
                             "net's RMSE against the real discounted returns of whichever "
                             "attempts actually resolved as ball_out. Isolates whether the "
                             "value net can fit clean/unambiguous ball_out examples, vs. the "
                             "real dataset's sparse, noisy ball_out rows. 0 (default) = skip.")
    parser.add_argument("--synthetic-ball-out-t-max-s", type=float, default=0.7,
                        help="Max time-to-exit-boundary (seconds) sampled for the forced "
                             "carry (uniform in [0.1, this]). Default 0.7.")
    parser.add_argument("--synthetic-timeout-episodes", type=int, default=0,
                        help="If > 0, run this many REGULAR phase-1 episodes (real "
                             "env.reset(), real opponent sampling) with env.max_episode_s "
                             "overridden per-episode to a random value in "
                             "[--synthetic-timeout-s-min, --synthetic-timeout-s-max] (see "
                             "_synthetic_timeout_episode), to generate many more `timeout` "
                             "outcomes than the real dataset has. Rows are folded into both "
                             "the linear-regression baseline and the value net's training "
                             "batches, same as --synthetic-ball-out-episodes. 0 (default) = skip.")
    parser.add_argument("--synthetic-timeout-s-min", type=float, default=2.0,
                        help="Lower bound (seconds) for the randomized max_episode_s override. Default 2.0.")
    parser.add_argument("--synthetic-timeout-s-max", type=float, default=12.0,
                        help="Upper bound (seconds) for the randomized max_episode_s override. Default 12.0.")
    parser.add_argument("--trunk-hidden", type=int, default=None,
                        help="Override the fresh value_net's trunk_hidden (default: "
                             "network.exec_trunk_hidden from ai_config.json, same as production's "
                             "--separate-value-net critic). Capacity sweep knob -- if a bigger trunk "
                             "gets meaningfully lower val RMSE than the default, the current size "
                             "was likely underfitting, not just data/target-noise limited.")
    parser.add_argument("--value-hidden-dim", type=int, default=None,
                        help="Override the fresh value_net's value_head hidden layer width "
                             "(default: network.value_hidden_dim from ai_config.json). 0 = single "
                             "linear layer, no hidden layer at all. Capacity sweep knob, independent "
                             "of --trunk-hidden -- widens only the final value_head, not the shared trunk.")
    parser.add_argument("--entity-embed-dim", type=int, default=None,
                        help="Override the fresh value_net's OWN entity encoder embedding dim "
                             "(default: network.entity_embed_dim from ai_config.json). Free to change "
                             "independently of decision_net's entity encoder since value_net never "
                             "shares one here (see debug_value_network.py's frozen decision_net "
                             "note). Must be divisible by the network's attention head counts "
                             "(network.num_attention_heads / inter_player_attn_heads in "
                             "ai_config.json) or construction will raise.")
    args = parser.parse_args()

    if args.checkpoint and args.data:
        log.warning(
            "--checkpoint and --data both given: using --data as the dataset (no live "
            "rollout will be collected) and --checkpoint only to initialize decision_net/"
            "value_net's weights -- i.e. warm-starting value fitting from a trained "
            "checkpoint against a pre-recorded dataset instead of a fresh rollout."
        )
    if not args.checkpoint and not args.data:
        args.data = "demonstrations/phase_1_debug/"
    if args.checkpoint:
        _capacity_override_given = any(
            v is not None for v in (args.trunk_hidden, args.value_hidden_dim, args.entity_embed_dim)
        )
        if _capacity_override_given and not args.reset_separate_value_net:
            raise SystemExit(
                "--trunk-hidden/--value-hidden-dim/--entity-embed-dim change the fresh "
                "network's shape, which only exists when fitting a genuinely fresh "
                "value_net -- with --checkpoint, that's only the --reset-separate-value-net "
                "case (a fresh replacement for the checkpoint's separate value_net). "
                "Every other --checkpoint path loads real checkpoint weights, whose shape "
                "must match ai_config.json exactly; drop these overrides or add "
                "--reset-separate-value-net."
            )
    else:
        if args.reset_value_weights or args.reset_separate_value_net or args.reset_dir_log_std:
            raise SystemExit("--reset-value-weights/--reset-separate-value-net/"
                              "--reset-dir-log-std only apply with --checkpoint.")

    from footballcoach.ai.config import load_ai_config

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(int(load_ai_config().get("ppo", {}).get("main_process_torch_threads", 4)))

    from footballcoach.ai.bc.dataset import DemonstrationDataset
    from footballcoach.ai.models.decision_network import DecisionNetwork
    from footballcoach.ai.models.execution_network import ExecutionNetwork

    ckpt_trainer = None
    if args.checkpoint:
        from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

        _ckpt_peek = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        _ckpt_separate_value_net = "value_net" in _ckpt_peek
        del _ckpt_peek
        log.info(f"Checkpoint {args.checkpoint}: separate_value_net="
                 f"{_ckpt_separate_value_net} (auto-detected)")
        ckpt_trainer = PPOTrainer.from_config(
            device=torch.device("cpu"), inference_only=True,
            separate_value_net=_ckpt_separate_value_net,
        )
        ckpt_trainer.load_checkpoint(Path(args.checkpoint))
        if args.reset_dir_log_std:
            # Must happen BEFORE rollout collection (below) -- this changes
            # action sampling itself, not just the later fitting stage (see
            # _reset_dir_log_std's docstring). The single-process collection
            # path reuses ckpt_trainer.execution_net directly, so resetting
            # it here covers that case; the parallel path re-applies this
            # independently inside each worker (see _rollout_worker_entry),
            # since each worker loads its own separate copy of the checkpoint.
            _reset_dir_log_std(ckpt_trainer.execution_net)
            log.info("--reset-dir-log-std: reset move_dir_log_std/kick_dir_log_std to "
                     "config init values before rollout collection.")
        if args.data:
            # --data given alongside --checkpoint: use the pre-recorded dataset
            # instead of collecting a fresh rollout -- ckpt_trainer is still
            # built above and used below purely to initialize decision_net/
            # value_net's weights (warm start), same as the checkpoint-only
            # path, just without the (often much slower) rollout-collection
            # step. --rollout-steps/--n-parallel-envs/--progress-milestone-pct
            # are unused in this combination since no rollout is collected.
            ds = DemonstrationDataset.from_directory(args.data)
        else:
            _worker_torch_threads = (
                args.worker_torch_threads if args.worker_torch_threads is not None
                else int(load_ai_config().get("ppo", {}).get("worker_torch_threads", 1))
            )
            ds = _build_rollout_dataset(
                ckpt_trainer, args.checkpoint, phase_id=1, rollout_steps=args.rollout_steps,
                n_parallel_envs=args.n_parallel_envs, seed=args.seed,
                separate_value_net=_ckpt_separate_value_net,
                worker_torch_threads=_worker_torch_threads,
                reset_dir_log_std=args.reset_dir_log_std,
                progress_milestone_pct=args.progress_milestone_pct,
            )
    else:
        ds = DemonstrationDataset.from_directory(args.data)
    log.info(f"Loaded {len(ds):,} rows total")
    log.info(f"has_rewards={ds.has_rewards}")

    valid_idx = ds.valid_indices()
    log.info(f"valid_indices(): {len(valid_idx):,} rows "
              f"({100.0 * len(valid_idx) / len(ds):.1f}% of total)")

    # classify_outcome()/row_outcomes() report the ground-truth episode
    # outcome (win/loss/ball_out/invalid/timeout) FROM THE TRAINEE'S
    # PERSPECTIVE ONLY -- e.g. "win" means the trainee reached the box, not
    # whichever player a given row's self_feat happens to describe.
    # valid_idx (used for value-net TRAINING, intentionally) includes BOTH
    # the trainee's own rows AND a non-immobile secondary player's own rows
    # in the same episode -- for the secondary player, a "win" episode was
    # actually THEIR loss. Any diagnostic that groups by outcome must
    # therefore restrict to the trainee's own rows (is_trainee==1) or it
    # silently mixes the trainee's winning returns with the losing
    # opponent's returns under the same "win" bucket, producing nonsense
    # like a negative min inside the "win" outcome. Training itself is
    # unaffected -- it still uses valid_idx (both perspectives), which is
    # correct there since self_feat is genuinely self-relative.
    trainee_valid_idx = valid_idx[ds._is_trainee[valid_idx] > 0.5]

    returns = ds.compute_returns(gamma=args.gamma)
    log.info(f"Returns over ALL rows: mean={returns.mean():.3f} std={returns.std():.3f} "
              f"min={returns.min():.3f} max={returns.max():.3f}")
    returns_valid = returns[valid_idx]
    log.info(f"Returns over valid_indices(): mean={returns_valid.mean():.3f} "
              f"std={returns_valid.std():.3f}")

    _log_dataset_distribution(ds, valid_idx, returns, trainee_valid_idx=trainee_valid_idx)
    _log_reward_component_breakdown(ds, trainee_valid_idx)
    if ds.has_reward_components:
        for _outc in ("win", "loss", "ball_out", "invalid", "timeout"):
            _log_reward_component_breakdown(ds, trainee_valid_idx, outcome_filter=_outc)
    _log_returns_by_outcome(ds, trainee_valid_idx, returns, label="trainee's own valid rows")
    _log_episode_total_reward_by_outcome(ds, trainee_valid_idx, label="trainee's own valid rows")

    valid_only = not args.include_immobile_self
    train_idx, val_idx = ds.split_train_val_indices(val_frac=args.val_frac, valid_only=valid_only)
    n_train_eps = ds.n_episodes(train_idx) if len(train_idx) else 0
    n_val_eps = ds.n_episodes(val_idx) if len(val_idx) else 0
    log.info(f"Train/val split (valid_only={valid_only}): "
              f"{len(train_idx):,} train rows across {n_train_eps} episodes  |  "
              f"{len(val_idx):,} val rows across {n_val_eps} episodes")
    if len(val_idx) == 0:
        raise SystemExit("No val rows -- dataset needs >= 2 complete episodes. "
                          "Record more episodes; lowering --val-frac won't help; "
                          "check `dones`.")

    # Per-row outcome lookup (see DemonstrationDataset.outcome_by_row()),
    # cached over the FULL dataset (not train_idx/val_idx, which may be
    # shuffled/non-contiguous) so any later row chunk can be indexed
    # directly into it regardless of iteration order. Requires ground-truth
    # episode outcomes (re-record demonstrations if this dataset predates
    # meta_episode_outcomes) -- falls back to an all-"n/a" lookup rather
    # than raising, so the rest of this script's diagnostics still run.
    if ds.has_episode_outcomes:
        outcome_by_row = ds.outcome_by_row()
    else:
        log.warning("Dataset has no ground-truth episode outcomes -- "
                    "per-outcome value-loss breakdown will be skipped "
                    "(re-record with the updated record_demonstrations.py).")
        outcome_by_row = np.full(len(ds), "n/a", dtype=object)

    # Inverse-frequency weight per outcome, computed over train_idx only so
    # val's loss (and its early-stop signal) stays unweighted/comparable.
    # Normalized to mean 1.0 over train rows so the overall loss scale (and
    # therefore --lr) is unaffected -- only the relative per-row weighting
    # changes, not the total gradient magnitude for a "flat" outcome mix.
    # Shared by both the linear-regression baseline below and the value net.
    outcome_weight_by_row: np.ndarray | None = None
    if args.outcome_reweight:
        train_outcomes_full = outcome_by_row[train_idx]
        outcomes_unique, outcome_counts = np.unique(train_outcomes_full, return_counts=True)
        inv_freq = {o: len(train_outcomes_full) / c for o, c in zip(outcomes_unique, outcome_counts)}
        mean_inv_freq = float(np.mean([inv_freq[o] for o in train_outcomes_full]))
        norm_weight = {o: min(w / mean_inv_freq, args.outcome_reweight_max) for o, w in inv_freq.items()}
        outcome_weight_by_row = np.ones(len(ds), dtype=np.float32)
        for o, w in norm_weight.items():
            outcome_weight_by_row[outcome_by_row == o] = w
        log.info(f"--- Outcome reweighting enabled (train rows only, capped at "
                 f"{args.outcome_reweight_max}x): {norm_weight} ---")

    synthetic_obs_list: list = []
    synthetic_returns = np.array([], dtype=np.float64)
    synthetic_outcomes = np.array([], dtype=object)
    if args.synthetic_ball_out_episodes > 0 or args.synthetic_timeout_episodes > 0:
        # Each generator builds and drives its own ScenarioEnv instance and
        # is otherwise pure Python/NumPy (holds the GIL almost the whole
        # time) -- a thread pool would just take turns, not actually run
        # both concurrently. Use separate processes for real parallelism;
        # both generators are already side-effect-free functions returning
        # plain (list, ndarray, ndarray) results, so they pickle cleanly.
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as pool:
            ball_out_future = (
                pool.submit(
                    _generate_synthetic_ball_out_set,
                    args.synthetic_ball_out_episodes, args.synthetic_ball_out_t_max_s, args.seed,
                )
                if args.synthetic_ball_out_episodes > 0 else None
            )
            timeout_future = (
                pool.submit(
                    _generate_synthetic_timeout_set,
                    args.synthetic_timeout_episodes, args.synthetic_timeout_s_min,
                    args.synthetic_timeout_s_max, args.seed,
                )
                if args.synthetic_timeout_episodes > 0 else None
            )
            if ball_out_future is not None:
                synthetic_obs_list, synthetic_returns, synthetic_outcomes = ball_out_future.result()
            if timeout_future is not None:
                to_obs_list, to_returns, to_outcomes_raw = timeout_future.result()
                if to_obs_list:
                    # Tagged "timeout_<outcome>" (rather than reusing bare outcome
                    # names) so this set's breakdown stays distinguishable from the
                    # ball-out set's above even after both are concatenated.
                    to_outcomes = np.array([f"timeout_{o}" for o in to_outcomes_raw], dtype=object)
                    synthetic_obs_list = synthetic_obs_list + to_obs_list
                    synthetic_returns = np.concatenate([synthetic_returns, to_returns])
                    synthetic_outcomes = np.concatenate([synthetic_outcomes, to_outcomes])

    _run_linear_regression(
        ds, train_idx, val_idx, returns, outcome_weight_by_row,
        synthetic_obs_list=synthetic_obs_list, synthetic_returns=synthetic_returns,
        synthetic_outcomes=synthetic_outcomes,
    )

    if args.checkpoint:
        # Loaded checkpoint's decision_net -- frozen either way (this script
        # never trains it), same role as the fresh-random one below: produce
        # decision_heads context for the execution/value network's forward
        # pass. Unlike the fresh-network path, this one is the REAL policy
        # that generated ds's rollout, so decision_heads context matches
        # what the checkpoint actually saw.
        decision_net = ckpt_trainer.decision_net
        decision_net.eval()
        for p in decision_net.parameters():
            p.requires_grad_(False)

        if ckpt_trainer.separate_value_net and not args.reset_separate_value_net:
            # Real separate value_net from the checkpoint: no BC-primed trunk
            # to protect (see PPOTrainer.pretrain_value()'s separate_value_net
            # branch) -- train it fully unfrozen, same as production.
            value_net = ckpt_trainer.value_net
            if args.reset_value_weights:
                _reset_value_head(value_net)
            optimizer = torch.optim.Adam(
                value_net.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.weight_decay
            )
            _capacity_desc = "checkpoint's separate value_net" + (
                " (value_head reset)" if args.reset_value_weights else "")
        elif ckpt_trainer.separate_value_net and args.reset_separate_value_net:
            # Ignore the checkpoint's saved value_net weights entirely and
            # fit a brand new one instead -- still uses the checkpoint's
            # decision_net above for context. Capacity overrides are only
            # meaningful here (see the argparse guard above).
            value_net = ExecutionNetwork.from_config(
                trunk_hidden_override=args.trunk_hidden,
                value_hidden_dim_override=args.value_hidden_dim,
                entity_embed_dim_override=args.entity_embed_dim,
            )
            optimizer = torch.optim.Adam(
                value_net.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.weight_decay
            )
            _capacity_desc = "fresh separate value_net (--reset-separate-value-net)"
        else:
            if args.reset_separate_value_net:
                log.warning("--reset-separate-value-net has no effect: this checkpoint was "
                            "not trained with --separate-value-net.")
            # Non-separate checkpoint: value_net IS execution_net -- freeze
            # trunk/encoders (PPOTrainer._get_value_pretrain_freeze_params(),
            # the exact "value pretrain" freezing convention used in
            # production) and train only value_head, mirroring
            # PPOTrainer.pretrain_value()'s non-separate branch.
            value_net = ckpt_trainer.execution_net
            if args.reset_value_weights:
                _reset_value_head(value_net)
            for p in ckpt_trainer._get_value_pretrain_freeze_params():
                p.requires_grad_(False)
            optimizer = torch.optim.Adam(
                list(value_net.value_head.parameters()),
                lr=args.lr, eps=1e-5, weight_decay=args.weight_decay,
            )
            _capacity_desc = "checkpoint's execution_net, trunk/encoders frozen (value_head only)" + (
                " (value_head reset)" if args.reset_value_weights else "")
        n_params = sum(p.numel() for p in value_net.parameters())
        n_trainable = sum(p.numel() for p in value_net.parameters() if p.requires_grad)
        log.info(f"value_net: {_capacity_desc}  total_params={n_params:,}  "
                 f"trainable_params={n_trainable:,}")
    else:
        # Fresh decision net (frozen, only used to produce decision_heads context
        # for the execution network's forward pass -- mirrors how the value net
        # is always fed decision_heads in the real training loop, but here it is
        # NOT trained, isolating "can a value net fit returns from obs alone".
        decision_net = DecisionNetwork.from_config()
        decision_net.eval()
        for p in decision_net.parameters():
            p.requires_grad_(False)

        value_net = ExecutionNetwork.from_config(
            trunk_hidden_override=args.trunk_hidden,
            value_hidden_dim_override=args.value_hidden_dim,
            entity_embed_dim_override=args.entity_embed_dim,
        )
        n_params = sum(p.numel() for p in value_net.parameters())
        log.info(
            f"value_net capacity: trunk_hidden={value_net.trunk[0].out_features}  "
            f"value_hidden_dim={args.value_hidden_dim if args.value_hidden_dim is not None else '(config default)'}  "
            f"entity_embed_dim={args.entity_embed_dim if args.entity_embed_dim is not None else '(config default)'}  "
            f"total_params={n_params:,}"
        )
        optimizer = torch.optim.Adam(
            value_net.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.weight_decay
        )

    ret_std_train = float(np.std(returns[train_idx])) if len(train_idx) else 1.0
    ret_var_train = max(ret_std_train ** 2, 1e-6)

    from footballcoach.ai.ppo.bc import (
        AI_TYPE_IMMOBILE, AI_TYPE_NEURAL, AI_TYPE_RULES, _I_OPPONENT_AI_TYPE,
    )
    _OPP_TYPE_NAME = {AI_TYPE_RULES: "rules", AI_TYPE_IMMOBILE: "immobile", AI_TYPE_NEURAL: "neural"}

    def _run_epoch(idx: np.ndarray, train: bool) -> tuple[float, float, dict[str, tuple[float, int]], dict[str, tuple[float, int]]]:
        """Return (raw_mse, normalized_mse, per_opponent_type, per_outcome)
        averaged over all rows in idx. Both breakdown dicts map name ->
        (raw_mse, n_rows). When train=True and synthetic rows were generated
        (--synthetic-ball-out-episodes), those rows are mixed into the SAME
        training batches (not a separate pass) -- they're tagged with a
        "synthetic_" prefix on their real (ball_out/win) outcome in the
        per-outcome breakdown so their contribution stays visible without a
        second training loop."""
        total_sq_err = 0.0
        n_rows = 0
        opp_sq_err: dict[str, float] = {}
        opp_n_rows: dict[str, int] = {}
        outc_sq_err: dict[str, float] = {}
        outc_n_rows: dict[str, int] = {}
        value_net.train(train)
        epoch_idx = idx.copy()
        if train:
            np.random.shuffle(epoch_idx)

        def _step(obs_dict, ret_batch, opp_type_col, outcome_col, row_weights):
            nonlocal total_sq_err, n_rows
            with torch.set_grad_enabled(train):
                with torch.no_grad():
                    d_heads = decision_net(
                        obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
                        obs_dict["ball_feat"], obs_dict["global_feat"],
                    )
                e_heads = value_net(
                    obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
                    obs_dict["ball_feat"], obs_dict["global_feat"], d_heads,
                    obs_dict.get("self_ai_type"), obs_dict.get("other_ai_type"),
                )
                pred = e_heads.value.squeeze(-1)
                per_row_sq = (pred - ret_batch) ** 2
                if train and row_weights is not None:
                    loss = (torch.from_numpy(row_weights) * per_row_sq).mean()
                else:
                    loss = per_row_sq.mean()
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
                    optimizer.step()
            # Reported/logged MSE is always the UNWEIGHTED mean squared error
            # (per_row_sq), even when the backward pass above used a weighted
            # loss -- keeps every logged number comparable across runs regardless
            # of --outcome-reweight.
            per_row_sq_err = per_row_sq.detach()
            total_sq_err += float(per_row_sq_err.sum())
            n_rows += len(ret_batch)
            if opp_type_col is not None:
                for code, name in _OPP_TYPE_NAME.items():
                    row_mask = opp_type_col == code
                    if not row_mask.any():
                        continue
                    opp_sq_err[name] = opp_sq_err.get(name, 0.0) + float(per_row_sq_err[row_mask].sum())
                    opp_n_rows[name] = opp_n_rows.get(name, 0) + int(row_mask.sum())
            for outcome in np.unique(outcome_col):
                row_mask_np = outcome_col == outcome
                outc_sq_err[outcome] = outc_sq_err.get(outcome, 0.0) + float(
                    per_row_sq_err.numpy()[row_mask_np].sum()
                )
                outc_n_rows[outcome] = outc_n_rows.get(outcome, 0) + int(row_mask_np.sum())

        for obs_dict, ret_batch, chunk in _iterate_over_with_indices(ds, epoch_idx, returns, args.batch_size):
            row_weights = (
                outcome_weight_by_row[chunk] if train and outcome_weight_by_row is not None else None
            )
            _step(obs_dict, ret_batch, ds._labels[chunk, _I_OPPONENT_AI_TYPE], outcome_by_row[chunk], row_weights)

        if train and synthetic_obs_list:
            synthetic_outcome_col_full = np.array(
                [f"synthetic_{o}" for o in synthetic_outcomes], dtype=object
            )
            for obs_dict, ret_batch, chunk in _iterate_synthetic_batches(
                synthetic_obs_list, synthetic_returns, args.batch_size, shuffle=True,
            ):
                _step(obs_dict, ret_batch, None, synthetic_outcome_col_full[chunk], None)

        raw_mse = total_sq_err / max(n_rows, 1)
        per_opponent_type = {
            name: (opp_sq_err[name] / max(opp_n_rows[name], 1), opp_n_rows[name])
            for name in opp_sq_err
        }
        per_outcome = {
            name: (outc_sq_err[name] / max(outc_n_rows[name], 1), outc_n_rows[name])
            for name in outc_sq_err
        }
        return raw_mse, raw_mse / ret_var_train, per_opponent_type, per_outcome

    log.info(f"Fitting fresh separate value network: {args.epochs} epochs, "
             f"lr={args.lr}, weight_decay={args.weight_decay}, batch_size={args.batch_size}, "
             f"train_ret_std={ret_std_train:.3f}, outcome_reweight={args.outcome_reweight}")

    # RMSE is in the same units as the MC return itself (MSE is squared
    # units, which is why a return range of [-4, 5] can show MSE values
    # like 40).
    def _log_epoch_result(
        epoch_label: str,
        train_raw, train_norm, train_by_opp, train_by_outc,
        val_raw, val_norm, val_by_opp, val_by_outc,
    ) -> None:
        log.info(
            f"{epoch_label}  "
            f"train_rmse={math.sqrt(train_raw):.4f} (norm={math.sqrt(train_norm):.4f})  "
            f"val_rmse={math.sqrt(val_raw):.4f} (norm={math.sqrt(val_norm):.4f})"
        )
        for name in sorted(set(train_by_opp) | set(val_by_opp)):
            train_mse, train_n = train_by_opp.get(name, (float("nan"), 0))
            val_mse, val_n = val_by_opp.get(name, (float("nan"), 0))
            log.info(
                f"    opponent={name:<9}  train_rmse={math.sqrt(train_mse):.4f} (n={train_n})  "
                f"val_rmse={math.sqrt(val_mse):.4f} (n={val_n})"
            )
        for name in sorted(set(train_by_outc) | set(val_by_outc)):
            train_mse, train_n = train_by_outc.get(name, (float("nan"), 0))
            val_mse, val_n = val_by_outc.get(name, (float("nan"), 0))
            log.info(
                f"    outcome={name:<12}  train_rmse={math.sqrt(train_mse):.4f} (n={train_n})  "
                f"val_rmse={math.sqrt(val_mse):.4f} (n={val_n})"
            )

    # Baseline: evaluate the network's loss on train/val BEFORE any gradient
    # step (train=False on both -- purely a forward pass, no optimizer.step())
    # so epoch 1's improvement can be read against a real starting point
    # instead of assumed. Uses the SAME _run_epoch() as the training loop
    # below, just never called with train=True first.
    _log_epoch_result(
        f"epoch   0/{args.epochs} (baseline, no training yet)",
        *_run_epoch(train_idx, train=False), *_run_epoch(val_idx, train=False),
    )

    best_val_norm = float("inf")
    _patience_ctr = 0
    for epoch in range(1, args.epochs + 1):
        train_raw, train_norm, train_by_opp, train_by_outc = _run_epoch(train_idx, train=True)
        val_raw, val_norm, val_by_opp, val_by_outc = _run_epoch(val_idx, train=False)
        _log_epoch_result(
            f"epoch {epoch:3d}/{args.epochs}",
            train_raw, train_norm, train_by_opp, train_by_outc,
            val_raw, val_norm, val_by_opp, val_by_outc,
        )
        if val_norm < best_val_norm:
            best_val_norm = val_norm
            _patience_ctr = 0
        else:
            _patience_ctr += 1
            if args.patience > 0 and _patience_ctr >= args.patience:
                log.info(f"Early stopping at epoch {epoch}/{args.epochs} "
                         f"(val normalized MSE did not improve for {args.patience} epochs).")
                break

    log.info(f"Best val normalized MSE achieved: {best_val_norm:.4f} "
             f"(RMSE={math.sqrt(best_val_norm):.4f}; "
             f"<1.0 = better than predicting the mean; <0.5 = useful critic)")
    if best_val_norm > 0.85:
        log.warning(
            "Val normalized MSE stayed >= 0.85 even for a FRESH network fit "
            "directly on MC returns -- this points to a data/signal problem "
            "(reward calc, observation encoding, or return computation), not "
            "a PPO/optimizer issue. Inspect the reward component breakdown "
            "and check whether returns actually correlate with anything "
            "observable in obs_self_feat/obs_ball_feat/obs_global_feat."
        )

    # --- Per-row residuals on val set for the two post-training diagnostics ---
    residual_by_row = np.zeros(len(ds), dtype=np.float32)
    value_net.eval()
    with torch.no_grad():
        for obs_dict, ret_batch, chunk in _iterate_over_with_indices(ds, val_idx, returns, args.batch_size):
            d_heads = decision_net(
                obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
                obs_dict["ball_feat"], obs_dict["global_feat"],
            )
            e_heads = value_net(
                obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
                obs_dict["ball_feat"], obs_dict["global_feat"], d_heads,
                obs_dict.get("self_ai_type"), obs_dict.get("other_ai_type"),
            )
            pred = e_heads.value.squeeze(-1).numpy()
            residual_by_row[chunk] = ret_batch.numpy() - pred

    _log_component_correlation(ds, val_idx, args.gamma)
    _episode_residual_correlation(ds, val_idx, args.gamma, residual_by_row)
    _save_worst_episode_match_log(ds, val_idx, residual_by_row, args.worst_episode_log_path)

    if synthetic_obs_list:
        _score_synthetic_ball_out_with_net(
            decision_net, value_net, synthetic_obs_list, synthetic_returns, synthetic_outcomes,
        )


if __name__ == "__main__":
    main()

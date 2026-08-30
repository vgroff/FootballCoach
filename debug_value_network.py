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
import random
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



def _compute_outcome_norm_weights(train_outcomes: np.ndarray, max_weight: float) -> dict:
    """Inverse-frequency weight per outcome value present in
    ``train_outcomes``, normalized to mean 1.0 (so the overall loss scale --
    and therefore --lr -- is unaffected, only the relative per-row weighting
    changes) and capped at ``max_weight`` (uncapped weights blow up when an
    outcome has only a handful of rows, e.g. 5 'loss' rows in 146k -- a
    single such row would then dominate a minibatch's gradient outright,
    which looks like training collapse, not a computation bug). See
    --outcome-reweight/--outcome-reweight-max."""
    outcomes_unique, outcome_counts = np.unique(train_outcomes, return_counts=True)
    inv_freq = {o: len(train_outcomes) / c for o, c in zip(outcomes_unique, outcome_counts)}
    mean_inv_freq = float(np.mean([inv_freq[o] for o in train_outcomes]))
    return {o: min(w / mean_inv_freq, max_weight) for o, w in inv_freq.items()}


def _build_outcome_weight_array(outcome_by_row: np.ndarray, norm_weight: dict) -> np.ndarray:
    """Per-row weight array from a fixed outcome->weight mapping (see
    _compute_outcome_norm_weights()) -- any outcome value not present in
    norm_weight (e.g. an "incomplete" trailing-episode row, or an outcome
    that simply didn't occur in whatever sample norm_weight was computed
    from) defaults to a neutral 1.0."""
    weights = np.ones(len(outcome_by_row), dtype=np.float32)
    for o, w in norm_weight.items():
        weights[outcome_by_row == o] = w
    return weights


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
    predicted), across all complete val episodes. HIGH |correlation| for a
    component that has non-trivial variance means that component's swings
    show up in the net's ERROR, i.e. the net hasn't learned to account for
    it -- a candidate "hardest to predict" signal. LOW |correlation| means
    the opposite: whatever this component contributes is already reflected
    in the net's prediction, leaving no residual trace, even though the
    component itself has real variance (well-tracked). Verified with a
    synthetic check (a component the predictor fully captures correlates
    ~0 with the residual; one it fully ignores correlates ~1) -- don't
    trust the direction from intuition alone, it's easy to get backwards."""
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
    rows.sort(key=lambda r: -abs(r[1]))
    for k, corr, std in rows:
        log.info(f"  {lbl_map.get(k, k):<16}  {corr:>+7.3f}  {std:>9.4f}")
    if rows:
        log.info("  (sorted worst-tracked first: HIGH |corr| = this component's variance "
                  "shows up in the net's error (poorly captured); LOW |corr| despite real "
                  "variance = already well-accounted-for in the prediction. Read alongside "
                  "the per-component MC-return magnitude above.)")


def _log_feature_error_correlation(
    ds, val_idx: np.ndarray, returns_val_all: np.ndarray, residual_by_row: np.ndarray,
) -> None:
    """Correlate per-ROW interpretable observation features (not reward
    components) against the value network's per-row squared error, across
    every val row -- answers "what kind of situation is hardest to predict
    the return for", complementing _episode_residual_correlation()'s
    per-episode reward-component view (which only sees the 7 named reward
    components, not raw observable state like possession/distance/stamina).

    Uses Pearson correlation, which is well-defined for the 0/1 flags here
    too (equivalent to point-biserial correlation), so one table covers
    continuous and binary features alike without separate binning logic.
    ``corr_sqerr`` is the primary signal -- high |corr| means error
    concentrates where that feature is high (positive) or low (negative).
    ``corr_signed`` shows which DIRECTION the net is biased in when that
    feature is high (residual = actual - predicted, so positive = the net
    UNDERESTIMATES the return there) -- two features can have the same
    |corr_sqerr| for very different reasons (systematic bias vs. pure
    noise), which corr_signed disambiguates.

    Field indices below are PlayerFeatures/BallFeatures field-declaration
    order (see obs/schema.py -- field position IS the array index, same
    convention already used by _episode_rows_to_match_log()'s _VEL_X etc.)."""
    residual = residual_by_row[val_idx]
    sq_err = residual ** 2
    self_feat = ds._self_feat[val_idx]
    ball_feat = ds._ball_feat[val_idx]
    global_feat = ds._global_feat[val_idx]

    # PlayerFeatures attribute block (schema.py field order, indices 13-20):
    # top_speed, acceleration, kick_power, kick_precision, dribbling,
    # ball_control, tackling, stamina_attr -- these are FIXED per-player
    # (drawn once at spawn), not per-row state like possession/distance
    # above, so a correlation here answers "does the net do worse for
    # certain player BUILDS" rather than "certain situations". Included
    # because push-kick behaviour (orders.py's _try_push_kick) is directly
    # driven by the top_speed/kick_power ratio (reference_speed from
    # top_speed, max_kick_speed from kick_power) -- a fast, weak-kicking
    # player saturates power_fraction against its 1.0 clamp differently
    # than a slow, hard-kicking one, so it's a real candidate explanation
    # for player-dependent weirdness (e.g. the many-tiny-kicks degenerate
    # mode seen in one real worst-episode trace) that per-row situational
    # features can't see at all.
    top_speed = self_feat[:, 13]
    kick_power = self_feat[:, 15]
    features = {
        "|return| (target)":     np.abs(returns_val_all[val_idx]),
        "time_remaining_norm":   global_feat[:, 1],
        "self_has_possession":   self_feat[:, 23],
        "self_is_controlling":   self_feat[:, 25],
        "self_ball_distance":    self_feat[:, 5],
        "self_ball_closing_spd": self_feat[:, 8],
        "self_speed":            self_feat[:, 11],
        "self_stamina":          self_feat[:, 12],
        "ball_is_loose":         ball_feat[:, 10],
        "ball_height":           ball_feat[:, 2],
        "ball_speed":            np.linalg.norm(ball_feat[:, 3:6], axis=1),
        "is_trainee_row":        ds._is_trainee[val_idx],
        "attr_top_speed":        top_speed,
        "attr_acceleration":     self_feat[:, 14],
        "attr_kick_power":       kick_power,
        "attr_kick_precision":   self_feat[:, 16],
        "attr_dribbling":        self_feat[:, 17],
        "attr_ball_control":     self_feat[:, 18],
        "attr_tackling":         self_feat[:, 19],
        "attr_stamina_attr":     self_feat[:, 20],
        "attr_top_speed_minus_kick_power": top_speed - kick_power,
    }
    log.info("--- Per-row feature vs. value-error correlation (val rows) ---")
    log.info(f"  {'feature':<24}  {'corr_sqerr':>10}  {'corr_signed':>11}")
    rows = []
    for name, vals in features.items():
        if vals.std() < 1e-8:
            continue
        corr_sq = float(np.corrcoef(vals, sq_err)[0, 1])
        corr_signed = float(np.corrcoef(vals, residual)[0, 1])
        rows.append((name, corr_sq, corr_signed))
    rows.sort(key=lambda r: -abs(r[1]))
    for name, corr_sq, corr_signed in rows:
        log.info(f"  {name:<24}  {corr_sq:>+10.3f}  {corr_signed:>+11.3f}")
    if rows:
        log.info("  (corr_sqerr: high |corr| = error concentrates where this feature is "
                  "high/low. corr_signed: positive = net UNDERESTIMATES the return when "
                  "this feature is high (residual=actual-predicted grows with it), "
                  "negative = net OVERESTIMATES there.)")


def _episode_rows_to_match_log(
    ds, start: int, end: int, *,
    returns_by_row: np.ndarray | None = None,
    residual_by_row: np.ndarray | None = None,
    has_prediction: np.ndarray | None = None,
) -> list[dict]:
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

    Every event carrying a "player_pos" also carries a matching "player_vel"
    (3-tuple, same m/s convention as "ball_vel") -- current speed at that
    exact sample, not derivable from position deltas alone since samples
    aren't evenly spaced in real time (kick/tackle callback rows). The
    "start" event's player_positions additionally carries "attributes" (the
    8 PlayerAttributes fields, all [0,1]) per player -- fixed for the whole
    episode (drawn once at spawn), so captured once there rather than
    repeated on every subsequent event. The "episode_end" event also carries
    "seed" (the int this episode's scenario was built with, or None for
    datasets predating meta_episode_seeds -- see DemonstrationDataset.
    episode_seed()) so a worst-episode export can be rebuilt exactly later
    via scripts/replay_episode.py, not just read as a static log.

    When ``returns_by_row``/``residual_by_row``/``has_prediction`` are given
    (all full-``ds``-length arrays, e.g. straight from _run_val_diagnostics'
    own residual_by_row/returns_val plus a `has_prediction[val_idx] = True`
    mask), every event carrying "self"'s own position also gets
    "predicted_value" and "actual_return" (predicted = actual - residual,
    per debug_value_network.py's own residual convention: return -
    predicted) for THAT row -- not just once at episode end. Previously the
    net's prediction was only ever visible as a single number
    (worst_residual) on the episode_end event, computed from the episode's
    FIRST row -- there was no way to see whether the prediction tracked the
    unfolding trajectory (e.g. dropping as the ball visibly heads toward
    the boundary) or sat oblivious to it. `has_prediction` guards against
    stamping a misleading 0.0 on a row `valid_indices()`/val_idx excluded
    (e.g. a non-decision-step callback row) -- residual_by_row/returns_by_row
    are only meaningful where has_prediction is True. Omitted (None, the
    default) on all three: no predicted_value/actual_return fields at all,
    exactly the prior behaviour.
    """
    from footballcoach.ai.ppo.bc import (
        _I_AI_TYPE, _I_KICK_THIS_TICK, _I_OPPONENT_AI_TYPE, _I_TACKLE_ATTEMPT,
        AI_TYPE_IMMOBILE, AI_TYPE_NEURAL, AI_TYPE_RULES,
    )

    _AI_TYPE_NAME = {AI_TYPE_RULES: "rules", AI_TYPE_IMMOBILE: "immobile", AI_TYPE_NEURAL: "neural"}
    # PlayerFeatures column indices (see schema.py field order).
    _VEL_X, _VEL_Y, _HAS_POSS, _POS_X, _POS_Y = 9, 10, 23, 30, 31
    _ATTACKING_DIR = 27
    # Attribute block (schema.py fields 13-20, all [0,1] from PlayerAttributes)
    # -- FIXED per player for the whole episode (drawn once at spawn), unlike
    # every other field this function reads, so it's only captured once, on
    # the "start" event, rather than repeated on every position sample.
    _ATTR_TOP_SPEED, _ATTR_ACCEL, _ATTR_KICK_POWER, _ATTR_KICK_PRECISION = 13, 14, 15, 16
    _ATTR_DRIBBLING, _ATTR_BALL_CONTROL, _ATTR_TACKLING, _ATTR_STAMINA_ATTR = 17, 18, 19, 20
    # BallFeatures column indices (height_m is already in real metres, not
    # normalized -- see schema.py's BallFeatures.height_m).
    _BALL_POS_X, _BALL_POS_Y, _BALL_POS_Z, _BALL_VEL_X, _BALL_VEL_Y = 0, 1, 2, 3, 4
    # Pitch half-diagonal -- ai/obs/encoder.py normalizes BOTH position
    # AND velocity x/y by this SAME constant for players and the ball alike
    # (encoder.py: "pos_x=ball.position.x / half_diag", "pos_y=... /
    # half_diag" -- NOT separate 52.5/34.0 per-axis divisors; schema.py's
    # docstrings used to (wrongly) imply per-axis normalization and have
    # since been corrected to match this). Using 52.5/34.0 under-scales y
    # by ~1.84x and x by ~1.19x, silently producing positions that look
    # well within bounds when the real position is actually near/at the
    # boundary -- this is exactly what made early debugging of ball-out
    # episodes so confusing, and the same bug resurfaced in a standalone
    # diagnostic script (diagnose_crossing_head.py) that didn't reuse this
    # function -- see ai/knowledge.md.
    import math
    _HALF_DIAG = math.hypot(52.5, 34.0)

    # Real elapsed seconds per TIMED sample (see record_demonstrations.py's
    # `sample_interval_s` / ai_config.json's bc.demo_sample_interval_s).
    # Previously this reconstruction used raw `row - start` as "time_s",
    # which is NOT real time -- kick/tackle callbacks insert extra rows
    # between timed samples, and even ignoring those, one row-index step
    # is NOT one second. On one real ball_out episode this made an 18-row
    # episode display as "18.0s" when the true duration (verified against
    # the player's own recorded speed_mps and this same divisor) was closer
    # to 4.5s -- a ~4x error that made a clean, fast sprint-and-intercept
    # look like a slow multi-second jog. Approximated as
    # (index within trainee_rows) * sample_interval_s -- exact when no
    # kick/tackle mid-interval callback has inserted an extra trainee row
    # (the common case), a slight underestimate otherwise (those extra rows
    # don't represent a new timed sample, but do advance the index by 1).
    from footballcoach.ai.config import load_ai_config
    _sample_interval_s = float(load_ai_config()["bc"]["demo_sample_interval_s"])

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

    def _attrs(feat) -> dict[str, float]:
        """PlayerAttributes snapshot (all [0,1]) -- fixed for the whole
        episode, so callers only need this once per player, not per row."""
        return {
            "top_speed": round(float(feat[_ATTR_TOP_SPEED]), 3),
            "acceleration": round(float(feat[_ATTR_ACCEL]), 3),
            "kick_power": round(float(feat[_ATTR_KICK_POWER]), 3),
            "kick_precision": round(float(feat[_ATTR_KICK_PRECISION]), 3),
            "dribbling": round(float(feat[_ATTR_DRIBBLING]), 3),
            "ball_control": round(float(feat[_ATTR_BALL_CONTROL]), 3),
            "tackling": round(float(feat[_ATTR_TACKLING]), 3),
            "stamina_attr": round(float(feat[_ATTR_STAMINA_ATTR]), 3),
        }

    def _pred_fields(row: int) -> dict[str, float]:
        """predicted_value/actual_return for `row`, or {} if prediction data
        wasn't passed in or this row has none (see docstring above)."""
        if has_prediction is None or not has_prediction[row]:
            return {}
        actual = round(float(returns_by_row[row]), 4)
        predicted = round(actual - float(residual_by_row[row]), 4)
        return {"predicted_value": predicted, "actual_return": actual}

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

    def _opp_feat(row: int):
        _exists_row = ds._exists_mask[row]
        _opp_slot = int(_exists_row.argmax()) if _exists_row.any() else 0
        return ds._other_feat[row, _opp_slot]

    # BUG FIX (verified against raw ds._is_trainee/_self_feat/_other_feat
    # data before applying): trainee and opponent rows are INTERLEAVED
    # within an episode's row range (see compute_returns()'s docstring --
    # every timed sample appends the trainee's own row AND the opponent's
    # own row, back to back, order not guaranteed). Each row's self_feat/
    # bc_label is THAT ROW's OWNER's own perspective -- on an
    # opponent-owned row (is_trainee=0), ds._self_feat[row] is the
    # OPPONENT's own state (e.g. an immobile opponent correctly pinned at
    # its spawn point) and ds._other_feat[row] is really the TRAINEE as
    # seen by the opponent, NOT "self"/"opponent" from the trainee's point
    # of view this reconstruction is trying to tell a story from. Using
    # every row blindly and always labelling self_feat "self" therefore
    # SWAPS the two players' identities every other row -- verified
    # directly against raw rows: it produced a nonsensical "immobile
    # opponent teleports to wherever the ball is and gains possession every
    # tick" narrative for an episode whose raw data is actually completely
    # clean (opponent provably pinned at one spot the whole time, trainee's
    # possession stable, position progressing smoothly). Restricting to
    # only the trainee's OWN rows fixes this -- each such row already has a
    # fully correct self=trainee/other=opponent view, so the opponent's own
    # interleaved rows aren't needed at all (also true for `start`/`end`
    # themselves -- neither is guaranteed to land on a trainee row).
    trainee_rows = [r for r in range(start, end + 1) if ds._is_trainee[r] > 0.5]
    if not trainee_rows:
        raise ValueError(
            f"_episode_rows_to_match_log: no trainee-owned row in [{start}, {end}] -- "
            f"every episode must have at least one (its own is_trainee marking is broken, "
            f"or start/end don't actually bound one complete episode)."
        )
    ep_start, ep_end = trainee_rows[0], trainee_rows[-1]
    # Real elapsed time only advances on genuine decision-step rows
    # (is_decision_step==1, 0.5s apart) -- kick/tackle-callback rows and the
    # trailing terminal row are_decision_step==0 (one real physics tick
    # later, not a new 0.5s sample) and must NOT advance the clock.
    # Previously this used a flat enumerate() index over ALL trainee rows
    # regardless of is_decision_step, so every callback/terminal row
    # inflated the count by a false extra 0.5s -- confirmed against real
    # data: a timeout episode with 40 true decision-steps (20.0s, matching
    # ds.classify_outcome's own 40 rows independently) displayed as 21.5s
    # (43 * 0.5s) purely because 4 of its 44 trainee rows were non-decision
    # callback/terminal rows that still each advanced the old flat index.
    _row_to_i: dict[int, int] = {}
    _i = -1
    for r in trainee_rows:
        if ds._is_decision_step[r] > 0.5:
            _i += 1
        _row_to_i[r] = max(_i, 0)

    def _t(row: int) -> float:
        """Real elapsed seconds for `row` -- see _sample_interval_s's own
        comment. `row` may not itself be trainee-owned (e.g. `end`); falls
        back to the nearest trainee row's index rather than raising."""
        if row in _row_to_i:
            return round(_row_to_i[row] * _sample_interval_s, 3)
        nearest = min(trainee_rows, key=lambda r: abs(r - row))
        return round(_row_to_i[nearest] * _sample_interval_s, 3)

    self_ai_type = _AI_TYPE_NAME.get(float(ds._labels[ep_start][_I_AI_TYPE]), "unknown")
    opp_ai_type = _AI_TYPE_NAME.get(float(ds._labels[ep_start][_I_OPPONENT_AI_TYPE]), "unknown")

    start_self_feat = ds._self_feat[ep_start]
    start_ball_feat = ds._ball_feat[ep_start]
    start_opp_feat = _opp_feat(ep_start)
    # "team": "left"/"right" is a DISPLAY label only (picks marker colour in
    # scripts/visualise_match_log.py) -- must be derived from the real
    # PlayerFeatures.attacking_direction field (+1.0 attacks +x, -1.0 attacks
    # -x), NOT hardcoded. Previously this was hardcoded "left"/"right"
    # regardless of which way the player actually attacks, which silently
    # mislabelled every player whose real attacking_direction is -1.0 --
    # made a legitimate box_possession-scoring run (ball correctly deep in
    # the TRUE opponent box on the -x side) look like it ended nowhere near
    # either box when checked against the wrong (+x) side.
    self_attacks_pos_x = float(start_self_feat[_ATTACKING_DIR]) > 0.0
    self_team = "left" if self_attacks_pos_x else "right"
    opp_team = "right" if self_attacks_pos_x else "left"
    events: list[dict] = [{
        "time_s": 0.0, "event": "start",
        "ball_pos": _ball_pos3(start_ball_feat),
        "ball_vel": _vel3(start_ball_feat, _BALL_VEL_X, _BALL_VEL_Y),
        "player_positions": {
            "self": {"pos": _pos3(start_self_feat, _POS_X, _POS_Y),
                     "vel": _vel3(start_self_feat, _VEL_X, _VEL_Y),
                     "team": self_team, "ai_type": self_ai_type,
                     "attributes": _attrs(start_self_feat),
                     **_pred_fields(ep_start)},
            "opponent": {"pos": _pos3(start_opp_feat, _POS_X, _POS_Y),
                         "vel": _vel3(start_opp_feat, _VEL_X, _VEL_Y),
                         "team": opp_team, "ai_type": opp_ai_type,
                         "attributes": _attrs(start_opp_feat)},
        },
    }]

    # Periodic position snapshots so the visualiser's player-track lines
    # don't sit frozen between sparse events (possession_change/kick/
    # tackle_attempt) -- previously "consistency" only ever appeared once,
    # right before episode_end (see below), so a long quiet stretch of
    # dribbling/running with no kicks (e.g. the timeout episode's last 9s)
    # rendered as a dead straight line with no sense of how the players
    # actually moved through it. Reuses the same "consistency" event type
    # (small muted marker, not a "real" match event) at a fixed real-time
    # cadence instead of just at the end.
    _CONSISTENCY_INTERVAL_S = 2.0
    _last_snapshot_t = float("-inf")

    # kick_this_tick (label index 12) is 1.0 for BOTH a real, physics-
    # executed kick AND a mere "armed" approach tick with no ball contact
    # yet (see ai/ppo/bc.py: `kick_this_tick = 1.0 if (player.kicked_this_tick
    # or player.kick_armed) else 0.0`) -- the two are not separately stored,
    # so a naive `label[_I_KICK_THIS_TICK] > 0.5` filter shows an "armed but
    # never touched the ball" tick as if it were a real kick.
    #
    # A previous version of this filter used "self has possession again on
    # the next recorded row" as the real-kick test. That test is now WRONG:
    # since the engine's one-touch armed-kick fix (Match.
    # _update_loose_ball_pickup's kick_armed branch), a real kick taken
    # while merely armed (the common case for a chasing player) grants and
    # releases possession within the SAME physics tick, by design (no
    # CONTROLLING_BALL, no lingering possession) -- so has_poss never shows
    # True on ANY sampled row for that kick, even though it genuinely
    # struck the ball (confirmed against raw recorded data: repeated real
    # velocity bumps on a chase with has_poss=False throughout). The
    # possession-based test silently hid every one-touch redirect.
    #
    # The reliable signal instead: is_decision_step==0 marks a row inserted
    # by the engine's own on_kick/on_tackle callback (see record_
    # demonstrations.py), and on_kick only ever fires from a REAL executed
    # kick (Player.kick_direct/kick_with_direction), never from merely
    # being armed -- so `kick_this_tick AND is_decision_step==0` on the
    # SAME row is a genuine physics-level kick, independent of whether
    # possession lingers afterward.
    _is_decision_step_by_i = [ds._is_decision_step[r] > 0.5 for r in trainee_rows]

    last_possessor: str | None = None
    for _i, row in enumerate(trainee_rows):
        t = _t(row)
        self_feat = ds._self_feat[row]
        ball_feat = ds._ball_feat[row]
        ball_pos = _ball_pos3(ball_feat)
        pos = _pos3(self_feat, _POS_X, _POS_Y)
        vel = _vel3(self_feat, _VEL_X, _VEL_Y)
        # The (single, in phase 1) opponent occupies a RANDOMIZED slot in
        # other_feat each row (see ai/obs/encoder.py's rng.sample slot
        # shuffle) -- find it via exists_mask rather than assuming slot 0.
        # has_possession there signals the OPPONENT gaining/losing the ball,
        # which self_feat's has_possession alone can't distinguish from a
        # loose ball.
        other_feat = _opp_feat(row)
        other_pos = _pos3(other_feat, _POS_X, _POS_Y)
        other_vel = _vel3(other_feat, _VEL_X, _VEL_Y)
        if row != ep_end and t - _last_snapshot_t >= _CONSISTENCY_INTERVAL_S:
            events.append({
                "time_s": t, "event": "consistency", "player_id": "self",
                "player_pos": pos, "player_vel": vel, "ball_pos": ball_pos,
                **_pred_fields(row),
            })
            events.append({
                "time_s": t, "event": "consistency", "player_id": "opponent",
                "player_pos": other_pos, "player_vel": other_vel, "ball_pos": ball_pos,
            })
            _last_snapshot_t = t
        self_has_poss = self_feat[_HAS_POSS] > 0.5
        opp_has_poss = other_feat[_HAS_POSS] > 0.5
        possessor = "self" if self_has_poss else "opponent" if opp_has_poss else None
        if possessor != last_possessor and (possessor is not None or last_possessor is not None):
            events.append({
                "time_s": t, "event": "possession_change",
                "possessor_id": possessor, "ball_pos": ball_pos,
                "player_id": possessor,
                "player_pos": pos if possessor == "self" else
                    other_pos if possessor == "opponent" else None,
                "player_vel": vel if possessor == "self" else
                    other_vel if possessor == "opponent" else None,
                **(_pred_fields(row) if possessor == "self" else {}),
            })
            last_possessor = possessor
        label = ds._labels[row]
        if label[_I_KICK_THIS_TICK] > 0.5 and not _is_decision_step_by_i[_i]:
            events.append({
                "time_s": t, "event": "kick", "player_id": "self",
                "player_pos": pos, "player_vel": vel, "ball_pos": ball_pos,
                **_pred_fields(row),
            })
        if label[_I_TACKLE_ATTEMPT] > 0.5:
            events.append({
                "time_s": t, "event": "tackle_attempt", "player_id": "self",
                "player_pos": pos, "player_vel": vel, "ball_pos": ball_pos,
                **_pred_fields(row),
            })

    # ball_feat is shared/global (not player-relative), so reading it at the
    # raw episode `end` row is fine regardless of which player owns that
    # row -- only self_feat/other_feat (player-relative) need ep_end.
    end_ball_feat = ds._ball_feat[end]
    end_self_feat = ds._self_feat[ep_end]
    end_opp_feat = _opp_feat(ep_end)
    end_breakdown = _reward_breakdown(start, end)
    end_t = _t(end)
    end_outcome = ds.classify_outcome(end) if ds.has_episode_outcomes else None
    # The seed this episode's scenario was built with -- see Ball.
    # episode_seed()'s own docstring. None for datasets recorded before
    # meta_episode_seeds existed (re-record to enable replay). Lets any
    # worst-episode export be rebuilt exactly via scripts/replay_episode.py
    # --seed <this value>, instead of only ever inspecting this static log.
    end_seed = ds.episode_seed(end) if ds.has_episode_seeds else None
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
        "player_vel": _vel3(end_self_feat, _VEL_X, _VEL_Y),
        "ball_pos": _ball_pos3(end_ball_feat),
        **_pred_fields(ep_end),
    })
    events.append({
        "time_s": end_t, "event": "consistency", "player_id": "opponent",
        "player_pos": _pos3(end_opp_feat, _POS_X, _POS_Y),
        "player_vel": _vel3(end_opp_feat, _VEL_X, _VEL_Y),
        "ball_pos": _ball_pos3(end_ball_feat),
    })
    # NOTE: `end_ball_feat`/`end_t` are simply whatever the LAST recorded row
    # of the episode is -- this function never extrapolates or replays
    # physics to guess at a state that wasn't recorded. Data recorded
    # BEFORE the record_demonstrations.py fix (see ScenarioLoop.
    # last_completed_trial_match / ScenarioEnv.last_terminal_match /
    # _record_terminal_now in record_demonstrations.py) has its last row
    # one physics tick BEFORE the true terminal state (env.step() rebuilds
    # the next trial's match before returning, so the tick that actually
    # triggered the outcome was never captured) -- re-record to get the
    # real final tick; this reconstruction only ever shows what's actually
    # in the file.

    events.append({
        "time_s": end_t, "event": "episode_end",
        "ball_pos": _ball_pos3(end_ball_feat),
        "reward_total": round(float(ds._rewards[start:end + 1].sum()), 4),
        "reward_components": end_breakdown,
        "reward_cumulative": end_breakdown,
        "outcome": end_outcome,
        "seed": end_seed,
        # Both players' TRUE final recorded positions, same shape as the
        # "start" event's player_positions -- lets the visualiser draw a
        # dedicated end-of-episode marker for each player (not just the
        # ball) and close out each player's trajectory at its real
        # endpoint rather than only via the muted "consistency" dot.
        "player_positions": {
            "self": {"pos": _pos3(end_self_feat, _POS_X, _POS_Y),
                     "vel": _vel3(end_self_feat, _VEL_X, _VEL_Y), "team": self_team,
                     **_pred_fields(ep_end)},
            "opponent": {"pos": _pos3(end_opp_feat, _POS_X, _POS_Y),
                         "vel": _vel3(end_opp_feat, _VEL_X, _VEL_Y), "team": opp_team},
        },
    })
    return events


def _save_worst_episode_match_log(
    ds, val_idx: np.ndarray, residual_by_row: np.ndarray, out_path: str,
    returns_by_row: np.ndarray | None = None,
) -> None:
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
    skimmable.

    `returns_by_row` (the same actual-return array `residual_by_row` was
    computed against, e.g. `returns_val`) is optional -- when given, every
    self-position event in the saved log also carries predicted_value/
    actual_return for that row (see _episode_rows_to_match_log), not just
    a single residual number on episode_end. `has_prediction` is derived
    from `val_idx` here (rows outside it never had a real forward pass this
    diagnostics run) rather than trusting residual_by_row's own default-0.0
    fill, which is indistinguishable from a genuine zero residual.

    ALSO saves one more file, independent of the per-outcome selection
    above: the single complete val episode with the most REAL kicks (see
    _count_real_kicks below), as "<stem>_most_kicks<suffix>" -- a cheap,
    separate diagnostic lens (a chaotic thrash of kicks looks nothing like
    a clean episode even when the outcome/residual alone wouldn't flag it
    as unusual)."""
    import json
    from footballcoach.ai.ppo.bc import _I_KICK_THIS_TICK

    def _count_real_kicks(start: int, end: int) -> int:
        """Real, physics-executed kicks only -- kick_this_tick AND NOT
        is_decision_step (see _episode_rows_to_match_log's own docstring:
        kick_this_tick alone also fires for a merely-armed approach with no
        ball contact yet), restricted to trainee-owned rows to match
        exactly which rows become "kick" events in the saved log."""
        is_trainee_slice = ds._is_trainee[start:end + 1]
        labels_slice = ds._labels[start:end + 1]
        is_decision_step_slice = ds._is_decision_step[start:end + 1]
        real_kick = (
            (is_trainee_slice > 0.5)
            & (labels_slice[:, _I_KICK_THIS_TICK] > 0.5)
            & (is_decision_step_slice <= 0.5)
        )
        return int(real_kick.sum())

    ranges = ds.episode_row_ranges(val_idx)
    if not ranges:
        log.info("--- No complete val episodes to extract worst-case match logs from ---")
        return

    by_outcome: dict[str, list[tuple[int, int]]] = {}
    for start, end in ranges:
        outcome = ds.classify_outcome(end) if ds.has_episode_outcomes else "unknown"
        by_outcome.setdefault(outcome, []).append((start, end))

    # episode_row_ranges(val_idx) deliberately returns val_idx's OWN
    # first/last row per episode, which -- per its own docstring -- can be
    # a proper SUBSET of the full-dataset episode whenever val_idx is
    # filtered (val_idx here excludes some rows; see valid_indices()).
    # Using that truncated end directly as _episode_rows_to_match_log's
    # `end` would omit the episode's real terminal row entirely -- confirmed
    # against real data: a timeout episode's true dones=1 boundary was 3
    # rows past what episode_row_ranges(val_idx) reported, silently cutting
    # off the actual final state. Resolve to the TRUE full-dataset boundary
    # before reconstructing, so the saved match log always ends on the
    # episode's real last row.
    full_ranges = ds._full_dataset_episode_row_ranges()
    full_ends = np.array([e for _s, e in full_ranges], dtype=np.int64)

    def _resolve_full_range(any_row: int) -> tuple[int, int]:
        idx = int(np.searchsorted(full_ends, any_row, side="left"))
        return full_ranges[idx]

    has_prediction = None
    if returns_by_row is not None:
        has_prediction = np.zeros(len(ds), dtype=bool)
        has_prediction[val_idx] = True

    out_path = Path(out_path)
    for outcome in sorted(by_outcome):
        outcome_ranges = by_outcome[outcome]
        worst_start, worst_end = max(outcome_ranges, key=lambda r: abs(residual_by_row[r[0]]))
        worst_residual = float(residual_by_row[worst_start])
        true_start, true_end = _resolve_full_range(worst_start)
        events = _episode_rows_to_match_log(
            ds, true_start, true_end,
            returns_by_row=returns_by_row, residual_by_row=residual_by_row,
            has_prediction=has_prediction,
        )
        # Metadata folded into the episode_end event as extra keys (ignored
        # by scripts/visualise_match_log.py, which only reads specific known
        # fields) rather than wrapped in a top-level dict -- the file itself
        # must be a bare event list to match real MatchLogger.save() dumps,
        # so this is directly viewable with:
        #   uv run python scripts/visualise_match_log.py <this file>
        events[-1]["residual"] = worst_residual
        events[-1]["row_range"] = [int(true_start), int(true_end)]
        events[-1]["n_episodes_this_outcome"] = len(outcome_ranges)
        outcome_path = out_path.with_name(f"{out_path.stem}_{outcome}{out_path.suffix}")
        with open(outcome_path, "w") as f:
            json.dump(events, f, indent=2)
        log.info(f"--- Worst val episode for outcome={outcome} ({len(outcome_ranges)} "
                  f"episode(s)): rows [{true_start}, {true_end}], "
                  f"residual={worst_residual:+.3f} -- saved match log to {outcome_path} ---")

    most_kicks_start, most_kicks_end = max(ranges, key=lambda r: _count_real_kicks(r[0], r[1]))
    kick_count = _count_real_kicks(most_kicks_start, most_kicks_end)
    true_start, true_end = _resolve_full_range(most_kicks_start)
    events = _episode_rows_to_match_log(
        ds, true_start, true_end,
        returns_by_row=returns_by_row, residual_by_row=residual_by_row,
        has_prediction=has_prediction,
    )
    events[-1]["kick_count"] = kick_count
    events[-1]["row_range"] = [int(true_start), int(true_end)]
    most_kicks_path = out_path.with_name(f"{out_path.stem}_most_kicks{out_path.suffix}")
    with open(most_kicks_path, "w") as f:
        json.dump(events, f, indent=2)
    log.info(f"--- Val episode with the most kicks ({kick_count}, outcome="
              f"{events[-1].get('outcome')}): rows [{true_start}, {true_end}] -- "
              f"saved match log to {most_kicks_path} ---")


def _iterate_over_with_indices(ds, idx, returns, batch_size, device: torch.device | None = None):
    """Like _iterate_over(shuffle=False) but also yields the row-index chunk,
    so callers can scatter per-row outputs (e.g. residuals) back into a
    dataset-sized array. ``device`` (default: CPU, matching _to_tensor's own
    default) -- pass the training device so batches land there directly
    instead of needing a separate transfer per step."""
    from footballcoach.ai.bc.dataset import _build_ai_type_arrays, _to_tensor

    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        if len(chunk) == 0:
            continue
        self_ai_type, other_ai_type = _build_ai_type_arrays(
            ds._labels[chunk], ds._exists_mask[chunk]
        )
        obs_dict = {
            "self_feat":   _to_tensor(ds._self_feat[chunk], device),
            "other_feat":  _to_tensor(ds._other_feat[chunk], device),
            "exists_mask": _to_tensor(ds._exists_mask[chunk], device),
            "ball_feat":   _to_tensor(ds._ball_feat[chunk], device),
            "global_feat": _to_tensor(ds._global_feat[chunk], device),
            "self_ai_type":  _to_tensor(self_ai_type, device),
            "other_ai_type": _to_tensor(other_ai_type, device),
            "labels": _to_tensor(ds._labels[chunk], device),
            "row_idx": torch.as_tensor(chunk, dtype=torch.long, device=device),
        }
        ret_batch = _to_tensor(returns[chunk], device)
        yield obs_dict, ret_batch, chunk


def _iterate_synthetic_batches(
    obs_list: list, returns_arr: np.ndarray, batch_size: int, shuffle: bool = False,
    device: torch.device | None = None,
):
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
            "self_feat":     _to_tensor(self_feat[chunk], device),
            "other_feat":    _to_tensor(other_feat[chunk], device),
            "exists_mask":   _to_tensor(exists_mask[chunk], device),
            "ball_feat":     _to_tensor(ball_feat[chunk], device),
            "global_feat":   _to_tensor(global_feat[chunk], device),
            "self_ai_type":  _to_tensor(self_ai_type[chunk], device),
            "other_ai_type": _to_tensor(other_ai_type[chunk], device),
        }
        ret_batch = _to_tensor(returns_arr[chunk], device)
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
    outcomes: np.ndarray | None = None, device: torch.device | None = None,
) -> None:
    """Score the TRAINED value_net against the fixed synthetic set's real
    discounted returns (see _generate_synthetic_ball_out_set), broken down
    per outcome (ball_out vs. win) when *outcomes* is given."""
    from footballcoach.ai.physics_pretrain.physics_value_net import PhysicsEncoderValueNet

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
        self_feat_t = torch.from_numpy(self_feat).to(device)
        other_feat_t = torch.from_numpy(other_feat).to(device)
        exists_mask_t = torch.from_numpy(exists_mask).to(device)
        ball_feat_t = torch.from_numpy(ball_feat).to(device)
        global_feat_t = torch.from_numpy(global_feat).to(device)
        self_ai_type_t = torch.from_numpy(self_ai_type).to(device)
        other_ai_type_t = torch.from_numpy(other_ai_type).to(device)
        d_heads = decision_net(self_feat_t, other_feat_t, exists_mask_t, ball_feat_t, global_feat_t)
        # PhysicsEncoderValueNet is never reachable here in practice -- guarded
        # upfront in main() (--physics-encoder-value-net is incompatible with
        # --synthetic-ball-out-episodes/--synthetic-timeout-episodes, the only
        # callers of this function) -- isinstance check kept as defensive
        # insurance only; synthetic ObservationBatch rows have no BC-label
        # source, so this would raise inside forward() if ever hit.
        extra_kwargs = {"labels": None} if isinstance(value_net, PhysicsEncoderValueNet) else {}
        e_heads = value_net(
            self_feat_t, other_feat_t, exists_mask_t, ball_feat_t, global_feat_t, d_heads,
            self_ai_type_t, other_ai_type_t,
            **extra_kwargs,
        )
        pred = e_heads.value.squeeze(-1).cpu().numpy()

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


def _save_value_net_checkpoint(
    path: str, kind: str, state_dict: dict, optimizer, epoch: int, best_val_norm: float, extra_meta: dict,
) -> None:
    """Save THIS script's own value_net training progress -- see
    --value-checkpoint-dir/--init-value-checkpoint. ``kind`` tags which
    branch produced this checkpoint ("physics_encoder_value_net" or
    "fresh_execution_net") so a later --init-value-checkpoint can refuse to
    load one into the wrong branch with a clear error instead of a silent
    shape mismatch. ``extra_meta`` carries the branch-specific construction
    args needed to rebuild an identically-shaped network before loading
    ``state_dict`` back in (e.g. the physics branch's checkpoint paths/
    mlp_hidden, or the fresh branch's trunk_hidden/value_hidden_dim/
    entity_embed_dim)."""
    from pathlib import Path

    ckpt = {
        "kind": kind,
        "state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_norm": best_val_norm,
        **extra_meta,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)


def _move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    """optimizer.load_state_dict() restores Adam's exp_avg/exp_avg_sq as
    whatever device they were SAVED on (always CPU here, see torch.save's
    map_location="cpu" in _load_value_net_checkpoint) -- these don't follow
    the model's own .to(device) call, so a resumed run on GPU would
    otherwise hit a device-mismatch error (or silently use stale CPU state)
    the moment optimizer.step() first runs. Call once right after
    load_state_dict()."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _load_value_net_checkpoint(path: str, expected_kind: str) -> dict:
    """Load a checkpoint written by _save_value_net_checkpoint(), raising a
    clear SystemExit if it was saved by a different branch (kind mismatch)
    rather than letting load_state_dict() fail with an opaque shape error."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("kind") != expected_kind:
        raise SystemExit(
            f"--init-value-checkpoint {path} was saved as kind={ckpt.get('kind')!r}, expected "
            f"{expected_kind!r} -- checkpoints from --physics-encoder-value-net and the fresh-"
            f"ExecutionNetwork path are not interchangeable."
        )
    return ckpt


def _peek_file_stats(path) -> tuple[int, int]:
    """Cheap per-file (n_episodes, n_rows), WITHOUT decompressing the big
    observation arrays -- mirrors DemonstrationDataset.from_files()'s own
    "Pass 1" (see its docstring): np.load() on a .npz is lazy per-array, so
    reading just meta_episode_outcomes'/bc_labels' shapes never touches
    obs_other_feat/obs_self_feat/etc. Used by _EpochFilePool to size a
    --max-episodes-per-epoch working set (n_episodes) and by
    PhysicsEncoderValueNet.compute_features_for_files_cached() to know each
    file's row range within a from_files()-built dataset (n_rows), without
    paying the full decompression cost for the whole directory up front --
    the entire point of this feature."""
    with np.load(path) as data:
        n_rows = int(data["bc_labels"].shape[0])
        n_episodes = int(data["meta_episode_outcomes"].shape[0]) if "meta_episode_outcomes" in data.files else n_rows
        return n_episodes, n_rows


class _EpochFilePool:
    """Manages a --max-episodes-per-epoch training file pool: holds a
    "working set" of files sized to reach roughly max_episodes episodes,
    replacing a resample_frac fraction of it with a fresh random draw from
    the rest of the pool every epoch (see resample()). A real class (not
    module functions) since it owns real cross-epoch state (which files are
    currently active, each file's episode/row counts) that would otherwise
    need threading through main()'s closures by hand.

    Kept deliberately simple: file SELECTION isn't reproduced across a
    --init-value-checkpoint resume (only model/optimizer state is) -- a
    resumed run just starts with a fresh random initial sample. Good enough
    since training data composition doesn't need bit-exact reproducibility
    the way model weights do.
    """

    def __init__(self, files: list, max_episodes: int, resample_frac: float, rng: random.Random):
        # Parallelized across a thread pool -- same rationale as
        # DemonstrationDataset.from_files()'s own thread pool (see its
        # _MAX_LOAD_WORKERS): each np.load()+shape-peek is real zlib
        # decompression + zip-directory parsing, and this runs over the
        # ENTIRE training file pool (thousands of files) once at startup.
        import concurrent.futures
        _n_workers = min(8, _os.cpu_count() or 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=_n_workers) as ex:
            stats = dict(zip(files, ex.map(_peek_file_stats, files)))
        self._episode_counts = {f: n_ep for f, (n_ep, _n_rows) in stats.items()}
        self.row_counts = {f: n_rows for f, (_n_ep, n_rows) in stats.items()}
        self._pool = list(files)
        self._rng = rng
        self._max_episodes = max_episodes
        self._resample_frac = resample_frac
        self._working_set: list = []
        self._fill_to_target()

    def _fill_to_target(self) -> None:
        candidates = [f for f in self._pool if f not in self._working_set]
        self._rng.shuffle(candidates)
        total = sum(self._episode_counts[f] for f in self._working_set)
        for f in candidates:
            if total >= self._max_episodes:
                break
            self._working_set.append(f)
            total += self._episode_counts[f]

    def resample(self) -> None:
        """Call at the start of every epoch AFTER the first (the initial
        working set from __init__ is used for that one)."""
        n_drop = round(self._resample_frac * len(self._working_set))
        if n_drop > 0:
            drop = set(self._rng.sample(self._working_set, n_drop))
            self._working_set = [f for f in self._working_set if f not in drop]
        self._fill_to_target()

    @property
    def files(self) -> list:
        return list(self._working_set)

    @property
    def n_episodes(self) -> int:
        return sum(self._episode_counts[f] for f in self._working_set)


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
    parser.add_argument("--physics-encoder-value-net", action="store_true", default=False,
                        help="Replace the normal ExecutionNetwork-based value_net with a tiny "
                             "MLP over two FROZEN, independently-pretrained physics encoders' "
                             "latents (ball, masked to zero whenever the ball isn't loose; "
                             "trainee's own player state, unmasked) plus time-remaining/elapsed, "
                             "an is-possessed flag, and the opponent's relative position (2 raw "
                             "scalars, not entity attention) -- no decision-network context. "
                             "Diagnostic: isolates whether "
                             "physics-only pretraining already carries reward-to-go signal. "
                             "Requires --physics-ball-checkpoint/--physics-player-checkpoint. "
                             "Requires BC labels for heading/desired-direction/speed-mode "
                             "reconstruction (available from --data demo recordings and from "
                             "--checkpoint rollout collection alike); incompatible with "
                             "--synthetic-ball-out-episodes/--synthetic-timeout-episodes (no "
                             "BC-label source for those rows) and with --reset-value-weights/"
                             "--reset-separate-value-net/--trunk-hidden/--value-hidden-dim/"
                             "--entity-embed-dim (none apply to this net's architecture -- use "
                             "--physics-value-mlp-hidden instead).")
    parser.add_argument("--physics-ball-checkpoint", type=str, default=None,
                        help="Frozen BallDynamicsEncoder checkpoint path, e.g. "
                             "checkpoints/physics_pretrain/ball_encoder_63.midtrain_latest_train.pt. "
                             "Required with --physics-encoder-value-net.")
    parser.add_argument("--physics-player-checkpoint", type=str, default=None,
                        help="Frozen PlayerDynamicsEncoder checkpoint path, e.g. "
                             "checkpoints/physics_pretrain/player_encoder_45.midtrain_latest_train.pt. "
                             "Required with --physics-encoder-value-net.")
    parser.add_argument("--physics-value-mlp-hidden", type=int, default=64,
                        help="Hidden width of the small trainable MLP head on top of the two "
                             "frozen latents + time/possession context. Only meaningful with "
                             "--physics-encoder-value-net.")
    parser.add_argument("--value-checkpoint-dir", type=str, default=None,
                        help="If given, saves THIS script's own value_net training progress here "
                             "every epoch: <dir>/latest.pt (overwritten every epoch) and "
                             "<dir>/best_val.pt (whenever val normalized MSE improves). Works for "
                             "both --physics-encoder-value-net and the default fresh-"
                             "ExecutionNetwork path (not the --checkpoint branch, which loads an "
                             "already-trained PPO checkpoint -- a different, pre-existing concept "
                             "with its own resume semantics). Default: not saved.")
    parser.add_argument("--init-value-checkpoint", type=str, default=None,
                        help="Resume this script's own value_net + optimizer state + epoch/best-"
                             "val bookkeeping from a checkpoint previously written by "
                             "--value-checkpoint-dir (e.g. .../latest.pt or .../best_val.pt). Must "
                             "match the current branch (--physics-encoder-value-net vs the fresh "
                             "path) and, for the physics branch, the same --physics-ball-checkpoint/"
                             "--physics-player-checkpoint/--physics-value-mlp-hidden -- raises a "
                             "clear error rather than a silent shape mismatch otherwise. "
                             "Independent of --checkpoint.")
    parser.add_argument("--max-episodes-per-epoch", type=int, default=None,
                        help="If given, don't load the whole --data directory into RAM up front. "
                             "Instead hold a bounded-size TRAINING working set of files (sized to "
                             "reach roughly this many episodes), and every --epochs-per-reload "
                             "epochs replace a --epoch-resample-frac fraction of it with a fresh "
                             "random draw from the rest of the directory -- caps memory to roughly "
                             "one reload's worth of raw data instead of the entire dataset, at the "
                             "cost of re-reading the swapped-in fraction from disk each reload "
                             "(only the newly-added files are actually re-read from disk -- files "
                             "that survive a resample are spliced in from memory, no re-decompress). "
                             "The VALIDATION set is a SEPARATE, fixed set of files "
                             "held out once at the start and never resampled (comparing 'best val' "
                             "against a moving target would defeat early stopping/checkpointing). "
                             "With --physics-encoder-value-net, each file's frozen-encoder features "
                             "are cached per-file (not per-epoch) the first time that file is drawn, "
                             "so files that survive a resample never get re-encoded. Requires --data "
                             "(incompatible with --checkpoint). --outcome-reweight's per-outcome "
                             "weights are computed ONCE from the initial train working set (assumed "
                             "representative of the full pool) and reused unchanged across every "
                             "later resample. Skips the one-time pre-training dataset-"
                             "distribution/linear-regression/reward-breakdown diagnostics (those "
                             "would only describe the initial sample, not the full pool) -- default "
                             "None = today's behaviour, load the whole directory once, unchanged.")
    parser.add_argument("--epoch-resample-frac", type=float, default=0.5,
                        help="Fraction of the training working set replaced every reload (only "
                             "meaningful with --max-episodes-per-epoch). 0.5 (default) = swap half "
                             "the files each reload. 1.0 = fully fresh random sample each reload. "
                             "0.0 = load once and never resample (degenerates to today's static "
                             "behaviour, just restricted to a --max-episodes-per-epoch-sized subset "
                             "instead of the whole directory).")
    parser.add_argument("--epochs-per-reload", type=int, default=1,
                        help="How many epochs to train on the current resampled working set before "
                             "swapping in the next --epoch-resample-frac fraction (only meaningful "
                             "with --max-episodes-per-epoch). 1 (default) = resample every epoch, "
                             "unchanged from before this flag existed. Raise this if the reload cost "
                             "(logged as 'epoch N data reload:') is a meaningful fraction of your "
                             "actual training time per epoch -- e.g. 5 means the working set only "
                             "gets swapped every 5th epoch, cutting reload overhead 5x at the cost of "
                             "the model seeing a less-frequently-refreshed data sample.")
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

    if args.physics_encoder_value_net:
        if not args.physics_ball_checkpoint or not args.physics_player_checkpoint:
            raise SystemExit("--physics-encoder-value-net requires both "
                              "--physics-ball-checkpoint and --physics-player-checkpoint.")
        if args.synthetic_ball_out_episodes or args.synthetic_timeout_episodes:
            raise SystemExit("--physics-encoder-value-net is incompatible with "
                              "--synthetic-ball-out-episodes/--synthetic-timeout-episodes "
                              "(no BC-label source for those rows).")
        if args.reset_value_weights or args.reset_separate_value_net or args.trunk_hidden or \
           args.value_hidden_dim or args.entity_embed_dim:
            raise SystemExit("--reset-value-weights/--reset-separate-value-net/"
                              "--trunk-hidden/--value-hidden-dim/--entity-embed-dim don't apply "
                              "to --physics-encoder-value-net's architecture -- use "
                              "--physics-value-mlp-hidden instead.")

    if args.max_episodes_per_epoch is not None:
        if args.checkpoint:
            raise SystemExit("--max-episodes-per-epoch requires --data (loading the whole "
                              "directory in one shot is inherent to the --checkpoint branch's "
                              "rollout-collection/warm-start paths) -- incompatible with "
                              "--checkpoint.")
        if not (0.0 <= args.epoch_resample_frac <= 1.0):
            raise SystemExit(f"--epoch-resample-frac must be in [0.0, 1.0], got "
                              f"{args.epoch_resample_frac!r}.")
        if args.max_episodes_per_epoch <= 0:
            raise SystemExit(f"--max-episodes-per-epoch must be positive, got "
                              f"{args.max_episodes_per_epoch!r}.")
        if args.epochs_per_reload <= 0:
            raise SystemExit(f"--epochs-per-reload must be positive, got "
                              f"{args.epochs_per_reload!r}.")
    elif args.epochs_per_reload != 1:
        raise SystemExit("--epochs-per-reload only applies with --max-episodes-per-epoch.")

    from footballcoach.ai.config import load_ai_config

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(int(load_ai_config().get("ppo", {}).get("main_process_torch_threads", 4)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    _resampling_enabled = args.max_episodes_per_epoch is not None

    from footballcoach.ai.bc.dataset import DemonstrationDataset
    from footballcoach.ai.models.decision_network import DecisionNetwork
    from footballcoach.ai.models.execution_network import ExecutionNetwork
    from footballcoach.ai.physics_pretrain.physics_value_net import PhysicsEncoderValueNet

    ckpt_trainer = None
    if args.checkpoint:
        from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

        _ckpt_peek = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        _ckpt_separate_value_net = "value_net" in _ckpt_peek
        del _ckpt_peek
        log.info(f"Checkpoint {args.checkpoint}: separate_value_net="
                 f"{_ckpt_separate_value_net} (auto-detected)")
        ckpt_trainer = PPOTrainer.from_config(
            device=device, inference_only=True,
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
    elif _resampling_enabled:
        ds = None  # not used in this mode -- see the dedicated setup block below
    else:
        ds = DemonstrationDataset.from_directory(args.data)

    if _resampling_enabled:
        _setup_start = time.monotonic()
        all_files = sorted(Path(args.data).glob("*.npz"))
        if not all_files:
            raise SystemExit(f"--max-episodes-per-epoch: no .npz files found in {args.data}")
        _file_rng = random.Random(args.seed)
        _file_rng.shuffle(all_files)
        n_val_files = max(1, round(len(all_files) * args.val_frac))
        val_files, train_files_pool = all_files[:n_val_files], all_files[n_val_files:]
        if not train_files_pool:
            raise SystemExit("--max-episodes-per-epoch: --val-frac leaves no files for the "
                              "training pool -- lower --val-frac or add more recorded files.")
        log.info(f"--max-episodes-per-epoch: {len(all_files):,} files total -- "
                 f"{len(val_files)} reserved for a FIXED val set, {len(train_files_pool)} in "
                 f"the resampled training pool")

        ds_val = DemonstrationDataset.from_files(val_files)
        valid_only = not args.include_immobile_self
        val_idx = ds_val.valid_indices() if valid_only else np.arange(len(ds_val))
        if len(val_idx) == 0:
            raise SystemExit("No val rows -- dataset needs >= 2 complete episodes. "
                              "Record more episodes; lowering --val-frac won't help; check `dones`.")
        returns_val = ds_val.compute_returns(gamma=args.gamma)
        outcome_by_row_val = (
            ds_val.outcome_by_row() if ds_val.has_episode_outcomes
            else np.full(len(ds_val), "n/a", dtype=object)
        )
        log.info(f"  fixed val set: {len(ds_val):,} rows across {ds_val.n_episodes()} episodes "
                 f"({len(val_idx):,} valid_idx rows)")

        file_pool = _EpochFilePool(
            train_files_pool, args.max_episodes_per_epoch, args.epoch_resample_frac, _file_rng,
        )
        file_feature_cache: dict = {}  # persists across epochs -- see PhysicsEncoderValueNet.compute_features_for_files_cached
        # train_file_slices' KEY ORDER is ds_train's actual row order (insertion
        # order == the order files were successfully loaded in) -- every
        # consumer of "what order is ds_train in" (the resample reload below,
        # compute_features_for_files_cached) must use list(train_file_slices)
        # for that, NOT file_pool.files (which is just the target working set,
        # not necessarily in ds_train's row order once resampling reorders
        # kept-vs-newly-loaded files -- see the resample block below).
        ds_train, train_file_slices = DemonstrationDataset.from_files_with_offsets(file_pool.files)
        train_idx = ds_train.valid_indices() if valid_only else np.arange(len(ds_train))
        returns_train = ds_train.compute_returns(gamma=args.gamma)
        outcome_by_row_train = (
            ds_train.outcome_by_row() if ds_train.has_episode_outcomes
            else np.full(len(ds_train), "n/a", dtype=object)
        )
        _reload_cadence = ("every epoch" if args.epochs_per_reload == 1
                           else f"every {args.epochs_per_reload} epochs")
        log.info(f"  initial train working set: {len(file_pool.files)} files, "
                 f"{file_pool.n_episodes} episodes, {len(ds_train):,} rows "
                 f"({len(train_idx):,} valid_idx rows)  [target {args.max_episodes_per_epoch} "
                 f"episodes, {args.epoch_resample_frac:.0%} resampled {_reload_cadence}]")
        log.info(f"  --max-episodes-per-epoch setup: {time.monotonic() - _setup_start:.1f}s")

        # Per-outcome weight MAPPING is computed ONCE here, from the initial
        # train working set (assumed representative of the full pool, per
        # the user's call) -- only the per-row weight ARRAY gets rebuilt on
        # every later resample (see the reload block below), reusing this
        # same fixed mapping rather than recomputing inverse frequencies
        # against each epoch's smaller, noisier sample.
        outcome_reweight_norm_weight: dict | None = None
        outcome_weight_by_row: np.ndarray | None = None
        if args.outcome_reweight:
            outcome_reweight_norm_weight = _compute_outcome_norm_weights(
                outcome_by_row_train[train_idx], args.outcome_reweight_max,
            )
            outcome_weight_by_row = _build_outcome_weight_array(
                outcome_by_row_train, outcome_reweight_norm_weight,
            )
            log.info(f"--- Outcome reweighting enabled (from initial train working set, "
                     f"capped at {args.outcome_reweight_max}x, reused unchanged across "
                     f"resamples): {outcome_reweight_norm_weight} ---")
        synthetic_obs_list: list = []
        synthetic_returns = np.array([], dtype=np.float64)
        synthetic_outcomes = np.array([], dtype=object)
        if args.synthetic_ball_out_episodes > 0 or args.synthetic_timeout_episodes > 0:
            raise SystemExit("--max-episodes-per-epoch + --synthetic-ball-out-episodes/"
                              "--synthetic-timeout-episodes isn't supported yet.")
        log.info("--max-episodes-per-epoch: running the one-time dataset-distribution/reward-"
                 "breakdown/MC-return diagnostics against the INITIAL TRAIN WORKING SET only "
                 "(a resampled sample, not the full pool) -- treat as indicative, not exact.")
        trainee_train_idx = train_idx[ds_train._is_trainee[train_idx] > 0.5]
        _log_dataset_distribution(ds_train, train_idx, returns_train, trainee_valid_idx=trainee_train_idx)
        _log_reward_component_breakdown(ds_train, trainee_train_idx)
        if ds_train.has_reward_components:
            for _outc in ("win", "loss", "ball_out", "invalid", "timeout"):
                _log_reward_component_breakdown(ds_train, trainee_train_idx, outcome_filter=_outc)
        _log_returns_by_outcome(ds_train, trainee_train_idx, returns_train, label="initial train sample")
        _log_episode_total_reward_by_outcome(ds_train, trainee_train_idx, label="initial train sample")
    else:
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
            norm_weight = _compute_outcome_norm_weights(outcome_by_row[train_idx], args.outcome_reweight_max)
            outcome_weight_by_row = _build_outcome_weight_array(outcome_by_row, norm_weight)
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

        # Non-resampling mode: train and val share the same underlying
        # dataset/returns/outcome lookup (just different row slices) --
        # aliasing them onto the ds_train/ds_val names lets _run_epoch()
        # use ONE selection rule (is_val_data) uniformly in both modes,
        # rather than needing its own separate branch for this case.
        ds_train = ds_val = ds
        returns_train = returns_val = returns
        outcome_by_row_train = outcome_by_row_val = outcome_by_row

    # Overridden below (in the physics-encoder / fresh-ExecutionNetwork
    # branches only -- see --init-value-checkpoint) if resuming this
    # script's own prior training progress.
    start_epoch = 1
    resumed_best_val_norm: float | None = None

    if args.physics_encoder_value_net:
        # decision_net is still built (existing call-site contract) but its
        # output is unused by PhysicsEncoderValueNet.forward() -- reuse the
        # checkpoint's own decision_net if one was loaded (--checkpoint given
        # alongside this flag), otherwise a fresh throwaway one, same
        # "frozen, context only" role as the other two branches below.
        decision_net = ckpt_trainer.decision_net if ckpt_trainer is not None else DecisionNetwork.from_config()
        decision_net = decision_net.to(device)
        decision_net.eval()
        for p in decision_net.parameters():
            p.requires_grad_(False)

        _obs_cfg = load_ai_config().get("observation", {})
        _curr_cfg = load_ai_config().get("curriculum", {})
        value_net = PhysicsEncoderValueNet.from_checkpoints(
            args.physics_ball_checkpoint, args.physics_player_checkpoint,
            mlp_hidden=args.physics_value_mlp_hidden,
            live_ball_spin_nn_norm_rad_s=float(_obs_cfg.get("ball_spin_nn_norm_rad_s", 55.0)),
            live_height_norm_m=float(_obs_cfg.get("height_norm_m", 3.0)),
            time_norm_max_s=float(_obs_cfg.get("time_remaining_norm_max_s", 7200.0)),
            max_episode_s=float(_curr_cfg.get("phase1_max_episode_s", 60.0)),
        )
        # MUST move to device before constructing the optimizer -- .to()
        # replaces each Parameter's underlying tensor; an optimizer built
        # from the pre-move parameters would keep training stale CPU copies
        # never touched by any forward/backward pass on GPU.
        value_net = value_net.to(device)
        optimizer = torch.optim.Adam(
            value_net.mlp.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.weight_decay
        )
        n_params = sum(p.numel() for p in value_net.mlp.parameters())
        log.info(f"value_net: physics-encoder diagnostic net (ball+player frozen encoders + "
                 f"{args.physics_value_mlp_hidden}-wide MLP head), trainable_params={n_params:,}")
        # Encoders/aux heads/canonicalization are all frozen -- nothing about
        # their output depends on self.mlp's (the only trainable part)
        # weights, so precompute+cache once over every row that'll ever be
        # looked up instead of recomputing the same frozen pipeline every
        # single epoch. --max-episodes-per-epoch needs TWO separate caches
        # (the static val set's, and the per-epoch-resampled train set's --
        # see _run_epoch's is_val_data-keyed swap of value_net._feature_cache
        # right before each call) since ds_train's row layout changes every
        # epoch while ds_val's never does; outside that mode ds_train IS
        # ds_val (aliased above), so one combined cache covers both, exactly
        # like before this feature existed.
        _precompute_start = time.monotonic()
        if _resampling_enabled:
            val_feature_cache = value_net.precompute_and_cache_features(ds_val, val_idx)
            train_feature_cache = value_net.compute_features_for_files_cached(
                ds_train, list(train_file_slices), file_pool.row_counts, file_feature_cache,
            )
            log.info(f"  precomputed+cached physics-encoder features for val ({len(val_idx):,} "
                     f"rows) + initial train working set ({len(ds_train):,} rows) in "
                     f"{time.monotonic() - _precompute_start:.1f}s")
        else:
            train_feature_cache = val_feature_cache = value_net.precompute_and_cache_features(
                ds_train, np.concatenate([train_idx, val_idx]),
            )
            log.info(f"  precomputed+cached physics-encoder features for "
                     f"{len(train_idx) + len(val_idx):,} rows in "
                     f"{time.monotonic() - _precompute_start:.1f}s")
        if args.init_value_checkpoint:
            _vckpt = _load_value_net_checkpoint(args.init_value_checkpoint, "physics_encoder_value_net")
            if (_vckpt.get("physics_ball_checkpoint") != args.physics_ball_checkpoint or
                    _vckpt.get("physics_player_checkpoint") != args.physics_player_checkpoint or
                    _vckpt.get("physics_value_mlp_hidden") != args.physics_value_mlp_hidden):
                raise SystemExit(
                    f"--init-value-checkpoint {args.init_value_checkpoint} was trained with "
                    f"physics_ball_checkpoint={_vckpt.get('physics_ball_checkpoint')!r}, "
                    f"physics_player_checkpoint={_vckpt.get('physics_player_checkpoint')!r}, "
                    f"physics_value_mlp_hidden={_vckpt.get('physics_value_mlp_hidden')!r} -- "
                    f"doesn't match the current run's --physics-ball-checkpoint/"
                    f"--physics-player-checkpoint/--physics-value-mlp-hidden. Resuming into a "
                    f"differently-shaped/differently-sourced net would silently corrupt training."
                )
            value_net.mlp.load_state_dict(_vckpt["state_dict"])
            optimizer.load_state_dict(_vckpt["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, device)
            start_epoch = int(_vckpt["epoch"]) + 1
            resumed_best_val_norm = float(_vckpt["best_val_norm"])
            log.info(f"  resumed from {args.init_value_checkpoint}: epoch={_vckpt['epoch']} "
                     f"best_val_norm={resumed_best_val_norm:.4f} -- continuing at epoch {start_epoch}")
    elif args.checkpoint:
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
            value_net = value_net.to(device)  # before optimizer construction, see physics-branch comment
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
        decision_net = DecisionNetwork.from_config().to(device)
        decision_net.eval()
        for p in decision_net.parameters():
            p.requires_grad_(False)

        value_net = ExecutionNetwork.from_config(
            trunk_hidden_override=args.trunk_hidden,
            value_hidden_dim_override=args.value_hidden_dim,
            entity_embed_dim_override=args.entity_embed_dim,
        )
        # MUST move to device before constructing the optimizer -- see the
        # physics-encoder branch's identical comment.
        value_net = value_net.to(device)
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
        if args.init_value_checkpoint:
            _vckpt = _load_value_net_checkpoint(args.init_value_checkpoint, "fresh_execution_net")
            if (_vckpt.get("trunk_hidden") != args.trunk_hidden or
                    _vckpt.get("value_hidden_dim") != args.value_hidden_dim or
                    _vckpt.get("entity_embed_dim") != args.entity_embed_dim):
                raise SystemExit(
                    f"--init-value-checkpoint {args.init_value_checkpoint} was trained with "
                    f"trunk_hidden={_vckpt.get('trunk_hidden')!r}, "
                    f"value_hidden_dim={_vckpt.get('value_hidden_dim')!r}, "
                    f"entity_embed_dim={_vckpt.get('entity_embed_dim')!r} -- doesn't match the "
                    f"current run's --trunk-hidden/--value-hidden-dim/--entity-embed-dim. Resuming "
                    f"into a differently-shaped net would silently corrupt training."
                )
            value_net.load_state_dict(_vckpt["state_dict"])
            optimizer.load_state_dict(_vckpt["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, device)
            start_epoch = int(_vckpt["epoch"]) + 1
            resumed_best_val_norm = float(_vckpt["best_val_norm"])
            log.info(f"  resumed from {args.init_value_checkpoint}: epoch={_vckpt['epoch']} "
                     f"best_val_norm={resumed_best_val_norm:.4f} -- continuing at epoch {start_epoch}")

    ret_std_train = float(np.std(returns_train[train_idx])) if len(train_idx) else 1.0
    ret_var_train = max(ret_std_train ** 2, 1e-6)

    from footballcoach.ai.ppo.bc import (
        AI_TYPE_IMMOBILE, AI_TYPE_NEURAL, AI_TYPE_RULES, _I_OPPONENT_AI_TYPE,
    )
    _OPP_TYPE_NAME = {AI_TYPE_RULES: "rules", AI_TYPE_IMMOBILE: "immobile", AI_TYPE_NEURAL: "neural"}

    def _run_epoch(
        idx: np.ndarray, train: bool, is_val_data: bool = False,
    ) -> tuple[float, float, dict[str, tuple[float, int]], dict[str, tuple[float, int]], dict[str, float] | None]:
        """Return (raw_mse, normalized_mse, per_opponent_type, per_outcome,
        grad_norm_stats) averaged over all rows in idx. grad_norm_stats is
        {"mean": ..., "max": ...} (the pre-clip gradient L2 norm from every
        optimizer.step() this epoch) when train=True, else None (no backward
        pass ran). Both breakdown dicts map name ->
        (raw_mse, n_rows). When train=True and synthetic rows were generated
        (--synthetic-ball-out-episodes), those rows are mixed into the SAME
        training batches (not a separate pass) -- they're tagged with a
        "synthetic_" prefix on their real (ball_out/win) outcome in the
        per-outcome breakdown so their contribution stays visible without a
        second training loop.

        ``is_val_data`` selects WHICH dataset ``idx`` indexes into
        (ds_val/returns_val/outcome_by_row_val vs. ds_train/returns_train/
        outcome_by_row_train) -- deliberately separate from ``train`` (which
        only controls whether gradients/optimizer.step() happen): the
        epoch-0 baseline call below evaluates train_idx with train=False
        (pure forward pass, no gradient) but it's still TRAIN data. Outside
        --max-episodes-per-epoch, ds_train IS ds_val (same object, aliased)
        so this selection is a no-op either way. For --physics-encoder-value-net,
        also swaps value_net's active feature cache to match (train_feature_cache
        is rebuilt fresh every epoch by the caller; val_feature_cache is
        static) -- see PhysicsEncoderValueNet.compute_features_for_files_cached()."""
        from footballcoach.ai.progress import ProgressReporter

        ds_cur = ds_val if is_val_data else ds_train
        returns_cur = returns_val if is_val_data else returns_train
        outcome_by_row_cur = outcome_by_row_val if is_val_data else outcome_by_row_train
        if isinstance(value_net, PhysicsEncoderValueNet):
            value_net._feature_cache = val_feature_cache if is_val_data else train_feature_cache

        total_sq_err = 0.0
        n_rows = 0
        opp_sq_err: dict[str, float] = {}
        opp_n_rows: dict[str, int] = {}
        outc_sq_err: dict[str, float] = {}
        outc_n_rows: dict[str, int] = {}
        # Pre-clip gradient L2 norm (clip_grad_norm_ always returns this,
        # whether or not it actually clips) -- tracked per optimizer.step()
        # to help diagnose "loss isn't moving": a norm collapsing toward 0
        # means vanishing gradients (nothing left to learn from), a norm
        # pinned near/above the clip threshold (1.0, see below) every step
        # means clipping is constantly kicking in and likely bottlenecking
        # progress, neither of which is visible from the loss curve alone.
        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        n_grad_steps = 0
        value_net.train(train)
        epoch_idx = idx.copy()
        if train:
            np.random.shuffle(epoch_idx)

        def _step(obs_dict, ret_batch, opp_type_col, outcome_col, row_weights):
            nonlocal total_sq_err, n_rows, grad_norm_sum, grad_norm_max, n_grad_steps
            with torch.set_grad_enabled(train):
                with torch.no_grad():
                    d_heads = decision_net(
                        obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
                        obs_dict["ball_feat"], obs_dict["global_feat"],
                    )
                extra_kwargs = (
                    {"labels": obs_dict.get("labels"), "row_idx": obs_dict.get("row_idx")}
                    if isinstance(value_net, PhysicsEncoderValueNet) else {}
                )
                e_heads = value_net(
                    obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
                    obs_dict["ball_feat"], obs_dict["global_feat"], d_heads,
                    obs_dict.get("self_ai_type"), obs_dict.get("other_ai_type"),
                    **extra_kwargs,
                )
                pred = e_heads.value.squeeze(-1)
                per_row_sq = (pred - ret_batch) ** 2
                if train and row_weights is not None:
                    loss = (torch.from_numpy(row_weights).to(device) * per_row_sq).mean()
                else:
                    loss = per_row_sq.mean()
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    total_norm = float(torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0))
                    optimizer.step()
                    grad_norm_sum += total_norm
                    grad_norm_max = max(grad_norm_max, total_norm)
                    n_grad_steps += 1
            # Reported/logged MSE is always the UNWEIGHTED mean squared error
            # (per_row_sq), even when the backward pass above used a weighted
            # loss -- keeps every logged number comparable across runs regardless
            # of --outcome-reweight.
            per_row_sq_err = per_row_sq.detach()
            total_sq_err += float(per_row_sq_err.sum())
            n_rows += len(ret_batch)
            # .cpu() before numpy-boolean-mask indexing below -- opp_type_col/
            # outcome_col are plain numpy arrays, and indexing/",numpy()"-ing a
            # CUDA tensor with/into one raises; harmless no-op on CPU.
            per_row_sq_err_np = per_row_sq_err.cpu().numpy()
            if opp_type_col is not None:
                for code, name in _OPP_TYPE_NAME.items():
                    row_mask = opp_type_col == code
                    if not row_mask.any():
                        continue
                    opp_sq_err[name] = opp_sq_err.get(name, 0.0) + float(per_row_sq_err_np[row_mask].sum())
                    opp_n_rows[name] = opp_n_rows.get(name, 0) + int(row_mask.sum())
            for outcome in np.unique(outcome_col):
                row_mask_np = outcome_col == outcome
                outc_sq_err[outcome] = outc_sq_err.get(outcome, 0.0) + float(
                    per_row_sq_err_np[row_mask_np].sum()
                )
                outc_n_rows[outcome] = outc_n_rows.get(outcome, 0) + int(row_mask_np.sum())

        progress = ProgressReporter(len(epoch_idx), prefix="  val:   " if is_val_data else "  train: ")
        for obs_dict, ret_batch, chunk in _iterate_over_with_indices(
            ds_cur, epoch_idx, returns_cur, args.batch_size, device=device,
        ):
            row_weights = (
                outcome_weight_by_row[chunk] if train and outcome_weight_by_row is not None else None
            )
            _step(
                obs_dict, ret_batch, ds_cur._labels[chunk, _I_OPPONENT_AI_TYPE],
                outcome_by_row_cur[chunk], row_weights,
            )
            progress.update(n_rows, postfix=f"loss={total_sq_err / max(n_rows, 1):.4f}")

        if train and synthetic_obs_list:
            synthetic_outcome_col_full = np.array(
                [f"synthetic_{o}" for o in synthetic_outcomes], dtype=object
            )
            for obs_dict, ret_batch, chunk in _iterate_synthetic_batches(
                synthetic_obs_list, synthetic_returns, args.batch_size, shuffle=True, device=device,
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
        grad_norm_stats = (
            {"mean": grad_norm_sum / n_grad_steps, "max": grad_norm_max}
            if n_grad_steps > 0 else None
        )
        return raw_mse, raw_mse / ret_var_train, per_opponent_type, per_outcome, grad_norm_stats

    log.info(f"Fitting fresh separate value network: {args.epochs} epochs, "
             f"lr={args.lr}, weight_decay={args.weight_decay}, batch_size={args.batch_size}, "
             f"train_ret_std={ret_std_train:.3f}, outcome_reweight={args.outcome_reweight}")

    # RMSE is in the same units as the MC return itself (MSE is squared
    # units, which is why a return range of [-4, 5] can show MSE values
    # like 40).
    def _log_epoch_result(
        epoch_label: str,
        train_raw, train_norm, train_by_opp, train_by_outc, train_grad_norm,
        val_raw, val_norm, val_by_opp, val_by_outc, val_grad_norm,
    ) -> None:
        # train_grad_norm is None on the epoch-0 baseline call (train=False,
        # no backward pass ever ran) -- val_grad_norm is always None (no
        # gradients on val) and isn't printed.
        grad_norm_str = (
            f"  grad_norm={train_grad_norm['mean']:.4f} (max={train_grad_norm['max']:.4f})"
            if train_grad_norm is not None else ""
        )
        log.info(
            f"{epoch_label}  "
            f"train_rmse={math.sqrt(train_raw):.4f} (norm={math.sqrt(train_norm):.4f})  "
            f"val_rmse={math.sqrt(val_raw):.4f} (norm={math.sqrt(val_norm):.4f})"
            f"{grad_norm_str}"
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
        *_run_epoch(train_idx, train=False, is_val_data=False),
        *_run_epoch(val_idx, train=False, is_val_data=True),
    )

    def _save_value_net_now(epoch: int, current_best_val_norm: float, filename: str) -> None:
        if not args.value_checkpoint_dir:
            return
        if isinstance(value_net, PhysicsEncoderValueNet):
            _save_value_net_checkpoint(
                f"{args.value_checkpoint_dir}/{filename}", "physics_encoder_value_net",
                value_net.mlp.state_dict(), optimizer, epoch, current_best_val_norm,
                extra_meta={
                    "physics_ball_checkpoint": args.physics_ball_checkpoint,
                    "physics_player_checkpoint": args.physics_player_checkpoint,
                    "physics_value_mlp_hidden": args.physics_value_mlp_hidden,
                },
            )
        elif not args.checkpoint:
            _save_value_net_checkpoint(
                f"{args.value_checkpoint_dir}/{filename}", "fresh_execution_net",
                value_net.state_dict(), optimizer, epoch, current_best_val_norm,
                extra_meta={
                    "trunk_hidden": args.trunk_hidden,
                    "value_hidden_dim": args.value_hidden_dim,
                    "entity_embed_dim": args.entity_embed_dim,
                },
            )
        else:
            log.warning("--value-checkpoint-dir has no effect with --checkpoint (the PPO-"
                        "checkpoint-warm-start branch) -- only --physics-encoder-value-net and "
                        "the fresh-ExecutionNetwork path support saving this script's own "
                        "training progress.")

    def _run_val_diagnostics(epoch_label: str) -> None:
        """Recompute per-row val residuals (a full forward pass over
        val_idx, separate from _run_epoch's own val pass since that only
        keeps the running loss, not per-row predictions) and run the
        component-correlation/feature-correlation/worst-episode diagnostics
        against them. Called periodically during training (see the epoch
        loop, same cadence as --epochs-per-reload) and once more after the
        loop ends, so a long run gets visibility into how these correlations
        evolve instead of only a single snapshot at the very end."""
        residual_by_row = np.zeros(len(ds_val), dtype=np.float32)
        value_net.eval()
        if isinstance(value_net, PhysicsEncoderValueNet):
            # Defensive, not load-bearing given _run_epoch's own call order
            # (it always ends on an is_val_data=True call, which already
            # leaves this set correctly) -- but don't rely on that.
            value_net._feature_cache = val_feature_cache
        with torch.no_grad():
            for obs_dict, ret_batch, chunk in _iterate_over_with_indices(
                ds_val, val_idx, returns_val, args.batch_size, device=device,
            ):
                d_heads = decision_net(
                    obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
                    obs_dict["ball_feat"], obs_dict["global_feat"],
                )
                extra_kwargs = (
                    {"labels": obs_dict.get("labels"), "row_idx": obs_dict.get("row_idx")}
                    if isinstance(value_net, PhysicsEncoderValueNet) else {}
                )
                e_heads = value_net(
                    obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
                    obs_dict["ball_feat"], obs_dict["global_feat"], d_heads,
                    obs_dict.get("self_ai_type"), obs_dict.get("other_ai_type"),
                    **extra_kwargs,
                )
                pred = e_heads.value.squeeze(-1).cpu().numpy()
                residual_by_row[chunk] = ret_batch.cpu().numpy() - pred

        log.info(f"=== Val diagnostics @ {epoch_label} ===")
        _log_component_correlation(ds_val, val_idx, args.gamma)
        _episode_residual_correlation(ds_val, val_idx, args.gamma, residual_by_row)
        _log_feature_error_correlation(ds_val, val_idx, returns_val, residual_by_row)
        _save_worst_episode_match_log(
            ds_val, val_idx, residual_by_row, args.worst_episode_log_path,
            returns_by_row=returns_val,
        )

    best_val_norm = resumed_best_val_norm if resumed_best_val_norm is not None else float("inf")
    _patience_ctr = 0
    for epoch in range(start_epoch, args.epochs + 1):
        if _resampling_enabled and epoch > start_epoch and (epoch - start_epoch) % args.epochs_per_reload == 0:
            # Skip on the very first epoch of a run (whether fresh or
            # resumed) -- file_pool's own __init__ already gave us that
            # epoch's initial working set; resampling from epoch 2 onward
            # avoids throwing away half of what was just loaded before
            # training on it even once. --epochs-per-reload > 1 further
            # skips this on every epoch except every Nth one (counted from
            # start_epoch, so a --init-value-checkpoint resume mid-cycle
            # doesn't matter -- it just restarts the N-epoch counter from
            # wherever it resumed, same as a fresh run would from epoch 0).
            # Incremental reload: only files NEW to the working set this epoch
            # get re-read from disk (DemonstrationDataset.from_files_with_offsets()
            # -- the expensive, zlib-decompressing step). Files that survive
            # the resample get their rows sliced straight out of the CURRENT
            # ds_train (still alive at this point -- see concat_slices()) and
            # spliced into the new one via concat_slices(), no disk I/O at
            # all. Previously this called from_files() on the ENTIRE working
            # set every epoch regardless of how much of it actually changed
            # -- with a typical --epoch-resample-frac (e.g. 0.3), that meant
            # re-decompressing ~70% of the pool from scratch every single
            # epoch for no reason.
            _reload_start = time.monotonic()
            file_pool.resample()
            kept_files = [f for f in file_pool.files if f in train_file_slices]
            new_files = [f for f in file_pool.files if f not in train_file_slices]
            pieces = [(ds_train, train_file_slices[f]) for f in kept_files]
            piece_files = list(kept_files)
            if new_files:
                ds_new, new_file_slices = DemonstrationDataset.from_files_with_offsets(new_files)
                for f in new_files:
                    if f in new_file_slices:  # defensive: skip any unreadable-file race, see from_files()
                        pieces.append((ds_new, new_file_slices[f]))
                        piece_files.append(f)
            n_new_files = len(piece_files) - len(kept_files)
            ds_train = DemonstrationDataset.concat_slices(pieces)
            # Rebuild the offset map for the NEW ds_train -- its row order is
            # piece_files' order (kept files first, then newly-loaded ones),
            # which is generally NOT file_pool.files' order, so any later
            # consumer of "what order is ds_train in" must use
            # list(train_file_slices), not file_pool.files.
            train_file_slices = {}
            _off = 0
            for f, (_src, sl) in zip(piece_files, pieces):
                n_rows = sl.stop - sl.start
                train_file_slices[f] = slice(_off, _off + n_rows)
                _off += n_rows
            train_idx = ds_train.valid_indices() if valid_only else np.arange(len(ds_train))
            returns_train = ds_train.compute_returns(gamma=args.gamma)
            outcome_by_row_train = (
                ds_train.outcome_by_row() if ds_train.has_episode_outcomes
                else np.full(len(ds_train), "n/a", dtype=object)
            )
            if outcome_reweight_norm_weight is not None:
                # Reuse the FIXED mapping computed once from the initial
                # working set (see setup above) -- only the per-row array
                # needs rebuilding, since ds_train's row layout just changed.
                outcome_weight_by_row = _build_outcome_weight_array(
                    outcome_by_row_train, outcome_reweight_norm_weight,
                )
            if isinstance(value_net, PhysicsEncoderValueNet):
                train_feature_cache = value_net.compute_features_for_files_cached(
                    ds_train, piece_files, file_pool.row_counts, file_feature_cache,
                )
            log.info(
                f"  epoch {epoch} data reload: {time.monotonic() - _reload_start:.1f}s "
                f"({n_new_files} new file(s) loaded from disk, {len(kept_files)} kept in "
                f"memory from last epoch (no re-read/re-decompress, encoder features "
                f"reused), {len(ds_train):,} rows, {file_pool.n_episodes} episodes)"
            )

        train_raw, train_norm, train_by_opp, train_by_outc, train_grad_norm = _run_epoch(
            train_idx, train=True, is_val_data=False,
        )
        val_raw, val_norm, val_by_opp, val_by_outc, val_grad_norm = _run_epoch(
            val_idx, train=False, is_val_data=True,
        )
        _log_epoch_result(
            f"epoch {epoch:3d}/{args.epochs}",
            train_raw, train_norm, train_by_opp, train_by_outc, train_grad_norm,
            val_raw, val_norm, val_by_opp, val_by_outc, val_grad_norm,
        )
        # Same cadence as the data reload above (both keyed off
        # --epochs-per-reload, default 1 = every epoch outside
        # --max-episodes-per-epoch) -- these diagnostics involve their own
        # full forward pass over val_idx (to get per-row predictions, not
        # just the running loss _run_epoch already computed), so tying them
        # to the reload cadence keeps a long run's overhead bounded instead
        # of paying for correlation/worst-episode analysis every epoch.
        if (epoch - start_epoch) % args.epochs_per_reload == 0:
            _run_val_diagnostics(f"epoch {epoch}/{args.epochs}")
        is_best = val_norm < best_val_norm
        if is_best:
            best_val_norm = val_norm
            _patience_ctr = 0
        else:
            _patience_ctr += 1
        _save_value_net_now(epoch, best_val_norm, "latest.pt")
        if is_best:
            _save_value_net_now(epoch, best_val_norm, "best_val.pt")
        if not is_best and args.patience > 0 and _patience_ctr >= args.patience:
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

    # Final diagnostics snapshot for the truly-last epoch reached (natural
    # completion or early stopping) -- may repeat the last periodic call
    # from inside the loop if the run happened to stop on a reload-cadence
    # epoch, which is harmless.
    _run_val_diagnostics("final")

    if synthetic_obs_list:
        _score_synthetic_ball_out_with_net(
            decision_net, value_net, synthetic_obs_list, synthetic_returns, synthetic_outcomes,
            device=device,
        )


if __name__ == "__main__":
    main()

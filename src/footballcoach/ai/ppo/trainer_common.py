"""Module-level helpers shared by ppo_trainer.py and its PPOTrainer mixin modules."""
from __future__ import annotations

import logging
import torch
import dataclasses
import math
from typing import Optional
import numpy as np
from footballcoach.ai.action.schema import DecisionAction, DecisionHeadsRaw
from footballcoach.entities.action_opportunity import OPP_ACTION_KEYS


log = logging.getLogger("footballcoach.ai.ppo")

def _load_state_dict_tolerant(module: torch.nn.Module, ckpt_sd: dict, label: str) -> None:
    """Load ckpt_sd into module, skipping any param whose shape no longer
    matches (e.g. value_head grew an extra input column since the
    checkpoint was saved) instead of raising. Skipped params keep their
    fresh random init; everything else loads normally. Lets checkpoints
    saved under an older network architecture keep loading after additive
    changes to value_head/value_ai_type_channel sizing, at the cost of
    those specific params needing to be retrained (typically fine, since
    such changes are usually paired with --reset-value-weights anyway)."""
    own_sd = module.state_dict()
    compatible = {}
    skipped = []
    for k, v in ckpt_sd.items():
        if k in own_sd and own_sd[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)
    module.load_state_dict(compatible, strict=False)
    if skipped:
        log.warning(
            f"{label}: skipped {len(skipped)} shape-mismatched param(s) "
            f"(network architecture changed since checkpoint was saved), "
            f"keeping fresh init for: {skipped}"
        )

# Reward component short-key → display label mapping (order = display order).
# Used by both the per-rollout log and the pre-training diagnostic in train.py.
REWARD_COMP_LABELS: list[tuple[str, str]] = [
    ("out",   "ball_out"),
    ("box",   "box_possession"),
    ("spd",   "speed_bonus"),
    ("lterm", "opponent_box"),
    ("tout",  "timeout"),
    ("sprint", "sprint_penalty"),
    ("appr",  "approach"),
    ("prog",  "progress"),
]

# All phase-1 StepInfo.trial_outcome values (see ScenarioEnv.step()) that
# outcome_breakdown() below always reports a percentage for, even when a
# given rollout/eval happens to have zero of them (0% rather than silently
# omitted) — win/loss stay first for backward-compat with old log-scrapers
# that split on "/".
_PHASE1_OUTCOME_KEYS: list[tuple[str, str]] = [
    ("box_possession", "win"),
    ("opponent_box_possession", "loss"),
    ("timeout", "tout"),
    ("miss", "miss"),
    ("invalid", "inval"),
]

def outcome_breakdown(outcomes: list[str]) -> str:
    """Format a list of StepInfo.trial_outcome strings as
    "win%/loss%/tout%/miss%/inval%[/other%]" -- a fuller breakdown than just
    win/loss so a swing in win% can be traced to (e.g.) more timeouts vs.
    more losses vs. more ball-out-of-play, instead of both being lumped into
    an invisible remainder. "inval" is a ball-out with no toucher at all
    (nobody's fault, e.g. a bad initial placement) as opposed to "miss"
    (the last toucher is blamed). 'other' only appears if some outcome value
    isn't one of the known keys above (e.g. a phase-2 "goal"/"dispossessed"
    leaking through, or an "unknown" from a missing info object).
    """
    n = len(outcomes)
    if n == 0:
        return "n/a"
    parts = [f"{outcomes.count(key) / n * 100:.1f}%" for key, _ in _PHASE1_OUTCOME_KEYS]
    known = sum(outcomes.count(key) for key, _ in _PHASE1_OUTCOME_KEYS)
    other = n - known
    if other > 0:
        parts.append(f"{other / n * 100:.0f}%")
    return "/".join(parts)

# Slash-joined short labels matching outcome_breakdown()'s column order, for
# a one-time "vs[...]:" legend prefix on log lines instead of repeating the
# key names on every vs_rules/vs_immobile/vs_neural segment.
_PHASE1_OUTCOME_LEGEND = "/".join(label for _, label in _PHASE1_OUTCOME_KEYS)


_DECISION_HEADS_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(DecisionHeadsRaw))

def _detach_decision_heads(d_heads: DecisionHeadsRaw) -> DecisionHeadsRaw:
    """Return a copy of d_heads with every field detached (see separate_value_net).

    ball_physics_full/self_physics_full/other_physics_full (see
    "Frozen physics-dynamics encoders" in ai/knowledge.md) are None when
    that feature is disabled (the default) -- skip detach() for those,
    same None-passthrough every other consumer of DecisionHeadsRaw already
    has to handle."""
    return dataclasses.replace(
        d_heads, **{
            name: (v.detach() if (v := getattr(d_heads, name)) is not None else None)
            for name in _DECISION_HEADS_FIELD_NAMES
        }
    )

# Execution-network head name -> nn.Module attribute name, used for the
# per-head gradient-norm / logit-drift diagnostics in _ppo_update().
_GRAD_NORM_SPIKE_TOP_K = 15  # how many params to show in a grad-norm-spike breakdown, see PPOTrainer.grad_norm_spike_k

_GRAD_NORM_SPIKE_MIN_SAMPLES = 20  # don't check for spikes until the running history has at least this many samples

EXEC_HEAD_MODULES: list[tuple[str, str]] = [
    ("move_direction", "move_direction"),
    ("exec_move", "exec_move_logit"),
    ("sprint", "sprint_logit"),
    ("kick", "kick_logit"),
    ("kick_direction", "kick_direction"),
    ("kick_power", "kick_power"),
    ("kick_spin", "kick_spin"),
    ("tackle_attempt", "tackle_attempt_logit"),
]

def _maybe_log_grad_norm_spike(
    trainer: "PPOTrainer", label: str, epoch_i: int, mb_i: int, raw_grad_norm: float,
) -> None:
    """Adaptive grad-norm-spike diagnostic, shared by PPO/BC/DAgger's
    minibatch loops (_ppo_update, pretrain_combined()'s Phase 1, dagger.py's
    train_bc_epochs()). Maintains a bounded rolling history of raw grad
    norms on the trainer (trainer._grad_norm_history, most-recent
    _GRAD_NORM_SPIKE_HISTORY_MAXLEN samples pooled across ALL THREE training
    loops -- deliberately not reset per-epoch/per-phase, since "is this
    minibatch's grad norm unusually large FOR THIS RUN" is the question, and
    a brand new run naturally starts the window empty again anyway).

    Logs a [grad norm SPIKE] DEBUG line (visible with train.py's --verbose
    flag) with a top-K per-parameter-tensor breakdown whenever raw_grad_norm
    exceeds (running_mean +
    trainer.grad_norm_spike_k * running_std) of that history -- adaptive to
    whatever scale a given run/config actually produces, instead of a fixed
    absolute number that needs re-tuning any time the typical scale changes
    (confirmed necessary: a real run with mean~5 max~200 needs a very
    different absolute bar than a small smoke-test with mean~4.6 max~14.3).
    Requires at least _GRAD_NORM_SPIKE_MIN_SAMPLES observations before
    checking at all -- an empty/tiny history has a meaningless mean/std and
    would otherwise flag the first few minibatches of every run as "spikes"
    against almost no baseline. The just-observed raw_grad_norm is appended
    to the history AFTER computing the bar, so it never inflates its own
    baseline.

    Must be called with .grad still populated on trainer.decision_net/
    execution_net (i.e. after backward(), before optimizer.step()/
    zero_grad()) -- the per-parameter breakdown reads those .grad tensors
    directly. `label` is a short tag (e.g. "PPO", "BC", "DAgger") included
    in the log line so it's clear which training loop an outlier fired from.
    """
    if trainer.grad_norm_spike_k is None:
        return
    history = trainer._grad_norm_history
    if len(history) >= _GRAD_NORM_SPIKE_MIN_SAMPLES:
        running_mean = float(np.mean(history))
        running_std = float(np.std(history))
        spike_bar = running_mean + trainer.grad_norm_spike_k * running_std
        if raw_grad_norm > spike_bar:
            from footballcoach.ai.obs.schema import BallFeatures, GlobalFeatures, PlayerFeatures
            # "First touch" Linear layers -- the ones whose in_features IS
            # the raw named feature schema (not a hidden/intermediate
            # width) -- keyed by parameter-name SUFFIX (matched via
            # str.endswith below) so both decision_net's and execution_net's
            # copies match regardless of module-path prefix.
            _first_touch_layer_schemas = {
                "entity_encoder.per_entity_mlp.0.weight": PlayerFeatures,
                "entity_encoder.ball_query_proj.weight": BallFeatures,
                "entity_encoder.global_query_proj.weight": GlobalFeatures,
                "self_mlp.0.weight": PlayerFeatures,
                "ball_mlp.0.weight": BallFeatures,
                "global_mlp.0.weight": GlobalFeatures,
            }
            _named_params = [
                (f"{_net_name}.{_n}", _n, _p)
                for _net_name, _net in (("decision_net", trainer.decision_net), ("execution_net", trainer.execution_net))
                for _n, _p in _net.named_parameters()
                if _p.grad is not None
            ]
            _named_grad_norms = [(_full, _p.grad.norm().item()) for _full, _n, _p in _named_params]
            _named_grad_norms.sort(key=lambda kv: kv[1], reverse=True)
            _top_spike = _named_grad_norms[:_GRAD_NORM_SPIKE_TOP_K]
            _spike_str = "  ".join(f"{name}={norm:.2f}" for name, norm in _top_spike)
            # DEBUG, not WARNING -- gated behind train.py's --verbose flag
            # (sets the "footballcoach.ai" parent logger to DEBUG, which
            # this "footballcoach.ai.ppo" child inherits). Was WARNING while
            # this diagnostic was actively hunting the crossing_head/
            # crosses_logit scale bug (see physics_encoders.py's _apply_
            # logit_sigmoid) -- now that that's fixed, remaining spikes are
            # the ordinary "categorical/context flags carry a lot of
            # gradient" pattern, not something worth surfacing by default.
            log.debug(
                f"[grad norm SPIKE][{label}] epoch={epoch_i} mb={mb_i} raw={raw_grad_norm:.2f} "
                f"(running mean={running_mean:.2f} std={running_std:.2f} "
                f"bar=mean+{trainer.grad_norm_spike_k:.1f}*std={spike_bar:.2f}, n={len(history)}) -- "
                f"top {len(_top_spike)} params: {_spike_str}"
            )
            # Per-input-COLUMN gradient norm for whichever "first touch"
            # Linear layers (the ones whose in_features are the raw named
            # feature schema itself, not a hidden/intermediate width) appear
            # among this spike's parameters -- a genuine causal decomposition
            # of the SAME weight.grad tensor already read above (weight.grad
            # has shape (out_features, in_features), and column i is BY
            # CONSTRUCTION exactly the gradient signal attributable to input
            # feature i: grad_W[j,i] = sum_batch grad_output[b,j]*input[b,i]).
            # Free -- no extra backward pass, no hooks, no per-sample loop,
            # just a different reduction on a tensor we already have. This is
            # the principled alternative to the raw-feature-range heuristic
            # below, which only measures coincidental correlation with
            # magnitude and can mislead (e.g. distance_m/ball_distance_m
            # routinely reach ~2.0 for two entities near opposite pitch
            # corners -- a mundane geometric fact, unrelated to gradient
            # magnitude, that would otherwise look like a smoking gun).
            _seen_param_ids: set[int] = set()
            for _full, _n, _p in _named_params:
                if id(_p) in _seen_param_ids:
                    continue  # shared tensor (e.g. share_entity_encoder) already reported once
                for _suffix, _schema in _first_touch_layer_schemas.items():
                    if _n.endswith(_suffix):
                        _seen_param_ids.add(id(_p))
                        _cols_str = _describe_weight_grad_columns(_p, _schema)
                        if _cols_str:
                            log.debug(f"  [grad norm SPIKE][{label}] {_full} per-input-column grad norm: {_cols_str}")
                        break
    history.append(raw_grad_norm)

_GRAD_COLUMN_TOP_K = 5  # how many named input columns to show per weight tensor in a grad-norm-spike breakdown

def _describe_weight_grad_columns(
    weight: "torch.Tensor", schema_dataclass: "type | None", top_k: int = _GRAD_COLUMN_TOP_K,
) -> str:
    """Per-INPUT-COLUMN gradient norm for a Linear layer's weight tensor
    (shape (out_features, in_features)). weight.grad[:, i] is, BY
    CONSTRUCTION, exactly the gradient signal attributable to input feature
    i (grad_W[j,i] = sum_batch grad_output[b,j] * input[b,i]) -- so this is
    a genuine causal decomposition of an already-computed gradient tensor,
    unlike _describe_feature_extremes' raw-magnitude heuristic above, which
    only measures coincidental correlation with input VALUE and can mislead
    (confirmed in practice: distance_m/ball_distance_m routinely reach ~2.0
    for two entities merely near opposite pitch corners -- a mundane
    geometric fact with no causal relationship to gradient magnitude, that
    still looked like a smoking gun under the magnitude-only heuristic).

    Free to compute -- reads weight.grad.norm(dim=0), no extra backward
    pass, no hooks, no per-sample loop, just a different reduction on a
    tensor already sitting in memory after backward().

    Only meaningful for a layer's FIRST touch of named raw features -- a
    schema_dataclass is expected to have exactly as many fields as
    weight.shape[1] (in_features); if it doesn't (e.g. physics-encoder
    features concatenated on, extending in_features beyond the named
    schema -- see DecisionNetwork's self_dim_eff/ball_dim_eff), falls back
    to anonymous "col_i" labels rather than mis-naming columns.
    """
    if weight.grad is None:
        return ""
    col_norms = weight.grad.norm(dim=0)
    names = None
    if schema_dataclass is not None:
        _candidate_names = [f.name for f in dataclasses.fields(schema_dataclass)]
        if len(_candidate_names) == col_norms.shape[0]:
            names = _candidate_names
    if names is None:
        names = [f"col_{i}" for i in range(col_norms.shape[0])]
    k = min(top_k, len(names))
    top_vals, top_idx = col_norms.topk(k)
    return "  ".join(f"{names[i]}={v:.3f}" for i, v in zip(top_idx.tolist(), top_vals.tolist()))

def _ai_types(obs_dict: dict) -> tuple:
    """Extract (self_ai_type, other_ai_type) tensors from an obs dict, or
    (None, None) if absent — DecisionNetwork/ExecutionNetwork.forward()
    default to all-zero one-hots in that case. Centralised here so every
    ``decision_net(...)``/``execution_net(...)`` call site in this file uses
    the identical fallback behaviour. See ai/knowledge.md "Opponent-AI-type
    (value-only)".
    """
    return obs_dict.get("self_ai_type"), obs_dict.get("other_ai_type")

def _episode_abs_adv_means(
    track_ids: list, dones, advantages,
) -> list[float]:
    """Per-episode mean(|advantage|), for episode-seed replay (see
    ``PPOTrainer._update_episode_replay``'s highest-mean-|advantage| seed
    selection). Segments on the TRAINEE's own ``done > 0.5`` events (one
    entry per completed trainee episode, in order) so the result lines up
    1:1 with ``stats["episode_rewards"]``/``stats["episode_seeds"]`` --
    those are populated the same way, only ever appended when the
    trainee's own ``done`` fires. A trailing incomplete episode (no
    terminal trainee ``done=1`` row yet) is silently dropped, matching
    those two stats lists' own behavior.

    Unlike the previous (``_trainee_episode_abs_adv_means``) version of
    this function, each episode's mean is now taken over EVERY row in that
    episode's span -- the trainee's own rows AND any interleaved
    secondary-track rows -- not just the trainee's. This is deliberate,
    not an oversight: whenever a secondary/opponent row exists in the
    buffer at all, it is -- today -- ALWAYS the current, live network
    playing itself. See ai/ppo/batched_rollout_worker.py's "Secondary
    neural players" docstring section: ``_batched_worker_main`` passes
    ``secondary_trainer=trainer``, the exact same weights as the trainee's
    own network, and ``_secondary_neural_candidates()`` excludes
    rules-based/immobile secondaries entirely (they never had a
    ``NeuralPlayerAI``/``last_transition`` to add to the buffer in the
    first place) -- so a rules/immobile-opponent episode simply never puts
    a non-"trainee" row in the buffer, and this function reduces to
    trainee-only for those episodes automatically, with no explicit
    filtering needed. A secondary row's own ``done`` is always set to the
    SAME shared, env-level ``done`` as the trainee's for that tick (see
    scenario_env.py's ``last_secondary_results`` construction -- one trial
    ends the same way for every player on the pitch at once), so every
    secondary row for an episode arrives strictly between that episode's
    previous trainee-``done`` row and its own trainee-``done`` row (the
    trainee row for a given tick is always added to the buffer BEFORE that
    tick's secondary rows -- see ``_batched_worker_main``/single-process
    ``_collect()``'s ``buffer.add()`` ordering) -- segmenting on trainee
    ``done`` events alone still captures every secondary row exactly once,
    in the right episode, with no separate secondary-side bookkeeping
    needed. Closing a segment is deliberately deferred until the NEXT
    trainee row is seen (not done immediately at the trainee ``done=1``
    row) specifically so that tick's own trailing secondary rows -- which
    arrive right after it, still done=1, before the next episode's first
    trainee row -- are pulled into the segment being closed, not the next
    one.

    CAUTION: if a frozen/older-checkpoint opponent is ever added (this has
    been discussed as a real future possibility), the "any secondary row
    implies current network" invariant this function leans on breaks --
    at that point this needs an explicit per-row "is this the live
    network" signal (not currently threaded through the buffer) before it
    can keep including secondary rows unconditionally. Don't assume this
    still holds without checking when that lands.

    ``dones``/``advantages`` accept plain lists OR anything indexable with
    ``[int] -> float`` (e.g. a ``RolloutBuffer.dones`` list alongside a
    freshly computed ``compute_gae()`` advantages list) -- both are always
    the SAME length as ``track_ids`` (one entry per buffer row).
    """
    means: list[float] = []
    seg: list[int] = []
    pending_close = False
    for i, t in enumerate(track_ids):
        if t == "trainee" and pending_close:
            means.append(sum(abs(advantages[j]) for j in seg) / len(seg))
            seg = []
            pending_close = False
        seg.append(i)
        if t == "trainee" and dones[i] > 0.5:
            pending_close = True
    if pending_close:
        means.append(sum(abs(advantages[j]) for j in seg) / len(seg))
    return means

def value_mse_by_outcome(
    pred: torch.Tensor, target: torch.Tensor, outcomes: list[str],
) -> dict[str, tuple[float, int, float, float]]:
    """Per-outcome (raw MSE, n_rows, target_mean, target_mean_sq) breakdown
    of a value-prediction batch, shared by every value-fitting call site
    (Phase 0 demo warm-up, PPO rollout value pre-training, and the PPO value
    loss itself) so "is the critic doing worse on losses/timeouts than wins"
    can be answered the same way everywhere instead of duplicating the
    group-by logic per caller. ``outcomes`` must be the same length as
    ``pred``/``target`` (e.g. a dataset's ``row_outcomes()`` for demo rows,
    or a rollout batch's ``step_outcomes`` list, with "" -> "unknown").
    Empty/missing outcome strings are grouped under "unknown".

    ``target_mean``/``target_mean_sq`` (mean of target, mean of target**2 --
    NOT variance/std directly) are the ground-truth return's own first two
    raw moments for this group, in the same "per-call mean, multiply by n to
    re-accumulate a sum across chunks" convention ``mse`` already uses --
    callers accumulating this across multiple calls (e.g.
    ``_accum_value_by_outcome`` below) should do the same
    ``value * n``-then-divide-by-total-n dance for these two fields that
    they already do for ``mse``, then derive
    ``std = sqrt(max(target_mean_sq - target_mean**2, 0))`` once at the
    final, fully-accumulated point -- NOT per chunk (mean/mean_sq combine
    linearly across chunks via weighted averaging, std does not).
    """
    if len(outcomes) == 0:
        return {}
    target_np = target.detach().float().cpu().numpy()
    sq_err = (pred.detach() - target).float().cpu().numpy() ** 2
    out: dict[str, tuple[float, int, float, float]] = {}
    labels = np.array([o if o else "unknown" for o in outcomes], dtype=object)
    for name in np.unique(labels):
        mask = labels == name
        n = int(mask.sum())
        out[str(name)] = (
            float(sq_err[mask].sum() / max(n, 1)),
            n,
            float(target_np[mask].mean()),
            float((target_np[mask] ** 2).mean()),
        )
    return out

def format_outcome_rmse_breakdown(by_outcome: dict[str, tuple[float, int, float, float]]) -> str:
    """One-line ``outcome=rmse(gt=mean±std, n)`` summary of
    ``value_mse_by_outcome()``'s output, for appending to an existing log
    line without a full extra multi-line block -- callers that want the
    fuller multi-line form (see debug_value_network.py) can iterate the
    dict themselves instead.

    Takes the sqrt of each outcome's mean squared error before formatting,
    so the displayed RMSE is in the same units as the value/return itself
    (reward-scale) rather than squared units -- easier to eyeball against
    the return std printed elsewhere on the same log line. The ground-truth
    ``gt=mean±std`` alongside it is what the critic is actually being
    scored against for that outcome group -- an RMSE of e.g. 0.66 reads very
    differently against a gt std of 0.45 (worse than just predicting the
    mean) than against a gt std of 2.0 (a solid fit).

    ``by_outcome`` values are ``(mse, n, target_mean, target_mean_sq)`` --
    the LAST two are target_mean_sq, not variance/std; std is derived here
    as ``sqrt(max(target_mean_sq - target_mean**2, 0))``, matching
    ``value_mse_by_outcome()``'s own docstring on why that derivation must
    happen only once, at the final fully-accumulated point.
    """
    if not by_outcome:
        return ""
    return "  ".join(
        f"{name}={math.sqrt(max(mse, 0.0)):.3f}"
        f"(gt={gt_mean:+.2f}±{math.sqrt(max(gt_mean_sq - gt_mean ** 2, 0.0)):.2f}, n={n})"
        for name, (mse, n, gt_mean, gt_mean_sq) in sorted(by_outcome.items())
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_numpy(action: DecisionAction, exec_samples: dict) -> dict[str, np.ndarray]:
    """Flatten a DecisionAction + raw execution samples to a numpy dict for the rollout buffer.

    ``exec_samples`` may also carry the action-opportunity flags (``OPP_ACTION_KEYS``,
    attached by ``ScenarioEnv`` when ``ppo.log_action_opportunity_stats`` /
    ``ppo.mask_untriggered_actions`` is on); they are copied through so they become
    ``action/opp_*`` batch tensors. Absent otherwise (=> the batch has no such keys).
    """
    out = _action_dict(action, exec_samples)
    for _k in OPP_ACTION_KEYS:
        if _k in exec_samples:
            out[_k] = exec_samples[_k]
    return out

def _action_dict(action: DecisionAction, exec_samples: dict) -> dict[str, np.ndarray]:
    return {
        "shoot": np.array([action.shoot], dtype=np.float32),
        "pass_": np.array([action.pass_], dtype=np.float32),
        "move": np.array([action.move], dtype=np.float32),
        "tackle": np.array([action.tackle], dtype=np.float32),
        "get_possession_extra": np.array([action.get_possession_extra], dtype=np.float32),
        "mark": np.array([action.mark], dtype=np.float32),
        "hold_position": np.array([action.hold_position], dtype=np.float32),
        "pass_target": np.array([action.pass_target], dtype=np.float32),
        "tackle_target": np.array([action.tackle_target], dtype=np.float32),
        "mark_target": np.array([action.mark_target], dtype=np.float32),
        "exec_move": exec_samples["exec_move"],
        "sprint": exec_samples["sprint"],
        "kick": exec_samples["kick"],
        "tackle_attempt": exec_samples["tackle_attempt"],
        "move_dir_raw": exec_samples["move_dir_raw"],
        "kick_dir_raw": exec_samples["kick_dir_raw"],
        "kick_power_raw": exec_samples["kick_power_raw"],
        "kick_spin_raw": exec_samples["kick_spin_raw"],
        "move_region_center_raw": action.move_region_center_raw,
        "move_region_size_raw": np.array([action.move_region_size_raw], dtype=np.float32),
        "move_arrival_speed_raw": np.array([action.move_arrival_speed_raw], dtype=np.float32),
    }

def _merge_worker_batches(batches: list[dict], release_inputs: bool = False) -> dict:
    """Concatenate per-worker ``RolloutBuffer.as_tensors()`` dicts along dim 0.

    Each worker's batch must already have GAE applied independently (its
    ``advantages``/``returns`` were computed against its OWN trailing
    bootstrap value) -- concatenating raw transitions across worker
    boundaries BEFORE computing GAE would corrupt advantage estimates by
    treating unrelated episodes/workers as one continuous trajectory.

    ``release_inputs=True`` deletes each key from every input dict as soon as
    that key has been concatenated, so peak memory is ONE full copy of the
    data plus a single key's worth of extra, instead of the inputs AND the
    merged result alive together (~2x) -- the rollout-end RAM spike this
    exists to flatten. The input dicts are left EMPTY afterwards, so only
    pass it when the caller is done with them (the streaming rollout
    consumers are). Default False = inputs untouched, as before.
    """
    merged: dict = {}
    for key in list(batches[0].keys()):
        if key in ("reward_comps_raw", "step_outcomes", "track_ids"):
            merged[key] = [x for b in batches for x in b[key]]
        else:
            merged[key] = torch.cat([b[key] for b in batches], dim=0)
        if release_inputs:
            for b in batches:
                del b[key]
    return merged

def _split_batch_releasing(
    batch: dict, train_mask: np.ndarray, val_mask: Optional[np.ndarray],
) -> tuple[dict, Optional[dict]]:
    """Split a merged rollout batch into ``(train, val)`` row subsets (boolean
    masks over rows; ``val_mask=None`` -> ``val`` is None) while EMPTYING
    ``batch`` key by key as each key is sliced, so the full batch and its two
    slices are never all alive together (peak ~1 full copy plus one key,
    instead of ~2 full copies). ``batch`` is left empty afterwards -- the
    caller must have read anything it still needs from it first (e.g.
    ``batch["returns"]`` for a normalisation std) and must not use it again.
    The list-valued keys (``reward_comps_raw``/``step_outcomes``/
    ``track_ids``) are sliced as Python lists, exactly like the tensors."""
    list_keys = ("reward_comps_raw", "step_outcomes", "track_ids")
    train_idx = torch.from_numpy(np.where(train_mask)[0]).long()
    train_list = train_idx.tolist()
    val_idx = torch.from_numpy(np.where(val_mask)[0]).long() if val_mask is not None else None
    val_list = val_idx.tolist() if val_idx is not None else None
    train: dict = {}
    val: Optional[dict] = {} if val_idx is not None else None
    for key in list(batch.keys()):
        v = batch.pop(key)
        if key in list_keys:
            train[key] = [v[i] for i in train_list]
            if val is not None:
                val[key] = [v[i] for i in val_list]
        else:
            train[key] = v[train_idx]
            if val is not None:
                val[key] = v[val_idx]
        del v
    return train, val


def _lr_warmup_scale(step_idx: int, warmup_steps: int) -> float:
    """Linear LR-warmup multiplier for the ``step_idx``-th (0-based) optimizer
    step of a cycle: ``(step_idx + 1) / warmup_steps`` capped at 1.0, so the
    first step runs at ``1/warmup_steps`` of the base lr and step
    ``warmup_steps - 1`` onward at the full lr. ``warmup_steps <= 0`` = off."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step_idx + 1) / warmup_steps)

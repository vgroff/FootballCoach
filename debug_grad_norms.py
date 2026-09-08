"""Explain where a real PPO update's gradient norm actually comes from,
broken down by NETWORK LAYER (trunk/entity_encoder/each individual head)
and by STEP within one real episode.

Motivated by a live training-log observation: `[grad clip] main`'s pre-clip
norm (mean=5.5, max=184.6) looked surprisingly large next to
`[exec head grad norm]`'s move_direction=2.275, raising the question of
whether move_direction was somehow leaking into the "main" clip group
instead of being properly isolated into "dir" (see ppo_trainer.py's
`direction_param_ids`/`_non_direction_params`/`_direction_params`).

A direct parameter-count check already answers the "leak" question: with
this repo's current ai_config.json network sizes, `direction_param_ids`
(move_direction + kick_direction + their two log_std scalars) covers only
247 of decision_net+execution_net's 273,139 total parameters (0.09%) --
"main" is EVERYTHING else, dominated by the SHARED entity_encoder (43,264
params, literally the same nn.Module instance feeding both networks when
network.share_entity_encoder=true) plus decision_net's own trunk (59,592)
and execution_net's own smaller trunk (16,224). A combined-group L2 norm
over ~273K parameters is essentially guaranteed to dwarf a single tiny
head's own ~100-parameter norm regardless of per-parameter gradient
scale -- there is no leak, just very different parameter-set sizes being
compared. This script goes one step further: it doesn't just report that
static parameter-count fact, it measures REAL per-layer/per-step gradient
norms from a real trained checkpoint on a real episode, so "which layer is
actually driving `main`'s norm on any given step" is empirical, not just
inferred from parameter counts.

How it works
------------
1. Loads a trained checkpoint via PPOTrainer.load_checkpoint() (same format
   train.py/--checkpoint/--from-pretrained read).
2. Rolls the loaded policy out, closed-loop, for exactly ONE real episode
   (reusing debug_policy_net.py's `_build_replay_env` -- same opponent-roll
   reproduction logic as replay_episode.py), collecting genuine
   action/log_prob/value/reward/bc_label transitions exactly like
   PPOTrainer.train()'s own rollout loop does (not the DAgger-style dummy
   buffer debug_policy_net.py's own _collect_dagger_rollout uses -- this
   needs real fields for GAE/PPO, not just BC labels).
3. Computes GAE advantages/returns over that one episode and packs it into
   the same batch shape `_ppo_update()` expects, then calls the REAL,
   unmodified `PPOTrainer._ppo_update()` with `minibatch_size=1` and
   `n_epochs=1` -- one gradient step per episode step, in the exact same
   loss composition (policy + value + entropy + BC-aux) real training uses.
   Nothing about `_ppo_update()` itself is changed; this script only
   configures the trainer's minibatch size/epoch count before calling it.
4. To read out the per-layer breakdown without touching `_ppo_update()`'s
   source, this script temporarily monkeypatches `torch.nn.utils.
   clip_grad_norm_` and `torch.randperm` (restored in a `finally` block)
   just for the duration of that one call:
     - `torch.randperm` is intercepted so this script knows which shuffled
       minibatch index corresponds to which real (chronological) episode
       step -- `_ppo_update()` draws `indices = torch.randperm(n)` exactly
       once (n_epochs=1) and slices minibatches out of it in order.
     - `clip_grad_norm_` is intercepted only for the specific call whose
       parameter list exactly equals decision_net.parameters() +
       execution_net.parameters() combined (the pre-clip "raw" measurement
       call `_ppo_update()` already makes, at `max_norm=inf`, BEFORE the
       real main/dir clipping calls run) -- every other clip_grad_norm_
       call (the small per-head EXEC_HEAD_MODULES diagnostics, the real
       main/dir clipping calls themselves) is passed straight through
       unmodified, so production's actual clipping behaviour is completely
       unaffected by this script; only an extra READ of the (still
       unclipped, at that point) .grad tensors happens, in addition to
       whatever `_ppo_update()` was already going to do with them.
   At that interception point, this script groups the touched parameters
   by TOP-LEVEL SUBMODULE (trunk, entity_encoder, self_mlp, each individual
   head, ...) using a precomputed `id(param) -> label` map, and computes an
   L2 norm per group. Because the groups are a strict partition of the
   full parameter set, `sum(group_norm**2 for every group) ==
   raw_norm**2` exactly (Pythagorean/Euclidean norm decomposition) -- so
   recombining just the two "dir" groups (move_direction, kick_direction)
   reproduces the exact same "dir" number `_ppo_update()`'s own separate
   clip call would report, and recombining everything else reproduces
   "main", without needing to intercept those two calls separately at all.

Caveat (stated plainly, not hidden): minibatch_size=1 means EVERY step gets
its own real optimizer.step() within this single `_ppo_update()` call, so
by the last few steps of a long episode the weights have drifted slightly
from the loaded checkpoint (same as any real minibatch loop -- this is not
an artifact unique to this script). Advantage normalization is unaffected
(it's computed once over the whole episode's advantages before any
minibatch runs, not per-minibatch), so that particular failure mode (a
single-row batch normalizing its own advantage to a degenerate 0) does not
apply here.

Usage:
    uv run python debug_grad_norms.py --checkpoint checkpoints/latest.pt --seed 12345
    uv run python debug_grad_norms.py --checkpoint checkpoints/latest.pt --top-n 15 --json-out results/grad_norms.json
"""
from __future__ import annotations

# Must run BEFORE numpy (or torch, which imports it) is first imported
# ANYWHERE in this process -- see debug_value_network.py's identical block
# for the full rationale (OpenBLAS thread-pool sizing at first import).
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import contextlib
import json
import logging
import math
import random as _random
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("footballcoach.ai.debug_grad_norms")


def _top_level_name(param_name: str) -> str:
    """First real submodule component of a named_parameters() key.

    trainer.decision_net/execution_net are CanonicalNetworkWrapper instances
    (see obs/canonical.py) -- every parameter name is actually prefixed
    "_wrapped.<real name>" (the wrapper registers the real DecisionNetwork/
    ExecutionNetwork as its own "_wrapped" submodule so .parameters() etc.
    recurse into it transparently). Strip that prefix before taking the
    first component, otherwise every single parameter in the whole network
    collapses into one "_wrapped" bucket.
    """
    parts = param_name.split(".")
    if parts[0] == "_wrapped":
        parts = parts[1:]
    return parts[0]


def _build_param_layer_labels(trainer) -> dict[int, str]:
    """id(param) -> "<group>.<top-level submodule name>" for every parameter
    in decision_net + execution_net.

    entity_encoder/ball_mlp/global_mlp are shared (literally the same
    nn.Module instance feeding both networks) whenever
    network.share_entity_encoder=true in ai_config.json -- a shared
    module's parameters are labelled "SHARED.<name>" regardless of which
    network's .parameters() iteration order happens to touch them first,
    so the breakdown doesn't arbitrarily attribute shared gradient to only
    one of the two networks.
    """
    labels: dict[int, str] = {}
    for name, p in trainer.decision_net.named_parameters():
        top = _top_level_name(name)
        labels[id(p)] = f"decision_net.{top}"
    for name, p in trainer.execution_net.named_parameters():
        top = _top_level_name(name)
        pid = id(p)
        if pid in labels:
            # Already claimed by decision_net -- this is one of the shared
            # modules (entity_encoder/ball_mlp/global_mlp), re-label as
            # shared rather than leaving it misattributed to decision_net
            # alone.
            labels[pid] = f"SHARED.{top}"
        else:
            labels[pid] = f"execution_net.{top}"
    return labels


def _collect_one_episode(trainer, seed: int, max_steps: int, deterministic: bool):
    """Roll trainer's current policy out closed-loop for exactly one real
    episode, collecting genuine (obs, action, log_prob, value, reward, done,
    bc_label) transitions -- the same fields PPOTrainer.train()'s own
    rollout loop stores (see ppo_trainer.py train(), lines ~1034-1074),
    reused here rather than reimplemented. Secondary-player transitions
    (shared-weight training data from a neural secondary opponent) are
    deliberately NOT collected -- this script is about the trainee's own
    gradient contribution on one episode, mixing in a second track would
    conflate two different players' states into one "episode" breakdown.

    Returns (buffer, summary) where summary is {"outcome", "n_steps",
    "total_reward", "seed"}.
    """
    from debug_policy_net import _build_replay_env
    from footballcoach.ai.ppo.bc import phase1_labels
    from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
    from footballcoach.ai.ppo.ppo_trainer import _action_to_numpy

    env = _build_replay_env(trainer, seed, deterministic=deterministic)
    buf = RolloutBuffer()
    total_reward = 0.0
    outcome = None
    n_steps = 0
    for n_steps in range(1, max_steps + 1):
        next_obs, reward, done, info = env.step()
        tr = env.last_trainee_transition
        if tr is None:
            if done:
                break
            continue
        label = phase1_labels(env, player_id="trainee")
        buf.add(
            obs=tr["obs"],
            action=_action_to_numpy(tr["action"], tr["raw_exec"]),
            log_prob=float(tr["log_prob"]),
            value=float(tr["value"]),
            reward=reward,
            done=1.0 if done else 0.0,
            bc_label=label.to_array(),
            head_log_probs=tr.get("head_log_probs"),
            reward_comps=dict(getattr(env, "last_reward_components", {})),
            step_outcome=(info.trial_outcome or "") if (done and info is not None) else "",
        )
        total_reward += reward
        if done:
            outcome = info.trial_outcome
            break
    return buf, {"outcome": outcome, "n_steps": n_steps, "total_reward": total_reward, "seed": seed}


def _angle_deg(vec2: torch.Tensor) -> float:
    """atan2 angle in degrees of a single (2,)-shaped raw direction vector
    (not necessarily unit-length -- callers pass both the raw sampled action
    and the network's own L2-normalized mean, and only the ANGLE, not the
    magnitude, is meaningful for either)."""
    v = vec2.detach().flatten()
    return math.degrees(math.atan2(float(v[1]), float(v[0])))


def _angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """Smallest signed-magnitude difference between two angles in degrees,
    wrapped into [0, 180] -- plain subtraction would report e.g. 350 deg for
    two directions actually only 10 deg apart across the wraparound."""
    d = abs(a_deg - b_deg) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _register_move_dir_input_hook(trainer) -> tuple["torch.utils.hooks.RemovableHandle", dict]:
    """Forward hook on execution_net's real move_direction Linear layer,
    capturing the L2 norm of ITS INPUT (the execution trunk's final hidden
    activation `h`) on every forward call.

    Why this matters for "why is the weight-gradient norm so big": for a
    Linear layer `y = W @ h + b`, the gradient of any scalar loss w.r.t. W
    is the OUTER PRODUCT `dL/dW = (dL/dy) (h)^T`, whose Frobenius norm is
    EXACTLY `||dL/dy|| * ||h||` (norms of an outer product multiply
    exactly, not just approximately). So the weight-gradient norm this
    script reports for "execution_net.move_direction" isn't only a function
    of "how wrong the prediction was" (||dL/dy||, e.g. the (sample-mean)/
    sigma^2 term from the Gaussian log_prob) -- it's ALSO scaled by how
    large the trunk's own output activations happen to be feeding into that
    layer, which this hook measures directly instead of assuming.
    """
    state = {"input_norm": None}

    def _hook(module, inputs, output):
        state["input_norm"] = float(inputs[0].detach().norm().item())

    handle = trainer.execution_net._wrapped.move_direction.register_forward_hook(_hook)
    return handle, state


def _move_dir_why(trainer, frame_locals: dict, batch: dict, hook_state: dict) -> dict:
    """Everything needed to explain THIS step's move_direction gradient:
    how far the current policy's predicted mean direction has drifted from
    the direction that was actually sampled/stored at rollout-collection
    time (the bigger this drift, the bigger d(log_prob)/d(mean) gets for a
    Normal distribution -- see DirectionHead/execution_network.py's own
    "KL spikes if too tight" note: sensitivity scales as 1/std^2), the
    current log_std (a tighter distribution amplifies the same angular
    drift into a much bigger log_prob swing), the resulting move_dir-
    specific ratio (isolates JUST this head's contribution, unlike the
    aggregate PPO ratio which mixes in every head), this step's advantage
    (a large |advantage| scales every head's policy-gradient term together,
    independent of move_dir specifically), and whether exec_move was even
    "on" this tick (move_dir's PPO log_prob -- see _per_head_new_log_probs's
    exec_move_mask -- is masked to 0 when it wasn't, though the BC-aux
    direction loss is NOT gated by exec_move, so it can still contribute
    even then).

    Reads _ppo_update()'s own local variables via frame introspection
    (mb_idx/d_heads/e_heads/mb_actions/mb_adv) rather than recomputing any
    of this independently -- avoids any risk of this diagnostic's own
    numbers subtly disagreeing with what production actually computed.
    """
    from footballcoach.ai.ppo.rollout_buffer import HEAD_LP_KEYS

    mb_idx = frame_locals["mb_idx"]
    d_heads = frame_locals["d_heads"]
    e_heads = frame_locals["e_heads"]
    mb_actions = frame_locals["mb_actions"]
    mb_adv = frame_locals["mb_adv"]
    # The AGGREGATE ratio (exp(new_log_probs - old_log_probs), summed over
    # all 15 heads' log_probs) -- NOT move_dir's own isolated ratio -- is
    # what actually scales the policy-gradient term flowing into EVERY
    # head's own log_prob, including move_dir's (new_log_probs is a SUM
    # over heads, so d(policy_loss)/d(lp_movedir) = d(policy_loss)/
    # d(new_log_probs), the same aggregate-ratio-scaled quantity every
    # other head gets, regardless of what move_dir's own individual ratio
    # is). Using move_dir_ratio here instead was a real bug in an earlier
    # version of this diagnostic -- confirmed by comparing the
    # reconstructed vs. observed weight-grad norm across several steps: it
    # matched well when the two ratios happened to be similar and was off
    # by 10-20x on a step where they diverged.
    aggregate_ratio = float(frame_locals["ratio"].item()) if "ratio" in frame_locals else None
    clip_range = frame_locals.get("clip")

    with torch.no_grad():
        stored_raw = mb_actions["move_dir_raw"]  # (1, 2) -- action sampled at collection time
        current_mean = e_heads.move_direction  # (1, 2) -- unit vector, CURRENT policy's mean
        log_std_move = trainer.execution_net.move_dir_log_kappa.to(trainer.device)

        exec_move_on = bool(mb_actions["exec_move"].item() > 0.5)
        # _per_head_new_log_probs()/_recompute_log_prob() gate move_dir's
        # PPO log_prob by exec_move_mask -- when exec_move is off this tick,
        # production's real move_dir log_prob (and hence its policy-
        # gradient contribution) is exactly 0, which is exactly why dir_norm
        # reads 0.0 on such steps too (confirmed live). Mirror that masking
        # here instead of reporting a nonzero log_prob/ratio for a
        # contribution that was never actually part of the loss.
        move_dir_new_lp = float(
            trainer._move_dir_head(current_mean, log_std_move).log_prob(stored_raw).item()
        ) if exec_move_on else 0.0
        move_dir_old_lp = None
        if "head_log_probs" in batch:
            col = HEAD_LP_KEYS.index("move_dir")
            move_dir_old_lp = float(batch["head_log_probs"][mb_idx, col].item())
        angle_stored = _angle_deg(stored_raw)
        angle_current_mean = _angle_deg(current_mean)

        # Analytical d(log_prob)/d(mean) magnitude for an isotropic Normal:
        # ||sample - mean|| / sigma^2 (verified directly against autograd --
        # see the standalone sanity check run alongside this script). This
        # is the SAME quantity that drives KL between old/new policy for a
        # fixed-variance Gaussian (KL ~ (delta_mean)^2 / sigma^2) -- sigma
        # amplifies both the KL AND this gradient term identically, it is
        # not "either/or".
        # STALE as of the von Mises migration: move_dir is no longer an
        # isotropic Gaussian on the unit vector (see VonMisesDirectionHead in
        # ai/action/distributions.py) -- this formula (and std_move/
        # log_std_move below) is kept renamed-but-otherwise-unmodified so the
        # script still runs, but the reconstructed gradient estimate below is
        # no longer analytically correct for the current distribution. A real
        # fix would use the von Mises log_prob's actual derivative w.r.t.
        # theta_mean instead.
        std_move = float(torch.exp(log_std_move.clamp(
            trainer.move_dir_log_kappa_min, trainer.move_dir_log_kappa_max
        )).mean().item())
        diff_norm = float((stored_raw - current_mean).norm().item())
        d_logprob_d_mean_norm = (diff_norm / (std_move ** 2)) if exec_move_on else 0.0

        trunk_activation_norm = hook_state.get("input_norm")
        # d(mean)/d(raw_vector)'s Jacobian, for mean=raw/(||raw||+eps), has
        # operator norm ~1/||raw|| -- the estimate below implicitly treats
        # this as ~1 (assumes ||raw_vector||~1, i.e. the pre-normalize
        # output is already close to unit length). Reported so a bad match
        # can be checked against this specific assumption instead of staying
        # a mystery: a small ||raw_vector|| here amplifies the REAL gradient
        # well beyond what the estimate (which omits this factor) predicts.
        raw_move_norm = float(e_heads.move_direction_unnormalized.norm().item())

        result = {
            "advantage": float(mb_adv.item()),
            "exec_move_on": exec_move_on,
            "move_dir_log_std": float(log_std_move.mean().item()),  # actually log_kappa now, key name kept for output-format stability
            "move_dir_std": std_move,
            "move_dir_new_logprob": move_dir_new_lp,
            "stored_action_angle_deg": angle_stored,
            "current_mean_angle_deg": angle_current_mean,
            # How far the CURRENT policy's predicted mean has drifted from
            # the direction actually sampled/stored at collection time --
            # the primary driver of a large d(log_prob)/d(mean) under a
            # tight Gaussian (see docstring).
            "mean_vs_stored_angle_diff_deg": _angle_diff_deg(angle_stored, angle_current_mean),
            "d_logprob_d_mean_norm": d_logprob_d_mean_norm,
            "trunk_activation_norm": trunk_activation_norm,
            "raw_move_norm": raw_move_norm,
        }
        if move_dir_old_lp is not None:
            result["move_dir_old_logprob"] = move_dir_old_lp
            result["move_dir_logprob_shift"] = move_dir_new_lp - move_dir_old_lp
            result["move_dir_ratio"] = math.exp(
                max(min(move_dir_new_lp - move_dir_old_lp, 40.0), -40.0)
            )
            # Back-of-envelope reconstruction of the OBSERVED weight-grad
            # norm, purely to sanity-check the mechanism end to end:
            # ||dL/dW||_F == ||dL/d(mean)|| * ||trunk activation h||
            # (outer-product norm identity, exact) and ||dL/d(mean)|| ==
            # |advantage * ratio_AGGREGATE| * d_logprob_d_mean_norm (chain
            # rule through the PPO surrogate objective's un-clipped branch:
            # new_log_probs sums all 15 heads' log_probs, so d(policy_loss)/
            # d(lp_movedir) is scaled by the SAME aggregate ratio every
            # other head's log_prob gets -- move_dir's own isolated ratio
            # is NOT the right multiplier here, only useful as its own
            # diagnostic). Still approximate: ignores the clip(ratio, 1-eps,
            # 1+eps) kink (flagged separately below via clip_active) and the
            # per-sample importance weight (a no-op for this collection's
            # default weight=1.0). Reported ALONGSIDE the real observed norm
            # (this record's "layers" dict) for direct comparison, never
            # used to override it.
            if trunk_activation_norm is not None and exec_move_on and aggregate_ratio is not None:
                result["aggregate_ratio"] = aggregate_ratio
                if clip_range is not None:
                    result["clip_active"] = not (1.0 - clip_range <= aggregate_ratio <= 1.0 + clip_range)
                result["estimated_weight_grad_norm"] = (
                    abs(result["advantage"] * aggregate_ratio) * d_logprob_d_mean_norm * trunk_activation_norm
                )
        _bkdn = frame_locals.get("_bkdn")
        if isinstance(_bkdn, dict) and "direction" in _bkdn:
            result["bc_direction_loss"] = float(_bkdn["direction"])
        return result


@contextlib.contextmanager
def _grad_norm_capture(
    trainer, layer_labels: dict[int, str], dir_top_level: set[str], batch: dict, hook_state: dict,
):
    """Context manager: monkeypatches torch.randperm + clip_grad_norm_ for
    its duration (both restored on exit, including on exception) to record
    a per-step, per-layer gradient-norm breakdown from the NEXT
    `trainer._ppo_update(batch, progress)` call made inside the `with`
    block. `batch` must be the EXACT SAME dict object passed to that
    `_ppo_update()` call (augmentation disabled by the caller, so
    `_ppo_update()` never reassigns it internally) -- used to look up each
    step's OLD (rollout-collection-time) per-head log_probs for the
    move_direction "why" diagnostics below. Yields a list that gets filled
    with one dict per step:
    {"mb_order_i": int, "step_idx": int, "raw_norm": float,
     "main_norm": float, "dir_norm": float, "layers": {label: norm},
     "why_move_dir": {...}}.

    Only intercepts the ONE clip_grad_norm_ call in _ppo_update() whose
    parameter list is EXACTLY decision_net.parameters() +
    execution_net.parameters() combined (the pre-clip "raw" measurement,
    max_norm=inf, called before the real main/dir clipping) -- every other
    call (per-head EXEC_HEAD_MODULES diagnostics, the real main/dir clip
    calls) is passed straight through untouched, so production's actual
    clipping/training behaviour is completely unaffected. See module
    docstring for the full rationale, including why summing per-layer
    squared norms exactly reproduces "main"/"dir" without intercepting
    those calls separately.
    """
    all_ids = {id(p) for p in list(trainer.decision_net.parameters()) + list(trainer.execution_net.parameters())}

    records: list[dict] = []
    state = {"indices": None, "call_i": 0}

    _orig_randperm = torch.randperm
    _orig_clip = torch.nn.utils.clip_grad_norm_

    def _patched_randperm(n, *a, **kw):
        result = _orig_randperm(n, *a, **kw)
        # Always take the LATEST permutation, and reset the per-permutation
        # call counter -- _ppo_update() draws a fresh torch.randperm(n) at
        # the top of every epoch (and its own separate value-only-
        # continuation loop, if that ever triggers), so call_i must be
        # relative to whichever permutation is currently in effect, not a
        # running total across all of them.
        state["indices"] = result.tolist()
        state["call_i"] = 0
        return result

    def _patched_clip_grad_norm_(parameters, max_norm, *a, **kw):
        params_list = list(parameters)
        ids = [id(p) for p in params_list]
        if set(ids) == all_ids:
            group_sumsq: dict[str, float] = {}
            for p, pid in zip(params_list, ids):
                if p.grad is None:
                    continue
                sumsq = float(p.grad.detach().pow(2).sum().item())
                label = layer_labels.get(pid, "UNKNOWN")
                group_sumsq[label] = group_sumsq.get(label, 0.0) + sumsq
            dir_sumsq = sum(v for k, v in group_sumsq.items() if k.split(".")[-1] in dir_top_level)
            main_sumsq = sum(v for k, v in group_sumsq.items() if k.split(".")[-1] not in dir_top_level)
            layer_norms = {k: v ** 0.5 for k, v in group_sumsq.items()}
            call_i = state["call_i"]
            state["call_i"] += 1
            if state["indices"] is not None and call_i < len(state["indices"]):
                step_idx = state["indices"][call_i]
            else:
                # Defensive fallback (should not normally happen): some
                # other torch.randperm call desynced the counter. Fall back
                # to the raw call order rather than crashing a diagnostic
                # script -- the aggregate/spikiest-step stats are still
                # valid, only step_idx's mapping to a real episode tick is
                # suspect for this record.
                step_idx = call_i
            try:
                why_move_dir = _move_dir_why(trainer, sys._getframe(1).f_locals, batch, hook_state)
            except Exception as exc:  # noqa: BLE001 -- best-effort diagnostic, never fatal
                why_move_dir = {"error": str(exc)}
            records.append({
                "mb_order_i": call_i,
                "step_idx": int(step_idx),
                "raw_norm": (main_sumsq + dir_sumsq) ** 0.5,
                "main_norm": main_sumsq ** 0.5,
                "dir_norm": dir_sumsq ** 0.5,
                "layers": layer_norms,
                "why_move_dir": why_move_dir,
            })
        return _orig_clip(parameters, max_norm, *a, **kw)

    torch.randperm = _patched_randperm
    torch.nn.utils.clip_grad_norm_ = _patched_clip_grad_norm_
    try:
        yield records
    finally:
        torch.randperm = _orig_randperm
        torch.nn.utils.clip_grad_norm_ = _orig_clip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Trained checkpoint to load (PPOTrainer.load_checkpoint() format).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Episode seed to replay (opponent roll reproduced exactly, see "
                             "debug_policy_net.py's _build_replay_env). Default: random.")
    parser.add_argument("--max-episode-steps", type=int, default=500,
                        help="Safety cap on decision steps for the collected episode. Default: 500.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Roll out with deterministic (mode/mean) actions instead of PPO's "
                             "normal stochastic sampling. Default: off (stochastic, matching real "
                             "rollout collection).")
    parser.add_argument("--separate-value-net", action="store_true",
                        help="Set if the checkpoint was trained with --separate-value-net.")
    parser.add_argument("--progress", type=float, default=0.0,
                        help="Schedule progress fraction in [0,1] fed to _ppo_update() (controls "
                             "clip/lr/bc/entropy coefficient schedules) -- 0.0 = start-of-training "
                             "values. Default: 0.0.")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Print full per-layer breakdown for the N steps with the largest "
                             "main_norm. Default: 10.")
    parser.add_argument("--device", type=str, default=None,
                        help="PyTorch device. Default: auto-detect (cuda if available, else cpu).")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Optional path to dump the full per-step/per-layer breakdown as JSON "
                             "for further analysis/plotting.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    log.info(f"Device: {device}")

    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

    trainer = PPOTrainer.from_config(
        device=device, inference_only=True, separate_value_net=args.separate_value_net,
    )
    trainer.optimizer = torch.optim.Adam(
        list(trainer.decision_net.parameters()) + list(trainer.execution_net.parameters()),
        lr=1e-4, eps=1e-5,
    )
    if args.separate_value_net:
        # inference_only=True skips __init__'s "not inference_only" branch
        # entirely, so self.value_net_optimizer is never built -- but
        # _ppo_update() unconditionally calls self.value_net_optimizer.
        # zero_grad()/.step() whenever self.separate_value_net is True
        # (confirmed live: AttributeError on 'NoneType' without this).
        trainer.value_net_optimizer = torch.optim.Adam(
            trainer.value_net.parameters(), lr=1e-4, eps=1e-5,
        )
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"--checkpoint {ckpt_path} does not exist")
    # load_checkpoint() gates the "value_net" key behind `if self.value_net
    # is not None:` and silently skips it otherwise -- no warning. Loading a
    # --separate-value-net checkpoint without also passing --separate-value-
    # net here leaves execution_net.value_head at whatever it was during
    # real training (untrained dead weight, since that head's own gradient
    # path is disconnected from the loss whenever separate_value_net=True) —
    # then THIS script would route new_values through it anyway,
    # manufacturing a large, fake value-loss gradient that has nothing to do
    # with the checkpoint's real training dynamics. Confirmed live: this is
    # exactly what produced a misleading "value_head is a steady top
    # contributor" result on a --separate-value-net checkpoint run without
    # this flag. Check both directions explicitly rather than trusting the
    # caller to remember the flag.
    _ckpt_peek = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _ckpt_has_value_net = "value_net" in _ckpt_peek
    if _ckpt_has_value_net and not args.separate_value_net:
        raise SystemExit(
            f"{ckpt_path} contains a 'value_net' key (trained with --separate-value-net) "
            f"but --separate-value-net was not passed to this script. Re-run with "
            f"--separate-value-net, otherwise execution_net's own (untrained, dead) value "
            f"head gets wired into the loss instead of the real critic, producing bogus "
            f"gradient-norm numbers for it."
        )
    if not _ckpt_has_value_net and args.separate_value_net:
        raise SystemExit(
            f"--separate-value-net was passed but {ckpt_path} has no 'value_net' key "
            f"(it wasn't trained with a separate value net). Re-run without --separate-value-net."
        )
    del _ckpt_peek
    trainer.load_checkpoint(ckpt_path)

    # PPOTrainer.__init__(inference_only=True) hard-codes self.direction_param_ids
    # = set() (no optimizer/param-group setup needed for pure inference) --
    # confirmed by reading __init__ directly. _ppo_update()'s own main/dir
    # split reads self.direction_param_ids live, so left as-is this would
    # make _non_direction_params ("main") silently include EVERY parameter
    # (nothing to exclude) and _direction_params ("dir") empty -- not
    # reflective of real training, where inference_only=False populates this
    # correctly. Reconstruct it here exactly the way __init__ does for a
    # real (non-inference-only) trainer, from the UNWRAPPED execution_net
    # (trainer.execution_net._wrapped) -- named_parameters() on the
    # CanonicalNetworkWrapper itself would see "_wrapped.move_direction..."
    # names instead of "move_direction...", the same prefix issue
    # _top_level_name() above works around.
    _move_dir_ids = {
        id(p) for name, p in trainer.execution_net._wrapped.named_parameters()
        if name.startswith("move_direction.") or name == "move_dir_log_kappa"
    }
    _kick_dir_ids = {
        id(p) for name, p in trainer.execution_net._wrapped.named_parameters()
        if name.startswith("kick_direction.") or name in ("kick_dir_log_kappa", "kick_dir_z_log_std")
    }
    trainer.direction_param_ids = _move_dir_ids | _kick_dir_ids
    log.info(f"Reconstructed direction_param_ids: {len(trainer.direction_param_ids)} params "
             f"(move_direction={len(_move_dir_ids)}, kick_direction={len(_kick_dir_ids)})")

    layer_labels = _build_param_layer_labels(trainer)
    dir_top_level = {"move_direction", "kick_direction", "move_dir_log_kappa", "kick_dir_log_kappa", "kick_dir_z_log_std"}

    seed = args.seed if args.seed is not None else _random.randint(0, 2**31 - 1)
    log.info(f"Collecting one episode (seed={seed}, deterministic={args.deterministic})...")
    buf, summary = _collect_one_episode(trainer, seed, args.max_episode_steps, args.deterministic)
    n = len(buf)
    if n < 2:
        raise SystemExit(
            f"Collected episode has only {n} trainee decision step(s) -- need at least 2 for a "
            f"meaningful per-step breakdown (and for advantage normalization to be non-degenerate). "
            f"Try a different --seed."
        )
    log.info(
        f"Episode: seed={seed}  outcome={summary['outcome']}  steps={n}  "
        f"total_reward={summary['total_reward']:.3f}"
    )

    advantages, returns = buf.compute_gae(trainer.gamma, trainer.lam, 0.0)
    batch = buf.as_tensors(advantages, returns)

    trainer.minibatch_size = 1
    trainer.n_epochs = 1
    # Disable _ppo_update()'s per-minibatch KL early-stop for this run.
    # minibatch_size=1 makes each step's ratio/KL a single-sample, unaveraged
    # quantity -- much noisier than real training's real minibatches -- so
    # it can trip the early stop after only a handful of steps. When it
    # does, _ppo_update() runs a separate "value-only continuation" loop
    # over the same batch afterward (its own torch.randperm() call, its own
    # minibatch loop) whose gradient-norm measurement calls this script's
    # monkeypatch cannot distinguish from the main loop's -- confirmed via
    # a real run: 51 real episode steps produced 58 captured records with
    # step_idx values running past 50, once the main loop's early stop at
    # mb28 triggered a second, separately-indexed pass. Setting target_kl
    # very high keeps this a single, clean, exactly-one-record-per-step
    # pass -- this only affects THIS diagnostic call (a throwaway trainer
    # instance whose weights are never saved), never real training.
    trainer.target_kl = 1e9
    # Disable geometric-flip/slot-permutation batch augmentation for this
    # run too -- it expands the batch (confirmed live: 43 real episode
    # steps became 86 augmented rows), which breaks the one-record-per-
    # real-step assumption this script's step_idx recovery depends on, and
    # its internal row-shuffling uses its own torch.randperm-adjacent calls
    # that collided with this script's monkeypatch (observed IndexError).
    # Same "throwaway diagnostic trainer" reasoning as target_kl above.
    trainer.augment_n_slot_shuffles = 0
    trainer.decision_net.train()
    trainer.execution_net.train()

    hook_handle, hook_state = _register_move_dir_input_hook(trainer)
    try:
        with _grad_norm_capture(trainer, layer_labels, dir_top_level, batch, hook_state) as records:
            trainer._ppo_update(batch, args.progress)
    finally:
        hook_handle.remove()

    if len(records) != n:
        log.warning(
            f"Expected {n} per-step grad-norm records, got {len(records)} -- "
            f"_ppo_update()'s internal structure may have changed since this script was written "
            f"(e.g. the 'raw' pre-clip measurement call was removed/renamed). Reporting whatever "
            f"was captured."
        )
    records.sort(key=lambda r: r["step_idx"])

    # --- Static structural context: parameter counts per layer, independent
    # of this episode's data -- explains WHY a combined "main" norm is
    # structurally larger than any single head's own norm regardless of
    # per-parameter gradient scale (see module docstring). ---
    from collections import defaultdict
    _counts_by_label: dict[str, int] = defaultdict(int)
    for name, p in trainer.decision_net.named_parameters():
        lbl = layer_labels[id(p)]
        _counts_by_label[lbl] += p.numel()
    for name, p in trainer.execution_net.named_parameters():
        lbl = layer_labels[id(p)]
        _counts_by_label[lbl] += p.numel()
    total_params = sum(_counts_by_label.values())
    dir_params = sum(v for k, v in _counts_by_label.items() if k.split(".")[-1] in dir_top_level)
    log.info("")
    log.info(f"--- Structural context: {total_params:,} total params in decision_net+execution_net ---")
    log.info(
        f"  'dir' group (move_direction+kick_direction+log_stds): {dir_params:,} params "
        f"({100 * dir_params / total_params:.2f}% of total)"
    )
    log.info(f"  'main' group (everything else): {total_params - dir_params:,} params "
             f"({100 * (total_params - dir_params) / total_params:.2f}% of total)")
    log.info("  Top layers by parameter count:")
    for lbl, cnt in sorted(_counts_by_label.items(), key=lambda kv: -kv[1])[:8]:
        log.info(f"    {lbl:<32} {cnt:>8,}  ({100 * cnt / total_params:.1f}%)")

    # --- Per-step summary table ---
    log.info("")
    log.info(f"--- Per-step main/dir grad norm ({n} steps, chronological order) ---")
    log.info(f"  {'step':>4}  {'main_norm':>10}  {'dir_norm':>9}  {'top layer (this step)':<28}  {'top layer norm':>14}")
    for r in records:
        top_layer, top_norm = max(r["layers"].items(), key=lambda kv: kv[1])
        log.info(
            f"  {r['step_idx']:>4}  {r['main_norm']:>10.4f}  {r['dir_norm']:>9.4f}  "
            f"{top_layer:<28}  {top_norm:>14.4f}"
        )

    # --- Aggregate per-layer stats across the whole episode ---
    all_labels = sorted({lbl for r in records for lbl in r["layers"]})
    log.info("")
    log.info(f"--- Aggregate per-layer grad norm across all {len(records)} steps ---")
    log.info(f"  {'layer':<32} {'mean':>10} {'std':>9} {'max':>10} {'max@step':>9}")
    agg_rows = []
    for lbl in all_labels:
        vals = np.array([r["layers"].get(lbl, 0.0) for r in records])
        max_i = int(vals.argmax())
        agg_rows.append((lbl, vals.mean(), vals.std(), vals.max(), records[max_i]["step_idx"]))
    agg_rows.sort(key=lambda row: -row[1])
    for lbl, mean, std, mx, max_step in agg_rows:
        log.info(f"  {lbl:<32} {mean:>10.4f} {std:>9.4f} {mx:>10.4f} {max_step:>9}")

    # --- Spikiest steps, full breakdown ---
    spikiest = sorted(records, key=lambda r: -r["main_norm"])[: args.top_n]
    log.info("")
    log.info(f"--- Top {len(spikiest)} spikiest steps (by main_norm), full per-layer breakdown ---")
    for r in spikiest:
        sorted_layers = sorted(r["layers"].items(), key=lambda kv: -kv[1])
        top3 = ", ".join(f"{lbl}={v:.3f}" for lbl, v in sorted_layers[:3])
        log.info(
            f"  step={r['step_idx']:>4}  main={r['main_norm']:.4f}  dir={r['dir_norm']:.4f}  "
            f"top3=[{top3}]"
        )
        w = r.get("why_move_dir", {})
        if "error" in w:
            log.info(f"           why(move_dir): <unavailable: {w['error']}>")
        elif w:
            _shift = w.get("move_dir_logprob_shift")
            _ratio = w.get("move_dir_ratio")
            log.info(
                f"           why(move_dir): adv={w['advantage']:+.3f}  exec_move_on={w['exec_move_on']}  "
                f"log_kappa={w['move_dir_log_std']:.3f} (kappa≈{math.exp(w['move_dir_log_std']):.3f})"
            )
            log.info(
                f"           stored_action_angle={w['stored_action_angle_deg']:+.1f}°  "
                f"current_mean_angle={w['current_mean_angle_deg']:+.1f}°  "
                f"(mean has drifted {w['mean_vs_stored_angle_diff_deg']:.1f}° from what was sampled)"
            )
            if _shift is not None:
                if not w["exec_move_on"]:
                    log.info(
                        "           move_dir logprob: masked (exec_move off this tick) -- "
                        "PPO's move_dir policy-gradient term is exactly 0 here (dir_norm above "
                        "confirms it); any nonzero dir_norm on this step comes from the BC-aux "
                        "loss instead, see bc_direction_loss below if present"
                    )
                else:
                    log.info(
                        f"           move_dir logprob: old={w['move_dir_old_logprob']:+.3f}  "
                        f"new={w['move_dir_new_logprob']:+.3f}  shift={_shift:+.3f}  "
                        f"-> move_dir-only ratio={_ratio:.3f}"
                    )
            if "bc_direction_loss" in w:
                log.info(f"           bc_direction_loss={w['bc_direction_loss']:.4f}")
            if "estimated_weight_grad_norm" in w:
                _observed = r["layers"].get("execution_net.move_direction")
                _clip_note = ""
                if w.get("clip_active"):
                    _clip_note = "  [aggregate ratio outside clip band -- estimate may be off]"
                log.info(
                    f"           chain: |d(logprob)/d(mean)|={w['d_logprob_d_mean_norm']:.3f}"
                    f"  x  ||trunk activation h||={w['trunk_activation_norm']:.3f}"
                    f"  x  |adv*ratio_agg|={abs(w['advantage'] * w['aggregate_ratio']):.3f}"
                    f" (ratio_agg={w['aggregate_ratio']:.3f})"
                    f"  ≈ {w['estimated_weight_grad_norm']:.1f}"
                    + (f"   (observed: {_observed:.1f})" if _observed is not None else "")
                    + _clip_note
                )
                log.info(
                    f"           (||raw pre-normalize move_direction output||={w['raw_move_norm']:.3f} -- "
                    f"estimate above assumes this is ~1; if it's much smaller, the real normalize-Jacobian "
                    f"amplifies the gradient well beyond the estimate)"
                )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "checkpoint": str(ckpt_path),
                "episode": summary,
                "structural_param_counts": dict(_counts_by_label),
                "records": records,
            }, f, indent=2)
        log.info(f"\nFull per-step/per-layer data written to {out_path}")


if __name__ == "__main__":
    main()

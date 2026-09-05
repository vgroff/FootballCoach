"""Sanity-check the BC training + policy-network pipeline end to end.

Picks the first recorded demonstration episode that (a) ends in a trainee
win and (b) lasts more than 10 real seconds, BC-overfits a FRESH
decision_net/execution_net on ONLY that episode's rows until the loss
bottoms out near its analytic floor, then replays the SAME episode
(identical seed, spawn state, and opponent-type roll) with the overfit
network driving the trainee deterministically -- and reports whether the
outcome/duration come out (almost) the same as the original recording.

If they don't, that points to a structural bug somewhere in the BC
training loop, label canonicalization, network forward pass, or
action-application pipeline: a network that can't even memorize and
reproduce ONE episode it was directly trained on is broken before you ever
get to questions of generalization.

Deliberately does NOT go through PPOTrainer.pretrain_combined() -- that
also does value pretraining and live rollout collection (its Phase 2/3),
neither relevant to a single-episode imitation sanity check. Every actual
computation below (network forward pass, label canonicalization, the BC
loss itself, dataset/env construction, exact-episode replay) reuses the
same production code pretrain_combined()/train.py/replay_episode.py use --
only the orchestration loop around them is specific to this script.

Usage:
    uv run python debug_policy_net.py --demonstrations demonstrations/phase1_rules_immobile
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
import functools
import logging
import math
import random as _random
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("footballcoach.ai.debug_policy")


def _find_first_win_episode(ds, min_duration_s: float):
    """First episode (recording order) with outcome == 'win' and duration
    > min_duration_s. Returns (start, end_inclusive, duration_s, seed).

    Duration computed exactly like debug_value_network.py's own episode
    display: count of is_decision_step==1 rows among the episode's
    TRAINEE-owned rows, times demo_sample_every_n_decisions *
    decision_interval_s -- NOT dones.sum() (double-counts) or a flat
    row-index span (inflated by kick/tackle callback rows).
    """
    from footballcoach.ai.config import load_ai_config
    cfg = load_ai_config()
    sample_interval_s = (
        float(cfg["bc"]["demo_sample_every_n_decisions"])
        * float(cfg["observation"]["decision_interval_s"])
    )
    for start, end in ds.episode_row_ranges(np.arange(len(ds))):
        if ds.classify_outcome(end) != "win":
            continue
        n_decision_steps = sum(
            1 for r in range(start, end + 1)
            if ds._is_trainee[r] > 0.5 and ds._is_decision_step[r] > 0.5
        )
        duration_s = n_decision_steps * sample_interval_s
        if duration_s > min_duration_s:
            seed = ds.episode_seed(end)
            if seed is None:
                log.warning(
                    f"  Episode at rows [{start},{end}] (duration={duration_s:.1f}s) "
                    f"qualifies but has no recorded seed (dataset predates "
                    f"meta_episode_seeds) -- can't replay it, skipping."
                )
                continue
            return start, end, duration_s, seed
    raise SystemExit(
        f"No 'win' episode longer than {min_duration_s}s with a recorded seed "
        f"found in this dataset."
    )


def _bc_overfit_episode(
    trainer, ds, start: int, end: int, opt: torch.optim.Optimizer, *,
    max_epochs: int, batch_size: int, log_every: int, target_loss_margin: float,
    extra_obs: dict | None = None, extra_bc_labels: "torch.Tensor | None" = None,
) -> tuple[float, float, int]:
    """BC-train trainer.decision_net/execution_net on ONLY rows [start, end]
    until bc_loss - floor <= target_loss_margin, or max_epochs is reached.

    Additive margin above the floor, not a multiplicative ratio: with no
    label smoothing (this script's default, matching bc.dec_label_smoothing/
    exec_label_smoothing's own 0.0 defaults), compute_bc_loss_floor()
    legitimately returns exactly 0.0 (a deterministic 0/1 target has zero
    target entropy), which would make any floor*ratio threshold permanently
    unreachable. bc_loss - floor ("bc_adj") is compute_bc_loss_floor()'s own
    documented way to compare loss values independent of the smoothing
    offset -- see its docstring.

    Mirrors PPOTrainer.pretrain_combined()'s Phase 1 per-minibatch step
    (ppo_trainer.py) exactly -- same forward pass / label canonicalization /
    loss function -- just scoped to one episode's rows via
    iterate_minibatches(indices_override=...).

    extra_obs/extra_bc_labels: optional additional (already batched, already
    on `device`) rows concatenated onto every minibatch this call produces --
    used by _dagger_loop() to mix in aggregated on-policy (state, rules-AI
    counterfactual label) pairs alongside the original episode's own
    recorded rows, without this function needing to know DAgger exists.
    extra_bc_labels must be in the same raw/world-frame convention as
    ds._labels (uncanonicalized) -- canonicalize_bc_labels() below is applied
    AFTER concatenation, uniformly to both sources.

    opt is passed in (not built here) and must be trainer.optimizer itself --
    main() builds it there and assigns it to that attribute BEFORE any
    --checkpoint load, so PPOTrainer.load_checkpoint()/_save_checkpoint_to()'s
    existing self.optimizer state_dict save/restore (guarded on
    self.optimizer is not None) picks it up automatically across a resume,
    with no separate optimizer-state mechanism needed here.

    Returns (final_mean_loss, floor, epochs_run).
    """
    from footballcoach.ai.ppo.bc import bc_loss_from_tensor, compute_bc_loss_floor, direction_magnitude_reg
    from footballcoach.ai.obs.canonical import canonicalize_bc_labels, x_sign_of
    from footballcoach.ai.ppo.ppo_trainer import _ai_types

    ep_rows = np.arange(start, end + 1)
    ep_valid = np.intersect1d(ds.valid_indices(), ep_rows)
    if len(ep_valid) == 0:
        raise SystemExit(f"Episode rows [{start},{end}] have zero valid BC rows -- can't train on this episode.")
    log.info(f"BC-overfitting on episode rows [{start},{end}] ({len(ep_valid)} valid rows, batch_size={batch_size})")

    # pos_weight_kick/tackle_attempt: auto-compute from the FULL dataset,
    # exactly like pretrain_combined() does -- this one episode alone is far
    # too small a sample to estimate class imbalance from on its own.
    _pos_weights = ds.compute_pos_weights()
    pos_weight_kick = _pos_weights["kick"]
    pos_weight_tackle_attempt = _pos_weights["tackle_attempt"]

    device = trainer.device
    trainer.decision_net.train()
    trainer.execution_net.train()
    floor = None
    mean_loss = float("nan")
    epoch = 0
    all_params = list(trainer.decision_net.parameters()) + list(trainer.execution_net.parameters())
    for epoch in range(1, max_epochs + 1):
        losses = []
        grad_norms: list[float] = []
        head_losses: dict[str, list[float]] = {}
        dir_raw_norms: list[float] = []
        for obs_dict, bc_labels in ds.iterate_minibatches(
            batch_size, shuffle=True, device=device, indices_override=ep_valid,
        ):
            if extra_obs is not None:
                obs_dict = {k: torch.cat([v, extra_obs[k]], dim=0) for k, v in obs_dict.items()}
                bc_labels = torch.cat([bc_labels, extra_bc_labels], dim=0)
            sat, oat = _ai_types(obs_dict)
            d_heads = trainer.decision_net(
                obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
                obs_dict["ball_feat"], obs_dict["global_feat"], sat, oat,
            )
            e_heads = trainer.execution_net(
                obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
                obs_dict["ball_feat"], obs_dict["global_feat"], d_heads, sat, oat,
            )
            bc_labels = canonicalize_bc_labels(bc_labels, x_sign_of(obs_dict["self_feat"]))
            # direction_loss_mode: production bc_loss_from_tensor() handles
            # natively (bc.py) -- forwarding trainer._bc_dir_loss_mode
            # (config-driven, --direction-loss-mode overrides it for this
            # run only) through to the same call real BC pretraining uses.
            loss, _bkdn = bc_loss_from_tensor(
                bc_labels, d_heads, e_heads,
                pos_weight_kick=pos_weight_kick,
                pos_weight_tackle_attempt=pos_weight_tackle_attempt,
                direction_loss_mode=trainer._bc_dir_loss_mode,
                return_breakdown=True,
            )
            # Raw-vector magnitude regularizer -- standalone/unconditional
            # (see direction_magnitude_reg()'s docstring in bc.py), same
            # trainer._bc_dir_mag_reg_coef (config-driven,
            # --bc-dir-mag-reg-coef overrides it for this run only) real BC
            # pretraining now always applies too.
            loss = loss + direction_magnitude_reg(e_heads, trainer._bc_dir_mag_reg_coef)
            # Diagnostic: printed every epoch regardless of mode/coef so you
            # can see whether ||raw|| is well-behaved.
            dir_raw_norms.append(e_heads.move_direction_unnormalized.norm(dim=-1).mean().item())
            if floor is None:
                floor = compute_bc_loss_floor(
                    bc_labels, pos_weight_kick=pos_weight_kick,
                    pos_weight_tackle_attempt=pos_weight_tackle_attempt,
                )
            opt.zero_grad()
            loss.backward()
            # Same clip production Phase 1 BC training applies
            # (ppo_trainer.py:2298-2302, gated on self._bc_max_grad_norm),
            # missing here before now -- without it, once this tiny 48-row
            # dataset gets driven to near-zero loss, BCE gradients near
            # saturation stay steep rather than vanishing, so a single
            # unclipped step can overshoot into a bad region (confirmed live:
            # loss sat at ~0.0002 for ~300 epochs then spiked to ~1.9 in the
            # space of 20 epochs before slowly re-recovering over hundreds
            # more) -- worse with a resumed (non-cold) optimizer, whose
            # accumulated momentum keeps pushing in the pre-spike direction
            # for a few more steps before correcting.
            # clip_grad_norm_ always returns the PRE-clip total norm across
            # all_params regardless of whether it actually rescales anything
            # -- pass float("inf") as the limit when clipping is disabled so
            # this still measures without ever touching the gradients, same
            # "measure only" trick ppo_trainer.py's raw_grad_norm uses.
            clip_limit = trainer._bc_max_grad_norm if trainer._bc_max_grad_norm is not None else float("inf")
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, clip_limit).item()
            opt.step()
            losses.append(loss.item())
            grad_norms.append(grad_norm)
            for k, v in _bkdn.items():
                head_losses.setdefault(k, []).append(v)
        mean_loss = float(np.mean(losses))
        if epoch == 1 or epoch % log_every == 0:
            mean_grad_norm = float(np.mean(grad_norms))
            mean_dir_raw_norm = float(np.mean(dir_raw_norms))
            head_str = "  ".join(f"{k}={np.mean(v):.4f}" for k, v in head_losses.items())
            log.info(
                f"  [bc overfit] epoch {epoch}/{max_epochs}  loss={mean_loss:.4f}  floor={floor:.4f}  "
                f"grad_norm={mean_grad_norm:.3f}"
                + (f"  (clip={trainer._bc_max_grad_norm})" if trainer._bc_max_grad_norm is not None else "")
                + f"  move_dir_raw_norm={mean_dir_raw_norm:.3f}"
                + f"  (mode={trainer._bc_dir_loss_mode}"
                + (f", mag_reg_coef={trainer._bc_dir_mag_reg_coef}" if trainer._bc_dir_mag_reg_coef > 0.0 else "")
                + ")"
            )
            log.info(f"    [head losses] {head_str}")
        if floor is not None and (mean_loss - floor) <= target_loss_margin:
            log.info(
                f"  [bc overfit] converged at epoch {epoch}: "
                f"loss={mean_loss:.4f}  floor={floor:.4f}  (loss-floor={mean_loss - floor:.4f} <= {target_loss_margin})"
            )
            return mean_loss, floor, epoch
    log.warning(f"  [bc overfit] did not reach target after {max_epochs} epochs (final loss={mean_loss:.4f}, floor={floor:.4f})")
    return mean_loss, floor, epoch


def _build_replay_env(trainer, seed: int, *, deterministic: bool = True):
    """Build + reset a phase-1 ScenarioEnv with trainer.decision_net/
    execution_net driving the trainee, seeded/opponent-matched exactly like
    the original recording. Returns the reset env, ready to env.step().
    """
    from footballcoach.ai.curriculum.envs import build_env, bc_label_fn_for_phase_player
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.scripts.replay_episode import _phase1_opponent_probs
    from footballcoach.rules_ai import Phase1RulesAI

    trainer.decision_net.eval()
    trainer.execution_net.eval()

    env = build_env(PHASES_BY_ID[1])
    # ScenarioEnv's own built-in mechanism for "policy net plays": set this
    # BEFORE reset(), and reset() auto-assigns NeuralPlayerAI to the trainee
    # (scenario_env.py) -- no manual player.ai / apply_nn_action.py call needed.
    env.sample_action_fn = functools.partial(trainer._sample_action, deterministic=deterministic)
    # (player, match) -> BCLabel, computed INSIDE NeuralPlayerAI.act() at the
    # same instant as the observation -- see that method's own bc_label_fn
    # docstring and phase1_labels_for_player's "CRITICAL -- CALLER-SIDE
    # TIMING" section (bc.py) for why this can't be a separate call after
    # env.step() returns (_collect_dagger_rollout() used to do exactly
    # that; fixed 2026-09, same fix as ai/ppo/dagger.py's own).
    env.bc_label_fn = bc_label_fn_for_phase_player(1)
    env.reset(seed=seed)

    # Second, OVERRIDING opponent-type roll -- exact same logic as
    # ai/scripts/replay_episode.py's build_replay_match() (see its docstring
    # for why: record_demonstrations.py never trusts build_1v1_scenario's own
    # internal seed-determined roll, it re-rolls independently from a fresh
    # random.Random(seed) and folds any "neural" region into "rules"). That
    # function builds its own raw Match rather than accepting one, so it
    # can't be called directly against ScenarioEnv's -- duplicated inline
    # here for the same reason replay_episode.py's own run_headless()
    # duplicates it inline too. Keep in sync with build_replay_match() by
    # hand if that ever changes.
    opponent_rules_prob, opponent_immobile_prob = _phase1_opponent_probs()
    match = env._loop.match
    opponent = match.player_by_id("opponent")
    roll = _random.Random(seed).random()
    if roll < opponent_rules_prob:
        opponent.ai = Phase1RulesAI()
        match._opponent_use_rules_ai, match._opponent_is_immobile = True, False
    elif opponent_immobile_prob is not None and roll >= opponent_rules_prob + opponent_immobile_prob:
        opponent.ai = Phase1RulesAI()
        match._opponent_use_rules_ai, match._opponent_is_immobile = True, False
    else:
        # ai stays None (Phase1RulesAI.decide() and others key off
        # `opponent.ai is None` as their "can this opponent ever move/
        # contest" signal) -- but current_order is set directly so the
        # opponent still moves like a real player instead of Match.
        # _apply_movement's "no order this tick" branch, which now RAISES
        # rather than silently coasting. Same fix as replay_episode.py's
        # apply_phase1_opponent_roll() (this is its duplicate, see above).
        from footballcoach.orders import JogOrder
        opponent.ai = None
        opponent.current_order = JogOrder(direction=opponent.velocity)
        match._opponent_use_rules_ai, match._opponent_is_immobile = False, True
    return env


# Fields RolloutBuffer.add() appends to, one entry per row -- kept in sync
# with rollout_buffer.py's own field list by hand (that module has no public
# "give me every per-row field name" accessor). Used below to extend/
# subsample a buffer used purely as an in-memory (obs, bc_label) store (all
# other fields carry dummy placeholder values -- see _collect_dagger_rollout)
# without duplicating RolloutBuffer's storage/stacking logic.
_ROLLOUT_BUFFER_FIELDS = [
    "obs", "actions", "log_probs", "values", "rewards", "dones",
    "bc_labels", "head_log_probs", "weights", "reward_comps",
    "step_outcomes", "track_ids",
]


def _extend_buffer(dst, src) -> None:
    for f in _ROLLOUT_BUFFER_FIELDS:
        getattr(dst, f).extend(getattr(src, f))


def _subsample_buffer(buf, max_size: int, rng: _random.Random) -> None:
    """In-place: if buf has more than max_size rows, keep a uniform random
    subset of exactly max_size rows -- the "start randomly dropping past a
    certain point" eviction policy requested for the DAgger aggregated
    buffer. A plain uniform re-sample of the whole (already fully
    materialized, in-memory) buffer rather than a streaming reservoir
    sampler -- simpler, and equivalent since this only ever runs between
    DAgger iterations on the buffer's full current contents.
    """
    n = len(buf)
    if n <= max_size:
        return
    keep = sorted(rng.sample(range(n), max_size))
    for f in _ROLLOUT_BUFFER_FIELDS:
        vals = getattr(buf, f)
        setattr(buf, f, [vals[i] for i in keep])


def _dagger_extra_tensors(buf, device):
    """Pack an aggregated-rows buffer into the (obs_dict, bc_labels) shape
    _bc_overfit_episode's extra_obs/extra_bc_labels params expect, via
    RolloutBuffer.as_tensors() -- the same stacking production PPO training
    uses to turn a rollout into network-ready tensors (ppo_trainer.py),
    reused here rather than re-implemented. Returns (None, None) for an
    empty buffer (nothing to concatenate yet -- DAgger iteration 1, before
    any rollout has been collected).
    """
    if len(buf) == 0:
        return None, None
    t = buf.as_tensors([0.0] * len(buf), [0.0] * len(buf))
    extra_obs = {k[len("obs/"):]: v.to(device) for k, v in t.items() if k.startswith("obs/")}
    extra_bc_labels = t["bc_labels"].to(device)
    return extra_obs, extra_bc_labels


def _collect_dagger_rollout(trainer, seed: int, max_steps: int):
    """Roll the CURRENT policy out closed-loop from `seed`, deterministically,
    with NO divergence stopping -- the whole point of DAgger is to visit
    wherever the policy actually goes (including off the original recorded
    trajectory) and get those states correctly labelled, not to bail out at
    the first sign of drift like _replay_episode_step_by_step does.

    Every visited state is labelled via phase1_labels_for_player(), called
    from INSIDE NeuralPlayerAI.act() at the same instant the observation is
    encoded (see _build_replay_env's env.bc_label_fn wiring, and
    NeuralPlayerAI.bc_label_fn's own docstring in rules_ai.py) -- the SAME
    mechanism production's on-policy BC-aux-loss training and
    ai/ppo/dagger.py's collect_dagger_rollout() use. Previously this called
    phase1_labels(env, "trainee") SEPARATELY, after env.step() had already
    returned -- a real, since-fixed bug (see phase1_labels_for_player()'s
    "CRITICAL -- CALLER-SIDE TIMING" docstring section in bc.py): that call
    site saw POST-movement state, one physics tick later than the
    observation it was meant to accompany, and the counterfactual's own
    exploratory order.execute() call could additionally desync match.rng
    for any REAL kick that fired afterward (see the "CRITICAL -- SHARED
    RNG STREAM" section of the same docstring). Fixed here the same way as
    production.

    Returns (buffer, summary) -- buffer is a RolloutBuffer holding one row
    per decision step (obs + bc_label, all other fields dummy placeholders,
    see RolloutBuffer.add() call below); summary is
    {"outcome": str|None, "n_steps": int, "total_reward": float}.
    """
    from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer

    env = _build_replay_env(trainer, seed, deterministic=True)
    buf = RolloutBuffer()
    total_reward = 0.0
    outcome = None
    n_steps = 0
    with torch.no_grad():
        for n_steps in range(1, max_steps + 1):
            _, reward, done, info = env.step()
            tr = env.last_trainee_transition
            if tr is not None:
                total_reward += reward
                # Computed INSIDE NeuralPlayerAI.act() (see
                # _build_replay_env's env.bc_label_fn wiring), at the same
                # instant as tr["obs"] -- already a numpy array.
                buf.add(
                    obs=tr["obs"], action={}, log_prob=0.0, value=0.0,
                    reward=0.0, done=0.0, bc_label=tr.get("bc_label"),
                )
            if done:
                outcome = info.trial_outcome
                break
    return buf, {"outcome": outcome, "n_steps": n_steps, "total_reward": total_reward}


def _dagger_loop(
    trainer, ds, start: int, end: int, seed: int, opt: torch.optim.Optimizer, *,
    dagger_iterations: int, bc_epochs_per_iter: int, buffer_max_size: int,
    max_replay_steps: int, batch_size: int, log_every: int,
    target_loss_margin: float, rng_seed: int,
) -> tuple[float, float, int]:
    """DAgger (Ross, Gordon & Bagnell 2011): alternate BC-training on
    (original episode rows + aggregated on-policy rows) with rolling the
    resulting policy out and aggregating newly-visited states, each labelled
    with what Phase1RulesAI would actually do from there. Directly addresses
    the exposure-bias/covariate-shift gap plain single-episode BC has --
    pure BC never sees "recover from a slightly-off state" examples, so
    small per-step errors compound in closed-loop replay even at ~zero
    training loss (see this session's earlier step-by-step replay: BC loss
    0.0002 but a wrong discrete decision-head flip at step 10 still blew up
    to 2.67m divergence by step 12). Aggregating the policy's OWN visited
    states (correctly labelled) directly teaches it to recover from exactly
    the kind of drift it produces, instead of only ever seeing the original
    trajectory's states.

    Reuses _bc_overfit_episode (via extra_obs/extra_bc_labels) and
    _collect_dagger_rollout (via phase1_labels(), production's own on-policy
    BC label source) -- no BC loss / label-generation logic duplicated here,
    only the outer aggregate-and-retrain loop itself.

    Returns (final_mean_loss, final_floor, total_epochs_run) -- same shape
    as _bc_overfit_episode's return, so main()'s checkpoint-save/summary
    code doesn't need to know which path produced it.
    """
    from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
    rng = _random.Random(rng_seed)
    aggregated = RolloutBuffer()
    mean_loss = floor = float("nan")
    total_epochs = 0
    for it in range(1, dagger_iterations + 1):
        extra_obs, extra_bc_labels = _dagger_extra_tensors(aggregated, trainer.device)
        log.info(f"[dagger] iteration {it}/{dagger_iterations}: BC training ({len(aggregated)} aggregated rows so far)")
        mean_loss, floor, epochs_run = _bc_overfit_episode(
            trainer, ds, start, end, opt,
            max_epochs=bc_epochs_per_iter, batch_size=batch_size,
            log_every=log_every, target_loss_margin=target_loss_margin,
            extra_obs=extra_obs, extra_bc_labels=extra_bc_labels,
        )
        total_epochs += epochs_run
        new_buf, summary = _collect_dagger_rollout(trainer, seed, max_replay_steps)
        _extend_buffer(aggregated, new_buf)
        _subsample_buffer(aggregated, buffer_max_size, rng)
        log.info(
            f"[dagger] iteration {it}/{dagger_iterations} rollout: outcome={summary['outcome']}  "
            f"steps={summary['n_steps']}  reward={summary['total_reward']:.2f}  "
            f"aggregated_buffer_size={len(aggregated)}"
        )
    return mean_loss, floor, total_epochs


# PlayerFeatures column indices (see schema.py field order) -- same constants
# debug_value_network.py uses, copied verbatim rather than re-derived: this
# exact hardcoded-index approach has bitten this codebase before when
# re-derived independently elsewhere (diagnose_crossing_head.py silently used
# the wrong per-axis normalization -- see ai/knowledge.md), so the established,
# already-fixed reference values are reused as-is.
_POS_X, _POS_Y, _VEL_X, _VEL_Y = 30, 31, 9, 10
# Pitch half-diagonal -- ai/obs/encoder.py normalizes BOTH position AND
# velocity x/y by this SAME constant (not separate per-axis divisors).
_HALF_DIAG = math.hypot(52.5, 34.0)


def _fmt_head_line(
    action_name: str, exec_move: int, sprint: int, kick: int, tackle_attempt: int,
    move_dir_deg: float, kick_dir_xyz, kick_power,
) -> str:
    """Shared formatting for the full execution-head breakdown, used by both
    _decode_original_step and _decode_replay_step so the two sides line up
    column-for-column. kick_dir_xyz/kick_power are None when kick == 0 (no
    kick this tick -- those heads' values are meaningless/untrained on
    non-kick rows, matching bc.py's own kicked_mask gating).

    kick_spin is deliberately omitted: it's a permanently-frozen head with
    zero physical effect right now (apply_nn_action.py hardcodes zero spin
    regardless of what this head outputs -- see agent_plans/
    spin_implementation_plan.md section 0), so displaying it would only add
    noise, not information.
    """
    kick_dir_str = (
        f"({kick_dir_xyz[0]:+.2f},{kick_dir_xyz[1]:+.2f},{kick_dir_xyz[2]:+.2f})"
        if kick_dir_xyz is not None else "n/a"
    )
    kick_power_str = f"{kick_power:.2f}" if kick_power is not None else "n/a"
    return (
        f"action={action_name:<14} exec_move={exec_move} sprint={sprint} kick={kick} tackle_attempt={tackle_attempt}  "
        f"move_dir={move_dir_deg:+6.1f}deg  kick_dir={kick_dir_str:<18}  kick_power={kick_power_str}"
    )


def _decode_original_step(ds, row: int) -> tuple[tuple[float, float], tuple[float, float], str]:
    """(pos_xy, vel_xy, full_decision_line) for one recorded trainee
    decision-step row, decoded from the dataset's own stored observation
    (self_feat) and BC label (bc_labels) -- the rules-AI's actual recorded
    action, not a re-derivation."""
    from footballcoach.ai.ppo.bc import (
        _I_DIR_X, _I_DIR_Y, _I_EXEC_MOVE, _I_GP_EXTRA, _I_HOLD, _I_KICK_DIR_X,
        _I_KICK_DIR_Y, _I_KICK_DIR_Z, _I_KICK_POWER, _I_KICK_THIS_TICK, _I_MARK,
        _I_MOVE, _I_PASS, _I_SHOOT, _I_SPRINT, _I_TACKLE, _I_TACKLE_ATTEMPT,
    )
    feat = ds._self_feat[row]
    pos = (round(float(feat[_POS_X]) * _HALF_DIAG, 2), round(float(feat[_POS_Y]) * _HALF_DIAG, 2))
    vel = (round(float(feat[_VEL_X]) * _HALF_DIAG, 2), round(float(feat[_VEL_Y]) * _HALF_DIAG, 2))
    label = ds._labels[row]
    action_name = "none"
    # Same priority order as ai/action/gating.py's _HEAD_ORDER (winner-take-all
    # tie-break: shoot > pass > move > tackle > get_possession > mark > hold) --
    # get_possession is a real selectable action (gating.py's SelectedAction),
    # not just the gp_extra "headroom above tackle_prob" auxiliary signal, so
    # it belongs in this list alongside the other 6 or a real rules-AI
    # get_possession decision would always misreport as "none" here.
    for name, col in [("shoot", _I_SHOOT), ("pass", _I_PASS), ("move", _I_MOVE),
                       ("tackle", _I_TACKLE), ("get_possession", _I_GP_EXTRA),
                       ("mark", _I_MARK), ("hold", _I_HOLD)]:
        if label[col] > 0.5:
            action_name = name
            break
    exec_move = int(label[_I_EXEC_MOVE] > 0.5)
    sprint = int(label[_I_SPRINT] > 0.5)
    kick = int(label[_I_KICK_THIS_TICK] > 0.5)
    tackle_attempt = int(label[_I_TACKLE_ATTEMPT] > 0.5)
    dir_angle = math.degrees(math.atan2(float(label[_I_DIR_Y]), float(label[_I_DIR_X])))
    kick_dir_xyz = (
        (float(label[_I_KICK_DIR_X]), float(label[_I_KICK_DIR_Y]), float(label[_I_KICK_DIR_Z]))
        if kick else None
    )
    kick_power = float(label[_I_KICK_POWER]) if kick else None
    decision = _fmt_head_line(action_name, exec_move, sprint, kick, tackle_attempt,
                               dir_angle, kick_dir_xyz, kick_power)
    return pos, vel, decision


def _decode_replay_decision(match) -> str:
    """Full execution-head breakdown for the trainee's most recent decision,
    via NeuralPlayerAI's cached GatingResult (the winner-take-all-resolved
    action that was actually applied to the player, same object
    apply_nn_action.py consumes) -- NOT position/velocity, see
    _replay_episode_step_by_step's own docstring for why those are read
    separately (pre-decision, not post-decision) to stay time-aligned with
    the recorded rows."""
    trainee = match.player_by_id("trainee")
    gating = getattr(trainee.ai, "_last_gating", None)
    if gating is None:
        return "n/a"
    action_name = str(getattr(gating.selected, "name", gating.selected)).lower()
    exec_move = int(gating.exec_move)
    sprint = int(gating.sprint)
    kick = int(gating.kick_this_tick)
    tackle_attempt = int(gating.tackle_attempt)
    dir_angle = (
        math.degrees(math.atan2(float(gating.move_direction[1]), float(gating.move_direction[0])))
        if gating.move_direction is not None else float("nan")
    )
    kick_dir_xyz = (
        (float(gating.kick_direction[0]), float(gating.kick_direction[1]), float(gating.kick_direction[2]))
        if kick and gating.kick_direction is not None else None
    )
    kick_power = gating.kick_power_fraction if kick else None
    return _fmt_head_line(action_name, exec_move, sprint, kick, tackle_attempt,
                           dir_angle, kick_dir_xyz, kick_power)


def _read_trainee_pos_vel(match) -> tuple[tuple[float, float], tuple[float, float]]:
    """Real engine position/velocity for the trainee, right now -- no obs
    decode needed, this is live Match state."""
    trainee = match.player_by_id("trainee")
    return (
        (round(trainee.position.x, 2), round(trainee.position.y, 2)),
        (round(trainee.velocity.x, 2), round(trainee.velocity.y, 2)),
    )


def _replay_episode_step_by_step(
    trainer, ds, start: int, end: int, seed: int, *, max_divergence_m: float = 1.0,
    stop_on_divergence: bool = True, max_replay_steps: int = 500,
) -> dict:
    """Step the recorded (rules-AI) episode and a deterministic replay
    (overfit policy) side by side, ONE decision interval at a time, logging
    an alternating rules/neural line pair per step.

    stop_on_divergence=True (default): stops as soon as the trainee's
    position diverges by more than max_divergence_m between the two, or when
    either sequence naturally ends first -- the original "does this even
    memorize one episode" sanity-check behaviour.

    stop_on_divergence=False: ignores max_divergence_m as a stop condition
    (divergence is still measured and reported) and keeps stepping the
    replay past the end of the original recording -- printing NEURAL-only
    lines once the original rows run out -- until the replay's own episode
    ends (done=True) or max_replay_steps is hit. Use this once the policy is
    expected to actually complete the scenario on its own (e.g. after
    _dagger_loop), so the final report shows the real outcome/reward instead
    of an early divergence bail-out.

    Relies on bc.demo_sample_every_n_decisions == 1 (every recorded row is
    exactly one decision interval, no gaps) for the original rows and
    env.step() calls to correspond 1:1 by index -- true for the current
    config; if that key is ever raised, this stops being step-for-step
    accurate and needs re-deriving.

    Returns a dict: {"stopped_reason": "diverged"|"replay_ended"|
    "original_exhausted"|"max_steps_exhausted", "n_steps": int,
    "final_distance_m": float, "max_distance_m": float,
    "replay_outcome": str|None, "replay_duration_s": float|None,
    "total_reward": float} -- replay_outcome/duration are only set when the
    replay reached its own natural end (stopped_reason == "replay_ended").
    """
    trainee_rows = [
        r for r in range(start, end + 1)
        if ds._is_trainee[r] > 0.5 and ds._is_decision_step[r] > 0.5
    ]
    n_orig_steps = len(trainee_rows)
    loop_steps = n_orig_steps if stop_on_divergence else max(n_orig_steps, max_replay_steps)
    env = _build_replay_env(trainer, seed, deterministic=True)
    match = env._loop.match
    log.info(
        f"  [step-by-step] {n_orig_steps} original decision steps; "
        + (f"stop on >{max_divergence_m}m divergence" if stop_on_divergence
           else f"run to completion (max {loop_steps} steps)")
    )

    # Position/velocity as they stood right BEFORE each decision is made --
    # captured here (the true initial spawn state, right after reset(),
    # matching what trainee_rows[0]'s self_feat represents) and then updated
    # to the post-step state after each env.step() call, for use as the NEXT
    # iteration's pre-decision state. This must NOT be read right after
    # env.step() (as an earlier version of this function did): that reads
    # the state at the END of that decision interval -- one full interval
    # later than what the recorded row represents (the state the rules-AI
    # actually saw when making ITS decision) -- silently comparing apples to
    # oranges, one step out of phase. Confirmed from a real run: the old
    # code's NEURAL-step-i position matched RULES-step-(i+1)'s position
    # almost exactly, meaning the two trajectories were actually tracking
    # closely the whole time and the ~1m "divergence" that triggered an
    # early stop was mostly this phase offset, not real behavioural drift.
    new_pos, new_vel = _read_trainee_pos_vel(match)

    dist = 0.0
    max_dist = 0.0
    total_reward = 0.0
    i = 0
    with torch.no_grad():
        for i in range(loop_steps):
            have_orig = i < n_orig_steps
            if have_orig:
                orig_pos, orig_vel, orig_decision = _decode_original_step(ds, trainee_rows[i])
            _, reward, done, info = env.step()
            total_reward += reward
            new_decision = _decode_replay_decision(match)

            if have_orig:
                log.info(f"  {i:6d}  RULES   pos=({orig_pos[0]:+7.2f},{orig_pos[1]:+7.2f})  vel=({orig_vel[0]:+6.2f},{orig_vel[1]:+6.2f})")
                log.info(f"  {i:6d}  RULES     {orig_decision}")
            log.info(f"  {i:6d}  NEURAL  pos=({new_pos[0]:+7.2f},{new_pos[1]:+7.2f})  vel=({new_vel[0]:+6.2f},{new_vel[1]:+6.2f})")
            log.info(f"  {i:6d}  NEURAL    {new_decision}")

            if have_orig:
                dist = math.hypot(orig_pos[0] - new_pos[0], orig_pos[1] - new_pos[1])
                max_dist = max(max_dist, dist)
            new_pos, new_vel = _read_trainee_pos_vel(match)  # post-step state -> next iteration's pre-decision state
            if have_orig and dist > max_divergence_m and stop_on_divergence:
                log.info(f"  [step-by-step] STOPPED at step {i}: positions diverged by {dist:.2f}m (> {max_divergence_m}m)")
                return {"stopped_reason": "diverged", "n_steps": i + 1, "final_distance_m": dist,
                        "max_distance_m": max_dist, "replay_outcome": None, "replay_duration_s": None,
                        "total_reward": total_reward}
            if done:
                duration_s = info.ticks_elapsed * env._dt_s
                log.info(
                    f"  [step-by-step] replay episode ended at step {i}: outcome={info.trial_outcome}  "
                    f"duration={duration_s:.1f}s  reward={total_reward:.2f}  max_divergence={max_dist:.2f}m"
                )
                return {"stopped_reason": "replay_ended", "n_steps": i + 1, "final_distance_m": dist,
                        "max_distance_m": max_dist, "replay_outcome": info.trial_outcome,
                        "replay_duration_s": duration_s, "total_reward": total_reward}
    if stop_on_divergence:
        log.info(f"  [step-by-step] reached end of original recording ({n_orig_steps} steps), no divergence >{max_divergence_m}m")
        return {"stopped_reason": "original_exhausted", "n_steps": n_orig_steps, "final_distance_m": dist,
                "max_distance_m": max_dist, "replay_outcome": None, "replay_duration_s": None,
                "total_reward": total_reward}
    log.info(f"  [step-by-step] hit max_replay_steps ({loop_steps}) without the replay episode ending")
    return {"stopped_reason": "max_steps_exhausted", "n_steps": loop_steps, "final_distance_m": dist,
            "max_distance_m": max_dist, "replay_outcome": None, "replay_duration_s": None,
            "total_reward": total_reward}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demonstrations", type=str, required=True,
                        help="Directory of recorded demonstration .npz files (e.g. demonstrations/phase1_rules_immobile).")
    parser.add_argument("--min-duration-s", type=float, default=10.0,
                        help="Minimum episode duration (real seconds) to qualify. Default: 10.0.")
    parser.add_argument("--bc-lr", type=float, default=1e-3,
                        help="BC learning rate. Default: ai_config.json bc.bc_learning_rate.")
    parser.add_argument("--bc-max-epochs", type=int, default=1000,
                        help="Max BC epochs over the single episode before giving up. Default: 300.")
    parser.add_argument("--bc-batch-size", type=int, default=1048,
                        help="Minibatch size -- one episode's valid rows are almost always fewer than this, "
                             "so training is effectively full-batch. Default: 2048.")
    parser.add_argument("--bc-log-every", type=int, default=10,
                        help="Log BC loss every N epochs. Default: 20.")
    parser.add_argument("--bc-target-loss-margin", type=float, default=0.00005,
                        help="Stop once (mean BC loss - floor) <= this margin (floor = "
                             "compute_bc_loss_floor()'s analytic minimum for this batch, "
                             "usually 0.0 with no label smoothing). Default: 0.0001.")
    parser.add_argument("--bc-max-grad-norm", type=str, default=None,
                        help="Gradient-norm clip for this script's own BC optimizer step (same "
                             "nn.utils.clip_grad_norm_ call production Phase 1 BC training applies, "
                             "gated on bc.max_grad_norm in ai_config.json). Default: whatever "
                             "bc.max_grad_norm resolves to (currently null in ai_config.json -- i.e. "
                             "NO clipping by default, matching production's current behaviour exactly). "
                             "Pass a float (e.g. 0.5) to clip for just this run without touching shared "
                             "config -- useful since long high-epoch-count runs on a tiny, already "
                             "near-zero-loss dataset can otherwise take a single unclipped step that "
                             "overshoots into a bad region (observed live: stable ~0.0002 loss for "
                             "~300 epochs, then a spike to ~1.9 over the next 20). Pass 'none' to force "
                             "clipping off even if bc.max_grad_norm is set to a number in config.")
    parser.add_argument("--direction-loss-mode", type=str, default=None, choices=["cosine", "mse"],
                        help="Override bc.direction_loss_mode (ai_config.json) for this run only -- "
                             "'cosine' (default in config): 1 - cosine_similarity on the L2-normalized "
                             "move_direction output, scale-invariant in the raw vector's magnitude. "
                             "'mse': plain MSE on the raw (pre-normalize) output against the (unit-"
                             "length) target -- no division-by-norm in this loss's gradient, same "
                             "fixed point as 'cosine' for two unit vectors, better-conditioned "
                             "gradients getting there. See bc_loss_from_tensor()'s docstring (bc.py) "
                             "for the full rationale. Default (unset): whatever bc.direction_loss_mode "
                             "resolves to in config.")
    parser.add_argument("--bc-dir-mag-reg-coef", type=float, default=None,
                        help="Override bc.direction_mag_reg_coef (ai_config.json) for this run only -- "
                             "only used when the resolved direction_loss_mode is 'cosine'. Coefficient "
                             "for an added (||raw_move_direction|| - 1)^2 term pulling the raw output's "
                             "magnitude back toward 1 without changing the cosine loss itself. Try "
                             "0.01-0.1. The per-epoch log line always prints the mean ||raw|| "
                             "regardless, so you can see whether it needs one. Default (unset): "
                             "whatever bc.direction_mag_reg_coef resolves to in config.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the FRESH network's initial weights (not the replayed episode's own recorded seed).")
    parser.add_argument("--device", type=str, default=None,
                        help="PyTorch device. Default: auto-detect (cuda if available, else cpu).")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/overfit/latest.pt",
                        help="Where to save the overfit decision_net/execution_net weights after BC "
                             "training (via PPOTrainer._save_checkpoint_to() -- same checkpoint format "
                             "load_checkpoint()/--checkpoint/--from-pretrained read). "
                             "Default: checkpoints/overfit/latest.pt (overwritten each run).")
    parser.add_argument("--max-divergence-m", type=float, default=1.0,
                        help="Stop the step-by-step replay as soon as the trainee's position diverges "
                             "from the original recording by more than this many metres. Default: 1.0.")
    parser.add_argument("--checkpoint", type=str, nargs="?", const="__use_checkpoint_path__", default=None,
                        help="Resume BC training from an existing checkpoint (via PPOTrainer."
                             "load_checkpoint()) instead of fresh random-init weights. Pass with no "
                             "value to load from --checkpoint-path itself -- i.e. 'continue from what "
                             "this script last saved' -- or give an explicit path to load from "
                             "somewhere else. No effect on --checkpoint-path, which is always where "
                             "the result gets saved afterward, whether or not you continued from it.")
    parser.add_argument("--dagger-iterations", type=int, default=0,
                        help="Enable DAgger (Ross/Gordon/Bagnell 2011): alternate BC training with "
                             "closed-loop rollouts of the current policy, aggregating each visited "
                             "state labelled by what Phase1RulesAI would actually do from there "
                             "(phase1_labels() -- the same live rules-AI query production's on-policy "
                             "BC-aux-loss training uses). Directly targets the exposure-bias gap plain "
                             "single-episode BC has: near-zero training loss doesn't bound closed-loop "
                             "rollout error, because BC never sees 'recover from a slightly-off state' "
                             "examples. Default: 0 (disabled -- single BC pass + divergence-stopping "
                             "replay, the original behaviour of this script).")
    parser.add_argument("--dagger-bc-epochs-per-iter", type=int, default=200,
                        help="Max BC epochs per DAgger iteration (same early-stop-on-floor logic as "
                             "--bc-max-epochs applies within each iteration). Default: 200.")
    parser.add_argument("--dagger-buffer-max-size", type=int, default=2000,
                        help="Cap on the aggregated on-policy (state, label) buffer -- once exceeded, "
                             "a uniform random subset of this size is kept (oldest and newest rows "
                             "equally likely to be dropped). Default: 2000.")
    parser.add_argument("--max-replay-steps", type=int, default=500,
                        help="Safety cap on env ticks for any closed-loop rollout not bounded by the "
                             "original recording's own length -- used by each DAgger rollout, and by "
                             "the final replay when --dagger-iterations > 0 (which runs to completion "
                             "instead of stopping at divergence). Default: 500.")
    args = parser.parse_args()
    if args.checkpoint == "__use_checkpoint_path__":
        args.checkpoint = args.checkpoint_path

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    log.info(f"Device: {device}")

    from footballcoach.ai.bc.dataset import DemonstrationDataset
    from footballcoach.ai.config import load_ai_config
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

    ds = DemonstrationDataset.from_directory(args.demonstrations)
    start, end, orig_duration_s, seed = _find_first_win_episode(ds, args.min_duration_s)
    log.info(
        f"Selected episode: rows [{start},{end}]  outcome=win  "
        f"duration={orig_duration_s:.1f}s  seed={seed}"
    )

    cfg = load_ai_config()
    bc_lr = args.bc_lr if args.bc_lr is not None else float(cfg["bc"].get("bc_learning_rate", 1e-4))

    trainer = PPOTrainer.from_config(device=device, inference_only=True)
    # Assigned to trainer.optimizer (rather than kept as a local variable)
    # BEFORE any --checkpoint load below, specifically so PPOTrainer.
    # load_checkpoint()'s existing self.optimizer state_dict restore (only
    # engages when self.optimizer is not None) has something to load into --
    # and so _save_checkpoint_to() picks it up automatically afterward. Same
    # single-param-group Adam every run, so load_checkpoint()'s param-group-
    # count check always matches across a resume between two runs of this
    # script.
    trainer.optimizer = torch.optim.Adam(
        list(trainer.decision_net.parameters()) + list(trainer.execution_net.parameters()),
        lr=bc_lr, eps=1e-5,
    )
    # --bc-max-grad-norm overrides trainer._bc_max_grad_norm (set by
    # PPOTrainer.__init__ from ai_config.json's bc.max_grad_norm, itself
    # None right now -- no clipping -- unless explicitly set) for just this
    # script's own optimizer step, without touching shared config.
    if args.bc_max_grad_norm is not None:
        if args.bc_max_grad_norm.strip().lower() == "none":
            trainer._bc_max_grad_norm = None
        else:
            trainer._bc_max_grad_norm = float(args.bc_max_grad_norm)
        log.info(f"--bc-max-grad-norm: overriding BC grad-norm clip to {trainer._bc_max_grad_norm}")
    # --direction-loss-mode / --bc-dir-mag-reg-coef override trainer.
    # _bc_dir_loss_mode / _bc_dir_mag_reg_coef (set by PPOTrainer.__init__
    # from ai_config.json's bc.direction_loss_mode / direction_mag_reg_coef)
    # for just this run, without touching shared config -- same pattern as
    # --bc-max-grad-norm above.
    if args.direction_loss_mode is not None:
        trainer._bc_dir_loss_mode = args.direction_loss_mode
        log.info(f"--direction-loss-mode: overriding to {trainer._bc_dir_loss_mode!r}")
    if args.bc_dir_mag_reg_coef is not None:
        trainer._bc_dir_mag_reg_coef = args.bc_dir_mag_reg_coef
        log.info(f"--bc-dir-mag-reg-coef: overriding to {trainer._bc_dir_mag_reg_coef}")
    if args.checkpoint is not None:
        checkpoint_load_path = Path(args.checkpoint)
        if checkpoint_load_path.exists():
            trainer.load_checkpoint(checkpoint_load_path)
            log.info(f"Continuing BC training from checkpoint: {checkpoint_load_path}")
        else:
            log.warning(
                f"--checkpoint {checkpoint_load_path} does not exist -- "
                f"starting from fresh random-init weights instead."
            )
    if args.dagger_iterations > 0:
        log.info(
            f"DAgger enabled: {args.dagger_iterations} iteration(s), "
            f"{args.dagger_bc_epochs_per_iter} BC epoch(s)/iteration, "
            f"buffer cap {args.dagger_buffer_max_size}"
        )
        mean_loss, floor, epochs_run = _dagger_loop(
            trainer, ds, start, end, seed, trainer.optimizer,
            dagger_iterations=args.dagger_iterations,
            bc_epochs_per_iter=args.dagger_bc_epochs_per_iter,
            buffer_max_size=args.dagger_buffer_max_size,
            max_replay_steps=args.max_replay_steps,
            batch_size=args.bc_batch_size, log_every=args.bc_log_every,
            target_loss_margin=args.bc_target_loss_margin, rng_seed=args.seed,
        )
    else:
        mean_loss, floor, epochs_run = _bc_overfit_episode(
            trainer, ds, start, end, trainer.optimizer,
            max_epochs=args.bc_max_epochs, batch_size=args.bc_batch_size,
            log_every=args.bc_log_every, target_loss_margin=args.bc_target_loss_margin,
        )

    # Save right after BC training (not after replay, which is read-only
    # anyway) so the overfit weights are on disk even if replay itself hits
    # an unrelated error -- same primitive train.py's --pretrain-from-
    # checkpoint/--from-pretrained read back via load_checkpoint().
    #
    # Only overwrite if this run's loss actually beats whatever's already
    # saved there -- --checkpoint-path doubles as both the resume source
    # (--checkpoint's bare-flag default) and the save destination, so
    # without this, ANY run (even a short one that never converges) would
    # silently clobber a previously well-converged checkpoint, which is
    # exactly what happened testing --checkpoint earlier this session.
    # _save_checkpoint_to() itself carries no loss metadata (it's a shared
    # primitive also used by pretrain_combined(), not worth bloating with a
    # debug-script-specific field for this), so the comparison value lives
    # in a small sidecar JSON next to the checkpoint instead.
    import json
    checkpoint_path = Path(args.checkpoint_path)
    loss_sidecar_path = checkpoint_path.with_suffix(".loss.json")
    prev_loss = None
    if loss_sidecar_path.exists():
        prev_loss = json.loads(loss_sidecar_path.read_text()).get("bc_loss")
    if prev_loss is None or mean_loss < prev_loss:
        trainer._save_checkpoint_to(checkpoint_path)
        loss_sidecar_path.write_text(json.dumps({"bc_loss": mean_loss}))
        log.info(
            f"Saved overfit checkpoint: {checkpoint_path}  (loss={mean_loss:.4f}"
            + (f", previous={prev_loss:.4f})" if prev_loss is not None else ", no previous checkpoint)")
        )
    else:
        log.info(
            f"Kept existing checkpoint: {checkpoint_path}  (its loss={prev_loss:.4f} <= "
            f"this run's loss={mean_loss:.4f}, not overwriting)"
        )

    stop_on_divergence = args.dagger_iterations <= 0
    log.info(f"Replaying episode seed={seed} deterministically with the overfit policy (step by step)...")
    log.info("=" * 70)
    result = _replay_episode_step_by_step(
        trainer, ds, start, end, seed, max_divergence_m=args.max_divergence_m,
        stop_on_divergence=stop_on_divergence, max_replay_steps=args.max_replay_steps,
    )
    log.info("=" * 70)

    if result["stopped_reason"] == "replay_ended":
        replay_outcome = DemonstrationDataset._OUTCOME_LABEL_MAP.get(
            result["replay_outcome"], result["replay_outcome"]
        )
        log.info(f"Original : outcome=win               duration={orig_duration_s:.1f}s")
        log.info(
            f"Replayed : outcome={replay_outcome:<8} ({result['replay_outcome']})  "
            f"duration={result['replay_duration_s']:.1f}s  reward={result['total_reward']:.2f}  "
            f"(ran {result['n_steps']} steps, max divergence={result['max_distance_m']:.2f}m)"
        )
        outcome_match = replay_outcome == "win"
        duration_close = abs(result["replay_duration_s"] - orig_duration_s) <= max(2.0, 0.25 * orig_duration_s)
        if outcome_match and duration_close:
            log.info("PASS: replayed outcome matches and duration is close.")
        elif outcome_match:
            log.warning("PARTIAL: outcome matches but duration diverged more than expected.")
        else:
            log.warning("FAIL: replayed outcome does not match the original recording.")
    elif result["stopped_reason"] == "diverged":
        log.warning(
            f"FAIL: trainee position diverged from the original recording by "
            f"{result['final_distance_m']:.2f}m (> {args.max_divergence_m}m) after only "
            f"{result['n_steps']} of the original episode's decision steps."
        )
    elif result["stopped_reason"] == "max_steps_exhausted":
        log.warning(
            f"FAIL: replay ran {result['n_steps']} steps (max_replay_steps) without reaching a "
            f"terminal outcome. reward so far={result['total_reward']:.2f}  "
            f"max divergence from original={result['max_distance_m']:.2f}m"
        )
    else:  # original_exhausted
        log.info(
            f"PASS: replay tracked the original recording within {args.max_divergence_m}m "
            f"for all {result['n_steps']} decision steps (replay itself hadn't ended yet)."
        )
    log.info(
        f"(BC training: {epochs_run} epoch(s), final loss={mean_loss:.4f}, floor={floor:.4f}, "
        f"loss-floor={mean_loss - floor:.4f})"
    )


if __name__ == "__main__":
    main()

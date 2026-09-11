"""Batched multi-environment rollout collection.

``rollout_worker.py`` runs ``n_processes`` worker PROCESSES, each with
its own single ``ScenarioEnv``, each doing its own batch-of-1 CPU network
forward pass whenever its trainee needs a decision. Profiled this session:
rollout collection is ~88% CPU neural-network inference (not physics), and a
direct benchmark showed 6 separate batch=1 CPU forward passes take 51.2s for
a fixed workload where 1 batch=6 CPU call takes only 10.5s (~4.88x) -- nearly
all of that gap is Python/PyTorch dispatch overhead on tiny batch-of-1 calls,
not actual compute. This module gets that speedup by running
``envs_per_process`` environments TOGETHER in one process, sharing one
trainer, and batching every environment's trainee decision into ONE
``PPOTrainer._sample_action_batch()`` call per round instead of one
``_sample_action()`` call per environment per decision.

Near-zero behavior change: ``ScenarioEnv.step()`` is architecturally almost
always exactly one full decision interval (``NeuralPlayerAI``'s
``decision_interval_ticks`` is constructed equal to
``ScenarioEnv._ticks_per_decision``, the same loop bound ``step()`` uses
internally -- ``ai/env/scenario_env.py:152,299,313,414``), so the trainee is
*usually* due for a decision on the very first physics tick of every
``env.step()`` call, whether continuing an existing episode or immediately
following a fresh ``env.reset()`` (which itself needs no decision at all).
That's what lets calling every environment's next ``env.step()`` (or
``env.reset()``, for one that just finished) together, once per round, keep
every environment's decisions naturally, mostly synchronized -- no waiting,
no held-back episodes, no global tick counter needed.

NOT a hard guarantee, though: ``ScenarioEnv.step()`` has at least one
legitimate early-exit (trainee already in the opponent box with possession
-- see its own comment) that can end a decision interval several ticks
short, leaving the NEXT round's would-be decision genuinely not due yet for
that one env. An earlier version of this module asserted every env must be
due at the start of a round and crashed a real training run the first time
this fired in practice -- ``collect()`` now checks
``ai.is_due_for_decision()`` first and simply skips a not-due env from that
round's batch (see ``collect()``'s own docstring for the full mechanism);
its ``_precomputed_result`` stays unset, so its own ``env.step()`` call
falls through to ``act()``'s completely normal cached-gating path for that
tick -- identical to what happens for that same tick in the unbatched
``rollout_worker.py`` path, so this is not a behavior change, just
tolerance for something the original design incorrectly assumed could never
happen.

Mechanism, per round:
  1. For each env: check its trainee's ``NeuralPlayerAI.is_due_for_decision()``;
     for due envs only, call ``prepare()`` (see ``rules_ai.py`` -- the exact
     same call that would fire naturally as the first physics tick inside
     the upcoming ``env.step()``, just invoked one line earlier so its obs
     can be pooled with every other due env's). Not-due envs are skipped
     entirely this round (see above).
  2. Concatenate the collected obs dicts (due envs only) into one batch;
     call ``trainer._sample_action_batch(...)`` ONCE.
  3. Stash each due env's result on its trainee AI's ``_precomputed_result``.
  4. Call ``env.step()`` normally for every env (due or not) -- a due env's
     internal first tick's ``player.ai.act()`` (fired by the completely
     UNMODIFIED ``Match._process_orders()``) sees the precomputed result and
     consumes it via ``apply()`` instead of calling ``sample_action_fn``
     again; every later tick within that same ``env.step()`` call, and every
     tick of a not-due env's ``env.step()`` call, behaves exactly as today
     (cached-gating reapplication via ``act()``'s normal, un-precomputed path).
  5. Normal per-env bookkeeping, mirroring ``rollout_worker.py``'s
     ``_collect()`` line for line, just once per env per round instead of
     once per worker process.

Secondary neural players (e.g. a self-play opponent) are OUT OF SCOPE for
batching in this first pass -- they still decide via their own
``NeuralPlayerAI.act()`` synchronously, one at a time, exactly as today.
Only the trainee's decision is batched. Worth revisiting if a curriculum
phase leans on neural self-play heavily; phase 1's default opponent is
immobile/rules-based, so this is a reasonable scope for now, not a silently
swept-under-the-rug limitation.

Two non-obvious things discovered (the hard way, via a failing test) while
building this:

- ``BatchedEnvGroup.collect()`` NEVER calls ``env.sample_action_fn`` or the
  trainee AI's own ``.sample_action_fn`` -- every decision is always
  precomputed via ``trainer._sample_action_batch(...)`` before ``env.step()``
  runs, so ``NeuralPlayerAI.act()``'s ``sample_action_fn`` fallback branch is
  never reached in batched collection. ``collect()`` exposes its own
  ``deterministic``/``deterministic_decision``/``deterministic_direction``
  kwargs (forwarded straight to ``_sample_action_batch``) as the actual,
  single source of truth for sampling behavior -- setting them on
  ``env.sample_action_fn`` instead is silently ignored.
- Despite the above, ``env.sample_action_fn`` still MUST be set (to
  anything, even a plain ``trainer._sample_action`` never actually called)
  before ``env.reset()`` runs -- ``ScenarioEnv.reset()`` itself checks it to
  decide whether/how to wire a ``NeuralPlayerAI`` onto the trainee at all;
  without it the trainee's ``.ai`` stays ``None`` and every ``prepare()``
  call in this module crashes with ``AttributeError``.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Optional

import torch

log = logging.getLogger("footballcoach.ai.ppo.batched_rollout_worker")


def _stack_obs_dicts(obs_dicts: list) -> dict:
    """Stack a list of unbatched obs dicts (as ``ObservationBatch.to_torch_dict()``
    produces) into one batched dict, ready for ``_sample_action_batch``."""
    keys = obs_dicts[0].keys()
    return {k: torch.stack([o[k] for o in obs_dicts]) for k in keys}


def _new_episode_stats() -> dict:
    return {
        "episode_rewards": [],
        "episode_outcome_labels": [],
        "secondary_episode_rewards": [],
        "episode_outcomes_vs_rules": [],
        "episode_outcomes_vs_neural": [],
        "episode_outcomes_vs_immobile": [],
        "episode_comp_list": [],
        "episode_durations_s": [],
    }


class BatchedEnvGroup:
    """Owns ``n_envs`` ``ScenarioEnv`` instances and steps them together,
    one round at a time, batching the trainee's decision network call
    across all of them. Single-process, directly testable without any
    multiprocessing involved -- see ``_batched_worker_main`` below for the
    Process+Pipe wrapper real parallel collection uses.
    """

    def __init__(self, envs: list, trainer, seeds: Optional[list] = None) -> None:
        assert len(envs) > 0, "BatchedEnvGroup needs at least one env"
        if seeds is not None:
            assert len(seeds) == len(envs), "seeds must have exactly one entry per env"
        self.envs = envs
        self.trainer = trainer
        from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
        self._buffers = [RolloutBuffer() for _ in envs]
        # Always resets every env exactly once here, regardless of whether
        # the caller already reset it -- pass `seeds` for reproducible
        # construction (e.g. tests comparing against a controlled reference
        # run) rather than relying on a pre-reset env, which this would
        # otherwise silently discard.
        if seeds is not None:
            self._last_obs = [env.reset(seed=s) for env, s in zip(envs, seeds)]
        else:
            self._last_obs = [env.reset() for env in envs]

    def _trainee_player_and_ai(self, env):
        player = env.match.player_by_id(env.trainee_player_id)
        return player, player.ai

    def collect(
        self, n_steps: int,
        deterministic: bool = False,
        deterministic_decision: bool = False,
        deterministic_direction: bool = False,
    ) -> list[dict]:
        """Collect until at least ``n_steps`` total trainee+secondary steps
        have been recorded across the whole group (mirrors
        ``rollout_worker.py``'s ``_collect(n_steps, ...)`` stopping
        condition, just incrementing by however many envs/secondaries
        contribute a step each round instead of exactly 1). Returns a list
        of ``len(self.envs)`` dicts, each shaped exactly like
        ``rollout_worker.py._collect()``'s return value
        (``buffer``/``last_value``/``stats``) -- a drop-in per-env
        replacement for what one whole rollout_worker.py process returns
        today.

        ``deterministic``/``deterministic_decision``/``deterministic_direction``:
        forwarded to every ``_sample_action_batch`` call this round -- real
        rollout collection always leaves these at the default (``False``,
        matching ``rollout_worker.py``'s own always-stochastic
        ``_sample_action`` usage); set them for a fully reproducible,
        RNG-free comparison (e.g. tests). NOTE: each env's own
        ``env.sample_action_fn`` is NOT consulted here for envs that ARE due
        this round (that indirection only matters for the single-env
        ``act()`` fallback path) -- these three explicit kwargs are the
        actual, single source of truth for how THIS group samples envs it
        batches. The one exception: an env that is NOT yet due this round
        (see below) steps itself completely normally, including its own
        potential ``sample_action_fn`` fallback if it somehow becomes due
        mid-``env.step()`` -- structurally the same as any single-env
        rollout, since this method never touches that env at all this round.

        Not every env is guaranteed to be due for a decision at the start of
        every round: ``ScenarioEnv.step()`` almost always runs a full
        ``decision_interval_ticks``-tick interval (see this module's
        top docstring), but it has at least one legitimate early-exit
        (trainee already in the opponent box with possession -- further
        ticks would only accumulate spurious negative progress) that can
        end an interval several ticks short. When that happens, the
        NEXT round's would-be decision is not actually due yet -- calling
        ``prepare()`` on it would legitimately return ``None`` (exactly the
        cached-gating-reapply behavior the unbatched path already relies
        on). Each round therefore checks ``ai.is_due_for_decision()`` FIRST
        (a pure peek, see its own docstring for why calling ``prepare()``
        itself to check would silently desync the cadence) and only
        includes due envs in this round's batch; a not-due env's
        ``_precomputed_result`` is simply left ``None``, so its own
        ``env.step()`` call falls through to ``act()``'s normal per-tick
        path (which will itself re-check due-ness and correctly just
        re-apply cached gating) -- identical to what the unbatched
        rollout_worker.py path already does for that same tick, so this
        adds no behavior change, just tolerance for an env not being due.
        """
        from footballcoach.ai.ppo.ppo_trainer import _action_to_numpy

        n = len(self.envs)
        stats = [_new_episode_stats() for _ in range(n)]
        episode_reward_accum = [0.0] * n
        secondary_episode_reward_accum = [0.0] * n
        episode_comp_accum = [dict() for _ in range(n)]

        collected = 0
        while collected < n_steps:
            # --- 1. Prepare each DUE env's trainee decision for this round.
            # --- (see docstring above for why not every env is guaranteed
            # to be due, and why is_due_for_decision() is checked first
            # rather than just calling prepare() and branching on None).
            obs_dicts = []
            players_and_ais = []
            for env in self.envs:
                player, ai = self._trainee_player_and_ai(env)
                if not ai.is_due_for_decision():
                    continue
                obs_dict = ai.prepare(player, env.match, 0)
                assert obs_dict is not None, (
                    "is_due_for_decision() said this env was due, but "
                    "prepare() returned None anyway -- the two are now out "
                    "of sync (see is_due_for_decision()'s docstring for the "
                    "exact condition it checks)."
                )
                obs_dicts.append(obs_dict)
                players_and_ais.append((player, ai))

            # --- 2 & 3. ONE batched network call over the due envs only,
            # then stash results (not-due envs' _precomputed_result stays
            # None -- see docstring above). ---
            if obs_dicts:
                obs_batch = _stack_obs_dicts(obs_dicts)
                results = self.trainer._sample_action_batch(
                    obs_batch,
                    deterministic=deterministic,
                    deterministic_decision=deterministic_decision,
                    deterministic_direction=deterministic_direction,
                )
                for (player, ai), result in zip(players_and_ais, results):
                    ai._precomputed_result = result

            # --- 4 & 5. Step every env; per-env bookkeeping (mirrors
            # rollout_worker.py._collect() exactly, once per env). ---
            for i, env in enumerate(self.envs):
                next_obs, reward, done, info = env.step()
                tr = env.last_trainee_transition
                if tr is None:
                    # Structurally shouldn't happen given the precomputed
                    # result is always set before env.step() runs (see
                    # assert above) -- kept for parity/robustness with
                    # rollout_worker.py's identical defensive branch rather
                    # than assuming it can truly never fire.
                    if done:
                        stats[i]["episode_rewards"].append(episode_reward_accum[i])
                        episode_reward_accum[i] = 0.0
                        stats[i]["episode_outcome_labels"].append(
                            info.trial_outcome if (info is not None and info.trial_outcome is not None) else "unknown"
                        )
                        self._last_obs[i] = env.reset()
                    else:
                        self._last_obs[i] = next_obs
                    continue

                bc_label_arr = tr.get("bc_label")
                self._buffers[i].add(
                    obs=tr["obs"],
                    action=_action_to_numpy(tr["action"], tr["raw_exec"]),
                    log_prob=float(tr["log_prob"]),
                    value=float(tr["value"]),
                    reward=reward,
                    done=1.0 if done else 0.0,
                    bc_label=bc_label_arr,
                    head_log_probs=tr.get("head_log_probs"),
                    reward_comps=dict(getattr(env, "last_reward_components", {})),
                    step_outcome=(info.trial_outcome or "") if (done and info is not None) else "",
                )
                collected += 1

                for sec in getattr(env, "last_secondary_results", []):
                    self._buffers[i].add(
                        obs=sec["obs"],
                        action=_action_to_numpy(sec["action"], sec["raw_exec"]),
                        log_prob=sec["log_prob"],
                        value=sec["value"],
                        reward=sec["reward"],
                        done=sec["done"],
                        bc_label=None,
                        weight=self.trainer._secondary_weight,
                        track_id=sec["player_id"],
                    )
                    secondary_episode_reward_accum[i] += sec["reward"]
                    if sec["done"]:
                        stats[i]["secondary_episode_rewards"].append(secondary_episode_reward_accum[i])
                        secondary_episode_reward_accum[i] = 0.0
                    collected += 1

                episode_reward_accum[i] += reward
                for _k, _v in getattr(env, "last_reward_components", {}).items():
                    episode_comp_accum[i][_k] = episode_comp_accum[i].get(_k, 0.0) + _v

                if done:
                    stats[i]["episode_rewards"].append(episode_reward_accum[i])
                    episode_reward_accum[i] = 0.0
                    stats[i]["episode_outcome_labels"].append(
                        info.trial_outcome if (info is not None and info.trial_outcome is not None) else "unknown"
                    )
                    if episode_comp_accum[i]:
                        stats[i]["episode_comp_list"].append(dict(episode_comp_accum[i]))
                    episode_comp_accum[i] = {}
                    if info is not None and info.trial_outcome is not None:
                        if info.is_rules_episode:
                            stats[i]["episode_outcomes_vs_rules"].append(info.trial_outcome)
                        elif info.is_immobile_episode:
                            stats[i]["episode_outcomes_vs_immobile"].append(info.trial_outcome)
                        else:
                            stats[i]["episode_outcomes_vs_neural"].append(info.trial_outcome)
                    if info is not None:
                        stats[i]["episode_durations_s"].append(info.ticks_elapsed * env._dt_s)
                    self._last_obs[i] = env.reset()
                else:
                    self._last_obs[i] = next_obs

        # Bootstrap value per env (never concatenate raw transitions across
        # envs before this -- see _train_parallel()'s identical per-worker
        # discipline).
        out = []
        for i, env in enumerate(self.envs):
            last_value = self.trainer._bootstrap_last_values(env, self._last_obs[i], self._buffers[i])
            out.append({"buffer": self._buffers[i], "last_value": last_value, "stats": stats[i]})
        return out

    def clear_buffers(self) -> None:
        for b in self._buffers:
            b.clear()


def _batched_worker_main(
    conn, phase_id: int, base_seed: int, worker_idx: int, envs_per_process: int,
    separate_value_net: bool = False, worker_torch_threads: int = 1,
) -> None:
    """Entry point run inside each persistent batched-worker process. Must
    stay picklable/top-level (mirrors rollout_worker.py's ``_worker_main``,
    including the critical BLAS-threading env-var ordering)."""
    import os
    _t = str(max(1, worker_torch_threads))
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = _t

    import random

    import torch as _torch

    from footballcoach.ai.curriculum.envs import build_env, bc_label_fn_for_phase_player
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _load_state_dict_tolerant

    _torch.set_num_threads(max(1, worker_torch_threads))

    phase = PHASES_BY_ID[phase_id]
    bc_label_fn = bc_label_fn_for_phase_player(phase_id)

    trainer = PPOTrainer.from_config(
        device=_torch.device("cpu"), inference_only=True, separate_value_net=separate_value_net,
    )
    if phase.frozen_heads:
        trainer.set_frozen_heads(phase.frozen_heads)

    envs = []
    for j in range(envs_per_process):
        seed = base_seed + j
        _torch.manual_seed(seed)
        random.seed(seed)
        env = build_env(phase)
        env.sample_action_fn = trainer._sample_action
        env.bc_label_fn = bc_label_fn
        envs.append(env)

    group = BatchedEnvGroup(envs, trainer)

    while True:
        try:
            msg = conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        cmd = msg.get("cmd")
        if cmd == "collect":
            results = group.collect(msg["n_steps"])
            conn.send(results)
            group.clear_buffers()  # after send: results[i]["buffer"] IS this same object
        elif cmd == "set_weights":
            _load_state_dict_tolerant(trainer.decision_net, msg["decision_net"], "decision_net")
            _load_state_dict_tolerant(trainer.execution_net, msg["execution_net"], "execution_net")
            if msg.get("value_net") is not None and trainer.value_net is not None:
                _load_state_dict_tolerant(trainer.value_net, msg["value_net"], "value_net")
            conn.send({"ok": True})
        elif cmd == "close":
            break
        else:
            log.warning(f"batched_rollout_worker[{worker_idx}]: unknown cmd {cmd!r}")
    conn.close()


@dataclass
class BatchedRolloutWorkerHandle:
    """Main-process handle to one spawned batched-worker process (owns
    ``envs_per_process`` environments internally)."""
    process: "mp.process.BaseProcess"
    conn: "mp.connection.Connection"
    worker_idx: int

    def collect(self, n_steps: int) -> None:
        self.conn.send({"cmd": "collect", "n_steps": n_steps})

    def recv_result(self) -> list:
        return self.conn.recv()

    def set_weights(self, decision_state: dict, execution_state: dict, value_state: Optional[dict]) -> None:
        self.conn.send({
            "cmd": "set_weights",
            "decision_net": decision_state,
            "execution_net": execution_state,
            "value_net": value_state,
        })
        self.conn.recv()  # block until applied, keeps weight sync deterministic

    def close(self) -> None:
        try:
            self.conn.send({"cmd": "close"})
        except (BrokenPipeError, OSError):
            pass
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()


def spawn_batched_workers(
    phase_id: int, n_processes: int, envs_per_process: int, base_seed: int,
    separate_value_net: bool = False, worker_torch_threads: int = 1,
) -> list[BatchedRolloutWorkerHandle]:
    """Spawn ``n_processes`` batched-worker processes, each internally
    owning ``envs_per_process`` environments (total envs =
    ``n_processes * envs_per_process``). ``envs_per_process=1`` reduces to
    exactly today's ``rollout_worker.py`` topology (for empirical A/B
    comparison, at the same ``n_processes``); ``n_processes=1`` is the
    "everything in one process" extreme this module was built to test;
    anything in between is a tunable hybrid.
    """
    ctx = mp.get_context("spawn")
    handles: list[BatchedRolloutWorkerHandle] = []
    for i in range(n_processes):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_batched_worker_main,
            args=(
                child_conn, phase_id, base_seed + i * envs_per_process, i, envs_per_process,
                separate_value_net, worker_torch_threads,
            ),
            daemon=True,
        )
        proc.start()
        handles.append(BatchedRolloutWorkerHandle(process=proc, conn=parent_conn, worker_idx=i))
    return handles


def close_batched_workers(handles: list[BatchedRolloutWorkerHandle]) -> None:
    for h in handles:
        h.close()

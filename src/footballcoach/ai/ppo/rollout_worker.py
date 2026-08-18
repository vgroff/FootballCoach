"""Subprocess worker for parallel PPO rollout collection.

Enabled via ``ppo.n_parallel_envs > 1`` in ai_config.json (see
ai_trainer_knowledge.md "Parallel rollout collection"). Each worker owns a
full, independent copy of the environment AND the policy networks — there is
no shared memory / batched inference across workers, so the existing
single-env sampling code in ``PPOTrainer._sample_action`` runs completely
unmodified inside each worker process. This sidesteps the fact that action
sampling happens synchronously deep inside ``Match.step()`` (via
``NeuralPlayerAI.act()``), which would otherwise block any scheme that tries
to batch inference across environments.

Physics stepping is pure-Python/CPU-bound, so real OS processes (not
threads) are required to get parallel speedup past the GIL.

Each worker is driven by ``RolloutWorkerHandle`` (constructed in the main
process) over a ``multiprocessing.Pipe``, using this simple protocol:

    -> {"cmd": "collect", "n_steps": int, "progress": float}
    <- {"buffer": RolloutBuffer, "stats": {...}}

    -> {"cmd": "set_weights", "decision_net": state_dict, "execution_net": state_dict,
        "value_net": state_dict | None}
    <- {"ok": True}

    -> {"cmd": "close"}
    (worker process exits, no reply)
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("footballcoach.ai.ppo.rollout_worker")


def _worker_main(
    conn, phase_id: int, seed: int, worker_idx: int, separate_value_net: bool = False,
    worker_torch_threads: int = 1, progress_value=None,
) -> None:
    """Entry point run inside each worker process. Must stay picklable/top-level."""
    # Must happen BEFORE numpy (or anything importing it, e.g. torch) is
    # first imported in this fresh spawned process -- numpy's underlying BLAS
    # (OpenBLAS here) reads these env vars at import/first-use to size its
    # OWN internal thread pool, entirely independent of torch.set_num_threads()
    # below. Left unset, OpenBLAS defaults to one thread per LOGICAL core
    # (16 on a typical 8c/16t desktop) PER WORKER PROCESS -- with
    # n_parallel_envs=6 that's up to 96 BLAS threads fighting over 16 cores,
    # which is why steps/s per worker gets WORSE as n_parallel_envs increases
    # past a small number, even though torch itself was already correctly
    # capped. The physics engine and observation encoder are numpy-heavy
    # (not torch), so this is not a hypothetical concern -- it is the
    # dominant oversubscription source at higher worker counts.
    import os
    _t = str(max(1, worker_torch_threads))
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = _t

    import random

    import torch

    from footballcoach.ai.curriculum.envs import build_env, bc_label_fn_for_phase
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
    from footballcoach.ai.progress import ProgressReporter

    torch.set_num_threads(max(1, worker_torch_threads))
    torch.manual_seed(seed)
    random.seed(seed)

    phase = PHASES_BY_ID[phase_id]
    env = build_env(phase)
    bc_label_fn = bc_label_fn_for_phase(phase_id)

    # inference_only=True: no optimizer construction needed, this process
    # never runs a gradient step — the main process owns training. Must match
    # the main process's separate_value_net setting, otherwise trainer.value_net
    # is None here and set_weights' value_net sync below silently no-ops (see
    # ai_trainer_knowledge.md "stored value != fresh value at collection time").
    trainer = PPOTrainer.from_config(
        device=torch.device("cpu"), inference_only=True, separate_value_net=separate_value_net,
    )
    if phase.frozen_heads:
        trainer.set_frozen_heads(phase.frozen_heads)
    env.sample_action_fn = trainer._sample_action

    obs = env.reset()
    buffer = RolloutBuffer()

    def _collect(n_steps: int, progress: float) -> dict:
        nonlocal obs
        episode_rewards: list[float] = []
        episode_reward_accum = 0.0
        secondary_episode_rewards: list[float] = []
        secondary_episode_reward_accum = 0.0
        episode_outcomes_vs_rules: list[str] = []
        episode_outcomes_vs_neural: list[str] = []
        episode_outcomes_vs_immobile: list[str] = []
        # Combined outcome per episode, in the SAME order as episode_rewards
        # below (unlike the vs_rules/vs_neural/vs_immobile split above, which
        # only appends conditionally and so can't be zipped back against
        # episode_rewards index-for-index) -- lets the caller pair each
        # episode's total reward with its outcome regardless of opponent type.
        episode_outcome_labels: list[str] = []
        episode_comp_accum: dict[str, float] = {}
        episode_comp_list: list[dict[str, float]] = []
        episode_durations_s: list[float] = []
        # live=False: several worker processes share one inherited terminal,
        # so a live \r-updating bar per worker would stomp on the others' --
        # coarse milestone lines are safe since each is newline-terminated.
        # When progress_value is given (a shared ctx.Value, see
        # spawn_workers()), the caller polls it to render ONE aggregate live
        # bar across all workers instead, so this reporter's per-step
        # rendering is skipped entirely -- only its FINAL .finish() summary
        # line still prints (see the bottom of this function).
        progress_reporter = ProgressReporter(n_steps, prefix=f"[worker {worker_idx}] ", live=False)
        if progress_value is not None:
            with progress_value.get_lock():
                progress_value.value = 0

        collected = 0
        while collected < n_steps:
            next_obs, reward, done, info = env.step()
            tr = env.last_trainee_transition
            if tr is None:
                if done:
                    episode_rewards.append(episode_reward_accum)
                    episode_reward_accum = 0.0
                    episode_outcome_labels.append(
                        info.trial_outcome if (info is not None and info.trial_outcome is not None) else "unknown"
                    )
                    obs = env.reset()
                else:
                    obs = next_obs
                continue

            from footballcoach.ai.ppo.ppo_trainer import _action_to_numpy

            bc_label_arr = None
            if bc_label_fn is not None:
                bc_label_arr = bc_label_fn(env).to_array()

            buffer.add(
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
                buffer.add(
                    obs=sec["obs"],
                    action=_action_to_numpy(sec["action"], sec["raw_exec"]),
                    log_prob=sec["log_prob"],
                    value=sec["value"],
                    reward=sec["reward"],
                    done=sec["done"],
                    bc_label=None,
                    weight=trainer._secondary_weight,
                )
                secondary_episode_reward_accum += sec["reward"]
                if sec["done"]:
                    secondary_episode_rewards.append(secondary_episode_reward_accum)
                    secondary_episode_reward_accum = 0.0
                collected += 1

            episode_reward_accum += reward
            for _k, _v in getattr(env, "last_reward_components", {}).items():
                episode_comp_accum[_k] = episode_comp_accum.get(_k, 0.0) + _v

            if done:
                episode_rewards.append(episode_reward_accum)
                episode_reward_accum = 0.0
                episode_outcome_labels.append(
                    info.trial_outcome if (info is not None and info.trial_outcome is not None) else "unknown"
                )
                if episode_comp_accum:
                    episode_comp_list.append(dict(episode_comp_accum))
                episode_comp_accum = {}
                if info is not None and info.trial_outcome is not None:
                    if info.is_rules_episode:
                        episode_outcomes_vs_rules.append(info.trial_outcome)
                    elif info.is_immobile_episode:
                        episode_outcomes_vs_immobile.append(info.trial_outcome)
                    else:
                        episode_outcomes_vs_neural.append(info.trial_outcome)
                if info is not None:
                    episode_durations_s.append(info.ticks_elapsed * env._dt_s)
                obs = env.reset()
            else:
                obs = next_obs
            if progress_value is not None:
                with progress_value.get_lock():
                    progress_value.value += 1
            else:
                progress_reporter.update(collected)
        progress_reporter.finish(collected, n_episodes=len(episode_outcome_labels))

        # Bootstrap value for the state AFTER the last stored step, same as
        # the single-process path — required for correct per-worker GAE.
        with torch.no_grad():
            last_obs_dict = {k: v.unsqueeze(0) for k, v in obs.to_torch_dict().items()}
            last_value = trainer._get_value(last_obs_dict)

        return {
            "buffer": buffer,
            "last_value": last_value,
            "stats": {
                "episode_rewards": episode_rewards,
                "episode_outcome_labels": episode_outcome_labels,
                "secondary_episode_rewards": secondary_episode_rewards,
                "episode_outcomes_vs_rules": episode_outcomes_vs_rules,
                "episode_outcomes_vs_neural": episode_outcomes_vs_neural,
                "episode_outcomes_vs_immobile": episode_outcomes_vs_immobile,
                "episode_comp_list": episode_comp_list,
                "episode_durations_s": episode_durations_s,
            },
        }

    while True:
        try:
            msg = conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        cmd = msg.get("cmd")
        if cmd == "collect":
            result = _collect(msg["n_steps"], msg["progress"])
            conn.send(result)
            buffer.clear()  # after send: result["buffer"] IS this same object
        elif cmd == "set_weights":
            trainer.decision_net.load_state_dict(msg["decision_net"])
            trainer.execution_net.load_state_dict(msg["execution_net"])
            if msg.get("value_net") is not None and trainer.value_net is not None:
                trainer.value_net.load_state_dict(msg["value_net"])
            conn.send({"ok": True})
        elif cmd == "close":
            break
        else:
            log.warning(f"rollout_worker[{worker_idx}]: unknown cmd {cmd!r}")
    conn.close()


@dataclass
class RolloutWorkerHandle:
    """Main-process handle to one spawned worker (process + pipe)."""
    process: "mp.process.BaseProcess"
    conn: "mp.connection.Connection"
    worker_idx: int

    def collect(self, n_steps: int, progress: float) -> None:
        self.conn.send({"cmd": "collect", "n_steps": n_steps, "progress": progress})

    def recv_result(self) -> dict:
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


def spawn_workers(
    phase_id: int, n_workers: int, base_seed: int, separate_value_net: bool = False,
    worker_torch_threads: int = 1, progress_value=None,
) -> list[RolloutWorkerHandle]:
    """Spawn ``n_workers`` rollout worker processes for curriculum phase ``phase_id``.

    ``worker_torch_threads`` is read HERE (in the parent, via
    ``ppo.worker_torch_threads`` / ``PPOTrainer.worker_torch_threads``) and
    passed down as a plain int, rather than each worker reading
    ``ai_config.json`` itself -- the value must be applied to BLAS env vars
    before numpy/torch are first imported in the fresh child process (see
    ``_worker_main``'s docstring comment), which is earlier than a config
    load would otherwise happen.

    ``progress_value``: optional shared ``ctx.Value("l", 0)`` (created by
    the SAME "spawn" context as this function uses, so it can be inherited
    correctly) that every worker increments once per collected step --
    lets the caller render one aggregate live progress bar across all
    workers instead of each worker printing its own. Must be created and
    passed at PROCESS-CREATION time like this (via ``Process(args=...)``);
    a raw ``Synchronized`` value can't be sent to an already-running worker
    afterwards (unlike a ``multiprocessing.Manager`` proxy). ``None``
    (default) = workers fall back to their own per-worker milestone lines,
    unchanged from before this parameter existed -- the regular PPO
    training loop (``PPOTrainer._train_parallel()``) does not pass this,
    only ``_collect_value_pretrain_rollout()``'s parallel path does.
    """
    ctx = mp.get_context("spawn")
    handles: list[RolloutWorkerHandle] = []
    for i in range(n_workers):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_worker_main,
            args=(child_conn, phase_id, base_seed + i, i, separate_value_net, worker_torch_threads,
                  progress_value),
            daemon=True,
        )
        proc.start()
        handles.append(RolloutWorkerHandle(process=proc, conn=parent_conn, worker_idx=i))
    return handles


def close_workers(handles: list[RolloutWorkerHandle]) -> None:
    for h in handles:
        h.close()

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


def _worker_main(conn, phase_id: int, seed: int, worker_idx: int, separate_value_net: bool = False) -> None:
    """Entry point run inside each worker process. Must stay picklable/top-level."""
    import random

    import torch

    from footballcoach.ai.config import load_ai_config
    from footballcoach.ai.curriculum.envs import build_env, bc_label_fn_for_phase
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer

    cfg = load_ai_config()
    torch.set_num_threads(max(1, int(cfg.get("ppo", {}).get("worker_torch_threads", 1))))
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
        episode_comp_accum: dict[str, float] = {}
        episode_comp_list: list[dict[str, float]] = []
        episode_durations_s: list[float] = []

        collected = 0
        while collected < n_steps:
            next_obs, reward, done, info = env.step()
            tr = env.last_trainee_transition
            if tr is None:
                if done:
                    episode_rewards.append(episode_reward_accum)
                    episode_reward_accum = 0.0
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
) -> list[RolloutWorkerHandle]:
    """Spawn ``n_workers`` rollout worker processes for curriculum phase ``phase_id``."""
    ctx = mp.get_context("spawn")
    handles: list[RolloutWorkerHandle] = []
    for i in range(n_workers):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_worker_main,
            args=(child_conn, phase_id, base_seed + i, i, separate_value_net),
            daemon=True,
        )
        proc.start()
        handles.append(RolloutWorkerHandle(process=proc, conn=parent_conn, worker_idx=i))
    return handles


def close_workers(handles: list[RolloutWorkerHandle]) -> None:
    for h in handles:
        h.close()

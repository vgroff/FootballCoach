"""Persistent subprocess worker pool for parallel seeded evaluation during
PPO training (see ai/ppo/ppo_trainer.py's ``_train_parallel()``).

``run_seeded_evaluation_parallel()`` in seeded_eval.py spawns a fresh
``multiprocessing.Pool`` -- and therefore pays a full torch/footballcoach
re-import in every worker, ~4-8s each on this project's dev machine -- on
EVERY call. That's fine for a one-off caller (evaluate.py CLI, a single ad
hoc eval), but periodic eval during training calls it every rollout cycle
(every ~1-2 minutes over a multi-hour run), so the same import tax gets
paid hundreds of times over one run for no reason. These workers are
spawned ONCE at the start of ``_train_parallel()`` and stay alive for the
whole run instead, mirroring ``ai/ppo/rollout_worker.py``'s persistent
``Process`` + ``Pipe`` pattern exactly (that module already proves this
pattern out for rollout collection) -- only the current policy weights get
pushed to already-running workers each cycle, via ``set_weights``.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("footballcoach.ai.eval.eval_worker")


def _eval_worker_main(conn, separate_value_net: bool, worker_torch_threads: int = 1) -> None:
    """Entry point run inside each persistent eval worker process. Must
    stay picklable/top-level (mirrors rollout_worker.py's _worker_main)."""
    # Must happen BEFORE numpy/torch are first imported in this fresh
    # spawned process -- see rollout_worker.py's _worker_main for the full
    # explanation (OpenBLAS oversubscription otherwise).
    import os
    _t = str(max(1, worker_torch_threads))
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = _t

    import torch

    from footballcoach.ai.eval.seeded_eval import run_seeded_evaluation
    from footballcoach.ai.ppo.ppo_trainer import _build_eval_env_factory, rebuild_inference_trainer

    torch.set_num_threads(max(1, worker_torch_threads))

    trainer = None  # rebuilt on the first set_weights message
    while True:
        msg = conn.recv()
        cmd = msg["cmd"]
        if cmd == "close":
            return
        elif cmd == "set_weights":
            trainer = rebuild_inference_trainer(
                msg["decision_net"], msg["execution_net"], separate_value_net, msg["value_net"],
            )
            conn.send({"ok": True})
        elif cmd == "eval":
            assert trainer is not None, "eval requested before any set_weights"
            env_factory = _build_eval_env_factory(msg["use_rules_ai"], msg["max_episode_s"])
            result = run_seeded_evaluation(
                env_factory, trainer._sample_action, msg["seeds"], msg["repeats_per_seed"],
                msg.get("win_outcome", "box_possession"),
            )
            conn.send(result)
        else:
            raise ValueError(f"unknown eval worker command: {cmd!r}")


@dataclass
class EvalWorkerHandle:
    process: mp.Process
    conn: object
    worker_idx: int

    def set_weights(self, decision_state: dict, execution_state: dict, value_state: Optional[dict]) -> None:
        self.conn.send({
            "cmd": "set_weights",
            "decision_net": decision_state,
            "execution_net": execution_state,
            "value_net": value_state,
        })
        self.conn.recv()  # block until applied, keeps weight sync deterministic

    def eval(self, seed_chunk: list[int], repeats_per_seed: int, use_rules_ai: bool,
             max_episode_s: float, win_outcome: str = "box_possession") -> None:
        """Fire-and-forget: dispatch the eval, collect the result separately
        via recv_result() once ALL workers have been dispatched (lets
        workers run in parallel instead of one at a time)."""
        self.conn.send({
            "cmd": "eval",
            "seeds": seed_chunk,
            "repeats_per_seed": repeats_per_seed,
            "use_rules_ai": use_rules_ai,
            "max_episode_s": max_episode_s,
            "win_outcome": win_outcome,
        })

    def recv_result(self):
        return self.conn.recv()

    def close(self) -> None:
        try:
            self.conn.send({"cmd": "close"})
        except (BrokenPipeError, OSError):
            pass
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()


def spawn_eval_workers(
    n_workers: int, separate_value_net: bool = False, worker_torch_threads: int = 1,
) -> list[EvalWorkerHandle]:
    """Spawn ``n_workers`` persistent eval worker processes. Call once per
    training run (see ``_train_parallel()``); reuse the returned handles for
    every periodic eval via ``set_weights`` + ``eval`` instead of spawning a
    fresh Pool each time."""
    ctx = mp.get_context("spawn")
    handles: list[EvalWorkerHandle] = []
    for i in range(n_workers):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_eval_worker_main,
            args=(child_conn, separate_value_net, worker_torch_threads),
            daemon=True,
        )
        proc.start()
        handles.append(EvalWorkerHandle(process=proc, conn=parent_conn, worker_idx=i))
    return handles


def close_eval_workers(handles: list[EvalWorkerHandle]) -> None:
    for h in handles:
        h.close()

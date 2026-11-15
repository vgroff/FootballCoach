"""Batched-rollout trace snapshot: seeded in-process BatchedEnvGroup collections
of phase 1 (with episode-seed replay), hashed row by row together with the
returned stats. Two runs of the same code compare exactly, so a refactor of the
rollout workers must give 0 differences.

Usage (from the repo root):
    python scripts/batched_rollout_trace_snapshot.py OUT.json
    python scripts/batched_rollout_trace_snapshot.py --compare A.json B.json
"""
import hashlib
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
torch.set_num_threads(1)


class _CounterSeededRandom(random.Random):
    """Unseeded Random() takes the next counter value instead of OS entropy."""
    _created = 0

    def __init__(self, x=None):
        if x is None:
            _CounterSeededRandom._created += 1
            x = 500_000 + _CounterSeededRandom._created
        super().__init__(x)


random.Random = _CounterSeededRandom


def _h(x) -> str:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes() + str(x.shape).encode()).hexdigest()[:16]
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=repr).encode()).hexdigest()[:16]


def run() -> dict:
    from footballcoach.ai.config import load_ai_config
    from footballcoach.ai.curriculum.envs import build_env, bc_label_fn_for_phase_player
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.ppo.batched_rollout_worker import BatchedEnvGroup
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

    cfg = load_ai_config()
    cfg["network"]["ball_physics_encoder_checkpoint"] = None
    cfg["network"]["player_physics_encoder_checkpoint"] = None
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    trainer = PPOTrainer.from_config(device=torch.device("cpu"), inference_only=True)
    phase = PHASES_BY_ID[1]
    trainer.set_frozen_heads(phase.frozen_heads)
    envs = []
    for _ in range(4):
        env = build_env(phase)
        env.sample_action_fn = trainer._sample_action
        env.bc_label_fn = bc_label_fn_for_phase_player(1)
        envs.append(env)
    record = {}
    for batch_secondary in (False, True):
        random.seed(21)
        torch.manual_seed(21)
        group = BatchedEnvGroup(envs, trainer, seeds=[100, 101, 102, 103],
                                secondary_trainer=trainer if batch_secondary else None)
        rounds = []
        replay = None
        for r in range(3):
            random.seed(31 + r)
            torch.manual_seed(31 + r)
            results = group.collect(2000, replay_seeds=replay)
            out = []
            for res in results:
                buf = res["buffer"]
                adv, ret = buf.compute_gae(0.99, 0.95, res["last_value"])
                tensors = buf.as_tensors(adv, ret)
                out.append({
                    "rows": len(buf.rewards) if hasattr(buf, "rewards") else None,
                    "tensors": {k: _h(v) for k, v in sorted(tensors.items()) if isinstance(v, torch.Tensor)},
                    "stats": {k: _h(v) for k, v in sorted(res["stats"].items())},
                    "seeds": res["stats"].get("episode_seeds", []),
                })
            # Replay the first two recorded seeds of each env next round.
            replay = [s for res in results for s in res["stats"].get("episode_seeds", [])[:2] if s is not None]
            rounds.append(out)
            group.clear_buffers()
        record[f"batch_secondary={batch_secondary}"] = rounds
    return record


def compare(a_path: str, b_path: str) -> int:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    if a == b:
        print("0 differences")
        return 0
    diffs = []

    def walk(x, y, p):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    diffs.append(f"{p}/{k}: missing in one side")
                else:
                    walk(x[k], y[k], f"{p}/{k}")
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{p}[{i}]")
        elif x != y:
            diffs.append(f"{p}: {str(x)[:80]} != {str(y)[:80]}")

    walk(a, b, "")
    print(f"{len(diffs)} differences")
    for d in diffs[:20]:
        print("  ", d)
    return 1


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    rec = run()
    with open(sys.argv[1], "w") as f:
        json.dump(rec, f, sort_keys=True)
    print("wrote", sys.argv[1], {k: [sum(e["rows"] or 0 for e in rnd) for rnd in v] for k, v in rec.items()})

"""Golden snapshot for the ppo_trainer.py refactor: every number, weight hash and
(timing-stripped) log line from a fixed-seed run through the methods being moved.
Usage: golden.py OUT.json   (run from the repo root)"""
import functools
import hashlib
import io
import json
import logging
import os
import random
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
torch.set_num_threads(1)


class _CounterSeededRandom(random.Random):
    """An unseeded Random() normally seeds from OS entropy; here it takes the next
    counter value, so seeds depend only on creation order."""
    _created = 0

    def __init__(self, x=None):
        if x is None:
            _CounterSeededRandom._created += 1
            x = 100_000 + _CounterSeededRandom._created
        super().__init__(x)


random.Random = _CounterSeededRandom

from footballcoach.ai.config import load_ai_config

cfg = load_ai_config()
cfg["ppo"]["mask_untriggered_actions"] = False
cfg["ppo"]["n_epochs"] = 2
cfg["ppo"]["minibatch_size"] = 64
cfg["ppo"]["augment_n_slot_shuffles"] = 0
cfg["ppo"]["value_pretrain_n_processes"] = 1
cfg.setdefault("eval", {})["eval_n_parallel_workers"] = 1
cfg["network"]["ball_physics_encoder_checkpoint"] = None
cfg["network"]["player_physics_encoder_checkpoint"] = None
for k, v in {"ball_mlp_hidden": 16, "global_mlp_hidden": 16, "entity_hidden": 16}.items():
    if k in cfg["network"]:
        cfg["network"][k] = v

out_path = Path(sys.argv[1])
record: dict = {}
log_buffer = io.StringIO()
handler = logging.StreamHandler(log_buffer)
handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
root = logging.getLogger("footballcoach")
root.addHandler(handler)
root.setLevel(logging.INFO)


def seed_all(s):
    torch.manual_seed(s)
    random.seed(s)
    np.random.seed(s)


def tensor_hash(module_or_tensors) -> str:
    h = hashlib.sha256()
    items = module_or_tensors.state_dict().items() if hasattr(module_or_tensors, "state_dict") else module_or_tensors
    for name, t in items:
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def nets_hash(t) -> dict:
    out = {"decision": tensor_hash(t.decision_net), "execution": tensor_hash(t.execution_net)}
    if getattr(t, "value_net", None) is not None:
        out["value"] = tensor_hash(t.value_net)
    return out


def plain(x):
    if isinstance(x, torch.Tensor):
        return [float(v) for v in x.detach().cpu().flatten()[:64]]
    if isinstance(x, np.ndarray):
        return [float(v) for v in x.flatten()[:64]]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0])) if "time" not in str(k)}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x[:64]]
    if isinstance(x, float):
        return repr(x)
    if isinstance(x, (int, str, bool)) or x is None:
        return x
    return type(x).__name__


def stage(name):
    def deco(fn):
        seed_all(1234)
        try:
            record[name] = plain(fn())
        except Exception as e:  # recorded, so a refactor that changes failure behaviour shows up too
            record[name] = f"EXCEPTION {type(e).__name__}: {e}"
        return fn
    return deco


from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
import tests.ai_scenario.test_batched_rollout as B

seed_all(1234)
trainer = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=True)
if PHASES_BY_ID[1].frozen_heads:
    trainer.set_frozen_heads(PHASES_BY_ID[1].frozen_heads)
record["init_weights"] = nets_hash(trainer)


def env():
    e = B._make_envs(trainer, 1, max_episode_s=3.0)[0]
    e.sample_action_fn = trainer._sample_action
    return e


@stage("sampling")
def _():
    e = env()
    e.reset()
    obs = e._get_obs()
    obs_dict = obs.to_torch_dict()
    out = {
        "stochastic": trainer._sample_action(obs_dict),
        "deterministic": trainer._sample_action(obs_dict, deterministic=True, deterministic_decision=True, deterministic_direction=True),
    }
    batch = {k: torch.stack([v, v, v]) for k, v in obs_dict.items()}
    out["batch"] = trainer._sample_action_batch(batch)
    return out


@stage("rollout_update")
def _():
    e = env()
    batch, stats = trainer._collect_value_pretrain_rollout(e, 300, phase_id=None, use_gae=True)
    metrics = trainer._ppo_update(batch, progress=0.1)
    return {"batch_returns": batch["returns"], "batch_log_probs": batch["log_probs"], "stats": stats,
            "metrics": metrics, "weights": nets_hash(trainer)}


@stage("log_summary")
def _():
    e = env()
    batch, _ = trainer._collect_value_pretrain_rollout(e, 200, phase_id=None, use_gae=True)
    metrics = trainer._ppo_update(batch, progress=0.2)
    trainer._log_rollout_summary(
        metrics=metrics, steps_per_sec=1000.0, episode_rewards=[0.5, -0.2, 1.0, 0.0], secondary_episode_rewards=[0.1],
        episode_outcomes_vs_rules=["box_possession", "timeout"], episode_outcomes_vs_immobile=["box_possession"],
        episode_outcomes_vs_neural=[], rollout_components={"approach": 0.1, "progress": 0.2},
        episode_comp_list=[{"approach": 0.1, "progress": 0.2}], episode_durations_s=[2.0, 3.0],
        comp_step_stats={}, n_reward_comp_steps=10, episode_kick_counts=[0, 1, 2, 5],
        episode_armed_kick_counts=[0, 1, 0, 2],
    )
    return nets_hash(trainer)


@stage("checkpoint_roundtrip")
def _():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ck.pt"
        trainer._save_checkpoint_to(path)
        other = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=True)
        if PHASES_BY_ID[1].frozen_heads:
            other.set_frozen_heads(PHASES_BY_ID[1].frozen_heads)
        step = other.load_checkpoint(path)
        keys = sorted(torch.load(path, map_location="cpu", weights_only=False).keys())
        return {"step": step, "keys": keys, "loaded": nets_hash(other), "same": nets_hash(other) == nets_hash(trainer)}


@stage("seeded_eval")
def _():
    from footballcoach.ai.eval.seeded_eval import run_seeded_evaluation
    try:
        from footballcoach.ai.ppo.trainer_eval import _build_eval_env_factory
    except ImportError:
        from footballcoach.ai.ppo.trainer_eval import _build_eval_env_factory
    det = functools.partial(trainer._sample_action, deterministic=True, deterministic_decision=True, deterministic_direction=True)
    r = run_seeded_evaluation(_build_eval_env_factory(use_rules_ai=True, max_episode_s=4.0), det, [7, 8, 9], repeats_per_seed=1)
    return {"rewards": r.rewards, "outcomes": r.outcomes_list, "lengths": r.episode_lengths_s}


@stage("eval_vs_rules")
def _():
    trainer._eval_vs_rules(max_episode_s=4.0)
    return "ran"


@stage("pretrain_value")
def _():
    out = trainer.pretrain_value(env(), n_steps=200, n_epochs=1, lr=1e-4, batch_size=64)
    return {"out": out, "weights": nets_hash(trainer)}


@stage("ppg_value_refit")
def _():
    out = trainer.ppg_value_refit(env(), n_steps=200, epochs=1, batch_size=64, num_rollouts=1)
    return {"out": out, "weights": nets_hash(trainer)}


@stage("consistency_refit")
def _():
    out = trainer.consistency_refit(env(), n_steps=200, epochs=1, num_rollouts=1, val_episode_fraction=0.3, lr=1e-3, batch_size=64, kappa_deg=22.0)
    return {"out": out, "weights": nets_hash(trainer)}


TIMING = re.compile(r"(\d+(\.\d+)?\s*(s|ms|steps/s|/s|it/s|sec))\b|speed=\S+|elapsed\S*|took \S+|\d+\.\d+s\b|eta \S+", re.I)
lines = []
for line in log_buffer.getvalue().splitlines():
    if "progress" in line.lower() or "%" in line and "steps/s" in line:
        continue
    line = re.sub(r"\S*tmp[A-Za-z0-9_]{6,}\S*", "<tmp>", line)
    lines.append(TIMING.sub("<t>", line))
record["log_lines"] = lines
record["log_line_count"] = len(lines)
out_path.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
print("stages:", {k: ("EXC" if isinstance(v, str) and v.startswith("EXCEPTION") else "ok") for k, v in record.items() if k not in ("log_lines",)})
print("log lines:", len(lines))

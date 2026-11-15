"""Env-trace snapshot for ScenarioEnv refactors: every public per-step output
(reward, done, StepInfo, reward components, secondary results, player/ball
state, observation hash) from fixed-seed episodes over several opponent
regimes, written as JSON so two runs can be compared exactly.

Usage (from the repo root):
    python scripts/env_trace_snapshot.py OUT.json
    python scripts/env_trace_snapshot.py --compare A.json B.json

Covers self-play (neural secondary), rules and immobile opponents, with
always_compute_secondary_reward both on and off, and BC labelling on (as in
training: the BC sandbox snapshots/restores player callbacks and match.rng).
Private env attributes are deliberately not recorded, so a refactor that
renames or removes internal state still compares cleanly.
"""
import dataclasses
import functools
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
            x = 200_000 + _CounterSeededRandom._created
        super().__init__(x)


random.Random = _CounterSeededRandom

_CONFIGS = {
    "self_play": dict(rules=0.0, immobile=0.0, always=False),
    "self_play_always": dict(rules=0.0, immobile=0.0, always=True),
    "rules": dict(rules=1.0, immobile=0.0, always=False),
    "rules_always": dict(rules=1.0, immobile=0.0, always=True),
    "immobile_always": dict(rules=0.0, immobile=1.0, always=True),
}
_STEPS = 250
_MAX_EPISODE_S = 8.0


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [repr(float(v)) for v in x.ravel()]
    if isinstance(x, torch.Tensor):
        return _jsonable(x.detach().cpu().numpy())
    if isinstance(x, (np.floating, float)):
        return repr(float(x))
    if isinstance(x, (np.integer, int, bool, str)) or x is None:
        return x if not isinstance(x, np.integer) else int(x)
    if dataclasses.is_dataclass(x):
        return _jsonable(dataclasses.asdict(x))
    return repr(x)


def _hash(x) -> str:
    return hashlib.sha256(json.dumps(_jsonable(x), sort_keys=True).encode()).hexdigest()[:16]


def _player_state(match):
    out = {}
    for p in sorted(match.players, key=lambda q: q.player_id):
        out[p.player_id] = [repr(p.position.x), repr(p.position.y), repr(p.velocity.x), repr(p.velocity.y),
                            repr(p.stamina), p.kick_count, p.tackle_fire_count]
    b = match.ball
    out["ball"] = [repr(b.position.x), repr(b.position.y), repr(b.position.z), b.possessed_by]
    return out


def run() -> dict:
    from footballcoach.ai.env.scenario_env import ScenarioEnv
    from footballcoach.ai.curriculum.envs import bc_label_fn_for_phase_player
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    trainer = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=True)
    record = {}
    for ci, (name, c) in enumerate(_CONFIGS.items()):
        torch.manual_seed(1000 + ci)
        defn = ScenarioDefinition(
            key=f"trace_{name}", label=name, description=name,
            build=functools.partial(build_1v1_scenario, opponent_rules_prob=c["rules"],
                                    opponent_immobile_prob=c["immobile"]),
        )
        env = ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=1,
                          secondary_player_ids=["opponent"], max_episode_s=_MAX_EPISODE_S,
                          rng=random.Random(500 + ci))
        env.always_compute_secondary_reward = c["always"]
        env.sample_action_fn = trainer._sample_action
        env.bc_label_fn = bc_label_fn_for_phase_player(1)
        seed = 9000 + 100 * ci
        obs = env.reset(seed=seed)
        steps = [{"reset_obs": _hash(obs.to_torch_dict())}]
        for _ in range(_STEPS):
            obs, reward, done, info = env.step()
            match = env.last_terminal_match if done else env._loop.match
            steps.append({
                "reward": repr(float(reward)), "done": done, "info": _jsonable(info),
                "components": _jsonable(env.last_reward_components),
                "secondary": [
                    {k: _jsonable(v) for k, v in r.items()
                     if k in ("player_id", "reward", "done", "step_outcome", "reward_components",
                              "log_prob", "value", "illegal_action")}
                    | {"rest": _hash({k: v for k, v in r.items() if k not in ("reward_components",)})}
                    for r in env.last_secondary_results
                ],
                "trainee_transition": None if env.last_trainee_transition is None
                else _hash(env.last_trainee_transition),
                "state": _player_state(match),
                "obs": _hash(obs.to_torch_dict()),
            })
            if done:
                seed += 1
                obs = env.reset(seed=seed)
                steps.append({"reset_obs": _hash(obs.to_torch_dict())})
        record[name] = steps
        n_done = sum(1 for s in steps if s.get("done"))
        n_sec = sum(len(s.get("secondary", [])) for s in steps)
        print(f"{name}: {len(steps)} records, {n_done} episodes ended, {n_sec} secondary rows")
    return record


def compare(a_path: str, b_path: str) -> int:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    diffs = []

    def walk(x, y, p):
        if type(x) is not type(y):
            diffs.append((p, x, y))
        elif isinstance(x, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    diffs.append((f"{p}/{k}", "missing in one side"))
                else:
                    walk(x[k], y[k], f"{p}/{k}")
        elif isinstance(x, list):
            if len(x) != len(y):
                diffs.append((p, f"len {len(x)} vs {len(y)}"))
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{p}[{i}]")
        elif x != y:
            diffs.append((p, x, y))

    walk(a, b, "")
    print(f"{len(diffs)} differences")
    for d in diffs[:20]:
        print("  ", d)
    return 1 if diffs else 0


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    out = sys.argv[1]
    rec = run()
    with open(out, "w") as f:
        json.dump(rec, f, sort_keys=True)
    print("wrote", out)

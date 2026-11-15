"""PPO-update trace snapshot for refactors of PPOUpdateMixin._ppo_update.

Runs ``_ppo_update`` from a fixed seed under a set of configuration variants
chosen to reach every branch of the method (separate/shared value net, physics
encoders on/off, action masking, augmentation + flip consistency, held-out
split + epoch-0 baseline, KL early stop with value-only and BC-only
continuations, the high-KL diagnostic block, shared value gradient, kappa
regularisers, LR warmup, a batch missing the optional keys). For each variant it
records the returned metrics, hashes of every network and optimizer state, the
RNG states after the call, and the full timing-stripped DEBUG log. Two runs of
the same code compare exactly, so a pure restructuring must give 0 differences.

Usage (from the repo root):
    python scripts/ppo_update_trace_snapshot.py OUT.json
    python scripts/ppo_update_trace_snapshot.py --compare A.json B.json

Set PPO_TRACE_ONLY=name1,name2 to run only some variants (e.g. for quick
mutation checks; compare such a file only against one made the same way).

Each variant also lists which log tags it produced ("coverage"), so it is
visible which branches a "0 differences" result actually exercised.
"""
import copy
import hashlib
import io
import json
import logging
import os
import random
import re
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
            x = 400_000 + _CounterSeededRandom._created
        super().__init__(x)


random.Random = _CounterSeededRandom

_NO_PHYSICS = {"network": {"ball_physics_encoder_checkpoint": None, "player_physics_encoder_checkpoint": None}}

# name -> (config overrides applied before construction, trainer attribute
#          overrides, schedule overrides {name: constant}, batch keys to drop,
#          separate_value_net, n_updates)
_VARIANTS = {
    "base": (_NO_PHYSICS, {}, {}, [], True, 2),
    "physics_on": ({}, {}, {}, [], True, 1),
    "shared_value": (_NO_PHYSICS, {}, {}, [], False, 1),
    "plain": (_NO_PHYSICS, {"mask_untriggered_actions": False, "log_action_opportunity_stats": False,
                            "dual_clip_c": 0.0, "lr_warmup_steps": 0}, {}, [], True, 1),
    "aug_val_baseline": (_NO_PHYSICS, {"augment_n_slot_shuffles": 1, "_log_flip_consistency": True,
                                       "val_episode_fraction": 0.3, "log_epoch_zero_baseline": True}, {}, [], True, 1),
    "early_stop_bc": (_NO_PHYSICS, {"target_kl": -1.0, "value_only_continuation_epochs": 2,
                                    "bc_only_continuation_epochs": 2, "bc_only_continuation_coeff": None,
                                    "minibatch_size": 64},
                      {"bc": 0.3, "bc_kick": 0.2, "bc_tackle": 0.1}, [], True, 1),
    "early_stop_shared_bc_coeff": (_NO_PHYSICS, {"target_kl": -1.0, "value_only_continuation_epochs": 1,
                                                 "bc_only_continuation_epochs": 1, "bc_only_continuation_coeff": 0.5,
                                                 "minibatch_size": 64},
                                   {"bc": 0.3}, [], False, 1),
    "kl_diag_regs": (_NO_PHYSICS, {"target_kl": 1e9, "move_dir_log_kappa_reg_coef": 0.01,
                                   "kick_dir_log_kappa_reg_coef": 0.02, "kick_dir_z_log_std_reg_coef": 0.03,
                                   "lr_warmup_steps": 3},
                     {"lr": 0.05}, [], True, 1),
    "kl_diag_shared": (_NO_PHYSICS, {"target_kl": 1e9}, {"lr": 0.05}, [], False, 1),
    "share_value_grad": (_NO_PHYSICS, {"share_value_grad_with_decision": True, "decision_value_coef": 0.3,
                                       "minibatch_size": 64}, {}, [], True, 1),
    "minimal_batch": (_NO_PHYSICS, {"mask_untriggered_actions": False, "log_action_opportunity_stats": False},
                      {}, ["head_log_probs", "step_outcomes", "reward_comps_raw", "dones"], True, 1),
    "minimal_batch_kl_diag": (_NO_PHYSICS, {"mask_untriggered_actions": False, "log_action_opportunity_stats": False,
                                            "target_kl": 1e9},
                              {"lr": 0.05}, ["head_log_probs", "step_outcomes", "reward_comps_raw"], True, 1),
}
# Value dropout consumes the torch RNG inside forward passes, so these two catch any
# reordering of the network calls (not just changed arithmetic).
_DROPOUT = {"network": {**_NO_PHYSICS["network"], "value_dropout": 0.1}}
_VARIANTS["value_dropout"] = (_DROPOUT, {"minibatch_size": 64}, {}, [], True, 1)
_VARIANTS["value_dropout_shared_early_stop"] = (_DROPOUT, {"minibatch_size": 64, "target_kl": -1.0,
                                                           "value_only_continuation_epochs": 1},
                                                {"lr": 0.05}, [], False, 1)
_INFO_ONLY = {"base"}   # also run at INFO level, since DEBUG changes which blocks execute

_COVERAGE_TAGS = [
    "[DIAG e0 mb0]", "[early stop", "[value-only continuation]", "[bc-only continuation]",
    "[bc-only continuation grad clip]", "ratio percentiles", "[per-head new lp means", "[top-",
    "[held-out policy]", "epoch 0 (pre-training baseline", "[ratio spike]", "[per-head KL]",
    "[per-head policy_loss", "[value RMSE by outcome]", "[advantage |.| by episode", "[policy heads]",
    "[direction param delta]", "[exec continuous", "[grad norm]",
]

TIMING = re.compile(r"(\d+(\.\d+)?\s*(s|ms|steps/s|/s|it/s|sec))\b|speed=\S+|elapsed\S*|took \S+|\d+\.\d+s\b|eta \S+", re.I)


def _seed_all(s):
    torch.manual_seed(s)
    random.seed(s)
    np.random.seed(s)


def _hash_bytes(*chunks) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c if isinstance(c, bytes) else str(c).encode())
    return h.hexdigest()[:16]


def _module_hash(m) -> str:
    return _hash_bytes(*(name.encode() + t.detach().cpu().contiguous().numpy().tobytes()
                         for name, t in m.state_dict().items()))


def _optim_hash(opt) -> str:
    if opt is None:
        return None
    sd = opt.state_dict()
    chunks = [json.dumps(sd["param_groups"], sort_keys=True, default=str)]
    for k in sorted(sd["state"]):
        for name, v in sorted(sd["state"][k].items()):
            chunks.append(f"{k}/{name}")
            chunks.append(v.detach().cpu().contiguous().numpy().tobytes() if isinstance(v, torch.Tensor) else repr(v))
    return _hash_bytes(*chunks)


def _plain(x):
    if isinstance(x, torch.Tensor):
        return [repr(float(v)) for v in x.detach().cpu().flatten()]
    if isinstance(x, np.ndarray):
        return [repr(float(v)) for v in x.flatten()]
    if isinstance(x, (np.floating, float)):
        return repr(float(x))
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0])) if "time" not in str(k)}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (int, str, bool)) or x is None:
        return x
    return type(x).__name__


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict):
            _deep_update(dst.setdefault(k, {}), v)
        else:
            dst[k] = v


def _run_variant(name, cfg_over, attr_over, sched_over, drop_keys, separate, n_updates, level):
    from footballcoach.ai.config import load_ai_config
    from footballcoach.ai.ppo.bc import _I_VALID
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    import tests.ai_scenario.test_batched_rollout as B

    cfg = load_ai_config()
    saved = copy.deepcopy(cfg)
    try:
        _deep_update(cfg, {"ppo": {"n_epochs": 3}})
        _deep_update(cfg, cfg_over)
        _seed_all(1234)
        trainer = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=separate)
    finally:
        cfg.clear()
        cfg.update(saved)
    if PHASES_BY_ID[1].frozen_heads:
        trainer.set_frozen_heads(PHASES_BY_ID[1].frozen_heads)
    for k, v in attr_over.items():
        assert hasattr(trainer, k), f"{name}: trainer has no attribute {k}"
        setattr(trainer, k, v)
    for k, v in sched_over.items():
        assert hasattr(trainer.schedules, k), f"{name}: schedules has no {k}"
        setattr(trainer.schedules, k, lambda progress, _v=v: _v)

    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    root = logging.getLogger("footballcoach")
    old_level = root.level
    root.addHandler(handler)
    out = {"updates": []}
    try:
        for u in range(n_updates):
            _seed_all(2000 + u)
            env = B._make_envs(trainer, 1, max_episode_s=3.0)[0]
            env.sample_action_fn = trainer._sample_action
            batch, _ = trainer._collect_value_pretrain_rollout(env, 200, phase_id=None, use_gae=True)
            if sched_over.get("bc", 0.0) > 0.0:
                # This collector stores no valid BC labels, so the BC paths would only ever see
                # all-invalid rows; deterministic synthetic labels marked valid reach them instead.
                g = torch.Generator().manual_seed(77 + u)
                labels = torch.rand(batch["bc_labels"].shape, generator=g)
                labels[:, _I_VALID] = 1.0
                batch["bc_labels"] = labels
            for k in drop_keys:
                batch.pop(k, None)
            _seed_all(3000 + u)
            root.setLevel(level)
            log_buffer.seek(0)
            log_buffer.truncate()
            try:
                metrics = _plain(trainer._ppo_update(batch, progress=0.1 + 0.3 * u))
            except Exception as e:  # recorded, so a change in failure behaviour shows up too
                metrics = f"EXCEPTION {type(e).__name__}: {e}"
            root.setLevel(old_level)
            lines = [TIMING.sub("<t>", ln) for ln in log_buffer.getvalue().splitlines()]
            rec = {
                "metrics": metrics,
                "decision": _module_hash(trainer.decision_net),
                "execution": _module_hash(trainer.execution_net),
                "value": _module_hash(trainer.value_net) if getattr(trainer, "value_net", None) is not None else None,
                "optimizer": _optim_hash(trainer.optimizer),
                "value_optimizer": _optim_hash(getattr(trainer, "value_net_optimizer", None)),
                "rng_torch": _hash_bytes(torch.get_rng_state().numpy().tobytes()),
                "rng_random": _hash_bytes(repr(random.getstate())),
                "rng_numpy": _hash_bytes(repr(np.random.get_state())),
                "rng_aug": _hash_bytes(repr(trainer._aug_rng.bit_generator.state)
                                       if hasattr(trainer._aug_rng, "bit_generator")
                                       else repr(trainer._aug_rng.getstate())
                                       if hasattr(trainer._aug_rng, "getstate") else type(trainer._aug_rng).__name__),
                "grad_norm_history": [repr(float(v)) for v in trainer._grad_norm_history],
                "log": lines,
                "coverage": sorted({t for t in _COVERAGE_TAGS if any(t in ln for ln in lines)}),
            }
            out["updates"].append(rec)
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    return out


def run() -> dict:
    record = {}
    only = [v for v in os.environ.get("PPO_TRACE_ONLY", "").split(",") if v]
    for name, spec in _VARIANTS.items():
        if only and name not in only:
            continue
        levels = [("debug", logging.DEBUG)] + ([("info", logging.INFO)] if name in _INFO_ONLY else [])
        for lname, level in levels:
            key = f"{name}@{lname}"
            record[key] = _run_variant(name, *spec, level)
            ups = record[key]["updates"]
            exc = [u["metrics"] for u in ups if isinstance(u["metrics"], str)]
            print(f"{key:34s} log_lines={[len(u['log']) for u in ups]} "
                  f"coverage={ups[-1]['coverage']}" + (f" EXC={exc}" if exc else ""))
    return record


def compare(a_path: str, b_path: str) -> int:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    diffs = []

    def walk(x, y, p):
        if type(x) is not type(y):
            diffs.append((p, str(x)[:200], str(y)[:200]))
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
            diffs.append((p, str(x)[:200], str(y)[:200]))

    walk(a, b, "")
    print(f"{len(diffs)} differences")
    for d in diffs[:30]:
        print("  ", d)
    return 1 if diffs else 0


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    rec = run()
    with open(sys.argv[1], "w") as f:
        json.dump(rec, f, sort_keys=True)
    print("wrote", sys.argv[1])

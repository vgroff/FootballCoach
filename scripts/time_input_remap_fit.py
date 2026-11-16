"""Choose and check the weight remap for the time-remaining input fix.

Until the fix, NeuralPlayerAI computed time remaining as
max_episode_s - ticks / 30 while training ticks are observation.sim_dt_s
(0.04 s), so the training rows' clock ran at 0.83x and one tick late: at the
true end of an episode it still read about max_episode_s / 6 (~4.5 s). The
fix (2026-11-16) encodes the true seconds
left; ai/ppo/time_input_remap.py folds f_old = a + b * f_new into a
checkpoint's weights (f = log1p(seconds) / log1p(time_remaining_norm_max_s),
which every network reads linearly). f_old also depends on each episode's
length and the compression is a curve, so no (a, b) is exact; this script

1. plays seeded episodes of every phase with the checkpoint and keeps each
   training row's state (asserting the recorded time feature, now the true
   clock, is reproduced exactly from the env's tick count; the old value at
   the same instant is computed from the old formula);
2. fits candidate (a, b) by weighted least squares over every tick of every
   episode length each phase can draw, with a chosen share of the weight on
   the last END_S seconds (the end of the episode, where the old clock and
   the true one differ most), plus the plain fit over the visited rows for
   reference;
3. evaluates each candidate COUNTERFACTUALLY on those real states: each state
   is put at a set of points on the clock (true time left 0.5 s ... the whole
   episode), and the original checkpoint on the OLD clock value at that point
   is compared with the remapped checkpoint on the TRUE value. Natural rows
   near the end of an episode are rare (~1 in 40), so this is what measures
   the end of the episode; it holds the state fixed, so it measures the remap
   alone, not a change in which states occur.

Usage (repo root):
    python scripts/time_input_remap_fit.py CHECKPOINT [--episodes N] [--phases 1,2] [--states N]
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import statistics
from pathlib import Path

import torch

from footballcoach.ai.config import load_ai_config
from footballcoach.ai.curriculum.envs import build_env, build_role_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
from footballcoach.ai.ppo.time_input_remap import TIME_REMAINING_COL, remap_time_remaining_input
from footballcoach.ai.ppo.trainer_common import _ai_types

NORM = math.log1p(float(load_ai_config()["observation"]["time_remaining_norm_max_s"]))
BINS = (("<2s", 0.0, 2.0), ("2-4s", 2.0, 4.0), ("4-8s", 4.0, 8.0), ("8-16s", 8.0, 16.0), (">=16s", 16.0, 1e9))
CLOCK_POINTS_S = (0.5, 1.5, 3.0, 6.0, 10.0, 14.0, 20.0, 28.0)
HEADS = ("move_logit", "gp_logit", "kick_logit", "sprint_logit")


def _env(pid: int):
    ph = PHASES_BY_ID[pid]
    if not ph.is_role_based:
        return build_env(ph)
    return build_role_env(ph, controller_weights={r: {"neural": 1} if r == "attacker" else {"rules": 1}
                                                  for r in set(ph.roles.values())})


def f_old_at(horizon: float, ticks: int) -> float:
    """The old feature at env tick count ``ticks`` (NeuralPlayerAI counts the
    current tick before encoding, and counted ticks as 1/30 s)."""
    return math.log1p(max(0.0, horizon - (ticks + 1) / 30.0)) / NORM


def f_new_at(horizon: float, ticks: int, dt: float) -> float:
    return math.log1p(max(0.0, horizon - ticks * dt)) / NORM


def collect(trainer, phases, episodes):
    """Rows (phase, obs, f_old, f_new, horizon, dt) of the trainee's real decisions."""
    rows = []
    for pid in phases:
        env = _env(pid)
        env.sample_action_fn = trainer._sample_action
        for s in range(episodes):
            env.reset(seed=s)
            done = False
            while not done:
                ticks, horizon = env._episode_ticks, env.max_episode_s
                _, _, done, _ = env.step()
                tr = env.last_trainee_transition
                if tr is None:
                    continue
                obs = {k: torch.as_tensor(v).clone() for k, v in tr["obs"].items()}
                f_rec = float(obs["global_feat"][TIME_REMAINING_COL])
                f_new = f_new_at(horizon, ticks, env._dt_s)
                if abs(f_new - f_rec) > 1e-5:
                    raise AssertionError(f"time reconstruction off: {f_new} vs recorded {f_rec}")
                rows.append((pid, obs, f_old_at(horizon, ticks), f_new, horizon, env._dt_s))
    return rows


def outputs(trainer, obs, f):
    d = {k: v.unsqueeze(0).clone() for k, v in obs.items()}
    d["global_feat"][..., TIME_REMAINING_COL] = f
    sf, of, em, bf, gf = d["self_feat"], d["other_feat"], d["exists_mask"], d["ball_feat"], d["global_feat"]
    sat, oat = _ai_types(d)
    with torch.no_grad():
        dh = trainer.decision_net(sf, of, em, bf, gf, sat, oat)
        eh = trainer.execution_net(sf, of, em, bf, gf, dh, sat, oat)
        return {"move_logit": float(dh.move_logit), "gp_logit": float(dh.get_possession_raw),
                "kick_logit": float(eh.kick_logit), "sprint_logit": float(eh.sprint_logit),
                "value": float(trainer._get_value(d))}


def grid_pairs(phases, end_s: float, end_share: float):
    """(f_old, f_new, weight) at every tick of 11 episode lengths per phase
    (max_episode_s + [0, variability]); phases weighted equally, and within a
    phase ``end_share`` of the weight on ticks with under ``end_s`` left."""
    out = []
    for pid in phases:
        env = _env(pid)
        base, var, dt = env._base_max_episode_s, env._max_episode_s_variability_s, env._dt_s
        pairs = []
        for j in range(11):
            horizon = base + var * j / 10
            for ticks in range(int(horizon / dt)):
                pairs.append((f_old_at(horizon, ticks), f_new_at(horizon, ticks, dt), horizon - ticks * dt < end_s))
        n_end = sum(p[2] for p in pairs)
        n_rest = len(pairs) - n_end
        for fo, fn, is_end in pairs:
            w = (end_share / n_end) if is_end else ((1.0 - end_share) / n_rest)
            out.append((fo, fn, w))
    return out


def fit(pairs):
    fo = torch.tensor([p[0] for p in pairs], dtype=torch.float64)
    fn = torch.tensor([p[1] for p in pairs], dtype=torch.float64)
    w = torch.tensor([p[2] for p in pairs], dtype=torch.float64).sqrt()
    sol = torch.linalg.lstsq(torch.stack([w, w * fn], 1), (w * fo).unsqueeze(1)).solution
    return tuple(float(x) for x in sol.flatten())


def counterfactual(trainer, candidates, states):
    """{candidate: {bin: {head: [|dp|...], "value": [|dv|...]}}} over each
    state placed at every CLOCK_POINTS_S point within its episode length."""
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))
    res = {name: {b[0]: {k: [] for k in (*HEADS, "value")} for b in BINS} for name in candidates}
    for _pid, obs, _fo, _fn, horizon, dt in states:
        for t_left in CLOCK_POINTS_S:
            if t_left >= horizon:
                continue
            ticks = round((horizon - t_left) / dt)
            fo, fn = f_old_at(horizon, ticks), f_new_at(horizon, ticks, dt)
            base = outputs(trainer, obs, fo)
            label = next(b[0] for b in BINS if b[1] <= horizon - ticks * dt < b[2])
            for name, net in candidates.items():
                new = outputs(net, obs, fn)
                for k in HEADS:
                    res[name][label][k].append(abs(sig(new[k]) - sig(base[k])))
                res[name][label]["value"].append(abs(new["value"] - base["value"]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--phases", default="1,2")
    ap.add_argument("--states", type=int, default=300, help="real states used for the counterfactual evaluation")
    args = ap.parse_args()
    phases = [int(p) for p in args.phases.split(",")]
    trainer = PPOTrainer.from_config(device=torch.device("cpu"))
    trainer.load_checkpoint(Path(args.checkpoint))
    rows = collect(trainer, phases, args.episodes)
    print(f"rows {len(rows)} " + " ".join(f"phase{p}={sum(r[0] == p for r in rows)}" for p in phases))

    fits = {"visited rows": fit([(r[2], r[3], 1.0) for r in rows]),
            # end_s beyond any episode: every tick is an "end" tick, i.e. even weighting
            "clock, even": fit(grid_pairs(phases, 1e9, 1.0))}
    for end_s, share in ((8.0, 0.5), (8.0, 0.75), (4.0, 0.5)):
        fits[f"clock, {int(share * 100)}% on last {int(end_s)}s"] = fit(grid_pairs(phases, end_s, share))
    candidates = {"no remap": trainer}
    for name, (a, b) in fits.items():
        net = copy.deepcopy(trainer)
        remap_time_remaining_input(net, a=a, b=b)
        candidates[name] = net
        print(f"{name:<22} f_old = {a:.6f} + {b:.6f} * f_new")
    rng = random.Random(0)
    states = rng.sample(rows, min(args.states, len(rows)))
    res = counterfactual(trainer, candidates, states)
    print(f"\nchange vs the original checkpoint on its old clock, by true time left "
          f"({len(states)} real states x {len(CLOCK_POINTS_S)} clock points; probability changes, mean/max)")
    for name in candidates:
        print(f"-- {name}")
        for label, _lo, _hi in BINS:
            r = res[name][label]
            if not r["value"]:
                continue
            parts = [f"{k[:-6]} {statistics.mean(r[k]):.4f}/{max(r[k]):.3f}" for k in HEADS]
            parts.append(f"value {statistics.mean(r['value']):.3f}/{max(r['value']):.3f}")
            print(f"   {label:>6}  " + "  ".join(parts))


if __name__ == "__main__":
    main()

"""Rescale the move/kick direction heads' final Linear layers so their raw
(pre-normalize) output has median norm ~1. Policy-preserving: every consumer
uses the L2-normalized vector (atan2 / normalize), kappa is a separate
parameter. Writes a NEW checkpoint without optimizer state (Adam moments of
the rescaled weights would be stale), then verifies on real observations that
deterministic actions, values and log-probs are unchanged.
usage: rescale_direction_heads.py in.pt out.pt
"""
import sys, random, copy, numpy as np, torch
from footballcoach.ai.curriculum.envs import build_env, build_role_env
from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
from footballcoach.ai.ppo.distill_pretrain import _student_forward
src, dst = sys.argv[1], sys.argv[2]
torch.set_num_threads(2)
tr = PPOTrainer.load_for_inference(src)


def rows_from(env_builder, n_eps, seed0):
    out = []
    for k in range(n_eps):
        env = env_builder()
        env.sample_action_fn = tr._sample_action
        env.rng = random.Random(k)
        env.reset(seed=seed0 + k)
        done = False
        while not done:
            _, _, done, _ = env.step()
            for t in [env.last_trainee_transition] + list(getattr(env, "last_secondary_results", [])):
                if t is not None and t.get("obs") is not None:
                    out.append(t["obs"])
    return out


ctrl = {"attacker": "neural", "defender": "neural", "goalkeeper": "neural"}
def p2():
    e = build_role_env(PHASES_BY_ID[2], controller_weights={r: {c: 1} for r, c in ctrl.items()})
    e.controller_overrides = ctrl
    e.scenario_kwargs = {**e.scenario_kwargs, "no_keeper_prob": 0.0}
    return e
rows = rows_from(p2, 12, 7000) + rows_from(lambda: build_env(PHASES_BY_ID[1]), 12, 8000)
obs = {k: torch.as_tensor(np.stack([np.asarray(r[k]) for r in rows])) for k in rows[0]}
with torch.no_grad():
    d0, e0 = _student_forward(tr, obs)
mn = e0.move_direction_unnormalized.norm(dim=-1)
kn = e0.kick_direction_unnormalized.norm(dim=-1)
s_move, s_kick = float(mn.median()), float(kn.median())
print(f"{len(rows)} rows: raw norm median move {s_move:.3f} (p10 {mn.quantile(.1):.2f}, p90 {mn.quantile(.9):.2f}) "
      f"kick {s_kick:.3f} (p10 {kn.quantile(.1):.2f}, p90 {kn.quantile(.9):.2f})")

ck = torch.load(src, map_location="cpu", weights_only=False)
ex = ck["execution_net"]
for name, s in (("move_direction", s_move), ("kick_direction", s_kick)):
    ex[f"{name}.weight"] = ex[f"{name}.weight"] / s
    ex[f"{name}.bias"] = ex[f"{name}.bias"] / s
ck.pop("optimizer", None)
ck["_direction_heads_rescaled"] = {"move": s_move, "kick": s_kick, "from": src}
torch.save(ck, dst)

tr2 = PPOTrainer.load_for_inference(dst)
with torch.no_grad():
    d1, e1 = _student_forward(tr2, obs)
n1 = e1.move_direction_unnormalized.norm(dim=-1)
print(f"after: raw norm median move {n1.median():.3f}, kick {e1.kick_direction_unnormalized.norm(dim=-1).median():.3f}")
print("max |normalized move dir diff|", float((e1.move_direction - e0.move_direction).abs().max()),
      " kick", float((e1.kick_direction - e0.kick_direction).abs().max()),
      " value", float((e1.value - e0.value).abs().max()))
# deterministic and seeded-stochastic actions identical on 300 rows
bad = 0
for k in range(min(300, len(rows))):
    o = {kk: v[k] for kk, v in obs.items()}
    a = tr._sample_action(o, deterministic=True); b = tr2._sample_action(o, deterministic=True)
    torch.manual_seed(k); c = tr._sample_action(o); torch.manual_seed(k); d = tr2._sample_action(o)
    for x, y in ((a, b), (c, d)):
        if abs(float(x[1]) - float(y[1])) > 1e-4 or not np.allclose(np.asarray(x[4]["move_direction"]), np.asarray(y[4]["move_direction"]), atol=1e-5):
            bad += 1
print(f"action/log-prob mismatches over {min(300, len(rows))} rows x 2 modes: {bad}")

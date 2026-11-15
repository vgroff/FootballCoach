"""Play a role-based curriculum phase (default: phase 2, 1v2) with every role on
its rules AI, through the real training env (curriculum/envs.py build_env ->
RoleScenarioEnv), and report the outcome mix and each role's mean terminal
reward. Seeded, so a run is reproducible.

Usage (from the repo root):
    python scripts/phase2_rules_playthrough.py [--phase 2] [--episodes 300] [--seed 0]
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.getcwd())


def run(phase_id: int, episodes: int, seed: int) -> dict:
    from footballcoach.ai.curriculum.envs import build_role_env
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID

    phase = PHASES_BY_ID[phase_id]
    all_rules = {role: {"rules": 1} for role in set(phase.roles.values())}
    env = build_role_env(phase, controller_weights=all_rules)
    outcomes: collections.Counter = collections.Counter()
    reward_sums: dict[str, float] = collections.defaultdict(float)
    durations: list[float] = []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done, info = False, None
        while not done:
            _, _, done, info = env.step()
        outcomes[info.trial_outcome] += 1
        durations.append(info.ticks_elapsed * env._dt_s)
        for pid, r in env.episode_rewards_by_player(info.trial_outcome).items():
            reward_sums[phase.roles[pid]] += r
    return {
        "phase": phase.name,
        "episodes": episodes,
        "outcomes": {k: outcomes.get(k, 0) for k in env.outcome_keys},
        "mean_reward_by_role": {role: reward_sums[role] / episodes for role in sorted(reward_sums)},
        "mean_duration_s": sum(durations) / len(durations),
        "max_duration_s": max(durations),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = run(args.phase, args.episodes, args.seed)
    print(f"{res['phase']}: {res['episodes']} all-rules episodes, "
          f"mean {res['mean_duration_s']:.1f}s (max {res['max_duration_s']:.1f}s)")
    for k, v in res["outcomes"].items():
        print(f"  {k:20s} {v:5d}  {100.0 * v / res['episodes']:5.1f}%")
    print("  mean terminal reward: " + "  ".join(f"{r}={v:+.3f}" for r, v in res["mean_reward_by_role"].items()))


if __name__ == "__main__":
    main()

"""
Diagnostic: what episode returns does the rules-based AI get on the Phase 1 scenario?

Drives the trainee with BC labels (the rules-based move-toward-ball / push-toward-goal
logic) directly as orders, then scores every episode to get the return distribution.

Run: uv run python debug_rulesai_score.py
"""
import functools
import numpy as np
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.bc import phase1_labels
from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition

N_EPISODES = 500

# Match training params exactly: slower ball, longer timeout
defn = ScenarioDefinition(
    key="dbg", label="dbg", description="dbg",
    build=functools.partial(build_1v1_scenario, ball_max_speed_mps=4.0),
)
env = ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=1, max_episode_s=240.0)

episode_returns = []
episode_lengths = []

obs = env.reset()
ep_return = 0.0
ep_len = 0

while len(episode_returns) < N_EPISODES:
    # Get rules-based label, then build a fake action dict that applies the
    # label's move_direction/sprint/etc. exactly as if the network had output
    # them, driving the env through the normal action interface.
    label = phase1_labels(env)

    import numpy as _np
    move_dir = label.move_direction if label.move_direction is not None else _np.array([1.0, 0.0])
    env_action = {
        "decision_probs": {
            "shoot": 0.0,
            "pass_": 0.0,
            "move": float(label.move),
            "tackle": 0.0,
            "get_possession": float(label.get_possession_extra),
            "mark": 0.0,
            "hold_position": 0.0,
        },
        "execution_physical": {
            "move_direction": move_dir,
            "sprint": bool(label.sprint > 0.5),
            "kick_this_tick": False,
            "kick_direction": _np.array([1.0, 0.0]),
            "kick_power_fraction": 0.5,
            "kick_spin": _np.zeros(3),
            "tackle_attempt": False,
        },
        "decision_physical": {
            "move_region_center_m": _np.zeros(2),
            "move_region_size_m": 2.0,
            "move_arrival_speed_mps": 5.0,
        },
        "target_slots": {"pass_": 0, "tackle": 0, "mark": 0},
        "slot_player_ids": [None] * 21,
        "decision": None,
        "execution": None,
    }

    next_obs, reward, done, info = env.step(env_action)
    ep_return += reward
    ep_len += 1

    if done:
        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        ep_return = 0.0
        ep_len = 0
        obs = env.reset()
    else:
        obs = next_obs

returns = np.array(episode_returns)
lengths = np.array(episode_lengths)

print(f"\n=== Rules-based AI on Phase 1 ({N_EPISODES} episodes) ===")
print(f"Episode return:  mean={returns.mean():.2f}  std={returns.std():.2f}")
print(f"                 min={returns.min():.2f}  max={returns.max():.2f}")
print(f"                 p10={np.percentile(returns,10):.2f}  p50={np.percentile(returns,50):.2f}  p90={np.percentile(returns,90):.2f}")
print(f"Episode length:  mean={lengths.mean():.1f} steps  (max_ep=240 steps @ 0.5s each)")
print(f"Reached box:     {(returns >= 5.0).sum()}/{N_EPISODES} = {(returns >= 5.0).mean()*100:.1f}%  (terminal bonus=+5)")
print(f"Top 10%:         return >= {np.percentile(returns,90):.2f}")
print(f"Top 1%:          return >= {np.percentile(returns,99):.2f}")

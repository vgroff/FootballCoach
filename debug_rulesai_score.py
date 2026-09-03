"""
Diagnostic: what episode returns does the rules-based AI get on the Phase 1 scenario?

Drives the trainee with Phase1RulesAI directly (assigned as the player's own
.ai, not via ScenarioEnv's sample_action_fn/NeuralPlayerAI machinery -- that
path is for a neural network's action dict, which rules-AI driving doesn't
use), then scores every episode to get the return distribution.

Run: uv run python debug_rulesai_score.py
"""
import numpy as np
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.rules_ai import Phase1RulesAI
from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition

N_EPISODES = 500


def _real_phase1_max_episode_s() -> float:
    """The actual phase-1 episode timeout (ai_config.json["curriculum"]
    ["phase1_max_episode_s"]) -- see curriculum/phases.py's
    PHASE_1_GET_POSSESSION.env_kwargs, what real training actually passes to
    ScenarioEnv. Matches outcome_baseline.py's own helper of the same name."""
    from footballcoach.ai.curriculum.phases import PHASE_1_GET_POSSESSION
    return float(PHASE_1_GET_POSSESSION.env_kwargs["max_episode_s"])


# ball_max_speed_mps is deliberately NOT passed here -- omitting it lets
# build_1v1_scenario fall back to its own config-driven default, which reads
# ai_config.json's phase1_scenario.ball_max_speed_mps directly, so this
# script always tracks whatever that config value currently is instead of
# hardcoding a number that can silently go stale.
#
# HISTORY (2026-09-03): this value used to silently differ across the
# codebase -- curriculum/envs.py (real PPO/BC training), replay_episode.py,
# and evaluate.py's --baseline-only path each hardcoded their OWN literal
# (10.0, 10.0, and 4.0 respectively) instead of reading
# phase1_scenario.ball_max_speed_mps, so the config value had no effect on
# any of them regardless of what it was set to -- the same "two disconnected
# sources of truth" shape as this session's earlier max-spin finding. All
# four call sites (this script included) were fixed together to read the
# config value instead of hardcoding a literal; see
# agent_plans/physics_update.md for context if that file still references
# this, and git blame on curriculum/envs.py for the actual fix commit.
def _build(*args, **kwargs):
    match = build_1v1_scenario(*args, **kwargs)
    match.player_by_id("trainee").ai = Phase1RulesAI()
    return match


defn = ScenarioDefinition(
    key="dbg", label="dbg", description="dbg",
    build=_build,
)
# sample_action_fn is left at its default (None) deliberately -- ScenarioEnv
# only overrides the trainee's .ai with a NeuralPlayerAI when a sampling
# function is supplied (see ScenarioEnv.reset()); leaving it None means the
# Phase1RulesAI assigned in _build() above stays in place and drives the
# trainee for real, and env.step() (no action argument -- that API was
# removed) advances the match through it directly.
env = ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=1, max_episode_s=_real_phase1_max_episode_s())

episode_returns = []
episode_lengths = []
episode_reached_box = []

obs = env.reset()
ep_return = 0.0
ep_len = 0
ep_box = False

while len(episode_returns) < N_EPISODES:
    next_obs, reward, done, info = env.step()
    ep_return += reward
    ep_len += 1
    if env.last_reward_components.get("box", 0.0) > 0.0:
        ep_box = True

    if done:
        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        episode_reached_box.append(ep_box)
        ep_return = 0.0
        ep_len = 0
        ep_box = False
        obs = env.reset()
    else:
        obs = next_obs

returns = np.array(episode_returns)
lengths = np.array(episode_lengths)
reached_box = np.array(episode_reached_box)

print(f"\n=== Rules-based AI on Phase 1 ({N_EPISODES} episodes) ===")
print(f"Episode return:  mean={returns.mean():.2f}  std={returns.std():.2f}")
print(f"                 min={returns.min():.2f}  max={returns.max():.2f}")
print(f"                 p10={np.percentile(returns,10):.2f}  p50={np.percentile(returns,50):.2f}  p90={np.percentile(returns,90):.2f}")
print(f"Episode length:  mean={lengths.mean():.1f} steps  (max_ep={env._max_episode_ticks} steps, max_episode_s={_real_phase1_max_episode_s()})")
# Checked directly against the reward.py "box" component (box_possession_terminal
# fires) rather than a hardcoded return threshold -- an earlier version of this
# script assumed the terminal bonus was +5 (it's actually box_possession_terminal
# alone, 3.0 in the live config, plus a variable speed/stamina component on top),
# so a `returns >= 5.0` check silently undercounted real box-reaching episodes.
print(f"Reached box:     {reached_box.sum()}/{N_EPISODES} = {reached_box.mean()*100:.1f}%")
print(f"Top 10%:         return >= {np.percentile(returns,90):.2f}")
print(f"Top 1%:          return >= {np.percentile(returns,99):.2f}")

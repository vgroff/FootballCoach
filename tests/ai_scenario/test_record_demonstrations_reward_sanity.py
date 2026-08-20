"""End-to-end sanity check on record_demonstrations.py's recorded rewards.

This is the test that SHOULD have existed before today: every other reward
test (tests/ai_unit/test_reward.py) only unit-tests ``phase1_reward()`` in
isolation, one call at a time -- none of them exercise the actual recording
pipeline (``record_episodes()``) end-to-end and check that a REAL recorded
episode's reward stays within a sane bound. That gap is exactly why two
separate bugs (a row/reward misattribution when a kick/tackle callback fired
mid-``env.step()``, and ``_pending_reward`` leaking across episode
boundaries) both shipped into real recorded data before being caught by a
human staring at aggregate statistics rather than by CI.

The bound here is deliberately generous (nowhere close to tight
per-component reward-shaping values) -- this is a coarse "did something
put an astronomically wrong number in the data" tripwire, not a precise
reward-design check. It uses Phase1RulesAI on both sides (fast, no
checkpoint needed, and both sides kick/tackle -- push-kicks from
Phase1RulesAI's own box-approach logic and tackles from GetPossessionOrder
-- so it exercises the on_kick/on_tackle-callback-mid-step() interleaving
this bug class lives in, unlike a neural-driven trainee which barely kicks
at all).
"""
from __future__ import annotations

import numpy as np

from footballcoach.ai.curriculum.envs import bc_label_fn_for_phase, build_env
from footballcoach.ai.curriculum.phases import CurriculumPhase
from footballcoach.ai.scripts.record_demonstrations import record_episodes

# Generous ceiling derived from ai_config.json's phase-1 reward terms
# (box_possession_terminal=2.0, speed_bonus_scale=4.0,
# gain_possession_bonus=1.0 -- can fire multiple times per episode on
# repeated possession changes, loss_of_possession_penalty=-0.9,
# ball_out_penalty=-4.0, stamina_penalty small) -- real clean episode
# totals observed in practice top out around +12 to +15. 50 is comfortably
# above any legitimate episode, and three orders of magnitude below the
# corruption this test guards against (real corrupted data showed totals
# in the hundreds to +1479).
PER_EPISODE_TOTAL_BOUND = 50.0
# A single tick's reward is the sum of AT MOST one of each component firing
# together (box + speed + poss + stamina, roughly) -- never more than ~10.
PER_ROW_REWARD_BOUND = 10.0


def test_recorded_episode_rewards_stay_within_sane_bounds():
    phase = CurriculumPhase(
        name="p1", phase_id=1, scenario_key="phase1_1v1", env_kwargs={"max_episode_s": 15.0}
    )
    env = build_env(phase)
    label_fn = bc_label_fn_for_phase(1)
    env.always_compute_secondary_reward = True

    result = record_episodes(
        env, label_fn, n_episodes=300, scenario_key="phase1_1v1", phase_id=1,
        sample_interval_s=0.2,
        opponent_rules_prob=1.0, opponent_immobile_prob=0.0,
    )

    rewards = result["rewards"]
    dones = result["dones"]
    is_trainee = result["is_trainee"]

    bad_rows = np.nonzero(np.abs(rewards) > PER_ROW_REWARD_BOUND)[0]
    assert len(bad_rows) == 0, (
        f"{len(bad_rows)} row(s) have a per-tick reward exceeding "
        f"+/-{PER_ROW_REWARD_BOUND} (e.g. row {bad_rows[0]}: {rewards[bad_rows[0]]}) -- "
        f"no single tick's reward should ever be this large; this is the exact signature "
        f"of a corrupted/misattributed row."
    )

    # Per-player, per-episode undiscounted totals (gamma=1 sum), computed
    # directly from the raw arrays -- deliberately NOT using
    # DemonstrationDataset.compute_returns() here, so this test doesn't
    # silently stop catching a recording-side bug if a future dataset-side
    # bug happened to also mask it.
    n = len(rewards)
    running = {0.0: 0.0, 1.0: 0.0}  # keyed by is_trainee value
    prev_had_done = False
    worst = 0.0
    for i in range(n - 1, -1, -1):
        is_done = dones[i] > 0.5
        if is_done and not prev_had_done:
            running[0.0] = 0.0
            running[1.0] = 0.0
        key = 1.0 if is_trainee[i] > 0.5 else 0.0
        running[key] += float(rewards[i])
        worst = max(worst, abs(running[key]))
        prev_had_done = is_done

    assert worst <= PER_EPISODE_TOTAL_BOUND, (
        f"a player's per-episode undiscounted reward total reached {worst:.3f}, "
        f"exceeding the sane ceiling of {PER_EPISODE_TOTAL_BOUND} -- real reward "
        f"components (box_possession_terminal=2.0, speed_bonus_scale=4.0, ...) cannot "
        f"legitimately sum this high; this indicates a recording-pipeline bug "
        f"(row misattribution or a leaked/uncleared per-player reward accumulator)."
    )

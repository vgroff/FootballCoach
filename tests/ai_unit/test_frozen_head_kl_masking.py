"""Regression test: frozen decision heads must be masked in the per-head
log_prob array stored at sample time (``_sample_action``'s ``head_log_probs``),
matching the masking already applied by ``_per_head_new_log_probs()`` at PPO
update time.

Before the fix, ``head_log_probs`` stored the raw (unmasked) Bernoulli
log_prob for every frozen head, while ``_per_head_new_log_probs()`` zeroed
those same entries when recomputing the "after" side of the per-head KL
diagnostic. This produced a spurious per-head KL (= the raw sampled
log_prob, not a real KL divergence) for every frozen head -- e.g.
`shoot=-0.14 pass_=-0.12 tackle=-0.12` in training logs, always negative,
even though these heads receive no gradient and should show ~0 KL.
"""
from __future__ import annotations

from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
from footballcoach.ai.ppo.rollout_buffer import HEAD_LP_KEYS
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario

_FROZEN = [
    "shoot_logit", "pass_logit", "tackle_logit",
    "get_possession_raw", "mark_logit", "hold_position_logit",
]
# Maps decision-net attribute name -> HEAD_LP_KEYS entry it controls.
_FROZEN_LP_KEYS = ["shoot", "pass_", "tackle", "gp_extra", "mark", "hold"]


def _make_env() -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="frozen_head_kl_1v1",
        label="Frozen head KL masking test",
        description="Regression test for per-head KL masking of frozen heads",
        build=build_1v1_scenario,
    )
    return ScenarioEnv(
        definition=defn,
        trainee_player_id="trainee",
        phase=1,
        max_episode_s=30.0,
    )


def test_head_log_probs_zero_for_frozen_heads():
    trainer = PPOTrainer.from_config()
    trainer.set_frozen_heads(_FROZEN)

    env = _make_env()
    env.sample_action_fn = trainer._sample_action
    obs = env.reset()

    result = trainer._sample_action(obs.to_torch_dict())
    # _sample_action returns an 8-tuple; head_log_probs is stored in the
    # transition dict built by ScenarioEnv/NeuralPlayerAI, not directly in
    # this tuple -- call the private method's returned pieces instead by
    # inspecting env.last_trainee_transition after a step(), which is what
    # actually gets stored in the rollout buffer.
    next_obs, reward, done, info = env.step()
    tr = env.last_trainee_transition
    assert tr is not None, "No trainee transition recorded -- fixture issue"
    head_log_probs = tr.get("head_log_probs")
    assert head_log_probs is not None, "head_log_probs missing from transition"

    key_to_idx = {k: i for i, k in enumerate(HEAD_LP_KEYS)}
    for key in _FROZEN_LP_KEYS:
        idx = key_to_idx[key]
        assert head_log_probs[idx] == 0.0, (
            f"head_log_probs[{key!r}] = {head_log_probs[idx]} -- expected exactly "
            "0.0 since this head is frozen for the current curriculum phase. "
            "If nonzero, the per-head KL diagnostic will report a spurious "
            "negative 'KL' for this head (see module docstring)."
        )

    # Sanity: non-frozen heads should generally be nonzero (log_prob of an
    # actual Bernoulli sample is virtually never exactly 0.0).
    for key in ("exec_move", "kick", "tackle_attempt"):
        idx = key_to_idx[key]
        assert head_log_probs[idx] != 0.0, (
            f"head_log_probs[{key!r}] is exactly 0.0 -- unexpected for an "
            "unfrozen head, this test's own sanity check may be broken."
        )

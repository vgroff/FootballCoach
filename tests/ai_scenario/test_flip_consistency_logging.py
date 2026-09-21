"""Coverage for the [PPO] `flip` consistency log line (PPOTrainer._flip_consistency_stats and its capture inside
_recompute_old_log_probs_for_augmented_batch). The properties that matter: the id/flip blocks are paired row-for-row,
the capture does not change the recomputed log-probs, and the number moves in the right direction."""
from __future__ import annotations

import functools
import logging
import math
import random

import pytest
import torch

from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.obs.augment import BALL_FLIP_Y_IDX, PLAYER_FLIP_Y_IDX, augment_batch
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario, phase1_training_on_tick

_PHASE = PHASES_BY_ID[1]


def _make_env(trainer) -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="flip_consistency_test", label="flip consistency logging test", description="1v1 short episodes",
        build=functools.partial(build_1v1_scenario, opponent_rules_prob=1.0, opponent_immobile_prob=0.0),
        on_tick=phase1_training_on_tick,
    )
    env = ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=1, max_episode_s=2.0)
    env.sample_action_fn = trainer._sample_action
    return env


@pytest.fixture()
def trainer():
    torch.manual_seed(0)
    # y_canonical (live config) makes the network exactly flip-invariant, i.e. flip KL == 0 by construction; the stat only has
    # something to measure with it off, so pin it rather than depend on the ambient config.
    t = PPOTrainer.from_config(device=torch.device("cpu"), separate_value_net=False, y_canonical=False)
    if _PHASE.frozen_heads:
        t.set_frozen_heads(_PHASE.frozen_heads)
    t.augment_n_slot_shuffles = 1
    return t


def _augmented(trainer, symmetric_obs: bool = False):
    batch, _ = trainer._collect_value_pretrain_rollout(_make_env(trainer), 400, phase_id=None)
    if symmetric_obs:
        # y-signed features all zero -> the flipped copy is identical to the original
        for k, cols in (("obs/self_feat", PLAYER_FLIP_Y_IDX), ("obs/other_feat", PLAYER_FLIP_Y_IDX), ("obs/ball_feat", BALL_FLIP_Y_IDX)):
            batch[k] = batch[k].clone()
            batch[k][..., cols] = 0.0
    n = len(batch["returns"])
    aug = augment_batch(batch, 1, random.Random(0))
    aug.update(trainer._precompute_physics_full(aug, 256))
    return aug, n


def test_capture_pairs_rows_and_does_not_change_the_recomputed_log_probs(trainer):
    aug, n = _augmented(trainer)
    lp_plain, hl_plain = trainer._recompute_old_log_probs_for_augmented_batch(aug)
    pair = {}
    lp_cap, hl_cap = trainer._recompute_old_log_probs_for_augmented_batch(aug, pair_out=pair)
    assert torch.equal(lp_plain, lp_cap) and torch.equal(hl_plain, hl_cap)
    assert len(pair["id"]["move_direction"]) == n and len(pair["flip"]["move_direction"]) == n
    # the flip block really is the flip_y copy: an independent forward pass over just those rows agrees
    from footballcoach.ai.ppo.ppo_trainer import _ai_types
    idx = torch.arange(n, 2 * n)[:64]
    mb = {k.replace("obs/", ""): aug[k][idx] for k in aug if k.startswith("obs/")}
    sat, oat = _ai_types(mb)
    with torch.no_grad():
        d = trainer.decision_net(mb["self_feat"], mb["other_feat"], mb["exists_mask"], mb["ball_feat"], mb["global_feat"], sat, oat,
                                 ball_physics_full=mb.get("ball_physics_full"), self_physics_full=mb.get("self_physics_full"),
                                 other_physics_full=mb.get("other_physics_full"))
        e = trainer.execution_net(mb["self_feat"], mb["other_feat"], mb["exists_mask"], mb["ball_feat"], mb["global_feat"], d, sat, oat)
    assert torch.allclose(pair["flip"]["move_direction"][:64], e.move_direction, atol=1e-5)
    assert torch.allclose(pair["flip"]["exec_move_logit"][:64], e.exec_move_logit, atol=1e-4)


def test_kl_is_positive_on_generic_states_and_bernoulli_kl_is_zero_when_the_flip_is_a_no_op(trainer):
    aug, n = _augmented(trainer)
    pair = {}
    trainer._recompute_old_log_probs_for_augmented_batch(aug, pair_out=pair)
    stats = trainer._flip_consistency_stats(aug, pair)
    assert stats["n"] == n and math.isfinite(stats["kl"]) and stats["kl"] > 0.0

    aug_s, n_s = _augmented(trainer, symmetric_obs=True)
    pair_s = {}
    trainer._recompute_old_log_probs_for_augmented_batch(aug_s, pair_out=pair_s)
    sym = trainer._flip_consistency_stats(aug_s, pair_s)
    for head in ("exec_move", "sprint", "kick", "tackle_attempt", "move", "gp_extra"):
        assert sym["per_head"][head] == pytest.approx(0.0, abs=1e-5), head
    assert sym["metrics"]["exec_move_sign_disagree_pct"] == 0.0


def test_ppo_update_reports_the_stat_only_when_enabled_and_logs_the_line(trainer, caplog):
    def _batch():
        env = _make_env(trainer)
        b, _ = trainer._collect_value_pretrain_rollout(env, 500, phase_id=None, use_gae=True)
        return b

    trainer._log_flip_consistency = False
    assert trainer._ppo_update(_batch(), progress=0.0)["flip_consistency"] is None
    trainer._log_flip_consistency = True
    metrics = trainer._ppo_update(_batch(), progress=0.0)
    fc = metrics["flip_consistency"]
    assert fc is not None and math.isfinite(fc["kl"]) and fc["kl"] > 0.0
    assert set(fc["metrics"]) >= {"move_dir_err_p50_deg", "move_dir_err_p90_deg", "exec_move_sign_disagree_pct"}

    with caplog.at_level(logging.INFO, logger="footballcoach.ai.ppo"):
        trainer._log_rollout_summary(
            metrics=metrics, steps_per_sec=1.0, episode_rewards=[0.0], secondary_episode_rewards=[],
            episode_outcomes_vs_rules=[], episode_outcomes_vs_immobile=[], episode_outcomes_vs_neural=[],
            rollout_components={}, episode_comp_list=[], episode_durations_s=[], comp_step_stats={}, n_reward_comp_steps=0,
        )
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "flip     consistency KL=" in text and "hard-decision disagree%" in text and "state pairs, pre-update weights" in text

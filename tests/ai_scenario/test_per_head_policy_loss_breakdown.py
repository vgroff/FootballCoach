"""Tests for two policy_loss diagnostics added to _ppo_update(), both aimed
at tracking down WHY pol_mean stays positive instead of trending negative:

1. The per-head COUNTERFACTUAL policy_loss breakdown -- the "[per-head
   policy_loss (counterfactual, ...)]" log line. See all_head_policy_loss's
   own comment in ppo_trainer.py for the exact semantics: for each head,
   ratio_h = exp(new_head_lp - old_head_lp) is computed as if every OTHER
   head stayed at ratio=1 (unchanged from the rollout-collection policy),
   then run through the same clipped-surrogate formula as the real scalar
   policy_loss. This is NOT a strict decomposition -- the real per-sample
   ratio is the PRODUCT of every head's own ratio (since log-probs sum,
   ratio = exp(sum) = product of per-head exp(delta)), so the 15 per-head
   numbers do not sum to the scalar policy_loss. It's a counterfactual for
   comparing relative contribution, not a partition of the total.

   Reuses test_frozen_head_kl_masking.py's core fact (head_log_probs/
   _per_head_new_log_probs already correctly zero out frozen heads) rather
   than re-testing that masking itself -- this file only tests the NEW
   breakdown built on top of it.

2. Per-SAMPLE policy_loss percentiles, every epoch -- the "[epoch N/M
   policy_loss percentiles, n=...]" line -- to see the distribution's shape
   (skew, outliers) rather than just pol_mean. A progressive pool across
   that epoch's minibatches (not one clean snapshot, since the network is
   being updated minibatch-by-minibatch through the epoch) -- an
   approximation, same caveat as pol_mean itself, but still useful for
   spotting skew/outliers.
"""
from __future__ import annotations

import math

import pytest
import torch

from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _action_to_numpy
from footballcoach.ai.ppo.rollout_buffer import HEAD_LP_KEYS, RolloutBuffer
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario

_ROLLOUT_STEPS = 40


def _make_env() -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="per_head_policy_loss_test",
        label="Per-head policy_loss breakdown test",
        description="Smoke-test 1v1 environment for the per-head policy_loss diagnostic",
        build=build_1v1_scenario,
    )
    return ScenarioEnv(
        definition=defn, trainee_player_id="trainee", phase=1, max_episode_s=30.0,
    )


def _collect_rollout(env: ScenarioEnv, trainer: PPOTrainer, n: int):
    """Same shape as test_separate_value_net.py's helper, but also stores
    head_log_probs (test_batched_rollout.py's pattern) -- without this the
    stored "old" per-head log-probs would silently fall back to zeros
    (RolloutBuffer.add's default), making the new diagnostic's ratio
    computation degenerate rather than a real before/after comparison."""
    buffer = RolloutBuffer()
    env.sample_action_fn = trainer._sample_action
    env.reset()
    last_obs = None
    for _ in range(n):
        next_obs, reward, done, _info = env.step()
        tr = env.last_trainee_transition
        if tr is not None:
            buffer.add(
                obs=tr["obs"], action=_action_to_numpy(tr["action"], tr["raw_exec"]),
                log_prob=tr["log_prob"], value=tr["value"], reward=reward,
                done=1.0 if done else 0.0, head_log_probs=tr.get("head_log_probs"),
            )
            last_obs = next_obs
        if done:
            env.reset()
    if last_obs is None:
        last_obs = next_obs
    return buffer, last_obs


def _run_ppo_update(trainer: PPOTrainer, buffer: RolloutBuffer, last_obs):
    with torch.no_grad():
        last_obs_dict = {
            k: v.unsqueeze(0).to(trainer.device) for k, v in last_obs.to_torch_dict().items()
        }
        last_value = trainer._get_value(last_obs_dict)
    advantages, returns = buffer.compute_gae(trainer.gamma, trainer.lam, last_value)
    batch = buffer.as_tensors(advantages, returns)
    assert "head_log_probs" in batch
    assert not torch.all(batch["head_log_probs"] == 0.0), (
        "fixture produced all-zero head_log_probs -- the diagnostic would be "
        "exercised degenerately (see _collect_rollout's docstring)"
    )
    return trainer._ppo_update(batch, progress=0.0)


class TestPerHeadPolicyLossBreakdown:
    def test_log_line_lists_every_active_head_with_finite_values(self, caplog):
        """Masked/frozen/inactive heads (PPOTrainer._inactive_head_lp_keys --
        curriculum-frozen decision heads plus kick_spin, which is
        permanently frozen regardless of curriculum phase) are dropped from
        this line entirely rather than shown as a content-free residual
        (ratio=1 collapses the counterfactual policy_loss to
        -mean(this update's advantages), identical across every masked
        head, not a real per-head signal -- see _inactive_head_lp_keys'
        own docstring)."""
        trainer = PPOTrainer.from_config()
        env = _make_env()
        buffer, last_obs = _collect_rollout(env, trainer, _ROLLOUT_STEPS)

        with caplog.at_level("INFO"):
            _run_ppo_update(trainer, buffer, last_obs)

        lines = [
            r.message for r in caplog.records
            if r.message.startswith("  [per-head policy_loss")
        ]
        assert len(lines) == 1, f"expected exactly one summary line, got {len(lines)}"
        line = lines[0]

        inactive = trainer._inactive_head_lp_keys()
        assert inactive, (
            "expected at least kick_spin to be permanently frozen -- if this "
            "changed, _kick_spin_frozen's default changed too, update this test"
        )
        active_keys = [k for k in HEAD_LP_KEYS if k not in inactive]

        # Parse "key=+0.0123" pairs (values are signed, fixed 4dp) and check
        # every one is finite -- a NaN/Inf here would print as literal
        # "nan"/"inf" text, which float() will happily parse, so also guard
        # against that explicitly.
        body = line.split("]", 1)[1].strip()
        parsed_keys = []
        for token in body.split("  "):
            if not token:
                continue
            _key, _val_str = token.split("=")
            val = float(_val_str)
            assert math.isfinite(val), f"{_key}={_val_str} is not finite"
            parsed_keys.append(_key)
        assert sorted(parsed_keys) == sorted(active_keys), (
            f"expected exactly the active heads, no masked/frozen ones -- "
            f"got={sorted(parsed_keys)} expected={sorted(active_keys)}"
        )

    def test_frozen_head_shows_ratio_of_exactly_one(self, caplog):
        """A frozen head's stored old_log_prob AND freshly-recomputed
        new_log_prob are both masked to exactly 0.0 (see
        test_frozen_head_kl_masking.py) -- so this diagnostic's ratio_h for
        that head must be exp(0-0)=1 exactly, for every row, every
        minibatch. Checked directly (not just "value looks small") by
        re-deriving what a ratio-of-1 breakdown must equal: the per-sample
        advantage itself (mb_w defaults to 1), which is NOT necessarily 0
        for an arbitrary minibatch -- unlike the val_episode_fraction=0
        "ratio=1 for every row -> policy_loss=0" case, this is only
        FULL-BATCH-normalized, not per-minibatch zero-mean, so this test
        checks the ratio/mechanism directly instead of asserting the
        resulting number is 0."""
        frozen = ["shoot_logit", "pass_logit", "tackle_logit"]
        trainer = PPOTrainer.from_config()
        trainer.set_frozen_heads(frozen)
        env = _make_env()
        buffer, last_obs = _collect_rollout(env, trainer, _ROLLOUT_STEPS)

        with torch.no_grad():
            last_obs_dict = {
                k: v.unsqueeze(0).to(trainer.device) for k, v in last_obs.to_torch_dict().items()
            }
            last_value = trainer._get_value(last_obs_dict)
        advantages, returns = buffer.compute_gae(trainer.gamma, trainer.lam, last_value)
        batch = buffer.as_tensors(advantages, returns)

        frozen_lp_keys = ["shoot", "pass_", "tackle"]
        key_to_idx = {k: i for i, k in enumerate(HEAD_LP_KEYS)}
        for key in frozen_lp_keys:
            idx = key_to_idx[key]
            assert torch.all(batch["head_log_probs"][:, idx] == 0.0), (
                f"stored head_log_probs for frozen head {key!r} is not all-zero -- "
                "this test's premise (frozen heads store 0.0) doesn't hold, "
                "see test_frozen_head_kl_masking.py"
            )

        mb_obs = {k.replace("obs/", ""): batch[k][:8].to(trainer.device)
                  for k in batch if k.startswith("obs/")}
        mb_actions = {k.replace("action/", ""): batch[k][:8].to(trainer.device)
                      for k in batch if k.startswith("action/")}
        sf, of, em = mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"]
        bf, gf = mb_obs["ball_feat"], mb_obs["global_feat"]
        from footballcoach.ai.ppo.ppo_trainer import _ai_types
        sat, oat = _ai_types(mb_obs)
        with torch.no_grad():
            d_heads = trainer.decision_net(sf, of, em, bf, gf, sat, oat)
            e_heads = trainer.execution_net(sf, of, em, bf, gf, d_heads, sat, oat)
            new_head_lp = trainer._per_head_new_log_probs(d_heads, e_heads, mb_actions, em)
        for key in frozen_lp_keys:
            idx = key_to_idx[key]
            assert torch.all(new_head_lp[:, idx] == 0.0), (
                f"freshly-recomputed head_log_probs for frozen head {key!r} is not "
                "all-zero -- ratio_h would not actually be 1 for this head, "
                "making the counterfactual breakdown wrong for frozen heads"
            )


class TestEpochPolicyLossPercentiles:
    """The "[policy percentiles, n=...]" line -- per-SAMPLE (not
    per-minibatch-mean) TRAIN policy_loss distribution, reported every
    epoch (like the held-out percentiles), requested to check whether the
    [policy] mean's positive value is a broad shift or a few outliers
    dragging the mean. Unlike the held-out numbers, this is a progressive
    pool across that epoch's minibatches (rows reflect whatever weights
    existed when each minibatch ran, not one clean end-of-epoch snapshot)
    -- an approximation, not an exact per-epoch distribution, but still
    useful for spotting skew/outliers."""

    def test_percentiles_logged_every_epoch_monotonic_and_finite(self, caplog):
        trainer = PPOTrainer.from_config()
        env = _make_env()
        buffer, last_obs = _collect_rollout(env, trainer, _ROLLOUT_STEPS)

        with caplog.at_level("INFO"):
            _run_ppo_update(trainer, buffer, last_obs)

        # startswith (not a plain substring check) so this doesn't also
        # pick up "[held-out policy percentiles, ...]" -- not relevant in
        # this file's fixture (val_episode_fraction stays at its 0.0
        # default here) but kept precise regardless.
        lines = [
            r.message for r in caplog.records
            if r.message.strip().startswith("[policy percentiles")
        ]
        n_epoch_lines = sum(
            1 for r in caplog.records if r.message.startswith("  == epoch ")
        )
        assert n_epoch_lines >= 1, "no epoch ran -- test is vacuous"
        # <= n_epoch_lines since KL early-stop can cut the epoch loop short
        # partway through a minibatch, same caveat as the held-out test.
        assert len(lines) == n_epoch_lines, (
            f"expected one percentile line per epoch that actually ran "
            f"({n_epoch_lines}), got {len(lines)}"
        )

        pcts = ["p0", "p1", "p10", "p50", "p90", "p99", "p100"]
        for line in lines:
            # "n=1,234" appears in the tag itself, before the pN= pairs.
            n_reported = int(line.split("n=")[1].split("]")[0].replace(",", ""))
            assert n_reported > 0

            body = line.split("]", 1)[1].strip()
            values = []
            for pct in pcts:
                assert f"{pct}=" in body, f"{pct} missing from percentile line: {line}"
            for token in body.split("  "):
                if not token:
                    continue
                _key, _val_str = token.split("=")
                val = float(_val_str)
                assert math.isfinite(val), f"{_key}={_val_str} is not finite"
                values.append(val)
            assert len(values) == len(pcts)
            assert values == sorted(values), f"percentiles are not monotonically non-decreasing: {values}"


class TestEpochPolicyLossPerHeadBreakdown:
    """The "[policy heads] shoot=... move_dir=..." line -- per-epoch
    counterfactual per-head policy_loss breakdown, requested to answer
    "which heads are contributing to [policy] mean at each epoch" directly,
    rather than only at the very end of the whole update (the pre-existing
    "[per-head policy_loss (counterfactual, ...)]" line). Same formula/
    semantics as that end-of-run line (see this file's own module
    docstring) -- just sliced to one epoch's minibatches via the same
    _epoch_slice_start window [policy]/[policy percentiles] already use,
    instead of averaged over the whole update."""

    def test_logged_every_epoch_with_every_active_head_finite(self, caplog):
        """Masked/frozen/inactive heads are dropped from this line too (see
        TestPerHeadPolicyLossBreakdown.test_log_line_lists_every_active_head_with_finite_values's
        own docstring for why -- same PPOTrainer._inactive_head_lp_keys
        filter, same rationale)."""
        trainer = PPOTrainer.from_config()
        env = _make_env()
        buffer, last_obs = _collect_rollout(env, trainer, _ROLLOUT_STEPS)
        inactive = trainer._inactive_head_lp_keys()
        active_keys = [k for k in HEAD_LP_KEYS if k not in inactive]

        with caplog.at_level("INFO"):
            _run_ppo_update(trainer, buffer, last_obs)

        n_epoch_lines = sum(
            1 for r in caplog.records if r.message.startswith("  == epoch ")
        )
        assert n_epoch_lines >= 1, "no epoch ran -- test is vacuous"
        lines = [
            r.message for r in caplog.records
            if r.message.strip().startswith("[policy heads]")
        ]
        # <= n_epoch_lines since KL early-stop can cut the epoch loop short
        # partway through a minibatch, same caveat as the percentile line.
        assert len(lines) == n_epoch_lines, (
            f"expected one per-head line per epoch that actually ran "
            f"({n_epoch_lines}), got {len(lines)}"
        )

        for line in lines:
            body = line.split("]", 1)[1].strip()
            parsed_keys = []
            for token in body.split("  "):
                if not token:
                    continue
                key, val_str = token.split("=")
                assert key in HEAD_LP_KEYS, f"unexpected head {key!r} in: {line}"
                assert math.isfinite(float(val_str)), f"{key}={val_str} is not finite"
                parsed_keys.append(key)
            assert sorted(parsed_keys) == sorted(active_keys), (
                f"expected exactly the active heads, no masked/frozen ones -- "
                f"got={sorted(parsed_keys)} expected={sorted(active_keys)}"
            )


class TestRolloutSummaryEntropyDropsInactiveHeads:
    """The "[PPO] ... entropy  shoot=... move_dir=..." breakdown line inside
    _log_rollout_summary's multi-line rollout summary -- same
    PPOTrainer._inactive_head_lp_keys filter as the per-head KL/policy_loss
    lines above, applied to metrics['entropy_breakdown'] before building
    the display string. Unlike KL/policy_loss (which are only ever
    approximately or artifactually zero for a masked head), entropy for a
    masked head is an intentional, exact 0.0 set inside _compute_entropy
    itself (see that method's own docstring) -- still dropped rather than
    shown, since a guaranteed-0.0 entry for an untrained head is exactly as
    uninformative as the KL/policy_loss cases.

    Calls _log_rollout_summary directly with a REAL metrics dict (from an
    actual _ppo_update() call, so entropy_breakdown is real data, not
    hand-built) and empty/trivial values for the other rollout-bookkeeping
    params -- the same empty initial forms train()'s own loop uses before
    any episode has completed this rollout (episode_rewards=[], etc.), so
    this exercises an already-load-bearing code path, not a synthetic edge
    case."""

    def test_entropy_breakdown_state_omits_inactive_heads(self, caplog):
        """Asserts on trainer._prev_entropy_breakdown (the ALREADY-FILTERED
        dict _log_rollout_summary stores right after building the display
        string -- see that method's own "_prev_entropy_breakdown = dict(
        _ent_bkdn)" line) rather than parsing the rendered multi-line log
        text: the "heads" section a few lines below entropy in the same
        rollout summary reuses several of the SAME short names (move=,
        exec_move=, sprint=) for a completely different metric (action-
        taken proportions), so scanning the combined message's raw text for
        "key=value" tokens can't reliably tell the two sections apart.
        Asserting on the dict that's the direct input to string-building
        tests the actual filtering behavior without that ambiguity."""
        trainer = PPOTrainer.from_config()
        env = _make_env()
        buffer, last_obs = _collect_rollout(env, trainer, _ROLLOUT_STEPS)
        inactive = trainer._inactive_head_lp_keys()
        assert inactive, (
            "expected at least kick_spin to be permanently frozen -- if this "
            "changed, _kick_spin_frozen's default changed too, update this test"
        )

        metrics = _run_ppo_update(trainer, buffer, last_obs)
        assert metrics.get("entropy_breakdown"), "fixture produced no entropy_breakdown -- test is vacuous"
        active_keys = [k for k in HEAD_LP_KEYS if k not in inactive]

        with caplog.at_level("INFO"):
            trainer._log_rollout_summary(
                metrics=metrics, steps_per_sec=1.0,
                episode_rewards=[], secondary_episode_rewards=[],
                episode_outcomes_vs_rules=[], episode_outcomes_vs_immobile=[],
                episode_outcomes_vs_neural=[], rollout_components={},
                episode_comp_list=[], episode_durations_s=[],
                comp_step_stats={}, n_reward_comp_steps=0,
            )

        assert trainer._prev_entropy_breakdown is not None, (
            "_log_rollout_summary should have set _prev_entropy_breakdown "
            "from a non-empty entropy_breakdown"
        )
        assert sorted(trainer._prev_entropy_breakdown.keys()) == sorted(active_keys), (
            f"expected exactly the active heads, no masked/frozen ones -- "
            f"got={sorted(trainer._prev_entropy_breakdown.keys())} expected={sorted(active_keys)}"
        )
        for k in inactive:
            assert k not in trainer._prev_entropy_breakdown

        # Sanity check the rendered log text too (loosely): at least one
        # active head's own entropy value should appear literally, and the
        # combined message should exist -- doesn't try to delineate exact
        # line boundaries (see docstring above), just confirms something
        # was actually logged.
        _lines = [r.message for r in caplog.records if "[PPO] step=" in r.message]
        assert len(_lines) == 1
        _sample_key = active_keys[0]
        _sample_val = f"{trainer._prev_entropy_breakdown[_sample_key]:.4f}"
        assert f"{_sample_key}={_sample_val}" in _lines[0]

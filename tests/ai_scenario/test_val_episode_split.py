"""Tests for the held-out validation-episode diagnostic (PPOTrainer.
val_episode_fraction / _split_train_val_episodes / _eval_val_episode_losses,
see __init__'s val_episode_fraction comment).

Purely a diagnostic reported once per PPO epoch (the "[held-out policy]"/
"[held-out value]" log lines in _ppo_update) -- never used for early
stopping or any other training decision. These tests guard:

  1. _split_train_val_episodes is TRACK-AWARE: with a neural secondary
     opponent producing a second, interleaved chronological track (see
     RolloutBuffer.track_ids's own docstring), every held-out episode's rows
     must all belong to a single track -- a flat dones-only scan over the
     concatenated batch would sometimes stitch together rows from BOTH
     tracks that merely happen to sit inside the same
     [prev_done+1, done] flat span.
  2. Every row ends up in exactly one of train/val (no row dropped or
     duplicated), tensor and list-keyed batch fields alike.
  3. val_episode_fraction=0.0 (default) never even calls the split --
     zero behavior change to _ppo_update.
  4. _eval_val_episode_losses never mutates any parameter (no grad, no
     optimizer step) -- held-out rows genuinely never train the network.
  5. End to end: _ppo_update with val_episode_fraction>0 still actually
     trains (params change) while reporting finite held-out losses.

Mirrors test_secondary_opponent_gae.py's/test_episode_replay.py's fixture
patterns: a fully neural secondary opponent (opponent_rules_prob=0.0,
opponent_immobile_prob=0.0) for a genuine second track, built directly via
ScenarioEnv/build_1v1_scenario with explicit kwargs (not curriculum/envs.py's
build_env(), which reads the live, concurrently-mutable ai_config.json) --
and a short max_episode_s so a modest rollout budget reliably produces
several COMPLETE episodes per track, which the split needs to be
non-vacuous.
"""
from __future__ import annotations

import copy
import functools
import math

import pytest
import torch

from footballcoach.ai.config import load_ai_config
from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ai.models.decision_network import DecisionNetwork
from footballcoach.ai.models.execution_network import ExecutionNetwork
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _action_to_numpy, _trimmed_mean_p10_p90
from footballcoach.ai.ppo.rollout_buffer import HEAD_LP_KEYS, RolloutBuffer
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario

_ROLLOUT_STEPS = 90
_SHORT_MAX_EPISODE_S = 2.0


def _make_env_with_neural_opponent() -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="val_episode_split_test",
        label="Val episode split test",
        description=(
            "1v1 with a fully neural secondary opponent, short episodes so "
            "a modest rollout budget produces several complete episodes "
            "per track."
        ),
        build=functools.partial(
            build_1v1_scenario, opponent_rules_prob=0.0, opponent_immobile_prob=0.0,
        ),
    )
    return ScenarioEnv(
        definition=defn, trainee_player_id="trainee", phase=1,
        secondary_player_ids=["opponent"], max_episode_s=_SHORT_MAX_EPISODE_S,
    )


def _collect_rollout_with_secondary(env: ScenarioEnv, trainer: PPOTrainer, n: int):
    buffer = RolloutBuffer()
    env.sample_action_fn = trainer._sample_action
    env.reset()
    next_obs = None
    for _ in range(n):
        next_obs, reward, done, info = env.step()
        tr = env.last_trainee_transition
        if tr is not None:
            buffer.add(
                obs=tr["obs"], action=_action_to_numpy(tr["action"], tr["raw_exec"]),
                log_prob=tr["log_prob"], value=tr["value"], reward=reward,
                done=1.0 if done else 0.0, head_log_probs=tr.get("head_log_probs"),
                step_outcome=(info.trial_outcome or "") if (done and info is not None) else "",
            )
        for sec in getattr(env, "last_secondary_results", []):
            buffer.add(
                obs=sec["obs"], action=_action_to_numpy(sec["action"], sec["raw_exec"]),
                log_prob=sec["log_prob"], value=sec["value"], reward=sec["reward"],
                done=sec["done"], track_id=sec["player_id"],
                head_log_probs=sec.get("head_log_probs"),
                step_outcome=sec.get("step_outcome", ""),
            )
        if done:
            env.reset()
    return buffer, next_obs


@pytest.fixture(scope="module")
def trainer() -> PPOTrainer:
    cfg = copy.deepcopy(load_ai_config())
    # Small enough that a from-scratch _ppo_update over a tiny rollout stays
    # fast, but >1 epoch so the per-epoch val-loss log line actually runs
    # more than once.
    cfg["ppo"]["n_epochs"] = 3
    cfg["ppo"]["minibatch_size"] = 8
    return PPOTrainer(
        decision_net=DecisionNetwork.from_config(),
        execution_net=ExecutionNetwork.from_config(),
        cfg=cfg,
    )


@pytest.fixture
def multi_track_batch(trainer: PPOTrainer) -> dict:
    """A real collected rollout with GAE applied, both tracks present."""
    env = _make_env_with_neural_opponent()
    buffer, next_obs = _collect_rollout_with_secondary(env, trainer, _ROLLOUT_STEPS)
    assert set(buffer.track_ids) == {"trainee", "opponent"}, (
        "fixture didn't produce both tracks this run -- rerun or increase "
        "_ROLLOUT_STEPS; the track-awareness assertions below would be "
        "vacuous otherwise"
    )
    last_values = trainer._bootstrap_last_values(env, next_obs, buffer)
    advantages, returns = buffer.compute_gae(trainer.gamma, trainer.lam, last_values)
    return buffer.as_tensors(advantages, returns)


def _make_synthetic_two_track_batch() -> tuple[dict, list[tuple[str, list[int]]]]:
    """Hand-built batch with two tracks interleaved 1:1 (mirrors
    RolloutBuffer.track_ids's own docstring: "trainee_t0, secondary_t0,
    trainee_t1, secondary_t1, ..."), each with several episodes of known,
    DIFFERENT lengths -- deliberately chosen so a flat, track-naive
    ``dones > 0.5`` scan over the flat array would slice episode boundaries
    completely differently (and wrongly) than a per-track scan.

    Also stamps each row's "log_probs" entry with its own original flat
    index, purely as a fingerprint so the test can identify exactly which
    original rows a split landed on without assuming any row-order/
    contiguity invariant about the split's output.

    Returns ``(batch, episodes)`` where ``episodes`` is the independently
    (not implementation-derived) computed list of
    ``(track, flat_indices)`` -- note each episode's own flat indices are
    NOT contiguous (they skip every other row, which belongs to the other,
    interleaved track).
    """
    trainee_lens = [3, 4, 5]
    opponent_lens = [4, 4, 4]

    def _dones_for(lens: list[int]) -> list[float]:
        out: list[float] = []
        for length in lens:
            out.extend([0.0] * (length - 1) + [1.0])
        return out

    trainee_dones = _dones_for(trainee_lens)
    opponent_dones = _dones_for(opponent_lens)
    assert len(trainee_dones) == len(opponent_dones) == 12

    n = 24
    dones = [0.0] * n
    track_ids = [""] * n
    for i in range(12):
        dones[2 * i] = trainee_dones[i]
        dones[2 * i + 1] = opponent_dones[i]
        track_ids[2 * i] = "trainee"
        track_ids[2 * i + 1] = "opponent"

    episodes: list[tuple[str, list[int]]] = []
    pos = 0
    for length in trainee_lens:
        episodes.append(("trainee", [2 * (pos + k) for k in range(length)]))
        pos += length
    pos = 0
    for length in opponent_lens:
        episodes.append(("opponent", [2 * (pos + k) + 1 for k in range(length)]))
        pos += length

    batch = {
        "dones": torch.tensor(dones, dtype=torch.float32),
        "track_ids": track_ids,
        "log_probs": torch.arange(n, dtype=torch.float32),
    }
    return batch, episodes


class TestSplitIsTrackAware:
    def test_split_keeps_every_episode_intact_and_track_correct(self, trainer):
        trainer.val_episode_fraction = 0.5
        batch, episodes = _make_synthetic_two_track_batch()

        train_batch, val_batch = trainer._split_train_val_episodes(batch)
        assert val_batch is not None

        val_indices = {int(x) for x in val_batch["log_probs"].tolist()}
        train_indices = {int(x) for x in train_batch["log_probs"].tolist()}
        assert val_indices.isdisjoint(train_indices)
        assert val_indices | train_indices == set(range(24))

        n_val_episodes = 0
        for track, flat_indices in episodes:
            span = set(flat_indices)
            # A flat, track-naive scan (or a track-aware scan that then
            # wrongly masks a *contiguous* [start:end+1] range instead of
            # this episode's own specific, non-contiguous rows) would let
            # this exact episode straddle train/val -- this assertion is
            # what either bug fails.
            assert span <= val_indices or span <= train_indices, (
                f"{track} episode {sorted(span)} was split across train/val"
            )
            if span <= val_indices:
                n_val_episodes += 1
        assert n_val_episodes == 3, f"expected 3 of 6 episodes held out, got {n_val_episodes}"

    def test_train_and_val_partition_every_row_exactly_once(self, trainer, multi_track_batch):
        trainer.val_episode_fraction = 0.5
        n_total = len(multi_track_batch["dones"])
        train_batch, val_batch = trainer._split_train_val_episodes(multi_track_batch)
        assert val_batch is not None
        assert len(train_batch["dones"]) + len(val_batch["dones"]) == n_total
        # List-keyed fields (see RolloutBuffer.as_tensors) must be split the
        # same way as the tensor fields -- same lengths on both sides.
        for b in (train_batch, val_batch):
            assert len(b["track_ids"]) == len(b["dones"])
            assert len(b["step_outcomes"]) == len(b["dones"])

    def test_too_few_episodes_skips_split(self, trainer):
        """<2 complete episodes: split must be a no-op (everything trains),
        not a crash or a degenerate 100%/0% split."""
        trainer.val_episode_fraction = 0.5
        tiny_batch = {
            "dones": torch.tensor([0.0, 0.0, 1.0]),
            "track_ids": ["trainee", "trainee", "trainee"],
            "log_probs": torch.zeros(3),
        }
        train_batch, val_batch = trainer._split_train_val_episodes(tiny_batch)
        assert val_batch is None
        assert train_batch is tiny_batch


class TestDisabledByDefault:
    def test_zero_fraction_never_calls_split(self, trainer, multi_track_batch, monkeypatch):
        trainer.val_episode_fraction = 0.0

        def _boom(self, batch):
            raise AssertionError("_split_train_val_episodes must not be called when disabled")

        monkeypatch.setattr(PPOTrainer, "_split_train_val_episodes", _boom)
        metrics = trainer._ppo_update(multi_track_batch, progress=0.0)
        assert math.isfinite(metrics["policy_loss"])


class TestEvalNeverTrains:
    def test_eval_val_episode_losses_does_not_change_params(self, trainer, multi_track_batch):
        trainer.val_episode_fraction = 0.5
        _train_batch, val_batch = trainer._split_train_val_episodes(multi_track_batch)
        assert val_batch is not None

        before = [p.detach().clone() for p in trainer.decision_net.parameters()] + [
            p.detach().clone() for p in trainer.execution_net.parameters()
        ]
        (
            pol_loss, val_loss, kl, per_sample_pol, val_loss_trimmed, _val_loss_trimmed2,
            pol_loss_trimmed, _pol_loss_trimmed2, _surr_mean, _surr_trimmed, _surr_trimmed2,
            _head_pol,
        ) = trainer._eval_val_episode_losses(val_batch, clip=trainer.schedules.clip(0.0))
        after = [p.detach().clone() for p in trainer.decision_net.parameters()] + [
            p.detach().clone() for p in trainer.execution_net.parameters()
        ]

        assert math.isfinite(pol_loss)
        assert math.isfinite(val_loss)
        assert math.isfinite(kl)
        assert math.isfinite(val_loss_trimmed)
        assert math.isfinite(pol_loss_trimmed)
        assert len(per_sample_pol) == len(val_batch["log_probs"])
        assert torch.isfinite(per_sample_pol).all()
        assert per_sample_pol.mean().item() == pytest.approx(pol_loss, abs=1e-4), (
            "per-sample policy_loss tensor's mean must match the returned scalar"
        )
        for b, a in zip(before, after):
            assert torch.equal(b, a), "held-out validation eval must never change any parameter"


class TestEndToEnd:
    def test_ppo_update_reports_val_losses_and_still_trains(self, trainer, multi_track_batch):
        trainer.val_episode_fraction = 0.5
        before = [p.detach().clone() for p in trainer.decision_net.parameters()]

        metrics = trainer._ppo_update(multi_track_batch, progress=0.0)

        after = [p.detach().clone() for p in trainer.decision_net.parameters()]
        assert any(not torch.equal(b, a) for b, a in zip(before, after)), (
            "decision_net params did not change -- held-out split must not "
            "have silently swallowed the whole training set"
        )
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                assert math.isfinite(val), f"PPO metrics['{key}']={val} is not finite"

    def test_val_percentile_line_logged_every_epoch_monotonic(self, trainer, multi_track_batch, caplog):
        """Unlike the train-side "[policy percentiles]" line, val_batch is
        fixed and cheap to re-evaluate, so "[held-out policy percentiles]"
        is logged every epoch, not just the first."""
        trainer.val_episode_fraction = 0.5
        with caplog.at_level("INFO"):
            trainer._ppo_update(multi_track_batch, progress=0.0)

        lines = [
            r.message for r in caplog.records
            if r.message.strip().startswith("[held-out policy percentiles")
        ]
        # <= n_epochs (KL early-stop can cut the loop short), but should
        # fire for every epoch that actually ran, not just the first.
        assert 1 <= len(lines) <= trainer.n_epochs
        pcts = ["p0", "p1", "p10", "p50", "p90", "p99", "p100"]
        for line in lines:
            body = line.split("]", 1)[1].strip()
            values = []
            for pct in pcts:
                assert f"{pct}=" in body
            for token in body.split("  "):
                if not token:
                    continue
                _key, _val_str = token.split("=")
                values.append(float(_val_str))
            assert len(values) == len(pcts)
            assert all(math.isfinite(v) for v in values)
            assert values == sorted(values), f"percentiles not monotonic: {values}"

    def test_policy_loss_and_kl_are_exactly_zero_at_ratio_one_before_any_training(self, trainer, multi_track_batch):
        """Ground truth for why val_episode_policy_loss/val_episode_kl
        aren't generalization diagnostics: at ratio=1 (i.e. before any
        gradient step touches a row), the clipped surrogate collapses to
        -mean(adv_normalized), which is exactly 0 by construction
        (advantages are normalized to zero mean), and the approx-KL
        estimator mean(old_lp - new_lp) is exactly 0 since new_lp == old_lp
        -- for ANY subset of rows, not something specific to being "held
        out". Confirmed here on both the val split and an equal-size sample
        of rows that are about to be trained on (but haven't yet)."""
        trainer.val_episode_fraction = 0.5
        train_batch, val_batch = trainer._split_train_val_episodes(multi_track_batch)
        assert val_batch is not None
        n = len(train_batch["log_probs"])
        sample_idx = torch.randperm(n)[:len(val_batch["log_probs"])]
        train_sample = {
            k: (v[sample_idx] if torch.is_tensor(v) else [v[i] for i in sample_idx.tolist()])
            for k, v in train_batch.items()
        }
        clip = trainer.schedules.clip(0.0)

        (
            val_pol_loss, _, val_kl, val_per_sample_pol, _, _, _, _,
            val_surr_mean, val_surr_trimmed, val_surr_trimmed2, _,
        ) = trainer._eval_val_episode_losses(val_batch, clip)
        (
            train_pol_loss, _, train_kl, train_per_sample_pol, _, _, _, _,
            train_surr_mean, train_surr_trimmed, train_surr_trimmed2, _,
        ) = trainer._eval_val_episode_losses(train_sample, clip)
        assert val_pol_loss == pytest.approx(0.0, abs=1e-5)
        assert train_pol_loss == pytest.approx(0.0, abs=1e-5)
        assert val_kl == pytest.approx(0.0, abs=1e-5)
        # (r-1)*A is IDENTICALLY 0 per-row at ratio=1 (not just ~0 via a
        # normalized-mean-zero cancellation the way -mean(A) is) -- an even
        # cleaner degenerate case than policy_loss's own.
        assert val_surr_mean == pytest.approx(0.0, abs=1e-6)
        assert train_surr_mean == pytest.approx(0.0, abs=1e-6)
        assert val_surr_trimmed == pytest.approx(0.0, abs=1e-6)
        assert train_surr_trimmed == pytest.approx(0.0, abs=1e-6)
        assert val_surr_trimmed2 == pytest.approx(0.0, abs=1e-6)
        assert train_surr_trimmed2 == pytest.approx(0.0, abs=1e-6)
        assert train_kl == pytest.approx(0.0, abs=1e-5)

        # The MEAN is exactly 0, but individual rows are NOT -- at ratio=1,
        # surr1=surr2=adv_normalized exactly, so per_sample_policy_loss_i =
        # -adv_normalized_i (a real, varied per-row value; only its mean
        # cancels to 0 by construction). Confirm that exact relationship
        # rather than the wrong stronger claim "every row is 0" (caught by
        # this test itself failing when it asserted that).
        def _normalized_adv(b: dict) -> torch.Tensor:
            adv = b["advantages"]
            return (adv - adv.mean()) / (adv.std() + 1e-8)

        assert torch.allclose(val_per_sample_pol, -_normalized_adv(val_batch), atol=1e-4)
        assert torch.allclose(train_per_sample_pol, -_normalized_adv(train_sample), atol=1e-4)


class TestTrimmedMeanP10P90:
    """Pure math tests for _trimmed_mean_p10_p90 -- the robust companion to
    val_episode_value_loss's plain mean (see that function's own docstring
    and _eval_val_episode_losses' val_episode_value_loss_p10_p90 return
    value): drops the bottom/top 10% before averaging, so a single
    badly-mis-valued state's huge squared-error term can't single-handedly
    dominate the reported number the way it can dominate a plain mean."""

    def test_matches_plain_mean_when_no_outliers(self):
        x = torch.linspace(0.0, 1.0, steps=1000)
        trimmed = _trimmed_mean_p10_p90(x)
        assert trimmed == pytest.approx(float(x.mean()), abs=0.01), (
            "a smoothly-varying, symmetric distribution's trimmed mean "
            "should equal its plain mean regardless of trim fraction -- "
            "trimming only matters when there ARE outliers"
        )

    def test_a_single_huge_outlier_is_excluded(self):
        # 999 ordinary values around 1.0, plus one absurd outlier -- the
        # outlier is the top 1/1000 = 0.1%, comfortably inside the top-10%
        # cut, so it must be fully excluded from the trimmed mean.
        x = torch.cat([torch.ones(999), torch.tensor([1_000_000.0])])
        trimmed = _trimmed_mean_p10_p90(x)
        assert trimmed == pytest.approx(1.0, abs=0.05), (
            f"expected the huge outlier to be excluded, got trimmed mean={trimmed}"
        )
        # Sanity: the PLAIN mean, by contrast, is completely dominated by it.
        assert float(x.mean()) > 900.0

    def test_empty_tensor_returns_nan(self):
        assert math.isnan(_trimmed_mean_p10_p90(torch.zeros(0)))


class TestValEpisodeLossesTrimmed:
    """p10-p90 AND p25-p75 trimmed companions on the HELD-OUT set's own
    losses -- the "[held-out value]" line's p10_p90=/p25_p75= (squared-error
    terms, so >= 0) and the "[held-out policy]" line's p10_p90=/p25_p75=
    (clipped-surrogate terms, sign-mixed -- see [held-out policy]'s own
    "not a generalization diagnostic" caveat, which both trimmed versions
    inherit unchanged). p25_p75 is the more aggressive, symmetric
    interquartile mean, logged alongside (not instead of) p10_p90."""

    def test_returned_and_finite_and_logged(self, trainer, multi_track_batch, caplog):
        trainer.val_episode_fraction = 0.5
        _train_batch, val_batch = trainer._split_train_val_episodes(multi_track_batch)
        assert val_batch is not None

        clip = trainer.schedules.clip(0.0)
        with caplog.at_level("INFO"):
            (
                _pol, val_loss, _kl, _per_sample, val_loss_trimmed, val_loss_trimmed2,
                pol_loss_trimmed, pol_loss_trimmed2, _surr_mean, _surr_trimmed, _surr_trimmed2,
                _head_pol,
            ) = trainer._eval_val_episode_losses(val_batch, clip)
        assert math.isfinite(val_loss_trimmed)
        assert val_loss_trimmed >= 0.0, "trimmed mean of non-negative squared-error terms can't be negative"
        assert math.isfinite(val_loss_trimmed2)
        assert val_loss_trimmed2 >= 0.0, "trimmed mean of non-negative squared-error terms can't be negative"
        assert math.isfinite(pol_loss_trimmed)
        assert math.isfinite(pol_loss_trimmed2)
        assert math.isfinite(_surr_mean)
        assert math.isfinite(_surr_trimmed)
        assert math.isfinite(_surr_trimmed2)

        with caplog.at_level("INFO"):
            trainer._ppo_update(multi_track_batch, progress=0.0)
        val_loss_lines = [
            r.message for r in caplog.records if r.message.strip().startswith("[held-out value]")
        ]
        pol_loss_lines = [
            r.message for r in caplog.records if r.message.strip().startswith("[held-out policy]")
        ]
        assert len(val_loss_lines) >= 1, "[held-out value] line must appear in the per-epoch log output"
        assert len(pol_loss_lines) >= 1, "[held-out policy] line must appear in the per-epoch log output"
        for line in val_loss_lines:
            val = float(line.split("p10_p90=")[1].split("(")[0])
            assert math.isfinite(val)
            assert val >= 0.0
            val2 = float(line.split("p25_p75=")[1].split("(")[0])
            assert math.isfinite(val2)
            assert val2 >= 0.0
        for line in pol_loss_lines:
            val = float(line.split("p10_p90=")[1].split()[0])
            assert math.isfinite(val)
            val2 = float(line.split("p25_p75=")[1].split()[0])
            assert math.isfinite(val2)


class TestHeldOutPerHeadPolicyLossBreakdown:
    """The "[held-out policy heads] shoot=... move_dir=..." line -- same
    per-head counterfactual breakdown as the train-side "[policy heads]"
    line (see test_per_head_policy_loss_breakdown.py's
    TestEpochPolicyLossPerHeadBreakdown), computed on the held-out set
    instead. Regression coverage for a real bug found and fixed alongside
    this feature: the secondary/opponent-player buffer.add() calls in
    ppo_trainer.py/batched_rollout_worker.py/rollout_worker.py never passed
    head_log_probs, silently defaulting every secondary row's stored
    per-head log-probs to zero (RolloutBuffer.add()'s default) even though
    NeuralPlayerAI.last_transition sets real values identically for trainee
    and secondary players -- this corrupted every per-head KL/policy_loss
    diagnostic for the majority of rows whenever
    curriculum.phase1_opponent_neural_ratio > 0. Confirmed fixed via a
    direct numerical check: sum(per-head KL) vs. the scalar approx_kl for
    the same _ppo_update() call went from off by ~0.3-1.7 (with the bug) to
    matching to float precision (~3e-5) after passing head_log_probs
    through for secondary rows too -- see this module's own
    _collect_rollout_with_secondary fixture helper, fixed the same way."""

    def test_returned_and_logged_when_head_log_probs_present(self, trainer, multi_track_batch, caplog):
        trainer.val_episode_fraction = 0.5
        _train_batch, val_batch = trainer._split_train_val_episodes(multi_track_batch)
        assert val_batch is not None
        assert "head_log_probs" in val_batch
        assert not torch.all(val_batch["head_log_probs"] == 0.0), (
            "fixture produced all-zero head_log_probs -- this test would be "
            "exercised degenerately (regression-test the fixture fix above, "
            "not this feature)"
        )

        clip = trainer.schedules.clip(0.0)
        *__, head_pol = trainer._eval_val_episode_losses(val_batch, clip)
        assert head_pol is not None
        assert head_pol.shape == (len(HEAD_LP_KEYS),)
        assert torch.isfinite(head_pol).all()

        with caplog.at_level("INFO"):
            trainer._ppo_update(multi_track_batch, progress=0.0)
        lines = [
            r.message for r in caplog.records
            if r.message.strip().startswith("[held-out policy heads]")
        ]
        assert len(lines) >= 1, "[held-out policy heads] line must appear in the per-epoch log output"
        # Masked/frozen/inactive heads (PPOTrainer._inactive_head_lp_keys)
        # are dropped from this line -- same rationale/filter as the
        # train-side "[policy heads]" line, see
        # test_per_head_policy_loss_breakdown.py's
        # TestEpochPolicyLossPerHeadBreakdown docstring.
        inactive = trainer._inactive_head_lp_keys()
        active_keys = [k for k in HEAD_LP_KEYS if k not in inactive]
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


class TestTrainSideMeansTrimmed:
    """p10-p90 AND p25-p75 trimmed companions on the TRAIN-side
    "[policy]"/"[value]" summary lines -- [policy]'s p10_p90=/p25_p75=
    (policy, sign-mixed) and [value]'s p10_p90=/p25_p75= (value head,
    squared-error terms so >= 0). Distinct from "[held-out policy]"/
    "[held-out value]"'s own p10_p90=/p25_p75= (the HELD-OUT set's trimmed
    losses -- "val" is overloaded in this file between "value head" and
    "held-out validation set", see _epoch_sample_value_loss's own comment
    in ppo_trainer.py -- hence neither TRAIN-side line uses "val" as its
    tag at all)."""

    def test_logged_every_epoch_finite(self, trainer, multi_track_batch, caplog):
        with caplog.at_level("INFO"):
            trainer._ppo_update(multi_track_batch, progress=0.0)

        n_epoch_lines = sum(1 for r in caplog.records if r.message.startswith("  == epoch "))
        assert n_epoch_lines >= 1, "no epoch ran -- test is vacuous"

        pol_lines = [r.message for r in caplog.records if r.message.strip().startswith("[policy]")]
        val_lines = [r.message for r in caplog.records if r.message.strip().startswith("[value]")]
        assert len(pol_lines) == n_epoch_lines, (
            "[policy] should appear in every epoch's summary output, "
            f"got {len(pol_lines)} for {n_epoch_lines} epoch(s)"
        )
        assert len(val_lines) == n_epoch_lines, (
            "[value] should appear in every epoch's summary output, "
            f"got {len(val_lines)} for {n_epoch_lines} epoch(s)"
        )
        for line in pol_lines:
            val = float(line.split("p10_p90=")[1].split()[0])
            assert math.isfinite(val)
            val2 = float(line.split("p25_p75=")[1].split()[0])
            assert math.isfinite(val2)
        for line in val_lines:
            val = float(line.split("p10_p90=")[1].split("(")[0])
            assert math.isfinite(val)
            val2 = float(line.split("p25_p75=")[1].split("(")[0])
            assert math.isfinite(val2)
            assert val >= 0.0, "trimmed mean of non-negative squared-error terms can't be negative"


class TestSurrogateDiagnostic:
    """"[policy surrogate]"/"[held-out policy surrogate]" -- E[(r-1)*A],
    UNCLIPPED and without the dual-clip floor, logged as a mean + p10_p90 +
    p25_p75 trimmed companions (same _trimmed_mean_p10_p90/
    _trimmed_mean_p25_p75 helpers as every other per-sample line here).
    Mathematically == -[policy]/-[held-out policy]'s own mean whenever the
    clip/dual-clip floor are inactive for a row (see
    _eval_val_episode_losses' own docstring) -- NOT a different signal in
    the common case, just a clip-invariant one, so its value is specifically
    in exposing rows where clipping DOES engage (which [policy]/[held-out
    policy] intentionally floor away to protect the real gradient step)."""

    def test_logged_every_epoch_finite_train_and_held_out(self, trainer, multi_track_batch, caplog):
        trainer.val_episode_fraction = 0.5
        with caplog.at_level("INFO"):
            trainer._ppo_update(multi_track_batch, progress=0.0)

        n_epoch_lines = sum(1 for r in caplog.records if r.message.startswith("  == epoch "))
        assert n_epoch_lines >= 1, "no epoch ran -- test is vacuous"

        train_lines = [
            r.message for r in caplog.records if r.message.strip().startswith("[policy surrogate]")
        ]
        held_out_lines = [
            r.message for r in caplog.records
            if r.message.strip().startswith("[held-out policy surrogate]")
        ]
        assert len(train_lines) == n_epoch_lines, (
            "[policy surrogate] should appear in every epoch's summary output, "
            f"got {len(train_lines)} for {n_epoch_lines} epoch(s)"
        )
        assert len(held_out_lines) == n_epoch_lines, (
            "[held-out policy surrogate] should appear in every epoch's summary "
            f"output, got {len(held_out_lines)} for {n_epoch_lines} epoch(s)"
        )
        for line in train_lines:
            mean_val = float(line.split("mean=")[1].split()[0])
            trimmed_val = float(line.split("p10_p90=")[1].split()[0])
            trimmed_val2 = float(line.split("p25_p75=")[1].split()[0])
            assert math.isfinite(mean_val)
            assert math.isfinite(trimmed_val)
            assert math.isfinite(trimmed_val2)
        for line in held_out_lines:
            mean_val = float(line.split("mean=")[1].split()[0])
            trimmed_val = float(line.split("p10_p90=")[1].split()[0])
            trimmed_val2 = float(line.split("p25_p75=")[1].split()[0])
            assert math.isfinite(mean_val)
            assert math.isfinite(trimmed_val)
            assert math.isfinite(trimmed_val2)

    def test_held_out_surrogate_mean_matches_negative_policy_loss_at_ratio_one(
        self, trainer, multi_track_batch,
    ):
        """Direct numerical check of the documented identity: before any
        training touches a row (ratio=1 everywhere), policy_loss and
        surrogate_mean must be negatives of each other to float precision
        -- not just both individually ~0, but the actual relationship."""
        trainer.val_episode_fraction = 0.5
        _train_batch, val_batch = trainer._split_train_val_episodes(multi_track_batch)
        assert val_batch is not None
        clip = trainer.schedules.clip(0.0)
        (
            pol_loss, _val_loss, _kl, _per_sample, _v_p90, _v_p75, _p_p90, _p_p75,
            surr_mean, _surr_p90, _surr_p75, _head_pol,
        ) = trainer._eval_val_episode_losses(val_batch, clip)
        assert pol_loss == pytest.approx(-surr_mean, abs=1e-5)


class TestProperOldLogProbForAugmentedBatch:
    """PPOTrainer._recompute_old_log_probs_for_augmented_batch() -- replaces
    augment_batch()'s cheap tiled old_log_prob (correct only for the
    identity-flip copies) with the TRUE log pi_old(action|obs) for every
    row, flip_y-augmented copies included. Called from _ppo_update()
    whenever augment_n_slot_shuffles > 0, before this update's first
    gradient step (network still exactly theta_old at that point).

    Explicitly forces augment_n_slot_shuffles on (rather than relying on
    the live ai_config.json's own value, which this repo's own hazard
    notes say can drift from concurrent edits) so this test is meaningful
    regardless of the live config."""

    def test_recomputed_shapes_and_finite(self, trainer, multi_track_batch):
        trainer.augment_n_slot_shuffles = 1
        try:
            from footballcoach.ai.obs.augment import augment_batch
            augmented = augment_batch(multi_track_batch, 1, trainer._aug_rng)
            log_probs, head_log_probs = trainer._recompute_old_log_probs_for_augmented_batch(augmented)
            assert log_probs.shape == augmented["log_probs"].shape
            assert torch.isfinite(log_probs).all()
            if "head_log_probs" in augmented:
                assert head_log_probs is not None
                assert head_log_probs.shape == augmented["head_log_probs"].shape
                assert torch.isfinite(head_log_probs).all()
        finally:
            trainer.augment_n_slot_shuffles = 0

    def test_train_side_epoch_zero_reads_zero_with_augmentation_on(
        self, trainer, multi_track_batch, caplog,
    ):
        """The exact gap a real training log surfaced: WITHOUT this fix,
        the augmented TRAIN batch's epoch-0 [policy]/[policy surrogate]
        mean read a small but real nonzero value (the flip_y copies'
        borrowed old_log_prob approximation error), while [held-out
        policy] (never augmented) read exactly 0.0000 at the same ratio=1
        snapshot. With the fix, train must read ~0 too, since old_log_prob
        is now properly recomputed for every augmented row."""
        trainer.augment_n_slot_shuffles = 1
        trainer.log_epoch_zero_baseline = True
        try:
            with caplog.at_level("INFO"):
                trainer._ppo_update(multi_track_batch, progress=0.0)
        finally:
            trainer.augment_n_slot_shuffles = 0
            trainer.log_epoch_zero_baseline = False

        epoch0_idx = next(
            i for i, r in enumerate(caplog.records) if r.message.startswith("  == epoch 0")
        )
        epoch1_idx = next(
            i for i, r in enumerate(caplog.records)
            if i > epoch0_idx and r.message.startswith("  == epoch 1")
        )
        baseline_lines = [r.message for r in caplog.records[epoch0_idx + 1:epoch1_idx]]

        pol_line = next(l for l in baseline_lines if l.strip().startswith("[policy]"))
        pol_mean = float(pol_line.split("mean=")[1].split()[0])
        kl = float(pol_line.split("kl=")[1].split()[0])
        assert pol_mean == pytest.approx(0.0, abs=1e-3)
        assert kl == pytest.approx(0.0, abs=1e-3)

        surr_line = next(l for l in baseline_lines if l.strip().startswith("[policy surrogate]"))
        surr_mean = float(surr_line.split("mean=")[1].split()[0])
        assert surr_mean == pytest.approx(0.0, abs=1e-3)


class TestEpochZeroBaseline:
    """PPOTrainer.log_epoch_zero_baseline (default False) -- an opt-in
    "== epoch 0 (pre-training baseline, ratio=1 exactly) ==" block logged
    BEFORE epoch 1, in the same [policy]/[policy surrogate]/[policy
    percentiles]/[policy heads]/[value] (and held-out equivalents) format
    as every real per-epoch block, evaluated under current weights before
    any gradient step this update has run. Reuses _eval_val_episode_losses
    for both the train batch and val_batch, so policy_loss/kl/surrogate
    read trivially ~0 (ratio=1 exactly everywhere -- same identity
    TestPolicyLossZeroAtRatioOne-style tests elsewhere in this file rely
    on), which is expected, not something these tests need to re-litigate."""

    def test_disabled_by_default_no_epoch_zero_line(self, trainer, multi_track_batch, caplog):
        assert trainer.log_epoch_zero_baseline is False
        with caplog.at_level("INFO"):
            trainer._ppo_update(multi_track_batch, progress=0.0)
        lines = [r.message for r in caplog.records if "epoch 0" in r.message]
        assert lines == [], f"epoch-0 baseline must not appear when disabled, got: {lines}"

    def test_enabled_logs_epoch_zero_before_epoch_one(self, trainer, multi_track_batch, caplog):
        trainer.log_epoch_zero_baseline = True
        try:
            with caplog.at_level("INFO"):
                trainer._ppo_update(multi_track_batch, progress=0.0)
        finally:
            trainer.log_epoch_zero_baseline = False

        epoch_header_lines = [
            i for i, r in enumerate(caplog.records)
            if r.message.startswith("  == epoch ")
        ]
        assert len(epoch_header_lines) >= 2, "expected an epoch-0 header AND at least epoch 1's own"
        assert "epoch 0" in caplog.records[epoch_header_lines[0]].message
        assert "epoch 1" in caplog.records[epoch_header_lines[1]].message

        # Every line between the epoch-0 and epoch-1 headers is the baseline
        # block -- same tag set as a real epoch, minus [grad clip]/kl-early-
        # stop (neither applies before any gradient step exists).
        baseline_lines = [
            r.message for r in caplog.records[epoch_header_lines[0] + 1:epoch_header_lines[1]]
        ]
        joined = "\n".join(baseline_lines)
        for tag in ("[policy]", "[policy surrogate]", "[value]"):
            assert tag in joined, f"{tag} missing from epoch-0 baseline block:\n{joined}"

        pol_line = next(l for l in baseline_lines if l.strip().startswith("[policy]"))
        pol_mean = float(pol_line.split("mean=")[1].split()[0])
        kl = float(pol_line.split("kl=")[1].split()[0])
        assert pol_mean == pytest.approx(0.0, abs=1e-4)
        assert kl == pytest.approx(0.0, abs=1e-4)

        surr_line = next(l for l in baseline_lines if l.strip().startswith("[policy surrogate]"))
        surr_mean = float(surr_line.split("mean=")[1].split()[0])
        assert surr_mean == pytest.approx(0.0, abs=1e-4)

        assert any("[grad clip] N/A" in l for l in baseline_lines)

    def test_enabled_includes_held_out_baseline_when_val_split_active(
        self, trainer, multi_track_batch, caplog,
    ):
        trainer.log_epoch_zero_baseline = True
        trainer.val_episode_fraction = 0.5
        try:
            with caplog.at_level("INFO"):
                trainer._ppo_update(multi_track_batch, progress=0.0)
        finally:
            trainer.log_epoch_zero_baseline = False
            trainer.val_episode_fraction = 0.0

        epoch0_idx = next(
            i for i, r in enumerate(caplog.records) if r.message.startswith("  == epoch 0")
        )
        epoch1_idx = next(
            i for i, r in enumerate(caplog.records)
            if i > epoch0_idx and r.message.startswith("  == epoch 1")
        )
        baseline_lines = [r.message for r in caplog.records[epoch0_idx + 1:epoch1_idx]]
        joined = "\n".join(baseline_lines)
        assert "[held-out policy]" in joined
        assert "[held-out policy surrogate]" in joined
        assert "[held-out value]" in joined

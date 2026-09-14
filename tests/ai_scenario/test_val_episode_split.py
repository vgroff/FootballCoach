"""Tests for the held-out validation-episode diagnostic (PPOTrainer.
val_episode_fraction / _split_train_val_episodes / _eval_val_episode_losses,
see __init__'s val_episode_fraction comment).

Purely a diagnostic reported once per PPO epoch (the val_episode_policy_loss=/
val_episode_value_loss= log line in _ppo_update) -- never used for early
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
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _action_to_numpy, _trimmed_mean_p1_p99
from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
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
        next_obs, reward, done, _info = env.step()
        tr = env.last_trainee_transition
        if tr is not None:
            buffer.add(
                obs=tr["obs"], action=_action_to_numpy(tr["action"], tr["raw_exec"]),
                log_prob=tr["log_prob"], value=tr["value"], reward=reward,
                done=1.0 if done else 0.0,
            )
        for sec in getattr(env, "last_secondary_results", []):
            buffer.add(
                obs=sec["obs"], action=_action_to_numpy(sec["action"], sec["raw_exec"]),
                log_prob=sec["log_prob"], value=sec["value"], reward=sec["reward"],
                done=sec["done"], track_id=sec["player_id"],
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
        pol_loss, val_loss, kl, per_sample_pol, val_loss_trimmed, pol_loss_trimmed = (
            trainer._eval_val_episode_losses(val_batch, clip=trainer.schedules.clip(0.0))
        )
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
        """Unlike the train-side "[epoch 1 policy_loss percentiles]" line
        (epoch 0 only), val_batch is fixed and cheap to re-evaluate, so its
        percentile breakdown is logged every epoch."""
        trainer.val_episode_fraction = 0.5
        with caplog.at_level("INFO"):
            trainer._ppo_update(multi_track_batch, progress=0.0)

        lines = [
            r.message for r in caplog.records
            if "val_episode_policy_loss percentiles" in r.message
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

        val_pol_loss, _, val_kl, val_per_sample_pol, _, _ = trainer._eval_val_episode_losses(val_batch, clip)
        train_pol_loss, _, train_kl, train_per_sample_pol, _, _ = trainer._eval_val_episode_losses(train_sample, clip)
        assert val_pol_loss == pytest.approx(0.0, abs=1e-5)
        assert train_pol_loss == pytest.approx(0.0, abs=1e-5)
        assert val_kl == pytest.approx(0.0, abs=1e-5)
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


class TestTrimmedMeanP1P99:
    """Pure math tests for _trimmed_mean_p1_p99 -- the robust companion to
    val_episode_value_loss's plain mean (see that function's own docstring
    and _eval_val_episode_losses' val_episode_value_loss_p1_p99 return
    value): drops the bottom/top 1% before averaging, so a single
    badly-mis-valued state's huge squared-error term can't single-handedly
    dominate the reported number the way it can dominate a plain mean."""

    def test_matches_plain_mean_when_no_outliers(self):
        x = torch.linspace(0.0, 1.0, steps=1000)
        trimmed = _trimmed_mean_p1_p99(x)
        assert trimmed == pytest.approx(float(x.mean()), abs=0.01), (
            "a smoothly-varying distribution's trimmed mean should be close "
            "to its plain mean -- trimming only matters when there ARE outliers"
        )

    def test_a_single_huge_outlier_is_excluded(self):
        # 999 ordinary values around 1.0, plus one absurd outlier -- the
        # outlier is the top 1/1000 = 0.1%, comfortably inside the top-1%
        # cut, so it must be fully excluded from the trimmed mean.
        x = torch.cat([torch.ones(999), torch.tensor([1_000_000.0])])
        trimmed = _trimmed_mean_p1_p99(x)
        assert trimmed == pytest.approx(1.0, abs=0.05), (
            f"expected the huge outlier to be excluded, got trimmed mean={trimmed}"
        )
        # Sanity: the PLAIN mean, by contrast, is completely dominated by it.
        assert float(x.mean()) > 900.0

    def test_empty_tensor_returns_nan(self):
        assert math.isnan(_trimmed_mean_p1_p99(torch.zeros(0)))


class TestValEpisodeLossesTrimmed:
    """p1-p99 trimmed companions on the HELD-OUT set's own losses --
    val_episode_value_loss_p1_p99 (squared-error terms, so >= 0) and
    val_episode_policy_loss_p1_p99 (clipped-surrogate terms, sign-mixed --
    see val_episode_policy_loss's own "not a generalization diagnostic"
    caveat, which this trimmed version inherits unchanged)."""

    def test_returned_and_finite_and_logged(self, trainer, multi_track_batch, caplog):
        trainer.val_episode_fraction = 0.5
        _train_batch, val_batch = trainer._split_train_val_episodes(multi_track_batch)
        assert val_batch is not None

        clip = trainer.schedules.clip(0.0)
        with caplog.at_level("INFO"):
            _pol, val_loss, _kl, _per_sample, val_loss_trimmed, pol_loss_trimmed = (
                trainer._eval_val_episode_losses(val_batch, clip)
            )
        assert math.isfinite(val_loss_trimmed)
        assert val_loss_trimmed >= 0.0, "trimmed mean of non-negative squared-error terms can't be negative"
        assert math.isfinite(pol_loss_trimmed)

        with caplog.at_level("INFO"):
            trainer._ppo_update(multi_track_batch, progress=0.0)
        val_loss_lines = [r.message for r in caplog.records if "val_episode_value_loss_p1_p99=" in r.message]
        pol_loss_lines = [r.message for r in caplog.records if "val_episode_policy_loss_p1_p99=" in r.message]
        assert len(val_loss_lines) >= 1, "val_episode_value_loss_p1_p99 must appear in the per-epoch log line"
        assert len(pol_loss_lines) >= 1, "val_episode_policy_loss_p1_p99 must appear in the per-epoch log line"
        for line in val_loss_lines:
            val = float(line.split("val_episode_value_loss_p1_p99=")[1].split("(")[0])
            assert math.isfinite(val)
            assert val >= 0.0
        for line in pol_loss_lines:
            val = float(line.split("val_episode_policy_loss_p1_p99=")[1].split()[0])
            assert math.isfinite(val)


class TestTrainSideMeansTrimmed:
    """p1-p99 trimmed companions on the TRAIN-side "[epoch N/M] pol_mean=...
    val_mean=..." summary line -- pol_mean_p1_p99 (policy, sign-mixed) and
    val_mean_p1_p99 (value head, squared-error terms so >= 0). Distinct from
    val_episode_policy_loss_p1_p99/val_episode_value_loss_p1_p99 (the
    HELD-OUT set's own trimmed losses -- "val" is overloaded in this file
    between "value head" and "held-out validation set", see
    _epoch_sample_value_loss's own comment in ppo_trainer.py)."""

    def test_logged_every_epoch_finite(self, trainer, multi_track_batch, caplog):
        with caplog.at_level("INFO"):
            trainer._ppo_update(multi_track_batch, progress=0.0)

        n_epoch_lines = sum(
            1 for r in caplog.records if r.message.startswith("  [epoch ") and "pol_mean=" in r.message
        )
        assert n_epoch_lines >= 1, "no epoch ran -- test is vacuous"

        pol_lines = [r.message for r in caplog.records if "pol_mean_p1_p99=" in r.message]
        val_lines = [r.message for r in caplog.records if "val_mean_p1_p99=" in r.message]
        assert len(pol_lines) == n_epoch_lines, (
            "pol_mean_p1_p99 should appear in every epoch's summary line, "
            f"got {len(pol_lines)} for {n_epoch_lines} epoch(s)"
        )
        assert len(val_lines) == n_epoch_lines, (
            "val_mean_p1_p99 should appear in every epoch's summary line, "
            f"got {len(val_lines)} for {n_epoch_lines} epoch(s)"
        )
        for line in pol_lines:
            val = float(line.split("pol_mean_p1_p99=")[1].split()[0])
            assert math.isfinite(val)
        for line in val_lines:
            val = float(line.split("val_mean_p1_p99=")[1].split("(")[0])
            assert math.isfinite(val)
            assert val >= 0.0, "trimmed mean of non-negative squared-error terms can't be negative"

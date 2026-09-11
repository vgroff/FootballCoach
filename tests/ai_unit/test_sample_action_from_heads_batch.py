"""Hard-guarantee tests for ``PPOTrainer._sample_action_from_heads_batch`` --
the vectorized tail added to make batched rollout collection actually fast
(see ai/ppo/batched_rollout_worker.py and the throughput investigation in
training_runs.md: the network forward pass was already batched this session,
but the per-row sampling/log_prob/decanonicalization tail was not, and
profiled at ~53% of decision time despite batching -- this file's job is to
make sure vectorizing that tail didn't silently change what it computes).

Unlike test_sample_action_batch.py (which exercises the vectorized tail
through REAL network forward passes, comparing against a per-row reference
built from the SAME code), this file hand-constructs synthetic
DecisionHeadsRaw/ExecutionHeadsRaw batches so individual rows can be forced
into specific gating states (exec_move=True/False x kick=True/False, frozen
vs unfrozen heads, differing x_sign) that would be impractical to hit
reliably by sampling real network outputs. This targets exactly the
highest-risk part of the vectorization flagged before writing it: the
per-row CONDITIONAL masking (sprint/move_dir gated by exec_move, kick_dir/
kick_power/kick_spin gated by kick, frozen-head masking) that got converted
from Python `if` statements to `_recompute_log_prob`/`_per_head_new_log_probs`'s
tensor-mask multiplication -- a bug there wouldn't crash, it would silently
corrupt the PPO log_prob/loss for exactly the rows in the "wrong" gating
state, which is the worst kind of bug to ship (it trains, just wrong).

HEAD_LP_KEYS index order (matches _per_head_new_log_probs's stack and the
old per-row head_log_probs array exactly -- verified by reading both side by
side before writing this file):
  0 shoot, 1 pass_, 2 move, 3 tackle, 4 get_possession_extra, 5 mark,
  6 hold_position, 7 exec_move, 8 sprint, 9 kick, 10 tackle_attempt,
  11 move_dir, 12 kick_dir, 13 kick_power, 14 kick_spin
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from footballcoach.ai.action.schema import DecisionHeadsRaw, ExecutionHeadsRaw
from footballcoach.ai.obs.schema import MAX_OTHER_PLAYERS
from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _slice_dataclass_row

_ATOL = 1e-4

# HEAD_LP_KEYS indices, named for readability in the tests below.
_IDX_SHOOT, _IDX_PASS, _IDX_MOVE, _IDX_TACKLE = 0, 1, 2, 3
_IDX_GP_EXTRA, _IDX_MARK, _IDX_HOLD = 4, 5, 6
_IDX_EXEC_MOVE, _IDX_SPRINT, _IDX_KICK, _IDX_TACKLE_ATTEMPT = 7, 8, 9, 10
_IDX_MOVE_DIR, _IDX_KICK_DIR, _IDX_KICK_POWER, _IDX_KICK_SPIN = 11, 12, 13, 14


@pytest.fixture(scope="module")
def trainer():
    t = PPOTrainer.from_config(device=torch.device("cpu"), inference_only=True, separate_value_net=True)
    t.decision_net.eval()
    t.execution_net.eval()
    t.value_net.eval()
    return t


def _make_heads_batch(
    K: int, *, exec_move_logits, kick_logits, seed: int, max_other: int = MAX_OTHER_PLAYERS,
    latent_dim: int = 8,
):
    """Synthetic (batch=K) DecisionHeadsRaw/ExecutionHeadsRaw with
    controlled exec_move_logit/kick_logit per row (everything else random
    but reproducible) -- lets tests force specific gating combinations that
    would be impractical to hit reliably through real network sampling."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g) * 0.5

    d_heads = DecisionHeadsRaw(
        shoot_logit=r(K, 1), pass_logit=r(K, 1), move_logit=r(K, 1), tackle_logit=r(K, 1),
        get_possession_raw=r(K, 1), mark_logit=r(K, 1), hold_position_logit=r(K, 1),
        pass_target_logits=r(K, max_other), tackle_target_logits=r(K, max_other),
        mark_target_logits=r(K, max_other),
        move_region_center=r(K, 2), move_region_size=r(K, 1), move_arrival_speed=r(K, 1),
        region_of_play_center=r(K, 2), region_of_play_size=r(K, 1), attack_defence_raw=r(K, 1),
        latent_vector=r(K, latent_dim), value=r(K, 1),
    )
    e_heads = ExecutionHeadsRaw(
        move_direction=r(K, 2), move_direction_unnormalized=r(K, 2),
        exec_move_logit=torch.tensor(exec_move_logits, dtype=torch.float32).view(K, 1),
        sprint_logit=r(K, 1),
        kick_logit=torch.tensor(kick_logits, dtype=torch.float32).view(K, 1),
        kick_direction=r(K, 3), kick_direction_unnormalized=r(K, 3),
        kick_power=r(K, 1), kick_spin=r(K, 3), tackle_attempt_logit=r(K, 1),
        value=r(K, 1),
    )
    value_batch = r(K, 1).squeeze(-1)
    x_sign_batch = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(K)])
    exists_mask = torch.ones(K, max_other)
    return d_heads, e_heads, value_batch, x_sign_batch, exists_mask


def _repeat_row0(obj, K: int):
    """Broadcast row 0 of a dataclass-of-tensors to K identical rows --
    used to isolate x_sign as the ONLY thing differing between rows."""
    kwargs = {}
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        kwargs[f.name] = None if val is None else val[0:1].expand(K, *val.shape[1:]).clone()
    return type(obj)(**kwargs)


def _reference_rows(trainer, d_heads, e_heads, value_b, x_sign_b, em, **kwargs):
    """The per-row ground truth: K individual calls to the OLD, unchanged
    _sample_action_from_heads -- what _sample_action_batch called before
    this vectorization existed."""
    K = int(value_b.shape[0])
    return [
        trainer._sample_action_from_heads(
            _slice_dataclass_row(d_heads, i), _slice_dataclass_row(e_heads, i),
            float(value_b[i]), float(x_sign_b[i]), em[i:i + 1], **kwargs,
        )
        for i in range(K)
    ]


def _assert_field_close(a, b, label: str) -> None:
    if isinstance(a, np.ndarray):
        np.testing.assert_allclose(a, b, atol=_ATOL, err_msg=label)
    elif isinstance(a, bool):
        assert a == b, label
    elif isinstance(a, (int, float)):
        assert a == pytest.approx(b, abs=_ATOL), label
    else:
        assert a == b, label


class TestGatingMatrixMatchesReference:
    """All 4 combinations of exec_move x kick, deterministic mode, compared
    row-by-row against the OLD per-row _sample_action_from_heads -- the
    ground truth this vectorization must never diverge from."""

    def test_all_four_gating_combinations(self, trainer):
        K = 4
        exec_move_logits = [10.0, 10.0, -10.0, -10.0]
        kick_logits = [10.0, -10.0, 10.0, -10.0]
        d_heads, e_heads, value_b, x_sign_b, em = _make_heads_batch(
            K, exec_move_logits=exec_move_logits, kick_logits=kick_logits, seed=42,
        )

        # deterministic=True (not just deterministic_decision): the
        # continuous direction/kick_power/kick_spin heads still SAMPLE
        # stochastically under deterministic_decision alone, and a batched
        # .sample() call consumes the global RNG stream in a different
        # order than K sequential per-row .sample() calls (true in general
        # for elementwise ops, and unavoidable for VonMises' rejection
        # sampler specifically -- see TestStochasticLogProbSelfConsistency's
        # docstring) -- so those heads would legitimately draw DIFFERENT
        # values here and make this an apples-to-oranges log_prob
        # comparison. Full determinism removes all sampling from the
        # picture, isolating exactly what this test wants to check: the
        # masking/gating/decanonicalization logic.
        batched = trainer._sample_action_from_heads_batch(
            d_heads, e_heads, value_b, x_sign_b, em, deterministic=True,
        )
        reference = _reference_rows(
            trainer, d_heads, e_heads, value_b, x_sign_b, em, deterministic=True,
        )

        for i in range(K):
            (ref_action, ref_lp, ref_value, ref_probs, ref_exec,
             ref_dec, ref_tgt, ref_raw, ref_hlp) = reference[i]
            (bat_action, bat_lp, bat_value, bat_probs, bat_exec,
             bat_dec, bat_tgt, bat_raw, bat_hlp) = batched[i]

            # Sanity: confirm the forced gating actually landed where intended,
            # so this test can't pass vacuously on a row that never exercised
            # the combination its logits were set up for.
            assert bat_exec["exec_move"] == (exec_move_logits[i] > 0), f"row {i} exec_move setup"
            assert bat_exec["kick_this_tick"] == (kick_logits[i] > 0), f"row {i} kick setup"

            assert ref_lp == pytest.approx(bat_lp, abs=_ATOL), f"row {i} total log_prob"
            assert ref_value == pytest.approx(bat_value, abs=_ATOL), f"row {i} value"
            np.testing.assert_allclose(ref_hlp, bat_hlp, atol=_ATOL, err_msg=f"row {i} head_log_probs")
            for f in dataclasses.fields(ref_action):
                _assert_field_close(
                    getattr(ref_action, f.name), getattr(bat_action, f.name), f"row {i} action.{f.name}",
                )
            for k in ref_exec:
                _assert_field_close(ref_exec[k], bat_exec[k], f"row {i} execution_physical[{k}]")
            for k in ref_dec:
                _assert_field_close(ref_dec[k], bat_dec[k], f"row {i} decision_physical[{k}]")
            for k in ref_raw:
                np.testing.assert_allclose(ref_raw[k], bat_raw[k], atol=_ATOL, err_msg=f"row {i} raw_exec[{k}]")

    def test_gated_heads_are_exactly_zero_not_near_zero(self, trainer):
        """When exec_move=False, sprint/move_dir head_log_probs must be
        EXACTLY 0.0 (not a small nonzero float from a masking-order bug) --
        same for kick_dir/kick_power/kick_spin when kick=False."""
        K = 2
        d_heads, e_heads, value_b, x_sign_b, em = _make_heads_batch(
            K, exec_move_logits=[-10.0, -10.0], kick_logits=[-10.0, -10.0], seed=43,
        )
        batched = trainer._sample_action_from_heads_batch(
            d_heads, e_heads, value_b, x_sign_b, em, deterministic_decision=True,
        )
        for i in range(K):
            hlp = batched[i][8]
            assert hlp[_IDX_SPRINT] == 0.0, f"row {i} sprint should be exactly 0 (exec_move=False)"
            assert hlp[_IDX_MOVE_DIR] == 0.0, f"row {i} move_dir should be exactly 0 (exec_move=False)"
            assert hlp[_IDX_KICK_DIR] == 0.0, f"row {i} kick_dir should be exactly 0 (kick=False)"
            assert hlp[_IDX_KICK_POWER] == 0.0, f"row {i} kick_power should be exactly 0 (kick=False)"

    def test_active_heads_are_nonzero(self, trainer):
        """The flip side of the above -- when exec_move/kick ARE active, the
        gated heads must actually contribute (rules out a masking bug that
        zeroes everything unconditionally, which the previous test alone
        couldn't catch)."""
        K = 1
        d_heads, e_heads, value_b, x_sign_b, em = _make_heads_batch(
            K, exec_move_logits=[10.0], kick_logits=[10.0], seed=44,
        )
        batched = trainer._sample_action_from_heads_batch(
            d_heads, e_heads, value_b, x_sign_b, em, deterministic_decision=True,
        )
        hlp = batched[0][8]
        assert hlp[_IDX_SPRINT] != 0.0
        assert hlp[_IDX_MOVE_DIR] != 0.0
        assert hlp[_IDX_KICK_DIR] != 0.0
        assert hlp[_IDX_KICK_POWER] != 0.0


class TestFrozenHeadsMasking:
    def test_frozen_heads_zero_across_the_batch_and_match_reference(self, trainer):
        K = 3
        d_heads, e_heads, value_b, x_sign_b, em = _make_heads_batch(
            K, exec_move_logits=[10.0] * K, kick_logits=[10.0] * K, seed=45,
        )
        trainer.set_frozen_heads(["shoot_logit", "mark_logit"])
        try:
            # deterministic=True, not deterministic_decision -- see the
            # comment in TestGatingMatrixMatchesReference.test_all_four_gating_combinations
            # for why an exact log_prob comparison needs full determinism.
            batched = trainer._sample_action_from_heads_batch(
                d_heads, e_heads, value_b, x_sign_b, em, deterministic=True,
            )
            reference = _reference_rows(
                trainer, d_heads, e_heads, value_b, x_sign_b, em, deterministic=True,
            )
            for i in range(K):
                hlp = batched[i][8]
                assert hlp[_IDX_SHOOT] == 0.0, f"row {i} shoot should be masked to 0 (frozen)"
                assert hlp[_IDX_MARK] == 0.0, f"row {i} mark should be masked to 0 (frozen)"
                assert batched[i][1] == pytest.approx(reference[i][1], abs=_ATOL), (
                    f"row {i} total log_prob vs per-row reference under frozen heads"
                )
                np.testing.assert_allclose(reference[i][8], hlp, atol=_ATOL, err_msg=f"row {i} head_log_probs")
        finally:
            # Un-freeze so this doesn't leak into other tests sharing `trainer`.
            for name in ("shoot_logit", "mark_logit"):
                for p in getattr(trainer.decision_net, name).parameters():
                    p.requires_grad_(True)
            assert not trainer._ppo_lp_masked_heads, "cleanup failed to unfreeze heads"


class TestPerRowDecanonicalization:
    def test_x_sign_applied_per_row_not_broadcast(self, trainer):
        """Two rows with IDENTICAL heads except x_sign differs -- outputs
        must mirror independently (row 0's x-component = -row 1's,
        y/z unchanged), never both mirrored the same way (which would
        happen if x_sign were accidentally collapsed to a scalar somewhere,
        e.g. only x_sign_batch[0] used for the whole batch)."""
        K = 2
        d_heads, e_heads, value_b, _, em = _make_heads_batch(
            K, exec_move_logits=[10.0, 10.0], kick_logits=[10.0, 10.0], seed=46,
        )
        d_heads = _repeat_row0(d_heads, K)
        e_heads = _repeat_row0(e_heads, K)
        x_sign_b = torch.tensor([1.0, -1.0])

        batched = trainer._sample_action_from_heads_batch(
            d_heads, e_heads, value_b, x_sign_b, em, deterministic=True,
        )
        exec0, exec1 = batched[0][4], batched[1][4]
        np.testing.assert_allclose(exec0["move_direction"][0], -exec1["move_direction"][0], atol=1e-5)
        np.testing.assert_allclose(exec0["move_direction"][1:], exec1["move_direction"][1:], atol=1e-5)
        np.testing.assert_allclose(exec0["kick_direction"][0], -exec1["kick_direction"][0], atol=1e-5)
        np.testing.assert_allclose(exec0["kick_direction"][1:], exec1["kick_direction"][1:], atol=1e-5)

        dec0, dec1 = batched[0][5], batched[1][5]
        np.testing.assert_allclose(dec0["move_region_center_m"][0], -dec1["move_region_center_m"][0], atol=1e-5)
        np.testing.assert_allclose(dec0["move_region_center_m"][1], dec1["move_region_center_m"][1], atol=1e-5)

        # And they must match the per-row reference exactly, not just be
        # mutually consistent with each other.
        reference = _reference_rows(trainer, d_heads, e_heads, value_b, x_sign_b, em, deterministic=True)
        for i in range(K):
            np.testing.assert_allclose(
                reference[i][4]["move_direction"], batched[i][4]["move_direction"], atol=_ATOL,
                err_msg=f"row {i} move_direction vs reference",
            )
            np.testing.assert_allclose(
                reference[i][5]["move_region_center_m"], batched[i][5]["move_region_center_m"], atol=_ATOL,
                err_msg=f"row {i} move_region_center_m vs reference",
            )


class TestStochasticLogProbSelfConsistency:
    def test_reported_log_prob_matches_recompute_from_returned_raw_samples(self, trainer):
        """For REAL (non-deterministic) sampling, feed each row's own
        returned raw samples back into the single-row reference's
        _compute_log_prob and confirm it matches the batched log_prob
        reported for that row. This catches "vectorized code computed
        log_prob against the wrong row's params/samples" bugs WITHOUT
        needing RNG-stream equivalence between batched and per-row sampling
        (which is not guaranteed for VonMises rejection sampling -- a
        batched accept/reject loop draws a different number of underlying
        random values, in a different order, than K independent per-row
        rejection loops, even from the same seed)."""
        K = 6
        d_heads, e_heads, value_b, x_sign_b, em = _make_heads_batch(
            K,
            exec_move_logits=[8.0, -8.0, 8.0, -8.0, 8.0, -8.0],
            kick_logits=[8.0, 8.0, -8.0, -8.0, 8.0, -8.0],
            seed=47,
        )
        batched = trainer._sample_action_from_heads_batch(d_heads, e_heads, value_b, x_sign_b, em)

        for i in range(K):
            action, lp, value, probs, exec_phys, dec_phys, tgt, raw, hlp = batched[i]
            samples = {
                "shoot": torch.tensor([[action.shoot]]), "pass_": torch.tensor([[action.pass_]]),
                "move": torch.tensor([[action.move]]), "tackle": torch.tensor([[action.tackle]]),
                "gp_extra": torch.tensor([[action.get_possession_extra]]),
                "mark": torch.tensor([[action.mark]]), "hold": torch.tensor([[action.hold_position]]),
                "pass_tgt": torch.tensor([action.pass_target]), "tackle_tgt": torch.tensor([action.tackle_target]),
                "mark_tgt": torch.tensor([action.mark_target]),
                "exec_move": torch.tensor(raw["exec_move"]).view(1, 1),
                "sprint": torch.tensor(raw["sprint"]).view(1, 1),
                "kick": torch.tensor(raw["kick"]).view(1, 1),
                "tackle_attempt": torch.tensor(raw["tackle_attempt"]).view(1, 1),
                "move_dir_raw": torch.tensor(raw["move_dir_raw"]).view(1, 2),
                "kick_dir_raw": torch.tensor(raw["kick_dir_raw"]).view(1, 3),
                "kick_power_raw": torch.tensor(raw["kick_power_raw"]).view(1, 1),
                "kick_spin_raw": torch.tensor(raw["kick_spin_raw"]).view(1, 3),
            }
            row_d = _slice_dataclass_row(d_heads, i)
            row_e = _slice_dataclass_row(e_heads, i)
            with torch.no_grad():
                recomputed_lp = float(trainer._compute_log_prob(row_d, row_e, samples, em[i:i + 1]))
            assert lp == pytest.approx(recomputed_lp, abs=_ATOL), f"row {i} log_prob self-consistency"


class TestRandomizedFuzz:
    """Broad-spectrum sweep with realistic (non-extreme) random logits --
    catches anything the hand-picked edge cases above might miss, e.g.
    boundary behavior near sigmoid(logit)==0.5, or interactions between
    frozen heads and gating happening simultaneously."""

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_matches_reference_across_random_batches(self, trainer, seed):
        K = 8
        g = torch.Generator().manual_seed(seed)
        exec_move_logits = (torch.randn(K, generator=g) * 3).tolist()
        kick_logits = (torch.randn(K, generator=g) * 3).tolist()
        d_heads, e_heads, value_b, x_sign_b, em = _make_heads_batch(
            K, exec_move_logits=exec_move_logits, kick_logits=kick_logits, seed=seed + 1000,
        )
        # deterministic=True -- see TestGatingMatrixMatchesReference's comment
        # for why an exact log_prob comparison needs full determinism, not
        # just deterministic_decision.
        batched = trainer._sample_action_from_heads_batch(
            d_heads, e_heads, value_b, x_sign_b, em, deterministic=True,
        )
        reference = _reference_rows(
            trainer, d_heads, e_heads, value_b, x_sign_b, em, deterministic=True,
        )
        for i in range(K):
            assert reference[i][1] == pytest.approx(batched[i][1], abs=_ATOL), f"seed {seed} row {i} log_prob"
            np.testing.assert_allclose(
                reference[i][8], batched[i][8], atol=_ATOL, err_msg=f"seed {seed} row {i} head_log_probs",
            )

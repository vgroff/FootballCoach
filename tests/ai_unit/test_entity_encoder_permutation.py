"""Regression test guarding the entity encoder's permutation-invariance
guarantee: which of the MAX_OTHER_PLAYERS slots the real opponent occupies
in other_feat/exists_mask must never affect the heads that actually drive
in-game behavior -- not just value.

Context: ScenarioEnv randomly shuffles which slot the opponent lands in on
every decision (ai/obs/encoder.py's rng.sample() call), relying entirely on
EntityEncoder's shared-per-slot-MLP + masked-attention-pool design (see its
own docstring) to make that shuffle behaviorally inert. This was verified
empirically session-side (2026-10-08): running identical seeded episodes
with the opponent-slot rng left deliberately unseeded still reproduced
byte-identical full trajectories (every ball position and value prediction)
under a deterministic policy. test_ai_type_side_channel.py's own
TestValueSideChannelPermutationInvariance already guards this for the
value-only ai-type side channel specifically (and documents WHY it matters:
that side channel's PREDECESSOR implementation, a flatten+Linear, was NOT
permutation-invariant, and slot-shuffling was the training-time workaround
for that bug). This file extends the same guarantee to the main entity
encoder feeding every POLICY head, which that file doesn't cover.

Three different categories of output exist here, found by writing this test
(the target-logit non-invariance below was NOT previously known/documented):

  1. INVARIANT fields (shoot_logit, kick_direction, kick_power, value,
     latent_vector, etc.) -- must be EXACTLY equal regardless of slot, since
     they describe something about the whole scene/trunk, not one candidate.

  2. EQUIVARIANT fields (other_physics_full -- a per-slot frozen-encoder
     passthrough, shape (batch, MAX_OTHER_PLAYERS, dim)) -- moving the real
     player to a different slot correctly moves this value to that slot too
     (it's computed per-slot by a SHARED encoder, no cross-slot mixing), so
     the real invariant is "the value AT the real player's slot", checked by
     gathering that slot in each variant before comparing.

  3. NOT invariant, by construction -- a real, previously-undocumented
     finding: pass_target_logits/tackle_target_logits/mark_target_logits are
     plain ``nn.Linear(trunk_hidden, MAX_OTHER_PLAYERS)`` (decision_network.py),
     i.e. output index 5 and output index 17 have completely independent
     learned weight rows -- NOT a pointer/attention mechanism over per-slot
     embeddings like EntityEncoder's own pooling. So the raw logit "for
     whichever player is in slot 5" is a genuinely different number than
     "for whichever player is in slot 17", even for the identical player.
     This is the same anti-pattern the value side-channel used to have
     (see its own docstring) -- just not yet fixed here.

     Confirmed CURRENTLY behaviorally inert for every phase these heads are
     actually used in today (always exactly one or zero unmasked
     candidates): MaskedCategorical's mask sets every invalid slot's
     probability to exactly 0 regardless of its raw logit (softmax of
     [L_valid, -inf, -inf, ...] = probability 1.0 on the one valid slot no
     matter what L_valid's value is), so sample/log_prob/entropy are
     unaffected by this non-invariance in a single-candidate scenario.
     TestTargetLogitsKnownNonInvariance below formally pins both halves of
     this: the raw logits DO differ by slot (so a future regression-fix
     doesn't silently go unnoticed), and the resulting masked distribution
     is nonetheless degenerate/inert in the single-candidate case. This
     would NOT hold in a future multi-candidate phase (2+ unmasked slots,
     e.g. full 11v11 team play) -- flagged as a known latent issue, not
     exercised by any current phase's tests.
"""
import torch

from footballcoach.ai.obs.schema import (
    AI_TYPE_ONE_HOT_DIM,
    BALL_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    MAX_OTHER_PLAYERS,
    PLAYER_FEATURE_DIM,
)
from footballcoach.ai.models.decision_network import DecisionNetwork
from footballcoach.ai.models.execution_network import ExecutionNetwork

# Category 3 above -- known NOT invariant, excluded from the blanket check
# and tested separately (TestTargetLogitsKnownNonInvariance).
_KNOWN_NON_INVARIANT_FIELDS = {"pass_target_logits", "tackle_target_logits", "mark_target_logits"}
# Category 2 above -- per-slot passthrough, EQUIVARIANT (gather at the real slot).
_PER_SLOT_PASSTHROUGH_FIELDS = {"other_physics_full"}


def _build_decision_inputs(slot: int, batch_size: int = 1, seed: int = 0):
    torch.manual_seed(seed)
    self_feat = torch.randn(batch_size, PLAYER_FEATURE_DIM)
    ball_feat = torch.randn(batch_size, BALL_FEATURE_DIM)
    global_feat = torch.randn(batch_size, GLOBAL_FEATURE_DIM)
    self_ai_type = torch.zeros(batch_size, AI_TYPE_ONE_HOT_DIM)
    self_ai_type[:, 2] = 1.0  # self is neural

    real_player_feat = torch.randn(PLAYER_FEATURE_DIM)
    real_player_ai_type = torch.tensor([1.0, 0.0, 0.0])  # is_rules

    other_feat = torch.zeros(batch_size, MAX_OTHER_PLAYERS, PLAYER_FEATURE_DIM)
    exists_mask = torch.zeros(batch_size, MAX_OTHER_PLAYERS)
    other_ai_type = torch.zeros(batch_size, MAX_OTHER_PLAYERS, AI_TYPE_ONE_HOT_DIM)
    other_feat[0, slot] = real_player_feat
    exists_mask[0, slot] = 1.0
    other_ai_type[0, slot] = real_player_ai_type
    return self_feat, other_feat, exists_mask, ball_feat, global_feat, self_ai_type, other_ai_type


def _assert_heads_match(heads_a, heads_b, slot_a: int, slot_b: int):
    mismatches = []
    for f in heads_a.__dataclass_fields__:
        if f in _KNOWN_NON_INVARIANT_FIELDS:
            continue
        a = getattr(heads_a, f)
        b = getattr(heads_b, f)
        if a is None and b is None:
            continue  # e.g. physics-encoder passthrough fields, disabled by default
        if f in _PER_SLOT_PASSTHROUGH_FIELDS:
            a_cmp, b_cmp = a[:, slot_a], b[:, slot_b]
        else:
            a_cmp, b_cmp = a, b
        if not torch.allclose(a_cmp, b_cmp, atol=1e-5):
            mismatches.append(f)
    assert not mismatches, (
        f"These outputs changed when the real player moved from slot {slot_a} "
        f"to slot {slot_b} (should be permutation-invariant/equivariant): {mismatches}"
    )


class TestDecisionNetworkPermutationInvariance:
    def test_all_heads_unchanged_when_real_player_moved_to_different_slot(self):
        net = DecisionNetwork.from_config()
        net.eval()
        with torch.no_grad():
            heads_a = net(*_build_decision_inputs(slot=0))
            heads_b = net(*_build_decision_inputs(slot=17))
        _assert_heads_match(heads_a, heads_b, slot_a=0, slot_b=17)

    def test_all_heads_unchanged_across_every_slot(self):
        """Broader sweep than slot 0 vs 17 -- every slot must agree with slot 0."""
        net = DecisionNetwork.from_config()
        net.eval()
        with torch.no_grad():
            heads_ref = net(*_build_decision_inputs(slot=0))
            for slot in (1, 5, 10, 20, MAX_OTHER_PLAYERS - 1):
                heads_s = net(*_build_decision_inputs(slot=slot))
                _assert_heads_match(heads_ref, heads_s, slot_a=0, slot_b=slot)


class TestExecutionNetworkPermutationInvariance:
    def test_all_heads_unchanged_when_real_player_moved_to_different_slot(self):
        d_net = DecisionNetwork.from_config()
        e_net = ExecutionNetwork.from_config()
        d_net.eval()
        e_net.eval()
        with torch.no_grad():
            self_feat, other_feat_a, exists_mask_a, ball_feat, global_feat, self_ai_type, other_ai_type_a = (
                _build_decision_inputs(slot=0)
            )
            d_heads_a = d_net(self_feat, other_feat_a, exists_mask_a, ball_feat, global_feat, self_ai_type, other_ai_type_a)
            e_heads_a = e_net(
                self_feat, other_feat_a, exists_mask_a, ball_feat, global_feat, d_heads_a,
                self_ai_type, other_ai_type_a,
            )

            _, other_feat_b, exists_mask_b, _, _, _, other_ai_type_b = _build_decision_inputs(slot=17)
            d_heads_b = d_net(self_feat, other_feat_b, exists_mask_b, ball_feat, global_feat, self_ai_type, other_ai_type_b)
            e_heads_b = e_net(
                self_feat, other_feat_b, exists_mask_b, ball_feat, global_feat, d_heads_b,
                self_ai_type, other_ai_type_b,
            )
        # ExecutionHeadsRaw has no per-candidate (MAX_OTHER_PLAYERS-shaped)
        # fields at all -- every field here must be an exact invariant.
        _assert_heads_match(e_heads_a, e_heads_b, slot_a=0, slot_b=17)


class TestTargetLogitsKnownNonInvariance:
    """Pins down the category-3 finding from this file's module docstring:
    pass_target_logits/tackle_target_logits/mark_target_logits are NOT
    permutation-invariant (plain per-slot-indexed nn.Linear, not a pointer/
    attention mechanism) but are currently behaviorally inert because every
    phase that uses them today only ever has one real candidate."""

    def test_raw_target_logits_do_differ_by_slot(self):
        """Documents the actual non-invariance, so a future fix (making
        these heads permutation-equivariant) is noticed here as an
        intentional change, not an unexplained test failure."""
        net = DecisionNetwork.from_config()
        net.eval()
        with torch.no_grad():
            heads_a = net(*_build_decision_inputs(slot=0))
            heads_b = net(*_build_decision_inputs(slot=17))
        for f in ("pass_target_logits", "tackle_target_logits", "mark_target_logits"):
            a = getattr(heads_a, f)[:, 0]
            b = getattr(heads_b, f)[:, 17]
            assert not torch.allclose(a, b, atol=1e-4), (
                f"{f} no longer differs by slot -- if this head was made "
                "permutation-equivariant, update this test's expectation "
                "(and consider deleting it / folding it into the main "
                "invariance check above)."
            )

    def test_masked_categorical_is_inert_to_the_non_invariance_with_one_candidate(self):
        """The thing that actually matters for current behavior: with
        exactly one unmasked slot, MaskedCategorical must put probability
        1.0 there regardless of that slot's raw (slot-dependent) logit
        value -- sample/log_prob/entropy must not depend on WHICH slot the
        real candidate occupies, only on the fact that it's the only valid
        one."""
        from footballcoach.ai.action.distributions import MaskedCategorical

        net = DecisionNetwork.from_config()
        net.eval()
        with torch.no_grad():
            for slot in (0, 5, 20):
                _, _, exists_mask, _, _, _, _ = _build_decision_inputs(slot=slot)
                heads = net(*_build_decision_inputs(slot=slot))
                dist = MaskedCategorical(heads.pass_target_logits, exists_mask)
                probs = dist.probs()
                assert torch.allclose(probs[0, slot], torch.tensor(1.0), atol=1e-6)
                assert torch.isclose(dist.entropy()[0], torch.tensor(0.0), atol=1e-6)

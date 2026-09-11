"""Direct unit coverage for ScenarioEnv._possession_transition_step() --
previously had ZERO dedicated tests despite backing the "poss" reward
component and possession-based progress gating for BOTH the trainee and
every secondary player (self-play doubles the exposure: it's called once
per player per tick, against the SAME match.ball.possessed_by ground
truth, with each player's own independent counters).

The method exists specifically to distinguish a REAL turnover (possession
settles onto a DIFFERENT player) from a harmless momentary loose ball (a
push-kick dribble touch, or a player knocking the ball loose and
immediately re-collecting it themselves) -- see its own docstring. That
mid-interval nuance (gain -> lose -> regain, or lose -> loose -> settle
elsewhere, all within a single decision interval) is exactly the kind of
thing a naive before/after comparison would get wrong, and exactly the
kind of thing no existing test exercises directly for either player.

TestPossessionTransitionStep exercises the pure staticmethod directly
(every branch its own docstring describes). TestSymmetricTwoPlayerSequence
mirrors scenario_env.py's ACTUAL calling pattern -- one independent call
per player per tick, both fed the same possessed_by ground truth -- and
proves a direct tackle between two players produces EXACTLY matched
gain/loss counts on the same tick, the property self-play's reward
attribution (see test_self_play_reward_attribution.py) implicitly depends
on but never verified at this level.
"""
from __future__ import annotations

from footballcoach.ai.env.scenario_env import ScenarioEnv

_step = ScenarioEnv._possession_transition_step


class TestPossessionTransitionStep:
    def test_simple_gain_from_loose_ball(self):
        poss_prev, pending_loss, gained, lost = _step("p1", "p1", False, False, 0, 0)
        assert poss_prev is True
        assert pending_loss is False
        assert gained == 1
        assert lost == 0

    def test_holding_possession_does_not_recount_as_a_gain(self):
        poss_prev, pending_loss, gained, lost = _step("p1", "p1", True, False, 0, 0)
        assert poss_prev is True
        assert gained == 0, "already holding the ball -- must not increment gained_count again"
        assert lost == 0

    def test_not_possessing_and_ball_elsewhere_is_a_noop(self):
        poss_prev, pending_loss, gained, lost = _step("p1", "p2", False, False, 0, 0)
        assert poss_prev is False
        assert pending_loss is False
        assert gained == 0
        assert lost == 0

    def test_direct_loss_to_another_player_same_tick_counts_immediately(self):
        """p1 had it; this tick p2 (a different, concrete player) has it --
        a confirmed turnover, counted the SAME tick, no deferral needed."""
        poss_prev, pending_loss, gained, lost = _step("p1", "p2", True, False, 0, 0)
        assert poss_prev is False
        assert pending_loss is False
        assert lost == 1

    def test_loss_to_loose_ball_defers_counting(self):
        """p1 had it; this tick nobody does (loose ball) -- not yet a
        confirmed turnover, since p1 might immediately re-collect it."""
        poss_prev, pending_loss, gained, lost = _step("p1", None, True, False, 0, 0)
        assert poss_prev is False
        assert pending_loss is True
        assert lost == 0, "must not count as lost until the loose ball actually resolves"

    def test_loose_ball_regained_by_same_player_cancels_silently(self):
        """The exact 'never really lost' case the docstring calls out:
        p1 -> loose (pending) -> p1 again, with no OTHER player ever
        holding it in between. Net effect must be a complete no-op --
        NOT a lost+gained pair, which would wrongly show up as a real
        turnover-and-recovery in the reward/progress accounting."""
        poss_prev, pending_loss, gained, lost = _step("p1", None, True, False, 0, 0)
        assert (poss_prev, pending_loss) == (False, True)
        poss_prev, pending_loss, gained, lost = _step(
            "p1", "p1", poss_prev, pending_loss, gained, lost,
        )
        assert poss_prev is True
        assert pending_loss is False
        assert gained == 0, "regaining a ball that was only ever loose, never taken by anyone else, is not a gain"
        assert lost == 0, "and therefore not a loss either -- net zero across both ticks"

    def test_loose_ball_settling_on_someone_else_confirms_the_turnover(self):
        """p1 -> loose (pending) -> p2 picks it up: NOW it's a confirmed
        turnover, counted on the tick it actually resolves, not the tick
        the ball first went loose."""
        poss_prev, pending_loss, gained, lost = _step("p1", None, True, False, 0, 0)
        assert (poss_prev, pending_loss) == (False, True)
        poss_prev, pending_loss, gained, lost = _step(
            "p1", "p2", poss_prev, pending_loss, gained, lost,
        )
        assert poss_prev is False
        assert pending_loss is False
        assert lost == 1

    def test_loose_ball_still_loose_next_tick_stays_pending(self):
        poss_prev, pending_loss, gained, lost = _step("p1", None, True, False, 0, 0)
        poss_prev, pending_loss, gained, lost = _step(
            "p1", None, poss_prev, pending_loss, gained, lost,
        )
        assert pending_loss is True
        assert lost == 0

    def test_multi_tick_gain_lose_regain_lose_elsewhere_sequence(self):
        """A full, hand-computed sequence spanning every branch, mirroring
        a real tackle-heavy decision interval: gain, direct loss, regain,
        loose, someone else settles it. Final counts must match exactly."""
        poss_prev, pending_loss, gained, lost = False, False, 0, 0
        sequence = [
            "p1",   # tick 1: gains -> gained=1
            "p2",   # tick 2: direct loss to p2 -> lost=1
            "p1",   # tick 3: p1 regains directly -> gained=2
            None,   # tick 4: p1 loses it, loose -> pending, not yet counted
            "p3",   # tick 5: p3 settles it -> confirmed turnover -> lost=2
        ]
        for possessed_by in sequence:
            poss_prev, pending_loss, gained, lost = _step(
                "p1", possessed_by, poss_prev, pending_loss, gained, lost,
            )
        assert gained == 2
        assert lost == 2
        assert poss_prev is False
        assert pending_loss is False


class TestSymmetricTwoPlayerSequence:
    """Mirrors scenario_env.py's REAL calling pattern: _possession_transition_step
    is called ONCE PER PLAYER PER TICK (independently, own counters), all
    fed the SAME match.ball.possessed_by ground truth for that tick -- see
    the trainee/secondary call sites in ScenarioEnv.step(). This is the
    property self-play's reward attribution rests on: a single physical
    event (a tackle) must produce CONSISTENT, exactly-matched signals for
    both players involved, never a double-count or a dropped event on
    either side."""

    def test_direct_tackle_produces_exactly_matched_gain_and_loss(self):
        """trainee holds it, then opponent tackles it away this same tick --
        trainee's own call must show lost=1 and opponent's own call must
        show gained=1, on the SAME tick, with neither side double-counting."""
        trainee_state = (True, False, 0, 0)   # (poss_prev, pending_loss, gained, lost)
        opponent_state = (False, False, 0, 0)

        possessed_by = "opponent"  # ball changes hands to the opponent this tick
        trainee_state = _step("trainee", possessed_by, *trainee_state)
        opponent_state = _step("opponent", possessed_by, *opponent_state)

        _tr_poss_prev, _tr_pending, tr_gained, tr_lost = trainee_state
        _op_poss_prev, _op_pending, op_gained, op_lost = opponent_state
        assert tr_lost == 1 and tr_gained == 0
        assert op_gained == 1 and op_lost == 0

    def test_loose_ball_between_two_players_never_double_counts(self):
        """trainee loses it to a loose ball, opponent doesn't have it
        either yet -- NEITHER side should count anything until the ball
        actually settles on somebody."""
        trainee_state = (True, False, 0, 0)
        opponent_state = (False, False, 0, 0)

        trainee_state = _step("trainee", None, *trainee_state)
        opponent_state = _step("opponent", None, *opponent_state)

        assert trainee_state[1] is True  # pending_loss
        assert trainee_state[3] == 0     # not yet counted as lost
        assert opponent_state[2] == 0    # not counted as gained -- nobody has it yet
        assert opponent_state[3] == 0

        # Next tick: opponent picks up the loose ball.
        trainee_state = _step("trainee", "opponent", *trainee_state)
        opponent_state = _step("opponent", "opponent", *opponent_state)

        assert trainee_state[3] == 1   # NOW confirmed lost
        assert opponent_state[2] == 1  # and gained, same tick, matched 1:1

    def test_ball_bouncing_between_neither_player_never_credits_either(self):
        """A loose ball that settles on a THIRD party (e.g. a goalkeeper,
        or in a >1v1 scenario another secondary player) must not credit
        either the trainee's or this opponent's own counters at all --
        it's simply not their event."""
        trainee_state = (False, False, 0, 0)
        opponent_state = (False, False, 0, 0)

        for possessed_by in [None, "third_party", "third_party"]:
            trainee_state = _step("trainee", possessed_by, *trainee_state)
            opponent_state = _step("opponent", possessed_by, *opponent_state)

        assert trainee_state[2:] == (0, 0)
        assert opponent_state[2:] == (0, 0)

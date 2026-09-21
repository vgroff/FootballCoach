"""Per-player, per-decision-interval record of whether a kick / armed tackle COULD have taken effect.

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

Why this exists: agent_plans/masked_action_training_plan.md. The kick gate, the
kick direction/power heads and the tackle gate are only causally relevant when
the action can actually resolve; training them on every row is pure noise.
Whether it could resolve is only knowable after the decision interval has
played out, so the ENGINE records it (this tracker, fed by hooks in
``Match``/``NeuralPlayerAI``) and the env reads it at the end of the interval.

Definitions (all per decision interval = ticks from one fresh decision of this
player up to, not including, its next one -- or episode end):

* ``kick opportunity`` at tick t: the player held the ball at the moment its
  kick intent would be applied in tick t (start-of-tick possession, evaluated
  in ``NeuralPlayerAI.apply`` on the decision tick and ``_reapply_cached_gating``
  on every later tick), OR it gained the ball through a LOOSE-BALL PICKUP
  during tick t (``Match._update_loose_ball_pickup``). Possession won by a
  tackle is deliberately NOT a kick opportunity: it depends on the player's own
  tackle bit (see the cross-head rule below).
* ``tackle opportunity`` at tick t: an opposing player was the ball carrier
  within tackle-contact range of this player at the point
  ``Match._check_armed_tackles`` runs, with both players available -- the
  exact predicate the armed-tackle resolution uses, evaluated for EVERY
  player whether or not it armed a tackle.
* ``kick_fired``: ``Player.kick_count`` advanced during the interval.
* ``tack_fired``: an ARMED tackle by this player reached
  ``Match._attempt_tackle_contact`` (``Player.tackle_fire_count`` advanced).

Independence rule (I1 in the plan). The masked policy gradient is only unbiased
if "an opportunity existed" does not itself depend on the gate bit being
trained. That holds for the FIRST opportunity tick of an interval (the world is
identical up to it for either bit value) but not afterwards: a fired kick
releases the ball (changes whether an opposing carrier can appear), a won
tackle gives this player the ball (creates a kick opportunity). So with
``cross_head_cutoff`` (the default) only whichever head's opportunity comes
FIRST counts; the other head's later opportunity in the same interval is
dropped. The raw (uncut) markers are kept so the size of that cut is measured.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OpportunityFlags:
    kick_opp: bool          # cut (independence rule applied) -- trains the kick gate
    tack_opp: bool          # cut -- trains the tackle gate
    kick_fired: bool        # kick_count advanced this interval -- trains kick_dir / kick_power
    tack_fired: bool        # armed tackle reached contact this interval
    kick_opp_raw: bool      # uncut, for diagnostics / invariants
    tack_opp_raw: bool


# Keys these flags occupy in a rollout row's raw_exec / the batch's ``action/*`` tensors.
OPP_ACTION_KEYS = (
    "opp_kick", "opp_kick_raw", "fired_kick", "opp_tack", "opp_tack_raw", "fired_tack", "opp_partial",
)


@dataclass
class ActionOpportunity:
    interval_start_tick: int = 0
    kick_count_at_start: int = 0
    tackle_fire_count_at_start: int = 0
    kick_first_tick: int = -1
    tack_first_tick: int = -1

    def begin_interval(self, tick: int, kick_count: int, tackle_fire_count: int) -> None:
        self.interval_start_tick = tick
        self.kick_count_at_start = kick_count
        self.tackle_fire_count_at_start = tackle_fire_count
        self.kick_first_tick = -1
        self.tack_first_tick = -1

    def note_kick_opp(self, tick: int) -> None:
        if self.kick_first_tick < 0:
            self.kick_first_tick = tick

    def note_tack_opp(self, tick: int) -> None:
        if self.tack_first_tick < 0:
            self.tack_first_tick = tick

    def flags(self, kick_count: int, tackle_fire_count: int, cross_head_cutoff: bool = True) -> OpportunityFlags:
        k, t = self.kick_first_tick, self.tack_first_tick
        raw_k, raw_t = k >= 0, t >= 0
        if cross_head_cutoff:
            kick_opp = raw_k and (not raw_t or k <= t)
            tack_opp = raw_t and (not raw_k or t <= k)
        else:
            kick_opp, tack_opp = raw_k, raw_t
        return OpportunityFlags(
            kick_opp=kick_opp,
            tack_opp=tack_opp,
            kick_fired=kick_count > self.kick_count_at_start,
            tack_fired=tackle_fire_count > self.tackle_fire_count_at_start,
            kick_opp_raw=raw_k,
            tack_opp_raw=raw_t,
        )

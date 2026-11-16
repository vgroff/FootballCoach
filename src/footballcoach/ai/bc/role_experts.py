"""Rules-AI experts that label a NEURAL player's states in a role-based phase.

For a neural player whose role has BC switched on, ``RulesExpert`` asks that
role's own rules AI (the one the scenario built for it) what it would do at
each of the player's decisions, and labels that order with
``order_labels.order_label`` (nothing in the real match changes).

Rules AIs remember things between decisions: the order they are carrying out
(a run to a target, a shot), and their own fields (the keeper's close-down
commitment and its cooldown, the attacker's progress check). The expert keeps
that memory -- a SHADOW of the order the rules AI would hold, and the AI
object itself -- and keeps it TRUE TO THE EPISODE AS ACTUALLY PLAYED by the
neural player, not to what the expert itself would have done:

- every decision is made on the real current state (the neural player's
  position, the real ball and opponents);
- the shadow order advances on that real state and is dropped when it
  completes or replaced when it hands over (ShootOrder -> MoveOrder), as in
  real rules play;
- a per-AI reconciler (RECONCILERS) corrects the AI's own memory where the
  neural player did something else: e.g. if the expert keeper decided to
  close down but the neural keeper did not, the expert is not left
  "committed"; if the neural keeper closes down on its own, the expert counts
  as committed; the close-down cooldown starts when the NEURAL keeper stops;
  if the neural attacker kicked, the expert's run/shot plan is over.

What the neural player actually did is read from its own last decision
(``NeuralPlayerAI._last_gating``) and the player's real kick count.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from footballcoach.ai.ppo.bc import BCLabel
    from footballcoach.engine.match import Match
    from footballcoach.entities.player import Player


def _unwrap(ai) -> list:
    """[ai, its wrapped inner AI, ...] (scenario wrappers such as
    _IdleFallbackAI keep the real AI in ``_inner``)."""
    chain = [ai]
    while getattr(chain[-1], "_inner", None) is not None:
        chain.append(chain[-1]._inner)
    return chain


def _neural_selected(player) -> Optional[object]:
    """The SelectedAction of the neural player's last decision, or None."""
    g = getattr(player.ai, "_last_gating", None)
    return getattr(g, "selected", None)


# --- reconcilers: correct a rules AI's memory to what the neural player did ---

def _reconcile_keeper(expert: "RulesExpert", player, match) -> None:
    """StagedGoalkeeperAI: its close-down commitment follows the NEURAL
    keeper's own choice (deciding to win the ball = closing down)."""
    from footballcoach.ai.action.gating import SelectedAction
    from footballcoach.orders import GetPossessionOrder

    ai = expert.core
    actual = _neural_selected(player) is SelectedAction.GET_POSSESSION
    was_actual = expert.state.get("actual_closing_down", False)
    if ai._closing_down and not actual:
        # The expert is committed but the neural keeper is not closing down:
        # drop the commitment. If the neural keeper WAS closing down and has
        # now stopped, that stop starts the cooldown; if it never went (the
        # expert only just decided to), there is nothing to cool down from.
        if was_actual:
            ai._close_down_ended_s = match.time_s
        ai._closing_down = False
        if isinstance(expert.shadow, GetPossessionOrder):
            expert.shadow = None
    elif actual and not ai._closing_down:
        # The neural keeper is closing down on its own: as far as the rules
        # keeper's memory goes, it is committed (its keep/exit test applies).
        ai._closing_down = True
        if not isinstance(expert.shadow, GetPossessionOrder):
            expert.shadow = GetPossessionOrder()
    expert.state["actual_closing_down"] = actual


def _reconcile_attacker(expert: "RulesExpert", player, match) -> None:
    """BallCarrierAttackerAI: the rules attacker's run is only ever ended by
    its own shot. If the neural attacker kicked since the last decision and
    the ball has gone (a shot, or a touch it did not recover), the run / shot
    plan is over, as after a rules shot, instead of the expert still running
    its old route. If it kicked but has the ball again (a dribble touch it
    recovered), the run goes on."""
    kicks = player.kick_count
    if kicks > expert.state.get("kick_count", kicks) and match.ball.possessed_by != player.player_id:
        expert.shadow = None
        expert.core._prev_dist_to_target = None
    expert.state["kick_count"] = kicks


def _reconcilers() -> dict:
    from footballcoach.rules_ai import BallCarrierAttackerAI, Phase1RulesAI, StagedGoalkeeperAI
    return {
        StagedGoalkeeperAI: _reconcile_keeper,
        BallCarrierAttackerAI: _reconcile_attacker,
        Phase1RulesAI: None,          # decides afresh every call; nothing to reconcile
    }


class RulesExpert:
    """See the module docstring. One per labelled neural player per episode.

    ``label(player, match)`` is a ``bc_label_fn`` (NeuralPlayerAI calls it at
    each of the player's decisions, at the instant its observation is
    encoded)."""

    def __init__(self, rules_ai, opening_order, player) -> None:
        reconcilers = _reconcilers()
        self.ai = rules_ai
        chain = _unwrap(rules_ai)
        self.core = chain[-1]
        if type(self.core) not in reconcilers:
            raise ValueError(f"no BC expert support for {type(self.core).__name__} (add a reconciler, "
                             "or None if it keeps no memory, in role_experts._reconcilers)")
        self._reconcile: Optional[Callable] = reconcilers[type(self.core)]
        # Decide at every call: the expert is asked once per neural decision,
        # which is already the rules AI's own cadence.
        for a in chain:
            if hasattr(a, "decision_interval_ticks"):
                a.decision_interval_ticks = 1
        self.shadow = opening_order
        self.state: dict = {"kick_count": player.kick_count}

    def label(self, player: "Player", match: "Match") -> "BCLabel":
        from footballcoach.ai.bc.order_labels import order_label
        from footballcoach.ai.ppo.bc import BCLabel
        from footballcoach.orders import GetPossessionOrder

        if self._reconcile is not None:
            self._reconcile(self, player, match)
        order = self._decide(player, match)
        if order is None:
            self.shadow = None
            return BCLabel.invalid()
        if isinstance(order, GetPossessionOrder):
            # Its possession callback is registered on the player inside each
            # counterfactual and removed again by the restore; let it register
            # afresh every time instead of believing it still is.
            order._callback_registered = False
        label, cf = order_label(player, match, order)
        self.shadow = None if cf["completed"] else (cf["order_after"] or order)
        return label

    def _decide(self, player, match):
        """The rules AI's decision from the shadow order on the real current
        state, with everything the decision may touch restored (it can turn
        the player, draw from match.rng for a shot target, and log)."""
        saved_player, saved_ball = dict(vars(player)), dict(vars(match.ball))
        rng_state, log_cb = match.rng.getstate(), match.log_callback
        match.log_callback = None
        player.current_order = self.shadow
        try:
            self.ai.act(player, match, 0)
            return player.current_order
        finally:
            match.rng.setstate(rng_state)
            match.log_callback = log_cb
            for obj, saved in ((player, saved_player), (match.ball, saved_ball)):
                d = vars(obj)
                for k in [k for k in d if k not in saved]:
                    del d[k]
                d.update(saved)

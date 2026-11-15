"""RoleScenarioEnv: the env for role-based curriculum phases (phase 2 onward).

A role-based phase names each player's role (attacker, defender,
goalkeeper, ...), gives every role weights over controllers ("rules",
"neural"), and pays terminal-only rewards from one table per role keyed by
the trial's outcome. Everything phase-specific is passed in as data (see
curriculum/envs.py's build_role_env), so a new phase of this kind needs no
code here.

Phase 1 keeps ScenarioEnv's own step() unchanged; this subclass overrides
only reset() and step() and reuses the rest (observation encoding, the
opportunity flags, the match logger, last_terminal_match).

The trainer-facing contract is ScenarioEnv's: per step, the "trainee"
transition plus a list of secondary results. Here the trainee is simply the
first neural player of the episode (in role order) and every other neural
player is a secondary, re-chosen at each reset. An episode in which every
role rolled "rules" has no neural player: fine for playthroughs and evals,
not for training (it produces no rows).
"""
from __future__ import annotations

import random
from typing import Callable, Optional

from footballcoach.ai.env.reward import EMAFilter, terminal_outcome_reward, validate_terminal_reward_table
from footballcoach.ai.env.scenario_env import ScenarioEnv, StepInfo
from footballcoach.ai.obs.schema import ObservationBatch
from footballcoach.engine.match import Match
from footballcoach.ui.scenarios import ScenarioDefinition, ScenarioLoop

CONTROLLERS: tuple[str, ...] = ("rules", "neural")


def _validate_controller_weights(roles: dict, weights: dict) -> None:
    for role in set(roles.values()):
        if role not in weights:
            raise ValueError(f"no controller weights for role {role!r}")
        w = weights[role]
        unknown = set(w) - set(CONTROLLERS)
        if unknown:
            raise ValueError(f"role {role!r}: unknown controller(s) {sorted(unknown)}; known: {CONTROLLERS}")
        if any(float(v) < 0 for v in w.values()) or sum(float(v) for v in w.values()) <= 0:
            raise ValueError(f"role {role!r}: controller weights must be >= 0 and not all 0, got {w}")


def roll_controller(weights: dict, rng: random.Random) -> str:
    """One draw from unnormalised {controller: weight}; consumes exactly one
    rng.random() even when only one controller has weight."""
    total = sum(float(weights.get(c, 0.0)) for c in CONTROLLERS)
    x = rng.random() * total
    for c in CONTROLLERS:
        x -= float(weights.get(c, 0.0))
        if x < 0:
            return c
    return next(c for c in reversed(CONTROLLERS) if float(weights.get(c, 0.0)) > 0)


class RoleScenarioEnv(ScenarioEnv):
    """Terminal-reward env for a role-based phase.

    Args:
        definition: The scenario (a ui/scenarios.py ScenarioDefinition).
        roles: player_id -> role, in the order the trainee is chosen from.
        controller_weights: role -> {"rules": w, "neural": w}, rolled per role
            at every reset. ``controller_overrides`` (attribute, below) pins
            roles instead, for evals.
        reward_table: role -> outcome -> reward (terminal only).
        outcome_keys: the outcome vocabulary ``classify_outcome`` returns,
            "timeout" included; the reward table must cover exactly these.
        classify_outcome: (raw_outcome, final_match, attacking_team) -> outcome.
        terminal_raw_outcomes: raw detect_trial_outcome() keys that end a trial.
        phase_name: label used in the episode tag (logging).
        Remaining arguments are ScenarioEnv's.
    """

    def __init__(
        self,
        definition: ScenarioDefinition,
        *,
        roles: dict[str, str],
        controller_weights: dict[str, dict[str, float]],
        reward_table: dict,
        outcome_keys,
        classify_outcome: Callable,
        terminal_raw_outcomes: frozenset,
        phase: int,
        phase_name: str,
        max_episode_s: float,
        **kwargs,
    ):
        if not roles:
            raise ValueError("a role-based phase needs at least one role")
        _validate_controller_weights(roles, controller_weights)
        if "timeout" not in outcome_keys:
            raise ValueError("outcome_keys must include 'timeout' (the env's own clock)")
        validate_terminal_reward_table(reward_table, sorted(set(roles.values())), outcome_keys)
        first_id = next(iter(roles))
        super().__init__(definition, trainee_player_id=first_id, max_episode_s=max_episode_s,
                         phase=phase, secondary_player_ids=[], **kwargs)
        self.roles: dict[str, str] = dict(roles)
        self.controller_weights = controller_weights
        self.reward_table = reward_table
        self.outcome_keys = tuple(outcome_keys)
        self._classify_outcome = classify_outcome
        self._terminal_raw_outcomes = frozenset(terminal_raw_outcomes)
        self.phase_name = phase_name
        # role -> controller, pinning those roles instead of rolling them
        # (evals); roles not listed are still rolled.
        self.controller_overrides: dict[str, str] = {}
        # Training needs a neural trainee in every episode (the batched rollout
        # worker batches the trainee's decisions). When set, rolls with no
        # neural player are redrawn, i.e. controllers follow the configured
        # weights CONDITIONED on at least one neural player.
        self.require_neural_player: bool = False
        # This episode's player_id -> controller, and a stable text label of it.
        self.episode_controllers: dict[str, str] = {}
        self.episode_tag: str = ""
        self._attacking_team = None

    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> ObservationBatch:
        """Build a new trial, roll each role's controller and wire neural players.

        With ``seed`` the controller rolls are seeded too (from a string derived
        from it, so they do not shift the scenario's own draws), so replaying a
        seed replays the same controller assignment."""
        from footballcoach.rules_ai import NeuralPlayerAI

        build_kwargs = {**self.scenario_kwargs, "sim_dt_s": self._dt_s}
        if seed is not None:
            build_kwargs["seed"] = seed
        self._loop = ScenarioLoop(
            definition=self.definition,
            max_trials=0,
            rng_reduction=self.rng_reduction,
            linger_s=self.linger_s,
            kwargs=build_kwargs,
            timeout_ticks=10**9,
            terminal_outcomes=self._terminal_raw_outcomes,
        )
        if self._max_episode_s_variability_s > 0.0:
            self.max_episode_s = self._base_max_episode_s + self.rng.uniform(0.0, self._max_episode_s_variability_s)
        else:
            self.max_episode_s = self._base_max_episode_s
        self._ema.reset()
        self._episode_ticks = 0
        self._box_bounds_cache = {}
        self._max_episode_ticks = max(1, int(self.max_episode_s / self._dt_s))
        self.last_trainee_transition = None
        self.last_secondary_results = []
        self.last_terminal_match = None
        self.last_reward_components = {}

        match = self._loop.match
        carrier = match.player_by_id_or_none(match.ball.possessed_by)
        if carrier is None:
            raise ValueError(f"{self.definition.key}: an attack/defence trial must start with a player on the ball")
        self._attacking_team = carrier.team
        for pid in self.roles:
            match.player_by_id(pid)   # every role must exist in the built match

        roll_rng = random.Random(f"controllers:{seed}") if seed is not None else self.rng
        self.episode_controllers = self._roll_controllers(roll_rng)
        neural = [pid for pid, c in self.episode_controllers.items() if c == "neural"]
        if neural and self.sample_action_fn is None:
            raise ValueError(f"{self.phase_name}: roles rolled neural ({neural}) but no sample_action_fn is set")
        self.trainee_player_id = neural[0] if neural else next(iter(self.roles))
        self.secondary_player_ids = neural[1:]
        self.episode_tag = f"{self.phase_name}[" + ",".join(
            f"{self.roles[pid]}={self.episode_controllers[pid]}" for pid in self.roles
        ) + "]"

        for pid in neural:
            if pid != self.trainee_player_id and pid not in self._sec_ema:
                self._sec_ema[pid] = EMAFilter.from_config()
        for pid in self.secondary_player_ids:
            self._sec_ema[pid].reset()
        for pid in neural:
            player = match.player_by_id(pid)
            ema = self._ema if pid == self.trainee_player_id else self._sec_ema[pid]
            player.ai = NeuralPlayerAI(
                self.sample_action_fn,
                decision_interval_ticks=self._ticks_per_decision,
                max_episode_s=self.max_episode_s,
                ema_smoothed=ema.smoothed,
                rng=self.rng,
                bc_label_fn=self.bc_label_fn if pid == self.trainee_player_id else None,
            )
            # The scenario's scripted opening order belongs to the rules AI.
            player.current_order = None

        self._episode_index += 1
        self._match_logger.reset()
        match.match_logger = self._match_logger
        self._match_logger.record_start(match)
        return self._get_obs()

    def _roll_controllers(self, rng: random.Random) -> dict[str, str]:
        """player_id -> controller for one episode (overrides pinned, the rest
        rolled), redrawn while require_neural_player is set and nothing is
        neural."""
        for role, ctrl in self.controller_overrides.items():
            if ctrl not in CONTROLLERS:
                raise ValueError(f"controller override {ctrl!r} for role {role!r}; known: {CONTROLLERS}")

        def _can_be_neural(role: str) -> bool:
            pinned = self.controller_overrides.get(role)
            return pinned == "neural" if pinned is not None else float(self.controller_weights[role].get("neural", 0)) > 0

        if self.require_neural_player and not any(_can_be_neural(r) for r in self.roles.values()):
            raise ValueError(f"{self.phase_name}: require_neural_player is set but no role can be neural")
        while True:
            out = {}
            for pid, role in self.roles.items():
                ctrl = self.controller_overrides.get(role)
                out[pid] = ctrl if ctrl is not None else roll_controller(self.controller_weights[role], rng)
            if not self.require_neural_player or "neural" in out.values():
                return out

    def step(self, *, timers_already_advanced: bool = False) -> tuple[ObservationBatch, float, bool, StepInfo]:
        """Advance one decision interval; pay terminal rewards when the trial ends."""
        if self._loop is None:
            raise RuntimeError("Call reset() before step()")
        match = self._loop.match
        trainee = match.player_by_id(self.trainee_player_id)
        info = StepInfo()
        self.last_trainee_transition = None

        raw_outcome: Optional[str] = None
        kicks = armed_kicks = 0
        for tick_idx in range(self._ticks_per_decision):
            tick_done = self._loop.step(skip_state_timers=(timers_already_advanced and tick_idx == 0))
            self._episode_ticks += 1
            if trainee.kicked_this_tick:
                kicks += 1
                if trainee.kick_armed:
                    armed_kicks += 1
            if tick_done:
                # The loop has already built the next trial; the one that ended
                # is last_completed_trial_match.
                raw_outcome = self._loop.last_outcome
                break

        trial_ended = raw_outcome is not None
        timed_out = not trial_ended and self._episode_ticks >= self._max_episode_ticks
        done = trial_ended or timed_out
        final_match: Match = self._loop.last_completed_trial_match if trial_ended else match
        outcome = None
        if done:
            outcome = "timeout" if timed_out else self._classify_outcome(
                raw_outcome, final_match, attacking_team=self._attacking_team)
            if outcome not in self.outcome_keys:
                raise ValueError(f"classifier returned {outcome!r}, not in this phase's outcome_keys")
            info.trial_outcome = outcome

        def _reward(pid: str) -> tuple[float, dict]:
            if not done:
                return 0.0, {}
            return terminal_outcome_reward(self.reward_table, self.roles[pid], outcome)

        reward, self.last_reward_components = _reward(self.trainee_player_id)
        info.ticks_elapsed = self._episode_ticks
        info.trainee_kicks_this_step = kicks
        info.trainee_armed_kicks_this_step = armed_kicks
        info.episode_tag = self.episode_tag

        if getattr(trainee.ai, "last_transition", None) is not None:
            self.last_trainee_transition = trainee.ai.last_transition
            if self._opp_flags_enabled:
                self.last_trainee_transition = self._with_opportunity_flags(self.last_trainee_transition, trainee, done)
            info.illegal_action = self.last_trainee_transition.get("illegal_action", False)

        self.last_secondary_results = []
        for pid in self.secondary_player_ids:
            sec_player = match.player_by_id(pid)
            transition = getattr(sec_player.ai, "last_transition", None)
            if transition is None:
                continue
            if self._opp_flags_enabled:
                transition = self._with_opportunity_flags(transition, sec_player, done)
            sec_reward, sec_comps = _reward(pid)
            self.last_secondary_results.append({
                **transition,
                "player_id": pid,
                "reward": sec_reward,
                "done": 1.0 if done else 0.0,
                "step_outcome": outcome or "",
                "reward_components": dict(sec_comps),
            })

        if done:
            self._match_logger.notify_episode_end(
                final_match.time_s, final_match.ball.position, outcome, reward, self.last_reward_components)
        else:
            self._match_logger.accumulate_reward(reward, self.last_reward_components)
        if not done:
            self.last_terminal_match = None
        elif trial_ended:
            self.last_terminal_match = self._loop.last_completed_trial_match
        else:
            self.last_terminal_match = self._loop.match
        return self._get_obs(), reward, done, info

    def episode_rewards_by_player(self, outcome: str) -> dict[str, float]:
        """Every role player's terminal reward for ``outcome`` (whatever its
        controller), e.g. for playthrough summaries."""
        return {pid: terminal_outcome_reward(self.reward_table, role, outcome)[0] for pid, role in self.roles.items()}

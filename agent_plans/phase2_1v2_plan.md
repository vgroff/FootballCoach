> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

# Phase 2: 1v2 (attacker vs defender + goalkeeper)

Written 2026-11-15. Goal: add the UI's 1v2 scenario as curriculum phase 2,
with terminal-only rewards for three roles, trainable alongside phase 1 from
an existing phase-1 checkpoint, built so that a phase 3 is mostly a config
entry plus (at most) a new outcome classifier.

## Decisions (from the user, 2026-11-15)

- **Rewards are terminal-only**, from one table per role: attacker,
  defender, goalkeeper. "Defence" in the outcome list means the whole
  defending team. Values are placeholders for now.
- **Controllers per role**, configurable: first all rules (playthrough
  only), then neural attacker and defender with a rules goalkeeper, and
  eventually a neural goalkeeper too.
- **Phase 1 stays untouched**: same rewards, same code path, verified
  bit-identical with `scripts/env_trace_snapshot.py`,
  `scripts/golden_snapshot.py` and `scripts/ppo_update_trace_snapshot.py`.
- **Episode ends** on: goal; the keeper getting the ball (saved); a
  defender getting the ball (defence possession; a clearance always starts
  with the defender taking possession, so it lands here); the ball going out
  (attacker shot over the attacked goal line = off target; any other
  attacker-caused out = "attacker out", a heavy attacker penalty); timeout.
  Own goals cannot happen (the episode ends on defensive possession first).
- **Frozen heads**: same as phase 1. Shooting uses the execution kick head.
  Unfreezing tackle/pass/mark would need per-row head masks and is out of
  scope.
- Everything that varies by phase (env construction, controllers, outcome
  vocabulary, rewards, logging tags, evals) must be data on the phase, not
  `if phase == N` branches.

## Facts this plan relies on (verified 2026-11-15)

- `build_1v2_scenario` exists with rules AIs for all three players. It takes
  no `seed` and no `sim_dt_s`, which `ScenarioEnv.reset()` always passes, so
  the builder needs both parameters.
- `detect_trial_outcome` already returns goal / saved / dispossessed / miss
  for 1v2 (box possession is off for that definition).
- A goalkeeper's reach and jumping come from the possession model whenever
  `is_goalkeeper` is set, so a neural keeper needs no new action head.
- Observations have room for 21 other players, an `is_goalkeeper` flag and a
  20-slot task-ID one-hot (`task_id_{phase-1}`), so a phase-1 checkpoint
  loads into a phase-2 run unchanged.
- Every training loop drains one neural "trainee" per env plus a list of
  neural "secondaries"; the batched worker requires the trainee to be
  neural. Phase 2 therefore picks the trainee per episode as the first
  neural player, and the rest become secondaries. A phase-2 episode with no
  neural player is valid for playthroughs and evals, not for training.
- `ScenarioEnv._latest_outcome()` returns the outcome with the highest
  cumulative count, not the most recent one. Harmless for phase 1 (only miss
  and goal can end a trial there, and both are treated as miss); wrong for
  any phase with more outcomes. Fixed for the new env.
- The old phase-2 stub (penalty scenario, `phase2_reward`) has never run
  (its builder rejects `sim_dt_s`). It is replaced.

## Design

### Phase definition (`ai/curriculum/phases.py`)
`CurriculumPhase` gains optional data fields: `roles` (player_id -> role),
`controllers_config_key` (where its per-role controller weights live in
`ai_config.json`), `reward_config_key`, `build_env` (a factory, replacing
`curriculum/envs.py`'s if/elif), `outcome_keys`, and `eval_specs`. Phase 1
keeps its current builder; only the dispatch changes.

### Outcomes (`ai/env/outcome.py`)
`classify_attack_defence_outcome(raw_outcome, final_match, attacking_team)`
maps the loop's raw outcome plus the final tick's state to
`goal / saved / defence_possession / shot_off_target / attacker_out /
timeout`. It is team-based (attacking team = initial carrier's team;
`is_goalkeeper` for saves), so it also fits 2v2, 2v3 and so on.

### Rewards (`ai/env/reward.py`, `ai_config.json`)
`terminal_outcome_reward(table, role, outcome)` returns `(reward, comps)`
from `reward.phase2.terminal[role][outcome]`; a missing entry raises. No
shaping terms.

### Env (`ai/env/role_env.py`)
`RoleScenarioEnv(ScenarioEnv)` overrides `reset()` and `step()` only:
- `reset()`: builds the match, rolls each role's controller from the phase's
  weights, replaces rolled-neural players' AI with `NeuralPlayerAI` (and
  clears their scripted order), and sets this episode's `trainee_player_id`
  and `secondary_player_ids`.
- `step()`: ticks the decision interval; ends on the loop's outcome or the
  env timeout; classifies the outcome; pays each neural player its role's
  terminal reward; fills `last_trainee_transition` / `last_secondary_results`
  in the existing format, plus an `episode_tag` describing the controllers.
`ScenarioEnv` itself is not edited.

### Mixed-phase training (`curriculum.training_phase_mix`)
`{"1": weight, "2": weight}`; default `{"1": 1}` is today's behaviour. Each
rollout env slot is assigned a phase deterministically from the weights.
Weights are per env slot; phase-2 episodes are much shorter, so the logged
row share per phase is what to tune against.

### Logging and evals
- Episode outcomes are tallied per `episode_tag` instead of the hardcoded
  vs_rules / vs_immobile / vs_neural lists. Phase-1 tags keep those names, so
  phase-1 logs read the same.
- Each phase lists eval specs (controller overrides, seeds, episodes); the
  trainer runs every spec of every phase in the mix with one generic runner.

### Starting from a checkpoint
Load into a fresh run directory. The phase-2 task-ID column and the
`is_goalkeeper` column were always 0 in phase 1, so their first-layer
weights never got a gradient and are still at random init (to verify on a
real checkpoint before acting). Opt-in: zero those input columns on load.
Value refit (PPG) before PPO to recalibrate the value head.

## Stages

1. **Phase-2 env, no trainer changes.** Builder seed/sim_dt_s; outcome
   classifier; reward table; `RoleScenarioEnv`; phase-2 definition and
   registry dispatch; rules playthrough test and
   `scripts/phase2_rules_playthrough.py`. Phase 1 bit-identical.
2. **Mixed-phase rollouts and per-tag logging** in all three training paths
   and both rollout workers. Phase 1 bit-identical with the mix at its
   default.
3. **Per-phase evals.**
4. **Checkpoint start**: measure the unused-column claim, then the opt-in
   zeroing; document the run recipe. No training runs.

## Findings during stage 1

- Training envs tick at `observation.sim_dt_s` = 0.04 s; the UI at 1/30 s.
  With rules AIs over the same 2000 seeds, 1v2 gives 39.7% goals in the env
  and 34.1% in a 1/30 s loop; an independent 0.04 s loop plus the classifier
  reproduces the env trial for trial (0 mismatches). Not a bug, but UI
  outcome rates are not training outcome rates.
- Rules 1v2 at 0.04 s, 2000 seeds: goal 793, defence_possession 641, saved
  550, timeout 10, shot_off_target 6, attacker_out 0. The rules attacker
  almost never misses, so the attacker_out / off-target branches are covered
  by constructed-state unit tests, not by play.
- `detect_trial_outcome` reports a teammate's reception as "saved": the UI's
  2v2 scenario ends all 49 traced trials as "saved". The 1v2 classifier raises
  on that case; a phase with several attackers must fix it first.

## Status

- [x] Stage 1 (2026-11-15): builder seed/sim_dt_s, ScenarioLoop.last_outcome
  (and ScenarioEnv._latest_outcome now uses it), attack/defence classifier,
  terminal reward tables, RoleScenarioEnv, data-driven build_env, rules
  playthrough. Phase 1 verified unchanged: env trace identical apart from
  the new, always-empty StepInfo.episode_tag; engine trace and golden
  identical.
- [x] Stage 2 (2026-11-15): curriculum/mix.py (parse/validate, env-slot
  assignment, phase-tagged replay seeds), per-env phases in the batched
  workers (build_worker_envs), RoleScenarioEnv.require_neural_player,
  per-tag outcome/reward tallies and summary lines, train.py --phase-mix,
  non-batched paths refuse a mix. Phase-1 batched rollouts verified unchanged
  with scripts/batched_rollout_trace_snapshot.py (differences = only the two
  new, empty stats keys; a replay-order mutation is caught). Not mixed yet:
  PPG refit / value pre-training (primary phase only) and evals (stage 3).
- [ ] Stage 3
- [ ] Stage 4

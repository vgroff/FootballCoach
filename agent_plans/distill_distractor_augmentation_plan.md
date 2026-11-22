# Distill refit: distractor-player augmentation (invariance teaching)

Status: idea from the user, 2026-11-22. Not implemented. To be tried if the
distill refit really hits a wall before the user decides on PPO; tell the user
when starting it.

Scope (user, 2026-11-22): a phase-2 attacking-phase tool only, used where
the extra players are certainly irrelevant. It is NOT for 5v5 or any
cooperative phase: there far players matter (passing, support), and
cooperation will be taught there instead.

## Idea

On phase-2 (and optionally phase-1) rollout states, add synthetic players
in the third of the pitch away from play:
- a goalkeeper and/or defender on the attacker's team;
- an attacker on the defender's team.

Each gets a random position, velocity, attributes and stamina. They are not
simulated, only slotted into the observation.

The frozen teachers score the clean observation, and the student must
produce the same output on the augmented one. The student learns that
far-away, irrelevant bodies do not change the action. This is standard
augmentation-invariance (consistency) distillation, and it prepares the
network for crowded 5v5 observations.

## Why it fits the current code

- The entity encoder takes a variable number of other players through
  `exists_mask` (`schema.MAX_OTHER_PLAYERS` = 21 slots).
- The refit already scores teacher and student on separate input versions:
  `teacher_anchor` uses the obs without the precomputed physics features,
  and the student recomputes physics features from its own obs.

## Build (single-source: use the real encoder)

- All 41 `PlayerFeatures` fields are derivable from a player's position,
  velocity and attributes plus the self and ball state: `rel_dx/dy`,
  `distance_m`, `ball_rel_*`, `ball_vel_rel_*`, `ball_closing_speed`, pos,
  heading, flags, attributes and desired-movement fields.
- Do not re-derive them in tensor space: that duplicates the encoder
  (single-source rule).
- Instead, in the rollout workers at collection time, add synthetic
  `Player` objects to a copy of the match's player list (no sim step) and
  call the real `encode_observation` on it.
- Store both versions for an augmented row: the clean obs for the teacher
  anchor and the augmented obs for the student. RAM is tight (refit process
  ~16.5 GB with 2 aggregated cycles), so augment a fraction of rows (e.g.
  0.3), as an opt-in `distill_pretrain.distractor_fraction` (default 0).
- Value target: unchanged returns (the far players are irrelevant to the
  outcome by construction).
- Counterfactual copies: augment them the same way (the task id and the
  distractors are independent).

## Placement rules

- Far from the ball and from every real player (e.g. more than 25 m from
  both, inside the far third), never between the ball and either goal.
- Use valid attribute ranges (sample the scenario generator's attribute
  distribution) and plausible velocities.
- Distractors never hold the ball, are never the last toucher, and are
  never offside-relevant.

## Risks

- **The wrong invariance for 5v5.** A far teammate is a pass option in 5v5,
  so "ignore far teammates" could fight passing later. Keep distractors
  only where they truly cannot matter for the current task, make the
  augmentation easy to switch off, and drop it before the passing work
  (agent_plans/phase2_second_attacker_passing_plan.md).
- The keeper flag (`is_goalkeeper`) on a distractor keeper: the refit skips
  counterfactual copies of keeper ROWS (`goalkeeper_rows`, own-player flag),
  which is unaffected because distractors are other players, never self.

## How to measure

- Paired KL of the student on augmented vs clean versions of the same rows
  (should converge to 0).
- The normal-row KL against the frozen reference (`holdout_reference`) must
  not get worse.
- Behaviour evals unchanged (seeded, power sigma matched).

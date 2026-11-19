# Engine luck in the advantage (parked, 2026-11-19)

Status: **parked** (user's call). Fresh engine dice on episode replay
(`ppo.episode_replay_fresh_engine_rng`, on) already stop replay from
re-running hand-picked luck; this note is for if we want more later.
Nothing here is implemented or tested.

## Measured (single checkpoint, phase2_run34/checkpoint82)

115 phase-2 eval seeds, AI attacker trainee vs rules defender + keeper,
network sampling fresh every play. Score = the replay picker's per-episode
mean |GAE advantage| (gamma 0.99, lam 0.96).

| pair | Spearman | top-35% overlap (chance 35%) |
|---|---|---|
| fresh dice, play 1 vs 2 / 1 vs 3 / 2 vs 3 | 0.71 / 0.71 / 0.77 | 57 / 68 / 70% |
| the seed's own dice, play 1 vs 2 | 0.88 | 82% |
| the seed's dice vs fresh dice | 0.63 | 62% |

- Seeds explain ~71% of the score's variance with fresh dice: the replay
  pile mostly selects genuinely hard situations, not luck. (Earlier estimate
  that the selection was mostly luck was wrong.)
- The engine dice add the rest of the repeatability (0.73 -> 0.88 Spearman):
  that is the luck the old full-seed replay re-ran.
- Score vs V(s0): Spearman -0.48 (big surprises come from the harder starts;
  not looked into).
- Outcome agreement for comparison: same dice 88.7%, fresh dice 76.5%.
- A phase-2 episode makes ~2 engine draws, both at the decisive kick / tackle.

Probe scripts were in the session scratchpad (`adv_rank_probe.py`,
`dice_share_probe.py`); single process, ~20 min each.

## Idea: subtract the luck of each engine roll (control variate)

At every engine roll, luck = V(state after the roll) - E over the roll's
outcomes [V(state after)]. Subtracting it from the advantage is still
unbiased (it has zero mean under the roll's own distribution, however wrong
V is); it only removes variance. Known technique: AIVAT (Burch et al. 2018)
for poker evaluation; for PPO it is an extra baseline at chance nodes.

- **Tackles (cheap):** the engine knows the success probability p, so
  luck ~= (success - p) * (V_won - V_lost). Two value calls per tackle.
- **Kicks (dearer):** the noise is continuous: clone the match, re-roll the
  kick noise K times, simulate a few ticks, average V. ~2 rolls per episode,
  so affordable but not free.
- **Also helps:** replay selection on luck-free |advantage| (situation only),
  and lower-variance evals (the AIVAT use).

Open questions: which V to use after the roll (value of the trainee only, or
per role); how many ticks after a kick before V sees the result; whether the
gain is worth it next to fresh-dice replay and seeded evals.

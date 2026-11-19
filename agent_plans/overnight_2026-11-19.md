# 2026-11-19: harder phase-2 start, keeper tweaks (run31)

User: the attacker has an easy time -- start further back, random starting
speed/direction for all players (as phase 1), and see whether the rules
keeper can save more (close the angle, close down).

## Measurements (scratchpad keeper_sweep*.py; run30/checkpoint53 as the AI
attacker; defender + keeper rules; 115 phase-2 eval seeds; the rules attacker
is deterministic per seed, so its rows are 115 independent episodes)

Scenario, 6 m keeper (goals scored, rules / AI attacker): current 15.7 /
87.4%; start 25-45 m 5.2 / 75.9%; random velocity 10.4 / 61.7%; both 6.1 /
57.4% (defender's ball wins vs the AI 10.2 -> 38.5%).

Keeper angle positioning (new, StagedGoalkeeperAI._guard_target): stand on
the goal-centre -> ball line, fraction f of the ball distance off the line
(cap m). Current scenario, rules attacker conceded: off 15.7%, f0.05/2 m
13.0%, f0.10/3 m 27.0%, f0.15/4 m 29.6%, f0.20/6 m 30.4% (saves 31 -> 18%);
AI attacker 86.5-87.3% for all. Worse -> left OFF (code kept, default off).

Close-down entry distance (angle off), conceded rules / AI attacker (AI shots
off target): 6 m 15.7 / 86.7 (1.0%); 8 m 15.7 / 79.6 (7.9%); 10 m 15.7 /
45.3 (39.6%); 12 m 16.5 / 23.4 (58.8%); 14 m 21.7 / 15.7 (59.7%). The AI
attacker (trained only vs the 6 m keeper) shoots wide against a keeper
rushing out; expected to recover with training (untested).

## User's calls -> run31
- Keeper close-down 10 m (exit 12 m), global (orders.json).
- Phase 2: attacker 18-40 m from goal, random_initial_velocity on.
- Run30's mix kept (50:50 phases; attacker 20:80, defender 30:70 rules:neural).
Run31 from run30/checkpoint53, checkpoints/phase2_run31, log
scratchpad/ppo_run31.log; watchdog moved. Phase-2 evals not comparable with
earlier runs (new start + keeper).

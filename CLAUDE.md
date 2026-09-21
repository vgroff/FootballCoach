# FootballCoach — agent instructions

## Never use `git stash`

Multiple agents work in this repo concurrently (possibly in parallel
sessions/worktrees against the same working tree). `git stash` is a single
shared stack — one agent's stash can be popped, dropped, or shadowed by
another agent's concurrent stash operation, silently losing someone's work.

- Do not run `git stash` (including `push`/`pop`/`apply`/`drop`) for any
  reason, including "just checking if a test failure pre-exists my changes."
- To compare against a clean baseline, use a disposable worktree
  (`git worktree add`) or `git diff`/`git show` against a specific commit
  instead of stashing the current changes away.
- To set aside in-progress edits, copy the file(s) elsewhere or commit to a
  scratch branch instead.

## Never run destructive git commands, and always dry-run deletes

An agent cleaning up its own scratch script once ran a delete command that
also swept up three unrelated, pre-existing, never-committed files sitting
in the repo root — permanently lost, since they were untracked (no git
history to recover from).

- Do not run destructive git commands (`git reset --hard`, `git clean`,
  force-push, branch deletion, etc.) unless the user explicitly asked for
  that specific operation.
- Before running ANY command that deletes files (`rm`, `del`, `git clean`,
  etc.), first run its dry-run/list-only form (e.g. `rm` with `-n` where
  supported, or just `ls`/`find`/`git clean -n` on the same pattern first)
  and check the exact file list it would touch — especially when cleaning up
  your own scratch/temp files, since a too-broad glob or path can silently
  catch someone else's work sitting in the same directory.
- If a delete command's target list includes anything you didn't create
  this session, stop and ask before proceeding.

## Investigate, don't hand-wave: try to disprove your own explanation

When the user says something looks wrong ("smells like a bug", "that seems
weird", "why can't it do X when it has the information"), treat it as a
hypothesis to test, not a preference to talk them out of. In this repo, deep
dives on suspicions like these have surfaced real bugs or design flaws far more
often than not (the user's estimate: ~80%). Examples from one session: a value
net that "plateaued" turned out to ignore near-certain ball-out misses; an
"episode ends when the ball goes out" rule turned out to fire 0.5 m / 1.0 m
past the painted line (never Law 9's whole-ball-over-the-line); a claim that
returns are deterministic under a deterministic policy was false (the sim rolls
random kick error and tackles).

- **Prove it, and try to break it.** Before presenting a cause, run the check
  that would refute it, not just the one that would confirm it. A confirming
  result is where to go deeper (counterfactuals, held-out data, a different
  slice), not where to stop.
- **Don't stop at the first hurdle.** A failed or awkward probe is a reason to
  fix the probe, not to conclude "not worth it". Budget for several rounds.
- **Audit your own metric and conditioning.** Check the metric measures what you
  claim (e.g. `-value` is a bad "miss score" because value is dominated by who
  is winning), and slice on the variable that matters before calling something
  "not decisive" (pooling slow drifts with fast ones hid the decisive cases).
  Verify assumptions against the code (grep it) instead of reasoning from what
  you think it does.
- **Prefer white-box evidence.** Reconstruct the real inputs and confirm they
  reproduce recorded outputs (rules out plumbing bugs), then do counterfactual
  edits on the real network, before drawing design conclusions.
- **Answer the question that was asked.** Do not steer toward the next
  milestone (e.g. "let's resume PPO") while the user's question is still open.
  Offer next steps after the evidence, not instead of it.
- **Label what is tested vs guessed.** Say "untested idea" for the latter,
  retract early and plainly when a result disproves you, and correct any
  notes/knowledge files you already wrote.
- **This is not deference.** Keep disagreeing when the evidence supports it and
  say when the user is wrong; debate is wanted. The ask is curiosity plus
  rigour: be the one who tries to falsify your own hypotheses first.

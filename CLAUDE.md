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

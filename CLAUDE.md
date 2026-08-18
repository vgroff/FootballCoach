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

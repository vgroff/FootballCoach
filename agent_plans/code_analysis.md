# Code Analysis — FootballCoach

> **Status as of 2026-10-12 (reconciliation pass):** mostly still accurate
> and still open — this is a deliberately-curated "what's left" list (per
> the note below, already-fixed items were removed by prior agents), and
> re-verification confirms nearly every remaining item is still true today,
> with current file:line refs refreshed inline. Two items changed:
> **§7's `tests/ai_scenario/` single-file claim is now stale/resolved**
> (that directory has grown to 27 test files — see inline note), and
> **§6's `PlayerAttributes.average` docstring claim is partially stale**
> (the docstring itself was removed at some point; the method is now
> undocumented rather than mis-documented, though it's still used in
> production UI code under a test-sounding name — see inline note). One
> new confirmed finding: **§7's "N >= 1000" audit request turned up a real
> inconsistency** — `N_TRIALS` is 200 in `test_pass_balance.py`, 500 in
> `test_shoot_balance.py`, 2000 in `test_penalty_balance.py`, and at least
> one deliberate `n_trials=50` for a non-hard-target visual table — see
> inline note. Everything else (structural/architectural items in §3,
> magic numbers in §4, doc-drift items in §5, remaining test-coverage gaps
> in §7, and the UI note in §8) was spot-checked against current code and
> is still open as described, just with line numbers refreshed where they
> had drifted.

> **Note for future agents:** Sections 1 and 2 (bugs and duplicate logic) and
> most of the mechanical items in sections 4, 5, 6, and 8 have already been
> fixed and removed from this file. What remains is either (a) a genuine
> structural/design decision that needs a human call before touching (see
> the `Developer Note`s in section 3), or (b) an item intentionally left
> alone because fixing it requires a design/behavioral judgment call, not
> just a mechanical refactor (each such item has a short note explaining
> why). When picking up further items from this list: do the mechanical/
> low-risk fixes without asking, flag anything that changes behavior or
> needs a judgment call, run the full test suite
> (`uv run pytest tests/unit tests/scenario tests/balance -q`) after each
> change, and remove completed items from this file as you go.

Automated read-through of the non-AI parts of the codebase (engine, entities,
orders, actions, steering, rules_ai, ui, mathutils, generation, config,
tests, top-level debug scripts, and `scripts/`). The `src/footballcoach/ai/`
package itself was intentionally **not** reviewed here (per request), except
`rules_ai.py` which lives at the top level.

Findings are grouped by module. File paths are relative to the repo root.
Line numbers are approximate (code may have shifted slightly since review).

---

## 3. Structural / architectural criticisms

- **`ball.possessed_by` is a player-id string, not a player reference** —
  flagged directly in [Idea2.md] by the user, and confirmed as a real,
  pervasive cost: every site needing the actual carrier must call
  `Match.player_by_id(...)`, a linear O(n) scan over `self.players`, done
  many times per tick from `possession.py`, `match.py`, `orders.py`, and
  `tackling.py`. It also requires scattered `try/except KeyError` guards for
  stale ids (e.g. `MarkOrder.execute`'s `except KeyError: return True # target
  gone`). This will get worse as player counts grow toward full 11v11. If the
  string-id-as-source-of-truth is kept for serialization/observation-encoding
  reasons, at minimum add a per-tick `dict[str, Player]` cache in `Match`
  rather than repeated linear scans. Developer Note: We tried fixing this before and it's a pain because circular references, does it seem easy actually? I agree that it should be done

- **`engine/match.py` is a "does everything" file** — tick loop, order
  processing, possession syncing, ball pickup, head-on-tackle detection, GK
  immunity/box logic, pass leading, and interception math all live in one
  `Match` class. Worth splitting interception math out (see above) and
  possibly the GK-box/immunity logic into its own module, since
  `goalkeeping.py` already exists as a natural home for GK-specific logic. 
  Developer Note: Sounds good. Anything else that could be split out from match without too much hassle?

- **Anthropometric constants live in `physics.json["player"]`** rather than
  `attributes.json` — `knee_height_m`, `waist_height_m`, `height_m`,
  `radius_m` describe body shape, which is arguably closer to
  "attributes/generation" in spirit than "physics," even though these
  particular values are universal rather than per-player. At minimum, a
  comment in `attributes.json` cross-referencing where body-shape constants
  actually live would help. Developer Note: yeah, better in attributes

- **`Scoreboard.score_for()` / `Pitch.is_goal()` use raw `"left"`/`"right"`
  strings** rather than the `Team` enum that already exists elsewhere in the
  codebase — this stringly-typed convention flows from `Pitch.is_goal()`
  through `check_goal()` to `Scoreboard.score_for()`, and exists purely to
  justify a runtime `ValueError(f"unknown goal side: {side}")` guard that a
  real enum would make unnecessary (caught statically instead). Developer Note: fix it

- **`Player.current_order: object | None` and `desired_speed_mode: object |
  None` are untyped** to dodge circular imports with `orders.py` — a
  `TYPE_CHECKING`-guarded type alias (the codebase already does this for
  `Match` elsewhere) would restore type-checking value cheaply. Developer Note: do it

- **`Ball`'s `position`/`velocity`/`spin` use `= None  # type:
  ignore[assignment]` plus a `__post_init__`** to work around dataclass
  mutable-default issues, whereas `Player` (in the same codebase) already
  uses the more idiomatic `field(default_factory=Vector3.zero)` for the same
  purpose. Inconsistent pattern between two entity classes for the exact same
  problem — worth aligning `Ball` to `Player`'s approach and dropping the
  `type: ignore`. Developer Note: sounds like a straightforward fix?

- **No single canonical epsilon constant** — `1e-6`, `1e-9`, `1e-12` all
  appear scattered across modules (e.g. `Pitch.is_goal()`,
  `Ball.is_grounded()`) for conceptually similar "avoid divide-by-zero /
  treat as touching" checks, with no shared name explaining which tolerance
  to use where. Developer note: yeah fix that, make it a constant somewhere

- **`Vector3`'s module docstring calls it "a tiny, fast 3D vector wrapper
  around numpy,"** but internally all arithmetic is plain Python floats;
  numpy is only used by the rarely-called `from_array`/`as_array` conversion
  helpers. The docstring's framing overstates numpy's role and could mislead
  readers about performance characteristics. Developer Note: why isn't it numpy? 
  Could it be changed easily for better performance?

---

## 4. Magic numbers / literals that should be config-driven

- `orders.py`'s `MoveOrder.arrival_tolerance_m = 0.3` and
  `overshoot_timeout_s = 0.5` are dataclass field defaults rather than
  config values, inconsistent with the project's stated "everything tunable
  lives in JSON" convention. Left as-is: these are dataclass field defaults
  (not local constants), so making them config-driven means deciding
  whether every `MoveOrder` construction site should look up a JSON default
  or whether the field defaults stay as fallbacks with JSON only overriding
  them — a small design choice rather than a pure mechanical move.

---

## 5. Documentation that seems stale, unconvincing, or drift-prone

- **The same tuned numeric worked-examples appear in multiple places** and
  will silently drift out of sync: `kicking.py`'s `kick_sigma_rad` docstring
  gives a specific numeric example ("a 25m auto-pass at eff_power≈0.20 gets
  barely any extra error, multiplier ≈ 1.01"); `possession.py`'s control-time
  model has similarly narrated worked examples; `tackling.py`'s
  `attempt_tackle` docstring re-explains logic that's *also* fully
  re-explained in `physics.json`'s `_comment_speed` key; and
  `match.py`'s `_effective_dribbling` docstring math is likewise duplicated
  in a `physics.json` comment. Each of these is "explained" in two or more
  places, and since the underlying constants have clearly been retuned
  repeatedly (per project history), there's no guarantee prose and code have
  stayed in sync. I'd treat every such docstring number as suspect until
  cross-checked against the current `physics.json` values. (Left as-is:
  picking one canonical location and rewriting the other to reference it is
  a documentation-structure decision, not a pure mechanical fix.)
- **`match.py`'s big "ordering subtlety" tick-order comment is duplicated
  near-verbatim in `engine/knowledge.md`.** Same drift risk — a future
  tick-order change is likely to update one copy and forget the other.
- **`goalkeeping.py`'s `predict_goal_line_crossing` explicitly ignores
  drag/Magnus effects**, self-described as the same simplification used in
  `kicking.solve_launch_pitch_rad`. This is an honest, self-aware
  approximation — but there is no balance test that measures *how wrong*
  this becomes for heavily-spun/curved shots. The explanation is plausible
  but its magnitude is untested, so I'd flag it as "explained but not
  validated" rather than settled.
- **Balance-test target bands documented as being right at the edge of
  pass/fail** — the project's own plan-doc history notes cases like
  "precision=0.5 corner aim: 79.95% vs 50-80% target" and "precision=0.8
  corner: 94.95% vs 85-95% target," both essentially touching the boundary.
  These are self-flagged as close calls in the project's own notes but
  don't appear to have been revisited since. A tiny incidental change to
  `angle_error_scale_rad` could flip either test from pass to fail without
  the underlying game feel actually changing — this reads as under-tuned
  rather than solid, despite being "documented." (This is a balance/tuning
  decision, not a mechanical fix — leave for a dedicated tuning pass.)

---

## 6. Easy wins

- `PlayerAttributes.average(value: float = 0.5)` in
  [entities/attributes.py](src/footballcoach/entities/attributes.py) is documented as
  "mainly for tests" but is actually used directly in production UI code
  ([ui/scenarios.py](src/footballcoach/ui/scenarios.py)) and sandbox scripts, not
  just tests — contradicting its own docstring. Left as-is: renaming/moving
  it requires deciding what the *intended* production API should be, not
  just a mechanical cleanup.
  **PARTIALLY STALE (checked 2026-10-12):** the method
  (`entities/attributes.py:27-29`) no longer has any docstring at all — the
  "mainly for tests" claim it's accused of contradicting isn't there to
  contradict anymore. The underlying point still holds though: it's a
  bare `@staticmethod` with a test-sounding name, still used directly in
  production (`ui/scenarios.py:450,914-916`). Still worth the same
  decision (rename vs. accept as production API), just reframe as
  "undocumented, test-sounding name used in production" rather than
  "contradicts its own docstring."
- Several of the same balance-test target percentages (e.g. "precision=0.5
  centre: >95% scored") are written out in prose in **at least four
  independent places** (design docs, `physics.json` comments, the balance
  test assertions themselves, and `scripts/grid_search_kick_params.py`'s
  docstring) — any future retune has four places to remember to update, and
  will likely miss some. Consider a single source of truth (e.g. the test
  file itself, with docs/scripts referencing it rather than re-stating
  numbers). (Left as-is: picking the canonical location is a documentation-
  structure decision.)

---

## 7. Test coverage gaps

- **`tests/ai_scenario/` contains only `test_smoke.py`**, a single file,
  versus `tests/scenario/`'s 10 files for the engine. If AI scenario-level
  coverage is meant to eventually mirror the engine's scenario tier, this is
  a visible, significant gap (noted for completeness even though the `ai/`
  package itself was out of scope for deep review).
  **RESOLVED/SUPERSEDED (checked 2026-10-12):** `tests/ai_scenario/` now
  contains **27 test files** (e.g. `test_batched_rollout.py`,
  `test_dual_clip_ppo.py`, `test_masked_action_training.py`,
  `test_ppg_value_refit.py`, `test_val_episode_split.py`, and more),
  essentially at parity with `tests/scenario/`'s now-19 files. This gap no
  longer exists; no further action needed here.
- No test explicitly named/targeting exact-tangent or grazing-hit edge cases
  for `resolve_ball_block_by_inactive_players`'s ray-circle intersection math
  in [engine/collision.py](src/footballcoach/engine/collision.py) (quadratic
  `discriminant == 0` case) — worth confirming coverage exists somewhere, or
  adding it if not.
  **Still open (checked 2026-10-12):** grepped the whole `tests/` tree for
  `resolve_ball_block_by_inactive_players` — zero matches. Not only is the
  discriminant-==-0 edge case untested, the function has **no test coverage
  at all** under its own name (it may be exercised indirectly through
  scenario/balance tests, but nothing targets it specifically). Gap is
  real, and slightly worse than originally described.
- `tests/knowledge.md` itself documents a prior incident where a balance
  test saturated near 100% regardless of GK attributes, later tightened —
  a good sign it's been caught once, but there's no confirmation any other
  current balance test has the same "too easy, saturates near 0%/100%"
  problem; worth a follow-up audit given this exact failure mode has already
  occurred once. (Not independently re-audited this pass — out of scope for
  a documentation-only reconciliation; still an open to-do.)
- No visible confirmation that all `tests/balance/*.py` files uniformly use
  `N >= 1000` trials as the design docs mandate — worth a quick grep/audit
  to make sure no test regressed to a lower trial count (which could
  reintroduce flakiness).
  **CONFIRMED STILL TRUE, with specifics (checked 2026-10-12):**
  `N_TRIALS` is **200** in `tests/balance/test_pass_balance.py:25`, **500**
  in `tests/balance/test_shoot_balance.py:18`, and **2000** in
  `tests/balance/test_penalty_balance.py:26` — not uniform, and two of the
  three are well below 1000. Additionally,
  `test_pass_balance.py:119` uses a hardcoded `n_trials=50` for
  `test_pass_success_rate_table_across_precision_and_distance` (explicitly
  commented as "not a hard-target test," a visual table rather than a
  pass/fail assertion, so lower N there may be intentional) — but the
  200/500/2000 spread across the three hard-target files looks like
  genuine, unexplained drift rather than a documented design choice.
- `tests/balance/results/` (e.g. `latest_results.json`) is a committed
  build/output artifact in version control — likely intentional per the
  README's "inspect without re-running pytest" rationale, but worth
  confirming it's not accidentally accumulating diff noise on every
  balance-test run.

---

## 8. UI-specific notes (non-blocking)

- `Renderer`'s emoji-font detection in
  [ui/renderer.py](src/footballcoach/ui/renderer.py) (probing render height to
  detect bitmap emoji fonts) is inherently platform-fragile and has no test
  coverage confirming a font is actually found under a headless
  (`SDL_VIDEODRIVER=dummy`) CI environment. Not added yet: there is currently
  no test file/home for `ui/renderer.py` or `ui/camera.py` at all (no
  `tests/ui/` directory exists), so adding this smoke test means first
  deciding where UI tests should live — a small structural decision rather
  than a pure mechanical addition.


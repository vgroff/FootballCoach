# Function-Preserving Network Widening ("Seam Widening")

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

## 0. Status

**Not started — planning only.** Written up on request so the design is
captured somewhere durable before any implementation begins. Scope (which
layer(s) to widen first, script vs. other delivery mechanism) is not yet
decided — see §6.

## 1. The ask

Widen layers of the pretrained `DecisionNetwork`/`ExecutionNetwork` (give a
layer more units to grow model capacity for continued training) such that:

1. The network's output is **exactly unchanged** immediately after
   widening (no discontinuity in the loss/policy the moment training
   resumes), and
2. The new units are **not dead** — they must receive real, nonzero
   gradient from the very first training step, so they actually start
   learning rather than sitting inert forever.

This is the standard "Net2WiderNet" function-preserving network-growing
problem (Chen et al. 2016, "Net2Net: Accelerating Learning via Knowledge
Transfer"). The naive approach — zero-initializing everything about a new
unit — satisfies (1) but fails (2): a new unit's gradient depends on its
*outgoing* weight (the next layer's column that reads it) being nonzero, so
zeroing that column blocks gradient forever, and the unit never moves off
its useless initial state.

## 2. The recipe

For a widened layer, split the new parameters by direction:

- **New incoming weights** (the rows/columns producing the new units'
  *activations* from their existing inputs) get a normal **random** init —
  the same scheme already used for the rest of that layer (e.g.
  `nn.Linear`'s default Kaiming-uniform). **Not zero.**
- **New outgoing weights** (the columns of the **next** layer's weight
  matrix that would read the new units as additional input) get a **zero**
  init.

This is the opposite of the naive "zero the new unit's own weights"
approach, and it resolves the exact tension in §1:

- **Output unchanged at t=0**: the next layer computes
  `y = W_old @ h_old + W_new_out @ h_new + b`. Since `W_new_out = 0`,
  `h_new`'s value is irrelevant — `y` is identical to before widening,
  for every input, exactly (not approximately).
- **New units are not dead**: `dL/d(W_new_out) = dL/dy * h_new`, which is
  nonzero on the very first backward pass as long as `h_new` (computed from
  the new units' *randomly initialized* incoming weights) is nonzero for
  the given input. So `W_new_out` moves away from zero after step 1. From
  step 2 onward, `dL/d(W_new_in) = dL/dy * W_new_out * ...` also becomes
  nonzero, since `W_new_out` is no longer exactly zero. One-step bootstrap
  delay, not a dead end.

If the new unit uses a ReLU (true for every hidden layer in this
network — see §3), a random incoming-weight init means roughly half the new
units are already active (`h_new > 0`) for a typical input, so the
"boostrap" gradient in step 1 is not a rare event — it fires for close to
half the new units on close to every batch, immediately.

## 3. Where this is an exact fit in this codebase, and where it isn't

### 3.1 Exact fit: the plain `Linear -> ReLU` branches

Reading `src/footballcoach/ai/models/decision_network.py` and
`execution_network.py`: `self_mlp`, `ball_mlp`, `global_mlp`, `trunk`, every
scalar/categorical/continuous output head, `decision_mlp` (execution-only),
and `value_head` (when `value_hidden_dim > 0`) are **all** plain
`nn.Linear` + `nn.ReLU()`, with **no normalization layer** anywhere in that
signal path. No `LayerNorm`/`BatchNorm` means no running statistic gets
perturbed by adding new units — the recipe in §2 gives **exact** function
preservation for these layers, not an approximation. Concretely, these
hidden widths are safe, "v1" widening targets:

- `self_mlp_hidden` (`ai_config.json` → `network.self_mlp_hidden`, default
  64) — feeds into `trunk`'s concatenated input at a fixed offset (see
  §4.1 for why this matters).
- `ball_mlp_hidden` (`network.ball_mlp_hidden`, default 32) — same
  concatenation caveat.
- `global_mlp_hidden` (`network.global_mlp_hidden`, default 32) — same.
- `trunk_hidden` (`network.trunk_hidden`, default 256) — widening the
  **first** `trunk` layer's output means the **second** `trunk` layer's
  input also grows; both the first layer's new incoming weights (random)
  and the second layer's new-column outgoing-to-first-layer weights (zero)
  need touching (a two-layer version of the same recipe, applied twice in
  sequence — see §4.2). The trunk's output then also feeds *every* head
  AND `value_head` — each of those heads' weight matrices grows a new zero
  column too (widening `trunk_hidden` has the largest fan-out of any single
  target here).
- `decision_mlp_hidden` (execution-network-only, `network.decision_mlp_hidden`,
  default 64) — same pattern as `self_mlp_hidden`.
- `value_hidden_dim` (when nonzero) — the value head's own optional hidden
  layer.

### 3.2 Not an exact fit: `entity_embed_dim` / the attention path

`entity_encoder.py`'s `EntityEncoder` — used inside **both** networks — is
**not** a clean fit. It contains:

- `nn.LayerNorm(embed_dim)` (`ln_query`, `ln_kv`, and `ln_inter` when
  inter-player attention is enabled): LayerNorm normalizes **across the
  full embedding vector** (mean/variance over all `embed_dim` channels).
  Adding new channels with a nonzero (random-init-driven) activation
  changes that mean/variance for the *existing* channels too, even though
  the new channels' own downstream outgoing weights are zero — so the
  existing channels' normalized values shift, breaking exact output
  preservation the moment normalization is involved. This is the exact
  "Net2Net doesn't compose cleanly with normalization layers" caveat noted
  in the original Net2Net literature.
- `nn.MultiheadAttention(embed_dim, num_heads)`: bundles a fused
  `in_proj_weight` (`3*embed_dim x embed_dim`, for Q/K/V) and `out_proj`
  (`embed_dim x embed_dim`), and requires `embed_dim % num_heads == 0`.
  Widening `embed_dim` isn't "add new output units to a Linear" — every
  head's per-head dimension (`embed_dim / num_heads`) changes too unless
  `num_heads` is grown in proportion, which is a structurally different
  operation (adding whole new attention heads, which then need *their own*
  zero-init-output trick applied at the `out_proj` level, not a per-channel
  one).

**Conclusion: scope v1 to §3.1's plain-MLP targets only.** `entity_embed_dim`
widening is a real, separate, higher-effort project (approximate at best,
without a more involved LayerNorm-aware or head-count-aware construction)
and should not block getting value out of the easy, exact cases first.

## 4. Implementation complexity specific to this codebase

### 4.1 Concatenated trunk input — widening must target the right offset

`trunk_input_dim = entity_embed_dim + self_mlp_hidden + ball_mlp_hidden +
global_mlp_hidden` (`decision_network.py`), and analogously
`entity_embed_dim + self_mlp_hidden + ball_mlp_hidden + global_mlp_hidden +
decision_mlp_hidden` for `ExecutionNetwork`. `trunk[0]`'s weight matrix
(`nn.Linear(trunk_input_dim, trunk_hidden)`) has its input axis laid out as
**four (or five) concatenated sub-blocks in a fixed order**. Widening e.g.
`self_mlp_hidden` from 64 to 96 means:

- `self_mlp[0]` (the `Linear(self_dim, 64)`) gets 32 new **rows** (output
  units) — random init.
- `trunk[0]` (the `Linear(trunk_input_dim, trunk_hidden)`) gets 32 new
  **columns** (input units) — zero init — inserted at the position
  corresponding to `self_mlp`'s segment within the concatenation, i.e.
  immediately after the `entity_embed_dim` columns and before the
  `ball_mlp_hidden` columns, **not** appended at the end of the whole
  matrix. Getting this offset wrong silently corrupts the model (the new
  zero columns would multiply against the *wrong* sub-vector's activations,
  and the *old* columns would shift to the wrong sub-vector too) without
  raising any shape error, since the column count still matches — this is
  the single most likely source of a silent bug in this whole plan and
  needs its own dedicated unit test (see §7).

`ExecutionNetwork`'s `decision_mlp` segment sits at the *end* of its
five-block concatenation, which is the one case where "append at the end"
happens to be correct — still worth writing generically rather than relying
on that coincidence, since the four-block `DecisionNetwork` case doesn't
have that luxury.

### 4.2 Two-layer trunk — widening `trunk_hidden` touches three tensors, not one

`trunk = Sequential(Linear(trunk_input_dim, trunk_hidden), ReLU,
Linear(trunk_hidden, trunk_hidden), ReLU)`. Widening `trunk_hidden` means:

1. `trunk[0]`: new **output rows** (from `trunk_input_dim`) — random init.
2. `trunk[2]`: new **input columns** (reading `trunk[0]`'s new outputs) —
   zero init, AND new **output rows** (since `trunk[2]`'s output is also
   `trunk_hidden`-sized and feeds every head) — random init. These are two
   independent parts of the *same* weight tensor (rows vs. columns) and
   both need touching, in the same widening pass, for `trunk[2]` alone.
3. Every head (`shoot_logit`, ..., `value_head`, `latent_vector`, etc.) and
   (execution-net only) `decision_query_proj`: new **input columns** —
   zero init. A head being zero-init on its new columns means that head's
   output is *also* exactly unchanged by the trunk widening, which is
   necessary — the whole point is that widening is invisible to everything
   downstream, all the way to the actual action distributions.

### 4.3 Adam optimizer state has no entry for new params

New parameters have no history in `self.optimizer`'s per-param `m`/`v`
running moment estimates — there's no meaningful way to "fill in" prior
Adam state for a tensor that didn't exist before. `PPOTrainer.load_checkpoint`
already has a `reset_optimizer: bool` path (used today for `--reset-optimizer`,
also already handles a "saved param-group count differs from current"
mismatch by skipping optimizer-state restore entirely and warning — see
`load_checkpoint`'s existing `saved_n_groups != current_n_groups` branch).
A widened checkpoint should always be loaded with `--reset-optimizer` (or
via that existing shape-mismatch fallback) — old Adam statistics for the
*unwidened* portions of a layer, computed under the old shapes, don't cleanly
carry over either even where the tensor identity nominally survived, since
the widened tensor is now a different shape than what Adam's state dict
would expect for that key. Adam simply restarts (all params, not just new
ones) — acceptable, since the point of widening is to keep the *policy's
behavior* continuous, not the optimizer's momentum.

### 4.4 `_load_state_dict_tolerant` is the wrong tool for this

`ppo_trainer.py`'s existing `_load_state_dict_tolerant(module, ckpt_sd,
label)` (used by `load_checkpoint` for every net) already tolerates a
shape mismatch — but by **skipping** the mismatched tensor entirely and
leaving the *fresh random init* in place for that whole tensor. That's
correct behavior for its actual purpose (loading an old checkpoint into a
network whose architecture changed for unrelated reasons, e.g. a new
value-head input column from an added side-channel) but is **not**
function-preserving: it would re-randomize an *entire* widened weight
matrix (both the old, previously-trained columns/rows AND the new ones),
throwing away all pretraining for that layer. Widening must be done as an
explicit, standalone **checkpoint transform** (§5) that surgically extends
each tensor in place (old sub-block copied verbatim, new sub-block
randomly or zero initialized per §2/§4.1/§4.2) — never routed through
`_load_state_dict_tolerant`'s skip-on-mismatch path.

### 4.5 Shared vs. independent encoders

`DecisionNetwork`/`ExecutionNetwork`'s `from_config()` classmethods both
accept optional `shared_entity_encoder`/`shared_ball_mlp`/`shared_global_mlp`
args for weight-sharing between the two networks, but the actual PPO
training construction site (`PPOTrainer.__init__`, both the primary
`DecisionNetwork.from_config()` / `ExecutionNetwork.from_config()` calls)
does **not** pass them — `decision_net` and `execution_net` each get their
own independent `entity_encoder`/`ball_mlp`/`global_mlp` weights today. This
simplifies v1: widening `ball_mlp_hidden` for `decision_net` and for
`execution_net` are two **independent** transforms (different tensors, no
shared-weight aliasing to worry about). Confirm this hasn't changed before
implementing (grep for `shared_entity_encoder=` at the `PPOTrainer.__init__`
construction site) — if sharing is ever turned on there, a widened shared
module must be widened exactly once and both networks' state dicts pointed
at the same result, not widened twice independently.

The optional `self.value_net` (`separate_value_net=True`, its own
independently-constructed `ExecutionNetwork.from_config(...)`) is a further
separate copy again, with its own `trunk_hidden`-equivalent sizing — needs
its own transform call if in use, same recipe.

## 5. Proposed delivery mechanism: standalone checkpoint-transform script

A new script, e.g. `src/footballcoach/ai/scripts/widen_checkpoint.py`
(exact location/naming TBD at implementation time), that:

1. Loads an existing checkpoint (`decision_net`/`execution_net`/optionally
   `value_net` state dicts) the same way `PPOTrainer.load_checkpoint` does.
2. Takes explicit target widths for whichever of §3.1's dims the user wants
   grown (e.g. `--self-mlp-hidden 96 --trunk-hidden 384`), defaulting every
   *unspecified* dim to its current value (a true no-op for anything not
   named).
3. For each widened dim, applies the §2 recipe tensor-by-tensor: build a
   **new**, larger tensor, copy the old tensor into the corresponding
   sub-block unchanged, fill the new incoming sub-block with a fresh
   `nn.Linear`-equivalent random init (reuse `nn.Linear`'s own reset
   scheme — e.g. instantiate a throwaway `nn.Linear` at the new shape and
   read its initialized weight/bias for the new rows, rather than
   hand-rolling an init formula that could drift from PyTorch's own
   default), and the new outgoing sub-block with zeros — using §4.1's
   segment-offset bookkeeping for every concatenated-input layer
   (`trunk[0]` for `DecisionNetwork`; `trunk[0]` for `ExecutionNetwork`,
   which has one extra segment) and §4.2's two-tensor treatment for
   `trunk_hidden` specifically.
4. Also updates `ai_config.json`'s `network` section to the new widths (or
   prints a diff for the user to apply) — **the config is the source of
   truth every `from_config()` call reads from next run**, so the
   transformed checkpoint and the config must agree, or the next training
   run will construct networks at the OLD shape and `load_checkpoint`
   will silently fall back to `_load_state_dict_tolerant`'s skip-and-
   reinit path for every widened tensor (see §4.4) — exactly undoing the
   point of this whole exercise, with no error raised.
5. Writes a new checkpoint file (never overwrites the input — this is a
   config/architecture-changing transform, not an in-place update), and
   drops the `optimizer`/`value_net_optimizer` keys entirely (§4.3) so a
   forgotten `--reset-optimizer` flag on the next `load_checkpoint` call
   can't accidentally restore stale, wrong-shaped Adam state — the training
   script already tolerates a missing `optimizer` key.
6. Logs, per transformed tensor, old shape → new shape, so a run's console
   output/log file has a permanent record of exactly what was widened and
   when (mirroring this codebase's general preference for traceable,
   explicit diagnostics over silent transforms).

## 6. Open — needs a decision before implementation starts

- **Which layer(s) to widen first.** `trunk_hidden` has the largest
  fan-out (touches every head) and is the most likely actual bottleneck
  for added capacity, but is also the most involved single transform
  (§4.2). `self_mlp_hidden`/`ball_mlp_hidden`/`global_mlp_hidden` are
  simpler (single incoming tensor + one concatenated-offset outgoing edit)
  and would be a lower-risk first implementation to prove the mechanism
  end-to-end before tackling `trunk_hidden`.
- **`decision_net` vs. `execution_net` vs. both.** Independent transforms
  (§4.5) — can be scoped to just one network first.
- **Whether to also support widening via a config-only "grow on next
  `from_config()` construction" path** instead of (or in addition to) an
  offline checkpoint script — the offline-script approach (§5) is simpler
  and keeps `from_config()` itself free of any widening-specific logic, at
  the cost of being a separate manual step in the workflow. No strong
  reason surfaced yet to prefer the more invasive option.
- **`entity_embed_dim`/attention widening (§3.2)** — explicitly deferred,
  not part of this plan's v1 scope.

## 7. Tests (once scope from §6 is picked)

- **Exact-preservation regression test**: build a small `DecisionNetwork`/
  `ExecutionNetwork` pair, run a forward pass on a random batch, apply the
  widening transform, run the SAME batch through the widened network, and
  assert the two outputs are bit-for-bit (or `atol=1e-6`) identical for
  every field in `DecisionHeadsRaw`/`ExecutionHeadsRaw` — the core
  correctness property this whole plan exists to guarantee.
- **Gradient-flow test**: after widening, run one optimizer step on a loss
  that depends on the new units' output, then assert the new incoming
  weights' `.grad` is nonzero (proves §2's "not dead" claim directly,
  rather than just trusting the math).
- **Concatenation-offset test (§4.1)**: the single highest-risk spot for a
  silent bug — construct a case where `self_mlp_hidden`/`ball_mlp_hidden`/
  `global_mlp_hidden` are all *different* sizes (so a wrong offset
  provably scrambles the wrong sub-vectors, rather than accidentally
  working because two segments happen to be the same width) and assert
  exact-preservation still holds.
- **Checkpoint round-trip test**: transform a real (small, test-fixture)
  checkpoint, reload it via `PPOTrainer.load_checkpoint(..., reset_optimizer=True)`
  end-to-end (real code path, not a hand-rolled loader), and confirm no
  shape-mismatch warnings are logged (i.e. `_load_state_dict_tolerant`
  reports zero skipped keys — proof the transform script's output shapes
  exactly match what a `from_config()` build at the new widths expects).

## 8. Files likely touched (not exhaustive — plan only)

- New: `src/footballcoach/ai/scripts/widen_checkpoint.py` (or similar) —
  the transform script itself (§5).
- `src/footballcoach/ai/config/ai_config.json` — new widths written/
  diffed here (§5 step 4).
- Test file(s) under `tests/ai_scenario/` or `tests/ai_unit/` for §7.
- No changes anticipated to `decision_network.py`/`execution_network.py`/
  `entity_encoder.py` themselves, or to `ppo_trainer.py`'s training loop —
  this is purely an offline, between-runs transform on saved weights; the
  network *classes'* `forward()`/`from_config()` code doesn't need to know
  widening ever happened, it just gets constructed at the new
  `ai_config.json` widths next run like any other config change.

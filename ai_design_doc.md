# AI Design Document

Status: design agreed, implementation not yet started. This document is the
canonical write-up of the neural-network / PPO player-AI design discussed in
chat (the source-of-truth `ai_plan.md` in the repo root is the user's
personal scratch notes and is never read by an AI agent - see
`/memories/repo/rules.md` - this document is the AI-facing distillation of
that discussion, safe to read and extend in future sessions).

This is a living document - update it as decisions change or firm up during
implementation, the same way `engine/knowledge.md` etc. are kept current.

## 1. Goal / framing

Football Coach's engine and rules-based `orders`/`actions` layer already
exist and are complete (see root `README.md` and `engine/knowledge.md`).
The next phase is to control players with neural networks trained via PPO
(Proximal Policy Optimization), so that:

- The human "coach" can eventually set tactics/formations and nudge
  individual player decisions (e.g. "+20% shoot probability") rather than
  micromanaging every action.
- Players act autonomously and (eventually) competently within a full
  match, with a curriculum of increasingly complex training scenarios
  bootstrapped from the existing rules-based orders.

## 2. High-level architecture

### 2.1 Two networks per player, shared/common weights

Each player is driven by **two networks run in sequence each decision tick**:

1. **Decision network** - looks at the world and decides *what kind of
   thing to do* (shoot? pass? move? tackle? mark? hold position?) and the
   high-level parameters of that (pass target, move region, etc).
2. **Execution network** - given the decision network's full output, decides
   the actual low-level motor output for this tick (move direction,
   jog/sprint, kick probability/direction/power/spin, tackle attempt).

All players share **one set of weights** for each network (not one network
instance per player) - i.e. a single decision-network policy and a single
execution-network policy, both trained across all players/positions
simultaneously, at least "for now" per the user's framing (specialising by
position/role is a possible future extension, not in scope yet).

### 2.2 Observation encoding

Inputs to both networks include, at minimum:

- **Ball state**: position, velocity, spin.
- **Every player's state** (self + all 21 others in a full 11v11 match):
  position, velocity, stamina, attributes, plus flags:
  - which team the player is on
  - whether they currently have possession
  - whether they are in "inactive" state (tackled cooldown)
  - whether they are currently in `CONTROLLING_BALL` (first-touch) state
  - whether they are a goalkeeper
  - (room for more flags as needed, e.g. current order/action)
- **Match context**: current score, time left in the game, pitch
  length/width, goal size, box/penalty-area dimensions, ball bounce
  (restitution) coefficient.
- **Attacking direction flag** - confirmed by user: each player's
  observation must encode which way they are attacking (`Team.LEFT` attacks
  +x / `Team.RIGHT` attacks -x, per `engine/offside.py`'s convention), so
  that a relative/normalized encoding still lets the network reason about
  "towards my goal" vs "towards their goal" consistently, and so pitches can
  potentially be mirrored for data augmentation later without confusing the
  network about attacking direction.

**Relative encoding.** Rather than raw absolute (x, y) coordinates for the
ball and other players, use **relative offset (dx, dy) from the observing
player, normalized by pitch dimensions, plus a redundant distance
scalar** - not pure polar (angle, distance), because raw angles wrap around
discontinuously (a small perturbation across the 180°/-180° boundary looks
like a huge input jump, which NNs handle poorly). A (dx, dy) vector already
encodes both direction and distance smoothly with no discontinuity, and
distance is included redundantly anyway since it's cheap and helps the
network (avoids relying purely on vector magnitude implicitly).

**Full padded entity layout, decided now rather than deferred.** Per user's
explicit choice, the observation's "other players" block is sized for the
**full 21-other-player (11v11) case from the very start**, even though
early curriculum scenarios (1v1, 1v2 etc.) only ever populate 1-2 of those
slots. Unused slots are zero-filled and carry an explicit "exists" mask bit
(see masking, section 2.4) so the network can distinguish "no player here"
from "a real player at the origin." This avoids reworking the observation
space (and re-training from scratch) when the curriculum scales up to 3v3
and 5v5 later - the trade-off is more up-front complexity (masking must be
correct even in the simplest scenarios) in exchange for architectural
continuity across the whole curriculum. Also, populate these slots randomly each time, so that the network learns this invariance.

### 2.3 Shared-weight entity encoder + attention

For the 21-other-player block: each other player's feature vector is passed
through **the same small per-entity MLP** ("shared weights" - the user's
own intuition, confirmed as a sound approach and standard practice in
entity-based multi-agent RL, e.g. AlphaStar/OpenAI Five style architectures)
producing one embedding per player. This is combined via:

- an **attention layer** over all 21 embeddings (query from the observing
  player's own embedding, keys/values from the other-player embeddings), so
  the network can learn to focus on relevant players (the ball carrier, the
  nearest opponent, a marked target, etc.) rather than needing a fixed
  positional mapping between "slot index" and "which player."
- Max/mean pooling is a simpler complementary or fallback option, but
  attention is preferred here because per-player logits are *also* needed
  directly for the pass-target and tackle-target softmax heads (section
  2.5), so the network needs per-entity outputs anyway, not just a single
  pooled summary vector.

Rough shape: `(other-player features) -> shared per-entity MLP -> per-entity
embeddings -> attention (query = self embedding) -> pooled context vector`,
concatenated with `(self features) -> dense` and `(global/match context) ->
dense`, then through shared trunk layers before splitting into the decision
heads.

Note on compute: sharing weights across entities does **not** reduce
per-tick FLOPs (the shared MLP still runs once per entity present) - the
benefit is in reduced parameter count / combined training signal (all 21
"other player" examples train the same weights) and architectural
flexibility across roster sizes, not raw inference speed. This was
discussed explicitly and should not be assumed to be a speed optimisation.

### 2.4 Decision network outputs (the "heads")

The decision network outputs, per player per decision tick:

- **Shoot probability** (independent sigmoid, not part of a softmax - see
  section 2.6).
- **Pass probability** + pass target: a softmax over up to 21 "other
  player" slots (fixed max roster size, not tied to the current scenario's
  actual player count - see masking below).
- **Move probability** + region to move to (region modelled as a square,
  minimum 1m x 1m up to 4m x 4m) + arrival speed.
- **Tackle probability** + tackle target: softmax over up to 21 slots (same
  masking scheme as pass target - "11 softmaxed outputs" in the user's
  original note referred to a smaller roster size example; the general
  mechanism is a fixed max-roster-size softmax regardless of exact number).
- **Get-possession probability** - always constrained to be at least the
  tackle probability (can be higher, e.g. when going for a loose ball
  rather than dispossessing an opponent).
- **Mark probability** + mark target (softmax over other players, same
  masking). Reward shaping: reward being close to the mark target and
  positioned between them and the ball more aggressively than plain
  proximity, and reward interceptions specifically.
- **Hold-position probability** - behaviourally identical to Move (uses the
  same move-region output), but the network is penalised much more
  aggressively for straying outside the designated region while this head
  is active, versus Move's softer region preference.
- **Region of play** - a region (minimum 15m x 15m, up to 40m x 40m)
  denoting a preferred general playing area; players are rewarded for
  staying in/near this region but only mildly, not harshly, penalised for
  leaving it.
- **Attack/defence weighting** (0-1 scalar) - see section 2.7 for the
  smoothing/EMA design agreed for this.
- **Latent vector** - a free vector passed through to the execution
  network and (per the curriculum, section 4) trained even during phases
  where most other heads are frozen, so it can carry information the
  execution network needs before the rest of the decision network is
  unfrozen.

**Masking mechanism (explained per user request, now understood/agreed).**
Because the softmax pass/tackle/mark target heads must always output a
fixed-size vector (21 slots) regardless of how many players actually exist
in the current scenario (1v1 up to 11v11), unused slots are masked out
*before* the softmax: `masked_logits = raw_logits + mask`, where
`mask[i] = 0` for a slot with a real player this tick and `mask[i] = -inf`
for an empty/non-existent slot. Since `exp(-inf) = 0`, masked slots receive
exactly zero probability mass and can never be sampled/selected, while the
remaining real slots' softmax is computed purely relative to each other
(mathematically identical to running softmax over just the real slots, but
implemented with a fixed-width tensor so the network architecture doesn't
need to change size between scenarios). This is a standard technique for
variable-size discrete choice sets in fixed-size networks. The same "exists"
mask bit doubles as the "is this slot a real player" flag mentioned in
section 2.2's padded entity layout.

### 2.5 Execution network outputs

Given the full decision-network output (all heads, including the ones not
"selected" this tick) plus the usual observation inputs, the execution
network outputs the actual low-level per-tick motor actions, mapping onto
the existing `orders.py`/`actions.py` action surface:

- move direction
- jog/sprint choice
- kick probability
- kick direction, power, spin
- perform-tackle probability

The engine must guardrail against illegal actions (e.g. kicking without
possession, tackling out of range) - the AI is *punished* for attempting an
illegal action rather than the engine silently ignoring it, so the network
actually learns not to attempt them. (Whether the existing engine already
guards all of these cases needs to be checked/extended during
implementation - not yet audited as of this document.)

### 2.6 Action selection: gating rule vs. PPO log_prob (explained per user request)

This was a point of confusion, resolved as follows, and it's important
enough to restate precisely for future readers of this document/code:

There are **two distinct, separate concerns** which must not be conflated:

1. **In-game action gating (deterministic, non-differentiable, runs
   post-hoc):** Each of the ~6 action-probability heads (shoot, pass, move,
   tackle, get-possession, mark, hold-position) is an **independent sigmoid
   output** (not a softmax across each other) - explicitly *not* mutually
   exclusive at the raw-probability level, since e.g. get-possession and
   tackle can sensibly co-occur. To decide what the player's body actually
   does *this tick*, apply a simple rule: if any head's probability exceeds
   50%, the single highest one is treated as "selected" (effectively forced
   to 1.0) and all others as "not selected" (forced to 0) for the purposes
   of driving the execution network / engine this tick. This is **just a
   downstream post-processing / gating rule** on top of the raw
   probabilities - it has no bearing on the training math and is not part
   of the differentiable computation graph.

2. **PPO training (the log_prob the algorithm actually needs):** PPO's
   clipped surrogate objective needs, for whatever action was actually
   taken, the probability the *current* policy assigns to that action, so
   it can form the ratio `π_new(a|s) / π_old(a|s)`. With **independent
   Bernoulli distributions per head** (which is exactly what independent
   sigmoids give you), this is simple and well-defined: each head's
   `log_prob` is `log(p)` if that head "fired" or `log(1-p)` if it didn't,
   with **no argmax/threshold anywhere in this calculation**. Each head can
   be trained as its own independent Bernoulli action (own advantage
   signal/reward shaping per head, as described throughout section 2.4's
   per-head reward notes) using completely standard PPO mechanics.

**The trap to avoid:** if the winner-take-all gating rule from (1) were
mistakenly folded into the differentiable/backprop path (e.g. trying to
make the *choice of which head wins* itself trainable via backprop through
the argmax), you'd hit the standard non-differentiability problem of
hard argmax/threshold operations, which is normally worked around with
tricks like Gumbel-softmax / straight-through estimators. **None of that
machinery is needed here** as long as the two concerns are kept cleanly
separate in the implementation: train each head's raw Bernoulli probability
via ordinary PPO log_prob/advantage/clipping math first (this is where
backprop happens), and only *afterwards*, entirely outside the gradient
graph, apply the deterministic winner-take-all rule to translate those
probabilities into the one action actually executed against the engine this
tick. Keep this separation explicit in code (e.g. a `select_action()`
utility that is plain Python/no-grad, clearly distinct from the policy's
`forward()`/`log_prob()` methods) to avoid ever needing a straight-through
estimator by accident.

### 2.7 Attack/defence weighting - long-latency scalar (agreed design)

The attack/defence weighting output (0-1) is meant to represent a
long-term strategic stance that changes reward weighting during training
(higher defence weight rewards tackling/keeping possession more; higher
attack weight rewards progressing upfield/shooting/scoring more), and per
the user's original note, should change only slowly/with high latency
under normal play, but be allowed to shift more quickly right after a goal.

**Agreed implementation approach:** rather than trying to bake this
smoothing behaviour *inside* the neural network itself, implement it as an
**exponential moving average (EMA) filter external to the network**:

- The network's raw sigmoid output for this head is the *instantaneous
  target* value.
- A smoothed value is maintained per-player (outside the network, in the
  environment/policy-wrapper code) via a standard EMA:
  `smoothed = alpha * smoothed_prev + (1 - alpha) * raw_output`, with a
  slow-moving `alpha` (close to 1) in normal play - shifts to a
  fast-moving `alpha` (further from 1) for a short window immediately
  following a goal, then reverts to the slow value.
- The **previous tick's smoothed value is fed back into the network as an
  input** on the next decision tick (part of the "current player state"
  observation block, self features), so the network is always aware of
  what attack/defence stance is actually in effect (its own raw output
  alone would not tell it that, since the smoothed value lags behind).

This keeps the mechanism simple, fully controllable/tunable outside the
network (no need to reverse-engineer or constrain the network's internal
dynamics to produce a particular latency), and transparent to debug - it's
literally just an EMA with a state-dependent smoothing constant.

## 3. RL algorithm and implementation choice

### 3.1 Decision: custom PPO loop in PyTorch (not stable-baselines3)

**Chosen: a custom PPO training loop, hand-written in PyTorch**, rather
than using stable-baselines3 (SB3) or another off-the-shelf RL library.
This decision was discussed explicitly and **this section documents why,
and notes that it is revisitable** if it turns out to be the wrong call
during implementation.

**Why not SB3:** SB3 provides a mature, well-tested PPO implementation
(GAE advantage estimation, clipped surrogate loss, value function, entropy
bonus, minibatch/epoch scheduling, LR/clip-range annealing, vectorized
rollout collection, checkpointing, TensorBoard logging) - all correct and
widely used "get this subtly wrong and training silently underperforms"
plumbing, for free. However, SB3's `ActorCriticPolicy` and action-space
handling assume a single `Discrete`/`MultiDiscrete`/`Box` action space; it
has no built-in action distribution for "9ish independent sigmoid
(Bernoulli) heads + multiple masked-categorical (softmax-over-variable-N)
heads + several continuous vector heads (move target, arrival speed,
region box, attack/defence scalar, latent vector), all combined into one
action, with masking on the categorical slots." Using SB3 here would
require subclassing `ActorCriticPolicy` to inject a custom
`features_extractor` (the entity encoder/attention block) *and* writing a
custom `Distribution` class combining Bernoulli + masked-Categorical +
Normal/Beta sub-distributions with a combined `log_prob`/`entropy`/`sample`
- at which point most of the actual complexity is still bespoke code
fighting against a framework's internal shape assumptions (buffer storage,
`predict()`, action-space validation), rather than work saved by the
framework.

**Why custom is preferred here:** given the user's explicit choice to
build the full-complexity action space (all decision heads from day one)
and the full padded entity/attention encoder from day one (not deferred),
SB3's main selling point - fast path to training a *simple* action space -
doesn't really apply to this project. A custom loop means the network,
action distribution, and masking are just our own PyTorch modules from the
start, with no adapter/translation layer, and it stays easy to extend
later for coach-influence adjustments (e.g. "+20% shoot probability" is
just arithmetic on our own tensors, not something that has to be threaded
through SB3's `predict()`/policy internals).

**Trade-off acknowledged:** a custom loop means we own writing (and
correctly debugging) the PPO algorithm itself - rollout buffer, GAE(λ)
advantage estimation, the clipped surrogate objective, value loss, entropy
bonus, multiple epochs over shuffled minibatches, gradient clipping,
learning-rate/clip-range scheduling. This is well-documented and there are
good single-file reference implementations to follow the structure of
(e.g. CleanRL's `ppo_continuous_action.py` style - explicit, readable, not
a heavyweight abstraction layer), but subtle bugs in this kind of code
(GAE errors especially) tend to be *silent* - training just plateaus at
mediocre performance rather than crashing - so care and testing/validation
against toy problems is warranted before trusting results on the full
football task.

**This choice can be revisited.** If the custom loop proves to be a
maintenance burden, or a simpler/partial action space turns out to be
preferable after all (e.g. if the MVP scope narrows further), switching to
SB3 (or another library) remains an option - nothing about the engine/env
design below is SB3-incompatible, it would just require the adapter layer
described above. Any future agent/session picking this back up should read
this section before assuming the custom-loop decision is immutable.

### 3.2 Stack

- PyTorch (network modules, custom PPO loop). Not yet added as a
  dependency in `pyproject.toml` as of this document - to be added when
  implementation begins.
- The existing `Match` (from `engine/match.py`) is the environment the PPO
  loop drives; no separate physics/simulation reimplementation is needed -
  per the existing engine's explicit design goal (see root `README.md`,
  "The engine is UI-agnostic... a pygame renderer or an RL training loop
  are both future consumers of the same `Match.step()` API").

## 4. Training curriculum (as discussed, restated for reference)

The user's proposed curriculum, in order:

1. **General play-toward-ball AI** (1v1, one player each team): decision
   network's specialised-execution-related outputs are frozen; only the
   latent vector is trained. Random placement, attributes, stamina,
   running direction; ball randomly placed/moving with random spin and a
   randomised (mostly-normal, sigma 0.08) restitution coefficient; random
   time remaining (90% normal 0-120 min, 10% low 1-20s, actually ending the
   scenario at zero). Reward: closing distance to ball, gaining/keeping
   possession, progressing the ball toward the opponent's goal; scenario
   ends on box possession or a 2-minute cap; punished if the ball goes out
   after being touched. Opponent progression: immobile -> sometimes
   rules-based AI -> sometimes older AI generations. During this phase,
   the Move/Get-Possession decision-head *targets* used for training are
   set directly from rules-based logic (Get Possession when the ball is
   loose; Get Possession + Tackle when the opponent has it; Move toward
   the box when this player has it) - i.e. supervised-style bootstrapping
   of those specific heads even while "training" via the RL loop.
2. **Shooting**: scenarios spanning empty-goal, keeper-in-goal (rules-based
   Save order), static defender, attacker+defender, free kicks, with
   randomised-but-reasonable parameters/positions/attributes. Reward: time
   to shoot (faster better), shot on target, goal scored (best). Decision
   network specifically rewarded for outputting Shoot > 50%; latent vector
   also trained. Bootstrapped from the general network's weights (or a
   single shared network); can start by playing against/learning from the
   rules-based Save/Shoot orders, optionally extended with a shoot
   direction parameter rather than always aiming centre.
3. **Passing, Moving, Tackling** (similar structure): immobile players
   allowed; can use the shooting/moving AI to help train tackling (tackling
   trained after those, per this ordering); moving scenarios should include
   navigating around other players (static, moving, deliberately
   obstructing) with reaching an endpoint quickly as the reward.
4. **Continuous aggregation**: each new focus phase's training set
   continues to include earlier scenario types (shooting, passing, moving,
   etc.) to avoid catastrophic forgetting while a new skill is being
   emphasised.
5. **Rest of the decision network**: unfreeze all non-latent heads; retrain
   using the same scenarios but now learning the *decisions themselves*
   (not just hardcoded target heads); progress to chained-order scenarios
   (pass -> move -> shoot, etc.), first with no defenders then immobile
   ones. Introduce negative/probability-smoothing training on decisions
   that went poorly (e.g. an awkward-angle shot against a good keeper
   should be discouraged in favour of repositioning). Also train the
   *inverse*: if a head's probability is deliberately held below 50% in a
   scenario, the execution network is punished for still carrying out that
   action (e.g. shooting on goal while shoot-probability is suppressed, or
   passing to a teammate while pass-probability is suppressed) - this
   "forces" the decision and execution networks to actually rely on these
   nodes for the corresponding behaviour, rather than the execution network
   learning to ignore them.
6. **Simultaneous attack/defence training** once both sides are
   independently competent.
7. **RNG reduction schedule**: start slightly boosted (~0.55) to make early
   learning easier, gradually reduced to the game's default expected value
   (0.3) as training progresses.
8. **Scale-up**: eventually move to full 3v3 and then 5v5 with everything
   running fully end-to-end (no frozen heads, no hardcoded bootstrapping).

Throughout, rules-based `orders`/`actions` are explicitly used both as a
source of positive/bootstrapping examples and as a scripted adversary/ally,
consistent with the existing engine's design.

Pitch/goal randomisation during training: 80% standard pitch/goal/box
dimensions, 20% smaller pitches (down to 1/3 size, with some variation in
side ratio) and/or smaller goals (down to half size).

## 5. Agreed MVP scope (first concrete implementation target)

Two initial experiments, chosen by the user as the fastest path to a
working end-to-end training loop, to validate the infrastructure before
building out the full curriculum:

1. **GetPossession/Move experiment** - decision network chooses between
   Move and GetPossession (with the associated move target/region/arrival
   speed parameters), in a 1v1-style scenario as described in curriculum
   phase 1. Per the user's explicit preference (overriding the initially
   proposed narrower "binary + target only" scope), **all decision-network
   heads are present in the network's input/output from the very
   beginning**, even heads that are irrelevant/unused/untrained in this
   specific experiment (matching the user's stated general principle of
   training with all inputs and outputs present from day one, per
   `ai_plan.md`'s notes) - heads not relevant to this experiment are simply
   not the focus of the reward signal, rather than being architecturally
   absent.
2. **Shoot experiment** - decision network trained to output Shoot
   probability appropriately (per curriculum phase 2); the goalkeeper
   remains rules-based (`actions.save`) throughout. Per the user's explicit
   choice, **both the decision (shoot probability) and the execution
   network's aim_point/power/spin are trained together** from the start
   (not decision-only with the existing rules-based `actions.shoot()`
   execution as an interim simplification).

**Entity encoder scope for the MVP:** per the user's explicit choice, the
**full padded 21-other-player entity layout (with existence masking) is
built now**, even though these two MVP experiments only ever populate 1-2
of those slots - see section 2.2's rationale (avoids reworking the
observation/encoder architecture later when scaling to 3v3/5v5).

## 6. Open items / not yet decided

These are flagged so a future session doesn't assume they're settled:

- Exact reward function coefficients/shaping for each head (only
  qualitative direction has been agreed so far, e.g. "reward closing
  distance to ball," not exact formulas).
- Exact network sizes/hyperparameters (layer widths, attention head count,
  learning rate, clip range, GAE lambda/gamma, etc.) - to be chosen during
  implementation and tuned empirically, likely exposed via config similar
  to the engine's `physics.json`/`attributes.json` pattern.
- Whether/how the existing engine already guardrails all illegal actions
  the execution network could attempt (kicking without possession, etc.) -
  needs an audit; extend `engine/` as needed and punish illegal attempts in
  the reward function. - Developer note: yes, they engine should definitely guard illegal actions like kicking without possession
- How often the networks run relative to the engine's fixed 1/30s tick -
  user confirmed this should be a tunable parameter (networks need not run
  every tick), but no specific default has been chosen yet.
    - let's say every 0.5s for now
- Whether stable-baselines3 (or another library) should be reconsidered
  later - see section 3.1's explicit note that the custom-loop choice is
  revisitable.
- Position/role specialisation (separate weights per position) - explicitly
  deferred, "for now" shared weights across all players regardless of
  role.
- Higher-order strategies (Hold Possession/Defend/Conserve Energy as coach
  inputs with different loss functions) - explicitly flagged by the user as
  speculative/future, not part of the current design scope.
- Intelligent weight initialisation (e.g. a skip-connection from
  ball-direction/distance/spin/velocity straight to the move-direction
  output, initialised so "move toward the ball" is a good prior from the
  start) - raised as an idea by the user, not yet decided whether/how to
  implement.

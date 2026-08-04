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

## 6. Repository / module layout for the AI code

New top-level package, parallel to `src/footballcoach/engine/`,
`src/footballcoach/ui/` etc. Nothing here touches the existing engine
package structure - the AI code is purely a *consumer* of `Match`,
`orders`, `actions`, `entities`, `config`, exactly like `ui/` is today.

```
src/footballcoach/
  ai/
    __init__.py
    config/
      ai_config.json          # network sizes, PPO hyperparams, decision
                               # interval, reward coefficients - mirrors the
                               # engine's physics.json/attributes.json pattern
    obs/
      __init__.py
      schema.py                # dataclasses describing the observation
                                # layout (self, other-players, ball, global)
      encoder.py                # Match -> ObservationBatch (numpy/torch),
                                # relative encoding, masking, padding to 21
    action/
      __init__.py
      schema.py                 # dataclasses describing raw network output
                                # (decision heads + execution heads)
      distributions.py          # MaskedCategorical, IndependentBernoulli,
                                # combined ActionDistribution
      gating.py                 # select_action(): the non-differentiable
                                # winner-take-all + threshold rule (2.6)
      to_orders.py               # ActionOutput -> orders.py Order objects,
                                # incl. illegal-action detection/penalty flags
    models/
      __init__.py
      entity_encoder.py          # shared per-entity MLP + attention block
      decision_network.py        # trunk + all decision heads
      execution_network.py       # trunk + all execution heads
      value_network.py           # critic (shared trunk optional, see 8.2)
    ppo/
      __init__.py
      rollout_buffer.py           # storage + GAE(lambda) computation
      ppo_trainer.py               # the training loop itself (see section 8)
      schedules.py                 # LR / clip-range / rng_reduction schedules
    env/
      __init__.py
      match_env.py                  # Gym-like wrapper around Match + a
                                     # scenario definition; step()/reset()
      scenario_env.py                # adapts ui/scenarios.py ScenarioDefinition
                                     # + ScenarioLoop into a trainable env
      reward.py                      # per-head reward shaping functions
                                     # (section 9)
    curriculum/
      __init__.py
      phases.py                     # curriculum phase definitions (section 4)
      opponent_pool.py               # rules-based / frozen-checkpoint
                                     # opponent sampling (self-play pool)
    scripts/
      train.py                       # CLI entry point: `uv run python -m
                                     # footballcoach.ai.scripts.train --phase 1`
      evaluate.py                    # run N trials headless, report stats
                                     # (mirrors tests/balance/'s reporting style)
tests/
  ai_unit/                            # fast, deterministic tests for obs
                                     # encoding, masking, distributions, GAE
  ai_scenario/                        # short end-to-end smoke tests: a few
                                     # PPO updates on a toy env don't crash
                                     # and the loss actually changes
```

New dependency to add to `pyproject.toml`: `torch` (CPU is fine to start;
GPU only matters once training volume gets large - this project's episodes
are cheap/short single-player physics steps, not image renders, so CPU
throughput should be adequate for the MVP experiments). Suggest adding it
as a `dev`/optional group first (`[dependency-groups] ai = ["torch>=2.x"]`)
so the base game install stays lightweight for anyone who just wants to
play, and `uv sync --group ai` pulls in the training stack.

## 7. Observation schema (concrete)

### 7.1 Constants

```python
# ai/config: mirrors physics.json's approach - tunable, not hardcoded, but
# shown here with concrete defaults for clarity.
MAX_OTHER_PLAYERS = 21          # full 11v11 minus self
SELF_FEATURE_DIM = ...          # see below, computed from the fields
OTHER_PLAYER_FEATURE_DIM = ...  # per-entity feature vector length
GLOBAL_FEATURE_DIM = ...
DECISION_INTERVAL_S = 0.5        # networks run every 0.5s of sim time, not
                                  # every 1/30s engine tick (~15 ticks/decision)
```

`DECISION_INTERVAL_S = 0.5` is the user's confirmed starting value for how
often the decision+execution networks actually run (open item from section
6.1 resolved). Between decisions, the player continues executing the
*last* chosen order/action via the existing engine order state machine
(e.g. a `MoveOrder` or `ShootOrder` set by `to_orders.py`) exactly the same
way a human-issued UI order persists tick-to-tick today - no special engine
change is needed for this, since `orders.py` was already designed for
"assign an order, it persists until complete/replaced." The execution
network's *very* low-level outputs (move direction this instant, kick
this instant) are the exception - see 7.4/9 for how these interact with
sub-decision-interval ticks.

### 7.2 Per-player feature vector (used for both "self" and each "other" slot)

```python
@dataclass
class PlayerFeatures:
    # Relative to observing player, normalized by pitch half-length/width
    # (see 7.3 for the exact normalization convention). For the *self* slot
    # this is (0, 0) - not omitted, so the self/other feature vectors have
    # identical shape and the entity encoder's shared MLP can (optionally)
    # also be reused for the self embedding if desired.
    rel_dx: float
    rel_dy: float
    distance_m: float            # redundant with rel_dx/rel_dy, aids learning

    velocity_x: float            # world-frame, normalized by that player's
    velocity_y: float             # own effective_top_speed (not a global
                                  # constant) so the scale is meaningful
                                  # regardless of attribute-driven top speed
    speed_mps: float              # magnitude, redundant with vx/vy

    heading_sin: float             # sin/cos of heading_rad - avoids the
    heading_cos: float             # angle-wraparound discontinuity (2.2)

    stamina: float                  # 0-1, already normalized

    # Attributes (all already 0-1 in PlayerAttributes)
    top_speed: float
    acceleration: float
    kick_power: float
    kick_precision: float
    dribbling: float
    ball_control: float
    tackling: float
    stamina_attr: float             # the attribute, distinct from current stamina

    # Flags (all 0/1 floats, not bools, for direct tensor packing)
    is_own_team: float
    is_self: float                    # 1.0 only for the observing player's own slot
    has_possession: float
    is_inactive_tackled: float
    is_controlling_ball: float
    is_goalkeeper: float
    attacking_direction: float           # +1.0 if attacking +x, -1.0 if attacking -x
                                        # (Team.LEFT / Team.RIGHT convention,
                                        # engine/offside.py) - present on
                                        # every player slot (each player's own
                                        # attacking direction), not just self,
                                        # since a marker needs to know their
                                        # mark target's attacking direction too

    exists: float                       # 1.0 for a real player in this slot,
                                        # 0.0 for a padded/absent slot - see 7.5
```

### 7.3 Normalization convention

- Positions: expressed relative to the *observing* player
  (`rel_dx = other.x - self.x`, `rel_dy = other.y - self.y`), then divided
  by `pitch.length_m / 2` and `pitch.width_m / 2` respectively, so values
  are roughly in [-1, 1] even as pitch dimensions vary across randomised
  training scenarios (section 4's pitch-size randomisation). `distance_m`
  is normalized by `pitch half-diagonal` for the same reason.
- Velocities: normalized per-player by that player's own
  `effective_top_speed` (from `engine/movement.py`) rather than a single
  global max speed constant, so "0.5" always means "roughly half that
  player's personal top speed" regardless of their `top_speed` attribute -
  this keeps the feature meaningful and roughly attribute-invariant, which
  should help shared-weight training generalise across the full attribute
  range (0.2-0.9ish per `generation/knowledge.md`'s tiers).
- Ball spin: normalized by a configured max plausible spin (from
  `physics.json`'s Magnus-effect constants) similarly.

### 7.4 Ball feature vector

```python
@dataclass
class BallFeatures:
    rel_dx: float
    rel_dy: float
    distance_m: float
    height_m: float              # NOT normalized the same way as x/y - use
                                  # a fixed divisor (e.g. 3.0m) since height
                                  # doesn't scale with pitch size
    velocity_x: float
    velocity_y: float
    velocity_z: float
    spin_x: float
    spin_y: float
    spin_z: float
    is_possessed: float           # 0/1
    is_loose: float                # 1 - is_possessed, redundant but explicit
```

### 7.5 Global / match-context feature vector

```python
@dataclass
class GlobalFeatures:
    score_diff: float             # (own_team_goals - opp_team_goals), NOT
                                  # raw scores, so it's meaningful regardless
                                  # of which team is "left"/"right" this match
    time_remaining_s: float        # normalized by a fixed max (e.g. 7200s
                                  # i.e. 120 min, matching curriculum phase 1's
                                  # random time-remaining spec) - clipped/
                                  # log-scaled optionally so the 1-20s
                                  # "urgent" scenarios are still distinguishable
                                  # after normalization (a plain linear /7200
                                  # squashes them all to ~0 - recommend
                                  # log1p or a separate "is_urgent" flag, see
                                  # note below)
    pitch_length_m: float          # raw, or normalized against the standard
    pitch_width_m: float            # 105m/68m - both are informative: the
                                  # network needs to know actual scale isn't
                                  # always standard
    goal_width_m: float
    goal_height_m: float
    box_length_m: float
    box_width_m: float
    ball_restitution_coefficient: float
    rng_reduction: float             # the network should know how noisy the
                                  # current game/training setting is
    attack_defence_smoothed: float    # this player's own EMA-smoothed
                                  # attack/defence value fed back in (2.7) -
                                  # technically a per-player, not global,
                                  # feature; placed in self features in the
                                  # actual tensor packing (listed here for
                                  # narrative completeness)
```

**Time-remaining normalization note:** a plain linear normalisation by a
large max (7200s) makes the curriculum's explicit 10%-of-the-time
"1-20 seconds left" scenarios nearly indistinguishable from "2 minutes
left" once squashed to a [0,1] range (both ~0.003 vs ~0.017). Recommend
either (a) `log1p(time_remaining_s) / log1p(max_time_s)`, which spreads out
the low end, or (b) an explicit additional `is_final_20s` binary flag
alongside the normalized continuous value. Flagging this now since it's an
easy thing to get subtly wrong and only notice much later when the "urgent
endgame" scenario type never actually trains differently from a normal one.

### 7.6 Padding / masking for the other-players block (concrete)

- Build a list of up to `MAX_OTHER_PLAYERS` (21) `PlayerFeatures` for
  whichever real players other than self exist this tick (could be as few
  as 1 in a 1v1 scenario).
- **Randomised slot assignment (added per user note):** real players are
  shuffled into a *randomly chosen* subset of the 21 slots each time an
  observation is built (not always slots 0..k-1) - this is important so the
  network learns genuine *permutation invariance / doesn't overfit to slot
  index* (e.g. "slot 0 is usually the ball carrier" would be a spurious
  correlation in a fixed-assignment scheme, especially harmful once scaling
  from 1v1 - where slot assignment barely varies - up to 11v11). Concretely:
  `slot_indices = random.sample(range(MAX_OTHER_PLAYERS), k=len(other_players))`,
  assign real players to those slots, zero-fill + `exists=0.0` on the rest.
- Remaining (21 - k) slots: every field zeroed, **except** `exists = 0.0`
  explicitly set (all other flags naturally read as 0 too, so a padded
  slot looks like "a same-team, non-GK, no-possession player standing
  exactly on top of the observer with zero velocity" if `exists` weren't
  there to disambiguate it - this is exactly why the mask bit is
  necessary, not just a nice-to-have).
- The **attention mask** applied inside the entity encoder (section 8) uses
  this same `exists` bit: padded slots' attention scores are set to `-inf`
  before the softmax over keys, identically to the pass/tackle/mark target
  masking described in section 2.4 - it is, in fact, the same masking
  *technique* applied in two different places (entity attention keys vs.
  action-head softmax targets), worth recognising as one recurring pattern
  rather than two separate mechanisms.

## 8. Network architecture (concrete modules)

### 8.1 Entity encoder + attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityEncoder(nn.Module):
    """Shared per-entity MLP + multi-head attention pooling, used for the
    up-to-21 other-player slots. Query = self embedding; keys/values = other
    players' embeddings. Padded slots are masked out of the attention
    softmax so they contribute exactly zero, regardless of their (zeroed)
    feature values.
    """

    def __init__(self, entity_feature_dim: int, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.per_entity_mlp = nn.Sequential(
            nn.Linear(entity_feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        # Reused for the self-embedding too (same shared weights) - self is
        # just another "entity" with rel_dx=rel_dy=0, is_self=1.
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(
        self,
        self_features: torch.Tensor,     # (batch, entity_feature_dim)
        other_features: torch.Tensor,    # (batch, MAX_OTHER_PLAYERS, entity_feature_dim)
        exists_mask: torch.Tensor,        # (batch, MAX_OTHER_PLAYERS), 1.0/0.0
    ) -> torch.Tensor:                    # returns (batch, embed_dim) context vector
        self_embed = self.per_entity_mlp(self_features).unsqueeze(1)   # (batch, 1, embed_dim)
        other_embed = self.per_entity_mlp(other_features)               # (batch, 21, embed_dim)

        # nn.MultiheadAttention expects a boolean "key_padding_mask" where
        # True means "ignore this key" - i.e. the INVERSE of exists_mask.
        key_padding_mask = exists_mask < 0.5   # (batch, 21), True = padded/absent

        context, _attn_weights = self.attention(
            query=self_embed, key=other_embed, value=other_embed,
            key_padding_mask=key_padding_mask,
        )
        return context.squeeze(1)   # (batch, embed_dim)
```

Note: `nn.MultiheadAttention`'s `key_padding_mask` already implements
exactly the "-inf before softmax" masking rule described conceptually in
section 2.4/7.6 - this is the built-in PyTorch mechanism for it, no need to
hand-roll the masked-softmax for this particular use (hand-rolling is,
however, still needed for the pass/tackle/mark *action* target heads,
since those aren't attention layers - see 8.3).

**Edge case: zero other players.** Even a 1v1 scenario has exactly 1 other
player (the opponent), so `other_embed`/`exists_mask` are never *entirely*
masked in the current curriculum. If a future scenario ever had literally
zero other players (e.g. a solo dribbling drill), `nn.MultiheadAttention`
with an all-True `key_padding_mask` for one batch row can produce NaNs
(softmax over an empty/all -inf set is undefined) - guard for this
explicitly (e.g. skip attention and use a learned "no-opponents" embedding
constant) if such a scenario is ever added.

### 8.2 Shared trunk + decision heads

```python
class DecisionNetwork(nn.Module):
    def __init__(self, self_dim, ball_dim, global_dim, entity_embed_dim=64,
                 trunk_hidden=256, latent_dim=32):
        super().__init__()
        self.entity_encoder = EntityEncoder(entity_feature_dim=self_dim)  # same feature schema as self/other
        self.self_mlp = nn.Sequential(nn.Linear(self_dim, 64), nn.ReLU())
        self.ball_mlp = nn.Sequential(nn.Linear(ball_dim, 32), nn.ReLU())
        self.global_mlp = nn.Sequential(nn.Linear(global_dim, 32), nn.ReLU())

        trunk_input_dim = 64 + 64 + 32 + 32  # entity_context + self + ball + global
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, trunk_hidden), nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(),
        )

        # --- Heads ---
        # Independent Bernoulli action-probability heads (raw logits; apply
        # sigmoid only where a probability is actually needed, e.g. for
        # gating/logging - keep raw logits for the Bernoulli distribution's
        # log_prob, which is numerically more stable from logits directly).
        self.shoot_logit = nn.Linear(trunk_hidden, 1)
        self.pass_logit = nn.Linear(trunk_hidden, 1)
        self.move_logit = nn.Linear(trunk_hidden, 1)
        self.tackle_logit = nn.Linear(trunk_hidden, 1)
        self.get_possession_raw = nn.Linear(trunk_hidden, 1)   # see 8.2.1 re: constraint vs tackle
        self.mark_logit = nn.Linear(trunk_hidden, 1)
        self.hold_position_logit = nn.Linear(trunk_hidden, 1)

        # Categorical (masked) target heads - one logit per possible slot.
        self.pass_target_logits = nn.Linear(trunk_hidden, MAX_OTHER_PLAYERS)
        self.tackle_target_logits = nn.Linear(trunk_hidden, MAX_OTHER_PLAYERS)
        self.mark_target_logits = nn.Linear(trunk_hidden, MAX_OTHER_PLAYERS)

        # Continuous heads - output raw values; squash/scale to physical
        # ranges (metres, m/s) in the action-decoding step (to_orders.py),
        # NOT inside the network, so the network's own numeric output stays
        # well-conditioned (roughly unit-scale) for the Normal distribution.
        self.move_region_center = nn.Linear(trunk_hidden, 2)     # (x, y), tanh-squashed later
        self.move_region_size = nn.Linear(trunk_hidden, 1)        # scalar, sigmoid+scale to 1-4m
        self.move_arrival_speed = nn.Linear(trunk_hidden, 1)       # sigmoid+scale to 0-top_speed
        self.region_of_play_center = nn.Linear(trunk_hidden, 2)
        self.region_of_play_size = nn.Linear(trunk_hidden, 1)       # sigmoid+scale to 15-40m
        self.attack_defence_raw = nn.Linear(trunk_hidden, 1)         # sigmoid -> 0-1

        self.latent_vector = nn.Linear(trunk_hidden, latent_dim)     # unbounded, passed to execution net

        # Per PPO's usual actor-critic setup, decide whether the value
        # function shares the trunk (faster, more sample-efficient at small
        # scale, but couples actor/critic gradients through a shared body -
        # see 9.4 for the trade-off and recommendation) or has its own
        # separate trunk (safer/decoupled, more parameters).
        self.value_head = nn.Linear(trunk_hidden, 1)   # shared-trunk variant

    def forward(self, self_feat, other_feat, exists_mask, ball_feat, global_feat):
        entity_ctx = self.entity_encoder(self_feat, other_feat, exists_mask)
        h = torch.cat([
            entity_ctx,
            self.self_mlp(self_feat),
            self.ball_mlp(ball_feat),
            self.global_mlp(global_feat),
        ], dim=-1)
        h = self.trunk(h)
        return DecisionHeadsRaw(
            shoot_logit=self.shoot_logit(h),
            pass_logit=self.pass_logit(h),
            move_logit=self.move_logit(h),
            tackle_logit=self.tackle_logit(h),
            get_possession_raw=self.get_possession_raw(h),
            mark_logit=self.mark_logit(h),
            hold_position_logit=self.hold_position_logit(h),
            pass_target_logits=self.pass_target_logits(h),
            tackle_target_logits=self.tackle_target_logits(h),
            mark_target_logits=self.mark_target_logits(h),
            move_region_center=self.move_region_center(h),
            move_region_size=self.move_region_size(h),
            move_arrival_speed=self.move_arrival_speed(h),
            region_of_play_center=self.region_of_play_center(h),
            region_of_play_size=self.region_of_play_size(h),
            attack_defence_raw=self.attack_defence_raw(h),
            latent_vector=self.latent_vector(h),
            value=self.value_head(h),
        )
```

#### 8.2.1 Enforcing "get-possession >= tackle" constraint

The spec requires `get_possession_probability >= tackle_probability` always
(get-possession can exceed tackle, e.g. chasing a loose ball, but never be
lower). Rather than outputting `get_possession_probability` directly as an
independent sigmoid (which the network could easily violate), derive it
compositionally so the constraint is structurally guaranteed:

```python
tackle_prob = torch.sigmoid(heads.tackle_logit)
extra_prob = torch.sigmoid(heads.get_possession_raw)   # network's own "how much MORE"
# get_possession is tackle_prob plus some fraction of the remaining headroom
# up to 1.0 - guarantees get_possession_prob in [tackle_prob, 1.0] always.
get_possession_prob = tackle_prob + extra_prob * (1.0 - tackle_prob)
```

This is a standard "residual/headroom" parameterization - it keeps the
constraint exact (not just encouraged via a loss penalty) at zero extra
engineering cost, and both `tackle_logit` and `get_possession_raw` remain
ordinary independent parameters the network can freely learn, with
`get_possession_raw=0 -> get_possession_prob == tackle_prob` (the "always
at least as often as tackle" floor) and `get_possession_raw=1 ->
get_possession_prob == 1.0` (always attempt to get the ball this tick,
maximal). Log-prob for PPO purposes should be computed on `tackle_logit`
and `get_possession_raw` as two independent Bernoulli parameters directly
(their own raw sigmoids), *not* on the derived `get_possession_prob` value,
since the latter isn't a simple Bernoulli parameter of an independent
action - it's cleaner to think of the actual PPO action space as
`(tackle_intent, get_possession_extra)` and only compose them into the
"effective get-possession probability" at the gating/execution stage.

### 8.3 Masked categorical distribution for target heads

```python
class MaskedCategorical:
    """Categorical distribution over MAX_OTHER_PLAYERS slots, masking out
    non-existent players via -inf before softmax (see 2.4/7.6)."""

    def __init__(self, logits: torch.Tensor, exists_mask: torch.Tensor):
        # exists_mask: (batch, MAX_OTHER_PLAYERS), 1.0 = real player.
        masked_logits = logits.masked_fill(exists_mask < 0.5, float("-inf"))
        self.dist = torch.distributions.Categorical(logits=masked_logits)

    def sample(self) -> torch.Tensor:
        return self.dist.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()

    def mode(self) -> torch.Tensor:
        return self.dist.probs.argmax(dim=-1)
```

`torch.distributions.Categorical` already normalises `exp(-inf)` to exactly
0 probability internally and its `log_prob`/`entropy` handle this
correctly (no NaN) as long as *at least one* slot per row is unmasked -
same "all slots masked" edge case caveat as section 8.1's attention note
applies here too (a scenario with literally no valid pass/tackle targets at
all should never call `.sample()`/`.log_prob()` on this distribution for
that row - guard at the call site, e.g. skip the pass-target loss term
entirely for players with zero teammates).

### 8.4 Independent Bernoulli heads

```python
class IndependentBernoulli:
    """Thin wrapper so all the ~7-9 sigmoid action-probability heads share
    one consistent log_prob/entropy/sample interface, built directly from
    logits (numerically stabler than sigmoid-then-Bernoulli-from-probs)."""

    def __init__(self, logits: torch.Tensor):
        self.dist = torch.distributions.Bernoulli(logits=logits)

    def sample(self) -> torch.Tensor:
        return self.dist.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()

    def prob(self) -> torch.Tensor:
        return torch.sigmoid(self.dist.logits)
```

### 8.5 Continuous heads (move target, arrival speed, region, attack/defence)

Two standard options for continuous PPO action heads - **recommend Option
B (Normal with learned/state-dependent std)** for this project:

**Option A - Beta distribution**, naturally bounded to [0,1], good when the
output truly has hard physical bounds (e.g. `attack_defence` and
`move_region_size` after sigmoid do have hard bounds) but is slightly more
fiddly to parameterize stably (needs `alpha, beta > 0`, typically via
`softplus(raw) + 1`) and has less precedent/tooling than Gaussian PPO heads.

**Option B - Gaussian (Normal) with `tanh`/`sigmoid` squashing applied
*after* sampling**, the standard approach used by most continuous-control
PPO implementations (e.g. MuJoCo benchmarks): network outputs an unbounded
mean and a (state-independent or state-dependent) log-std; sample from
`Normal(mean, std)`; squash the *sample* through `tanh` (mapped to [-1,1])
or `sigmoid` (mapped to [0,1]) before rescaling into the physical range
(metres, m/s). This requires a log-prob correction for the squashing
Jacobian if being fully rigorous (as in SAC's original paper), but PPO
implementations very commonly skip this correction in practice (treating
the *pre-squash* Gaussian sample as the "action" for log_prob purposes,
and only squashing for the actual physical/engine-facing value) - simpler,
and works fine in practice for this kind of application; flag as a known
minor approximation rather than a bug if adopted.

```python
class SquashedNormalHead:
    """mean/log_std parameterization; produces both an unconstrained action
    for PPO log_prob purposes and the squashed physical-range value for
    actually building an order."""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor,
                 low: float, high: float):
        self.low, self.high = low, high
        std = torch.exp(log_std.clamp(-5, 2))   # clamp for numerical stability
        self.dist = torch.distributions.Normal(mean, std)

    def sample_raw(self) -> torch.Tensor:
        return self.dist.sample()

    def log_prob(self, raw_action: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(raw_action).sum(dim=-1)

    def to_physical(self, raw_action: torch.Tensor) -> torch.Tensor:
        squashed = torch.sigmoid(raw_action)   # or tanh, remapped, per head
        return self.low + squashed * (self.high - self.low)

    def mode_physical(self) -> torch.Tensor:
        return self.to_physical(self.dist.mean)
```

Recommended (mean, low, high) per continuous head:
- `move_region_center`: unconstrained mean, squashed via `tanh` to
  `[-1, 1]` in normalized rel-position space, rescaled to actual
  metres/pitch-relative offset in `to_orders.py`.
- `move_region_size`: `sigmoid`-squashed to `[1.0, 4.0]` metres.
- `move_arrival_speed`: `sigmoid`-squashed to `[0, effective_top_speed]`
  (a per-player value, computed in `to_orders.py`, not inside the network).
- `region_of_play_size`: `sigmoid`-squashed to `[15.0, 40.0]` metres.
- `attack_defence_raw`: `sigmoid`-squashed to `[0, 1]` directly (this is
  the *instantaneous* target fed into the EMA smoother of section 2.7, not
  the smoothed value itself).

**State-dependent vs. global log_std:** simplest starting point is a single
global `nn.Parameter` per head (not computed from the trunk), which is the
most common/stable starting configuration in PPO continuous-control
literature (e.g. CleanRL's continuous PPO) - start there; only move to a
state-dependent std (an extra linear layer per head) if entropy collapse or
insufficient exploration is observed empirically.

### 8.6 Execution network

Structurally similar shape (entity encoder + trunk), but its input also
concatenates the *entire* decision-network output (all heads, both the
"selected" ones and the ones gated off this tick - per section 2.5's
explicit requirement), and its outputs map directly onto the low-level
motor action surface:

```python
class ExecutionNetwork(nn.Module):
    def __init__(self, self_dim, ball_dim, global_dim, decision_output_dim,
                 entity_embed_dim=64, trunk_hidden=256):
        super().__init__()
        self.entity_encoder = EntityEncoder(entity_feature_dim=self_dim)
        self.self_mlp = nn.Sequential(nn.Linear(self_dim, 64), nn.ReLU())
        self.ball_mlp = nn.Sequential(nn.Linear(ball_dim, 32), nn.ReLU())
        self.global_mlp = nn.Sequential(nn.Linear(global_dim, 32), nn.ReLU())
        self.decision_mlp = nn.Sequential(nn.Linear(decision_output_dim, 64), nn.ReLU())

        trunk_input_dim = 64 + 64 + 32 + 32 + 64
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, trunk_hidden), nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(),
        )

        self.move_direction = nn.Linear(trunk_hidden, 2)   # sin/cos-style unit vector, see below
        self.sprint_logit = nn.Linear(trunk_hidden, 1)       # Bernoulli: sprint vs jog
        self.kick_logit = nn.Linear(trunk_hidden, 1)          # Bernoulli: kick this instant?
        self.kick_direction = nn.Linear(trunk_hidden, 2)
        self.kick_power = nn.Linear(trunk_hidden, 1)           # sigmoid -> 0-1 power_fraction
        self.kick_spin = nn.Linear(trunk_hidden, 3)
        self.tackle_attempt_logit = nn.Linear(trunk_hidden, 1)  # Bernoulli
```

`move_direction`/`kick_direction`: the linear layer outputs a raw 2-vector
which is **L2-normalized to a unit vector inside `forward()`** (`v / (||v||
+ eps)`) before being stored in `ExecutionHeadsRaw`. This is preferable to
outputting an angle (avoids wraparound discontinuity, differentiable
everywhere) and, crucially, **constrains the distribution mean to the unit
circle**, bounding max mean-shift between any two policy versions to 2.
Consequently the KL contribution of these heads is O(1) per step rather
than the ~2000 that occurred when the raw vector could drift to magnitude
25–50 — so direction heads are **included in the PPO log_prob ratio** like
all other heads, with no special treatment needed.

For PPO purposes: the unit-normalized mean is the parameter of an isotropic
2D Gaussian (`Normal(mean, std)` per component, independent). The *noisy
sample* drawn from this Gaussian is stored as `move_dir_raw`/`kick_dir_raw`
in the rollout buffer and L2-normalized again to obtain the physical unit
vector used by the engine. Log_prob is computed on the stored raw sample
under the current mean, exactly as for `SquashedNormalHead` (section 8.5).

## 9. PPO training loop - detailed, with explicit options

### 9.1 What "one trajectory step" means here

Given `DECISION_INTERVAL_S = 0.5s` and the engine's `dt = 1/30s`, one
"environment step" for PPO purposes = **one decision interval**, not one
engine tick: `MatchEnv.step(action)` internally (a) converts the sampled
action into orders via `to_orders.py`, (b) assigns those orders to the
player(s) being trained, (c) calls `Match.step()` in a loop ~15 times
(0.5s / (1/30s)) advancing the underlying simulation, (d) computes the
reward accumulated over those 15 ticks (section 9's reward section, 10
below), (e) builds the next observation. This means a PPO "episode" (e.g.
one curriculum-phase-1 trial capped at 2 minutes) is ~240 decision steps
long (120s / 0.5s) - short enough that rollout buffers stay small and
GAE/return computation is cheap, which is convenient for fast iteration on
the MVP experiments.

Other players in the scenario not currently being trained (e.g. a
rules-based opponent, or a frozen older-generation checkpoint per the
curriculum's self-play pool) are simply driven by whatever those
mechanisms already are - the `orders`/`actions` layer for rules-based ones,
or a separate frozen policy's forward pass (no gradient) for older-AI
opponents - `MatchEnv` just needs to re-issue their orders/actions each
decision interval (or every tick, if that opponent AI already works
per-tick, e.g. `SaveOrder`/`MarkOrder`'s persistent-duty design already
means "issue once, it stays in effect" so it doesn't need re-issuing at
all).

### 9.2 Reusing the UI scenarios for training data (explicit note, per user request)

**`src/footballcoach/ui/scenarios.py` already contains built, tested
scenario constructors (`build_penalty_scenario`, `build_tackle_scenario`,
the 1v1/1v2/2v2 builders, etc.) and `ScenarioLoop`, which already knows how
to run a scenario to completion, detect the outcome (goal/dispossessed/
timeout/ball-out), and rebuild a fresh randomised trial. This is valuable,
already-correct training-data infrastructure and should be reused rather
than reimplemented:**

- `env/scenario_env.py` should **wrap `ScenarioDefinition` + `ScenarioLoop`
  directly**, rather than building a parallel/duplicate scenario-generation
  path. Concretely: `ScenarioEnv.__init__(scenario: ScenarioDefinition,
  trainee_player_id: str, **scenario_kwargs)` holds a `ScenarioLoop`
  internally; `ScenarioEnv.reset()` calls (or waits for) the loop's
  automatic match-rebuild-on-trial-end and returns the first observation;
  `ScenarioEnv.step(action)` assigns the action's orders to the trainee
  player, calls `ScenarioLoop.step()` in the required sub-loop (per 9.1),
  and translates the loop's outcome (goal / dispossessed / timeout /
  ball-out - see `ui/scenarios.py`'s `ScenarioLoop` for the exact outcome
  enum) into a terminal reward + `done` flag.
- This reuse is exactly why `ScenarioDefinition.build` functions already
  taking `rng_reduction` and scenario-specific kwargs (separation
  distance, attribute ranges, etc.) matters for training: the *same*
  randomisation knobs already exposed to the UI's scenario-picker screen
  (`ScenarioParam` list) double as the curriculum's difficulty/domain-
  randomisation knobs during training, with zero duplicate code - e.g.
  phase 1's "vary separation, vary attributes" requirement is already
  exactly what `build_tackle_scenario`'s `separation_min_m` /
  `separation_max_m` / `tackler_tackling_min/max` /
  `dribbler_dribbling_min/max` kwargs provide.
- Where a curriculum phase needs a scenario shape that doesn't exist yet in
  `ui/scenarios.py` (e.g. a dedicated "1 attacker + keeper, empty vs
  static-defender vs moving-defender" progression for phase 2's shooting
  curriculum), **add it to `ui/scenarios.py` as a new `ScenarioDefinition`
  builder** (following the existing file's conventions/docstring style)
  rather than creating a separate training-only scenario module - this
  keeps a single source of truth for scenario construction that's usable
  both from the UI (for a human to visually spot-check what the AI is
  training against) and from the training loop, and keeps the balance-test
  suite's existing pattern (`tests/scenario/test_scenario_loop.py` already
  exercises every entry in `SCENARIOS` generically) automatically covering
  any new scenario added this way too.
- The rules-based `orders`/`actions` "AI logic" already embedded in these
  scenario builders (e.g. `build_tackle_scenario`'s use of
  `ChaseTackleOrder`/`GetPossessionOrder`, `SaveOrder` for keepers) is
  precisely the source of the "positive examples / scripted adversary or
  ally" the curriculum (section 4) repeatedly calls for - no separate
  scripted-opponent implementation is needed for the MVP experiments; it
  already exists and is already tested.

### 9.3 Rollout buffer + GAE(lambda)

```python
@dataclass
class RolloutBuffer:
    observations: list   # ObservationBatch per step
    actions: list         # ActionOutput per step (raw, pre-gating - the
                          # thing log_prob was computed against)
    log_probs: list        # summed log_prob across ALL heads for this step
    values: list             # critic's value estimate at this step
    rewards: list             # scalar reward this step (see 9.5)
    dones: list                # 1.0 if this step ended the episode

    def compute_gae(self, gamma: float, lam: float, last_value: float):
        advantages = [0.0] * len(self.rewards)
        last_gae = 0.0
        for t in reversed(range(len(self.rewards))):
            next_value = self.values[t + 1] if t + 1 < len(self.values) else last_value
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = [a + v for a, v in zip(advantages, self.values)]
        return advantages, returns
```

Standard GAE(lambda), `gamma≈0.99`, `lam≈0.95` as sane starting defaults
(these plus all other hyperparameters belong in `ai_config.json`, not
hardcoded - per section 6's `ai_config.json`, and consistent with the
open-items note that exact hyperparameters are not yet decided/tuned).
**Common silent bug to watch for** (flagged per section 3.1's warning that
GAE bugs are silent): off-by-one on `next_value`/`dones` alignment - the
value used for `delta` at step `t` must be the value estimate *after*
transitioning (i.e. for the state the episode is *in* at `t+1`, not `t`),
and `next_non_terminal` must zero out bootstrapping across an episode
boundary. Writing a small unit test with a toy 3-step deterministic reward
sequence and hand-computed expected advantages (in `tests/ai_unit/`) before
trusting this against the real environment is strongly recommended.

### 9.4 Actor/critic sharing - explicit options

**Option A - Shared trunk (single network, two heads: policy heads + one
value head off the same trunk features)**, as sketched in section 8.2's
`DecisionNetwork.value_head`. Pros: fewer total parameters, faster to
train per environment step (one forward pass computes both), and at small
network scale (this project's MVP) is usually stable. Cons: the value
loss's gradient and the policy loss's gradient both flow through the same
trunk, so a large-magnitude value loss early in training (value function
starts essentially random) can destabilise the policy's shared features
before the value function calibrates - usually mitigated with a `vf_coef`
weight (e.g. 0.5) down-weighting the value loss relative to the policy
loss in the combined objective (see 9.6), and/or a warmup period.

**Option B - Fully separate networks** (separate trunk for policy vs.
value, doubling parameter count and forward-pass cost, no gradient
interference). Cleaner in theory, standard in some PPO implementations
(e.g. many robotics continuous-control setups use separate networks), at
the cost of more compute and more total parameters to tune.

**Recommendation for this project: start with Option A (shared trunk)** -
given the modest network sizes needed for the MVP experiments (1v1/1v2
scenarios, not full 11v11 matches yet) and the value of fast iteration,
shared-trunk is the more common default in PPO reference implementations
for small-to-medium tasks (e.g. OpenAI's original PPO paper and most
CleanRL PPO variants default to a shared trunk for discrete-action tasks,
though CleanRL's continuous-control variant actually uses fully separate
networks). If value loss appears to destabilise policy training empirically
(watch for policy entropy collapsing or KL divergence spiking right when
value loss is large), switch to Option B - this is a cheap architectural
change to make later (the value head is already factored out as its own
`nn.Linear` in the sketch above) and doesn't require redesigning the rest
of the system.

Note this decision applies **independently to each of the two networks**
(decision network's own actor/critic split, execution network's own
actor/critic split) - it would also be reasonable to have the execution
network's value function estimate be a *different* value function from the
decision network's (e.g. different reward horizons/weightings per section
9.5's per-head rewards), or to share a single combined value estimate
across both - this is an open design choice, default recommendation:
**one shared value estimate for the whole per-player action this tick**
(decision + execution combined), since ultimately both networks are being
optimised toward the same overall episode return, just at different levels
of abstraction; revisit if the two networks need genuinely different
training paces (e.g. execution-network-only phases of the curriculum,
section 4).

### 9.5 Per-head log_prob combination

For a single environment step, log_prob is a **sum across every head that
was "active"/relevant this step** - not every head is included every step
(e.g. `pass_target_logits`'s log_prob only makes sense/should only be
included if `pass_logit`'s Bernoulli sample was 1 this step, or during
early bootstrapped phases, if the rules-based target was actually a pass).
Concretely:

```python
def combined_log_prob(heads: DecisionHeadsRaw, action: DecisionAction, exists_mask) -> torch.Tensor:
    lp = 0.0
    lp = lp + IndependentBernoulli(heads.shoot_logit).log_prob(action.shoot)
    lp = lp + IndependentBernoulli(heads.pass_logit).log_prob(action.pass_)
    lp = lp + IndependentBernoulli(heads.move_logit).log_prob(action.move)
    lp = lp + IndependentBernoulli(heads.tackle_logit).log_prob(action.tackle)
    lp = lp + IndependentBernoulli(heads.get_possession_raw).log_prob(action.get_possession_extra)
    lp = lp + IndependentBernoulli(heads.mark_logit).log_prob(action.mark)
    lp = lp + IndependentBernoulli(heads.hold_position_logit).log_prob(action.hold_position)

    # Target heads: only meaningfully sampled/relevant if the corresponding
    # intent action was 1 this step (masked contribution otherwise - use
    # action.pass_ as a 0/1 multiplier so an irrelevant target doesn't
    # contribute gradient/log_prob noise on steps where it wasn't used).
    lp = lp + action.pass_ * MaskedCategorical(heads.pass_target_logits, exists_mask).log_prob(action.pass_target)
    lp = lp + action.tackle * MaskedCategorical(heads.tackle_target_logits, exists_mask).log_prob(action.tackle_target)
    lp = lp + action.mark * MaskedCategorical(heads.mark_target_logits, exists_mask).log_prob(action.mark_target)

    # Continuous heads similarly gated by relevance where applicable
    # (move_region/arrival_speed only meaningfully "used" if move or
    # hold_position fired - but simplest starting point: just always
    # include them un-gated, since they're cheap/always well-defined
    # continuous outputs, unlike the categorical targets which need an
    # actual target player to exist. Recommend starting ungated for
    # continuous heads and revisiting if this causes gradient noise.)
    ...
    return lp
```

This per-head-sum approach treats the whole multi-head action as one
factorized joint distribution (`log P(all heads) = sum(log P(each
head))`), which is the standard, correct way to combine independent action
components for PPO's ratio calculation - `π_new(a|s)/π_old(a|s)` becomes
`exp(sum(new_log_probs) - sum(old_log_probs))`, exactly analogous to how
`MultiDiscrete` action spaces are already handled in most RL libraries,
just generalised to a mix of Bernoulli/Categorical/Normal components.

### 9.6 Clipped surrogate objective + full loss

```python
def ppo_loss(new_log_probs, old_log_probs, advantages, values, returns,
             entropy, clip_range=0.2, vf_coef=0.5, ent_coef=0.01):
    ratio = torch.exp(new_log_probs - old_log_probs)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalize

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(values, returns)   # consider PPO's optional value clipping too

    entropy_loss = -entropy.mean()   # encourage exploration

    total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
    return total_loss, dict(policy_loss=policy_loss.item(), value_loss=value_loss.item(),
                              entropy=entropy.mean().item(), approx_kl=(old_log_probs - new_log_probs).mean().item())
```

Standard multi-epoch PPO update: for each rollout collection (e.g. a batch
of full episodes/trials), run `K` epochs (e.g. 4-10) over the buffer in
shuffled minibatches, recomputing `new_log_probs`/`values`/`entropy` each
epoch (since the policy has changed), while `old_log_probs`/`advantages`/
`returns` are fixed from the original rollout. Track `approx_kl` and
early-stop epochs for a batch if it exceeds a threshold (e.g. 0.02) -
standard PPO safety valve against a batch of stale advantages pushing the
policy too far in one update.

### 9.7 Illegal-action handling (developer note: engine should guard, confirmed)

Per the user's explicit confirmation, **the engine should guard against
illegal actions being physically possible** (e.g. `KickOrder`/`ShootOrder`
already appear to require possession implicitly via `kick_ball`'s
mechanics reading `ball.possessed_by` - this needs auditing/confirming
during implementation, not yet verified as of this document) - i.e. an
illegal action attempt should be a safe no-op at the engine level, not
something that could corrupt match state, **and** the AI should be
separately punished for attempting it via the reward function (a negative
reward term applied whenever `to_orders.py` detects the action-to-order
translation was attempted against an illegal precondition, e.g. shooting
without possession, tackling out of range) - both protections should exist
simultaneously (engine safety net + reward-shaping deterrent), not one
instead of the other.

## 10. Reward shaping (concrete starting formulas - all coefficients tunable)

These are **starting points for the MVP experiments only** (per section 5),
to be tuned empirically exactly like the engine's balance constants -
expressed here as concrete formulas so implementation has something
concrete to start from, not just "reward getting the ball" prose.

### 10.1 Experiment 1 (GetPossession/Move)

Per-decision-step reward, accumulated over the ~15 engine ticks in that
interval:

```python
def phase1_reward(prev_ball_dist, curr_ball_dist, has_possession_now,
                   gained_possession_this_step, ball_progress_toward_goal_m,
                   ball_went_out_after_touch, illegal_action_attempted,
                   reached_opponent_box_with_possession, timed_out) -> float:
    r = 0.0
    r += 0.05 * (prev_ball_dist - curr_ball_dist)     # shaping: closing distance to ball
    if gained_possession_this_step:
        r += 1.0                                        # one-off bonus for winning the ball
    if has_possession_now:
        r += 0.1 * ball_progress_toward_goal_m           # progressing the ball upfield
    if ball_went_out_after_touch:
        r -= 1.0
    if illegal_action_attempted:
        r -= 0.2
    if reached_opponent_box_with_possession:
        r += 5.0   # terminal bonus, episode ends
    # timed_out: no terminal bonus/penalty, just ends
    return r
```

### 10.2 Experiment 2 (Shoot)

```python
def phase2_reward(shot_taken_this_step, ticks_since_episode_start,
                   shot_on_target, goal_scored, illegal_action_attempted,
                   possession_lost_to_keeper) -> float:
    r = 0.0
    if shot_taken_this_step:
        # Faster shots rewarded more - decaying bonus the longer the player
        # waited before shooting.
        r += 1.0 * max(0.0, 1.0 - ticks_since_episode_start / MAX_EPISODE_TICKS)
        if shot_on_target:
            r += 2.0
        if goal_scored:
            r += 10.0   # terminal
    if illegal_action_attempted:
        r -= 0.2
    if possession_lost_to_keeper:
        r -= 0.5   # mild - not the end of the world, but shouldn't be the norm
    return r
```

Additionally, per section 5, the **decision network's `shoot_logit` head
specifically** needs an auxiliary loss/reward term encouraging
`shoot_probability > 0.5` in scenarios where shooting was actually the
correct behaviour-bootstrapping target (rules-based comparison, exactly
like phase 1's Move/GetPossession bootstrapping) - i.e. during the initial
bootstrap sub-phase, add a supervised cross-entropy term between
`shoot_logit` and the rules-based "should have shot here" binary label,
alongside (not instead of) the PPO reward-driven loss, then anneal the
supervised term's weight to zero as training progresses into pure RL.

## 11. Illegal-action / guardrail engine audit checklist (to action during implementation)

Concrete checklist for auditing `engine/` before/while wiring up
`to_orders.py`, since the user confirmed illegal actions must be guarded:

- [ ] `KickOrder`/`ShootOrder` execution: confirm `Match._process_orders`
      already no-ops (rather than erroring or producing a nonsensical kick)
      if the ordering player does not currently have `ball.possessed_by ==
      player.player_id`. If not, add the guard.
- [ ] `TackleOrder`/`ChaseTackleOrder`/`GetPossessionOrder`: confirm
      attempting a tackle while `player.state == INACTIVE_TACKLED` is
      already prevented (per `is_available_to_tackle()` in
      `entities/player.py`) at every call site, not just some.
- [ ] `PassOrder`: same possession precondition as `KickOrder`.
- [ ] `SaveOrder`: confirm a non-goalkeeper issuing this order is a safe
      no-op rather than undefined behaviour (per docstring, "Goalkeeper-only" -
      verify this is actually enforced somewhere, not just documented).
- [ ] Whatever guard behaviour is found/added, `to_orders.py` must be able
      to *detect* "this action would have been illegal" independently
      (i.e. don't rely on silently-no-op engine behaviour alone - the
      reward function needs an explicit boolean to penalise), e.g. by
      checking the same precondition in Python before/alongside assigning
      the order, or by having the engine expose a small result/status object
      from `_process_orders` indicating whether each order was legal this
      tick.

## 12. Testing strategy for the AI code

Mirroring the existing engine's unit/scenario/balance test-tier convention:

- **`tests/ai_unit/`** (fast, deterministic, no torch training loop): 
  - Observation encoder: given a hand-built `Match` state, assert the
    produced tensors have correct shape, correct masking (`exists` bits,
    padded slots zeroed), correct normalization (e.g. a player exactly at
    the pitch boundary produces `rel_dx ≈ ±1.0`), and correct random slot
    shuffling (assert two calls with different RNG seeds place the same
    real player in different slots, but always produce identical *content*
    modulo slot permutation).
  - `MaskedCategorical`/`IndependentBernoulli`/`SquashedNormalHead`: verify
    `log_prob`/`entropy`/`sample` don't NaN, masked slots truly get zero
    probability, squashing stays within declared physical bounds.
  - GAE: small hand-computed toy sequence (section 9.3's note).
  - `select_action()` gating rule (section 2.6): given a hand-built set of
    head probabilities, assert the correct single winner is selected above
    50% threshold, and that below-50%-for-everything correctly selects
    "no action" (or whichever the design's convention is for that case -
    **open item, not yet decided: what should happen if EVERY head is below
    50%?** e.g. hold-position by default, or the previous tick's action
    persists, or a dedicated small "no-op movement" default - flag this
    explicitly as needing a decision before `gating.py` can be finalized).
- **`tests/ai_scenario/`**: short smoke tests that a handful of PPO update
  steps on a tiny toy environment (not necessarily the full football env -
  a synthetic environment with a known optimal policy, e.g. "reward is
  higher the closer a single continuous action value is to a fixed target"
  is a good sanity check independent of football-specific complexity) run
  without crashing and visibly reduce loss / increase reward over a few
  iterations - catches gross implementation bugs before spending compute
  on the real football scenarios.
- **`ai/scripts/evaluate.py`**: mirrors `tests/balance/`'s reporting
  convention (full stats, not just pass/fail) - run N trials of a given
  curriculum-phase scenario with the current policy checkpoint, report
  win-rate/goal-rate/average-time-to-shoot/etc., written to a results file
  the same way `tests/balance/results/latest_results.json` works today.

## 13. Open items / not yet decided

These are flagged so a future session doesn't assume they're settled:

- Exact reward function coefficients/shaping for each head - section 10
  now gives concrete starting formulas for the two MVP experiments, but
  these are explicitly starting points to tune empirically, not final
  values; reward shaping for the later curriculum phases (passing,
  tackling, marking, region-of-play, attack/defence-weighted rewards) is
  still only qualitative.
- Exact network sizes/hyperparameters (layer widths, attention head count,
  learning rate, clip range, GAE lambda/gamma, etc.) - section 8/9 gives
  concrete starting defaults (e.g. `trunk_hidden=256`, `embed_dim=64`,
  `gamma=0.99`, `lam=0.95`, `clip_range=0.2`), but these are all
  first-guess values to be tuned empirically once training actually runs,
  exposed via `ai/config/ai_config.json` matching the engine's
  `physics.json`/`attributes.json` pattern.
- **RESOLVED**: illegal-action guarding - confirmed by user, the engine
  must guard illegal actions (e.g. kicking without possession) as a safe
  no-op, *and* the reward function must separately penalise the attempt.
  See section 9.7 and the concrete audit checklist in section 11 for what
  still needs verifying/adding in `engine/`.
- **RESOLVED**: decision interval - confirmed by user as `0.5s` to start
  (section 7.1's `DECISION_INTERVAL_S`), i.e. the decision+execution
  networks run roughly every 15 engine ticks, not every tick.
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
- **New, from section 12's test-design discussion**: what should the
  `select_action()` gating rule (section 2.6) do when *every* action-
  probability head is below 50% this decision tick? Candidate options not
  yet chosen between: (a) default to Hold-Position, (b) let the previous
  decision's action persist unchanged, (c) a dedicated small "no-op/idle"
  behaviour distinct from Hold-Position's region-penalty semantics. Needs
  a decision before `action/gating.py` can be finalized.
- Actor/critic sharing (section 9.4): recommended default is a shared
  trunk per network (decision network's own actor/critic, execution
  network's own actor/critic, one combined value estimate per player per
  decision step) - flagged as easy to change later, not fully locked in.
- Squashed-Gaussian log_prob correction for continuous heads (section 8.5):
  deliberately *not* applying the tanh/sigmoid-Jacobian correction some
  algorithms (e.g. SAC) use, as is common practice for PPO - flagged as a
  known, deliberate approximation rather than an oversight, revisit only if
  continuous-head training behaves oddly.
- Whether execution-network-only phases of the curriculum need a
  *different* value function/reward horizon from the decision network -
  section 9.4 flags this as an open question with a default recommendation
  (share one value estimate) rather than a settled answer.

## 14. Reuse of `ui/scenarios.py` for training (summary pointer)

See section 9.2 for the full explanation - **short version for anyone
skimming this document**: do not build a separate/duplicate scenario or
scripted-opponent system for training. `src/footballcoach/ui/scenarios.py`
(`ScenarioDefinition`, `ScenarioLoop`, and the various `build_*` functions
like `build_tackle_scenario`) already provides tested, randomisable,
outcome-detecting scenario construction with rules-based AI logic built in
via the existing `orders`/`actions` layer - `ai/env/scenario_env.py` should
wrap these directly, and any new scenario shape a curriculum phase needs
should be added to `ui/scenarios.py` itself (as a new `ScenarioDefinition`),
not to a parallel training-only module.

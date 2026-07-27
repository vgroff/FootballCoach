# generation/

Generates `PlayerAttributes` for players, per the design goal that skills
should be correlated (e.g. faster players tend to accelerate faster) rather
than independently sampled, and that different competition "tiers" should
land in different, tunable bands.

## Approach

1. Attributes are sampled jointly from a multivariate Gaussian, not 8
   independent univariate Gaussians. The correlation matrix (from
   `config/attributes.json["correlations"]`) is turned into a covariance
   matrix (`corr * sigma^2`) and `numpy.random.Generator.multivariate_normal`
   draws all 8 attributes for a player in one call, preserving the
   requested correlations (e.g. `top_speed:acceleration = 0.6`).
2. The result is clipped to `[0, 1]` (some tail samples will naturally fall
   outside this range for a Gaussian - clipping is the simplest, cheapest
   fix and matches the "rare to see >0.9" intuition from Idea.md, since
   clipping doesn't inflate the population *at* 1.0 meaningfully for
   realistic sigma values).
3. "Tiers" (`premier_league`, `championship`, ..., `league_three`, `generic`)
   are just different `(mean, sigma)` pairs applied uniformly to all 8
   attributes before the correlation/clipping step. Per Idea.md's explicit
   requirement: Premier League squads should land ~0.6-0.85, League Three
   ~0.2-0.3, while still being "competent" (not near 0). See
   `tests/unit/test_attribute_generation.py` and
   `tests/balance/` for tests validating these bands empirically.

## Justification for the correlation magnitudes

The three correlations in `attributes.json["correlations"]` are a first,
intuition-based pass - there was no explicit numeric target from Idea.md for
these (unlike e.g. penalty scoring rates), so treat the magnitudes as a
starting point to tune by feel rather than something derived from a hard
constraint:

- `top_speed:acceleration = +0.6` - a fairly strong positive correlation,
  reflecting that raw athleticism (fast-twitch muscle, general leg power)
  drives both traits together in real athletes. Strong enough to feel like
  a real "athletic" archetype exists, not so strong (e.g. 0.9+) that the two
  attributes become redundant.
- `kick_precision:ball_control = +0.4` - a moderate correlation for "close
  technical skill" - technically gifted players tend to be decent at both,
  but plenty of real players are lopsided (great control, average passing
  range, or vice versa), so this is deliberately weaker than the
  speed/acceleration link.
- `tackling:dribbling = -0.2` - a mild negative correlation, modelling a
  soft trade-off between "destroyer" and "creator" player archetypes
  without making them mutually exclusive (a 0.7 dribbler can still roll a
  reasonable tackling score; this isn't a hard opposite-ends-of-a-spectrum
  design).

If squads feel too homogeneous (every fast player is also a great
accelerator with no exceptions) or too random (no sense of player
"archetypes" at all), these are the first constants to adjust - increase
magnitude for a stronger archetype effect, decrease/remove for more
independent, unpredictable squads.

## Justification for the tier mean/sigma values

Verified analytically (using the normal CDF, since each attribute is
marginally $\mathcal{N}(\text{mean}, \text{sigma}^2)$ before clipping/
correlation reshuffle - correlation only affects the *joint* distribution
across attributes for a single player, not each attribute's individual
marginal spread):

| Tier | mean | sigma | Target band | P(in band) | P(attr > 0.9) |
|---|---|---|---|---|---|
| `premier_league` | 0.72 | 0.08 | 0.6-0.85 (user spec) | 88.1% | 1.2% (rare, as intended) |
| `championship` | 0.55 | 0.09 | 0.35-0.7 (interpolated) | 93.9% | ~0% |
| `league_one` | 0.40 | 0.09 | 0.2-0.55 (interpolated) | 93.9% | ~0% |
| `league_two` | 0.32 | 0.08 | 0.15-0.45 (interpolated) | 93.1% | ~0% |
| `league_three` | 0.25 | 0.07 | 0.2-0.3 (user spec) | 52.5% | ~0% |
| `generic` | 0.50 | 0.20 | n/a (population baseline) | n/a | 2.3% |

Only `premier_league` and `league_three` had explicit bands from Idea.md;
the middle tiers (`championship`, `league_one`, `league_two`) were
interpolated by eye to space the English football pyramid out sensibly
between them and haven't been balance-tested as rigorously - if you add
explicit tests/targets for those tiers, check the band-fraction the same
way (see the calculation this table is based on, in the chat history /
re-derivable via the normal CDF given mean/sigma/band). `league_three`'s
52.5%-in-band figure looks low compared to the others, but that's because
its target band (0.2-0.3) is deliberately narrow (0.1 wide vs. 0.25-0.35
for the others) - the *sigma* (0.07) is still tight relative to that band,
it's just a smaller window by design.

The `generic` tier (mean=0.5, sigma=0.2) is the "population baseline" used
when no tier is specified - it's what gives the "`P(attr > 0.9) ≈ 2.3%`,
i.e. rare" intuition from Idea.md's original brief, before any tier-specific
skew is applied.

## Usage

```python
from footballcoach.generation import generate_attributes, generate_squad

striker_attrs = generate_attributes(tier="premier_league")
squad_attrs = generate_squad(size=11, tier="championship")
```

Pass an explicit `rng` (either a `random.Random` or a
`numpy.random.Generator`) for reproducible generation in tests; omit it for
a fresh, unseeded generator in normal game use.

## Extending

To add a new tier, add a `(mean, sigma)` entry under
`config/attributes.json["tiers"]` - no code changes needed. To add a new
correlation, add a `"attrA:attrB": rho` entry under
`config/attributes.json["correlations"]` (both attribute names must be in
`attribute_order`). Keep `rho` in `[-1, 1]` and be mindful that adding many
strong correlations can make the covariance matrix ill-conditioned/non-PSD;
if `multivariate_normal` starts raising warnings about that, reduce the
magnitude of the added correlations.

# DSL updates: the next version

Companion to `reference_math.tex` (the current DSL) and
`reference_math_andrew_notes.txt` (the review).  Decisions from the review
discussion, October 2026.  The programmatic shape is in
`dsl_interface.md`; this note is the math.

What does not change: the two models and the identity `Phi_l = F_{l-1}`,
the window estimator with AIS, the cluster expansion, the toys, the
read-based factors (pair, count, unary) and painted channels of the
current DSL.

## Committed

### C1. Each channel owns its sampler

The forward kernel of a level is any kernel that leaves
`exp(-(E_l + Phi_l))` invariant at `T = 1`; which one is the channel's
choice.  Enumerating Gibbs for small domains (`TableChannel`), the
bit-sliced sampler for tiles (`BitTileChannel`), candidate-set Gibbs for
large domains (`CoordChannel`: current value, neighbours' coherent
continuations, random proposals, chosen without reading the value being
resampled).  Training uses the channel's own sampler inside AIS, so `F`
is the free energy as that sampler realises it.

Learned potentials live only on coarse channels: `F_0 = 0`, so the tile
level's energy is entirely designed and the tile sampler never sees a
learned term.  The bias from learning enters small-domain samplers only.

*The criterion for a latent.*  With the coarse level clamped, the fine
kernel's autocorrelation time should be close to one; if it is not, the
latent is badly defined (Brandt and Ron).  This is the test that the
fixed sweep budget `S_l` is enough, and it is measured per channel set.

*Post-relaxation.*  After the `S_l` honoured sweeps, `p` sweeps with the
parent constraint loosened (soft honour, or none) repair local scales
where `Phi` above was inaccurate, at the cost of drifting from the coarse
sample.  A knob per level, default 0.

*Candidates from the allowed set.*  A candidate-set sampler proposes from
the values the parents' writes admit at the site, not from `Dom_c`, so
forbidden values cost nothing.  The cap `K` on the set makes the work
per site constant whatever the size of the domain.

*Dormancy (the no-op).*  When the parents' writes into a block admit
exactly one value, the channel's cells in that block are fixed at it
before the level is sampled; the sweep skips them.  Decided once per
block when the level's inputs are assembled (the chain is top-down, the
parents are final), one bit per (channel, block).  It costs nothing
elsewhere: the admitted value is the reference `r`, whose energy is zero
in the reference gauge; a dormant block has one configuration, so its
free energy is zero and the parent's `Phi` needs no term for it; for a
certificate, `r` has mass 0 and `d = INF`, valid with no witness, so an
active neighbour sees a dead end and no bookkeeping runs on the dormant
side; windows exclude dormant blocks from the fine region.  The sampler
needs one generic rule (a single admissible candidate means fixed) and a
block-level fast path (an active bit per block tested before the block's
cells; for the bit-sliced sampler, before its planes).

### C2. Learned potentials bias the sampler additively over the candidates it exposes

For a coarse channel `c`, `Phi_c` is a sum of local learned terms,

    Phi_c(z_c) = sum_p psi_theta(z_p, z_{N(p)}),

with `N(p)` a declared neighbourhood.  The general per-site form is a
value feature against a context encoding,

    psi_theta(z_p, ctx) = e(z_p) . h_theta(ctx),

which is what any network that outputs energies over a set of values
looks like at its last layer; `h_theta` may be anything, and **nothing
below relies on it being linear**.

*How it enters sampling.*  The channel's sampler, at site `p`, is
drawing over a candidate set `C_p` (all of `Dom_c`, or the subset of C1).
It hands `C_p` to the potential and receives an additive energy vector

    Delta_p(t) = Phi_c(z with z_p = t) - Phi_c(z),   t in C_p,

which it adds to the designed energies before the draw.  The sampler
normalises over `C_p` itself, so `Delta_p` is needed only up to a per-site
constant.

*Two admissible forms.*  (i) A difference of one joint `Phi_c`, including
the `psi` terms of every neighbour `q` whose context contains `p`; exact
by construction.  (ii) A directly learned conditional: energies over
`C_p` predicted from the context at `p` alone.  This is what Brandt and
Ron's P+ tables are (`reading.md`); it need not be the conditional of
any joint, and the pair consistency condition (for neighbouring sites
`p, q` and values `t, t'`: the product of one-site odds around the square
`(t,t') -> (t',t') -> (t',t) -> (t,t)` must close) is reported as a fit
diagnostic.  Because the training targets come from one consistent
distribution the violation is bounded by model error.  Prefer (i) where
it is cheap; (ii) is the general case and the one trained by
pseudo-likelihood below.

*Ways to compute it* (the potential's choice, invisible to the sampler):

- *Materialised*: when `psi` decomposes into unary and pair terms and
  `D_c` is small, precompute `u_c(v)`, `g_{c,d}(v, v')` once and look
  them up.  Zero learned-model calls at sampling time.
- *One pass*: `h_theta(ctx_p)` once per site visit, one dot product per
  candidate.  A joint difference only when `h` is linear in the
  neighbours' features; otherwise it is form (ii).
- *Batched delta*: re-evaluate the `|N(p)| + 1` terms touching `p` for
  each candidate, one batch over `C_p`.  Form (i) for any `psi`.

*Feature set.*  `e : Dom_c -> R^k` is a property of the channel, fixed
before fitting: learned (token embeddings), derived from what `v` asks
of the finer level, or mixed.  Example of a derived set, the **stamp**:
paint `v` alone in its block's frame and take the indicator over
(footprint cell, paint value).  With `h` linear and the pair matrix tied
to the stencil geometry this is the paint potential of
`reference_math` section 4 over coarse values, exact on held-out pairs;
one-hot and learned features were not (`convpot_test.md`).  The stamp is
one option, not the design.

*Training: two estimators for the same object.*  Both deliver the
per-site candidate energies up to a constant, i.e. log ratios of the fine
partition function between candidates; neither needs `Z` itself.

- *Pseudo-likelihood on joint samples* (preferred where available).
  Sample the bidirectional model `p*` offline, fine and coarse together;
  train the conditional by cross-entropy: softmax over the sampler's
  candidate set, logits `= -(E_c(t) + Delta_p(t))` with the designed
  energy as a fixed offset so only `F` is learned.  Counting a table per
  context is the tabular case.  No windows, no AIS, no partition
  function: the fine level is integrated out by having been sampled.
  Joint samples are free when the latent is a deterministic view of the
  level below (block the fine samples); for a free latent with a
  footprint they need collapsed moves (latent and footprint together),
  which is the cost.
- *Window free energies* (`reference_math` section 4).  `F` per window
  by AIS on the fine channel's own sampler, ridge or Adam on
  `x(W) . theta = y(W)`.  Needs no joint sampling; pays a partition
  function estimate per window; fits `F` as the budgeted kernel realises
  it.  The route for free latents.
- Where both apply (Potts: the mid level is a view of the tiles under
  hard honour) they cross-check each other.

*Adaptive features.*  Grow the feature set or stencil where the residual
(or the consistency violation) is significant and nowhere else, matching
truncation error to statistical error (Brandt and Ron's neighbourhood
tree).  Near-locality, the conditional changing by `O(exp(-c r))` under
changes at distance `r`, is why starting small is safe.

*Baseline and a design rule.*  The support of `Phi` (forbids from
overlapping hard writes, honourability) is computed, never learned.
Run first with the finite part zero plus `p` relaxed sweeps per level
(post-relaxation: honour loosened, the fine level repairs local scales).
The residual against the oracle splits into what relaxation repaired
(local) and what it did not (what `Phi` must carry).  Rule: make every
constraint you cannot afford to get wrong hard, so it lives in the
support; the finite part carries preferences only.

Recursion upward is unchanged.  Colouring of `c` follows `N(p)` plus the
reach of the features.

*Not on the tile level.*  `F_0 = 0`; the tile sampler never sees a
learned term.

### C3. Documentation

Done in `reference_math.tex`: factor as any local energy per candidate
(tables, convpot, certificate); pair = one offset applied everywhere,
neighbourhood as the declared offsets; `Phi_l` enters the kernel as
same-level rows; candidate-set Gibbs for large domains; promotion
motivated.  To do: drop *honour* as a category (a hard parent factor),
state the direction rule once (a channel reads levels `>= l` only; finer
levels reach it through `Phi_l`).

### C4. Constraints among latents

A constraint between latents that leaves no fine-level signature (two
discs that merely overlap when both only request "fill") is not induced
and is never learned.  It is a designed factor at the latents' own level,
computed by a formula over the two values' footprints (`FormulaRow` in
the interface: distance, overlap, containment), generic across types.
An effect that the fine level induces but non-locally (support or
connectivity threading between two latents through tiles neither writes)
is invisible to any local feature of the writes; the fit residual is its
only trace.  The remedy is the same principle one level up: a coarse
certificate (support or connectivity at block resolution), so the
constraint is designed where it is local.  Neither is a learning problem.

### C5. Scaling the library

Hundreds of latent types (trees, houses, castles, mountains) is the
large-domain case of C1 with the hierarchy supplying the sparsity.  No
rule per type pair.

- *Families, one level up.*  A biome / role channel at a coarse level
  writes a hard mask over **families** into the latent channels below:
  a designed pair table biome x family, a few dozen entries.  "Trees
  never in a desert" is one entry.  The same write makes the tree
  channel dormant there (C1).
- *Each type on the level matching its footprint.*  One slot channel per
  level (8-blocks: trees; 32: houses; 128: mountains), value =
  (family, variant), family a view.  A 128-block has one mountain slot
  and never considers trees; most slots are `r`, which is always a
  finite candidate.  Kind / variant as a promotion when the variant
  space is large: exact draw on the small kind channel, candidate-set on
  the variant.
- *Constant work per block.*  Candidates from the allowed families,
  capped at `K` (C1).  Work per site is `K x (designed rows + bias)`
  regardless of the library size.
- *Interactions never enumerated.*  Types interact through the children
  they both write to; the learned potential scores that from their
  features (C2), so no tree x house parameter exists.  Rules with no
  fine signature are formula factors (C4).  The context encoder sees the
  parent's mask, so `Phi` is biome-conditioned without a model per biome.
  Parameters scale with the feature dimension, not the type count.
- *Adding a type.*  A schematic (its stamp or other derived features), a
  family, a level.  Zero new learned parameters with derived features;
  one initialised row with learned ones, which is the case for mixed
  features: derived to start, learned dimensions where the residual asks.
- *Types without a fixed footprint* (rivers, roads) are not slot values:
  each is a channel with a certificate, dormant wherever its family is
  masked out, paying only on the corridor the biome level allows.
- *Biome composition from examples* is a count factor (biome value x
  family counts in its blocks) fitted from a corpus; the DSL has the
  factor, the estimator has not been run on it.

## Deferred: the bias field

Considered and set aside until several channel sets write into one tile
level (open item 3 of `reference.md`), where the painted alphabet starts
enumerating combinations of writers.  Kept here for then.

- *Field as primitive.*  Every channel exposes `B_a : Lambda_a -> R^{D_a}`;
  coarser channels write additively (`B_p += T[alpha(.), beta(z_b(q))]`),
  count factors as block writes `C_q(n) = T[beta(z_b(q)), n]`, painters
  as a compression `B_p += Theta[pi_p]`.  Circles: disc `(kappa, 0)` +
  ring `(0, kappa)` = the old `conflict` value; `n` rings = `(0, n kappa)`.
- *Communication rules.*  Direction (writes from levels `>= l` only),
  range (a declared radius in parent cells), same-level writes
  re-applied on change, fields assembled once per level before its sweeps.
- *Response functional.*  `Phi` as a property of the fine channel,
  `R_a(B) = sum_s f_0(B_s) + corrections`, `f_0(B_s) = -log sum_t
  exp(-B_s(t))` exact when fine cells are independent given the field.
  One `R` per fine channel shared by every writer.  Scoring a coarse
  candidate then overlays its write on the field over its footprint, a
  fine-level touch at sampling time, which is what C2 avoids.
- *Inbox typing.*  `BitInbox` (context slots + quantised row tables) vs
  `DenseInbox`; writers typed by inbox.

## Open

- Q1. Per channel: learned, derived, or mixed features?  Start derived
  and small, grow adaptively (C2); decide on circles + Potts first.
- Q4. Joint samples for free latents: collapsed moves on a periodic test
  world, or windows + AIS only.  Decide on circles (objects with
  offsets) by comparing the two estimators' tables.
- Q2. The feature of an exemplar coordinate: its window's statement
  (the masked exemplar patch) is a stamp by construction; is that the
  whole feature or does it need learned dimensions for texture?
- Q3. The first real library (C5): which level each type sits on, and
  whether the biome x family table stays small once roles are counted
  from a corpus.

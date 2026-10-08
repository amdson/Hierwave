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

which it adds to the designed energies before the draw.  The contract is
that `Delta_p` is a difference of the one joint `Phi_c`: it includes the
terms `psi` of every neighbour `q` whose context contains `p`, not only
`p`'s own term.  Own-term-only energies (`e(t) . h(ctx_p)` alone) are a
dependency network with no joint, and are not allowed: they break the
identity and make the result sweep-order dependent.

*Ways to meet the contract* (the potential's choice, invisible to the
sampler):

- *Materialised*: when `psi` decomposes into unary and pair terms and
  `D_c` is small, precompute `u_c(v)`, `g_{c,d}(v, v')` once and look
  them up.  Zero learned-model calls at sampling time.
- *One pass*: compute `h_theta(ctx_p)` once per site visit, then one dot
  product per candidate.  Meets the contract exactly only when `h` is
  linear in the neighbours' features (the bilinear form), because then
  the neighbours' terms fold into the dot product.
- *Batched delta*: for each candidate, re-evaluate the `|N(p)| + 1` terms
  that touch `p`, as one batch over `C_p`.  Exact for any `psi`; cost
  `|C_p| x (|N(p)| + 1)` local evaluations, done at the coarse level.

*Feature set.*  `e : Dom_c -> R^k` is a property of the channel, fixed
before fitting: learned (token embeddings), derived from what `v` asks
of the finer level, or mixed.  Example of a derived set, the **stamp**:
paint `v` alone in its block's frame and take the indicator over
(footprint cell, paint value).  With `h` linear and the pair matrix tied
to the stencil geometry this is the paint potential of
`reference_math` section 4 over coarse values, exact on held-out pairs;
one-hot and learned features were not (`convpot_test.md`).  The stamp is
one option, not the design.

*Training.*  Unchanged: windows, `y(W)`, AIS on the fine channel's own
sampler; `theta` by ridge when `Phi` is linear in it, Adam otherwise.
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

- Q1. Per channel: learned, derived, or mixed features?  Decide on
  circles + Potts first, where the exact `Phi` is known, then roots.
- Q2. The feature of an exemplar coordinate: its window's statement
  (the masked exemplar patch) is a stamp by construction; is that the
  whole feature or does it need learned dimensions for texture?
- Q3. The first real library (C5): which level each type sits on, and
  whether the biome x family table stays small once roles are counted
  from a corpus.

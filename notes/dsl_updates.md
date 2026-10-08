# DSL updates: the next version

Companion to `reference_math.tex` (the current DSL) and
`reference_math_andrew_notes.txt` (the review).  Decisions from the review
discussion, October 2026.  The programmatic shape is in
`dsl_interface.md`; this note is the math; the test problem and the
staged build plan are `circles_biome_test.md`.

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
sample.  A knob per level, default 0.  Measured (`circles_biome_test`
stage 3): harmless once the finite part is in, destructive before it,
because a loosened constraint weaker than the unary gap between values
(ten nats between two families' presence costs) is simply overridden.
Rule: never relax with a constraint softer than the unary gap, and only
after the unaries are learned.

*Candidates from the allowed set.*  A candidate-set sampler proposes from
the values the parents' writes admit at the site, not from `Dom_c`, so
forbidden values cost nothing.  The cap `K` on the set makes the work
per site constant whatever the size of the domain.  As built (stage 7b):
the parent hard rows are evaluated once per level at `init`, into one
deduplicated admitted list per distinct parent context (a few lists per
world, each the size of what one mask admits); same-level hard rows and
the cap run on the list only.  Per-site work is then flat in the library
size (1.5-1.8 us at `K = 8` from 5 to 20 families, where the full pass
rose to 3.4); what still grows with `D` is the memory of the `D x D`
support and pair tables, a cache effect, to be removed by deriving the
support from footprint overlap per family pair rather than tabulating
it per value pair.

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
it is cheap; (ii) is the general case and the one the bootstrap below
trains.

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

*Training: bootstrap from `q` toward `p*`.*  Level by level from the
tiles up, with the level below already installed.  At level `l`:

1. *Contexts from `q`.*  Run the forward chain; its states are the
   contexts.  Accuracy is spent where the sampler goes (the
   imitation-learning argument: query the expert at the learner's
   states, not the expert's).
2. *Targets from `p*`, locally.*  At each active site `p`, the collapsed
   conditional of `p*` over the sampler's candidate set `C_p`:
   `pi_p(t) = softmax_t(-(E_l(t) + F_{l-1}(t; ctx)))`, with
   `F_{l-1}(t; ctx)` the free energy of the fine region of a one-site
   window with halo, by AIS on level `l-1`'s own sampler with its
   installed potential (`reference_math` section 4), one run per
   candidate; an exact hook where one exists.  Never a global sample of
   `p*`: one level, one window, the shipping kernel at the shipping
   budget, so the cost is bounded by construction and the floor is the
   AIS standard error.  Evaluating only the capped set is consistent.
   **The target must integrate out everything the sampler draws after
   the site**: every finer level, including the site's own children.  A
   one-site conditional of `p*` that reads a finer level (`p*(T | mid)`,
   `p*(T | slots)`, the frozen-site S1f) is the wrong target: it is
   nearly one-hot on the current value, the fit returns `q`'s own
   conditional, and the result is the "leak" of `potts_test.md` (stage
   6b: the same contexts give 0.22 of the oracle's top contrast with the
   frozen target and 0.96-1.00 with the collapsed one).  AIS over the
   child sampler is collapsed by construction; an exact hook must be
   too.  Inside the AIS, rows that read only parents go into `p_0` at
   full strength and are never annealed; annealing a parent row (`inf`
   replaced by `L`) was biased by 0.5-1 nat at 175x the cost.  The
   child's own window is its written cells; neighbours are a fixed
   boundary through the annealed rows.
3. *Move the context toward `p*`.*  Having computed `pi_p`, draw from
   it.  One sweep of this is one `p*`-invariant step on `q`'s sample at
   no extra cost; `K` sweeps give contexts `K` steps closer to `p*`.
   Measured: with collapsed targets `K` buys nothing (stages 5, 6b);
   the leak it was meant to close was a target error.  Default 0.
4. *Update `q`.*  Fit `theta` by cross-entropy of `pi_p` against the
   learned conditional `softmax(-(E_l(t) + Delta_p(t; theta)))` over
   `C_p`, with the designed energy a fixed offset so only `F` is
   learned.  Pseudo-likelihood with the response Rao-Blackwellised: all
   of `pi_p` is used, not one draw from it.
5. Resample contexts under the new `theta` and repeat.  A context's
   targets are fixed numbers once the level below is installed, so the
   dataset of (context, targets) aggregates across iterations and is
   refit from scratch each time; nothing goes stale and there is no
   negative phase.

This is the self-play of `potts_test.md` with the oracle's collapsed
conditionals replaced by AIS: `K = 0` is S1, `K` steps is S3 (CD-K),
`K -> inf` is S0.  The fixed point is a `q` whose conditionals equal
`p*`'s at the contexts `q` visits; with a potential rich enough to
represent them `q`'s kernel is then `p*`-invariant and `q` differs from
`p*` only by how far `S_l` sweeps mix, which the autocorrelation
criterion of C1 tests.  With a potential that cannot (always), `K` is
the dial on where the error goes; the Potts leak (S1 ~0.7 of S0 at the
top, S3 ~0.9) is its measurement.

*Other sources of targets.*  Where joint samples of `p*` are free (a
latent that is a view of the level below: block the fine samples; a toy
with closed-form collapsed moves; an annotated corpus) the target at a
site is the one-hot of the sampled value and step 2 costs nothing; the
same loss, the same code.  Multi-site windows regressed on `F` directly
(the estimator of `reference_math` section 4) are the same AIS runs
with a regression loss; kept as the diagnostic for pair structure.
Where several sources apply (Potts; circles) they cross-check, which is
what the test uses them for.

*What the recursion still costs.*  `F_{l-1}` at level `l` is computed
under level `l-1`'s installed approximation, so error below enters the
targets above.  That is also the point: `F` as the budgeted kernel
realises it.  Training fails only where generation fails: a fine kernel
that does not mix with the coarse values clamped (C1).  Measured: on
circles the recursion costs nothing visible (the top fit through AIS
over the mid sampler equals the exact collapsed hook within eval
noise); on Potts at kappa 1 the top reaches 0.85 of the oracle's
contrast and the rest is attributed to the learned mid tables below.

*Training worlds are periodic.*  Production worlds have no edge.  An
edge in a toy world makes edge objects cheaper (clipped rings), biases
the position-free potentials fitted on edge-and-interior contexts, and
leaks into interior statistics; restricting the evaluation to the
interior does not recover the gate, a torus does (stage 5b).

*Adaptive features.*  Grow the feature set or stencil where the residual
(or the consistency violation) is significant and nowhere else, matching
truncation error to statistical error (Brandt and Ron's neighbourhood
tree).  Near-locality, the conditional changing by `O(exp(-c r))` under
changes at distance `r`, is why starting small is safe.

*Baseline and a design rule.*  The support of `Phi` (forbids from
overlapping hard writes, honourability) is computed, never learned.
The baseline is support plus the zeroth-order unaries: the free-energy
cost of each value in isolation (an object's 18-28 nats; `-log` of the
number of admitted configurations for a mask), which is computed from
the stamps, not learned.  With the finite part at zero the forward
fills every admitted slot and the biome level collapses onto the value
with the fewest children, so "support only" measures nothing (stage 3).
With the unaries in, the residual against the oracle is what the pair
terms must carry (shared-ring attraction: occupancy 0.40 vs 0.73,
contact 0.20 vs 0.34).  Rule: make every constraint you cannot afford
to get wrong hard, so it lives in the support; the finite part carries
preferences only.  Measured outcome (stages 4-5b): with stamp features
and collapsed targets the pair terms are exact on held-out pairs and on
a family never seen in training, conflicts are zero everywhere, and the
end-to-end forward is within eval noise of the oracle on a periodic
world.

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
- Q4. Measured where exact targets exist: AIS targets equal the exact
  hook within noise, the recursion costs nothing on circles and 0.15 of
  the top contrast on Potts at kappa 1, `K` buys nothing.  Still open on
  objects over a Potts texture (`circles_biome_test.md`, stretch), where
  no exact target exists.
- Q5. Support memory: the `D x D` tables per offset (13 MB at 321
  values) are the remaining cost that grows with the library; derive
  the support from footprint overlap per family pair instead.
- Q6. The bit tile sampler behind the `ChannelSampler` protocol, a
  tempered kernel for coordinate channels (so AIS can run over them),
  and certificates with a cap (stage 8).
- Q2. The feature of an exemplar coordinate: its window's statement
  (the masked exemplar patch) is a stamp by construction; is that the
  whole feature or does it need learned dimensions for texture?
- Q3. The first real library (C5): which level each type sits on, and
  whether the biome x family table stays small once roles are counted
  from a corpus.

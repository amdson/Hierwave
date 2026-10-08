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

### C2. Learned potentials are conv potentials over per-value features

For a coarse channel `c`, `Phi_c` is the conv potential of
`reference_math` section 5 with the value embedding made explicit:

    Phi_c(z_c) = sum_p [ a . e(z_p) + sum_{d != 0} e(z_p)^T A_d e(z_{p+d}) + head ]

where `e : Dom_c -> R^k` is the channel's **feature set**: a vector per
value, fixed for the channel, from which every interaction is computed.
`e` and the structure of `A_d` are the two choices; the rest (windows,
`y(W)`, AIS, ridge or Adam, recursion upward) is the existing estimator.

- *Learned features* (token embeddings): `e` the rows of a free `(D, k)`
  matrix, `A_d` free, fitted jointly by Adam on windows.  Generalises to
  unseen value pairs only as far as the embedding extrapolates; at the
  circles mid level (`D = 17`) it did not (`convpot_test.md`).
- *Derived features*: `e(v)` built programmatically from what `v`
  requests of the finer level, so unseen values have features before any
  pair is seen.  Example, the **stamp**: paint `v` alone in its block's
  frame; `e(v)` is the indicator over (footprint cell, paint value).  Tie
  `A_d` to the geometry, `e(v)^T A_d e(v') = sum over cell pairs within
  the stencil of g_delta(paint_v(s), paint_{v'}(s'))`, and the free
  parameters are a stencil `(u, {g_delta})` of a few dozen numbers.  This
  is the paint potential of section 4 written over coarse values; it was
  exact on held-out pairs.  Higher-order merges (three stamps on one
  cell) are omitted at this order and show up in the fit residual.
- Mixed: derived features with a free `A_d`, or derived plus a few
  learned dimensions, when the residual says the stencil is too rigid.

Evaluation is a coarse-level operation and never reads the fine grid at
sampling time.  Small `D`: materialise `u_c(v)`, `g_{c,d}(v, v')` over all
values and the coarse adjacencies once (exact for a bilinear form) and run
them as ordinary rows.  Large `D`: compute `e(t)^T A_d e(z_{p+d})` on
demand for each candidate, cost `k` per neighbour (or the footprint for a
stamp).  The feature set is a property of the channel (`features(v)` in
the interface).

Colouring of `c` follows the interaction's reach: with features that
spill into neighbouring blocks and an 8-neighbour stencil, diagonal
blocks interact, so four colours `(x mod 2, y mod 2)`.

### C3. Documentation

Done in `reference_math.tex`: factor as any local energy per candidate
(tables, convpot, certificate); pair = one offset applied everywhere,
neighbourhood as the declared offsets; `Phi_l` enters the kernel as
same-level rows; candidate-set Gibbs for large domains; promotion
motivated.  To do: drop *honour* as a category (a hard parent factor),
state the direction rule once (a channel reads levels `>= l` only; finer
levels reach it through `Phi_l`).

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
- Q3. Constraints among latents with no fine-level signature (two discs
  that merely overlap) are designed coarse factors, computed by a
  formula, not learned.  Induced but non-local effects (support between
  two objects) need a coarse certificate.  Neither is in the DSL yet.

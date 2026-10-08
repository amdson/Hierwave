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

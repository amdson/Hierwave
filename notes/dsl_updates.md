# DSL updates: the next version

Companion to `reference_math.tex` (the current DSL) and
`reference_math_andrew_notes.txt` (the review).  Each item: what changes,
why, the math, what it replaces.  Built iteratively; numbers in brackets
refer to the review notes.

What does not change: the two models and the identity `Phi_l = F_{l-1}`,
the sampler (Gumbel-max site draws, colouring, the certificate's joint
draw), the window estimator with AIS, the cluster expansion, the toys.
The updates are to how levels communicate and what the learned potential
takes as input.

## U1. The bias field is the communication primitive  [3, 6]

**Change.**  Every channel `a` exposes a field `B_a : Lambda_a -> R^{D_a}`,
one energy per candidate value per cell, starting at zero.  Coarser
channels *write* into it additively.  The kernel at cell `p` scores
candidate `t` by

    e_p(t) = B_p(t) + C_q(n_{-p} + alpha(t)) + (same-level pair terms at p with z_p = t)

(block writes `C_q` defined below) and never reads a coarser channel
directly.  `+inf` entries in the field
are hard.

**Why.**  A parent read at a fixed offset is already a write: cell `q` of
`b` contributes the vector `T[alpha(.), beta(z_b(q))]` to every `p` it
covers.  Making the write the primitive buys three things: position
dependence by construction (what painters bolt on today), any number of
writers compress to one `D_a`-vector per cell, and the learned potential
can take the field as input (U4).

**Math.**  A table parent read `(a alpha, b beta, delta)` becomes the write

    for q in Lambda_b, p with q_b(p, delta) = q:   B_p(.) += T[alpha(.), beta(z_b(q))]

A count factor `(a alpha, b beta)` is also a write, of a function rather
than a vector: cell `q` of `b` writes the row it selects,

    C_q(n) = T[beta(z_b(q)), n],   n in {0, ..., |block(q)|},

into its block, and the kernel at `p in block(q)` scores `t` by
`C_q(n_{-p} + alpha(t))` with `n_{-p}` the sum of `alpha` over the other
cells of the block (a maintained per-block counter).  The parent writes;
the siblings' values the kernel reads are a same-level interaction (U2
rule 3), so the top-down convention holds for every factor.  Two write
types, then: a *cell write* `B_p(.)` and a *block write* `C_q(.)`.

**Example (circles).**  An object's disc demands dirt, `(kappa, 0)` on
`(air, dirt)`; its ring demands air, `(0, kappa)`.  A tile in one disc and
another ring gets `(kappa, kappa)`, the old `conflict` value; a tile in
`n` rings gets `(0, n kappa)`, the old `f_a(n)`.  The painted alphabet
`{free, dirt, air, conflict}` was enumerating sums of writes.

**Replaces.**  `Factors` (pair with `h_b > h_a`), `Painted channels`,
`Honour factors` in reference_math.tex.  Painters survive as a
*compression*: a painted value `pi_p in {0..V-1}` with a lookup
`Theta in R^{V x D_a}` writes `B_p += Theta[pi_p]`.

## U2. Communication rules  [6]

**Change.**  Stated once, for the compiler to check:

1. *Direction.*  During generation a channel at level `l` receives writes
   from levels `>= l` only.  Finer levels reach coarser ones only through
   the learned potential `Phi_l`, at training time.
2. *Range.*  Cell `q` of `b` may write to cells `p` of `a` with
   `h_b >= h_a` and `|q_b(p, 0) - q|_inf <= rho_ab`, a declared radius in
   `b`'s cells (`rho = 0`: the block of `q`; `rho = 1`: the block and its
   eight neighbours, the painters' spill today).
3. *Same level.*  Channels at one level interact through pair tables and
   writes within a declared radius; a sampled channel's writes are
   re-applied when it changes (the painter refresh today).  Open: whether
   same-level writes between sampled channels are allowed or only from
   designed/painted ones, see Q2.
4. *Schedule.*  Fields of level `l` are assembled once from `z_{>l}`
   before the level's `S_l` sweeps; same-level writes are maintained
   incrementally.

**Replaces.**  The scattered statements "homed on its finer side",
"top-down", "painters' spill".

## U3. Fewer named cases  [4, 5]

**Change.**
- *Honour* is dropped as a category.  A write with `+inf` entries is a
  hard write; `honourable` (a coarse configuration whose field admits a
  finite completion) stays, defined where the support of `Phi` needs it.
- *Factor* means any local energy per candidate.  The kinds are: writes
  (U1), same-level tables, the conv potential, the certificate.  Tables
  are the kernel's fast path and what training materialises, not a
  restriction of the DSL.
- *Neighbourhood* [1, 2].  A same-level pair factor is one offset applied
  at every cell (translation-invariant, position-free table).  A channel
  declares its neighbourhood `N_a subset Z^2`; its pair factors are over
  `N_a` and `max |delta|` sets the colouring radius.  Nothing else about
  pairs changes.

**Replaces.**  `Honour factors`, the opening of `Factors`, "the one
computed factor".

## U4. The learned potential takes the field as input  [the NOTE]

**Change.**  `Phi_l(z_l | z_{>l})` is written as a function of the fine
field `B` that `z_l` (and `z_{>l}`) induce on the level below, not of
`z_l`'s values or a painted alphabet:

    Phi_l(z_l) ~= F_theta(B(z_l)),
    F_theta(B) = sum_s f_0(B_s) + sum_{delta in O} sum_s g_delta(B_s, B_{s+delta}) + ...

over fine cells `s`.  First choice for the per-cell term, exact when the
fine cells are independent given the field (circles):

    f_0(B_s) = -log sum_t exp(-B_s(t))

which is also AIS's `log Z_0` per site.  So `Phi = sum_s f_0(B_s) +
(learned correction)`, and the correction is what the fine level's own
interactions (same-level tables, certificate) add.  Linear-in-theta
parametrisations of the correction: features of `B_s` and of pairs
`(B_s, B_{s+delta})`.  The paint potential is the special case where
`B_s = Theta[pi_s]` and the features are one-hot in `pi_s`.

**Why.**  Generalises across writers without enumerating their
combinations (the `conflict` value), and makes the same `Phi` serve
several channel sets writing into one tile level (open item 3 of
reference.md).

**Training.**  Unchanged: windows, `y(W)`, AIS with `p_0 ~ exp(-sum_s
B_s)` (U4's `f_0` is `-log Z_0`), ridge on the correction's features.
Materialisation: tables when `D_c` is small; otherwise evaluate
`F_theta` on the candidate's field footprint at sampling time.

**Replaces.**  `Paint potential`, `Materialisation` in part.

## U5. Promotion, restated  [7]

Unchanged in substance.  Use case: a hard factor on a view `alpha` of a
large proposal-updated channel `u` (an exemplar coordinate) rejects most
candidates and can leave none.  Promote: a small channel `w` carries the
hard factors and is updated exactly; `u` pays `cost(w, alpha(u))`, a
write from `w` into `u`'s field and from `u` into `w`'s.  Marginalising
`w` gives the original model with the hard factors softened by `cost`.
Cost should be a normalised conditional (`-log p(w | u)`), not an
indicator penalty (`channels_system.tex`).

## Open questions

- Q1 [3]. Resolved: count factors are block writes (U1).
- Q2 [6]. Same-level writes between two *sampled* channels: allowed (then
  both refresh on change, cost `D` per neighbour update) or only from
  designed and painted channels?
- Q3 [5]. The certificate under fields: it is a same-level computed
  factor over `(t, d)`.  Does a coarse level ever write into `d`'s field
  (e.g. forcing trunk at a planned root)?
- Q4 [U4]. First correction features to try: `B_s` itself (linear),
  `f_0(B_s)`, products `f_0(B_s) f_0(B_{s+delta})`, or one-hot of the
  argmin of `B_s`.  Decide on circles + Potts where the exact `Phi` is
  known.
- Q5. Memory: a field is `H W D` floats per channel at the tile level;
  fine for `D = 2`, not for `D ~ 64` kinds.  Sparse writes (most cells
  untouched) or a painted compression (U1) where it matters.

## What this does to reference_math.tex

| section | change |
|---|---|
| 1 Objects: Factors | open with the general notion; writes replace parent reads; neighbourhood declared |
| 1 Objects: Honour | removed (U3) |
| 1 Objects: Certificate | "computed factor" wording (done); Q3 |
| 1 Objects: Painted channels | becomes "Fields and writes" (U1) + rules (U2); painters as compression |
| 1 Objects: Promotion | motivation (done); cost as writes (U5) |
| 2 The two models | unchanged; `E_l` is now field + same-level terms |
| 3 Sampler | `e_p(t) = B_p(t) + ...`; otherwise unchanged |
| 4 Training | paint potential -> field potential (U4) |
| 5 Convpot, 6 Self-play, 7 Toys | unchanged; circles example moves to U1 |

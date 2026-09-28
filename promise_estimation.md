# Promises: the abstraction, its interface with p, and how their statistics are estimated

Companion to `promises.md` (the design) and `texture_sampling.md` (the
synthesis stage). This document explains what a promise *is* as a random
variable, what the guarantee rests on, and where the numbers in the promise
tables come from. It is written against the code as of milestones 1–2
(`castlegen/hier.py`, `castlegen/promise.py`,
`castlegen/quantities/connectivity.py`, `castlegen/corpus.py`).

## 1. Two distributions

The default model is the Boltzmann distribution over tile grids,

    p(x) ∝ exp(−E(x) / T),   E = Σ pair tables (Eh, Ev) + Σ unaries,

defined by a JSON tile set. It knows about sockets and local preferences and
nothing else. A **guarantee** G is a global predicate on grids: "all rooms
form one network", "no river cell is a dead end", "every solid run is
supported". What we want to sample is

    p_G(x) = p(x) · 1[G(x)] / Z_G,

p conditioned on the guarantee. Two things make this hard: G is global, so
Gibbs on p cannot be made to respect it locally, and p_G may put its mass on
configurations p rarely produces (p at T = 0.6 has ~188 disconnected pieces
per thousand cells; p_G has one).

The promise system is a way to sample an approximation of p_G with only
local operations, such that G holds *exactly* regardless of how good the
approximation is. Those two properties are delivered by two different parts
of the machinery, and keeping them separate is the whole point:

- **G holds** because of the promise *language* (the summary, its merge rule
  and fulfilment). This is hand designed and proven, not learned.
- **The output resembles p_G** to the extent that the promise *statistics*
  are estimated well and the fine-level filler samples p conditioned on the
  promises. This is the learned part, and it can be bad without breaking G.

## 2. Promises as deterministic functions of the tiles

Fix a block hierarchy: level-0 blocks are K×K cells, a level-(ℓ+1) block is
2×2 level-ℓ blocks. For a quantity, define on every block b of side k:

    summarize(x_b)  -> state      exact summary of the block's cells
    abstract(state) -> promise    projection onto a small finite set Π

and write A_b(x) = abstract(summarize(x_b)). The promise of a block is
therefore a **function of the tiles**, not a free variable. Two requirements
make this a usable hierarchical variable:

1. **Exact merge.** There is `merge_promises` with
   A_parent(x) = merge_promises(A_c00(x), A_c01(x), A_c10(x), A_c11(x))
   for every x. Then the whole pyramid of promises, at every level, is a
   deterministic function of the tiles, computable bottom-up from level 0
   without looking at the tiles again.
2. **Self-satisfaction.** `satisfied(A_b(x), summarize(x_b))` is true for
   every x. A block always honours its own abstraction. This sounds
   tautological, and for a well-formed language it is; the connectivity
   implementation of milestone 2 violates it (section 7), which is why its
   statistics are meaningless.

Consistency between neighbours is not an extra assumption: if two blocks come
from the same grid, their facing sides describe the same seam, so
`consistent(A_a, A_b)` holds automatically. The **soundness theorem** is the
converse used at generation time: if a level-0 promise grid is consistent
everywhere, merges to a top-level value that implies G, and every K-block
honours its promise, then G holds for the tiles. For connectivity: a
connected promise graph plus every block joining its open sides internally
gives one network. For flow: balanced fluxes plus no dead ends inside blocks
gives no dead ends anywhere.

## 3. The induced distribution q

Because A is a function, p_G induces a distribution over promise pyramids:

    q(π) = Σ_x p_G(x) · 1[A(x) = π].

q is the object we estimate. It is a distribution over consistent pyramids
only (inconsistent ones have probability zero), and its marginal at each
level ℓ is a distribution over grids of Π-values. Nothing about q is chosen
by hand; it is whatever p_G does, viewed through the abstraction.

Generation reverses the arrow:

    π ~ q̂                     (top-down, section 5)
    x ~ p(x | A(x) = π)       (fine level, section 6)

If q̂ = q and the fine sampler were exact, the output would be exactly p_G:
p_G(x) = Σ_π q(π) p_G(x | π), and p_G(x | π) = p(x | A(x) = π) whenever π
implies G, which every consistent π does. Errors in q̂ shift which promise
pyramids are drawn; errors in the fine sampler shift the tiles within a
pyramid. Neither can break G.

Two consequences worth stating:

- If A is nearly constant under p_G, q is nearly a point mass and the
  promise level is uninformative but harmless. This is what happened on
  castle at K = 16: 94% of cells are rooms, so every block is "open on all
  sides" and the corpus has one promise value. The guarantee still comes
  from fulfil; the tables just don't help.
- If A is very fine, q is spread over many values and the corpus cannot
  estimate it. The language should be as coarse as exact merge permits, and
  its *energy* should have far fewer parameters than |Π|² (section 5.3).

## 4. Where the corpus comes from

q is defined through p_G, and we cannot sample p_G exactly. `corpus.py`
approximates it:

1. Sample p by Gibbs (`exemplar.gibbs`) on 128×128 tori, the layout's
   temperature schedule, 24 seeds.
2. Push each sample into G with the cheapest available repair: for
   connectivity `exemplar.connect` (a shortest-path carve joining every
   piece to the largest), for flow the fulfil router, for support the
   fill-down. Call this map R.
3. Compute A at every level of every sample (`abstract ∘ summarize`), giving
   24 promise pyramids, i.e. at h = 16/32/64/128 on a 128 map, 1536/384/96/24
   block values.

The distribution of step 2 is R#p, the push-forward of p through the repair,
not p_G. R#p puts all its mass on G, but its density on G is p(x) weighted by
how many p-samples R sends to x, not by p(x) alone. The bias is the repair's
preference for short, cheap joins. A better corpus is any sampler closer to
p_G: the contract sampler in `hier_probe.py` (exact within its own
language), rejection sampling at small sizes, or importance reweighting of
repaired samples by exp(−ΔE/T) where ΔE is the energy the repair added. None
of these is implemented; the tables currently see R#p.

Held-out check for the corpus size: fit on 20 seeds, score the other 4 under
the fitted model, compare with the same score on the training seeds.

## 5. The model q̂ and how its parameters are estimated

### 5.1 Form

q̂ is a chain of level-wise Markov random fields, coarse to fine:

    q̂(π) = q̂_L(π_L) · Π_{ℓ<L} q̂_ℓ(π_ℓ | π_{ℓ+1})

    q̂_ℓ(π_ℓ | π_{ℓ+1}) ∝ 1[consistent(π_ℓ)] · 1[merge(π_ℓ) = π_{ℓ+1}]
                          · exp(−Σ_b u_ℓ(π_b) − Σ_{seams} e_ℓ(π_a, π_b, dir) − Σ_b c_ℓ(π_b, π_parent(b)))

The indicator terms are the hard part and are not learned. The energies
u, e, c are the learned part. This form is chosen because generation needs
exactly one conditional: the distribution of a parent's four children given
the parent and the neighbouring parents' children (the halo). Under an MRF
that conditional is local and computable by enumeration over the consistent
refinements, which is what `PromiseVar.candidates`/`energy` do.

### 5.2 What is estimated now: counts

`Tables.fit` (in `promise.py`) estimates, per level h, from the corpus grids:

- **unary** u_h(π) = −log p̂(π), with p̂ the smoothed frequency of value π
  among blocks at level h (additive smoothing α = 0.5 over |Π|);
- **pair** e_h(π_a, π_b, dir) = −PMI = −log [p̂(π_a, π_b) / (p̂(π_a) p̂(π_b))]
  for horizontally and vertically adjacent blocks, from smoothed
  co-occurrence counts;
- **parent** c_h(π_child, π_parent, pos) likewise, for each of the four child
  positions.

These are moment-matching estimates of a *different* model (independent
pairs) and only approximate the MRF's maximum-likelihood tables, but they are
cheap, deterministic, and the correct thing to try first. They are stored in
`cache/promise-tables-<key>.npz`, keyed by the corpus key.

A subtlety of using counts here: the consistency indicator already forbids
most pairs, so the pair counts of consistent pairs carry the same information
twice (once as −∞ on inconsistent entries, once as low PMI). That is harmless
for sampling but means the fitted energies are not calibrated
probabilities. Pseudo-likelihood (5.3) fixes that.

### 5.3 What should replace counts: pseudo-likelihood with few parameters

Maximum pseudo-likelihood: for every corpus parent at level ℓ, the log
probability of its observed refinement under the conditional in 5.1, given
its halo, summed over parents, maximised in the energies. It is local (one
enumeration per parent), it accounts for the hard constraints properly
(inconsistent refinements get zero mass and do not distort the fit), and it
is what the sampler actually uses, so it optimises the right thing.

Parametrise the energies log-linearly instead of as dense tables:

    u_ℓ(π) = θ_ℓ · f(π),   e_ℓ(π_a, π_b) = φ_ℓ · g(π_a, π_b),   c_ℓ = ψ_ℓ · h(π_child, π_parent)

with a handful of hand-chosen features per quantity, such as number of
open sides, number of groups, whether a group continues across a seam, total
absolute flux, number of springs. Then a level has tens of parameters, the
corpus of a few hundred blocks per level determines them, and |Π| can grow
to what exact merge needs (52 for side partitions) without the tables
becoming sparse. The choice of features is the only remaining hand design in
the learned part, and it is a choice of *what statistics to match*, not of
what values are likely.

### 5.4 The top level

The top level has no parent. For a finite map it is sampled by Gibbs over
seam bits with a global check (for connectivity: open blocks connected).
Its statistics come from the same tables at the top h. For an infinite map it
is a deterministic hashed pattern and carries no learned statistics at all;
everything learned lives in the refinements below it.

## 6. The interface with p at the fine level

The promise pyramid ends at K-blocks. Three mechanisms connect it to tiles,
in increasing order of principle:

- **Coupling into synthesis** (`CoordPromiseCoupling`). During texture
  synthesis, each candidate exemplar coordinate at level h ≥ K is charged
  λ·cost(u, π), the number of promise conditions the exemplar window at u
  violates. It biases which patches are copied. In practice it barely acts:
  synthesis corrects only at h ≤ 8, so the charge is applied via the
  ancestor block's promise at fine levels only.
- **Fulfil.** After tiles exist, each K-block is edited inside its own
  boundary until `satisfied`. This is a deterministic map from
  (tiles, promise) to tiles, an approximation of sampling
  p(x_b | π_b, boundary). It is exact on G and biased on p in the same way
  the corpus repair is: it prefers the cheapest edit. Its edit count is the
  measure of how far the promise was from what synthesis produced.
- **Feasibility-masked Gibbs** (phase 2, `hier_probe.feasible` generalised).
  Sample the tiles of a block from p with every candidate that would make
  the block unable to honour its promise given −∞ energy. This *is* MCMC on
  p(x_b | A(x_b) = π_b), the fine-level conditional of section 3, and it
  removes the repair bias. It costs a block-local summary per candidate,
  which `conn.py` makes incremental.

The repair stage of the pipeline sits between fulfil and the output and must
not undo promises; it currently runs with node signatures forbidden and
fulfilled cells protected. Phase 2 removes the need for that by folding the
promise into the Gibbs mask.

## 7. Diagnostics: is the estimate meaningful?

These numbers say whether q̂ is estimating anything, independently of
whether G holds (it always does).

| Check | Meaning | Healthy | Milestone 2 castle |
|---|---|---|---|
| corpus blocks satisfying their own abstraction | requirement 2 of §2 | 1.0 by construction | 0.0 |
| distinct promise values in the corpus per level | q is not a point mass | several | 1 |
| consistent refinements per parent, mean | the language has room | ≫ 1 | 5 with halo |
| synthesized K-blocks satisfying their promise *before* fulfil | q̂ asks for likely things | comparable to 1 − seam rate | 0.0 |
| fulfil edits, fraction of cells | distance between q̂ and synthesis | a few percent | 13% |
| held-out pseudo-likelihood vs. training | corpus large enough | close | not measured |
| kind / door TV against p samples (`connmetrics.bias`) | output still resembles p | ≤ baseline | worse (0.07–0.09 vs 0.03–0.05) |

The castle column is what a broken abstraction looks like. The milestone-2
connectivity language sets `abstract` to "sides crossed" but `satisfied` to
"sides crossed *and* exactly one component in the block". Row 1 is zero
because no natural block is one component; the promise asks for something
p_G does not do, so rows 4, 5 and 7 follow. Row 2 is a separate effect: at
castle density the honest abstraction is constant anyway.

The fix for row 1 is a language where `satisfied(abstract(s), s)` always
holds and merge is still exact. For connectivity that is the partition of
open sides into internally-connected groups (the port partition of
`conn.py`, coarsened to sides): a block with three separate corridors
passing through it has an honest value, and the pairing-across-a-seam
problem that motivated the one-component rule is caught by union-find in
the merge. The cost is |Π| = 52, which is why 5.3 matters.

Flow and support have honest abstractions as designed: net flux per side is
computed from the cells, every valid block satisfies its flux, and corpus
blocks are valid before mining.

## 8. Summary

- A promise is `abstract(summarize(block))`, a function of the tiles, with an
  exact merge up the hierarchy. The guarantee is a theorem about consistent
  promise grids plus per-block fulfilment; it does not depend on any
  estimate.
- The learned object is q, the distribution of promise pyramids under p
  conditioned on the guarantee. It is estimated from a corpus of repaired
  p-samples, which approximates p_G with a repair bias, by fitting a
  level-wise MRF whose hard constraints are consistency and merge and whose
  soft energies are, today, smoothed counts and PMI, and should become
  log-linear pseudo-likelihood fits with few parameters per level.
- The fine level samples p conditioned on the promises, today by cheapest
  edit (fulfil), eventually by masked Gibbs.
- The test that the estimate is meaningful is not the guarantee (always
  true) but whether blocks satisfy their own abstraction, whether the
  promise values vary, and how much fulfil has to edit.

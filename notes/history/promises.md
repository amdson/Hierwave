# Hierarchical variables: one multiscale MRF for coordinates, promises and labels

Status: history (G1). Superseded by notes/dsl_updates.md + castlegen/channels. Kept for the record.

Design handoff for the next implementation pass. Read with
`texture_sampling.md` (the current synthesis stage) and the module
docstrings named below. Nothing here is implemented except the pieces listed
in section 2.

## 1. The idea

The current pipeline is already a hierarchical MRF, implicitly. At every
level of the Gaussian stack each grid cell holds one variable, its exemplar
coordinate S[p]. The level is initialised from the parent (upsample plus
jitter), then a few correction passes update each cell to the candidate that
minimises a local energy (the neighbourhood match against coherent
candidates). The fine tile is a deterministic function of the finest
coordinate, E[S].

This design makes that structure explicit and generalises it. A level is a
grid of blocks; every block carries a tuple of **hierarchical variables**.
Each variable type defines how it is initialised from the parent, what
candidates a local update considers, and what local energy scores them.
Sampling a level is one generic routine: initialise from the parent, then
sweep, updating every variable of every block against its neighbours, its
parent and the other variables of its block. The fine level (tiles under p)
is the same routine with the tile as the variable.

Three variable types are planned:

| Variable | Domain | Refinement from parent | Local energy | Bottom-up summary | Constraints |
|---|---|---|---|---|---|
| Coordinate (exists) | exemplar position | upsample + jitter | neighbourhood match | none | soft |
| Promise (new) | small abstract set | enumerated consistent refinements | learned tables | exact (`summarize`, `merge`) | hard |
| Appearance label (later) | k classes | sampled from learned tables | learned tables | lossy (clustering) | soft |
| Tile (exists, level −1) | signatures | E[S] or Gibbs from p | p's pair tables and unaries | itself | sockets, structures |

A promise is the special case of a hierarchical variable whose summary is
exact and whose pair and parent couplings are hard. That exactness is what
turns local updates into a global guarantee (every room reaches the gate,
rivers have no dead ends, solids are supported), which is the reason to add
the type. The coordinate variable is the special case with a nonparametric
energy (the exemplar) and no bottom-up summary. The framework is the same
for both, and cross-couplings between variables of one block are ordinary
energy terms.

Locality is the non-negotiable property: every update reads a bounded
neighbourhood at its own level plus the parent, so the whole thing chunks
for infinite generation. Hierarchy artifacts are accepted.

## 2. What already exists

| Piece | Where | Role |
|---|---|---|
| Base model p, tile sets, structures | `castlegen/legacy/tileset.py`, `castlegen/legacy/tilesets/*.json` | tile variable's energy |
| Coordinate variable: stack, upsample, jitter, correction | `castlegen/legacy/texsyn.py` (`Analysis`, `synthesize`, `_correct`) | first instance of the framework, to be refactored into it |
| Exact block connectivity summaries with merge | `castlegen/legacy/conn.py` (`_Summary`) | connectivity promise's `summarize`/`merge` |
| Contract sampler for connectivity | `castlegen/legacy/hier_probe.py` | prototype of the promise variable at fan-out 4, hand-set energies, hard-coded |
| Feasibility-masked Gibbs on tiles | `hier_probe.make_sampler` (`feasible`) | tile–promise coupling |
| Connectivity metrics | `castlegen/legacy/connmetrics.py` | acceptance metrics |
| Shortest-path carve | `castlegen/legacy/exemplar.py` `connect`, `local_connect`, `open_table` | connectivity `fulfil` |
| Masked Gibbs, plain Gibbs | `castlegen/legacy/exemplar.py` `repair`, `gibbs` | tile sampling inside blocks |
| Stage pipeline with metrics | `castlegen/legacy/pipeline.py` | integration point |
| River hack | `castlegen/rivers.py` | deleted once the flow promise exists |

## 3. The framework

### 3.1 Levels and blocks

Levels are indexed from the top L down to 0, then the tile level −1. A
level-ℓ block covers K·2^ℓ fine cells on a side; level 0 blocks are K×K
(K = 16 to start). Fan-out 2: a parent has four children. This differs from
`hier_probe` (fan-out 4) so that enumerating a parent's refinements stays
cheap. The coordinate variable's stack has its own per-level resolution;
align it so that a coordinate level coincides with each block level (the
coordinate lives on the block grid at that level).

A block's **sides** are N E S W; a **seam** is the boundary shared with a
neighbour. A parent's side is the concatenation of two children's sides.

### 3.2 Variable type

```python
class VarType:
    name: str
    hard: bool                      # infeasible candidates get -inf, not a finite penalty

    # ---- top-down
    def top(self, level, shape_or_infinite, rng) -> grid:
        """Values at the top level: a sample (finite) or a deterministic hashed
        pattern (infinite mode)."""
    def init_children(self, parent_value, level, rng) -> dict[(0|1, 0|1), value]:
        """Initial values for the four children.  Coordinates: upsample + jitter.
        Promises: any consistent refinement.  Labels: sample from the table."""
    def candidates(self, block, level, ctx) -> list[value]:
        """Proposal set for a local update.  Coordinates: the 3x3 coherent set.
        Promises: refinements consistent with the parent and current
        neighbours (section 5).  Tiles: all signatures."""
    def energy(self, value, block, level, ctx) -> float:
        """Local energy of putting `value` in this block given its neighbours,
        its parent and the block's other variables.  Sum of unary, pair terms
        per side, parent coupling, and cross-variable couplings (3.4).
        Return +inf for infeasible candidates when hard."""

    # ---- bottom-up (optional; exact for promises, lossy for labels, absent for coordinates)
    def summarize(self, ts, tiles, y0, x0, k) -> value_or_state
    def merge(self, children: dict, k) -> value_or_state
    def abstract(self, state) -> value                     # state -> variable domain
    def satisfied(self, value, state) -> bool

    # ---- fine-level enforcement (promises only)
    def fulfil(self, ts, tiles, y0, x0, k, value, halo, rng, budget) -> (tiles, cost)
```

For promises the bottom-up half must be a homomorphism: `abstract(merge(states))`
equals the promise-level merge of `abstract(state)` for each child. That is
the soundness condition, and it is what makes hard local constraints imply
the global property. Test it (section 8).

### 3.3 The level sampler

One routine for every level, including the tile level:

```
sample_level(ℓ, parent_grid):
    for each variable type v:                      # in a fixed order, e.g. promise, coordinate, label
        grid[v] = v.init_children over parent_grid[v]
    for sweep in 1..n_ℓ:
        for colour in checkerboard (2 colours at fan-out 2; 4 subpasses as texsyn does):
            for each block b of that colour, in parallel:
                for each variable type v:
                    C = v.candidates(b, ℓ, ctx)
                    e = [v.energy(c, b, ℓ, ctx) for c in C]
                    grid[v][b] = sample(C, exp(-e / T_ℓ))       # T_ℓ -> 0 recovers texsyn's argmin
    return grid
```

Texsyn's correction is this loop with one variable, T = 0, 3×3 coherent
candidates and κ on the alternative candidate. `hier_probe`'s base sampler
is this loop at the tile level with `feasible` supplying the −∞ entries. The
promise refinement (section 5) is this loop over promise variables with
candidates restricted to consistent refinements.

Everything a block reads is its own level within a bounded radius plus its
parent, so a window of any level is determined by a padded window of the
parent, exactly as in `texture_sampling.md`'s determinism argument. Use the
hashed noise in `castlegen/legacy/core.py` for all randomness so that chunks agree.

### 3.4 Couplings between variables of one block

These are ordinary energy terms in `energy`, evaluated by whichever variable
is being updated:

- **coordinate ↔ promise.** The estimated cost to make the exemplar block at
  that coordinate honour the promise. Precomputed table over exemplar
  positions at that level × promise values (for connectivity: positions ×
  16). Zero when the exemplar block already satisfies the promise, so texture
  is copied unchanged wherever it can be. Scales with a weight λ. This is the
  "soft prior during synthesis" half.
- **tile ↔ promise.** Hard: a tile candidate that makes its K-block's exact
  state unsatisfiable, given the block's other cells, is infeasible. This is
  `hier_probe.feasible`. Cheaper alternative used first: sample tiles freely,
  then `fulfil` (section 6).
- **tile ↔ coordinate.** The tile equals E[S] initially; in phase 2 (section
  7) the coordinate is dropped and the coupling is replaced by the label
  tilt.
- **label ↔ tile.** A per-kind unary offset inside the block (an exponential
  tilt of p).
- **promise ↔ label, promise ↔ promise, label ↔ label** across neighbours and
  parent: learned tables, section 6.

### 3.5 Bottom-up pass

After sampling level ℓ−1, recompute each variable's summary from the
children and compare with the value held at level ℓ. For promises the two
must agree; disagreement is a bug in `fulfil` or in soundness, and the test
suite treats it as a failure. For labels the disagreement rate is a
diagnostic (the clustering is lossy). For coordinates there is nothing to
compare. The summaries feed the metrics in section 8.

### 3.6 Milestone 1: texsyn as the coordinate variable

Index framework levels by the cell side h in fine cells. A texsyn level l has
h = 2^(L−l) and a grid of side n/h, so one texsyn cell already is one h×h
block; texsyn's levels are the framework's levels. A variable type declares
the h range it exists at: coordinates at every h down to 1, promises at
h ≥ K, tiles at h = 1. Grids of different variables at the same h have the
same shape, so couplings are elementwise lookups.

`Analysis` (stack, PCA, coherence sets) is untouched and becomes the
per-level model held by `CoordVar`. The rest of `texsyn.py` maps as:

| texsyn today | framework |
|---|---|
| `S = zeros(n/m, n/m)` at l = 0 | `CoordVar.top` |
| upsample block in `synthesize` | `CoordVar.init_children`: child (dy,dx) = parent + ((2dy−1)h/2, (2dx−1)h/2) mod m |
| jitter line | second half of `init_children`, using `core.noise` instead of `np.random` |
| `first_corrected`, `corrections`, `final_corrections` | `CoordVar.sweeps(h)` |
| the 18 stacked candidates in `_correct` | `CoordVar.candidates(level, colour)`: 3×3 coherent set and C2 partners, minus hΔ |
| NS/NE squared distance × κ, plus β pair energy at h = 1 | `CoordVar.energy(cands, level, colour)` |
| the four (i,j) subpasses, `argmin`, write-back | the generic sampler at T = 0 |
| `seam_mask`, `features`, `_blur`, `_pca` | unchanged |

The sampler is vectorised per checkerboard colour, not per block; the
per-block signatures in 3.2 are the semantics, the implementation takes a
level and a colour and returns arrays, as `_correct` does with `np.roll`.

```python
def run(types, couplings, n, ctx):
    prev = None
    for h in (2**j for j in range(L, -1, -1)):            # coarse to fine
        lv = Level(h, shape=(n // h, n // h))
        for v in types:
            if v.exists(h):
                lv.vars[v.name] = v.top(lv, ctx) if prev is None or not v.exists(2 * h) \
                                  else v.init_children(prev.vars[v.name], lv, ctx)
        for s in range(max(v.sweeps(h) for v in types)):
            for colour in COLOURS:                          # texsyn's (0,0),(1,1),(1,0),(0,1)
                for v in types:
                    if s >= v.sweeps(h) or not v.exists(h): continue
                    C = v.candidates(lv, colour, ctx)       # (a, b, nC, ...)
                    e = v.energy(C, lv, colour, ctx)        # (a, b, nC)
                    for cp in couplings.involving(v):
                        e += cp.energy(v, C, lv, colour, ctx)
                    lv.vars[v.name][colour] = pick(C, e, v.T(h), noise(...))
        prev = lv
    return lv
```

`pick` is `argmin` at T = 0 and Gumbel-max with hashed noise otherwise; that
is the only capability the sampler adds beyond texsyn. Couplings are
separate objects so `CoordVar` never knows about promises. The coordinate ↔
promise coupling holds a precomputed table `cost[h][u, promise]` (the
fulfil-cost estimate of the h×h exemplar window at u) and returns λ times a
lookup; at h < K it consults the ancestor block's promise by integer
division of the cell index.

Behaviour changes, both intended: jitter comes from hashed noise (needed
for chunking), and `synthesize` becomes a wrapper that builds `CoordVar`
from the analysis and the r, κ and correction settings, runs it alone and
returns the finest grid, so the pipeline stage is unchanged.

Equivalence test: `run([CoordVar])` and the old `synthesize` with the same
injected jitter array and T = 0 give the same S cell for cell. Keep the old
`_correct` until that passes, then delete it.

## 4. Promise variable types

Choose each promise language to be as broad as possible while keeping the
promise-level merge exact and `fulfil` always possible. Broad means many
concrete blocks satisfy it, so refinement has options and the corpus tables
have support.

### 4.1 Connectivity

Nodes are `is_room` tiles plus the gate; adjacent nodes are joined when their
facing sockets connect (`Dh`/`Dv`). Guarantee: every node reaches the gate
(finite) or lies in one infinite network (infinite).

- **State.** `conn._Summary`: boundary node cells (ports), the partition of
  ports by internal reachability, the count of components touching no port,
  the gate's class. `conn.py` computes and merges this already.
- **Promise.** A 4-bit side set `open ⊆ {N,E,S,W}` meaning: all open sides
  are mutually reachable inside the block, no enclosed components, and a
  block with no open side holds no nodes. Plus a `has_gate` bit in finite
  mode. Sixteen values, thirty-two with the gate.
- **Pair constraint.** A seam is open on both sides or closed on both.
- **Merge.** Parent side bit = OR of its two children's bits. Consistent iff
  internal seams agree and the graph on the four children with edges at open
  internal seams connects every child that has an open side. That condition
  is what makes the merge exact.
- **Refinement candidates.** Enumerate internal seam bits and the split of
  each parent side bit across two children; filter by the condition above and
  by agreement with the neighbours' current children on the shared seams. A
  few thousand cases at most.
- **Fulfil.** `exemplar.connect`'s carve within the block plus margin r, with
  one required crossing cell per open side, chosen by hashing the seam so both
  blocks pick the same cell; wall off or join closed components; delete nodes
  from blocks with no open side. Carving `tag` tiles is always allowed, so
  every promise is achievable.
- **Artifact.** A block cannot hold two separate networks exiting through
  different sides, and at scale K·2^ℓ the layout is tree-like where p might
  be loopier.

`hier_probe.py` implements exactly this contract. Port it; do not rewrite it.

### 4.2 Flow (rivers)

Prerequisite: oriented water in the wilds tile set (rotations of a `water`
kind plus `source` and `sink` kinds). A cell's flow crosses the seam it
points at. Replaces `rivers.py`.

- **State.** Net outward flux per side and the count of sources minus sinks
  inside; conservation holds by construction.
- **Promise.** The same, clipped to a small range (flux ∈ {−2..2}, net
  sources ∈ {−1,0,1}), balanced.
- **Pair constraint.** Flux out of one side equals flux into the facing side.
- **Merge.** Add children's fluxes per parent side; internal seams cancel.
  Exact.
- **Refinement candidates.** Internal seam fluxes and side splits such that
  every child balances. Small integer enumeration.
- **Fulfil.** Route each unit of incoming flux to a side or sink that needs
  it by a shortest-path carve of oriented water with capacity bookkeeping;
  fill dead ends. Structures get a margin so a route always exists.
- **Artifact.** Flux is quantized and small. Loops inside a block are neutral
  and left to p.

Land connectivity and flow coexist through bridges, which are road tiles with
water sockets on their sides: nodes for connectivity, water for flow. Both
promises live in the same block tuple; `fulfil` runs flow first, then
connectivity with the water protected.

### 4.3 Support (side view, phase 3)

Needs a side-view tile set with a gravity direction. Rule: every maximal
horizontal solid run in row y lies within a solid run in row y+1, except on
the ground row.

- **State.** Full bottom-row and top-row solid masks; per vertical side, the
  rows where a run crosses; an interior pass flag.
- **Promise.** Bottom and top masks at full resolution (bit vectors, split
  on refinement, learned with a chain model along the seam); per vertical
  side one integer h: runs cross the seam exactly on the lowest h rows.
  Restricting crossings to a down-closed row set is what makes the vertical
  seam sound with one number.
- **Pair constraint.** Horizontal seam: each run in the upper block's bottom
  mask lies within a run of the lower block's top mask. Vertical seam: equal h.
- **Merge.** Masks concatenate; the internal horizontal seam is checked by
  containment, the internal vertical seam by equal h; the parent's h composes
  from the lower child's h and, if it fills the child, the upper child's h.
  Exact.
- **Fulfil.** Fill solids below unsupported runs down to the next supported
  row; clear crossings above h; set edge columns.
- **Artifact.** No cantilevers across vertical seams unless full-height from
  the block's bottom.

## 5. Sampling a promise level

Promise candidates for a block are not free choices; four siblings must
merge to their parent. So update promises per parent, not per block: a
parent's candidate set is `refine(parent_value, halo)`, the consistent 2×2
assignments given the neighbouring parents' current children on the shared
seams. Score a candidate by the learned energy summed over the four children,
their internal seams, their seams to the halo, and their parent couplings.
Checkerboard over parents; two or three sweeps. Initialise every parent with
the first enumerated refinement so the level is consistent from the start.
This is the section 3.3 loop with the "block" being the parent's 2×2 group.

Top level: finite maps sample a consistent grid directly (for connectivity
`hier_probe`'s top-level Gibbs with a global check is fine, the top level is
small). Infinite maps use `top`: a deterministic hashed pattern that is
consistent and globally valid by construction, e.g. all seams open with
hashed closures that keep every local 3×3 connected; a fixed drainage
direction with hashed meanders; ground at a fixed depth. The regularity sits
at scale K·2^L and is the accepted artifact.

## 6. The tile level

Two phases behind the same interface; both end with `fulfil` per K-block, in
a fixed order over quantities, protecting earlier edits.

- **Phase 1, coordinates.** Keep the coordinate variable through level 0
  with the coordinate ↔ promise coupling from 3.4 active at the levels where
  a coordinate cell is a K-block or larger. Tiles are E[S]. Then `fulfil`,
  then `repair` for socket violations with fulfilled cells protected. This
  reuses texsyn and gives pictures soon.
- **Phase 2, tilted Gibbs.** Drop the coordinate variable. Each K-block is
  sampled from p by Gibbs with the label tilt and, optionally, the hard tile
  ↔ promise coupling from `hier_probe.feasible`; otherwise `fulfil` after a
  fixed number of sweeps, then a few protected sweeps. No exemplar at
  generation time. This is a proper conditional of a tilted p.

Compare the two on section 8 metrics before removing phase 1.

## 7. Learning

Hand designed: promise languages, their merges, refinement, fulfilment; the
coordinate energy is the exemplar. Learned: every table in the promise and
label energies. Do not hand-set them (the `J` in `hier_probe` goes away).

### 7.1 Model per level

    E_ℓ(g) = Σ_b u_ℓ(g_b) + Σ_seams pair_ℓ(g_a, g_b, orientation) + Σ_b parent_ℓ(g_b, g_parent(b))

over the tuple (promise, label) per block, with −∞ on inconsistent seams and
on children that do not merge to their parent. Dense tables for small
domains (16 × 16 for connectivity promises), chain factorisation along the
seam for bit-vector promises.

### 7.2 Corpus

The single exemplar has sixteen K-blocks; it is not a training set. Generate
one: sample p on 128×128 tori with Gibbs over many seeds, then make each
sample satisfy the quantities with the existing tools (`exemplar.connect`
or the contract sampler for connectivity, the flow fulfiller for rivers).
These approximate "p conditioned on the guarantee", the distribution the
promise sampler should reproduce. Run `abstract ∘ summarize` and the label
clustering over every block at every level to get variable grids per level.
Cache them; script under `castlegen/` with a fixed seed list.

### 7.3 Fitting

Pseudo-likelihood per level in JAX: the conditional of each block's tuple
given its neighbours and parent, with inconsistent entries masked to −∞
before the softmax so probability is spent only on consistent
configurations. Small L2 toward tables fitted on unconditioned p samples as a
smoother. Empirical counts with additive smoothing are the baseline.

Diagnostics per level: mean number of consistent refinements per parent, and
the fraction of corpus parents whose observed refinement has low probability
under the fit. Mostly single-option parents mean the promise language is too
fine for the corpus.

### 7.4 Appearance labels

k-means on block kind histograms plus door density from the corpus; each
class's tilt (per-kind unary offsets) is fitted so that Gibbs under p + tilt
reproduces the class centroid histogram. Labels enter the level tables
jointly with promises so that co-occurrences such as "dense district" with
"many open sides" are learned rather than assumed. `texsyn.Analysis`'s PCA
neighbourhood features are a reasonable clustering input.

## 8. Integration, metrics, tests

### 8.1 Stages

In `castlegen/legacy/pipeline.py`:

- `hier`: runs the level sampler from the top down to level 0 for the
  configured variable types and stores the per-level grids in
  `ctx.cache["levels"]`. Params: variable types, K, levels, tables path, top
  pattern, per-level temperatures and sweep counts, jitter r for
  coordinates.
- `tiles`: phase 1 (E[S]) or phase 2 (tilted Gibbs).
- `fulfil`: per K-block, per quantity; records edit counts; sets
  `ctx.cache["protect"]`.

`repair` stays after `fulfil`. `synthesize` becomes a thin alias for `hier`
with only the coordinate variable, which is the first refactor milestone.
`connect` remains as a baseline. `rivers` is deleted once flow works.

Metrics to add: per quantity and level, the fraction of blocks whose exact
summary satisfies their promise; max `fulfil` edits per block; label
disagreement rate. Existing global metrics (`comps_per_1k`, `main_share`,
`bad`, `bias`) remain the acceptance numbers.

### 8.2 Tests

- **Framework equivalence.** `hier` with only the coordinate variable, T = 0,
  reproduces `texsyn.synthesize` cell for cell on the same seed.
- **Soundness.** Random grids: `merge` of children's `summarize` equals the
  parent's `summarize`; `abstract(merge(...))` equals the promise merge of
  the abstractions. For every promise type.
- **Refinement.** Every candidate from `refine` merges to the parent and is
  consistent with the halo, and equals the brute-force filter over all child
  tuples where that is feasible.
- **Fulfilment.** Random blocks, promises and halos: terminates, `satisfied`
  holds, edits stay within block plus margin.
- **Bottom-up.** After a full run, every level's promise equals the summary
  recomputed from the tiles.
- **End to end.** Castle 128×128: one component containing the gate. Wilds
  with flow: no water cell with out-degree 0 except sinks.
- **Determinism and chunking.** Same seed, same output; a 64×64 window
  generated in isolation with the padded parent halo equals the same window
  cut from a 128×128 run.

### 8.3 Acceptance for phase 1

- Castle: `main_share` = 1.0 in finite mode; `fulfil` edits a few percent of
  cells; max edits per block bounded and reported (`local_connect` currently
  reaches about 35 per 16-block window at r = 4).
- Wilds: continuous rivers under the flow promise with `rivers.py` removed;
  villages still complete.
- Kind and door TV distance against p samples (`connmetrics.bias`) no worse
  than the current pipeline.

## 9. Plan

1. Refactor `texsyn` into the framework: `VarType` for coordinates, the level
   sampler, `hier` stage. Pass the equivalence test. No behaviour change.
2. Connectivity promise `VarType` from `conn.py` and `hier_probe.py`, fan-out
   2, K = 16. Soundness and refinement tests. Finite-mode top level.
3. Corpus generator, empirical-count tables, promise level sampling, the
   coordinate ↔ promise coupling table. Bottom-up metrics.
4. `fulfil` for connectivity from `exemplar.connect` with seam-hash crossings.
   `fulfil` stage. Castle acceptance.
5. Oriented water, flow promise, delete `rivers.py`. Wilds acceptance.
6. Pseudo-likelihood fitting; compare with counts on 7.3 diagnostics.
7. Infinite mode: `top` patterns, chunk test.
8. Labels and phase 2 tile level; compare with phase 1; retire the
   coordinate variable from generation if it wins.
9. Support promise with a side-view tile set.

## 10. Pitfalls

- Promises must be abstract. Concrete boundary states leave refinement with
  one option and turn the coarse level into corpus tiling.
- The promise merge must stay exact. "May connect" gives no guarantee.
- No hand-set energies for learned variables.
- No global checks below the finite top level.
- Seam decisions (crossing cells, flux cells) are computed from the seam's
  coordinates and seed so both blocks agree without communication.
- Promised quantities come out more regular than p. For anything that only
  needs to be statistically plausible, use a soft variable (label or
  coupling), not a promise.
- Keep the coordinate variable's stack levels aligned with block levels, or
  the coordinate ↔ promise coupling has no block to refer to.

## Implementation notes (milestone 1)

- `castlegen/legacy/hier.py`: generic framework, no texsyn import. `Level(h, shape,
  vars)`, `Ctx(seed, levels, data)`, `VarType` (with `h_min`/`h_max` for
  `exists`, and a `salt` that separates noise streams), `Coupling`, `pick`,
  `run(types, couplings, n, ctx=None)`. `run` returns the finest `Level`;
  every level (coarse to fine) is kept in `ctx.levels`.
- `castlegen/legacy/texsyn.py`: `CoordVar(hier.VarType)` wraps `Analysis`;
  `synthesize` is a wrapper (same signature plus optional `T`, `jitter`,
  `ctx`). The old code stays as `_synthesize_legacy` / `_correct_legacy`
  (with `jitter` and `levels_out` hooks) for `tests/legacy/test_hier.py`; delete
  both once nobody needs the comparison.
- Noise: `hier.noise(seed, level, step, colour, y, x, slot) -> float32 in (0,1)`,
  a numpy port of `core.noise`, bit-identical (tested). Jitter uses
  `level = log2 h`, `step = hier.INIT_STEP`, `colour = 0`, `slot = axis`;
  Gumbel-max uses `step = sweep`, `colour` = subpass index, `slot` = candidate.
- Deviation: `CoordVar.top` is zeros plus jitter, not bare zeros, because
  texsyn jitters the top level too (`r[0]` is 1 in the shipped pipelines).
- Deviation: subpasses whose colour has no cells (grid side 1) are skipped;
  legacy ran them on empty slices, so this changes nothing.
- `jitter` override: a list indexed by texsyn level l of int arrays
  (n/h, n/h, 2), or a callable `(l, h, shape)`. Equivalence at T = 0 with injected
  jitter holds cell for cell at every level for three settings (incl. β, all levels).
- Timing (64 -> 128, best of 3): legacy 0.728 s, hier 0.725 s.
- Output changes only through the jitter source. Mean village count over 24 wilds
  seeds is 14.2 (legacy) vs 14.5 (hier).
- The pipeline `hier` stage (8.1) is not added yet; `synthesize` still is the stage.

## Implementation notes (milestone 2)

- Modules: `castlegen/legacy/promise.py` (`Tables`, `PromiseVar` base with the per-parent sweep and energy,
  `CoordPromiseCoupling`), `castlegen/legacy/quantities/connectivity.py` (`State`, `summarize`, `merge`, `abstract`,
  `satisfied`, `merge_promises`, `consistent`, `refine`/`brute_refine`, `ConnectivityPromise` with `top`,
  `init_children`, `proposals`, `window_costs`, `fulfil`), `castlegen/legacy/corpus.py`, pipeline stages `hier`,
  `tiles`, `fulfil`, spec `pipelines/castle_promise.json`.
- `hier.run`: `VarType.group` (1 = per cell, unchanged) and `VarType.assign`. With `group = 2` the colours run
  over the grid of parents, `candidates` returns whole joint updates (a, b, nC, M cells) and `assign` writes
  them. CoordVar is untouched; the equivalence tests still pass.
- Per-parent update: a parent sets its children's 4 internal seams and owns the split of its E and S sides
  (writing the facing bits into the neighbours' children); N/W splits belong to the neighbours. With a fixed
  halo on all four sides no split could ever change (deadlock), hence the ownership. Every proposal is checked
  against the merge of the up to 3 parents it touches, so levels stay consistent; tested at top grid side 1, 2, 4.
- Language made precise: O != {} requires ONE block-local component that crosses every side in O (several
  components could pair up across a closed seam and break the guarantee); a closed parent has empty children.
  "Crossing" = boundary node joined to the node across the edge, so `merge` needs no door checks. `abstract`
  = sides crossed (a projection; states with several components abstract but do not satisfy).
- Top: Gibbs over seam bits (not block values) so pair consistency holds by construction; global check = open
  blocks connected, at least one open. No coupling at the top level.
- Tables: counts + additive smoothing (alpha 0.5), unary -log p, pair/parent terms = negative PMI.
  `cache/promise-tables-<key>.npz` (arrays `u{h}`, `pairE{h}` (2,16,16), `parentE{h}` (4,16,16) [pos, child,
  parent], raw counts). Corpus `cache/corpus-<key>.npy`: 24 samples of p, 128x128, layout's T schedule + clean
  + global connect. Finding: at T = 0.6 p is ~94 % nodes and EVERY corpus block at every level is O = 15, and
  no corpus 16-block is a single component; the tables are degenerate and the sampler outputs all-15.
- Coupling: cost tables per h in {16, 32, 64} computed in ~4 s per run (not cached on disk). Castle coords are
  only corrected at h <= 8, so in practice only the h < K ancestor lookup acts.
- Fulfil: exemplar.connect (cheap joins, no deletion) then a forced shortest path that may overwrite nodes or
  solids (cost 3) or delete a component, whichever edits fewer cells; hub paths as last resort. Blocks that
  already keep the block-local contract are skipped, so a second fulfil is a no-op when nothing broke.
- Deviation, stage order: hier, tiles, repair, fulfil(protect_nodes), repair(no_new_nodes), fulfil. A plain
  repair after fulfil broke up to 87 % of blocks; protecting all nodes and forbidding node signatures
  (`exemplar.gibbs(forbid=...)`) keeps connectivity but leaves fulfil's own violations (bad ~0.011-0.017 vs
  baseline 0.002). The repair stage no longer clears protected orphans.
- Gaps: phase 2, labels, flow, infinite mode, chunk test; pseudo-likelihood fit; kind TV vs exemplar is worse
  than the baseline (0.068/0.089 vs 0.047/0.030); promise variety needs a corpus that is not all-15.

## Implementation notes (milestone 3)

- Tile sets: kinds take `"flow": "N"|"E"|"S"|"W"|"lake"` (rotated with the tile; a list makes one kind per side,
  mass shared) and `"source": true`; flow is part of the signature key. `ts.flow_dir` (-1/0..3), `ts.water`,
  `ts.lake`, `ts.source`. `"flow_rules"` adds soft pair terms (into_water, into_dry, head_on, parallel, anti).
  Wilds: `river` (4 sigs), `spring` (4), `lake`; bridges carry flow (straight E/W or N/S across the road, corner
  S or W via `bridge_corner`/`bridge_corner_w`, tee, cross); the built-in wall is no longer water (unary 6, grey).
- Rule: a flowing cell drains into water, and is a spring, pointed into, or next to a lake. Lakes exempt, and
  they feed their neighbours (so rivers can leave lakes). `flow.flow_violations` -> dead_ends / unfed.
- Promise: side fluxes f_d = #out - #in in {-2..2} (625 values); the internal net source is implied (lakes absorb
  any amount). Convention: a side carries exactly |f| crossings, one direction only. State = crossings +
  count of rule-breaking cells judged with true torus neighbours, so merge is exact and satisfaction everywhere
  at K means zero violations globally; no top-level check. Soundness analogue of the closed-seam pairing: a
  block may point across a side only at its declared out-crossings (block_ok checks this with a virtual ring
  that is dry except at declared crossings), so two blocks cannot "satisfy" each other through an undeclared seam.
- Crossing cells: |f| offsets in [1, k-2] in a seam-hashed order, skipping cells within 1 of a structure on
  either side (fulfil never moves structures, so both blocks agree). Connectivity crossings skip water/structure.
- Fulfil: every out-crossing gets its own source (unused in-crossing, spring, lake-fed cell; else a new spring
  ~k/3 in), every other in-crossing drains into a route or lake (else a new small lake); Dijkstra with water
  cheap (0.3), block rim and structure margins excluded, roads crossed straight as bridges. Then every other
  flowing cell is dried (bridges -> roads) and nearby plain land re-fitted greedily to its neighbours.
- Connectivity on wilds: `cross_socket: "land"`, carve tag `road`, deletions become lakes (flow-safe);
  `exemplar.connect` and `_force_join` cross rivers only by straight bridge jumps that keep the flow.
- Stages: `hier` takes `quantities` + per-quantity dicts; `fulfil` runs them in order, passing flow's
  protect mask (water + structures) to connectivity; `repair` gained `no_new_water`. Wilds order: hier, tiles,
  repair, fulfil, repair(no_new_water), fulfil. `rivers.py`/`rivers` stage deleted; `wilds.json` is the
  baseline without it (synthesize, repair, connect with tag road).
- Deviation, corpus: p on wilds holds ~0.01 % water at T 0.6-1.0 (rivers vanish under relaxation), so the
  flow corpus uses `corpus_source: "synth"` (texsyn from the exemplar + repair), made valid by
  `flow.make_valid` (top-down projection of the sample's own seam fluxes, then fulfil) and one global connect.
- Exemplar: `stroke` of a flowing kind lays it oriented; unfed placed cells become springs, dead ends lakes;
  layout `"flow": true` prunes unplaced water that breaks the rule. The wilds exemplar has 0 violations.
- Gaps: rivers are sparse and short (|f| <= 2 per side at every level, and the sampler tends to low flux);
  fulfil routes are 1 wide with staircases; bad ~0.01 vs baseline 0.000; conn tables on the wilds corpus are
  nearly all 15; pseudo-likelihood (plfit) not used; the flow corpus takes ~20 min to build once.

## Implementation notes (milestone 4)

- Tile set `cliffs.json` (side view): air, stone, soil, turf, ore, one socket; wall tagged `solid` (unary 8).
  `tileset.py`: `"solid"` tag -> `ts.solid`; rules gained `"above"` (ordered, vertical) and `"beside"`. Turf under
  matter costs 3.0 (`bad`). p is ~25 % solid floating blobs at T 0.9 and all air at 0.6, hence layout `"corpus_T"`.
- Rule: a solid outside the band (bottom G = 8 rows) needs a solid below; band cells are exempt and count as support.
  Needs G < K. Language (`quantities/support.py`, I = 2, V = 16): state = bottom-row need mask, top-row solid mask,
  interior violations; merge exact. satisfied = viol 0, need => bh, tf => full (a block with viol 0 satisfies its
  abstraction). Validity: ground blocks bh = 0, others tf => bh. Seam: upper.bh => lower.tf.
- Soundness: consistent grid + satisfied K-blocks => every bottom-row need sits on a fully solid top interval below.
  A consistent parent level makes child seams consistent, so refinement is a product of two independent child
  columns and never depends on the halo (no seam ownership). Not complete: valid grids can abstract inconsistently,
  and K-consistency does not imply 2K-consistency, so corpus grids go through `fix` (global bottom-up fill/clear),
  `project` (top-down nearest refinement) and per-block fulfil; the mined pyramid is then consistent (tested).
- Stages: hier -> tiles -> ground (new: paints and protects the band) -> repair -> fulfil -> repair(keep_solidity,
  via gibbs/repair `keep_class`) -> fulfil; `cliffs.json` (synthesize, repair) is the baseline. Metrics: `solid`,
  `unsupported`. Exemplar: `place` gained `fill`/`blob` (optional `"seed": true`); layout `"support": true` runs
  `fix`. The rule forbids overhangs and caves: the floating chunk becomes a pillar, the cave a shaft.
- Results, seed 1 / 2: 0 unsupported, satisfaction 1.000 at 16/32/64, bad 0 (baseline seed 1: 222 unsupported).
  Before fulfil 0.406/0.188/0.000 and 0.266/0.125/0.000; fulfil changes 28 % / 35 % of cells and solid falls from
  ~0.4 to 0.10: flat ground plus the bottom block row. With flat tables fulfil changes 38 % (solid 0.61, slabs).
- Corpus (p, T 0.9, 24 x 128^2): solid 0.33 raw, 0.097 after `fix` (clearing a floating blob beats propping it up);
  `project` adds 0.8 %. Values: h16 5 distinct (1526/1536 are 0), h >= 32 all 0; refinements with halo 57/113/225,
  single-option parents 0.75/0.50/0.00. plfit one-hot held-out: h16 0.097 vs flat 1.354, h32 0.010 vs 2.708.
  I = 4: 13 values at h16, still all 0 above, 12.6k-50k refinements per parent, so I = 2 ships.
- Gaps: a non-ground block with bh = 0 must be empty and a seam carries support only under a K/2-wide full top row,
  so terrain above the bottom block row survives only as slabs; the p corpus is flat ground; no overhangs or caves.

# The biome circles test: the next-version DSL on a toy with exact answers

The test problem and build plan for `dsl_updates.md` (C1-C5) and
`dsl_interface.md`.  It is `notes/circles_test.md` with three changes:
hard honour, a second object family, and a biome level that masks
families.  Everything the new design must produce has a reference here:
the mid tables are closed form (tiles independent given the objects), the
oracle gives joint samples by collapsed moves, and the stamp fit
(`paintpot_test.md`) is already exact on this geometry.  The companion
notes apply unless stated: training schemes and measurements from
`potts_test.md`, window estimator from `induce_test.md` /
`paintpot_test.md`.

Constraints: numba, single core (`NUMBA_NUM_THREADS=1`), figures under
`images/`, nothing model-specific in the generic layer, the numba kernel
(not `bitgibbs.c`) for every sampler here.

## The model

Tiles (h = 1): `air = 0`, `dirt = 1`, field `mu` per air tile.

Mid (`BM = 8`): one slot channel `obj`, `D = 33`:

| values | meaning |
|---|---|
| 0 | absent (the reference value `r`) |
| 1..16 | disc, centre `(cy, cx)` in `{2..5}^2`, value `1 + (cy-2)*4 + (cx-2)`; footprint the 13-cell radius-2 disc |
| 17..32 | bar, top-left `(ry, rx)` with `ry` in `{2..5}`, `rx` in `{1..4}`, value `17 + (ry-2)*4 + (rx-1)`; footprint the 2 x 4 rectangle |

Views: `self` (identity), `present` (0/1), `fam` (0 none, 1 disc, 2 bar).
Every object demands dirt on its footprint and air on its ring (the
footprint's 8-neighbour dilation minus the footprint: 24 cells for a disc,
16 for a bar); footprints never leave the block, rings spill at most one
tile into a neighbouring block.

Top (`BT = 2` mid cells, 16 tiles): `biome`, `D = 4`: `0 none`, `1 discs`,
`2 bars`, `3 both`.  Views `pal` (identity), `allow_disc`, `allow_bar`
(0/1, the two bits of the value).

Painted channels (deterministic, fixed, refreshed when the source changes):

- `allow`, h = BM, D = 4: the parent biome's value copied into each of its
  four slots.  Painted from `biome`.
- `dem`, h = 1, D = 4: `0 free`, `1 dirt`, `2 air`, `3 conflict` (a
  footprint of one object and a ring of another).  Painted from `obj`.

Designed factors, all position free:

| name | kind | reads | table |
|---|---|---|---|
| `mask` | pair on `obj`, (obj, fam) x (allow, val), offset (0,0) | parent | `INF` when the family's bit is clear in `allow`, else 0 (hard) |
| `pres` | unary (obj, self) | | `-b_disc` on disc values, `-b_bar` on bar values, 0 on absent |
| `honour` | pair on `tile`, (tile, col) x (dem, val), offset (0,0) | parent | free 0; dirt-demand `INF` on air; air-demand `INF` on dirt; conflict `INF` on both (hard) |
| `mu` | unary (tile, col) | | `[mu, 0]` |
| `bio_u` | unary (biome, pal) | | `[0, 4 log 2, 4 log 2, 4 log 3]` (see below) |

No designed obj-obj, biome-biome, or further unaries.  `kappa` and `lam`
of the old test are gone: both honours are hard.

*Per-tile free energies.*  Given the objects the tiles are independent:

| demand | Z | `-log Z` at mu = 0.3 |
|---|---|---|
| free | `1 + e^-mu` | `f0 = -0.554` |
| dirt | `1` | `0` |
| air | `e^-mu` | `mu = 0.3` |
| conflict | `0` | `+INF` |

So an object's free-energy cost over empty neighbours is `F_disc = 13
(0 - f0) + 24 (mu - f0) ~ 27.7`, `F_bar = 8 (0 - f0) + 16 (mu - f0) ~
18.1` (the code's `object_cost(fam)` is the reference); a ring tile
shared by two objects gains `f0 - mu = -0.854`; a conflict tile is
forbidden.  Presence bonuses `b_fam = F_fam - log 16`, so an isolated
allowed slot is 50/50 absent / present whatever the family, and the
learned unary carries only what neighbours and the boundary induce.
The induced biome unary at zeroth order (slots independent, 50/50) is
`-4 log Z_slot` with `Z_slot = 1, 2, 2, 3` for none / discs / bars /
both, i.e. `0, -2.77, -2.77, -4.39`; `bio_u` cancels it so `p*`'s biome
marginal is roughly uniform and the learned `bio_u` has a known target
up to interaction corrections.

*What is induced, and where it lives.*  Mid: the **support** (two objects
whose footprint and ring overlap are forbidden; a pairwise relation
between slots at the 8 neighbouring offsets, derived from the stamps) and
the **finite part** (shared-ring attraction, up to 4 tiles between
adjacent discs; presence corrections at the boundary).  Top: the unary
above plus a pair term through objects sharing rings across the block
edge, and the interaction of the mask with the support (a "both" block
next to a "discs" block has fewer admissible configurations than next to
"none").

*Dormancy.*  `biome = none` admits only `absent` in its four slots: the
slots are fixed at `r` before the mid level is sampled and skipped by the
sweep; their free energy is 0 exactly, so `Phi_top(none) = 0` with no
learned term.  This is the C1 no-op in its smallest form.

## Learned potentials

Mid: a paint potential over the stamp features (`paintpot.py`, OFF8),
i.e. `psi` pairwise, tied to the stencil, over (footprint cell, paint
value) indicators of the candidate and its neighbours.  Trained by either
estimator below; **materialised** into `obj_h`, `obj_v`, two diagonal
pair tables (33 x 33 each) and `obj_u` (33) for the sampler, with the
support rows (`INF` entries) added from the stamps, not from the fit.
The support tables are `conflict_d[t, t'] = INF` iff painting `t` at `p`
and `t'` at `p + d` produces a conflict tile, `d` over the 8 offsets.

Top: `bio_h`, `bio_v` (4 x 4), `bio_u` (4), tabular (one-hot features;
`D = 4`).

Gauge as in Potts: zero mean per table, double centering when comparing
pair tables.

## Interfaces

### `castlegen/channels/circles_biome.py`, mirroring `circles.py`

```
class CirclesBiome:
    def __init__(self, nty, ntx, BM=8, BT=2, mu=0.3, b=None)   # b: dict fam -> bonus, default presence_bonus(fam)
    D (=33), P (=4), BM, BT, nty, ntx, nmy, nmx, H, W
    FAM : (D,) 0/1/2;  STAMP[t] : list of (dy, dx, demand) in the block frame (footprint = dirt, ring = air)
    def object_cost(self, fam); def presence_bonus(self, fam)
    def channels(self) -> (biome, obj, tile, allow, dem)
    def designed_factors(self) -> [mask, pres, honour, mu, bio_u]
    def support_factors(self) -> 8 hard pair factors on obj derived from STAMP (conflict_d)
    def learned_factors(self, theta) -> obj_h, obj_v, obj_d1, obj_d2, obj_u, bio_h, bio_v, bio_u
    def theta0(self)
    def model(self, chans, theta=None, extra=()) -> Model      # allow, dem fixed everywhere
    def paint_allow(self, biome, allow); def paint_dem(self, obj, dem, window=None)
    def admissible(self, obj, i, j) -> (D,) bool                # mask and support at slot (i, j) given the other slots
    def stats(self, biome, obj, tile) -> dict
        # fitted: obj_h, obj_v, obj_d1, obj_d2, obj_u, bio_h, bio_v, bio_u (frequencies)
        # monitors: present_disc, present_bar, conflict (must be 0), contact, bio_hist (4,),
        #           dormant (fraction of slots fixed), edge_air_top (held out, as before)
    def reference(self) -> dict   # exact mid tables by tile enumeration (INF on the support), double-centred
                                  # finite part; top unary at zeroth order; top pairs = None (oracle only)
    def symmetrise(self, stats)   # dihedral group with the induced maps on offsets (bars: 2 x 4 is not
                                  # square, so only the maps that fix the bar shape: identity, mirrors)
    def render(self, biome, obj, tile)

class Forward:      # biome sweeps, paint allow, obj init (dormancy) + sweeps, paint dem, tile sweeps
class Oracle:
    def top_probs(self, i, j) -> (P,)     # p*(T | its 4 slots): bio_u on the values whose mask admits every present family
    def top_move(self, i, j)              # draw T, repaint allow
    def mid_probs(self, i, j) -> (D,)     # p*(o | allow, neighbours' objects), footprint + spilled ring tiles
                                          #   integrated out in closed form, 0 on inadmissible values
    def mid_move(self, i, j)              # draw o, repaint dem locally, redraw tiles whose demand changed
    def tile_sweep(self, n=1)
    def sweep(self, n=1, tile_sweeps=2)   # top moves, mid moves, tile sweeps
    def moments(self, burn, sweeps)
    def joint_samples(self, burn, every, n) -> iterator of (biome, obj, tile) states   # oracle contexts; SampledTargets
```

The top move is the plain conditional `p*(T | slots)`: the biome affects
nothing but admissibility, so given the slots it is a unary over the
values that admit every present family.  The chain mixes between
families through states where all four slots are absent; a block is
50/50 per slot, so this is not slow.  (The old independence-Metropolis
move collapsed over the slots is not needed; keep it only if the biome
autocorrelation turns out long.)

### Generic layer (`castlegen/channels`), the `dsl_interface.md` shape

The old `Model.compile` / `kernel.sweep` path stays; these wrap it.

```
class Sampler:                          # Channel.sweep & co. of dsl_interface.md; one per channel
    TableSampler(model, home, K=None)   # the numba kernel; K caps the candidate set (below)
    def init(rng)                       # paint parents' writes, then dormancy: a site whose hard rows admit
                                        #   one value is set to it and fixed; a block with every site fixed
                                        #   sets active[block] = 0 and the kernel skips it before its cells
    def sweep(rng); def relax(rng, p); def energy(); def sweep_tempered(rng, beta); def unary_logZ()
    def candidates(y, x) -> int[]       # the set C_p the kernel draws over (all admissible, or K of them)

class Bias:                             # dsl_interface.md Bias
    MaterialisedBias(rows)              # pair / unary rows: handled by the kernel as today, no new code
    BatchedDeltaBias(features, psi)     # Python: for each candidate, paint it locally, feature delta, psi;
                                        #   used in training and in the invariance test, not in production
```

Kernel additions (`kernel.py`):

- *Candidate cap.*  With `K` set, the kernel draws over `K` values: the
  current value plus `K - 1` drawn uniformly from the admissible values at
  the site (admissible = finite energy under the hard rows, computed first
  from the hard rows alone).  The Gibbs draw over that set is a valid
  kernel for `exp(-E)` because the set is chosen without reading `z_p`
  beyond including it (`coord.py` invariant).  `K = None` is the current
  full enumeration.
- *Block fast path.*  `active : (rows / hb, cols / hb) bool` per home
  channel, tested before a block's cells.
- *Hard rows first.*  `_energies` evaluates rows with `INF` entries
  before the others and skips the rest for candidates already at `INF`,
  so forbidden values cost one lookup.

Training (`train.py` additions; the bootstrap of C2):

```
class Targets:                          # p*'s collapsed conditional at a site over the candidate set
    def at(self, model, home, y, x, cand) -> (len(cand),) probabilities
class ExactTargets(Targets)             # Oracle.mid_probs / top_probs restricted to cand (this toy)
class AISTargets(Targets)               # softmax(-(E_designed + F)) with F by AIS on a one-site window with
                                        #   halo over the child channel's sampler, one run per candidate;
                                        #   K, M, L as in induce.ais_log_z
class SampledTargets(Targets)           # one-hot of the value in a free joint sample (Potts mid: blocked tiles)

def fit(model, home, targets, iters, n_contexts, K, features, l2) -> theta
    # for it in iters: n_contexts forward runs at the current theta; at every active site of `home`
    # pi = targets.at(...) over the sampler's candidate set, appended to the dataset; K sweeps that draw
    # z_p ~ pi in place (K = 0 is S1, K > 0 is S3 / CD-K); then refit theta from scratch on the whole
    # dataset: sum of KL(pi || softmax(-(E_designed + Delta_theta))), Delta from BatchedDeltaBias
    # (paint potential: linear in theta, convex, L-BFGS).  Returns theta and reports the held-out KL,
    # the pair consistency violation of the targets and of the fit, and the dataset size.
def fit_windows(...)                    # the existing paintpot / induce path, unchanged: the diagnostic
def autocorr_clamped(model, home, parent, sweeps) -> float
    # the fine kernel's integrated autocorrelation time of its energy with `parent` clamped (C1 criterion)
```

`Trainer.install` = materialise (`paintpot.materialise`) + support rows
from `support_factors()` added to `extra`.

## Tests (`tests/test_circles_biome.py`)

- `STAMP` sizes: 13 + 24 for discs, 8 + 16 for bars; no footprint leaves
  its block; ring spill at most one cell.
- `support_factors()` against brute force: paint every pair of values at
  each offset on a 3 x 3 block grid, `INF` iff a conflict tile appears.
- `mid_probs` against brute force over the footprint tiles on a reduced
  footprint (radius-1 disc, 1 x 2 bar, `BM = 4`) and against
  `reference()`'s per-tile formula for a hand-placed pair: disc at (3, 5)
  left and bar at (2, 1) right, counting the shared ring tiles by hand.
- `top_probs` against brute force over the four slots' admissibility.
- Dormancy: a `none` block has all four slots fixed at 0 after `init`,
  `active` clear, and `Model.energy("obj")` unchanged by its sweep.
- Candidate cap invariance: on `CirclesBiome(2, 2)` with the exact
  tables installed, 2000 sweeps at `K = None` and at `K = 8` give the same
  `present_disc`, `present_bar`, `obj_h` within 3 standard errors.
- `fit` with `ExactTargets` on oracle contexts recovers the materialised
  finite part to 1e-2 and reports a consistency violation below 1e-2;
  `AISTargets` with the per-tile exact hook in place of AIS gives the
  same `pi` as `ExactTargets` to 1e-9 at ten random sites.
- Energy identity: `Model.energy("tile")` equals the numpy sum of `mu`
  air tiles with every demanded tile satisfied (and `INF` otherwise).

## Experiment

`nty = ntx = 6` (12 x 12 slots, 96 x 96 tiles), `mu = 0.3`, oracle
`BURN = 50`, `SWEEPS = 400`, forward `S_T = S_M = 30`, `S_F = 20`, final
stats on 64 runs, eval noise as in Potts.  Script
`notes/experiments/circles_biome.py`, outputs `images/cbio_*`, results
appended here.  Stages are gated; do not start a stage before the
previous gate passes.

1. **Interface.**  `Sampler`, `MaterialisedBias`, candidate cap, block
   fast path, hard-rows-first.  Gate: `tests/test_circles.py`,
   `test_potts.py`, `test_paintpot.py` pass through the wrappers, and the
   circles `eta0.5` S0 run reproduces its final L1s within eval noise.
2. **Model and oracle.**  Gate: the tests above; an oracle render with
   `conflict = 0`, both families present, dormant blocks empty.
3. **Baseline: support only.**  Forward with `designed + support`, no
   finite part, post-relaxation `p = 0` and `p = 2` (mask softened to
   `lam = 3` for the `p` sweeps).  Report every monitor against the
   oracle and the top histogram against uniform.  Expected: `conflict`
   0, `present` right in isolation, `contact` low (no shared-ring
   attraction), `bio_hist` uniform where the oracle is not (the induced
   unary is 2.8-4.4 nats).  This is the residual `Phi` must carry.
   Report per-site time at `K = None` and `K = 8`, and the fraction of
   slots dormant.  Post-relaxation has little to repair in this toy
   (tiles are independent given the objects); report it and move on.
4. **The mid level: the bootstrap, exact and AIS targets.**  `fit` with
   stamp features, `iters = 5`, `n_contexts = 8`, in a 2 x 2 grid of
   settings: targets `ExactTargets` / `AISTargets` (AIS `K = 32, M = 16`
   over the one-site window's 37-tile region with halo) and `K = 0` (S1)
   / `K = 3` (S3).  Plus `fit_windows` on 3 x 3 slot windows with the
   exact hook as the diagnostic.  Report, double-centred on the finite
   part: max abs error against `reference()` on held-out value pairs,
   slope and corr, the consistency violation of targets and fit, the
   held-out KL, the dataset size and wall time per setting, and
   `autocorr_clamped("obj", "biome")`.  Gate: exact targets within 0.05
   of the reference on held-out pairs at `K = 0`.  What the grid measures:
   the AIS column against the exact column is the cost of estimated
   targets; `K = 3` against `K = 0` is what moving the contexts buys
   (the circles leak was S1 0.33 of S0 at the top, so expect it to
   matter at the top more than here).  Decides Q1 (stay with derived
   features if the gate holds) and the circles part of Q4.
5. **The top level.**  Install the mid bias; `fit` on `biome` with the
   same grid (`ExactTargets` = `Oracle.top_probs` restricted to the
   admissible values; `AISTargets` runs the `obj` sampler with its bias
   in, over the one-biome-cell window of 4 slots plus halo: the
   recursion).  Compare to oracle moments and the zeroth-order unary;
   check `Phi_top(none) = 0` exactly from every setting.  End to end:
   forward with everything installed vs oracle on every monitor,
   `edge_air_top` held out.  Gate: within eval noise on fitted features;
   `conflict` 0.
6. **Potts cross-check.**  `fit` on the Potts mid level (kappa 8, J 0.1
   and kappa 1, J 0.3) with `SampledTargets` (blocked oracle tiles),
   `ExactTargets` (`Oracle.mid_probs`) and `AISTargets` (transfer matrix
   replaced by AIS over the block).  Report the three tables against each
   other and against the published S1; one short run.
7. **Scaling probe.**  `N` random families (connected stamps of 6-14
   cells inside an 8-block, 16 offsets each) for `N` in 2, 5, 10, 20, with
   a biome whose mask admits a random pair.  Report per-site time, dormant
   fraction and parameter count against `N`; stamp features should give
   zero new parameters per family and flat time at `K = 8`.
8. **Migration.**  `roots.py`, `ground.py` and `coord.py` onto `Sampler`
   (certificate and candidate-set kernels behind the interface).  Gate:
   their existing tests and demo images unchanged.  This is the test that
   the interface is sufficient for the real sets; it does not need
   anything learned.

Stretch, after 8: objects on a Potts texture (footprint tiles coupled by
`J`), where no exact target exists.  The real case of Q4: `AISTargets`
only, judged by the end-to-end monitors against a long oracle chain on a
periodic world, with `K` as the dial.

## Contracts fixed before the build

- *The one-site window.*  For site `p` of channel `c` at level `l` and
  candidate `t`: coarse context = the state's values at every other cell
  of levels `>= l` (no reference value needed, the context is a whole
  state); fine region `R(p)` = the fine cells any candidate at `p` writes
  (here the block plus its one-cell spill border, 10 x 10 tiles),
  dilated by the reach of the fine channel's same-level factors; fine
  cells outside `R(p)` fixed at the state's own fine values.  `F(t; ctx)`
  is the free energy of `R(p)` under `z_p = t` with that boundary; only
  differences across `t` at one site are used.
- *Candidate features at a site.*  `fit` takes a callback
  `feats(model, home, y, x, cand) -> (len(cand), nfeat)`: for each
  candidate the paint-potential feature vector (`paintpot.features`,
  OFF8) of the local paint with `cand[i]` at `(y, x)`, minus the same
  with `absent` there, computed over `R(p)` only.  `CirclesBiome`
  provides it as `stamp_features`.  With `psi` linear the dataset record
  is `(X: (n_cand, nfeat), e: (n_cand,) designed, pi: (n_cand,))` and the
  fit is logistic regression on `X`.
- *Site order during the `K` steps.*  `fit` visits sites in the kernel's
  own order: colour classes from `Model.compile(home).colours`, raster
  within a class, so a `K` step is one sweep of a valid `p*` kernel.
- *The `Targets` protocol* (duck-typed, no shared base class needed):
  `at(model, home, y, x, cand) -> (len(cand),) float64` summing to 1,
  computed from the model's current state.

## Work packages

Four packages start at once; A adds and never renames, so the others
build on today's kernel and merge without conflict.

| package | builds | owns | depends on | gate |
|---|---|---|---|---|
| A. Sampler layer | stage 1: `Sampler` wrapping the kernel, candidate cap `K`, block active bit, hard rows first | `castlegen/channels/sampler.py` (new), additions to `kernel.py`, `core.py` | nothing | old circles / Potts / paintpot tests pass through the wrapper; `test_sampler.py` for cap invariance and dormancy on `Potts(1,1)` |
| B. Model and oracle | stage 2: `CirclesBiome`, painters, `support_factors`, `reference`, `stamp_features`, oracle, render, tests; stamps as data so stage 7 is a parameterisation | `castlegen/channels/circles_biome.py`, `tests/test_circles_biome.py` | current `core.py` only | the tests above; an oracle render with zero conflicts, both families, dormant blocks |
| C. Trainer | `ExactTargets`, `SampledTargets`, `fit`, `autocorr_clamped` | `castlegen/channels/targets.py` (new), additions to `train.py`, `tests/test_targets.py` | `paintpot.features`, the existing Potts and circles oracles | stage 6 on Potts and `fit` on the old circles with `Oracle.mid_probs` targets: tables match the published S1 |
| D. AIS targets | `AISTargets` on the one-site window over the child channel's sampler | `castlegen/channels/aistargets.py` (new), `tests/test_aistargets.py` | `induce.ais_log_z`; the `Targets` protocol | equals an exact hook to 1e-9 on the old circles; within the AIS standard error without it |

Integration order: A first; stage 3 (A + B); stages 4-5 (B + C + D);
stage 7 (A + B); stage 8 (A only, by whoever built A).

## Corrections after the build

- *Symmetry.*  The default bar offsets (`ry` 2..5, `rx` 1..4) are not
  mirror-symmetric inside the block, so `p*`'s exact symmetry group for
  the default families is the identity only; `group()` finds the valid
  elements for any family set, and the test uses a symmetric bar range.
- *The biome marginal is not uniform.*  `bio_u0` cancels only the
  zeroth-order induced unary; shared-ring attraction and cheaper edge
  objects push `p*` to about discs 0.52, both 0.39, bars 0.08, none 0.01
  (confirmed by exact enumeration on `CirclesBiome(1,1)`).  So dormant
  blocks are rare at the default dials; raise `bio_u0[none]`'s
  competitors or lower `b_disc` when the dormancy path needs exercising.
- *Support tables.*  Four canonical offsets suffice (`Model.compile`
  adds the reflections); the diagonal tables are zero for the defaults
  since rings spill sideways only.
- *`stamp_features` paints a 12 x 12 window*, two tiles beyond the
  block, so the OFF8 pairs across the edge of `R(p)` match the
  full-grid difference exactly.
- The designed biome unary is `bio_u0`; `bio_u` is the learned one.
  `Forward` adds the support factors by default.

## Results

### Stage 3: support-only baseline

Script `notes/experiments/circles_biome.py` with `STAGE=3` (results
`images/cbio_stage3.json`, log `images/cbio_stage3_log.txt`), 88 s single
core.  Renders `images/cbio_base_oracle.png`, `cbio_base_p0.png` (theta0, p
= 0), `cbio_base_p2.png` (theta0, p = 2), `cbio_base_dormant.png` (dormancy
dial: oracle left, forward `+bu+ou` right; none blocks hatched).

Config: `CirclesBiome(6, 6)` (12 x 12 slots, 96 x 96 tiles), default dials
(mu 0.3, `b` = 24.94 / 15.33 = `F_fam - log 16` with `F_disc` 27.71,
`F_bar` 18.10, `bio_u0` = 0 / 2.77 / 2.77 / 4.39).  Oracle 50 + 400 sweeps
(3.5 s; `symmetrise` is the identity for the default families).  Forward
`Forward(use_sampler=True)`: biome Sampler, obj Sampler with `hb = BT`
(init: dormancy from the hard `mask` row), `S_M` sweeps at cap `K`, then `p`
sweeps of `relax` under the same channels with the mask at `lam = 3`
(`soft_mask_factor`; support stays hard, dormant slots stay fixed), tile
Sampler.  `S_T = S_M = 30`, `S_F = 20`, designed + support.  RUNS 64, eval
noise = L1 between two independent 64-run evaluations (seeds 555 / 556;
the K = None and K = 8 rows share seeds, so their biome samples are
identical).  Three settings of the learned tables, each at (p, K) = (0,
None), (0, 8), (2, None):

- `theta0`: as specified, no finite part at all.
- `+bu`: plus the zeroth-order top unary `reference(centred=False)["bio_u"]`
  (= `-bio_u0` exactly: the biome prior becomes uniform).
- `+bu+ou`: plus the induced mid unary `obj_u = F(o) - F(absent)`
  (interior closed form, `reference()["obj_u"] - pres_e`: an isolated
  allowed slot 50/50).

The two extra settings were added because the designed dials are offsets
of induced unaries (`pres` cancels `F_fam`, `bio_u0` cancels the
zeroth-order slot entropy), so with theta0 the unaries dominate everything
else; `+bu+ou` is the baseline the stage was meant to measure (the pair
part of `Phi` missing, nothing else).

New monitors: `pd_allowed` / `pb_allowed` = present per family among the
slots whose biome admits it ("in isolation", computed in the script);
`mask_viol` = slots whose family the biome forbids (non-zero only under
relaxation; added to `CirclesBiome.stats`).

**Monitors, default dial** (`edge_air_top` held out):

| | present_disc | present_bar | pd_allowed | pb_allowed | conflict | contact | bio_hist (none/discs/bars/both) | dormant | mask_viol | edge_air_top |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.671 | 0.158 | 0.729 | 0.349 | 0 | 0.344 | .011/.534/.069/.386 | 0.011 | 0 | 0.541 |
| theta0 p0 K=None | 0.069 | 0.054 | 1.000 | 0.815 | 0 | 0.214 | .877/.056/.054/.012 | 0.877 | 0 | 0.440 |
| theta0 p0 K=8 | 0.069 | 0.054 | 1.000 | 0.815 | 0 | 0.209 | .877/.056/.054/.012 | 0.877 | 0 | 0.439 |
| theta0 p2 K=None | 0.129 | 0.000 | 1.000 | 0.003 | 0 | 0.344 | .871/.060/.055/.014 | 0.871 | 0.055 | 0.450 |
| +bu p0 K=None | 0.474 | 0.258 | 1.000 | 0.527 | 0 | 0.259 | .268/.242/.258/.232 | 0.268 | 0 | 0.524 |
| +bu p0 K=8 | 0.474 | 0.258 | 1.000 | 0.527 | 0 | 0.247 | .268/.242/.258/.232 | 0.268 | 0 | 0.528 |
| +bu p2 K=None | 0.760 | 0.001 | 1.000 | 0.001 | 0 | 0.351 | .239/.246/.265/.250 | 0.239 | 0.265 | 0.548 |
| +bu+ou p0 K=None | 0.187 | 0.214 | 0.396 | 0.437 | 0 | 0.207 | .268/.242/.258/.232 | 0.268 | 0 | 0.477 |
| +bu+ou p0 K=8 | 0.188 | 0.213 | 0.395 | 0.435 | 0 | 0.189 | .268/.242/.258/.232 | 0.268 | 0 | 0.478 |
| +bu+ou p2 K=None | 0.199 | 0.220 | 0.388 | 0.415 | 0 | 0.201 | .239/.246/.265/.250 | 0.239 | 0.013 | 0.479 |

Monitor eval noise (|difference| of the two evaluations) is 0.000-0.018
on present, 0.001-0.04 on contact, 0.02-0.04 (L1) on bio_hist, < 0.004 on
edge_air_top.  A single |difference| understates the spread: a 256-run
check of `+bu+ou` gives contact 0.202 +- 0.003 (K None) and 0.203 +- 0.004
(K 8) at S_M 30, 0.198 / 0.205 at S_M 100 (so the 0.207 / 0.189 pair above
is a 2-sigma draw, and 30 sweeps are mixed).

**L1 to the oracle on the fitted features, default dial:**

| | obj_h | obj_v | obj_d1 | obj_d2 | obj_u | bio_h | bio_v | bio_u |
|---|---|---|---|---|---|---|---|---|
| theta0 p0 K=None | 1.611 | 1.601 | 1.575 | 1.579 | 1.413 | 1.919 | 1.924 | 1.732 |
| theta0 p0 K=8 | 1.608 | 1.600 | 1.574 | 1.573 | 1.413 | 1.919 | 1.924 | 1.732 |
| theta0 p2 K=None | 1.558 | 1.573 | 1.530 | 1.531 | 1.401 | 1.917 | 1.915 | 1.720 |
| +bu p0 K=None | 0.881 | 0.867 | 0.710 | 0.726 | 0.523 | 1.247 | 1.184 | 0.892 |
| +bu p0 K=8 | 0.892 | 0.887 | 0.729 | 0.733 | 0.533 | 1.247 | 1.184 | 0.892 |
| +bu p2 K=None | 0.968 | 0.987 | 0.825 | 0.818 | 0.518 | 1.211 | 1.146 | 0.848 |
| +bu+ou p0 K=None | 1.213 | 1.206 | 1.170 | 1.179 | 0.997 | 1.247 | 1.184 | 0.892 |
| +bu+ou p0 K=8 | 1.229 | 1.200 | 1.174 | 1.177 | 0.999 | 1.247 | 1.184 | 0.892 |
| +bu+ou p2 K=None | 1.189 | 1.179 | 1.139 | 1.143 | 0.971 | 1.211 | 1.146 | 0.848 |
| eval noise theta0 p0 K=None | 0.109 | 0.121 | 0.112 | 0.111 | 0.033 | 0.025 | 0.040 | 0.019 |
| eval noise theta0 p0 K=8 | 0.105 | 0.113 | 0.110 | 0.098 | 0.029 | 0.025 | 0.040 | 0.019 |
| eval noise theta0 p2 K=None | 0.091 | 0.088 | 0.091 | 0.095 | 0.030 | 0.060 | 0.060 | 0.029 |
| eval noise +bu p0 K=None | 0.318 | 0.333 | 0.354 | 0.378 | 0.063 | 0.095 | 0.080 | 0.037 |
| eval noise +bu p0 K=8 | 0.345 | 0.349 | 0.367 | 0.360 | 0.084 | 0.095 | 0.080 | 0.037 |
| eval noise +bu p2 K=None | 0.203 | 0.198 | 0.217 | 0.210 | 0.053 | 0.110 | 0.080 | 0.034 |
| eval noise +bu+ou p0 K=None | 0.231 | 0.233 | 0.250 | 0.235 | 0.047 | 0.095 | 0.080 | 0.037 |
| eval noise +bu+ou p0 K=8 | 0.246 | 0.245 | 0.252 | 0.245 | 0.059 | 0.095 | 0.080 | 0.037 |
| eval noise +bu+ou p2 K=None | 0.243 | 0.238 | 0.265 | 0.253 | 0.053 | 0.110 | 0.080 | 0.034 |

**Timings** (ms per forward run, mean of 64; obj = the `S_M` capped sweeps,
init = `Sampler.init`; us = obj sweep time per active slot per sweep):

| | wall | biome | obj init | obj | relax | tile | us / active site |
|---|---|---|---|---|---|---|---|
| theta0 p0 K=None | 39.4 | 0.53 | 0.10 | 1.00 | 0 | 36.9 | 1.89 |
| theta0 p0 K=8 | 41.3 | 0.57 | 0.10 | 0.79 | 0 | 38.8 | 1.49 |
| theta0 p2 K=None | 41.5 | 0.58 | 0.12 | 1.12 | 0.25 | 38.5 | 2.02 |
| +bu p0 K=None | 46.4 | 0.70 | 0.11 | 6.38 | 0 | 37.9 | 2.02 |
| +bu p0 K=8 | 37.4 | 0.51 | 0.08 | 3.20 | 0 | 32.9 | 1.01 |
| +bu+ou p0 K=None | 39.5 | 0.50 | 0.08 | 4.91 | 0 | 33.3 | 1.55 |
| +bu+ou p0 K=8 | 40.3 | 0.55 | 0.10 | 3.33 | 0 | 35.3 | 1.05 |
| +bu+ou p2 K=None | 43.8 | 0.55 | 0.11 | 5.49 | 0.60 | 36.1 | 1.67 |

The tile level is 85-95% of a forward run (9216 tiles x 20 sweeps, ~0.19
us per tile update).  Per obj site, the cap K = 8 costs 1.0-1.5 us against
1.6-2.0 us at K = None (D = 33): about 1.6-1.9x, not 33 / 8, presumably
because the hard rows are still read for every value to find the
admissible set the cap draws from.

**Dormancy.**  Dial: `bio_u0 + 2.5` nats on every non-none value (the
suggested setting; the oracle's none fraction is steep in this dial and
its biome autocorrelation long there: 0.11-0.13 at +2.0, 0.21-0.62 at +2.5
over 400-sweep chains).  Oracle 100 + 800 sweeps: bio_hist .238 / .399 /
.070 / .293.  Forward at p = 0:

| | dormant | present_disc | present_bar | pd_allowed | pb_allowed | conflict | contact | obj ms | us / active site |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.238 | 0.493 | 0.132 | 0.712 | 0.366 | 0 | 0.343 | | |
| theta0 K=None | 0.989 | 0.004 | 0.007 | 1.000 | 0.929 | 0 | 0.067 | 0.24 | 5.1 |
| theta0 K=8 | 0.989 | 0.004 | 0.007 | 1.000 | 0.929 | 0 | 0.039 | 0.24 | 5.1 |
| +bu K=None | 0.790 | 0.144 | 0.066 | 1.000 | 0.477 | 0 | 0.254 | 1.63 | 1.80 |
| +bu K=8 | 0.790 | 0.144 | 0.066 | 1.000 | 0.477 | 0 | 0.262 | 1.05 | 1.16 |
| +bu+ou K=None | 0.790 | 0.057 | 0.060 | 0.391 | 0.432 | 0 | 0.187 | 1.63 | 1.80 |
| +bu+ou K=8 | 0.790 | 0.057 | 0.054 | 0.408 | 0.396 | 0 | 0.204 | 1.02 | 1.13 |

(L1 on the fitted features: theta0 1.49-1.52 obj pairs / 1.79 bio_h, +bu
0.96-1.03 / 1.21, +bu+ou 1.17-1.20 / 1.21, eval noise 0.02-0.16; the K = 8
rows within 0.01 of K = None.)  The theta0 rows' 5 us per active site is
the fixed per-sweep cost spread over 1% active slots, not a per-site cost.
The clean measurement is a probe with the biome set by hand (non-none
values uniform, theta0, 20 repeats of init + 30 obj sweeps):

| K | none fraction | active | init ms | 30 sweeps ms | us / active site |
|---|---|---|---|---|---|
| None | 0 | 1.00 | 0.12 | 8.41 | 1.95 |
| None | 0.25 | 0.75 | 0.12 | 6.05 | 1.87 |
| None | 0.5 | 0.50 | 0.08 | 3.75 | 1.74 |
| None | 0.75 | 0.25 | 0.08 | 1.91 | 1.76 |
| None | 1 | 0 | 0.07 | 0.15 | |
| 8 | 0 | 1.00 | 0.07 | 4.32 | 1.00 |
| 8 | 0.25 | 0.75 | 0.08 | 3.33 | 1.03 |
| 8 | 0.5 | 0.50 | 0.07 | 2.23 | 1.03 |
| 8 | 0.75 | 0.25 | 0.07 | 1.17 | 1.08 |
| 8 | 1 | 0 | 0.07 | 0.14 | |

The obj time is linear in the active fraction; a fully dormant world costs
0.14 ms for 30 sweeps (the block loop and seeding, 2-3% of a fully active
one) plus a 0.07 ms init.  No conflict in any run; dormant slots stay at 0
through the sweeps and the relaxation (`tests/test_circles_biome.py`).

### What the numbers show (stage 3)

(a) *The support works, and is the only thing that is right.*  Conflict
is 0 in every run of every setting, relaxed or not, capped or not; the
dormant slots stay absent.  Everything else is off by far more than eval
noise, at both levels.

(b) *What the finite part must carry.*  The default dials put two large
unaries into the induced part, so with theta0 the forward is dominated by
them, not by the missing pairs:
- the mid unary `F_fam` (27.7 / 18.1 nats), against which `pres` (-24.9 /
  -15.3) was set: without it every admitted slot is filled (pd_allowed 1.00
  against 0.73, pb_allowed 0.81 against 0.35: bars stop only at the support
  and at the larger disc bonus in "both" blocks).  "present right in isolation" holds only once `obj_u =
  F(o)` is in (`+bu+ou`: 0.40 / 0.44, under the isolated 0.5 because the
  support crowds neighbours).
- the top unary: theta0 gives none 0.88, not uniform (`bio_u0` 2.8-4.4
  nats with nothing cancelling it).  With the zeroth-order unary the
  histogram is uniform (.27 / .24 / .26 / .23) against the oracle's .011 /
  .534 / .069 / .386: L1 0.89 on bio_u, 1.2 on bio_h / bio_v (noise
  0.04-0.1) is what the beyond-zeroth-order top terms (shared-ring
  attraction across block edges, mask x support) must carry.
- with both unaries in, the mid residual is the shared-ring attraction:
  disc occupancy 0.40 against 0.73 (+0.33 to carry), bars 0.44 against 0.35
  (-0.09, presumably crowded out by discs in "both" blocks), contact 0.20
  against 0.34, edge_air_top 0.477 against 0.541; obj pair L1 1.2 against
  noise 0.25.  The `+bu` setting is closer in obj L1 (0.71-0.88) only
  because "fill every slot" happens to be nearer the oracle's dense disc
  packing than 50/50 is.

(c) *Post-relaxation changes little, as expected, once the unaries are
right.*  `+bu+ou` at p = 2: every monitor moves by <= 0.02, L1s within
eval noise, mask_viol 0.013.  Tiles are independent given the objects, so
there is nothing local for the relaxed obj sweeps to repair.  Without the
mid unary it is destructive: the soft mask (lam 3) is outweighed by the
9.6-nat difference between `b_disc` and `b_bar`, so bar-only blocks fill
with discs (pb_allowed 0.81 -> 0.003, mask_viol 0.05 at theta0, 0.26 at
+bu).  Relaxation is only as good as the unaries under it: `lam` must
exceed the largest designed-unary gap the finite part is meant to cancel,
or the relaxation must run after the unaries are learned.

(d) *Cap invariance.*  K = 8 against K = None: the fitted-feature L1s
agree within 0.02 (noise 0.1-0.35) in every setting, present and
pd/pb_allowed within 0.002-0.04, contact within 0.02 (2 sigma at 64 runs;
a 256-run check gives 0.202 +- 0.003 against 0.203 +- 0.004).  The biome
level is identical by construction (shared seeds, the cap is on obj only).

(e) *Dormancy costs nothing.*  The obj time is linear in the active
fraction, a fully dormant world costs 2-3% of a fully active one for its
block loop, and `init` is 0.07 ms.  The obj level is 1-15% of a forward
run; tiles dominate.

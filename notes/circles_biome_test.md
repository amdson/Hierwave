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

### Stages 4-5: the bootstrap fit

Script `notes/experiments/circles_biome_fit.py` (callbacks and
`materialise` in `castlegen/channels/biome_fit.py`, test
`tests/test_biome_fit.py`), outputs `images/cbio_stage45.json`,
`images/cbio_stage45_heldout.json` (`HELDOUT=1`), `images/cbio_fit_tables.png`,
renders `images/cbio_fit_oracle.png`, `images/cbio_fit_top_<targets>_K<K>.png`,
log `images/cbio_stage45_log.txt`.  Wall time 1453 s + 90 s (holdout run).

Config: `nty = ntx = 6`, default dials, forward `S_T = S_M = 30`, `S_F =
20` on the old (non-Sampler) path, `train.fit` with `iters = 5`,
`n_contexts = 8`, `l2 = 1e-4`, holdout 0.1, 8 consistency probes per
context; AIS `K = 32`, `M = 16`, `L = 30`, linear schedule.  Mid: stamp
features (68: 4 unary + 4 x 16 OFF8 pair counts over `dem`), designed
offset from the designed + support model (mask, pres, `INF` on the
support: `fit` drops the inadmissible values before calling the
targets), K-step = draw from `pi`, `dem` repainted, tiles whose demand
changed redrawn.  Tables by `materialise`: single objects and pairs
painted on a 5 x 5 block canvas (`u[0]`, `g_d[0, .]` exactly 0); the
test checks it reproduces `PAIR` / `obj_u` to 1e-9 from
`theta = (fz on free / dirt / air, 0 elsewhere)` (the true psi is a
unary over the paint, so the feature set contains it exactly).

Mid targets: `exact` = `Oracle.mid_probs`; `ais` = `AISTargets("tile",
block + spill border)` with the default split: honour reads `dem`, so it
is in `p_0`, nothing is annealed and AIS is exact (max L1 to `mid_probs`
6e-14 over 20 sites); `aisH` = the same with `anneal_names=["honour"]`
(`INF -> L`), the estimated-targets column, run on 4 contexts with 1
probe each because every probe costs `2 D` window runs (an AIS run is
10 ms here against 0.057 ms for the exact split).

#### Stage 4, the mid level

Double-centred over the finite entries against `reference()`.
"Held-out" = present pairs `(t, t')` that never occurred as (candidate,
neighbour value) in a data record (probes excluded); `n` per offset h /
v / d1 / d2, of 1002 / 1014 / 1024 / 1024 finite present pairs; in
brackets those with a nonzero interaction.

| setting | records | held-out n | held-out max err | all max err | obj_u err | slope / corr h, v | viol targets / fit | KL held | s | tau obj\|biome, tile\|obj |
|---|---|---|---|---|---|---|---|---|---|---|
| exact K0 | 692 | 52/26/62/60 (2/2/0/0) | **0.0005** | 0.0010 | 0.0003 | 1.000/1.000, 1.000/1.000 | 7e-15 / 2e-14 | 7e-6 | 27 | 1.20, 0.91 |
| exact K3 | 2768 | 2/0/0/0 | 0.0001 | 0.0009 | 0.0003 | 1.000/1.000, 1.000/1.000 | 8e-15 / 3e-14 | 2e-6 | 41 | 1.23, 1.39 |
| ais K0 | 692 | 52/26/62/60 | 0.0005 | 0.0010 | 0.0003 | 1.000/1.000, 1.000/1.000 | 9e-14 / 3e-14 | 7e-6 | 44 | 1.20, 0.91 |
| ais K3 | 2768 | 2/0/0/0 | 0.0001 | 0.0009 | 0.0003 | 1.000/1.000, 1.000/1.000 | 9e-14 / 3e-14 | 2e-6 | 64 | 1.23, 1.39 |
| aisH K0 (4 ctx) | 340 | 130/179/203/317 | 0.86 | 0.97 | 0.80 | 0.81/0.976, 0.67/0.81 | 21 / 2e-14 | 0.62 | 315 | 1.09, 1.05 |
| aisH K3 (4 ctx) | 1360 | 10/6/30/47 | 0.50 | 1.06 | 0.90 | 0.69/0.985, 0.67/0.987 | 27 / 1e-14 | 0.67 | 509 | 1.23, 1.02 |

Gate (exact, K = 0, held-out within 0.05): **passed**, 0.0005.  Named
shared-ring entries of `obj_h` (double-centred; the two most attractive
conflict-free pairs, raw -3.42 each): disc 4 (centre (2, 5)) left of
disc 9 (centre (4, 2)): reference -2.243, exact / ais -2.243, aisH K0
-2.08, aisH K3 -1.25; bar 20 left of bar 17: -2.447, -2.446 / -2.447,
-2.06, -1.76.

AIS with honour annealed (20 sites, every candidate): L1 to `mid_probs`
median 0.89, max 1.54; `(log Z - exact) / se` median -4.6, 35% within 3
se.  The delta-method se saturates at 1 (one chain dominates the
weights: se p90 0.98, max 1.00), so it is not an error bar here; the
estimate is biased low by several nats, differently per candidate.

The held-out count above is small because the forward contexts cover
almost every pair.  89% of the slots are dormant at stage 4, since
`bio_u0` without the learned top unary puts 88% of the biomes at none,
but the remaining slots still see nearly all values next to each other.
The strict test (`HELDOUT=1`) uses exact targets at K = 0, with every
biome forced to admit one family, so the other family's 16 values never
occur:

| contexts admit | pairs with an unseen value: n (interacting), max err h / v / d1 / d2 | unseen-unseen max err h / v | obj_u max err |
|---|---|---|---|
| discs only | 756 (65) / 768 (16) / 768 / 768: 0.0003 / 0.0002 / 0.0001 / 0.00005 | 0.0003 / 0.00004 | 0.032 |
| bars only | 746 (83) / 758 (50) / 768 / 768: 0.0008 / 0.0015 / 0.0007 / 0.0004 | 0.0008 / 0.0014 | 0.040 |

#### Stage 5, the top level

Mid bias installed: the `exact K0` stamp theta materialised (`obj_*`),
the same for every top setting.  Tabular `bio_h`, `bio_v`, `bio_u`,
designed offset `bio_u0`.  Targets:

- `exact` = `Oracle.top_probs`, i.e. `p*(T | its four slots)`, *not*
  collapsed over the slots.
- `ais` = `AISTargets("obj", the cell's 2 x 2 slots)`, with
  `after_change` = paint allow + dormancy (slots of a none cell set to
  absent and fixed).  The annealed rows are `obj_h/v/d1/d2` and `sup_*`
  (same-level, `INF -> L = 30`): a support conflict costs 30 nats
  instead of being forbidden, a weight of e^-30 per violation, negligible
  at the end of the anneal.  `p_0` = pres + obj_u + mask (inf kept).  The
  neighbouring slots are a fixed boundary, and their rows to the window
  are kept.
- `hook` = the same window with log Z by enumeration over the admissible
  values under the installed tables (`biome_fit.Bench.top_hook`, checked
  against brute force through `Model.energy`).  This is the exact
  collapsed target, added as a third column.

K-step: paint allow, dormancy, `Oracle.mid_move` on the four slots.
Oracle moments: 50 + 400 sweeps (group = identity).

L1 to the oracle moments, 64 forward runs with everything installed
(`edge_air_top` held out):

| setting | records | KL held | viol targets | s | obj_h | obj_u | bio_h | bio_v | bio_u | present_disc | present_bar | contact | dormant | edge_air_top | conflict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact K0 | 1440 | 0.46 | 1e-16 | 12 | 1.385 | 1.221 | 1.793 | 1.768 | 1.470 | 0.554 | 0.056 | 0.100 | 0.684 | 0.093 | 0 |
| exact K3 | 5760 | 0.83 | 1e-16 | 13 | 0.986 | 0.879 | 1.384 | 1.373 | 1.049 | 0.406 | 0.029 | 0.092 | 0.460 | 0.073 | 0 |
| ais K0 | 1440 | 0.028 | 1.4 | 65 | 0.386 | 0.240 | 0.222 | 0.194 | 0.143 | 0.104 | 0.026 | 0.054 | 0.032 | 0.016 | 0 |
| ais K3 | 5760 | 0.033 | 1.3 | 131 | 0.386 | 0.237 | 0.216 | 0.186 | 0.137 | 0.100 | 0.023 | 0.054 | 0.030 | 0.015 | 0 |
| hook K0 | 1440 | 0.021 | 4e-16 | 51 | 0.396 | 0.244 | 0.236 | 0.212 | 0.153 | 0.107 | 0.029 | 0.054 | 0.034 | 0.016 | 0 |
| hook K3 | 5760 | 0.028 | 7e-16 | 135 | 0.377 | 0.231 | 0.205 | 0.170 | 0.129 | 0.097 | 0.024 | 0.050 | 0.030 | 0.014 | 0 |
| mid only (no top term) | | | | | 1.656 | 1.488 | 1.906 | 1.917 | 1.731 | 0.616 | 0.127 | 0.152 | 0.866 | 0.106 | 0 |
| eval noise | | | | | 0.291 | 0.053 | 0.064 | 0.071 | 0.036 | 0.013 | 0.002 | 0.003 | 0.007 | 0.002 | 0 |

(obj_v / d1 / d2 track obj_h within 0.02 in every row.)  The fit's
consistency violation is 1e-15 to 3e-14 everywhere.  Both the tabular
top potential and the mid stamp potential are joint potentials (form
(i)), so they are closed by construction.

Biome histogram (none / discs / bars / both):

| source | none | discs | bars | both |
|---|---|---|---|---|
| oracle | 0.014 | 0.531 | 0.074 | 0.381 |
| hook K3 | 0.044 | 0.469 | 0.108 | 0.379 |
| ais K0 | 0.046 | 0.462 | 0.114 | 0.378 |
| exact K0 | 0.698 | 0.094 | 0.125 | 0.083 |
| exact K3 | 0.474 | 0.230 | 0.138 | 0.158 |

Learned `bio_u` in the gauge `bio_u[none] = 0`:

| setting | none | discs | bars | both |
|---|---|---|---|---|
| zeroth order | 0 | -2.77 | -2.77 | -4.39 |
| exact K0 | 0 | -0.94 | -1.74 | -2.62 |
| exact K3 | 0 | -2.39 | -2.15 | -3.88 |
| ais K0 | 0 | -3.73 | -3.02 | -5.24 |
| ais K3 | 0 | -3.66 | -2.95 | -5.18 |
| hook K0 | 0 | -3.74 | -2.97 | -5.24 |
| hook K3 | 0 | -3.78 | -3.00 | -5.27 |

The collapsed fits sit about 1 nat below zeroth order on discs and both,
and 0.2-0.9 nats below on bars.  That is the shared-ring attraction and
the cheaper edge objects, which make `p*` prefer discs.  Learned `bio_h`
/ `bio_v` double-centred are small (|entries| <= 0.19).  The oracle's
`-log(p(T,T') / p(T) p(T'))` pattern reaches 0.6 (max difference 0.62
for the collapsed fits, 0.71-0.97 for exact).  That pattern is a
marginal-correlation statistic of a strongly unary-driven chain, not the
coupling, so the comparison is qualitative only.

`Phi_top(none)`: the AIS `log Z` of the none candidate is exactly 0.0
(se 0) at all 36 cells of a fresh context in every setting, and so is
the hook's.  There is one configuration (dormancy fixes the four slots
at absent), and materialisation gives `u[0] = 0` and `g_d[0, .] = 0`, so
`F = 0` with no learned term.  AIS against the hook at the other three
candidates: |delta log Z| max 0.23-0.47, se median 0.085-0.11, 99-100%
within 3 se, at 3.3 ms per run.  `tau(obj | biome)` at the final theta
is 1.2-1.6 sweeps.

The residual on the collapsed rows is the world boundary.  Presence on
edge / interior slots:

| run | edge | interior |
|---|---|---|
| oracle | 0.90 | 0.79 |
| forward (hook K3) | 0.72 | 0.78 |
| forward obj, oracle's biome clamped | 0.77 | 0.78 |
| oracle obj, same biomes | 0.91 | 0.77 |

An object at the world edge loses its off-grid ring tiles (~0.85 nat
each).  `stamp_features` sees this, because its paint is clipped, but
the position-free materialised tables cannot carry it.  That accounts
for the whole of the `obj_u` and `present_disc` gap and most of
`bio_u`'s.

**What the numbers show.**

1. *Stamp features generalise to unseen pairs, exactly.*  The true
   induced potential is a unary over the `dem` paint, which the feature
   set contains.  So 692 records give every table to 1e-3, and a fit
   that never saw a bar (or a disc) recovers every pair involving it to
   1.5e-3.  Q1 stays with derived features.
2. *The cost of estimated targets.*  On this model the default
   `AISTargets` is exact at 0.06 ms per window, because honour is a
   parent row and the tiles are independent given `dem`.  Forcing honour
   into the anneal makes AIS 175 times dearer and badly wrong at K = 32,
   M = 16.  It is biased several nats low, the targets are inconsistent
   by 20 nats, and the tables are off by 0.5-1 with the attraction
   shrunk (slope 0.7-0.8).  The reported se saturates and does not flag
   this.  The lesson is the split, not the AIS budget: anything that
   reads only parents belongs in `p_0`.  At the top, where AIS does real
   work (obj pairs and support annealed), it matches the exact collapsed
   hook within its se, and the fits and end-to-end monitors of `ais` and
   `hook` agree within eval noise.
3. *What K buys.*  Nothing at the mid level: the fit is already exact at
   K = 0, and K = 3 is 4x the records and covers every pair.  At the top
   it depends on the targets:
   - With `top_probs`, the non-collapsed `p*(T | slots)`, the bootstrap
     is the S1 leak in its purest form.  Contexts from `q` are 88% none
     (all slots absent), and their target is `softmax(-bio_u0)`, which
     favours none again, so the fit learns only a third of the unary.
     K = 3 moves the contexts and halves the gap (dormant 0.68 -> 0.46,
     bio_u L1 1.47 -> 1.05) but stays far off.
   - With the collapsed targets (`ais`, `hook`), K changes nothing
     measurable, because the target no longer depends on where the slots
     are.

   So the top target must be collapsed over the level below (AIS or a
   hook), which is what C2 says.  `ExactTargets(top_probs)` is the wrong
   object for a sampler that draws `T` before its slots.
4. `Phi_top(none) = 0` exactly, from every setting.
5. *End to end.*  `conflict` is 0 everywhere.  The collapsed settings
   take the none share of the biome histogram from 0.87 (mid only) to
   0.04, against the oracle's 0.01, and cut every L1 by 4-12x.  But the
   gate (within eval noise on the fitted features) **fails**: bio_u
   0.13-0.15 against noise 0.036, obj_u 0.23-0.24 against 0.053, and
   present_disc 0.10 against 0.013, with obj_h at noise.  The cause is
   the world-boundary discount above, not the top fit.  The fix is a
   boundary-aware install (an edge-class `obj_u`, or the batched-delta
   stamp potential at edge slots), not tuning; the dials were not
   touched.

Deviations:

- `aisH` ran on 4 contexts with 1 probe each, because of the cost.
- `fit_windows` was not run.
- The top AIS window is the four written slots, not dilated.  The
  neighbouring slots enter as a fixed boundary through the annealed
  rows' window, so `log Z(none) = 0` holds in absolute value.
- The `hook` column is an addition.
- Every top setting uses the `exact K0` mid tables.

### Stage 7: scaling probe

Script `notes/experiments/circles_biome_scale.py` (results
`images/cbio_stage7.json`, log `images/cbio_stage7_log.txt`), 47 s single
core for the whole stage including the optional fit (two other agents were
running on the machine, so absolute times carry some load noise; the cost
probe interleaves its configurations to share it).  Figure
`images/cbio_scaling.png` (per-site time and parameter count against N, log
axes); render `images/cbio_scale20.png` (N = 20: oracle left, forward
`+bu+ou` K = 8 right; none blocks hatched).

Config: `CirclesBiome(6, 6, families=random_families(N, rng),
masks=random_pair_masks(N, rng))`, N in 2, 5, 10, 20, default dials (mu
0.3, `b_fam = F_fam - log 16` per family, `bio_u0 = 4 log(1 + #admitted)` =
0 / 4.39 x 7).

- `random_families` (new, additive): each family is a 4-connected set of
  6-14 cells grown from a seed by random frontier additions, its bounding
  box kept within 5 x 5 (so at least 16 in-block anchors exist), shapes
  distinct up to translation; anchors = every in-block position, 16 drawn at
  random when there are more (always more here).  Rings and presence
  bonuses follow from the footprint as for discs and bars.
- `masks=` (new constructor argument): the biome's values are an explicit
  list of admitted-family bitmasks (`C.MASKS`); default `arange(2^N)`, so
  every existing test is untouched.  `FAMOK`, `bio_u0`, the `allow_<name>`
  views, `_group`'s biome map (an element is kept only if the mask list is
  closed under its family permutation; identity always) and `render`'s
  dormant hatching (`MASKS[biome] == 0`) read `MASKS`; `paint_allow`,
  `dormant_of`, `stats` (bio_hist over `P = len(masks)` values, mask_viol),
  `reference()["bio_u"]`, `top_probs` already went through `FAMOK` / `P`.
- `random_pair_masks(N, rng)`: `[0] + 7` pairs, consecutive pairs of fresh
  random permutations, distinct while distinct pairs remain; D_biome = 8 at
  every N.  N = 2 has one pair, repeated 7 times (so the biome is none vs
  both at prior odds 1 : 7); at N = 20 the 7 pairs cover 14 of the 20
  families (a pair biome cannot cover 20 families with 7 values; the other
  6 are never admitted).
- Learned tables: `+bu+ou` of stage 3, i.e. `theta0` with `bio_u =
  reference(centred=False)["bio_u"]` and `obj_u = reference()["obj_u"] -
  pres` (installed in full: the pair reference is cheap, see build times);
  the four obj pair tables are installed as zero D x D tables as in stage 3
  (so the K = None soft rows include them).
- Oracle 20 + 100 sweeps per N; forward `Forward(use_sampler=True, hb =
  BT)`, `S_T = S_M = 30`, `S_F = 20`, 16 runs per K, K in None, 8.  Per-site
  time is stage 3's: obj stage time / (active slots x S_M).  Cost probe:
  every biome set to a random non-none value (active fraction 1), obj from
  empty, 30 obj sweeps, 40 reps (median), four samplers interleaved per rep:
  the full forward model and a hard-rows-only model (mask + support; `pres`
  and every learned table removed), each at K = None and K = 8.

**Per N** (forward times are means of 16 runs; probe us are medians of 40):

| N | D_obj | build ms (support + pair ref) | oracle conflict | families seen / admitted | fw us/site K=None | fw us/site K=8 | probe us/site K=None | probe us/site K=8 | obj ms/run None / 8 | wall ms/run None / 8 | dormant | conflict | present fw (None / 8) | present oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 33 | 3.0 (1.4) | 0 | 2 / 2 | 2.03 | 1.12 | 2.09 | 1.14 | 7.7 / 4.2 | 46.7 / 44.0 | 0.127 | 0 | 0.570 / 0.576 | 0.868 |
| 5 | 81 | 7.8 (6.0) | 0 | 5 / 5 | 3.46 | 1.65 | 2.88 | 1.62 | 13.1 / 6.2 | 61.0 / 47.5 | 0.127 | 0 | 0.573 / 0.561 | 0.909 |
| 10 | 161 | 15.0 (9.6) | 0 | 10 / 10 | 4.57 | 2.48 | 4.40 | 2.31 | 17.3 / 9.4 | 68.1 / 58.4 | 0.127 | 0 | 0.564 / 0.566 | 0.912 |
| 20 | 321 | 35.4 (25.3) | 0 | 14 / 14 | 6.67 | 3.50 | 5.92 | 3.00 | 25.1 / 13.2 | 85.1 / 72.8 | 0.127 | 0 | 0.554 / 0.585 | 0.882 |

Parameters and memory (the obj level):

| N | D_obj | stamp features (nfeat) | tabular 4 D^2 + D | support entries 4 D^2 | INF density | oracle s (120 sweeps) |
|---|---|---|---|---|---|---|
| 2 | 33 | 68 | 4 389 | 4 356 | 1.0% | 0.8 |
| 5 | 81 | 68 | 26 325 | 26 244 | 1.7% | 1.5 |
| 10 | 161 | 68 | 103 845 | 103 684 | 1.6% | 2.2 |
| 20 | 321 | 68 | 412 485 | 412 164 | 1.5% | 3.8 |

`nfeat` is `stamp_features(...).shape[1]` (OFF8 pairs over the 4-value
dem paint): 68 at every N, by construction.  The support (and the exact
pair reference, computed alongside it on the 3 x 3 block canvas) builds in
25 ms at D = 321, so the pair reference was affordable and installed; the
"single-object unary only" fallback was not needed.  The support tables
hold 4 D^2 = 412k entries at N = 20 (as float64 tables with their
reflections, 8 x 321^2 x 8 bytes = 6.6 MB if packed that way), 1.5% of them
INF.

**Hard-row cost** (probe, us per active site per sweep, medians):

| N | D | full K=None | full K=8 | hard only K=None | hard only K=8 | hard / full at K=8 | soft rows at K=8 (full - hard) | soft rows at K=None |
|---|---|---|---|---|---|---|---|---|
| 2 | 33 | 2.09 | 1.14 | 1.61 | 0.89 | 78% | 0.25 | 0.48 |
| 5 | 81 | 2.88 | 1.62 | 1.96 | 1.39 | 86% | 0.23 | 0.92 |
| 10 | 161 | 4.40 | 2.31 | 2.65 | 2.00 | 87% | 0.31 | 1.75 |
| 20 | 321 | 5.92 | 3.00 | 3.33 | 2.76 | 92% | 0.24 | 2.59 |

Packed obj rows: 19, of which 9 hard (the mask, a pair to `allow`, and the
support at 8 directed offsets; 5 hard at N = 2, whose two shapes never meet
diagonally so the diagonal support tables carry no INF and pack as soft) and
10 soft (`pres`, `obj_u`, the 4 pair tables x 2 directions).  The hard pass
runs every hard row over all D values before the cap draws (`_site_cap`:
`_rows(..., 0, nhard, cand, D, ...)`); the `skip` test makes a row cheap on
already-forbidden values but still visits them.  At K = 8 the soft rows on
the capped set cost a flat 0.23-0.31 us at every N; everything that grows
with N is the hard pass: full K=8 and hard-only K=8 grow with the same
slope, 6.5 ns per value of D ((3.00 - 1.14) / 288 and (2.76 - 0.89) / 288).
At K = None the soft rows add a second O(D) term (13.3 ns per value in
total).  So at D = 321 the hard pass over all D is 92% of a capped site
update and is the remaining O(D) cost.

What would remove it: the mask is a parent row, constant through the obj
sweeps, and with pair biomes it admits 1 + 2 x 16 = 33 of the 321 values
(10%) at every active slot, independent of N.  A per-block admissible list
computed once at `init` from the parent hard rows (C1's `A_p`; `Sampler.init`
already computes `admissible()` under exactly those rows, and throws it away
except for dormancy) would let the kernel run the same-level hard rows (the
support) and the soft rows only over `A_p`, making the hard pass O(|A_p|) =
O(16 x families per biome), constant in N.  The expected per-site cost is
then the N = 2 figure (~1.1 us at K = 8).  Storage is one list per biome
value (8 here), not per site, since `A_p` depends on the slot only through
its `allow`.

**Optional: the mid fit at N = 5** (`train.fit`, `ExactTargets(Oracle.
mid_probs)`, `stamp_features`, K = 0, 3 iterations, 4 contexts, via
`biome_fit.Bench` / `materialise`; 22 s): 172 records (52 / 120 / 172
cumulative), held-out KL 3.3e-7 / 3.8e-7 / 3.1e-7 by iteration (train
2.0e-6), 68 parameters.  The materialised pair tables (double-centred over
their finite entries, support INF from the stamps) against `reference()` on
the 15 316 finite value pairs (t, t' > 0) never seen as (candidate,
neighbour) in the data passes: max abs error 6.4e-4, mean 3.3e-5, slope
0.99995, corr 1 - 1.3e-8, against a reference range of 5.24 nats.  The
contexts are few because the Bench forward at theta0 leaves most top
blocks none (the designed `bio_u0` uncancelled), and still the fit
generalises to every unseen pair: the stamp features are exact for this
geometry, so they need only enough records to identify 68 numbers.

### What the numbers show (stage 7)

(a) *Per-site work at K = 8 is not yet constant in N.*  It grows from 1.1
to 3.0 us (probe) as D goes 33 -> 321, against 2.1 -> 5.9 us at K = None:
the cap halves the cost at every N, but both grow linearly in D.  The
probe decomposes it: the soft rows on the capped set are flat (0.24 us),
and the growth is entirely the hard pass, which visits all D values with
every hard row (92% of a capped update at N = 20).  The fix is the C1
admissible list from the parent rows at init, after which the per-site
work should be the N = 2 cost for any N with two families per biome.  This
is a kernel change (`_site_cap` over a per-site or per-allow candidate
list), not a model change; it is left for the Sampler owner.

(b) *The parameter count is flat with stamp features.*  68 features at
every N, against 4.4k -> 412k tabular entries (4 D^2 + D); adding a family
adds zero learned parameters.  At N = 5 the exact-target fit reproduces the
reference pair tables on 15k unseen pairs to 6e-4 nats from 172 records,
so the derived features generalise across 5 random shapes as they did for
disc and bar.  The support is derived from the stamps (25 ms at D = 321)
and costs memory quadratic in D (412k entries); it is not learned.

(c) *Dormancy does its job with sparse masks.*  Conflict is 0 in every
oracle sweep and every forward run at every N and K (mask violations 0
in the new test; not monitored in the script).
The dormant fraction is 0.127 at every N (the none share of a uniform
8-value biome, 1/8), and none blocks are skipped whole (`hb = BT`); `init`
costs 0.10-0.18 ms.  But with sparse masks dormancy covers only the none
blocks: the useful sparsity is inside the active blocks (90% of the values
masked out at N = 20), and today the kernel pays for those values on
every visit, which is (a).

(d) *The rest scales as expected.*  The forward run is 44 -> 73 ms at K = 8
(the obj stage 4 -> 13 ms, tiles ~37-55 ms whose work does not depend on N;
the spread follows machine load), the oracle 0.8 -> 3.8 s per 120 sweeps
(its collapsed move enumerates all D).  Present fractions (forward 0.55-0.59
against the oracle's 0.87-0.91) are off for the reasons of stage 3: the
pair part of `Phi` is missing and the oracle keeps almost no none blocks;
that is not what this stage measures.

Deviations: the masks cannot cover 20 families with 7 pairs (14 covered at
N = 20; "every family appears" is checked against the admitted ones); at
N = 2 the 7 pairs are one pair repeated; random stamps are kept within a 5
x 5 box so that every family has 16 offsets; the cost probe uses 40
interleaved reps (medians) rather than stage 3's 20 sequential ones, because
of load from the concurrent agents.

### Stage 8: migration

roots, ground and coord run through one protocol,
`sampler.ChannelSampler` (name, init, sweep(n, T, seed), energy,
candidates; relax, sweep_tempered and unary_logZ are optional).
`sampler.generate(levels, S, rng, painters)` runs the forward chain with
no per-model code.  `notes/experiments/migrate_combined.py`
(chan_combined3 as level 8 = [Sampler(u8)], level 1 =
[CoordSampler(joint)], with painters for the surface and the refinement)
reproduces `images/chan_combined3_sheet.png` pixel for pixel.

Interface changes: `S[l]` may be a list of (n, seed) chunks so that
scripts which reseed can be replayed; `Sampler` keeps an optional soft
model for `relax`; under a certificate, dormancy fixes only a mass-0
value, with d = INF, so it is a dead end to its neighbours, and a single
admitted value with mass stays active so its d is drawn.  The
certificate kernel already supported dormancy and inactive blocks at
K = None and needed no change.  Ground is a plain table set (no parent
rows, so init is a no-op).  For coordinate channels, `CoordSampler`
wraps `CoordKernel.sweep` / `sweep_joint` bit-identically and adds a
total `energy()`; they have no hard parent rows (the parent's refinement
is soft honour `mu` on `uref`), so `init` is a no-op.

What did not fit: the joint (u, t, d) kernel is one sampler owning three
channels, so "each channel owns its sampler" becomes "each sampled group
owns one"; the coordinate kernel has no colouring, no temperature in its
own mode, and no tempered kernel for AIS.

Remaining: the bit tile sampler (`bitgibbs.c`) behind the same protocol;
certificates with a candidate cap K (asserted off: the certificate draw
enumerates the whole domain); a tempered kernel and `unary_logZ` for
coordinate channels so AIS can run over them; dormancy for coordinate
channels if the parents' writes ever become hard.

Tests: `tests/test_migrate.py` (7).  The full suite has 8 pre-existing
failures unrelated to the channel code (`pipeline.py:217` NameError in
test_flow / test_promise / test_support; data files missing in
test_blockconn / test_blockfield).

# The biome circles test, stage 5b

Follows `notes/circles_biome_test.md` ("Stages 4-5: the bootstrap fit").

### Stage 5b: the boundary

The stage 5 end-to-end gate failed, and the failure was attributed to the
world boundary.  This stage tests whether the boundary is the whole story.

Script `notes/experiments/circles_biome_edge.py`, output
`images/cbio_stage5b.json`, log `images/cbio_stage5b_log.txt`, renders
`images/cbio_edge_oracle.png` and `images/cbio_edge_forward.png` (torus,
final setting), and `images/cbio_edge_forward_open.png` (open world, stage 5
potentials).  Wall time 177 s.

Config: `NT = 6`, forward `S = (30, 30, 20)` on the old path, 64 runs per
eval, oracle 50 + 400 sweeps, seeds as in stage 5.  The "all" row of part 1
reproduces the stage 5 `hook K0` row exactly (obj_h 0.396, obj_u 0.244,
bio_u 0.153, present_disc 0.107).

- **Eval noise** = L1 between two independent 64-run forward evals (seeds
  901 / 902), as in stage 5.
- **Oracle noise** = L1 between two independent 400-sweep oracle chains.

A forward-vs-oracle L1 carries both, so a fit with no bias sits at roughly
`sqrt((noise^2 + oracle_noise^2) / 2)`, i.e. 0.7-0.9 x eval noise.  The gate
is the stage 5 one: every fitted feature (`obj_h/v/d1/d2`, `obj_u`, `bio_h/v`,
`bio_u`) has L1 <= eval noise, and `conflict = 0`.

**Periodic world.**  The generic kernel cannot wrap: a pair row whose
neighbour is off the grid reads `pad` or vanishes (`kernel._energies`), and
`kernel.py` is not ours to change.  So `CirclesBiome(nty, ntx,
periodic=True)` (added to `circles_biome.py`) stores the torus padded with a
ghost ring one top cell wide (8 x 8 top cells for the 6 x 6 torus).

- **Ghost cells.**  Ghost cells are fixed and are exact copies of the
  opposite interior cells (`C.sync`).  Every interior site therefore reads
  its wrapped neighbours through the unchanged kernel, `stamp_features`,
  `_delta` and `Bench.top_hook`.
- **Forward.**  The forward sweeps one class of a torus-proper colouring at
  a time and syncs after each class: `(y + x) % 2` at the top, 2 x 2 at the
  mid level, one class for tiles.  This is exact Gibbs on the torus, not a
  lagged copy.
- **Oracle.**  The oracle writes a mid move to every copy of the slot,
  repaints and redraws around each copy, then syncs.
- **Paints and stats.**  `paint_dem` / `paint_allow` sync.  `stats` crops the
  torus and wraps every pair, ring and top-edge band.
- **Tests.**  The new tests check the wrapped paint against an independent
  modular painter, and `mid_probs` against direct torus painting.  They also
  check the tile energy identity on a periodic 2 x 2 world, that ghosts stay
  copies after oracle and forward runs, and that `stats_region(margin=0)` is
  `stats`.
- **Not supported.**  The Sampler path (`use_sampler`) is not wired for the
  torus and asserts.

#### 1. Interior-only evaluation (open world, stage 5 potentials)

The interior is defined per level:

- `int`: top cells, slots and tiles at least one block of their own level
  from the edge (4 x 4 top cells, 10 x 10 slots).
- `deep`: the slots and tiles inside the interior top cells (8 x 8 slots).

L1 to the oracle, with the eval noise below each row:

| region | obj_h | obj_v | obj_d1 | obj_d2 | obj_u | bio_h | bio_v | bio_u | present_disc | present_bar | dormant | contact | bio_hist | edge_air_top | conflict | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 0.396 | 0.390 | 0.391 | 0.391 | **0.244** | **0.236** | **0.212** | **0.153** | 0.107 | 0.029 | 0.034 | 0.054 | 0.153 | 0.016 | 0 | fail |
| noise | 0.291 | 0.304 | 0.345 | 0.333 | 0.053 | 0.064 | 0.071 | 0.036 | 0.013 | 0.002 | 0.007 | 0.003 | 0.036 | 0.002 | 0 | |
| oracle noise | 0.138 | 0.134 | 0.142 | 0.149 | 0.035 | 0.062 | 0.056 | 0.042 | 0.014 | 0.008 | 0.004 | 0.001 | 0.042 | 0.001 | 0 | |
| int | 0.317 | 0.316 | 0.342 | 0.340 | **0.089** | 0.108 | **0.126** | 0.035 | 0.036 | 0.013 | 0.025 | 0.008 | 0.035 | 0.010 | 0 | fail |
| noise | 0.369 | 0.368 | 0.417 | 0.395 | 0.080 | 0.161 | 0.083 | 0.057 | 0.017 | 0.002 | 0.007 | 0.005 | 0.057 | 0.003 | 0 | |
| oracle noise | 0.165 | 0.167 | 0.179 | 0.184 | 0.043 | 0.106 | 0.095 | 0.034 | 0.008 | 0.001 | 0.006 | 0.003 | 0.034 | 0.000 | 0 | |
| deep | 0.389 | 0.402 | 0.430 | 0.422 | 0.101 | 0.108 | **0.126** | 0.035 | 0.028 | 0.003 | 0.012 | 0.013 | 0.035 | 0.010 | 0 | fail |
| noise | 0.453 | 0.474 | 0.543 | 0.512 | 0.102 | 0.161 | 0.083 | 0.057 | 0.021 | 0.002 | 0.009 | 0.002 | 0.057 | 0.004 | 0 | |

Moments (oracle / forward):

| region | present_disc | present_bar | dormant | contact | edge_air_top |
|---|---|---|---|---|---|
| all | 0.665 / 0.558 | 0.159 / 0.188 | 0.014 / 0.048 | 0.342 / 0.396 | 0.539 / 0.523 |
| int | 0.615 / 0.579 | 0.172 / 0.185 | 0.018 / 0.043 | 0.385 / 0.377 | 0.536 / 0.526 |
| deep | 0.604 / 0.576 | 0.188 / 0.185 | 0.024 / 0.036 | 0.373 / 0.386 | 0.534 / 0.524 |

Biome histogram, none / discs / bars / both:

| | all | int |
|---|---|---|
| oracle | 0.014 / 0.531 / 0.074 / 0.381 | 0.024 / 0.484 / 0.108 / 0.383 |
| forward | 0.048 / 0.458 / 0.117 / 0.377 | 0.036 / 0.467 / 0.112 / 0.385 |

Presence on the world-edge ring of slots / elsewhere:

- oracle 0.912 / 0.787;
- forward 0.707 / 0.764.

The open world's interior is not the torus either.  The L1 of the
open-world `int` oracle to the torus oracle is:

- bio_u 0.052;
- present_disc 0.039;
- dormant 0.025;
- obj_u 0.080.

So in a 6 x 6 world the edge reaches the interior through the slot pairs and
the biome chain.

#### 2. Periodic world: refit and evaluate

**Mid fit (stage 4 exact K0 on the torus).**

- Size and time: 692 records, 16 s.
- Fit quality: consistency violation 5e-15 (targets) / 2e-14 (fit).
- Error to `reference()` (double-centred, finite present pairs): max 9e-5;
  obj_u 2e-5.
- Against the open-world exact-K0 tables: max difference 1e-3 on the
  admissible entries.

The mid potential does not see the boundary, as expected: the induced
potential is a unary over the paint, and the edge only removes paint.

**Top fit (stage 5 hook K0 on the torus).**

- Size and time: 1440 records, 56 s.
- Fit quality: KL held 0.023, consistency violation 9e-16.

Learned `bio_u`, gauge none = 0:

| world | none | discs | bars | both |
|---|---|---|---|---|
| zeroth order | 0 | -2.77 | -2.77 | -4.39 |
| open (stage 5) | 0 | -3.74 | -2.97 | -5.24 |
| torus | 0 | -2.46 | -1.82 | -3.19 |

The stage 5 fit's 1.0-1.3 nats of extra pull toward discs and both was the
cheaper edge objects.  On the torus the learned unary sits 0.3-1.2 nats
*above* zeroth order: the support between neighbouring slots costs
configurations.

**End to end on the torus**, L1 to the torus oracle:

| setting | obj_h | obj_v | obj_d1 | obj_d2 | obj_u | bio_h | bio_v | bio_u | present_disc | present_bar | dormant | contact | edge_air_top | conflict | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torus fits, S = 30 | 0.249 | 0.242 | 0.257 | 0.260 | 0.064 | 0.062 | 0.065 | 0.035 | 0.021 | 0.002 | 0.008 | 0.008 | 0.002 | 0 | **pass** |
| eval noise | 0.296 | 0.307 | 0.331 | 0.310 | 0.068 | 0.085 | 0.067 | 0.048 | 0.026 | 0.016 | 0.007 | 0.003 | 0.004 | 0 | |
| oracle noise | 0.128 | 0.133 | 0.132 | 0.134 | 0.033 | 0.061 | 0.052 | 0.026 | 0.008 | 0.003 | 0.002 | 0.001 | 0.006 | 0 | |
| torus fits, repeat eval | 0.235 | 0.242 | 0.249 | 0.243 | 0.057 | 0.070 | 0.069 | 0.040 | 0.009 | 0.000 | 0.004 | 0.006 | 0.003 | 0 | at noise (bio_v 0.069 / 0.067) |
| open-world thetas on the torus | 0.244 | 0.241 | 0.261 | 0.250 | 0.058 | **0.114** | **0.112** | **0.067** | 0.009 | 0.002 | 0.009 | 0.001 | 0.005 | 0 | fail |
| torus fits, S_T = S_M = 100 | 0.235 | 0.226 | 0.240 | 0.240 | 0.041 | 0.109 | 0.099 | 0.065 | 0.012 | 0.011 | 0.000 | 0.005 | 0.006 | 0 | (single eval, see below) |

Moments of the torus run:

- Biome histogram, oracle 0.043 / 0.473 / 0.116 / 0.369, forward 0.034 /
  0.475 / 0.106 / 0.385.
- present_disc 0.576 / 0.597.
- Presence on the torus's first and last rows and columns / elsewhere:
  oracle 0.759 / 0.767, forward 0.784 / 0.783.  The edge excess is gone.

#### 4. What remains, against noise

Single 64-run evals compared with a single noise draw are a coin flip at
this level.  So the final setting was evaluated four times per budget (4 x
64 runs) against a 2000-sweep oracle chain:

- `L1 pool` is the 256-run mean against the long oracle.
- `noise pool` is L1(runs 1-128, runs 129-256).  A bias-free `L1 pool`
  should be about half of it plus the long oracle's own noise.
- The long oracle against the 400-sweep oracle: obj_h 0.105, obj_u 0.034,
  bio_u 0.037.

| budget | quantity | obj_h | obj_v | obj_d1 | obj_d2 | obj_u | bio_h | bio_v | bio_u | present_disc | present_bar | dormant | contact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S = 30 | mean single-eval L1 | 0.217 | 0.221 | 0.232 | 0.230 | 0.041 | 0.053 | 0.052 | 0.030 | 0.006 | 0.003 | 0.002 | 0.002 |
| | L1 pool | 0.114 | 0.118 | 0.125 | 0.124 | 0.023 | 0.027 | 0.027 | 0.016 | 0.003 | 0.002 | 0.001 | 0.000 |
| | noise pool | 0.209 | 0.219 | 0.230 | 0.221 | 0.037 | 0.046 | 0.038 | 0.009 | 0.004 | 0.003 | 0.002 | 0.002 |
| S = 100 | mean single-eval L1 | 0.215 | 0.223 | 0.233 | 0.225 | 0.046 | 0.059 | 0.058 | 0.034 | 0.007 | 0.006 | 0.002 | 0.003 |
| | L1 pool | 0.116 | 0.118 | 0.118 | 0.119 | 0.027 | 0.042 | 0.045 | 0.029 | 0.003 | 0.003 | 0.002 | 0.003 |
| | noise pool | 0.215 | 0.221 | 0.233 | 0.217 | 0.038 | 0.063 | 0.066 | 0.049 | 0.013 | 0.013 | 0.002 | 0.005 |

**What the numbers show.**

1. **The boundary is the whole story.**
   - *Torus.*  With the boundary removed, the same pipeline passes the
     stage 5 gate: the stage 4 exact-K0 mid fit, then the stage 5 hook-K0
     top fit, then the forward.  Every fitted feature is within eval noise
     (bio_v at noise on a repeat eval) and conflict is 0.
   - *Monitors.*  Every monitor is at noise as well: present_disc 0.021
     against 0.026, dormant 0.008 against 0.007.  The open world showed
     0.107 and 0.034.
   - *What the torus removed.*  It removed the only thing that changed.
     Mid tables, targets, features, dials and sweep budget are the same as
     in stage 5, and the mid theta agrees with the open-world one to 1e-3.
2. **Why interior-only evaluation is not enough.**  On the open world,
   restricting to the interior cuts the residual 2-7x:
   - obj_u 0.244 -> 0.089, bio_u 0.153 -> 0.035, present_disc 0.107 ->
     0.036.
   - bio_u and bio_h pass on the interior.

   The interior gate still fails, for two reasons:
   - *Biased fit.*  The top theta is position free but was fitted on
     contexts mixing edge and interior cells.  Its unary is the average:
     1.0-1.3 nats too attractive for an interior cell.  Installed on the
     torus, the open-world thetas give bio_u 0.067 and bio_h 0.114, against
     0.035 and 0.062 for the torus fit.
   - *Leaking edge.*  In a 6 x 6 world the edge reaches the interior
     through the slot pairs and the biome chain.  The open-world interior
     oracle differs from the torus oracle by bio_u 0.052 and present_disc
     0.039.

   So the edge both biases the fit and contaminates the evaluation region.
   Masking the evaluation cannot undo the first.  A periodic training world
   removes both.
3. **The remaining residual is noise.**
   - *Mid level.*  Pooled over 256 runs against a 2000-sweep oracle, every
     feature's L1 is at the level its noise predicts: obj_h 0.114 against
     about 0.105 expected (half of noise pool), obj_u 0.023 against about
     0.019, before the long oracle's own noise is added.
   - *Top level.*  The largest pooled ratio is in the top features (bio_u
     0.016 against noise pool 0.009 at S = 30).  But the same quantity at
     S = 100 is 0.029 against 0.049, so neither budget shows a bias that
     the other does not.
   - *What bias is left.*  Any residual top-level bias is below about
     0.02-0.03 in bio_u L1 at this sample size.  Its probable source, if
     real, is the top pair terms: bio_h / bio_v are tabular nearest-
     neighbour only and the target is collapsed under the K = 0 contexts.
     Mid pairs beyond the 4 offsets cannot contribute, since the support
     and the shared-ring term are exactly nearest-neighbour.
4. **The sweep budget is not a factor.**
   - The pooled L1 at S = 100 matches S = 30 within noise on every feature.
   - S = 100 against S = 30 directly: obj_u 0.077 against noise 0.068,
     bio_u 0.050 against 0.048.

   The chains are mixed at 30 sweeps: tau(obj | biome) is 1.2-1.6 sweeps.
5. **No edge-aware unary was needed.**  Item 3, an `obj_u_edge` per edge
   class, was not built, because the periodic world passes.  It would only
   serve a bounded toy world.

**Consequence for the design docs.**  Toy worlds used for training and
end-to-end gates should be periodic, because production worlds are infinite
and have no edge.  An edge biases the position-free learned potentials and
leaks into the interior statistics.

Deviations:

- The periodic world is a padded torus with ghost copies and a per-colour
  sync, not a wrapping kernel, because the generic kernel's off-grid rule
  cannot wrap.  This is exact Gibbs on the torus for the old path; the
  Sampler path is not supported.
- `materialise` / `all_tables` take the open-world `CirclesBiome` (C0).  The
  tables do not depend on the world, and the padded `dem_of` would treat a
  5 x 5 canvas as a world grid.  `ghost_width` now returns 0 for non-world
  shapes as a guard.
- Additions beyond the brief:
  - oracle noise;
  - the `deep` interior;
  - the open-world thetas on the torus;
  - the pooled 4 x 64-run residual against a 2000-sweep oracle.
- Not rerun on the torus: stage 4/5 AIS columns and K = 3.

### Stage 6b: the Potts leak with collapsed targets

The question: was the Potts leak (S1's top tables at ~0.7 of S0's,
notes/potts_test.md "top_h / top_v: the leak") a target error and not a
context error?  S1's top target is p*(T | mid), a one-site conditional that
reads the level below T.  The target a top-down sampler needs, since it draws
T before its mid cells, is the collapsed conditional: T's mid cells (and
their tiles) integrated out.

Script: `notes/experiments/potts_collapsed.py`.  Output:
`images/potts_collapsed.json`, `images/potts_collapsed_tables.png` (top_h
double-centred per target type next to S0's published table).  Whole run
156 s on one core.

Config: nty = ntx = 6 (12 x 12 mid, 48 x 48 tiles), lam = 3, kappa 8 / J 0.1
and kappa 1 / J 0.3.  All fits are `train.fit`, iters 5, n_contexts 8, l2
1e-4, tabular feats, holdout 0.1.  Contexts are fresh forward runs (30 / 30 /
20 sweeps) at the current tables, and they share their Channel objects with
an `Oracle`.

- Mid: `ExactTargets(Oracle.mid_probs)`, K = 0, top tables 0 (stage 6
  setup a), refitted at both settings.  kappa 8: mid_h by d = -0.398 /
  +0.001 / +0.396 (reference BM J d = -0.40 / 0 / +0.40).  kappa 1: -0.295 /
  +0.018 / +0.259 (published S1 -0.29 / +0.02 / +0.26, S0 -0.31 / +0.02 /
  +0.26).  The mid tables are then fixed for every top fit.
- Top: top_h, top_v and top_u are fitted with the fitted mid tables
  installed.  The designed offset is 0, because no designed factor is homed
  on top (`designed_energies(orc.model, "top")`).  The learned top tables
  enter only through the fit's X theta, as in biome_fit.  Target types:
  - **frozen**: `ExactTargets(softmax(-site_energies("top", below=True)))`,
    i.e. S1's p*(T | mid) = softmax(-lam #(own mid cells outside palette)).
  - **ais**: `AISTargets(child "mid", region = the cell's 2 x 2 mid block,
    designed_e 0, after_change none, K 32, M 16, L 30)`.  p_0 holds lam (the
    parent row) and mid_u; mid_h / mid_v are annealed.  As in the circles
    run, the region is the written cells only.  The neighbouring mid cells
    are a fixed boundary, with their mid_h / mid_v rows into the window
    kept.
  - **hook**: the same window, log Z computed exactly by a column transfer
    matrix over the window (`window_logz`, checked against brute force).
  - **hook4** (extra): the window dilated by the mid tables' reach (one mid
    cell), so 4 x 4 cells, exact.  The fixed boundary then sits at distance
    2 from T's cells, and the ring cells keep their lam rows to the
    neighbouring tops.
  - K = 3 (frozen and hook): T is drawn from the target, then
    `Oracle.mid_move` runs at its 4 mid cells (a collapsed draw given T and
    the boundary tiles, with the block's tiles redrawn).
- Eval:
  - Forward L1 to the oracle moments (50 + 400 sweeps, symmetrised as in
    potts_train.py, whose `symmetrise` is copied because that script runs on
    import), over two independent 64-run evals (L1 a / L1 b).
  - eval noise = L1 between those two evals.
  - edge_same_top (held out).
  - The fit's final `viol_targets` (the consistency of the targets; the
    fit's own is about 1e-15 everywhere).
  - tau of the mid kernel with top clamped (`autocorr_clamped(fw.model,
    "mid", ["top"])`, at the final tables).

#### Contrast (double-centred learned top tables; one run each)

| setting | target | K | disjoint - same (h / v) | one-colour - same (h / v) | ratio to S0 | edge_same_top (eval a / b) |
|---|---|---|---|---|---|---|
| kappa 8, J 0.1 | S0 published | | 0.73 / 0.71 | 0.31 / 0.32 | 1 | .397 (oracle .395) |
| | S1 published | | 0.50 / 0.53 | 0.25 / 0.26 | 0.72 | .381 |
| | S3 published | 3 | 0.67 / 0.70 | 0.30 / 0.29 | 0.95 | .392 |
| | frozen | 0 | 0.26 / 0.05 | 0.09 / 0.07 | 0.22 | .359 / .369 |
| | frozen | 3 | 0.58 / 0.62 | 0.28 / 0.25 | 0.83 | .396 / .381 |
| | ais | 0 | 0.70 / 0.67 | 0.31 / 0.30 | 0.96 | .392 / .390 |
| | hook | 0 | 0.71 / 0.68 | 0.31 / 0.30 | 0.96 | .392 / .392 |
| | hook | 3 | 0.71 / 0.72 | 0.30 / 0.31 | 0.99 | .394 / .392 |
| | hook4 | 0 | 0.72 / 0.72 | 0.31 / 0.32 | 1.00 | .391 / .392 |
| kappa 1, J 0.3 | S0 published | | 0.60 / 0.58 | 0.26 / 0.26 | 1 | .365 (oracle .365) |
| | S1 published | | 0.38 / 0.39 | 0.20 / 0.19 | 0.65 | .356 |
| | S3 published | 3 | 0.52 / 0.52 | 0.22 / 0.24 | 0.88 | .361 |
| | frozen | 0 | 0.00 / -0.20 | -0.06 / -0.09 | -0.16 | .350 / .354 |
| | frozen | 3 | 0.19 / 0.42 | 0.09 / 0.11 | 0.52 | .355 / .355 |
| | ais | 0 | 0.50 / 0.50 | 0.23 / 0.23 | 0.85 | .359 / .359 |
| | hook | 0 | 0.50 / 0.50 | 0.23 / 0.23 | 0.85 | .359 / .359 |
| | hook | 3 | 0.49 / 0.49 | 0.22 / 0.23 | 0.83 | .358 / .360 |
| | hook4 | 0 | 0.50 / 0.51 | 0.23 / 0.23 | 0.85 | .359 / .359 |

The ratio to S0 is (disjoint h + v) / (S0's published h + v).  The
contrast is invariant under double-centring, so these numbers compare
directly with potts_test.md's.

#### L1 to the oracle moments (64-run eval a; eval b in brackets for the top keys)

| setting | target | mid_h | mid_v | mid_u | top_h | top_v | top_u |
|---|---|---|---|---|---|---|---|
| kappa 8 | frozen K0 | 0.133 | 0.158 | 0.107 | 0.215 (0.214) | 0.300 (0.227) | 0.118 (0.098) |
| | frozen K3 | 0.111 | 0.116 | 0.073 | 0.168 (0.186) | 0.187 (0.163) | 0.100 (0.095) |
| | ais K0 | 0.051 | 0.054 | 0.034 | 0.096 (0.131) | 0.077 (0.093) | 0.043 (0.075) |
| | hook K0 | 0.045 | 0.051 | 0.026 | 0.082 (0.125) | 0.070 (0.089) | 0.038 (0.065) |
| | hook K3 | 0.042 | 0.040 | 0.023 | 0.067 (0.104) | 0.050 (0.078) | 0.030 (0.045) |
| | hook4 K0 | 0.038 | 0.041 | 0.019 | 0.073 (0.077) | 0.058 (0.054) | 0.031 (0.024) |
| | eval noise (range over runs) | 0.06-0.09 | 0.06-0.08 | 0.03-0.06 | 0.10-0.16 | 0.09-0.11 | 0.03-0.07 |
| kappa 1 | frozen K0 | 0.114 | 0.145 | 0.079 | 0.238 (0.224) | 0.331 (0.245) | 0.108 (0.069) |
| | frozen K3 | 0.121 | 0.119 | 0.086 | 0.204 (0.187) | 0.146 (0.112) | 0.090 (0.060) |
| | ais K0 | 0.097 | 0.100 | 0.071 | 0.134 (0.111) | 0.119 (0.093) | 0.073 (0.053) |
| | hook K0 | 0.099 | 0.101 | 0.073 | 0.134 (0.110) | 0.123 (0.092) | 0.073 (0.051) |
| | hook K3 | 0.095 | 0.099 | 0.070 | 0.121 (0.103) | 0.105 (0.083) | 0.069 (0.030) |
| | hook4 K0 | 0.100 | 0.103 | 0.077 | 0.135 (0.107) | 0.122 (0.077) | 0.076 (0.028) |
| | eval noise (range over runs) | 0.06-0.07 | 0.06-0.09 | 0.03-0.04 | 0.08-0.15 | 0.08-0.11 | 0.04-0.06 |

Published S0 for scale: kappa 8 top_h / top_v 0.081 / 0.073, kappa 1 0.087 /
0.071.

#### Diagnostics

Consistency violation of the targets (final iteration, max over probes):

- frozen and hook: 1e-15.  These targets do not depend on the neighbouring
  top at all.  frozen reads only T's own mid cells; the 2 x 2 hook reads
  the fixed neighbouring mid cells, never the neighbouring T.  So the
  closure check is trivially passed.
- ais: 0.32 (kappa 8) / 0.28 (kappa 1).  This is AIS noise around the
  hook's 0: the closure sums 8 log-probabilities.
- hook4: 0.35 / 0.19, exact, so this is a real inconsistency.  The ring
  cells read the neighbouring tops, and windowed conditionals with a frozen
  boundary at distance 2 are not conditionals of one joint.  It costs
  nothing in the learned tables.

The fit's own violation is about 1e-15 throughout.  tau(mid | top clamped)
is 1.0-1.4 for every run (the mid kernel mixes in about one sweep).

AIS vs the exact hook, on a fresh context at the final tables, over all 36
cells x 4 candidates:

| setting | dlogZ mean | max abs dlogZ | AIS se median / p90 / max (during the fit) | within 3 se | target L1 mean / max | ms per AIS run | fit seconds (AIS / hook) |
|---|---|---|---|---|---|---|---|
| kappa 8 | +0.003 | 0.104 | 0.032 / 0.047 / 0.089 | 98% | 0.024 / 0.053 | 0.93 | 15.7 / 1.4 |
| kappa 1 | +0.004 | 0.063 | 0.024 / 0.035 / 0.067 | 98% | 0.016 / 0.032 | 1.00 | 16.9 / 1.3 |

#### What the numbers show

(a) **The collapsed target closes the leak at K = 0 at kappa 8.**  With T's
mid cells integrated out under lam and the installed mid tables, the top
contrast is 0.96 of S0 by AIS and 0.96 by the exact hook.  The dilated
exact window gives 1.00.  The fitted-feature L1s are at or below eval noise
on every key, and edge_same_top is .392 against oracle .395 and S0 .397.
For comparison, S1 published 0.72 and .381, and S3 (K = 3 end-to-end)
published 0.95.  K buys nothing: hook K3 gives 0.99 against hook K0's 0.96,
inside the +-0.05 spread between entries that are equal by symmetry.

(b) **At kappa 1 the collapsed target gives 0.85, against S1's 0.65 and
S3's 0.88.**  ais, hook and hook4 agree to 0.01 and K = 3 gives 0.83, so
the 0.15 that remains is not in the top target's window or in the AIS.  It
is likely in what the top target integrates against, the mid tables:

- At kappa 1 the top-top coupling is induced partly through tile
  correlations across the top edge that a pairwise mid table carries only
  in projection.
- The mid tables' own target `Oracle.mid_probs` conditions on the boundary
  tiles of the neighbouring blocks.
- The mid L1s at kappa 1 sit at 0.10 against noise 0.07 in every variant,
  while at kappa 8 they are at noise.

This attribution was not separately tested.  edge_same_top is .359 (oracle
.365, S1 .356, S3 .361).

(c) **The leak was a target error.**  The context is the same in every row:
forward runs at the current tables, one pass, K = 0.  Only the target
changes.  frozen to collapsed moves the contrast from 0.22 to 0.96 at kappa
8 and from -0.16 to 0.85 at kappa 1.  Fitted by train.fit's
pseudo-likelihood regression, the frozen target is much worse than
published S1's RB moment matching (0.22 / -0.16 against 0.72 / 0.65), and
noisy (h / v 0.26 / 0.05).  The reason: p*(T | mid) is nearly a one-hot on
the current T, since the mid cells were drawn from that T.  The regression
therefore returns the forward's own q(T | neighbours), a fixed point at any
theta, and the fit stays near its start.  K = 3 with collapsed mid moves
lets the mid cells move under T and recovers part of it (0.83 / 0.52), as
S3 did.  The collapsed target needs none of this.

(d) **AIS against the exact hook at the top.**  The two give
indistinguishable tables (contrast 0.70 / 0.67 against 0.71 / 0.68 at kappa
8; identical at kappa 1).  log Z agrees to 0.003 on average, with 98% of
values within 3 se, a median se of 0.02-0.03 at K = 32, M = 16, and
per-site target L1 of 0.02.  AIS costs 1 ms per run, 11x the hook's fit
time (16 s against 1.4 s per fit); both are negligible.

(e) **The window.**  The circles convention (the written cells only, the
neighbouring mid cells a fixed boundary through the annealed rows) is
enough.  The 2 x 2 window's target never reads the neighbouring tops.  The
top coupling is learned through the forward's correlation between a
neighbouring top and its mid cells on the shared edge.  It is
self-consistent (violation 0) and matches S0 at kappa 8.  Dilating by the
mid tables' reach reads the neighbouring tops directly through the ring's
lam rows.  That moves 0.96 to 1.00 at kappa 8 and nothing at kappa 1, at
18-25x the hook's cost, and it makes the targets mildly inconsistent.

(f) **Consequence for the design docs** (notes/dsl_updates.md C2, the
targets of reference_math section 4).  The target at a site must integrate
out everything below the site's level that the sampler draws after the
site, including the site's own children at the home level's child level.
A one-site conditional of p* that reads a finer level is the frozen-site
target:

- p*(T | mid) here.
- The plain mid conditional with the tiles fixed (S1f).
- p*(T | slots) on the biome circles.

It reproduces the leak, and under train.fit's regression it does worse than
leak: the fit stalls at the forward's own conditional.  The fix is in the
target, not in the contexts or K.  The collapsed target over the child
sampler with its installed learned potential, the parent row in p_0, and
the window's written cells with the neighbouring child cells as a fixed
boundary, is enough at K = 0.  An exact hook, where one exists, is a cheap
drop-in for AIS.  What it cannot fix is error already in the child level's
learned potential: that propagates up, which is the kappa 1 residual.

Deviations from the brief:

- Mid tables were refitted at both settings.  images/bootstrap_potts.json
  holds kappa 8 only, and the refit matches it: -0.398 / +0.001 / +0.396
  against -0.398 / +0.001 / +0.396.
- Extra rows: hook K = 3 and the dilated hook4.
- The K-step after_change is the collapsed `Oracle.mid_move` at T's 4 mid
  cells, so a K-step is a p*-invariant block move.
- One training run per row.  Expect about +-0.05 on a contrast entry.

### Stage 7b: the admissible list

The fix queued by stage 7 (notes/circles_biome_stage7.md, "Hard-row cost"):
dsl_updates C1's candidates-from-the-allowed-set, built into the Sampler.
Script `notes/experiments/circles_biome_scale2.py` (results
`images/cbio_stage7b.json`, log `images/cbio_stage7b_log.txt`, 53 s);
the stage 7 script is unchanged.

**Design as built.**

- `kernel.adm_lists(..., rows_sel, D)` (new): for every home site, the
  energies of all D values under the parent rows (`Sampler.parent_rows()`:
  hard unaries and hard pairs whose other channel is not home, the same
  rows dormancy reads), and the list of finite ones with their parent
  energy.  Lists are deduplicated by content (a 64-bit hash of (value,
  energy bits), confirmed by full comparison) into CSR arrays `adm_ptr`,
  `adm_idx` (ascending values), `adm_e`, with `adm_id[y, x]` (int32) naming
  the site's list.  Memory O(sites + #distinct x |A|): 4.3 kB at every N
  here (7 lists of 33), against 46 kB for a dense (sites x D) bool at
  N = 20.  Build time O(sites x D x #parent rows), once per `init`.
- `Sampler(..., adm=True)` (new, default off): `init()` builds the lists,
  takes dormancy from them (a list of length 1; identical to the
  `admissible()` route), and records the remaining hard rows (`sib_rows`, the
  same-level ones: here the support at 8 offsets) as contiguous runs
  `[f0, f1)` of packed rows.  `sweep()` then runs `kernel.sweep_adm`;
  `candidates(y, x)` returns its set (`kernel.site_candidates_adm`).
  Invariant (docstring): the lists are valid until the parents change;
  `init()` again after they do; `refresh()` drops them (back to `sweep_cap`
  until the next `init`).  Certificate channels, `relax` and the tempered
  kernels keep `sweep_cap` (their parents are softened or the draw needs all
  D).
- `kernel.sweep_adm` / `_site_adm` (new): at a site, copy the list and its
  parent energies, run the sib hard rows on the list only (skipping values
  already inf), then exactly `_site_cap`'s logic on what remains: K = None,
  every listed value, soft rows on all of them; K > 0, `cand[0] = z_p`
  (energy inf if z_p is unlisted), the n listed values != z_p that the sib
  rows admit in ascending order, the same partial Fisher-Yates
  (`randint(0, n - j)`, energies swapped alongside), soft rows on the capped
  set, Gumbel-max draw.  The set of n finite candidates and their order are
  those of `sweep_cap` (which drops the same values as inf), and `_draw`
  consumes one uniform per finite entry, so the random stream is the same:
  **bit-identical** to `sweep_cap` for the same seed, not just equal in
  distribution.  The one caveat is floating-point order: the parent energy is
  summed first, so if a parent row with nonzero finite entries were packed
  after a sib row the sum could differ in the last bit (not the case in any
  model here: the masks are 0 / inf).
- One detail mattered for speed: evaluating the sib rows one `_rows` call
  per row (`f, f + 1`) cost ~0.4 us per site more than one call over a range
  (it made the adm path *slower* than the old one at N = 2, 5).  The sib rows
  are therefore passed as runs; here they are one run (the parent mask is
  packed first).

**Tests** (`tests/test_sampler.py`, additive): `test_adm_identity` (grids
equal to `sweep_cap` after `init` + 4 sweeps, same seed, for Potts(2, 2)
top/mid/tile with and without a hard palette parent row, Circles(2, 2)
top/mid/tile with and without a hard slot row, the toy with a hard parent
mask, K in None / 2 / 4 (K skipped on certificate homes); and CirclesBiome(2,
2) with 5 random families and pair masks, obj, K None / 8, 10 sweeps);
`test_adm_lists_and_dormancy` (the list at every site = `admissible()` under
the parent rows; dormant = lists of length 1 = `C.dormant_of(allow)`; at most
one list per biome value; dormant sites unchanged by 20 sweeps; every value
drawn admissible; `candidates` = z_p plus admissible values);
`test_adm_long_run_biome` (6000 sweeps each, 30 batches: family histogram,
value parity and energy, old path vs adm path with different seeds, |z| < 4
at K None and 8; observed max |z| 1.2 / 1.6).  In the scale script the
forwards on both paths end in the same states (present fractions equal to
the last digit at every N and K).

**Per-site cost** (us per active obj site per sweep; probe at active
fraction 1, S_M = 30 sweeps, 80 interleaved reps, medians; "stage 7" is the
stage 7 table; "old" is `sweep_cap` re-measured in this run, interleaved
with "adm", so the old/adm comparison shares the load; the machine carried
two other experiments, which is why "old" sits above stage 7 at N >= 5):

| N | D | K=8 stage 7 | K=8 old | **K=8 adm** | K=None stage 7 | K=None old | **K=None adm** | hard only K=8 old / adm | K=8 adm, tables spread 10 x 10 | forward K=8 old / adm | forward K=None old / adm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 33 | 1.14 | 0.99 | **0.96** | 2.09 | 1.81 | **1.79** | 0.76 / 0.73 | 1.09 | 1.01 / 0.98 | 1.89 / 1.80 |
| 5 | 81 | 1.62 | 1.91 | **1.47** | 2.88 | 3.38 | **2.48** | 1.61 / 1.18 | 2.04 | 2.10 / 1.83 | 3.67 / 2.78 |
| 10 | 161 | 2.31 | 2.83 | **1.77** | 4.40 | 5.29 | **2.82** | 2.46 / 1.38 | 2.73 | 2.97 / 1.96 | 5.34 / 2.97 |
| 20 | 321 | 3.00 | 3.43 | **1.66** | 5.92 | 6.87 | **2.56** | 3.03 / 1.23 | 2.82 | 3.81 / 2.18 | 7.19 / 2.72 |

Init and lists:

| N | init ms old / adm (probe, 144 sites) | distinct lists (all biomes non-none) | list lengths | list bytes | dense bool bytes |
|---|---|---|---|---|---|
| 2 | 0.077 / 0.133 | 1 | 33 (= all of D) | 1 120 | 4 752 |
| 5 | 0.124 / 0.215 | 7 | 33 each | 4 336 | 11 664 |
| 10 | 0.167 / 0.296 | 7 | 33 each | 4 336 | 23 184 |
| 20 | 0.187 / 0.316 | 7 | 33 each | 4 336 | 46 224 |

Packed obj rows 19; hard 5 at N = 2 (mask + 4 orthogonal supports) and 9 at
N >= 5 (mask + 8 supports); 1 parent row (the mask) at every N.

### What the numbers show (stage 7b)

(a) *The O(D) hard pass is gone.*  At N = 20 the capped update drops from
3.43 to 1.66 us (2.1x) and the full one from 6.87 to 2.56 us (2.7x); the
hard-only probe from 3.03 to 1.23 us.  From N = 5 to N = 20 (D 81 -> 321,
4x) the adm cost at K = 8 is 1.47 / 1.77 / 1.66, flat within the load noise,
where the old path grows 1.91 -> 3.43; at K = None it is 2.48 / 2.82 / 2.56
against 3.38 -> 6.87.  The cost now follows |A_p| = 33, not D.  Every
active site is visited over 33 values whatever N is, as predicted.

(b) *The claim "flat at the N = 2 cost (~1.1 us)" holds only from N = 5 on.*
N = 2 is cheaper (0.96) for a structural reason, not a D one: its two shapes
never meet diagonally, so only 4 of the 8 supports are hard (5 hard rows, not
9) and the other 4 run as soft rows on the 8 capped values instead of hard
rows on 33.  At N = 2 the list is all of D, so adm = old there (0.96 vs 0.99:
no overhead from the list).  The fair baseline for N >= 5 is ~1.5 us.

(c) *What remains that depends on D is memory, not work.*  The work per site
is now constant (33 values x 8 sib rows, 8 values x 10 soft rows), but the
tables the rows read are D x D: 16 pair tables (8 support, 8 learned pair
directions) are 0.84 MB at D = 81 and 13.2 MB at D = 321, so the per-candidate
lookups `tab[c, v_b]` fall out of L1 then L2 as D grows.  The cache check
reads the same values from tables spread 10 x 10 apart in memory (100x the
footprint): +0.13 us at N = 2, +0.6 at N = 5, +1.0 at N = 10, +1.2 at N = 20,
so the residual N = 5 -> 20 drift (and much of the remaining gap to N = 2)
is in the memory hierarchy.  Remedies, not done here: store the tables
value-of-neighbour-major (the 33 admitted values are two runs of 16
consecutive anchors, so a column read is a few cache lines), or evaluate
support and pair energies from the stamp features rather than D^2 tables.
The table storage (4 D^2 support entries) is the remaining quadratic
memory cost, unchanged by this stage.

(d) *What is O(D) and is fine.*  `adm_lists` runs the parent rows over all
D at every site once per `init`: 0.13 -> 0.32 ms per init (1.7x the old
`admissible()` dormancy pass, which it replaces), against 30 sweeps of
~0.2-0.4 ms each at the obj level.  It could be O(#distinct contexts x D) by
keying on the parent values instead of the site, but at these sizes it is
not worth it.  The five per-sweep scratch arrays are size D (allocation,
not work).  The list memory is O(sites) for `adm_id` plus one list per
distinct parent context (7-8 here), not O(sites x D).

(e) *Forward runs.*  With dormancy (12.7% none) the forward obj stage at
K = 8 goes 3.81 -> 2.18 us per active site at N = 20 and 7.19 -> 2.72 at
K = None, with identical samples (bit-identical kernel).  The
gains are slightly smaller than in the probe because the forward obj grid
starts empty and fills, and its sites are partly dormant (skipped in both).

Deviations: `adm` is an opt-in constructor flag (default off) so that the
concurrently running stage 4-5 experiments, which build Samplers through
`Forward`, are untouched; `Forward` does not expose it (circles_biome.py
was not edited), and the scale2 script switches it on through a `Forward`
subclass.  Since the path is bit-identical, flipping the default (or
passing it from `Forward`) is safe once those runs are done.  The machine
was loaded (load average 6-8) during the timed run; the old/adm comparison
is interleaved and fair, the absolute numbers are ~10-20% above an idle
machine.

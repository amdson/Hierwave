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

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
    def joint_samples(self, burn, every, n) -> iterator of (biome, obj, tile) states   # the JointSampler
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

Training (`train.py` additions; the two estimators of C2):

```
def fit_conditional(model, home, samples, features, steps, lr) -> theta
    # pseudo-likelihood: for every joint sample and every active site p of `home`, cross-entropy of the
    # sampled value over the sampler's candidate set, logits = -(E_designed(t) + Delta_p(t; theta)).
    # Delta_p from BatchedDeltaBias.  Linear psi (paint potential) -> convex; use Adam or L-BFGS.
    # Returns theta, the pair consistency violation (max over neighbouring active sites and value
    # pairs of the closure error of the one-site odds), and the training cross-entropy.
def fit_windows(...)                    # the existing paintpot / induce path, unchanged
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
- `fit_conditional` on data from the closed form itself (slots drawn from
  `mid_probs` with the tiles integrated out) recovers the materialised
  finite part to 1e-2 and reports a consistency violation below 1e-2.
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
4. **The mid level, both estimators.**  (a) `fit_conditional` on
   `Oracle.joint_samples(50, 5, 400)`; (b) `fit_windows` with the exact
   per-tile hook and with AIS (`K = 32, M = 16`) on 3 x 3 slot windows.
   Both with stamp features.  Report, double-centred on the finite part:
   max abs error against `reference()` on held-out value pairs, slope and
   corr, the pair consistency violation of (a), the two fits against each
   other, wall time, and `autocorr_clamped("obj", "biome")`.  Gate: both
   within 0.05 of the reference on held-out pairs; this decides Q1 (stay
   with derived features if so) and the circles half of Q4 (which
   estimator is cheaper for the same error).  (b) is the production
   route; (a) exists here only because collapsed moves are closed form,
   and its job is to measure what the recursion costs.
5. **The top level.**  Install the mid bias; fit `bio_*` by (a)
   `fit_conditional` with oracle top moves and (b) `fit_windows` with AIS
   on the `obj` channel (the recursion: AIS runs the mid sampler with its
   bias in).  Compare to oracle moments and the zeroth-order unary;
   check `Phi_top(none) = 0` exactly from both fits.  End to end: forward
   with everything installed vs oracle on every monitor, `edge_air_top`
   held out.  Gate: within eval noise on fitted features; `conflict` 0.
6. **Potts cross-check.**  Both estimators on the Potts mid level (kappa
   8, J 0.1 and kappa 1, J 0.3), where block samples are free.  Report
   the two tables against each other and against S1; one short run.
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
`J`), where collapsed moves are no longer closed form.  The real case of
Q4: compare windows + AIS against collapsed moves on a periodic world.

## Results

(none yet)

# The circles test: induced repulsion between objects, two levels deep

Second toy for self-play training of coarse tables (companion of
`notes/potts_test.md`, which has the training schemes, the trainer and the
measurements; everything there applies unless stated here).  Where Potts
has dense tables and entropy, this one has sparse geometric tables with
closed-form free energies: tiles are independent given the objects, so
every induced coupling is an exact sum over tiles and can be checked by
hand.  It is also closer to the real system: objects with footprints,
honour of a painted shape, and a parent that places its child at a
corner.

Constraints: numba, single core (`NUMBA_NUM_THREADS=1`), figures under
`images/`, nothing model-specific in the generic layer.

## The model

Tiles: `air = 0`, `dirt = 1`.  Field `mu` per air tile (dirt preferred).

Mid (blocks of `BM = 8`): an object, absent (`0`) or present with its disc
centre at offset `(cy, cx)` in `{2..5}^2` inside the block: `D = 17`,
value `1 + (cy - 2) * 4 + (cx - 2)`.  The disc is the 13-cell Euclidean
disc of radius 2 about the centre (never leaves the block); the ring is
its 8-neighbour dilation minus the disc (16 cells; spills at most one
cell into a neighbouring block when the centre is at 2 or 5).  The object
demands dirt on the disc and air on the ring, cost `kappa` per violated
tile (soft honour).

Top (blocks of `BT = 2` mid cells, 16 tiles): a corner `T in {0: NW, 1:
NE, 2: SW, 3: SE}`.  It demands that the mid cell at that corner holds an
object and the other three are empty, cost `lam` per mid cell that
disagrees.

### Position-dependent honour: painted channels

Both honour terms depend on where a cell sits inside its parent, which a
pair table over views cannot express (the layer's tables are position
free, as every real channel set is).  As in the sugar test, the
dependence is carried by deterministic painted channels, refreshed
whenever their source changes:

- `slot`, h = BM, D = 2: `slot[i, j] = 1` iff mid cell `(i, j)` is the
  corner its top parent chose.  Painted from `top`.
- `dem`, h = 1, D = 3 (`0` free, `1` dirt, `2` air): the union of all
  objects' demands (disc = dirt, ring = air); where a disc and a ring
  overlap, a fourth value `3` conflict (the tile is demanded both ways).
  Painted from `mid`.  So `D = 4`.

Honour factors, both over painted channels, position free:

- `lam`: pair homed on `mid`, a = (`mid`, `present`), b = (`slot`, `val`),
  offset (0,0), table `lam * [present != slot]`.
- `kappa`: pair homed on `tile`, a = (`tile`, `col`), b = (`dem`, `val`),
  offset (0,0), table: free -> 0; dirt-demand -> `kappa * [col == air]`;
  air-demand -> `kappa * [col == dirt]`; conflict -> `kappa` for either
  colour.
- `mu`: unary on (`tile`, `col`), `[0, mu]` wait: air = 0 costs `mu`,
  dirt = 1 costs 0, so the table is `[mu, 0]`.

No designed mid-mid, top-top, mid unary or top unary.  Learned (forward
only, from zero): `mid_h`, `mid_v` (17 x 17), `mid_u` (17), `top_h`,
`top_v` (4 x 4), `top_u` (4), zero-mean gauge as in Potts.

### Exact free energies (the reference tables)

Given the objects, tiles are independent.  Per tile, `-log Z`:

| tile's demand | Z | -log Z |
|---|---|---|
| free | `1 + e^-mu` | `f0` |
| dirt (one disc) | `1 + e^-(mu + kappa)` | `f_d` |
| air (one or more rings) | `e^-mu + e^-kappa` (dirt violates one ring) ... with `n` rings: `e^-mu + e^-(n kappa)` | `f_a(n)` |
| conflict (disc + ring) | `e^-kappa + e^-(mu + kappa)` | `f_c` |

So `F(objects) = sum over tiles of -log Z(tile)`, and the induced pair
term between two objects `o, o'` in adjacent blocks is `F(o, o') - F(o,
-) - F(-, o') + F(-, -)` with the other blocks empty.  Per conflict tile
it costs `f_c - f_d - f_a(1) + f0 ~ kappa - log 2` for large kappa; per
shared ring tile it gains `f_a(2) - 2 f_a(1) + f0 ~ -mu`.  `Potts`-style
`reference()` returns these exact tables (`mid_h`, `mid_v` by enumerating
the 17 x 17 placements, `mid_u` from the single-object footprint in an
interior block) and the top tables at `lam -> inf` by enumerating the two
corner objects' offsets: `top_h[T, T'] = -log sum_{o, o'} exp(-(u(o) +
u(o') + pair(o, o')))` where the two mid cells are the chosen corners of
two horizontally adjacent top cells (they are horizontally adjacent mid
cells when `T` is an east corner and `T'` the west corner on the same
row, diagonal when on different rows, otherwise more than one block apart
and the term is zero), and likewise `top_v`.

Learned tables are identifiable only up to row and column terms (the
unaries absorb them), so compare learned and reference pair tables after
double centering (subtract row means and column means, add the grand
mean), and unaries after centering.

Dials: `kappa = 4` (`kappa - log 2 ~ 3.3` per conflict tile, up to a few
tiles), `mu = 0.3` (up to ~5 shared ring tiles, ~ -1.5), `lam = 3`.  No
ordering transition to worry about; objects do not order.

## Interfaces

`castlegen/channels/circles.py`, mirroring `potts.py` so the training
driver can be shared:

```
class Circles:
    def __init__(self, nty, ntx, BM=8, BT=2, kappa=4.0, mu=0.3, lam=3.0)
    q (=2 tile colours), D (=17), P (=4), BM, BT, nty, ntx, nmy, nmx, H, W
    OY, OX : (D,) centre offsets (-1 for absent); DISC, RING: lists of (dy, dx) relative to the centre
    def channels(self) -> (top, mid, tile, slot, dem)      # views: top "pal" (identity, named pal so the driver's names match), mid "self" (identity) and "present"; tile "col"; slot "val"; dem "val"
    def designed_factors(self) -> [lam, kappa, mu]
    def learned_factors(self, theta) -> as Potts (names mid_h, mid_v, mid_u, top_h, top_v, top_u; pair views mid "self", top "pal")
    def theta0(self)
    def model(self, chans, theta) -> Model  (all five channels; slot and dem fixed=True everywhere)
    def paint_slot(self, top, slot); def paint_dem(self, mid, dem)
    def stats(self, top, mid, tile) -> dict  # fitted: mid_h, mid_v, mid_u, top_h, top_v, top_u (frequencies);
        # monitors: `present` (fraction of mid cells with an object), `viol` (fraction of demanded tiles violated),
        # `conflict` (fraction of tiles with demand 3), `contact` (fraction of adjacent present pairs whose
        # footprints touch or overlap), `slot_viol` (fraction of mid cells disagreeing with slot),
        # `edge_air_top` (fraction of air among tiles in the one-tile band each side of a top-block edge; held out)
    def reference(self) -> dict  # exact tables as above (mid_h, mid_v, mid_u, top_h, top_v, top_u), double-centred
    def symmetrise(self, stats) -> stats  # average over the dihedral group (transpose, mirrors) with the induced maps on corners and offsets; exact symmetries of p*
    def render(self, top, mid, tile) -> (H, W, 3) uint8  # dirt brown, air light; mid/top block edges; disc centres marked
class Forward:  # as Potts; run(): top sweeps, paint slot, mid sweeps, paint dem, tile sweeps
class Oracle:
    def mid_probs(self, i, j) -> (D,)     # p*(o | slot, neighbours' objects) with the block's tiles AND the spilled ring tiles integrated out exactly (closed form per tile); the tiles outside the footprint are untouched
    def mid_move(self, i, j)              # draw o, repaint dem locally, redraw every tile whose demand changed from its exact conditional
    def top_probs_mid(self, i, j) -> (P,) # p*(T | mid): prod over the 4 mid cells of exp(-lam [present != slot(T, cell)])
    def top_move_mid(self, i, j)          # draw from top_probs_mid, repaint slot
    def top_probs(self, i, j) -> (P,)     # collapsed over the 4 mid cells given the tiles: prod over cells of sum_o exp(-lam[..] - kappa * violations(o, tiles))
    def top_move(self, i, j)              # draw T, repaint slot, redraw the 4 mid cells from p*(o | T, tiles), repaint dem
    def tile_sweep(self, n=1)             # generic kernel on "tile"
    def sweep(self, n=1, tile_sweeps=2)   # top moves, mid moves, tile sweeps
    def moments(self, burn, sweeps) -> dict (symmetrised)
```

The driver's hooks (see `notes/experiments/selfplay_train.py`): target
conditionals for the top come from `Oracle.top_probs_mid` here (the
painted `slot` is not a factor the generic `site_probs` can see), from
`site_probs(m_star, "top", below=True)` for Potts; mid from `mid_probs`
in both.  The S2 relaxation of the (top, mid) pair with no tiles: top by
`top_move_mid`, mid by `m_full.sweep("mid", 1)` with the slot repainted
after each top sweep.  The frozen diagnostic S1f uses
`Oracle.mid_probs_plain(i, j)`: the one-site conditional with the tiles
fixed, `exp(-lam[..] - kappa * violations(o, tiles))`.

Tests (`tests/channels/test_circles.py`): `mid_probs` against brute force over the
block's tiles on a reduced footprint (use `BM = 4` with a radius-1 disc
if the class allows `R` as a parameter, else enumerate only the footprint
tiles, which is exact because tiles are independent); `reference()`
`mid_h` against the per-tile formula for a hand-placed pair
(centre (3, 5) left, centre (3, 2) right: count the conflict and shared
tiles by hand in the test); `top_probs` and `top_probs_mid` against brute
force; `paint_dem` conflict marking; `Forward.run` / `Oracle.sweep` on
`Circles(2, 2)`; the energy identity (`Model.energy("tile")` equals the
numpy sum of `kappa` violations and `mu` air).

## Experiment

Same schemes and measurements as Potts (S0, S1, S2, S3 with CD-K
accumulated gaps, S1f), `nty = ntx = 6` (12 x 12 mid, 96 x 96 tiles),
one kappa setting (4) first.  Report the learned `mid_h` and `top_h`
against `reference()` (double-centred, max abs error and the pattern),
the top-table sparsity (entries that should be zero), the leak ordering
S1 / S2 / S3 / S0 on the top contrast, and the held-out `edge_air_top`.
Outputs `images/circles_*`, results appended here.

## Corrections after the build

- The ring (8-neighbour dilation of the 13-cell disc, minus the disc) has
  24 cells, not 16; the footprint is 37 tiles.  Spill is still at most one
  cell.
- `dem` has one air value, so a tile in two rings pays `kappa` once if
  dirt; the spec's `f_a(n)` is `f_a(1)`.  At the dials: a shared ring tile
  gains `f0 - f_a = -0.83`, a conflict tile costs `f_c - f_d - f_a + f0 =
  +2.63`.  Pairs that only share ring tiles attract (up to 4 shared tiles,
  -3.3), pairs with conflicts repel; the induced top coupling is net
  attractive, raw `top_h` is -0.69 at NE->NW and SE->SW and exactly 0
  elsewhere.
- An object costs `F_obj ~ 26.95` in free energy (tile entropy lost over
  the footprint), so with `lam = 3` and no other term `p*` has no objects.
  Raising `lam` past that freezes the top level instead (given the tiles
  the corner is determined, so no top conditional moves).  Fix, as in the
  sugar test: a designed presence bonus `pres`, unary `-b` on present
  with `b = F_obj - log(#offsets)` by default, so an isolated block is
  50/50 absent / present and `lam = 3` tilts it.  The learned `mid_u`
  then carries only what the neighbours and the boundary induce.
- `top_move` is a Metropolis independence move with the factorised
  `top_probs` as proposal and the exact 17^4 sum as target (tiles shared
  between the four mid cells); exact, but it can stick from a start with
  heavily overlapping objects, so the oracle starts with one object per
  chosen corner.
- `symmetrise` is exact on `p*` only for `nty == ntx`.

## Results

Script: `notes/experiments/selfplay_train.py` with `MODEL=circles` (output
`images/circles_selfplay_<run>.json`, logs `images/circles_log_<run>.txt`,
markdown `images/circles_selfplay_<run>_<run>_mu0.3_k4.md`); analysis of the
learned tables `notes/experiments/circles_analyse.py <json> [<figure>]`.
Figures `images/circles_<run>_k4_curves.png`, `_tables.png` (the driver's,
one colour scale per row: the learned 17 x 17 mid_h are washed out next to
the reference, use the next one), `_tables_scaled.png` (each panel its own
scale, plus the oracle's top pair pattern), renders `_oracle.png`,
`_untrained.png`, `_<scheme>.png`.  Two runs:

- `eta0.5`: the plain step, ETA = 0.5 on every table, schemes S0 S1 S2 S3
  (K = 3) S1f S2rb S3rb.
- `fwdscale`: a per-entry preconditioned step (new env `SCALE=fwd`): gap /
  max(forward frequency, 0.1 / table size), each entry's step clipped to
  0.5 nats, ETA = 0.1 (so the step is ~ eta log(p_fwd / p_target)); schemes
  S0 S1 S2 S3 S1f.  A 40-step test at ETA 0.5 / clip 1.0 oscillated (top_u
  of S0 went to [.56 -.89 -.46 .78]); ETA 0.1 / clip 0.5 was the one setting
  tried after it.  The preconditioner uses only the forward's own counts.

Config both: dials kappa 4, mu 0.3, lam 3, b = F_obj - log 16 = 24.18
(F_obj = 26.95); nty = ntx = 6 (12 x 12 mid, 96 x 96 tiles), RUNS = 8,
STEPS = 200, K = 3, BURN = 50, SWEEPS = 400, S_T = S_M = 30, S_F = 20, curves
every 10 steps on 16 eval runs, final stats on 64 runs, "eval noise" = L1
between two independent 64-run evaluations of the same tables (as Potts).
Oracle moments 50 + 400 sweeps: 6.3 s (14 ms / sweep).  ETA = 1.0 was not
run: the reason for the second run is the pair table, whose entries have
frequencies ~5e-4 and move ~1e-4 nats per step at ETA 0.5; doubling ETA does
not change that.

**Why a second run.**  At ETA 0.5 the presence level is learned (below) but
the present-present block of mid_h is hardly learned by any scheme in 200
steps (least-squares slope on the reference 0.06 for S0 / S3, 0.16 for
S1 / S2).  The step is per-site frequency, and a single offset pair has
frequency ~ 0.35^2 / 256.  The preconditioned run learns that block (slope
1.2-1.5) but ends with too many objects (present .41-.50 vs .35).  Neither
run is a clean converged result; both are reported.

Untrained gap magnitudes (|g|_1, per site): S0 mid_h 1.80, mid_u 1.29,
top_h 0.28; S1 mid_h 2.22, mid_u 1.06, top_h 0.35; S3 mid_h 1.64, mid_u
1.29, top_h 0.28.  The RB gaps are as large as S0's at both levels here
(on Potts the RB top gaps were ~3x smaller).

### Run eta0.5 (plain step, ETA = 0.5)

L1 distance to the oracle moments on the fitted features (final, 64 forward runs):

| | mid_h | mid_v | mid_u | top_h | top_v | top_u |
|---|---|---|---|---|---|---|
| untrained | 1.784 | 1.788 | 1.294 | 0.187 | 0.183 | 0.045 |
| S0 | 0.225 | 0.245 | 0.067 | 0.072 | 0.093 | 0.036 |
| S1 | 0.190 | 0.169 | 0.043 | 0.150 | 0.136 | 0.016 |
| S2 | 0.178 | 0.165 | 0.042 | 0.128 | 0.103 | 0.042 |
| S3 | 0.235 | 0.246 | 0.067 | 0.083 | 0.110 | 0.030 |
| S1f | 1.785 | 1.788 | 1.294 | 0.195 | 0.173 | 0.025 |
| S2rb | 0.193 | 0.176 | 0.045 | 0.200 | 0.156 | 0.046 |
| S3rb | 1.783 | 1.788 | 1.294 | 0.144 | 0.152 | 0.067 |
| eval noise S0 | 0.119 | 0.133 | 0.025 | 0.058 | 0.096 | 0.035 |
| eval noise S1 | 0.114 | 0.121 | 0.030 | 0.068 | 0.081 | 0.029 |
| eval noise S2 | 0.115 | 0.114 | 0.029 | 0.066 | 0.092 | 0.029 |
| eval noise S3 | 0.117 | 0.124 | 0.020 | 0.069 | 0.109 | 0.032 |
| eval noise S1f | 0.200 | 0.201 | 0.044 | 0.081 | 0.068 | 0.028 |
| eval noise S2rb | 0.122 | 0.123 | 0.029 | 0.097 | 0.082 | 0.038 |
| eval noise S3rb | 0.204 | 0.205 | 0.041 | 0.100 | 0.071 | 0.016 |

Monitors (edge_air_top held out):

| | present | viol | conflict | contact | slot_viol | edge_air_top | s/step |
|---|---|---|---|---|---|---|---|
| oracle | 0.3532 | 0.0201 | 0.0000 | 0.5236 | 0.1127 | 0.4773 |  |
| untrained | 1.0000 | 0.0243 | 0.0022 | 0.3522 | 0.7500 | 0.5719 |  |
| S0 | 0.3854 | 0.0237 | 0.0006 | 0.4376 | 0.1769 | 0.4794 | 0.20 |
| S1 | 0.3613 | 0.0226 | 0.0004 | 0.4822 | 0.1237 | 0.4795 | 0.23 |
| S2 | 0.3613 | 0.0228 | 0.0004 | 0.4862 | 0.1224 | 0.4784 | 0.24 |
| S3 | 0.3865 | 0.0234 | 0.0006 | 0.4289 | 0.1773 | 0.4796 | 0.56 |
| S1f | 1.0000 | 0.0243 | 0.0022 | 0.3520 | 0.7500 | 0.5720 | 0.23 |
| S2rb | 0.3630 | 0.0226 | 0.0004 | 0.4829 | 0.1260 | 0.4804 | 0.24 |
| S3rb | 1.0000 | 0.0243 | 0.0022 | 0.3538 | 0.7500 | 0.5717 | 0.57 |

Presence: the mid_u L1 (16-run curve) of S0 is 0.29 at step 40, 0.19 at
step 100, 0.10 at 200; S1 / S2 stay at the untrained 1.29 until step ~50,
then 0.49 at step 80 and 0.06 at step 100 (present ~ oracle by step 100).
S1's RB mid_u gap is smaller than S0's while present = 1 (1.06 vs 1.29)
and saturates; once objects appear it converges faster than S0.

### Run fwdscale (preconditioned step, ETA = 0.1, clip 0.5)

| | mid_h | mid_v | mid_u | top_h | top_v | top_u |
|---|---|---|---|---|---|---|
| untrained | 1.784 | 1.788 | 1.294 | 0.187 | 0.183 | 0.045 |
| S0 | 0.151 | 0.158 | 0.131 | 0.141 | 0.122 | 0.063 |
| S1 | 0.331 | 0.322 | 0.284 | 0.099 | 0.079 | 0.025 |
| S2 | 0.317 | 0.309 | 0.280 | 0.112 | 0.105 | 0.049 |
| S3 | 0.167 | 0.175 | 0.124 | 0.113 | 0.105 | 0.045 |
| S1f | 1.785 | 1.789 | 1.294 | 0.243 | 0.222 | 0.097 |
| eval noise S0 | 0.119 | 0.136 | 0.030 | 0.086 | 0.076 | 0.029 |
| eval noise S1 | 0.143 | 0.141 | 0.031 | 0.075 | 0.081 | 0.043 |
| eval noise S2 | 0.147 | 0.145 | 0.048 | 0.084 | 0.089 | 0.040 |
| eval noise S3 | 0.112 | 0.115 | 0.033 | 0.071 | 0.077 | 0.036 |
| eval noise S1f | 0.198 | 0.193 | 0.043 | 0.107 | 0.079 | 0.046 |

| | present | viol | conflict | contact | slot_viol | edge_air_top | s/step |
|---|---|---|---|---|---|---|---|
| oracle | 0.3532 | 0.0201 | 0.0000 | 0.5236 | 0.1127 | 0.4773 |  |
| untrained | 1.0000 | 0.0243 | 0.0022 | 0.3522 | 0.7500 | 0.5719 |  |
| S0 | 0.4187 | 0.0201 | 0.0000 | 0.4799 | 0.1846 | 0.4763 | 0.19 |
| S1 | 0.4951 | 0.0207 | 0.0000 | 0.3923 | 0.2529 | 0.4886 | 0.22 |
| S2 | 0.4934 | 0.0205 | 0.0000 | 0.4016 | 0.2523 | 0.4887 | 0.24 |
| S3 | 0.4110 | 0.0206 | 0.0000 | 0.4746 | 0.1758 | 0.4746 | 0.56 |
| S1f | 1.0000 | 0.0241 | 0.0020 | 0.3522 | 0.7500 | 0.5722 | 0.23 |

The mid_u L1 curves flatten by step ~40 (S0 0.16 -> 0.14, S1 0.35 -> 0.28
from step 40 to 200): present is stuck high, not converging slowly.  My
guess, not checked: mid_u and the absent row / column of mid_h, mid_v are
redundant carriers of presence and their preconditioned steps (scaled by
very different frequencies) work against each other.

### mid_h vs the exact reference

Double-centred.  "pp" = the 16 x 16 present-present block, where the
induced coupling lives (83% of its reference entries are exactly 0 before
centering; raw range -3.32, four shared ring tiles, to +1.94).  Entries
by (left centre)|(right centre), learned (reference, double-centred):
A = (3,5)|(3,2) conflict, B = (2,5)|(2,2) conflict, C = (3,5)|(3,3) shared
ring, D = (3,4)|(3,2) shared ring.

| run | scheme | max abs err | pp slope / corr | mean attract / repel / zero entries (ref -1.37 / +2.37 / +0.11) | A (+2.27) | B (+2.50) | C (-2.01) | D (-2.01) |
|---|---|---|---|---|---|---|---|---|
| eta0.5 | S0 | 2.59 | 0.06 / 0.67 | -0.12 / +0.03 / +0.01 | +0.00 | +0.07 | -0.13 | -0.13 |
| | S1 | 2.25 | 0.16 / 0.84 | -0.24 / +0.18 / +0.03 | +0.14 | +0.25 | -0.29 | -0.29 |
| | S2 | 2.25 | 0.16 / 0.84 | -0.24 / +0.18 / +0.03 | +0.14 | +0.25 | -0.29 | -0.28 |
| | S3 | 2.60 | 0.06 / 0.66 | -0.11 / +0.02 / +0.01 | -0.00 | +0.07 | -0.11 | -0.12 |
| | S1f, S3rb | 2.87, 2.88 | 0.00 | 0 / 0 / 0 | 0 | 0 | 0 | 0 |
| | S2rb | 2.25 | 0.16 / 0.84 | (= S1) | +0.14 | +0.25 | -0.29 | -0.28 |
| fwdscale | S0 | 1.36 | 1.17 / 0.91 | -1.69 / +2.94 / +0.13 | +2.89 | +2.59 | -2.45 | -2.78 |
| | S1 | 3.31 | 1.46 / 0.89 | -1.76 / +4.69 / +0.06 | +5.11 | +3.07 | -2.77 | -3.12 |
| | S2 | 2.90 | 1.43 / 0.90 | -1.78 / +4.52 / +0.07 | +4.78 | +2.92 | -2.66 | -3.23 |
| | S3 | 1.96 | 1.23 / 0.89 | -1.76 / +3.05 / +0.14 | +3.16 | +3.14 | -2.45 | -2.78 |
| | S1f | 3.05 | 0.01 / 0.21 | +0.03 / +0.17 / +0.00 | +0.21 | +0.20 | +0.04 | +0.03 |

Every scheme that learns at all gets the signs right: conflicts repel,
shared rings attract, the zero entries stay near zero.  At ETA 0.5 the
magnitudes are 6-16% of the reference.  Preconditioned, they are 1.2-1.5x
too large; S1 / S2 overshoot the repulsions most (mean +4.7 vs +2.4).  A
repulsion is learned from a pair the forward rarely produces, so it is
poorly determined beyond "large".  Entries with a repulsion that large
are never sampled, so their excess costs nothing in the fitted L1.  mid_v
is the same (slopes within 0.03).

**mid_u.**  The reference unary (designed -b included) has present - absent
= log 16 = 2.77, so the learned mid_u should have present - absent =
b + log 16 = F_obj = 26.95.  Learned (mean over offsets minus absent): eta0.5
S0 18.8, S1 21.1, S2 21.1, S3 19.1; fwdscale S0 11.6, S1 16.0, S2 16.0, S3
11.6; S1f 0.05 / 1.6, S3rb 0.08.  The rest sits in the absent row and
column of mid_h / mid_v (a unary under another name).  The effective gap
for a cell with four absent neighbours (mid_u plus the four pair terms) is
eta0.5 S0 29.4, S1 26.8, S2 26.8, S3 29.4; fwdscale S0 27.1, S1 27.1, S2
27.2, S3 27.0, against F_obj = 26.95.  So the ~27 nats are reached, by
step ~40 (S0) / ~100 (S1) at ETA 0.5; S0 / S3 overshoot by 2.5 nats at ETA
0.5 (present .385 vs .353).  The spread over present offsets is 0.1-0.8
(reference 0 for interior blocks; boundary blocks differ).

### top_h vs the oracle and the reference; the leak

The oracle's top pair pattern (relative excess over independence, rows =
left corner T, cols = right corner T', order NW NE SW SE):

    NW: -.02 +.34 -.14 +.16      NE: +.13 -.02 -.31 -.14
    SW: -.14 +.16 -.02 +.34      SE: -.31 -.14 +.13 -.02

i.e. NE->NW and SE->SW +13% (the reference's adjacent-corner attraction),
NW->NE and SW->SE +34% (two objects one block apart that the reference
leaves at 0: the induced coupling has a second-neighbour range through
the empty mid cells between them), NE->SW and SE->NW -31%.
`-log(p / p p)` double-centred, the target shape the learned tables are
compared to: NE->NW = SE->SW = -0.31, NW->NE = SW->SE = -0.14, NE->SW =
SE->NW = +0.18.  The reference (raw: -0.69 at NE->NW and SE->SW, 0
elsewhere) double-centred: -0.43 there, +0.26 at NE->SW / SE->NW, +-0.09
elsewhere.  Neither the oracle nor any learned table is sparse: the
entries the reference has at raw 0 are not ~0 in the oracle at lam = 3
(NW->NE is the second-largest attraction), so "sparsity" is not a test
any scheme can pass here.

Top contrast (energy units, from the learned tables): mean over
{NE->SW, SE->NW} minus mean over {NE->NW, SE->SW, NW->NE, SW->SE}, after
double centering; top_v by the transpose map (NE <-> SW).  The oracle's
`-log(p/pp)` contrast is 0.40 and the reference's 0.52.  Corr / slope are
for learned top_h on the oracle's `-log(p/pp)`.

| run | scheme | contrast h / v | ratio to S0 (h+v) | top_h corr / slope vs oracle | corr vs reference |
|---|---|---|---|---|---|
| eta0.5 | S0 | 0.34 / 0.27 | 1 | 0.99 / 0.85 | 0.93 |
| | S1 | 0.11 / 0.10 | 0.33 | 0.87 / 0.23 | 0.83 |
| | S2 | 0.26 / 0.23 | 0.80 | 0.96 / 0.60 | 0.81 |
| | S3 | 0.47 / 0.35 | 1.35 | 0.98 / 1.09 | 0.92 |
| | S1f | -0.00 / -0.07 | -0.12 | 0.00 / 0.00 | -0.11 |
| | S2rb | -0.04 / 0.07 | 0.04 | -0.06 / -0.03 | 0.03 |
| | S3rb | 0.39 / 0.45 | 1.38 | 0.94 / 1.05 | 0.83 |
| fwdscale | S0 | 0.24 / 0.29 | 1 | 0.96 / 0.58 | 0.94 |
| | S1 | 0.30 / 0.24 | 1.03 | 0.88 / 0.73 | 0.87 |
| | S2 | 0.38 / 0.37 | 1.42 | 0.84 / 0.80 | 0.84 |
| | S3 | 0.33 / 0.28 | 1.15 | 0.86 / 0.70 | 0.88 |
| | S1f | -0.06 / -0.10 | -0.31 | -0.67 / -0.16 | -0.74 |

Learned top_h double-centred (rows T = NW NE SW SE), for the shape:
eta0.5 S0 `-.01 -.10 +.12 -.01; -.26 +.01 +.15 +.11; +.13 -.00 +.00 -.13;
+.14 +.09 -.27 +.03`, S1 `-.00 +.00 -.01 +.01; -.07 -.03 +.05 +.05; +.02
-.00 +.04 -.05; +.06 +.03 -.08 -.01`; fwdscale S1 `-.11 +.02 +.10 -.01;
-.25 +.02 +.16 +.07; +.16 -.11 -.02 -.03; +.20 +.06 -.24 -.02`.  Every
learning scheme gets NE->NW and SE->SW as the strongest attraction and
NE->SW / SE->NW as the strongest repulsion.  NW->NE / SW->SE is weakly
negative in S0 (-0.10, -0.13) and less consistent in the others.

One training run per scheme.  The contrast's run-to-run noise was not
measured; the top_h L1 eval noise (0.06-0.10) is the same size as the
differences between schemes' top_h L1.

### Frozen sites, diagnostics, oscillation, drift

- S1f (mid sites by the one-site conditional with tiles fixed) learns
  nothing at the mid level in either run: an object's 37-tile footprint
  pins it, present stays 1.0, and mid_u, mid_h are 0 to two decimals at
  ETA 0.5.  At the top it learns a small table with the wrong sign
  (contrast -0.12 / -0.31 of S0; corr -0.67 with the oracle pattern in the
  preconditioned run).  With every mid cell present, p*(T | mid) is flat
  over T, so the top gap is noise plus whatever the frozen mid produces.
  Same finding as Potts: the collapsed mid conditional is what makes S1
  work.
- S2rb learns the S1 mid tables and no top coupling (contrast 0.04 of S0),
  as on Potts.  S3rb learns no mid tables (present 1.0) and a top table
  1.38x S0's contrast, from relaxations that start in an all-present
  forward state.  On Potts S3rb's top was 0.6 of S0.
- Oscillation: none at ETA 0.5.  Preconditioned at ETA 0.5 / clip 1 it
  oscillated (the 40-step test); at ETA 0.1 / clip 0.5 the S1 / S2 top_h
  curves spike at step 20 (0.42 / 0.44) and then settle.
- S3 drift: not clearly present.  The 16-run top_h curve of S3 is
  0.17-0.22 throughout at ETA 0.5; preconditioned it is 0.16-0.19 then
  0.26 at the last eval (one point; the 64-run final is 0.113 against
  noise 0.071).

Timing (s per training step, 8 forward runs of 30/30/20 sweeps, single
core): S0 0.19-0.20, S1 0.22-0.23, S2 0.24, S3 0.56 (K = 3 oracle sweeps
at 14 ms each, x 8 runs), S1f 0.23, S2rb 0.24, S3rb 0.57.  eta0.5 run (7
schemes) ~9 min; fwdscale run (5 schemes) ~5.5 min; tests ~1.5 min.

### What the numbers show

(a) *The induced top coupling is learned by S1, given a step that learns
the mid pair table.*  The induced coupling goes through objects and is not
sparse at lam = 3.  At ETA 0.5, S1's mid pair table is at 16% of the
reference and its top table is the right shape but small (contrast 0.33
of S0, slope 0.23 on the oracle pattern).  Preconditioned, S1's mid_h has
the right sign everywhere (too large) and its top table matches S0's
contrast (1.03) with corr 0.88 to the oracle pattern.  S1f learns nothing:
the collapsed mid conditional is needed, as on Potts.  The learned mid
tables carry the physics the spec predicts: conflicts repel, shared rings
attract, non-touching pairs ~0, and the absent-vs-present field reaches
F_obj ~ 27 nats once mid_u and the absent rows of the pair tables are
counted together.

(b) *Leak ordering.*  At ETA 0.5 the ordering is S1 (0.33) < S2 (0.80) <
S0 (1) < S3 (1.35).  S1 < S2 < S0 is as on Potts; S3 above S0 is not.
Preconditioned there is no leak: S1 1.03, S3 1.15, S2 1.42.  With one run
per scheme and two step rules that disagree, the circles data do not
establish a leak ordering.  What they show is that S1's top contrast
tracks how well its mid tables are learned: low when mid_h is at 16%,
S0-level when mid_h is fully grown.  On the held-out edge_air_top
(oracle .477, untrained .572) every learning scheme is within .003 of the
oracle at ETA 0.5.  Preconditioned, S0 / S3 are within .003 and S1 / S2
are at .489, tracking their excess presence (present .49).

(c) *Differences from Potts.*  (1) The step size, not the target, limits
the run: a dense 4 x 4 Potts pair table learns at ETA 0.5, a sparse
17 x 17 one with 5e-4-frequency entries does not.  Neither plain nor
preconditioned steps converged all of mid_u, mid_h, present together in
200 steps.  (2) The RB top gaps are as large as S0's here (0.35 vs 0.28
untrained), not 3x smaller.  (3) S3 sits above S0 in contrast at ETA 0.5,
and S3rb learns a large top table without a mid level.  On Potts both
were below S0.  (4) The tops' second-neighbour coupling (NW->NE +34% in
the oracle, 0 in the lam -> inf reference) is real at lam = 3, so the
reference top tables are only a qualitative check here.  Fitted L1s are
at or above eval noise (mid_h noise 0.11-0.15, top_h 0.06-0.10) and do not
rank S0-S3.

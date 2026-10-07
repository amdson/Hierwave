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

Tests (`tests/test_circles.py`): `mid_probs` against brute force over the
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

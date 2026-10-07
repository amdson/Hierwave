# The Potts test: self-play training of coarse tables on a three-level hierarchy

A toy with strong, legible induced couplings, an exact oracle, and a dial
from "every table computable by hand" to "coarse sites frozen under hard
honour".  It measures whether couplings between top-level latents, induced
only through the middle level and the tiles, are learnable by local
self-play, and how much the per-level recursion leaks against end-to-end
training.  Companion of the sugar test (`castlegen/channels/sugar.py`,
`notes/channels_demo.md`), whose coupling is entropic and subtle; here it
is energetic.

Constraints from the constitution and memory: numba kernels, single core
(`NUMBA_NUM_THREADS=1`), figures saved under `images/`, no rules specific
to this model inside the generic layer.

## The model

q = 4 colours, cyclic distance `d(c, c') = min(|c - c'|, q - |c - c'|)` in
{0, 1, 2}.  Three channels (names fixed):

| channel | h | domain | view |
|---|---|---|---|
| `tile` | 1 | colour 0..q-1 | `col` (identity) |
| `mid`  | BM = 4 | block colour 0..q-1 | `col` (identity) |
| `top`  | BM * BT = 8 (BT = 2 mid cells per side) | palette 0..P-1, P = q; palette k = {k, k+1 mod q} | `pal` (identity) |

Grid: `nty x ntx` top cells, so `(nty*BT) x (ntx*BT)` mid cells and
`(nty*BT*BM) x (ntx*BT*BM)` tiles.

Designed factors (the bidirectional energy `p*`), nothing else:

- `J`: tile-tile pair at offsets (0,1) and (1,0), table `J * d(c, c')`,
  `pad_b = pad_a = -1` (the term vanishes off the grid).  Name `J_h`, `J_v`.
- `kappa` (honour M -> F): pair factor homed on `tile`, a = (`tile`,`col`),
  b = (`mid`,`col`), offset (0,0), table `kappa * [c != m]`.  Name `kappa`.
- `lam` (honour T -> M): pair factor homed on `mid`, a = (`mid`,`col`),
  b = (`top`,`pal`), offset (0,0), table `lam * [m not in palette(p)]`.
  Name `lam`.

There is no designed mid-mid, top-top, mid unary or top unary term, so
every coupling at those levels is induced: mid-mid by the tiles across a
block edge (at kappa -> inf exactly `BM * J * d(m, m')` per shared edge),
top-top by the mid cells on the shared edge through the mid-mid coupling,
i.e. by the tiles two levels down.  Palette overlap (identical / one
shared colour / disjoint) is not a copy of the distance pattern, so the
top-top table has to be genuinely induced.

Learned factors (the forward model only), initialised at zero:

| name | factor |
|---|---|
| `mid_h`, `mid_v` | pair (`mid`,`col`)-(`mid`,`col`) at (0,1), (1,0), table (q, q), pad -1 |
| `mid_u` | unary (`mid`,`col`), (q,) |
| `top_h`, `top_v` | pair (`top`,`pal`)-(`top`,`pal`) at (0,1), (1,0), table (P, P), pad -1 |
| `top_u` | unary (`top`,`pal`), (P,) |

Gauge: every learned table is kept zero-mean (subtract the mean after
each step); the absolute level of a unary or table is not identifiable.

Forward procedure (the model being fitted): `Model` over all three
channels with designed + learned factors; `sweep("top", S_T)`, then
`sweep("mid", S_M)`, then `sweep("tile", S_F)`, top-down with the generic
kernel.  The top kernel sees only its learned tables (no designed term
reaches it top-down); the mid kernel sees `lam` and its learned tables;
the tile kernel sees `kappa` and `J`.

Dials: `J = 1`, `lam = 3`, `kappa in {8 (effectively hard), 1}`.  At
kappa = 8 the tiles are nearly deterministic given mid and the tables are
checkable by hand; at kappa = 1 there is entropy and roughening.

## Interfaces (fixed so the pieces can be built in parallel)

### `castlegen/channels/potts.py`

```
Q = 4
def dist(c, cp) -> cyclic distance (works on arrays)
class Potts:
    def __init__(self, nty, ntx, BM=4, BT=2, q=4, J=1.0, kappa=8.0, lam=3.0)
    # sizes
    q, P, BM, BT, nty, ntx, nmy, nmx (mid grid), H, W (tiles)
    PAL : (P, q) bool, PAL[p, c] = c in palette p
    def channels(self) -> (top, mid, tile)   # fresh Channel objects, names "top"/"mid"/"tile", views as above
    def designed_factors(self) -> list[Factor]          # J_h, J_v, kappa, lam
    def learned_factors(self, theta: dict) -> list[Factor]   # names as in the table; theta[name] is the table
    def theta0(self) -> dict                            # zeros of the right shapes
    def model(self, top, mid, tile, theta) -> Model     # all channels, designed + learned
    def stats(self, top, mid, tile) -> dict             # see below
    def render(self, top, mid, tile) -> (H, W, 3) uint8 # tiles coloured, mid/top block edges drawn
class Forward:
    def __init__(self, P: Potts, theta: dict, seed=0)
    top, mid, tile, model (rebuilt in run)
    def run(self, S_T=30, S_M=30, S_F=20, fresh=True) -> stats dict
class Oracle:
    """Two-way sampler of p* with collapsed moves (exact)."""
    def __init__(self, P: Potts, seed=0)
    top, mid, tile, model (designed factors only)
    def mid_logZ(self, i, j) -> (q,) float     # log Z of the block's tiles for each m, given boundary tiles (outside the block) and NOT the top; -inf never occurs
    def mid_probs(self, i, j) -> (q,) float    # p*(m | top parent, boundary tiles) with the block's tiles integrated out: softmax(-lam*[m not in pal] + mid_logZ)
    def mid_move(self, i, j)                   # draw m from mid_probs, then redraw the block's tiles exactly (backward sampling)
    def top_probs(self, i, j) -> (P,) float    # p*(T | tiles, other tops) with the 2x2 mid cells integrated out given the tiles: prod over the 4 mid cells of sum_m exp(-lam*[m not in pal] - kappa*mismatch(block tiles, m))
    def top_move(self, i, j)                   # draw T from top_probs, then redraw its 4 mid cells from p*(m | T, block tiles)
    def tile_sweep(self, n=1)                  # generic kernel sweeps of "tile" (reads mid)
    def sweep(self, n=1, tile_sweeps=2)        # per sweep: all top moves (random order), all mid moves (random order), tile_sweeps tile sweeps
    def moments(self, burn, sweeps) -> dict    # averaged stats over `sweeps` after `burn`
```

Collapsed mid move: column transfer matrix over the BM x BM block, column
state = BM colours (q^BM = 256 states at BM = 4).  Column weight: vertical
`J d` within the column, `kappa` mismatches against m, and the `J d` terms
against the fixed tiles above and below the block and (for the first /
last column) to the left / right.  Transition weight between columns:
`J d` per row.  Forward filter with per-column normalisation gives log Z;
backward sampling redraws the block.  Validate `mid_logZ` against brute
force at BM = 2 (q^4 = 256 states) in `tests/test_potts.py`, and
`top_probs` against brute force over the 4 mid cells.

`stats` returns, all as frequencies (pair tables normalised to sum 1 over
the pairs counted, unaries over the cells):

- fitted features: `mid_h`, `mid_v` (q, q), `mid_u` (q,), `top_h`, `top_v`
  (P, P), `top_u` (P,).  These keys coincide with the learned table names.
- monitors (tile level, what the user sees): `tile_d` (3,) fraction of
  tile 4-neighbour pairs at distance 0/1/2; `mismatch` fraction of tiles
  with colour != their mid colour; `pal_viol` fraction of mid cells
  outside their top's palette; `edge_same_mid` fraction of tile pairs that
  straddle a mid-block edge with equal colour; `edge_same_top` the same
  across top-block edges.  `edge_same_top` is held out from every fit.

### Generic layer additions (`core.py`, `kernel.py`, new `train.py`)

`Model.compile(home, below=False)`: with `below=True` the packing also
includes the factors homed on finer channels that read `home`, so the
kernel's candidate energies at a `home` site are the full conditional of
the joint (bidirectional).  Two new row kinds in `Packed.fac`:

- kind 3, below-pair: a pair factor homed on a finer channel `a` with
  `b == home` and offset (0, 0).  At a `home` site p with candidate v the
  term is `sum over fine cells s in block(p) of tab[va(z_s), vb(v)]`.
  Offsets other than (0, 0) on a below-pair are not supported (assert).
- kind 4, below-count: a count factor with `b == home`: `tab[vb(v), s]`
  with `s` the block sum of `va` over the fine channel.

`kernel._energies` handles kinds 3 and 4; `sweep` and `total_energy` are
unchanged for kinds 0-2 (`total_energy` with `below=True` packing must not
double count: skip kinds 3 and 4 there, they are already counted on the
fine side).  Tests: the bidirectional site energies at a mid site equal
the brute-force energy differences of the full joint, on a tiny Potts
model (use `potts.Potts(1, 1, BM=2)` if available, else build three
channels by hand).

`train.py` (generic, nothing Potts-specific):

```
def counts(model, names) -> dict name -> array
    # sufficient statistics of the named learned factors at the current state:
    # each pair counted once (a at p, b at q), unaries per cell; same shapes as the tables
def site_probs(model, home, T=1.0, below=True) -> (rows, cols, D)
    # the kernel's candidate energies at every site of `home` (below=True packing),
    # softmax(-e/T); fixed cells get a one-hot on their current value
def rb_gap(model, home, names, probs) -> dict name -> array
    # Rao-Blackwellised expected change of counts(model, names) if each site of `home`
    # were resampled, one at a time from the current state, with probability probs[y, x, :]:
    #   sum_p sum_t probs[p, t] (N(z with z_p = t) - N(z))
    # only the named factors homed on `home` (pairs with b == home, unaries) change; others give zeros.
    # `probs` may come from site_probs or from any other p*-invariant one-site kernel (e.g. a collapsed move).
def step(theta, gap, prior=None, eta=1.0, lam_prior=0.0, scale=None) -> new theta
    # theta[k] += eta * gap[k] / scale[k]  - lam_prior * (theta[k] - prior[k]); then zero-mean per table.
    # Convention: gap = (counts under the forward model) - (target counts), per site, so that an
    # over-produced pair gains energy.
```

Sign convention: energies, `E = sum theta[n]`, probability `exp(-E)`.  A
feature over-produced by the forward model relative to the target gets a
higher energy: `theta += eta * (N_forward - N_target)`.

### Training schemes (`notes/experiments/potts_train.py`)

Every scheme updates all six learned tables jointly with `step`, from
`theta0`, for `STEPS` steps, each step averaging over `RUNS` forward runs
(`Forward.run`, fresh).  The target side of the gap differs:

- **S0, oracle PCD** (the reference, as in the sugar test): target =
  `Oracle.moments(BURN, SWEEPS)` once, up front; gap = forward stats -
  oracle stats on the fitted features.  This is moment matching against
  the equilibrium of `p*`.
- **S1, per-level RB, one step**: from each forward sample, the
  Rao-Blackwellised expected change of the learned counts under one-site
  moves of `p*`: top sites with `site_probs(model, "top", below=True)`
  (reads mid through `lam`), mid sites with the collapsed conditional
  `Oracle.mid_probs` evaluated at the forward state (tiles integrated out;
  reads top and boundary tiles).  gap = -(expected change).  No
  relaxation; the top's gradient is evaluated at the forward's own mid
  state.
- **S2, two-level recursion for the top** (the "reuse the learned middle
  potentials" scheme): as S1 for the mid tables; for the top tables,
  first relax the (top, mid) pair for `K` sweeps under
  `E_lam + theta_mid` with NO tiles (mid sites by `site_probs(model,
  "mid", below=False)` which sees `lam` and the learned mid tables only;
  top sites by `site_probs(model, "top", below=True)`), then take the RB
  expected change at the relaxed state.
- **S3, end-to-end CD-K**: relax the full state for `K` sweeps of the
  oracle's kernel (`Oracle.sweep`-style moves applied to the forward's
  state: top moves, collapsed mid moves, tile sweeps), then the RB
  expected change of all learned tables at the relaxed state (top by
  `site_probs` below=True, mid by `mid_probs`).  K = 0 recovers S1; K ->
  inf recovers S0.
- Optional **S1-frozen**: S1 with the mid sites using the plain one-site
  conditional `site_probs(model, "mid", below=True)` instead of the
  collapsed one, to show the frozen-site problem at kappa = 8.

Mid sites also use the collapsed conditional for the *forward* side of
the RB gap?  No: the RB gap is a single quantity, the expected change
under the target kernel from the forward sample; the "forward counts"
never appear separately.  gap[name] = -rb_gap[name] / n_sites.

For S0 the gap is the moment difference, per site.  Use the same `eta`
scale for all schemes (per-site units) and report the learning curves.

### Measurements (`notes/experiments/potts_train.py`, outputs in `images/`)

For both kappa settings, for each scheme:

1. Learning curves: the fitted features' gap to the oracle moments per
   step (the oracle moments are the ground truth for every scheme, even
   those that never see them).
2. Final forward stats vs oracle stats on the fitted features and all
   monitors, especially the held-out `edge_same_top`.
3. The learned tables: `mid_h` against `BM * J * d(m, m')` (gauge-fixed:
   subtract the mean of each), `top_h` as a heatmap; the gap between
   schemes on `top_h` is the leak measurement.
4. Renders: one oracle sample, one forward sample untrained, one trained
   per scheme (`images/potts_<kappa>_<scheme>.png`), plus a figure of the
   tables (`images/potts_<kappa>_tables.png`).
5. Timing per step per scheme.

Results: `images/potts_results.json` and a results section appended to
this file (tables of numbers, one paragraph of what they show).

Sizes to start: `nty = ntx = 6` (12 x 12 mid, 48 x 48 tiles), `RUNS = 4`,
`STEPS = 150`, `K = 3`, `BURN = 50`, `SWEEPS = 200`, `S_T = S_M = 30`,
`S_F = 20`.  Reduce if a scheme's step exceeds ~2 s.

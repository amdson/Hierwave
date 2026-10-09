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
force at BM = 2 (q^4 = 256 states) in `tests/channels/test_potts.py`, and
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

Correction (found in the first run): for S2 and S3 the gap is the CD-K
form, the accumulated count change over the relaxation plus the RB
expected change of one more move at the relaxed state,
`gap = -(N(relaxed) - N(forward) + rb_gap(relaxed)) / n_sites`.  The RB
term alone at the relaxed state goes to zero as K -> inf (the relaxed
state is stationary), so it cannot recover S0; the accumulated form does.
The RB-only versions are kept as diagnostics `S2rb`, `S3rb`, and both
fail (results below).

Dials, corrected: `J = 1` puts `p*` in its ordered phase at both kappa
values (at kappa -> inf the mid level is a Potts model with coupling
`BM * J`, far above critical), and every scheme then collapses into one
colour sector through the unaries.  Use `J = 0.1` at kappa 8 and `J = 0.3`
at kappa 1, both disordered with clear induced correlations.  The oracle
moments are symmetrised over the exact symmetries of `p*` (colour
rotation and reflection with the matching palette map, lattice transpose
and mirror) to remove the colour-sector bias of a finite chain.

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

## Results

Script: `notes/experiments/potts_train.py` (env-var config; output
`images/potts_results.json`, one key per `k<kappa>_J<J>`, with oracle and
forward stats, learned tables, curves, timing, eval noise; logs
`images/potts_log_*.txt`).  Figures per setting: `images/potts{tag}_k{kappa}_curves.png`,
`_tables.png`, renders `_oracle.png`, `_untrained.png`, `_<scheme>.png`;
tag `_J0.1` / `_J0.3` for the disordered-phase runs, no tag for J = 1.

**The spec's dials (J = 1, lam = 3) put p* deep in its ordered phase, at
both kappa.**  At kappa -> inf the induced mid coupling `BM J d` is a q = 4
clock model = two decoupled Ising models with coupling 4J; Ising is
critical at 0.44, so J = 1 is ~9 x critical (and the tile level alone, K = J,
is already ordered).  A 450-sweep oracle chain at J = 1 sits in one colour
(kappa 8: mid_u = [0, 0, .013, .988], edge_same_top 0.994; kappa 1: mid_u =
[.10, 0, .07, .83], edge_same_top 0.913).  Oracle scan (6 x 6 top, 50 + 200
sweeps, colour fractions on the same-colour mid pair `tr(mid_h)` / top pair
`tr(top_h)`, uncorrelated = 0.25): kappa 8: J = 0.1 -> .50 / .36, J = 0.15 ->
.62 / .43 (colour fractions drifting, near critical), J = 0.2 -> .76 / .49
(ordering), J >= 0.3 ordered; kappa 1: J = 0.1 -> .38 / .26, J = 0.3 -> .46
/ .31, J = 0.5 ordered.  So the main runs use **J = 0.1 at kappa = 8 and J =
0.3 at kappa = 1** (disordered, with clear induced correlations at both
coarse levels); the spec's J = 1 runs are reported after them.

Changes from the spec, all in the script:

- *Oracle target symmetrised* over the symmetries of p* (colour maps c ->
  +-c + r with the induced palette map, lattice transpose / mirror).  Exact
  for p*; it removes the colour-sector bias of a finite chain (at J = 0.1 the
  raw oracle mid_u was [.23 .22 .28 .27]).  Forward stats are not symmetrised.
- *S2 and S3 include the accumulated count change*: gap = -(N(relaxed) -
  N(forward) + RB change at the relaxed state) / n_sites.  This is the
  expected count change over the K + 1 steps of the target kernel from the
  forward sample, the version for which K = 0 is S1 and K -> inf is S0, as
  the spec states.  The literal "RB change at the relaxed state only" is run
  as a diagnostic (`S2rb`, `S3rb`, kappa 8 only): it is a different
  estimator (the K-th step's change, not the change from the forward sample)
  and it fails (below).
- Target kernel: top sites `site_probs(m_star, "top", below=True)` on a
  designed-only model (`Oracle.model`), mid sites `Oracle.mid_probs` over
  all mid cells; counts read by `rb_gap` on the forward's full model; the
  forward's channels are the oracle's channel objects, so all act on one
  state.  Startup self-check: top conditionals = softmax(-lam * #mid cells
  outside the palette) at a random state, and the RB change of `top_u` =
  sum over sites of (probs - one-hot(current)).  Untrained gap magnitudes
  (|g|_1, kappa 8 J 0.1): S0 mid_h .31 top_h .34; S1 mid_h .26 top_h .11;
  S3 (K = 3) mid_h .23 top_h .15 — RB gaps are the same order as S0 on the
  mid tables and ~3x smaller on the top tables (one step of a slow top chain).

Config (main runs): nty = ntx = 6, RUNS = 8, STEPS = 200, K = 3, BURN = 50,
SWEEPS = 400, S_T = S_M = 30, S_F = 20, **ETA = 0.5** for every scheme,
learning curves every 10 steps on 16 eval runs (fixed seeds), final stats on
64 runs.  ETA = 2.0 (the start value) oscillated: at J = 0.1 S0 ended worse
than untrained (mid_h L1 .49 vs .29) and S3 drifted (top_h .55); ETA = 0.25
was stable but S1 had not converged in 150 steps (edge_same_top .355 vs
.395).  J = 1 runs: RUNS = 4, STEPS = 150, SWEEPS = 200, ETA = 0.5.

"eval noise" rows: L1 between two independent 64-run evaluations of the same
trained tables; final L1 values below roughly that level are not
distinguishable from the oracle.  The learning curves (16 eval runs) are
noisier still (~0.1 per key) and say little beyond "converged by ~100
steps"; the learned tables are the informative output.

### kappa = 8, J = 0.1

L1 distance to the oracle moments on the fitted features (final, 64 forward runs):

| | mid_h | mid_v | mid_u | top_h | top_v | top_u |
|---|---|---|---|---|---|---|
| untrained | 0.287 | 0.279 | 0.018 | 0.175 | 0.250 | 0.045 |
| S0 | 0.078 | 0.069 | 0.052 | 0.081 | 0.073 | 0.050 |
| S1 | 0.064 | 0.053 | 0.045 | 0.075 | 0.101 | 0.048 |
| S2 | 0.070 | 0.062 | 0.048 | 0.092 | 0.084 | 0.061 |
| S3 | 0.103 | 0.101 | 0.070 | 0.128 | 0.108 | 0.073 |
| S1f | 0.284 | 0.274 | 0.018 | 0.192 | 0.236 | 0.052 |
| S2rb | 0.067 | 0.078 | 0.009 | 0.202 | 0.250 | 0.122 |
| S3rb | 0.241 | 0.191 | 0.083 | 0.137 | 0.138 | 0.078 |
| eval noise S0 | 0.073 | 0.075 | 0.062 | 0.097 | 0.117 | 0.056 |
| eval noise S1 | 0.066 | 0.061 | 0.048 | 0.104 | 0.100 | 0.050 |
| eval noise S2 | 0.054 | 0.049 | 0.041 | 0.087 | 0.082 | 0.036 |
| eval noise S3 | 0.075 | 0.072 | 0.051 | 0.091 | 0.078 | 0.042 |
| eval noise S1f | 0.049 | 0.045 | 0.015 | 0.097 | 0.075 | 0.047 |
| eval noise S2rb | 0.057 | 0.055 | 0.032 | 0.090 | 0.095 | 0.020 |
| eval noise S3rb | 0.063 | 0.060 | 0.044 | 0.123 | 0.119 | 0.060 |

Monitors (tile_d = fraction of tile pairs at distance 0/1/2; edge_same_top held out):

| | tile_d | mismatch | pal_viol | edge_same_mid | edge_same_top | s/step |
|---|---|---|---|---|---|---|
| oracle | 0.882/0.101/0.017 | 0.0007 | 0.0220 | 0.500 | 0.395 |  |
| untrained | 0.849/0.119/0.032 | 0.0007 | 0.0474 | 0.359 | 0.248 |  |
| S0 | 0.883/0.100/0.017 | 0.0007 | 0.0199 | 0.506 | 0.397 | 0.10 |
| S1 | 0.882/0.101/0.018 | 0.0007 | 0.0202 | 0.498 | 0.381 | 0.18 |
| S2 | 0.883/0.100/0.017 | 0.0007 | 0.0196 | 0.505 | 0.395 | 0.19 |
| S3 | 0.883/0.100/0.017 | 0.0007 | 0.0200 | 0.505 | 0.392 | 0.57 |
| S1f | 0.849/0.118/0.032 | 0.0007 | 0.0472 | 0.361 | 0.250 | 0.10 |
| S2rb | 0.876/0.104/0.021 | 0.0007 | 0.0225 | 0.474 | 0.339 | 0.18 |
| S3rb | 0.857/0.113/0.031 | 0.0007 | 0.0456 | 0.392 | 0.293 | 0.60 |

mid_h vs BM J d(m, m') zero-meaned (d = 0 / 1 / 2 reference -0.40 / +0.00 / +0.40); learned mean over entries by distance, and max |mid_h - ref|; top_h rows:

| | mid_h d=0 | d=1 | d=2 | max abs err | top_h (rows p = 0..3) |
|---|---|---|---|---|---|
| S0 | -0.39 | +0.02 | +0.36 | 0.05 | -0.31 -0.02 +0.38 -0.05; -0.02 -0.38 -0.04 +0.38; +0.41 -0.02 -0.30 -0.06; -0.00 +0.40 +0.00 -0.36 |
| S1 | -0.40 | +0.01 | +0.38 | 0.03 | -0.24 -0.02 +0.23 +0.02; -0.01 -0.26 +0.01 +0.25; +0.26 -0.02 -0.19 +0.02; +0.00 +0.27 -0.01 -0.32 |
| S2 | -0.39 | +0.01 | +0.38 | 0.04 | -0.28 -0.02 +0.31 -0.03; +0.00 -0.32 -0.04 +0.32; +0.33 +0.02 -0.22 -0.07; -0.01 +0.33 +0.03 -0.34 |
| S3 | -0.39 | +0.02 | +0.36 | 0.05 | -0.31 -0.03 +0.36 +0.00; -0.04 -0.31 +0.03 +0.37; +0.33 -0.03 -0.27 -0.04; +0.01 +0.37 -0.06 -0.37 |
| S1f | -0.00 | +0.00 | +0.00 | 0.40 | -0.01 +0.00 -0.01 +0.07; +0.01 -0.05 +0.05 +0.01; -0.01 +0.03 +0.01 -0.04; +0.06 -0.02 -0.04 -0.05 |
| S2rb | -0.40 | +0.01 | +0.38 | 0.04 | +0.06 -0.04 +0.04 -0.10; +0.02 +0.04 -0.09 +0.06; -0.04 +0.05 -0.05 +0.04; -0.09 -0.01 +0.02 +0.07 |
| S3rb | -0.02 | +0.02 | -0.02 | 0.49 | -0.39 +0.03 +0.21 +0.10; +0.07 -0.25 +0.01 +0.19; +0.17 +0.04 -0.19 -0.03; +0.12 +0.22 -0.06 -0.24 |
### kappa = 1, J = 0.3

L1 distance to the oracle moments on the fitted features (final, 64 forward runs):

| | mid_h | mid_v | mid_u | top_h | top_v | top_u |
|---|---|---|---|---|---|---|
| untrained | 0.229 | 0.220 | 0.018 | 0.144 | 0.220 | 0.045 |
| S0 | 0.075 | 0.078 | 0.054 | 0.087 | 0.071 | 0.043 |
| S1 | 0.067 | 0.070 | 0.050 | 0.103 | 0.126 | 0.063 |
| S2 | 0.082 | 0.081 | 0.060 | 0.118 | 0.126 | 0.081 |
| S3 | 0.040 | 0.043 | 0.028 | 0.082 | 0.064 | 0.020 |
| eval noise S0 | 0.055 | 0.049 | 0.035 | 0.085 | 0.080 | 0.036 |
| eval noise S1 | 0.071 | 0.080 | 0.060 | 0.115 | 0.104 | 0.056 |
| eval noise S2 | 0.067 | 0.076 | 0.056 | 0.105 | 0.105 | 0.050 |
| eval noise S3 | 0.064 | 0.066 | 0.053 | 0.101 | 0.109 | 0.057 |

Monitors (tile_d = fraction of tile pairs at distance 0/1/2; edge_same_top held out):

| | tile_d | mismatch | pal_viol | edge_same_mid | edge_same_top | s/step |
|---|---|---|---|---|---|---|
| oracle | 0.477/0.402/0.121 | 0.3700 | 0.0266 | 0.392 | 0.365 |  |
| untrained | 0.464/0.409/0.127 | 0.3814 | 0.0474 | 0.364 | 0.331 |  |
| S0 | 0.479/0.400/0.121 | 0.3710 | 0.0252 | 0.397 | 0.365 | 0.10 |
| S1 | 0.476/0.403/0.121 | 0.3735 | 0.0272 | 0.391 | 0.356 | 0.19 |
| S2 | 0.476/0.402/0.122 | 0.3730 | 0.0273 | 0.392 | 0.358 | 0.18 |
| S3 | 0.476/0.402/0.122 | 0.3734 | 0.0267 | 0.394 | 0.361 | 0.59 |

mid_h vs BM J d(m, m') zero-meaned (d = 0 / 1 / 2 reference -1.20 / +0.00 / +1.20); learned mean over entries by distance, and max |mid_h - ref|; top_h rows:

| | mid_h d=0 | d=1 | d=2 | max abs err | top_h (rows p = 0..3) |
|---|---|---|---|---|---|
| S0 | -0.31 | +0.02 | +0.26 | 0.94 | -0.27 -0.02 +0.34 -0.06; -0.03 -0.31 -0.03 +0.30; +0.32 -0.00 -0.23 -0.04; -0.00 +0.34 -0.01 -0.31 |
| S1 | -0.29 | +0.02 | +0.26 | 0.95 | -0.18 -0.02 +0.15 +0.02; -0.00 -0.22 +0.03 +0.22; +0.20 +0.02 -0.14 -0.02; +0.01 +0.18 -0.00 -0.24 |
| S2 | -0.28 | +0.02 | +0.25 | 0.96 | -0.22 -0.03 +0.23 -0.00; +0.03 -0.27 -0.03 +0.22; +0.24 +0.05 -0.13 -0.08; -0.01 +0.26 -0.00 -0.25 |
| S3 | -0.30 | +0.02 | +0.25 | 0.96 | -0.20 -0.02 +0.27 -0.04; -0.04 -0.24 -0.02 +0.27; +0.30 -0.00 -0.18 -0.04; +0.06 +0.28 -0.04 -0.35 |
### top_h / top_v: the leak

Contrast of the learned top tables, `disjoint - same` = mean over p of
theta[p, p+2] - theta[p, p] (palettes with no shared colour vs identical),
and `one-colour - same` = mean of theta[p, p+-1] - theta[p, p].  One
training run per scheme; the spread among entries that are equal by
symmetry is about +-0.05.

| setting | scheme | disjoint - same (h / v) | one-colour - same (h / v) | ratio to S0 (disjoint, h+v) |
|---|---|---|---|---|
| kappa 8, J 0.1 | S0 | 0.73 / 0.71 | 0.31 / 0.32 | 1 |
| | S1 | 0.50 / 0.53 | 0.25 / 0.26 | 0.72 |
| | S2 | 0.61 / 0.63 | 0.28 / 0.29 | 0.86 |
| | S3 | 0.67 / 0.70 | 0.30 / 0.29 | 0.95 |
| | S1f | 0.02 / 0.02 | 0.04 / 0.00 | 0.03 |
| | S2rb | -0.02 / -0.03 | -0.05 / -0.01 | -0.03 |
| | S3rb | 0.46 / 0.42 | 0.30 / 0.21 | 0.61 |
| kappa 1, J 0.3 | S0 | 0.60 / 0.58 | 0.26 / 0.26 | 1 |
| | S1 | 0.38 / 0.39 | 0.20 / 0.19 | 0.65 |
| | S2 | 0.45 / 0.47 | 0.21 / 0.24 | 0.78 |
| | S3 | 0.52 / 0.52 | 0.22 / 0.24 | 0.88 |

Timing (s per training step, RUNS = 8 forward runs of 30/30/20 sweeps,
single core): S0 0.10, S1 0.18-0.19, S2 0.18-0.19, S3 0.57-0.59 (K = 3
oracle sweeps at 17 ms each per run), S1f 0.10, S2rb 0.18, S3rb 0.60.
Oracle moments 450 sweeps: 7.5 s.  A whole kappa = 8 setting (7 schemes x
200 steps + evals) took ~6.5 min, kappa = 1 ~3.5 min.

### What the numbers show (disordered-phase runs)

(a) *The top-top coupling is learned by S1.*  At both settings S1 learns
the right shape of top_h / top_v: identical palettes favoured, disjoint
palettes penalised, one shared colour in between, with the same ordering as
S0 — through the collapsed mid conditional and one top step, with no tiles
read at the top.  The mid tables from S1 match `BM J d` at kappa = 8 to
0.03 (max abs error; d = 0 / 1 / 2 learned -0.40 / +0.01 / +0.38 vs -0.40 /
0 / +0.40); S0, S2, S3 match to 0.04-0.05.  At kappa = 1 every scheme learns
mid_h ~ -0.30 / +0.02 / +0.26, well below `BM J d` = -1.2 / 0 / +1.2, as
expected (the kappa -> inf formula does not apply; S0 agrees, so this is
the true induced coupling, not a failure).

(b) *Leak.*  S1's top tables are ~0.7 of S0's (0.72 at kappa 8, 0.65 at
kappa 1); S2 recovers part (0.86 / 0.78), S3 most (0.95 / 0.88); this
ordering S1 < S2 < S3 < S0 holds in both settings and in h and v
separately.  On the held-out `edge_same_top` the leak is visible but small:
oracle .395 / S0 .397 / S1 .381 / S2 .395 / S3 .392 (kappa 8); oracle .365
/ S0 .365 / S1 .356 / S2 .358 / S3 .361 (kappa 1), untrained .248 / .331.
The fitted-feature L1s do not separate S0-S3: all are at the eval-noise
level (0.04-0.13 per key vs noise 0.04-0.12), except S3 at kappa 8, which is
somewhat above (mid_h .10 vs noise .075, top_h .13 vs .09) and whose curve
drifted upward after step ~130 (it was .12 / .18 at step 100); at ETA = 2
the same drift was larger.  I would not call S3 converged-and-better than
S0 on the evidence of one run; the table shape says it is closest to S0.

(c) *Frozen sites.*  At kappa = 8, S1f (mid sites with the plain one-site
conditional, tiles fixed) learns nothing at the mid level (mid_h exactly 0
to two decimals: a mid cell with 16 tiles under it has its current colour
pinned by 16 kappa = 128) and therefore nothing at the top (top_h contrast
0.02); its forward stats equal the untrained ones (edge_same_top .250 vs
.248, mid_h L1 .284 vs .287).  The collapsed conditional is what makes S1
work.

Diagnostics: the literal "RB change at the relaxed state only" fails.
S2rb learns the mid tables (they are the S1 mid gap) but no top coupling
(contrast -0.02; edge_same_top .339): after K top sweeps under p*(T | M) at
fixed relaxed mid the top is at its conditional equilibrium, so the
one-step RB change is ~0 regardless of theta_top.  S3rb learns no mid
tables (mid_h ~0) for the same reason (the collapsed mid moves equilibrate
the mid given the tiles), and a top table at 0.6 of S0 from a forward
whose mid level is untrained (edge_same_top .293).

### The spec's dials (J = 1, ordered phase)

Both kinds of target fail here, differently.  The RB schemes S1, S2, S3 all
end with the forward in a single colour sector chosen by the learned
unaries (top_u ~ +-1.0 favouring the two palettes that share one colour,
e.g. S1 kappa 8 top_u = [-.92 -.95 .86 1.02]); a one-colour state is a
fixed point of the one-site p* kernel, so its RB gap vanishes and the
mid_h coupling stops growing at d = 0 ~ -0.9 (kappa 8) / -0.5 (kappa 1),
far from `BM J d` = -4.  Their monitors match a single oracle chain closely
(edge_same_top .995 / .995 / .998 vs .994 at kappa 8; .965 vs .913 at kappa
1, over-ordered), but the fitted-feature L1 against the symmetrised target
is ~1.5 (all mass on one colour, every run).  S0 against the symmetrised
target stays symmetric but under-orders (edge_same_top .853 vs .994; at
kappa 1 .820 vs .913) and has a worse `pal_viol` than untrained (.110 vs
.047 at kappa 8): the forward cannot produce lattice-wide order from local
top-down sweeps on 12 x 12 mid cells.  S1f again learns nothing.
S2rb / S3rb are partly ordered (edge_same_top .549 / .671).  These runs do
not measure the leak; the top tables are not comparable across schemes.
### kappa = 8, J = 1

L1 distance to the oracle moments on the fitted features (final, 64 forward runs):

| | mid_h | mid_v | mid_u | top_h | top_v | top_u |
|---|---|---|---|---|---|---|
| untrained | 1.279 | 1.271 | 0.018 | 0.492 | 0.568 | 0.045 |
| S0 | 0.196 | 0.207 | 0.122 | 0.205 | 0.209 | 0.168 |
| S1 | 1.496 | 1.496 | 1.492 | 1.234 | 1.233 | 0.986 |
| S2 | 1.497 | 1.497 | 1.492 | 1.236 | 1.240 | 0.990 |
| S3 | 1.498 | 1.498 | 1.496 | 1.235 | 1.235 | 0.989 |
| S1f | 1.283 | 1.272 | 0.012 | 0.511 | 0.599 | 0.018 |
| S2rb | 0.613 | 0.611 | 0.327 | 0.539 | 0.590 | 0.049 |
| S3rb | 1.359 | 1.353 | 1.021 | 0.877 | 0.907 | 0.646 |
| eval noise S0 | 0.201 | 0.208 | 0.182 | 0.155 | 0.164 | 0.101 |
| eval noise S1 | 0.005 | 0.004 | 0.003 | 0.066 | 0.026 | 0.020 |
| eval noise S2 | 0.004 | 0.004 | 0.002 | 0.031 | 0.030 | 0.009 |
| eval noise S3 | 0.004 | 0.002 | 0.002 | 0.055 | 0.029 | 0.018 |
| eval noise S1f | 0.053 | 0.047 | 0.017 | 0.085 | 0.096 | 0.028 |
| eval noise S2rb | 0.090 | 0.099 | 0.077 | 0.088 | 0.076 | 0.033 |
| eval noise S3rb | 0.059 | 0.061 | 0.048 | 0.090 | 0.106 | 0.057 |

Monitors (tile_d = fraction of tile pairs at distance 0/1/2; edge_same_top held out):

| | tile_d | mismatch | pal_viol | edge_same_mid | edge_same_top | s/step |
|---|---|---|---|---|---|---|
| oracle | 0.999/0.001/0.000 | 0.0000 | 0.0000 | 0.996 | 0.994 |  |
| untrained | 0.850/0.118/0.032 | 0.0001 | 0.0474 | 0.359 | 0.248 |  |
| S0 | 0.980/0.012/0.007 | 0.0000 | 0.1100 | 0.917 | 0.853 | 0.05 |
| S1 | 0.999/0.001/0.000 | 0.0000 | 0.0048 | 0.995 | 0.995 | 0.08 |
| S2 | 0.999/0.001/0.000 | 0.0000 | 0.0031 | 0.995 | 0.995 | 0.08 |
| S3 | 0.999/0.001/0.000 | 0.0000 | 0.0048 | 0.997 | 0.998 | 0.28 |
| S1f | 0.850/0.118/0.032 | 0.0001 | 0.0468 | 0.358 | 0.244 | 0.05 |
| S2rb | 0.939/0.048/0.013 | 0.0000 | 0.0245 | 0.742 | 0.549 | 0.09 |
| S3rb | 0.938/0.048/0.013 | 0.0000 | 0.0228 | 0.736 | 0.671 | 0.29 |

mid_h vs BM J d(m, m') zero-meaned (d = 0 / 1 / 2 reference -4.00 / +0.00 / +4.00); learned mean over entries by distance, and max |mid_h - ref|; top_h rows:

| | mid_h d=0 | d=1 | d=2 | max abs err | top_h (rows p = 0..3) |
|---|---|---|---|---|---|
| S0 | -2.70 | +0.89 | +0.91 | 3.18 | -0.72 -0.34 +1.43 -0.34; -0.33 -0.73 -0.28 +1.34; +1.35 -0.39 -0.73 -0.29; -0.28 +1.38 -0.30 -0.76 |
| S1 | -0.86 | +0.24 | +0.38 | 3.67 | -0.58 -0.74 +0.29 +0.27; -0.72 -0.67 +0.31 +0.30; +0.25 +0.27 +0.04 +0.16; +0.31 +0.34 +0.09 +0.08 |
| S2 | -0.83 | +0.25 | +0.33 | 3.74 | +0.03 +0.25 +0.36 +0.11; +0.32 -0.69 -0.77 +0.33; +0.31 -0.75 -0.61 +0.29; +0.13 +0.34 +0.30 +0.06 |
| S3 | -0.61 | +0.21 | +0.18 | 3.88 | -0.66 -0.80 +0.39 +0.29; -0.78 -0.72 +0.38 +0.39; +0.36 +0.29 -0.03 +0.11; +0.32 +0.41 +0.07 -0.02 |
| S1f | +0.00 | -0.00 | +0.00 | 4.00 | +0.07 +0.01 +0.02 -0.04; -0.03 -0.01 -0.01 -0.01; +0.06 -0.01 -0.01 -0.02; -0.03 -0.05 +0.05 -0.00 |
| S2rb | -1.27 | +0.25 | +0.76 | 3.27 | +0.07 -0.13 -0.00 -0.00; +0.00 +0.14 -0.20 +0.10; -0.03 -0.08 +0.12 -0.00; -0.00 +0.05 +0.05 -0.09 |
| S3rb | -0.21 | +0.09 | +0.03 | 3.98 | -0.47 +0.05 +0.27 -0.03; +0.17 -0.36 +0.02 +0.21; +0.16 +0.28 -0.25 +0.17; -0.20 +0.21 +0.21 -0.44 |

### kappa = 1, J = 1

L1 distance to the oracle moments on the fitted features (final, 64 forward runs):

| | mid_h | mid_v | mid_u | top_h | top_v | top_u |
|---|---|---|---|---|---|---|
| untrained | 1.198 | 1.190 | 0.018 | 0.468 | 0.543 | 0.045 |
| S0 | 0.508 | 0.517 | 0.467 | 0.402 | 0.411 | 0.310 |
| S1 | 1.517 | 1.517 | 1.497 | 1.237 | 1.243 | 0.987 |
| S2 | 1.516 | 1.516 | 1.496 | 1.236 | 1.241 | 0.986 |
| S3 | 1.517 | 1.517 | 1.497 | 1.237 | 1.243 | 0.987 |
| eval noise S0 | 0.260 | 0.254 | 0.234 | 0.133 | 0.159 | 0.089 |
| eval noise S1 | 0.001 | 0.001 | 0.000 | 0.055 | 0.027 | 0.025 |
| eval noise S2 | 0.001 | 0.002 | 0.000 | 0.067 | 0.038 | 0.032 |
| eval noise S3 | 0.002 | 0.002 | 0.001 | 0.070 | 0.046 | 0.038 |

Monitors (tile_d = fraction of tile pairs at distance 0/1/2; edge_same_top held out):

| | tile_d | mismatch | pal_viol | edge_same_mid | edge_same_top | s/step |
|---|---|---|---|---|---|---|
| oracle | 0.954/0.045/0.001 | 0.0238 | 0.0003 | 0.934 | 0.913 |  |
| untrained | 0.783/0.196/0.021 | 0.1917 | 0.0474 | 0.570 | 0.508 |  |
| S0 | 0.886/0.101/0.014 | 0.2421 | 0.0482 | 0.855 | 0.820 | 0.04 |
| S1 | 0.965/0.034/0.001 | 0.0199 | 0.0060 | 0.965 | 0.965 | 0.08 |
| S2 | 0.965/0.035/0.001 | 0.0199 | 0.0063 | 0.965 | 0.965 | 0.08 |
| S3 | 0.965/0.034/0.001 | 0.0199 | 0.0060 | 0.965 | 0.965 | 0.28 |

mid_h vs BM J d(m, m') zero-meaned (d = 0 / 1 / 2 reference -4.00 / +0.00 / +4.00); learned mean over entries by distance, and max |mid_h - ref|; top_h rows:

| | mid_h d=0 | d=1 | d=2 | max abs err | top_h (rows p = 0..3) |
|---|---|---|---|---|---|
| S0 | -2.22 | +0.62 | +0.98 | 3.05 | -0.72 -0.30 +1.26 -0.32; -0.25 -0.75 -0.24 +1.29; +1.23 -0.29 -0.73 -0.28; -0.24 +1.30 -0.25 -0.70 |
| S1 | -0.53 | +0.19 | +0.15 | 4.16 | -0.60 +0.26 +0.31 -0.73; +0.27 +0.14 +0.14 +0.24; +0.24 +0.09 +0.17 +0.34; -0.69 +0.31 +0.26 -0.76 |
| S2 | -0.48 | +0.18 | +0.13 | 4.16 | -0.62 +0.26 +0.32 -0.76; +0.31 +0.10 +0.09 +0.30; +0.30 +0.07 +0.13 +0.32; -0.69 +0.29 +0.33 -0.77 |
| S3 | -0.39 | +0.14 | +0.11 | 4.11 | -0.60 +0.27 +0.34 -0.78; +0.27 +0.07 +0.09 +0.33; +0.34 +0.11 +0.11 +0.31; -0.73 +0.34 +0.29 -0.77 |
# Conv potentials: a learned 3 x 3 energy over value embeddings

Follow-up of `notes/induce_test.md`.  The tabular pair fit reproduced the
circles mid level exactly but left a large residual at the top (0.6-0.8
of the target RMS) with only horizontal and vertical pairs.  This note
specifies a learned potential over 8 adjacencies as a new generic factor
kind, fitted by the same window free-energy regression with JAX + Adam,
and the circles test that decides whether a nonlinearity is needed.

## Definition (the one thing both the kernel and the fit implement)

For a coarse channel with domain size D, an embedding `E : (D, k)`, and
the 3 x 3 offsets `d in {-1, 0, 1}^2` indexed `0..8` (row-major, index 4
is the centre), the energy of a grid `z` is

```
E(z) = sum_p [ U(z_p) + B_p(z) + H_p(z) ]
U(z_p)  = a . e(z_p)                                       unary (k,)
B_p(z)  = sum_{d != centre} e(z_p)^T A_d e(z_{p+d})        bilinear, A_d : (k, k), 8 of them
H_p(z)  = v . phi( sum_d W_d e(z_{p+d}) + b )              plaquette head, W_d : (k, m), b, v : (m,)
```

`phi` is softplus; `H` is absent when `m = 0`.  Off-grid neighbours use
the embedding of the `pad` value (the level's reference value) when
`pad >= 0`, else the zero vector (then their bilinear terms vanish and
they add nothing to the head's pre-activation).  Every ordered pair
`(p, p+d)` contributes its own bilinear term; nothing is symmetrised
(`A_{-d}` and `A_d^T` are both learned; the fit may tie them, the kernel
does not care).  No gauge is imposed; the regression target fixes it.

## Generic layer: factor kind `CONVPOT` (`core.py`, `kernel.py`)

`Factor.convpot(a=(chan, view), E, a_vec, A, W, b, v, pad=-1, name)`
with `E : (Vа, k)` indexed by the VIEW value (as every factor is; the
identity view makes it the domain), `A : (9, k, k)` (centre ignored),
`W : (9, k, m)`, `b, v : (m,)`; `m = 0` allowed.  Homed on `chan`,
radius 1 (the plaquette head needs the 3 x 3 windows that contain p,
which read p's neighbours at distance 2; use radius 2 and the
5-colouring).  Packed as rows of kind 5 whose `tab` slot points at a
tuple of these arrays (extend `Packed` with a `convs` tuple and put the
index in the row).

Kernel, candidate energies at site p for every t (added to `out[t]` by
`_energies`): the change-free form
```
out[t] += a . e_t
        + sum_{d} [ e_t^T A_d e(z_{p+d}) + e(z_{p+d})^T A_{-d} e_t ]       (pad rules as above)
        + sum_{q : p in window(q)} v . phi( h_q^{(p -> t)} + b )
```
where `h_q^{(p -> t)}` is q's pre-activation with `z_p` replaced by `t`;
the simple implementation recomputes `h_q` for the nine `q` from scratch
per candidate (cost `9 * 9 * k * m` per candidate, fine for the toys;
the incremental version `h_q + W_{p-q}(e_t - e_{z_p})` is an
optimisation for later).  `total_energy` adds `sum_p [U + B + H]`
exactly as defined.  `Model.compile(home, below=...)` is unchanged for
other kinds; `describe` names the kind.

Tests (`tests/test_convpot.py`): a numpy reference `convpot_energy(grid,
params, pad)` written directly from the definition; `total_energy`
equals it on random grids and params (with and without `pad`, with
`m = 0` and `m > 0`); `site_energies` differences `e[t] - e[t']` equal
the reference's energy differences; a sweep runs under the 5-colouring;
a convpot that encodes a known pair table (`k = D`, one-hot `E`, `A_d` =
the table for `d` = east, zero elsewhere, `m = 0`) gives the same
conditionals as `Factor.pair` with that table.

## The fit (`castlegen/channels/convfit.py`, JAX)

- `params`: dict `{"a", "A", "W", "b", "v"}` with `E` fixed (random) or
  learned (`learn_E=True`).
- `energy_window(params, E, window_values, pad)`: the definition above
  on a small grid, in JAX, vectorised over a batch of windows.  Must
  equal the numpy reference of the kernel tests to 1e-6 (test it:
  `tests/test_convfit.py` imports the same numpy reference).
- `fit(windows, targets, D, k, m, E=None, pad, steps, lr, l2, seed,
  tie=True)`: Adam on `mean (energy_window(z) - energy_window(ref) -
  target)^2 + l2 * |params|^2`, where the reference window is all `pad`;
  `tie=True` constrains `A_{-d} = A_d^T` and `W` unconstrained.  Returns
  params, training curve, residual RMS and target RMS (the diagnostics of
  `induce_test.md`).  `m = 0` is the bilinear model (also provide
  `fit_bilinear_ls`, the closed-form least squares for `m = 0` with
  fixed `E`, as the fast path and the check on Adam).
- `materialise_pairs(params, E, pad) -> (u : (D,), g : (9, D, D))`:
  evaluate the potential's implied unary and pair tables by energies of
  one- and two-cell configurations on a reference background (exact for
  `m = 0`, the pair projection for `m > 0`), for comparison with
  `reference()` and for the `(v, v')` held-out test.
- `to_factor(params, E, view, pad, name) -> Factor.convpot(...)`.

Embeddings (`induce_circles.py` hooks, model-facing):
- `embed_identity(D, k, seed)`: Gaussian rows, one per value (the random
  table-row baseline; cannot generalise across values).
- `embed_patch(P, level, k, seed)`: for `mid`, the object's demand map
  on the block plus its one-tile ring (10 x 10, channels dirt / air)
  flattened and projected by a fixed Gaussian matrix to `k` (the
  painted-view embedding; absent = zero map); for `top`, the 2 x 2 slot
  pattern projected likewise.  The reference value's embedding is
  whatever the projection gives it (not forced to zero).

Windows: `induce.py`'s window machinery with 3 x 3 coarse windows
(needed to identify a 3 x 3 kernel), exterior at the reference value,
exact `F` for the mid level, AIS over the mid channel for the top level
(the mid level carrying the exact tabular tables of `induce_test.md`
stage B).  Also keep the 2 x 2 windows from before for a like-for-like
residual comparison.

## The test on circles

Dials as in `circles_test.md`.  `k = 16` for identity embeddings, `k =
32` for patch embeddings; `m in {0, 32}`.

1. **Top level, the residual question.**  Fit the top potential on 3 x 3
   top windows (N = 2000, AIS `K = 256, M = 4`): (a) tabular h+v pairs
   as in `induce_test.md` (reference point, 0.6-0.8); (b) bilinear
   `m = 0` on 8 offsets, identity embedding; (c) `m = 32`.  Report
   residual RMS / target RMS for each, the materialised `top_h` vs the
   oracle pattern (correlation, contrast as defined in
   `circles_test.md`), and the materialised diagonal tables.  The
   decision rule: if (b) brings the residual under ~0.2 of target, the
   nonlinearity is not needed at the top; if (c) does and (b) does not,
   it is.
2. **Mid level, generalisation across values.**  Hold out 40% of the
   present-present `(v, v')` combinations: draw 3 x 3 windows and reject
   any containing a held-out adjacent pair (any of the 8 offsets).  Fit
   (b) and (c) with identity embeddings and with patch embeddings.
   Score the materialised pair tables on seen vs held-out entries
   against `reference()` (double-centred: max error, slope,
   correlation).  Identity embeddings should fail on held-out entries;
   patch embeddings are the claim under test.
3. **End to end.**  Forward model with convpot factors at both levels
   (best variants from 1 and 2, fitted on all pairs) vs oracle moments:
   fitted keys, monitors, held-out `edge_air_top`, 64 runs, eval noise,
   next to the induce-tables row of `induce_test.md`.  Kernel time per
   forward run with convpot factors vs tables.
4. **Cost.**  Fit time per variant, AIS time, kernel time.

Outputs `images/convpot_circles_*`, JSON `images/convpot_circles.json`,
results appended here.

## Results

Script `notes/experiments/convpot_circles.py` (stages `123`; window targets
cached in `images/convpot_circles_windows.npz`, everything else in
`images/convpot_circles.json`, log `images/convpot_circles_log.txt`),
figures `notes/experiments/convpot_circles_figs.py`.  Hook added to
`circles.py`: `Circles.model(chans, theta, extra=())` and
`Forward(C, theta, seed, extra=())` append extra factors, and
`learned_factors` leaves out any key missing from theta (so `theta = {}` +
two convpot factors replaces all six tables).  Oracle moments are read from
`images/induce_circles.json` (same `Circles(6, 6)`, same code and seed).

### Choices made while running

- Mid tables inside the top windows: exact F, 2 x 2 windows, N = 4000
  (residual 0.0003 / 59.89, as induce stage B).
- Top windows: AIS K = 256, M = 4, N = 2000 uniform corners, exterior and
  reference at NW (= `pad` 0); 32 halos (forward mid kernel under all-NW
  tops), `F(r, halo)` per halo at 4 K.  N was not reduced (AIS 69 s).
- Every top fit is also done on an 80 / 20 split; 'test' = held-out window
  residual.  This matters: (c) has more parameters than windows.
- **Floor.**  The halo makes the target `F(z, h) - F(r, h)` depend on h,
  which no potential of z can see.  Measured on 50 z x 8 halos: within-z sd
  0.914 = **0.54 of the target RMS** (AIS se alone 0.20).  So I also ran
  3 x 3 windows **without halo** (free window boundary, as the "no halo"
  row of `induce_test.md`), whose floor is AIS noise only: 50 z x 4 repeats,
  within-z sd 0.237 = 0.20 of target.
- (b) uses orthonormal rows (D = 4 < k = 16); it is the same model class as
  (b0) and gives identical numbers.  (c) / (c8): m = 32 / 8 from the (b) ls
  solution, Adam 4000 steps, lr 3e-3 -> 3e-5, l2 1e-6; (creg) the same at
  l2 1e-3 (the head is shrunk away: identical to (b)).  (b0hv): the
  8-offset one-hot model with the diagonal A set to zero (h + v + the pad
  terms), to tell diagonals from pad terms.
- Mid: 3 x 3 windows, exact F, presence 0.5.  Held-out set: 40% of the 136
  unordered present-present pairs `{v, v'}` (54, seed 20: hand entries B and
  D held out, A and C seen), symmetric so that a pair is excluded at all 8
  offsets in both orientations; 4000 accepted of 16490 drawn (acceptance
  0.24).  All-pairs fit: 6000 windows.  Ridge l2 = 1e-6 (per-window mean
  loss), plus a sweep.  Scores are in the **reference gauge** (row / column
  absent = 0), in which both `C.reference(centred=False)` and
  `materialise_pairs` on an absent background are exactly defined; mid_h and
  mid_v pooled (2 x 155 seen, 2 x 101 held-out ordered pp entries).
  Double-centring the full table mixes held-out errors into the seen
  entries (e.g. hand A, seen and exact in the reference gauge, comes out
  +1.59 vs +2.27 double-centred for b0), so it is not used for the seen /
  held-out split.

### 1. Top level, the residual question

Residual / target RMS (all = fit on all windows, train / test = the 80 / 20
split); contrast and correlations of the materialised `top_h = g[east]`,
`top_v = g[south]` as in `induce_test.md` (oracle contrast 0.40, reference
0.52).

| windows | variant | rel all | rel train | rel test | contrast h / v | top_h corr / slope vs oracle | top_v corr vs oracle | corr vs reference |
|---|---|---|---|---|---|---|---|---|
| 3x3 halo (target 1.680, floor 0.54) | (a) tabular h+v | 0.558 | 0.561 | 0.567 | 0.34 / 0.29 | 0.91 / 0.83 | 0.86 | 0.98 |
| | (b0) one-hot, 8 offsets | 0.543 | 0.545 | 0.548 | 0.35 / 0.31 | 0.90 / 0.84 | 0.89 | 0.99 |
| | (b0hv) one-hot, no diagonals | 0.551 | 0.554 | 0.547 | 0.35 / 0.32 | 0.89 / 0.84 | 0.90 | 0.98 |
| | (b) orth. k 16 | 0.543 | 0.545 | 0.548 | 0.35 / 0.31 | 0.90 / 0.84 | 0.89 | 0.99 |
| | (c) k 16, m 32 | 0.500 | 0.485 | **0.603** | 0.31 / 0.29 | 0.89 / 0.68 | 0.83 | 0.88 |
| | (c8) k 16, m 8 | 0.509 | 0.498 | 0.606 | 0.38 / 0.35 | 0.90 / 0.91 | 0.89 | 0.97 |
| | (creg) m 32, l2 1e-3 | 0.543 | 0.545 | 0.548 | 0.35 / 0.31 | 0.90 / 0.84 | 0.89 | 0.99 |
| 2x2 halo (target 0.996) | (a) tabular h+v | 0.666 | 0.664 | 0.672 | 0.31 / 0.30 | 0.92 / 0.77 | 0.89 | 0.98 |
| | (b0) one-hot | 0.660 | 0.655 | 0.685 | 0.30 / 0.32 | 0.93 / 0.74 | 0.89 | 0.98 |
| | (b0hv) | 0.666 | 0.663 | 0.681 | 0.30 / 0.31 | 0.93 / 0.75 | 0.90 | 0.98 |
| 3x3 no halo (target 1.179, floor 0.20) | (a) tabular h+v | 0.356 | 0.358 | 0.359 | 0.25 / 0.25 | 0.82 / 0.59 | 0.81 | 0.88 |
| | (b0) one-hot | **0.197** | 0.196 | **0.203** | 0.34 / 0.33 | 0.91 / 0.80 | 0.91 | 0.99 |
| | (b0hv) | 0.200 | 0.200 | 0.202 | 0.34 / 0.33 | 0.91 / 0.80 | 0.91 | 0.99 |
| | (b) orth. k 16 | 0.197 | 0.196 | 0.203 | 0.34 / 0.33 | 0.91 / 0.80 | 0.91 | 0.99 |
| | (c) m 32 | 0.195 | 0.191 | 0.208 | 0.34 / 0.33 | 0.91 / 0.80 | 0.91 | 0.99 |
| | (c8) m 8 | 0.193 | 0.192 | 0.206 | 0.33 / 0.33 | 0.91 / 0.79 | 0.90 | 0.99 |

(2 x 2 with halo: 0.666 for (a) here; `induce_test.md` C had 0.632 / 1.079
= 0.59 at K 128, M 16.  Same picture.)

Materialised tables, double-centred (rows NW NE SW SE = this cell, columns
= the neighbour), 3 x 3 halo windows:

    oracle -log(p/pp) top_h: +0.00 -0.14 +0.13 +0.01; -0.31 +0.00 +0.18 +0.13; +0.13 +0.01 +0.00 -0.14; +0.18 +0.13 -0.31 +0.00
    (b0) top_h:   +0.09 -0.08 +0.07 -0.08; -0.30 +0.07 +0.18 +0.05; +0.07 -0.09 +0.07 -0.06; +0.14 +0.10 -0.32 +0.09
    (b0) top_v:   +0.05 +0.05 -0.04 -0.06; +0.08 +0.07 -0.09 -0.06; -0.26 +0.17 +0.06 +0.04; +0.13 -0.29 +0.08 +0.08
    (b0) diag SE: +0.02 -0.00 +0.03 -0.05; -0.03 +0.01 +0.02 -0.00; +0.03 -0.02 -0.03 +0.03; -0.02 +0.01 -0.02 +0.03   (dc rms 0.026)
    (b0) diag SW: +0.02 -0.01 -0.01 -0.00; -0.07 +0.04 +0.05 -0.02; +0.03 -0.05 -0.01 +0.04; +0.03 +0.02 -0.03 -0.02   (dc rms 0.033)
    (c)  diag SE: +0.03 -0.09 +0.11 -0.05; +0.03 +0.07 +0.00 -0.10; -0.02 -0.11 -0.03 +0.16; -0.04 +0.12 -0.07 -0.01   (dc rms 0.079)
    no halo (b0) diag SE / SW: dc rms 0.012 / 0.011 (all entries |x| <= 0.03)
    for scale: top_h / top_v dc rms 0.12-0.14

Figure `images/convpot_circles_top.png` (rows top_h, top_v, diag SE, diag
SW; columns reference, oracle, (a), (b0), (b), (c), (b0) and (c) no halo).

### 2. Mid level, generalisation across values

Held-out fit (4000 windows without any held-out pair; target RMS 90.3).
'windows w/ held-out' = relative residual on the 4476 all-pairs windows
that contain a held-out pair.  Hand entries in the reference gauge
(reference: A +1.94, B +1.94, C -2.49, D -2.49; * = held out).

| variant | in-sample rel | windows w/ held-out rel | seen max err / slope / corr | held max err / slope / corr | held rms pred / ref | A / B* / C / D* | fit s |
|---|---|---|---|---|---|---|---|
| (b0) one-hot k 17 | 4.7e-06 | 0.007 | 0.00 / 1.000 / 1.000 | 3.18 / 0.034 / -0.056 | 0.16 / 0.80 | +1.94 / -0.15 / -2.49 / -0.14 | 0.2 |
| (b) identity k 32 | 7.4e-07 | 0.011 | 0.00 / 1.000 / 1.000 | 5.43 / 0.036 / -0.007 | 0.90 / 0.80 | +1.94 / -0.82 / -2.49 / +0.17 | 1.5 |
| (c) identity m 32 | 8.0e-05 | 0.011 | 0.01 / 1.000 / 1.000 | 5.44 / 0.037 / -0.006 | 0.90 / 0.80 | +1.93 / -0.84 / -2.49 / +0.17 | 78 |
| (b-patch) k 32 | 1.1e-06 | 0.012 | 0.00 / 1.000 / 1.000 | 4.46 / 0.032 / -0.027 | 0.99 / 0.80 | +1.94 / -0.45 / -2.49 / +1.09 | 1.8 |
| (c-patch) m 32 | 3.2e-06 | 0.012 | 0.00 / 1.000 / 1.000 | 4.45 / 0.032 / -0.027 | 0.99 / 0.80 | +1.94 / -0.45 / -2.49 / +1.09 | 72 |

(b0)'s held-out entries are the ridge prior: `A = 0` on them, which in the
reference gauge reads -0.15 (minus the fitted pad-pair terms; the same for
every held-out entry).

Ridge sweep (m = 0, held-out fit): identity l2 1e-4 / 1e-2 / 1: held corr
-0.007 / 0.013 / 0.033 (seen corr 1.000 / 0.995 / 0.712); patch: -0.028 /
-0.084 / -0.038 (seen corr 1.000 / 0.824 / 0.193).  No ridge makes either
embedding generalise.

All-pairs fit (6000 windows, target RMS 126.9): every variant reproduces
the reference, pp max err 0.000 (b0, b), 0.001 (patch), 0.011 ((c), Adam
not fully back at the ls optimum); in-sample rel <= 8.5e-05; materialised
diagonal (SE) tables exactly 0 (max 0.000 for m = 0): the 3 x 3 mid free
energy is pair additive over h + v only.  Figure
`images/convpot_circles_mid.png` (pp block of mid_h, held-out entries
x-marked; scatter fitted vs reference).

### 3. End to end

Variants: top (b0) on 3 x 3 halo windows, all 2000 (ties (b) and (creg) on
test residual; the automatic argmin first picked (creg), whose dead m = 32
head costs kernel time, so stage 3 was rerun with `TOPV=b03`); mid (b0) on
all pairs.  64 forward runs, seed 555; eval noise vs seed 556.

| | mid_h | mid_v | mid_u | top_h | top_v | top_u | ms / run |
|---|---|---|---|---|---|---|---|
| induce_test D: exact-mid + top(exact-mid), 2x2 tables | 0.176 | 0.165 | 0.111 | 0.136 | 0.164 | 0.040 | |
| tables: exact-mid + top (a, 3x3) | 0.176 | 0.165 | 0.114 | 0.125 | 0.162 | 0.027 | 27.8 |
| mixed: exact-mid tables + top convpot (b0) | 0.168 | 0.154 | 0.096 | 0.151 | 0.154 | 0.046 | 27.1 |
| convpot: mid (b0) + top (b0) | 0.169 | 0.162 | 0.106 | 0.151 | 0.154 | 0.046 | 32.4 |
| mixed, top convpot (b0, no-halo windows) | 0.161 | 0.151 | 0.093 | 0.144 | 0.155 | 0.020 | 27.1 |
| eval noise tables (a, 3x3) | 0.090 | 0.096 | 0.028 | 0.069 | 0.078 | 0.024 | |
| eval noise mixed (b0) | 0.104 | 0.095 | 0.032 | 0.094 | 0.116 | 0.030 | |
| eval noise convpot | 0.116 | 0.092 | 0.028 | 0.094 | 0.116 | 0.030 | |
| eval noise mixed (no halo) | 0.100 | 0.097 | 0.029 | 0.090 | 0.127 | 0.018 | |

Monitors (edge_air_top held out):

| | present | viol | conflict | contact | slot_viol | edge_air_top |
|---|---|---|---|---|---|---|
| oracle | 0.3532 | 0.0201 | 0.0000 | 0.5236 | 0.1127 | 0.4773 |
| tables (a, 3x3) | 0.3146 | 0.0206 | 0.0000 | 0.5954 | 0.0817 | 0.4776 |
| mixed (b0) | 0.3222 | 0.0207 | 0.0000 | 0.6003 | 0.0880 | 0.4793 |
| convpot (b0 + b0) | 0.3174 | 0.0199 | 0.0000 | 0.5945 | 0.0826 | 0.4779 |
| mixed (no halo) | 0.3240 | 0.0207 | 0.0000 | 0.5989 | 0.0890 | 0.4803 |

Kernel time per sweep (single core, 6 x 6 tops / 12 x 12 mids): tables top
0.011 ms, mid 0.090 ms; convpot m = 0 top 0.021 ms, mid 0.279 ms; convpot
m = 32 top (the (creg) run) 0.678 ms.  Per forward run (30 + 30 + 20
sweeps) the tile level dominates: 27-32 ms either way.  Renders
`images/convpot_circles_fwd_{tables,mixed,convpot,mixednohalo}.png`.

### 4. Cost

AIS: 3 x 3 halo windows 69 s (34 ms / window) + `F(r, halo)` 5 s; floor
(400 windows) 14 s; 2 x 2 38 s (19 ms) + 3 s; 3 x 3 no halo 65 s (32 ms) +
floor 6 s.  Mid exact F (10000 windows of 3 x 3): 0.7 s.  Fits: closed-form
ls 0.0-0.1 s (top), 0.2-2 s (mid, k 32: 4128 features); Adam top m 32
44-48 s, m 8 13-14 s (4000 steps, 2000 windows); mid m 32 72-134 s (3000
steps, 4000-6000 windows).  Forward evals ~2 s per configuration (64 + 64
runs).  Stages 1-3 total ~15 min wall (including reruns of the fits).

### What the numbers show

- **The top residual is the halo, not missing structure.**  With the halo,
  every model sits at the floor: (a) 0.567, (b0) 0.548 test vs floor 0.54.
  Without the halo (floor 0.20) the 8-offset bilinear model reaches the
  floor, 0.197 in-sample / 0.203 test, while tabular h + v stays at 0.36.
  The gain over (a) is **not** in the diagonals: (b0hv) with the diagonal A
  zeroed gives 0.200, and the materialised diagonal tables are 0.01 dc rms
  (no halo) / 0.03 (halo) vs 0.13 for h / v.  What (a) lacks is the pair
  terms with the exterior (`pad`) cells, i.e. per-value edge corrections a
  free window boundary needs; with the halo these are swamped by the
  halo-dependent part of the target.  The 0.6-0.8 residual of
  `induce_test.md` was halo variance (plus 2 x 2 edge effects), not
  non-additivity.
- **Decision (the note's rule): no nonlinearity at the top.**  (b) is at
  the 0.20 floor on no-halo windows; (c) m = 32 / 8 does not improve test
  residual (0.208 / 0.206 vs 0.203 no halo; 0.603 / 0.606 vs 0.548 with
  halo, i.e. it overfits the halo noise, train 0.485) and its materialised
  tables get noisier (diag dc rms 0.08, top_h slope 0.68).
- **The materialised top tables** have the oracle's shape: corr 0.90-0.91,
  slope 0.80-0.84, contrast 0.34-0.35 / 0.31-0.33 (oracle 0.40) for (b0)
  both with and without halo; (a) on no-halo windows is worse (slope 0.59,
  contrast 0.25: the missing edge terms leak into the pair table).
- **Mid generalisation fails for both embeddings.**  Seen entries are exact
  for all variants; held-out entries have corr -0.06 to +0.03 with the
  reference for one-hot, identity and patch alike, at any ridge.  Reason:
  with D = 17 and the absent row zero, the patch embedding has rank 16 = the
  number of present values, so `e_v^T A e_w` is a reparametrisation of the
  full table and the only inductive bias is the ridge geometry of `A` in
  the projected space, which does not encode "two demand maps overlap
  after a block shift" (the true pair term is bilinear in the
  *unprojected, shifted* maps, but a random 32-dim projection followed by
  an unconstrained k x k A does not prefer it).  The head adds nothing
  (c = b).  The window-level residual on windows containing held-out pairs
  is only ~1% because the targets (RMS 90-127) are dominated by the
  unaries; table scores are the right test here.
- **End to end** convpot matches the tabular induced model: mid stats
  within noise of the tables (mid_h 0.169 vs 0.176, noise ~0.09-0.12),
  top_h 0.151 vs 0.125-0.136 and top_v 0.154 vs 0.162-0.164 (differences
  within about 1.5x eval noise; top noise 0.07-0.12).  The same edge /
  presence deficit as `induce_test.md` (present 0.32 vs 0.35, slot_viol
  0.08 vs 0.11); held-out edge_air_top 0.478 (oracle 0.477).  No
  improvement from the convpot because on circles the tables were already
  the right model.
- **Cost.**  An m = 0 convpot is 2x (top) to 3x (mid) a pair table per
  sweep and invisible per run; an m = 32 head is 60x a table at the top
  (0.68 vs 0.011 ms / sweep) with the simple non-incremental head kernel.

### Things that looked wrong / caveats

- Adam on the mid (c) all-pairs fit did not get back to the ls optimum it
  started from (pp max err 0.011, rel 8.5e-05 vs 5e-07): the head's random
  init perturbs it and 3000 steps at lr 1e-3 do not recover; harmless here.
- The 'best by test residual' choice for the top is a tie between (b0),
  (b), (creg) to 3 decimals; picking (creg) silently costs 30x kernel time
  for a dead head.  Prefer the m = 0 variant on ties.
- The (b0) held-out entries are the ridge prior only up to the reference
  gauge shift (-0.15 here); they are not exactly 0 after
  `materialise_pairs`.

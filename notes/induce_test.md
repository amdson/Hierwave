# Induced potentials: the combining algorithm, tested on circles

The project is a DSL for simple HMRFs (channel sets) plus an algorithm for
combining them.  Sets interact only by competing for shared children, so
combining them correctly means one thing: for every coarse channel,
estimate the free energy the levels below induce on it and hand it to the
forward kernel as a potential (history/channels.tex, "Direction, and the free
energy of the level below").  This note specifies that estimator as a
generic algorithm, `castlegen/channels/induce.py`, and its test on the
circles toy (`notes/circles_test.md`), where the exact answer is known.

No moment matching anywhere.  The coarse values are always clamped, so no
coarse site ever has to move (the frozen-site problem of self-play does
not arise), and no pair has to occur in a sample to be priced (the D^2
sample-count problem does not arise).

## The algorithm (`induce.py`)

Inputs: a model with a coarse channel `c` and the channel(s) below it,
`decode` (the painters), a reference value `r` of `c` (absent / corner 0),
a window shape (coarse cells), a free-energy estimator, a potential
parametrisation.  Output: unary and pair tables (per adjacency direction)
over `c`'s values for the forward kernel, plus the regression residual.

1. **Windows.**  Draw coarse configurations `z_W` on a small window of
   `c` (2 x 2 cells to start; 3 x 3 later), values from a proposal
   distribution: uniform over values with presence tilted toward the
   forward model's current marginal, or windows cut from forward samples
   (mix both; record which).  The exterior of the window is at the
   reference value `r`, and the fine region is the window's blocks plus
   whatever the painters spill outside (one tile ring here).  Fine cells
   outside the fine region are at the reference fine distribution (free
   tiles under the field; for coupled tiles: fixed to a sample under
   reference coarse values, i.e. a fine halo).
2. **Free energy per window.**  `F(z_W) = -log Z_fine(z_W)`, the fine
   level integrated out on the fine region with the coarse window
   clamped, by one of:
   - `exact`: a model hook (`Circles.window_free_energy`, closed form per
     tile; Potts: transfer matrix), for validation;
   - `ais`: generic annealed importance sampling on the fine Model of the
     window: reference distribution `p_0` = the fine channel's unaries
     only (independent sites, `log Z_0` in closed form), target `p_1` =
     the full fine conditional; interpolate `E_beta = E_unary + beta *
     E_rest` with `E_rest` the pair / honour factors; hard (`inf`)
     entries are replaced by a large finite `L` (30) for the anneal, so
     `Z_1` is the hard `Z` up to `e^-L` per potential violation.  `K`
     beta steps (geometric or linear), one generic-kernel sweep at each,
     `log w += -(beta_{k+1} - beta_k) * E_rest(state)`; `M` independent
     chains; `log Z_1 = log Z_0 + logmeanexp(log w)`.  Report the
     estimator's spread across chains as its error bar.  The budget `K *
     M` is the knob; the kernel is the forward model's own fine kernel,
     so this is "the free energy as realised by that kernel" when `K` is
     small.
   The target for regression is `F(z_W) - F(r_W)` (the all-reference
   window), which cancels the exterior's own terms; what the exterior
   contributes through the boundary is a per-value constant and lands
   in the unaries.
3. **Fit the additive potential.**  `F(z_W) - F(r_W) ~ sum_cells u(z_p) +
   sum_adjacent g_dir(z_p, z_q)` over the window's cells and internal
   adjacencies.  Parametrisations:
   - `tables`: `u` is a (D,) vector, `g_h`, `g_v` are (D, D); features are
     counts, the fit is ridge regression (closed form).  Entries never
     seen in any window get the ridge prior (0).
   - `net` (second stage, not in this test): `u = u_phi(decode(v))`,
     `g = g_phi(decode(v), decode(v'), dir)`, fitted by least squares with
     autodiff; same targets.
   The residual of the fit is the part of `F` that is not additive over
   cells and adjacencies (triples and beyond): report its RMS against
   the target's RMS.  That is the pair-truncation check.
4. **Materialise** the tables into the forward model's learned factors
   (`mid_u`, `mid_h`, `mid_v` here), zero-mean gauge as before.
5. **Recurse upward.**  For the next coarse channel (`top`), the "fine
   level" is `mid` with its learned tables and the designed `lam`
   honour; `exact` is unavailable, `ais` runs over the mid channel of
   the window (2 x 2 top cells = 4 x 4 mid cells; `p_0` = mid unaries,
   i.e. presence bonus + learned `mid_u`; anneal `lam` and the learned
   pair tables).  Reference top value: corner 0.
6. **Iterate** once more with windows drawn from the forward model now
   that it has tables (the halo / proposal dependence), and report how
   much the tables moved.

## The test on circles

Dials as in circles (kappa 4, mu 0.3, lam 3, presence bonus default),
`Circles(6, 6)` for the forward model and the oracle.

A. **Estimator check.**  On 200 random 2 x 2 mid windows, `ais` vs
   `exact` `F(z_W) - F(r_W)`: mean error and spread as a function of the
   budget (`K` in {8, 16, 32, 64}, `M` in {4, 16}).  Also one check of
   the `inf -> L` replacement does not matter here (circles has no hard
   factor; set `L` anyway).
B. **Mid tables from exact `F`.**  Fit `mid_u`, `mid_h`, `mid_v` by
   ridge from N = 4000 windows (uniform values, presence 0.5).  Compare
   to `Circles.reference()` double-centred: max abs error, slope,
   correlation, the four hand entries (conflict / shared ring pairs).
   Residual RMS vs target RMS.  Then the same from `ais` at the budget A
   picks.
C. **Top tables.**  With the mid tables materialised, fit `top_u`,
   `top_h`, `top_v` by `ais` over the mid channel on 2 x 2 top windows
   (N = 2000).  Compare to the oracle's top pattern (the leak
   measurement of `circles_test.md`: NE->NW / SE->SW enhanced, NE->SW /
   SE->NW depleted, NW->NE enhanced) and to `reference()['top_h']`.
D. **End to end.**  Forward model with the fitted tables (no moment
   matching) vs oracle moments: the six fitted keys, all monitors
   including the held-out `edge_air_top`, 64 runs, eval noise as before.
   Compare with the best self-play row of `circles_test.md`.
E. **Iteration.**  Redraw windows from forward samples, refit, report the
   table movement and the end-to-end change.
F. **Cost.**  Wall time per stage, single core.

Outputs `images/induce_circles_*` (tables figure with reference / exact-fit
/ ais-fit / top vs oracle, estimator-error figure, renders), JSON
`images/induce_circles.json`, results appended here.

## Results

Code: `castlegen/channels/induce.py` (generic: `ais_log_z`, `ais_se`,
`windows`, `fit_tables`, `materialise`), `castlegen/channels/induce_circles.py`
(the circles hooks `MidWindows`, `TopWindows`, `reference_halos`), tests
`tests/channels/test_induce.py`.  Run: `notes/experiments/induce_circles.py` (stages A-E,
output `images/induce_circles.json`, log `images/induce_circles_log.txt`),
`notes/experiments/induce_circles_report.py` (these tables, figures),
`notes/experiments/induce_circles_edge.py` (the edge diagnostic below).
Figures: `images/induce_circles_estimator.png` (A), `images/induce_circles_tables.png`
(reference / oracle / exact-F fit / ais-F fit / iterated, mid_h and top_h),
renders `images/induce_circles_oracle.png`, `images/induce_circles_fwd_exact_exact.png`,
`_fwd_exact_exact_nohalo.png`, `_fwd_ais_ais.png`, `_fwd_iter.png`.

Implementation choices, where the spec left room or I departed from it:

- **Reference gauge in the fit.**  `fit_tables` drops the pair features with
  a reference value on either side, so `g(r, .) = g(., r) = 0` (then each
  table is zero-meaned).  Without it the 2 x 2 fit is not identified in a
  way that matters: a cell has one horizontal adjacency inside the window
  and two in the bulk, so `g_h(a, b) += s(a) + s(b)`, `u -= s` leaves every
  window target unchanged but changes the bulk energy by `s` per cell.  In
  the reference gauge `u(a) = F(a, rest r) - F(r)` and `g` is the ANOVA pair
  term against `r`, which is what the bulk needs (exact when there are no
  triples).  Signature `fit_tables(wv, targets, D, ridge, ref=0)`.
- **AIS details.**  `Z` is over the free cells of `home` with
  `E = Model.energy(home)` (fixed cells' terms are constants inside it);
  `log Z_0` closed form from the packed unary rows; the anneal is done by
  scaling the packed tables (one compile per window, a table tuple per
  beta), `E_rest` by `kernel.total_energy` with the unary tables zeroed.
  Order per step: sweep at `beta_k`, then `log w -= (beta_{k+1} - beta_k)
  E_rest`.  Linear schedule throughout (geometric implemented, not run).
  Error bar: delta-method SE of `log mean w`.
- **Budget picked by A: K = 256, M = 4** (outside the listed grid): at the
  same cost as K = 64, M = 16 (41 vs 40 ms / window) its error RMS is 0.45
  vs 0.63 and its bias +0.09 vs +0.17.  Top windows: K = 128, M = 16
  (16 free sites, mean se 0.10).
- **Top windows.**  Fine region = the 4 x 4 mid cells of the 2 x 2 top
  window; the halo (a one-mid-cell ring, the reach of the mid pair tables)
  fixed to a sample of the forward mid kernel under all-NW tops, as the
  note says.  32 halo samples, each window draws one, `F(r_W, halo)` per
  halo by AIS at 4x K, target `F(z_W, halo) - F(r_W, halo)`.  A no-halo
  variant (window edge = free boundary) is reported next to it.
- **Mid windows.**  2 x 2 mid cells, the fine region their 18 x 18 tiles
  (blocks + one-tile ring, `dem` painted and fixed); exact = `sum fz[dem]`.
- **Test (4)** runs at K = 256, M = 16, not K = 32, M = 16: at K = 32 only
  80% of windows are within 3 SE (row K 32 M 16 of A), and the 5 test
  windows passed for 3 of 6 seeds.  At K = 256 all 8 seeds tried pass.
- **E was cut**: 2000 mid windows (1000 forward-cut + 1000 uniform) instead
  of 4000, to stay in the time budget; the top windows had 512 forward-cut
  (a loop bound, not intended: the forward-cut top list stopped filling when
  the mid list was full) + 1488 uniform.

### A. Estimator check (200 windows, 2 x 2 mid, presence 0.5)

Error = AIS minus exact `F(z_W) - F(r_W)` (nats; `F(r_W)` by AIS is exact here, no `E_rest`); 'se' = the delta-method standard error the estimator reports; 'within 3 se' = fraction of windows with |error| <= 3 se.  Mean error by number of objects in the window (1 / 2 / 3 / 4).

| K | M | mean err | sd | rms | mean se | within 3 se | mean err by #objects 1 / 2 / 3 / 4 | ms / window |
|---|---|---|---|---|---|---|---|---|
| 8 | 4 | +5.976 | 4.732 | 7.623 | 0.774 | 0.28 | +2.13 / +5.64 / +9.69 / +15.23 | 1.4 |
| 16 | 4 | +2.601 | 2.500 | 3.608 | 0.702 | 0.45 | +0.92 / +2.69 / +4.09 / +5.33 | 2.6 |
| 32 | 4 | +1.240 | 1.462 | 1.917 | 0.609 | 0.65 | +0.53 / +1.38 / +1.91 / +1.30 | 5.1 |
| 64 | 4 | +0.480 | 0.936 | 1.052 | 0.525 | 0.84 | +0.23 / +0.41 / +0.71 / +1.83 | 10.2 |
| 8 | 16 | +4.552 | 3.471 | 5.724 | 0.705 | 0.29 | +1.37 / +4.54 / +7.16 / +12.91 | 5.3 |
| 16 | 16 | +1.573 | 1.939 | 2.497 | 0.607 | 0.60 | +0.48 / +1.31 / +2.69 / +5.20 | 10.2 |
| 32 | 16 | +0.494 | 0.934 | 1.056 | 0.508 | 0.80 | +0.11 / +0.39 / +0.90 / +1.62 | 20.6 |
| 64 | 16 | +0.167 | 0.611 | 0.634 | 0.365 | 0.90 | +0.01 / +0.23 / +0.29 / -0.13 | 40.2 |
| 128 | 16 | +0.072 | 0.321 | 0.329 | 0.261 | 0.96 | +0.05 / +0.04 / +0.11 / +0.38 | 80.4 |
| 256 | 16 | +0.022 | 0.224 | 0.225 | 0.186 | 0.98 | +0.01 / +0.02 / +0.04 / +0.03 | 161.4 |
| 256 | 4 | +0.088 | 0.446 | 0.454 | 0.292 | 0.90 | +0.05 / +0.11 / +0.12 / -0.01 | 41.0 |

Windows with 0 objects: 10 (error exactly 0).  `L` = 30 vs 5 with the same seed: max |dF| = 0.0e+00 (circles has no inf entry).

### B. Mid tables

N = 4000 windows, uniform offsets, presence 0.5, ridge 1e-3, reference gauge (row / column 'absent' of the pair tables 0).  Residual RMS / target RMS: exact 0.0003 / 59.89; ais (K = 256, M = 4) 0.369 / 59.99 (per-window AIS error RMS 0.431, mean se 0.300).  Wall: exact 0.1 s, ais 565 s.

| tables | mid_h max abs err | mid_h pp slope / corr | mid_v pp slope / corr | mean attract / repel / zero (ref -1.37 / +2.37 / +0.11) | A (+2.27) | B (+2.50) | C (-2.01) | D (-2.01) | mid_u present - absent / spread (F_obj 26.95 / 0) | mid_u + pres max err |
|---|---|---|---|---|---|---|---|---|---|---|
| exact F | 0.00 | 1.000 / 1.000 | 1.000 / 1.000 | -1.37 / +2.37 / +0.11 | +2.27 | +2.50 | -2.01 | -2.01 | 26.95 / 0.00 | 0.00 |
| ais F (K 256, M 4) | 0.68 | 1.001 / 0.965 | 0.971 / 0.963 | -1.35 / +2.40 / +0.10 | +2.19 | +2.44 | -2.03 | -2.02 | 27.00 / 0.03 | 0.06 |
| ais F, iterated (E) | 2.74 | 0.791 / 0.796 | 0.971 / 0.869 | -1.17 / +1.42 / +0.12 | +0.79 | +0.87 | -1.55 | -1.68 | 26.99 / 0.05 | 0.12 |

For comparison (circles_test.md): self-play fwdscale S0 max err 1.36, pp slope / corr 1.17 / 0.91; eta0.5 S1 2.25, 0.16 / 0.84.

### C. Top tables

N = 2000 windows of 2 x 2 top cells (uniform corners), reference corner 0 (NW), `ais` over the window's 4 x 4 mid cells.  'halo': the one-mid-cell ring around them fixed to a sample of the forward mid kernel under all-NW tops (32 halo samples, `F(r_W)` per halo by AIS at 4x K); 'no halo': the ring left out (free boundary).  Contrast = mean{NE->SW, SE->NW} - mean{NE->NW, SE->SW, NW->NE, SW->SE} of the double-centred table (top_v through NE <-> SW); oracle `-log(p/pp)` contrast 0.40, reference 0.52.

| mid tables used | halo | residual / target RMS | mean AIS se | contrast h / v | top_h corr / slope vs oracle | top_v corr vs oracle | corr vs reference | wall s |
|---|---|---|---|---|---|---|---|---|
| exact | yes | 0.632 / 1.079 | 0.106 | 0.28 / 0.36 | 0.91 / 0.74 | 0.89 | 0.96 | 70 |
| exact | no | 0.247 / 0.505 | 0.103 | 0.28 / 0.28 | 0.84 / 0.68 | 0.84 | 0.85 | 61 |
| ais | yes | 0.673 / 1.106 | 0.104 | 0.36 / 0.38 | 0.87 / 0.79 | 0.90 | 0.98 | 69 |
| ais, iterated (E) | yes | 0.821 / 1.083 | 0.106 | 0.35 / 0.34 | 0.85 / 0.77 | 0.95 | 0.95 | 70 |

Double-centred top_h (rows T = NW NE SW SE):

    oracle -log(p/pp): +0.00 -0.14 +0.13 +0.01; -0.31 +0.00 +0.18 +0.13; +0.13 +0.01 +0.00 -0.14; +0.18 +0.13 -0.31 +0.00
    reference:         +0.09 -0.09 +0.09 -0.09; -0.43 +0.09 +0.26 +0.09; +0.09 -0.09 +0.09 -0.09; +0.26 +0.09 -0.43 +0.09
    exact              +0.06 -0.11 +0.11 -0.06; -0.24 +0.08 +0.09 +0.08; +0.05 -0.08 +0.09 -0.06; +0.13 +0.10 -0.29 +0.05
    exact_nohalo       +0.04 -0.18 +0.17 -0.03; -0.15 +0.05 +0.11 -0.02; -0.01 +0.01 +0.03 -0.03; +0.12 +0.12 -0.32 +0.08
    ais                +0.10 -0.09 +0.06 -0.06; -0.27 +0.07 +0.16 +0.05; -0.01 -0.03 +0.11 -0.07; +0.18 +0.06 -0.33 +0.08
    iterated (E)       +0.08 -0.08 +0.09 -0.09; -0.32 +0.12 +0.12 +0.08; -0.00 -0.04 +0.05 -0.01; +0.24 +0.00 -0.26 +0.02

Self-play for comparison (circles_test.md): eta0.5 S0 contrast 0.34 / 0.27, corr / slope 0.99 / 0.85; fwdscale S1 0.30 / 0.24, 0.88 / 0.73.

### D. End to end (no moment matching)

L1 to the oracle moments on the fitted features (64 forward runs; oracle 50 + 400 sweeps, symmetrised; same seeds and conventions as circles_test.md):

| | mid_h | mid_v | mid_u | top_h | top_v | top_u |
|---|---|---|---|---|---|---|
| untrained | 1.784 | 1.788 | 1.294 | 0.187 | 0.183 | 0.045 |
| exact-mid + top(exact-mid) | 0.176 | 0.165 | 0.111 | 0.136 | 0.164 | 0.040 |
| exact-mid + top(exact-mid, no halo) | 0.175 | 0.164 | 0.115 | 0.163 | 0.171 | 0.054 |
| ais-mid + top(ais-mid) | 0.178 | 0.180 | 0.122 | 0.148 | 0.161 | 0.040 |
| iterated (E): ais-mid + top, forward windows | 0.166 | 0.173 | 0.102 | 0.150 | 0.154 | 0.036 |
| eval noise exact-mid + top(exact-mid) | 0.092 | 0.094 | 0.025 | 0.074 | 0.092 | 0.032 |
| eval noise exact-mid + top(exact-mid, no halo) | 0.091 | 0.092 | 0.026 | 0.093 | 0.085 | 0.029 |
| eval noise ais-mid + top(ais-mid) | 0.086 | 0.084 | 0.017 | 0.081 | 0.082 | 0.033 |
| eval noise iterated (E): ais-mid + top, forward windows | 0.099 | 0.102 | 0.025 | 0.087 | 0.078 | 0.030 |
| self-play eta0.5 S0 | 0.225 | 0.245 | 0.067 | 0.072 | 0.093 | 0.036 |
| self-play eta0.5 S1 | 0.190 | 0.169 | 0.043 | 0.150 | 0.136 | 0.016 |
| self-play eta0.5 S2 | 0.178 | 0.165 | 0.042 | 0.128 | 0.103 | 0.042 |
| self-play fwdscale S0 | 0.151 | 0.158 | 0.131 | 0.141 | 0.122 | 0.063 |

Monitors (edge_air_top held out):

| | present | viol | conflict | contact | slot_viol | edge_air_top |
|---|---|---|---|---|---|---|
| oracle | 0.3532 | 0.0201 | 0.0000 | 0.5236 | 0.1127 | 0.4773 |
| untrained | 1.0000 | 0.0243 | 0.0022 | 0.3522 | 0.7500 | 0.5719 |
| exact-mid + top(exact-mid) | 0.3153 | 0.0207 | 0.0000 | 0.5944 | 0.0833 | 0.4770 |
| exact-mid + top(exact-mid, no halo) | 0.3139 | 0.0209 | 0.0000 | 0.5959 | 0.0826 | 0.4768 |
| ais-mid + top(ais-mid) | 0.3129 | 0.0211 | 0.0000 | 0.5973 | 0.0801 | 0.4776 |
| iterated (E): ais-mid + top, forward windows | 0.3226 | 0.0206 | 0.0000 | 0.5861 | 0.0884 | 0.4786 |
| self-play eta0.5 S0 | 0.3854 | 0.0237 | 0.0006 | 0.4376 | 0.1769 | 0.4794 |
| self-play eta0.5 S1 | 0.3613 | 0.0226 | 0.0004 | 0.4822 | 0.1237 | 0.4795 |
| self-play eta0.5 S2 | 0.3613 | 0.0228 | 0.0004 | 0.4862 | 0.1224 | 0.4784 |
| self-play fwdscale S0 | 0.4187 | 0.0201 | 0.0000 | 0.4799 | 0.1846 | 0.4763 |

### E. Iteration

Windows: mid 1000 cut from forward samples of the D 'ais-mid + top(ais-mid)' model (presence 0.327) + 1000 uniform; top 512 forward-cut + 1488 uniform; halos regenerated with the refit mid tables.  Mid residual / target RMS 0.291 / 51.23.  Table movement against D's tables (double-centred pairs, centred unaries):

| table | max abs change | rms change | rms of the D table |
|---|---|---|---|
| mid_h | 2.553 | 0.482 | 0.755 |
| mid_v | 1.706 | 0.454 | 0.734 |
| mid_u | 0.091 | 0.041 | 6.353 |
| top_h | 0.061 | 0.044 | 0.137 |
| top_v | 0.071 | 0.040 | 0.141 |
| top_u | 0.083 | 0.057 | 0.269 |

Edge diagnostic (`induce_circles_edge.py`; oracle 50 + 400 sweeps, forward
64 runs of the D exact-mid + top model): present on edge mid cells / interior
/ all: oracle 0.533 / 0.274 / 0.353, forward 0.299 / 0.323 / 0.315.

### Timing (F), single core

A 77 s (all 11 budgets, 200 windows each); B exact 0.1 s, B ais 565 s (4000
windows at 41 ms nominal: the machine was loaded during that run, 48% CPU;
the same budget ran at 41 ms / window in E, so ~165 s unloaded); C 61-70 s
per top fit (2000 windows + 32 halos); D 3 s per forward configuration (64 +
64 runs), oracle moments 6.4 s; E 82 s mid + 70 s top + 6 s eval.  Total
~17 min wall.  Tests 5 s.

### What the numbers show

- *Estimator.*  AIS is biased upward in F (Z underestimated), the bias
  grows with the number of objects (demanded tiles) and falls with K;
  M helps much less than K at equal cost.  The reported SE is too small at
  small K (28-65% within 3 SE for K <= 32); at K >= 128 it is about right
  (96-98%).
- *Mid tables.*  From exact F the 2 x 2 fit reproduces `reference()` to
  1e-3 (residual 3e-4 of 60 nats): the circles mid free energy is pair
  additive on a 2 x 2 window, with no diagonal or triple terms, so the
  pair truncation is exact here.  From AIS (K 256, M 4, 4000 windows) the
  pp block has slope 1.00, corr 0.965, max error 0.68, hand entries within
  0.08, mid_u gap 27.00 vs F_obj 26.95.  This beats every self-play run
  (best: fwdscale S0 slope 1.17, corr 0.91, max err 1.36).
- *Top tables.*  The fit is far from pair additive at the top: residual
  RMS is 59-76% of the target RMS (AIS se is only 0.10, so this is real
  non-additivity: triples through the mid level, and the position-dependent
  halo terms the 2 x 2 window cannot express).  Even so the fitted top_h
  has the oracle's shape: NE->NW and SE->SW the strongest attraction,
  NE->SW / SE->NW repelled; corr 0.84-0.91 with the oracle's
  `-log(p/pp)`, slope 0.68-0.79, contrast 0.28-0.38 (oracle 0.40), i.e.
  between self-play fwdscale S1 and eta0.5 S0.  NW->NE (-0.14 in the
  oracle) comes out -0.09 to -0.11 with the halo, -0.18 without.  The
  halo raises the correlation with the oracle (0.91 vs 0.84) and the
  vertical contrast.
- *End to end.*  With no moment matching the forward model is at the
  self-play level on the mid pair stats (mid_h L1 0.166-0.178 vs best
  self-play 0.151-0.178, eval noise ~0.09).  top_h 0.136-0.163 is at the
  self-play S1 / fwdscale level, worse than eta0.5 S0 (0.072); top_v
  0.154-0.171 is worse than every self-play row (0.093-0.136; noise ~0.08).
  mid_u is worse (0.10-0.12 vs 0.04-0.07): present is 0.313-0.323 vs the
  oracle's 0.353, contact 0.59 vs 0.52, slot_viol 0.08 vs 0.11.  Most of
  the presence gap is the grid edge: the induced mid_u is the interior one,
  while in p* an object whose ring spills off the grid is cheaper, so the
  oracle has 0.53 present on edge mid cells against the forward's 0.30.
  In the interior the forward has too many objects (0.32 vs 0.27); I did
  not find why (the top tables' residual is the candidate: slot_viol and
  contact are both off in the direction of more objects at their corners).
  The held-out edge_air_top is 0.477-0.479 (oracle 0.477).
- *Iteration.*  Refitting on forward-cut windows moved the top tables by
  0.04-0.06 rms (tables ~0.14 rms) and the end-to-end stats within eval
  noise.  The mid pair tables got worse (pp slope 0.79, hand entries A, B
  +0.79 / +0.87 vs +2.27 / +2.50): forward samples almost never contain a
  conflicting pair, and with 1000 uniform windows instead of 4000 many pp
  entries are seen ~2 times or never (unseen entries go to the ridge prior
  0).  That is a coverage effect of the cut and the proposal, not a
  proposal dependence of F: the circles mid F has none (exact and pair
  additive), so here the iteration can only lose.  The mixing proposal
  needs the uniform part at full size.

# Induced potentials: the combining algorithm, tested on circles

The project is a DSL for simple HMRFs (channel sets) plus an algorithm for
combining them.  Sets interact only by competing for shared children, so
combining them correctly means one thing: for every coarse channel,
estimate the free energy the levels below induce on it and hand it to the
forward kernel as a potential (channels.tex, "Direction, and the free
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

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

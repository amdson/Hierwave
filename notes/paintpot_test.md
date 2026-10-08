# Paint potentials: the induced free energy as a local energy over the painted map

Follow-up of `notes/convpot_test.md`, whose per-value embeddings could not
generalise to held-out value pairs: a random projection of a value's
patch in its own block frame cannot express the overlap of two shifted
demand maps, which is what the induced coupling is.  This note places the
learned potential where the free energy lives: on the fine lattice, as a
local energy over the painted channels.

Any coarse value is a painted map, so a potential that only sees painted
tiles generalises across values by construction, and the painter's
composition rule (conflicts, shared rings) is inside the input rather
than something the potential has to learn.  Linear for now: tables over
painted values; the pair tables over painted cells are where a
nonlinearity would go later.

## Definition

Painted channel(s) `c` at fine resolution with values `0..V-1` (circles
mid: `dem`, V = 4 (free / dirt / air / conflict); circles top: `slot` at
mid resolution, V = 2).  The paint potential is

```
F_theta(paint) = sum_t u(paint_t) + sum_{d in OFF} sum_t g_d(paint_t, paint_{t+d})
```

with `OFF` the 4 independent offsets of the 8-neighbourhood (east, south,
south-east, south-west), `u : (V,)`, `g_d : (V, V)`; off-grid partners
contribute nothing.  Linear in `theta = (u, g)`, so the fit is ridge
regression on count features.  Optional radius-2 offsets (`OFF2`) as a
flag, for the top level.

## Fit (generic, `castlegen/channels/paintpot.py`)

- `features(paint, V, offsets) -> (V + len(offsets) * V * V,)` counts.
- Windows as in `induce.py`, WITH the halo cells as part of the painted
  region: the regressor for a window is `features(paint(z_W | halo)) -
  features(paint(ref_W | halo))` and the target `F(z_W | halo) - F(ref_W
  | halo)`; the halo's own terms cancel in both, and the window-halo
  cross terms are features, not noise (the halo floor of
  `convpot_test.md` disappears).  The model provides `paint_region(z_W,
  halo) -> painted map over the fine region` (window + halo blocks +
  spill).
- `fit(X, y, l2) -> theta`, closed form; residual RMS / target RMS.
- `materialise(theta, paint_pair, D, offsets_coarse) -> (u_c : (D,), g_c
  : (ndir, D, D))`: `u_c(v) = F_theta(paint of v alone on a reference
  background) - F_theta(reference)`, `g_c[d](v, v') = F_theta(paint of v
  at p and v' at p + d) - u_c(v) - u_c(v') - F_theta(reference)`, with
  `paint_pair(v, v', d)` the model's painter on a region large enough to
  hold both footprints and their spill.  Exact for a linear potential.
  These tables go into the forward kernel exactly as before.

## Test on circles

1. **Mid level, generalisation (the claim).**  Same held-out set as
   `convpot_test.md` stage 2 (seed 20, 54 of 136 pairs held out, hand
   entries B and D held out).  3 x 3 mid windows with a one-block halo,
   exact `F`.  Fit the paint potential (unary only; unary + 8-offset
   pairs) on the accepted windows; materialise `mid_h`, `mid_v` and the
   diagonals; score seen and held-out entries against `reference()` in
   the reference gauge (as the convpot run did): max error, slope,
   correlation, the hand entries.  Also report the fitted `u` against
   the per-tile `-log Z` values (`f0, f_d, f_a, f_c` of
   `circles_test.md`), which it should recover up to a constant, and
   whether the pair tables over painted cells are ~0 (tiles are
   independent given demands, so they should be).
2. **Top level, the halo fix.**  3 x 3 top windows with halo (the same
   windows and AIS targets as `convpot_test.md` stage 1 if saved in
   `images/convpot_circles_windows.npz`, else regenerate at K = 256, M =
   4, N = 2000); painted channel `slot` at mid resolution, offsets up to
   radius 2.  Report residual / target against the convpot run's halo
   floor (0.54) and no-halo floor (0.20); materialised `top_h`, `top_v`
   vs the oracle pattern (correlation, contrast).  For like-for-like,
   also fit the bilinear convpot (b0) with halo cells added as features
   (pairs between window and halo cells), to separate "halo as input"
   from "paint potential".
3. **End to end**: forward with the materialised tables (mid from the
   all-pairs paint fit, top from stage 2) vs oracle, 64 runs, next to
   the table and convpot rows.
4. **Potts smoke test (stretch, only if time allows)**: the painted
   channel at the tile level is the block colour itself; fit the paint
   potential on 3 x 3 mid windows with exact (transfer-matrix) `F` from
   `potts.py`'s oracle, and compare the materialised `mid_h` to `BM J d`
   at kappa 8 and to the self-play tables at kappa 1.  This is the case
   with coupled tiles, where the painted pair terms must do the work.

Outputs `images/paintpot_circles_*`, JSON `images/paintpot_circles.json`,
results appended here.

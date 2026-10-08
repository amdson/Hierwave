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

## Results

Code: `castlegen/channels/paintpot.py` (features / fit / energy /
materialise / canonical), with hooks appended to `induce_circles.py`
(`paint_region_mid`, `mid_window_F`, `paint_single_mid`, `paint_pair_mid`,
`paint_region_top`, `paint_single_top`, `paint_pair_top`; 5 x 5 coarse
canvas, centre (2, 2)).  Tests in `tests/test_paintpot.py` (6 pass):
- features match a brute-force loop;
- energy = features . theta;
- an exact fit recovers a planted theta on exhaustive 3 x 3 binary windows;
- **materialise with u = fz and no pairs reproduces
  `reference(centred=False)` mid_h / mid_v to < 1e-9, mid_u to < 1e-9
  after adding the pres term, diagonals 0**;
- the region F matches `MidWindows.window_free_energy_exact` differences.

Scripts: `notes/experiments/paintpot_circles.py` (stages `1`, `a`, `b`,
`3`), figures `notes/experiments/paintpot_circles_figs.py`, Potts
`notes/experiments/paintpot_potts.py`, identifiability helper
`notes/experiments/paintpot_rank.py`.  JSON in
`images/paintpot_circles.json` and `images/paintpot_potts.json`; windows in
`images/paintpot_circles_windows.npz`; logs in
`images/paintpot_circles_log.txt` and `images/paintpot_potts_log.txt`.

### Choices made while running

- Stage 1 windows are 5 x 5 mid grids (3 x 3 window + one-block halo),
  presence 0.5, uniform offset.  The held-out rejection is applied to the
  **whole 5 x 5 grid** (any held-out pair at any of the 8 offsets, window or
  halo).  That drops acceptance to 0.0100 (4000 of 398857 drawn; convpot
  had 0.24 on 3 x 3 without halo).  Region = 5 x 5 blocks + one-tile ring
  (42 x 42 tiles).  Target = sum of fz[dem] over the region, z vs the absent
  window under the same halo.  Ridge l2 = 1e-6 on the per-window mean loss.
- Stage 2: the saved convpot windows (`images/convpot_circles_windows.npz`,
  3 x 3 with halo) have a halo of **mid** values (a fixed one-mid-cell ring
  sampled by the forward mid kernel under reference tops), not top values,
  and the halos themselves were not saved.  I regenerated the 32 halos from
  the same rng stream (check: `F(r, halo)` by AIS reproduced bit-exactly,
  max |diff| 0).  A paint over `slot` with the halo tops at the reference
  cannot see that halo, so stage 2 has two parts:
  - **2a** (saved targets, mid-ring halo): the painted region is 8 x 8 at
    mid resolution, with slot inside.  The ring cells are painted either as
    `2 + 2 slot + present(halo)` (V = 6) or as `2 + halo mid value` (V = 19).
  - **2b** (new windows, the note's design): 5 x 5 top grids, i.e. the 3 x 3
    window plus a ring of 16 halo **top** cells (uniform).  The mid cells of
    all 25 blocks are free (10 x 10, free outer boundary, mid tables =
    convpot's exact-mid 2 x 2 tables).  Target `F(z | H) - F(r | H)` with
    **both** F by AIS, K = 256, **M = 8**, N = 2000, 80 / 20 split.  M is
    doubled from the brief's 4 because each target is a difference of two
    AIS runs (a deviation from the protocol).  Floor = within-window sd of
    the target over 4 independent AIS repeats of both F (50 windows).

  The like-for-like bilinear control (one-hot b0 with the halo tops as
  extra cells, same A tables, halo-halo terms cancel) is only defined in
  2b: in 2a the halo and the window are at different resolutions.
- Stage 3 top = 2b "paint slot V2 OFF8" (best test residual with the fewest
  features; OFF2 ties).  Mid = the all-pairs unary+OFF8 fit.

### 1. Mid level, held-out generalisation

Held-out set as convpot (seed 20: 54 of 136 unordered pp pairs, hand B and
D held out).  Target RMS 72.3 for the held-out fit (4000 windows) and 126.6
for the all-pairs fit (6000).  Scores are in the reference gauge, mid_h and
mid_v pooled (2 x 155 seen, 2 x 101 held-out entries; reference rms 0.84).
Hand entries (mid_h, materialised / reference) are the same in every row
below: A +1.939 / +1.939, B* +1.939 / +1.939, C -2.490 / -2.490, D* -2.490
/ -2.490.

| fit | variant | in-sample rel | seen max err / slope / corr | held max err / slope / corr | diag (SE / SW) rms | mid_u max err | fit s |
|---|---|---|---|---|---|---|---|
| held-out | unary | 8.4e-08 | 1.8e-05 / 1.0000 / 1.0000 | 1.8e-05 / 1.0000 / 1.0000 | 4e-13 / 4e-13 | 1.3e-07 | 0.6 |
| held-out | unary + OFF8 | 1.7e-08 | 6.9e-06 / 1.0000 / 1.0000 | 6.9e-06 / 1.0000 / 1.0000 | 3.7e-08 / 5.0e-08 | 6.7e-08 | 0.8 |
| all pairs | unary | 2.6e-08 | 5.6e-06 / 1.0000 / 1.0000 | 5.6e-06 / 1.0000 / 1.0000 | 4e-13 / 4e-13 | 5.0e-08 | 0.6 |
| all pairs | unary + OFF8 | 5.7e-09 | 2.1e-06 / 1.0000 / 1.0000 | 2.1e-06 / 1.0000 / 1.0000 | 1.4e-08 / 1.3e-08 | 4.4e-08 | 0.7 |

For comparison, convpot stage 2 on the same held-out set had held-out corr
-0.06 to +0.03 for one-hot, identity and patch embeddings, with hand B*
between -0.15 and +1.09 against +1.94.

Fitted paint parameters:
- Unary only: u (centred) = -1.3427 / -0.8018 / -0.5128 / +2.6573 for free /
  dirt / air / conflict.  This is fz centred, to 1e-5 (fz = -0.5544,
  -0.0135, +0.2756, +3.4456).
- Unary + OFF8: the paint parameters are **not identified**.  The 68 window
  feature differences have rank 11: 16 pair features never vary, and the
  disc and ring shapes fix most pair counts given the unary counts.  The
  min-norm ridge solution spreads fz over the pair tables: the canonical
  (double-centred) paint pair tables have rms 0.107 and max 0.224, and u is
  off fz by up to 1.14.  The truth (u = fz, g = 0) is in the solution set
  (|X (theta - theta_0)| rms 7e-07).  The materialised coarse tables are
  invariant to this, so they are exact anyway (table above).  "Paint pair
  tables ~0" therefore holds for the truth but cannot be read off the OFF8
  fit.  It is the unary-only fit that confirms the mid free energy is a
  paint potential with unary terms only.

Figure `images/paintpot_circles_mid.png`: reference and materialised pp
block of mid_h with held-out entries x-marked, their difference at 1e-5
scale, and a scatter of all pp entries, seen and held out.

### 2. Top level

**2a: saved convpot windows (3 x 3, mid-ring halo).**  Target RMS 1.680.
Halo floor 0.54 of target, as measured in convpot.  AIS-only floor
sqrt(mean se^2 + mean se_r^2) = 0.242 = 0.14 of target.  Oracle contrast
0.40, reference 0.52.

| variant | features | rel all | rel train | rel test | contrast h / v | top_h corr / slope vs oracle | top_v corr | corr vs ref | dc rms h / v | diag SE / SW dc rms |
|---|---|---|---|---|---|---|---|---|---|---|
| (b0) one-hot convpot, no halo features | 68 | 0.543 | 0.545 | 0.548 | 0.35 / 0.31 | 0.90 / 0.84 | 0.89 | 0.99 | 0.141 / 0.124 | 0.026 / 0.033 |
| paint slot V2 OFF8 (halo not painted) | 18 | 0.564 | 0.567 | 0.554 | 0.33 / 0.28 | 0.92 / 0.76 | 0.91 | 1.00 | 0.125 / 0.106 | 0.007 / 0.008 |
| paint slot V2 OFF2 (halo not painted) | 50 | 0.552 | 0.556 | 0.543 | 0.34 / 0.34 | 0.90 / 0.82 | 0.93 | 0.99 | 0.137 / 0.131 | 0.009 / 0.021 |
| paint slot + halo presence V6 OFF8 | 150 | 0.423 | 0.424 | 0.426 | 0.37 / 0.32 | 0.92 / 0.87 | 0.92 | 1.00 | 0.143 / 0.127 | 0.009 / 0.011 |
| paint slot + halo presence V6 OFF2 | 438 | 0.406 | 0.406 | 0.417 | 0.36 / 0.31 | 0.92 / 0.86 | 0.91 | 1.00 | 0.140 / 0.123 | 0.009 / 0.016 |
| paint slot + halo mid value V19 OFF8 | 1463 | **0.144** | 0.144 | **0.152** | 0.34 / 0.33 | 0.92 / 0.81 | 0.92 | 0.99 | 0.133 / 0.131 | 0.013 / 0.011 |

(b0) reproduces convpot's 0.543 / 0.548 exactly (same split).

**2b: new windows with a ring of halo top cells.**  Target RMS 1.230.  AIS
floor 0.399 = 0.32 of target.  Mean se 0.252 for F(z|H) and 0.249 for
F(r|H).

| variant | features | rel all | rel train | rel test | contrast h / v | top_h corr / slope vs oracle | top_v corr | corr vs ref | dc rms h / v | diag SE / SW dc rms |
|---|---|---|---|---|---|---|---|---|---|---|
| (b0) one-hot convpot, window only | 68 | 0.616 | 0.606 | 0.668 | 0.30 / 0.33 | 0.88 / 0.74 | 0.95 | 0.97 | 0.125 / 0.123 | 0.015 / 0.028 |
| (b0+halo) one-hot convpot, halo tops as cells | 68 | 0.324 | 0.320 | 0.342 | 0.33 / 0.35 | 0.90 / 0.78 | 0.93 | 0.99 | 0.131 / 0.133 | 0.011 / 0.011 |
| paint slot V2 OFF8 | 18 | 0.329 | 0.328 | **0.332** | 0.34 / 0.35 | 0.92 / 0.81 | 0.92 | 1.00 | 0.133 / 0.136 | 0.012 / 0.009 |
| paint slot V2 OFF2 | 50 | 0.328 | 0.326 | 0.334 | 0.33 / 0.35 | 0.91 / 0.79 | 0.92 | 0.99 | 0.131 / 0.134 | 0.010 / 0.007 |

Materialised tables for 2b paint slot V2 OFF8 (double-centred; rows NW NE
SW SE = this cell, columns = the neighbour):

    oracle -log(p/pp) top_h: +0.00 -0.14 +0.13 +0.01; -0.31 +0.00 +0.18 +0.13; +0.13 +0.01 +0.00 -0.14; +0.18 +0.13 -0.31 +0.00
    paint top_h:             +0.07 -0.07 +0.07 -0.07; -0.30 +0.07 +0.15 +0.07; +0.07 -0.07 +0.07 -0.07; +0.16 +0.07 -0.30 +0.07
    paint top_v:             +0.07 +0.07 -0.07 -0.07; +0.07 +0.07 -0.07 -0.07; -0.30 +0.16 +0.07 +0.07; +0.16 -0.30 +0.07 +0.07
    diagonals: all |x| <= 0.03 (dc rms 0.012 / 0.009)

Figure `images/paintpot_circles_top.png`: rows top_h, top_v, diag SE and
diag SW; columns oracle, reference, convpot (b0) 2a, 2a paint V6 OFF2, 2b
(b0+halo), 2b paint OFF8 and 2b paint OFF2.

### 3. End to end

Mid: the all-pairs unary+OFF8 materialised tables (= reference to 2e-6).
Top: 2b paint slot V2 OFF8 (top_h, top_v, top_u); the second paint row adds
the materialised SE / SW tables as extra pair factors.  64 runs at seed 555,
eval noise against seed 556, oracle moments from
`images/induce_circles.json`.  The first two rows are copied from
`convpot_test.md`.

| | mid_h | mid_v | mid_u | top_h | top_v | top_u | ms / run |
|---|---|---|---|---|---|---|---|
| tables: exact-mid + top (a, 3x3) (convpot note) | 0.176 | 0.165 | 0.114 | 0.125 | 0.162 | 0.027 | 27.8 |
| convpot: mid (b0) + top (b0) (convpot note) | 0.169 | 0.162 | 0.106 | 0.151 | 0.154 | 0.046 | 32.4 |
| paint: mid (unary+OFF8) + top (2b OFF8) h+v | 0.181 | 0.164 | 0.113 | 0.116 | 0.157 | 0.021 | 26.8 |
| paint: same + top diagonals | 0.182 | 0.164 | 0.112 | 0.118 | 0.157 | 0.023 | 25.3 |
| eval noise, paint h+v | 0.098 | 0.095 | 0.029 | 0.079 | 0.084 | 0.033 | |
| eval noise, paint + diagonals | 0.096 | 0.095 | 0.029 | 0.087 | 0.084 | 0.036 | |

Monitors (edge_air_top held out):

| | present | viol | conflict | contact | slot_viol | edge_air_top |
|---|---|---|---|---|---|---|
| oracle | 0.3532 | 0.0201 | 0.0000 | 0.5236 | 0.1127 | 0.4773 |
| tables (a, 3x3) | 0.3146 | 0.0206 | 0.0000 | 0.5954 | 0.0817 | 0.4776 |
| convpot (b0 + b0) | 0.3174 | 0.0199 | 0.0000 | 0.5945 | 0.0826 | 0.4779 |
| paint h+v | 0.3136 | 0.0208 | 0.0000 | 0.5964 | 0.0812 | 0.4768 |
| paint + diagonals | 0.3146 | 0.0206 | 0.0000 | 0.5960 | 0.0824 | 0.4767 |

### 4. Potts (stretch)

There is no cheap exact route for 3 x 3 mid windows (a 12-tile frontier has
4^12 states).  I used **2 x 2 mid windows, exact**, by a site-by-site
transfer matrix over an 8-tile frontier (4^8 states; checked against brute
force on 3 x 3 tiles).  Free boundary, no halo, all 256 window colourings;
target F(z) - F(all colour 0).  Painted channel = block colour broadcast to
the tiles (V = 4); paint potential unary / +OFF8 / +OFF2 over the 8 x 8
tiles, materialised on a 5 x 5 block canvas.  Tables are double-centred and
averaged by cyclic distance d = 0 / 1 / 2; the spread within a distance
class is <= 1.2e-3 everywhere.  Exact F takes 8.5 s per setting.

| setting | BM J d | exact 1 x 2 window pair | paint unary rel | paint OFF8 rel | paint OFF8 mid_h (= mid_v) | diag SE | self-play S0 mid_h | S1 mid_h |
|---|---|---|---|---|---|---|---|---|
| kappa 8, J 0.1 | -0.40 / 0 / +0.40 | -0.399 / 0.000 / +0.399 | 0.42 | 4.2e-06 | -0.399 / 0.000 / +0.399 | 0.000 / 0.000 / 0.000 | -0.392 / +0.015 / +0.362 | -0.399 / +0.007 / +0.385 |
| kappa 1, J 0.3 | -1.20 / 0 / +1.20 | -0.280 / +0.012 / +0.255 | 0.41 | 3.5e-03 | -0.281 / +0.013 / +0.256 | -0.020 / 0.000 / +0.019 | -0.308 / +0.022 / +0.264 | -0.288 / +0.016 / +0.257 |
| kappa 8, J 1 (ordered) | -4.0 / 0 / +4.0 | -3.999 / 0.000 / +3.999 | 0.42 | 5.2e-06 | -3.999 / 0.000 / +3.999 | 0.000 / 0.000 / 0.000 | -2.703 / +0.894 / +0.915 | -0.864 / +0.243 / +0.378 |

OFF2 gives the same tables as OFF8 to 3 decimals (rel 4.0e-06 / 3.5e-03 /
5.0e-06).  The kappa 8, J 1 setting is the ordered phase, where
`potts_test.md` reports that the self-play runs break; its row is only for
completeness.

### Cost

- Stage 1: windows + exact F + features 14 s (400k draws for the
  rejection); each fit 0.6-0.8 s.
- 2a: halo regeneration + F(r) check 4 s; fits < 2 s each (V19, with 1463
  features, is the slowest).
- 2b: AIS 571 s for 4000 runs (143 ms per run, 100 free mid cells, M = 8),
  plus 56 s for the floor.
- Stage 3: ~2 min (4 x 64 forward runs).
- Potts: 26 s for three settings.

Total compute ~15 min.

### What the numbers show

- **Mid: the claim holds, exactly.**  The paint potential over `dem`
  generalises to held-out value pairs with no loss.  Held-out entries match
  `reference()` to 2e-5 (max), slope and corr 1.0000, and hand B* and D* are
  exact; every convpot embedding had held-out corr ~0.  The unary-only fit
  recovers fz up to a constant (1e-5), confirming that the 3 x 3 mid free
  energy with halo is a paint potential with unary terms only.  The
  materialised diagonals are 0 (< 1e-7).  This holds by construction (the
  painter carries all the value structure), which is the point.
- **Top: with the halo painted, the residual is at the AIS floor.**
  - 2b (halo as top cells): paint OFF8 gives 0.332 test against a floor of
    0.32.  The bilinear convpot with the halo tops as cells lands in the
    same place (0.342).  So the gain over (b0) window-only (0.668) comes from
    "halo as input", not from "paint potential".  The paint version gets
    there with 18 parameters instead of 68 and gives the cleanest tables
    (corr vs reference 1.00, diagonals <= 0.03).  OFF2 adds nothing (0.334).
  - 2a (saved windows, mid-ring halo): painting only halo presence takes the
    residual from 0.548 to 0.417-0.426.  Painting the halo mid values
    reaches 0.152 test against the 0.14 AIS floor, so the halo's mid values
    account for convpot's 0.54 "halo floor".  But V19 is a value-indexed
    table on the halo side (1463 nominal features), so it is not a
    value-generalising paint.  It shows where the variance is; it is not a
    model to deploy.
- **The materialised top tables are no closer to the oracle than
  convpot's.**  All variants give the same pattern: corr vs oracle
  0.90-0.92, slope 0.74-0.87, contrast 0.33-0.37 (oracle 0.40), corr vs the
  lam -> inf reference 0.99-1.00.  Removing the halo variance changed the
  residual, not the tables.
- **End to end is within eval noise of the tables and convpot rows.**  Mid
  0.18 / 0.16 / 0.11 (noise 0.10 / 0.10 / 0.03); top_h 0.116 vs 0.125 /
  0.151 (noise 0.08-0.09); top_v 0.157 vs 0.154-0.162.  The same presence
  and slot_viol deficit remains (present 0.314 vs 0.353; slot_viol 0.081 vs
  0.113), and held-out edge_air_top is 0.477 (oracle 0.477).  Adding the top
  diagonals changes nothing.
- **Potts (2 x 2 exact windows): pair terms over painted tiles do the
  work.**  Unary-only leaves 0.41-0.42 of the target; unary + OFF8 fits to
  4e-6 (kappa 8) / 3.5e-3 (kappa 1).  The materialised mid_h equals the
  exact two-block free-boundary pair term.
  - At kappa 8, J 0.1 it is BM J d to 1e-3, and self-play S1 agrees to
    0.014.
  - At kappa 1, J 0.3 it is -0.281 / +0.013 / +0.256: far from BM J d (as
    expected), within 0.01 of S1 (-0.288 / +0.016 / +0.257) and within 0.03
    of S0.  A small diagonal (+-0.02) appears.

### Things that looked wrong / caveats

- The paint parameters at the mid level are badly unidentified (rank 11 of
  68 with OFF8).  Only the materialised tables and the unary-only fit are
  meaningful; reading the fitted `g` directly will mislead.
- The note's stage 2 premise ("the same windows if saved") does not hold:
  the saved halo is a ring of mid values, which a paint over `slot` with
  the halo tops at the reference cannot see.  Hence the 2a / 2b split; 2b
  used M = 8, not 4.
- The 2b floor (0.32 of target) is higher than convpot's no-halo floor
  (0.20), because each target differences two AIS runs over 100 free cells.
  "At the floor" here leaves room for structure at the 0.1-0.2 level that
  this test cannot see.
- The Potts windows are 2 x 2 with a free boundary and no halo, so the
  materialised coupling is the free-boundary two-block term, not the
  infinite-lattice induced coupling.  The two coincide at kappa 8; at
  kappa 1 the agreement with self-play suggests they are close, but that is
  not a direct check.  The note asked for 3 x 3 windows; not done (cost).
- The stage 1 rejection is stricter than convpot's (the whole 5 x 5 grid is
  checked), so the held-out windows hold fewer objects.  That is irrelevant
  for an exact unary potential, but the protocol is not byte-identical.

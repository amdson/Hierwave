# The biome circles test, stage 5b

Follows `notes/circles_biome_test.md` ("Stages 4-5: the bootstrap fit").

### Stage 5b: the boundary

The stage 5 end-to-end gate failed, and the failure was attributed to the
world boundary.  This stage tests whether the boundary is the whole story.

Script `notes/experiments/circles_biome_edge.py`, output
`images/cbio_stage5b.json`, log `images/cbio_stage5b_log.txt`, renders
`images/cbio_edge_oracle.png` and `images/cbio_edge_forward.png` (torus,
final setting), and `images/cbio_edge_forward_open.png` (open world, stage 5
potentials).  Wall time 177 s.

Config: `NT = 6`, forward `S = (30, 30, 20)` on the old path, 64 runs per
eval, oracle 50 + 400 sweeps, seeds as in stage 5.  The "all" row of part 1
reproduces the stage 5 `hook K0` row exactly (obj_h 0.396, obj_u 0.244,
bio_u 0.153, present_disc 0.107).

- **Eval noise** = L1 between two independent 64-run forward evals (seeds
  901 / 902), as in stage 5.
- **Oracle noise** = L1 between two independent 400-sweep oracle chains.

A forward-vs-oracle L1 carries both, so a fit with no bias sits at roughly
`sqrt((noise^2 + oracle_noise^2) / 2)`, i.e. 0.7-0.9 x eval noise.  The gate
is the stage 5 one: every fitted feature (`obj_h/v/d1/d2`, `obj_u`, `bio_h/v`,
`bio_u`) has L1 <= eval noise, and `conflict = 0`.

**Periodic world.**  The generic kernel cannot wrap: a pair row whose
neighbour is off the grid reads `pad` or vanishes (`kernel._energies`), and
`kernel.py` is not ours to change.  So `CirclesBiome(nty, ntx,
periodic=True)` (added to `circles_biome.py`) stores the torus padded with a
ghost ring one top cell wide (8 x 8 top cells for the 6 x 6 torus).

- **Ghost cells.**  Ghost cells are fixed and are exact copies of the
  opposite interior cells (`C.sync`).  Every interior site therefore reads
  its wrapped neighbours through the unchanged kernel, `stamp_features`,
  `_delta` and `Bench.top_hook`.
- **Forward.**  The forward sweeps one class of a torus-proper colouring at
  a time and syncs after each class: `(y + x) % 2` at the top, 2 x 2 at the
  mid level, one class for tiles.  This is exact Gibbs on the torus, not a
  lagged copy.
- **Oracle.**  The oracle writes a mid move to every copy of the slot,
  repaints and redraws around each copy, then syncs.
- **Paints and stats.**  `paint_dem` / `paint_allow` sync.  `stats` crops the
  torus and wraps every pair, ring and top-edge band.
- **Tests.**  The new tests check the wrapped paint against an independent
  modular painter, and `mid_probs` against direct torus painting.  They also
  check the tile energy identity on a periodic 2 x 2 world, that ghosts stay
  copies after oracle and forward runs, and that `stats_region(margin=0)` is
  `stats`.
- **Not supported.**  The Sampler path (`use_sampler`) is not wired for the
  torus and asserts.

#### 1. Interior-only evaluation (open world, stage 5 potentials)

The interior is defined per level:

- `int`: top cells, slots and tiles at least one block of their own level
  from the edge (4 x 4 top cells, 10 x 10 slots).
- `deep`: the slots and tiles inside the interior top cells (8 x 8 slots).

L1 to the oracle, with the eval noise below each row:

| region | obj_h | obj_v | obj_d1 | obj_d2 | obj_u | bio_h | bio_v | bio_u | present_disc | present_bar | dormant | contact | bio_hist | edge_air_top | conflict | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 0.396 | 0.390 | 0.391 | 0.391 | **0.244** | **0.236** | **0.212** | **0.153** | 0.107 | 0.029 | 0.034 | 0.054 | 0.153 | 0.016 | 0 | fail |
| noise | 0.291 | 0.304 | 0.345 | 0.333 | 0.053 | 0.064 | 0.071 | 0.036 | 0.013 | 0.002 | 0.007 | 0.003 | 0.036 | 0.002 | 0 | |
| oracle noise | 0.138 | 0.134 | 0.142 | 0.149 | 0.035 | 0.062 | 0.056 | 0.042 | 0.014 | 0.008 | 0.004 | 0.001 | 0.042 | 0.001 | 0 | |
| int | 0.317 | 0.316 | 0.342 | 0.340 | **0.089** | 0.108 | **0.126** | 0.035 | 0.036 | 0.013 | 0.025 | 0.008 | 0.035 | 0.010 | 0 | fail |
| noise | 0.369 | 0.368 | 0.417 | 0.395 | 0.080 | 0.161 | 0.083 | 0.057 | 0.017 | 0.002 | 0.007 | 0.005 | 0.057 | 0.003 | 0 | |
| oracle noise | 0.165 | 0.167 | 0.179 | 0.184 | 0.043 | 0.106 | 0.095 | 0.034 | 0.008 | 0.001 | 0.006 | 0.003 | 0.034 | 0.000 | 0 | |
| deep | 0.389 | 0.402 | 0.430 | 0.422 | 0.101 | 0.108 | **0.126** | 0.035 | 0.028 | 0.003 | 0.012 | 0.013 | 0.035 | 0.010 | 0 | fail |
| noise | 0.453 | 0.474 | 0.543 | 0.512 | 0.102 | 0.161 | 0.083 | 0.057 | 0.021 | 0.002 | 0.009 | 0.002 | 0.057 | 0.004 | 0 | |

Moments (oracle / forward):

| region | present_disc | present_bar | dormant | contact | edge_air_top |
|---|---|---|---|---|---|
| all | 0.665 / 0.558 | 0.159 / 0.188 | 0.014 / 0.048 | 0.342 / 0.396 | 0.539 / 0.523 |
| int | 0.615 / 0.579 | 0.172 / 0.185 | 0.018 / 0.043 | 0.385 / 0.377 | 0.536 / 0.526 |
| deep | 0.604 / 0.576 | 0.188 / 0.185 | 0.024 / 0.036 | 0.373 / 0.386 | 0.534 / 0.524 |

Biome histogram, none / discs / bars / both:

| | all | int |
|---|---|---|
| oracle | 0.014 / 0.531 / 0.074 / 0.381 | 0.024 / 0.484 / 0.108 / 0.383 |
| forward | 0.048 / 0.458 / 0.117 / 0.377 | 0.036 / 0.467 / 0.112 / 0.385 |

Presence on the world-edge ring of slots / elsewhere:

- oracle 0.912 / 0.787;
- forward 0.707 / 0.764.

The open world's interior is not the torus either.  The L1 of the
open-world `int` oracle to the torus oracle is:

- bio_u 0.052;
- present_disc 0.039;
- dormant 0.025;
- obj_u 0.080.

So in a 6 x 6 world the edge reaches the interior through the slot pairs and
the biome chain.

#### 2. Periodic world: refit and evaluate

**Mid fit (stage 4 exact K0 on the torus).**

- Size and time: 692 records, 16 s.
- Fit quality: consistency violation 5e-15 (targets) / 2e-14 (fit).
- Error to `reference()` (double-centred, finite present pairs): max 9e-5;
  obj_u 2e-5.
- Against the open-world exact-K0 tables: max difference 1e-3 on the
  admissible entries.

The mid potential does not see the boundary, as expected: the induced
potential is a unary over the paint, and the edge only removes paint.

**Top fit (stage 5 hook K0 on the torus).**

- Size and time: 1440 records, 56 s.
- Fit quality: KL held 0.023, consistency violation 9e-16.

Learned `bio_u`, gauge none = 0:

| world | none | discs | bars | both |
|---|---|---|---|---|
| zeroth order | 0 | -2.77 | -2.77 | -4.39 |
| open (stage 5) | 0 | -3.74 | -2.97 | -5.24 |
| torus | 0 | -2.46 | -1.82 | -3.19 |

The stage 5 fit's 1.0-1.3 nats of extra pull toward discs and both was the
cheaper edge objects.  On the torus the learned unary sits 0.3-1.2 nats
*above* zeroth order: the support between neighbouring slots costs
configurations.

**End to end on the torus**, L1 to the torus oracle:

| setting | obj_h | obj_v | obj_d1 | obj_d2 | obj_u | bio_h | bio_v | bio_u | present_disc | present_bar | dormant | contact | edge_air_top | conflict | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torus fits, S = 30 | 0.249 | 0.242 | 0.257 | 0.260 | 0.064 | 0.062 | 0.065 | 0.035 | 0.021 | 0.002 | 0.008 | 0.008 | 0.002 | 0 | **pass** |
| eval noise | 0.296 | 0.307 | 0.331 | 0.310 | 0.068 | 0.085 | 0.067 | 0.048 | 0.026 | 0.016 | 0.007 | 0.003 | 0.004 | 0 | |
| oracle noise | 0.128 | 0.133 | 0.132 | 0.134 | 0.033 | 0.061 | 0.052 | 0.026 | 0.008 | 0.003 | 0.002 | 0.001 | 0.006 | 0 | |
| torus fits, repeat eval | 0.235 | 0.242 | 0.249 | 0.243 | 0.057 | 0.070 | 0.069 | 0.040 | 0.009 | 0.000 | 0.004 | 0.006 | 0.003 | 0 | at noise (bio_v 0.069 / 0.067) |
| open-world thetas on the torus | 0.244 | 0.241 | 0.261 | 0.250 | 0.058 | **0.114** | **0.112** | **0.067** | 0.009 | 0.002 | 0.009 | 0.001 | 0.005 | 0 | fail |
| torus fits, S_T = S_M = 100 | 0.235 | 0.226 | 0.240 | 0.240 | 0.041 | 0.109 | 0.099 | 0.065 | 0.012 | 0.011 | 0.000 | 0.005 | 0.006 | 0 | (single eval, see below) |

Moments of the torus run:

- Biome histogram, oracle 0.043 / 0.473 / 0.116 / 0.369, forward 0.034 /
  0.475 / 0.106 / 0.385.
- present_disc 0.576 / 0.597.
- Presence on the torus's first and last rows and columns / elsewhere:
  oracle 0.759 / 0.767, forward 0.784 / 0.783.  The edge excess is gone.

#### 4. What remains, against noise

Single 64-run evals compared with a single noise draw are a coin flip at
this level.  So the final setting was evaluated four times per budget (4 x
64 runs) against a 2000-sweep oracle chain:

- `L1 pool` is the 256-run mean against the long oracle.
- `noise pool` is L1(runs 1-128, runs 129-256).  A bias-free `L1 pool`
  should be about half of it plus the long oracle's own noise.
- The long oracle against the 400-sweep oracle: obj_h 0.105, obj_u 0.034,
  bio_u 0.037.

| budget | quantity | obj_h | obj_v | obj_d1 | obj_d2 | obj_u | bio_h | bio_v | bio_u | present_disc | present_bar | dormant | contact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S = 30 | mean single-eval L1 | 0.217 | 0.221 | 0.232 | 0.230 | 0.041 | 0.053 | 0.052 | 0.030 | 0.006 | 0.003 | 0.002 | 0.002 |
| | L1 pool | 0.114 | 0.118 | 0.125 | 0.124 | 0.023 | 0.027 | 0.027 | 0.016 | 0.003 | 0.002 | 0.001 | 0.000 |
| | noise pool | 0.209 | 0.219 | 0.230 | 0.221 | 0.037 | 0.046 | 0.038 | 0.009 | 0.004 | 0.003 | 0.002 | 0.002 |
| S = 100 | mean single-eval L1 | 0.215 | 0.223 | 0.233 | 0.225 | 0.046 | 0.059 | 0.058 | 0.034 | 0.007 | 0.006 | 0.002 | 0.003 |
| | L1 pool | 0.116 | 0.118 | 0.118 | 0.119 | 0.027 | 0.042 | 0.045 | 0.029 | 0.003 | 0.003 | 0.002 | 0.003 |
| | noise pool | 0.215 | 0.221 | 0.233 | 0.217 | 0.038 | 0.063 | 0.066 | 0.049 | 0.013 | 0.013 | 0.002 | 0.005 |

**What the numbers show.**

1. **The boundary is the whole story.**
   - *Torus.*  With the boundary removed, the same pipeline passes the
     stage 5 gate: the stage 4 exact-K0 mid fit, then the stage 5 hook-K0
     top fit, then the forward.  Every fitted feature is within eval noise
     (bio_v at noise on a repeat eval) and conflict is 0.
   - *Monitors.*  Every monitor is at noise as well: present_disc 0.021
     against 0.026, dormant 0.008 against 0.007.  The open world showed
     0.107 and 0.034.
   - *What the torus removed.*  It removed the only thing that changed.
     Mid tables, targets, features, dials and sweep budget are the same as
     in stage 5, and the mid theta agrees with the open-world one to 1e-3.
2. **Why interior-only evaluation is not enough.**  On the open world,
   restricting to the interior cuts the residual 2-7x:
   - obj_u 0.244 -> 0.089, bio_u 0.153 -> 0.035, present_disc 0.107 ->
     0.036.
   - bio_u and bio_h pass on the interior.

   The interior gate still fails, for two reasons:
   - *Biased fit.*  The top theta is position free but was fitted on
     contexts mixing edge and interior cells.  Its unary is the average:
     1.0-1.3 nats too attractive for an interior cell.  Installed on the
     torus, the open-world thetas give bio_u 0.067 and bio_h 0.114, against
     0.035 and 0.062 for the torus fit.
   - *Leaking edge.*  In a 6 x 6 world the edge reaches the interior
     through the slot pairs and the biome chain.  The open-world interior
     oracle differs from the torus oracle by bio_u 0.052 and present_disc
     0.039.

   So the edge both biases the fit and contaminates the evaluation region.
   Masking the evaluation cannot undo the first.  A periodic training world
   removes both.
3. **The remaining residual is noise.**
   - *Mid level.*  Pooled over 256 runs against a 2000-sweep oracle, every
     feature's L1 is at the level its noise predicts: obj_h 0.114 against
     about 0.105 expected (half of noise pool), obj_u 0.023 against about
     0.019, before the long oracle's own noise is added.
   - *Top level.*  The largest pooled ratio is in the top features (bio_u
     0.016 against noise pool 0.009 at S = 30).  But the same quantity at
     S = 100 is 0.029 against 0.049, so neither budget shows a bias that
     the other does not.
   - *What bias is left.*  Any residual top-level bias is below about
     0.02-0.03 in bio_u L1 at this sample size.  Its probable source, if
     real, is the top pair terms: bio_h / bio_v are tabular nearest-
     neighbour only and the target is collapsed under the K = 0 contexts.
     Mid pairs beyond the 4 offsets cannot contribute, since the support
     and the shared-ring term are exactly nearest-neighbour.
4. **The sweep budget is not a factor.**
   - The pooled L1 at S = 100 matches S = 30 within noise on every feature.
   - S = 100 against S = 30 directly: obj_u 0.077 against noise 0.068,
     bio_u 0.050 against 0.048.

   The chains are mixed at 30 sweeps: tau(obj | biome) is 1.2-1.6 sweeps.
5. **No edge-aware unary was needed.**  Item 3, an `obj_u_edge` per edge
   class, was not built, because the periodic world passes.  It would only
   serve a bounded toy world.

**Consequence for the design docs.**  Toy worlds used for training and
end-to-end gates should be periodic, because production worlds are infinite
and have no edge.  An edge biases the position-free learned potentials and
leaks into the interior statistics.

Deviations:

- The periodic world is a padded torus with ghost copies and a per-colour
  sync, not a wrapping kernel, because the generic kernel's off-grid rule
  cannot wrap.  This is exact Gibbs on the torus for the old path; the
  Sampler path is not supported.
- `materialise` / `all_tables` take the open-world `CirclesBiome` (C0).  The
  tables do not depend on the world, and the padded `dem_of` would treat a
  5 x 5 canvas as a world grid.  `ghost_width` now returns 0 for non-world
  shapes as a guard.
- Additions beyond the brief:
  - oracle noise;
  - the `deep` interior;
  - the open-world thetas on the torus;
  - the pooled 4 x 64-run residual against a 2000-sweep oracle.
- Not rerun on the torus: stage 4/5 AIS columns and K = 3.

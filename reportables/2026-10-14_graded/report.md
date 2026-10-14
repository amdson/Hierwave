# Graded (smooth) violation measures (ideas #21) — PAUSED 2026-10-14 (coordinator's pause order)

Agent: graded. CPU only (cores 0-7, JAX_PLATFORMS=cpu); GPU 0 was at 30 % at every check, GPU 1 never used. No training was run; nothing is running now.

## 1. What is built (all on disk, tests pass)

**`castlegen/graded.py`** — stable API, documented at the top of the file (numpy, host-side; no jax twin yet):
- `instance_distances(problem, state, region) -> {factor name: (d, loc)}`: per hard-factor instance an integer distance d >= 0 (d = 0 iff the instance is satisfied), in `energy.instances` order, with `loc` = the cells each instance reads (levels and coordinates; -1 for pad reads) plus `level` and `violated`.
- `level_mass(problem, state, region) -> int[T+1]`: D_l, the graded violation mass per level.
- `graded_field(problem, state, region, level) -> float[(H>>level, W>>level)]`: per cell the summed d of the hard instances (any level) touching it; finer cells map to their ancestor, coarser cells spread their full d (not split) over the cells they cover, count instances cover the parent's footprint, an instance counts once per cell. Zero everywhere iff the state is valid. This is the input feature the #19 / guided-update agents asked for.
- `graded(problem, state, region) -> Graded` (viol, nhard, mass per level; per_factor -> (viol, mass, n); per-instance d; `hist()`), `describe(problem)`, `factor_distance`.

**Construction (generic, from the tables only).** d = min over feasible entries of the L1 movement, with |v - v'| on an axis whose feasible sets are intervals with monotone endpoints (a "monotone band": <=, >=, =; sums of count factors always), Hamming 1[v != v'] on any other axis (edit distance), and infinite movement on a fixed cell (designed channel, pad read). What it gives:
- ground: `support_l` d = B(p) - A(below) = unsupported occupied columns (one per floating tile at L0); `honourA` d = full-column shortfall; `honourB` d = occupied-column excess. All 21 hard factors classified ordered.
- toys: |sum - count|, |rho2 - 8|. columns: counts |sum - n|; the `column` pair is not a band (k = 0 may sit above 0 or 12) so d = 1 = binary.
- dimers: all four hard factors are over unordered directions and every value has a feasible partner, so d = 1 per violated instance: **graded = binary on dimers** (a per-factor measure cannot see a world-level matching edit distance). The idea's second bullet therefore says nothing new on dimers.

**Tests.** `tests/test_graded.py`: 17 passed (CPU, 26 s): d = 0 <=> H = 0 and identical counts to energy.py on all five suite problems (probes, exact toy samples, random states, untrained cfast raw/final draws, open windows with pads); hand checks (1 and 3 unsupported columns -> 1 and 3; parent A raised by k -> honour shortfall k; count one / two off -> 1 / 2; toy_modes rho2 = 0 -> 8 per block; a dimers rhombus conflict -> two distance-1 instances); band detection; a designed fixed side with an unordered table.

**Graded penalty (approved first test, addendum (c)) — code built, tests mostly pass, NOT yet run as an experiment.**
- `castlegen/variants/_graded_core.py`: `CondGraded` (CondJax + closed graded distance `nvg` per candidate; pads / designed cells fixed as in graded.py), `train_pass_graded` (cfast's fast train_pass recording `nvg`, cost = lam_l * nvg[a]; masks still binary, so draws are cfast's), `build_train` (cfast's K = 0 fused update with ell = lam_l * nvg; cone returns, critic, lambda control on the BINARY rate unchanged).
- `castlegen/variants/cgpen.py`: `cgpen` (CFast subclass) and `ckpop_hard_gpen` (= `ckpop_hard_pen` from ckernel_own.py with the 32 penalty steps replaced by cgpen's fused update via `train_fns`).
- `tests/variants/test_cgpen.py` (CPU, 105 s): cost identity (draw costs sum to lam . D(x) from graded.py) PASS; gradient check vs the exact grad of sum_x q(x) lam . D(x) on `_gradcheck.tiny_problem`, zero and wrong critic, 300 x 16: PASS (proj_z -3.2 / 0.76 zero critic, -1.6 / 0.72 wrong critic; all |z| < 4); same draws as cfast pen_only and tot = lam . D on toy_support: PASS. The power test (estimator rejected against the BINARY penalty's gradient) FAILS for lack of power (hotelling_p 0.57 / 0.29): on tiny_problem D > H only on rare states; marked xfail with that reason. Run 1800 batches or test on a problem with wider bands before trusting a difference.

## 2. Diagnostic 1: graded mass next to counts on the saved own-draws models (no retraining)

Script `scripts/diag1_run.sh` -> `scripts/diag1_gen.py` (one CPU process per model, 16 held-out eval worlds, seeds EVAL_SEED0 + i, W = 128, raw and final), `scripts/diag1_summ.py` -> `runs/diag1_summary.md`; raw per-world records in `runs/diag1_<model>.json`, log `runs/diag1.log`. Models: cfast300 (cfine's 300-update cfast, collapsed, dirt 0.96), cr2 `ckpop_hard_pen` s0, wave6 `ckpop_rkl123x2` C and `_cg40` D, wave4 `ckpop_hard` W1 and `ckpop_hard_rkl` W2b (300 s).

**Reading.** Are the residual violations "nearly valid"?
- At L0 every violated instance has d = 1 (a single floating tile): yes.
- At L1 mass per violated instance is 1.2-1.9 (one or two unsupported columns of a 2-column block): mostly yes.
- At L2 it is 2-3.3 (of a maximum 4): these are half-to-whole-block misses, not single-column ones.
- At L3-L4 the rare instances carry d = 4-11 (honourA4top with d = 8-10: the top certificate promises a whole L3 block's worth of full columns the children do not have). Repair makes this worse at the coarse levels: w6_D final L4 mass 18 (two instances, d = 7 and 11) from raw 0; w6_C final L3 18 vs raw 12. This is the repair-created coarse failure of change request #3, now with a size: a repaired coarse certificate ends up 7-11 columns off.
- Own-draws arms per world: cr2 `ckpop_hard_pen` 13/16 worlds valid after repair, the three failing worlds carry mass 1, 1, 10 (the last: one L2 instance d = 3 plus honourB3 d = 2 plus fine misses). wave6 C/D: 6/16 and 8/16 valid; failing worlds mass 1-19. So the failures are a few worlds with one coarse miss that drags several finer ones along, not a diffuse dusting.
- cfast300 (collapsed): 0/16 valid; mass per violated instance 1.5-2.8 at L1-L2.

### Per level (16 held-out eval worlds each, W = 128; budget 2e-4)

| model | kind | L | violated | frac | mass D | D / instances | D per violated | worlds with a violation |
|---|---|---|---|---|---|---|---|---|
| cfast300 | raw | 0 | 57 | 2.2e-04 | 57 | 2.2e-04 | 1.00 | 13/16 |
| cfast300 | raw | 1 | 202 | 3.1e-03 | 380 | 5.8e-03 | 1.88 | 16/16 |
| cfast300 | raw | 2 | 55 | 3.4e-03 | 154 | 9.4e-03 | 2.80 | 16/16 |
| cfast300 | raw | 3 | 3 | 7.3e-04 | 19 | 4.6e-03 | 6.33 | 2/16 |
| cfast300 | raw | 4 | 0 | 0.0e+00 | 0 | 0.0e+00 | nan | 0/16 |
| cfast300 | final | 0 | 50 | 1.9e-04 | 50 | 1.9e-04 | 1.00 | 13/16 |
| cfast300 | final | 1 | 35 | 5.3e-04 | 57 | 8.7e-04 | 1.63 | 9/16 |
| cfast300 | final | 2 | 61 | 3.7e-03 | 89 | 5.4e-03 | 1.46 | 16/16 |
| cfast300 | final | 3 | 2 | 4.9e-04 | 7 | 1.7e-03 | 3.50 | 2/16 |
| cfast300 | final | 4 | 1 | 2.0e-03 | 6 | 1.2e-02 | 6.00 | 1/16 |
| cr2_hard_pen_s0 | raw | 0 | 3 | 1.1e-05 | 3 | 1.1e-05 | 1.00 | 2/16 |
| cr2_hard_pen_s0 | raw | 1 | 4 | 6.1e-05 | 5 | 7.6e-05 | 1.25 | 3/16 |
| cr2_hard_pen_s0 | raw | 2 | 4 | 2.4e-04 | 9 | 5.5e-04 | 2.25 | 2/16 |
| cr2_hard_pen_s0 | raw | 3 | 0 | 0.0e+00 | 0 | 0.0e+00 | nan | 0/16 |
| cr2_hard_pen_s0 | raw | 4 | 0 | 0.0e+00 | 0 | 0.0e+00 | nan | 0/16 |
| cr2_hard_pen_s0 | final | 0 | 3 | 1.1e-05 | 3 | 1.1e-05 | 1.00 | 2/16 |
| cr2_hard_pen_s0 | final | 1 | 2 | 3.1e-05 | 3 | 4.6e-05 | 1.50 | 2/16 |
| cr2_hard_pen_s0 | final | 2 | 3 | 1.8e-04 | 6 | 3.7e-04 | 2.00 | 1/16 |
| cr2_hard_pen_s0 | final | 3 | 0 | 0.0e+00 | 0 | 0.0e+00 | nan | 0/16 |
| cr2_hard_pen_s0 | final | 4 | 0 | 0.0e+00 | 0 | 0.0e+00 | nan | 0/16 |
| w6_C_rkl123x2 | raw | 0 | 23 | 8.8e-05 | 23 | 8.8e-05 | 1.00 | 9/16 |
| w6_C_rkl123x2 | raw | 1 | 37 | 5.6e-04 | 46 | 7.0e-04 | 1.24 | 11/16 |
| w6_C_rkl123x2 | raw | 2 | 18 | 1.1e-03 | 38 | 2.3e-03 | 2.11 | 8/16 |
| w6_C_rkl123x2 | raw | 3 | 3 | 7.3e-04 | 12 | 2.9e-03 | 4.00 | 3/16 |
| w6_C_rkl123x2 | raw | 4 | 0 | 0.0e+00 | 0 | 0.0e+00 | nan | 0/16 |
| w6_C_rkl123x2 | final | 0 | 15 | 5.7e-05 | 15 | 5.7e-05 | 1.00 | 8/16 |
| w6_C_rkl123x2 | final | 1 | 15 | 2.3e-04 | 18 | 2.7e-04 | 1.20 | 7/16 |
| w6_C_rkl123x2 | final | 2 | 9 | 5.5e-04 | 25 | 1.5e-03 | 2.78 | 6/16 |
| w6_C_rkl123x2 | final | 3 | 5 | 1.2e-03 | 18 | 4.4e-03 | 3.60 | 4/16 |
| w6_C_rkl123x2 | final | 4 | 0 | 0.0e+00 | 0 | 0.0e+00 | nan | 0/16 |
| w6_D_rkl123x2_cg40 | raw | 0 | 17 | 6.5e-05 | 17 | 6.5e-05 | 1.00 | 8/16 |
| w6_D_rkl123x2_cg40 | raw | 1 | 17 | 2.6e-04 | 28 | 4.3e-04 | 1.65 | 7/16 |
| w6_D_rkl123x2_cg40 | raw | 2 | 11 | 6.7e-04 | 28 | 1.7e-03 | 2.55 | 6/16 |
| w6_D_rkl123x2_cg40 | raw | 3 | 4 | 9.8e-04 | 27 | 6.6e-03 | 6.75 | 4/16 |
| w6_D_rkl123x2_cg40 | raw | 4 | 0 | 0.0e+00 | 0 | 0.0e+00 | nan | 0/16 |
| w6_D_rkl123x2_cg40 | final | 0 | 10 | 3.8e-05 | 10 | 3.8e-05 | 1.00 | 5/16 |
| w6_D_rkl123x2_cg40 | final | 1 | 10 | 1.5e-04 | 19 | 2.9e-04 | 1.90 | 4/16 |
| w6_D_rkl123x2_cg40 | final | 2 | 8 | 4.9e-04 | 15 | 9.2e-04 | 1.88 | 5/16 |
| w6_D_rkl123x2_cg40 | final | 3 | 3 | 7.3e-04 | 15 | 3.7e-03 | 5.00 | 2/16 |
| w6_D_rkl123x2_cg40 | final | 4 | 2 | 3.9e-03 | 18 | 3.5e-02 | 9.00 | 2/16 |
| w4_W1_hard | raw | 0 | 52 | 2.0e-04 | 52 | 2.0e-04 | 1.00 | 15/16 |
| w4_W1_hard | raw | 1 | 46 | 7.0e-04 | 58 | 8.9e-04 | 1.26 | 14/16 |
| w4_W1_hard | raw | 2 | 12 | 7.3e-04 | 23 | 1.4e-03 | 1.92 | 8/16 |
| w4_W1_hard | raw | 3 | 3 | 7.3e-04 | 20 | 4.9e-03 | 6.67 | 2/16 |
| w4_W1_hard | raw | 4 | 0 | 0.0e+00 | 0 | 0.0e+00 | nan | 0/16 |
| w4_W1_hard | final | 0 | 38 | 1.4e-04 | 38 | 1.4e-04 | 1.00 | 12/16 |
| w4_W1_hard | final | 1 | 18 | 2.7e-04 | 26 | 4.0e-04 | 1.44 | 8/16 |
| w4_W1_hard | final | 2 | 5 | 3.1e-04 | 10 | 6.1e-04 | 2.00 | 4/16 |
| w4_W1_hard | final | 3 | 5 | 1.2e-03 | 21 | 5.1e-03 | 4.20 | 4/16 |
| w4_W1_hard | final | 4 | 0 | 0.0e+00 | 0 | 0.0e+00 | nan | 0/16 |
| w4_W2b_hard_rkl | raw | 0 | 29 | 1.1e-04 | 30 | 1.1e-04 | 1.03 | 13/16 |
| w4_W2b_hard_rkl | raw | 1 | 10 | 1.5e-04 | 13 | 2.0e-04 | 1.30 | 8/16 |
| w4_W2b_hard_rkl | raw | 2 | 6 | 3.7e-04 | 19 | 1.2e-03 | 3.17 | 6/16 |
| w4_W2b_hard_rkl | raw | 3 | 2 | 4.9e-04 | 7 | 1.7e-03 | 3.50 | 1/16 |
| w4_W2b_hard_rkl | raw | 4 | 0 | 0.0e+00 | 0 | 0.0e+00 | nan | 0/16 |
| w4_W2b_hard_rkl | final | 0 | 23 | 8.8e-05 | 23 | 8.8e-05 | 1.00 | 13/16 |
| w4_W2b_hard_rkl | final | 1 | 6 | 9.2e-05 | 10 | 1.5e-04 | 1.67 | 5/16 |
| w4_W2b_hard_rkl | final | 2 | 7 | 4.3e-04 | 23 | 1.4e-03 | 3.29 | 5/16 |
| w4_W2b_hard_rkl | final | 3 | 0 | 0.0e+00 | 0 | 0.0e+00 | nan | 0/16 |
| w4_W2b_hard_rkl | final | 4 | 0 | 0.0e+00 | 0 | 0.0e+00 | nan | 0/16 |

### Per factor (summed over the 16 worlds; violated instances, graded mass)

| model | kind | factor | violated | mass |
|---|---|---|---|---|
| cfast300 | raw | honourA3top | 1 | 2 |
| cfast300 | raw | honourB2bot | 91 | 167 |
| cfast300 | raw | honourB2top | 87 | 173 |
| cfast300 | raw | honourB3bot | 1 | 1 |
| cfast300 | raw | honourB3top | 1 | 1 |
| cfast300 | raw | honourB4bot | 1 | 8 |
| cfast300 | raw | honourB4top | 1 | 7 |
| cfast300 | raw | support0 | 57 | 57 |
| cfast300 | raw | support1 | 24 | 40 |
| cfast300 | raw | support2 | 52 | 150 |
| cfast300 | raw | support3 | 1 | 4 |
| cfast300 | final | honourB1bot | 1 | 1 |
| cfast300 | final | honourB1top | 2 | 2 |
| cfast300 | final | honourB2bot | 3 | 4 |
| cfast300 | final | honourB2top | 8 | 12 |
| cfast300 | final | honourB3bot | 43 | 46 |
| cfast300 | final | honourB3top | 1 | 1 |
| cfast300 | final | support0 | 47 | 47 |
| cfast300 | final | support1 | 24 | 41 |
| cfast300 | final | support2 | 17 | 42 |
| cfast300 | final | support3 | 2 | 7 |
| cfast300 | final | support4 | 1 | 6 |
| cr2_hard_pen_s0 | raw | honourB2bot | 2 | 2 |
| cr2_hard_pen_s0 | raw | honourB3bot | 1 | 2 |
| cr2_hard_pen_s0 | raw | honourB3top | 1 | 2 |
| cr2_hard_pen_s0 | raw | support0 | 3 | 3 |
| cr2_hard_pen_s0 | raw | support1 | 2 | 3 |
| cr2_hard_pen_s0 | raw | support2 | 2 | 5 |
| cr2_hard_pen_s0 | final | honourB2bot | 1 | 1 |
| cr2_hard_pen_s0 | final | honourB3bot | 1 | 1 |
| cr2_hard_pen_s0 | final | honourB3top | 1 | 2 |
| cr2_hard_pen_s0 | final | support0 | 3 | 3 |
| cr2_hard_pen_s0 | final | support1 | 1 | 2 |
| cr2_hard_pen_s0 | final | support2 | 1 | 3 |
| w6_C_rkl123x2 | raw | honourA2top | 7 | 10 |
| w6_C_rkl123x2 | raw | honourA4top | 1 | 8 |
| w6_C_rkl123x2 | raw | honourB1bot | 3 | 3 |
| w6_C_rkl123x2 | raw | honourB1top | 5 | 5 |
| w6_C_rkl123x2 | raw | honourB2bot | 7 | 9 |
| w6_C_rkl123x2 | raw | honourB3bot | 6 | 15 |
| w6_C_rkl123x2 | raw | honourB3top | 3 | 7 |
| w6_C_rkl123x2 | raw | support0 | 15 | 15 |
| w6_C_rkl123x2 | raw | support1 | 23 | 27 |
| w6_C_rkl123x2 | raw | support2 | 9 | 16 |
| w6_C_rkl123x2 | raw | support3 | 2 | 4 |
| w6_C_rkl123x2 | final | honourA2top | 2 | 2 |
| w6_C_rkl123x2 | final | honourA3top | 2 | 9 |
| w6_C_rkl123x2 | final | honourA4top | 1 | 10 |
| w6_C_rkl123x2 | final | honourB2bot | 2 | 2 |
| w6_C_rkl123x2 | final | honourB3bot | 4 | 10 |
| w6_C_rkl123x2 | final | honourB4bot | 1 | 2 |
| w6_C_rkl123x2 | final | honourB4top | 1 | 2 |
| w6_C_rkl123x2 | final | support0 | 15 | 15 |
| w6_C_rkl123x2 | final | support1 | 11 | 14 |
| w6_C_rkl123x2 | final | support2 | 3 | 6 |
| w6_C_rkl123x2 | final | support3 | 2 | 4 |
| w6_D_rkl123x2_cg40 | raw | honourA2top | 3 | 6 |
| w6_D_rkl123x2_cg40 | raw | honourA3top | 1 | 4 |
| w6_D_rkl123x2_cg40 | raw | honourA4top | 2 | 16 |
| w6_D_rkl123x2_cg40 | raw | honourB1bot | 3 | 3 |
| w6_D_rkl123x2_cg40 | raw | honourB1top | 1 | 1 |
| w6_D_rkl123x2_cg40 | raw | honourB2bot | 4 | 5 |
| w6_D_rkl123x2_cg40 | raw | honourB2top | 2 | 3 |
| w6_D_rkl123x2_cg40 | raw | honourB3bot | 4 | 10 |
| w6_D_rkl123x2_cg40 | raw | honourB3top | 2 | 4 |
| w6_D_rkl123x2_cg40 | raw | support0 | 13 | 13 |
| w6_D_rkl123x2_cg40 | raw | support1 | 8 | 14 |
| w6_D_rkl123x2_cg40 | raw | support2 | 4 | 10 |
| w6_D_rkl123x2_cg40 | raw | support3 | 2 | 11 |
| w6_D_rkl123x2_cg40 | final | honourA2top | 2 | 5 |
| w6_D_rkl123x2_cg40 | final | honourA3top | 1 | 2 |
| w6_D_rkl123x2_cg40 | final | honourA4top | 1 | 6 |
| w6_D_rkl123x2_cg40 | final | honourB2bot | 1 | 2 |
| w6_D_rkl123x2_cg40 | final | honourB2top | 1 | 2 |
| w6_D_rkl123x2_cg40 | final | honourB3bot | 3 | 3 |
| w6_D_rkl123x2_cg40 | final | honourB3top | 1 | 1 |
| w6_D_rkl123x2_cg40 | final | support0 | 10 | 10 |
| w6_D_rkl123x2_cg40 | final | support1 | 6 | 10 |
| w6_D_rkl123x2_cg40 | final | support2 | 3 | 9 |
| w6_D_rkl123x2_cg40 | final | support3 | 2 | 9 |
| w6_D_rkl123x2_cg40 | final | support4 | 2 | 18 |
| w4_W1_hard | raw | honourA2top | 5 | 8 |
| w4_W1_hard | raw | honourA3top | 1 | 1 |
| w4_W1_hard | raw | honourA4bot | 1 | 7 |
| w4_W1_hard | raw | honourB1bot | 1 | 1 |
| w4_W1_hard | raw | honourB1top | 1 | 1 |
| w4_W1_hard | raw | honourB2bot | 13 | 14 |
| w4_W1_hard | raw | honourB2top | 5 | 7 |
| w4_W1_hard | raw | honourB3bot | 7 | 12 |
| w4_W1_hard | raw | support0 | 50 | 50 |
| w4_W1_hard | raw | support1 | 23 | 29 |
| w4_W1_hard | raw | support2 | 4 | 10 |
| w4_W1_hard | raw | support3 | 2 | 13 |
| w4_W1_hard | final | honourA1top | 1 | 1 |
| w4_W1_hard | final | honourA2top | 1 | 2 |
| w4_W1_hard | final | honourA4bot | 1 | 5 |
| w4_W1_hard | final | honourB2bot | 4 | 6 |
| w4_W1_hard | final | honourB2top | 2 | 3 |
| w4_W1_hard | final | honourB3bot | 2 | 2 |
| w4_W1_hard | final | honourB4bot | 2 | 4 |
| w4_W1_hard | final | support0 | 37 | 37 |
| w4_W1_hard | final | support1 | 11 | 15 |
| w4_W1_hard | final | support2 | 3 | 8 |
| w4_W1_hard | final | support3 | 2 | 12 |
| w4_W2b_hard_rkl | raw | honourB1bot | 3 | 4 |
| w4_W2b_hard_rkl | raw | honourB1top | 1 | 1 |
| w4_W2b_hard_rkl | raw | honourB2bot | 3 | 4 |
| w4_W2b_hard_rkl | raw | honourB3bot | 3 | 8 |
| w4_W2b_hard_rkl | raw | honourB4top | 1 | 2 |
| w4_W2b_hard_rkl | raw | support0 | 25 | 25 |
| w4_W2b_hard_rkl | raw | support1 | 7 | 9 |
| w4_W2b_hard_rkl | raw | support2 | 3 | 11 |
| w4_W2b_hard_rkl | raw | support3 | 1 | 5 |
| w4_W2b_hard_rkl | final | honourB1bot | 1 | 1 |
| w4_W2b_hard_rkl | final | honourB2bot | 3 | 5 |
| w4_W2b_hard_rkl | final | honourB3bot | 3 | 12 |
| w4_W2b_hard_rkl | final | honourB3top | 1 | 2 |
| w4_W2b_hard_rkl | final | support0 | 22 | 22 |
| w4_W2b_hard_rkl | final | support1 | 3 | 5 |
| w4_W2b_hard_rkl | final | support2 | 3 | 9 |

### Distance histogram (level, d) -> violated instances, summed over worlds

- cfast300 raw: L0 d=1: 57, L1 d=1: 91, L1 d=2: 44, L1 d=3: 67, L2 d=1: 3, L2 d=2: 5, L2 d=3: 47, L3 d=4: 1, L3 d=7: 1, L3 d=8: 1
- cfast300 final: L0 d=1: 50, L1 d=1: 14, L1 d=2: 20, L1 d=3: 1, L2 d=1: 46, L2 d=2: 6, L2 d=3: 5, L2 d=4: 4, L3 d=3: 1, L3 d=4: 1, L4 d=6: 1
- cr2_hard_pen_s0 raw: L0 d=1: 3, L1 d=1: 3, L1 d=2: 1, L2 d=2: 3, L2 d=3: 1
- cr2_hard_pen_s0 final: L0 d=1: 3, L1 d=1: 1, L1 d=2: 1, L2 d=1: 1, L2 d=2: 1, L2 d=3: 1
- w6_C_rkl123x2 raw: L0 d=1: 23, L1 d=1: 28, L1 d=2: 9, L2 d=1: 6, L2 d=2: 5, L2 d=3: 6, L2 d=4: 1, L3 d=2: 2, L3 d=8: 1
- w6_C_rkl123x2 final: L0 d=1: 15, L1 d=1: 12, L1 d=2: 3, L2 d=1: 2, L2 d=2: 2, L2 d=3: 2, L2 d=4: 2, L2 d=5: 1, L3 d=2: 4, L3 d=10: 1
- w6_D_rkl123x2_cg40 raw: L0 d=1: 17, L1 d=1: 6, L1 d=2: 11, L2 d=1: 4, L2 d=2: 1, L2 d=3: 2, L2 d=4: 4, L3 d=4: 1, L3 d=7: 1, L3 d=8: 2
- w6_D_rkl123x2_cg40 final: L0 d=1: 10, L1 d=1: 2, L1 d=2: 7, L1 d=3: 1, L2 d=1: 5, L2 d=2: 1, L2 d=4: 2, L3 d=4: 1, L3 d=5: 1, L3 d=6: 1, L4 d=7: 1, L4 d=11: 1
- w4_W1_hard raw: L0 d=1: 52, L1 d=1: 34, L1 d=2: 12, L2 d=1: 4, L2 d=2: 6, L2 d=3: 1, L2 d=4: 1, L3 d=5: 1, L3 d=7: 1, L3 d=8: 1
- w4_W1_hard final: L0 d=1: 38, L1 d=1: 11, L1 d=2: 6, L1 d=3: 1, L2 d=1: 2, L2 d=2: 2, L2 d=4: 1, L3 d=2: 2, L3 d=5: 1, L3 d=6: 2
- w4_W2b_hard_rkl raw: L0 d=1: 28, L0 d=2: 1, L1 d=1: 7, L1 d=2: 3, L2 d=2: 2, L2 d=3: 1, L2 d=4: 3, L3 d=2: 1, L3 d=5: 1
- w4_W2b_hard_rkl final: L0 d=1: 23, L1 d=1: 2, L1 d=2: 4, L2 d=2: 3, L2 d=3: 1, L2 d=4: 2, L2 d=6: 1

### Per world: total violated instances / total mass (all levels)

| model | kind | valid worlds | violated per world | mass per world | line_match | dirt |
|---|---|---|---|---|---|---|
| cfast300 | raw | 0/16 | [6, 19, 17, 11, 27, 10, 30, 14, 7, 13, 29, 17, 26, 42, 22, 27] | [14, 29, 36, 18, 54, 28, 40, 40, 15, 38, 58, 32, 41, 81, 40, 46] | 0.500 | 0.962 |
| cfast300 | final | 0/16 | [2, 6, 8, 3, 12, 4, 12, 4, 3, 4, 18, 7, 19, 28, 12, 7] | [2, 6, 13, 3, 16, 4, 17, 4, 3, 4, 32, 10, 31, 37, 18, 9] | 0.500 | 0.962 |
| cr2_hard_pen_s0 | raw | 11/16 | [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 6, 0] | [3, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 10, 0] | 0.552 | 0.434 |
| cr2_hard_pen_s0 | final | 13/16 | [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0] | [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10, 0] | 0.552 | 0.434 |
| w6_C_rkl123x2 | raw | 3/16 | [2, 0, 4, 17, 1, 2, 7, 3, 0, 8, 1, 0, 5, 14, 8, 9] | [4, 0, 8, 19, 1, 6, 11, 4, 0, 10, 1, 0, 13, 19, 13, 10] | 0.571 | 0.507 |
| w6_C_rkl123x2 | final | 6/16 | [0, 0, 3, 10, 1, 0, 1, 3, 0, 5, 0, 0, 2, 6, 7, 6] | [0, 0, 7, 11, 1, 0, 2, 4, 0, 6, 0, 0, 11, 11, 16, 7] | 0.571 | 0.507 |
| w6_D_rkl123x2_cg40 | raw | 4/16 | [1, 8, 0, 11, 0, 2, 8, 1, 0, 2, 1, 1, 2, 0, 1, 11] | [1, 19, 0, 23, 0, 6, 19, 1, 0, 2, 2, 1, 3, 0, 1, 22] | 0.562 | 0.499 |
| w6_D_rkl123x2_cg40 | final | 8/16 | [1, 7, 0, 8, 0, 0, 4, 0, 0, 2, 1, 0, 1, 0, 0, 9] | [1, 19, 0, 19, 0, 0, 14, 0, 0, 2, 2, 0, 2, 0, 0, 18] | 0.562 | 0.499 |
| w4_W1_hard | raw | 0/16 | [4, 3, 3, 8, 11, 11, 14, 3, 7, 6, 4, 2, 5, 2, 16, 14] | [4, 4, 3, 8, 13, 14, 18, 4, 8, 6, 10, 2, 6, 2, 18, 33] | 0.628 | 0.513 |
| w4_W1_hard | final | 2/16 | [0, 2, 0, 3, 6, 7, 9, 1, 5, 5, 1, 1, 4, 2, 9, 11] | [0, 2, 0, 3, 6, 10, 11, 2, 5, 5, 6, 1, 7, 2, 10, 25] | 0.628 | 0.513 |
| w4_W2b_hard_rkl | raw | 2/16 | [3, 0, 0, 5, 1, 2, 4, 2, 3, 3, 3, 1, 4, 10, 2, 4] | [5, 0, 0, 6, 4, 2, 5, 2, 6, 3, 3, 1, 5, 20, 3, 4] | 0.599 | 0.495 |
| w4_W2b_hard_rkl | final | 2/16 | [3, 0, 0, 4, 1, 1, 2, 2, 2, 3, 1, 1, 2, 10, 2, 2] | [5, 0, 0, 4, 2, 1, 2, 3, 5, 3, 1, 1, 3, 21, 3, 2] | 0.599 | 0.495 |


## 3. Not done (paused)

- Diagnostic 2 (exactopt landscape, binary vs graded penalty): not started.
- The graded-penalty experiment: `ckpop_hard_gpen` 300 s seed 0 on GPU 1 (after the coordinator's queue: wait for `pgrep -f "bench_ck.sh|arm.sh|bench_pop.sh"` empty and FOLLOWUP_DONE in cr2_penalty/runs/gpu1_followup.nohup, GPU 1 at 0 %, CUDA_VISIBLE_DEVICES=1, cores 8-15), with a copy of `reportables/2026-10-14_cr2_penalty/scripts/arm_eval.py` (replace the variant name by `ckpop_hard_gpen`); compare with cr2_penalty report section 3 (`ckpop_hard_pen`: within budget on 3 of 4 seeds, line_match 0.56-0.59, F +0.03) and wave6 `ckpop_hard_rkl` (L2 3.5e-4, line_match 0.60-0.63, F -0.09). Then dimers 120 s from `reportables/2026-10-14_dimers_kernel/runs/cfast60.pkl` with `dk_loop_pen.py` adapted (construct `cgpen` instead of `cfast(pen_only)`; its `rkl_steps` already uses `c.fn["fused"]`, so only the constructor line changes). Also log the raw graded mass per level during training (`tot` of the fused tuple is lam . D per world).
- A jax.numpy twin of `graded_field` for use inside a jitted revising step (the numpy version is host-side).
- Expectation to test (from diagnostic 1): the graded penalty mainly changes the coarse levels' signal (d up to 11 there), while at L0 it is identical to the binary one; on dimers it is identical everywhere.

## 4. Where a graded signal could enter, with constitution status

| place | what changes | status |
|---|---|---|
| diagnostics (this report, bench-side reporting) | nothing in the sampler | allowed |
| training penalty lam_l D_l in place of lam_l H_l (`cgpen`, `ckpop_hard_gpen`) | the objective's penalty term (constraint 4) | approved as a first test, addendum 2026-10-14 (c), training only |
| repair acceptance (redraw sites in order of d; prefer the candidate with the smallest closed d among the least-violating set) | a repair detail; the least-violating set stays binary | allowed if the fixed one-round, fixed-window structure stays; a change of the repair's choice rule, log it |
| masks / lookahead (least-violating set = argmin of graded d + lookahead) | the generation conditional (constraint 2's designed step; also changes which states are reachable) | needs approval; not built, not recommended before the penalty test |
| revising steps reading `graded_field` as an input feature (#19 / guided updates) | more information to the learned part | allowed freedom ("more information to learned parts") inside the #19 amendment |
| annealed / MH kernels pricing violations by lam * d instead of lam * H | the kernel-route amendment prices at a fixed lam <= 20 per violation | an amendment change; proposal only |

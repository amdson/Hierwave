# dstage: fixed k block sweeps on collapsed cfast dimers draws (2026-10-15)

Question: do k in {10,30,60,120} block sweeps take collapsed cfast60 dimers draws to h/v/d within 0.05 of 0.5/0.25/0.25 with violations <= 2e-4, and at what cost per world?

## Answer: NO (never within 120 sweeps). The diagonal reaches tolerance (3x3, by 30 sweeps) but h and v settle at 0.38/0.38, 0.12-0.13 off the 0.5/0.25 target, and the 2x2 negative control stays at d = 0.

Run: `python reportables/2026-10-15_dstage/scripts/dstage.py runs/dstage.json 64` (env.sh 0, taskset -c 6-7; adapted from dimers_kernel chain_relax.py). 64 worlds, start = cfast60 trained draws (0.466/0.534/0.000, viol 1.5e-4/cell). One sweep = one FineBlockKernel.sweep (random offset of 4, whole 32x64 strip). lam None = infeasible moves; lam 20 = soft. GPU 0 was shared with gpen_p11, so ms are inflated and noisy; gpu_before/after.txt in runs/. One seed.

| block | lam | k | h/v/d | viol/cell | cells changed (orientation) | ms/world (cumulative to k) |
|---|---|---|---|---|---|---|
| 2x2 | inf | 10 / 30 / 60 / 120 | .499/.501/0 ; .504/.496/0 ; .505/.495/0 ; .507/.493/0 | 1.5e-4 all | .48 / .49 / .50 / .50 | 74 / 75 / 76 / 78 |
| 2x2 | 20 | 10 | .499/.501/.001 | 1.2e-4 | .48 | 76 |
| 2x2 | 20 | 30 / 60 / 120 | d = .002 / .003 / .004 | 9e-5 / 9e-5 / 8e-5 | .50 | 77 / 78 / 81 |
| 3x3 | inf | 10 | .423/.426/.151 | 4.6e-5 | .54 | 119 |
| 3x3 | inf | 30 | .399/.395/.206 | 4.6e-5 | .57 | 126 |
| 3x3 | inf | 60 | .385/.394/.221 | 4.6e-5 | .58 | 137 |
| 3x3 | inf | 120 | .383/.384/.233 | 4.6e-5 | .59 | 160 |
| 3x3 | 20 | 10 / 30 / 60 / 120 | d = .152 / .207 / .221 / .235; h,v .38-.42 | 6.1e-5 | .54-.59 | 98 / 107 / 121 / 150 |

(ms/world is total time of the first k sweeps divided by 64 worlds, so ~3-6 ms per world per 3x3 sweep, ~0.1 s total at k=10-30; sweeps after the first few are fast, the first-sweep overhead dominates.)

- Violations: 3x3 is valid throughout (4.6e-5 to 6e-5, inside 2e-4); 2x2 is at the start level (1.5e-4, inside 2e-4).
- Tolerance per component: d within 0.05 of 0.25 from k=30 (3x3: 0.206), but h and v are 0.10-0.15 off at every k, and still drifting toward 1/3 each. Plateau 0.38/0.38/0.23 matches the earlier 600-cycle result. Per dimers_kernel the bulk target by triangular symmetry is about 1/3 each (0.5/0.25/0.25 is the exact 4x4 value), so the 3x3 chain may be near equilibrium for the strip and the 0.5/0.25/0.25 criterion may itself be wrong for this region; I did not resolve this.
- 2x2 changes about half the cells (h<->v flips) yet conserves the diagonal count: d = 0 with infeasible moves; the tiny d with lam 20 comes from soft violations.

## Not done
One seed; a1_cfast60.pkl (post-ML starts) not run; k=4 blocks not run (too large for GPU); no equilibrium target measured for this strip.

## Next
Measure the strip's true h/v/d (long k=3 chain from a rhombi start meeting the squares start, or exact 4xW transfer) before using 0.5/0.25/0.25 as a gate.

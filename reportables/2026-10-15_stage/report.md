# stage: minimum budget of the fixed post-pass stage, its roughness overshoot, and a bench row (2026-10-15)

Agent stage (wave 2), GPU 1 (shared with p4fill / dstage light jobs), cores 8-15. ideas #22, p10's `v1b d5` stage.
Objective of the models: ckpop_hard_pen's (kernel-route ML + penalty-only reverse KL, cr2_penalty); training untouched.

## Answers
1. **Smallest adequate budget: R = 30, F = 150** (128/128 worlds valid, zero violations at every level; R = 15 leaves 2 worlds).
   **The roughness overshoot is a transient, not a kernel defect.** Roughness peaks at R 30 (1.84) and then falls to the chain
   value: 1.65 at R 120, **1.39 at R 240 (chain 1.40)**. It follows the offset relaxation (pass draw +4.6 rows → +0.30, chain −0.15),
   so at the minimum budget the stage is mid-transient: valid but not at the target's statistics.
2. **Bench row `ckpop_hard_pen_stage` (300 s × 2 seeds): final violations 0 at every level (pooled CI ≤ 1.1e-4 at L0-L2), but not
   accepted.** It fails **raw ≈ final** (roughness 1.10/1.33 raw → 2.20/1.91 final; fill_err 0.026 → 0.04), **finite support**
   (mismatch 0.073) and probes (flagged as for every variant). The raw≈final failure is the guard firing: the stage changes the
   terrain statistics instead of only repairing.
3. **Finite support: the bench check does cover the stage and it fails (0.073).** Reason: the stage's sweep noise is jax.random
   keyed per sweep (array position), not by absolute coordinate, and colour_gibbs/block kernels only run on periodic layouts. Computed
   reach below: finite but ~2.6e5 fine columns one-sided.

## Q1: what was run
`scripts/stage_q1.py` (copy of p10.py + the cmix chain observables: offset = mean(first-dirt row − designed line), circular roughness,
Fourier modes of the deviation). Driver `scripts/run_q1.sh`: hard_pen saved params, **128** eval worlds, W 128, arms
`v1b:5:{15,30,60,120,240}:150` (all levels free; per round 1 single-cell graded λ20 sweep + JaxBlockGibbs L1 + FineBlockKernel 2x2;
then cert reset + F cert-only graded sweeps). One noise seed per arm. Chain reference: cmix `trace_hard.npz`, sweeps 1500-2000, 32 chains.

| arm | viol L0..L4 | valid/128 | tiles chg | line_match | dirt | floating | rough (bench / circ) | offset | mean c1 (re,im) | mean abs c1 | fill p50/max | stage-1 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pass raw | 44,46,23,3,0 | 75 | – | 0.559 | 0.432 | 1.40 | 1.02 / 1.05 | +4.63 | −0.047,−0.057 | 1.26 | | |
| pass + own repair | 35,23,21,3,1 | 90 | – | 0.559 | 0.432 | 1.27 | 1.02 / 1.05 | +4.63 | −0.046,−0.057 | 1.26 | | |
| R15 F150 | 2,2,1,0,0 | 126 | 2.5 % | 0.570 | 0.444 | 0.09 | 1.75 / 1.76 | +3.89 | −0.046,−0.027 | 1.08 | 21/68 | (compile) |
| **R30 F150** | **0,0,0,0,0** | **128** | 3.5 % | 0.579 | 0.453 | 0 | **1.84** / 1.84 | +3.34 | −0.039,−0.022 | 0.96 | 21/60 | 0.9 |
| R60 F150 | 0,0,0,0,0 | 128 | 4.8 % | 0.601 | 0.467 | 0 | 1.81 / 1.80 | +2.45 | −0.047,+0.009 | 0.74 | 22/56 | 1.7 |
| R120 F150 | 0,0,0,0,0 | 128 | 6.6 % | 0.638 | 0.485 | 0 | 1.65 / 1.65 | +1.26 | −0.017,+0.017 | 0.53 | 22/60 | 3.3 |
| R240 F150 | 0,0,0,0,0 | 128 | 8.0 % | 0.667 | 0.500 | 0 | **1.39** / 1.39 | +0.30 | −0.096,+0.026 | 0.50 | 21/62 | 6.9 |
| chain (hard, late) | 0 | – | – | – | – | 0 | – / **1.40** | −0.15 | −0.21,−0.14 (other seeds) | 0.48 | | |
| probe (designed line) | 0 | 128 | – | 0.886 | 0.505 | 0 | 0.19 / 0.19 | 0.00 | | 0.17 | | |

Stage-1 seconds are for 128 worlds batched (≈7 ms/world at R 30); the refill adds ≈1.5 s per 128 worlds.
Mean c1 (re, im) is seed-dependent (chain seeds 0-7, eval seeds differ); compare mean |c1|.

Diagnostic (p4fill's suggestion; `scripts/run_tight.sh`, `runs/hard_pen_tight.log`): refill started from ground's tight abstraction
(`lift`: A = full columns, B = occupied) instead of random certificates, same R 30 stage, 128 worlds:

| refill start | F | viol | valid | first valid sweep p50/max |
|---|---|---|---|---|
| tight (lift) | 20 | 0,0,0,0,0 | 128 | 1 / 1 |
| random | 20 | 24,37,28,4,0 | 61 | 17 / 20 (67 never) |
| tight (lift) | 150 | 0,0,0,0,0 | 128 | 1 / 1 |
| random | 150 | 0,0,0,0,0 | 128 | 21 / 60 |

The tight start is problem-specific (constraint 6) unless certificates become declared abstraction maps; the variant keeps the random start.

## Q2: bench row
Module `castlegen/variants/cstage.py`: `ckpop_hard_pen_stage` subclasses `ckpop_hard_pen` (training unchanged); `generate` takes the
pass's raw draw and returns final = stage(raw), R = 30, F = 150 (cfg `stage_R`, `stage_F`); 2x2 block offset from the fixed cycle
r mod 4 (no host rng); certificate reset values from `castlegen.bench.noise.hash32` keyed by (seed, tag, level, Y, absolute X).
The pass's own repair is not used. Smoke: `scripts/smoke.py` (`runs/smoke*.log`).

Command: `python -m castlegen.bench run ckpop_hard_pen_stage ground --train 300 --seeds 2 --tag stage` (`scripts/run_bench.sh`,
`runs/bench.log`); `python -m castlegen.bench board` regenerated `results/board.md`. GPU 1 shared; SM clock before 210 MHz (idle),
after 1560 MHz, no throttle flag (runs/nvidia_*_bench.txt).

| row | seeds | updates (ML/rkl) | final viol counts L0..L4 (pooled) | worst final | raw worst | raw ≈ final | seeds agree | support | probes | rho L0..L4 | starved % L0..L4 | F | gen s/world |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ckpop_hard_pen (2026-10-14 warm) | 4 | 1176-1184 | 25,14,5,1,0 | 2.4e-4 (fail, seed 2) | 4.9e-4 | ok | ok | ok | flagged | 0.31/0.79/0.68/0.47/−0.15 | 0/5/11/24/26 | 0.034 | 0.03-0.48 |
| **ckpop_hard_pen_stage** | 2 | 912 / 1168 | **0,0,0,0,0** | **0** | 7.3e-4 | **FAIL** (roughness, fill_err) | ok | **FAIL 0.073** | flagged | 0.28/0.76/0.77/0.27/−0.06 | 0/4/9/26/22 | 0.078 | 2.6 / 1.3 |

Per seed (stage row): roughness raw → final 1.10 → 2.20 and 1.33 → 1.91; line_match 0.534 → 0.550, 0.573 → 0.592; dirt 0.396 → 0.427,
0.432 → 0.451; floating 4.2 → 0, 1.9 → 0; fill_err 0.026 → 0.040, 0.026 → 0.037. F is computed on raw draws, so it reflects training
(fewer updates under the shared GPU: 912 and 1168 vs ~1180), not the stage. Seed 0 got fewer updates (first-run setup 69 s).

Reading: the stage does the repair (zero violations at every level, pooled CI upper 7e-6/2.8e-5/1.1e-4 at L0-L2; L3/L4 still straddle
the budget for lack of instances), but at R 30 it is half way through the relaxation from the pass draw toward the chain equilibrium,
so roughness goes up by 0.6-1.1 and fill_err by 50 %: the bench's raw≈final guard says the stage is doing the sampler's job.
At R 240 roughness would match the chain (1.39) but it would still not match raw (the raw pass is the smoother, lower one).

## Q3: finite support
The bench check (two open windows at x0 = 0 and 64, compare overlaps) runs the variant's own `generate`, so it covers the stage.
Open windows: pass on the cpass margin-extended window (2752 columns each side), then the stage on region + 128 columns each side
(STAGE_PAD) cut from it, treated as periodic. Result: raw 0 mismatch, **final 0.069-0.073 mismatch** (smoke: rho4 1.0, rho1 0.40,
t 0.048). Cause: (a) the sweeps' Gumbel noise comes from jax.random per sweep, i.e. by array position, so the two windows draw different
noise at the same absolute site; (b) colour_gibbs and the block kernels require periodic layouts, so open windows are wrapped.
Fixing (a) needs a coordinate-keyed copy of colour_gibbs' `_step_world` and of the block kernels' Gumbel draws (not done).

Computed reach (one-sided, fine columns): single-cell sweep of level l = 16 colour steps × r 3 cells × 2^l → 48 × 31 = 1488 per round;
JaxBlockGibbs L1 = 4 colour classes × (block 2 + window 6) = 32; FineBlockKernel = 2 classes × (2 + 3) = 10; so ≈ 1530 per round,
× R 30 = 4.6e4; refill = 150 sweeps × 48 × (2+4+8+16) = 2.2e5. **Total ≈ 2.6e5 fine columns** (+ the pass's 2752), finite and fixed
(constraint 2), but ~2000× the 128-column strip; padding it is not practical, so finite support rests on the effective (coupled-noise)
radius, which is unmeasured until the noise is coordinate-keyed.

## Constitution
- Constraint 2: the stage is a fixed-count local repair (R 30 rounds, F 150 sweeps, fixed block offset cycle); reach above. Noise is
  not coordinate-keyed (bench finite-support check fails), so chunk-independent generation is not demonstrated.
- Constraint 4: objective and training untouched (subclass; only `generate` changes).
- Constraint 7: λ 20 inside the sweeps; the raw≈final failure is reported, not hidden.
- Freeing rho4 (d = T+1) makes the pass an initialiser of a fixed-length MCMC (p10's caveat); the raw≈final failure is the measurable form.

## What failed
- raw ≈ final (roughness, fill_err) and finite support on the bench row; not accepted.
- R 15 (2 worlds invalid at 128).
- One noise seed per Q1 arm; tiles-changed differs by 0.1 % between identical-key stage-1 runs (GPU scatter nondeterminism likely).

## Next (one line each)
- Coordinate-keyed Gumbel in a copied colour_gibbs step + block kernels, then rerun the support check with STAGE_PAD 128/256 to measure the effective radius.
- Fix the transient at its source: the pass draw sits +4.6 rows low; a stage cannot be both short and statistics-preserving from there.
- Batch the stage over worlds in the bench (2.6 s/world now, dispatch-bound; 7 ms/world batched).

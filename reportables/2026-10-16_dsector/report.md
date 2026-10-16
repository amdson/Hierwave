# dsector: does a sector-changing, p*-invariant refresh kernel fix the dimers population and the pass? (2026-10-16)

## Answer
- **Population: YES.** A worm / alternating-loop kernel brings every start to the exact 0.351/0.324/0.324 in **one sweep** (within 0.005 on each fraction) and stays there to 300 sweeps.
- **Pass: NO (partly).** Trained on that population the pass's own-draw h/v/d is inside the gate (0.347/0.330/0.323 over the last 300 iterations of a 300 s run; no diagonal collapse), but **4.4 % of hard instances per cell are violated and 0 % of draws are valid**. The orientation mix is fixed; validity is not. No bench was run (the pre-check fails on violations; the greedy baseline fails the same way at 3.9e-2).

## Kernel (`castlegen/variants/_dimer_loop_kernel.py`)
Worm excursion on perfect matchings: pick a cell t uniformly, delete its dimer (t,h); the head h draws one of its six directions with prob 1/6 (off-strip = null step); if the neighbour is t, close (add (h,t)); else delete the neighbour's dimer (w,z), add (h,w), head := z. Excursions longer than `lmax` = 1024 steps are rejected (M stays). Every step is reversible with prob 1/6, open is 1/N, close 1/6, so an excursion and its reversal have equal probability; the cap keeps this (same length). So the kernel is symmetric, uniform p* (no soft terms) is invariant. Uses only the matching constraint and the six neighbours (no orientation knowledge). Long loops wrap the periodic x direction, which 3x3 blocks cannot.
Test `tests/variants/test_dimer_loop_kernel.py` (4 passed, 35 s): exact enumeration of all matchings on 4x3 and 2x4 periodic-x strips and every excursion up to a step cap gives transition matrix T with `T == T.T` (1e-12), row sums <= 1, new states valid; sweeps keep validity, cap 1 restores; repair makes valid.
Sweep = H*W = 2048 excursion attempts per world. Acceptance (closed within cap) 0.63; 46 % of attempts change the matching. Cost 4.6 ms per sweep per world (8 numba threads, cores 8-15).

## Population alone (`scripts/pop_sweeps.py`, 96 worlds per start, lmax 1024; runs/pop_l1024.{log,json})
Starts: collapsed cfast60 draws (the 87 % that are valid), probe tilings (mode 2), all horizontal. Exact: 0.3513/0.3244/0.3244.

| start | 0x | 1x | 10x | 100x | 300x |
|---|---|---|---|---|---|
| collapsed | .466/.534/0 | .348/.328/.325 | .351/.326/.323 | .351/.325/.323 | .349/.324/.327 |
| tiling | .503/.252/.245 | .347/.325/.328 | .351/.324/.325 | .350/.324/.326 | .351/.324/.326 |
| all horizontal | 1/0/0 | .352/.325/.323 | .352/.323/.326 | .351/.324/.325 | .352/.326/.322 |

Three starts in three different sectors agree to 0.004 after one sweep (per-world std 0.017). Within 0.01 criterion met.

## Pass trained on the refreshed population (`scripts/dsec_loop.py`, copy of dk_loop.py, cfast init, own draws as contexts, draw-wise ML 8 updates/iteration, nctx 32; GPU 1 shared with an Opus diagnostic, cores 8-15; gpu_before/after*.txt)
Own draws are 5-15 % violated, so before the worm they are projected onto perfect matchings (`repair`: unreciprocated cells become monomers, each walks as a worm head to another monomer). This projection is **not** p*-invariant; it is a repair of invalid draws only, and the worm then erases the start. Net effect: the ML population is close to exact uniform p* samples, independent of q. Stated plainly: that is the kernel-route amendment at its limit (the kernel mixes fully), not a gentle improvement of q.

| run | updates | refined pop h/v/d | pass own-draw h/v/d (mean of last 100 iters) | viol/cell | valid draws |
|---|---|---|---|---|---|
| 60 s, no repair (runs/pre60_worm_norepair) | 1744 | invalid (worm skipped all worlds) | .33-.39 each, noisy | 0.054 | 0 |
| 60 s, block sweeps + MT + repair + worm | 160 | .351/.325/.324 | .375/.31/.31 (first iters) | 0.105 | 0 |
| 60 s, repair + worm only (`fb_on=0 mt_on=0`) | 968 | .351/.325/.324 | ~.33-.35 each (32 worlds, std .02) | 0.055 | 0 |
| 300 s, same | 5088 | .351/.324/.324 | **.347/.330/.323** | **0.044** (plateau from 150 s) | 0 |

-log-lik loss per cell plateaus at 0.479 (dimer entropy about 0.43). Violations fall 0.153 -> 0.087 -> 0.052 -> 0.047 -> 0.044 and then flatten.

## What failed / caveats
- First 60 s run: worm did nothing because refined own draws were never valid (block sweeps do not repair 15 % violations); added `repair`.
- The 3 s/iteration of the block+MT+repair variant cost 10x the updates; the worm-only variant is 0.45 s/iteration.
- Validity of the pass is not fixed by exact-sample ML: a fixed-step local pass with this window/net leaves 4.4 % of hard instances violated. The orientation shift (0.5/0.5/0 collapse) was the training population's sector mix, and that is cured; a violation stage / penalty (as for ground) is the remaining gap. Not tested here.
- Only the `dk_loop` pre-check driver was used; the kernel is not wired into `ckernel_pop` or the bench (no variant registered), and no bench rows exist. 1 seed, 32-world contexts.

## Next
1. Add the worm to the pop-refresh as a registered variant and combine with the penalty / fixed stage that drives validity (the stage keeps ground at 0 violations).
2. Check a bench row only after the pass reaches <1e-3 violations.

# Wave C: validity for the dimers pass (follow-up, same day)
Question: does the worm refresh inside a `cfitpen_ml2`-style schedule, with and without the fixed stage (R 30), give a valid pass on dimers?

Built: `castlegen/variants/cdimerworm.py` registers `cdimerworm` (cfitpen_ml2 schedule: 16 ML updates and 32 penalty-only reverse-KL steps per iteration, lam 0-20, refresh 0.1 of nctx 32; the population refresh is one worm sweep after the repair projection; init = cfast default, since the ground init does not fit the dimers layout; ckernel_own's level-1/2 chains do not exist at T = 0, so `_build_uncached` builds only ML step, energy and generator) and `cdimerworm_stage` (`staged = True`, generation = CKPopHardPenStage's). Test: `tests/variants/test_cdimerworm.py` (1 passed). `loop_refresh` now sets unrepairable leftover monomers to code 0 (world stays invalid) after a no-penalty run crashed the energy check with p = -1.
Scripts: `scripts/precheck.py` (300 s train, then 16 held-out periodic worlds, raw and final), `scripts/smoke.py`. Runs: `runs/wc_stage_pen.*`, `runs/wc_stage_nopen.*`. GPU 1 shared with an Opus diagnostic, cores 8-15; clocks `runs/gpu_wc*`: before 1.8 GHz, 84 C (the after files were read once the card was idle, 210 MHz, so they say nothing about the run).

| arm (300 s, seed 0) | updates | draws | h/v/d | viol/cell | valid |
|---|---|---|---|---|---|
| penalty steps (rkl_n 32), raw | 9488 | own pass | .362/.308/.330 | 0.0296 | 0 % |
| penalty steps, stage R 30 final | | staged | .352/.318/.331 | 0.0043 | 0 % |
| no penalty (rkl_n 0), raw | 18608 | own pass | .346/.328/.327 | 0.0304 | 0 % |
| no penalty, stage R 30 final | | staged | .346/.322/.332 | 0.0042 | 0 % |

Per-world std of h/v/d about 0.017-0.02 over 16 worlds. Population during training: 100 % valid, .351/.324/.324 (worm). Penalty lambda hit its cap 20 by iteration ~40; the pass's penalty raw violation fraction at the end 0.008.

Answer: **validity did not come.** The orientation mix is within 0.02 of the exact values with or without the stage (staged final .352/.318/.331, within 0.01 on h and d but v is 0.006 low and the worlds are not valid), but violations per cell are 0.030 raw and 0.004 after the stage (about 8.5 violated instances per 2048-cell world); 0 of 16 worlds are valid in every arm. The penalty steps do not change the raw violation rate (0.0296 vs 0.0304) in 300 s, though they halve the updates. Bench not run (criterion: 0 final violations).
Not tried: stage R above 30, longer training, a pass window/net change. Next: the stage at R 100-120 on these params (generation-only), and a longer ML fit; the violation floor 0.03 raw is the same as the untrained pass (0.05) and the ML loss plateau (0.479 vs entropy 0.43), so the pass lacks capacity or window for the matching constraint.

# dgate: dimers under the corrected gate (0.351/0.324/0.324 +-0.02), 2026-10-16

## Answers
1. **No trained sampler passes** (valid or not). Closest: cbound_lam 0.44-0.47/0.39/0.14-0.17 (off by 0.09-0.12 in h and 0.16-0.18 in d). Only the untrained `greedy` baseline lands in the box (seed 0: .369/.312/.319), with 3.9% violations per cell, so it is not a valid sampler; its seed 1 (.375/.308/.316) misses on h by 0.024.
2. **Not within 480 sweeps, from either start, at 3x3.** 4x4 could not be run. Both starts stall off the gate and at different places, so the stage does not reach the gate from collapsed draws at any tested budget.

## 1. Record (h/v/d; violations per cell; gate = all three within 0.02)
| sampler (source) | h/v/d raw | viol/cell | gate |
|---|---|---|---|
| cfast b300 / gate3 / chot (bench) | .48-.51 / .49-.52 / .00-.001 | 1-2e-4 | fail |
| cfast b300smoke (bench) | .122/.878/0 | 8e-4 | fail |
| cpass v1, ctemp variants (bench) | .90-1.0 / .00-.10 / .00 (or d .98-.99) | 1e-4-1.3e-3 | fail |
| cbound_lam (bench, 2 seeds) | .439/.395/.166 ; .470/.388/.141 | 1e-4 | fail |
| cfast60 (dimers_kernel) | .466/.534/0 | 1.5e-4 | fail |
| rkl16 / rkl32 (final iter, raw) | .51/.48/.01 ; .496/.504/0 | 2.5e-3 ; 2.4e-4 | fail |
| dimers_pen32 (cr2_penalty, iter 280 raw) | .487/.513/0 | 3e-5 | fail |
| a1_cfast60 / a1_init / a2_cfast60 / a2_init (raw, last iter) | .338/.369/.293 ; .329/.366/.305 ; .360/.331/.309 ; .356/.307/.337 | 5.1e-2 to 5.8e-2 | a2_cfast60 and a2_init are inside +-0.02 on h/v/d but violate 5% of cells (a1: d off by 0.03 or v off by 0.045); the untrained init does as well as trained, so this is the penalty-free kernel regime, not a valid sampler. fail on violations |
| greedy baseline (bench) | .369/.312/.319 ; .375/.308/.316 | 3.9e-2 | in box (seed 0) but invalid |

Source: results/bench.jsonl per_seed stats; runs/*.jsonl last record. The pkls store parameters, not draws, so I used the jsonl logs.

## 2. Stage runs (64 worlds, lam 20, GPU 1 shared with stage2, cores 6-7; gpu_before/after.txt)
Command: `python reportables/2026-10-16_dgate/scripts/dgate_stage.py runs/<cfg>.json 64 <k3|mix>`. Starts: collapsed cfast60 draws (.466/.534/0, viol 1.5e-4) and probe tiling mode 2 (valid, .505/.248/.248). Logs every 30 sweeps in runs/k3.log, runs/mix.log; ms/world is cumulative time to k, shared GPU.

| stage | start | k=30 | 60 | 120 | 240 | 480 | viol/cell | ms/world at 240 (480) |
|---|---|---|---|---|---|---|---|---|
| 3x3 | collapsed | .399/.395/.207 | .386/.389/.225 | .385/.385/.230 | .381/.383/.236 | .382/.382/.237 | 1.5-3e-5 | 85 (154) |
| 3x3 | tiling | .410/.293/.297 | | .398/.301/.301 | .389/.304/.307 | .382/.310/.307 | 0 | 69 (139) |
| 1-site + 3x3 | collapsed | | .390/.391/.218 | .383/.387/.230 | .382/.384/.234 | | 3-4.6e-5 | 117 |
| 1-site + 3x3 | tiling | | .403/.297/.300 | .396/.300/.305 | .389/.305/.306 | | 0 | 106 |
| 4x4 | | not run | | | | | | |

- Gate reached at no checkpoint (every 30 sweeps to 480 for 3x3, to 240 for mixed), either start. Convergence criterion (both starts within 0.02) not met: the starts disagree by d = 0.237 vs 0.307 and v = 0.38 vs 0.31, and both h stay near 0.38 (0.03 above 0.351).
- The two starts settle in different places at flat rates (collapsed: no movement from k=120 to 480; tiling: h drifting down 0.001 per 30 sweeps). That is two sectors that local moves do not connect on this time scale (consistent with deq: h/v-rich vs h/d-rich), neither equal to the exact mean. Hypothesis, not tested: local block moves conserve the periodic-x winding/tilt sector, while the exact fractions average over sectors.
- Adding single-site moves changes nothing (single-site is frozen on hard dimers).
- 4x4: `FineBlockKernel(k=4)` builds (numpy, 3 s) but `internal_pruned` has 2,884,844 completions per tile, above its max_out=2e6 assert (`FineBlockKernel` does not pass max_out through; I did not edit shared code). Even raised, 64 worlds x 2.9M completions x ~100 tiles per sweep is not light on a shared 16 GB card. So 4x4 is unsupported as the code stands (runs/k4.log has the assert).
- Violations stay inside 2e-4 throughout.

## Not done / next
One seed, 64 worlds. 4x4 not run. Next: test the sector hypothesis cheaply (print winding per row / a conserved flux of the start vs end), and a start-mixture stage (random sector start). Without a sector-changing move a fixed local stage cannot hit 0.351/0.324/0.324.

# speed-1: faster fine-level amortised loop (sweep2a arm B, ground levels 0-2)

2026-10-14. Same kernel, same conditional, same weighted-ML objective. The GPU (RTX 3090) was shared with 2-3 other agents all session, so absolute timings vary by up to 2x between runs; before/after pairs below were run back to back where stated.

## Where the time went (old code, runs/profile_old.log, 120 s, arm B, 32 contexts, 6 cycles, 4 updates/iter)
| item | s / iteration | share |
|---|---|---|
| probe contexts (host numpy) | 0.080 | 5 % |
| own draws (gen pass + host flatten) + stats | 0.044 | 3 % |
| Gibbs (2,0) x 6 cycles | 0.084 | 6 % |
| **Gibbs (1,1) x 6 cycles** | **0.737** | **50 %** |
| energy curve | 0.011 | 1 % |
| **teacher-forced log q + ML update (4 updates)** | **0.524** | **35 %** |
| total | 1.48 | 21.6 states/s, 2.7 updates/s |

Causes. (1,1): the energy was evaluated for every one of the 480 completions of every block for every instance row (about 14k rows x 480 x 32 worlds per colour class). ML: the MLP sat inside the 16-step colour scan, so autodiff backpropagated through a scan (step 43 ms at B=8, forward only 7 ms).

## What was changed (all semantics-preserving)
1. **Factored block energy** (castlegen/kernels/amortised.py, `JaxBlockGibbs`, default `fast=True`; old path kept as `sweep_ref`). Every instance row of ground has either all members inside the block (energy depends only on the completion: computed once at construction, world independent) or exactly one member inside (energy is a per-world field over that variable's domain, <= 6 values: computed per sweep, then summed over the block's variables with one exact one-hot matmul at HIGHEST precision). Anything else falls back to the old path. Same Gumbel draws: with the same key the new sweep gives bit-identical states (runs/check_fast.log: 0.0 differing entries over 3 sweeps, both kernels). (1,1) sweep of 32 worlds 235 ms -> 17.5 ms (13x); (2,0) 4.8 -> 1.1 ms.
2. **Teacher forcing split** (castlegen/kernels/logq.py): `_collect_inputs` scans the colours once with no parameters (features, soft, mask, forced values; stop_gradient), then the MLPs run over all 16 colours of a channel at once (`_forced_pass`; old one kept as `_forced_pass_ref`). Gradient vs old: max diff 2.4e-3 on max |grad| 150 (float32 order), loss value identical to 7 digits (runs/bench_ml.log). Step B=8: 43 -> 9.8 ms; B=16: 77 -> 17 ms.
3. **Per-level teacher forcing**: `score_levels=(0,1,2)` skips the scans of the frozen levels 3-4 (their q does not depend on the trained parameters; coarse values are filled from the state). Gradients of levels 0-2 identical (same max diff as above); the logged loss and infeasible counts then cover levels 0-2 only (log key `loss_per_cell_fineonly`). Extra gain small (8.7 vs 9.8 ms) once item 2 is in.
4. **Fused loop pieces**: `make_refine_scan` (all cycles of [(2,0),(1,1)] in one lax.scan jit, energy only at start and end), `make_multi_update` (the 4 updates in one lax.scan jit, one host sync), GPU-side flatten of own draws, and a worker thread that builds the next iteration's probe lifts and designed values while the GPU runs (the probe construction is host numpy: 80 ms per 16 probes).

## Before / after (arm B, 32 contexts, 6 cycles, 4 updates/iter)
Profile pairs ran back to back on the same cores (profile_old.py then run_loop_fast.py with PROF=1, 60 s each); a second fast profile (prof_fast4, 40 s) was in a quieter GPU window.

| | old | new (back to back) | new (quiet window) |
|---|---|---|---|
| s / iteration | 1.61 (1.48 earlier run) | 0.43 | 0.24 |
| contexts refined / s | 19.9 (21.6) | 74.6 | 134 |
| ML updates / s | 2.5 (2.7) | 9.3 | 16.8 |
| ctx (probe+own+stats) s/iter | 0.154 | 0.09 | 0.05 |
| Gibbs, 6 cycles, s/iter | 0.93 | 0.134 | 0.071 |
| ML, 4 updates, s/iter | 0.52 | 0.20 | 0.11 |

Speed-up 3.7x (contended) to 6x (quiet). Quiet-window microbenchmarks: Gibbs (1,1) sweep 13x, ML step 4.4x. What is left: ML (about 45 %), Gibbs, own-draw generation (the full pass for 16 states) and the host-side designed values. Remaining ML time is larger than the 4 x 10 ms microbenchmark because of GPU sharing and the per-iteration host syncs of the diagnostics.

## Nothing changed check (300 s arm B with the fast code, runs/fastB_s0, 564 iterations, 2256 updates versus about 552 for sweep2a B s0; eval_ground.py unchanged, 32 seeds x 4 probes)
| | upd | L3 fill cost | L3 soft | L3 own | L4 fill | L4 own | line_match L3 clamp | nv L3 fill |
|---|---|---|---|---|---|---|---|---|
| sweep2a B s0 | 552 | -0.106 | 0.140 | 0.35 | 0.68 | 0.73 | 0.734 | 2.0e-4 |
| sweep2a B s1 | 1016 | -0.111 | 0.138 | 0.33 | 0.64 | 0.71 | 0.733 | 1.9e-4 |
| **fast B s0** | 2256 | **-0.114** | **0.129** | 0.31 | 0.60 | 0.69 | 0.743 | 1.0e-4 |

Same plateau (sweep2a already found the loop saturates after 100-300 updates); the small differences are within seed noise or slightly better with more updates. Own-draw line_match stays 0.508 (levels 3-4 untrained). Different RNG stream than sweep2a (probe jitters from a separate generator), so runs are not bit-comparable; the kernel and ML pieces are individually verified against the old code (above).

## Tests
tests/kernels/test_amortised.py (added: fast sweep == reference sweep for (2,0) and (1,1), lam 20 and None, with a violating state; refine scan), tests/kernels/test_speed_ml.py (new: score_levels gradients equal, multi-update equals sequential), test_logq.py (tolerance 1e-5 -> 1e-4 for the sum-order change), test_exact_block.py: all pass (17 + 9 passed).

## Not done / failed
- No constitution change needed. Possible next: own-draw generation (full pass) is now the biggest non-ML context cost; the (1,1) kernel could skip blocks whose outside values did not change, not built.
- Timings are not clean: shared GPU.

## Files
Modified: castlegen/kernels/amortised.py, castlegen/kernels/logq.py, tests/kernels/test_amortised.py, tests/kernels/test_logq.py. Created: tests/kernels/test_speed_ml.py, reportables/2026-10-14_speed1/ (report.md, run_loop_fast.py, run_loop_old.py, profile_old.py, check_fast.py, bench_ml.py, env.sh, amortised_old.py, logq_old.py, eval_ground.py, clamp_tools.py, gate.py copies, runs/). Loop command: python reportables/2026-10-14_speed1/run_loop_fast.py B <seed> <budget_s> 6 <outdir>.

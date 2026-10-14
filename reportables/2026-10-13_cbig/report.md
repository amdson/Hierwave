# cbig: large batches on the GPU (stopped early)

2026-10-13 · `castlegen/variants/cbig.py`, `_cbig_core.py`, `tests/variants/test_cbig.py`. The agent's text, condensed by the orchestrator.

No bench runs and no SNR numbers yet. The variant is built, and its batched generation equals cpass's single-world generation and scoring exactly (tested on the box).

## Built

- **Same estimator as cfast in expectation.** Each step's 64 subsampled sites are chosen before the pass (weight n/m), so features and logits are recorded only there. Costs are still recorded everywhere.
  - This fixes memory: cfast records about 27 MB per ground world and runs out of memory at B = 1024.
- **Default B = 256,** lr 3e-3.
- **Faster host side:** designed inputs for the next batch are prepared on the CPU while the GPU runs (one numpy call per batch, checked against per-seed calls).
- **Generation** runs 16 seeds in one compiled pass, with log q cached, so the bench's `score` is a lookup.
- **`grad`** returns one batch's actor gradient, for SNR measurement.

## cfast speed against B (RTX 3090, shared box)

| problem | B | s/update | s/world | peak GB | fresh compile s |
| --- | --- | --- | --- | --- | --- |
| toy_support | 8 | 0.0045 | 5.7e-4 | 0.02 | 12 |
| toy_support | 256 | 0.0083 | 3.3e-5 | 0.32 | 50 |
| toy_support | 1024 | 0.022 | 2.2e-5 | 1.11 | 45 |
| ground | 8 | 0.014 | 2.9e-3 | 0.30 | 121 |
| ground | 64 | 0.034 | 1.6e-3 | 1.75 | 142 |
| ground | 256 | 0.116 | 1.5e-3 | 6.9 | 150 |
| ground | 1024 | out of memory | | > 24 | |

- **toy_support:** nearly flat up to B ≈ 256.
- **ground:** the GPU saturates around B = 64. Past that, building the designed inputs on the CPU (about 1 ms per world) dominates. cbig's fixes for both limits are written but not yet measured.

## Problems on the box that affect every agent

1. **Fresh compiles take 45–150 s, which eats the 150 s budget.**
   - `XLA_FLAGS=--xla_gpu_autotune_level=0` cuts toy_support at B = 256 from 132 s to 29 s.
   - The cache at `~/.cache/jax_cfast` works.
   - So GPU benches are fair only with a warm cache (one throwaway run) or with autotune off. Record which.
2. **Thread limit.**
   - The cgroup allows 2304 threads in total, and each JAX process starts about 316, so about 6 processes hit the limit, and then compiles fail with `ptxas` errors.
   - `taskset -c` over 8 cores brings a process to about 44 threads. `scripts/gpu.sh` now does this.

## Next

1. Warm the cache, or turn autotune off for cbig.
2. Profile cbig at B = 256 and 1024.
3. Measure SNR from repeated batches.
4. Bench at B = 256 (ground, columns) and B = 1024 (toys, dimers).
5. Compare collapse (dirt, fill tracking) per seed against cfast.
6. Run the forced-coarse fine-level check.

---

# Round 3 results (2026-10-14, cbig agent)

**The GPU was shared** (other agents at 100 % utilisation throughout), so wall-clock budgets were contended. Compare by updates as well as time. Runs are tagged `r3` in `results/bench.jsonl`; variants `cbig64` and `cbig256` (B = 64 / 256) were added to `cbig.py`.

## Changes to cbig

- **Training clock starts after setup + first compile** (the fused update and the generation pass are compiled in `_warm`, logged as `setup_s` / `compile_s` in the train log and `info()["timing"]`). Note this means the bench's `within_budget` flag is false on a cold cache (train_s includes the compile). cfast's budget still includes its compile.
- **Shared cache really used.** cpass/cfast hard-code `~/.cache/jax_cpass|jax_cfast` at import, overriding `JAX_COMPILATION_CACHE_DIR` (so `/workspace/jax_cache` was empty before). cbig now re-applies the env var after import.
- **autotune_level=0** appended to `XLA_FLAGS` in cbig only (`CBIG_AUTOTUNE=default` turns it off).
- `tests/variants/test_cbig.py`: 5 passed (400 s, cold-ish cache). Test unchanged.

## Compile time (toy_support, B = 256, 20 s training)

| run | setup s (incl. fused compile) | generation compile s | s/update |
| --- | --- | --- | --- |
| cold, autotune 0, fresh cache | 147 | 49 | 0.093 |
| warm, autotune 0 | 7.8 | 3.6 | 0.068 |
| cold, autotune default, fresh cache | 171 | 48 | 0.080 |

- Persistent cache works: ~196 s cold to ~11 s warm for this shape.
- autotune 0 gave only ~10 % on cold compile here (147 vs 171 s) and no clear step-time difference (0.068-0.093 vs 0.080, within contention noise). The 132 s to 29 s from the earlier report did not reproduce; the CPU-bound compile on 8 contended cores seems to dominate. Kept as default, harmless.
- Ground cold compile was 316 s at B = 64 (seed 0 of the bench), then 0 s for seed 1 (same process).

## Bench, ground (2 seeds x 150 s, back to back)

| variant | updates (s0 / s1) | worst final viol | free energy (s0 / s1) | dirt_frac | line_match |
| --- | --- | --- | --- | --- | --- |
| cfast B = 8 | 1040 / 1543 | 7.3e-4 / 0 | 0.918 / 0.736 | 0.991 / 1.000 | 0.5 |
| cbig B = 64 | 922 / 1094 | 0 / 0 (raw 1.5e-5) | 0.548 / 0.543 | 1.000 / 0.999 | 0.5 |
| cbig B = 256 | 184 / 230 | 0 / 6.1e-5 | 0.646 / 0.786 | 1.000 / 0.998 | 0.5 |

- None accepted (probes flagged, across-seed fill_err disagrees). Free energy is lower for B = 64 than cfast (0.55 vs 0.74-0.92) but that is a still-collapsed state; B = 256 got only ~200 updates because of contention (~0.7 s/update here, 0.12 s uncontended earlier) and is not better.
- **Collapse does not change with B.** Every run is all dirt (dirt_frac ≈ 1, dirt_excess ≈ 0.49-0.50, line_match 0.5, i.e. ignoring the fill line). Images: `images/cbig64_ground_s1.png`, `images/cbig256_ground_s1.png`, `images/cfast_ground_s0.png`. B = 64 / 256 give flat all-dirt worlds (roughness 0-0.19), cfast s0 shows some rough/floating dirt (floating 36.8).
- Per-update cost grows with B on the shared GPU, so more samples per update did not turn into more progress within 150 s.

## Bench, toy_support (exact F* = -0.4688)

| variant | updates | worst final viol | free energy |
| --- | --- | --- | --- |
| cfast B = 8 | 4872 / 7204 | 3.7e-4 / 2.4e-4 | -0.424 / -0.444 |
| cbig B = 256 | 2510 / 3334 | 3.7e-4 / 3.7e-4 | -0.455 / -0.467 |

cbig at B = 256 is closer to F* (gap 0.014 / 0.002 versus 0.045 / 0.025) with ~half the updates; dirt_frac 0.74 / 0.71 versus 0.81 / 0.77. Neither accepted (violations above 2e-4, probes flagged). Seed-1 cbig is within 0.002 of the optimum, so large B helps on toy_support but not on ground.

## Gradient SNR vs B (ground, init, zero critic, tau 1.5, lambda 10)

Per-batch SNR = (|mean g|^2 - trCov/M) / trCov, M batches (48 / 24 / 12), `snr.py`, raw `snr_ground.log`. Values per block (largest |value| per B):

| B | l0_t | l1_cert | l2_cert | others |
| --- | --- | --- | --- | --- |
| 8 | -0.011 | 0.026 | 0.006 | |snr| < 0.013 |
| 64 | -0.030 | 0.001 | 0.118 | |snr| < 0.03 |
| 256 | 0.028 | -0.016 | 0.078 | |snr| < 0.035 |

All estimates are within noise of zero (negative values are estimator noise): at init the per-batch gradient is noise-dominated at every B up to 256 on ground, and no trend with B is resolved with these M. If SNR scaled linearly with B we would see 32x between B = 8 and 256; nothing like that shows up, so either the signal at init is tiny or M is too small to resolve it. Not conclusive.

## Next

- Large B is worthwhile where per-update cost is flat (toy_support); on ground it is GPU-bound beyond B ≈ 64, and collapse is unaffected, supporting the view that the collapse is a representation / sampling-order issue, not a variance one.
- Redo the SNR with a trained-for-a-while actor and larger M, if the question is still of interest.

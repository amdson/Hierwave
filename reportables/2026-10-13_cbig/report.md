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

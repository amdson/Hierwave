# speed-3: faster generation (own draws), identical draws

2026-10-14. GPU (RTX 3090) shared with 3+ other agents the whole time: kernel durations and wall times swing 2-4x between runs (median per call is often 3x the minimum). All comparisons below are interleaved old/new in the same process; "min" is the quiet-window figure, "median" the contended one. Old code = reportables/2026-10-14_speed3/_cfast_core_old.py, _cpass_core_old.py (copies before the change).

## Where the time goes (ground, W = 64, vmapped train_pass record=False = the ckernel/cfast own-draw generator)
- 144 channel-steps per world (5 levels x 16 colours x 1-2 channels), tiny per step (1 to 256 sites per world). Old code: ~2990 GPU kernels per call (21 per channel-step).
- Ablation (runs/ablate_ground_*.log, noisy): features + MLP is ~75 % of the pass at B = 32 (removing either removes both, since the MLP consumes the features); cond (`lay.cond`, mask/lookahead) and the argmax/state scatter are small; Gumbel hashing 10-30 %. Level split (level-only runs, not additive): level 1 (two channels, 503 features) is the largest, then levels 0, 2, 3; level 4 (1 site) ~ launch overhead only.
- Profiler (prof.py; kernel times inflated by sharing): dominant kernels are the window-gather + concatenate fusions that build `feat` (levels 0/1) and the MLP GEMMs. At B = 32 the pass is launch-latency bound (~4.6 us/kernel x 18 kernels x 144 steps ~ 12 ms); at B = 128 it is bandwidth bound on the feature tensor.
- Host side (host_prof.py, runs/host_prof.log): `problem.designed_values` is numpy, ~0.6 ms per world on ground (21.7 ms for 32 worlds, 57 ms for 128) plus 1-3 ms stack/transfer; this is as large as the device pass in quiet windows (B = 32: 8-12 ms). Not changed (bench/problem.py and ground.py are not mine): proposal, see below.

## What changed (castlegen/variants/_cfast_core.py; `_cpass_core.coloured_pass` delegates to it when record=False, critic_extra=None, resid off)
1. Everything depending only on (level, colour): site coordinates, window gather indices, in-range flags, position one-hot, row coordinate: built once in numpy (`_geom`), enter the scan as xs.
2. Gumbel noise for all 16 colours of a channel computed in one batch before the scan (same hash, same per-domain-value layout; replay faithfulness untouched).
3. Context features read from an encoded image per level (embedding x known, known; one array per level, columns per channel) kept in the scan carry and updated by the same scatter that writes the draw; the parent level's image is the previous level's final carry. One gather per window instead of a per-channel read + embedding gather + mask. Layouts with count-residual features (`resid`) keep the old path.
Tried and dropped: optimization_barrier on the gathered window (no gain, runs/abn1), separate image per channel (v1, equal within noise).

## Equality
- tests/variants/test_gen_speed.py (new): old vs new on 6 layouts (toy_support periodic and strip x0 = 8, toy_modes, dimers, ground periodic W = 64 and strip W = 48 x0 = 16): train_pass record and not, with and without override; coloured_pass totals and log q; forced scoring. train_pass records (draws, features, logits, costs) bit-identical on CPU; coloured_pass worlds bit-identical, its per-step totals / log q equal to rtol 1e-5 (max 4.8e-7 abs on CPU: summation order).
- GPU: final states bit-identical (0 differing cells) at B = 32 and 128 on ground and toy_support (abn.py, abn_final.log); 160 worlds in total each run, generated with random nonzero output layers (scale 1). Max logit difference on GPU not measured separately (draw equality is the criterion; CPU float bound above).

## Before / after (final code, interleaved in one process, W = 64; runs/abn_final2.log is a quiet-GPU window, p10 ~ median)
| problem | B | old ms (min / med) | new ms (min / med) | speedup | worlds/s old | worlds/s new |
|---|---|---|---|---|---|---|
| ground | 32 | 8.0 / 8.5 | 7.3 / 7.6 | 1.10x | 3998 | 4411 |
| ground | 128 | 17.2 / 23.1 | 15.7 / 22.5 | 1.10x (med 1.03x) | 7448 | 8171 |
| toy_support | 32 | 1.5 / 1.7 | 1.7 / 1.8 | 0.9x | 20786 | 19301 |
| toy_support | 128 | 2.2 / 2.4 | 1.8 / 2.1 | 1.2x | 58406 | 69211 |
Contended-window runs of the same comparison (runs/abn_final.log, abn1, abn2): ground B = 32 1.1-1.3x, B = 128 1.1-1.3x (min), medians 1.0-1.15x. Honest summary: **about 1.1x on ground, nothing measurable on toys; the gain is the size of the run-to-run noise**. The first profile (GPU busy 85 -> 32 ms per call, 2990 -> 2550 kernels) was taken in a contended window and overstates it.
Single-world `generate()` (cpass; coloured_pass + repair + host copy): ground 15.2 -> 12.1 ms median (min 9.6 vs 10.4), toy_support 4.3 -> 3.3 ms median (gen_single.log).

## Why not more
Kernel count per channel-step is already ~18 and the remaining time is the feature tensor (levels 0/1) and the MLP GEMMs. Bigger wins would change numerics (first-layer matmul as a sum over window gathers: different summation order, draws could flip) or need CUDA graphs. Per-iteration, own draws are 16-32 worlds = 5-10 ms of ~500 ms; the host `designed_values` is the same size.

## Proposals (not built)
- Batched/cached `designed_values` for ground (depends only on seed and x-columns of a 1-D line; vectorise over seeds) and/or build next iteration's designed arrays on the prefetch thread (speed-1's thread): saves ~0.6 ms/world.
- Pass `record=False` own draws through `fast_coloured_pass` (already used by `generate`, `score`).

## Tests run (CPU, box flags): runs/tests_cpu.log (first block there is the superseded strict version of my own test; final run runs/tests_gen_speed.log: 6 passed). All passed: test_cstart 11, test_logq 8, test_creplay 8, test_cfast_cone 10, test_cpass_gradient 7, test_gen_speed 6. Note: running them in one pytest process hit a CPU-JIT 'Failed to materialize symbols' error (environment), so each file ran in its own process.
tests/variants/test_cpass_gradient.py, test_cfast_cone.py, test_creplay.py, test_cstart.py, tests/kernels/test_logq.py, tests/variants/test_gen_speed.py.

## Files
Modified: castlegen/variants/_cfast_core.py (train_pass rewritten, fast_coloured_pass added, old body kept as _train_pass_slow for resid), castlegen/variants/_cpass_core.py (coloured_pass delegates). Created: tests/variants/test_gen_speed.py; reportables/2026-10-14_speed3/ (report.md, _cfast_core_old.py, _cpass_core_old.py, bench_gen.py, ab.py, abn.py, ablate.py, prof.py, host_prof.py, gen_single.py, env.sh, runs/).

# speed-1b: faster coarse kernel cycle (block_mh, coarse_flip, coarse Gibbs) in ckernel

2026-10-14. Same proposals, same MH ratios, same public API. GPU shared with other agents throughout, so absolute times swing by 2-5x between runs; the old/new pairs that matter are run in the same process or back to back.

## Profile (ckernel iteration on ground: 32 contexts, 4 flips + 4 L2 MTM moves per world, K 4 / 8, 8 ML updates; runs/profile_old2.log is the pre-change coarse kernels, profile_new2.log the new)
| s / iteration | old | new |
|---|---|---|
| contexts (host probes + own draws) | 0.19 | 0.09 |
| coarse flips (L3, L4, K=4) | 2.04 | 0.059 |
| coarse Gibbs L4 + L3 | 0.13 | 0.024 |
| L2 MTM moves (K=8) | 4.01 | 0.035 |
| fine Gibbs (3 cycles) | 0.12 | 0.025 |
| ML (8 updates, all levels) | 0.61 | 0.26 |
| total | 7.1 | 0.50 |
| flip moves / s | 63 | 2180 |
| L2 MTM moves / s | 32 | 3630 |
| updates / s | 1.1 | 16.1 |

The old column ran while tests used the GPU, so it is inflated. In-process pair (bench_chain.py, 4 moves x 32 worlds, min of 6, GPU at 99 % from other jobs): MTM L2 K=8 7029 -> 124 ms (57x), flip L3/L4 K=4 1239 -> 279 ms (4.4x). An earlier contended baseline (before any change, other load) had flip 0.42 s, MTM 0.28 s, total 2.1 s per iteration. Honest range of the gain: 4x to 50x on the moves, about 4x to 14x on the iteration. Acceptance statistics over the 6 profiled iterations are identical to all printed digits in old and new (flip L3 0.6369, L4 0.3930, MTM L2 0.8958).

## Where the time was
A move's pass ran all 16 colour steps of every level <= Lmax over the whole image, with the block selected by a mask, for K candidates and then once more for the reverse scoring. It is latency bound (hundreds of sequential tiny steps), so the work, not the FLOPs, mattered: 2 passes in sequence x 80 steps per move.

## Changes (castlegen/kernels/block_mh.py, coarse_flip.py, amortised.py)
1. `compact_pass`: only the block's sites (static slots ceil(2^(Lmax-l)/4)^2 per colour step, dynamic origin) and only colour steps that contain a block site (at most min(16, 4^(Lmax-l)) per level; the active colours are found per world in pass order). The key sequence of the full pass is regenerated and the Gumbel noise of each step is drawn at full-site shape then gathered, so equal keys give equal candidates.
2. Candidates and the reverse (teacher-forced) scoring of the old block run as one K+1 batch (`use_forced` flag on the last row): one sequential pass per move instead of two.
3. Public API unchanged: `block_pass(..., blk=None)` still the full-image pass (`block_pass_ref`, `flip_pass_ref` kept); `mh_move`, `flip_move`, `make_chain`, `make_flip_chain`, `make_accept` take the same arguments and return the same records. Edits only reach processes started afterwards.
4. Energy differences: the pass already sums soft/violations only over block sites, i.e. exactly the factors touching the block, so no separate field/internal split was needed for the MH ratio.
5. Coarse Gibbs (`JaxBlockGibbs`, L4 has 39321 completions in chunks of 4096): the factored fast path of speed-1 now also covers chunked kernels (noise reproduced chunk by chunk). L4 sweep 410 -> 145 ms, L3 255 -> 28 ms for 32 worlds; bit-identical states (runs/check_gibbs.log in this folder's check_gibbs.py output).

## Equivalence (check_equiv.py, runs/check_equiv.log; old modules block_mh_old.py / coarse_flip_old.py loaded side by side, 64 moves each, same keys)
Decisions agree in 100 % of moves for MTM L1/L2/L3, flip uniform and q at L3/L4. Max |log r| difference 5e-3 (float summation order), max dE difference 0.0. New states identical in all moves, except one move in 64 in the first run of "flip q L4" (a near tie in a Gumbel argmax; check_tie.py with 256 such moves: all identical, runs/check_tie.log). Whole 6-iteration chains give identical acceptance rates (above).

## 300 s ckernel run (run_ck.py unchanged from sweep2b, default cfg; setup incl. compile 54 s in both)
| | updates | s / iter | gate trained L3 fill | L4 fill | own cost | line_match own | dirt own |
|---|---|---|---|---|---|---|---|
| sweep2b ck300 | 440 | ~4.7 | -0.110 | -0.125 | -0.084 | 0.626 | 0.513 |
| **fast** (runs/ck300_fast) | **2312** | 0.86 | -0.113 | -0.146 | -0.119 | 0.703 | 0.499 |

5.3x the updates in the same 300 s; the gate (eval_ground.py unchanged) is at least as good, own line_match up (0.70 vs 0.63). Part of the gain over sweep2b also comes from speed-1's ML and fine-Gibbs speedups, which sweep2b did not yet have.

## Tests
tests/kernels/test_block_mh.py, test_coarse_flip.py (8 passed), test_amortised.py, new tests/kernels/test_speed_coarse.py (compact == reference passes: draws, log q, soft, violations, flip modes; 2 passed 1 skipped).

## Left / suggestions (ckernel.py is not mine)
ML is now 50 % of an iteration (8 single updates with host syncs; `make_multi_update` and `score_levels` from speed-1 would fuse them) and contexts 20 % (thread prefetch as in run_loop_fast). Remaining flip cost: sequential 4 moves x K+1 passes; moves in a world are state dependent.

## Files
Modified: castlegen/kernels/block_mh.py, castlegen/kernels/coarse_flip.py, castlegen/kernels/amortised.py. Created: tests/kernels/test_speed_coarse.py; reportables/2026-10-14_speed1b/ (report.md, block_mh_old.py, coarse_flip_old.py, profile_ck.py, check_equiv.py, check_tie.py, check_gibbs.py, bench_chain.py, run_ck.py + eval copies, runs/).

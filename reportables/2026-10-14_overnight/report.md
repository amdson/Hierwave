# Overnight session, 2026-10-14: the kernel route, pushed further

Started 08:01 UTC. Budget: six hours, two Opus + up to four Sonnet agents. Constitution frozen (amendment 2026-10-14 in force); change requests go to `notes/constitution_change_requests.md`.

Goals: (1) ground's coarse levels trained by the kernel route; (2) own-draw line_match off 0.50 and the v2 bench; (3) speed-ups for the kernels, ML update and pass; (4) breadth once ground moves.

## Wave log (newest first)

(waves are appended here as they finish)

### Wave 1: sweep2b, coarse kernels and the first own-draw line-following (commits through df29fb5)
- Coarse flip and any-level single-cell Gibbs kernels built and verified (8 tests; |z| <= 1.25 on toys; dropping the q ratio gives z 800).
- Bottom-up hypothesis confirmed: with kernel-a's fitted fine levels, q-subtree proposals from probe lifts are accepted .79 / .44 / .03 at L2 / L3 / L4 (cfast: 0 / 0 / 0); the L2 log-ratio gap shrinks from -35 to -1.4 nats.
- `ckernel` (fine Gibbs + MT subtree + coarse kernels, ML on all levels, contexts = probe lifts + own draws): own-draw line_match 0.50 -> 0.62 in 440 updates, the first variant above 0.50; dirt 0.83 -> 0.52; v2 probe rho 0.82-0.86 at L1-3, starvation <= 10 % per level (cfast 76-100 %); F -0.112 vs cfast 0.774. NOT accepted: worst violations 3.2e-3 (budget 2e-4), plain ML has no violation pressure.
- The gain comes from probe-lift contexts: an own-draws-only run stays collapsed (dirt 0.90, line_match 0.50). Ruling: training the coarse levels on probe lifts is outside the frozen rules (constitution change request #1, with the numbers); the probe arm is kept as a labelled upper bound, the main line must move own draws out of dirt with kernels applied to own draws.
- Wave 2 launched: wave2-explore (own draws only: bottom-up curriculum, tempered/multiple-try coarse flips, annealed subtree blocks, delayed acceptance) and wave2-bench (violation pressure via kernel pricing, state weighting and the mixed reverse-KL term; bench the upper-bound arm; dimers check). speed-1 (fine loop profiling) and speed-2 part 2 (valid annealed blocks) still running.

### Wave 1 (partial): speed-2, colour Gibbs primitive (commit 2a88434)
- Generic GPU-batched single-cell exact Gibbs over a colour class, all worlds at once, plus annealed block moves with AIS weights. 22 tests pass (conditionals match brute force at every level/channel; invariance max |z| 2.3; AIS log Z within 0.013 of exact on both toys).
- Throughput on ground (W = 64, 64 worlds): level-0 sweep 1.4e8 cells/s; annealed level-3 subtree block (tau 8 -> 1, 32 steps) 670 blocks/s, 173 at 128 steps. Kernel-a's per-block joint kernel: 2.3e3 blocks/s.
- Failure: annealed blocks from random starts keep about 10 hard violations each (single-cell moves cannot cross the hard count constraints), so they are not yet usable training states. Follow-up launched: anneal lambda too, start from valid states, interleave kernel-a's joint fine-block move.
- Still running at this point: sweep2b (coarse flips / ckernel), speed-1 (fine loop profiling).


## Starting point
Sweep 1 and sweep 2a are committed (kernel_a, kernel_b, kernel_c, sweep2a reports). sweep2b (coarse flips, coarse Gibbs, `ckernel` variant) is still running at the start of this session.

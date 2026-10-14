# Overnight session, 2026-10-14: the kernel route, pushed further

Started 08:01 UTC. Budget: six hours, two Opus + up to four Sonnet agents. Constitution frozen (amendment 2026-10-14 in force); change requests go to `notes/constitution_change_requests.md`.

Goals: (1) ground's coarse levels trained by the kernel route; (2) own-draw line_match off 0.50 and the v2 bench; (3) speed-ups for the kernels, ML update and pass; (4) breadth once ground moves.

## Wave log (newest first)

(waves are appended here as they finish)

### Wave 1 (partial): speed-2, colour Gibbs primitive (commit 2a88434)
- Generic GPU-batched single-cell exact Gibbs over a colour class, all worlds at once, plus annealed block moves with AIS weights. 22 tests pass (conditionals match brute force at every level/channel; invariance max |z| 2.3; AIS log Z within 0.013 of exact on both toys).
- Throughput on ground (W = 64, 64 worlds): level-0 sweep 1.4e8 cells/s; annealed level-3 subtree block (tau 8 -> 1, 32 steps) 670 blocks/s, 173 at 128 steps. Kernel-a's per-block joint kernel: 2.3e3 blocks/s.
- Failure: annealed blocks from random starts keep about 10 hard violations each (single-cell moves cannot cross the hard count constraints), so they are not yet usable training states. Follow-up launched: anneal lambda too, start from valid states, interleave kernel-a's joint fine-block move.
- Still running at this point: sweep2b (coarse flips / ckernel), speed-1 (fine loop profiling).


## Starting point
Sweep 1 and sweep 2a are committed (kernel_a, kernel_b, kernel_c, sweep2a reports). sweep2b (coarse flips, coarse Gibbs, `ckernel` variant) is still running at the start of this session.

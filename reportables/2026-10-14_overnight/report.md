# Overnight session, 2026-10-14: the kernel route, pushed further

Started 08:01 UTC. Budget: six hours, two Opus + up to four Sonnet agents. Constitution frozen (amendment 2026-10-14 in force); change requests go to `notes/constitution_change_requests.md`.

Goals: (1) ground's coarse levels trained by the kernel route; (2) own-draw line_match off 0.50 and the v2 bench; (3) speed-ups for the kernels, ML update and pass; (4) breadth once ground moves.

## Wave log (newest first)

(waves are appended here as they finish)

### Wave 2 (partial): wave2-bench, violation pressure on the upper-bound arm (probe-lift contexts, change request #1)
- exp(-20V) weighting does nothing: probe-refined states have zero violations but nearly all own-refined states violate, so the weights drop them and the fit becomes probe-only ML. The 3.2e-3 is 13 support3 violations in 16 worlds: the coarse levels choose certificates the children cannot realise next to their neighbours; repair then creates L4 violations.
- Mixing cfast's reverse-KL update (32 steps per iteration, lr 1e-3) brings violations to 5e-5..4.3e-4 at line_match 0.57-0.59; at cfast's lr 3e-3 it collapses the model (dirt 0.99).
- Bench `ckernel_rkl32`, 2 seeds, 300 s: seed 1 within budget, seed 0 misses by 4 of 16384 instances at L2 (2.4e-4). F -0.012 vs cfast 0.774; line_match 0.56 / 0.58; v2 rho 0.33 / 0.80 / 0.47 / 0.75 / 0.47; starvation 0 / 4 / 10 / 24 / 42 % (the RKL mix re-introduces starvation at L3-4 against ML-only's 10 / 3.5 %). No accepted row yet. The bench compiled cold on seed 0 (254 of 300 s), so the row is also budget-starved.
- Follow-up launched (wave2-bench2): reverse-KL on levels 2-4 only at low lr, ML on 0-2, infeasible-priced L4 kernels, fine sweep after every coarse move, warm compile before the bench.

### Wave 2 (partial): dimers-kernel
- Generic k x k fine-block Gibbs for T = 0 problems (`castlegen/kernels/fine_block.py`, 7 tests exact on 4x8 periodic). On dimers, k = 2 blocks can only do square flips and never change the diagonal count; k = 3 blocks contain every rhombus and do: pure kernel chains take cfast's collapsed draws (.47/.53/0) to .38/.39/.23 in 600 cycles, though two different starts had not met yet (not equilibrated). k = 4 is too large for the GPU path on the bench region (2.9e6 completions).
- So local block moves CAN move the orientation mix on dimers, which answers the open question from the kernel-route discussion, given a big enough block.
- The ML loop on dimers lost validity (own-draw violations rise from 2e-4 to ~0.05 per cell within 50 updates, both with hard-priced blocks and with exp(-20V) weights), so there is no trained orientation result yet. Same disease as ground's 3.2e-3: plain ML has no violation pressure. Waiting for wave2-bench's violation fix before a dimers rerun.

### Wave 2 (partial): speed-1, fine loop 3.7-6x faster
- Factored block energy (internal part once + per-world field; bit-identical Gibbs states, 13x on the (1,1) sweep), teacher forcing over all 16 colour steps at once (4.4x on the ML step), level skipping for frozen levels, cycles and updates fused under lax.scan, context prefetch. Arm B: 0.24-0.43 s/iteration against 1.61; 134 states/s against 20; 16.8 updates/s against 2.5. Gradients within 2.4e-3 of the old code (float32 summation order; test_logq tolerance 1e-5 -> 1e-4).
- 300 s gate unchanged within noise (L3 fill -0.114, soft 0.129, line_match 0.743, L4 fill 0.60). Own-draw line_match 0.508: the fine loop alone still does not move own draws. 26 tests pass.
- Follow-up launched (speed-1b): the coarse kernel cycle (block_mh, coarse_flip), which is now the main cost (~0.7 s per ckernel update).

### Wave 2 (partial): speed-2 part 2, valid annealed blocks (commit after 2a88434; 27 colour-Gibbs tests pass)
- `castlegen/kernels/anneal.py`: `anneal_block_valid`. Annealed L3 / L4 subtree blocks started from VALID states (probe lifts or q's own draws; tau 8 -> 1, lambda fixed at 20, hard final phase, 32 steps, 0.10 s per 64 blocks) are 98-100 % valid; 16-17 % (L3) and 25-34 % (L4) of all-dirt q-draw blocks become partially filled. Random starts stay 0-2 % valid; a lambda ramp destroys valid starts.
- AIS weights from these non-equilibrium starts are heuristic (ESS 1.5-2 of 64 from q draws; 16-18 from probes). Passed to wave2-explore as the exploration move for own draws.
- dimers-kernel (Sonnet) launched: generic k x k fine-block Gibbs for T = 0 and the kernel loop on dimers, to see whether local block moves can rebalance the three orientations.

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

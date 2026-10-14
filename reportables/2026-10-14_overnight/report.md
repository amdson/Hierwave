# Overnight session, 2026-10-14: the kernel route, pushed further

Started 08:01 UTC. Budget: six hours, two Opus + up to four Sonnet agents. Constitution frozen (amendment 2026-10-14 in force); change requests go to `notes/constitution_change_requests.md`.

Goals: (1) ground's coarse levels trained by the kernel route; (2) own-draw line_match off 0.50 and the v2 bench; (3) speed-ups for the kernels, ML update and pass; (4) breadth once ground moves.

## Wave log (newest first)

(waves are appended here as they finish)

### Wave 4: wave4-pop, violation pressure on the own-draws-only arm (not accepted yet)
- Hard final colour-Gibbs sweeps (lambda = inf) keep the chain population 100 % valid and own-draw line_match at 0.636, but own-draw violations stay at 2.8e-4..1.1e-3 per level: plain ML still leaks. Adding cfast's reverse-KL on levels 0-3 (lr 1e-3) cuts violations 2-5x at a cost of about 0.025 line_match.
- Bench `ckpop_hard_rkl` (own draws only, inside the frozen rules), 2 seeds, 300 s: NOT accepted. Seed 0 lost 171 s to compiling despite the warm-up and ends with 5.9e-3 at L4; seed 1 (52 s setup, 608 ML + 2433 RKL updates) fails only at L2 (7.9e-4) with F -0.091, line_match 0.62. Mean F -0.047 vs cfast 0.774. L3-4 probe starvation 21-33 % (rho <= 0.31 there): L4 has no violation pressure and the coarse levels are still partly starved.
- Wave 5 launched (wave5-pop, fresh agent): fix the compile budget (single batch shape, warm-up with the bench's exact shapes), add scaled L4 reverse-KL pressure, bench 2 seeds.

### Wave 3: wave3-explore, own draws leave the dirt state INSIDE the frozen rules (main result so far)
- `ckernel_pop` (castlegen/variants/ckernel_own.py): persistent chain populations (3 % refreshed from fresh own draws each iteration) plus 20 colour-Gibbs sweeps per iteration, kernels applied to own draws only, no probe contexts anywhere in training. One seed, 300 s: own-draw line_match 0.507 -> 0.645, dirt 0.851 -> 0.508, own-draw dirt per L4 row 0.00 / 0.12 / 0.91 / 1.00 against the fill's 0.00 / 0.12 / 0.88 / 1.00. Clamp gap closed at L3 and L4 (fill / own: -0.112 / -0.111 and -0.138 / -0.144), slightly better than the probe-context arm.
- Both ingredients matter: persistence alone 0.545, sweeps alone 0.574, together 0.645. The win is accumulation of the 1-2.5 % accepted dirt->partial moves over a persistent population, not any single big move.
- Not bench-ready: own-draw violations 9.4e-4 per cell (cfast 1.9e-5), only 3 % of draws valid; plain ML has no violation pressure. Wave 4 launched (wave4-pop): hard final sweeps so the ML targets are valid states, and/or the reverse-KL-on-L0-3 recipe, then a 2-seed bench.
- Certificate-aware moves: lambda-only tempered transitions (20 -> 5) accept 0.74-0.82 at L4 with 8-9 % dirt->partial per move; 20 -> 2 is rejected (violations survive the cooling). Freeing one column of an L4 block consistently changes 46 cells across 5 levels (16 / 16 / 8 / 4 / 2). The bottom-up certificate proposal was not built.

### Wave 3 (partial): wave2-bench2, closest to an accepted row so far (upper-bound arm, change request #1)
- Level-split objectives, all with lambda = inf in the coarse kernels (refined coarse states now have zero coarse violations). ML off at L3-4 collapses the line (dirt 0.95): the coarse line-following on this arm comes from ML on the probe contexts, consistent with the own-draw-only failure.
- `ckernel_rkl123` (ML on all levels, reverse-KL on L0-3): bench 2 seeds, warm compile (setup 254 -> 75 s), speed-1b kernels. Seed 1 within budget at every level: F -0.098, line_match 0.65, rho .38/.81/.58/.82/.60, starved 0/2.8/8.1/10.9/2.8 %. Seed 0 fails at L3 (7.3e-4). `ckernel_rkl32inf` (RKL on L4 too) clears L3 but pushes L4 starvation to 33-36 % and fails across-seed agreement.
- Remaining violations are 1-4 instances per level in 16 worlds, about seed-to-seed noise. Follow-up launched (wave3-bench): a weaker separate L4 reverse-KL step, then a 4-seed bench of the chosen arm to measure the pass rate and the evaluation's own noise.

### Wave 3 (partial): dimers-kernel part 2
- Adding cfast's reverse-KL steps to the dimers kernel loop (N = 32 per iteration, lr 1e-3) restores validity (3.4e-4 per cell, near budget) but the own-draw mix collapses back to cfast's two orientations (diagonal 0.000-0.011); refined states still carry 0.07-0.09 diagonal. ML only keeps three orientations (.36/.32/.32) at 5.6e-2 violations.
- Ground and dimers now show the same trade-off: ML gives diversity but leaks mass onto violating states; the full reverse-KL mix fixes violations and brings collapse back. Written up as constitution change request #2: ML plus a penalty-only violation term (the λ·E_q[violations] part of reverse KL without the energy and entropy terms). Not built (frozen).

### Wave 3 (partial): speed-1b, coarse kernels batched
- Block-only passes, K + 1 candidates scored in one batch, factored coarse Gibbs: ckernel iteration 7.1 s -> 0.50 s (the 7.1 was contended; a fair range is 4-14x). Flips 63 -> 2180 moves/s, L2 multiple-try 32 -> 3630, updates 1.1 -> 16.1 per s. Acceptance decisions identical to the old code on the same keys; log ratios within 5e-3.
- A 300 s ckernel run (upper-bound arm) now gets 2312 updates against 440, own-draw line_match 0.70 against 0.63, L4 fill -0.146 against -0.125. Remaining cost is the ML step and context generation inside ckernel.py.
- Speed summary for the night so far: fine loop 3.7-6x (speed-1), coarse cycle ~4-14x (speed-1b), colour Gibbs sweeps at 1.4e8 cells/s and annealed blocks at 670 blocks/s (speed-2).

### Wave 2: wave2-explore, own draws only (negative, with a diagnosis)
- Five arms (bottom-up curriculum + coarse flips K = 16; + annealed L4 blocks; tempered transitions tau <= 2 and <= 6; hot-end lambda 1), one seed, 300 s each, no probe contexts: own-draw line_match stays at 0.50 in all five; dirt 0.84-0.90. The clamp-gate gap closes in every arm (own contexts fit), but that is the fine levels fitting own contexts, not exploration.
- Diagnosis: rho is NOT what is stuck (rho4 in the top row already follows the fill, 0.02 vs 0.00). The stuck channel is the B certificate (occupied columns, 0.66 in the top row); the hard support and honour constraints make the tiles follow it, and freeing a column means lowering B consistently bottom-up through four levels. Flips redraw certificates high from q; tempered transitions at lambda 20 do not cross; lowering lambda at the hot end leaves violations that get rejected (median log acceptance -756). The ~5 % accepted dirt->partial moves are lost because contexts are redrawn fresh every iteration.
- Annealed L4 blocks from full-support q draws (exact AIS weights) were 1.5 % valid at 32 steps (speed-2's 98 % used valid own-block starts).
- Wave 3 launched (wave3-explore): persistent chain populations so accepted moves accumulate; certificate-aware bottom-up proposals with the certificate factors tempered inside the proposal only.
- Also this wave: notes/methodology.md extended with the kernel-route sections (status table; TBD markers for the runs still in flight).

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

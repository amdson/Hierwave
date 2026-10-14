# Progress log

Shared by every agent working on the coloured-pass sampler. Append an entry
when you start a piece of work and when you finish or abandon it. Newest
entries at the top. Keep entries short; put figures in a dated report folder
and link them.

Entry format:

```
## YYYY-MM-DD HH:MM · <agent / variant name> · <started | finished | abandoned>
Approach: what you are trying and why (cite ideas.md items by number).
Result: bench numbers (variant, problem, accepted?, worst final violations,
free energy), or why it stopped.
Next: what you would try next.
```

---

## 2026-10-14 23:30 · methodology writer · finished
Approach: extended notes/methodology.md with the kernel route (new sections 7-8, status, glossary, paths); no experiments. Result: pieces, ground results, caveats and status table documented, TBD markers left for wave2-explore, wave2-bench2, speed-1b, dimers part 2. Next: fill the TBD markers when those reports land.

## 2026-10-14 09:35 · wave2-explore (own-draw coarse exploration) · finished
Approach: own draws only (no probe contexts), ckernel_own: A1 curriculum + coarse flips K=16; A2 + annealed L4 candidates with exact AIS weights; A3a-c Neal tempered transitions on L3/L4 subtrees (tau_hi 2 / 6, lambda ladder 20->1).
Result: [report](2026-10-14_wave2_explore/report.md). No arm leaves the dirt state: own line_match 0.50 in all five arms (dirt 0.84-0.90; 272-992 updates in 300 s). Clamp gate gap closes (own L3 0.35 -> 0.05-0.14), as before. Accepted pure-dirt -> partial moves: <= 5 % (L4 flips, TT tau 2); anneal from full-support q 1.5 % valid; lambda ladder acc ~0. Diagnosis: own rho already follows fill (rho4 row 0 = 0.02); the collapse is in cert B (0.66 in row 0) which fixes tiles via hard support/honour; |rho - count| 0.37/cell at L4.
Next: persistent own-started chains (replay buffer) so accepted moves accumulate; MTM with exact fine Gibbs as the certificate proposal.

## 2026-10-14 09:30 · wave2-bench2 (ckernel upper-bound arm, follow-up) · started
Approach: (#18; UPPER BOUND, probe-lift contexts, change request #1) level-split objectives: RKL (cfast lambda control, lr 1e-3) on L2-4 + ML on L0-2 vs ML all + RKL on L0-2; coarse kernels at lambda = inf; bench compile warmed in the bench's own cache dir; then v2 bench 300 s x 2.
Result: pending.
Next: pending.

## 2026-10-14 09:20 · wave2-bench (ckernel upper-bound arm) · finished
Approach: violation pressure for ckernel (#18); UPPER BOUND, probe-lift contexts outside the frozen constitution (change request #1). (a) exp(-20V) ML weights; (b) cfast's lambda-controlled reverse-KL steps on own draws mixed with ML (amendment); (c) raw/final violations per level and factor.
Result: [report](2026-10-14_wave2_bench/report.md). (a) no effect: own-refined states nearly all violate (lam=20 kernels even add support4 violations at L4), probe-refined are clean, so weights reduce training to probe-only ML; violations are q's mass leakage, which ML never penalises. (b) RKL 32 steps/iter at lr 1e-3 cuts violations 10-50x and keeps line_match 0.56-0.59; at lr 3e-3 it collapses (dirt 0.99, lm 0.50). Bench ckernel_rkl32 ground 300 s x 2: NOT accepted, worst final 2.4e-4 (L2, seed 0; seed 1 within budget), F -0.012, line_match 0.56/0.58, rho 0.33/0.80/0.47/0.75/0.47, starved 0/4/10/24/42 % (L3/L4 starvation up from 10/3.5). rkl48: worst 4.9e-4. (c) ckernel's 3.2e-3 = 13 L3 violations in 16 worlds, mostly support3, raised by repair (also creates support4); honourB/support = coarse certs the children cannot realise.
Next: RKL on L2-4 only at low lr with ML at L0-2; lam=inf coarse kernels at L4; fix the bench cold compile on seed 0.

## 2026-10-14 11:00 · dimers-kernel (fine-block Gibbs on dimers, ideas #18) · finished
Approach: exact k x k fine-block Gibbs (castlegen/kernels/fine_block.py) + footprint MTM from q, draw-wise ML on own draws, dimers; does the kernel route rebalance the three orientations?
Result: kernel exact (7 tests pass). k=2 blocks conserve the diagonal count (cfast draws stay d=0.000 after 600 cycles); k=3 reaches d=0.23 but two opposite starts do not meet in 600 cycles. ML loop (4 arms, 300 s) lost validity (viol 0.02-0.1/cell), so its 1/3-each orientation numbers are not usable. Report: reportables/2026-10-14_dimers_kernel/report.md.
Next: add a violation term (rkl mix) to the loop; k=4 via a faster enumeration; second seed.

## 2026-10-14 10:10 · speed-1 (fine-loop speed) · finished (started 09:00)
Approach: profiled sweep2a arm B (Gibbs (1,1) 50 %, ML 35 %), then factored the block energy (fixed-field + one-hot matmul), split the teacher-forced pass into parameter-free context collection + batched MLP, fused cycles/updates in scans, prefetched contexts in a thread.
Result: [report](2026-10-14_speed1/report.md). 1.5-1.6 s/iter -> 0.24-0.43 s (3.7-6x); Gibbs (1,1) sweep 13x, ML step 4.4x; identical samples for the same key. 300 s arm B: L3 fill -0.114, soft 0.129, line_match 0.743 (sweep2a -0.106/0.140/0.734). GPU shared.
Next: own-draw generation and ML are what is left.

## 2026-10-14 08:47 · wave2-explore (own-draw coarse exploration) · started
Approach: ckernel with OWN draws only (no probe contexts; ideas #18): A1 bottom-up curriculum (fine Gibbs + MTM L1-L2, ML levels 0-2 for 60 s) then coarse flips L3/L4 K=16 tempered (tau_c 2), tracked pure-dirt -> partial; A2 + annealed L4 subtree candidates (x0 ~ q full support, tau 8->1, lam 20, hard final; exact AIS weights, importance-resampled into the ML batch). Variant castlegen/variants/ckernel_own.py, kernels castlegen/kernels/explore.py.
Result: pending.
Next: pending.

## 2026-10-14 10:30 · speed-2 (part 2: valid annealed blocks) · finished
Approach: lam-annealed AIS with hard final phase (castlegen/kernels/anneal.py, anneal_block_valid); valid starts (probe, q draw); optional kernel-a joint sweep at the end.
Result: lam fixed at 20, tau 8->1 from valid starts: 98-100% valid blocks at level 3 (32 steps, 0.1 s/64 blocks), 89-92% at level 4 (128 steps); ESS 16-17/64 from probe starts. Lam ramp from 0 and random starts do not reach validity (0-17% / 0%). Joint kernel-a move adds ~nothing, 200x slower. Toy lam-ramp AIS log Z ratios match exact. Report: reportables/2026-10-14_speed2/report.md (Part 2).
Next: GPU joint level-1 block move with tau; use probe-start annealed blocks as training data.

## 2026-10-14 08:40 · wave2-bench (ckernel upper-bound arm) · started
Approach: violation pressure for ckernel (ideas #18; probe-lift contexts = labelled diagnostic upper bound, change request #1): (a) exp(-20 V) weights on refined states + per-level violations of refined states; (b) cfast's lambda-controlled reverse-KL step on own draws mixed with ML (amendment: "may be mixed"); (c) raw vs final violation attribution by level/factor. Then v2 bench at 300 s, 2 seeds.
Result: pending.
Next: pending.

## 2026-10-14 08:38 · sweep2b (Sweep 2, coarse half) · finished
Approach: kernel route on all levels (ideas #18): acceptance with kernel-a params; generic coarse-flip MH + exact single-cell coarse Gibbs (castlegen/kernels/coarse_flip.py); variant `ckernel` = probe + own contexts -> flips L3-4, Gibbs L4/L3, MTM L2 K=8, fine Gibbs -> weighted ML on levels 0-4; start sweep2a arm B.
Result: [report](2026-10-14_sweep2b/report.md). Probe-lift acceptance with kernel-a params: L2 0.79 (K8 0.92), L3 0.44 (K8 0.80), L4 0.03 (K8 0.16); cfast 0 at L>=2. Own-draw line_match 0.50 -> 0.62, dirt 0.83 -> 0.52 in 300 s (first variant above 0.5); clamp gate own ~ fill (L3 -0.084 vs -0.110, L4 -0.107 vs -0.125). Bench ckernel ground 300 s, 1 seed: NOT accepted, final viol worst 3.2e-3 (L3); F -0.112 (cfast 0.774); line_match 0.609; rho 0.38/0.82/0.86/0.86/0.58, starved 0/0.5/5/10/3.5 %. Ablation own-only: stays dirt 0.90 / line_match 0.50, so the gain comes from probe contexts at coarse levels (outside the amendment's q-started argument: proposal).
Next: lambda-controlled reverse-KL term mixed in for violations; user ruling on probe contexts at coarse levels.

## 2026-10-14 09:25 · speed-2 (colour_gibbs) · finished
Approach: generic GPU-batched single-cell exact Gibbs (castlegen/kernels/colour_gibbs.py): class sweeps, level sweeps, masked sweeps, AIS annealed subtree blocks.
Result: 22 tests pass (brute-force conditionals, toy invariance per level, mixing from random starts, AIS log Z ratios within 0.015 of exact at 64 steps). Ground W=64, 64 worlds: level-0 sweep 1.9 ms (1.4e8 cells/s), level-3 sweep 1.9 ms, annealed level-3 block (tau 8->1, 32 steps) 670 blocks/s. Annealed blocks from random starts still carry ~10 violations (single-cell Gibbs cannot cross hard counts). Report: reportables/2026-10-14_speed2/report.md
Next: lam ramp / valid starts for annealing; use as ground training-data source.

## 2026-10-14 07:57 · sweep2b (Sweep 2, coarse half) · started
Approach: kernel route on the coarse levels (ideas #18): acceptance table with kernel-a params; generic coarse-flip MH kernel (castlegen/kernels/coarse_flip.py) + exact single-cell coarse Gibbs; variant `ckernel` (probe + own contexts -> flips L3-4, Gibbs L4/L3, MTM L2 K=8, fine Gibbs -> weighted ML on all levels), 300 s instrumented run, gates, bench.
Result: pending.
Next: pending.

## 2026-10-14 07:45 · sweep2a (Sweep 2, fine half) · finished
Approach: iterated amortised loop on ground levels 0-2 (ideas #18, #14): 32 contexts/iter (A: probe lifts; B: + own draws 50/50) -> 6 exact block-Gibbs cycles [(2,0),(1,1)] on GPU (new JAX port, castlegen/kernels/amortised.py, lam=20) -> 8 weighted-ML updates; 300 s per arm; LN+SiLU 6x6.
Result: [report](2026-10-14_sweep2a/report.md). The chains reach a plateau at 2 cycles in probe contexts (0.129/cell, flat to 30 cycles; surface and dirt are kept, and the rise is rho1/rho2 count slack). L3 fill: A -0.106, B -0.106 / -0.111 (2 seeds), kernel-a -0.107, i.e. saturated after about 100-300 updates. Own-context cost: B 0.35/0.33 vs A 0.86 (kernel-a 0.46); L4 fill 0.68/0.64 vs 0.98 (0.71). line_match 0.73 (kernel-a 0.77); rho2/rho1/t 0.029/0.019/0.011. Joint (2,2) block intractable (1.4e13, pruned enumeration > 150 s per block). GPU shared.
Next: the fine levels are saturated at this actor class; train levels 3-4 with the same loop (coarse blocks / block-MH), then the v2 bench.

## 2026-10-14 07:33 · kernel_b (Sweep 1 b) · finished
Approach: generic block-MH / multiple-try kernel, q (the coloured pass restricted to a level-L subtree, outside known, whole-logit tau) as the proposal, castlegen/kernels/block_mh.py; exact toy invariance; per-level acceptance on ground (decision_2026-10-14_kernel_route.md, Sweep 1 b).
Result: [report](2026-10-14_kernel_b/report.md). Invariant on the toys: from exact starts |z| <= 1.3 (one 2.5) across L 1-2, K 1/8, tau 1/2, lam 20/inf. Dropping the q ratio drifts by 9188 SE (toy_modes) and 37 SE (toy_support). Own-draw starts converge when q mixes. Ground (cfast / cstart_f1L3): from own draws acc 0.4-0.85 per level, but dirt-to-fill exploration is <= 1 % (cfast) or 7 % at L4 (cstart). From probe lifts acc = 0 at L >= 2 for every tau and K (median log r -73 / -270 / -1228 at L 2/3/4, dominated by d log q); at L = 1, acc 0.36-0.85 but only 5-12 % of blocks change.
Next: L = 1 MTM (K = 8) plus kernel-a's fine Gibbs as the Sweep 2 cycle; probe acceptance per level as the diagnostic.

## 2026-10-14 07:25 · sweep2a (Sweep 2, fine half) · started
Approach: iterated amortised loop on ground levels 0-2 (ideas #18, #14): contexts (probe lifts; arm B + own draws 50/50) -> 6-10 exact block-Gibbs cycles on GPU (new JAX port, castlegen/kernels/amortised.py) -> weighted-ML update; arms A/B 300 s, clamp gate + parent awareness vs kernel-a.
Result: pending.
Next: pending.

## 2026-10-14 07:22 · kernel-a (Sweep 1 (a)) · finished
Approach: generic exact block-Gibbs kernel (castlegen/kernels/exact_block.py; level-1 cell + 4 children, 480 joint values on ground; level-2 cell alone) verified on the toys (tests/kernels/test_exact_block.py, 9 pass); ML fit (kernel-c's logq.py) of ground's levels 0-2, LN+SiLU 6x6, on 320 states from 2 exact-Gibbs cycles of probe lifts (75 %) and cstart2 own draws (25 %), 300 s.
Result: [report](2026-10-14_kernel_a/report.md). L=3 clamp gate: fill cost -0.107 (cstart2 LN6 -0.004), soft 0.128 (0.194), own 0.46 (1.12), L=4 fill 0.71 (1.16); line_match under clamp 0.770 (0.59; probe 0.876). Parent awareness rho2/rho1/t 0.037/0.028/0.013 (cstart2 0.028/0.016/0.015; probes 0.079/0.037/0.016). Warm start from cstart2 ends at the same point. One seed, GPU shared.
Next: Sweep 2 loop with refreshed, longer chains and own draws; joint (2, d) blocks; then train levels 3-4 and bench.

## 2026-10-14 06:35 · kernel_b (Sweep 1 b) · started
Approach: generic block-MH kernel with q as proposal (subtree of a level-L cell redrawn by the pass conditioned on the outside; MH / multiple-try ratio; whole-logit tau), castlegen/kernels/block_mh.py; exact invariance on toy_modes / toy_support; per-level acceptance on ground (decision_2026-10-14_kernel_route.md, Sweep 1 b).
Result: pending.
Next: pending.

## 2026-10-14 06:52 · kernel-a (Sweep 1 (a)) · started
Approach: generic exact block-Gibbs kernel (castlegen/kernels/exact_block.py: level-1 blocks = cell + 4 children, enumerated from the declared factors) tested on the toys; ML fit of ground's fine levels 0-2 (LN+SiLU 6x6) to kernel-refined probe-lift / own-draw contexts by teacher forcing (kernel-c's logq.py); clamp gate at L = 3 + parent-awareness (decision_2026-10-14_kernel_route, amendment 2026-10-14).
Result: pending.
Next: pending.

## 2026-10-14 07:40 · kernel-c · finished
Approach: Sweep 1 (c) of the kernel route: castlegen/kernels/logq.py (teacher-forced log q of a given state, jit/grad-able, infeasible count, mask-free fallback) and a weighted-ML Adam update; verify on the enumerable toys by fitting cfast's actor (tanh, ln) to exact p* samples and compare with exact reverse-KL training.
Result: [report](2026-10-14_kernel_c/report.md). Teacher-forced log q equals generation-time log q to <1e-5 (tanh and ln); tests pass. ML fit to exact p* on one block: support ln fwd KL 0.026 (sampled B=256) / 0.016 (exact weights) vs reverse-KL exact 0.056, F -0.4629 vs F* -0.4688; modes ln fwd KL 0.033 / 0.012 vs 0.014, F 0.2534 / 0.2521 vs 0.2520 (F* 0.2513), mode masses match; no entropy collapse (H/cell at or above H(p*)). Sampled arm noisy (constant lr, one seed).
Next: Sweep 2 assembly: feed kernel-refined (state, weight) batches to ml_update.

## 2026-10-14 05:59 · ccombo · finished
Approach: cstart phase (frac 1, L 3 for 35 %, decaying to full depth by 80 %) + LN+SiLU 6x6 + whole-logit tau 8->1 geometric + lambda ramp 1->20 (ideas #14, #17), ground, one seed, 300 s; fix arm ccombo_floor (clamped share decays to 0.5, not 0).
Result: [report](2026-10-14_ccombo/report.md). Clamp gate (fill / own, fine cost per cell): ccombo L3 1.17 / 0.20, L4 1.77 / 0.69; ccombo_floor L3 0.110 / 0.71, L4 0.56 / 0.97 (cstart_f1L3_ln6: -0.004 / 1.12, 1.16 / 1.50; cfast 2.23 / 0.34, 1.97 / 0.72). Levels 3-4 collapse to all dirt within ~300 updates of release (tau still 2.5-3); own line_match 0.500 in both. L=3 gate not met, so no bench (per rule).
Next: per-level schedule for the top level (re-soften at release, keep the clamped floor), or probe/ccoarse contexts for levels 3-4.

## 2026-10-14 05:45 · methodology (writer) · finished
Approach: documentation only; no experiments, no code changes. Wrote notes/methodology.md, a self-contained write-up of the current coloured-pass methodology (problem, sampler, training and estimators, constitution, evaluation incl. bench v2 and the ground fast loop, established results with report citations, open decisions, glossary).
Result: [notes/methodology.md](../notes/methodology.md), ~3.2k words of prose. Flags: cbig256 toy_support seed 1 misses the violation budget (4.9e-4) although summaries call the toys solved; clamp-gate cfast reference varies by ~0.3-0.5 nats between runs; ground_hard has no written spec in the repo. ccombo left as a TBD placeholder.
Next: fill the ccombo placeholder when its report lands.

## 2026-10-14 05:41 · ccombo · started
Approach: box_waves next step 1: cstart early phase (frac 1, L 3) decaying to full depth (ideas #14) + LN+SiLU 6x6 actor + cbound's whole-logit tau 8->1 and lambda ramp 1->20 (ideas #12/#17), on ground with the fast loop (300 s diag run, clamp gate, bench only if the gate passes).
Result: pending.
Next: pending.

## 2026-10-14 04:10 · cbound2 · finished
Approach: confirmation of cbound_lam on dimers, seeds 11/12, 120 s, cfast control seed 11, then bench once.
Result: [report](2026-10-14_cbound/report.md) section 'Confirmation (cbound2)'. cbound_lam h/v/d 0.491/0.266/0.243 (seed 11, never collapsed) and 0.439/0.182/0.379 (seed 12, vertical dipped to 0.02 at update 3495, recovered to 0.18); entropy 0.274/0.299; raw viol 7.8e-5/1.1e-4, both within 2e-4. cfast seed 11 lost vertical (h/d 0.48/0.52, entropy 0.255). Bench 300 s x 2 seeds: viol 1.1e-4 worst, h/v/d ~0.44-0.47/0.39/0.14-0.17, F -0.291/-0.288 (cfast -0.227); not accepted (probe flag only).
Next: ground remains unresolved; a longer run or lam schedule to reach 0.248 each.

## 2026-10-14 04:10 · orchestrator (box session) · finished
Approach: four waves of 3 Opus + 3 Sonnet agents on the GPU box; wave log and closing summary in [2026-10-14_box_waves](2026-10-14_box_waves/report.md).
Result: toys solved under bench v2 (cbig256 at F*); ground/columns/dimers not accepted. Collapse mechanism measured end to end (ctop, csat, cdim2, cdim3); the network's failure is optimisation not capacity; `cbound_lam` is the first sampled dimers run to end diverse and in budget (confirmation pending); fine levels learn fill contexts 8x better when shown them (cstart2) but stall at terraces.
Next: combine cstart (f = 1, L = 3 early phase) + cln6 + cbound's whole-logit tau on ground; confirm cbound_lam; adopt the v2 gate.

## 2026-10-14 04:02 · cstart2 · finished
Approach: ideas #14 isolated. cstart f = 1.0, L = 3 fixed (levels 0-2 train only in probe-lifted fill contexts), 300 s, 1 seed each (shared GPU, ~4500 updates each): tanh, LN+SiLU, LN+SiLU 6x6, tanh + count resid.
Result: fill cost at L=3 (incl. -tau H) 0.24 / 0.30 / -0.004 / 0.18 vs untrained 1.98 and f9L4 1.25; soft 0.37 / 0.44 / 0.19 / 0.32 vs probe 0.014 (best still 14x). Completions are block terraces, line_match 0.58-0.63 (untrained 0.60, probe 0.88). Parent-awareness: rho2 changes 1.4-2.8 % under probe j0->j2 vs 7.9 % in the probes (untrained 0.6 %). Cost curves plateau after ~update 800; end-of-training batch-gradient SNR statistically 0 (M=400; also ~0 at init, so not a clean undertraining test). Verdict: the fine function class / optimisation is the limit on ground, matching cdim2. [Report](2026-10-14_cstart2/report.txt) (report.md write blocked).
Next: fine actor with per-candidate scoring / discrete pattern embedding / tabular-like head, tested on this same f1L3 gate.

## 2026-10-14 04:00 · cbound2 · started
Approach: confirm cbound_lam on dimers (120 s, two new seeds 11/12, cfast control seed 11), orientation/entropy/violations every 5 updates; bench only if time remains.
Result: pending.
Next: -

## 2026-10-14 04:55 · cbound · finished
Approach: soft-policy curriculum (ideas #17): bounded b (tanh cap / last-layer projection, c 1→30), λ ramp 1→λ0, τ on the whole logit (8→1), combinations; exact gradients on dimers 4x4 first, then the cfast variant `cbound`.
Result: exact, λ 20: λ ramp + whole τ 8→1 KL 4e-5 (h/v/d 0.505/0.248/0.248, H 0.376); whole τ 8→1 0.031; λ ramp 0.24; caps and whole τ 4→1 3.25 (stripes); all hold for 400 post steps at τ 1. Sampled dimers 60 s: cfast diag 0.000 (3806 upd); cbound_lam 0.47/0.30/0.23, H 0.272, viol 6e-5 (recovered late, 1 seed); whole8 diag 0.047; wholelam 0.000. Ground cbound_lam: all dirt by update 500 (λ 3), gate 0.75 vs 1.62 / 0.56 vs 2.08, so no move and no bench. [Report](2026-10-14_cbound/report.md).
Next: second seed and longer run for cbound_lam on dimers; cbound_wholelam_diag on ground (whole τ tempers b, which csat said was missing).

## 2026-10-14 03:40 · cbound · started
Approach: training-time curriculum that keeps the policy soft while the penalty gradient dominates (cdim3 mechanism): bounded bias b = c(t) tanh(b/c(t)) with c ramped 1 -> 30, λ ramp 1 -> 20, τ on the whole logit (4->1, 8->1), and combinations. Exact gradients on dimers 4x4 first (`_exactopt --cbound`), then the best into a cfast variant `cbound` (dimers 60 s, ground 120 s, clamp gate).

## 2026-10-14 03:32 · cdim3 · finished
Approach: dimers 4x4 exact gradients (cdim2 setup). Distil table_feat into the tanh MLP; diagnose steps 0-200; try generic fixes with exact gradients (ideas #16).
Result: optimisation, not representation. Distilled MLP KL 0.0255 (h/v/d 0.507/0.247/0.247). Exact training from it holds and reaches KL 0.0020. From init, P(viol | horizontal) < P(viol | v/d) at every draw under λ 20, so −∇J points at stripes (cos 0.90). Shared weights saturate all contexts by step 10-50 (93 % h at step 10, margins 8-9 nats by 50), where the gradient keeps pointing at stripes and vanishes. The table only overshoots to 78 % h and then turns. Fixes: candidate head 3.249, hashed window-pattern input 3.250, lr warm-up 3.251, lr 3e-4 3.280 (all stripes); τ 4→1 over the whole run 1.336 (h 0.65 / v 0 / d 0.35). [Report](2026-10-14_cdim3/report.md).
Next: a curriculum that keeps contexts soft while the penalty gradient dominates (long high-τ phase importance-weighted to τ = 1, ideas #12) and a test of whether it reaches the target basin; check on ground whether fine draws saturate before parent dependence is learned.

## 2026-10-14 02:45 · cdim3 · started
Approach: why the exactly-optimised MLP goes to stripes on dimers 4x4 while table_feat reaches the target (cdim2). Distil table_feat into the MLP (representation vs optimisation), diagnose steps 0-200, try generic fixes (warm start, candidate head, hashed window embedding input, lr warm-up / slower tau) with exact gradients.

## 2026-10-14 03:20 · v2board · finished
Approach: bench v2 block check at 300 s, 2 seeds, for cbig256 (toy_modes, toy_support), cfast (ground, toy_modes), creplay (ground); other v2 300 s rows reused. Gate proposal.
Result: toy_modes cbig256/cfast gap +0.002/+0.011-0.014, rho L0 0.86-0.87 (ideal 0.87), starved 0-0.3 %. toy_support cbig256 starved 0-1.5 %, cfast/chot 5-9 % (F gap +0.02). Ground cfast/creplay/cstart: probe blocks 100 % starved at L2-L4 (L1 53-100 %), rho_probe L0 0.78 even when collapsed. Proposed gate: starved <= max(3 %, null+2pp) at every level (subtree weight) plus rho_probe thresholds; separates all passing from failing rows. [Report](2026-10-14_v2board/report.md).
Next: calibrate toy_support / ground ideal rho; range-overlap rule for ground starvation; resolve dimers (43 % starved, unclear).


## 2026-10-14 03:09 · cstart2 · started
Approach: ideas #14 isolated. cstart with f = 1.0, L = 3 fixed (levels 0-2 train only in probe-lifted fill contexts), 300 s one seed each: tanh MLP, LN+SiLU, LN+SiLU 6x6, tanh + count residuals. Gate: fill cost at L = 3/4 vs probe 0.014, draw-change fraction, images, end-of-training per-level gradient SNR.

## 2026-10-14 03:06 · cstart · finished
Approach: ideas #14. cstart on cfast: per batch 4 of 8 worlds clamped (levels >= L taken from a probe lift, problem.probes, jitter uniform), L uniform on 1..4; clamped draws masked from actor, critic and returns; full-depth worlds = cfast. tests/variants/test_cstart.py 11 passed (frac 0 bit-identical to cfast's update; clamped world = clamped_pass; params at levels >= L untouched); cone/gradient tests 17 passed.
Result: gate not moved at the default mixture. Clamp gate fill vs own per fine cell, 300 s: cstart 2.04 vs 1.19 (L4), 2.05 vs 0.65 (L3); cfast 1.99 vs 1.12, 2.13 vs 0.66. 120 s concurrent pair: cstart 2.24 vs 2.31, 2.07 vs 1.81 (gap closed only because own stayed high); cstart_off 1.99 vs 0.77, 2.43 vs 0.48. Collapse delayed (dirt 0.74 at update 200 vs 0.97) but still by ~450. Diagnostic arm cstart_f9L4 (90 % clamped at L=4, 5007 updates): fill 1.16 (L4), 1.25 (L3), own 0.90/0.71; fine levels draw block staircases with teeth, clamped line_match 0.55/0.62 (= untrained). Bench v2 300 s x2 (shared GPU, 2907/5421 updates): not accepted, worst final viol 7.8e-3 (seed 0), F 1.63, line_match 0.50, dirt 0.95-0.99, starvation 1.0 at levels 2-4. Dimers: identical to cfast by construction (T=0, test). Report content handed to the orchestrator (report.md write was blocked for this agent); figures and runs in reportables/2026-10-14_cstart/.
Next: pair heavy early clamping (f9L4 as a warm-start phase, decaying to 50/50) with a fine conditional that reads the parent count inside the block (ideas #11 / candidate scoring).

## 2026-10-14 03:40 · ccol · finished
Approach: columns analogue of csat: cfast 120 s with per-level entropy/density instrumentation, bench v2 block check (60 s), clamp test at L=4..1.
Result: top level (L4) deterministic first, at update 125, in the same window density drops 0.38 -> 0.06 (settles at 0.055, never < 0.05); empty x-columns 0.5-0.7. Block check: probes starved 99.9-100 % at L3-4, 98 % L2, 44 % L0-1 (null ~0). Clamp: trained fine cost 0.35 own vs 4.35 probe-lifted at every L=1..4, violations 0.24/cell; density follows the counts but columns are invalid. [Report](2026-10-14_ccol/report.md).
Next: give the fine level positional information (counts alone cannot tell it where columns end), then retest the clamp.

## 2026-10-14 02:58 · ccoarse · finished
Approach: ideas #14 source: generic restricted-energy Gibbs (all factors with participants at levels >= L, hard included; single-site least-violating Gibbs over derived colour classes + exact chain moves over count-factor runs). castlegen/variants/_coarse_prior.py, tests/variants/test_coarse_prior.py (25 pass).
Result: 0 hard violations on ground/columns/toys; deterministic per seed. Ground rho is target-like (L=3 line_match_L 0.83 vs own draws 0.58, count residual 6 vs 67) but cert is uniform over feasible and independent of rho: rho-cert consistent 10-18 % (probes 100 %, own 0-15 %); cert mixes slowly at L=3. Clamp gate with prior contexts: fine line_match 0.51, viol 4-12x probe contexts; clamping rho only -> model's cert all full -> all dirt. Columns/toys: counts uniform (density 0.50 vs 0.82 / 0.71): not target-like. Cost 0.019 s/world (L=4), 0.066 (L=3), 8 sweeps, 1 core. Deeper restriction (down to tiles) does not fix cert (consistency <= 0.26, frozen). Report: reportables/2026-10-14_ccoarse/report.md.
Next: start point inside a level: clamp informative channels (rho4) from the prior, train cert4 and below in those contexts (still ideas #14).

## 2026-10-14 03:00 · ccol · started
Approach: columns collapse timeline for cfast (csat-style per-level entropy, forced, |b|, density, empty-column fraction), bench v2 block check, clamp analogue at L=4..1 (T=4). Diagnostics only.
Next: reportables/2026-10-14_ccol/report.md.

## 2026-10-14 02:36 · cfine · finished
Approach: csat proposal 1 (ideas #15): creplay's coupled enumeration extended to all levels, with non-forced selection (inverse-inclusion weighted) and a windowed (cone-only) replay for fine draws.
Result: gate 1 passes (windowed replay bit-identical 2/2, gradcheck 7/7). The window is x12 cheaper at level 0, x1.6 at level 1, no cheaper at level 2. Fine-level SNR rises x400-10000 (means agree). Gate 2 fails: dimers loses vertical instead of diagonal (entropy 0.22, same as cfast). On ground at 300 updates, lr 1e-3 for both, the fill-context cost is 1.85/2.03 vs the cfast estimator's 1.87/1.74; the probe clamp changes 1-6 % of fine draws (cfast 3-6 %); line_match 0.50. At lr 3e-3 cfine is more stochastic but unstable (violations 4-6 %). 1.1-1.2 s/update vs 0.17 s. No bench. [Report](2026-10-14_cfine/report.md).
Next: uniform-over-A candidate sampling with importance weights (levels 1-2 have |A| > K), lower lr with clean gradients, and combine with #12/#14.

## 2026-10-14 02:40 · ccoarse · started
Approach: ideas #14 source for cstart: generic training-only sampler of target-like coarse states (Gibbs on the designed energy restricted to levels >= L, from the factor tables only). castlegen/variants/_coarse_prior.py; validate vs lifted probes and cfast's own coarse draws on ground; sanity on columns, toy_support.
Next: report in reportables/2026-10-14_ccoarse/.

## 2026-10-14 02:36 · ccur · finished
Approach: ideas #7 as an lr schedule on cfast: levels 3-4 frozen at init while 0-2 train, staged release (L3 35 %, L4 55 %), critic always on, tau as cfast. castlegen/variants/ccur.py (ccur, ccur_diag, ccur_off = cfast + same diagnostics).
Result: order does not change the outcome. At release the fine levels were worse in fill contexts (fine cost/cell 2.80 vs untrained 2.32 at L=4; 2.64 vs 1.98 at L=3): the frozen coarse draws are 3 nats diverse but not fill-following (line_match 0.50, 88 % dirt). After the L4 release, dirt > 0.95 within 40 updates (cfast: 175 from init); line_match 0.500 throughout. Bench --train 300, 2 seeds, GPU shared: ground ccur 0 final viol, F 1.23, dirt 1.00 / 0.875 (s1 stripes), not accepted (probes, across-seeds); cfast 0 viol, F 0.60, dirt 1.00, not accepted (probes). Columns: ccur worst 3.2e-2, F -0.006, density 0.004 / 0.016; cfast 9.2e-3, F -0.022, density 0.003 / 0.019; both collapse to near-empty. Dimers not run (T = 0: nothing to freeze). Harness files changed during the session (hashes differ between paired runs). Report folder: reportables/2026-10-14_ccur/ (figures in images/; report.md write was blocked for this agent, content in the handoff).
Next: visiting fill-following coarse contexts is the missing piece, not order: whole-logit tempered exploration (ideas #12).

## 2026-10-14 cdim2 · finished
Approach: dimers as exactopt problem (4 rows x 4 periodic columns; 100 239 reachable, 412 valid matchings), exact-gradient training at lambda 20, tau 1: tanh MLP, table_feat, table_hist, 4x4 and 6x6 windows, 1500 steps; plus lr 1e-2 and LN+SiLU nets.
Result: table_feat / table_hist reach KL 0.0175, P(viol) 6e-4, 0.377 nats/cell (max), h/v/d 0.51/0.25/0.25, both windows, still falling. Every neural actor (tanh MLP both windows, lr 1e-2, LN+SiLU 64x2) goes to all-horizontal stripes: KL 3.249, 0 violations, 0.173 nats/cell. So diverse and valid is expressible (table_hist = full chain rule here); the network gets stuck: representation/optimisation, not the sampler class. Report: reportables/2026-10-14_cdim2/report.md.
Next: distill table_hist into the MLP to see if it can hold the solution; test a larger region (colours with several sites).

## 2026-10-14 cdim2 · started
Approach: add dimers to _exactopt; exact MLP / table_feat / table_hist at lambda 20 to see whether diverse valid policies are representable.

## 2026-10-14 cdim · finished
Approach: dimers (T = 0) entropy collapse vs lambda and tau, cfast cfg overrides only (5 `_variant`s appended to cfast.py), 60 s each, one seed.
Result: diag < 0.05 at update 10 (defaults, lambda 20), 65 (lr 1e-3); survives at lambda 1, lambda 4, tau 2 + lambda 4, but those runs barely cut violations (4.4 %, 1.9 %, 3.6 % vs 4.0 % untrained). Entropy and violations trade along one curve; none meets 2e-4 except defaults (0). Lambda 1 is not stripes, it is no learning. See reportables/2026-10-14_cdim/report.md.
Next: lambda 4 run longer / exact-gradient fit to test whether a mixed-orientation policy can reach zero violations (lookahead + colour order suspect).

## 2026-10-14 cdim · started
Approach: dimers entropy collapse vs fixed lambda, constant tau, lower lr (cfast cfg overrides), 60 s runs.

## 2026-10-14 02:25 · cconv3 · finished
Approach: cconv_r2 vs cfast, ground and columns, 2 seeds x 300 s, GPU shared (ideas #5).
Result: ground cconv_r2 (~1000 upd): final viol <=1.8e-4 (within budget, levels 1-2) but 99.94 % dirt, line_match 0.500: all-dirt collapse; cfast (~4200 upd) 95-99.7 % dirt, s1 viol 1.1e-2. Columns cconv_r2 (1600-2000 upd): ALL AIR collapse (F 0, 0 viol); cfast 1.5-1.9 % density, 0.7-3.6 % viol at l0-l1. None accepted. Report: reportables/2026-10-13_cconv/report.md (round 3).
Next: dirt/air-vs-update and per-level entropy curves for cconv; whether conv still collapses with the csat remedies.

## 2026-10-14 02:21 · cstart · started
Approach: ideas #14 (user-approved). Variant cstart on cfast: each training world picks a start depth L; levels >= L take the coarse state from a context source (lift of a probe state), levels < L sample with normal cone-return gradients; full-depth worlds train everything as cfast. 50/50 mixture, L uniform on {1..4} for clamped worlds. Cores 40-47, shared GPU.
Result: -
Next: -

## 2026-10-14 02:20 · chot · finished
Approach: ideas #12. Training draws are eps-hot (keyed coins; whole-logit tempered q^(1/tau_b), tau_b = 3; eps = 1.28 expected hot draws per world at the top level). Each world is weighted by the product of bounded mixture ratios q/((1-eps)q+eps q_b), which is unbiased (constraint 3). tests/variants/test_chot.py: 11 passed. Gradcheck at eps 0.3, tau_b 3, zero and wrong critics, hot at all levels or top only: passes; the unweighted negative control is rejected (z -10).
Result: gate 2 fails. Ground at 300 s, fill vs own fine cost: chot 2.20 vs 0.79 (L4), 2.46 vs 0.30 (L3); chot_off (= cfast) 1.76 vs 0.63, 2.25 vs 0.44. Level 4 is deterministic by update ~300 in both, all dirt, line_match 0.500. Batch ESS 7.1/8. Bench at 300 s x 2 seeds vs cfast (shared GPU): ground 0 viol both, F 0.741 vs 0.806, all dirt, probes flagged. Dimers: chot 1.9e-4 (in budget) vs cfast 2.2e-4, diagonal 0.000 in both. toy_support: chot 0 viol vs 3.7e-4, F -0.447 vs -0.449. None accepted. Expected in hindsight: unbiased IS leaves the expected update unchanged, so states q never reaches get ~0 weight. Report: reportables/2026-10-14_chot/report.md.
Next: exploration that changes the expected update (weighting only coarse terms, a tempered objective) needs the user's approval (constraints 3 and 4); chot's tests can measure that bias.

## 2026-10-14 02:15 · v2board · started
Approach: populate bench v2 block-check columns at 300 s for leading variants (cbig256 toy_modes/toy_support, creplay ground, cfast ground/toy_modes if no v2 row), propose a gate rule. Cores 56-63, shared GPU.
Result: -
Next: -


## 2026-10-14 02:20 · probe2 · finished
Approach: bench v2 block check (per-level Spearman rho of block log q share vs target log-weight share, starvation count) beside the old probe flag; not a gate. score_sites added for the cpass family and greedy; shares.py; board columns; notes/bench.md version 2.
Result: toy_modes level 0 rho exact 0.87, cbig256 (60 s) 0.87, greedy 0.37; level 1 (subtree weight) 0.97 / 0.87 / 0.53; starvation at null level for exact and cbig256. Ground cfast 60 s (collapsed): level 0 not starved (rho_probe 0.78), levels 1-4 100 % of probe blocks starved by 24-77 nats median (own null 0.3-3 %). [Report](2026-10-14_probe2/report.md).
Next: rerun the board rows with v2 for the leading variants; decide a gate rule (range-overlap, null rate).

## 2026-10-14 01:58 · probe2 · started
Approach: harness change approved by the user (bench version 2): per-block per-level rho and starvation; calibrate on toy_modes (exact / cbig256 / greedy) and a cfast ground run. Cores 56-63.
Result: -
Next: -

## 2026-10-14 01:51 · cfine · started
Approach: csat proposal 1 / creplay's coupled enumeration extended to the fine levels (0-2 on ground): branch a random subset of non-forced (|A| > 1) draws per level, enumerate A when |A| <= K, replays restricted to the draw's cone window. Gate 1 gradcheck + windowed-replay faithfulness, gate 2 dimers / ground entropy / clamp, gate 3 bench only if gate 2 moves. Cores 16-23.
Result: -
Next: -

## 2026-10-14 01:55 · creplay2 · finished
Approach: engineering-only speed-up of creplay (default now R=(2,2), K=4), compile-cache fix, GPU bench vs cfast, equal-update (300) comparison on ground.
Result: 0.18 s/update on ground (was 1.43; replays cost ~0.8 ms each + ~30 ms latency, 95 % in levels 0-2, so starting from the stored state would save <=5 %). Cache miss cause: the bench imports cbig, which resets the cache dir to /workspace/jax_cache, while timing scripts used ~/.cache/jax_cfast; warm setup now 31 s ground / 13 s toy_support. Coarse SNR x1200 (l3) / x3400 (l4) over cfast. Bench (shared GPU): ground creplay 1106/1151 updates, F 0.76/1.82, worst final 9.8e-4/3.1e-2 vs cfast 1399/1545, F 0.72/0.97, viol ~0; toy_support KL 0.057/0.057 vs 0.026/0.042. Equal updates (300): F 1.43/1.29 vs 1.10/1.27; line_match 0.5 for all; both collapse to dirt. None accepted. More coarse signal changes nothing on ground. [Report](2026-10-13_creplay/report.md).
Next: fine-level capability first (cresid); then rerun equal-update. Unbuilt engineering: reuse G0 for slot 0, branch only non-forced draws, smaller coarse lr.

## 2026-10-14 · chot · started
Approach: ideas #12. cfast trained on eps-hot behaviour draws (each draw hot w.p. eps, keyed by coordinate noise; hot draws sample the whole-logit tempered q^(1/tau_b)), every world's actor terms weighted by the product of bounded mixture ratios q/((1-eps)q+eps q_b) over its hot levels (superset of every term's ancestral closure: unbiased, constraint 3). Gate 1 gradcheck, Gate 2 clamp eval at 300 s, Gate 3 bench vs cfast at 300 s. Cores 40-47.
Result: -
Next: -

## 2026-10-14 01:32 · ccur · started
Approach: ideas #7 as an lr schedule on cfast (engineering only): actor lr 0 for levels 3-4 (frozen at init) while levels 0-2 train, then staged release fine to coarse (L3 at 35 %, L4 at 55 % of post-setup time); critic trains at all levels; tau anneals as cfast. csat diagnostics plus line_match per update. Gate: ground 300 s + clamp eval (L = 4, 3), at release and at the end; bench ccur vs cfast at --train 300, 2 seeds. Cores 24-31.
Result: -
Next: -

## 2026-10-14 02:10 · ctop · finished
Approach: top-level (level 4) draws of ground under cfast at init, 50 and 200 updates: coupled replays with the draw overridden to the probe's fill-following value or to all dirt; cone return split by term and level; plus a joint clamp of the whole top level (probe vs all dirt).
Result: at init the joint comparison is a tie (partial fill cheaper by 0.09 per fine cell at tau 2, dearer by 0.016 at tau 1); single draws: cert4 prefers dirt already (+0.018 per cell, 80 % of sites, lambda*viol), rho4 prefers fill (-0.011). After 50 updates all dirt is cheaper by +0.31 per cell, after 200 by +0.90 (soft energy at level 0). The preference for dirt is built by the fine levels' first updates. Report: reportables/2026-10-14_ctop/report.md.
Next: ccur's bottom-up order must beat the first-50-update drift; check cert4 candidate gradients (enumeration) early.


## 2026-10-14 01:28 · cconv3 · started
Approach: cconv_r2 vs cfast at 300 s/seed x 2 seeds on ground and columns (cores 32-39), smoke warm-up first. Report: reportables/2026-10-13_cconv/report.md (round 3).
Result: -
Next: -


## 2026-10-14 01:30 · cln · finished
Approach: cfg options on cfast: net (tanh | ln = LayerNorm + SiLU 64x2) and win (4x4 | 6x6), with the cone and margin following the window; variants cln (ln, 4x4) and cln6 (ln, 6x6). Tests: test_cln.py 11 passed (cone, finite support, defaults unchanged); test_cfast_cone + test_cresid 17 passed.
Result: GPU shared, 2 x 150 s, B = 8, none accepted (probes flagged everywhere). KL per block, s0 / s1: toy_support cln6 1.28 / 0.78, cln 0.45 / 0.90, cfast 0.70 / 0.58 (cfast misses the budget, 5.9e-3); toy_modes cln6 0.26 / 0.26, cln 0.16 / 0.22, cfast 0.15 / 0.16. Ground: all three collapse to all dirt (line ignored); cln6 s0 misses the budget (1.5e-3) after a 125 s compile. Microbenchmark: 5.1 / 6.1 / 6.0 ms per update (cfast / cln / cln6), but the bench gave cln* 2.5-4x fewer updates (contention). The exact-gradient gains do not transfer at B = 8: variance binds. Report: reportables/2026-10-13_cln/report.md.
Next: fixed-update-count comparison; cln6 at large B (cbig).

## 2026-10-14 baselines300 · finished
Approach: cfast 300 s x 2 seeds on all five problems, cbig256 300 s on the toys (ideas: none, baselines).
Result: nothing accepted. cfast F gap to F* toy_support 0.016/0.026 (150 s: 0.055), toy_modes 0.011/0.011; cbig256 gap 0.003/-0.003 (support), 0.003/0.002 (modes), exact stats and seeds ok, probes still flagged. Ground still all dirt (viol 3e-5), columns still near-all air (viol 1.5e-2), dimers diagonal still 0 (viol 1.2e-4). Doubling the budget changed no acceptance and no collapse. Report reportables/2026-10-14_baselines300/report.md.
Next: compare variants against these 300 s rows.

## 2026-10-14 baselines300 · started
Approach: cfast at 300 s/seed on all five problems plus cbig256 on the toys.
Result: -
Next: -

## 2026-10-14 01:20 · creplay2 · started
Approach: make creplay cheap enough to compare with cfast at equal wall clock (engineering only): profile one update, cut replay count/cost, fix the compile-cache miss, then GPU bench vs cfast (ground, toy_support) and an equal-update (300) comparison on ground.
Result: -
Next: -

## 2026-10-14 probes · finished
Approach: scored 512 toy_modes probes with a retrained cbig256 (60 s) against exact log p; tracked dimers orientation fractions untrained and through cfast training.
Result: toy_modes: 0/512 probes unreachable, no scoring bug; flag is a genuine, very powerful world-level test (shift 4.06 nats = KL(q||p)+KL(p||q) at F gap 0.0035/cell; flag needs gap under ~0.0005/cell) plus a 16-vs-16 sample-size weakness (exchangeable draws flag 53%). dimers: diagonal is present untrained (0.318), lost within ~20 updates in step with entropy collapse (0.78 to 0.21 nats/cell); not structural.
Next: relabelled-diagonal test to see if the loser follows geometry or label.

## 2026-10-14 probes · started
Approach: diagnose toy_modes probe flags (reachability/target mass/scoring) and dimers missing diagonal orientation (untrained vs trained).
Result: pending.
Next: see reportables/2026-10-14_probes/report.md.


## 2026-10-14 csat · finished
Approach: cfast on ground with per-level saturation diagnostics (entropy over non-forced draws, forced fraction, b spread, dirt), then actor lr 1e-3 and a τ anneal over the whole budget (engineering freedoms), each gated by the clamp test. 150 s per arm, one seed, shared GPU.
Result: the collapse starts at the top level. cert4 goes deterministic and dirt reaches 0.95 at about update 200 (lr 1e-3: 225, while rho3/2/1 are still at 2–3 nats). Lr 1e-3 keeps the fine levels stochastic (final H 0.37 / 1.37 / 0.70), but they stay context-blind (95–100 % identical draws under clamping). Gate, fill vs own: 1.58 vs 0.57 at L = 4 and 2.04 vs 0.37 at L = 3 (default 1.92 vs 1.22 and 2.00 vs 0.93; tauall 1.88 vs 0.71 and 2.11 vs 0.44). Every arm is all dirt, line_match 0.500. The bench was not run. [Report](2026-10-14_csat/report.md).
Next: enumerated per-candidate replays at fine count draws (allowed freedom), plus ideas #12 with whole-logit tempering (τ does not temper b).

## 2026-10-14 csat · started
Approach: test whether the fine-level learned bias saturates early (deterministic rho draws, never visiting fill contexts) on ground: per-level entropy / forced fraction / |b| logged over training for cfast, then two engineering levers (lr, tau schedule, clipping; allowed freedoms) with the clamp gate from cresid.
Result: pending.
Next: see reportables/2026-10-14_csat/report.md.

## 2026-10-14 cconv (round 2) · finished
Approach: equal-update comparison cconv_r2 / cconv / cfast (ideas #5); report: reportables/2026-10-13_cconv/report.md (round 2).
Result: at ~equal updates conv F is better or equal (toy_modes KL 0.026 vs 0.077; toy_support 0.01-0.04 lower) but toy_support violations are 10-50x higher early; at 150 s cconv_r2 (2.5-3.8k updates) F -0.441/-0.451 vs cfast (10-15k) -0.411/-0.449. None accepted.
Next: cconv_r2 on ground/columns at B=16 with longer budget.

## 2026-10-14 cconv (round 2) · started
Approach: added cconv_r2; equal-update and equal-wall-clock comparisons on toy_support/toy_modes.

## 2026-10-14 01:05 · cconv · finished
Approach: first GPU bench of cconv (ideas #5) vs cfast, 2 seeds x 150 s, shared GPU.
Result: cconv worse than cfast everywhere and 5-10x fewer updates. toy_support F -0.395 vs -0.416 (F* -0.4688), worst final viol 1.8e-2 vs 2e-3; toy_modes F 0.29-0.33 vs 0.268 (F* 0.2513); ground (B=16 after B=64 OOM) viol 7.6e-2, streaky dirt, no fill line, one seed 0 updates (cold compile 240 s); columns viol 0.18 vs 3e-2. None accepted. [Report](2026-10-13_cconv/report.md).
Next: r=2/width 16 (3x faster) with a longer run to test whether the gap is throughput.

## 2026-10-14 00:20 · cconv · started
Approach: bench cconv vs cfast on toy_support, toy_modes, ground, columns.
Result: -
Next: -


## 2026-10-14 01:00 · creplay · finished
Approach: finished the importance-weighted coupled-replay estimator (t1 everywhere, t2 dropped at branched levels, (N_c/R)·Σ_selected c_i). Tests: replay faithfulness 4/4, gradcheck 4/4 (CPU). GPU shared with other agents.
Result:
- Per-world SNR at init on ground: level 4 0.010 → 142, level 3 0.014 → 93 (variance /13 900 and /6 800; means agree).
- But updates are slow: 1.4–2.0 s/update on ground at R = (8,8) / (16,16), K = 4, vs 0.07 s for cfast. Compile takes 140–350 s and misses the cache, so seed 0 ends up untrained.
- Ground bench: creplay worst final viol 6.3e-2 (s0 untrained; s1 4.7e-3), F 1.626, 0 / 201 updates. cfast: 1.5e-3, F 0.792, 1142 / 2028 updates. creplay still collapses to ~98 % dirt.
- toy_support: creplay s1 KL 0.026 (3953 updates) vs cfast 0.031 (6001).
- Not accepted. Columns not run. [Report](2026-10-13_creplay/report.md)
Next: cut replay cost (K = 2, top level only, larger B), fix the compile-cache miss, rerun ground and columns at equal updates, run the clamp diagnostic.

## 2026-10-14 01:30 · fixes (round 2) · finished
Approach: relaxed cpass.py:256 assert (rtol 1e-2, atol 1); CPU cpass tests 7 passed; GPU benches 2 x 150 s warmed, shared GPU.
Result: cpass ground no crash: F 1.12/0.77, 280/506 updates, all dirt, line_match 0.5. ctrunc_full ground: F 0.715/0.595 (Mac 0.72), 338/470 updates, fine viol 1.4e-4, levels 3-4 about 1e-2, dirt 0.94-0.96, no fill line. cfast columns fair budget (5468/8333 updates): still almost all air (density 0.06/0.02), viol 1e-2. None accepted. Board regenerated.
Next: see 2026-10-13_gpu_baselines/report.md round 2.

## 2026-10-14 01:10 · fixes (round 2) · started
Approach: relax cpass.py:256 assert to rtol 1e-2/atol 1 (float32 GPU reduction order), then GPU bench (2 x 150 s, warmed) cpass ground, ctrunc_full ground, cfast columns rerun; regenerate board. Shared GPU, cores 56-63.
Result: pending.
Next: report in 2026-10-13_gpu_baselines/report.md.

## 2026-10-14 01:25 · cbig2 · finished
Approach: batch size alone on toy_modes/toy_support (see started entry), tag r4, shared GPU.
Result: toy_modes F gap to F*=0.2513: cfast 0.0069/0.0119, cbig64 0.0044/0.0040, cbig256 0.0034/0.0036, cbig1024 0.0056/0.0027 (exact MLP would be 0.0008): 32x batch closes ~half to two thirds, plateaus, B=1024 no better. toy_support cbig256 gap 0.007/0.020 vs cfast 0.027, plateaued; only probes fail. Nothing accepted (probes flagged on all). Report: reportables/2026-10-13_cbig/report.md (round 4).
Next: 6x6 actor window + LN/SiLU at B=256; probes failure (tail coverage) needs its own diagnostic.

## 2026-10-14 01:00 · cbig2 · started
Approach: does B=256/1024 alone close the toy_modes variance gap (exactopt: sampled B=8 KL 0.19/block vs exact 0.013)? Bench cfast, cbig64, cbig256, cbig1024 on toy_modes and cfast, cbig256 on toy_support, 2 seeds x 150 s, shared GPU, tag r4. Added `cbig1024` variant line to cbig.py.
Next: report in 2026-10-13_cbig/report.md.

## 2026-10-14 01:00 · cln · started
Approach: acting on exactopt (ideas.md #10, #5): per-variant cfg options for the actor network (LayerNorm + SiLU 64x2; default tanh) and the actor window (6x6, offsets -2..3; default 4x4), with the cone and margin following the window. Variants cln (LN+SiLU, 4x4) and cln6 (LN+SiLU, 6x6) on cfast.
Result: pending.
Next: bench cln6 / cln / cfast back to back on toy_support, toy_modes, ground.

## 2026-10-14 00:45 · exactopt · finished
Approach: ideas.md #10 round 3, exact gradients on enumerable blocks (CPU): collected the toy_support / tiny runs, then a representation study (LN+SiLU, width/depth, lr, schedule) and an information study (actor window; same-colour independence).
Result: variance dominates at B = 8 (sampled 0.18 vs exact 0.013 on modes; 1.56 vs 0.30 / 0.012 on support). The tanh MLP's support plateau (0.30) is optimisation, not capacity: lr 1e-2 gives 0.012 and LN+SiLU 64x2 gives 0.0084 (table 0.0074). The modes 0.012 floor is the 4x4 actor window: 6x6 gives table 0 and MLP 0.0009. On tiny, the history table floors at 0.037 from same-colour independence (the chain-rule diagnostic gives 0.002); the 8-wide window takes table_feat from 0.32 to 0.07. Report: reportables/2026-10-13_exactopt/report.md.
Next: LN+SiLU with a 6x6 window as the actor default, then sampled training at large B to close the variance gap.

## 2026-10-14 01:10 · cresid · finished
Approach: generic count-residual features (parent view minus known siblings' sum, unknown siblings, parent-unknown flag, per declared count factor), variant cresid = cfast + features; clamp eval as gate.
Result: gate failed. Fine cost in fill contexts 2.02 vs 0.74 own (L=4; cfast 1.97 vs 0.72), 2.45 vs 0.38 (L=3; cfast 2.23 vs 0.34); rho3/2/1 still 97-100 % identical across contexts, deterministic (log q ~ 0). Bench ground 2x150 s (shared GPU): cresid worst final 6.1e-5 (in budget), F 1.41, 477/707 updates; cfast 2.4e-4 (misses), F 0.86, 803/765 updates; both all dirt, not accepted. Tests pass (new test_cresid, test_cfast_cone, finite support 0). Report: reportables/2026-10-13_cresid/report.md.
Next: check whether bias saturation (deterministic rho early) blocks learning; lower actor lr / slower tau anneal; residual in cconv's candidate head.

## 2026-10-14 00:40 · cresid · started
Approach: per declared count factor, a feature for the children: parent view minus known siblings' view sum, unknown siblings in the subset, parent-unknown flag (generic, from declarations; "more information to learned parts"). Variant cresid = cfast + these features. Gate: clamp eval at L=4/3.

## 2026-10-14 00:45 · clamp · finished
Approach: clamp diagnostic: levels >= L clamped to probe-lifted (fill-following) coarse states vs own coarse draw, same noise; cfast 150 s on ground (1357 updates, shared GPU) and toy_modes; untrained reference.
Result: hypothesis confirmed, gap created by training. Ground: trained fine cost below L=4 is 1.97 nats/cell in fill contexts vs 0.72 own (+1.26); below L=3 2.23 vs 0.34; the probe's own soft energy is 0.014, so fine KL >= ~1.96/cell. Untrained: no gap (2.32 vs 2.47). Gap is count-prior soft energy; fine violations rise only to 0.3-0.5 %/level, forced fractions are unchanged. Trained rho3..rho1 draws are bit-identical whatever the parent (context-blind); the count prior is visible only to the last child. toy_modes: fine KL 0.36 nats/block in non-mode contexts (untrained 1.00), 0 in modes; the coarse level over-selects modes (0.70 vs p* 0.55). Report: reportables/2026-10-13_clamp/report.md.
Next: give the fine learned parts the parent-count residual (parent rho minus known siblings, number unknown) and use this clamp test as the gate.

## 2026-10-14 00:08 · clamp · started
Approach: test the fill-avoidance hypothesis directly: cfast trained 150 s on ground (GPU, shared) and toy_modes; training pass with levels >= L clamped to probe-lifted (fill-following) coarse states vs the model's own coarse draw, same noise; per-level cost, violations, forced / no-option fractions, trained vs untrained. Diagnostic only, no variant changes.
Result: pending.
Next: report in reportables/2026-10-13_clamp/.

## 2026-10-14 · cbig · finished
Approach: compile cache + autotune, test fix, large-B bench (reportables/2026-10-13_cbig/report.md).
Result: test_cbig 5 passed. Warm cache cuts toy_support B=256 setup 196 s -> 11 s (autotune 0 only ~10 % cold). Ground (GPU shared): cfast B=8 1040/1543 upd F 0.92/0.74; cbig B=64 922/1094 upd F 0.55/0.54; B=256 184/230 upd F 0.65/0.79; all collapse to all dirt regardless of B. toy_support: cbig256 F -0.455/-0.467 vs cfast -0.424/-0.444 (F* -0.4688), none accepted. SNR at init ~0 for B=8..256.
Next: SNR on a partly trained actor; large B only pays where step cost is flat.

## 2026-10-13 · cbig · started
Approach: round-3 pass on cbig: compile time (shared cache, autotune_level=0), fix test, bench B=64/256 vs cfast B=8 on ground, toy_support B=256, SNR vs B if time.
Result: running.
Next: bench.


## 2026-10-13 23:55 · exactopt · started
Approach: ideas.md #10 round 3. Collect toy_support / tiny exact runs; representation study on the toy_support block (LN+SiLU, wider/deeper, lr) with exact gradients; information study (wider actor window, same-step independence) on toy_modes / tiny. CPU only, cores 40-47.
Result: pending.
Next: verdict per problem in reportables/2026-10-13_exactopt/report.md.

## 2026-10-13 23:52 · creplay · started
Approach: finish creplay's importance-weighted refactor (coupled coarse replays, ideas.md coupled-replay item): at branched levels keep t1 for every draw, drop every score term t2, add (N_c/R)·Σ_selected c_i (enumeration or LOO). Gradcheck on CPU, then time R and bench creplay vs cfast on the GPU (toy_support, ground).
Result: pending.
Next: GPU bench, coarse SNR before/after.

## 2026-10-14 00:55 · baselines · finished
Approach: fresh GPU baselines (shared GPU, 2 seeds x 150 s): greedy and cfast on all five problems, cpass on toy_support and ground.
Result: nothing accepted. cfast final viol: toy_support 0, toy_modes 0, ground 1.0e-3 (rerun; first run seed 0 got 0 updates), dimers 2.0e-4, columns 7.3e-2. cpass toy_support 2.3e-3; cpass ground crashes on GPU (assertion in cpass.py:256, float32 branch-replay mismatch). ctrunc_full not run. Draws: ground all/mostly dirt, columns all air, dimers lose diagonal. Report: reportables/2026-10-13_gpu_baselines/report.md.
Next: look at the cpass float32 assertion; rerun starved seeds (cfast columns seed 0, 283 updates) when the GPU is quieter.

## 2026-10-13 23:51 · baselines · started
Approach: fresh GPU baseline board: greedy, cfast on all 5 problems; then cpass (toy_support, ground), ctrunc_full (ground). GPU shared.
Result: pending.
Next: report in reportables/2026-10-13_gpu_baselines/.

## 2026-10-13 · cconv · stopped early
Approach: ideas.md #5 on cfast: per-level conv trunk (3x3 stem + residual pre-LN SiLU 3x3 blocks, fixed radius r), dense each step; per-channel linear + candidate-scoring head (s(t), nv, lookahead, value embedding); loss re-evaluates the trunk on (2r+1)^2 patches at subsampled sites; critic on stop-grad trunk features; cone and margin from the conv window. Files: castlegen/variants/cconv.py, _cconv_core.py, tests/variants/test_cconv.py, reportables/2026-10-13_cconv/.
Result: implementation done; 15/15 tests pass on the box (dense == patch trunk, loss recomputes pass log q, scoring, changes-inside-cone on all 5 problems, open windows agree on toy_support and ground). No bench runs yet (session ended). Box note: pids.max 2304 and each jax process uses about 300-500 threads; run under `taskset -c <8 cpus>` (44 threads) or compiles die with pthread_create / ptxas errors.
Next: bench cconv vs cfast on the box (toy_support gate, dimers, ground, columns), B=64, r=4, under taskset.

## 2026-10-13 · crb · stopped early
Approach: ideas.md #8 on cfast: cone returns use each later draw's expected cost lbar_j = sum_a q_j(a) l_j(a) instead of its realised cost (tower-property unbiased).
Result: gradcheck passes (proj 1.017/0.989; negative control rejected z=10/-31). toy_support variance at frozen params: no reduction (ratio 0.99-1.01 at init; 0.99/0.95/0.73 at l0/l1/l2 after 1500 updates with trained critic). No benches run (stopped). [Report](2026-10-13_crb/report.md).
Next: ground variance check before any bench; joint (per-step) RB or combine with creplay. Box note: pids limit -> use taskset.

## 2026-10-13 · cmulti · stopped early
Approach: ideas.md #3 on cfast: P = 2 passes per level (pass 1 = cfast's pass; pass 2 redraws every site colour by colour given all current neighbours, own actor). Objective = auxiliary-variable upper bound E_q[S(x_P) + lam c + tau(log q(traj) - log r(x_{<P} | x_P))], learned per-site backward kernel r; needs user approval. Tests (11) pass on the box: P=1 = cfast, telescoping costs = S + log q(traj) - log r exactly, cone containment, finite support.
Result: partial. GPU, 2 x 150 s: cfast toy_support viol 0, F -0.432 / -0.441. cmulti toy_support seed 0 viol 0, F_ub in info (bench F = None: score() unavailable). Smoke run (1 x 60 s): cmulti dimers collapses to 100 % vertical, log q(traj) ~ 0, so pass 2 does not break the collapse. On a valid matching, single-site redraws are frozen. Dimers and columns benches did not run: the box's pid limit killed XLA thread creation. scripts/chain2.sh (taskset, retry) is ready but was not started.
Next: run chain2.sh; block (pair) moves in later passes for frozen hard constraints; fewer colours.

## 2026-10-13 · cbig · stopped early
Approach: cfast on the GPU box with large batches; pre-selected subsample recording (memory), vectorised + overlapped designed inputs, batched cached generation with cached log q. Folder 2026-10-13_cbig/.
Result: no bench yet. cfast speed on 3090: toy_support flat to B~256 (4.5 ms @8, 22 ms @1024); ground saturates at B~64 (14/34/116 ms @8/64/256, OOM @1024, host designed inputs > GPU time). Fresh GPU compile 45-150 s eats the 150 s budget (autotune_level=0: 132 -> 29 s on toy_support); box pids limit 2304 vs ~316 threads/JAX process causes ptxas launch failures (taskset to 8 cores -> 44 threads).
Next: warm cache or scoped autotune off; profile cbig; SNR vs B; benches at B=256/1024; check collapse/fill tracking vs B.

## 2026-10-13 · exactopt (diagnostic) · stopped early
Approach: ideas.md #10. Exact J and its gradient by enumerating the reachable outcomes, with deduplicated draw contexts; mlp, feature-table and full-history-table actors, compared with cpass and cfast sampled training from the same init. Problems: tiny, and single toy_support / toy_modes blocks (non-periodic 4x4).
Result: machinery and tests built (5/6 tests pass; the history-table convergence test was lengthened and its rerun is unconfirmed). Partial toy_modes numbers only (KL from 2.04): exact mlp 0.90 at step 150; at 20 steps exact 0.57-0.81 against sampled cpass 1.79 and cfast 1.43. No verdict. The full runs are still going on the box (nohup, /tmp/eo_modes.log). [Report](2026-10-13_exactopt/report.md).
Next: pull /tmp/eo_runs, run toy_support and tiny, plot, and apply the decision rule.

## 2026-10-13 23:59 · creplay · stopped early
Approach: cfast + coupled replays for many coarse draws per world (cone-scored, enumerated when |A| <= K, else LOO), batched over (world, draw, action) on the GPU. Folder 2026-10-13_creplay/.
Result: no bench. Replay faithfulness tests pass (4 problems). Gradcheck passes in all 4 configurations (enumerated, LOO, random subsets, wrong critic; pre-refactor code). Ground on the GPU: 4096 replays/update = 6.1 s/update and 222 s compile, so it does not fit the budget as configured. Per branched draw, the across-action spread (~30) is ~1/35 of the return noise (~1000-1500): about 10^3x variance reduction per draw. Code is mid-refactor (importance-weighted subset estimator: t1 for all draws, (N/R)·Σ coupled terms for a random subset), so it does not run yet. Fill-context clamp diagnostic written (_creplay_clamp.py), not run.
Next: finish the refactor, R = 16 per level, bench against cfast on the GPU, run the clamp diagnostic.

## 2026-10-13 · exactopt (diagnostic) · started
Approach: ideas.md #10. Enumerate every reachable draw outcome of small hierarchies (gradcheck tiny; a single toy_support block; a single toy_modes block), precompute each draw's context once, and optimise the cpass actor with the exact gradient of J = E_q[S + lam H + tau log q]; compare exact KL curves with cpass- and cfast-estimator training from the same init and step count; a tabular actor (per feature context, and per full history) separates representation from optimisation.
Result: pending.
Next: -

## 2026-10-13 · Codex / cfast_head · finished
Approach: ideas.md #6(b) only on cfast: smooth absolute-domain cosine head, original inputs/critic/estimator. Cheap agent implemented; parent reviewed and benchmarked.
Result: ground, 2 x 150 s: worst final viol 9.16e-5 versus fresh cfast 6.59e-3; F 1.204 versus 0.813. Head seeds collapse to 50% dirt stripes / 100% dirt; both fail coverage probes and across-seed agreement. Nothing accepted. Updates 585/622 versus 926/825. Ten cone tests and four new head tests pass; gradient equivalence rechecked with actual cone returns. [Report](2026-10-13_cfast_head/report.md).
Next: keep as an ablation; creplay and cbig already address lower-variance credit. Columns and mask-relative follow-ups not run because coverage did not improve.

## 2026-10-13 23:10 · creplay · started
Approach: cfast plus coupled replays for many coarse draws per world (all draws at the top level, a random subset at the next), K actions each under the world's stored noise, each replay scored by the draw's dependency cone; enumeration when |A| <= K, leave-one-out baseline otherwise; replays batched over (world, draw, action) on the GPU. Gradcheck on the tiny enumerable problem; bench on the GPU box against cfast.
Result: pending.
Next: -

## 2026-10-13 · cconv · started
Approach: ideas.md #5 on cfast. Actor = per-level conv trunk (input planes: the level's channels embedded, known and exists flags, parent planes upsampled, position-in-parent, row), 3x3 stem + residual pre-LayerNorm 3x3 SiLU blocks at fixed radius r, recomputed densely each step; per-channel linear head plus candidate-scoring head (site features + value embedding, s(t), nv(t), lookahead). Loss re-evaluates the trunk on (2r+1)^2 patches at the subsampled sites (state at the step rebuilt from the final state and the static schedule). Critic reads stop-gradient trunk features. Cone and margin use the conv reach. Bench on the GPU box vs cfast.
Result: pending.
Next: -

## 2026-10-13 · cmulti · started
Approach: ideas.md #3 on cfast: P passes per level (pass 1 = cfast's coloured pass; passes 2..P resample every site, colour by colour, given all current neighbours, own actor per pass). log q(final) is intractable, so the objective is the auxiliary-variable upper bound E_q[S(x_P) + lam c + tau(log q(trajectory) - log r(earlier | x_P))] with a learned per-site backward kernel r (needs user approval before candidacy). Bench on the box vs cfast: dimers, columns, toy_support.
Result: pending.
Next: -

## 2026-10-13 · cbig · started
Approach: cfast on the GPU box with large batches (B = 8 ... 1024 worlds per update): profile s/update and memory vs B, remove whatever stops scaling, measure per-batch gradient SNR vs B, batch/jit-cached generation; bench toy_support, toy_modes, dimers, ground, columns (+ cfast on the box for ground, toy_support).
Result: pending.
Next: -

## 2026-10-13 · Codex / cfast_head · started
Approach: isolate ideas.md #6(b): cfast with the smooth cosine output head (8 terms per declared ordered view, 0.1 residual, absolute coordinates), unchanged input features and critic. Cheaper agent implements isolated variant; parent reviews and runs checks. User permits up to two concurrent jobs.
Result: pending. Paired ground benchmark: cfast versus cfast_head, two seeds x 150 seconds, 16 held-out worlds per seed.
Next: assess violations per level, dirt excess, floating cells, probe availability, and throughput; advance to columns only if promising.

## 2026-10-13 16:50 · cordinal · finished
Approach: ideas.md #6 on cpass, estimator untouched: (a) thermometer + one-hot inputs, (b) cosine-basis actor head (8 per ordered view) + 0.1 residual for D > 8, (c) basis over the range of A. Folder 2026-10-13_cordinal/.
Result: nothing accepted. toy_support viol 2e-3 (one repaired cell), F -0.387 (KL 0.082 vs cpass 0.095); ground viol 0.029 vs 0.070, F 1.846 vs 1.918, floating 153/651 vs ~1130, more dirt; columns viol 0.15 (= cpass), F -0.645, seed 0 collapsed to empty, 90 vs 150 updates (critic input 3-5x larger under (a)). Ground ablation: a only 0.090 (= cpass), a+b 0.043, a+b+c 0.029.
Next: critic back on cpass encoding (remove throughput confound), rerun columns; combine the smooth head with cfast's throughput.

## 2026-10-13 15:12 · cfast · finished
Approach: cpass with dependency-cone returns (conservative read-graph superset, box sums on device), critic on actor hidden layer (no recorded critic features), unbiased subsampled loss (64 sites/step/world), no replays, one fused jitted update, compiled functions shared across seeds, tau anneal after setup, lr 3e-3. Folder 2026-10-13_cfast/.
Result: 5-12x faster per update (ground 0.43 vs 5.2 s, columns 0.16 vs 0.89 s, same load); cone cuts gradient variance 357x / 8x / 1.8x at levels 0/1/2 (ground). Bench (2 x 150 s): toy_support viol 2e-3, F -0.441 (KL 0.028); ground 648/549 updates, viol 0 / 2.9e-2, F 0.885 but collapse to all dirt; columns 1110 updates, viol 2.1e-2, F -0.05, collapse to air (density 0.02-0.06). Nothing accepted.
Next: entropy-preserving schedules (ideas #1) on top of cfast; coupled replays for many coarse draws; smaller actor window for tighter cones.

## 2026-10-13 15:15 · ctrunc · finished
Approach: truncated-depth bootstrapped training (ideas.md #2). Generator/objective as cpass. A coarse-to-fine chain with batches shrinking toward the fine level (ground B = 1,1,4,16,64), local returns (box R = 3) truncated d levels down, and a per-cell value v_m (sampler encoder) regressed on d-step targets. No replays. Variants: ctrunc (d = 1), ctrunc_d2, ctrunc_full.
Result: nothing accepted. 2-5x cpass's updates/s (ground 2-3/s, columns 4-6/s).
- toy_support: F -0.441 (d1) / -0.445 (full) vs cpass -0.374 (opt -0.469); worst final viol 3.9e-3 / 4.9e-4; stats still off (flat blocks 15-25 %).
- ground: F 1.05 (d1) / 0.74 (d2) / 0.72 (full) vs 1.92. Fine-level viol 4-15x lower, but level 4 is 4-8 % after repair; d2 seed 0 is near-valid (1e-3) only by filling everything with dirt.
- columns: d1 viol 0.10, density 0.49; full collapses to an empty world.
- V: unbiased in the mean (|bias| < 0.015 nats per fine cell) but weakly informative (corr 0.1-0.7, rmse about sd). d = 1 is the worst depth.
Next: full-depth chain as default plus an entropy or soft-constraint schedule (ideas.md #1); a repair that doesn't push violations to level 4. Report text is in the agent's final message (the report.md write was refused for subagents).

## 2026-10-13 16:00 · cordinal · started
Approach: ideas.md #6 on cpass, estimator untouched: (a) thermometer (16 bins) + one-hot (n <= 32) input codes for views > 8 values; (b) smooth actor head for D > 8, b = sum_k c_k cos(pi k u) + 0.1 r_v; (c) u relative to the range of the least-violating set A. Bench toy_support, ground, columns vs cpass.
Result: pending.

## 2026-10-13 15:40 · gradcheck (cpass estimator) · finished
Approach: spec test 5 on an enumerable 2-level strip (44 614 reachable outcomes): exact J and grad J by enumeration vs the mean of cpass's actor-loss gradient over 2000 x 16 worlds per config (tau 1.5, lambda (2, 3), P(violation) 0.61).
Result: unbiased. Projection on grad J 0.97-1.03 (|z| <= 1.4) in every config: zero / trained / wrong critic, branches enumerated (K 4) and sampled with LOO (K 2), branch term alone, unbranched top draws alone; mean cost matches J. No bug. Variance is the issue: per-world SNR 0.15-0.36 on the tiny strip, ~1e-3 on toy_support and undetectable on dimers at init (zero critic). Actor grad clip binds 1.7 % (toy_support l0) / 5 % (dimers) of batches: a small, technically biasing component. Test: tests/variants/test_cpass_gradient.py; data in 2026-10-13_gradcheck/.
Next: variance reduction (dependency-cone returns, better critic, larger B); entropy loss is optimisation, not estimator bias.

## 2026-10-13 14:50 · ctemp · finished
Approach: cpass plus per-level τ, actor conditioned on log τ_l, schedules anneal / mix (all levels, τ ~ log-U[1, 3]) / coarse (only non-finest levels heated), last 50 % at τ = 1; λ controller only on τ = 1 batches; λ soft starts tried (ideas #1). Diagnostics: training-batch stats, exact coarse/fine KL split on the toys (reportables/2026-10-13_ctemp/kl_split.py).
Result: no improvement, nothing accepted. ctemp (coarse) vs cpass:
- toy_support: KL 0.095 / 0.089 vs 0.096 / 0.092, viol 4.9e-4.
- toy_modes: KL 0.012 / 0.014 vs 0.013 / 0.011, viol 0.
- dimers: collapse 96 % H / 99 % D vs 91 % / 99.9 % H, viol 1.3e-3.

Diagnosis:
- Dimers collapse at τ = 1 exactly when violations drop. A λ ramp from 1 only delays it, and the run then collapses to diagonal instead.
- toy_support has exact stats at λ = 1 (14 % violations); flat blocks return as λ → 20.
- toy_modes fine KL is 0.6–1.0 nats/block in non-mode contexts, so the coarse level avoids them. Heating the coarse levels cut fine KL 0.22 → 0.15 but raised coarse KL 0.10 → 0.16.
Next: lower-variance credit for coarse and early draws (cone returns, replays at every level), then retry the coarse τ schedule; more passes per level for dimers (ideas #3).

## 2026-10-13 14:30 · ctrunc · started
Approach: truncated-depth training (ideas.md #2, approved exception to constraint 3). Generator and objective as cpass; training is a coarse-to-fine chain with batches shrinking toward fine levels (on-policy start states, many cheap coarse worlds), returns local to the draw and truncated d levels down, the rest replaced by a learned per-cell value v_m (same encoders, finite window) trained by d-step bootstrapped regression across levels. No coupled replays.
Result: pending.
Next: bench ground, columns, toy_support; compare d = 1 with full depth.

## 2026-10-13 14:10 · gradcheck (cpass estimator) · started
Approach: spec test 5: exact J(theta) and grad J by enumerating every draw outcome of a tiny 2-level periodic problem, against the mean of cpass's actor-loss gradient over many batches (zero / trained / wrong critic, enumerated and sampled coupled replays, tau != 1, lambda > 0 with violations).
Result: pending.
Next: -

## 2026-10-13 14:10 · ctemp · started
Approach: entropy loss / mode collapse of cpass on toy_support, toy_modes, dimers. Diagnose first (free-energy gap and orientation fractions during training), then τ / λ schedules and a τ-conditioned actor trained over a range of τ (ideas #1), ending at τ = 1.
Result: –
Next: –

## 2026-10-13 14:03 · cfast · started
Approach: throughput for cpass on ground and columns: profile, dependency-cone returns (G_i summed over a conservative superset of draw i's read-graph descendants, via per-step box sums), cheaper/fewer replays, compiled-function reuse.
Result: pending.
Next: -

## 2026-10-13 13:55 · orchestrator / cpass · finished
Approach: the final spec as written (castlegen/variants/cpass.py): coloured pass with one-step lookahead mask, per-(level, channel) MLP bias, full-future returns, critic control variate, K = 4 coupled replays of one coarsest draw per world, bounded penalties, τ 2 → 1, one repair round.
Result: [report](2026-10-13_cpass-first-results/report.md).
- toy_support: viol 2.4e-4, KL 0.095 nats/cell.
- toy_modes: viol 0, KL 0.013.
- dimers: viol 8e-4, but collapsed to horizontal.
- ground: 7e-2, unchanged (60 updates).
- columns: 1.6e-1, density halved.
- Nothing accepted.
Next: gradient test on an enumerable case; entropy loss (ideas #1); throughput on ground and columns.

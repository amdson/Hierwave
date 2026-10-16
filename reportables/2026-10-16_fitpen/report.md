# fitpen: rebalance ML fit against the penalty in ckpop_hard_pen (2026-10-16, wave 3)

**Question.** Can the pass fit its kernel-refined population (pop line_match ~0.64; own draws 0.55) while the penalty-only term keeps L2 under 2e-4, by rebalancing fit vs penalty (engineering freedoms only)?

**Answer: yes, with a caveat (2 seeds).** Doubling the ML steps per iteration (`cfitpen_ml2`: upi 8 -> 16, the same 32 penalty steps, lr, λ control and kernels) raises bench line_match from 0.56-0.59 to 0.61-0.63. That is the reverse-KL mix's structure. L2 stays under budget on both seeds (1.2e-4, 1.8e-4). The cost is that pooled L2 doubles (1.5e-4 vs 7.6e-5; the CI straddles the budget). Seed 1 misses the budget on a single L3 instance, the same failure mode as ckpop_hard_pen's seed 2. Not accepted.

Objective: weighted ML on own draws + p*-invariant kernels (kernel-route amendment) plus cfast `pen_only` penalty steps (change request #2). Generation is unchanged. λ stays within [0, 20] (constraint 7). The GPU was shared (one light Sonnet job on GPU 0); SM clock 1560 MHz with no throttle reasons before every run.

## Choice of arms (and why not (a)/(c))
- (a) gate penalty on own raw violations under budget: in ckpop_hard_pen the own raw rates at L1-L3 stay at 1.5-10e-4, above the controller's 1e-4 target (λ_1..3 pinned at 20), so the gate would almost never fire at L1-L3. Not run.
- (c) λ ramp: the penalty steps use their own Adam state, and with pen_only the loss is Σ λ_l H_l. A common scale on λ therefore cancels in Adam's update, and only the ratios between levels matter. A ramp of λ_max would be close to a no-op. Not run. (This is also why lr and step count, not λ, move the cr2 trade-off.)
- Run instead: (b) `cfitpen_ml2` = 2x ML steps per iteration, and (b') `cfitpen_mllr2` = 2x ML lr (same step counts and iteration rate).

## Fast loop: 120 s, seed 0, GPU 0, back to back, bench-style eval on 16 worlds (`runs/arm_*`)

| arm | ML / pen upd | ms/iter | pop lm (last 10 it) | own8 lm (last 5 logs) | eval lm | eval raw L0/1/2/3 | train own raw L2 (last 10) | eval final L0/1/2/3/4 | F | offset rows L0/1/2/3/4 (raw, eval) |
|---|---|---|---|---|---|---|---|---|---|---|
| ckpop_hard_pen | 472 / 1889 | 2052 | 0.638 | 0.554 | 0.556 | 4.6e-5 / 1.5e-4 / 5.5e-4 / 1.5e-3 | 5.1e-4 | 3.8e-5 / 1.2e-4 / 1.8e-4 / 7.3e-4 / 5.9e-3 | +0.070 | +3.12 / 3.26 / 2.38 / 0.53 / 0.02 |
| cfitpen_ml2 (upi 16) | 848 / 1697 | 2252 | 0.657 | 0.599 | **0.591** | 5.0e-5 / 3.1e-4 / **5.5e-4** / 4.9e-4 | 8.4e-4 | 3.8e-5 / 1.7e-4 / 7.3e-4 / 0 / 0 | −0.079 | +0.80 / 0.70 / 0.40 / 1.09 / 0.02 |
| cfitpen_mllr2 (ml_lr 2e-3) | 472 / 1889 | 2032 | 0.650 | 0.596 | 0.583 | 9.2e-5 / **9.5e-4** / 3.7e-4 / 7.3e-4 | 1.1e-3 | 7.6e-5 / 4.0e-4 / 6.7e-4 / 2.4e-4 / 0 | −0.029 | −0.08 / 0.17 / −0.77 / −0.34 / 0.03 |

λ at the end: 16/20/20/20 (baseline) and 20/20/20/20 (both arms). At 120 s the final-violation columns are not converged; the baseline itself shows 5.9e-3 at L4. The offset column uses the measure from the offset agent (reportables/2026-10-16_offset): implied surface row from the level-l counts minus the designed line, on the 16 eval worlds' raw draws. More ML weight removes most of the mid-level dirt deficit (L0 +3.1 -> +0.8 / −0.1 rows). From the mllr2 run on, the per-iteration log also records `offset` for own8 and the population (pop +0.1 rows at every level).

ml2 met the bench gate: eval L2 raw 1.0x the baseline (1.6x on training draws), with line_match at the 0.59 line (own8 0.60). mllr2 trades L1 instead (6x raw).

## Bench: `cfitpen_ml2` ground, 300 s x 2 seeds, warm (run 20261016-052448_cfitpen_ml2_ground_fitpen)

Command: `scripts/bench.sh cfitpen_ml2 2` (= `scripts/bench_warm.py bench cfitpen_ml2 300 2`, the cr2 wrapper with tag fitpen: warm compile in-process, harness untouched). GPU 0 shared; SM 1560 MHz, no throttling, before the run.

| row | seed | ML / pen upd | final counts L0/1/2/3/4 | worst | budget | F | lm | rough | ρ 0-4 | starved % 0-4 |
|---|---|---|---|---|---|---|---|---|---|---|
| cfitpen_ml2 | 0 | 2128 / 4257 | 15 / 5 / 2 / 0 / 0 | 1.2e-4 | pass | −0.083 | **0.609** | 1.50 | .36/.63/.84/.42/−.26 | 0/3.4/8.0/23.9/29.4 |
| cfitpen_ml2 | 1 | 2128 / 4257 | 7 / 3 / 3 / **1** / 0 | 2.4e-4 (L3) | fail | −0.080 | **0.629** | 1.28 | .40/.82/.73/.40/−.22 | 0/5.8/10.7/16.7/42.9 |
| ckpop_hard_pen (cr2, 4 seeds) | 0-3 | ~1180 / ~4720 | L2: 0/1/1/3 | 3.1e-5..2.4e-4 | 3/4 | −0.04..+0.11 | 0.557-0.585 | 0.95-1.21 | | 0-5/9-13/21-26/23-32 |

Pooled (Clopper-Pearson 95 %): ml2 L1 6.1e-5, **L2 1.5e-4 [5.0e-5, 3.6e-4]**, L3 1.2e-4 (one event), L4 0. The baseline's pooled L2 is 7.6e-5 [2.5e-5, 1.8e-4]. Verdict fields: final violations fail (one L3 instance), raw vs final ok, across-seed ok, finite support ok, old probe flag fires. Mean F −0.082 (baseline +0.034). Board regenerated (`results/board.md`).

For comparison, cr2's pen16 (halved penalty steps) reached lm 0.626 with L2 4.9e-4 (fail). At the same structure, ml2 holds L2 at 1.2-1.8e-4. More fit is therefore not the same as less penalty: it moves the trade-off curve, though it does not remove it (L2 roughly doubles vs the baseline and roughness rises to 1.3-1.5).

## Failed / caveats
- 2 seeds only; one seed fails on a single L3 instance (1/4096). Pooled L2 CI straddles the budget.
- Fast-loop arms ran on one seed for 120 s. Final violations at 120 s are noisy (a few instances).
- (a) and (c) were not run (reasons above, by mechanism, not measured).
- Zero-change test: `tests/variants/test_cfitpen.py`, 2 passed. Params, λ and population are bit-identical to ckpop_hard_pen after one iteration on GPU. The offset logging, added afterwards, only adds log keys.

## Files
`castlegen/variants/cfitpen.py` (cfitpen, cfitpen_ml2, cfitpen_mllr2, cfitpen_pe2; knobs upi, ml_lr, pen_every; per-level offset in `_ground_stats`), `tests/variants/test_cfitpen.py`, `scripts/{chain.sh,bench.sh,bench_warm.py,offset_eval.py}`, `runs/`.

## Next
Run 2 more seeds of cfitpen_ml2 to settle the pooled L2. Then try upi 12, or ml2 with the L3 penalty steps scaled up (rkl_level_scale {3: 2}), to buy back the L3/L2 margin while keeping lm ≥ 0.61.

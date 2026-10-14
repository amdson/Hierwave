# cr2-penalty: ML plus the penalty-only violation term (change request #2, adopted)

2026-10-14, evening session · new box (2x RTX A4000 16 GB, 24 cores; `notes/box_rules_2026-10-14.md`, last section) · one GPU per job, no other load unless stated.

**Decision (user, in chat):** change request #2 adopted: the penalty term Σ_l λ_l E_q[H_l] may be added to the weighted-ML objective on its own, with the same λ control and the same unbiased estimator as in reverse-KL training. Recorded as an addendum in `notes/coloured_pass_constitution.md` (pending paste into the doc) and as the status line of #2 in `notes/constitution_change_requests.md`.

## What was built

- **`cfast` option `pen_only`** (`castlegen/variants/cfast.py`): after the training pass, every draw's recorded cost becomes λ_l × violations(a) instead of soft + λ_l × violations + τ log q; the actor loss's per-draw term ℓ(a) likewise. Everything else is cfast's fused update unchanged: the same draws, cone returns (box sums of the new costs), the critic as control variate (fitted to the new returns), the subsampled loss, and the adaptive λ_l ≤ 20 driven by the raw violation rate. K > 0 (coupled replays) is not supported with it.
- **Variants.** Own draws only (inside the rules): `ckpop_hard_pen` = `ckpop_hard_rkl` with `pen_only=True` (32 penalty steps per iteration, lr 1e-3, levels 0-3, λ restarted at 10); `ckpop_pen_all` the same on levels 0-4. Upper-bound arm (probe-lift contexts, change request #1): `ckernel_pen4w3f` = `ckernel_rkl4w3f` with `pen_only=True`; `ckernel_pen_all` without the L4 scale.
- **Dimers loop:** `scripts/dk_loop_pen.py`, the dimers-kernel loop with `pen_only=1`.
- **Scripts for this box:** `scripts/env.sh [gpu]`, `scripts/arm.sh` / `arm_eval.py` (instrumented 300 s arm + the bench's own eval), `scripts/bench_ck.sh` (warm smoke + 4-seed bench of a ckernel variant), `summ_dimers.py`, `pool.py`.

## 1. Gradient check (tests/variants/test_cpen.py, 6 tests pass, CPU 2.5 min)

Tiny enumerable two-level problem (spec test 5 machinery), τ = 1.5, λ = (2, 3), random nonzero actor, cone returns; exact ∇J_pen by enumerating every reachable outcome, J_pen = Σ_x q(x) λ·H(x).

- **Cost identity:** the summed draw costs of a world equal λ·H(x) from the reference energy (32 worlds, exact).
- **Unbiased:** with a zero critic and a deliberately wrong critic, the mean estimator gradient matches ∇J_pen on both parameter blocks (300 batches of 16; projection ratio 1.02 / 1.06 at level 1; level 0 passed at the thresholds but was 3σ off in that sample). A longer run with the zero critic (`scratchpad` diagnostic, seeds disjoint from the test's) settles it as noise: at 1800 batches the projection ratio is 1.013 (level 0, |∇| 2.4) and 0.994 (level 1, |∇| 55.9), z = 0.19 / −0.55 along ∇J_pen, Hotelling p 0.44 / 0.86, cosine 1.000; the z-scores shrink with M instead of growing.
- **Power:** the same estimator is rejected against the full ∇J (Hotelling p = 0, z = −23 / +12).
- **Draws unchanged:** on toy_support `pen_only` gives bit-identical states and violation counts to cfast; only the costs and loss differ, and the per-world cost sums to λ·violations.

## 2. Dimers pre-check (fast-loop step 1): negative

`dk_loop_pen.py`, start = the collapsed cfast 60 s model (h/v/d 0.467/0.533/0.000), k = 2 and 3 fine-block Gibbs + 4x4 multiple-try redraws, 8 ML updates (lr 1e-3) + **32 penalty-only steps** per iteration, 120 s, one seed, GPU 0 alone. References: part 2's reverse-KL mix (N = 32, 300 s) and the ML-only arm (`reportables/2026-10-14_dimers_kernel/`). Means over the last 10 iterations; `images/dimers_orient.png`.

| arm | ML updates | own draws h/v/d | refined h/v/d | viol/cell own | valid own worlds | λ end |
| --- | --- | --- | --- | --- | --- | --- |
| **pen32** (penalty only) | 2240 (120 s) | 0.512 / 0.488 / **0.000** | 0.481 / 0.454 / 0.065 | **7.9e-5** | 92 % | 19.6 |
| rkl32 (reverse-KL mix) | 1256 (300 s) | 0.496 / 0.503 / 0.001 | 0.472 / 0.459 / 0.069 | 3.4e-4 | 69 % | 20 |
| ML only (a2) | 1800 (300 s) | 0.358 / 0.323 / 0.319 | 0.353 / 0.325 / 0.322 | 5.6e-2 | 0 % | – |

- **Validity:** the penalty-only term keeps own draws inside the budget (7.9e-5 per cell, 92 % of worlds valid), better than the reverse-KL mix at the same step count (3.4e-4), and 1.8x the updates per second (no soft/entropy terms in the returns changes nothing in cost; the GPU was simply unshared).
- **Diversity:** the diagonal does **not** appear in own draws (0.000 throughout; the h/v split swings between 0.17/0.83 and 0.93/0.07 from iteration to iteration). The kernel-refined states carry d = 0.05-0.09 as in part 2, so ML is trained on a diagonal every iteration and the 32 penalty steps still remove it.
- **Reading.** On dimers the collapse pressure is not (only) the designed-energy and entropy terms: the penalty term alone, at λ ≈ 20 with a 2e-4 target, pushes q off the diagonal, consistent with cdim3's mechanism (under a dominant λ the locally safest move is a stripe, and shared weights carry it to every context). Change request #2 separates validity from the energy/entropy collapse, not from the penalty's own mode-seeking. So on dimers it does not reach the three-orientation target either; the valid three-orientation reference stays cbound_lam (λ ramp).

## 3. Ground: own-draws-only arm `ckpop_hard_pen` (inside the rules)

`ckpop_hard_rkl` (wave4-6: persistent population, 20 colour-Gibbs sweeps, 2 hard final sweeps, ML on all levels) with its 32 reverse-KL steps per iteration replaced by 32 **penalty-only** steps (lr 1e-3, levels 0-3, λ restarted at 10, cfast's λ control). Own draws only; no probe contexts anywhere in training.

### 3.1 Instrumented arm, seed 0, 300 s, GPU 0 alone (`runs/arm_ckpop_hard_pen_s0/`)

1136 ML + 4545 penalty updates (setup 1.0 s after a 350 s cold warm-up). Bench-style eval on 16 held-out worlds: **final violations within budget at every level** (L0 1.1e-5, L1 3.1e-5, L2 1.8e-4, L3 0, L4 0), F +0.038, line_match 0.552, dirt 0.43, ρ .30/.78/.72/.52/−.02, starved 0/3.7/9.6/25/25 %. Trajectory (`loop.jsonl`): the population leaves the dirt state as in ckpop_hard_rkl (population line_match 0.64-0.65, 94-100 % valid from iteration 33 on), own-draw line_match plateaus at 0.55-0.57 from update ~400; λ_1..λ_3 saturate at 20 (raw own-draw rates at those levels stay above the controller's 1e-4 target), λ_0 decays to 4. Reference (wave5-pop arm A, same recipe with reverse KL, throttled 3090): line_match 0.57, F −0.026, L1-L2 final 5.3e-4 / 9.2e-4, L4 2e-3.

### 3.2 Bench, 4 seeds × 300 s, GPU 0 alone (run 20261014-193641_ckpop_hard_pen_ground_warm): NOT accepted, 3 of 4 seeds within the violation budget

Warm (setup 1.0-1.1 s), 1176-1184 ML + 4705-4737 penalty updates per seed; SM clock 1560 MHz, no throttling.

| seed | final violated instances L0/1/2/3/4 (of 262144 / 65536 / 16384 / 4096 / 512) | worst | viol budget | F/cell | line_match | dirt | rough | valid raw worlds | ρ 0-4 | starved % 0-4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 6 / 2 / 0 / 0 / 0 | 3.1e-5 | **pass** | +0.038 | 0.557 | 0.44 | 0.95 | 9 / 16 | .31/.79/.78/.46/−.17 | 0/4.3/13.2/23.3/22.9 |
| 1 | 0 / 1 / 1 / 0 / 0 | 6.1e-5 | **pass** | +0.114 | 0.561 | 0.41 | 1.17 | 13 / 16 | .30/.75/.48/.33/−.17 | 0/5.5/10.9/26.2/28.1 |
| 2 | 13 / 6 / 1 / **1** / 0 | 2.4e-4 (L3) | fail | +0.025 | 0.559 | 0.46 | 1.21 | 5 / 16 | .30/.80/.75/.37/−.10 | 0/4.2/8.9/24.0/22.8 |
| 3 | 6 / 5 / 3 / 0 / 0 | 1.8e-4 | **pass** | −0.039 | 0.585 | 0.50 | 1.13 | 3 / 16 | .34/.81/.70/.70/−.15 | 0/4.8/9.4/21.2/31.6 |

Pooled final rates (Clopper-Pearson 95 %): L0 2.4e-5; L1 5.3e-5 [2.9, 9.0]e-5; **L2 7.6e-5 [2.5e-5, 1.8e-4], the whole interval under budget** (ckpop_hard_rkl in wave6: 3.5e-4 [2.2, 5.3]e-4, interval entirely over); L3 6.1e-5 (one event; wave6: 1.8e-4); L4 0.

Verdict fields: final violations fail (seed 2's single L3 instance; zero are allowed per seed), **across-seed agreement passes on every statistic** (first ckpop bench to do so; wave5/6 failed on roughness), raw vs final agree, finite support holds, the old probe flag fires (uninformative; 3-13 valid raw worlds of 16). Mean F +0.034.

### 3.3 Reading

- **The penalty-only term does what it was adopted for: it separates validity from the energy/entropy collapse pressure.** Own draws of the in-rules arm are inside the budget at L0-L2 with margin (L2 at 0.4x the budget, against 1.75x for the reverse-KL mix) and L4 stays at zero without any L4 term; the one miss is a single L3 instance on one seed, at the level where one instance is already 2.4e-4.
- **It costs structure.** Against the reverse-KL mix (wave6, 4 seeds): line_match 0.56-0.59 vs 0.60-0.63, F +0.03 vs −0.09, coarse ρ at L3 0.33-0.70 vs 0.25-0.57 (similar), L4 ρ ≈ −0.15 vs < 0, L3/L4 starvation 21-26 / 23-32 % vs 18-38 %. The dirt fraction per seed (0.41-0.50) and the population's line_match (0.64) are unchanged, so the loss is in how well the pass reproduces its own kernel-refined population under the extra pressure: with λ_1..3 pinned at 20, 32 penalty steps per 8 ML steps pull the fine conditionals toward the locally safe (flatter) completions. Images: `images/bench/ckpop_hard_pen_ground_s*.png`: flat terrain at the fill's mean height with isolated one-cell spikes and no hill-following, the same picture as the own-draws reverse-KL arms in `reportables/2026-10-14_best_samples/`; the accepted probe-context seeds (`images/bench/ckernel_rkl4w3f_l2x2_ground_s0.png`) follow the hills.
- **Dimers showed the same trade in its pure form** (section 2): valid, but the penalty alone keeps the diagonal out of own draws and makes the h/v split unstable.
- **Follow-ups queued (section 3.4):** less penalty pressure (16 steps; lr 5e-4) to see whether line_match recovers while L2 stays under budget.

### 3.4 Lower-pressure arms (seed 0, 300 s, GPU 1 alone, bench-style eval)

| arm | penalty steps / lr | ML / pen upd | final viol L0/1/2/3/4 | viol budget | F/cell | line_match | dirt | ρ 0-4 | starved % 0-4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ckpop_hard_pen (3.1) | 32 / 1e-3 | 1136 / 4545 | 1.1e-5 / 3.1e-5 / 1.8e-4 / 0 / 0 | pass | +0.038 | 0.552 | 0.43 | .30/.78/.72/.52/−.02 | 0/3.7/9.6/25/25 |
| ckpop_pen16 | 16 / 1e-3 | 1392 / 2785 | 8.4e-5 / 4.6e-5 / **4.9e-4** / 0 / 0 | fail (L2) | −0.110 | 0.626 | 0.50 | .39/.75/.85/.61/−.20 | 0/1.2/8.1/21/38 |
| ckpop_pen_lr5 | 32 / 5e-4 | 1176 / 4705 | 8.0e-5 / 1.8e-4 / **3.7e-4** / **1.2e-3** / 0 | fail (L2, L3) | −0.118 | 0.621 | 0.50 | .37/.79/.83/.65/−.16 | 0/1.8/7.7/16/37 |
| ckpop_hard_rkl (wave6, 4 seeds) | 32 RKL / 1e-3 | ~900 / ~3600 | L2 pooled 3.5e-4 | fail | −0.090 | 0.60-0.63 | 0.49 | | 18-38 at L3/L4 |

Halving the pressure (steps or lr) restores the structure of the reverse-KL arm (line_match 0.62-0.63, F −0.11, dirt 0.50, L1-L3 ρ 0.75-0.85) and loses the budget at L2 (2-2.5x over) and, for the lr arm, at L3. So within this family validity and structure trade monotonically along the pressure axis; the penalty-only term shifts the curve (at equal structure it is more valid than the reverse-KL mix: pen16 L2 4.9e-4 at lm 0.63 vs rkl 3.5e-4 pooled at lm 0.60-0.63 is not better, though; at equal validity it is at lm 0.55) rather than breaking the trade-off. The next lever is not pressure but a better-shaped signal (graded distances, guided revisions: `reportables/2026-10-14_graded/`, `_guided_step/`, `_crevise/`).

## 4. Ground: upper-bound arm `ckernel_pen4w3f` (probe-lift contexts, change request #1)

`ckernel_rkl4w3f` with the penalty-only term in place of the reverse-KL mix (32 steps, lr 1e-3, L4 scale 0.3, λ control). Bench 4 x 300 s, GPU 1 alone (run 20261014-201533_ckernel_pen4w3f_ground_cr2), 1720-1984 ML updates per seed. **NOT accepted, but 4 of 4 seeds within the violation budget** (first 4-of-4 on ground on any arm).

| seed | final violated instances L0/1/2/3/4 | worst | F/cell | line_match | rough | valid raw | ρ 0-4 | starved % 0-4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 4 / 1 / 2 / 0 / 0 | 1.2e-4 | −0.041 | 0.589 | 0.68 | 8 | .30/.83/.69/.78/.68 | 0/4.2/10.9/12.0/1.3 |
| 1 | 4 / 1 / 1 / 0 / 0 | 6.1e-5 | −0.051 | 0.577 | 1.10 | 11 | .35/.84/.79/.80/.44 | 0/4.7/7.3/13.5/28.8 |
| 2 | 0 / 1 / 0 / 0 / 0 | 1.5e-5 | −0.023 | 0.557 | 0.46 | 13 | .35/.81/.76/.79/.52 | 0/4.8/10.8/20.9/5.4 |
| 3 | 0 / 0 / 1 / 0 / 0 | 6.1e-5 | +0.008 | 0.575 | 0.48 | 14 | .29/.72/.83/.75/.64 | 0/5.3/8.9/20.8/3.3 |

Pooled: L0 7.6e-6, L1 1.1e-5, **L2 6.1e-5 [1.7e-5, 1.6e-4]**, L3 0, L4 0. Fails: across-seed roughness (0.46-1.10), the old probe flag. Mean F −0.027.

Against the reverse-KL mix on the same arm (section 5: F −0.13, line_match 0.66-0.71, pooled L2 2.4e-4, 3 of 8 seeds): validity is solved on every seed and structure falls to the own-draws level (0.56-0.59). The probe contexts' advantage in line_match disappears under the penalty: the term pulls the fine conditionals toward flat, safe completions whatever the coarse contexts were trained on. Images `images/bench/ckernel_pen4w3f_ground_s*.png`.

## 5. Wave5-bench rows re-run on a cool GPU (upper-bound arm, unchanged objective)

Both rows of `reportables/2026-10-14_wave5_bench/` were training-starved by thermal throttling on the 3090 (288-352 updates on the affected seeds). Re-run here on GPU 1 alone (A4000, SM clock 1560-1905 MHz throughout, no throttle reason), warm smoke first, `--seeds 4 --train 300`. Objective unchanged (reverse-KL mix); upper-bound arm (probe-lift contexts, change request #1).

### 5.1 `ckernel_rkl4w3f_l2x2` (L2 reverse-KL scale 2): NOT accepted, 2 of 4 seeds pass

Run 20261014-192539_ckernel_rkl4w3f_l2x2_ground_cr2 (`results/bench.jsonl`). Every seed got 1816-1976 ML updates (7264-7904 reverse-KL steps), i.e. the intended training; the 3090's well-trained seeds had 920-1344.

| seed | ML / RKL upd | final violated instances L0/1/2/3/4 (of 262144 / 65536 / 16384 / 4096 / 512) | worst | viol budget | F/cell | line_match | ρ 0-4 | starved % 0-4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1976 / 7904 | 7 / 4 / 2 / 0 / 0 | 1.2e-4 (L2) | **pass** | −0.138 | 0.696 | .39/.85/.59/.81/.63 | 0/1.1/7.6/10.7/5.0 |
| 1 | 1856 / 7424 | 21 / 18 / **5** / **1** / 0 | 3.1e-4 (L2) | fail | −0.135 | 0.658 | .34/.84/.81/.82/.57 | 0/2.0/8.8/12.4/12.2 |
| 2 | 1816 / 7264 | 9 / 4 / 2 / 0 / 0 | 1.2e-4 (L2) | **pass** | −0.127 | 0.663 | .39/.78/.61/.81/.58 | 0/1.0/8.1/13.2/4.0 |
| 3 | 1824 / 7296 | 22 / 3 / **7** / 0 / 0 | 4.3e-4 (L2) | fail | −0.127 | 0.669 | .43/.76/.53/.73/.55 | 0/3.9/7.6/13.9/7.5 |

Pooled final rates (Clopper-Pearson 95 %): L0 5.6e-5 [4.3, 7.3]e-5; L1 1.1e-4 [0.74, 1.6]e-4; **L2 2.4e-4 [1.4, 4.0]e-4 (straddles)**; L3 6.1e-5 (1 event); L4 0.

- With the intended update count the arm is better than wave5's row on every summary: 2 of 4 seeds pass (1 of 4 before), mean F −0.132 (−0.061), line_match 0.66-0.70 (0.57-0.66), L3/L4 starvation 11-14 / 4-12 % (12-24 / 6-31 %).
- L2 remains the binding level: the pooled rate straddles the budget, and the two failing seeds have 5 and 7 L2 instances where 3 are allowed. One L3 event in 16384 (seed 1).
- Across-seed agreement fails on roughness (0.93-1.50 across seeds), as in every ckernel bench so far; the old probe flag fires.
- So the wave5 conclusion stands in a cleaner form: the L2 scale-2 step fixes L2 on about half the seeds and the rest miss by 2-4 instances; nothing here is budget-starved any more.

### 5.2 `ckernel_rkl4w3f_l2x15` (L2 scale 1.5): NOT accepted, 1 of 4 seeds pass

Run 20261014-195125_ckernel_rkl4w3f_l2x15_ground_cr2, 1824-1984 ML updates per seed, GPU 1 alone.

| seed | final violated instances L0/1/2/3/4 | worst | viol budget | F/cell | line_match | ρ 0-4 | starved % 0-4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 15 / 5 / 2 / 0 / 0 | 1.2e-4 | **pass** | −0.135 | 0.672 | .42/.86/.60/.82/.59 | 0/2.8/7.6/10.9/7.0 |
| 1 | 17 / 9 / **7** / **2** / 0 | 4.9e-4 (L3) | fail | −0.114 | 0.672 | .39/.76/.83/.81/.57 | 0/2.6/8.5/13.8/9.4 |
| 2 | 8 / 3 / 2 / **3** / 0 | 7.3e-4 (L3) | fail | −0.133 | 0.713 | .44/.76/.67/.76/.57 | 0/2.3/7.3/12.5/5.6 |
| 3 | 11 / 6 / **5** / 0 / 0 | 3.1e-4 (L2) | fail | −0.141 | 0.708 | .41/.80/.76/.71/.55 | 0/2.2/6.4/10.0/9.5 |

Pooled: L0 4.9e-5; L1 8.8e-5; L2 2.4e-4 [1.4, 4.0]e-4 (same as scale 2); **L3 3.1e-4 [1.0, 7.1]e-4 (5 events)**; L4 0. Mean F −0.131, line_match 0.67-0.71, across-seed roughness fails.

- Scale 1.5 does not keep L3 at zero: 5 L3 events against 1 at scale 2 and 0 at scale 1 (wave3/4). So the hoped-for "scale between 1 and 2 that keeps L3 clean" is not there; L3 events look seed-driven rather than scale-driven.
- Both reruns together (8 seeds, ~1900 updates each): the upper-bound arm with the reverse-KL mix sits at pooled L2 2.4e-4 and passes 3 of 8 seeds. The wave5 numbers were indeed training-starved, and the clean picture is: F −0.13, line_match 0.66-0.71, L2 at the budget edge, occasional L3 singles.

## Verdict

- **Change request #2 does what it was adopted for.** With the penalty-only term in place of the reverse-KL mix, the violation budget is met on 3 of 4 own-draws seeds (pooled L2 interval entirely under 2e-4; the miss is one L3 instance) and on 4 of 4 probe-context seeds (pooled L2 6.1e-5, L3/L4 zero), with across-seed agreement passing on the own-draws arm. No row is accepted: the own-draws miss is a single instance, and every row fails the old probe flag (uninformative) or roughness agreement.
- **It does not break the validity/structure trade-off, it moves along it.** Full pressure: valid, line_match 0.55-0.59, F −0.03..+0.04. Half pressure: line_match 0.62-0.63, F −0.11, L2 2-2.5x over budget. On dimers: valid but no diagonal and an unstable h/v split. The penalty term alone is mode-seeking at λ = 20 (cdim3's mechanism), so dropping the energy and entropy terms removes one collapse pressure and keeps another.
- **Clean numbers for the wave5 rows:** at ~1900 updates per seed the reverse-KL probe-context arm sits at pooled L2 2.4e-4, F −0.13, line_match 0.66-0.71, 3 of 8 seeds pass; L3 singles are seed-driven, not scale-driven.
- **Next (the user's direction, agents running):** a better-shaped violation signal rather than more pressure: graded distances in the penalty (addendum c, `reportables/2026-10-14_graded/`), guided revising steps that read the graded field (`_guided_step/`, `_crevise/`); and the acceptance gate (v2, eval size) so single L3 instances stop deciding rows.

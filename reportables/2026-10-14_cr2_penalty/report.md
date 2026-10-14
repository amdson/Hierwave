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

(pending)

## 4. Ground: upper-bound arm `ckernel_pen4w3f` (probe-lift contexts, change request #1)

(pending)

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

### 5.2 `ckernel_rkl4w3f_l2x15` (L2 scale 1.5)

(pending)

## Verdict

(pending)

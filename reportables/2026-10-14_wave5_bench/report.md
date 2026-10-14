# wave5-bench: L2-only reverse-KL scale on the ckernel arm (UPPER-BOUND ARM)

2026-10-14 · agent wave4-bench (follow-up) · ground · RTX 3090 shared (wave6-pop's jobs; its bench finished before mine started) · cores 48-55.

> **Upper bound: probe-lift contexts (change request #1).** Everything here is ckernel: half of every training batch is probe lifts, outside the frozen constitution.
> **Objectives:** wave4-bench's `ckernel_rkl4w3f`, which is:
> - draw-wise weighted ML exp(-20 V) on kernel-refined states, on all levels;
> - cfast's λ-controlled reverse-KL (τ = 1, own draws, λ0 = 10, λ ≤ 20), lr 1e-3, 32 steps per 8 ML updates;
> - coarse kernels at λ = inf;
> - the fused loop (`fast=True`).
>
> New in this run: the L2 parameter step of the reverse-KL update is scaled by s (`rkl_level_scale={2: s, 4: 0.3}`; with Adam this is the L2 lr × s). L1 and L3 are unchanged; L4 stays at 0.3.

## 1. Arms (300 s, one seed, run back to back, start sweep2a arm B), 32 held-out worlds with the v2 block check

| arm | ML / RKL upd | final viol L0/1/2/3/4 (counts in 32 worlds) | L2 rate | line_match | starved % L3 / L4 | ρ L1/L2/L3/L4 |
|---|---|---|---|---|---|---|
| wave4 rkl4w3f (s = 1, reference, 4-way contention) | 504 / 2016 | 22 / 31 / 20 / 0 / 0 | 6.1e-4 | 0.63 | 16.1 / 7.0 | .84/.59/.83/.66 |
| **l2x2 (s = 2)** | 792 / 3168 | 7 / 15 / **5** / 2 / 0 | **1.5e-4** | 0.628 | 13.2 / 6.0 | .83/.58/.77/.62 |
| l2x3 (s = 3) | 664 / 2656 | 34 / 27 / 18 / 5 / 0 | 5.5e-4 | 0.606 | 13.5 / 11.1 | .80/.75/.76/.64 |

- **Rule** (lowest pooled L2 rate with line_match ≥ 0.60 and L3 starvation ≤ 17 %): both arms qualify, and **l2x2 is chosen** (L2 1.5e-4 vs 5.5e-4).
- **s = 3 is worse at L2 and also adds L3 violations.**
- **Even l2x2 has 2 final L3 violations in 32 worlds**, where the s = 1 runs had 0. This showed up again in the bench.
- **Timing (runs/timing_summary.txt):**
  - l2x2: 1.86 s/iteration, of which RKL 1.24 s and ML 0.17 s; 4.3 ML updates/s.
  - l2x3: 2.16 s/iteration.
  - This is a less-contended GPU than in wave4-bench, where the same loop took 3.45 s/iteration.

## 2. Bench: `--seeds 4 --train 300`, ckernel_rkl4w3f_l2x2

**Upper bound: probe-lift contexts (change request #1).**

- **Setup:** run 20261014-120214_ckernel_rkl4w3f_l2x2_ground, harness v2.
  - Warm: the arm_l2x2 run compiled through the bench's import path and shapes; setup was 69-102 s.
  - **Seeds 2-3 got only 344 / 288 ML updates, against 1344 / 920 for seeds 0-1.** Something else loaded the GPU during their training, so seeds 2-3 are undertrained.
- **Images:** images/bench/.

| seed | ML / RKL upd | final viol L0/1/2/3/4 (violated instances in 16 worlds) | worst | pass | F/cell | line_match | ρ 0/1/2/3/4 direct [subtree] | starved % 0/1/2/3/4 |
|---|---|---|---|---|---|---|---|---|
| 0 | 1344 / 5376 | 3.1e-5 (8) / 1.2e-4 (8) / 1.2e-4 (2) / **2.4e-4 (1)** / 0 | 2.4e-4 (L3) | no | -0.123 | 0.660 | .37/.78[.85]/.72[.71]/.84[.70]/.63[.34] | 0/3.4/9.6/12.1/5.5 |
| 1 | 920 / 3680 | 4.6e-5 (12) / 1.8e-4 (12) / 1.2e-4 (2) / 0 / 0 | 1.8e-4 (L1) | **yes** | -0.117 | 0.645 | .37/.79[.85]/.69[.66]/.71[.57]/.57[.28] | 0/2.4/8.9/13.2/14.8 |
| 2 | 344 / 1376 | 5.3e-5 (14) / **2.1e-4 (14)** / **3.1e-4 (5)** / **2.4e-4 (1)** / 0 | 3.1e-4 (L2) | no | -0.015 | 0.574 | .33/.78[.86]/.32[.30]/.78[.62]/.56[.28] | 0/4.2/11.8/19.8/11.0 |
| 3 | 288 / 1152 | 4.6e-5 (12) / 9.2e-5 (6) / **3.7e-4 (6)** / **2.4e-4 (1)** / 0 | 3.7e-4 (L2) | no | +0.009 | 0.568 | .40/.68[.75]/.30[.34]/.78[.66]/.43[.11] | 0/4.7/12.2/24.1/30.8 |

- **Pass count:** 1 of 4 seeds meets the budget at every level. Bench mean F is -0.061 (cfast 0.774).
- **The across-seed agreement check fails**, because the two undertrained seeds differ. Raw vs final agrees and finite support holds.

### Pooled final rates, Poisson 95 % intervals (4 seeds × 16 worlds)

| level | count / instances | rate [95 %] | wave4 rkl4w3f (s = 1) |
|---|---|---|---|
| L0 | 46 / 1 048 576 | 4.4e-5 [3.2e-5, 5.9e-5] | 3.0e-5 |
| L1 | 40 / 262 144 | 1.5e-4 [1.1e-4, 2.1e-4] | 1.2e-4 |
| L2 | 15 / 65 536 | **2.3e-4** [1.3e-4, 3.8e-4] | 2.9e-4 [1.75e-4, 4.5e-4] |
| L3 | 3 / 16 384 | **1.8e-4** [0.4e-4, 5.4e-4] | 0 |
| L4 | 0 / 2 048 | 0 [0, 1.8e-3] | 0 |

- **Well-trained seeds 0-1 alone** (920-1344 updates):
  - L2: 4 / 32 768 = 1.2e-4 [0.3e-4, 3.1e-4];
  - L1: 20 / 131 072 = 1.5e-4;
  - L3: 1 / 8 192.

## Verdict

**Not accepted: 1 of 4 seeds passes** (seed 1, worst 1.8e-4 at L1).

- **The ×2 L2 step fixes L2 where training got its normal update count.**
  - Seeds 0-1 have 2 L2 violations each, within the 3 allowed.
  - The pooled L2 rate is lower than at s = 1, but the intervals overlap, so the improvement is not significant.
- **The failures are now spread across levels and are partly budget-starved:**
  - Seed 0 fails at L3 by one violation (1 of 4 096; zero are allowed).
  - Seeds 2-3 lost about 75 % of their updates to GPU contention. They fail at L2, at L3 (one each) and on line_match (0.57). Their L3 starvation is 20-24 % and seed 3's L4 starvation is 31 %.
- **The L3 violations are new compared with s = 1** (0 of 16 384 in both earlier 4-seed benches). They appeared in the l2x2 arm (2 in 32 worlds), in l2x3 (5) and in the bench (3 / 16 384, pooled 1.8e-4). At 3 events this is not separable from noise, but it is consistent with a stronger L2 step pushing violations up to L3.
- **A clean read needs a bench without contention**, since 2 of 4 seeds were undertrained. An L2 scale between 1 and 2 (e.g. 1.5) might keep L3 at zero. Not run (time cap).

## Files
- **castlegen/variants/ckernel.py:** variants `ckernel_rkl4w3f_l2x2` and `ckernel_rkl4w3f_l2x3`.
- **reportables/2026-10-14_wave5_bench/:**
  - **scripts/:** run_ck2.py, eval_viol.py, bench_row.py, timing.py, env.sh, ckernel_snapshot.py.
  - **runs/:** arm_l2x2/, arm_l2x3/ (loop.jsonl, params), arm_*.log, eval_arm_l2x2/3.json and .log, timing_summary.txt, bench_ckernel_rkl4w3f_l2x2.log, bench_row.txt.
  - **images/bench/:** ckernel_rkl4w3f_l2x2_ground_s0-3.png.

## Part 2: ckernel_rkl4w3f_l2x15 (L2 scale 1.5), bench alone on an "idle" GPU (UPPER-BOUND ARM)

**Upper bound: probe-lift contexts (change request #1).**

- **Variant:** `ckernel_rkl4w3f_l2x15`, which is `rkl_level_scale={2: 1.5, 4: 0.3}` with everything else as in Part 1.
- **Warm:** a 40 s smoke run through the bench's import path (scripts/run_ck2.py; runs/smoke_l2x15). Its setup took 123 s, so the cache was cold for the new constants.
- **Bench:** `--seeds 4 --train 300`, run 20261014-123103_ckernel_rkl4w3f_l2x15_ground. No other GPU process was visible during the bench.

### The GPU was throttled, so this row is not the intended test
- **Updates:** every seed got only **328-352 ML updates** (41-44 iterations, about 4.8 s per iteration), against about 1300 expected. The chip was throttling, not shared:
  - After the bench, `nvidia-smi` showed 100 % utilisation, SM clock **255 MHz of 2100**, temperature 90 °C, P2, and active throttle reason 0x20 (software thermal slowdown); see runs/nvidia_smi_after_l2x15.txt.
  - So the GPU ran at about 1/8 clock.
  - This most likely also explains the 288-344 updates of Part 1's seeds 2-3, which ran from about 12:15. That is thermal throttling, not another agent's job as I assumed in Part 1.
- **Consequence:** this row measures an undertrained model (a quarter of the updates). It does not test the L2 scale of 1.5.

| seed | setup s | ML / RKL upd | final viol L0/1/2/3/4 (violated instances in 16 worlds) | worst | pass | F/cell | line_match | ρ 0/1/2/3/4 direct [subtree] | starved % 0/1/2/3/4 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 85 | 328 / 1312 | 8.0e-5 (21) / **2.3e-4 (15)** / **6.1e-4 (10)** / **9.8e-4 (4)** / 0 | 9.8e-4 (L3) | no | -0.007 | 0.600 | .37/.75[.83]/.43[.42]/.73[.52]/.61[.29] | 0/4.1/11.7/18.3/9.7 |
| 1 | 100 | 352 / 1408 | 2.7e-5 (7) / 1.2e-4 (8) / **4.3e-4 (7)** / 0 / 0 | 4.3e-4 (L2) | no | -0.011 | 0.582 | .34/.79[.84]/.78[.74]/.78[.62]/.56[.30] | 0/4.1/10.3/23.6/9.7 |
| 2 | 101 | 336 / 1344 | 1.1e-4 (29) / **2.3e-4 (15)** / **7.9e-4 (13)** / **1.5e-3 (6)** / 0 | 1.5e-3 (L3) | no | +0.011 | 0.559 | .35/.84[.88]/.71[.66]/.79[.67]/.54[.31] | 0/3.8/11.1/20.1/10.3 |
| 3 | 101 | 344 / 1376 | 8.8e-5 (23) / **2.3e-4 (15)** / **6.7e-4 (11)** / **4.9e-4 (2)** / 0 | 6.7e-4 (L2) | no | -0.012 | 0.581 | .36/.83[.87]/.67[.63]/.83[.72]/.51[.21] | 0/4.0/11.5/16.1/19.3 |

- **Pass count:** 0 of 4 seeds. Bench mean F is -0.005.
- **Across-seed check fails** on roughness; the other stats agree. Raw vs final agrees and finite support holds.

### Pooled final rates, Poisson 95 % intervals

| level | count / instances | rate [95 %] |
|---|---|---|
| L0 | 80 / 1 048 576 | 7.6e-5 [6.1e-5, 9.5e-5] |
| L1 | 53 / 262 144 | 2.0e-4 [1.5e-4, 2.6e-4] |
| L2 | 41 / 65 536 | 6.3e-4 [4.5e-4, 8.5e-4] |
| L3 | 12 / 16 384 | 7.3e-4 [3.8e-4, 1.3e-3] |
| L4 | 0 / 2 048 | 0 |

### Verdict
**Not accepted: 0 of 4 seeds pass.**

| level | allowed per seed (16 worlds) | observed per seed | excess |
|---|---|---|---|
| L2 | 3 | 7-13 | 4-10 instances, on all 4 seeds |
| L3 | 0 | 2-6 on seeds 0, 2, 3 | 2-6 instances |
| L1 | 13 | 15 on seeds 0, 2, 3 | 2 instances |

- **These are the numbers of a model at a quarter of its usual update count.** They match the undertrained seeds 2-3 of Part 1, and the 300-500-update runs of earlier waves.
- **No conclusion about scale 1.5 can be drawn.**
- **What would test it:** re-run the same command once `nvidia-smi` shows the SM clock back near 2 GHz (check `clocks.sm` and throttle reasons before benching), and confirm about 1300 updates on seed 0 before trusting the row.
- **Part 1's bench is affected too:** seeds 2-3 were throttled, so the clean evidence for L2 scale 2 is its seeds 0-1 only.

### Files (Part 2)
- **castlegen/variants/ckernel.py:** variant `ckernel_rkl4w3f_l2x15`.
- **runs/:** smoke_l2x15/ and smoke_l2x15.log, bench_ckernel_rkl4w3f_l2x15.log, bench_row_l2x15.txt, nvidia_smi_after_l2x15.txt.
- **images/bench/:** ckernel_rkl4w3f_l2x15_ground_s0-3.png.

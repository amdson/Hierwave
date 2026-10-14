# wave4-bench: fused ckernel loop, L1-L2 pressure arm, and a second 4-seed bench (UPPER-BOUND ARM)

2026-10-14 · agent wave4-bench · ground · RTX 3090 shared with 2-4 other jobs throughout (wave5-pop arms and bench, other agents' tests) · cores 48-55.

> **Upper bound: probe-lift contexts (change request #1).** Every number here comes from ckernel: half of every training batch is probe lifts, which is outside the frozen constitution.
> **Objectives (unchanged from wave3-bench's ckernel_rkl4w3):**
> - draw-wise weighted ML (exp(-20 V)) on kernel-refined states, on all levels;
> - cfast's λ-controlled reverse-KL (fused, τ = 1, own draws, λ0 = 10, λ ≤ 20), lr 1e-3, 32 steps per 8 ML updates;
> - L4 reverse-KL step scaled by 0.3;
> - coarse kernels at λ = inf.

## 1. Throughput (castlegen/variants/ckernel.py, cfg `fast=True`)

### What changed
- **Reverse-KL steps:** the 32 steps of an iteration now run as one jitted `lax.scan`.
  - Each step is the old body: cfast's `fused`, then the `rkl_levels` restore and the `rkl_level_scale` scaling, then cfast's λ control.
  - The λ control runs on the device in float32. The old code ran it on the host in float64.
  - Every step in an iteration uses the same rkey, as in the old loop, which keys on `self.updates`.
- **ML updates:** the 8 updates run as one jitted scan, with per-update minibatch indices and weights drawn exactly as before.
- **Host work in a worker thread:** probe lifts, the designed values of own draws and the designed values of the RKL seeds are built one iteration ahead.
- **Old path kept:** `fast=False` is the default and keeps the old loop, so ckernel_own and wave5-pop are unaffected. The public methods are unchanged and only new ones were added.
- **Variants:** `ckernel_rkl4w3f` is rkl4w3 with `fast=True`. `ckernel_rkl4w3f_l12x2` adds `rkl_level_scale={1: 2, 2: 2, 4: 0.3}`.

### Equivalence (scripts/check_fast.py, runs/check_fast.log)
- **RKL, 4 steps from one snapshot:**
  - The raw-violation vectors are identical.
  - The max parameter difference between the old and fused paths is 1.3e-5, against a step size of 4e-3 (float summation order).
  - λ agrees to 1e-6.
- **ML multi-update vs sequential:** parameters and losses are bit-identical (difference 0.0).
- **Tests:** tests/kernels pass, 73 passed and 1 skipped (runs/tests_kernels.log).

### Timing (scripts/timing.py, runs/timing_summary.txt; s per iteration)

| run | contention | ctx | refine | diag | ML (8 upd) | RKL (32 steps) | s / iter | ML upd / s |
|---|---|---|---|---|---|---|---|---|
| old loop, thr_old (150 s) | about 3 GPU jobs | 0.149 | 0.374 (incl. diag) | - | 0.441 | **2.096** | 2.99 | 2.67 |
| fast, arm_f (300 s) | 4 GPU jobs | 0.137 | 0.398 | 0.122 | 0.404 | **2.426** | 3.45 | 2.32 |
| microbench, 4 RKL steps, compiled (check_fast) | 1-2 jobs | | | | | old 0.33 s, fused 0.18 s | | |

- **Fusion removes host overhead but not the GPU time.** The RKL half takes 65-75 % of an iteration, at about 45-75 ms per step depending on contention. That time is the GPU work of a B = 8 coloured pass plus its gradient, which is latency bound, and fusion does not remove it. ML went from 0.44 to 0.40 s.
- **Updates per 300 s did not change measurably under this contention.**
  - Bench ML updates: 832-1256 (wave3: 816-1248).
  - Bench setup: 67-93 s (wave3: 38-75 s), because the new scans compile longer.
- **Honest verdict: no net throughput gain in the bench.**
- The real lever would be fewer, larger RKL steps (e.g. 16 steps at B = 16). That changes the optimiser schedule, so it is a different arm, not a semantics-preserving speed-up. Not tried.

## 2. L1-L2 pressure arms (300 s, one seed each, run in parallel, start sweep2a arm B), evaluated on 32 held-out worlds with the v2 block check

**Upper bound: probe-lift contexts (change request #1).**

| arm | ML / RKL upd | final viol L0/1/2/3/4 (counts in 32 worlds) | pooled L1-L2 | line_match | starved % L3 / L4 | ρ L1/L2/L3/L4 |
|---|---|---|---|---|---|---|
| rkl4w3f (scale L4 0.3) | 504 / 2016 | 4.2e-5 (22) / 2.4e-4 (31) / 6.1e-4 (20) / 0 / 0 | 3.1e-4 | 0.63 | **16.1** / 7.0 | .84/.59/.83/.66 |
| rkl4w3f_l12x2 (+ L1, L2 × 2) | 320 / 1280 | 5.9e-5 (31) / 2.2e-4 (29) / 5.5e-4 (18) / 1.2e-4 (1) / 0 | 2.9e-4 | **0.568** | **21.5** / 6.1 | .74/.69/.75/.60 |

- **Rule** (lowest pooled L1-L2 rate among arms with L3/L4 starvation ≤ 15 % and line_match ≥ 0.60): **no arm qualifies.**
  - l12x2 fails both conditions (line_match 0.568, L3 starvation 21.5 %).
  - rkl4w3f misses only the starvation limit, by 1.1 points at L3. I chose it as the closest arm.
- **The arm pair is not a clean comparison.** Both arms were starved of updates by 4-way GPU contention: 504 and 320 ML updates, against about 1000 in a bench.
- The ×2 L1-L2 step did not lower pooled L1-L2 violations beyond noise (2.9e-4 vs 3.1e-4). It cost line_match (0.63 to 0.57) and L3 starvation (16 % to 22 %).

## 3. Bench: `--seeds 4 --train 300`, ckernel_rkl4w3f

**Upper bound: probe-lift contexts (change request #1).**

- **Setup:** run 20261014-111542_ckernel_rkl4w3f_ground, harness v2.
  - Warm: the compile cache was warmed by the arm_f run, which used the bench's import path (`load_variants`, cache /workspace/jax_cache) and the same shapes.
  - The bench ran alone among benches, but the GPU was shared with the tail of other agents' tests.
- **Images:** images/bench/.

| seed | setup s | ML / RKL upd | final viol L0/1/2/3/4 (violated instances in 16 worlds) | worst | pass | F/cell | line_match | ρ 0/1/2/3/4 direct [subtree] | starved % 0/1/2/3/4 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 67 | 1256 / 5024 | 5.0e-5 (13) / 1.1e-4 (7) / **4.3e-4 (7)** / 0 / 0 | 4.3e-4 (L2) | no | -0.115 | 0.616 | .36/.82[.88]/.79[.74]/.76[.63]/.63[.34] | 0/3.3/7.5/15.6/5.4 |
| 1 | 93 | 848 / 3392 | 1.9e-5 (5) / 1.1e-4 (7) / **3.7e-4 (6)** / 0 / 0 | 3.7e-4 (L2) | no | -0.111 | 0.600 | .40/.80[.85]/.87[.84]/.76[.59]/.57[.23] | 0/2.7/6.8/15.7/15.5 |
| 2 | 86 | 832 / 3328 | 1.1e-5 (3) / 1.1e-4 (7) / 6.1e-5 (1) / 0 / 0 | 1.1e-4 (L1) | **yes** | -0.090 | 0.625 | .35/.79[.85]/.81[.77]/.82[.67]/.54[.20] | 0/2.1/10.0/16.4/10.4 |
| 3 | 93 | 1184 / 4736 | 3.8e-5 (10) / 1.5e-4 (10) / **3.1e-4 (5)** / 0 / 0 | 3.1e-4 (L2) | no | -0.121 | 0.670 | .38/.79[.85]/.85[.82]/.78[.61]/.55[.30] | 0/2.0/8.2/13.5/7.9 |

- **Pass count:** 1 of 4 seeds meets the budget at every level. Bench mean F is -0.109 (cfast 0.774). Not accepted.

### Pooled final rates with Poisson 95 % intervals (4 seeds × 16 worlds)

| level | count / instances | rate [95 %] | wave3 rkl4w3 |
|---|---|---|---|
| L0 | 31 / 1 048 576 | 3.0e-5 [2.0e-5, 4.2e-5] | 2.9e-5 |
| L1 | 31 / 262 144 | **1.2e-4** [0.8e-4, 1.7e-4] | 1.9e-4 |
| L2 | 19 / 65 536 | **2.9e-4** [1.75e-4, 4.5e-4] | 3.7e-4 |
| L3 | 0 / 16 384 | 0 [0, 2.3e-4] | 0 |
| L4 | 0 / 2 048 | 0 [0, 1.8e-3] | 0 |

## Verdict

**Not accepted: 1 of 4 seeds passes** (seed 2, worst 1.1e-4 at L1). The failing level is **L2 on all three failing seeds**: 5-7 violations in 16 384 instances, where at most 3 are allowed (2.4-3.6 × the 2e-4 budget).

- **L2 is a real but small miss.**
  - The pooled L2 rate of 2.9e-4 has a 95 % interval of 1.75e-4 to 4.5e-4. Its upper part is above budget, so this is most likely signal.
  - A true rate of 2.9e-4 passes a seed's L2 check (≤ 3 of 16 384) with probability about 0.3.
  - The rate is lower than wave3's 3.7e-4, but the two intervals overlap.
- **L1 is now within budget.** The pooled rate is 1.2e-4, with an upper bound of 1.7e-4 (wave3 was at the budget, 1.9e-4). The ×2 scaling was not used in the bench, so this is seed and run variation, not a fix.
- **L3 and L4 still have 0 violations on all 4 seeds.**
- **L3 starvation is borderline.** It is 15.6 / 15.7 / 16.4 % on three seeds, against wave3's 12-15 %. L4 starvation is 5-16 %.
- **What would move L2:** more RKL pressure at L2 alone (scaling L1 too cost line_match in the arm), or more updates per 300 s from fewer, larger RKL batches. Both are untested here and both stay inside cfast's admitted reverse-KL configuration, so no constitution change is needed.

## Files
- **castlegen/variants/ckernel.py:**
  - cfg `fast`;
  - methods `_rkl_scan_build`, `_rkl_steps_fast`, `_ml_multi_build`, `_host_prep`, `_submit_prep`, `_ctx_from_prep`, `_train_fast`;
  - variants `ckernel_rkl4w3f` and `ckernel_rkl4w3f_l12x2`.
- **reportables/2026-10-14_wave4_bench/:**
  - **scripts/:** check_fast.py, timing.py, ckernel_before.py (pre-change copy), plus run_ck2.py, eval_viol.py, bench_row.py, show.py, factor_table.py and env.sh (from wave3-bench).
  - **runs/:** check_fast.log, tests_kernels.log, timing_summary.txt, thr_old/, thr_fast/, arm_f/, arm_l12x2/ (loop.jsonl, params), eval_arm_*.json and .log, bench_ckernel_rkl4w3f.log, bench_row.txt.
  - **images/bench/:** ckernel_rkl4w3f_ground_s0-3.png.

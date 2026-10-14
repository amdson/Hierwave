# wave3-bench: a weaker L4 reverse-KL step, and the pass rate over 4 seeds (UPPER-BOUND ARM)

2026-10-14 · agent wave2-bench · ground · RTX 3090 shared with other agents · cores 48-55.

> **Upper bound: probe-lift contexts (change request #1).** This is ckernel: half of every training batch is probe lifts, which is outside the frozen constitution.
> **Objectives:**
> - draw-wise weighted ML (exp(-20 V)) on kernel-refined states, on all levels;
> - cfast's λ-controlled reverse-KL (fused, τ = 1, own draws, λ0 = 10, λ ≤ 20) at lr 1e-3, 32 steps per 8 ML updates, on L0-3;
> - the L4 reverse-KL parameter step scaled by s, so the L4 lr is 1e-3 × s (`rkl_level_scale={4: s}`; with Adam the step is linear in lr);
> - coarse kernels at λ = inf.
>
> Code: speed-1b's batched kernels.

## 1. Arms (300 s, one seed, instrumented, start sweep2a arm B), evaluated on 32 held-out worlds with the v2 block check

| arm | L4 RKL lr | ML / RKL upd | final viol L0/1/2/3/4 (32 worlds) | worst | line_match | starved % L3 / L4 | ρ pooled L3 / L4 [subtree] |
|---|---|---|---|---|---|---|---|
| rkl123 (bench2) | 0 | 464 / 1857 | 6e-5 / 1.1e-4 / 2.1e-4 / 6.1e-4 / 0 | 6.1e-4 (L3) | 0.60 | 19.9 / 3.0 (bench: 14.0, 10.9 / 2.3, 2.8) | 0.59[0.38] / 0.66[0.40] |
| rkl32inf (bench2) | 1e-3 | 472 / 1889 | 6e-5 / 1.0e-4 / 2.1e-4 / 0 / 0 | 2.1e-4 (L2) | 0.57 | 21.0 / **37.1** | 0.80[0.68] / 0.47[0.04] |
| rkl4w1 | 1e-4 | 440 / 1761 | 1.1e-4 / 4.7e-4 / 1.3e-3 / **3.1e-3** / 0 | 3.1e-3 (L3) | 0.60 | 23.5 / 4.3 | 0.82[0.68] / 0.66[0.43] |
| **rkl4w3** | **3e-4** | 440 / 1761 | 6e-5 / 3.2e-4 / 2.7e-4 / 2.4e-4 / 0 | 3.2e-4 (L1) | 0.61 | **12.5 / 11.5** | 0.82[0.67] / 0.62[0.29] |

- rkl4w1 and rkl4w3 each lost about 135 s of their 300 s to setup.
- **Rule** (lowest worst violation among arms with L3/L4 starvation ≤ 15 %): rkl4w3 is the only arm that meets the starvation limit in this eval, so it is chosen.
  - rkl123 meets the limit in its bench (11-14 %) but not in this eval (20 %).
  - rkl4w3 also has fewer raw L3 violations: 1 in 32 worlds, against 6 for rkl123.
- **rkl4w1 is worse than rkl123 even at L3.** One seed's run-to-run variation is large: 20 vs 6 raw L3 violations between two runs whose only difference is a 0.1-scaled L4 step. Single-seed comparisons of arms that differ by a few violations are not reliable.

## 2. Bench: `--seeds 4 --train 300`, ckernel_rkl4w3 (warm compile first: a 30 s smoke run in the bench's cache dir; the bench ran alone)

**Upper bound: probe-lift contexts (change request #1).**

Run 20261014-101123_ckernel_rkl4w3_ground (harness v2, git 21c9f19+dirty). Setup took 75 s on seed 0 and 38-43 s on seeds 1-3.

| seed | ML / RKL upd | final viol L0/1/2/3/4 (violated instances in 16 worlds) | worst | meets budget? | F/cell | line_match | dirt | ρ pooled 0/1/2/3/4 direct [subtree] | starved % 0/1/2/3/4 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 816 / 3265 | 4.6e-5 (12) / **3.2e-4 (21)** / **7.9e-4 (13)** / 0 / 0 | 7.9e-4 (L2) | no | -0.083 | 0.644 | 0.51 | 0.36/0.71[0.78]/0.76[0.73]/0.79[0.63]/0.65[0.42] | 0/5.3/8.2/13.5/3.8 |
| 1 | 920 / 3681 | 1.9e-5 (5) / **2.1e-4 (14)** / **2.4e-4 (4)** / 0 / 0 | 2.4e-4 (L2) | no | -0.127 | 0.665 | 0.49 | 0.39/0.83[0.87]/0.81[0.76]/0.68[0.51]/0.59[0.28] | 0/1.8/7.0/14.8/8.7 |
| 2 | 1112 / 4449 | 1.9e-5 (5) / 1.4e-4 (9) / 1.2e-4 (2) / 0 / 0 | 1.4e-4 (L1) | **yes** | -0.123 | 0.665 | 0.52 | 0.40/0.80[0.85]/0.85[0.82]/0.85[0.72]/0.56[0.25] | 0/2.0/6.6/12.2/12.4 |
| 3 | 1248 / 4993 | 1.9e-5 (5) / 7.6e-5 (5) / **3.1e-4 (5)** / 0 / 0 | 3.1e-4 (L2) | no | -0.121 | 0.662 | 0.52 | 0.41/0.82[0.87]/0.82[0.80]/0.75[0.58]/0.56[0.25] | 0/2.5/6.9/13.7/13.2 |

- **Pass count:** 1 of 4 seeds meets the budget at every level. Bench mean F is -0.113 (cfast 0.774).
- **Starvation:** L3 / L4 starvation stays at or below 15 % on every seed (12-15 % / 4-13 %).
- **Verdict items:** across-seed stats agree, raw vs final agrees, finite support holds. The v1 probe flag fires on seeds 1-3 (uninformative per BOX_RULES).

## 3. Evaluation noise: what one violation is worth

The bench uses n_eval = 16 held-out worlds per seed (W = 128, H = 64). Hard instances per level over 16 worlds, and the pass rule `worst < 2e-4`:

| level | hard instances (16 worlds) | one violation = | max violations that still pass |
|---|---|---|---|
| L0 | 262 144 | 3.8e-6 | 52 |
| L1 | 65 536 | 1.5e-5 | 13 |
| L2 | 16 384 | 6.1e-5 | 3 |
| **L3** | **4 096** | **2.4e-4** | **0** |
| **L4** | **512** | **2.0e-3** | **0** |

- **At L3 and L4 a single violation fails the bench.**
- **Poisson chance that a model whose true violation rate is r at every level passes a given level on one seed:**

| true rate r | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| 5e-5 (a quarter of the budget) | 1.00 | 1.00 | 0.99 | 0.81 | 0.97 |
| 1e-4 (half the budget) | 1.00 | 0.99 | 0.92 | **0.66** | 0.95 |
| 1.5e-4 | 0.98 | 0.88 | 0.77 | 0.54 | 0.93 |
| 2e-4 (at the budget) | 0.51 | 0.56 | 0.59 | 0.44 | 0.90 |

- **The rule is much stricter than "rate below 2e-4".** A model at half the budget everywhere passes all levels on one seed with probability about 0.66 × 0.92 × 0.95 ≈ 0.57, and on both seeds of a 2-seed bench with probability about 0.33. At L3, 16 worlds cannot tell 1e-4 from 0 with any confidence.

## Verdict

**Not accepted: 1 of 4 seeds meets the budget at every level** (seed 2, worst 1.4e-4).

- **The L3 / L4 problem is solved on this arm.** L3 and L4 have 0 final violations on all 4 seeds (0 of 16 384 and 0 of 2 048 pooled instances), and L3 / L4 starvation stays at or below 15 %.
  - The weaker L4 step (lr 3e-4) does what neither rkl123 (L3 violations) nor rkl32inf (L4 starvation 33-36 %) did.
- **The misses are at L1 and L2, and they are signal, not noise.**
  - **L2:** 24 violations pooled over 4 × 16 384 instances = 3.7e-4 (Poisson 95 % interval about 2.4e-4 to 5.5e-4). That is above the 2e-4 budget, and a true rate there passes a seed with probability below 0.5.
  - **L1:** 49 / 262 144 = 1.9e-4, at the budget, so passes are a coin flip.
  - **L3:** 0 / 16 384 gives a 95 % upper bound of 1.8e-4, so the L3 rate is consistent with passing.
- **The misses track update count.** Seed 0 had the fewest updates (816; 75 s setup) and the most violations (13 at L2); seeds 2-3 had 1100-1250 updates.
- **Next.** Every remaining violation is at L1-L2, so more RKL pressure there (e.g. scale the L1-2 step up or add steps) or more updates per 300 s are the levers. ckernel.py's ML / context half of the iteration is now the bottleneck (speed-1b's note: `make_multi_update`, prefetch).

## Files
- **castlegen/variants/ckernel.py:**
  - cfg `rkl_level_scale` ({level: s}; the RKL step of that level is scaled by s);
  - variants ckernel_rkl4w1 (s = 0.1) and ckernel_rkl4w3 (s = 0.3).
- **reportables/2026-10-14_wave3_bench/:**
  - scripts/: run_ck2.py, eval_viol.py (+ v2 block check), show.py, bench_row.py, factor_table.py, env.sh
  - runs/: rkl4w1, rkl4w3, smoke_rkl4w3 (loop.jsonl, params, logs); eval_*.json / .log; bench_ckernel_rkl4w3.log
  - images/bench/

# wave2-bench2: level-split objectives, λ = inf coarse kernels, warmed bench (UPPER-BOUND ARM)

2026-10-14 · agent wave2-bench · ground · RTX 3090 shared with other agents · cores 48-55 · runs two at a time.

> **Upper bound: probe-lift contexts (change request #1).** This is ckernel: half of every training batch is probe lifts.
> It is outside the frozen constitution.
> **Objectives:** draw-wise weighted ML (exp(-20 V)) on kernel-refined states, mixed with cfast's λ-controlled reverse-KL update (fused, τ = 1, own draws, λ0 = 10, λ ≤ 20, lr 1e-3, 32 steps per 8 ML updates). Each arm restricts the RKL step, the ML step, or both, to the levels given.

**Code.**
- **Bench runs** (and the smoke warm-ups and evals) use speed-1b's batched block_mh / coarse_flip / amortised, which were committed before they started.
- **Section 1's four arms** (split, fine_rkl, rkl32inf, rkl0123) also started after the coordinator's speed-1b notice; the split and fine_rkl update counts suggest these two did not get the faster kernels.
- **All of them** use ckernel.py with this follow-up's edits.

## 1. Arms (300 s, one seed, start sweep2a arm B; own raw violated fraction, mean of the last 3 diagnostics, 16 own worlds each)

| arm | ML levels | RKL levels | coarse kernels | ML / RKL upd | own raw L0/1/2/3/4 | line_match | dirt |
|---|---|---|---|---|---|---|---|
| split (coordinator's (1)) | 0-2 | 2-4 | λ = inf | 176 / 705 | 2.1e-4 / 7.8e-4 / **7.7e-3** / 9.8e-4 / 0 | **0.500** | **0.95** |
| fine_rkl | 0-4 | 0-2 | λ = inf | 176 / 705 | 4.2e-4 / 2.4e-3 / 4.0e-3 / 3.7e-3 / 0 | 0.558 | 0.58 |
| **rkl32inf** | 0-4 | 0-4 | λ = inf | 472 / 1889 | 1.1e-4 / 2.4e-4 / 2.4e-4 / 1.6e-4 / 0 | 0.584 | 0.47 |
| **rkl0123** | 0-4 | 0-3 (L4 excluded) | λ = inf | 464 / 1857 | 8.5e-5 / 4.9e-4 / 9.8e-4 / 4.9e-4 / 0 | 0.615 | 0.49 |

The split and fine_rkl arms lost 135 s to a cold compile in the bench's cache dir, so their update counts are lower.

**Held-out eval** (32 worlds; v2 block check run in the same process, own pool 128):

| arm | final viol L0/1/2/3/4 | ρ pooled 0/1/2/3/4 direct [subtree] | starved % 0/1/2/3/4 |
|---|---|---|---|
| rkl32inf | 6.0e-5 / 1.0e-4 / 2.1e-4 / 0 / 0 | 0.34 / 0.69[0.76] / 0.62 / 0.80[0.68] / 0.47[0.04] | 0 / 5.7 / 9.8 / 21.0 / **37.1** |
| rkl0123 | 6.0e-5 / 1.1e-4 / 2.1e-4 / **6.1e-4** / 0 | 0.35 / 0.75[0.82] / 0.80 / 0.59[0.38] / 0.66[0.40] | 0 / 3.1 / 9.3 / 19.9 / **3.0** |

**Findings.**
- **(1) The level split the coordinator asked for collapses the line.** With ML switched off at L3-4, those levels get only reverse-KL, and the arm is back in cfast's dirt mode: dirt 0.95, line_match 0.500.
  - Its L2 violations also stay high (7.7e-3), because L2's violations are caused by L3 choices.
  - The line-following on this arm comes from ML on probe-lift contexts at the coarse levels. Taking that away removes it.
- **RKL only on the fine levels (L0-2) is too weak** at the coarse levels: about 4e-3 at L2/L3.
- **RKL on L4 is the trade-off.**
  - **The L3 violations need RKL at L4.** With RKL on L0-3 only, L3 still has 5-6 violations in 32 worlds (support3 / honourB4: L4 certs the L3 children cannot realise next to their neighbours). With RKL on L4, the cone returns charge those L3 costs to the L4 draws, and L3 goes to 0.
  - **That costs L4 starvation:** 37 % with RKL on L4, against 3 % without, which is about the ML-only ckernel level of 3.5 %.
- **(2) The support3 mechanism and the kernels.** With λ = inf, every coarse move (single-cell Gibbs at L3/L4 and the flips) is conditioned on every touching hard instance. That includes the honour counts against the cell's current children and support against its neighbours.
  - So a refined coarse state always keeps a valid realisation: its current subtree.
  - The flips redraw the subtree from q, and the MH ratio at λ = inf rejects any violation in it. This is the invariant form of "run the fine sweep and drop invalid blocks". Dropping blocks on outcome outside an acceptance ratio is not p*-invariant, so I did not build that.
  - Refined coarse states have had 0 coarse violations ever since λ = inf: refined L4 is 0 in every arm, where λ = 20 gave up to 9e-2.
  - The support3 violations at generation time are q's mass leakage, and only RKL at L4/L3 removes them, as above.
- **(3) Compile warm-up.**
  - The bench imports every variant (`load_variants`), and in this environment that leaves the JAX cache dir at /workspace/jax_cache (BOX_RULES' JAX_COMPILATION_CACHE_DIR). My earlier runners imported only ckernel/cfast, which leaves ~/.cache/jax_cfast.
  - scripts/run_ck2.py and eval_viol.py now call `load_variants()` first, so they share the bench's cache.
  - A 40 s smoke run of each variant (setup 65 s with speed-1b's new kernels) came before the bench. Result: bench setup took 73-76 s on seed 0 and 38-40 s on seed 1, against 254 s cold for seed 0 last time. Both seeds made 490-660 ML updates.

## 2. Bench (v2, `--seeds 2 --train 300`, both variants run concurrently on a shared GPU)

**Upper bound: probe-lift contexts (change request #1).**

| variant (objective levels) | seed | ML / RKL upd | final viol L0/1/2/3/4 | F/cell | line_match | dirt | ρ pooled 0/1/2/3/4 direct [subtree] | starved % 0/1/2/3/4 |
|---|---|---|---|---|---|---|---|---|
| cfast (board, 300 s; RKL only) | | | worst 1.9e-5 | 0.774 | ~0.50 | | 0.21/-0.30/-0.74/-0.70/-0.79 | 1/76/100/100/100 |
| ckernel_rkl32 (previous wave; RKL all, λ = 20 kernels) | 0 / 1 | 248/993, 192/769 | worst 2.4e-4 (L2) | -0.012 | 0.56/0.58 | 0.49 | 0.33/0.80/0.47/0.75/0.47 | 0/4/10/24/42 |
| **ckernel_rkl32inf** (ML 0-4, RKL 0-4, coarse λ = inf) | 0 | 496 / 1985 | 8.8e-5 / **2.1e-4** / **2.4e-4** / 0 / 0 | -0.042 | 0.563 | 0.49 | 0.32/0.76[0.83]/0.61[0.58]/0.68[0.52]/0.51[0.05] | 0/4.2/9.6/22.6/**36.0** |
| | 1 | 664 / 2657 | 3.8e-5 / 1.5e-4 / **4.3e-4** / 0 / 0 | -0.084 | 0.592 | 0.49 | 0.39/0.84[0.89]/0.85[0.82]/0.74[0.60]/0.45[0.14] | 0/3.1/9.1/14.7/**32.5** |
| **ckernel_rkl123** (ML 0-4, RKL 0-3, coarse λ = inf) | 0 | 488 / 1953 | 1.1e-4 / **2.1e-4** / **2.4e-4** / **7.3e-4** / 0 | -0.082 | 0.619 | 0.50 | 0.38/0.78[0.85]/0.65[0.63]/0.70[0.55]/0.65[0.44] | 0/3.8/8.6/14.0/**2.3** |
| | 1 | 648 / 2593 | 2.3e-5 / 1.1e-4 / 1.8e-4 / 0 / 0 **(within budget)** | -0.098 | 0.646 | 0.50 | 0.38/0.81[0.86]/0.58[0.55]/0.82[0.70]/0.60[0.43] | 0/2.8/8.1/10.9/**2.8** |

Runs: 20261014-094739_ckernel_rkl32inf_ground and 20261014-094739_ckernel_rkl123_ground. Harness v2; git 4d27bcc+dirty.

- **Other verdict items.**
  - rkl32inf: across-seed agreement fails on roughness; raw vs final agrees; finite support holds; the v1 probe flag is n/a / n/a.
  - rkl123: across-seed and raw-vs-final agree; finite support holds; the v1 probe flag is n/a on seed 0 and set on seed 1. That flag is uninformative per BOX_RULES, but it is still part of the verdict.
- **Updates.** These are 2-3x the previous wave's, thanks to the warm-up and speed-1b's kernels. The two benches shared the GPU with each other and with other agents.

## Verdict
**No arm meets the 2e-4 budget on both seeds.**
- **Closest by the coordinator's preference (low starvation): ckernel_rkl123** (ML on all levels, RKL on L0-3, λ = inf coarse kernels).
  - **Seed 1 is within budget at every level** (worst 1.8e-4). F is -0.098 per cell, line_match 0.646, and starvation at L3/L4 is 10.9 % / 2.8 %.
  - **Seed 0 fails:** L3 is 7.3e-4 (3 violations in 16 worlds), and L1/L2 are 2.1e-4/2.4e-4 (14 and 4 violations).
  - Its L4 starvation (2-3 %) matches ML-only ckernel's, and its L3 starvation (11-14 %) is about ML-only ckernel's 10 %.
- **ckernel_rkl32inf** (RKL also on L4) removes L3 violations on both seeds. It still fails at L1/L2 (worst 4.3e-4), and L4 starvation is 33-36 %.
- **The residual violations are 1-4 per level in 16 worlds** on every arm, about the size of the seed-to-seed noise. Repair moves some of them upward: raw L2 6e-5 becomes final 1.8e-4.
- **Next step, if pursued:** RKL at L4 with a small separate lr or a step-count ratio, to clean L3 without starving L4. Or count the L3 violations against the L4 draws only through the cone (already the case), with fewer L4 steps. Not tried, for lack of time.

## Files
- **castlegen/variants/ckernel.py** (edited):
  - cfg `rkl_levels` (the RKL step's changes to other levels are reverted) and `coarse_lam` (λ of the coarse flips and Gibbs; inf = infeasible);
  - variants ckernel_split, ckernel_fine_rkl, ckernel_rkl32inf, ckernel_rkl123 (RKL on L0-3; the name is historical).
- **reportables/2026-10-14_wave2_bench2/:**
  - scripts/: run_ck2.py (bench-order imports), eval_viol.py (+ v2 block check), factor_table.py, show.py, bench_row.py, env.sh
  - runs/: split, fine_rkl, rkl32inf, rkl0123 and smoke_* (loop.jsonl, params, logs); eval_*.json / .log; bench_*.log
  - images/bench/

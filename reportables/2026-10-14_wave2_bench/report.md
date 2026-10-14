# wave2-bench: violation pressure for ckernel (UPPER-BOUND ARM)

2026-10-14 · agent wave2-bench · ground · RTX 3090 **shared** with three other agents · cores 48-55.

> **Upper bound: probe-lift contexts, outside the frozen constitution (change request #1).** Every number below is for
> ckernel with half its training contexts from probe lifts. That arm is a labelled diagnostic, not a candidate.
> **Objectives:** weighted ML on kernel-refined states (amendment 2026-10-14), plus, in the B arms, cfast's
> λ-controlled reverse-KL update on the sampler's own τ = 1 draws. The amendment allows the two to be mixed.

All arms start from sweep2a arm B params, as sweep2b's runs did, and use the kernel cycle from sweep2b (flips at L3/L4, coarse Gibbs, MTM at L2, fine Gibbs, kernels at λ = 20 unless stated).

## 1. Violation arms (one seed each, 300 s including setup, instrumented)

The table reports own raw violated fraction per level for L0 / L1 / L2 / L3, plus line_match and dirt. All are measured on the last diagnostic batch: 16 own worlds at W = 128. The column "eval raw / final" uses 32 held-out bench-noise worlds (scripts/eval_viol.py). L4 raw is 0 in every arm.

| arm | ML upd | RKL upd | setup s | own raw viol L0/1/2/3 (last diag) | line_match | dirt | eval raw L0/1/2/3 | eval final L0/1/2/3/4 |
|---|---|---|---|---|---|---|---|---|
| sweep2b ck300 (ML, w = 1) | 440 | 0 | 50 | n/a | 0.619 | 0.52 | 2.3e-4 / 8.1e-4 / 1.6e-3 / 9.8e-4 | 1.9e-4 / 4.3e-4 / 9.8e-4 / 1.5e-3 / 9.8e-4 |
| **A** ML, w = exp(-20 V) | 328 | 0 | 58 | 2e-4 / 7e-4 / 2.1e-3 / 1.2e-3 | 0.618 | 0.51 | 2.0e-4 / 6.6e-4 / 1.7e-3 / 1.3e-3 | 1.5e-4 / 3.8e-4 / 1.1e-3 / 1.6e-3 / 2.0e-3 |
| C as A, kernels at λ = inf | 128 | 0 | 184 (cold) | identical to A at the same update | 0.549 | 0.58 | 7.0e-4 / 2.2e-3 / 6.2e-3 / 9.3e-3 | 6.5e-4 / 1.6e-3 / 3.7e-3 / 8.2e-3 / 6.8e-3 |
| B (A + 4 RKL/iter, λ from init) | 128 | 65 | 160 (cold) | 5e-4 / 1.5e-3 / 3.8e-3 / 6.6e-3 | 0.549 | 0.51 | 7.2e-4 / 2.2e-3 / 6.3e-3 / 1.0e-2 | 5.8e-4 / 1.5e-3 / 4.3e-3 / 8.2e-3 / 7.8e-3 |
| **B2** A + 16 RKL/iter, lr 1e-3, λ0 = 10 | 144 | 289 | 105 | 2e-4 / 1.6e-3 / 3.7e-3 / 1.5e-3 | 0.566 | 0.47 | 1.8e-4 / 6.9e-4 / 1.4e-3 / 8.5e-4 | 1.3e-4 / 3.8e-4 / 1.1e-3 / 9.8e-4 / 9.8e-4 |
| **B4** A + 32 RKL/iter, lr 1e-3, λ0 = 10 | 176 | 705 | 132 | 0 / 3e-4 / 1e-4 / 0 | 0.570 | 0.50 | 1.1e-4 / 3.9e-4 / 4.0e-4 / 1.2e-4 | 5.1e-5 / 2.2e-4 / 4.3e-4 / 1.2e-4 / 0 |
| B3 A + 32 RKL/iter, **lr 3e-3** (cfast's) | 232 | 929 | 74 | 1e-4 / 1e-4 / 3e-4 / 0 | **0.500** | **0.99** | | |

![violation arms](images/violation_arms.png)

The plot shows L2 and L3 own raw violated fraction (floored at 1e-5 for the log axis; the dotted line is the 2e-4 budget) and line_match against ML updates.

### (a) Pricing in the kernels and exp(-20 V) weights
- **The kernels already priced violations at λ = 20.** All of them get `kernel_lam = 20`: JaxBlockGibbs `lam`, and the MH / flip `_logp`. That part needed no change.
- **The refined training states do violate, and only the own-context ones do.**
  - Probe-refined states have 0 violations at every level and every diagnostic.
  - Own-refined states keep about half their raw violations. In arm A, the fraction of own-refined states with any violation is 1.0 early and 0.75 at the end, against roughly 0.6-1.0 for the raw draws.
  - Per level, early in A: raw 4e-3 / 1.0e-2 / 3.1e-2 / 5.3e-2 / 0 becomes refined 2.8e-3 / 6.9e-3 / 2.1e-2 / 4.2e-2 / **9.2e-2**.
  - At the end of A: raw 2e-4 / 7e-4 / 2.1e-3 / 1.2e-3 / 0 becomes refined 0 / 1e-4 / 1.2e-3 / 1.5e-3 / 0.
- **The λ = 20 kernels create L4 violations.** support4 goes from 0 raw to 11 in 16 worlds after refinement. The coarse Gibbs and flip moves trade a hard support violation (cost 20) for a larger soft gain (noise4 / prior4). So at λ = 20 the kernels' target is not the hard target at L4. At λ = inf (arm C), refined L4 stays 0.
- **What the weighting actually does.** Because nearly every own-refined world violates somewhere (W = 128, about 4-10 violations per world early on), exp(-20 V) sends those weights to about 0. ESS is 3.6-5.5 of 8, which means ML trains on the probe-refined states only.
- **Arms A and C produce bit-identical own raw draws for 128 updates.** C used λ = inf kernels. The identical draws prove the own states got zero weight in both arms.
- **Conclusion for (a): weighting changed nothing.** The eval violations of A equal sweep2b's ck300 within noise. The model's violations are not inherited from violating targets: its training targets in A are all valid. **They are leakage of q's mass onto violating states, which plain ML never penalises.** No weighting of the ML targets can fix that.

### (b) Mixing in cfast's λ-controlled reverse-KL update
- **Implementation.** After each ML block, `rkl_n` calls to cfast's fused update run at τ = 1 on fresh own draws: B = 8, cone returns, critic h1, sub = 64. λ is restarted at cfast's λ0 = 10 and updated by cfast's controller (vbar / eps / eta, λ ≤ 20, eps from the measured repair ratio). The ML and RKL steps use separate Adam states.
- **Result: reverse-KL removes the violations, and at lr 1e-3 it does not kill the line-following.**
  - At equal ML-update counts, about 128-144, B2 has own raw violations 5-10x below A/C: eval raw 1.4e-3 vs 6.2e-3 at L2, and 8.5e-4 vs 9.3e-3 at L3. Line_match is the same (0.564 vs 0.556).
  - B4 (32 steps) reaches 1e-4 to 4e-4 at every level by update 150, with line_match 0.57-0.59 and dirt 0.49.
- **At cfast's own lr (3e-3) the collapse pressure wins.** B3 collapsed to the dirt mode by update 72: dirt 0.99, line_match 0.500. Violations there are about 1e-4, so it kills both the line-following and the violations.
- **So the mix is a balance.** The RKL step count times its lr sets violation pressure against collapse pressure. 32 × 1e-3 per ML iteration of 8 updates (B4) works. 32 × 3e-3 does not.

### (c) Where the 3.2e-3 comes from (raw vs final, by level and factor)

The table gives violated instances in 32 eval worlds, by home level. Count factors are homed at the children's level, `support{l}` at level l.

| model | L0 raw → final | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| ck300 | supp 112, hB 8 → supp 96, hB 3 | supp 61, hB 42, hA 3 → 41 / 16 / 0 | supp 17, hB 31, hA 4 → 18 / 11 / 3 | **supp 6**, hB 1, hA 1 → **supp 7, hB 5** | 0 → **supp 1** |
| A | 97 / 7 / 1 → 73 / 2 / 2 | 56 / 28 / 3 → 41 / 7 / 2 | 20 / 29 / 7 → 17 / 15 / 3 | supp 8, hB 3 → supp 11, hB 2 | 0 → supp 2 |
| B4 | supp 31, hB 27 → 22 / 3 (+2 hA) | 23 / 27 / 1 → 16 / 12 / 1 | 5 / 8 → 5 / 7 / 2 | 1 → 1 | 0 → 0 |

(supp = support, hA / hB = honourA / honourB.)

- **The bench's worst level, L3, is the coarsest level with many instances, and it is where repair adds violations.**
  - Raw L3 is small in count: 8-12 violations in 32 worlds, mostly `support3`.
  - Repair raises L3 (ck300: 8 → 12; bench: 2.9e-3 → 3.2e-3) and creates `support4` violations at L4 from 0. Repair never increases the total, but it moves violations upward by redrawing coarse cells.
  - **Small-count effect.** L3 has only 256 hard instances per W = 128 world (4096 in the 16 bench worlds). A single violation in the 16 worlds is 2.4e-4, already over budget. At L4, one violation is 2e-3. The 3.2e-3 at L3 is 13 violations in 16 worlds.
- **Factor type.**
  - The lookahead mask keeps every single draw inside the least-violating set. Violations therefore appear only where all values of the cell violate something.
  - **`support` (B(p) ≤ A(p below), same level) dominates at L0, L1 and L3.** **`honourB` (children's occupied columns ≤ parent's B) dominates at L2.**
  - Both are "the parent certificate cannot be realised by the children given their neighbours":
    - a cell's cert is squeezed between its parent's honour counts, which its siblings may already have used up, and support against the row below, which belongs to a different parent;
    - so the coarse levels put mass on cert combinations of adjacent parents that the next level cannot realise jointly.
  - **This is the ML-trained coarse levels putting mass on certs the fine levels cannot realise.** It shrinks as the coarse levels get reverse-KL pressure: B4 raw L3 is 1 violation in 32 worlds, against 6-8 for the ML-only arms.

## 2. Bench (v2, 300 s, 2 seeds, shared GPU)

Runs: `python -m castlegen.bench run <v> ground --seeds 2 --train 300`. ckernel_rkl = B2 config (16 RKL / iter); ckernel_rkl32 = B4 config.

**Upper bound: probe-lift contexts, outside the frozen constitution (change request #1).**

| variant (objective) | seed | ML upd | RKL upd | final viol L0/1/2/3/4 | worst | F/cell | line_match | dirt | ρ pooled per level 0/1/2/3/4, direct [subtree] | starved % 0/1/2/3/4 |
|---|---|---|---|---|---|---|---|---|---|---|
| cfast (board, 300 s; RKL only) | | | | | 1.9e-5 | 0.774 | ~0.50 | | 0.21/-0.30/-0.74/-0.70/-0.79 | 1/76/100/100/100 |
| ckernel (sweep2b, 1 seed; ML only) | 0 | 368 | 0 | 3.5e-4/7.5e-4/1.5e-3/3.2e-3/2.0e-3 | 3.2e-3 | -0.112 | 0.609 | 0.53 | 0.38/0.82/0.86/0.86/0.58 | 0/0.5/5.4/10/3.5 |
| ckernel_rkl (ML + RKL 16) | 0 † | 64 | 129 | 1.2e-3/4.5e-3/1.5e-2/2.0e-2/1.2e-2 | | 1.184 | 0.512 | 0.72 | 0.28/0.67[0.71]/0.30/0.69/0.46[0.53] | 0/6.5/9.7/17.7/0.8 |
| ckernel_rkl (ML + RKL 16) | 1 | 520 | 1041 | 1.3e-4/2.7e-4/5.5e-4/0/0 | 2.0e-2 (seed 0) | -0.045 | 0.621 | 0.53 | 0.42/0.75[0.82]/0.62/0.83[0.70]/0.53[0.18] | 0/5.1/8.3/15.0/21.7 |
| **ckernel_rkl32 (ML + RKL 32)** | 0 | 248 | 993 | 4.2e-5/1.5e-4/**2.4e-4**/0/0 | **2.4e-4 (L2)** | -0.026 | 0.560 | 0.49 | 0.35/0.80[0.86]/0.43/0.76[0.62]/0.48[0.07] | 0/3.9/10.0/24.4/41.2 |
| **ckernel_rkl32 (ML + RKL 32)** | 1 | 192 | 769 | 1.9e-5/3.1e-5/1.8e-4/0/0 | | 0.001 | 0.585 | 0.49 | 0.32/0.80[0.87]/0.50/0.75[0.65]/0.47[-0.03] | 0/4.3/10.2/22.7/43.8 |
| ckernel_rkl48 (ML + RKL 48) | 0 | 112 | 673 | 4.2e-5/3.5e-4/4.9e-4/2.4e-4/0 | 4.9e-4 (L2) | 0.036 | 0.558 | 0.49 | 0.28/0.81[0.87]/0.25/0.52[0.24]/0.46[0.06] | 0.2/5.2/12.4/27.1/43.5 |
| ckernel_rkl48 (ML + RKL 48) | 1 | 424 | 2545 | 5.0e-5/1.5e-4/2.4e-4/2.4e-4/0 | | -0.053 | 0.583 | 0.48 | 0.38/0.76[0.83]/0.66/0.76[0.62]/0.43[-0.01] | 0/3.6/10.0/23.9/41.5 |

† ckernel_rkl seed 0 spent 254 s of its 300 s in setup, a cold compile inside the bench process; its seed 1 took 63 s. The bench process ends up with a different JAX compile-cache directory than the runner scripts because of import order, as sweep2b also found. Seed 0 is not a measurement.

- **ckernel_rkl32: not accepted, by one violation.**
  - The worst final value is 2.4e-4 at L2 on seed 0: 4 of 16 384 instances, against a 2e-4 budget. Seed 1 is within budget at every level (worst 1.8e-4).
  - F is -0.012 per cell (cfast 0.774). line_match is 0.56-0.58 (ckernel ML-only: 0.61). Across-seed stats agree, raw vs final agree, and finite support holds.
  - The old v1 probe flag fires on both seeds. Per the box rules it is uninformative, but it is still part of the verdict.
- **ckernel_rkl48 is worse: not accepted.**
  - The worst final value is 4.9e-4 at L2.
  - Across-seed stats fail. Seed 0 made only 112 ML updates under contention, against 424 for seed 1.
  - More RKL did not buy the last violations. The residual is 1-4 violations per level in 16 worlds, and the run-to-run noise is of the same size.
- **The violation pressure starves the top levels in the v2 check.** Starvation at L3 / L4 is 23-24 % / 41-44 % (ML-only ckernel: 10 / 3.5 %), and L4 subtree ρ falls to about 0. Levels 1-2 are unchanged (ρ 0.80 [0.86], starvation 4 / 10 %).
- **Updates achieved:** about 190-250 ML and 770-990 RKL per seed. The GPU was shared, and the arms ran two at a time.

## What failed
- **Weighting the ML targets (a) does nothing for violations,** since all the training targets that get weight are already valid. It also silently turns ckernel into probe-only ML (own states get zero weight).
- **λ = 20 kernels add L4 violations to own states.**
- **RKL at cfast's lr collapses the line (B3).**
- **No accepted row.** rkl32 misses by one L2 violation on one seed, and pays for it with coarse-level starvation (L3 / L4).
- **Compile noise.** Cold compiles cost 105-254 s of the 300 s in several runs, so the update counts are not comparable between arms.

## Next
- **Separate the objectives by level.** Apply RKL to levels 2-4 only at a lower lr, and keep ML with probe contexts at levels 0-2, to get violations down without starving the L3 / L4 probes.
- **Run λ = inf at L4 in the coarse kernels.** The λ = 20 kernels create support4 violations.
- **Pre-compile the shapes in the bench process,** or align its cache dir, so that seed 0 gets its full budget.

## Files
- castlegen/variants/ckernel.py (edited):
  - cfg `ml_wlam`, `rkl_n`, `rkl_lr`, `rkl_lam0`, `diag_every`;
  - λ = inf kernels;
  - per-level / per-factor violation diagnostics;
  - variants ckernel_w20, ckernel_rkl, ckernel_rkl32, ckernel_rkl48, ckernel_inf (bench rows: ckernel_rkl 20261014-085528, ckernel_rkl32 20261014-090151, ckernel_rkl48 20261014-091425);
  - defaults keep sweep2b's training; the only change to plain `ckernel` is the per-level diagnostic every 2nd iteration (diag_every=2), about 0.3 s each.
- scripts/: run_ck.py (sweep2b's runner), eval_viol.py, factor_table.py, show.py, bench_row.py, plot_arms.py, env.sh, ckernel_sweep2b_orig.py (the pre-edit copy).
- runs/: armA_w20, armB_rkl, armB2_rkl16, armB3_rkl32, armB4_rkl32lr1, armC_inf (loop.jsonl, params_ground.pkl, logs); eval_*.json; bench_*.log.
- images/violation_arms.png; images/bench/ (bench images).

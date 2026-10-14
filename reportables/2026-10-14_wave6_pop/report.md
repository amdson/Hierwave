# wave6-pop: L1-L3 pressure arms and a 4-seed bench of ckpop_hard_rkl

**Own draws only, inside the frozen constitution** (amendment 2026-10-14). Problem: ground. Agent wave6-pop (formerly wave5-pop), cores 40-47. The GPU was shared: during both arms, wave4-bench's 4-seed bench was running on it, and other agents' jobs were running during this bench.

**Objective.** Weighted ML (weights 1, all levels) on a persistent population of own τ = 1 draws. The population is refined by p*-invariant kernels: tempered transitions at L4/L3, MTM at L2, fine Gibbs, 20 colour-Gibbs sweeps at λ = 20, and 2 hard sweeps. This is mixed with cfast's λ-controlled reverse-KL update on levels 0-3: 32 steps per iteration, lr 1e-3, λ ≤ 20. Every run used wave5's per-process compile memo and the warm-up wrapper (scripts/bench_warm.py; the bench harness is not edited).

## 1. L1-L3 pressure arms

Each arm: one seed, 300 s, evaluated with the bench's own `eval_seed` (16 worlds, v2 block check; nothing written to bench.jsonl). The two arms ran back to back.

| arm | updates ML / RKL | final violated instances L0/L1/L2/L3/L4 | pooled L1-L3 rate | F | line_match | starved % L3 / L4 |
|---|---|---|---|---|---|---|
| C `ckpop_rkl123x2` (reverse-KL step x2 at L1-L3) | 536 / 2145 | 17/18/11/5/1 | 34/86016 = 4.0e-4 | -0.050 | **0.571** | **30.3** / 25.0 |
| D `ckpop_rkl123x2_cg40` (C + 40 colour-Gibbs sweeps) | 488 / 1953 | 10/10/8/3/2 | 21/86016 = 2.4e-4 | -0.024 | **0.562** | **29.7** / 20.8 |
| ref `ckpop_hard_rkl`, wave5 bench seed 0 | 912 / 3649 | 11/6/5/1/0 | 12/86016 = 1.4e-4 | -0.090 | 0.608 | 22.8 / 23.1 |

**Rule:** lowest pooled L1-L3 violation rate among arms with line_match ≥ 0.58 and L3/L4 starvation no worse than 23 / 34 %.
- Neither arm qualifies. Both have line_match below 0.58 and L3 starvation of about 30 %.
- Both arms are also worse than the reference on L1-L3 violations, with **caveats**:
  - Each arm got only 488-536 updates against the reference's 912. wave4-bench's bench shared the GPU throughout the arms (this was not avoidable in the 70 min).
  - The 40-sweep arm pays for its sweeps in updates.
- The bench therefore stays with `ckpop_hard_rkl`. Curves: images/arms_curves.png.

## 2. Bench: `ckpop_hard_rkl ground --seeds 4 --train 300`, warmed

Run 20261014-114022_ckpop_hard_rkl_ground_warm (bench v2). I started it after wave4-bench's bench had finished, so the two benches did not run at the same time. The warm-up took 124 s, outside the bench clock, in the same process.

Violation columns give violated instances in 16 worlds, out of 262144 / 65536 / 16384 / 4096 / 512 at L0 / L1 / L2 / L3 / L4. The budget of 2e-4 allows at most 52 / 13 / 3 / 0 / 0.

| seed | setup | updates ML / RKL | viol L0/L1/L2/L3/L4 | worst frac | pass viol | F/cell | line_match | roughness | ρ pool L0-L4 | starved % L0-L4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.1 s | 768 / 3073 | 6/3/**5**/**1**/0 | 3.1e-4 | fail | -0.103 | 0.629 | 1.13 | .33/.75/.58/.48/-.20 | 0/2.8/8.5/22.1/37.5 |
| 1 | 2.7 s | 696 / 2785 | 7/8/**4**/0/0 | 2.4e-4 | fail (by 1 instance) | -0.087 | 0.619 | 1.14 | .38/.82/.73/.25/-.21 | 0/2.6/9.7/26.4/30.7 |
| 2 | 2.7 s | 768 / 3073 | 35/10/3/**2**/0 | 4.9e-4 | fail | -0.072 | 0.598 | 1.93 | .39/.76/.72/.57/-.19 | 0/4.2/9.9/17.9/28.4 |
| 3 | 1.2 s | 904 / 3617 | 10/8/**11**/0/0 | 6.7e-4 | fail | -0.099 | 0.605 | 1.23 | .36/.73/.57/.47/-.14 | 0/2.9/9.8/21.4/33.9 |

Pooled final rates over the 4 seeds, with Clopper-Pearson 95 % intervals:

| level | rate | 95 % CI | against the 2e-4 budget |
|---|---|---|---|
| L0 | 58/1048576 = 5.5e-5 | [4.2e-5, 7.2e-5] | under |
| L1 | 29/262144 = 1.1e-4 | [7.4e-5, 1.6e-4] | under |
| **L2** | **23/65536 = 3.5e-4** | **[2.2e-4, 5.3e-4]** | **over** |
| L3 | 3/16384 = 1.8e-4 | [3.8e-5, 5.4e-4] | straddles |
| L4 | 0/2048 | [0, 1.8e-3] | straddles |

**Verdict: NOT accepted. 0 of 4 seeds pass the violation check.**

| check | result |
|---|---|
| final violations | **fail.** Worst 6.7e-4 (seed 3, L2, 3.4x budget). Seed 1 is the closest, at 2.4e-4 (1.2x): 4 L2 instances where 3 are allowed. |
| across seeds | **fail** on roughness. Seed 2 has 1.93 against 1.13-1.23 for the others. dirt, line_match, floating and fill_err agree. |
| probes (v1 flag) | **fail.** Seeds 0 and 1 are flagged. Seeds 2 and 3 are None: only 1 of 16 raw worlds was valid, and the check needs at least 2. |
| raw vs final | pass |
| finite support | pass |
| within budget | pass (300-304 s; setup 1.1-2.7 s on every seed) |
| F | mean -0.090 per cell |

**Reading.**
- The compile fix holds on all 4 seeds: setup was 1.1-2.7 s, and updates were 696-904 on a shared GPU.
- **The binding failure is L2.** Its pooled rate of 3.5e-4 is over budget with the whole 95 % interval above 2e-4. L1 is under budget. L3 and L4 are compatible with the budget given 16 worlds per seed.
- Each seed would have to cut its L2 rate by about 2x to pass.
- Roughness disagreement is driven by one seed out of four (seed 2, 1.93), as was wave5's seed 1 (1.89). Roughness looks bimodal across seeds rather than noisy.
- L3/L4 starvation is 18-26 % at L3 and 28-38 % at L4 on every seed, above the 20 % target, and ρ at L4 is about -0.2. The coarse levels never see target-like states inside the rules.
- What I would want, outside the frozen rules: change requests #2 (penalty-only violation term) and #3 (violation price on the repaired output).

## Files
- castlegen/variants/ckernel_own.py: new variants `ckpop_rkl123x2` and `ckpop_rkl123x2_cg40`.
- reportables/2026-10-14_wave6_pop/
  - report.md
  - images/: arms_curves.png, C_eval_s0.png, D_eval_s0.png, bench/ckpop_hard_rkl_ground_s{0..3}.png
  - scripts/: arm_eval.py, bench_warm.py, summarize.py, rho.py, plot_arms.py, pool.py
  - runs/: C_rkl123x2/, D_rkl123x2_cg40/, *.log, bench.log, bench_run/, bench_row.json, pool.txt

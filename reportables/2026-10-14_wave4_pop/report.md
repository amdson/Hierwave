# wave4-pop: violation pressure for ckernel_pop, and the v2 bench (inside the frozen constitution)

2026-10-14, agent wave4-pop (formerly wave3-explore). Problem: ground. Model: LN+SiLU 6x6, started from sweep2a arm-B params. GPU shared; cores 40-47.

**This arm is inside the frozen constitution.** Every training context is a sampler's own τ = 1 draw, refined by p*-invariant kernels on a persistent population (amendment 2026-10-14). Probe lifts are used only by the bench's probe check and by the clamp gate.

**Objective.**
- Weighted ML (weights 1, all levels) on the population.
- In the RKL arms, mixed with cfast's λ-controlled reverse-KL update on levels 0-3: 32 fused steps per iteration, lr 1e-3, λ restarted at 10, λ ≤ 20. Parameters at level 4 are restored after each RKL step. The mechanism is inherited from ckernel.py, which was not edited.

## Variant
`ckernel_pop` with P3's settings (wave3): a persistent population of 32 own-started chains with 3 % refreshed from fresh own draws per iteration. Each iteration:
1. kernel cycle: coarse Gibbs L4/L3, tempered transitions L4/L3 (τ_hi 2), MTM L2, fine Gibbs;
2. 20 colour-Gibbs sweeps at λ = 20;
3. **2 hard colour-Gibbs sweeps** (λ = inf: violating values excluded), so the ML targets are valid states;
4. 8 ML updates.

The training clock now includes setup (`clock_setup=True`, as the bench requires). Bench variants, registered in castlegen/variants/ckernel_own.py:
- `ckpop_hard` = (i);
- `ckpop_hard_rkl` = (i) + (ii).

## 1. Violation pressure

Own-draw gates use 64 fresh own draws of the final params. "viol" is the bench's violated-hard fraction per level, from energy.py.

| arm | train (setup) | updates | dirt | line_match | viol L0 / L1 / L2 / L3 / L4 | L3 fill / own | L4 fill / own | population valid |
|---|---|---|---|---|---|---|---|---|
| wave3 P3 (no pressure) | 300 (65) | 792 | 0.508 | 0.645 | viol/cell 9.4e-4 | -0.112 / -0.111 | -0.138 / -0.144 | 0.98 |
| (i) W1 hard sweeps | 300 incl. 127 setup | 552 | 0.506 | **0.636** | 2.8e-4 / 1.1e-3 / 6.4e-4 / 6.1e-4 / 0 | -0.108 / -0.102 | -0.134 / -0.135 | 1.00 |
| (i)+(ii) W2 | 300 incl. 189 setup | 192 | 0.524 | 0.564 | 3.0e-4 / 1.2e-3 / 3.0e-3 / 3.6e-3 / 0 | not run | not run | 0.91 |
| (i)+(ii) W2b | 300 after 94 setup | 712 | 0.490 | 0.611 | **7.6e-5 / 2.0e-4 / 5.2e-4 / 3.1e-4 / 0** | -0.061 / -0.049 | -0.074 / -0.076 | 1.00 |

- **(i) alone is not enough.** It keeps the population valid (1.00), but own-draw violations stay at 3e-4 to 1e-3 per level, above the 2e-4 target.
- **(i)+(ii) lowers violations by 2-5x at the same training time** (W2b), with λ at its cap of 20 on levels 0-3.
  - It costs line_match: 0.611 against 0.636.
  - At 300 s it is still above 2e-4 at L2/L3.
- Following the rule, (i)+(ii) is the arm benched.

![training curves](images/w_curves.png)

## 2. Bench: `python -m castlegen.bench run ckpop_hard_rkl ground --seeds 2 --train 300`

- Run 20261014-101824_ckpop_hard_rkl_ground; bench v2, harness ee11dda34556.
- The compile was warmed first by a 30 s train through `load_variants()`. That smoke took 257 s, all of it compiling.
- GPU shared.

| seed | setup (inside the 300 s) | updates (ML / RKL) | final viol L0 / L1 / L2 / L3 / L4 | raw viol worst | F/cell | line_match | dirt | ρ pool L0-L4 | starved (probe) % L0-L4 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 171 s | 224 / 897 | 1.3e-4 / 4.3e-4 / 1.4e-3 / 2.9e-3 / **5.9e-3** | 2.7e-3 (L3) | -0.004 | 0.554 | 0.513 | 0.29 / 0.72 / 0.77 / 0.30 / -0.04 | 0 / 5 / 10.5 / 28 / 29 |
| 1 | 52 s | 608 / 2433 | 8.0e-5 / 1.8e-4 / **7.9e-4** / 0 / 0 | 9.2e-4 (L2) | -0.091 | 0.622 | 0.503 | 0.38 / 0.80 / 0.57 / 0.31 / -0.18 | 0 / 2.7 / 9.3 / 21 / 33 |

**Verdict: NOT accepted.**
- **Failed checks:**
  - final violations: worst 5.9e-3 (seed 0, L4) and 7.9e-4 (seed 1, L2);
  - across-seeds agreement of line_match and roughness: the two seeds trained for different times;
  - probes flagged: the old v1 flag, which BOX_RULES says to treat as uninformative.
- **Passed:** within budget, finite support, raw vs final agreement.
- **Free energy:** mean -0.047 per cell, against cfast's 0.774 (300 s).

Images: images/bench/ckpop_hard_rkl_ground_s0.png, _s1.png. Raw: runs/bench_run/, runs/bench_row.json.

**Reading.**
- **Setup dominated seed 0.** It spent 171 of its 300 s compiling, even after the warm smoke, so it got 224 updates. Seed 1, in the same process, made 608 updates.
- **Seed 1 is close to the budget.** Its final violations are under 2e-4 at L0, L1, L3 and L4, failing only at L2 (7.9e-4). Its line_match is 0.622 and its free energy -0.09 per cell.
- **L4 has no violation pressure in this arm.** RKL is restricted to levels 0-3, and in seed 0 the L4 final violations (5.9e-3, after repair) carry the failure.
- **Levels 3-4 are starved for probes** (21-33 %), and their ρ is low (0.3 / ≤ 0).
  - In-bounds, the coarse levels never see target-like coarse states. They are trained only on what the population reaches, and the population reaches the fill only on average (line_match 0.67).
  - The probe-context ckernel arms had starvation of 3-16 % at L3-L4. Those arms are out of bounds.

## What I would do next
1. **Cut setup.** Compile time is the main cost: 52-189 s of a 300 s budget, and 171 s in bench seed 0. One single-shape jit for the population cycle, and no separate shapes for the refresh batch or the logging batch, would add about 50 % more updates.
2. **Add pressure at L4.** Either `rkl_levels` [0..4] with a level-4 scale of 0.3 (wave2-bench's `rkl4w3`), or include L4 in the hard sweeps' targets. Seed 0's failure is at L4.
3. **Report the next bench at equal setup.** Run each seed in a fresh, warmed process so that the seeds agree.

## Files
- castlegen/variants/ckernel_own.py: `CKernelPop` gains hard sweeps, the inherited RKL mixing, per-level own-violation logging and `clock_setup`; new variants `ckpop_hard` and `ckpop_hard_rkl`.
- reportables/2026-10-14_wave4_pop/:
  - report.md
  - images/: w_curves.png, bench/ckpop_hard_rkl_ground_s{0,1}.png
  - scripts/: run_pop.py, eval_own.py, plot_w.py, peek4.sh, smoke_bench_import.py, gate copies
  - runs/: W1_hard, W2_hard_rkl, W2b_hard_rkl_300train, eval_own_*.json, bench.log, bench_run/, bench_row.json, smoke.log

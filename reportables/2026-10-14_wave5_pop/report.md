# wave5-pop: compile budget, L4 pressure and a warmed bench for ckpop_hard_rkl

**Own draws only, inside the frozen constitution** (amendment 2026-10-14). Problem: ground. Agent wave5-pop, cores 40-47. The GPU was shared with other agents' jobs throughout.

**Objective.** Weighted ML (weights 1, all levels) on a persistent population of own τ = 1 draws. The population is refined by p*-invariant kernels: tempered transitions at L4/L3, MTM at L2, fine Gibbs, 20 colour-Gibbs sweeps at λ = 20, and 2 hard sweeps. This is mixed with cfast's λ-controlled reverse-KL update (32 steps per iteration, lr 1e-3, λ ≤ 20). The reverse-KL step is on levels 0-3 for `ckpop_hard_rkl`, and on 0-4 with the L4 step scaled for the new arms. Probe lifts are used only by the bench's checks.

## 1. Compile budget: fixed (setup 171 s -> 1.2 s per seed)

**Diagnosis** (scripts/prof_setup.py, runs/prof1.log). I profiled through the bench's import path, in one process, as run.py does.
- The persistent cache works. It is `/workspace/jax_cache` (the effective dir after load_variants()). It produced 25 hits that saved 315 s of backend compile.
- **The setup is Python tracing, not compiling.** A cold process makes 340 compile requests. These cost:
  - 34.5 s of jaxpr tracing;
  - 15.4 s of lowering;
  - 15.3 s of cache retrieval;
  - 27 s of backend compile for the programs under the 1 s persistence threshold, which are never cached.
- The total is 85 s on a quiet GPU and 171 s under contention.
- **A separate warm-up process cannot remove tracing**, because traces are not persisted. That is why the wave4 smoke run did not help.
- **Seed 1 paid again (49-52 s), in the same process.** Every instance built new `jax.jit` wrappers (gen, fine and coarse Gibbs, MTM, tempered transitions, ML step), so it re-traced all of them.
- A 25 s stall after iteration 1 came from the logging batch (8 own draws), a shape that had not been compiled.

**Fix** (castlegen/variants/ckernel_own.py):
- The built kernels and jitted callables are memoised per process. The key is the problem, the width, the full cfg (minus the warm-up-only keys) and the init λ baked into gen. This is pure compile caching: no parameters, rng or population are shared.
- Every shape the loop uses is warmed in setup: the refresh batch, `_pop_stats` on it, the population scatter and the logging batch.
- `jax_persistent_cache_min_compile_time_secs = 0` is set during train, so the sub-second compiles persist too.
- `max_iters` was added, used only by warm-up instances.

**Wrapper** (scripts/bench_warm.py; the harness is not edited). It builds and trains one warm-up instance (seed 99, 3 iterations) through `load_variants()` in the bench process. It then calls `castlegen.bench.run.run(...)` unchanged. This is the "fresh warmed process" for every seed.

**Smoke** (`bench_warm.py smoke ckpop_hard_rkl 60 2`, runs/smoke60.log):

| | warm-up | seed 0 setup | seed 1 setup | updates in 60 s (ML / RKL) |
|---|---|---|---|---|
| wrapper | 263 s (two cold processes at once) | **1.3 s** | **1.8 s** | 136 / 545, 120 / 481 |

In the bench itself, both seeds' setup was 1.2 s and the warm-up took 134 s. Without the wrapper, the first seed of a fresh process still pays about 85 s of setup (quiet GPU), or more under contention.

## 2. L4 pressure: neither arm qualifies

New variants: `ckpop_rkl4w3` and `ckpop_rkl4w1`. Both run reverse-KL on levels 0-4, with the L4 parameter step scaled by 0.3 and 0.1 respectively. Each arm had one seed and 300 s of training, and was evaluated with the bench's own `eval_seed` (16 worlds, v2 block check; nothing written to bench.jsonl).

The two arms ran at the same time on a shared GPU, so each got only 344 ML updates. Alone, an arm gets 688 updates (runs/A_rkl4w3_try1_evalcrash: training log only, because its eval crashed on a path bug in my script).

| arm | updates ML / RKL | final viol L0 / L1 / L2 / L3 / L4 | F | line_match | ρ pool L0-L4 | starved % L0-L4 |
|---|---|---|---|---|---|---|
| A rkl4w3 (L4 x0.3) | 344 / 1377 | 6.5e-5 / 5.3e-4 / 9.2e-4 / 0 / 2.0e-3 | -0.026 | 0.568 | .36/.76/.65/.16/-.40 | 0/4.1/8.5/24.3/**81.0** |
| B rkl4w1 (L4 x0.1) | 344 / 1377 | 1.0e-4 / 2.4e-4 / 7.9e-4 / 1.7e-3 / 2.0e-3 | -0.043 | 0.582 | .34/.82/.81/.25/-.16 | 0/4.3/10.3/22.5/**34.3** |
| ref: wave4 bench (L0-3 only) | 224-608 | worst 5.9e-3 (L4) / 7.9e-4 (L2) | -0.05 | 0.55-0.62 | | L3/L4 21-33 |

- **Reverse-KL at L4 adds no violation pressure.** The pass's raw L4 violations are always 0, so cfast's adaptive λ_4 decays from 10 to 0 within 120-250 s (images/arms_curves.png). The L4 step is then pure energy plus entropy pressure, and it raises L4 starvation (81 % / 34 %).
- **The L4 failures are made by repair.** L3 picks certificates that the children cannot realise, and repair turns that into one L4 instance in 16 worlds. This alone gives 2.0e-3, because one L4 instance is 1/512.
- **Rule** (lowest worst violation among arms with L3/L4 starvation ≤ 20 % and line_match ≥ 0.58): no arm qualifies, including the reference.
  - I benched `ckpop_hard_rkl` (reverse-KL on L0-3), the closest on starvation.
  - It is the same variant as wave4, now with the compile fix, so the bench is a direct test of goal 1.
- What I would want instead is outside the frozen rules: violation pricing on the repaired output at L4. This is change request #3 in notes/constitution_change_requests.md.

## 3. Bench: `ckpop_hard_rkl ground --seeds 2 --train 300`, warmed

Run 20261014-110428_ckpop_hard_rkl_ground_warm (bench v2), launched through scripts/bench_warm.py. GPU shared.

| seed | setup | updates ML / RKL | final viol L0 / L1 / L2 / L3 / L4 | raw worst | F/cell | line_match | dirt | roughness | ρ pool L0-L4 | starved % L0-L4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.2 s | 912 / 3649 | 4.2e-5 / 9.2e-5 / **3.1e-4** / **2.4e-4** / 0 | 7.3e-4 | -0.090 | 0.608 | 0.505 | 1.07 ± 0.07 | .37/.70/.60/.37/-.06 | 0/5.0/7.7/22.8/23.1 |
| 1 | 1.2 s | 904 / 3617 | 1.6e-4 / **2.4e-4** / **5.5e-4** / **9.8e-4** / 0 | 8.5e-4 | -0.094 | 0.633 | 0.501 | 1.89 ± 0.06 | .43/.82/.68/.59/-.20 | 0/3.8/9.5/19.1/34.1 |

For scale, the seed 0 misses are 5 instances at L2 (out of 16384) and 1 at L3 (out of 4096). The seed 1 misses are 16, 9 and 4 instances at L1, L2 and L3.

**Verdict: NOT accepted.**

| check | result |
|---|---|
| final violations < 2e-4 | **fail.** Worst 9.8e-4 (seed 1, L3, 4.9x budget). Seed 0's worst is 3.1e-4 (L2, 1.5x). |
| raw vs final | pass |
| across seeds | **fail** on roughness (1.07 vs 1.89). dirt, line_match, floating and fill_err agree. |
| probes (v1 flag) | **fail.** Seed 0 flagged. Seed 1 is None, because it had 0 valid raw worlds of 16 and the check needs at least 2. BOX_RULES calls the v1 flag uninformative, but the verdict still requires it. |
| finite support | pass (0 mismatch) |
| within budget | pass (302 s) |
| F | -0.092 per cell (cfast at 300 s: 0.774) |

**Against wave4's bench of the same variant:**
- Setup is 1.2 / 1.2 s against 171 / 52 s, and updates are 912 / 904 against 224 / 608.
- The worst violation is 9.8e-4 against 5.9e-3. The L4 violations are now 0 on both seeds, so the L4 failure was undertraining caused by compile time.
- line_match now agrees across seeds (0.61 / 0.63).
- L3/L4 starvation is unchanged at 19-34 %.

This is the closest arm. The final-violation check fails by 1.5x (seed 0) to 4.9x (seed 1). The structural failure is that own raw worlds are rarely valid as whole worlds (5 and 0 of 16). That blocks the probe check and needs the violation rate per instance at L1-L3 to fall another 5-10x.

Images: images/bench/ckpop_hard_rkl_ground_s{0,1}.png, images/arms_curves.png, images/{A_rkl4w3,B_rkl4w1}_eval_s0.png.

## Files
- castlegen/variants/ckernel_own.py: per-process build memo, warm-up shapes, `cache_all`, `max_iters`; new variants `ckpop_rkl4w3` and `ckpop_rkl4w1`. The pre-wave5 copy is scripts/ckernel_own.before_wave5.py.
- scripts/: prof_setup.py, bench_warm.py (bench wrapper), arm_eval.py, summarize.py, rho.py, plot_arms.py.
- runs/: prof1.log, smoke60.log, A_rkl4w3/, B_rkl4w1/, A_rkl4w3_try1_evalcrash/, *.log, bench.log, bench_run/ (train logs), bench_row.json.
- notes/constitution_change_requests.md: #3 (violation price on the repaired output at the coarsest level).

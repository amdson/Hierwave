# crevise: revising (diffusion-style) generation trained by the path KL (ideas #19, Amendment 2026-10-14 (b))

2026-10-14 evening · agent crevise · **PAUSED by the coordinator's order at ~22:15 UTC with two runs still going (see "Running now").** Box: 2x RTX A4000, GPU 0 only, cores 16-23 (GPU) / 0-7 (CPU tests). GPU clocks recorded before / after every chain (`runs/gpu_*.txt`): 1560-1920 MHz SM, 39-66 C, no throttling. Objective: **path KL** for crevise (every log entry carries `objective: path_kl`), reverse KL for the cfast controls.

## Answer (numbers first)

- **Tests:** the 9 fast tests pass (R = 0 = cfast **bit for bit**: same draws, costs, violations, identical parameters after a fused update; cost identity on toy_support / ground / dimers to 2e-3; cone containment on 3 problems; overlapping open windows agree). Gradient check (first run, unguided, near-deterministic random policy): mean estimator gradient = exact path-KL gradient by trajectory enumeration with zero and wrong critics (proj_z −0.95 / −0.93, Hotelling p 0.64 / 0.65, cost mean = J), but the first negative control (returns without the revising costs) was **not** rejected (p 0.56) because that policy had H(traj) = 0.03 nats: no power. The test was changed to a soft policy (randomize scale 0.3), guided features on, plus a second control (no returns at all); the two-level test crashed on a shape bug in the gap helper, fixed. **The re-run of the full file is in progress** (`runs/test_all.log`, 3 passes so far).
- **Bound gap (enumerable flat problem, τ = 1, λ = 20, `runs/tiny_gap.json`):** path KL − exact reverse KL of x_0 = 3.1-4.0 nats per world for the untrained policy (|M| log D = 4.2-5.5 minus H(old | x_0) 1.1-1.5); exactly |M| log D for a near-deterministic policy. Never negative; the per-trajectory cost identity holds to 1e-5.
- **Dimers, unguided (R 2, ρ 0.5, 300 s, 23 075 updates): FAILED.** Trajectory entropy 1.19 → 0.33 nats/cell by update 25, 0.00 by update 4 500; final h/v/d 0.041 / 0.443 / 0.516, raw violations **5.6e-3** (final 4.7e-3, budget 2e-4), λ at the cap. cfast control (180 s, 18 373 updates): 0.489 / 0.511 / 0.0004, raw 1.2e-4, final 9.3e-5, entropy 0.226. Revisions did not stop the collapse and left 30x cfast's violations.
- **Ground patch (8 x 16, T = 2, MCMC reference dirt 0.641, soft 33.0):** cfast (180 s, ~9 000 updates) dirt **0.651**, soft 32.5, line match 0.89, L1 of the P(dirt) map to the reference **0.051**, final violations 1.5e-5 / 1.2e-4 / 0: close to the reference on this small patch. Unguided crevise (300 s, 7 265 updates): dirt **0.936**, soft 63.7, L1 0.295, final violations 9e-5 / 1.8e-4 / 0: the all-dirt collapse, much worse than cfast.
- **Guided crevise (violation-field features):** built and smoke-tested (ground, open window); dimers and ground-patch runs are **running now**, results not yet in.
- **Bench:** not run (paused before step 3; both prototypes that finished fail step 1).

## Running now (read these when they finish)

| job | command (from the repo root, `scripts/env_gpu.sh` sourced, `taskset -c 16-23`) | log / output | expected end | how to read |
| --- | --- | --- | --- | --- |
| guided dimers | `python scripts/dimers_run.py crevise 300 25 4 runs/dimers_crevise_guided.json` (chain `scripts/guided_chain.sh`, PID in `runs/guided_chain.pid`) | `runs/dimers_crevise_guided.log`, json | ~22:25 UTC | last `FINAL {...}` line: `raw` = h/v/d, `raw_viol`, `final_viol` (budget 2e-4), `logq_traj` (−H(traj)/cell), `F_ub`; compare the unguided row above and cfast |
| guided ground patch | `python scripts/ground_patch_train.py crevise 300 256 runs/ground_patch_crevise_guided.json` (same chain, starts after dimers) | `runs/ground_patch_crevise_guided.log`, json | ~22:35 UTC | lines `raw {...}` / `final {...}`: dirt (reference 0.641), soft (33.0), `L1_to_mcmc`, `viol_frac` per level; `CHAIN_DONE` marks the end |
| test re-run | `JAX_PLATFORMS=cpu taskset -c 0-7 python -m pytest tests/variants/test_crevise.py -q -s` (PID `runs/test_all.pid`) | `runs/test_all.log` | ~22:30 UTC | final pytest summary line; the `zero` / `wrong` / neg-control dicts print proj_z, Hotelling p |

`python scripts/plot.py` regenerates `images/dimers_curves.png` and `images/ground_patch_pdirt.png` (add the guided jsons to its file lists to include them).

## What was built

Files: `castlegen/variants/_crevise_core.py` (schedule, pass with revising steps, violation field, cones `RCone`, margin), `castlegen/variants/crevise.py` (variants `crevise` = R 2, ρ 0.5, guided; `crevise_noguide`, `crevise_r0`, `crevise_r1`, `crevise_sep_noguide`), `castlegen/variants/_crevise_gradcheck.py` (trajectory enumeration with fixed erase coins, exact path KL + gradient, exact reverse KL of x_0), `tests/variants/test_crevise.py` (15 tests), `scripts/` (env, dimers probe, ground patch + MCMC reference, plots, chains), `runs/`, `images/`.

**Generation.** Levels T..0. At level l: cfast's coloured pass, then R revising steps. Step m: (1) the **declared noising r**: every (sampled channel, site) of level l is erased independently with probability ρ = 0.5 by a coin hashed from (seed, "revise:<channel>#m", level, Y, X): coordinate-keyed, fixed, not learned, independent of the state (the guided-step agent's selection rule did not exist yet; a fixed rate keeps r trivially declared); (2) erased cells are unknown until redrawn; the 16 colours are swept again and the erased cells are redrawn given everything known at that moment with the same least-violating mask and lookahead, their own Gumbel noise ("<channel>#rev<m>") and their own actor. Then one repair round. Finite support: `margin(problem, R, guide)` = Σ_l (16·3 + R·16·(6 guided / 3) + 16·2 + 8)·2^l fine cells (ground guided default 5 760, unguided 3 328).

**Guided revising steps** (coordinator's addition, ideas #21, allowed freedom; cfg `guide`, default on). Before the step erases anything, every (sampled channel, cell) of level l and l + 1 gets the number of violated hard instances touching it in the current state (nv of the current value from the local conditional). The revising actor reads cfast's features plus this field over the actor's same-level 4 x 4 and parent 3 x 3 windows. **Stand-in:** `castlegen/graded.py` exists (numpy, per instance) but has no per-cell JAX `graded_field`, so the field is the binary count; the swap is one function (`violation_field`). The field is fixed for the step; `RCone` widens the guided steps' reach by the window cells' factor scopes.

**Objective: path KL as per-draw costs.** P = p*(x_0) Π r(x_k | x_{k−1}) with r undoing the revising steps backwards (each cell erased with probability ρ, refilled **uniformly** over D values) and then erasing the level; the mask coins cancel between Q and P, so KL(Q ‖ P) + log Z = E_Q[S(x_0) + Σ λ_l H_l(x_0) + τ log Q(traj) + τ Σ_erased log D]. Costs: pass draw soft(a) + λ nv(a) + τ log q(a) (cfast's); redraw o → a: [soft_h(a) − soft_h(o)] + λ[nv_h(a) − nv_h(o)] + τ(log q(a) + log D), closed terms in the hybrid state (earlier colours of the step at new values, this and later colours at old values), which telescope to E(x^{m+1}) − E(x^m). log D carries no gradient. R = 0 is exactly cfast. Training: cfast's fused update (cone returns over the pass + revisions step list, all-action + score-function terms, h1 critic per (level, channel, step), subsampled loss, λ ≤ 20, τ 2 → 1). `score()` is None for R > 0; `info()["bound"]` gives F_ub = S + log Q(traj) + |M| log D per cell (upper bound on F). Gap = Σ_erased log D − H(old | x_0): up to 2 ρ R log 6 = 1.79 nats/cell on dimers, so the objective rewards trajectory entropy, not only x_0's.

## Dimers (one seed, 4 held-out raw worlds every 25 updates, final on 16 worlds)

| run | updates | final h / v / d (raw) | raw viol | final viol | entropy / cell |
| --- | --- | --- | --- | --- | --- |
| crevise_noguide (R 2, ρ 0.5, 300 s) | 23 075 | 0.041 / 0.443 / 0.516 | 5.6e-3 | 4.7e-3 | H(traj) 0.000 (F_ub 1.792 = |M| log D) |
| crevise guided (300 s) | running | | | | |
| cfast control (180 s) | 18 373 | 0.489 / 0.511 / 0.0004 | 1.2e-4 | 9.3e-5 | 0.226 |
| cbound_lam reference | | 0.47 / 0.30 / 0.23 | 6e-5 | | 0.27 |

![dimers curves](images/dimers_curves.png)

Unguided trajectory: update 1: 0.38 / 0.32 / 0.30, viol 4.3 %, H(traj) 1.19; update 25: v 0.84, H 0.33; update 50: v 0.99, H 0.03; flips between h and v stripes at 1 % violations for 3 000 updates while λ climbs to 20; all-diagonal and deterministic at zero violations by update 4 500; λ decays to 0.8, the entropy term reopens the policy, violations return, λ back to 20, and the run freezes deterministic at 0.6 % violations for 15 000 updates. Throughput 77 updates/s vs cfast 102/s.

## Ground patch (8 x 16, T = 2; `scripts/ground_patch.py`: the MCMC report's patch tiled to width 16, same seams)

Reference: annealed-importance sampler, 540 s one core, 56 runs all valid and distinct, ESS 12.2, weighted dirt 0.641, soft 33.0 (`runs/ground_patch_mcmc.json`). 256 held-out worlds per variant.

| run | updates | dirt | soft | line match | raw viol L0 / L1 / L2 | final viol | L1 P(dirt) to ref |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reference | | 0.641 | 33.0 | | 0 | 0 | 0 |
| cfast (180 s) | ~9 000 | 0.651 | 32.5 | 0.892 | 9e-5 / 5.5e-4 / 0 | 1.5e-5 / 1.2e-4 / 0 | 0.051 |
| crevise_noguide (300 s) | 7 265 | 0.936 | 63.7 | 0.751 | 2.6e-4 / 0 / 0 | 9e-5 / 1.8e-4 / 0 | 0.295 |
| crevise guided (300 s) | running | | | | | | |

![ground patch P(dirt)](images/ground_patch_pdirt.png)

## What failed, plainly

1. Unguided crevise on dimers: deterministic, 0.6 % violations (30x budget), two orientations; worse than cfast.
2. Unguided crevise on the ground patch: all dirt (0.94 vs 0.64) where cfast matches the reference to 0.05.
3. The bound is loose by construction with uniform r (ρ R log D per cell); score() is None for R > 0, so the bench cannot compute F or the probes (fails acceptance on the board regardless).
4. First negative control had no power (deterministic test policy); fixed, re-run pending.
5. Bench not run; guided results pending.

## Proposals (not built; constraint cited)

- **Designed fixed r** (refill ∝ exp(−s − λ_fixed nv) over all D values, generic from the tables): tightens the bound where the target's conditionals are peaked; a different declared r (amendment (b)).
- **Field-dependent erase rate** ρ(field): a fixed declared function of the state, allowed by the coordinator's note; the mask terms then no longer cancel and add a log-ratio cost per step; not built (no selection rule available yet).
- An entropy floor or trajectory-level τ changes the objective (constraint 4); noted only.

## Next

Read the two guided runs and the test re-run (table above). If guided dimers keeps three orientations within budget, bench it (`python -m castlegen.bench run crevise dimers --seeds 2 --train 300`); otherwise try the designed fixed r (proposal 1) on the tiny problem first (gap) and then dimers, and port a per-cell graded distance to JAX to replace the binary stand-in.

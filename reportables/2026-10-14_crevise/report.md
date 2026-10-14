# crevise: revising (diffusion-style) generation trained by the path KL (ideas #19, Amendment 2026-10-14 (b))

2026-10-14 evening · agent crevise · box: 2x RTX A4000, GPU 0 only, cores 16-23 (GPU) / 0-7 (CPU tests); GPU clocks recorded before / after every chain (`runs/gpu_*.txt`): 1560-1920 MHz SM under load, 39-66 C, never throttled. Objective: **path KL** for every crevise run (each training log entry carries `objective: path_kl`), reverse KL for the cfast controls. One seed per run.

## Answer (numbers first)

- **Tests: 18 / 18 pass** after a fix (`tests/variants/test_crevise.py`, CPU). R = 0 reproduces cfast **bit for bit** (same draws, costs, violations; identical parameters after a fused update). Cost identity on toy_support / ground / dimers: summed draw costs = S(x_0) + λ·H(x_0) + τ(log Q(traj) + |M| log D) to 2e-3. Gradient check against the exact path-KL gradient by trajectory enumeration (flat tiny problem, 10 502 trajectories, 6 erased cells, guided, soft policy, 300 batches x 16 worlds): pass actor proj_z 0.34 / −0.20 (zero / wrong critic), Hotelling p 0.10 / 0.09; **revising actor proj_z 0.28 / −0.18, p 0.19 / 0.12, proj_ratio 1.03 / 0.91**; negative controls rejected (returns without the revising costs: p 0.0, proj_z −21 and 5.8; no returns: p 0.0, proj_z −124 and 12). Two-level problem, one erased coarse cell (119 814 trajectories): both actors unbiased (proj_z −0.83 / −0.75, p 0.23 / 0.11). Finite support: a replaced draw changes nothing outside its cone (4 cases, guided and unguided); overlapping open windows agree exactly (3 cases, raw and final).
- **A bias was found and fixed in this session:** the first version's all-action term used the closed energy terms of the state the redraw conditions on, while the actual redraw cost uses the hybrid-state terms (other erased cells at their old values); the gradient check on the guided flat problem caught it (revising actor proj_ratio 1.43, p 3e-15), and the first negative control, run with a near-deterministic test policy, had had no power. **All runs marked "biased term" below were trained with that estimator**; only the last dimers run uses the corrected one.
- **Bound gap (enumerable flat problem, τ = 1, λ = 20):** path KL − exact reverse KL of x_0 = 3.1-4.1 nats per world (16 cells, 6-8 erased) for every policy looked at: the |M| log D charge (4.2-5.5) minus what the generator's posterior over the old values keeps random (H(old | x_0) 1.1-1.5 for the untrained policy, 0.1 for the gradient-test policy, 0 for a deterministic one). Two-level problem with one erased coarse cell: gap 0.44 of log 3 = 1.10. Never negative; per-trajectory identity to 1e-5.
- **Dimers (h / v / d, violations per hard instance, 300 s):** unguided, biased term: 0.041 / 0.443 / 0.516, raw **5.6e-3**, deterministic (H(traj) = 0). Guided, biased term: 0.793 / 0.170 / 0.037, raw **2.4e-2**, deterministic. **Guided, fixed estimator: 0.000 / 0.000 / 1.000, raw 3.1e-5 (budget met), deterministic** (1 975 updates: this run was 10x slower per update, see below). cfast control: 0.489 / 0.511 / 0.0004, raw 1.2e-4, entropy 0.226. cbound_lam reference 0.47 / 0.30 / 0.23. **Every crevise configuration collapsed to one or two orientations and to a deterministic trajectory**; the collapse happens as the violations fall (cdim's curve), at update 25-650.
- **Ground patch (8 x 16, T = 2; MCMC reference dirt 0.641, soft 33.0):** cfast dirt **0.651**, soft 32.5, line match 0.89, L1 of the P(dirt) map 0.051 (close to the reference). Unguided crevise (biased term): dirt **0.936**, soft 63.7, L1 0.295. Guided crevise (biased term): dirt **0.941**, soft 75.0, L1 0.300. Both crevise runs are the all-dirt collapse; not re-run with the fixed estimator (GPU time cap).
- **Bench:** not run; no configuration passed step 1 (dimers pre-check).
- **Verdict: negative.** The path-KL revising sampler, with a fixed-rate erase r and uniform refill, is more mode-seeking than cfast, not less: the trajectory entropy goes to zero and generation becomes a deterministic function of the coins. Proposal for a designed fixed r below.

## What was built

Files: `castlegen/variants/_crevise_core.py` (schedule, pass with revising steps, violation field, `RCone`, margin), `castlegen/variants/crevise.py` (variants `crevise` = R 2, ρ 0.5, guided; `crevise_noguide`, `crevise_r0`, `crevise_r1`, `crevise_sep_noguide`), `castlegen/variants/_crevise_gradcheck.py` (trajectory enumeration with fixed erase coins, exact path KL and gradient, exact reverse KL of x_0), `tests/variants/test_crevise.py` (18 tests), `scripts/` (env, dimers probe, ground patch + MCMC reference, plots, chains, smoke), `runs/` (logs, json, GPU clocks), `images/`.

**Generation.** Levels T..0. At level l: cfast's coloured pass (16 colours, least-violating mask with lookahead, coordinate-keyed Gumbel noise), then R = 2 revising steps. Step m: (1) **the declared noising r**: every (sampled channel, site) of level l is erased independently with probability ρ = 0.5 by a coin hashed from (seed, "revise:<channel>#m", level, Y, X): coordinate-keyed, fixed, not learned, independent of the state; (2) erased cells are unknown until redrawn; the 16 colours are swept again and the erased cells are redrawn given everything known at that moment (unerased cells, redrawn cells of earlier colours, coarser levels) with the same mask and lookahead, their own Gumbel noise ("<channel>#rev<m>") and their own actor. Then cpass's one repair round. Fixed work: 1 + R sweeps per level; throughput 77 updates/s against cfast's 102/s on dimers (launch-latency bound at B = 8). Finite support: `margin(problem, R, guide)` = Σ_l (16·3 + R·16·(6 if guided else 3) + 16·2 + 8)·2^l fine cells (ground: 5 760 guided, 3 328 unguided), verified by the cone and window tests.

**Guided revising steps** (coordinator's addition, ideas #21; allowed freedom "more information to learned parts"; cfg `guide`, default on). Before a step erases anything, every (sampled channel, cell) of level l and of level l + 1 gets the number of violated hard instances touching it in the current state (nv of the current value from the same local conditional the draw uses). The revising actor reads cfast's features plus this field over the actor's same-level 4 x 4 and parent 3 x 3 windows (zero outside the region). The field is fixed for the step, so the draw reads only values fixed before it; `RCone` widens a guided step's dependency reach by the factor scopes of its window cells. **Graded field: not swapped in.** `castlegen/graded.py` landed during this work with `graded_field(problem, state, region, level)` (numpy, host side, per instance); the pass is one jitted vmapped function, so a host field would need a `pure_callback` between revising steps (one host call per world, level and step: on ground 5 levels x 2 steps x B worlds of numpy instance scans per update, roughly the cost of the bench's energy() per world, i.e. tens of ms each and the end of the fused update's speed) and a JAX twin of the generic distance tables was out of this session's budget. The binary stand-in is what every run used (on dimers graded = binary anyway, per graded.py). The swap point is the single function `violation_field`. **Cell selection of `_guided_step.py`** (`select_cells`: every cell with a nonzero field, dilated by 1, plus descendants) was read and **not adopted** as the noising: a deterministic state-dependent erase set is not a valid fixed r for the path KL, because P's r must regenerate the earlier state from the later one and a valid x^{m+1} has an empty field, so any change Q made would have zero probability under P (infinite KL). It would be valid as a mixture ρ(cell) = ρ_0 + (1 − ρ_0)·1[field > 0, dilated] read from the later state, at the price of a log-ratio of the two mask probabilities in every revising cost; not built.

**Objective: the path KL as per-draw costs.** Q = the generator's trajectory law (pass values, erase masks, redrawn values, level by level). P = p*(x_0) Π r(x_k | x_{k−1}) with r, read from x_0 backwards, undoing the revising steps of a level in reverse order (each cell independently with probability ρ, refilled **uniformly** over its D values) and then erasing the level. The mask coins have the same law under Q and P and cancel, so KL(Q ‖ P) + log Z = E_Q[S(x_0) + Σ_l λ_l H_l(x_0) + τ log Q(traj) + τ Σ_erased log D] (τ = 1 at the end). Per-draw costs that sum to the integrand: pass draw of a: soft(a) + λ nv(a) + τ log q(a) (cfast's); redraw of an erased cell from old o to a: [soft_h(a) − soft_h(o)] + λ[nv_h(a) − nv_h(o)] + τ(log q(a) + log D), with the closed terms in the hybrid state (erased cells of earlier colours of the step at their new values, of this and later colours at their old values); colour by colour these telescope to E(x^{m+1}) − E(x^m), and the per-draw change of closed violations gives H_l(x_0) for the λ control. The all-action term uses the same hybrid terms for every candidate value (the fix). log D carries no gradient. R = 0 is cfast exactly. Training: cfast's fused update (cone returns over the pass + revisions step list, all-action + score-function terms, one h1 critic per (level, channel, step), subsampled loss, λ ≤ 20, τ 2 → 1 over 20 %, expected-draw normalisation per level). `score()` is None for R > 0 (log q(x_0) intractable): the bench's F and probe metrics would be missing. `info()["bound"]` gives F_ub = S + log Q(traj) + |M| log D per cell and its parts.

**Bound gap.** KL(Q ‖ P) = KL(q(x_0) ‖ p*) + E KL(Q(earlier | x_0) ‖ r(earlier | x_0)) = reverse KL + Σ_erased log D − H_Q(old values | x_0, masks). With uniform refilling the charge is ρ R log D per cell (dimers 1.79, tiles 0.69 nats) and only trajectory entropy earns it back; a deterministic generator pays it in full. The objective therefore rewards entropy of the old values given the final state, which the runs never achieved.

## Tests (CPU cores 0-7; `runs/test_all.log` first run, `runs/test_gradcheck_fixed.log` after the fix)

| test | what | result |
| --- | --- | --- |
| test_r0_matches_cfast | toy_support W 64, R = 0: draws, costs, violations identical; parameters after one update identical; score = cpass's | pass |
| test_cost_identity[toy_support / ground / dimers] | 3 worlds, R 1-2, ρ 0.4-0.5, guided on ground and dimers; Σ costs vs reference energy + τ(log Q + |M| log D); telescoped dv = H_l(x_0) | pass |
| test_flat_enumeration | 10 502 trajectories, Σ Q = 1 ± 1e-4 | pass |
| test_flat_unbiased[zero / wrong] | mean gradient vs exact ∇J; both actor blocks | pass after the fix (zero: pass block proj_z 0.34, p 0.10; rev block 0.28, p 0.19, ratio 1.03; wrong: −0.20 / −0.18, p 0.09 / 0.12); **failed before** (rev block ratio 1.43, p 3e-15) |
| test_flat_negative_control[rev / all] | returns without revising costs / without returns | rejected (p 0.0, proj_z −21 & 5.8; −124 & 12) |
| test_flat_bound_gap | path KL ≥ reverse KL of x_0; per-trajectory identity < 1e-3 | pass (gap 4.07, |M| log D 4.16, identity 9e-6) |
| test_two_level_unbiased | gradcheck's two-level problem, R = (0, 1), one erased coarse cell, 119 814 trajectories, 200 batches | pass (proj_z −0.83 / −0.75, p 0.23 / 0.11; gap 0.44) |
| test_changes_inside_cone[4] | replaced draw changes nothing outside its cone; replay changes nothing; coins never move | pass |
| test_open_windows_agree[3] | overlapping open windows agree, raw and final (toy_support; dimers, ground guided) | pass |

Bound gap on the flat problem (T = 0, 2 x 8 binary cells, R = 1, ρ = 0.5, τ = 1, λ = 20; `runs/tiny_gap.json`), nats per world:

| policy | coins | erased | trajectories | distinct x_0 | path KL + log Z | reverse KL(x_0) + log Z | gap | |M| log D | H(traj) | H(x_0) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| untrained (b = 0) | 1 | 7 | 12 531 | 1 508 | 6.853 | 3.291 | 3.562 | 4.852 | 6.69 | 5.40 |
| untrained (b = 0) | 2 | 8 | 27 370 | 1 155 | 5.612 | 1.567 | 4.044 | 5.545 | 6.48 | 4.98 |
| untrained (b = 0) | 3 | 6 | 10 502 | 998 | 3.276 | 0.151 | 3.124 | 4.159 | 5.84 | 4.81 |
| random, near-deterministic | 1 | 7 | 12 531 | 1 508 | 2.423 | −2.429 | 4.852 | 4.852 | 0.03 | 0.03 |
| random, near-deterministic | 3 | 6 | 10 502 | 998 | 1.780 | −2.379 | 4.159 | 4.159 | 0.03 | 0.03 |
| gradient-test policy (scale 0.3, guided) | 3 | 6 | 10 502 | 998 | 0.461 | −3.611 | 4.072 | 4.159 | | |

## Dimers (one seed; 4 held-out raw worlds every 25 updates, final on 16 worlds; `scripts/dimers_run.py`)

| run | estimator | updates (300 s) | final h / v / d (raw) | raw viol | final viol | H(traj) or entropy / cell |
| --- | --- | --- | --- | --- | --- | --- |
| crevise_noguide (R 2, ρ 0.5) | biased term | 23 075 | 0.041 / 0.443 / 0.516 | 5.6e-3 | 4.7e-3 | 0.000 |
| crevise guided | biased term | 22 430 | 0.793 / 0.170 / 0.037 | 2.4e-2 | 2.1e-2 | 0.000 |
| crevise guided | **fixed** | 1 975 | 0.000 / 0.000 / 1.000 | 3.1e-5 | 3.1e-5 | 0.002 |
| cfast control (180 s) | | 18 373 | 0.489 / 0.511 / 0.0004 | 1.2e-4 | 9.3e-5 | 0.226 |
| cbound_lam reference | | | 0.47 / 0.30 / 0.23 | 6e-5 | | 0.27 |

![dimers curves](images/dimers_curves.png)

- Unguided (`runs/dimers_crevise.log`): H(traj) 1.19 → 0.33 by update 25, 0.03 by update 50; h/v stripe flipping at 1 % violations for 3 000 updates while λ climbs to 20; all-diagonal and deterministic at zero violations by update 4 500; λ decays to 0.8, the entropy term reopens the policy, violations return, λ back to 20 and the run freezes deterministic at 0.6 % violations for 15 000 updates (zero gradient through the saturated softmax).
- Guided, biased term (`runs/dimers_crevise_guided.log`): the field features keep three orientations and H(traj) 0.5-0.7 for 1 000 updates (violations stuck at 4 %), then diagonal-dominated (0.30 / 0.12 / 0.58, 3.5 %) and from update 8 500 a horizontal-dominated deterministic state with 2.5 % violations; λ at 20 from update 1 000 on.
- Guided, fixed estimator (`runs/dimers_crevise_guided_fixed.log`): three orientations with 3-4 % violations up to update 475 (0.22 / 0.33 / 0.44), then diagonal takes over as violations fall (update 650: 0.13 / 0.02 / 0.86, 1.7 %), all diagonal and within budget at update 1 000. The budget is met with a single orientation: more collapsed than cfast's two. This run managed only 1 975 updates (0.15 s per update against 0.013 s for the earlier runs with the same shapes; the warm run before it showed the same slowness, GPU clocks normal, no other GPU process; the only code difference is two extra recorded (16, n, D) arrays per revising group; not diagnosed).
- Reading: the revising steps never acted as symmetry breaking by propagation. The policy that cuts violations is a deterministic one, and because the redraws are deterministic functions of the coins too, nothing is left to re-pair conflicting cells differently. Trajectory entropy, which the path KL rewards, fell to zero in every run.

## Ground patch (8 x 16, T = 2; `scripts/ground_patch.py`)

The MCMC report's `ground_small` patch tiled to width 16 (every level's width must be a multiple of 4; the seam between the copies is the same row-3 / row-2 step as the 8-periodic wrap). Reference: the same annealed-importance sampler, 540 s one CPU core, 56 runs all valid and distinct, ESS 12.2, weighted dirt 0.641 (8 x 8 report: 0.638), weighted soft 33.0 (`runs/ground_patch_mcmc.json`). 256 held-out worlds per variant.

| run | estimator | updates | dirt | soft | line match | raw viol L0 / L1 / L2 | final viol | L1 P(dirt) to ref | max abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference (MCMC) | | | 0.641 | 33.0 | | 0 | 0 | 0 | 0 |
| cfast (180 s) | | ~9 000 | 0.651 | 32.5 | 0.892 | 9e-5 / 5.5e-4 / 0 | 1.5e-5 / 1.2e-4 / 0 | 0.051 | 0.39 |
| crevise_noguide (300 s) | biased term | 7 265 | 0.936 | 63.7 | 0.751 | 2.6e-4 / 0 / 0 | 9e-5 / 1.8e-4 / 0 | 0.295 | 0.90 |
| crevise guided (300 s) | biased term | 8 958 | 0.941 | 75.0 | 0.742 | 1.4e-4 / 0 / 0 | 3e-5 / 0 / 0 | 0.300 | 0.91 |

![ground patch P(dirt)](images/ground_patch_pdirt.png)

cfast tracks the reference on this small patch (the full-size ground collapse is a different regime); both crevise runs are all dirt except the two empty top-left blocks, with λ_0 at the cap. Not re-run with the fixed estimator.

## What failed, plainly

1. The first estimator was biased in the revising steps' all-action term; the gradient check with a deterministic test policy and the first negative control did not have the power to see it. Fixed and re-verified; four of the five training runs predate the fix.
2. Dimers: every configuration ends deterministic with one or two orientations; the only run within budget (guided, fixed) is all diagonal. cfast keeps two orientations within budget; cbound_lam three.
3. Ground patch: all dirt (0.94 vs 0.64) where cfast matches the reference (0.65, L1 0.05).
4. The bound is loose by construction with a uniform r (ρ R log D per cell), and score() is None for R > 0, so no R > 0 configuration can be accepted by the bench as it stands.
5. The fixed-estimator run was 10x slower per update for an unknown reason; bench and ground re-runs not done; graded field not swapped in (host-side only); one seed everywhere.

## Proposal: a designed fixed r (not built)

**What.** Keep the generator, the erase coins and the per-draw machinery; change P's refill of an erased cell from uniform to the designed local conditional of the unerased state, r(a | x_¬M) ∝ exp(−s(a) − λ_r nv(a)) over all D values (s, nv the closed terms given the unerased cells; λ_r a declared constant, e.g. 10), which has full support, is generic from the tables (no problem-specific code, constraint 1) and is fixed (not learned). The revising cost becomes τ(log q(a) − log r(o | x_¬M)) + the energy difference, with o the old value: one more cond call per revising colour step, nothing else changes; the gradient test machinery covers it unchanged.

**Why.** The gap E KL(Q(old | x_0) ‖ r(old | x_0)) is charged in full against trajectory entropy with uniform r: 1.79 nats per cell on dimers, where the target's conditionals are nearly deterministic (a cell's partner is fixed by its neighbours), so the objective tells the pass to be random where the target is not and then punishes every redraw by log 6 unless the old value was uniformly random. A designed r matches those peaked conditionals, so the charge becomes small wherever the old value was already locally consistent and large only where it was in conflict: the bound then measures what the revisions repair instead of taxing them. On the tiny problem the gap can be computed exactly for both choices with `_crevise_gradcheck.py` before any GPU run.

**Constraint.** Amendment 2026-10-14 (b) allows "a fixed noising process declared with the sampler"; the designed r is fixed and declared, but it is a different r from the one declared here (uniform), and the objective's value changes with it, so it is listed as a proposal under amendment (b) / constraint 4 rather than built.

A second, cheaper variant for the symmetry question: the field-dependent erase mixture above (coordinator's note) as a declared r, with the mask log-ratio in the costs; it keeps revisions concentrated where conflicts are, but it does not fix the uniform-refill charge, which the runs identify as the main cost.

## Next

1. Compute the exact gap with uniform vs designed r on the tiny problems (CPU, minutes) before deciding whether to build the designed r.
2. If it is built: dimers pre-check (60 s), then the ground patch, with the fixed estimator; profile the 10x slowdown first (compare `fused` timings with and without `soft_c` / `nv_c` recorded).
3. JAX twin of `graded_field` (or a host callback) to replace the binary stand-in; test equality against `graded.graded_field` on random states.

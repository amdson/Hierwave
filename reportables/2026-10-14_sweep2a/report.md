# sweep2a: the iterated amortised loop on ground's fine levels (Sweep 2, fine half)

2026-10-14 · ground, LN+SiLU 6x6 actor (net="ln", win=(-2,3)), levels 0-2 trained, levels 3-4 frozen at the cstart2 init (untrained). **Objective: weighted maximum likelihood (weights 1) on states from p*-invariant exact block-Gibbs kernels** (constitution, Amendment 2026-10-14). Not reverse KL. The GPU (RTX 3090) was shared with another agent. Arms A s0 and B s0 also ran concurrently with each other, so B s1, which ran alone, got 1016 updates against about 555.

## Pieces
- **castlegen/kernels/amortised.py** (generic: factors + layout + kernels):
  - `JaxBlockGibbs` is a JAX/GPU port of kernel-a's ExactBlockGibbs (shared-product path), batched over worlds. It reuses that class's blocks, colouring and instance groups and samples by chunked Gumbel-max.
  - Its conditional energies equal the numpy kernel's exactly (all 4 colour classes, (2,0) and (1,1)). Total energy equals `total_energy`. Test: tests/kernels/test_amortised.py, 1 pass, 29 s.
  - Speed: one (1,1) sweep of 32 worlds takes 0.57 s, about 50x the numpy kernel per world.
  - Also in the file: `make_energy` (per-world soft / violations / fine-level soft), `refine` (cycles with a per-cycle energy curve) and `make_ml_update_drawwise` (see the infeasible-draws note).
- **Kernel λ:** violations are priced at λ = 20 inside the kernels, not treated as infeasible. Own-draw contexts carry fixed coarse violations, and under the hard target those would freeze every block they touch.
- **Loop (run_loop.py):**
  - Each iteration takes 32 contexts. Arm A uses probe lifts only (jitter 0-3). Arm B uses 16 probe lifts + 16 own draws of the current params from iteration 2 on.
  - Each context gets 6 cycles of [(2,0), (1,1)] exact Gibbs on levels 0-2. Levels 3+ stay at the context.
  - Then 8 Adam updates (B = 8, lr 1e-3) of levels 0-2 on the refined states.
  - 300 s per arm, about 2.4-4.4 s per iteration.
- **Joint level-2 block with its subtree:** not tractable.
  - (2,2), i.e. rho2, cert2 + 4 x (rho1, cert1) + 16 tiles: joint 1.4e13. The pruned enumeration of a single block did not finish in 150 s.
  - (2,1) without tiles: pruned by internal-only hard instances, 2.4M completions at slot 8 of 10. Pruned per block with the outside values, 10,625 per block at 0.04 s each on CPU, but the set differs per block, so it cannot be batched. It is also nearly redundant, because the tiles pin rho1.
  - Not built.

## Chain convergence (images/energy_per_cycle.png, runs/curve/curve.json; 30 cycles, 32 contexts)
- **Probe contexts:** soft energy per cell goes 0.020 → 0.115 → 0.129 at cycle 2, then stays flat at 0.128-0.131 through cycle 30. **Kernel-a's 2-cycle states (0.128) were already at this plateau.** Blocks changed per sweep at the plateau: level-2 about 330/512, level-1 about 700/2048.
- **Own draws of the init model:** 2.69 → 1.07 by cycle 2, then 1.036 at cycle 30, still drifting down slowly. Most of the remainder is coarse (levels 3-4) energy, which these kernels cannot touch.
- **The refined probe states do not drift toward rough or dirt states** (runs/refined_view.json, images/refined_probe_states.png; 16 probes, 10 cycles):
  - The dirt fraction is unchanged (0.496 → 0.497), and so is the tile-tile pair energy (0.0087 → 0.0089 per cell).
  - The whole rise is in latent-count slack: t-rho1 count 0 → 0.058, rho1-rho1 0.004 → 0.022, rho1-rho2 count 0 → 0.016, rho2-rho3 count 0 → 0.012, rho2-rho2 0.002 → 0.008.
  - So the exact conditional, given level-3+ parents, keeps the probe's surface but puts entropy into rho1/rho2 that no longer equal the exact lifted counts. The surface also wanders within the parents' slack: line_match drops from 0.87 (j0) to 0.74, and from 0.76 to 0.73 averaged over jitters.
  - The ML target is therefore p*'s fine conditional, which is a little different from the probe, at about 0.13/cell rather than 0.014.

## Gate (eval_ground.py, unchanged; 32 seeds x 4 probes; cstart2's λ)

| model | upd | L=3 fill cost | L=3 soft | L=3 own | L=4 fill | L=4 own | line_match L=3 clamp | rho2 / rho1 / t change | nv/cell L=3 fill | nv/cell own |
|---|---|---|---|---|---|---|---|---|---|---|
| probes | | | 0.014 | | | | 0.876 | 0.079 / 0.037 / 0.016 | | |
| cstart2 LN6 (reverse KL) | 4485 | -0.004 | 0.194 | 1.12 | 1.16 | 1.50 | 0.59 | 0.028 / 0.016 / 0.015 | | |
| kernel-a ML from init | 1330 | -0.107 | 0.128 | 0.46 | 0.71 | 0.84 | 0.770 | 0.037 / 0.028 / 0.013 | 1.3e-4 | 0.011 |
| **A: probes, iterated** | 560 | -0.106 | 0.148 | 0.86 | 0.98 | 1.24 | 0.732 | 0.028 / 0.019 / 0.011 | 3.2e-4 | 0.025 |
| **B: probes + own, s0** | 552 | -0.106 | 0.140 | **0.35** | **0.68** | **0.73** | 0.734 | 0.029 / 0.019 / 0.011 | 2.0e-4 | 0.016 |
| **B: probes + own, s1** | 1016 | **-0.111** | **0.138** | **0.33** | **0.64** | **0.71** | 0.733 | 0.029 / 0.020 / 0.011 | 1.9e-4 | 0.016 |

(L=4 own includes the untrained level-3 draws. Own-context line_match stays at about 0.51 in every arm, because levels 3-4 are untrained.)

## Loop logs (runs/*/loop.jsonl, images/loop_curves.png)
- **ML loss per cell (levels 0-2):**
  - Arm A goes 0.55 → 0.33 and is flat after about 50 s (about 100 updates).
  - Arm B goes 0.60 → 0.38 and is flat after about 150 s. Its contexts are harder.
  - **The loop saturates after about 100-300 updates. Doubling the updates (B s1: 1016) changes nothing beyond seed noise.**
- **Arm B own draws' soft/cell:** 2.53 → 1.07 (fine-level part 2.09 → 0.64). After 6 cycles the same draws are at 1.03 (fine 0.60). The model's own fine levels end within about 0.04/cell of their refinement.
- **Refined probe contexts:** 0.129 ± 0.002 throughout, in both arms.
- **Infeasible draws under teacher forcing:** probe contexts 0. Own contexts 2-4e-4 of fine draws, meaning about 50 % of own refined states have at least one draw outside the pass's least-violating set; every such state sits in a violating coarse context. `make_ml_update` would drop those whole states, so the loop uses `make_ml_update_drawwise`, which drops only the infeasible draws (about 3 per state) and keeps the rest of the state's conditionals.

## Answers
- **Saturation.** The iterated loop reaches kernel-a's one-shot fit and stops there: L3 fill -0.106/-0.111 against -0.107. Refreshing contexts, running more cycles and doubling the updates did not move the probe-clamped gate. The ceiling is not chain equilibration, since chains reach their plateau in 2 cycles. It is either the actor class (about 0.33 nats/cell of ML loss in probe contexts) or the fact that the refined target itself sits at 0.13/cell with line_match about 0.73.
- **Own draws (B) help where A cannot:** own-context cost 0.35 vs 0.86 (kernel-a 0.46, which had 25 % own draws of cstart2), L=4 fill 0.68 vs 0.98, and fewer violations.
- **Regressions vs kernel-a:** line_match under the clamp 0.73 vs 0.77, and parent awareness rho2 0.029 vs 0.037 and rho1 0.019 vs 0.028. Both are consistent with the iterated target being the 6-cycle equilibrium (line_match 0.73) rather than kernel-a's mix of 1- and 2-cycle states (0.75). Not significant across one seed of kernel-a.

## Caveats
- A and B s0 shared the GPU with each other and with another agent. B s1 ran with more GPU, so it got 1016 updates.
- One seed for A. Levels 3-4 are untrained, so own contexts are coarse-violating (0.02-0.04 violations per cell).
- The kernel uses λ = 20, not the hard target. For probe contexts the two are practically the same: no violations appear.

## Files
- castlegen/kernels/amortised.py
- tests/kernels/test_amortised.py
- Scripts: run_loop.py, check_kernel.py, refined_view.py, plot.py; eval_ground.py, clamp_tools.py, gate.py (copies from kernel-a)
- runs/: curve/, smoke/, armA_s0/, armB_s0/, armB_s1/ (loop.jsonl, params_ground.pkl, ground_summary.json, change.json), logs, refined_view.json
- images/: energy_per_cycle.png, loop_curves.png, refined_probe_states.png, clamped_completions_arm*.png (own draws in the top row)

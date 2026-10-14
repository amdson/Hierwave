# kernel-a: exact fine-block Gibbs + ML fit of ground's fine levels (Sweep 1 (a))

2026-10-14 · ground, LN+SiLU 6x6 actor (net="ln", win=(-2,3)), one seed. GPU (RTX 3090) shared with other agents; cores 40-47 shared with kernel-b/kernel-c jobs. Weighted-ML training on kernel-refined states is approved for training (constitution, Amendment 2026-10-14).

## Pieces

**Kernel (castlegen/kernels/exact_block.py).** Generic exact block Gibbs, built only from the factor declarations and the layout:
- Every factor instance is listed once as variable ids in a flat state. The listing follows the same rules as bench/energy.py (periodic x, pads, within, by_pos, pad_a).
- A block spec (L, depth) covers a level-L cell's sampled channels plus those of its descendants down to L - depth.
- The kernel enumerates every joint completion and computes each one's conditional energy from the instances touching the block. With lam=None (used here) the hard target applies and violating completions are excluded; lam=20 is also supported. It then samples the exact conditional.
- Blocks that share no instance are updated in parallel, using a greedy colouring of the block conflict graph (4 colours on ground).
- Small blocks use the full shared product. Large ones are built child-first, pruned by the hard instances and cached.
- On ground: (1, 1) = rho1, cert1 and 4 tiles, 480 joint values, about 0.9 s per sweep of 2048 blocks. (2, 0) = rho2, cert2, 255 values.
- No ground-specific code.

**Tests (tests/kernels/test_exact_block.py, 9 pass, 75 s):**
- Instance energy equals bench energy on all three problems.
- Each block's conditional energy differences equal full-energy differences (toy_support, ground).
- Level-2 full blocks (pruned path), started from an all-zero invalid state, reproduce problem.exact on toy_support and toy_modes within 4 SE (99 sweeps x 32 blocks).
- Invariance: level-1 blocks alone are not ergodic on the toys (the hard counts pin rho2). Started from exact samples (60 chains, 10 sweeps), the stats stay at exact within 4 SE on both toys, and the blocks do move.

**ML update.** I used kernel-c's castlegen/kernels/logq.py (`make_ml_update`: teacher-forced log q, Adam, per-level mask). Its file existed when I got there, so I did not write a duplicate. Levels 0-2 train; levels 3-4 are frozen.

## Experiment

**Data (make_data.py):**
- 160 ground worlds: 75 % probe lifts (jitter 0-3, uniform) and 25 % own full-depth draws of cstart2's trained cstart_f1L3_ln6.
- Each world gets 2 cycles of [(2,0) then (1,1)] exact Gibbs. Levels >= 3 stay at the context.
- Both post-cycle states are kept: 320 states, 237 s on 7 processes.
- Per sweep, 30-36 % of level-1 blocks and 55-64 % of level-2 cells change.
- Probe-context energy rises from 0.019 to 0.128 per cell, so the chain is not equilibrated after 2 cycles. line_match goes from 0.77 (probes, all jitters) to 0.75.

**Fit (train_ml.py):**
- B = 8, lr 1e-3, 300 s, about 1330 updates per arm.
- Arm `ml_init` starts from the cstart2 init; arm `ml_warm` starts from cstart2's trained LN6.
- Held-out log q per fine cell at levels 0/1/2 goes from -0.015/-0.35/-0.17 to -0.009/-0.162/-0.090, flat after about update 500. Both arms end at the same values (images/ml_heldout.png).
- No infeasible (out-of-mask) forced draws.

## Gate (eval_ground.py from cstart2, unchanged; 32 seeds x 4 probes; cstart2's λ (0, 6.3, 6.1, 0, 0))

| model | L=3 fill cost | (λ=20) | L=3 soft | L=3 own | L=4 fill | L=4 own | line_match L=3 clamp |
|---|---|---|---|---|---|---|---|
| untrained | 1.92 | 2.06 | 2.41 | 1.82 | 2.18 | 2.20 | 0.60 |
| cstart2 LN6 (reverse KL, 4485 upd) | -0.004 | -0.004 | 0.194 | 1.12 | 1.16 | 1.50 | 0.59 |
| **ML on kernel states, from init** | **-0.107** | -0.104 | **0.128** | **0.46** | **0.71** | 0.84 | **0.770** |
| ML, warm from cstart2 | -0.105 | -0.100 | 0.127 | 0.49 | 0.75 | 0.87 | 0.758 |
| probe itself (j0) | | | 0.014 | | | | 0.876 |

Violations: 1-3e-4 per cell. Own-context line_match is still 0.51, because levels 3-4 are untrained there.

**Parent awareness** (fraction of draws that change, probe j0 vs j2 clamp, L = 3):

| | rho2 | rho1 | t |
|---|---|---|---|
| probes | 0.079 | 0.037 | 0.016 |
| cstart2 best (LN6) | 0.028 | 0.016 | 0.015 |
| ML from init | 0.037 | 0.028 | 0.013 |
| ML warm | 0.033 | 0.023 | 0.012 |

**Images.** images/clamped_completions.png: the ML model's clamped completions follow the probe's hill with 1-2 row dirt teeth. cstart2's LN6 draws 2-3 flat block terraces instead.

## Answer
- Yes: under maximum likelihood on exact local conditionals, the same LN+SiLU 6x6 class leaves the block-terrace plateau that reverse KL stuck at.
- line_match under the clamp is 0.77 vs 0.59, against 0.75 for the training states themselves and 0.88 for the probe. Fill soft energy is 0.128, equal to the data's 0.128. Own-context cost falls 1.12 -> 0.46.
- rho1 is now near probe-level parent responsiveness (0.028 vs 0.037). rho2 is at about half (0.037 vs 0.079).
- The fit matches its data. The remaining gap to p* is in the data: only 2 Gibbs cycles from probe starts, so the states sit between the probe and p*'s fine conditional.

## Caveats
- One seed, 320 training states from 160 worlds.
- The held-out split is by state, so the two sweeps of one world can land on both sides.
- t's change fraction (0.013) is below the probes' 0.016 and not above untrained: tiles are still mostly pinned by level-1 certificates.
- Own-draw contexts carry about 86 coarse violations per world, kept as context.
- The pruned-path cache key had a bug, fixed before the final test run. It affected only the toys' speed, not ground, which uses the shared 480-product path.

## Next
- More, and longer, chains: equilibrate further and refresh the data during training (Sweep 2 loop).
- Include own draws of the current model.
- Add (2, 1) or (2, 2) blocks so rho2/cert2 move jointly with their subtree.
- Then let levels 3-4 train the same way and run the v2 bench.

Files: make_data.py, train_ml.py, eval_ground.py / clamp_tools.py / gate.py (copies from cstart2), plot.py; runs/ (data_a.npz, ml_init/, ml_warm/, logs, smoke); images/.

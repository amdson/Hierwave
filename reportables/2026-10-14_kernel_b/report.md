# kernel_b: generic block-MH kernel with q as the proposal (Sweep 1 b)

2026-10-14 · agent kernel-b · RTX 3090, GPU shared with other agents (cores 48-55) · about 60 min wall time.
Scope: `notes/decision_2026-10-14_kernel_route.md`, Sweep 1 (b). The constitution amendment of 2026-10-14 approves this kernel as training machinery.

## What was built

The kernel is in `castlegen/kernels/block_mh.py`. It is generic: it reads only the factors (through the pass's CondJax) and the layout.

- **Block.** A block is a level-L cell plus its whole subtree, meaning every sampled channel at levels l ≤ L under that cell. Everything else is fixed.
- **Proposal q(x_B | x_out).** This is the coloured pass restricted to the block.
  - Outside values are known from the start.
  - The pass's own steps (levels ≤ L, 16 colours, channels in order) draw only the block's sites.
  - Each draw uses the pass's masked conditional with whole-logit tempering, `(-soft - b) / tau`, and fresh `jax.random` Gumbel noise.
  - Teacher-forcing the old block values through the same function gives q(x_B | x_out).
- **Target ratio.** The closed energies that the pass accumulates on the block's steps count every block-touching factor instance exactly once. So E_B(x') − E_B(x) is the exact energy difference, with lam = 20 or hard (inf).
- **Move.** This is independent-proposal multiple-try MH: K candidates, weights w = p*/q, pick j ∝ w, accept with min(1, Σw / (Σ_{i≠j} w + w_x)). K = 1 reduces to plain MH.
- **Per-move log.** Each move records the level, block size, log r, accepted, dE, d log q and violations.
- **Delayed acceptance is not built.** The full block pass is a single vmapped scan, so a coarse-factor screen would save no compute.
- **Unit checks** (`scripts/units.py`, `tests/kernels/test_block_mh.py`):
  - The block energy delta matches `energy.py` to 3e-6.
  - The log q of a draw equals its teacher-forced log q (difference 0.0).
  - Draws follow exp(log q): on a toy_support L = 1 block, TV = 0.0016 and chi² = 0.96 with 3 dof over 20 000 draws.

## Invariance on the toys (8 × 16 periodic region, 8 iid blocks, 128 chains, lam = 20 unless noted)

Max |z| is taken over all exact stats, with z = (estimate − exact) / SE and the SE computed from the spread between chains. Burn-in is the first third of each chain.

| problem | q | L | K, tau | start | moves/chain | acc | max abs z | verdict |
|---|---|---|---|---|---|---|---|---|
| toy_modes | bad (random head, scale 3) | 2 | 1, 1 | exact | 20 000 | 0.06 | **1.3** | no drift |
| toy_modes | bad, **q ratio dropped** | 2 | 1, 1 | exact | 20 000 | 0.39 | **9188** (mode_B → 0.999) | caught |
| toy_modes | bad | 2 | 8, 1 | exact | 6 000 | 0.13 | 1.1 | no drift |
| toy_modes | bad | 2 | 1, 2 | exact | 10 000 | 0.07 | 1.1 | no drift |
| toy_modes | mild (scale 1) | 2 | 1, 1 | exact | 10 000 | 0.11 | 0.9 | no drift |
| toy_modes | bad | 1 (lam inf) | 1, 1 | exact | 2 000 | 0.84 | 1.0 | no drift |
| toy_modes | untrained | 2 | 1, 1 | own | 20 000 | 0.19 | 1.6 | converged |
| toy_modes | untrained | 1+2 | 1, 1 | own | 20 000 | 0.90 / 0.19 | 1.4 | converged |
| toy_modes | untrained | 2 | 8, 1 | own | 10 000 | 0.62 | 1.4 | converged |
| toy_modes | trained (cfast, clamp pkl) | 1+2 | 1, 1 | own | 2 000 | 0.92 / 0.62 | 2.1 | converged |
| toy_modes | mild | 2 | 1, 1 | own | 20 000 | 0.12 | 3.4 (mode_other) | not converged (see below) |
| toy_modes | bad | 2 | 1, 1 | own | 3 000 | 0.30 | 81 | not mixed (q almost never proposes L/T modes) |
| toy_support | bad | 2 | 1, 1 | exact | 20 000 | 0.001 | 1.3 | no drift (weak: barely moves) |
| toy_support | bad, **q ratio dropped** | 2 | 1, 1 | exact | 20 000 | 0.014 | **37.5** (flat 0.33 vs 0.096) | caught |
| toy_support | bad | 1+2 (lam inf) | 1, 1 | exact | 20 000 | 0.62 / 0.001 | 1.2 | no drift |
| toy_support | bad | 2 | 8, 2 | exact | 6 000 | 0.06 | 2.5 (dirt_frac) | borderline |
| toy_support | untrained | 2 | 1, 1 | own | 20 000 | 0.05 | 1.8 | converged |
| toy_support | untrained | 1+2 | 1, 1 | own | 3 000 | 0.78 / 0.07 | 1.9 | converged |
| toy_support | untrained | 2 | 8, 1 | own | 10 000 | 0.30 | 0.9 | converged |
| toy_support | mild | 1+2 | 1, 1 | own | 20 000 | 0.74 / 0.03 | 6.1 (dirt_frac) | not converged |

![toy invariance](images/toy_invariance.png)

**Reading.**
- **Exact starts.** Every chain started from exact samples stays at the exact stats (|z| ≤ 1.3, one 2.5). Dropping the q ratio sends the same chains off by thousands of SE on toy_modes and 37 SE on toy_support. The test therefore has the power to catch a missing proposal-density term.
- **Starts from own draws.** These converge to exact when q mixes (untrained, trained, K = 8).
- **The two own-draw failures** (mild q: z 3.4 on toy_modes, 6.1 on toy_support) come with block acceptance of 0.03-0.12. The same mild q is exact-invariant from exact starts (z 0.9), so I read these as incomplete mixing, not bias. On toy_support with L = 1 and lam = 20, block counts can only move through L = 2 moves accepted at 3 %.
- **Weak spot.** The toy_support bad-q exact-start run accepts 0.1 % of moves, so its "no drift" says little. The toy_modes runs carry the invariance claim.

## Ground: acceptance of subtree resampling per level (lam = 20)

Setup: 256 moves per cell; each move uses a random world out of 64 and a uniformly random level-L cell, and moves are not chained. "own" means raw pass draws at tau = 1; "probe" means probe lifts.

Each cell shows acc / moved, where moved means accepted and the block actually changed. Block sizes are 6 / 26 / 106 / 426 sites at L = 1 / 2 / 3 / 4.

**cfast** (clamp `params_ground.pkl`, 1357 updates; own draws are 100 % dirt)

| start | L | t1 K1 | t2 K1 | t1 K8 | t2 K8 |
|---|---|---|---|---|---|
| own | 1 | .84/.29 | .81/.30 | .96/.50 | .99/.49 |
| own | 2 | .64/.57 | .62/.58 | .84/.79 | .96/.93 |
| own | 3 | .49/.49 | .38/.38 | .62/.62 | .62/.62 |
| own | 4 | .56/.56 | .17/.17 | .94/.94 | .68/.68 |
| probe | 1 | .36/.05 | .34/.07 | .47/.07 | .53/.07 |
| probe | 2-4 | **0/0** | 0/0 | 0/0 | 0/0 |

**cstart_f1L3** (cstart2 run; own draws are 90 % dirt)

| start | L | t1 K1 | t2 K1 | t1 K8 | t2 K8 |
|---|---|---|---|---|---|
| own | 1 | .85/.18 | .81/.24 | .93/.53 | .98/.62 |
| own | 2 | .68/.50 | .67/.57 | .91/.83 | .92/.91 |
| own | 3 | .45/.45 | .41/.41 | .59/.59 | .67/.67 |
| own | 4 | .42/.42 | .34/.34 | .82/.82 | .78/.78 |
| probe | 1 | .62/.06 | .64/.10 | .85/.12 | .87/.11 |
| probe | 2-4 | **0/0** | 0/0 | 0/0 | 0/0 |

**Untrained q** (reference): from own draws, acceptance at L = 1-4 is 0.71 / 0.51 / 0.33 / 0.38 (K = 1, tau = 1). From probes it is 0.74 / 0.04 / 0 / 0 (with K = 8 it is 0.98 / 0.22 / 0 / 0).

**Exploration (dirt to fill).** An accepted move counts if its block was all dirt in the fine tiles and is not all dirt afterwards. All counts below are summed over the four (tau, K) settings at each level.

| q | L = 1 | L = 2 | L = 3 | L = 4 |
|---|---|---|---|---|
| cfast | 0 / 1024 | 0 / 1024 | 0 / 1024 | 9 / 1024 |
| cstart_f1L3 | 2 / 876 | 0 / 855 | 10 / 701 | 31 / 423 |
| untrained | 6 / 853 | 9 / 782 | 12 / 611 | 14 / 373 |

**Reading.**
- **Own draws: acceptance is high but uninformative.** Under q's own draws, acceptance only measures the spread of p*/q over q's own support. A collapsed q proposes another all-dirt block and accepts it. Exploration away from all-dirt is about 1 % or less of accepted moves for cfast. For cstart it is 7 % at L = 4.
- **Probes give the diagnostic: q's conditionals are badly wrong from L = 2 up.** Starting from a fill-following state, no proposal is ever accepted at L ≥ 2, for any tau or K, for both trained q's.
  - Median log r at tau = 1 is −73 / −270 / −1228 at L = 2 / 3 / 4 for cfast.
  - The d log q term dominates: q assigns the probe's own block 35 / 194 / 827 nats less than its proposal.
  - The energy difference also favours the probe: median dE of the proposal is +12 / +30 / +246.
  - So q puts its mass on blocks that are both worse under p* and far from the target's typical set.
  - Even the untrained q fails at L ≥ 3, so subtree resampling from target-like states needs blocks no larger than L = 1-2.
- **L = 1 does move probes, but only a little:** about 0.36-0.85 acceptance, but only 5-12 % of blocks actually change.

## What failed / caveats

- **Two own-draw toy runs with a mild bad q did not converge** in 20 000 moves (z 3.4 and 6.1). Exact-start runs with the same q show no drift, so I read these as slow mixing, not bias.
- **The toy_support K = 8, tau = 2 exact-start run sits at z = 2.5** on dirt_frac. It is borderline and was not repeated for lack of time.
- **The toy_support bad-q exact start barely moves** (acceptance 0.001).
- **Delayed acceptance is not built** (no compute saving in this implementation).
- **Ground used one move per state, not chains.** The probe numbers are a snapshot of q's conditionals, not a mixing study.
- **Acceptance on ground used lam = 20.** Proposal violations were 0-16 % at L = 4 from probes.

## Next

- Use L = 1 (K = 8) MTM moves, plus exact fine Gibbs (kernel a), as the kernel cycle on ground. Larger blocks only make sense once q's conditionals on target-like parents improve.
- Use "probe acceptance per level" as the Sweep 2 diagnostic: it is 0 for the current q at L ≥ 2.
- Repeat the toy_support K = 8 exact-start run with more chains.

## Files

- `castlegen/kernels/block_mh.py` (new; the package `__init__` was already created by kernel-c)
- `tests/kernels/test_block_mh.py`
- Scripts in `scripts/`:
  - `common.py`, `units.py`
  - `toy_chain.py`, `run_toys.sh`, `list*.txt`, `summ.py`, `plot_toys.py`
  - `ground_accept.py`, `run_ground.sh`, `ground_all.sh`
- `runs/`: toy JSON results (`toy_*.json`), ground results (`ground_*.json`), logs
- `images/toy_invariance.png`

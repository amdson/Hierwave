# speed-2: GPU-batched generic single-cell Gibbs (colour_gibbs)

2026-10-14 · agent speed-2 · RTX 3090 shared with other agents · about 75 min wall time.

## What was built (`castlegen/kernels/colour_gibbs.py`)

Generic: it reads only the layout (`lay.cond`, i.e. CondJax) and the factor instances (`exact_block.build_instances`, imported, not modified).

- `gibbs_sweep(lay, state, level, channel, colour, tau, lam, key, mask=None)`: resamples every cell of one (level, channel, colour) class in all B worlds in parallel, from `exp(-(soft + lam*nv)/tau)` over the full channel domain (unary, pair, parent-child, count factors all included). `lam=None` excludes violating values (untempered hard target; a cell with no feasible value stays).
- `sweep_level` (16 colours via `lax.scan`, channels in declared order), `sweep_all` (levels coarse to fine), optional `mask` (dict channel -> bool (B,h,w), a different mask per world).
- `subtree_mask(lay, L, cells)`: the sampled-channel cells at levels <= L under one level-L cell per world.
- `anneal_block(lay, state, mask, key, tau_hi, n_steps, lam)`: AIS inside the mask. beta_j from 1/tau_hi to 1 (geometric in tau by default). The weight is `log w = sum_j (beta_j - beta_{j-1}) * (-E(x_{j-1}))`, where x_j comes from one masked sweep at beta_j over the mask's levels. E is tracked incrementally from the sweeps' exact energy differences, starting from the absolute energy of the instances touching the mask (`state_energy`), so log w is absolute. It is the log of an unbiased estimate of Z_1/Z_beta0 of the mask's conditional given the outside, if x_0 ~ pi_beta0.
- `state_energy`, `conditional`, `conflicts`: absolute energy of the instances touching a mask; the exact (soft, nv) per site and value; colouring check.

**Colouring validity.** `conflicts(lay)` lists every factor instance (periodic wrap, pads, `within`, `by_pos`, `pad_a` rows, counts) with two cells of one (channel, colour) class. Result: none for toy_modes, toy_support and ground (all factor types: unary, same-level pair with offsets (0,1), (1,0), parent-level pair, count). A count factor couples a parent with four children; siblings differ in (Y&1, X&1), hence in colour, and the parents are on another level. If a problem did have such a factor, the class is split by greedy colouring of its conflict graph and the groups are updated one after another (implemented, `ngroups > 1`; not exercised because no declared problem needs it). Self-coupled cells (a pair with offset 0 mod the strip) are not supported; the check would report them.

Hard-limit note: with lam=None (or large lam) single-cell moves cannot cross hard count constraints (flipping one tile violates its rho1). Single-cell Gibbs therefore needs a finite lam (or block moves) to mix; see the ground result below.

## Tests (`tests/kernels/test_colour_gibbs.py`, 22 pass, ~1.5 min total)

Exact references for the toys use the lam-penalised target (lam = 2, so single cells can move), computed by enumerating the 2^16 block states and summing over (rho1 x4, rho2) per block: exact block distribution, exact stats (toys' own stat functions plus mean rho2 and mean violations per block), exact log Z at any beta, and an exact sampler. The energy bookkeeping behind them is cross-checked against `energy.py`.

| test | result |
|---|---|
| colouring: no conflicts on toy_modes, toy_support, ground | pass |
| `state_energy` equals `energy.energy` (soft, violations) on all three problems | pass |
| (b) conditional equals brute force: for random cells at every level and channel of toy_modes, toy_support, ground, the full-energy difference over every domain value equals the `lay.cond` conditional (soft atol 2e-3, violation counts exact) | pass |
| (b) a class sweep on 20 000 copies of one state draws each cell from the softmax (toy_support lam 3; ground lam 3 and lam=None) | pass (max z < 4.5; zero draws on excluded values) |
| (a) invariance, exact samples + 6 sweeps of ONE level (0, 1, 2) | max abs z: modes 2.25 / 1.72 / 1.20; support 1.23 / 1.10 / 1.26 |
| (a) mixing from uniform-random starts, 200 all-level sweeps, 256 chains | max abs z: modes 0.94, support 0.90 |
| (c) AIS, tau 4 -> 1, 64 steps, 4000 runs, subtree of one level-2 cell | log mean exp(log w) vs exact log Z ratio: modes -11.510 +- 0.014 vs -11.505; support -5.998 +- 0.008 vs -5.985 |

AIS also with 8 steps: modes -11.568 +- 0.064, support -6.033 +- 0.026 (ESS fraction 0.06 / 0.27, versus 0.53 / 0.78 at 64 steps): the estimate converges to the exact ratio as steps grow.

Negative control: sweeping exact samples at tau = 1.5 instead of 1 gives max |z| = 43 on both toys, so the invariance test does catch a wrong conditional temperature. (`runs/test_numbers.json`, `scripts/test_numbers.py`.)

Test notes: per-chain means over the 8 blocks; SE from the spread across chains, floored at 2e-3 (an almost-never-hit statistic like empty_block_frac has zero sample spread). The level-wise test at level 1 and 2 conditions on the other levels, as it must.

## Throughput on ground (W = 64, 64 worlds, tau 8 -> 1, lam = 20; `runs/throughput.json`, `scripts/throughput.py`)

GPU shared with two other agents; times vary with their load (the first, contended run gave 9 ms / 9.6 ms / 0.75 s for the three lines below; the quiet run is shown).

| job | work | time | rate |
|---|---|---|---|
| level-0 sweep (16 colours, channel t) | 64 x 4096 = 262 144 cell updates | 1.9 ms | 1.4e8 cells/s (3.0e7 when contended) |
| level-3 sweep (rho3, cert3) | 64 x 64 x 2 = 8 192 cell updates | 1.9 ms | 4.2e6 cells/s (8.6e5 contended) |
| annealed level-3 subtree block, 32 steps, batch of 64 blocks (106 sampled cells per block: levels 3..0 under one level-3 cell) | 64 blocks x 32 steps | 0.095 s (0.75 s contended) | 670 blocks/s (86 contended) |
| same, 128 steps | | 0.37 s | 173 blocks/s |

First compile of each is 1-6 s (warm cache: under 1 s).

Comparison (different kernels, so per update they are not like for like):
- kernel-a's exact block Gibbs (python per-block loop, numpy, level (1,1): 6 sampled cells per block): 0.9 s for a sweep of 2048 blocks = 2.3e3 blocks/s = 1.4e4 sampled cells/s. A level-0 colour_gibbs sweep updates 1e4 times as many cells per second. It does single cells, kernel-a does exact 480-value joint blocks; the useful comparison is mixing per second, not measured here.
- The other session's annealed MCMC (Mac, one core, 8 x 8 patch, 102 valid samples in 5 min = 0.34 samples/s): colour_gibbs anneals 670 level-3 blocks/s (a level-3 block spans 8 x 8 fine tiles on W = 64, the same area as that patch's size). That is 2000x the sample rate, but with a caveat that matters:

**Failure / limitation: the annealed blocks are not valid.** Single-cell Gibbs from a uniform-random start leaves violations: mean violations touching the block at the end are 10.9 (32 steps) and 8.8 (128 steps) at lam = 20, 11.8 / 10.0 at lam = 60, and 0 of 64 blocks reach zero (`runs/anneal_quality.json`). The mean energy drops from 1359 to 313 (lam 20, 32 steps) and to 268 (128 steps). The ground_mcmc session reached valid samples by annealing lam to hard over 150 steps from a burn-in at lam = 0.5; this primitive anneals tau (soft energy and lam*nv together) with lam fixed, and has no lam schedule or block move. The sampled weights have a large spread (std 107 of log w over 64 blocks, mean -325), so they are poorly conditioned on ground too. What the primitive guarantees is the exact kernel and the correct AIS weight; whether a given schedule reaches valid states is a schedule/start question (e.g. start from q's own draws or the exact block kernel's states, a lam ramp, or hard-excluded sweeps after a valid start). I did not tune this.

## Files

Created: `castlegen/kernels/colour_gibbs.py`, `tests/kernels/test_colour_gibbs.py`, `reportables/2026-10-14_speed2/{report.md, scripts/throughput.py, scripts/test_numbers.py, scripts/anneal_quality.py, runs/throughput.json, runs/test_numbers.json, runs/anneal_quality.json}`. Modified: `reportables/progress.md` (entry at top). `images/` is empty: no figure was needed.

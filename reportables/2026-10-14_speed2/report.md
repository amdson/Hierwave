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

# Part 2: valid annealed blocks

New file `castlegen/kernels/anneal.py`: `anneal_block_valid(lay, state, mask, key, tau_hi, n_steps, lam_kind, lam_end, final, n_hard, joint, joint_sweeps)` returns `(state, logw, valid, info)`. Scripts `scripts/valid_blocks.py`; runs `runs/valid_blocks.json` (const / ramp), `valid_blocks2.json` (const / geo, hard final), `valid_blocks2_joint.json`. Tests added to `tests/kernels/test_colour_gibbs.py` (27 pass in total).

**Weight with a lambda schedule.** Target at step j: f_j(x) = exp(-beta_j (S(x) + lam_j N(x))), beta_j = 1/tau_j (tau 8 -> 1 geometrically), S soft energy, N violations of the instances touching the mask. x_j is one masked sweep invariant for f_j. The AIS weight is

log w = sum_{j=1..n} [log f_j(x_{j-1}) - log f_{j-1}(x_{j-1})] = - sum_j [ beta_j (S_{j-1} + lam_j N_{j-1}) - beta_{j-1} (S_{j-1} + lam_{j-1} N_{j-1}) ],

with S, N of x_{j-1} tracked exactly. Hard final phase (lam -> inf): f_hard = exp(-beta_n S) 1[N=0], so f_hard/f_n = 1[N=0]: log w is -inf for blocks with N > 0 and unchanged otherwise; further hard sweeps leave f_hard invariant. Exact only if x_0 ~ f_0 (toy tests); from probe or q starts the weights are heuristic.

**Toy check.** lam ramps 0 -> 2 (linear) and 0.5 -> 2 (geometric), tau 4 -> 1, 64 steps, 4000 runs, exact starts: log-mean-exp of log w equals the exact log Z(beta=1, lam=2) - log Z(beta0, lam0) within 4 bootstrap SE (or 0.02), toy_modes and toy_support, both schedules: 4 tests pass. A further test checks that hard-final weights are finite only for valid blocks.

**Ground table** (W = 64, 64 blocks, tau 8 -> 1, lam_end = 20, GPU shared). "const" = lam 20 throughout (tau annealing only); "geo" = lam 0.5 -> 20 geometric, "ramp" = 0 -> 20 linear (ramp rows from the first run, hard final adds nothing to the lam rows). Validity = 0 violations touching the block. ESS is over all 64 weights (lam final; with hard final the invalid blocks get weight 0). Changed = fraction of the block's cells (all channels) differing from the start. Pure->part = fraction of blocks that start all-dirt or all-air in t and end partially filled.

| level | steps | start | lam | valid | ESS/64 | changed | dirt start -> end | pure->part | s/64 blocks |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 32 | random | const | 0.00 | 0.0 | 0.60 | .50 -> .48 | 0 | 0.10 |
| 3 | 32 | probe | const | **1.00** | **15.7** | 0.10 | .45 -> .45 | 0 | 0.10 |
| 3 | 32 | probe | geo | 0.16 | 1.1 | 0.28 | .45 -> .47 | 0.72 | 0.10 |
| 3 | 32 | probe | ramp | 0.17 | 1.9 | 0.30 | .45 -> .47 | 0.73 | 0.10 |
| 3 | 32 | q draw | const | **0.98** | 1.5 | 0.21 | 1.00 -> .92 | 0.17 | 0.10 |
| 3 | 32 | q draw | geo | 0.17 | 1.0 | 0.50 | 1.00 -> .64 | 0.94 | 0.10 |
| 3 | 128 | random | const | 0.02 | 1.0 | 0.63 | .50 -> .48 | 0 | 0.44 |
| 3 | 128 | probe | const | **0.98** | **17.5** | 0.10 | .45 -> .45 | 0.02 | 0.37 |
| 3 | 128 | probe | geo | 0.81 | 2.9 | 0.12 | .45 -> .45 | 0.14 | 0.37 |
| 3 | 128 | probe | ramp | 0.88 | 1.1 | 0.12 | .45 -> .45 | 0.09 | 0.36 |
| 3 | 128 | q draw | const | **1.00** | 2.0 | 0.24 | 1.00 -> .88 | 0.16 | 0.37 |
| 3 | 128 | q draw | geo | 0.55 | 1.7 | 0.47 | 1.00 -> .65 | 0.61 | 0.38 |
| 4 | 32 | random | const | 0.00 | 0.0 | 0.60 | .50 -> .49 | 0 | 0.12 |
| 4 | 32 | probe | const | 1.00 | 1.0 | 0.10 | .45 -> .46 | 0.02 | 0.12 |
| 4 | 32 | probe | geo | 0.00 | 0.0 | 0.38 | .45 -> .47 | 0.72 | 0.12 |
| 4 | 32 | q draw | const | 0.98 | 1.0 | 0.18 | 1.00 -> .95 | 0.25 | 0.12 |
| 4 | 128 | random | const | 0.00 | 0.0 | 0.62 | .50 -> .49 | 0 | 0.46 |
| 4 | 128 | probe | const | 0.92 | 5.6 | 0.11 | .45 -> .46 | 0.08 | 0.44 |
| 4 | 128 | probe | geo | 0.08 | 1.0 | 0.24 | .45 -> .47 | 0.70 | 0.44 |
| 4 | 128 | q draw | const | 0.89 | 3.6 | 0.24 | 1.00 -> .88 | 0.34 | 0.57 |
| 4 | 128 | q draw | geo | 0.05 | 1.0 | 0.52 | 1.00 -> .61 | 0.98 | 0.44 |

Starts are valid in 100% of blocks for probe and q draws (the q draws of the trained cfast ground model have 0 violations touching these blocks), and 0% for random.

**Reading.**
- Valid starts with tau-only annealing and lam fixed at 20 give valid states: 98-100% at level 3 (92%/89% at level 4, 128 steps), at 670 blocks/s (32 steps) or 170 blocks/s (128 steps). With probe starts the ESS is 16-17 of 64 at level 3, so the weights are usable there; q-draw starts give ESS 1.5-3.6.
- The lam ramp from 0 does not help: lam = 0 at the start erases the valid start, and the ramp needs ~128 steps to recover (81-88% at level 3, 5-8% at level 4). It makes bigger changes (the all-dirt q-draw blocks become partially filled in 61-98% of blocks, dirt 1.00 -> 0.65) but only 16-55% of those are valid. Random starts never become valid with any lam schedule (level 3: 0-2% at 128 steps; 9-12 violations left at 32 steps, 21-34 at level 4), and the hard final phase does not repair them (n_hard = 2 sweeps).
- So the useful setting is the tempered perturbation of a valid start (const lam 20). The perturbation is local: 10% of cells change from a probe, 20-25% from a q draw. A dirt block can become partially filled: from q draws (all-dirt blocks) 16-17% of blocks do at level 3 (dirt 1.00 -> 0.88-0.92) and 25-34% at level 4; from probes 0-2% (probe blocks are mostly already partly filled; 83% of level-3 blocks were pure, and ~0% of those changed).
- Weights from probe/q starts are not exact AIS weights (the start is not pi_beta0), so ESS describes the weight spread only.

**Joint fine-block moves** (kernel-a `ExactBlockGibbs(p, Region(64), level=1, lam=20)`, imported unmodified). Its API has no tau and runs numpy over the whole region, one world at a time, so it is applied once after the anneal (tau = 1 target, p*-invariant, weight unchanged). Cost: about 20 s for 64 worlds (68 s with its pruned-enumeration cache cold) versus 0.1 s for the anneal. Level 3, 32 steps, hard final: validity random 0.00 -> 0.00 (mean violations 11.1 -> 8.6), probe/const 1.00 -> 1.00, q draw/const 0.98 -> 0.98, probe/geo 0.16 -> 0.23, q draw/geo 0.17 -> 0.20. It barely helps and is 200x slower than the anneal; a GPU version with tau would be needed to interleave it, which I did not build (edit of exact_block.py is not mine).

**Recommendation.** Use `anneal_block_valid(..., lam_kind="const", lam_end=20, final="hard")` from probe or q-draw starts at tau_hi = 8, 32 steps for level 3 (100%/98% valid, 0.1 s per 64 blocks), 128 steps for level 4 (89-92%). It cannot generate valid blocks from random starts.

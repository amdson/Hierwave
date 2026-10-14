# sweep2b: the kernel route on the coarse levels, packaged as variant `ckernel` (Sweep 2, coarse half)

2026-10-14 · agent sweep2b · ground, LN+SiLU 6x6 actor (net="ln", win=(-2,3)), one seed · RTX 3090, GPU shared with sweep2a and other agents · cores 48-55.
Objective: **weighted maximum likelihood (weights 1) on states refined by p*-invariant kernels**, for all levels 0-4 (constitution, Amendment 2026-10-14). There is no reverse-KL term. Generation is cfast's coloured pass, unchanged.

## 1. Bottom-up acceptance check (kernel-b's ground_accept.py, kernel-a params)

**Setup.**
- Params: kernel-a `runs/ml_init`. Levels 0-2 are ML-fit on exact-Gibbs refined states. **Levels 3-4 are at cstart2's init, untrained**; nothing was copied from cfast.
- One move per (state, random level-L cell), with no chaining; 256 moves per cell; λ = 20.
- "probe" starts from probe lifts; "own" starts from the model's own τ = 1 draws, which are 85 % dirt.
- Each cell below gives acceptance / fraction of moves that changed the block, followed by the median log acceptance ratio (log r).

| L | K | τ | own: acc/moved, med log r | probe: acc/moved, med log r | probe med d log q | probe med dE | cfast probe acc (kernel-b) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | .89/.36, 0.0 | .93/.16, 0.0 | 0.0 | 0.0 | .36 |
| 1 | 8 | 1 | .94/.42, 0.0 | .99/.15, 0.0 | 0.0 | 0.0 | .47 |
| 2 | 1 | 1 | .51/.48, -0.6 | **.79/.54, -0.1** | -1.4 | +1.8 | 0 |
| 2 | 1 | 2 | .41/.40, -1.5 | .28/.23, -1.8 | -1.6 | +3.6 | 0 |
| 2 | 8 | 1 | .70/.69, 0.0 | **.92/.75, 0.0** | -1.5 | +1.8 | 0 |
| 2 | 8 | 2 | .78/.77, 0.0 | .80/.61, -0.2 | -0.9 | +1.8 | 0 |
| 3 | 1 | 1 | .40/.40, -2.7 | **.44/.44, -1.4** | -6.4 | +8.6 | 0 |
| 3 | 1 | 2 | .34/.34, -8.8 | .00/.00, -10.3 | -9.2 | +19.8 | 0 |
| 3 | 8 | 1 | .62/.62, 0.0 | **.80/.80, 0.0** | -7.6 | +8.2 | 0 |
| 3 | 8 | 2 | .59/.59, 0.0 | .04/.04, -4.4 | -6.0 | +11.9 | 0 |
| 4 | 1 | 1 | .38/.38, -48 | .03/.03, -86 | -33 | +123 | 0 |
| 4 | 1 | 2 | .28/.28, -73 | .00/.00, -228 | -44 | +269 | 0 |
| 4 | 8 | 1 | .73/.73, 0.0 | **.16/.16, -10.6** | -28 | +42 | 0 |
| 4 | 8 | 2 | .66/.66, 0.0 | .00/.00, -61 | -36 | +102 | 0 |

![acceptance per level](images/accept_per_level.png)

**Answer: the hypothesis holds, and more strongly than expected.**
- Once the fine levels are right, q-subtree proposals from probe lifts are accepted at L2 (0.79, or 0.92 with K = 8). They are also accepted at L3 (0.44, or 0.80 with K = 8), even though level 3's own channels are still at init.
- The L2 proposal gap d log q shrinks from -35 nats (cfast) to -1.4; at L3 it shrinks from -194 to -6.4.
- L4 remains hard: acceptance is 0.16 with K = 8, and the median dE is +42.
- τ = 2 hurts from probes at L >= 3.

Raw: runs/accept_kA_mlinit.json; script scripts/ground_accept.py (kernel-b's, with kernel-a's layout).

## 2. Coarse kernels (castlegen/kernels/coarse_flip.py)

**Flip move.**
- Pick a level-L cell. Propose its own sampled channels from a flip distribution with full support over the pass's least-violating set given the outside. Two modes: `uniform`, or `q` = the pass conditional tempered at τ_c.
- Then redraw the cell's whole subtree from q at τ.
- Accept with the exact multiple-try MH ratio. Both the proposal and its reverse density are flip density × subtree pass density, the reverse obtained by teacher forcing the old values. The energy comes from block_mh's closed-energy argument.
- Mode `q` with τ_c = τ reproduces block_mh.mh_move bit for bit (tested).
- Delayed acceptance is not built, for the same reason as kernel-b: the whole block pass is one scan, so a screen saves no compute.

**Exact single-cell coarse Gibbs.** This is ExactBlockGibbs(L, depth = 0), with the GPU version JaxBlockGibbs(L, 0). It was already generic over L, so `coarse_gibbs()` only builds it. The cell's channels are enumerated jointly: 2925 completions at L3 and 39 321 at L4, chunked. They are scored by every touching instance: count prior to parent and children, smoothness, honour certificates, support and top fill noise. The move is rejection-free.

**Tests.**
- tests/kernels/test_coarse_flip.py: 5 pass, together with kernel-b's 3. They check:
  - flip draw log q = teacher-forced log q, also for the flip part alone;
  - block dE = energy.py dE;
  - q mode = mh_move;
  - the L3/L4 Gibbs conditional on ground = full-energy differences.
- Toy invariance (scripts/toy_flip.py, 128 chains, 1000 moves per chain, λ = 20, L = 2, exact starts unless "own"):

| problem | q | kernel | K | acc | max abs z | note |
|---|---|---|---|---|---|---|
| toy_modes | bad | flip uniform | 1 | .063 | 1.16 | rho2 is pinned by a unary (rho2 = 8), so the flip is trivial and this tests only the subtree part |
| toy_modes | bad | flip uniform, **q ratio dropped** | 1 | .395 | **800** | power check |
| toy_modes | bad | flip q, τ_c = 2 | 1 | .063 | 1.16 | same pinned rho2 |
| toy_modes | bad | flip + Gibbs(2,0) | 1 | .063 | 1.16 | Gibbs changed 0 % (rho2 pinned) |
| toy_modes | untrained | flip uniform, own start | 8 | .627 | 1.25 | converged |
| toy_support | bad | flip uniform | 1 | .010 | 0.42 | weak: few moves accepted |
| toy_support | bad | flip + Gibbs(2,0), λ = inf | 1 | .010 | 0.42 | |
| toy_support | untrained | flip uniform | 1 | .051 | 0.86 | rho2 genuinely flips here (no unary) |
| toy_support | untrained | flip uniform, **q ratio dropped** | 1 | .090 | **153** | power check |
| toy_support | untrained | flip q, τ_c = 2 | 4 | .179 | 3.53 | see below |
| toy_support | untrained | flip uniform, own start | 4 | .204 | 3.53 | see below |

- The two runs at z = 3.53 both come from empty_block_frac, whose exact value is 5.5e-4. Each run saw a single empty block where about 4.5 were expected. Its SE then equals the estimate, so this z is degenerate and not evidence of drift. Every other stat in those runs is within |z| <= 1.24.
- A K >= 2 toy_support check with a frequent stat is still missing; kernel-b's MTM had a similar borderline z = 2.5 there.

## 3-4. Variant `ckernel` (castlegen/variants/ckernel.py) and the run

**Loop.** Each iteration takes 32 contexts: 16 probe lifts (jitter 0-3) and 16 own τ = 1 draws. Kernel cycle, all at λ = 20:
- 4 coarse flips per world, L uniform over {3, 4}, mode q with τ_c = 2, K = 4;
- exact single-cell Gibbs sweeps at L4, then L3;
- 4 MTM q-subtree moves per world at L2, K = 8;
- 3 cycles of exact fine Gibbs [(2,0), (1,1)].

Then 8 weighted-ML updates (B = 8, lr 1e-3, all levels; drawwise infeasible drop from amortised.py).

**Start.** sweep2a's arm-B params (runs/armB_s0): levels 0-2 trained by the fine loop, levels 3-4 at cstart2's init. So the bench's 300 s comes on top of sweep2a's 300 s fine-level run.

**Cost.** About 4.5 s per iteration: 3 s refine, 1.2 s ML. Setup takes 50 s warm and 270 s cold.

**Instrumented run (runs/ck300, 300 s, 440 updates, 55 iterations).**

| update | own dirt | own line_match | refined-own line_match | soft/cell own → refined | own viol/cell | infeasible draws/state | acc flip L3 / L4 / MTM L2 |
|---|---|---|---|---|---|---|---|
| 8 | 0.83 | 0.504 | 0.502 | 1.10 → 0.62 | 0.0187 | 8.1 | .60 / .32 / .87 |
| 104 | 0.75 | 0.517 | 0.516 | 0.65 → 0.44 | | 2.2 | .56 / .27 / .91 |
| 232 | 0.58 | 0.580 | 0.583 | 0.27 → 0.22 | 0.0025 | 0.41 | .53 / .23 / .88 |
| 424 | 0.52 | **0.619** | 0.619 | 0.20 → 0.18 | 0.0011 | 0.33 | .68 / .33 / .97 |

![own draws over training](images/own_draws_training.png)

- **Own-draw line_match rises above 0.5 for the first time (0.50 → 0.62).** Dirt falls 0.83 → 0.52, and own soft energy falls 1.10 → 0.20 per cell; probe-refined states sit at 0.14.
- Coarse single-cell Gibbs changes 90 % of L4 cells and 80 % of L3 cells per sweep. This is certificate and count slack.
- **The kernels barely move the own draws' line_match:** the refined value is within 0.007 of the raw one at every point. The pull toward the line therefore comes from the ML fit to the refined probe-context states. See the ablation below.

**Clamp gate.** L3/L4 fine cost per fine cell, fill-following vs own context (kernel-a's gate.py):

| model | L=3 fill | L=3 own | L=4 fill | L=4 own | L3-clamp line_match |
|---|---|---|---|---|---|
| cfast (board note) | 0.72 | 2.23 | 0.34 | 1.97 | |
| kernel-a ML | -0.107 | 0.46 | 0.71 | 0.84 | 0.770 |
| start (sweep2a arm B) | -0.106 | 0.346 | 0.683 | 0.725 | |
| **ckernel 300 s** | **-0.110** | **-0.084** | **-0.125** | **-0.107** | 0.739 |

- The own coarse contexts now cost about the same as fill-following ones; the gap is gone at both L = 3 and L = 4.
- Violations: 2e-4 per cell in the clamped fill context.

**Parent awareness** (fraction of draws changed, probe j0 vs j2 clamp, L = 3):

| model | rho2 | rho1 | t |
|---|---|---|---|
| probes | 0.079 | 0.037 | 0.016 |
| kernel-a ML | 0.037 | 0.028 | 0.013 |
| ckernel | 0.032 | 0.021 | 0.011 |

This is unchanged, at about half of the probes' rate. The fine levels were already saturated.

## Bench (v2, `python -m castlegen.bench run ckernel ground --seeds 1 --train 300`)

| variant | train s | updates | final viol (worst, level) | raw viol worst | F/cell | line_match | dirt | ρ per level 0/1/2/3/4 (pool) | starved % 0/1/2/3/4 |
|---|---|---|---|---|---|---|---|---|---|
| cfast (board, 300 s) | 300 | | 1.9e-5 | 2.7e-5 | 0.774 | ~0.50 | | 0.21/-0.30/-0.74/-0.70/-0.79 | 1/76/100/100/100 |
| **ckernel** | 300 (97 s setup) | 368 | **3.2e-3 (L3)** | 2.9e-3 | **-0.112** | **0.609** | 0.527 | **0.38/0.82/0.86/0.86/0.58** | **0/0.5/5.4/10/3.5** |

Run 20261014-081110_ckernel_ground; image images/bench/ckernel_ground_s0.png.

- **Not accepted, because of violations.** The worst final violation fraction is 3.2e-3 at L3. Per level, final viol is 3.5e-4 / 7.5e-4 / 1.5e-3 / 3.2e-3 / 2.0e-3, which is 100x cfast's.
- The probe check is new: ρ is positive at every level, and starvation falls from 76-100 % to at most 10 %.
- The free energy is 0.89 nats per cell below cfast's.
- The first bench attempt (runs/bench_cold) spent 274 s of the 300 s compiling, because the bench process uses another compile cache than my runner. It made 8 updates and is not a result. The rerun was warm, but setup still took 97 s.

## Ablation: own draws only (own_frac = 1, no probe contexts; the strict reading of the amendment)
runs/ck300_own: the same loop with 32 own draws per iteration and no probe contexts. Setup was 174 s, because the new shape compiled cold, so the run made 280 updates.

| update | own dirt | own line_match | refined line_match | soft own → refined | own viol/cell | acc flip L3 / L4 / MTM L2 |
|---|---|---|---|---|---|---|
| 8 | 0.84 | 0.504 | 0.503 | 1.10 → 0.62 | 0.018 | .57 / .56 / .80 |
| 136 | 0.90 | 0.502 | 0.502 | 0.67 → 0.51 | 0.006 | .38 / .55 / .86 |
| 264 | 0.90 | 0.501 | 0.501 | 0.56 → 0.47 | 0.005 | .58 / .42 / .80 |

- **Own-only stays collapsed.** Dirt stays at 0.90 and line_match at 0.50. In the main run at the same update count (264) they were 0.55 and 0.59.
- The kernels lower energy within the dirt mode. Soft energy goes 1.10 → 0.56, but they never leave that mode: kernel-b's exploration problem, now with coarse flips too.
- **So the line_match rise in ckernel is due to the probe-lift contexts, which ML fits at the coarse levels, not to the coarse kernels.**

## What failed / caveats
- **Violations.** ML carries no violation pressure beyond the refined states having fewer violations (λ = 20 kernels: own 1.1e-3 → refined 3e-4 per cell at the end). Starting with coarse levels at init means about 1.9e-2 per cell to work off, and 300 s takes it to about 1e-3. cfast reaches 1e-4 only through its λ controller.
- **Attribution.** The line_match gain comes through ML on probe-started chains, not through the coarse kernels moving own draws. Flips are accepted at 23-68 %, but they rarely change the dirt / line state of own draws.
- **Probe-started chains are not covered by the KL(qK ‖ p*) ≤ KL(q ‖ p*) argument.** The argument needs q-started chains. Probe contexts for ALL levels go beyond ideas #14 (fine levels on arbitrary parents) and are a proposal, not covered by the amendment, until the user rules on them. The ablation above is the strictly-covered variant.
- **Other.** One seed. The bench start is warm (sweep2a's 300 s). The GPU was shared. The toy flip tests are weak at K >= 2 on toy_support.

## Next
- Add the λ-controlled reverse-KL term (allowed: "may be mixed"), or price violations at λ = inf in the coarse kernels, to bring violations down to budget.
- Take the coarse levels' start from a short reverse-KL warm-up so that own draws begin with few violations.
- Put the coarse flip proposals on own draws only with a tempered flip at L4 (acc 0.23 now), so that the strictly-covered variant gets coarse movement.

## Files
- New: castlegen/kernels/coarse_flip.py, castlegen/variants/ckernel.py, tests/kernels/test_coarse_flip.py.
- reportables/2026-10-14_sweep2b/:
  - scripts/: ground_accept.py, common.py, toy_flip.py, run_toys*.sh, run_ck.py, plot_*.py, eval_ground.py / clamp_tools.py / gate.py (kernel-a copies), env.sh
  - runs/: accept_*.json, toy_*.json, ck300/, ck300_own/, bench logs
  - images/

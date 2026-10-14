# wave3-explore: persistent own-started chains move ground's own draws to the ground line

2026-10-14, agent wave3-explore. Problem: ground, LN+SiLU 6x6, one seed per arm. Start: sweep2a arm-B params, so levels 0-2 are fit and levels 3-4 are at init.

**Objective.** Weighted ML (weights 1) on all levels, on states produced by p*-invariant kernels at λ = 20 applied to the sampler's OWN τ = 1 draws (amendment 2026-10-14). **No probe lifts were used in training**; they were used only in the clamp gate. Generation is unchanged.

**Setup.** GPU shared with other agents. Cores 40-47. Each arm trained for 300 s after setup (setup 65-180 s). All arms ran on speed-1b's batched block_mh / coarse_flip / amortised code. Of wave2's arms, A3c also had all of it, A3b only the new block_mh / coarse_flip, and A1, A2, A3a mostly or entirely the old code.

## 1. Persistent chains (variant `ckernel_pop`, in castlegen/variants/ckernel_own.py)

**Population.** 32 states, initialised from own draws. Each iteration:
1. replace the oldest `refresh` fraction of the population with fresh own draws;
2. apply one kernel cycle to the whole population: exact coarse Gibbs L4, L3; tempered transitions at L4 and L3 (τ_hi 2, n 16, 4 moves each); MTM at L2 (K = 8); 3 cycles of exact fine Gibbs;
3. optionally, `cg_sweeps` × speed-2's colour-Gibbs sweep over all levels (single-site, λ = 20);
4. run 8 ML updates on the population.

Every member is an own draw followed by p*-invariant kernels; the chain just runs longer.

**Own-draw gates** use 64 fresh τ = 1 own draws of the final params (scripts/eval_own.py). The per-iteration own stats in the training logs come from only 1-3 fresh draws in P2/P3, so they are noisy.

| arm | updates | own dirt | own line_match (sd) | top-row B4 | own viol/cell | L3 fill / own | L4 fill / own | population dirt / line_match / valid |
|---|---|---|---|---|---|---|---|---|
| start | 0 | 0.851 | 0.507 (0.011) | 0.666 | 1.6e-2 | -0.106 / 0.346 | 0.683 / 0.725 | |
| wave2 A3a (same cycle, no persistence) | 384 | 0.88 | 0.50 | | 4e-3 | -0.100 / 0.102 | 0.090 / 0.120 | |
| P1 persistent, refresh 10 % | 784 | 0.709 | 0.513 (0.010) | 0.434 | 5.7e-3 | not run | not run | 0.68 / 0.52 / 0.07 |
| P2 persistent, refresh 3 % | 736 | 0.623 | **0.545** (0.024) | 0.210 | 4.6e-3 | -0.108 / -0.032 | -0.108 / -0.064 | 0.62 / 0.55 / 0.26 |
| **P3 = P2 + 20 colour-Gibbs sweeps per iteration** | 792 | **0.508** | **0.645** (0.030) | 0.000 | **9.4e-4** | **-0.112 / -0.111** | **-0.138 / -0.144** | 0.50 / 0.67 / **0.98** |
| P4 = P3 without persistence (refresh 100 %) | 904 | 0.556 | 0.574 (0.036) | 0.055 | 2.3e-3 | -0.111 / -0.085 | -0.127 / -0.118 | 0.55 / 0.58 / 0.19 |
| ref: sweep2b probe-context arm (out of bounds) | 440 | 0.52 | 0.619 | | 1.1e-3 | -0.110 / -0.084 | -0.125 / -0.107 | |

**Dirt by L4 block row (top to bottom).**
- Fill-implied: 0.00 / 0.12 / 0.88 / 1.00.
- P3 own draws: 0.00 / 0.12 / 0.91 / 1.00.
- Start: 0.60 / 0.87 / 0.95 / 0.99.

**Answers.**
- **The accepted moves do accumulate, and q follows.**
  - With persistence alone (P2), own-draw line_match leaves 0.50: 0.545 ± 0.003 SE over 64 draws, still rising at the end.
  - With persistence plus cheap single-site colour-Gibbs sweeps (P3), own draws reach dirt 0.51 and line_match 0.645, row by row on the fill.
  - The clamp-gate gap is gone: own context ≤ fill context at both L3 and L4, better than the out-of-bounds probe-context arm.
  - This is the first in-bounds own-draw result above 0.50.
- **Where the gain comes from.** Both parts contribute:
  - the colour-Gibbs sweeps alone (P4) give 0.574;
  - persistence alone (P2) gives 0.545;
  - both together (P3) give 0.645.
- **Refresh rate.** 10 % refresh (P1) is too fast at this budget.
- **The population leaves dirt through many small steps, not big moves.**
  - Tempered-transition dirt→partial acceptances stay at 0.8-2.5 % of pure-dirt proposals: L4 35/4123 in P3, 57/5219 in P2.
  - The population's top-row B4 falls from 0.8 to 0 (P3: by about 200 updates).
- **Violations are still too high for the bench.** P3's own draws have 9.4e-4 violations per cell, only 3 % of them valid. That is better than ckernel's 3.2e-3, but far above cfast's 1.9e-5. The population itself is 98 % valid. Not benched.

![population and own-draw curves](images/pop_curves.png)
![own draws](images/own_draws.png)
![population states](images/own_draws_population.png)

## 2. Certificate-aware moves (scripts/diag_tt.py, runs/diag_tt.json)

**What I could build within the time.** Tempering only the certificate factors is not separable inside `lay.cond`. But on ground **every hard factor is a certificate factor** (support B ≤ A below, row honour), while the count prior and smoothness are soft. So tempering only λ (τ = 1, λ_j geometric from 20 down to λ_hi, Neal's tempered transitions with an exact MH ratio; `make_tempered_transition(..., tau_hi=1, lam_hi=...)`) tempers exactly the certificate and support constraints. I could not keep support untempered while tempering honour, for the same reason. Tests: the λ-ladder invariance test is in tests/kernels/test_explore.py (7 pass).

**Per move**, 256 moves per row; "dirt→part" is accepted pure-dirt→partial over pure-dirt proposals; ms per move is contended:

| L | move | start | acc | median log a | dirt→part | ms/move |
|---|---|---|---|---|---|---|
| 4 | TT τ 1→2 (λ 20) | own | .94 | 0 | 7/109 (6.4 %) | 16 |
| 4 | TT τ 1→2 | P2 population | .22 | -2.9 | 2/107 (1.9 %) | 18 |
| 4 | λ only 20→5, n 32 | own | .82 | 0 | 9/111 (8.1 %) | 32 |
| 4 | λ only 20→5, n 32 | P2 population | .74 | -0.1 | 10/114 (8.8 %) | 32 |
| 4 | λ only 20→2, n 32 / n 128 | own | .23 / .40 | -25 / -6.6 | 0 / 0 | 30 / 27 |
| 3 | TT τ 1→2 | own / P2 | .83 / .54 | 0 / -0.7 | 4.9 % / 0 % | 3 |
| 3 | λ only 20→5 | own / P2 | .87 / .95 | 0 / 0 | 4.8 % / 2.5 % | 6 |
| 3 | λ only 20→2, n 128 | own / P2 | .38 / .36 | -2.3 / -1.7 | 1.4 % / 0.8 % | 100 |

- λ-only tempering down to 5 is the best per-move certificate move: about 8-9 % dirt→partial at L4, against 2-6 % for τ-tempering, at twice the cost.
- Going lower (λ → 2) is rejected because violations survive the cooling.
- **Candidate (a) was not built:** a bottom-up proposal with tempered-certificate exact fine Gibbs and coarse certificates drawn given the children, with its density tracked. Step 1 already moved own draws, and (a) needs a per-factor conditional that the layout does not expose.
- These moves were not put into a training arm; P3's success came from many cheap single-site sweeps on persistent chains, not from a large move.

**What a column-freeing move must change** (scripts/lift_count.py, exact lifts of a fine state and the same state with the top k tiles of one column cleared):

| k cleared | cells changed in total (levels 0 / 1 / 2 / 3 / 4) | cert B changed (L1-L4) | cert A changed |
|---|---|---|---|
| 1 | 9 (1/2/2/2/2) | 0 | 1 per level |
| 4 | 14 (4/4/2/2/2) | 2/1/0/0 | 2/1/1/1 |
| 16 (one column of a top L4 block) | 46 (16/16/8/4/2) | 8/4/2/1 | 8/4/2/1 |

- Freeing one column of an L4 block is a 46-cell, 5-level change in a tight lift.
- In own draws, which carry certificate slack, single-site steps can still take it one cell at a time:
  - lower A top-down;
  - clear tiles top-down, which the count prior favours because rho is already low;
  - lower B bottom-up.
- That sequence needs many sweeps, which persistent chains plus colour-Gibbs sweeps provide.

## Caveats
- One seed per arm.
- The per-iteration own-draw logging in P2/P3 used 1 fresh draw, so the final numbers come from 64-draw evaluations.
- P1's clamp gate was not run.
- The GPU was shared.
- Update counts differ by arm (736-904).
- Not benched: violations are over budget.

## Next
1. **Bench `ckernel_pop` with P3's settings** at 300 s, after adding violation pressure. Options: colour-Gibbs at λ = inf (hard) in the final sweeps, or the λ-controlled reverse-KL term (allowed: "may be mixed").
2. **Raise line_match from 0.645** by addressing surface roughness (column noise in the own draws), with more fine sweeps on the population.
3. **Check the step-1 result on dimers/columns** (breadth).

## Files
- castlegen/variants/ckernel_own.py: `CKernelPop` (`ckernel_pop`) added.
- castlegen/kernels/explore.py: λ ladder in `make_tempered_transition`.
- tests/kernels/test_explore.py: λ-ladder invariance test (7 pass).
- reportables/2026-10-14_wave3_explore/:
  - report.md
  - images/: pop_curves.png, own_draws.png, own_draws_population.png
  - scripts/: run_pop.py, eval_own.py, diag_tt.py, lift_count.py, plot_pop.py, render_own.py, peek.sh, gate copies
  - runs/: P1-P4 (loop.jsonl, params_ground.pkl, population.pkl, ground_summary.json), eval_own_*.json, diag_tt.json, lift_count.json, logs

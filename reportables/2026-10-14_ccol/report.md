# ccol: columns collapse timeline for cfast, bench v2 block check, clamp analogue (2026-10-14)

Agent ccol, cores 48-55, GPU shared with other jobs. One training seed (seed 0), cfast defaults (B = 8), columns (T = 4, so levels 0..4: fine k, coarse counts n1..n4). Diagnostics only: `ccol_train.py` subclasses csat's diagnostic (`csat_cols.py` is csat.py with its dirt measure `t` replaced by `k > 0`; no training or generation code changed). Diagnostic batch = 8 fixed worlds, every 25 updates, current tau and lambda. Target (exact): density 0.817, empty x-column fraction ~0.

## 1. Timeline (120 s training, 3476 updates; `images/ccol_timeline.png`)

![](images/ccol_timeline.png)

Entropy = mean entropy of q over non-forced draws (nats). Density / empty from the diagnostic worlds.

| update | tau | density | empty x-col | H L4 (top) | H L3 | H L2 | H L1 | H L0 (fine) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.0 | 0.56 | 0.00 | 5.55 | 3.25 | 1.91 | 1.15 | 1.51 |
| 50 | 1.83 | 0.50 | 0.003 | 5.34 | 2.81 | 1.55 | 0.94 | 1.46 |
| 100 | 1.76 | 0.38 | 0.002 | 4.17 | 2.57 | 1.42 | 0.34 | 1.33 |
| **125** | 1.72 | **0.064** | 0.13 | **0.16** | 1.78 | 1.15 | 0.75 | 0.88 |
| 150 | 1.67 | 0.055 | 0.33 | 0.00 | 1.53 | 0.63 | 0.10 | 0.71 |
| 300 | 1.46 | 0.060 | 0.61 | 0.00 | 0.20 | 0.33 | 0.15 | 0.89 |
| 1000 | 1.0 | 0.055 | ~0.7 | 0.00 | ~0.6 | 0.02 | ~0.1 | ~0.35 |
| 3350 | 1.0 | 0.055 | 0.57 | 0.00 | 0.89 | 0.01 | 0.00 | 0.17 |

First update (25-grid) at which a level's entropy is below 0.3 nats: **L4 at 125**, L1 at 150, L3 at 300 (it climbs back to 0.6-1.0 later and stays stochastic), L2 at 325, L0 at 1350 (and only to 0.17 at the end). Density: 0.38 at 100, 0.064 at 125, and **it never crosses below 0.05**: it settles at 0.054-0.060 (so "falls below 0.05" is not reached; it falls below 0.1 at update 125, between two grid points, together with L4). Empty-column fraction exceeds 0.3 at 150 and wanders 0.5-0.7 afterwards. The top level is the first to go deterministic, in the same 25-update window in which density falls from 0.38 to 0.06. Forced fraction at L4 is 0 throughout (nothing forces the top draw; it is just deterministic). Forced fraction at L0 rises to 0.97, at L1 0.9, L2 0.88 (the fine levels are forced by hard constraints once the context is air). Mean |b| at L0 grows steadily (2.2 at 125 to 10 at the end); lambda ended at [20, 0, 0, 0, 0] (L0 violations stay above eps: raw fine violation 0.19 in the bench baseline).

Caveat: 25-update grid, 8 diagnostic worlds, one seed. Early updates have tau > 1 (tau = 1 from ~700).

## 2. Bench v2 block check (cfast, columns, `--train 60 --seeds 1`, tag ccol, warmed; 3304 updates, GPU shared)

Final F = -0.045 (the all-air state has weight 1, F near 0; exact target is far lower), final violations 1.2 % (level 0), density 0.0625 against 0.817 exact, empty x-column fraction 0.53 against ~0. Not accepted. Block = 2x2 patch with one parent. Direct weight is constant on the coarse levels (hard count equalities), so direct rho is n/a at levels 1-4 and the subtree weight is the usable one (as in probe2).

| level | rho subtree: pool / probes alone / own alone | starved % probes | starved % own (null) | median probe deficit (nats) |
| --- | --- | --- | --- | --- |
| 0 | -0.48 / 0.00 / 0.57 (direct = subtree) | 44.3 | 0.0 | 0.03 |
| 1 | -0.41 / 0.94 / 0.02 | 43.9 | 0.0 | 3.5 |
| 2 | -0.66 / 0.47 / 0.58 | 98.3 | 0.0 | 32 |
| 3 | -0.69 / 0.83 / -0.31 | 99.9 | 0.02 | 12 |
| 4 | -0.81 / -0.08 / n/a (own draws identical) | 100 | 0.0 | 89 |

Reading. Levels 3 and 4 starve essentially every probe block (99.9-100 %, null 0-0.02 %), level 2 98 %, levels 0 and 1 about 44 % (worst level-0 deficits ~25 nats on the probe's bottom-edge blocks, median only 0.03: the fine level assigns low mass to a minority of probe blocks, mostly column ends). Pooled rho is negative at every level (Simpson: own and probe clusters sit on opposite sides), so read the probe/own columns; both are unreliable here because the own draws are nearly constant (all air). Weights of own and probe blocks barely overlap (caveat as in probe2), so the starvation percentages saturate. One seed, 60 s, bench v2 harness.

## 3. Clamp analogue (trained 120 s model; 32 seeds x 4 probes; `eval_columns.py`)

Levels >= L are replaced by the coarse counts of an exact-sample probe column state (lifted: n1..n4 are the cell counts per 2^l block), the levels below L are drawn by the model with the same noise as the own run. T = 4, so L = 4, 3, 2, 1 are all possible; L = 1 clamps only n1 and leaves level 0 (the fine k) free. Cost = soft + lambda * violations + tau * log q (trained lambda = [20, 0, 0, 0, 0], tau = 1; untrained lambda = 10 each), summed over the levels below L, per fine cell. Sanity: the stats pass reproduces `_creplay_clamp.clamped_pass` at L = 1 (35808.64 vs 35808.65 trained).

| model | L | fine cost / cell, own coarse | fine cost / cell, probe-lifted coarse | violations per cell (level 0), own / probe | density, own / probe | empty x-col, own / probe |
| --- | --- | --- | --- | --- | --- | --- |
| untrained | 4 | 1.39 | 1.58 | 0.246 / 0.306 | 0.58 / 0.87 | 0 / 0 |
| untrained | 3 | 1.42 | 1.68 | 0.246 / 0.319 | 0.58 / 0.88 | 0 / 0 |
| untrained | 2 | 1.49 | 1.77 | 0.246 / 0.321 | 0.58 / 0.89 | 0 / 0 |
| untrained | 1 | 1.58 | 1.94 | 0.246 / 0.330 | 0.58 / 0.90 | 0 / 0 |
| trained | 4 | 0.34 | **3.86** | 0.020 / 0.220 | 0.055 / 0.81 | 0.53 / 0 |
| trained | 3 | 0.35 | **4.34** | 0.020 / 0.243 | 0.055 / 0.81 | 0.53 / 0 |
| trained | 2 | 0.35 | **4.39** | 0.020 / 0.246 | 0.055 / 0.81 | 0.53 / 0 |
| trained | 1 | 0.35 | **4.35** | 0.020 / 0.244 | 0.055 / 0.81 | 0.53 / 0 |

Reading. The trained model's cost per cell rises 11-12x (0.35 to 4.35) when it is given the probe's counts, and almost all of it is level-0 hard violations: nv 0.24 per cell times lambda 20 = 4.9. The fine level does follow the counts in aggregate (density 0.81 against the probe's 0.817, no empty columns), so counts reach it, but it cannot turn them into valid columns: about one violated column constraint per 4 cells, which is as bad as the untrained model (0.25-0.33) or worse. The result does not improve from L = 4 down to L = 1 (4.35 at L = 1): n1 gives the count per 2x2 patch and no more, with no positional information, and the fine level has to place the column cells itself with nothing to tell it where column ends and gaps are. Level 0 in the own run is 97 % forced (air context), in the probe run it is 68-76 % forced.

## 4. Same shape as ground, or different?

**Partly the same, with one difference that matters.**

Same: the top level goes deterministic first (L4 below 0.3 nats at update 125, the same window in which density falls 0.38 -> 0.06; ground: cert4 at 200 with dirt reaching 0.95), and the collapse is to the cheap valid mode (all air here, all dirt there; the empty x-column fraction rises to 0.5-0.7). Levels 3, 2, 1 follow within 25-200 updates and the fine level ends with a high forced fraction (0.97) and a large learned bias (|b| 10), i.e. it never sees an occupied context after update ~150. The block check starves probes from level 2 up (98-100 %), and the same coarse-first ordering holds.

Different: (a) L1 goes deterministic at 150, not later, and L3 never really locks (0.6-1.0 nats at the end); on ground rho3 and rho2 went deterministic at ~350. (b) The fine level's failure is not context blindness alone: when the coarse counts demand occupied columns it does produce the right density but makes invalid columns (0.24 violations per cell), because columns' coarse channels carry only counts, no positions. On ground the clamped fine draws ignored the line (line_match 0.5 to 0.6) but did not violate hard constraints in the same way (not re-measured here). So the fine level of columns is under-informed even in good contexts (also true in the untrained model, 0.25-0.33 violations per cell at any L), which ground's fine levels were not shown to be. (c) Probe-lifted counts are not harder at lower L: the gap is flat from L = 4 to L = 1, so there is no depth at which the fine level is competent here.

## Files and conditions
`ccol_train.py` (train + log), `csat_cols.py`, `eval_columns.py`, `clamp_tools.py` (copy of csat's), `plot_ccol.py`, `run/train_columns.jsonl` (per-25-update diagnostics), `run/columns_summary.json`, `run/params_columns.pkl`, `images/ccol_timeline.png`. Bench row: `results/bench.jsonl`, tag ccol (a ccolwarm 25 s row precedes it; ignore). Cores 48-55, GPU shared; the 120 s run includes about 70 s of separate warm-up compile (training clock 120 s). All of this is one seed.

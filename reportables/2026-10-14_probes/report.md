# Probes: why toy_modes is flagged, why dimers loses the diagonal

2026-10-14, probes agent. GPU shared with other jobs (wall-clock budgets contended). Scripts in this folder: `probeA.py`, `probeB.py`, `probeB_fine.py`, `probeB_viol.py`. No harness or variant code changed. Params are not saved by the bench, so A retrains.

## A. toy_modes probe flags

Setup: cbig256, seed 0, 60 s training (1196 updates, fresh params, not the r4 run). 512 own raw draws and 512 probes (the problem's own `probes()`, exact samples from p), each scored with the variant's `score()`. Exact log p = sum over the 32 blocks of log p_block from the 2^16 enumeration.

| quantity | value |
| --- | --- |
| F per cell on own draws (F* = 0.2513) | 0.2548 (gap 0.0035) |
| own draws valid / q finite | 512 / 512 |
| probes with log q = -inf (unreachable) | 0 / 512 |
| est. KL(q\|\|p) per world (per cell) | 1.83 (0.0036) |
| est. KL(p\|\|q) per world (per cell) | 2.23 (0.0043) |
| d = -S - log q, own: mean, std | -130.47, 1.75 |
| d, probes: mean, std | -126.42, 2.45 |
| shift probes - own | 4.06 = 1.83 + 2.23 (exactly KL(q\|\|p)+KL(p\|\|q)) |
| fraction of probes above own 99th pct (512 vs 512) | 0.447 |
| same, 16 own vs 16 probes (bench size), resampled | P(flag) = 1.00, mean frac 0.66 |
| same, probes replaced by exchangeable own draws, 16 vs 16 | P(flag) = 0.53 |

Figure: `probesA.png` (histogram of d, and d against probe exact log p).

Verdict on the three hypotheses:
1. **Unreachable probes: no.** 0 of 512 have log q = -inf.
2. **Tiny-mass probes: not the cause, a minor modifier.** Probes are exact samples from p, so they are typical states (per-block exact log p from -2 to -17; only 6.8 % of probe blocks below -9). The lowest-mass quartile of probes is flagged more often (58 % above threshold) than the highest-mass quartile (22 %), corr(d, log p) = -0.32, but even the best quartile is far above the 1 % tail.
3. **Scoring bug: no.** The probe-minus-own shift of d equals KL(q||p)+KL(p||q) computed independently from exact log p to 3 digits (4.06). score() and the generation log q agree (own draws give F = 0.2548 through score()).

What holds instead: **the test is a world-level likelihood-ratio test and is very powerful.** d = log(p/q) + const, so own draws sit at -KL(q||p) and probes at +KL(p||q) with spread sqrt(2 KL) over 32 blocks. A near-optimal q (0.0035 nats per cell off) is 2.3 sigma away from p at world level. Gaussian estimate: the flag (>5 % above the 99th percentile) stays off only when KL is below about 0.25 nats per world, about 0.0005 per cell, which means F within about 0.0005 of F*. The flag is therefore **real but not informative about usability**: it says q is not p to better than 0.0005 nats/cell, and no round-4 run was that close.

Second, independent weakness: at the bench's n_eval = 16 the check has 16 own valid draws and 16 probes, so the "99th percentile" is essentially the maximum of 16 and a single probe above it gives 6.25 % > 5 %. Even exchangeable draws trip the flag 53 % of the time.

Suggestion (proposal only; harness frozen): use the same sample sizes of at least 200 each, or compare the means via KL(q||p)+KL(p||q) with a threshold stated in nats per cell, so that "flagged" separates a 0.003 from a 0.03 gap.

## B. dimers: the missing diagonal

Orientation fractions (horizontal / vertical / diagonal), mean over 8 held-out raw worlds (final draws in brackets for the untrained rows). -log q per cell is the entropy estimate from score().

| sampler | h | v | diag | -log q/cell |
| --- | --- | --- | --- | --- |
| greedy (untrained) | 0.364 | 0.318 | 0.318 | n/a |
| cfast lr = 0 (untrained, tau = 1) | 0.364 [0.361] | 0.318 [0.321] | 0.318 [0.318] | 0.781 |
| cfast after 150 s (5475 updates) | 0.523 | 0.476 | 0.0004 | 0.225 |

**The diagonal is present untrained** (0.318, same as greedy; at lr = 0 the head output is zero so draws equal greedy's). There is no structural obstruction in the colour order, lookahead or declaration: the untrained violation rate per hard factor is the same for the E, S and SE offsets (5.2 / 5.2 / 5.3 %), and the three orientations are drawn about equally. It is lost in training.

Trajectory (`dimers_orientation.png`, `B_cfast_fine.json`, `B_cfast.json`). Fine run: 8 s budget, so tau anneals fast; first update is at t = 2.8 s after compile.

| update | diag | vertical | horizontal | -log q/cell | raw viol |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.296 | 0.336 | 0.368 | 0.779 | 0.0386 |
| 5 | 0.142 | 0.491 | 0.367 | 0.692 | 0.0361 |
| 10 | 0.041 | 0.588 | 0.371 | 0.490 | 0.0247 |
| 15 | 0.014 | 0.634 | 0.352 | 0.358 | 0.0148 |
| 20 | 0.005 | 0.694 | 0.301 | 0.268 | 0.0080 |
| 24 | 0.002 | 0.765 | 0.233 | 0.209 | 0.0053 |
| 82 (150 s run, tau 1.86) | 0.0007 | 0.649 | 0.351 | 0.240 | 0.0068 |
| 432 | 0.0001 | 0.586 | 0.414 | 0.219 | 0.0001 |
| 617 | 0.0001 | 0.502 | 0.498 | 0.223 | 0.0000 |
| 5475 | 0.0004 | 0.476 | 0.523 | 0.225 | ~0 |

- The diagonal is gone within about 20 updates (about 3 s), before the violations are removed, and the sampler's entropy falls from 0.78 to 0.21 nats per cell in the same window. So it **coincides with entropy collapse**: it is the first thing the head removes, then it slowly rebalances h/v to 0.5/0.5 over the next ~600 updates as violations go to 0.
- Final -log q is 0.225 nats per cell, roughly half the triangular-lattice dimer entropy (about 0.43 per site, my recollection, not verified here); the sampler is in a two-orientation low-entropy mosaic with zero violations.
- Both bench seeds of the earlier cfast run also had diagonal = 0.000, so it is reproducible, not a symmetry-breaking accident.

What I did not determine: why the diagonal is the orientation removed rather than h or v. The measured untrained statistics are symmetric across the three offsets, so the asymmetry has to come from the learned head (actor window, features, or the cone returns). Untested guesses: the diagonal pair (dy, dx) = (1, 1) is the only one whose partner lies in both a different row and a different column colour, so its costs sit furthest apart in the draw order; or the head learns "avoid the diagonal" as the cheapest feature direction to cut violations. A cheap next test is the same 8 s run with the diagonal's table entries permuted (relabel SE as S) and see whether the loser follows the geometry or the label; that needs a problem variant and was out of scope here.

## Failures and caveats
- A retrained cbig256 (60 s, one seed), not the r4 run's params (the bench does not save them). F gap 0.0035 matches the r4 range (0.003).
- B used one seed of cfast, 8 worlds per measurement; the generate calls inside the training loop slightly slow the training clock.
- Gaussian threshold estimate (0.25 nats per world) is analytic, not simulated.

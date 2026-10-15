# rgap: exact bound gap, uniform vs designed refill r

**Question.** With the designed fixed r(a | x_notM) ∝ exp(-s(a) - lam_r nv(a)), is the path-KL bound gap below a third of the uniform gap and below ~0.3 nats per cell, so a revising sampler is not taxed into determinism?

**Answer: no.** For generators that keep entropy the designed r removes only 54-59 % of the gap (ratio 0.41-0.46 at best lam_r 2-5; lam_r 20 is no better than uniform). Trained generators collapse to deterministic anyway. The per-cell 0.3 criterion is met by uniform too on these binary toys, so it does not discriminate.

## What ran
`castlegen/variants/_rgap.py` (copy of the enumeration that also accumulates -log r_des(old | x_notM) of every erased cell, cond on the state with the step's whole erased set unknown; C' = C - |M| log D + sum -log r_des). Driver `scripts/run_gap.py <flat|two> <coin seed> <train secs>`, `scripts/small_lam.py`, `scripts/logz.py`. CPU `taskset -c 0-3`, `JAX_PLATFORMS=cpu`. tau = 1. Flat: 2x8 binary, R = 1, rho 0.5, guided, lam 20, coins 3 (6 erased, 10 502 traj) and 1 (7 erased, 12 531). Two-level: R = (0,1), coins 2, 1 erased coarse cell, 119 814 traj, lam (2,3). Uniform numbers reproduce the crevise report (3.124, 3.562). Enumeration 10-40 s, training 60 s per generator (Adam lr 0.05, exact gradient, 3-33 steps), whole thing ~10 CPU-minutes.
Generators: untrained; trained 60 s by exact path KL with uniform r; trained 60 s with designed r (lam_r 10; extra). (b) exact-p* generator: not constructible (the actor only adds a bias to the lookahead-masked logits), not done.

## Table (gap = J_path - reverse KL(x_0), nats per world; per cell = /16 flat, /20 two-level; charge in brackets: |M| log D for uniform, E[-log r(old)] for designed)
| problem | generator | reverse KL(x_0) (no log Z) | uniform | des lam_r=2 | des 5 | des 10 | des 20 |
|---|---|---|---|---|---|---|---|
| flat c3 | untrained | 0.151 | 3.124 (4.159) | 1.266 | 1.323 | 1.913 | 3.123 |
| flat c3 | trained uniform | -0.002 | 4.159 | | 4.413 | 4.404 | 4.404 |
| flat c3 | trained des10 | -2.400 | 4.159 | | 0.923 | 0.913 | 0.913 |
| flat c1 | untrained | 3.291 | 3.562 (4.852) | 1.634 | 1.656 | 2.441 | 4.060 |
| flat c1 | trained uniform | -1.800 | 4.852 | | 1.316 | 1.289 | 1.289 |
| flat c1 | trained des10 | -2.400 | 4.852 | | 1.316 | 1.289 | 1.289 |
| two c2 | untrained | -6.954 | 0.259 (1.099) | | 0.046 | 0.045 | 0.045 |
| two c2 | trained uniform | -2.528 | 0.554 | | 0.520 | 0.520 | 0.520 |
| two c2 | trained des10 | -7.744 | 0.379 | | 0.197 | 0.197 | 0.197 |

Per cell (flat, /16): uniform 0.20-0.22 untrained, 0.26-0.30 trained; designed lam_r 2-5 untrained 0.08-0.10; trained-designed 0.06-0.08. lam_r 0 (unary only) is worse than uniform (3.73, 4.63). Brute-force flat problem (lam 20): min E -2.4 (2 states), log Z 6.85, so the optimal reverse KL is -6.85 and the deterministic min-energy state (-2.4) is 4.4 nats from it.

## Reading
- Broad generator (untrained, H(x_0) ~5): designed r at best 0.41-0.46 of uniform, above one third. Large lam_r makes it worse: an old value that violated at erase time costs ~lam_r, so lam_r 20 equals uniform.
- Both short trainings (30 steps, lr 0.05, probably too hard) went deterministic (H(x_0) = 0), with uniform r (reverse KL -0.002 / -1.80, not even the min energy) and with designed r (-2.40, the min-energy state). A deterministic generator pays the full charge; the designed charge is 0.9-1.3 there (0.2-0.3x of uniform) only because the pass already was locally consistent. Whether the optimum of J' keeps entropy was not tested (not a converged optimisation, one seed each).
- Two-level has one erased cell, so 0.045-0.52 are single-cell numbers.

## Failed / not done
Exact-p* generator (b); converged training; D > 2 (dimers D = 6 only via the formula); the designed-r training collapse is not shown to be the tax versus optimiser speed.

## Next
None for this r; if the idea continues, a lower-tax r shape (e.g. including the draw's own lookahead) would need a new exact-gap check first.

# Coloured-pass benchmark harness

The fixed yardstick for sampler variants. Goals and validity rules are in the
constitution doc ("Coloured-pass sampler: constitution",
https://claude.ai/code/artifact/f76368c5-8d7c-44b3-86b8-f2292232e69a); the
starting design is the "Coloured-pass sampler: final implementation spec".

## What is frozen

`castlegen/bench/` (problem format, noise, energy, variant interface, problem
suite, runner, leaderboard) is frozen. Variants import it and never edit it.
Every result records a hash of these files, so results from a modified harness
are visible. Changes to the harness or a problem need the user's approval and
a version bump.

Variants live in `castlegen/variants/<name>.py` (one module per variant, plus
private helper modules prefixed `_`) and may use anything else.

## World, regions and samples

- A strip of `H` fine rows, infinite in x; rows grow downward. Levels 0
  (fine) to T (coarse); a level-l cell covers 2^l x 2^l fine cells; 2 x 2
  refinement, child position k = 2 (Y & 1) + (X & 1).
- `Region(W)`: periodic strip of circumference W (training, scoring, metrics).
  `Region(W, periodic=False, x0)`: window of the infinite strip (finite-support
  test). W and x0 are multiples of 2^T.
- A sample is `{channel: int array (H >> l, W >> l)}`.

## Problems (declarations, not code)

Channels with views; unary, pair (same level or parent, optional
position-in-parent table, pads for rows off the strip, optional confinement to
a level-L block) and count (sum of a child view over a subset of child
positions) factors; tables over views; inf = hard. Designed channels are fixed
functions of absolute coordinates. See `castlegen/bench/problem.py`.

Suite v1: `toy_support`, `toy_modes` (enumerable, exact answers), `ground`
(the certland target), `dimers` (triangular lattice, hard, no slack),
`columns` (tall columns, exact per-column answers). Trees and curves wait on a
certificate design reviewed by the user. All targets are "easy": short-range
correlations, no criticality, no phase coexistence.

## Variant interface

`castlegen/bench/variant.py`: `train(budget_s, log)`, `generate(seed, region)
-> {"raw", "final"}`, `score(x, region) -> log q | None`, `info()`.
All generation noise comes from `castlegen.bench.noise`, keyed by absolute
coordinates.

## A bench run

`python -m castlegen.bench run <variant> <problem> [--seeds 2] [--train 150]`

Per training seed: construct, `train(budget)`, then on held-out noise seeds:

1. **Violations**: raw and final violated fraction of hard instances per level
   on `n_eval` periodic worlds.
2. **Free energy** (when scorable): mean over raw draws of S(x) + log q(x) per
   fine cell, the reverse-KL objective at tau = 1 up to a constant. Lower is
   better.
3. **Stats**: the problem's per-world statistics, raw and final, mean and
   standard error; compared with exact values where the problem has them.
4. **Finite support**: two overlapping open windows from the same seed must
   agree exactly on their overlap, at every level, raw and final.
5. **Probes** (when scorable; the old flag, kept): d(y) = -S(y) - log q(y) of valid probe states
   against the 99th percentile of d over the variant's own valid draws.
6. An image of one final world in `images/bench/`.

Across seeds: stats must agree within tolerance. Verdict = the constitution's
acceptance: final violations under 2e-4 at every level, exact stats matched,
no probe set flagged (more than 5 % above the 99th percentile), stats agree
raw vs final and across seeds, finite support holds.

Results append to `results/bench.jsonl`; `python -m castlegen.bench board`
writes `results/board.md` (latest run per variant x problem).

## Budgets

A bench run takes 5-10 minutes on the Mac (default 2 seeds x 150 s training
plus evaluation), single-threaded, at most 2-3 runs at once. No sweeps or
longer runs without a stated, very good reason.

## Version 2 (2026-10-14): block check

User-approved harness change, version 1 -> 2 (`BENCH_VERSION` in `run.py`; the
harness hash changes, so every older board row is marked "(old harness)").  The
v1 probe flag is a world-level likelihood-ratio test and trips on exchangeable
draws about half the time at n = 16 vs 16
(`reportables/2026-10-14_probes/report.md`).  It stays on the board for now;
the **block check** is added beside it and is **not an acceptance gate yet**.

- New optional `Variant.score_sites(x, region) -> {channel: (h, w) per-draw log q}`
  (sums to `score`; the bench checks this on the first world and records n/a if not).
  Implemented for the cpass family (cpass, cfast, cbig, ...) and greedy; others: n/a.
- `castlegen/bench/shares.py`.  A block at level l is the 2 x 2 patch of level-l sites with
  one parent; at the coarsest level T the same rule gives 2 x 2 top cells (as if a level T+1
  existed).  Log q share = sum of the block's per-draw log q.  Target log-weight share =
  -soft energy - 10 * violated hard instances (LAMBDA_SCORE = 10, cpass lam0) of the factors
  homed at level l; each instance goes to the block of its finer-side cell, a same-level
  pair across a block boundary gives half to each block ("direct").  Because direct weight is
  constant at levels whose factors are hard equalities (toy coarse levels), a "subtree" weight
  (direct share plus those of all blocks below) is reported in brackets.
- Pool: 128 own raw worlds and 32 probe seeds (all worlds of `problem.probes`).  Per level:
  Spearman rho pooled, on probes alone, on own alone (None when a share is constant);
  starvation = fraction of probe blocks whose log q share lies more than 5 nats below the
  own-draw Theil-Sen line (log q share on weight) at their weight, plus the same fraction of
  own blocks (a null) and the five worst probe blocks (level, block, deficit).
- Board: two new columns, rho per level (pooled) and starved % per level, levels 0 / 1 / ... / T,
  direct [subtree].

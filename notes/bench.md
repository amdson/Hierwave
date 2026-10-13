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
5. **Probes** (when scorable): d(y) = -S(y) - log q(y) of valid probe states
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

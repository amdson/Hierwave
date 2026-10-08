### Stage 6b: the Potts leak with collapsed targets

The question: was the Potts leak (S1's top tables at ~0.7 of S0's,
notes/potts_test.md "top_h / top_v: the leak") a target error and not a
context error?  S1's top target is p*(T | mid), a one-site conditional that
reads the level below T.  The target a top-down sampler needs, since it draws
T before its mid cells, is the collapsed conditional: T's mid cells (and
their tiles) integrated out.

Script: `notes/experiments/potts_collapsed.py`.  Output:
`images/potts_collapsed.json`, `images/potts_collapsed_tables.png` (top_h
double-centred per target type next to S0's published table).  Whole run
156 s on one core.

Config: nty = ntx = 6 (12 x 12 mid, 48 x 48 tiles), lam = 3, kappa 8 / J 0.1
and kappa 1 / J 0.3.  All fits are `train.fit`, iters 5, n_contexts 8, l2
1e-4, tabular feats, holdout 0.1.  Contexts are fresh forward runs (30 / 30 /
20 sweeps) at the current tables, and they share their Channel objects with
an `Oracle`.

- Mid: `ExactTargets(Oracle.mid_probs)`, K = 0, top tables 0 (stage 6
  setup a), refitted at both settings.  kappa 8: mid_h by d = -0.398 /
  +0.001 / +0.396 (reference BM J d = -0.40 / 0 / +0.40).  kappa 1: -0.295 /
  +0.018 / +0.259 (published S1 -0.29 / +0.02 / +0.26, S0 -0.31 / +0.02 /
  +0.26).  The mid tables are then fixed for every top fit.
- Top: top_h, top_v and top_u are fitted with the fitted mid tables
  installed.  The designed offset is 0, because no designed factor is homed
  on top (`designed_energies(orc.model, "top")`).  The learned top tables
  enter only through the fit's X theta, as in biome_fit.  Target types:
  - **frozen**: `ExactTargets(softmax(-site_energies("top", below=True)))`,
    i.e. S1's p*(T | mid) = softmax(-lam #(own mid cells outside palette)).
  - **ais**: `AISTargets(child "mid", region = the cell's 2 x 2 mid block,
    designed_e 0, after_change none, K 32, M 16, L 30)`.  p_0 holds lam (the
    parent row) and mid_u; mid_h / mid_v are annealed.  As in the circles
    run, the region is the written cells only.  The neighbouring mid cells
    are a fixed boundary, with their mid_h / mid_v rows into the window
    kept.
  - **hook**: the same window, log Z computed exactly by a column transfer
    matrix over the window (`window_logz`, checked against brute force).
  - **hook4** (extra): the window dilated by the mid tables' reach (one mid
    cell), so 4 x 4 cells, exact.  The fixed boundary then sits at distance
    2 from T's cells, and the ring cells keep their lam rows to the
    neighbouring tops.
  - K = 3 (frozen and hook): T is drawn from the target, then
    `Oracle.mid_move` runs at its 4 mid cells (a collapsed draw given T and
    the boundary tiles, with the block's tiles redrawn).
- Eval:
  - Forward L1 to the oracle moments (50 + 400 sweeps, symmetrised as in
    potts_train.py, whose `symmetrise` is copied because that script runs on
    import), over two independent 64-run evals (L1 a / L1 b).
  - eval noise = L1 between those two evals.
  - edge_same_top (held out).
  - The fit's final `viol_targets` (the consistency of the targets; the
    fit's own is about 1e-15 everywhere).
  - tau of the mid kernel with top clamped (`autocorr_clamped(fw.model,
    "mid", ["top"])`, at the final tables).

#### Contrast (double-centred learned top tables; one run each)

| setting | target | K | disjoint - same (h / v) | one-colour - same (h / v) | ratio to S0 | edge_same_top (eval a / b) |
|---|---|---|---|---|---|---|
| kappa 8, J 0.1 | S0 published | | 0.73 / 0.71 | 0.31 / 0.32 | 1 | .397 (oracle .395) |
| | S1 published | | 0.50 / 0.53 | 0.25 / 0.26 | 0.72 | .381 |
| | S3 published | 3 | 0.67 / 0.70 | 0.30 / 0.29 | 0.95 | .392 |
| | frozen | 0 | 0.26 / 0.05 | 0.09 / 0.07 | 0.22 | .359 / .369 |
| | frozen | 3 | 0.58 / 0.62 | 0.28 / 0.25 | 0.83 | .396 / .381 |
| | ais | 0 | 0.70 / 0.67 | 0.31 / 0.30 | 0.96 | .392 / .390 |
| | hook | 0 | 0.71 / 0.68 | 0.31 / 0.30 | 0.96 | .392 / .392 |
| | hook | 3 | 0.71 / 0.72 | 0.30 / 0.31 | 0.99 | .394 / .392 |
| | hook4 | 0 | 0.72 / 0.72 | 0.31 / 0.32 | 1.00 | .391 / .392 |
| kappa 1, J 0.3 | S0 published | | 0.60 / 0.58 | 0.26 / 0.26 | 1 | .365 (oracle .365) |
| | S1 published | | 0.38 / 0.39 | 0.20 / 0.19 | 0.65 | .356 |
| | S3 published | 3 | 0.52 / 0.52 | 0.22 / 0.24 | 0.88 | .361 |
| | frozen | 0 | 0.00 / -0.20 | -0.06 / -0.09 | -0.16 | .350 / .354 |
| | frozen | 3 | 0.19 / 0.42 | 0.09 / 0.11 | 0.52 | .355 / .355 |
| | ais | 0 | 0.50 / 0.50 | 0.23 / 0.23 | 0.85 | .359 / .359 |
| | hook | 0 | 0.50 / 0.50 | 0.23 / 0.23 | 0.85 | .359 / .359 |
| | hook | 3 | 0.49 / 0.49 | 0.22 / 0.23 | 0.83 | .358 / .360 |
| | hook4 | 0 | 0.50 / 0.51 | 0.23 / 0.23 | 0.85 | .359 / .359 |

The ratio to S0 is (disjoint h + v) / (S0's published h + v).  The
contrast is invariant under double-centring, so these numbers compare
directly with potts_test.md's.

#### L1 to the oracle moments (64-run eval a; eval b in brackets for the top keys)

| setting | target | mid_h | mid_v | mid_u | top_h | top_v | top_u |
|---|---|---|---|---|---|---|---|
| kappa 8 | frozen K0 | 0.133 | 0.158 | 0.107 | 0.215 (0.214) | 0.300 (0.227) | 0.118 (0.098) |
| | frozen K3 | 0.111 | 0.116 | 0.073 | 0.168 (0.186) | 0.187 (0.163) | 0.100 (0.095) |
| | ais K0 | 0.051 | 0.054 | 0.034 | 0.096 (0.131) | 0.077 (0.093) | 0.043 (0.075) |
| | hook K0 | 0.045 | 0.051 | 0.026 | 0.082 (0.125) | 0.070 (0.089) | 0.038 (0.065) |
| | hook K3 | 0.042 | 0.040 | 0.023 | 0.067 (0.104) | 0.050 (0.078) | 0.030 (0.045) |
| | hook4 K0 | 0.038 | 0.041 | 0.019 | 0.073 (0.077) | 0.058 (0.054) | 0.031 (0.024) |
| | eval noise (range over runs) | 0.06-0.09 | 0.06-0.08 | 0.03-0.06 | 0.10-0.16 | 0.09-0.11 | 0.03-0.07 |
| kappa 1 | frozen K0 | 0.114 | 0.145 | 0.079 | 0.238 (0.224) | 0.331 (0.245) | 0.108 (0.069) |
| | frozen K3 | 0.121 | 0.119 | 0.086 | 0.204 (0.187) | 0.146 (0.112) | 0.090 (0.060) |
| | ais K0 | 0.097 | 0.100 | 0.071 | 0.134 (0.111) | 0.119 (0.093) | 0.073 (0.053) |
| | hook K0 | 0.099 | 0.101 | 0.073 | 0.134 (0.110) | 0.123 (0.092) | 0.073 (0.051) |
| | hook K3 | 0.095 | 0.099 | 0.070 | 0.121 (0.103) | 0.105 (0.083) | 0.069 (0.030) |
| | hook4 K0 | 0.100 | 0.103 | 0.077 | 0.135 (0.107) | 0.122 (0.077) | 0.076 (0.028) |
| | eval noise (range over runs) | 0.06-0.07 | 0.06-0.09 | 0.03-0.04 | 0.08-0.15 | 0.08-0.11 | 0.04-0.06 |

Published S0 for scale: kappa 8 top_h / top_v 0.081 / 0.073, kappa 1 0.087 /
0.071.

#### Diagnostics

Consistency violation of the targets (final iteration, max over probes):

- frozen and hook: 1e-15.  These targets do not depend on the neighbouring
  top at all.  frozen reads only T's own mid cells; the 2 x 2 hook reads
  the fixed neighbouring mid cells, never the neighbouring T.  So the
  closure check is trivially passed.
- ais: 0.32 (kappa 8) / 0.28 (kappa 1).  This is AIS noise around the
  hook's 0: the closure sums 8 log-probabilities.
- hook4: 0.35 / 0.19, exact, so this is a real inconsistency.  The ring
  cells read the neighbouring tops, and windowed conditionals with a frozen
  boundary at distance 2 are not conditionals of one joint.  It costs
  nothing in the learned tables.

The fit's own violation is about 1e-15 throughout.  tau(mid | top clamped)
is 1.0-1.4 for every run (the mid kernel mixes in about one sweep).

AIS vs the exact hook, on a fresh context at the final tables, over all 36
cells x 4 candidates:

| setting | dlogZ mean | max abs dlogZ | AIS se median / p90 / max (during the fit) | within 3 se | target L1 mean / max | ms per AIS run | fit seconds (AIS / hook) |
|---|---|---|---|---|---|---|---|
| kappa 8 | +0.003 | 0.104 | 0.032 / 0.047 / 0.089 | 98% | 0.024 / 0.053 | 0.93 | 15.7 / 1.4 |
| kappa 1 | +0.004 | 0.063 | 0.024 / 0.035 / 0.067 | 98% | 0.016 / 0.032 | 1.00 | 16.9 / 1.3 |

#### What the numbers show

(a) **The collapsed target closes the leak at K = 0 at kappa 8.**  With T's
mid cells integrated out under lam and the installed mid tables, the top
contrast is 0.96 of S0 by AIS and 0.96 by the exact hook.  The dilated
exact window gives 1.00.  The fitted-feature L1s are at or below eval noise
on every key, and edge_same_top is .392 against oracle .395 and S0 .397.
For comparison, S1 published 0.72 and .381, and S3 (K = 3 end-to-end)
published 0.95.  K buys nothing: hook K3 gives 0.99 against hook K0's 0.96,
inside the +-0.05 spread between entries that are equal by symmetry.

(b) **At kappa 1 the collapsed target gives 0.85, against S1's 0.65 and
S3's 0.88.**  ais, hook and hook4 agree to 0.01 and K = 3 gives 0.83, so
the 0.15 that remains is not in the top target's window or in the AIS.  It
is likely in what the top target integrates against, the mid tables:

- At kappa 1 the top-top coupling is induced partly through tile
  correlations across the top edge that a pairwise mid table carries only
  in projection.
- The mid tables' own target `Oracle.mid_probs` conditions on the boundary
  tiles of the neighbouring blocks.
- The mid L1s at kappa 1 sit at 0.10 against noise 0.07 in every variant,
  while at kappa 8 they are at noise.

This attribution was not separately tested.  edge_same_top is .359 (oracle
.365, S1 .356, S3 .361).

(c) **The leak was a target error.**  The context is the same in every row:
forward runs at the current tables, one pass, K = 0.  Only the target
changes.  frozen to collapsed moves the contrast from 0.22 to 0.96 at kappa
8 and from -0.16 to 0.85 at kappa 1.  Fitted by train.fit's
pseudo-likelihood regression, the frozen target is much worse than
published S1's RB moment matching (0.22 / -0.16 against 0.72 / 0.65), and
noisy (h / v 0.26 / 0.05).  The reason: p*(T | mid) is nearly a one-hot on
the current T, since the mid cells were drawn from that T.  The regression
therefore returns the forward's own q(T | neighbours), a fixed point at any
theta, and the fit stays near its start.  K = 3 with collapsed mid moves
lets the mid cells move under T and recovers part of it (0.83 / 0.52), as
S3 did.  The collapsed target needs none of this.

(d) **AIS against the exact hook at the top.**  The two give
indistinguishable tables (contrast 0.70 / 0.67 against 0.71 / 0.68 at kappa
8; identical at kappa 1).  log Z agrees to 0.003 on average, with 98% of
values within 3 se, a median se of 0.02-0.03 at K = 32, M = 16, and
per-site target L1 of 0.02.  AIS costs 1 ms per run, 11x the hook's fit
time (16 s against 1.4 s per fit); both are negligible.

(e) **The window.**  The circles convention (the written cells only, the
neighbouring mid cells a fixed boundary through the annealed rows) is
enough.  The 2 x 2 window's target never reads the neighbouring tops.  The
top coupling is learned through the forward's correlation between a
neighbouring top and its mid cells on the shared edge.  It is
self-consistent (violation 0) and matches S0 at kappa 8.  Dilating by the
mid tables' reach reads the neighbouring tops directly through the ring's
lam rows.  That moves 0.96 to 1.00 at kappa 8 and nothing at kappa 1, at
18-25x the hook's cost, and it makes the targets mildly inconsistent.

(f) **Consequence for the design docs** (notes/dsl_updates.md C2, the
targets of reference_math section 4).  The target at a site must integrate
out everything below the site's level that the sampler draws after the
site, including the site's own children at the home level's child level.
A one-site conditional of p* that reads a finer level is the frozen-site
target:

- p*(T | mid) here.
- The plain mid conditional with the tiles fixed (S1f).
- p*(T | slots) on the biome circles.

It reproduces the leak, and under train.fit's regression it does worse than
leak: the fit stalls at the forward's own conditional.  The fix is in the
target, not in the contexts or K.  The collapsed target over the child
sampler with its installed learned potential, the parent row in p_0, and
the window's written cells with the neighbouring child cells as a fixed
boundary, is enough at K = 0.  An exact hook, where one exists, is a cheap
drop-in for AIS.  What it cannot fix is error already in the child level's
learned potential: that propagates up, which is the kappa 1 residual.

Deviations from the brief:

- Mid tables were refitted at both settings.  images/bootstrap_potts.json
  holds kappa 8 only, and the refit matches it: -0.398 / +0.001 / +0.396
  against -0.398 / +0.001 / +0.396.
- Extra rows: hook K = 3 and the dilated hook4.
- The K-step after_change is the collapsed `Oracle.mid_move` at T's 4 mid
  cells, so a K-step is a p*-invariant block move.
- One training run per row.  Expect about +-0.05 on a contrast entry.

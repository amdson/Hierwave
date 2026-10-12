# Marian's 3D WFC modules under infinite annealed Gibbs (2026-10-12)

Code: `castlegen/marian.py` (loader), `castlegen/infinite.py` (`TileModel3D`,
`_anneal3d`, `_block3d`, `generate`/`World` dispatch); experiment
`notes/experiments/marian_infinite.py` (table, renders),
`notes/experiments/marian_probe.py` (composition, wildcard placement);
tests `tests/test_marian.py`.  Data: `cache/marian` (MIT, parsed as text),
plus `Assets/Game.unity` fetched from the repo for the boundary settings.

## Module set
- 454 modules (121 prototypes x distinct rotations), weight =
  `Prototype.Probability` (Slot.CollapseRandom; Empty 10, Solid 1, most
  0.005-2).  `allowed[d][a, b]` from `PossibleNeighbors`: module i is bit
  i % 8 of byte i // 8 of the hex string (long[] of little-endian words).
- Directions (Orientations.cs) -> our (layer, row south, col east):
  0 LEFT -x = col - 1, 1 DOWN = layer - 1, 2 BACK -z = row + 1,
  3 RIGHT +x = col + 1, 4 UP = layer + 1, 5 FORWARD +z = row - 1.
- Symmetry: `allowed[d][a, b] != allowed[opp d][b, a]` for 312 pairs per
  horizontal direction, 374 per vertical one (0.15% of pairs).  Every listed
  pair fits by the face connectors (Module.Fits); the mismatches come from
  SimplifyNeighborData pruning one side.  WFC needs both lists (each side's
  support), so a pair is forbidden if either list omits it.
- Boundary (Game.unity): MapHeight 6, no slots outside; EnforceConnector
  Down 1 on layer 0 (stand on ground) and Up 0 on layer 5 (open sky),
  propagated by arc consistency on a 1 x 6 x 1 self-tiling default column.
  Domains per layer, raw -> propagated: 245 -> 221, 454, 454, 454, 454 -> 425,
  131 -> 127.  Layer 0 never holds air.  The sampler treats a layer's domain
  as hard (never drawn outside it); propagated domains give slightly fewer
  wildcards than raw ones (S = 32: 0.025 against 0.028 per cell).

## Sampler
Grid (L, H, W), chunks stay 2D columns holding all L layers.  Parity
(layer + y + x) & 1; draw key `_u01(seed, y, x, (h << 8) | layer)`.  Flat
and block (B x B columns x all layers, shift, same cone) kernels share
`_site3`: O(6 T) table adds per update (F as uint8 (6, T, T) rows), about
2 us per update at T = 454.  Wildcard fix pass as in 2D (parity 0,
lexicographic); misfits 0 in every run.  Exact across requests, chunk order,
negative coordinates (tests; a pad of 3 instead of 2S + 1 breaks overlap).

## Results (64^2 columns x 6 layers, seeds 0-2; lone 32^2 chunk timed alone)

Default ramps gamma 2 -> 8 (`gamma0` as listed), beta 1 throughout unless
hot (beta0 0.1), quench 4 sweeps at gamma inf; `q0_g4` = no quench, gamma
2 -> 4.  Blocks: shift, one global anneal over Sb K inner sweeps.

| schedule | wildcards / cell | / column | columns with one | lone 32^2 chunk |
|---|---|---|---|---|
| S32 gamma0 2 | 0.0251 | 0.151 | 0.145 | 2.8 s |
| S32 gamma0 0.5 | 0.0397 | 0.238 | 0.231 | 2.9 s |
| S32 quench 16 | 0.0612 | 0.367 | 0.329 | 2.8 s |
| S32 gamma1 32 | 0.0674 | 0.404 | 0.348 | 2.8 s |
| S32 all lexicographic | 0.1973 | 1.184 | 0.750 | 2.9 s |
| S32 q0_g4 | 0.0187 | 0.112 | 0.111 | 2.7 s |
| S64 gamma0 2 | 0.0060 | 0.036 | 0.036 | 16.2 s |
| S64 gamma0 0.5 | 0.0099 | 0.059 | 0.059 | 17.2 s |
| S64 hot | 0.0215 | 0.129 | 0.127 | 19.4 s |
| S64 q0_g4 | 0.0145 | 0.087 | 0.087 | 15.8 s |
| B8 Sb2 K16 | 0.0371 | 0.223 | 0.199 | 1.5 s |
| B8 Sb2 K32 | 0.0124 | 0.074 | 0.072 | 3.0 s |
| B8 Sb3 K32 | 0.0052 | 0.031 | 0.031 | 7.1 s |
| B8 Sb3 K32 gamma0 0.5 | 0.0101 | 0.060 | 0.060 | 7.3 s |
| B8 Sb3 K32 gamma 4 -> 16 | 0.0992 | 0.595 | 0.450 | 7.2 s |
| B8 Sb3 K32 q0_g4 | 0.0177 | 0.106 | 0.106 | 8.0 s |
| B16 Sb2 K32 | 0.0085 | 0.051 | 0.049 | 7.4 s |
| B8 Sb3 K64 | 0.0029 | 0.018 | 0.018 | 13.6 s |
| B8 Sb4 K32 | 0.0029 | 0.018 | 0.018 | 13.0 s |
| B8 Sb3 K32 hot | 0.0203 | 0.122 | 0.118 | 7.7 s |
| S128 gamma0 0.5 | 0.0020 | 0.012 | 0.012 | 113 s |

Misfits after the fix pass: 0 in every run.  Timings single core, partly
with two runs in parallel.

- What helped: more sweeps (S 32 -> 64: 4x fewer wildcards), block Gibbs
  (B8 Sb3 K32 beats flat S64 at 2.3x less time per lone chunk; Sb4 or K64
  halve it again), gamma0 2 over 0.5, propagated layer domains.
- What hurt: hard constraints too early or too long.  Gamma inf for more
  than the last few sweeps, or a gamma ramp to 16-32, freezes defects in
  place (all-lexicographic S32: 0.20 per cell); defects need finite gamma to
  move and annihilate.  A short run prefers no quench (S32 q0_g4 0.019 vs
  0.025), a long one the quench (S64 0.006 vs 0.0145).
- The world is flat.  At T = 1 the product of Marian's weights (Empty 10
  against <= 2 for anything else) makes air dominant: beta0 = 1 runs end
  with layers 2-5 ~100% air, layer 1 62% air, layer 0 floor / solid, i.e.
  one-storey terraces and stairs (58 prototypes used, 77% Empty).  A hot
  start (beta0 0.1) freezes in the opposite phase: layers 0-4 filled
  (Solid 23%, interior corners, tunnels), air only on layer 5, 99
  prototypes, 2% wildcards concentrated on the top layers.  Neither looks
  like Marian's towns: his WFC is sequential (min-entropy collapse with
  propagation), whose output distribution is not the Gibbs measure of the
  weights.  Matching it needs learned energies (e.g. per-layer unaries or
  a coarse height field fitted to WFC output), not schedule tuning.
- Images: images/marian_topdown.png, images/marian_slice.png (B8 Sb3 K32,
  seed 0, 96^2; top-down = colour of the highest non-air module per column
  (prototype hash, shaded by height), wildcards magenta; slice = three rows,
  layer 5 at the top); images/marian_topdown_hot.png,
  images/marian_slice_hot.png (hot start).

## Open
- Speed: ~2 us per update; a C port (bitset rows, compatible-only scan)
  would give the same 5-10x as the city's fast mode.
- Statistics: fit per-layer (or coarse-level) energies so the Gibbs
  measure reproduces WFC's height / prototype marginals.

## 2x2 Metropolis-Hastings groups (2026-10-12, first stats)
`castlegen/plaquette.py`: inside each 3D block solve, a group sweep clears
2x2 cells on one layer and refills them in order NW, NE, SW, SE from
single-site conditionals given the outside and the refilled cells; accepted
with the independence-sampler ratio (exact detailed balance; lexicographic at
gamma = inf).  Exact across requests.  Default AIR_WEIGHTS, B8 Sb3 shift
global, 48^2 columns, seeds 0-2 (`notes/experiments/marian_mh2x2.py`):

| run | wildcards | time / 48^2 | accept (finite / hard) |
|---|---|---|---|
| base K32 | 0.0125 | 10.7 s | |
| MH in the last quarter, K32 | 0.0119 | 10.9 s | 0.74 / 0.75 |
| MH every inner sweep, K32 | 0.0082 | 23.4 s | 0.66 / 0.76 |
| MH every inner sweep, K16 | 0.0164 | 11.8 s | 0.64 / 0.73 |
| base K64 | 0.0053 | 18.7 s | |

- Moves are accepted often (64-76%), but at equal time plain extra sweeps
  win (K64: 0.53% against 0.82% at 23 s).  The 2x2 groups lie in one layer;
  the defects may need vertical or larger joint moves, untested.
- With the rising air profile the wildcard rate is 1.25% at K32 (Marian's
  uniform weights gave 0.52%): taller structure is harder.

## Wildcard experiments, round 2 (2026-10-12)
All at the default AIR_WEIGHTS, B8 shift global, 48^2 columns, seeds 0-2,
numba single core, one batch per table so times compare
(`notes/experiments/marian_sched.py` + `marian_sched_runs.py`; seed spread
about +-0.0007).

Group shapes (`plaquette.py`: 'h' 2x2 in a layer, 'vx' / 'vy' 2 layers x 2
cols / rows, None = skip):

| run | wildcards | time |
|---|---|---|
| base K32 / K48 / K64 / K96 | 0.0125 / 0.0072 / 0.0053 / 0.0032 | 9.2 / 13.6 / 18.0 / 26.7 s |
| MH h every sweep, K32 | 0.0082 | 27 s |
| MH vx every sweep, K32 | 0.0055 | 27 s |
| MH vx every other sweep, K32 | 0.0078 | 17 s |
| MH vx, vy alternate with skips, K32 | 0.0065 | 18 s |
| MH vx, K24 | 0.0083 | 20 s |
| MH vx, vy in the last half, K48 | 0.0053 | 22 s |

Vertical groups beat horizontal ones, but no group variant beats plain
sweeps at equal time.

Exact domino moves at gamma = inf (enumerate pairs within min + 1
violations per cell; exact lexicographic Gibbs on the pair): K32 0.0115, K64
0.0054, ~free.  The wildcard analysis
(`notes/experiments/marian_wildcards.py`) finds wildcards on layers 1-3, 82%
of them fit if one single neighbour were removed, with varied neighbour
signatures (no missing module); dominoes show the blocking neighbour is
itself pinned, so the defects are longer chains.

Schedule shape at Sb3 K64 (equal cost unless noted):

| change | wildcards |
|---|---|
| reference (gamma 2 -> 8, quench 4) | 0.0053 |
| gamma 2 -> 5 | 0.0044 |
| gamma 2 -> 4 | 0.0037 |
| gamma 1.5 -> 5 | 0.0034 |
| gamma 1 -> 8 | 0.0060 |
| gamma 3 -> 8 | 0.0269 |
| gamma 2 -> 12 | 0.0075 |
| quench 1 / 12 | 0.0049 / 0.0058 |
| Sb4 K48 / Sb6 K32 (1.4x / 2.2x time) | 0.0043 / 0.0043 |
| B16 (2.5x time) | 0.0030 |

The hard-weight ramp matters most and is sharp (gamma0 3 is 5x worse than
2): a gentle ramp 1.5 -> 5 halves wildcards at no cost.  With it: K32 0.0062
(C, 4 seeds of 96^2: 0.0064, 0.94 s per lone chunk), K48 0.0040, K64 0.0034,
K96 (2 -> 5) 0.0027.  `export_models.marian_model` now uses 1.5 -> 5 at K32.

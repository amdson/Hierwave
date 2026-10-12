# Infinite worlds by annealed checkerboard Gibbs (2026-10-12)

Code: `castlegen/infinite.py`; experiment `notes/experiments/infinite_gibbs.py`;
tests `tests/test_infinite.py`; images `images/infinite_roads.png`,
`images/infinite_biomes.png`.

## Model and schedule
A pair MRF over T tiles: soft energy `u`, `Eh`, `Ev` (finite) and hard
forbidden pairs `Hh`, `Hv`.  Sweep s draws each cell from
`exp(-(beta[s] E + gamma[s] V))`, V = violated pairs at the cell.
`gamma = inf` is lexicographic: fewest violations first, then soft at
`beta[s]`, so a contradiction keeps the least-bad tile.  Default
`Schedule.anneal`: geometric ramps beta 0.1 -> 1, gamma 0.5 -> 8, then 4
quench sweeps at beta 1, gamma inf (it ends at T = 1, a sample, not an argmin).

## Infinite access
Every draw takes its uniform from `hash(seed, y, x, half-step)`; parity is
global `(y + x) & 1`; the schedule depends on the sweep only.  So a cell is
`f(seed, y, x)`.  Light cone: one cell per half-step, PAD = 2S per side; each
half-step updates only the cells at margin >= h (a truncated pyramid).
`World` stores C x C chunks in a dict and generates the missing chunks of a
request as one batch (bounding rectangle, padded).

## Results (S = 64, single core)
- Chunks requested one by one in random order equal one batch exactly; two
  overlapping requests agree; a larger pad changes nothing.
- The cone is tight: at 95% of the pad 16 (roads) and 582 (biomes) of 96^2
  cells still differ, so 2S is needed, not a fraction of it.
- Cost: requesting single 32^2 chunks costs 1921 updates per stored cell
  (30x the ideal 64); a 384^2 world in two batches 146 (2.3x).  About
  0.35 s per 128^2 at S = 64 for T = 8-12.
- biomes (8 elevation bands, a step of more than one band forbidden, soft
  smoothness): violations/cell 0.031, 0.002, 0 at S = 8, 16, >= 32.
- roads (arm-matching tiles without dead ends, so a lone arm is a
  contradiction): violations/cell 0.0016 at S = 64, 0.0001 at S = 128; they
  stay in the world as broken road ends.

## Next
- Cache the intermediate sweeps of past batches so a neighbouring request
  computes only its new strip (the exact cells at sweep t are the batch
  shrunk by 2t), if single-chunk requests matter.
- Coarse levels: a level at stride s with S_c sweeps adds s * 2 S_c to the
  pad; random access survives.
- Run on the real tile set when it lands (build a `TileModel` from it).

## City tiles (2026-10-12)
`castlegen/citytiles.py` (from branch worktree-city-tiles, cherry-picked):
3907 dual-grid Wang tiles, sockets (corner, corner, feature), weights w.
`SocketModel` runs them with u = -log w and V = mismatched sockets, O(T)
compares per update, no T x T tables (~9 us per update at T = 3907).
`bulk` is a statistics mode (every interior cell updated each half-step,
fixed border; not exact).  Experiment `notes/experiments/infinite_city.py`.

| S | beta0 | violations / cell | corner zones |
|---|---|---|---|
| 32 | 0.1 | 0.137 | P .20 R .30 C .45 D .01 I .04 |
| 32 | 0.3 | 0.066 | P .02 R .84 C .13 I .01 |
| 32 | 1.0 | 0 | R 1.0 |
| 64 | 0.1 | 0.041 | P .10 R .38 C .50 I .01 |
| 64 | 0.3 | 0.0018 | R .97 C .03 |
| 128 | 0.1 | 0.011 | P .06 R .47 C .48 |
| 128 | 0.3 | 0 | R 1.0 |
| 256 | 0.1 | 0 | P .002 R .64 C .36 |

- Exact worlds: 256^2 from 32^2 chunks in two batches equals one batch
  (S = 32, 33 s, 61 updates per stored cell).
- Every clean result is low in variety: the tile weights' equilibrium at
  T = 1 is one or two big districts (R has the top weight and every zone
  boundary costs x0.12).  Parks, industry, water, highways and airports exist
  only as transients of a hot start and die out as the run anneals (they
  survive in the hot S = 32 world at 0.13 violations per cell).  Slower
  annealing does not fix it, since it only gets closer to that equilibrium.
- The missing structure is large scale: the other agent's coarse district
  plan is what brought downtown, rings, industry and a river under WFC.
  Next: a coarse level (stride s, its own World, hashed by level) whose
  zones bias the corners of the fine tiles, per the hierarchical plan.
  Images: images/infinite_city_hot.png (varied, defective),
  images/infinite_city_bulk256.png (clean, two zones),
  images/infinite_city.png (clean, all R).

## Wildcard fix pass and diversity weights (2026-10-12)
- Upweighting rare zones only changes the winner (P 82%).  Cheaper zone
  boundaries keep variety: BOUNDARY_W 0.5 with P, I, D at 1.0 keeps P C I D W
  at S = 128 with ~0.02 misfits per cell (a speckled patchwork, no layout;
  rail and highways never form).
- `SocketModel(wild_u=...)`: one extra half-step on parity 0, lexicographic;
  a cell with no fitting tile takes the wildcard id T (fits any socket).
  Every pair has one parity-0 cell, so the output has no misfit; the pad
  grows by one, chunks stay exact.  Wildcards per cell at S = 32, 64, 128:
  0.10, 0.047, 0.019 (about 20 per 32^2 chunk at S = 128), in small
  diagonal pairs on district fronts.  Image:
  images/infinite_city_fix_bw.5_rare.png (wildcards magenta).

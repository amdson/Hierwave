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

# castlegen

Fixed-budget hierarchical Boltzmann sampling of tile-grid castles in JAX.
Design document: the "MVP framework: 12,000-room castle generation on a 2D grid" doc in Claude.

## What exists

- `castlegen/core.py` — tile encoding (802 tiles: 50 types × 16 door masks, wall, gateway),
  coordinate-hashed noise (murmur3 finaliser; chunk-invariant), checkerboard coordinates.
- `castlegen/base.py` — dense masked base conditional (all 802 tiles scored per active site),
  Gumbel-max site update, Jacobi step of the distance equality, violation count.
- `castlegen/schedule.py` — 20-step base schedule with temperature and pin ramps, d relaxation,
  the two fallback rules. Enough to run the oracle-plan experiment.
- `tests/` — chunk invariance of the noise, d Jacobi fixed point equals exact BFS, a full
  base run ends with zero violations and one gateway.

## Not yet implemented

Coarse levels (Gaussian Gibbs on h, coarse d, count splitting), window resampling,
the offline pipeline (reference sampler, ξ, CCA projection, GMRF fits), the parameter blob
loader (`params/`), and the stress-test harness. Order of work: offline reference sampler
→ oracle-plan experiment → coarse levels.

## Run

    pip install -e ".[dev]"
    pytest

## Conventions fixed for the C++ port

- Noise: `noise(castle, level, step, colour, y, x, slot)` → uniform in (0,1); sampling by Gumbel-max.
- Gateway on the top row with a south-facing door (`GATE_MASK = 4`); perimeter is wall.
- `d` is int16 with 32767 as ∞; walls hold ∞, gateway holds 0.
- Door bits: 0 north, 1 east, 2 south, 3 west; a door exists iff both facing bits are set.

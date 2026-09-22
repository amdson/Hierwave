# castlegen

Fixed-budget hierarchical Boltzmann sampling of tile-grid castles in JAX.
Design document: the "MVP framework: 12,000-room castle generation on a 2D grid" doc in Claude.

## End-to-end test

[Open in Colab](https://colab.research.google.com/github/amdson/Hierwave/blob/main/notebooks/e2e_colab.ipynb)
— `notebooks/e2e_colab.ipynb` clones this repo, runs the unit tests, generates one 192 × 192 castle
with the base level only, asserts the validity invariants against an exact BFS, plots it, and times a
vmapped batch. Or locally: `python -m castlegen.e2e`.

## What exists

- `castlegen/core.py` — tile encoding (802 tiles: 50 types × 16 door masks, wall, gateway),
  coordinate-hashed noise (murmur3 finaliser; chunk-invariant), checkerboard coordinates.
- `castlegen/base.py` — dense masked base conditional (all 802 tiles scored per active site,
  with the reachability term), Gumbel-max site update, door-bit sync (doors are edge variables
  owned by the colour just updated), Jacobi step of the distance equality, violation count.
- `castlegen/e2e.py` — full-size run, invariant checks against exact BFS, batch timing.
- `castlegen/schedule.py` — hashed random init, 20-step base schedule with temperature and pin
  ramps, d relaxation, a fixed number of fallback rounds. Enough to run the oracle-plan experiment.
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
- Door bits: 0 north, 1 east, 2 south, 3 west. Doors are edge variables: the cell being updated
  chooses its bits, and after each checkerboard step the other colour copies its facing bits
  (`sync_bits`), so facing bits agree on every edge. A bit toward a wall is infeasible.
- Base-level behaviour without a coarse plan: the castle grows only within the light cone of the
  d relaxation from the gateway; everything else is walled by the fallback. That is expected, and
  is what the coarse levels exist to fix.

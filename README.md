# castlegen

Fixed-budget hierarchical Boltzmann sampling of tile-grid castles in JAX.
Design document: the "MVP framework: 12,000-room castle generation on a 2D grid" doc in Claude.

## End-to-end test

[Open in Colab](https://colab.research.google.com/github/amdson/Hierwave/blob/main/notebooks/e2e_colab.ipynb)
— `notebooks/e2e_colab.ipynb` clones this repo, runs the unit tests, generates one 192 × 192 castle
with the base level only, asserts the validity invariants against an exact BFS, renders it, times a
vmapped batch, and previews the tile set's local rules. Or locally:
`python -m castlegen.e2e --png castle.png` and `python -m castlegen.preview --png preview.png`.

## Tile sets

A tile set is a JSON file (`castlegen/tilesets/demo.json`; format in the `castlegen/tileset.py`
docstring). It lists sockets and which pairs form a door, kinds with fixed sockets per side, rotations,
variants and a render spec, and energy rules over tags. The sampler never sees individual types: each
(kind, distinct rotation) is one *signature*, all pairwise terms fold into two signature tables
`Eh[left, right]` and `Ev[top, bottom]`, and the concrete type (rotation, variant values) is drawn per
cell at the end from hashed noise. Variants multiply the type count without touching the energy, so a
spec with 10⁶ types still samples over a handful of signatures (see `tests/test_tileset.py`).

Rendering (`castlegen/render.py`): ASCII glyphs, or a px × px sprite per cell whose colour comes from a
variant's `render`, else the kind's (`{"color": "#rrggbb"}` or `{"image": "file.png"}`), else a random
colour per (kind, variant). Room sprites get borders from the sockets: a line for wall, a gap for door.

## What exists

- `castlegen/core.py` — coordinate-hashed noise (murmur3 finaliser; chunk-invariant), checkerboard coordinates.
- `castlegen/tileset.py` — tile-set loader and compiler (signatures, pair tables, door tables, type draw).
- `castlegen/base.py` — dense base conditional (all signatures scored per active site, with the
  reachability term), Gumbel-max site update, Jacobi step of the distance equality, violation count.
- `castlegen/render.py`, `castlegen/preview.py` — ASCII / image rendering; local-rules preview.
- `castlegen/e2e.py` — full-size run, invariant checks against exact BFS, batch timing.
- `castlegen/schedule.py` — hashed random init, 100-sweep annealed base schedule (T 4 → 1) with pin
  ramps, d relaxation. No repair step: the output can hold rooms with violated hard terms.
- `tests/` — chunk invariance of the noise, d Jacobi fixed point equals exact BFS, a batch of
  base runs ends with zero violations and one gateway, tile-set compilation and pair energies,
  a 10⁶-type spec, rendering.

## Not yet implemented

Coarse levels (Gaussian Gibbs on h, coarse d, count splitting), window resampling,
the offline pipeline (reference sampler, ξ, CCA projection, GMRF fits), the parameter blob
loader (`params/`), and the stress-test harness. Order of work: offline reference sampler
→ oracle-plan experiment → coarse levels.

## Run

    pip install -e ".[dev]"      # dev includes Pillow for image tiles
    pytest

## Conventions fixed for the C++ port

- Noise: `noise(castle, level, step, colour, y, x, slot)` → uniform in (0,1); sampling by Gumbel-max.
- Grid state is signature ids (int32); the last two signatures are wall and gateway.
- Gateway on the top row with a south door; the perimeter holds no rooms.
- `d` is int16 with 32767 as ∞; non-rooms (wall, solid kinds) hold ∞, gateway holds 0.
- Sockets are fixed per kind and rotate with it (sides N, E, S, W; rotation clockwise). A door exists
  on an edge iff the two facing sockets connect; it is a function of the two signatures, so nothing is
  modelled on edges.
- Temperature divides the energy only; a kind's `unary` sets its base mass exp(−unary), shared by its
  rotations and variants, and is not tempered.
- Type ids: kind-major, then rotation, then variant values (first axis most significant).
- Base-level behaviour without a coarse plan: the castle grows only within the light cone of the
  d relaxation from the gateway, and with the demo rules it usually stops at the first dead-end
  chamber. That is expected, and is what the coarse
  levels exist to fix. `castlegen.preview` shows the local rules without that limitation.

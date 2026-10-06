# Channels demo: ground + roots on one tile kernel

Plan for the first test of the interface in `notes/channels.tex` (sections
7, 10, 16): two channel sets written separately against a generic layer, then
combined by taking the union of their factors, with no edit to the kernel.
Numba kernel, not the bit sampler.

## Demo

Side view, 2D grid of tiles.  Four channels:

| channel  | level | domain                              | kernel                       |
|----------|-------|-------------------------------------|------------------------------|
| `surf`   | 8     | rows of the chunk under the surface, 0..8 | designed: value-noise heightmap, sampled once |
| `tile`   | 1     | signatures of the tile set          | generic Gibbs (numba)        |
| `cert`   | 1     | d in {0..D} + INF                   | joint with `tile` at each site (certificate predicate) |
| (root exemplar) | 1 | enters as counted tables, not as a variable | — |

Ground set (`castlegen/channels/ground.py`):

- `surf` channel: 1D value noise over chunk columns gives the surface height;
  the chunk value is how many of its 8 rows lie under the surface.  Honourable
  under support by construction (a chunk above a non-full chunk is empty).
- hard seam factor on the tile view `solid`, vertical: solid needs solid below
  (the support rule of `generic.support_rule`); below the map counts as solid.
- soft honour factor `count`: the number of solid tiles in a chunk against
  8 * surf, a quadratic table.
- soft pair tables over the view `ground` (sky / soil / stone; root kinds read
  as soil), from the tile set's rules.

Root set (`castlegen/channels/roots.py`):

- view `root` (none / earth / trunk / r1 / r2 / r3), view `mass`
  (0 / 0 / 3 / 1 / 2 / 3), view `trunk`.
- soft pair tables over `root` counted from a hand-painted ASCII exemplar
  (negative PMI, smoothed) plus a unary from its marginal.
- hard contact factor: a root or trunk tile never touches sky, except the
  trunk's north side, which must be sky (the trunk stands at the surface).
- certificate `cert`: d per cell.  Valid at p iff mass_p = 0, or p is a trunk
  with d_p = 0, or some 4-neighbour q has mass_q >= mass_p > 0 and d_q < d_p
  < INF.  The tree rule of channels.tex section 14 with mass in place of the
  join bits.  Drawn jointly with the tile at each site by enumerating (t, d),
  the neighbours' validity included (dependants), as blockfield does with its
  (g, d) field.  A small cost DELTA * d keeps chains short.

Combination (`notes/experiments/chan_combined.py`): the union of the two
factor lists on one tile channel whose views are the union of the views.
Expected to need no new code beyond the script; knob changes only if the two
sets fight (e.g. the root tables pulling density away from `surf`).

## Generic layer (`castlegen/channels/core.py`, `kernel.py`)

- `Channel(name, h, D, views)`: grid `(H / h, W / h)` int32; `views[name]`
  is a `(D,)` int array (a view of the domain).
- `Factor`: a table over views.  Kinds:
  - `pair`: `(a: chan, view) at p` with `(b: chan, view) at q`, `q = (p * h_a)
    // h_b + off` in b's cells.  Same-level neighbours (`off = (0, 1)`) and
    parent reads (`b` coarser) alike.  `pad`: b's view value off the grid,
    or -1 to skip the term there.
  - `count`: `(a: chan, view) at p` with the SUM of view `b` over the finer
    channel's cells inside the block of p.  Homed on the fine channel.
  - `unary`: a table over one view.
  - hard = entries of `inf`.  No separate type.
- `Certificate(tile_chan, mass_view, root_view, cert_chan, D, delta)`: the one
  computed factor; the kernel has a branch for it.  (channels.tex section 16:
  a computed factor produces its shape directly.)
- `Model(channels, factors, certs)`: `compile()` packs every channel grid into
  one buffer, every view into one buffer, every table into one buffer, and
  each factor into a descriptor row.  For a channel being updated it lists
  the factors homed on it plus the reflections of same-level pair factors
  (transposed table, negated offset).  Factors homed below a channel are
  ignored when that channel is updated: top-down only (section 12).
  Colouring: radius = max offset (+1 with a certificate, dependant reads);
  2 colours at radius 1, 5 at radius 2 as `(x + 2 y) mod 5`.
- `kernel.sweep`: one sweep of a level-1 channel, colour by colour, each site
  drawing its value from the exact conditional over all D candidates (Gumbel
  over finite energies; a site with no finite candidate keeps its value and
  is counted as a violation).  With a certificate, candidates are (t, d).
- Designed channels (`surf`) write their grid themselves and have no kernel.
- Schedule: hand-declared in the scripts (levels top-down, N sweeps at level
  1).  No ramps, T = 1, fixed sweep counts.

## Order of work

1. core + kernel, with a test: a pair MRF on a small torus reproduces
   `bitgibbs.exact`-style conditionals (or simply: support never violated,
   count honoured).
2. ground alone: `notes/experiments/chan_ground.py` -> `images/chan_ground.png`.
3. roots alone on a flat clamped ground: `chan_roots.py` -> `images/chan_roots.png`.
4. combined: `chan_combined.py` -> `images/chan_combined.png`; report which
   knobs, if any, had to move.

## What this does not test

Promotion of a coordinate view, exemplar coordinates (the roots use counted
pair tables, not `generic.py`'s coordinates), the bit sampler, the
honourability checker, calibration.  Those are later.

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

## Results (2026-10-06)

Code: `castlegen/channels/{core,kernel,ground,roots}.py`, tests in
`tests/test_channels.py`, scripts `notes/experiments/chan_{ground,roots,combined}.py`,
images `images/chan_{ground,roots,combined}.png`.

Ground alone (128 x 256, 60 sweeps, 7 ms/sweep): 0 unsupported cells, chunk
solid count within 0.9 of 8 * surf on average, 92% of chunks exact.

Roots alone (64 x 192, flat ground, 1000 sweeps, 8 ms/sweep, 4 trunks): 0
rule violations, 0 dangling ports, 0 root-sky contacts, 120 root cells,
mass 65 / 34 / 21 thin to thick, degree 6 / 100 / 14 tips / chains / forks.
Two design changes were forced on the way and are in the module docstrings:
  - roots as plain mass cells with pairwise attraction fill blobs (a blob
    has more good pairs than a chain), so roots became port tiles with join
    bits in the certificate, as channels.tex section 14 says;
  - single-site Gibbs cannot nucleate a port pair, so the port seam is soft
    (`dangle`), and the exemplar's mass / degree statistics are counted as
    marginals, since its tips are all mass 1 and the joint table priced
    the thick tip every growing root passes through at ~8.
  The certificate d uses Dmax 4096 with delta 0.05: with a geometric draw
  the d-sum over a chain of n cells is prod 1 / (exp(delta i) - 1), neutral
  while delta * depth < 1, and the range must hold depth / delta.

Combined (96 x 256, 1500 sweeps, 25 ms/sweep): the union of the factor
lists on one tile channel, no kernel edit, no new code beyond the script.
0 unsupported, 0 certificate violations, 4 of 4 trunks at the surface, 75
root cells.  One knob moved: `grow` 1.0 -> 1.5, because the ground's
texture term sees root cells as earth and in fine-grained stone they lose
the rock cohesion.  Gating the texture off root kinds entirely (a zero row
in the ground view) was tried and is wrong: the trunk then loses the solid
cohesion every surface cell has and is never placed.  The right gate is
narrower than "transparent": roots are earth-like solids to the ground.

Trenches: with the tile level started blank, the ground rose for ~50
sweeps and a trunk was placed as soon as the rising ground entered its
marked chunk, at that chunk's bottom row; the hard sky-above-trunk rule and
the support rule then locked a one-wide shaft above it while the ground
rose up to 7 more rows.  Fixed by starting level 1 as a consistent
refinement of surf (`ground.init_tiles`, as promise.py's child
initialisation): trunk depth below the neighbouring surface went from 8
to 0..2 cells, the residual being the surface's own fluctuation.

Not yet good: the soil / stone texture is salt-and-pepper at T = 1 with
these pair energies, and root systems are modest (about 20 cells per
trunk in the combined run) and grow slowly, one tip cell per sweep at
best, since every extension passes a dangling port.

## Results, second pass: masked exemplar coordinates (2026-10-06)

The thin wiry roots of the first pass were the pair statistics' fault:
pair terms carry no shape, and the exemplar was itself a one-cell-wide
skeleton.  `castlegen/channels/coord.py` adds the coordinate channel of
channels.tex sections 5 and 9: each cell holds an index into a thick,
masked ASCII exemplar; the view `alpha` is FREE (masked out: the ground's
cell, no root allowed), EARTH (the ring below the surface around the
roots: any earth tile) or a specific root / trunk tile; one pair factor at
offset (0, 0) couples tile to alpha(u) at cost nu per mismatch.  The
certificate, port seam and contact rules are unchanged and keep acting on
the tiles; the counted tables are switched off (`roots.factors(counted=False)`).

Kernel: (u, t, d) are drawn jointly per site (`coord.joint_sweep`), using
the tile kernel's factor energies and certificate weights (factored out
as `kernel.site_weights`).  Separate u and t sweeps could not nucleate:
the coordinate and the tile must change together, and with 45 root kinds
against one soil, label entropy beat any mismatch cost small enough to
let them change in turn.  Coordinate energy: lam per incoherent
neighbour, except that two FREE cells owe each other nothing, so a
verbatim copy of the exemplar's masked region surrounded by FREE
continuations is the energy minimum.  Candidates per site: the current
value, the four coherent continuations, KR random coordinates, KT
coordinates whose alpha equals the current tile (how a placed trunk
finds its coordinate).  KR = 8 gave blobs: inside a thick band every
interior coordinate matches, so jumps turn the copy into a patchwork.
KR = 0 copies verbatim; KR = 1 keeps the shape with variation (a doubled
taproot, a jagged crown).  Knobs: nu 6, lam 1, dangle 1 (ports now only
mark an unfinished copy), KR 1, KT 1.

Roots alone (64 x 192, 300 joint sweeps, 15 ms/sweep, 4 trunks): 0 rule
violations, 414 root cells, mass 28 / 102 / 284 thin to thick, 683
coordinate cells in the mask, 5 mismatches.  images/chan_roots2.png.

Combined with the ground (96 x 256, 400 joint sweeps, 39 ms/sweep): no
knob moved from the roots-only run.  0 unsupported, 0 certificate
violations, 3 of 3 trunks, 227 root cells, 7 mismatches.  One exemplar
fix on the way: the earth ring must start below the trunk's row, or the
exemplar asks for earth beside the trunk where the real surface has sky,
and the trunk flickers.  images/chan_combined2.png.

## What this does not test

Promotion of a coordinate view, exemplar coordinates (the roots use counted
pair tables, not `generic.py`'s coordinates), the bit sampler, the
honourability checker, calibration.  Those are later.

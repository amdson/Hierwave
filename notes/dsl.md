# Structure DSL and a unified HMRF: discussion notes (2026-10-02)

A record of ideas from a design discussion.  Nothing here is implemented, and
nothing here is a decision.  Starting points in the code: the rigid object
channel (castlegen/objects.py, notes/experiments/macro.py), the promise levels
(castlegen/generic.py JointLevel), the tile stage (blockconn, blockfield) and
the bit-sliced sampler (castlegen/bitgibbs.py).

## 1. Current model

A hierarchical MRF with three kinds of variables:
- coordinates u_h (exemplar windows), mostly with soft interactions (E_c, E_par);
- promises v_h (support class; connectivity ports and keys), defined from tile data;
- tiles at h = 1, with pair energies and hard rules; objects add a code channel o.
Sampled top-down, tiles last; every higher level constrains the statistics of
the levels below it, promises most strongly.  So far few hard constraints sit on
high-level variables, and there is little aggressive search at high levels to
satisfy constraints.

## 2. Objects as macro-tiles

- The object code o = base[k] + dy w + dx is a hidden channel.  The pair check
  nbr[o_p, d] >= 0 => o_q = nbr[o_p, d], applied from both sides, makes objects
  whole or absent.
- A coordinate at level h ranges over every exemplar position at tile
  resolution, so an object can sit at any tile offset, and it is present from
  the coarsest level whose windows contain it.  A coordinate change at level h
  moves a whole window at once; single-site tile updates cannot move a rigid
  object.
- nbr generalised from one successor to a set of successors gives an adjacency
  grammar over macro-tiles: composites ("A's east edge meets B's west edge"),
  stretchable bands (an edge code whose successor is itself), variants (several
  templates sharing border codes), wildcard cells (a code with no forced tile).
  With sets, rigidity is no longer transitive and sampling becomes constraint
  satisfaction (WFC-style contradictions), unless choices are confined to regions
  whose extent a parent has already committed.

Theory: the recognizable picture languages (REC) are exactly the projections of
local picture languages (Giammarresi–Restivo), and pair (domino) checks suffice.
"Visible tiles + hidden code channel + pair checks" therefore expresses anything
a 2D finite-state description can.  Finding a valid tiling is NP-hard on finite
regions and undecidable on the infinite plane.

## 3. A DSL for structures

Formalism discussed: split grammars (as in CGA shape grammars, Müller et al.).
A rule splits a rectangle into named sub-rectangles: fixed, repeat (stretch) or
alternative (variant).  Leaves are tiles, free regions (texture) or other rules.
The derivation tree lines up with the sampling hierarchy: a coarse cell commits
to the rule covering it and where its splits fall (a promise), and finer levels
resolve inside.  The framing relates to L-systems (top-down rewriting) and WFC
(local adjacency constraints).

Layers for users:
1. tiles: the existing tile set JSON with sockets;
2. parts: ASCII grids (the existing "grid" placements) with annotations;
3. rules: composition on top, e.g. `castle := ring(wall, 3) around courtyard`, and
   named global properties (`connected`, `supported`).

Example part:
```
part temple
  rows:  "###########"
         "#....A....#"
         "#.........#"
         "#####.#####"
  stretch x: cols 2..8
  stretch y: rows 1..2
  free: '.'
  port S: row 3, col 5
```

### Painted masks

Users paint masks over an exemplar and annotate them:
- region: rigid copy (the current object channel);
- band inside a region: stretch x / stretch y;
- hole inside a region: free (texture fills it, the frame stays exact);
- port: a cell or edge strip where the structure connects;
- one label on several regions: the same part; differing instances are variants.
Mask nesting gives the grammar tree.  A compiler could propose relations seen in
the exemplar ("every temple touches a path on its S side; make it hard?") for the
user to confirm.  Masks mark the line between authored (hard) and learned (soft)
content.  From one painted instance it is ambiguous what is meant to vary;
annotations or several instances resolve that.

Compiler outputs from one spec:
1. an exemplar generator (parts re-placed at sampled sizes and variants on a
   synthesised background), so the exemplar stays regenerable;
2. code tables (tmpl, successor sets) for the tile-level hard check;
3. port lists for the connectivity promise.
Compiler checks mentioned: masks nest; bands span their region; a stretch band
connects across its own repeat seam (so stretching keeps internal connectivity);
every variant reaches its ports; inconsistent paint is rejected with the
offending pixels.  Storage: painted masks as extra layers next to the tile layer,
as cache/castle_macro_obj.npy already is for codes.

### Ports

- Local half: the port cell opens on side d and the outside cell joins it.  A pair
  check whose successor set covers one cell and does not propagate.
- Global half: reachability, not locally checkable.  Carried by the connectivity
  promise: the object is a super-node with a precomputed port pattern (a block
  port pattern of irregular size), and the coarse cell holding a port commits an
  open side toward it.  This relates to the macro.py run in which a temple was
  unreachable and the connectivity term outweighed LO, breaking the object.
- Split: codes for what is local and exact; promises for what needs information
  to travel (reachability, support).

## 4. A unified model and sampler

Gibbs over cells, each update conditioned on (a) same-level neighbours within a
radius r, (b) the cell's other channels, (c) the parent cell.

| factor | soft | hard |
|---|---|---|
| same-level neighbours | E_c, pair energies | seam compat, object checks, socket joins |
| within one cell | cost(u, v), -logz | (t, o) consistency, (g, d) validity |
| parent -> child | E_par, parent cost | promise refinement |

Two update kinds: small domains (tiles, codes, promises, g) enumerated exactly,
bit-sliced (each factor a table row indexed by a neighbour's value); large
domains (coordinates, m^2 values) by MH with candidate proposals.

Rules discussed:
- Certificates: a nonlocal property is not a factor; it comes with a witness
  channel whose local checks imply it (connected: distance field or promise keys;
  supported: column classes; port reachable: an exit in the promise graph).
- Honourability: with top-down sampling and no revisiting, every hard parent ->
  child relation must leave a nonempty set of child states for every parent value
  in every context its constraints allow (constitution 9).  Checkable on small
  cases.  Joint promises (support + connectivity) are a case where it may fail.
- Coarse levels have few cells, so heavy search there (annealing hard terms in
  from soft, many sweeps, block updates) is cheap.  A structure is easiest to
  move at the level where it is one cell.
- Not covered by this scheme: bottom-up passes (revisiting parents given
  children, multigrid style), and factors beyond radius r.

## 5. Channels as abstraction maps

Building the HMRF = defining channels at levels and the interaction terms
between them.  A channel at level h as alpha_h(window), a deterministic summary of
the tiles under it:
- support class: which columns hold solids;
- ports: which sides open and which exits connect;
- object code: which template cell;
- coordinate: the window itself, as found in the exemplar, compared softly.
Properties: tables come from counting alpha over exemplar windows; if alpha_h is
compositional (computable from the children's alpha_{h/2} plus seams), the hard
parent -> child relation follows and the honourability check is mechanical; the
sampler does not have to discover a channel's meaning.

Sources of terms: authored (masks, annotations, sockets: hard rules), derived
(compiled certificates and level relations), learned (soft tables from counts).
Channel properties discussed: compositional, small (bit-sliceable), and tied to a
hard rule the coordinates' soft statistics do not reliably deliver.  Support,
connectivity and objects each arose that way.

## 6. Coordinates as weakened statements

A coordinate is a statement about what the tiles under its cell look like, after
masking and blurring.
- Blurring: a level-h coordinate states only scale-h detail; the levels form a
  scale space.  E_c (PCA'd patch features) and E_par are already blurs; an
  explicit h-dependent blur relates to how lambda_c and sigma are set per level.
- Masking (the user's framing): a mask weakens a coordinate's request, a statement
  that pixels in a particular area do not matter.  Not yet worked out.
- Variants raised in discussion: fixed masks (painted object or free regions, or
  features owned by a promise, so the same evidence is not counted twice); a
  sampled mask channel with a cost per masked area, e.g. a likelihood ratio
  against a fallback model, where the parent abstains on masked children (an
  outlier or line process, as in image MRFs).  Uses mentioned: hard constraints
  taking cells from a coordinate, two coordinates covering parts of one region,
  stepping stones between coordinates.  Quantity mentioned: masked fraction per
  level.
- Spectrum: rigid object (no blur, no mask, hard) -> promise (exact summary, hard)
  -> coordinate (blurred and masked, soft).  Each channel is (alpha, region / mask,
  scale, comparison).

## 7. Multiple interacting exemplars

- Several exemplars as several coordinate channels, each with its own alpha_k
  (different features, regions or scales).  Example: mountains made of castle
  tiles, with a terrain channel (solid bit, support) and a settlement channel
  (rooms, connectivity), each with its own promises.
- Ground truth for the coupling: p(upper) ∝ Z(upper), the summed weight of the
  lower-level resolutions consistent with all upper variables (F = -log Z).
- Approximations discussed:
  - factorise: F ≈ sum_k F_k + sum over declared interacting pairs F_kl;
  - couple through alpha rather than u: (castle promise x support class) is
    16 x 16, (castle coordinate x mountain coordinate) is 16k x 16k;
  - self-play: run the fine sampler under many upper configurations, record the
    energy reached or failure, fit F_kl (honourability is the F = inf case);
    relates to the constitution's "cross-scale consistency is a training goal";
  - no learned coupling: channels interact only through tiles and hard rules.
- Many exemplars in a full game: roles (terrain, settlement, dungeon, decoration)
  as channels that fix alpha and interaction terms; exemplars fill roles and share
  the role's alpha features; a role's coordinate ranges over the concatenated
  library, so it also selects the exemplar; spatial coherence from E_c / E_par,
  optionally a coarse style channel gating the library.  Couplings are role x
  role over shared features; adding an exemplar to a role adds no coupling
  tables.
- Alternative: a single coordinate channel over one concatenated library, with
  roles as different alpha features of that one coordinate.  Interactions
  between roles then come from exemplars containing them together (generated by a
  layout program, since the exemplar is regenerable), and hard interactions from
  joint promises.  Role combinations that never appear together in an exemplar
  are not produced.

## 8. Fast exemplar matching

Concern: with many exemplars, nearest-neighbour search and its memory become
expensive even with good asymptotics.  Idea: a restriction of Gibbs where each
latent has one or more predefined neighbourhoods of candidate values, sampled
from very quickly, with neighbourhoods designed for mixing.

Precedent (cited from memory, titles and venues unverified):
- Ashikhmin, "Synthesizing natural textures", I3D 2001: coherent candidates.
- Tong et al., "Synthesis of bidirectional texture functions on arbitrary
  surfaces", SIGGRAPH 2002: k-coherence, precomputed k similar coordinates per
  exemplar position.
- Lefebvre & Hoppe, "Parallel controllable texture synthesis", SIGGRAPH 2005;
  "Appearance-space texture synthesis", 2006: k-coherence on the GPU, k ~ 2.
- Barnes et al., "PatchMatch", SIGGRAPH 2009; "Generalized PatchMatch", ECCV 2010:
  propagation plus random search, no index.
- Cohen et al., "Wang tiles for image and texture generation", SIGGRAPH 2003:
  precomputed tile set, generation by lookup.
- Merrell, "Example-based model synthesis", I3D 2007; Gumin, WFC (2016):
  precomputed adjacency lists.
- Zanella, "Informed proposals for local MCMC in discrete spaces", JASA 2020:
  fixed neighbourhoods N(x), sampling within them by a balancing function of pi.
- Grathwohl et al., "Oops I took a gradient", ICML 2021: discrete sampling with
  gradient-informed neighbourhoods.
- Peskun ordering (1973); Liu, Liang & Wong, "Multiple-try Metropolis", JASA 2000.
- Malkov & Yashunin, HNSW, TPAMI 2020: navigable small-world graphs, memory
  O(N M).
- Jégou, Douze & Schmid, "Product quantization for nearest neighbor search",
  TPAMI 2011.

Correctness: a candidate set that depends only on the neighbours (not the site's
own value), sampled exactly, plus an epsilon uniform fallback, is a valid MH move
(generic.py relies on this).  Candidates that depend on the current value need a
symmetric graph or the MH correction.  Mixing design = a sparse graph on the
state space whose pi-weighted walk has a large spectral gap.

Design sketched: per coordinate a fixed candidate set of
- the coherent shifts from its 8 neighbours,
- the parent's prolongation,
- k ~ 4-8 links in a symmetric kNN / small-world graph over all exemplars'
  coordinates (a few long links crossing exemplars),
- epsilon uniform;
sampled exactly within the set (MH-corrected for value-dependent parts).
Measures mentioned: autocorrelation of the coordinate field per level, seed
diversity, cross-exemplar switching rate, compared with the current kNN-MH at
matched cost.  Cross-exemplar mixing depends on long links or a coarse style
channel.

Memory: candidate tables are small (16k coordinates x k = 8 x 4 B ~ 0.5 MB per
exemplar per level).  The E_c feature tables dominate (25 slots x m^2 x D floats
per level).  Options: int8 or product-quantized features; features computed on
the fly from the 1-byte exemplar tiles and a small per-tile table; coarse levels
ranging over a stride-h/4 lattice of coordinates.

## 9. Workload per step

Coordinate update:
- candidate set of fixed size (~18 with k = 8), no deduplication (duplicates
  counted consistently in the proposal probability), so branch-free;
- per candidate: Delta E_c over 25 slots x D (~300 multiply-adds at D ~ 12),
  fixed-size parent and cost reads; hard promise checks mask candidates rather
  than change their number;
- MH reverse candidate set for value-dependent parts roughly doubles the graph
  part; acceptance changes outcomes, not work;
- estimate ~6-7k multiply-adds, ~1 us per site on one core; a 128^2 map at
  h = 16 is 64 cells.
- Variable part: memory latency.  Graph-link candidates are random reads into
  large tables; with each coordinate's 25 slot projections contiguous (~300 B in
  int8, ~5 cache lines) the misses per step are bounded by ~k x 5 lines.  Coherent
  candidates have good locality.
- The number of steps needed (mixing) is not fixed by this.

Channel interaction terms:
- constant cost, independent of exemplar count: couplings F_kl over alpha values
  (e.g. 16 x 16), hard parent -> child masks, object checks (<= 5 codes, 4 pair
  checks), certificate loops (10 values of g, RK keys);
- per-coordinate cost(u, v) rows grow with total coordinates x sum of V; as
  4-bit bit-plane stacks a 16-value channel is ~10 bytes per coordinate (~10 MB
  per channel per level at a million coordinates), one fixed read per candidate,
  directly usable by the bit-sliced adder; plus alpha(u), ~1 byte;
- joint updates: the joint domain is the product of the channels updated
  together (support + connectivity: 16 x 16 x RK ~ 1-4k); options: block updates
  over channels linked by hard terms with Gibbs across the rest, and a bound on
  joint block size; blocking fixes the work and slows mixing for hard-coupled
  channels;
- per-step work as a formula: candidates x (features + per-channel cost reads +
  declared couplings) + sum over joint blocks of their domain products; it can be
  reported at compile time.

## 10. Complexity

The combination of these pieces is close to the edge of manageable complexity.
A smaller core discussed: one channel interface (domain, alpha, table factors of
the three kinds), one update (fixed-size candidate set, table-driven exact draw),
and a compile-time cost report; the existing coordinates, promises and object
codes fit it.  Pieces outside that core: multiple coordinate channels and learned
couplings, sampled masks, successor sets beyond stretch bands, the rule layer,
joint updates over more than two channels.

Possible experiments mentioned:
1. refactor existing channels onto one interface; existing experiments reproduce;
2. fixed-size candidate sets over a concatenated library of 2-3 exemplars:
   per-step cost as the library grows, exemplar switching at the top level;
3. mountain-castle with one coordinate channel and joint support + connectivity
   promises, from a generated combined exemplar;
4. stretch and port annotations;
5. ports as super-nodes in the connectivity promise;
6. an honourability checker for hard parent -> child relations;
7. two coordinate channels coupled only via tiles and hard rules, then with
   self-play-fitted F_kl.

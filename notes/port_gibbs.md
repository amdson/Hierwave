# Port-factorized bit-sliced Gibbs: proposed algorithm (2026-10-04)

A design written up from a discussion.  Nothing here is implemented.  It
targets the lowest level (tiles) only, and a GPU (Apple M2 via Metal) as the
eventual backend.  Starting points in the code: castlegen/csrc/bitgibbs.c
(bit-sliced levels, `acc_*`, `bg_sweep_pairs`, `bg_sweep_field`),
castlegen/bitgibbs.py, castlegen/blockfield.py, castlegen/ports.py.

The idea in one paragraph: tiles interact only through small per-side port
codes.  A site's conditional is computed for all tiles at once as bit masks
(bit t = tile t), where each mask is a tiny relation table evaluated as a
truth table over the transposed port bits.  Energies are a handful of integer
levels summed by a bit-sliced adder; a tile is drawn exactly with integer
arithmetic (popcount, one multiply-high, a fixed-step select).  More than 64
tiles are handled by Gibbs within one block of a rotating partition of the
tiles.  Auxiliary certificate fields (g, d) combine per-side masks with OR/AND
chosen by scalar predicates on the neighbours' field values.

## 1. Terms

| Term | Definition |
|---|---|
| tile, S | The site variable t ∈ {0 .. S−1}. |
| site p, neighbour n_d | A grid cell; n_d is its neighbour on side d ∈ {N, E, S, W} = {0, 1, 2, 3}; opp(d) = (d + 2) mod 4. |
| port byte port[t][d] | 8 bits attached to side d of tile t.  A tile's ports are one u32. |
| group g, val_g(q) | A fixed set of (usually 3) bit positions of a port byte; val_g(q) is the value of those bits in byte q (0 .. 7).  Default split 3 + 3 + 2. |
| relation R_g^d | A bit matrix over (own value a, facing value b) for group g on side d: bit (b << 3) \| a of one u64.  Hard: 1 = allowed. |
| row | row_g^d(b) = (R_g^d >> 8b) & 0xFF, the 8-entry truth table over own values a given the facing value b. |
| level relation | E_g^d(a, b) ∈ {0 .. 2^FB − 1} ∪ {∞}, stored as FB level-bit relations Rb_g^d[i] (bit i of the level) and one forbid relation Rf_g^d. |
| word, NW | A u64 holding the bits of 64 tiles; NW = ⌈S / 64⌉ words per mask (NW = 1 with tile subsets, §6). |
| mask | NW words, bit t for tile t. |
| plane P[d][i] | The mask of tiles whose port bit i on side d is set: the port table transposed. 32 planes per word. |
| eval3(row, x0, x1, x2) | The mask of tiles whose group value a (bits from planes x0..x2) has row bit a set: a 3-level mux tree with planes as selectors and row bits (as all-zero / all-one words) as leaves.  ~21 bit ops per word. |
| factor, factor stack | A term the site sees, as FB level planes plus one forbid plane (masks).  The format `acc_add` consumes. |
| accumulator A | AB bit planes (AB = 5) holding each tile's summed level, plus a bad plane; a carry past bit AB − 1 sets bad. |
| level l, NL | Summed level 0 .. NL − 1, NL = 2^AB = 32.  Energy ≈ unit · l. |
| cw[l] | Integer weight of level l: 2^(31 − l) (unit = ln 2, exact), or round(2^40 e^(−unit·l)). |
| level mask M_l, count n_l | Tiles whose accumulated level is l, legal and not bad; n_l = popcount(M_l). |
| Z | Σ_l n_l cw[l], the (integer) normaliser of the site conditional. |
| select(w, k) | Position of the k-th set bit of word w (k < popcount(w)); fixed 6-step binary search. |
| legal_p | Per-site mask of allowed tiles (from a parent level or a painted mask). |
| partition Π, block B | Π splits the tiles into blocks of ≤ 64; block_of_Π(t) is the block containing t (§6). |
| aux fields g_p, d_p | Per-site certificate values (§7): g ∈ {0 .. G, INF = G + 1}, d ∈ {0 .. D}. |
| colouring | A map from sites to colour classes updated in parallel; 2-colour checkerboard (x + y) mod 2, or 5-colour (x + 2y) mod 5. |

## 2. Model

Target distribution over a tiling T (restricted to legal tiles):

    π(T) ∝ Π_p cw-weight( u(t_p) + Σ_{edges (p, q), side d of p} Σ_g E_g^d(val_g(port[t_p][d]), val_g(port[t_q][opp d])) )

i.e. the energy of an edge is a sum over port groups of small level tables,
plus a per-tile (or per-site) unary u.  A hard model is the special case
E ∈ {0, ∞}.  Any total level ≥ NL is treated as forbidden (truncation of the
target; budget levels so that typical sums stay well below 32, or use AB = 6).

Symmetry (required): an edge's energy must be the same whichever endpoint is
updated, so

    E_g^d(a, b) = E_g^{opp d}(b, a)        for all g, d, a, b.

With one relation per group shared by all sides this means each R is a
symmetric 8 × 8 bit matrix (R == transpose(R)).  Separate horizontal and
vertical relations cost 2 u64 per relation instead of 1.

Expressivity: a constraint between two tiles must factor as an AND (hard) or a
sum (soft) over groups.  Any relation *within* a group is free (e.g. "at least
one of these 3 channels joins").  Equality of a code of any width factors bit
by bit (object glue codes: 0 = no object, else the expected partner part).
Not expressible: disjunctions or interactions across groups, arbitrary learned
S × S tables (approximate them and correct by MH, §5.3), non-local rules (need
certificates, §7).

## 3. Bit-sliced representation

Precomputed once per tile set:
- planes P[d][i] for d < 4, i < 8 (32 masks);
- relations: per group, 1 u64 (hard) or FB + 1 u64 (levelled), per
  orientation if not shared;
- unary stack(s) for u;
- the weights cw.

Per site, the neighbour on side d contributes its facing byte
q_d = port[t_{n_d}][opp d] (a site never needs its neighbours' tile ids,
only their port bytes).  For group g, b = val_g(q_d), and the factor mask is

    eval3(row_g^d(b), P[d][g0], P[d][g1], P[d][g2])

where g0..g2 are the group's bit positions.  A 2-bit group uses a 2-level mux
(or pads with an all-zero plane).

## 4. Site update, hard model

    cand = legal_p
    for d in 0..3:
        q = facing byte of n_d
        for g: cand &= eval3(row_g^d(val_g(q)), P[d][g0], P[d][g1], P[d][g2])
    n = Σ_w popcount(cand[w])
    k = mulhi32(rand32, n)                    # uniform in [0, n)
    w* = word holding the k-th bit (branchless prefix walk), k -= bits before it
    t_p = 64 w* + select(cand[w*], k)

Never empty: under a 2-colouring, the current tile was a candidate of every
neighbour's last update, so it is in cand whenever the state is consistent.
Start from a consistent state (a valid tiling, e.g. an exemplar patch or a
self-compatible tile).  An empty cand signals an inconsistent state (assert).

## 5. Site update, soft levels

### 5.1 Accumulate
    A = 0
    for d, g: acc_add(A, factor stack with plane i = eval3(row of Rb_g^d[i]), forbid = eval3(row of Rf_g^d))
    acc_add(A, unary stack)                   # per tile, or per site
    other per-site terms: acc_addc(A, level, mask)   # a constant level on a mask

acc_add is the bit-sliced ripple adder of bitgibbs.c (sum = a ⊕ x ⊕ c,
carry = a x ∨ c (a ⊕ x)); a carry past the top bit or a set forbid bit marks
the tile bad.

### 5.2 Draw
1. Counts: for l < NL, n_l = popcount(M_l), where M_l = legal ∧ ¬bad ∧
   ∧_b (A_b if bit b of l else ¬A_b).
2. Level: r = mulhi64(rand64, Z); walk l with a running prefix until
   r < Σ_{l' ≤ l} n_l' cw[l'].
3. Tile: k = (r − prefix) / cw[l] (uniform in [0, n_l)); rebuild M_l (one
   AND chain over the AB planes) and select its k-th bit.

Only the counts are kept for all levels; the chosen level's mask is rebuilt.
This avoids holding 32 × NW masks (register pressure on a GPU).

### 5.3 Exact float energies (optional)
If the port model approximates float energies E_true (e.g. learned pair
energies), use the quantized draw as a proposal and accept with
exp(−(E_true(t) − E_true(t0)) + unit (l(t) − l(t0))), as bg_sweep_pairs does.
The target is then exact and a poor fit shows up only as a lower acceptance
rate.  Requires cw = exp(−unit l).

## 6. More than 64 tiles: partition-restricted Gibbs

A schedule picks a partition Π_s of the tiles into blocks of ≤ 64 for sweep s.
Site p samples within B = block_of_{Π_s}(t_p), its *current* tile's block,
with the same update as §4/§5 on a one-word mask.

Correctness rule: the subset offered must be chosen with the same probability
whichever of its members is current.  A partition satisfies it (every member
of B gets B), so each step is exact Gibbs on B and preserves π; any sequence
of such steps preserves π.  Not valid: "a scheduled set of 64, plus the
current tile if missing" (moves out of it cannot be reversed).

Ergodicity: the union of the partitions used must connect all tiles (two
tiles are linked if some partition puts them in one block).

Schedule: mostly class-aligned partitions (tiles with the same port signature
or class in one block, so most of a site's conditional mass is in-block),
with a random or shifted partition every k-th sweep for ergodicity.

Data: per block, its own planes (32 words, 256 bytes); S/64 blocks per
partition, a few partitions resident.  The index block_of(t_p) differs per
thread, so these live in shared (threadgroup) memory.

Cost: the one-word kernel plus one indexed load (≈ S/64 times cheaper than
NW = S/64 words).  Price: slower mixing between tiles in different blocks
(probability ≈ 63 / (S − 1) of sharing a block under a random partition).
Never-empty and MH (§5.3) carry over (t and t0 are in the same block).

## 7. Auxiliary certificate fields

Hard constraints on site-level auxiliary variables enter as masks: the
neighbours' auxiliary values are scalars the thread reads, and they choose
*which* per-side masks are combined (OR / AND across sides is ordinary bit
logic; only the per-side masks must factor over groups).  Requirement: the
constraint depends on the tile only through a few masks (per-side join masks,
exit, node, opening on side d).

Worked instance: the connectivity certificate of bg_sweep_field.

Definitions (within one k × k block; neighbours across the block seam do not
count as witnesses):
- J_d: mask of tiles joined to n_d on side d, from a hard join relation over
  the port groups (one eval3 per group, as in §4).
- X_p: per-site exit mask (openings on class ports, nodes at a sink root).
- N: mask of node tiles.
- q = n_d is *eligible* for p if q is in p's block, g_q < INF and d_q < d_p.
- p is *valid* if g_p = INF, or t_p ∈ X_p, or some eligible q has
  g_q + [t_p not joined to t_q] ≤ g_p.
- q is a *dependant* of p if q is in p's block, g_q < INF, t_q ∉ X_q, and q
  has no valid witness other than p.

Joint draw of (t_p, g_p) given d_p and everything else.  For each g' ∈ {0 .. INF}:

    W_g' = ALL                                         if g' = INF
         = X_p ∨ (ALL if some eligible q has g_q + 1 ≤ g')
               ∨ OR_{eligible q on side d, g_q = g'} J_d      otherwise
    for each dependant q on side d:
        g' = INF or d_p ≥ d_q or g' > g_q  →  L = ∅
        g' = g_q                           →  AND J_d
        g' < g_q                           →  no constraint
    L_g' = legal_p ∧ W_g' ∧ (dependant constraints)

The energy of tile t at g' is A's level plus a g-cost of c_N(g') on nodes
and c_O(g') elsewhere (μ g' and ε g' in level units, must lie on the level
grid).  The accumulator A is built once; per g' only counts are needed:

    Z_g' = Σ_l n_l(L_g' ∧ N) cw[l + c_N(g')] + Σ_l n_l(L_g' ∧ ¬N) cw[l + c_O(g')]

(cw of an index ≥ NL is 0).  Draw g' by Z_g', then the level and the tile
within L_g' as in §5.2.

Then (g_p, d_p) | t_p exactly (scalar, as in bg_sweep_field): for each g, the
feasible d form an interval [lo, hi] (lo from the witnesses, hi from the
dependants); weight exp(−cost(t) g − δ d) summed in closed form over the
interval; draw g, then d within the interval.

Colouring: a site's update reads its neighbours' witnesses (distance 2), so
two sites updated together must not share a neighbour.  The 5-colouring
(x + 2y) mod 5 gives every same-colour pair Manhattan distance ≥ 3 (5 is the
minimum: a plus-shaped neighbourhood needs 5 distinct colours).  Pair-only
models (§4, §5) need just the checkerboard.  Total work is the same; only the
number of phases differs.

Never empty: the current (t_p, g_p) remains valid as long as the dependant
checks are reproduced exactly (port blockfield's valid_skip and its tests).

Other per-site terms of bg_sweep_field (block port patterns, seam rules,
sink-root penalty) are constant levels on masks (acc_addc) and carry over
unchanged.

## 8. Sweep schedule

    for sweep s:
        choose partition Π_s                        (if S > 64)
        for colour c in colouring:                  (2 or 5 phases)
            in parallel over sites p of colour c:
                read neighbours' port bytes (and aux values, 2-hop for §7)
                update t_p (§4 or §5), restricted to block_of(t_p)
                update (g_p, d_p) (§7)
                write t_p (and its 4 facing port bytes)

Random numbers: counter-based, a hash of (seed, site, sweep, phase, draw
index), so the result is independent of thread scheduling and position-hashed
for infinite worlds.

## 9. GPU mapping (Metal, M2)

- Registers are 32-bit; a u64 mask is two registers, 64-bit ops cost ~2.
  Consider writing the kernel natively on u32 halves.
- Arrays stay in registers only with compile-time indices: fix NW (= 1 with
  §6) at compile time, unroll word loops, and gather a dynamic word with
  x |= cand[w] & −(w == w*) instead of cand[w*].
- Do not keep 32 level masks (§5.2 counts then rebuilds).
- Planes and relations: identical for all threads → constant memory or
  registers.  Per-block planes indexed by block_of(t_p) → threadgroup memory.
- Tile/port grid and aux fields in device memory; ~10–20 bytes per site
  update, far below the bandwidth limit.
- One thread per site; one dispatch per colour phase.
- Prototype route from Python: MLX `mx.fast.metal_kernel`.

## 10. Cost estimates (unmeasured)

Base M2 (10 GPU cores): ≈ 1.8 × 10¹² 32-bit int ops/s.  Estimates include a
2–4× discount for occupancy, RNG and overheads.

| Kernel | 32-bit ops / site (NW = 1) | Site updates / s | 16 × 16 block sweeps / s |
|---|---|---|---|
| Hard (§4) | ~500 | ~10⁹ | ~4 × 10⁶ |
| Soft, FB = 2 (§5) | ~2 500 | ~2 × 10⁸ | ~10⁶ |
| Soft + aux (§7, G ≈ 8) | ~10 000 | ~5 × 10⁷ | ~2 × 10⁵ |

Unknowns: popcount throughput on Apple GPUs, register spills in the soft and
aux kernels (2–5× if they spill).  Scales roughly with core count (M2 Pro
~2×, M2 Max ~4×).

## 11. Validation plan

1. Ports from castle.json / castle_dense.json (socket + glue codes): rebuild
   hard compatibility from the port relations and diff against Dh / Dv and
   the current forbid planes (target: zero mismatches).
2. CPU reference in C: port-based bg_sweep_pairs and bg_sweep_field; compare
   single-site conditionals exactly against the table versions (bitgibbs.exact).
3. Partition scheme: on a tile set with S > 64, compare autocorrelation times
   against full NW-word Gibbs; report slowdown per sweep against speedup per
   sweep.
4. Metal prototype of §4 on a 1024² grid; measure site updates / s against
   bitgibbs_bench on one CPU core.

## 12. Open questions

- Learning the port codes and per-group tables from exemplar pair energies
  (alternate: fit tables by least squares with codes fixed; re-assign each
  tile side's code by enumerating 256 values with tables fixed).
- Mixing of hard models under single-site moves (rigid objects can freeze);
  block updates (2 × 2, row segments) within the same mask framework.
- Saturation budget: level ranges of the group tables against AB = 5.

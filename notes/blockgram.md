# Block-compiled split grammar: towers (2026-10-07)

`castlegen/blockgram/{core,towers}.py`, `notes/experiments/blockgram_towers.py`,
images/blockgram_towers{,_zoom}.png.  Follows the negative result in
channels_demo.md ("Towers of counted rooms").

Core: nodes are fixed-width int records (type, box, hashed seed, attrs);
each type declares the level at which it splits into children.  Levels 32
and 8 hold K = 64 slots per block; a block's slots are its parent's with
every record splitting at >= its level replaced by the children that
overlap it (broadcast, so replicas agree and no seam is checked).  Leaves
paint at level 1 in (z, seed) order.  Owner level (96-column regions):
the only sampled state, Gibbs with position-hashed noise, 6 sweeps, a
light-cone margin of 2 * 6 + 2 regions.

Grammar: region -> tower (n in 10..15 or absent, on a global 6-row story
lattice above the terrain), bridge to the next tower (owner Gibbs: both
towers must hold the story above their ground floor, the deck must clear
the terrain), trees.  Tower -> floor chain drawn exactly by backward
messages (state = 5 room widths in {5, 6, 8, 9}, stair room, stair
direction; the hole may not open onto a wall or the next stair; door rooms
and the roof footprint are unaries; soft terms favour floors that differ),
foundation, roof (battlements, pitched, turret, spire, dome), balconies,
entrance stairs.  Floor -> shell + 5 room leaves (stair or furniture by kind).

Result, 960 columns: 10 towers, every one valid by a checker that reads
tile types only (10..15 floors, 5 rooms and 1 stair per floor, a hole
above every top step, the top floor's to the roof).  15 chunks of 64
columns rendered from empty caches equal the whole render tile for tile.
Max slots per block 8 at level 32, 7 at level 8.  0.9 s per 960 columns.
A window at x = 10^6 is valid too.

One calibration lesson repeated: with a flat per-bridge bonus every gap got
a bridge, since 6..10 valid stories outnumber "none".  Fixed by making the
bonus the log-odds of a bridge and the story uniform (+ log #stories).

## Compiling the grammar to levels (2026-10-08)

`castlegen/blockgram/{compile,tower_spec}.py`, `notes/experiments/blockgram_compile.py`.
The spec declares choice instances (domain, deps, anchor, reads of the
ground as boxes, position flag) and hard factors as predicates; no level is
named.  The compiler samples feasible derivations and decides per kind:
reads -> finest level whose 3 x 3 block window holds every read box (E);
no reads but coupled -> collapsed with its dependents or, with none, its
finest partner (I); neither -> stateless.  Factors across levels: peers at
one level that are not local to each other cannot coordinate, so the coarse
side quantifies over them when that loses <= 25% of its values, else the
kind whose lift helps most is lifted; one local cluster -> existence
projection.  Finer parts read coarser ancestors through the broadcast frame.
Honourability: for alternative coarse values allowed by the projections, a
backtracking search one level down.

Result, 12 contexts:
- no bridges (2 s): 96 entrance, n, present, span, floor chain; 32 x, base;
  8 foot, path; room kinds stateless.  No decisions needed; 0 of 953
  coarse values without a completion.
- bridges (66 s): base lifted to 96 (two towers' bases cannot coordinate at
  32; quantifying loses 65%), n lifted (27%), bridges quantified over x and
  span (0% loss); x and span at 32 with the floor chain collapsed there;
  foot, path at 8.  0 of 1038 without a completion.
Against the hand design: same split (record / placement / footing), with
three differences that follow from the spec rather than from rules: the
entrance sits at 96 because its read covers both sides of the tower; base is
at 96 for every tower (a static compiler cannot express "only if bridged");
the floor chain collapses late (with span at 32) rather than in the record.
Bugs found on the way, all fixed: absent instances in the geometry, reads
written as "where is the ground" instead of "is ground in my box", a false
dependency (base on x through its anchor), stale decisions across rounds,
lifts undone by the collapse rule, clusters mixed across levels, random
search on chains in the honourability check.
Not yet done: executing the compiled plan (channels + kernels) and rendering.

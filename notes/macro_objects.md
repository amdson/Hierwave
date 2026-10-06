# Macro objects: discussion notes (2026-10-05)

A record of findings and ideas from the macro-object sessions.  Code:
castlegen/macroobj.py (door matching), castlegen/macrocontact.py (contact
connectivity); experiments in notes/experiments/macro_*.py.

## 1. Decoupling hard and soft updates (idea, not implemented)

Problem: hard variables (doors or contacts, overlap, connectivity) need many
sweeps; soft exemplar matching (kNN) needs only a handful of updates but is
expensive per update.  Doing both at every block visit pays the kNN cost on
every sweep.

Idea: make the match its own variable.  z = matched exemplar coordinate per
block or per window; x = rooms.

    p(x, z) ∝ exp(-E_hard(x) - E_soft(x, z))

E_soft(x, z) is an O(1) table lookup per candidate: agreement of the
candidate with the exemplar's room at z (+ the block's offset).  Alternate:
- many x-sweeps with z fixed: the hard work; the soft term adds almost no
  cost per candidate;
- an occasional z-update with x fixed: the kNN / k-coherence step,
  re-matching each block's or window's neighbourhood to the exemplar.

Still Gibbs: each step leaves the joint invariant, and nothing requires equal
update rates, so z can be updated as rarely as wanted (every ~10 sweeps, 3-5
times a run).  The kNN cost is paid a handful of times, not per visit.

The current pipeline is the one-update case: the pasted parent labelling is a
fixed z for the whole fine stage, and nu (penalty off the parent) is the O(1)
soft term.  Missing: re-matching z during the fine stage so it follows x
(today the fine stage edits 20-27% of blocks and keeps being pulled toward a
parent it has left).

Proposed test (full-scale map, contact mode):
1. coarse stage as now -> z0 and the parent;
2. fine stage in chunks of ~10 sweeps; after each chunk re-match z by
   k-coherence on the current rooms (candidates: neighbours' z shifted, plus a
   few random), at window or block level; rebuild the parent from z;
3. compare drift from the parent and components against the fixed-parent run.

Cautions:
- update z rarely relative to how fast x moves (a slowly moving target);
  frequent updates can oscillate;
- nu is all-or-nothing per block; a graded agreement (same family at a nearby
  phase, overlap of footprints) would let x meet z partway;
- the descriptor for matching room neighbourhoods is open: a rasterised
  coarse family-occupancy grid (simplest, fits ANN / k-coherence) or a set
  distance over (family, offset) (exact, slower).

Rough cost of kNN placements (35 min CPU baseline for 1000^2 blocks, ~40 us
per block update):

| where the matching sits | extra cost |
|---|---|
| inside every candidate's energy | 100-1000x (impossible with ~1300 candidates per block) |
| one brute-force query per block per sweep | +0.3-1 ms per block, 10-25x |
| the same with an ANN index | ~1.5-2x |
| k-coherence per block every sweep | ~1.2-1.5x |
| k-coherence every few sweeps, or per window | a few % |

Contact mode lowers the hard side too: no lam ramp, and the region test
reached one component within 4 sweeps, so the split might be 5-10 hard
sweeps per soft update.

## 2. Door matching vs contact connectivity (findings)

Door matching (macroobj): doors are part of the room variant (sealed
subsets); attachments need exact door alignment.
- A pure component-count energy fails on the 16x16 region test (5/56 at
  beta = 8; beta = 32 changed nothing): after ~10 sweeps the count never
  changes.  The forest heuristic (orphan penalty per room) gets 54/56, by
  dissolving orphans and regrowing, not by merging.
- In a state with no unmatched door, no single-block move can merge two
  components: every free-facing door after lifting block b was attached to b,
  so a new attachment only reaches b's own component.  Census of the stuck
  states: 0 merging moves out of ~3-13k placements, with either tile set.
- A larger palette (macro_palette: 24 families, random door layouts) does not
  change this; holding lam low longer helps count only slightly (palette,
  80% low: 7/16).

Contact connectivity (macrocontact): rooms are shapes (full variants); two
rooms are connected when >= minc slot cells of door-capable walls face each
other flush; doors are chosen at render.
- Census on the stuck door-mode states: opening every flush contact would
  leave 1-2 components (from 2-9).
- Region test: pure count 52/56, forest 51/56, plain 0/56; one component by
  the first check (sweep 4).  Every failure is a transient island at T = 1;
  two final sweeps at T = 0.1 fixed all four traced cases (QUENCH option).
- Exact block heat-bath over all full variants x 64 phases, ~3.5 s per
  region run (door mode ~1.6 s).

Full-scale door-mode map at 1000^2 blocks (macro_dense, the conn settings):
369k rooms, 5094 components, largest 2909, 97% of rooms in components >= 20;
35 min CPU.  The patchwork is the door-mode merging limit; contact mode at
full scale is the next comparison.

## 3. Other threads

- Structured outputs (e.g. a five-floor round tower, different rooms per
  floor): exact counts, global shape, all-different and cross-floor
  alignment are not local.  Hybrid: a split grammar (dsl.md section 3)
  emits regions (masks), fixed objects (stairs, required rooms) and port
  promises; the macro sampler fills each region under connectivity, as in
  the region test.  Needs: mask regions instead of rectangles, per-region
  palette / mu, fixed objects with ports, counts left to the grammar.
  Softer option: positional channels (radius, angle, floor) for style.
- Render-time doors currently use a global union-find over the contact
  graph; the proper version hands "these contacts may hold doors, these must
  connect" to the tile-level port / connectivity sampler as promises.

# Hierwave constitution

The goals behind design decisions. When a choice is unclear, check it
against these.

## Purpose
1. Generate 2D and 3D terrain for voxel worlds: structures (castles,
   cliffs, ...) that look like a given example, plausible locally and at
   every scale.
2. Enforce hard structural constraints an example cannot reliably teach
   (support, connectivity).

## What a correct output is
3. A *sample* from a distribution p, not an optimum: variety matters, so
   methods sample at T = 1 rather than taking the argmin.
4. The exemplar pipeline only approximates p, and the exemplar can be
   regenerated at will. It is never ground truth to match, and its
   statistics are not evidence for or against a scheme.
5. Plausible maps beat exact coverage: forbidding rare structures is fine
   if maps still look right.

## Locality (infinite worlds)
6. The sampler takes a fixed number of steps using local context only.
   Cost per cell is independent of world size; no information has to travel
   across the map.
7. Worlds are infinite, position-hashed and generated chunk by chunk, so no
   global check is allowed, not even a small one at the top level.

## How it is generated
8. Coarse to fine: each level refines its parent. Coarse levels are learned
   from exemplar windows, never hand-designed.
9. Hard constraints are carried as promises: a parent commits and its
   children can always honour it. Every promise is locally easy to resolve
   at every level.

## Non-goals
- Consistency across scales: a map generated at one scale need not match a
  finer map of the same place generated later. That would be nice, but it
  belongs among training goals, not constraints.

## Fixed state
- Every cell carries a fixed set of variables of fixed size (no
  variable-length lists such as per-part keys). Compromise on exact output
  statistics rather than grow the state.

## Method taste
10. General and principled: no rules specific to one tile set or example.
    Boundary conditions are allowed.
11. No hybrids and no clamps, except as temporary test scaffolding.
12. Quick, minimal experiments before building machinery.

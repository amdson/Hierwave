# P5: do joint block moves fix refinement under clamped coarse levels?  (p5, 2026-10-15)

**Answer: yes.** Joint blocks refine in 5-10 sweeps where single-cell Gibbs gets 1/32 (tight) and 2/16 (chain) in 200 sweeps.

Setup = cmix section 2 (`refine.py`): ground W = 128, lam = 20 binary penalty, levels >= d clamped at a valid state, levels < d uniform-random start (same starts per arm), "valid" = 0 violations by `colour_gibbs.state_energy`. Tight = 32 worlds (probe seeds 0..31); chain = the 16 worlds in `final_binary_lam20.npz` (only 16 exist). One run per arm. GPU 1 was shared with p10; SM clock 1560 MHz, no throttle (`runs/gpu_*.txt`).

## Table: worlds valid after k sweeps (tight n=32 / chain n=16)
| d (free levels) | kernel | k=5 | 10 | 20 | 40 | 200 | tile cells changed | ms/sweep |
|---|---|---|---|---|---|---|---|---|
| 1 (tiles only), tight | single-cell | 0 | 0 | 1 | 1 | 1 | 1.7 % | 2.1 |
| 1, tight | tile block 2x2 (16 joint) | 31 | 32 | 32 | 32 | 32 | 1.7 % | 2.1 |
| 1, tight | tile block 4x4 (65536 joint) | 32 | 32 | 32 | 32 | 32 | 1.7 % | 54 |
| 1, chain | single-cell | 0 | 1 | 1 | 2 | 2 | 1.4 % | 1.7 |
| 1, chain | tile block 2x2 | 15 | 16 | 16 | 16 | 16 | 1.3 % | 1.3 |
| 1, chain | tile block 4x4 | 16 | 16 | 16 | 16 | 16 | 1.3 % | 23 |
| 2 (levels 0-1 free), tight | single-cell | 0 | 0 | 0 | 0 | 21 (median first 69) | 2.4 % | 3.1 |
| 2, tight | parent+4 children (480 joint) | 3 | 32 | 32 | 32 | 32 | 2.5 % | 4.2 |
| 2, chain | single-cell | 0 | 0 | 0 | 0 | 12 (median 87) | 2.1 % | 2.8 |
| 2, chain | parent+4 children | 0 | 12 | 16 | 16 | 16 | 2.1 % | 2.6 |

Median first-valid sweep: d=1 blocks 5; d=2 block 7 (tight) / 9 (chain). Sweep times are JIT-warmed, synced, 32 worlds (tight) or 16 (chain), contended GPU, so only roughly comparable; a block sweep costs about the same as a single-cell sweep (2x2, parent+children) or 10-25x more (4x4).

## Reading
- **The clamp-respecting fix is the 2x2 tile block, not parent+children.** At d=1 the parent (level 1) is clamped, so a (1,1) block cannot be used there (it would resample level 1). The tile-only 2x2 block (`FineBlockKernel`, k=2, existing code, no new kernel) already supplies the paired flips the row-honour factors need: 5 sweeps. 4x4 gains nothing over 2x2 here.
- **Parent+children (`JaxBlockGibbs(p, reg, 1, 1)`) is the d=2 comparison** (levels 0 and 1 free, both unclamped): 32/32 and 16/16 valid by sweep 10-20 vs 21/32 and 12/16 after 200 for single-cell; first-valid ~8 vs ~70-90 sweeps. Free-level violations in the single-cell arm start in the thousands and take ~60 sweeps just to get near zero.
- Single-cell at d=1 stays stuck at 2.4-2.6 violations/world, reproducing cmix section 2 (it got a few valid at 15-20 sweeps, then none).
- Finished refinements stay close to the source (1.3-2.5 % of tiles differ), as in cmix.

## Dependency reach of one block sweep (fixed-step accounting)
A sweep = sequential colour classes; blocks of one class share no factor instance. Colour counts measured: parent+children block 4 classes, 2x2 tile block 2 classes (single-cell: 16 colours per channel per level). A cell's new value depends on its block's boundary and, through the earlier classes, on neighbouring blocks: reach is at most `ncol` block rings, i.e. <= 4 level-1 cells per side for (1,1) and <= 2 blocks (4 tiles) plus the random offset for 2x2. I did not measure the factor window width; a 5-10 sweep refinement multiplies these. For the finite-support invariant, state it as (sweeps) x (ncol) x (block + factor window), and test it with the perturbation test.

## Caveats / failures
- Offsets for FineBlockKernel are chosen by host rng per sweep (a mixture of invariant kernels; fine for training, for generation the offset schedule must be fixed).
- Tight source has no slack by construction; chain d=2 block needs 10-20 sweeps, not 5.
- Not run: section 1 four-start mixing test; graded penalty; d=4. One run per arm, no seeds.

## Next
Use the 2x2 tile block (or a fixed offset cycle) for P10's resample under a clamped level; budget ~10 sweeps.

Commands: `bash reportables/2026-10-15_p5/scripts/all.sh` (script `scripts/p5.py SOURCE ARM SWEEPS`; arms s1 f2 f4 s2 b11); logs `runs/all.log`, `runs/*.json`.

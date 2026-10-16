# dthin: thin-block Gibbs on dimers, 2026-10-16

## Answer: NOT ANSWERED (no sweeps ran). Shape support and completion counts only.
No h/v/d numbers exist. Every attempted run died with CUDA OOM before the first sweep finished.

## Shape support (kernels take square k x k only; thin blocks built in `castlegen/variants/_dthin.py`, 46 lines, subclass of FineBlockGibbs + orientation-alternating ThinKernel)
Completions per tile (internal-pruned, dimers, p domain 6, edge designed). Cap in FineBlockKernel/internal_pruned is 2,000,000.
| shape | completions | under 2M cap |
|---|---|---|
| 3x4 | 169,764 | yes |
| 2x6 | 580,792 | yes |
| 3x5 | 1,834,326 | yes |
| 2x7 | > 2.4M (at slot 12) | no |
| 2x8 | > 30.7M (at slot 15, partial) | no |
| 2x10 | > 36M (partial) | no |
| 2x12 | > 48M (partial) | no |
| 3x6 | 19,889,698 | no |
| 3x8 | > 30.7M (partial) | no |
So of the requested shapes only 2x6 is under the cap; 2x8, 2x12, 3x6 are not (thin does not buy much on this problem: 2x8 is already above square 4x4's 2.9M). Also W=64 is not divisible by 6 or 12; tiles use nx = 64//kw with a random offset per kernel (valid p*-invariant mixture, some columns uncovered per offset).

## Runs (all failed)
Note: gpu_after shows GPU 0 at 1 MiB at the end, so the 12 GB holder (possibly my own first killed process lingering) was gone only after the time cap; a rerun is now possible.
Command: `taskset -c 16-23 python reportables/2026-10-16_dthin/scripts/dthin_stage.py runs/t2x6.json <nworlds> 2x6` (adapted from dgate_stage.py; 3 offsets per orientation, lam 20, alternate 2x6 / 6x2 per sweep, starts as dgate).
GPU 0 was NOT idle: another process (not mine, host pid 2398259) held 12.2 GB of the 16 GB for the whole session (nvidia-smi, runs/gpu_*.txt). With ~4 GB free:
- 64 worlds: stalled in OOM retries in the start-generation step.
- chunked 16 and 4 worlds: `ncomps [580792, 580792]` built, then RESOURCE_EXHAUSTED allocating 1.73 GiB (jit_squeeze, constant in the number of worlds) in the first sweep. So 2x6 does not fit in 4 GB at any world count with this code; 3x5 (3x more completions) would not either.

## Next (one line each)
- Rerun 2x6 on a GPU with >= 8 GB free (or once the 12 GB holder exits); expected cost is far above dgate's 85 ms/world (580k vs thousands of completions).
- Prior for the answer: likely no, since the only in-cap thin shape (2x6, 3x5) carries a chain only 5-6 cells long, against the sector change needing chains of length ~ the 32-row height.
- A chain-rotating move for dimers would need a shape beyond the enumeration cap (needs non-enumerative block exact sampling, e.g. a transfer-matrix over a 2 x W strip); that is a generic 'exact strip' kernel, still shapes from the grid (constraint 1 holds: thin blocks are grid shapes, not a dimers-specific move).

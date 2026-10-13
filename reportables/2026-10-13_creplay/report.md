# creplay: coupled replays for many coarse draws (stopped early)

2026-10-13 · `castlegen/variants/creplay.py`, `_creplay_core.py`, `_creplay_clamp.py`, `tests/variants/test_creplay.py`, `scripts/timing.py`.

**Status: stopped early. No bench runs. The code is partway through a refactor and does not currently run** (details below).

## What was built
- **Estimator.** Built on cfast. At the top `blev` levels, R_l draws per world are chosen uniformly and independently of the trajectory (all of them by default at the top level). Each gets K actions: all of A when |A| <= K, otherwise the drawn action plus K−1 i.i.d. draws from q. Each action is replayed from scratch under the world's coordinate-keyed noise and scored by the draw's cone return (cfast's `Cone`). Enumerated draws get the exact Σ_a ∇q(a)[ℓ(a)+G^a]; sampled draws use the all-action term plus leave-one-out.
- **Batching.** Selection, action sampling, replays, cone sums and the losses all run in one jitted function. Replays are vmapped over (world, draw, action) and chunked with `lax.map`.
- **Replay faithfulness: 4/4 tests pass** (toy_support, columns, ground, dimers; on the box, CPU). The checks:
  - Draws outside the replaced draw's cone are bit-identical to the original pass.
  - Costs outside the cone match to float32 round-off. They are not bitwise equal, because XLA's MLP reduction order depends on the vmapped batch size.
  - Replaying the drawn action reproduces G_i.
  - The batched cone return equals the brute-force sum over the cone's members.
- **Gradcheck** (tiny enumerable problem, `_gradcheck` machinery, 300 × 16 worlds): **all 4 configurations pass** on the pre-refactor code: every draw branched and enumerated (K = 4), top level sampled with leave-one-out (K = 2), random subsets at both levels, and a wrong fixed critic.
- **No windowed replay.** At the branched levels the cones cover 94–99.9 % of later draws on ground, and these draws come first in the schedule, so a window would save almost nothing.

## Numbers (GPU box, ground, B = 8)
- cfast-equivalent update (no branches): **0.028 s**, 57 s compile.
- Replays at R = (64 at level 4, 64 at level 3), K = 4, which is 4096 full-pass replays per update: **6.1 s/update** and **222 s compile**. The compile alone is longer than the 150 s budget, so that configuration cannot be benched as-is.
- Per-draw signal against noise, from the replay diagnostics at init:
  - **Across-action spread of the cone return** (std over the K actions): about 27 at level 3 and 39 at level 4.
  - **Spread of the return itself across draws:** about 1500 and 1000.
  - **What that means:** a coupled comparison removes roughly (1000/30)² ≈ 10³× the variance per branched draw. This is consistent with cfast's coarse SNR of about 0.02.
- **Other diagnostics:**
  - replay_err = 0: replays reproduce the original returns.
  - **Enumerable draws** (|A| <= 4): 22 % (l4), 32 % (l3).
  - **Forced draws** (|A| = 1): 15 % / 27 %.
  - **Mean |A|:** 152 / 40.

## Key design insight (the half-done refactor)
- **Why partial branching does little as built.** With R < N_c, the unbranched draws keep their score-function terms, and their noise dominates the level's gradient.
- **The fix being written.** At a branched level, keep the low-variance all-action term t1 for every draw. Drop every draw's score term t2. Add (N_c/R)·Σ_selected c_i, where c_i is the coupled replay term (enumeration Σ_a ∇q(a)(G^a−Q(a)), or LOO). This is unbiased under uniform selection that is independent of the trajectory. It lets R be small, e.g. 16 per level, about 1 s/update.
- **Code state.**
  - Done: the single batched replay for all levels (`replay_all`, cone tables padded to the longest), which cuts compile.
  - Not done: `actor_loss` still reads `r["sel"]`, which no longer exists.
  - To finish: `branch_terms` should return c_i, and `actor_loss` should drop t2 at branch levels and add w_l·Σ c_i. Then rerun `test_creplay.py`, adding the `loo_subset` configs to the importance-weighted path.

## Coordinator's question (fill-following contexts)
I wrote `_creplay_clamp.py` (`clamped_pass`): the training pass with every draw at levels >= L clamped to given values, e.g. a probe's lifted coarse channels. It was not run. The planned measurement:
- Train cfast and creplay for 150 s each.
- Compare finer-level cost per cell and violations under probe-clamped coarse states (L = 4 and 3) against the model's own coarse states.

**This question is open.** If the fine levels fail in fill-following contexts, stronger coarse credit will only make avoiding partial fills more efficient.

## Next
1. Finish the importance-weighted refactor and rerun the gradcheck.
2. Time R = (16, 16), K = 4 on the GPU.
3. Bench cfast and creplay on the box: ground, columns, toy_support, dimers.
4. Run the clamp diagnostic.
5. Measure coarse SNR per world before/after (the `grad_variance.py` method, with `agrad`).

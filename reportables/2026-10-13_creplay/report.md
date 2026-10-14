# creplay: coupled replays for many coarse draws

2026-10-13 · `castlegen/variants/creplay.py`, `_creplay_core.py`, `_creplay_clamp.py`, `tests/variants/test_creplay.py`, `scripts/timing.py`.

**Status (round 3, 2026-10-14 01:00):** the importance-weighted refactor is finished and passes all tests (4/4 replay faithfulness, 4/4 gradcheck). First GPU benches against cfast are below: **creplay is not better than cfast on either problem**, mostly because it is 25–30× slower per update on ground and its compile (140–350 s) eats the 150 s budget of the first seed. Ground still collapses to all dirt. The round-2 notes follow the new section.


## Round 3 (2026-10-14): importance-weighted estimator, tests, first GPU benches

GPU shared with 3–5 other agents' jobs (100 % utilisation, up to 24 GB in use) throughout; all timings are contended.

### Estimator (done)
Per draw, sum_a grad q(a)[ell(a)+G^a] = t1 + c_i with t1 = sum_a grad q(a)[ell(a)+Q(a)] (exact) and c_i the return-dependent part.
- Non-branched levels: cfast unchanged (t1 + score term t2, subsampled at `sub` = 64 sites per step with weight n/m).
- Branched levels (top `blev` = 2): t1 for every draw (same subsampling), **every t2 dropped**, plus (N_c/R)·Σ over the R uniformly selected draws of c_i:
  - enumeration (|A| <= K): Σ_a grad q(a)(G^a − Q(a));
  - sampled: (1/K) Σ_k grad log q(a_k)[(G^k − Q(a_k)) − mean_{j≠k}(G^j − Q(a_j))].
- Selection is uniform without replacement over the structurally valid (channel, colour, site) triples, independent of the trajectory (validity is structural only), so the estimator is unbiased (hard constraint 3). Critic loss unchanged (cone-return MSE plus MSE on the replayed actions).
- Code: `branch_terms` returns Σ c_i; `actor_loss` drops t2 at branch levels and adds the weighted sum. Default config is now `R=(8, 8)`, `Kr=4`.

### Tests (CPU, cores 16–23)
- `test_replay_faithful`: **4/4 pass** (toy_support, columns, ground, dimers), 7 min.
- `test_creplay_unbiased`: **4/4 pass** (enum_both, loo_top, loo_subset, wrong_critic; 300 × 16 worlds), 8.7 min. loo_subset (R = (2, 5) of N_c) and wrong_critic (R = (3, 6)) exercise the importance-weighted path with R < N_c. No tolerances changed.

### Timing (GPU, ground, B = 8, K = 4)
| R (l4, l3) | replays/update | s/update | compile |
| --- | --- | --- | --- |
| (16, 16) of (64, 256) | 512 | **2.03** | 226 s |
| (8, 8) | 256 | **1.43** | 306 s |
| bench, seed 1, R = (8, 8) | 256 | 0.75 (201 updates in 150 s) | — |
| cfast (bench, same time) | 0 | 0.074 (2028 updates) | 42 s |

On toy_support, R = (8, 8): 0.033 s/update, 105 s compile. No configuration reached ≤ 1 s/update in isolation on ground; (8, 8) was the closest, so it is the default. Replay diagnostics at init, ground R = (16, 16): replay_err ≤ 0.004, enumerable 26 % (l4) / 29 % (l3), forced 16 % / 23 %, mean |A| 149 / 40, across-action spread of the cone return 28 / 26 vs its across-draw std 1009 / 1593 (unchanged from round 2).

**Compile is not cached for creplay.** The timing runs compiled the same configuration minutes before each bench, yet the bench's first seed spent 350 s (ground) and 143 s (toy_support) in setup. cfast's warm setup is 16–42 s. Not diagnosed. (Note: cfast.py points the cache at `~/.cache/jax_cfast`, overriding `JAX_COMPILATION_CACHE_DIR`; the creplay entries do land there.)

### Bench (2 seeds × 150 s, tag `creplay_pair`, run back to back / concurrently on the shared GPU)
| problem | variant | updates (s0 / s1) | worst final viol | F (s0 / s1) | accepted |
| --- | --- | --- | --- | --- | --- |
| ground | cfast | 1142 / 2028 | 1.5e-3 (l3, s0); s1 = 0 | 0.889 / 0.694 (mean 0.792) | no (probes, roughness across seeds) |
| ground | creplay | **0 / 201** | 6.3e-2 (s0, untrained); s1 4.7e-3 | 2.048 / 1.204 | no |
| toy_support | cfast | 4488 / 6001 | 2.4e-4 | −0.417 / −0.438 (KL 0.051 / 0.031) | no (stats, probes) |
| toy_support | creplay | **167 / 3953** | 2.0e-3 (s0) | −0.383 / −0.443 (KL 0.086 / 0.026) | no (stats, probes) |

- Seed 0 of creplay is effectively untrained on both problems (compile ate the budget), so only seed 1 is a comparison.
- toy_support seed 1: creplay KL 0.026 with 3953 updates vs cfast 0.031 with 6001: about equal, within seed noise.
- ground seed 1: 201 updates are too few to judge the estimator. Final violations 4.7e-3 (l2) vs cfast 0.
- **Collapse on ground: yes.** creplay s1 is 98 % dirt (roughness 1.17, ragged air pockets only at the top; `images/creplay_ground_s1.png`); cfast s0 is 97 % dirt with 1-cell stripes, s1 100 % dirt (`images/cfast_ground_s0.png`, `_s1.png`). creplay's lower fill_err (0.014 vs 0.05–0.06) comes from having trained less (the untrained seed 0 has 0.017), not from following the fill.
- Columns: not run. The first launch (with the SNR job) died silently with empty logs at about 00:50 (cause unknown: not OOM-reported; possibly another agent's pkill); no time to retry.

### Coarse SNR per world (ground, at init, critic zeroed, 24 worlds, B = 1)
`scripts/snr.py` uses the grad_variance.py method with `agrad`: the same worlds and parameters, comparing the cfast-equivalent estimator (blev = 0) with creplay at R = (8, 8), K = 4. SNR = |mean|² / E|g − mean|² per world, using the pooled mean. Raw numbers are in `snr_ground.json`.

| level | var cfast | var creplay | variance ratio | SNR cfast | SNR creplay | mean diff² / se² |
| --- | --- | --- | --- | --- | --- | --- |
| 4 (branched, 8 of 64) | 5.4e7 | 3.9e3 | **13 900×** | 0.010 | **142** | 0.99 |
| 3 (branched, 8 of 256) | 8.8e6 | 1.3e3 | **6 800×** | 0.014 | **93** | 1.31 |
| 2 | 1.08e6 | 1.08e6 | 1.0 | 0.023 | 0.023 | 0 |
| 1 | 4.7e4 | 5.5e4 | 0.86 | 0.023 | 0.020 | 0.65 |
| 0 | 56.6 | 63.0 | 0.90 | 0.055 | 0.050 | 0.24 |

- **Means agree** at every level (diff²/se² ≈ 1): a second check of unbiasedness, this time on ground.
- **Branched levels:** per-world SNR rises from about 0.01 to about 100, so one ground world now gives a usable level-3/4 gradient with only 8 of 64 / 256 draws replayed. The gain has two sources:
  - dropping the 56 / 248 unbranched score terms (t1 is exact);
  - the coupled comparison: across-action spread is about 25, against an across-draw spread of 1000–1600.
- **Unbranched levels:** levels 0–1 differ only through a different subsample (the random key is split differently), and level 2 is identical. Levels 0–2 keep an SNR of 0.02–0.05, so they are now the noisy levels.
- **Caveats:** measured at init only, with the critic off, on 24 worlds.
- **The estimator does what it was meant to do. The bench failure is throughput** (201 vs 2028 updates), not variance.

### What I'd do next
1. Make the replays cheap enough for ~1000 updates in 150 s on ground: the cost is about 0.8 s fixed plus ~2.3 ms per replay at B = 8 (each replay is a full sequential pass). Options inside the allowed freedoms: fewer actions (K = 2), branching only the top level, larger B to amortise the fixed cost, or windowed replay at level 3 (cones cover 94 %, so small gain).
2. Find why the creplay compile misses the persistent cache; until then the first bench seed is untrained.
3. Then rerun ground and columns for equal update counts, and the clamp diagnostic (`_creplay_clamp.py`, still unrun) which asks whether fine levels can realise fill-following contexts at all; exactopt's toy_modes result says variance limits there, but ground's collapse may still be the representation problem.

## Round 2 notes (2026-10-13)

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

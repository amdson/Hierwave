# creplay: coupled replays for many coarse draws

2026-10-13 · `castlegen/variants/creplay.py`, `_creplay_core.py`, `_creplay_clamp.py`, `tests/variants/test_creplay.py`, `scripts/timing.py`.

**Status (round 3b, creplay2, 2026-10-14 01:50):** replays are now cheap enough to compare at equal wall clock: **0.18 s/update** on ground (R = (2, 2), K = 4; was 1.43 s at R = (8, 8)), 1106–1151 updates in 150 s against cfast's 1399–1545, and warm compiles now hit the cache (setup 31 s ground, 13 s toy_support; was 140–350 s). Coarse SNR at the new default is still ×1200 (l3) / ×3400 (l4) over cfast. **It changes nothing that matters on ground:** both variants collapse to dirt with 1-cell air stripes and line_match 0.5 (no fill following), and at equal updates (300) creplay's F is equal or worse (1.43 / 1.29 vs 1.10 / 1.27). On toy_support creplay is worse (KL 0.057 vs 0.026–0.042). Round-3b section first, then the earlier round-3 and round-2 notes.

**Status (round 3, 2026-10-14 01:00):** the importance-weighted refactor is finished and passes all tests (4/4 replay faithfulness, 4/4 gradcheck). First GPU benches against cfast are below: **creplay is not better than cfast on either problem**, mostly because it is 25–30× slower per update on ground and its compile (140–350 s) eats the 150 s budget of the first seed. Ground still collapses to all dirt. The round-2 notes follow the new section.

## Round 3b (creplay2, 2026-10-14): cheap replays, compile cache, benches

GPU shared with 3–5 other agents' jobs throughout (other benches, cbig, cresid); every timing is contended and moved by ±50 % between runs. Scripts: `scripts/profile.py`, `scripts/profile2.py` (interleaved replay timings), `scripts/timing.py` (now imports the bench's variant registry first), `scripts/snr.py` (optional output tag), `scripts/table.py`, `scripts/eqplot.py`.

### 1. Where an update's time goes (ground, B = 8)
| part | time | note |
| --- | --- | --- |
| original pass, 8 worlds (`run_pass`) | 15 ms | |
| fused update without branches (blev = 0, = cfast's estimator) | 56 ms | |
| replay pass, N worlds (`replay_costs`, min of 4 interleaved rounds) | N=32: 34 ms · 64: 48 · 128: 95 · 256: 226 · 512: 508 | ~30 ms latency floor, then **~0.75–1.0 ms per replay**, linear |
| one replay split by level (N = 256, level-restricted passes) | l4 1.4 ms · l3 2.4 · l2 6.6 · l1 28 · l0 37 | **levels 3–4 are 5 % of a replay** |
| cone integral of the replay costs | 0.2–3 ms | negligible |
| losses and both Adam steps | fused − batch ≈ 0 | negligible |

- **The "0.8 s fixed cost" of round 3 was not real.** It was a two-point line fit under heavy contention. An update is the base update (~0.06 s) plus the replay pass, which costs ~30 ms of step latency (144 sequential steps) plus ~0.8 ms per replayed world, all of it compute in levels 0–2.
- **Starting replays from the stored state at the branched step would save ≤ 5 %.** Branched draws are at levels 3–4, the first 64 of 144 steps, and those steps are cheap; the fine levels below must be recomputed because the cones cover 94–99.9 % of them. Not built.
- **lax.map chunking:** not the cost. With one chunk the loop is now skipped (`nch == 1` path); no measurable change, kept.
- **Designed inputs (host):** not measured separately; cfast pays the same.

### 2. Cutting the cost (engineering only, estimator unchanged)
`fused` timing per configuration (`scripts/timing.py`, same process, contended GPU; compile is cold, in the bench's cache dir):

| config | replays/update (B·ΣR·K) | s/update | compile |
| --- | --- | --- | --- |
| blev = 0 (cfast estimator) | 0 | 0.056 | 54 s |
| R = (8, 8), K = 4 (round-3 default) | 512 | 0.43 (1.43 in round 3, heavier contention) | 27 s* |
| R = (1, 1), K = 2 | 32 | 0.165 | 120 s |
| R = (2, 2), K = 2 | 64 | 0.20 | 128 s |
| **R = (2, 2), K = 4 (new default)** | 128 | **0.18** | 91 s |
| R = (1, 1), K = 2, B = 32 | 128 | 0.265 | 130 s |

\* partly cached. (Round 3's table counted half the replays.) Below ~128 replays the cost sits at the latency floor, so (2, 2) with K = 4 costs the same as (1, 1) with K = 2 and carries more signal. Larger B amortises nothing extra: B = 32 at the same 128 replays is slower (0.265 s).

Coarse per-world SNR at init (24 worlds, critic zeroed, `snr_ground_r11k2.json`, `snr_ground_r22k4.json`):

| level | cfast | R=(1,1), K=2 | R=(2,2), K=4 | R=(8,8), K=4 (round 3) |
| --- | --- | --- | --- | --- |
| 4 | 0.010 | 11 (×1100) | **35 (×3400)** | 142 |
| 3 | 0.014 | 1.3 (×95) | **17 (×1200)** | 93 |
| 0–2 | 0.02–0.06 | unchanged | unchanged | unchanged |

Means agree with cfast at every level (diff²/se² 0.2–1.3). Tests after the change (CPU, cores 16–23): replay faithfulness toy_support + ground, gradcheck enum_both + loo_subset: **4/4 pass** (6 min; columns, dimers, loo_top and wrong_critic not rerun: the change only touches the one-chunk path and the default R).

### 3. Compile cache: found and fixed
- **Cause:** different cache directories, not different shapes. `cpass.py`/`cfast.py` set `~/.cache/jax_cpass` then `~/.cache/jax_cfast` at import; `cbig.py` resets it to `$JAX_COMPILATION_CACHE_DIR` at import. The bench imports every variant (`load_variants`), so it reads and writes `/workspace/jax_cache`; round 3's timing script imported only creplay, so it warmed `~/.cache/jax_cfast`. Verified by printing `jax.config.jax_compilation_cache_dir` both ways.
- **Fix:** `scripts/timing.py` (and `profile2.py`) call `load_variants()` first. Not changed in cfast/cpass (not mine). Agents warming a compile should use a bench smoke run or import the registry first.
- **Effect:** creplay bench setup 31 s on ground (cold smoke: 83 s) and 13 s on toy_support (cold: 73 s), against 140–350 s in round 3. The second seed reuses the in-process compile (0.3 s).

### 4. Bench (2 seeds × 150 s, tag `creplay2`, run back to back, shared GPU)
| problem | variant | updates s0 / s1 | worst final viol s0 / s1 | F s0 / s1 | stats | accepted |
| --- | --- | --- | --- | --- | --- | --- |
| ground | cfast | 1399 / 1545 | 0 / 7.6e-6 | 0.716 / 0.970 | dirt 1.00 / 0.97, line 0.5, fill_err 0.06 / 0.07 | no |
| ground | creplay | 1106 / 1151 | 9.8e-4 (l3) / **3.1e-2 (l3)** | 0.763 / **1.824** | dirt 0.95 / 0.89, line 0.5 / 0.51, fill_err 0.08 / 0.09 | no |
| toy_support | cfast | 7144 / 6870 | 1.2e-4 / 1.2e-4 | −0.443 / −0.427 (KL 0.026 / 0.042) | | no |
| toy_support | creplay | 4569 / 4271 | 1.2e-4 / 0 | −0.412 / −0.412 (KL 0.057 / 0.057) | | no |

- creplay now gets 75–80 % of cfast's updates on ground and 60–65 % on toy_support.
- **Ground seed 1 of creplay never settled:** level-3 raw violations swung between 0 and 0.13 for the whole run, λ hit the cap (20), and the cost per world rose from 11k to 15.7k. Seed 0 settled (cost 6.4k, raw violations 0 at the end).
- **Images:** both creplay seeds are dirt with narrow air columns hanging from the top (`images/r3_creplay_ground_s0.png`, `_s1.png`), the same collapse as cfast (`images/r3_cfast_ground_s0.png`, `_s1.png`). No fill line.

### 5. Equal updates on ground (300 each, tag `creplay2_eq300`)
`creplay_cap300` (default config) against `creplay_b0_cap300` (blev = 0: cfast's estimator in the same code). Both stop at 300 updates and anneal τ by update count.

| | creplay s0 / s1 | cfast estimator s0 / s1 |
| --- | --- | --- |
| F | 1.427 / 1.292 | 1.095 / 1.272 |
| worst final viol | 2.2e-3 (l3) / **0.28 (l4)** | 3.7e-3 (l2) / 2.6e-3 (l2) |
| dirt_frac | 0.95 / 0.87 | 0.96 / 0.97 |
| line_match | 0.50 / 0.50 | 0.50 / 0.50 |
| fill_err | 0.016 / 0.048 | 0.058 / 0.043 |
| roughness | 5.8 / 5.8 | 1.9 / 2.1 |

![equal-update curves](images/r3_eq300_curves.png)

- **More coarse signal does not change ground's outcome.** At 300 updates neither variant follows the fill (line_match 0.5 everywhere). creplay is slightly less dirty and rougher, its F is equal or worse, and seed 1 ends with level-3 raw violations rising and 28 % violations at level 4 after repair (repair pushes them to the top).
- **The coarse levels move, but toward avoidance.** Precise level-3/4 gradients follow the downstream costs, and those costs come from the fine levels: they cost +1.3–1.9 nats/cell in fill-following contexts (clamp's finding). Exact credit for that cost only teaches the coarse levels to avoid partial fills more reliably, as the round-2 hypothesis predicted. The levels that now limit ground are 0–2: low SNR (0.02–0.05) and limited capability in ragged contexts.
- The level-3 violation swings (seed 1 in both the 150 s and the 300-update runs) suggest the coarse step size is now too large for the cleaner gradient. A lower coarse learning rate is an allowed freedom; it was not tried.

### What I'd do next
1. Don't spend more on coarse variance for ground until the fine levels can build fill-following contexts (cresid's work). Then rerun the equal-update comparison: creplay should matter only once the downstream costs reward following the fill.
2. Further engineering, both unbiased and not built:
   - Reuse the drawn action's return G0 for replay slot 0 (saves 1/K of replays).
   - Select branch draws only among non-forced ones (|A| ≥ 2, 16–23 % are forced), weight N_nf/R. A forced draw's c_i is identically zero, so this stays unbiased; it needs a gradcheck.
   - Fold the replays into the original pass's batch to pay the ~30 ms step latency once.
3. Try a smaller coarse learning rate with creplay (the level-3 oscillation).
4. Columns and dimers are still not benched with creplay.



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

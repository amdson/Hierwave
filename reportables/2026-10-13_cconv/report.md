# cconv (ideas #5): first GPU bench, loses to cfast everywhere

2026-10-13 · `castlegen/variants/cconv.py`, `_cconv_core.py`, `tests/variants/test_cconv.py` (15 pass on the box). The agent's text, condensed by the orchestrator.

See Bench results below (added 2026-10-14).

## Built

- **Trunk, one per level.**
  - Inputs: the level's channels (zeroed where unknown) with known and exists flags, the parent planes upsampled 2×2, position in parent, and the row fraction.
  - Layers: a 3×3 stem, then r − 1 residual blocks `h += conv3x3(SiLU(LN(h)))`. Off-grid cells are zero-padded, and x wraps on periodic strips.
  - Precision is fp32 `HIGHEST`, because TF32 made the pass and the loss disagree on log q by 1e-3.
- **Heads:** a full-domain linear head plus a candidate-scoring head, which reads the site features and, for each candidate, its embedding, s(t), min(nv, 3) and min(look, 3). Both start at zero.
- **Training pass:** cfast's, with the trunk recomputed densely at every step. The loss re-runs the trunk on (2r+1)² patches at the subsampled sites.
- **Cone and margin** use the conv reach.
- **Defaults:** B = 64, r = 4, width 32, lr 1e-3.

## Tests

- The dense trunk equals the patch trunk.
- The loss recomputes the pass's log q, and scoring matches the pass.
- Every change caused by replacing one draw lies inside its cone, on all 5 problems.
- Overlapping open windows agree (toy_support, ground).

## Next (old plan, superseded by the bench section)

1. Profile seconds per update at B = 64 on ground; compiles are slow (about 50 s for generation).
2. Bench toy_support (the gate), dimers, ground and columns against cfast on the box, under `taskset`.
3. Annotate ideas #5.

## Bench results (2026-10-14, RTX 3090, SHARED with 3-5 other agents' jobs)

Wall-clock budgets are contended, so compare updates, not seconds. Default 2 seeds x 150 s, cconv then cfast back to back per problem (two problems at a time on cores 32-35 / 36-39). cconv: B=64, r=4, width 32 (ground and columns: variant `cconv_b16`, B=16, see below). cfast: its default B=8. Different B means updates are not like for like; cconv updates are ~8x (or 2x at B=16) bigger.

Smoke (toy_support, 1 seed, 60 s, warm cache): 242 updates, setup 10.5 s, F -0.395. Passes the 50-update bar, so no radius/width change was made. Profile at B=64 (warm): r4/w32 0.63 s/update, r2/w16 0.20 s/update. Cold compile of the fused step is the real cost: 130-240 s (first-ever compile ate a whole 60 s budget: 0 updates). I also removed a line in cconv.py that pointed the JAX cache at ~/.cache/jax_cconv, so it now uses the shared cache.

| problem | variant | accepted | worst final viol | raw viol | F/cell (s0, s1) | updates s0 / s1 | probes | seeds agree |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toy_support (F* -0.4688) | cconv | no | 1.8e-2 (l1) | 3.9e-3 | -0.3955, -0.3953 | 170 / 317 | flagged (s1; s0 none valid) | yes |
| | cfast | no | 2.0e-3 (l1) | 3.7e-4 | -0.4159, -0.4063 | 2579 / 6018 | flagged | yes |
| toy_modes (F* 0.2513) | cconv | no | 0 | 0 | 0.3348, 0.2866 | 220 / 368 | flagged | NO |
| | cfast | no | 0 | 0 | 0.2683, 0.2681 | 4271 / 6108 | flagged | NO |
| ground | cconv_b16 | no | 7.6e-2 (l4; 7e-3 at l0) | 5.0e-2 | 2.05, 2.11 (incl. violations) | 0 (compile 240 s) / 75 | none valid | yes |
| | cfast | no (budget ok) | 1.5e-5 | 2.7e-5 | 0.800, 0.714 | 1312 / 1589 | flagged | NO |
| columns | cconv_b16 | no | 0.18 (l1) | 0.19 | -1.03, -1.08 | 95 / 85 | none valid | yes |
| | cfast | no | 3.0e-2 (l1) | 1.5e-2 | -0.043, -0.018 | 7974 / 7398 | none valid | NO |

Exact stats mismatch for both variants on toy_support, toy_modes and columns. Ground has no exact stats.

## Plain statements

- cconv is worse than cfast on all four problems at equal wall clock: toy_support F -0.395 vs -0.416 (KL about 0.07 vs 0.05 nats/cell), toy_modes F 0.29-0.33 vs 0.268, ground and columns far worse.
- It is much slower per step: 170-370 updates in 150 s against 2.6k-8k for cfast, about 5-10x fewer updates even with 8x the batch on toy problems. It is probably undertrained rather than shown to be a worse representation; this bench cannot separate the two. B=64 on ground ran out of GPU memory (7 GB allocation on a shared card), so ground and columns used B=16, which is a deviation from the default.
- Ground: cconv does NOT collapse to all dirt (cfast does, all dirt as before), but it is not good either: mostly dirt with vertical air streaks, no visible fill line, violations 0.7-7.6% per level against the 2e-4 budget; seed 0 got zero updates because the cold compile took the whole budget, so that seed is the untrained net. Ground comparison is therefore weak (75 updates on one seed).
- Columns: cconv_b16 (~90 updates) has 18% violations, an image of patchy dirt/air.
- cfast toy_modes was killed once by another agent's pkill and rerun (same conditions, a few minutes later).
- Image caveat: images/bench is overwritten by anyone running the same variant and problem; cfast_ground_s0 may come from the baselines agent's run (all dirt either way).

Images: `images/cconv_b16_ground_s0.png`, `cfast_ground_s0.png`, `cconv_b16_columns_s0.png`, `cfast_columns_s0.png`, `cconv_toy_support_s0.png`, `cfast_toy_support_s0.png`, `cconv_toy_modes_s0.png`, `cfast_toy_modes_s0.png`.

## Next

Equal-updates comparison needs a faster step: r=2/width 16 is 3x faster (0.20 s/update at B=64); one run at that setting with 600 s on toy_support would say whether the gap is throughput. Not run (time box, and one change allowed).

# GPU baselines: greedy, cfast, cpass on the Vast RTX 3090

2026-10-13/14 · agent "baselines" · no code changed. Bench: `python -m castlegen.bench run <variant> <problem>` (2 seeds x 150 s, harness 22dadfc76132). Board: `results/board.md` (regenerated after every problem).

**Caveat: the GPU was shared** with up to four other agents' jobs throughout, and all runs used 8 pinned CPU cores (56-63). Wall-clock budgets were contended, so the number of updates is the honest measure of training done. Compile time comes out of the 150 s budget (a cold or contended compile can eat most of it).

## Verdict

- **Nothing is accepted.** Every row below has accepted = no. Probes are flagged for every trained variant that was scored.
- **greedy (floor)** violates 7.3e-1 (toy_support), 5.9e-1 (toy_modes), 2.1e-1 (ground), 4.0e-2 (dimers), 7.5e-1 (columns).
- **cfast** meets the 2e-4 violation budget only on toy_support (0) and toy_modes (0). Ground final viol 1.0e-3 (second run), dimers 2.0e-4 (just at the budget), columns 7.3e-2. Where violations are low the draws are degenerate (see images).
- **cpass crashed on ground** twice (see below), so there is no GPU cpass-on-ground number. cpass on toy_support: 2.3e-3, F -0.383/-0.389, only 278/399 updates.
- **ctrunc_full on ground was not run** (cut for time after cpass ground crashed and cfast ground needed a rerun).

## Failures and anomalies

1. **cpass ground crashes on the GPU.** `cpass.py:256` asserts that the replayed branch cost `Gk[:,0]` matches `g0` (rtol 1e-3, atol 0.1). It fired twice, at the first update, on worlds with cost about 1.7e4: e.g. 17582.2 vs 17643.6, 16419.8 vs 16440.1 (relative 1e-3 to 3.5e-3). Likely float32 non-associativity between the two code paths on the GPU (the Mac passed with 60 updates). Not fixed (no code edits allowed). An earlier launch was killed externally ("Terminated" after 3 min, probably another agent's pkill); this was not the assertion.
2. **cfast ground, first run (`20261014-000542`)**: seed 0 achieved **0 updates** (train took 282 s: compile and contention ate the whole budget), so its row is an untrained model with cfast's loop. Seed 1 had 1575 updates. Worst final viol 6.3e-2, F 2.05 / 4.06. I reran it (`20261014-004640`, warm cache): 2611 / 909 updates. The table below shows the rerun (the board's latest); first-run images are `cfast_ground_s0/s1.png`.
3. **cfast columns**: seed 0 had only 283 updates (compile contention), seed 1 had 2653. Seeds disagree on everything.
4. The "gen s/world" for seed 0 includes JIT compile (0.7-6 s); seed 1 is the steady value (0.03-0.11 s).
5. F is the raw free energy per fine cell (omits hard penalties), lower is better. On ground there is no exact value; F above greedy's 1.29 means a worse draw distribution than greedy even if violations are fewer.

## Tables

### toy_support (exact F* = -0.4688)

| variant | accepted | worst final viol | worst raw viol | F/cell (s0 / s1) | updates (s0 / s1) | exact stats | raw=final | seeds agree | probes flagged (s0,s1) | train s | gen s/world (s0 / s1) | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | no | 7.29e-01 | 7.29e-01 | -1.0014 / -0.9984 | - / - | FAIL | ok | ok | None,None | 0 | 0.10 / 0.08 | 20261013-235201_greedy_toy_support |
| cfast | no | 0.00e+00 | 0.00e+00 | -0.4133 / -0.4148 | 4385 / 3360 | FAIL | ok | FAIL | True,True | 150 | 0.78 / 0.05 | 20261013-235205_cfast_toy_support |
| cpass | no | 2.32e-03 | 2.44e-03 | -0.3832 / -0.3894 | 278 / 399 | FAIL | ok | ok | True,True | 150 | 0.19 / 0.07 | 20261014-003317_cpass_toy_support |

Stats (mean, final draws), per seed:

- greedy s0: dirt_frac=0.819, col_height=3.277, empty_block_frac=0.002, flat_block_frac=0.158
- greedy s1: dirt_frac=0.822, col_height=3.286, empty_block_frac=0.000, flat_block_frac=0.146
- cfast s0: dirt_frac=0.851, col_height=3.404, empty_block_frac=0.000, flat_block_frac=0.172
- cfast s1: dirt_frac=0.813, col_height=3.253, empty_block_frac=0.000, flat_block_frac=0.170
- cpass s0: dirt_frac=0.872, col_height=3.487, empty_block_frac=0.002, flat_block_frac=0.426
- cpass s1: dirt_frac=0.866, col_height=3.463, empty_block_frac=0.006, flat_block_frac=0.363

### toy_modes (exact F* = 0.2513)

| variant | accepted | worst final viol | worst raw viol | F/cell (s0 / s1) | updates (s0 / s1) | exact stats | raw=final | seeds agree | probes flagged (s0,s1) | train s | gen s/world (s0 / s1) | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | no | 5.94e-01 | 5.94e-01 | -0.1759 / -0.1877 | - / - | FAIL | ok | ok | None,None | 0 | 0.14 / 0.10 | 20261013-235758_greedy_toy_modes |
| cfast | no | 0.00e+00 | 0.00e+00 | 0.2633 / 0.2709 | 2475 / 5287 | FAIL | ok | FAIL | True,True | 150 | 1.79 / 0.03 | 20261013-235804_cfast_toy_modes |

Stats (mean, final draws), per seed:

- greedy s0: mode_L=0.014, mode_R=0.014, mode_T=0.016, mode_B=0.012, mode_other=0.945, broken_bonds=5.877
- greedy s1: mode_L=0.008, mode_R=0.023, mode_T=0.021, mode_B=0.020, mode_other=0.928, broken_bonds=5.637
- cfast s0: mode_L=0.234, mode_R=0.199, mode_T=0.211, mode_B=0.186, mode_other=0.170, broken_bonds=5.305
- cfast s1: mode_L=0.238, mode_R=0.238, mode_T=0.242, mode_B=0.219, mode_other=0.062, broken_bonds=4.547

### ground

| variant | accepted | worst final viol | worst raw viol | F/cell (s0 / s1) | updates (s0 / s1) | exact stats | raw=final | seeds agree | probes flagged (s0,s1) | train s | gen s/world (s0 / s1) | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | no | 2.06e-01 | 2.06e-01 | 1.2881 / 1.3095 | - / - | FAIL | ok | ok | None,None | 0 | 0.75 / 1.44 | 20261014-000457_greedy_ground |
| cfast | no | 1.04e-03 | 4.15e-03 | 0.6956 / 1.3481 | 2611 / 909 | FAIL | ok | FAIL | True,True | 150 | 0.77 / 0.10 | 20261014-004640_cfast_ground |

Stats (mean, final draws), per seed:

- greedy s0: dirt_frac=0.729, dirt_excess=0.222, roughness=1.420, line_match=0.512, floating=5066.875, fill_err=0.017
- greedy s1: dirt_frac=0.728, dirt_excess=0.227, roughness=1.527, line_match=0.489, floating=5094.875, fill_err=0.016
- cfast s0: dirt_frac=1.000, dirt_excess=0.493, roughness=0.000, line_match=0.500, floating=0.000, fill_err=0.097
- cfast s1: dirt_frac=0.911, dirt_excess=0.410, roughness=6.613, line_match=0.503, floating=12.875, fill_err=0.046

### dimers

| variant | accepted | worst final viol | worst raw viol | F/cell (s0 / s1) | updates (s0 / s1) | exact stats | raw=final | seeds agree | probes flagged (s0,s1) | train s | gen s/world (s0 / s1) | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | no | 4.04e-02 | 4.04e-02 | -0.7684 / -0.7765 | - / - | FAIL | ok | ok | None,None | 0 | 0.27 / 0.14 | 20261014-001855_greedy_dimers |
| cfast | no | 2.02e-04 | 2.02e-04 | -0.2258 / -0.2276 | 4626 / 5032 | FAIL | ok | ok | True,True | 150 | 0.32 / 0.03 | 20261014-001907_cfast_dimers |

Stats (mean, final draws), per seed:

- greedy s0: horizontal=0.369, vertical=0.312, diagonal=0.319, flip=0.235, flip_cells=0.693
- greedy s1: horizontal=0.375, vertical=0.308, diagonal=0.316, flip=0.237, flip_cells=0.695
- cfast s0: horizontal=0.488, vertical=0.512, diagonal=0.000, flip=0.347, flip_cells=0.928
- cfast s1: horizontal=0.503, vertical=0.496, diagonal=0.000, flip=0.346, flip_cells=0.924

### columns

| variant | accepted | worst final viol | worst raw viol | F/cell (s0 / s1) | updates (s0 / s1) | exact stats | raw=final | seeds agree | probes flagged (s0,s1) | train s | gen s/world (s0 / s1) | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | no | 7.50e-01 | 7.50e-01 | -1.9783 / -1.9767 | - / - | FAIL | ok | ok | None,None | 0 | 0.21 / 0.20 | 20261014-002505_greedy_columns |
| cfast | no | 7.29e-02 | 2.59e-02 | -0.0712 / -0.0182 | 283 / 2653 | FAIL | FAIL | FAIL | None,None | 150 | 3.07 / 0.06 | 20261014-002516_cfast_columns |

Stats (mean, final draws), per seed:

- greedy s0: density=0.890, bottom_density=0.071, gap_len=0.230, empty_xcol_frac=0.000
- greedy s1: density=0.891, bottom_density=0.072, gap_len=0.090, empty_xcol_frac=0.000
- cfast s0: density=0.062, bottom_density=0.021, gap_len=-4.195, empty_xcol_frac=0.666
- cfast s1: density=0.008, bottom_density=0.003, gap_len=-79.819, empty_xcol_frac=0.790


## Images and what the draws look like

Images in `images/` (copied from `images/bench/`). One final world per seed.

- **ground, greedy** (`greedy_ground_s0.png`): speckle. About 73 % dirt with air holes everywhere, no structure, no fill line (floating dirt 5000+ cells). Violations 21 %.
- **ground, cfast** (rerun, `cfast_ground_rerun_s0/s1.png`): both seeds collapse toward dirt. Seed 0 is **100 % dirt, completely flat** (roughness 0, floating 0): all dirt, ignores the fill line (line_match 0.50 = chance). Seed 1 (909 updates) is 91 % dirt with a ragged air region at the top: vertical air slivers and a few wide air blobs along the top rows, still not following any fill line (line_match 0.503). First run: seed 0 (untrained) shows regular vertical air stripes at the top, seed 1 all dirt (98 %).
- **columns, greedy** (`greedy_columns_s0.png`): dense, noisy columns (density 0.89 vs the target about 0.82) with scattered white gaps; 75 % violations.
- **columns, cfast** (`cfast_columns_s0/s1.png`): **almost all air**: a sparse lattice of isolated single cells (density 0.06 and 0.008; 67-79 % of x-columns empty). Same degenerate "air" mode as in the cfast report.
- **dimers, greedy / cfast** (`greedy_dimers_s0.png`, `cfast_dimers_s0.png`): greedy has all three orientations (about 0.37/0.31/0.32) with 4 % violations. cfast meets the budget (2e-4) but the diagonal orientation is **absent** (0.000), only horizontal 0.49-0.50 and vertical 0.50-0.51: a two-orientation mosaic of small patches, not the full three-orientation measure.

## Comparison with the Mac numbers (CONTEXT.md table)

| problem | Mac | GPU here | difference |
| --- | --- | --- | --- |
| greedy toy_support / ground / columns / dimers | 0.73 / 0.21 / 0.75 / 0.04 | 0.73 / 0.21 / 0.75 / 0.04 | identical (deterministic, untrained) |
| cfast toy_support | KL 0.028 (F about -0.441), viol 2e-3 / 4.9e-4 | F -0.413 / -0.415 (KL about 0.055), viol 0 | fewer violations, worse F; only 3360-4385 updates vs 8600-9000 on the Mac, because the GPU was shared and B = 8 |
| cfast ground | viol 0 / 2.9e-2, F 0.885, all dirt (648 / 549 updates) | rerun viol 0 / 1.0e-3, F 0.70 / 1.35, dirt 1.00 / 0.91 (2611 / 909 updates) | 1.4-4x more updates than the Mac but the same all-dirt collapse; F noisy across seeds |
| cfast columns | viol 2e-2, collapses to air, about 1110 updates | viol 7.3e-2, collapses to air; 283 / 2653 updates | same collapse; seed 0 starved |
| cpass toy_support | 2.4e-4, F -0.374 | 2.3e-3, F -0.383 / -0.389 | F slightly better, violations 10x higher (278-399 updates; stochastic) |
| cpass ground | 0.07, F 1.92 (about 60 updates) | **crashed** (assertion, above) | GPU differs |
| cpass / cfast dimers, toy_modes | cpass dimers 8e-4 collapse horizontal | cfast dimers 2e-4, collapses to two orientations; cfast toy_modes F 0.263/0.271 (F* 0.2513) | no Mac cfast value for these in CONTEXT.md |

Where the GPU differs: update counts are bigger on ground and toy_support (cfast) but smaller than the Mac for toy_support, seed-to-seed spread is much larger because compile time under contention changes how many updates each seed gets, and cpass on ground crashes on the GPU. The qualitative failures (all dirt, all air) are the same as on the Mac.

## Files

- Raw data: `results/bench.jsonl`, per-run logs `results/runs/<run>/train_s*.jsonl`. Driver logs: `/tmp/baselines_run*.log`.
- `images/`: greedy and cfast images for ground, columns, dimers.

# Baselines at 300 s per seed (cfast, cbig256)

2026-10-14 · agent "baselines300" · no code changed. `python -m castlegen.bench run <variant> <problem> --train 300` (2 seeds, harness 22dadfc76132, tag `b300`), one run at a time on cores 48-55, compile warmed by a `--train 20 --seeds 1` smoke run first (tag `b300smoke`). Board regenerated after each run. Images: `images/b300_<variant>_<problem>_s<k>.png`.

**The GPU was shared** (about 5 jobs, 100 % utilisation). Wall-clock budgets are contended, so quote updates, not seconds. Note that cbig256's clock starts after compile (train_s 310/300 shows it), cfast's includes compile. Caution on the update counts: 300 s runs got 2.5-7x more updates than the 150 s runs, far more than 2x. The 150 s runs were run under heavier contention and with compile inside the budget, so the updates are not comparable like with like. Other agents' runs also carry tag `b300` in `results/bench.jsonl` (e.g. ccur); the runs below are the cfast/cbig256 ones started at 01:20-02:32.

## Verdict

- **Nothing is accepted at 300 s either.** Every row: accepted = no; probes flagged on every scored row.
- **Doubling the budget did not change the qualitative picture.** Collapses are the same: ground all dirt (dirt_frac 1.00 / 0.99, line_match 0.50), columns almost all air (density 0.06 / 0.02), dimers no diagonal (0.000). No new collapses, no collapse fixed.
- **F improved a little** on toy_support (-0.413 to -0.453/-0.443 for cfast) and ground; toy_modes cfast improved by 0.01. Gap to F* is still 0.011-0.026 per cell for cfast, which is why the probe flag (a world-level test) stays on.
- **Violations**: toy_support cfast got slightly worse (0 to 1.2e-4 at level 0, but one seed has 2/512 violations at level 1: 3.9e-3, so it fails the 2e-4 budget); ground improved (1.0e-3 to 3.4e-5, now inside the budget, but the draws are all dirt); columns improved 7.3e-2 to 1.5e-2 (still fails; raw != final); dimers 2.0e-4 to 1.2e-4 (inside budget).
- **Acceptance flips**: none. The one sub-criterion that changed: cfast toy_modes "seeds agree" is now ok (was FAIL); cfast toy_support seeds agree still FAIL; cfast dimers seeds agree ok both times; cfast ground exact/seeds agree FAIL.
- **cbig256 is the best on both toys** (as before) and now has exact stats ok and seeds agree ok on both toys. It is within 0.002-0.003 of F* on toy_modes and about 0.002 of F* on toy_support (mean of 16 worlds; seed 1 even reads below F*, i.e. within evaluation noise). Still fails: probes flagged, toy_support final viol 4.9e-4 > 2e-4.

## Tables (300 s | 150 s from `reportables/2026-10-13_gpu_baselines/report.md`)

F is the raw free energy per fine cell, lower is better. Worst final viol is the worst over levels and seeds. Flags: probe flag per seed, seeds-agree.

### toy_support (F* = -0.4688)

| variant | budget | accepted | worst final viol | F/cell s0 / s1 | gap to F* s0 / s1 | updates s0 / s1 | probes flagged | seeds agree | exact stats |
|---|---|---|---|---|---|---|---|---|---|
| cfast | 300 s | no | 3.9e-03 (level 1, 2/512; level 0: 2.4e-4 / 1.2e-4) | -0.4532 / -0.4428 | 0.0156 / 0.0260 | 20283 / 32385 | yes, yes | FAIL | FAIL |
| cfast | 150 s | no | 0 | -0.4133 / -0.4148 | 0.0555 / 0.0540 | 4385 / 3360 | yes, yes | FAIL | FAIL |
| cbig256 | 300 s | no | 4.9e-04 | -0.4662 / -0.4720 | 0.0026 / -0.0032 | 3540 / 4791 | yes, yes | ok | ok |
| cbig256 | 150 s (r4, board) | no | 1.2e-04 | -0.4553 (mean) | about 0.014 | - | yes | ok | FAIL |

### toy_modes (F* = 0.2513)

| variant | budget | accepted | worst final viol | F/cell s0 / s1 | gap to F* s0 / s1 | updates s0 / s1 | probes flagged | seeds agree | exact stats |
|---|---|---|---|---|---|---|---|---|---|
| cfast | 300 s | no | 0 | 0.2630 / 0.2619 | 0.0117 / 0.0106 | 12552 / 13497 | yes, yes | ok | FAIL |
| cfast | 150 s | no | 0 | 0.2633 / 0.2709 | 0.0120 / 0.0196 | 2475 / 5287 | yes, yes | FAIL | FAIL |
| cbig256 | 300 s | no | 0 | 0.2546 / 0.2535 | 0.0033 / 0.0022 | 5521 / 8619 | yes, yes | ok | ok |
| cbig256 | 150 s (r4, board) | no | 0 | 0.2548 (mean) | about 0.0035 | - | yes | ok | FAIL |

toy_modes stats (final, cfast 300 s): mode_L/R/T/B = 0.250/0.211/0.230/0.166 (s0), 0.244/0.234/0.273/0.145 (s1), mode_other 0.143 / 0.104 (150 s: 0.170 / 0.062). cbig256 300 s: other 0.203 / 0.209, closer to the exact stats (its exact check passes).

### ground (no F*)

| variant | budget | accepted | worst final viol | F/cell s0 / s1 | updates s0 / s1 | probes | seeds agree | draws |
|---|---|---|---|---|---|---|---|---|
| cfast | 300 s | no | 3.4e-05 | 0.8134 / 1.1146 | 4397 / 4116 | yes, yes | FAIL | dirt_frac 1.000 / 0.991, roughness 0 / 1.10, line_match 0.500 / 0.500: all dirt, ignores the fill line |
| cfast | 150 s (rerun) | no | 1.0e-03 | 0.6956 / 1.3481 | 2611 / 909 | yes, yes | FAIL | dirt_frac 1.000 / 0.911, line_match 0.50 |

F on ground is noisy across seeds (spread 0.3) and is not lower at 300 s: the all-dirt mode gives a flat F around 0.7-0.8 and seed 1 is worse.

### columns (no F*)

| variant | budget | accepted | worst final viol | F/cell s0 / s1 | updates s0 / s1 | seeds agree | raw=final | draws |
|---|---|---|---|---|---|---|---|---|
| cfast | 300 s | no | 1.5e-02 | -0.0474 / -0.0247 | 11635 / 7427 | FAIL | FAIL | density 0.062 / 0.019, empty x-columns 0.49 / 0.63: almost all air |
| cfast | 150 s | no | 7.3e-02 | -0.0712 / -0.0182 | 283 / 2653 | FAIL | FAIL | density 0.062 / 0.008, empty x-columns 0.67 / 0.79 |

Greedy for scale is density 0.89, F -1.98. cfast's F of about -0.03 means a far worse distribution than greedy even though violations are lower. Probes not scored (None).

### dimers (no F*)

| variant | budget | accepted | worst final viol | F/cell s0 / s1 | updates s0 / s1 | probes | seeds agree | draws |
|---|---|---|---|---|---|---|---|---|
| cfast | 300 s | no | 1.2e-04 | -0.2266 / -0.2258 | 8504 / 3872 | yes, yes | ok | horizontal 0.504 / 0.490, vertical 0.496 / 0.510, diagonal 0.000, flip 0.346 / 0.348 |
| cfast | 150 s | no | 2.0e-04 | -0.2258 / -0.2276 | 4626 / 5032 | yes, yes | ok | horizontal 0.49-0.50, vertical 0.50-0.51, diagonal 0.000 |

Dimers is the same to three digits: the budget changes nothing. Violations pass but the diagonal orientation is absent, F is not below the (-0.77) greedy value only because F omits the penalties.

## Failures and caveats

- Seed spread: updates vary by 2x between seeds on columns and dimers (GPU contention), and cbig256 gets only 3.5k-8.6k updates in 300 s on these toys vs cfast's 12k-32k, because B = 256 vs 8; cbig256 wins on F per wall-clock regardless.
- toy_support cfast final viol: the 3.9e-3 is two violating samples out of 512 at level 1 in seed 1; level 0 is 1.2e-4. The harness counts the worst level.
- F for cbig256 toy_support seed 1 (-0.4720) is below F*; this is evaluation noise on 16 worlds, not a better-than-exact sampler.
- `bench.jsonl` also holds smoke runs (tag `b300smoke`, 20 s, 1 seed); the board's "latest run" entries for cfast were overwritten by the full runs except where noted (the board still showed the dimers smoke row until the dimers run finished; regenerate if in doubt).
- Raw logs: `/tmp/b300.log`, per-run logs `results/runs/<run>/`. Runs: cfast toy_support 20261014-012012, toy_modes 013144, ground 014436, columns 015714, dimers 020854; cbig256 toy_support 022032, toy_modes 023236 (all `_b300`).

## What to use

Compare a new variant against the 300 s rows above, not the 150 s ones: cfast improves with budget on the toys (F gap roughly halves on toy_support), but not at all on the hard problems.

# cresid: count-residual features for the learned bias (2026-10-14)

Agent: cresid. Acts on the clamp finding (`../2026-10-13_clamp/report.md`). The GPU was shared with up to four other jobs throughout.

## Answer

**The gate failed. The feature is in place and correct, but the trained bias does not use it.** After 150 s (1820 updates) the fine levels in fill-following contexts cost as much as cfast's do:

| fine cost per fine cell (nats) | own coarse | fill-following (probe-clamped) | gap |
| --- | --- | --- | --- |
| cfast (clamp), below L = 4 | 0.72 | 1.97 | +1.26 |
| **cresid, below L = 4** | **0.74** | **2.02** | **+1.28** |
| cfast (clamp), below L = 3 | 0.34 | 2.23 | +1.88 |
| **cresid, below L = 3** | **0.38** | **2.45** | **+2.07** |
| probe's own soft energy | | 0.014 / 0.013 | |

The untrained numbers are identical to clamp's (2.47 vs 2.32 and 2.07 vs 1.98), because the new features feed W1 and W3 starts at 0, so b = 0.

**The rho draws are still context-blind.** With the same noise, rho3, rho2 and rho1 are 97.3 %, 100 % and 99.9 % identical whether the coarse levels come from the model or from a probe (`same_check.txt`; cfast scored 100 %). The draws at levels 2 and 3 are deterministic (log q per cell is −0.000 and −0.002), and their means depend only on the row: rho3 is 41 / 21 / 41 / 21 / 58 … by row. Per-factor soft energy in fill contexts (L = 4 clamp) is still dominated by prior1 (t vs rho1), at 1.33 nats per cell against 1.21 for cfast (`factor_breakdown.json`).

**Likely reason (not tested).** The bias saturates early: the rho draws become deterministic, so q is one-hot. Once that happens the score-function and expectation terms carry almost no gradient for new inputs. The training contexts are also the model's own (cert near full, t forced to dirt), and there the children's rho do not change t. So the residual never pays off in the contexts that training visits. Extra information to the learned part cannot help while that part is saturated and never sees the contexts where the information matters.

## Bench, ground (2 seeds × 150 s; both runs at the same time on the shared GPU, cores 24–27 and 28–31)

| | cresid | cfast |
| --- | --- | --- |
| updates (s0 / s1) | 477 / 707 | 803 / 765 |
| worst final violations | **6.1e-5 (within budget)** | 2.4e-4 (misses the 2e-4 budget) |
| raw violations, worst | 7.6e-5 | 1.2e-2 |
| free energy per cell | 1.41 (s0 1.23, s1 1.60) | **0.86** (0.86, 0.86) |
| dirt fraction | 0.999 / 0.984 | 1.000 / 0.940 |
| line_match | 0.5 / 0.5 | 0.5 / 0.5 |
| probes flagged | yes / yes | yes / – |
| finite support mismatch | 0 | 0 |
| accepted | no | no |

Final λ: cresid [13.9, 14.8, 14.8, 14.7, 5.2] and [10.5, 13.7, 16.9, 13.1, 2.9]; cfast [10.3, 9.4, 17.0, 9.7, 2.0] and [9.7, 9.7, 17.7, 11.6, 2.4]. cresid meets the violation budget on both seeds, but at worse F with fewer updates. With 2 seeds and contended updates, I don't count that as a real difference. **None of the draws follow the fill line.** Every run is all dirt, apart from thin air columns (images below). Columns and toy_support were not run (out of time).

An earlier cresid bench chain (run `20261014-003438_cresid_ground`) was killed partway through by something outside my scripts. Its seed 0 alone gave final violations of up to 6.3e-2 and F 2.05. It is not used above.

![](images/bench_ground_images.png)
![](images/ground_total_below_L.png)
![](images/ground_per_level_bars.png)
![](images/ground_clamped_examples.png)

The clamped examples are the same block-aligned staircases that cfast produced.

## What was built

- `castlegen/variants/_cpass_core.py`: `Layout(..., resid=False)`. When on, for every declared count factor E[vb(parent), Σ_{k∈subset} va(child_k)] whose children live at level l, each site of that level gets 6 features:
  - whether the site is in the subset;
  - parent unknown;
  - (vb(parent) − Σ known siblings' va) / max sum;
  - unknown siblings in the subset / |subset|;
  - the residual per unknown sibling / (Va − 1), clipped to ±2;
  - the parent value / max sum.

  This covers every count factor at every level, including ground's rho priors and both row-honour counts, derived only from the declarations. The features read only the parent and the site's 2 × 2 block, which are inside ACTOR_WIN. The critic (cfast's "h1" critic) sees them through the actor's first layer. The option is off by default, so every other variant is unchanged.
- `cpass.py`: the layout passes `cfg["resid"]`. `cfast.py`: the layout cache key includes it, and the new variant is `_variant("cresid", resid=True)`.
- `tests/variants/test_cresid.py`, 7 tests, all passing:
  - the feature dimension is unchanged when the option is off;
  - the values match a direct computation on a lifted ground state with hidden siblings;
  - the cone contains every change with the feature on (ground, toy_modes, columns, with random W3);
  - finite support: overlapping open windows agree, raw and final, on toy_modes and ground with random W3.

  `tests/variants/test_cfast_cone.py` still passes (10 tests), and the bench's finite-support check was 0 on every run.
- Scripts here are clamp's, with the layout's `resid` switched on: `train_save.py` (takes a variant name), `eval_ground.py`, `same_check.py`, `factor_breakdown.py`, `plots.py`. The eval's check against `clamped_pass` matches to float precision.

## Next

1. Find out whether saturation is the cause: log per-level entropy over training, and see whether the rho channels go deterministic before 200 updates. If they do, try a lower lr for the actor or a slower τ anneal (engineering freedoms), and rerun this gate.
2. The fine levels never see fill contexts during training, so a feature alone cannot fix this. Training fine levels on target-like coarse contexts changes the context distribution, which makes it a proposal (constraints 3 and 4), not built.
3. Pair the residual with cconv's candidate-scoring head. A residual scored per candidate (|residual − candidate|) is something a saturated MLP bias cannot express easily.

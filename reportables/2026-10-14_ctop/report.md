# ctop: is the top level's collapse a rational response to the untrained fine levels? (2026-10-14)

Agent: ctop. Diagnostics only (no training or generation code changed). Ground, cfast defaults (B = 8, lr 3e-3, seed 0), GPU shared with other jobs. Checkpoints: init (lr = 0, tau = 2, lambda = 10), 50 updates (tau 1.90) and 200 updates (tau 1.54, lambda = 12 at levels 0-3, 8 at level 4), from one training run, evaluated at each checkpoint's own tau (and at tau = 1 and 2, `res_fixedtau`/`res_joint`). 16 worlds (seeds 4,000,000+), all 64 top-level draws per world (rho4 and cert4 at 32 sites each), so 1024 draws per checkpoint.

## Answer

**At init the objective does not clearly prefer all dirt. By 50 updates it does, strongly.** The preference is created by the first ~50 updates of fine-level training, not present in the untrained fine levels.

Joint comparison (all level-4 draws clamped together to a fill-following probe vs to all dirt, same noise, fine levels as they are; nats per fine cell, own level-4 cost included; probe with 1 row of jitter):

| checkpoint | tau | probe - dirt (levels 0-3) | probe - dirt (levels 0-4) | sign |
| --- | --- | --- | --- | --- |
| init | 2.0 | -0.063 | **-0.093** | partial fill cheaper |
| init | 1.0 | +0.038 | **+0.016** (probe0 +0.006, probe3 +0.040) | about even, dirt slightly cheaper |
| 50 updates | 1.90 | +0.332 | **+0.306** | all dirt cheaper |
| 200 updates | 1.54 | +0.924 | **+0.901** | all dirt cheaper |

(per fine cell; 8192 cells per world, so +0.3 is about 2,500 nats per world.) At tau = 2, which is where training starts, partial fill is cheaper by 0.09 per cell. At tau = 1 it is a tie to within the probe jitter (+0.006 to +0.040). The untrained fine levels are not hostile to fill contexts; the clamp report's finding that fill contexts are no costlier than the model's own at init is confirmed. After 50 updates the dirt context cost fell from 2.31 to 1.96 per cell (tau = 1) while the probe context cost did not move (2.34 to 2.38): the fine levels have learned dirt first. After 200 updates dirt context costs 1.09 against 1.97 for fill.

So the collapse is a rational response of the top level to the fine levels **as they are after a few dozen updates**, not to the untrained ones. Only the draw-by-draw picture at init is mixed (next section).

## Single-draw coupled replays (the requested comparison)

For each top-level draw (world, channel, colour, site) the draw is replayed under the world's own noise with its value overridden to (probe) the fill-following lifted value at that site or (dirt) all dirt (rho4 = 256, cert4 = (16, 16)); G is the cone return (later draws in the draw's dependency cone; replay of the actual value reproduces cfast's cone return to 2e-3). Differences are G(probe) - G(dirt) per fine cell of the world (cones at level 4 cover almost the whole world). "In A" means both values lie in the least-violating set the sampler can choose from (dirt always is; the probe's cert is in A for 40 % of cert4 draws, its rho for all). Positive: partial is more expensive downstream. SE is across the 16 worlds. `by term` is soft / lambda*viol / tau*log q; `by level` is L0 to L4.

Own cost of the draw is not in G. At b = 0 it is the same for every value (soft + tau log q = -tau log Z), and it stays within 0.0005 per cell at all checkpoints, so G decides.

Draws whose probe value differs from all dirt (674 of 1024; the rest sit under the line where the probe is full), both in A:

| checkpoint | set | n | mean G(probe) - G(dirt), /cell | fraction partial cheaper | by term (soft / lam viol / tau log q) | by level L0..L4 | actual - dirt, /cell |
| --- | --- | --- | --- | --- | --- | --- | --- |
| init | all differing | 472 | -0.0024 ± 0.0014 | 0.57 | -0.0092 / +0.0141 / -0.0073 | +.0037 +.0024 -.0001 -.0073 -.0011 | -0.0028 |
| init | rho4 | 337 | **-0.0107 ± 0.0014** | **0.72** | -0.0105 / 0 / -0.0002 | +.0005 -.0003 -.0008 -.0095 -.0007 | -0.0115 |
| init | cert4 (in A) | 135 | **+0.0181 ± 0.0019** | **0.20** | -0.0060 / +0.0490 / -0.0248 | +.0115 +.0089 +.0016 -.0017 -.0021 | +0.0189 |
| 50 | all differing | 468 | +0.0090 ± 0.0016 | 0.45 | +0.0033 / +0.0109 / -0.0051 | +.0104 +.0027 +.0034 -.0065 -.0010 | +0.0047 |
| 50 | rho4 | 337 | -0.0033 ± 0.0017 | 0.61 | -0.0033 / 0 / -0.0002 | +.0007 +.0011 +.0041 -.0085 -.0007 | -0.0045 |
| 50 | cert4 (in A) | 131 | **+0.0405 ± 0.0022** | **0.02** | +0.0202 / +0.0382 / -0.0179 | +.0352 +.0069 +.0015 -.0014 -.0017 | +0.0282 |
| 200 | all differing | 465 | +0.0208 ± 0.0007 | 0.34 | +0.0203 / +0.0011 / -0.0005 | +.0178 +.0019 +.0047 -.0030 -.0006 | +0.0040 |
| 200 | rho4 | 337 | +0.0048 ± 0.0007 | 0.47 | +0.0049 / 0 / -0.0001 | +.0011 +.0024 +.0057 -.0037 -.0007 | +0.0030 |
| 200 | cert4 (in A) | 128 | **+0.0629 ± 0.0013** | **0.00** | +0.0605 / +0.0041 / -0.0017 | +.0617 +.0007 +.0020 -.0011 -.0005 | +0.0070 |

Without restricting to A (all 674 differing draws): +0.0075, +0.0188, +0.0207 per cell at init, 50, 200, with partial cheaper in 42 %, 33 %, 27 % of draws. Over all 1024 draws: +0.0051, +0.0125, +0.0137 per cell, partial cheaper in 27 %, 22 %, 17 %. "actual - dirt" (the drawn value against dirt): +0.0014, +0.0052, +0.0019 over all draws; cheaper in 43 %, 42 %, 43 %.

Reading it:
- **rho4 prefers the fill at init** (72 % of sites, 0.011 per cell, i.e. 88 nats per draw), mostly through the level-3 children and the soft term (the count prior between rho3 and rho4). The preference weakens by 50 updates (0.003) and flips by 200 (+0.005).
- **cert4 prefers dirt at every checkpoint, including init**, and the fill cert is cheaper in only 20 % (init), 2 % (50), 0 % (200) of draws. A single partial certificate costs 0.018 per cell at init (about 150 nats per draw), 0.041 at 50 and 0.063 at 200. cert4 is where the dirt fraction comes from: the actual draw is already all-dirt in 39 % of cert4 draws at init, 38 % at 50, 67 % at 200 (rho4 is almost never dirt: 5 %, 1 %, 0 %).
- **Which term (task 2).** At init the cost of a partial certificate is lambda*violations (+0.049 per cell, lambda = 10: the untrained fine levels, with b = 0, violate honour and support constraints around an unusual certificate), partly offset by tau*log q (-0.025, a partial context keeps the fine levels' q more spread) and a small soft term (-0.006). The cost sits at levels 0 and 1 (+0.012, +0.009). By 50 updates the violation term stays (+0.038) and soft energy turns positive (+0.020). By 200 updates it is almost entirely **soft energy at level 0** (+0.0605 of +0.0629; violations fall to +0.004; the soft terms are not broken down further here, but the clamp report attributes the level-0 gap to the t/rho1 count prior). So the term changes with training: violations first, then soft energy at the finest level, consistent with the clamp report's "count prior ignores context" mechanism.
- Joint clamp, by term, levels 0-3 (probe1 - dirt, native tau): init soft -0.123, lambda*viol +0.195, tau*log q -0.135; 50 updates +0.276, +0.157, -0.101; 200 updates +0.898, +0.047, -0.022. So at init violations alone make partial fill dearer (+0.2), but the soft energy and entropy terms favour it; by 200 updates soft energy at level 0 (+0.69 to +0.77 of the per-level total) dominates.
- By level in the joint clamp (probe1 - dirt, tau = 1): init L0 +0.079, L1 +0.043, L2 +0.007, L3 -0.091; 200 updates L0 +0.690, L1 +0.062, L2 +0.153, L3 -0.030. Level 3 always favours the probe (-0.09 to -0.03; I did not investigate why), but is outweighed by level 0.

## Statement for the question (task 3)

At init (untrained fine levels, b = 0), taken one top-level draw at a time, the objective **already prefers all dirt for the certificate (cert4)**, by 0.018 per fine cell (about 150 nats per draw; 80 % of sites), but prefers the fill for rho4 (0.011 per cell, 72 % of sites). The two cancel when the whole top level is changed together: partial fill is **cheaper by 0.09 per cell at tau = 2** and **dearer by 0.016 per cell at tau = 1** (a tie within probe jitter, range +0.006 to +0.040). So the untrained fine levels do not by themselves make all dirt rational. The preference for dirt appears within 50 updates (+0.31 per cell, about 2,500 nats per world, at tau 1.9) and reaches +0.90 by 200 updates, when cert4 goes deterministic. The top level then collapses to the objective-minimising choice given fine levels that have just trained on dirt contexts.

Implications:
- A bottom-up schedule (ccur) or any fix of the fine levels' behaviour in fill contexts addresses a real, trained-in cost (fill context 1.97 vs dirt 1.09 per cell after 200 updates, the clamp-report gap), but it has to act early: the cost gap is built in the first 50 updates, during the tau 2 phase, while the top-level draws are still mixed. Per-draw, the cert4 preference for dirt exists from the start (violations at init), so the top level never gets a good signal for partial certificates, and the first 50 updates add soft-energy cost on top.
- Variance and bias saturation are not needed to explain the start of the collapse: the gradient sign at cert4 points to dirt already at init in 80 % of sites with exact coupled returns. Why cert4 rather than rho4 moves first in training is not tested here.

## Figures

![](images/ctop_cost_diff.png)

`images/ctop_cost_diff.png`: partial minus all-dirt downstream cost per fine cell, split by term (bars) with the total (dot), at the three checkpoints. Panels: rho4 draws, cert4 draws (both in A), and all top-level draws clamped together (bars are levels 0-3 by term, the dot includes level 4).

## Method

- `train_snap.py`: cfast on ground, 150 s tau schedule (so tau is 1.90 at update 50, 1.54 at update 200); snapshots params, lambda and tau at updates 0, 50, 200 (`snaps.pkl`). The compile was warmed first.
- `ctop_eval.py`: per checkpoint, base cfast pass (`train_pass`) for the actual draws, A, soft, nv, log q; then coupled replays for every top-level draw and each candidate value (actual, probe jitter 1, all dirt) with the override of `_cfast_core.train_pass` copied into a per-term version (soft, lambda*nv, tau*log q of the drawn values, no change in draws). Cone returns by integral images (`_cfast_cone.Cone`), split by 3 terms x 5 levels with traced level masks. Sanity: `sanity.py` (replay of the actual value reproduces the cone return to 0.002 on values around 10^4). Outputs `res_native.pkl`, `res_fixedtau.pkl` (tau = 1 and 2 for all checkpoints), summaries by `analyse.py`.
- `ctop_joint.py`, `clamp_tools.py` (copy of csat's `clamped_pass_stats`), `joint_an.py`: all level-4 draws clamped to probe lifts (jitter 0, 1, 3) or to `ground.lift` of an all-dirt tile state, or left free ("own"); cost of the levels below by term and level; 16 worlds, same noise.
- `make_report.py`: figure and joint numbers. `res_*_summary.json` hold all tables (including tau = 1 and 2 versions of the single-draw tables).

## Caveats

- One training seed, 16 worlds, B = 8 and a 150 s tau schedule; the 50 and 200 update checkpoints are at tau 1.90 and 1.54. At fixed tau the pattern is the same (tau = 1 and 2 summaries): the single-draw partial - dirt over all differing draws at tau = 2 is +0.0051 (init), +0.0124 (50); the joint numbers are in the table above.
- Single-draw overrides change one block while the others keep their sampled values; the joint clamp changes all. The two answers differ at init (single-draw cert4 +0.018, joint -0.093 at tau 2), which says the cert4 and rho4 effects and block interactions matter; the joint version is the better proxy for what the top level optimises, the single-draw version for the gradient it receives.
- The probe's cert is outside A for 60 % of cert4 draws (the sampler could not pick it); those are excluded from the "in A" rows but included in the "all differing" rows.
- "All dirt" is a lifted all-ones tile state (every top block full); the sampler's own dirt (cert4 = (16, 16)) in a draw with a partial rho4 is a mix, as in the single-draw test.
- Not tested: whether the signs change with a longer-trained but non-collapsed top level, or with lambda below 10 at init.

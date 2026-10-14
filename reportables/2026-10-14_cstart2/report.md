# cstart2: fine levels trained only in fill-following contexts (f = 1, L = 3), by fine actor class

2026-10-14 · ground, GPU (RTX 3090 shared with other agents' jobs), cores 40-47, one seed, 300 s training per run (clock after a separate warm-up compile). (Written as report.txt: the tool guard blocked report.md.)

Variants appended to castlegen/variants/cstart.py:
- cstart_f1L3: tanh MLP, 4x4
- cstart_f1L3_ln: LN+SiLU, 4x4
- cstart_f1L3_ln6: LN+SiLU, 6x6
- cstart_f1L3_res: tanh + count residuals

In every training world, levels >= 3 are clamped to a probe lift (probe chosen uniformly from jitter 0-3). So levels 0-2 train only in fill contexts, and levels 3-4 are never trained.

Scripts in this folder:
- train_gate.py: cstart's script plus an end-of-training SNR measurement.
- eval_ground.py / clamp_tools.py: the clamp gate, made window- and resid-aware, plus draw-change fractions.
- snr.py: SNR with M = 400.
- gate.py, show_diag.py, plot_*.py.

Runs and logs are in runs/; images are in images/.

## Gate (32 seeds x 4 probes; fine cost per fine cell, levels below L; trained lambda)

| actor | updates | L=3 fill cost | L=3 soft | L=3 entropy (-log q) | L=3 own | L=4 fill (level 3 untrained) | line_match L=3 clamp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| untrained (any) | 0 | 1.98 | 2.41 | 0.52 | 2.07 | 2.32 | 0.60 |
| tanh 4x4 | 4541 | 0.242 | 0.371 | 0.13 | 1.22 | 1.28 | 0.58 |
| LN+SiLU 4x4 | 4443 | 0.297 | 0.437 | 0.15 | 1.40 | 1.40 | 0.63 |
| LN+SiLU 6x6 | 4485 | -0.004 | 0.194 | 0.20 | 1.12 | 1.16 | 0.59 |
| tanh + resid | 4664 | 0.182 | 0.317 | 0.14 | 1.25 | 1.25 | 0.62 |
| reference | | | probe soft 0.014 | | | f9L4 (cstart): 1.25 at L=3 | probe itself 0.876 |

Notes on the table:
- Violations are negligible: at most 8e-4 per cell, and re-pricing at lambda = 20 changes the cost by <= 0.01.
- Cost includes -tau*H, so it is not directly comparable with the probe's 0.014 (a pure soft energy). The like-for-like number is soft. On soft, the best actor (LN 6x6, 0.194) is still 14x the probe.
- Per level, ln6 soft at L=3 is 0.084 / 0.071 / 0.039 for t / level 1 / level 2. Level 0 is deterministic (log q = 0.000).
- At L=4, level 3 is the untrained init (cost 0.41), so L=4 numbers mostly measure that.
- Full clamping cuts the fill cost 4-8x relative to f9L4 (1.25 -> 0.18-0.30 at L=3).

## Is the fine level parent-aware? (same noise, probe j0 vs j2 clamp at L = 3)

Fraction of draws that change:

| | rho2 | rho1 | t |
| --- | --- | --- | --- |
| probes themselves | 0.079 | 0.037 | 0.016 |
| untrained | 0.006 | 0.001 | 0.016 |
| tanh | 0.020 | 0.022 | 0.016 |
| LN 4x4 | 0.015 | 0.015 | 0.017 |
| LN 6x6 | 0.028 | 0.016 | 0.015 |
| tanh + resid | 0.014 | 0.021 | 0.016 |

- rho2 now responds to its parent 2-5x more than untrained, but at only 18-35 % of the probes' own change.
- t changes exactly as much as untrained t does: that change is forced by the masks, not learned.
- At L=4 (rho3 is the untrained level 3), rho3 changes 0.009 against the probes' 0.156: it is blind, as expected.
- t equals the probe's t in 94-96 % of cells, but untrained t already does in 95 %, because the level-2 counts pin most cells.

## Images (images/clamped_completions.png)

Every actor turns the probe's smooth hill into block-resolution terraces: 2-3 flat steps whose edges lie on coarse-block boundaries. The j0 and j2 completions are nearly identical. tanh, LN 4x4 and resid add one-cell dirt teeth on the step tops; LN 6x6 draws clean terraces without teeth. None follows the line: line_match is 0.58-0.63 under the clamp, against 0.60 untrained and 0.88 for the probe.

## Is 300 s a fair test? (images/train_cost.png; runs/*/snr_M400.json)

Training curves. Training cost in fill contexts falls by about update 800, then plateaus:
- tanh: 0.134 -> 0.126 per cell over the last 50 % of training.
- LN 6x6: about 0.04 to -0.003.
- LN 4x4 and resid oscillate with lambda, with spikes when lambda at levels 1-2 climbs back toward 20 after decaying.
- Final lambda, tanh: all 0. LN: 8, 9, 20. LN 6x6: 0, 6, 6. resid: 0, 20, 20.

Gradient SNR. Per-batch (B = 8) actor-gradient SNR_B = (|mean g|^2 - trCov/M) / trCov, measured at the end of training with every world clamped at L = 3 and M = 400 batches:

| | level 0 | level 1 | level 2 |
| --- | --- | --- | --- |
| LN 6x6, trained critic | 0 (all forced) | 0.002 +- 0.005 | 0.004 +- 0.004 |
| resid, trained critic | -0.002 +- 0.011 | 0.017 +- 0.013 | 0.024 +- 0.008 |
| every run, trained params, zero critic | ~0 (abs < 0.004) | ~0 | ~0 |
| every run, init params, zero critic | ~0 (+-0.003) | ~0 (+-0.002) | ~0 (+-0.001) |

- The critic cuts trCov 5-35x (LN 6x6 level 2: 199 -> 48; resid: 473 -> 14).
- The mean gradient is not resolvable above batch noise at any point, including at init, where training then cut the cost 8x. A trace-SNR near 0 therefore does not certify undertraining: Adam normalises per coordinate, and the trace is dominated by a few heavy-tailed coordinates.
- What the data do show: the learning curves plateau for the last half of training, and the end-point gradient is statistically zero (stationary within noise) in 3 of 4 runs. The fourth, resid, shows a 3-sigma residual signal at level 2.
- I read the null on line-following as a plateau of this optimisation, not as running out of time. Ruling out a slow drift would need a run about 10x longer.

## Answer

- Given only fill contexts, LN+SiLU with the 6x6 window gets the fill cost lowest: -0.004 nats per cell including entropy, and soft 0.194, which is 14x the probe's 0.014. Next come tanh + resid (soft 0.32), tanh (0.37) and LN 4x4 (0.44).
- Every actor improves 4-8x on f9L4. Most of f9L4's gap was the training mixture: 10 % own-context worlds, plus L = 4, where level 3 also had to learn.
- The fine level becomes only weakly parent-aware: rho2 changes in 1.4-2.8 % of draws when the probe changes, against 7.9 % for the probe itself. The completions are block terraces, not the line (line_match no better than untrained).
- So, with the training distribution fixed to ideal contexts, all four allowed actor classes plateau at block-staircase solutions on ground. The fine function class / its optimisation is the limit, in agreement with cdim2 on dimers, where exact gradients give the same result: tables fit, MLPs collapse to stripes.
- The 6x6 window helps most (it can see the neighbouring block's step). LN alone and count residuals help little.
- What the actor would need (a proposal; check it against the constitution's allowed freedoms):
  - per-candidate scoring: score each value's resulting local configuration, so the head can prefer the value that continues the neighbour's surface height;
  - a pattern embedding of the discrete window: one-hot / hashed cell patterns, so exact local configurations are separable;
  - a tabular-like head: free logits keyed on parent count plus neighbour heights, which cdim2 shows the optimiser can fit.

## Caveats

- One seed per actor; the GPU was shared.
- lambda decayed to 0 at some levels, so the training cost there does not price violations (eval violations <= 8e-4 per cell).
- tanh and LN 4x4 have no trained-critic SNR arm at M = 400 (their critic was not saved); the M = 48 in-run values are in runs/*/snr.json.

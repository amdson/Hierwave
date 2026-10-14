# cstart (ideas #14): fine levels trained on probe-lifted coarse contexts at a mixed start depth, no pass

2026-10-14 · `castlegen/variants/cstart.py`, `_cstart_core.py`, `tests/variants/test_cstart.py` (11 pass; clamp fraction 0 is bit-identical to cfast; a clamped world reproduces `clamped_pass`; parameters at levels ≥ L stay unchanged). Written by the orchestrator from the agent's final message (its report write was blocked); the agent's `images/` and `runs/` are in this folder.

**Design.** Each training world is, with probability f, clamped at levels ≥ L (L uniform in {1..4}) to the lift of a probe state; the rest run at full depth. Clamped worlds train only the draws below L. Default f = 0.5.

## Clamp gate (ground; fine cost per fine cell, fill-following vs own context; shared GPU)

| run | updates | L = 4 fill / own | L = 3 fill / own |
| --- | --- | --- | --- |
| cstart, 300 s | 3023 | 2.04 / 1.19 | 2.05 / 0.65 |
| cfast, 300 s (cfine's run) | – | 1.99 / 1.12 | 2.13 / 0.66 |
| cstart, 120 s | 994 | 2.24 / 2.31 | 2.07 / 1.81 |
| cstart_off (= cfast), 120 s | 1436 | 1.99 / 0.77 | 2.43 / 0.48 |
| cstart_f9L4, 300 s (90 % clamped, always L = 4) | 5007 | 1.16 / 0.90 | 1.25 / 0.71 |

Probe reference: 0.014.
- At 120 s the gap closes only because cstart's own-context cost stays high.
- f9L4 halves the fill cost, but the clamped fine levels draw block staircases with regular dirt teeth: line_match under the clamp 0.55 / 0.62, the same as untrained (0.56 / 0.61). Its own draws are still 94 % dirt.
- Draws changed by the probe clamp (L = 4 / L = 3): 0.32 / 0.35 overall for cstart at 300 s, but **rho3 changes in only 1.2 %**: the fine count draws are still blind to their parent.
- Collapse delayed, not prevented: dirt 0.74 at update 200 (cfast 0.97), passes 0.95 around update 450 (cfast about 150). λ at the cap of 20 at levels 1 and 3.

## Bench (v2, `--train 300`, 2 seeds; 2907 / 5421 updates)
Not accepted. Worst final violations 7.8e-3 (seed 0), F 1.63, line_match 0.50, dirt 0.95-0.99. Starvation 1.0 at levels 2-4, 0.53 / 1.0 at level 1. (Run before the gate result was in; by the fast-loop order it should not have run.) Dimers: identical to cfast by construction (T = 0).

## Verdict
Visiting fill-following contexts is necessary but not sufficient. Shown the contexts constantly (f9L4), the fine MLP still does not learn to read the parent count and builds staircases. Together with cdim2 (on dimers, exact gradients: free logits over the same features reach the target, every MLP collapses to stripes), the remaining suspect is the fine actor's function class / optimisation, not the training distribution. Next: an early heavy-clamp phase decaying to 50/50, combined with a fine actor that can actually use the parent features (candidate scoring, pattern embedding; see cdim3).

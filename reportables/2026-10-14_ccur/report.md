# ccur (ideas #7): bottom-up level schedule does not change the outcome

2026-10-14 · `castlegen/variants/ccur.py`. Written by the orchestrator from the agent's final message (the agent's own report write was blocked); scripts, `runs/` and `images/curves.png` are the agent's.

**Design.** cfast with levels 3-4 frozen (parameters restored after each update; Adam moments keep accumulating) while the fine levels train; staged release fine to coarse (L3 at 35 % of the budget, L4 at 55 %). Critic trains at every level throughout; τ anneals as in cfast.

## Gate (ground, 300 s; fine cost per fine cell, own coarse context / fill-following coarse context)

| model | L = 4 own / fill | L = 3 own / fill |
| --- | --- | --- |
| untrained | 2.47 / 2.32 | 2.07 / 1.98 |
| ccur at release (update 1056) | 1.91 / **2.80** | 1.49 / **2.64** |
| ccur, end (3014 updates) | 0.81 / 2.07 | 0.49 / 2.13 |
| ccur_off (= cfast + diagnostics, 3366 updates) | 0.84 / 1.86 | 0.54 / 2.11 |

- While the coarse levels were frozen, the fine levels got **worse** in fill contexts. The frozen coarse draws are diverse (about 3 nats) but do not follow the fill (line_match 0.50, 88 % dirt), so the fine levels had nothing fill-like to learn from.
- After release the top level still collapses, and faster: L4 released at update 1635, dirt passed 0.95 at 1675, cert4 deterministic by 1725 (cfast: 175-250 updates from init). line_match was 0.500 at every update in both runs.

## Bench (`--train 300`, 2 seeds, back to back, shared GPU)

| | worst final viol | F | dirt | line_match | updates | accepted |
| --- | --- | --- | --- | --- | --- | --- |
| ground ccur | 0 | 1.23 | 1.00 / 0.875 (seed 1 ends in stripes) | 0.50 | 2907 / 4007 | no (probes flagged, seeds disagree) |
| ground cfast | 0 | 0.60 | 1.00 | 0.50 | 3891 / 4702 | no (probes flagged) |
| columns ccur | 3.2e-2 | −0.006 | – | – | 8341 / 4225 | no |
| columns cfast | 9.2e-3 | −0.022 | – | – | 5598 / 4346 | no |

On columns both collapse to nearly empty (density 0.004 / 0.016 vs 0.003 / 0.019). Dimers not run (T = 0: nothing to freeze, ccur = cfast).

## Caveats
- The harness changed mid-session (bench v2), so the paired runs carry different harness hashes; the first ccur columns attempt crashed in `shares.py`, the rerun worked.

## Verdict
Training order is not the lever. The fine levels need fill-following coarse contexts during training, which neither the model's own draws nor a frozen random prior provide. See ideas #14 (`cstart`).

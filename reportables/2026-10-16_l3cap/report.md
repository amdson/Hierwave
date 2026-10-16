# l3cap: is the high-slack L3 deficit a capacity or window limit of the L3 head?

**Answer: no.** Neither doubling the L3 hidden width, widening the actor window from 6x6 to 8x8, nor both reduces the high-slack surface-crossing deficit; the largest change is +0.65 nats (worse), the seed-to-seed spread of the control is 0.05-0.2 nats on probes and the arm-vs-control differences are within +-0.7. No arm cuts it by 1 nat. Raw violations are not systematically changed.

GPU 0 was shared with other agents' jobs (SM clock 1725-1905 MHz, 76-77 C, no throttle after runs); all arms got the same 1904 updates (119 iterations x 16), taken from the control's 300 s run; the wider arms needed 316-406 s of wall for the same updates.

## What was run
`castlegen/variants/l3cap.py` (variants `l3cap_ctl|w2|c2|wc2`), driver `scripts/driver.sh` (-> `run_arm.sh`, `train_arm.py`, `summarize.py`). Each arm starts from `2026-10-16_longfit2/runs/big300/params_s<seed>.pkl` and runs clongfit's recipe (ML only, upi 16, nctx 64, ml_B 32) for exactly 119 iterations (= the control's 300 s count), fresh Adam state in every arm. Then q2 (raw pass draws, 128 worlds), relax4's `relaxcheck.py`, `classes.py` and `decomp.py` (relaxed probes) on the continued params.
- (a) ctl: unchanged. (b) w2: **per-level**: only the level-3 nets (all channels) have hidden width 64 -> 128 (`core.mlp` is shape-generic, so no change to the pass). (c) c2: window 6x6 (-2..3) -> 8x8 (-3..4). The window is one Layout-wide setting (features, cone, margin), so it cannot be per-level without touching the generic pass structure; **c2 changes all levels**. (d) wc2: both.
- Interpretation note: in this recipe `nctx` is the size of the persistent own-draw population, not an actor context width, so I read "context nctx x2" as the actor window. The population size was not varied.
- Widening is function-preserving (hidden units duplicated, outgoing weights split at random so duplicates separate; new window features have zero input weights); checked: all four arms generate identical draws at step 0 (`scripts/check.py`).

## Table (full: `runs/table.md`; starved = deficit > 5 nats; relaxed probes; hi_cross = slack > 1.5 and dirt in (0.05, 0.95), 25.6 % of L3 blocks)
| arm | seed | upd | wall s | raw line_match | raw viol L0-L4 (128 worlds) | hi_cross starved % probe/own | hi_cross med deficit probe/own | slack0 starved %/med def (probe) | rho(q,D) hi_cross probe/own | rho(q,D) slack0 | L3 starved % probe/own | L3 rho | L4 starved % probe/own | L4 rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctl | 0 | 1904 | 300 | 0.664 | 187,87,15,0,0 | 25.7/25.6 | 2.91/3.10 | 0.1/-0.59 | .35/.32 | .48 | 6.8/7.3 | .48 | 9.2/1.7 | .34 |
| w2 | 0 | 1904 | 379 | 0.665 | 186,91,17,1,0 | 26.6/31.0 | 3.28/3.39 | 0.0/-0.60 | .36/.30 | .42 | 6.9/8.3 | .44 | 8.7/1.7 | .35 |
| c2 | 0 | 1904 | 406 | 0.668 | 162,113,33,0,0 | 25.0/26.8 | 2.92/3.10 | 0.1/-0.56 | .32/.32 | .52 | 6.4/7.2 | .50 | 6.5/1.5 | .33 |
| wc2 | 0 | 1904 | 384 | 0.667 | 141,80,20,1,0 | 29.8/26.8 | 3.56/3.39 | 0.1/-0.53 | .35/.29 | .47 | 8.0/7.5 | .46 | 7.3/1.4 | .37 |
| ctl | 1 | 1904 | 301 | 0.658 | 187,93,28,1,0 | 23.1/22.8 | 2.95/2.97 | 0.0/-0.66 | .34/.30 | .49 | 6.0/6.4 | .49 | 10.1/1.7 | .33 |
| c2 | 1 | 1904 | 316 | 0.661 | 173,129,38,1,0 | 24.5/30.3 | 3.25/3.44 | 0.0/-0.61 | .27/.26 | .45 | 6.4/8.4 | .47 | 9.5/2.0 | .32 |

Seed 1 was run for ctl and c2 (c2 = arm with the lowest seed-0 probe deficit, 2.92, chosen automatically; it is a tie with ctl, not a winner).

## Reading
- High-slack deficit (probe / own): ctl 2.91 / 3.10 (s0), 2.95 / 2.97 (s1); w2 3.28 / 3.39; c2 2.92 / 3.10 (s0), 3.25 / 3.44 (s1); wc2 3.56 / 3.39. Starved share of hi_cross 23-30 % for every arm. Slack-0 blocks have no deficit in any arm. rho(q, D) on hi_cross stays 0.27-0.36, no arm moves it up.
- The larger heads do not fit the blocks better with the same 1904 updates. Caveat: this is 300 s-equivalent of continuation from a model trained 300 s; a bigger head could need more updates than the control to use its capacity, which was not tested. The result says the deficit is not removed cheaply by capacity or window size, not that no amount would.
- Raw violations: wc2 has fewer at L0-L2 (141/80/20 vs 187/87/15), c2 s0 fewer at L0 but more at L1-L2 (162/113/33), c2 s1 more at L1-L2 (173/129/38 vs 187/93/28); the control's own run-to-run spread is 187/87/15 vs 187/93/28 (two seeds), so with one run per arm these are not distinguishable from noise except possibly wc2's L0, which is a single run. Raw line_match 0.658-0.668 everywhere.
- Nothing regressed at L4 beyond noise: relaxed-probe starved 6.5-10.1 % (ctl 9.2 / 10.1), rho 0.32-0.37; c2 s0 and wc2 are lower (6.5, 7.3 %), c2 s1 is 9.5 %.
- Single run per arm and seed, fresh Adam in every arm.

## What failed / next
- Nothing crashed. Wider models are 1.05-1.35x slower per update (wall 316-406 s vs 300 s for the same updates; the GPU was shared so ratios are rough).
- Next (one line each): the deficit tracks certificate slack not capacity, so the lever is the L3 certificate's training signal (e.g. a certificate-aware target or more ML weight on high-slack crossing blocks), not a bigger L3 net; if capacity is still wanted, run w2 for 3-4x the updates before concluding.

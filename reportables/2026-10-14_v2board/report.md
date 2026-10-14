# v2board: bench v2 block check across the leading variants at 300 s, and a gate proposal

2026-10-14. Bench version 2 (harness hash ee11dda3). 300 s per seed, 2 seeds, cores 56-63. **The GPU was shared** with 4-5 other jobs, so update counts are contended and not comparable across rows run at different times. Nothing in the harness or variants was changed. Tables below come from `results/bench.jsonl` via `tables.py`; raw dump in `collect_out.txt`; gate evaluation in `gate.py` / `gate.md`.

## Rows (all v2, 300 s, 2 seeds)
Mine (tag `v2board`): cbig256 toy_modes, cbig256 toy_support, creplay ground, cfast ground, cfast toy_modes. Already v2 at 300 s from other agents, not duplicated: cfast and chot toy_support (tag chot_gate3), cstart ground (tag cstart), cfast dimers (b300), columns rows (ccur, cfast). The baselines300 agent also produced a second cbig256 toy_modes and toy_support run (tag b300); it agrees with mine within noise (listed in `gate.md`). The board (`results/board.md`) was regenerated.

Reading the tables: "w" is the weight used: `dir` = direct target share, `sub` = subtree share (used on toy levels whose direct weight is constant, where direct rho is n/a). rho columns: pooled / probes alone / own alone. "starved % probe" = probe blocks more than 5 nats below the own-draw line; "own null %" = same fraction on own blocks. Worst final violations are the max over levels of the final fraction (budget 2e-4). Old probe flag: FLAG = v1 flag fired. Gap = F minus F* (toy only; ground has no F*).

## Tables
Per-seed tables follow (generated). Key reading per problem:

- toy_modes (F* 0.2513): cbig256 gap +0.0022 / +0.0023, cfast +0.0108 / +0.0139. Violations 0. Old flag fires on all four. New check: rho L0 0.86-0.87 (exact-p ideal 0.87) on every row, probes 0.87-0.88, level-1 subtree 0.85-0.89 (ideal 0.97, cbig256 at 60 s earlier 0.87), starvation 0-0.3 % (null 0-1.7 %). Direct starvation at level 1 is 6-7 % on cfast and 2 % on cbig256, because direct weight is constant there; use the subtree weight (0-0.3 %).
- toy_support (F* -0.4688): cbig256 gap 0.0000 / +0.0012, cfast +0.018 / +0.021, chot +0.024 / +0.019. Worst final violation 1.2e-4 to 4.9e-4 (cbig256 s1 4.9e-4 and cfast 3.7e-4 exceed 2e-4; chot 0). rho L0 0.71-0.72 on all (no exact ideal computed), subtree L1 0.74-0.85, L2 0.49-0.68. Starvation: cbig256 0-1.5 %, cfast 1.0-7.5 %, chot 8.2-8.6 % (subtree weight). So here the check ranks the variants the same way as F: the two with a larger F gap are the ones with 5-9 % of probe blocks starved.
- ground (no F*): cfast F 0.76 / 0.79, creplay 1.14 / 1.33, cstart 1.48 / 1.79 (cstart has final violations up to 7.8e-3). All collapsed (images `images/cfast_ground_s0.png`, `images/creplay_ground_s0.png`: all dirt plus a few vertical stripes, no fill line). rho probes L0 0.77-0.78 on all (including the collapsed ones). Levels 2-4: 100 % of probe blocks starved for every variant and seed except creplay s0 level 3 (31 %, median deficit 0.1 nats); level 1 is 53 % (cfast s0, cstart s0) or 100 %. Median deficit 22-180 nats. rho on probes alone at levels 1-4 is 0.01-0.78 (mostly below 0.6); pooled rho is negative (Simpson, own and probe clusters on opposite sides), so pooled rho is uninformative on ground.
- dimers (cfast, 300 s): F -0.2266 / -0.2258, violations 1.2e-4. rho n/a (probes' weight constant), but 43-44 % of probe blocks are starved at level 0 with a median deficit of only 0.2 nats (a heavy tail). Own null 0-0.1 %. I cannot say whether this is a real failure (probes may be states the target gives mass and the sampler does not, e.g. one orientation) or an artefact of probe design on a hard problem with no slack; no image or exact reference to check. Treat dimers as unresolved.


#### toy_modes

| variant (seed) | F (gap) | worst final viol | old probe flag | updates | level | w | rho pool | rho probes | rho own | starved % probe | own null % | median deficit (nats) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cbig256 s0 | 0.2535 (+0.0022) | 0.0e+00 | FLAG | 5467 | 0 | dir | 0.87 | 0.87 | 0.87 | 0.0 | 0.0 | 0.0 |
|  | | | | | 1 | sub | 0.88 | 0.89 | 0.88 | 0.0 | 0.0 | 0.1 |
|  | | | | | 2 | sub | n/a | n/a | n/a | 0.0 | 0.0 | 0.0 |
| cbig256 s1 | 0.2536 (+0.0023) | 0.0e+00 | FLAG | 3208 | 0 | dir | 0.87 | 0.88 | 0.87 | 0.0 | 0.0 | 0.0 |
|  | | | | | 1 | sub | 0.89 | 0.89 | 0.88 | 0.0 | 0.0 | 0.0 |
|  | | | | | 2 | sub | n/a | n/a | n/a | 0.0 | 0.0 | 0.0 |
| cfast s0 | 0.2621 (+0.0108) | 0.0e+00 | FLAG | 42201 | 0 | dir | 0.86 | 0.87 | 0.84 | 0.0 | 0.0 | 0.0 |
|  | | | | | 1 | sub | 0.85 | 0.89 | 0.79 | 0.0 | 0.0 | 0.1 |
|  | | | | | 2 | sub | n/a | n/a | n/a | 0.0 | 0.0 | 0.0 |
| cfast s1 | 0.2652 (+0.0139) | 0.0e+00 | FLAG | 31155 | 0 | dir | 0.86 | 0.87 | 0.84 | 0.0 | 0.0 | 0.0 |
|  | | | | | 1 | sub | 0.86 | 0.88 | 0.84 | 0.3 | 0.0 | 0.1 |
|  | | | | | 2 | sub | n/a | n/a | n/a | 0.0 | 0.0 | 0.0 |

#### toy_support

| variant (seed) | F (gap) | worst final viol | old probe flag | updates | level | w | rho pool | rho probes | rho own | starved % probe | own null % | median deficit (nats) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cbig256 s0 | -0.4688 (+0.0000) | 1.2e-04 | FLAG | 4178 | 0 | dir | 0.72 | 0.71 | 0.73 | 0.0 | 0.0 | 0.0 |
|  | | | | | 1 | sub | 0.81 | 0.81 | 0.81 | 0.0 | 0.0 | 0.1 |
|  | | | | | 2 | sub | 0.49 | 0.52 | 0.47 | 0.0 | 0.1 | -0.0 |
| cbig256 s1 | -0.4676 (+0.0012) | 4.9e-04 | FLAG | 4878 | 0 | dir | 0.72 | 0.71 | 0.74 | 0.0 | 0.0 | 0.0 |
|  | | | | | 1 | sub | 0.83 | 0.81 | 0.85 | 1.5 | 0.0 | 0.0 |
|  | | | | | 2 | sub | 0.54 | 0.56 | 0.51 | 0.1 | 0.2 | 0.1 |
| cfast s0 | -0.4509 (+0.0179) | 3.7e-04 | FLAG | 17292 | 0 | dir | 0.71 | 0.72 | 0.71 | 3.6 | 0.0 | 0.0 |
|  | | | | | 1 | sub | 0.76 | 0.74 | 0.77 | 1.0 | 0.0 | 0.4 |
|  | | | | | 2 | sub | 0.60 | 0.61 | 0.56 | 5.4 | 0.0 | 0.2 |
| cfast s1 | -0.4475 (+0.0213) | 3.7e-04 | FLAG | 12678 | 0 | dir | 0.74 | 0.71 | 0.76 | 3.6 | 0.0 | 0.0 |
|  | | | | | 1 | sub | 0.79 | 0.79 | 0.75 | 7.5 | 0.0 | 0.1 |
|  | | | | | 2 | sub | 0.67 | 0.68 | 0.57 | 2.6 | 0.2 | 0.3 |

#### ground

| variant (seed) | F (gap) | worst final viol | old probe flag | updates | level | w | rho pool | rho probes | rho own | starved % probe | own null % | median deficit (nats) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cfast s0 | 0.7585 | 1.9e-05 | FLAG | 9647 | 0 | dir | 0.32 | 0.78 | 0.33 | 2.6 | 0.0 | 0.0 |
|  | | | | | 1 | dir | -0.02 | 0.24 | -0.26 | 53.0 | 1.2 | 42.8 |
|  | | | | | 2 | dir | -0.74 | 0.51 | 0.13 | 100.0 | 0.0 | 154.6 |
|  | | | | | 3 | dir | -0.64 | 0.65 | 0.51 | 100.0 | 0.0 | 154.9 |
|  | | | | | 4 | dir | -0.76 | 0.55 | -0.62 | 100.0 | 0.0 | 181.1 |
| cfast s1 | 0.7889 | 0.0e+00 | FLAG | 10657 | 0 | dir | 0.09 | 0.78 | n/a | 0.0 | 0.0 | 0.0 |
|  | | | | | 1 | dir | -0.58 | 0.17 | 0.66 | 100.0 | 0.2 | 51.6 |
|  | | | | | 2 | dir | -0.74 | 0.58 | 0.11 | 100.0 | 0.0 | 63.9 |
|  | | | | | 3 | dir | -0.75 | 0.42 | -0.52 | 100.0 | 0.0 | 167.7 |
|  | | | | | 4 | dir | -0.82 | 0.24 | -0.82 | 100.0 | 0.1 | 173.4 |
| creplay s0 | 1.1370 | 3.1e-05 | FLAG | 1668 | 0 | dir | 0.45 | 0.77 | 0.68 | 2.8 | 0.0 | 0.0 |
|  | | | | | 1 | dir | -0.64 | 0.28 | 0.76 | 100.0 | 0.5 | 44.6 |
|  | | | | | 2 | dir | -0.67 | 0.48 | -0.16 | 100.0 | 0.4 | 63.6 |
|  | | | | | 3 | dir | 0.13 | 0.76 | 0.18 | 31.4 | 0.4 | 0.1 |
|  | | | | | 4 | dir | -0.76 | 0.36 | -0.46 | 100.0 | 0.0 | 148.9 |
| creplay s1 | 1.3286 | 6.1e-05 | FLAG | 2028 | 0 | dir | 0.17 | 0.78 | 0.36 | 2.8 | 0.0 | 0.0 |
|  | | | | | 1 | dir | -0.65 | 0.24 | 0.33 | 100.0 | 0.6 | 51.5 |
|  | | | | | 2 | dir | -0.79 | 0.26 | -0.26 | 100.0 | 2.1 | 105.9 |
|  | | | | | 3 | dir | -0.71 | 0.64 | -0.21 | 100.0 | 1.7 | 70.2 |
|  | | | | | 4 | dir | -0.76 | 0.30 | -0.41 | 100.0 | 0.2 | 171.2 |
| cstart s0 | 1.4762 | 7.8e-03 | None | 2907 | 0 | dir | 0.62 | 0.78 | 0.47 | 2.9 | 0.0 | 0.0 |
|  | | | | | 1 | dir | 0.03 | 0.32 | 0.30 | 53.0 | 0.0 | 53.0 |
|  | | | | | 2 | dir | -0.67 | 0.48 | 0.78 | 100.0 | 0.0 | 104.3 |
|  | | | | | 3 | dir | -0.38 | 0.73 | 0.76 | 100.0 | 0.5 | 72.8 |
|  | | | | | 4 | dir | -0.65 | 0.43 | 0.40 | 100.0 | 0.0 | 123.1 |
| cstart s1 | 1.7878 | 0.0e+00 | None | 5421 | 0 | dir | 0.25 | 0.78 | 0.46 | 2.9 | 0.0 | 0.0 |
|  | | | | | 1 | dir | -0.81 | 0.44 | -0.25 | 100.0 | 1.1 | 22.6 |
|  | | | | | 2 | dir | -0.67 | 0.39 | 0.66 | 100.0 | 0.1 | 152.9 |
|  | | | | | 3 | dir | -0.75 | 0.47 | -0.32 | 100.0 | 0.1 | 123.8 |
|  | | | | | 4 | dir | -0.64 | 0.01 | 0.74 | 100.0 | 0.1 | 165.5 |

#### dimers

| variant (seed) | F (gap) | worst final viol | old probe flag | updates | level | w | rho pool | rho probes | rho own | starved % probe | own null % | median deficit (nats) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cfast s0 | -0.2266 | 1.2e-04 | FLAG | 8504 | 0 | dir | 0.01 | n/a | 0.03 | 43.4 | 0.1 | 0.2 |
| cfast s1 | -0.2258 | 1.1e-04 | FLAG | 3872 | 0 | dir | -0.00 | n/a | 0.01 | 44.4 | 0.0 | 0.3 |

## Proposed gate rule (proposal only, harness unchanged)

Per seed, using the subtree weight (it equals direct at level 0 and on ground, and is the only informative one on toy coarse levels):

- **A (primary). Starvation at every level: starved_probe(l) <= max(3 %, own-null(l) + 2 pp).** Basis: passing rows have 0-1.5 % (cbig256, cfast toy_modes; null 0-0.2 %); failing ones 5.4-8.6 % (toy_support cfast, chot) and 31-100 % (ground, columns). A threshold in 3-5 % separates them; 3 % keeps the two toy_support variants with F gap +0.02 on the fail side. Only five rows sit between 1.5 and 8.6 %, so the number is a first guess.
- **B (secondary). rho on probes alone >= 0.8 x the exact-p ideal at level 0 where an exact ideal exists (toy_modes: >= 0.70 for ideal 0.87); elsewhere rho on probes alone >= 0.4 at every level with a defined rho.** Passing: toy_modes 0.87-0.89, toy_support 0.44-0.81. Not discriminating on ground at level 0 (0.77-0.78 for all, collapsed or not), so B only bites on ground at L1 (0.17-0.44) and in columns (<= 0.23). B is weaker than A; the 0.4 is near the lowest passing toy_support value (0.44) and would need an exact-ideal calculation for toy_support and ground to calibrate.
- Do not use pooled rho (Simpson on ground), and do not use direct weight on levels whose direct weight is constant.

Result of A and B per seed (`gate.md`): pass: cbig256 toy_modes (4/4 seeds incl. the baselines300 repeat), cfast toy_modes (2/2), cbig256 toy_support (4/4). Fail: cfast toy_support (2/2, by A at 5.4 and 7.5 %), chot toy_support (2/2, 8.2-8.6 %), every ground row (cfast, creplay, cstart; 6/6), every columns row (6/6), dimers cfast (2/2, by A, unresolved as above).

Agreement with the images and F: ground rows all fail and all images are collapsed (no fill line), so agreement is complete there, but note the old flag also failed them and the check adds the level (L1-L4, whole coarse stack). On toy_modes the gate passes the two samplers that are within 0.002-0.014 nats/cell of F*, where the old flag fires on all. I did not view toy images (not informative); on toy_support the gate's verdict agrees with the F ordering but cfast's final violations (3.7e-4) already fail acceptance independently. Where the gate has not been tested: a sampler that is partly working on ground (the one near miss is creplay s0 level 3, 31 %); no row yet has starvation between 10 and 50 % except ground L1 at 53 %.

## Caveats
- Starvation saturates at 100 % on ground: own and probe weights barely overlap, so "at the same energy" is an extrapolation; add a range-overlap rule (probe weight within the own 5-95 % range) before relying on counts there (probe2 report).
- Two seeds per row, shared GPU, one 300 s budget; thresholds are first guesses from 8 variant-problem combinations.
- cstart rows are another agent's, included because they are v2 at 300 s; I did not verify its settings.
- Ideal rho exists only for toy_modes. toy_support's L0 0.71 is the value reached at F = F*, which suggests the toy_support ideal is about 0.71 too.

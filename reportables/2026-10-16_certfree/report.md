# certfree: certificate-free pass, certificates derived after (ideas #22 target B)

Agent certfree, 2026-10-16 overnight, wave A. GPU 0 shared throughout (recur, longfit2 and cdata jobs ran on GPU 0/1 at the same time; times and updates/s are contended). Cores 0-3.

## Question
Can the coloured pass train and sample only rho and tiles (certificate-free ground), derive the certificates afterwards by the tight map, and finish with the fixed stage with a short certificate refill from the derived certificates (F 20 instead of random F 150), with the full ground state valid?

User decision cited: dropping certificates changes the target (the tile marginal loses ground's log #valid-certificates term), accepted for this line (ideas.md #22, target B). The derivation is declared as an abstraction map of the level below (constraint 6, generic form): the parent's A is the largest value allowed by every declared `honourA` count factor (min over child rows of the row sum of A), B the smallest allowed by every `honourB` factor (max over child rows of the row sum of B), read from the factors' subsets. On a support-valid terrain it equals ground's `lift` (tested). Constraints 1, 2 (everything local, fixed counts, stage keyed by absolute coordinates), 3-5 unchanged; the stage is the 2026-10-16 amendment.

## What was built
- `castlegen/variants/_certfree_problem.py`: `make_ground_cf` (ground without cert channels, honour and support1..4; hard factor left: support0; same rho, tiles, soft energies, count prior; probes filtered), `derive_certs` (numpy / jnp).
- `castlegen/variants/ccertfree.py`: `ccertfree` = `cfitpen_ml2` training on ground_cf (init=None: the ml2 warm start from sweep2a has a different input width, so this is cold), generation = pass -> derive -> stage on the FULL ground (R terrain rounds, then certificates re-derived from the stage's tiles, then F refill sweeps from them) -> full ground state. `staged = True`. cfg `stage_R` (30), `stage_F` (20). Not registered in the bench problem registry (frozen file); the variant builds ground_cf internally from the bench's ground, scoring and the bench checks run on full ground.
- `tests/variants/test_certfree.py` (5 pass, CPU): cf problem has only support0 hard; derived == lift and zero violations on 25 random support-valid terrains; honour holds on arbitrary tiles; jnp == numpy; zero-stage variant output is pass + derived certs with honour satisfied.

## Numbers
Training 30 min (1800 s incl. 262 s setup), cold start, seed 0: `runs/train30/` (params.pkl, loop.jsonl, train_summary.json). 10176 ML updates, 20353 penalty updates, 6.6 ML updates/s after setup; lam = [19.8, 0, 0, 0, 0] (only the tile level has a hard factor). cfitpen_ml2 300 s bench: 2128 ML updates (7.1/s, warm). Same order; both contended.

Stage table, 128 held-out worlds, W 128, F 20 (`scripts/tables.py`, `runs/tables30.json`; chain reference rough 1.40, offset -0.15; ml2 rows from `2026-10-16_combo/runs/q2_ml2_300s.json`, F 150 random refill, 300 s warm-started pass):

| pass | R | offset | rough (circ) | mean abs c1 | line_match | viol L0..L4 | valid/128 | tiles changed |
|---|---|---|---|---|---|---|---|---|
| certfree 30 min | raw (pass + derived certs) | +0.93 | 1.22 | – | 0.666 | 51,70,41,17,9 | – | – |
| certfree 30 min | 15 | +0.56 | 1.39 | 0.40 | 0.674 | 0,0,0,0,0 | 128 | 2.4 % |
| certfree 30 min | 30 | +0.48 | 1.43 | 0.40 | 0.674 | 0,0,0,0,0 | 128 | 2.8 % |
| ml2 300 s | raw | +0.72 | 1.40 | 0.81 | 0.628 | 126,176,9,6,0 | 24 | – |
| ml2 300 s | 30 | +0.33 | 1.46 | 0.68 | 0.653 | 0,0,0,0,0 | 128 | 2.9 % |

Refill: at R 30 the output is identical for F = 0, 5, 20 (`tables30_F0.json`, `_F5.json`): the certificates derived from the stage's tiles are already valid (0 violations), so the refill is not needed for validity; it only randomizes among valid certificates. Raw violations (pass tiles, derived certs) are the pass's own support0 violations at L0 and their images above.

Bench, `python -m castlegen.bench run ccertfree ground --train 300 --seeds 2 --tag certfree` (300 s cold, 2080 updates, run `results/runs/20261016-093626_ccertfree_ground_certfree`, log `runs/bench.log`): final violations 0 at every level, both seeds; finite-support mismatch 0 (stage with the derivation included); seeds agree; line_match 0.651 / 0.629, roughness 1.37 / 1.51, free energy -0.117 / -0.088; probe flag fails (frac above 1.0 both, as for every staged variant so far); raw vs final not gated (staged). Gen 0.09-1.7 s per world (batched, cache-dependent).

Block check on the full ground state (v2; rho per level pool, starved % of probe blocks, seed 0 / seed 1): rho 0.23/0.73/0.84/0.39/-0.28 and 0.26/0.70/0.80/0.35/-0.29; starved 0.0/0.3/0.1/8.0/7.4 and 0.0/0.0/5.9/11.8/2.1. cfitpen_ml2 (board): 0/4/9/20/35. Caveat: certificate channels have no q term here (score_sites pads zeros), so the cert blocks are not the same objects as in ml2's check; rho and starvation of rho/tile blocks look better at L3-L4 but this is not a like-for-like comparison. A check on ground_cf itself is not done.

## Answer
Yes for validity and the stage: the pipeline works end to end, 128/128 valid at R 15 and R 30 with 0 violations at every level, finite support passes, and the 30-minute cold-start cf pass reaches line_match 0.674, roughness matching the chain (1.39-1.43 vs 1.40) and |c1| 0.40 (ml2 0.68), offset +0.5 (chain -0.15, ml2 +0.33 at R 30). The refill from derived certificates is not needed at all (F 0 = F 20). Not matched: training budget (1800 s cold vs ml2 300 s warm); the 300 s bench with the same budget gives the same line_match as ml2 within seed spread (0.651/0.629 vs 0.646/0.665), no gain and no loss. Probe flag still fails.

## Failed / not done
- No warm start (param shapes differ); no matched cold-start cfitpen_ml2 control, so "better at 30 min" is not separated from "longer training".
- Block check not run on ground_cf; coarse offset (+0.5 at R 30) remains above ml2's +0.33.
- Stage cost per world not timed separately (GPU contended).

## Next (one line each)
- Matched control: cold cfitpen_ml2 for 30 min on ground, same table.
- Block check / starvation on ground_cf (cf probes already provided).
- R sweep to 100/240 for the offset; warm start by transferring rho/tile nets with a re-initialised first layer.

Files: `castlegen/variants/{_certfree_problem,ccertfree}.py`, `tests/variants/test_certfree.py`, `reportables/2026-10-16_certfree/{scripts/{train,tables,smoke}.py, runs/}`.

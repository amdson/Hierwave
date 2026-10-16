# starve: fit problem or target problem? (agent starve, 2026-10-16)

Question: why is `cfitpen_ml2_stage` starved at coarse levels (v2 block check)? Is the training population covered by q (fit) or starved too (target)?
GPU 1 was shared with cdata (smi before/after in runs/). Cores 4-5. Not a bench run; no seeds beyond one params file (seed 0).

## Run
`taskset -c 4-5 python scripts/starve.py <variant> <params.pkl> runs/<x>.npz 320` then `scripts/analyse.py`.
starve.py: own draws (128 worlds, generate()["raw"]) and probes (128 worlds) exactly as `block_check`; population = CKernelPop's training loop (refresh 1/32, refine + 20 cg + 2 hard sweeps) re-run 320 iterations from pass draws with params frozen, snapshots at it 64,80,...,320 (17 x 32 = 544 states, ages <=32 iterations). All three sets are scored with the same `score_sites`/`weight_shares_signed` and the same own-draw Theil-Sen line; starved = >5 nats below it. Slack = mean (B-A) of the 4 cert sites of the block (columns, raw); dirt = mean rho/h^2.
Params: `arm_cfitpen_ml2/params.pkl` (stage variant class, raw draws) and `arm_ckpop_hard_pen/params.pkl`.

## Starved % of blocks (direct weight), cfitpen_ml2 (reproduces board 0/4/9/20/35: here 0/1.6/9.1/23.2/40.6)
| level | own | probes | population | Spearman rho own/probe/pop |
|---|---|---|---|---|
| L1 | 0.8 | 1.6 | 2.7 | .54/.45/.63 |
| L2 | 2.3 | 9.1 | 9.3 | .65/.64/.58 |
| L3 | 6.1 | 23.2 | 24.5 | .64/.72/.42 |
| L4 | 2.6 | 40.6 | 2.3 | .28/.25/.34 |

arm_ckpop_hard_pen (earlier fit), probes / population: L2 9.2/10.6, L3 28.9/36.4, L4 25.1/4.0. Own L3 mean slack hard_pen 0.39 vs ml2 0.76 (probes 0.87, pop 0.92): the fit change made the pass draw more high-slack L3 blocks, starved fraction at L3 fell for probes (28.9 to 23.2) and pop (36.4 to 24.5); L4 probe starvation rose (25 to 41).

## What the starved blocks are
- L2/L3 (probe and population alike): blocks with slack > 0 that the surface crosses. Starved fraction by slack at L3: slack 0 -> 0%, (0,1.5] 20-25%, (1.5,3.5] 69-92%, >3.5 87-98%. By dirt at L3: mixed blocks (dirt 0.2-0.6) are 83-100% starved in probes and 99-100% in the population; pure blocks 13-18%. Starved L3 blocks: mean slack 2.8-3.2, dirt 0.43-0.46; non-starved: slack 0.16-0.32 (mostly 0), dirt 0.52.
- The pass does draw such blocks (own L3 slack>1.6: 23% of blocks vs probes 24%, pop 27%; slack histograms are alike) but gives them low q relative to the line: it is not that high-slack certificates are never drawn; they are drawn and under-weighted per block.
- L4: starvation exists only for probes (pop 2.3%). Within the own-draw weight range (-9.7..-2.4) probes are still 30-71% starved, pop 1-2%. Not tied to slack (probe starved slack 3.6 vs ok 3.9), dirt (48% starved at dirt<0.2, 35% at >0.6) or jitter amplitude (37-45% for all four). Probe L4 weights are higher (median -2.1) than any population/own mass (median -5.5) and probe log q is ~3 nats lower: probes are smooth, fill-exact, low-entropy states the pass/population never produce at L4.

## Answer
- L2-L3: fit problem. The training population is starved under q exactly as the probes are (24.5% vs 23.2% at L3, 9.3 vs 9.1 at L2): the population contains these blocks, q gives them low q. The pass has not fitted its own refined population at the block level (also visible: pop rho at L3 0.42).
- L4: target/support problem, not fit: q fits the population (2.3% starved) but the population does not cover probe-like L4 blocks, so no amount of fitting on this population fixes the 40%; it needs population states with probe-like L4 structure (not allowed as probe lifts under the kernel route) or a decision that this probe set is atypical under p*. Negative L4 rho with rho .28/.25/.34 is weak for all three sets, so the "negative L4 rank correlation" of the board is not reproduced in this pool (own-line deficits only).
Caveats: one params file; weights are single-state energies (probes are low entropy, so extrapolating the own-draw line to them at L4 is partly a metric artefact); population is rebuilt under frozen final params, not the training-time population.

Files: scripts/starve.py, scripts/analyse.py, runs/{ml2,hp}.npz, runs/{ml2,hp}_analysis.txt, runs/*.log.

# relax4: clongfit (ML-only fit, 300 s) on 4 seeds, relaxed vs exact-lift probes, and L3 block classes

GPU 0 was shared with other agents' jobs (clocks 1875 MHz, 77 C, no throttle after training). **Confound:** seeds 0 and 1 (longfit2 params) got 912 / 928 updates in 300 s; seeds 2 and 3 (trained here, same recipe, `longfit2/scripts/train.py clongfit 300 <s>`) got 1920 / 1888, because the GPU was less contended. Results below do not track update count (seeds 2, 3 are not better at L3).

## Commands
`scripts/train23.sh` (seeds 2, 3 -> `runs/train/params_s{2,3}.pkl`), then `scripts/analyse.sh`: `l4source/scripts/relaxcheck.py` (full v2 `block_check`, exact-lift and relaxed probes = tiles fixed + 10 hard Gibbs sweeps of levels 4..1) on the four models plus cfitpen_ml2, then `scripts/classes.py` (starve's measure on L3, relaxed probes). Outputs `runs/relaxcheck.{json,log}`, `runs/classes.{json,log}`. Variant `clongfit` raw draws, as in l4source (not the bench's `clongfit_stage` run).

## Table 1: starved % of probe blocks per level, exact-lift -> relaxed (own-draw starved % in last column; rho_pool in 2nd table)
| model | L1 | L2 | L3 | L4 | own L3 / L4 |
|---|---|---|---|---|---|
| clongfit s0 | 0.5 -> 0.9 | 4.4 -> 3.0 | 14.3 -> 7.8 | 31.0 -> 4.6 | 8.3 / 2.0 |
| clongfit s1 | 0.3 -> 0.8 | 4.7 -> 3.5 | 13.3 -> 8.0 | 30.6 -> 8.2 | 8.4 / 2.4 |
| clongfit s2 | 0.3 -> 0.7 | 4.3 -> 2.9 | 12.8 -> 6.9 | 36.1 -> 10.0 | 8.0 / 1.8 |
| clongfit s3 | 0.3 -> 0.8 | 3.9 -> 2.5 | 13.7 -> 7.1 | 31.4 -> 8.4 | 7.7 / 2.2 |
| cfitpen_ml2 | 1.6 -> 3.2 | 9.1 -> 9.3 | 23.2 -> 21.9 | 40.6 -> 3.9 | 6.1 / 2.6 |

L0 is 0 everywhere.

rho_pool, exact-lift -> relaxed:
| model | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| s0 | .84 -> .67 | .86 -> .67 | .51 -> .53 | -.16 -> +.31 |
| s1 | .85 -> .68 | .85 -> .67 | .48 -> .52 | -.13 -> +.32 |
| s2 | .86 -> .68 | .86 -> .66 | .36 -> .48 | -.16 -> +.32 |
| s3 | .86 -> .67 | .87 -> .67 | .46 -> .49 | -.12 -> +.33 |
| ml2 | .83 -> .58 | .82 -> .61 | .40 -> .54 | -.25 -> +.31 |

## Table 2: L3 block classes under relaxed probes (starved = deficit > 5 nats below the own-draw Theil-Sen line)
hi_cross = block-mean slack B-A > 1.5 and dirt in (0.05, 0.95) (surface-crossing); slack0 = slack exactly 0. Fraction = share of L3 blocks; "q share" = the class's share of total log q share summed over blocks (log q is negative, so a share above the block fraction means the class carries more log-q cost); "share of starved" = fraction of all starved blocks in the class.

| model | class | probe: frac blocks | starved | median deficit (nats) | q share | share of starved | own: starved | own median deficit |
|---|---|---|---|---|---|---|---|---|
| s0 | hi_cross | .256 | 28.9 % | 3.61 | .343 | 95.6 % | 29.3 % | 3.33 |
| s0 | slack0 | .667 | 0.1 % | -0.48 | .579 | 0.6 % | 0.0 % | -0.54 |
| s1 | hi_cross | .256 | 31.2 % | 3.83 | .345 | 97.9 % | 31.0 % | 3.41 |
| s1 | slack0 | .667 | 0.1 % | -0.52 | .577 | 0.6 % | 0.0 % | -0.62 |
| s2 | hi_cross | .256 | 26.2 % | 3.37 | .342 | 97.9 % | 29.8 % | 3.44 |
| s2 | slack0 | .667 | 0.0 % | -0.61 | .580 | 0.0 % | 0.0 % | -0.65 |
| s3 | hi_cross | .256 | 26.5 % | 3.53 | .342 | 95.9 % | 28.4 % | 3.48 |
| s3 | slack0 | .667 | 0.1 % | -0.52 | .580 | 1.0 % | 0.0 % | -0.59 |
| cfitpen_ml2 (relaxed, this run) | hi_cross | .256 | 81.4 % | 8.32 | .411 | 95.4 % | 22.5 % | 2.23 |
| cfitpen_ml2 (relaxed, this run) | slack0 | .667 | 0.1 % | -0.16 | .513 | 0.3 % | 0.1 % | -0.35 |

Starve's cfitpen_ml2 values (exact-lift probes, report `2026-10-16_starve`): starved fraction by slack at L3 is 0 % for slack 0, 20-25 % for (0, 1.5], 69-92 % for (1.5, 3.5], 87-98 % above 3.5; mixed (surface-crossing) blocks 83-100 % starved. Relaxed probes here give 81.4 % for ml2 hi_cross, consistent: the L3 deficit of ml2 is not a probe artefact.

## Answer
The L3 fit problem is gone for the no-penalty fit on every seed, not seed-dependent. Probe L3 starvation under relaxed probes is 6.9-8.0 % on all four seeds (ml2 21.9 %), close to each seed's own-draw level (7.7-8.4 %). High-slack surface-crossing blocks are starved 26-31 % for clongfit on probes, the same as on its own draws (28-31 %) and with the same median deficit (3.4-3.8 vs 3.3-3.5 nats); for ml2 they are 81 % on probes against 22 % on own draws (median deficit 8.3 vs 2.2). Slack-0 blocks are never starved. So the residual 26-31 % in the hi_cross class is the own-draw spread of the line itself, not a probe-specific under-weighting. Not clean: L4 under relaxed probes is 4.6 % (s0) but 8.2 / 10.0 / 8.4 % on seeds 1-3, above ml2 (3.9 %) and own draws (1.8-2.4 %); L4 rho stays +0.31..+0.33 on all seeds, so this is mild and not a rank failure, but seed 0 understated it. L1-L2 rho_pool dropped 0.84 -> 0.67 under relaxation on all seeds (as in l4source). Only one training run per seed, and seeds 2-3 had twice the updates of 0-1.

## What failed / next
- Nothing crashed. The update-count mismatch (912 vs 1900) is a confound for the seed comparison; seeds 0-1 could be rerun at equal updates if it matters.
- Next: look at why L4 relaxed starvation is 8-10 % on seeds 1-3 (is it again a probe-reference effect, e.g. 10 sweeps not mixed at L4, or a real fit gap), using l4source's `relax_{1,10,100}` sweep counts.

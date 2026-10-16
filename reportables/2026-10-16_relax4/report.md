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

## Deficit decomposition (follow-up, 13:05 UTC; no training; `scripts/decomp.py`, `runs/decomp.{json,log}`)
Per L3 block of the 4 clongfit models, own draws (128 worlds) and relaxed probes (128 worlds), classes as above. **A certificate-based log-weight share as specified does not exist in the bench:** `problems/ground.py` has no soft term in A or B; the certificate enters the target only through hard constraints (support, row honour) and the soft count prior / smoothness on rho. `shares.weight_shares_signed` returns the level-3 factors' energies (direct, D) and D plus all finer levels (subtree, S); the cert channel contributes 0 or -inf there. So (ii) is replaced by the subtree share S and by the certificate slack B-A itself. Deficit = line(own) - log q share, nats, line fitted on own draws separately for D and for S.

| class (median over 4 seeds; range) | set | rho(q, D) | rho(q, S) | deficit vs D | deficit vs S | median D-S | median slack | q-per-slack coef (nats), D held fixed | rho(deficit, slack) |
|---|---|---|---|---|---|---|---|---|---|
| hi_cross | own | .30-.31 | .50-.59 | 3.3-3.5 | 1.0-1.4 | +55..+57 | 3 | -1.1..-1.4 | .33-.42 |
| hi_cross | probe | .27-.36 | .20-.24 | 3.4-3.8 | 2.2-2.7 | +46 | 4 | -0.7..-0.95 | .11-.20 |
| slack0 | own | .54-.58 | .22-.27 | -0.5..-0.65 | -0.4 | +30 | 0 | n/a | n/a |
| slack0 | probe | .46-.53 | .23-.26 | -0.5..-0.6 | -0.1..-0.4 | +30 | 0 | n/a | n/a |

D-S is large and positive (D is a 8x8-block-level term only; S adds the fine levels, which are negative), so the two shares are on different scales (line slopes 0.81 vs 0.13) and each is compared to q through its own own-draw line.

Reading, plainly:
- q does not cleanly track either weight on high-slack crossing blocks. rho(q, D) is only 0.27-0.36 there (vs 0.46-0.58 for slack-0 blocks), and rho(q, S) is 0.5-0.6 on own draws but 0.20-0.24 on probes.
- The median deficit (3.4-3.8 nats) is the same size as the median slack (3-4), but this is a coincidence of medians, not an identity: the rank correlation of deficit with slack is only 0.11-0.42, and at fixed D, q's log share falls by 0.7-1.4 nats per unit of slack (OLS coefficient, 1.0-1.4 on own draws). So q is paying roughly one nat per unit of certificate slack that the direct weight does not charge: consistent with a block carrying B-A free certificate choices having lower per-sample log q (entropy of the certificate conditional) without a matching weight, but this is not proved here.
- Against the subtree weight the deficit is smaller on own draws (1.0-1.4 nats) but still 2.2-2.7 on probes, so the subtree does not explain it either.
- Hence neither "q fits the certificate and the gap is the certificate's" nor "q is a poor fit of the weight" is established. What is established: on probes and own draws alike the high-slack blocks sit about 3.5 nats under the direct line, the same size as their median slack, with a loose slack dependence; slack-0 blocks have no deficit. The test the coordinator wanted (an A/B-based weight) needs a definition of the pass's training target on certificates that the bench does not expose; I did not construct one.

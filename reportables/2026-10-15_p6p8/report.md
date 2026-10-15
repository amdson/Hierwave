# p6p8: fill tail at scale (P8) and derived-channel entropy (P6)
GPU 0 was shared with gpen_p11 (cores 6-7). Clocks before/after in runs/gpu_*.txt. Scripts in scripts/ (fill_tail.py, p6.py).

## P8: does the graded fill tail stay bounded (F ~ 80) as W grows?
Setup: `fill_len.py` logic (certificate-free terrain after 300 hard sweeps of mix.stripped, random certificates, certificate-only sweeps up to 400, first sweep with N = 0 per world). 32 worlds at W=128, 96 worlds at W=512 and 2048 (24 fill seeds x 4 start families).
Command: `taskset -c 6-7 python reportables/2026-10-15_p6p8/scripts/fill_tail.py <W> <NF>` (NF = 8 for W=128, 24 otherwise).

Fraction of worlds still invalid after k sweeps (a world = one W-wide strip):

| W | penalty | p50 | p90 | max | >20 | >40 | >60 | >80 | >120 | F for 2e-4 (exp. tail fit) |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | graded | 36 | 55 | 67 | 1 | .34 | .03 | 0 | 0 | ~125 |
| 128 | bin 20 | 44 | 87 | 164 | 1 | .59 | .28 | .13 | .03 | ~270 |
| 128 | bin 8 | 50 | 87 | 297 | .97 | .59 | .31 | .19 | .06 | ~330 |
| 512 | graded | 48 | 70 | 81 | 1 | .83 | .20 | .01 | 0 | ~140 |
| 512 | bin 20 | 70 | 123 | 162 | 1 | .93 | .67 | .36 | .11 | ~310 |
| 512 | bin 8 | 156 | 344 | >400 (44/96 valid) | 1 | 1 | .97 | .93 | .86 | no |
| 2048 | graded | 67 | 90 | 121 | 1 | 1 | .59 | .20 | .01 | ~165 |
| 2048 | bin 20 | 99 | 150 | 181 | 1 | 1 | .99 | .77 | .25 | ~310 |
| 2048 | bin 8 | none valid in 400 | | | 1 | 1 | 1 | 1 | 1 | no |

F(2e-4) is a crude extrapolation: exponential fit to the upper half of first-valid times (graded tail scale ~12 sweeps at every W; bin 20 ~27-31; bin 8 ~36), F = median + scale x ln(0.5/2e-4). With 96 worlds nothing below ~1e-2 is measured; treat as +/-30.

Answer: **No for "F ~ 80 enough"; yes for "bounded growth".** Graded does not blow up: its tail scale stays ~12 sweeps, but the whole distribution shifts by about +12 sweeps per 4x in W (median 36 -> 48 -> 67, max 67 -> 81 -> 121), the usual log W of a max over cells. At W=2048, F=80 leaves 20% invalid and F=120 leaves 1%; F ~ 165-170 reaches 2e-4. Binary 20 has the same log W shift but a 2.3x heavier tail (F ~ 310 at every W: 2x graded's); binary 8 is not a usable fill penalty at scale (no world valid in 400 sweeps at W=2048). Binary's max/graded's max: 2.4x at 128, 2.0x at 512, 1.5x at 2048 on observed maxima, but the tail-scale ratio is constant at ~2.3x. Binary's tail does not grow with W beyond the common log W shift.
Caveat: these are from-random-certificate fills on a single-site chain; a fill from the tight abstraction (P4) or a larger budget at a lower-W block would change the picture.

## P6: entropy of a channel given everything else (derived-channel rule)
Measure: plug-in entropy of the exact hard Gibbs conditional of each channel (softmax of -soft over values with nv = 0, from `colour_gibbs.conditional`), averaged over all cells of a colour sweep and over states. This is H(c | all neighbours, parent, children), a lower bound on H(c | level below). It is not the exact quantity asked for.
Ground: 16 valid worlds (cgraded final_binary_lam20.npz) at W=128, hard sweeps, states after 100/200/300 sweeps (all valid). Columns: 16 exact iid samples at W=128.

| ground level | rho H (nats/cell) | rho support | cert H (nats/cell) | cert support | rho/cert | cert H per fine cell |
|---|---|---|---|---|---|---|
| 1 | 0.559 | 5 | 0.030 | 1.05 | 19x | 7.5e-3 |
| 2 | 1.301 | 17 | 0.052 | 1.08 | 25x | 3.3e-3 |
| 3 | 2.116 | 65 | 0.107 | 1.18 | 20x | 1.7e-3 |
| 4 | 2.687 | 257 | 0.217 | 1.38 | 12x | 8.5e-4 |
| tiles (level 0) | 0.006 | 1.01 | | | | |

Cert H follows the slack (0.011/0.045/0.151/0.52): same ordering, about 0.4-0.7x of it. A threshold anywhere in 0.25-0.5 nats/cell separates certificates from rho at every level, but the gap shrinks with level (12x at level 4), and level 4 cert (0.22) is above where a "about zero" rule would drop it. Note the level-0 tile (a real channel) is also ~0.006 here, so the measure alone does not say "derived": tiles are near-deterministic given their surroundings but are not scaffolding.

| columns channel | H (nats/cell) | support |
|---|---|---|
| n1, n2, n3, n4 | 0.000 | 1.00 |
| k (fine) | 0.000 | 1.00 |

Columns' coarse channels are exactly derived (hard count equalities): H = 0, so the rule drops them, as expected. But k is also H = 0 given its neighbours: the length-12 chain constraint leaves each single cell no alternative.
Stripped single-site chain on columns (300 hard sweeps, 16 worlds, W=128): validity kept (0 violations) with and without coarse channels, but **0 cells change in any sweep**, from exact samples, under hard sweeps and under penalty 20. From an empty start the chain also stays empty (valid, density 0). Autocorrelation is undefined (frozen). So stripping coarse channels costs nothing in validity or mixing here, because mixing is already zero: single-site Gibbs on columns is non-ergodic with or without them.

Answer: **Yes on ground** (certs separate from rho by 12-25x, ordered with slack), **partly on columns**: the coarse channels are exactly derived (H = 0) but the rule cannot distinguish them from k, which is frozen for a different reason (hard length-12 support). The entropy rule identifies near-deterministic channels, not ones whose removal is harmless to ergodicity; for that the stripped-chain mixing test is also needed.

## What failed / caveats
- First columns run used the bool allowed-table as the factor table (my bug: 122375 violations); fixed, rerun in runs/p6_columns.log.
- P6 entropy is of the full conditional, not of the level-below-only conditional; ground used 16 worlds (not long chains), W=128.
- P8 tail extrapolations are exponential fits to 96 worlds.

## Next (one line each)
- P8: give P3 a fill budget of F ~ 170 sweeps at W=2048 (graded), or test the tight-abstraction fill (P4) whose tail may be shorter.
- P6: compute H(c | level below only) by marginalising neighbours, and add an ergodicity test (block-move acceptance) to the rule.

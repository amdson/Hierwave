# p4fill: fill from the tight abstraction (P4)
GPU 1 was shared (stage agent); clocks in runs/gpu_*.txt. Cores 4-5. Script: scripts/p4.py (copy of p6p8 fill_tail.py logic).
Commands: `taskset -c 4-5 python reportables/2026-10-15_p4fill/scripts/p4.py 512 16` (64 worlds) and `... 2048 8` (32 worlds). Terrain: 300 hard sweeps of mix.stripped (certificate-free). Results: runs/p4_512_n16.json, runs/p4_2048_n8.json.
Arms: rand_grad (random certs, graded lam 20, cert cells only); tight_grad (ground `lift` certs, same graded fill); tight_hard (lift certs, hard sweeps, extra).

## Table: valid fraction / A_tight - A at levels 1-4 (mean over all worlds)
| W | arm | k=0 | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|---|
| 512 | rand graded valid | 0 | 0 | 0 | 0 | .19 | 1.0 |
| 512 | tight graded valid | 1 | 1 | 1 | 1 | 1 | 1 |
| 512 | rand graded gap | - | .004/.004/-.03/-.16 | .006/.012/.023/.073 | .007/.017/.042/.133 | .007/.018/.048/.163 | .007/.0185/.050/.169 |
| 512 | tight graded gap | 0 | .007/.017/.043/.147 | .007/.018/.047/.153 | .007/.018/.048/.164 | .007/.019/.052/.167 | .007/.019/.053/.176 |
| 512 | tight hard gap | 0 | .007/.016/.043/.147 | .007/.018/.047/.158 | .007/.018/.047/.163 | .007/.019/.049/.160 | .007/.019/.050/.168 |
| 2048 | rand graded valid | 0 | 0 | 0 | 0 | 0 | .75 |
| 2048 | tight graded valid | 1 | 1 | 1 | 1 | 1 | 1 |
| 2048 | tight graded gap | 0 | .007/.017/.043/.145 | .007/.018/.048/.158 | .007/.019/.050/.167 | .007/.019/.052/.168 | .007/.019/.050/.169 |
| 2048 | rand graded gap (k=80) | | | | | | .007/.020/.051/.168 |
| 2048 | tight hard gap | 0 | .007/.017/.043/.144 | .007/.018/.048/.157 | .007/.019/.051/.171 | .007/.019/.051/.169 | .007/.019/.051/.171 |

(Rand-graded gap at k<=10 is of invalid states and not meaningful; tight arms are valid at every checked k.)

## Answer
**Yes.** Tight start is valid at every k (64/64 and 32/32 worlds at all checked k, graded and hard), so the tail (F ~ 80-165 for random start, W=2048 random still 25% invalid at 80) is gone. Slack reaches its plateau in ~5 sweeps at levels 1-2, ~10-20 at levels 3-4, the same at W=512 and 2048 (no visible W dependence). Plateau values 0.007/0.019/0.051/0.17 are identical to the random start's late values (k=80), so the tight start is not biased relative to the fill's own equilibrium. They are about 3x tighter than the chain reference 0.011/0.045/0.151/0.52, which is the known certificate-free-terrain effect (cmix §6: terrain not weighted by the number of valid certificate assignments), not a fill bias. Level 4 creeps slightly (0.147 -> 0.17 over 5 -> 80), so use ~20 sweeps.
Only checked at the coarse k grid; validity checked at those k only (not every sweep), but the hard-sweep arm keeps validity by construction.

## Constitution status
The tight value is ground's `lift` (problem-specific). It is generic only if certificates are declared as abstraction maps of the level below; otherwise constraint 6 blocks it. This is a diagnostic. The fill itself is fixed-count local sweeps (constraint 2 fine).

## What failed / caveats
Nothing failed. Slack was not compared to the chain reference on the same (p-weighted) terrain, only to the random-start fill on certificate-free terrain. Stationarity beyond 80 sweeps not checked.

## Next
Replace the random-start fill in P3/P10 with the tight start at F ~ 20 (cuts the fill budget ~8x at W=2048), if the DSL declares certificates as abstraction maps.

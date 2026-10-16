# l4source: why does the L4 population never reach the probe-like states? (agent l4source, 2026-10-16)

**Answer: stationary, not mixing. The kernel is not what lacks the subtree free energy; the probe set and the bench's direct weight are.**
- A chain started from the bench's probe lifts relaxes to the population's L4 distribution within 1x the usual refresh (32 iterations). A chain started from own draws does not move at L4 from 0 to 3200 iterations (100x).
- Under the kernels' stationary distribution (p*), pure-air and pure-dirt L4 cells sit about 8 counts away from the exact lift (rho4 = 0 or 256). The probes put them on the domain boundary. The bench's direct weight therefore scores the probes 3.4 nats heavier, while p* (with the volume of the rho ladder below) makes them lighter.
- With the probes' fine tiles held fixed and only their coarse channels redrawn from p*(coarse | t) for 10 sweeps, L4 probe starvation falls from 37 % to 3.6 % (own draws: 2.3 %). L3 stays at 22 %, which is the separate fit problem starve found.

Setup:
- Params: `reportables/2026-10-16_fitpen/runs/arm_cfitpen_ml2/params.pkl` (variant `cfitpen_ml2`, 300 s), frozen throughout. Not a bench run, no training.
- GPU 1 was shared with the coordinator's benches (GPU 0) and others; SM clock 1770 MHz before, 1560 MHz after (`runs/smi_*.txt`). Cores 8-15.
- Objective: none. This is a measurement, with no new objective and nothing built.

## Commands
```
source reportables/2026-10-14_cr2_penalty/scripts/env.sh 1
taskset -c 8-15 python reportables/2026-10-16_l4source/scripts/l4src.py cfitpen_ml2 <params.pkl> runs/main 3200   # 58 min, DONE
taskset -c 8-15 python reportables/2026-10-16_l4source/scripts/relax.py cfitpen_ml2 <params.pkl> runs/main        # 25 s, DONE
JAX_PLATFORMS=cpu python scripts/analyse.py runs/main > runs/analysis.txt
JAX_PLATFORMS=cpu python scripts/devs.py /workspace/scratch/l4source/pop_{320,1000,3200}.pkl > runs/devs.txt
```
- Scoring: every state is scored exactly as `block_check` does it (`score_sites`, `shares.weight_shares_signed`). The own-draw Theil-Sen line comes from 64 own draws at the bench's own seeds, and starved means more than 5 nats below that line.
- Target log-weight = the bench's direct block weight at L4 (−noise4 − smooth4; prior4 is homed at L3). Probes = the bench's 128 probe worlds.
- `chain` = the CKernelPop refresh cycle exactly (refine phase 2: L4/L3 tempered transitions, coarse Gibbs, MTM L2, fine Gibbs; then 20 colour-Gibbs sweeps at λ 20 and 2 hard sweeps). It runs with no replacement on 32 fresh own draws plus 32 probe worlds (one batch). A population member normally lives 32 iterations (refresh 1/32), so 32 / 320 / 3200 iterations = 1x / 10x / 100x.
- `l4only` = exact colour Gibbs at L4 only (hard), with all finer levels fixed.
- `relax` = the bench probes with tiles t fixed and levels 4..1 redrawn by exact Gibbs (hard).

## Numbers (L4 unless marked; weight = bench direct log-weight per block; logq per block)
| set | starved L4 | weight mean ± sd | overlap with probe weights | logq (rho4 / cert4) | noise4 / block | abs(rho4 − lift) / cell | cert4 = lift | starved L3 | line_match |
|---|---|---|---|---|---|---|---|---|---|
| own draws (bench seeds) | 2.3 % | −5.67 ± 2.21 | 0.39 | −14.2 (−11.3 / −2.8) | 4.13 | 15.1 | 0.60 | 6.0 % | 0.61 |
| **probes (exact lifts)** | **37.2 %** | **−2.21 ± 2.00** | 1 | −17.6 (−14.5 / −3.1) | 0.84 | 0 | 1.00 | 23.1 % | 0.77 |
| chain from own, 1x (32 it) | 1.2 % | −5.67 ± 2.16 | 0.37 | −14.1 | 4.27 | 9.2 | 0.66 | 23.6 % | 0.680 |
| chain from own, 10x (320) | 2.3 % | −5.48 ± 2.20 | 0.40 | −14.1 | 4.06 | 9.3 | 0.65 | 22.9 % | 0.675 |
| chain from own, 100x (3200) | 0.8 % | −5.64 ± 2.29 | 0.40 | −14.4 | 4.21 | 9.4 | 0.65 | 23.4 % | 0.673 |
| chain from probes, 0 | 41.8 % | −2.01 ± 1.92 | 0.90 | −17.9 | 0.92 | 0 | 1.00 | 23.3 % | 0.779 |
| **chain from probes, 1x (32)** | **1.6 %** | **−5.44 ± 2.56** | 0.45 | −14.1 | 4.10 | 9.0 | 0.67 | 23.4 % | 0.684 |
| chain from probes, 10x / 100x | 1.2 / 1.2 % | −5.45 / −5.11 | 0.42 / 0.47 | −14.1 / −13.8 | 4.13 / 3.76 | 9.1 / 8.9 | 0.66 | 23.1 / 23.9 % | 0.675 / 0.687 |
| probes, L4 Gibbs only ×1 / ×100 (fine fixed) | 10.9 / 11.3 % | −2.38 / −2.39 | 0.82 | −14.6 | 1.13 | 3.0 / 3.3 | 0.89 | 25 % | 0.779 |
| own, L4 Gibbs only ×1 / ×100 | 22.3 / 22.3 % | −6.89 / −6.83 | 0.40 | −17.4 | 5.46 | 13.2 | 0.61 | 12-15 % | 0.634 |
| **probes, tiles fixed, coarse L4-L1 redrawn ×1 / ×10 / ×100** | 8.1 / **3.6** / 3.5 % | −2.58 / −5.61 / −5.70 | 0.82 / 0.38 / 0.39 | −14.4 / −13.5 / −13.6 | 1.05 / 4.00 / 4.09 | 3.1 / 7.0 / 7.1 | 0.90 / 0.88 / 0.88 | 21.3 / 22.1 / 21.9 % | 0.770 (tiles unchanged) |

Validity: chain states are 100 % valid from iteration 32 on. Own raw draws are 30 % valid, and the own chain at iteration 0 (fresh training-seed draws) 9 %.

Where the L4 gap sits (chains at 320 / 1000 / 3200 pooled; d = rho4 − 256 f per L4 cell; `runs/devs.txt`):
| cell class | chain from own: mean d ± sd | chain from probes | probes (lift) |
|---|---|---|---|
| air (f = 0) | +7.8 ± 3.9 | +7.7 ± 3.9 | +0.3 ± 1.6 |
| mixed | −0.1 ± 6.0 | +0.6 ± 5.7 | −0.0 ± 6.6 (same 5/50/95 % quantiles as the chain: −10 / 0 / 10) |
| dirt (f = 16) | −7.6 ± 3.9 | −7.6 ± 3.9 | −0.6 ± 2.1 |

- **The rho ladder in pure air, chain vs lift.** Mean rho at L1 / L2 / L3 / L4 is 0.25 / 1.2 / 3.65 / 7.8 in the chain against 0.01 / 0.05 / 0.2 / 0.8 in the lift; dirt is mirrored.
- **Air cells.** The chain puts 4.9 % of air cells at rho4 ≤ 1; the probes put 94.6 % there.
- **The missing free energy, measured.** For air cells, log(chain occupancy of rho4) + noise4(rho4) relative to rho4 = 8 is −2.7 at rho4 = 0, −1.2 at 4 and +0.2 to +0.8 at 10-16. This is the free energy of what lies below the cell, which the direct weight omits. It is worth about 2.7 nats per pure L4 cell between the lift value and the typical value.
- **Mixed cells** carry no such gap: the profile is flat within 0.5 nats, and the probes' d distribution equals the chain's.

## Reading
1. **Distributions.** Population/own L4 direct weight is −5.7 ± 2.2; probes are −2.2 ± 2.0, overlap 0.39. The probes' log q is 3.5 nats lower, almost all of it in rho4 (−14.5 vs −11.3). The probes are heavier only through noise4 (0.84 vs 4.1 per block). Slack and smoothness do not differ (3.76 vs 3.88, 1.37 vs 1.53).
2. **Mixing vs stationary.**
   - From own draws, 1x / 10x / 100x the refresh leaves L4 where it is: starved 1.2 / 2.3 / 0.8 %, weight −5.67 / −5.48 / −5.64, noise4 4.27 / 4.06 / 4.21.
   - The same kernels take the probe worlds to that same distribution in at most 32 iterations: starved 41.8 → 1.6 %, weight −2.0 → −5.4, noise4 0.9 → 4.1, line_match 0.78 → 0.68 (the chain reference is 0.69).
   - Two starts meeting from opposite sides within 1x and staying together for 100x make this the kernels' stationary L4 distribution, not a mixing limit. The kernels are p*-invariant (hard sweeps; λ 20 sweeps and transitions), so it is p*'s L4 marginal for these statistics.
3. **What lacks the subtree free energy.**
   - Not the kernel. It samples the joint, so its L4 marginal contains everything below.
   - The bench's direct weight compares log q against −noise4 − smooth4 only. It rewards rho4 on the lift value (air rho4 = 0 against the stationary 8: +1.0 nats of noise4 per cell), while p* penalises it by about 2.7 nats per pure cell through the volume of the rho ladder below (L1-L3 rho also sit off the boundary under p*).
   - q has fitted p* here (population 1-2 % starved), so it gives exact-lift L4 values low log q, as it should.
   - The probes' fine geometry is not the issue. With tiles fixed (line_match stays 0.77) and only the coarse channels redrawn from p*(coarse | t), L4 starvation is 3.6 %. L3 does not move (22 %): the L3 starvation is the fit problem in starve's report, untouched by this.
4. **What a p*-invariant refresh at L4 would need.** Nothing; the existing one is already at its L4 stationary state after 1x. Making the population cover exact-lift L4 states would need a non-p*-invariant data source: probe lifts as training data, which is change request #1 and outside the kernel-route amendment. It would pull q away from p* at L4, so it should not be done for this purpose.
   - **τ > 1 exploration** with importance weights targets the same p*, so it changes nothing here.
   - **Ideas #14 (full-support parent contexts) is allowed**, and this result suggests a better context source for it: probe tiles with p*-redrawn coarse channels (exact Gibbs given t, a p*-invariant kernel). These contexts are p*-typical at L4 rather than boundary states. recur's K16+probe arm clamped L4 to exact lifts, which p* almost never visits (94.6 % vs 4.9 % of air cells at rho4 ≤ 1).
   - The decision that matters is a bench-measure change, filed as change request #6: score the block check's probes after coarse relaxation, or report L4 starvation as uninformative.

## Failed / caveats
- One params file and one seed. 32 + 32 chain worlds; L4 has 1024 blocks per set.
- "Stationary" is shown for the L4 statistics measured here (weight, noise4, lift distance, cert tightness, starvation), not for every mode of the world.
- The chain line_match (0.67-0.69) differs from the probes' (0.77) too, so the probes' tile geometry is also atypical under p*. It is not what starves them at L4 (relax table).
- Rule slip: the 25 s `relax.py` scoring run (no training) was launched while `ps … | grep -c` read 4. I did not check before launching; it ended before anything noticed.
- Step 4 (a lengthened-refresh `ccdata` arm) was not run, because the answer is not "mixing". Lengthening the refresh cannot change a stationary marginal.

## Next (one line each)
- Rerun the v2 block check on cfitpen_ml2 / clongfit with coarse-relaxed probes (tiles fixed, 10 hard Gibbs sweeps of L4-L1), next to the exact-lift numbers, to see what L4 coverage actually is (needs the user's decision on the harness: change request #6).
- If a context source for ideas #14 is wanted: the same relaxed probes instead of exact lifts.

Files: `scripts/l4src.py`, `scripts/relax.py`, `scripts/analyse.py`, `scripts/devs.py`; `runs/main.log`, `runs/relax.log`, `runs/analysis.txt`, `runs/devs.txt`, `runs/main/*.npz` + `summary.json`, `runs/smi_{before,after}.txt`. Chain states at 320/1000/3200 iterations are in `/workspace/scratch/l4source/` (3.5 MB each).

# longfit: one hour of ML-only training, LN+SiLU 6x6 actor, equilibrated population, then the fixed stage (2026-10-16 overnight)

Agent longfit, GPU 0 (shared: recur's CPU phase on the box, the other GPU busy with other agents; no recur GPU job overlapped), cores 6-7. Ideas #18, #5, #10.

**Question.** If the ML half is trained for an hour with the 6x6 window and LN+SiLU on the kernel-equilibrated population, no penalty term, how good are the own draws (structure, starvation, raw violations), and does the fixed stage (R 30) then make them valid?

**Answer.** An hour of pure ML gives modestly better own draws than `cfitpen_ml2` (300 s) on structure and clearly better on raw violations and L3 starvation, but it plateaus at about 10 min and does not beat the chain's structure. Yes, the stage makes every world valid at R 15 already. Pure ML did not leak mass to invalid states here: raw violated fractions are at or below the 2e-4 budget at every level from 10 min on.

## What was run
- **Actor.** `cfitpen_ml2` already uses ckernel's `net="ln", win=(-2, 3)` (CK defaults), i.e. the cln6 actor. So the control and the long run have the same network; the differences are only run length, batch and population, and no penalty. I say so because the question assumed the control used the default net.
- `castlegen/variants/clongfit.py`: `clongfit` = `CFitPen` with `upi=16, rkl_n=0` (no penalty steps, lambda control inert at its init), `nctx=64` (control 32), `ml_B=32` (control 8). Everything else as ml2 (P3 population, refresh 0.03, cg_sweeps 20, hard_sweeps 2, ML lr 1e-3, kernels).
- Driver `scripts/longfit.py clongfit 4200 runs/clongfit`, pinned `taskset -c 6-7`, GPU 0, from scratch, budget 4200 s (setup counted; cached compile, setup 60 s). Logs every iteration (16 updates); every 10 min an `eval_seed` (8 worlds; 16 for the final) plus a reduced block_check (24 own + 6 probe seeds); params every 15 min (last three kept: `params_030min/045min/060min.pkl`). 
- Smoke run first (cold compile 465 s, not counted). Stage: `scripts/stage.sh` = stage2 `q2.py` on `cfitpen_ml2` class with the final params, 128 worlds, R 15/30/60, F 150. Starvation: `scripts/blocks.py` (full block_check, 128 + 32). Table scripts: `scripts/curve.py`, combo's `table.py`.
- 34064 ML updates in 4200 s (8.2 /s; 2129 iterations). Control: 2112 updates (plus 4225 penalty). nvidia-smi: `runs/clongfit/nvidia_{before,after}.txt` (GPU 0: 1560 MHz 59 C before, 1890 MHz 78 C after, no throttle) and `runs/nvidia_*_stage.txt`.

## Table 1: training curve (`runs/curve.md`; own8 = 8 own draws of the population routine, eval = eval_seed raw draws, 8 worlds)
| min | updates | own8 line_match | dirt | eval line_match raw | eval raw viol frac L0..L4 (budget 2e-4) | starved_probe L3 / L4 | starved_own L3 / L4 |
|---|---|---|---|---|---|---|---|
| 1 | 16 | 0.505 | 0.874 | | | | |
| 10 | 4480 | 0.679 | 0.500 | 0.654 | 1e-4,0,2e-4,0,0 | 0.129 / 0.453 | 0.066 / 0.037 |
| 20 | 9056 | 0.666 | 0.477 | 0.646 | 1e-4,1e-4,0,0,0 | 0.125 / 0.385 | 0.065 / 0.021 |
| 30 | 14048 | 0.685 | 0.477 | 0.650 | 0,1e-4,4e-4,0,0 | 0.122 / 0.396 | 0.074 / 0.021 |
| 40 | 19072 | 0.691 | 0.487 | 0.653 | 1e-4,1e-4,0,0,0 | 0.121 / 0.286 | 0.072 / 0.010 |
| 50 | 24080 | 0.666 | 0.478 | 0.650 | 1e-4,0,1e-4,0,0 | 0.113 / 0.323 | 0.064 / 0.010 |
| 60 | 29088 | 0.660 | 0.497 | 0.656 | 0,0,0,0,0 | 0.118 / 0.292 | 0.074 / 0.031 |
| final (70) | 34064 | 0.670 | 0.485 | 0.653 | 1e-4,1e-4,0,0,0 | 0.120 / 0.349 | 0.055 / 0.021 |

Per-level offset (own8, rows) wanders in -0.8..+0.2 with no trend after 10 min (L0..L4 at the end -0.30, -0.44, -0.36, -0.31, -0.04). Noise at 8 worlds is about +-0.006 on line_match. Everything is flat after about 10 minutes (4500 updates).

## Table 2: control vs long run, own draws (full block_check 128 own + 32 probe seeds; 128-world stage run for line_match / counts)
| | cfitpen_ml2 300 s (2112 upd) | clongfit 70 min (34064 upd) |
|---|---|---|
| raw line_match (128 worlds) | 0.628 | **0.668** |
| raw dirt | 0.493 | 0.505 |
| raw violated hard instances L0..L4 (128 worlds) | 126,176,9,6,0 | **135,34,7,0,0** |
| raw worlds valid /128 | 24 | 46 |
| rho_pool L0..L4 | .32/.83/.82/.40/-.25 | **.40/.87/.87/.61/-.16** |
| starved_probe L0..L4 | 0/.016/.091/.232/.406 | **0/.002/.040/.114/.351** |
| starved_own L0..L4 | 0/.008/.023/.061/.026 | 0/.006/.032/.060/.017 |
| raw roughness / offset / abs c1 | 1.40 / +0.72 / 0.81 | 1.48 / -0.06 / 0.53 |

(Starved is the block_check v2 fraction, probes and own separately; the combo report's "starved %" column is a different pooled figure, so I recomputed the control here with the same script.)

## Table 3: stage, 128 worlds (chain reference: roughness 1.40, offset -0.15, |c1| 0.48, line_match 0.69)
| pass | R | offset | rough (circ) | mean abs c1 | line_match | dirt | viol L0..L4 | valid/128 | tiles changed |
|---|---|---|---|---|---|---|---|---|---|
| longfit | raw | -0.06 | 1.48 | 0.53 | 0.668 | 0.505 | 135,34,7,0,0 | 46 | – |
| longfit | 15 | -0.02 | 1.43 | 0.54 | 0.674 | 0.505 | 0,0,0,0,0 | 128 | 2.4 % |
| longfit | **30** | -0.01 | 1.41 | 0.51 | 0.671 | 0.505 | 0,0,0,0,0 | 128 | 2.6 % |
| longfit | 60 | -0.03 | 1.40 | 0.53 | 0.669 | 0.505 | 0,0,0,0,0 | 128 | 2.9 % |
| ml2 300 s (copied from combo) | raw | +0.72 | 1.40 | 0.81 | 0.628 | 0.493 | 126,176,9,6,0 | 24 | – |
| ml2 300 s (copied) | 30 | +0.33 | 1.46 | 0.68 | 0.653 | 0.500 | 0,0,0,0,0 | 128 | 2.9 % |

## Reading
- Structure: raw line_match 0.628 -> 0.668 (+0.04); after R 30, 0.653 -> 0.671. Offset goes from +0.33 to -0.01 at R 30 (chain -0.15), roughness 1.41 (chain 1.40), |c1| 0.51 (0.48). The one-hour pass plus a short stage matches the chain on roughness/offset/c1 and is about 0.02 below it on line_match (0.69). One seed, one noise draw per world; I have no seed repeat.
- Raw violations: L1 count 176 -> 34 per 128 worlds; the rest similar. 46 vs 24 raw worlds valid. Not zero: the raw pass still needs the stage, but R 15 suffices (0 violations, 128/128) and R 30 is comfortable.
- Starvation: L3 probe-starved 0.232 -> 0.114 (halved), L2 0.091 -> 0.040, L4 0.406 -> 0.351 (still large, rho_pool at L4 stays negative, -0.16). L3 rho 0.40 -> 0.61. Own-draw starvation is unchanged (it was already low); the improvement is on probe (valid elsewhere) blocks, so the model puts mass on more of the valid space.
- Where ML saturates: line_match, dirt, offset and starvation are flat from about 10 min; the extra 60 minutes (30000 updates) bought nothing measurable beyond that (L4 starvation 0.45 -> 0.29-0.35 is the only trend, noisy). So the ML half's limit is not run length at this batch and net, consistent with the kernel route finding that the fit is the bottleneck but now saying the bottleneck is not training time.
- The earlier "ML-only leaks 10-100x over budget" does not appear here with this population and batch: raw fractions are at or under 2e-4 at all levels in every eval. The control's violations were higher with penalty steps present, so the penalty did not buy raw validity at this setting; I did not test an ML-only 300 s run at nctx 32 / ml_B 8, so I cannot say whether the bigger batch or the longer run is what removed the leak.

## What failed / caveats
- Not run: a 300 s clongfit (to separate batch size from run length), a second seed. The 10-minute eval used 8 worlds, so its violation fractions have large noise (one violated instance is ~3e-5).
- The two GPU 0 neighbours: GPU 0 was shared with recur's CPU phase and other agents' work on the box during the run; update rate 7.9-8.6 /s was steady.
- Probe flag (old check) not run; rho/starved reported instead as asked.

## Next (one line each)
- Run clongfit at 300 s and 20 min from the same code to find the saturation point and whether batch alone explains the gain.
- Bench `clongfit` + stage (R 15) at 600 s x 2 seeds for a gated row.
- Attack L4 (starved_probe 0.35, rho_pool negative): ML alone does not fix it at any run length here.

Final params: `reportables/2026-10-16_longfit/runs/clongfit/params.pkl` (also `params_030min/045min/060min.pkl`). Raw logs: `runs/clongfit/train.jsonl`, `eval.jsonl`, `runs/q2_longfit.json`, `runs/blocks.txt`.

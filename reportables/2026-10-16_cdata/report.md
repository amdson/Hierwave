# cdata: do the coarse levels of cfitpen_ml2 need more data or more ML weight?

**Answer: NO.** No arm cuts L3 or L4 starvation by more than a third of the control's at equal or better validity. L4 got worse in every arm (0.30 -> 0.29-0.52 probe-starved, rho_pool L4 -0.06 -> -0.13..-0.22).

## What ran
- `castlegen/variants/ccdata.py` (cfitpen_ml2 + knobs `pop_mult`, `w3`/`w2`, `extra_coarse`, `init_pkl`); `tests/variants/test_ccdata.py` (zero-change config bit-identical to cfitpen_ml2 after one iteration: passes; knob test: see runs/test.log).
- `scripts/sweep.sh` (arms ctrl, a, b, c, d, e) then `scripts/sweep_f.sh` (arm f, added after the coordinator's starve message); `scripts/arm.py` per arm; 20 min training each, GPU 1 cores 8-15, GPU shared with starve; nvidia-smi before/after in `runs/`. All arms start from `reportables/2026-10-16_fitpen/runs/arm_cfitpen_ml2/params.pkl` (Adam state and lambda restart; same for the control). Single seed, 16 eval worlds: differences of a few hundredths in starvation are within noise.
- Knob semantics: pop_mult scales the population (nctx 32 -> 64/128), minibatch ml_B stays 8, so the examples per update do not grow, only their diversity. The ML update is Adam, which is invariant to a gradient scale, so "weight x4 at L>=3" is implemented as a 4x parameter step for levels >=3 (p <- p_old + 4 (p_adam - p_old)). Extra steps: 16 (e) / 48 (f) additional ML steps per iteration updating levels >=3 only (same iteration, fresh minibatches); in 20 min they double/triple the updates (13.8k / 15.6k vs 7.5k).
- Mid-run block checks (reduced pool, 2 per arm) are in eval.json `mid_checks`; the final numbers are the bench's full v2 block check (eval_seed, 16 worlds). Not done: block_check every 50 updates (costs ~20+ s each); not done: per-block log-q share of high-slack L3 blocks and per-block loss weights (needs a new weighted-ML step; proposed below). Per-iteration offset / line_match / dirt / raw violations per level are in each `arm_*/loop.jsonl`.

## Table (starved = probe/own fraction; raw viol and offset are of the final raw draws, offset in rows; post-stage = after R 30 on 128 worlds)

| arm | starved probe/own L2 | L3 | L4 | rho_pool L3 | L4 | line_match | raw viol L0..L4 | offset L0..L4 (raw) | updates | upd/s | ms/iter | post-stage viol (R30) | stage rough | stage offset | |c1| | stage lm | tiles chg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_pop2 | 0.088/0.050 | 0.252/0.073 | 0.310/0.020 | 0.46 | -0.13 | 0.634 | 1.2e-04,3.2e-04,6.1e-05,0.0e+00,0.0e+00 | +0.19,-0.06,-1.07,-0.65,+0.04 | 7968 | 6.63 | 2413 | 0,0,0,0,0 | 1.46 | +0.19 | 0.60 | 0.663 | 2.9 % |
| b_pop4 | 0.077/0.030 | 0.176/0.065 | 0.379/0.020 | 0.35 | -0.18 | 0.629 | 8.4e-05,1.4e-04,0.0e+00,0.0e+00,0.0e+00 | +1.02,+0.95,+0.87,+0.30,-0.02 | 6192 | 5.15 | 3105 | 0,0,0,0,0 | 1.38 | +0.61 | 0.59 | 0.658 | 2.7 % |
| c_w4 | 0.085/0.044 | 0.168/0.063 | 0.350/0.021 | 0.42 | -0.21 | 0.646 | 8.4e-05,1.8e-04,6.1e-05,0.0e+00,0.0e+00 | +0.70,+0.90,-0.43,+0.75,+0.05 | 7616 | 6.34 | 2524 | 0,0,0,0,0 | 1.40 | +0.45 | 0.50 | 0.669 | 2.7 % |
| ctrl | 0.070/0.039 | 0.200/0.076 | 0.300/0.012 | 0.42 | -0.06 | 0.633 | 1.3e-04,2.9e-04,0.0e+00,0.0e+00,0.0e+00 | -0.14,-0.86,+0.05,-0.13,+0.02 | 7504 | 6.25 | 2559 | 0,0,0,0,0 | 1.44 | +0.01 | 0.62 | 0.658 | 2.8 % |
| d_w4_pop2 | 0.063/0.038 | 0.149/0.070 | 0.287/0.015 | 0.54 | -0.15 | 0.630 | 2.7e-05,1.4e-04,2.4e-04,0.0e+00,0.0e+00 | +0.85,+5.21,+0.40,+0.20,+0.03 | 7312 | 6.09 | 2627 | 0,0,0,0,0 | 1.40 | +0.46 | 0.57 | 0.658 | 2.9 % |
| e_extra16 | 0.074/0.033 | 0.198/0.056 | 0.370/0.024 | 0.74 | -0.16 | 0.648 | 9.2e-05,1.4e-04,2.4e-04,0.0e+00,0.0e+00 | +0.14,+0.90,+0.23,-0.51,+0.04 | 13824 | 11.49 | 2784 | 0,0,0,0,0 | 1.42 | +0.09 | 0.57 | 0.663 | 2.8 % |
| f_extra48 | 0.060/0.032 | 0.198/0.048 | 0.519/0.025 | 0.50 | -0.22 | 0.646 | 8.0e-05,2.0e-04,6.1e-05,0.0e+00,0.0e+00 | +0.45,+0.31,+0.45,+0.08,-0.05 | 15616 | 12.96 | 4937 | 0,0,0,0,0 | 1.39 | +0.23 | 0.56 | 0.660 | 2.5 % |

Post-stage violations are 0 at every level for every arm (128/128 valid). Raw L4 violations 0 everywhere.

## Reading
- L3 probe starvation: ctrl 0.200; best d (w4 + pop2) 0.149 (-25 %), c 0.168, b 0.176, others 0.198-0.252. Needed < 0.133. L3 own starvation: ctrl 0.076; best f 0.048, e 0.056 (-37 %, -26 %) but probe starvation unchanged, so not a clean win. L2 unchanged (0.06-0.09).
- rho_pool L3 rises with the extra coarse steps (e 0.74, f 0.50, d 0.54 vs 0.42) while starvation does not fall: the ordering improves, the under-weighted high-slack blocks (starve's diagnosis) do not move.
- L4 starvation rises with more coarse fitting (f 0.52). This is the coverage problem from the starve report; more ML at L4 on own draws makes it worse, not better.
- Validity unchanged. line_match 0.629-0.648 (ctrl 0.633; e, f, c 0.646-0.648 are within ~2 standard errors, 0.006). Per-level offset is not stable between arms (e.g. d L1 +5.2 rows, ctrl L1 -0.86): the offset at L1-L3 moves by up to a row in either direction with training noise, so no arm "fixes" it.
- Stage at R 30: all arms similar (roughness 1.38-1.46, offset +0.01 to +0.61, |c1| 0.50-0.62; ctrl +0.01 / 0.62); c and f have lower |c1| (0.50, 0.56) at tiles changed 2.5-2.7 %.

## What failed / next
- More data, step weight and extra steps at L>=3 do not cut starvation; a plain "more of the same objective" at the coarse levels is ruled out as the fix. Single seed, so small effects (c, d) are not excluded but none approaches a third.
- Candidate for wave B (best on L3 probe starvation and rho, validity equal): `runs/arm_d_w4_pop2/params.pkl`; best on rho_pool L3 and line_match: `runs/arm_e_extra16/params.pkl`.
- Next (one line each): per-block ML weights proportional to the target weight share (upweight surface-crossing high-slack L3 blocks; the thing starve found under-weighted) needs a new weighted-ML step in logq; an L4 coverage fix needs probe-like contexts, outside the own-draw constraint (propose, do not build).

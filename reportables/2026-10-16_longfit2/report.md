# longfit2: does clongfit's recipe reproduce at a bench-legal budget, and which knob carries it? (2026-10-16 overnight)

GPU 0 (shared with other agents' JAX jobs, 2 of 3 training slots taken; cores 16-23). nvidia-smi: before 1560 MHz / 60 C; after 1845-1860 MHz / 80-81 C, throttle reason 0x4 (power cap) only, never below 1500 MHz (`runs/nvidia_before.txt`, `runs/*/nvidia_after_s*.txt`).

**Answer.** No arm reproduces the longfit gain, and the batch is not what carries it. At 300 s and 600 s, clongfit (ml_B 32, nctx 64) and the small-batch control (ml_B 8, nctx 32) give the same raw structure (line_match 0.653-0.670) and the same raw violations (L3 1-6 per 128 worlds, L0-L2 in the 100s), with 11-31 of 128 raw worlds valid vs longfit's 46 and the 300 s ml2 control's 24. The small batch gets 4000-4100 updates in 600 s, the large one 2250-2450, and the small batch is not worse. What longfit had that these lack is length (34k updates, 70 min); but longfit's own curve says line_match was flat from 10 min, so the remaining difference is raw validity and L4/L3 starvation, which here only appear as slow trends. The gate (raw line_match >= 0.66 and 0 raw L3-L4 violations) was met by no arm and seed (L3 violations 1-6 everywhere), so **no bench was run**.

## What was run
- `scripts/sweep.sh 0 16-23` (launched once, DONE marker): compile warm-up of both shapes (45 s, uncounted), then per arm seeds 0,1: `scripts/train.py <variant> <budget> <seed>` (variants `clongfit`, `clongfit_small` from castlegen/variants/clongfit.py: rkl_n 0, upi 16; small = ml2's nctx 32, ml_B 8; budget includes setup), then stage2 `q2.py` (128 worlds, raw own draws and stage v1b R 15 F 150, same measures as longfit), then longfit's `blocks.py` (block_check, 128 own + 32 probe seeds).
- Params: `runs/<arm>/params_s<seed>.pkl`. Per-arm raw files: `q2_s*.json`, `blocks_s*.txt`, `train_s*.jsonl`, `updates_s*.json`. Table: `scripts/summ.py`.
- Arms: big300 = clongfit 300 s; big600 = clongfit 600 s; small600 = clongfit_small 600 s.
- Not seed-matched to longfit: seeds 0 and 1 here (longfit used seed 0, 4200 s).

## Table (128 worlds each; chain reference line_match 0.69 / roughness 1.40 / offset -0.15 / |c1| 0.48)
Columns: updates; raw = unrepaired own draws; offset/rough/|c1| are raw; R15 = after the fixed stage R 15; rho and starved are block_check v2 per level L0..L4.

| arm | seed | upd | raw lm | raw dirt | raw viol L0-4 | raw valid | offset | rough | |c1| | R15 lm | R15 viol | R15 valid | rho L0-4 | starved_probe L0-4 | starved_own L0-4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| big300 | 0 | 912 | 0.661 | 0.512 | 224,140,40,6,0 | 22 | -0.48 | 1.51 | 0.55 | 0.665 | 9,8,6,2,0 | 125 | 0.403/0.843/0.858/0.508/-0.162 | 0.000/0.004/0.044/0.143/0.310 | 0.000/0.007/0.034/0.083/0.019 |
| big300 | 1 | 928 | 0.653 | 0.506 | 296,191,67,6,0 | 11 | -0.14 | 1.52 | 0.61 | 0.660 | 11,9,6,2,0 | 122 | 0.399/0.854/0.852/0.478/-0.134 | 0.000/0.003/0.047/0.133/0.306 | 0.000/0.007/0.036/0.084/0.024 |
| big600 | 0 | 2256 | 0.665 | 0.507 | 174,91,35,4,0 | 31 | -0.18 | 1.43 | 0.56 | 0.670 | 13,11,5,2,0 | 123 | 0.386/0.852/0.854/0.447/-0.270 | 0.000/0.003/0.042/0.117/0.407 | 0.000/0.007/0.037/0.074/0.017 |
| big600 | 1 | 2448 | 0.667 | 0.505 | 165,111,27,1,0 | 31 | -0.02 | 1.50 | 0.53 | 0.673 | 4,4,2,0,0 | 126 | 0.407/0.859/0.872/0.504/-0.171 | 0.000/0.002/0.037/0.116/0.387 | 0.000/0.007/0.033/0.068/0.021 |
| small600 | 0 | 4000 | 0.663 | 0.506 | 179,108,33,3,0 | 21 | -0.13 | 1.44 | 0.59 | 0.668 | 3,1,2,0,0 | 127 | 0.402/0.859/0.872/0.424/-0.172 | 0.000/0.003/0.039/0.123/0.349 | 0.000/0.007/0.028/0.086/0.021 |
| small600 | 1 | 4144 | 0.670 | 0.503 | 236,145,35,4,0 | 15 | +0.05 | 1.58 | 0.47 | 0.678 | 9,6,5,2,0 | 125 | 0.394/0.855/0.851/0.370/-0.154 | 0.000/0.004/0.046/0.116/0.312 | 0.000/0.007/0.033/0.076/0.016 |

Reference rows (from longfit report): ml2 300 s control: raw lm 0.628, raw viol 126,176,9,6,0, 24 valid, offset +0.72, rough 1.40, c1 0.81, starved_probe .000/.016/.091/.232/.406; clongfit 70 min: raw lm 0.668, viol 135,34,7,0,0, 46 valid, offset -0.06, 1.48, 0.53, starved_probe .000/.002/.040/.114/.351; R 15: lm 0.674, 128/128 valid.

## Reading
- Raw line_match: big300 0.661/0.653, big600 0.665/0.667, small600 0.663/0.670 vs ml2 0.628 and clongfit-70min 0.668. The line_match gain over ml2 is already there at 300 s and is the same with the small batch: it comes from dropping the penalty steps (all of 300 s is ML), not from ml_B/nctx and not from length. Two seeds, noise between seeds about 0.005-0.01.
- Offset: ml2 +0.72 becomes -0.48..+0.05 in all arms (chain -0.15), also without the long run. Roughness 1.43-1.58 (chain 1.40), |c1| 0.47-0.61 (0.48). Seed spread in offset and roughness is larger than any arm difference.
- Starvation: starved_probe L3 0.143/0.133 (big300), 0.117/0.116 (big600), 0.123/0.116 (small600) vs 0.232 (ml2) and 0.114 (70 min): reaches the long-run value by 600 s regardless of batch; L4 0.31-0.41, no better than before (long run 0.35). rho_pool L3 0.37-0.51 (long run 0.61), L4 negative in every arm.
- Raw violations: L1 140/191 (big300), 91/111 (big600), 108/145 (small600) vs 34 in the 70 min run; raw valid 11-31 vs 46. Violations fall with length (big300 -> big600) and the 70-minute value is not reached. Batch size does not separate them (small600 4000 updates vs big600 2300 are the same within seed noise). This is the one place longfit's gain is not reproduced.
- Stage R 15: 122-127 of 128 worlds valid in these arms (residual violations 0-13 per level, L0/L1 mostly), vs 128/128 for the 70-min params. R 30 was used in the combo table; I did not rerun R 30 here. R 15 therefore needs the longer fit or a larger R at 300-600 s.
- Rate: clongfit does 3.0-3.1 updates/s on this contended GPU (longfit measured 8.2/s on an emptier box), clongfit_small 6.7-6.9/s.

## What failed
- No arm met the bench gate; no bench row and no `clongfit_stage` bench run. The variant `castlegen/variants/clongfit_stage.py` (clongfit + stage, R 15, `staged = True`) is registered and imports (checked), but has not been benched.
- Update counts differ by box load (GPU shared); arms are at equal wall clock, not equal updates.
- Only 2 seeds; spread in raw valid worlds (11 vs 22, 31 vs 31, 21 vs 15) is large.
- No 20-30 min arm, so I cannot say where raw violations reach the 70-min level.

## Next
- Cheapest bench candidate: `clongfit_small` + stage with R 30 at 600 s (4000 updates, best update rate, stage R 15 left 1-9 violations per level); needs a gate decision since L3 violations are 3-4 per 128 worlds raw.
- A 20 min clongfit_small run (2 seeds) to see whether raw L3/L4 violations reach 0 before 70 min.

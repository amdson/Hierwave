# stage2: the fixed stage made finite-support, warm vs random start, bench rows (2026-10-16)

Agent stage2 (wave 3), GPU 1 (shared with the Sonnet agent dgate; light jobs), cores 8-15. ideas #22, Amendment 2026-10-16.
Training is ckpop_hard_pen's (unchanged); only `generate` differs.

## Answers
1. **Finite support: PASSES** (bench check, mismatch 0 at every level, raw and final, both seeds, R 30 / F 150).
   Two fixes: (a) every Gumbel draw of the stage is keyed by (world seed, phase/round/sweep/channel or block class, level,
   Y, absolute X, value) through `castlegen.bench.noise`; (b) open windows are padded by the stage's declared reach.
   **The wave-2 reach estimate (2.6e5 fine columns) was ~100x too high.** The exact light cone of the schedule
   (boolean propagation over the factor-instance graph through the stage's exact update order) is **~64-66 fine
   columns per round** (rounds 1-4: 40/104/167/233 right, 36/70/101/121 left) and the certificate refill adds **≤ 10 in
   total, independent of F** (certificates couple only inside a top-level column; there is no horizontal certificate
   factor). Declared: 80 per round + 32 → **R 30 / F 150: 2432 fine columns**, smaller than the pass's own margin (2752).
   **Exact cone of the full R 30 / F 150 schedule: 1904 columns right, 688 left** (runs/reach_R30.log; right increments
   63-66 per round, the refill adds 8/10 and saturates), inside the declared 2432. So the stage is paddable at every R we care about; R 30 stays the smallest valid budget (wave-2: R 15 leaves 2/128).
2. **The learned pass is worth a lot to the stage** (table below). At equal budget, the pass start reaches the chain's
   statistics at R 240 (offset +0.36, roughness 1.40; chain −0.15 / 1.40) and is valid at every R; the random start
   (cmix §7 recipe, λ 20) is still invalid at R 240 (26/128 worlds with violations, roughness 2.98) and is 0/128 valid at
   R 30 and R 100; all-dirt is valid but still −5 rows off at R 240 (−17 at R 100). **Random at R 240 does not match the
   pass at R 100** (pass R 100: valid 128/128, offset +1.60, rough 1.64). Under this stage, cmix's "~100 sweeps" from
   random does not hold (that was a single-site chain with its own schedule, not this kernel mix).
   But: the stage is still a chain with a warm start (amendment wording): at R 30 it sits mid-relaxation (roughness 1.83 vs raw 1.05).
3. **Bench row `ckpop_hard_pen_stage` (R 30, F 150, 300 s × 2): final violations ≤ 1.5e-5 (worst level), support ok,
   seeds agree; not accepted only because probes are flagged** (as for every variant; see box rules on the probe
   check). Raw vs final is reported, not gated: roughness 1.09→2.17 and 1.33→1.90. Second row `ckpop_hard_pen_stage120` (R 120): 0 final violations, support ok, roughness 0.88/1.15 → 1.75/1.73, fill_err unchanged (0.029 → 0.030).

## Q1: what changed (castlegen/variants/cstage.py, castlegen/variants/_cstage_ck.py)
- `_cstage_ck.py`: copies of colour_gibbs' per-class step and of JaxBlockGibbs' sweep (fast path and one-chunk slow path for
  the 2x2 tile blocks) whose only difference is the noise source: `noise.gumbel/uniform(seed, tag, level, Y, X_abs, value)`,
  tag = `tag_of("cstage/<phase>/<round>/<level or class>/<channel>")`; a block's draw is keyed by its first variable's
  (level, Y, X) and the completion index. The kernels themselves are not edited.
- `Stage.run(st, seeds, R, F, x0)`: batched over worlds (per-world seeds); offset cycle r mod 4; certificate reset keyed
  by (seed, level, Y, X) as before; x0 must be a multiple of 64 (colours and block tilings are absolute).
- `generate`: periodic worlds are made 32 consecutive seeds at a time in one batched stage call and cached (deterministic
  in the seed, so results are unchanged); open windows: the pass on region ± (2752 + reach), the stage on region ± reach
  treated as periodic (the wrap seam is beyond the light cone), crop.
- Reach: `scripts/reach.py W NR` (runs/reach.log aligned seed; runs/reach2.log seed misaligned by one column, single steps + 4 rounds;
  runs/reach_R30.log the full 30 rounds + 3 refill sweeps on W 6144, CPU).

Smoke (`scripts/smoke.py`, hard_pen saved params, 128 periodic worlds W 128, then the bench's two-window check):

| R / F | pad (fine cols) | viol L0..L4 (pooled over the worlds run) | tiles changed | 32 worlds wall (pass + stage + host energy) | support mismatch |
|---|---|---|---|---|---|
| 4 / 10 | 384 | 67,94,62,28,2 (64 worlds) | – | 0.8 s | 0 |
| **30 / 150** | **2432** | **0,0,0,0,0** (128 worlds) | **3.3 %** | 1.6-1.7 s (≈ 50 ms/world) | **0** |

## Q2: warm vs random start at equal budget
`scripts/q2.py` (copy of wave-2 stage_q1.py + an all-dirt start), `scripts/run_q2.sh`, `runs/q2.json`, `runs/q2.log`.
128 worlds, W 128, hard_pen params, the stage of wave 2 (same kernels; jax.random noise — periodic, so irrelevant here),
F 150 random-start refill after every arm. Random = every sampled channel uniform (cmix §7: random tiles/rho, λ 20);
all-dirt = t = 1, every channel lifted (valid). Chain reference (cmix trace_hard late): offset −0.15, roughness 1.40, |c1| 0.48.

| start | R | offset | rough (circ) | mean abs c1 | line_match | dirt | viol L0..L4 | valid/128 | tiles changed vs pass |
|---|---|---|---|---|---|---|---|---|---|
| pass raw | 0 | +4.63 | 1.05 | 1.26 | 0.559 | 0.432 | 44,46,23,3,0 | 75 | – |
| pass | 30 | +3.35 | 1.83 | 0.96 | 0.579 | 0.453 | 0 | 128 | 3.4 % |
| pass | 100 | +1.60 | 1.64 | 0.58 | 0.627 | 0.480 | 0 | 128 | 6.1 % |
| pass | 240 | **+0.36** | **1.40** | 0.48 | 0.668 | 0.499 | 0 | 128 | 8.0 % |
| random | 30 | −8.45 | 16.85 | 2.18 | 0.505 | 0.494 | 15236,5731,2578,659,100 | 0 | |
| random | 100 | −1.31 | 12.88 | 1.70 | 0.514 | 0.492 | 4620,1790,533,68,1 | 0 | |
| random | 240 | +0.37 | 2.98 | 0.87 | 0.611 | 0.499 | 46,30,3,0,0 | 102 | |
| all-dirt | 30 | −24.62 | 1.81 | 1.86 | 0.500 | 0.890 | 0 | 128 | |
| all-dirt | 100 | −17.26 | 2.46 | 1.82 | 0.500 | 0.775 | 0 | 128 | |
| all-dirt | 240 | −4.99 | 2.76 | 1.13 | 0.567 | 0.583 | 0 | 128 | |

Stage seconds per 128 worlds (R 240): 6.7-7.0 s for every start. One noise seed per arm.

## Q3: bench rows
Command: `scripts/run_bench.sh [variant]` = `python -m castlegen.bench run <variant> ground --train 300 --seeds 2 --tag stage2`
(runs/bench_stage.log, runs/bench120.log); `reportables/2026-10-14_cr2_penalty/scripts/pool.py` (runs/pool_R30.txt, runs/pool_R120.txt; pool reads the last row, so the R 30 row was pooled from a one-line copy); `python -m castlegen.bench board`.
GPU 1 shared with dgate; nvidia-smi before/after in runs/nvidia_*_bench_*.txt.

| row | R/F | reach pad | updates (s0/s1) | final viol counts L0..L4 (pooled, 2x16 worlds) | worst final | raw worst | raw→final (seed 0; seed 1) | seeds agree | support | probes | rho L0..L4 | starved % L0..L4 | F | gen s/world (s0/s1) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ckpop_hard_pen_stage (wave 2, unkeyed) | 30/150 | 128 (wrapped) | 912/1168 | 0,0,0,0,0 | 0 | 7.3e-4 | rough 1.10→2.20; 1.33→1.91 | ok | FAIL 0.073 | flagged | | | 0.078 | 2.6/1.3 |
| **ckpop_hard_pen_stage** | 30/150 | 2432 | 904/1168 | **2,1,0,0,0** (L0 CI ≤ 1.4e-5, L1 ≤ 4.3e-5, L2 ≤ 1.1e-4) | 1.5e-5 | 7.3e-4 | rough 1.09→2.17; 1.33→1.90. lm 0.537→0.552; 0.573→0.590. dirt 0.396→0.423; 0.432→0.450. fill_err 0.028→0.040; 0.026→0.034. floating 4.0→0.3; 1.9→0 | ok | **ok (0)** | flagged | 0.29/0.79/0.76/0.40/−0.11 | 0/3/9/25/27 | 0.059 | 1.36/0.10 |
| **ckpop_hard_pen_stage120** | 120/150 | 9664 | 896/1152 | **0,0,0,0,0** (L0 CI ≤ 7.0e-6, L1 ≤ 2.8e-5, L2 ≤ 1.1e-4) | 0 | 4.6e-5 | rough 0.88→1.75; 1.15→1.73. lm 0.540→0.622; 0.562→0.641. dirt 0.416→0.484; 0.414→0.479. fill_err 0.029→0.030; 0.027→0.029 | ok | **ok (0)** | flagged | 0.28/0.79/0.67/0.39/−0.17 | 0/4/10/27/27 | 0.064 | 1.38/0.18 |

Both rows: not accepted, the only failing gate is probes (flagged for every variant on the board); final violations, seeds-agree
and finite support pass; raw vs final reported, not gated (staged). L3/L4 CIs straddle the budget for lack of instances
(as for every row). F is computed on raw draws (training), so it reflects the 896-1168 updates of a shared GPU, not the stage.
The bench evaluates 16 worlds per seed; roughness of the final draw (1.7-2.2) is still above the chain's 1.40 at both R
(R 120 is closer; Q2 says R 240 is needed). GPU 1 SM clock before/after: 1560/1875 MHz (R 30), 210 (idle)/210 MHz (R 120),
no thermal flag; dgate shared the GPU.

Stage cost: ≈ 50 ms/world for pass + R 30 stage batched 32 at a time (bench gen 0.10 s/world on seed 1 including wasted
batch members; 1.36 s on seed 0 including compiles). Fraction changed: 3.3 % of tiles (R 30); all certificate channels are
reset and refilled by design.

## Constitution
- Amendment 2026-10-16 / constraint 2: fixed R, F, offset cycle; noise keyed by absolute coordinates; declared reach
  R × 80 + 32 fine columns, padded on open windows; the bench support check passes with the stage included.
- Constraint 1: every kernel is derived from the declared factors (colour Gibbs, exact block Gibbs); the reach is computed
  from the factor-instance graph, not chosen per scenario. The tight (lift) refill is not used (constraint 6).
- Constraints 3/4: training untouched; the stage contributes no gradient.
- Constraint 7: λ 20 in the stage.

## What failed
- Bench rows not accepted: probes flagged (every variant flags; the redesigned probe check is pending per box rules).
- At R 30 the stage still moves roughness from ~1.1-1.3 to ~1.9-2.2 (mid-relaxation, as wave 2 found); it reaches the chain's
  roughness only at R 240.
- First attempt at keyed noise mismatched (500 cells) because x0 was baked into a cached jit; fixed (x0 traced).

## Next (one line each)
- Fix the pass's +4.6-row offset at the source (offset agent's report): a better start would shorten the stage, since at equal budget the start dominates.
- Compute the exact cone for R 120 too (only R 30 was run in full; R 120's 9664 rests on the ≤ 66/round increments).
- Gate the stage's own relaxation by its autocorrelation: R 240 ≈ the slow-mode tail, so the stage is "a chain with a warm start" by the amendment's wording.

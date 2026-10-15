# kblock: block-Gibbs sweeps in ckpop_hard_pen's population refresh (2026-10-15)

**Answer: no.** At 120 s (seed 0, equal updates) adding exact block sweeps to the population refresh does not raise population or own-draw line_match beyond noise, and a 4x dose gives nothing. No bench was run (the brief's stop rule).

## Question
Does adding p5's block kernels (parent+4-children at level 1, and 2x2 tile blocks) to `ckpop_hard_pen`'s population refresh lift own-draw line_match above 0.59 at the same validity?

## Premise check (read before the numbers)
`ckpop_hard_pen` already refreshes with block kernels. Each iteration, its phase-2 `refine` (inherited from `ckernel.py` and `ckernel_own.py`) runs:
- coarse block Gibbs at L4 and L3;
- tempered transitions at L4 and L3;
- 4 L2 multiple-try moves;
- **3 x (J20, J11)**, where J11 = `JaxBlockGibbs(level 1, depth 1)`, p5's parent+children kernel, at λ = 20.

After `refine` come 20 colour-Gibbs sweeps at λ = 20 and then the 2 hard single-site sweeps. So the hypothesis "the refresh is only single-site hard sweeps" does not hold. The one kernel that was missing is the 2x2 tile block (F2).

## What was built
`castlegen/variants/ckblock.py`:
- `ckpop_hard_pen_kb` appends `kb_rounds` rounds of (optional single-site sweep, J11, F2) to the end of `refine`, before the colour-Gibbs and hard sweeps.
- The added kernels are hard (`kb_lam=None`: violating completions excluded).
- Variants: `_kb` (1 round), `_kb4` (4 rounds), `_kb0` (0 rounds).
- Identity check (`scripts/ident.py`): after one iteration, `_kb0` matches `ckpop_hard_pen` bit for bit (all 90 parameter leaves and the population).
- p*-invariance of J11 and F2 is tested in the existing kernel tests and was not re-proved here.

Constitution status: these are training-only kernels under the amendment of 2026-10-14 (exact block Gibbs on small blocks; hard violations treated as infeasible). Generation is unchanged. The objective is weighted ML plus the penalty-only term, as in ckpop_hard_pen.

## Runs
- Command: `scripts/arm.sh <variant> 0 120 fast`. This is a copy of the cr2_penalty instrumented arm: warm-up instance, 120 s of training, then the bench's `eval_seed` on 16 held-out worlds. Nothing was written to `results/bench.jsonl`.
- Runs were back to back on GPU 0, which is shared (GPU 1 was busy, at 83 C). GPU 0's SM clock was 1560 MHz before each run. The nvidia-smi snapshots are in `runs/fast_*/nvsmi_*.txt`.
- Summaries come from `scripts/loopsum.py`.

| arm | updates (ML / pen) | kb cost per refresh | tiles changed by kb per refresh | pop line_match (upd 400+) | pop valid | own8 line_match (upd 400+) | **eval line_match** | eval F | eval final violations L0/1/2/3/4 (×1e-4) |
|---|---|---|---|---|---|---|---|---|---|
| ckpop_hard_pen | 480 / 1921 | – (refine 959 ms) | – | 0.638 | 0.994 | 0.554 | **0.546** | +0.123 | 0.31 / 0.76 / 1.8 / 7.3 / 39 |
| kb (1 round) | 472 / 1889 | 9 ms | 1.1 % | 0.649 | 0.990 | 0.584 | **0.565** | +0.069 | 0.57 / 1.4 / **6.7** / 7.3 / 20 |
| kb4 (4 rounds) | 464 / 1857 | 23 ms | 1.5 % | 0.642 | 0.996 | 0.556 | **0.545** | +0.096 | 0.31 / 0.76 / 0.61 / 2.4 / 39 |

- Population line_match is the same within 0.01 in all three arms, at every stage of training (0.55 at updates 0-200, 0.63-0.64 at 200-400).
- The block kernels are cheap (1-2 % of refresh time) and move 1-1.5 % of tiles per refresh, but this does not change what the ML step sees.
- The +0.02 eval line_match of the 1-round arm is one seed, and it disappears at 4 rounds. For scale, the 4 seeds of ckpop_hard_pen at 300 s spread 0.56-0.59.
- Validity varies seed-like across arms: 1 round is worse at L2, 4 rounds better. None of the 120 s runs is within budget at L3/L4 yet. That is expected at 120 s and is not a kb effect.

## What failed
- The hypothesis's premise: the refresh already contains the parent+children block kernel three times per iteration.
- Adding the 2x2 tile block, even 4x per iteration, does not raise population or own-draw line_match.
- A first warm-up recompiled F2 for each of its 4 offsets inside the timed loop (2.8-4.3 s per iteration). This is fixed in `ckblock.py`: all offsets are compiled once before the clock.

## Interpretation
The population already has more structure than the pass (0.64 vs 0.55). As cr2_penalty §3 found, the gap is in how the pass fits its population under λ_1..3 = 20 penalty pressure, not in the targets. Stronger targets cannot close a gap that lies in the fit.

## Next (one line each)
- Do not pursue a stronger refresh. The lever is the fit/pressure side, e.g. the penalty on fewer steps once the population is valid, or the ML weight versus the penalty steps.
- If the refresh is revisited: p10's 60-round composition per iteration costs about 30x the refine time and is only worth trying if population line_match is the bottleneck, which it is not.

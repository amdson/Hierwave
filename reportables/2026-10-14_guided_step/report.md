# Guided revising step for a frozen model (ideas #19-#21 direction; started as #20 (a) jitter repair)

**Status: PAUSED 2026-10-14 ~20:40 UTC on the coordinator's pause order (user usage).** Built and tested; the
16-world measurements are running detached (section 4). Nothing was benched; no training was run.

History: this task started as ideas #20 (a) (jitter repair, folder `2026-10-14_jitter_repair`, module
`_jitter_repair.py`). The user dropped the repair line (20:25); the jitter repair is kept only as an offline
baseline (`castlegen/variants/_jitter_repair.py`, not wired into any variant; `cjrepair.py` was never created).

## 1. What is built

- `castlegen/variants/_guided_step.py` (generic, reads only factor declarations + the Layout):
  - `HardMaps(lay)`: exact per-instance violation indicator maps of every hard factor in JAX (mirrors
    `bench/energy.py`: periodic / window, pads, within, by_pos, pad_a edge rows). `per_level()` equals
    energy.py's `E.viol` (tested on toy_support, ground, columns).
  - `violation_field(hm, state) -> {level: (h, w) int}`: violated hard instances touching each cell (the binary
    stand-in for #21's graded field; same shape and zero set, so `castlegen/graded.py` can replace it).
  - `select_cells(lay, field, dilate=1, descend=True)`: cells with nonzero field, dilated by 1 cell at their level,
    plus all descendants of selected coarse cells (so redrawn parents get their children redrawn after them).
  - `guided_step(lay, params, state, field, seed, step_index, tau=1, dilate=1, descend=True)
     -> (new_state, log_q_step, cells_changed)`: erase the selected sampled cells, then run q's own coloured pass
    (levels T..0, 16 colours, channels in order; least-violating mask with lookahead, logits -soft/tau - b_theta,
    Gumbel-max) writing only selected cells; the rest is teacher-forced context. Noise keyed by
    (seed, "guided{step}:{channel}", level, Y, absolute X). Dense: every site is computed, so the step has a
    fixed cost whatever the field is (no cap needed). `log_q_step` = sum of log q of the redrawn values.
  - Variant B slot: if an actor's input width equals the pass features + 2, the step appends
    `field_feats` = log(1 + field) at the site and at its parent. Frozen models never see it; it is for the
    #19 agent's path-KL training (the #19 agent can call `guided_step` directly; `seed` is the world seed int).
  - `guided_revise(lay, params, state, seed, R, record=False)`: R steps; info = log_q, cells changed and
    violations per level before each step.
  - Radius: `margin_step` = sum_l (16 x 3 + 2 dilate + 2) << l fine cells per step (one restricted coloured
    pass + the field's 1-cell reach + the dilation); `guided_margin(problem, R)` = R x that, rounded to P << T.
    Ground (T = 4): 1612 -> 1664 fine cells per step (cpass's own margin is 2752).
- `castlegen/variants/cguided.py`: `ckpop_hard_pen_gs2/_gs4`, `ckpop_hard_rkl_gs2/_gs4`: the base variants with
  generate() = pass -> R guided steps -> cpass one-round repair (cfg gs_R, gs_dilate, gs_descend, gs_repair);
  training identical to the base, so saved parameters evaluate without retraining. Not benched.
- `castlegen/variants/_jitter_repair.py` (baseline only): after cpass's repair, R rounds; per round, levels T..1,
  16 colour groups: each level-L cell whose cell-plus-children block touches a violated instance is erased and
  redrawn from the designed conditional with lookahead (coordinate-keyed noise); accepted only if the violated
  instances touching the block (exact block window sums) do not increase. Total violations never increase.
- `tests/variants/test_guided_step.py`: 7 tests, all pass (61 s, CPU): maps == energy.py (3 problems); the step is
  deterministic in (seed, coordinates), changes only selected cells, counts changes exactly; a valid state is a
  fixed point (0 cells changed); finite support: two overlapping open windows (toy_support, x0 = 0 and 16,
  padded by margin + guided_margin) agree exactly after pass + 1 step; jitter repair never increases violations
  and is deterministic.

## 2. Baseline numbers so far (CPU smoke, saved `ckpop_hard_pen` seed-0 params, 2 held-out worlds, W = 128)

`runs/smoke_pen_cpu.log`. Violated instances summed over the 2 worlds, L0..L4; "changed" = cells differing
from the raw draw per world; stats are means over the 2 worlds (raw: line_match 0.595, dirt 0.407, rough 1.43).

| state | viol L0..L4 | graded mass L0..L4 | changed/world L0..L4 | valid worlds | line_match / dirt / rough |
|---|---|---|---|---|---|
| raw | 0/1/1/0/0 | 0/1/3/0/0 | 0 | 0/2 | .595 / .407 / 1.43 |
| cpass repair (1 round) | 0/1/0/0/0 | 0/1/0/0/0 | 0/0.5/0.5/0/0 | 1/2 | .595 / .407 / 1.43 |
| jitter repair R = 1 (= R = 2) | 0/0/0/0/0 | 0 | 1/2/0.5/0/0 | 2/2 | .595 / .407 / 1.43 |
| guided steps R = 1 (= R 2..8; field is 0 after step 1) | 0/0/0/0/0 | 0 | 22.5/31.5/11/0/0 | 2/2 | .592 / .406 / 1.39 |

Reading (2 worlds only, not evidence yet): one guided step from q's own conditional cleared both raw violations
(including the L1 one the cpass repair cannot fix) but rewrote 65 cells per world against the jitter repair's 3.5;
steps 2-8 did nothing because the field was already zero. Times per world on CPU (cores 4-7, after compile):
pass 0.037 s, cpass repair 0.007 s, jitter R = 1 0.023 s, R = 2 0.041 s, 8 guided steps 0.30 s (0.038 s per step,
about one pass each). Compile: pass 3 s, jitter R = 2 66 s, 8 guided steps 69 s.

## 3. What is running (detached; read when done)

- CPU, cores 0-7: `scripts/run_offline.sh 0-7 ckpop_hard_pen <cr2 pen_s0 params.pkl> pen 0 16 8 runs/pen_s0_cpu.json`,
  log `runs/pen_s0_cpu.log` (16 held-out worlds of training seed 0, R up to 8). The final table in the log (and
  the json) has, per state (raw, repair, jitter1/2, guided1..8, guided1..8+repair): violations and graded mass per
  level, cells changed per world per level, valid worlds, stats; the last line has time per world and cells
  changed per step. PID in `runs/pids.txt`.
- GPU 1, gated: `scripts/gpu1_queue.sh` (log `runs/gpu1_queue.log`, prints the gate status every 60 s) waits until
  the coordinator's queue is gone, `gpu1_followup.nohup` has FOLLOWUP_DONE and GPU-1 utilisation is 0 %, then runs
  one process at a time on cores 8-15: pen_s0 (cr2 arm), rkl_w2b (wave4 W2b `ckpop_hard_rkl` 300 s), rkl123x2_C
  (wave6 C), hard_w1 (wave4 W1 `ckpop_hard`): `runs/<name>.log` / `.json`, nvidia-smi before/after in runs/.
  Prints ALL_DONE at the end; timeout 4 h.
- Reading the result: the question is whether `guided{R}` reaches 0 violations at every level on 16 worlds, at
  which R, with how many cells changed per world (compare `jitter1` and `repair`), and whether the raw-vs-revised
  stats move by more than the bench tolerances (line_match 0.03, dirt 0.03, roughness 0.15, floating 0.5,
  fill_err 0.01).

## 4. What is next
- Read the four json/log results and fill the before/after table per model (R = 0, 1, 2, 4, 8; +repair).
- If one step clears most violations but rewrites many cells: shrink the selection (dilate 0, descend only the
  children of changed parents) and measure distortion again; the field is the lever.
- Hand `guided_step` to the #19 agent (variant B: retrain with the 2 field features; the step's log_q is the
  path-KL proposal term).
- Replace the binary field by `castlegen/graded.py`'s distances (same interface) and test whether graded guidance
  selects fewer cells for the same clearance.
- A bench of `ckpop_hard_pen_gs2` only if the offline numbers justify it (user decision; none run here).

## Files
castlegen/variants/_guided_step.py, cguided.py, _jitter_repair.py (baseline); tests/variants/test_guided_step.py;
reportables/2026-10-14_guided_step/{report.md, scripts/ (env.sh, smoke_toy.py, offline_eval.py, run_offline.sh,
run_tests.sh, gpu1_queue.sh), runs/ (tests.log, smoke_pen_cpu.log/.json, pen_s0_cpu.log, gpu1_queue.log, pids.txt)}.

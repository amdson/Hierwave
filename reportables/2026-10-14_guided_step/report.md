# Guided revising step for a frozen model q (user direction for ideas #19-#21; started as #20 (a))

Agent jitter-repair / guided-step, 2026-10-14 evening. Problem: ground. **No training, no bench.** Four saved
own-draws models evaluated offline on 16 held-out worlds each (the bench's eval seeds of training seed 0,
periodic W = 128). Compute: tests and the smoke on CPU (cores 0-7); the four 16-world evaluations on GPU 1
(A4000, SM 1560 MHz at the end, no throttling), launched by a gate script only after the coordinator's queue had
printed FOLLOWUP_DONE and GPU-1 utilisation was 0 %; each took under 4 minutes.

History: the task started as ideas #20 (a) (jitter repair; folder `2026-10-14_jitter_repair`). The user dropped
the repair line at 20:25; the jitter repair is kept only as an offline baseline (`castlegen/variants/_jitter_repair.py`,
wired into no variant; `cjrepair.py` was never created). Then paused 20:40-21:10 (usage) with the evaluations detached.

## 1. Result in one paragraph

**Repeated guided steps from q's own conditional (variant A) do not reach zero violations.** Each step erases
every cell touching a violated hard instance plus its 1-cell neighbourhood and descendants and redraws them from
q (the coloured pass with lookahead, teacher-forced context). The redraw re-creates violations at q's own rate, so
the violated-instance count fluctuates instead of falling: pen_s0 11 (raw) -> 14 / 20 / 10 / 9 after 1 / 2 / 4 / 8
steps (with a new L3 violation at step 8), rkl_w2b 47 -> 41 / 30 / 30 / 15, C 81 -> 62 / 36 / 17 / 28,
W1 113 -> 81 / 56 / 26 / 20. It touches 15-320 cells per world per step (30-370 per world in total). The
accept-if-not-worse baselines are monotone and touch 1-25 cells per world: cpass's one-round repair 8 / 36 / 44 / 66,
the jitter repair (R = 2) 6 / 17 / 17 / 32. Distortion: all six statistics agree raw vs revised under the bench's
rule (3 SE + tolerance) for every model and R, and line_match / dirt / roughness / fill_err move by at most
0.006 / 0.007 / 0.08 / 0.0002; floating moves most (W1 11.4 -> 4.1) because the revision removes support violations
(so does any repair). Cost: one guided step = one pass, 5.4 ms per world on the A4000 (pass 5.5 ms, cpass repair
1.9 ms, jitter R = 1 8 ms). On the bench's per-level budget (52 / 13 / 3 / 0 / 0 instances in 16 worlds) the pen_s0
model passes with cpass's repair and with the jitter repair, and fails after 2 or 8 guided steps (L2 = 4; one L3
instance). **So: no, with the actor as trained, more guided steps do not converge; the field as an actor input
(variant B) needs training before it can mean anything, which is the #19 agent's path-KL route.**

## 2. What is built

- `castlegen/variants/_guided_step.py` (generic: factor declarations + the Layout only; jit/vmap-able):
  - `HardMaps(lay)`: exact per-instance violation indicator maps of every hard factor in JAX (mirrors
    `bench/energy.py`: periodic / window, pads, within, by_pos, pad_a edge rows); `per_level()` equals energy.py's
    `E.viol` (tested on toy_support, ground, columns).
  - `violation_field(hm, state) -> {level: (h, w) int}`: violated hard instances touching each cell. The binary
    stand-in for #21's graded field: same shape and zero set, so `castlegen/graded.py` distances plug into the same
    `field` argument.
  - `select_cells(lay, field, dilate=1, descend=True)`: cells with a nonzero field, dilated by `dilate` cells at their
    level (x wraps on the strip, is cut on a window), plus every descendant of a selected coarse cell.
  - `guided_step(lay, params, state, field, seed, step_index, tau=1, dilate=1, descend=True)
    -> (new_state, log_q_step, cells_changed)`: erase the selected sampled cells; run q's coloured pass (levels T..0,
    16 colours, channels in declared order; least-violating mask with lookahead; logits -soft/tau - b_theta(features);
    Gumbel-max) writing only selected cells, everything else teacher-forced as known context. Noise keyed by
    (seed, "guided{step}:{channel}", level, Y, absolute X). Dense: every site is computed and the mask decides what is
    written, so the cost is fixed whatever the field (no cap needed). `log_q_step` = sum of log q of the redrawn values.
  - Variant B slot: if an actor's input width equals the pass features + 2, `field_feats` = log(1 + field) at the site
    and at its parent is appended. Frozen models never see it.
  - `guided_revise(lay, params, state, seed, R, record=False)`: R steps; info = log_q, cells changed, violations per
    level before each step (and the states, with record).
  - Dependency radius: `margin_step` = sum_l (16 x 3 + 2 dilate + 2) << l fine cells per step (one restricted coloured
    pass, the field's 1-cell reach, the dilation); `guided_margin(problem, R)` = R x that rounded to P << T: ground
    1664 per step (cpass's own margin is 2752). Hand-off notes for the crevise agent are in the module docstring.
- `castlegen/variants/cguided.py`: `ckpop_hard_pen_gs2/_gs4`, `ckpop_hard_rkl_gs2/_gs4`: base variant + R guided
  steps + cpass repair in generate(); training identical to the base (unbenched; see section 5).
- `castlegen/variants/_jitter_repair.py` (baseline): after cpass's repair, R rounds; per round, levels T..1 and 16
  colour groups: each level-L cell whose cell-plus-children block touches a violated instance is erased and redrawn
  from the designed conditional with lookahead (coordinate-keyed noise), kept only if the violated instances touching
  the block (exact window sums over the maps; groups are instance-disjoint) do not increase. Total violations never
  increase. Radius: 16 x 3 cells per level per round (declared in `jitter_margin`).
- `tests/variants/test_guided_step.py`: 7 tests, all pass (61 s, CPU): maps == energy.py on three problems; the step is
  deterministic in (seed, coordinates), changes only selected cells and counts changes exactly; a valid state is a
  fixed point; finite support: two overlapping open windows (toy_support, x0 = 0 and 16, padded by
  margin + guided_margin) agree exactly after pass + 1 step; the jitter repair never increases violations and is
  deterministic.

## 3. Offline tables (16 held-out worlds per model; `scripts/offline_eval.py`, tables by `scripts/tables.py`)

Columns: violated hard instances per level summed over the 16 worlds (bench/energy.py); "budget" = every level within
the bench's 2e-4 (at most 52 / 13 / 3 / 0 / 0); graded mass per level (castlegen/graded.py `level_mass`); cells
changed from the raw draw per world (all levels); worlds with zero violations; stats and the bench's raw-vs-final
rule (|diff| <= 3 SE + tol; tol line_match 0.03, dirt 0.03, roughness 0.15, floating 0.5, fill_err 0.01).
`guidedR` = R guided steps from the raw draw; `+repair` = followed by cpass's one-round repair. Graded mass equals
the binary count at L0 (one floating tile = distance 1) and exceeds it where a certificate is several columns off.

**`ckpop_hard_pen` cr2 arm seed 0 (bench 3/4 seeds)** (`runs/pen_s0.json`; lam = [4.3, 20.0, 20.0, 20.0, 0.0])

| state | violated L0/L1/L2/L3/L4 (16 worlds) | total | budget | graded mass L0..L4 | cells changed / world | valid worlds | stats vs raw (bench rule) |
|---|---|---|---|---|---|---|---|
| raw | 3/4/4/0/0 | 11 | fail | 3/5/9/0/0 | 0.0 | 11/16 | lm 0.552 dirt 0.434 rough 0.94 float 1.8 (all agree) |
| repair | 3/2/3/0/0 | 8 | pass | 3/3/6/0/0 | 0.6 | 13/16 | lm 0.552 dirt 0.434 rough 0.94 float 1.8 (all agree) |
| jitter1 | 2/1/3/0/0 | 6 | pass | 2/2/5/0/0 | 2.9 | 15/16 | lm 0.553 dirt 0.434 rough 0.94 float 1.5 (all agree) |
| jitter2 | 2/1/3/0/0 | 6 | pass | 2/2/5/0/0 | 3.0 | 15/16 | lm 0.553 dirt 0.434 rough 0.94 float 1.5 (all agree) |
| guided1 | 4/8/2/0/0 | 14 | pass | 4/10/6/0/0 | 30.4 | 13/16 | lm 0.553 dirt 0.435 rough 0.93 float 0.9 (all agree) |
| guided2 | 6/10/4/0/0 | 20 | fail | 6/15/11/0/0 | 37.1 | 15/16 | lm 0.554 dirt 0.435 rough 0.94 float 1.9 (all agree) |
| guided4 | 2/6/2/0/0 | 10 | pass | 2/7/4/0/0 | 34.1 | 15/16 | lm 0.552 dirt 0.435 rough 0.94 float 0.6 (all agree) |
| guided8 | 4/2/2/1/0 | 9 | fail | 4/4/8/8/0 | 32.6 | 15/16 | lm 0.552 dirt 0.434 rough 0.94 float 2.0 (all agree) |
| guided4+repair | 1/1/4/0/0 | 6 | fail | 1/2/6/0/0 | 33.9 | 15/16 | lm 0.552 dirt 0.435 rough 0.94 float 0.5 (all agree) |
| guided8+repair | 4/2/2/1/0 | 9 | fail | 4/4/6/8/0 | 32.6 | 15/16 | lm 0.552 dirt 0.434 rough 0.94 float 2.0 (all agree) |

cells changed per guided step (mean over worlds): [30.4, 15.6, 25.9, 18.7, 17.2, 22.7, 16.9, 19.2]; time per world (GPU 1 unless noted): pass 5.6 ms, repair 1.9 ms, jitter R=1 8.0 ms, R=2 14.1 ms, 8 guided steps 44.0 ms (5.5 ms per step)

**`ckpop_hard_rkl` wave4 W2b, 300 s** (`runs/rkl_w2b.json`; lam = [20.0, 20.0, 20.0, 20.0, 0.0])

| state | violated L0/L1/L2/L3/L4 (16 worlds) | total | budget | graded mass L0..L4 | cells changed / world | valid worlds | stats vs raw (bench rule) |
|---|---|---|---|---|---|---|---|
| raw | 29/10/6/2/0 | 47 | fail | 30/13/19/7/0 | 0.0 | 2/16 | lm 0.599 dirt 0.495 rough 1.60 float 6.2 (all agree) |
| repair | 23/6/7/0/0 | 36 | fail | 23/10/23/0/0 | 1.3 | 2/16 | lm 0.599 dirt 0.495 rough 1.60 float 5.7 (all agree) |
| jitter1 | 13/2/6/0/0 | 21 | fail | 13/4/23/0/0 | 15.9 | 5/16 | lm 0.599 dirt 0.495 rough 1.62 float 4.1 (all agree) |
| jitter2 | 9/2/6/0/0 | 17 | fail | 9/4/25/0/0 | 19.1 | 7/16 | lm 0.599 dirt 0.495 rough 1.62 float 3.4 (all agree) |
| guided1 | 22/10/7/2/0 | 41 | fail | 22/17/25/15/0 | 148.6 | 4/16 | lm 0.600 dirt 0.496 rough 1.60 float 4.9 (all agree) |
| guided2 | 18/5/4/3/0 | 30 | fail | 18/7/14/23/0 | 164.3 | 4/16 | lm 0.601 dirt 0.496 rough 1.63 float 4.2 (all agree) |
| guided4 | 18/3/7/2/0 | 30 | fail | 18/6/21/16/0 | 175.0 | 5/16 | lm 0.601 dirt 0.496 rough 1.63 float 3.8 (all agree) |
| guided8 | 12/3/0/0/0 | 15 | pass | 12/5/0/0/0 | 193.4 | 7/16 | lm 0.600 dirt 0.495 rough 1.66 float 2.7 (all agree) |
| guided4+repair | 15/3/2/2/0 | 22 | fail | 15/4/6/10/0 | 174.8 | 5/16 | lm 0.601 dirt 0.496 rough 1.63 float 3.5 (all agree) |
| guided8+repair | 11/1/2/0/0 | 14 | pass | 11/2/3/0/0 | 193.6 | 7/16 | lm 0.600 dirt 0.495 rough 1.66 float 2.2 (all agree) |

cells changed per guided step (mean over worlds): [148.6, 59.6, 54.3, 38.2, 43.3, 55.8, 51.7, 7.3]; time per world (GPU 1 unless noted): pass 5.4 ms, repair 1.9 ms, jitter R=1 7.8 ms, R=2 13.7 ms, 8 guided steps 42.8 ms (5.3 ms per step)

**`ckpop_rkl123x2` wave6 C** (`runs/rkl123x2_C.json`; lam = [19.6, 20.0, 20.0, 20.0, 0.0])

| state | violated L0/L1/L2/L3/L4 (16 worlds) | total | budget | graded mass L0..L4 | cells changed / world | valid worlds | stats vs raw (bench rule) |
|---|---|---|---|---|---|---|---|
| raw | 23/37/18/3/0 | 81 | fail | 23/46/38/12/0 | 0.0 | 3/16 | lm 0.571 dirt 0.507 rough 0.71 float 4.2 (all agree) |
| repair | 15/15/9/5/0 | 44 | fail | 15/18/25/18/0 | 3.4 | 6/16 | lm 0.571 dirt 0.507 rough 0.71 float 4.1 (all agree) |
| jitter1 | 6/4/5/3/0 | 18 | fail | 6/6/15/15/0 | 22.4 | 8/16 | lm 0.570 dirt 0.507 rough 0.70 float 1.2 (all agree) |
| jitter2 | 5/4/5/3/0 | 17 | fail | 5/5/16/15/0 | 23.4 | 9/16 | lm 0.570 dirt 0.507 rough 0.71 float 1.1 (all agree) |
| guided1 | 21/29/9/3/0 | 62 | fail | 21/45/23/13/0 | 158.4 | 6/16 | lm 0.572 dirt 0.506 rough 0.73 float 4.9 (all agree) |
| guided2 | 14/16/4/2/0 | 36 | fail | 14/22/5/2/0 | 180.1 | 10/16 | lm 0.575 dirt 0.507 rough 0.70 float 2.2 (all agree) |
| guided4 | 5/3/7/2/0 | 17 | fail | 5/4/20/3/0 | 188.3 | 12/16 | lm 0.573 dirt 0.508 rough 0.69 float 2.0 (all agree) |
| guided8 | 11/10/5/2/0 | 28 | fail | 11/16/12/6/0 | 190.6 | 12/16 | lm 0.573 dirt 0.507 rough 0.69 float 7.6 (all agree) |
| guided4+repair | 5/1/5/3/0 | 14 | fail | 5/1/14/6/0 | 188.4 | 12/16 | lm 0.573 dirt 0.508 rough 0.69 float 2.0 (all agree) |
| guided8+repair | 10/6/5/1/0 | 22 | fail | 10/10/14/5/0 | 190.4 | 12/16 | lm 0.573 dirt 0.507 rough 0.69 float 7.6 (all agree) |

cells changed per guided step (mean over worlds): [158.4, 97.8, 46.8, 44.4, 36.1, 29.1, 26.6, 28.1]; time per world (GPU 1 unless noted): pass 5.5 ms, repair 1.9 ms, jitter R=1 7.7 ms, R=2 13.7 ms, 8 guided steps 43.5 ms (5.4 ms per step)

**`ckpop_hard` wave4 W1 (no violation term)** (`runs/hard_w1.json`; lam = [0.0, 6.3, 6.1, 0.0, 0.0])

| state | violated L0/L1/L2/L3/L4 (16 worlds) | total | budget | graded mass L0..L4 | cells changed / world | valid worlds | stats vs raw (bench rule) |
|---|---|---|---|---|---|---|---|
| raw | 52/46/12/3/0 | 113 | fail | 52/58/23/20/0 | 0.0 | 0/16 | lm 0.628 dirt 0.513 rough 1.86 float 11.4 (all agree) |
| repair | 38/18/5/5/0 | 66 | fail | 38/26/10/21/0 | 4.6 | 2/16 | lm 0.628 dirt 0.513 rough 1.85 float 8.8 (all agree) |
| jitter1 | 18/13/3/5/0 | 39 | fail | 18/20/10/21/0 | 31.3 | 4/16 | lm 0.629 dirt 0.513 rough 1.87 float 5.1 (all agree) |
| jitter2 | 13/12/3/4/0 | 32 | fail | 13/17/10/15/0 | 35.8 | 6/16 | lm 0.629 dirt 0.513 rough 1.87 float 4.2 (all agree) |
| guided1 | 46/26/9/0/0 | 81 | fail | 46/34/21/0/0 | 323.6 | 4/16 | lm 0.627 dirt 0.519 rough 1.88 float 10.0 (all agree) |
| guided2 | 27/21/7/1/0 | 56 | fail | 27/26/16/6/0 | 347.6 | 5/16 | lm 0.626 dirt 0.520 rough 1.93 float 6.1 (all agree) |
| guided4 | 18/6/2/0/0 | 26 | pass | 18/8/5/0/0 | 369.4 | 9/16 | lm 0.622 dirt 0.520 rough 1.92 float 3.9 (all agree) |
| guided8 | 15/4/1/0/0 | 20 | pass | 15/6/3/0/0 | 374.1 | 9/16 | lm 0.624 dirt 0.520 rough 1.94 float 4.1 (all agree) |
| guided4+repair | 14/3/2/0/0 | 19 | pass | 14/4/5/0/0 | 369.3 | 9/16 | lm 0.622 dirt 0.520 rough 1.93 float 3.4 (all agree) |
| guided8+repair | 15/4/0/0/0 | 19 | pass | 15/7/0/0/0 | 373.9 | 9/16 | lm 0.624 dirt 0.520 rough 1.94 float 4.1 (all agree) |

cells changed per guided step (mean over worlds): [323.6, 105.8, 78.6, 39.8, 18.8, 31.6, 12.1, 14.6]; time per world (GPU 1 unless noted): pass 5.7 ms, repair 1.9 ms, jitter R=1 8.1 ms, R=2 14.5 ms, 8 guided steps 45.3 ms (5.7 ms per step)

CPU check: `runs/pen_s0_cpu.log` (cores 0-7) reproduces `runs/pen_s0.log` (GPU 1) bit for bit on every count; CPU
times per world: pass 6 ms, 8 guided steps 43 ms, jitter R = 2 14 ms (compile 3 / 136 / 90 s).

## 4. Reading

- **Cells changed per step never go to zero** (pen_s0: 30, 16, 26, 19, 17, 23, 17, 19; W1: 324, 106, 79, 40, 19, 32,
  12, 15): the selection keeps finding violations because the redraw creates new ones. A redraw from q at a site
  violates at about q's own raw rate; with ~30-300 redrawn cells per world per step and raw rates of 1e-5 to 2e-4 per
  instance the recreated count is comparable to what was removed. The coarse selection (descend) also redraws cert3
  cells and adds L3 violations the raw pass did not have (pen_s0 steps 3, 6, 7, 8; rkl_w2b steps 1-6).
- **Where it helps:** on the two worst models (C, W1), 4-8 steps cut the total 3-5x (81 -> 17, 113 -> 20), roughly
  the jitter repair's level (17, 32), at 10-20x the cells changed; on the model closest to acceptance (pen_s0) it
  does nothing useful: the cpass repair and the jitter repair already pass the budget, the guided steps do not.
- **Distortion is small by the bench's rule** on every model; the statistic that moves is floating, which is itself a
  violation count. The real cost is the 30-370 rewritten cells per world: the output is no longer q's draw on those
  cells, which is exactly what the raw-vs-final check cannot see at 16 worlds.
- **Accept rules win over resampling** for a frozen q: both monotone baselines reduce violations at every step and
  touch 1-25 cells. A guided step with the jitter repair's acceptance (keep the redraw only if the touched violations
  do not increase) would be a third variant; it was not built (pause order; no more building without the table
  justifying it).

## 5. Generation-time use: proposal only, not run

The table does not justify a bench of `ckpop_hard_pen_gs2/_gs4` (section 1), so none was run and no new variant was
benched. If a guided step is ever used at generation time: R fixed steps with the declared margin are inside
constraint 2 (fixed count, local windows, finite support, coordinate-keyed noise; tested). But the bench scores
log q of the RAW draw and treats the revised output as "final", exactly like repair, so in the bench's eyes a guided
step on a frozen q is a repair, outside the trained density, and counts only through final violations and the
raw-vs-final check. For it to be part of the sampler (its density scored, its log q in the objective) the revising
steps must be in the trained generator and objective, which is Amendment 2026-10-14 (b) (path KL, #19): that is where
variant B belongs. `guided_step` returns the per-step log q for that purpose.

## 6. What failed / was not done
- The #20 (a) first test (4-seed bench of `ckpop_hard_rkl` with the jitter repair) was not run: user decision.
- Variant B was not measured: without training it is a slot (the saved actors have the narrower input).
- Graded guidance (castlegen/graded.py in JAX) was not used for selection, only reported; the zero set is the same
  as the binary field's, so the selection would be identical; the values matter only as actor inputs (B).

## 7. Next
- For the #19 agent: train with the field features (actor input width + 2) and the revising steps in the path-KL
  objective; `guided_step` gives the step's log q.
- If a repair is wanted after all: the jitter repair at R = 1 (accept rule, 3-25 cells per world) is the cheaper and
  monotone option; its bench is a 25-minute run per variant.
- A guided step with an acceptance rule (one line in `guided_step`), measured the same way, to separate "where q
  redraws" from "what it keeps".

## Files
castlegen/variants/_guided_step.py, cguided.py, _jitter_repair.py (baseline); tests/variants/test_guided_step.py;
reportables/2026-10-14_guided_step/: report.md; scripts/ (env.sh, smoke_toy.py, offline_eval.py, run_offline.sh,
run_tests.sh, gpu1_queue.sh, tables.py); runs/ (tests.log, smoke_pen_cpu.*, pen_s0_cpu.*, pen_s0.*, rkl_w2b.*,
rkl123x2_C.*, hard_w1.*, tables.md, gpu1_queue.log, nvidia_smi_before/after.txt, pids.txt). No images (no training
curves; the bench images were not produced because no bench ran).

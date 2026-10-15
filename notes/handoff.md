# Handoff: 2026-10-15 waves (two waves, nine agents, all finished)

Read `reportables/2026-10-15_waves/report.md` (coordinator log with every agent's numbers, the synthesis, decisions for the user and recommended next lines). Per-agent reports: `reportables/2026-10-15_{p5,rgap,p6p8,p10,gpen_p11,p4fill,dstage,deq,kblock,stage}/`. Board regenerated (`results/board.md`, row `ckpop_hard_pen_stage`). New variants: `cnocq.py`, `ckblock.py`, `cstage.py`, `_rgap.py` (all registered under new names, none accepted). No jobs are running. Everything is pushed.

Short version: ruled out the designed-r revising sampler, the graded penalty, dropping certificate log q, and block kernels in the training population. Signal: a fixed post-pass MCMC stage makes own draws fully valid and, with enough rounds, chain-like (line_match 0.667) but fails raw~final and finite support as built; the tight-abstraction certificate refill is valid by construction (F 20); the pass draw has a +4.6-row offset bias at the source; the dimers gate 0.5/0.25/0.25 is wrong (exact 0.351/0.324/0.324). Decisions pending for the user are listed in the waves report.

---

# Handoff: state at pause, 2026-10-14 evening session (new box)

**Paused because the user ran out of usage.** Read this section first; the sections below are the earlier sessions' handoffs (still valid for context).

## Box
A different Vast instance from the 3090 box: 2x RTX A4000 16 GB, 24 cores, pid cap 1792, no volume (recycle wipes it). Repo = real git clone at `/workspace/Hierwave`, branch `coloured-pass`; `/venv/main` has jax[cuda13] 0.11.2. Rules and flags: `notes/box_rules_2026-10-14.md` (last section), `reportables/2026-10-14_cr2_penalty/scripts/env.sh`.

## Decisions taken by the user this session (recorded in `notes/coloured_pass_constitution.md` addenda and `reportables/ideas.md`)
- Change request #2 ADOPTED (addendum): ML + penalty-only term. Built and tested: `reportables/2026-10-14_cr2_penalty/report.md`.
- Ideas #19 (revising sampler, path KL; amendment (b) pushed by the user), #20 (jitter), #21 (graded violation distances) pulled. User direction: guided, repeated updates with the graded signal, NOT draw-K search; repair line dropped. Graded mass in the training penalty approved for a first test (addendum (c)).

## Results this session (details and tables in `reportables/2026-10-14_cr2_penalty/report.md`)
- Penalty-only estimator unbiased (tests/variants/test_cpen.py; 1800-batch diagnostic).
- Own-draws arm `ckpop_hard_pen` (in the rules), 4-seed bench: pooled L2 7.6e-5 (interval entirely under budget), 3 of 4 seeds within budget (the miss = one L3 instance), across-seed agreement passes; NOT accepted (that instance + the old probe flag). Cost: line_match 0.56-0.59 (vs 0.60-0.63), F +0.03 (vs -0.09).
- Dimers: penalty-only keeps draws valid (7.9e-5) but no diagonal; h/v split unstable.
- Wave5 reruns on a cool GPU (reverse-KL mix, probe-context arm): l2x2 2 of 4, l2x15 1 of 4; pooled L2 2.4e-4 both; L3 singles seed-driven. ~1900 updates per 300 s here.
- Upper-bound penalty arm `ckernel_pen4w3f` bench: seeds 0-2 within budget (F -0.02..-0.05), seed 3 was running at pause.

## Running / queued at pause (no tokens needed; results land in files)
- GPU 1 queue: `reportables/2026-10-14_cr2_penalty/runs/`: `ub_pen/bench_ckernel_pen4w3f.log` (4-seed bench; row appended to results/bench.jsonl), then `arm_ckpop_pen16_s0/` and `arm_ckpop_pen_lr5_s0/` (lower-pressure own-draws arms, eval.json each); `gpu1_followup.nohup` prints FOLLOWUP_DONE at the end. Summaries: `scripts/pool.py results/bench.jsonl` (last row), `scripts/summarize.py <armdir>`, `scripts/rho.py <eval.json>`.
- Agents (paused with state written in their report.md; they may have launched detached jobs, see their reports): `reportables/2026-10-14_crevise/` (idea #19, GPU 0), `reportables/2026-10-14_guided_step/` (guided revising step, was #20), `reportables/2026-10-14_graded/` (idea #21 + approved graded-penalty test).

## What to do first next session
1. Read the three agent reports and the pending sections 3.4 / 4 of the cr2_penalty report; fill them from the run files above (bench rows via pool.py; arm evals via summarize.py), regenerate `results/board.md` (`python -m castlegen.bench board`), commit, push.
2. Decisions still pending for the user: change request #1 (probe-lift contexts), #3 (post-repair pricing), the v2 acceptance gate / dropping the old probe flag, eval size (one L3 instance in 16 worlds = 2.4e-4).
3. Recommended lines (coordinator's view, in the cr2 report's verdict and the chat): guided revising steps with the graded field (#19 + #21), graded penalty first test, lambda ramp in the penalty mix, deeper (one level) lookahead on certificates.

---

# Handoff: coloured-pass research loop (2026-10-13, evening)

For a Claude session continuing this work, e.g. running on the Vast GPU box. Read this file, then the documents listed in `notes/coloured_pass_constitution.md`'s start-here section.

## The user's standing rules

These are not all in the repo elsewhere.

- **Constitution:** `notes/coloured_pass_constitution.md` (exported from the user's doc).
  - Only the user changes it.
  - Anything outside its allowed freedoms is a proposal, not an experiment. That includes biased estimators, objective changes and special-case moves.
  - Approved exception: truncated-depth returns (ideas #2).
- **Generic algorithm only:** no problem-specific steps, regions or features. Fix failures by giving the learned parts more information.
- **Test problems stay "easy":** short-range correlations, no criticality or phase coexistence.
- **Experiments last 5–10 minutes.** No parameter sweeps or long runs without a very good, stated reason.
- **Reporting:**
  - each piece of work gets a dated `reportables/<date>_<topic>/report.md` with figures;
  - log progress in `reportables/progress.md`, newest first;
  - read the user's `reportables/ideas.md` and annotate it, never editing their text.
- **Compute:** the Mac runs at most one job; everything else runs on the GPU box.
- **Git:** work on branch `coloured-pass`. Commit and push after each round, never to main, never force-push.
- **Other agents:** a GPT/Codex agent ("astra") also works on this project, with files `cfast_head*` and an `astra_` prefix. It will move to branch `astra`. Don't touch its files.
- **Style:** be concise. Report failures plainly; good-looking curves are not success.

## The GPU box

- Vast.ai RTX 3090 24 GB, CUDA 13, 80 CPUs, `ssh -p 26359 root@137.175.22.196`.
- Repo at `/workspace/hierwave`; Python environment `/venv/main`, with jax[cuda13] 0.11.2 already installed.
- On the box itself, run directly:
  `cd /workspace/hierwave && source /venv/main/bin/activate && PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false python -m castlegen.bench run <variant> <problem>`
  (`scripts/gpu.sh` is the Mac-side wrapper).
- No persistent volume: destroying or recycling the instance wipes it, so push to git often. Stopping and restarting is safe.
- If you work in a git clone on the box rather than the rsynced copy, note that `/workspace/hierwave` was populated by rsync and isn't a git checkout.
- **WARNING: thread limit.**
  - The container's pid limit (pids.max 2304) is shared by every process, and each default JAX process holds about 400 threads. With several processes running, the GPU client fails: "Failed to launch ptxas / pthread_create failed".
  - Cap threads per process: `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`, `taskset -c <few cores>`, and for CPU-only jobs `JAX_PLATFORMS=cpu`.
  - Run only a few JAX processes at once.
- **Left running by round 2 (nohup):**
  - exactopt's runs: `/tmp/eo_modes.log` → `/tmp/eo_runs/*.json`, `/tmp/eo_support_enum.log`, `/tmp/eo_test.log`.
  - Check `ps aux | grep python` for others.
- **Round 2 status:**
  - creplay stopped mid-refactor and is currently broken; see its report for the fix plan.
  - exactopt stopped early, but early numbers show exact training well ahead of the sampled estimators on the toy_modes block.
  - Check each `reportables/2026-10-13_<name>/report.md`.
- GPU results go to `results/gpu/` (separate board). The GPU only pays off with large batches; B = 8 is slower than the Mac.

## Where things stand

- **Harness:** `castlegen/bench/`, frozen. Suite: toy_support, toy_modes, ground, dimers, columns.
- **Round 1** (reports in `reportables/2026-10-13_*`):
  - cpass's estimator is unbiased but extremely noisy.
  - cfast (cone returns) makes training 5–12× faster.
  - ctrunc (truncated depth) and cordinal (smooth head) gave partial gains; ctemp (schedules) gave none.
- **Codex's cfast_head:** ground meets the violation budget, but every run collapses to all dirt or 50 % stripes and ignores the designed fill line.
- **Working hypothesis:** the fine levels can't reliably build partially filled, structured contexts, so the coarse levels learn to avoid them. Coarse credit and variance fixes alone may only make that avoidance more efficient.
- **Running at handoff** (agents launched from the Mac session; if that session died, they died too, so check `reportables/progress.md` and `results/gpu/bench.jsonl` for what finished):
  - `cbig`: cfast at large batches on the GPU.
  - `creplay`: coupled replays for many coarse draws, within cones.
  - `cconv`: conv trunk (ideas #5).
  - `cmulti`: multi-pass levels (ideas #3). Its objective may need the user's approval.
  - `crb`: candidate-averaged downstream costs (ideas #8).
  - `exactopt`: exact-gradient optimisation diagnostic (ideas #10).

## Next steps the user has agreed to

1. Finish and collect round 2, then compare everything on the GPU board.
2. Use exactopt's verdict (variance vs representation or sampling order) to choose round 3. If representation or order, target fine-level capability: cconv-style context with candidate scoring, revisable passes, deeper lookahead, and training fine levels on target-like coarse contexts.
3. Queued: a variance-minimising critic (ideas #9); low-to-high depth-1 scheduling (ideas #7, low priority).

## Box session, 2026-10-14 (orchestrated waves)

Read `reportables/2026-10-14_box_waves/report.md` first: it summarises all four waves, the collapse mechanism as measured, what is solved (the toys under bench v2) and the recommended next steps. User decisions taken during the session (bench budget 300 s, ideas #12 and #14 in scope, bench v2 probe check, ground as the focus problem with the fast loop) are in `notes/box_rules_2026-10-14.md`.

## Overnight box session, 2026-10-14 (kernel route)
Read `reportables/2026-10-14_overnight/report.md` ("What the night established", then the wave log) and `notes/methodology.md` §7-9. Decisions pending in `notes/constitution_change_requests.md` (#1-#3). Main variants: `ckpop_hard_rkl` (own draws only, castlegen/variants/ckernel_own.py) and `ckernel_rkl4w3f` (upper bound, ckernel.py). Kernels in `castlegen/kernels/`.

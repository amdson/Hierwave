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

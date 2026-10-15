# Shared brief for the 2026-10-15 waves (read fully before acting)

You are one of several agents working in parallel in `/workspace/Hierwave` (branch `coloured-pass`) on this Vast box. The coordinator (not you) commits; **never run git**. The user is not watching; your report files are the deliverable.

## Read first (in this order, skim where you already know it)
1. `notes/coloured_pass_constitution.md` (hard constraints; cite by number; nothing outside the allowed freedoms gets built, it gets written as a proposal).
2. `notes/box_rules_2026-10-14.md`, last three sections (new box, flags, kill rules).
3. `reportables/ideas.md` (the user's ideas; annotate under the item you touch, never edit the user's text).
4. The files your task names.

## The user's instruction for this session
"There's a ton of ideas now. Don't try all of them exhaustively. Rule them out quickly, or check for signal quickly, then look for promising directions. Use the time effectively." Your job is one sharp question with a yes/no answer in **under ~60 minutes of wall clock**, not a thorough study. Stop early if the answer is already clear.

## Box rules (mandatory)
- Environment: `source reportables/2026-10-14_cr2_penalty/scripts/env.sh <gpu>` (activates `/venv/main`, sets PYTHONPATH, thread caps, JAX cache, `CUDA_VISIBLE_DEVICES`). For CPU-only JAX add `JAX_PLATFORMS=cpu`.
- 2x RTX A4000 16 GB, 24 cores, **pid cap 1792**. Your GPU and core range are in your task. Pin with `taskset -c <range>`. **Run at most one JAX process at a time.** Before launching one, check `cat /sys/fs/cgroup/pids.current`; if it is above 1400, wait.
- Never `pkill -f python` or any pattern; kill only PIDs you launched. Record PIDs.
- Before and after each GPU run: `nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv` into your runs folder. Say in the report that the GPU was shared.
- Each experiment 5-10 min. No sweeps. Warm the compile with a short smoke run before a timed run.
- Run jobs with `nohup ... > runs/<name>.log 2>&1 &` and poll the log; do not block on a foreground run longer than a couple of minutes.

## Files you may touch
Only your own: `castlegen/variants/<yourvariant>.py`, `castlegen/variants/_<yourvariant>_*.py`, `tests/variants/test_<yourvariant>*.py`, your report folder `reportables/2026-10-15_<name>/` (with `runs/`, `images/`, `scripts/`), plus append-only edits to `reportables/progress.md` (a `started` entry now, a `finished` entry at the end, newest first under the `---` line, format at the top of the file; re-read right before editing, others edit it too) and annotations under the relevant `reportables/ideas.md` item. Never edit `castlegen/bench/`, `cpass.py`, `_cpass_core.py`, `cfast.py`, `_cond_jax.py`, `_local.py`, the constitution, or another agent's files. To change shared code, subclass or copy into your module.

## Report
`reportables/2026-10-15_<name>/report.md`: the question, what you ran (exact commands), a numbers table, the yes/no answer, what failed, what you would do next (one line each). Plain statements of failure; curves and images are not success. Figures only if they carry the answer.

Final message to the coordinator: **≤ 250 words, numbers first, the yes/no answer in the first line**, which files you wrote, what you would try next. No code.

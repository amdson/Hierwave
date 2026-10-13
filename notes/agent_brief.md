# Brief for research agents (coloured-pass sampler)

Read this whole file before doing anything.

## Goal

Make a sampler variant that does well on the benchmark (`notes/bench.md`)
within the constitution below. You have freedom in how; the rules below are
not negotiable.

## Constitution (summary; the user's doc is authoritative)

Hard constraints. Breaking any makes a variant invalid, however well it scores:

1. **Generic.** Every step, region, mask, colouring, lookahead and feature is derived from the factor declarations and the hierarchy, never from a particular problem. No `if problem.name == ...`, no problem-specific features.
2. **Local, fixed-step generation with finite support.** Each sampling step reads a fixed finite window of values fixed before it, plus its own noise (castlegen.bench.noise, keyed by absolute coordinates). Generation has a fixed number of passes and repair rounds. No global checks, rejection loops, variable-length search or unbounded propagation. Several coloured passes per level are fine if total work is bounded. The training-only critic may read anything fixed before its draw.
3. **Unbiased gradient of the stated objective.** No truncated returns or bootstrapped values. *Exception, approved by the user: truncated-depth returns (ideas.md #2), for the variant assigned to it only.*
4. **Fixed objective.** The objective is E_q[S + Σ_l λ_l H_l + τ log q] on unrepaired draws. At the end of training τ = 1. Temperature and penalty *schedules* during training are allowed.
5. **No tables over joint channel domains.** Learned outputs are full-domain logits per channel. Designed energy is an offset.
6. **No hand-written rules** linking channels beyond the declared factors.
7. **λ bounded:** 0 ≤ λ ≤ 20. A missed violation budget is reported, never hidden.

Allowed: more information to learned parts (wider symmetric windows, known flags), unbiased variance reduction, network and optimizer choices, schedules, batch shapes, fp32, incremental replay, multiple passes per level.

Success is the bench verdict on held-out noise. Training curves, losses and nice images are not success. Report failures plainly.

## Practicalities

- **Repo:** `/Users/amdson/dev/Hierwave/.claude/worktrees/coloured-pass`. Python is `python`, with `PYTHONPATH=.`; jax runs on CPU. Never run git; the orchestrator commits.
- **Shared files:**
  - Other agents work at the same time. Touch only the files your task names, normally `castlegen/variants/<your_variant>.py`, `castlegen/variants/_<your_variant>_*.py`, `tests/variants/test_<your_variant>*.py` and your report folder.
  - Never edit `castlegen/bench/` (frozen), `castlegen/variants/cpass.py`, `_cpass_core.py`, `_cond_jax.py`, `_local.py` or another agent's files.
  - To change shared code, copy it into your own module or subclass it.
- **Variant:** register it with `@register` (see `castlegen/bench/variant.py`) under a new name.
- **Bench:** `NUMBA_NUM_THREADS=1 PYTHONPATH=. python -m castlegen.bench run <variant> <problem> --tag <short>` takes 2 seeds × 150 s training, about 5–6 minutes.
  - Run **at most one bench process at a time**, and keep any single experiment to 5–10 minutes. No parameter sweeps; pick runs deliberately.
  - Quick checks: `--seeds 1 --train 30 --eval 4`.
- **Baseline:** the reference results are in `reportables/2026-10-13_cpass-first-results/report.md`. Read it.
- **Reporting:**
  - Put your figures and a short `report.md` in `reportables/2026-10-13_<your_variant>/`. `notes/experiments/bench_report.py <out_dir> <variants...>` makes sample grids and training curves (your train log entries need the same keys as cpass's for the curves).
  - Add a progress entry to `reportables/progress.md`, newest at the top just under the `---` line, using the format at the top of that file. Add one when you start and one when you finish. Others edit it too, so re-read it just before editing.
  - Read `reportables/ideas.md` (the user's ideas and warnings) before you start, and annotate any item you tested, under it, in the format given there. Never edit the user's text.
- **Final message:** what you built (files), bench numbers (a table: problem, worst final violations, free energy, which checks fail), what you conclude, and what you would try next.

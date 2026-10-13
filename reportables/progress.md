# Progress log

Shared by every agent working on the coloured-pass sampler. Append an entry
when you start a piece of work and when you finish or abandon it. Newest
entries at the top. Keep entries short; put figures in a dated report folder
and link them.

Entry format:

```
## YYYY-MM-DD HH:MM · <agent / variant name> · <started | finished | abandoned>
Approach: what you are trying and why (cite ideas.md items by number).
Result: bench numbers (variant, problem, accepted?, worst final violations,
free energy), or why it stopped.
Next: what you would try next.
```

---

## 2026-10-13 15:12 · cfast · finished
Approach: cpass with dependency-cone returns (conservative read-graph superset, box sums on device), critic on actor hidden layer (no recorded critic features), unbiased subsampled loss (64 sites/step/world), no replays, one fused jitted update, compiled functions shared across seeds, tau anneal after setup, lr 3e-3. Folder 2026-10-13_cfast/.
Result: 5-12x faster per update (ground 0.43 vs 5.2 s, columns 0.16 vs 0.89 s, same load); cone cuts gradient variance 357x / 8x / 1.8x at levels 0/1/2 (ground). Bench (2 x 150 s): toy_support viol 2e-3, F -0.441 (KL 0.028); ground 648/549 updates, viol 0 / 2.9e-2, F 0.885 but collapse to all dirt; columns 1110 updates, viol 2.1e-2, F -0.05, collapse to air (density 0.02-0.06). Nothing accepted.
Next: entropy-preserving schedules (ideas #1) on top of cfast; coupled replays for many coarse draws; smaller actor window for tighter cones.

## 2026-10-13 15:15 · ctrunc · finished
Approach: truncated-depth bootstrapped training (ideas.md #2). Generator/objective as cpass. A coarse-to-fine chain with batches shrinking toward the fine level (ground B = 1,1,4,16,64), local returns (box R = 3) truncated d levels down, and a per-cell value v_m (sampler encoder) regressed on d-step targets. No replays. Variants: ctrunc (d = 1), ctrunc_d2, ctrunc_full.
Result: nothing accepted. 2-5x cpass's updates/s (ground 2-3/s, columns 4-6/s).
- toy_support: F -0.441 (d1) / -0.445 (full) vs cpass -0.374 (opt -0.469); worst final viol 3.9e-3 / 4.9e-4; stats still off (flat blocks 15-25 %).
- ground: F 1.05 (d1) / 0.74 (d2) / 0.72 (full) vs 1.92. Fine-level viol 4-15x lower, but level 4 is 4-8 % after repair; d2 seed 0 is near-valid (1e-3) only by filling everything with dirt.
- columns: d1 viol 0.10, density 0.49; full collapses to an empty world.
- V: unbiased in the mean (|bias| < 0.015 nats per fine cell) but weakly informative (corr 0.1-0.7, rmse about sd). d = 1 is the worst depth.
Next: full-depth chain as default plus an entropy or soft-constraint schedule (ideas.md #1); a repair that doesn't push violations to level 4. Report text is in the agent's final message (the report.md write was refused for subagents).

## 2026-10-13 16:00 · cordinal · started
Approach: ideas.md #6 on cpass, estimator untouched: (a) thermometer (16 bins) + one-hot (n <= 32) input codes for views > 8 values; (b) smooth actor head for D > 8, b = sum_k c_k cos(pi k u) + 0.1 r_v; (c) u relative to the range of the least-violating set A. Bench toy_support, ground, columns vs cpass.
Result: pending.

## 2026-10-13 15:40 · gradcheck (cpass estimator) · finished
Approach: spec test 5 on an enumerable 2-level strip (44 614 reachable outcomes): exact J and grad J by enumeration vs the mean of cpass's actor-loss gradient over 2000 x 16 worlds per config (tau 1.5, lambda (2, 3), P(violation) 0.61).
Result: unbiased. Projection on grad J 0.97-1.03 (|z| <= 1.4) in every config: zero / trained / wrong critic, branches enumerated (K 4) and sampled with LOO (K 2), branch term alone, unbranched top draws alone; mean cost matches J. No bug. Variance is the issue: per-world SNR 0.15-0.36 on the tiny strip, ~1e-3 on toy_support and undetectable on dimers at init (zero critic). Actor grad clip binds 1.7 % (toy_support l0) / 5 % (dimers) of batches: a small, technically biasing component. Test: tests/variants/test_cpass_gradient.py; data in 2026-10-13_gradcheck/.
Next: variance reduction (dependency-cone returns, better critic, larger B); entropy loss is optimisation, not estimator bias.

## 2026-10-13 14:50 · ctemp · finished
Approach: cpass plus per-level τ, actor conditioned on log τ_l, schedules anneal / mix (all levels, τ ~ log-U[1, 3]) / coarse (only non-finest levels heated), last 50 % at τ = 1; λ controller only on τ = 1 batches; λ soft starts tried (ideas #1). Diagnostics: training-batch stats, exact coarse/fine KL split on the toys (reportables/2026-10-13_ctemp/kl_split.py).
Result: no improvement, nothing accepted. ctemp (coarse) vs cpass:
- toy_support: KL 0.095 / 0.089 vs 0.096 / 0.092, viol 4.9e-4.
- toy_modes: KL 0.012 / 0.014 vs 0.013 / 0.011, viol 0.
- dimers: collapse 96 % H / 99 % D vs 91 % / 99.9 % H, viol 1.3e-3.

Diagnosis:
- Dimers collapse at τ = 1 exactly when violations drop. A λ ramp from 1 only delays it, and the run then collapses to diagonal instead.
- toy_support has exact stats at λ = 1 (14 % violations); flat blocks return as λ → 20.
- toy_modes fine KL is 0.6–1.0 nats/block in non-mode contexts, so the coarse level avoids them. Heating the coarse levels cut fine KL 0.22 → 0.15 but raised coarse KL 0.10 → 0.16.
Next: lower-variance credit for coarse and early draws (cone returns, replays at every level), then retry the coarse τ schedule; more passes per level for dimers (ideas #3).

## 2026-10-13 14:30 · ctrunc · started
Approach: truncated-depth training (ideas.md #2, approved exception to constraint 3). Generator and objective as cpass; training is a coarse-to-fine chain with batches shrinking toward fine levels (on-policy start states, many cheap coarse worlds), returns local to the draw and truncated d levels down, the rest replaced by a learned per-cell value v_m (same encoders, finite window) trained by d-step bootstrapped regression across levels. No coupled replays.
Result: pending.
Next: bench ground, columns, toy_support; compare d = 1 with full depth.

## 2026-10-13 14:10 · gradcheck (cpass estimator) · started
Approach: spec test 5: exact J(theta) and grad J by enumerating every draw outcome of a tiny 2-level periodic problem, against the mean of cpass's actor-loss gradient over many batches (zero / trained / wrong critic, enumerated and sampled coupled replays, tau != 1, lambda > 0 with violations).
Result: pending.
Next: -

## 2026-10-13 14:10 · ctemp · started
Approach: entropy loss / mode collapse of cpass on toy_support, toy_modes, dimers. Diagnose first (free-energy gap and orientation fractions during training), then τ / λ schedules and a τ-conditioned actor trained over a range of τ (ideas #1), ending at τ = 1.
Result: –
Next: –

## 2026-10-13 14:03 · cfast · started
Approach: throughput for cpass on ground and columns: profile, dependency-cone returns (G_i summed over a conservative superset of draw i's read-graph descendants, via per-step box sums), cheaper/fewer replays, compiled-function reuse.
Result: pending.
Next: -

## 2026-10-13 13:55 · orchestrator / cpass · finished
Approach: the final spec as written (castlegen/variants/cpass.py): coloured pass with one-step lookahead mask, per-(level, channel) MLP bias, full-future returns, critic control variate, K = 4 coupled replays of one coarsest draw per world, bounded penalties, τ 2 → 1, one repair round.
Result: [report](2026-10-13_cpass-first-results/report.md).
- toy_support: viol 2.4e-4, KL 0.095 nats/cell.
- toy_modes: viol 0, KL 0.013.
- dimers: viol 8e-4, but collapsed to horizontal.
- ground: 7e-2, unchanged (60 updates).
- columns: 1.6e-1, density halved.
- Nothing accepted.
Next: gradient test on an enumerable case; entropy loss (ideas #1); throughput on ground and columns.

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

## 2026-10-13 · cmulti · stopped early
Approach: ideas.md #3 on cfast: P = 2 passes per level (pass 1 = cfast's pass; pass 2 redraws every site colour by colour given all current neighbours, own actor). Objective = auxiliary-variable upper bound E_q[S(x_P) + lam c + tau(log q(traj) - log r(x_{<P} | x_P))], learned per-site backward kernel r; needs user approval. Tests (11) pass on the box: P=1 = cfast, telescoping costs = S + log q(traj) - log r exactly, cone containment, finite support.
Result: partial. GPU, 2 x 150 s: cfast toy_support viol 0, F -0.432 / -0.441. cmulti toy_support seed 0 viol 0, F_ub in info (bench F = None: score() unavailable). Smoke run (1 x 60 s): cmulti dimers collapses to 100 % vertical, log q(traj) ~ 0, so pass 2 does not break the collapse. On a valid matching, single-site redraws are frozen. Dimers and columns benches did not run: the box's pid limit killed XLA thread creation. scripts/chain2.sh (taskset, retry) is ready but was not started.
Next: run chain2.sh; block (pair) moves in later passes for frozen hard constraints; fewer colours.

## 2026-10-13 · cbig · stopped early
Approach: cfast on the GPU box with large batches; pre-selected subsample recording (memory), vectorised + overlapped designed inputs, batched cached generation with cached log q. Folder 2026-10-13_cbig/.
Result: no bench yet. cfast speed on 3090: toy_support flat to B~256 (4.5 ms @8, 22 ms @1024); ground saturates at B~64 (14/34/116 ms @8/64/256, OOM @1024, host designed inputs > GPU time). Fresh GPU compile 45-150 s eats the 150 s budget (autotune_level=0: 132 -> 29 s on toy_support); box pids limit 2304 vs ~316 threads/JAX process causes ptxas launch failures (taskset to 8 cores -> 44 threads).
Next: warm cache or scoped autotune off; profile cbig; SNR vs B; benches at B=256/1024; check collapse/fill tracking vs B.

## 2026-10-13 · exactopt (diagnostic) · stopped early
Approach: ideas.md #10. Exact J and its gradient by enumerating the reachable outcomes, with deduplicated draw contexts; mlp, feature-table and full-history-table actors, compared with cpass and cfast sampled training from the same init. Problems: tiny, and single toy_support / toy_modes blocks (non-periodic 4x4).
Result: machinery and tests built (5/6 tests pass; the history-table convergence test was lengthened and its rerun is unconfirmed). Partial toy_modes numbers only (KL from 2.04): exact mlp 0.90 at step 150; at 20 steps exact 0.57-0.81 against sampled cpass 1.79 and cfast 1.43. No verdict. The full runs are still going on the box (nohup, /tmp/eo_modes.log). [Report](2026-10-13_exactopt/report.md).
Next: pull /tmp/eo_runs, run toy_support and tiny, plot, and apply the decision rule.

## 2026-10-13 23:59 · creplay · stopped early
Approach: cfast + coupled replays for many coarse draws per world (cone-scored, enumerated when |A| <= K, else LOO), batched over (world, draw, action) on the GPU. Folder 2026-10-13_creplay/.
Result: no bench. Replay faithfulness tests pass (4 problems). Gradcheck passes for all-enumerated branching (others unfinished). Ground on the GPU: 4096 replays/update = 6.1 s/update and 222 s compile, so it does not fit the budget as configured. Per branched draw, the across-action spread (~30) is ~1/35 of the return noise (~1000-1500): about 10^3x variance reduction per draw. Code is mid-refactor (importance-weighted subset estimator: t1 for all draws, (N/R)·Σ coupled terms for a random subset), so it does not run yet. Fill-context clamp diagnostic written (_creplay_clamp.py), not run.
Next: finish the refactor, R = 16 per level, bench against cfast on the GPU, run the clamp diagnostic.

## 2026-10-13 · exactopt (diagnostic) · started
Approach: ideas.md #10. Enumerate every reachable draw outcome of small hierarchies (gradcheck tiny; a single toy_support block; a single toy_modes block), precompute each draw's context once, and optimise the cpass actor with the exact gradient of J = E_q[S + lam H + tau log q]; compare exact KL curves with cpass- and cfast-estimator training from the same init and step count; a tabular actor (per feature context, and per full history) separates representation from optimisation.
Result: pending.
Next: -

## 2026-10-13 · Codex / cfast_head · finished
Approach: ideas.md #6(b) only on cfast: smooth absolute-domain cosine head, original inputs/critic/estimator. Cheap agent implemented; parent reviewed and benchmarked.
Result: ground, 2 x 150 s: worst final viol 9.16e-5 versus fresh cfast 6.59e-3; F 1.204 versus 0.813. Head seeds collapse to 50% dirt stripes / 100% dirt; both fail coverage probes and across-seed agreement. Nothing accepted. Updates 585/622 versus 926/825. Ten cone tests and four new head tests pass; gradient equivalence rechecked with actual cone returns. [Report](2026-10-13_cfast_head/report.md).
Next: keep as an ablation; creplay and cbig already address lower-variance credit. Columns and mask-relative follow-ups not run because coverage did not improve.

## 2026-10-13 23:10 · creplay · started
Approach: cfast plus coupled replays for many coarse draws per world (all draws at the top level, a random subset at the next), K actions each under the world's stored noise, each replay scored by the draw's dependency cone; enumeration when |A| <= K, leave-one-out baseline otherwise; replays batched over (world, draw, action) on the GPU. Gradcheck on the tiny enumerable problem; bench on the GPU box against cfast.
Result: pending.
Next: -

## 2026-10-13 · cconv · started
Approach: ideas.md #5 on cfast. Actor = per-level conv trunk (input planes: the level's channels embedded, known and exists flags, parent planes upsampled, position-in-parent, row), 3x3 stem + residual pre-LayerNorm 3x3 SiLU blocks at fixed radius r, recomputed densely each step; per-channel linear head plus candidate-scoring head (site features + value embedding, s(t), nv(t), lookahead). Loss re-evaluates the trunk on (2r+1)^2 patches at the subsampled sites (state at the step rebuilt from the final state and the static schedule). Critic reads stop-gradient trunk features. Cone and margin use the conv reach. Bench on the GPU box vs cfast.
Result: pending.
Next: -

## 2026-10-13 · cmulti · started
Approach: ideas.md #3 on cfast: P passes per level (pass 1 = cfast's coloured pass; passes 2..P resample every site, colour by colour, given all current neighbours, own actor per pass). log q(final) is intractable, so the objective is the auxiliary-variable upper bound E_q[S(x_P) + lam c + tau(log q(trajectory) - log r(earlier | x_P))] with a learned per-site backward kernel r (needs user approval before candidacy). Bench on the box vs cfast: dimers, columns, toy_support.
Result: pending.
Next: -

## 2026-10-13 · cbig · started
Approach: cfast on the GPU box with large batches (B = 8 ... 1024 worlds per update): profile s/update and memory vs B, remove whatever stops scaling, measure per-batch gradient SNR vs B, batch/jit-cached generation; bench toy_support, toy_modes, dimers, ground, columns (+ cfast on the box for ground, toy_support).
Result: pending.
Next: -

## 2026-10-13 · Codex / cfast_head · started
Approach: isolate ideas.md #6(b): cfast with the smooth cosine output head (8 terms per declared ordered view, 0.1 residual, absolute coordinates), unchanged input features and critic. Cheaper agent implements isolated variant; parent reviews and runs checks. User permits up to two concurrent jobs.
Result: pending. Paired ground benchmark: cfast versus cfast_head, two seeds x 150 seconds, 16 held-out worlds per seed.
Next: assess violations per level, dirt excess, floating cells, probe availability, and throughput; advance to columns only if promising.

## 2026-10-13 16:50 · cordinal · finished
Approach: ideas.md #6 on cpass, estimator untouched: (a) thermometer + one-hot inputs, (b) cosine-basis actor head (8 per ordered view) + 0.1 residual for D > 8, (c) basis over the range of A. Folder 2026-10-13_cordinal/.
Result: nothing accepted. toy_support viol 2e-3 (one repaired cell), F -0.387 (KL 0.082 vs cpass 0.095); ground viol 0.029 vs 0.070, F 1.846 vs 1.918, floating 153/651 vs ~1130, more dirt; columns viol 0.15 (= cpass), F -0.645, seed 0 collapsed to empty, 90 vs 150 updates (critic input 3-5x larger under (a)). Ground ablation: a only 0.090 (= cpass), a+b 0.043, a+b+c 0.029.
Next: critic back on cpass encoding (remove throughput confound), rerun columns; combine the smooth head with cfast's throughput.

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

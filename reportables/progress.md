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

## 2026-10-14 01:30 · cln · finished
Approach: cfg options on cfast: net (tanh | ln = LayerNorm + SiLU 64x2) and win (4x4 | 6x6), with the cone and margin following the window; variants cln (ln, 4x4) and cln6 (ln, 6x6). Tests: test_cln.py 11 passed (cone, finite support, defaults unchanged); test_cfast_cone + test_cresid 17 passed.
Result: GPU shared, 2 x 150 s, B = 8, none accepted (probes flagged everywhere). KL per block, s0 / s1: toy_support cln6 1.28 / 0.78, cln 0.45 / 0.90, cfast 0.70 / 0.58 (cfast misses the budget, 5.9e-3); toy_modes cln6 0.26 / 0.26, cln 0.16 / 0.22, cfast 0.15 / 0.16. Ground: all three collapse to all dirt (line ignored); cln6 s0 misses the budget (1.5e-3) after a 125 s compile. Microbenchmark: 5.1 / 6.1 / 6.0 ms per update (cfast / cln / cln6), but the bench gave cln* 2.5-4x fewer updates (contention). The exact-gradient gains do not transfer at B = 8: variance binds. Report: reportables/2026-10-13_cln/report.md.
Next: fixed-update-count comparison; cln6 at large B (cbig).

## 2026-10-14 baselines300 · started
Approach: cfast at 300 s/seed (2 seeds) on toy_support, toy_modes, ground, columns, dimers; cbig256 at 300 s on the toys; compare with 150 s baselines. Cores 48-55, one run at a time, smoke warm-up first.
Result: -
Next: -

## 2026-10-14 01:20 · creplay2 · started
Approach: make creplay cheap enough to compare with cfast at equal wall clock (engineering only): profile one update, cut replay count/cost, fix the compile-cache miss, then GPU bench vs cfast (ground, toy_support) and an equal-update (300) comparison on ground.
Result: -
Next: -

## 2026-10-14 probes · finished
Approach: scored 512 toy_modes probes with a retrained cbig256 (60 s) against exact log p; tracked dimers orientation fractions untrained and through cfast training.
Result: toy_modes: 0/512 probes unreachable, no scoring bug; flag is a genuine, very powerful world-level test (shift 4.06 nats = KL(q||p)+KL(p||q) at F gap 0.0035/cell; flag needs gap under ~0.0005/cell) plus a 16-vs-16 sample-size weakness (exchangeable draws flag 53%). dimers: diagonal is present untrained (0.318), lost within ~20 updates in step with entropy collapse (0.78 to 0.21 nats/cell); not structural.
Next: relabelled-diagonal test to see if the loser follows geometry or label.

## 2026-10-14 probes · started
Approach: diagnose toy_modes probe flags (reachability/target mass/scoring) and dimers missing diagonal orientation (untrained vs trained).
Result: pending.
Next: see reportables/2026-10-14_probes/report.md.


## 2026-10-14 csat · finished
Approach: cfast on ground with per-level saturation diagnostics (entropy over non-forced draws, forced fraction, b spread, dirt), then actor lr 1e-3 and a τ anneal over the whole budget (engineering freedoms), each gated by the clamp test. 150 s per arm, one seed, shared GPU.
Result: the collapse starts at the top level. cert4 goes deterministic and dirt reaches 0.95 at about update 200 (lr 1e-3: 225, while rho3/2/1 are still at 2–3 nats). Lr 1e-3 keeps the fine levels stochastic (final H 0.37 / 1.37 / 0.70), but they stay context-blind (95–100 % identical draws under clamping). Gate, fill vs own: 1.58 vs 0.57 at L = 4 and 2.04 vs 0.37 at L = 3 (default 1.92 vs 1.22 and 2.00 vs 0.93; tauall 1.88 vs 0.71 and 2.11 vs 0.44). Every arm is all dirt, line_match 0.500. The bench was not run. [Report](2026-10-14_csat/report.md).
Next: enumerated per-candidate replays at fine count draws (allowed freedom), plus ideas #12 with whole-logit tempering (τ does not temper b).

## 2026-10-14 csat · started
Approach: test whether the fine-level learned bias saturates early (deterministic rho draws, never visiting fill contexts) on ground: per-level entropy / forced fraction / |b| logged over training for cfast, then two engineering levers (lr, tau schedule, clipping; allowed freedoms) with the clamp gate from cresid.
Result: pending.
Next: see reportables/2026-10-14_csat/report.md.

## 2026-10-14 cconv (round 2) · started
Approach: equal-update comparison cconv_r2 / cconv / cfast on toy_support and toy_modes (~300 updates), then cconv_r2 vs cfast equal wall clock (ideas #5).
Result: pending.
Next: see report.

## 2026-10-14 01:05 · cconv · finished
Approach: first GPU bench of cconv (ideas #5) vs cfast, 2 seeds x 150 s, shared GPU.
Result: cconv worse than cfast everywhere and 5-10x fewer updates. toy_support F -0.395 vs -0.416 (F* -0.4688), worst final viol 1.8e-2 vs 2e-3; toy_modes F 0.29-0.33 vs 0.268 (F* 0.2513); ground (B=16 after B=64 OOM) viol 7.6e-2, streaky dirt, no fill line, one seed 0 updates (cold compile 240 s); columns viol 0.18 vs 3e-2. None accepted. [Report](2026-10-13_cconv/report.md).
Next: r=2/width 16 (3x faster) with a longer run to test whether the gap is throughput.

## 2026-10-14 00:20 · cconv · started
Approach: bench cconv vs cfast on toy_support, toy_modes, ground, columns.
Result: -
Next: -


## 2026-10-14 01:00 · creplay · finished
Approach: finished the importance-weighted coupled-replay estimator (t1 everywhere, t2 dropped at branched levels, (N_c/R)·Σ_selected c_i). Tests: replay faithfulness 4/4, gradcheck 4/4 (CPU). GPU shared with other agents.
Result:
- Per-world SNR at init on ground: level 4 0.010 → 142, level 3 0.014 → 93 (variance /13 900 and /6 800; means agree).
- But updates are slow: 1.4–2.0 s/update on ground at R = (8,8) / (16,16), K = 4, vs 0.07 s for cfast. Compile takes 140–350 s and misses the cache, so seed 0 ends up untrained.
- Ground bench: creplay worst final viol 6.3e-2 (s0 untrained; s1 4.7e-3), F 1.626, 0 / 201 updates. cfast: 1.5e-3, F 0.792, 1142 / 2028 updates. creplay still collapses to ~98 % dirt.
- toy_support: creplay s1 KL 0.026 (3953 updates) vs cfast 0.031 (6001).
- Not accepted. Columns not run. [Report](2026-10-13_creplay/report.md)
Next: cut replay cost (K = 2, top level only, larger B), fix the compile-cache miss, rerun ground and columns at equal updates, run the clamp diagnostic.

## 2026-10-14 01:10 · fixes (round 2) · started
Approach: relax cpass.py:256 assert to rtol 1e-2/atol 1 (float32 GPU reduction order), then GPU bench (2 x 150 s, warmed) cpass ground, ctrunc_full ground, cfast columns rerun; regenerate board. Shared GPU, cores 56-63.
Result: pending.
Next: report in 2026-10-13_gpu_baselines/report.md.

## 2026-10-14 01:25 · cbig2 · finished
Approach: batch size alone on toy_modes/toy_support (see started entry), tag r4, shared GPU.
Result: toy_modes F gap to F*=0.2513: cfast 0.0069/0.0119, cbig64 0.0044/0.0040, cbig256 0.0034/0.0036, cbig1024 0.0056/0.0027 (exact MLP would be 0.0008): 32x batch closes ~half to two thirds, plateaus, B=1024 no better. toy_support cbig256 gap 0.007/0.020 vs cfast 0.027, plateaued; only probes fail. Nothing accepted (probes flagged on all). Report: reportables/2026-10-13_cbig/report.md (round 4).
Next: 6x6 actor window + LN/SiLU at B=256; probes failure (tail coverage) needs its own diagnostic.

## 2026-10-14 01:00 · cbig2 · started
Approach: does B=256/1024 alone close the toy_modes variance gap (exactopt: sampled B=8 KL 0.19/block vs exact 0.013)? Bench cfast, cbig64, cbig256, cbig1024 on toy_modes and cfast, cbig256 on toy_support, 2 seeds x 150 s, shared GPU, tag r4. Added `cbig1024` variant line to cbig.py.
Next: report in 2026-10-13_cbig/report.md.

## 2026-10-14 01:00 · cln · started
Approach: acting on exactopt (ideas.md #10, #5): per-variant cfg options for the actor network (LayerNorm + SiLU 64x2; default tanh) and the actor window (6x6, offsets -2..3; default 4x4), with the cone and margin following the window. Variants cln (LN+SiLU, 4x4) and cln6 (LN+SiLU, 6x6) on cfast.
Result: pending.
Next: bench cln6 / cln / cfast back to back on toy_support, toy_modes, ground.

## 2026-10-14 00:45 · exactopt · finished
Approach: ideas.md #10 round 3, exact gradients on enumerable blocks (CPU): collected the toy_support / tiny runs, then a representation study (LN+SiLU, width/depth, lr, schedule) and an information study (actor window; same-colour independence).
Result: variance dominates at B = 8 (sampled 0.18 vs exact 0.013 on modes; 1.56 vs 0.30 / 0.012 on support). The tanh MLP's support plateau (0.30) is optimisation, not capacity: lr 1e-2 gives 0.012 and LN+SiLU 64x2 gives 0.0084 (table 0.0074). The modes 0.012 floor is the 4x4 actor window: 6x6 gives table 0 and MLP 0.0009. On tiny, the history table floors at 0.037 from same-colour independence (the chain-rule diagnostic gives 0.002); the 8-wide window takes table_feat from 0.32 to 0.07. Report: reportables/2026-10-13_exactopt/report.md.
Next: LN+SiLU with a 6x6 window as the actor default, then sampled training at large B to close the variance gap.

## 2026-10-14 01:10 · cresid · finished
Approach: generic count-residual features (parent view minus known siblings' sum, unknown siblings, parent-unknown flag, per declared count factor), variant cresid = cfast + features; clamp eval as gate.
Result: gate failed. Fine cost in fill contexts 2.02 vs 0.74 own (L=4; cfast 1.97 vs 0.72), 2.45 vs 0.38 (L=3; cfast 2.23 vs 0.34); rho3/2/1 still 97-100 % identical across contexts, deterministic (log q ~ 0). Bench ground 2x150 s (shared GPU): cresid worst final 6.1e-5 (in budget), F 1.41, 477/707 updates; cfast 2.4e-4 (misses), F 0.86, 803/765 updates; both all dirt, not accepted. Tests pass (new test_cresid, test_cfast_cone, finite support 0). Report: reportables/2026-10-13_cresid/report.md.
Next: check whether bias saturation (deterministic rho early) blocks learning; lower actor lr / slower tau anneal; residual in cconv's candidate head.

## 2026-10-14 00:40 · cresid · started
Approach: per declared count factor, a feature for the children: parent view minus known siblings' view sum, unknown siblings in the subset, parent-unknown flag (generic, from declarations; "more information to learned parts"). Variant cresid = cfast + these features. Gate: clamp eval at L=4/3.

## 2026-10-14 00:45 · clamp · finished
Approach: clamp diagnostic: levels >= L clamped to probe-lifted (fill-following) coarse states vs own coarse draw, same noise; cfast 150 s on ground (1357 updates, shared GPU) and toy_modes; untrained reference.
Result: hypothesis confirmed, gap created by training. Ground: trained fine cost below L=4 is 1.97 nats/cell in fill contexts vs 0.72 own (+1.26); below L=3 2.23 vs 0.34; the probe's own soft energy is 0.014, so fine KL >= ~1.96/cell. Untrained: no gap (2.32 vs 2.47). Gap is count-prior soft energy; fine violations rise only to 0.3-0.5 %/level, forced fractions are unchanged. Trained rho3..rho1 draws are bit-identical whatever the parent (context-blind); the count prior is visible only to the last child. toy_modes: fine KL 0.36 nats/block in non-mode contexts (untrained 1.00), 0 in modes; the coarse level over-selects modes (0.70 vs p* 0.55). Report: reportables/2026-10-13_clamp/report.md.
Next: give the fine learned parts the parent-count residual (parent rho minus known siblings, number unknown) and use this clamp test as the gate.

## 2026-10-14 00:08 · clamp · started
Approach: test the fill-avoidance hypothesis directly: cfast trained 150 s on ground (GPU, shared) and toy_modes; training pass with levels >= L clamped to probe-lifted (fill-following) coarse states vs the model's own coarse draw, same noise; per-level cost, violations, forced / no-option fractions, trained vs untrained. Diagnostic only, no variant changes.
Result: pending.
Next: report in reportables/2026-10-13_clamp/.

## 2026-10-14 · cbig · finished
Approach: compile cache + autotune, test fix, large-B bench (reportables/2026-10-13_cbig/report.md).
Result: test_cbig 5 passed. Warm cache cuts toy_support B=256 setup 196 s -> 11 s (autotune 0 only ~10 % cold). Ground (GPU shared): cfast B=8 1040/1543 upd F 0.92/0.74; cbig B=64 922/1094 upd F 0.55/0.54; B=256 184/230 upd F 0.65/0.79; all collapse to all dirt regardless of B. toy_support: cbig256 F -0.455/-0.467 vs cfast -0.424/-0.444 (F* -0.4688), none accepted. SNR at init ~0 for B=8..256.
Next: SNR on a partly trained actor; large B only pays where step cost is flat.

## 2026-10-13 · cbig · started
Approach: round-3 pass on cbig: compile time (shared cache, autotune_level=0), fix test, bench B=64/256 vs cfast B=8 on ground, toy_support B=256, SNR vs B if time.
Result: running.
Next: bench.


## 2026-10-13 23:55 · exactopt · started
Approach: ideas.md #10 round 3. Collect toy_support / tiny exact runs; representation study on the toy_support block (LN+SiLU, wider/deeper, lr) with exact gradients; information study (wider actor window, same-step independence) on toy_modes / tiny. CPU only, cores 40-47.
Result: pending.
Next: verdict per problem in reportables/2026-10-13_exactopt/report.md.

## 2026-10-13 23:52 · creplay · started
Approach: finish creplay's importance-weighted refactor (coupled coarse replays, ideas.md coupled-replay item): at branched levels keep t1 for every draw, drop every score term t2, add (N_c/R)·Σ_selected c_i (enumeration or LOO). Gradcheck on CPU, then time R and bench creplay vs cfast on the GPU (toy_support, ground).
Result: pending.
Next: GPU bench, coarse SNR before/after.

## 2026-10-14 00:55 · baselines · finished
Approach: fresh GPU baselines (shared GPU, 2 seeds x 150 s): greedy and cfast on all five problems, cpass on toy_support and ground.
Result: nothing accepted. cfast final viol: toy_support 0, toy_modes 0, ground 1.0e-3 (rerun; first run seed 0 got 0 updates), dimers 2.0e-4, columns 7.3e-2. cpass toy_support 2.3e-3; cpass ground crashes on GPU (assertion in cpass.py:256, float32 branch-replay mismatch). ctrunc_full not run. Draws: ground all/mostly dirt, columns all air, dimers lose diagonal. Report: reportables/2026-10-13_gpu_baselines/report.md.
Next: look at the cpass float32 assertion; rerun starved seeds (cfast columns seed 0, 283 updates) when the GPU is quieter.

## 2026-10-13 23:51 · baselines · started
Approach: fresh GPU baseline board: greedy, cfast on all 5 problems; then cpass (toy_support, ground), ctrunc_full (ground). GPU shared.
Result: pending.
Next: report in reportables/2026-10-13_gpu_baselines/.

## 2026-10-13 · cconv · stopped early
Approach: ideas.md #5 on cfast: per-level conv trunk (3x3 stem + residual pre-LN SiLU 3x3 blocks, fixed radius r), dense each step; per-channel linear + candidate-scoring head (s(t), nv, lookahead, value embedding); loss re-evaluates the trunk on (2r+1)^2 patches at subsampled sites; critic on stop-grad trunk features; cone and margin from the conv window. Files: castlegen/variants/cconv.py, _cconv_core.py, tests/variants/test_cconv.py, reportables/2026-10-13_cconv/.
Result: implementation done; 15/15 tests pass on the box (dense == patch trunk, loss recomputes pass log q, scoring, changes-inside-cone on all 5 problems, open windows agree on toy_support and ground). No bench runs yet (session ended). Box note: pids.max 2304 and each jax process uses about 300-500 threads; run under `taskset -c <8 cpus>` (44 threads) or compiles die with pthread_create / ptxas errors.
Next: bench cconv vs cfast on the box (toy_support gate, dimers, ground, columns), B=64, r=4, under taskset.

## 2026-10-13 · crb · stopped early
Approach: ideas.md #8 on cfast: cone returns use each later draw's expected cost lbar_j = sum_a q_j(a) l_j(a) instead of its realised cost (tower-property unbiased).
Result: gradcheck passes (proj 1.017/0.989; negative control rejected z=10/-31). toy_support variance at frozen params: no reduction (ratio 0.99-1.01 at init; 0.99/0.95/0.73 at l0/l1/l2 after 1500 updates with trained critic). No benches run (stopped). [Report](2026-10-13_crb/report.md).
Next: ground variance check before any bench; joint (per-step) RB or combine with creplay. Box note: pids limit -> use taskset.

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
Result: no bench. Replay faithfulness tests pass (4 problems). Gradcheck passes in all 4 configurations (enumerated, LOO, random subsets, wrong critic; pre-refactor code). Ground on the GPU: 4096 replays/update = 6.1 s/update and 222 s compile, so it does not fit the budget as configured. Per branched draw, the across-action spread (~30) is ~1/35 of the return noise (~1000-1500): about 10^3x variance reduction per draw. Code is mid-refactor (importance-weighted subset estimator: t1 for all draws, (N/R)·Σ coupled terms for a random subset), so it does not run yet. Fill-context clamp diagnostic written (_creplay_clamp.py), not run.
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

# recur: recursive (K-sample) coarse training, amendment 2026-10-16 (b) (agent recur, overnight 2026-10-16)

Question: if the coarse levels (L3, L4) of ground are trained with the K-sample recursive cost −log Ẑ_K, using cfitpen_ml2's fitted fine levels (L0-L2), does coarse starvation at L3 / L4 go down, without losing validity (with the stage) or structure?

Objective statement (constitution, "which objective produced a result"):
- Fine levels (L0-L2): cfitpen_ml2's own objective, unchanged. That is weighted ML on the kernel-refined own-draw population (kernel-route amendment) plus pen-only steps (change request #2), restricted to L0-L2.
- Coarse levels (L3, L4): the amendment (b) recursive cost. This is a biased experiment under constraint 3.
- Generation is unchanged (the coloured pass, plus the fixed stage R 30 / F 150 where stated).
- l0 = 3: L3 and L4 are "coarse". Reason: starvation sits at L3/L4, and the L3 footprint (8×8 fine cells, 84 fine draws) is the largest subtree whose importance weights are still usable (below).

GPU 0 was shared (the longfit agent on GPU 0 until ~09:10; other agents on GPU 1). nvidia-smi before/after each arm is in `runs/sweep/<arm>/`.

## Code
- `castlegen/variants/_crecur_core.py`:
  - `split_pass`: cfast's train_pass with the noise of levels ≥ l0 keyed by a coarse seed and that of levels < l0 by a fine seed. K passes with one coarse seed therefore give K fine completions of the same clamped coarse state; this is asserted.
  - An optional clamp of the coarsest level, used for ideas #14 contexts.
  - `foot_sums`: the per-coarse-cell subtree cost.
  - `build`: the jitted recursive step.
- `castlegen/variants/_crecur_exact.py`: exact enumeration and checks on the two-level problem.
- `castlegen/variants/crecur.py`: the variants `crecur` and `crecur_stage`.
- `tests/variants/test_crecur.py`: 5 tests, all pass, CPU, 94 s.

The coarse gradient per coarse draw i is cfast's all-action term on its own cost (soft + λ nv + log q) plus log q(a_i) · (G_i − b_i). The baseline b_i is the leave-one-out mean of G over the other worlds of the batch at the same (channel, colour, site), which makes it a zero-mean control variate.

## Step 1: estimator and guard (1), exact check (CPU)

Problem: `_gradcheck.tiny_problem` (T = 1; 16 binary fine cells; 4 coarse cells with 3 values), λ = (2, 3).
- 56 reachable coarse states; 44 614 reachable outcomes.
- F_supp(c) = −log Σ_{t: q(t|c)>0} exp(−E_fine(c,t)) is the subtree free energy over the pass's support, which is what any estimator with q-proposals converges to.
- F_all(c) sums over all 2^16 t. It lies 0.4-0.8 nats below F_supp at λ = (2, 3): the lookahead masks exclude violating fine states. This gap shrinks to ~0 at λ = 20.
- R_K is computed through `split_pass`: 3000 coarse draws × 16 fine completions.
- The bias of R_K for K = 1/4/16 uses the first 1/4/16 completions.
- Commands: `scripts/exact_check.py runs/exact_check.json` and `scripts/exact_grad_b0.py runs/exact_grad_b0.json` (CPU, `JAX_PLATFORMS=cpu taskset -c 16-23`).

Mean bias of −log Ẑ_K vs exact F_supp, q(c)-weighted, ± s.e.:

| fine model | exact KL(q_fine‖p*) | K = 1 | K = 4 | K = 16 |
|---|---|---|---|---|
| (a) exact conditionals (t_k ~ p*(t\|c) by enumeration) | 0 | 0 (max \|err\| 9e-16) | 0 (9e-16) | 0 (3e-15) |
| (b0) pass, fine output layer zeroed (prior-only fine level) | 1.547 | 1.586 ± 0.031 | 0.816 ± 0.022 | 0.361 ± 0.015 |
| (b1) pass, random fine parameters | 5.161 | 5.153 ± 0.045 | 5.061 ± 0.046 | 4.785 ± 0.048 |
| (b2) pass, random fine parameters, output ×3 (over-confident) | 5.255 | 5.255 | 5.255 | 5.253 |

- The bias falls with K and is 0 with exact fine conditionals.
- At K = 1 the bias equals the exact KL, as it should (E[R_1] = F + KL).
- With a bad fine model K = 16 barely helps (b1).
- With an over-confident, near-deterministic one, K does nothing at all (b2): all K completions coincide.

The bias depends on the context. With (b0), per coarse state:
- c = (1,1,1,1): KL 0.82; bias 0.89 / 0.40 / 0.17 at K = 1 / 4 / 16.
- c = (1,0,0,0): KL 2.36; bias 2.43 / 1.35 / 0.56.

So at K = 1 the second coarse state is penalised 1.5 nats more than the first purely through fine misfit. At K = 16 the spread is 0.39. This is the starvation mechanism the amendment warns about, and K shrinks it by about 4×.

Coarse-gradient check, model (b0), 250 batches × 16 worlds:

| K | vs exact ∇ joint J (reverse KL) | vs exact ∇ marginal J_m = Σ_c q(c)(E_c + log q(c) + F_supp(c)) |
|---|---|---|
| 1 | proj ratio 0.91, proj z −0.87, cos 0.96 | cos −0.28 |
| 4 | proj ratio 0.51, cos 0.87 | cos −0.01 |
| 16 | cos −0.06 | proj ratio 0.49, cos 0.89 |

- At K = 1 the estimator is the unbiased joint reverse-KL gradient. This is also checked with (b1): proj z 0.70, Hotelling p 0.45.
- As K grows, the gradient rotates from the joint gradient towards the marginal (true coarse) gradient.
- At K = 16 the gradient is still about halfway there: proj ratio 0.49 on ∇J_m.
- On this problem the joint and marginal gradients are anti-aligned (cos −0.50). The fine misfit dominates what the coarse level is told.

### The literal (whole-world) form is degenerate on ground
Measured on ground with cfitpen_ml2's params (CPU, B = 2-8, K = 4-16):
- Whole world: the log importance weights of whole-world fine completions under one coarse state have a std of **47-79 nats** across replicas. Normalised ESS = 1/K (0.062 at K = 16). Ẑ_K is the single best completion, and R_K ≈ min_k cost_k + log K. This is best-of-K, not a better free-energy estimate.
- The dependency cone of an L4 draw is the whole world. That of an L3 draw is ~45-70 % of it. Cone-local R_K is just as degenerate (ESS 0.25 = 1/K at K = 4).
- Per-coarse-cell subtree (the footprint: fine draws of L0-L2 under the cell): log-weight std **2.9-3.5 nats at L3** (ESS 0.46-0.47 at K = 16, about 7 effective samples) and **7.2-8.5 nats at L4** (ESS 0.18-0.20, about 3).

The training therefore uses the per-cell subtree form (`rec_mode = "footonly"`): G_i = coarse cone return (coarse draws only) + R_K(footprint of i). This form has a further bias. It drops the cost change of fine draws outside i's footprint that read a_i (neighbouring subtrees through the parent window and the cross-block factors). That truncation is a bias even at K = 1, of the ideas #2 / ctrunc kind.

A hybrid, `rec_mode = "foot"`, is also implemented and tested: the K-sample term replaces only the footprint part of the full cone return. It is exactly unbiased at K = 1 (test). But its return std on ground is 100-230 nats, so the K-sample change (2-3 nats per L3 cell) is lost in the noise. It was not used for the sweep.

## Step 2: guard (2), clamp gate for the fitted fine levels
Command: `scripts/clamp_gate.py runs/clamp_gate_fitpen.json fitpen_ml2=<params>` (CPU). 32 seeds × 4 probe lifts; fine cost per fine cell (sum of soft + λ nv + log q of the levels below L, τ = 1):

| params | L = 4 own | L = 4 probe-lift | L = 3 own | L = 3 probe-lift |
|---|---|---|---|---|
| cfitpen_ml2 (`arm_cfitpen_ml2/params.pkl`) | −0.074 ± 0.002 | −0.066 ± 0.002 | −0.049 ± 0.002 | −0.070 ± 0.001 |
| cfast (2026-10-14 reference) | 0.72 | 1.97 | 0.34 | 2.23 |

- The fitted fine levels show no large context-dependent gap, unlike cfast.
- At L = 4, probe contexts cost +0.008 per fine cell more than own contexts. That is about +1.9 nats per L4 block, most of it at L3 (−0.017 vs −0.025 per cell).
- At L = 3, probe contexts are cheaper than own contexts (−0.070 vs −0.049).

So the recursion should not starve probe-like L3 contexts through fine misfit. At L4 a ~2-nat bias against probe blocks remains, and K = 16 (L4 ESS ~3) removes only part of it.

(Steps 3-4 below.)

## Interim (10:15 UTC, 2 of 5 arms done)
Sweep: `scripts/sweep_recur.py runs/sweep 1800`. Arms back to back on GPU 0, each started from `arm_cfitpen_ml2/params.pkl` with λ kept (λ control unchanged).
- Fine levels keep training as in cfitpen_ml2; they are not frozen in any arm.
- Recursion arms: 8 recursive steps per iteration, B = 8, footonly mode, lr 1e-3.
- Final evaluation is the bench evaluation with the stage (16 worlds), plus the full v2 block check (128 own worlds, 32 probe seeds).

| arm | iters / ML upd / rec upd | starved % L2 / L3 / L4 (probe, direct) | rho L3 / L4 | raw viol L0-L4 → staged | lm staged (raw) | rough | F | first-dirt offset / \|c1\| |
|---|---|---|---|---|---|---|---|---|
| control (cfitpen_ml2 cont.) | 518 / 8288 / 0 | 7.8 / 22.1 / 47.4 | 0.38 / −0.25 | 7.6e-5, 2.1e-4, 6.1e-5, 0, 0 → 0 | 0.660 (0.633) | 1.50 | −0.149 | +0.34 / 0.58 |
| K = 16 | 321 / 5136 / 2569 | 6.5 / 26.7 / **97.6** | 0.31 / −0.46 | 6.1e-5, 7.6e-5, 0, 0, 0 → 0 | 0.639 (0.599) | 1.45 | −0.137 | −0.01 / 0.91 |

- Recursion at K = 16 does not reduce starvation.
- L4 collapses: 98 % of probe L4 blocks are starved, against 47 % for the control. L3 is slightly worse: 27 vs 22 %.
- Validity holds; structure moves (offset to ~0, |c1| up to 0.91, line_match down).
- During training the subtree ESS rose (L3 0.3 → 0.6) and the subtree cost per L3 cell fell (+3.8 → −6.6 nats). The coarse levels moved towards contexts where the fine completions are cheap and certain. That is the mode-seeking behaviour of reverse KL on the coarse level with a lower-biased subtree term, not coverage.
- λ_0 and λ_1 decay to 0 in both arms (λ control: raw violations stay below ε). This is the same in the control.

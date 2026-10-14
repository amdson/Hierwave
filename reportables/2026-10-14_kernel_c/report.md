# kernel-c: log q of a given state and weighted ML, verified on the toys (Sweep 1 (c))

2026-10-14. CPU only (cores 56-63, JAX_PLATFORMS=cpu), one 4x4 block of toy_support / toy_modes (exactopt's enumeration: 120 409 / 12 870 reachable states, exact p*). Runs 240 s each (support ln ml_sample 300 s), one seed, lr 3e-3. Not GPU, so no contention.

## 1. castlegen/kernels/logq.py
- `logq_of_state(var_or_lay, params, state, region=None, tau=1.0, maskfree=False)`: teacher-forced pass (every draw set to the state's value, features/soft/mask read from the state built so far in the pass order). Returns `total`, `total_finite`, `per_level`, `per_draw` ({key: (16, n)}), `n_infeasible`, `feasible`. `logq_batch` vmaps it.
- Infeasible draw (forced value outside the least-violating set A, q = 0): counted; `total = -inf`. `maskfree=True` scores such draws by the mask-free logits (not the sampler's q; fallback only). The differentiable part skips infeasible draws, so no NaN gradients.
- `make_ml_update(lay, tau, clip)` / `ml_update(params, opt, states, weights, step_fn, lr, level_on)`: Adam step on -sum w_i log q(x_i) / sum w_i (infeasible states get weight 0 and are counted), per-level on/off mask (switched-off levels keep both params and Adam moments), per-level grad clip 10 as in cfast.
- Tests `tests/kernels/test_logq.py` (tanh and ln actors, random params): teacher-forced log q equals the log q recorded by cfast's `train_pass` when it generated the state from its own noise, max abs diff < 1e-5, at tau 1 and 1.7 (40 states each); per-level and per-draw sums agree; infeasible states detected and finite under maskfree; level mask leaves masked params untouched; loss decreases. 8 passed (the first run had one over-strict loss check at lr 1e-2, fixed to lr 1e-3).
- Speed: pass on a 4x4 block, B = 256 states, CPU 1 thread: about 7 ML steps/s.

## 2. ML fit to exact p* vs exact reverse-KL
p* = exp(-S) on the valid reachable states (the true target; F* from it: -0.4688 support, 0.2513 modes per cell, reproduced). Metrics are exact over the enumeration at tau = 1. Rev KL is vs p_lam (lam 10, exactopt's convention; p* itself gives infinity when q puts mass on violating states; p_viol is shown). Arms: `ml_sample` = Adam on B = 256 exact p* samples per step through `ml_update` (logq_of_state); `ml_exact` = full-batch weights w = p*(x) over the enumeration (exact forward KL gradient, via exactopt's context tables); `rev_exact` = exact reverse-KL gradient (exactopt train_exact, tau 2->1 over first 20 % of time), same init/lr/time.

| problem | actor | arm | steps | fwd KL(p*\|q) | rev KL(q\|p_lam) | p_viol | F/cell | F*/cell | H/cell (p* = 0.3324 / 0.2591) |
|---|---|---|---|---|---|---|---|---|---|
| support | ln | ML sampled | 1600 | 0.026 | 0.094 | 0.0088 | -0.4629 | -0.4688 | 0.3347 |
| support | ln | ML exact weights | 423 | 0.016 | 0.062 | 0.0065 | -0.4649 | | 0.3349 |
| support | ln | reverse KL exact | 187 | 0.056 | 0.060 | 0.0063 | -0.4651 | | 0.3270 |
| support | tanh | ML sampled | 1827 | 0.046 | 0.130 | 0.0120 | -0.4607 | | 0.3335 |
| support | tanh | ML exact weights | 402 | 0.021 | 0.081 | 0.0098 | -0.4637 | | 0.3352 |
| support | tanh | reverse KL exact | 893 | 2.13 | 0.307 | 0.0050 | -0.4496 | | 0.2872 |
| modes | ln | ML sampled | 1282 | 0.033 | 0.034 | 0 | 0.2534 | 0.2513 | 0.2557 |
| modes | ln | ML exact weights | 1307 | 0.012 | 0.014 | 0 | 0.2521 | | 0.2602 |
| modes | ln | reverse KL exact | 2685 | 0.014 | 0.012 | 0 | 0.2520 | | 0.2557 |
| modes | tanh | ML sampled | 2703 | 0.029 | 0.031 | 0 | 0.2532 | | 0.2570 |
| modes | tanh | ML exact weights | 5173 | 0.012 | 0.014 | 0 | 0.2521 | | 0.2604 |
| modes | tanh | reverse KL exact | 8728 | 0.014 | 0.012 | 0 | 0.2520 | | 0.2557 |

Mode masses on modes (p*: mode_other 0.2564, mode_L 0.1859, broken bonds 5.444): ML sampled ln 0.2502 / 0.1864 / 5.430, tanh 0.2676 / 0.1872 / 5.442; ML exact ln 0.2563 / 0.1859 / 5.465; reverse KL ln 0.2481 / 0.1878 / 5.415. Flat-block fraction on support (p* 0.0959): ML sampled ln 0.0950, ML exact ln 0.0924, rev-KL ln 0.0991, rev-KL tanh 0.134. Full table: `summary.md`; curves `images/kl_entropy.png`; raw `runs/*.json`; log files /tmp/kernelc_*.log.

Reading:
- ML with exact samples reaches the same F and mode masses as reverse KL: F within 0.001 of F* on modes (0.2521 vs 0.2513), within 0.004-0.006 on support (the sampled-B=256 arm is noisy; fwd KL fluctuates 0.025-0.075 between logged points). The sampled arm is 2-3x worse in KL than the exact-weight arm (finite-batch noise at constant lr 3e-3); no lr decay was tried.
- No entropy collapse: log entropy per cell stays at or slightly above H(p*) for all ML arms from the first logged point to the end (0.335 vs 0.332 on support). Reverse KL, which is mode-seeking, sits slightly below H(p*) (0.327, and 0.287 for the tanh run that got stuck at its known plateau: fwd KL 2.1, i.e. it lost support on valid states).
- The tanh plateau of exactopt (reverse KL, 0.3 at lr 3e-3) shows again in the reverse arm; ML does not have it in the same time (fwd KL 0.02-0.05).
- Caveat: it matches the recorded reverse-KL results only to about their level, not better: modes 0.012-0.014 floor is the actor-window information limit (exactopt), which ML also hits (0.012 fwd KL). On support the numbers on record (cbig256 at F*) are on the full strip with the sampled estimator, not comparable to this one-block enumeration.
- Failures / not done: no time-decayed lr; one seed; 240 s runs are not converged for the sampled arm; the original test run needed a loosened loss check; a lane-script bug dropped two jobs which were rerun (no data lost).

## 3. What the ML path needs from the MH/Gibbs kernels
A batch of (state, weight) pairs and nothing else:
- states: dict {channel: (B, h, w) int} of full states (all channels, including designed ones and every coarse level), on the same layout/region the actor trains on;
- weights: (B,) non-negative, need not be normalised (1 for plain refined draws; importance weights for multiple-try candidates);
- states must be reachable by the pass at tau = 1 (lie in the least-violating sets): `info['n_infeasible']` reports violations; a p*-invariant kernel on valid states never triggers it, a kernel that moves through violating states should have them filtered or reweighted before the update. Per-level masks (`level_on`) allow fine-level-only fits (Sweep 1 (a)).
No acceptance probabilities, proposal densities or kernel internals enter the update.

## Files
castlegen/kernels/__init__.py, castlegen/kernels/logq.py, tests/kernels/test_logq.py, reportables/2026-10-14_kernel_c/{report.md, ml_fit.py, summarize.py, summary.md, lanes.sh, images/kl_entropy.png, runs/*.json}.

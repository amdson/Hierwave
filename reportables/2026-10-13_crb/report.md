# crb: Rao-Blackwellised cone returns (ideas.md #8). Stopped early.

Files: `castlegen/variants/crb.py`, `_crb_core.py`, `tests/variants/test_crb.py` (5 tests pass on the box; the gradient checks run on CPU), `scripts/gradcheck.py`, `scripts/gradvar.py`.

## Method
- crb is cfast with K = 0.
- Every cone return now uses the expected cost ℓ̄_j = Σ_{a∈A_j} q_j(a)[s_j(a) + λ n_j(a) + τ log q_j(a)] in place of each later draw's realised cost ℓ_j(a_j). ℓ̄_j comes from the soft, nv, mask and logq values the pass already records.
- Draws, the direct all-action term, the critic and the subsampling are unchanged. Returns are stop-gradient.
- `rb=False` (registered as `crb_off`) gives cfast's estimator exactly; a test checks this.

**Why it stays unbiased.**
- Take a draw j > i at a strictly later step. A cone never contains i itself or any draw of i's step.
- ∇log q_i(a_i) is a function of h_j, the state fixed before j's step.
- a_j ~ q_j(·|h_j) with fresh noise, so E[ℓ_j(a_j) | h_j] = ℓ̄_j.
- By the tower property, E[∇log q_i ℓ_j] = E[∇log q_i ℓ̄_j].

## Gradient check
Tiny enumerable problem, 1000 batches × 16 worlds. The table gives the projection of the mean estimate on ∇J (1 means unbiased).

| estimator | l0 | l1 | Hotelling p |
| --- | --- | --- | --- |
| crb, zero critic | 1.017 (z 1.1) | 0.989 (z −0.8) | 0.24 / 0.38 |
| cfast, zero critic | 1.019 | 0.982 | 0.27 / 0.50 |
| crb, wrong critic | 1.016 | 0.936 (z −1.5) | 0.015 / 0.35 |
| negative control (expectation under uniform-over-A instead of q) | 2.67 (z 10) | −4.47 (z −31) | 1e-24 / 1e-157 |

crb's estimate is consistent with no bias, and the check rejects the negative control. Per-world SNR barely moves: crb 0.221 / 0.119 against cfast 0.197 / 0.119.

## Variance at frozen parameters (toy_support)
Paired comparison: same worlds and same loss subsample. B = 8 on the GPU box.

| parameters, critic | level | var ratio (cfast / crb) | per-second ratio | SNR per batch, cfast → crb |
| --- | --- | --- | --- | --- |
| init (critic off = on) | l0 | 1.01 | 0.99 | 0.047 → 0.048 |
| init (critic off = on) | l1 | 1.00 | 0.98 | not measurable (0) |
| init (critic off = on) | l2 | 0.99 | 0.98 | 0.0025 → 0.0024 |
| after 1500 crb updates, critic off | l0, l1, l2 | 1.00 | 1.15 (timing noise) | — |
| after 1500 crb updates, critic on | l0 | 0.99 | — | 0.029 → 0.028 |
| after 1500 crb updates, critic on | l1 | 0.95 | — | 0.100 → 0.094 |
| after 1500 crb updates, critic on | l2 | 0.73 | — | 0.046 → 0.033 |

- **Cost:** about 0.02 s/update for both; the extra arithmetic is negligible.
- **Return spread is the same:** the std of G is 18.9 / 21.5 / 24.0 (cfast) against 18.9 / 21.7 / 24.5 (crb).
- **Means agree:** the paired-difference |d|²/se² is 0.2 to 1.8, so the mean gradients agree.

**Why there is no gain.** Per-draw sampling noise is not the dominant term on toy_support. Most of G's spread comes from contexts h_j, which earlier draws in the cone keep randomising. Rao-Blackwellising each term separately also does not guarantee lower variance: the terms Σ(ℓ_j − ℓ̄_j) are correlated with the later ℓ̄_k, because a_j changes h_k. With a trained critic, crb was worse at level 2 (0.73×), which is consistent with that correlation being negative.

## Not done (stopped by the coordinator)
- The gradient-variance run on ground.
- The benches on toy_support, ground, columns and dimers.
- The fill_err, line_match and dirt numbers on ground.

Nothing is running on the box from this agent.

## Conclusion and next steps
- crb is unbiased and costs nothing, but on toy_support it gives no variance reduction, and slightly negative at level 2 with a trained critic.
- I do not expect a bench gain. A ground variance measurement (`scripts/gradvar.py ground 256 300`) would confirm this before any bench is spent.
- A more promising direction is to Rao-Blackwellise jointly: condition on a whole later step (each cone's step-boxes), or combine with creplay's coupled replays, which remove the downstream context noise that per-term RB cannot.

## Box note
The box hits its cgroup pids limit (2304); each JAX process spawns about 400 threads. The result is "Failed to launch ptxas" and aborts. Running under `taskset -c <8 cores>` cuts a process to about 44 threads. Small gradient checks run fine with `JAX_PLATFORMS=cpu`.

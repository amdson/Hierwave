# exactopt (ideas #10): stopped early, no verdict yet

2026-10-13 · `castlegen/variants/_exactopt.py`, `tests/variants/test_exactopt.py`. The agent's text, saved by the orchestrator.

## Built

- **`Enum`:** every reachable outcome of the pass, from `_gradcheck.reachable_states`. One forced pass records each draw's context and action, with contexts deduplicated per network. Log q for any parameters is then a gather plus a log-softmax.
- **Exact quantities:** J = E_q[S + λH + τ log q], its gradient, the KL against the reachable set, the coarse-marginal KL, and the problem stats under q.
- **Actors:**
  - `mlp`: the cpass actor.
  - `table_feat`: one free logit vector per distinct feature context.
  - `table_hist`: one per (draw, full state before its step). This is the most the draw order and masks allow.
- **Sampled baselines, same init and step count:** cpass's update, and cfast's fused update (with the cone switched off). τ goes 2 → 1 over the first 20 % of steps, with λ = 10.
- **Problems:**
  - the gradcheck tiny problem;
  - toy_support and toy_modes on one 4×4 block (non-periodic H = W = 4, block-local factors, so the per-block target is exact).
- **Sizes:** toy_modes has 12 870 outcomes; toy_support has 120 409.

## Tests

5 of 6 pass:
- context-table log q matches the forced pass to within 1e-4;
- Σq = 1;
- the exact gradient matches finite differences.

The test that the full-history table reaches the target (tiny problem) got KL 3.50 → 0.047 in 600 steps and was still falling. The 2000-step rerun was unfinished at shutdown.

## Numbers so far (toy_modes block, KL per block in nats; initial KL 2.043)

| run | steps | KL |
| --- | --- | --- |
| exact mlp, lr 3e-3 | 150 | 0.90 (mode_other 0.63, target 0.256) |
| exact mlp (smoke) | 20 | 0.81 |
| exact table_feat (smoke) | 20 | 0.59 |
| exact table_hist (smoke) | 20 | 0.57 |
| cpass sampled, B = 8 (smoke) | 20 | 1.79 |
| cfast sampled, B = 8 (smoke) | 20 | 1.43 |

**Early reading:** exact training is well ahead of the sampled estimators. The smoke rows use a 4-step τ anneal, so they aren't comparable with the 150-step row.

## Still running on the box (nohup, with timeouts)

- `/tmp/eo_modes.log` → `/tmp/eo_runs/*.json`: 5 runs × 2000 steps, about 1 h, may hit the 3000 s timeout.
- `/tmp/eo_support_enum.log` → `/tmp/exactopt_enum_support.pkl`.
- `/tmp/eo_test.log`: the pytest rerun.

## GPU box warning

JAX on the GPU failed with "Failed to launch ptxas / pthread_create failed": the container's pid limit (pids.max 2304) is shared by every process, and each default jax process holds about 400 threads. This agent ran on the CPU instead, with `JAX_PLATFORMS=cpu` and `taskset -c` over 4 cores (about 22 threads). Cap threads per process.

## Next

1. Pull `/tmp/eo_runs/*.json`, then run `plot.py <runs> reportables/2026-10-13_exactopt`.
2. Run toy_support (its enumeration is cached) and tiny with `--steps 1000`.
3. Interpret:
   - exact mlp plateaus above table_feat → representation;
   - table_feat above table_hist → information or draw order;
   - exact ≈ 0 while sampled does not reach it → variance.

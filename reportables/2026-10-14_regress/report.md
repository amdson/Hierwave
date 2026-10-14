# regress: full test suite, 2026-10-14

Cores 64-71, CPU only, no code changed. The single `-x` run was not feasible (see below), so the suite was run in pieces.

## Totals (approximate: the suite was split and some pieces re-run)
- Passed ~ 270 (bench, channels, kernels) + 198 (speed kernels, legacy) + ~105 (variants, bitgibbs, gen_speed) ~ 570. Failed: 1 persistent. Skipped: 1. xfailed: 10. Errors: 2 (legacy, missing data).
- Wall time ~ 95 min total (first -x run 45 min to its first failure at 270 tests; whole-suite serial cost is ~75 min of test time).

## Failures
1. tests/variants/test_exactopt.py::test_full_history_table_reaches_target — FAILS, reproducible (2 runs): `assert curve[-1]["kl"] < 0.01` got 0.047 (initial 3.50). Convergence tolerance, 2000 steps at lr 0.1; the test is dated Oct 13 23:42 and not touched since. Not fixed (not a one-line import; judging the tolerance needs the exactopt author). File: castlegen/variants/exactopt (agent exactopt).
2. tests/legacy/test_exchain.py (2 tests) — ERROR at fixture: `IndexError`, no `cache/exemplar-cliffs-*.npy` in cwd. Missing legacy data file, unrelated to tonight's edits.

## Transient (passed on rerun, not code failures)
- tests/kernels/test_speed_coarse.py::test_compact_equals_reference[toy_support-2]: `JaxRuntimeError: Failed to materialize symbols ... bitcast_gather_fusion` in the first run; passed on rerun (and toy_modes-1 also passed).
- Two pytest processes died with `Fatal Python error: Aborted` inside XLA compile (once in compilation_cache put_executable_and_time during test_cmulti, once in backend_compile_and_load during test_cstart/test_gen_speed). The reruns of the same files passed (test_gen_speed 6/6, cmulti..ctrunc, bitgibbs). Likely shared-cache/compile contention or memory pressure on the shared box, not a test problem. Worth knowing: concurrent pytest runs against the shared JAX cache are flaky.
- No edits to explore.py, ckernel*.py, _cpass_core/_cfast_core caused failures that I could see: all of tests/kernels (incl. test_explore, test_block_mh, test_fine_block, test_logq) and tests/variants other than exactopt pass.

## 15 slowest (first run, tests/bench + channels + kernels)
204.7s channels/test_sampler.py::test_adm_identity
174.6s kernels/test_fine_block.py::test_invariance_numpy[3-offset2]
100.2s kernels/test_logq.py::test_infeasible_and_maskfree[tanh]
93.3s channels/test_sampler.py::test_wrapper_identity
90.9s channels/test_circles_biome.py::test_masks_random_families_run
86.6s kernels/test_speed_coarse.py::test_compact_equals_reference[toy_support-2]
82.5s channels/test_circles_biome.py::test_forward_sampler_matches_old
80.4s channels/test_treeland.py::test_forward_valid_and_attached
72.7s kernels/test_amortised.py::test_fast_sweep_equals_reference
70.4s channels/test_treeland.py::test_tree_ais_targets_finite
69.3s kernels/test_exact_block.py::test_full_block_gibbs_reproduces_exact[toy_modes]
69.2s kernels/test_explore.py::test_annealed_candidates_unbiased[toy_support]
59.2s kernels/test_logq.py::test_infeasible_and_maskfree[ln]
56.2s kernels/test_block_mh.py::test_exact_start_no_drift_and_power
53.7s channels/test_sampler.py::test_hard_rows_first
Slower elsewhere: legacy/test_generic.py::test_mh_batch_keeps_pibar_invariant 180s, test_mh_keeps_pibar_invariant 114s.
No single file exceeded 10 min except channels (about 20 min total).

Logs: pytest.log, pytest_rest.log, pytest_variants*.log, pytest_v4.log, pytest_gen.log in this folder.

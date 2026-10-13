# Gradient check: cpass's estimator is unbiased; variance is the problem

2026-10-13 · `castlegen/variants/_gradcheck.py`, `tests/variants/test_cpass_gradient.py` (7 tests pass, about 2 min). The agent's text, saved by the orchestrator.

**Verdict.** On an enumerable problem, the mean of cpass's actor gradient equals the exact ∇J within sampling noise (1–2 %) in every configuration. The entropy loss and mode collapse are not estimator bias. The gradient's signal-to-noise ratio is tiny: about 0.01 per batch on toy_support, so each update is roughly 99 % noise.

## Setup

- **Test problem:** a two-level periodic strip (T = 1, H = 2, W = 8).
  - 16 binary fine cells and 4 coarse cells with 3 values.
  - Hard rules: support, no adjacent dirt (this crosses blocks and traps the lookahead), count bands, no adjacent c = 2.
  - P(violation) under q = 0.61.
- **Exact answer:** 44 614 reachable outcomes, enumerated. J and ∇J come by differentiating the forced-pass log q.
  - Σq = 1.0000001.
  - A finite difference gives 36.38 against |∇J| = 36.25.
- **Settings:** τ = 1.5, λ = (2, 3), random nonzero parameters, 2000 batches of 16 worlds per configuration.

## Results (projection of the mean gradient on ∇J, divided by |∇J|)

| configuration | fine (l0) | coarse (l1) | worst \|z\| |
| --- | --- | --- | --- |
| zero critic, K = 4 (branches enumerated) | 0.993 | 0.996 | 1.0 |
| coupled-branch term alone, K = 4 | – | 1.010 | 1.1 |
| zero critic, K = 2 (sampled, leave-one-out) | 1.006 | 0.995 | 0.4 |
| coupled-branch term alone, K = 2 | – | 1.026 | 1.4 |
| wrong fixed critic, K = 4 / 2 | 0.983 | 0.998 / 0.969 | 1.8 |
| trained critic, K = 4 / 2 | 1.009 | 1.005 / 1.007 | 1.3 |

- **Multivariate tests:** whitened T² tests over 17 directions pass (p from 0.02 to 0.95).
- **Sampling matches scoring:** the pass samples exactly the q that forced scoring assigns (z = 0.5 on 205 k worlds).
- **Negative control:** returns truncated to the next 4 steps are rejected, so the test can detect bias.

## Signal-to-noise (zero critic)

| case | SNR per world | SNR per draw |
| --- | --- | --- |
| tiny, W = 8 | 0.25 / 0.18 | 0.016 / 0.046 |
| tiny, W = 32 | 0.13 / 0.05 | 0.002 / 0.003 |
| toy_support at init (l0 / l1 / l2) | 1.3e-3 / 7e-4 / 1.2e-3 | 2.5e-6 / 6e-6 / 4e-5 |
| dimers at init | below ~6e-4 | – |

- **Cause:** each draw's return sums every later cost in the world, about 700 terms on toy_support. So the noise grows with world size while each draw's signal stays local.
- **Critic:** a trained critic doubles the coarse-level SNR on the tiny problem (0.16 → 0.36) and does nothing for the fine level. A wrong critic cuts the SNR by 3–10×.

## Minor: gradient clipping

Clipping each level's gradient to norm 10 binds on some batches:
- **toy_support:** actor 1.7 % (l0); critic up to 12 %.
- **dimers:** actor 5.2 %; critic 44 %.

This slightly biases the expected update, which the spec allows. It also starves the critic on dimers.

## Next

Variance, not bias:
- dependency-cone returns;
- a critic that helps the fine level;
- larger B;
- give the critic more room.

## Files

- `castlegen/variants/_gradcheck.py`, `tests/variants/test_cpass_gradient.py`.
- `cpass.py`: one added line exposing `lay.actor_loss` and `lay.critic_loss`; behaviour unchanged.
- Scripts and data in this folder:
  - `run_gradcheck.py` → `results.json`
  - `split_check.py`
  - `sampling_check.py`
  - `snr_scaling.py` → `snr_scaling.json`
  - `clip_rate.py`
  - `diag_*.py`

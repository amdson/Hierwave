# Coloured-pass sampler: constitution

Exported 2026-10-13 from the user's doc
(https://claude.ai/code/artifact/f76368c5-8d7c-44b3-86b8-f2292232e69a, rev 53).
The doc is authoritative; if they disagree, ask the user.

## For a newcomer (not part of the doc)

**Start here, in this order:**
1. This file.
2. `notes/bench.md`: the harness, the bench command and the acceptance rules.
3. `notes/agent_brief.md`: practical rules for agents.
4. `results/board.md`: current scores.
5. `reportables/2026-10-13_*/report.md`: round-1 findings.
6. `reportables/ideas.md`: the user's ideas.

**Approved exception (2026-10-13):** truncated-depth (biased, bootstrapped) returns, ideas.md #2, are allowed as an experiment despite hard constraint 3.

## Purpose

Build a learned, fixed-step, local sampler for hierarchical discrete grids that rarely violates hard factors and stays diverse. It is trained by reverse KL on its own complete draws, with no training chain. This page decides whether a change is valid; the spec decides how things are done, and this page wins any conflict.

## Goals

1. **Low error.** Fewer than 2 × 10⁻⁴ violated hard instances per level after repair, measured on independent noise.
2. **Diverse output.** Minimize soft energy plus τ × log q at τ = 1, which is reverse KL to the Gibbs target exp(−S − Σ λ_l H_l). Exact sampling from p* is not required.
3. **Most modes maintained.** Mode mass matches on enumerable tests, and valid states from other sources are not starved (coverage probe). True mode coverage may be intractable, so this is a soft, aspirational goal.
4. **Fixed generation cost.** A fixed number of coloured passes per level (several allowed), plus a fixed repair budget. Total work per site is bounded. Generation never stalls.
5. **Runs on GPU.** Batched over sites, worlds and replays.

## Hard constraints

A change that breaks any of these is invalid, however well it scores. Cite them by number.

1. **Generic.** Every step, region, mask, colouring, lookahead and feature is derived from the factor declarations and the hierarchy. Nothing is chosen because of a particular scenario's constraints or a failure analysis of one.
2. **Local, fixed-step generation.** Every sampling step reads only a fixed, finite window of values fixed before it, plus its own noise. Every pass, level and repair round has a fixed count, so each output site depends on finitely many noise values and designed inputs. That finite support is what lets any chunk of an infinite world be generated on its own. No global checks, rejection loops, variable-length search or unbounded propagation are allowed. The training-only critic is exempt, because it never runs at generation time.
3. **Unbiased gradient of the stated objective.** No truncated returns, bootstrapped values or dropped cost terms. A critic or baseline may only act as a control variate whose expected contribution is zero.
4. **The objective is fixed.** Costs are the designed energies plus λ_l × violations plus τ × log q, computed on unrepaired draws. Coarse costs are never rescaled by level, and fine-level costs never leave coarse returns.
5. **No joint-domain tables.** Learned outputs are full-domain logits per channel. Designed energy is an offset; anything a lower level induces at a coarser one is learned, not hand-written.
6. **No hand-written bounds or rules** linking channels beyond the declared factors. Hard factors appear only as declared tables, the lookahead derived from them, the penalty and repair.
7. **Penalty is bounded.** 0 ≤ λ_l ≤ 20. A missed budget is reported, never hidden by a larger penalty or a longer repair.

## Allowed freedoms

These are valid without asking, as long as the hard constraints hold.

- **More information to learned parts.** Wider symmetric windows, known flags, boundary summaries on all sides, current τ and λ as inputs.
- **Wide training-only inputs.** The critic may read any value fixed before its draw, including the whole level. It never reads the draw itself, same-step draws or later states.
- **Unbiased variance reduction.** Coupled replays with shared noise, leave-one-out baselines, branching at any level, enumerating small candidate sets, larger B or K.
- **Engineering.** Network sizes, optimizer, learning rates, clipping, τ schedule, batch shapes, fp32 on GPU and incremental replay. Numba stays as the reference that the tests check against.
- **Defaults.** Every number in the spec's defaults table may change, except the acceptance budget and the λ cap.

Each freedom is logged with its measured effect.

## Amendment 2026-10-14: kernel-route training (user decision)

Adopted by the user on 2026-10-14 after the reverse-KL-only route stalled on ground (see `notes/methodology.md` §6 and `notes/decision_2026-10-14_kernel_route.md`). It is frozen for the duration of the two kernel-route sweeps.

Constraints 3 and 4 are relaxed **for training only**, as follows.

- The training update may be the gradient of a weighted maximum-likelihood objective, −Σ_i w_i log q(x_i), over states x_i produced by applying p*-invariant Markov kernels to the sampler's own draws, and over multiple-try candidates with their importance weights. Because KL(qK ‖ p*) ≤ KL(q ‖ p*) for any p*-invariant kernel K, this is an improvement direction for the stated goal without requiring the chain to mix.
- Admissible kernels: exact block Gibbs on small blocks given their boundary; Metropolis-Hastings subtree resampling with the sampler (optionally tempered) as the proposal, including multiple-try; coarse-cell flips with the sampler redrawing the subtree, with delayed acceptance using the coarse-level factors as a screen; perturbation of the sampler's latent noise; annealed block proposals. Each kernel must be defined only from the declared factors, the hierarchy and the sampler itself: no problem-specific moves (constraint 1 applies unchanged).
- Hard violations inside a kernel's acceptance ratio are priced at a fixed λ ≤ 20 or treated as infeasible; the choice is logged.
- Generation is unchanged: the fixed-step local pass, never a chain (constraint 2 applies unchanged). Constraints 5, 6 and 7, and all invariants, apply unchanged.
- Reverse-KL training remains valid and may be mixed with the above; which objective produced a result is stated in every report.
- New diagnostics are admitted as evidence: per-kernel, per-level acceptance rates, and where the kernels move mass in the sampler's own draws.

## Invariants checked in code

These are asserted or tested on every change. A failure blocks all experiments until it is fixed.

- **Cost identity.** Σ_i ℓ_i(a_i) = C(x), to floating-point tolerance.
- **Never stalls.** Every valid state has a finite forced probability.
- **Finite support.** The compiler computes each output site's dependency radius through every pass, level and repair round, and asserts that it is a fixed constant. A test perturbs the noise outside that radius and checks the site does not change.
- **Normalised.** On enumerable cases, probabilities sum to one.
- **Replay is faithful.** Draws outside D_i reproduce bit-identically. Gumbel noise is stored per domain value, not per candidate.
- **Batched equals sequential.** The two agree under fixed noise, including at seams.
- **Gradient matches.** On a tiny enumerable hierarchy, the estimator's mean equals the exact ∇J, with zero, accurate and deliberately wrong critics.
- **Repair never increases** total violations at any redraw.

## What counts as success

A run succeeds only if all of these hold on noise it was not tuned on:

- The repaired violation budget is met at every level.
- Mode mass agrees on enumerable tests within the declared tolerance.
- No probe set is flagged.
- Structural statistics of raw and repaired output agree with each other and across independent training seeds.

These do **not** count as success: stable training curves, low gradient variance, low loss, good-looking images or a good result on one seed. Results are reported per level, with metrics and an image saved in `images/`. Failures and skipped steps are reported as such.

## Process

- Only the user changes this page. Claude proposes changes and never applies them silently.
- Anything outside the allowed freedoms needs the user's approval before it is built, even as an experiment. That includes biased estimators, truncated returns, special-case moves and a different objective.
- When a result tempts a shortcut, write the shortcut down as a proposal and cite the constraint it would break.
- The training chain (i-SIR, certland) is superseded. Its code stays only as a test reference and a source of probe states.

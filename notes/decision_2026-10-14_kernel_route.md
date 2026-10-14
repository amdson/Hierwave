# User decision, 2026-10-14 (box session, second round): the kernel route

**Decision (user, in chat):** run two research sweeps on amortised-MCMC training of the coloured-pass sampler, given that exact sampling of p* is intractable.

**Premise.** Any Markov kernel K that leaves p* invariant satisfies KL(qK || p*) <= KL(q || p*). So a few MCMC steps applied to q's own draws give a distribution closer to p*, and fitting q toward those refined states (weighted maximum likelihood = forward KL on refined samples) is a monotone improvement direction. No mixing is required. Training consults only the target's energy (still ab initio). Generation is unchanged: the fixed-step local pass.

**Kernels in scope** (each p*-invariant; cycles of them are too):
- exact block Gibbs on small fine blocks given their boundary (enumeration, rejection-free);
- subtree resample with q as proposal (optionally tempered, tau > 1), Metropolis-Hastings-corrected, multiple-try (block i-SIR) allowed;
- coarse-cell flip (uniform or tempered q) plus q subtree redraw, with delayed acceptance using the coarse-level factors as a screen;
- noise perturbation (Crank-Nicolson step on the sampler's latent noise) for correlated proposals on larger blocks;
- annealed block proposals (reserve).
Block sizes are set adaptively from acceptance (target 20-40 %). Log acceptance per kernel and per level.

**Proposed constitution amendment (for the user to paste if adopted).** Constraint 3 ("unbiased gradient of the stated objective") and constraint 4 ("fixed objective") are relaxed for training as follows: the training update may be the gradient of a weighted maximum-likelihood objective over states produced by applying p*-invariant Markov kernels to the sampler's own draws (and over multiple-try candidates with their importance weights). The kernels must be generic (defined from the declared factors and the sampler, with no problem-specific moves), and generation remains the fixed-step local pass. Everything else (locality, generic algorithm, no joint-domain tables, no hand-written bounds, bounded penalty) is unchanged.

**Sweep 1 (parallel, verify the pieces):** (a) exact fine-block Gibbs + ML fit of the fine levels on ground, gated by the clamp test and the parent-awareness check; (b) generic block-MH kernel with q as proposal, invariance verified on toy_modes (exact stats); (c) log q of a given state via the pass with overrides, and a weighted-ML update, verified by fitting exact enumerated samples on the toys.
**Sweep 2 (assemble):** the amortised loop on ground: draws -> kernel cycle -> ML update; gate = clamp test, parent-awareness, v2 bench; acceptance per level as the new diagnostic.

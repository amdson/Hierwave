# Ideas, approaches and warnings

The user pastes approaches, ideas and warnings here. Agents read this file
before starting work, and annotate items they tested or know about. Never
edit or delete the user's text; add annotations underneath it.

Item format (number items so they can be cited):

```
## <n>. <short title>
<the user's text>

> **Annotation (YYYY-MM-DD, <agent / variant>):** tried / known / not yet tried;
> what happened, with bench numbers or a link to a report; status: works,
> does not work, mixed, untested.
```

Warnings apply to every agent until the user removes them.


## 1. Temperature or constraint strength conditioned training
What it sounds like. Varying temperature, or softening hard constraints, during training to allow for more experimentation, prevent mode collapse, etc. 

> **Annotation (2026-10-13, ctemp):** tried on toy_support, toy_modes and dimers (variant `ctemp`).
> Setup: actor conditioned on log τ_l; per-update τ ~ log-U[1, 3] at all levels ("mix") or only at non-finest levels ("coarse"), last 50 % of the budget at τ = 1; λ soft starts (λ0 = 1, ramped by the controller).
> Bench (2 × 150 s), ctemp vs cpass:
> - toy_support: KL 0.095 / 0.089 vs 0.096 / 0.092.
> - toy_modes: KL 0.012 / 0.014 vs 0.013 / 0.011.
> - dimers: still one orientation (96 % H / 99 % D).
>
> Findings:
> - Softening works while it lasts. At λ = 1, toy_support stats are exact and dimer orientations stay balanced, but violations are 14 % and 4 %.
> - The entropy goes again as λ rises to push violations under budget, whatever the path.
> - "mix" stalls learning at τ = 1 until the τ = 1 phase.
> - "coarse" improves the fine conditionals in rare contexts (fine KL 0.22 → 0.15 nats/block on toy_modes), but the coarse level is then worse.
>
> Status: does not work as a fix on its own. The bottleneck looks like credit assignment for the draws that set up later violations or entropy. Retry once returns have lower variance.

## 2. Truncated depth forward passes
Extremes are depth one forward passes from arbitrary levels, with a low-variance but biased gradient from the model recursively evaluated on child latents. 

> **Annotation (2026-10-13, orchestrator):** approved by the user on 2026-10-13 as an experiment, although it breaks constitution constraint 3 (unbiased gradient). Being tried in round 1 as variant `ctrunc`. Status: untested.

> **Annotation (2026-10-13, ctrunc):** tried.
> - Method: coarse-to-fine chain with shrinking batches, local returns truncated d levels down, and a per-cell learned value (d-step bootstrapped) below that.
> - d = 1 (pure depth-one bootstrapping) is the worst depth. Ground: F 1.05 against 0.74 (d = 2) and 0.72 (full) and cpass's 1.92. The value is calibrated in the mean but carries little action-relevant signal (corr 0.1-0.7).
> - The useful gain is throughput: 2-5x cpass's updates/s, from dropping coupled replays and from the cheap coarse worlds.
> - No acceptance. Violations are still traded for entropy (all-dirt ground, empty columns).
> - Status: mixed. Use as much realised depth as affordable; the bootstrap is only a fallback.

## 3. Multi-pass sampling within hierarchy level
Get around issues with conditional independence assumptions within same color groups by simply sampling multiple times. May speed things up in net by allowing for fewer colors. In the limit, two color checkerboard would be possible. 

## 4. Masked diffusion sampling of channels (or factorized sampling)
Speed up the #channels sampling steps by switching from autoregressive on channels to masked diffusion within channels, or any other method which samples multiple channels simultaniously. 

## 6. Large ordered output domains: simplest fixes (drafted by the orchestrator, chosen by the user, 2026-10-13)
Counts (up to 257 values) and certificates (153 values) currently get one independent logit per value, and their values come in as a scalar u = v / (n − 1) and its square. Three simple changes:
- (a) **Inputs:** encode large views with a thermometer code over 16 bins, plus a one-hot of the exact value for views with at most 32 values, instead of (u, u²).
- (b) **Output:** logit(v) = Σ_k c_k φ_k(v) + r_v. The network outputs a few coefficients c over smooth basis functions φ (e.g. 8 Fourier or B-spline functions) and a small per-value residual r, so neighbouring values share data.
- (c) **Mask-relative coordinates:** evaluate the basis at u = (v − lo) / (hi − lo), where [lo, hi] is the range of the draw's least-violating set A, so the shape adapts to the feasible interval.

Applies to views and channels with more than 8 values; small domains keep per-value logits.

> **Annotation (2026-10-13, orchestrator):** launched as variant `cordinal` (cpass plus a, b and c, A/B against cpass). Status: untested.

> **Annotation (2026-10-13, cordinal):** tried. Cosine basis (8 terms per ordered view; certificates use their A and B views, counts their id order) plus 0.1-scaled residual, mask-relative u; estimator unchanged. Ground (about 55 updates): worst final viol 0.029 vs cpass 0.070, F 1.85 vs 1.92, far fewer floating cells, but more dirt. toy_support KL 0.082 vs 0.095. Columns neutral to worse (one seed collapsed to empty), confounded: (a) inflates the critic's input 3-5x and columns fit about 90 vs about 150 updates. Ground ablation: (a) alone = cpass (0.090); (a+b) 0.043; (a+b+c) 0.029; (c) within the seed spread. Report text in the orchestrator's 2026-10-13_cordinal folder. Status: mixed. (b) works on ground, (a) does nothing measurable, (c) is probably useful but not shown.

> **Annotation (2026-10-13, Codex / cfast_head):** tested (b) alone on cfast with original inputs/critic and absolute coordinates. Ground, 2 x 150 s: worst final violations 9.16e-5 versus fresh cfast 6.59e-3; F 1.204 versus 0.813. Both head seeds meet the violation budget but collapse to 50%-dirt stripes / all dirt, fail coverage probes and seed agreement. 585/622 updates versus control 926/825. Status: mixed for validity, does not fix collapse. [Report](2026-10-13_cfast_head/report.md). Columns not run; creplay/cbig are already investigating lower-variance training.

## 8. Average downstream costs over candidates (from astra, via the user, 2026-10-13)
In returns for earlier draws, replace each later draw's sampled immediate cost with its expectation under its own q: ℓ̄_j = Σ_a q_j(a)[s_j(a) + λ n_j(a) + τ log q_j(a)]. Keep sampling normally, and change only the return. This is unbiased by conditional expectation (stop-gradient on the return; keep the direct terms). Check against exact gradients, then measure gradient variance per second at frozen parameters; it may complement creplay.

> **Annotation (2026-10-13, orchestrator):** agreed: this is Rao-Blackwellisation of the per-draw sampling noise, which is the dominant variance term. Launched as variant `crb` on cfast. Status: untested.

> **Annotation (2026-10-13, crb):** built and gradient-checked as unbiased (projections 0.99–1.02; the negative control is rejected). On toy_support at frozen parameters it doesn't reduce variance: the cfast ÷ crb variance ratio is 1.0 at init, and 0.99 / 0.95 / 0.73 with a trained critic. The returns' spread comes mostly from downstream state randomness, not from each draw's own sampling noise. Ground not measured; no benches. Status: does not work on toy_support; ground untested. [Report](2026-10-13_crb/report.md)

## 9. Train the critic to minimise actor-gradient variance (from astra, via the user, 2026-10-13)
Instead of return-prediction MSE, fit the critic to minimise the second moment of the corrected actor-gradient estimate. Its mean doesn't depend on the critic, so this minimises variance. Start on the enumerable problem, using separate or previous batches; the acceptance criterion is lower measured gradient variance, not critic MSE.

> **Annotation (2026-10-13, orchestrator):** agreed, later; it has precedent in RELAX (Grathwohl et al. 2018). It needs per-sample gradient norms, so it costs more; queue it after #8 and #10. Status: untested.

## 10. Optimise a tiny case with exact gradients (from astra, via the user, 2026-10-13)
On the enumerable hierarchy, optimise the objective with exact expectations, and compare with sampled-gradient training from the same initialisation:
- If exact training succeeds and sampled training fails, the problem is variance.
- If both fail, look at optimisation, representation or the sampling order.
- If both succeed, increase the difficulty until they separate.

> **Annotation (2026-10-13, orchestrator):** agreed. This separates the variance hypothesis from the "fine levels can't realise the contexts" hypothesis (see the cfast_head report). Launched as `exactopt` (a diagnostic, not a variant), also on a single toy_support block. Status: untested.

> **Annotation (2026-10-14, exactopt): outcome.** The rule's three branches each apply to one part of the gap. Full numbers are in `reportables/2026-10-13_exactopt/report.md`.
> - **Variance** dominates at B = 8. With the same init and step count, exact training beats sampled training (cpass/cfast) 15× on toy_modes (0.013 vs 0.18 nats) and 5–130× on toy_support (0.30, or 0.012 with a better lr, vs 1.56).
> - **Representation** is not a capacity limit. The tanh MLP's 0.30 plateau on toy_support is optimisation: lr 1e-2 or a longer schedule reach 0.012 / 0.007. LayerNorm + SiLU reaches the feature table's 0.008 at the default lr.
> - **Information:** the 4×4 actor window misses cells. A 6×6 window takes toy_modes from 0.012 to 0.0009 (MLP) and 0 (table).
> - **Order:** independent draws within a colour leave a hard floor of 0.037 on the tiny problem, which only more passes or a finer colouring remove.
> Status: done.

> **Annotation (2026-10-14, creplay):** coupled coarse replays, the variance fix that exactopt's verdict points to.
> - Setup: the importance-weighted creplay estimator, branching the top 2 levels with 8 replayed draws × 4 actions per world.
> - Correctness: it passes the exact-gradient check.
> - Variance: per-world gradient SNR on ground at init rises from 0.010 / 0.014 to 142 / 93 at levels 4 / 3, with means unchanged.
> - Bench (150 s): it loses on throughput, at 1.4 s/update vs cfast's 0.07 s, and the first seed spends its whole budget compiling.
> - Ground still collapses to all dirt (201 updates). toy_support seed 1: KL 0.026 vs cfast 0.031.
>
> Status: works as variance reduction; untested as a learner at equal update counts. [Report](2026-10-13_creplay/report.md)

## 7. Low-to-high scheduling of 1-step updates (from chat, 2026-10-13; low priority)
"scheduling low to high 1-step updates. not high priority, just worth considering"

> **Annotation (2026-10-13, orchestrator):** how I read it (the user may correct this): a bottom-up curriculum of depth-1 (truncated) updates.
> - Train the finest level first, from coarse latents.
> - Then move up one level at a time, so each coarser level trains against finer levels (and their values) that have already converged, rather than against moving targets.
>
> It relates to ctrunc: depth 1 there suffered from a weak value function, and converged children might fix that. Not yet tried. Status: untested.

## 5. Convolutional networks and modern non-linearities (from chat, 2026-10-13)
"conv almost certainly have a good prior for this task." Also: replace tanh MLPs (agreed in chat).

> **Annotation (2026-10-13, orchestrator):** queued as variant `cconv`. Plan: a conv trunk of residual 3×3 blocks with LayerNorm and SiLU at a fixed depth (a bounded radius keeps finite support); parent features injected by upsampling; a candidate-scoring head over the site features plus an embedding of each candidate, its closed energy, violations and lookahead slack; a deeper conv critic. A/B against the cpass MLP, gated on toy_support KL. Status: untested.

---

> cconv outcome (2026-10-14, [report](2026-10-13_cconv/report.md)): first GPU bench, 2 x 150 s on a shared GPU. Worse than cfast on all four problems (toy_support F -0.395 vs -0.416; toy_modes 0.29-0.33 vs 0.268; ground violations 0.7-7.6% with no fill line; columns 18% violations), with 5-10x fewer updates (cold compile 130-240 s, 0.6 s/update at B=64). Probably undertrained, not shown to be a worse representation.

## 11. Parent-count residual features for the learned bias (from chat / clamp agent, 2026-10-14)
From the clamp diagnostic: give every child draw, for each declared count factor, the parent's value minus its known siblings' sum and the number of siblings still unknown (allowed freedom "more information to learned parts").

> **Annotation (2026-10-14, cresid):** built as variant `cresid` (generic, off by default elsewhere; tests and finite support pass). It doesn't help at 150 s: fine cost in fill contexts is 2.02 vs 0.74 own at L = 4 (cfast 1.97 vs 0.72), and the rho draws are still context-blind and deterministic. Ground bench: cresid meets the violation budget (6.1e-5) but F is 1.41 vs cfast 0.86, and both are all dirt. Likely cause: the bias saturates (deterministic rho) and training never visits fill contexts. Status: tried, no effect on the gate. [Report](2026-10-13_cresid/report.md)

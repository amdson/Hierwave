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

## 5. Convolutional networks and modern non-linearities (from chat, 2026-10-13)
"conv almost certainly have a good prior for this task." Also: replace tanh MLPs (agreed in chat).

> **Annotation (2026-10-13, orchestrator):** queued as variant `cconv`. Plan: a conv trunk of residual 3×3 blocks with LayerNorm and SiLU at a fixed depth (a bounded radius keeps finite support); parent features injected by upsampling; a candidate-scoring head over the site features plus an embedding of each candidate, its closed energy, violations and lookahead slack; a deeper conv critic. A/B against the cpass MLP, gated on toy_support KL. Status: untested.

---

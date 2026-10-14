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

> **Annotation (2026-10-14, cln): outcome of acting on it.** Selectable actor net (LN + SiLU) and window (6×6) on cfast as variants `cln` / `cln6`, with the cone and finite-support tests passing. In sampled training (B = 8, 150 s, shared GPU) neither beats cfast: toy_modes KL per block is 0.26 (cln6) and 0.16–0.22 (cln) against 0.15–0.16 (cfast); ground collapses to all dirt for all three. Consistent with variance binding at B = 8; the comparison is confounded by 2.5–4× fewer updates under contention (true per-update cost +20 %). See `reportables/2026-10-13_cln/report.md`. Status: done; retry at large B.

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

> **Annotation (2026-10-14, ccur):** tested as a per-level lr schedule on cfast (engineering only).
> - Setup: levels 3–4 frozen at init while levels 0–2 train, then released fine to coarse (L3 at 35 %, L4 at 55 % of 300 s); ground.
> - Result: the order does not change the outcome. While the coarse levels were frozen, the fine levels got worse in fill-following contexts (fine cost per cell 2.80 vs 2.32 untrained at L = 4; 2.64 vs 1.98 at L = 3). The untrained coarse levels are diverse (3 nats) but not fill-following (line_match 0.50, 88 % dirt), so there was nothing fill-like to learn from.
> - After release, the top level went to all dirt within 40 updates (cfast: 175 from init).
> - Bench (300 s, 2 seeds): budget met by both (0 final violations), neither accepted (probes flagged). F 1.23 vs cfast 0.60; line_match 0.50 in both; ccur seed 1 ended in the stripes mode.
>
> Status: tried, no gain on ground. Not applicable to dimers (T = 0). Columns: see the report. [Report](2026-10-14_ccur/report.md)

## 5. Convolutional networks and modern non-linearities (from chat, 2026-10-13)
"conv almost certainly have a good prior for this task." Also: replace tanh MLPs (agreed in chat).

> **Annotation (2026-10-13, orchestrator):** queued as variant `cconv`. Plan: a conv trunk of residual 3×3 blocks with LayerNorm and SiLU at a fixed depth (a bounded radius keeps finite support); parent features injected by upsampling; a candidate-scoring head over the site features plus an embedding of each candidate, its closed energy, violations and lookahead slack; a deeper conv critic. A/B against the cpass MLP, gated on toy_support KL. Status: untested.

---

> cconv outcome (2026-10-14, [report](2026-10-13_cconv/report.md)): first GPU bench, 2 x 150 s on a shared GPU. Worse than cfast on all four problems (toy_support F -0.395 vs -0.416; toy_modes 0.29-0.33 vs 0.268; ground violations 0.7-7.6% with no fill line; columns 18% violations), with 5-10x fewer updates (cold compile 130-240 s, 0.6 s/update at B=64). Probably undertrained, not shown to be a worse representation.

## 11. Parent-count residual features for the learned bias (from chat / clamp agent, 2026-10-14)
From the clamp diagnostic: give every child draw, for each declared count factor, the parent's value minus its known siblings' sum and the number of siblings still unknown (allowed freedom "more information to learned parts").

> **Annotation (2026-10-14, cresid):** built as variant `cresid` (generic, off by default elsewhere; tests and finite support pass). It doesn't help at 150 s: fine cost in fill contexts is 2.02 vs 0.74 own at L = 4 (cfast 1.97 vs 0.72), and the rho draws are still context-blind and deterministic. Ground bench: cresid meets the violation budget (6.1e-5) but F is 1.41 vs cfast 0.86, and both are all dirt. Likely cause: the bias saturates (deterministic rho) and training never visits fill contexts. Status: tried, no effect on the gate. [Report](2026-10-13_cresid/report.md)

## 12. Exploration at higher τ, importance-weighted to τ = 1 (from chat, user, 2026-10-14)
The user: "exploring at higher T is definitely in scope." Draw training worlds at τ_b > 1 (or a mixture of τ values), and weight each draw's gradient contribution by the importance ratio to the τ = 1 objective, so the gradient of the stated objective stays unbiased (constraint 3) while the fine levels keep visiting contexts a deterministic τ = 1 policy would never produce. Also agreed in the same exchange: the bench training budget is 300 s per seed from now on (the session runs autonomously).

> **Annotation (2026-10-14, orchestrator):** motivated by the clamp and cresid findings (fine levels deterministic, context-blind; cresid's extra information unusable because the contexts are never visited). Queued as variant `chot` on the next free agent. Status: untested.

> **Outcome (2026-10-14, chot):** built as variant `chot` on cfast. Hot draws are eps-hot, with coins keyed by the coordinate noise. They sample the whole-logit tempered q^(1/tau_b). Each world is weighted by the product of bounded mixture ratios q/((1-eps)q+eps q_b) over its hot levels. The gradcheck passes at eps 0.3, tau_b 3, with zero and wrong critics; the unweighted control is rejected.
> - At 300 s on ground it does not help. Fill vs own fine cost is 2.20 vs 0.79 (L = 4) and 2.46 vs 0.30 (L = 3), against 1.76 vs 0.63 and 2.25 vs 0.44 for cfast through the same path. The top level is still deterministic by update ~300, and everything is all dirt.
> - Bench, 300 s x 2 seeds: ground, dimers (diagonal still 0.000) and toy_support are all within noise of cfast, and none is accepted.
> - The reason is structural. An unbiased importance-weighted estimator has the same expected update as on-policy training, so contexts q reaches with probability ~0 get weight ~0 however often they are visited. Making exploration change what the fine levels learn needs a biased estimator or a different objective (constraints 3 and 4), which is the user's call.
>
> Status: tried, no effect. [Report](2026-10-14_chot/report.md)

## 13. Fine-level saturation diagnostics and the lr / τ levers (from chat / csat agent, 2026-10-14)
I tested cresid's hypothesis that the fine-level bias saturates early, so training never visits fill-following contexts. I logged per-level entropy, the forced fraction and the spread of b during cfast training on ground, then tried actor lr 1e-3 and a τ anneal over the whole budget (engineering freedoms).

> **Outcome (2026-10-14, csat):** the hypothesis is wrong about the order of events.
> - The dirt collapse starts at the top level. cert4 goes deterministic, and dirt reaches 0.95 at about update 200–250. With lr 1e-3, rho3, rho2 and rho1 still have 2–3 nats of entropy at that point and never go deterministic.
> - Lr 1e-3 keeps the fine levels stochastic, but they stay context-blind: their draws are 95 / 99 / 100 % identical under probe-clamped coarse levels. On the gate, fill vs own cost is 1.58 vs 0.57 at L = 4 and 2.04 vs 0.37 at L = 3 (default: 1.92 vs 1.22 and 2.00 vs 0.93). The slower anneal changed nothing.
> - Every arm is all dirt, with line_match 0.500.
> - τ is a weak lever because it tempers only the soft term, not b (b spread 15–60 nats). Exploration at higher τ (#12) must temper the whole logit.
>
> Status: tried, no effect. A proposal (per-candidate enumeration at fine count draws, plus whole-logit tempering for #12) is in the [report](2026-10-14_csat/report.md).

## 14. Fine levels trained on arbitrary parent states, mixed start depth (from chat, user, 2026-10-14)
The user's argument: reverse KL splits into a coarse term plus the fine conditional KL averaged over coarse states; for any coarse state the optimal fine conditional is the target's, so training the fine levels on parent states from any full-support distribution (target-like lifts of probe states, the fill line plus the count prior, or the model's own draws) has the same minimiser as the stated objective. It also removes the collapse mechanism: the coarse levels minimise the free energy of their own fine levels, which is inflated exactly where the fine levels are bad, so they avoid those states; a fine conditional trained everywhere removes the inflation.

Form: each training world picks a start depth L from a schedule; above L the coarse state comes from a mixture of own draws and target-like states; below L the pass samples as usual with its normal cone-return gradients; full-depth worlds keep training the coarse levels. Generation unchanged.

Constitution note: the fine parameters' gradient is a differently weighted average over contexts, so it is not the unbiased gradient of J as written (constraint 3), though the optimum is unchanged. **Approved by the user in chat as in scope.**

> **Annotation (2026-10-14, orchestrator):** queued as variant `cstart` for the next free Opus agent; gate = the fast loop and the clamp test. Status: untested.

> **Annotation (2026-10-14, cstart): outcome.** Built as `cstart` (probe-lift contexts via `problem.probes`, 50 % clamped worlds, L uniform 1-4; clamped draws masked from loss and returns; frac 0 is bit-identical to cfast). At the default mixture the clamp gate does not move: fill-context fine cost 2.04 / 2.05 per cell at L = 4 / 3 after 300 s vs cfast 1.99 / 2.13; the collapse is delayed (dirt 0.74 vs 0.97 at update 200) but happens by ~450; bench 300 s x 2 not accepted (F 1.63, line_match 0.50, starvation 1.0 at levels 2-4). A 90 %-clamped-at-L=4 arm halves the fill cost (1.16 / 1.25) but the fine levels still ignore where the line runs inside a block (clamped line_match = untrained), and the own draws still collapse. Reweighting contexts helps only with a lot of fill-context data and does not make the fine conditional parent-aware within a block. Status: tried; next, pair with parent-aware fine features (#11) and an early high-clamp phase. Report: reportables/2026-10-14_cstart/.

> **Annotation (2026-10-14, ccombo):** an f = 1, L = 3 phase (35 %), decaying to full depth, with LN 6x6 + whole-logit τ 8→1 + λ ramp 1→20: levels 3-4 lock into dirt within ~300 updates of release and the fine levels then forget fill contexts (L = 3 fill 1.17). With a 0.5 clamped floor kept to the end, fill beats own at the fine levels (L = 3 0.11 vs 0.71, L = 4 0.56 vs 0.97; cfast 2.23 vs 0.34), but own line_match stays 0.500. Status: tried, gate not passed, no bench. Report: reportables/2026-10-14_ccombo/.

## 15. Coupled enumeration at the fine levels (from chat / csat agent, 2026-10-14)
csat's proposal 1: the fine count draws on ground go deterministic and context-blind, and a deterministic policy gets no score-function gradient for actions it never takes. Branch a random subset of non-forced fine draws per world (inverse-inclusion weighted, unbiased), enumerate A when |A| ≤ K with coupled replays scored by the cone, and replay only inside the cone window. Inside the allowed freedoms ("branching at any level, enumerating small candidate sets").

> **Annotation (2026-10-14, cfine agent):** built as variant `cfine` ([report](2026-10-14_cfine/report.md)).
> - **Tests:** windowed replay bit-identical to the full replay (2/2), gradcheck 7/7.
> - **Speed and signal:** the windowed replay is ×12 cheaper at level 0 and ×1.6 at level 1, no cheaper at level 2. Fine-level per-world SNR rises ×400–10 000, with means agreeing.
> - **Ground: no effect.** At 300 updates, with lr 1e-3 for both, the fill-context cost is 1.85 / 2.03 against the cfast estimator's 1.87 / 1.74 (L = 4 / 3). The probe clamp changes 1–6 % of fine draws (cfast 3–6 %), line_match is 0.50, and dirt is 0.92.
> - **At lr 3e-3** cfine stays more stochastic, but its violations rise to 4–6 %.
> - **Dimers:** an orientation still dies (vertical instead of diagonal).
> - **Cost:** 1.1–1.2 s/update against 0.17 s, so the bench was not run.
> - **Limitation:** in trained states only level 0 (|A| = 2) is fully enumerated. At levels 1–2, |A| > K and LOO samples from a near-deterministic q, so unvisited actions still get little signal.
> - **Next:** sample candidates uniformly over A with importance weights (unbiased), or K ≥ 5 at level 1. Combine with #12 / #14, because the gradient only acts in visited contexts.
> - Status: tested, negative on ground.

## 16. Distil the exact table solution into the neural actor (from chat / cdim2 agent, 2026-10-14)
cdim2's untested idea: initialise from, or regularise toward, the free-logit solution to see whether the MLP can hold the diverse dimers solution at all. This would separate representation from optimisation.

> **Annotation (2026-10-14, cdim3): outcome.**
> - **It is optimisation.** The tanh 64x2 MLP distils table_feat to exact KL 0.0255. Exact KL training from that fit stays diverse and reaches KL 0.0020 (h/v/d 0.501/0.249/0.250).
> - **Mechanism.** From the symmetric init, the λ penalty makes horizontal the safest move at every draw. The shared weights saturate that move in all contexts within 10-50 steps, and the gradient then keeps pointing at stripes and vanishes.
> - **Cheap generic fixes from init all fail.** Candidate head 3.249, hashed window-pattern input 3.250, lr warm-up 3.251, lr 3e-4 3.280. τ 4→1 over the whole run gives 1.336 (vertical dies).
> - **Ground.** The same mechanism plausibly explains the context-blind fine levels.
> - Status: tried. The neural class is sufficient; next, a curriculum that keeps contexts soft early (#12). Report: reportables/2026-10-14_cdim3/.


## 17. Soft-policy curriculum while the penalty dominates: bounded bias, λ ramp, whole-logit τ (from chat / cdim3 agent, 2026-10-14)
Keep the shared-weight actor soft until it has learned context-dependent structure. Candidates: (a) bound b by c(t), ramped from 1 nat upward; (b) ramp λ from 1 to its controlled value; (c) anneal τ on the whole logit, q ∝ exp((−s − b)/τ); (d) combinations. All are engineering freedoms.

> **Annotation (2026-10-14, cbound): outcome.**
> - **Exact gradients (dimers 4x4, λ 20).** λ ramp + whole-logit τ 8→1 reaches KL 4e-5 with h/v/d 0.505/0.248/0.248. Whole τ 8→1 alone reaches KL 0.031. Both stay diverse for 400 steps at τ = 1. The λ ramp alone stays diverse but stalls at KL 0.24. Bounding b (tanh cap or last-layer projection) and whole τ 4→1 go to stripes (3.25): the cap limits margins, not direction.
> - **Sampled cfast (dimers, 60 s).** Every arm keeps three orientations while violations are about 1 %, then drops the diagonal when violations fall. The drop is to 2-5 % rather than cfast's 0 %. With the λ ramp alone the diagonal recovered to 0.23 at 6e-5 violations (entropy 0.272 vs cfast 0.224; one seed, late in the run). The exact winner collapsed to 0.
> - **Ground (λ ramp).** All dirt by update 500 at λ ≈ 3. Gate 0.75/1.62 (L4), 0.56/2.08 (L3): no move.
> - Status: tested. It works under exact gradients; it is partial and unconfirmed under sampled gradients; it is negative on ground with the λ ramp. Next: a second seed for cbound_lam on dimers, and whole-logit τ on ground. Report: reportables/2026-10-14_cbound/.

> **Annotation (2026-10-14, ccombo):** on ground combined with #14's clamped phase, whole-logit τ 8→1 (geometric) + λ ramp 1→20 did not keep levels 3-4 soft at their release (deterministic within ~300 updates at τ 2.5-3; all dirt). Report: reportables/2026-10-14_ccombo/.


## 18. Kernel route (amortised MCMC, weighted ML) (from chat, 2026-10-14)

> **Annotation (2026-10-14, kernel-c):** log q of a given state by teacher forcing (castlegen/kernels/logq.py) matches generation-time log q to 1e-5, and weighted ML on exact toy samples reaches the reverse-KL free energy and mode masses with no entropy collapse (H/cell at H(p*)). Report: reportables/2026-10-14_kernel_c/.

> **Annotation (2026-10-14, kernel-a):** exact block-Gibbs refined contexts + ML fit of ground's fine levels (LN+SiLU 6x6, 300 s) leave reverse KL's terrace plateau: clamped line_match 0.77 (cstart2 0.59, probe 0.88), L=3 fill soft 0.128 (0.194), own 0.46 (1.12); rho2/rho1 change 0.037/0.028 (probes 0.079/0.037). Kernel: castlegen/kernels/exact_block.py. Report: reportables/2026-10-14_kernel_a/report.md.

> **Annotation (2026-10-14, kernel_b, Sweep 1 b):** the generic block-MH / multiple-try kernel with q as proposal (castlegen/kernels/block_mh.py) is exactly invariant on the toys: |z| <= 1.3 from exact starts, and dropping the q ratio drifts by thousands of SE. On ground, from probe lifts, acceptance is 0 at L >= 2 for both trained q (cfast, cstart_f1L3), any tau in {1, 2} and K in {1, 8}. At L = 1 it is 0.36-0.85, but few blocks change. From own draws it is high, but less than 1-7 % of accepted moves leave all-dirt. Usable block size today: L = 1. Report: reportables/2026-10-14_kernel_b/.

> **Annotation (2026-10-14, sweep2a, Sweep 2 fine half):** the iterated loop (GPU exact block Gibbs, castlegen/kernels/amortised.py; 6 cycles; ML on levels 0-2) saturates at kernel-a's one-shot fit after about 100-300 updates: L=3 fill -0.106/-0.111. Probe chains plateau at 0.129/cell by cycle 2: the surface is kept and the rise is rho1/rho2 count slack. Adding own draws (arm B) cuts own-context cost from 0.86 to 0.34 and L=4 fill from 0.98 to 0.66. Joint (2,2) blocks are intractable. Report: reportables/2026-10-14_sweep2a/report.md.

> **Annotation (2026-10-14, sweep2b, Sweep 2 coarse half):** with kernel-a's fine levels, q-subtree moves from probe lifts are accepted at L2/L3 (0.92/0.80 at K=8; cfast 0); variant `ckernel` (coarse flips + coarse Gibbs + L2 MTM + fine Gibbs, ML on all levels) lifts own-draw line_match to 0.61-0.62 (bench F -0.112 vs cfast 0.774, rho > 0 at every level), but violations are 100x cfast (3.2e-3, not accepted) and an own-only ablation stays collapsed (0.50): the gain comes from probe contexts, not the coarse kernels. Report: reportables/2026-10-14_sweep2b/report.md.

[dimers-kernel 2026-10-14, #18] Fine-block Gibbs on dimers: k=2 conserves the diagonal count (cfast draws stay d=0.000), k=3 reaches d=0.23 from collapsed draws; ML-only loop lost validity (viol 0.02-0.1/cell), so no usable orientation result. See reportables/2026-10-14_dimers_kernel/report.md.

> **Annotation (2026-10-14, wave2-bench, UPPER BOUND: probe-lift contexts, change request #1):** ML-only ckernel's violations are q's mass leakage, not bad targets (exp(-20V)-weighted ML on all-valid targets leaves them unchanged); mixing cfast's lambda-controlled reverse-KL (32 steps/iter, lr 1e-3) removes them while keeping line_match 0.56-0.59 (lr 3e-3 collapses to dirt). Bench ckernel_rkl32 300 s x 2: worst final 2.4e-4 at L2 (one violation over budget, one seed), F -0.012, but L3/L4 probe starvation rises to 24/42 %. Report: reportables/2026-10-14_wave2_bench/report.md.

> **Annotation (2026-10-14, wave2-explore):** own-draws-only kernels (coarse flips K=16, AIS-weighted annealed L4 subtrees, Neal tempered transitions at L3/L4, with/without a lambda ladder) leave own line_match at 0.50 in 300 s; own rho already follows fill, the collapsed variable is cert B (hard support/honour chain), accepted dirt->partial moves <= 5 %. Report: reportables/2026-10-14_wave2_explore/report.md.

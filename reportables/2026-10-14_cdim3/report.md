# cdim3: why the exactly-optimised MLP goes to stripes on dimers (representation or optimisation?)

2026-10-14 · CPU only (cores 16-23, `JAX_PLATFORMS=cpu`), exact gradients over the enumerated reachable set of cdim2's region (dimers, 4 x 4 periodic, 100 239 reachable outcomes, 412 valid matchings, λ 20, Adam, clip 10). Code: additions to `castlegen/variants/_exactopt.py` (existing kinds unchanged): `distill`, `Enum.b_unique`, `Enum.ctx_weights`, actor kinds `cnet-*` (candidate-scoring head) and `hnet-*` (hashed window-pattern embeddings as inputs), `train_exact(anneal, warmup, hook)`, CLI `--anneal --warmup --init --save`. Scripts here: `diag.py` (task 2), `distill.py` (task 1), `plot.py`; runs in `runs/`.

## Answer
**Optimisation, not representation.** The tanh 64x2 MLP on the same features fits table_feat's diverse policy to KL 0.026 by supervised distillation, and exact KL training started from that fit keeps it and improves to **KL 0.0020** (better than the table's 0.0177). From the default init, every neural actor tried (tanh, candidate head, hashed-pattern inputs, lr 3e-4, lr warm-up) reaches all-horizontal stripes (KL 3.25) by step 100-300. A slow τ schedule (4 → 1 over the whole run) stops at a different two-orientation state (KL 1.34, no vertical dimers).

## 1. Distillation (representation test)
Target: table_feat after 1500 exact steps (KL 0.0177, h/v/d 0.506/0.247/0.247). Loss Σ_c w_c KL(π_table(·|c) ‖ π_MLP(·|c)), with w_c = expected visits under the target, over all 66 757 feature contexts. Full-batch Adam at lr 3e-3.

| distilled actor | steps | distill loss | exact KL to target | P(viol) | entropy/cell | h / v / diag |
| --- | --- | --- | --- | --- | --- | --- |
| tanh MLP 64x2 (cpass) | 1000 | 1e-4 | 0.089 | | | 0.506 / 0.247 / 0.247 |
| tanh MLP 64x2 (cpass) | 4000 | <1e-5 | **0.0255** | 8e-4 | 0.377 | 0.507 / 0.247 / 0.247 |

The MLP fits the policy to well below the 0.1 bar, so the features hold what is needed and a smooth tanh MLP over the one-hot window can express it. Not run: the uniform-weighted variant (`all`) and LN 256x3. Both were queued, but I stopped them for time after the 64x2 result had settled the question.

## 2. The stripe attractor (figure)
![stripe attractor](images/stripe_attractor.png)

Panels: orientation mix; per-draw P_q(horizontal) in colour order (on 4 x 4 the colour order is raster: draws 0-3 are row 0); the q-weighted policy entropy per draw; the cosine between the descent direction −∂J/∂θ and ∂ log Q(16 stripe outcomes)/∂θ. Step 0 is shared (grey). Every 10 steps up to step 200, using cdim2's schedule.

| step | MLP h / v / d | MLP Q(stripes) | MLP cos(−∇J, ∇log Q_s) | table h / v / d | table Q(stripes) | table cos |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.44 / 0.28 / 0.28 | 0.005 | +0.90 | same | same | +0.44 |
| 10 | 0.93 / 0.03 / 0.04 | 0.73 | +0.89 | 0.49 / 0.26 / 0.26 | 0.03 | +0.49 |
| 50 | 1.00 / 0 / 0 | 0.998 | +0.46 | 0.78 / 0.12 / 0.11 | 0.42 | −0.23 |
| 100 | 1.00 / 0 / 0 | 0.999 | +0.62 | 0.69 / 0.17 / 0.15 | 0.15 | −0.11 |
| 200 | 1.00 / 0 / 0 | 0.999 | +0.78 (‖∇J‖ 0.05) | 0.61 / 0.21 / 0.18 | 0.09 | −0.22 |

- **The initial pull is toward horizontal, and it comes from the hard penalty, not from the colour order's empty neighbourhoods.** Under the uniform masked init P(viol) = 0.85, and the conditional violation probability is lower when a draw goes horizontal than when it goes vertical or diagonal, at every draw (h 0.80-0.87 vs v/d 0.84-0.90). A horizontal pair closes within the current raster row. A vertical or diagonal pair leaves an obligation on the next row that the one-step lookahead does not see. Under λ = 20, "go horizontal" is the steepest descent for both actors at step 0 (cos +0.90 MLP, +0.44 table). The first-drawn sites are not special: draw 0's P(h) is 0.5, set by the edge mask.
- **The MLP moves every context at once.** One shared "prefer E/W" direction in the weights shifts all 66 757 contexts together. By step 10 the MLP is 93 % horizontal with logit margins of 3.5. By step 50 it is saturated: margins 8-9 nats, with only the row-start E-vs-W choice left (entropy log 2 at draws 0, 4, 8, 12). The table moves each context separately and overshoots only to 78 % h at step 50. Then its gradient turns away from stripes (cos −0.2), because the entropy term wins in the still-soft contexts.
- **At stripes the MLP's gradient keeps pointing toward stripes** (cos +0.5 to +0.8 from step 100 on) and vanishes (‖∇J‖ 16 → 0.05). Given that every other context is saturated horizontal, a vertical or diagonal deviation at any one context is followed by horizontal draws that violate. So each context's best response is horizontal, and the entropy gradient through a saturated softmax is exponentially small. This is a self-consistent local optimum of the shared-weight actor. The diverse solution is also stable for the same network (§3a), so the outcome is decided in the first ~10 steps.

## 3. Generic fixes, exact gradients
All runs: λ 20, Adam, clip 10, same seed/init as cdim2. Rows are 600 steps unless noted.

| fix | setting | KL to target | P(viol) | entropy/cell | h / v / diag |
| --- | --- | --- | --- | --- | --- |
| baseline tanh MLP (cdim2, 1500 steps) | lr 3e-3, τ 2→1 over 20 % | 3.249 | 0 | 0.173 | 1.00 / 0 / 0 |
| (a) warm start from distilled MLP | lr 3e-3, τ = 1 | **0.0020** | 6e-5 | 0.376 | 0.501 / 0.249 / 0.250 |
| (b) candidate-scoring head `cnet-tanh-64x2` (h Wo + Σ w SiLU(hA + e(t)E + c), e(t) = one-hot(t), s(t)) | lr 3e-3, τ 2→1 over 50 % | 3.249 | 5e-5 | 0.173 | 1.00 / 0 / 0 |
| (c) hashed window-pattern embeddings as inputs `hnet-tanh-64x2` (full window → 4096 x 16, all rows used; 3 x 3 around site + in-block position → 4096 x 16, 4028 patterns into 2547 rows) | lr 3e-3, τ 2→1 over 50 % | 3.250 | 6e-5 | 0.173 | 1.00 / 0 / 0 |
| (d1) lr warm-up 200 steps + τ 2→1 over the whole run | lr 3e-3 | 3.251 | 1e-4 | 0.173 | 1.00 / 0 / 0 |
| (d2) lower lr | lr 3e-4, τ 2→1 over 50 % | 3.280 (still falling slowly) | 1e-3 | 0.175 | 1.00 / 0 / 0 |
| (d3) τ 4→1 over the whole run | lr 3e-3 | 1.336 | 2e-4 | 0.293 | 0.647 / **0** / 0.353 |

- (a) is the only arm that holds the diverse solution, and it gets better than the table. Under exact gradients the diverse optimum is a stable basin for the same MLP.
- (b) and (c) add expressiveness (candidate scoring; a near-table input over local patterns), but the problem is the path, not the class. Both are 100 % horizontal by step 50, as fast as the plain MLP.
- (d1) and (d2) slow the first steps, but the descent direction at init is the same, so they only reach stripes later. (d3) keeps the entropy term dominant for longer. The MLP then settles in a different symmetric-breaking state (horizontal plus diagonal, no vertical), the same kind of "one orientation dies" seen in cfine on dimers. It is better than stripes, but not the target.
- Limitation: one region (4 x 4 periodic, one site per colour), one seed per arm, and 600 steps for the fix arms (cdim2's stripe runs were flat from step 200 to 1500).

## 4. Verdict and ground
- **Which fix works:** only starting in the right basin (a). None of the cheap generic changes to the network or input (b, c), nor the lr and τ levers (d), stops the collapse on their own. The τ-high schedule is the only one that moves the end state. Since (a) needs a diverse teacher, the generic version is a curriculum that keeps the actor's contexts soft while the penalty gradient is strongest. Examples: high-τ exploration importance-weighted back to τ = 1 (ideas #12, approved) over a longer horizon, or a per-context entropy floor early on. The second option changes the objective, so it would be a proposal under constraint 3/4. Untested here.
- **Ground (one line, not a run):** the same mechanism plausibly explains the context-blind fine levels. The penalty gradient at init favours one locally safe action everywhere (dirt / parent-independent counts). The shared weights carry that direction to every context, parent included, within a few steps. Once saturated, any parent-specific deviation is punished by the other, already saturated draws, and the entropy gradient vanishes. So the network has the capacity to read the parent features but never gets a gradient that rewards using them.

# Clamp diagnostic: do fine levels fail in fill-following contexts? (2026-10-13/14)

Agent: clamp. Diagnostic only, with no changes to training or generation code. The GPU was shared with up to four other jobs.

## Answer

**Yes, and training creates the gap.** On ground, with the coarse levels clamped to a valid state that follows the fill (a probe, lifted exactly), the trained cfast fine levels cost **1.97 nats per fine cell** below L = 4. In the model's own coarse context they cost **0.72**, so the gap is **+1.26**. Below L = 3 the figures are **2.23 vs 0.34 (+1.88)**. The probe's own soft energy at those levels is only 0.014 (L = 4) and 0.013 (L = 3) per cell, so a valid completion that is almost free exists. That gives a lower bound on the fine-level KL in fill contexts of about **1.96 nats per fine cell, or about 500 nats per top block**.

Untrained (τ = 1, b = 0), there is **no gap**: fill contexts are slightly *cheaper* than the model's own (random) contexts, at 2.32 vs 2.47 (L = 4) and 1.98 vs 2.07 (L = 3). So training does not make the fine levels good at partial fills. It lowers fine cost only in the contexts the coarse level moves to: near-full certificates, where `t` is forced to all dirt.

Violations and forced draws explain only a small part of the gap. The trained fine levels in fill contexts violate 0.3–0.5 % at levels 1–3 (L = 4 clamp), against 0 in their own contexts. Draws with no valid option are 0.3–0.5 % against 0. Forced fractions are about the same in both (L0 ≈ 0.99–1.0, L1 ≈ 0.50, L2 ≈ 0.46, L3 ≈ 0.41–0.44). Almost all of the gap is **soft energy from the count prior**: children's rho does not match the parent's rho.

## Mechanism (ground)

**The trained rho channels at levels 3, 2 and 1 ignore their context.** With the same noise, rho3, rho2 and rho1 are **bit-identical (100 %)** whether rho4 and cert4 come from the model or from a probe (8 worlds; `same_check.py`). Untrained they are 98–100 % identical too.

The cause is in the conditional. A count prior κ(Σ children − parent)²/h_c² enters `cond`'s soft term only for the **last** of the four children. The lookahead is hard-only, so the first three children see the parent only through the learned bias b. At 150 s (1357 updates, B = 8) b has not learned to read the parent count. It has made the rho draws deterministic (log q ≈ 0 at levels 2 and 3) and independent of context.

Per-factor soft energy, trained, per fine cell:

| model / context | prior4 | prior3 | prior2 | prior1 (t vs rho1) |
| --- | --- | --- | --- | --- |
| own coarse | 0.36 | 0.24 | 0.07 | 0.18 |
| probe-clamped L ≥ 4 | 0.57 | 0.24 | 0.07 | 1.21 |
| probe-clamped L ≥ 3 | – | 0.80 | 0.20 | 1.35 |

The certificates carry the structure, so `t` follows the hard cert channels and not the soft counts. In the images below, the clamped generations are block-aligned staircases (16- or 8-column steps), not the designed line.

**What the coarse level does with this.** The trained top level still tracks the fill in rho4: the mean of |rho4 − 256 f| is 18.7, with mean rho4 = 136 against 127 for 256 f. But it sets cert4 near full (mean A = 13.5, B = 16 of 16), which forces all dirt below (dirt fraction 1.000). rho4 sits between the fill and the fixed rho3 sum (about 168). So the "avoidance" in this run is cert4 → full, plus every lower rho paying a fixed mismatch.

## toy_modes, L = 1 (exact)

Setup: rho2 = 8 and rho1 clamped to each of the 85 quadrant-count contexts, 16 worlds × 32 blocks each. The fine KL per block is computed exactly as E_q[S + log q] + log Z(c).

| | untrained | trained (2897 updates, CPU) |
| --- | --- | --- |
| fine KL, mode contexts (fully forced) | 0 | 0 |
| fine KL, non-mode contexts (p\*-weighted) | **1.00** | **0.36** |
| fine violations | 0 | 0 |
| forced fraction, non-mode | 0.71 | 0.67 |
| own-draw mode-context mass (p\* = 0.55) | 0.09 | **0.70** |
| p\* mass the coarse level would give modes if optimal given this fine level, ∝ p\* e^{−KL_fine} | 0.75 | 0.63 |

The trained coarse level over-selects mode contexts, at 0.70 against 0.55 under p\*. This is the direction the fine deficit predicts, and slightly beyond the optimal tilt (0.63). The non-mode fine deficit of 0.36 nats per block is at the low end of ctemp's 0.6–1.0.

Total trained KL per block, decomposed: coarse 0.16 plus fine under its own contexts 0.095, giving 0.255. With p\* contexts the fine part alone would be 0.16.

## Figures

- `images/ground_total_below_L.png`: fine cost below L, own vs fill-following, untrained vs trained.
- `images/ground_per_level_bars.png`: per level, cost, raw violations, forced fraction and no-option fraction, with SE bars (32 seeds × 4 probes with jitter 0–3 rows). The black ticks show the probe's own soft energy.
- `images/ground_clamped_examples.png`: generated tiles with own vs clamped coarse levels, from the same noise.
- `images/modes_fine_kl_by_context.png`, `images/modes_clamped_examples.png`.

![](images/ground_total_below_L.png)
![](images/ground_per_level_bars.png)
![](images/ground_clamped_examples.png)
![](images/modes_fine_kl_by_context.png)

## Method

- **Training.** cfast with its default config (B = 8, lr 3e-3) on ground, seed 0, for 150 s after a warm-up compile in the same process: 1357 updates on the shared GPU, final λ = [4.8, 7.1, 7.4, 5.4, 0]. My first attempt spent all 150 s on compiling (264 s setup) and got 0 updates; that is why the warm-up was added. toy_modes: 150 s on the CPU, 2897 updates. Untrained = the initial parameters (W3 = 0, so b = 0) at τ = 1. Parameters are in `params_*.pkl`.
- **The pass.** `clamp_tools.clamped_pass_stats` is a copy of `_creplay_clamp.clamped_pass` that also records soft, nv, log q, |A| and no-option per level. Its per-level costs match `clamped_pass` to float precision; the check is printed in `/tmp/clamp_eval_ground.log`.
- **Contexts.** Fill-following contexts are the bench's own ground probes: dirt below the jittered designed line, every coarse channel lifted exactly by `ground.lift`. No new lift helper was needed. Own and clamped runs use the same noise seeds, 3,000,000 + i.
- **Measures.** Cost is soft + λ·nv + τ log q of the drawn values, at τ = 1 with each model's own λ, per fine cell (64 × 128). Violations come from `energy.py` on the raw state, which uses the bench's level attribution.
- **Scripts.** `train_save.py`, `eval_ground.py`, `eval_modes.py`, `probe_soft.py`, `factor_breakdown.py`, `same_check.py`, `plots.py`. Results are in `ground_summary.json`, `modes_summary.json`, `factor_breakdown.json` and `probe_soft.json`.

## Caveats

- One training seed per problem and 150 s only, which is short. The context-blind rho channels may partly be an early-training state; longer runs and cbig's large B could differ.
- The cost gap is measured on cfast. I did not check cfast_head (Codex), which actually collapses.
- The fill contexts are exact probe lifts. The model's own top level never produces these exact certificates, so they are partly off-distribution for the fine b (a covariate shift, which is part of the hypothesis).

## Implications for round 3

Better coarse credit would only optimise around a fine level that cannot hit parent counts. The cheapest fix inside the constitution is to give the learned parts the parent count residual explicitly. Two options:
- a feature: the parent's rho minus the sum of its known siblings, with the number of unknown siblings, which is allowed under "more information to learned parts";
- the cconv candidate-scoring head given that residual.

Then re-run this clamp test as the gate: the fine cost in fill contexts should approach the probe's roughly 0.01 per cell. Training fine levels on target-like coarse contexts (HANDOFF step 2) is the other lever. It needs care, because it changes the sampling distribution of contexts. Under constraint 4 and the objective, that is a proposal, not something I built.

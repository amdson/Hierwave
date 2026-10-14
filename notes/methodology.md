# Coloured-pass sampler: methodology and state of play

Status as of 2026-10-14, after the box session's four agent waves. This note is for readers who have not followed the daily reports. Every number comes from a report in `reportables/`, cited by folder: [clamp] means `reportables/2026-10-13_clamp/`. `notes/coloured_pass_constitution.md` overrides this note if they disagree.

## 1. Problem statement

**What is sampled.** A world is a strip of H fine rows, infinite in x, described at levels l = 0 (finest) to T (coarsest). A level-l cell covers 2^l x 2^l fine cells and has four children one level down (2 x 2 refinement). Each level carries *channels*: integer fields such as the fine tile, or coarse abstractions such as a block's solid-tile count ("rho") or a column certificate. *Designed* channels, such as a target fill line, are fixed functions of absolute coordinates and are never sampled.

The target is a Gibbs distribution defined by declared *factors*: unary, pair (same level or parent-child) and count (a parent against the sum of its children). Each table entry is either a finite *soft* energy or infinite, i.e. *hard*. The sampler aims at p(x) ∝ exp(−S(x) − Σ_l λ_l H_l(x)), with S the soft energy and H_l the number of violated hard instances at level l.

**Local, fixed-step, learned.**
- *Local:* each sampling step reads only a fixed finite window of values already drawn, plus its own noise.
- *Fixed-step:* a fixed number of passes and repair rounds; no rejection loops, global checks or variable-length search.

Together these give each output cell a finite dependency radius. Any chunk of an infinite world can then be generated on its own, at bounded cost per cell, batched on a GPU.
- *Learned:* a neural bias corrects a designed local conditional.

The predecessor (certland, an i-SIR training chain) froze at coarse levels and depended on its start state [notes/context.md].

**Test problems** (`castlegen/bench/problems/`; all short-range, no criticality or phase coexistence):

| problem | structure | what it tests |
| --- | --- | --- |
| toy_support | T = 2, iid 4 x 4 blocks; hard "dirt only on dirt", smoothness, dirt bonus; coarse counts hard | exact answers (F* = −0.4688/cell); hard support |
| toy_modes | T = 2, 4 x 4 blocks with exactly 8 of 16 filled (hard), ferromagnetic bonds | exact answers (F* = 0.2513); four modes whose masses must match |
| ground | T = 4, H = 64; rho and (A, B) certificates at levels 1-4, tiles at level 0; soft count prior, smoothness, top-level fill term; hard support and row honour | a coarse plan (the surface line) that the fine levels must realise |
| dimers | T = 0, perfect matchings of the triangular lattice, uniform | purely hard, no slack; diversity = three orientation fractions |
| columns | T = 4; fine channel = position in a length-12 column; coarse counts hard, no positions | coarse coordination of long vertical structure |

## 2. The sampler

**Schedule.** Top-down, levels T to 0. Within a level, cells are split into 16 *colours* by (Y mod 4, X mod 4); within a colour, the channels are drawn in declared order. A *step* is one (level, colour, channel), and all its sites are drawn in parallel.

**One draw.** For each candidate value a, the designed conditional computes three quantities from the factor tables:
- the *closed* soft energy s(a), from terms whose other members are all known;
- the closed hard violations c(a);
- a one-step *lookahead*: 1 for each hard instance with unknown members that no completion can satisfy.

The *least-violating set* A minimises c + lookahead. The draw samples

q(a) ∝ 1[a ∈ A] · exp(−s(a)/τ − b_θ(a)).

Here b_θ is the learned *bias*. It is an MLP (tanh 64x2 by default; LN+SiLU optional [cln]) reading a 4 x 4 same-level window, a 3 x 3 parent window, known/in-bounds flags, τ and λ. Gumbel-max noise is hashed from absolute coordinates, so overlapping windows of the infinite strip agree exactly. Generation runs at τ = 1. Then one fixed *repair* round, coarse to fine, redraws every site touching a violated hard instance among its least-violating values, ∝ exp(−soft).

**How a parent constrains its children** (the owner's question: does parent density determine child density?). On ground, not exactly:
- **Counts are soft.** The count prior is κ(Σ children's rho − parent rho)²/h_c², with κ = 1 and h_c the children's side. A mismatch of h_c tiles costs 1 nat; twice that costs 4.
  - The term is closed only when all four children are known, so it enters the designed conditional only at the last child. The first three see the parent only through b [clamp].
  - Trained cfast models ignore the parent count: rho3, rho2 and rho1 draws were bit-identical whether the coarse levels came from the model or a fill-following probe [clamp].
- **Certificates are hard.** A certificate (A, B) promises A full columns and allows at most B occupied ones.
  - Row honour (children's A sum ≥ parent A, B sum ≤ parent B) and support are hard, so they act through the mask and lookahead.
  - Tiles follow certificates, not counts: a near-full certificate forces all dirt, and clamped completions are block-aligned staircases [clamp, csat].

On the toys and columns, counts are hard equalities and pin the children's total exactly. On columns they say nothing about where columns sit [ccol].

## 3. Training

**Objective.** Reverse KL on the sampler's own complete, unrepaired draws:

J(θ) = E_q[ S(x) + Σ_l λ_l H_l(x) + τ log q(x) ].

The three terms are the designed soft energy, the penalised hard violations and the (negative) entropy. At τ = 1, J = KL(q ‖ p_λ) − log Z. The cost splits exactly over draws, ℓ_i = s_i + λ_l c_i + τ log q_i; this "cost identity" is checked in code.

**Why reverse KL.** It needs only the sampler's own draws: no target samples and no chain. The price is mode-seeking: nothing pulls mass toward states the sampler never visits. Section 6 is largely this price.

**Gradient.** Each draw contributes two terms:
- an all-action term, Σ_a ∇q(a)[ℓ(a) + Q(a)];
- a score-function term, ∇log q(a_i)·(G_i − Q(a_i)).

The *return* G_i sums the costs of later draws. The *critic* Q predicts the return. It is a training-only control variate, so it can read wide windows and leaves the gradient unbiased. On a tiny enumerable case the estimator is unbiased within 1-2 %, but the per-batch SNR is about 0.01 at B = 8 and halves when the world doubles [notes/context.md].

**λ and τ.** λ_l starts at 10. A slow controller moves it toward a target violation rate (1e-4 to 1e-3), clipped to [0, 20]. τ anneals from 2 to 1 over the first 20 % of the budget. τ divides only s, not b; once b spreads over 15-60 nats, τ barely matters [csat].

**Truncated depth** (ideas #2, approved exception) bootstraps returns below depth d. In ctrunc, depth 1 was worst (ground F 1.05 vs 0.72 at full depth). Its real gain was 2-5x throughput [ideas.md #2].

**Estimator improvements** (all unbiased, gradient-checked):

| variant | what it does | effect | ground |
| --- | --- | --- | --- |
| cfast [cfast] | *cone returns*: G_i sums only costs draw i can influence | variance /357 at level 0, /8 at level 1, ~1x at levels 3-4; 5-12x faster | all dirt |
| creplay [creplay] | *coupled replays* at the top two levels: alternative actions under the same noise | coarse SNR x1200-3400 at 0.18 s/update (x6800-13900 at 1.4 s) | unchanged at equal updates |
| cfine [cfine] | coupled enumeration at fine levels, cone-window replays | fine SNR x400-10000 | no effect |
| cbig [cbig] | B = 256 on the GPU | toys reach F* | B = 8/64/256 all dirt |
| crb [ideas #8] | expected instead of sampled later costs | no variance gain on toy_support | not run |
| chot [chot] | ε-hot whole-logit exploration, importance-weighted to τ = 1 | same expected update as on-policy, by construction | no effect |

**Exact-gradient tool (exactopt).** On blocks small enough to enumerate, J and its gradient are computed exactly. Training one network both ways separates estimator noise from optimisation and representation.

Findings on the toys [exactopt]:
- Sampled training at B = 8 ended 5-15x above exact training, so variance was the toys' main limit.
- The tanh MLP's toy_support plateau was the learning rate. LN+SiLU reaches 0.0084 against the feature table's 0.0074.
- The 4 x 4 window leaves a 0.012-nat floor on toy_modes; 6 x 6 gives 0.0009.
- Same-colour independence leaves a hard 0.037-nat floor on the tiny layout.

It was later extended to dimers [cdim2, cdim3, cbound].

## 4. The constitution

The constitution decides validity, whatever a change scores:
1. **Generic.** Every step, mask, colouring and feature derives from the factor declarations and the hierarchy, never from one scenario ("we need a generic algorithm").
2. **Local, fixed-step generation.** The critic is exempt.
3. **Unbiased gradient** of the stated objective. No truncation, no bootstrapping, no dropped terms.
4. **Fixed objective.** No rescaling by level.
5. **No joint-domain tables.** Learned outputs are full-domain logits per channel. What lower levels induce at coarser ones is learned.
6. **No hand-written bounds or rules** linking channels beyond the declared factors.
7. **0 ≤ λ ≤ 20.** A missed budget is reported, never hidden.

**Allowed without asking:**
- more information for the learned parts;
- any training-only critic input;
- unbiased variance reduction;
- engineering (networks, optimiser, schedules).

Anything else is a proposal citing the constraint it breaks.

**Approved exceptions:**
- truncated depth (ideas #2);
- higher-τ exploration importance-weighted back to τ = 1 (#12);
- fine levels trained on parent states from any full-support distribution, at a mixed start depth (#14). This reweights contexts, so it is not the unbiased gradient of J as written, but it has the same minimiser.

## 5. Evaluation

**Bench harness** (`castlegen/bench/`, frozen; `notes/bench.md`). Each run trains 300 s per seed, 2 seeds. On held-out noise it then reports:
- raw and repaired violations per level (budget 2e-4 at every level);
- free energy F = E[S + log q] per fine cell (lower is better);
- statistics against exact values and across seeds;
- a finite-support test;
- a probe check;
- an image.

Acceptance needs all of them. Good curves or images do not count.

**Bench v2 probe check** [probe2, v2board]. *Probes* are valid states from an independent source (designed ground states, exact toy samples).
- *Why it replaced the old check.* The old flag compared whole-world −S − log q of probes with the 99th percentile of the sampler's own draws. At 16 vs 16 worlds it fired on exchangeable draws 53 % of the time [probes].
- *Blocks and shares.* v2 works per *block*: a 2 x 2 patch of level-l sites with one parent. It compares each block's *log q share* (the sum of its per-draw log q) with its *target log-weight share* (−soft − 10 x violations attributed to the block).
- *What it reports, per level:* Spearman ρ between the two shares, and *starvation*. Starvation is the fraction of probe blocks more than 5 nats below the own-draw Theil-Sen fit, with the own-block fraction as a null.
- *Details.* Where direct weight is constant, a "subtree" weight is used instead. On toy_modes the exact ideal is ρ 0.87 at level 0.
- *Proposed gate (v2board):* starved ≤ max(3 %, null + 2 pp) at every level, and probe ρ at level 0 ≥ 0.8 x ideal. Not yet adopted.

**Fast loop for ground** (user decision). The full bench runs only if steps 1-3 pass:
1. Dimers, 60 s. If an orientation dies, stop.
2. Ground, 60-120 s, logging per-level entropy, dirt fraction and *line_match*: agreement of the surface with the designed line, 0.5 chance, about 0.88 for probes.
3. *Clamp gate*: fine cost per cell (soft + λ·violations + τ log q) below level L, with the coarse levels clamped to a fill-following probe vs the model's own.
   - cfast reference: 1.97 vs 0.72 at L = 4 and 2.23 vs 0.34 at L = 3. A probe's own soft energy is 0.014 [clamp].
   - The reference moves: 1.99/1.12 and 2.13/0.66 in one 300 s run [cstart], and 1.76/0.63 and 2.25/0.44 in another [chot]. Treat differences below about 0.3 nats as noise.
4. Bench.

## 6. What is established

**Toys: solved, with one violation miss.** cbig256 reaches F* on toy_support (gap 0.0000 / +0.0012) and is +0.0022 / +0.0023 from it on toy_modes [v2board, baselines300]. It has exact stats, seed agreement, 0-1.5 % starved blocks (null up to 1.7 %) and level-0 ρ 0.87 = ideal. Batch size was the toys' only problem. The summaries omit one caveat: toy_support seed 1 has final violations of 4.9e-4, above budget, so strictly it is not accepted.

**Estimator fixes do not move ground.** None of these changed the outcome: replays, enumeration, large B, LN+SiLU / 6 x 6 / conv actors, count-residual features, schedules, bottom-up curricula, unbiased exploration. Ground stays all dirt, line_match 0.50 [box_waves]. At 300 s cfast meets the violation budget (1.9e-5), but levels 2-4 have 100 % of probe blocks starved, by 64-181 nats median; level 0 is fine [v2board].

**The collapse mechanism on ground:**
1. *Near tie at init.* With all top-level draws clamped together, fill is 0.093/cell cheaper than all dirt at τ = 2 and 0.016 dearer at τ = 1 (within jitter). Per single draw, rho4 prefers fill (72 % of sites) but cert4 already prefers dirt (80 %), through the untrained fine levels' violations [ctop].
2. *Fine levels improve faster in dirt contexts.* After 50 updates the dirt-context fine cost falls from 2.31 to 1.96 while the fill-context cost stays put (2.34 → 2.38). The dirt preference reaches +0.31/cell (about 2,500 nats/world), and +0.90 by update 200 [ctop].
3. *cert4 locks.* Its entropy drops below 0.3 nats and dirt passes 0.95 around update 200 [csat]. Across arms the lock lands between updates 125 and 300.
4. *Fill contexts are never revisited.* Level-0 draws are 99-100 % forced within about 300 updates [csat]. Reverse KL trains only visited contexts, and importance weighting gives unvisited ones weight about zero [chot].
5. *The fine MLP saturates to a parent-independent safe action.* Even when kept stochastic (lr 1e-3), fine draws are 95-100 % identical under probe-clamped coarse levels [csat].

Training creates the gap: untrained, fill contexts are slightly cheaper than the model's own (2.32 vs 2.47 at L = 4) [clamp].

**Columns and dimers follow the same script.**
- *Columns* [ccol]:
  - The top level turns deterministic at update 125 while density falls from 0.38 to 0.06 (target 0.817).
  - Probes are 99.9-100 % starved at levels 3-4.
  - Under probe counts the fine level still cannot build valid columns (0.24 violations per cell), because counts carry no positions.
- *Dimers* [cdim, cbound2]:
  - cfast loses one orientation within 10-100 updates. Which one dies depends on the seed; ending with two is stable.
  - Entropy and violations trade along one curve for every λ in [1, 20].

**Capacity vs optimisation (dimers, exact gradients).** The test patch is 4 x 4 periodic, with 412 valid matchings.
- Free per-context logits reach KL 0.0175 with the target mix (h/v/d 0.51/0.25/0.25). Every neural actor goes to horizontal stripes, KL 3.249 [cdim2].
- The sampler class can express the target, and so can the MLP: distilled from the table it fits to KL 0.026, then exact training improves it to 0.0020 [cdim3].
- What fails is the path. Under λ = 20 the horizontal move is locally safest everywhere (descent cosine 0.90 toward stripes). Shared weights carry it to all contexts, with 8-9-nat margins by step 50; then the gradient vanishes.
- A candidate head, hashed pattern inputs, warm-up and lower lr all still give stripes.

**Soft early phase on dimers (cbound).**
- *Exact gradients:* a λ ramp 1 → 20 plus whole-logit tempering, q ∝ exp((−s − b)/τ) with τ 8 → 1, reaches KL 4e-5.
- *Sampled training:* the λ ramp alone (`cbound_lam`) is the first sampled run to end with three orientations within budget. Four runs:

| run | h / v / d | raw violations | note |
| --- | --- | --- | --- |
| seed 0, 60 s | 0.47 / 0.30 / 0.23 | 6e-5 | |
| seed 11, 120 s | 0.49 / 0.27 / 0.24 | 7.8e-5 | |
| seed 12, 120 s | 0.44 / 0.18 / 0.38 | 1.1e-4 | vertical collapsed to 0.02, recovered only partly |
| 300 s bench, 2 seeds | diagonal 0.14-0.17 | up to 1.1e-4 | F −0.29 vs cfast −0.23 |

- *What is still off:* the mix, and entropy (0.27-0.30 vs 0.377). The exact winner, λ ramp plus whole-logit τ, collapsed under sampled training.
- *On ground:* the λ ramp is still all dirt by update 500 (λ ≈ 3). Ground's driver is soft energy, not the penalty [cbound, ctop].

**Training on fill contexts (#14).**
- *cstart* (50/50 mixture, random start depth) only delayed the collapse: dirt reached 0.95 at about update 450 vs 150. The gate did not move (2.04 vs 1.19 at L = 4) [cstart].
- *cstart2* clamped every world at L = 3 to a probe lift [cstart2]:
  - Fill-context cost at L = 3 fell to 0.18-0.30/cell (−0.004 for LN+SiLU 6 x 6), against 2.23 for cfast and 1.25 for cstart's 90 %-clamped arm. box_waves rounds this to "8x better".
  - The best actor's soft energy is 0.194, 14x the probe's.
  - All actors stall at block terraces: line_match 0.58-0.63 (untrained 0.60, probe 0.88), plateauing after about update 800.

**Parent-awareness stays weak.** When the clamped probe changes, rho2 changes in 1.4-2.8 % of draws, against 7.9 % for the probes (0.6 % untrained). Tile changes are mask-forced [cstart2]. Count-residual features did nothing [cresid].

**Coarse prior (ccoarse).** Gibbs sampling on the designed energy restricted to levels ≥ L [ccoarse]:
- On ground it gives fill-following counts (line_match 0.83 at L = 3 vs 0.58 for own draws).
- It gives uniform certificates: only 10-18 % of blocks are rho-certificate consistent, against 100 % for probes. Since tiles follow certificates, fine line_match is 0.51.
- On columns and the toys it gives uniform counts.

**Net:** on ground, fill-context training and a soft early phase both look necessary, and neither is sufficient yet [box_waves].

## 7. Current status and open directions

**In progress: ccombo.** A sibling agent is combining three pieces on ground, gated by the clamp test and v2 starvation:
- a cstart early phase in fill contexts (f = 1, L = 3), decaying to full-depth training;
- the LN+SiLU 6 x 6 actor;
- cbound's schedules.

No result yet.

<!-- ccombo result: TBD -->

**Open user decisions:**
- *Adopt the v2 gate* and drop the old probe flag.
- *Target-like certificates.* Options:
  - (a) Clamp only rho from the prior and let the model sample and train cert onward. ccoarse argues this is within #14.
  - (b) A certificate-aware coarse prior.
  - (c) Train the coarse levels on repaired own draws.

  (b) and (c) raise questions under constraints 1, 4 and 6.
- *ground_hard.* Named as an open decision; no specification exists in the repo yet.

**The per-candidate / tabular-head proposal.** cstart2 and box_waves suggest a fine actor that can key on the parent, tested first with exact gradients on a small ground block:
- per-candidate scoring;
- a discrete pattern embedding;
- a tabular-like head with free logits keyed on parent count and neighbour heights.

The first two are engineering freedoms. But on dimers a candidate head and hashed pattern inputs collapsed like the MLP [cdim3], so expressiveness alone is not the obstacle.

The third is a proposal, not a freedom, for two reasons. A table over joint values of several channels (parent count x neighbour heights) is what constraint 5 forbids. Keying on "neighbour heights" is also a ground-specific feature, which conflicts with constraint 1. It needs approval, and probably a generic formulation, before even an experiment.

**Other threads:**
- The same-colour floor needs more passes or a finer colouring (ideas #3; cmulti's objective awaits approval).
- Columns needs positional information at the fine level.
- Dimers' v2 row (43 % of probe blocks starved, 0.2-nat median deficit) is unexplained [v2board].

## 8. Glossary

- **Colour**: the class (Y mod 4, X mod 4); one colour's sites are drawn in parallel.
- **Least-violating set A**: values minimising closed violations plus one-step lookahead.
- **Bias b_θ**: the learned logit correction; q ∝ 1[A] exp(−s/τ − b).
- **Repair**: one fixed round redrawing sites that touch violated hard instances.
- **Return G_i**: the cost of later draws that draw i can influence (its *cone*).
- **Critic**: a training-only return predictor, used as a control variate.
- **Coupled replay**: re-running a draw with other actions under the same noise.
- **λ_l, τ**: the per-level violation penalty (≤ 20) and the temperature.
- **F, F***: the free energy per fine cell, and its exact optimum.
- **Probe**: a valid state from an independent source.
- **Starvation**: a probe block's log q share more than 5 nats below the own-draw fit.
- **Clamp gate**: fine cost below L in fill-following vs own coarse contexts.
- **line_match**: agreement with the designed fill line.
- **Start depth L**: in cstart, levels ≥ L are clamped to a supplied context.

## 9. Where to find things

- **Rules:** `notes/coloured_pass_constitution.md`, `notes/box_rules_2026-10-14.md`, `reportables/ideas.md`.
- **Harness:** `castlegen/bench/` (including `shares.py`, the v2 check) and `castlegen/bench/problems/`; documented in `notes/bench.md`; results in `results/board.md`.
- **Sampler:**
  - `castlegen/variants/_cpass_core.py`: pass, features, repair.
  - `_cond_jax.py`: conditional and lookahead.
  - `cpass.py`: training.
  - `cfast.py` and `_cfast_cone.py`: cone returns.
  - Variants: `creplay.py`, `cbig.py`, `_cln_core.py`, `cstart.py`, `cbound.py`, `_coarse_prior.py`.
  - `_exactopt.py`: exact-gradient tool.
- **Reports:** `reportables/<date>_<name>/report.md`; session summary in `reportables/2026-10-14_box_waves/report.md`; log in `reportables/progress.md`. Hand-over notes: `notes/context.md`, `notes/handoff.md`.

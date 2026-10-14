# Coloured-pass sampler: methodology and state of play

Status as of the evening of 2026-10-14, after the box session's four agent waves and the afternoon-to-overnight switch to the kernel route (section 7). This note is for readers who have not followed the daily reports. Every number comes from a report in `reportables/`, cited by folder: [clamp] means `reportables/2026-10-13_clamp/`. `notes/coloured_pass_constitution.md` overrides this note if they disagree.

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

## 7. The kernel route (amortised MCMC training)

Reverse KL stalled on ground (section 6). On 2026-10-14 the user relaxed constraints 3 and 4 for training only (amendment in `notes/coloured_pass_constitution.md`; decision in `notes/decision_2026-10-14_kernel_route.md`). Reports are cited by folder as before; [overnight] is the wave log `reportables/2026-10-14_overnight/report.md`. Every result below was trained by weighted maximum likelihood (ML) on kernel-refined states unless a mix with reverse KL is stated.

### 7.1 Premise

If K is a Markov kernel that leaves p* invariant, then KL(qK ‖ p*) ≤ KL(q ‖ p*). Applying K to q's draws gives states closer to p*, and fitting q to them by weighted ML (minimise −Σ w_i log q(x_i)) is an improvement direction. The chain does not need to mix. Training still consults only the target energy, and generation is unchanged: the fixed-step local pass, never a chain. The cost is that the route is only as good as the states the kernels reach: from a collapsed q, a local kernel returns states near the collapsed mode (section 8). Admitted kernels: exact block Gibbs, MH subtree resampling with q as proposal (multiple-try allowed), coarse flips, annealed blocks. Each is defined only from the factors, the hierarchy and the sampler. Violations inside an acceptance ratio are priced at a fixed λ ≤ 20 or treated as infeasible. All code is in `castlegen/kernels/`.

### 7.2 Teacher-forced log q and weighted ML (kernel_c)

`logq.py` runs the pass with every draw set to a given state's value (*teacher forcing*) and returns log q; a forced value outside the least-violating set has q = 0 and is counted as infeasible. `make_ml_update` is an Adam step on −Σ w_i log q(x_i) / Σ w_i. Check: teacher-forced log q equals the log q recorded when cfast generated the state, to under 1e-5 (40 states, τ = 1 and 1.7; 8 tests).

Fit to exact p* on one enumerated 4 x 4 block, LN+SiLU actor, one seed, 240 s [kernel_c]:

| problem | arm | fwd KL(p* ‖ q) | F per cell (F*) |
| --- | --- | --- | --- |
| toy_support | ML on 256 exact samples | 0.026 | −0.4629 (−0.4688) |
| toy_support | exact reverse KL | 0.056 | −0.4651 |
| toy_modes | ML on 256 exact samples | 0.033 | 0.2534 (0.2513) |
| toy_modes | exact reverse KL | 0.014 | 0.2520 |

ML matches reverse KL in free energy and mode masses, and its entropy stays at or above the target's (0.335 against 0.332 per cell on toy_support), where reverse KL sits below. The tanh reverse-KL arm lost support (forward KL 2.13); tanh ML did not (0.046). The sampled arm is 2-3 times worse in KL than exact weights (batch noise, constant lr). The update needs only (state, weight) pairs; acceptance probabilities never enter it.

### 7.3 Exact fine-block Gibbs (kernel_a, sweep2a)

`exact_block.py` enumerates every joint completion of a block, prices it by all touching factor instances, and samples the exact conditional. A block (L, depth) is a level-L cell's channels plus those of its descendants down to L − depth; blocks sharing no instance run in parallel (4 colours on ground). On ground (1, 1) is rho1, cert1 and four tiles (480 values), (2, 0) is rho2 and cert2 (255). Joint (2, 2) blocks have 1.4e13 completions and were not built. Tests (9): conditional energy differences equal full-energy differences; blocks from an invalid start reproduce the exact toy statistics within 4 SE; exact samples stay exact.

Experiment [kernel_a]: 160 worlds (75 % probe lifts, 25 % cstart2 own draws), two Gibbs cycles, ML on levels 0-2, levels 3-4 frozen. Clamp gate, one seed:

| model | L=3 fill cost | L=3 soft | L=3 own | line_match, L=3 clamp |
| --- | --- | --- | --- | --- |
| cstart2 (reverse KL) | −0.004 | 0.194 | 1.12 | 0.59 |
| ML on kernel states | −0.107 | 0.128 | 0.46 | 0.770 |
| probe | | 0.014 | | 0.876 |

The network class that stalled at block terraces under reverse KL now follows the hill (1-2 rows of dirt teeth, not flat terraces). rho2 responsiveness is 0.037 against the probes' 0.079, rho1 0.028 against 0.037; own-context line_match stays 0.51 as levels 3-4 are untrained.

The iterated loop [sweep2a] (arm A probe lifts only; arm B half own draws; six cycles and 8 updates per iteration) shows the chains plateau after two cycles: probe-context soft energy goes 0.020 to 0.129 and stays at 0.128-0.131 through cycle 30. So p*'s fine conditional given the coarse parents sits at about 0.13 per cell, not the probe's 0.014, with line_match about 0.73. L=3 fill cost is −0.106 (A), −0.106 and −0.111 (B, 552 and 1016 updates); the loop saturates after 100-300 updates. Arm B helps own contexts (L=3 own 0.35 against 0.86 for A).

### 7.4 MH subtree kernel (kernel_b)

A block is a level-L cell plus every sampled channel below it. The proposal is the coloured pass restricted to the block, tempered by whole-logit (−soft − b)/τ; teacher forcing the old values gives the reverse density. The move is independent *multiple-try* MH: K candidates with weights p*/q, one picked ∝ weight, accepted with min(1, Σw / (Σ_{i≠j} w + w_x)); K = 1 is plain MH.

Verification: block energy delta equals `energy.py` to 3e-6; draws follow exp(log q) (TV 0.0016). On the toys, chains from exact samples stay exact (|z| ≤ 1.3 with a deliberately bad q; one borderline 2.5). Dropping the q ratio gives |z| 9188 (toy_modes) and 37.5 (toy_support), so the test has power. Two own-draw runs with a mild bad q did not converge (z 3.4, 6.1; acceptance 0.03-0.12), read as slow mixing since the same q is invariant from exact starts. Delayed acceptance was not built: the block pass is one scan, so a screen saves no compute.

**Ground acceptance** [kernel_b]: one move per (state, random cell), 256 per cell, λ = 20, τ = 1, K = 1 (K = 8 is higher). Cells are acceptance / fraction changed.

| q, start | L=1 | L=2 | L=3 | L=4 |
| --- | --- | --- | --- | --- |
| cfast, own draws (100 % dirt) | .84/.29 | .64/.57 | .49/.49 | .56/.56 |
| cfast or cstart_f1L3, probe lifts | .36-.62/.05-.06 | 0 | 0 | 0 |

Acceptance from own draws is uninformative: a collapsed q proposes another all-dirt block and accepts it (dirt to fill in at most 1 % of accepted moves for cfast, 7 % at L=4 for cstart_f1L3). **From probe lifts the acceptance is 0 % at L ≥ 2 for every trained q, τ and K.** For cfast the median log ratio is −73 / −270 / −1228 at L = 2 / 3 / 4: q gives the probe's block 35 / 194 / 827 nats less than its own proposal, and the proposal costs 12 / 30 / 246 nats more energy. The reverse-KL coarse conditionals were badly wrong.

**After the fine levels were fitted by ML** [sweep2b]: same procedure, but q is kernel_a's fit (levels 3-4 untrained). Probe-lift acceptance, K = 1 (K = 8), τ = 1:

| L=1 | L=2 | L=3 | L=4 |
| --- | --- | --- | --- |
| .93 (.99) | **.79** (.92) | **.44** (.80) | .03 (.16) |

The two tables use different q's and are not comparable cell for cell (L=1 differs, .36 against .93, for this reason). Probe moves went from 0 / 0 / 0 to .79 / .44 / .03 at L = 2 / 3 / 4, and the L=2 proposal gap shrank from −35 to −1.4 nats. This confirmed the bottom-up hypothesis: once the fine levels are right, a q subtree proposal is a usable kernel one level up. τ = 2 hurts from probes at L ≥ 3.

### 7.5 Coarse flips and single-cell Gibbs (sweep2b)

`coarse_flip.py` picks a level-L cell, proposes its channels from a flip distribution with full support (`uniform`, or the pass conditional tempered at τ_c), redraws the subtree from q, and accepts with the exact multiple-try ratio. With the q flip at τ_c = τ it equals kernel_b's move bit for bit. Coarse single-cell Gibbs enumerates a cell's channels jointly (2925 completions at L3, 39 321 at L4), rejection-free. Toy invariance (128 chains): |z| ≤ 1.25 on toy_modes, ≤ 0.86 on toy_support at K = 1; dropping the q ratio gives 800 and 153. Two K = 4 toy_support runs show z = 3.53, both from `empty_block_frac` (exact 5.5e-4; one empty block seen where about 4.5 were expected, so the SE equals the estimate). On toy_modes rho2 is pinned by a unary, so the flip is trivial there. A K ≥ 2 toy_support check with a frequent statistic is missing.

### 7.6 Batched colour Gibbs and annealed blocks (speed2)

`colour_gibbs.py` resamples all cells of one (level, channel, colour) class in all worlds at once from exp(−(soft + λ·violations)/τ). No two cells of a class share a factor instance on the toys or ground, so this is a valid parallel Gibbs step (22 tests; conditionals equal brute force; a negative control at τ = 1.5 gives |z| 43). A level-0 sweep runs at 1.4e8 cells/s (3e7 contended). `anneal_block` is annealed importance sampling inside a block (τ 8 → 1, one masked sweep per step); on the toys its weights recover the exact log Z ratio (modes −11.510 ± 0.014 against −11.505; support −5.998 ± 0.008 against −5.985).

**Random starts fail; valid starts work.** From random starts 0 of 64 blocks reach zero violations (about 10 remain at L3), because single-cell moves cannot cross hard counts; a λ ramp from 0 erases a valid start. Annealing τ only, λ fixed at 20, hard final phase, from valid starts [speed2 part 2]:

| level, steps, start | valid | ESS of 64 | all-dirt blocks made partly filled |
| --- | --- | --- | --- |
| L3, 32, probe | 1.00 | 15.7 | |
| L3, 32, q draw | 0.98 | 1.5 | 17 % |
| L4, 128, probe | 0.92 | 5.6 | |
| L4, 128, q draw | 0.89 | 3.6 | 34 % |

Cost is 0.10 s per 64 blocks at L3 (32 steps) and 0.44-0.57 s at L4 (128 steps). The perturbation is local (10-25 % of cells change), and AIS weights from non-equilibrium starts are heuristic.

### 7.7 Fine blocks for T = 0 (dimers_kernel)

`fine_block.py` gives k x k tiles of level-0 cells at a random offset (7 tests, exact on 4 x 8 periodic dimers). A k = 2 block can only flip a 2 x 2 square, which conserves the number of diagonal dimers; a rhombus needs 2 x 3 cells, which k = 3 contains (15 480 completions; k = 4 has 2.9e6, too large for the GPU path). Pure kernel chains without learning, 600 cycles, h / v / d [dimers_kernel]:

| start | k = 2 | k = 3 |
| --- | --- | --- |
| cfast draws (.467 / .533 / 0) | .509 / .491 / **.000** | .382 / .390 / .228 |
| probe rhombi (.495 / 0 / .505) | .420 / .075 / .505 | .400 / .207 / .393 |

So local blocks can restore the third orientation, but only with k ≥ 3, and they mix slowly: chains from opposite starts did not meet, so the stationary mix is unmeasured (bulk target 1/3 each). With ML on refined own draws, validity was lost within about 50 updates in all four arms (own violations 2e-4 to 0.05-0.1 per cell), with hard-priced blocks and exp(−20V) weights alike. The orientation fractions of those draws are fractions of violating states and say nothing about the target.

### 7.8 Speed-ups (speed1)

Half the fine-loop time was the (1, 1) Gibbs sweep and a third the ML step. Changes (semantics-preserving): factored block energy (13 times on the sweep, bit-identical states); teacher forcing over the 16 colour steps at once (4.4 times on the ML step, gradients within 2.4e-3 from float32 order, so the test tolerance went from 1e-5 to 1e-4); frozen levels skipped; cycles and updates fused under `lax.scan`. Result: 0.24-0.43 s per iteration against 1.61 (3.7 times contended to 6 times quiet; the GPU was shared), 16.8 ML updates/s against 2.5. The 300 s gate is unchanged (L3 fill −0.114, line_match 0.743), and own-draw line_match is 0.508: a faster fine loop does not move own draws. The coarse cycle is now the main cost, about 0.7 of 4.5 s per ckernel iteration.

## 8. What the route established on ground

**Fine levels follow the hill.** Under ML on exact-Gibbs states, levels 0-2 leave the reverse-KL terrace plateau: L=3 fill cost −0.107 against −0.004, clamp line_match 0.77 against 0.59 [kernel_a]; 0.73-0.74 in the iterated, faster loop, saturating within about 300 updates [sweep2a, speed1]. The ceiling is the equilibrated conditional itself (soft 0.13 per cell, line_match about 0.73).

**ckernel.** The variant `ckernel` (`castlegen/variants/ckernel.py`) per iteration: 32 contexts (16 probe lifts, jitter 0-3, and 16 own τ = 1 draws); four coarse flips per world (L in {3, 4}, q mode, τ_c = 2, K = 4); single-cell Gibbs at L4 then L3; four K = 8 MTM moves at L2; three fine Gibbs cycles; then 8 ML steps on all levels. All kernels at λ = 20. Own draws over 440 updates [sweep2b]:

| update | dirt | own line_match | own soft/cell | own violations/cell |
| --- | --- | --- | --- | --- |
| 8 | 0.83 | 0.504 | 1.10 | 0.0187 |
| 424 | 0.52 | **0.619** | 0.20 | 0.0011 |

This is the first variant above own-draw line_match 0.50. The coarse gap of the clamp gate closes: L=3 own −0.084 against fill −0.110 (cfast 2.23 against 0.72). On the v2 bench (1 seed, 368 updates; line_match 0.609 there) probe rho is 0.38 / 0.82 / 0.86 / 0.86 / 0.58 at levels 0-4, starvation 0 / 0.5 / 5.4 / 10 / 3.5 % (cfast 1 / 76 / 100 / 100 / 100 %), F −0.112 per cell against 0.774.

**Caveat 1: violations.** The worst final violation fraction is 3.2e-3 at L3 (budget 2e-4), so the row is not accepted. Plain ML has no violation pressure.

**Caveat 2: the gain comes from probe-lift contexts, outside the frozen rules.** The own-draws-only ablation (32 own draws per iteration, 280 updates) stays collapsed: dirt 0.90, line_match 0.50, though the kernels lower energy inside the dirt mode (soft 1.10 to 0.56). Even in the main run the refined states' line_match is within 0.007 of the raw draws', so the kernels do not move draws; ML on refined probe-start chains does. The amendment covers kernels on q's own draws, and the inequality in 7.1 needs q-started chains. ideas #14 admits fine levels conditioned on arbitrary parents, not coarse levels trained to reproduce probes' coarse states. Change request #1 (`notes/constitution_change_requests.md`) records this; the orchestrator's ruling is that the probe arm stays as a labelled upper bound and the main line must move own draws out of the dirt state with kernels on own draws only.

**Violation arms** (upper-bound arm; one seed each, 300 s including setup, shared GPU) [wave2_bench]:
- *Weighting by exp(−20V) does nothing.* Probe-refined states have no violations and nearly every own-refined world has some, so the weights drop the own states and the fit becomes probe-only ML (arms with λ = 20 and λ = ∞ kernels gave bit-identical own draws for 128 updates). The violations are q's mass leaking onto violating states, which ML on valid targets never penalises.
- *λ = 20 kernels create L4 violations* (support4 from 0 to 11 in 16 worlds), by trading a hard violation for a larger soft gain; at λ = ∞ L4 stays 0.
- *The 3.2e-3* is 13 support3 violations in 16 worlds (one violation is 2.4e-4 at L3): the coarse levels choose certificates the children cannot realise next to their neighbours, and repair then creates L4 violations.
- *Mixing in cfast's λ-controlled reverse-KL update* works at lr 1e-3 (32 steps per iteration: own raw violations 1e-4 to 4e-4 at every level, line_match 0.57-0.59) and collapses at cfast's lr 3e-3 (dirt 0.99, line_match 0.500).

Bench, 300 s, 2 seeds, upper-bound arm:

| variant | worst final violations | F/cell | line_match | starved % 0 / 1 / 2 / 3 / 4 |
| --- | --- | --- | --- | --- |
| ckernel_rkl32, seed 0 | 2.4e-4 (L2; 4 of 16 384 instances) | −0.026 | 0.560 | 0 / 3.9 / 10 / 24 / 41 |
| ckernel_rkl32, seed 1 | 1.8e-4 (within budget) | 0.001 | 0.585 | 0 / 4.3 / 10 / 23 / 44 |
| ckernel_rkl48, seed 0 | 4.9e-4 | 0.036 | 0.558 | 0.2 / 5.2 / 12 / 27 / 44 |

rkl32 misses by one violation on one seed. The violation pressure re-introduces starvation at L3-L4 (about 24 % and 42 %, against 10 % and 3.5 % for ML only). rkl48 is worse, and its seed 0 got 112 ML updates under contention. Seed 0 of ckernel_rkl (16 steps) spent 254 s of 300 s compiling and is not a measurement.

**Dimers.** k = 3 blocks restore the third orientation (7.7), but ML loses validity in every arm, so there is no trained orientation result. The valid references remain cbound_lam (0.47 / 0.30 / 0.23, violations 6e-5) and cfast (0.51 / 0.49 / 0.00).

**Status at the time of writing** (ground, 300 s, v2 bench):

| variant | own-draw line_match | worst final violations | v2 starvation | accepted? |
| --- | --- | --- | --- | --- |
| cfast (reverse KL) | 0.50 | 1.9e-5 | 76-100 % at levels 1-4 | no: all dirt |
| ckernel (probe + own contexts) | 0.609 | 3.2e-3 | 10 % at L3 | no: violations; outside frozen rules |
| ckernel_rkl32 (probe + own) | 0.56 / 0.585 | 2.4e-4 / 1.8e-4 | 24 % at L3, 42 % at L4 | no: one seed over budget; outside frozen rules |
| ckernel, own draws only (not benched) | 0.50 | 5e-3 own raw | not measured | no: collapsed |

Two reports disagree on small items: ckernel's line_match is 0.609 on the bench, 0.619 in the instrumented run and in wave2_bench's table, and its L3 final violation is 3.2e-3 on the bench against 1.5e-3 in wave2_bench's own 32-world evaluation of the same recipe (different runs and noise).

## 9. Current status and open directions

**Where things stand.** The reverse-KL route's last experiment was ccombo [ccombo]: a fill-context early phase, the LN+SiLU 6 x 6 actor and cbound's schedules, one seed per arm, 300 s. It failed the clamp gate, so the bench was not run: levels 3-4 locked into all-dirt certificates within about 300 updates after release, and own-draw line_match stayed 0.500. Keeping half of each batch clamped (`ccombo_floor`) fixed L=4 (fill cost 0.56 against 1.16) but not L=3 (0.110). That is the point at which the project switched to the kernel route (section 7). The kernel route moves the fine levels (section 8) and, with probe-lift contexts, the coarse levels, but no variant is accepted. The main line, kernels on own draws only, is still collapsed to the dirt state.

**Open questions:**
- **(a) Own draws leaving the dirt state, with kernels applied to own draws only.** wave2-explore: in progress. It uses a bottom-up curriculum, tempered or multiple-try coarse flips, annealed subtree blocks from valid starts (7.6, handed over as the exploration move) and delayed acceptance. `<!-- wave2-explore result: TBD -->`
- **(b) Violation pressure without re-introducing collapse or starvation.** wave2-bench2: in progress. It applies reverse KL to levels 2-4 only at low lr and ML to levels 0-2, prices L4 kernels as infeasible, runs a fine sweep after every coarse move, and warms the compile cache before the bench. `<!-- wave2-bench2 result: TBD -->`
- **(c) The user's decision on change request #1:** admit kernel-refined probe lifts as training contexts for all levels, or keep them as an upper-bound diagnostic. If admitted, the bench's probe check no longer tests generalisation to unseen target-like states and needs held-out probes with different jitter or seeds.
- **(d) Coarse kernel speed.** speed-1b: in progress. The coarse cycle (block_mh, coarse_flip) is about 0.7 s of each 4.5 s ckernel iteration. `<!-- speed-1b result: TBD -->`
- **(e) Dimers with the reverse-KL mix.** Part 2 of dimers_kernel (ML plus a violation term, as in ckernel_rkl). `<!-- dimers part 2: TBD -->`

**Other user decisions still open** (from before the kernel route):
- *Adopt the v2 gate* and drop the old probe flag (the v1 flag still fires and still counts in bench verdicts).
- *ground_hard.* Named as an open decision; no specification exists in the repo yet.
- *Target-like certificates by reverse-KL means* (options (a)-(c) of ccoarse). Change request #1 is the kernel-route version of the same question.

**The per-candidate / tabular-head proposal** (tested first with exact gradients on a small ground block) stays unapproved: a table over joint values of parent count and neighbour heights is what constraint 5 forbids, and "neighbour heights" is ground-specific (constraint 1). The kernel results make it less urgent for the fine levels, which already follow the hill, though parent responsiveness is still half the probes' for rho2.

**Other threads:**
- The same-colour floor needs more passes or a finer colouring (ideas #3; cmulti's objective awaits approval).
- Columns needs positional information at the fine level. The kernel route has not been run on columns.
- Dimers' v2 row (43 % of probe blocks starved, 0.2-nat median deficit) is unexplained [v2board].
- Open verification gaps: a K ≥ 2 toy_support invariance check with a frequent statistic; k = 4 fine blocks on dimers; a second seed for almost every kernel-route result.

## 10. Glossary

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
- **Amortised MCMC**: training q toward states produced by applying MCMC kernels to its own draws, so that no chain has to mix; generation stays the fixed-step pass.
- **p*-invariant kernel**: a Markov kernel K that leaves the target unchanged; then KL(qK ‖ p*) ≤ KL(q ‖ p*).
- **Teacher forcing**: running the pass with every draw set to a given state's value, to read off log q of that state.
- **Weighted ML**: the update −Σ w_i log q(x_i) on states x_i with weights w_i.
- **AIS weight**: the annealed-importance-sampling weight of a block, tracked from exact energy differences along a τ schedule; unbiased only from an exact starting distribution.
- **ESS**: effective sample size of a set of weights, (Σw)² / Σw²; here given out of 64 blocks.
- **Multiple-try (MTM)**: an MH move with K candidates, one picked in proportion to p*/q and accepted with a corrected ratio; K = 1 is plain MH.
- **Delayed acceptance**: screening a proposal with the cheap coarse factors before the full ratio. Specified in the amendment, not built.
- **Colour Gibbs sweep**: exact single-cell Gibbs on all cells of one (level, channel, colour) class in parallel; valid because no factor instance holds two cells of a class.
- **Own draws vs probe lifts (as contexts)**: contexts are the states the kernels start from. Own draws are the sampler's τ = 1 output; probe lifts are designed valid states lifted to all levels. Probe-lift contexts are outside the frozen rules (change request #1).
- **Upper-bound arm**: a run that uses probe-lift contexts, kept as a labelled diagnostic.

## 11. Where to find things

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
- **Kernel route:**
  - `castlegen/kernels/`: `logq.py` (teacher forcing, ML update), `exact_block.py` and `amortised.py` (exact block Gibbs, GPU port, loop pieces), `block_mh.py` (MH and multiple-try subtree), `coarse_flip.py`, `colour_gibbs.py`, `anneal.py`, `fine_block.py` (T = 0), tests in `tests/kernels/`.
  - Variant: `castlegen/variants/ckernel.py` (with the `ckernel_rkl*` configurations).
  - Decision and rules: `notes/decision_2026-10-14_kernel_route.md`, the amendment in `notes/coloured_pass_constitution.md`, `notes/constitution_change_requests.md`.
  - Reports (`reportables/2026-10-14_<name>/report.md`): `kernel_c`, `kernel_a`, `kernel_b`, `sweep2a`, `sweep2b`, `speed1`, `speed2` (parts 1-2), `dimers_kernel`, `wave2_bench`. Running: `wave2_explore`, `speed1b`.
  - Wave log, newest first: `reportables/2026-10-14_overnight/report.md`.

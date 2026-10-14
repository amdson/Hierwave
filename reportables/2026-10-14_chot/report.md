# chot: ε-hot exploration, importance-weighted back to τ = 1 (2026-10-14)

Agent: chot. Builds ideas.md #12, which the user approved. The GPU was shared with up to four other agents' jobs throughout. All training used the 300 s budget.

## Answer

**The estimator is correct, but it does not help.** Gate 1 passes: the weighted gradient is unbiased on the tiny enumerable problem, with ε = 0.3, τ_b = 3, zero and wrong critics, and the unweighted negative control is rejected. Gate 2 fails: after 300 s on ground, the fine levels are still deterministic and still pay a large cost in fill-following contexts (2.20 against 0.79 nats per cell at L = 4, and 2.46 against 0.30 at L = 3), which is no better than cfast trained through the same code path (1.76 against 0.63, and 2.25 against 0.44). The level-4 entropy collapses at the same update (about 300) in both runs, and every own draw is all dirt (line_match 0.500).

**Why this was expected, in hindsight.** The two properties conflict. An unbiased estimator of ∇J at τ = 1 has the *same expected update* whatever the behaviour policy. Exploration with importance weights changes only the variance of the update, never its mean. A context that q reaches with probability e^−15 enters the weighted gradient with weight about e^−15, however often the hot draws visit it. So once the top level saturates, the fine levels are still never trained on fill contexts *in expectation*. Exploration that changes what the fine levels learn has to change the expected update. That means a biased estimator or a different objective, such as a tempered or mixture objective, or training fine levels on fill contexts. Under constraints 3 and 4 that is a proposal for the user, not an experiment (see the end of this report).

## Gate 1: gradient check (tests/variants/test_chot.py, 11 tests pass)

The tiny enumerable problem from `_gradcheck.py` (T = 1, 4 coarse and 16 fine cells), at τ = 1 and λ = (2, 3), with 300 batches of B = 16 worlds:

| case | level | proj_z | rand_z | Hotelling p | proj_ratio ± rel_noise |
| --- | --- | --- | --- | --- | --- |
| hot at all levels, zero critic | l0 / l1 | 0.62 / −0.87 | 0.95 / −0.64 | 0.37 / 0.21 | 1.016 ± 0.027 / 0.979 ± 0.030 |
| hot at all levels, wrong critic | l1 | −2.27 | −1.71 | 0.11 | 0.84 ± 0.08 |
| top level only, zero critic | l1 | −1.09 | −0.82 | 0.06 | 0.98 ± 0.03 |
| top level only, wrong critic | l0 / l1 | 0.87 / −3.19 | 0.31 / −2.58 | 0.38 / 0.023 | 1.36 ± 0.44 / 0.80 ± 0.07 |
| **negative control** (hot draws, weights dropped) | l0 / l1 | **−10.7 / −10.4** | | **0.0 / 0.0** | rejected |

The wrong-critic rows passed the test's thresholds, but they leaned negative (−2.3 and −3.2), so I reran them on fresh seeds with 800 batches (`gc_wrong.py`). Every |z| was below 1: hot at all levels, l1 proj_z −0.18 with ratio 0.992 ± 0.047; top level only, l1 −0.04 with ratio 0.999 ± 0.043; ε = 0 (cfast's estimator), l1 0.17. The first run's lean was seed noise from the wrong critic's heavy tails.

The other tests:
- With ε = 0, the pass reproduces cfast's `train_pass` bit for bit, on toy_support and ground.
- `chot_off`'s fused update equals cfast's.
- Replays are faithful with hot draws on (ε = 0.3), on toy_support, ground and dimers:
  - the hot coins never move;
  - replacing one draw changes draws, costs and ratios only inside its cone;
  - replaying the drawn value reproduces the pass exactly.

## The estimator (castlegen/variants/_chot_core.py, chot.py)

- **Behaviour.** Each draw at a hot level is hot with probability ε. The coin is `noise.uniform(seed, "hot:" + channel, level, Y, X)`, so replays redraw the same coins. A hot draw samples from the **whole-logit** tempered conditional q_b ∝ q^(1/τ_b) over A, using the same per-value Gumbel noise. q is the target policy at the current τ. Following csat, tempering only the designed term cannot move a saturated b.
- **Ratio.** The coin is independent auxiliary noise, so each draw marginally follows b = (1 − ε) q + ε q_b. Each draw carries ρ = q / b, which is bounded by 1 / (1 − ε). By convexity, χ²(q‖b) ≤ ε χ²(q‖q_b), so this has lower variance than weighting only the hot draws by q / q_b.
- **Which draws enter the weight.** For a loss term f, let C be the ancestral closure of the draws f depends on. Then E_b[f ∏_{m∈S} ρ_m] = E_q[f] for any S ⊇ C. To see it, sum out the draws outside C in reverse step order: nothing in C reads them, so each factor sums to 1. chot takes S = every draw of the hot levels in the world. That is a superset of every term's closure, so it needs one weight W per world, which multiplies that world's all-action, score and critic terms. A per-term closure would have lower variance, but on ground a fine draw's backward closure reaches almost all of level 4 anyway.
- **Rest of training.** The critic is trained unweighted on behaviour draws; it is only a control variate. The λ controller uses importance-weighted violation counts. Generation is cfast's, unchanged.
- **Defaults.**
  - Hot levels: `hot="top"`, i.e. level T only.
  - ε = n_hot / (number of draws at the hot levels), with n_hot = 1.28 expected hot draws per world. This bounds Var W whatever the world size. On ground it gives ε = 0.02 over 64 level-4 draws; on toy_support 0.04; on dimers 0.000625.
  - τ_b = 3.
- **Registered variants.** `chot_all` (every level, same expected count per world), `chot_t2` (the top two levels), and `chot_off` (ε = 0: cfast through chot's code path, with chot's entropy logging). Neither chot_all nor chot_t2 was run.

## Gate 2: ground, 300 s, clamp evaluation

Setup:
- `train_save.py`: warm compile, then 300 s on seed 0.
- `eval_ground.py`: cresid's and csat's clamp evaluation, 32 seeds × 4 probes.
- chot and chot_off trained at the same time on cores 40–43 and 44–47.

| run | updates | L = 4 own | L = 4 fill | L = 3 own | L = 3 fill | own dirt | own line_match | clamped line_match L4 / L3 | final λ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| untrained | 0 | 2.47 | 2.32 | 2.07 | 1.98 | 0.88 | 0.503 | 0.561 / 0.606 | 10 |
| chot_off (= cfast) | 3202 | 0.63 | 1.76 | 0.44 | 2.25 | 1.000 | 0.500 | 0.540 / 0.606 | 0 |
| **chot** | 3157 | 0.79 | **2.20** | 0.30 | **2.46** | 1.000 | 0.500 | 0.561 / 0.590 | 0 |

References: the 150 s cfast run gave 0.72 / 1.97 and 0.34 / 2.23; the probe's own cost is 0.014.

![](images/gate.png)

**Entropy over training** (training batch, non-forced draws, logged every 10 updates). The ESS curve is a 20-update running mean; its ends are padding artefacts.

![](images/train_entropy.png)

- **Level 4** collapses identically in both runs: H < 0.3 nats by update 300 and 0.00 from about update 1000.
- **The fine levels do not stay stochastic.** chot's level 0 went deterministic *earlier* (about update 350, against about 1000 for chot_off). Levels 2 and 3 end at 0.01 nats in chot, against 0.02 / 0.1 in chot_off. This is one seed per arm, so I don't read anything into the difference.
- **The weights were mild.** Batch ESS averaged 7.07 of 8 (minimum 2.7), with max W = 1.88 over the run. The hot fraction at level 4 was 2.0–2.5 % as designed.

So the variance cost was small, and so was any effect. Hot draws at a saturated level put the alternatives into worlds whose weight is about q(alt) / (ε q_b(alt)), which is close to 0.

## Gate 3: bench, 300 s × 2 seeds, chot against cfast back to back

The two chains ran at the same time, chot on cores 40–43 and cfast on 44–47, with chot's start about 40 s behind. The GPU was shared with other agents. Both runs used `--train 300` with tag `chot_gate3`. Each row gives seed 0 / seed 1.

| problem | variant | updates | worst final violations | worst raw violations | F per cell (mean) | stats | probes | accepted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ground | cfast | 4219 / 4143 | 0 / 0 | 3.8e-6 | 0.758 / 0.855 (0.806) | all dirt (dirt 0.999 / 1.000), line_match 0.500 / 0.500 | flagged | no (probes) |
| ground | **chot** | 3727 / 3976 | 0 / 0 | 0 | 0.765 / 0.718 (0.741) | all dirt (1.000 / 1.000), line_match 0.500 / 0.500 | flagged | no (probes) |
| dimers | cfast | 9067 / 5351 | 2.2e-4 / 2.2e-4 (misses 2e-4) | 2.3e-4 | −0.2279 | h / v / diag 0.479 / 0.520 / 0.001 and 0.512 / 0.488 / 0.000 | flagged | no |
| dimers | **chot** | 7902 / 5151 | 1.9e-4 / 9.3e-5 | 2.0e-4 | −0.2261 | h / v / diag 0.462 / 0.538 / **0.000** and 0.496 / 0.504 / **0.000** | flagged | no (probes) |
| toy_support | cfast | 17292 / 12678 | 3.7e-4 / 3.7e-4 (misses) | 3.7e-4 | −0.4509 / −0.4475 (−0.4492) | exact stats missed (dirt 0.765 / 0.768) | flagged | no |
| toy_support | **chot** | 14405 / 11169 | 0 / 0 | 0 | −0.4447 / −0.4498 (−0.4472) | exact stats missed (dirt 0.774 / 0.782) | flagged | no (stats, probes) |

The exact optimum on toy_support is F* = −0.4688.

- **Ground.** Both variants are all dirt and ignore the fill line. chot's F is 0.065 lower, which is within the seed spread (cfast's seeds differ by 0.10).
- **Dimers.** The diagonal orientation does **not** survive in chot: it is 0.000 on both seeds, as with cfast. chot meets the violation budget and cfast just misses it (2.2e-4). With 2 seeds, a 300 s budget and runs whose update counts differ by up to 1.8×, I don't count that as a real difference.
- **toy_support.** chot is at 0 violations and cfast at 3.7e-4. chot's F is 0.002 worse.
- **Updates.** chot got 6–19 % fewer updates per run. The per-update overhead is the hot logits and ratio, plus contention.

Images: `images/bench_<variant>_<problem>_s<seed>.png`.

![](images/bench_chot_ground_s0.png)
![](images/bench_cfast_ground_s0.png)
![](images/bench_chot_dimers_s0.png)
![](images/bench_cfast_dimers_s0.png)

## Proposal (not built; needs the user's approval, constraints 3 and 4)

Any change that makes the fine levels learn fill contexts once the top level has saturated must change the *expected* update. The options:
- (a) Optimise a tempered or mixture objective: KL from the behaviour mixture, or J_τ with τ held above 1 at the coarse levels for part of training. That is an objective change (constraint 4).
- (b) Drop or clip the importance weights on fine-level terms, i.e. train the fine levels on behaviour contexts. That is a biased estimator of ∇J (constraint 3). It is equivalent to the "train fine levels on target-like coarse contexts" proposal in the cresid and csat reports.
- (c) Within the constitution, prevent the coarse collapse itself instead of exploring around it. One example is csat's per-candidate enumeration at the count draws.

chot's estimator and its tests are the right infrastructure for (b) as a controlled experiment. For example, add a flag that weights only coarse-level terms, then measure the bias against the exact gradient on the tiny problem.

## Files

- `castlegen/variants/_chot_core.py`: the hot pass, weights and fused update.
- `castlegen/variants/chot.py`: the variants and the training loop. cfast is unchanged.
- `tests/variants/test_chot.py`: the 11 tests.
- In this folder:
  - `gc_wrong.py`: the wrong-critic rerun.
  - `train_save.py`, `eval_ground.py`, `clamp_tools.py`, `gate.py`, `line_match.py`: the clamp gate, adapted from csat.
  - `plot_train.py`, `plot_gate.py`: the figures.
  - `runs/<run>/`: train logs, parameters and summaries.
- The evaluation's clamped-example image failed to save (missing directory). The numbers were already written, so I did not rerun it.

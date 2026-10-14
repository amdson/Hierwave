# cdim2: can the sampler class express diverse, valid dimers? (exact optimisation)

2026-10-14 · CPU only (cores 32-39, `JAX_PLATFORMS=cpu`), exact gradients over the enumerated reachable set. Code: `dimers<H>[x<W>]` problem in `castlegen/variants/_exactopt.py` (existing kinds and problems unchanged; additions: `dimers_problem`, `reachable_designed` for the designed edge channel, `dimer_stats`, `Enum.des`, T = 0 case in `kl_coarse`). Plot: `plot.py`; runs: `runs/`; figure `images/exact_dimers.png`.

## Region
Triangular-lattice dimers, hard matching + designed top/bottom edge, T = 0, no soft terms. **4 rows x 4 columns, periodic in x** (the only legal choices are periodic widths that are multiples of 4; a non-periodic block would drop the matching factors at its edge, so it is not used). 6x6 is not a legal periodic width; "6x6" below means the actor window.
- Reachable set: **100 239** outcomes (enumerates in 40 s). **412 are valid perfect matchings** (= all of them; the target puts 2e-16 on invalid outcomes at lambda 20). log 412 / 16 = **0.376 nats/cell** is the maximum entropy. Target orientation mix: h 0.505, v 0.248, diag 0.248.
- Contexts: 4x4 window 66 757 feature contexts, 6x6 window 127 736 (= the 127 736 history contexts). Every colour holds exactly one site on 4x4, so **table_hist is the full chain rule: any distribution on the reachable set**. 4x8 was not tried (reachable set would be roughly the square of this one).

## Results (lambda 20, tau 1, 1500 steps, exact gradients, tau annealed 2 -> 1, same init)
| actor | window | lr | KL to target | P(violation) | entropy/cell | h / v / diag |
| --- | --- | --- | --- | --- | --- | --- |
| tanh MLP (cpass) | 4x4 | 3e-3 | 3.249 | 0 | 0.173 | 1.00 / 0 / 0 |
| tanh MLP | 6x6 | 3e-3 | 3.249 | 0 | 0.173 | 1.00 / 0 / 0 |
| table_feat | 4x4 | 3e-2 | 0.0177 | 6e-4 | 0.377 | 0.506 / 0.247 / 0.247 |
| table_feat | 6x6 | 3e-2 | 0.0175 | 6e-4 | 0.377 | 0.506 / 0.247 / 0.247 |
| table_hist | 4x4 | 3e-2 | 0.0175 | 6e-4 | 0.377 | 0.506 / 0.247 / 0.247 |
| table_hist | 6x6 | 3e-2 | 0.0175 | 6e-4 | 0.377 | 0.506 / 0.247 / 0.247 |
| extra: tanh MLP | 4x4 | 1e-2 | 3.249 | 0 | 0.173 | 1.00 / 0 / 0 |
| extra: LN+SiLU 64x2 | 4x4 | 3e-3 | 3.249 | 0 | 0.173 | 1.00 / 0 / 0 |
| extra: LN+SiLU 256x3 | 6x6 | 3e-3 | 3.249 at step 150 (killed, same stripes) | 0 | 0.173 | 1.00 / 0 / 0 |

The MLP KL 3.249 = log(412/16): q is uniform on the **16 all-horizontal matchings** (two offsets per row, four rows): stripes, exactly zero violations, entropy log 16 / 16 = 0.173 nats/cell. All four neural runs reach this by step ~200 and stay (flat to 1500 steps). The tables are still falling at step 1500 (0.0177 -> 0.0175 over the last 50 steps, ~0.001 per 50): P(violation) 6e-4 is the remaining KL, not a floor.

![curves](images/exact_dimers.png)

## Answer
- A zero-violation policy with the **full entropy (0.377 nats/cell) and the target mix (50 / 25 / 25 % h / v / diag)** is representable in this class: table_hist (the most the order and masks allow) reaches KL 0.0175 and P(viol) 6e-4 and is still improving. So the order, one-step lookahead and masks are **not** the limit, at least on 4x4 periodic.
- table_feat, which sees only what the actor window shows (same information as the MLP), gets there as well, with both windows. On a 4-wide periodic region the 4x4 window already covers every column, so the 6x6 window changes nothing; the window question is therefore not tested at scale here.
- The tanh MLP (two lrs, and LN+SiLU at 64x2 and, briefly, 256x3) goes to **horizontal stripes with zero violations and 0.173 nats/cell, 46 % of the maximum**, and does not leave. The same features and the same objective are solved by free logits, so this is representation / optimisation of the neural actor on exact gradients, not training noise and not the sampler class. Exact gradients do not rescue it: the stripe state is a stable stationary point (all neural variants, three lrs/architectures, converge to it within 200 steps).
- Caveat: one tiny region (16 cells, one site per colour). It shows representability, not that the learned-network limit persists at 32 x 64 where colours have many sites (same-colour independence, the 0.037-nat floor measured on `tiny`, could matter there). The tables are also unconstrained per-context logits, a far bigger class than a shared network over windows. Why the network picks stripes (the shared weights across contexts and lambda 20 making the one-orientation basin attractive while the diagonal dies early) was not diagnosed; an untested idea is to initialise or regularise toward the tables (distillation) to see whether the MLP can hold the target solution at all.

## What this says for the cdim result
Training finds stripes because a neural actor, exactly optimised, finds them too. By the decision rule (table_hist diverse and valid, MLP not) the gap is representation/optimisation of the network, not the sampler class: structural remedies (more passes, colouring, deeper lookahead) are not shown to be needed on dimers.

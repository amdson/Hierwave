# exactopt (ideas #10): exact-gradient diagnostic, verdict

2026-10-13/14 · `castlegen/variants/_exactopt.py` (diagnostic, not a bench variant), `tests/variants/test_exactopt.py`.
CPU only (`JAX_PLATFORMS=cpu`, 4 cores per job), so GPU contention does not affect any number here. All KLs are exact, per enumerated block, in nats, at τ = 1, λ = 10, same init for every MLP run. τ anneals 2 → 1 over the first 20 % of steps.

## Verdict

| problem | exact tanh MLP | feature table | history table | sampled (B = 8) | limit |
| --- | --- | --- | --- | --- | --- |
| toy_modes block (2000 steps) | 0.0126 | 0.0116 | 4e-7 | cpass 0.175, cfast 0.191 | **variance** first (sampled 15× worse); a 0.012 **information** floor from the actor window |
| toy_support block (1000 steps) | 0.303 (plateau) | 0.0074 | 0.0056 | cpass 1.59, cfast 1.56 | **variance** (sampled 5× worse than even the stuck MLP) plus **optimisation** of the tanh MLP; not capacity, not information |
| tiny (gradcheck, 1000–2000 steps) | 0.313 | 0.319 (0.070 with an 8-wide window) | 0.037 (floor) | cpass 1.11, cfast 0.96 | **information** (mostly the window: 0.32 → 0.07) and **order** (same-colour independence, 0.037-nat floor) |

Using the report's decision rule:
- **Exact ≈ 0 where sampled is not → variance.** True on all three: with the same function class, init and step count, the sampled estimators end 3–15× above the exact MLP (modes 0.175 vs 0.013; support 1.56 vs 0.30, and vs 0.012 for the same MLP at lr 1e-2; tiny 0.96 vs 0.31). Variance is the main limit at B = 8.
- **MLP above feature table → representation.** On support the gap (0.30 vs 0.007) is **optimisation, not capacity**: the same tanh 64×2 MLP reaches 0.012 at lr 1e-2, and 0.0074 at lr 3e-3 with a 3000-step schedule (stopped at step 2000). LayerNorm + SiLU at the default lr reaches 0.0084 with no tuning, and 0.021 by step 200 at 256×3. On modes and tiny the MLP equals the feature table, so no representation gap exists there.
- **Feature table above history table → information or order.** On modes this is all the actor window: widening it to 6×6 takes the feature table from 0.0116 to 0 and the MLP from 0.0126 to 0.0009. On tiny, the history table itself floors at 0.037, because sites of the same colour are drawn independently. On support the two tables agree (0.0074 vs 0.0056, both still falling), so there is no information gap there; the "0.013" quoted earlier was a step-675 value.

![studies](images/studies.png)

## 1. Collected runs

All JSONs are in `runs/`; the per-problem figures are `images/curves_{modes,support,tiny}.png` (made with `plot.py`), and the round-3 studies are in `images/studies.png` (made with `plot_studies.py`).

| run | modes (2000) | support (1000) | tiny (1000) |
| --- | --- | --- | --- |
| exact tanh MLP, lr 3e-3 | 0.0126 | 0.303 | 0.313 |
| exact feature table, lr 3e-2 | 0.0116 (flat from step 500) | 0.0074 (falling) | 0.319 |
| exact history table, lr 3e-2 | 4e-7 | 0.0056 (falling) | 0.047 (0.037 at lr 0.1–0.3, 2000 steps) |
| sampled cpass, B = 8 | 0.175 | 1.589 | 1.113 |
| sampled cfast, B = 8 (no cone) | 0.191 | 1.560 | 0.959 |

On support, the sampled runs end with 1.3–1.7 % violating draws (target 0.04 %) and a flat-block fraction of 0.35 (target 0.096). The stuck exact MLP ends at 0.13 flat blocks, and the tables at 0.096–0.097.

## 2. Representation study (toy_support block, exact gradients, 1000 steps)

New actor kinds `net-<tanh|ln>-<width>x<depth>` (ln = Linear → LayerNorm → SiLU; zero output layer, so b = 0 at step 0 as for the cpass MLP). The inputs are the same recorded feature contexts, so only the function class and optimisation vary.

| actor | lr | KL @250 | KL @500 | KL @1000 |
| --- | --- | --- | --- | --- |
| tanh 64×2 (cpass MLP) | 3e-3 | 0.343 | 0.323 | **0.303** |
| tanh 64×2 | 1e-2 | 0.090 | 0.090 | **0.012** |
| tanh 64×2 | 3e-2 | 0.93 | 0.91 | 0.90 (too high) |
| tanh 64×2, 3000-step schedule | 3e-3 | – | 0.109 | 0.021 (0.0074 at step 2000; stopped there for time) |
| tanh 256×3 | 3e-3 | 0.094 | 0.080 | 0.042 |
| **LN+SiLU 64×2** | 3e-3 | – | 0.027 | **0.0084** |
| LN+SiLU 256×3 | 3e-3 | – | – | 0.021 at step 200 (killed by the job timeout at step 200) |
| feature table | 3e-2 | 0.072 | 0.022 | 0.0074 |

**Reading:** the tanh MLP has enough capacity (it reaches the table's 0.007–0.012 when lr or schedule allow). The 0.30 is a plateau: it sits there for about 800 steps at lr 1e-2 before dropping, and it does not escape at lr 3e-3 within 1000 steps with the fast anneal. LN + SiLU removes the plateau at the default lr, and its 64×2 version matches the feature table. This supports the user's switch to LN + SiLU. The fresh-init tanh 64×2 control was not run (timeout).

## 3. Information study

**What the feature context lacks.** `ACTOR_WIN` reads same-level offsets −1..2 from the 2×2 parent block's corner (a 4×4 window) and parent offsets −1..1. On a 4×4 block, every level-0 draw misses one full row and one column of the block (7 of 16 cells). The parent window already covers the whole level-1 grid, and the known/in-bounds flags give the position. Nothing else is missing: on every problem, the feature vector determines the closed soft energies and the mask (n contexts = n feature contexts).

**Test: widen the same-level window to −2..3 (6×6)** (`--win=-2:3`, re-enumerates the contexts).
- toy_modes: the feature table goes from 0.0116 to **0.0000**, and the number of feature contexts becomes equal to the number of history contexts (52 181). The tanh MLP goes from 0.0126 to **0.0009** at 2000 steps.
- So the modes floor is entirely the window. Ferro bonds and the block count couple a draw to cells outside the 4×4 window through the not-yet-drawn cells.
- tiny, with an 8-wide same-level window covering the whole periodic width (`--win=-3:4`, parent window unchanged): the feature table goes from 0.319 to **0.070** at 1000 steps. The history table reaches 0.047 at the same lr and steps. The feature contexts grow from 2 268 to 22 498, against 100 908 history contexts. Most of tiny's 0.28-nat information gap is the window; the rest is probably the parent window, which still misses one level-1 column (untested).

Widening the window is an allowed freedom ("wider symmetric windows"). Its cost is feature dimension: level 0 goes from 132 to 212 inputs on modes. A conv trunk (cconv) gets the same information through depth.

## 4. tiny: is the history table's 0.047 a floor? Yes: same-colour independence

- More steps and a higher lr converge to **0.0370** (lr 0.1) and **0.0368** (lr 0.3) at 2000 steps; both are flat from step 1000. So 0.047 was slow convergence onto a floor of about 0.037, not a still-falling curve.
- **Cause.** tiny is H = 2, periodic W = 8, so each level-0 colour (Y mod 4, X mod 4) holds two sites, X and X + 4. They are drawn in the same step, independently given the prefix. Under the target they are correlated through the cells still to be drawn (with W = 8 they share neighbours via the wrap). No per-site policy can express that correlation.
- **Check.** A diagnostic actor `table_seq` conditions each draw also on the earlier same-step draws, which is the full chain rule; it is not a valid sampler. It falls to **0.0021** at 2000 steps and is still falling, while p_viol drops to 0.0013.
- So the 0.037 is the coloured-pass order's own limit on this layout, not a mask or lookahead effect.
- **Why the 4×4 blocks reach 0.** On a 4×4 non-periodic block, every colour holds exactly one site, so the history table is the full chain rule.
- **Implication for real problems.** Same-colour sites are 4 apart. The loss is a correlation between draws at distance 4 that is mediated by undrawn cells. It is small for short-range targets, but it is a hard floor that no network or window can remove. Removing it needs more passes or a finer colouring (the cmulti direction, where the constitution allows several passes).

## Proposals (outside allowed freedoms: none needed)

Everything above uses allowed freedoms (window, network, lr, schedule), apart from the diagnostic `table_seq`, which is a measurement only.

## Next

1. Default the actor to LN + SiLU, with the 6×6 same-level window, then rerun the sampled comparison. The remaining gap is then variance alone: sampled 0.18 vs exact 0.001 on modes.
2. Variance is the binding constraint at B = 8. Rerun the sampled curves on the blocks at B = 64–256 (cbig) to measure how quickly the gap closes.
3. To measure the same-colour floor at a realistic width, enumerate a wider tiny (W = 12) to see whether it shrinks with site distance.

## Code added (existing kinds unchanged)

- `_exactopt.py`:
  - `net-<act>-<w>x<d>` actors;
  - `table_seq` (diagnostic);
  - `--win lo:hi[,plo:phi]` rebinds `core.ACTOR_WIN` before building the variant and enumeration, with a per-window enum cache;
  - per-run lr via `exact_<kind>@<lr>`.
- `plot_studies.py`.

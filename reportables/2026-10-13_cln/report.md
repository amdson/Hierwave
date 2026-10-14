# cln: LayerNorm + SiLU actor and 6×6 actor window on cfast

2026-10-14 · acts on exactopt's verdict (ideas #10, #5). Both changes are allowed freedoms: engineering (network) and more information to learned parts (wider symmetric window). GPU shared with 3–4 other agents' jobs throughout; every wall-clock number below is contended.

## Code
- `castlegen/variants/_cln_core.py` (new): `ln_mlp_init` / `ln_mlp` (Linear → LayerNorm → SiLU, 64×2, zero output layer as in the cpass MLP), `hidden1` (the actor's first hidden layer for cfast's h1 critic), `window(lo, hi)`, `win_reach`, `net_init`.
- `_cpass_core.py` (small edits): `Layout(..., win=None)` with an `actor_win` property (None → `ACTOR_WIN`, read at trace time, so exactopt's rebinding still works); `coloured_pass` reads `lay.actor_win`; `mlp` dispatches to `ln_mlp` when the params carry LayerNorm gains; `margin(problem, win=None)` takes the read radius as max(3, window reach).
- `_cfast_core.py` reads `lay.actor_win`; `cpass.py` builds the layout window from `cfg["win"]`, initialises the actor from `cfg["net"]` and computes the margin from the window; `cfast.py`: `CFG` gains `net="tanh"`, `win=None` (defaults leave every existing variant unchanged); the cone uses `lay.actor_win`; layout and compiled-function caches key on win and net.
- Variants: `cln` (`net="ln"`, 4×4), `cln6` (`net="ln"`, `win=(-2, 3)`).
- Caveat: `_creplay_core.py`, `_crb_core.py` and `_cbig_core.py` still build their cones from `core.ACTOR_WIN`. They are correct at the default, but setting `win` on those variants would give a cone that is too narrow. Their owners should switch to `lay.actor_win` first.

## Tests (CPU)
- `tests/variants/test_cln.py`, **11 passed**:
  - defaults unchanged (same tanh init bit for bit, 4×4 window, same margin);
  - for the 6×6 window: every change lies inside the cone (toy_modes, toy_support, ground, columns), the cone reach widens, and the training pass equals cpass's pass;
  - finite support (overlapping open windows agree, raw and final) for cln6 on toy_modes and ground, and for cln on toy_support.
- The existing `test_cfast_cone.py` and `test_cresid.py` still pass (**17 passed**).
- Margin: the 6×6 window reads at most 3 same-level cells from the site, which equals the parent window's reach that the margin already assumed. So the margin is unchanged (the formula now follows the window anyway).

## Bench (GPU, shared; 2 seeds × 150 s; B = 8; KL = 16 (F − F*) per 4×4 block)
| problem | variant | updates s0 / s1 | worst final viol | F (s0 / s1) | KL/block (s0 / s1) | accepted |
| --- | --- | --- | --- | --- | --- | --- |
| toy_support | cln6 | 2661 / 4659 | 1.2e-4 | −0.389 / −0.420 | 1.28 / 0.78 | no (probes, exact stats) |
| toy_support | cln | 3732 / 2627 | 1.2e-4 | −0.441 / −0.412 | 0.45 / 0.90 | no (probes, exact stats) |
| toy_support | cfast | 8936 / 11450 | **5.9e-3** (s0) | −0.425 / −0.433 | 0.70 / 0.58 | no (+ violation budget, seeds) |
| toy_modes | cln6 | 2486 / 4644 | 0 | 0.2675 / 0.2672 | 0.260 / 0.255 | no (probes, exact stats) |
| toy_modes | cln | 3661 / 2561 | 0 | 0.2614 / 0.2653 | 0.161 / 0.223 | no |
| toy_modes | cfast | 9028 / 11420 | 0 | 0.2605 / 0.2614 | **0.147 / 0.162** | no |
| ground | cln6 | 257 / 1554 (125 s compile) | **1.5e-3** (s0) | 1.43 / 0.65 | – | no; all dirt |
| ground | cln | 622 / 2245 (90 s compile) | 6.1e-5 | 0.96 / 0.76 | – | no; all dirt |
| ground | cfast | 3163 / 2104 (16 s compile) | 3.8e-6 | 0.70 / 0.64 | – | no; all dirt |

Finite support held in every run. Probes were flagged in every run.
- **Ground collapses for all three.** dirt_frac is 0.985–1.0 and line_match 0.5 in every seed; the fill line is ignored. In cln6 s0, short streaks of air hang below the top (floating 10.2).
- **Toy runs:** on these runs the LN/6×6 variants are not better than cfast. cfast's toy_modes KL is the lowest, and cln6's the highest.

![curves](images/curves.png)

| | cfast | cln | cln6 |
| --- | --- | --- | --- |
| ground s1 | ![](images/cfast_ground_s1.png) | ![](images/cln_ground_s1.png) | ![](images/cln6_ground_s1.png) |
| toy_modes s1 | ![](images/cfast_toy_modes_s1.png) | ![](images/cln_toy_modes_s1.png) | ![](images/cln6_toy_modes_s1.png) |
| toy_support s1 | ![](images/cfast_toy_support_s1.png) | ![](images/cln_toy_support_s1.png) | ![](images/cln6_toy_support_s1.png) |

## Cost per update
- **Microbenchmark:** the fused update on toy_modes, in one process, interleaved, 5 × 20 calls on the shared GPU. Medians: cfast **5.1 ms**, cln **6.1 ms** (+20 %, LayerNorm), cln6 **6.0 ms** (the window adds about 0 on top of LN; level-0 inputs go from 132 to 212).
- **Bench update counts:** in the bench, cln and cln6 got 2.5–4× fewer updates than cfast. That is much more than the 20 % measured above, so most of the difference is GPU contention, which varied between the back-to-back runs. That makes the toy comparisons confounded by update count.
- **Ground compile time:** compilation is a real cost on ground. cln6 took 125 s of the first seed's 150 s budget (257 updates), cln 90 s and cfast 16 s; part of that is cfast's warm shared cache.

## Reading
- **No gain at the sampled level.** The exact-gradient gains (LN: 0.30 → 0.008; 6×6: 0.013 → 0.001 on modes) do not show up in sampled B = 8 training at 150 s. That matches exactopt's main verdict: variance, not representation or window, binds at B = 8 (sampled KL is 0.15–1.3 per block, against 0.001–0.008 exact).
- **The comparison is confounded.** The LN variants also got far fewer updates under contention.
- **Ground:** neither change affects the all-dirt collapse.

## Next
- Rerun with a fixed update count, not fixed wall clock, or on a quiet GPU.
- Combine cln6 with large B (cbig), where variance no longer dominates. The exactopt numbers say that is where the window and LN should pay off.

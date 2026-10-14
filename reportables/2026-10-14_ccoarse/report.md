# ccoarse: restricted-energy Gibbs as a source of target-like coarse states (ideas #14)

2026-10-14 · `castlegen/variants/_coarse_prior.py`, `tests/variants/test_coarse_prior.py` (25 pass). Scripts here: `validate_ground.py`, `clamp_check.py`, `sanity_counts.py`; numbers in `ground_metrics.json`, `clamp_check.json`, `counts_metrics.json`. CPU numpy on 1 core; the model draws used clamp's saved cfast params (`2026-10-13_clamp/params_ground.pkl`) on the shared GPU. 64 worlds, seeds 3,000,000+.

**Verdict.** The sampler works as specified (generic, hard coarse factors always satisfied, deterministic per seed, cheap at L = 4), but on ground it is target-like only in rho. The certificates come out uniform over the hard-feasible set and independent of rho: only 10-18 % of blocks have A·h ≤ rho ≤ B·h, against 100 % for probe lifts. The fine levels follow cert, not rho, so these contexts behave like random ones in the clamp gate (fine line_match 0.51). On columns and the toys the restricted energy is purely hard, so the counts come out uniform: not target-like. **Do not plug it into cstart as the main source yet;** see "What would work" below.

## What it does
For start depth L it keeps every factor whose participants all sit at levels ≥ L (hard ones included) and runs, per sweep, coarse to fine:
- **Single-site Gibbs** per channel over colour classes derived from the factor offsets and count siblings. Each site is restricted to its least-violating values, so an invalid start gets repaired and a valid state stays valid; after that it is exact Gibbs on the restricted energy at temperature τ.
- **Chain move** over runs of count factors that are the only coupling between two channels (columns n1→…→n4, the toys' rho1→rho2). It is an exact joint draw by upward sum messages and downward sampling. Single-site moves cannot move hard equalities; this move can.

It starts from all-min, all-max, a mix by seed parity, or a given state (for example the model's own draw). Randomness comes from `bench.noise` keyed by (seed, step, index), so a world's draw does not depend on the rest of its batch. Nothing in it refers to channel names.

## Ground, coarse metrics (64 worlds; "line" = the jitter-0 probe lift)

| L | source | dirt | abs err vs line | line_match_L | A/h | B/h | A==B | rho–cert consistent | count residual | hard viol/world |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | probe lift | 0.497 | 0.004 | 0.997 | 0.38 | 0.61 | 0.68 | **1.00** | – | 0 |
| 4 | cfast untrained, own | 0.498 | 0.018 | 0.993 | 0.75 | 0.88 | 0.52 | 0.12 | – | 0 |
| 4 | cfast trained, own | 0.532 | 0.075 | 0.972 | 0.84 | 1.00 | 0.75 | 0.00 | – | 0 |
| 4 | prior, 8 sweeps | 0.497 | 0.016 | 0.994 | 0.44 | 0.55 | 0.35 | 0.18 | – | 0 |
| 4 | prior, 32 sweeps | 0.497 | 0.018 | 0.995 | 0.46 | 0.56 | 0.34 | 0.16 | – | 0 |
| 3 | probe lift | 0.497 | 0.005 | 0.988 | 0.44 | 0.55 | 0.84 | **1.00** | 0.0 | 0 |
| 3 | cfast untrained, own | 0.505 | 0.345 | 0.635 | 0.82 | 0.87 | 0.79 | 0.15 | 72.7 | 12.9 |
| 3 | cfast trained, own | 0.656 | 0.389 | 0.579 | 0.97 | 1.00 | 0.88 | 0.10 | 67.3 | 0 |
| 3 | prior, 8 sweeps | 0.498 | 0.104 | 0.829 | 0.47 | 0.51 | 0.81 | 0.10 | 5.9 | 0 |
| 3 | prior, 32 sweeps | 0.498 | 0.103 | 0.839 | 0.47 | 0.51 | 0.75 | 0.12 | 6.0 | 0 |

Notes on the columns:
- line_match_L is the coarse analogue of `ground.stats` line_match: blocks within one block of the line, with "rho/h² > 0.5" compared against the line's fraction.
- The count residual is mean |Σ children rho3 − rho4|.
- The hard violations are the restricted hard factors (support, row honour) at levels ≥ L.

What the table shows:
- **rho is good.** It follows the fill at L = 4 (as well as the model's draws), and at L = 3 it is much better than the model's own draws (line_match 0.83 vs 0.58, residual 6 vs 67). It is blurred vertically inside a level-4 block, because only the count sum constrains rho3, so abs err is 0.10 vs 0.005 for probe lifts.
- **cert is not target-like.** No soft factor touches cert, so its restricted marginal is uniform over the support/honour-feasible set. Nothing ties it to rho, because the rho–cert link exists only through the tiles.
- **cert mixes slowly at L = 3.** Starting all-min vs all-max, A/h after 8 / 32 / 128 sweeps is 0.13 / 0.29 / 0.46 vs 0.81 / 0.66 / 0.49. At L = 4 the two starts agree by 32 sweeps.

Images (rows: probe lift, untrained own, trained own, prior with 8 sweeps; rho/h² light = air; cert R = A/h, G = B/h): `images/ground_L4.png`, `images/ground_L3.png`. In the prior rows rho follows the line, while cert is a support-driven vertical gradient (L = 4) or sits near its start (L = 3, world 0 started min, world 1 max).

## Clamp gate with prior contexts (fine levels below L, 64 worlds)

| L | model | context | fine cost/cell | fine viol/cell | fine dirt | fine line_match |
|---|---|---|---|---|---|---|
| 4 | untrained | probe lift | 2.32 | 0.019 | 0.57 | 0.54 |
| 4 | untrained | prior (all) | 2.82 | 0.085 | 0.66 | 0.52 |
| 4 | untrained | prior rho4 only, cert drawn by model | 2.49 | 0.041 | 0.88 | 0.50 |
| 4 | untrained | own | 2.49 | 0.041 | 0.88 | 0.50 |
| 4 | trained | probe lift | 1.96 | 0.002 | 0.62 | 0.53 |
| 4 | trained | prior (all) | 1.69 | 0.025 | 0.76 | 0.51 |
| 4 | trained | prior rho4 only | 0.89 | 0 | 1.00 | 0.50 |
| 4 | trained | own | 0.71 | 0 | 1.00 | 0.50 |
| 3 | untrained | probe lift | 1.97 | 0.008 | 0.53 | 0.58 |
| 3 | untrained | prior (all) | 2.14 | 0.034 | 0.54 | 0.51 |
| 3 | trained | probe lift | 2.21 | 0.0005 | 0.55 | 0.56 |
| 3 | trained | prior (all) | 2.18 | 0.009 | 0.54 | 0.51 |
| 3 | trained | prior rho4, rho3 only | 1.00 | 0 | 1.00 | 0.50 |
| 3 | trained | own | 0.34 | 0 | 1.00 | 0.50 |

With prior contexts the fine draw has 4-12× the violations of probe contexts, because cert and rho conflict. When only rho is clamped, the model draws "all full" cert regardless of rho and the fine levels follow cert: fine output stays all dirt. The fine levels read cert, not rho.

## Columns and toys (count-only coarse channels; 2 sweeps; probes are exact target draws)

| problem, L | channel | prior mean / sd | probe mean / sd | hard viol |
|---|---|---|---|---|
| columns, 4 | n4/256 | 0.51 / 0.28 | 0.82 / 0.03 | 0 |
| columns, 2 | n2/16 | 0.50 / 0.31 | 0.82 / 0.18 | 0 |
| toy_support, 1 | rho1/4 | 0.50 / 0.35 | 0.71 / 0.34 | 0 |
| toy_modes, 1 | rho1/4 | 0.50 / 0.34 | 0.50 / 0.44 | 0 |

Zero violations everywhere (the tests check min, max and mixed starts and repair from a random invalid start). The marginals are uniform on the count range, as the restricted energy implies (a test checks it exactly on toy_support). They are not target-like: everything soft in these problems lives at level 0.

## Cost per world (1 CPU core, 8 sweeps)
- **Ground:** L = 4 takes 0.019 s (0.005 s at 2 sweeps); L = 3 takes 0.066 s (0.017 s at 2 sweeps).
- **Columns:** 0.05-0.07 s at L ≤ 3 with 2 sweeps (one chain move is already an exact draw).
- **Toys:** under 1 ms.

cfast's ground update takes about 0.07 s for the whole batch, so drawing 32 clamped worlds per update would cost 0.6-2 s synchronously. Use a pool precomputed on spare cores (any full-support distribution is allowed under #14), or 2 sweeps at L = 4.

## Tried and failed: deeper restriction
Running the Gibbs from L' < L (down to the tiles, L' = 0) and keeping levels ≥ L does not fix cert. rho–cert consistency reaches 0.26 at most (L' = 0, 32 sweeps, 2 s per world), and rho gets worse (L = 3 line_match 0.76), because single-site moves on the hard honour/support lattice are nearly frozen. This is certland's failure mode again.

## What would work (proposals for cstart / the orchestrator)
1. **Start point inside a level** (still ideas #14, generic): clamp only the channels the prior is informative on, i.e. those touched by a soft restricted factor (`informative_channels(problem, L)`: ground L = 4 gives `rho4`, L = 3 gives `rho4, rho3`; empty on columns and the toys). Then let the model sample *and train* everything after the clamped prefix in declared order, including cert_L. ground declares rho_l before cert_l, so at L = 4 "clamp rho4, start at cert4" is a valid prefix. The cert policy then gets gradient in fill-following rho contexts, which neither the own draws nor this prior provide. At L = 3, cert4 sits between rho4 and rho3, so the prefix is only rho4 + cert4, with cert4 the model's own.
2. On columns and the toys the prior adds nothing over uniform counts. Keep probes or own draws there.

## Interface for cstart
```python
from castlegen.variants._coarse_prior import coarse_contexts, coarse_clamp_batch, informative_channels
ctx = coarse_contexts(problem, seeds, Ls, region=None, sweeps=8, tau=1.0, init="mix", chans=None)
# -> {chan: (B, h, w) int64} for every sampled channel at levels >= 1 (or `chans`);
#    world b filled at levels >= Ls[b], 0 below and for Ls[b] > T (cstart's contexts() format)
cl = coarse_clamp_batch(problem, seeds, L, region=None, sweeps=8, tau=1.0, init="mix")
# -> {chan at levels >= L: (B, h, w) int32}, the vmapped clamped_pass clamp format
```

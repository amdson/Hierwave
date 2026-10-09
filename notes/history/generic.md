# The generic constrained texture sampler

Status: history (G1c). Superseded by notes/dsl_updates.md + castlegen/channels. Kept for the record.

Implementation: `castlegen/legacy/generic.py`. Tests: `tests/legacy/test_generic.py`. Experiments: `notes/experiments/legacy/generic_*.py`, whose figures go to `images/generic_*.png`.

The goal is to generate tile maps that look like an exemplar, while *guaranteeing* hard constraints, without any code specific to one constraint, one tile set or one view.

The design keeps a strict division of labour:
- the **texture** supplies appearance and local layout;
- **promises** supply long-range structure and make global constraints local;
- the **rules** are hard local relations between tiles.

Every stage samples. Nothing projects onto a feasible set, and nothing takes an argmin.

## 1. Target

For a tile map x (n × W), latent exemplar coordinates S = {S_h}, and level-K promise grids P_c (one per constraint c):

```
p(x, S | P) ∝ exp(-(E_ex(x, S) + E_loc(x)) / T) · Π_rules 1[rule(x)] · Π_c 1[leaf_c(x) = P_c]
```

**Boundary conditions.** x is periodic horizontally. Vertically, the exemplar's own top row continues forever above the map, and its bottom row continues forever below it. This is the same convention as texture synthesis with `bounds="edge"` (`texsyn.Analysis`, `castlegen/legacy/texsyn.py:65`). No tile is referred to by name: the boundary tiles are read from `E[0]` and `E[-1]`.

**Exemplar energy E_ex.** Multi-scale patch matching against the exemplar on one-hot tile features (`Phi = eye(n_sig) / √2`), i.e. the image itself, as in texture optimisation.
- For each scale h in `w` (default in `generate`: {4: 5.12, 8: 20.48, 16: 81.92}, i.e. 8·h²/25):
  - the features are blurred with a Gaussian of σ = h/2;
  - they are sampled at the centres of an h × h coarse grid, giving `B` (`TileChain._refresh`, `generic.py:581`; kernels `_rows_kernel` / `_cols_kernel`, `castlegen/legacy/exchain.py:57-65`);
  - each sample's 5 × 5 neighbourhood at spacing h (`OFF`, `exchain.py:48`) is compared with the exemplar neighbourhood at its latent coordinate S_h.
- The exemplar bank `_patches_bank` (`generic.py:508`) holds every exemplar column and every centre row from 4h above to 4h below the exemplar, with the same edge continuation. So pure-sky and pure-ground patches are available.
- `VPAD` virtual coarse rows (`exchain.py:49`) carry the boundary features.
- Marginalising over S turns the patch distance into a soft-min, so the chain needs no nearest-neighbour search.

**Local energy E_loc** (`fit_local`, `generic.py:165`): a unary term per tile plus horizontal and vertical pair terms. It is learned from the exemplar by pseudo-likelihood, with the sampler's boundary continuation. It covers the detail finer than the patch scales. It predicts 99% of exemplar tiles from their neighbours, and puts negligible mass on tiles the exemplar never uses.

**Rules.** Hard pair relations, as `(2, S, S)` bool arrays: `[0]` left|right, `[1]` upper/lower. The only rule so far is `support_rule` (`generic.py:157`): a solid tile needs a solid tile below. Boundary rows are checked like any other neighbour.

## 2. The constraint interface

| member | meaning |
|---|---|
| `K`, `V` | leaf block side; number of values |
| `leaf(blocks)` | `(..., K, K)` tiles → value of each K × K block |
| `merge(children)` | `(..., 2, 2)` child values `[dy, dx]` → the parent value (exact, by definition) |
| `compat` | `(2, V, V)` hard seam rule between adjacent level-K blocks, chosen so that a level-K grid is realisable **iff** every seam is compatible |
| `dist(a, b)` | distance between values, used only by the texture coupling |
| `witness(P, fill, empty)` | a plain tile map realising a compatible grid (constructive proof of the "if" direction; used in tests and renders) |
| `build_order(R, C)` | the order blocks are realised in by the sequential witness |
| `block_start(P, by, bx, x, …)` | a realisation of one block that is valid given the blocks realised before it |
| `extendable(blk, by, bx, P)` | whether a block keeps the blocks not yet realised realisable |

The last three describe how to build a realisation block by block. The sampling inside each block (`guided_witness`) is generic.

The only implementation is `AverageFill` (`generic.py:65`), the average-height promise:
- **Leaf value:** each block has two sub-columns of w = K/2 columns. b_j = ⌊solid cells in sub-column j / K⌋ ∈ 0..w, so V = (w+1)² = 81 at K = 16.
- **Merge:** a parent's b_j = ⌊(sum of the four child sub-column values under it) / 4⌋.
- **Seam rule (vertical):** `b_u == 0 or b_u <= b_l` per sub-column. Upper matter needs at least b_u full columns below it, and the lower block holds at most b_l. There is no horizontal rule.
- **Distance:** L1 over the sub-columns. This is the one hand-chosen piece; the generic default would be Potts.
- **Build order:** bottom-up, since support points downward.
- **Extendable:** at least b_u full columns in each sub-column, where b_u belongs to the block above.
- **Block start** (`generic.py:123`): see stage 3.

`TileChain` takes lists of constraints and grids, and `_allowed` checks all of them. So far only one constraint has been run.

## 3. Stages

`generate` (`generic.py:762`) runs all three stages for one seed. It optionally takes a precomputed texture (`x_tex`) and a separate seed for stages 2–3 (`sub_seed`).

### Stage 1: texture (`texture`, `generic.py:740`)

Texture synthesis in the style of Lefebvre & Hoppe, as a hierarchical variable:
- `texsyn.CoordVar` (`texsyn.py:183`) is run by `hier.run` (`castlegen/legacy/hier.py:168`).
- The exemplar is bounded (`Analysis(..., bounds="edge")`): out-of-map neighbours read sky above and ground below (`CoordVar._nb`, `texsyn.py:204`).
- Corrections sample by Gumbel-max at `coord_T` (`hier.pick`, `hier.py:156`); they run from level 1, with `kappa=4`.
- **Top level.**
  - **Default:** one cell as large as the map. Its coordinate is one uniform exemplar position, and every level below is an exact offset of it (`r=(1, 1)` jitters only the top). So the layout is essentially one random crop of the exemplar.
  - **`h_top=h`:** every cell of side h gets an independent uniform exemplar coordinate. `h_top=32` gives a varied surface distribution; 16 is mostly sky with floating scraps.

The output `x_tex = E[S]` carries appearance and layout, but satisfies no constraint.

### Stage 2: promises (`PromiseSampler`, `generic.py:447`)

Only the level-K grid P_K is a hard variable. The coarser grids P_h = merge^h(P_K) (`pyramid`, `generic.py:264`) are exact functions of it, and contribute only soft energy:

```
E(P_K) = Σ_h [ Σ u_h(P_h) + Σ pair_h(neighbours) + Σ parent_h(P_h, P_2h) + lam · Σ dist(P_h, T_h) ]
```

- **Tables.** Two fits of the same energy form:
  - `fit_tables` (`generic.py:282`): counts with additive smoothing. A unary term −log p; pair terms as negative PMI (periodic horizontally, no vertical wrap); parent terms as negative PMI per child position.
  - `fit_tables_pl` (`generic.py:349`): **pseudo-likelihood**. Every block's conditional given the rest of its grid is fitted to the corpus. That's exactly the distribution the heat-bath draws from, with merges, compat and boundary values included. It uses a ridge penalty l2 and is fitted with jax and L-BFGS; the candidate grids are precomputed as parameter indices (`_energy_index`). `pl_nll` evaluates held-out pseudo-NLL.
- **Texture coupling.** T_h = merge^h(leaf(x_tex)) are the texture's own values. Each unit of L1 disagreement at each level costs `lam`, so the probability falls by e^−lam per unit at `T_prom = 1`.
- **Feasibility is exact and local.** P_K is compatible at every seam; the boundary blocks come from `boundary_values` (`generic.py:274`).
- **Sampler** (`sweep`, `generic.py:481`): an exact heat-bath per block over all V values, restricted to values compatible with the four neighbours.
- **Start.** All-sky. It is metastable under single-block moves, so `run` (`generic.py:496`) anneals geometrically from `T_hot = 10` to T over the first half of the sweeps, then samples at T.

### Stage 3: tiles (`TileChain`, `generic.py:527`)

**Start: `guided_witness`** (`generic.py:203`). A random feasible realisation of P_K. It satisfies every rule and every promise, which `valid` asserts.
- Blocks are realised one at a time in `con.build_order`. Each starts from `con.block_start`, then gets a few single-site heat-bath sweeps of

  q(block) ∝ exp(−E_loc(block | realised cells) − #(cells differing from the hint) / τ)

  - E_loc counts only pairs with realised cells and the boundary.
  - The hint term is optional (hint = the texture, τ = 0.1 by default).
  - Sweeps are raster scans alternating bottom-up and top-down, so a run of cells can grow or shrink in one sweep.
  - Candidates are limited to tiles that keep the block's leaf value, every rule against realised cells and the boundary, and `con.extendable`.
- **`AverageFill.block_start`**, per sub-column:
  - **Total:** drawn uniformly from the realisable part of the bin.
  - **Full columns:** only under a non-empty block above. Then b_j of them, the widest support the block's own mass allows (at least the b_u needed), as one run of consecutive supported columns at a random position.
  - **The rest:** spread evenly, bottom-aligned, over the other supported columns.
  - So a partly filled block starts as a plateau, and a block above can spread its matter.
  - Earlier versions started from b_j scattered full columns. Those are thin, and single-site moves can't move them afterwards (see §5), so they became needles.

**Move: single-site heat-bath over all tiles** (`site_sweep`, `generic.py:710`).
- `_allowed` (`generic.py:672`) keeps only tiles for which every rule holds against the four neighbours (boundaries included), and every constraint's `leaf` of the site's block still equals its promise.
- Allowed tiles are weighted by exp(−(ΔE_ex + ΔE_loc) / T).
- `dE` (`generic.py:689`) is exact. E_ex is quadratic in B, so

  ```
  ΔE_h = w_h Σ_s [ 2 (c(s) B(s) − Q(s)) · ΔB(s) + c(s) ΔB(s)² ]
  ```

  where c(s) counts the real neighbourhoods containing sample s, and Q(s) sums their matched exemplar values (`_Q`, `generic.py:592`). ΔE_loc involves only the four neighbours.
- **Parallel batches:** sites more than 3·h_max + 1 apart share no coarse sample, no rule and no K-block. So each batch of an offset grid is updated exactly in parallel.

**Latent coordinates.** Every `s_every` sweeps, `resample_S` (`generic.py:613`) updates every S_h(s); given x they are independent.
- **Exact heat-bath** over all exemplar positions when (coarse samples × bank size) ≤ `exact_budget`.
- **Metropolis–Hastings** otherwise (`_mh_S`, `generic.py:628`); only needed if fine scales h = 1, 2 are used. The proposal depends only on x and the other S_h:
  - with probability ε, uniform over the bank;
  - otherwise uniform over a fixed list: the k nearest bank patches in PCA space, plus the neighbours' coordinates shifted back by their offset. A neighbour off the map contributes a kNN entry instead.

  So q(u) = ε/N + (1−ε)·mult(u)/L, and the acceptance ratio is exact.

**Deliberately removed**, relative to `exchain.ExChain` / `average_var.tile_stage`:
- moves restricted to the same solidity class;
- column height-jump moves;
- `project` (greedy argmin toward the texture);
- the ground band;
- the tile set's pair energy and base mass;
- hard-coded "air" and "stone" boundary features.

## 4. Verified

`tests/legacy/test_generic.py`:
- the plain witness realises its promises and satisfies the support rule;
- the seam rule is necessary: the leaf grid of every random heightmap is compatible;
- the guided witness is feasible for any hint, even random invalid maps, and reproduces a feasible terrain-like hint to more than 99%;
- predicted ΔE_ex + ΔE_loc equals the actual change for every allowed move tried (scales 1, 2, 8);
- rules, promises and `B` bookkeeping (drift < 1e-9) hold across sweeps;
- with x fixed, the MH update of S_h reaches the exact conditional (error falls like 1/√steps);
- the fitted E_loc predicts the exemplar's tiles and gives unused tiles negligible mass.

## 5. Known limitations and open questions

- **Single-site moves cannot relocate a full column.** A sub-column's total must stay within its bin, which is exactly K cells (one column) wide, and `extendable` requires b_u full columns. When b_u = b, moving a full column would need (b+1)·K cells at some point, so full-column positions are frozen for good. That applies to the tile chain and to the witness's sweeps. The fix under consideration: constraints supply value-preserving moves (for `AverageFill`, swapping two columns within a sub-column across a vertical run of blocks), accepted by Metropolis–Hastings.
- **The witness builds bottom-up.** A lower block commits its full columns before the block above is shaped. The widest-support rule makes this benign for plateaus; the block above still has to stand where those columns are.
- **One-way stages.** Texture → promises → tiles. Texture synthesis never sees the promises, and the tile chain's S_h are separate from the texture's S. So the output is not a sample of one joint model.
- **The promise model is only as good as its 16-map corpus.** Pseudo-likelihood tables are about twice as variable as the corpus.
- **Cost at 128².**
  - Texture analysis: about 26 s, cached across seeds.
  - Tile-chain setup: about 5 s. One tile sweep: about 0.8 s.
  - Guided witness: a few seconds.
  - Pseudo-likelihood table fit: about a minute.
- **Hand-set parameters:** `w`, T, `coord_T`, `lam`, `T_hot`, τ, `AverageFill.dist`, and the ridge penalties.

## 6. Findings, in order

1. **Coarse patch scales only ⇒ speckle** (`images/generic.png`). A single wrong tile barely changes any blurred sample at σ ≥ 2, so unused tiles and speckle appeared.
2. **Fine patch scales (h = 1, 2) are slow and not enough** (`images/generic_fine.png`, about 290 s per seed). They were replaced by E_loc, which gives clean materials at about 80 s per seed (`images/generic_local.png`).
3. **The texture pull `lam`** (`notes/experiments/legacy/generic_lam.py`, `images/generic_lam.png`), with count tables:

   | `lam` | layout |
   |---|---|
   | 0 | collapses to empty |
   | 1 | flattened |
   | ≥ 4 | follows the texture |

4. **The start is the output unless it comes from the texture.** From the plain witness, the comb teeth remained after 50 sweeps. A texture-guided start at τ = 0.1 reproduces the texture wherever it is feasible (`images/generic_guided.png`). The chain then drifts toward rounder shapes; comparing E(start) with E(final) is still to do.
5. **Diversity** (`images/generic_diversity.png`). With the texture fixed, stages 2–3 add almost none (1.3% of cells differ between downstream seeds, against 17% between textures). The default texture is one random crop, and seeds 1–3 happened to land on nearly the same crop.
6. **Finer top levels** (`images/generic_top16.png`, `images/generic_top32.png`). 32 px gives a varied surface distribution, lower than the corpus (0.13 solid).
7. **Pseudo-likelihood tables** (`notes/experiments/legacy/generic_tables.py`, `images/generic_tables.png`):

   | | held-out pseudo-NLL per block | fill, tables alone | top height, tables alone |
   |---|---|---|---|
   | counts | 1.72 | 0.01 | 3 |
   | pseudo-likelihood, l2 = 1 | 0.94 | 0.45 ± 0.20 | 80 ± 25 |
   | corpus | – | 0.39 ± 0.08 | 71 ± 11 |

8. **With the old witness, 32 px textures and pseudo-likelihood tables** (`images/generic_combo.png`): `lam` from 1 to 4 barely changes anything (all within about 2% of the texture, 0.13 solid). The needles came from the old witness.
9. **Witness fixes** (`notes/experiments/legacy/generic_render.py`, `images/generic_render.png`). Realising corpus promises:

   | start blocks | needles per map | E_loc gap to the true map |
   |---|---|---|
   | scattered full columns | 5–10 | about 2,500 |
   | plateau + minimal full columns | 0–3 | about 1,000 |
   | plateau + widest support | 0–1 (true maps: 0–1) | about 1,000 |

   The narrow towers that remain in table samples come from the tables' own small-fill high blocks.
10. **Rule-aware texture synthesis** (`RuleCoordVar`, `texture(..., rule, gamma)`, `notes/experiments/legacy/generic_rule.py`, `images/generic_rule.png`). Corrections pay γ × the fraction of rule-forbidden pairs on each seam between a candidate's implied h × h exemplar window and its neighbours' windows (or the boundary rows). It uses only the generic rule. Coupling the texture to the promises instead would not remove needles: promises sampled from a texture with a floating slab prefer propping it up with a 1 px pillar over deleting it.

    | 32 px start, 4 seeds | γ = 0 | γ = 10 | γ ≥ 100 |
    |---|---|---|---|
    | unsupported texture cells | 0, 173, 10, 14 | 0, 28, 0, 0 | 0, 0, 0, 0 |

    At γ = 100 textures are supported, higher (0.16 solid) and more varied (0.181 of cells differ between seeds, against 0.139).
11. **Hint-aware `block_start`.** With a hint, the sub-column total and the position of the full-column run are drawn with weight exp(−mismatch / τ), not uniformly. This removed towers split into pillars, whose runs used to land away from the texture's tower edge.

    With γ = 100 textures, `lam = 4` and pseudo-likelihood tables, the outputs follow their textures closely, with 0 needles on all 4 seeds (`images/generic_rule.png`). Now that the texture is feasible, a high `lam` costs nothing.

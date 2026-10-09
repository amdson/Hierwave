Soft Tile Synthesis: Parallel Controllable Texture Synthesis as a WFC Alternative
Sep 25, 2026 · @Andrew
Summary
Run Lefebvre and Hoppe's parallel controllable texture synthesis (paratexsyn, SIGGRAPH 2005) on a tile map instead of an image, as a soft, parallel, multiscale alternative to Wave Function Collapse (WFC). The output is a map of exemplar coordinates, so every output cell is a real exemplar cell and locally valid regions are exact copies; adjacency violations occur only at seams between copied patches.
The algorithm is generic: it needs only an embedding of tiles into a vector space and a linear blur on that space. That generality is why it is the first milestone. Multiple exemplars, label control and tile symmetries all reduce to bookkeeping on the coordinate, and a later hard-constraint stage (WFC on seams) reuses the same coordinate map.
What you gain over WFC: order-independent parallel evaluation, deterministic windows of an unbounded output, per-scale randomness sliders, and no contradictions. What you give up: hard adjacency validity and long-range constraint propagation.
Background: the 2005 algorithm
Paratexsyn synthesizes a coordinate map, not colors. Given an m×m exemplar E, the output at level l is S_l[p] ∈ [0,m)², and the displayed value is E[S_L[p]] where L = log2 m. Three properties follow from storing coordinates: coherent candidates are cheap (the neighbor's coordinate plus an offset), upsampling is exact, and any attribute attached to E can be looked up after synthesis.
Gaussian stack. Level l of the stack is E blurred with a kernel of width proportional to h_l = 2^(L−l), kept at full m×m resolution. A pyramid level is the stack level sampled on the sublattice h_l·Z²; the stack is the union of all h_l² translates of the pyramid. The 5×5 neighborhood at level l samples E_l[u + h_l δ], δ ∈ {−2..2}², so it spans 4·h_l exemplar cells. The blur is what makes those sparse samples a valid summary of the surrounding h_l × h_l cells. Coordinates stay unquantized at every level, which is what makes jitter continuous and avoids coarse-grid alignment artifacts.
Preprocessing
Preprocess(E):
  E_aug := E padded to 2m×2m (tile if toroidal)
  for l in 0..L:
    h_l := 2^(L-l)
    E_l := GaussianFilter(E_aug, sigma ∝ h_l), restricted to m×m
    for each u:
      N_E_l(u) := concat of E_l[u + h_l·δ] for δ in {-2..2}²   # 5×5 window
    P_l := top-k PCA basis of {N_E_l(u)}
    for each u:
      Ñ_E_l(u) := P_l · N_E_l(u)
      C_l(u) := [u, u']   # u' = best 7×7 match with |u'-u| ≥ 0.05 m
Synthesis
Synthesize(r_0..r_L):          # r_l ∈ [0,1]: per-level jitter amplitude
  S_{-1} := all (0,0)
  for l in 0..L:
    S_l := Upsample(S_{l-1}, l)
    S_l := Jitter(S_l, l)
    if l > 2:
      repeat c=2 times: S_l := Correct(S_l, l)
  return S_L

Upsample(S_par, l):
  for each p, Δ ∈ {0,1}²:
    S_l[2p+Δ] := (S_par[p] + h_l·(Δ − (½,½))) mod m

Jitter(S, l):
  for each p:
    J := floor(h_l · r_l · H(p) + (½,½))    # H: Z² → [-1,1]², a hash
    S[p] := (S[p] + J) mod m

Correct(S, l):
  for (i,j) in [(0,0),(1,1),(1,0),(0,1)]:      # 4 interleaved subpasses
    S_new := S
    for each p with p mod 2 = (i,j), in parallel:
      Ñ_S(p) := P_l · concat of E_l[S[p+δ]] for δ in {-2..2}²
      best := ∞
      for Δ ∈ {-1,0,1}², i_c ∈ {1,2}:
        u := (C_l(S[p+Δ])[i_c] − h_l·Δ) mod m
        cost := ‖Ñ_S(p) − Ñ_E_l(u)‖² · (1 if i_c=1 else κ)   # κ > 1
        if cost < best: best, u_best := cost, u
      S_new[p] := u_best
    S := S_new
  return S
With all r_l = 0 the output is the exemplar tiled exactly. Correction is skipped at levels 0–2 because at level 0 every stack sample equals the mean and at levels 1–2 it locks coarse feature alignment.
Determinism. Each subpass reads a radius-2 window; c·s² = 8 subpasses give a dependency radius of 16. A window W_l is computed deterministically from the padded window W_l ⊕ 16, which needs ⌈W_l/2⌉ ⊕ 16 at the coarser level, and so on to level 0.
Known weakness. The cost is an L2 distance on a 5×5 window, so a misplaced edge and a slightly recolored flat region are penalized comparably. Structures larger than 4·h_l at the finest corrected level are assembled from fragments whose seams the metric does not see. The paper's own examples of failure are text and other semantic structures.
Relationship to Wave Function Collapse
Paratexsyn is WFC's overlapping model with the constraint relaxed from "overlap error = 0" to "argmin overlap error over ≤ 18 candidates", plus a multiscale prior and no propagation.
WFC's overlapping model assigns each output cell a pattern id, which is an exemplar coordinate u naming an N×N window, under the hard constraint that adjacent cells' windows agree on their overlap. Paratexsyn assigns each cell S[p] and minimizes ‖N_S(p) − N_E(u)‖² over candidates C(S[p+Δ]) − h_l Δ. The coherent candidate (i_c = 1) is "extend the neighbor's window", which has zero overlap error by construction. A seam is any p with S[p+Δ] − S[p] ≠ Δ for some Δ, and adjacency violations can only occur at seams.
Property
WFC (overlapping)
Paratexsyn
Local validity
exact
argmin over candidates; violations at seams
Propagation
to arbitrary distance
none beyond the 5×5 window
Scale structure
none
multiscale pyramid, per-level jitter r_l
Evaluation
sequential, global
parallel, order-independent
Windowed determinism
no
yes, dependency radius 16
Failure mode
contradiction, backtrack or restart
seam artifacts
Novelty
any arrangement propagation allows
rearrangement of exemplar patches
Neither method enforces global constraints (connectivity, counts) beyond what WFC's propagation happens to imply.
Design A: categorical paratexsyn
The synthesis loop is unchanged. Everything specific to tiles lives in the exemplar format, the embedding φ, the stack construction, and the coherence-set predicate.
Exemplar format. An m×m integer array E_id ∈ {1..T}^(m×m), m a power of two. Sources, by which WFC model you come from:
• Overlapping model: the input bitmap; E_id is the pixel value. This case is the original paper with φ = color.
• Hand-authored tilemap: use as-is.
• Simple-tiled model: there is no spatial exemplar, and paratexsyn cannot run without one. Run WFC once at m×m (toroidal if possible) and use its output as E_id. It is valid everywhere and stationary by construction.
Embedding. φ: tiles → R^d, stacked per cell. Start with the one-hot row only.
Channel
Dim
Source
Role
Tile one-hot
T
E_id
base metric; ‖φ(a)−φ(b)‖² = 2·[a≠b]
Socket one-hot, per edge
4·S
tile definitions
incompatible neighbors far apart
Region label one-hot
R
author or segmentation
texture-by-numbers control, stationarity
Signed distance to label boundary
1
distance transform
ASTS-style edge and shape integrity
Learned or hand-set embedding
e
later
graded tile similarity
Give channels weights w_c by scaling each by √w_c before PCA, so a socket mismatch can cost more than a tile mismatch.
Stack. E_l = G_σl * φ(E), σ_l ∝ h_l, on the padded 2m×2m exemplar. On one-hot data E_l[u] is the tile-type composition around u at scale h_l. Keep it as float; do not renormalize or argmax. Skip the paper's PCA-to-2-color-channels step.
Padding: the paper mirrors, which is wrong for tiles (mirrored rows can violate adjacency; asymmetric tiles have no mirror). In order of preference: (1) toroidal exemplar, tiled; (2) generate at 2m×2m and address the central m×m; (3) replicate-pad and mark padded cells invalid as coherence targets.
Neighborhoods and PCA. N_E_l(u) has dimension 25d. For T ≈ 20–50 expect 16–32 PCA components, not 6; check explained variance. A 3×3 window at the finest level is worth trying, since a tile is already a feature.
Coherence sets. C_l(u) = [u, u'], u' the best 7×7 match at distance ≥ 0.05 m. Two changes: break ties with the hash (repeated regions produce many exact ties, and a deterministic tie-break makes one u' dominate); and put an optional predicate compatible(u, u') on the candidate, initially always true. Design B fills it in.
Synthesis loop. As in the background pseudocode. Two details: with a non-toroidal exemplar, store S as coordinates into the padded exemplar and clamp candidates to the interior rather than wrapping; and implement the 4-subpass checkerboard in the paper's order. A single full parallel pass oscillates on categorical data because there is no soft gradient to damp it.
Output and diagnostics. T[p] = E_id[S_L[p]]. Also emit the seam mask M[p] = [∃Δ: S_L[p+Δ] − S_L[p] ≠ Δ] and the adjacency-violation mask. The violation rate as a function of (r_l) is the primary diagnostic: exactly 0 at r = 0, growing with coarse-level jitter. Violations outside M indicate a bug.
Extensions
All three keep the invariant "output = coordinates, tiles are looked up", so the synthesis loop is untouched; only stack construction and coherence-set precomputation change.
Multiple exemplars reduce to one exemplar with a disconnected domain. Concatenate E^1..E^n into one array, so S[p] = (e, u) is a coordinate in a larger domain. The paper's border rule (u within 2 cells of a border ⇒ C(u) ∌ u) already forces neighborhoods that would straddle a boundary to jump elsewhere; the union just has more borders. Upsampling and jitter stay inside E^e (mod is per-component, u mod m_e). Cross-exemplar transitions happen only through the second entry of C(e,u), computed by the same 7×7 search over the union in the shared φ space. Since all exemplars share the tile vocabulary, φ is common and cross-exemplar distances are meaningful.
Label control (texture-by-numbers). Left alone, the union picks exemplars by local similarity, so one exemplar dominates. To steer it, paint a target label field Λ[p] over the output and add a term to the correction cost:
cost(u) = ‖Ñ_S(p) − Ñ_E(u)‖² + λ · ‖ℓ(u) − Λ_l[p]‖²
where ℓ(u) ∈ R^n is the one-hot exemplar id of u and Λ_l is a mipmap of Λ. This is Hertzmann's texture-by-numbers; Busto, Eisenacher, Lefebvre and Stamminger (2010) put exactly this term in the parallel runtime. The known difficulty is the transition band: with no exemplar containing an E^1 → E^2 transition, neighborhoods there match nothing and correction degrades to noise. Fixes in the literature are transition exemplars and faster search (Disney's parallel coherent random walk). For tiles, the adjacency rules define what a transition may be, which is Design B's seam solver.
Symmetry group coordinates. Let S[p] = (e, g, u), g ∈ D4 or the subgroup the tileset supports: cell p shows exemplar e at u transformed by g. The neighborhood at p reads E^e[u + g⁻¹δ] with φ transformed accordingly (permute socket channels, map tile ids to rotated variants). Coherence becomes (e, g, u + g⁻¹Δ); jitter is unchanged; coherence sets include rotated matches by searching the rotated stacks. This is WFC's symmetry expansion done at the coordinate level, multiplying the effective exemplar by |G| at zero storage cost. A tileset with no rotational variants collapses to the trivial group.
Further out. Continuously parameterized exemplars (Matusik et al. 2005) make e a point in a simplex; procedural exemplars replace the array with a function E(u) on an unbounded domain and drop the mod m.
Design B: soft layout, hard seams
Use Design A's coordinate map at a coarse level to decide the macro layout, copy exemplar blocks verbatim where the map is contiguous, and run WFC only on the seams. Solve cost then scales with seam length, not area.
1. Coarse exemplar. Pick a block size B (8–16 tiles). Run paratexsyn down to the level l* with h_l* = B, using the categorical stack. Output: one exemplar coordinate S_l*[p] per output block.
2. Classify blocks. For each block, test its 4 neighbors: if S[p+Δ] − S[p] = B·Δ, the two blocks are exemplar-contiguous and the tiles are copied verbatim with guaranteed validity (this is synthesis magnification with the fine exemplar as E_H). Only blocks on a coordinate discontinuity need solving.
3. Seam solving. For each seam, run WFC on a strip of width w straddling it, with hard boundary constraints from the copied tiles on both sides and domains weighted by the referenced exemplar region's tile distribution. Corners where four blocks meet are w×w squares with four boundary constraints.
4. Solvable by construction. Precompute on the exemplar's pattern-adjacency graph which pairs of windows (u, u') admit a valid bridge of width w (for a 1D seam, w-step reachability in the adjacency graph; corners need a 2D version). Restrict the coherence predicate compatible(u, u') to bridgeable pairs. Then the seam WFC never contradicts and the pipeline stays deterministic per window with dependency radius w.
5. Determinism. Seed each seam solver from the hash H(p) at the seam's coarse position so any window is recomputable independently. Sequential chunked WFC breaks this; overlapping padded seam regions with fixed-radius dependency preserve it.
Gained over plain WFC: parallelism, unbounded windowed output, per-scale randomness controlling how often the layout jumps between exemplar regions, solve cost proportional to seam length. Lost: macro-novelty is bounded by B and the jitter, since the output only recombines exemplar blocks.
Related prior work to read first: Merrell's Example-based Model Synthesis (2007), from which WFC descends and which already does block-wise parallel modification under hard adjacency; and the hierarchical or nested WFC variants that use a coarse WFC to constrain a fine one. Design B differs in that the coarse layer is soft, multiscale and coordinate-based, which is what buys the copy-when-contiguous shortcut in step 2.
Limitations and open questions
Large structures. A motif of diameter D survives only if (a) some corrected level has D ≲ 4·h_l and the motif is distinguishable from everything else at that level's blur, and (b) jitter is near zero at every level with h_l < D. At levels where the motif spans several coarse pixels, per-pixel jitter fragments it with probability ≈ 1 − (1−q)^(n−1) for n ≈ (D/h_l + 1)² pixels and q ≈ P(h_l r_l H(p) rounds to nonzero). Where D < h_l the motif is sub-pixel and moves as a unit. Condition (b) is arrangeable with the sliders or a painted R_E, but it means the motif only appears as a rigid copy. This is why the method does well with terrain features and poorly with text, where every glyph is a similar-sized blob.
Transitions between exemplars. The one real research question in the extensions. With no exemplar containing a transition, the metric has nothing to match in the band between two exemplars. Candidate answers: authored or synthesized transition exemplars; the adjacency-rule seam solver of Design B; or accepting a violation rate.
Stationarity. Synthesis reproduces neighborhoods, not layouts. A hand-drawn level with distinct regions yields a mixture, not the regions, unless the region label channel and a painted Λ are used.
Global constraints. Neither paratexsyn nor WFC enforces connectivity, counts or reachability. Any such requirement needs a separate post-check or a different model.
Metric blindness. L2 on one-hot cannot express adjacency legality; two tiles can be visually similar and adjacency-incompatible. Socket channels mitigate this; only Design B removes it.
Open questions
[ ] How many PCA components are needed for T ≈ 30 tiles with socket channels?
[ ] Does a 3×3 finest-level window lose measurable quality on tile data?
[ ] What violation rate is acceptable for the intended use, and at what (r_l) is it reached?
[ ] For Design B, what seam width w makes most exemplar window pairs bridgeable?
Implementation plan
Build the pure-coherence version first; each later step is independently testable against the violation-rate curve.
Step
Scope
Test
1
Exemplar loader, one-hot φ, categorical Gaussian stack
stack level 0 equals the global tile histogram
2
Synthesis loop with C_l(u) = [u] only, no PCA, 4-subpass correction
r = 0 reproduces the exemplar tiling exactly; violations ⊆ seam mask
3
PCA on neighborhoods
same output as step 2 within tolerance; speed gain
4
k = 2 coherence sets with hashed tie-break
more variety at fixed r; violation rate unchanged or lower
5
Socket channels and channel weights
violation rate drops at fixed r
6
compatible(u, u') predicate, initially always true
hook in place for Design B
7
Multi-exemplar union, label term λ
painted Λ is respected; transition band quality
8
Symmetry group coordinates
rotated tiles appear; validity preserved under g
9
Design B: block classification, seam WFC, bridgeability precompute
zero violations; solve time ∝ seam length
Steps 1–2 are a few hundred lines of numpy; the categorical stack is the only nontrivial part. Preprocessing cost is O(m² · 49d) per level for the coherence search, so watch d rather than m; m = 64 or 128 is enough for tiles.
References
• Lefebvre, S. and Hoppe, H. Parallel Controllable Texture Synthesis. SIGGRAPH 2005. PDF
• Lefebvre, S. and Hoppe, H. Appearance-Space Texture Synthesis. SIGGRAPH 2006. Project page — signed feature distance in the matching vector; the fix for sharp structures.
• Wei, L.-Y. and Levoy, M. Order-Independent Texture Synthesis. Stanford TR 2002 / 2003 — the parallel correction scheme paratexsyn extends.
• Ashikhmin, M. Synthesizing Natural Textures. I3D 2001 — coherence search.
• Tong, X. et al. Synthesis of Bidirectional Texture Functions on Arbitrary Surfaces. SIGGRAPH 2002 — k-coherence.
• Hertzmann, A. et al. Image Analogies. SIGGRAPH 2001 — texture-by-numbers.
• Busto, P., Eisenacher, C., Lefebvre, S. and Stamminger, M. Instant Texture Synthesis by Numbers. VMV 2010 — label term in the parallel runtime.
• Manke, F. and Wünsche, B. Analysis of Appearance Space Attributes for Texture Synthesis and Morphing. IVCNZ 2009. IEEE
• Kaspar, A. et al. Self Tuning Texture Optimization. Eurographics 2015. EG digital library — automatic guidance channels for mid-scale structure.
• Zhou, Y. et al. Non-Stationary Texture Synthesis by Adversarial Expansion. SIGGRAPH 2018. arXiv
• Merrell, P. Example-based Model Synthesis. I3D 2007 — WFC's ancestor; block-wise parallel modification under hard adjacency.
• Gumin, M. WaveFunctionCollapse. 2016. GitHub repository.
• Matusik, W. et al. Texture Design Using a Simplicial Complex of Morphable Textures. SIGGRAPH 2005.
• Zhang, J. et al. Synthesis of Progressively-Variant Textures on Arbitrary Surfaces. SIGGRAPH 2003 — texton masks.
• Barnes, C. and Zhang, F.-L. A Survey on Patch-based Synthesis: GPU Implementation and Optimization. 2020. arXiv
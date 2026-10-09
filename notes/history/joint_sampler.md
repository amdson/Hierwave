# Joint sampler: plan

Status: history (G1c). Superseded by notes/dsl_updates.md + castlegen/channels. Kept for the record.

Checkerboard Gibbs version of Lefebvre–Hoppe over exemplar coordinates and
promise values at every level, with parent terms.  Replaces the texture →
promises → tiles split in castlegen/legacy/generic.py.

## Target

p(x) ∝ exp(−E(x)) · 1[x satisfies the constraints], approximated top-down:
at each level h, sample p(u_h, v_h | u_2h, level-h consistency).

## State (per level h = h_top … 1, grid n/h × n/h)

n = output side, m = exemplar side (m² possible coordinates; cliffs_big: m = 256).

- u[p]  exemplar coordinate (centre of an h × h exemplar window)
- v[p]  promise value, V small (SurfaceClass at scale h: E/S/P/F, V = 4; h = 1: E/F)
- h = 1: tile x[p] = E[u[p]], v[p] = leaf_1(x[p]) (the solid bit)

## Terms (all defined, none learned)

- E_c(u)   symmetric patch energy: Σ_q ‖N_S(q) − N_E(u_q)‖²
           N_S(q) = exemplar features at u of q's 5×5 neighbours, N_E(u) = exemplar's own
           5×5 neighbourhood at u.  Changing u_p touches the 25 terms q with |q − p| ≤ 2
           (not L–H's one-sided cost at p)
- E_par(u) Σ_p ‖wrap(u_p − (u_parent(p) + off_p))‖² / (2 (r h)²)   parent prolongation
           = −log q_jit(u | prolong(u_2h)): an energy, not just the init, so it enters
           the joint (u, v) conditional.  Unary during level h's sweeps (u_2h fixed).
           Needs a soft kernel with r > 0 at every level: today's uniform box with
           r = [1, 1, 0, …] would be a hard constraint and freeze the fine levels.
- cost(u,v) −log p_jit(v | u): histogram of leaf_h over exemplar windows within r·h of u,
           a (m², V) table per level, smoothed toward the level's marginal:
           p(v | u) = (c_uv + α p_h(v)) / (N_u + α).  Zero counts would add false hard
           constraints (Z[u] = 0, dead-end cells); backing off to p_h keeps v's that are
           impossible at level h (e.g. S and P at h = 1) at zero
- compat   hard same-level seam table (compat_h; the support argument is scale-free);
           h = 1: the pair rule

Cell p's joint conditional:
  π(u, v) ∝ exp(−ΔE_c(u) − E_par(u) − cost(u, v)) · 1[v compatible with the 4 neighbour v's]

## Pseudocode

```
for h in h_top, h_top/2, …, 1:
    if h == h_top:                                  # h_top = 32: coord and promise grids matched,
        u ← uniform exemplar coordinates            # no parent, so E_par ≡ 0; exact sweeps (16 cells
                                                    # on 128², one per colour), annealed T_hot → T
    else:
        u ← prolong(u_2h) + jitter(r h)
    v ← sample v | u (exact, per cell, given compat with already-set neighbours)
    for s in 1..sweeps(h):
        for colour in colours(spacing 5):          # cells ≥ 5 apart share no E_c term, no seam
            for p in colour (in parallel):
                U  ← candidates(p)                  # coherent + C2 + parent prolongation + ε uniform
                if exact (|U| = m²):
                    Z[u] ← Σ_v exp(−cost(u,v)) · 1[compat(v, nbr v)]
                    u    ~ exp(−ΔE_c(u) − E_par(u)) · Z[u]
                else:                               # MH on u with v marginalised
                    u'   ~ q(· | p)
                    accept w.p. min(1, π̄(u') q(u | p) / π̄(u) q(u' | p)),  π̄(u) = exp(−ΔE_c − E_par) · Z[u]
                v ~ exp(−cost(u, ·)) · 1[compat(·, nbr v)]
x ← E[u_1]
```

## Guarantees

- Exact: support at h = 1 (compat there is the rule).
- Approximate: at h > 1, samples ≈ p(u | u_2h, consistency at level h).
- Not guaranteed: realisability across levels. v_2h and v_h have no hard merge
  relation (for support, no small V is closed under merge).

## Steps

1. E_c symmetric patch energy and ΔE_c for one cell, reusing texsyn.Analysis levels
   (El, NE, PCA).  Test: ΔE_c = E_c(after) − E_c(before) on random grids.
2. E_par, and colouring with spacing 5.  Test: a parallel batch gives the same result as
   sequential single-cell updates.
3. cost_h tables and compat_h for SurfaceClass at every h.  Test: cost rows are
   normalised; compat_1 = the support rule.
4. Joint (u, v) update, exact and MH.  Test: on a small exemplar, the MH marginal
   matches exact enumeration.
5. Level loop plus the h = 1 tile readout.  Test: the output satisfies the rule, and
   diversity across seeds.
6. Experiment: compare against generic.py (texture, promise, tiles) at matched cost.

## Open

- h_top = 32 is swept, not clamped (on target).  Risk: metastable top (as the all-sky
  PromiseSampler start); anneal, and check diversity across seeds.  Fallback: clamp u at
  h_top (T = ∞, as texture(h_top=32)).
- At h_top the 5-wide patch wraps a 4-wide periodic grid onto itself: count the self-overlap
  in ΔE_c.
- Candidate proposal q (reuse TileChain's kNN + coherent + ε MH).
- Weights: E_c vs E_par vs cost, and r per level.  λ_c is a free parameter; generic.ex_weight
  derives it from the target (λ_c = E_ex's w_h = 8 h² / 25 with E_ex's features; exact up
  to fill = exemplar windows, finer E_ex terms and E_loc ignored).  Revisit: fit by
  pseudo-likelihood on target samples (the exemplar alone sends λ_c → ∞).
- A hard cross-level v relation, if a closed small-V value space turns up.

## Status (2026-09-28)

- Levels h = 16 → 1 run end to end (notes/experiments/legacy/joint_multi.py): kNN MH, free sides,
  two padding rows above and below with their own patch terms and sampled columns.
- h = 1: v = the tile's solid bit (E|E / F|F), hard cost, so compat is the support rule;
  E_loc = E_pair / t_loc − Σ logz joins the conditional.  Output is support-valid.
- Colours at spacing 5 are batched (JointLevel(fast=True), sweep_fast); the single-cell
  methods are the reference, tested against it.  Speedup 1–4× (overhead per colour is
  25 × 24 small gathers); slower than the reference at h = 16.

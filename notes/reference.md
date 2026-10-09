# Hierwave reference

A compact description of the project as of 2026-10-08: goals, the
objects the DSL has, the two probability models, the sampler, and the
training (combining) algorithm, with enough detail to reimplement.  Longer
arguments live in `constitution.md`, `history/channels.tex`, `channels_system.tex`
and the per-experiment notes (`potts_test.md`, `circles_test.md`,
`induce_test.md`, `convpot_test.md`, `paintpot_test.md`).

## 1. Goals

- Generate 2D/3D voxel terrain and structures that look like given
  examples, plausible at every scale, with hard structural constraints
  (support, connectivity) always satisfied.
- Output is a *sample* from a distribution, at T = 1, not an optimum.
- Infinite, position-hashed worlds generated chunk by chunk: a fixed
  number of local steps per cell, no global check at any level.
- Coarse to fine: each level refines the one above.  Designed coarse
  interactions are allowed; interactions *induced* by lower levels are
  learned, never hand-tuned.
- Fixed per-cell state; general mechanisms, no rules for one tile set.

The project is a DSL for simple hierarchical MRFs (channel sets) plus an
algorithm for combining them: sets interact only by competing for shared
children, and the algorithm computes the couplings that competition
induces.

## 2. Objects (the DSL)

**Channel** `(name, h, D, views)`: a grid of `(H/h, W/h)` values in
`0..D-1` at block size `h` (`h = 1`: tiles).  A **view** is a `(D,)`
integer array mapping each value to a small feature; factors read views,
never raw values.  `fixed` mask: cells the kernel never updates.

**Factor**: a table over views, energy units, `inf` = hard.
- `pair(a, b, off, table, pad_b, pad_a)`: `E[va(z_a[p]), vb(z_b[q])]`,
  `q = (p * h_a) // h_b + off` in b's cells; same-level neighbours
  (`off = (0, 1)`) and parent reads (b coarser) alike.  `pad`: b's view
  off the grid, `-1` = the term vanishes.
- `count(a, b, table)`: `E[vb(z_b[q]), sum of va over the block of q]`,
  b coarser than a.
- `unary(a, table)`.
- `convpot(a, E, a_vec, A, W, b, v, pad)` (section 6; bilinear 8-offset
  potential over value embeddings, optional softplus head).
A factor is homed on its finer (`a`) side.  Top-down only: a coarser
channel never reads a finer one during generation.

**Certificate** (the one computed factor; connectivity/support): tile
channel with views `mass`, `trunk`, join bits; a `d` channel at the same
level, `D = Dmax + 2`, `INF = Dmax + 1`.  Cell p is valid iff
`mass_p = 0 and d_p = INF`, or p is a trunk with `d_p = 0`, or `d_p < INF`
and some joined 4-neighbour q has `mass_q >= mass_p` and `d_q < d_p`.
Soft cost `delta * d`.  `tree = True`: exactly one parent per root cell.

**Painted channel**: a deterministic function of a latent, at finer
resolution, stored as a channel with `fixed = True` (sugar's stone,
circles' `dem` and `slot`).  Carries position-dependent honour (which
tile of a block is in the disc) that position-free tables cannot; it is
both the refinement step and the input of the learned potential.

**Promotion**: replace a deterministic view `alpha(z_c)` of a large
channel by a channel `w` with domain `V` and a cost factor
`cost(w, alpha(z_c))`; move hard factors to read `w`.  The cost table is
the free energy of realising `w` under `z_c` (section 5), or the
exemplar's `-log p(w | u)` as its counting stand-in.

**Model** `(H, W, channels, factors, certs)`.  Composition of channel
sets = union of their channel and factor lists on shared channels.

**Channel sets so far** (`castlegen/channels/`): `ground` (designed
value-noise surface `surf` at h = 8, support seam, count honour, soft
contact tables), `roots` (exemplar-counted tables, contact rule,
certificate), `coord` (masked exemplar coordinate channel `u` with view
`alpha`, coherence kernel over candidate coordinates, coarse `u8`
windows), and the toys `sugar`, `potts`, `circles` (section 7).

## 3. The two models

Levels `h_top, ..., 1`; `z_h` the state of level h; `E_h(z_h | z_{>h})`
the sum of the factors homed on level h (they read h and coarser levels).

**Bidirectional model (the ideal, `p*`)**
```
p*(z) ∝ exp( - sum_h E_h(z_h | z_{>h}) )
```
the undirected joint of every declared factor.  Influence runs both ways
through shared variables.  It is the specification; it is never sampled
in production.

**Forward model (the chain, what ships)**
```
q(z) = q(z_top) q(z_{top/2} | z_top) ... q(z_1 | z_2)
```
each stage the generic Gibbs kernel run for a fixed number of sweeps at
T = 1 on level h with levels above clamped, under `E_h + Phi_h`, where
`Phi_h(z_h | z_{>h})` is the *learned potential* of level h.

**The identity.**  Writing `F_{h/2}(z_h) = -log Z_{h/2}(z_h) = -log sum_{z_{h/2}}
exp(-E_{h/2}(z_{h/2} | z_h) - ...)` (the free energy of everything
below h given `z_h`, recursively corrected), the chain reproduces `p*`
exactly when `Phi_h = F_{h/2}` for every level, and the kernels
equilibrate.  So the learned potential of a level is, by definition, the
free energy the levels below induce on it, and composition is correct
when those potentials are right.  `F` is approximated by a cluster
expansion into local terms (unary, pair over adjacencies; higher orders
when measured), justified when the fine correlation length is below the
block size (off-critical content).

**Support and finite part.**  Hard rules are not learned: the support of
`Phi_h` comes from honourability (a coarse value whose children cannot
complete is `inf`); the learned part is finite.

## 4. The sampler (`kernel.py`)

Per level, `Model.compile(home)` packs: every factor homed on `home`,
reflections of same-level pairs (transposed table, negated offset), the
grids and views of all channels, the certificate.  Colouring by factor
radius: 2 colours at radius 1 (`(x + y) mod 2`), 5 at radius 2
(`(x + 2y) mod 5`), sequential beyond.

Sweep: for each colour, each non-fixed site p: candidate energies
`e[t] = sum over packed rows` for every `t in 0..D-1`; draw `t` by
Gumbel-max over finite `e[t]/T` (exact conditional).  A site with no
finite candidate keeps its value and counts as a violation.  With a
certificate the site draws `(t, d)` jointly: for each t the valid d form
an interval (above the smallest witness `d`, below the smallest `d` of a
dependant that p must keep valid), whose geometric sum under
`exp(-delta d)` is closed form; t by Gumbel-max over summed weights, then
d within its interval.  Dependants read their other neighbours, hence
radius 2.

`compile(home, below=True)` adds rows for factors homed on finer channels
that read `home` (kind 3 below-pair: `sum over fine cells s in block(p)
of tab[va(z_s), vb(v)]`; kind 4 below-count), giving the full `p*`
conditional at a coarse site.  Used by diagnostics and the self-play
trainer, never by generation.

Schedule: levels top-down; designed channels written once; each sampled
channel initialised as a consistent refinement of the level above
(painted), then `S` sweeps.  No ramps, no annealing, no global checks.

## 5. The training (combining) algorithm

Input: a model, a coarse channel `c` at level h, the corrected levels
below it, its painter(s), a reference value `r` of `c`.  Output: `Phi_h`
as tables (or an on-the-fly potential) for the forward kernel.  Nothing
samples `c`; no coarse site ever moves; no value pair has to occur.

**5.1 Windows.**  Draw configurations `z_W` of `c` on a small window (3 x
3 cells; 2 x 2 for pair-only fits) plus a one-cell **halo** of `c` (values
drawn from the proposal, later from forward samples); exterior at `r`.
Fine region = the window's and halo's blocks plus the painters' spill.

**5.2 Free energy per window** `F(z_W | halo) = -log Z` of the fine
region with `c` clamped:
- `exact` hooks where available (transfer matrix; per-tile product when
  tiles are independent given the paint);
- `ais` (generic, `induce.ais_log_z`): `p_0` = the fine channel's unary
  factors only (`log Z_0 = sum_sites log sum_t exp(-E_unary(t))`, closed
  form), `p_1` = the full fine conditional, `E_beta = E_unary + beta *
  E_rest`, `inf -> L` (30) inside the anneal; `K` beta steps (linear
  from 1e-3 to 1), one kernel sweep per step, `log w += -(beta_{k+1} -
  beta_k) E_rest(state)`; `M` chains; `log Z = log Z_0 + logmeanexp(log
  w)`, spread of `log w` as the error bar.  Budgets that worked: `K =
  256, M = 4..8`; bias at small K is positive and the reported SE is
  optimistic below `K ~ 64`.  Run with the forward model's own fine
  kernel so `F` is the free energy *as that kernel realises it at that
  budget*.
Target per window: `y = F(z_W | halo) - F(r_W | halo)`.

**5.3 Paint potential** (`paintpot.py`, the parametrisation that
generalises).  Over the painted channel(s) at fine resolution with values
`0..V-1`:
```
F_theta(paint) = sum_t u(paint_t) + sum_{d in OFF8} sum_t g_d(paint_t, paint_{t+d})
```
`OFF8` = east, south, south-east, south-west (the 4 independent offsets of
the 8-neighbourhood); off-grid partners contribute nothing.  Linear in
`theta`.  Regressor `x = features(paint(z_W | halo)) - features(paint(r_W
| halo))` (counts of unary values and offset pairs); the halo's own
terms cancel, the window-halo cross terms are features.  Fit: ridge,
closed form.  Diagnostics: residual RMS / target RMS (= what is not
local in the painted map at this order), rank of the feature matrix.

**5.4 Materialise** coarse tables for the forward kernel:
`u_c(v) = F_theta(paint(v alone on r background)) - F_theta(paint(r))`,
`g_c[d](v, v') = F_theta(paint(v at p, v' at p + d)) - u_c(v) - u_c(v') -
F_theta(paint(r))` for the coarse adjacencies `d` (east, south, and the
diagonals), computed by painting pairs.  Exact for a linear potential.
For large D or many channels, evaluate the paint potential on candidates
at sampling time instead (paint the candidate's footprint, sum the
local energy): a convolution over painted images.

**5.5 Recurse upward.**  For the next coarser channel the "fine level" is
`c` carrying `E_c + Phi_c`; `p_0` = its unaries (designed + learned
`u_c`), anneal the rest.  Then iterate once with windows cut from forward
samples (halo distribution) and report how much the tables moved.

**5.6 What was tried and retired** (numbers in section 7):
- *Tables over latent identities* fitted by moment matching: `D^2`
  entries, each moved only when its pair occurs; unsampled entries
  unbounded.  Fails at D = 17.
- *Per-value embeddings with a bilinear/conv potential* (`convfit.py`,
  `CONVPOT`): fits seen pairs exactly, held-out pairs at correlation 0,
  because a projection of a value's patch in its own frame cannot
  express overlap of shifted maps.  The bilinear 8-offset form is still
  the right *shape* at coarse levels (reaches the AIS floor); the
  softplus plaquette head adds nothing and overfits.
- *Self-play moment matching* (`train.py`): forward sample `z`, a local
  `p*`-invariant kernel, Rao-Blackwellised expected count change
  `sum_p sum_t p*(t | z_-p) [N(z, z_p = t) - N(z)]`, step `theta += eta *
  gap`.  Works when coarse sites can move (collapsed moves in the toys),
  gives exactly zero gradient under hard honour (frozen sites), ordered
  per-level < recursion < end-to-end CD-K < oracle on the leak, and
  fails in the ordered phase.  Kept as a *drift diagnostic* of the
  forward output against `p*`, not as the trainer.

## 6. The convpot factor (kept for coarse-level pair potentials)

For a channel with embedding `E : (V, k)` and 3 x 3 offsets `d`
(row-major, index 4 = centre):
```
E(z) = sum_p [ a . e(z_p) + sum_{d != centre} e(z_p)^T A_d e(z_{p+d})
             + v . softplus( sum_d W_d e(z_{p+d}) + b ) ]
```
off-grid neighbours use `E[pad]` if `pad >= 0`, else drop out.  Kernel
cost per candidate `9 k` (bilinear) `+ 9 k m` (head, incremental).  With
one-hot `E` and `m = 0` it is the 8-offset pair-table model; fit by
closed-form ridge (`fit_bilinear_ls`), or Adam (`fit`) for `m > 0`.

## 7. Toys and measured results

| toy | levels | induced coupling | oracle |
|---|---|---|---|
| sugar | objects (8-blocks) / tiles | entropic: shared ring cells of a hard-square lattice gas | collapsed Gibbs, column transfer matrix |
| potts | palettes (8) / block colours (4) / tiles, cyclic-distance coupling J, honour kappa, lam | energetic, dense tables; `mid_h = BM J d` at kappa -> inf | collapsed mid (256-state transfer matrix), exact top |
| circles | corners (16) / disc objects with offsets (8, D = 17) / dirt-air tiles, soft honour kappa, field mu, presence bonus b | sparse geometric: +2.63 per conflict tile, -0.83 per shared ring tile | closed form per tile |

Key numbers (details in the notes):
- Self-play (Potts, kappa 8, J 0.1): mid tables exact for every scheme;
  top contrast as fraction of oracle: per-level 0.72, recursion 0.86,
  end-to-end 0.95.  J = 1 (ordered) fails.  Circles: plain step too slow
  on rare entries, preconditioned step overshoots.
- Induce, tabular, circles mid from AIS: slope 1.00, corr 0.965, unary
  27.00 vs `F_obj` 26.95.  Top residual 0.6-0.8 of target: later shown to
  be halo variance (0.54 floor), not non-additivity.
- Convpot: bilinear 8-offset reaches the AIS floor at the top without
  halo (0.20); head adds nothing; held-out value pairs corr ~0 for both
  embeddings.
- Paint potential: circles held-out pairs exact (corr 1.0000, max err
  2e-5, 18-68 parameters); top at the AIS floor with the halo painted
  (0.33 vs 0.32); Potts `mid_h` = -0.399/0/+0.399 vs +-0.40 at kappa 8,
  and -3.999/0/+3.999 vs +-4.0 at J = 1 where self-play broke; painted
  pair terms needed there (unary-only leaves 0.42).
- End to end (forward with learned tables vs oracle moments): within
  evaluation noise for tables, convpot and paint; a finite-grid boundary
  artefact lowers presence (0.31 vs 0.35).

Runtime: ~130 ns per tile update at D = 2 (numba, single core); coarse
levels and learned potentials negligible; a 256^2 chunk at ~64 kinds
estimated 1.5-3 s on the reference kernel; bit-sliced C / GPU planned.

## 8. Open items, in order

1. The estimator on a real set (ground + roots): hard factors, the
   certificate's cost, AIS with `inf -> L`.
2. Chunk-by-chunk generation with halos and position-hashed randomness;
   refinement from stored coarse state (also gives distance LOD: far
   regions exist as coarse paint only).
3. Several sets on one tile level: painted pair features across channel
   pairs; a small conv over stacked painted channels when triples matter.
4. Estimator variance (bridge sampling between neighbouring windows,
   chain reuse); periodic test worlds; a finer top-level floor.
5. Promotion costs from the same estimator.
6. Fast kernel (bit-sliced, GPU), 3D (6 neighbours, 3D support
   certificate), Luanti.
7. The painter as a declared element of a channel set.

## 9. Code map

| file | role |
|---|---|
| `castlegen/channels/core.py` | Channel, Factor (pair/count/unary/convpot), Certificate, Model, `compile(home, below)` |
| `kernel.py` | numba Gibbs sweep, certificate joint draw, `total_energy`, convpot evaluation |
| `train.py` | self-play diagnostics: `counts`, `site_probs`, `rb_gap`, `step` |
| `induce.py` | windows, `ais_log_z`, tabular `fit_tables`, `materialise` |
| `paintpot.py` | paint potential: `features`, `fit`, `energy`, `materialise` |
| `convfit.py`, `convref.py` | JAX fit of the convpot, numpy reference energy |
| `induce_circles.py` | circles hooks: windows, exact F, painters, embeddings |
| `ground.py`, `roots.py`, `coord.py` | real channel sets |
| `sugar.py`, `potts.py`, `circles.py` | toys with exact oracles (`Forward`, `Oracle`, `reference()`) |
| `sampler.py` | the Channel-sampler interface: `ChannelSampler`, `Sampler`, `generate`, the admitted list |
| `targets.py` | bootstrap targets: p*'s collapsed conditional at one site (`ExactTargets`, `SampledTargets`) |
| `aistargets.py` | `AISTargets`: the same conditional by AIS on the one-site window |
| `biome_fit.py` | biome circles bootstrap callbacks and stamp-feature materialisation |
| `embed_fit.py` | learned value embeddings (bilinear / MLP) fitted by hand gradients, materialised to tables |
| `circles_biome.py` | biome circles toy: object families masked by a biome, exact `Forward` / `Oracle`, periodic |
| `notes/experiments/selfplay_train.py` | self-play driver (potts / circles hooks) |
| `notes/experiments/induce_circles.py`, `convpot_circles.py`, `paintpot_circles.py`, `paintpot_potts.py` | the estimator experiments |
| `tests/channels/test_*.py` | regression suite (oracles vs brute force, kernel vs reference, fits recover planted tables) |

## 10. External work

Real-space renormalisation (the frame: block variables, renormalised
couplings = induced free energy):
- Niemeijer & van Leeuwen, "Renormalization theory for Ising-like spin
  systems", in Domb & Green, *Phase Transitions and Critical Phenomena*
  vol. 6 (1976): cluster/cumulant expansions, truncation to short-range
  couplings.
- Swendsen, "Monte Carlo renormalization group", PRL 42 (1979); Swendsen
  et al., PRB (1984): renormalised couplings by correlation matching
  (= self-play moment matching).  Wu & Car, arXiv:1707.08683,
  variational MCRG.
- Ron, Swendsen & Brandt, "Inverse Monte Carlo renormalization group
  transformations for critical phenomena", PRL 89 (2002): fine from
  coarse, on demand (= the forward chain).  Bachtis, Aarts, Di Renzo &
  Lucini, "Inverse renormalization group in quantum field theory",
  arXiv:2107.00466; review PoS EuroPLEx2023.
- Li & Wang, "Neural network renormalization group", PRL 121 (2018),
  arXiv:1802.02840.  Koch-Janusz & Ringel, "Mutual information, neural
  networks and the renormalization group", Nat. Phys. 14 (2018).
- Gidas, "A renormalization group approach to image processing
  problems", PAMI 11 (1989); Pérez & Heitz, "Restriction of a Markov
  random field on a graph and multiresolution statistical image
  modeling", IEEE Trans. Inf. Theory 42 (1996).

Coarse-graining and free-energy potentials:
- Shell, "The relative entropy is fundamental to multiscale and inverse
  thermodynamic problems", J. Chem. Phys. 129 (2008).
- Lyubartsev & Laaksonen, "Calculation of effective interaction
  potentials from radial distribution functions: a reverse Monte Carlo
  approach", PRE 52 (1995).  Reith, Pütz & Müller-Plathe, iterative
  Boltzmann inversion, J. Comput. Chem. 24 (2003).
- Noid et al., "The multiscale coarse-graining method", J. Chem. Phys.
  128 (2008).  Wang et al., "Machine learning of coarse-grained molecular
  dynamics force fields" (CGnets), ACS Cent. Sci. 5 (2019); Husic et al.,
  "Coarse graining molecular dynamics with graph neural networks", J.
  Chem. Phys. 153 (2020).
- Neal, "Annealed importance sampling", Stat. Comput. 11 (2001); Neal,
  "Sampling from multimodal distributions using tempered transitions",
  Stat. Comput. 6 (1996).

Energy-based learning, biasing, the procedure as the model:
- Tieleman, "Training restricted Boltzmann machines using approximations
  to the likelihood gradient" (PCD), ICML 2008.
- Nijkamp, Hill, Zhu & Wu, "Learning non-convergent non-persistent
  short-run MCMC toward energy-based model", NeurIPS 2019.  Xie et al.,
  "Cooperative training of descriptor and generator networks", PAMI 2018.
- Pitera & Chodera, "On the use of experimental observations to bias
  simulated ensembles", JCTC 8 (2012).  Ganchev et al., "Posterior
  regularization for structured latent variable models", JMLR 11 (2010).
- Zhu, Wu & Mumford, FRAME, Neural Comput. 9 (1997) / IJCV 27 (1998).
  Della Pietra, Della Pietra & Lafferty, "Inducing features of random
  fields", PAMI 19 (1997).

Texture synthesis and procedural generation:
- Paget & Longstaff, "Texture synthesis via a noncausal nonparametric
  multiscale Markov random field", IEEE TIP 7(6) (1998).  De Bonet,
  "Multiresolution sampling procedure for analysis and synthesis of
  texture images", SIGGRAPH 1997.
- Beukman, Ingram, Liu & Rosman, "Hierarchical WaveFunction Collapse",
  AIIDE 2023.  Alaka & Bidarra, "Hierarchical semantic wave function
  collapse", FDG 2023.  Merrell, "Example-based model synthesis", I3D
  2007.
- Willsky, "Multiresolution Markov models for signal and image
  processing", Proc. IEEE 90 (2002).

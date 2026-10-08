# DSL interface

The programmatic shape of `dsl_updates.md` (committed part): what a
channel, its sampler, its feature set and the trainer expose to each
other.  Pseudocode, Java-ish.  Comments give the symbol from
`reference_math.tex` each method implements.  Storage, incremental
updates, quantisation and site order live behind these interfaces and
are not part of the math.  The factor language (pair, count, unary,
painted channels, certificate) is the current one.  The build order is
in `circles_biome_test.md`.

Two decisions shape it.  **A channel samples itself**: the sampler is a
method of the channel, so the tile channel runs the bit-sliced sampler
while a coarse channel runs plain Gibbs.  **The learned potential is an
additive bias over the candidates the sampler exposes**: whatever set a
site is drawing over, the potential is handed that set and returns one
energy per candidate; how it computes them (materialised tables, a
context encoder, a batched exact delta) is its own business.

## Channel

```java
interface Channel {
  String name();
  int h();                         // block size h_c
  int D();                         // |Dom_c|
  View view(String name);          // alpha : Dom_c -> {0..V-1}
  IntGrid z();                     // z_c on Lambda_c
  BoolGrid fixed();                // phi_c
  Features features();             // e : Dom_c -> R^k, null for a channel with no learned potential
  List<Bias> biases();             // Phi_c as additive energies over a candidate set; empty on tiles

  void init(Rng rng);              // painted initialisation: a consistent refinement of the level above;
                                   //   then dormancy (C1): a site whose parents admit one value is fixed at it,
                                   //   and a block with every site fixed is skipped as a unit by sweep
  void relax(Rng rng, int p);      // post-relaxation: p sweeps with the parent constraint loosened (C1)
  void sweep(Rng rng);             // one sweep of the channel's own kernel at T = 1 under E_l + Phi_l,
                                   //   reading levels >= l only.  Any kernel invariant for exp(-(E_l + Phi_l)).
                                   //   At site p with candidate set C_p (Dom_c or a subset) the kernel does
                                   //     e = designed energies over C_p;  for (Bias b : biases()) b.add(this, p, C_p, e);
                                   //     draw over C_p from e;  notify locals and biases of the change.

  // training and diagnostics only
  double energy();                 // E_l(z_l | z_{>l}) + Phi_l as the sampler realises it
  void sweepTempered(Rng rng, double beta);   // p_beta ~ exp(-E_un - beta E_rest): AIS's kernel
  double unaryLogZ();              // log Z_0 = sum_s log sum_t exp(-E_un,s(t))
}

class TableChannel   implements Channel { List<Local> locals; }
                                   // reference_math section 3: enumerate Dom_c, colouring by the
                                   //   factors' reach, Gumbel-max, certificate joint draw.  Small D.
class BitTileChannel implements Channel { }
                                   // castlegen/bitgibbs.c: rows as quantised level planes, bit-sliced
                                   //   adder, integer draw.  Designed factors only (F_0 = 0).
class CoordChannel   implements Channel { Exemplar ex; int K, K_t; }
                                   // castlegen/channels/coord.py: candidate set (current value,
                                   //   neighbours' coherent continuations, K random, K_t
                                   //   tile-consistent) chosen without reading z_p, proposals drawn
                                   //   from the values the parents admit; Gumbel-max on it.
```

Same-level factors are constructor arguments of the implementation (a
`Local` list for the numba kernel, rows for the bit sampler), not part of
the interface.

## Local factors (what a TableChannel's kernel sums)

```java
interface Local {                  // one term of E_l + Phi_l at the home channel
  int reach();                     // in home cells; sets the colouring
  void score(Channel home, int y, int x, double[] e);   // e[t] += this term with z_p = t
  void onChange(Channel home, int y, int x, int from, int to);   // incremental state
}

class PairRow     implements Local { Channel b; View a, bv; int[] off; double[][] T; }  // reads level >= l
class CountRow    implements Local { }
class UnaryRow    implements Local { }
class Certificate implements Local { double[] weights(...); int drawD(int t, Rng rng); }
class FormulaRow  implements Local { Footprints f; }   // designed latent-level rule computed from the two
                                                       //   values' footprints (distance, overlap); C4
```

## Bias (C2): the learned potential as the sampler sees it

```java
interface Bias {                   // Phi_c = sum_p psi_theta(z_p, z_{N(p)})
  int reach();                     // of N(p) plus the features' spill; joins the colouring
  void add(Channel c, int y, int x, int[] cand, double[] e);
                                   // e[i] += Phi_c(z with z_p = cand[i]) - Phi_c(z)  for every i.
                                   //   Up to a per-site constant.  Form (i): a difference of the one joint
                                   //   Phi_c, including the psi terms of every neighbour q with p in N(q).
                                   //   Form (ii): a directly learned conditional at p, with the pair
                                   //   consistency check reported (C2).
  void onChange(Channel c, int y, int x, int from, int to);   // cached context, if any
}

class MaterialisedBias implements Bias { double[] u; double[][][] g_d; }
                                   // psi pairwise-decomposable and D_c small: u_c(v), g_{c,d}(v, v')
                                   //   precomputed once; add() is lookups.  No model calls at sampling time.
class OnePassBias      implements Bias { Features e; Encoder h; }
                                   // h_theta(ctx_p) once per visit, e[i] += e(cand[i]) . h.  Meets the
                                   //   contract only when h is linear in the neighbours' features
                                   //   (bilinear); the implementation asserts that.
class BatchedDeltaBias implements Bias { Features e; LocalTerm psi; }
                                   // for each candidate, re-evaluate the |N(p)| + 1 psi terms touching p,
                                   //   as one batch over cand.  Exact for any psi.
```

## Features (C2)

```java
interface Features {               // e : Dom_c -> R^k
  int k();
  double[] of(int v);              // e(v), fixed for the channel once built
  Structure structure();           // optional: ties on psi's parameters the features imply (e.g. a
                                   //   stencil geometry for stamps); null when psi is free
}

class LearnedFeatures implements Features { double[][] E; }
                                   // rows of a free (D, k) matrix, fitted with psi (Adam); token embeddings
class StampFeatures   implements Features { Painter pi; int spill; }
                                   // derived: paint v alone in its block's frame; e(v) = indicator over
                                   //   (footprint cell, paint value), k = cells x V.  structure() may tie
                                   //   a pairwise psi to the stencil (then psi = the paint potential,
                                   //   parameters (u, g)).  Built once; the fine grid is not read at
                                   //   sampling time.  One option among feature sets, not the design.
class MixedFeatures   implements Features { Features derived; LearnedFeatures extra; }
```

A feature set is built when the channel is, from the channel's own
declaration (its painter, its exemplar, or nothing but `D` and `k`).
`Trainer` fits `psi`'s parameters and any learned feature dimensions; it
does not choose the feature set.

## Schedule

```java
class Level { int h; List<Channel> channels; int S; }

class Model {
  List<Level> levels;              // coarsest first
  void generate(Rng rng) {
    for (Level l : levels)
      for (Channel c : l.channels) { c.init(rng); for (int s = 0; s < l.S; s++) c.sweep(rng); }
  }                                // painters refresh inside sweep/onChange, as today
}
```

## Training (reference_math section 4)

```java
interface FreeEnergy { double estimate(Model m, Channel c, Window w); }   // F(z_W | halo)
class ExactHook implements FreeEnergy { }
class AIS       implements FreeEnergy { int K, M; double L; }   // fine channel's unaryLogZ, sweepTempered, energy

class Window {                     // z_W, z_halo, ref outside; the fine region R(W)
  Window(Channel c, Shape shape, int ref, Proposal p, Rng rng);
}

interface JointSampler { Sample next(); }   // p* offline: block fine samples (latent = view), or collapsed moves

class Trainer {
  Params fitWindows(Model m, Channel c, int ref, FreeEnergy F, Proposal p, int nWindows, double lambda);
                                   // ridge when Phi is linear in theta, Adam otherwise, on x(W) . theta = y(W)
  Params fitConditional(Model m, Channel c, JointSampler js, int nSamples);
                                   // pseudo-likelihood: cross-entropy of the sampled value over the sampler's
                                   //   candidate set, logits = -(E_c + Delta); designed energy a fixed offset.
                                   //   Reports the pair consistency violation and the fine kernel's
                                   //   autocorrelation with the coarse level clamped.
  Bias install(Params theta, Channel c);
                                   // MaterialisedBias when psi is pairwise and D_c small; OnePassBias when
                                   //   h is linear; BatchedDeltaBias otherwise.  Added to c.biases().
                                   //   Recursion upward: the next coarser channel's AIS runs c's
                                   //   sweepTempered with this bias in.
}
```

## The training algorithm

```
# Levels l = 1 (tiles) .. L.  Designed E_l(z_l | z_{>l}) given at every level.  Phi_1 = 0.
# For l >= 2:  Phi_l = F_{l-1} = -log sum_{z_{l-1}} exp(-(E_{l-1} + Phi_{l-1})),
# approximated as  support_l (computed)  +  sum_p psi_theta(z_p, z_{N(p)})  (fitted).
# The sampler only needs Delta_p(t) = Phi_l(z with z_p = t) - Phi_l(z) over its candidate set C_p.

train(model, jointSampler or null, nSamples, nWindows):
    samples = jointSampler ? jointSampler.draw(nSamples) : null
        # one offline run of p*, every level, full states; reused by every level below.
        # Collapsed moves for a free latent (Q4); block the fine samples for a latent that is a view.
    for l = 2 .. L:                                   # bottom up: estimator B needs Phi_{l-1} installed
        for c in channels(l) with a learned potential:
            S = support(c)                            # hard rows derived from the stamps / honourability
            theta = samples ? fitConditional(model, c, samples)
                            : fitWindows(model, c, nWindows)
            c.locals += S
            c.biases += install(theta, c)
            diagnostics(c)                            # consistency violation (A) or residual (B),
                                                      # autocorrClamped(children of c | c), held-out monitors

# estimator A: pseudo-likelihood on joint samples
fitConditional(model, c, samples):
    theta = 0
    repeat until converged (Adam; L-BFGS when psi is linear in theta, the loss is convex):
        loss = 0
        for z in samples:
            model.load(z restricted to levels >= l); repaint      # levels < l discarded: integrated out
            for p in sites of c that are active (not dormant, not fixed):
                C = c.candidates(p)                   # admissible values at p, or the K-subset containing z_p
                e = [E_designed_c(p, t) for t in C]   # designed rows; hard rows give INF
                d = [Delta_p(t; theta)  for t in C]   # BatchedDelta: psi terms touching p with z_p = t, minus
                                                      #   at z_p (paint potential: theta . (x_p(t) - x_p(z_p)))
                loss += -log softmax(-(e + d))[z_p]
        theta -= lr * grad(loss + l2 |theta|^2)
    return theta
    # reports held-out cross-entropy and the pair consistency violation: max over neighbouring
    # active sites p, q and values t, t' of the closure error of the one-site odds around the square

# estimator B: window free energies
fitWindows(model, c, nWindows):
    for i in 1 .. nWindows:
        W   = proposal.window(c)                      # values on the window cells + halo, r outside
        y_i = F_{l-1}(W) - F_{l-1}(W_ref)             # AIS on level l-1's own sampler (Phi_{l-1} installed)
                                                      #   over the fine region R(W); exact hook when available
        x_i = features(W) - features(W_ref)           # sum over window sites of the stamp-pair features
    theta = ridge(X, y)                               # Adam when psi is not linear in theta
    return theta
    # reports the residual on held-out windows and the AIS standard error (the floor)

F(W) by AIS:  log Z = unaryLogZ() + log mean_m exp(-sum_k (beta_{k+1} - beta_k) E_rest(z^{(m)}_k)),
              z_k advanced by sweepTempered(beta_k);  F = -log Z.

install(theta, c):
    if psi pairwise and D_c small:  MaterialisedBias (u[v], g_d[v, v'] tabulated once -> rows)
    elif h_theta linear:            OnePassBias
    else:                           BatchedDeltaBias(features, psi_theta)

generate(model):
    for l = L .. 1:
        for c in channels(l):
            c.init(rng)                               # paint parents' writes; dormancy
            for s in 1 .. S_l:  c.sweep(rng)          # at p: e over C_p; b.add(C_p, e) for b in biases; draw
            c.relax(rng, p_l)
```

- Estimator A needs no recursion: the marginal of `p*` over levels
  `>= l` has one-site conditionals `exp(-(E_l + F_{l-1}))` exactly, so
  one sample set serves every level, in any order.  The bottom-up order
  is for estimator B, whose `F_{l-1}` is computed by sampling level
  `l-1` under its installed approximation.
- Both deliver per-site candidate energies up to a constant; neither
  computes a partition function of level `l`.
- The support is never in the fit: hard rows from the stamps go into
  the locals first; the sampled value is always admissible and forbidden
  candidates are at `INF` designed energy, so they drop out of the softmax.
- The candidate set in training matches the sampler's: with a cap, the
  sampled value is included and the cross-entropy is over the subset,
  which is consistent since the conditional restricted to a subset is
  proportional to the same weights.

## What the interface hides

- The sampler: colouring and Gumbel-max, bit-sliced rows and the integer
  draw, the candidate set of a coordinate channel; certificate
  bookkeeping; painter refresh.
- How a `Bias` computes its energies: tables, an encoder, a batch.
- AIS internals.

## What the math fixes

- `sweep` reads levels `>= l` only and leaves `exp(-(E_l + Phi_l))`
  invariant; `energy`, `sweepTempered`, `unaryLogZ` make it AIS's kernel,
  so training measures the kernel that ships.
- A `Bias` returns candidate energies up to a per-site constant over the
  set it is handed, so the draw is the learned conditional of `E_l + Phi_l`
  whatever the sampler's candidate set and whatever the bias's internals;
  the joint-difference form makes that conditional exact, the direct form
  exact up to the reported consistency violation.
- `Phi_c` is evaluated at `c`'s level and never reads the fine grid at
  sampling time.
- A feature set is fixed before fitting; what is fitted is `psi`'s
  parameters and learned feature dimensions if any.

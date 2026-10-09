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
                                   //   then the admitted list (C1): the parent hard rows evaluated once into
                                   //   one list per distinct parent context, which candidates() and sweep()
                                   //   draw from; dormancy = a list of length one, fixed, and a block with
                                   //   every site fixed is skipped as a unit by sweep.  Valid until the
                                   //   parents change.
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

interface Targets {                // p*'s collapsed conditional at one site, over the sampler's candidate set
  double[] at(Model m, Channel c, int y, int x, int[] cand);   // pi_p over cand, sums to 1
}
class AISTargets   implements Targets { FreeEnergy F; }   // softmax(-(E_c + F_{l-1}(t; ctx))), F by AIS on a
                                                          //   one-site Window with halo, one run per candidate;
                                                          //   the general case
class ExactTargets implements Targets { }                 // closed-form hook (toys)
class SampledTargets implements Targets { JointSampler js; }   // one-hot of the value in a free joint sample
                                                          //   (latent = view: blocked fine data; annotated corpus)

class Trainer {
  Params fit(Model m, Channel c, Targets T, int iters, int nContexts, int K, double lambda);
                                   // the bootstrap of dsl_updates C2: contexts from the forward chain,
                                   //   targets pi_p from T at every active site, K sweeps drawing from pi_p
                                   //   to move contexts toward p* (K = 0: S1; K > 0: CD-K / S3), cross-entropy
                                   //   of pi_p against softmax(-(E_c + Delta_theta)) over cand; the dataset of
                                   //   (context, pi_p) aggregates over iters and is refit each time.
                                   //   Reports the pair consistency violation of the targets and the fit, the
                                   //   held-out cross-entropy, and autocorrClamped(children of c | c).
  Params fitWindows(Model m, Channel c, int ref, FreeEnergy F, Proposal p, int nWindows, double lambda);
                                   // the same AIS runs on multi-site windows with a regression loss on
                                   //   x(W) . theta = y(W); diagnostic for pair structure
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

train(model, iters, nContexts, K):
    for l = 2 .. L:                                   # bottom up: the targets at l need Phi_{l-1} installed
        for c in channels(l) with a learned potential:
            S = support(c); c.locals += S             # hard rows derived from the stamps / honourability
            T = c.targets()                           # AISTargets in general; ExactTargets / SampledTargets
                                                      #   where free.  Never a global sampler of p*.
            theta = fit(model, c, T, iters, nContexts, K)
            c.biases += install(theta, c)
            diagnostics(c)                            # consistency violation, held-out cross-entropy,
                                                      # autocorrClamped(children of c | c), monitors

# the bootstrap: sample from q, move toward p*, update q
fit(model, c, T, iters, nContexts, K):
    data = []                                         # (context features, candidate set, pi_p) per site; aggregates
    theta = 0
    for it in 1 .. iters:
        for n in 1 .. nContexts:
            z = model.generate(theta)                 # 1. a context from q: the forward chain, levels >= l
            for k in 0 .. K:                          # K = 0: S1.  K > 0: CD-K (S3).  K -> inf: S0.
                for p in active sites of c (not dormant, not fixed), in the kernel's colour order:
                    C  = c.candidates(p)              # admissible values, or the capped set containing z_p
                    pi = T.at(model, c, p, C)         # 2. p*'s collapsed conditional over C:
                                                      #    softmax(-(E_c(t) + F_{l-1}(t; ctx))), F by AIS on
                                                      #    a one-site window with halo, one run per t in C
                    data += (ctx(p), C, pi)
                    if k < K: z_p ~ pi; repaint       # 3. one p*-invariant step on q's sample, free
        theta = argmin over theta of                  # 4. update q: refit from scratch on all data
            sum over data of  KL( pi || softmax(-(e + d)) )  +  l2 |theta|^2
                where e[t] = E_designed_c(p, t), d[t] = Delta_p(t; theta)      # BatchedDelta; the designed
                                                      #   energy is a fixed offset, only F is learned
            # convex when psi is linear in theta (paint potential): L-BFGS; Adam otherwise
    return theta
    # reports: held-out KL; the pair consistency violation of pi (targets) and of the fit (max over
    # neighbouring active sites p, q and values t, t' of the closure error of the one-site odds)

F(t; ctx) by AIS over the fine region R of the one-site window at p with z_p = t, halo from ctx, r outside:
    log Z = unaryLogZ() + log mean_m exp(-sum_k (beta_{k+1} - beta_k) E_rest(z^{(m)}_k)),
    z_k advanced by sweepTempered(beta_k) of level l-1's sampler with Phi_{l-1} in;  F = -log Z.
    Only differences across t at one site matter, so the halo's own free energy cancels.

fitWindows(model, c, nWindows):                       # diagnostic: multi-site windows, regression loss
    for i in 1 .. nWindows:
        W   = proposal.window(c)
        y_i = F_{l-1}(W) - F_{l-1}(W_ref);  x_i = features(W) - features(W_ref)
    theta = ridge(X, y)

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

- The targets are local: one level, one site's window, the shipping
  kernel at the shipping budget.  Nothing samples `p*` globally.  The
  bottom-up order is because `F_{l-1}` is computed under level `l-1`'s
  installed approximation.
- A target integrates out everything the sampler draws after the site,
  including the site's own children.  `AISTargets` does by construction;
  an `ExactTargets` hook must be collapsed too.  A one-site conditional
  of `p*` that reads a finer level is the frozen-site target and
  reproduces the leak (C2, stage 6b).  Parent rows sit in the AIS's
  `p_0` and are never annealed.
- A target is a fixed number once the level below is installed, so the
  dataset aggregates across iterations; there is no negative phase and
  nothing goes stale.  `K` moves the contexts from `q` toward `p*` at
  `K` times the AIS cost and no new mechanism: the draw is from the
  `pi_p` already computed.
- The loss is on the object the sampler uses: the conditional over its
  own candidate set at the contexts it visits.  With a sampled target it
  is pseudo-likelihood; with a computed one it is pseudo-likelihood with
  the response Rao-Blackwellised.  Neither computes a partition function
  of level `l`.
- The only failure mode is the one generation has: a fine kernel that
  does not mix with the coarse values clamped, which the C1 criterion
  flags.
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

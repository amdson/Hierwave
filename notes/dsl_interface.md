# DSL interface

The programmatic shape of `dsl_updates.md` (committed part): what a
channel, its sampler, its feature set and the trainer expose to each
other.  Pseudocode, Java-ish.  Comments give the symbol from
`reference_math.tex` each method implements.  Storage, incremental
updates, quantisation and site order live behind these interfaces and
are not part of the math.  The factor language (pair, count, unary,
painted channels, certificate) is the current one.

Two decisions shape it.  **A channel samples itself**: the sampler is a
method of the channel, so the tile channel runs the bit-sliced sampler
while a coarse channel runs plain Gibbs.  **A channel has a feature
set**: a vector per value, from which its learned potential is computed;
how the vector is obtained (learned, derived from the finer level, mixed)
is the channel's choice.

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

  void init(Rng rng);              // painted initialisation: a consistent refinement of the level above
  void sweep(Rng rng);             // one sweep of the channel's own kernel at T = 1 under E_l + Phi_l,
                                   //   reading levels >= l only.  Any kernel invariant for exp(-(E_l + Phi_l)).

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
                                   //   tile-consistent) chosen without reading z_p; Gumbel-max on it.
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
class ConvRow     implements Local { Features e; double[] a; double[][][] A_d; Head head; }
                                   // Phi_c (C2): e[t] += a.e(t) + sum_d e(t)^T A_d e(z_{p+d}) + head;
                                   //   on demand from features, k per neighbour.  Large D.
class MaterialisedRows implements Local { double[] u; double[][][] g_d; }
                                   // the same Phi_c precomputed over Dom_c x Dom_c (small D):
                                   //   u_c(v), g_{c,d}(v, v'); plain unary + pair lookups
```

## Features (C2)

```java
interface Features {               // e : Dom_c -> R^k
  int k();
  double[] of(int v);              // e(v), fixed for the channel once built
  PairStructure structure();       // how A_d is parametrised: FREE, or tied to a geometry
}

class LearnedFeatures implements Features { double[][] E; }
                                   // rows of a free (D, k) matrix, fitted with A_d (Adam); token embeddings
class StampFeatures   implements Features { Painter pi; int spill; }
                                   // derived: paint v alone in its block's frame; e(v) = indicator over
                                   //   (footprint cell, paint value), k = cells x V.  structure() ties
                                   //   A_d to the stencil: e(v)^T A_d e(v') = sum over cell pairs within
                                   //   offsets O of g_delta(paint_v(s), paint_v'(s')); parameters (u, g).
                                   //   Built once; the fine grid is not read at sampling time.
class MixedFeatures   implements Features { Features derived; LearnedFeatures extra; }
```

A feature set is built when the channel is, from the channel's own
declaration (its painter, its exemplar, or nothing but `D` and `k`).
`Trainer` fits the parameters of `A_d`, `a` and any learned dimensions;
it does not choose the feature set.

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

class Trainer {
  ConvParams fit(Model m, Channel c, int ref, FreeEnergy F, Proposal p, int nWindows, double lambda);
                                   // ridge (features fixed, A_d tied: linear in theta) or Adam (free A_d /
                                   //   learned features) on x(W) . theta = y(W)
  Local install(ConvParams theta, Channel c);
                                   // MaterialisedRows when D_c is small, ConvRow otherwise; added to
                                   //   c's locals beside its designed rows.  Recursion upward: the next
                                   //   coarser channel's AIS runs c's sweepTempered with this row in.
}
```

## What the interface hides

- The sampler: colouring and Gumbel-max, bit-sliced rows and the integer
  draw, the candidate set of a coordinate channel; certificate
  bookkeeping; painter refresh.
- Whether `Phi_c` is materialised or computed on demand.
- AIS internals.

## What the math fixes

- `sweep` reads levels `>= l` only and leaves `exp(-(E_l + Phi_l))`
  invariant; `energy`, `sweepTempered`, `unaryLogZ` make it AIS's kernel,
  so training measures the kernel that ships.
- `Phi_c` is a function of `c`'s values through `features()` and the
  fitted `A_d`; it is evaluated at `c`'s level and never reads the fine
  grid at sampling time.
- A feature set is fixed before fitting; what is fitted is `a`, `A_d`,
  the head, and learned feature dimensions if any.

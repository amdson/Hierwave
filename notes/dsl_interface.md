# DSL interface

The programmatic shape of `dsl_updates.md`: what a channel, a write, a
local factor, the kernel and the trainer expose to each other.  Pseudocode,
Java-ish.  Comments give the symbol from `reference_math.tex` /
`dsl_updates.md` each method implements.  Storage, incremental updates and
scheduling live behind these interfaces and are not part of the math.

## Core objects

```java
interface Channel {
  String name();
  int h();                         // block size h_c; level = h
  int D();                         // |Dom_c|
  View view(String name);          // alpha : Dom_c -> {0..V-1}, an int[D]
  IntGrid z();                     // z_c on Lambda_c = (H/h) x (W/h)
  BoolGrid fixed();                // phi_c: cells the kernel never updates
  Field field();                   // B_c: the energies coarser channels wrote (U1)
}

interface Field {                  // B : Lambda_c -> R^D, +inf allowed
  void clear();
  void add(int y, int x, double[] e);         // B_p += e           (cell write)
  double[] at(int y, int x);                  // B_p
  BlockRow block(Channel src, View a, int qy, int qx);   // C_q for a count write
}

interface BlockRow {               // C_q : {0..|block|} -> R, with a maintained counter
  void set(double[] row);          // C_q(n) = T[beta(z_b(q)), n]
  int countExcluding(int y, int x);   // n_{-p}: sum of alpha over the block minus p
  double at(int n);                // C_q(n)
}
```

## Communication (coarse to fine)

```java
interface Writer {                 // one declared parent-to-child term (U1, U2)
  Channel src();                   // b, coarser or equal
  Channel dst();                   // a, h_a <= h_b
  int radius();                    // rho_ab, in src cells (U2 rule 2)

  void write(Field dst);           // apply all of src's writes into dst.field()
  Delta delta(int qy, int qx, int v);   // what src cell q would write if z_b(q) = v:
                                   //   a sparse field change over block(q) +- radius.
                                   //   Used (i) to refresh after q changes,
                                   //   (ii) by Potential to score a coarse candidate.
}

// Implementations
class TableWrite implements Writer { View a, b; int[] offset; double[][] T; }
                                   // B_p += T[a(.), b(z_b(q))], q = q_b(p, offset)
class CountWrite implements Writer { View a, b; double[][] T; }
                                   // block(q).row = T[b(z_b(q)), .]
class Painter implements Writer  { IntGrid paint; double[][] Theta; }
                                   // pi_p from z_src, then B_p += Theta[pi_p]  (compression)
class PromotionCost implements Writer { View alpha; double[][] cost; }
                                   // w -> u: B_u(p) += cost(w_p, alpha(.)); and the reverse
```

## Local factors (same level, read by the kernel)

```java
interface Local {                  // a same-level term of E_l
  int radius();                    // sets the colouring
  void score(Channel home, int y, int x, double[] e);   // e[t] += this term with z_p = t
  void onChange(Channel home, int y, int x, int from, int to);  // keep incremental state
}

class PairTable implements Local { int[] offset; double[][] T; boolean reflected; }
class CountTerm  implements Local { }     // e[t] += C_q(n_{-p} + a(t)); counter kept in BlockRow
class ConvPot    implements Local { }     // section 5 of reference_math
class Certificate implements Local {      // joint (t, d) draw; the kernel special-cases it
  double[] weights(Channel home, int y, int x, double T);   // w_t, with [a_t, b_t)
  int drawD(int t, Rng rng);
}
class OnTheFly   implements Local { Potential phi; Writer[] out; }
                                   // e[t] += phi.energy(delta footprint of t at p): Phi evaluated
                                   // at sampling time when D is too large for tables
```

## Kernel and schedule

```java
class Kernel {
  Kernel(Channel home, List<Local> locals, double T);
  void sweep(Rng rng);             // colours by max radius; per non-fixed site:
                                   //   e = home.field().at(p); for L in locals: L.score(...)
                                   //   t = gumbelMax(-e / T);  home.z()[p] = t;  onChange(...)
}

class Level {
  int h; List<Channel> channels;
  List<Writer> writesIn;           // from levels >= h (direction, U2 rule 1)
  List<Local>  locals;             // E_l's same-level part, plus Phi_l's materialised tables
  int S;                           // sweeps, the budgeted kernel
}

class Model {
  List<Level> levels;              // coarsest first
  void generate(Rng rng) {
    for (Level l : levels) {
      for (Channel c : l.channels) c.field().clear();
      for (Writer w : l.writesIn) w.write(w.dst().field());
      for (Channel c : l.channels) init(c);          // painted initialisation: argmin_t B_p(t)
      for (int s = 0; s < l.S; s++) new Kernel(c, l.locals, 1.0).sweep(rng);
    }
  }
}
```

## Training (section 4 of reference_math, U4)

```java
interface FreeEnergy {             // F(z_W | z_halo) of the fine region
  double estimate(Model m, Channel c, Window w);   // exact hook or AIS
}

interface Potential {              // F_theta over a field region (U4)
  double[] features(FieldRegion r);   // linear in theta: N(.)
  double energy(FieldRegion r);       // F_theta = features . theta
}

class Window {                     // z_W, z_halo, r outside; builds the fine region's field
  Window(Channel c, Shape shape, int ref, Proposal p, Rng rng);
  FieldRegion fineField(Model m);  // run c's writers restricted to R(W)
}

class Trainer {
  Potential fit(Model m, Channel c, int ref, FreeEnergy F, Proposal p,
                int nWindows, double lambda);      // ridge on x(W) . theta = y(W)
  List<Local> materialise(Potential phi, Channel c, int ref);  // u_c, g_{c,d} as tables,
                                                   // or an OnTheFly local when D_c is large
}
```

## What the interface hides

- How `Field` is stored: dense `(rows, cols, D)`, or sparse when most
  cells receive no write (Q5 in `dsl_updates.md`).
- When writes are re-applied: all at once before a level's sweeps
  (coarse to fine), or incrementally through `Writer.delta` when a
  same-level writer changes.
- Counters for block writes, the certificate's witness bookkeeping,
  convpot's incremental head.
- Colouring and site order inside `Kernel.sweep`.
- AIS internals (`K`, `M`, `L`, the schedule) inside a `FreeEnergy`.

## What the math fixes

- `Field.add` is additive and order-independent: `B_p` is a sum over
  writers, so writers never need to know about each other.
- `Writer` goes coarse to fine only (`h_src >= h_dst`); the kernel reads
  `Field` and `Local`s, never a coarser channel's `z`.
- `Local.score` returns the exact conditional's energy up to a constant,
  so `Kernel.sweep` is a Gibbs sweep of `E_l + Phi_l`.
- `Potential.features` is a function of the field only, so `Phi` is
  defined through the writes, not through the coarse values (U4).

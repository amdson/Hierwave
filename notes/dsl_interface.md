# DSL interface

The programmatic shape of `dsl_updates.md`: what a channel, a write, the
schedule and the trainer expose to each other.  Pseudocode, Java-ish.
Comments give the symbol from `reference_math.tex` / `dsl_updates.md`
each method implements.  Storage, incremental updates, quantisation and
scheduling live behind these interfaces and are not part of the math.

Two decisions shape it.  **A channel samples itself**: the sampler is a
method of the channel, not a shared kernel, so the tile channel can run
the bit-sliced sampler while a coarse channel runs plain Gibbs.  **A
channel declares its own inbox**: the surface coarser channels write
into is typed per channel (bit planes for tiles, dense energies for
small coarse channels), and writers are typed against it.  The math sees
one thing, the field `B_p(t)`, which every inbox can report.

## Channel

```java
interface Channel<I extends Inbox> {
  String name();
  int h();                         // block size h_c
  int D();                         // |Dom_c|
  View view(String name);          // alpha : Dom_c -> {0..V-1}, an int[D]
  IntGrid z();                     // z_c on Lambda_c
  BoolGrid fixed();                // phi_c

  I inbox();                       // what coarser channels write into (U1)

  void init(Rng rng);              // painted initialisation: a consistent refinement of the
                                   //   inbox (argmin_t B_p(t) with hard entries respected)
  void sweep(Rng rng);             // one sweep of the channel's own sampler at T = 1
                                   //   under E_l + Phi_l: inbox + same-level terms, nothing coarser

  // for training and diagnostics only
  double energy();                 // E_l(z_l | z_{>l}) + Phi_l as the sampler realises it
  void sweepTempered(Rng rng, double beta);   // p_beta ~ exp(-E_un - beta E_rest): AIS's kernel
  double unaryLogZ();              // log Z_0 = sum_s log sum_t exp(-E_un,s(t))
}
```

A channel implementation owns its same-level factors (pair tables,
convpot, the certificate, a count term's sibling counter, an on-the-fly
`Phi`).  They are constructor arguments of the implementation, not part
of the interface: the bit-sliced tile channel stores them as rows, the
table channel as a `Local` list for a numba kernel.

```java
class TableChannel implements Channel<DenseInbox> { List<Local> locals; }
                                   // reference_math section 3: colouring, Gumbel-max, certificate joint draw
class BitTileChannel implements Channel<BitInbox> { }
                                   // castlegen/bitgibbs.c: rows as level planes, bit-sliced adder, integer draw
```

## Inbox

```java
interface Inbox {
  void clear();
  double[] fieldAt(int y, int x);  // B_p, as the sampler will see it (after any quantisation):
                                   //   the field of U1/U4; the trainer reads only this
}

class DenseInbox implements Inbox {            // B stored as doubles, (rows, cols, D)
  void add(int y, int x, double[] e);          // B_p += e
  BlockRow block(Channel src, View a, int qy, int qx);   // C_q for a count write
}

class BitInbox implements Inbox {              // tiles: K context slots, each a small int per cell
  int slots();                                 //   plus a row table; the sampler sums the rows
  int bits(int k);                             // slot k holds v in 0 .. 2^bits - 1
  void setRows(int k, double[][] T);           // rows T[v][t] over tiles, quantised to FB level
                                               //   planes + a forbid plane (+inf)
  void setContext(int k, IntGrid v);           // which row applies at each cell
                                               // fieldAt(p) = sum_k T_k[v_k(p)][.]
}
```

The two inboxes are the same object to the math (`fieldAt`), different
to the writer: a dense inbox takes energies, a bit inbox takes a context
and a row table.  The quantisation in `BitInbox` is part of "the free
energy as the kernel realises it".

## Writers (coarse to fine)

```java
interface Writer<I extends Inbox> {   // one declared parent-to-child term (U1, U2)
  Channel src();                   // b, coarser or equal
  Channel<I> dst();                // a, h_a <= h_b
  int radius();                    // rho_ab in src cells (U2 rule 2)
  void write(I inbox);             // apply all of src's current writes
  Delta delta(int qy, int qx, int v);   // the change src cell q makes with z_b(q) = v: refresh
                                   //   after q changes, and scoring a coarse candidate in Phi
}

// into a DenseInbox
class TableWrite    implements Writer<DenseInbox> { View a, b; int[] off; double[][] T; }
                                   // B_p += T[a(.), b(z_b(q))], q = q_b(p, off)
class CountWrite    implements Writer<DenseInbox> { View a, b; double[][] T; }
                                   // block(q).row = T[b(z_b(q)), .]
class PromotionCost implements Writer<DenseInbox> { View alpha; double[][] cost; }

// into a BitInbox: the parent supplies a context per cell and the rows
class ContextWrite  implements Writer<BitInbox>  { int slot; View b; double[][] rows; }
                                   // v(p) = b(z_b(q_b(p, off))), rows[v][t] = T[a(t), v]
class PaintWrite    implements Writer<BitInbox>  { int slot; Painter pi; double[][] Theta; }
                                   // v(p) = pi_p from z_src, rows = Theta: the compression of U1
```

A `TableWrite` and a `ContextWrite` are the same term of the math; the
writer's job is to put it in the form the destination's sampler reads.

## Schedule

```java
class Level {
  int h; List<Channel> channels;
  List<Writer> writesIn;           // from levels >= h (U2 rule 1)
  int S;                           // sweeps, the budgeted kernel
}

class Model {
  List<Level> levels;              // coarsest first
  void generate(Rng rng) {
    for (Level l : levels) {
      for (Channel c : l.channels) c.inbox().clear();
      for (Writer w : l.writesIn) w.write(w.dst().inbox());
      for (Channel c : l.channels) { c.init(rng); for (int s = 0; s < l.S; s++) c.sweep(rng); }
    }
  }
}
```

Same-level writers between two sampled channels (Q2 in
`dsl_updates.md`), if allowed, go through `Writer.delta` after each
change; for a bit inbox that means rewriting a context plane, so the
cost is per changed cell, not per sweep.

## Training (reference_math section 4, U4)

```java
interface FreeEnergy {             // F(z_W | z_halo) of the fine region
  double estimate(Model m, Channel c, Window w);
}
class ExactHook implements FreeEnergy { }                  // transfer matrix, per-tile product
class AIS implements FreeEnergy {  // K, M, L, schedule hidden; uses the fine channel's own
  int K, M; double L;              //   unaryLogZ, sweepTempered and energy, so F is the
}                                  //   free energy as that channel's sampler realises it

interface Potential {              // F_theta over a field region (U4)
  double[] features(FieldRegion r);   // N(.), linear in theta; r is a grid of fieldAt(p)
  double energy(FieldRegion r);
}

class Window {                     // z_W, z_halo, ref outside; the fine region R(W)
  Window(Channel c, Shape shape, int ref, Proposal p, Rng rng);
  FieldRegion fineField(Model m);  // run c's writers restricted to R(W), read fieldAt
}

class Trainer {
  Potential fit(Model m, Channel c, int ref, FreeEnergy F, Proposal p,
                int nWindows, double lambda);           // ridge on x(W) . theta = y(W)
  List<Writer> materialise(Potential phi, Channel c, int ref);
                                   // u_c, g_{c,d} as tables into c's own inbox / same-level
                                   //   factors, or an on-the-fly term scoring Writer.delta
}
```

## What the interface hides

- Inbox storage: dense doubles, or context planes and quantised rows.
- The sampler: colouring and Gumbel-max, or bit-sliced rows and an
  integer draw; the certificate's witness bookkeeping; count counters.
- When writes are re-applied: all at once before a level's sweeps, or per
  changed cell through `Writer.delta`.
- AIS internals inside `FreeEnergy`.

## What the math fixes

- `fieldAt` is a sum over writers (additive, order-independent), so
  writers never know about each other; a bit inbox realises the sum by
  summing rows.
- Writers go coarse to fine; `sweep` reads the inbox and same-level
  terms, never a coarser channel's `z`.
- `sweep` is a Gibbs-type kernel under `E_l + Phi_l` at `T = 1`, whatever
  its implementation, and `energy`, `sweepTempered`, `unaryLogZ` make it
  usable as AIS's kernel, so training measures the kernel that ships.
- `Potential.features` depends on `fieldAt` only, so `Phi` is defined
  through the writes (U4), and the same `Phi` fits any inbox type.

## Open

- A count write into a `BitInbox`: the row depends on the sibling count
  `n_{-p}`, which the bit-sliced sampler would have to maintain itself
  (one slot whose context the channel computes).  Support it there, or
  keep count terms to dense channels?
- Quantisation: `fieldAt` reports quantised energies, so `Phi` is fitted
  to the quantised kernel.  Fine for training; for diagnostics against
  `p*` the unquantised `E` is also wanted (`energy(quantised=false)`).

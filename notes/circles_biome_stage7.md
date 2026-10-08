### Stage 7: scaling probe

Script `notes/experiments/circles_biome_scale.py` (results
`images/cbio_stage7.json`, log `images/cbio_stage7_log.txt`), 47 s single
core for the whole stage including the optional fit (two other agents were
running on the machine, so absolute times carry some load noise; the cost
probe interleaves its configurations to share it).  Figure
`images/cbio_scaling.png` (per-site time and parameter count against N, log
axes); render `images/cbio_scale20.png` (N = 20: oracle left, forward
`+bu+ou` K = 8 right; none blocks hatched).

Config: `CirclesBiome(6, 6, families=random_families(N, rng),
masks=random_pair_masks(N, rng))`, N in 2, 5, 10, 20, default dials (mu
0.3, `b_fam = F_fam - log 16` per family, `bio_u0 = 4 log(1 + #admitted)` =
0 / 4.39 x 7).

- `random_families` (new, additive): each family is a 4-connected set of
  6-14 cells grown from a seed by random frontier additions, its bounding
  box kept within 5 x 5 (so at least 16 in-block anchors exist), shapes
  distinct up to translation; anchors = every in-block position, 16 drawn at
  random when there are more (always more here).  Rings and presence
  bonuses follow from the footprint as for discs and bars.
- `masks=` (new constructor argument): the biome's values are an explicit
  list of admitted-family bitmasks (`C.MASKS`); default `arange(2^N)`, so
  every existing test is untouched.  `FAMOK`, `bio_u0`, the `allow_<name>`
  views, `_group`'s biome map (an element is kept only if the mask list is
  closed under its family permutation; identity always) and `render`'s
  dormant hatching (`MASKS[biome] == 0`) read `MASKS`; `paint_allow`,
  `dormant_of`, `stats` (bio_hist over `P = len(masks)` values, mask_viol),
  `reference()["bio_u"]`, `top_probs` already went through `FAMOK` / `P`.
- `random_pair_masks(N, rng)`: `[0] + 7` pairs, consecutive pairs of fresh
  random permutations, distinct while distinct pairs remain; D_biome = 8 at
  every N.  N = 2 has one pair, repeated 7 times (so the biome is none vs
  both at prior odds 1 : 7); at N = 20 the 7 pairs cover 14 of the 20
  families (a pair biome cannot cover 20 families with 7 values; the other
  6 are never admitted).
- Learned tables: `+bu+ou` of stage 3, i.e. `theta0` with `bio_u =
  reference(centred=False)["bio_u"]` and `obj_u = reference()["obj_u"] -
  pres` (installed in full: the pair reference is cheap, see build times);
  the four obj pair tables are installed as zero D x D tables as in stage 3
  (so the K = None soft rows include them).
- Oracle 20 + 100 sweeps per N; forward `Forward(use_sampler=True, hb =
  BT)`, `S_T = S_M = 30`, `S_F = 20`, 16 runs per K, K in None, 8.  Per-site
  time is stage 3's: obj stage time / (active slots x S_M).  Cost probe:
  every biome set to a random non-none value (active fraction 1), obj from
  empty, 30 obj sweeps, 40 reps (median), four samplers interleaved per rep:
  the full forward model and a hard-rows-only model (mask + support; `pres`
  and every learned table removed), each at K = None and K = 8.

**Per N** (forward times are means of 16 runs; probe us are medians of 40):

| N | D_obj | build ms (support + pair ref) | oracle conflict | families seen / admitted | fw us/site K=None | fw us/site K=8 | probe us/site K=None | probe us/site K=8 | obj ms/run None / 8 | wall ms/run None / 8 | dormant | conflict | present fw (None / 8) | present oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 33 | 3.0 (1.4) | 0 | 2 / 2 | 2.03 | 1.12 | 2.09 | 1.14 | 7.7 / 4.2 | 46.7 / 44.0 | 0.127 | 0 | 0.570 / 0.576 | 0.868 |
| 5 | 81 | 7.8 (6.0) | 0 | 5 / 5 | 3.46 | 1.65 | 2.88 | 1.62 | 13.1 / 6.2 | 61.0 / 47.5 | 0.127 | 0 | 0.573 / 0.561 | 0.909 |
| 10 | 161 | 15.0 (9.6) | 0 | 10 / 10 | 4.57 | 2.48 | 4.40 | 2.31 | 17.3 / 9.4 | 68.1 / 58.4 | 0.127 | 0 | 0.564 / 0.566 | 0.912 |
| 20 | 321 | 35.4 (25.3) | 0 | 14 / 14 | 6.67 | 3.50 | 5.92 | 3.00 | 25.1 / 13.2 | 85.1 / 72.8 | 0.127 | 0 | 0.554 / 0.585 | 0.882 |

Parameters and memory (the obj level):

| N | D_obj | stamp features (nfeat) | tabular 4 D^2 + D | support entries 4 D^2 | INF density | oracle s (120 sweeps) |
|---|---|---|---|---|---|---|
| 2 | 33 | 68 | 4 389 | 4 356 | 1.0% | 0.8 |
| 5 | 81 | 68 | 26 325 | 26 244 | 1.7% | 1.5 |
| 10 | 161 | 68 | 103 845 | 103 684 | 1.6% | 2.2 |
| 20 | 321 | 68 | 412 485 | 412 164 | 1.5% | 3.8 |

`nfeat` is `stamp_features(...).shape[1]` (OFF8 pairs over the 4-value
dem paint): 68 at every N, by construction.  The support (and the exact
pair reference, computed alongside it on the 3 x 3 block canvas) builds in
25 ms at D = 321, so the pair reference was affordable and installed; the
"single-object unary only" fallback was not needed.  The support tables
hold 4 D^2 = 412k entries at N = 20 (as float64 tables with their
reflections, 8 x 321^2 x 8 bytes = 6.6 MB if packed that way), 1.5% of them
INF.

**Hard-row cost** (probe, us per active site per sweep, medians):

| N | D | full K=None | full K=8 | hard only K=None | hard only K=8 | hard / full at K=8 | soft rows at K=8 (full - hard) | soft rows at K=None |
|---|---|---|---|---|---|---|---|---|
| 2 | 33 | 2.09 | 1.14 | 1.61 | 0.89 | 78% | 0.25 | 0.48 |
| 5 | 81 | 2.88 | 1.62 | 1.96 | 1.39 | 86% | 0.23 | 0.92 |
| 10 | 161 | 4.40 | 2.31 | 2.65 | 2.00 | 87% | 0.31 | 1.75 |
| 20 | 321 | 5.92 | 3.00 | 3.33 | 2.76 | 92% | 0.24 | 2.59 |

Packed obj rows: 19, of which 9 hard (the mask, a pair to `allow`, and the
support at 8 directed offsets; 5 hard at N = 2, whose two shapes never meet
diagonally so the diagonal support tables carry no INF and pack as soft) and
10 soft (`pres`, `obj_u`, the 4 pair tables x 2 directions).  The hard pass
runs every hard row over all D values before the cap draws (`_site_cap`:
`_rows(..., 0, nhard, cand, D, ...)`); the `skip` test makes a row cheap on
already-forbidden values but still visits them.  At K = 8 the soft rows on
the capped set cost a flat 0.23-0.31 us at every N; everything that grows
with N is the hard pass: full K=8 and hard-only K=8 grow with the same
slope, 6.5 ns per value of D ((3.00 - 1.14) / 288 and (2.76 - 0.89) / 288).
At K = None the soft rows add a second O(D) term (13.3 ns per value in
total).  So at D = 321 the hard pass over all D is 92% of a capped site
update and is the remaining O(D) cost.

What would remove it: the mask is a parent row, constant through the obj
sweeps, and with pair biomes it admits 1 + 2 x 16 = 33 of the 321 values
(10%) at every active slot, independent of N.  A per-block admissible list
computed once at `init` from the parent hard rows (C1's `A_p`; `Sampler.init`
already computes `admissible()` under exactly those rows, and throws it away
except for dormancy) would let the kernel run the same-level hard rows (the
support) and the soft rows only over `A_p`, making the hard pass O(|A_p|) =
O(16 x families per biome), constant in N.  The expected per-site cost is
then the N = 2 figure (~1.1 us at K = 8).  Storage is one list per biome
value (8 here), not per site, since `A_p` depends on the slot only through
its `allow`.

**Optional: the mid fit at N = 5** (`train.fit`, `ExactTargets(Oracle.
mid_probs)`, `stamp_features`, K = 0, 3 iterations, 4 contexts, via
`biome_fit.Bench` / `materialise`; 22 s): 172 records (52 / 120 / 172
cumulative), held-out KL 3.3e-7 / 3.8e-7 / 3.1e-7 by iteration (train
2.0e-6), 68 parameters.  The materialised pair tables (double-centred over
their finite entries, support INF from the stamps) against `reference()` on
the 15 316 finite value pairs (t, t' > 0) never seen as (candidate,
neighbour) in the data passes: max abs error 6.4e-4, mean 3.3e-5, slope
0.99995, corr 1 - 1.3e-8, against a reference range of 5.24 nats.  The
contexts are few because the Bench forward at theta0 leaves most top
blocks none (the designed `bio_u0` uncancelled), and still the fit
generalises to every unseen pair: the stamp features are exact for this
geometry, so they need only enough records to identify 68 numbers.

### What the numbers show (stage 7)

(a) *Per-site work at K = 8 is not yet constant in N.*  It grows from 1.1
to 3.0 us (probe) as D goes 33 -> 321, against 2.1 -> 5.9 us at K = None:
the cap halves the cost at every N, but both grow linearly in D.  The
probe decomposes it: the soft rows on the capped set are flat (0.24 us),
and the growth is entirely the hard pass, which visits all D values with
every hard row (92% of a capped update at N = 20).  The fix is the C1
admissible list from the parent rows at init, after which the per-site
work should be the N = 2 cost for any N with two families per biome.  This
is a kernel change (`_site_cap` over a per-site or per-allow candidate
list), not a model change; it is left for the Sampler owner.

(b) *The parameter count is flat with stamp features.*  68 features at
every N, against 4.4k -> 412k tabular entries (4 D^2 + D); adding a family
adds zero learned parameters.  At N = 5 the exact-target fit reproduces the
reference pair tables on 15k unseen pairs to 6e-4 nats from 172 records,
so the derived features generalise across 5 random shapes as they did for
disc and bar.  The support is derived from the stamps (25 ms at D = 321)
and costs memory quadratic in D (412k entries); it is not learned.

(c) *Dormancy does its job with sparse masks.*  Conflict is 0 in every
oracle sweep and every forward run at every N and K (mask violations 0
in the new test; not monitored in the script).
The dormant fraction is 0.127 at every N (the none share of a uniform
8-value biome, 1/8), and none blocks are skipped whole (`hb = BT`); `init`
costs 0.10-0.18 ms.  But with sparse masks dormancy covers only the none
blocks: the useful sparsity is inside the active blocks (90% of the values
masked out at N = 20), and today the kernel pays for those values on
every visit, which is (a).

(d) *The rest scales as expected.*  The forward run is 44 -> 73 ms at K = 8
(the obj stage 4 -> 13 ms, tiles ~37-55 ms whose work does not depend on N;
the spread follows machine load), the oracle 0.8 -> 3.8 s per 120 sweeps
(its collapsed move enumerates all D).  Present fractions (forward 0.55-0.59
against the oracle's 0.87-0.91) are off for the reasons of stage 3: the
pair part of `Phi` is missing and the oracle keeps almost no none blocks;
that is not what this stage measures.

Deviations: the masks cannot cover 20 families with 7 pairs (14 covered at
N = 20; "every family appears" is checked against the admitted ones); at
N = 2 the 7 pairs are one pair repeated; random stamps are kept within a 5
x 5 box so that every family has 16 offsets; the cost probe uses 40
interleaved reps (medians) rather than stage 3's 20 sequential ones, because
of load from the concurrent agents.

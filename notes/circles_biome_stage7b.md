### Stage 7b: the admissible list

The fix queued by stage 7 (notes/circles_biome_stage7.md, "Hard-row cost"):
dsl_updates C1's candidates-from-the-allowed-set, built into the Sampler.
Script `notes/experiments/circles_biome_scale2.py` (results
`images/cbio_stage7b.json`, log `images/cbio_stage7b_log.txt`, 53 s);
the stage 7 script is unchanged.

**Design as built.**

- `kernel.adm_lists(..., rows_sel, D)` (new): for every home site, the
  energies of all D values under the parent rows (`Sampler.parent_rows()`:
  hard unaries and hard pairs whose other channel is not home, the same
  rows dormancy reads), and the list of finite ones with their parent
  energy.  Lists are deduplicated by content (a 64-bit hash of (value,
  energy bits), confirmed by full comparison) into CSR arrays `adm_ptr`,
  `adm_idx` (ascending values), `adm_e`, with `adm_id[y, x]` (int32) naming
  the site's list.  Memory O(sites + #distinct x |A|): 4.3 kB at every N
  here (7 lists of 33), against 46 kB for a dense (sites x D) bool at
  N = 20.  Build time O(sites x D x #parent rows), once per `init`.
- `Sampler(..., adm=True)` (new, default off): `init()` builds the lists,
  takes dormancy from them (a list of length 1; identical to the
  `admissible()` route), and records the remaining hard rows (`sib_rows`, the
  same-level ones: here the support at 8 offsets) as contiguous runs
  `[f0, f1)` of packed rows.  `sweep()` then runs `kernel.sweep_adm`;
  `candidates(y, x)` returns its set (`kernel.site_candidates_adm`).
  Invariant (docstring): the lists are valid until the parents change;
  `init()` again after they do; `refresh()` drops them (back to `sweep_cap`
  until the next `init`).  Certificate channels, `relax` and the tempered
  kernels keep `sweep_cap` (their parents are softened or the draw needs all
  D).
- `kernel.sweep_adm` / `_site_adm` (new): at a site, copy the list and its
  parent energies, run the sib hard rows on the list only (skipping values
  already inf), then exactly `_site_cap`'s logic on what remains: K = None,
  every listed value, soft rows on all of them; K > 0, `cand[0] = z_p`
  (energy inf if z_p is unlisted), the n listed values != z_p that the sib
  rows admit in ascending order, the same partial Fisher-Yates
  (`randint(0, n - j)`, energies swapped alongside), soft rows on the capped
  set, Gumbel-max draw.  The set of n finite candidates and their order are
  those of `sweep_cap` (which drops the same values as inf), and `_draw`
  consumes one uniform per finite entry, so the random stream is the same:
  **bit-identical** to `sweep_cap` for the same seed, not just equal in
  distribution.  The one caveat is floating-point order: the parent energy is
  summed first, so if a parent row with nonzero finite entries were packed
  after a sib row the sum could differ in the last bit (not the case in any
  model here: the masks are 0 / inf).
- One detail mattered for speed: evaluating the sib rows one `_rows` call
  per row (`f, f + 1`) cost ~0.4 us per site more than one call over a range
  (it made the adm path *slower* than the old one at N = 2, 5).  The sib rows
  are therefore passed as runs; here they are one run (the parent mask is
  packed first).

**Tests** (`tests/test_sampler.py`, additive): `test_adm_identity` (grids
equal to `sweep_cap` after `init` + 4 sweeps, same seed, for Potts(2, 2)
top/mid/tile with and without a hard palette parent row, Circles(2, 2)
top/mid/tile with and without a hard slot row, the toy with a hard parent
mask, K in None / 2 / 4 (K skipped on certificate homes); and CirclesBiome(2,
2) with 5 random families and pair masks, obj, K None / 8, 10 sweeps);
`test_adm_lists_and_dormancy` (the list at every site = `admissible()` under
the parent rows; dormant = lists of length 1 = `C.dormant_of(allow)`; at most
one list per biome value; dormant sites unchanged by 20 sweeps; every value
drawn admissible; `candidates` = z_p plus admissible values);
`test_adm_long_run_biome` (6000 sweeps each, 30 batches: family histogram,
value parity and energy, old path vs adm path with different seeds, |z| < 4
at K None and 8; observed max |z| 1.2 / 1.6).  In the scale script the
forwards on both paths end in the same states (present fractions equal to
the last digit at every N and K).

**Per-site cost** (us per active obj site per sweep; probe at active
fraction 1, S_M = 30 sweeps, 80 interleaved reps, medians; "stage 7" is the
stage 7 table; "old" is `sweep_cap` re-measured in this run, interleaved
with "adm", so the old/adm comparison shares the load; the machine carried
two other experiments, which is why "old" sits above stage 7 at N >= 5):

| N | D | K=8 stage 7 | K=8 old | **K=8 adm** | K=None stage 7 | K=None old | **K=None adm** | hard only K=8 old / adm | K=8 adm, tables spread 10 x 10 | forward K=8 old / adm | forward K=None old / adm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 33 | 1.14 | 0.99 | **0.96** | 2.09 | 1.81 | **1.79** | 0.76 / 0.73 | 1.09 | 1.01 / 0.98 | 1.89 / 1.80 |
| 5 | 81 | 1.62 | 1.91 | **1.47** | 2.88 | 3.38 | **2.48** | 1.61 / 1.18 | 2.04 | 2.10 / 1.83 | 3.67 / 2.78 |
| 10 | 161 | 2.31 | 2.83 | **1.77** | 4.40 | 5.29 | **2.82** | 2.46 / 1.38 | 2.73 | 2.97 / 1.96 | 5.34 / 2.97 |
| 20 | 321 | 3.00 | 3.43 | **1.66** | 5.92 | 6.87 | **2.56** | 3.03 / 1.23 | 2.82 | 3.81 / 2.18 | 7.19 / 2.72 |

Init and lists:

| N | init ms old / adm (probe, 144 sites) | distinct lists (all biomes non-none) | list lengths | list bytes | dense bool bytes |
|---|---|---|---|---|---|
| 2 | 0.077 / 0.133 | 1 | 33 (= all of D) | 1 120 | 4 752 |
| 5 | 0.124 / 0.215 | 7 | 33 each | 4 336 | 11 664 |
| 10 | 0.167 / 0.296 | 7 | 33 each | 4 336 | 23 184 |
| 20 | 0.187 / 0.316 | 7 | 33 each | 4 336 | 46 224 |

Packed obj rows 19; hard 5 at N = 2 (mask + 4 orthogonal supports) and 9 at
N >= 5 (mask + 8 supports); 1 parent row (the mask) at every N.

### What the numbers show (stage 7b)

(a) *The O(D) hard pass is gone.*  At N = 20 the capped update drops from
3.43 to 1.66 us (2.1x) and the full one from 6.87 to 2.56 us (2.7x); the
hard-only probe from 3.03 to 1.23 us.  From N = 5 to N = 20 (D 81 -> 321,
4x) the adm cost at K = 8 is 1.47 / 1.77 / 1.66, flat within the load noise,
where the old path grows 1.91 -> 3.43; at K = None it is 2.48 / 2.82 / 2.56
against 3.38 -> 6.87.  The cost now follows |A_p| = 33, not D.  Every
active site is visited over 33 values whatever N is, as predicted.

(b) *The claim "flat at the N = 2 cost (~1.1 us)" holds only from N = 5 on.*
N = 2 is cheaper (0.96) for a structural reason, not a D one: its two shapes
never meet diagonally, so only 4 of the 8 supports are hard (5 hard rows, not
9) and the other 4 run as soft rows on the 8 capped values instead of hard
rows on 33.  At N = 2 the list is all of D, so adm = old there (0.96 vs 0.99:
no overhead from the list).  The fair baseline for N >= 5 is ~1.5 us.

(c) *What remains that depends on D is memory, not work.*  The work per site
is now constant (33 values x 8 sib rows, 8 values x 10 soft rows), but the
tables the rows read are D x D: 16 pair tables (8 support, 8 learned pair
directions) are 0.84 MB at D = 81 and 13.2 MB at D = 321, so the per-candidate
lookups `tab[c, v_b]` fall out of L1 then L2 as D grows.  The cache check
reads the same values from tables spread 10 x 10 apart in memory (100x the
footprint): +0.13 us at N = 2, +0.6 at N = 5, +1.0 at N = 10, +1.2 at N = 20,
so the residual N = 5 -> 20 drift (and much of the remaining gap to N = 2)
is in the memory hierarchy.  Remedies, not done here: store the tables
value-of-neighbour-major (the 33 admitted values are two runs of 16
consecutive anchors, so a column read is a few cache lines), or evaluate
support and pair energies from the stamp features rather than D^2 tables.
The table storage (4 D^2 support entries) is the remaining quadratic
memory cost, unchanged by this stage.

(d) *What is O(D) and is fine.*  `adm_lists` runs the parent rows over all
D at every site once per `init`: 0.13 -> 0.32 ms per init (1.7x the old
`admissible()` dormancy pass, which it replaces), against 30 sweeps of
~0.2-0.4 ms each at the obj level.  It could be O(#distinct contexts x D) by
keying on the parent values instead of the site, but at these sizes it is
not worth it.  The five per-sweep scratch arrays are size D (allocation,
not work).  The list memory is O(sites) for `adm_id` plus one list per
distinct parent context (7-8 here), not O(sites x D).

(e) *Forward runs.*  With dormancy (12.7% none) the forward obj stage at
K = 8 goes 3.81 -> 2.18 us per active site at N = 20 and 7.19 -> 2.72 at
K = None, with identical samples (bit-identical kernel).  The
gains are slightly smaller than in the probe because the forward obj grid
starts empty and fills, and its sites are partly dormant (skipped in both).

Deviations: `adm` is an opt-in constructor flag (default off) so that the
concurrently running stage 4-5 experiments, which build Samplers through
`Forward`, are untouched; `Forward` does not expose it (circles_biome.py
was not edited), and the scale2 script switches it on through a `Forward`
subclass.  Since the path is bit-identical, flipping the default (or
passing it from `Forward`) is safe once those runs are done.  The machine
was loaded (load average 6-8) during the timed run; the old/adm comparison
is interleaved and fair, the absolute numbers are ~10-20% above an idle
machine.

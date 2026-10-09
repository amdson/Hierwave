# Trees on terrain: the recursion with a sampled fine level, limbs, and the tree exemplar copied (2026-10-09)

`castlegen/channels/{trees,treeland,limbs,actais,treecopy}.py`,
`notes/experiments/{trees_fit,treeland_gen,tree_exemplar}.py`, `tests/channels/test_{trees,treeland,treecopy}.py`.
The first test of the bootstrap trainer (`dsl_updates.md` C2, `train.fit`)
where the level below the fitted one is itself sampled with same-level
couplings, so the collapsed target is a genuine free energy; then the first
structured type (trees with connected limbs of non-increasing mass, the
root set mirrored above ground) and what its free energy does to the
estimators.

## Stage A: tree slots over a chain terrain (`trees.py`)

Side view, H = 40 rows, 24 interior slot columns of 8 plus a margin of 2
fixed-absent slots per side (no periodicity: the chain's correlation
length is a few columns, and the margin keeps every record interior).

- `surf` (h = 8): the band chunk's rows under the surface, the column height
  h in 0..8.  Energy kappa (h - 4)^2 per column (through a pair to the
  painted chunk-row channel) and c |h_j - h_{j+1}| between neighbours (a
  pair over the chunk values, exact because the fill structure makes the
  per-row differences add up).  kappa 0.1, c 1.
- `tree` (h = 8): a slot, D = 65: absent or a trunk at (row r, column c) of
  the slot.  Unary -b per present value, b = -1.5 (oracle presence 0.64).
  The trunk sits on the surface: the slot paints a demand `dem8` into its
  column (the height the terrain must have there), a hard pair on `surf`.
- Tiles painted (sky / soil / trunk).  No limbs in stage A.

Given the trees, the terrain is a chain over chunk columns with pinned
columns: a transfer matrix gives its log Z (the exact collapsed target, the
`hook` of `AISTargets`) and exact samples (the oracle's terrain move; block
Gibbs with the slot move p*(t | terrain) is the oracle of p*).  The induced
potential on the trees is -log of the terrain's marginal at a pinned
column (a unary over the trunk height) and the negative PMI between pinned
heights at slot distance 1, 2.  Learned: tabular `tree_u` over the view
`hclass` (0 absent, 1 + r) and `tree_p1`, `tree_p2` over hclass pairs at
slot offsets 1, 2 (90 parameters); designed offset = the presence unary.
Targets over the window of the slot's own column dilated by `reach` chunk
columns each side, the terrain outside held (AISTargets' convention):
`hook` exact, `ais` = AISTargets over the `surf` sampler (the pins and the
height prior in p_0, the smoothness annealed, K 32, M 16).  4 iterations x
8 contexts, K in {0, 3}, probes 2 per context.  End to end: 64 forward runs
(30 tree sweeps, 30 surf sweeps) against two independent oracle chains
(100 burn, 2000 sweeps; their L1 distance is the eval noise).

| stat | oracle A | oracle B | theta0 | hook K0 | hook K3 | ais K0 | hook K0, reach 3 |
|---|---|---|---|---|---|---|---|
| present | 0.642 | 0.638 | 0.943 | 0.646 | 0.639 | 0.649 | 0.648 |
| p1_both | 0.412 | 0.407 | 0.893 | 0.417 | 0.405 | 0.422 | 0.421 |
| p1_eq (adjacent trees, same height) | 0.225 | 0.220 | 0.128 | 0.232 | 0.228 | 0.235 | 0.230 |
| p1_far (adjacent, differ >= 2) | 0.044 | 0.042 | 0.553 | 0.050 | 0.043 | 0.048 | 0.050 |
| p2_eq | 0.159 | 0.154 | 0.115 | 0.165 | 0.159 | 0.167 | 0.156 |
| terr_rough (mean adjacent height difference) | 0.585 | 0.587 | 2.433 | 0.599 | 0.603 | 0.603 | 0.617 |
| terr_mean | 4.03 | 3.97 | 3.42 | 3.85 | 3.94 | 3.87 | 3.87 |
| L1 to oracle A (13 stats) | - | 0.0083 | 0.742 | 0.0249 | 0.0144 | 0.0238 | 0.0228 |
| KL held out | | | | 0.037 | 0.035 | 0.040 | 0.034 |
| fit seconds | | | | 30 | 40 | 338 | 50 |

- At theta0 the forward puts a tree in nearly every slot at a random
  height and pins the terrain to it (roughness 2.4 against 0.59); the fit
  recovers presence, the height histogram (max bin error 0.02), the
  adjacent-height statistics and the terrain's roughness.  Residual 2-3x
  the eval noise (K3 1.7x), mostly in `terr_mean` (the forward's terrain
  is 0.1-0.2 lower: 30 single-chunk sweeps from a flat start, not the
  learned potential).
- AIS targets match the exact hook (L1 0.0238 vs 0.0249, same tables to
  corr 0.92 / 0.77 with the oracle's -PMI at distance 1 / 2) at 11x the
  cost: the estimated-targets recursion works where the child mixes.
- The window boundary is not the residual: reach 3 (a window of 7 columns)
  gives the same L1 as reach 1.  The pair consistency
  of the targets is 1e-14 only at iteration 0 (every slot full) and ~1 nat
  after: the targets at p and q are conditionals given different fine
  boundaries, so their one-site odds need not close; the fit's own closure
  is 1e-15 (a pairwise potential).  Not a bug; a property of windowed
  targets worth remembering when reading `viol_targets`.
- A bug found: `AISTargets.log_z` left the painted channels at the last
  candidate's value after restoring the site, so the next site in the
  colour order saw a stale neighbour (here a spurious pin; 0.009 max
  change in a target).  Fixed (repaint after restoring); the numbers above
  did not move at this precision.

## Stage B: limbs (`limbs.py`, `treeland.py`)

The terrain comes first here and the trees read it (the tower grammar's
order, the one where "trees react to the terrain" is literal): level 2 is
`surf` then `tree` (same h = 8, the slot reads surf at offset 0 through a
hard placement pair, admitted list = absent + the 8 columns at the surface
row); level 1 is the tile channel with `limbs.KINDS` (sky, soil, trunk with
a north port, limb{mass 1..3}_{1..4 ports}: 48 kinds), the root set's
views and statistics counted by `roots.exemplar_stats` from a hand-painted
tree exemplar through `roots.tables(rows=...)`, contact rules mirrored
(limbs never touch soil, a trunk has soil below and none above), and the
root certificate with `tree=True` (a forest rooted at the trunks, mass
non-increasing away from them).  Painted `ground` and `tdem` force soil and
the trunk cell.

### Generation (`treeland_gen.py`, 64 runs, 24 slots, grow 1, dangle 4, 150 tile sweeps)

The exemplar is the big root system of `coord.py` (40 x 84, painted from
strokes for the exemplar-coordinates pass, 31 % of its cells branch
points) flipped vertically; the small root exemplar mirrored by hand
(`limbs.EXEMPLAR_SMALL`, 88 % of its cells degree 2) was tried first and
gives wandering chains with under 1 % branch points at any grow / dangle /
sweep setting (19 cells per tree at grow 3).

- Valid: 0 certificate violations in 64 runs, 22640 limb cells all
  attached to a trunk (fraction 1.0000), 0 multi-parent cells; 3 dangling
  ports per tree (the tips; soft).  1.4 s per run.
- 23.0 limb cells per tree (sd of the mean 0.4); presence 0.646 (the
  designed value: no learned terms).  Degree histogram about 11 % tips,
  72 % chains, 15 % branch points, 1 % four-way; mass about 38 / 19 / 43 %
  for 1 / 2 / 3: bushy, thick at the trunk, thin green tips, wide and low
  like the root system it was counted from (images/treeland_sheet.png).
  grow 2 / 3 give 49 / 74 cells per tree and thickets.
- The trees react to the terrain through the hard rules alone: trees whose
  slot has a relief of 2 or more rows grow 21.0 limb cells against 23.2 on
  flatter ground (n 170 vs 822); with the ground one row higher on the
  right the limbs sit 1.6 cells further left (n 218, sd of the mean 0.45),
  one row higher on the left 0.5 cells further right (n 220), two rows
  1.3 / 1.7 (n 52 / 74), consistent in sign.  Presence does not depend on
  the relief (0.64 / 0.65 / 0.69 at relief 0 / 1 / 2+, within noise), as
  designed: the forward has no term for the limbs' free energy.

### Learning: the limbs' free energy is not estimable with the shipping tools

What the tree level should learn is the limbs' free energy given the trunk
and the terrain around it (room for limbs: a tree on a slope or beside a
cliff has fewer limb configurations, so p* places it less often there).
Two estimators of log Z over the slot's limb window (16 rows x 16 columns
around the trunk, the tiles outside held), differences across the
candidates absent / trunk at column 0 / 3 / 7, two seeds each:

| estimator | model | K, M | seed-to-seed max difference of log Z differences (nats) | standard error |
|---|---|---|---|---|
| `AISTargets` (couplings annealed from independent sites) | soft, no certificate | 32, 16 | 10 (grow 1), 10 (grow 2), 3.5 (grow 3) | ~1 (saturated) |
| same | same | 128, 16 | 6, 6.5, 4.8; 13 with a 24 x 20 window | ~0.7 |
| `ActivityAIS` (couplings on, the limb activity annealed from +MU) | soft | 50..200, 8..16 | 13..26 | ~1 |
| `ActivityAIS` with the certificate kernel, boundary parent chains kept | certificate | 50..200, 8..16 | 3.3..13 | ~1 |

- The soft model (no certificate) is not a usable surrogate: its window
  log Z is ~200 nats even with no trunk, because detached limb fragments
  with matched ports cost only their unaries.  Everything the certificate
  does at generation is missing from the soft free energy.
- With the certificate (`actais.py`: every intermediate distribution keeps
  all constraints including the certificate, only (1 - beta) MU per limb
  cell is annealed, z_0 is the empty window with the trunk at d = 0 and the
  parent chains of outside cells that depend on the window kept fixed,
  log Z_0 = -E(z_0) exactly), the estimate is still off by 3-13 nats
  between seeds at K up to 200, and the candidates differ by 1-3 nats.  The
  limb structure needs ~150 sweeps to grow under single-site moves; no
  affordable schedule is near equilibrium at any beta, and the weights
  (Delta beta x MU x ~20 cells per step) carry that lag.
- A target noisy by several nats is not merely imprecise: the softmax over
  candidates goes to the luckiest one, the expected target is near uniform,
  and the fit learns nothing.  So the stage B fit was not run.

The diagnosis is the C1 criterion in its sharpest form: the fine kernel
does not mix with the coarse values clamped, and the recursion inherits
that.  Ways out, none built:

1. Joint growth moves in the tile kernel (add / remove a limb segment),
   the same need the tower negative result found (`channels_demo.md`
   2026-10-06: a kernel that draws a whole row jointly).
2. Limbs as coarse records (segments with length and direction, as the
   compiled grammar's values) whose free energy is enumerable; the limb
   tiles become a painter.  This is the blockgram direction applied to
   the first structured type.
3. For the terrain-first ordering specifically, the surf level's targets
   are exact once the tree level is installed (enumeration over the
   window's slots), so only the tree level's limb free energy is the
   problem.

## Stage C: the procedural tree exemplar copied with guarantees (`treecopy.py`)

The tree exemplar meant all along was the legacy one: the procedural
picture of `notes/experiments/legacy/tree_texsyn.py` (64 x 64; branches
that split, leaf clumps at the tips, grass and soil), which the old
texture synthesis copied with no guarantees.  `castlegen/channels/treecopy.py`
copies it through the masked exemplar-coordinates pipeline of the roots
(`coord.py`: coordinate channel, joint (u, t, d) kernel, `Coarse` window
channel and refinement) under the tree certificate.  The trick that makes
the roots machinery apply unchanged is to run upside down: flipped
vertically, a tree hanging from a ground line at the top is a root system
(the medium, air, takes the root view's EARTH code; the ground is the
"sky"), so the surface rule, the level-8 certificate and the refinement
need no mirror image; `render()` flips back.

Exemplar parse: bark mass = taxicab distance to the nearest non-bark,
non-ground cell (1..3), leaves mass 1; the 4-connected component holding
the trunk cell is kept (five leaf specks off the bark are dropped); a
breadth-first spanning tree from the trunk gives the ports and the mass is
clamped non-increasing along it, so the exemplar itself satisfies
`Certificate(tree=True)` (`check_exemplar` 0).  Kinds: air, grass, soil,
trunk (mass 3, three ports), limb{1..3}, leaf, dark leaf x 15 port sets,
79 in all.

Guarantees, hard in the kernel: the certificate (every tree cell on a join
path to a trunk, mass non-increasing from it, exactly one parent); contact
(tree cells never touch the ground, a trunk has ground below and none
above, a thin limb or leaf never rests on the ground); free cells hold air
or tree cells only, the ground and the trunks are fixed cells.  Soft: the
coupling to the exemplar's statement (nu 6), coherence (lam 1), dangling
ports (1), the refinement (mu 2).

| schedule | tree cells (5 trunks; exemplar 476 each) | leaves | violations / multi-parent | attached | trunks | in ground | mismatched | seconds |
|---|---|---|---|---|---|---|---|---|
| sequential growth, 600 joint sweeps | 323 | 16 | 0 / 0 | 1.000 | 5 | 0 | 18 | 39 |
| sequential, 4000 sweeps | 495 | 9 | 0 / 0 | 1.000 | 5 | 0 | 15 | 230 |
| coarse (100 level-8 sweeps) + refinement | 2337 | 1107 | 0 / 0 | 1.000 | 5 | 0 | 0 | 1 |
| coarse + 60 joint sweeps | 2344 | 1112 | 0 / 0 | 1.000 | 5 | 0 | 47 | 5 |

Sequential growth stalls at a fifth of the exemplar (the copy climbs the
trunk as a comb and dissolves at the thin tips as fast as it grows); the
coarse seeding places 24 windows per trunk and the refinement writes a
complete copy, leaves on tips 1.000, which the joint sweeps then vary
(47 coordinate mismatches, 41 dangling ports, 0.999 of leaves on tips).
images/tree_exemplar_{exemplar,coarse,seq}.png, images/tree_exemplar.json
(solid kind colours; `render_ports` draws the root set's port sprite, which
makes one-cell branches look like wire and a comb of the trunk).

Two bugs in `coord.Coarse` found on the way, both fixed and both loosening
the roots case correctly:
- The edge signatures required equal mass on both sides of every port
  crossing a chunk edge, so two true neighbour windows of the exemplar were
  incompatible wherever a branch thins across the edge; the placement
  stalled at 5 windows per trunk.  The match is now the certificate's rule
  (the parent's mass at least the child's; `sig_ok`).
- A window whose parent-ward crossing faced an empty chunk was forbidden
  outright, so two windows whose branch crosses their shared edge in both
  directions waited for each other forever (a leaf clump's spanning tree
  does this).  The crossing is now a cut at a finite cost; the level-8
  certificate still wants a placed parent-ward neighbour on some side and
  the refinement prunes what the placed windows leave orphaned.  43 -> 120
  windows for 5 trunks.

Diversity by recombination (soft coarse coherence, hard edge signatures;
`LAM8` in the script).  With `lam8` finite alone the windows recombine
freely and the copies differ (mean pairwise Jaccard distance of the
aligned tree masks 0.75 at lam8 2, 0.89 at 0.5, against 0.03 verbatim),
every guarantee holding, but the edge signatures match a trunk window's
top to its own bottom, so trunks tile themselves to the sky and trunk
segments land above branches.  The missing constraint is height: a window
keeps its exemplar row (`align`: a hard pair between the window's ey and a
painted chunk-row channel, ey = 8 x the chunk row), which forbids a trunk
anywhere but on the ground and a crown anywhere but at its height, and
leaves recombination free sideways.  With it, one exemplar has little to
recombine at a given height: 1-6 recombined seams in 5 copies, and the
diversity that remains (0.17-0.35) is mostly pruning, crowns cut at seams
whose signatures match nothing else (105-391 cells pruned, 1266-1979
cells against 2342 verbatim).  Recombination needs more material at
each height than one tree offers.

Not learned, copied: the soft terms are the coordinate coupling and
coherence as in the roots second pass; a learned potential on top of the
copy is the same open item as for the limbs.

## What this adds to the design

- The recursion with estimated targets works on a fine level that mixes
  (stage A, AIS within noise of the exact hook); the stretch case of
  `todo.md` group 1 is answered for a chain terrain, not yet for a Potts
  texture.
- The first structured type generates correctly under the certificate on a
  sampled terrain and reacts to it through the hard rules; the soft
  reaction (placement) needs the limb free energy, which neither AIS
  variant can estimate because of single-site growth mixing.
- `AISTargets` repaints after restoring (bug fix); `train._site_record`
  skips a target with no finite candidate; `roots.tables` takes an
  exemplar, and the limb set counts the big root exemplar flipped (the
  small one gives no branching); `actais.ActivityAIS` is kept as the
  estimator to extend with joint moves.
- The tree exemplar copies with every guarantee holding and no sequential
  growth (stage C); the root machinery transfers to any structure that
  hangs from a surface by running it upside down.
- The terrain-first ordering (tree slots at the terrain's level reading it)
  is the shape the tower grammar needs: a type's hard reads of the ground
  stay hard pairs against a same-level channel, and the learned part is the
  finer levels' free energy only.

Images: images/tree_exemplar_{exemplar,coarse,seq}.png, images/trees_{oracle,theta0,hook_K0,hook_K3,ais_K0}.png,
images/trees_tables.png, images/trees_stageA.json, images/treeland_sheet.png,
images/treeland_gen.json.

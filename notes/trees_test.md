# Trees on terrain: the recursion with a sampled fine level, and limbs (2026-10-09)

`castlegen/channels/{trees,treeland,limbs,actais}.py`,
`notes/experiments/{trees_fit,treeland_gen}.py`, `tests/channels/test_{trees,treeland}.py`.
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

### Generation (`treeland_gen.py`, 64 runs, 24 slots, grow 3, dangle 4, 150 tile sweeps)

- Valid: 0 certificate violations in 64 runs, 18749 limb cells all
  attached to a trunk (fraction 1.0000), 0 multi-parent cells; 0.6
  dangling ports per tree (soft).  1.4 s per run.
- 18.9 limb cells per tree (sd of the mean 0.2); presence 0.646 (the
  designed value: no learned terms).  Mass histogram roughly 45 / 25 / 30 %
  for 1 / 2 / 3; degree histogram 3 % tips, 97 % chains, under 1 %
  branch points: the limbs are wandering chains thickest at the trunk
  (images/treeland_sheet.png), the same look the counted root tables gave
  before the exemplar-coordinates pass, and a knob scan (grow 1..3, dangle
  2..4, 60..150 sweeps) moves size but not branching.
- The trees react to the terrain through the hard rules alone: trees whose
  slot has a relief of 2 or more rows grow 17.5 limb cells against 19.2 on
  flatter ground (n 170 vs 510); with the ground one row higher on one
  side, the limbs sit 1.5 cells further from it (n 218, sd of the mean
  0.45), the reverse slope 0.4 (n 220); at slopes of 2 the counts are too
  small (n 52-74) and change sign.  Presence does not depend on the relief
  (0.64 / 0.65 / 0.69 at relief 0 / 1 / 2+, within noise), as designed:
  the forward has no term for the limbs' free energy.

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
  exemplar; `actais.ActivityAIS` is kept as the estimator to extend with
  joint moves.
- The terrain-first ordering (tree slots at the terrain's level reading it)
  is the shape the tower grammar needs: a type's hard reads of the ground
  stay hard pairs against a same-level channel, and the learned part is the
  finer levels' free energy only.

Images: images/trees_{oracle,theta0,hook_K0,hook_K3,ais_K0}.png,
images/trees_tables.png, images/trees_stageA.json, images/treeland_sheet.png,
images/treeland_gen.json.

# Handoff (2026-10-09, evening)

For whoever picks the repository up next, including a fresh session.
Where things stand and what to do first.  `todo.md` is the full list of
future work; this note is the state.

## Since the morning handoff

- The blockgram work was committed and merged with the reorganisation
  (`main` = `origin/main`; the stale worktrees under `.claude/worktrees/`
  can be removed with `git worktree remove --force` + `prune`).
- `trees_test.md` (the trees on terrain test): stage A validates the
  recursion with AIS targets over a sampled chain terrain (within noise of
  the exact hook; 2-3x eval noise end to end).  Stage B generates the
  first structured type (limbs under the certificate, `limbs.py`,
  `treeland.py`) on a sampled terrain, valid and reacting to the terrain
  through the hard rules; its placement cannot be learned because the
  limb free energy is not estimable by AIS (`actais.py` kept; the fix is
  joint growth moves or limbs as records, `todo.md` groups 1, 2, 4).
- `AISTargets.log_z` now repaints after restoring the site (it left the
  last candidate painted); `train._site_record` skips targets with no
  finite candidate; `roots.tables(rows=...)`.  144 tests pass.

## State of `main`

`main` = `origin/main` plus the commits of 2026-10-09 (blockgram merged,
trees test):

- `castlegen/channels/` is the current generic implementation: channels,
  factors, certificates, per-channel samplers with admitted lists and
  dormancy, the bootstrap trainer (`train.fit`, `AISTargets`,
  `ExactTargets`, `actais.ActivityAIS`), the biome circles toy, the trees
  test (`trees.py`, `treeland.py`, `limbs.py`).  144 tests pass:
  `NUMBA_NUM_THREADS=1 python -m pytest tests/channels -q`.
- `castlegen/blockgram/` is the compiled split grammar (`blockgram.md`);
  executing its plan through the channels layer is `todo.md` group 4.
- Everything earlier is under `castlegen/legacy/` with a status line per
  module and `tests/legacy/` (10 known failures xfail by name).
- The design is in `dsl_updates.md`, `dsl_interface.md`; the test notes in
  the order of `notes/README.md`; `reference_math.tex` is not yet updated
  for the bootstrap trainer, collapsed targets, admitted lists, periodic
  worlds.

## The questions in `reference_math_andrew_notes.txt`

Seven questions on the math note.  The design answers some already;
the rest are open items in `todo.md`:

- Pair energies over arbitrary pairs vs neighbourhoods: the DSL's pair
  factors are over a fixed offset set per factor (the stencil), so a
  neighbourhood by construction; `dsl_updates.md` C2.  The note's sum
  over p with a fixed offset is that.
- Forbid parent reads, writes only: that is the direction rule
  (`dsl_updates.md` C3 to-do, `todo.md` group 6): a channel reads levels
  >= its own only; finer levels reach it through the learned potential.
- "Aren't all factors honour": agreed; C3 drops honour as a category
  (a hard parent factor) and defines honourable inline.  `todo.md` group 6.
- Certificates as "the one computed factor", and why factors are table
  entries: a certificate is a computed predicate (connectivity, support)
  that a table cannot express; formula factors among latents (C4) are
  the generalisation and are specified, not built.  `todo.md` group 3.
- Painted channels as compression of an additive bias field, with
  explicit rules for which channels in which blocks communicate and in
  which direction: deferred in `dsl_updates.md` ("Deferred bias field")
  until several channel sets write into one tile level; `todo.md`
  group 5.  The comment added to `reference_math.tex` says the same.
- Promotion: yes, the term for decoupling cheaply updated hard variables
  (an edge signature promoted a level up) from expensively updated soft
  ones below; research question 2 in `channels_system.tex`, `todo.md`
  group 3.

## What to do first

In the order of `todo.md`:

1. The stretch case: objects over a Potts-coupled tile level, AIS targets
   only, judged end to end on a periodic world.  First test of the
   recursion with estimated targets.  Script to start from:
   `notes/experiments/circles_biome_fit.py`; model in
   `castlegen/channels/circles_biome.py` (a Potts `J` among the tiles
   goes into `designed_factors` at the tile level; `support_factors`
   stays as it is, it is the object level's hard pairs).
2. The first real type through the C5 plumbing (slot channel per level,
   biome x family mask, variants as promotions).
3. Executing the compiled tower plan through the channels layer
   (`todo.md` group 4): record-valued channels, collapsed heads,
   projections as certificates, lifted reads, stateless leaves.  The
   compiler's output is `Compiled.table()` in `castlegen/blockgram/compile.py`;
   the negative result to beat is in `channels_demo.md` (2026-10-06).

## How this session worked

Up to five agents on disjoint files in one worktree, no git from the
agents, tests and commits by the coordinator per package, stage notes
folded into `circles_biome_test.md`.  Single core throughout
(`NUMBA_NUM_THREADS=1`).  Figures go to `images/` (gitignored).

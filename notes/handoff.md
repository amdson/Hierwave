# Handoff (2026-10-09)

For whoever picks the repository up next, including a fresh session.
Where things stand, what is not yet in git, how to bring it in, and
what to do first.  `todo.md` is the full list of future work; this note
is the state.

## State of `main`

`origin/main` at 990c13e (PR #2) plus `todo.md` and this note (PR #3):

- `castlegen/channels/` is the current generic implementation: channels,
  factors, certificates, per-channel samplers with admitted lists and
  dormancy, the bootstrap trainer (`train.fit`, `AISTargets`,
  `ExactTargets`), the biome circles toy that validates it, and
  `embed_fit.py` for the feature-set comparison.  132 tests pass:
  `NUMBA_NUM_THREADS=1 python -m pytest tests/channels -q`.
- Everything earlier is under `castlegen/legacy/` with a status line per
  module and `tests/legacy/` (10 known failures xfail by name).
- The design is in `dsl_updates.md` (decisions and measurements),
  `dsl_interface.md` (interface and the training algorithm),
  `circles_biome_test.md` (the test problem and every stage's result),
  `reference_math.tex` (mathematics; not yet updated for the bootstrap
  trainer, collapsed targets, admitted lists, periodic worlds).
- What the build established: support is computed from the writes, the
  finite part is learned; stamp features are exact including for a
  never-seen family; targets must integrate out everything drawn after
  the site (collapsed targets), and with them the number of p* steps
  buys nothing; the end-to-end chain matches the oracle within eval noise
  on a periodic world; per-site cost is flat in the library size with
  admitted lists; parameters are flat (68 against 412k tabular).

## Not in git: the main checkout at `/Users/amdson/dev/Hierwave`

That checkout is at 0b6db4d (the commit before the proofread PR, on the
old layout) and holds work that is on no branch:

Untracked, all of it the block-compiled grammar and the tower stress test:

    castlegen/blockgram/{__init__,core,towers,tower_spec,compile}.py
    castlegen/channels/tower.py
    notes/blockgram.md
    notes/experiments/{blockgram_towers,blockgram_compile,tower_windows}.py
    notes/reference_math_andrew_notes.txt      (questions on the math note)

Modified against 0b6db4d:

    notes/channels_demo.md     appends "Towers of counted rooms: a negative result"
    notes/reference_math.tex   one added comment: adopt the bias field as the
                               DSL's communication primitive
    notes/reference_math.pdf   rebuilt (tracked; upstream rebuilt it too)

Every import in the untracked files resolves on the new layout
(`castlegen.blockgram` is new; `castlegen.channels` did not move).
`castlegen.egg-info/` is a build artefact; ignore it.

### Bringing it in

`git pull` will refuse as it stands, because the reorganisation also
edited `channels_demo.md` (six citations to `history/channels.tex`) and
the proofread edited `reference_math.tex`.  Commit first, then merge:

    cd /Users/amdson/dev/Hierwave
    git add castlegen/blockgram castlegen/channels/tower.py notes/blockgram.md \
            notes/experiments/blockgram_towers.py notes/experiments/blockgram_compile.py \
            notes/experiments/tower_windows.py notes/reference_math_andrew_notes.txt \
            notes/channels_demo.md notes/reference_math.tex
    git commit -m "blockgram: split grammars compiled to a block pyramid; tower grammar and its compiler; towers of counted rooms (negative result)"
    git pull --no-rebase            # merge origin/main

Expected conflicts: `channels_demo.md` (keep both: the upstream citation
edits and the appended tower section), `reference_math.tex` (keep both:
the proofread and the one added comment), and `reference_math.pdf`
(binary; take either, then rebuild it from the merged `.tex`).
Then `git push`, and the worktrees under `.claude/worktrees/` can go
(`git worktree prune` after removing the directories).

If the merge is unwelcome, the equivalent is a branch off `origin/main`
with the same files copied in; nothing in the list collides with a move.

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

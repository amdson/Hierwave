# Todo: every future addition under consideration

Collected 2026-10-09 from `dsl_updates.md` (Open, Deferred, C3-C5),
`dsl_interface.md`, `circles_biome_test.md` (stretch, stage 7b/8/4c/5b
remainders), `channels_system.tex` (research questions), `reading.md`,
`history/todo.txt` (the items the new design did not already settle),
and the memory notes.  Grouped by what they are; within a group, the
order I would do them.  Each item names its source.

## 1. Validation still owed on the toys

- [ ] **Objects over a textured fine level** (the stretch case; Q4).
      Footprint tiles coupled by a Potts `J`, so no exact collapsed
      target exists: `AISTargets` only, judged end to end against a
      long oracle chain on a periodic world.  This is the first test of
      the recursion where the AIS targets are not exact.
      `circles_biome_test.md` stretch, `dsl_updates.md` Q4.
- [ ] **Adaptive features / stencil growth.**  Grow the feature set or
      offsets where the residual or the consistency violation is
      significant; never tested, every fit so far used a fixed OFF8
      stencil.  Needed where induced couplings are not nearest-neighbour
      (the circles top level had second-neighbour terms).
      `dsl_updates.md` C2 "Adaptive features", `reading.md`.
- [ ] **Potts kappa 1 residual.**  The collapsed top fit reached 0.85 of
      the oracle's contrast; attributed to the learned mid tables below
      but not tested separately.  `circles_biome_test.md` stage 6b.
- [ ] **Learned embeddings with unvisited rows pinned**, and mixed
      features with the pin, as the recipe for latents with no painter.
      Stage 4c showed the leak; the fix is described, not run.
      `dsl_updates.md` Q1.
- [ ] **Window regression (`fitWindows`) as the pair-structure
      diagnostic** was never run on the biome model.  `circles_biome_test.md`
      stage 4 "Not run".
- [ ] **Top-level AIS and K = 3 on the periodic world** were not rerun
      after the torus fix (only exact-hook K = 0).  Stage 5b.

## 2. Samplers and performance

- [ ] **Bit tile sampler behind `ChannelSampler`** (`bitgibbs.c`): the
      tile level is 85-95% of a forward run.  Numba kernel stays the
      reference until the bit sampler is trusted.  `dsl_updates.md` Q6,
      memory `numba-kernel-first`, `history/todo.txt` LATER.
- [ ] **Support derived from footprint overlap per family pair**, not a
      `D x D` table per value pair: the remaining cost that grows with
      the library is table memory (13 MB at 321 values; a cache effect
      at ~1 us/site).  `dsl_updates.md` Q5, stage 7b.
- [ ] **Certificates with a candidate cap**: the certificate draw
      enumerates the whole domain; `K` is asserted off there.  Stage 8.
- [ ] **Tempered kernel and `unary_logZ` for coordinate channels**, so
      AIS targets can run over an exemplar-coordinate child.  Stage 8.
- [ ] **Dormancy for coordinate channels** if their parents' writes ever
      become hard.  Stage 8.
- [ ] **Hard-rows-first as the default compile path** (currently opt-in
      through `Sampler`; existing callers keep the old summation order).
      Package A deviation 1.
- [ ] **Re-`init()` rule**: admitted lists and dormancy are valid until
      the parents change; a sampler that is reused across parent changes
      must call `init()` again.  Make the schedule do it, or assert.
      Stage 7b docstring.
- [ ] **Block-order random stream**: with `hb > 1` the visit order inside
      a colour class is block by block, so the stream differs from the
      raster kernel.  Decide whether bit-identity across `hb` matters.
      Package A deviation 4.
- [ ] GPU after the bit sampler.  `history/todo.txt` LATER.

## 3. The library and the real channel sets (C4, C5)

- [ ] **The first real type** (C5): slot channel per level, biome x
      family mask one level up, type on the level matching its
      footprint, variant as a promotion when the variant space is large.
      None of the C5 plumbing exists outside the toy.  `dsl_updates.md`
      Q3, C5.
- [ ] **A hierarchical type with sub-structures and a connectivity rule**
      (the library building): a 32-block footprint writing a mask into
      8-block slots, sub-structures as slot families, reachability as a
      certificate or formula factor.  The honest measurement of the
      marginal cost of a type.
- [ ] **Formula factors among latents** (C4: distance, overlap,
      containment from footprints; `FormulaRow`) are specified, not
      implemented.
- [ ] **Coarse certificate** (C4: support or connectivity at block
      resolution for effects the fine level induces non-locally).
      Specified, not implemented.
- [ ] **Biome composition from a corpus**: the count factor biome value x
      family counts exists in the DSL; the estimator has not been run on
      it.  C5.
- [ ] **Types without a fixed footprint** (rivers, roads) as channels with
      a certificate, dormant where their family is masked out.  C5.
- [ ] **Exemplar coordinate features** (Q2): is the masked exemplar patch
      the whole feature, or does it need learned dimensions for texture.
- [ ] **Third channel set on ground + roots with zero hand tuning**, the
      original research question 1 (calibration between interacting
      channel sets).  The bootstrap trainer is the intended answer; it
      has not been run on ground + roots.  `channels_system.tex`,
      `history/todo.txt` NOW/NEXT.
- [ ] **Learned coarse tables for ground + roots** by the bootstrap
      (replacing `learn_knobs.py`'s moment-matched knobs): per-window
      unary and soft pair tables over edge signatures.  `history/todo.txt`
      NOW.
- [ ] **Promotion as a promise channel** (research question 2: edge
      signature promoted, smoothed `-log p(v|u)` above, hard honour
      below; measure stuck chunks and seam quality).  `channels_system.tex`.
- [ ] **Joint blocks rule** (research question 3: which variables a
      kernel must draw jointly, inferred from the factor graph under a
      product-domain cap).  Stage 8 found the joint (u, t, d) coordinate
      kernel is one sampler owning three channels; the rule is still by
      hand.
- [ ] **Diversity sheet** (research question 4: several painted systems;
      distinct seams vs validity).
- [ ] **Chunk-by-chunk generation with halos** against the global run
      (research question 5; constitution 6-7).  Untested in the channels
      layer.

## 4. Training and features

- [ ] **Multi-site windows and the one-site window as settings of one
      routine** (`fitWindows` folded into `Trainer.fit`).  `dsl_interface.md`.
- [ ] **Post-relaxation rule enforced**: never relax with a constraint
      softer than the unary gap, and only after the unaries are learned;
      currently a knob with a documented footgun.  C1.
- [ ] **Requests versus fitted parameters** (beta vs theta: a requested
      statistic as a multiplier, chain tables fitted to the joint's own
      moments so untargeted features don't drift).  `channels_system.tex`
      "What requesting a statistic means", `history/todo.txt`.
- [ ] **The bias field as a communication primitive** (deferred until
      several channel sets write into one tile level): field per channel,
      additive writes, response functional, inbox typing.
      `dsl_updates.md` Deferred.
- [ ] **Per-factor energy breakdown and ablation in the compiler.**
      `history/todo.txt` NEXT.
- [ ] **AIS standard error under-reports at the default K, M** (delta
      method from 16 chains); widen or use more chains when a target's
      error matters.  Package D deviation 5.

## 5. Documentation

- [ ] **`reference_math.tex`**: drop *honour* as a category (a hard
      parent factor; define "honourable" inline), state the direction
      rule once (a channel reads levels `>= l` only; finer levels reach
      it through `Phi_l`).  C3 "To do".
- [ ] **`reference_math.tex`** does not yet describe the bootstrap
      trainer, collapsed targets, the admitted list or the periodic-world
      rule; `dsl_updates.md` does.  Fold in once the stretch case has
      run.
- [ ] **`reference.md`**: still the pre-build compact reference apart
      from the code map.
- [ ] History notes cite each other by bare filenames as if from
      `notes/`; harmless.

## 6. Infrastructure and legacy

- [ ] `castlegen/legacy/pipeline.py:219` uses an undefined name (`an`);
      three legacy tests xfail on it.
- [ ] `cache/castle_ex.npy` is not in the repo; two legacy tests xfail.
- [ ] JAX remains a dependency of the legacy JAX-era modules and of
      `channels/convfit.py` (lazy); decide whether to drop convfit's JAX
      path now that `embed_fit.py` has numpy fits.
- [ ] `notebooks/e2e_colab.ipynb` updated to legacy imports, not executed.
- [ ] `chi_quick.py` (legacy) reads `sys.argv` at import time.
- [ ] Images are gitignored; the test notes reference figures that only
      exist where the scripts were run.  Decide whether to commit a
      curated set.

## 7. 3D

- [ ] **Luanti backend**: own minimal game, `singlenode` mapgen, a trusted
      mod calling the C sampler through LuaJIT FFI, `chunksize = 4`;
      `.vox` export for inspection before that.  Memory `luanti-for-3d`,
      `history/todo.txt` LATER.  Not before the stretch case and the
      first real type.

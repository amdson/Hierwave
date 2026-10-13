# cconv (ideas #5): built and tested, stopped before benches

2026-10-13 · `castlegen/variants/cconv.py`, `_cconv_core.py`, `tests/variants/test_cconv.py` (15 pass on the box). The agent's text, condensed by the orchestrator.

No bench numbers yet.

## Built

- **Trunk, one per level.**
  - Inputs: the level's channels (zeroed where unknown) with known and exists flags, the parent planes upsampled 2×2, position in parent, and the row fraction.
  - Layers: a 3×3 stem, then r − 1 residual blocks `h += conv3x3(SiLU(LN(h)))`. Off-grid cells are zero-padded, and x wraps on periodic strips.
  - Precision is fp32 `HIGHEST`, because TF32 made the pass and the loss disagree on log q by 1e-3.
- **Heads:** a full-domain linear head plus a candidate-scoring head, which reads the site features and, for each candidate, its embedding, s(t), min(nv, 3) and min(look, 3). Both start at zero.
- **Training pass:** cfast's, with the trunk recomputed densely at every step. The loss re-runs the trunk on (2r+1)² patches at the subsampled sites.
- **Cone and margin** use the conv reach.
- **Defaults:** B = 64, r = 4, width 32, lr 1e-3.

## Tests

- The dense trunk equals the patch trunk.
- The loss recomputes the pass's log q, and scoring matches the pass.
- Every change caused by replacing one draw lies inside its cone, on all 5 problems.
- Overlapping open windows agree (toy_support, ground).

## Next

1. Profile seconds per update at B = 64 on ground; compiles are slow (about 50 s for generation).
2. Bench toy_support (the gate), dimers, ground and columns against cfast on the box, under `taskset`.
3. Annotate ideas #5.

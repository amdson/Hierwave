# p10: P10, the user's pipeline (pass with certificates, then delete-and-resample below d) (2026-10-15)

Agent p10. GPU 1 (shared with p5's light MCMC jobs), cores 8-15. Diagnostic only: no training, no shared code touched. Objective of the saved models: cr2's `ckpop_hard_pen` / `ckpop_pen16` (kernel-route ML + penalty-only reverse KL, see cr2_penalty). ideas #22, cmix proposals P10.

## Question
After the coloured pass, delete everything below level d, resample the terrain below d with a **fixed** number of sweeps under the clamped levels >= d, then refill the certificates below d with a **fixed** number of graded certificate-only sweeps. Does this take a trained own-draws model under the 2e-4 budget at every level with the terrain statistics kept?

## Answer
**Only in variant 2 (d = T+1: certificates are planning-only, every level is resampled).**
- `v1b d5 R60 F80` gave **64/64 valid worlds and 0 violations at every level for both models**. This arm frees every sampled level, including the top rho4. At each of the R rounds it runs one single-cell graded sweep of all levels, one level-1 parent+4-children block sweep and one 2x2 tile block sweep.
- Tiles changed: 3.1 % (pen16) and 4.7 % (hard_pen).
- Cost: about 27 ms per world on the A4000, against about 1.3 s per world for the pass draw.
- **Variant 1 (d <= 4, top certificates clamped) fails at fixed budget.** At 32 worlds, one world per model keeps a coarse honour/support miss at L2-L4 under every (d, R, F) tried, up to R = 100 and including the block kernels.
- **The literal pipeline fails outright.** Deleting the certificates below d, then resampling the terrain with honour_d dropped and refilling, makes the refill infeasible against the kept level-d certificates: 0/16 worlds valid.
- **Strict variant 2 (rho4 clamped, cert4 refilled) is borderline.** hard_pen: 63/64 valid (one L1 refill miss, under budget). pen16: 63/64 valid with one L2 instance (2.4e-4), **over budget by one instance**.

## What was run
Script `scripts/p10.py <label> <variant|greedy> <params> <n> <out.json> <arms>` (arms `kind:d:R:F[:del]`). Drivers: `scripts/run32.sh`, `runblk.sh`, `runv2.sh`, `runv2_64.sh`, `runv2r_64.sh`. Logs and JSON are in `runs/`.

**Setup.**
- Pass draw = the `raw` (unrepaired) output of `v.generate` on held-out eval worlds EVAL_SEED0+i, W = 128. The loader is graded/diag1's.
- All sweeps use colour_gibbs at tau = 1 with the λ = 20 graded penalty (cgraded `graded_problem`). The block kernels are p5's (binary λ = 20).

**Arms.**
- `v1` (variant 1, "certificates derived"):
  - Stage 1: R single-cell sweeps of levels d-1..0 on all channels (rho, cert, t) of full ground, starting from the pass draw. Because cert_{d-1} is carried, row honour of level d stays a hard factor.
  - Stage 2: the certificates below d are reset to uniform random and refilled by F certificate-only sweeps.
- `v1b`: as `v1`, but each round also runs `JaxBlockGibbs(level 1, depth 1)` and `FineBlockKernel(k = 2)`, following p5's message. A level-2 block kernel did not build within 110 s, so it is not used.
- `v2`: `v1b` at d = 5 with rho4 masked (clamped).
- `plain` (the literal pipeline): stage 1 runs on ground with the certificates below d removed, together with every factor that reads them (honour_d dropped; support0 kept). Stage 2 is the same refill.
- `:1` (DEL): the terrain below d is also reset to random.

**Counting.** Violations are hard instances per level on full ground (host `energy`).
- 16 worlds: 1 instance = 6.1e-5 / 2.4e-4 / 9.8e-4 / 3.9e-3 at L0 / L1 / L2 / L3.
- 32 worlds: half of that. 64 worlds: a quarter of it.
- So any L2+ violation is over budget at every n here. Zero at L3/L4 cannot show 2e-4: there are only 1024 / 128 instances at 64 worlds.

## Numbers
Final violations per level [L0..L4] (sum over worlds), worlds valid, tiles changed vs the pass draw, line_match / roughness / floating / dirt. The probe reference (64 worlds): line_match 0.882, roughness 0.20, floating 0, dirt 0.507.

| model | arm | n | viol L0..L4 | valid | tiles chg | lm | rough | float | dirt | fill p50/max |
|---|---|---|---|---|---|---|---|---|---|---|
| hard_pen | pass raw | 64 | 20,20,12,1,0 | 41 | - | 0.558 | 1.01 | 1.41 | 0.437 | - |
| hard_pen | pass final (repair) | 64 | 16,12,10,1,0 | 47 | - | 0.558 | 1.01 | 1.36 | 0.437 | - |
| hard_pen | v1 d2 R30 F80 | 16 | 2,1,4,0,0 | 14 | 0.7 % | 0.555 | 1.08 | 1.50 | 0.433 | 3/5 |
| hard_pen | v1 d3 R30 F80 | 16 | 2,1,2,0,0 | 15 | 0.9 % | 0.557 | 1.06 | 1.50 | 0.435 | 8/24 |
| hard_pen | plain d2 R30 F80 | 16 | 532,1186,4,0,0 | **0** | 4.6 % | 0.562 | 1.91 | 0 | 0.439 | none |
| hard_pen | plain d3 R30 F80 | 16 | 222,250,463,0,0 | **0** | 6.5 % | 0.592 | 1.73 | 0 | 0.468 | none |
| hard_pen | v1 d2 R30 F80 DEL | 16 | 385,951,4,0,0 | 0 | 2.3 % | 0.558 | 2.69 | 112 | 0.434 | none |
| hard_pen | v1 d4 R100 F80 | 32 | 2,1,2,1,0 | 30 | 5.2 % | 0.611 | 1.58 | 0.31 | 0.483 | 17.5/40 |
| hard_pen | v1b d4 R100 F80 | 32 | 0,0,0,1,0 | 31 | 7.1 % | 0.658 | 1.51 | 0 | 0.500 | 15/60 |
| hard_pen | v2 (rho4 clamped) R60 F80 | 64 | 0,1,0,0,0 | 63 | 5.1 % | 0.609 | 1.91 | 0 | 0.475 | 23/53 |
| hard_pen | **v1b d5 R30 F80** | 64 | **0,0,0,0,0** | **64** | 3.3 % | 0.577 | 1.79 | 0 | 0.456 | 20/37 |
| hard_pen | **v1b d5 R60 F80** | 64 | **0,0,0,0,0** | **64** | 4.7 % | 0.601 | 1.82 | 0 | 0.470 | 20.5/53 |
| hard_pen | v1 d5 R100 F80 | 64 | 1,0,0,0,0 (under) | 63 | 3.7 % | 0.585 | 1.77 | 0.03 | 0.460 | 21/42 |
| pen16 | pass raw | 64 | 92,46,23,3,0 | 20 | - | 0.629 | 1.41 | 5.36 | 0.498 | - |
| pen16 | pass final (repair) | 64 | 70,15,22,3,0 | 22 | - | 0.629 | 1.42 | 4.08 | 0.498 | - |
| pen16 | v1 d4 R100 F80 | 16 | 2,0,0,0,0 (under) | 14 | 3.0 % | 0.649 | 1.44 | 0.38 | 0.505 | 18.5/36 |
| pen16 | v1 d4 R100 F80 | 32 | 8,2,3,2,0 | 28 | 2.8 % | 0.661 | 1.41 | 1.56 | 0.505 | 16.5/46 |
| pen16 | v1b d4 R100 F80 | 32 | 2,1,3,1,0 | 31 | 3.3 % | 0.661 | 1.43 | 0.50 | 0.507 | 15/51 |
| pen16 | plain d3 R30 F80 | 16 | 135,133,156,0,0 | **0** | 3.4 % | 0.634 | 1.36 | 0 | 0.494 | none |
| pen16 | v2 (rho4 clamped) R60 F80 | 64 | 2,1,**1**,0,0 | 63 | 3.2 % | 0.662 | 1.45 | 0.11 | 0.504 | 22/50 |
| pen16 | v1b d5 R30 F80 | 64 | 2,1,**1**,0,0 | 63 | 2.8 % | 0.649 | 1.49 | 0.11 | 0.503 | 22/48 |
| pen16 | **v1b d5 R60 F80** | 64 | **0,0,0,0,0** | **64** | 3.1 % | 0.656 | 1.45 | 0 | 0.503 | 22/57 |
| pen16 | v1 d5 R100 F80 | 64 | 1,2,0,0,0 (under) | 63 | 2.8 % | 0.654 | 1.44 | 0.03 | 0.502 | 22/50 |
| greedy | v1 d4 R300 F80 | 16 | 1119,458,126,129,0 | 0 | 32 % | 0.516 | 6.59 | 678 | 0.775 | none |

Full logs: `runs/{hard_pen,hard_pen2,pen16,pen16_32,hard_pen_32,hard_pen_blk,pen16_blk,hard_pen_v2,pen16_v2,hard_pen_v2_64,pen16_v2_64,hard_pen_v2r_64,pen16_v2r_64,greedy_16}.log`. Each JSON lists the violating worlds with their per-factor counts.

## Reading
- **Refill feasibility is the crux, and it is all-or-nothing.**
  - When the certificates below d are carried through the terrain resample (v1), the refill reaches validity whenever stage 1 ended valid: median 2-22 sweeps, max 77 (under F = 80). F = 40 missed once in 64.
  - When they are dropped (plain), the resampled terrain stops matching the kept level-d promises and the refill is infeasible in every world. This is the proposal's "watch out", confirmed.
- **Residual failures are single-world coarse traps.** Every variant-1 miss is one world per 32 with honourB3/honourA4 plus support2/3 instances: a clamped coarse certificate that the children cannot honour by single-cell moves (hard_pen world 14, pen16 world 26). Freeing the level that holds the bad promise (d = T+1) clears them. Freeing levels 0-1 with block kernels clears the fine misses (L0 floating tiles), not the coarse ones.
- **Structure.**
  - pen16 is kept or improved: line_match 0.629 -> 0.656, roughness 1.41 -> 1.45, floating 5.4 -> 0, dirt 0.498 -> 0.503 (probe 0.507).
  - hard_pen improves on line_match (0.558 -> 0.601) and dirt (0.437 -> 0.470), but **roughness rises 1.01 -> 1.82** (probe 0.20). The tau = 1 MCMC moves the rugged surface toward the chain's own (rougher) equilibrium. That is a change, not a repair.
  - Terrain changes 3-5 % of tiles. The sampler is not the problem for these two models.
  - The untrained greedy pass is hopeless as a start: 0/16 valid after 300 rounds, 32 % of tiles changed.
- **Cost of the winning arm.**
  - Per round: 1 single-cell sweep of all 5 levels + 1 level-1 block sweep + 1 tile block sweep. Then F = 80 certificate-only sweeps of levels 4..1.
  - Wall clock: about 1.0 s + 0.7 s per 64 worlds warm (shared GPU), about 27 ms per world. Compile is 17-46 s once.
  - FineBlockKernel picks its offset with host rng. Generation would need a fixed offset cycle.
- **Finite support (stated, not tested).** All stages are fixed-count local steps, so constraint 2 holds in principle. The added dependency radius is bounded by (R + F) × per-sweep reach.
  - Per-sweep reach is at most 15 cells per level per sweep (16 colours updated in sequence), so at most 15 × (16+8+4+2+1) = 465 fine columns per round of all levels, plus the block kernels' 2-cell reach.
  - For R = 60, F = 80 the loose bound is about 6e4 fine columns: finite and constant, but far wider than the W = 128 strip. The effective radius is unmeasured.
- **Constitution.**
  - The stage is a fixed-budget repair (goal 4, constraint 2).
  - Variant 2 discards the pass's coarse certificates, which is the behaviour change the user accepted (ideas #22 decision).
  - Freeing rho4 as well (d = T+1) means the pass draw is only an initialisation of a fixed-length MCMC run, not a plan that is kept. Whether that still counts as "the coloured pass" is the user's call.
  - The per-level budget is a statement about 64 worlds. Zero at L2-L4 needs more worlds to confirm.

## What failed
- Variant 1 at every d <= 4 (one stuck world per model per 32).
- The literal (honour dropped) pipeline: 0/16.
- Deleting the terrain below d at R = 30: 0/16.
- Building the level-2 block kernel: more than 110 s.
- Strict variant 2 on pen16: one L2 instance over.
- hard_pen's roughness is not kept.
- The greedy arm.
- Each arm is one noise seed. The 16-world pen16 v1 d4 "under budget" flipped to over at 32 worlds, so the 64-world zeros are not robust evidence.

## Next (one line each)
- Rerun `v1b d5 R60 F80` on 256 worlds (and W = 512) with 2 noise seeds to put a real bound on L2-L4.
- Measure the effective dependency radius by perturbing the noise outside a window (the finite-support test).
- Add a parent+children block at level 2-3 (pruned components) to see whether d <= 4 can be rescued without freeing the top.
- Check whether hard_pen's roughness rise is the target's (long chain) or the stage's.

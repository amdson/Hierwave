# offset: where the pass draw's +4.6 row surface offset comes from (2026-10-16)

Agent offset (wave 3), GPU 0 shared (fitpen priority; SM clock and throttle flags in runs/nvidia_*.txt), cores 4-5.
Script `scripts/offset.py`; runs: hard_pen params, 96 eval worlds (seeds EVAL_SEED0+i), W 128; pen16 params, 64 worlds.
Command: `taskset -c 4-5 python scripts/offset.py hard reportables/2026-10-14_cr2_penalty/runs/arm_ckpop_hard_pen_s0/params.pkl 96 runs/hard.json`.

Measure: implied surface at level l = H - (sum over Y of rho_l)/2^l per level-l column (rho_0 = t), minus the designed line
(from fill, strip-averaged). Offset = mean over columns and worlds, convention of stage (positive = surface LOWER = LESS dirt;
the draws have dirt 0.436 vs 0.508 for probes, so the premise "too much dirt" has the wrong sign). "fd" = first-dirt row offset of t.
Clamp = the model's own fine levels with all channels at levels >= L replaced by a probe's lift (same noise as the raw draw;
unclamped run reproduces raw exactly, 100% of tiles). SE of offsets 0.04-0.19.

| arm | rho4 | rho3 | rho2 | rho1 | t (mass) | fd offset | rough | abs c1 | dirt |
|---|---|---|---|---|---|---|---|---|---|
| hard_pen raw | +0.03 | +1.88 | +3.35 | +4.67 | +4.61 | +4.60 | 1.01 | 1.25 | 0.436 |
| hard_pen final (after cpass repair) | +0.03 | +1.88 | +3.35 | +4.67 | +4.61 | +4.60 | 1.01 | 1.25 | 0.436 |
| probe jitter 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.19 | 0.16 | 0.508 |
| probe jitter 2 | -0.01 | -0.01 | -0.01 | -0.01 | -0.01 | -0.01 | 1.41 | 0.19 | 0.508 |
| chain (hard, cmix; first-dirt only) | | | | | | -0.15 | 1.40 | 0.48 | |
| clamp L>=4 (rho4 exact), model draws rho3..t | 0 | +1.59 | +3.35 | +5.16 | +5.65 | +5.64 | 0.59 | 1.50 | 0.420 |
| clamp L>=3 | 0 | 0 | +0.80 | +1.33 | +0.47 | +0.46 | 1.36 | 0.41 | 0.501 |
| clamp L>=2 | 0 | 0 | 0 | +0.67 | +0.05 | +0.05 | 0.94 | 0.26 | 0.508 |
| clamp L>=1 (tiles only drawn) | 0 | 0 | 0 | 0 | -0.09 | -0.09 | 0.54 | 0.17 | 0.510 |
| (jitter-2 probe clamps, L4/L3/L2/L1 | | | | | | +5.15/+0.37/-0.02/-0.16 | 0.71/1.42/1.19/1.31 | 1.32/0.33/0.23/0.19 | |
| pen16 raw (64 worlds) | +0.00 | +0.58 | +0.33 | +1.37 | +0.54 | +0.50 | 1.41 | 0.79 | 0.498 |

## Verdict
- (a) as stated (rho4 carries it): NO. rho4 is +0.03 (it follows fill; the noise4 factor pins it).
- (b) tiles high inside correct blocks: NO. With rho >= 2 clamped, the drawn rho1 and tiles give +0.05 (tiles alone: -0.09, matching the chain's -0.15 / -0.105).
- (c) repair-created: NO. raw and final are identical in every column (repair moves no mass; +4.60 both).
- Source: the **mid-level rho draws rho3|rho4, rho2|rho3, rho1|rho2**. Each draws too little dirt given its parent: increments +1.6..1.9, +1.5..1.7, +1.3..1.8
  (clamp L>=4: 1.59, 3.35, 5.16 cumulative; raw: 1.88, 3.35, 4.67). The bias accumulates down the hierarchy, so it is a "coarse/mid-level
  mass bias" in the spirit of (a) but not at rho4. Clamping rho3 removes ~90% of it (+0.46 left, in rho2/rho1 draws).
  The count prior and smoothness make a parent's children sum to the parent only softly; the learned draws under-fill.
- (d) population bias: NO / mostly not. Training loop.jsonl (arm_ckpop_hard_pen_s0, last iter): kernel population dirt 0.492 vs probes 0.508 (about +0.8 rows if all
  of it were mass), against own draws 0.351 raw (0.436 at eval). The targets are near the line; the sampler fails to fit them.
  No saved population state exists (only summary stats), so the population offset itself is untested.
- Training-dependent: pen16 (different penalty weight) has +0.50 at the same measure with rough 1.41, c1 0.79, so this is a property of the hard_pen run, not of the architecture.
  rho1 is still +1.37 there.

## Failed / not done
- Chain per-level (mass) offsets not measured (cmix trace stores first-dirt heights only); chain compared on fd only.
- One noise draw per world; one params seed per arm.
- First launch crashed (clamped_pass hard-codes ACTOR_WIN; patched in-script to lay.actor_win, checked: unclamped == raw).
- A first pen16 launch hit another agent's half-edited variant file; re-run fine.

## Next
Targeted loss/feature on rho3..rho1 mass given the parent (e.g. a parent-count residual feature, ideas #11, or ML weighting of the child-sum mismatch); check whether pen16's smaller offset came from its penalty weight.

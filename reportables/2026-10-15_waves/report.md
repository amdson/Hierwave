# Waves 2026-10-15 (coordinator log)

User instruction: pull all ideas, run two waves (<=2 Opus, <=4 Sonnet), rule out quickly, look for signal, then focus. Shared brief: `notes/wave_brief_2026-10-15.md`.

## Wave 1 (launched 2026-10-15)
| agent | model | gpu / cores | question |
| --- | --- | --- | --- |
| gpen_p11 | Opus | GPU 0 / 16-23 | training side on ground own draws: graded penalty (ckpop_hard_gpen, addendum c) and P11-1 (drop certificate log q from the rkl mix): does either move line_match off ~0.57 while staying under budget? |
| p10 | Opus | GPU 1 / 8-15 | the user's pipeline P10: pass with certificates, delete-and-resample below d, refill; fixed budget. Does it get a saved own-draws model under budget with structure kept? |
| rgap | Sonnet | CPU / 0-3 | crevise: exact path-KL bound gap, uniform r vs the proposed designed r on the tiny problems. Is the designed r worth building? |
| p5 | Sonnet | GPU 1 (light) / 4-5 | P5: joint parent+children block Gibbs for refinement under clamped coarse levels (the paired-flip deadlock). |
| p6p8 | Sonnet | GPU 0 (light) / 6-7 | P6 (generic derived-channel rule: H(channel | level below)) and P8 (graded vs binary fill tail at W 512/2048). |

Results:
- **p5 (done, yes):** 2x2 tile-block Gibbs (existing FineBlockKernel) refines tiles under a clamped level 1 in ~5 sweeps (31/32 tight, 15/16 chain valid; single-cell 1/32 after 200 sweeps); with level 1 free, parent+4-children blocks (existing JaxBlockGibbs) 32/32 by sweep 10; 1.3-2.5 % tiles changed; 1-4 ms/sweep on 32 worlds. Forwarded to p10 for its resample step. Report: `reportables/2026-10-15_p5/`.
- **rgap (done, no):** on the tiny flat problem the designed r cuts the exact bound gap only to 0.41-0.46 of uniform (3.12 -> 1.27 nats/world at lam_r 2; lam_r 20 = uniform), not below a third; and generators trained by exact gradient under either r went deterministic in 60 s (reverse KL -0.002 / -2.4 vs optimum -6.85; short, one seed). Designed-r crevise is dropped for this session. Report: `reportables/2026-10-15_rgap/`.
- **p6p8 (done):** P8: the graded certificate fill's tail shifts ~+12 sweeps per 4x W (log W, i.e. a per-cell exponential tail of scale ~12 sweeps); F for 2e-4 per world ~125/140/165 at W 128/512/2048 (binary lam 20: ~310; binary lam 8 unusable). So a fixed F ~ 80 is not enough; F ~ 150-200 graded is the working number, or a fill that is valid by construction (P4). P6: on ground the certificate channels' conditional entropy is 12-25x below rho's at every level (0.03/0.05/0.11/0.22 vs 0.56/1.3/2.1/2.7 nats/cell), so an entropy rule separates them; on columns the coarse counts have H = 0 but the single-site chain is non-ergodic (0 cells move), so the rule needs an ergodicity test. Report: `reportables/2026-10-15_p6p8/`.
- **p10 (done, yes with a caveat):** the literal pipeline (drop certificates below d, resample, refill) is infeasible against the kept level-d certificates (0/16) and variant 1 at d <= 4 leaves one coarse trap per 32 worlds; but with every level freed (d = T+1; 60 rounds of graded single-cell + p5's block sweeps, row honour hard, then 80 refill sweeps) both saved models reach 64/64 valid, 0 violations at every level, 3-5 % of tiles changed, line_match 0.56->0.60 / 0.63->0.66, floating -> 0, at ~27 ms/world (pass draw 1.3 s/world). Cost: roughness rises 1.01->1.82 on hard_pen (probe 0.20; pen16 1.41->1.45), so the stage distorts the one statistic the sampler already had wrong. Report: `reportables/2026-10-15_p10/`.
- gpen_p11: pending.

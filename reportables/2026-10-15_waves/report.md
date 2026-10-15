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
- **gpen_p11 (done, no / no signal):** graded penalty `ckpop_hard_gpen` bench 300 s x 2: 1/2 seeds under budget, pooled L2 1.8e-4 (pen 0.76e-4 over 4 seeds), line_match 0.54/0.61, F +0.10/-0.00; raw violations during training are 15-35 % lower but the held-out coarse misses are the ones repair creates, which a raw-draw penalty cannot reach. P11-1 (drop certificate tau log q from the rkl mix, `ckpop_hard_rkl_nocq`): 120 s one seed, line_match 0.580 vs 0.586, certificate slack does not move toward the certificate-free reference (the ML half still fits certificates to kernel states with full multiplicity). Both dropped; P11-2 (ML targets from a certificate-free chain + fill) is the live training-side version. Report: `reportables/2026-10-15_gpen_p11/`.

## Wave 2 (launched while gpen_p11 was still running)
| agent | model | gpu / cores | question |
| --- | --- | --- | --- |
| stage | Opus | GPU 1 / 8-15 | p10's all-levels-free fixed stage, bench-grade: minimum rounds, roughness vs the chain reference (1.39), a bench row as a registered variant's repair (`ckpop_hard_pen_stage`), finite support. |
| p4fill | Sonnet | GPU 1 light / 4-5 | P4: certificate fill from the tight abstraction: does it remove the log-W fill tail, and does slack reach equilibrium? |
| dstage | Sonnet | GPU 0 light / 6-7 | dimers: can a fixed budget of block sweeps restore h/v/d diversity from collapsed cfast draws? |
| kblock | Opus | GPU 0 / 16-23 (after gpen_p11) | p5's block kernels inside ckpop_hard_pen's population refresh (kernel-route amendment): does stronger mixing of the ML targets raise own-draw line_match above 0.59 at the same validity? |
| deq | Sonnet | GPU 1 light / 4-5 | dimers: true equilibrium h/v/d on the bench strip; is the 0.5/0.25/0.25 gate right? |

Results:
- **p4fill (done, yes):** a certificate fill started from the tight abstraction is valid at every k (64/64 at W 512, 32/32 at W 2048, graded and hard), and its slack plateaus in ~5 sweeps (L1-2) / 10-20 sweeps (L3-4) at the same values the random-start fill reaches at k = 80 (.007/.019/.053/.176; the 3x gap to the full chain is the known certificate-free-terrain effect). Random start: 0 % valid through k = 20, 75-100 % at k = 80. So a fill of F ~ 20 from the tight start replaces F ~ 150-200 from random, **if** certificates are declared as abstraction maps of the level below (otherwise constraint 6; today it is ground's `lift`). Report: `reportables/2026-10-15_p4fill/`.
- **dstage (done, partial):** on collapsed cfast dimers draws (h/v/d 0.47/0.53/0.00), 3x3 block sweeps restore the diagonal to 0.15/0.21/0.22/0.23 at k = 10/30/60/120 with violations 5-6e-5; h/v settle at 0.38/0.38, which matches the long chain's own plateau (0.38/0.38/0.23), so either the strip's equilibrium is not 0.5/0.25/0.25 or the chain is slow there; 2x2 blocks conserve the diagonal (0.000). Cost ~120-160 ms/world (shared GPU). Diversity is restored by a fixed stage where training never managed it; the gate value needs checking. Report: `reportables/2026-10-15_dstage/`.
- stage, gpen_p11: pending.

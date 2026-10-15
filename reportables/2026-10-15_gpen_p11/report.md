# gpen_p11: graded penalty (ideas #21) and no certificate entropy (P11-1, ideas #22) on ground

Agent gpen_p11, 2026-10-15, GPU 0 (shared box: GPU 1 was in use by another agent during the runs; GPU 0 SM clock 1560 MHz, no throttle, at each before/after check, `runs/nv_*.txt`), cores 16-23. One GPU process at a time. All runs: own draws only, ground, the cr2 arm script (warm-up instance, then train, then the bench's frozen `eval_seed` on 16 held-out worlds).

## Answers
- **Q1, graded penalty `ckpop_hard_gpen` vs `ckpop_hard_pen`: no.** Line_match does not move (0.540-0.606 vs 0.557-0.585), and L2/L3 misses do not go down on held-out worlds (pooled L2 1.8e-4 vs 0.8e-4; seed 0 misses L3 and L4 in both of its 300 s runs). Training-time raw violation rates are 15-35 % lower at L1-L3, but that does not reach the evaluated output.
- **Q2, drop τ log q for cert1..cert4 in the reverse-KL cost (`ckpop_hard_rkl_nocq`) vs `ckpop_hard_rkl`: no signal at 120 s, 1 seed.** Line_match is unchanged (0.580 vs 0.586). Slack moves only partly towards the scheme's values: tighter A at L3-L4, looser B at L2-L3. The L2 miss is no better (15 vs 6 of 16384). No bench was run.

## Q1: graded penalty

Commands (from `/workspace/Hierwave`):
```
reportables/2026-10-15_gpen_p11/scripts/fast.sh fast120 120 0 ckpop_hard_pen ckpop_hard_gpen      # arm_eval.py of cr2, 120 s, seed 0
reportables/2026-10-15_gpen_p11/scripts/bench_after.sh ckpop_hard_gpen                         # bench_warm.py bench ckpop_hard_gpen 300 2
python reportables/2026-10-14_cr2_penalty/scripts/pool.py <row>                                 # runs/pool_*.txt
```

**Fast runs (120 s, seed 0, back to back).** Final = after repair, 16 eval worlds. Train raw = mean over the last 10 iterations of the reverse-KL steps' own-draw raw rate.

| arm | updates (ML / rkl) | final L0..L4 (counts) | train raw L0..L3 | line_match | own8 lm | dirt | F |
|---|---|---|---|---|---|---|---|
| ckpop_hard_pen | 416 / 1665 | 10, 6, 2, 3, 2 | 7e-5, 2.7e-4, 6.0e-4, 1.11e-3 | 0.542 | 0.558 | 0.429 | 0.175 |
| ckpop_hard_gpen | 440 / 1761 | 9, 8, 6, 6, 2 | 6e-5, 2.5e-4, 5.1e-4, 0.87e-3 | 0.540 | 0.554 | 0.420 | 0.147 |

Earlier 300 s seed-0 arms of the same pair, from other agents' folders and run at different times (pen: `2026-10-14_cr2_penalty/runs/arm_ckpop_hard_pen_s0`; gpen: `2026-10-14_graded/runs/arm_ckpop_hard_gpen_s0`):

| arm | updates | final L2 / L3 / L4 | train raw L1..L3 | line_match | F |
|---|---|---|---|---|---|
| pen | 1136 | 1.8e-4 / 0 / 0 | 3.2e-4, 5.0e-4, 4.7e-4 | 0.552 | 0.038 |
| gpen | 1112 | 0.6e-4 / 2.4e-4 / 1.95e-3 | 1.7e-4, 3.4e-4, 3.7e-4 | 0.548 | 0.057 |

**Bench (300 s x 2 seeds, warm, `results/bench.jsonl` run 20261015-084129_ckpop_hard_gpen_ground_warm), against the 4-seed ckpop_hard_pen row (20261014-193641).**

| | updates per seed | seeds within 2e-4 | pooled final L1 / L2 / L3 / L4 | line_match | F | L3 / L4 starved % |
|---|---|---|---|---|---|---|
| ckpop_hard_gpen (2 seeds) | 1176, 1128 | 1/2 (seed 0: L4 1/512, L3 2/4096, L2 5/16384) | 4.6e-5 / 1.8e-4 / 2.4e-4 / 9.8e-4 | 0.540, 0.606 | 0.104, -0.004 | 28, 24 / 30, 36 |
| ckpop_hard_pen (4 seeds) | 1176-1184 | 3/4 | 5.3e-5 / 7.6e-5 / 6.1e-5 / 0 | 0.557-0.585 | -0.039..0.114 | 21-26 / 23-32 |

Same seeds 0-1 only, pen: L2 1/32768, L3 0, L4 0. Not accepted (probe flag, and an across-seeds dirt_excess mismatch).

**Reading.** The graded mass lowers the training-time raw violation rate at L1-L3 a little. At L0 the graded mass equals the binary count (d = 1), so nothing changes there. On held-out worlds the coarse misses are no better and line_match is unchanged. Seed 0 failed the same way in two independent 300 s gpen runs. Its L3/L4 misses are the repair-created coarse failures of change request #3. The graded penalty is computed on raw draws, so it does not reach them.

## Q2: no τ log q for the certificate channels (P11-1)

**What was built.** `castlegen/variants/cnocq.py` registers `ckpop_hard_rkl_nocq`, a subclass of `ckpop_hard_rkl`. The cfg list `nocq_chans` defaults to cert1..cert4. These channels are chosen by name for this test only; a generic rule for which channels are derived is measured separately (P6).
- **How the cost changes.** The fused reverse-KL update is `cfast.CFast._build_train`, copied at import by source transform (three asserted replacements; shared files untouched). For draws of the listed channels, τ log q(a) is removed from the per-draw cost, so it leaves every return, and from the actor loss's immediate term.
- **What stays.** Those channels are still sampled, and their soft energies and λ·violations stay in the cost. The ML half is unchanged.
- **Test.** `tests/variants/test_cnocq.py` (CPU, 111 s) passes:
  - With an empty list, every output of one fused update on ground (W 64, B 2) is bit-identical to cfast's, which is ckpop_hard_rkl's reverse-KL step. The test runs at that fused-update level, not a whole ckpop iteration.
  - With the certificate list, draws and violations are identical, and the cost drops by exactly Σ log q(a) over the certificate draws.
- **Recorded** as change request #4 in `notes/constitution_change_requests.md`.

Command: `reportables/2026-10-15_gpen_p11/scripts/fast_cert.sh fast120 120 0 ckpop_hard_rkl ckpop_hard_rkl_nocq`. This is the cr2 arm script plus certificate slack on the same 16 eval worlds (`scripts/arm_eval_cert.py`).

| arm | updates (ML / rkl) | final L0..L4 (counts) | train raw L0..L3 | line_match | own8 lm | dirt | F |
|---|---|---|---|---|---|---|---|
| ckpop_hard_rkl | 464 / 1857 | 20, 19, 6, 2, 1 | 2.3e-4, 1.02e-3, 1.73e-3, 1.52e-3 | 0.586 | 0.595 | 0.500 | -0.046 |
| ckpop_hard_rkl_nocq | 464 / 1857 | 12, 12, 15, 1, 2 | 1.8e-4, 0.90e-3, 1.36e-3, 1.37e-3 | 0.580 | 0.591 | 0.480 | -0.062 |

Certificate slack on raw own draws, levels 1-4. The final draws give the same values to two digits.

| | A_tight − A | B − B_tight |
|---|---|---|
| ckpop_hard_rkl | 0.010 / 0.036 / 0.201 / 0.80 | 0.002 / 0.011 / 0.073 / 1.75 |
| ckpop_hard_rkl_nocq | 0.014 / 0.037 / 0.136 / 0.52 | 0.003 / 0.028 / 0.214 / 1.34 |
| cmix ref., full chain (valid states) | 0.011 / 0.045 / 0.151 / 0.52 | |
| cmix ref., scheme (certificate-free + fill) | 0.007 / 0.019 / 0.052 / 0.19 | |

**Reading.**
- Without the certificate entropy term, A tightens at L3-L4 but B loosens at L2-L3. The total slack B − A at L4 is unchanged (3.82 vs 3.82).
- Nothing moves towards the scheme's 3x tighter certificates.
- Line_match and the L2 miss do not improve; the L2 difference (15 vs 6 counts) is within one-seed, 16-world noise.
- Likely reason (not tested): the ML half of the iteration still fits every level's conditionals to kernel-refined states. The kernels keep ground's p invariant, including its certificate multiplicity, so only the reverse-KL half saw the change. Turning the change on for the ML half would mean ML on certificate-free terrain chains plus a fill (P11 candidate 2).

## What failed
- The graded penalty did not reduce held-out coarse misses: a seed-0 L4 miss at 300 s, twice.
- P11-1 had no effect on structure or slack at 120 s.
- No Q2 bench was run, because there was no signal to bench.

## Next (one line each)
- P11 candidate 2 (ML on certificate-free terrain chain plus certificate fill states), so that the ML half also drops certificate multiplicity; then check slack against 0.007/0.019/0.052/0.19.
- Price the repaired output's coarse misses (change request #3); neither penalty variant reaches them.

## Files
- `castlegen/variants/cnocq.py`, `tests/variants/test_cnocq.py`
- `reportables/2026-10-15_gpen_p11/scripts/` (`fast.sh`, `fast_cert.sh`, `arm_eval_cert.py`, `bench_after.sh`)
- `reportables/2026-10-15_gpen_p11/runs/` (logs, `loop.jsonl`/`eval.json` per arm, `pool_*.txt`, `nv_*.txt`)
- Appended: `notes/constitution_change_requests.md` (#4), `reportables/ideas.md` (#21, #22), `reportables/progress.md`

# Box session, wave log (2026-10-13 night → 2026-10-14)

Orchestrator's summary of the multi-agent waves on the GPU box. Each agent's own report is in its dated folder; this page only joins the dots. Numbers are from a shared GPU (3-6 jobs at a time), so wall-clock benches are contended; compare by updates where given.

## Wave 1 (six agents, ~75 min)

| agent | folder | verdict |
| --- | --- | --- |
| exactopt | 2026-10-13_exactopt | **Variance is the main limit at B = 8** on all three enumerable cases (sampled 5-15× above exact training of the same MLP). The toy_support MLP plateau was the learning rate, not capacity: LN+SiLU 64×2 at the default lr reaches 0.0084 (feature table 0.0074). The toy_modes floor (0.012) is the 4×4 actor window; 6×6 → 0.0009. tiny's floor (0.047) is two same-colour sites drawn independently. |
| clamp | 2026-10-13_clamp | **Fill-avoidance hypothesis confirmed, and training creates it.** Trained fine levels cost 1.97 vs 0.72 nats/cell (L = 4) and 2.23 vs 0.34 (L = 3) in fill-following vs own coarse contexts on ground; untrained no gap. Mechanism: fine rho draws are bit-identical whatever the parent; the count prior reaches only the last child; the top level sets certificates near full → all dirt. toy_modes: fine KL 0.36/block in non-mode contexts vs 0 in modes; coarse over-selects modes (0.70 vs 0.55). |
| creplay | 2026-10-13_creplay | Estimator finished and unbiased (gradcheck 4/4). Coarse SNR per world ×6800-13900 on ground. But 1.4 s/update and 300 s compile → loses the 150 s bench; ground still ~98 % dirt. |
| cbig | 2026-10-13_cbig | Persistent compile cache works (196 s → 11 s). Large B does not change ground's collapse (B = 8/64/256 all dirt). toy_support B = 256: F −0.467 vs F* −0.469, not accepted. |
| cconv | 2026-10-13_cconv | Loses everywhere with 5-10× fewer updates; undertrained, not separable from representation. Ground: not all dirt, but ignores the fill, 0.7-7.6 % violations. |
| baselines | 2026-10-13_gpu_baselines | GPU board for greedy/cfast on all five problems, cpass on toy_support. Nothing accepted; collapses as on the Mac (ground all dirt, columns all air, dimers never diagonal). cpass crashes on ground (float32 assert). |

**Reading after wave 1.** Three separate limits, all now measured: (1) sampling variance at the coarse levels (fixed in principle by creplay, at a cost); (2) the fine levels never learned to read the parent count, so they cannot build fill-following contexts and the coarse levels avoid them (cresid targets this); (3) the actor itself: tanh MLP optimises badly and the 4×4 window is too small (cln targets this). Large B alone does nothing for ground.

## Wave 2 (in flight)

- cresid (Opus): generic per-draw count-residual features from the declared count factors; gate = clamp's test; bench ground vs cfast.
- cln (Opus): LN+SiLU actor and 6×6 window as cfg options; `cln`, `cln6` variants; bench toy_support, toy_modes, ground.
- creplay2 (Opus): profile and cut the replay cost (replay from the stored state at the branched step, K = 2, top level only); compile-cache miss; equal-update comparison on ground.
- cbig2 (Sonnet): does B = 64/256 close the variance gap on the real toy_modes bench?
- fixes (Sonnet): cpass GPU assert tolerance; cpass and ctrunc_full on ground; fair cfast columns rerun.
- cconv2 (Sonnet): equal-update comparison cconv vs cfast on the toys.

## Process notes
- Box rules for agents: `/workspace/BOX_RULES.md` (core ranges, flags, no git by agents, no broad pkill, compile warm-up before benches).
- Bench images in `images/bench` are overwritten by concurrent runs: agents copy them into their own folders right after each run.
- cbig's training clock starts after compile; cfast's includes it (about 11 s warm).

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

## Wave 2 and 3 (results; agents' reports in their folders)

| agent | folder | verdict |
| --- | --- | --- |
| cresid | 2026-10-13_cresid | Generic count-residual features: no effect on the clamp gate. Fine rho draws stay 97-100 % identical under a probe clamp; levels 2-3 deterministic. |
| cln | 2026-10-13_cln | LN+SiLU actor and 6×6 window as cfg options (tests pass, defaults unchanged). No gain at B = 8 in sampled training; ground still all dirt. |
| creplay2 | 2026-10-13_creplay | 0.18 s/update (R = (2, 2), K = 4), compile-cache miss fixed. Coarse SNR ×1200-3400, and **more coarse signal changes nothing on ground** at equal updates. |
| cbig2 | 2026-10-13_cbig | B = 256 closes half to two thirds of the toy_modes gap (plateau at the 4×4-window floor); toy_support B = 256 within 0.007 of F*, fails only the (old) probe flag. |
| fixes | 2026-10-13_gpu_baselines | cpass GPU assert fixed; cpass/ctrunc_full on ground all dirt; columns all-air collapse survives a fair budget. |
| cconv2/3 | 2026-10-13_cconv | At equal updates the conv actor matches or beats cfast on F (throughput-limited); at 300 s it collapses exactly like cfast on ground and columns. Parked. |
| probes | 2026-10-14_probes | The old probe flag is a world-level LR test with n = 16: exchangeable draws trip it 53 %. Dimers' diagonal is present untrained and dies by update 20 with entropy collapse. |
| csat | 2026-10-14_csat | **Collapse starts at the top level** (cert4 deterministic ~update 200 as dirt passes 0.95); fine levels follow. lr 1e-3 keeps them stochastic but context-blind. τ tempers only the designed term, not b (15-60 nats spread). |
| ctop | 2026-10-14_ctop | At init the top level is near a tie (partial fill cheaper at τ 2). The all-dirt preference is **built within 50 updates** because the fine levels improve faster in dirt contexts. cert4 drives it. |
| chot | 2026-10-14_chot | ε-hot draws at τ_b with bounded importance weights: unbiased (gradcheck 11/11) and **cannot help by construction** (same expected update as on-policy). |
| ccur | 2026-10-14_ccur | Bottom-up schedule (ideas #7): frozen coarse levels give diverse but non-fill contexts; fine levels get worse in fill contexts; top level collapses within 90 updates of release. |
| cfine | 2026-10-14_cfine | Enumerated fine-level branching with windowed replays: unbiased, fine SNR 0.01 → 25-85, **no effect**: the gradient only acts in visited contexts. |
| cdim | 2026-10-14_cdim | Dimers: entropy and violations trade along one curve for every λ ∈ [1, 20] and τ. λ sets when stripes arrive, not whether. |
| probe2 | 2026-10-14_probe2 | **Bench v2** (user-approved): per-block, per-level Spearman ρ and starvation count. cbig256 toy_modes at the exact-p ideal (ρ 0.87, 0 % starved); collapsed ground starves levels 1-4 by 24-77 nats/block, level 0 fine. |

**Reading after wave 3.** Every estimator-side fix is now built, unbiased and measured: coarse credit (creplay), fine credit (cfine), batch size (cbig), actor (cln, cconv), information (cresid), schedules (csat, ccur), unbiased exploration (chot). None moves ground, for one reason: reverse KL on the sampler's own draws only trains a level in the contexts it visits, the top level locks into all dirt within 50-200 updates because the untrained fine levels are cheaper there, and from then on the fill-following contexts are never visited. The user approved the two things that act on that directly:
- **ideas #12 / #14:** train the fine levels on target-like parent states at a mixed start depth (same minimiser, different context weighting): `cstart` (running), with a generic coarse-state source from the designed energy (`ccoarse`, running).
- **bench v2 probe check** (landed), so starvation is now measured per level instead of by a miscalibrated world-level flag.
Open structural question (cdim2, running): whether a diverse valid policy is representable at all in this sampler class on dimers, which would point at goal-4 freedoms (more passes, colouring, lookahead) rather than training.

## Decisions taken by the user during the session
Exploration at higher τ in scope (ideas #12); bench budget 300 s; ground is the focus problem with the fast loop (dimers pre-check → ground 120 s with entropy/dirt/line_match → clamp gate → bench); ideas #14 in scope; bench v2 probe check approved.

## Process notes
- Box rules for agents: `/workspace/BOX_RULES.md` (core ranges, flags, no git by agents, no broad pkill, compile warm-up before benches).
- Bench images in `images/bench` are overwritten by concurrent runs: agents copy them into their own folders right after each run.
- cbig's training clock starts after compile; cfast's includes it (about 11 s warm).

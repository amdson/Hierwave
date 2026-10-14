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


## Wave 4 (results) and closing summary (04:10 UTC)

| agent | folder | verdict |
| --- | --- | --- |
| baselines300 | 2026-10-14_baselines300 | 300 s board: cbig256 within 0.003 of F* on both toys with exact stats and seed agreement; ground/columns/dimers collapses unchanged at double budget. |
| v2board | 2026-10-14_v2board | Bench v2 rows for the leaders; **proposed gate** (starved ≤ max(3 %, null + 2 pp) at every level; probe ρ at level 0 ≥ 0.8 × ideal) passes cbig256 on both toys and cfast toy_modes, fails every ground/columns/dimers run. For the user. |
| ccoarse | 2026-10-14_ccoarse | Generic coarse-state sampler from the designed energy (tests 25/25): fill-following counts on ground (line_match 0.83) but uniform certificates; uninformative on columns/toys. |
| ccol | 2026-10-14_ccol | Columns collapse has ground's shape (top level deterministic at update 125); fine level cannot realise valid columns even under probe counts (no positions in the count channels). |
| cstart | 2026-10-14_cstart | Probe-lifted contexts at mixed depth, 50/50: no gate pass; collapse delayed (dirt 0.95 at ~450 vs 150), not prevented. Most of its gap was the mixture (see cstart2). |
| cdim2 | 2026-10-14_cdim2 | **Exact gradients on 4×4 dimers: tables reach the diverse zero-violation target (KL 0.018); every MLP collapses to stripes.** The sampler class is not the limit. |
| cdim3 | 2026-10-14_cdim3 | **It is optimisation, not representation**: the distilled MLP fits (KL 0.026, then trains to 0.002). Under λ 20 the early gradient points at the locally safe action and shared weights saturate every context within 50 steps. |
| cbound | 2026-10-14_cbound | **λ ramp 1→20 + whole-logit τ 8→1: KL 4e-5 with exact gradients.** Sampled dimers: `cbound_lam` ends with three orientations (0.47/0.30/0.23) inside the violation budget, one seed. Ground: still all dirt at λ ≈ 3 (soft terms, not the penalty). |
| cstart2 | 2026-10-14_cstart2 | Fine levels shown only fill contexts (f = 1, L = 3): all actors 8× better than cfast's fine levels; LN+SiLU 6×6 best (soft 0.194 vs probe 0.014) but all stall at block terraces; parent-awareness still weak. |
| cbound2 | 2026-10-14_cbound | **Confirmed on two more seeds:** `cbound_lam` keeps three orientations and meets the 2e-4 budget (h/v/d 0.49/0.27/0.24 and 0.44/0.18/0.38; viol 7.8e-5, 1.1e-4). 300 s bench: F −0.29 vs cfast −0.23, raw viol 1.1e-4, only the old probe flag fires. Mix and entropy are still off the target (0.25 each, 0.377). cfast ends with two orientations in every seed; which one dies is seed-dependent. |

### What the session established (numbers in the folders)

1. **The toys are solved** under the new probe check: cbig256 at F* on toy_support and within 0.002 on toy_modes, 0-1.5 % starved blocks (null up to 1.7 %). Variance (batch size) was their only problem.
2. **Every estimator-side fix is built, unbiased and does not move ground**: coarse coupled replays (creplay, ×10⁴ SNR, 0.18 s/update), fine enumeration (cfine), big batches, LN+SiLU / 6×6 / conv actors, count-residual features, schedules, unbiased exploration (chot: provably the same expected update). The old probe flag was miscalibrated; bench v2 measures starvation per level.
3. **The collapse mechanism, measured end to end:** at init the top level is near a tie; within 50 updates the fine levels get better at dirt than at fill contexts; the top level, with exact local cost and few draws, locks into full certificates; from then on fill contexts are never visited; the fine MLP saturates into one parent-independent safe action through its shared weights. Columns and dimers follow the same script.
4. **The network's failure is optimisation, not capacity** (cdim2/cdim3): with exact gradients a table solves dimers and the MLP does not, yet the MLP fits the table's solution when distilled and then trains below it. Keeping the policy soft while the penalty dominates (λ ramp + whole-logit τ) solves dimers exactly, and on sampled dimers `cbound_lam` is the first run to end diverse and in budget.
5. **On ground, context plus the soft early phase are both needed and neither is sufficient yet**: fine levels shown only fill contexts learn them 8× better but stall at terraces (cstart2); the soft schedule alone still collapses the top level because ground's driver is the soft energy, not λ (cbound, ctop).

### Recommended next steps
1. **Combine on ground:** `cstart` at f = 1, L = 3 for an early phase, with the LN+SiLU 6×6 actor and cbound's whole-logit τ schedule, decaying to full-depth training; gate = clamp test + v2 starvation per level. (This is ideas #14 + #17 + the cln options; all built, not yet combined.)
2. **Confirm `cbound_lam` on dimers** at 300 s with 2 seeds through the v2 bench (cbound2's confirmation is appended to the cbound report).
3. **Adopt the v2 gate** (user decision) and drop the old probe flag.
4. **Parent-awareness on ground** is still weak even in fill contexts: test a tabular-like or per-candidate head keyed on parent count and neighbour heights with exact gradients on a small ground block (exactopt has the machinery).
5. Generic target-like certificates remain open (ccoarse): a certificate-aware coarse prior, or training the coarse levels on repaired own draws.

### Housekeeping
- Everything is committed and pushed on `coloured-pass` (HEAD listed in progress.md). No agent processes are left running. The rsync copy at `/workspace/hierwave` is stale; the clone at `/workspace/hierwave_git` is the box's working tree.
- Box rules and all user decisions of the session: `/workspace/BOX_RULES.md` (outside the repo; copied to `notes/box_rules_2026-10-14.md`).

## Decisions taken by the user during the session
Exploration at higher τ in scope (ideas #12); bench budget 300 s; ground is the focus problem with the fast loop (dimers pre-check → ground 120 s with entropy/dirt/line_match → clamp gate → bench); ideas #14 in scope; bench v2 probe check approved.

## Process notes
- Box rules for agents: `/workspace/BOX_RULES.md` (core ranges, flags, no git by agents, no broad pkill, compile warm-up before benches).
- Bench images in `images/bench` are overwritten by concurrent runs: agents copy them into their own folders right after each run.
- cbig's training clock starts after compile; cfast's includes it (about 11 s warm).

# ccombo: mixed-start-depth phase + LN+SiLU 6x6 + whole-logit τ + λ ramp, on ground

2026-10-14 · agent ccombo · cores 40-47 (pytest on 48-55) · one shared RTX 3090 (it was idle during these runs) · one seed per arm · 300 s of training per run. Each run's clock starts after a separate warm-up compile of about 72 s.

**Bottom line.**
- The full combination (`ccombo`) fails the clamp gate. The coarse levels 3-4 lock into all-dirt certificates within about 300 updates of being released. The fine levels then lose their fill-context fit, because the rest of the run trains them only in their own (dirt) contexts.
- The one fix I tried (`ccombo_floor`) keeps a clamped share of 0.5 at the end of training. It fixes L = 4: fill cost 0.56, against 1.16 for cstart_f1L3_ln6. It does not reach cstart_f1L3_ln6's L = 3 number (0.110 vs -0.004; soft 0.317 vs 0.194). The own draws still never follow the line (line_match 0.500).
- Because the gate's L = 3 condition fails, I did not run the 300 s bench. That follows the task rule.

## What was built
`castlegen/variants/ccombo.py`, class `CCombo(CStart)`. It has its own train loop, which merges cstart's fused clamped-world update (`_cstart_core.fused_ctx`, unchanged) with cbound's schedules. No change to `_cstart_core` was needed: the fused update already takes τ and λ per call, and whole-logit τ needs only the last-layer mapping around the step. Generation is cfast's, and the last layer is mapped back to τ = 1 at the end of training.

All schedules are functions of `ftr`, the fraction of post-setup training time; the pure function is `schedule(cfg, ftr)`.

| schedule | default (`ccombo`) | cfg keys |
| --- | --- | --- |
| start depth (ideas #14) | Phase 1 (ftr < 0.35): every world clamped at L = 3 to a probe lift. Decay (0.35 to 0.80): the clamped fraction falls linearly 1 → 0 and clamped worlds draw L from {3, 4}. After 0.80: full-depth worlds only. | `p1, decay_end, frac1, frac_end, Ls1, Ls2` |
| whole-logit τ (cbound `whole`) | q ∝ exp((−s − b_raw)/τ), with τ falling geometrically 8 → 1 over the first 0.8. The stored layer is W_eff = W_raw/τ, each Adam step on it is divided by τ, and it is rescaled when τ changes (cbound's mapping). | `whole, tau0, tau1, anneal, tau_geom` |
| λ ramp (cbound `lam_start`) | λ in use rises linearly 1 → 20 over the first 0.5, with the λ control frozen during the ramp. The control takes over from 20 afterwards. λ stays ≤ 20 throughout. | `lam_start, lam_end, lam_frac` |
| actor | `net="ln"`, `win=(-2, 3)` (LN+SiLU, 6x6) | |

Registered variants:
- `ccombo`: everything above.
- `ccombo_notau`: cfast's τ 2 → 1 on −s only. Registered but not run.
- `ccombo_nophase`: no clamped worlds (control). Registered but not run.
- `ccombo_floor`: the fix, `frac_end = 0.5`.

Tests:
- `tests/variants/test_ccombo.py` (new) checks the schedule values and monotonicity, and that the start-depth draw follows the schedule: 2/2 pass.
- `tests/variants/test_cstart.py`: 11/11 pass (CPU).

## Step 2: ground, 300 s, diagnostics every 10 updates
The diagnostics use 8 held-out worlds and the τ and λ in use. They come from csat's per-level entropy and dirt, plus line_match.

![ccombo](images/diag_ccombo.png)
![ccombo_floor](images/diag_ccombo_floor.png)

| run | updates | H at the end, L0 / L1 / L2 / L3 / L4 | dirt (end) | line_match | λ at the end |
| --- | --- | --- | --- | --- | --- |
| ccombo | 5655 | 0.43 / 0.71 / 1.59 / 0.00 / 0.00 | 0.997 | 0.500 | 3.4 / 0.3 / 1.7 / 0 / 0 |
| ccombo_floor | 5146 | 0.00 / 0.67 / 0.16 / 0.03 / 0.01 | 0.89 | 0.500 | 10 / 20 / 20 / 20 / 0 |

**Phase 1 (updates 0 to about 1650).**
- Levels 3-4 are untrained and stay at their init entropy (3.1-3.6).
- The fine levels lose entropy slowly (L2 2.5 → 1.8).
- Dirt sits at 0.80-0.83, below the untrained 0.86, but line_match stays at 0.50-0.51.

**Decay.**
- L3 and L4 collapse within about 300 updates of their release (L4 entropy 3.3 → 0.6 by update 2100), while τ is still 2.5-3.
- Dirt goes to 0.95 by update 2100, and to 0.99 once frac reaches 0 (`ccombo`).
- This is the same release-collapse ccur saw with frozen coarse levels.

**Fine-level entropy.** In `ccombo`, entropy survives past update 300 and to the end of the run at levels 0-2. That entropy is in dirt contexts, so it is not useful diversity. In `ccombo_floor`, levels 0 and 2 go nearly deterministic by the end.

## Step 3: clamp gate
The gate uses cstart2's window-aware `eval_ground.py`: 32 seeds × 4 probes, τ = 1, the trained λ, fine cost per fine cell over levels below L.

| run | updates | L=3 fill | L=3 soft | L=3 −log q | L=3 own | L=4 fill | L=4 soft | L=4 own | line_match, L=3 clamp | own dirt / line_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| untrained | 0 | 1.98 | 2.41 | 0.52 | 2.07 | 2.32 | 2.72 | 2.47 | 0.60 | 0.88 / 0.505 |
| cfast (clamp report) | | 2.23 | | | 0.34 | 1.97 | | 0.72 | | 1.00 / 0.50 |
| cstart_f1L3_ln6 (cstart2) | 4485 | **−0.004** | **0.194** | 0.20 | 1.12 | 1.16 | 1.41 | 1.50 | 0.59 | 0.85 / 0.504 |
| **ccombo** | 5655 | 1.17 | 1.40 | 0.23 | 0.20 | 1.77 | 2.00 | 0.69 | 0.59 | 1.00 / 0.500 |
| **ccombo_floor** | 5146 | 0.110 | 0.317 | 0.22 | 0.71 | **0.56** | 0.77 | 0.97 | 0.61 | 0.89 / 0.500 |
| probe itself | | | 0.014 | | | | | | 0.88 | |

Per-level breakdown, from `runs/*/ground_summary.json`:
- **ccombo, L = 3 fill:** levels 1-2 cost 0.46 and 0.54. Their fill skill from phase 1 was overwritten.
- **ccombo_floor, L = 3 fill:** levels 1-2 cost −0.11 and 0.04.
- **ccombo_floor, L = 4 fill:** level 3 costs 0.30, against the untrained 0.41 that cstart2 carried.
- **Violations:** at most 5e-3 at level 2 under the L = 4 clamp (ccombo_floor), and ≤ 1.2e-3 elsewhere.

What the numbers show:
- **The fill-vs-own gap reverses with the floor.** ccombo_floor's fine levels are cheaper in fill contexts than in their own (0.11 vs 0.71 at L = 3, 0.56 vs 0.97 at L = 4). cfast is the opposite way round (2.23 vs 0.34). Plain ccombo is back to cfast's sign (1.17 vs 0.20).
- **The top level still does not choose fill contexts.** Own line_match is 0.500, and the top-level entropy is 0.01.
- **Parent-awareness is still weak.** Comparing probe j0 with j2, the rho2 draws change 1.1 % (ccombo) and 2.2 % (floor), against 2.8 % for ln6 and 7.9 % for the probes themselves. rho3 at L = 4 changes 0.1-0.3 %.

Clamped completions: `images/clamped_examples_ccombo*.png`.

## Step 4: bench not run
The gate requires L = 3 to be at least as good as cstart_f1L3_ln6. ccombo gives 1.17 and ccombo_floor 0.110, against −0.004. Following the task rule, I did not run the bench. The one fix (the floor) is spent.

## What failed, and the diagnosis
1. **The coarse levels collapse at release.** Whole-logit τ at 2.5-3 and λ around 17 did not keep L3-L4 soft once they started training. L4 went deterministic in about 300 updates, the same mechanism as ccur and ctop.
   - Training levels 0-2 in fill contexts first did not make fill-following certificates cheaper for the top level in practice. At release, the top level sees the fine levels' cost in its own draws' contexts, and those are mostly dirt (0.82).
2. **Without a clamped floor, the fine levels forget fill contexts.** In ccombo the last 45 % of updates are own-draw only, in dirt, and the L = 3 fill cost ends at 1.17 (the phase-1 value was not measured; no mid-run snapshot).
3. **With the floor, the fine levels keep fill contexts, but at a lower quality than all-clamped training.** L = 3 is 0.110 vs −0.004, because half of the batches are now own/dirt contexts and the run is shorter in updates.

## Next (proposals)
- **Treat the top level separately from the fine-level schedule.** Release it while the clamped floor stays at 0.5 or higher, with a much higher τ at release (whole-logit τ restarted from 8 for the coarse levels only, or a per-level τ schedule). Both stay within the allowed freedoms: τ schedule and per-level learning rates.
- **Or train levels 3-4 on probe-lifted contexts too:** clamp at L = 5 is not available, but the top level could start from probe certificates via ccoarse's generic coarse prior. Ideas #14 already approves target-like parent states.
- **Not run (registered):** `ccombo_notau` and `ccombo_nophase` ablations. A second seed is needed before any of these numbers count.

## Files
- Code: `castlegen/variants/ccombo.py` (new) and `tests/variants/test_ccombo.py` (new).
- Scripts in this folder: `train_diag.py` (cstart2's train_gate.py without the SNR part), `run.sh`, `ev.sh` (runs cstart2's eval_ground.py), `gate.py`, `plot_diag.py`.
- `runs/<arm>/`: params_ground.pkl, train_ground.jsonl, ground_summary.json, change.json, plus all logs.

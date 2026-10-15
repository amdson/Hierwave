# Proposals: certificates as cheap scaffolding, terrain as the hard part (2026-10-15)

These follow from the cmix findings ([report](report.md), sections 1–6). Each proposal states its question, setup, measurements, cost and constitution status (constraint numbers refer to notes/coloured_pass_constitution.md). They are ordered by priority. Nothing here has been run yet.

**Findings they build on:**
- Certificates given terrain are easy: valid in about 20–40 sweeps, at equilibrium statistics.
- Terrain sampling is the slow part, and committed certificates make it 2–4× slower.
- Block-alternating terrain and high certificates fails: the certificates block drift.
- A certificate-free terrain chain plus a certificate fill always ends valid. But its certificates are 3× tighter at levels 3–4 than under ground's p, because p weights each terrain by its number of valid certificate assignments.

---

## Decision needed first (user)
**Which terrain target?**
- **(A)** ground's p as declared: certificates are free variables with uniform weight, so terrains are weighted by #valid certificates.
- **(B)** certificates as derived scaffolding: the terrain distribution comes from the soft energies only (the certificate-free target), and certificates are filled afterwards.

(B) changes the declared target, so it needs your approval (constraint 4: the objective is fixed). P1 sizes how much the choice matters; P2 tests whether (A) can be kept cheaply.

## P1. Size the certificate-counting term exactly (diagnostic)
- **Question:** how far is p_strip(terrain) from p(terrain), per cell and per level? Does the gap grow with T?
- **Setup:**
  - Tiny periodic ground variants (for example H = 8, W = 4–8, T = 2–3). Enumerate every support-valid tile configuration, i.e. monotone columns.
  - For each one, count the valid certificate assignments exactly (brute force or a DP over the certificate tree). Sum rho out exactly; it appears in both targets.
  - Compare the two terrain marginals.
- **Measure:**
  - KL(p_strip ‖ p) per cell;
  - certificate slack under both targets, against the 3× gap seen at W = 128;
  - which terrain features the weight rewards (block-aligned full columns?).
- **Cost:** minutes on the Mac. **Status:** diagnostic; no constraint touched.

## P2. Local correction for the certificate count (keeps target A)
- **Question:** can a certificate-free terrain chain sample p's terrain marginal using an acceptance ratio of local certificate counts?
- **Setup:**
  - Terrain moves (single-cell or block) on certificate-free ground, accepted with an extra factor #certs(new) / #certs(old).
  - Estimate that ratio exactly inside a finite window around the move: the certificate cells whose factors the changed tiles reach (their ancestor cone and its support neighbours). Grow the window until the ratio stabilises.
  - Validate on P1's enumerable cases, then compare with the full chain's certificate slack at W = 128 (0.011 / 0.045 / 0.151 / 0.52).
- **Measure:** exactness against enumeration; window size needed; τ and cost against the full chain (τ 83, 0.053 s/sweep) and the uncorrected scheme.
- **Cost:** 1–2 h to build, 5-min runs. **Status:** a sampler kernel (kernel-route amendment), generic (windows derived from the factor reach). At generation time it must use fixed windows and fixed counts (constraint 2).
- **Caveat:** AIS on a related count already failed for growing structures (memory: limb free energy). Exact window counts are the only version worth trying.

## P3. Fixed-step generation along the scheme (target B)
- **Question:** generate the terrain top-down without certificates, then fill the certificates with a fixed-step local fill. Does it meet the 2 × 10⁻⁴ violation budget at a fixed fill budget?
- **Setup:**
  - Terrain from the untrained greedy pass and from a cfast-style trained pass, on the certificate-free problem (rho hierarchy + tiles, support0 via the lookahead).
  - Then F graded λ = 20 certificate-only sweeps, F ∈ {40, 60, 80, 120}, on the bench strip (W = 128) and on wide windows (W = 512, 2048) to see the tail.
  - Fill variants:
    1. from random;
    2. from the tight abstraction (A = full columns, B = occupied). This is valid by construction, so the fill only has to spread slack (see P4).
- **Measure:**
  - certificate violations per level against the budget;
  - fill failure rate per cell against F;
  - certificate slack against the equilibrium values;
  - finite-support window test;
  - terrain statistics against the bench.
- **Cost:** Mac runs of 5–10 min; training runs on the box. **Status:**
  - The fill is a fixed-count local step (constraint 2 is fine).
  - Training the terrain sampler on the certificate-free target is a change of objective (constraint 4) and needs approval (decision B).
  - The untrained greedy-pass arm is diagnostic.

## P4. Fill from the tight abstraction instead of random
- **Question:** the tight certificates (A = full columns, B = occupied columns) are always valid. How many hard certificate-only sweeps from there reach equilibrium slack?
- **Setup:** on full-chain terrains, compare against the exact conditional (`certs.py` reference). Record slack per level after k = 0, 5, 10, 20, 40 sweeps.
- **Measure:** sweeps to equilibrium slack per level. Validity is 100 % at every step by construction.
- **Cost:** ~5 min. **Status:**
  - The tight value is ground's `lift`, problem-specific code.
  - It is generic only if certificates are declared as abstraction maps of the level below (the DSL direction: channels as abstraction maps).
  - Otherwise constraint 6 (no hand-written rules linking channels) blocks it, and it stays a diagnostic.

## P5. Joint parent + children block moves (fixes the paired-flip deadlock)
- **Question:** single-cell moves can't refine tiles under tight level-1 certificates (0/16 valid, section 2). Does exact block Gibbs over a parent cell and its 4 children (all channels) fix refinement and speed up mixing?
- **Setup:**
  - castlegen/kernels/exact_block.py / fine_block.py (existing).
  - Rerun section 2's refinement (tiles only free, tight and chain certificates) and section 1's 4-start mixing test on full ground.
- **Measure:** refinement validity and sweeps; τ and shift recovery per wall-second against single-cell Gibbs.
- **Cost:** ~10 min. **Status:** generic (blocks from the hierarchy), kernel route; ideas #15 / #20(a).

## P6. Generic rule for which certificate channels are derived
- **Question:** can "drop or derive a channel when its conditional entropy given the level below is about zero" be measured generically? Does certificate-free single-site Gibbs survive on the other suite problems?
- **Setup:**
  - From long valid chains, estimate H(channel_l | level l − 1) for every coarse channel of ground, columns and the toys. On ground the slack is 0.011 / 0.046 / 0.16 / 0.54 at levels 1–4.
  - Then run the stripped single-site chain on columns: length-12 constraints with hard count-only coarse channels.
- **Measure:** entropy per channel; validity and mixing on columns with and without its coarse channels.
- **Cost:** ~10 min. **Status:** diagnostic. Turning it into a rule touches constraint 1 (generic) and needs the rule written from declarations only.

## P7. Speeding up the intrinsic slow mode (surface drift) on certificate-free ground
- **Question:** without certificates, τ is still 30–45 sweeps, and a 6-row shift takes about 105 sweeps. Do generic block moves or tempering help per wall-second?
- **Setup:**
  - Subtree block MH and coarse flips (castlegen/kernels/block_mh.py, coarse_flip.py) on the certificate-free problem.
  - A replica stack on λ for support0 only (graded swaps gave 5× round trips on full ground).
- **Measure:** τ and shift recovery per wall-second against the plain certificate-free chain.
- **Cost:** ~10 min. **Status:** diagnostic or kernel route.

## P8. Graded vs binary fill tail at scale
- **Question:** in fills of 32 worlds, graded's worst case was 73–77 sweeps against binary's 87–208. Does that tail gap hold or grow at W = 512 / 2048? A fixed fill budget is set by the tail per cell, not the median.
- **Setup:** `fill_len.py` at larger W, both penalties, binary also at λ ∈ {8, 20}.
- **Measure:** fill-failure survival curve per cell against sweeps.
- **Cost:** ~5–10 min. **Status:** diagnostic; feeds P3's choice of F.

## P9. Annealing redo with a shallow bottom (low priority)
- **Question:** cycles down to λ = 1 never ended valid (the surface floated about 25 rows). Do shallow cycles (λ 20 → 6 → 20), or tempering only the soft energy with hard constraints kept, give valid ends and real decorrelation per cycle?
- **Setup:** `mix.py` cyc arms with new schedules, cool-down 50–100 sweeps.
- **Measure:** valid fraction at cycle ends; terrain change per cycle; per-second effective samples against the hard chain.
- **Cost:** ~5 min. **Status:** diagnostic.

---

**Suggested order:**
1. P1 and P4 (cheap, and they inform decision A/B).
2. P5 (the deadlock fix is useful under either target).
3. P2 if you choose A, or P3 + P8 if you choose B.
4. P6 and P7.
5. P9 last.

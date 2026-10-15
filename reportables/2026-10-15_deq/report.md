# deq: true equilibrium orientation fractions of the dimers bench strip (2026-10-15)

Question: what are the equilibrium h/v/d of the bench dimers problem (32 rows, open top/bottom, x periodic/window, uniform over perfect matchings of the square grid + (1,1) diagonal), and is 0.5/0.25/0.25 the right gate?

## Answer: NO. Exact bulk-strip value is h/v/d = 0.351/0.324/0.324 (about 1/3 each, h +0.018 from the top/bottom edge rows). The gate 0.5/0.25/0.25 is wrong for the bench geometry; the k=3 chain plateau 0.38/0.38/0.23 is also not equilibrium.

Method: exact, no MCMC. Kasteleyn/Pfaffian on the planar triangular graph (orientation found by search so every triangle has an odd number of clockwise edges), P(edge uv) = |A_uv (A^-1)_vu|. Check: sum of edge probabilities = N/2 exactly in every run. Script `scripts/exact.py` (CPU only, cores 4-5, one run, seconds). Open window in x (planar); the model is gapped with short correlation, so central columns equal the periodic strip up to exponentially small terms (central-32 values identical to 4 digits for W = 64, 128, 192).

| case | h / v / d |
|---|---|
| 32 x 128, central 32 columns (strip bulk, bench height) | 0.3513 / 0.3244 / 0.3244 |
| 32 x 192, central 32 columns | 0.3513 / 0.3244 / 0.3244 |
| 32 x 64, central 32 columns | 0.3513 / 0.3244 / 0.3244 |
| 32 x 64 whole window (x ends included) | 0.347 / 0.334 / 0.319 |
| per row, 32 x 128 centre: row 0 / row 1 / row 3 / row 8 | .461/.27/.27 ; .328/.336/.336 ; .333/.334/.334 ; 1/3 each |
| 4 x 8 periodic-x strip, brute-force enumeration (26 817 matchings, matches dimers_kernel count) | 0.437 / 0.291 / 0.273 |
| 6 x 6, 4 x 4 periodic-x strip enumerated | 0.379/0.317/0.304 ; 0.404/0.313/0.283 |

- Why h is high: only the top and bottom rows are affected. A top-row cell cannot pair N/NW, so those cells are pushed to E/W (row 0: 0.46 h). The edge effect decays within 2 rows; the 32-row mean is h +0.018. Bulk is the lattice-symmetric 1/3 each.
- Finite-size: small strips (4 rows) have h 0.44-0.45 because the edge rows are a large share; at 32 rows the edge is 1/16 of rows. So 0.5/0.25/0.25 is not a small-case enumeration of this problem: I could not reproduce 0.505/0.248/0.248 with periodic-x strips (4x4, 4x8, 6x6) or open windows (4x4: .446/.446/.107). It appears only in ideas.md (line ~205, "dimers 4x4 exact gradients", a trained-model KL result, not a uniform-matching enumeration) and dimers_kernel report; its provenance as a target is unverified.
- Chain plateaus are not equilibrium: 3x3 block Gibbs from collapsed (h/v-rich) draws sits at 0.38/0.38/0.23 and from rhombi (h/d-rich) at 0.40/0.21/0.39 (dimers_kernel), both 0.03-0.17 from the exact 0.351/0.324/0.324. Those are slow-mode (sector) plateaus of the k=3 kernel, consistent with the opposite-start chains not meeting. dstage's suspicion that the 3x3 plateau is the equilibrium is not supported.
- Not done: no MCMC runs (the exact answer made the multi-start chains unnecessary; GPU not used); 4x4 blocks not run; the periodic cylinder was not computed exactly (central-column argument above).

## Recommendation
Gate dimers on h/v/d = 0.351/0.324/0.324 with the existing bench tolerance 0.02 (so h in 0.331-0.371, v and d in 0.304-0.344), using the central/whole-strip mean over many worlds; it is exact, so no sampler needs to be trusted to define it. The existing `tol` of 0.02 is fine. A sampler that stalls at 0.38/0.38/0.23 or 0.5/0.25/0.25 fails it. Per-row profile (row 0 h = 0.46) is a further exact check.

## Next
One line: set the bench `exact` for dimers from `scripts/exact.py` (coordinator decision; bench frozen), and rerun dstage-style chains against the 1/3 target, with k=3 chains plus 4x4 or long-range moves to cross the h/v-rich vs h/d-rich sectors.

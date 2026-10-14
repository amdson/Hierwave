| run | steps | fwd KL(p*\|q) | rev KL(q\|p_lam10) | p_viol | F/cell | F* | H/cell | H(p*)/cell | stats (q) | stats (p*) |
|---|---|---|---|---|---|---|---|---|---|---|
| modes_ln_ml_exact | 1307 | 0.0116 | 0.0142 | 0.0000 | 0.2521 | 0.2513 | 0.2602 | 0.2591 | mode_other 0.2563, mode_L 0.1859, broken_bonds 5.4651 | mode_other 0.2564, mode_L 0.1859, broken_bonds 5.4437 |
| modes_ln_ml_sample | 1282 | 0.0325 | 0.0338 | 0.0000 | 0.2534 | 0.2513 | 0.2557 | 0.2591 | mode_other 0.2502, mode_L 0.1864, broken_bonds 5.4302 | mode_other 0.2564, mode_L 0.1859, broken_bonds 5.4437 |
| modes_ln_rev_exact | 2685 | 0.0143 | 0.0117 | 0.0000 | 0.2520 | 0.2513 | 0.2557 | 0.2591 | mode_other 0.2481, mode_L 0.1878, broken_bonds 5.4150 | mode_other 0.2564, mode_L 0.1859, broken_bonds 5.4437 |
| modes_tanh_ml_exact | 5173 | 0.0117 | 0.0142 | 0.0000 | 0.2521 | 0.2513 | 0.2604 | 0.2591 | mode_other 0.2568, mode_L 0.1859, broken_bonds 5.4675 | mode_other 0.2564, mode_L 0.1859, broken_bonds 5.4437 |
| modes_tanh_ml_sample | 2703 | 0.0292 | 0.0314 | 0.0000 | 0.2532 | 0.2513 | 0.2570 | 0.2591 | mode_other 0.2676, mode_L 0.1872, broken_bonds 5.4418 | mode_other 0.2564, mode_L 0.1859, broken_bonds 5.4437 |
| modes_tanh_rev_exact | 8728 | 0.0143 | 0.0118 | 0.0000 | 0.2520 | 0.2513 | 0.2557 | 0.2591 | mode_other 0.2489, mode_L 0.1888, broken_bonds 5.4155 | mode_other 0.2564, mode_L 0.1859, broken_bonds 5.4437 |
| support_ln_ml_exact | 423 | 0.0155 | 0.0621 | 0.0065 | -0.4649 | -0.4688 | 0.3349 | 0.3324 | flat_block_frac 0.0924 | flat_block_frac 0.0959 |
| support_ln_ml_sample | 1600 | 0.0259 | 0.0940 | 0.0088 | -0.4629 | -0.4688 | 0.3347 | 0.3324 | flat_block_frac 0.0950 | flat_block_frac 0.0959 |
| support_ln_rev_exact | 187 | 0.0559 | 0.0595 | 0.0063 | -0.4651 | -0.4688 | 0.3270 | 0.3324 | flat_block_frac 0.0991 | flat_block_frac 0.0959 |
| support_tanh_ml_exact | 402 | 0.0205 | 0.0806 | 0.0098 | -0.4637 | -0.4688 | 0.3352 | 0.3324 | flat_block_frac 0.0936 | flat_block_frac 0.0959 |
| support_tanh_ml_sample | 1827 | 0.0459 | 0.1295 | 0.0120 | -0.4607 | -0.4688 | 0.3335 | 0.3324 | flat_block_frac 0.0938 | flat_block_frac 0.0959 |
| support_tanh_rev_exact | 893 | 2.1320 | 0.3073 | 0.0050 | -0.4496 | -0.4688 | 0.2872 | 0.3324 | flat_block_frac 0.1343 | flat_block_frac 0.0959 |

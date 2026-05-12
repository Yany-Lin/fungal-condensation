# Framework Validation Report

## Executive Summary

After excluding failed `Rhizopus`/Black plates from species-level comparisons, the viable fungal comparison is clean:

- `Aspergillus` dry-zone width: 298.12 um.
- `Mucor` dry-zone width: 139.86 um.
- Measured ratio: 2.13x.
- Morphology-predicted absorbing-capacity ratio: 2.01x.
- Gap: about 6%.

The strongest supported claim is that colony architecture, summarized as tissue fraction times structure thickness, explains the viable Asp/Muc difference in dry-zone width. The secondary condensation observables remain directionally concordant but are not individually significant at n = 5 vs 5.

## Mechanistic Frameworks Tested

### Effective Medium Diffusion: Rejected

The tortuosity hypothesis predicts that denser `Aspergillus` tissue should impede vapor transport more strongly than `Mucor`.

Test: solve the steady-state Laplace equation in the pore phase of 3D colony masks and compute D_eff/D0.

| Metric | `Aspergillus` | `Mucor` | Ratio | p |
|---|---:|---:|---:|---:|
| D_eff/D0 | 0.524 +/- 0.125 | 0.535 +/- 0.175 | 0.98 | 0.866 |

Interpretation: both resolved pore networks are well above percolation threshold. Transport resistance through the resolved pore phase does not differentiate the genera.

### Absorbing Capacity: Supported

The absorbing-capacity framework treats the colony as a projected reservoir of hygroscopic material:

`A = f_tissue x d_structure`

where `f_tissue` is tissue area fraction and `d_structure` is median distance-transform thickness.

| Metric | `Aspergillus` | `Mucor` | Ratio | Cohen d | Welch p |
|---|---:|---:|---:|---:|---:|
| Tissue fraction | 0.300 +/- 0.008 | 0.258 +/- 0.019 | 1.16 | 3.04 | 1.1e-5 |
| Thickness | 7.21 +/- 0.26 um | 4.16 +/- 0.53 um | 1.73 | 7.58 | 7.1e-11 |
| Absorbing capacity | 2.16 +/- 0.11 | 1.07 +/- 0.16 | 2.01 | 8.22 | 3.1e-13 |

Prediction check: 2.01x predicted vs 2.13x measured. This is a 6% gap.

## Counterintuitive Control: Specific Surface Area

`Mucor` has higher 2D specific surface area:

| Metric | `Aspergillus` | `Mucor` | Ratio | p |
|---|---:|---:|---:|---:|
| Specific surface area | 0.148 1/um | 0.262 1/um | 0.56 | 7.0e-7 |

If surface-to-volume ratio alone determined sink strength, `Mucor` would be expected to win. It does not. The result supports bulk absorbing material per footprint rather than perimeter per tissue area as the relevant resolved-scale descriptor.

## Figure 4 Concordance

Figure 4 uses direction-normalized ratios so values above 1 always indicate the `Aspergillus`-favored direction.

### Condensation Observables: Green vs White Only

| Observable | Ratio | p | Interpretation |
|---|---:|---:|---|
| delta, dry-zone width | 2.13 | 9.1e-5 | strong |
| Size gradient | 1.07 | 0.752 | direction only |
| tau50 gradient | 1.25 | 0.462 | direction only |
| d* | 1.10 | 0.663 | direction only |

The old `White + Black` pooling is not used. Black represents failed/dead `Rhizopus`, not viable `Mucor` morphology.

### Morphological Metrics

| Metric | Ratio | p | Role |
|---|---:|---:|---|
| FFT spectral slope alpha | 1.16 | 0.0014 | main texture metric |
| Tissue thickness | 1.73 | 7.1e-11 | main geometry metric |
| Hessian tubeness CV | 1.70 | 9.8e-4 | main unit-morphology metric |
| Erosion survival | 2.13 | 7.9e-5 | compactness metric |

All eight Figure 4 ratios point in the `Aspergillus`-favored direction. The binomial sign test is p = 0.008. Because several metrics share images, masks, or derived components, this should be described as directional concordance, not as eight independent validations.

## Reviewer-Level Caveats

- The 3D segmentation threshold was calibrated to a validated thickness range. The genus ordering is strong, but a threshold-sensitivity analysis should be cited whenever the thickness result is discussed.
- Hessian tubeness CV was chosen after exploratory screening of Hessian-derived metrics. It is biologically interpretable and magnification-independent, but the metric-selection process should be transparent.
- Light microscopy has n = 6 vs n = 3. Exact Mann-Whitney p-values are bounded at 0.024 even under perfect separation, so effect-size precision is limited.
- Absorbing capacity is a 2D projected proxy. The close 2.01x vs 2.13x agreement is strong consistency for these two genera, not a universal predictive law.
- FFT alpha measures image texture and high-frequency power. It should not be described as reconstructing literal pore geometry or true nanoscale surface area.
- Specific surface area is a 2D perimeter/tissue-area proxy, not true 3D surface area.

## Final Mechanistic Statement

At the resolved optical scale, `Aspergillus` forms thicker, more spatially heterogeneous, more erosion-resistant tissue structures than viable `Mucor`. The product of tissue fraction and structure thickness gives a simple absorbing-capacity proxy that quantitatively matches the measured dry-zone width ratio. Tortuosity and resolved specific surface area do not explain the direction of the effect.

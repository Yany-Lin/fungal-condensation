# Methodology: Hyphal Architecture Analysis

## Overview

This analysis compares viable `Aspergillus` and `Mucor` colonies only. The failed `Rhizopus`/Black condition is excluded from Figure 4 species-level comparisons because colonies did not establish. It may remain in Figure 3 only as low-source failed-inoculum data along the universal condensation axis.

The main mechanistic claim is deliberately narrow: a 2D projected absorbing-capacity proxy,

`A = f_tissue x d_structure`,

predicts the viable Asp/Muc dry-zone ratio well. The measured dry-zone ratio is 2.13x, and the morphology-derived prediction is 2.01x, a 6% gap.

## Segmentation

### 3D Colony Surface ROIs

Input images are manually cropped in-focus regions from reflected-light colony-surface photographs, calibrated at 0.94 um/px.

The 3D masks use adaptive dark-response thresholding:

1. Gaussian smoothing, sigma = 1 px.
2. Local mean subtraction, sigma = 32 px.
3. Dark response = local mean - smoothed image.
4. Threshold = mean(dark response) + 0.5 x std.
5. Binary closing followed by opening.

The threshold was selected by parameter sweep to reproduce the biologically validated thickness range. The final means are 7.21 um for `Aspergillus` and 4.16 um for `Mucor`. Sensitivity across threshold multipliers 0.3-0.7 gives Asp/Muc thickness ratios of 1.67-1.73, with Welch p < 2.4e-9 at every threshold; see `segmentation_threshold_sensitivity.csv`.

### Light Microscopy

Input images are 16-bit TIF brightfield images at 10x, 20x, and 40x magnification. Tissue is segmented with the adaptive dark-response method from `analyze_fungi.py`, using genus-specific intensity caps and percentile thresholds to capture broad `Aspergillus` tissue masses and fine `Mucor` filaments.

The light-microscopy sample size is small: n = 6 `Aspergillus` fields and n = 3 `Mucor` fields. Welch tests are reported as primary tests, and exact Mann-Whitney tests are used as non-parametric confirmation. With n = 6 vs 3, the smallest possible two-sided exact Mann-Whitney p-value is 0.024, so this limitation must be disclosed.

## Main Figure 4 Metrics

### Panel C: FFT Spectral Slope

Tile-based 2D FFT is computed on 3D colony-surface ROIs. Each ROI is divided into 256 x 256 px tiles with 128 px stride, windowed with a Hann function, transformed, radially averaged, and fit on the log-log power spectrum over 0.02-0.40 cyc/px.

Result:

| Genus | alpha |
|---|---:|
| `Aspergillus` | -3.01 +/- 0.11 |
| `Mucor` | -3.51 +/- 0.38 |

Welch p = 0.0014, Cohen d = 1.85. A shallower slope means more high-frequency image texture. GLCM contrast is retained only as spatial-domain cross-validation of this FFT result.

### Panel D: Tissue Thickness

Tissue thickness is the median Euclidean distance-transform value within the 3D tissue mask.

| Genus | Thickness |
|---|---:|
| `Aspergillus` | 7.21 +/- 0.26 um |
| `Mucor` | 4.16 +/- 0.53 um |

Ratio = 1.73, Welch p = 7.1e-11, Cohen d = 7.58.

### Panel E: Hessian Tubeness CV

Multi-scale Hessian tubeness is computed on light microscopy images at sigma = 1, 2, 4, 8, and 16 px. The retained response is the maximum non-negative larger Hessian eigenvalue across scales. Tubeness CV is std/mean of tubeness values within tissue pixels.

Interpretation: high CV indicates mixed morphological phases, with solid conidial cores plus filamentous edges. Low CV indicates more uniformly filamentous tissue.

| Genus | Tubeness CV |
|---|---:|
| `Aspergillus` | 1.038 +/- 0.168 |
| `Mucor` | 0.610 +/- 0.064 |

Ratio = 1.70, Welch p = 9.8e-4, Cohen d = 2.93. The Asp/Muc ratio is stable across magnification: 1.71x at 10x, 1.76x at 20x, and 1.65x at 40x.

Important caveat: tubeness CV was selected after exploratory Hessian metric screening. It should be presented transparently as the biologically interpretable winner from a candidate sweep, not as a blindly pre-registered endpoint.

### Panel F: Erosion Survival

Erosion survival is the fraction of tissue pixels remaining after 10 binary erosion iterations.

| Genus | Erosion survival |
|---|---:|
| `Aspergillus` | 0.791 +/- 0.103 |
| `Mucor` | 0.372 +/- 0.025 |

Ratio = 2.13, Welch p = 7.9e-5, Cohen d = 4.75. Exact Mann-Whitney gives U = 18, p = 0.024, the lower bound possible at n = 6 vs 3.

## Absorbing Capacity

Absorbing capacity is defined as:

`A = f_tissue x d_structure`

where `f_tissue` is 3D tissue area fraction and `d_structure` is median structure thickness.

| Metric | `Aspergillus` | `Mucor` | Ratio |
|---|---:|---:|---:|
| Tissue fraction | 0.300 | 0.258 | 1.16 |
| Thickness | 7.21 um | 4.16 um | 1.73 |
| Absorbing capacity | 2.16 | 1.07 | 2.01 |

The morphology-derived capacity ratio, 2.01x, predicts the clean viable-fungal dry-zone ratio, 2.13x, within 6%.

This is a 2D projected proxy for hygroscopic mass per footprint. It is strong model consistency for two viable genera, not a general predictive law.

## Condensation Observables

Figure 4 validation uses `Green` (`Aspergillus`, n = 5) vs `White` (`Mucor`, n = 5) only.

| Observable | Direction-normalized ratio | Cohen d | Welch p |
|---|---:|---:|---:|
| delta, dry-zone width | 2.13 | 6.10 | 9.1e-5 |
| Size gradient | 1.07 | 0.21 | 0.752 |
| tau50 gradient | 1.25 | 0.50 | 0.462 |
| d* | 1.10 | 0.29 | 0.663 |

Only delta is individually significant after excluding failed Rhizopus/Black. The three secondary observables are directionally concordant but not significant at n = 5 vs 5.

## Negative And Cross-Validation Analyses

Effective medium diffusion/tortuosity is rejected as the differentiating mechanism:

| Metric | `Aspergillus` | `Mucor` | p |
|---|---:|---:|---:|
| D_eff/D0 | 0.524 | 0.535 | 0.866 |

Specific surface area is higher in `Mucor`, so simple surface-to-volume ratio cannot explain stronger `Aspergillus` depletion.

GLCM contrast cross-validates FFT ordering but is not the primary texture metric. FFT is retained because it uses the full frequency-domain spectrum rather than a single nearest-neighbor contrast statistic.

## Statistical Policy

- Welch unequal-variance t-tests are primary pairwise tests.
- Cohen d uses pooled standard deviation.
- Exact Mann-Whitney tests are reported where useful, especially for n = 6 vs 3 light microscopy.
- Directional concordance is reported as a binomial sign test across direction-normalized Figure 4 ratios.
- Correlated metrics are not treated as independent mechanistic confirmations.

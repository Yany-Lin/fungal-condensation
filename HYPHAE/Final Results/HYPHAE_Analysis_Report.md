# Structural Morphology of Aspergillus vs Mucor Colony Surfaces

## Summary

Three independent quantitative metrics, measured across three imaging scales, demonstrate that Aspergillus and Mucor colonies differ in how they organize tissue. Aspergillus concentrates material into regularly-spaced dense conidial clusters separated by narrow channels. Mucor distributes material as a uniform filamentous network with irregular, wide gaps. This architectural difference explains why Aspergillus produces depletion zones 2.1x wider than Mucor (delta = 298 +/- 16 um vs 140 +/- 33 um): clustered morphology creates more condensation surface and tighter vapor-trapping channels.

---

## Metric 1: FFT Spectral Slope (Colony Surface Texture)

**Method.** Colony-surface photographs were divided into tiles (512x512 px, stride 256). Tiles were quality-gated by Laplacian variance (focus) and intensity standard deviation (texture). For qualifying tiles, the 2D power spectrum was azimuthally averaged and fitted as a power law over 0.01-0.45 cyc/px. The slope alpha quantifies texture complexity: shallower alpha = more fine-scale structure.

**Result.**

| | Aspergillus | Mucor |
|---|---|---|
| Colony surface | -2.960 +/- 0.024 (n=14) | -3.092 +/- 0.101 (n=7) |
| 3D macro (ROI) | -3.221 +/- 0.086 (n=13) | -3.460 +/- 0.240 (n=10) |

Colony surface: Welch t = 3.43, p = 0.013, Cohen's d = 2.21 [1.08, 3.33].
3D macro: Welch t = 3.00, p = 0.012, Cohen's d = 1.41 [0.49, 2.32].

**Interpretation.** Aspergillus surfaces have a shallower spectral slope at both imaging scales, indicating more fine-scale texture from the granular surface of packed conidial masses. The result replicates across two independent image sets collected on different dates.

---

## Metric 2: Local Density Coefficient of Variation (Tissue Heterogeneity)

**Method.** Light microscopy images of disaggregated tissue (10X, 20X, 40X) were segmented via Otsu thresholding. The field of view was divided into 256x256 um patches and the foreground fraction computed per patch. The coefficient of variation (CV = std/mean) across patches quantifies spatial heterogeneity of tissue distribution. This metric is preparation-independent: it measures how tissue is distributed regardless of total amount.

**Result.**

| | Aspergillus (n=6) | Mucor (n=3) |
|---|---|---|
| Local density CV | 0.687 +/- 0.110 | 0.392 +/- 0.034 |

Welch t = 6.04, p = 0.0007, Cohen's d = 3.12. Survives Benjamini-Hochberg correction (p_BH = 0.012).

Consistent across magnifications:

| Magnification | Aspergillus | Mucor | Ratio |
|---|---|---|---|
| 10X | 0.607 | 0.355 | 1.71 |
| 20X | 0.668 | 0.400 | 1.67 |
| 40X | 0.785 | 0.421 | 1.86 |

**Interpretation.** Aspergillus tissue is spatially heterogeneous (CV = 0.69): dense conidial masses alternate with nearly empty regions. Mucor tissue is uniform (CV = 0.39): filaments distribute evenly regardless of observation scale. The Asp/Muc ratio is constant (~1.7x) across 10X, 20X, and 40X magnification, confirming this reflects intrinsic morphology, not preparation artifact.

---

## Metric 3: Lacunarity (Gap Distribution)

**Method.** 3D colony-surface photographs were manually cropped to in-focus regions of interest (13 Aspergillus, 11 Mucor; calibration 0.94 um/px). ROIs were binarized via Otsu thresholding and analyzed using the gliding-box lacunarity algorithm. Lacunarity measures the variance-to-mean ratio of tissue mass across sliding windows, quantifying how "gappy" the spatial distribution is at each measurement scale.

**Result.**

| | Aspergillus (n=13) | Mucor (n=11) |
|---|---|---|
| Lacunarity (largest scale) | 1.305 +/- 0.156 | 2.041 +/- 0.542 |

Welch t = -4.36, p = 0.001, Cohen's d = -1.92. Mann-Whitney U = 2, p = 0.0001.

**Interpretation.** Mucor has significantly higher lacunarity: at large spatial scales, its tissue alternates between dense patches and wide empty regions in an irregular pattern. Aspergillus tissue, despite being organized into discrete conidial clusters, distributes those clusters more regularly across the field -- creating a uniform "mesh" of condensation surfaces. This regularity is what makes Aspergillus an effective vapor sink: humid air encounters condensation surfaces at predictable intervals rather than passing through large unstructured voids.

---

## Mechanistic Link to Hygroscopy

The depletion zone width delta depends on condensation surface area and vapor transport geometry.

1. **Surface texture (FFT alpha).** Aspergillus conidial balls create a rough surface with more high-frequency structure, increasing the effective condensation area per unit colony area.

2. **Tissue heterogeneity (CV).** The alternating dense/empty pattern in Aspergillus creates a channel network where water vapor becomes confined between opposing condensation surfaces.

3. **Regular spacing (lacunarity).** Aspergillus clusters are uniformly spaced (low lacunarity), ensuring that the channel network covers the entire colony surface. Mucor's irregular gaps leave large unserved regions.

Together: more texture + more channels + regular coverage = wider depletion zone.

---

## Statistical Summary

| Metric | Scale | Asp | Muc | Cohen's d | p |
|---|---|---|---|---|---|
| FFT alpha | Colony surface | -2.960 | -3.092 | 2.21 | 0.013 |
| FFT alpha | 3D macro | -3.221 | -3.460 | 1.41 | 0.012 |
| Local density CV | Light Microscopy | 0.687 | 0.392 | 3.12 | 0.0007 |
| Lacunarity | 3D macro | 1.305 | 2.041 | -1.92 | 0.001 |

All tests: Welch's t (unequal variance). Effect sizes: Cohen's d with pooled SD.

---

## Files in This Folder

| File | Contents |
|---|---|
| final_figure.pdf | Publication figure (5 panels: A-B representative ROIs, C-E boxplots) |
| fft_spectral_slopes.csv | Per-image FFT alpha (14 Asp + 7 Muc, colony surface) |
| fragmentation_results.csv | Per-ROI lacunarity + fragmentation (13 Asp + 11 Muc, 3D) |
| light_micro_top_discriminators.csv | Top 15 light microscopy metrics with BH correction |
| convergent_evidence_table.csv | All significant metrics in one table |
| effect_size_forest_plot.pdf | Supplementary: forest plot of all 15 metrics across 3 scales |
| archive/ | All supporting files (Hessian, Laplacian, full stats, etc.) |

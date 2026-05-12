# FigureHyphae: Hyphal Architecture Analysis

This folder contains the reproducible Figure 4 analysis for the viable `Aspergillus` vs `Mucor` comparison.

`Black`/failed `Rhizopus` is excluded from Figure 4 species-level comparisons because colonies did not establish. It may remain in Figure 3 only as failed-inoculum or low-source data along the universal condensation axis.

## Current Main Result

Absorbing capacity is defined as:

`A = f_tissue x d_structure`

where `f_tissue` is projected tissue fraction and `d_structure` is median distance-transform thickness.

The capacity ratio is 2.01x. The clean viable-fungal dry-zone ratio is 2.13x. The prediction gap is about 6%.

## Current Figure 4 Metrics

- Panel C: FFT spectral slope alpha on 3D colony-surface ROIs.
- Panel D: median tissue thickness from distance transform.
- Panel E: Hessian tubeness CV from light microscopy.
- Panel F: erosion survival from light microscopy.
- Panel G: direction-normalized ratios for four condensation observables and four morphology metrics.
- Panel H: absorbing-capacity decomposition, 2.01x predicted vs 2.13x measured.

## Reproduction

Run the final rebuild script:

```bash
python code/step_final_rebuild_all.py
```

This regenerates the main Figure 4 panels, supplementary figures S18-S22, and `output/framework_validation_comprehensive.csv`, then syncs the handoff package.

Earlier step scripts are retained for provenance:

- `step0_calibrate_segmentation.py`: 3D segmentation calibration.
- `step1_compute_all_metrics.py`: legacy metric computation.
- `step2_tortuosity_test.py`: effective diffusivity/tortuosity negative test.
- `step3_interfacial_area.py`: interfacial and absorbing-capacity metrics.
- `step4_framework_validation.py`: legacy validation; superseded by `step_final_rebuild_all.py` for final Figure 4.
- `step5_fft_crossvalidation.py`: FFT/GLCM cross-validation.

## Key Outputs

- `figures/figure4_panels_CF.{pdf,png,svg}`
- `figures/figure4_panel_GH.{pdf,png,svg}`
- `figures/figS18_fft_analysis.{pdf,png,svg}`
- `figures/figS19_hessian_lm.{pdf,png,svg}`
- `figures/figS20_erosion_demo.{pdf,png,svg}`
- `figures/figS21_tubeness_cv.{pdf,png,svg}`
- `figures/figS22_absorbing_capacity.{pdf,png,svg}`
- `figures/figS_seg3d_sensitivity.{pdf,png,svg}`
- `output/framework_validation_comprehensive.csv`
- `output/segmentation_threshold_sensitivity.csv`

## Reviewer Caveats

- The 3D threshold was calibrated, so cite threshold sensitivity.
- Hessian tubeness CV was selected after exploratory screening, so disclose metric selection.
- Light microscopy sample size is n = 6 vs n = 3; exact Mann-Whitney p-values are bounded at 0.024.
- The absorbing-capacity model is a 2D proxy and should be described as consistency for two viable genera, not universal validation.

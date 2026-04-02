# Fungal Hyphae Reorganize Condensation Fields as Distributed Hygroscopic Sinks

Code and analysis pipeline for reproducing all quantitative figures and tables in the manuscript.

## Repository Structure

```
├── FigureSchematic/        # Figure 1: Schematic panels B, C (droplet count & mean radius)
│   └── code/
│       ├── step0_segment_droplets.py    # Cellpose cyto3 segmentation
│       ├── step1_batch_process.py       # EDT + raycast distance computation
│       ├── step2_beysens_profile.py     # Beysens scaling (β = 1/3, 1)
│       └── step3_figure_panels.py       # Generate panels B, C
│
├── FigureHGAggregate/      # Figure 2: Hydrogel aggregate panels E–L
│   └── code/
│       ├── step1_batch_process.py       # Batch segmentation + distance
│       ├── step2_compute_metrics.py     # Tanh fit, zone metrics, δ
│       ├── step4_panels_BC.py           # Panels F, G (radial profiles)
│       ├── step5_heatmap.py             # Panel C (heatmap, Figure 3)
│       ├── step6_panels_GH.py           # Panels E, H (size gradient)
│       └── test_tracking/
│           ├── track_droplets.py        # Forward-lifetime tracking
│           └── make_manuscript_panels.py # Panels I, J, K, L (survival)
│
├── FigureFungi/            # Figure 3: Fungal panels B, D, E, F, I, J
│   └── code/
│       ├── step1_batch_process.py       # Batch process fungal trials
│       ├── step2_compute_metrics_fungi.py # Fungal-specific metrics
│       ├── step3_panel_B_universal_Rstar.py # Panel B (universal R*)
│       ├── step5_universal_panels.py    # Panels D, E (universal collapse)
│       └── make_panel_delta_strip.py    # Panel F (δ swarm plot)
│
├── FigureRSR/              # Figure 4: Rain-shadow-ridge panels B–G
│   └── code/
│       ├── step1_figure_RSR.py          # Panel B (RSR scatter)
│       ├── step2_rsr_metrics_and_universal_plots.py  # Universal metrics + panels
│       ├── make_panel_E_rsr.py          # Panel E (KM survival)
│       ├── make_panel_zone_metric.py    # Zone metric analysis
│       └── make_rsr_mask_overlays.py    # Mask overlay visualizations
│
├── FigureSupplementary/    # Supplementary Figures S1–S17
│   └── code/
│       ├── supp_common.py               # Shared utilities and paths
│       ├── supp_S1_km_sensitivity.py    # KM sensitivity grid
│       ├── supp_S2_segmentation_validation.py
│       ├── supp_S3_per_trial_KM.py      # Individual KM curves
│       ├── supp_S4_beysens.py           # Beysens profiles (all trials)
│       ├── supp_S5_epsilon.py           # Evaporation rate analysis
│       ├── supp_S6_grid.py              # Universal collapse grid
│       ├── supp_S7_Rd_grid.py           # R(d) scatter grid
│       ├── supp_S8_KM_grid.py           # KM survival grid
│       ├── supp_S9_bootstrap.py         # Bootstrap CIs
│       ├── supp_S10_tanh_diagnostics.py # Tanh fit diagnostics
│       ├── supp_S11_K_analysis.py       # Evaporation coefficient K
│       ├── supp_S12_stats.py            # Statistical summary
│       ├── supp_S13_fungi_Rd_grid.py    # Fungal R(d) grid
│       ├── supp_S14_universality_stats.py
│       ├── supp_S15_bayesian.py         # Bayesian universality
│       ├── supp_S16_blind_prediction.py # Blind prediction validation
│       ├── supp_S17_spectral.py         # Spectral density
│       ├── supp_S17_delta_computation.py
│       ├── supp_Rd_all_trials.py
│       └── supp_delta_raycast.py
│
├── additions/              # Extended statistical analyses
│   ├── 2_ANCOVA_universality/           # ANCOVA + Bayesian universality tests
│   ├── 3_bootstrap_CIs/                # Bootstrap 95% CIs for all regressions
│   ├── 4_K_distance_evaporation/       # Evaporation coefficient K analysis
│   ├── 5_cox_PH_model/                 # Cox proportional hazards model
│   ├── 6_aft_panel_D/                  # Accelerated failure time analysis
│   └── 7_log_beta_visuals/             # Log-beta visualization
│
├── FigureTable/            # Supplementary metrics tables
│   └── code/
│       └── step4_universal_table.py     # Universal comparison table
│
├── Hyphal Analysis/        # Hyphal spacing FFT analysis
│   ├── hyphae_fft_density.py            # FFT transect analysis
│   └── spectral_slope_analysis.py       # Spectral slope statistics
│
├── Figures/                # Assembled publication figures (PDF + SVG)
├── Supplementary/          # Assembled supplementary figures
└── supplementary_information.tex  # LaTeX source for SI
```

## System Requirements

### Operating System
Tested on macOS 14+ (Apple Silicon). Should work on any OS with Python 3.10+.

### Python Dependencies
| Package | Tested Version | Purpose |
|---------|---------------|---------|
| numpy | >= 1.24 | Array operations |
| scipy | >= 1.11 | Curve fitting, EDT, statistics |
| matplotlib | >= 3.8 | Figure generation |
| pandas | >= 2.0 | Data handling |
| scikit-image | >= 0.21 | Morphological operations, regionprops |
| Pillow | >= 10.0 | Image I/O |
| cellpose | >= 3.0 | Droplet segmentation (cyto3 model) |
| lifelines | >= 0.27 | Kaplan-Meier survival analysis |
| h5py | >= 3.8 | HDF5 probability map I/O |
| statsmodels | >= 0.14 | ANCOVA, Cox PH (additions/) |
| scikit-learn | >= 1.3 | Bayesian universality (additions/) |

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Typical install time: ~3 minutes on a standard machine.

## Demo / Reproducing Figures

### Quick demo (figure generation from pre-computed outputs)

Each `Figure*/output/` directory contains pre-computed intermediate data (CSV files) and output figures (SVG/PDF/PNG). To regenerate figures from intermediate data:

```bash
# Figure 1 panels B, C
python FigureSchematic/code/step3_figure_panels.py

# Figure 2 panels F, G
python FigureHGAggregate/code/step4_panels_BC.py

# Figure 2 panels E, H
python FigureHGAggregate/code/step6_panels_GH.py

# Figure 2 panels I, J, K, L
python FigureHGAggregate/code/test_tracking/make_manuscript_panels.py

# Figure 3 panel C (heatmap)
python FigureHGAggregate/code/step5_heatmap.py

# Figure 3 panel B
python FigureFungi/code/step3_panel_B_universal_Rstar.py

# Figure 3 panels D, E
python FigureFungi/code/step5_universal_panels.py

# Figure 3 panel F
python FigureFungi/code/make_panel_delta_strip.py

# Figure 3 panels I, J
python "Hyphal Analysis/spectral_slope_analysis.py"

# Figure 4 panel B
python FigureRSR/code/step1_figure_RSR.py

# Figure 4 panel E (KM survival)
python FigureRSR/code/make_panel_E_rsr.py

# Figure 4 universal metrics
python FigureRSR/code/step2_rsr_metrics_and_universal_plots.py

# Metrics table
python FigureTable/code/step4_universal_table.py
```

Typical run time: < 30 seconds per script on a standard machine.

### Full pipeline (from raw images)

The full pipeline starting from raw microscopy images requires the raw data hosted on Zenodo. Place raw data directories under each `Figure*/raw_data/` and run the `step0`/`step1` scripts first, then proceed with subsequent steps in numerical order.

Steps involving Cellpose segmentation (`step0_segment_droplets.py`) take ~2-5 minutes per trial on a machine with GPU; ~10-20 minutes on CPU only.

## Key Algorithms

1. **Droplet segmentation**: Cellpose cyto3 model (diameter = 90 px, flow_threshold = 0.2, cellprob_threshold = 1.0) with 3 px morphological erosion
2. **Distance computation**: Euclidean distance transform (EDT) from source boundary; raycast method (100 boundary samples) for dry-zone width delta
3. **Radial profile fitting**: Hyperbolic tangent model R(d) for size transition; broken-stick regression R(r') for size gradient
4. **Survival analysis**: Forward-lifetime Kaplan-Meier from seed time t = 15 min; coalescence events right-censored
5. **Beysens scaling**: log-log R(t) with beta = 1/3 (diffusion-limited) and beta = 1 (coalescence-dominated)

## Raw Data

Raw microscopy images (*.jpg, *.tif), ilastik probability maps (*.h5), and NPY segmentation masks are hosted on Zenodo. Processed CSV data (in `raw_data/aggregate_edt/` and `output/` folders) are included in this repository and sufficient to reproduce all figure panels without external data.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19167978.svg)](https://doi.org/10.5281/zenodo.19167978)

## License

This project is licensed under the MIT License -- see [LICENSE](LICENSE) for details.

## Citation

If you use this code, please cite:

> Lin, Y.J., Feng, L., Khan, A., Park, K.-C. & Jung, S. Fungal Hyphae Reorganize Condensation Fields as Distributed Hygroscopic Sinks. *Nature Communications* (2025). https://doi.org/10.5281/zenodo.19167978

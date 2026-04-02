# Supplementary Figures — Index & Manuscript Reference Map

Scripts in `FigureSupplementary/code/`. Shared utilities in `supp_common.py`.
All outputs PDF + SVG only. Data derived from aggregate_edt CSVs and track histories.

**6 supplementary figures (S1–S6).**

---

## S1 — KM Sensitivity Analysis
**File:** `FigureS1_km_sensitivity.pdf`
**Script:** `supp_S1_km_sensitivity.py`
**Data:** `FigureHGAggregate/code/test_tracking/output/`
**Shows:** 3×3 grid (panels a–i) of τ₅₀(d) profiles under 9 parameter combos (bin width 100/200/400 µm × min tracks 5/10/30). Canonical (200 µm, 10 tracks, blue border) matches main analysis. Confirms τ₅₀ estimates are not sensitive to binning choices.
**Manuscript reference:** Methods §Survival-time gradient — *"τ₅₀ estimates were robust to bin-width and minimum-count choices (Supplementary Fig. S1)."*

---

## S2 — Beysens Growth Profiles
**File:** `FigureS2_beysens_profiles.pdf`
**Script:** `supp_S4_beysens.py`
**Data:** `FigureHGAggregate/raw_data/aggregate_edt/`, `FigureFungi/raw_data/aggregate_edt/`
**Shows:** 7-panel column (panels a–g), one exemplar per condition. Log-log R(t) coloured by distance r' with fitted power-law lines and β=1/3 / β=1 reference lines. Validates d²-law growth across all 7 conditions.
**Manuscript reference:** Results §Time-resolved dynamics — *"Droplet growth follows diffusion-limited t^(1/3) scaling across all conditions (Supplementary Fig. S2)."*

---

## S3 — R(d) Profiles — All 35 Trials
**File:** `FigureS3_Rd_all_trials.pdf`
**Script:** `supp_Rd_all_trials.py`
**Data:** `FigureHGAggregate/raw_data/aggregate_edt/`, `FigureFungi/raw_data/aggregate_edt/`
**Shows:** 7×5 grid (7 conditions × 5 replicates = 35 panels). Each panel: ghost scatter + mean ± SEM band + Δ marker. Trial ID top-right. Demonstrates spatial R(d) gradient reproducibility across all lab trials (both hydrogels and fungi).
**Manuscript reference:** Results §Three metrics scale linearly / Universal collapse — *"R(d) profiles for all 35 lab trials are shown in Supplementary Fig. S3."*

---

## S4 — KM Survival Curves (Distance-Stratified, All 20 Hydrogel Trials)
**File:** `FigureS4_KM_survival_grid.pdf`
**Script:** `supp_S8_KM_grid.py`
**Data:** `FigureHGAggregate/code/test_tracking/output/`
**Shows:** 4×5 grid of KM survival curves per hydrogel trial, stratified by 4 distance bands (0.9/1.5/2.1/2.9 mm). Dots mark τ₅₀. Extends Fig. 2I–J to all 20 trials — the within-trial τ₅₀ distance gradient is present in every replicate.
**Manuscript reference:** Results §Three metrics scale linearly — *"Distance-stratified KM survival curves for all 20 hydrogel trials are shown in Supplementary Fig. S4."*

---

## S5 — Bootstrap Confidence Intervals
**File:** `FigureS5_bootstrap_CIs.pdf`
**Script:** `supp_S9_bootstrap.py`
**Data:** `additions/3_bootstrap_CIs/`
**Shows:** Bootstrap 95% CIs (N = 10,000 resamples) for calibration regression slopes (Figs. 2F, 2H, 2K, 2L). Confirms all regression estimates are statistically robust.
**Manuscript reference:** Methods §Statistical analysis — *"Bootstrap confidence intervals on calibration slopes (N = 10,000 resamples; Supplementary Fig. S5)."*

---

## S6 — Delta Raycast Methodology
**File:** `FigureS6_delta_raycast.pdf`
**Script:** `supp_delta_raycast.py`
**Data:** `{HG,Fungi}/raw_data/aggregate_edt/{tid}_edt_droplets.csv` + `{tid}_boundary_polygon.csv`
**Shows:** (a–c) Droplet scatter coloured by distance, boundary polygon, and raycast lines for three exemplar hydrogel trials (agar Δ≈100 µm, 1:1 NaCl Δ≈421 µm, 2:1 NaCl Δ≈681 µm) — illustrates how Δ grows with water-activity depression. (d) Bar chart of Δ for all 35 trials.
**Manuscript reference:** Methods §Dry-zone width — *"Δ raycast methodology and values for all 35 trials are shown in Supplementary Fig. S6."*

---

## SupplementaryRSR.pdf (unnumbered standalone)
Extended RSR field figure. Reference as *"Supplementary RSR Data"* in main text.

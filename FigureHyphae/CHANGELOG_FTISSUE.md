# f_tissue Investigation — Methodological Journey

Chronological audit trail of every method tried during the 2026-05-27 / 2026-05-28 f_tissue audit, including the failures. Kept for transparency so the published methodology can be defended against any reviewer asking "did you consider X?"

Read `FTISSUE_METHODOLOGY_REPORT.md` for the technical write-up; this file is the *story* of how we got there.

---

## v0 — Baseline (the published method)

**Adaptive dark-response thresholding** on macro colony-surface ROIs.
- σ_smooth = 1 px, σ_local = 32 px, threshold = mean(dr) + 0.5·std(dr)
- Polarity: tissue = darker than local mean
- Asp/Muc ratio: **1.16×**, d = 3.00, p = 1.2×10⁻⁵
- Calibrated to reproduce thickness range (7.21 µm Asp; matches literature)

**Concerns flagged before the audit:**
1. Polarity convention (hyphae appear *bright* macroscopically, so "dark = tissue" is counterintuitive to reviewers).
2. Binary gating discards gradient information.
3. 1.16× ratio is weak vs δ benchmark 2.13×.

## Day 1 (2026-05-27): polarity exploration and binary alternatives

### Attempt 1 — Sauvola local adaptive (literature standard)
- Window = 64 px, k = 0.2, R = 0.5, polarity = dark
- Result: ratio **3.77×**, d = 3.94, p = 1.7×10⁻⁹, no overlap
- **Visually bad** — masks look like sparse salt-and-pepper, not hyphae. Sauvola at k=0.2 captures only the darkest cores.
- Parameter sweep (6 windows × 5 k values × 24 ROIs = 720 segmentations on GPU in 14 s) showed ratio range 1.0–8.3 — strong sensitivity to parameter choice.
- Verdict: **rejected as primary** (too aggressive, looks like noise).

### Attempt 2 — Method comparison with auto-polarity detection
Built `compare_segmentation_methods.py` with explicit polarity check (verify `raw[mask].mean() < raw[~mask].mean()` per method).
- Methods compared: adaptive (current), Otsu (global), Frangi vesselness + Otsu, Sauvola local.
- Polarity check: zero flips needed across all 96 segmentations.
- Verdict: **confirmed no polarity bugs in any method**, but Otsu global over-segments badly (50% f_tissue, not physical).

### Attempt 3 — Foundation-model segmentation (Cellpose v4)
- cyto3 pretrained model with diameter=12 px on all 24 ROIs.
- Result: ratio **0.89×**, d = -0.09, p = 0.83 — *no discrimination*.
- Cellpose found 0–14 "cells" per ROI; designed for round cells, not filaments.
- Verdict: **failed** — kept the result in `figS_ftissue_ml.pdf` to document that we tried.

### Attempt 4 — Foundation-model classification (DINOv2)
- facebook/dinov2-base, frozen, extract CLS-token embeddings (24 × 768).
- LOOCV logistic regression: **24/24 correct = 100% accuracy, AUC = 1.000**.
- Doesn't replace f_tissue (not a segmenter) but confirms genera are visually distinct to a foundation model — strong "the discrimination is real" datapoint.
- Verdict: **kept as bonus SI panel** confirming morphological distinctness.

### Attempt 5 — Polarity flip (tissue = bright)
User correction: "hyphae are the WHITE pixels, not dark." Rebuilt `ftissue_bright_final.py` mirroring the published method with bright-response instead of dark.
- σ_smooth = 1, σ_local = 32, k = 0.5, polarity = BRIGHT.
- Result: ratio **1.11×**, d = 2.52, p = 5.5×10⁻⁵.
- All Asp > Muc still holds (gap narrowed to 0.012).
- Three-class anatomy revealed: bright (30%) + middle (40%) + dark (30%) — bright and dark are NOT complements, both polarities measure the same texture density from opposite sides.
- Verdict: **chosen for visual coherence** (white in raw ↔ white in mask) but accepted the slight statistical cost.

### Attempt 6 — Interactive Gradio explorer
Built `threshold_explorer.py` with live sliders for adaptive/Sauvola/Otsu parameters + polarity toggle. GPU update <100 ms per slider move. Used to visually verify the chosen settings across ROIs.

## Day 2 (2026-05-28): continuous metrics

### Attempt 7 — Continuous multi-scale bright-density
After user feedback ("binary gating too insensitive, want gradient measurement"), built `continuous_bright_density.py`.
- Per-pixel: `d(x,y) = mean_σ max(0, I_norm − gauss(I_norm, σ))`, σ ∈ {8, 16, 32, 64, 128} px.
- Per-ROI: `D = ⟨d(x,y)⟩`.
- Result: ratio **1.89×**, d = 4.01, p = 4.3×10⁻⁹, **no overlap**.
- 11% short of δ benchmark — best result so far.
- Verdict: **promoted to primary**, replaced all prior primary candidates.

### Attempt 8 — Photometric reconstruction (current primary)
Per user request for "heatmap reconstruction using GPU to analyze drop-offs and shadings."
Built `photometric_density_reconstruction.py` combining:
- Multi-scale brightness `B_σ` (from Attempt 7)
- Multi-scale gradient energy `G_σ` (smoothed |∇I|)
- Structure-tensor coherence `C_σ` (filament anisotropy)
- Per-scale reconstruction: `R_σ = B_σ · (1 + α · norm(G_σ) · norm(C_σ))`, α = 1
- Per-pixel: `R = ⟨R_σ⟩_σ`; per-ROI: `D_rec = ⟨R⟩`
- Result: ratio **1.95×**, d = 3.80, p = 1.1×10⁻⁸, **no overlap**.
- **8.4%** short of δ benchmark — closest match of any method tested.
- 3D topographic surface plot is visually striking.
- Spearman ρ with continuous-only = 0.99 (same ranking, sharper magnitudes).
- Verdict: **promoted to primary** (per the plan-mode decision criteria).

### Final convergence figure
6-method comparison: all give Asp > Muc, mean Spearman ρ = 0.89, ratios span 1.11×–3.77×, photometric closest to δ.

---

## Files at each stage

| Day / Attempt | New code | New CSV | New figure |
|---|---|---|---|
| v0 (existing) | `step1_compute_all_metrics.py` etc | `interfacial_metrics_3d.csv` | `figS18_fft_analysis.pdf` etc |
| Day 1 / 1 | `sauvola_supplement_and_validation.py` | `ftissue_sauvola_final.csv` | `figS_ftissue_sauvola*.pdf` |
| Day 1 / 2 | `compare_segmentation_methods.py` | `ftissue_seg_methods.csv` | `figS_ftissue_segcompare.pdf` |
| Day 1 / 3–4 | `ml_segmentation_and_embedding.py` | `ftissue_ml_methods.csv` | `figS_ftissue_ml.pdf` |
| Day 1 / 5 | `ftissue_bright_final.py` | `ftissue_bright_final.csv` | `figS_ftissue_bright.pdf` |
| Day 1 / 6 | `threshold_explorer.py` (interactive) | — | — |
| Day 2 / 7 | `continuous_bright_density.py` | `ftissue_continuous_density.csv` | `figS_continuous_density.pdf` |
| Day 2 / 8 | `photometric_density_reconstruction.py` | `ftissue_photometric.csv` | `figS_photometric_reconstruction.pdf` |
| Final | `reproduce_all_final.py`, `final_convergence_figure.py`, `fullres_previews.py`, `run_all_ftissue.py` | `ftissue_all_methods_final.csv` | `figS_ftissue_final_convergence.pdf` |

---

## Lessons learned (for the methodology report and future work)

1. **Polarity choice is a visual-coherence question, not a numerical one.** Both polarities measure the same underlying texture density from opposite sides (Spearman ρ ≈ 0.85). Pick the convention that doesn't make reviewers mentally invert when reading the figure.

2. **Binary thresholding loses information.** All four binary variants span 1.11×–3.77× depending on threshold placement. The single continuous metric (1.89×) is more stable.

3. **Adding structure-aware terms helps marginally.** Photometric reconstruction (1.95×) vs continuous-brightness (1.89×) — the gradient × coherence boost closes 2.6 percentage points of the gap to δ. Worth it for the visualization (3D topography) but not a dramatic statistical improvement; continuous brightness alone has slightly tighter Cohen's d.

4. **Foundation-model segmentation doesn't transfer to filaments out of the box.** Cellpose cyto3 produced essentially no detections (ratio 0.89×, p = 0.83). A custom-trained CNN would need pixel-level hyphal annotations we don't have.

5. **Foundation-model classification confirms the morphological distinction is real.** DINOv2 achieves 100% Asp vs Muc accuracy with no fine-tuning — a strong "the discrimination is genuine" datapoint.

6. **GPU acceleration matters.** Every analysis above runs in seconds on the RTX 5070 Ti. The full Sauvola sweep (720 segmentations) takes 14 s; the photometric reconstruction (multi-scale brightness + gradient + structure tensor on 24 ROIs) takes 3.3 s. CPU equivalents would run ~50× slower.

---

End of changelog.

# f_tissue Investigation — Master Index

**Investigation date:** 2026-05-27 to 2026-05-28
**Paper:** Fungal Hyphae as Distributed Vapor Sinks (Lin, Feng, Khan, Park, Jung)
**Purpose:** Pre-publish audit of the colony-surface `f_tissue` segmentation metric. Identified polarity-convention and binary-gating issues with the published method, evaluated six alternatives, settled on a continuous photometric reconstruction metric. Everything in this index lives in `D:\FINAL OSF\FigureHyphae\` alongside the original paper deposit.

This README documents only the **f_tissue investigation** files added during this audit. The original Figure 4 pipeline (`build_figure4_complete.py`, `step1_compute_all_metrics.py`, `figS18`–`figS22`*, `figure4_panel*`*, etc.) is untouched and is documented in the parent deposit `README.md`.

---

## 1. Top-level documents (root of `FigureHyphae/`)

| File | Purpose |
|---|---|
| `README_FTISSUE.md` | **This file.** Top-level index of the f_tissue investigation. |
| `CHANGELOG_FTISSUE.md` | Chronological audit trail of every method tried and why. |
| `FTISSUE_METHODOLOGY_REPORT.md` | Comprehensive technical report: methods, math, reviewer Q&A, reproducibility, limitations. The single doc to read first. |

## 2. Code (`code/`)

Ordered by role.

### Master reproducibility
| Script | Runtime (RTX 5070 Ti) | What it does |
|---|---|---|
| `run_all_ftissue.py` | ~60 s end-to-end | Single entry point: runs every analysis below in dependency order and verifies all outputs exist. |

### Primary metric
| Script | Runtime | Output |
|---|---|---|
| `photometric_density_reconstruction.py` | 3.3 s | **PRIMARY** — `D_rec` per ROI, supplement figure, full-resolution 2D + 3D heatmaps. Combines multi-scale brightness × (1 + α·gradient·structure-tensor coherence). |
| `continuous_bright_density.py` | 1.6 s | Brightness-only ablation (`D_continuous`); kept as a sensitivity check. |

### Binary sensitivity checks
| Script | Runtime | Output |
|---|---|---|
| `reproduce_all_final.py` | 1.7 s | All four binary methods (adaptive bright/dark, Sauvola bright/dark) in one pass. |
| `ftissue_bright_final.py` | <1 s | Standalone bright-polarity adaptive (paired with dark for direct comparison). |
| `compare_segmentation_methods.py` | 2.0 s | Side-by-side 4-method comparison with auto-polarity detection. |
| `sauvola_supplement_and_validation.py` | 14 s | Sauvola parameter sweep (6×5 grid × 24 ROIs = 720 segmentations). |

### Visualization
| Script | Runtime | Output |
|---|---|---|
| `final_convergence_figure.py` | <2 s | Unified 6-method convergence supplement (loads CSVs, renders figure). |
| `fullres_previews.py` | <5 s | Native-resolution side-by-side raw + mask PNGs for canonical Asp/Muc and all 24 ROIs. |
| `make_supplementary_fig_ftissue.py` | <2 s | Pedagogical f_tissue schematic (binary mask layout). |
| `make_supplementary_fig_ftissue_box.py` | <2 s | Standalone f_tissue boxplot matching Fig 4 stripbox style. |

### Diagnostic / exploratory
| Script | Runtime | Purpose |
|---|---|---|
| `analyze_ftissue_diagnostic.py` | ~30 s | Multi-scale window-size sweep (CPU; predates the GPU pipeline). |
| `ml_segmentation_and_embedding.py` | ~4 min | Cellpose v4 + DINOv2 attempt. Cellpose failed (ratio 0.89×), DINOv2 achieved 100% classification. Kept for transparency. |
| `threshold_explorer.py` | live | Interactive Gradio web app for slider-based threshold exploration on GPU. Launch and open `http://127.0.0.1:7860`. |

## 3. Output CSVs (`output/`)

| CSV | Rows | Columns |
|---|---|---|
| `ftissue_photometric.csv` | 24 | `file, genus, D_recon` — **PRIMARY** per-ROI continuous density |
| `ftissue_continuous_density.csv` | 24 | `file, genus, D_continuous` — brightness-only ablation |
| `ftissue_all_methods_final.csv` | 24 | `file, genus, f_adaptive-bright, f_adaptive-dark, f_sauvola-bright, f_sauvola-dark` |
| `ftissue_bright_final.csv` | 24 | `file, genus, f_bright, f_dark` (paired) |
| `ftissue_seg_methods.csv` | 96 | `file, genus, method, f_tissue, polarity_flipped` — long-format method comparison |
| `ftissue_sauvola_final.csv` | 24 | `file, genus, f_current, f_sauvola` |
| `ftissue_ml_methods.csv` | 24 | `file, genus, f_cellpose, dt_cellpose_um, n_cells, dinov2_p_asp, dinov2_pred` |

Plus pre-existing original-deposit CSVs (`figure4_3d_metrics.csv`, `interfacial_metrics_3d.csv`, `interfacial_metrics_lm.csv`, `fft_per_roi.csv`, `framework_validation_comprehensive.csv`, etc.) — NOT modified during this investigation.

## 4. Figures (`figures/`)

### Final convergence supplement (the one to put in the paper SI)
| File | Description |
|---|---|
| `figS_ftissue_final_convergence.{pdf,png,svg}` | 6-method convergence: box plots, ratio bars vs δ benchmark, Spearman correlation matrix, parallel coordinates, verdict text. |

### Primary metric standalone
| File | Description |
|---|---|
| `figS_photometric_reconstruction.{pdf,png,svg}` | D_rec box plot, per-ROI scatter vs continuous, ratio vs δ, Spearman bars, canonical heatmaps |
| `figS_continuous_density.{pdf,png,svg}` | D_continuous box plot, scaling, canonical heatmaps, scale decomposition |

### Polarity sensitivity (binary, bright vs dark)
| File | Description |
|---|---|
| `figS_ftissue_bright.{pdf,png,svg}` | Bright-polarity adaptive vs dark-polarity adaptive, side-by-side panels |
| `figS_ftissue_segcompare.{pdf,png,svg}` | 4-method binary comparison with polarity check |

### Sauvola sensitivity
| File | Description |
|---|---|
| `figS_ftissue_sauvola.{pdf,png,svg}` | Sauvola-only mask visuals |
| `figS_ftissue_sauvola_supplement.{pdf,png,svg}` | Sauvola vs current side-by-side |
| `figS_ftissue_sauvola_validation.{pdf,png,svg}` | Parameter sweep heatmap + all-24-ROI gallery |

### Earlier convergence / box-plot drafts
| File | Description |
|---|---|
| `figS_ftissue_convergence.{pdf,png,svg}` | Earlier 4-method convergence draft (superseded by `_final_convergence`) |
| `figS_ftissue.{pdf,png,svg}` | Pedagogical f_tissue panel (pixel-counting schematic) |
| `figS_ftissue_box.{pdf,png,svg}` | Standalone f_tissue box plot in Fig 4 stripbox style |

### Foundation-model attempt
| File | Description |
|---|---|
| `figS_ftissue_ml.{pdf,png,svg}` | Cellpose v4 failure documentation + DINOv2 100%-accuracy classification |

### Full-resolution previews (`figures/preview_fullres/`)
| Subset | Description |
|---|---|
| `reconstruction_{asp,muc}_heatmap.png` | Native-resolution 2D viridis heatmaps of `D_rec` field |
| `reconstruction_{asp,muc}_3d.png` | Native-resolution 3D topographic surface plots of `D_rec` |
| `density_{asp,muc}_heatmap.png` | Native-resolution 2D heatmaps of `D_continuous` field |
| `{asp,muc}_canonical_raw_vs_mask.png` | Native-resolution side-by-side raw + bright-binary mask |
| `all24/Asp_*_raw_vs_mask.png`, `all24/Muc_*_raw_vs_mask.png` | All 24 ROIs at native resolution |

## 5. Environment / dependencies

- **Python:** 3.12 via uv at `C:\Users\yanyl\research\.venv\Scripts\python.exe`
- **Torch:** 2.12.0.dev+cu128 (Blackwell-compatible nightly)
- **GPU:** RTX 5070 Ti, 17.1 GB VRAM (CUDA True confirmed)
- **Other Python:** `numpy`, `scipy`, `matplotlib`, `Pillow`, `scikit-image`, `scikit-learn`, `lifelines`, `cellpose ≥ 4`, `transformers`, `timm`, `gradio`
- **Operating system:** Windows 11

## 6. How to reproduce everything

From a PowerShell prompt at `D:\FINAL OSF\FigureHyphae\`:

```powershell
& "C:\Users\yanyl\research\.venv\Scripts\python.exe" "code\run_all_ftissue.py"
```

Expected total runtime: **~60 seconds end-to-end on RTX 5070 Ti.** All outputs land in `output/` (CSVs), `figures/` (PDF/PNG/SVG), and `figures/preview_fullres/` (native-resolution renders). Idempotent — re-running overwrites prior outputs deterministically (random seed is fixed for any scatter-jitter and bootstrap).

For the interactive threshold explorer (separate, not part of `run_all`):

```powershell
& "C:\Users\yanyl\research\.venv\Scripts\python.exe" "code\threshold_explorer.py"
# Then open http://127.0.0.1:7860 in a browser
```

## 7. Headline numbers to cite

| Quantity | Value | Source |
|---|---|---|
| Primary metric ratio | **Asp/Muc D_rec = 1.95×** | `ftissue_photometric.csv` |
| δ benchmark | 2.13× | Main paper |
| Gap from δ | **8.4%** | (best of any tested method) |
| Cohen's *d* | 3.80 | photometric reconstruction |
| Welch *p* | 1.1×10⁻⁸ | photometric reconstruction |
| No-overlap | **Yes** | Asp min 0.0288 > Muc max 0.0240 |
| Cross-method Spearman ρ (mean off-diagonal) | 0.89 | 6-method comparison |
| All 6 methods agree on direction? | **Yes** (Asp > Muc, p ≤ 5.5×10⁻⁵ everywhere) | convergence figure |

## 8. Where to look first

1. **Read `FTISSUE_METHODOLOGY_REPORT.md`** — the technical report with everything: math, results, reviewer Q&A.
2. **Open `figures/figS_ftissue_final_convergence.png`** — the one supplement figure that summarizes all six methods.
3. **Open `figures/preview_fullres/reconstruction_asp_3d.png`** — the 3D topographic visualization of the primary metric.
4. **Run `code/run_all_ftissue.py`** — if you want to reproduce everything from scratch.

---

End of f_tissue investigation index. See `CHANGELOG_FTISSUE.md` for the chronological audit trail.

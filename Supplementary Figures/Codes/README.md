# Supplementary figure regeneration scripts

Scripts to regenerate `SuppFigN.pdf` files in the parent directory.

## Mapping

| SuppFig | Source script | Notes |
|---|---|---|
| 1 | (manual) | Chamber schematic — drawn in Illustrator |
| 2 | `supp_Rd_all_trials.py` | Per-trial R(d) profiles, 35 trials |
| 3 | `supp_S4_beysens.py` | Droplet growth curves, 7 conditions |
| 4 | `supp_S8_KM_grid.py` | KM survival curves, 20 hydrogel trials |
| 5 | `supp_S1_km_sensitivity.py` | τ₅₀ binning sensitivity |
| 6 | `supp_S9_bootstrap.py` | Bootstrap CIs on Fig. 2 regressions |
| 7 | `supp_delta_raycast.py` | Dry-zone raycast methodology |
| 8 | (manual) | Field RSR scatter — assembled in Illustrator |
| 9–14 | `hyphae_SuppFig9to14.py` | Hyphae morphology supplementary figures |
| 15 | `supp_S18_chamber_stability.py` | Chamber temperature/humidity stability |

## Dependencies

- Python 3.11+
- `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-image`, `Pillow`, `lifelines`

## How to run

From this directory:

```bash
python supp_Rd_all_trials.py    # SuppFig2
python supp_S4_beysens.py       # SuppFig3
# ...etc
python hyphae_SuppFig9to14.py   # SuppFigs 9–14
```

`supp_common.py` is a shared helper imported by every `supp_*.py` script.

`hyphae_SuppFig9to14.py` is self-contained; it writes outputs to
`../../FigureHyphae/figures/` (hard-coded paths inside the script).
The other `supp_*.py` scripts write to a sibling `_validation_output/`
directory under `supplementary_figures/`.

## Reproducibility note

All scripts produce pixel-identical outputs to the shipped PDFs except:
- **SuppFig6** (`supp_S9_bootstrap.py`): bootstrap uses a non-fixed RNG seed,
  so the confidence-band geometry shifts very slightly each run. The
  regression statistics and bands are statistically identical.

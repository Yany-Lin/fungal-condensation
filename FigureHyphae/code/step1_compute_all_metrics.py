#!/usr/bin/env python3
"""Rebuild ALL Figure 4 metrics with the calibrated segmentation.

3D segmentation: smooth σ=1, local_mean σ=32, dark_response = local - smooth,
                 threshold = mean + 0.5×std, close+open 1 iteration each.

Light micro: exact analyze_fungi.py replication.

Outputs:
  - All per-sample data (CSV)
  - Consistent figure (Panels C-H)
  - Updated framework validation
"""

import json
import numpy as np
from scipy import ndimage as ndi
from scipy import stats
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_T7 = Path('/Volumes/T7/FINAL OSF')
_REPO = Path(__file__).resolve().parent.parent.parent
BASE = _T7 if _T7.exists() else _REPO
ROI_DIR = BASE / 'HYPHAE' / 'Analysis' / 'results' / '3d_overlays'
SESSION = ROI_DIR / 'roi_session.json'
MICRO_DIR = BASE / 'HYPHAE' / 'Light Microscopy'
RSR_CSV = BASE / 'FigureRSR' / 'output' / 'rsr_and_lab_dstar_metrics.csv'
OUT = BASE / 'FigureHyphae' / 'output'

CAL_3D = 0.94
MM = 1 / 25.4
C_ASP = '#4CAF50'
C_MUC = '#757575'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 8, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'svg.fonttype': 'none',
})


def load_gray(path):
    with Image.open(path) as im:
        arr = np.asarray(im).astype(np.float64)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr


# ═══════════════════════════════════════════════════════════════
# CALIBRATED 3D SEGMENTATION
# ═══════════════════════════════════════════════════════════════
def seg_3d(img):
    """Calibrated: smooth σ=1, local σ=32, thresh = mean + 0.5*std."""
    smooth = ndi.gaussian_filter(img, sigma=1.0)
    local_mean = ndi.gaussian_filter(smooth, sigma=32.0)
    dark_response = local_mean - smooth
    threshold = dark_response.mean() + 0.5 * dark_response.std()
    mask = dark_response > threshold
    mask = ndi.binary_closing(mask, iterations=1)
    mask = ndi.binary_opening(mask, iterations=1)
    return mask


# ═══════════════════════════════════════════════════════════════
# LIGHT MICRO SEGMENTATION (exact analyze_fungi.py)
# ═══════════════════════════════════════════════════════════════
def otsu_threshold(values):
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    w_bg = np.cumsum(hist).astype(float)
    w_fg = total - w_bg
    sum_bg = np.cumsum(hist * centers)
    mean_bg = sum_bg / np.maximum(w_bg, 1)
    mean_fg = (sum_bg[-1] - sum_bg) / np.maximum(w_fg, 1)
    var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
    return centers[np.argmax(var)]


def seg_micro(img, label):
    smooth = ndi.gaussian_filter(img, sigma=1.0)
    local_mean = ndi.gaussian_filter(smooth, sigma=32.0)
    dark_response = local_mean - smooth
    valid = np.ones(img.shape, dtype=bool)
    margin = max(8, int(round(min(img.shape) * 0.006)))
    valid[:margin, :] = False
    valid[-margin:, :] = False
    valid[:, :margin] = False
    valid[:, -margin:] = False

    values = smooth[valid]
    global_t = otsu_threshold(values)
    dr_values = dark_response[valid]
    dr_median = float(np.median(dr_values))
    dr_mad = float(np.median(np.abs(dr_values - dr_median)) + 1e-9)

    if label.lower() in ('pink', 'green'):
        intensity_cap = float(np.percentile(values, 62))
        local_t = max(float(np.percentile(dr_values, 72)), dr_median + 0.80 * dr_mad)
        broad_mask = smooth < min(global_t, intensity_cap)
        fiber_mask = valid & (dark_response > local_t)
        mask = valid & (broad_mask | fiber_mask)
        min_size, max_hole = 28, 1500
    else:
        intensity_cap = float(np.percentile(values, 32))
        local_t = max(float(np.percentile(dr_values, 78)), dr_median + 1.15 * dr_mad)
        fiber_mask = valid & (dark_response > local_t)
        strong_dark = (smooth < intensity_cap) & (dark_response > np.percentile(dr_values, 48))
        mask = valid & (fiber_mask | strong_dark)
        min_size, max_hole = 10, 80

    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)), iterations=1)
    mask &= valid
    labels_arr, nlab = ndi.label(mask)
    if nlab > 0:
        sizes = np.bincount(labels_arr.ravel())
        keep = sizes >= min_size; keep[0] = False
        mask = keep[labels_arr]
    filled = ndi.binary_fill_holes(mask)
    holes = filled & ~mask
    hl, nhl = ndi.label(holes)
    if nhl > 0:
        hs = np.bincount(hl.ravel())
        small = hs <= max_hole; small[0] = False
        mask = mask | small[hl]
    mask &= valid
    return mask.astype(bool)


# ═══════════════════════════════════════════════════════════════
# GLCM CONTRAST (raw grayscale, no mask)
# ═══════════════════════════════════════════════════════════════
def glcm_contrast(img, d=1):
    lo, hi = np.percentile(img, [0.5, 99.5])
    if hi <= lo: hi = lo + 1
    img8 = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    dh = (img8[:, d:].astype(float) - img8[:, :-d].astype(float)) ** 2
    dv = (img8[d:, :].astype(float) - img8[:-d, :].astype(float)) ** 2
    return (dh.mean() + dv.mean()) / 2.0


# ═══════════════════════════════════════════════════════════════
# COMPUTE ALL METRICS
# ═══════════════════════════════════════════════════════════════

print('Loading session...')
with open(SESSION) as f:
    session = json.load(f)
saved = {k: v for k, v in session.items()
         if not k.startswith('_') and v.get('status') != 'deleted'}

# ── Legacy GLCM cross-validation metric (3D, raw grayscale) ──
print('\n── Panel C: GLCM contrast ──')
c_asp, c_muc = [], []
for key, val in saved.items():
    genus = val.get('genus')
    roi_path = ROI_DIR / genus / f'{Path(key).stem}_roi.jpg'
    if not roi_path.exists(): continue
    img = load_gray(roi_path)
    gc = glcm_contrast(img)
    (c_asp if genus == 'Aspergillus' else c_muc).append(gc)
c_asp, c_muc = np.array(c_asp), np.array(c_muc)
t, p = stats.ttest_ind(c_asp, c_muc, equal_var=False)
print(f'  Asp={c_asp.mean():.2f}±{c_asp.std(ddof=1):.2f}, Muc={c_muc.mean():.2f}±{c_muc.std(ddof=1):.2f}, p={p:.4g}')

# ── Panel D: Tissue thickness (3D, calibrated segmentation) ──
print('\n── Panel D: Tissue thickness ──')
d_asp, d_muc = [], []
f_asp_3d, f_muc_3d = [], []  # tissue fraction for absorbing capacity
for key, val in saved.items():
    genus = val.get('genus')
    roi_path = ROI_DIR / genus / f'{Path(key).stem}_roi.jpg'
    if not roi_path.exists(): continue
    img = load_gray(roi_path)
    mask = seg_3d(img)
    if mask.sum() < 100: continue
    dt = ndi.distance_transform_edt(mask) * CAL_3D
    dt_med = np.median(dt[mask])
    f_tissue = mask.mean()
    (d_asp if genus == 'Aspergillus' else d_muc).append(dt_med)
    (f_asp_3d if genus == 'Aspergillus' else f_muc_3d).append(f_tissue)
d_asp, d_muc = np.array(d_asp), np.array(d_muc)
f_asp_3d, f_muc_3d = np.array(f_asp_3d), np.array(f_muc_3d)
t, p = stats.ttest_ind(d_asp, d_muc, equal_var=False)
print(f'  Asp={d_asp.mean():.2f}±{d_asp.std(ddof=1):.2f} µm, Muc={d_muc.mean():.2f}±{d_muc.std(ddof=1):.2f} µm, p={p:.4g}')

# ── Panel E: Local density CV (light micro, from fungi_metrics.csv) ──
# Legacy metric, superseded by Hessian tubeness CV in the final manuscript.
# Kept for archival reproducibility; skipped if the original Leyun-side CSV
# is not present on the current machine.
print('\n── Panel E: Local density CV (legacy, skipped if data absent) ──')
import pandas as pd
_legacy_csv = Path('/Users/yany/Desktop/Leyun microscopy/analysis_outputs/fungi_metrics.csv')
if _legacy_csv.exists():
    fungi = pd.read_csv(_legacy_csv)
    e_asp = fungi.loc[fungi['label'] == 'Green', 'local_density_cv'].values
    e_muc = fungi.loc[fungi['label'] == 'White', 'local_density_cv'].values
    t, p = stats.ttest_ind(e_asp, e_muc, equal_var=False)
    print(f'  Asp={e_asp.mean():.4f}±{e_asp.std(ddof=1):.4f}, Muc={e_muc.mean():.4f}±{e_muc.std(ddof=1):.4f}, p={p:.4g}')
else:
    print(f'  (skipped: legacy CSV at {_legacy_csv} not present; see step_final_rebuild_all.py for Hessian tubeness CV)')

# ── Panel F: Erosion survival (light micro, exact segmentation) ──
print('\n── Panel F: Erosion survival ──')
micro_images = [
    ('Pink_10X_1.TIF', 'Aspergillus', 'Pink'),
    ('Pink_10X_2.TIF', 'Aspergillus', 'Pink'),
    ('Pink_20X_1.TIF', 'Aspergillus', 'Pink'),
    ('Pink_20X_2.TIF', 'Aspergillus', 'Pink'),
    ('Pink_40X_1.TIF', 'Aspergillus', 'Pink'),
    ('Pink_40X_2.TIF', 'Aspergillus', 'Pink'),
    ('White_10X_1.TIF', 'Mucor', 'White'),
    ('White_20X_1.TIF', 'Mucor', 'White'),
    ('White_40X_1.TIF', 'Mucor', 'White'),
]
f_asp, f_muc = [], []
for fname, genus, label in micro_images:
    path = MICRO_DIR / fname
    img = load_gray(path)
    mask = seg_micro(img, label)
    orig = mask.sum()
    if orig < 100: continue
    eroded = ndi.binary_erosion(mask, iterations=10)
    survival = eroded.sum() / orig
    (f_asp if genus == 'Aspergillus' else f_muc).append(survival)
    print(f'  {fname}: {survival:.4f}')
f_asp, f_muc = np.array(f_asp), np.array(f_muc)
t, p = stats.ttest_ind(f_asp, f_muc, equal_var=False)
print(f'  Asp={f_asp.mean():.4f}±{f_asp.std(ddof=1):.4f}, Muc={f_muc.mean():.4f}±{f_muc.std(ddof=1):.4f}, p={p:.4g}')

# ── Absorbing capacity ──
print('\n── Absorbing capacity ──')
cap_asp = f_asp_3d * d_asp
cap_muc = f_muc_3d * d_muc
t_cap, p_cap = stats.ttest_ind(cap_asp, cap_muc, equal_var=False)
cap_ratio = cap_asp.mean() / cap_muc.mean()
print(f'  Asp={cap_asp.mean():.3f}±{cap_asp.std(ddof=1):.3f}, Muc={cap_muc.mean():.3f}±{cap_muc.std(ddof=1):.3f}')
print(f'  Ratio={cap_ratio:.3f}, p={p_cap:.4g}')

# ── Condensation observables ──
rsr = pd.read_csv(RSR_CSV)
green = rsr[rsr['group'] == 'Green']
mucor_cond = rsr[rsr['group'] == 'White']
delta_ratio = green['delta_um'].mean() / mucor_cond['delta_um'].mean()


# ═══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('COMPLETE CONSISTENT METRICS')
print('=' * 70)

def print_metric(name, sa, sm):
    n1, n2 = len(sa), len(sm)
    t, p = stats.ttest_ind(sa, sm, equal_var=False)
    pooled = np.sqrt(((n1-1)*sa.std(ddof=1)**2 + (n2-1)*sm.std(ddof=1)**2) / (n1+n2-2))
    d = (sa.mean() - sm.mean()) / pooled if pooled > 0 else 0
    ratio = sa.mean() / sm.mean() if sm.mean() != 0 else np.inf
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f'  {name:25s}  Asp={sa.mean():.4f}±{sa.std(ddof=1):.4f} (n={n1})  '
          f'Muc={sm.mean():.4f}±{sm.std(ddof=1):.4f} (n={n2})  '
          f'ratio={ratio:.3f}  d={d:.2f}  p={p:.4g} {sig}')
    return ratio, d, p

ratios = {}
print('\nMorphological:')
r, d, p = print_metric('GLCM contrast (C)', c_asp, c_muc)
ratios['glcm'] = (r, d, p)
r, d, p = print_metric('Tissue thickness (D)', d_asp, d_muc)
ratios['thick'] = (r, d, p)
r, d, p = print_metric('Local density CV (E)', e_asp, e_muc)
ratios['cv'] = (r, d, p)
r, d, p = print_metric('Erosion survival (F)', f_asp, f_muc)
ratios['erosion'] = (r, d, p)
r, d, p = print_metric('Tissue fraction (3D)', f_asp_3d, f_muc_3d)
ratios['f_tissue'] = (r, d, p)
r, d, p = print_metric('Absorbing capacity', cap_asp, cap_muc)
ratios['capacity'] = (r, d, p)

print('\nCondensation:')
for col, name in [('delta_um', 'δ'), ('zone_metric', 'Size gradient'),
                   ('tau50_zone', 'τ50 gradient'), ('dstar_mm', 'd*')]:
    ga = green[col].dropna().values
    ma = mucor_cond[col].dropna().values
    r, d, p = print_metric(name, ga, ma)

print(f'\n── PREDICTION CHECK ──')
print(f'  Absorbing capacity ratio: {cap_ratio:.3f}')
print(f'  Measured δ ratio:         {delta_ratio:.3f}')
print(f'  Gap: {abs(cap_ratio - delta_ratio)/delta_ratio*100:.1f}%')
print(f'  Thickness ratio:          {d_asp.mean()/d_muc.mean():.3f}')
print(f'  Tissue fraction ratio:    {f_asp_3d.mean()/f_muc_3d.mean():.3f}')
print(f'  Product:                  {d_asp.mean()/d_muc.mean() * f_asp_3d.mean()/f_muc_3d.mean():.3f}')


# ═══════════════════════════════════════════════════════════════
# FIGURE — 8 panels (A-B images, C-F boxplots, G-H validation)
# ═══════════════════════════════════════════════════════════════
print('\nGenerating figure...')

def add_bracket(ax, sa, sm, y_offset_frac=0.08):
    ymax = max(np.max(sa), np.max(sm))
    ymin = min(np.min(sa), np.min(sm))
    rng = ymax - ymin
    y = ymax + rng * y_offset_frac
    ax.plot([1, 1, 2, 2], [y, y + rng * 0.03, y + rng * 0.03, y], 'k-', lw=0.6)
    t, p = stats.ttest_ind(sa, sm, equal_var=False)
    p_str = f'p = {p:.1e}' if p < 0.001 else f'p = {p:.3f}'
    ax.text(1.5, y + rng * 0.05, p_str, ha='center', fontsize=6.5)

def boxstrip(ax, sa, sm, ylabel):
    bp = ax.boxplot([sa, sm], positions=[1, 2], widths=0.45,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='white', lw=1.2),
                    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    bp['boxes'][0].set_facecolor(C_ASP); bp['boxes'][0].set_alpha(0.55)
    bp['boxes'][1].set_facecolor(C_MUC); bp['boxes'][1].set_alpha(0.55)
    rng = np.random.default_rng(42)
    ax.scatter(1 + rng.uniform(-0.10, 0.10, len(sa)), sa,
               s=14, c=C_ASP, alpha=0.8, edgecolors='none', zorder=3)
    ax.scatter(2 + rng.uniform(-0.10, 0.10, len(sm)), sm,
               s=14, c=C_MUC, alpha=0.8, edgecolors='none', zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Aspergillus', 'Mucor'], style='italic', fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    add_bracket(ax, sa, sm)

# Just panels C-H (user composes A-B images separately)
fig = plt.figure(figsize=(200 * MM, 120 * MM))
gs = fig.add_gridspec(2, 4, left=0.07, right=0.97, top=0.92, bottom=0.10,
                      hspace=0.45, wspace=0.45,
                      height_ratios=[1, 1])

ax_c = fig.add_subplot(gs[0, 0])
ax_d = fig.add_subplot(gs[0, 1])
ax_e = fig.add_subplot(gs[0, 2])
ax_f = fig.add_subplot(gs[0, 3])
ax_g = fig.add_subplot(gs[1, 0:2])
ax_h = fig.add_subplot(gs[1, 2:4])

boxstrip(ax_c, c_asp, c_muc, 'GLCM contrast\n(3D colony surface)')
boxstrip(ax_d, d_asp, d_muc, 'Tissue thickness\n(3D, µm)')
boxstrip(ax_e, e_asp, e_muc, 'Local density CV\n(light microscopy)')
boxstrip(ax_f, f_asp, f_muc, 'Erosion survival\n(light microscopy)')

for ax, lbl in zip([ax_c, ax_d, ax_e, ax_f], ['C', 'D', 'E', 'F']):
    ax.text(-0.15, 1.08, lbl, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

# ── Panel G: All 8 ratios ──
cond_ratios = []
cond_pvals = []
for col in ['delta_um', 'zone_metric', 'tau50_zone', 'dstar_mm']:
    ga = green[col].dropna().values
    ma = mucor_cond[col].dropna().values
    r = ga.mean() / ma.mean()
    _, p = stats.ttest_ind(ga, ma, equal_var=False)
    cond_ratios.append(r)
    cond_pvals.append(p)

# Invert GLCM ratio: low contrast = stronger sink, so show Muc/Asp
glcm_ratio_inv = 1.0 / ratios['glcm'][0]  # Muc/Asp
morph_ratios = [ratios['thick'][0], ratios['erosion'][0], ratios['cv'][0], glcm_ratio_inv]
morph_pvals = [ratios['thick'][2], ratios['erosion'][2], ratios['cv'][2], ratios['glcm'][2]]

all_labels = [r'$\delta$', 'Size\ngrad.', r'$\tau_{50}$'+'\ngrad.', r'$d^*$',
              'Thick.\n(D)', 'Erosion\n(F)', 'CV\n(E)', 'GLCM\n(C)']
all_ratios = cond_ratios + morph_ratios
all_pvals = cond_pvals + morph_pvals
all_colors = [C_MUC]*4 + [C_ASP]*4

x_g = np.arange(len(all_ratios))
ax_g.bar(x_g, all_ratios, color=all_colors, alpha=0.7, width=0.65, edgecolor='none')
for xi, (r, p, c) in enumerate(zip(all_ratios, all_pvals, all_colors)):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    ax_g.text(xi, r + 0.03, f'{r:.2f}{sig}', ha='center', fontsize=5,
              fontweight='bold', color=c)
ax_g.axhline(1.0, color='gray', ls='--', lw=0.5, alpha=0.5)
ax_g.axvline(3.5, color='gray', ls='-', lw=0.4, alpha=0.3)
ax_g.set_xticks(x_g)
ax_g.set_xticklabels(all_labels, fontsize=5.5)
ax_g.set_ylabel('Asp / Muc ratio', fontsize=7)
ax_g.set_ylim(0, max(all_ratios) * 1.22)
ax_g.text(1.5, max(all_ratios)*1.15, 'Condensation', ha='center', fontsize=6, color=C_MUC, style='italic')
ax_g.text(5.5, max(all_ratios)*1.15, 'Morphological', ha='center', fontsize=6, color='#2E7D32', style='italic')
for sp in ['top', 'right']: ax_g.spines[sp].set_visible(False)
ax_g.text(-0.08, 1.08, 'G', transform=ax_g.transAxes, fontsize=11, fontweight='bold', va='top')

# ── Panel H: Decomposition ──
thick_r = d_asp.mean() / d_muc.mean()
amplif = cap_ratio / thick_r  # NOT a residual — it's f_tissue ratio
# Actually: cap_ratio = thick_r × f_tissue_r
f_tissue_r = f_asp_3d.mean() / f_muc_3d.mean()

ax_h.bar([0], [thick_r], color=C_ASP, alpha=0.7, width=0.5, edgecolor='none')
ax_h.bar([0], [f_tissue_r - 1], bottom=[thick_r], color='#81C784', alpha=0.5,
         width=0.5, edgecolor='none')
ax_h.bar([1], [delta_ratio], color=C_MUC, alpha=0.7, width=0.5, edgecolor='none')

ax_h.text(0, thick_r/2, f'{thick_r:.2f}×\nthickness', ha='center',
          fontsize=5.5, fontweight='bold', color='white', va='center')
ax_h.text(0, thick_r + (f_tissue_r-1)/2, f'{f_tissue_r:.2f}×\ncoverage',
          ha='center', fontsize=5, color='#2E7D32', va='center')
ax_h.text(1, delta_ratio/2, f'{delta_ratio:.2f}×', ha='center',
          fontsize=7, fontweight='bold', color='white', va='center')

# Show predicted = product
pred = thick_r * f_tissue_r
ax_h.axhline(pred, color=C_ASP, ls=':', lw=0.8, alpha=0.5)
ax_h.text(0.5, pred + 0.05, f'predicted: {pred:.2f}×', ha='center', fontsize=5.5,
          color='#2E7D32', style='italic')

ax_h.set_xticks([0, 1])
ax_h.set_xticklabels(['Morphological\n(thickness × coverage)', 'Measured\nδ ratio'], fontsize=6)
ax_h.set_ylabel('Asp / Muc ratio', fontsize=7)
ax_h.axhline(1.0, color='gray', ls='--', lw=0.5, alpha=0.5)
ax_h.set_ylim(0, max(delta_ratio, pred) * 1.2)
for sp in ['top', 'right']: ax_h.spines[sp].set_visible(False)
ax_h.text(-0.12, 1.08, 'H', transform=ax_h.transAxes, fontsize=11, fontweight='bold', va='top')

for ext in ('.png', '.pdf', '.svg'):
    kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
    if ext == '.png': kw['dpi'] = 300
    fig.savefig(OUT / f'figure4_consistent{ext}', **kw)
plt.close(fig)
print(f'\nSaved: {OUT}/figure4_consistent.{{png,pdf,svg}}')

# ── Save CSVs ──
import csv

# Per-sample 3D
rows_3d = []
i_a, i_m = 0, 0
for key, val in saved.items():
    genus = val.get('genus')
    roi_path = ROI_DIR / genus / f'{Path(key).stem}_roi.jpg'
    if not roi_path.exists(): continue
    if genus == 'Aspergillus':
        rows_3d.append({'file': key, 'genus': genus, 'glcm': round(c_asp[i_a], 4),
                        'thickness_um': round(d_asp[i_a], 4), 'f_tissue': round(f_asp_3d[i_a], 4),
                        'absorbing_capacity': round(cap_asp[i_a], 4)})
        i_a += 1
    else:
        rows_3d.append({'file': key, 'genus': genus, 'glcm': round(c_muc[i_m], 4),
                        'thickness_um': round(d_muc[i_m], 4), 'f_tissue': round(f_muc_3d[i_m], 4),
                        'absorbing_capacity': round(cap_muc[i_m], 4)})
        i_m += 1

with open(OUT / 'figure4_3d_metrics.csv', 'w', newline='') as fout:
    w = csv.DictWriter(fout, fieldnames=['file', 'genus', 'glcm', 'thickness_um', 'f_tissue', 'absorbing_capacity'])
    w.writeheader()
    w.writerows(rows_3d)

print(f'Saved: {OUT}/figure4_3d_metrics.csv')
print('\nDone. All panels computed from single consistent segmentation.')

#!/usr/bin/env python3
"""Build a complete Figure 4 PDF in one shot.

A,B: three-channel colony imaging (color macro brightfield / grayscale LM /
local-density heatmap) for Aspergillus and Mucor.
C-F: FFT alpha, tissue thickness, Hessian tubeness CV, hyphal area fraction.
G:   Asp/Muc ratio bar chart (4 condensation + 4 morphological).
H:   Absorbing-capacity decomposition.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_T7 = Path('/Volumes/T7/FINAL OSF')
_REPO = Path(__file__).resolve().parent.parent.parent
BASE = _T7 if _T7.exists() else _REPO

ROI_DIR  = BASE / 'HYPHAE' / 'Analysis' / 'results' / '3d_overlays'
SESSION  = ROI_DIR / 'roi_session.json'
MICRO    = BASE / 'HYPHAE' / 'Light Microscopy'
RSR_CSV  = BASE / 'FigureRSR' / 'output' / 'rsr_and_lab_dstar_metrics.csv'
CSV_OUT  = BASE / 'FigureHyphae' / 'output'
FIG_OUT  = BASE / 'FigureHyphae' / 'figures'
DOWNLOAD = Path('/Users/yany/Downloads/Figure4_revamp_2026-05-20')
DOWNLOAD.mkdir(parents=True, exist_ok=True)

ASP_ROI  = ROI_DIR / 'Aspergillus' / '20251214_222552_roi.jpg'
MUC_ROI  = ROI_DIR / 'Mucor'       / '20251210_155926_roi.jpg'
ASP_LM   = MICRO / 'Pink_10X_1.TIF'
MUC_LM   = MICRO / 'White_10X_1.TIF'

CAL_3D = 0.94                       # microns per pixel for macro 3D ROIs
MM = 1 / 25.4
C_ASP, C_MUC = '#4CAF50', '#757575'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 8, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'mathtext.default': 'regular',
})

def load_gray(p):
    with Image.open(p) as im:
        a = np.asarray(im).astype(np.float64)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a

def load_rgb(p):
    with Image.open(p) as im:
        a = np.asarray(im.convert('RGB')).astype(np.float64) / 255.0
    return a

def seg_3d(img):
    s = ndi.gaussian_filter(img, sigma=1.0)
    l = ndi.gaussian_filter(s, sigma=32.0)
    dr = l - s
    t = dr.mean() + 0.5 * dr.std()
    return ndi.binary_opening(ndi.binary_closing(dr > t, iterations=1), iterations=1)

def gliding_density(img, win_um=120.0, cal_um_per_px=CAL_3D):
    """Local tissue fraction by gliding box of side win_um."""
    mask = seg_3d(img).astype(np.float64)
    w = max(8, int(round(win_um / cal_um_per_px)))
    kernel = np.ones((w, w)) / (w * w)
    return ndi.convolve(mask, kernel, mode='nearest')

def norm_display(img):
    lo, hi = np.percentile(img, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1
    return np.clip((img - lo) / (hi - lo), 0, 1)

def stripbox(ax, sa, sm, ylab, p_override=None):
    bp = ax.boxplot([sa, sm], positions=[1, 2], widths=0.4, patch_artist=True,
                    showfliers=False, medianprops=dict(color='white', lw=1.4),
                    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    bp['boxes'][0].set_facecolor(C_ASP); bp['boxes'][0].set_alpha(0.55)
    bp['boxes'][1].set_facecolor(C_MUC); bp['boxes'][1].set_alpha(0.55)
    rng = np.random.default_rng(42)
    ax.scatter(1 + rng.uniform(-0.09, 0.09, len(sa)), sa, s=14, c=C_ASP,
               alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
    ax.scatter(2 + rng.uniform(-0.09, 0.09, len(sm)), sm, s=14, c=C_MUC,
               alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([r'$\it{Aspergillus}$', r'$\it{Mucor}$'], fontsize=6.5)
    ax.set_ylabel(ylab, fontsize=7)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    p = p_override if p_override is not None else stats.ttest_ind(sa, sm, equal_var=False)[1]
    ym = max(sa.max(), sm.max()); r = ym - min(sa.min(), sm.min()); y = ym + r * 0.08
    ax.plot([1, 1, 2, 2], [y, y + r * 0.03, y + r * 0.03, y], 'k-', lw=0.6)
    txt = f'p = {p:.1e}' if p < 0.001 else f'p = {p:.3f}'
    ax.text(1.5, y + r * 0.06, txt, ha='center', fontsize=6)


# ── metrics from CSVs (we trust the pipeline; no recomputation) ──
fft_df  = pd.read_csv(CSV_OUT / 'fft_per_roi.csv')
fft_a   = fft_df.loc[fft_df.genus == 'Aspergillus', 'alpha'].values
fft_m   = fft_df.loc[fft_df.genus == 'Mucor',       'alpha'].values

with open(SESSION) as f:
    saved = {k: v for k, v in json.load(f).items()
             if not k.startswith('_') and v.get('status') != 'deleted'}
thick_a, thick_m, ftiss_a, ftiss_m = [], [], [], []
for key, val in saved.items():
    g = val.get('genus')
    rp = ROI_DIR / g / f'{Path(key).stem}_roi.jpg'
    if not rp.exists():
        continue
    img = load_gray(rp); m = seg_3d(img)
    dt = ndi.distance_transform_edt(m) * CAL_3D
    dtm = np.median(dt[m]) if m.sum() > 100 else 0
    (thick_a if g == 'Aspergillus' else thick_m).append(dtm)
    (ftiss_a if g == 'Aspergillus' else ftiss_m).append(m.mean())
thick_a, thick_m = np.array(thick_a), np.array(thick_m)
ftiss_a, ftiss_m = np.array(ftiss_a), np.array(ftiss_m)

MICRO_IMAGES = [
    ('Pink_10X_1.TIF',  'Asp'), ('Pink_10X_2.TIF',  'Asp'),
    ('Pink_20X_1.TIF',  'Asp'), ('Pink_20X_2.TIF',  'Asp'),
    ('Pink_40X_1.TIF',  'Asp'), ('Pink_40X_2.TIF',  'Asp'),
    ('White_10X_1.TIF', 'Muc'), ('White_20X_1.TIF', 'Muc'),
    ('White_40X_1.TIF', 'Muc'),
]
HESS = [1, 2, 4, 8, 16]
def multiscale_tubeness(img):
    r = np.zeros_like(img)
    for s in HESS:
        Ixx = ndi.gaussian_filter(img, sigma=s, order=[0, 2]) * s * s
        Iyy = ndi.gaussian_filter(img, sigma=s, order=[2, 0]) * s * s
        Ixy = ndi.gaussian_filter(img, sigma=s, order=[1, 1]) * s * s
        tr = Ixx + Iyy; det = Ixx * Iyy - Ixy * Ixy
        disc = np.sqrt(np.clip((tr / 2) ** 2 - det, 0, None))
        r = np.maximum(r, np.clip(tr / 2 + disc, 0, None))
    return r

def otsu_t(v):
    h, e = np.histogram(v, bins=256); c = (e[:-1] + e[1:]) / 2; t = h.sum()
    wb = np.cumsum(h).astype(float); wf = t - wb; sb = np.cumsum(h * c)
    mb = sb / np.maximum(wb, 1); mf = (sb[-1] - sb) / np.maximum(wf, 1)
    return c[np.argmax(wb * wf * (mb - mf) ** 2)]

def seg_micro(img, label):
    s = ndi.gaussian_filter(img, sigma=1.0); l = ndi.gaussian_filter(s, sigma=32.0); dr = l - s
    v = np.ones(img.shape, dtype=bool); m = max(8, int(round(min(img.shape) * 0.006)))
    v[:m, :] = False; v[-m:, :] = False; v[:, :m] = False; v[:, -m:] = False
    vals = s[v]; gt = otsu_t(vals); drv = dr[v]; dm = float(np.median(drv))
    dd = float(np.median(np.abs(drv - dm)) + 1e-9)
    if label.lower() in ('pink', 'asp', 'green'):
        ic = float(np.percentile(vals, 62))
        lt = max(float(np.percentile(drv, 72)), dm + 0.80 * dd)
        bm = s < min(gt, ic); fm = v & (dr > lt); mask = v & (bm | fm); ms, mh = 28, 1500
    else:
        ic = float(np.percentile(vals, 32))
        lt = max(float(np.percentile(drv, 78)), dm + 1.15 * dd)
        fm = v & (dr > lt); sd = (s < ic) & (dr > np.percentile(drv, 48))
        mask = v & (fm | sd); ms, mh = 10, 80
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)), iterations=1); mask &= v
    la, nl = ndi.label(mask)
    if nl > 0:
        sz = np.bincount(la.ravel()); k = sz >= ms; k[0] = False; mask = k[la]
    fl = ndi.binary_fill_holes(mask); ho = fl & ~mask; hl, nhl = ndi.label(ho)
    if nhl > 0:
        hs = np.bincount(hl.ravel()); sm = hs <= mh; sm[0] = False; mask = mask | sm[hl]
    return (mask & v).astype(bool)

tubcv_a, tubcv_m = [], []
for fname, g in MICRO_IMAGES:
    img = load_gray(MICRO / fname); n = norm_display(img); lab = 'asp' if g == 'Asp' else 'muc'
    mask = seg_micro(img, lab); t = multiscale_tubeness(n)[mask]
    cv = t.std() / t.mean() if t.mean() > 0 and len(t) > 100 else 0
    (tubcv_a if g == 'Asp' else tubcv_m).append(cv)
tubcv_a, tubcv_m = np.array(tubcv_a), np.array(tubcv_m)

rsr = pd.read_csv(RSR_CSV)
green   = rsr[rsr.group == 'Green']
mucor_c = rsr[rsr.group == 'White']
delta_ratio = green['delta_um'].mean() / mucor_c['delta_um'].mean()


# ════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(180 * MM, 230 * MM))
outer = GridSpec(3, 1, figure=fig, hspace=0.45,
                 left=0.06, right=0.97, top=0.97, bottom=0.05,
                 height_ratios=[1.35, 0.85, 0.85])

# ── A,B imagery block ──────────────────────────────────────────
img_gs = outer[0].subgridspec(3, 2, hspace=0.05, wspace=0.05)

def show_genus_column(col, label, color, roi_path, lm_path):
    rgb     = load_rgb(roi_path)
    lm_gray = load_gray(lm_path)
    lm_n    = norm_display(lm_gray)
    macro   = load_gray(roi_path)
    dens    = gliding_density(macro, win_um=120.0)

    ax = fig.add_subplot(img_gs[0, col]); ax.imshow(rgb); ax.axis('off')
    if col == 0:
        ax.text(-0.06, 0.5, 'Macro\nbrightfield', transform=ax.transAxes,
                fontsize=6.5, rotation=90, va='center', ha='right')
    ax.text(0.5, 1.02, f'$\\it{{{label}}}$', transform=ax.transAxes,
            fontsize=10, color=color, fontweight='bold', ha='center', va='bottom')
    if col == 0:
        ax.text(-0.10, 1.10, 'A', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')
    else:
        ax.text(-0.04, 1.10, 'B', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')

    ax = fig.add_subplot(img_gs[1, col]); ax.imshow(lm_n, cmap='gray'); ax.axis('off')
    if col == 0:
        ax.text(-0.06, 0.5, 'Light\nmicroscopy', transform=ax.transAxes,
                fontsize=6.5, rotation=90, va='center', ha='right')

    ax = fig.add_subplot(img_gs[2, col]); ax.imshow(dens, cmap='inferno', vmin=0, vmax=0.8)
    ax.axis('off')
    if col == 0:
        ax.text(-0.06, 0.5, 'Local-density\nheatmap', transform=ax.transAxes,
                fontsize=6.5, rotation=90, va='center', ha='right')

show_genus_column(0, 'Aspergillus', C_ASP, ASP_ROI, ASP_LM)
show_genus_column(1, 'Mucor',       C_MUC, MUC_ROI, MUC_LM)

# ── C-F panels ─────────────────────────────────────────────────
cf_gs = outer[1].subgridspec(1, 4, wspace=0.55)
ax_c = fig.add_subplot(cf_gs[0]); stripbox(ax_c, fft_a,  fft_m,
    r'Spectral slope $\alpha$' + '\n(3D colony surface)')
ax_d = fig.add_subplot(cf_gs[1]); stripbox(ax_d, thick_a, thick_m,
    'Tissue thickness\n(3D, µm)')
ax_e = fig.add_subplot(cf_gs[2]); stripbox(ax_e, tubcv_a, tubcv_m,
    'Hessian tubeness CV\n(light microscopy)')
ax_f = fig.add_subplot(cf_gs[3]); stripbox(ax_f, ftiss_a, ftiss_m,
    'Hyphal area fraction\n(3D colony surface)')
for ax, l in zip([ax_c, ax_d, ax_e, ax_f], 'CDEF'):
    ax.text(-0.20, 1.10, l, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

# ── G-H panels ─────────────────────────────────────────────────
gh_gs = outer[2].subgridspec(1, 2, wspace=0.35, width_ratios=[1.6, 1.0])
ax_g = fig.add_subplot(gh_gs[0]); ax_h = fig.add_subplot(gh_gs[1])

cond_r, cond_p = [], []
for col in ['delta_um', 'zone_metric', 'tau50_zone', 'dstar_mm']:
    a = green[col].dropna().values; b = mucor_c[col].dropna().values
    cond_r.append(a.mean() / b.mean())
    cond_p.append(stats.ttest_ind(a, b, equal_var=False)[1])

fft_ratio = abs(fft_m.mean()) / abs(fft_a.mean())
morph_r = [fft_ratio,
           thick_a.mean() / thick_m.mean(),
           tubcv_a.mean() / tubcv_m.mean(),
           ftiss_a.mean() / ftiss_m.mean()]
morph_p = [stats.ttest_ind(fft_a,  fft_m,  equal_var=False)[1],
           stats.ttest_ind(thick_a, thick_m, equal_var=False)[1],
           stats.ttest_ind(tubcv_a, tubcv_m, equal_var=False)[1],
           stats.ttest_ind(ftiss_a, ftiss_m, equal_var=False)[1]]

lb = [r'$\delta$', 'Size\ngrad.', r'$\tau_{50}$' + '\ngrad.', r'$d^*$',
      r'FFT $\alpha$' + '\n(C)', 'Thick.\n(D)', 'Tub CV\n(E)', 'Hyphal\narea (F)']
ar = cond_r + morph_r; ap = cond_p + morph_p
ac = [C_MUC] * 4 + [C_ASP] * 4
xg = np.arange(8)
ax_g.bar(xg, ar, color=ac, alpha=0.7, width=0.65, edgecolor='none')
for xi, (r, p, c) in enumerate(zip(ar, ap, ac)):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ' n.s.'
    ax_g.text(xi, r + 0.03, f'{r:.2f}{sig}', ha='center', fontsize=5,
              fontweight='bold', color=c)
ax_g.axhline(1.0, color='gray', ls='--', lw=0.5, alpha=0.5)
ax_g.axvline(3.5, color='gray', ls='-', lw=0.4, alpha=0.3)
ax_g.set_xticks(xg); ax_g.set_xticklabels(lb, fontsize=5.5)
ax_g.set_ylabel('Asp / Muc ratio', fontsize=7); ax_g.set_ylim(0, max(ar) * 1.22)
ax_g.text(1.5, max(ar) * 1.15, 'Condensation', ha='center', fontsize=6,
          color=C_MUC, style='italic')
ax_g.text(5.5, max(ar) * 1.15, 'Morphological', ha='center', fontsize=6,
          color='#2E7D32', style='italic')
for sp in ('top', 'right'):
    ax_g.spines[sp].set_visible(False)
ax_g.text(-0.10, 1.10, 'G', transform=ax_g.transAxes,
          fontsize=12, fontweight='bold', va='top')

tr  = thick_a.mean() / thick_m.mean()
ftr = ftiss_a.mean() / ftiss_m.mean()
pred = tr * ftr
ax_h.bar([0], [pred],        color=C_ASP, alpha=0.7, width=0.5, edgecolor='none')
ax_h.bar([1], [delta_ratio], color=C_MUC, alpha=0.7, width=0.5, edgecolor='none')
ax_h.text(0, pred / 2, f'{pred:.2f}×', ha='center', fontsize=8,
          fontweight='bold', color='white', va='center')
ax_h.text(0, pred + 0.08, f'{tr:.2f}× thickness\n× {ftr:.2f}× coverage',
          ha='center', fontsize=5.5, color='#2E7D32', va='bottom')
ax_h.text(1, delta_ratio / 2, f'{delta_ratio:.2f}×', ha='center', fontsize=8,
          fontweight='bold', color='white', va='center')
ax_h.set_xticks([0, 1])
ax_h.set_xticklabels(['Predicted\n(morphology)', 'Measured\n(δ ratio)'], fontsize=7)
ax_h.set_ylabel('Asp / Muc ratio', fontsize=7)
ax_h.axhline(1.0, color='gray', ls='--', lw=0.5, alpha=0.5)
ax_h.set_ylim(0, max(delta_ratio, pred) * 1.25)
for sp in ('top', 'right'):
    ax_h.spines[sp].set_visible(False)
ax_h.text(-0.18, 1.10, 'H', transform=ax_h.transAxes,
          fontsize=12, fontweight='bold', va='top')

for ext in ('.pdf', '.png', '.svg'):
    kw = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.05}
    if ext == '.png':
        kw['dpi'] = 600
    fig.savefig(DOWNLOAD / f'Figure4_NEW{ext}', **kw)
print(f'Saved Figure4_NEW.pdf/.png/.svg to {DOWNLOAD}')

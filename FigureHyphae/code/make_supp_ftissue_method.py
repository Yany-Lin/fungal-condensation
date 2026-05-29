#!/usr/bin/env python3
"""Supplementary figure documenting the photometric-reconstruction f_tissue method.

Replaces the prior binary-Otsu f_tissue computation.  Variable name f_tissue
is unchanged in the paper — only the method that produces it differs.

Panels:
  A: f_tissue boxplot (new method)
  B: Method derivation + canonical Aspergillus raw ROI + reconstruction heatmap
  C: Canonical Mucor raw ROI + reconstruction heatmap
  D: Method-vs-method comparison: new photometric vs old binary, plus four
     alternative segmentation methods, with the delta benchmark line
"""
import csv
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

BASE = Path('/Volumes/T7/FINAL OSF')
ROI_DIR = BASE / 'HYPHAE' / 'Analysis' / 'results' / '3d_overlays'
SESSION = ROI_DIR / 'roi_session.json'
CSV_DIR = BASE / 'FigureHyphae' / 'output'
OUT     = BASE / 'FigureHyphae' / 'figures'
DOWN    = Path('/Users/yany/Downloads/Figure4_revamp_2026-05-29_continuousD')

ASP_KEY = '20251214_222552.JPG'
MUC_KEY = '20251210_155926.JPG'
SCALES  = (8, 16, 32, 64, 128)
ALPHA   = 1.0
DELTA_RATIO = 2.13

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
        a = np.asarray(im).astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a


def normalize01(img):
    lo, hi = img.min(), img.max()
    return (img - lo) / max(hi - lo, 1e-6)


def reconstruct_density(img, sigmas=SCALES, alpha=ALPHA):
    """CPU numpy port of the photometric reconstruction (no GPU needed).
    Returns per-pixel R(x,y).
    """
    n = normalize01(img)
    s = ndi.gaussian_filter(n, sigma=1.0)
    gy, gx = np.gradient(s)
    grad_mag = np.hypot(gx, gy)
    R_accum = np.zeros_like(s, dtype=np.float64)
    eps = 1e-9
    for sig in sigmas:
        bg = ndi.gaussian_filter(s, sigma=sig)
        B  = np.clip(s - bg, 0, None)
        G  = ndi.gaussian_filter(grad_mag, sigma=sig)
        Sxx = ndi.gaussian_filter(gx * gx, sigma=sig)
        Syy = ndi.gaussian_filter(gy * gy, sigma=sig)
        Sxy = ndi.gaussian_filter(gx * gy, sigma=sig)
        tr = Sxx + Syy
        disc = np.sqrt(np.clip((Sxx - Syy) ** 2 + 4 * Sxy ** 2, 0, None))
        l1 = (tr + disc) / 2.0
        l2 = (tr - disc) / 2.0
        coherence = ((l1 - l2) / (l1 + l2 + eps)) ** 2
        Gn = G / max(G.max(), eps)
        Cn = coherence / max(coherence.max(), eps)
        R_accum = R_accum + B * (1.0 + alpha * Gn * Cn)
    return R_accum / len(sigmas)


# ── Load per-ROI values ──
df_rec = pd.read_csv(CSV_DIR / 'ftissue_photometric.csv')   # new method
df_bin = pd.read_csv(CSV_DIR / 'ftissue_all_methods_final.csv')  # old binaries
df_con = pd.read_csv(CSV_DIR / 'ftissue_continuous_density.csv')

f_a = df_rec.loc[df_rec.genus == 'Aspergillus', 'D_recon'].values
f_m = df_rec.loc[df_rec.genus == 'Mucor',       'D_recon'].values

ratio_new = f_a.mean() / f_m.mean()
welch_p   = stats.ttest_ind(f_a, f_m, equal_var=False).pvalue

# Per-ROI method ratios for comparison panel
methods = [
    ('photometric (new)',    'D_recon',          df_rec, '#1976D2', True),
    ('continuous bright',    'D_continuous',     df_con, '#5C6BC0', False),
    ('adaptive-dark (old)',  'f_adaptive-dark',  df_bin, '#9E9E9E', False),
    ('adaptive-bright',      'f_adaptive-bright',df_bin, '#42A5F5', False),
    ('Sauvola-dark',         'f_sauvola-dark',   df_bin, '#E53935', False),
    ('Sauvola-bright',       'f_sauvola-bright', df_bin, '#66BB6A', False),
]
ratios = []
for name, col, dfx, color, primary in methods:
    a = dfx.loc[dfx.genus == 'Aspergillus', col].values
    m = dfx.loc[dfx.genus == 'Mucor',       col].values
    ratios.append((name, a.mean() / m.mean(), color, primary))


# ── Load canonical images and run reconstruction ──
with open(SESSION) as f:
    saved = json.load(f)
canon = {}
for key, val in saved.items():
    if key.startswith('_') or val.get('status') == 'deleted':
        continue
    if key in (ASP_KEY, MUC_KEY):
        g = val['genus']
        rp = ROI_DIR / g / f'{Path(key).stem}_roi.jpg'
        img = load_gray(rp)
        R = reconstruct_density(img)
        canon[g] = {'img': img, 'R': R}
        print(f'  {g}: R mean = {R.mean():.4f}')


# ── Figure ──
fig = plt.figure(figsize=(180 * MM, 220 * MM))
gs = GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.32,
              left=0.07, right=0.96, top=0.95, bottom=0.05,
              height_ratios=[1.0, 1.0, 1.0, 1.0])


# Panel A: f_tissue boxplot (new method)
ax = fig.add_subplot(gs[0, 0])
bp = ax.boxplot([f_a, f_m], positions=[1, 2], widths=0.45, patch_artist=True,
                showfliers=False,
                medianprops=dict(color='white', lw=1.4),
                whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
bp['boxes'][0].set_facecolor(C_ASP); bp['boxes'][0].set_alpha(0.55)
bp['boxes'][1].set_facecolor(C_MUC); bp['boxes'][1].set_alpha(0.55)
rng = np.random.default_rng(42)
ax.scatter(1 + rng.uniform(-0.09, 0.09, len(f_a)), f_a, s=18, c=C_ASP,
           alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
ax.scatter(2 + rng.uniform(-0.09, 0.09, len(f_m)), f_m, s=18, c=C_MUC,
           alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
ax.set_xticks([1, 2])
ax.set_xticklabels([r'$\it{Asp}$', r'$\it{Muc}$'], fontsize=7)
ax.set_ylabel(r'$f_\mathrm{tissue}$', fontsize=9)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ym = max(f_a.max(), f_m.max()); rg = max(ym - min(f_a.min(), f_m.min()), 1e-9)
y = ym + rg * 0.08
ax.plot([1, 1, 2, 2], [y, y + rg * 0.03, y + rg * 0.03, y], 'k-', lw=0.6)
ax.text(1.5, y + rg * 0.06, f'p = {welch_p:.1e}', ha='center', fontsize=6)
ax.set_title(f'ratio = {ratio_new:.2f}×', fontsize=8, pad=3)
ax.text(-0.30, 1.14, 'A', transform=ax.transAxes, fontsize=12,
        fontweight='bold', va='top')


# Panel B: method-text + equation
ax = fig.add_subplot(gs[0, 1:])
ax.axis('off')
text_lines = [
    r'$\bf{Photometric\ reconstruction\ of\ }f_\mathrm{tissue}$',
    '',
    r'Per-pixel reconstruction at scale $\sigma$:',
    r'  $R_\sigma(x,y) = B_\sigma\,\bigl(1 + \alpha\,\widetilde{G}_\sigma\,\widetilde{C}_\sigma\bigr)$',
    '',
    r'  $B_\sigma$ — brightness above local Gaussian background',
    r'  $G_\sigma$ — gradient magnitude (smoothed)',
    r'  $C_\sigma$ — structure-tensor coherence $\in[0,1]$',
    r'  $\widetilde{\cdot}$ — rescaled to $[0,1]$ within ROI; $\alpha=1$',
    '',
    r'Scale ladder $\sigma \in \{8, 16, 32, 64, 128\}$ px (7.5–120 µm).',
    r'Per-ROI scalar $f_\mathrm{tissue} = \langle R_\sigma \rangle_{\sigma, x, y}$.',
    '',
    r'$\bf{No\ thresholding}$, no class-specific tuning — all the brightness gradient is retained.',
]
y = 0.97
for line in text_lines:
    ax.text(0.02, y, line, transform=ax.transAxes, fontsize=8, va='top', ha='left')
    y -= 0.075


# Row 2: canonical Aspergillus raw + reconstruction
def display_raw(img):
    lo, hi = np.percentile(img, [0.5, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1), 0, 1)


vmax_R = max(canon['Aspergillus']['R'].max(), canon['Mucor']['R'].max())

ax_ar = fig.add_subplot(gs[1, 0:2])
ax_ar.imshow(display_raw(canon['Aspergillus']['img']),
             cmap='gray', interpolation='nearest')
ax_ar.axis('off')
ax_ar.text(0.98, 0.04, 'Aspergillus', transform=ax_ar.transAxes, fontsize=8,
           ha='right', va='bottom', style='italic', color='white',
           fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.20', facecolor=C_ASP, alpha=0.85,
                     edgecolor='none'))
ax_ar.set_title('Raw ROI', fontsize=8, pad=3)
ax_ar.text(-0.02, 1.10, 'C', transform=ax_ar.transAxes, fontsize=12,
           fontweight='bold', va='top')

ax_ad = fig.add_subplot(gs[1, 2:4])
ax_ad.imshow(canon['Aspergillus']['R'], cmap='viridis',
             interpolation='nearest', vmin=0, vmax=vmax_R)
ax_ad.axis('off')
ax_ad.set_title(f'Photometric reconstruction '
                f'($f_\\mathrm{{tissue}} = {canon["Aspergillus"]["R"].mean():.4f}$)',
                fontsize=8, pad=3)
ax_ad.text(-0.02, 1.10, 'D', transform=ax_ad.transAxes, fontsize=12,
           fontweight='bold', va='top')


# Row 3: canonical Mucor raw + reconstruction
ax_mr = fig.add_subplot(gs[2, 0:2])
ax_mr.imshow(display_raw(canon['Mucor']['img']),
             cmap='gray', interpolation='nearest')
ax_mr.axis('off')
ax_mr.text(0.98, 0.04, 'Mucor', transform=ax_mr.transAxes, fontsize=8,
           ha='right', va='bottom', style='italic', color='white',
           fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.20', facecolor=C_MUC, alpha=0.85,
                     edgecolor='none'))
ax_mr.set_title('Raw ROI', fontsize=8, pad=3)
ax_mr.text(-0.02, 1.10, 'E', transform=ax_mr.transAxes, fontsize=12,
           fontweight='bold', va='top')

ax_md = fig.add_subplot(gs[2, 2:4])
im_R = ax_md.imshow(canon['Mucor']['R'], cmap='viridis',
                    interpolation='nearest', vmin=0, vmax=vmax_R)
ax_md.axis('off')
ax_md.set_title(f'Photometric reconstruction '
                f'($f_\\mathrm{{tissue}} = {canon["Mucor"]["R"].mean():.4f}$)',
                fontsize=8, pad=3)
ax_md.text(-0.02, 1.10, 'F', transform=ax_md.transAxes, fontsize=12,
           fontweight='bold', va='top')

# Single colorbar for the recon row
cbar_ax = fig.add_axes([0.92, 0.27, 0.012, 0.32])
cb = plt.colorbar(im_R, cax=cbar_ax)
cb.set_label(r'$R(x,y)$', fontsize=7)
cb.ax.tick_params(labelsize=6)


# Row 4: method comparison vs δ benchmark
ax = fig.add_subplot(gs[3, :])
xs = np.arange(len(ratios))
names  = [r[0] for r in ratios]
vals   = [r[1] for r in ratios]
colors = [r[2] for r in ratios]
primary = [r[3] for r in ratios]
ax.bar(xs, vals, color=colors, alpha=0.85, edgecolor='black', lw=0.5)
for x, v, p in zip(xs, vals, primary):
    ax.text(x, v + 0.06, f'{v:.2f}×', ha='center', fontsize=8,
            fontweight='bold' if p else 'normal')
ax.axhline(DELTA_RATIO, color='black', lw=0.8, ls='--', alpha=0.7)
ax.text(len(ratios) - 0.5, DELTA_RATIO + 0.06,
        f' δ benchmark = {DELTA_RATIO}×', va='bottom', ha='right',
        fontsize=8, color='black')
# Highlight primary
ax.bar([0], [vals[0]], color='none', edgecolor='#1976D2', lw=2.0,
       width=0.85, zorder=5)
ax.set_xticks(xs)
ax.set_xticklabels(names, fontsize=7, rotation=15, ha='right')
ax.set_ylabel(r'Asp / Muc $f_\mathrm{tissue}$ ratio', fontsize=8)
ax.set_ylim(0, max(max(vals), DELTA_RATIO) * 1.18)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.set_title('Method comparison.  The photometric reconstruction (boxed) '
             'is closest to the δ benchmark while remaining threshold-free.',
             fontsize=8, pad=4)
ax.text(-0.05, 1.10, 'G', transform=ax.transAxes, fontsize=12,
        fontweight='bold', va='top')


for ext in ('.pdf', '.svg', '.png'):
    kw = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.04}
    if ext == '.png':
        kw['dpi'] = 600
    fig.savefig(OUT  / f'figS_ftissue_method{ext}', **kw)
    fig.savefig(DOWN / f'figS_ftissue_method{ext}', **kw)
plt.close(fig)
print(f'\nSaved figS_ftissue_method.{{pdf,svg,png}}')
print(f'  primary new-method ratio = {ratios[0][1]:.3f}× (δ benchmark = {DELTA_RATIO}×)')
print(f'  old binary  ratio        = {ratios[2][1]:.3f}×')

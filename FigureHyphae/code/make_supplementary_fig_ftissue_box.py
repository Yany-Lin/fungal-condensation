#!/usr/bin/env python3
"""Single-panel f_tissue box plot in the exact Fig 4 C-F stripbox style.

Reuses the stripbox function and per-ROI f_tissue computation from
build_figure4_complete.py so this panel can be inserted anywhere
(supplementary, expanded Fig 4) without restyling.

Inputs: macro colony-surface ROIs (13 Asp + 11 Muc) at 0.94 µm/px.
Outputs figS_ftissue_box.{pdf,png,svg} sized to match one cf_gs cell
(approx. 45 mm wide x 65 mm tall, matching panels C/D/E/F of Fig 4).
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──
_T7 = Path('/Volumes/T7/FINAL OSF')
_WIN = Path(r'D:\FINAL OSF')
_REPO = Path(__file__).resolve().parent.parent.parent
BASE = next((p for p in (_T7, _WIN, _REPO) if p.exists()), _REPO)

ROI_DIR = BASE / 'HYPHAE' / 'Analysis' / 'results' / '3d_overlays'
SESSION = ROI_DIR / 'roi_session.json'
OUT = BASE / 'FigureHyphae' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

CAL_3D = 0.94
MM = 1 / 25.4
C_ASP, C_MUC = '#4CAF50', '#757575'

# Identical to build_figure4_complete.py rcParams
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


def seg_3d(img):
    s = ndi.gaussian_filter(img, sigma=1.0)
    l = ndi.gaussian_filter(s, sigma=32.0)
    dr = l - s
    t = dr.mean() + 0.5 * dr.std()
    return ndi.binary_opening(ndi.binary_closing(dr > t, iterations=1), iterations=1)


def stripbox(ax, sa, sm, ylab, p_override=None):
    """Verbatim copy from build_figure4_complete.py:84-104.
    Box + jittered scatter + Welch-t p-value bracket."""
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


def savefig(fig, name):
    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.03}
        if ext == '.png':
            kw['dpi'] = 600
        fig.savefig(OUT / f'{name}{ext}', **kw)
    plt.close(fig)
    print(f'  Saved: {name}')


# ── Per-ROI f_tissue (verbatim from build_figure4_complete.py:112-127) ──
with open(SESSION) as f:
    saved = {k: v for k, v in json.load(f).items()
             if not k.startswith('_') and v.get('status') != 'deleted'}

ftiss_a, ftiss_m = [], []
for key, val in saved.items():
    g = val.get('genus')
    rp = ROI_DIR / g / f'{Path(key).stem}_roi.jpg'
    if not rp.exists():
        continue
    img = load_gray(rp)
    m = seg_3d(img)
    if m.sum() < 100:
        continue
    (ftiss_a if g == 'Aspergillus' else ftiss_m).append(m.mean())

ftiss_a, ftiss_m = np.array(ftiss_a), np.array(ftiss_m)

print(f'Aspergillus (n={len(ftiss_a)}): mean={ftiss_a.mean():.3f}, '
      f'sd={ftiss_a.std(ddof=1):.3f}')
print(f'Mucor       (n={len(ftiss_m)}): mean={ftiss_m.mean():.3f}, '
      f'sd={ftiss_m.std(ddof=1):.3f}')
t, p = stats.ttest_ind(ftiss_a, ftiss_m, equal_var=False)
print(f'Welch t = {t:.3f}, p = {p:.3e}')

# ── Figure: one panel matched to Fig 4 cf_gs cell dimensions ──
# Fig 4 outer = 180 mm wide, cf_gs row = 1/3.05 of 230 mm tall ≈ 65 mm.
# 4 cells with wspace=0.55: each cell ≈ 36 mm wide. Pad to 45 mm for standalone.
fig, ax = plt.subplots(1, 1, figsize=(45 * MM, 65 * MM))
fig.subplots_adjust(left=0.30, right=0.95, top=0.88, bottom=0.14)

stripbox(ax, ftiss_a, ftiss_m,
         r'$f_{\mathrm{tissue}}$' + '\n(3D colony surface)')

# Panel letter slot (placeholder — change/remove when inserting)
ax.text(-0.40, 1.10, 'X', transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top')

savefig(fig, 'figS_ftissue_box')
print('Done.')

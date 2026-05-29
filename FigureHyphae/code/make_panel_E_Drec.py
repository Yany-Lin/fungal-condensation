#!/usr/bin/env python3
"""Standalone Panel E (D_rec boxplot) matched to published Figure 4 style.

Drop-in replacement for the f_tissue panel. Same dimensions / fontsizes / colors
as the existing C, D, E, F, G panels in Figures/Figure4.pdf so it composites
cleanly in Inkscape.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path('/Volumes/T7/FINAL OSF')
CSV  = BASE / 'FigureHyphae' / 'output' / 'ftissue_photometric.csv'
OUT  = BASE / 'FigureHyphae' / 'figures'
DOWN = Path('/Users/yany/Downloads/Figure4_revamp_2026-05-29_continuousD')
DOWN.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
C_ASP, C_MUC = '#4CAF50', '#757575'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 8, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'mathtext.default': 'regular',
})

df = pd.read_csv(CSV)
D_a = df.loc[df.genus == 'Aspergillus', 'D_recon'].values
D_m = df.loc[df.genus == 'Mucor',       'D_recon'].values

ratio = D_a.mean() / D_m.mean()
p = stats.ttest_ind(D_a, D_m, equal_var=False).pvalue
print(f'D_rec  Asp = {D_a.mean():.4f} ± {D_a.std(ddof=1):.4f} (n={len(D_a)})')
print(f'       Muc = {D_m.mean():.4f} ± {D_m.std(ddof=1):.4f} (n={len(D_m)})')
print(f'       ratio = {ratio:.3f}x   p = {p:.2e}')

# Layout: a top "illustration" strip (small) + boxplot below, mirroring the
# published Panel E styling (compact tissue forms clusters / loose tissue
# spreads thinly).  No raster thumbnails — those live in the Inkscape SVG
# and stay as-is.
fig, ax = plt.subplots(figsize=(50 * MM, 80 * MM))
fig.subplots_adjust(left=0.27, right=0.96, top=0.92, bottom=0.13)

bp = ax.boxplot([D_a, D_m], positions=[1, 2], widths=0.4, patch_artist=True,
                showfliers=False,
                medianprops=dict(color='white', lw=1.4),
                whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
bp['boxes'][0].set_facecolor(C_ASP); bp['boxes'][0].set_alpha(0.55)
bp['boxes'][1].set_facecolor(C_MUC); bp['boxes'][1].set_alpha(0.55)
rng = np.random.default_rng(42)
ax.scatter(1 + rng.uniform(-0.09, 0.09, len(D_a)), D_a, s=18, c=C_ASP,
           alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
ax.scatter(2 + rng.uniform(-0.09, 0.09, len(D_m)), D_m, s=18, c=C_MUC,
           alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
ax.set_xticks([1, 2])
ax.set_xticklabels([r'$\it{Aspergillus}$', r'$\it{Mucor}$'], fontsize=7)
ax.set_ylabel(r'$f_\mathrm{tissue}$' + '\n(3D colony surface)',
              fontsize=10)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

ym = max(D_a.max(), D_m.max())
rg = max(ym - min(D_a.min(), D_m.min()), 1e-9)
y = ym + rg * 0.08
ax.plot([1, 1, 2, 2], [y, y + rg * 0.03, y + rg * 0.03, y], 'k-', lw=0.6)
ax.text(1.5, y + rg * 0.06, f'p = {p:.1e}', ha='center', fontsize=6)

for ext in ('.pdf', '.svg', '.png'):
    kw = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.03}
    if ext == '.png':
        kw['dpi'] = 600
    fig.savefig(OUT  / f'panel_E_Drec{ext}', **kw)
    fig.savefig(DOWN / f'panel_E_Drec{ext}', **kw)
plt.close(fig)
print(f'\nSaved panel_E_Drec.{{pdf,svg,png}} to:')
print(f'  {OUT}')
print(f'  {DOWN}')

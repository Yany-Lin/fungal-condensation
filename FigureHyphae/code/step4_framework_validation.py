#!/usr/bin/env python3
"""Legacy framework validation — 4 condensation observables + 4 morphological metrics.

The final source of truth is step_final_rebuild_all.py. This legacy script has
been updated to use Green vs White only; Black/failed Rhizopus is not pooled
into the Figure 4 species-level Mucor comparator.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_T7 = Path('/Volumes/T7/FINAL OSF')
_REPO = Path(__file__).resolve().parent.parent.parent
BASE = _T7 if _T7.exists() else _REPO
RSR = BASE / 'FigureRSR' / 'output' / 'rsr_and_lab_dstar_metrics.csv'
HG  = BASE / 'FigureHGAggregate' / 'output' / 'hydrogel_metrics.csv'
OUT = BASE / 'FigureHyphae' / 'output'
FINAL = BASE / 'HYPHAE' / 'Final Results'

MM = 1 / 25.4
C_ASP = '#4CAF50'
C_MUC = '#757575'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 8, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'svg.fonttype': 'none',
})


def welch(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt(((n1-1)*a.std(ddof=1)**2 + (n2-1)*b.std(ddof=1)**2) / (n1+n2-2))
    d = (a.mean() - b.mean()) / pooled if pooled > 0 else 0
    return {'t': t, 'p': p, 'd': d, 'n1': n1, 'n2': n2,
            'mean1': a.mean(), 'mean2': b.mean(),
            'sem1': a.std(ddof=1)/np.sqrt(n1), 'sem2': b.std(ddof=1)/np.sqrt(n2),
            'ratio': a.mean() / b.mean() if b.mean() != 0 else np.inf}


rsr = pd.read_csv(RSR)
hg = pd.read_csv(HG)
green = rsr[rsr['group'] == 'Green']
mucor = rsr[rsr['group'] == 'White']

# ── 4 condensation observables ──
cond_cols = [
    ('delta_um',    r'$\delta$',              'Dry-zone width'),
    ('zone_metric', 'Size\ngrad.',            'Size gradient'),
    ('tau50_zone',  r'$\tau_{50}$'+'\ngrad.', 'Survival gradient'),
    ('dstar_mm',    r'$d^*$',                 'Half-attenuation'),
]

print('=' * 70)
print('4 CONDENSATION OBSERVABLES — Green (n=5) vs White/Mucor (n=5)')
print('=' * 70)

cond_r = []
for col, label, desc in cond_cols:
    r = welch(green[col], mucor[col])
    cond_r.append(r)
    sig = '***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else 'ns'
    print(f'  {desc:20s}  ratio={r["ratio"]:.2f}  d={r["d"]:.2f}  p={r["p"]:.4g} {sig}')

# ── 4 morphological metrics ──
morph_data = [
    (r'FFT $\alpha$'+'\n(C)', 'FFT spectral slope alpha', 1.165, 1.85, 0.00144),
    ('Thick.\n(D)',   'Tissue thickness',      1.732, 7.58, 7.1e-11),
    ('Tub CV\n(E)',   'Hessian tubeness CV',   1.701, 2.93, 0.000973),
    ('Erosion\n(F)',  'Erosion survival',      2.127, 4.75, 7.9e-5),
]

print(f'\n4 MORPHOLOGICAL METRICS')
for _, desc, ratio, d, p in morph_data:
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f'  {desc:20s}  ratio={ratio:.2f}  d={d:.2f}  p={p:.4g} {sig}')

# Concordance
n_conc = 8
p_sign = stats.binomtest(n_conc, 8, 0.5, alternative='greater').pvalue
print(f'\nConcordance: 8/8 same direction, sign test p = {p_sign:.4f}')

# Calibration decomposition
x_cal = hg['one_minus_aw'].values
y_cal = hg['delta_um'].values
slope, intercept, r_val, _, _ = stats.linregress(x_cal, y_cal)
delta_ratio = green['delta_um'].mean() / mucor['delta_um'].mean()
capacity_ratio = 2.012
amplif = delta_ratio / capacity_ratio
print(f'\nδ ratio = {delta_ratio:.2f} = absorbing capacity({capacity_ratio:.2f}) × residual({amplif:.2f})')
print(f'Absorbing capacity explains {capacity_ratio/delta_ratio*100:.0f}%')

# Cross-correlations
print(f'\nCross-correlations (all trials):')
all_d = rsr['delta_um'].values
all_t = rsr['tau50_zone'].values
all_z = rsr['zone_metric'].values
all_ds = rsr['dstar_mm'].values
groups = rsr['group'].values

for c1v, c2v, lab in [(all_d, all_z, 'δ vs size grad'),
                        (all_d, all_t, 'δ vs surv grad'),
                        (all_d, all_ds, 'δ vs d*'),
                        (all_z, all_t, 'size vs surv grad')]:
    v = ~np.isnan(c1v) & ~np.isnan(c2v)
    rs, ps = stats.spearmanr(c1v[v], c2v[v])
    print(f'  {lab:20s}  r={rs:.3f}  p={ps:.1e}')


# ═══════════════════════════════════════════════════════════════
# WIDE FIGURE — 3 panels
# ═══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(190 * MM, 62 * MM))
gs = fig.add_gridspec(1, 3, left=0.07, right=0.97, top=0.88, bottom=0.20,
                      wspace=0.35, width_ratios=[1.6, 1, 0.7])
axes = [fig.add_subplot(gs[i]) for i in range(3)]

# ── Panel A: All 8 ratios ──
ax = axes[0]

# Labels and data
a_labels = []
a_ratios = []
a_colors = []
a_pvals = []

# 4 condensation
for (col, label, desc), r in zip(cond_cols, cond_r):
    a_labels.append(label)
    a_ratios.append(r['ratio'])
    a_colors.append(C_MUC)
    a_pvals.append(r['p'])

# 4 morphological
for label, desc, ratio, d, p in morph_data:
    a_labels.append(label)
    a_ratios.append(ratio)
    a_colors.append(C_ASP)
    a_pvals.append(p)

x_a = np.arange(len(a_ratios))
ax.bar(x_a, a_ratios, color=a_colors, alpha=0.7, width=0.65, edgecolor='none')
for xi, (r, p, c) in enumerate(zip(a_ratios, a_pvals, a_colors)):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ' n.s.'
    ax.text(xi, r + 0.03, f'{r:.2f}{sig}', ha='center', fontsize=5,
            fontweight='bold', color=c)

ax.axhline(1.0, color='gray', ls='--', lw=0.5, alpha=0.5)
ax.axvline(3.5, color='gray', ls='-', lw=0.4, alpha=0.3)
ax.set_xticks(x_a)
ax.set_xticklabels(a_labels, fontsize=5.5)
ax.set_ylabel('Asp-favored ratio', fontsize=7)
ax.set_ylim(0, max(a_ratios) * 1.22)
ax.text(1.5, max(a_ratios)*1.15, 'Condensation', ha='center', fontsize=6, color=C_MUC, style='italic')
ax.text(5.5, max(a_ratios)*1.15, 'Morphological', ha='center', fontsize=6, color='#2E7D32', style='italic')
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
ax.text(-0.08, 1.07, 'A', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

# ── Panel B: δ vs τ50 gradient ──
ax = axes[1]
cmap = {'Agar': '#90CAF9', '0.5:1': '#42A5F5', '1:1': '#1E88E5', '2:1': '#0D47A1',
        'Green': C_ASP, 'White': '#9E9E9E', 'Black': '#424242'}
mmap = {'Agar': 'o', '0.5:1': 'o', '1:1': 'o', '2:1': 'o',
        'Green': 'D', 'White': 'D', 'Black': 'D'}
for grp in ['Agar', '0.5:1', '1:1', '2:1', 'Green', 'White', 'Black']:
    m = (groups == grp) & ~np.isnan(all_d) & ~np.isnan(all_t)
    if not m.any():
        continue
    ax.scatter(all_d[m], all_t[m], c=cmap[grp], marker=mmap[grp],
               s=22, alpha=0.8, edgecolors='white', linewidths=0.3,
               label=grp, zorder=3)

vm = ~np.isnan(all_d) & ~np.isnan(all_t)
sl, ic, _, _, _ = stats.linregress(all_d[vm], all_t[vm])
xf = np.linspace(0, 1100, 100)
ax.plot(xf, ic + sl * xf, 'k--', lw=0.8, alpha=0.4)
rs, ps = stats.spearmanr(all_d[vm], all_t[vm])
ax.text(0.95, 0.08, f'r = {rs:.2f}\np = {ps:.1e}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=6,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax.set_xlabel(r'$\delta$ (µm)', fontsize=7)
ax.set_ylabel(r'$\tau_{50}$ gradient', fontsize=7)
ax.set_xlim(-20, 1100)
ax.legend(fontsize=4.5, loc='upper left', ncol=2, handletextpad=0.2,
          columnspacing=0.4, borderpad=0.3, framealpha=0.8)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
ax.text(-0.12, 1.07, 'B', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

# ── Panel C: Decomposition ──
ax = axes[2]
ax.bar([0], [capacity_ratio], color=C_ASP, alpha=0.7, width=0.5, edgecolor='none')
ax.bar([0], [delta_ratio - capacity_ratio], bottom=[capacity_ratio], color='#81C784', alpha=0.5,
       width=0.5, edgecolor='none')
ax.bar([1], [delta_ratio], color=C_MUC, alpha=0.7, width=0.5, edgecolor='none')

ax.text(0, capacity_ratio/2, f'{capacity_ratio:.2f}×\ncapacity', ha='center',
        fontsize=5.5, fontweight='bold', color='white', va='center')
ax.text(0, capacity_ratio + (delta_ratio-capacity_ratio)/2, f'{amplif:.2f}×\nresidual',
        ha='center', fontsize=5, color='#2E7D32', va='center')
ax.text(1, delta_ratio/2, f'{delta_ratio:.2f}×', ha='center',
        fontsize=7, fontweight='bold', color='white', va='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Predicted', 'Measured'], fontsize=6)
ax.set_ylabel('Asp-favored ratio', fontsize=7)
ax.axhline(1.0, color='gray', ls='--', lw=0.5, alpha=0.5)
ax.set_ylim(0, delta_ratio * 1.15)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
ax.text(-0.15, 1.07, 'C', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

for ext in ('.png', '.pdf', '.svg'):
    kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
    if ext == '.png':
        kw['dpi'] = 300
    fig.savefig(OUT / f'framework_validation_final{ext}', **kw)
plt.close(fig)
print(f'\nSaved: {OUT}/framework_validation_final.{{png,pdf,svg}}')

# CSV
rows = []
for (col, _, desc), r in zip(cond_cols, cond_r):
    rows.append({'type': 'condensation', 'metric': desc,
                 'asp': round(r['mean1'], 4), 'muc': round(r['mean2'], 4),
                 'ratio': round(r['ratio'], 3), 'd': round(r['d'], 2), 'p': round(r['p'], 6)})
for _, desc, ratio, d, p in morph_data:
    rows.append({'type': 'morphological', 'metric': desc,
                 'asp': None, 'muc': None, 'ratio': ratio, 'd': d, 'p': p})
pd.DataFrame(rows).to_csv(FINAL / 'framework_validation_final.csv', index=False)
print(f'Saved: {FINAL}/framework_validation_final.csv')

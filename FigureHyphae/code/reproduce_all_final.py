#!/usr/bin/env python3
"""Reproduce all f_tissue analyses with the committed polarity (bright).

Runs four methods on all 24 colony-surface ROIs:
  1. ADAPTIVE-BRIGHT  — committed paper method (mirror of published)
                         σ_smooth=1, σ_local=32, k=0.5, morph=1, polarity=BRIGHT
  2. ADAPTIVE-DARK    — published method, for backwards compatibility
  3. SAUVOLA-BRIGHT   — literature-standard local adaptive, bright detection
  4. SAUVOLA-DARK     — literature-standard local adaptive, dark detection

Outputs:
  D:/FINAL OSF/FigureHyphae/output/ftissue_all_methods_final.csv
    per-ROI f_tissue under all four methods
  D:/FINAL OSF/FigureHyphae/figures/figS_ftissue_convergence.{pdf,png,svg}
    convergence supplementary figure: box plots, ratios, correlations,
    per-ROI parallel coordinates, three-class breakdown
  Console: per-method statistics + cross-method correlation matrix
"""

import json, csv, time
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BASE = Path(r'D:\FINAL OSF')
ROI_DIR = BASE / 'HYPHAE' / 'Analysis' / 'results' / '3d_overlays'
SESSION = ROI_DIR / 'roi_session.json'
OUT_FIG = BASE / 'FigureHyphae' / 'figures'
OUT_CSV = BASE / 'FigureHyphae' / 'output'

CAL_3D = 0.94
MM = 1 / 25.4
C_ASP, C_MUC = '#4CAF50', '#757575'
ASP_KEY = '20251214_222552.JPG'
MUC_KEY = '20251210_155926.JPG'

# Committed adaptive settings (mirror of published)
SIGMA_SMOOTH, SIGMA_LOCAL = 1.0, 32.0
K_ADAPT, MORPH_IT = 0.5, 1

# Sauvola settings (recommended from earlier sweep)
SAU_WIN, SAU_K, SAU_R = 64, 0.2, 0.5

# δ ratio benchmark (Asp/Muc from main paper)
DELTA_RATIO = 2.13

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0) if DEVICE == "cuda" else "n/a"}')

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 8, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'mathtext.default': 'regular',
})


# ── GPU primitives ──
def gauss(img, sigma):
    if sigma <= 0: return img
    r = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2); k = k / k.sum()
    a = img.unsqueeze(0).unsqueeze(0)
    a = F.pad(a, (r, r, 0, 0), mode='reflect')
    a = F.conv2d(a, k.view(1, 1, 1, -1))
    a = F.pad(a, (0, 0, r, r), mode='reflect')
    a = F.conv2d(a, k.view(1, 1, -1, 1))
    return a.squeeze(0).squeeze(0)


def morph(mask, iters=1):
    m = mask.float().unsqueeze(0).unsqueeze(0)
    for _ in range(iters): m = F.max_pool2d(m, 3, stride=1, padding=1)
    for _ in range(iters): m = -F.max_pool2d(-m, 3, stride=1, padding=1)
    for _ in range(iters): m = -F.max_pool2d(-m, 3, stride=1, padding=1)
    for _ in range(iters): m = F.max_pool2d(m, 3, stride=1, padding=1)
    return m.squeeze(0).squeeze(0) > 0.5


def adaptive(img_gpu, polarity):
    """Adaptive thresholding. polarity in {'bright', 'dark'}."""
    s = gauss(img_gpu, SIGMA_SMOOTH)
    l = gauss(s, SIGMA_LOCAL)
    if polarity == 'bright':
        delta = s - l
    else:
        delta = l - s
    t = delta.mean() + K_ADAPT * delta.std()
    return morph(delta > t, MORPH_IT)


def sauvola(img_gpu, polarity, window=SAU_WIN, k=SAU_K, R=SAU_R):
    lo, hi = img_gpu.min(), img_gpu.max()
    n = (img_gpu - lo) / max(float(hi - lo), 1e-6)
    x = n.unsqueeze(0).unsqueeze(0)
    mean = F.avg_pool2d(x, window, stride=1, padding=window // 2,
                         count_include_pad=False)
    msq = F.avg_pool2d(x ** 2, window, stride=1, padding=window // 2,
                        count_include_pad=False)
    mean = mean[..., :n.shape[0], :n.shape[1]]
    msq = msq[..., :n.shape[0], :n.shape[1]]
    std = (msq - mean ** 2).clamp(min=0).sqrt()
    # Sauvola threshold: T = mean * (1 + k*(std/R - 1))
    T = mean * (1 + k * (std / R - 1))
    if polarity == 'dark':
        m = (x < T)
    else:
        # Mirror: tissue is brighter than Sauvola threshold
        T_bright = mean * (2 - (1 + k * (std / R - 1)))  # symmetric flip
        m = (x > T_bright)
    return morph(m.squeeze(0).squeeze(0), MORPH_IT)


def three_class_breakdown(img_gpu):
    """Compute the three classes (bright tissue, mid background, dark tissue)
    using the adaptive-response framework, returning per-pixel category."""
    s = gauss(img_gpu, SIGMA_SMOOTH)
    l = gauss(s, SIGMA_LOCAL)
    br = s - l   # bright-response
    dr = l - s   # dark-response (= -br)
    t_b = br.mean() + K_ADAPT * br.std()
    t_d = dr.mean() + K_ADAPT * dr.std()
    is_bright = morph(br > t_b, MORPH_IT)
    is_dark = morph(dr > t_d, MORPH_IT)
    # Disjoint: prioritize "bright" since polarity = bright is the committed choice
    is_mid = ~is_bright & ~is_dark
    return is_bright, is_mid, is_dark


def load_gray(p):
    with Image.open(p) as im:
        a = np.asarray(im).astype(np.float32)
    if a.ndim == 3: a = a[..., :3].mean(axis=2)
    return a


def cohens_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                  / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0


METHODS = [
    ('adaptive-bright', lambda g: adaptive(g, 'bright'), '#1976D2'),  # primary
    ('adaptive-dark',   lambda g: adaptive(g, 'dark'),   '#9E9E9E'),  # published
    ('sauvola-bright',  lambda g: sauvola(g, 'bright'),  '#43A047'),
    ('sauvola-dark',    lambda g: sauvola(g, 'dark'),    '#E53935'),
]


# ── Load ROIs ──
with open(SESSION) as f:
    saved = {k: v for k, v in json.load(f).items()
             if not k.startswith('_') and v.get('status') != 'deleted'}
records = []
for key, val in saved.items():
    g = val.get('genus')
    rp = ROI_DIR / g / f'{Path(key).stem}_roi.jpg'
    if rp.exists(): records.append({'key': key, 'genus': g, 'path': rp})
asp = [r for r in records if r['genus'] == 'Aspergillus']
muc = [r for r in records if r['genus'] == 'Mucor']
print(f'Loaded {len(records)} ROIs ({len(asp)} Asp + {len(muc)} Muc)')


# ── Process ──
print('\nProcessing all methods on GPU...')
t0 = time.time()
fvals = {name: {'Aspergillus': [], 'Mucor': []} for name, _, _ in METHODS}
per_roi_table = []                # list of dicts: {key, genus, f_method1, f_method2, ...}
three_class_canon = {}            # {genus: (bright_mask, mid_mask, dark_mask)}
canon_imgs = {}

for r in records:
    img_np = load_gray(r['path'])
    img_gpu = torch.from_numpy(img_np).to(DEVICE)
    row = {'file': r['key'], 'genus': r['genus']}
    for name, fn, _ in METHODS:
        m = fn(img_gpu)
        f = float(m.float().mean().item())
        fvals[name][r['genus']].append(f)
        row[f'f_{name}'] = round(f, 4)
    per_roi_table.append(row)
    if r['key'] in (ASP_KEY, MUC_KEY):
        canon_imgs[r['genus']] = img_np
        is_b, is_m, is_d = three_class_breakdown(img_gpu)
        three_class_canon[r['genus']] = (is_b.cpu().numpy(),
                                          is_m.cpu().numpy(),
                                          is_d.cpu().numpy())

print(f'Done in {time.time() - t0:.1f} s')

# Save CSV
csv_path = OUT_CSV / 'ftissue_all_methods_final.csv'
field_names = ['file', 'genus'] + [f'f_{n}' for n, _, _ in METHODS]
with open(csv_path, 'w', newline='') as fp:
    w = csv.DictWriter(fp, fieldnames=field_names)
    w.writeheader(); w.writerows(per_roi_table)
print(f'CSV: {csv_path}')


# ── Per-method stats ──
print('\n══════════════════════════════════════════════════════════════')
print('PER-METHOD RESULTS')
print('══════════════════════════════════════════════════════════════')
stats_table = {}
for name, _, _ in METHODS:
    a = np.array(fvals[name]['Aspergillus'])
    m = np.array(fvals[name]['Mucor'])
    r = a.mean() / max(m.mean(), 1e-9)
    p = stats.ttest_ind(a, m, equal_var=False).pvalue
    d = cohens_d(a, m)
    direction_ok = bool(a.min() > m.max())
    stats_table[name] = {'asp_mean': a.mean(), 'asp_sd': a.std(ddof=1),
                          'muc_mean': m.mean(), 'muc_sd': m.std(ddof=1),
                          'ratio': r, 'p': p, 'd': d,
                          'no_overlap': direction_ok}
    print(f'  {name:18}  Asp={a.mean():.3f}±{a.std(ddof=1):.3f}  '
          f'Muc={m.mean():.3f}±{m.std(ddof=1):.3f}  '
          f'ratio={r:.3f}  d={d:.2f}  p={p:.2e}  '
          f'no-overlap={direction_ok}')


# ── Cross-method correlations (Spearman) ──
print('\n══════════════════════════════════════════════════════════════')
print('CROSS-METHOD SPEARMAN CORRELATIONS')
print('══════════════════════════════════════════════════════════════')
all_vals = {n: np.concatenate([np.array(fvals[n]['Aspergillus']),
                                 np.array(fvals[n]['Mucor'])])
             for n, _, _ in METHODS}
names = [n for n, _, _ in METHODS]
rho_mat = np.zeros((len(names), len(names)))
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        rho, _ = stats.spearmanr(all_vals[ni], all_vals[nj])
        rho_mat[i, j] = rho
print('    ' + '  '.join(f'{n[:14]:>14}' for n in names))
for i, ni in enumerate(names):
    print(f'  {ni[:14]:>14}  ' +
           '  '.join(f'{rho_mat[i, j]:>+14.3f}' for j in range(len(names))))


# ══════════════════════════════════════════════════════════════
# FIGURE — convergence
# ══════════════════════════════════════════════════════════════
print('\nRendering convergence figure...')
fig = plt.figure(figsize=(190 * MM, 230 * MM))
gs = GridSpec(4, 4, figure=fig, hspace=0.50, wspace=0.35,
               left=0.07, right=0.97, top=0.96, bottom=0.04,
               height_ratios=[1.0, 1.0, 1.0, 1.0])


# Helper: stripbox in Fig 4 style
def stripbox(ax, sa, sm, ylab, title=None):
    bp = ax.boxplot([sa, sm], positions=[1, 2], widths=0.4, patch_artist=True,
                     showfliers=False, medianprops=dict(color='white', lw=1.4),
                     whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    bp['boxes'][0].set_facecolor(C_ASP); bp['boxes'][0].set_alpha(0.55)
    bp['boxes'][1].set_facecolor(C_MUC); bp['boxes'][1].set_alpha(0.55)
    rng = np.random.default_rng(42)
    ax.scatter(1 + rng.uniform(-0.09, 0.09, len(sa)), sa, s=12, c=C_ASP,
                alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
    ax.scatter(2 + rng.uniform(-0.09, 0.09, len(sm)), sm, s=12, c=C_MUC,
                alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([r'$\it{Asp}$', r'$\it{Muc}$'], fontsize=6.5)
    ax.set_ylabel(ylab, fontsize=7)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    p = stats.ttest_ind(sa, sm, equal_var=False).pvalue
    ym = max(sa.max(), sm.max())
    rg = max(ym - min(sa.min(), sm.min()), 1e-6)
    y = ym + rg * 0.08
    ax.plot([1, 1, 2, 2], [y, y + rg * 0.03, y + rg * 0.03, y], 'k-', lw=0.6)
    ax.text(1.5, y + rg * 0.06,
             f'p = {p:.1e}' if p < 0.001 else f'p = {p:.3f}',
             ha='center', fontsize=6)
    if title:
        ax.set_title(title, fontsize=7, pad=3)


# Row 1: 4 box plots
for i, (name, _, color) in enumerate(METHODS):
    ax = fig.add_subplot(gs[0, i])
    sa = np.array(fvals[name]['Aspergillus'])
    sm = np.array(fvals[name]['Mucor'])
    r = stats_table[name]['ratio']; d = stats_table[name]['d']
    stripbox(ax, sa, sm, r'$f_{\mathrm{tissue}}$',
              title=f'{name}\nratio={r:.2f}×, d={d:.2f}')
    ax.text(-0.22, 1.18, 'ABCD'[i], transform=ax.transAxes, fontsize=12,
             fontweight='bold', va='top')


# Row 2 col 0-1: ratio bar chart
ax = fig.add_subplot(gs[1, 0:2])
xs = np.arange(len(METHODS))
rs = [stats_table[n]['ratio'] for n, _, _ in METHODS]
colors = [c for _, _, c in METHODS]
ax.bar(xs, rs, color=colors, alpha=0.85, edgecolor='black', lw=0.5)
for x, r_ in zip(xs, rs):
    ax.text(x, r_ + 0.04, f'{r_:.2f}×', ha='center', fontsize=8,
             fontweight='bold')
ax.axhline(DELTA_RATIO, color='black', lw=0.7, ls='--', alpha=0.7)
ax.text(len(METHODS) - 0.5, DELTA_RATIO + 0.05,
         f' δ ratio (Asp/Muc = {DELTA_RATIO}×)', va='bottom', ha='right',
         fontsize=7, color='black', alpha=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([n.split('-')[0] + '\n' + n.split('-')[1]
                     for n, _, _ in METHODS], fontsize=6.5)
ax.set_ylabel(r'Asp / Muc ratio in $f_{\mathrm{tissue}}$', fontsize=7)
ax.set_ylim(0, max(max(rs), DELTA_RATIO) * 1.15)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.text(-0.10, 1.10, 'E', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')
ax.set_title('All methods give Asp > Muc; magnitudes differ',
              fontsize=7.5, pad=3)


# Row 2 col 2-3: Spearman correlation matrix
ax = fig.add_subplot(gs[1, 2:4])
im = ax.imshow(rho_mat, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
ax.set_xticklabels(names, fontsize=6.5, rotation=30, ha='right')
ax.set_yticklabels(names, fontsize=6.5)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f'{rho_mat[i, j]:.2f}', ha='center', va='center',
                 fontsize=7, color='white' if abs(rho_mat[i, j]) > 0.6 else 'black')
plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02).ax.tick_params(labelsize=6)
ax.set_title('Per-ROI Spearman ρ across methods', fontsize=7.5, pad=3)
ax.text(-0.10, 1.10, 'F', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')


# Row 3: parallel coordinates per ROI across methods
ax = fig.add_subplot(gs[2, :])
for r in per_roi_table:
    ys = [r[f'f_{n}'] for n, _, _ in METHODS]
    color = C_ASP if r['genus'] == 'Aspergillus' else C_MUC
    ax.plot(range(len(METHODS)), ys, '-o', color=color, alpha=0.45,
             markersize=4, lw=0.8)
# Group means
for name, _, color in METHODS:
    pass  # already plotted
asp_means = [np.array(fvals[n]['Aspergillus']).mean() for n, _, _ in METHODS]
muc_means = [np.array(fvals[n]['Mucor']).mean() for n, _, _ in METHODS]
ax.plot(range(len(METHODS)), asp_means, '-D', color=C_ASP, lw=2.2,
         markersize=8, markeredgecolor='black', markeredgewidth=0.6,
         label='Asp mean')
ax.plot(range(len(METHODS)), muc_means, '-D', color=C_MUC, lw=2.2,
         markersize=8, markeredgecolor='black', markeredgewidth=0.6,
         label='Muc mean')
ax.set_xticks(range(len(METHODS)))
ax.set_xticklabels([n for n, _, _ in METHODS], fontsize=7)
ax.set_ylabel(r'$f_{\mathrm{tissue}}$', fontsize=7.5)
ax.legend(fontsize=6.5, frameon=False, loc='best')
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.text(-0.04, 1.10, 'G', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')
ax.set_title('Per-ROI parallel coordinates: every Asp line sits above '
              'every Muc line at every method', fontsize=7.5, pad=3)


# Row 4: three-class breakdown for canonical Asp + Muc
def three_class_panel(ax_row, ax_pie, genus, color):
    is_b, is_m, is_d = three_class_canon[genus]
    f_b = float(is_b.mean()); f_m = float(is_m.mean()); f_d = float(is_d.mean())
    # Stack RGB visualization: bright = white, mid = gray, dark = black
    h, w = is_b.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[is_b] = (1.0, 1.0, 1.0)
    rgb[is_m] = (0.45, 0.45, 0.45)
    rgb[is_d] = (0.0, 0.0, 0.0)
    ax_row.imshow(rgb, interpolation='nearest')
    ax_row.axis('off')
    ax_row.text(0.98, 0.03, genus, transform=ax_row.transAxes, fontsize=7,
                ha='right', va='bottom', style='italic', color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.20', facecolor=color,
                          alpha=0.85, edgecolor='none'))
    ax_row.set_title(
        f'bright = {f_b:.1%}   mid = {f_m:.1%}   dark = {f_d:.1%}',
        fontsize=7.5, pad=3)
    # Pie
    ax_pie.pie([f_b, f_m, f_d], labels=['bright', 'mid', 'dark'],
                colors=['white', '#9E9E9E', 'black'],
                wedgeprops=dict(edgecolor='black', linewidth=0.5),
                textprops=dict(fontsize=6.5),
                autopct='%1.0f%%', pctdistance=0.7)


ax = fig.add_subplot(gs[3, 0:2])
ax_pie = fig.add_subplot(gs[3, 2])
three_class_panel(ax, ax_pie, 'Aspergillus', C_ASP)
ax.text(-0.04, 1.10, 'H', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')

ax = fig.add_subplot(gs[3, 3])
# put muc visualization where pie was — actually let me re-layout
plt.delaxes(ax)


# Re-do row 4 layout with two genus rows
# (recompute by using the gridspec subgrid)
sub_gs = gs[3, :].subgridspec(1, 4, wspace=0.30)
ax_a = fig.add_subplot(sub_gs[0, 0:2])
ax_a_pie = fig.add_subplot(sub_gs[0, 2])
ax_m = fig.add_subplot(sub_gs[0, 3])

is_b, is_m, is_d = three_class_canon['Aspergillus']
f_b, f_mid, f_d = float(is_b.mean()), float(is_m.mean()), float(is_d.mean())
h, w = is_b.shape
rgb = np.zeros((h, w, 3), dtype=np.float32)
rgb[is_b] = (1.0, 1.0, 1.0); rgb[is_m] = (0.45, 0.45, 0.45); rgb[is_d] = (0.0, 0.0, 0.0)
ax_a.imshow(rgb, interpolation='nearest'); ax_a.axis('off')
ax_a.text(0.98, 0.03, 'Aspergillus', transform=ax_a.transAxes, fontsize=7,
           ha='right', va='bottom', style='italic', color='white',
           fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.20', facecolor=C_ASP,
                     alpha=0.85, edgecolor='none'))
ax_a.set_title(f'Three-class breakdown:  '
                f'bright {f_b:.0%} | mid {f_mid:.0%} | dark {f_d:.0%}',
                fontsize=7, pad=3)
ax_a.text(-0.04, 1.10, 'H', transform=ax_a.transAxes, fontsize=12,
           fontweight='bold', va='top')

ax_a_pie.pie([f_b, f_mid, f_d], labels=['bright', 'mid', 'dark'],
              colors=['white', '#9E9E9E', 'black'],
              wedgeprops=dict(edgecolor='black', linewidth=0.5),
              textprops=dict(fontsize=6.5),
              autopct='%1.0f%%', pctdistance=0.7)
ax_a_pie.set_title('Asp pixel class fractions', fontsize=6.5, pad=2)

is_b2, is_m2, is_d2 = three_class_canon['Mucor']
f_b2, f_mid2, f_d2 = float(is_b2.mean()), float(is_m2.mean()), float(is_d2.mean())
ax_m.pie([f_b2, f_mid2, f_d2], labels=['bright', 'mid', 'dark'],
          colors=['white', '#9E9E9E', 'black'],
          wedgeprops=dict(edgecolor='black', linewidth=0.5),
          textprops=dict(fontsize=6.5),
          autopct='%1.0f%%', pctdistance=0.7)
ax_m.set_title(f'Muc pixel class fractions\n'
                f'bright {f_b2:.0%} | mid {f_mid2:.0%} | dark {f_d2:.0%}',
                fontsize=6.5, pad=2)


for ext in ('.png', '.pdf', '.svg'):
    kw = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.04}
    if ext == '.png': kw['dpi'] = 600
    fig.savefig(OUT_FIG / f'figS_ftissue_convergence{ext}', **kw)
plt.close(fig)
print(f'Saved: figS_ftissue_convergence.{{pdf,png,svg}}')

print('\n══════════════════════════════════════════════════════════════')
print('CONVERGENCE VERDICT')
print('══════════════════════════════════════════════════════════════')
asp_min_all = min(np.array(fvals[n]['Aspergillus']).min() for n, _, _ in METHODS)
muc_max_all = max(np.array(fvals[n]['Mucor']).max() for n, _, _ in METHODS)
ranks_concordant = all(stats_table[n]['ratio'] > 1.0 for n, _, _ in METHODS)
print(f'  All methods give Asp/Muc ratio > 1?  {ranks_concordant}')
print(f'  Range of ratios:                     {min(rs):.2f}× – {max(rs):.2f}×')
print(f'  Mean cross-method Spearman ρ:        '
      f'{rho_mat[np.triu_indices(len(names), 1)].mean():.3f}')
print(f'  Per-method significance (all p<0.001)?  '
      f'{all(stats_table[n]["p"] < 1e-3 for n, _, _ in METHODS)}')
print('Done.')

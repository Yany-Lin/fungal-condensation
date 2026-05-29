#!/usr/bin/env python3
"""Continuous multi-scale bright-tissue-density metric (gradient-based).

Replaces binary thresholding with a CONTINUOUS measure that:
  - weights bright pixels heavily (thicker, whiter hyphae contribute more)
  - de-weights grays (intermediate brightness contributes less)
  - excludes darks (negative bright-response clipped to zero)
  - uses multiple spatial scales (σ = 8, 16, 32, 64, 128 px = 7.5–120 µm) so
    both thin filaments and dense patches add to the score
  - no hard threshold → no information loss from gating

Per-pixel density:   d(x,y) = mean_σ max(0, I_norm - gauss(I_norm, σ))
Per-ROI scalar:      D = mean(d(x,y))   (units: normalized intensity)

Outputs:
  D:/FINAL OSF/FigureHyphae/output/ftissue_continuous_density.csv
  D:/FINAL OSF/FigureHyphae/figures/figS_continuous_density.{pdf,png,svg}
  D:/FINAL OSF/FigureHyphae/figures/preview_fullres/density_{asp,muc}_heatmap.png
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
from matplotlib.colors import LinearSegmentedColormap

BASE = Path(r'D:\FINAL OSF')
ROI_DIR = BASE / 'HYPHAE' / 'Analysis' / 'results' / '3d_overlays'
SESSION = ROI_DIR / 'roi_session.json'
OUT_FIG = BASE / 'FigureHyphae' / 'figures'
OUT_CSV = BASE / 'FigureHyphae' / 'output'
PREV_DIR = OUT_FIG / 'preview_fullres'
PREV_DIR.mkdir(parents=True, exist_ok=True)

C_ASP, C_MUC = '#4CAF50', '#757575'
ASP_KEY = '20251214_222552.JPG'
MUC_KEY = '20251210_155926.JPG'
MM = 1 / 25.4
CAL_3D = 0.94

# Multi-scale background sigmas (px)
SCALES = (8, 16, 32, 64, 128)
SIGMA_SMOOTH = 1.0
DELTA_RATIO = 2.13   # δ benchmark

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
    H, W = img.shape
    # Cap kernel radius so reflect-padding stays within image dimensions
    r = min(r, H - 1, W - 1)
    x = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2); k = k / k.sum()
    a = img.unsqueeze(0).unsqueeze(0)
    a = F.pad(a, (r, r, 0, 0), mode='reflect')
    a = F.conv2d(a, k.view(1, 1, 1, -1))
    a = F.pad(a, (0, 0, r, r), mode='reflect')
    a = F.conv2d(a, k.view(1, 1, -1, 1))
    return a.squeeze(0).squeeze(0)


def normalize01(img_gpu):
    lo, hi = img_gpu.min(), img_gpu.max()
    return (img_gpu - lo) / max(float(hi - lo), 1e-6)


def multiscale_bright_density(img_gpu, sigmas=SCALES):
    """Per-pixel continuous bright-tissue density (GPU).
    Returns a 2-D tensor of the same shape as img_gpu, with values in [0, 1].
    """
    n = normalize01(img_gpu)
    s = gauss(n, SIGMA_SMOOTH)
    per_scale = []
    for sig in sigmas:
        bg = gauss(s, sig)
        br = torch.clamp(s - bg, min=0)
        per_scale.append(br)
    density = torch.stack(per_scale).mean(0)  # average across scales
    return density


def load_gray(p):
    with Image.open(p) as im:
        a = np.asarray(im).astype(np.float32)
    if a.ndim == 3: a = a[..., :3].mean(axis=2)
    return a


def cohens_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                  / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0


# ── Load and process ──
with open(SESSION) as f:
    saved = {k: v for k, v in json.load(f).items()
             if not k.startswith('_') and v.get('status') != 'deleted'}
records = []
for key, val in saved.items():
    g = val.get('genus')
    rp = ROI_DIR / g / f'{Path(key).stem}_roi.jpg'
    if rp.exists(): records.append({'key': key, 'genus': g, 'path': rp})
asp_recs = [r for r in records if r['genus'] == 'Aspergillus']
muc_recs = [r for r in records if r['genus'] == 'Mucor']
print(f'Loaded {len(records)} ROIs ({len(asp_recs)} Asp + {len(muc_recs)} Muc)')

print(f'\nMulti-scale: σ = {SCALES} px = {[s*CAL_3D for s in SCALES]} µm')
print('Processing on GPU...')
t0 = time.time()

density_a, density_m = [], []
canon_density = {}        # genus -> density map (numpy)
canon_imgs = {}           # genus -> raw img (numpy)

for r in records:
    img_np = load_gray(r['path'])
    img_gpu = torch.from_numpy(img_np).to(DEVICE)
    dens = multiscale_bright_density(img_gpu)
    scalar = float(dens.mean().item())
    r['D'] = scalar
    (density_a if r['genus'] == 'Aspergillus' else density_m).append(scalar)
    if r['key'] in (ASP_KEY, MUC_KEY):
        canon_imgs[r['genus']] = img_np
        canon_density[r['genus']] = dens.cpu().numpy()

elapsed = time.time() - t0
density_a = np.array(density_a); density_m = np.array(density_m)
print(f'Done in {elapsed:.1f} s')

# Stats
ratio = density_a.mean() / max(density_m.mean(), 1e-9)
p = stats.ttest_ind(density_a, density_m, equal_var=False).pvalue
d = cohens_d(density_a, density_m)
print('\n══════════════════════════════════════════════════════════════')
print('CONTINUOUS BRIGHT-DENSITY RESULTS')
print('══════════════════════════════════════════════════════════════')
print(f'  Asp:   {density_a.mean():.4f} ± {density_a.std(ddof=1):.4f}  '
       f'(min={density_a.min():.4f}, max={density_a.max():.4f})')
print(f'  Muc:   {density_m.mean():.4f} ± {density_m.std(ddof=1):.4f}  '
       f'(min={density_m.min():.4f}, max={density_m.max():.4f})')
print(f'  Ratio: {ratio:.3f}×,  Cohen d = {d:.2f},  Welch p = {p:.2e}')
print(f'  No-overlap (all Asp > all Muc)? {density_a.min() > density_m.max()}')
print(f'  δ benchmark for reference:      {DELTA_RATIO}×')

# CSV
csv_path = OUT_CSV / 'ftissue_continuous_density.csv'
with open(csv_path, 'w', newline='') as fp:
    w = csv.DictWriter(fp, fieldnames=['file', 'genus', 'D_continuous'])
    w.writeheader()
    for r in records:
        w.writerow({'file': r['key'], 'genus': r['genus'],
                     'D_continuous': round(r['D'], 6)})
print(f'\nCSV: {csv_path}')


# ── Figure: box plot + canonical heatmaps + parallel coords + scaling ──
fig = plt.figure(figsize=(190 * MM, 200 * MM))
gs = GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.30,
               left=0.06, right=0.97, top=0.96, bottom=0.04,
               height_ratios=[1.0, 1.0, 1.0, 1.0])

# A: continuous density box plot
ax = fig.add_subplot(gs[0, 0])
bp = ax.boxplot([density_a, density_m], positions=[1, 2], widths=0.4,
                 patch_artist=True, showfliers=False,
                 medianprops=dict(color='white', lw=1.4),
                 whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
bp['boxes'][0].set_facecolor(C_ASP); bp['boxes'][0].set_alpha(0.55)
bp['boxes'][1].set_facecolor(C_MUC); bp['boxes'][1].set_alpha(0.55)
rng = np.random.default_rng(42)
ax.scatter(1 + rng.uniform(-0.09, 0.09, len(density_a)), density_a,
            s=14, c=C_ASP, alpha=0.85, edgecolors='white',
            linewidths=0.3, zorder=3)
ax.scatter(2 + rng.uniform(-0.09, 0.09, len(density_m)), density_m,
            s=14, c=C_MUC, alpha=0.85, edgecolors='white',
            linewidths=0.3, zorder=3)
ax.set_xticks([1, 2])
ax.set_xticklabels([r'$\it{Aspergillus}$', r'$\it{Mucor}$'], fontsize=6.5)
ax.set_ylabel(r'$D$ (continuous bright density)', fontsize=7)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ym = max(density_a.max(), density_m.max())
rg = max(ym - min(density_a.min(), density_m.min()), 1e-6)
y = ym + rg * 0.08
ax.plot([1, 1, 2, 2], [y, y + rg * 0.03, y + rg * 0.03, y], 'k-', lw=0.6)
ax.text(1.5, y + rg * 0.06,
         f'p = {p:.1e}' if p < 0.001 else f'p = {p:.3f}',
         ha='center', fontsize=6)
ax.set_title(f'ratio = {ratio:.2f}×,  $d$ = {d:.2f}', fontsize=7, pad=3)
ax.text(-0.22, 1.18, 'A', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')

# B: scaling — D vs ROI index, sorted, colored by genus
ax = fig.add_subplot(gs[0, 1])
all_D = sorted([(r['D'], r['genus']) for r in records])
xs = np.arange(len(all_D))
for x, (D_, g) in enumerate(all_D):
    c = C_ASP if g == 'Aspergillus' else C_MUC
    ax.bar(x, D_, color=c, alpha=0.85, edgecolor='black', lw=0.4)
ax.set_xticks([])
ax.set_xlabel('ROIs sorted by $D$', fontsize=7)
ax.set_ylabel(r'$D$', fontsize=7)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.text(-0.22, 1.18, 'B', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')
ax.set_title('Per-ROI continuous density\n(green = Asp, gray = Muc)',
              fontsize=7, pad=3)

# C: ratio vs δ benchmark
ax = fig.add_subplot(gs[0, 2])
ax.bar([0], [ratio], color='#1976D2', alpha=0.85, edgecolor='black',
        lw=0.5, width=0.55)
ax.text(0, ratio + 0.05, f'{ratio:.2f}×', ha='center', fontsize=10,
         fontweight='bold')
ax.axhline(DELTA_RATIO, color='black', lw=0.7, ls='--', alpha=0.7)
ax.text(0.55, DELTA_RATIO, f' δ ({DELTA_RATIO}×)',
         va='center', ha='left', fontsize=7)
ax.set_xticks([0])
ax.set_xticklabels(['continuous\ndensity'], fontsize=7)
ax.set_ylabel('Asp / Muc ratio', fontsize=7)
ax.set_ylim(0, max(ratio, DELTA_RATIO) * 1.18)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.text(-0.22, 1.18, 'C', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')

# D: scale decomposition for canonical Asp
img_gpu = torch.from_numpy(canon_imgs['Aspergillus']).to(DEVICE)
n = normalize01(img_gpu); s = gauss(n, SIGMA_SMOOTH)
per_scale_a = []
for sig in SCALES:
    bg = gauss(s, sig)
    per_scale_a.append(float(torch.clamp(s - bg, min=0).mean().item()))
img_gpu = torch.from_numpy(canon_imgs['Mucor']).to(DEVICE)
n = normalize01(img_gpu); s = gauss(n, SIGMA_SMOOTH)
per_scale_m = []
for sig in SCALES:
    bg = gauss(s, sig)
    per_scale_m.append(float(torch.clamp(s - bg, min=0).mean().item()))

ax = fig.add_subplot(gs[0, 3])
xs = np.arange(len(SCALES))
ax.plot(xs, per_scale_a, '-o', color=C_ASP, ms=6, lw=1.4, label='Asp canon')
ax.plot(xs, per_scale_m, '-o', color=C_MUC, ms=6, lw=1.4, label='Muc canon')
ax.set_xticks(xs)
ax.set_xticklabels([f'{s}' for s in SCALES], fontsize=6.5)
ax.set_xlabel('scale σ (px)', fontsize=7)
ax.set_ylabel('per-scale bright excess', fontsize=7)
ax.legend(fontsize=6.5, frameon=False)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.text(-0.22, 1.18, 'D', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')
ax.set_title('Scale decomposition\n(canonical ROIs)', fontsize=7, pad=3)


# Row 2-3: heatmaps for canonical Asp + Muc (raw + density)
def render_density_rows(row_start, genus, color, label_pair):
    img = canon_imgs[genus]
    dens = canon_density[genus]
    lo, hi = np.percentile(img, [0.5, 99.5])
    nd = np.clip((img - lo) / max(hi - lo, 1), 0, 1)

    # Raw
    ax = fig.add_subplot(gs[row_start, 0:2])
    ax.imshow(nd, cmap='gray', interpolation='nearest'); ax.axis('off')
    ax.text(0.98, 0.03, genus, transform=ax.transAxes, fontsize=8,
             ha='right', va='bottom', style='italic', color='white',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.20', facecolor=color,
                       alpha=0.85, edgecolor='none'))
    ax.set_title('Raw ROI', fontsize=8, pad=3)
    ax.text(-0.02, 1.10, label_pair[0], transform=ax.transAxes, fontsize=12,
             fontweight='bold', va='top')

    # Continuous density heatmap (viridis)
    ax = fig.add_subplot(gs[row_start, 2:4])
    vmax = max(canon_density['Aspergillus'].max(),
                canon_density['Mucor'].max())
    im = ax.imshow(dens, cmap='viridis', interpolation='nearest',
                    vmin=0, vmax=vmax)
    ax.axis('off')
    ax.set_title(f'Continuous bright-tissue density   '
                  f'($D = {dens.mean():.4f}$)', fontsize=8, pad=3)
    ax.text(-0.02, 1.10, label_pair[1], transform=ax.transAxes, fontsize=12,
             fontweight='bold', va='top')
    return im


render_density_rows(1, 'Aspergillus', C_ASP, ('E', 'F'))
last_im = render_density_rows(2, 'Mucor', C_MUC, ('G', 'H'))
# Add a colorbar on the right
cbar_ax = fig.add_axes([0.92, 0.30, 0.015, 0.30])
cb = plt.colorbar(last_im, cax=cbar_ax)
cb.set_label(r'per-pixel density $d(x,y)$', fontsize=7)
cb.ax.tick_params(labelsize=6)


# Row 4: simple summary table-as-text
ax = fig.add_subplot(gs[3, :])
ax.axis('off')
summary_lines = [
    r'$\bf{Continuous\ multi-scale\ bright-tissue\ density}$ (this work)',
    '',
    rf'Per-pixel density   $d(x,y) = \langle \max(0,\ I_{{norm}} - \mathrm{{gauss}}(I_{{norm}}, \sigma)) \rangle_\sigma$,'
    rf'   $\sigma \in \{{{",".join(str(s) for s in SCALES)}\}}$ px',
    rf'Per-ROI scalar     $D = \langle d(x,y) \rangle$  over the full ROI',
    '',
    rf'Aspergillus:  $D = {density_a.mean():.4f} \pm {density_a.std(ddof=1):.4f}$  ($n = {len(density_a)}$)',
    rf'Mucor:        $D = {density_m.mean():.4f} \pm {density_m.std(ddof=1):.4f}$  ($n = {len(density_m)}$)',
    rf'Ratio:        ${ratio:.3f}\times$,    Cohen $d = {d:.2f}$,    Welch $p = {p:.2e}$',
    rf'No overlap?  {density_a.min() > density_m.max()}    ($\delta$ benchmark = {DELTA_RATIO}$\times$)',
    '',
    r'Compared to binary thresholding: brighter pixels contribute more,'
    r' grays contribute less, blacks contribute zero — no gating, all the gradient is retained.',
]
y = 0.95
for line in summary_lines:
    ax.text(0.04, y, line, transform=ax.transAxes, fontsize=7.5,
             va='top', ha='left')
    y -= 0.10


for ext in ('.png', '.pdf', '.svg'):
    kw = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.04}
    if ext == '.png': kw['dpi'] = 600
    fig.savefig(OUT_FIG / f'figS_continuous_density{ext}', **kw)
plt.close(fig)
print(f'Saved: figS_continuous_density.{{pdf,png,svg}}')


# ── Full-resolution density heatmaps for canonical ROIs ──
for genus, color in [('Aspergillus', C_ASP), ('Mucor', C_MUC)]:
    img = canon_imgs[genus]
    dens = canon_density[genus]
    lo, hi = np.percentile(img, [0.5, 99.5])
    nd = np.clip((img - lo) / max(hi - lo, 1), 0, 1)
    # Use figsize that gives native resolution
    h, w = img.shape
    fig_h = 12  # inches
    fig_w = fig_h * (2 * w + 8) / h
    fig2, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                                gridspec_kw={'wspace': 0.02})
    fig2.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.005)
    axes[0].imshow(nd, cmap='gray', interpolation='nearest'); axes[0].axis('off')
    axes[0].set_title('Raw ROI', fontsize=14)
    im2 = axes[1].imshow(dens, cmap='viridis', interpolation='nearest',
                          vmin=0, vmax=max(canon_density['Aspergillus'].max(),
                                            canon_density['Mucor'].max()))
    axes[1].axis('off')
    axes[1].set_title(
        f'Continuous bright-tissue density  ($D = {dens.mean():.4f}$)',
        fontsize=14)
    out_name = f'density_{"asp" if genus == "Aspergillus" else "muc"}_heatmap.png'
    fig2.savefig(PREV_DIR / out_name, dpi=200, bbox_inches='tight',
                  facecolor='white')
    plt.close(fig2)
    print(f'  Full-res heatmap: {PREV_DIR / out_name}')

print('\nDone.')

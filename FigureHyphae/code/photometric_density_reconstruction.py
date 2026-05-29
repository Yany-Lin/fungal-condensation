#!/usr/bin/env python3
"""Photometric density reconstruction — combines multi-scale brightness,
gradient energy (drop-offs), and structure-tensor coherence into a continuous
per-pixel tissue-density field.

Per-pixel: R(x,y) = mean_σ [ B_σ · (1 + α · norm(G_σ) · norm(C_σ)) ]
  B_σ — brightness magnitude (clipped positive bright-excess at scale σ)
  G_σ — gradient magnitude smoothed at scale σ
  C_σ — structure-tensor coherence (1 = filament, 0 = isotropic)
Per-ROI scalar D_recon = ⟨R(x,y)⟩.

Inputs: 24 colony-surface ROIs at 0.94 µm/px
Outputs:
  output/ftissue_photometric.csv                       — per-ROI D_recon
  figures/figS_photometric_reconstruction.{pdf,png,svg} — supplement figure
  figures/preview_fullres/reconstruction_{asp,muc}_heatmap.png
  figures/preview_fullres/reconstruction_{asp,muc}_3d.png

GPU: RTX 5070 Ti. Target <10 s end-to-end.
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from matplotlib.gridspec import GridSpec

BASE = Path(r'D:\FINAL OSF')
ROI_DIR = BASE / 'HYPHAE' / 'Analysis' / 'results' / '3d_overlays'
SESSION = ROI_DIR / 'roi_session.json'
OUT_FIG = BASE / 'FigureHyphae' / 'figures'
OUT_CSV = BASE / 'FigureHyphae' / 'output'
PREV_DIR = OUT_FIG / 'preview_fullres'
PREV_DIR.mkdir(parents=True, exist_ok=True)

CAL_3D = 0.94
MM = 1 / 25.4
C_ASP, C_MUC = '#4CAF50', '#757575'
ASP_KEY = '20251214_222552.JPG'
MUC_KEY = '20251210_155926.JPG'

# Same scales as the continuous-density metric, for direct comparison
SCALES = (8, 16, 32, 64, 128)
SIGMA_SMOOTH = 1.0
ALPHA = 1.0                 # gradient × coherence boost factor
DELTA_RATIO = 2.13

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}, '
       f'GPU: {torch.cuda.get_device_name(0) if DEVICE == "cuda" else "n/a"}')

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


def central_grad(img):
    """∂x, ∂y via central differences."""
    H, W = img.shape
    gx = torch.zeros_like(img); gy = torch.zeros_like(img)
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    return gx, gy


def norm_pos(t, eps=1e-9):
    """Normalize a non-negative tensor to [0,1] by its own max."""
    m = float(t.max().item())
    return t / max(m, eps)


def reconstruct_density(img_gpu, sigmas=SCALES, alpha=ALPHA):
    """Per-pixel photometric tissue-density reconstruction (multi-scale)."""
    n = normalize01(img_gpu)
    s = gauss(n, SIGMA_SMOOTH)
    gx, gy = central_grad(s)
    grad_mag = (gx * gx + gy * gy).sqrt()

    R_accum = torch.zeros_like(s)
    eps = 1e-9
    for sig in sigmas:
        bg = gauss(s, sig)
        B = torch.clamp(s - bg, min=0)                                # brightness
        G = gauss(grad_mag, sig)                                       # smoothed gradient
        Sxx = gauss(gx * gx, sig); Syy = gauss(gy * gy, sig)
        Sxy = gauss(gx * gy, sig)
        # 2x2 eigenvalues
        tr = Sxx + Syy
        disc = torch.sqrt(torch.clamp((Sxx - Syy) ** 2 + 4 * Sxy * Sxy,
                                        min=0))
        l1 = (tr + disc) / 2.0
        l2 = (tr - disc) / 2.0
        coherence = ((l1 - l2) / (l1 + l2 + eps)) ** 2                 # ∈ [0,1]
        # Normalize G and C to [0,1] for blending; B stays in its natural units
        Gn = norm_pos(G); Cn = norm_pos(coherence)
        R_sigma = B * (1.0 + alpha * Gn * Cn)
        R_accum = R_accum + R_sigma
    R = R_accum / len(sigmas)
    return R


def load_gray(p):
    with Image.open(p) as im:
        a = np.asarray(im).astype(np.float32)
    if a.ndim == 3: a = a[..., :3].mean(axis=2)
    return a


def cohens_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                  / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0


# ── Load ROIs ──
with open(SESSION) as f:
    saved = {k: v for k, v in json.load(f).items()
             if not k.startswith('_') and v.get('status') != 'deleted'}
records = []
for key, val in saved.items():
    g = val.get('genus')
    rp = ROI_DIR / g / f'{Path(key).stem}_roi.jpg'
    if rp.exists():
        records.append({'key': key, 'genus': g, 'path': rp})
print(f'Loaded {len(records)} ROIs')


# ── Process ──
print(f'\nMulti-scale: σ = {SCALES} px = {[s*CAL_3D for s in SCALES]} µm')
print(f'Alpha (gradient × coherence boost): {ALPHA}')
print('Processing on GPU...')
t0 = time.time()
D_a, D_m = [], []
canon = {}
for r in records:
    img_np = load_gray(r['path'])
    img_gpu = torch.from_numpy(img_np).to(DEVICE)
    R = reconstruct_density(img_gpu)
    D = float(R.mean().item())
    r['D_recon'] = D
    (D_a if r['genus'] == 'Aspergillus' else D_m).append(D)
    if r['key'] in (ASP_KEY, MUC_KEY):
        canon[r['genus']] = {
            'img': img_np,
            'R': R.cpu().numpy(),
        }
print(f'Done in {time.time() - t0:.1f} s')

D_a, D_m = np.array(D_a), np.array(D_m)
ratio = D_a.mean() / max(D_m.mean(), 1e-9)
p = stats.ttest_ind(D_a, D_m, equal_var=False).pvalue
d = cohens_d(D_a, D_m)
no_overlap = bool(D_a.min() > D_m.max())

print('\n══════════════════════════════════════════════════════════════')
print('PHOTOMETRIC RECONSTRUCTION RESULTS')
print('══════════════════════════════════════════════════════════════')
print(f'  Asp:   {D_a.mean():.4f} ± {D_a.std(ddof=1):.4f}  (n={len(D_a)})')
print(f'  Muc:   {D_m.mean():.4f} ± {D_m.std(ddof=1):.4f}  (n={len(D_m)})')
print(f'  Ratio: {ratio:.3f}×,  Cohen d = {d:.2f},  Welch p = {p:.2e}')
print(f'  No-overlap (all Asp > all Muc)? {no_overlap}')
print(f'  δ benchmark:                    {DELTA_RATIO}×')
gap_pct = abs(ratio - DELTA_RATIO) / DELTA_RATIO * 100
print(f'  Distance from δ: {gap_pct:.1f}%')

# CSV
csv_path = OUT_CSV / 'ftissue_photometric.csv'
with open(csv_path, 'w', newline='') as fp:
    w = csv.DictWriter(fp, fieldnames=['file', 'genus', 'D_recon'])
    w.writeheader()
    for r in records:
        w.writerow({'file': r['key'], 'genus': r['genus'],
                     'D_recon': round(r['D_recon'], 6)})
print(f'\nCSV: {csv_path}')


# ── Cross-correlation with existing methods ──
# Load existing CSVs
import csv as _csv
prev_cont = {row['file']: float(row['D_continuous'])
             for row in _csv.DictReader(open(OUT_CSV / 'ftissue_continuous_density.csv'))}
prev_bin = list(_csv.DictReader(open(OUT_CSV / 'ftissue_all_methods_final.csv')))
prev_bin_map = {row['file']: row for row in prev_bin}

methods = ['D_recon', 'D_continuous',
           'f_adaptive-bright', 'f_adaptive-dark',
           'f_sauvola-bright', 'f_sauvola-dark']
all_vals = {m: [] for m in methods}
for r in records:
    all_vals['D_recon'].append(r['D_recon'])
    all_vals['D_continuous'].append(prev_cont.get(r['key'], np.nan))
    for k in ('f_adaptive-bright', 'f_adaptive-dark',
              'f_sauvola-bright', 'f_sauvola-dark'):
        all_vals[k].append(float(prev_bin_map[r['key']][k]))

print('\nSpearman ρ of D_recon with existing methods:')
for k in methods[1:]:
    rho, _ = stats.spearmanr(all_vals['D_recon'], all_vals[k])
    print(f'  vs {k:24}: ρ = {rho:+.3f}')


# ── Figure ──
print('\nRendering supplement figure...')
fig = plt.figure(figsize=(190 * MM, 230 * MM))
gs = GridSpec(4, 4, figure=fig, hspace=0.50, wspace=0.32,
               left=0.06, right=0.96, top=0.96, bottom=0.04,
               height_ratios=[1.0, 1.0, 1.1, 1.1])

# A: D_recon box plot
ax = fig.add_subplot(gs[0, 0])
bp = ax.boxplot([D_a, D_m], positions=[1, 2], widths=0.4, patch_artist=True,
                 showfliers=False, medianprops=dict(color='white', lw=1.4),
                 whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
bp['boxes'][0].set_facecolor(C_ASP); bp['boxes'][0].set_alpha(0.55)
bp['boxes'][1].set_facecolor(C_MUC); bp['boxes'][1].set_alpha(0.55)
rng = np.random.default_rng(42)
ax.scatter(1 + rng.uniform(-0.09, 0.09, len(D_a)), D_a, s=14, c=C_ASP,
            alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
ax.scatter(2 + rng.uniform(-0.09, 0.09, len(D_m)), D_m, s=14, c=C_MUC,
            alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
ax.set_xticks([1, 2])
ax.set_xticklabels([r'$\it{Asp}$', r'$\it{Muc}$'], fontsize=6.5)
ax.set_ylabel(r'$D_{\mathrm{recon}}$', fontsize=8)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ym = max(D_a.max(), D_m.max())
rg = max(ym - min(D_a.min(), D_m.min()), 1e-9)
y = ym + rg * 0.08
ax.plot([1, 1, 2, 2], [y, y + rg * 0.03, y + rg * 0.03, y], 'k-', lw=0.6)
ax.text(1.5, y + rg * 0.06,
         f'p = {p:.1e}' if p < 0.001 else f'p = {p:.3f}',
         ha='center', fontsize=6)
ax.set_title(f'ratio = {ratio:.2f}×,  $d$ = {d:.2f}', fontsize=7, pad=3)
ax.text(-0.22, 1.16, 'A', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')

# B: per-ROI sorted by D_recon
ax = fig.add_subplot(gs[0, 1])
sorted_recs = sorted(records, key=lambda r: r['D_recon'])
for i, r in enumerate(sorted_recs):
    c = C_ASP if r['genus'] == 'Aspergillus' else C_MUC
    ax.bar(i, r['D_recon'], color=c, alpha=0.85, edgecolor='black', lw=0.4)
ax.set_xticks([])
ax.set_xlabel('ROIs sorted by $D_{\\mathrm{recon}}$', fontsize=7)
ax.set_ylabel(r'$D_{\mathrm{recon}}$', fontsize=8)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.text(-0.22, 1.16, 'B', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')
ax.set_title('Per-ROI score\n(green = Asp, gray = Muc)', fontsize=7, pad=3)

# C: scatter D_recon vs D_continuous
ax = fig.add_subplot(gs[0, 2])
sel_a = np.array([r['genus'] == 'Aspergillus' for r in records])
ax.scatter(np.array(all_vals['D_continuous'])[sel_a],
            np.array(all_vals['D_recon'])[sel_a],
            s=30, c=C_ASP, edgecolors='white', linewidths=0.4,
            label=f'Asp (n={sel_a.sum()})')
ax.scatter(np.array(all_vals['D_continuous'])[~sel_a],
            np.array(all_vals['D_recon'])[~sel_a],
            s=30, c=C_MUC, edgecolors='white', linewidths=0.4,
            label=f'Muc (n={(~sel_a).sum()})')
rho, _ = stats.spearmanr(all_vals['D_continuous'], all_vals['D_recon'])
ax.set_xlabel(r'$D_{\mathrm{continuous}}$', fontsize=7)
ax.set_ylabel(r'$D_{\mathrm{recon}}$', fontsize=7)
ax.legend(fontsize=6.5, frameon=False, loc='best')
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.text(-0.22, 1.16, 'C', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')
ax.set_title(f'Photometric vs continuous\nSpearman ρ = {rho:+.2f}',
              fontsize=7, pad=3)

# D: ratio vs δ benchmark + prior continuous
ax = fig.add_subplot(gs[0, 3])
ratios_cmp = [1.89, ratio]
labels_cmp = ['continuous\n(prior)', 'photometric\n(new)']
ax.bar([0, 1], ratios_cmp, color=['#9E9E9E', '#1976D2'], alpha=0.85,
        edgecolor='black', lw=0.5, width=0.55)
for x, rr in zip([0, 1], ratios_cmp):
    ax.text(x, rr + 0.04, f'{rr:.2f}×', ha='center', fontsize=9,
             fontweight='bold')
ax.axhline(DELTA_RATIO, color='black', lw=0.7, ls='--', alpha=0.7)
ax.text(1.55, DELTA_RATIO, f' δ ({DELTA_RATIO}×)',
         va='center', ha='left', fontsize=7)
ax.set_xticks([0, 1])
ax.set_xticklabels(labels_cmp, fontsize=7)
ax.set_ylabel('Asp / Muc ratio', fontsize=7)
ax.set_ylim(0, max(max(ratios_cmp), DELTA_RATIO) * 1.18)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.text(-0.22, 1.16, 'D', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')

# Row 2: spearman with all methods
ax = fig.add_subplot(gs[1, :])
xs = np.arange(len(methods))
rhos = []
for k in methods:
    if k == 'D_recon':
        rhos.append(1.0); continue
    rho_, _ = stats.spearmanr(all_vals['D_recon'], all_vals[k])
    rhos.append(rho_)
colors_rho = ['#1976D2'] + ['#9E9E9E'] * (len(methods) - 1)
ax.bar(xs, rhos, color=colors_rho, alpha=0.85, edgecolor='black', lw=0.5)
for x, r_ in zip(xs, rhos):
    ax.text(x, r_ + 0.02, f'{r_:.2f}', ha='center', fontsize=7,
             fontweight='bold')
ax.axhline(0.85, color='gray', lw=0.5, ls=':')
ax.set_xticks(xs)
ax.set_xticklabels(methods, fontsize=6.5, rotation=15)
ax.set_ylabel('Spearman ρ with $D_{\\mathrm{recon}}$', fontsize=7.5)
ax.set_ylim(0, 1.1)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.text(-0.04, 1.10, 'E', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')
ax.set_title(r'Per-ROI rank agreement: $D_{\mathrm{recon}}$ vs all other methods',
              fontsize=7.5, pad=3)


# Rows 3-4: canonical heatmaps (raw + R)
def display_raw(img_np):
    lo, hi = np.percentile(img_np, [0.5, 99.5])
    return np.clip((img_np - lo) / max(hi - lo, 1), 0, 1)


vmax_R = max(canon['Aspergillus']['R'].max(), canon['Mucor']['R'].max())

# Aspergillus row
ax_raw = fig.add_subplot(gs[2, 0:2])
img = canon['Aspergillus']['img']; nd = display_raw(img)
ax_raw.imshow(nd, cmap='gray', interpolation='nearest'); ax_raw.axis('off')
ax_raw.text(0.98, 0.03, 'Aspergillus', transform=ax_raw.transAxes, fontsize=8,
             ha='right', va='bottom', style='italic', color='white',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.20', facecolor=C_ASP,
                       alpha=0.85, edgecolor='none'))
ax_raw.set_title('Raw ROI', fontsize=8, pad=3)
ax_raw.text(-0.02, 1.10, 'F', transform=ax_raw.transAxes, fontsize=12,
             fontweight='bold', va='top')

ax_R = fig.add_subplot(gs[2, 2:4])
R_a = canon['Aspergillus']['R']
im_R = ax_R.imshow(R_a, cmap='viridis', interpolation='nearest',
                    vmin=0, vmax=vmax_R)
ax_R.axis('off')
ax_R.set_title(f'Reconstructed density ($D_{{\\mathrm{{recon}}}} = '
                f'{R_a.mean():.4f}$)', fontsize=8, pad=3)
ax_R.text(-0.02, 1.10, 'G', transform=ax_R.transAxes, fontsize=12,
           fontweight='bold', va='top')

# Mucor row
ax_raw_m = fig.add_subplot(gs[3, 0:2])
img_m = canon['Mucor']['img']; nd_m = display_raw(img_m)
ax_raw_m.imshow(nd_m, cmap='gray', interpolation='nearest'); ax_raw_m.axis('off')
ax_raw_m.text(0.98, 0.03, 'Mucor', transform=ax_raw_m.transAxes, fontsize=8,
                ha='right', va='bottom', style='italic', color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.20', facecolor=C_MUC,
                          alpha=0.85, edgecolor='none'))
ax_raw_m.set_title('Raw ROI', fontsize=8, pad=3)
ax_raw_m.text(-0.02, 1.10, 'H', transform=ax_raw_m.transAxes, fontsize=12,
                fontweight='bold', va='top')

ax_R_m = fig.add_subplot(gs[3, 2:4])
R_m = canon['Mucor']['R']
im_R = ax_R_m.imshow(R_m, cmap='viridis', interpolation='nearest',
                      vmin=0, vmax=vmax_R)
ax_R_m.axis('off')
ax_R_m.set_title(f'Reconstructed density ($D_{{\\mathrm{{recon}}}} = '
                  f'{R_m.mean():.4f}$)', fontsize=8, pad=3)
ax_R_m.text(-0.02, 1.10, 'I', transform=ax_R_m.transAxes, fontsize=12,
              fontweight='bold', va='top')


# Colorbar
cbar_ax = fig.add_axes([0.97, 0.10, 0.012, 0.30])
cb = plt.colorbar(im_R, cax=cbar_ax)
cb.set_label(r'$R(x,y)$', fontsize=7); cb.ax.tick_params(labelsize=6)


for ext in ('.png', '.pdf', '.svg'):
    kw = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.04}
    if ext == '.png': kw['dpi'] = 600
    fig.savefig(OUT_FIG / f'figS_photometric_reconstruction{ext}', **kw)
plt.close(fig)
print(f'Saved: figS_photometric_reconstruction.{{pdf,png,svg}}')


# ── Full-resolution heatmaps (2D + 3D surface) for canonical ROIs ──
def downsample(arr, target_max=400):
    """Downsample large arrays so 3D plot_surface is tractable."""
    h, w = arr.shape
    scale = max(h, w) // target_max
    if scale <= 1: return arr
    return arr[::scale, ::scale]


for genus, color in [('Aspergillus', C_ASP), ('Mucor', C_MUC)]:
    img = canon[genus]['img']
    R = canon[genus]['R']
    nd = display_raw(img)
    h, w = img.shape

    # 2D heatmap full-res
    fig2, axes = plt.subplots(1, 2,
                                figsize=(20, max(8, 20 * h / (2 * w))),
                                gridspec_kw={'wspace': 0.02})
    fig2.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.005)
    axes[0].imshow(nd, cmap='gray', interpolation='nearest'); axes[0].axis('off')
    axes[0].set_title('Raw ROI', fontsize=16)
    axes[1].imshow(R, cmap='viridis', interpolation='nearest',
                    vmin=0, vmax=vmax_R)
    axes[1].axis('off')
    axes[1].set_title(
        f'Photometric reconstruction  ($D_{{\\mathrm{{recon}}}} = {R.mean():.4f}$)',
        fontsize=16)
    out_name = f'reconstruction_{"asp" if genus == "Aspergillus" else "muc"}_heatmap.png'
    fig2.savefig(PREV_DIR / out_name, dpi=180, bbox_inches='tight',
                  facecolor='white')
    plt.close(fig2)
    print(f'  Full-res 2D heatmap: {PREV_DIR / out_name}')

    # 3D surface plot (downsampled for tractability)
    Rd = downsample(R, target_max=300)
    yy, xx = np.mgrid[0:Rd.shape[0], 0:Rd.shape[1]]
    fig3 = plt.figure(figsize=(14, 10))
    ax3 = fig3.add_subplot(111, projection='3d')
    surf = ax3.plot_surface(xx, yy, Rd, cmap='viridis',
                              linewidth=0, antialiased=False, vmin=0,
                              vmax=vmax_R, rcount=Rd.shape[0],
                              ccount=Rd.shape[1])
    ax3.set_box_aspect([Rd.shape[1], Rd.shape[0], min(Rd.shape) * 0.3])
    ax3.view_init(elev=45, azim=-60)
    ax3.set_axis_off()
    ax3.set_title(
        f'{genus}: tissue density topography  '
        f'($D_{{\\mathrm{{recon}}}} = {R.mean():.4f}$)',
        fontsize=16, color=color, fontweight='bold')
    fig3.colorbar(surf, ax=ax3, shrink=0.6, pad=0.02).set_label(
        r'$R(x,y)$', fontsize=12)
    out_name_3d = f'reconstruction_{"asp" if genus == "Aspergillus" else "muc"}_3d.png'
    fig3.savefig(PREV_DIR / out_name_3d, dpi=180, bbox_inches='tight',
                  facecolor='white')
    plt.close(fig3)
    print(f'  Full-res 3D surface: {PREV_DIR / out_name_3d}')


print('\nDone.')

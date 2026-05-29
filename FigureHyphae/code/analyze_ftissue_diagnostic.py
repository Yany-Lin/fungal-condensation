#!/usr/bin/env python3
"""f_tissue diagnostic — does this metric actually discriminate Asp vs Muc,
or is the 1.16× mean ratio masking something?

Probes four angles:
  1. Scale-dependence: how does Cohen's d evolve as the measurement window
     shrinks from whole-image to single-hypha?
  2. Distribution shape: at the discrimination-peak window, are the local
     f_tissue distributions different in shape (variance, skew, modality)?
  3. Spatial heterogeneity: CV of local f_tissue across each ROI.
  4. Visual: local-f_tissue heatmaps at the peak window for canonical ROIs.

Outputs: figS_ftissue_diagnostic.{pdf,png,svg}
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

ASP_ROI_KEY = '20251214_222552.JPG'
MUC_ROI_KEY = '20251210_155926.JPG'

# Window sizes for scale sweep (µm side length of square)
WIN_UM = np.array([5, 10, 20, 40, 80, 160, 320, 640])

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


def local_density(mask, win_um, cal_um_per_px=CAL_3D):
    w = max(2, int(round(win_um / cal_um_per_px)))
    kernel = np.ones((w, w)) / (w * w)
    return ndi.convolve(mask.astype(np.float64), kernel, mode='nearest')


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0


def savefig(fig, name):
    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.04}
        if ext == '.png':
            kw['dpi'] = 600
        fig.savefig(OUT / f'{name}{ext}', **kw)
    plt.close(fig)
    print(f'  Saved: {name}')


# ── Load all ROIs and pre-segment ──
with open(SESSION) as f:
    saved = {k: v for k, v in json.load(f).items()
             if not k.startswith('_') and v.get('status') != 'deleted'}

print('Segmenting all ROIs...')
asp_masks, muc_masks = [], []
asp_keys, muc_keys = [], []
for key, val in saved.items():
    g = val.get('genus')
    rp = ROI_DIR / g / f'{Path(key).stem}_roi.jpg'
    if not rp.exists():
        continue
    img = load_gray(rp)
    m = seg_3d(img)
    if m.sum() < 100:
        continue
    if g == 'Aspergillus':
        asp_masks.append(m); asp_keys.append(key)
    else:
        muc_masks.append(m); muc_keys.append(key)
print(f'  Aspergillus: {len(asp_masks)} ROIs')
print(f'  Mucor:       {len(muc_masks)} ROIs')


# ── Scale sweep: per-ROI mean + CV of local f at each window ──
print('\nScale sweep (this takes ~30 s)...')
def per_roi_stats(mask, win_um):
    """Mean and CV of local f_tissue for one ROI at window size win_um."""
    if win_um <= 0:
        v = np.array([mask.mean()])
    else:
        ld = local_density(mask, win_um)
        # Crop edge effects: trim a half-window border
        b = max(1, int(round(win_um / CAL_3D / 2)))
        ld = ld[b:-b, b:-b] if ld.shape[0] > 2 * b and ld.shape[1] > 2 * b else ld
        v = ld.ravel()
    return v.mean(), v.std(ddof=1) / max(v.mean(), 1e-9), v

asp_mean_by_win = []   # rows: ROIs, cols: window sizes
muc_mean_by_win = []
asp_cv_by_win = []
muc_cv_by_win = []

for w_um in WIN_UM:
    a_means, a_cvs, m_means, m_cvs = [], [], [], []
    for mk in asp_masks:
        mu, cv, _ = per_roi_stats(mk, w_um)
        a_means.append(mu); a_cvs.append(cv)
    for mk in muc_masks:
        mu, cv, _ = per_roi_stats(mk, w_um)
        m_means.append(mu); m_cvs.append(cv)
    asp_mean_by_win.append(a_means); muc_mean_by_win.append(m_means)
    asp_cv_by_win.append(a_cvs); muc_cv_by_win.append(m_cvs)

asp_mean_by_win = np.array(asp_mean_by_win)  # (n_win, n_asp)
muc_mean_by_win = np.array(muc_mean_by_win)
asp_cv_by_win = np.array(asp_cv_by_win)
muc_cv_by_win = np.array(muc_cv_by_win)

# Per-window Cohen's d for the per-ROI MEAN (this is what the box plot shows)
d_mean = np.array([cohens_d(asp_mean_by_win[i], muc_mean_by_win[i])
                    for i in range(len(WIN_UM))])
# Per-window Cohen's d for the per-ROI CV (heterogeneity discrimination)
d_cv = np.array([cohens_d(asp_cv_by_win[i], muc_cv_by_win[i])
                  for i in range(len(WIN_UM))])
# Welch p for the mean at each scale
p_mean = np.array([stats.ttest_ind(asp_mean_by_win[i], muc_mean_by_win[i],
                                     equal_var=False).pvalue
                    for i in range(len(WIN_UM))])
p_cv = np.array([stats.ttest_ind(asp_cv_by_win[i], muc_cv_by_win[i],
                                   equal_var=False).pvalue
                  for i in range(len(WIN_UM))])

print('\nScale sweep results:')
print(f'  {"win µm":>7} {"Asp mean":>10} {"Muc mean":>10} {"d(mean)":>9} '
      f'{"p(mean)":>10} {"d(CV)":>8} {"p(CV)":>10}')
for i, w in enumerate(WIN_UM):
    print(f'  {w:>7.0f} {asp_mean_by_win[i].mean():>10.3f} '
          f'{muc_mean_by_win[i].mean():>10.3f} {d_mean[i]:>9.2f} '
          f'{p_mean[i]:>10.2e} {d_cv[i]:>8.2f} {p_cv[i]:>10.2e}')

# Discrimination-peak window for CV (the heterogeneity signal)
peak_idx_cv = int(np.argmax(np.abs(d_cv)))
peak_idx_mean = int(np.argmax(np.abs(d_mean)))
W_PEAK = WIN_UM[peak_idx_cv]
print(f'\nPeak |d| for mean f_tissue: {WIN_UM[peak_idx_mean]:.0f} µm '
      f'(d={d_mean[peak_idx_mean]:.2f})')
print(f'Peak |d| for CV(local f_tissue): {W_PEAK:.0f} µm (d={d_cv[peak_idx_cv]:.2f})')


# ── Pool local f_tissue values at the peak-CV window for distribution ──
def pooled_local_values(masks, win_um):
    pool = []
    for mk in masks:
        _, _, v = per_roi_stats(mk, win_um)
        # Subsample to keep memory reasonable
        if len(v) > 50_000:
            v = np.random.default_rng(0).choice(v, 50_000, replace=False)
        pool.append(v)
    return np.concatenate(pool)

asp_pool = pooled_local_values(asp_masks, W_PEAK)
muc_pool = pooled_local_values(muc_masks, W_PEAK)


# ── Figure ──
fig = plt.figure(figsize=(180 * MM, 130 * MM))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.36,
                       left=0.07, right=0.97, top=0.95, bottom=0.08,
                       height_ratios=[1.0, 1.0])

# ── A: Cohen's d vs window size (mean and CV) ──
ax_a = fig.add_subplot(gs[0, 0])
ax_a.axhline(0, color='gray', lw=0.5)
ax_a.axhline(0.8, color='gray', lw=0.4, ls=':')
ax_a.axhline(-0.8, color='gray', lw=0.4, ls=':')
ax_a.plot(WIN_UM, d_mean, 'o-', color='#1976D2', lw=1.4, ms=5,
           label=r'mean $f_{\mathrm{tissue}}$')
ax_a.plot(WIN_UM, d_cv, 's-', color='#D81B60', lw=1.4, ms=5,
           label=r'CV of local $f_{\mathrm{tissue}}$')
ax_a.set_xscale('log')
ax_a.set_xlabel('Measurement window (µm)', fontsize=7.5)
ax_a.set_ylabel("Effect size, Cohen's $d$\n(Asp − Muc)", fontsize=7.5)
ax_a.legend(fontsize=6.5, loc='lower left', frameon=False)
ax_a.tick_params(labelsize=6.5)
for sp in ('top', 'right'):
    ax_a.spines[sp].set_visible(False)
ax_a.text(-0.18, 1.10, 'A', transform=ax_a.transAxes,
           fontsize=12, fontweight='bold', va='top')

# ── B: Welch p-value vs window size (log scale) ──
ax_b = fig.add_subplot(gs[0, 1])
ax_b.axhline(0.05, color='gray', lw=0.4, ls=':')
ax_b.axhline(0.001, color='gray', lw=0.4, ls=':')
ax_b.semilogy(WIN_UM, p_mean, 'o-', color='#1976D2', lw=1.4, ms=5,
               label=r'mean $f_{\mathrm{tissue}}$')
ax_b.semilogy(WIN_UM, p_cv, 's-', color='#D81B60', lw=1.4, ms=5,
               label=r'CV')
ax_b.set_xscale('log')
ax_b.invert_yaxis()
ax_b.set_xlabel('Measurement window (µm)', fontsize=7.5)
ax_b.set_ylabel('Welch $p$ (lower = stronger)', fontsize=7.5)
ax_b.legend(fontsize=6.5, loc='lower left', frameon=False)
ax_b.tick_params(labelsize=6.5)
for sp in ('top', 'right'):
    ax_b.spines[sp].set_visible(False)
ax_b.text(-0.18, 1.10, 'B', transform=ax_b.transAxes,
           fontsize=12, fontweight='bold', va='top')

# ── C: CV(local f) box plot at peak-discrimination window ──
ax_c = fig.add_subplot(gs[0, 2])
sa = asp_cv_by_win[peak_idx_cv]
sm = muc_cv_by_win[peak_idx_cv]
bp = ax_c.boxplot([sa, sm], positions=[1, 2], widths=0.4, patch_artist=True,
                   showfliers=False, medianprops=dict(color='white', lw=1.4),
                   whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
bp['boxes'][0].set_facecolor(C_ASP); bp['boxes'][0].set_alpha(0.55)
bp['boxes'][1].set_facecolor(C_MUC); bp['boxes'][1].set_alpha(0.55)
rng = np.random.default_rng(42)
ax_c.scatter(1 + rng.uniform(-0.09, 0.09, len(sa)), sa, s=14, c=C_ASP,
              alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
ax_c.scatter(2 + rng.uniform(-0.09, 0.09, len(sm)), sm, s=14, c=C_MUC,
              alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
ax_c.set_xticks([1, 2])
ax_c.set_xticklabels([r'$\it{Aspergillus}$', r'$\it{Mucor}$'], fontsize=6.5)
ax_c.set_ylabel(f'CV of local $f_{{\\mathrm{{tissue}}}}$\n'
                 f'(at {W_PEAK:.0f} µm window)', fontsize=7)
for sp in ('top', 'right'):
    ax_c.spines[sp].set_visible(False)
p = p_cv[peak_idx_cv]
ym = max(sa.max(), sm.max()); r = ym - min(sa.min(), sm.min())
y = ym + r * 0.08
ax_c.plot([1, 1, 2, 2], [y, y + r * 0.03, y + r * 0.03, y], 'k-', lw=0.6)
txt = f'p = {p:.1e}' if p < 0.001 else f'p = {p:.3f}'
ax_c.text(1.5, y + r * 0.06, txt, ha='center', fontsize=6)
ax_c.text(-0.30, 1.10, 'C', transform=ax_c.transAxes,
           fontsize=12, fontweight='bold', va='top')

# ── D, E: local-f heatmaps for canonical ROIs at peak window ──
asp_canon = [m for m, k in zip(asp_masks, asp_keys) if k == ASP_ROI_KEY]
muc_canon = [m for m, k in zip(muc_masks, muc_keys) if k == MUC_ROI_KEY]
asp_m = asp_canon[0] if asp_canon else asp_masks[0]
muc_m = muc_canon[0] if muc_canon else muc_masks[0]
asp_ld = local_density(asp_m, W_PEAK)
muc_ld = local_density(muc_m, W_PEAK)
vmin, vmax = 0.0, max(asp_ld.max(), muc_ld.max())

ax_d = fig.add_subplot(gs[1, 0])
im_d = ax_d.imshow(asp_ld, cmap='viridis', vmin=vmin, vmax=vmax,
                    interpolation='nearest')
ax_d.axis('off')
ax_d.set_title(f'Aspergillus: local $f_{{\\mathrm{{tissue}}}}$ '
                f'({W_PEAK:.0f} µm window)', fontsize=7,
                color=C_ASP, fontweight='bold', pad=3)
ax_d.text(-0.04, 1.10, 'D', transform=ax_d.transAxes,
           fontsize=12, fontweight='bold', va='top')

ax_e = fig.add_subplot(gs[1, 1])
im_e = ax_e.imshow(muc_ld, cmap='viridis', vmin=vmin, vmax=vmax,
                    interpolation='nearest')
ax_e.axis('off')
ax_e.set_title(f'Mucor: local $f_{{\\mathrm{{tissue}}}}$ '
                f'({W_PEAK:.0f} µm window)', fontsize=7,
                color=C_MUC, fontweight='bold', pad=3)
ax_e.text(-0.04, 1.10, 'E', transform=ax_e.transAxes,
           fontsize=12, fontweight='bold', va='top')

# Shared colorbar
cbar = fig.colorbar(im_e, ax=[ax_d, ax_e], shrink=0.7, pad=0.02, aspect=20,
                     orientation='vertical')
cbar.set_label(r'local $f_{\mathrm{tissue}}$', fontsize=6.5)
cbar.ax.tick_params(labelsize=6)

# ── F: pooled distribution of local f_tissue at peak window ──
ax_f = fig.add_subplot(gs[1, 2])
bins = np.linspace(0, 1, 50)
ax_f.hist(asp_pool, bins=bins, density=True, color=C_ASP, alpha=0.55,
           label=f'Asp  μ={asp_pool.mean():.2f}, σ={asp_pool.std():.2f}',
           edgecolor='none')
ax_f.hist(muc_pool, bins=bins, density=True, color=C_MUC, alpha=0.55,
           label=f'Muc  μ={muc_pool.mean():.2f}, σ={muc_pool.std():.2f}',
           edgecolor='none')
ax_f.set_xlabel(f'local $f_{{\\mathrm{{tissue}}}}$  ({W_PEAK:.0f} µm window)',
                 fontsize=7.5)
ax_f.set_ylabel('density', fontsize=7.5)
ax_f.legend(fontsize=6, loc='upper right', frameon=False)
ax_f.tick_params(labelsize=6.5)
for sp in ('top', 'right'):
    ax_f.spines[sp].set_visible(False)
ax_f.text(-0.18, 1.10, 'F', transform=ax_f.transAxes,
           fontsize=12, fontweight='bold', va='top')

savefig(fig, 'figS_ftissue_diagnostic')

# ── Print conclusion ──
print('\n══════════════════════════════════════════════════════════════')
print('CONCLUSION')
print('══════════════════════════════════════════════════════════════')
print(f'Global mean f_tissue is a weak discriminator on its own:')
print(f'  d(mean) at full image = {d_mean[-1]:.2f}, p = {p_mean[-1]:.2e}')
print(f'Discrimination is STRONGER for spatial heterogeneity:')
print(f'  d(CV) at full image   = {d_cv[-1]:.2f}, p = {p_cv[-1]:.2e}')
print(f'Peak |d| across scales:')
print(f'  mean: window={WIN_UM[peak_idx_mean]:.0f} µm, d={d_mean[peak_idx_mean]:.2f}')
print(f'  CV:   window={WIN_UM[peak_idx_cv]:.0f} µm, d={d_cv[peak_idx_cv]:.2f}')
print('\nDone.')

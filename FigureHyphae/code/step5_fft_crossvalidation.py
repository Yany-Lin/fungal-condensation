#!/usr/bin/env python3
"""Formal FFT spectral analysis on 3D colony ROIs.
Computes spectral slope α, peak wavelength, and high-frequency power ratio.
Cross-validates against GLCM, DT thickness, and absorbing capacity.
Generates publication-quality supplementary figure.
"""

import json, numpy as np
from scipy import ndimage as ndi
from scipy import stats
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROI_DIR = Path('/Volumes/T7/FINAL OSF/HYPHAE/Analysis/results/3d_overlays')
SESSION = ROI_DIR / 'roi_session.json'
OUT = Path('/Volumes/T7/FINAL OSF/FigureHyphae/figures')
CAL = 0.94  # µm/px
MM = 1/25.4
C_ASP = '#4CAF50'; C_MUC = '#757575'
TILE = 256; STRIDE = 128
FREQ_LO, FREQ_HI = 0.02, 0.40  # cycles/px for slope fit

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 8, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'mathtext.default': 'regular',
})

def load_gray(p):
    with Image.open(p) as im: arr = np.asarray(im).astype(np.float64)
    if arr.ndim == 3: arr = arr[..., :3].mean(axis=2)
    return arr

def seg_3d(img):
    s = ndi.gaussian_filter(img, sigma=1.0)
    l = ndi.gaussian_filter(s, sigma=32.0)
    dr = l - s; t = dr.mean() + 0.5 * dr.std()
    m = dr > t
    return ndi.binary_opening(ndi.binary_closing(m, iterations=1), iterations=1)

def glcm_contrast(img, d=1):
    lo, hi = np.percentile(img, [0.5, 99.5])
    if hi <= lo: hi = lo + 1
    i8 = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    dh = (i8[:, d:].astype(float) - i8[:, :-d].astype(float)) ** 2
    dv = (i8[d:, :].astype(float) - i8[:-d, :].astype(float)) ** 2
    return (dh.mean() + dv.mean()) / 2.0

def tile_radial(tile):
    """Radial power spectrum of a square tile."""
    n = tile.shape[0]
    c = tile - tile.mean()
    win = np.outer(np.hanning(n), np.hanning(n))
    fft = np.fft.fftshift(np.fft.fft2(c * win))
    p2d = np.abs(fft) ** 2
    cy, cx = n // 2, n // 2
    y, x = np.arange(n) - cy, np.arange(n) - cx
    yy, xx = np.meshgrid(y, x, indexing='ij')
    r = np.sqrt(xx**2 + yy**2).astype(int)
    rm = n // 2
    rf = np.clip(r.ravel(), 0, rm - 1)
    rs = np.bincount(rf, weights=p2d.ravel(), minlength=rm)
    rc = np.bincount(rf, minlength=rm).astype(float)
    rc[rc == 0] = 1
    freq = np.arange(rm) / n  # cycles per pixel
    return freq[1:], (rs / rc)[1:]

def fft_metrics(img, tile_size=TILE, stride=STRIDE):
    """Compute FFT spectral slope from tiled analysis."""
    h, w = img.shape
    all_powers = []
    for y0 in range(0, h - tile_size + 1, stride):
        for x0 in range(0, w - tile_size + 1, stride):
            t = img[y0:y0+tile_size, x0:x0+tile_size]
            if t.std() < 3.0:  # skip blank tiles
                continue
            f, p = tile_radial(t)
            all_powers.append(p)
    if not all_powers:
        return np.nan, np.nan, None, None
    mp = np.mean(all_powers, axis=0)
    f = np.arange(1, TILE // 2) / TILE

    # Fit slope in log-log space
    valid = (f >= FREQ_LO) & (f <= FREQ_HI) & (mp > 0)
    if valid.sum() < 10:
        return np.nan, np.nan, f, mp
    sl, ic, r, p, se = stats.linregress(np.log10(f[valid]), np.log10(mp[valid]))

    # High-frequency power ratio: fraction of total power above 0.15 cyc/px
    hf_mask = f > 0.15
    lf_mask = (f > 0.02) & (f <= 0.15)
    hf_ratio = mp[hf_mask].sum() / mp[lf_mask].sum() if mp[lf_mask].sum() > 0 else 0

    return sl, hf_ratio, f, mp


# ═══════════════════════════════════════════════════════════════
# COMPUTE ALL METRICS PER ROI
# ═══════════════════════════════════════════════════════════════

with open(SESSION) as f:
    session = json.load(f)
saved = {k: v for k, v in session.items()
         if not k.startswith('_') and v.get('status') != 'deleted'}

print('Computing FFT + GLCM + DT for all ROIs...')
rows = []
spectra = {'Aspergillus': [], 'Mucor': []}

for key, val in saved.items():
    genus = val.get('genus')
    roi_path = ROI_DIR / genus / f'{Path(key).stem}_roi.jpg'
    if not roi_path.exists():
        continue
    img = load_gray(roi_path)
    mask = seg_3d(img)
    dt_arr = ndi.distance_transform_edt(mask) * CAL
    dt_med = np.median(dt_arr[mask]) if mask.sum() > 100 else 0
    ft = mask.mean()
    gc = glcm_contrast(img)
    alpha, hf_ratio, freq, power = fft_metrics(img)
    cap = ft * dt_med

    rows.append({
        'file': key, 'genus': genus,
        'alpha': alpha, 'hf_ratio': hf_ratio,
        'glcm': gc, 'dt_um': dt_med, 'f_tissue': ft,
        'capacity': cap,
    })
    if freq is not None:
        spectra[genus].append((freq, power))
    print(f'  {key} ({genus}): α={alpha:.3f}, HF={hf_ratio:.4f}, '
          f'GLCM={gc:.1f}, DT={dt_med:.1f}µm, cap={cap:.2f}')


# ═══════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('FFT SPECTRAL ANALYSIS — FORMAL RESULTS')
print('=' * 70)

asp = [r for r in rows if r['genus'] == 'Aspergillus']
muc = [r for r in rows if r['genus'] == 'Mucor']

for metric, name, units in [
    ('alpha', 'Spectral slope α', ''),
    ('hf_ratio', 'High-freq power ratio', ''),
    ('glcm', 'GLCM contrast', ''),
    ('dt_um', 'DT thickness', 'µm'),
    ('capacity', 'Absorbing capacity', ''),
]:
    va = np.array([r[metric] for r in asp])
    vm = np.array([r[metric] for r in muc])
    t, p = stats.ttest_ind(va, vm, equal_var=False)
    u, pu = stats.mannwhitneyu(va, vm, alternative='two-sided')
    n1, n2 = len(va), len(vm)
    pooled = np.sqrt(((n1-1)*va.std(ddof=1)**2 + (n2-1)*vm.std(ddof=1)**2) / (n1+n2-2))
    d = (va.mean() - vm.mean()) / pooled if pooled > 0 else 0
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f'\n  {name} {units}')
    print(f'    Asp: {va.mean():.4f} ± {va.std(ddof=1):.4f} (n={n1})')
    print(f'    Muc: {vm.mean():.4f} ± {vm.std(ddof=1):.4f} (n={n2})')
    print(f'    Welch t={t:.3f}, p={p:.4g} {sig}')
    print(f'    Mann-Whitney U={u:.0f}, p={pu:.4g}')
    print(f'    Cohen d={d:.3f}')

# Cross-correlations
print('\n── Cross-correlations (all ROIs) ──')
all_alpha = np.array([r['alpha'] for r in rows])
all_glcm = np.array([r['glcm'] for r in rows])
all_dt = np.array([r['dt_um'] for r in rows])
all_cap = np.array([r['capacity'] for r in rows])
all_hf = np.array([r['hf_ratio'] for r in rows])

for x, y, xn, yn in [
    (all_alpha, all_glcm, 'α', 'GLCM'),
    (all_alpha, all_dt, 'α', 'DT'),
    (all_alpha, all_cap, 'α', 'Capacity'),
    (all_hf, all_glcm, 'HF ratio', 'GLCM'),
    (all_hf, all_dt, 'HF ratio', 'DT'),
    (all_glcm, all_dt, 'GLCM', 'DT'),
]:
    v = ~np.isnan(x) & ~np.isnan(y)
    if v.sum() < 5: continue
    rs, ps = stats.spearmanr(x[v], y[v])
    rp, pp = stats.pearsonr(x[v], y[v])
    print(f'  {xn:12s} vs {yn:10s}  Spearman r={rs:.3f} (p={ps:.4g})  Pearson r={rp:.3f} (p={pp:.4g})')


# ═══════════════════════════════════════════════════════════════
# FIGURE — 4 panels
# ═══════════════════════════════════════════════════════════════

print('\nGenerating figure...')

fig = plt.figure(figsize=(183 * MM, 130 * MM))
gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.35,
                      left=0.10, right=0.96, top=0.94, bottom=0.08)

# ── Panel A: Mean radial power spectra ──
ax = fig.add_subplot(gs[0, 0])
for genus, color, label in [('Aspergillus', C_ASP, 'Asp'), ('Mucor', C_MUC, 'Muc')]:
    specs = spectra[genus]
    all_p = np.array([p for _, p in specs])
    freq = specs[0][0]
    mean_p = np.mean(all_p, axis=0)
    sem_p = np.std(all_p, axis=0, ddof=1) / np.sqrt(len(specs))

    # Convert to spatial wavelength (µm)
    wl = 1.0 / (freq / CAL + 1e-12)  # µm per cycle

    ax.loglog(freq, mean_p, color=color, lw=1.5, label=f'{label} (n={len(specs)})')
    ax.fill_between(freq, mean_p - sem_p, mean_p + sem_p, color=color, alpha=0.15)

# Show fit region
ax.axvspan(FREQ_LO, FREQ_HI, alpha=0.08, color='blue')
ax.set_xlabel('Spatial frequency (cycles/px)', fontsize=7)
ax.set_ylabel('Radial power', fontsize=7)
ax.set_title('Mean radial power spectrum', fontsize=8, fontweight='bold')
ax.legend(fontsize=6, loc='lower left')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
ax.text(-0.14, 1.06, 'A', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')

# ── Panel B: Spectral slope boxplot ──
ax = fig.add_subplot(gs[0, 1])
va = np.array([r['alpha'] for r in asp])
vm = np.array([r['alpha'] for r in muc])

bp = ax.boxplot([va, vm], positions=[1, 2], widths=0.4, patch_artist=True, showfliers=False,
                medianprops=dict(color='white', lw=1.4),
                whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
bp['boxes'][0].set_facecolor(C_ASP); bp['boxes'][0].set_alpha(0.55)
bp['boxes'][1].set_facecolor(C_MUC); bp['boxes'][1].set_alpha(0.55)
rng = np.random.default_rng(42)
ax.scatter(1 + rng.uniform(-0.09, 0.09, len(va)), va,
           s=16, c=C_ASP, alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
ax.scatter(2 + rng.uniform(-0.09, 0.09, len(vm)), vm,
           s=16, c=C_MUC, alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
ax.set_xticks([1, 2])
ax.set_xticklabels([r'$\it{Aspergillus}$', r'$\it{Mucor}$'], fontsize=7)
ax.set_ylabel(r'Spectral slope $\alpha$', fontsize=7.5)
t, p = stats.ttest_ind(va, vm, equal_var=False)
ym = max(va.max(), vm.max()); r = ym - min(va.min(), vm.min())
y = ym + r * 0.08
ax.plot([1, 1, 2, 2], [y, y+r*0.03, y+r*0.03, y], 'k-', lw=0.6)
ax.text(1.5, y + r*0.06, f'p = {p:.3f}', ha='center', fontsize=6)
ax.set_title(r'Spectral slope $\alpha$ (steeper = smoother)', fontsize=7.5, fontweight='bold')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
ax.text(-0.14, 1.06, 'B', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')

# ── Panel C: α vs GLCM scatter (cross-validation) ──
ax = fig.add_subplot(gs[1, 0])
for r in rows:
    c = C_ASP if r['genus'] == 'Aspergillus' else C_MUC
    m = 'D' if r['genus'] == 'Aspergillus' else 'D'
    ax.scatter(r['alpha'], r['glcm'], c=c, s=20, alpha=0.8,
               edgecolors='white', linewidths=0.3, zorder=3)

rs_ag, ps_ag = stats.spearmanr(all_alpha, all_glcm)
# Fit line
sl, ic, _, _, _ = stats.linregress(all_alpha, all_glcm)
xfit = np.linspace(all_alpha.min(), all_alpha.max(), 50)
ax.plot(xfit, ic + sl * xfit, 'k--', lw=0.8, alpha=0.5)

ax.set_xlabel(r'Spectral slope $\alpha$', fontsize=7)
ax.set_ylabel('GLCM contrast', fontsize=7)
ax.set_title(r'$\alpha$ vs GLCM (cross-validation)', fontsize=7.5, fontweight='bold')
ax.text(0.95, 0.08, f'r = {rs_ag:.2f}\np = {ps_ag:.1e}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=6,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
ax.text(-0.14, 1.06, 'C', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')

# ── Panel D: α vs absorbing capacity scatter ──
ax = fig.add_subplot(gs[1, 1])
for r in rows:
    c = C_ASP if r['genus'] == 'Aspergillus' else C_MUC
    ax.scatter(r['alpha'], r['capacity'], c=c, s=20, alpha=0.8,
               edgecolors='white', linewidths=0.3, zorder=3)

rs_ac, ps_ac = stats.spearmanr(all_alpha, all_cap)
sl2, ic2, _, _, _ = stats.linregress(all_alpha, all_cap)
ax.plot(xfit, ic2 + sl2 * xfit, 'k--', lw=0.8, alpha=0.5)

ax.set_xlabel(r'Spectral slope $\alpha$', fontsize=7)
ax.set_ylabel('Absorbing capacity', fontsize=7)
ax.set_title(r'$\alpha$ vs absorbing capacity', fontsize=7.5, fontweight='bold')
ax.text(0.95, 0.08, f'r = {rs_ac:.2f}\np = {ps_ac:.1e}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=6,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=C_ASP,
                          markersize=6, label='Asp'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor=C_MUC,
                          markersize=6, label='Muc')]
ax.legend(handles=legend_elements, fontsize=6, loc='upper right')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
ax.text(-0.14, 1.06, 'D', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')

for ext in ('.png', '.pdf', '.svg'):
    kw = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.03}
    if ext == '.png': kw['dpi'] = 600
    fig.savefig(OUT / f'figS18_fft_analysis{ext}', **kw)
plt.close(fig)
print(f'\nSaved: figS18_fft_analysis')

# Save CSV
import csv
with open(OUT.parent / 'output' / 'fft_per_roi.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['file', 'genus', 'alpha', 'hf_ratio',
                                       'glcm', 'dt_um', 'f_tissue', 'capacity'])
    w.writeheader()
    w.writerows(rows)
print(f'Saved: output/fft_per_roi.csv')

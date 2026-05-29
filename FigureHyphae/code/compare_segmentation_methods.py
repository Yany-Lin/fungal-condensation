#!/usr/bin/env python3
"""Compare f_tissue segmentation methods with unified polarity.

Methods (all run on GPU, all output white = tissue):
  A. adaptive_current — current paper (Gaussian dark response,
     threshold = mean(dr) + 0.5*std)
  B. otsu_global      — global Otsu on raw intensity, illumination not corrected
  C. frangi_otsu      — multi-scale Hessian tubeness (Sato form,
     sigma = 1,2,4,8 px), then Otsu on the tubeness response. Designed for
     thin filamentous structures, captures fine hyphae the threshold methods
     miss.
  D. sauvola          — Sauvola local adaptive threshold (window = 64 px,
     k = 0.2). Parameter-light, handles illumination gradients.

Polarity is auto-enforced after each method: if raw[mask].mean() > raw[~mask].mean()
(tissue brighter than background), the mask is flipped. Tissue is always white.

Outputs:
  figS_ftissue_segcompare.{pdf,png,svg}
  ftissue_seg_methods.csv
"""

import json
import time
import csv
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
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
OUT_FIG = BASE / 'FigureHyphae' / 'figures'
OUT_CSV = BASE / 'FigureHyphae' / 'output'
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_CSV.mkdir(parents=True, exist_ok=True)

CAL_3D = 0.94
MM = 1 / 25.4
C_ASP, C_MUC = '#4CAF50', '#757575'

ASP_ROI_KEY = '20251214_222552.JPG'
MUC_ROI_KEY = '20251210_155926.JPG'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 8, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'mathtext.default': 'regular',
})


# ── GPU primitives ──
def _gauss_k1d(sigma, device):
    r = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=DTYPE)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    k = _gauss_k1d(sigma, img.device); r = (k.numel() - 1) // 2
    x = img.unsqueeze(0).unsqueeze(0)
    x = F.pad(x, (r, r, 0, 0), mode='reflect')
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, r, r), mode='reflect')
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    return x.squeeze(0).squeeze(0)


def gaussian_deriv(img, sigma, order_y, order_x):
    """Smoothed image with derivatives via finite-difference on the Gaussian-smoothed input."""
    s = gaussian_blur(img, sigma)
    out = s
    for _ in range(order_y):
        # central diff along axis 0
        out = torch.zeros_like(s)
        out[1:-1, :] = (s[2:, :] - s[:-2, :]) / 2.0
        s = out
    for _ in range(order_x):
        s2 = torch.zeros_like(out)
        s2[:, 1:-1] = (out[:, 2:] - out[:, :-2]) / 2.0
        out = s2
    return out * (sigma ** (order_y + order_x))  # gamma-normalize


def otsu(values_1d, nbins=512):
    vmin, vmax = float(values_1d.min()), float(values_1d.max())
    if vmax <= vmin:
        return vmin
    hist = torch.histc(values_1d, bins=nbins, min=vmin, max=vmax)
    centers = torch.linspace(vmin, vmax, nbins, device=values_1d.device)
    wb = torch.cumsum(hist, 0); wf = hist.sum() - wb
    sum_b = torch.cumsum(hist * centers, 0)
    sum_t = (hist * centers).sum()
    mb = sum_b / wb.clamp(min=1)
    mf = (sum_t - sum_b) / wf.clamp(min=1)
    var = wb * wf * (mb - mf) ** 2
    var = torch.where((wb > 0) & (wf > 0), var, torch.zeros_like(var))
    return float(centers[int(torch.argmax(var).item())])


def morph_open_close(mask, iters=1):
    m = mask.float().unsqueeze(0).unsqueeze(0)
    for _ in range(iters):
        m = F.max_pool2d(m, 3, stride=1, padding=1)
    for _ in range(iters):
        m = -F.max_pool2d(-m, 3, stride=1, padding=1)
    for _ in range(iters):
        m = -F.max_pool2d(-m, 3, stride=1, padding=1)
    for _ in range(iters):
        m = F.max_pool2d(m, 3, stride=1, padding=1)
    return m.squeeze(0).squeeze(0) > 0.5


def multiscale_tubeness(img, sigmas=(1, 2, 4, 8)):
    """Sato-style multi-scale tubeness for DARK lines on bright background.
    Returns max over scales of (larger Hessian eigenvalue) where positive.

    For dark line: Hessian has one large positive eigenvalue (across the line)
    and one near-zero eigenvalue (along the line). We return the positive
    larger eigenvalue.
    """
    response = torch.zeros_like(img)
    for s in sigmas:
        # Gaussian-smoothed Hessian via finite differences (gamma-normalized)
        Ixx = gaussian_deriv(img, s, 0, 2) - 0  # placeholder
        # Easier: smooth once, then compute discrete Hessian on the smooth
        sm = gaussian_blur(img, s)
        # Hessian components via central differences
        Ixx = torch.zeros_like(sm); Iyy = torch.zeros_like(sm); Ixy = torch.zeros_like(sm)
        Ixx[:, 1:-1] = sm[:, 2:] - 2 * sm[:, 1:-1] + sm[:, :-2]
        Iyy[1:-1, :] = sm[2:, :] - 2 * sm[1:-1, :] + sm[:-2, :]
        Ixy[1:-1, 1:-1] = (sm[2:, 2:] - sm[2:, :-2] - sm[:-2, 2:] + sm[:-2, :-2]) / 4.0
        Ixx, Iyy, Ixy = Ixx * (s * s), Iyy * (s * s), Ixy * (s * s)
        # Eigenvalues: λ = tr/2 ± sqrt((tr/2)² - det)
        tr = Ixx + Iyy
        det = Ixx * Iyy - Ixy * Ixy
        disc = torch.sqrt(torch.clamp((tr / 2) ** 2 - det, min=0))
        lam_large = tr / 2 + disc   # larger eigenvalue
        # Dark line on bright bg → larger eigenvalue is positive
        r = torch.clamp(lam_large, min=0)
        response = torch.maximum(response, r)
    return response


def sauvola_threshold(img, window=64, k=0.2, R=0.5):
    """Sauvola local adaptive threshold via integral images.
    Returns mask = (img < T(x,y)). For dark-on-bright (hyphae), tissue is darker
    than the local Sauvola threshold."""
    # Integral image trick: local mean via avg_pool, local std via E[X²] - E[X]²
    x = img.unsqueeze(0).unsqueeze(0)
    mean = F.avg_pool2d(x, window, stride=1, padding=window // 2,
                         count_include_pad=False)
    mean_sq = F.avg_pool2d(x ** 2, window, stride=1, padding=window // 2,
                            count_include_pad=False)
    var = (mean_sq - mean ** 2).clamp(min=0)
    std = torch.sqrt(var)
    # Crop back to original size if padding produced larger output
    mean = mean[..., :img.shape[0], :img.shape[1]]
    std = std[..., :img.shape[0], :img.shape[1]]
    T = mean * (1 + k * (std / R - 1))
    mask = img.unsqueeze(0).unsqueeze(0) < T
    return mask.squeeze(0).squeeze(0)


# ── Segmentation methods ──
def seg_adaptive_current(img):
    s = gaussian_blur(img, 1.0)
    l = gaussian_blur(s, 32.0)
    dr = l - s
    t = dr.mean() + 0.5 * dr.std()
    return morph_open_close(dr > t)


def seg_otsu_global(img):
    s = gaussian_blur(img, 1.0)
    t = otsu(s.ravel())
    return morph_open_close(s < t)  # dark = tissue


def seg_frangi_otsu(img):
    tub = multiscale_tubeness(img, sigmas=(1, 2, 4, 8))
    # Otsu on the positive tubeness values only (background is essentially 0)
    nz = tub[tub > 0]
    if nz.numel() < 1000:
        return torch.zeros_like(img, dtype=torch.bool)
    t = otsu(nz)
    return morph_open_close(tub > t)


def seg_sauvola(img):
    # Normalize to [0,1] first for stable Sauvola
    lo, hi = img.min(), img.max()
    n = (img - lo) / max(float(hi - lo), 1e-6)
    return morph_open_close(sauvola_threshold(n, window=64, k=0.2, R=0.5))


METHODS = [
    ('A: adaptive (current)',     seg_adaptive_current),
    ('B: Otsu (global)',          seg_otsu_global),
    ('C: Frangi + Otsu',          seg_frangi_otsu),
    ('D: Sauvola local',          seg_sauvola),
]


def enforce_polarity(mask, raw_img):
    """If raw[mask] is brighter than raw[~mask], the mask is inverted — flip it.
    Tissue must always be the DARKER class."""
    if mask.sum() == 0 or (~mask).sum() == 0:
        return mask, False
    mean_in = raw_img[mask].mean()
    mean_out = raw_img[~mask].mean()
    if mean_in > mean_out:
        return ~mask, True
    return mask, False


def load_gray(p):
    with Image.open(p) as im:
        a = np.asarray(im).astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a


# ── Process all ROIs ──
with open(SESSION) as f:
    saved = {k: v for k, v in json.load(f).items()
             if not k.startswith('_') and v.get('status') != 'deleted'}

print(f'\nProcessing 24 ROIs × {len(METHODS)} methods...')
t0 = time.time()
rows = []
fvals = {name: {'Aspergillus': [], 'Mucor': []} for name, _ in METHODS}
flips = {name: 0 for name, _ in METHODS}
masks_canon = {}
img_canon = {}

for key, val in saved.items():
    g = val.get('genus')
    rp = ROI_DIR / g / f'{Path(key).stem}_roi.jpg'
    if not rp.exists():
        continue
    img_np = load_gray(rp)
    img_gpu = torch.from_numpy(img_np).to(DEVICE)
    is_canon = (key == ASP_ROI_KEY) or (key == MUC_ROI_KEY)
    if is_canon:
        img_canon[g] = img_np
    for name, seg_fn in METHODS:
        raw_mask = seg_fn(img_gpu)
        mask, was_flipped = enforce_polarity(raw_mask, img_gpu)
        if was_flipped:
            flips[name] += 1
        f = float(mask.float().mean().item())
        fvals[name][g].append(f)
        rows.append({'file': key, 'genus': g, 'method': name,
                      'f_tissue': round(f, 4),
                      'polarity_flipped': int(was_flipped)})
        if is_canon:
            masks_canon[(g, name)] = mask.cpu().numpy()

elapsed = time.time() - t0
print(f'  Done in {elapsed:.1f} s on {DEVICE.upper()}')
print(f'  Polarity flips (out of 24): {flips}')

# Save CSV
csv_path = OUT_CSV / 'ftissue_seg_methods.csv'
with open(csv_path, 'w', newline='') as fp:
    w = csv.DictWriter(fp, fieldnames=['file', 'genus', 'method', 'f_tissue',
                                         'polarity_flipped'])
    w.writeheader(); w.writerows(rows)
print(f'  CSV: {csv_path}')

# Summary
print('\nPer-method results (unified polarity):')
print(f'  {"method":30} {"Asp":>14} {"Muc":>14} {"ratio":>8} {"p":>10} {"d":>6}')
ratios, ps, ds = {}, {}, {}
for name, _ in METHODS:
    a = np.array(fvals[name]['Aspergillus'])
    m = np.array(fvals[name]['Mucor'])
    r = a.mean() / m.mean()
    p = stats.ttest_ind(a, m, equal_var=False).pvalue
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) +
                  (len(m) - 1) * m.var(ddof=1)) / (len(a) + len(m) - 2))
    d = (a.mean() - m.mean()) / sp if sp > 0 else 0
    ratios[name], ps[name], ds[name] = r, p, d
    print(f'  {name:30} {a.mean():.3f}±{a.std(ddof=1):.3f}  '
          f'{m.mean():.3f}±{m.std(ddof=1):.3f}  {r:>6.3f}  {p:>10.2e}  {d:>5.2f}')


# ══════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(190 * MM, 165 * MM))
gs = fig.add_gridspec(3, 4, hspace=0.40, wspace=0.32,
                       left=0.06, right=0.97, top=0.95, bottom=0.05,
                       height_ratios=[1.0, 1.0, 1.0])

# Top row: 4 box plots (one per method)
for i, (name, _) in enumerate(METHODS):
    ax = fig.add_subplot(gs[0, i])
    sa = np.array(fvals[name]['Aspergillus'])
    sm = np.array(fvals[name]['Mucor'])
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
    ax.set_xticklabels([r'$\it{Asp}$', r'$\it{Muc}$'], fontsize=6.5)
    ax.set_ylabel(r'$f_{\mathrm{tissue}}$', fontsize=7.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    p = ps[name]
    ym = max(sa.max(), sm.max())
    r = max(ym - min(sa.min(), sm.min()), 1e-6)
    y = ym + r * 0.08
    ax.plot([1, 1, 2, 2], [y, y + r * 0.03, y + r * 0.03, y], 'k-', lw=0.6)
    txt = f'p = {p:.1e}' if p < 0.001 else f'p = {p:.3f}'
    ax.text(1.5, y + r * 0.06, txt, ha='center', fontsize=6)
    ax.set_title(f'{name}\nratio={ratios[name]:.3f}, d={ds[name]:.2f}',
                  fontsize=7, pad=4)
    ax.text(-0.22, 1.14, 'ABCD'[i], transform=ax.transAxes, fontsize=12,
             fontweight='bold', va='top')

# Middle row: canonical Aspergillus — raw + 4 masks
img_a = img_canon.get('Aspergillus')
if img_a is not None:
    lo, hi = np.percentile(img_a, [0.5, 99.5])
    nd_a = np.clip((img_a - lo) / max(hi - lo, 1), 0, 1)

ax_a_raw = fig.add_subplot(gs[1, 0])
ax_a_raw.imshow(nd_a, cmap='gray', interpolation='nearest')
ax_a_raw.axis('off')
ax_a_raw.text(0.98, 0.03, 'Aspergillus', transform=ax_a_raw.transAxes,
               fontsize=7, ha='right', va='bottom', style='italic',
               color='white', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.20', facecolor=C_ASP,
                         alpha=0.85, edgecolor='none'))
ax_a_raw.set_title('Raw ROI', fontsize=7.5, pad=3)
ax_a_raw.text(-0.04, 1.14, 'E', transform=ax_a_raw.transAxes,
               fontsize=12, fontweight='bold', va='top')

for j, (name, _) in enumerate(METHODS[:3]):  # show A, B, C
    ax = fig.add_subplot(gs[1, j + 1])
    mask = masks_canon.get(('Aspergillus', name))
    if mask is None:
        ax.axis('off'); continue
    ax.imshow(mask, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    ax.axis('off')
    ax.text(0.98, 0.03, f'$f = {mask.mean():.3f}$', transform=ax.transAxes,
             fontsize=7, ha='right', va='bottom', color='white',
             bbox=dict(boxstyle='round,pad=0.20', facecolor='black',
                       alpha=0.6, edgecolor='none'))
    ax.set_title(name.split(': ', 1)[1], fontsize=7.5, pad=3)

# Bottom row: canonical Mucor — raw + 4 masks (skip first col so D fits)
img_m = img_canon.get('Mucor')
if img_m is not None:
    lo, hi = np.percentile(img_m, [0.5, 99.5])
    nd_m = np.clip((img_m - lo) / max(hi - lo, 1), 0, 1)

ax_m_raw = fig.add_subplot(gs[2, 0])
ax_m_raw.imshow(nd_m, cmap='gray', interpolation='nearest')
ax_m_raw.axis('off')
ax_m_raw.text(0.98, 0.03, 'Mucor', transform=ax_m_raw.transAxes,
               fontsize=7, ha='right', va='bottom', style='italic',
               color='white', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.20', facecolor=C_MUC,
                         alpha=0.85, edgecolor='none'))
ax_m_raw.set_title('Raw ROI', fontsize=7.5, pad=3)
ax_m_raw.text(-0.04, 1.14, 'F', transform=ax_m_raw.transAxes,
               fontsize=12, fontweight='bold', va='top')

for j, (name, _) in enumerate(METHODS[:3]):
    ax = fig.add_subplot(gs[2, j + 1])
    mask = masks_canon.get(('Mucor', name))
    if mask is None:
        ax.axis('off'); continue
    ax.imshow(mask, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    ax.axis('off')
    ax.text(0.98, 0.03, f'$f = {mask.mean():.3f}$', transform=ax.transAxes,
             fontsize=7, ha='right', va='bottom', color='white',
             bbox=dict(boxstyle='round,pad=0.20', facecolor='black',
                       alpha=0.6, edgecolor='none'))
    ax.set_title(name.split(': ', 1)[1], fontsize=7.5, pad=3)

# Save
for ext in ('.png', '.pdf', '.svg'):
    kw = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.04}
    if ext == '.png':
        kw['dpi'] = 600
    fig.savefig(OUT_FIG / f'figS_ftissue_segcompare{ext}', **kw)
plt.close(fig)
print(f'\nSaved: figS_ftissue_segcompare.{{pdf,png,svg}}')

# Also render a second figure with Method D (Sauvola) masks for the canonical ROIs
fig2, axes = plt.subplots(2, 2, figsize=(110 * MM, 110 * MM))
fig2.subplots_adjust(hspace=0.15, wspace=0.08, top=0.92, bottom=0.04,
                      left=0.04, right=0.96)
for r, (genus, color, nd, key) in enumerate([
    ('Aspergillus', C_ASP, nd_a, ASP_ROI_KEY),
    ('Mucor',       C_MUC, nd_m, MUC_ROI_KEY)]):
    axes[r, 0].imshow(nd, cmap='gray', interpolation='nearest'); axes[r, 0].axis('off')
    axes[r, 0].text(0.98, 0.03, genus, transform=axes[r, 0].transAxes,
                     fontsize=7, ha='right', va='bottom', style='italic',
                     color='white', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.20', facecolor=color,
                               alpha=0.85, edgecolor='none'))
    if r == 0:
        axes[r, 0].set_title('Raw ROI', fontsize=8, pad=3)
    m = masks_canon.get((genus, 'D: Sauvola local'))
    if m is not None:
        axes[r, 1].imshow(m, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        axes[r, 1].axis('off')
        axes[r, 1].text(0.98, 0.03, f'$f = {m.mean():.3f}$',
                         transform=axes[r, 1].transAxes, fontsize=7,
                         ha='right', va='bottom', color='white',
                         bbox=dict(boxstyle='round,pad=0.20', facecolor='black',
                                   alpha=0.6, edgecolor='none'))
    if r == 0:
        axes[r, 1].set_title('D: Sauvola local', fontsize=8, pad=3)
for ext in ('.png', '.pdf', '.svg'):
    kw = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.04}
    if ext == '.png':
        kw['dpi'] = 600
    fig2.savefig(OUT_FIG / f'figS_ftissue_sauvola{ext}', **kw)
plt.close(fig2)
print('Saved: figS_ftissue_sauvola.{pdf,png,svg}')
print('Done.')

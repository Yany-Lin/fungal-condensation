#!/usr/bin/env python3
"""Final f_tissue computation with corrected polarity (tissue = bright pixels).

Mirror of the published adaptive-dark-response method, with the sign flipped:
  bright_response = smooth - local_mean    (positive where brighter than local)
  T = mean(br) + k * std(br)
  mask = br > T

Same σ_smooth=1.0, σ_local=32 px, k=0.5 as published — only the polarity changes.
Defensible as a single methodological correction.

Outputs:
  D:/FINAL OSF/FigureHyphae/figures/figS_ftissue_bright.{pdf,png,svg}
  D:/FINAL OSF/FigureHyphae/output/ftissue_bright_final.csv
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

# Defensible universal settings (mirror of published method)
SIGMA_SMOOTH = 1.0
SIGMA_LOCAL  = 32.0
K_THRESHOLD  = 0.5
MORPH_ITERS  = 1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0) if DEVICE == "cuda" else "n/a"}')
print(f'Settings: σ_smooth={SIGMA_SMOOTH}, σ_local={SIGMA_LOCAL}, k={K_THRESHOLD}, '
       f'polarity=BRIGHT (tissue = brighter than local mean)')

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


def seg_bright_response(img_gpu,
                          sigma_smooth=SIGMA_SMOOTH,
                          sigma_local=SIGMA_LOCAL,
                          k=K_THRESHOLD,
                          morph_iters=MORPH_ITERS):
    """Tissue = brighter than local mean (mirror of published dark-response)."""
    s = gauss(img_gpu, sigma_smooth)
    l = gauss(s, sigma_local)
    br = s - l                       # positive where brighter than local mean
    t = br.mean() + k * br.std()
    mask = br > t
    return morph(mask, morph_iters) if morph_iters > 0 else mask


def seg_dark_response(img_gpu,
                        sigma_smooth=SIGMA_SMOOTH,
                        sigma_local=SIGMA_LOCAL,
                        k=K_THRESHOLD,
                        morph_iters=MORPH_ITERS):
    """Original published method (tissue = darker)."""
    s = gauss(img_gpu, sigma_smooth)
    l = gauss(s, sigma_local)
    dr = l - s
    t = dr.mean() + k * dr.std()
    mask = dr > t
    return morph(mask, morph_iters) if morph_iters > 0 else mask


def load_gray(p):
    with Image.open(p) as im:
        a = np.asarray(im).astype(np.float32)
    if a.ndim == 3: a = a[..., :3].mean(axis=2)
    return a


# ── Load all ROIs ──
with open(SESSION) as f:
    saved = {k: v for k, v in json.load(f).items()
             if not k.startswith('_') and v.get('status') != 'deleted'}

records = []
for key, val in saved.items():
    g = val.get('genus')
    rp = ROI_DIR / g / f'{Path(key).stem}_roi.jpg'
    if not rp.exists(): continue
    records.append({'key': key, 'genus': g, 'path': rp})
print(f'Loaded {len(records)} ROIs')


# ── Process ──
print('\nProcessing on GPU...')
t0 = time.time()
f_bright = {'Aspergillus': [], 'Mucor': []}
f_dark   = {'Aspergillus': [], 'Mucor': []}
canon = {}
for r in records:
    img_np = load_gray(r['path'])
    img_gpu = torch.from_numpy(img_np).to(DEVICE)
    m_br = seg_bright_response(img_gpu)
    m_dk = seg_dark_response(img_gpu)
    f_b = float(m_br.float().mean().item())
    f_d = float(m_dk.float().mean().item())
    f_bright[r['genus']].append(f_b); f_dark[r['genus']].append(f_d)
    r['f_bright'] = f_b; r['f_dark'] = f_d
    if r['key'] in (ASP_KEY, MUC_KEY):
        canon[r['genus']] = {'img': img_np,
                              'mask_bright': m_br.cpu().numpy(),
                              'mask_dark': m_dk.cpu().numpy()}
print(f'Done in {time.time() - t0:.1f} s')

# Stats
b_a, b_m = np.array(f_bright['Aspergillus']), np.array(f_bright['Mucor'])
d_a, d_m = np.array(f_dark['Aspergillus']),   np.array(f_dark['Mucor'])
r_b = b_a.mean() / max(b_m.mean(), 1e-9)
r_d = d_a.mean() / max(d_m.mean(), 1e-9)
p_b = stats.ttest_ind(b_a, b_m, equal_var=False).pvalue
p_d = stats.ttest_ind(d_a, d_m, equal_var=False).pvalue
sp_b = np.sqrt(((len(b_a) - 1) * b_a.var(ddof=1) + (len(b_m) - 1) * b_m.var(ddof=1))
                / (len(b_a) + len(b_m) - 2))
cd_b = (b_a.mean() - b_m.mean()) / sp_b if sp_b > 0 else 0
sp_d = np.sqrt(((len(d_a) - 1) * d_a.var(ddof=1) + (len(d_m) - 1) * d_m.var(ddof=1))
                / (len(d_a) + len(d_m) - 2))
cd_d = (d_a.mean() - d_m.mean()) / sp_d if sp_d > 0 else 0

print('\n══════════════════════════════════════════════════════════════')
print('RESULTS')
print('══════════════════════════════════════════════════════════════')
print(f'BRIGHT polarity (tissue = white pixels = hyphae):')
print(f'  Asp:   {b_a.mean():.3f} ± {b_a.std(ddof=1):.3f}  (n={len(b_a)})')
print(f'  Muc:   {b_m.mean():.3f} ± {b_m.std(ddof=1):.3f}  (n={len(b_m)})')
print(f'  Ratio: {r_b:.3f}×,  Cohen d = {cd_b:.2f},  Welch p = {p_b:.2e}')
print()
print(f'DARK polarity (published method, for comparison):')
print(f'  Asp:   {d_a.mean():.3f} ± {d_a.std(ddof=1):.3f}')
print(f'  Muc:   {d_m.mean():.3f} ± {d_m.std(ddof=1):.3f}')
print(f'  Ratio: {r_d:.3f}×,  Cohen d = {cd_d:.2f},  Welch p = {p_d:.2e}')

# Save CSV
csv_path = OUT_CSV / 'ftissue_bright_final.csv'
with open(csv_path, 'w', newline='') as fp:
    w = csv.DictWriter(fp, fieldnames=['file', 'genus', 'f_bright', 'f_dark'])
    w.writeheader()
    for r in records:
        w.writerow({'file': r['key'], 'genus': r['genus'],
                     'f_bright': round(r['f_bright'], 4),
                     'f_dark': round(r['f_dark'], 4)})
print(f'\nCSV: {csv_path}')


# ── Figure: box plot in Fig 4 stripbox style + canonical visuals ──
fig = plt.figure(figsize=(180 * MM, 130 * MM))
gs = GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.30,
               left=0.07, right=0.97, top=0.94, bottom=0.05,
               height_ratios=[1.0, 1.0])


def stripbox(ax, sa, sm, ylab, ratio, d, p):
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
    ax.set_ylabel(ylab, fontsize=7.5)
    for sp_ in ('top', 'right'): ax.spines[sp_].set_visible(False)
    ym = max(sa.max(), sm.max())
    rg = max(ym - min(sa.min(), sm.min()), 1e-6)
    y = ym + rg * 0.08
    ax.plot([1, 1, 2, 2], [y, y + rg * 0.03, y + rg * 0.03, y], 'k-', lw=0.6)
    txt = f'p = {p:.1e}' if p < 0.001 else f'p = {p:.3f}'
    ax.text(1.5, y + rg * 0.06, txt, ha='center', fontsize=6)
    ax.set_title(f'ratio = {ratio:.2f}×,  $d$ = {d:.2f}', fontsize=7, pad=3)


# A: bright box
ax = fig.add_subplot(gs[0, 0])
stripbox(ax, b_a, b_m,
          r'$f_{\mathrm{tissue}}$' + '\n(bright = hyphae)', r_b, cd_b, p_b)
ax.text(-0.22, 1.18, 'A', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')

# B: dark box (for reference)
ax = fig.add_subplot(gs[0, 1])
stripbox(ax, d_a, d_m,
          r'$f_{\mathrm{tissue}}$' + '\n(dark = hyphae, published)',
          r_d, cd_d, p_d)
ax.text(-0.22, 1.18, 'B', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')

# C: paired per-ROI scatter
ax = fig.add_subplot(gs[0, 2])
ax.scatter(d_a, b_a, s=30, c=C_ASP, edgecolors='white', lw=0.4,
            label=f'Asp (n={len(b_a)})')
ax.scatter(d_m, b_m, s=30, c=C_MUC, edgecolors='white', lw=0.4,
            label=f'Muc (n={len(b_m)})')
ax.set_xlabel(r'$f_{\mathrm{tissue}}$ (dark polarity)', fontsize=7.5)
ax.set_ylabel(r'$f_{\mathrm{tissue}}$ (bright polarity)', fontsize=7.5)
ax.legend(fontsize=6.5, frameon=False, loc='best')
for sp_ in ('top', 'right'): ax.spines[sp_].set_visible(False)
ax.text(-0.22, 1.18, 'C', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')

# D: ratio comparison
ax = fig.add_subplot(gs[0, 3])
labels = ['dark\n(published)', 'bright\n(corrected)']
ratios = [r_d, r_b]
ax.bar([0, 1], ratios, color=['#9E9E9E', '#1976D2'], alpha=0.85,
        edgecolor='black', lw=0.5, width=0.55)
for x, rr in zip([0, 1], ratios):
    ax.text(x, rr + 0.04, f'{rr:.2f}×', ha='center', fontsize=8,
             fontweight='bold')
ax.axhline(2.13, color='black', lw=0.6, ls='--', alpha=0.6)
ax.text(1.5, 2.13, ' δ ratio (2.13×)', va='center', ha='left', fontsize=6.5,
         color='black', alpha=0.7)
ax.set_xticks([0, 1])
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel(r'Asp / Muc ratio in $f_{\mathrm{tissue}}$', fontsize=7.5)
ax.set_ylim(0, max(max(ratios), 2.5) * 1.18)
for sp_ in ('top', 'right'): ax.spines[sp_].set_visible(False)
ax.text(-0.22, 1.18, 'D', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')

# Bottom row: canonical visuals — raw + dark + bright side by side
def vis(row_i, genus, color, label):
    img = canon[genus]['img']
    m_dk = canon[genus]['mask_dark']
    m_br = canon[genus]['mask_bright']
    lo, hi = np.percentile(img, [0.5, 99.5])
    nd = np.clip((img - lo) / max(hi - lo, 1), 0, 1)
    ax = fig.add_subplot(gs[1, row_i * 2])
    ax.imshow(nd, cmap='gray', interpolation='nearest'); ax.axis('off')
    ax.text(0.98, 0.03, genus, transform=ax.transAxes, fontsize=7,
             ha='right', va='bottom', style='italic', color='white',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.20', facecolor=color,
                       alpha=0.85, edgecolor='none'))
    ax.text(-0.04, 1.14, label, transform=ax.transAxes, fontsize=12,
             fontweight='bold', va='top')
    ax.set_title('Raw ROI', fontsize=7.5, pad=3)
    # Side-by-side dark vs bright
    ax_m = fig.add_subplot(gs[1, row_i * 2 + 1])
    h, w = m_br.shape
    combo = np.zeros((h, 2 * w + 8))
    combo[:, :w] = m_dk.astype(float)
    combo[:, w:w + 8] = 0.5
    combo[:, w + 8:] = m_br.astype(float)
    ax_m.imshow(combo, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    ax_m.axis('off')
    ax_m.text(0.25, -0.04, f'dark\n$f={m_dk.mean():.3f}$',
               transform=ax_m.transAxes, ha='center', va='top', fontsize=6.5)
    ax_m.text(0.75, -0.04, f'bright\n$f={m_br.mean():.3f}$',
               transform=ax_m.transAxes, ha='center', va='top', fontsize=6.5)
    ax_m.set_title('dark  |  bright', fontsize=7.5, pad=3)


vis(0, 'Aspergillus', C_ASP, 'E')
vis(1, 'Mucor',       C_MUC, 'F')

for ext in ('.png', '.pdf', '.svg'):
    kw = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.04}
    if ext == '.png': kw['dpi'] = 600
    fig.savefig(OUT_FIG / f'figS_ftissue_bright{ext}', **kw)
plt.close(fig)
print(f'Saved: figS_ftissue_bright.{{pdf,png,svg}}')

# All-asp-greater-than-all-muc check
direction_ok = b_a.min() > b_m.max()
print(f'\nAll Asp > all Muc? {direction_ok} '
      f'(Asp min = {b_a.min():.3f}, Muc max = {b_m.max():.3f})')
print('\nDone.')

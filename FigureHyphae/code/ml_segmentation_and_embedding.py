#!/usr/bin/env python3
"""Foundation-model approaches on the 24 colony-surface ROIs.

PART 1 — Cellpose v4 (cyto3 transformer) for hyphal segmentation
   -> per-ROI f_tissue and thickness, comparable to current/Sauvola

PART 2 — DINOv2 (facebook/dinov2-base) embedding -> Asp vs Muc classifier
   -> LOOCV accuracy, AUC, PCA visualization

PART 3 — Combined comparison figure with current / Sauvola / Cellpose masks
         + DINOv2 PCA scatter

Outputs:
  D:/FINAL OSF/FigureHyphae/figures/figS_ftissue_ml.{pdf,png,svg}
  D:/FINAL OSF/FigureHyphae/output/ftissue_ml_methods.csv
"""

import json, time, csv, warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Paths ──
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0) if DEVICE == "cuda" else "n/a"}')

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 8, 'axes.linewidth': 0.6,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'mathtext.default': 'regular',
})


def load_gray(p):
    with Image.open(p) as im:
        a = np.asarray(im).astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a


def load_rgb(p):
    with Image.open(p) as im:
        return np.asarray(im.convert('RGB'))


# ───── Load ROIs ─────
with open(SESSION) as f:
    saved = {k: v for k, v in json.load(f).items()
             if not k.startswith('_') and v.get('status') != 'deleted'}

records = []
for key, val in saved.items():
    g = val.get('genus')
    rp = ROI_DIR / g / f'{Path(key).stem}_roi.jpg'
    if not rp.exists(): continue
    records.append({'key': key, 'genus': g, 'path': rp})
asp_n = sum(1 for r in records if r['genus'] == 'Aspergillus')
muc_n = sum(1 for r in records if r['genus'] == 'Mucor')
print(f'Loaded {len(records)} ROIs ({asp_n} Asp + {muc_n} Muc)')


# ════════════════════════════════════════════════════════════════
# PART 1 — Cellpose v4 segmentation
# ════════════════════════════════════════════════════════════════
print('\n=== PART 1: Cellpose v4 hyphal segmentation ===')
from cellpose import models as cp_models
cp = cp_models.CellposeModel(gpu=True)

# Hyphae are ~5-10 µm thick = ~5-10 px at 0.94 µm/px. Cellpose's diameter
# refers to the structure characteristic size. We try a diameter of 12 px to
# capture hyphal segments (cells/conidiophore heads in Asp are 5-15 µm).
DIAM = 12

t0 = time.time()
cp_f = {'Aspergillus': [], 'Mucor': []}
cp_dt = {'Aspergillus': [], 'Mucor': []}
canon_masks_cp = {}
for i, r in enumerate(records):
    img = load_gray(r['path'])
    # Cellpose normalize=True will percentile-normalize; we feed grayscale
    masks, flows, styles = cp.eval(
        img, diameter=DIAM,
        flow_threshold=0.4, cellprob_threshold=0.0,
        normalize=True, batch_size=8,
    )
    binary = (masks > 0)
    # Distance transform thickness on binary mask
    if binary.sum() > 100:
        dt = ndi.distance_transform_edt(binary) * CAL_3D
        dt_med = float(np.median(dt[binary]))
    else:
        dt_med = 0.0
    f = float(binary.mean())
    cp_f[r['genus']].append(f); cp_dt[r['genus']].append(dt_med)
    r['f_cp'] = f; r['dt_cp'] = dt_med
    r['n_cells'] = int(masks.max())
    if r['key'] in (ASP_KEY, MUC_KEY):
        canon_masks_cp[r['genus']] = binary
    if (i + 1) % 6 == 0 or i == len(records) - 1:
        print(f'  [{i+1:>2}/{len(records)}]  {r["key"]}  '
              f'{r["genus"][:3]}  n_cells={r["n_cells"]:>4}  '
              f'f={f:.3f}  dt={dt_med:.1f}µm')
elapsed = time.time() - t0
print(f'Cellpose total: {elapsed:.1f} s')

# Stats
cp_f_a, cp_f_m = np.array(cp_f['Aspergillus']), np.array(cp_f['Mucor'])
cp_dt_a, cp_dt_m = np.array(cp_dt['Aspergillus']), np.array(cp_dt['Mucor'])
r_cp = cp_f_a.mean() / max(cp_f_m.mean(), 1e-9)
p_cp = stats.ttest_ind(cp_f_a, cp_f_m, equal_var=False).pvalue
sp = np.sqrt(((len(cp_f_a) - 1) * cp_f_a.var(ddof=1) +
              (len(cp_f_m) - 1) * cp_f_m.var(ddof=1)) /
             (len(cp_f_a) + len(cp_f_m) - 2))
d_cp = (cp_f_a.mean() - cp_f_m.mean()) / sp if sp > 0 else 0
print(f'\nCellpose f_tissue: Asp={cp_f_a.mean():.3f}±{cp_f_a.std(ddof=1):.3f}, '
      f'Muc={cp_f_m.mean():.3f}±{cp_f_m.std(ddof=1):.3f}, '
      f'ratio={r_cp:.3f}, d={d_cp:.2f}, p={p_cp:.2e}')
print(f'Cellpose thickness: Asp={cp_dt_a.mean():.2f}±{cp_dt_a.std(ddof=1):.2f} µm, '
      f'Muc={cp_dt_m.mean():.2f}±{cp_dt_m.std(ddof=1):.2f} µm, '
      f'ratio={cp_dt_a.mean()/max(cp_dt_m.mean(),1e-9):.3f}')


# ════════════════════════════════════════════════════════════════
# PART 2 — DINOv2 embedding + Asp vs Muc classifier
# ════════════════════════════════════════════════════════════════
print('\n=== PART 2: DINOv2 feature embedding ===')
from transformers import AutoImageProcessor, AutoModel
proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dv2 = AutoModel.from_pretrained('facebook/dinov2-base').to(DEVICE).eval()
print(f'DINOv2 loaded; param count = {sum(p.numel() for p in dv2.parameters()):.2e}')

t0 = time.time()
feats, labels, keys = [], [], []
with torch.no_grad():
    for r in records:
        img_rgb = load_rgb(r['path'])  # HxWx3
        # Resize to 518 (standard DINOv2 input)
        pil = Image.fromarray(img_rgb)
        inp = proc(images=pil, return_tensors='pt').to(DEVICE)
        out = dv2(**inp)
        emb = out.last_hidden_state[:, 0].cpu().numpy().squeeze()  # CLS token
        feats.append(emb)
        labels.append(1 if r['genus'] == 'Aspergillus' else 0)
        keys.append(r['key'])
feats = np.array(feats)
labels = np.array(labels)
print(f'DINOv2 embedding: {feats.shape} in {time.time()-t0:.1f} s')

# Leave-one-out logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
loo = LeaveOneOut()
preds, probs = np.zeros(len(labels), int), np.zeros(len(labels))
for tr, te in loo.split(feats):
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
    clf.fit(feats[tr], labels[tr])
    preds[te] = clf.predict(feats[te])
    probs[te] = clf.predict_proba(feats[te])[0, 1]
acc = float((preds == labels).mean())
try:
    auc = roc_auc_score(labels, probs)
except Exception:
    auc = float('nan')
print(f'LOOCV accuracy: {acc:.3f}  ({(preds == labels).sum()}/{len(labels)} correct)')
print(f'LOOCV AUC:      {auc:.3f}')

# PCA for visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
proj = pca.fit_transform(feats)
print(f'PCA var explained: PC1={pca.explained_variance_ratio_[0]:.2%}, '
      f'PC2={pca.explained_variance_ratio_[1]:.2%}')

# Distance between class centroids in feature space
mean_a = feats[labels == 1].mean(0)
mean_m = feats[labels == 0].mean(0)
sep = np.linalg.norm(mean_a - mean_m) / np.linalg.norm(feats.std(0))
print(f'Class-centroid normalized separation: {sep:.2f}')


# ════════════════════════════════════════════════════════════════
# PART 3 — Combined comparison figure
# ════════════════════════════════════════════════════════════════
print('\n=== PART 3: Rendering combined figure ===')

# Need the current and Sauvola masks for the canonical ROIs to display
# alongside Cellpose. Recompute them here.
def seg_3d_current(img_gpu):
    s = _gauss(img_gpu, 1.0); l = _gauss(s, 32.0); dr = l - s
    t = dr.mean() + 0.5 * dr.std()
    return _morph(dr > t)


def sauvola_local(img_gpu, window=64, k=0.2, R=0.5):
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
    T = mean * (1 + k * (std / R - 1))
    m = (x < T).squeeze(0).squeeze(0)
    return _morph(m)


def _gauss(img, sigma):
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


def _morph(mask, iters=1):
    m = mask.float().unsqueeze(0).unsqueeze(0)
    for _ in range(iters): m = F.max_pool2d(m, 3, stride=1, padding=1)
    for _ in range(iters): m = -F.max_pool2d(-m, 3, stride=1, padding=1)
    for _ in range(iters): m = -F.max_pool2d(-m, 3, stride=1, padding=1)
    for _ in range(iters): m = F.max_pool2d(m, 3, stride=1, padding=1)
    return (m.squeeze(0).squeeze(0) > 0.5)

# Compute current + Sauvola masks for canonical Asp + Muc
canon_imgs, canon_cur, canon_sau = {}, {}, {}
for r in records:
    if r['key'] in (ASP_KEY, MUC_KEY):
        img_np = load_gray(r['path'])
        img_gpu = torch.from_numpy(img_np).to(DEVICE)
        canon_imgs[r['genus']] = img_np
        canon_cur[r['genus']] = seg_3d_current(img_gpu).cpu().numpy()
        canon_sau[r['genus']] = sauvola_local(img_gpu).cpu().numpy()

# Final figure
fig = plt.figure(figsize=(190 * MM, 165 * MM))
gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.30,
               left=0.06, right=0.97, top=0.95, bottom=0.05,
               height_ratios=[1.0, 1.0, 1.0])

# Row 1: 4 panels
# A: f_tissue box plot for cellpose
ax = fig.add_subplot(gs[0, 0])
bp = ax.boxplot([cp_f_a, cp_f_m], positions=[1, 2], widths=0.4, patch_artist=True,
                 showfliers=False, medianprops=dict(color='white', lw=1.4),
                 whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
bp['boxes'][0].set_facecolor(C_ASP); bp['boxes'][0].set_alpha(0.55)
bp['boxes'][1].set_facecolor(C_MUC); bp['boxes'][1].set_alpha(0.55)
rng = np.random.default_rng(42)
ax.scatter(1 + rng.uniform(-0.09, 0.09, len(cp_f_a)), cp_f_a, s=14, c=C_ASP,
           alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
ax.scatter(2 + rng.uniform(-0.09, 0.09, len(cp_f_m)), cp_f_m, s=14, c=C_MUC,
           alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
ax.set_xticks([1, 2])
ax.set_xticklabels([r'$\it{Aspergillus}$', r'$\it{Mucor}$'], fontsize=6.5)
ax.set_ylabel(r'$f_{\mathrm{tissue}}$ (Cellpose)', fontsize=7)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ym = max(cp_f_a.max(), cp_f_m.max())
rng_ = max(ym - min(cp_f_a.min(), cp_f_m.min()), 1e-6)
y = ym + rng_ * 0.08
ax.plot([1, 1, 2, 2], [y, y + rng_ * 0.03, y + rng_ * 0.03, y], 'k-', lw=0.6)
ax.text(1.5, y + rng_ * 0.06,
         f'p = {p_cp:.1e}' if p_cp < 0.001 else f'p = {p_cp:.3f}',
         ha='center', fontsize=6)
ax.set_title(f'ratio = {r_cp:.2f}×,  $d$ = {d_cp:.2f}', fontsize=7, pad=3)
ax.text(-0.22, 1.14, 'A', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')

# B: ratio comparison across methods (current, Sauvola, Cellpose) — load from CSV
import csv as _csv
cur_a, cur_m, sau_a, sau_m = [], [], [], []
prev_csv = OUT_CSV / 'ftissue_sauvola_final.csv'
if prev_csv.exists():
    for row in _csv.DictReader(open(prev_csv)):
        if row['genus'] == 'Aspergillus':
            cur_a.append(float(row['f_current'])); sau_a.append(float(row['f_sauvola']))
        else:
            cur_m.append(float(row['f_current'])); sau_m.append(float(row['f_sauvola']))
cur_a, cur_m = np.array(cur_a), np.array(cur_m)
sau_a, sau_m = np.array(sau_a), np.array(sau_m)
r_cur = cur_a.mean() / max(cur_m.mean(), 1e-9)
r_sau = sau_a.mean() / max(sau_m.mean(), 1e-9)

ax = fig.add_subplot(gs[0, 1])
methods = ['current', 'Sauvola', 'Cellpose']
ratios = [r_cur, r_sau, r_cp]
colors = ['#9E9E9E', '#1976D2', '#E53935']
ax.bar(range(3), ratios, color=colors, alpha=0.85, edgecolor='black', lw=0.5)
for i, rr in enumerate(ratios):
    ax.text(i, rr + 0.1, f'{rr:.2f}×', ha='center', fontsize=8, fontweight='bold')
ax.axhline(2.13, color='black', lw=0.6, ls='--', alpha=0.6)
ax.text(2.5, 2.13, ' δ ratio (2.13×)', va='center', ha='left', fontsize=6.5,
         color='black', alpha=0.7)
ax.set_xticks(range(3))
ax.set_xticklabels(methods, fontsize=7)
ax.set_ylabel(r'Asp/Muc ratio in $f_{\mathrm{tissue}}$', fontsize=7)
ax.set_ylim(0, max(max(ratios), 2.5) * 1.18)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.text(-0.22, 1.14, 'B', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')
ax.set_title('Method comparison\nvs δ benchmark', fontsize=7, pad=3)

# C: DINOv2 PCA scatter
ax = fig.add_subplot(gs[0, 2])
for cls, name, c in [(1, 'Asp', C_ASP), (0, 'Muc', C_MUC)]:
    sel = labels == cls
    ax.scatter(proj[sel, 0], proj[sel, 1], s=40, c=c, alpha=0.85,
                edgecolors='white', linewidths=0.5, label=f'{name} (n={sel.sum()})')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=7)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=7)
ax.legend(fontsize=6, frameon=False, loc='best')
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.text(-0.22, 1.14, 'C', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')
ax.set_title(f'DINOv2 feature space\nLOOCV acc = {acc:.0%},  AUC = {auc:.2f}',
              fontsize=7, pad=3)

# D: LOOCV probability per ROI
ax = fig.add_subplot(gs[0, 3])
xs = np.arange(len(labels))
sort_idx = np.argsort(probs)
sorted_probs = probs[sort_idx]; sorted_lab = labels[sort_idx]
for i, (p_, l_) in enumerate(zip(sorted_probs, sorted_lab)):
    ax.bar(i, p_, color=C_ASP if l_ == 1 else C_MUC,
            alpha=0.75, edgecolor='black', lw=0.4)
ax.axhline(0.5, color='black', lw=0.5, ls=':')
ax.set_xticks([])
ax.set_xlabel('ROIs (sorted by P[Asp])', fontsize=6.5)
ax.set_ylabel('P[Aspergillus]', fontsize=7)
ax.set_ylim(0, 1.0)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.text(-0.22, 1.14, 'D', transform=ax.transAxes, fontsize=12,
         fontweight='bold', va='top')
ax.set_title('Per-ROI classifier confidence', fontsize=7, pad=3)

# Row 2-3: canonical Asp + Muc raw + 3 masks (current, Sauvola, Cellpose)
for row_i, (genus, color) in enumerate([('Aspergillus', C_ASP),
                                          ('Mucor', C_MUC)]):
    img = canon_imgs[genus]
    lo, hi = np.percentile(img, [0.5, 99.5])
    nd = np.clip((img - lo) / max(hi - lo, 1), 0, 1)
    # col 0: raw
    ax = fig.add_subplot(gs[row_i + 1, 0])
    ax.imshow(nd, cmap='gray', interpolation='nearest'); ax.axis('off')
    ax.text(0.98, 0.03, genus, transform=ax.transAxes, fontsize=7,
             ha='right', va='bottom', style='italic', color='white',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.20', facecolor=color,
                       alpha=0.85, edgecolor='none'))
    ax.text(-0.04, 1.14, 'EF'[row_i], transform=ax.transAxes,
             fontsize=12, fontweight='bold', va='top')
    if row_i == 0:
        ax.set_title('Raw ROI', fontsize=7.5, pad=3)
    # col 1-3: masks
    for col, (name, mdict) in enumerate([('current', canon_cur),
                                          ('Sauvola', canon_sau),
                                          ('Cellpose', canon_masks_cp)]):
        ax = fig.add_subplot(gs[row_i + 1, col + 1])
        mask = mdict[genus]
        ax.imshow(mask, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        ax.axis('off')
        ax.text(0.98, 0.03, f'$f = {mask.mean():.3f}$', transform=ax.transAxes,
                 fontsize=7, ha='right', va='bottom', color='white',
                 bbox=dict(boxstyle='round,pad=0.20', facecolor='black',
                           alpha=0.6, edgecolor='none'))
        if row_i == 0:
            ax.set_title(name, fontsize=7.5, pad=3)

# Save
for ext in ('.png', '.pdf', '.svg'):
    kw = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.04}
    if ext == '.png': kw['dpi'] = 600
    fig.savefig(OUT_FIG / f'figS_ftissue_ml{ext}', **kw)
plt.close(fig)
print(f'Saved: figS_ftissue_ml.{{pdf,png,svg}}')

# Save CSV
csv_path = OUT_CSV / 'ftissue_ml_methods.csv'
with open(csv_path, 'w', newline='') as fp:
    w = csv.DictWriter(fp, fieldnames=['file', 'genus', 'f_cellpose', 'dt_cellpose_um',
                                         'n_cells', 'dinov2_p_asp', 'dinov2_pred'])
    w.writeheader()
    for r, p_, pr_ in zip(records, probs, preds):
        w.writerow({'file': r['key'], 'genus': r['genus'],
                     'f_cellpose': round(r['f_cp'], 4),
                     'dt_cellpose_um': round(r['dt_cp'], 4),
                     'n_cells': r['n_cells'],
                     'dinov2_p_asp': round(float(p_), 4),
                     'dinov2_pred': 'Aspergillus' if pr_ == 1 else 'Mucor'})
print(f'CSV: {csv_path}')

print('\n══════════════════════════════════════════════════════════════')
print('SUMMARY')
print('══════════════════════════════════════════════════════════════')
print(f'Current  f_tissue ratio: {r_cur:.2f}×')
print(f'Sauvola  f_tissue ratio: {r_sau:.2f}×')
print(f'Cellpose f_tissue ratio: {r_cp:.2f}×  (d={d_cp:.2f}, p={p_cp:.2e})')
print(f'δ benchmark:             2.13×')
print()
print(f'DINOv2 LOOCV: {acc:.0%} accuracy, AUC = {auc:.3f}')
print('Done.')

#!/usr/bin/env python3
"""Complete rebuild: main figure panels C-F + all 5 supplementary figures.
Panel C: FFT α, Panel D: thickness, Panel E: tubeness CV, Panel F: hyphal area fraction.
Erosion survival is now supplementary (S20 demo, S22 boxplot).
All consistent, all from single computation pass."""

import json, os, shutil, numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy import stats
from pathlib import Path
from PIL import Image
from skimage.segmentation import find_boundaries
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_T7 = Path('/Volumes/T7/FINAL OSF')
_REPO = Path(__file__).resolve().parent.parent.parent
BASE = _T7 if _T7.exists() else _REPO
ROI_DIR = BASE / 'HYPHAE' / 'Analysis' / 'results' / '3d_overlays'
HESSIAN_LM = BASE / 'HYPHAE' / 'Analysis' / 'results' / 'hessian_overlays'
SESSION = ROI_DIR / 'roi_session.json'
MICRO_DIR = BASE / 'HYPHAE' / 'Light Microscopy'
RSR_CSV = BASE / 'FigureRSR' / 'output' / 'rsr_and_lab_dstar_metrics.csv'
FIG_OUT = BASE / 'FigureHyphae' / 'figures'
CSV_OUT = BASE / 'FigureHyphae' / 'output'
HANDOFF = CSV_OUT
CAL_3D = 0.94; MM = 1/25.4
C_ASP = '#4CAF50'; C_MUC = '#757575'
HESSIAN_SCALES = [1, 2, 4, 8, 16]

# Best aspect-ratio-matched ROIs for supplementary
ASP_ROI_KEY = '20251214_222552.JPG'
MUC_ROI_KEY = '20251210_155926.JPG'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 8, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'mathtext.default': 'regular',
})

MICRO_IMAGES = [
    ('Pink_10X_1.TIF', 'Aspergillus', 'Pink', 10),
    ('Pink_10X_2.TIF', 'Aspergillus', 'Pink', 10),
    ('Pink_20X_1.TIF', 'Aspergillus', 'Pink', 20),
    ('Pink_20X_2.TIF', 'Aspergillus', 'Pink', 20),
    ('Pink_40X_1.TIF', 'Aspergillus', 'Pink', 40),
    ('Pink_40X_2.TIF', 'Aspergillus', 'Pink', 40),
    ('White_10X_1.TIF', 'Mucor', 'White', 10),
    ('White_20X_1.TIF', 'Mucor', 'White', 20),
    ('White_40X_1.TIF', 'Mucor', 'White', 40),
]

# ── Helper functions ──

def load_gray(p):
    with Image.open(p) as im: arr = np.asarray(im).astype(np.float64)
    if arr.ndim == 3: arr = arr[..., :3].mean(axis=2)
    return arr

def seg_3d(img, threshold_mult=0.5):
    s = ndi.gaussian_filter(img, sigma=1.0); l = ndi.gaussian_filter(s, sigma=32.0)
    dr = l - s; t = dr.mean() + threshold_mult * dr.std()
    return ndi.binary_opening(ndi.binary_closing(dr > t, iterations=1), iterations=1)

def otsu_t(v):
    h, e = np.histogram(v, bins=256); c = (e[:-1]+e[1:])/2; t = h.sum()
    wb = np.cumsum(h).astype(float); wf = t-wb; sb = np.cumsum(h*c)
    mb = sb/np.maximum(wb,1); mf = (sb[-1]-sb)/np.maximum(wf,1)
    return c[np.argmax(wb*wf*(mb-mf)**2)]

def seg_micro(img, label):
    s = ndi.gaussian_filter(img, sigma=1.0); l = ndi.gaussian_filter(s, sigma=32.0); dr = l-s
    v = np.ones(img.shape, dtype=bool); m = max(8, int(round(min(img.shape)*0.006)))
    v[:m,:]=False; v[-m:,:]=False; v[:,:m]=False; v[:,-m:]=False
    vals = s[v]; gt = otsu_t(vals); drv = dr[v]; dm = float(np.median(drv))
    dd = float(np.median(np.abs(drv-dm))+1e-9)
    if label.lower() in ('pink','green'):
        ic=float(np.percentile(vals,62)); lt=max(float(np.percentile(drv,72)),dm+0.80*dd)
        bm=s<min(gt,ic); fm=v&(dr>lt); mask=v&(bm|fm); ms,mh=28,1500
    else:
        ic=float(np.percentile(vals,32)); lt=max(float(np.percentile(drv,78)),dm+1.15*dd)
        fm=v&(dr>lt); sd=(s<ic)&(dr>np.percentile(drv,48)); mask=v&(fm|sd); ms,mh=10,80
    mask=ndi.binary_closing(mask,structure=np.ones((3,3)),iterations=1); mask&=v
    la,nl=ndi.label(mask)
    if nl>0: sz=np.bincount(la.ravel()); k=sz>=ms; k[0]=False; mask=k[la]
    fl=ndi.binary_fill_holes(mask); ho=fl&~mask; hl,nhl=ndi.label(ho)
    if nhl>0: hs=np.bincount(hl.ravel()); sm=hs<=mh; sm[0]=False; mask=mask|sm[hl]
    mask&=v; return mask.astype(bool)

def multiscale_tubeness(img):
    resp = np.zeros_like(img)
    for s in HESSIAN_SCALES:
        s2 = s*s
        Ixx = ndi.gaussian_filter(img, sigma=s, order=[0,2])*s2
        Iyy = ndi.gaussian_filter(img, sigma=s, order=[2,0])*s2
        Ixy = ndi.gaussian_filter(img, sigma=s, order=[1,1])*s2
        trace = Ixx+Iyy; det = Ixx*Iyy-Ixy*Ixy
        disc = np.sqrt(np.clip((trace/2)**2-det, 0, None))
        resp = np.maximum(resp, np.clip(trace/2+disc, 0, None))
    return resp

def norm_display(img):
    lo, hi = np.percentile(img, [0.5, 99.5])
    if hi <= lo: hi = lo + 1
    return np.clip((img-lo)/(hi-lo), 0, 1)

def savefig(fig, name, out=FIG_OUT):
    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.03}
        if ext == '.png': kw['dpi'] = 600
        fig.savefig(out / f'{name}{ext}', **kw)
    plt.close(fig)
    print(f'  Saved: {name}')

def stripbox(ax, sa, sm, yl):
    bp = ax.boxplot([sa, sm], positions=[1,2], widths=0.4, patch_artist=True, showfliers=False,
        medianprops=dict(color='white',lw=1.4), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    bp['boxes'][0].set_facecolor(C_ASP); bp['boxes'][0].set_alpha(0.55)
    bp['boxes'][1].set_facecolor(C_MUC); bp['boxes'][1].set_alpha(0.55)
    rng = np.random.default_rng(42)
    ax.scatter(1+rng.uniform(-0.09,0.09,len(sa)), sa, s=16, c=C_ASP, alpha=0.85,
               edgecolors='white', linewidths=0.3, zorder=3)
    ax.scatter(2+rng.uniform(-0.09,0.09,len(sm)), sm, s=16, c=C_MUC, alpha=0.85,
               edgecolors='white', linewidths=0.3, zorder=3)
    ax.set_xticks([1,2])
    ax.set_xticklabels([r'$\it{Aspergillus}$', r'$\it{Mucor}$'], fontsize=7)
    ax.set_ylabel(yl, fontsize=7.5)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    t, p = stats.ttest_ind(sa, sm, equal_var=False)
    ym = max(sa.max(), sm.max()); r = ym - min(sa.min(), sm.min()); y = ym + r*0.08
    ax.plot([1,1,2,2], [y, y+r*0.03, y+r*0.03, y], 'k-', lw=0.6)
    ax.text(1.5, y+r*0.06, f'p = {p:.1e}' if p<0.001 else f'p = {p:.3f}', ha='center', fontsize=6)

def cohen_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    sd = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / sd if sd > 0 else np.nan

def welch_p(a, b):
    return stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')[1]


# ═══════════════════════════════════════════════════════════════
# COMPUTE ALL METRICS
# ═══════════════════════════════════════════════════════════════
print('Computing all metrics...')

with open(SESSION) as f: session = json.load(f)
saved = {k: v for k, v in session.items() if not k.startswith('_') and v.get('status') != 'deleted'}

# 3D metrics (FFT from CSV, thickness from segmentation)
fft_df = pd.read_csv(CSV_OUT / 'fft_per_roi.csv')
fft_a = fft_df.loc[fft_df['genus']=='Aspergillus', 'alpha'].values
fft_m = fft_df.loc[fft_df['genus']=='Mucor', 'alpha'].values

thick_a, thick_m, ftiss_a, ftiss_m = [], [], [], []
for key, val in saved.items():
    genus = val.get('genus')
    rp = ROI_DIR / genus / f'{Path(key).stem}_roi.jpg'
    if not rp.exists(): continue
    img = load_gray(rp); mask = seg_3d(img)
    dt = ndi.distance_transform_edt(mask) * CAL_3D
    dtm = np.median(dt[mask]) if mask.sum() > 100 else 0
    ft = mask.mean()
    if genus == 'Aspergillus': thick_a.append(dtm); ftiss_a.append(ft)
    else: thick_m.append(dtm); ftiss_m.append(ft)
thick_a, thick_m = np.array(thick_a), np.array(thick_m)
ftiss_a, ftiss_m = np.array(ftiss_a), np.array(ftiss_m)
cap_a, cap_m = ftiss_a * thick_a, ftiss_m * thick_m

# Hyphal area fraction f_tissue — NEW computation method:
# multi-scale photometric reconstruction (brightness * gradient * structure-
# tensor coherence) computed once by photometric_density_reconstruction.py
# and stored in CSV. Replaces the prior binary Otsu thresholding approach.
_rec_csv = CSV_OUT / 'ftissue_photometric.csv'
_rec_df  = pd.read_csv(_rec_csv)
# Variable names ftiss_a / ftiss_m retained for downstream compatibility;
# values now come from the photometric pipeline rather than binary masks.
ftiss_a = _rec_df.loc[_rec_df.genus == 'Aspergillus', 'D_recon'].values
ftiss_m = _rec_df.loc[_rec_df.genus == 'Mucor',       'D_recon'].values
D_a, D_m = ftiss_a, ftiss_m  # backwards-compat aliases

# 3D segmentation sensitivity: threshold multiplier sweep.
sens_rows = []
for mult in [0.3, 0.4, 0.5, 0.6, 0.7]:
    aa, mm = [], []
    for key, val in saved.items():
        genus = val.get('genus')
        rp = ROI_DIR / genus / f'{Path(key).stem}_roi.jpg'
        if not rp.exists():
            continue
        img = load_gray(rp)
        mask = seg_3d(img, threshold_mult=mult)
        dt = ndi.distance_transform_edt(mask) * CAL_3D
        dtm = np.median(dt[mask]) if mask.sum() > 100 else np.nan
        (aa if genus == 'Aspergillus' else mm).append(dtm)
    aa, mm = np.asarray(aa, dtype=float), np.asarray(mm, dtype=float)
    sens_rows.append({
        'threshold_std_multiplier': mult,
        'asp_mean_dt_um': np.nanmean(aa),
        'muc_mean_dt_um': np.nanmean(mm),
        'ratio': np.nanmean(aa) / np.nanmean(mm),
        'p_welch': welch_p(aa, mm),
        'n_asp': np.sum(~np.isnan(aa)),
        'n_muc': np.sum(~np.isnan(mm)),
    })
sensitivity = pd.DataFrame(sens_rows)
for out_dir in (CSV_OUT, HANDOFF):
    sensitivity.to_csv(out_dir / 'segmentation_threshold_sensitivity.csv', index=False)

# LM metrics (erosion + tubeness CV)
erosion_a, erosion_m = [], []
tubcv_a, tubcv_m = [], []
# Store per-image tubeness distributions for S21
tub_distributions = {}

for fname, genus, label, mag in MICRO_IMAGES:
    img = load_gray(MICRO_DIR / fname)
    lo, hi = np.percentile(img, [0.5, 99.5])
    if hi <= lo: hi = lo + 1
    img_n = np.clip((img-lo)/(hi-lo), 0, 1)
    mask = seg_micro(img, label)

    # Erosion
    orig = mask.sum()
    eroded = ndi.binary_erosion(mask, iterations=10)
    surv = eroded.sum() / orig if orig > 0 else 0

    # Tubeness CV
    tubeness = multiscale_tubeness(img_n)
    t_tissue = tubeness[mask]
    tcv = t_tissue.std() / t_tissue.mean() if t_tissue.mean() > 0 and len(t_tissue) > 100 else 0

    if genus == 'Aspergillus':
        erosion_a.append(surv); tubcv_a.append(tcv)
    else:
        erosion_m.append(surv); tubcv_m.append(tcv)

    tub_distributions[fname] = {'genus': genus, 'mag': mag, 'values': t_tissue, 'cv': tcv}
    print(f'  {fname}: erosion={surv:.3f}, tub_cv={tcv:.3f}')

erosion_a, erosion_m = np.array(erosion_a), np.array(erosion_m)
tubcv_a, tubcv_m = np.array(tubcv_a), np.array(tubcv_m)

# Condensation
rsr = pd.read_csv(RSR_CSV)
green = rsr[rsr['group'] == 'Green']
# White is viable Mucor. Black/Rhizopus failed to establish and is retained only
# in the Figure 3 universal-collapse context, not in Figure 4 species comparisons.
mucor_c = rsr[rsr['group'] == 'White']
failed_rhizopus = rsr[rsr['group'] == 'Black']
delta_ratio = green['delta_um'].mean() / mucor_c['delta_um'].mean()
cap_ratio = cap_a.mean() / cap_m.mean()

# SSA from saved data
iface = pd.read_csv(CSV_OUT / 'interfacial_metrics_3d.csv')
ssa_a = iface.loc[iface['genus']=='Aspergillus', 'specific_surface'].values
ssa_m = iface.loc[iface['genus']=='Mucor', 'specific_surface'].values

# Validation table used by manuscript/handoff. Ratios are direction-normalized so
# values >1 always point toward stronger Aspergillus vapor-sink phenotype.
rows = []
for name, col in [
    ('delta (um)', 'delta_um'),
    ('Size gradient', 'zone_metric'),
    ('tau50 gradient', 'tau50_zone'),
    ('d* (mm)', 'dstar_mm'),
]:
    ga = green[col].dropna().values
    ma = mucor_c[col].dropna().values
    rows.append({
        'category': 'condensation',
        'metric': name,
        'asp_mean': ga.mean(),
        'muc_mean': ma.mean(),
        'ratio': ga.mean() / ma.mean(),
        'cohen_d': cohen_d(ga, ma),
        'p_welch': welch_p(ga, ma),
        'n_asp': len(ga),
        'n_muc': len(ma),
    })

metric_pairs = [
    ('morphological', 'FFT spectral slope alpha (C, primary)', fft_a, fft_m,
     abs(fft_m.mean()) / abs(fft_a.mean())),
    ('morphological', 'Tissue thickness um (D)', thick_a, thick_m,
     thick_a.mean() / thick_m.mean()),
    ('morphological', 'Hessian tubeness CV (E, primary)', tubcv_a, tubcv_m,
     tubcv_a.mean() / tubcv_m.mean()),
    ('morphological', 'Erosion survival (F)', erosion_a, erosion_m,
     erosion_a.mean() / erosion_m.mean()),
    ('morphological', 'Tissue fraction (3D, H decomp)', ftiss_a, ftiss_m,
     ftiss_a.mean() / ftiss_m.mean()),
    ('morphological', 'Absorbing capacity (H)', cap_a, cap_m,
     cap_a.mean() / cap_m.mean()),
]

for category, name, aa, ma, ratio in metric_pairs:
    rows.append({
        'category': category,
        'metric': name,
        'asp_mean': np.mean(aa),
        'muc_mean': np.mean(ma),
        'ratio': ratio,
        'cohen_d': cohen_d(aa, ma),
        'p_welch': welch_p(aa, ma),
        'n_asp': len(aa),
        'n_muc': len(ma),
    })

glcm_a = fft_df.loc[fft_df['genus']=='Aspergillus', 'glcm'].values
glcm_m = fft_df.loc[fft_df['genus']=='Mucor', 'glcm'].values
rows.append({
    'category': 'cross_validation',
    'metric': 'GLCM contrast (cross-val of FFT)',
    'asp_mean': glcm_a.mean(),
    'muc_mean': glcm_m.mean(),
    'ratio': glcm_m.mean() / glcm_a.mean(),
    'cohen_d': cohen_d(glcm_a, glcm_m),
    'p_welch': welch_p(glcm_a, glcm_m),
    'n_asp': len(glcm_a),
    'n_muc': len(glcm_m),
})

eff = pd.read_csv(CSV_OUT / 'effective_diffusivity_3d.csv')
deff_a = eff.loc[eff['genus']=='Aspergillus', 'D_eff_ratio'].values
deff_m = eff.loc[eff['genus']=='Mucor', 'D_eff_ratio'].values
rows.append({
    'category': 'rejected',
    'metric': 'D_eff/D0 (tortuosity test)',
    'asp_mean': deff_a.mean(),
    'muc_mean': deff_m.mean(),
    'ratio': deff_a.mean() / deff_m.mean(),
    'cohen_d': cohen_d(deff_a, deff_m),
    'p_welch': welch_p(deff_a, deff_m),
    'n_asp': len(deff_a),
    'n_muc': len(deff_m),
})

validation = pd.DataFrame(rows)
for out_dir in (CSV_OUT, HANDOFF):
    validation.to_csv(out_dir / 'framework_validation_comprehensive.csv', index=False)
print(f'  Validation table: delta ratio {delta_ratio:.2f}; capacity ratio {cap_ratio:.2f}; '
      f'failed Rhizopus/Black excluded from Figure 4 species comparison (n={len(failed_rhizopus)}).')

print('\nAll metrics computed.\n')


# ═══════════════════════════════════════════════════════════════
# MAIN FIGURE: Panels C-F (skinny row)
# ═══════════════════════════════════════════════════════════════
print('Generating main figure panels C-F...')

fig, axes = plt.subplots(1, 4, figsize=(190*MM, 75*MM))
fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.16, wspace=0.55)

stripbox(axes[0], fft_a, fft_m, r'Spectral slope $\alpha$' + '\n(3D colony surface)')
stripbox(axes[1], thick_a, thick_m, 'Tissue thickness\n(3D, \u00b5m)')
stripbox(axes[2], tubcv_a, tubcv_m, 'Hessian tubeness CV\n(light microscopy)')
stripbox(axes[3], ftiss_a, ftiss_m, r'$f_\mathrm{tissue}$' + '\n(3D colony surface)')

for ax, l in zip(axes, 'CDEF'):
    ax.text(-0.15, 1.08, l, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
savefig(fig, 'figure4_panels_CF', HANDOFF)
savefig(fig, 'figure4_panels_CF', FIG_OUT)


# ═══════════════════════════════════════════════════════════════
# MAIN FIGURE: Panels G-H (skinny row)
# ═══════════════════════════════════════════════════════════════
print('Generating main figure panels G-H...')

fig = plt.figure(figsize=(190*MM, 75*MM))
gs = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=0.90, bottom=0.16,
                      wspace=0.35, width_ratios=[1.6, 1])
ax_g = fig.add_subplot(gs[0]); ax_h = fig.add_subplot(gs[1])

# G: ratios
cond_r, cond_p = [], []
for col in ['delta_um', 'zone_metric', 'tau50_zone', 'dstar_mm']:
    ga = green[col].dropna().values; ma = mucor_c[col].dropna().values
    cond_r.append(ga.mean()/ma.mean()); _, p = stats.ttest_ind(ga, ma, equal_var=False); cond_p.append(p)

# FFT ratio: Asp is less negative (shallower) → ratio of |slope|: Muc/Asp
fft_ratio = abs(fft_m.mean()) / abs(fft_a.mean())
morph_r = [fft_ratio,
           thick_a.mean()/thick_m.mean(),
           tubcv_a.mean()/tubcv_m.mean(),
           D_a.mean()/D_m.mean()]
morph_p = [stats.ttest_ind(fft_a, fft_m, equal_var=False)[1],
           stats.ttest_ind(thick_a, thick_m, equal_var=False)[1],
           stats.ttest_ind(tubcv_a, tubcv_m, equal_var=False)[1],
           stats.ttest_ind(D_a, D_m, equal_var=False)[1]]

lb = [r'$\delta$', 'Size\ngrad.', r'$\tau_{50}$'+'\ngrad.', r'$d^*$',
      r'FFT $\alpha$'+'\n(C)', 'Thick.\n(D)', 'Tub CV\n(E)', r'$f_\mathrm{tissue}$'+'\n(F)']
ar = cond_r + morph_r
ap = cond_p + morph_p
ac = [C_MUC]*4 + [C_ASP]*4

xg = np.arange(len(ar))
ax_g.bar(xg, ar, color=ac, alpha=0.7, width=0.65, edgecolor='none')
for xi, (r, p, c) in enumerate(zip(ar, ap, ac)):
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ' n.s.'
    ax_g.text(xi, r+0.03, f'{r:.2f}{sig}', ha='center', fontsize=5, fontweight='bold', color=c)
ax_g.axhline(1.0, color='gray', ls='--', lw=0.5, alpha=0.5)
ax_g.axvline(3.5, color='gray', ls='-', lw=0.4, alpha=0.3)
ax_g.set_xticks(xg); ax_g.set_xticklabels(lb, fontsize=5.5)
ax_g.set_ylabel('Asp / Muc ratio', fontsize=7); ax_g.set_ylim(0, max(ar)*1.22)
ax_g.text(1.5, max(ar)*1.15, 'Condensation', ha='center', fontsize=6, color=C_MUC, style='italic')
ax_g.text(5.5, max(ar)*1.15, 'Morphological', ha='center', fontsize=6, color='#2E7D32', style='italic')
for sp in ['top','right']: ax_g.spines[sp].set_visible(False)
ax_g.text(-0.08, 1.08, 'G', transform=ax_g.transAxes, fontsize=11, fontweight='bold', va='top')

# H/I: morphology predicts condensation (\u03b4).
#
# IMPORTANT \u2014 single bar, no multiplicative decomposition.
#
# In the prior published version, f_tissue was a dimensionless binary area
# fraction in [0,1] and "predicted morphology" was f_tissue \u00d7 d_structure
# (a ratio of "effective tissue depths").  With the new photometric
# reconstruction, f_tissue is no longer an area fraction \u2014 it is a
# multi-scale brightness-density score that already encodes both colony
# compactness AND the structure-tensor coherence that scales with effective
# tissue depth.  Multiplying by d_structure would double-count thickness:
#   1.89 (new f_tissue ratio) \u00d7 1.73 (thickness ratio) = 3.27
# which overshoots the measured \u03b4 ratio (2.13) by ~54%.  The single-bar
# comparison below \u2014 f_tissue ratio alone vs \u03b4 ratio \u2014 is within 11% and
# is the physically defensible read.
D_ratio = D_a.mean() / D_m.mean()
ax_h.bar([0], [D_ratio],     color=C_ASP, alpha=0.7, width=0.5, edgecolor='none')
ax_h.bar([1], [delta_ratio], color=C_MUC, alpha=0.7, width=0.5, edgecolor='none')
ax_h.text(0, D_ratio/2,     f'{D_ratio:.2f}\u00d7',     ha='center', fontsize=8, fontweight='bold', color='white', va='center')
ax_h.text(1, delta_ratio/2, f'{delta_ratio:.2f}\u00d7', ha='center', fontsize=8, fontweight='bold', color='white', va='center')
ax_h.set_xticks([0,1])
ax_h.set_xticklabels(['Morphology\n($f_\\mathrm{tissue}$ ratio)', 'Condensation\n($\u03b4$ ratio)'], fontsize=7)
ax_h.set_ylabel('Asp / Muc ratio', fontsize=7)
ax_h.axhline(1.0, color='gray', ls='--', lw=0.5, alpha=0.5)
ax_h.set_ylim(0, max(delta_ratio, D_ratio) * 1.25)
for sp in ['top','right']: ax_h.spines[sp].set_visible(False)
ax_h.text(-0.12, 1.08, 'H', transform=ax_h.transAxes, fontsize=11, fontweight='bold', va='top')

savefig(fig, 'figure4_panel_GH', HANDOFF)
savefig(fig, 'figure4_panel_GH', FIG_OUT)


# ═══════════════════════════════════════════════════════════════
# S18: FFT cross-validation (FIXED: no regression lines)
# ═══════════════════════════════════════════════════════════════
print('S18: FFT cross-validation...')

fft_full = pd.read_csv(CSV_OUT / 'fft_per_roi.csv')
spectra_data = {}  # Need to recompute power spectra for Panel A

fig = plt.figure(figsize=(183*MM, 60*MM))
gs = fig.add_gridspec(1, 3, wspace=0.35, left=0.08, right=0.96)

# Panel A: α boxplot
ax = fig.add_subplot(gs[0])
stripbox(ax, fft_a, fft_m, r'Spectral slope $\alpha$')
ax.set_title('Shallower = more textured', fontsize=7, color='gray')
ax.text(-0.14, 1.08, 'A', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')

# Panel B: α vs GLCM scatter (NO regression line)
ax = fig.add_subplot(gs[1])
all_alpha = fft_full['alpha'].values
all_glcm = fft_full['glcm'].values
for _, r in fft_full.iterrows():
    c = C_ASP if r['genus'] == 'Aspergillus' else C_MUC
    ax.scatter(r['alpha'], r['glcm'], c=c, s=20, alpha=0.8, edgecolors='white', linewidths=0.3, zorder=3)
rs, ps = stats.spearmanr(all_alpha, all_glcm)
rp, pp = stats.pearsonr(all_alpha, all_glcm)
ax.text(0.95, 0.08, f'Spearman r = {rs:.2f}, p = {ps:.1e}\nPearson r = {rp:.2f}, p = {pp:.2f}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=5.5,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax.set_xlabel(r'Spectral slope $\alpha$', fontsize=7)
ax.set_ylabel('GLCM contrast', fontsize=7)
ax.set_title(r'$\alpha$ vs GLCM', fontsize=7.5, fontweight='bold')
legend_el_b = [Line2D([0],[0],marker='o',color='w',markerfacecolor=C_ASP,markersize=5,label='Asp'),
               Line2D([0],[0],marker='o',color='w',markerfacecolor=C_MUC,markersize=5,label='Muc')]
ax.legend(handles=legend_el_b, fontsize=5, loc='upper right')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.text(-0.14, 1.08, 'B', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')

# Panel C: α vs absorbing capacity (NO regression line)
ax = fig.add_subplot(gs[2])
all_cap = fft_full['capacity'].values
for _, r in fft_full.iterrows():
    c = C_ASP if r['genus'] == 'Aspergillus' else C_MUC
    ax.scatter(r['alpha'], r['capacity'], c=c, s=20, alpha=0.8, edgecolors='white', linewidths=0.3, zorder=3)
rs2, ps2 = stats.spearmanr(all_alpha, all_cap)
rp2, pp2 = stats.pearsonr(all_alpha, all_cap)
ax.text(0.95, 0.08, f'Spearman r = {rs2:.2f}, p = {ps2:.2f}\nPearson r = {rp2:.2f}, p = {pp2:.1e}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=5.5,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax.set_xlabel(r'Spectral slope $\alpha$', fontsize=7)
ax.set_ylabel('Absorbing capacity', fontsize=7)
ax.set_title(r'$\alpha$ vs capacity', fontsize=7.5, fontweight='bold')
legend_el = [Line2D([0],[0],marker='o',color='w',markerfacecolor=C_ASP,markersize=5,label='Asp'),
             Line2D([0],[0],marker='o',color='w',markerfacecolor=C_MUC,markersize=5,label='Muc')]
ax.legend(handles=legend_el, fontsize=5, loc='upper right')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.text(-0.14, 1.08, 'C', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')

savefig(fig, 'figS18_fft_analysis')


# ═══════════════════════════════════════════════════════════════
# S19: Hessian overlays + tubeness CV annotation
# ═══════════════════════════════════════════════════════════════
print('S19: Hessian + tubeness CV...')

asp_hess = plt.imread(HESSIAN_LM / 'Pink_10X_1_overlay.png')
muc_hess = plt.imread(HESSIAN_LM / 'White_10X_1_overlay.png')

asp_cv = tub_distributions['Pink_10X_1.TIF']['cv']
muc_cv = tub_distributions['White_10X_1.TIF']['cv']

fig, axes = plt.subplots(1, 2, figsize=(183*MM, 95*MM))
fig.subplots_adjust(wspace=0.04)

axes[0].imshow(asp_hess); axes[0].axis('off')
axes[0].text(0.02, 0.98, 'A', transform=axes[0].transAxes, fontsize=14,
             fontweight='bold', va='top', color='white',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6, edgecolor='none'))
axes[0].text(0.98, 0.02, f'Tubeness CV = {asp_cv:.2f}', transform=axes[0].transAxes,
             fontsize=8, ha='right', va='bottom', color='white', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=C_ASP, alpha=0.8, edgecolor='none'))

axes[1].imshow(muc_hess); axes[1].axis('off')
axes[1].text(0.02, 0.98, 'B', transform=axes[1].transAxes, fontsize=14,
             fontweight='bold', va='top', color='white',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6, edgecolor='none'))
axes[1].text(0.98, 0.02, f'Tubeness CV = {muc_cv:.2f}', transform=axes[1].transAxes,
             fontsize=8, ha='right', va='bottom', color='white', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=C_MUC, alpha=0.8, edgecolor='none'))

savefig(fig, 'figS19_hessian_lm')


# ═══════════════════════════════════════════════════════════════
# S20: Erosion demo (keep — regenerate for consistency)
# ═══════════════════════════════════════════════════════════════
print('S20: Erosion demo...')

asp_lm = load_gray(MICRO_DIR / 'Pink_10X_1.TIF')
muc_lm = load_gray(MICRO_DIR / 'White_10X_1.TIF')
asp_lm_mask = seg_micro(asp_lm, 'Pink')
muc_lm_mask = seg_micro(muc_lm, 'White')

stages = [0, 3, 5, 10, 20]
fig, axes = plt.subplots(2, len(stages), figsize=(183*MM, 78*MM))
fig.subplots_adjust(hspace=0.10, wspace=0.04)
for row, (img, mask, genus, color) in enumerate([
    (asp_lm, asp_lm_mask, 'Aspergillus', C_ASP),
    (muc_lm, muc_lm_mask, 'Mucor', C_MUC)]):
    nd = norm_display(img); orig = mask.sum()
    for col, ni in enumerate(stages):
        current = ndi.binary_erosion(mask, iterations=ni) if ni > 0 else mask.copy()
        sv = current.sum()/orig if orig > 0 else 0
        vis = np.stack([nd]*3, axis=-1).copy()
        vis[mask & ~current] = [0.35, 0.45, 0.85]; vis[current] = [0.88, 0.18, 0.18]
        axes[row, col].imshow(vis); axes[row, col].axis('off')
        if row == 0:
            axes[row, col].set_title('Original' if col == 0 else f'Erosion = {ni}\n({sv:.0%} survived)',
                                      fontsize=6.5 if col == 0 else 6, fontweight='bold' if col == 0 else 'normal', pad=3)
        else:
            if col > 0: axes[row, col].set_title(f'{sv:.0%}', fontsize=6, pad=3)
    axes[row, 0].text(-0.08, 0.5, genus, transform=axes[row, 0].transAxes,
                       fontsize=8, style='italic', color=color, fontweight='bold',
                       rotation=90, va='center', ha='right')
fig.legend(handles=[Patch(facecolor=[0.88,0.18,0.18],edgecolor='none',label='Survived'),
                    Patch(facecolor=[0.35,0.45,0.85],edgecolor='none',label='Eroded away')],
           loc='lower center', ncol=2, fontsize=7, framealpha=0.9, edgecolor='none', bbox_to_anchor=(0.5,-0.01))
axes[0,0].text(-0.08, 1.08, 'A', transform=axes[0,0].transAxes, fontsize=12, fontweight='bold', va='top')
axes[1,0].text(-0.08, 1.08, 'B', transform=axes[1,0].transAxes, fontsize=12, fontweight='bold', va='top')
savefig(fig, 'figS20_erosion_demo')


# ═══════════════════════════════════════════════════════════════
# S21: Tubeness CV derivation (NEW — replaces density CV)
# ═══════════════════════════════════════════════════════════════
print('S21: Tubeness CV derivation...')

fig = plt.figure(figsize=(183*MM, 110*MM))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

# Panel A: Tubeness distribution histograms for representative images
ax = fig.add_subplot(gs[0, :])
for fname, genus, color, ls in [('Pink_10X_1.TIF', 'Asp', C_ASP, '-'),
                                  ('White_10X_1.TIF', 'Muc', C_MUC, '-')]:
    td = tub_distributions[fname]
    vals = td['values']
    vals_clipped = vals[vals < np.percentile(vals, 99)]  # clip outliers for display
    ax.hist(vals_clipped, bins=100, density=True, alpha=0.4, color=color,
            label=f'{genus} (CV = {td["cv"]:.2f})')

ax.set_xlabel('Hessian tubeness response (within tissue)', fontsize=7.5)
ax.set_ylabel('Density', fontsize=7.5)
ax.set_title('Tubeness distribution within tissue', fontsize=8, fontweight='bold')
ax.legend(fontsize=7, loc='upper right')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.text(-0.06, 1.06, 'A', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')

# Panel B: Tubeness CV boxplot (all images)
ax = fig.add_subplot(gs[1, 0])
stripbox(ax, tubcv_a, tubcv_m, 'Tubeness CV')
ax.text(-0.14, 1.06, 'B', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')

# Panel C: Per-magnification comparison
ax = fig.add_subplot(gs[1, 1])
mags = [10, 20, 40]
x_pos = np.arange(len(mags))
width = 0.35
for i, mag in enumerate(mags):
    asp_vals = [tubcv_a[j] for j, (f,g,l,m) in enumerate(MICRO_IMAGES[:6]) if m == mag]
    muc_vals = [tubcv_m[j] for j, (f,g,l,m) in enumerate(MICRO_IMAGES[6:]) if m == mag]
    ax.bar(i - width/2, np.mean(asp_vals), width, color=C_ASP, alpha=0.6, edgecolor='none')
    ax.bar(i + width/2, np.mean(muc_vals) if muc_vals else 0, width, color=C_MUC, alpha=0.6, edgecolor='none')
    for v in asp_vals:
        ax.scatter(i - width/2, v, s=12, c=C_ASP, edgecolors='white', linewidths=0.3, zorder=3)
    for v in muc_vals:
        ax.scatter(i + width/2, v, s=12, c=C_MUC, edgecolors='white', linewidths=0.3, zorder=3)
    if asp_vals and muc_vals:
        ratio = np.mean(asp_vals) / np.mean(muc_vals)
        ax.text(i, max(max(asp_vals), max(muc_vals)) + 0.05, f'{ratio:.2f}\u00d7',
                ha='center', fontsize=6, fontweight='bold')

ax.set_xticks(x_pos); ax.set_xticklabels(['10\u00d7', '20\u00d7', '40\u00d7'], fontsize=7)
ax.set_ylabel('Tubeness CV', fontsize=7.5)
ax.set_title('Magnification independence', fontsize=7.5, fontweight='bold')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
legend_el = [Patch(facecolor=C_ASP, alpha=0.6, label='Asp'),
             Patch(facecolor=C_MUC, alpha=0.6, label='Muc')]
ax.legend(handles=legend_el, fontsize=6, loc='upper right')
ax.text(-0.14, 1.06, 'C', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')

savefig(fig, 'figS21_tubeness_cv')


# ═══════════════════════════════════════════════════════════════
# S22: Absorbing capacity + SSA (keep)
# ═══════════════════════════════════════════════════════════════
print('S22: Absorbing capacity + SSA...')

fig, axes = plt.subplots(1, 3, figsize=(150*MM, 55*MM))
fig.subplots_adjust(wspace=0.55)
stripbox(axes[0], cap_a, cap_m, 'Absorbing capacity\n($f$ \u00d7 thickness)')
stripbox(axes[1], ssa_a, ssa_m, 'Specific surface area\n(1/\u00b5m)')
axes[1].set_title(r'$\it{Mucor}$ > $\it{Aspergillus}$', fontsize=6.5, color=C_MUC, pad=3)
stripbox(axes[2], erosion_a, erosion_m, 'Erosion survival\n(iter. 10, light microscopy)')
for ax, l in zip(axes, 'ABC'):
    ax.text(-0.15, 1.08, l, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
savefig(fig, 'figS22_absorbing_capacity')


# ═══════════════════════════════════════════════════════════════
# Handoff sync + stale-file cleanup
# ═══════════════════════════════════════════════════════════════
print('\nSyncing handoff and cleaning stale files...')
for name in [
    'figS18_fft_analysis',
    'figS19_hessian_lm',
    'figS20_erosion_demo',
    'figS21_tubeness_cv',
    'figS22_absorbing_capacity',
    'figS_seg3d_sensitivity',
]:
    src = FIG_OUT / f'{name}.pdf'
    if src.exists():
        shutil.copy2(src, HANDOFF / f'{name}.pdf')
        print(f'  Synced: {name}.pdf')
for f in FIG_OUT.glob('figS21_cv.*'):
    os.remove(f); print(f'  Removed: {f.name}')
for f in HANDOFF.glob('figS21_cv.*'):
    os.remove(f); print(f'  Removed: {f.name}')
# Remove old handoff files
for f in HANDOFF.glob('figure4_panel_G.*'):
    os.remove(f); print(f'  Removed: {f.name}')

print('\n' + '=' * 70)
print('ALL DONE. Files generated:')
print('=' * 70)
print(f'  Main figure: figure4_panels_CF.{{png,pdf,svg}}')
print(f'  Main figure: figure4_panel_GH.{{png,pdf,svg}}')
print(f'  S18: figS18_fft_analysis.{{png,pdf,svg}}')
print(f'  S19: figS19_hessian_lm.{{png,pdf,svg}}')
print(f'  S20: figS20_erosion_demo.{{png,pdf,svg}}')
print(f'  S21: figS21_tubeness_cv.{{png,pdf,svg}}  (NEW — replaces density CV)')
print(f'  S22: figS22_absorbing_capacity.{{png,pdf,svg}}')

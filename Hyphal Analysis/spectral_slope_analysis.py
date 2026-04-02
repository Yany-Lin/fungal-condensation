#!/usr/bin/env python3
"""Spectral slope analysis comparing Aspergillus and Mucor hyphal surface texture."""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

THIS_DIR    = Path(__file__).parent
OUTPUT_DIR  = THIS_DIR  # stats/CSV stay here
FIGURE_DIR  = THIS_DIR.parent / 'FigureFungi' / 'output'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
TS = 7.0; LS = 8.5; PL = 12.0
LW = 0.6

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         TS,
    'axes.linewidth':    LW,
    'xtick.major.width': LW, 'ytick.major.width': LW,
    'xtick.major.size':  3.5, 'ytick.major.size':  3.5,
    'xtick.direction':   'out', 'ytick.direction':  'out',
    'svg.fonttype':      'none',
})

C_ASP  = '#4CAF50'
C_MUC  = '#757575'

DELTA_ASP = np.array([279.5, 316.0, 297.8, 285.6, 311.7])
DELTA_MUC = np.array([198.5, 126.2, 120.0, 131.6, 123.0])

FREQ_LO = 0.01
FREQ_HI = 0.45


def load_spectra(genus):
    df = pd.read_csv(THIS_DIR / f'{genus}_full_spectra.csv')
    freq = df['frequency'].values
    transect_cols = [c for c in df.columns if c != 'frequency']
    power = df[transect_cols].values  # (n_freq, n_transects)
    return freq, power, transect_cols


def compute_slope(freq, power_col):
    valid = np.isfinite(power_col) & (power_col > 0) & \
            (freq >= FREQ_LO) & (freq <= FREQ_HI)
    if valid.sum() < 10:
        return np.nan, np.nan, 0

    lf = np.log10(freq[valid])
    lp = np.log10(power_col[valid])
    sl, ic, r, p, se = stats.linregress(lf, lp)
    return sl, r**2, int(valid.sum())


def main():
    freq_a, pow_a, cols_a = load_spectra('Aspergillus')
    freq_m, pow_m, cols_m = load_spectra('Mucor')
    n_asp = pow_a.shape[1]
    n_muc = pow_m.shape[1]
    print(f"Aspergillus: {n_asp} transects, {len(freq_a)} frequency bins")
    print(f"Mucor:       {n_muc} transects, {len(freq_m)} frequency bins")

    slopes_asp = []
    r2_asp = []
    for j in range(n_asp):
        sl, r2, n = compute_slope(freq_a, pow_a[:, j])
        slopes_asp.append(sl)
        r2_asp.append(r2)

    slopes_muc = []
    r2_muc = []
    for j in range(n_muc):
        sl, r2, n = compute_slope(freq_m, pow_m[:, j])
        slopes_muc.append(sl)
        r2_muc.append(r2)

    slopes_asp = np.array(slopes_asp)
    slopes_muc = np.array(slopes_muc)
    r2_asp = np.array(r2_asp)
    r2_muc = np.array(r2_muc)

    valid_a = np.isfinite(slopes_asp)
    valid_m = np.isfinite(slopes_muc)
    sa = slopes_asp[valid_a]
    sm = slopes_muc[valid_m]
    print(f"\nValid slopes: Aspergillus {len(sa)}/{n_asp}, Mucor {len(sm)}/{n_muc}")
    print(f"Total transects: {len(sa) + len(sm)}")

    t_stat, t_p = stats.ttest_ind(sa, sm, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(sa, sm, alternative='two-sided')
    n1, n2 = len(sa), len(sm)
    pooled_sd = np.sqrt(((n1-1)*sa.std(ddof=1)**2 + (n2-1)*sm.std(ddof=1)**2) / (n1+n2-2))
    cohens_d = (sa.mean() - sm.mean()) / pooled_sd
    r_eff = 1 - (2*u_stat) / (n1*n2)

    report = []
    report.append("=" * 65)
    report.append("SPECTRAL SLOPE ANALYSIS: Aspergillus vs Mucor")
    report.append("=" * 65)
    report.append(f"\nFrequency range for slope fit: [{FREQ_LO}, {FREQ_HI}] cycles/px")
    report.append(f"\nAspergillus (Green fungus — images from Lab/Green/):")
    report.append(f"  n = {n1} transects")
    report.append(f"  slope = {sa.mean():.3f} ± {sa.std(ddof=1):.3f} (mean ± SD)")
    report.append(f"  SEM = {sa.std(ddof=1)/np.sqrt(n1):.4f}")
    report.append(f"  median = {np.median(sa):.3f}")
    report.append(f"  range = [{sa.min():.3f}, {sa.max():.3f}]")
    report.append(f"  mean R² of log-log fit = {r2_asp[valid_a].mean():.3f}")
    report.append(f"\nMucor (White fungus — images from VMS_JPG/W2/):")
    report.append(f"  n = {n2} transects")
    report.append(f"  slope = {sm.mean():.3f} ± {sm.std(ddof=1):.3f} (mean ± SD)")
    report.append(f"  SEM = {sm.std(ddof=1)/np.sqrt(n2):.4f}")
    report.append(f"  median = {np.median(sm):.3f}")
    report.append(f"  range = [{sm.min():.3f}, {sm.max():.3f}]")
    report.append(f"  mean R² of log-log fit = {r2_muc[valid_m].mean():.3f}")
    report.append("")
    report.append(f"Welch's t-test:  t = {t_stat:.3f}, p = {t_p:.3e}")
    report.append(f"Mann-Whitney U:  U = {u_stat:.1f}, p = {u_p:.3e}")
    report.append(f"Cohen's d = {cohens_d:.3f}")
    report.append(f"Effect size r = {r_eff:.3f}")
    report.append(f"\nShapiro-Wilk:")
    sw_a = stats.shapiro(sa)
    sw_m = stats.shapiro(sm)
    report.append(f"  Aspergillus: W = {sw_a.statistic:.4f}, p = {sw_a.pvalue:.4f}")
    report.append(f"  Mucor:       W = {sw_m.statistic:.4f}, p = {sw_m.pvalue:.4f}")

    # Levene's test for equal variances
    lev_stat, lev_p = stats.levene(sa, sm)
    report.append(f"\nLevene's test: F = {lev_stat:.3f}, p = {lev_p:.4f}")
    ks_stat, ks_p = stats.ks_2samp(sa, sm)
    report.append(f"Kolmogorov-Smirnov: D = {ks_stat:.3f}, p = {ks_p:.3e}")
    report.append(f"Green (Aspergillus): δ = {DELTA_ASP.mean():.1f} ± {DELTA_ASP.std(ddof=1)/np.sqrt(5):.1f} µm (mean ± SEM, n=5)")
    report.append(f"White (Mucor):       δ = {DELTA_MUC.mean():.1f} ± {DELTA_MUC.std(ddof=1)/np.sqrt(5):.1f} µm (mean ± SEM, n=5)")

    report_txt = '\n'.join(report)
    print(report_txt)

    with open(OUTPUT_DIR / 'spectral_slope_stats.txt', 'w') as f:
        f.write(report_txt)

    rows = []
    for j, col in enumerate(cols_a):
        if valid_a[j]:
            rows.append({'transect_id': col, 'genus': 'Aspergillus',
                         'slope': slopes_asp[j], 'r2': r2_asp[j]})
    for j, col in enumerate(cols_m):
        if valid_m[j]:
            rows.append({'transect_id': col, 'genus': 'Mucor',
                         'slope': slopes_muc[j], 'r2': r2_muc[j]})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'spectral_slope_results.csv', index=False)

    fig, (ax_i, ax_j) = plt.subplots(1, 2, figsize=(120*MM, 60*MM))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.18, wspace=0.45)

    bp = ax_i.boxplot([sa, sm], positions=[1, 2], widths=0.5,
                       patch_artist=True, showfliers=False,
                       medianprops=dict(color='white', lw=1.2),
                       whiskerprops=dict(lw=0.8),
                       capprops=dict(lw=0.8))
    bp['boxes'][0].set_facecolor(C_ASP)
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(C_MUC)
    bp['boxes'][1].set_alpha(0.6)

    rng = np.random.default_rng(42)
    jitter_a = rng.uniform(-0.15, 0.15, len(sa))
    jitter_m = rng.uniform(-0.15, 0.15, len(sm))
    ax_i.scatter(1 + jitter_a, sa, s=8, c=C_ASP, alpha=0.5, edgecolors='none', zorder=3)
    ax_i.scatter(2 + jitter_m, sm, s=8, c=C_MUC, alpha=0.5, edgecolors='none', zorder=3)

    ax_i.set_xticks([1, 2])
    ax_i.set_xticklabels(['Aspergillus', 'Mucor'], fontsize=TS, style='italic')
    ax_i.set_ylabel('Spectral slope $\\alpha$', fontsize=LS)

    y_max = max(sa.max(), sm.max()) + 0.15
    ax_i.plot([1, 1, 2, 2], [y_max, y_max+0.08, y_max+0.08, y_max], color='black', lw=0.8)
    ax_i.text(1.5, y_max + 0.10, f'p = {t_p:.1e}', ha='center', va='bottom', fontsize=TS-1)

    for sp in ['top', 'right']:
        ax_i.spines[sp].set_visible(False)
    ax_i.tick_params(labelsize=TS)
    ax_i.text(-0.18, 1.08, 'I', transform=ax_i.transAxes,
              fontsize=PL, fontweight='bold', va='top')

    slope_means = [sa.mean(), sm.mean()]
    slope_sds   = [sa.std(ddof=1), sm.std(ddof=1)]
    delta_means = [DELTA_ASP.mean(), DELTA_MUC.mean()]
    delta_sds   = [DELTA_ASP.std(ddof=1), DELTA_MUC.std(ddof=1)]

    ax_j.errorbar(slope_means[0], delta_means[0],
                  xerr=slope_sds[0], yerr=delta_sds[0],
                  fmt='o', color=C_ASP, markersize=7,
                  markeredgecolor='white', markeredgewidth=0.5,
                  capsize=3, capthick=0.8, elinewidth=0.8,
                  label='Aspergillus', zorder=3)
    ax_j.errorbar(slope_means[1], delta_means[1],
                  xerr=slope_sds[1], yerr=delta_sds[1],
                  fmt='o', color=C_MUC, markersize=7,
                  markeredgecolor='white', markeredgewidth=0.5,
                  capsize=3, capthick=0.8, elinewidth=0.8,
                  label='Mucor', zorder=3)
    ax_j.plot(slope_means, delta_means, color='gray', ls='--', lw=0.7, alpha=0.5, zorder=1)

    ax_j.set_xlabel('Spectral slope $\\alpha$', fontsize=LS)
    ax_j.set_ylabel('δ (µm)', fontsize=LS)
    ax_j.legend(fontsize=TS-1, framealpha=0.9, loc='best',
                handletextpad=0.3, borderpad=0.3, prop={'style': 'italic'})

    for sp in ['top', 'right']:
        ax_j.spines[sp].set_visible(False)
    ax_j.tick_params(labelsize=TS)
    ax_j.text(-0.22, 1.08, 'J', transform=ax_j.transAxes,
              fontsize=PL, fontweight='bold', va='top')

    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(FIGURE_DIR / f'panels_IJ{ext}', **kw)
    plt.close(fig)
    print(f"\nSaved panels_IJ.* to {FIGURE_DIR}")
    print(f"Saved spectral_slope_results.csv")
    print(f"Saved spectral_slope_stats.txt")


if __name__ == '__main__':
    main()

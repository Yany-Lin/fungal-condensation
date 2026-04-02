#!/usr/bin/env python3
"""
Comprehensive statistical supplement for Nature Communications manuscript.
Covers: pairwise comparisons, linearity tests, robustness checks,
        influence diagnostics, power analysis, and formatted tables.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from itertools import combinations
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE   = Path(__file__).resolve().parents[2]
OUTDIR = BASE / "additions" / "3_bootstrap_CIs"
OUTDIR.mkdir(parents=True, exist_ok=True)

uni  = pd.read_csv(BASE / "FigureTable/output/universal_metrics.csv")
hg   = pd.read_csv(BASE / "FigureHGAggregate/output/hydrogel_metrics.csv")
fun  = pd.read_csv(BASE / "FigureFungi/output/fungi_metrics.csv")

# Subset for the 6 vapor-sink groups (exclude Leaf rows)
vs = uni[uni["system"].isin(["Hydrogel", "Fungi"])].copy()

GROUPS = ["Agar", "0.5:1", "1:1", "2:1", "Green", "White", "Black"]
METRICS = ["delta_um", "zone_metric", "dtau50_dr", "max_slope"]
N_BOOT = 10000
RNG = np.random.default_rng(42)

def cliff_delta(x, y):
    """Cliff's delta effect size."""
    nx, ny = len(x), len(y)
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (nx * ny)

def cliff_interp(d):
    ad = abs(d)
    if ad < 0.147:   return "negligible"
    elif ad < 0.33:  return "small"
    elif ad < 0.474: return "medium"
    else:            return "large"

def bootstrap_ci(data, stat_fn=np.median, n_boot=N_BOOT, ci=0.95):
    boots = np.array([stat_fn(RNG.choice(data, size=len(data), replace=True))
                      for _ in range(n_boot)])
    lo = np.percentile(boots, 100 * (1 - ci) / 2)
    hi = np.percentile(boots, 100 * (1 + ci) / 2)
    return lo, hi

def bootstrap_r2_ci(x, y, n_boot=N_BOOT):
    """Bootstrap CI for R²."""
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = RNG.choice(n, size=n, replace=True)
        r = np.corrcoef(x[idx], y[idx])[0, 1]
        vals.append(r**2)
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

def theil_sen(x, y):
    res = stats.theilslopes(y, x)
    return res.slope, res.intercept, res.low_slope, res.high_slope

def cooks_distance(x, y):
    """Cook's D for simple OLS."""
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ b
    e = y - yhat
    p = 2
    n = len(y)
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(H)
    mse = np.sum(e**2) / (n - p)
    D = (e**2 / (p * mse)) * (h / (1 - h)**2)
    return D, 4.0 / n   # values, threshold

print("Section 1: Pairwise comparisons")

kw_rows = []
pw_rows = []

for met in METRICS:
    # Collect groups with valid data
    group_data = {}
    for g in GROUPS:
        vals = vs.loc[vs["group"] == g, met].dropna().values
        if len(vals) >= 2:
            group_data[g] = vals

    avail = list(group_data.keys())
    if len(avail) < 2:
        continue

    # Kruskal-Wallis
    arrays = [group_data[g] for g in avail]
    kw_stat, kw_p = stats.kruskal(*arrays)
    kw_rows.append({"metric": met, "groups_tested": ", ".join(avail),
                     "KW_H": f"{kw_stat:.3f}", "KW_p": f"{kw_p:.4g}"})

    # Pairwise Mann-Whitney with Bonferroni
    pairs = list(combinations(avail, 2))
    n_pairs = len(pairs)
    for g1, g2 in pairs:
        x, y = group_data[g1], group_data[g2]
        u_stat, u_p = stats.mannwhitneyu(x, y, alternative="two-sided")
        p_adj = min(u_p * n_pairs, 1.0)
        cd = cliff_delta(x, y)
        pw_rows.append({
            "metric": met,
            "group_A": g1, "median_A": f"{np.median(x):.4g}",
            "IQR_A": f"{np.percentile(x,25):.4g}–{np.percentile(x,75):.4g}",
            "group_B": g2, "median_B": f"{np.median(y):.4g}",
            "IQR_B": f"{np.percentile(y,25):.4g}–{np.percentile(y,75):.4g}",
            "U": f"{u_stat:.1f}", "p_raw": f"{u_p:.4g}",
            "p_bonf": f"{p_adj:.4g}",
            "Cliff_d": f"{cd:.3f}",
            "effect_size": cliff_interp(cd),
        })

kw_df = pd.DataFrame(kw_rows)
pw_df = pd.DataFrame(pw_rows)

print("Section 2: Linearity tests")

hg_metrics = ["delta_um", "max_slope", "r0_um", "alpha", "y_near", "y_far",
              "transition_width_um"]
lin_rows = []

x_aw = hg["one_minus_aw"].values

for met in hg_metrics:
    y_vals = hg[met].dropna()
    mask = hg[met].notna().values
    if mask.sum() < 5:
        continue
    x = x_aw[mask]
    y = y_vals.values

    # Linear fit
    sl_lin, ic_lin, r_lin, p_lin, se_lin = stats.linregress(x, y)
    ss_res_lin = np.sum((y - (sl_lin * x + ic_lin))**2)

    # Quadratic fit
    c2, c1, c0 = np.polyfit(x, y, 2)
    yhat_q = c2 * x**2 + c1 * x + c0
    ss_res_quad = np.sum((y - yhat_q)**2)

    # F-test for quadratic term
    n = len(x)
    df1 = 1  # extra parameter
    df2 = n - 3
    if df2 > 0 and ss_res_quad > 0:
        F = ((ss_res_lin - ss_res_quad) / df1) / (ss_res_quad / df2)
        p_f = 1 - stats.f.cdf(F, df1, df2)
    else:
        F, p_f = np.nan, np.nan

    lin_rows.append({
        "metric": met,
        "linear_R2": f"{r_lin**2:.4f}",
        "linear_p": f"{p_lin:.4g}",
        "quad_coeff": f"{c2:.4g}",
        "F_for_curvature": f"{F:.3f}" if np.isfinite(F) else "—",
        "p_curvature": f"{p_f:.4g}" if np.isfinite(p_f) else "—",
        "curvature_sig": "Yes" if (np.isfinite(p_f) and p_f < 0.05) else "No",
    })

lin_df = pd.DataFrame(lin_rows)

print("Section 3: Robustness checks")

# 3a. Regression robustness for hydrogel metrics
rob_rows = []
cook_records = []  # for influence plot

for met in hg_metrics:
    mask = hg[met].notna().values
    if mask.sum() < 5:
        continue
    x = x_aw[mask]
    y = hg.loc[mask, met].values
    ids = hg.loc[mask, "trial_id"].values

    # OLS
    sl, ic, r_ols, p_ols, se_sl = stats.linregress(x, y)
    # Theil-Sen
    ts_sl, ts_ic, ts_lo, ts_hi = theil_sen(x, y)
    pct_diff = 100 * (ts_sl - sl) / abs(sl) if sl != 0 else np.nan
    # Spearman
    rho, p_rho = stats.spearmanr(x, y)
    # R² bootstrap CI
    r2_lo, r2_hi = bootstrap_r2_ci(x, y)
    # Cook's distance
    D, thresh = cooks_distance(x, y)

    rob_rows.append({
        "metric": met,
        "OLS_slope": f"{sl:.4g}",
        "OLS_slope_SE": f"{se_sl:.4g}",
        "OLS_slope_95CI": f"[{sl - 1.96*se_sl:.4g}, {sl + 1.96*se_sl:.4g}]",
        "OLS_intercept": f"{ic:.4g}",
        "TheilSen_slope": f"{ts_sl:.4g}",
        "TheilSen_95CI": f"[{ts_lo:.4g}, {ts_hi:.4g}]",
        "slope_pct_diff": f"{pct_diff:.1f}%",
        "Pearson_r": f"{r_ols:.4f}",
        "Pearson_p": f"{p_ols:.4g}",
        "R2": f"{r_ols**2:.4f}",
        "R2_boot_95CI": f"[{r2_lo:.4f}, {r2_hi:.4f}]",
        "Spearman_rho": f"{rho:.4f}",
        "Spearman_p": f"{p_rho:.4g}",
    })

    for i, tid in enumerate(ids):
        cook_records.append({
            "metric": met, "trial_id": tid,
            "x": x[i], "y": y[i],
            "Cook_D": D[i], "threshold": thresh,
            "flagged": D[i] > thresh
        })

rob_df = pd.DataFrame(rob_rows)
cook_df = pd.DataFrame(cook_records)

# 3b. Same robustness for the universal vapor-sink metrics
# Key regressions from universal_metrics: zone_metric, S, dtau50_dr, delta_um vs 1-aw (hydrogels only)
uni_hg = vs[vs["system"] == "Hydrogel"].copy()
uni_hg["one_minus_aw"] = 1 - uni_hg["a_w"]

uni_rob_rows = []
for met in ["delta_um", "zone_metric", "dtau50_dr", "max_slope"]:
    mask = uni_hg[met].notna().values
    if mask.sum() < 5:
        continue
    x = uni_hg.loc[mask, "one_minus_aw"].values
    y = uni_hg.loc[mask, met].values
    ids = uni_hg.loc[mask, "trial_id"].values

    sl, ic, r_ols, p_ols, se_sl = stats.linregress(x, y)
    ts_sl, ts_ic, ts_lo, ts_hi = theil_sen(x, y)
    pct_diff = 100 * (ts_sl - sl) / abs(sl) if sl != 0 else np.nan
    rho, p_rho = stats.spearmanr(x, y)
    r2_lo, r2_hi = bootstrap_r2_ci(x, y)
    D, thresh = cooks_distance(x, y)

    # Intercept CI
    n = len(x)
    xbar = np.mean(x)
    ssx = np.sum((x - xbar)**2)
    mse = np.sum((y - (sl*x + ic))**2) / (n - 2)
    se_ic = np.sqrt(mse * (1/n + xbar**2/ssx))

    uni_rob_rows.append({
        "regression": f"{met} vs (1−aᵥ), hydrogels",
        "n": n,
        "OLS_slope": f"{sl:.4g}",
        "slope_SE": f"{se_sl:.4g}",
        "slope_95CI": f"[{sl - 1.96*se_sl:.4g}, {sl + 1.96*se_sl:.4g}]",
        "intercept": f"{ic:.4g}",
        "intercept_SE": f"{se_ic:.4g}",
        "intercept_95CI": f"[{ic - 1.96*se_ic:.4g}, {ic + 1.96*se_ic:.4g}]",
        "R2": f"{r_ols**2:.4f}",
        "R2_boot_95CI": f"[{r2_lo:.4f}, {r2_hi:.4f}]",
        "p_value": f"{p_ols:.4g}",
        "TheilSen_slope": f"{ts_sl:.4g}",
        "TheilSen_95CI": f"[{ts_lo:.4g}, {ts_hi:.4g}]",
        "slope_pct_diff_OLS_vs_TS": f"{pct_diff:.1f}%",
        "Spearman_rho": f"{rho:.4f}",
        "Spearman_p": f"{p_rho:.4g}",
    })

    for i, tid in enumerate(ids):
        cook_records.append({
            "metric": f"{met}_universal", "trial_id": tid,
            "x": x[i], "y": y[i],
            "Cook_D": D[i], "threshold": thresh,
            "flagged": D[i] > thresh
        })

uni_rob_df = pd.DataFrame(uni_rob_rows)
cook_df = pd.DataFrame(cook_records)

print("Section 4: Leaf comparison")

leaf = uni[uni["system"] == "Leaf"].copy()
healthy = leaf.loc[leaf["group"] == "Healthy", "dR_dr_um_per_mm"].dropna().values
diseased = leaf.loc[leaf["group"] == "Diseased", "dR_dr_um_per_mm"].dropna().values
# Also pearson_r
healthy_r = leaf.loc[leaf["group"] == "Healthy", "pearson_r"].dropna().values
diseased_r = leaf.loc[leaf["group"] == "Diseased", "pearson_r"].dropna().values

leaf_rows = []
for label, h, d in [("dR_dr (µm/mm)", healthy, diseased),
                     ("Pearson r", healthy_r, diseased_r)]:
    if len(h) >= 2 and len(d) >= 2:
        u_stat, u_p = stats.mannwhitneyu(h, d, alternative="two-sided")
        cd = cliff_delta(h, d)
        # Welch t-test as supplement
        t_stat, t_p = stats.ttest_ind(h, d, equal_var=False)
        leaf_rows.append({
            "metric": label,
            "Healthy_median": f"{np.median(h):.4g}",
            "Healthy_IQR": f"{np.percentile(h,25):.4g}–{np.percentile(h,75):.4g}",
            "Diseased_median": f"{np.median(d):.4g}",
            "Diseased_IQR": f"{np.percentile(d,25):.4g}–{np.percentile(d,75):.4g}",
            "MW_U": f"{u_stat:.1f}", "MW_p": f"{u_p:.4g}",
            "Welch_t": f"{t_stat:.3f}", "Welch_p": f"{t_p:.4g}",
            "Cliff_d": f"{cd:.3f}", "effect": cliff_interp(cd),
        })

leaf_df = pd.DataFrame(leaf_rows)

print("Section 5: Power analysis")

def min_detectable_f2(n, alpha=0.05, power=0.80, p=1):
    """
    Minimum detectable Cohen's f² for a single-predictor regression.
    Uses F-distribution inversion: f² = F_crit * (p / (n - p - 1))
    at the given power level.
    """
    df1 = p
    df2 = n - p - 1
    if df2 < 1:
        return np.nan
    # Critical F for significance
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    # NCP for desired power
    from scipy.optimize import brentq
    def power_eq(lam):
        return stats.ncf.sf(f_crit, df1, df2, lam) - power
    try:
        ncp = brentq(power_eq, 0, 200)
    except:
        return np.nan
    f2 = ncp / n
    return f2

def f2_to_r2(f2):
    return f2 / (1 + f2)

power_rows = []
for n in [15, 30]:
    f2 = min_detectable_f2(n)
    r2_min = f2_to_r2(f2)
    power_rows.append({
        "analysis": f"Regression (single predictor), n={n}",
        "min_detectable_f2": f"{f2:.4f}",
        "min_detectable_R2": f"{r2_min:.4f}",
        "alpha": 0.05, "power": 0.80,
    })

# Sample size for leaf comparison
# Observed effect size (Cohen's d)
if len(healthy) >= 2 and len(diseased) >= 2:
    pooled_std = np.sqrt(((len(healthy)-1)*np.std(healthy,ddof=1)**2 +
                          (len(diseased)-1)*np.std(diseased,ddof=1)**2) /
                         (len(healthy)+len(diseased)-2))
    cohens_d = abs(np.mean(healthy) - np.mean(diseased)) / pooled_std if pooled_std > 0 else np.nan

    # Required n per group for 80% power (two-sample t-test approximation)
    def power_ttest(n_per, d, alpha=0.05):
        df = 2*n_per - 2
        t_crit = stats.t.ppf(1 - alpha/2, df)
        ncp = d * np.sqrt(n_per / 2)
        return 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)

    # Brute-force search for required n
    n_req = "Could not determine"
    for n_try in range(3, 5001):
        if power_ttest(n_try, cohens_d) >= 0.80:
            n_req = n_try
            break

    power_rows.append({
        "analysis": f"Healthy vs Diseased leaf (observed Cohen's d = {cohens_d:.2f})",
        "min_detectable_f2": "—",
        "min_detectable_R2": "—",
        "alpha": 0.05,
        "power": f"n_required_per_group = {n_req}",
    })

power_df = pd.DataFrame(power_rows)

print("Section 6: Building manuscript table")

# Combine: group summary stats + regression stats
summary_rows = []

# Group summary for each metric
for met in METRICS:
    for g in GROUPS:
        vals = vs.loc[vs["group"] == g, met].dropna().values
        if len(vals) == 0:
            continue
        med = np.median(vals)
        q1, q3 = np.percentile(vals, [25, 75])
        summary_rows.append({
            "section": "Group summary",
            "metric": met,
            "group": g,
            "statistic": f"median = {med:.4g}, IQR = [{q1:.4g}, {q3:.4g}]",
        })

# Regression rows
for row in uni_rob_rows:
    summary_rows.append({
        "section": "Regression",
        "metric": row["regression"],
        "group": "—",
        "statistic": (f"slope = {row['OLS_slope']} ± {row['slope_SE']} "
                      f"{row['slope_95CI']}, "
                      f"R² = {row['R2']} {row['R2_boot_95CI']}, "
                      f"p = {row['p_value']}, "
                      f"ρ = {row['Spearman_rho']}"),
    })

summary_table = pd.DataFrame(summary_rows)

print("Saving CSVs")

# Master CSV with multiple sheets equivalent – use one big CSV with section markers
all_sections = []

def add_section(df, title):
    header = pd.DataFrame([{"": f"═══ {title} ═══"}])
    blank = pd.DataFrame([{}])
    all_sections.append(header)
    all_sections.append(df)
    all_sections.append(blank)

add_section(kw_df, "KRUSKAL-WALLIS TESTS")
add_section(pw_df, "PAIRWISE MANN-WHITNEY U (Bonferroni-corrected)")
add_section(lin_df, "LINEARITY TESTS (F-test for curvature)")
add_section(rob_df, "ROBUSTNESS: Hydrogel raw regressions (OLS vs Theil-Sen)")
add_section(uni_rob_df, "ROBUSTNESS: Universal metrics regressions")
add_section(leaf_df, "LEAF: Healthy vs Diseased")
add_section(power_df, "POWER ANALYSIS")
add_section(summary_table, "COMPLETE MANUSCRIPT STATISTICS")

# Flat CSV
master = pd.concat(all_sections, ignore_index=True)
master.to_csv(OUTDIR / "comprehensive_stats_table.csv", index=False)

# Also save individual tables for easy parsing
pw_df.to_csv(OUTDIR / "pairwise_comparisons.csv", index=False)
cook_df.to_csv(OUTDIR / "cook_distances.csv", index=False)

print("Saving LaTeX")

with open(OUTDIR / "comprehensive_stats_table.tex", "w") as f:
    f.write("% Auto-generated statistical supplement\n")
    f.write("% ════════════════════════════════════════════\n\n")

    f.write("\\begin{table}[htbp]\n\\centering\n")
    f.write("\\caption{Kruskal--Wallis tests across six vapor-sink groups.}\n")
    f.write("\\label{tab:kw}\n")
    f.write("\\begin{tabular}{lllr}\n\\toprule\n")
    f.write("Metric & Groups tested & $H$ & $p$ \\\\\n\\midrule\n")
    for _, r in kw_df.iterrows():
        f.write(f"{r['metric']} & {r['groups_tested']} & {r['KW_H']} & {r['KW_p']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    f.write("\\begin{longtable}{ll rr rr rrr l}\n")
    f.write("\\caption{Pairwise Mann--Whitney $U$ tests with Bonferroni correction "
            "and Cliff's $\\delta$ effect size.}\n")
    f.write("\\label{tab:pairwise}\\\\\n\\toprule\n")
    f.write("Metric & Pair & Med$_A$ & IQR$_A$ & Med$_B$ & IQR$_B$ & "
            "$U$ & $p_{\\text{adj}}$ & $\\delta$ & Effect \\\\\n\\midrule\n")
    f.write("\\endfirsthead\n")
    f.write("\\multicolumn{10}{c}{\\textit{(continued)}} \\\\\n\\toprule\n")
    f.write("Metric & Pair & Med$_A$ & IQR$_A$ & Med$_B$ & IQR$_B$ & "
            "$U$ & $p_{\\text{adj}}$ & $\\delta$ & Effect \\\\\n\\midrule\n")
    f.write("\\endhead\n")
    for _, r in pw_df.iterrows():
        pair = f"{r['group_A']} vs {r['group_B']}"
        f.write(f"{r['metric']} & {pair} & {r['median_A']} & "
                f"{r['IQR_A']} & {r['median_B']} & {r['IQR_B']} & "
                f"{r['U']} & {r['p_bonf']} & {r['Cliff_d']} & "
                f"{r['effect_size']} \\\\\n")
    f.write("\\bottomrule\n\\end{longtable}\n\n")

    f.write("\\begin{table}[htbp]\n\\centering\n")
    f.write("\\caption{Linearity tests: $F$-test for significance of quadratic term "
            "in hydrogel regressions (metric vs.\\ $1-a_w$).}\n")
    f.write("\\label{tab:linearity}\n")
    f.write("\\begin{tabular}{l rr rr l}\n\\toprule\n")
    f.write("Metric & Linear $R^2$ & Linear $p$ & Quad.~coeff & $F$ & $p_{\\text{curv}}$ & Sig? \\\\\n\\midrule\n")
    for _, r in lin_df.iterrows():
        f.write(f"{r['metric']} & {r['linear_R2']} & {r['linear_p']} & "
                f"{r['quad_coeff']} & {r['F_for_curvature']} & "
                f"{r['p_curvature']} & {r['curvature_sig']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
    f.write("\\caption{Regression robustness: OLS vs.\\ Theil--Sen slopes, "
            "Pearson $r$ vs.\\ Spearman $\\rho$, bootstrap $R^2$ CIs.}\n")
    f.write("\\label{tab:robustness}\n")
    f.write("\\begin{tabular}{l r l l r l rr}\n\\toprule\n")
    f.write("Regression & Slope $\\pm$ SE & Slope 95\\% CI & "
            "Theil--Sen [95\\% CI] & $R^2$ & $R^2$ boot CI & "
            "$\\rho$ & $p_\\rho$ \\\\\n\\midrule\n")
    for _, r in uni_rob_df.iterrows():
        f.write(f"{r['regression']} & {r['OLS_slope']} $\\pm$ {r['slope_SE']} & "
                f"{r['slope_95CI']} & {r['TheilSen_slope']} {r['TheilSen_95CI']} & "
                f"{r['R2']} & {r['R2_boot_95CI']} & "
                f"{r['Spearman_rho']} & {r['Spearman_p']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    if not leaf_df.empty:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Healthy vs.\\ Diseased leaf comparison.}\n")
        f.write("\\label{tab:leaf}\n")
        f.write("\\begin{tabular}{l rr rr rr rr l}\n\\toprule\n")
        f.write("Metric & Med$_H$ & IQR$_H$ & Med$_D$ & IQR$_D$ & "
                "$U$ & $p_U$ & $t$ & $p_t$ & $\\delta$ \\\\\n\\midrule\n")
        for _, r in leaf_df.iterrows():
            f.write(f"{r['metric']} & {r['Healthy_median']} & {r['Healthy_IQR']} & "
                    f"{r['Diseased_median']} & {r['Diseased_IQR']} & "
                    f"{r['MW_U']} & {r['MW_p']} & {r['Welch_t']} & {r['Welch_p']} & "
                    f"{r['Cliff_d']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    f.write("\\begin{table}[htbp]\n\\centering\n")
    f.write("\\caption{Power analysis results.}\n")
    f.write("\\label{tab:power}\n")
    f.write("\\begin{tabular}{l rr l}\n\\toprule\n")
    f.write("Analysis & Min $f^2$ & Min $R^2$ & Power / $n$ \\\\\n\\midrule\n")
    for _, r in power_df.iterrows():
        f.write(f"{r['analysis']} & {r['min_detectable_f2']} & "
                f"{r['min_detectable_R2']} & {r['power']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")


print("Generating influence diagnostics figure")

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

cook_df_plot = cook_df[~cook_df["metric"].str.contains("transition_width|y_near|y_far")]
unique_metrics = cook_df_plot["metric"].unique()
n_panels = len(unique_metrics)
ncols = min(3, n_panels)
nrows = int(np.ceil(n_panels / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.2*nrows))
if n_panels == 1:
    axes = np.array([axes])
axes = np.atleast_2d(axes)

for idx, met in enumerate(unique_metrics):
    ax = axes.flat[idx]
    sub = cook_df_plot[cook_df_plot["metric"] == met]
    colors = ["#d62728" if f else "#1f77b4" for f in sub["flagged"]]
    ax.bar(range(len(sub)), sub["Cook_D"].values, color=colors, edgecolor="none", width=0.7)
    ax.axhline(sub["threshold"].iloc[0], color="k", ls="--", lw=0.8, label=f"4/n = {sub['threshold'].iloc[0]:.3f}")
    ax.set_title(met.replace("_", " "), fontweight="bold")
    ax.set_ylabel("Cook's D")
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["trial_id"].values, rotation=45, ha="right", fontsize=6)
    ax.legend(fontsize=7, frameon=False)

# Remove empty panels
for idx in range(n_panels, nrows * ncols):
    axes.flat[idx].set_visible(False)

fig.tight_layout()
for ext in ["svg", "png", "pdf"]:
    fig.savefig(OUTDIR / f"influence_diagnostics.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Generating linearity tests figure")

lin_metrics = [m for m in hg_metrics if hg[m].notna().sum() >= 5]
n_panels = len(lin_metrics)
ncols = min(3, n_panels)
nrows = int(np.ceil(n_panels / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.5*nrows))
if n_panels == 1:
    axes = np.array([axes])
axes = np.atleast_2d(axes)

for idx, met in enumerate(lin_metrics):
    ax = axes.flat[idx]
    mask = hg[met].notna().values
    x = x_aw[mask]
    y = hg.loc[mask, met].values

    # Scatter
    ax.scatter(x, y, s=30, c="#1f77b4", edgecolor="white", linewidth=0.3, zorder=3)

    # Linear fit
    sl, ic, r_val, p_val, _ = stats.linregress(x, y)
    xfine = np.linspace(x.min() - 0.02, x.max() + 0.02, 200)
    ax.plot(xfine, sl*xfine + ic, "-", color="#1f77b4", lw=1.2, label="Linear")

    # Quadratic fit
    c2, c1, c0 = np.polyfit(x, y, 2)
    ax.plot(xfine, c2*xfine**2 + c1*xfine + c0, "--", color="#d62728", lw=1.2, label="Quadratic")

    # Find curvature p from lin_df
    curv_row = lin_df[lin_df["metric"] == met]
    if not curv_row.empty:
        p_c = curv_row.iloc[0]["p_curvature"]
        ax.set_title(f"{met.replace('_',' ')}\ncurvature p = {p_c}", fontweight="bold")
    else:
        ax.set_title(met.replace("_", " "), fontweight="bold")

    ax.set_xlabel("1 − $a_w$")
    ax.set_ylabel(met.replace("_", " "))
    ax.legend(fontsize=7, frameon=False)

for idx in range(n_panels, nrows * ncols):
    axes.flat[idx].set_visible(False)

fig.tight_layout()
for ext in ["svg", "png", "pdf"]:
    fig.savefig(OUTDIR / f"linearity_tests.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n" + "=" * 70)
print("COMPREHENSIVE STATS — KEY FINDINGS")
print("=" * 70)

print("\nKRUSKAL-WALLIS (omnibus across 6 groups):")
for _, r in kw_df.iterrows():
    print(f"  {r['metric']:20s}  H = {r['KW_H']:>8s}  p = {r['KW_p']}")

print(f"\nPAIRWISE COMPARISONS: {len(pw_df)} tests total")
sig = pw_df[pw_df["p_bonf"].astype(float) < 0.05]
print(f"  Significant after Bonferroni: {len(sig)}/{len(pw_df)}")
for _, r in sig.iterrows():
    print(f"    {r['metric']:15s}  {r['group_A']:5s} vs {r['group_B']:5s}  "
          f"p_adj = {r['p_bonf']:>8s}  Cliff δ = {r['Cliff_d']} ({r['effect_size']})")

print("\nLINEARITY TESTS (curvature significant at p<0.05?):")
for _, r in lin_df.iterrows():
    print(f"  {r['metric']:25s}  F = {r['F_for_curvature']:>8s}  "
          f"p = {r['p_curvature']:>8s}  Significant: {r['curvature_sig']}")

print("\nOLS vs THEIL-SEN slope difference:")
for _, r in rob_df.iterrows():
    print(f"  {r['metric']:25s}  OLS = {r['OLS_slope']:>10s}  "
          f"TS = {r['TheilSen_slope']:>10s}  diff = {r['slope_pct_diff']}")

n_flagged = cook_df["flagged"].sum()
print(f"\nINFLUENCE DIAGNOSTICS: {n_flagged} points flagged (Cook's D > 4/n)")
if n_flagged > 0:
    for _, r in cook_df[cook_df["flagged"]].iterrows():
        print(f"    {r['metric']:25s}  {r['trial_id']:12s}  D = {r['Cook_D']:.4f} (thresh = {r['threshold']:.4f})")

print("\nLEAF Healthy vs Diseased:")
for _, r in leaf_df.iterrows():
    print(f"  {r['metric']:20s}  MW p = {r['MW_p']:>8s}  Welch p = {r['Welch_p']:>8s}  "
          f"Cliff δ = {r['Cliff_d']} ({r['effect']})")

print("\nPOWER ANALYSIS:")
for _, r in power_df.iterrows():
    print(f"  {r['analysis']}")
    print(f"    min f² = {r['min_detectable_f2']}, min R² = {r['min_detectable_R2']}, {r['power']}")

print("\n" + "=" * 70)
print(f"Files saved to: {OUTDIR}/")
print("  comprehensive_stats_table.csv")
print("  comprehensive_stats_table.tex")
print("  pairwise_comparisons.csv")
print("  cook_distances.csv")
print("  influence_diagnostics.svg/png/pdf")
print("  linearity_tests.svg/png/pdf")
print("=" * 70)

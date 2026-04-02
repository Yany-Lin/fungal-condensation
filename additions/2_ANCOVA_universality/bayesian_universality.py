#!/usr/bin/env python3
"""
Bayesian and information-theoretic tests for universality across
hydrogel and fungal vapor-sink systems.

Tests whether the 30 trials (15 hydrogel, 15 fungi) follow a single
regression (universal) or require group-specific parameters.

Methods:
  A. AIC / BIC model comparison (universal vs two-line vs parallel vs six-line)
  B. Permutation test on slope difference (hydrogel vs fungi)
  C. Mixed-effects model (random intercept + slope by group)
  D. Partial-pooling visualisation
  E. Leave-one-group-out cross-validation
"""

import pathlib, warnings, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[2] / 'FigureTable' / 'output' / 'universal_metrics.csv'
OUT  = Path(__file__).resolve().parents[2] / 'additions' / '2_ANCOVA_universality'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

raw = pd.read_csv(DATA)
# keep only hydrogel + fungi rows (exclude Leaf)
df = raw[raw["system"].isin(["Hydrogel", "Fungi"])].copy()
df = df.dropna(subset=["delta_um"])

# convenience columns
df["delta"] = df["delta_um"]
df["system_label"] = df["system"]                         # Hydrogel / Fungi
df["condition"] = df["group"]                             # Agar, 1:1, 2:1, Green, White, Black

METRICS = {
    "dtau50_dr":   r"$\mathrm{d}\tau_{50}/\mathrm{d}r$",
    "zone_metric": "Zone metric",
}

# colours per condition
COND_COLORS = {
    "Agar":  "#1b9e77",
    "0.5:1": "#E67E22",
    "1:1":   "#d95f02",
    "2:1":   "#7570b3",
    "Green": "#e7298a",
    "White": "#66a61e",
    "Black": "#e6ab02",
}
SYS_COLORS = {"Hydrogel": "#0072B2", "Fungi": "#D55E00"}

def ols_fit(y, X):
    """Return OLS result; X should include constant if desired."""
    return sm.OLS(y, X).fit()


def aic_bic(res, n):
    """AIC and BIC from residual sum of squares."""
    rss = np.sum(res.resid**2)
    k   = res.df_model + 1          # +1 for variance parameter
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return aic, bic


def akaike_weights(aic_vec):
    d = np.array(aic_vec) - np.min(aic_vec)
    w = np.exp(-0.5 * d)
    return w / w.sum()


def model_comparison(df, ycol, ylabel):
    sub = df.dropna(subset=[ycol]).copy()
    y   = sub[ycol].values.astype(float)
    x   = sub["delta"].values.astype(float)
    n   = len(y)

    is_fungi = (sub["system_label"] == "Fungi").astype(float).values
    cond_dummies = pd.get_dummies(sub["condition"], drop_first=False)

    # Model 1 – universal
    X1 = sm.add_constant(x)
    r1 = ols_fit(y, X1)
    aic1, bic1 = aic_bic(r1, n)

    # Model 2 – two-line (separate slope + intercept for hydrogel vs fungi)
    X2 = np.column_stack([np.ones(n), x, is_fungi, x * is_fungi])
    r2 = ols_fit(y, X2)
    aic2, bic2 = aic_bic(r2, n)

    # Model 3 – parallel (same slope, different intercept)
    X3 = np.column_stack([np.ones(n), x, is_fungi])
    r3 = ols_fit(y, X3)
    aic3, bic3 = aic_bic(r3, n)

    # Model 4 – six-line (separate regression for each condition)
    cond_cols = list(cond_dummies.columns)
    ref = cond_cols[0]           # reference category
    X4_parts = [np.ones(n), x]
    for c in cond_cols[1:]:
        d = cond_dummies[c].values.astype(float)
        X4_parts.append(d)
        X4_parts.append(d * x)
    X4 = np.column_stack(X4_parts)
    r4 = ols_fit(y, X4)
    aic4, bic4 = aic_bic(r4, n)

    aics = [aic1, aic2, aic3, aic4]
    bics = [bic1, bic2, bic3, bic4]
    wts  = akaike_weights(aics)

    names = ["Universal", "Two-line", "Parallel", "Six-line"]
    kvals = [r1.df_model+1, r2.df_model+1, r3.df_model+1, r4.df_model+1]
    r2s   = [r1.rsquared, r2.rsquared, r3.rsquared, r4.rsquared]

    tbl = pd.DataFrame({
        "Model": names, "k": kvals, "R2": r2s,
        "AIC": aics, "BIC": bics, "Akaike_w": wts
    })
    tbl["dAIC"] = tbl["AIC"] - tbl["AIC"].min()
    tbl["dBIC"] = tbl["BIC"] - tbl["BIC"].min()

    return tbl, {"r1": r1, "r2": r2, "r3": r3, "r4": r4}


def permutation_test(df, ycol, n_perm=10_000, seed=42):
    sub = df.dropna(subset=[ycol]).copy()
    rng = np.random.default_rng(seed)

    def slope_diff(frame):
        h = frame[frame["system_label"] == "Hydrogel"]
        f = frame[frame["system_label"] == "Fungi"]
        if len(h) < 3 or len(f) < 3:
            return np.nan
        sh = stats.linregress(h["delta"], h[ycol]).slope
        sf = stats.linregress(f["delta"], f[ycol]).slope
        return sh - sf

    obs = slope_diff(sub)
    count = 0
    null_dist = np.empty(n_perm)
    labels = sub["system_label"].values.copy()
    for i in range(n_perm):
        rng.shuffle(labels)
        sub["system_label"] = labels
        null_dist[i] = slope_diff(sub)
    # restore
    sub["system_label"] = df.loc[sub.index, "system_label"].values

    p = np.mean(np.abs(null_dist) >= np.abs(obs))
    return obs, p, null_dist


def mixed_model(df, ycol):
    sub = df.dropna(subset=[ycol]).copy()
    sub["y"] = sub[ycol].astype(float)
    sub["x"] = sub["delta"].astype(float)
    sub["grp"] = sub["condition"].astype(str)

    md = smf.mixedlm("y ~ x", sub, groups=sub["grp"],
                      re_formula="~x")
    mdf = md.fit(reml=True)

    fe = smf.ols("y ~ x", data=sub).fit()

    ll_mixed = mdf.llf
    ll_fixed = fe.llf
    lr_stat = 2 * (ll_mixed - ll_fixed)
    p_lrt = stats.chi2.sf(lr_stat, df=3)

    re_cov = mdf.cov_re
    re_sd_intercept = np.sqrt(re_cov.iloc[0, 0]) if re_cov.shape[0] > 0 else 0
    re_sd_slope     = np.sqrt(re_cov.iloc[1, 1]) if re_cov.shape[0] > 1 else 0

    return mdf, fe, lr_stat, p_lrt, re_sd_intercept, re_sd_slope


def plot_partial_pooling(df, ycol, ylabel, mdf, re_sd_int, re_sd_slope, ax):
    sub = df.dropna(subset=[ycol]).copy()

    fe_int   = mdf.fe_params["Intercept"]
    fe_slope = mdf.fe_params["x"]

    xmin, xmax = sub["delta"].min() * 0.9, sub["delta"].max() * 1.1
    xline = np.linspace(xmin, xmax, 200)

    for cond in sorted(sub["condition"].unique()):
        c = COND_COLORS.get(cond, "grey")
        mask = sub["condition"] == cond
        ax.scatter(sub.loc[mask, "delta"], sub.loc[mask, ycol],
                   color=c, s=22, zorder=3, label=cond, edgecolors="white",
                   linewidths=0.3)
        re = mdf.random_effects.get(cond, {})
        ri = re.get("Intercept", re.get("Group", 0))
        rs = re.get("x", 0)
        yline = (fe_int + ri) + (fe_slope + rs) * xline
        ax.plot(xline, yline, color=c, lw=0.8, alpha=0.7)

    # universal line (thick)
    yuni = fe_int + fe_slope * xline
    ax.plot(xline, yuni, "k-", lw=2.0, alpha=0.85, label="Universal fit")

    ax.set_xlabel(r"$\delta$ ($\mu$m)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=2)

    # annotation
    ax.text(0.02, 0.97,
            f"RE SD(intercept) = {re_sd_int:.4f}\n"
            f"RE SD(slope) = {re_sd_slope:.2e}",
            transform=ax.transAxes, va="top", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.8))


def logo_cv(df, ycol):
    sub = df.dropna(subset=[ycol]).copy()
    groups = sorted(sub["condition"].unique())
    results = []
    for held_out in groups:
        train = sub[sub["condition"] != held_out]
        test  = sub[sub["condition"] == held_out]
        if len(test) < 2 or len(train) < 3:
            continue
        slope, intercept, _, _, _ = stats.linregress(train["delta"], train[ycol])
        pred = intercept + slope * test["delta"].values
        true = test[ycol].values
        ss_res = np.sum((true - pred)**2)
        ss_tot = np.sum((true - true.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        mae = np.mean(np.abs(true - pred))
        results.append({
            "held_out": held_out,
            "system": test["system_label"].iloc[0],
            "n_test": len(test),
            "R2": r2,
            "MAE": mae,
        })
    return pd.DataFrame(results)


report_lines = []

for ycol, ylabel in METRICS.items():
    sub = df.dropna(subset=[ycol])
    if len(sub) < 6:
        continue

    tag = ycol
    report_lines.append(f"\n{'='*60}")
    report_lines.append(f"  Metric: {ylabel}  ({ycol})")
    report_lines.append(f"{'='*60}")

    tbl, models = model_comparison(df, ycol, ylabel)
    report_lines.append("\nA. AIC / BIC model comparison")
    report_lines.append(tbl.to_string(index=False, float_format="%.4f"))
    best_aic = tbl.loc[tbl["AIC"].idxmin(), "Model"]
    best_bic = tbl.loc[tbl["BIC"].idxmin(), "Model"]
    report_lines.append(f"  >>> Preferred by AIC: {best_aic}")
    report_lines.append(f"  >>> Preferred by BIC: {best_bic}")
    tbl.to_csv(OUT / f"model_comparison_{tag}.csv", index=False)

    obs_diff, perm_p, null_dist = permutation_test(df, ycol)
    report_lines.append(f"\nB. Permutation test (slope difference hydrogel - fungi)")
    report_lines.append(f"   Observed slope diff = {obs_diff:.6f}")
    report_lines.append(f"   Permutation p-value = {perm_p:.4f}  (10 000 permutations)")

    # permutation histogram
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.hist(null_dist, bins=60, color="0.7", edgecolor="white", linewidth=0.3, density=True)
    ax.axvline(obs_diff, color="#D55E00", lw=1.5, label=f"Observed = {obs_diff:.4f}")
    ax.axvline(-obs_diff, color="#D55E00", lw=1.5, ls="--")
    ax.set_xlabel("Slope difference (Hydrogel - Fungi)")
    ax.set_ylabel("Density")
    ax.set_title(f"{ylabel}: permutation test (p = {perm_p:.3f})")
    ax.legend(frameon=False, fontsize=6.5)
    for fmt in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"permutation_{tag}.{fmt}")
    plt.close(fig)

    mdf, fe, lr_stat, p_lrt, re_sd_int, re_sd_slope = mixed_model(df, ycol)
    report_lines.append(f"\nC. Mixed-effects model (random intercept + slope by group)")
    report_lines.append(f"   Fixed intercept = {mdf.fe_params['Intercept']:.6f}")
    report_lines.append(f"   Fixed slope     = {mdf.fe_params['x']:.6f}")
    report_lines.append(f"   RE SD(intercept) = {re_sd_int:.6f}")
    report_lines.append(f"   RE SD(slope)     = {re_sd_slope:.2e}")
    report_lines.append(f"   LR statistic    = {lr_stat:.4f}")
    report_lines.append(f"   LRT p-value     = {p_lrt:.4f}")

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    plot_partial_pooling(df, ycol, ylabel, mdf, re_sd_int, re_sd_slope, ax)
    ax.set_title(f"{ylabel}: partial-pooling regression")
    for fmt in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"partial_pooling_{tag}.{fmt}")
    plt.close(fig)

    cv = logo_cv(df, ycol)
    report_lines.append(f"\nE. Leave-one-group-out cross-validation")
    report_lines.append(cv.to_string(index=False, float_format="%.4f"))
    cv.to_csv(OUT / f"logo_cv_{tag}.csv", index=False)

    # CV bar chart
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    colors = [COND_COLORS.get(g, "0.5") for g in cv["held_out"]]
    bars = ax.bar(cv["held_out"], cv["R2"], color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_ylabel("Prediction $R^2$")
    ax.set_title(f"{ylabel}: leave-one-group-out CV")
    ax.set_ylim(min(cv["R2"].min() - 0.15, -0.3), 1.05)
    for b, v in zip(bars, cv["R2"]):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    for fmt in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"logo_cv_{tag}.{fmt}")
    plt.close(fig)


fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
metric_list = [(ycol, ylabel) for ycol, ylabel in METRICS.items()
               if df.dropna(subset=[ycol]).shape[0] >= 6]

for row, (ycol, ylabel) in enumerate(metric_list):
    sub = df.dropna(subset=[ycol]).copy()

    # col 0: model comparison bar chart
    ax = axes[row, 0]
    tbl, _ = model_comparison(df, ycol, ylabel)
    names = tbl["Model"]
    ax.barh(names, tbl["Akaike_w"], color=["#0072B2", "#D55E00", "#009E73", "#CC79A7"])
    ax.set_xlabel("Akaike weight")
    ax.set_title(f"{ylabel}")
    for i, w in enumerate(tbl["Akaike_w"]):
        ax.text(w + 0.01, i, f"{w:.2f}", va="center", fontsize=6.5)

    # col 1: partial pooling
    ax = axes[row, 1]
    mdf_r, _, _, _, rei, res = mixed_model(df, ycol)
    plot_partial_pooling(df, ycol, ylabel, mdf_r, rei, res, ax)
    ax.set_title(f"{ylabel}: partial pooling")

    # col 2: LOGO CV
    ax = axes[row, 2]
    cv = logo_cv(df, ycol)
    colors = [COND_COLORS.get(g, "0.5") for g in cv["held_out"]]
    bars = ax.bar(cv["held_out"], cv["R2"], color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_ylabel("Prediction $R^2$")
    ax.set_title(f"{ylabel}: LOGO CV")
    ax.set_ylim(min(cv["R2"].min() - 0.15, -0.3), 1.05)
    for b, v in zip(bars, cv["R2"]):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    ax.tick_params(axis="x", rotation=30)

fig.tight_layout()
for fmt in ("svg", "png", "pdf"):
    fig.savefig(OUT / f"universality_composite.{fmt}")
plt.close(fig)


# ── save report ──
report = "\n".join(report_lines)
print(report)
(OUT / "universality_report.txt").write_text(report)
print(f"\nAll outputs saved to {OUT}/")

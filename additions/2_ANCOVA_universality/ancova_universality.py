#!/usr/bin/env python3
"""
Frequentist tests for universality of condensation metrics
across hydrogel and fungal vapor-sink systems.

Tests: ANCOVA / interaction tests, Chow structural-break test,
within-group slope comparison (z-test), Spearman/Pearson correlations,
equivalence testing (TOST), and publication-quality figures.
"""

import pathlib, warnings, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.api as sms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)

DATA = Path(__file__).resolve().parents[2] / 'FigureTable' / 'output' / 'universal_metrics.csv'
OUT  = Path(__file__).resolve().parents[2] / 'additions' / '2_ANCOVA_universality'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

df_all = pd.read_csv(DATA)
# Keep only hydrogel + fungi rows (exclude Leaf rows)
df = df_all[df_all["system"].isin(["Hydrogel", "Fungi"])].copy()
df["system_type"] = df["system"].map({"Hydrogel": "Hydrogel", "Fungi": "Fungi"})

# Panel 3D:  dtau50_dr  vs  delta_um
# Panel 3E:  zone_metric  vs  delta_um
# Make sure numerics are clean
for c in ["delta_um", "dtau50_dr", "zone_metric"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Convenience subsets
hydro = df[df["system_type"] == "Hydrogel"]
fungi = df[df["system_type"] == "Fungi"]

def ols_fit(x, y):
    """OLS with intercept, returns (slope, intercept, se_slope, se_intercept, r2, residuals, RSS, n)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.array(x[mask], dtype=float), np.array(y[mask], dtype=float)
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit()
    return dict(
        slope=res.params[1], intercept=res.params[0],
        se_slope=res.bse[1], se_intercept=res.bse[0],
        r2=res.rsquared, rss=np.sum(res.resid**2),
        n=len(x), resid=res.resid, x=x, y=y, result=res
    )


def pearson_spearman(x, y, label=""):
    mask = np.isfinite(x) & np.isfinite(y)
    x2, y2 = np.array(x[mask]), np.array(y[mask])
    pr, pp = stats.pearsonr(x2, y2)
    sr, sp = stats.spearmanr(x2, y2)
    return dict(label=label, pearson_r=pr, pearson_p=pp,
                spearman_rho=sr, spearman_p=sp, n=len(x2))


print("=" * 72)
print("A.  ANCOVA  /  interaction tests")
print("=" * 72)

ancova_results = {}
for yvar, ylabel in [("dtau50_dr", "dτ₅₀/dr"), ("zone_metric", "(R_far−R_near)/R_mid")]:
    sub = df[["delta_um", yvar, "system_type", "group"]].dropna()
    formula2 = f"{yvar} ~ delta_um * C(system_type, Treatment(reference='Hydrogel'))"
    m2 = ols(formula2, data=sub).fit()
    anova2 = sm.stats.anova_lm(m2, typ=2)

    formula6 = f"{yvar} ~ delta_um * C(group)"
    m6 = ols(formula6, data=sub).fit()
    anova6 = sm.stats.anova_lm(m6, typ=2)

    ancova_results[yvar] = dict(model2=m2, anova2=anova2, model6=m6, anova6=anova6)

    print(f"\n--- {ylabel} ---")
    print("  2-group ANCOVA  (Hydrogel vs Fungi)")
    interact_row = [r for r in anova2.index if ":" in r and "system_type" in r][0]
    sys_row = [r for r in anova2.index if "system_type" in r and ":" not in r][0]
    print(f"    Interaction (same slope?):  F = {anova2.loc[interact_row,'F']:.4f},  "
          f"p = {anova2.loc[interact_row,'PR(>F)']:.4g}")
    print(f"    Main effect  (same intercept?):  F = {anova2.loc[sys_row,'F']:.4f},  "
          f"p = {anova2.loc[sys_row,'PR(>F)']:.4g}")
    print(f"\n  6-group ANCOVA  (Agar, 1:1, 2:1, Green, White, Black)")
    interact_rows_6 = [r for r in anova6.index if "delta_um:C(group)" in r]
    if interact_rows_6:
        print(f"    Interaction (same slope?):  F = {anova6.loc[interact_rows_6[0],'F']:.4f},  "
              f"p = {anova6.loc[interact_rows_6[0],'PR(>F)']:.4g}")
    group_rows_6 = [r for r in anova6.index if r.startswith("C(group)") and "delta_um" not in r]
    if group_rows_6:
        print(f"    Main effect (group):  F = {anova6.loc[group_rows_6[0],'F']:.4f},  "
              f"p = {anova6.loc[group_rows_6[0],'PR(>F)']:.4g}")


print("\n" + "=" * 72)
print("B.  Chow test  (structural break)")
print("=" * 72)

chow_results = {}
for yvar, ylabel in [("dtau50_dr", "dτ₅₀/dr"), ("zone_metric", "(R_far−R_near)/R_mid")]:
    sub = df[["delta_um", yvar, "system_type"]].dropna()
    pooled = ols_fit(sub["delta_um"], sub[yvar])
    h = ols_fit(sub.loc[sub.system_type == "Hydrogel", "delta_um"],
                sub.loc[sub.system_type == "Hydrogel", yvar])
    f_ = ols_fit(sub.loc[sub.system_type == "Fungi", "delta_um"],
                 sub.loc[sub.system_type == "Fungi", yvar])

    RSS_pooled = pooled["rss"]
    RSS_sep = h["rss"] + f_["rss"]
    k = 2  # number of parameters per group (intercept + slope)
    n = pooled["n"]
    F_chow = ((RSS_pooled - RSS_sep) / k) / (RSS_sep / (n - 2 * k))
    p_chow = 1 - stats.f.cdf(F_chow, k, n - 2 * k)

    chow_results[yvar] = dict(F=F_chow, p=p_chow, df1=k, df2=n - 2 * k)
    print(f"\n  {ylabel}:  F({k},{n-2*k}) = {F_chow:.4f},  p = {p_chow:.4g}")


print("\n" + "=" * 72)
print("C.  Within-group slope comparison  (z-test)")
print("=" * 72)

slope_z = {}
for yvar, ylabel in [("dtau50_dr", "dτ₅₀/dr"), ("zone_metric", "(R_far−R_near)/R_mid")]:
    sub = df[["delta_um", yvar, "system_type"]].dropna()
    h = ols_fit(sub.loc[sub.system_type == "Hydrogel", "delta_um"],
                sub.loc[sub.system_type == "Hydrogel", yvar])
    f_ = ols_fit(sub.loc[sub.system_type == "Fungi", "delta_um"],
                 sub.loc[sub.system_type == "Fungi", yvar])
    z = (h["slope"] - f_["slope"]) / np.sqrt(h["se_slope"]**2 + f_["se_slope"]**2)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    slope_z[yvar] = dict(
        slope_h=h["slope"], se_h=h["se_slope"],
        slope_f=f_["slope"], se_f=f_["se_slope"],
        z=z, p=p
    )
    print(f"\n  {ylabel}")
    print(f"    Hydrogel slope = {h['slope']:.5f} ± {h['se_slope']:.5f}")
    print(f"    Fungi    slope = {f_['slope']:.5f} ± {f_['se_slope']:.5f}")
    print(f"    z = {z:.4f},  p = {p:.4g}")


print("\n" + "=" * 72)
print("D.  Spearman ρ alongside Pearson r")
print("=" * 72)

corr_rows = []

# Panels 3D, 3E (pooled, 30 trials)
sub3 = df[["delta_um", "dtau50_dr", "zone_metric"]].dropna(subset=["delta_um"])
corr_rows.append(pearson_spearman(sub3["delta_um"], sub3["dtau50_dr"].dropna().reindex(sub3.index), "Panel 3D: dτ₅₀/dr vs δ"))
corr_rows.append(pearson_spearman(sub3["delta_um"], sub3["zone_metric"].dropna().reindex(sub3.index), "Panel 3E: zone_metric vs δ"))

# Panels 2F, 2H, 2K — per-trial Pearson r values are stored in csv; compute Spearman too
# These are per-trial R(r) regressions already stored.  We can report the *cross-trial*
# regression of dR_dr vs delta if available.  Let's also report dR_dr vs delta.
if "dR_dr_um_per_mm" in df.columns:
    sub_dR = df[["delta_um", "dR_dr_um_per_mm"]].copy()
    sub_dR["dR_dr_um_per_mm"] = pd.to_numeric(sub_dR["dR_dr_um_per_mm"], errors="coerce")
    sub_dR["delta_um"] = pd.to_numeric(sub_dR["delta_um"], errors="coerce")
    sub_dR = sub_dR.dropna()
    if len(sub_dR) >= 3:
        corr_rows.append(pearson_spearman(sub_dR["delta_um"], sub_dR["dR_dr_um_per_mm"],
                                           "dR/dr vs δ (supplementary)"))

# Leaf panels: per-trial Pearson r stored in csv
leaf = df_all[df_all["system"] == "Leaf"].copy()
leaf["pearson_r"] = pd.to_numeric(leaf["pearson_r"], errors="coerce")
leaf["dR_dr_um_per_mm"] = pd.to_numeric(leaf["dR_dr_um_per_mm"], errors="coerce")
leaf2 = leaf.dropna(subset=["dR_dr_um_per_mm", "pearson_r"])
if len(leaf2) >= 3:
    corr_rows.append(pearson_spearman(leaf2["dR_dr_um_per_mm"], leaf2["pearson_r"],
                                       "Leaf: r vs dR/dr (cross-trial)"))

corr_df = pd.DataFrame(corr_rows)
print(corr_df.to_string(index=False))


print("\n" + "=" * 72)
print("E.  Equivalence testing  (TOST)  —  ±30 % of pooled slope")
print("=" * 72)

tost_results = {}
for yvar, ylabel in [("dtau50_dr", "dτ₅₀/dr"), ("zone_metric", "(R_far−R_near)/R_mid")]:
    sub = df[["delta_um", yvar, "system_type"]].dropna()
    pooled = ols_fit(sub["delta_um"], sub[yvar])
    h = ols_fit(sub.loc[sub.system_type == "Hydrogel", "delta_um"],
                sub.loc[sub.system_type == "Hydrogel", yvar])
    f_ = ols_fit(sub.loc[sub.system_type == "Fungi", "delta_um"],
                 sub.loc[sub.system_type == "Fungi", yvar])

    diff = h["slope"] - f_["slope"]
    se_diff = np.sqrt(h["se_slope"]**2 + f_["se_slope"]**2)
    margin = 0.30 * abs(pooled["slope"])

    t_upper = (diff - margin) / se_diff
    t_lower = (diff + margin) / se_diff
    df_tost = h["n"] + f_["n"] - 4
    p_upper = stats.t.cdf(t_upper, df_tost)
    p_lower = 1 - stats.t.cdf(t_lower, df_tost)
    p_tost = max(p_upper, p_lower)

    tost_results[yvar] = dict(diff=diff, se_diff=se_diff, margin=margin,
                               t_upper=t_upper, p_upper=p_upper,
                               t_lower=t_lower, p_lower=p_lower,
                               p_tost=p_tost, df=df_tost)
    print(f"\n  {ylabel}")
    print(f"    Pooled slope = {pooled['slope']:.5f}")
    print(f"    Equivalence margin = ±{margin:.5f}  (30 % of pooled slope)")
    print(f"    Slope difference (H − F) = {diff:.5f} ± {se_diff:.5f}")
    print(f"    Upper bound test: t = {t_upper:.4f},  p = {p_upper:.4g}")
    print(f"    Lower bound test: t = {t_lower:.4f},  p = {p_lower:.4g}")
    print(f"    TOST overall p = {p_tost:.4g}  {'→ EQUIVALENT' if p_tost < 0.05 else '→ NOT conclusively equivalent'}")


COLORS = {"Hydrogel": "#2166AC", "Fungi": "#B2182B"}
MARKERS_6 = {"Agar": "o", "1:1": "s", "2:1": "D",
             "Green": "^", "White": "v", "Black": "P"}
COLORS_6 = {"Agar": "#92C5DE", "1:1": "#4393C3", "2:1": "#2166AC",
            "Green": "#F4A582", "White": "#D6604D", "Black": "#B2182B"}

for yvar, ylabel, panel_label in [
    ("dtau50_dr", r"d$\tau_{50}$/d$r$  (s mm$^{-1}$)", "D"),
    ("zone_metric", r"($R_{\rm far}$−$R_{\rm near}$) / $R_{\rm mid}$", "E"),
]:
    sub = df[["delta_um", yvar, "system_type", "group"]].dropna()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    for grp, marker in MARKERS_6.items():
        g = sub[sub.group == grp]
        if len(g) == 0:
            continue
        ax.scatter(g["delta_um"], g[yvar], marker=marker,
                   c=COLORS_6[grp], s=36, edgecolors="k", linewidths=0.4,
                   zorder=3, label=grp)

    xrange = np.linspace(sub["delta_um"].min() - 30, sub["delta_um"].max() + 30, 200)
    for stype, col, ls in [("Hydrogel", COLORS["Hydrogel"], "--"),
                            ("Fungi", COLORS["Fungi"], "--")]:
        s = sub[sub.system_type == stype]
        fit = ols_fit(s["delta_um"], s[yvar])
        ax.plot(xrange, fit["intercept"] + fit["slope"] * xrange,
                color=col, ls=ls, lw=1.0, zorder=2)

    pooled = ols_fit(sub["delta_um"], sub[yvar])
    ax.plot(xrange, pooled["intercept"] + pooled["slope"] * xrange,
            color="k", ls="-", lw=1.5, zorder=2, label="Pooled")

    anova2 = ancova_results[yvar]["anova2"]
    interact_row = [r for r in anova2.index if ":" in r and "system_type" in r][0]
    sys_row = [r for r in anova2.index if "system_type" in r and ":" not in r][0]
    p_int = anova2.loc[interact_row, "PR(>F)"]
    p_sys = anova2.loc[sys_row, "PR(>F)"]
    p_chow = chow_results[yvar]["p"]

    ann = (f"ANCOVA interaction: p = {p_int:.3f}\n"
           f"ANCOVA intercept: p = {p_sys:.3f}\n"
           f"Chow test: p = {p_chow:.3f}")
    ax.text(0.03, 0.97, ann, transform=ax.transAxes, fontsize=6.5,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))

    ax.set_xlabel(r"$\delta$  ($\mu$m)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=6, loc="lower right", frameon=True, framealpha=0.9,
              edgecolor="0.7", handletextpad=0.3, borderpad=0.4)

    for fmt in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"ancova_panel3{panel_label}.{fmt}")
    plt.close(fig)
    print(f"\n  Saved ancova_panel3{panel_label}.svg / .png / .pdf")


rows_table = []
for yvar, ylabel in [("dtau50_dr", "dtau50/dr vs delta (3D)"),
                      ("zone_metric", "(R_far-R_near)/R_mid vs delta (3E)")]:
    anova2 = ancova_results[yvar]["anova2"]
    interact_row = [r for r in anova2.index if ":" in r and "system_type" in r][0]
    sys_row = [r for r in anova2.index if "system_type" in r and ":" not in r][0]

    rows_table.append([
        ylabel,
        "ANCOVA interaction (same slope?)",
        f"F = {anova2.loc[interact_row,'F']:.3f}",
        f"{anova2.loc[interact_row,'PR(>F)']:.4f}",
    ])
    rows_table.append([
        "",
        "ANCOVA main effect (same intercept?)",
        f"F = {anova2.loc[sys_row,'F']:.3f}",
        f"{anova2.loc[sys_row,'PR(>F)']:.4f}",
    ])
    rows_table.append([
        "",
        "Chow test (structural break)",
        f"F = {chow_results[yvar]['F']:.3f}",
        f"{chow_results[yvar]['p']:.4f}",
    ])
    sz = slope_z[yvar]
    rows_table.append([
        "",
        "Slope z-test",
        f"z = {sz['z']:.3f}",
        f"{sz['p']:.4f}",
    ])
    tost = tost_results[yvar]
    rows_table.append([
        "",
        "TOST equivalence (±30 %)",
        f"Δslope = {tost['diff']:.4f}",
        f"{tost['p_tost']:.4f}",
    ])

for _, row in corr_df.iterrows():
    rows_table.append([
        row["label"],
        "Pearson r / Spearman ρ",
        f"r = {row['pearson_r']:.3f} / ρ = {row['spearman_rho']:.3f}",
        f"{row['pearson_p']:.2e} / {row['spearman_p']:.2e}",
    ])

fig_tab, ax_tab = plt.subplots(figsize=(8.5, 0.35 * len(rows_table) + 1.0))
ax_tab.axis("off")
col_labels = ["Relationship", "Test", "Statistic", "p-value"]
table = ax_tab.table(cellText=rows_table, colLabels=col_labels,
                     loc="center", cellLoc="left")
table.auto_set_font_size(False)
table.set_fontsize(7)
table.auto_set_column_width([0, 1, 2, 3])
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#D9E2F3")
    table[0, j].set_text_props(weight="bold")

for fmt in ("svg", "png", "pdf"):
    fig_tab.savefig(OUT / f"ancova_summary_table.{fmt}")
plt.close(fig_tab)
print(f"\n  Saved ancova_summary_table.svg / .png / .pdf")


results_txt = []
results_txt.append("ANCOVA UNIVERSALITY TESTS — machine-readable summary\n")

for yvar, ylabel in [("dtau50_dr", "dτ₅₀/dr"), ("zone_metric", "zone_metric")]:
    results_txt.append(f"\n{'='*60}")
    results_txt.append(f"Metric: {ylabel}")
    results_txt.append(f"{'='*60}")

    anova2 = ancova_results[yvar]["anova2"]
    results_txt.append(f"\n2-group ANOVA table:\n{anova2.to_string()}")

    results_txt.append(f"\nChow test: F={chow_results[yvar]['F']:.6f}, p={chow_results[yvar]['p']:.6f}")

    sz = slope_z[yvar]
    results_txt.append(f"\nSlope z-test: z={sz['z']:.6f}, p={sz['p']:.6f}")
    results_txt.append(f"  Hydrogel slope={sz['slope_h']:.6f} ± {sz['se_h']:.6f}")
    results_txt.append(f"  Fungi slope={sz['slope_f']:.6f} ± {sz['se_f']:.6f}")

    tost = tost_results[yvar]
    results_txt.append(f"\nTOST: margin=±{tost['margin']:.6f}, p_tost={tost['p_tost']:.6f}")

results_txt.append(f"\n\n{'='*60}")
results_txt.append("Pearson / Spearman correlations")
results_txt.append(f"{'='*60}")
results_txt.append(corr_df.to_string(index=False))

(OUT / "ancova_results.txt").write_text("\n".join(results_txt))
print(f"\n  Saved ancova_results.txt")

print("\nAll analyses complete.")

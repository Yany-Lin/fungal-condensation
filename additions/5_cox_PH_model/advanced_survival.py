#!/usr/bin/env python3
"""
Advanced survival analysis for fungal vapor-sink paper.

Models:
  1. AFT (log-normal, Weibull) vs Cox PH — AIC comparison
  2. Size-stratified KM survival (3 × 2 panel)
  3. Time-varying coefficient model
  4. Mega-model pooling all trials with δ interaction
  5. Concordance-index comparison across covariate sets

Output figures saved as SVG in the same directory as this script.
"""

import pathlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import (
    CoxPHFitter,
    KaplanMeierFitter,
    WeibullAFTFitter,
    LogNormalAFTFitter,
    LogLogisticAFTFitter,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT     = Path(__file__).resolve().parents[2] / 'additions' / '5_cox_PH_model'
TRACKS  = Path(__file__).resolve().parents[2] / 'FigureHGAggregate' / 'code' / 'test_tracking' / 'output'
METRICS = Path(__file__).resolve().parents[2] / 'FigureTable' / 'output' / 'universal_metrics.csv'

plt.rcParams.update({
    "font.family":        "Arial",
    "font.size":          8,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "svg.fonttype":       "none",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "pdf.fonttype":       42,
})

COL_NEAR = "#d62728"
COL_FAR  = "#1f77b4"

def load_all_tracks():
    """Load every track-history CSV; derive trial_id from filename."""
    frames = []
    for fp in sorted(TRACKS.glob("*_track_histories.csv")):
        tid = fp.stem.replace("_track_histories", "")
        df = pd.read_csv(fp)
        df["trial_id"] = tid
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)

    # Derived columns
    pool["distance_mm"]  = pool["distance_um"] / 1000.0
    pool["lifetime_min"] = pool["lifetime_s"] / 60.0
    # event = 1 means droplet disappeared (died); 0 = censored (survived to end)
    pool["event"]        = (~pool["censored"]).astype(int)

    # Best available radius: prefer R_eq_seed, fall back to R_eq_birth
    pool["R0_um"] = pool["R_eq_seed"].where(pool["R_eq_seed"] > 0,
                                             pool["R_eq_birth"])
    pool["has_R0"] = pool["R0_um"] > 0
    pool.loc[pool["has_R0"], "log_R0_um"] = np.log(pool.loc[pool["has_R0"], "R0_um"])
    return pool


def merge_delta(pool):
    """Attach per-trial δ from universal_metrics."""
    met = pd.read_csv(METRICS, usecols=["trial_id", "delta_um", "group"])
    met = met.dropna(subset=["delta_um"])
    merged = pool.merge(met, on="trial_id", how="inner")
    merged["log_delta"] = np.log(merged["delta_um"].clip(lower=1))
    return merged


print("Loading track histories …")
pool_raw = load_all_tracks()
print(f"  → {len(pool_raw):,} droplets across {pool_raw['trial_id'].nunique()} trials")
print(f"    {pool_raw['event'].sum():,} events, "
      f"{(pool_raw['event']==0).sum():,} censored")
print(f"    {pool_raw['has_R0'].sum():,} with measured R₀")

pool = merge_delta(pool_raw)
print(f"  → {len(pool):,} droplets with δ after merge ({pool['trial_id'].nunique()} trials)")

# Basic quality: positive lifetime, finite distance
mask_base = (pool["lifetime_min"] > 0) & np.isfinite(pool["distance_mm"])
pool = pool.loc[mask_base].copy()
print(f"  → {len(pool):,} after base quality filter "
      f"({pool['event'].sum():,} events, {(pool['event']==0).sum():,} censored)\n")

# Subsets
pool_R0 = pool.loc[pool["has_R0"] & np.isfinite(pool["log_R0_um"])].copy()
print(f"  Subset with R₀: {len(pool_R0):,} droplets\n")

print("=" * 60)
print("1. AFT MODEL COMPARISON")
print("=" * 60)

surv_df = pool_R0[["lifetime_min", "event", "distance_mm", "log_R0_um"]].copy()

# Fit AFT models
waft = WeibullAFTFitter()
waft.fit(surv_df, duration_col="lifetime_min", event_col="event")

lnaft = LogNormalAFTFitter()
lnaft.fit(surv_df, duration_col="lifetime_min", event_col="event")

llaft = LogLogisticAFTFitter()
llaft.fit(surv_df, duration_col="lifetime_min", event_col="event")

# Cox PH for comparison (but AIC not directly comparable — use BIC too)
cox = CoxPHFitter()
cox.fit(surv_df, duration_col="lifetime_min", event_col="event")

# AFT AIC comparison (all use full likelihood, so these are comparable)
aft_models = {
    "Weibull AFT":      waft,
    "Log-Normal AFT":   lnaft,
    "Log-Logistic AFT": llaft,
}

aic_table = []
for name, m in aft_models.items():
    ll = m.log_likelihood_
    k  = m.summary.shape[0]
    aic = -2 * ll + 2 * k
    aic_table.append({"Model": name, "log-lik": ll, "k": k, "AIC": aic})
aic_df = pd.DataFrame(aic_table).sort_values("AIC")
aic_df["ΔAIC"] = aic_df["AIC"] - aic_df["AIC"].min()

print("\nAFT model AIC comparison:")
print(aic_df.to_string(index=False, float_format="{:.1f}".format))

# Cox PH concordance for reference
print(f"\nCox PH concordance index: {cox.concordance_index_:.4f}")

# Best AFT
best_aft_name = aic_df.iloc[0]["Model"]
best_aft = aft_models[best_aft_name]

print(f"\n--- Best AFT ({best_aft_name}) coefficients ---")
print(best_aft.summary.to_string(float_format="{:.4f}".format))

# Extract distance coefficient from best AFT
beta_dist = None
for k in best_aft.params_.index:
    if "distance" in str(k):
        beta_dist = best_aft.params_[k]
        break
if beta_dist is not None:
    print(f"\n  → Each mm further from source multiplies median lifetime "
          f"by exp({beta_dist:.4f}) = {np.exp(beta_dist):.3f}")
    print(f"    i.e., a {(np.exp(beta_dist)-1)*100:+.1f}% change per mm")

# Cox PH hazard ratio for distance
hr_dist = np.exp(cox.params_["distance_mm"])
print(f"\n  Cox PH: HR for distance = {hr_dist:.4f} per mm "
      f"(lower → protective → longer survival near source)")

fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

# Panel A: AIC bar chart
ax = axes[0]
colors_bar = ["#2ca02c" if row["ΔAIC"] == 0 else "#aaaaaa"
              for _, row in aic_df.iterrows()]
bars = ax.barh(aic_df["Model"], aic_df["ΔAIC"], color=colors_bar,
               edgecolor="k", linewidth=0.5)
ax.set_xlabel("ΔAIC (lower is better)")
ax.set_title("a  AFT model comparison", loc="left", fontweight="bold")
ax.invert_yaxis()
for bar, val in zip(bars, aic_df["ΔAIC"]):
    ax.text(val + aic_df["ΔAIC"].max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}", va="center", fontsize=7)

# Panel B: Best AFT coefficients (mu_ only)
ax = axes[1]
summ = best_aft.summary
mu_rows = [idx for idx in summ.index if "mu_" in str(idx)]
if mu_rows:
    sub = summ.loc[mu_rows].copy()
    labels = []
    for idx in sub.index:
        if isinstance(idx, tuple):
            labels.append(idx[1])
        else:
            labels.append(str(idx).replace("mu_:", ""))
    sub.index = labels
    sub = sub.loc[[l for l in sub.index if "Intercept" not in l]]

    coefs = sub["coef"]
    ci_lo = sub["coef lower 95%"]
    ci_hi = sub["coef upper 95%"]
    y_pos = np.arange(len(coefs))

    ax.errorbar(coefs, y_pos, xerr=[coefs - ci_lo, ci_hi - coefs],
                fmt="o", color="k", markersize=5, capsize=3, linewidth=1)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_yticks(y_pos)
    ylabels = []
    for l in coefs.index:
        if "distance" in l.lower():
            ylabels.append("Distance (mm)")
        elif "r0" in l.lower():
            ylabels.append("log R₀ (μm)")
        else:
            ylabels.append(l)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(f"AFT coefficient ({best_aft_name} μ)")
    ax.set_title("b  Effect sizes", loc="left", fontweight="bold")

fig.tight_layout()
fig.savefig(OUT / "aft_comparison.svg")
print(f"\n  Saved → {OUT / 'aft_comparison.svg'}")

print("\n" + "=" * 60)
print("2. SIZE-STRATIFIED SURVIVAL")
print("=" * 60)

q33, q66 = pool_R0["R0_um"].quantile([1/3, 2/3])
pool_R0["size_class"] = pd.cut(
    pool_R0["R0_um"],
    bins=[-np.inf, q33, q66, np.inf],
    labels=["Small", "Medium", "Large"],
)

d_med = pool_R0["distance_mm"].median()
pool_R0["dist_class"] = np.where(pool_R0["distance_mm"] <= d_med, "Near", "Far")

print(f"  Size tercile boundaries: {q33:.1f}, {q66:.1f} μm")
print(f"  Distance split at median: {d_med:.2f} mm")

fig, axes = plt.subplots(3, 2, figsize=(6.5, 7.5), sharey=True, sharex=True)

size_labels = ["Small", "Medium", "Large"]
dist_labels = ["Near", "Far"]

for i, sz in enumerate(size_labels):
    for j, dc in enumerate(dist_labels):
        ax = axes[i, j]
        sub = pool_R0[(pool_R0["size_class"] == sz) & (pool_R0["dist_class"] == dc)]

        kmf = KaplanMeierFitter()
        kmf.fit(sub["lifetime_min"], event_observed=sub["event"])
        kmf.plot_survival_function(
            ax=ax, color=COL_NEAR if dc == "Near" else COL_FAR,
            ci_show=True, ci_alpha=0.15, linewidth=1.2, label=None,
        )
        ax.get_legend().remove()
        med = kmf.median_survival_time_
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.4)
        if np.isfinite(med):
            ax.axvline(med, color="grey", linestyle=":", linewidth=0.4)
            ax.text(0.95, 0.92, f"τ₅₀ = {med:.1f} min",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                    color=COL_NEAR if dc == "Near" else COL_FAR)

        if i == 0:
            ax.set_title(f"{dc} source", fontweight="bold")
        if i == 2:
            ax.set_xlabel("Time (min)")
        if j == 0:
            ax.set_ylabel(f"{sz}\nSurvival probability")
        ax.set_ylim(-0.02, 1.02)

fig.text(0.01, 0.98, "Size-stratified Kaplan–Meier survival",
         fontsize=10, fontweight="bold", va="top")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "size_stratified_survival.svg")
print(f"  Saved → {OUT / 'size_stratified_survival.svg'}")

print("\n  Median survival (min) by size × distance:")
for sz in size_labels:
    row = []
    for dc in dist_labels:
        sub = pool_R0[(pool_R0["size_class"] == sz) & (pool_R0["dist_class"] == dc)]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["lifetime_min"], event_observed=sub["event"])
        m = kmf.median_survival_time_
        row.append(f"{m:.1f}" if np.isfinite(m) else ">obs")
    print(f"    {sz:8s}  Near={row[0]:>8s}  Far={row[1]:>8s}")

print("\n" + "=" * 60)
print("3. TIME-VARYING DISTANCE EFFECT")
print("=" * 60)

t_cuts = pool["lifetime_min"].quantile([1/3, 2/3]).values
print(f"  Time-axis cut-points: {t_cuts[0]:.1f}, {t_cuts[1]:.1f} min")

episodes = []
for t_lo, t_hi, label in [
    (0,         t_cuts[0], "early"),
    (t_cuts[0], t_cuts[1], "middle"),
    (t_cuts[1], pool["lifetime_min"].max() + 1, "late"),
]:
    ep = pool[["lifetime_min", "event", "distance_mm", "trial_id"]].copy()
    ep["start"] = t_lo
    ep["stop"]  = ep["lifetime_min"].clip(upper=t_hi)
    ep["ep_event"] = ((ep["lifetime_min"] <= t_hi) & (ep["event"] == 1)).astype(int)
    ep = ep[ep["lifetime_min"] > t_lo].copy()
    ep["period"] = label
    episodes.append(ep)

ep_df = pd.concat(episodes, ignore_index=True)
ep_df["duration"] = ep_df["stop"] - ep_df["start"]
ep_df = ep_df[ep_df["duration"] > 0].copy()

tv_results = []
for period in ["early", "middle", "late"]:
    sub = ep_df[ep_df["period"] == period].copy()
    sub_fit = sub[["duration", "ep_event", "distance_mm"]].rename(
        columns={"duration": "T", "ep_event": "E"})
    cph = CoxPHFitter()
    cph.fit(sub_fit, duration_col="T", event_col="E")
    s = cph.summary.loc["distance_mm"]
    tv_results.append({
        "Period": period, "n_at_risk": len(sub),
        "HR": np.exp(s["coef"]),
        "HR_lo": np.exp(s["coef lower 95%"]),
        "HR_hi": np.exp(s["coef upper 95%"]),
        "coef": s["coef"], "p": s["p"],
    })
    print(f"  {period:8s}  n={len(sub):>6,}  HR={np.exp(s['coef']):.4f}  "
          f"[{np.exp(s['coef lower 95%']):.4f}–{np.exp(s['coef upper 95%']):.4f}]  "
          f"p={s['p']:.2e}")

tv_df = pd.DataFrame(tv_results)

print("\n" + "=" * 60)
print("4. MEGA-MODEL (all trials, δ interaction)")
print("=" * 60)

mega_df = pool_R0[["lifetime_min", "event", "distance_mm", "log_R0_um",
                    "delta_um", "log_delta", "trial_id"]].copy()

# Standardize covariates for stability
for col in ["distance_mm", "log_R0_um", "log_delta"]:
    mu, sd = mega_df[col].mean(), mega_df[col].std()
    mega_df[f"{col}_z"] = (mega_df[col] - mu) / sd
    print(f"  {col}: mean={mu:.3f}, sd={sd:.3f}")

mega_df["dist_x_delta_z"] = mega_df["distance_mm_z"] * mega_df["log_delta_z"]

fit_cols = ["lifetime_min", "event",
            "distance_mm_z", "log_R0_um_z", "log_delta_z", "dist_x_delta_z"]

mega_cox = CoxPHFitter()
mega_cox.fit(mega_df[fit_cols], duration_col="lifetime_min", event_col="event")

print("\nMega-model summary (standardised covariates):")
print(mega_cox.summary.to_string(float_format="{:.4f}".format))

# Also fit on the FULL pool (distance + delta only, no R0) for maximum power
mega_full = pool[["lifetime_min", "event", "distance_mm", "log_delta"]].copy()
for col in ["distance_mm", "log_delta"]:
    mu, sd = mega_full[col].mean(), mega_full[col].std()
    mega_full[f"{col}_z"] = (mega_full[col] - mu) / sd
mega_full["dist_x_delta_z"] = mega_full["distance_mm_z"] * mega_full["log_delta_z"]

mega_cox_full = CoxPHFitter()
mega_cox_full.fit(
    mega_full[["lifetime_min", "event", "distance_mm_z", "log_delta_z", "dist_x_delta_z"]],
    duration_col="lifetime_min", event_col="event")

print(f"\nFull-pool mega-model (N={len(mega_full):,}, distance + δ only):")
print(mega_cox_full.summary.to_string(float_format="{:.4f}".format))

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))

# Panel a: time-varying HR
ax = axes[0]
x = np.arange(3)
hr  = tv_df["HR"].values
lo  = tv_df["HR_lo"].values
hi  = tv_df["HR_hi"].values
ax.errorbar(x, hr, yerr=[hr - lo, hi - hr],
            fmt="s-", color="k", markersize=6, capsize=4, linewidth=1.2)
ax.axhline(1, color="grey", linestyle="--", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(["Early", "Middle", "Late"])
ax.set_ylabel("Hazard ratio\n(per mm distance)")
ax.set_xlabel("Evaporation period")
ax.set_title("a  Time-varying HR", loc="left", fontweight="bold")

# Panel b: mega-model forest plot (full-pool)
ax = axes[1]
summ = mega_cox_full.summary
coefs = summ["coef"]
ci_lo = summ["coef lower 95%"]
ci_hi = summ["coef upper 95%"]
y_pos = np.arange(len(coefs))

pretty = {
    "distance_mm_z":   "Distance",
    "log_R0_um_z":     "log R₀",
    "log_delta_z":     "log δ",
    "dist_x_delta_z":  "Distance × δ",
}

ax.errorbar(coefs, y_pos, xerr=[coefs - ci_lo, ci_hi - coefs],
            fmt="o", color="k", markersize=5, capsize=3, linewidth=1)
ax.axvline(0, color="grey", linestyle="--", linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([pretty.get(c, c) for c in coefs.index])
ax.set_xlabel("Cox β (standardised)")
ax.set_title("b  Mega-model coefficients", loc="left", fontweight="bold")

for yi, idx in enumerate(coefs.index):
    p = summ.loc[idx, "p"]
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    xpos = ci_hi.loc[idx] + 0.015
    ax.text(xpos, yi, stars, va="center", fontsize=7, color="#333333")

# Panel c: interaction visualisation
ax = axes[2]
delta_vals = mega_full["log_delta_z"].quantile([0.1, 0.5, 0.9]).values
beta_dist_z = mega_cox_full.params_["distance_mm_z"]
beta_int    = mega_cox_full.params_["dist_x_delta_z"]

dist_grid = np.linspace(mega_full["distance_mm_z"].min(),
                        mega_full["distance_mm_z"].max(), 100)

for dv, lbl, col in zip(
    delta_vals,
    ["Low δ (10th %ile)", "Mid δ (median)", "High δ (90th %ile)"],
    ["#1b9e77", "#d95f02", "#7570b3"],
):
    log_hr = (beta_dist_z + beta_int * dv) * dist_grid
    ax.plot(dist_grid, np.exp(log_hr), color=col, linewidth=1.3, label=lbl)

ax.axhline(1, color="grey", linestyle="--", linewidth=0.5)
ax.set_xlabel("Distance (standardised)")
ax.set_ylabel("Relative hazard")
ax.set_title("c  δ modulates distance effect", loc="left", fontweight="bold")
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(OUT / "mega_model.svg")
print(f"\n  Saved → {OUT / 'mega_model.svg'}")

print("\n" + "=" * 60)
print("5. CONCORDANCE INDEX COMPARISON")
print("=" * 60)

# Use R0 subset for fair comparison across all models
covariate_sets = {
    "(a) Distance only":       ["distance_mm"],
    "(b) R₀ only":             ["log_R0_um"],
    "(c) Distance + R₀":       ["distance_mm", "log_R0_um"],
    "(d) Distance + R₀ + δ":   ["distance_mm", "log_R0_um", "log_delta"],
}

c_results = []
for label, cols in covariate_sets.items():
    df_fit = pool_R0[["lifetime_min", "event"] + cols].dropna()
    cph = CoxPHFitter()
    cph.fit(df_fit, duration_col="lifetime_min", event_col="event")
    ci = cph.concordance_index_
    c_results.append({"Model": label, "C-index": ci})
    print(f"  {label:30s}  C = {ci:.4f}")

# Also show distance-only on FULL pool
cph_dist_full = CoxPHFitter()
cph_dist_full.fit(pool[["lifetime_min", "event", "distance_mm"]],
                  duration_col="lifetime_min", event_col="event")
print(f"\n  Distance only (full pool, N={len(pool):,}): C = {cph_dist_full.concordance_index_:.4f}")

c_df = pd.DataFrame(c_results)

fig, ax = plt.subplots(figsize=(4.5, 2.8))
colors_c = ["#bdbdbd", "#bdbdbd", "#969696", "#2ca02c"]
bars = ax.barh(c_df["Model"], c_df["C-index"], color=colors_c,
               edgecolor="k", linewidth=0.5)
ax.set_xlabel("Concordance index")
ax.set_xlim(0.5, max(c_df["C-index"]) * 1.08)
ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.5, label="Random (0.5)")

for bar, val in zip(bars, c_df["C-index"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=7)

ax.set_title("Predictive accuracy", fontweight="bold", loc="left")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT / "concordance.svg")
print(f"\n  Saved → {OUT / 'concordance.svg'}")

with open(OUT / "advanced_survival_results.txt", "w") as f:
    f.write("ADVANCED SURVIVAL ANALYSIS — RESULTS SUMMARY\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Total droplets loaded: {len(pool):,} "
            f"({pool['event'].sum():,} events, "
            f"{(pool['event']==0).sum():,} censored)\n")
    f.write(f"Trials with δ: {pool['trial_id'].nunique()}\n")
    f.write(f"Subset with measured R₀: {len(pool_R0):,}\n\n")

    f.write("1. AFT MODEL COMPARISON  (R₀ subset)\n")
    f.write("-" * 40 + "\n")
    f.write(aic_df.to_string(index=False, float_format="{:.1f}".format) + "\n")
    if beta_dist is not None:
        f.write(f"\nBest AFT ({best_aft_name}): each mm further from source "
                f"multiplies median lifetime by {np.exp(beta_dist):.3f}\n")
    f.write(f"\n{best_aft_name} full summary:\n")
    f.write(best_aft.summary.to_string(float_format="{:.4f}".format) + "\n")
    f.write(f"\nCox PH HR for distance: {hr_dist:.4f}\n\n")

    f.write("2. SIZE-STRATIFIED SURVIVAL  (R₀ subset)\n")
    f.write("-" * 40 + "\n")
    f.write(f"Size tercile boundaries: {q33:.1f}, {q66:.1f} μm\n")
    f.write(f"Distance split at median: {d_med:.2f} mm\n\n")

    f.write("3. TIME-VARYING DISTANCE EFFECT  (full pool)\n")
    f.write("-" * 40 + "\n")
    f.write(tv_df.to_string(index=False, float_format="{:.4f}".format) + "\n\n")

    f.write("4. MEGA-MODEL  (full pool, distance + δ)\n")
    f.write("-" * 40 + "\n")
    f.write(f"N = {len(mega_full):,}\n")
    f.write(mega_cox_full.summary.to_string(float_format="{:.4f}".format) + "\n")
    f.write(f"\nMega-model with R₀ (N = {len(mega_df):,}):\n")
    f.write(mega_cox.summary.to_string(float_format="{:.4f}".format) + "\n\n")

    f.write("5. CONCORDANCE INDEX COMPARISON  (R₀ subset)\n")
    f.write("-" * 40 + "\n")
    f.write(c_df.to_string(index=False, float_format="{:.4f}".format) + "\n")
    f.write(f"\nDistance only on full pool (N={len(pool):,}): "
            f"C = {cph_dist_full.concordance_index_:.4f}\n")

print(f"\n  Results text → {OUT / 'advanced_survival_results.txt'}")
print("\nDone.")

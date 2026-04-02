#!/usr/bin/env python3
"""
K_universal.py  --  Evaporation-rate-constant analysis across 30 fungal vapour-sink trials.

Uses the d^2-law:  K = R0^2 / tau   (um^2 / s)
where R0 = R_eq_seed (converted to um) and tau = lifetime_s.

Produces:
    K_universal_vs_delta.{svg,png,pdf}   -- K_near/K_far vs delta
    Kstar_profile.{svg,png,pdf}          -- normalised K*(distance) for all trials
    dKdr_scatter.{svg,png,pdf}           -- dK/dr  vs  delta
    K_ratio_vs_dtau50dr.{svg,png,pdf}    -- predictive comparison
"""

import json, pathlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)

TRACK_DIR  = Path(__file__).resolve().parents[2] / 'FigureHGAggregate' / 'code' / 'test_tracking' / 'output'
CAL_DIRS   = [
    Path(__file__).resolve().parents[2] / 'FigureHGAggregate' / 'raw_data',
    Path(__file__).resolve().parents[2] / 'FigureFungi' / 'raw_data',
]
METRICS    = Path(__file__).resolve().parents[2] / 'FigureTable' / 'output' / 'universal_metrics.csv'
OUT        = Path(__file__).resolve().parents[2] / 'additions' / '4_K_distance_evaporation'
OUT.mkdir(parents=True, exist_ok=True)

GROUP_COLORS = {
    "Agar":  "#2ca02c",
    "0.5:1": "#E67E22",
    "1:1":   "#1f77b4",
    "2:1":   "#d62728",
    "Green": "#2ca02c",
    "White": "#7f7f7f",
    "Black": "#1a1a1a",
}
GROUP_MARKERS = {
    "Agar": "o", "0.5:1": "o", "1:1": "s", "2:1": "D",
    "Green": "^", "White": "v", "Black": "X",
}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica"],
    "font.size":         8,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "lines.linewidth":   1.0,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "mathtext.default":  "regular",
})

def load_calibration(trial: str) -> float:
    """Return um_per_px for *trial*."""
    for d in CAL_DIRS:
        p = d / trial / "calibration.json"
        if p.exists():
            with open(p) as f:
                cal = json.load(f)
            return cal["scale"]["pixel_size_um"]
    raise FileNotFoundError(f"No calibration for {trial}")


def compute_K_for_trial(trial: str, um_per_px: float) -> pd.DataFrame:
    """Return a dataframe with K and distance for each valid droplet."""
    csv = TRACK_DIR / f"{trial}_track_histories.csv"
    df = pd.read_csv(csv)

    # filter: uncensored, meaningful lifetime, enough frames
    df = df[df["death_cause"] != "censored"]
    df = df[df["lifetime_s"] > 30]
    df = df[df["n_frames"] >= 3]
    df = df[df["lifetime_s"] > 0]
    df = df[df["R_eq_seed"] > 0]

    # K = (R0_um)^2 / tau
    R0_um = df["R_eq_seed"] * um_per_px
    K = (R0_um ** 2) / df["lifetime_s"]

    out = pd.DataFrame({
        "distance_um": df["distance_um"].values,
        "R0_um":       R0_um.values,
        "lifetime_s":  df["lifetime_s"].values,
        "K":           K.values,
    })
    return out


def bin_K_vs_distance(df: pd.DataFrame, bin_width=50):
    """Return binned K(distance) with 50 um bins."""
    df = df.copy()
    df["bin"] = (df["distance_um"] // bin_width) * bin_width + bin_width / 2
    grp = df.groupby("bin")["K"].agg(["mean", "median", "count", "sem"])
    grp = grp[grp["count"] >= 3]
    return grp.reset_index()


metrics = pd.read_csv(METRICS)
# keep only trials with delta and track histories
track_trials = {p.stem.replace("_track_histories", "")
                for p in TRACK_DIR.glob("*_track_histories.csv")}
metrics = metrics[metrics["trial_id"].isin(track_trials)].copy()
metrics = metrics.dropna(subset=["delta_um"])
metrics = metrics.set_index("trial_id")

print(f"Analysing {len(metrics)} trials with delta values")

# Near/far zones: use quartile-based definition so every trial has data
# Near = lowest 25% of distances present; Far = highest 25%
FAR_CUTOFF = 1500   # um (fixed)

results = []

for trial in metrics.index:
    try:
        um_px = load_calibration(trial)
        df = compute_K_for_trial(trial, um_px)
        if len(df) < 10:
            print(f"  {trial}: only {len(df)} droplets -- skipping")
            continue

        # Adaptive near zone: lowest-distance quartile of droplets in this trial
        q25 = df["distance_um"].quantile(0.25)
        near = df[df["distance_um"] <= q25]
        far  = df[df["distance_um"] >= FAR_CUTOFF]

        K_near = near["K"].mean() if len(near) >= 3 else np.nan
        K_far  = far["K"].mean()  if len(far) >= 3 else np.nan
        ratio  = K_near / K_far   if (K_far > 0 and np.isfinite(K_near)) else np.nan

        # linear regression of K vs distance (dK/dr)
        slope, intercept, r, p, se = stats.linregress(df["distance_um"], df["K"])

        # binned profile
        binned = bin_K_vs_distance(df)

        results.append({
            "trial":     trial,
            "group":     metrics.loc[trial, "group"],
            "delta_um":  metrics.loc[trial, "delta_um"],
            "dtau50_dr": metrics.loc[trial, "dtau50_dr"],
            "dtau50_dr_sm": metrics.loc[trial, "dtau50_dr_sizematched"],
            "K_near":    K_near,
            "K_far":     K_far,
            "K_ratio":   ratio,
            "near_edge": q25,
            "dK_dr":     slope * 1000,   # per mm
            "dK_dr_r":   r,
            "dK_dr_p":   p,
            "n_drops":   len(df),
            "binned":    binned,
        })
        print(f"  {trial}: n={len(df):4d}  near<={q25:.0f}um  K_near={K_near:6.1f}  "
              f"K_far={K_far:6.1f}  ratio={ratio:.2f}  dK/dr={slope*1000:.3f}")
    except Exception as e:
        print(f"  {trial}: ERROR -- {e}")

res = pd.DataFrame(results)
print(f"\n{len(res)} trials processed successfully.")

summary = res.drop(columns=["binned"]).copy()
summary.to_csv(OUT / "K_summary_all_trials.csv", index=False)
print(f"Summary saved -> {OUT / 'K_summary_all_trials.csv'}")


fig, ax = plt.subplots(figsize=(3.5, 3.0))

valid = res.dropna(subset=["K_ratio", "delta_um"])

for grp, sub in valid.groupby("group"):
    ax.scatter(sub["delta_um"], sub["K_ratio"],
               c=GROUP_COLORS.get(grp, "gray"),
               marker=GROUP_MARKERS.get(grp, "o"),
               s=40, edgecolors="k", linewidths=0.4,
               label=grp, zorder=5)

# regression
x, y = valid["delta_um"].values, valid["K_ratio"].values
mask = np.isfinite(x) & np.isfinite(y)
x, y = x[mask], y[mask]
sl, ic, rr, pp, _ = stats.linregress(x, y)
xfit = np.linspace(x.min(), x.max(), 100)
ax.plot(xfit, sl * xfit + ic, "k--", lw=0.8, zorder=4)
ax.text(0.97, 0.05,
        f"$R^2 = {rr**2:.2f}$\n$P = {pp:.1e}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7)

ax.set_xlabel(r"Vapour-sink strength $\delta$ ($\mu$m)")
ax.set_ylabel(r"$K_{\mathrm{near}}$ / $K_{\mathrm{far}}$")
ax.legend(fontsize=6, frameon=False, loc="upper left")
ax.axhline(1, ls=":", c="gray", lw=0.5)

for ext in ("svg", "png", "pdf"):
    fig.savefig(OUT / f"K_universal_vs_delta.{ext}")
plt.close(fig)
print("Saved K_universal_vs_delta")


fig, ax = plt.subplots(figsize=(3.5, 3.0))

for grp, sub in res.groupby("group"):
    ax.scatter(sub["delta_um"], sub["dK_dr"],
               c=GROUP_COLORS.get(grp, "gray"),
               marker=GROUP_MARKERS.get(grp, "o"),
               s=40, edgecolors="k", linewidths=0.4,
               label=grp, zorder=5)

x, y = res["delta_um"].values, res["dK_dr"].values
mask = np.isfinite(x) & np.isfinite(y)
x, y = x[mask], y[mask]
sl, ic, rr, pp, _ = stats.linregress(x, y)
xfit = np.linspace(x.min(), x.max(), 100)
ax.plot(xfit, sl * xfit + ic, "k--", lw=0.8, zorder=4)
ax.text(0.97, 0.05,
        f"$R^2 = {rr**2:.2f}$\n$P = {pp:.1e}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7)

ax.set_xlabel(r"Vapour-sink strength $\delta$ ($\mu$m)")
ax.set_ylabel(r"d$K$/d$r$  ($\mu$m$^2$ s$^{-1}$ mm$^{-1}$)")
ax.legend(fontsize=6, frameon=False, loc="upper left")
ax.axhline(0, ls=":", c="gray", lw=0.5)

for ext in ("svg", "png", "pdf"):
    fig.savefig(OUT / f"dKdr_scatter.{ext}")
plt.close(fig)
print("Saved dKdr_scatter")


fig, ax = plt.subplots(figsize=(4.5, 3.0))

for _, row in res.iterrows():
    if np.isnan(row["K_far"]) or row["K_far"] <= 0:
        continue
    b = row["binned"]
    Kstar = b["mean"] / row["K_far"]
    ax.plot(b["bin"], Kstar,
            color=GROUP_COLORS.get(row["group"], "gray"),
            alpha=0.35, lw=0.8)

# Grand mean overlay
all_bins = []
for _, row in res.iterrows():
    if np.isnan(row["K_far"]) or row["K_far"] <= 0:
        continue
    b = row["binned"].copy()
    b["Kstar"] = b["mean"] / row["K_far"]
    all_bins.append(b[["bin", "Kstar"]])

if all_bins:
    big = pd.concat(all_bins)
    grand = big.groupby("bin")["Kstar"].agg(["mean", "sem", "count"])
    grand = grand[grand["count"] >= 3]
    ax.plot(grand.index, grand["mean"], "k-", lw=1.8, zorder=10, label="Grand mean")
    ax.fill_between(grand.index,
                    grand["mean"] - grand["sem"],
                    grand["mean"] + grand["sem"],
                    color="k", alpha=0.12, zorder=9)

ax.axhline(1, ls=":", c="gray", lw=0.5)
ax.set_xlabel(r"Distance from source ($\mu$m)")
ax.set_ylabel(r"$K^* = K\,/\,K_{\mathrm{far}}$")
ax.set_xlim(0, 3500)

# legend: one entry per group
handles = []
for grp in ["Agar", "0.5:1", "1:1", "2:1", "Green", "White", "Black"]:
    if grp in res["group"].values:
        handles.append(Line2D([0], [0], color=GROUP_COLORS[grp], lw=1.2, label=grp))
handles.append(Line2D([0], [0], color="k", lw=1.8, label="Grand mean"))
ax.legend(handles=handles, fontsize=6, frameon=False, loc="upper right")

for ext in ("svg", "png", "pdf"):
    fig.savefig(OUT / f"Kstar_profile.{ext}")
plt.close(fig)
print("Saved Kstar_profile")


fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

for ax_idx, (col, label) in enumerate([
    ("dtau50_dr", r"d$\tau_{50}$/d$r$ (s mm$^{-1}$)"),
    ("dtau50_dr_sm", r"d$\tau_{50}$/d$r$ size-matched (s mm$^{-1}$)")
]):
    ax = axes[ax_idx]
    valid2 = res.dropna(subset=["K_ratio", col])
    if len(valid2) < 5:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        continue

    for grp, sub in valid2.groupby("group"):
        ax.scatter(sub["K_ratio"], sub[col],
                   c=GROUP_COLORS.get(grp, "gray"),
                   marker=GROUP_MARKERS.get(grp, "o"),
                   s=36, edgecolors="k", linewidths=0.4,
                   label=grp, zorder=5)

    x2, y2 = valid2["K_ratio"].values, valid2[col].values
    mask2 = np.isfinite(x2) & np.isfinite(y2)
    x2, y2 = x2[mask2], y2[mask2]
    if len(x2) >= 3:
        sl2, ic2, rr2, pp2, _ = stats.linregress(x2, y2)
        xfit2 = np.linspace(x2.min(), x2.max(), 100)
        ax.plot(xfit2, sl2 * xfit2 + ic2, "k--", lw=0.8)
        ax.text(0.97, 0.05,
                f"$R^2 = {rr2**2:.2f}$\n$P = {pp2:.1e}$",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7)

    ax.set_xlabel(r"$K_{\mathrm{near}}$ / $K_{\mathrm{far}}$")
    ax.set_ylabel(label)
    if ax_idx == 0:
        ax.legend(fontsize=5.5, frameon=False, loc="upper left")

fig.tight_layout(w_pad=2)
for ext in ("svg", "png", "pdf"):
    fig.savefig(OUT / f"K_ratio_vs_dtau50dr.{ext}")
plt.close(fig)
print("Saved K_ratio_vs_dtau50dr")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Trials analysed: {len(res)}")

v = res.dropna(subset=["K_ratio"])
print(f"Trials with K_ratio: {len(v)}")
print(f"Mean K_near / K_far: {v['K_ratio'].mean():.2f} +/- {v['K_ratio'].sem():.2f}")
print(f"K_ratio range: [{v['K_ratio'].min():.2f}, {v['K_ratio'].max():.2f}]")
print(f"K_ratio < 1 in {(v['K_ratio'] < 1).sum()}/{len(v)} trials "
      f"(one-sample t-test vs 1: t={stats.ttest_1samp(v['K_ratio'], 1).statistic:.2f}, "
      f"P={stats.ttest_1samp(v['K_ratio'], 1).pvalue:.2e})")

# K_ratio vs delta
v1 = res.dropna(subset=["K_ratio", "delta_um"])
sl, ic, rr, pp, _ = stats.linregress(v1["delta_um"], v1["K_ratio"])
print(f"\nK_ratio vs delta:  R^2 = {rr**2:.3f},  P = {pp:.2e},  slope = {sl:.4f}")

# dK/dr vs delta
v2 = res.dropna(subset=["dK_dr", "delta_um"])
sl2, _, rr2, pp2, _ = stats.linregress(v2["delta_um"], v2["dK_dr"])
print(f"dK/dr  vs delta:   R^2 = {rr2**2:.3f},  P = {pp2:.2e}")

# dK/dr positive in how many trials?
print(f"dK/dr > 0 in {(res['dK_dr'] > 0).sum()}/{len(res)} trials "
      f"(one-sample t-test vs 0: t={stats.ttest_1samp(res['dK_dr'], 0).statistic:.2f}, "
      f"P={stats.ttest_1samp(res['dK_dr'], 0).pvalue:.2e})")

# K_ratio vs dtau50/dr
for col, name in [("dtau50_dr", "dtau50/dr"), ("dtau50_dr_sm", "dtau50/dr (size-matched)")]:
    v3 = res.dropna(subset=["K_ratio", col])
    if len(v3) >= 3:
        sl3, _, rr3, pp3, _ = stats.linregress(v3["K_ratio"], v3[col])
        print(f"K_ratio vs {name}:  R^2 = {rr3**2:.3f},  P = {pp3:.2e}")

print(f"\nAll outputs saved to {OUT}")

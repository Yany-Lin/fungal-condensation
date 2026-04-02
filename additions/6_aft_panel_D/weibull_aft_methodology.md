# Weibull AFT Model — Methodology for Droplet Survival Analysis

*A guide for lab members unfamiliar with survival analysis.*

---

## 1. What problem are we solving?

We tracked individual condensation droplets and recorded **how long each droplet survived** before evaporating. We want to know:

> *Do droplets farther from the vapor-absorbing source (hydrogel or fungus) survive longer? And does that gradient get stronger as the source gets drier?*

The naive approach — just take the mean lifetime at each distance — fails for two reasons:

1. **Censoring.** Many droplets are still alive when we stop recording. We know they survived *at least* until the end of the experiment, but not how much longer. Ignoring these droplets biases the mean downward.

2. **Confounding by size.** Larger droplets take longer to evaporate regardless of distance, because they have more water. If larger droplets happen to concentrate at certain distances, that creates a fake survival gradient.

Survival analysis is the branch of statistics built specifically to handle censoring. The Weibull AFT model handles both problems simultaneously.

---

## 2. The d²-law: why Weibull AFT is the right model

Before the statistics, the physics. A spherical droplet evaporating by diffusion obeys the **d²-law**:

```
R(t)² = R₀² − K · t
```

where R₀ is the initial radius, K is the evaporation rate constant (µm²/s), and t is time.

The droplet disappears when R(t) = 0, so its lifetime is:

```
T = R₀² / K(d)
```

where K(d) is the local evaporation rate at distance d from the source boundary. Taking logs:

```
log(T) = 2·log(R₀) − log(K(d))
       = α + β_d · d + β_R · log(R₀) + noise
```

This is exactly the **Accelerated Failure Time (AFT)** structure: log-lifetime is a linear function of covariates. The physics *derives* the model; it is not an assumption.

---

## 3. The Weibull AFT model, written out

The model we fit per trial is:

```
log(T_fwd) = α + β_d · d_mm + β_R · log(R₀_µm) + σ · ε
```

| Symbol | Meaning |
|--------|---------|
| `T_fwd` | Forward lifetime from t = 15 min (minutes) |
| `d_mm` | Distance from source boundary (mm) |
| `R₀_µm` | Initial droplet radius at t = 15 min (µm) |
| `α` | Intercept (log-lifetime when d = 0, R₀ = 1 µm) |
| `β_d` | Effect of distance on log-lifetime |
| `β_R` | Effect of log-size on log-lifetime |
| `σ` | Scale parameter (noise/variability) |
| `ε` | Error term with a **Gumbel** (extreme-value) distribution |

The Gumbel distribution for ε is what makes this **Weibull**: it gives the Weibull distribution for T itself. Weibull is the most physically motivated choice for evaporation because the failure rate (hazard) is allowed to change over time (unlike exponential, which assumes constant hazard).

### What β_d means

Because the model is on the log scale:

```
β_d > 0   →   farther droplets have longer lifetime
```

Specifically, moving 1 mm further from the source **multiplies median lifetime by exp(β_d)**. For our 2:1 NaCl data, β_d ≈ 0.46 (size-corrected), so:

```
exp(0.46) ≈ 1.58  →  each additional mm adds 58% to median lifetime
```

This is directly interpretable as a reduction in the local evaporation rate K(d).

### What β_R means

The d²-law predicts `log(T) = 2·log(R₀) + const`, so we expect β_R ≈ 2. In practice β_R is somewhat less than 2 because not all droplets evaporate cleanly (coalescence, condensation during recording, edge effects), but it should be positive and on the order of 1–2.

---

## 4. Handling censoring

A droplet is **censored** if:
- It was still alive at the end of the experiment
- It merged with another droplet (coalescence — we lose track of it)

Censored droplets get a flag `event = 0`; evaporated droplets get `event = 1`.

The Weibull AFT likelihood function handles these differently:

- **Event (event = 1):** contributes the probability *density* at the observed lifetime: f(T)
- **Censored (event = 0):** contributes the probability of *surviving at least* to the observed time: S(T) = P(lifetime > T)

By using both, the model extracts maximum information from every droplet, rather than throwing away the censored ones.

---

## 5. The "total" vs "size-corrected" β_d

We report two versions of β_d per trial:

**Total (β_d from distance-only model):**
```
log(T) = α + β_d · d_mm + σε
```
This is the raw distance gradient, including any confounding from size.

**Size-corrected (β_d from distance + log(R₀) model):**
```
log(T) = α + β_d · d_mm + β_R · log(R₀) + σε
```
This is the distance gradient *holding initial size constant*. It answers: "if two droplets had the same size at t = 15 min but lived at different distances, how much longer does the far one survive?"

The size-corrected β_d is the more rigorous metric because it strips out the confounding effect of a possible size-distance correlation. In our data, size-corrected R² (0.882) > total R² (0.809), confirming that controlling for size *sharpens* the signal.

---

## 6. What is AIC and why does Weibull win?

We compared three AFT distributional assumptions using **AIC (Akaike Information Criterion)**:

```
AIC = −2 · log-likelihood + 2 · (number of parameters)
```

Lower AIC = better. The penalty term (2k) prevents the model from just overfitting by adding parameters. ΔAIC > 10 is considered decisive evidence.

| Model | ΔAIC |
|-------|------|
| Weibull AFT | **0** (best) |
| Log-Logistic AFT | 3400 |
| Log-Normal AFT | 9718 |

Weibull wins decisively. This is consistent with the physics: the Weibull distribution's hazard rate `h(t) ∝ t^(ρ−1)` is a power law in time, which is what you expect for evaporation where the evaporation rate slows as the droplet shrinks (because the surface area decreases).

---

## 7. Why not Cox Proportional Hazards (the usual choice)?

Cox PH is the most commonly used survival model in biology. It assumes:

```
h(t | x) = h₀(t) · exp(β · x)
```

meaning the *ratio* of hazard rates between any two droplets is constant over time. For our droplets, this is **not true**. The distance effect on hazard ratio changes 3× over the experiment (HR = 0.29 early → 0.87 late), because as droplets shrink, the absolute difference in evaporation rate between near and far droplets becomes less consequential.

The AFT model doesn't require the hazard ratio to be constant. It just says log-lifetime is linear in covariates, which the d²-law guarantees.

Cox PH is still a valid rank-based comparison and gives directionally correct results. We use it in the main figure (Panel D) as a cross-check. But the Weibull AFT is more rigorous and gives higher R² (0.88 vs 0.81).

---

## 8. Step-by-step: what the code does

**File:** `additions/6_aft_panel_D/aft_panel_D.py`

For each trial:

1. **Load** `{trial_id}_track_histories.csv` — one row per tracked droplet
2. **Compute** `τ_fwd = (t_death_s − 900) / 60` — forward lifetime in minutes from the t = 15 min seed frame
3. **Convert** `R_eq_seed` from pixels to µm using `calibration.json`
4. **Filter** droplets with `n_frames ≥ 3`, `τ_fwd > 0`, `R₀ > 0` pixels
5. **Fit** `WeibullAFTFitter` from the `lifelines` library, two models:
   - Total: `[τ_fwd, event, d_mm]`
   - Adjusted: `[τ_fwd, event, d_mm, log_R₀_µm]`
6. **Extract** `β_d = params_[('lambda_', 'd_mm')]` — the distance coefficient
7. **Collect** per-trial (δ, β_d) pairs and regress across all 35 trials

---

## 9. The Panel D scatter plot

Each point is one trial. The x-axis is δ (µm), the average distance from the source boundary to the nearest droplet — a physical measure of how much the source suppresses condensation. The y-axis is the Weibull AFT β_d — how steeply survival increases with distance.

- **Black solid line:** linear regression, total β_d vs δ (R² = 0.81)
- **Red dashed line:** linear regression, size-corrected β_d vs δ (R² = 0.88)
- **Hydrogels (circles):** 20 trials across 4 water activity conditions
- **Fungi (diamonds):** 15 trials across 3 species

The slope of the regression is not arbitrary: by the d²-law and the physical definition of δ, stronger vapor sinks (lower a_w) produce larger δ AND steeper survival gradients, and the model recovers this.

---

## 10. Reporting in the Methods section

A suggested paragraph:

> "Per-trial survival gradients were quantified using a Weibull Accelerated Failure Time (AFT) model, which is mechanistically grounded in the d²-law of diffusive evaporation. For each trial, droplet forward lifetime τ (measured from t = 15 min) was modeled as log(τ) = α + β_d · d + β_R · log(R₀) + σε, where d is distance from the source boundary (mm), R₀ is initial radius at t = 15 min (µm), and ε follows a Gumbel distribution. Right-censoring was applied to droplets still present at experiment end and to droplets lost through coalescence. Models were fit using the `lifelines` package (v0.27) in Python. The distance coefficient β_d quantifies the fold-change in median lifetime per mm of distance: exp(β_d) is the lifetime multiplier. The size-corrected β_d (from the two-covariate model) was used as the primary per-trial gradient metric. The Weibull AFT was selected over Cox PH based on ΔAIC = 3400 (pooled across all trials) and its direct correspondence with the d²-law. AFT β_d values were then regressed against per-trial δ (the vapor-sink strength metric) across all 35 lab trials (20 hydrogel, 15 fungal)."

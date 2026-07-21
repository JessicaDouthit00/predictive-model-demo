# Predicting Patient Label from Systolic Blood Pressure Data

## 1. Data overview

Two files were provided:

- `fake_LCICM_bp_data.csv` — 859 rows of `(pat_id, timestamp, blood_pressure)`, i.e. irregularly-sampled
  systolic BP readings for 250 distinct patients (1–6 readings each, timestamps not evenly spaced).
- `fake_LCICM_labels.csv` — a binary `label` (0/1) for 300 patients. Labels are imbalanced: **200 label=0,
  100 label=1** (2:1) across the full label file.

## 2. Data quality issues found and how they were handled

| Issue | Finding | Decision |
|---|---|---|
| Patients with labels but **no BP rows at all** | 50 of 300 labeled patients (16.7%) have zero entries in the BP file | **Excluded from modeling.** There is no BP signal to learn from for these patients, so imputing them with population averages would only inject noise/bias. Their label balance (15 label=1 / 35 label=0) is similar to the rest of the data, suggesting missingness is not strongly informative — but this can't be confirmed with certainty, so it's flagged here rather than silently assumed. |
| Missing readings (`blood_pressure` is null) | 10 of 859 rows | **Dropped** those individual rows before aggregation; the timestamp is present but the value is not, so nothing can be computed from them. |
| Duplicate `(pat_id, timestamp)` pair | 1 case (patient 104, two different BP values logged at the identical timestamp) | **Kept both** as separate readings. Since features are aggregated per patient (mean/min/max/etc.), this has a negligible effect and isn't worth discarding data over — but it's worth knowing about for anyone auditing the raw file. |
| Extreme values (range: 30–197 mmHg) | Some values are physiologically implausible as isolated systolic readings (e.g., 30 mmHg) | **Not clipped or removed.** This is synthetic data with no metadata to distinguish "measurement error" from "intentionally extreme value," so arbitrarily filtering on a clinical plausibility threshold risks removing signal the label was designed around. This is flagged as a limitation, not silently fixed. |

**Net modeling set: 249 patients** (165 label=0 / 84 label=1 — same ~2:1 imbalance as the full label set,
confirming the exclusions didn't skew the class balance).

## 3. Feature engineering

Because each patient has a variable-length, irregularly-timed sequence of BP readings (not a fixed
number of visits), each patient's readings were collapsed into a fixed-length feature vector rather than
feeding raw sequences into a model — this is the right shape for a small tabular dataset (249 patients)
and keeps the model interpretable. Features per patient:

- `n_readings` — count of valid readings
- `mean_bp`, `std_bp`, `min_bp`, `max_bp`, `range_bp` — summary statistics of the reading distribution
- `first_bp`, `last_bp`, `delta_first_last` — earliest/latest reading and their difference (crude trend)
- `slope_bp` — OLS slope of BP vs. timestamp (trend over time, 0 if a patient has only 1 reading)
- `cv_bp` — coefficient of variation (std/mean), a scale-free measure of BP volatility
- `pct_ge_130`, `pct_ge_140` — fraction of a patient's readings at/above standard clinical thresholds
  (AHA "elevated"/Stage-1 and Stage-2 hypertension cutoffs), added as domain-informed features rather
  than relying purely on raw summary stats

All 13 features are computed only from valid (non-null) readings for that patient.

## 4. Exploratory findings

Label=1 patients show clearly higher and more variable BP than label=0 patients:

| | label=0 | label=1 |
|---|---|---|
| mean of per-patient mean BP | 97.0 | 111.2 |
| mean of per-patient max BP | 124.0 | 148.7 |
| mean of per-patient BP range | 52.5 | 76.1 |
| mean n_readings | 3.62 | 2.99 |

This is a genuine, sizeable separation (not just noise) — see `figures/mean_bp_by_label.png` and
`figures/bp_distribution.png`. There's also a mild association between label and `n_readings` (label=1
patients tend to have slightly fewer readings); this is noted in case it reflects something about how
the data/labels were generated, since it's a pattern the model does end up using.

## 5. Modeling approach

Given only 249 patients, a single train/test split would give a noisy, unreliable performance estimate,
so **repeated stratified 5-fold cross-validation (20 repeats = 100 folds total)** was used to get stable
estimates of out-of-sample performance. All features were standardized before Logistic Regression.
Class imbalance (2:1) was handled via `class_weight="balanced"` rather than resampling, to avoid
distorting the small dataset further.

Four models were compared:

1. **Majority-class baseline** — sanity check floor.
2. **Logistic Regression** (L2-regularized, balanced classes) — primary model, interpretable coefficients.
3. **Random Forest** (shallow, `max_depth=4`, regularized via `min_samples_leaf`) — nonlinear comparison.
4. **Gradient Boosting** (shallow trees, low learning rate) — nonlinear comparison.

## 6. Results

Repeated stratified 5-fold CV (mean ± std across 100 folds):

| Model | ROC-AUC | Avg. Precision (PR-AUC) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Majority-class baseline | 0.500 ± 0.000 | 0.337 ± 0.005 | 0.663 ± 0.005 | 0.000 | 0.000 | 0.000 |
| **Logistic Regression** | **0.820 ± 0.048** | 0.746 ± 0.072 | 0.763 ± 0.051 | 0.658 ± 0.094 | 0.656 ± 0.107 | 0.650 ± 0.075 |
| Random Forest | 0.867 ± 0.045 | 0.847 ± 0.045 | 0.841 ± 0.037 | 0.859 ± 0.095 | 0.645 ± 0.093 | 0.730 ± 0.066 |
| **Gradient Boosting** | **0.880 ± 0.049** | **0.865 ± 0.051** | **0.859 ± 0.041** | 0.900 ± 0.088 | 0.663 ± 0.106 | 0.757 ± 0.076 |

All three real models clear the 0.5-AUC baseline by a wide margin, confirming BP data carries real
predictive signal for the label. Tree-based models slightly outperform logistic regression, suggesting
some non-linear/threshold-like structure (consistent with the `pct_ge_140`-style features mattering).

**Out-of-fold confusion matrix** (Logistic Regression, threshold=0.5, `figures/confusion_matrix.png`):

```
              Pred 0   Pred 1
True 0          135       30
True 1           29       55
```

**ROC curve:** `figures/roc_curve.png`

**Logistic Regression coefficients** (standardized features, fit on full data — see
`figures/feature_coefficients.png` and `logreg_coefficients.csv`): the strongest positive drivers of
label=1 are `range_bp`, `pct_ge_140`, and `max_bp`; the strongest negative driver is `n_readings`, echoing
the exploratory finding above. `mean_bp` itself carries a small standardized coefficient once its
correlated cousins (`max_bp`, `range_bp`) are already in the model — a sign of multicollinearity among the
summary-stat features rather than mean BP being unimportant (it's very informative univariately, per the
exploratory table above).

## 7. Limitations & things I'd flag before using this for anything real

- **Small sample (249 patients).** CV estimates are much more stable than a single holdout split, but
  the confidence intervals on these metrics are still wide; treat the AUC ~0.82–0.88 range as "clearly
  better than chance," not as a precise number.
- **50 patients with no BP data can't be scored by this model at all** — a deployment consideration, not
  just a training-time one.
- **Extreme BP values weren't filtered.** If some of the 30–40 mmHg or 190+ mmHg readings are actually
  data-entry errors rather than intentional signal, that would change feature values for a handful of
  patients.
- **The `n_readings` association with label** is worth a second look before trusting this model further —
  it's the kind of thing that can be a genuine clinical pattern (e.g., sicker patients seen/measured less)
  or an artifact of how the synthetic labels/data were generated. I flagged it rather than either ignoring
  it or building a whole causal story around it.
- Feature multicollinearity (mean/min/max/range all correlated) makes individual logistic-regression
  coefficients harder to interpret in isolation; the tree-based models are less affected by this but are
  correspondingly less directly interpretable.

## Files in this submission

- `analysis.py` — full, runnable pipeline (load → clean → feature-engineer → model → evaluate → plots)
- `report.md` — this document
- `features_used_for_modeling.csv` — the 249×15 feature table actually used for training/evaluation
- `logreg_coefficients.csv` — standardized logistic regression coefficients
- `metrics_summary.json` — full CV metrics for all four models
- `figures/` — all plots referenced above

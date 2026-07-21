"""
LCICM Blood-Pressure -> Label Prediction
==========================================
End-to-end pipeline: load, clean, engineer patient-level features from
irregular longitudinal systolic BP readings, train/evaluate classifiers,
and save all figures + metrics used in the accompanying report.

Run: python3 analysis.py
Outputs land in ./figures/ and ./metrics/
"""

import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve, RocCurveDisplay,
)

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)

FIG_DIR = "figures"
import os
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
bp = pd.read_csv("/mnt/user-data/uploads/fake_LCICM_bp_data.csv")
labels = pd.read_csv("/mnt/user-data/uploads/fake_LCICM_labels.csv")

print(f"Raw BP rows: {len(bp)}  | Raw labeled patients: {len(labels)}")
print(f"BP columns: {list(bp.columns)} | dtypes:\n{bp.dtypes}")

# ---------------------------------------------------------------------------
# 2. DATA QUALITY / CLEANING DECISIONS (see report.md for full rationale)
# ---------------------------------------------------------------------------
n_missing_bp_values = bp["blood_pressure"].isna().sum()
n_dup_ts = bp.duplicated(subset=["pat_id", "timestamp"]).sum()
patients_with_any_bp = set(bp["pat_id"].unique())
patients_labeled = set(labels["pat_id"].unique())
no_bp_patients = sorted(patients_labeled - patients_with_any_bp)

print(f"\nRows with missing blood_pressure value: {n_missing_bp_values} (dropped)")
print(f"Duplicate (pat_id,timestamp) pairs: {n_dup_ts} (kept as separate readings; noted in report)")
print(f"Labeled patients with NO bp readings at all: {len(no_bp_patients)} (excluded from modeling per decision)")

# Drop rows where the reading itself is null -- can't use a reading with no value.
bp_clean = bp.dropna(subset=["blood_pressure"]).copy()

# Sanity-check plausible physiological range for systolic BP.
# Values are inspected but NOT clipped/removed: this is fake/synthetic data with no
# accompanying provenance to justify treating extremes as measurement error.
lo, hi = bp_clean["blood_pressure"].min(), bp_clean["blood_pressure"].max()
print(f"Systolic BP range after cleaning: [{lo}, {hi}]")

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING (patient-level, from irregular longitudinal readings)
# ---------------------------------------------------------------------------
def slope(sub: pd.DataFrame) -> float:
    """OLS slope of blood_pressure vs timestamp; 0.0 if <2 points."""
    if len(sub) < 2:
        return 0.0
    x = sub["timestamp"].values.astype(float)
    y = sub["blood_pressure"].values.astype(float)
    x = (x - x.mean())
    denom = (x ** 2).sum()
    if denom == 0:
        return 0.0
    return float((x * (y - y.mean())).sum() / denom)


def build_features(sub: pd.DataFrame) -> pd.Series:
    sub = sub.sort_values("timestamp")
    vals = sub["blood_pressure"].values.astype(float)
    n = len(vals)
    feats = {
        "n_readings": n,
        "mean_bp": vals.mean(),
        "std_bp": vals.std(ddof=0) if n > 1 else 0.0,
        "min_bp": vals.min(),
        "max_bp": vals.max(),
        "range_bp": vals.max() - vals.min(),
        "first_bp": vals[0],
        "last_bp": vals[-1],
        "delta_first_last": vals[-1] - vals[0],
        "slope_bp": slope(sub),
        "cv_bp": (vals.std(ddof=0) / vals.mean()) if vals.mean() != 0 and n > 1 else 0.0,
        "pct_ge_130": float(np.mean(vals >= 130)),  # elevated (AHA stage 1+) threshold
        "pct_ge_140": float(np.mean(vals >= 140)),  # stage 2 hypertension threshold
    }
    return pd.Series(feats)


feat_df = bp_clean.groupby("pat_id").apply(build_features)
feat_df.index.name = "pat_id"
feat_df = feat_df.reset_index()

data = labels.merge(feat_df, on="pat_id", how="inner")  # inner -> drops the 50 no-BP patients
print(f"\nModeling dataset: {len(data)} patients "
      f"(dropped {len(labels) - len(data)} with no BP readings, per agreed decision)")
print("Label balance in modeling set:")
print(data["label"].value_counts())

data.to_csv("features_used_for_modeling.csv", index=False)

FEATURE_COLS = [
    "n_readings", "mean_bp", "std_bp", "min_bp", "max_bp", "range_bp",
    "first_bp", "last_bp", "delta_first_last", "slope_bp", "cv_bp",
    "pct_ge_130", "pct_ge_140",
]
X = data[FEATURE_COLS].values
y = data["label"].values

# ---------------------------------------------------------------------------
# 4. EXPLORATORY PLOTS
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(bp_clean["blood_pressure"], bins=30, color="#4C72B0", edgecolor="white")
plt.xlabel("Systolic BP (mmHg)")
plt.ylabel("Count (readings)")
plt.title("Distribution of raw systolic BP readings")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/bp_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
for lab, grp in data.groupby("label"):
    plt.hist(grp["mean_bp"], bins=15, alpha=0.6, label=f"label={lab}")
plt.xlabel("Per-patient mean systolic BP")
plt.ylabel("Count (patients)")
plt.title("Mean BP per patient, by label")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/mean_bp_by_label.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
counts = data.groupby(["n_readings", "label"]).size().unstack(fill_value=0)
counts.plot(kind="bar", stacked=True, ax=plt.gca(), color=["#4C72B0", "#DD8452"])
plt.xlabel("Number of BP readings for patient")
plt.ylabel("Number of patients")
plt.title("Readings-per-patient vs label")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/n_readings_by_label.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. MODELING
# ---------------------------------------------------------------------------
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=RNG)

models = {
    "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=RNG)),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=5,
        class_weight="balanced", random_state=RNG),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=2, learning_rate=0.05, random_state=RNG),
}

scoring = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

results = {}
for name, model in models.items():
    cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    results[name] = {f"{k}_mean": float(np.mean(v)) for k, v in cv_res.items() if k.startswith("test_")}
    results[name].update({f"{k}_std": float(np.std(v)) for k, v in cv_res.items() if k.startswith("test_")})

print("\n=== Repeated Stratified 5-fold CV (20 repeats) ===")
for name, r in results.items():
    print(f"\n{name}")
    for metric in scoring:
        print(f"  {metric:20s}: {r[f'test_{metric}_mean']:.3f} +/- {r[f'test_{metric}_std']:.3f}")

with open("metrics_summary.json", "w") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------------------
# 6. Out-of-fold predictions for the primary model (Logistic Regression)
#    -> confusion matrix + ROC curve on held-out folds (single 5-fold pass)
# ---------------------------------------------------------------------------
single_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=RNG)
logreg = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=RNG)),
])
oof_proba = cross_val_predict(logreg, X, y, cv=single_cv, method="predict_proba")[:, 1]
oof_pred = (oof_proba >= 0.5).astype(int)

cm = confusion_matrix(y, oof_pred)
print("\nOut-of-fold confusion matrix (Logistic Regression, threshold=0.5):")
print(cm)

plt.figure(figsize=(4.5, 4))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.xticks([0, 1], ["Pred 0", "Pred 1"])
plt.yticks([0, 1], ["True 0", "True 1"])
plt.title("Out-of-fold Confusion Matrix (Logistic Regression)")
plt.colorbar()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y, oof_proba)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"LogReg (AUC={roc_auc_score(y, oof_proba):.3f})", color="#4C72B0")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Out-of-fold ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/roc_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. Coefficients (fit on full data) for interpretability
# ---------------------------------------------------------------------------
logreg.fit(X, y)
coefs = logreg.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({"feature": FEATURE_COLS, "coefficient": coefs}).sort_values(
    "coefficient", key=np.abs, ascending=False
)
coef_df.to_csv("logreg_coefficients.csv", index=False)
print("\nLogistic Regression coefficients (standardized features), full-data fit:")
print(coef_df.to_string(index=False))

plt.figure(figsize=(6, 4.5))
plt.barh(coef_df["feature"], coef_df["coefficient"], color="#4C72B0")
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Standardized coefficient")
plt.title("Logistic Regression feature coefficients")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/feature_coefficients.png", dpi=150)
plt.close()

print("\nDone. Figures in ./figures, metrics in metrics_summary.json, "
      "coefficients in logreg_coefficients.csv, modeling data in features_used_for_modeling.csv")

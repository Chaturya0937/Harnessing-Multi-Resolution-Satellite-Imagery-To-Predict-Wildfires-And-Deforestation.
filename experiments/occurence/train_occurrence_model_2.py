"""
train_occurrence_model_2.py
============================
Improved wildfire occurrence model. Key improvements over v1:

Problems fixed from v1:
  1. Early stopping at iteration 1 (3 of 5 folds) -- caused by
     min_data_in_leaf=50 being too large for a ~0.5% positive rate.
     Fixed: min_data_in_leaf=10, num_leaves=63.

  2. 4 out of 9 features had zero gain. Fixed by lowering min_child_samples
     and adding 'region' as a categorical feature.

  3. Severe class imbalance (~0.54% positive rate). Fixed by:
     - Using is_unbalance=True (more stable than manual scale_pos_weight),
     - Evaluating with PR-AUC in addition to ROC-AUC.

  4. Heavy regularisation combined with lr=0.03 prevented learning.
     Fixed: learning_rate=0.05, lambda_l2=0.5, min_gain_to_split=0.0.

  5. Best model was picked per-fold, not refitted on all data.
     Fixed: after CV, refit on full training data with chronological holdout.

Install:
    pip install lightgbm pandas scikit-learn matplotlib
"""

import os
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH   = r"C:\Users\Vardhan\Desktop\Coding\Minor Project\experiments\occurence\wildfire_occurrence_global.csv"
LABEL_COL  = "fire_label"
DATE_COL   = "date"
REGION_COL = "region"
DROP_COLS  = ["system:index", ".geo"]

FEATURE_COLS = [
    "temp_mean_k", "wind_speed", "precip_total_m", "dewpoint_k",
    "ndvi", "landcover", "slope_deg", "aspect_deg", "elevation_m",
    "region",   # categorical -- encodes biome / climate zone
]

REGION_SEASON_MONTHS = {
    "central_india":       [2, 3, 4, 5],
    "california":          [7, 8, 9, 10],
    "southeast_australia": [11, 12, 1, 2],
    "mediterranean":       [6, 7, 8, 9],
    "amazon_arc_of_fire":  [7, 8, 9, 10],
    "southern_africa":     [6, 7, 8, 9],
    "western_siberia":     [5, 6, 7, 8],
    "indonesia_sumatra":   [7, 8, 9, 10],
    "british_columbia":    [6, 7, 8, 9],
    "iberia":              [6, 7, 8, 9],
}

MODEL_OUT_PATH = "./models/wildfire_occurrence_lgbm_v2.txt"
N_FOLDS = 5

# LightGBM hyperparameters (v2) -- see docstring for rationale
LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            ["auc", "average_precision"],
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_data_in_leaf":  10,    # was 50; too large for 0.5% positive rate
    "max_depth":         6,
    "is_unbalance":      True,  # replaces manual scale_pos_weight
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "lambda_l1":         0.1,
    "lambda_l2":         0.5,   # was 1.0
    "min_gain_to_split": 0.0,
    "verbose":           -1,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    num_features = [c for c in FEATURE_COLS if c != REGION_COL]
    df = df.dropna(subset=num_features + [LABEL_COL])

    # Keep original region strings for diagnostics before encoding
    if REGION_COL in df.columns:
        df["_region_str"] = df[REGION_COL].astype(str)
        df[REGION_COL]    = df[REGION_COL].astype("category").cat.codes
        cat_features = [REGION_COL]
        feature_cols = FEATURE_COLS
    else:
        df["_region_str"] = "unknown"
        cat_features = []
        feature_cols = [c for c in FEATURE_COLS if c != REGION_COL]
        warnings.warn("'region' column not found -- excluded from features.")

    return df, feature_cols, cat_features


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def sanity_check_features(df: pd.DataFrame, feature_cols: list) -> None:
    print("\nFeature variation check:")
    for col in feature_cols:
        n_unique = df[col].nunique()
        std      = df[col].std()
        flag     = "  <-- LOW VARIATION" if (n_unique <= 2 or std == 0) else ""
        print(f"  {col}: {n_unique} unique values, std={std:.4f}{flag}")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def group_kfold_by_date(df: pd.DataFrame, n_folds: int = N_FOLDS):
    unique_dates = np.array(df[DATE_COL].unique())
    rng = np.random.RandomState(42)
    rng.shuffle(unique_dates)
    return np.array_split(unique_dates, n_folds)


def run_cv(df, feature_cols, cat_features, params, n_folds=N_FOLDS, label=""):
    folds = group_kfold_by_date(df, n_folds=n_folds)
    aucs, pr_aucs, best_iters = [], [], []

    for fold_idx, val_dates in enumerate(folds):
        train_df = df[~df[DATE_COL].isin(val_dates)]
        val_df   = df[ df[DATE_COL].isin(val_dates)]

        if val_df[LABEL_COL].sum() == 0 or train_df[LABEL_COL].sum() == 0:
            print(f"  Fold {fold_idx}: skipped (no positives in split)")
            continue

        X_tr, y_tr = train_df[feature_cols], train_df[LABEL_COL]
        X_va, y_va = val_df[feature_cols],   val_df[LABEL_COL]

        train_set = lgb.Dataset(X_tr, label=y_tr,
                                categorical_feature=cat_features)
        val_set   = lgb.Dataset(X_va, label=y_va, reference=train_set)

        model = lgb.train(
            params, train_set,
            num_boost_round=2000,
            valid_sets=[val_set], valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        probs    = model.predict(X_va, num_iteration=model.best_iteration)
        auc      = roc_auc_score(y_va, probs)
        pr_auc   = average_precision_score(y_va, probs)
        best_it  = model.best_iteration

        aucs.append(auc)
        pr_aucs.append(pr_auc)
        best_iters.append(best_it)
        tag = f"[{label}] " if label else ""
        print(f"  {tag}Fold {fold_idx} ({len(val_dates)} held-out dates): "
              f"AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  best_iter={best_it}")

    if aucs:
        print(f"  Mean AUC   : {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
        print(f"  Mean PR-AUC: {np.mean(pr_aucs):.4f} (+/- {np.std(pr_aucs):.4f})")
        print(f"  Avg best_iteration: {np.mean(best_iters):.0f}  "
              f"(was ~1 in v1; should now be >> 1)")
    return aucs, pr_aucs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df, feature_cols, cat_features = load_data(CSV_PATH)

    print(f"Loaded {len(df)} rows.")
    print(f"Positive rate: {df[LABEL_COL].mean():.4f}")
    print(f"Unique dates : {df[DATE_COL].nunique()}")
    regions = df["_region_str"].unique()
    print(f"Regions      : {len(regions)} ({', '.join(str(r) for r in regions[:5])}"
          f"{'...' if len(regions) > 5 else ''})")
    sanity_check_features(df, feature_cols)

    n_folds = min(N_FOLDS, df[DATE_COL].nunique())
    if n_folds < 3:
        raise ValueError(
            f"Only {df[DATE_COL].nunique()} unique dates -- not enough for CV. "
            "Fetch more dates before training."
        )

    # ---- Full-dataset CV ----
    print(f"\n=== Full-dataset {n_folds}-fold CV ===")
    aucs_full, _ = run_cv(df, feature_cols, cat_features, LGBM_PARAMS,
                          n_folds=n_folds, label="full")

    # ---- In-season-only diagnostic ----
    print("\n=== In-season-only diagnostic ===")
    months = pd.to_datetime(df[DATE_COL]).dt.month
    def _in_season(region_str, month):
        season = REGION_SEASON_MONTHS.get(region_str)
        return (month in season) if season else None

    df["_in_season"] = [
        _in_season(r, m) for r, m in zip(df["_region_str"], months)
    ]
    in_season_df = df[df["_in_season"] == True]
    print(f"In-season rows: {len(in_season_df)} "
          f"({100 * len(in_season_df) / len(df):.1f}% of total)")
    print(f"In-season positive rate: {in_season_df[LABEL_COL].mean():.4f}")

    n_folds_s = min(N_FOLDS, in_season_df[DATE_COL].nunique())
    if n_folds_s >= 3 and aucs_full:
        aucs_season, _ = run_cv(in_season_df, feature_cols, cat_features,
                                LGBM_PARAMS, n_folds=n_folds_s, label="in-season")
        if aucs_season:
            delta = np.mean(aucs_full) - np.mean(aucs_season)
            if delta > 0.05:
                print(
                    "\n*** AUC drops notably within fire season. The model may be "
                    "relying on seasonal timing. Evaluate primarily on in-season AUC. ***"
                )
            else:
                print(
                    "\nAUC stable within fire season -- genuine spatial signal. "
                    "Ready to scale up to all 10 regions."
                )
    else:
        print("Not enough in-season dates for sub-CV -- skipping.")

    # ---- Refit on full data with chronological holdout ----
    print("\n=== Final model refit (90% train / 10% chronological holdout) ===")
    df_sorted = df.sort_values(DATE_COL).reset_index(drop=True)
    split_idx  = int(0.9 * len(df_sorted))
    X_tr_f = df_sorted[feature_cols].iloc[:split_idx]
    y_tr_f = df_sorted[LABEL_COL].iloc[:split_idx]
    X_va_f = df_sorted[feature_cols].iloc[split_idx:]
    y_va_f = df_sorted[LABEL_COL].iloc[split_idx:]

    train_set_f = lgb.Dataset(X_tr_f, label=y_tr_f,
                              categorical_feature=cat_features)
    val_set_f   = lgb.Dataset(X_va_f, label=y_va_f, reference=train_set_f)

    final_model = lgb.train(
        LGBM_PARAMS, train_set_f,
        num_boost_round=2000,
        valid_sets=[val_set_f], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    final_probs = final_model.predict(X_va_f, num_iteration=final_model.best_iteration)
    final_auc   = roc_auc_score(y_va_f, final_probs)
    final_prauc = average_precision_score(y_va_f, final_probs)
    print(f"Final model -- holdout AUC: {final_auc:.4f}  PR-AUC: {final_prauc:.4f}")
    print(f"Best iteration: {final_model.best_iteration}")

    # ---- Best F1 threshold ----
    precisions, recalls, thresholds = precision_recall_curve(y_va_f, final_probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    if len(thresholds) > 0:
        best_idx    = int(np.argmax(f1_scores[:-1]))
        best_thresh = thresholds[best_idx]
        print(f"Best F1 threshold: {best_thresh:.3f} "
              f"(F1={f1_scores[best_idx]:.4f}, "
              f"Precision={precisions[best_idx]:.4f}, Recall={recalls[best_idx]:.4f})")

    # ---- Feature importance ----
    importance = pd.Series(
        final_model.feature_importance(importance_type="gain"),
        index=feature_cols,
    ).sort_values(ascending=False)
    print("\nFeature importance (gain):")
    print(importance.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Feature importance (v2)")
    plt.tight_layout()
    plt.savefig("./feature_importance_v2.png", dpi=120)
    print("\nSaved feature importance plot to ./feature_importance_v2.png")

    # ---- Save model ----
    os.makedirs("./models", exist_ok=True)
    final_model.save_model(MODEL_OUT_PATH)
    print(f"Saved model to {MODEL_OUT_PATH}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if aucs_full:
        print(f"  v1 mean CV AUC        : ~0.657  (best_iter ~1 -- broken)")
        print(f"  v2 mean CV AUC        : {np.mean(aucs_full):.4f}")
    print(f"  v2 final best_iter    : {final_model.best_iteration}  (should be >> 1)")
    print(f"\nNext: if in-season AUC >= 0.70, fetch all 10 regions' data")
    print(f"and retrain -- the script will scale automatically.")


if __name__ == "__main__":
    main()

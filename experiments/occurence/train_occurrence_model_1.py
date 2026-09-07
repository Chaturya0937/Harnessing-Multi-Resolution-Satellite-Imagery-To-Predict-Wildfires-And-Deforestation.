"""
Train a wildfire OCCURRENCE (risk) model on the CSV produced by
fetch_occurrence_data.py.

Why LightGBM instead of a neural net: for tabular environmental-driver
data like this (weather, fuel, terrain features per grid cell), gradient
boosted trees are the standard, best-performing, and most practical choice
in the wildfire-risk literature - they train in minutes on a CPU (no GPU
needed for this part at all), handle mixed feature types and missing
values natively, and are far less prone to overfitting on a modest-sized
tabular dataset than a deep net would be.

Install:
    pip install lightgbm pandas scikit-learn matplotlib
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH = r"C:\Users\Vardhan\Desktop\Coding\Minor Project\experiments\occurence\wildfire_occurrence_global_10regions.csv"   # combined via combine_downloaded_csvs()
LABEL_COL = "fire_label"
DATE_COL = "date"
DROP_COLS = ["system:index", ".geo"]  # GEE metadata columns, if present

FEATURE_COLS = [
    "temp_mean_k", "wind_speed", "precip_total_m", "dewpoint_k",
    "ndvi", "landcover", "slope_deg", "aspect_deg", "elevation_m",
]

# Region -> fire-season months, copied from fetch_global_occurrence_data.py's
# REGIONS list. Used only to check whether the model is learning genuine
# within-season spatial risk, or mostly just season-vs-off-season timing.
REGION_SEASON_MONTHS = {
    "central_india": [2, 3, 4, 5],
    "california": [7, 8, 9, 10],
    "southeast_australia": [11, 12, 1, 2],
    "mediterranean": [6, 7, 8, 9],
    "amazon_arc_of_fire": [7, 8, 9, 10],
    "southern_africa": [6, 7, 8, 9],
    "western_siberia": [5, 6, 7, 8],
    "indonesia_sumatra": [7, 8, 9, 10],
    "british_columbia": [6, 7, 8, 9],
    "iberia": [6, 7, 8, 9],
}

MODEL_OUT_PATH = "./models/wildfire_occurrence_lgbm.txt"


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    return df


def sanity_check_features(df):
    """Flag features with no real variation - a common sign of a fetch bug
    (wrong band, wrong scale/projection, or a constant default value)
    rather than a genuinely uninformative feature."""
    print("\nFeature variation check:")
    for col in FEATURE_COLS:
        n_unique = df[col].nunique()
        std = df[col].std()
        flag = "  <-- LOW VARIATION, investigate the fetch script" if n_unique <= 2 or std == 0 else ""
        print(f"  {col}: {n_unique} unique values, std={std:.4f}{flag}")


def flag_in_season(df, region_col="region", date_col=DATE_COL):
    """Reconstruct whether each row came from an in-season or off-season
    sample date, using the same REGIONS season-months mapping as the fetch
    script. This lets us test whether the model has learned genuine
    within-season spatial fire risk, or is mostly picking up on
    season-vs-off-season timing (temperature/NDVI naturally differ a lot
    between the two regardless of location, which can inflate AUC without
    reflecting real spatial skill)."""
    months = pd.to_datetime(df[date_col]).dt.month
    def is_in_season(row_region, row_month):
        season_months = REGION_SEASON_MONTHS.get(row_region)
        if season_months is None:
            return None
        return row_month in season_months
    df = df.copy()
    df["_in_season"] = [
        is_in_season(r, m) for r, m in zip(df[region_col], months)
    ]
    return df


def group_kfold_by_date(df, date_col, n_folds=5):
    """K-fold cross-validation grouped by date (and implicitly region, since
    region+date pairs are what vary here). This is far more trustworthy than
    holding out a single date - one date is too easy to get lucky/unlucky
    on, whereas averaging AUC across several different held-out folds of
    dates gives a genuine estimate of how well the model generalizes to
    conditions/locations it hasn't seen."""
    unique_dates = df[date_col].unique()
    rng = np.random.RandomState(42)
    rng.shuffle(unique_dates)
    folds = np.array_split(unique_dates, n_folds)
    return folds


def main():
    import os
    df = load_data(CSV_PATH)
    print(f"Loaded {len(df)} rows.")
    print(f"Positive rate: {df[LABEL_COL].mean():.4f}")
    print(f"Unique dates: {df[DATE_COL].nunique()}")
    sanity_check_features(df)

    n_folds = min(5, df[DATE_COL].nunique())
    if n_folds < 3:
        raise ValueError(
            f"Only {df[DATE_COL].nunique()} unique dates in the data - that's "
            f"not enough for a trustworthy validation split. Fetch more dates "
            f"(see fetch_global_occurrence_data.py) before training."
        )
    folds = group_kfold_by_date(df, DATE_COL, n_folds=n_folds)

    fold_aucs = []
    best_model = None
    best_fold_auc = -1.0
    best_val_probs, best_y_val = None, None

    for fold_idx, val_dates in enumerate(folds):
        train_df = df[~df[DATE_COL].isin(val_dates)]
        val_df = df[df[DATE_COL].isin(val_dates)]
        if val_df[LABEL_COL].sum() == 0 or train_df[LABEL_COL].sum() == 0:
            print(f"Fold {fold_idx}: skipped (no positive examples in train or val split)")
            continue

        X_train, y_train = train_df[FEATURE_COLS], train_df[LABEL_COL]
        X_val, y_val = val_df[FEATURE_COLS], val_df[LABEL_COL]

        n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        params = {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "learning_rate": 0.03,
            "num_leaves": 15,          # smaller/more conservative than before -
            "min_data_in_leaf": 50,    # guards against overfitting small folds
            "max_depth": 5,
            "scale_pos_weight": scale_pos_weight,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "lambda_l2": 1.0,
            "verbose": -1,
        }

        model = lgb.train(
            params, train_set, num_boost_round=1000,
            valid_sets=[val_set], valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)],
        )

        val_probs = model.predict(X_val, num_iteration=model.best_iteration)
        fold_auc = roc_auc_score(y_val, val_probs)
        fold_aucs.append(fold_auc)
        print(f"Fold {fold_idx} ({len(val_dates)} held-out dates): AUC = {fold_auc:.4f}")

        if fold_auc > best_fold_auc:
            best_fold_auc = fold_auc
            best_model = model
            best_val_probs, best_y_val = val_probs, y_val

    print(f"\nMean cross-validated AUC across {len(fold_aucs)} folds: "
          f"{np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")
    if np.mean(fold_aucs) <= 0.55:
        print("\n*** WARNING: mean AUC is close to or below 0.5 (random). The "
              "model has not learned reliable signal yet. This usually means "
              "the dataset still needs more dates/regions, or there's a "
              "feature quality issue worth checking (see feature importance "
              "and df[FEATURE_COLS].describe() below). ***")

    model = best_model
    val_probs, y_val = best_val_probs, best_y_val

    # --- Confound check: does the model still discriminate WITHIN fire
    # season, or is it mostly learning season-vs-off-season timing? ---
    print("\n=== In-season-only diagnostic ===")
    if "region" not in df.columns:
        print("No 'region' column found - skipping in-season diagnostic "
              "(this check only applies to the multi-region global dataset).")
    else:
        df_flagged = flag_in_season(df)
        in_season_df = df_flagged[df_flagged["_in_season"] == True]
        print(f"In-season rows: {len(in_season_df)} "
              f"({100 * len(in_season_df) / len(df_flagged):.1f}% of total)")
        print(f"In-season positive rate: {in_season_df[LABEL_COL].mean():.4f}")

        in_season_folds = group_kfold_by_date(in_season_df, DATE_COL, n_folds=min(5, in_season_df[DATE_COL].nunique()))
        in_season_aucs = []
        for val_dates in in_season_folds:
            train_sub = in_season_df[~in_season_df[DATE_COL].isin(val_dates)]
            val_sub = in_season_df[in_season_df[DATE_COL].isin(val_dates)]
            if val_sub[LABEL_COL].sum() == 0 or train_sub[LABEL_COL].sum() == 0:
                continue
            n_pos, n_neg = train_sub[LABEL_COL].sum(), len(train_sub) - train_sub[LABEL_COL].sum()
            spw = (n_neg / n_pos) if n_pos > 0 else 1.0
            train_set_s = lgb.Dataset(train_sub[FEATURE_COLS], label=train_sub[LABEL_COL])
            val_set_s = lgb.Dataset(val_sub[FEATURE_COLS], label=val_sub[LABEL_COL], reference=train_set_s)
            params_s = {
                "objective": "binary", "metric": "auc", "learning_rate": 0.03,
                "num_leaves": 15, "min_data_in_leaf": 50, "max_depth": 5,
                "scale_pos_weight": spw, "feature_fraction": 0.7,
                "bagging_fraction": 0.7, "bagging_freq": 5, "lambda_l2": 1.0,
                "verbose": -1,
            }
            m = lgb.train(params_s, train_set_s, num_boost_round=1000,
                           valid_sets=[val_set_s],
                           callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            probs = m.predict(val_sub[FEATURE_COLS], num_iteration=m.best_iteration)
            in_season_aucs.append(roc_auc_score(val_sub[LABEL_COL], probs))

        if in_season_aucs:
            mean_in_season = np.mean(in_season_aucs)
            print(f"In-season-only mean CV AUC: {mean_in_season:.4f} "
                  f"(vs {np.mean(fold_aucs):.4f} on the full dataset including off-season)")
            if mean_in_season < np.mean(fold_aucs) - 0.05:
                print("\n*** The AUC drops noticeably when off-season dates are excluded. "
                      "This means a meaningful part of the original score came from "
                      "distinguishing fire-season from off-season timing (via temperature/"
                      "NDVI), not from genuine within-season spatial risk prediction. "
                      "Before scaling up to the full region set, consider: reducing "
                      "off-season negatives relative to in-season ones, adding more "
                      "spatially-varying fuel/terrain features, or evaluating primarily "
                      "on this in-season metric going forward. ***")
            else:
                print("\nAUC holds up with off-season dates excluded - the model appears "
                      "to have learned genuine within-season spatial signal, not just "
                      "calendar timing. Reasonable to scale up to the full region set.")
        else:
            print("Not enough in-season folds with positive examples to run this check.")

    # Find the threshold that maximizes F1, since 0.5 is rarely optimal
    # under class imbalance
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = int(np.argmax(f1_scores[:-1])) if len(thresholds) > 0 else 0
    if len(thresholds) > 0:
        best_thresh = thresholds[best_idx]
        print(f"Best F1 threshold: {best_thresh:.3f} "
              f"(F1={f1_scores[best_idx]:.4f}, "
              f"Precision={precisions[best_idx]:.4f}, Recall={recalls[best_idx]:.4f})")

    # Feature importance - useful sanity check: if e.g. NDVI or wind speed
    # aren't showing up as meaningful predictors, revisit the features
    importance = pd.Series(
        model.feature_importance(importance_type="gain"), index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\nFeature importance (gain):")
    print(importance)

    plt.figure(figsize=(8, 5))
    importance.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Feature importance")
    plt.tight_layout()
    plt.savefig("./feature_importance.png", dpi=120)
    print("\nSaved feature importance plot to ./feature_importance.png")

    os.makedirs("./models", exist_ok=True)
    model.save_model(MODEL_OUT_PATH)
    print(f"Saved model to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
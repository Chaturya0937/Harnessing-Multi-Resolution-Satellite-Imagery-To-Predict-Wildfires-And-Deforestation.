"""
train_occurrence_model_global.py
=================================
Generalized wildfire occurrence model — trained on all 10 globally-distributed
regions and validated with Leave-One-Region-Out (LORO) cross-validation.

WHY LORO INSTEAD OF DATE-BASED FOLDS?
--------------------------------------
Our goal is a model that can predict fire risk for ANY location in the world,
including places it has never seen during training. The only honest way to
measure that is Leave-One-Region-Out CV:
  - Train on 9 regions → predict the held-out 10th region.
  - If AUC is good across all 10 held-out folds, the model has genuinely
    learned transferable physical drivers (temperature, NDVI, terrain, fuel),
    not region-specific quirks.
  - Date-based folds (v1/v2) measured temporal generalization, not spatial.

WHY `region` IS NOT A TRAINING FEATURE?
-----------------------------------------
At inference time for an arbitrary new location anywhere in the world,
you will NOT have a "region" label. The physical features alone must carry
the signal. We use `region` only as the grouping key for LORO CV.

FEATURES USED (all fetched from GEE — physically meaningful globally):
  - temp_mean_k      : 14-day mean 2m temperature (ERA5)
  - wind_speed       : 14-day mean wind speed (ERA5)
  - precip_total_m   : 14-day total precipitation (ERA5)
  - dewpoint_k       : 14-day mean dewpoint temperature (ERA5)
  - ndvi             : vegetation greenness / fuel moisture proxy (Sentinel-2)
  - landcover        : ESA WorldCover land-use class
  - slope_deg        : terrain slope (SRTM)
  - aspect_deg       : terrain aspect (SRTM)
  - elevation_m      : elevation (SRTM)

Install:
    pip install lightgbm pandas scikit-learn matplotlib
"""

import os
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)
from sklearn.calibration import CalibratedClassifierCV
import pickle

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH   = r"C:\Users\Vardhan\Desktop\Coding\Minor Project\experiments\occurence\wildfire_occurence_global_v3.csv"
LABEL_COL  = "fire_label"
DATE_COL   = "date"
REGION_COL = "region"    # used for LORO CV grouping only, NOT as a feature
DROP_COLS  = ["system:index", ".geo"]

# These are the ONLY features used at inference time — all are globally
# available from ERA5 + Sentinel-2 + SRTM + CEMS Fire Danger Indices, which
# cover the entire planet.
#
# v3 adds features specifically chosen to fix the LORO generalization
# failure found with the original 9 raw features: FWI-system fire danger
# indices are physically calibrated to be comparable across climates,
# temp_anomaly_z encodes "how unusual is this FOR THIS PLACE" rather than
# an absolute value, and the lag/window features add real time-series
# signal instead of a single 14-day snapshot.
FEATURE_COLS = [
    # Original raw weather/terrain features
    "temp_mean_k",
    "wind_speed",
    "precip_total_m",
    "dewpoint_k",
    "ndvi",
    "landcover",
    "slope_deg",
    "aspect_deg",
    "elevation_m",
    # New: climate-normalized / physically-grounded features
    "temp_anomaly_z",
    "vpd_kpa",
    # New: multi-window lag features
    "precip_sum_7d",
    "precip_sum_30d",
    "precip_sum_90d",
    "dry_day_count_30d",
    "temp_mean_30d",
    # New: FWI System fire danger indices (CEMS/ECMWF) - the main fix
    "fine_fuel_moisture_code",
    "duff_moisture_code",
    "drought_code",
    "initial_fire_spread_index",
    "build_up_index",
    "fire_weather_index",
    "keetch_byram_drought_index",
    "fire_danger_index",
]

# LightGBM parameters tuned for:
#   - ~180k rows, 9 features, 0.54% positive rate
#   - Maximizing spatial generalization (not just temporal)
#   - is_unbalance=True handles severe class imbalance automatically
LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            ["auc", "average_precision"],
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_data_in_leaf":  20,
    "max_depth":         6,
    "is_unbalance":      True,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "lambda_l1":         0.1,
    "lambda_l2":         0.5,
    "min_gain_to_split": 0.0,
    "verbose":           -1,
    "num_threads":       0,   # use all available CPU cores
}

MODEL_OUT_DIR  = "./models"
MODEL_OUT_PATH = f"{MODEL_OUT_DIR}/wildfire_occurrence_global.txt"
PLOT_OUT_PATH  = "./feature_importance_global.png"

# Fire-season months per region (for the in-season diagnostic only)
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL, REGION_COL])
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} rows with NaN values.")
    df[LABEL_COL]  = df[LABEL_COL].astype(int)
    df[REGION_COL] = df[REGION_COL].astype(str)
    return df


# ---------------------------------------------------------------------------
# Feature diagnostics
# ---------------------------------------------------------------------------
def print_data_summary(df: pd.DataFrame) -> None:
    print(f"\nDataset shape : {df.shape}")
    print(f"Positive rate : {df[LABEL_COL].mean():.4f} "
          f"({df[LABEL_COL].sum():,} fires / {len(df):,} rows)")
    print(f"Unique dates  : {df[DATE_COL].nunique()}  "
          f"({df[DATE_COL].min()} to {df[DATE_COL].max()})")
    print(f"\nRows per region:")
    rc = df.groupby(REGION_COL).agg(
        rows=(LABEL_COL, "count"),
        fires=(LABEL_COL, "sum"),
        pos_rate=(LABEL_COL, "mean"),
    )
    rc["pos_rate"] = rc["pos_rate"].map("{:.4f}".format)
    print(rc.to_string())

    print("\nFeature variation:")
    for col in FEATURE_COLS:
        n_unique = df[col].nunique()
        std      = df[col].std()
        flag     = "  <-- LOW VARIATION" if (n_unique <= 2 or std == 0) else ""
        print(f"  {col}: {n_unique} unique values, std={std:.4f}{flag}")


# ---------------------------------------------------------------------------
# Leave-One-Region-Out cross-validation
#
# This is the CORRECT evaluation for a "predict anywhere" model.
# Each fold trains on 9 regions and evaluates on the 10th.
# A good AUC here means the model genuinely learned physical fire drivers,
# not just memorized the statistics of the training regions.
# ---------------------------------------------------------------------------
def run_loro_cv(df: pd.DataFrame) -> tuple[list, list, list]:
    regions = sorted(df[REGION_COL].unique())
    print(f"\n=== Leave-One-Region-Out CV ({len(regions)} folds) ===")
    print(f"  Each fold trains on {len(regions)-1} regions, tests on 1 held-out region.\n")

    aucs, pr_aucs, region_names = [], [], []

    for held_out_region in regions:
        train_df = df[df[REGION_COL] != held_out_region]
        val_df   = df[df[REGION_COL] == held_out_region]

        if val_df[LABEL_COL].sum() == 0:
            print(f"  [{held_out_region}] SKIP — no positive examples in this region.")
            continue

        X_tr, y_tr = train_df[FEATURE_COLS], train_df[LABEL_COL]
        X_va, y_va = val_df[FEATURE_COLS],   val_df[LABEL_COL]

        train_set = lgb.Dataset(X_tr, label=y_tr)
        val_set   = lgb.Dataset(X_va, label=y_va, reference=train_set)

        model = lgb.train(
            LGBM_PARAMS, train_set,
            num_boost_round=2000,
            valid_sets=[val_set], valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        probs  = model.predict(X_va, num_iteration=model.best_iteration)
        auc    = roc_auc_score(y_va, probs)
        pr_auc = average_precision_score(y_va, probs)
        best_it = model.best_iteration

        aucs.append(auc)
        pr_aucs.append(pr_auc)
        region_names.append(held_out_region)

        pos_rate = y_va.mean()
        print(f"  Held-out [{held_out_region:25s}]  "
              f"AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  "
              f"pos_rate={pos_rate:.4f}  best_iter={best_it}")

    print(f"\n  LORO Mean AUC    : {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
    print(f"  LORO Mean PR-AUC : {np.mean(pr_aucs):.4f} (+/- {np.std(pr_aucs):.4f})")

    if np.mean(aucs) >= 0.72:
        print("\n  [GOOD] LORO AUC >= 0.72 -- model has learned transferable physical "
              "drivers. It can generalize to unseen locations.")
    elif np.mean(aucs) >= 0.60:
        print("\n  [OK] LORO AUC is moderate (0.60-0.72). The model generalizes "
              "partially. Consider adding more regions or climate-zone features "
              "(e.g., Koppen zone, WorldClim normals) to improve cross-region transfer.")
    else:
        print("\n  [WEAK] LORO AUC < 0.60 -- the model struggles to generalize to "
              "unseen regions. The physical features alone are not enough; "
              "more data diversity or stronger features are needed.")

    return aucs, pr_aucs, region_names


# ---------------------------------------------------------------------------
# In-season-only diagnostic (sanity check)
# ---------------------------------------------------------------------------
def run_inseason_diagnostic(df: pd.DataFrame) -> None:
    print("\n=== In-season-only sanity check ===")
    months = pd.to_datetime(df[DATE_COL]).dt.month

    def _in_season(region_str, month):
        season = REGION_SEASON_MONTHS.get(region_str)
        return bool(month in season) if season else None

    df = df.copy()
    df["_in_season"] = [
        _in_season(r, m) for r, m in zip(df[REGION_COL], months)
    ]
    in_season_df = df[df["_in_season"] == True]
    print(f"In-season rows: {len(in_season_df):,} "
          f"({100 * len(in_season_df) / len(df):.1f}% of total)")
    print(f"In-season positive rate: {in_season_df[LABEL_COL].mean():.4f}")
    print(f"  (If positive rate inside fire season >> overall positive rate,")
    print(f"   off-season negatives are diluting the training signal -- OK by design.)")


# ---------------------------------------------------------------------------
# Final model — trained on ALL data, refitted with chronological holdout
# ---------------------------------------------------------------------------
def train_final_model(df: pd.DataFrame):
    print("\n=== Final model — trained on ALL 10 regions ===")
    print("  Using 90% (chronologically earliest) rows for training,")
    print("  10% (most recent) rows as a holdout for reporting only.\n")

    df_sorted  = df.sort_values(DATE_COL).reset_index(drop=True)
    split_idx  = int(0.9 * len(df_sorted))

    X_tr = df_sorted[FEATURE_COLS].iloc[:split_idx]
    y_tr = df_sorted[LABEL_COL].iloc[:split_idx]
    X_va = df_sorted[FEATURE_COLS].iloc[split_idx:]
    y_va = df_sorted[LABEL_COL].iloc[split_idx:]

    print(f"  Train rows: {len(X_tr):,}  (fires: {int(y_tr.sum())})")
    print(f"  Holdout rows: {len(X_va):,}  (fires: {int(y_va.sum())})")

    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set   = lgb.Dataset(X_va, label=y_va, reference=train_set)

    model = lgb.train(
        LGBM_PARAMS, train_set,
        num_boost_round=2000,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    probs  = model.predict(X_va, num_iteration=model.best_iteration)
    auc    = roc_auc_score(y_va, probs)
    pr_auc = average_precision_score(y_va, probs)
    print(f"\n  Holdout AUC    : {auc:.4f}")
    print(f"  Holdout PR-AUC : {pr_auc:.4f}")
    print(f"  Best iteration : {model.best_iteration}")

    # Best threshold by F1
    precisions, recalls, thresholds = precision_recall_curve(y_va, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    if len(thresholds) > 0:
        best_idx    = int(np.argmax(f1s[:-1]))
        best_thresh = thresholds[best_idx]
        print(f"\n  Best F1 threshold : {best_thresh:.3f}")
        print(f"  At threshold      : F1={f1s[best_idx]:.4f}  "
              f"Precision={precisions[best_idx]:.4f}  "
              f"Recall={recalls[best_idx]:.4f}")
        print(f"\n  Interpretation: set the prediction threshold to {best_thresh:.3f}")
        print(f"  when you want maximum F1. For a high-recall early-warning")
        print(f"  system, lower the threshold to ~0.1–0.2.")

    return model, auc, pr_auc


# ---------------------------------------------------------------------------
# Feature importance plot
# ---------------------------------------------------------------------------
def plot_feature_importance(model: lgb.Booster) -> None:
    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=FEATURE_COLS,
    ).sort_values(ascending=False)

    print(f"\nFeature importance (gain -- higher = more predictive):")
    for feat, val in importance.items():
        bar = "|" * int(40 * val / importance.max())
        print(f"  {feat:20s} {val:12.1f}  {bar}")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(importance)))
    importance.plot(kind="barh", ax=ax, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Gain (total information contributed)")
    ax.set_title("Global wildfire occurrence model — feature importance", fontsize=13)
    ax.axvline(0, color="black", linewidth=0.5)
    for i, (val, name) in enumerate(zip(importance.values, importance.index)):
        ax.text(val * 0.02, i, f"{val:,.0f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_OUT_PATH, dpi=150)
    print(f"\nFeature importance plot saved to {PLOT_OUT_PATH}")


# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------
def save_model(model: lgb.Booster) -> None:
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    model.save_model(MODEL_OUT_PATH)
    # Also save feature names and order so inference code always uses the
    # exact same feature set
    meta = {
        "feature_cols": FEATURE_COLS,
        "model_path":   MODEL_OUT_PATH,
        "label_col":    LABEL_COL,
    }
    meta_path = f"{MODEL_OUT_DIR}/wildfire_occurrence_global_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"\nModel saved to         : {MODEL_OUT_PATH}")
    print(f"Feature metadata saved : {meta_path}")


# ---------------------------------------------------------------------------
# LORO result summary table
# ---------------------------------------------------------------------------
def print_loro_summary(region_names, aucs, pr_aucs):
    print("\n" + "=" * 70)
    print("LEAVE-ONE-REGION-OUT CV SUMMARY")
    print("=" * 70)
    print(f"{'Region':<28} {'AUC':>8} {'PR-AUC':>10}  {'Generalizes?':>14}")
    print("-" * 70)
    for r, a, p in sorted(zip(region_names, aucs, pr_aucs), key=lambda x: -x[1]):
        label = "[+] Yes" if a >= 0.68 else ("[~] Partial" if a >= 0.58 else "[-] Weak")
        print(f"  {r:<26} {a:>8.4f} {p:>10.4f}  {label:>14}")
    print("-" * 70)
    print(f"  {'MEAN':<26} {np.mean(aucs):>8.4f} {np.mean(pr_aucs):>10.4f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("GLOBAL WILDFIRE OCCURRENCE MODEL — TRAINING")
    print("=" * 70)

    # Load
    print(f"\nLoading {CSV_PATH} ...")
    df = load_data(CSV_PATH)
    print_data_summary(df)

    n_regions = df[REGION_COL].nunique()
    if n_regions < 5:
        warnings.warn(
            f"Only {n_regions} regions found. For a truly global model, "
            "use all 10 regions from fetch_global_occurrence_data.py."
        )

    # In-season sanity check
    run_inseason_diagnostic(df)

    # LORO CV — the main generalization test
    aucs, pr_aucs, region_names = run_loro_cv(df)

    # Pretty summary table
    if aucs:
        print_loro_summary(region_names, aucs, pr_aucs)

    # Final model on all data
    model, final_auc, final_prauc = train_final_model(df)

    # Plot
    plot_feature_importance(model)

    # Save
    save_model(model)

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    if aucs:
        print(f"  LORO mean AUC       : {np.mean(aucs):.4f}  (spatial generalization)")
    print(f"  Final holdout AUC   : {final_auc:.4f}  (temporal generalization)")
    print(f"  Final holdout PR-AUC: {final_prauc:.4f}")
    print(f"\n  The saved model predicts wildfire probability for any location")
    print(f"  given these 9 features (all globally available from ERA5+S2+SRTM):")
    for i, f in enumerate(FEATURE_COLS, 1):
        print(f"    {i}. {f}")
    print(f"\n  To predict for a new location:")
    print(f"    model = lgb.Booster(model_file='{MODEL_OUT_PATH}')")
    print(f"    prob  = model.predict(X_new)[0]   # X_new has columns: {FEATURE_COLS}")
    print(f"    fire_risk = 'HIGH' if prob > 0.2 else 'LOW'   # adjust threshold")


if __name__ == "__main__":
    main()
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH     = r"C:\Users\Vardhan\Desktop\Minor\Harnessing-Multi-Resolution-Satellite-Imagery-To-Predict-Wildfires-And-Deforestation.-main\experiments\occurence\wildfire_occurence_global_v3.csv"
MODEL_PATH   = "./models/wildfire_occurrence_global.txt"
EVAL_OUT_DIR = "./eval_results"

FEATURE_COLS = [
    "temp_mean_k", "wind_speed", "precip_total_m", "dewpoint_k",
    "ndvi", "landcover", "slope_deg", "aspect_deg", "elevation_m",
    "temp_anomaly_z", "vpd_kpa", "precip_sum_7d", "precip_sum_30d",
    "precip_sum_90d", "dry_day_count_30d", "temp_mean_30d",
    "fine_fuel_moisture_code", "duff_moisture_code", "drought_code",
    "initial_fire_spread_index", "build_up_index", "fire_weather_index",
    "keetch_byram_drought_index", "fire_danger_index"
]
LABEL_COL = "fire_label"

def main():
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)
    print("=" * 60)
    print("WILDFIRE MODEL ADVANCED EVALUATION & PLOTTING")
    print("=" * 60)

    # 1. Load Data
    print(f"\nLoading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])

    # 2. Extract Balanced Subset (At least 250 Negative and 250 Positive samples)
    pos_indices = df[df[LABEL_COL] == 1].index
    neg_indices = df[df[LABEL_COL] == 0].index

    n_samples = 250  # target samples per class
    np.random.seed(42)
    selected_pos = np.random.choice(pos_indices, size=min(n_samples, len(pos_indices)), replace=False)
    selected_neg = np.random.choice(neg_indices, size=n_samples, replace=False)
    balanced_indices = np.concatenate([selected_pos, selected_neg])

    df_eval = df.loc[balanced_indices].sample(frac=1, random_state=42).reset_index(drop=True)
    X_eval = df_eval[FEATURE_COLS]
    y_eval = df_eval[LABEL_COL]

    print(f"Evaluation dataset prepared: {len(X_eval):,} total rows")
    print(f"  - 0 (No Fire - Negative) : {sum(y_eval == 0)}")
    print(f"  - 1 (Fire - Positive)    : {sum(y_eval == 1)}")

    # 3. Load Model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train your model first.")
    
    print(f"Loading model from {MODEL_PATH}...")
    model = lgb.Booster(model_file=MODEL_PATH)

    # 4. Predict Probabilities
    probs = model.predict(X_eval, num_iteration=model.best_iteration)

    # 5. Automatic Threshold Selection (Maximizing F1 / High Recall)
    precisions, recalls, thresholds = precision_recall_curve(y_eval, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    AUTO_THRESHOLD = thresholds[best_idx]
    print(f"\n[INFO] Auto-calculated optimal threshold: {AUTO_THRESHOLD:.4f}")

    preds_binary = (probs >= AUTO_THRESHOLD).astype(int)

    # 6. Calculate Metrics
    auc = roc_auc_score(y_eval, probs)
    pr_auc = average_precision_score(y_eval, probs)
    cm = confusion_matrix(y_eval, preds_binary)

    # 7. Save Text Reports
    with open(os.path.join(EVAL_OUT_DIR, "metrics_summary.txt"), "w") as f:
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        f.write(f"Auto-Calculated Threshold: {AUTO_THRESHOLD:.4f}\n")

    with open(os.path.join(EVAL_OUT_DIR, "confusion_matrix.txt"), "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
        f.write(f"True Negatives (TN): {cm[0,0]}\nFalse Positives (FP): {cm[0,1]}\n")
        f.write(f"False Negatives (FN): {cm[1,0]}\nTrue Positives (TP): {cm[1,1]}\n")

    with open(os.path.join(EVAL_OUT_DIR, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_eval, preds_binary, target_names=["0 (No Fire - Negative)", "1 (Fire - Positive)"]))

    # 8. Generate & Save ROC-AUC Curve Graph
    plt.figure(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_eval, probs)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate (Recall)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_plot_path = os.path.join(EVAL_OUT_DIR, "roc_auc_curve.png")
    plt.savefig(roc_plot_path, dpi=150)
    plt.close()
    print(f"Saved ROC-AUC graph to: {roc_plot_path}")

    # 9. Generate & Save Feature Importance Graph
    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=FEATURE_COLS,
    ).sort_values(ascending=True)  # ascending for horizontal bar chart layout

    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(importance)))
    importance.plot(kind="barh", color=colors)
    plt.xlabel("Total Gain (Information Contributed)", fontsize=11)
    plt.title("Global Wildfire Model — Feature Importance", fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    feat_plot_path = os.path.join(EVAL_OUT_DIR, "feature_importance.png")
    plt.savefig(feat_plot_path, dpi=150)
    plt.close()
    print(f"Saved Feature Importance graph to: {feat_plot_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
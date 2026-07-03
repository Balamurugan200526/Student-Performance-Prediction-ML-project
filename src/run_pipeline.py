"""
run_pipeline.py
----------------
One-command orchestrator that runs the ENTIRE project pipeline:
    1. Load & clean raw data
    2. Generate EDA graphs -> outputs/graphs/
    3. Feature engineering
    4. Train & compare 5 models
    5. Evaluate the best model -> outputs/graphs/ + outputs/reports/
    6. Save best model + preprocessing artifacts -> models/
    7. Generate sample predictions -> outputs/predictions.csv

Run with:
    python src/run_pipeline.py
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import preprocessing as pp
from feature_engineering import add_derived_features, drop_unnecessary_columns
from visualization import run_full_eda
from train_model import build_dataset, train_and_compare, save_artifacts
from evaluation import full_evaluation

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE, "dataset", "student_performance.csv")
GRAPHS_DIR = os.path.join(BASE, "outputs", "graphs")
REPORTS_DIR = os.path.join(BASE, "outputs", "reports")
MODEL_DIR = os.path.join(BASE, "models")


def main():
    print("=" * 70)
    print("STEP 1: Load & clean data for EDA")
    print("=" * 70)
    df_raw = pp.load_data(RAW_PATH)
    df_clean = pp.handle_missing_values(df_raw)
    df_clean = pp.remove_duplicates(df_clean)
    df_clean = pp.handle_outliers(df_clean)

    print("\n" + "=" * 70)
    print("STEP 2: Exploratory Data Analysis (EDA)")
    print("=" * 70)
    run_full_eda(df_clean, out_dir=GRAPHS_DIR)

    print("\n" + "=" * 70)
    print("STEP 3: Feature Engineering + Model Training")
    print("=" * 70)
    X_train, X_test, y_train, y_test, encoders, scaler, feature_cols = build_dataset(RAW_PATH)
    results_df, best_name, best_model, all_models = train_and_compare(X_train, X_test, y_train, y_test)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)

    print("\n" + "=" * 70)
    print(f"STEP 4: Detailed Evaluation of Best Model ({best_name})")
    print("=" * 70)
    class_names = list(encoders["Final_Result"].classes_)
    metrics = full_evaluation(best_model, X_test, y_test, class_names, GRAPHS_DIR)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.insert(0, "Model", best_name)
    metrics_df.to_csv(os.path.join(REPORTS_DIR, "best_model_metrics.csv"), index=False)

    print("\n" + "=" * 70)
    print("STEP 5: Saving Model Artifacts")
    print("=" * 70)
    save_artifacts(best_model, encoders, scaler, feature_cols, MODEL_DIR)

    print("\n" + "=" * 70)
    print("STEP 6: Generating Sample Predictions File")
    print("=" * 70)
    sample_preds = X_test.copy()
    sample_preds["Actual_Result"] = [class_names[i] for i in y_test]
    sample_preds["Predicted_Result"] = [class_names[i] for i in best_model.predict(X_test)]
    preds_path = os.path.join(BASE, "outputs", "predictions.csv")
    sample_preds.to_csv(preds_path, index=False)
    print(f"[run_pipeline] Saved predictions -> {preds_path}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Best Model: {best_name}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

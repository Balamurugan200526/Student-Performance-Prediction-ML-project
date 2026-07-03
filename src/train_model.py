"""
train_model.py
---------------
Trains and compares five classification algorithms:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. Support Vector Machine (SVM)
    5. K-Nearest Neighbors (KNN)

The best model (by test-set accuracy) is saved to models/ using joblib,
along with the fitted LabelEncoders and StandardScaler so predict.py can
reproduce the exact same preprocessing at inference time.
"""

import os
import sys
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.dirname(__file__))
from preprocessing import full_preprocessing_pipeline, NUMERIC_COLS, TARGET_COL
from feature_engineering import add_derived_features, drop_unnecessary_columns


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
}


def build_dataset(raw_path: str):
    """
    Load + clean the raw CSV, add engineered features, encode & scale,
    then split into train/test sets. Returns everything needed for both
    training and later inference (encoders, scaler, feature column order).
    """
    import preprocessing as pp

    df = pp.load_data(raw_path)
    df = pp.handle_missing_values(df)
    df = pp.remove_duplicates(df)
    df = pp.handle_outliers(df)
    df = add_derived_features(df)
    df = drop_unnecessary_columns(df)

    df, encoders = pp.encode_categorical(df, fit=True)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X_train, X_test, y_train, y_test = pp.split_data(df, target_col=TARGET_COL)

    scale_cols = [c for c in feature_cols if c in df.select_dtypes(include="number").columns
                  and c not in ["Gender", "Participation", "Family_Support"]]

    X_train, scaler = pp.scale_features(X_train, scale_cols, fit=True)
    X_test, _ = pp.scale_features(X_test, scale_cols, scaler=scaler, fit=False)

    return X_train, X_test, y_train, y_test, encoders, scaler, feature_cols


def train_and_compare(X_train, X_test, y_train, y_test):
    """Train every candidate model, evaluate on the test set, and report results."""
    results = []
    fitted_models = {}

    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        results.append({"Model": name, "Accuracy": round(acc, 4), "F1_Score": round(f1, 4)})
        fitted_models[name] = model
        print(f"[train_model] {name}: Accuracy={acc:.4f}, F1={f1:.4f}")

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    print(f"\n[train_model] Best model: {best_model_name} (Accuracy={results_df.iloc[0]['Accuracy']})")
    return results_df, best_model_name, best_model, fitted_models


def save_artifacts(model, encoders, scaler, feature_cols, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "student_performance_model.pkl"))
    joblib.dump(encoders, os.path.join(model_dir, "label_encoders.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(model_dir, "feature_columns.pkl"))
    print(f"[train_model] Saved model + encoders + scaler + feature list -> {model_dir}")


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(BASE, "dataset", "student_performance.csv")
    model_dir = os.path.join(BASE, "models")
    reports_dir = os.path.join(BASE, "outputs", "reports")

    X_train, X_test, y_train, y_test, encoders, scaler, feature_cols = build_dataset(raw_path)
    results_df, best_name, best_model, all_models = train_and_compare(X_train, X_test, y_train, y_test)

    os.makedirs(reports_dir, exist_ok=True)
    results_df.to_csv(os.path.join(reports_dir, "model_comparison.csv"), index=False)

    save_artifacts(best_model, encoders, scaler, feature_cols, model_dir)

    print("\n[train_model] Training pipeline complete.")
    print(results_df)

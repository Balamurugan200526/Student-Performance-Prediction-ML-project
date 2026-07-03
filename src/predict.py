"""
predict.py
----------
Loads the saved model + preprocessing artifacts and predicts the
Final_Result (Pass / Average / Fail) for a NEW student record.

Can be used in two ways:
  1. As a library:  from predict import predict_student
  2. As a CLI:       python predict.py   (interactive prompts)

Both app.py (Streamlit) and this CLI reuse `predict_student()` so the
prediction logic lives in exactly one place.
"""

import os
import joblib
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")

NUMERIC_COLS = [
    "Age", "Attendance", "Study_Hours_Per_Day", "Previous_Marks",
    "Assignment_Score", "Internal_Marks", "Sleep_Hours", "Internet_Usage",
]
CATEGORICAL_COLS = ["Gender", "Participation", "Family_Support"]


def load_artifacts(model_dir: str = MODEL_DIR):
    """Load the trained model, encoders, scaler, and expected feature order."""
    model = joblib.load(os.path.join(model_dir, "student_performance_model.pkl"))
    encoders = joblib.load(os.path.join(model_dir, "label_encoders.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_columns.pkl"))
    return model, encoders, scaler, feature_cols


def _engineer_single_record(record: dict) -> dict:
    """Apply the SAME derived-feature formulas used during training to one record."""
    record = dict(record)  # copy
    record["Overall_Academic_Score"] = (
        0.4 * record["Previous_Marks"]
        + 0.3 * record["Assignment_Score"]
        + 0.3 * (record["Internal_Marks"] / 40 * 100)
    )
    study_hours = record["Study_Hours_Per_Day"] if record["Study_Hours_Per_Day"] != 0 else 0.1
    record["Study_Efficiency"] = record["Previous_Marks"] / study_hours
    record["Lifestyle_Balance"] = record["Sleep_Hours"] - (record["Internet_Usage"] * 0.5)
    return record


def predict_student(record: dict, model_dir: str = MODEL_DIR):
    """
    record: dict with keys matching the raw dataset columns, e.g.
        {
            "Gender": "Male", "Age": 18, "Attendance": 82.5,
            "Study_Hours_Per_Day": 3.5, "Previous_Marks": 70,
            "Assignment_Score": 75, "Internal_Marks": 30,
            "Sleep_Hours": 7, "Internet_Usage": 2.5,
            "Participation": "High", "Family_Support": "High"
        }

    Returns: (predicted_label: str, probability_dict: dict)
    """
    model, encoders, scaler, feature_cols = load_artifacts(model_dir)

    record = _engineer_single_record(record)
    df = pd.DataFrame([record])

    # Encode categoricals using the SAME fitted encoders from training
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        df[col] = le.transform(df[col].astype(str))

    # Scale numeric + engineered columns using the SAME fitted scaler
    scale_cols = [c for c in feature_cols if c in df.columns and c not in CATEGORICAL_COLS]
    df[scale_cols] = scaler.transform(df[scale_cols])

    # Ensure column order matches training exactly
    df = df[feature_cols]

    pred_encoded = model.predict(df)[0]
    pred_label = encoders["Final_Result"].inverse_transform([pred_encoded])[0]

    probability_dict = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        classes = encoders["Final_Result"].inverse_transform(range(len(proba)))
        probability_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

    return pred_label, probability_dict


def _interactive_cli():
    """Simple command-line interface for manual testing."""
    print("=== Student Performance Prediction ===")
    try:
        record = {
            "Gender": input("Gender (Male/Female): ").strip() or "Male",
            "Age": int(input("Age (16-19): ") or 18),
            "Attendance": float(input("Attendance % (0-100): ") or 75),
            "Study_Hours_Per_Day": float(input("Study Hours Per Day: ") or 3),
            "Previous_Marks": float(input("Previous Marks (0-100): ") or 60),
            "Assignment_Score": float(input("Assignment Score (0-100): ") or 65),
            "Internal_Marks": float(input("Internal Marks (0-40): ") or 25),
            "Sleep_Hours": float(input("Sleep Hours: ") or 6.5),
            "Internet_Usage": float(input("Internet Usage hrs/day: ") or 3),
            "Participation": input("Participation (Low/Medium/High): ").strip() or "Medium",
            "Family_Support": input("Family Support (Low/Medium/High): ").strip() or "Medium",
        }
    except Exception as e:
        print(f"Invalid input: {e}")
        return

    label, proba = predict_student(record)
    print(f"\nPredicted Final Result: {label}")
    if proba:
        print("Class probabilities:")
        for cls, p in proba.items():
            print(f"  {cls}: {p * 100:.2f}%")


if __name__ == "__main__":
    _interactive_cli()

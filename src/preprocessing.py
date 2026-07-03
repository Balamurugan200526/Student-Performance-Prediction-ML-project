"""
preprocessing.py
-----------------
Handles all data-cleaning steps required before modelling:
    1. Loading the raw CSV
    2. Handling missing values
    3. Removing duplicate rows
    4. Outlier detection & capping
    5. Encoding categorical variables
    6. Feature scaling
    7. Train-test split

Each function is intentionally small and single-purpose (modular design)
so it can be unit-tested and reused independently (e.g., inside
predict.py, where a single new record needs to go through the same
encoding/scaling pipeline used during training).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

CATEGORICAL_COLS = ["Gender", "Participation", "Family_Support"]
NUMERIC_COLS = [
    "Age", "Attendance", "Study_Hours_Per_Day", "Previous_Marks",
    "Assignment_Score", "Internal_Marks", "Sleep_Hours", "Internet_Usage",
]
TARGET_COL = "Final_Result"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    try:
        df = pd.read_csv(path)
        print(f"[preprocessing] Loaded {df.shape[0]} rows, {df.shape[1]} columns from {path}")
        return df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset not found at {path}") from e


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing numeric values with the column median (robust to outliers)
    and missing categorical values with the column mode.
    """
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    print("[preprocessing] Missing values handled (numeric -> median, categorical -> mode)")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, keeping the first occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "Student_ID"])
    after = len(df)
    print(f"[preprocessing] Removed {before - after} duplicate rows")
    return df.reset_index(drop=True)


def handle_outliers(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """
    Cap outliers using the IQR (Interquartile Range) method instead of
    deleting them, so we don't lose valuable student records. Any value
    beyond 1.5*IQR from Q1/Q3 is clipped to the boundary.
    """
    df = df.copy()
    cols = cols or NUMERIC_COLS
    for col in cols:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        if n_outliers:
            print(f"[preprocessing] Capped {n_outliers} outliers in '{col}'")
    return df


def encode_categorical(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Label-encode categorical columns (Gender, Participation, Family_Support)
    and the target column (Final_Result). Returns the encoded dataframe and
    the dictionary of fitted LabelEncoders so the SAME encoding can be
    reused at prediction time (critical - otherwise categories could be
    mapped to different numbers between training and inference).
    """
    df = df.copy()
    encoders = encoders or {}

    cols_to_encode = CATEGORICAL_COLS + ([TARGET_COL] if TARGET_COL in df.columns else [])

    for col in cols_to_encode:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    print(f"[preprocessing] Encoded categorical columns: {cols_to_encode}")
    return df, encoders


def scale_features(df: pd.DataFrame, feature_cols, scaler: StandardScaler = None, fit: bool = True):
    """
    Standardize numeric features to zero mean / unit variance.
    StandardScaler is used (rather than MinMax) because algorithms like
    Logistic Regression, SVM and KNN are distance/gradient based and
    perform better with standardized inputs.
    """
    df = df.copy()
    if fit:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    print(f"[preprocessing] Scaled {len(feature_cols)} numeric features")
    return df, scaler


def split_data(df: pd.DataFrame, target_col: str = TARGET_COL, test_size: float = 0.2, random_state: int = 42):
    """Stratified train-test split to preserve class proportions."""
    feature_cols = [c for c in df.columns if c not in [target_col, "Student_ID"]]
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[preprocessing] Train/Test split -> train: {X_train.shape[0]} rows, test: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


def full_preprocessing_pipeline(path: str):
    """
    Convenience function that runs the entire preprocessing pipeline
    end-to-end and returns train/test splits plus the fitted encoders
    and scaler (needed later for the prediction module).
    """
    df = load_data(path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df, encoders = encode_categorical(df, fit=True)

    feature_cols = [c for c in df.columns if c not in [TARGET_COL, "Student_ID"]]
    X_train, X_test, y_train, y_test = split_data(df)

    X_train, scaler = scale_features(X_train, NUMERIC_COLS, fit=True)
    X_test, _ = scale_features(X_test, NUMERIC_COLS, scaler=scaler, fit=False)

    return X_train, X_test, y_train, y_test, encoders, scaler, feature_cols


if __name__ == "__main__":
    result = full_preprocessing_pipeline("../dataset/student_performance.csv")
    print("[preprocessing] Pipeline complete.")

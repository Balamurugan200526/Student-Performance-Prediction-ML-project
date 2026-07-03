"""
feature_engineering.py
-----------------------
Adds derived features, performs correlation-based feature selection,
and reports feature importance once a tree-based model is available.

Design rationale:
- Study_Efficiency = Study_Hours_Per_Day combined with Previous_Marks
  captures how effectively a student converts study time into results,
  which is more informative than either raw column alone.
- Overall_Academic_Score blends previous marks, assignment score and
  internal marks into one composite indicator of academic consistency.
- Student_ID is dropped because it's a unique identifier with zero
  predictive value and can leak nothing but noise into the model.
"""

import pandas as pd
import numpy as np


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new engineered features from existing columns."""
    df = df.copy()

    # Composite academic score: weighted average of the three marks columns
    df["Overall_Academic_Score"] = (
        0.4 * df["Previous_Marks"]
        + 0.3 * df["Assignment_Score"]
        + 0.3 * (df["Internal_Marks"] / 40 * 100)  # normalize internal marks to /100
    )

    # Study efficiency: marks gained per hour of daily study (avoids div-by-zero)
    df["Study_Efficiency"] = df["Previous_Marks"] / df["Study_Hours_Per_Day"].replace(0, 0.1)

    # Lifestyle balance score: rewards adequate sleep, penalizes excess internet use
    df["Lifestyle_Balance"] = df["Sleep_Hours"] - (df["Internet_Usage"] * 0.5)

    print("[feature_engineering] Added: Overall_Academic_Score, Study_Efficiency, Lifestyle_Balance")
    return df


def drop_unnecessary_columns(df: pd.DataFrame, cols_to_drop=None) -> pd.DataFrame:
    """Remove columns that add no predictive value (e.g., unique IDs)."""
    cols_to_drop = cols_to_drop or ["Student_ID"]
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    print(f"[feature_engineering] Dropped columns: {existing}")
    return df


def correlation_analysis(df: pd.DataFrame, target_col: str = "Final_Result", top_n: int = 8):
    """
    Compute correlation of numeric features with the (encoded) target
    and return the top-N most correlated features. Useful both for
    reporting in EDA and for a lightweight feature-selection step.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        raise ValueError(f"'{target_col}' must be numeric-encoded before correlation analysis")

    corr_with_target = numeric_df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
    top_features = corr_with_target.head(top_n)
    print(f"[feature_engineering] Top {top_n} features correlated with target:\n{top_features}")
    return top_features


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from a fitted tree-based model
    (Random Forest / Decision Tree). Returns a sorted pandas Series.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model does not expose feature_importances_ (use a tree-based model)")

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    print(f"[feature_engineering] Feature importances:\n{importances}")
    return importances

"""
visualization.py
-----------------
Generates all Exploratory Data Analysis (EDA) graphs used in the project
report and notebook, and saves them into outputs/graphs/.

Every plotting function is standalone so it can be called individually
from the Jupyter notebook for interactive exploration.
"""

import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so this also works from scripts/servers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")

NUMERIC_COLS = [
    "Age", "Attendance", "Study_Hours_Per_Day", "Previous_Marks",
    "Assignment_Score", "Internal_Marks", "Sleep_Hours", "Internet_Usage",
]


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[visualization] Saved: {path}")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str):
    """
    Correlation heatmap: shows how strongly every pair of numeric
    features (and the encoded target) move together. Darker/brighter
    cells near +1 or -1 indicate strong relationships; values near 0
    indicate the features are largely independent.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    numeric_df = df.select_dtypes(include="number")
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Features")
    _save(fig, out_dir, "correlation_heatmap.png")


def plot_histograms(df: pd.DataFrame, out_dir: str):
    """
    Histograms: show the distribution/spread of each numeric feature,
    helping identify skewness and the general shape of the data.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=axes[i], color="steelblue")
            axes[i].set_title(f"Distribution of {col}")
    for j in range(len(NUMERIC_COLS), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Feature Distributions (Histograms)", fontsize=14)
    _save(fig, out_dir, "histograms.png")


def plot_boxplots(df: pd.DataFrame, out_dir: str):
    """
    Box plots: reveal the median, quartile spread, and outliers of each
    numeric feature -- essential for spotting data-entry errors and
    understanding variability across students.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        if col in df.columns:
            sns.boxplot(y=df[col], ax=axes[i], color="lightcoral")
            axes[i].set_title(f"Boxplot of {col}")
    for j in range(len(NUMERIC_COLS), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Feature Distributions (Box Plots) - Outlier Detection", fontsize=14)
    _save(fig, out_dir, "boxplots.png")


def plot_countplots(df: pd.DataFrame, out_dir: str):
    """
    Count plots: show class frequency for categorical columns (Gender,
    Participation, Family_Support) and the target (Final_Result),
    helping check for class imbalance.
    """
    cat_cols = ["Gender", "Participation", "Family_Support", "Final_Result"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(5 * len(cat_cols), 5))
    if len(cat_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cat_cols):
        sns.countplot(x=df[col], ax=ax, palette="viridis")
        ax.set_title(f"Count of {col}")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Categorical Feature Counts", fontsize=14)
    _save(fig, out_dir, "countplots.png")


def plot_pairplot(df: pd.DataFrame, out_dir: str):
    """
    Pair plot: visualizes pairwise relationships between a subset of key
    numeric features, colored by Final_Result, to spot separability
    between Pass/Average/Fail groups.
    """
    subset_cols = ["Attendance", "Study_Hours_Per_Day", "Previous_Marks", "Assignment_Score", "Final_Result"]
    subset_cols = [c for c in subset_cols if c in df.columns]
    pair_fig = sns.pairplot(df[subset_cols], hue="Final_Result", palette="Set2", diag_kind="kde")
    pair_fig.fig.suptitle("Pairwise Feature Relationships by Final Result", y=1.02)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pairplot.png")
    pair_fig.savefig(path, dpi=120)
    plt.close("all")
    print(f"[visualization] Saved: {path}")


def plot_result_vs_feature(df: pd.DataFrame, out_dir: str):
    """
    Box plots of key numeric features grouped by Final_Result -- makes
    it visually clear which factors most separate Pass/Average/Fail
    students (e.g., attendance and study hours typically differ a lot).
    """
    key_features = ["Attendance", "Study_Hours_Per_Day", "Previous_Marks", "Assignment_Score"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, col in zip(axes, key_features):
        sns.boxplot(x="Final_Result", y=col, data=df, ax=ax, palette="pastel")
        ax.set_title(f"{col} by Final Result")
    fig.suptitle("Key Features vs Final Result", fontsize=14)
    _save(fig, out_dir, "result_vs_features.png")


def run_full_eda(df: pd.DataFrame, out_dir: str = "../outputs/graphs"):
    """Run every EDA plotting function in sequence."""
    plot_correlation_heatmap(df, out_dir)
    plot_histograms(df, out_dir)
    plot_boxplots(df, out_dir)
    plot_countplots(df, out_dir)
    plot_result_vs_feature(df, out_dir)
    plot_pairplot(df, out_dir)
    print("[visualization] Full EDA complete.")

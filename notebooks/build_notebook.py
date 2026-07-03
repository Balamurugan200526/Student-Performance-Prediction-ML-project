"""
build_notebook.py
------------------
Programmatically builds Student_Performance_Prediction.ipynb using
nbformat, so the notebook content stays in sync with the src/ modules.
Run once to (re)generate the .ipynb file.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------
md("""# Student Performance Prediction Using Machine Learning
### AI & ML Internship Project — End-to-End Notebook

This notebook walks through the complete machine learning pipeline:
data loading, cleaning, exploratory data analysis (EDA), feature
engineering, model training & comparison, evaluation, and prediction
on new student records.

**Author:** Balamurugan S — B.Tech Information Technology
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib
""")

# ---------------------------------------------------------------
md("## 1. Imports & Setup")
code("""import sys, os
sys.path.append(os.path.join(os.getcwd(), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import preprocessing as pp
from feature_engineering import add_derived_features, drop_unnecessary_columns, correlation_analysis, get_feature_importance
from visualization import (
    plot_correlation_heatmap, plot_histograms, plot_boxplots,
    plot_countplots, plot_pairplot, plot_result_vs_feature
)
from train_model import build_dataset, train_and_compare, save_artifacts
from evaluation import compute_metrics, plot_confusion_matrix, print_classification_report, plot_roc_curves

sns.set_theme(style='whitegrid')
%matplotlib inline
""")

# ---------------------------------------------------------------
md("""## 2. Load the Dataset

The dataset (`student_performance.csv`) contains 600+ synthetic but
realistically-distributed student records with academic and lifestyle
features, and a target column `Final_Result` (Pass / Average / Fail).""")
code("""DATA_PATH = '../dataset/student_performance.csv'
df = pp.load_data(DATA_PATH)
df.head()
""")

code("""df.info()
""")

code("""df.describe(include='all')
""")

# ---------------------------------------------------------------
md("""## 3. Data Preprocessing

Steps performed:
1. Handle missing values (median for numeric, mode for categorical)
2. Remove duplicate rows
3. Detect & cap outliers using the IQR method
4. Encode categorical variables
5. Scale numeric features
6. Train-test split (stratified)""")

code("""print('Missing values before cleaning:')
print(df.isna().sum())
""")

code("""df_clean = pp.handle_missing_values(df)
df_clean = pp.remove_duplicates(df_clean)
df_clean = pp.handle_outliers(df_clean)
print('Missing values after cleaning:', df_clean.isna().sum().sum())
print('Shape after cleaning:', df_clean.shape)
""")

# ---------------------------------------------------------------
md("""## 4. Exploratory Data Analysis (EDA)

Below we visualize the cleaned dataset to understand feature
distributions, relationships, and how they connect to the target
variable `Final_Result`.""")

md("### 4.1 Correlation Heatmap\nShows how strongly numeric features move together.")
code("""plot_correlation_heatmap(df_clean, out_dir='../outputs/graphs')
""")

md("### 4.2 Histograms\nDistribution/spread of every numeric feature.")
code("""plot_histograms(df_clean, out_dir='../outputs/graphs')
""")

md("### 4.3 Box Plots\nMedian, quartiles, and outliers per feature.")
code("""plot_boxplots(df_clean, out_dir='../outputs/graphs')
""")

md("### 4.4 Count Plots\nClass frequency for categorical columns and the target.")
code("""plot_countplots(df_clean, out_dir='../outputs/graphs')
""")

md("### 4.5 Key Features vs Final Result\nHow Attendance, Study Hours, Previous Marks, and Assignment Score differ across Pass/Average/Fail.")
code("""plot_result_vs_feature(df_clean, out_dir='../outputs/graphs')
""")

md("### 4.6 Pair Plot\nPairwise relationships between key numeric features, colored by result.")
code("""plot_pairplot(df_clean, out_dir='../outputs/graphs')
""")

# ---------------------------------------------------------------
md("""## 5. Feature Engineering

We create three new engineered features:
- **Overall_Academic_Score** — weighted blend of Previous Marks, Assignment Score, Internal Marks
- **Study_Efficiency** — marks gained per hour of daily study
- **Lifestyle_Balance** — sleep hours adjusted for excess internet usage

We also drop `Student_ID` (a unique identifier with no predictive value).""")

code("""df_fe = add_derived_features(df_clean)
df_fe = drop_unnecessary_columns(df_fe)
df_fe.head()
""")

md("### 5.1 Correlation with Target\nTo understand which features matter most, we encode categoricals and check correlation with `Final_Result`.")
code("""df_encoded, encoders = pp.encode_categorical(df_fe, fit=True)
top_features = correlation_analysis(df_encoded, target_col='Final_Result', top_n=10)
top_features
""")

# ---------------------------------------------------------------
md("""## 6. Model Training & Comparison

We train and compare 5 classification algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

The `build_dataset()` helper (from `train_model.py`) applies the full
cleaning + feature engineering + encoding + scaling + split pipeline
in one call so training stays consistent with the standalone script.""")

code("""X_train, X_test, y_train, y_test, encoders, scaler, feature_cols = build_dataset(DATA_PATH)
results_df, best_name, best_model, all_models = train_and_compare(X_train, X_test, y_train, y_test)
results_df
""")

code("""plt.figure(figsize=(7,4))
sns.barplot(data=results_df, x='Model', y='Accuracy', palette='crest')
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------
md(f"""## 7. Feature Importance (Random Forest)

Tree-based models expose `feature_importances_`, showing which
features most influence the prediction.""")
code("""rf_model = all_models['Random Forest']
importances = get_feature_importance(rf_model, feature_cols)

plt.figure(figsize=(8,6))
importances.sort_values().plot(kind='barh', color='teal')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------
md("""## 8. Detailed Evaluation of the Best Model

Metrics explained:
- **Accuracy** — overall % of correct predictions
- **Precision** — of predicted Pass/Average/Fail, how many were correct
- **Recall** — of actual Pass/Average/Fail, how many were correctly found
- **F1-score** — harmonic mean of precision & recall
- **Confusion Matrix** — actual vs predicted class breakdown
- **ROC Curve / AUC** — separability of each class at varying thresholds""")

code(f"""class_names = list(encoders['Final_Result'].classes_)
y_pred = best_model.predict(X_test)

metrics = compute_metrics(y_test, y_pred)
metrics
""")

code("""plot_confusion_matrix(y_test, y_pred, class_names, out_dir='../outputs/graphs')
""")

code("""report = print_classification_report(y_test, y_pred, class_names)
print(report)
""")

code("""plot_roc_curves(best_model, X_test, y_test, class_names, out_dir='../outputs/graphs')
""")

# ---------------------------------------------------------------
md("""## 9. Save the Best Model

We persist the trained model along with the LabelEncoders and
StandardScaler used during training, so the exact same preprocessing
can be replayed at inference time (see `src/predict.py` and `app.py`).""")

code("""save_artifacts(best_model, encoders, scaler, feature_cols, model_dir='../models')
""")

# ---------------------------------------------------------------
md("""## 10. Predict on a New Student Record

Using the saved artifacts via `predict.py`, we can predict the result
for a brand-new student.""")

code("""from predict import predict_student

new_student = {
    'Gender': 'Female', 'Age': 18, 'Attendance': 88.0,
    'Study_Hours_Per_Day': 4.5, 'Previous_Marks': 78,
    'Assignment_Score': 82, 'Internal_Marks': 32,
    'Sleep_Hours': 7.0, 'Internet_Usage': 2.0,
    'Participation': 'High', 'Family_Support': 'High'
}

label, proba = predict_student(new_student, model_dir='../models')
print('Predicted Result:', label)
print('Class Probabilities:', proba)
""")

# ---------------------------------------------------------------
md("""## 11. Conclusion

This notebook demonstrated a complete, end-to-end machine learning
workflow for predicting student academic performance: data cleaning,
EDA, feature engineering, multi-model comparison, detailed evaluation,
and a reusable prediction pipeline — all deployed via a Streamlit web
app (`app.py`) for interactive use.

See `README.md` for setup instructions and `report_content.md` for the
full internship report content.""")

nb['cells'] = cells

with open('Student_Performance_Prediction.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created: Student_Performance_Prediction.ipynb")

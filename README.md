# 🎓 Student Performance Prediction Using Machine Learning

An end-to-end Machine Learning project that predicts whether a student will **Pass**, achieve an **Average** result, or **Fail**, based on academic and lifestyle factors such as attendance, study hours, previous marks, and more.

Built as a 1-month AI & Machine Learning internship project.

---

## 📌 Project Overview

Academic performance depends on a mix of measurable factors — attendance, study habits, prior scores, sleep, and lifestyle choices. This project builds a complete ML pipeline that:

- Cleans and explores a student dataset
- Engineers meaningful new features
- Trains and compares 5 classification algorithms
- Evaluates the best model with industry-standard metrics
- Deploys the model through an interactive Streamlit web app

---

## 🎯 Objectives

- Predict `Final_Result` (Pass / Average / Fail) from student data
- Build a complete, modular, end-to-end ML pipeline
- Demonstrate preprocessing, EDA, feature engineering, model training, evaluation, and deployment best practices

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| Web App | Streamlit |
| Notebook | Jupyter Notebook |

---

## 📊 Dataset Information

- **File:** `dataset/student_performance.csv`
- **Records:** 600+ synthetic student records (realistic distributions, intentional noise, a few missing values/outliers/duplicates to mimic real-world data)
- **Target column:** `Final_Result` (`Pass` / `Average` / `Fail`)

| Column | Description |
|---|---|
| Student_ID | Unique student identifier |
| Gender | Male / Female |
| Age | Student age (16–19) |
| Attendance | Attendance percentage |
| Study_Hours_Per_Day | Average daily study hours |
| Previous_Marks | Previous exam marks (%) |
| Assignment_Score | Assignment score (%) |
| Internal_Marks | Internal assessment marks (out of 40) |
| Sleep_Hours | Average daily sleep hours |
| Internet_Usage | Non-study internet usage (hrs/day) |
| Participation | Class participation level (Low/Medium/High) |
| Family_Support | Family academic support level (Low/Medium/High) |
| Final_Result | Target: Pass / Average / Fail |

> ⚠️ This is a **synthetic dataset** generated for academic/educational purposes — it does not represent real students.

---

## 📂 Project Structure

```
Student_Performance_Prediction/
│
├── dataset/
│   └── student_performance.csv
│
├── notebooks/
│   ├── Student_Performance_Prediction.ipynb
│   └── build_notebook.py
│
├── src/
│   ├── generate_dataset.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluation.py
│   ├── predict.py
│   ├── visualization.py
│   └── run_pipeline.py
│
├── models/
│   ├── student_performance_model.pkl
│   ├── label_encoders.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── outputs/
│   ├── graphs/
│   ├── reports/
│   └── predictions.csv
│
├── app.py
├── requirements.txt
├── README.md
└── report_content.md
```

---

## ⚙️ Installation Steps

1. **Clone / download the project folder**
   ```bash
   cd Student_Performance_Prediction
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Running Instructions

### 1. (Optional) Regenerate the dataset
```bash
python src/generate_dataset.py
```

### 2. Run the full pipeline (EDA + training + evaluation + save model)
```bash
python src/run_pipeline.py
```
This single command:
- Cleans the data
- Generates all EDA graphs → `outputs/graphs/`
- Trains & compares 5 ML models
- Evaluates the best model → `outputs/graphs/`, `outputs/reports/`
- Saves the trained model → `models/`
- Generates sample predictions → `outputs/predictions.csv`

### 3. Explore the Jupyter Notebook
```bash
jupyter notebook notebooks/Student_Performance_Prediction.ipynb
```

### 4. Make a prediction from the command line
```bash
python src/predict.py
```

### 5. Launch the Streamlit web app
```bash
streamlit run app.py
```

---

## 📈 Results

Five models were trained and compared on a held-out test set (20% of the data, stratified by class):

| Model | Accuracy | F1-score |
|---|---|---|
| **Logistic Regression** | **~59%** | **~0.59** |
| SVM | ~58% | ~0.58 |
| Random Forest | ~54% | ~0.54 |
| Decision Tree | ~47% | ~0.46 |
| KNN | ~43% | ~0.42 |

> Exact numbers vary slightly with the random seed. The moderate accuracy (rather than a suspiciously perfect one) is intentional — the synthetic dataset includes realistic noise, mirroring how real student outcomes are influenced by many unmeasured factors.

The best-performing model (Logistic Regression) is saved and served through the Streamlit app.

Detailed metrics, confusion matrix, classification report, and ROC curves are available in `outputs/graphs/` and `outputs/reports/`.

---

## 🚀 Future Scope

- Collect and train on real (anonymized) student data
- Add more behavioral features (e.g., quiz frequency, LMS engagement)
- Try ensemble/boosting models (XGBoost, LightGBM) and hyperparameter tuning
- Add explainability (SHAP/LIME) so predictions can be interpreted by teachers
- Deploy the Streamlit app to the cloud (Streamlit Community Cloud / Render)
- Build a teacher dashboard for batch predictions across an entire class

---

## 👤 Author

**Balamurugan S**
B.Tech Information Technology, A.V.C College of Engineering (Anna University)

---

## 📄 License

This project is created for academic/educational purposes as part of an AI & ML internship.

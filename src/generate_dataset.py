"""
generate_dataset.py
--------------------
Generates a realistic, synthetic dataset for the "Student Performance
Prediction Using Machine Learning" project.

Why synthetic data?
Real student records are private/sensitive, so for an academic project we
build a synthetic dataset whose statistical relationships mimic real-world
behaviour (e.g., higher attendance and study hours generally lead to better
results), while still containing natural noise so the ML problem is not
trivially easy. A fixed random seed makes the dataset reproducible.

Run:
    python src/generate_dataset.py
Output:
    dataset/student_performance.csv (600 records)
"""

import numpy as np
import pandas as pd
import os

# Fixed seed => reproducible dataset every time this script runs
np.random.seed(42)

N_RECORDS = 600  # more than the required minimum of 500


def generate_student_data(n=N_RECORDS):
    student_ids = [f"STU{str(i).zfill(4)}" for i in range(1, n + 1)]

    gender = np.random.choice(["Male", "Female"], size=n, p=[0.52, 0.48])
    age = np.random.randint(16, 20, size=n)  # typical school/college age range

    attendance = np.clip(np.random.normal(78, 12, n), 40, 100).round(1)
    study_hours = np.clip(np.random.normal(3.2, 1.4, n), 0.2, 8.0).round(1)
    previous_marks = np.clip(np.random.normal(65, 15, n), 20, 100).round(1)
    assignment_score = np.clip(np.random.normal(70, 14, n), 10, 100).round(1)
    internal_marks = np.clip(np.random.normal(28, 6, n), 5, 40).round(1)  # out of 40
    sleep_hours = np.clip(np.random.normal(6.5, 1.2, n), 3, 10).round(1)
    internet_usage = np.clip(np.random.normal(3.0, 1.8, n), 0, 10).round(1)  # hrs/day (non-study)
    participation = np.random.choice(["Low", "Medium", "High"], size=n, p=[0.3, 0.45, 0.25])
    family_support = np.random.choice(["Low", "Medium", "High"], size=n, p=[0.2, 0.45, 0.35])

    # ---- Build a latent "performance score" that drives the label ----
    # Positive influences: attendance, study_hours, previous_marks,
    # assignment_score, internal_marks, sleep_hours, family_support
    # Negative influence: excessive internet_usage (beyond productive use)
    support_map = {"Low": 0, "Medium": 1, "High": 2}
    participation_map = {"Low": 0, "Medium": 1, "High": 2}

    score = (
        0.18 * attendance
        + 6.0 * study_hours
        + 0.35 * previous_marks
        + 0.30 * assignment_score
        + 0.9 * internal_marks
        + 2.0 * sleep_hours
        - 2.2 * internet_usage
        + 5.0 * np.vectorize(support_map.get)(family_support)
        + 4.0 * np.vectorize(participation_map.get)(participation)
    )

    # Add natural random noise so results aren't perfectly deterministic
    score += np.random.normal(0, 12, n)

    # Convert the continuous latent score into 3 classes using percentile
    # thresholds so classes are reasonably balanced but not perfectly equal
    fail_threshold = np.percentile(score, 30)
    pass_threshold = np.percentile(score, 70)

    final_result = np.where(
        score < fail_threshold, "Fail",
        np.where(score < pass_threshold, "Average", "Pass")
    )

    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Gender": gender,
        "Age": age,
        "Attendance": attendance,
        "Study_Hours_Per_Day": study_hours,
        "Previous_Marks": previous_marks,
        "Assignment_Score": assignment_score,
        "Internal_Marks": internal_marks,
        "Sleep_Hours": sleep_hours,
        "Internet_Usage": internet_usage,
        "Participation": participation,
        "Family_Support": family_support,
        "Final_Result": final_result,
    })

    # ---- Inject a small amount of realistic messiness ----
    # 1) A handful of missing values (simulates incomplete records)
    for col in ["Attendance", "Sleep_Hours", "Assignment_Score"]:
        missing_idx = np.random.choice(df.index, size=int(0.015 * n), replace=False)
        df.loc[missing_idx, col] = np.nan

    # 2) A few exact duplicate rows (simulates data-entry duplication)
    dup_rows = df.sample(5, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 3) A few extreme outliers (simulates data-entry errors / edge cases)
    outlier_idx = np.random.choice(df.index, size=4, replace=False)
    df.loc[outlier_idx, "Study_Hours_Per_Day"] = [12.5, 0.0, 11.8, 0.1]

    return df


if __name__ == "__main__":
    df = generate_student_data()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "student_performance.csv")
    df.to_csv(out_path, index=False)
    print(f"Dataset generated with {len(df)} rows -> {out_path}")
    print(df["Final_Result"].value_counts())

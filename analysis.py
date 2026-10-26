import pandas as pd
from pathlib import Path

# Dataset file ka path
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "student-mat.csv"

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset nahi mila: {DATASET_PATH}\n"
        "Check karo ki CSV file data folder ke andar hai."
    )

df = pd.read_csv(DATASET_PATH, sep=";")

print("\nSTUDENT PERFORMANCE DATASET")
print("-" * 35)
print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 student records:")
print(df.head())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nDataset loaded successfully!")
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Basic statistical summary
print("\nSTATISTICAL SUMMARY")
print(df[["studytime", "absences", "G1", "G2", "G3"]].describe())

# 2. Final grade distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=15, kde=True)
plt.title("Distribution of Final Student Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(BASE_DIR / "final_grade_distribution.png")
plt.show()

# 3. Study time vs final grade
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="studytime", y="G3")
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time Category")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(BASE_DIR / "studytime_vs_grade.png")
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(8, 5))
correlation = df[["studytime", "absences", "G1", "G2", "G3"]].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Study Time, Absences and Grades")
plt.tight_layout()
plt.savefig(BASE_DIR / "grade_correlation.png")
plt.show()

print("\nGraphs saved successfully!")
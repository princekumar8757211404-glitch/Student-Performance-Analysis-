import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "student-mat.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "student_performance_model.pkl"

# Load dataset
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH, sep=";")

# Select features and target
features = ["studytime", "absences", "G1", "G2"]
target = "G3"

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model on unseen test data
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nSTUDENT PERFORMANCE MODEL")
print("-" * 35)
print(f"Total records: {len(df)}")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2): {r2:.3f}")

# Save trained model and feature names
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {"model": model, "features": features},
    MODEL_PATH
)

print(f"\nModel saved successfully: {MODEL_PATH}")

# Example prediction
example = pd.DataFrame(
    [[3, 4, 10, 12]],
    columns=features
)
predicted_grade = model.predict(example)[0]

print(f"Example predicted final grade: {predicted_grade:.1f}/20")
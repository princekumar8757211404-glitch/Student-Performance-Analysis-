import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "student-mat.csv"
MODEL_PATH = BASE_DIR / "student_performance_model.pkl"
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Student Performance Analysis")
st.write(
    "Explore student data and predict final grades "
    "using a trained Machine Learning model."
)

if not DATASET_PATH.exists():
    st.error("Dataset not found. Check the data folder.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Model not found. Run train_model.py first.")
    st.stop()

df = pd.read_csv(DATASET_PATH, sep=";")
saved_model = joblib.load(MODEL_PATH)
model = saved_model["model"]
features = saved_model["features"]

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Students", len(df))
col2.metric("Average Final Grade", f"{df['G3'].mean():.2f}/20")
col3.metric("Average Absences", f"{df['absences'].mean():.1f}")

st.subheader("Student Data")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Final Grade Distribution")
grade_counts = (
    df["G3"]
    .value_counts()
    .sort_index()
)
st.bar_chart(grade_counts)

st.subheader("Predict Final Grade")

studytime = st.selectbox(
    "Weekly study time category",
    options=[1, 2, 3, 4],
    index=1,
    help="1: less than 2 hours; 2: 2–5 hours; "
         "3: 5–10 hours; 4: more than 10 hours"
)

absences = st.number_input(
    "Number of school absences",
    min_value=0,
    max_value=100,
    value=4
)

g1 = st.slider("First period grade (G1)", 0, 20, 10)
g2 = st.slider("Second period grade (G2)", 0, 20, 12)

if st.button("Predict Final Grade"):
    input_data = pd.DataFrame(
        [[studytime, absences, g1, g2]],
        columns=features
    )
    prediction = float(model.predict(input_data)[0])
    prediction = max(0.0, min(20.0, prediction))

    st.success(f"Predicted final grade: {prediction:.1f}/20")
    st.caption(
        "This is an estimate from historical data, "
        "not a guaranteed actual grade."
    )

st.caption(
    "Dataset: UCI Student Performance dataset. "
    "Predictions are for educational demonstration."
)

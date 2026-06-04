import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------

# PAGE SETTINGS

# -------------------------------

st.set_page_config(
page_title="Student Performance Prediction",
page_icon="🎓",
layout="wide"
)

st.markdown("""
<style>

/* Soft Study Background */
.stApp{
    background:
    linear-gradient(
        rgba(15,23,42,0.65),
        rgba(15,23,42,0.65)
    ),
    url("https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?auto=format&fit=crop&w=1600&q=80");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Text */
h1, h2, h3, h4, h5, h6, p, label, span {
    color: white !important;
}

/* Buttons */
.stButton > button{
    background: linear-gradient(90deg,#22c55e,#16a34a);
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    border: none;
}

/* Metric Cards */
[data-testid="stMetric"]{
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

st.title("🎓 Student Performance Prediction System")
st.markdown("Predict whether a student will PASS or FAIL")

# -------------------------------

# LOAD DATASET

# -------------------------------

df = pd.read_csv("student-mat 1.csv")

# Create Target Variable

df["Result"] = df["G3"].apply(
lambda x: 1 if x >= 10 else 0
)

# Remove G3

df.drop("G3", axis=1, inplace=True)

# Encode Categorical Features

le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# -------------------------------

# TRAIN MODEL

# -------------------------------

X = df.drop("Result", axis=1)
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

model = RandomForestClassifier(
n_estimators=100,
random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

# -------------------------------

# ACCURACY

# -------------------------------

accuracy = accuracy_score(
y_test,
pred
)

st.metric(
"Model Accuracy",
f"{accuracy:.2%}"
)

# -------------------------------
# STUDENT INPUT FORM
# -------------------------------

st.subheader("📚 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    studytime = st.slider("Study Time", 1, 4, 2)
    failures = st.slider("Previous Failures", 0, 4, 0)

with col2:
    absences = st.slider("Absences", 0, 100, 5)

# Agar dataset me G1 aur G2 hain
g1 = st.number_input("First Period Grade (G1)", 0, 20, 10)
g2 = st.number_input("Second Period Grade (G2)", 0, 20, 10)

# -------------------------------

# PREDICTION

# -------------------------------

if st.button("🎯 Predict Result"):

    sample = X.iloc[[0]].copy()

    if "studytime" in sample.columns:
        sample["studytime"] = studytime

    if "failures" in sample.columns:
        sample["failures"] = failures

    if "absences" in sample.columns:
        sample["absences"] = absences

    if "G1" in sample.columns:
        sample["G1"] = g1

    if "G2" in sample.columns:
        sample["G2"] = g2

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.success("✅ Prediction: PASS")
    else:
        st.error("❌ Prediction: FAIL")



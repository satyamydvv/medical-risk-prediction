import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

# Page config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Title
st.title("🩺 Heart Disease Risk Prediction")
st.markdown("Enter patient health details to predict risk level")
st.markdown("---")

# ---- INPUTS ----
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2, 3])

feature_names = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak",
    "slope","ca","thal"
]

# ---- PREDICTION ----
if st.button("Predict Risk"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Model prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # ---- RESULT ----
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({probability:.2f})")
    else:
        st.success(f"✅ Low Risk ({probability:.2f})")

    st.subheader("📊 Risk Probability")
    st.progress(int(probability * 100))

    # =========================
    # 📈 FEATURE IMPORTANCE
    # =========================
    st.markdown("---")
    st.subheader("🧠 Key Risk Factors (Model Insight)")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_features = np.array(feature_names)[indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots()
    ax.bar(range(len(importances)), sorted_importances)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(sorted_features, rotation=90)
    ax.set_title("Feature Importance")

    st.pyplot(fig)

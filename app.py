import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Multiple Disease Prediction System",
    layout="wide"
)

# -------------------- ROBOTO FONT --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Roboto', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODELS --------------------
diabetes_model = pickle.load(open("saved_models/diabetes_model.sav", "rb"))
heart_disease_model = pickle.load(open("saved_models/heart_disease_model.sav", "rb"))
parkinsons_model = pickle.load(open("saved_models/parkinsons_model.sav", "rb"))
breast_cancer_model = pickle.load(open("saved_models/breast_cancer_model.sav", "rb"))

# -------------------- SIDEBAR --------------------
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction System",
        ["Diabetes Prediction", "Heart Disease Prediction",
         "Parkinsons Prediction", "Breast Cancer Prediction"],
        icons=["activity", "heart-pulse", "person", "gender-female"],
        default_index=0
    )

# =====================================================
# 🩸 DIABETES
# =====================================================
if selected == "Diabetes Prediction":
    st.title("🩸 Diabetes Prediction")

    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)

        with col1:
            Pregnancies = st.text_input("Number of Pregnancies", placeholder="e.g. 0")
            Glucose = st.text_input("Glucose Level (mg/dL)", placeholder="e.g. 120")
            BloodPressure = st.text_input("Blood Pressure (mmHg)", placeholder="e.g. 80")
            SkinThickness = st.text_input("Skin Thickness (mm)", placeholder="e.g. 20")

        with col2:
            Insulin = st.text_input("Insulin Level (µIU/mL)", placeholder="e.g. 85")
            BMI = st.text_input("BMI", placeholder="e.g. 24.5")
            DPF = st.text_input("Diabetes Pedigree Function", placeholder="e.g. 0.45")
            Age = st.text_input("Age", placeholder="e.g. 30")

        submit = st.form_submit_button("Check Diabetes")

    if submit:
        inputs = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                  Insulin, BMI, DPF, Age]

        if any(i.strip() == "" for i in inputs):
            st.warning("⚠️ Please fill in all fields")
        else:
            try:
                data = np.array([[int(Pregnancies), float(Glucose), float(BloodPressure),
                                  float(SkinThickness), float(Insulin), float(BMI),
                                  float(DPF), int(Age)]])
                prediction = diabetes_model.predict(data)

                if prediction[0] == 1:
                    st.error("❌ Person is likely to have Diabetes")
                else:
                    st.success("✅ Person is NOT likely to have Diabetes")
            except:
                st.error("❌ Please enter valid numeric values")

# =====================================================
# ❤️ HEART DISEASE
# =====================================================
if selected == "Heart Disease Prediction":
    st.title("❤️ Heart Disease Prediction")

    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input("Age", "e.g. 45")
            sex = st.text_input("Sex (1=Male, 0=Female)", "e.g. 1")
            cp = st.text_input("Chest Pain Type (0–3)", "e.g. 2")
            trestbps = st.text_input("Resting BP (mmHg)", "e.g. 120")

        with col2:
            chol = st.text_input("Cholesterol (mg/dL)", "e.g. 230")
            fbs = st.text_input("Fasting Blood Sugar (>120 = 1)", "e.g. 0")
            restecg = st.text_input("Rest ECG (0–2)", "e.g. 1")
            thalach = st.text_input("Max Heart Rate", "e.g. 150")

        with col3:
            exang = st.text_input("Exercise Induced Angina (1=Yes)", "e.g. 0")
            oldpeak = st.text_input("ST Depression", "e.g. 1.2")
            slope = st.text_input("Slope (0–2)", "e.g. 1")
            ca = st.text_input("Major Vessels (0–3)", "e.g. 0")
            thal = st.text_input("Thal (1–3)", "e.g. 2")

        submit = st.form_submit_button("Check Heart Disease")

    if submit:
        inputs = [age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]

        if any(i.strip() == "" for i in inputs):
            st.warning("⚠️ Please fill in all fields")
        else:
            try:
                data = np.array([[float(i) for i in inputs]])
                prediction = heart_disease_model.predict(data)

                if prediction[0] == 1:
                    st.error("❌ Person is likely to have Heart Disease")
                else:
                    st.success("✅ Person is NOT likely to have Heart Disease")
            except:
                st.error("❌ Invalid numeric input")

# =====================================================
# 🧠 PARKINSON'S
# =====================================================
if selected == "Parkinsons Prediction":
    st.title("🧠 Parkinson’s Disease Prediction")

    features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
        "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer",
        "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]

    inputs = []
    with st.form("parkinsons_form"):
        cols = st.columns(3)
        for i, feature in enumerate(features):
            with cols[i % 3]:
                inputs.append(st.text_input(feature, placeholder="e.g. 0.12"))

        submit = st.form_submit_button("Check Parkinson’s")

    if submit:
        if any(i.strip() == "" for i in inputs):
            st.warning("⚠️ Please fill in all fields")
        else:
            try:
                data = np.array([[float(i) for i in inputs]])
                prediction = parkinsons_model.predict(data)

                if prediction[0] == 1:
                    st.error("❌ Person is likely to have Parkinson’s Disease")
                else:
                    st.success("✅ Person is NOT likely to have Parkinson’s Disease")
            except:
                st.error("❌ Invalid numeric values")

# =====================================================
# 🎀 BREAST CANCER
# =====================================================
if selected == "Breast Cancer Prediction":
    st.title("🎀 Breast Cancer Prediction")

    features = [
        "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area",
        "Mean Smoothness", "Mean Compactness", "Mean Concavity",
        "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension"
    ]

    inputs = []
    with st.form("cancer_form"):
        col1, col2 = st.columns(2)
        for i, f in enumerate(features):
            with col1 if i < 5 else col2:
                inputs.append(st.text_input(f, placeholder="e.g. 14.5"))

        submit = st.form_submit_button("Check Breast Cancer")

    if submit:
        if any(i.strip() == "" for i in inputs):
            st.warning("⚠️ Please fill in all fields")
        else:
            try:
                data = np.array([[float(i) for i in inputs]])
                prediction = breast_cancer_model.predict(data)

                if prediction[0] == 1:
                    st.error("❌ Malignant Tumor Detected")
                else:
                    st.success("✅ Benign Tumor")
            except:
                st.error("❌ Invalid numeric input")


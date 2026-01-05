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
# ==================================================
# HEART DISEASE PAGE
# ==================================================
if selected == "Heart Disease Prediction":
    st.title("🫀Heart Disease Prediction")

    st.markdown('<div class="main-title">Heart Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Clinical parameters based heart risk analysis</div>', unsafe_allow_html=True)

    with st.form("heart_form"):
        age = st.text_input("Age (years)", placeholder="e.g. 45")
        sex = st.text_input("Sex (1 = Male, 0 = Female)", placeholder="e.g. 1")
        cp = st.text_input("Chest Pain Type (0–3)", placeholder="e.g. 2")
        trestbps = st.text_input("Resting Blood Pressure (mm Hg)", placeholder="e.g. 130")
        chol = st.text_input("Serum Cholesterol (mg/dL)", placeholder="e.g. 250")
        fbs = st.text_input("Fasting Blood Sugar > 120 mg/dL (1 = True, 0 = False)", placeholder="e.g. 0")
        restecg = st.text_input("Resting ECG Results (0–2)", placeholder="e.g. 1")
        thalach = st.text_input("Maximum Heart Rate Achieved", placeholder="e.g. 150")
        exang = st.text_input("Exercise Induced Angina (1 = Yes, 0 = No)", placeholder="e.g. 0")
        oldpeak = st.text_input("ST Depression Induced by Exercise", placeholder="e.g. 1.2")
        slope = st.text_input("Slope of Peak Exercise ST Segment (0–2)", placeholder="e.g. 1")
        ca = st.text_input("Number of Major Vessels (0–4)", placeholder="e.g. 0")
        thal = st.text_input("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible)", placeholder="e.g. 2")

        submit = st.form_submit_button("Check Heart Disease")

    if submit:
        fields = [
            age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak,
            slope, ca, thal
        ]

        if any(f.strip() == "" for f in fields):
            st.warning("⚠️ Please fill in all fields")
        else:
            try:
                data = [
                    float(age), float(sex), float(cp), float(trestbps),
                    float(chol), float(fbs), float(restecg), float(thalach),
                    float(exang), float(oldpeak), float(slope),
                    float(ca), float(thal)
                ]

                st.success(predict_heart(data))

            except ValueError:
                st.error("❌ Please enter valid numeric values only")


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




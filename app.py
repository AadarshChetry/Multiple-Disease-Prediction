import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Multiple Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS (ROBOTO FONT)
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Roboto', sans-serif;
}

.main-title {
    font-size: 42px;
    font-weight: 600;
    color: #0b5394;
    text-align: center;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    color: #555;
    font-size: 16px;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODELS (FIXED)
# --------------------------------------------------
diabetes_model = pickle.load(open("saved_models/diabetes_model.sav", "rb"))
heart_disease_model = pickle.load(open("saved_models/heart_disease_model.sav", "rb"))
parkinsons_model = pickle.load(open("saved_models/parkinsons_model.sav", "rb"))
breast_cancer_model = pickle.load(open("saved_models/breast_cancer_model.sav", "rb"))

# --------------------------------------------------
# PREDICTION FUNCTIONS
# --------------------------------------------------
def predict_diabetes(data):
    return "🟢 Person is NOT Diabetic" if diabetes_model.predict([data])[0] == 0 else "🔴 Person is Diabetic"

def predict_heart(data):
    return "🟢 No Heart Disease Detected" if heart_disease_model.predict([data])[0] == 0 else "🔴 Heart Disease Detected"

def predict_parkinsons(data):
    return "🟢 No Parkinson’s Disease" if parkinsons_model.predict([data])[0] == 0 else "🔴 Parkinson’s Disease Detected"

def predict_breast_cancer(data):
    return "🟢 Benign Tumor" if breast_cancer_model.predict([data])[0] == 0 else "🔴 Malignant Tumor"

# --------------------------------------------------
# SIDEBAR MENU
# --------------------------------------------------
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction System",
        ["Diabetes Prediction", "Heart Disease Prediction",
         "Parkinsons Prediction", "Breast Cancer Prediction"],
        icons=["activity", "heart-pulse", "person", "gender-female"],
        default_index=0
    )

# ==================================================
# DIABETES PAGE
# ==================================================
if selected == "Diabetes Prediction":

    st.markdown('<div class="main-title">Diabetes Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-based diabetes risk assessment</div>', unsafe_allow_html=True)

    with st.form("diabetes_form"):
        Pregnancies = st.text_input("Number of Pregnancies")
        Glucose = st.text_input("Glucose Level")
        BloodPressure = st.text_input("Blood Pressure")
        SkinThickness = st.text_input("Skin Thickness")
        Insulin = st.text_input("Insulin Level")
        BMI = st.text_input("BMI")
        DPF = st.text_input("Diabetes Pedigree Function")
        Age = st.text_input("Age")

        submit = st.form_submit_button("Check Diabetes")

    if submit:
        try:
            data = [
                int(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DPF), int(Age)
            ]
            st.success(predict_diabetes(data))
        except:
            st.error("❌ Enter valid numeric values")

# ==================================================
# HEART DISEASE PAGE
# ==================================================
if selected == "Heart Disease Prediction":

    st.markdown('<div class="main-title">Heart Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Early detection of heart disease</div>', unsafe_allow_html=True)

    with st.form("heart_form"):
        inputs = [st.text_input(f"Feature {i+1}") for i in range(13)]
        submit = st.form_submit_button("Check Heart Disease")

    if submit:
        try:
            data = [float(x) for x in inputs]
            st.success(predict_heart(data))
        except:
            st.error("❌ Enter valid numeric values")

# ==================================================
# PARKINSONS PAGE
# ==================================================
if selected == "Parkinsons Prediction":

    st.markdown('<div class="main-title">Parkinson’s Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Voice-based biomedical analysis</div>', unsafe_allow_html=True)

    features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
        "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer",
        "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]

    with st.form("parkinsons_form"):
        inputs = [st.text_input(f) for f in features]
        submit = st.form_submit_button("Check Parkinson’s")

    if submit:
        try:
            data = [float(x) for x in inputs]
            st.success(predict_parkinsons(data))
        except:
            st.error("❌ Enter valid numeric values")

# ==================================================
# BREAST CANCER PAGE (CORRECT)
# ==================================================
if selected == "Breast Cancer Prediction":

    st.markdown('<div class="main-title">Breast Cancer Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Tumor classification using ML</div>', unsafe_allow_html=True)

    features = [
        "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area",
        "Mean Smoothness", "Mean Compactness", "Mean Concavity",
        "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
        "Radius Error", "Texture Error", "Perimeter Error", "Area Error",
        "Smoothness Error", "Compactness Error", "Concavity Error",
        "Concave Points Error", "Symmetry Error", "Fractal Dimension Error",
        "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area",
        "Worst Smoothness", "Worst Compactness", "Worst Concavity",
        "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
    ]

    with st.form("breast_form"):
        inputs = [st.text_input(f) for f in features]
        submit = st.form_submit_button("Check Breast Cancer")

    if submit:
        try:
            data = [float(x) for x in inputs]
            st.success(predict_breast_cancer(data))
        except:
            st.error("❌ Enter valid numeric values")


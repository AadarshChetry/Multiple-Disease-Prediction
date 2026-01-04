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
# LOAD MODELS
# --------------------------------------------------
diabetes_model = pickle.load(open(
    "C:/Users/Aadarsh Chetry/Documents/6thSemProject/Multiple Disease Prediction/saved_models/diabetes_model.sav", "rb"
))

heart_disease_model = pickle.load(open(
    "C:/Users/Aadarsh Chetry/Documents/6thSemProject/Multiple Disease Prediction/saved_models/heart_disease_model.sav", "rb"
))

parkinsons_model = pickle.load(open(
    "C:/Users/Aadarsh Chetry/Documents/6thSemProject/Multiple Disease Prediction/saved_models/parkinsons_model.sav", "rb"
))

# --------------------------------------------------
# PREDICTION FUNCTIONS
# --------------------------------------------------
def predict_diabetes(data):
    return "🟢 Person is NOT Diabetic" if diabetes_model.predict([data])[0] == 0 else "🔴 Person is Diabetic"


def predict_heart(data):
    return "🟢 No Heart Disease Detected" if heart_disease_model.predict([data])[0] == 0 else "🔴 Heart Disease Detected"


def predict_parkinsons(data):
    return "🟢 No Parkinson’s Disease" if parkinsons_model.predict([data])[0] == 0 else "🔴 Parkinson’s Disease Detected"

# --------------------------------------------------
# SIDEBAR MENU
# --------------------------------------------------
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction System",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"],
        icons=["activity", "heart-pulse", "person"],
        default_index=0
    )

# ==================================================
# DIABETES PAGE
# ==================================================
if selected == "Diabetes Prediction":

    st.markdown('<div class="main-title">Diabetes Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-based diabetes risk assessment</div>', unsafe_allow_html=True)

    with st.form("diabetes_form"):
        Pregnancies = st.text_input("Number of Pregnancies", placeholder="e.g. 0")
        Glucose = st.text_input("Glucose Level (mg/dL)", placeholder="e.g. 120")
        BloodPressure = st.text_input("Blood Pressure (mmHg)", placeholder="e.g. 80")
        SkinThickness = st.text_input("Skin Thickness (mm)", placeholder="e.g. 20")
        Insulin = st.text_input("Insulin Level (µIU/mL)", placeholder="e.g. 85")
        BMI = st.text_input("Body Mass Index (BMI)", placeholder="e.g. 24.5")
        DPF = st.text_input("Diabetes Pedigree Function", placeholder="e.g. 0.45")
        Age = st.text_input("Age (years)", placeholder="e.g. 30")

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
            st.error("❌ Please enter valid numeric values in all fields")

# ==================================================
# HEART DISEASE PAGE
# ==================================================
if selected == "Heart Disease Prediction":

    st.markdown('<div class="main-title">Heart Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Early detection of heart disease</div>', unsafe_allow_html=True)

    with st.form("heart_form"):
        age = st.text_input("Age", placeholder="e.g. 45")
        sex = st.text_input("Sex (1 = Male, 0 = Female)", placeholder="e.g. 1")
        cp = st.text_input("Chest Pain Type (0–3)", placeholder="e.g. 2")
        trestbps = st.text_input("Resting Blood Pressure", placeholder="e.g. 130")
        chol = st.text_input("Cholesterol (mg/dL)", placeholder="e.g. 250")
        fbs = st.text_input("Fasting Blood Sugar > 120 (1/0)", placeholder="e.g. 0")
        restecg = st.text_input("Resting ECG (0–2)", placeholder="e.g. 1")
        thalach = st.text_input("Maximum Heart Rate", placeholder="e.g. 150")
        exang = st.text_input("Exercise Induced Angina (1/0)", placeholder="e.g. 0")
        oldpeak = st.text_input("Oldpeak", placeholder="e.g. 1.2")
        slope = st.text_input("Slope (0–2)", placeholder="e.g. 1")
        ca = st.text_input("CA (0–4)", placeholder="e.g. 0")
        thal = st.text_input("Thal (1–3)", placeholder="e.g. 2")

        submit = st.form_submit_button("Check Heart Disease")

    if submit:
        try:
            data = [
                float(age), float(sex), float(cp), float(trestbps),
                float(chol), float(fbs), float(restecg), float(thalach),
                float(exang), float(oldpeak), float(slope),
                float(ca), float(thal)
            ]
            st.success(predict_heart(data))
        except:
            st.error("❌ Please enter valid numeric values in all fields")

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
        inputs = [
            st.text_input(feature, placeholder="e.g. 0.012")
            for feature in features
        ]
        submit = st.form_submit_button("Check Parkinson’s Disease")

    if submit:
        try:
            data = [float(x) for x in inputs]
            st.success(predict_parkinsons(data))
        except:
            st.error("❌ Please enter valid numeric values in all fields")

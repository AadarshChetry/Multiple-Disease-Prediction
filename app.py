import streamlit as st
import pickle
import os
from streamlit_option_menu import option_menu
import numpy as np 

st.set_page_config(page_title="Multiple Disease Prediction", layout="wide", page_icon="🏥➕")


diabetes_model = pickle.load(open("C:/Users/Aadarsh Chetry/Documents/6thSemProject/Multiple Disease Prediction/saved_models/diabetes_model.sav", "rb"))
diabetes_scaler = pickle.load(open("C:/Users/Aadarsh Chetry/Documents/6thSemProject/Multiple Disease Prediction/saved_models/scaler.sav", "rb"))
heart_disease_model = pickle.load(open("C:/Users/Aadarsh Chetry/Documents/6thSemProject/Multiple Disease Prediction/saved_models/heart_disease_model.sav", "rb"))
heart_scaler = pickle.load(open("C:/Users/Aadarsh Chetry/Documents/6thSemProject/Multiple Disease Prediction/saved_models/heart_scaler.pkl", "rb"))
parkinsons_model = pickle.load(open("C:/Users/Aadarsh Chetry/Documents/6thSemProject/Multiple Disease Prediction/saved_models/parkinsons_model.sav", "rb"))
parkinsons_scaler = pickle.load(open("C:/Users/Aadarsh Chetry/Documents/6thSemProject/Multiple Disease Prediction/saved_models/parkinsons_scaler.pkl", "rb"))
breast_cancer_model = pickle.load(open("C:/Users/Aadarsh Chetry/Documents/6thSemProject/Multiple Disease Prediction/saved_models/breast_cancer_model.sav", "rb"))
breast_cancer_scaler = pickle.load(open("C:/Users/Aadarsh Chetry/Documents/6thSemProject/Multiple Disease Prediction/saved_models/breast_cancer_scaler.sav", "rb"))


with st.sidebar:
   selected = option_menu("Multiple Disease Prediction System",
                ["Diabetes Prediction",
                 "Heart Disease Prediction",
                 "Parkinson's Prediction",
                 "Breast Cancer Prediction"],
                 menu_icon='hospital-fill',
                 icons=['activity','heart-pulse','person', 'gender-female'],
                 default_index=0)


 # Diabetes Prediction Page   
if selected == "Diabetes Prediction":

    # Page Title
    st.title("🩸 Diabetes Prediction")

    Pregnancies = st.text_input("Number of Pregnancies", placeholder="e.g. 0")
    Glucose = st.text_input("Glucose Level (mg/dL)", placeholder="e.g. 120")
    BloodPressure = st.text_input("Blood Pressure (mmHg)", placeholder="e.g. 80")
    SkinThickness = st.text_input("Skin Thickness (mm)", placeholder="e.g. 20")
    Insulin = st.text_input("Insulin Level (µIU/mL)", placeholder="e.g. 85")
    BMI = st.text_input("Body Mass Index (BMI)", placeholder="e.g. 24.5")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function", placeholder="e.g. 0.45")
    Age = st.text_input("Age (years)", placeholder="e.g. 30")

    # code for Prediction
    diabetes_diagnosis = ""

    if st.button("Diabetes Test Result"):
    
        input_data = [
        
            float(Pregnancies),
            float(Glucose),
            float(BloodPressure),
            float(SkinThickness),
            float(Insulin),
            float(BMI),
            float(DiabetesPedigreeFunction),
            float(Age)
        ]

        input_data_scaled = diabetes_scaler.transform([input_data])
        diabetes_diagnosis = diabetes_model.predict(input_data_scaled)

    
        if diabetes_diagnosis[0] == 1:
               diabetes_diagnosis = "The person is diabetic"
        else:
              diabetes_diagnosis = "The person is not diabetic"

        st.success(diabetes_diagnosis)


# Heart Disease Prediction Page
if selected == "Heart Disease Prediction":

    st.title("❤️ Heart Disease Prediction")

    age = st.number_input("Age", min_value=1, max_value=120)

    sex = st.selectbox("Sex", ["Female", "Male"])
    sex = 0 if sex == "Female" else 1

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    )
    chest_pain = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(chest_pain)

    resting_bps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250)
    cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600)

    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", ["No", "Yes"])
    fasting_bs = 1 if fasting_bs == "Yes" else 0

    resting_ecg = st.selectbox(
        "Resting ECG Results",
        ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"]
    )
    resting_ecg = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(resting_ecg)

    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250)

    exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exercise_angina = 1 if exercise_angina == "Yes" else 0

    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, format="%.2f")

    st_slope = st.selectbox(
        "ST Slope",
        ["Upsloping", "Flat", "Downsloping"]
    )
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    heart_diagnosis = ""

    if st.button("Heart Disease Test Result"):

        input_data = [
            age, sex, chest_pain, resting_bps, cholesterol,
            fasting_bs, resting_ecg, max_hr,
            exercise_angina, oldpeak, st_slope
        ]

        # Scale input (since we used StandardScaler in training)
        input_data_scaled = heart_scaler.transform([input_data])
        probability = heart_disease_model.predict_proba(input_data_scaled)

        prediction = heart_disease_model.predict(input_data_scaled)
        probability = heart_disease_model.predict_proba(input_data_scaled)

        no_disease_prob = probability[0][0]
        disease_prob = probability[0][1]

        st.subheader("Prediction Result")

        if prediction[0] == 0:
            st.success(f"🟢 Low Risk of Heart Disease")
            st.write(f"Confidence: {no_disease_prob*100:.2f}%")
        else:
            st.error(f"🔴 High Risk of Heart Disease")
            st.write(f"Confidence: {disease_prob*100:.2f}%")

        st.write("---")
        st.write("### Detailed Probabilities")
        st.write(f"No Disease: {no_disease_prob*100:.2f}%")
        st.write(f"Heart Disease: {disease_prob*100:.2f}%")

        st.progress(float(disease_prob))



# Parkinson's Prediction Page
if selected == "Parkinson's Prediction":

    st.title("🧠 Parkinson’s Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        fo = st.number_input("MDVP:Fo(Hz)")
        fhi = st.number_input("MDVP:Fhi(Hz)")
        flo = st.number_input("MDVP:Flo(Hz)")
        jitter_percent = st.number_input("MDVP:Jitter(%)")
        jitter_abs = st.number_input("MDVP:Jitter(Abs)")
        rap = st.number_input("MDVP:RAP")
        ppq = st.number_input("MDVP:PPQ")
        ddp = st.number_input("Jitter:DDP")
        shimmer = st.number_input("MDVP:Shimmer")
        shimmer_db = st.number_input("MDVP:Shimmer(dB)")
        apq3 = st.number_input("Shimmer:APQ3")

    with col2:
        apq5 = st.number_input("Shimmer:APQ5")
        apq = st.number_input("MDVP:APQ")
        dda = st.number_input("Shimmer:DDA")
        nhr = st.number_input("NHR")
        hnr = st.number_input("HNR")
        rpde = st.number_input("RPDE")
        dfa = st.number_input("DFA")
        spread1 = st.number_input("spread1")
        spread2 = st.number_input("spread2")
        d2 = st.number_input("D2")
        ppe = st.number_input("PPE")

    if st.button("Parkinson's Test Result"):

        input_data = [
            fo, fhi, flo,
            jitter_percent, jitter_abs, rap, ppq, ddp,
            shimmer, shimmer_db, apq3, apq5,
            apq, dda, nhr, hnr,
            rpde, dfa, spread1, spread2, d2, ppe
        ]

        # Scale input
        input_data_scaled = parkinsons_scaler.transform([input_data])

        prediction = parkinsons_model.predict(input_data_scaled)
        probability = parkinsons_model.predict_proba(input_data_scaled)

        no_disease_prob = probability[0][0]
        disease_prob = probability[0][1]

        st.subheader("Prediction Result")

        if prediction[0] == 0:
            st.success("🟢 Low Risk of Parkinson's Disease")
            st.write(f"Confidence: {no_disease_prob*100:.2f}%")
        else:
            st.error("🔴 High Risk of Parkinson's Disease")
            st.write(f"Confidence: {disease_prob*100:.2f}%")

        st.write("---")
        st.write("### Detailed Probabilities")
        st.write(f"No Disease: {no_disease_prob*100:.2f}%")
        st.write(f"Parkinson's Disease: {disease_prob*100:.2f}%")

        st.progress(float(disease_prob))


if selected == "Breast Cancer Prediction":

# Breast Cancer Prediction Page
    if selected == "Breast Cancer Prediction":

        st.title("🎀 Breast Cancer Prediction")

        st.write("Enter the tumor measurement values below:")

        col1, col2 = st.columns(2)

        with col1:
            mean_radius = st.number_input("Mean Radius")
            mean_texture = st.number_input("Mean Texture")
            mean_perimeter = st.number_input("Mean Perimeter")
            mean_area = st.number_input("Mean Area")
            mean_smoothness = st.number_input("Mean Smoothness")
            mean_compactness = st.number_input("Mean Compactness")
            mean_concavity = st.number_input("Mean Concavity")
            mean_concave_points = st.number_input("Mean Concave Points")
            mean_symmetry = st.number_input("Mean Symmetry")
            mean_fractal_dimension = st.number_input("Mean Fractal Dimension")
            radius_error = st.number_input("Radius Error")
            texture_error = st.number_input("Texture Error")
            perimeter_error = st.number_input("Perimeter Error")
            area_error = st.number_input("Area Error")
            smoothness_error = st.number_input("Smoothness Error")

        with col2:
            compactness_error = st.number_input("Compactness Error")
            concavity_error = st.number_input("Concavity Error")
            concave_points_error = st.number_input("Concave Points Error")
            symmetry_error = st.number_input("Symmetry Error")
            fractal_dimension_error = st.number_input("Fractal Dimension Error")
            worst_radius = st.number_input("Worst Radius")
            worst_texture = st.number_input("Worst Texture")
            worst_perimeter = st.number_input("Worst Perimeter")
            worst_area = st.number_input("Worst Area")
            worst_smoothness = st.number_input("Worst Smoothness")
            worst_compactness = st.number_input("Worst Compactness")
            worst_concavity = st.number_input("Worst Concavity")
            worst_concave_points = st.number_input("Worst Concave Points")
            worst_symmetry = st.number_input("Worst Symmetry")
            worst_fractal_dimension = st.number_input("Worst Fractal Dimension")

        if st.button("Breast Cancer Test Result"):

            input_data = [
                mean_radius, mean_texture, mean_perimeter, mean_area,
                mean_smoothness, mean_compactness, mean_concavity,
                mean_concave_points, mean_symmetry, mean_fractal_dimension,
                radius_error, texture_error, perimeter_error, area_error,
                smoothness_error, compactness_error, concavity_error,
                concave_points_error, symmetry_error, fractal_dimension_error,
                worst_radius, worst_texture, worst_perimeter, worst_area,
                worst_smoothness, worst_compactness, worst_concavity,
                worst_concave_points, worst_symmetry, worst_fractal_dimension
            ]

            # Scale input
            input_scaled = breast_cancer_scaler.transform([input_data])

            prediction = breast_cancer_model.predict(input_scaled)
            probability = breast_cancer_model.predict_proba(input_scaled)

            malignant_prob = probability[0][0]   # 0 = Malignant
            benign_prob = probability[0][1]      # 1 = Benign

            st.subheader("Prediction Result")

            if prediction[0] == 0:
                st.error("🔴 Malignant Tumor Detected")
                st.write(f"Confidence: {malignant_prob*100:.2f}%")
            else:
                st.success("🟢 Benign Tumor")
                st.write(f"Confidence: {benign_prob*100:.2f}%")

            st.write("---")
            st.write("### Detailed Probabilities")
            st.write(f"Malignant: {malignant_prob*100:.2f}%")
            st.write(f"Benign: {benign_prob*100:.2f}%")

            st.progress(float(malignant_prob))

import numpy as np
import pickle
import streamlit as st

# Load model and scaler (use relative paths if possible)
loaded_model = pickle.load(open(
    "C:/Users/Aadarsh Chetry/Documents/6thSemProject/V2/saved_models/diabetes_model.sav", "rb"
))
scaler = pickle.load(open(
    "C:/Users/Aadarsh Chetry/Documents/6thSemProject/V2/saved_models/scaler.sav", "rb"
))


def diabetes_prediction(input_data):
    input_array = np.array(input_data, dtype=float).reshape(1, -1)
    std_data = scaler.transform(input_array)
    prediction = loaded_model.predict(std_data)

    if prediction[0] == 0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"


def main():
    st.title("Diabetes Prediction Web App")

    with st.form("diabetes_form"):
        Pregnancies = st.text_input("Number of Pregnancies", placeholder="e.g. 0")
        Glucose = st.text_input("Glucose Level (mg/dL)", placeholder="e.g. 120")
        BloodPressure = st.text_input("Blood Pressure (mmHg)", placeholder="e.g. 80")
        SkinThickness = st.text_input("Skin Thickness (mm)", placeholder="e.g. 20")
        Insulin = st.text_input("Insulin Level (µIU/mL)", placeholder="e.g. 85")
        BMI = st.text_input("Body Mass Index (BMI)", placeholder="e.g. 24.5")
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function", placeholder="e.g. 0.45")
        Age = st.text_input("Age (years)", placeholder="e.g. 30")

        submit = st.form_submit_button("Check Diabetes")

    if submit:
        fields = [
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]

        if any(field.strip() == "" for field in fields):
            st.warning("⚠️ Please fill in all fields before checking")
        else:
            try:
                input_data = [
                    int(Pregnancies),
                    float(Glucose),
                    float(BloodPressure),
                    float(SkinThickness),
                    float(Insulin),
                    float(BMI),
                    float(DiabetesPedigreeFunction),
                    int(Age)
                ]

                result = diabetes_prediction(input_data)
                st.success(result)

            except ValueError:
                st.error("❌ Please enter valid numeric values only")


if __name__ == "__main__":
    main()

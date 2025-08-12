import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from fpdf import FPDF
import re
import os


# loading the saved model

diabetes_model = pickle.load(open('trained_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_model_and_scaler.sav', 'rb'))

# sidebar for navigate

with st.sidebar:
    selected = option_menu("VitaRisk AI - Data-Driven Health Forecasting",["AI Risk Estimator for Diabetes", "Neural Risk Estimation for Heart Disease"], icons= ['activity', 'heart'], default_index = 0)

# Diabetes Prediction
if (selected == "AI Risk Estimator for Diabetes"):
    # page title
   # st.title("Diabetes Prediction System")
    import numpy as np
    import pickle
    import streamlit as st

    # Load the model and scaler
    loaded_model, loaded_scaler = pickle.load(
        open(r'D:\PythonProject\Diabetes Predicition System\trained_model.sav', 'rb'))


    def diabetes_prediction(input_data):
        try:
            input_data = [float(x) for x in input_data]
        except ValueError:
            return 'Please enter valid numerical values.', None

        input_np = np.asarray(input_data).reshape(1, -1)
        input_scaled = loaded_scaler.transform(input_np)

        prediction = loaded_model.predict(input_scaled)
        probability = loaded_model.predict_proba(input_scaled)[0][1]  # Prob of diabetic (class 1)

        if prediction[0] == 1:
            # Convert probability to level between 1–10
            diabetes_level = int(round(probability * 10))
            diabetes_level = max(1, min(diabetes_level, 10))
            return f'The Person Is Diabetic (Level: {diabetes_level}/10)', diabetes_level
        else:
            return 'The Person Is Not Diabetic', 0


    def main():
        st.set_page_config()
        st.title("AI Risk Estimator for Diabetes - (AIRED) ")

        # User inputs
        Pregnancies = st.text_input('Number Of Pregnancies')
        Glucose = st.text_input('Glucose Level')
        BloodPressure = st.text_input('Blood Pressure Value')
        SkinThickness = st.text_input('Skin Thickness Value')
        Insulin = st.text_input('Insulin Level')
        BMI = st.text_input('BMI Value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
        Age = st.text_input('Age')

        diagnosis = ''
        level = None

        if st.button('🧪 Get Diabetes Test Result'):
            input_list = [
                Pregnancies, Glucose, BloodPressure, SkinThickness,
                Insulin, BMI, DiabetesPedigreeFunction, Age
            ]
            diagnosis, level = diabetes_prediction(input_list)
            st.success(diagnosis)

            if level and level > 0:
                st.write("### Diabetes Risk Level")
                st.progress(level / 10)
                if level >= 8:
                    st.warning("⚠️ High risk — please consult a doctor.")
                elif level >= 5:
                    st.info("🟡 Moderate risk — monitor health closely.")
                else:
                    st.success("🟢 Mild risk — maintain healthy habits.")


    if __name__ == '__main__':
        main()

if (selected == "Neural Risk Estimation for Heart Disease") :
    st.title("Neural Risk Estimation for Heart Disease - (NREHD)")
    with open('heart_model_and_scaler.sav', 'rb') as f:
        model, scaler, feature_columns = pickle.load(f)
    st.write("Fill in your health details to estimate your heart risk level.")

    # Input Fields
    age = st.number_input("Your Age", 1, 120, 40)
    sex = st.radio("Gender", ["Male", "Female"])
    cp = st.selectbox("Do you feel chest pain during activity?", [
        "Type 2: Mild pain",
        "Type 3: Discomfort",
        "Type 4: No pain"
    ])
    trestbps = st.number_input("Usual Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol Level (mg/dL)", 100, 600, 190)
    fbs = st.radio("Is your fasting blood sugar over 120 mg/dL?", ["No", "Yes"])
    restecg = st.selectbox("Have you had any ECG issues?", ["No issues", "Mild issues", "Severe issues"])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.radio("Do you feel chest pain during exercise?", ["No", "Yes"])
    oldpeak = st.number_input("Heart Stress Level After Exercise (ST depression)", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope of heart rate during exercise", ["Increasing", "Flat", "Decreasing"])
    ca = st.selectbox("Number of blocked vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia type", ["Normal", "Fixed Defect", "Reversible Defect"])

    # Encoding
    sex_val = 1 if sex == "Male" else 0
    cp_val = {"Type 2: Mild pain": 1, "Type 3: Discomfort": 2, "Type 4: No pain": 3}[cp]
    fbs_val = 1 if fbs == "Yes" else 0
    restecg_val = {"No issues": 0, "Mild issues": 1, "Severe issues": 2}[restecg]
    exang_val = 1 if exang == "Yes" else 0
    slope_val = {"Increasing": 0, "Flat": 1, "Decreasing": 2}[slope]
    thal_val = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}[thal]

    # Final input array
    input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val,
                            restecg_val, thalach, exang_val, oldpeak, slope_val,
                            ca, thal_val]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    if st.button("🩺 Predict"):
        with st.spinner("Analyzing your heart health..."):
            prediction = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]
            risk_score = int(prob * 10)

            if prediction == 1:
                st.error(f"🚨 High Risk of Heart Disease Detected\nRisk Score: **{risk_score}/10**")
                health_tip = "Exercise regularly, reduce fatty foods, avoid smoking, and consult a cardiologist."
            else:
                st.success(f"✅ Low Risk of Heart Disease\nRisk Score: **{risk_score}/10**")
                health_tip = "Keep up the good lifestyle! Stay active and eat heart-healthy food."

            st.markdown(f"**Health Tip:** {health_tip}")

            # Chart
            fig, ax = plt.subplots()
            ax.bar(['Heart Risk'], [prob], width=0.4)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Risk Probability')
            st.pyplot(fig)

            # PDF Report
            clean_tip = re.sub(r'[^\x00-\x7F]+', '', health_tip)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, "Heart Disease Risk Report", ln=True)
            pdf.cell(0, 10, f"Risk Score: {risk_score}/10", ln=True)
            pdf.multi_cell(0, 10, f"Health Tip: {clean_tip}")
            pdf_path = "Heart_Risk_Report.pdf"
            pdf.output(pdf_path)

            # Download button
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Download Report", f, file_name="Heart_Risk_Report.pdf")

            os.remove(pdf_path)

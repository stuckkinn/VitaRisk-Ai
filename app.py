import pickle
import streamlit as st
import numpy as np
import time
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# Load models
diabetes_model = pickle.load(open('trained_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_model_and_scaler.sav', 'rb'))

# Sidebar navigation
st.set_page_config(page_title="VitaRisk-AI", page_icon="🫀", layout="centered")
with st.sidebar:
    selected = option_menu(
        "VitaRisk AI - Data-Driven Health Forecasting",
        ["AI Risk Estimator for Diabetes", "Neural Risk Estimation for Heart Disease"],
        icons=['activity', 'heart'],
        default_index=0
    )

# Countdown & progress helper
def countdown_timer(seconds=60):
    progress_bar = st.progress(0)
    status_text = st.empty()

    messages = [
        "🔍 Analyzing your health data...",
        "📊 Processing your inputs with AI...",
        "🧠 Crunching numbers and patterns...",
        "⚙️ Running advanced health risk models...",
        "⏳ Almost there — preparing your results..."
    ]

    for i in range(seconds, 0, -1):
        msg = messages[(seconds - i) % len(messages)]
        status_text.text(f"{msg} {i} seconds remaining")
        progress_bar.progress((seconds - i + 1) / seconds)
        time.sleep(1)

    status_text.text("✅ Analysis complete! Generating your report...")
    progress_bar.empty()


# ---------------- Diabetes Prediction ----------------
if selected == "AI Risk Estimator for Diabetes":
    loaded_model, loaded_scaler = pickle.load(open('trained_model.sav', 'rb'))

    def diabetes_prediction(input_data):
        try:
            input_data = [float(x) for x in input_data]
        except ValueError:
            return 'Please enter valid numerical values.', None

        input_np = np.asarray(input_data).reshape(1, -1)
        input_scaled = loaded_scaler.transform(input_np)
        prediction = loaded_model.predict(input_scaled)
        probability = loaded_model.predict_proba(input_scaled)[0][1]

        if prediction[0] == 1:
            diabetes_level = int(round(probability * 10))
            diabetes_level = max(1, min(diabetes_level, 10))
            return f'The Person Is Diabetic (Level: {diabetes_level}/10)', diabetes_level
        else:
            return 'The Person Is Not Diabetic', 0

    st.title("AI Risk Estimator for Diabetes - (AIRED)")

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
        countdown_timer(60)
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

# ---------------- Heart Disease Prediction ----------------
if selected == "Neural Risk Estimation for Heart Disease":
    st.title("Neural Risk Estimation for Heart Disease - (NREHD)")

    with open('heart_model_and_scaler.sav', 'rb') as f:
        model, scaler, feature_columns = pickle.load(f)

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

    sex_val = 1 if sex == "Male" else 0
    cp_val = {"Type 2: Mild pain": 1, "Type 3: Discomfort": 2, "Type 4: No pain": 3}[cp]
    fbs_val = 1 if fbs == "Yes" else 0
    restecg_val = {"No issues": 0, "Mild issues": 1, "Severe issues": 2}[restecg]
    exang_val = 1 if exang == "Yes" else 0
    slope_val = {"Increasing": 0, "Flat": 1, "Decreasing": 2}[slope]
    thal_val = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}[thal]

    input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val,
                            restecg_val, thalach, exang_val, oldpeak, slope_val,
                            ca, thal_val]])
    input_scaled = scaler.transform(input_data)

    if st.button("🩺 Predict"):
        countdown_timer(30)

        # Custom low-risk condition
        if cp == "Type 4: No pain" and trestbps < 150 and restecg == "No issues":
            prediction = 0
            prob = 0.2
            risk_score = 2
        else:
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

        fig, ax = plt.subplots()
        ax.bar(['Heart Risk'], [prob], width=0.4, color="red" if prediction else "green")
        ax.set_ylim(0, 1)
        ax.set_ylabel('Risk Probability')
        st.pyplot(fig)




st.subheader("💬 Feedback")
st.write("We value your feedback! Please fill in the form below to help us improve.")

# Feedback Form Inputs
name = st.text_input("Your Name")
review = st.selectbox("Model Review", ["Excellent", "Good", "Average", "Poor"])
feedback = st.text_area("Your Feedback")

if st.button("Submit Feedback"):
    if name.strip() and feedback.strip():
        try:
            # Email Setup
            sender_email = "stuckkinn01@gmail.com"  # Replace with your Gmail
            receiver_email = "shekharshashank310@gmail.com"  # Where feedback is sent
            app_password = "nxva fyme rqmf jyxg"  # From Gmail App Passwords

            subject = "📩 New Feedback for VitaRisk AI"
            body = f"""
            Name: {name}
            Model Review: {review}
            Feedback:
            {feedback}
            """

            # Create Email Message
            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = receiver_email
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            # Send Email via Gmail SMTP
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, app_password)
                server.sendmail(sender_email, receiver_email, msg.as_string())

            st.success("✅ Thank you for your feedback! It has been sent successfully.")
        except smtplib.SMTPAuthenticationError:
            st.error("❌ Authentication failed. Please check your email and App Password.")
        except Exception as e:
            st.error(f"❌ Failed to send feedback: {e}")
    else:
        st.warning("⚠️ Please enter your name and feedback before submitting.")
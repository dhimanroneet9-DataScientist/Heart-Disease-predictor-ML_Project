import joblib
import streamlit as st
from preprocess import build_features

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    return model

model = load_artifacts()

st.title("❤️ Heart Disease Risk Predictor")
st.caption("A logistic regression model trained on the Heart Failure Prediction dataset. "
           "Educational project only, not medical advice.")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 54)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox(
        "Chest pain type",
        ["ASY", "ATA", "NAP", "TA"],
        help="ASY = asymptomatic, ATA = atypical angina, NAP = non-anginal pain, TA = typical angina",
    )
    resting_bp = st.number_input("Resting blood pressure (mm Hg)", 60, 250, 130)
    cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 700, 240)
    fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dl", ["No", "Yes"])
with col2:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max heart rate achieved", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-induced angina", ["No", "Yes"])
    oldpeak = st.number_input("Oldpeak (ST depression)", -3.0, 7.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST slope", ["Up", "Flat", "Down"])

if st.button("Predict", type="primary"):
    X = build_features(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                       resting_ecg, max_hr, exercise_angina, oldpeak, st_slope)
    prob = model.predict_proba(X)[0, 1]

    st.subheader(f"Estimated risk: {prob:.1%}")
    st.progress(float(prob))
    if prob >= 0.5:
        st.error("The model predicts a HIGH likelihood of heart disease.")
    else:
        st.success("The model predicts a LOW likelihood of heart disease.")
    st.caption("Note: the model was trained on whole-number values, so Oldpeak is truncated to a whole number.")

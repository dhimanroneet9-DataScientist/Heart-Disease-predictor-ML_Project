import joblib
import streamlit as st
from preprocess import build_features

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 820px;}
    .hero {
        background: linear-gradient(135deg, #b91c3c 0%, #e11d48 45%, #fb7185 100%);
        padding: 2.2rem 1.8rem; border-radius: 20px; color: #fff; text-align: center;
        box-shadow: 0 10px 30px rgba(225, 29, 72, 0.25); margin-bottom: 1.6rem;
    }
    .hero h1 {color: #fff; font-size: 2.1rem; margin: 0 0 .3rem 0; line-height: 1.25;}
    .hero .by {font-size: 1.15rem; font-weight: 600; opacity: .95; margin-bottom: .6rem;}
    .hero p {margin: 0; opacity: .9; font-size: .98rem;}
    .section-title {font-weight: 700; font-size: 1.05rem; margin-bottom: .2rem;}
    div.stButton > button {
        width: 100%; border: none; border-radius: 12px; padding: .8rem 1rem;
        font-size: 1.1rem; font-weight: 700; color: #fff;
        background: linear-gradient(135deg, #b91c3c, #e11d48);
    }
    div.stButton > button:hover {filter: brightness(1.08); color: #fff;}
    .result {border-radius: 18px; padding: 1.6rem; text-align: center; margin-top: 1rem;}
    .result .label {font-size: .9rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 700;}
    .result .pct {font-size: 3.2rem; font-weight: 800; line-height: 1.1; margin: .2rem 0;}
    .result .msg {font-size: 1rem;}
    .bar {height: 14px; border-radius: 999px; background: rgba(0,0,0,.12); margin: .9rem auto 0; overflow: hidden;}
    .bar > div {height: 100%; border-radius: 999px;}
    .low  {background: #dcfce7; color: #14532d;}
    .mid  {background: #fef3c7; color: #78350f;}
    .high {background: #fee2e2; color: #7f1d1d;}
    .footer {text-align: center; opacity: .75; font-size: .85rem; margin-top: 2.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>❤️ Heart Disease Risk Predictor</h1>
        <div class="by">by Roneet Dhiman</div>
        <p>Enter a patient's clinical details and a machine learning model will estimate
        the likelihood of heart disease.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Inputs ----------
with st.container(border=True):
    st.markdown('<div class="section-title">🧍 Patient profile</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    age = c1.slider("Age", 18, 100, 54)
    sex = c2.selectbox("Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")
    fasting_bs = c3.selectbox("Fasting blood sugar > 120 mg/dl", ["No", "Yes"])

with st.container(border=True):
    st.markdown('<div class="section-title">🩺 Clinical measurements</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    resting_bp = c1.number_input("Resting blood pressure (mm Hg)", 60, 250, 130)
    cholesterol = c2.number_input("Cholesterol (mg/dl)", 0, 700, 240)
    chest_pain = c1.selectbox(
        "Chest pain type", ["ASY", "ATA", "NAP", "TA"],
        format_func=lambda x: {"ASY": "ASY – Asymptomatic", "ATA": "ATA – Atypical angina",
                               "NAP": "NAP – Non-anginal pain", "TA": "TA – Typical angina"}[x],
    )
    resting_ecg = c2.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

with st.container(border=True):
    st.markdown('<div class="section-title">🏃 Exercise test results</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    max_hr = c1.slider("Max heart rate achieved", 60, 220, 150)
    oldpeak = c2.number_input("Oldpeak (ST depression)", -3.0, 7.0, 1.0, step=0.1,
                              help="The model was trained on whole numbers, so decimals are truncated.")
    exercise_angina = c1.selectbox("Exercise-induced angina", ["No", "Yes"])
    st_slope = c2.selectbox("ST slope", ["Up", "Flat", "Down"])

# ---------- Prediction ----------
if st.button("🔍 Predict my risk"):
    X = build_features(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                       resting_ecg, max_hr, exercise_angina, oldpeak, st_slope)
    prob = float(model.predict_proba(X)[0, 1])

    if prob < 0.30:
        cls, label, color, msg = "low", "Low risk", "#16a34a", "The model sees a low likelihood of heart disease."
    elif prob < 0.60:
        cls, label, color, msg = "mid", "Moderate risk", "#d97706", "The model sees a moderate likelihood of heart disease."
    else:
        cls, label, color, msg = "high", "High risk", "#dc2626", "The model sees a high likelihood of heart disease."

    st.markdown(
        f"""
        <div class="result {cls}">
            <div class="label">{label}</div>
            <div class="pct">{prob:.1%}</div>
            <div class="msg">{msg}</div>
            <div class="bar"><div style="width:{prob*100:.1f}%; background:{color};"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- About ----------
with st.expander("ℹ️ About this project"):
    st.markdown(
        """
        - **Model:** Logistic Regression, chosen as the best of five models tested
          (Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM).
        - **Data:** Heart Failure Prediction dataset (918 patients, 11 clinical features).
        - **Test accuracy:** about 87%.
        - **Built with:** Python, scikit-learn and Streamlit.
        """
    )

st.markdown(
    '<div class="footer">Educational project by Roneet Dhiman. Not a medical device and not medical advice. '
    'Please consult a doctor for any health concerns.</div>',
    unsafe_allow_html=True,
)

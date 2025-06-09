import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Streamlit App Config ---
st.set_page_config(page_title="CKD Prediction (SVM)", layout="wide")
st.title("Chronic Kidney Disease (CKD) Prediction using SVM")

st.markdown("""
Enter the patient's details to predict the likelihood of Chronic Kidney Disease.
The prediction is based on a Support Vector Machine (SVM) model trained on 10 selected features.
""")

# --- Required Files ---
required_files = [
    'preprocessor.joblib',
    'rfe_selector.joblib',
    'selected_features.joblib',
    'support_vector_machine_model.joblib'
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"Missing required files: {', '.join(missing_files)}")
    st.stop()

# --- Load Components ---
try:
    preprocessor = joblib.load('preprocessor.joblib')
    rfe_selector = joblib.load('rfe_selector.joblib')
    selected_features = joblib.load('selected_features.joblib')
    svm_model = joblib.load('support_vector_machine_model.joblib')

    if not hasattr(svm_model, 'predict_proba'):
        st.warning("⚠️ Loaded SVM model does not support probability estimates. Confidence scores will be unavailable.")

except Exception as e:
    st.error(f"❌ Failed to load required components: {e}")
    st.stop()

# --- Feature Input UI ---
st.sidebar.header("Input Patient Data")

# Numerical inputs (selected)
sg = st.sidebar.slider("Specific Gravity (sg)", 1.005, 1.025, 1.015, 0.001)
al = st.sidebar.slider("Albumin (al)", 0, 5, 1)
bgr = st.sidebar.slider("Blood Glucose Random (bgr)", 22, 490, 120)
sc = st.sidebar.slider("Serum Creatinine (sc)", 0.4, 76.0, 1.2, 0.1)
hemo = st.sidebar.slider("Hemoglobin (hemo)", 3.1, 17.8, 12.0, 0.1)
pcv = st.sidebar.slider("Packed Cell Volume (pcv)", 9, 54, 40)

# Categorical inputs (selected)
ba = st.sidebar.selectbox("Bacteria (ba)", ['notpresent', 'present'])
htn = st.sidebar.selectbox("Hypertension (htn)", ['no', 'yes'])
dm = st.sidebar.selectbox("Diabetes Mellitus (dm)", ['no', 'yes'])
appet = st.sidebar.selectbox("Appetite (appet)", ['good', 'poor'])

# --- Create Full Input for Preprocessor ---
input_data = {
    'age': 40,           # Placeholder
    'bp': 80,            # Placeholder
    'sg': sg,
    'al': al,
    'su': 0,             # Placeholder
    'bgr': bgr,
    'bu': 40.0,          # Placeholder
    'sc': sc,
    'sod': 137.0,        # Placeholder
    'pot': 4.0,          # Placeholder
    'hemo': hemo,
    'pcv': pcv,
    'wc': 7500,          # Placeholder
    'rc': 4.5,           # Placeholder
    'rbc': 'normal',     # Placeholder
    'pc': 'normal',      # Placeholder
    'pcc': 'notpresent', # Placeholder
    'ba': ba,
    'htn': htn,
    'dm': dm,
    'cad': 'no',         # Placeholder
    'appet': appet,
    'pe': 'no',          # Placeholder
    'ane': 'no'          # Placeholder
}

input_df = pd.DataFrame([input_data])

# Order input columns properly
numerical_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
input_df = input_df[numerical_cols + categorical_cols]

# --- Predict Button ---
if st.sidebar.button("Predict"):
    try:
        # Preprocess
        preprocessed = preprocessor.transform(input_df)
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        all_preprocessed_features = numerical_cols + list(cat_features)
        preprocessed_df = pd.DataFrame(preprocessed, columns=all_preprocessed_features)

        # Feature selection
        final_input = preprocessed_df[selected_features]

        # Prediction
        prediction = svm_model.predict(final_input)[0]
        result = "CKD" if prediction == 1 else "Not CKD"

        st.subheader("Prediction Result")
        st.success(f"Predicted Outcome: **{result}**")

        # Confidence
        if hasattr(svm_model, 'predict_proba'):
            confidence_score = svm_model.predict_proba(final_input)[0][prediction]
            st.info(f"Confidence: **{confidence_score:.2%}**")
        else:
            st.warning("Confidence score not available for this model.")

        

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Model: Support Vector Machine | Features: Top 10 selected via RFE | © 2025 CKD Prediction")

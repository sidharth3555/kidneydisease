import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Streamlit App
st.set_page_config(page_title="CKD Prediction App", layout="wide")

st.title("Chronic Kidney Disease (CKD) Prediction App")
st.markdown("""
    Enter the patient's details below to predict the likelihood of Chronic Kidney Disease.
    The prediction is based on multiple machine learning models trained on the CKD dataset
    using 10 selected features, along with their confidence percentages.
""")


# Load the saved components
try:
    preprocessor = joblib.load('preprocessor.joblib')
    rfe_selector = joblib.load('rfe_selector.joblib')
    selected_features = joblib.load('selected_features.joblib')
    models = {
        'K-Nearest Neighbors': joblib.load('k-nearest_neighbors_model.joblib'),
        'Logistic Regression': joblib.load('logistic_regression_model.joblib'),
        'Random Forest': joblib.load('random_forest_model.joblib'),
        'Decision Tree': joblib.load('decision_tree_model.joblib'),
        'Support Vector Machine': joblib.load('support_vector_machine_model.joblib')
    }

    # --- DIAGNOSTIC CHECK FOR SVM MODEL ---
    if 'Support Vector Machine' in models:
        svm_model = models['Support Vector Machine']
        if hasattr(svm_model, 'predict_proba'):
            st.write("✅ SVM Model loaded with predict_proba capability.")
        else:
            st.write("❌ SVM Model loaded WITHOUT predict_proba capability.")
            st.write("This means the loaded .joblib file for SVM might be an older version, or there's a compatibility issue.")
            st.write("Please ensure you've replaced the 'support_vector_machine_model.joblib' file with the latest one provided and restarted the app.")
    # --- END DIAGNOSTIC CHECK ---

except FileNotFoundError as e:
    st.error(f"Error loading model components: {e}. Make sure all .joblib files are in the same directory.")
    st.stop()

# Define the full list of numerical and categorical columns used during training
numerical_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']


# Create input fields for the 10 selected features
st.sidebar.header("Patient Input Features (Selected)")

# Numerical inputs corresponding to selected features
sg = st.sidebar.slider("Specific Gravity (sg)", 1.005, 1.025, 1.015, 0.001)
al = st.sidebar.slider("Albumin (al)", 0, 5, 1)
bgr = st.sidebar.slider("Blood Glucose Random (bgr)", 22, 490, 120)
sc = st.sidebar.slider("Serum Creatinine (sc)", 0.4, 76.0, 1.2, 0.1)
hemo = st.sidebar.slider("Hemoglobin (hemo)", 3.1, 17.8, 12.0, 0.1)
pcv = st.sidebar.slider("Packed Cell Volume (pcv)", 9, 54, 40)

# Categorical inputs corresponding to selected features
ba = st.sidebar.selectbox("Bacteria (ba)", ['notpresent', 'present'])
htn = st.sidebar.selectbox("Hypertension (htn)", ['no', 'yes'])
dm = st.sidebar.selectbox("Diabetes Mellitus (dm)", ['no', 'yes'])
appet = st.sidebar.selectbox("Appetite (appet)", ['good', 'poor'])

# We need to create a full input DataFrame, even if some original features are not selected by RFE.
# The preprocessor expects all original columns. Placeholder values are used for non-selected features.
input_data = {
    'age': 40, # Placeholder, as 'age' was not selected.
    'bp': 80,  # Placeholder
    'sg': sg,
    'al': al,
    'su': 0,   # Placeholder
    'bgr': bgr,
    'bu': 40.0, # Placeholder
    'sc': sc,
    'sod': 137.0, # Placeholder
    'pot': 4.0, # Placeholder
    'hemo': hemo,
    'pcv': pcv,
    'wc': 7500, # Placeholder
    'rc': 4.5,  # Placeholder
    'rbc': 'normal', # Placeholder
    'pc': 'normal',  # Placeholder
    'pcc': 'notpresent', # Placeholder
    'ba': ba,
    'htn': htn,
    'dm': dm,
    'cad': 'no', # Placeholder
    'appet': appet,
    'pe': 'no', # Placeholder
    'ane': 'no'  # Placeholder
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Ensure the input_df columns are in the same order as expected by the preprocessor
original_cols_order = numerical_cols + categorical_cols
input_df = input_df[original_cols_order]


if st.sidebar.button("Predict"):
    try:
        # Preprocess the input data
        preprocessed_input = preprocessor.transform(input_df)

        # Get feature names after one-hot encoding for the preprocessed input
        all_preprocessed_features = numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
        preprocessed_input_df = pd.DataFrame(preprocessed_input, columns=all_preprocessed_features)

        # Filter the preprocessed input DataFrame using the RFE selected features
        final_input_for_prediction = preprocessed_input_df[selected_features]

        st.subheader("Prediction Results")
        prediction_results_data = []

        for name, model in models.items():
            prediction = model.predict(final_input_for_prediction)
            predicted_class = "CKD" if prediction[0] == 1 else "Not CKD"

            confidence = "N/A"
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(final_input_for_prediction)[0]
                # Assuming 0 for 'notckd' and 1 for 'ckd'
                # Get probability for the predicted class
                confidence_score = proba[1] if prediction[0] == 1 else proba[0]
                confidence = f"{confidence_score:.2%}"
            else:
                confidence = "Probabilities not available"

            prediction_results_data.append({
                "Model": name,
                "Prediction": predicted_class,
                "Confidence": confidence
            })

        results_df = pd.DataFrame(prediction_results_data)
        st.table(results_df)

        st.success("Prediction complete!")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please check the input values and ensure all model files are correctly loaded.")

# Add a section for model details and performance (optional, for completeness)
st.markdown("---")
st.subheader("Model Performance Summary (from training)")
st.info("""
    The models were evaluated using K-Fold Cross-Validation on the 10 selected features.
    The reported metrics are averages across the folds.
""")



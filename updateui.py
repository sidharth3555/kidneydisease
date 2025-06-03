import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Streamlit App Configuration - MUST be the first Streamlit command
st.set_page_config(page_title="CKD Prediction App", layout="wide")

st.title("Chronic Kidney Disease (CKD) Prediction App")
st.markdown("""
    Enter the patient's details below to predict the likelihood of Chronic Kidney Disease.
    The prediction is based on multiple machine learning models trained on the CKD dataset
    using 10 selected features.
""")

# --- Load the saved components ---
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
    model_evaluation_results = joblib.load('model_evaluation_results.joblib')
    model_confusion_matrices = joblib.load('model_confusion_matrices.joblib')
    # Load cleaned data for class distribution
    df_cleaned = pd.read_csv('cleaned_data_for_app.csv')

except FileNotFoundError as e:
    st.error(f"Error loading model components or data: {e}. Make sure all .joblib files and 'cleaned_data_for_app.csv' are in the same directory and you have run 't.py' script.")
    st.stop()

# --- DIAGNOSTIC CHECK FOR SVM MODEL (Optional, but good for debugging) ---
if 'Support Vector Machine' in models:
    svm_model = models['Support Vector Machine']
    if hasattr(svm_model, 'predict_proba'):
        st.sidebar.write("✅ SVM Model loaded with predict_proba capability.")
    else:
        st.sidebar.write("❌ SVM Model loaded WITHOUT predict_proba capability.")
        st.sidebar.write("Please re-run the 't.py' script to generate the correct SVM model and ensure 'probability=True'.")
# --- END DIAGNOSTIC CHECK ---


# Define the full list of numerical and categorical columns used during training
numerical_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']


# --- Patient Input Section ---
st.sidebar.header("Patient Input Features")

# Numerical inputs
age = st.sidebar.slider("Age", 2, 90, 40)
bp = st.sidebar.slider("Blood Pressure (bp)", 50, 180, 80)
sg = st.sidebar.slider("Specific Gravity (sg)", 1.005, 1.025, 1.015, 0.001)
al = st.sidebar.slider("Albumin (al)", 0, 5, 1)
su = st.sidebar.slider("Sugar (su)", 0, 5, 0)
bgr = st.sidebar.slider("Blood Glucose Random (bgr)", 22, 490, 120)
bu = st.sidebar.slider("Blood Urea (bu)", 1.5, 391.0, 40.0, 0.5)
sc = st.sidebar.slider("Serum Creatinine (sc)", 0.4, 76.0, 1.2, 0.1)
sod = st.sidebar.slider("Sodium (sod)", 104.0, 163.0, 137.0, 0.5)
pot = st.sidebar.slider("Potassium (pot)", 2.5, 47.0, 4.0, 0.1)
hemo = st.sidebar.slider("Hemoglobin (hemo)", 3.1, 17.8, 12.0, 0.1)
pcv = st.sidebar.slider("Packed Cell Volume (pcv)", 9, 54, 40)
wc = st.sidebar.slider("White Blood Cell Count (wc)", 2200, 26400, 7500)
rc = st.sidebar.slider("Red Blood Cell Count (rc)", 2.1, 8.0, 4.5, 0.1)

# Categorical inputs
rbc = st.sidebar.selectbox("Red Blood Cells (rbc)", ['normal', 'abnormal'])
pc = st.sidebar.selectbox("Pus Cell (pc)", ['normal', 'abnormal'])
pcc = st.sidebar.selectbox("Pus Cell Clumps (pcc)", ['notpresent', 'present'])
ba = st.sidebar.selectbox("Bacteria (ba)", ['notpresent', 'present'])
htn = st.sidebar.selectbox("Hypertension (htn)", ['no', 'yes'])
dm = st.sidebar.selectbox("Diabetes Mellitus (dm)", ['no', 'yes'])
cad = st.sidebar.selectbox("Coronary Artery Disease (cad)", ['no', 'yes'])
appet = st.sidebar.selectbox("Appetite (appet)", ['good', 'poor'])
pe = st.sidebar.selectbox("Pedal Edema (pe)", ['no', 'yes'])
ane = st.sidebar.selectbox("Anemia (ane)", ['no', 'yes'])

# Create a dictionary from inputs
input_data = {
    'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su, 'bgr': bgr, 'bu': bu,
    'sc': sc, 'sod': sod, 'pot': pot, 'hemo': hemo, 'pcv': pcv, 'wc': wc, 'rc': rc,
    'rbc': rbc, 'pc': pc, 'pcc': pcc, 'ba': ba, 'htn': htn, 'dm': dm,
    'cad': cad, 'appet': appet, 'pe': pe, 'ane': ane
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Order the columns of input_df as per the original training data
original_cols_order = numerical_cols + categorical_cols
input_df = input_df[original_cols_order]


# --- Prediction Button and Results ---
if st.sidebar.button("Predict"):
    try:
        # Preprocess the input data
        preprocessed_input = preprocessor.transform(input_df)

        all_preprocessed_features = numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
        preprocessed_input_df = pd.DataFrame(preprocessed_input, columns=all_preprocessed_features)

        # Filter the preprocessed input DataFrame using the RFE selected features
        final_input_for_prediction = preprocessed_input_df[selected_features]

        st.subheader("Prediction Results for Current Patient")
        prediction_results_data = []

        for name, model in models.items():
            prediction = model.predict(final_input_for_prediction)
            predicted_class = "CKD" if prediction[0] == 1 else "Not CKD"

            confidence = "N/A"
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(final_input_for_prediction)[0]
                # Assuming 0 for 'notckd' and 1 for 'ckd'
                confidence_score = proba[1] if prediction[0] == 1 else proba[0]
                confidence = f"{confidence_score:.2%}"
            else:
                confidence = "Probabilities not available (model not trained with probability=True)"

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


# --- Model Performance Overview ---
st.markdown("---")
st.subheader("Model Performance Overview")

# Table for Accuracy, Precision, Recall
st.write("#### Accuracy, Precision, and Recall")
metrics_df = pd.DataFrame.from_dict(model_evaluation_results, orient='index')
metrics_df = metrics_df[['accuracy', 'precision', 'recall']].round(4) # Round for display
st.table(metrics_df)


# Bar chart for Sensitivity and Specificity
st.write("#### Sensitivity and Specificity")
metrics_for_chart = []
for model_name, metrics in model_evaluation_results.items():
    metrics_for_chart.append({
        'Model': model_name,
        'Metric': 'Sensitivity',
        'Score': metrics['sensitivity']
    })
    metrics_for_chart.append({
        'Model': model_name,
        'Metric': 'Specificity',
        'Score': metrics['specificity']
    })
chart_df = pd.DataFrame(metrics_for_chart)

fig_sens_spec, ax_sens_spec = plt.subplots(figsize=(10, 6))
sns.barplot(x='Model', y='Score', hue='Metric', data=chart_df, ax=ax_sens_spec, palette='viridis')
ax_sens_spec.set_title('Model Sensitivity and Specificity')
ax_sens_spec.set_ylim(0, 1)
ax_sens_spec.set_ylabel('Score')
ax_sens_spec.set_xlabel('Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig_sens_spec)


# --- Confusion Matrices ---
st.write("#### Confusion Matrices")
col_count = 2 # Number of columns for confusion matrices
cols = st.columns(col_count)

for i, (name, cm) in enumerate(model_confusion_matrices.items()):
    with cols[i % col_count]:
        st.markdown(f"**{name} Confusion Matrix**")
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Not CKD', 'Predicted CKD'],
                    yticklabels=['Actual Not CKD', 'Actual CKD'], ax=ax_cm)
        ax_cm.set_ylabel('Actual Label')
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_title(f'{name}')
        st.pyplot(fig_cm)


# --- Class Distribution ---
st.write("#### Class Distribution in Dataset")
if 'class' in df_cleaned.columns:
    class_distribution = df_cleaned['class'].value_counts()

    fig_class_dist, ax_class_dist = plt.subplots(figsize=(7, 5))
    sns.barplot(x=class_distribution.index, y=class_distribution.values, ax=ax_class_dist, palette='pastel')
    ax_class_dist.set_title('Distribution of CKD vs. Not CKD')
    ax_class_dist.set_xlabel('Class')
    ax_class_dist.set_ylabel('Number of Patients')
    plt.tight_layout()
    st.pyplot(fig_class_dist)
else:
    st.warning("Class distribution could not be displayed as 'class' column not found in cleaned data.")


# --- Instructions for Running ---
st.sidebar.markdown("---")
st.sidebar.markdown("### How to Run This App:")
st.sidebar.markdown("1. **First, run the `t.py` script** (`python t.py`) to generate/update all the necessary `.joblib` model files and the `cleaned_data_for_app.csv`.")
st.sidebar.markdown("2. Ensure all `*.joblib` files and `cleaned_data_for_app.csv` are in the **same directory** as `app.py`.")
st.sidebar.markdown("3. Open your terminal or command prompt and navigate to the directory where you saved the files.")
st.sidebar.markdown("4. Run the command: `streamlit run app.py`")
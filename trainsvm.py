import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC # Import SVC
import joblib

# Load the dataset
# Ensure 'kidney_disease (1).csv' is in the same directory as this script
df = pd.read_csv('kidney_disease (1).csv')

# Drop the `id` column
df = df.drop('id', axis=1)

# Renaming the `classification` column for consistency
df.rename(columns={'classification': 'class'}, inplace=True)

# Replace '\t?' and '?' with NaN in the DataFrame
df.replace({'\t?': pd.NA, '?': pd.NA}, inplace=True)

# Clean up inconsistent values in 'dm', 'cad', 'class' columns
df['dm'] = df['dm'].replace({'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'})
df['cad'] = df['cad'].replace({'\tno': 'no'})
df['class'] = df['class'].replace({'ckd\t': 'ckd'})

# Convert `pcv`, `wc`, `rc` to numeric, coercing errors to NaN
df['pcv'] = pd.to_numeric(df['pcv'], errors='coerce')
df['wc'] = pd.to_numeric(df['wc'], errors='coerce')
df['rc'] = pd.to_numeric(df['rc'], errors='coerce')

# Define numerical and categorical columns (including 'class' for initial imputation)
numerical_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class'] # 'class' temporarily here for imputation

# Impute missing values for numerical columns with the mean
for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

# Impute missing values for categorical columns with the mode
for col in categorical_cols:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0] if not df[col].mode().empty else np.nan
        df[col] = df[col].fillna(mode_val)

# Separate target variable 'class'
X = df.drop('class', axis=1)
y = df['class']

# Convert 'class' column to numerical representation (0 for 'notckd', 1 for 'ckd')
y = y.map({'notckd': 0, 'ckd': 1})

# Re-define numerical and categorical columns from X after all cleaning/imputation
numerical_cols_X = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols_X = X.select_dtypes(include='object').columns.tolist()

# Outlier Handling for Numerical Columns using IQR method
for col in numerical_cols_X:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    X[col] = np.where(X[col] < lower_bound, lower_bound, X[col])
    X[col] = np.where(X[col] > upper_bound, upper_bound, X[col])

# Create preprocessing pipelines for numerical and categorical features
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols_X),
        ('cat', categorical_transformer, categorical_cols_X)
    ])

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Get feature names after one-hot encoding
feature_names = numerical_cols_X + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols_X))

# Convert preprocessed data to a DataFrame for RFE
X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)

# Feature Selection using RFE - Select 10 features
print("Performing Feature Selection using RFE (selecting 10 features)...")
estimator = LogisticRegression(random_state=42, solver='liblinear') # Using Logistic Regression as the RFE estimator
rfe_selector = RFE(estimator=estimator, n_features_to_select=10, step=1)
rfe_selector.fit(X_preprocessed_df, y)

# Get selected features
selected_features = X_preprocessed_df.columns[rfe_selector.support_]
print(f"Selected Features ({len(selected_features)}): {selected_features.tolist()}")

# Filter X_preprocessed_df to include only selected features
X_selected = X_preprocessed_df[selected_features]

# --- Train and Save ONLY the Support Vector Machine Model ---
print("\nTraining and saving the Support Vector Machine model with probability=True...")
svm_model = SVC(random_state=42, probability=True)
svm_model.fit(X_selected, y)

# Save the SVM model
joblib.dump(svm_model, 'support_vector_machine_model.joblib')
print("Support Vector Machine model saved: support_vector_machine_model.joblib")

# Also save the preprocessor, RFE selector, and selected features if they haven't been saved yet
# (They were generated in the context of X_selected)
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(rfe_selector, 'rfe_selector.joblib')
joblib.dump(selected_features, 'selected_features.joblib')
print("Preprocessor, RFE selector, and selected features saved.")
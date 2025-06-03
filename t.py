import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import joblib

# Load the dataset
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

# Define numerical and categorical columns
numerical_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']

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
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

# Outlier Handling for Numerical Columns using IQR method
for col in numerical_cols:
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
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Get feature names after one-hot encoding
feature_names = numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))

# Convert preprocessed data to a DataFrame for RFE
X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)

# Feature Selection using RFE - Select 10 features
print("Performing Feature Selection using RFE (selecting 10 features)...")
estimator = LogisticRegression(random_state=42, solver='liblinear')
rfe_selector = RFE(estimator=estimator, n_features_to_select=10, step=1)
rfe_selector.fit(X_preprocessed_df, y)

# Get selected features
selected_features = X_preprocessed_df.columns[rfe_selector.support_]
print(f"Selected Features ({len(selected_features)}): {selected_features.tolist()}")

# Filter X_preprocessed_df to include only selected features
X_selected = X_preprocessed_df[selected_features]

# Define the models
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42)
}

# K-Fold Cross-Validation and Evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}
confusion_matrices = {}

for name, model in models.items():
    print(f"\nTraining and evaluating {name}...")
    pipeline = Pipeline(steps=[
        ('model', model)
    ])

    scoring = ['accuracy', 'precision', 'recall']
    cv_results = cross_validate(pipeline, X_selected, y, cv=kf, scoring=scoring, return_estimator=True)

    # Store mean scores
    results[name] = {
        'accuracy': cv_results['test_accuracy'].mean(),
        'precision': cv_results['test_precision'].mean(),
        'recall': cv_results['test_recall'].mean()
    }

    # For confusion matrix, train one model on the entire selected dataset
    model.fit(X_selected, y)
    y_pred = model.predict(X_selected)
    cm = confusion_matrix(y, y_pred)
    confusion_matrices[name] = cm

# Display results
print("\n--- Model Evaluation Results (K-Fold Cross-Validation) ---")
for name, metrics in results.items():
    print(f"Model: {name}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print("-" * 30)

print("\n--- Confusion Matrices ---")
for name, cm in confusion_matrices.items():
    print(f"Model: {name}")
    print(cm)
    print("-" * 30)

# Save the models using joblib
model_paths = {}
for name, model in models.items():
    # Retrain the model on the full selected dataset before saving
    model.fit(X_selected, y)
    path = f'{name.replace(" ", "_").lower()}_model.joblib'
    joblib.dump(model, path)
    model_paths[name] = path
    print(f"Model saved: {path}")

# Save the preprocessor, RFE selector, and selected features
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(rfe_selector, 'rfe_selector.joblib')
joblib.dump(selected_features, 'selected_features.joblib')
print("Preprocessor, RFE selector, and selected features saved.")
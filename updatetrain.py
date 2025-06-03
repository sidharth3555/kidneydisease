# Import necessary libraries
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
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the plots directory exists
if not os.path.exists('model_plots'):
    os.makedirs('model_plots')

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
    'Support Vector Machine': SVC(random_state=42, probability=True) # probability=True is essential for confidence scores
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
        'recall': cv_results['test_recall'].mean(), # This is recall (sensitivity) for positive class (1)
    }

    # For confusion matrix, train one model on the entire selected dataset
    # And compute sensitivity and specificity from it
    model.fit(X_selected, y)
    y_pred = model.predict(X_selected)
    cm = confusion_matrix(y, y_pred)
    confusion_matrices[name] = cm

    # Calculate Sensitivity (True Positive Rate) and Specificity (True Negative Rate)
    # Assuming positive class is 1, negative class is 0
    # cm[0,0] = True Negatives (TN)
    # cm[0,1] = False Positives (FP)
    # cm[1,0] = False Negatives (FN)
    # cm[1,1] = True Positives (TP)

    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    results[name]['sensitivity'] = sensitivity
    results[name]['specificity'] = specificity


# --- Display Results in Console (Table Format) ---
print("\n" + "="*50)
print("--- Model Evaluation Results (K-Fold Cross-Validation) ---")
print("="*50)

metrics_df_display = pd.DataFrame.from_dict(results, orient='index')
metrics_df_display = metrics_df_display[['accuracy', 'precision', 'recall', 'sensitivity', 'specificity']].round(4)
print(metrics_df_display.to_string())
print("\n" + "="*50)


# --- Generate and Save Plots ---

# 1. Sensitivity and Specificity Bar Chart
print("\nGenerating Sensitivity and Specificity plot...")
metrics_for_chart = []
for model_name, metrics in results.items():
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

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Score', hue='Metric', data=chart_df, palette='viridis')
plt.title('Model Sensitivity and Specificity')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join('model_plots', 'sensitivity_specificity_bar_chart.png'))
plt.close()
print("Saved: model_plots/sensitivity_specificity_bar_chart.png")


# 2. Confusion Matrices (Heatmaps)
print("\nGenerating Confusion Matrices...")
for name, cm in confusion_matrices.items():
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Not CKD', 'Predicted CKD'],
                yticklabels=['Actual Not CKD', 'Actual CKD'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title(f'{name} Confusion Matrix')
    plt.tight_layout()
    filename = f'model_plots/{name.replace(" ", "_").lower()}_confusion_matrix.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")


# 3. Class Distribution Bar Chart
print("\nGenerating Class Distribution plot...")
if 'class' in df.columns:
    class_distribution = df['class'].value_counts()
    plt.figure(figsize=(7, 5))
    sns.barplot(x=class_distribution.index, y=class_distribution.values, palette='pastel')
    plt.title('Distribution of CKD vs. Not CKD in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Patients')
    plt.tight_layout()
    plt.savefig(os.path.join('model_plots', 'class_distribution_bar_chart.png'))
    plt.close()
    print("Saved: model_plots/class_distribution_bar_chart.png")
else:
    print("Cannot generate class distribution plot: 'class' column not found in DataFrame.")


# 4. Heatmap for Selected Features
print("\nGenerating Heatmap for 10 Selected Features...")
plt.figure(figsize=(10, 8))
# Calculate the correlation matrix for the selected features
correlation_matrix = X_selected.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of 10 Selected Features')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join('model_plots', 'selected_features_correlation_heatmap.png'))
plt.close()
print("Saved: model_plots/selected_features_correlation_heatmap.png")


# --- Save the Models and Preprocessing Components ---
print("\n" + "="*50)
print("--- Saving Models and Preprocessing Components ---")
print("="*50)

# Save the models using joblib
model_paths = {}
for name, model_instance in models.items():
    # Retrain the model on the full selected dataset before saving
    # (This ensures the saved model is trained on all available data with selected features)
    model_instance.fit(X_selected, y)
    path = f'{name.replace(" ", "_").lower()}_model.joblib'
    joblib.dump(model_instance, path)
    model_paths[name] = path
    print(f"Model saved: {path}")

# Save the preprocessor, RFE selector, and selected features
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(rfe_selector, 'rfe_selector.joblib')
joblib.dump(selected_features, 'selected_features.joblib')
print("Preprocessor, RFE selector, and selected features saved.")

print("\nAll tasks completed. Check your console for metrics and 'model_plots' directory for images.")
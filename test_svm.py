import joblib
import os

model_path = 'support_vector_machine_model.joblib'

if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found in the current directory.")
else:
    try:
        svm_model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")

        if hasattr(svm_model, 'predict_proba'):
            print("✅ The loaded SVM model HAS 'predict_proba' capability.")
        else:
            print("❌ The loaded SVM model DOES NOT HAVE 'predict_proba' capability.")
            print("This confirms the .joblib file is the issue.")

        # Optional: Print a snippet of the model to verify it's an SVC
        print(f"Model type: {type(svm_model)}")

    except Exception as e:
        print(f"An error occurred while loading or inspecting the model: {e}")
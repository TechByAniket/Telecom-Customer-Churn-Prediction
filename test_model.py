from pathlib import Path
import joblib

MODEL_PATH = Path("model") / "best_churn_model.pkl"

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")
print("Model type:", type(model))
print("Number of input features:", len(model.feature_names_in_))
print("Features:")
print(model.feature_names_in_)
print("Classes:", model.classes_)
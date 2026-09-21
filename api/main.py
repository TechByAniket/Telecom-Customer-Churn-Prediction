from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_churn_model.pkl"


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting telecom customer churn using a trained machine learning model.",
    version="1.0.0"
)


# ============================================================
# INPUT SCHEMA
# ============================================================

class ChurnInput(BaseModel):

    # Numerical features
    Age: float
    Number_of_Dependents: float
    Number_of_Referrals: float
    Tenure_in_Months: float
    Avg_Monthly_Long_Distance_Charges: float
    Avg_Monthly_GB_Download: float
    Monthly_Charge: float
    Total_Charges: float
    Total_Refunds: float
    Total_Extra_Data_Charges: float
    Total_Long_Distance_Charges: float
    Total_Revenue: float
    Satisfaction_Score: float
    CLTV: float
    Zip_Code: float
    Latitude: float
    Longitude: float
    Population: float

    # Categorical features
    Gender: str
    Under_30: str
    Senior_Citizen: str
    Married: str
    Dependents: str
    Referred_a_Friend: str
    Offer: str
    Phone_Service: str
    Multiple_Lines: str
    Internet_Service: str
    Internet_Type: str
    Online_Security: str
    Online_Backup: str
    Device_Protection_Plan: str
    Premium_Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Streaming_Music: str
    Unlimited_Data: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str
    City: str
    Lat_Long: str


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_churn(customer: ChurnInput):

    # Convert API input into dictionary
    input_dict = customer.model_dump()

    # Map API field names back to the exact training column names
    column_mapping = {
        "Number_of_Dependents": "Number of Dependents",
        "Number_of_Referrals": "Number of Referrals",
        "Tenure_in_Months": "Tenure in Months",
        "Avg_Monthly_Long_Distance_Charges": "Avg Monthly Long Distance Charges",
        "Avg_Monthly_GB_Download": "Avg Monthly GB Download",
        "Monthly_Charge": "Monthly Charge",
        "Total_Charges": "Total Charges",
        "Total_Refunds": "Total Refunds",
        "Total_Extra_Data_Charges": "Total Extra Data Charges",
        "Total_Long_Distance_Charges": "Total Long Distance Charges",
        "Total_Revenue": "Total Revenue",
        "Satisfaction_Score": "Satisfaction Score",
        "Zip_Code": "Zip Code",
        "Under_30": "Under 30",
        "Senior_Citizen": "Senior Citizen",
        "Referred_a_Friend": "Referred a Friend",
        "Phone_Service": "Phone Service",
        "Multiple_Lines": "Multiple Lines",
        "Internet_Service": "Internet Service",
        "Internet_Type": "Internet Type",
        "Online_Security": "Online Security",
        "Online_Backup": "Online Backup",
        "Device_Protection_Plan": "Device Protection Plan",
        "Premium_Tech_Support": "Premium Tech Support",
        "Streaming_TV": "Streaming TV",
        "Streaming_Movies": "Streaming Movies",
        "Streaming_Music": "Streaming Music",
        "Unlimited_Data": "Unlimited Data",
        "Paperless_Billing": "Paperless Billing",
        "Payment_Method": "Payment Method",
        "Lat_Long": "Lat Long"
    }

    # Rename fields to match the model's training schema
    for api_name, model_name in column_mapping.items():
        input_dict[model_name] = input_dict.pop(api_name)

    # Create DataFrame in the same feature order used during training
    input_data = pd.DataFrame(
        [input_dict],
        columns=model.feature_names_in_
    )

    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = model.predict(input_data)[0]

    # Probability of each class
    probabilities = model.predict_proba(input_data)[0]

    probability_dict = {
        str(cls): round(float(prob), 4)
        for cls, prob in zip(model.classes_, probabilities)
    }

    # Human-readable result
    result_labels = {
        0: "No Churn",
        1: "Churn"
    }

    return {
        "prediction": int(prediction),
        "prediction_label": result_labels.get(
            int(prediction),
            "Unknown"
        ),
        "probabilities": probability_dict,
        "churn_probability": round(
            float(probability_dict.get("1", 0.0)),
            4
        )
    }


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Customer Churn Prediction API",
        "status": "running"
    }
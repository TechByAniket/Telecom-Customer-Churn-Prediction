from pathlib import Path

import joblib
import streamlit as st

from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_churn_model.pkl"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Telecom Customer Churn",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# HEADER
# ============================================================

st.title("📊 Telecom Customer Churn Prediction")
st.markdown(
    "### AI-powered customer churn risk analysis"
)

st.info(
    "Enter the customer's profile and service information "
    "to estimate their churn risk."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎯 Customer Information")

st.sidebar.subheader("👤 Customer Profile")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

under_30 = st.sidebar.selectbox(
    "Under 30",
    ["Yes", "No"]
)

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    ["Yes", "No"]
)

married = st.sidebar.selectbox(
    "Married",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

number_of_dependents = st.sidebar.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=0
)

referred_a_friend = st.sidebar.selectbox(
    "Referred a Friend",
    ["Yes", "No"]
)

number_of_referrals = st.sidebar.number_input(
    "Number of Referrals",
    min_value=0,
    max_value=20,
    value=0
)

tenure_in_months = st.sidebar.number_input(
    "Tenure in Months",
    min_value=0,
    max_value=100,
    value=12
)

# ============================================================
# SERVICES & CONTRACT
# ============================================================

st.sidebar.subheader("📡 Services & Contract")

offer = st.sidebar.selectbox(
    "Offer",
    ["No Offer", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"]
)

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

avg_monthly_long_distance_charges = st.sidebar.number_input(
    "Avg Monthly Long Distance Charges",
    min_value=0.0,
    value=0.0
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["Yes", "No"]
)

internet_type = st.sidebar.selectbox(
    "Internet Type",
    ["No Internet", "DSL", "Fiber Optic", "Cable"]
)

avg_monthly_gb_download = st.sidebar.number_input(
    "Avg Monthly GB Download",
    min_value=0.0,
    value=0.0
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection_plan = st.sidebar.selectbox(
    "Device Protection Plan",
    ["Yes", "No", "No internet service"]
)

premium_tech_support = st.sidebar.selectbox(
    "Premium Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

streaming_music = st.sidebar.selectbox(
    "Streaming Music",
    ["Yes", "No", "No internet service"]
)

unlimited_data = st.sidebar.selectbox(
    "Unlimited Data",
    ["Yes", "No", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-Month", "One Year", "Two Year"]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Bank Withdrawal",
        "Credit Card",
        "Mailed Check",
        "Electronic Check"
    ]
)

# ============================================================
# BILLING & CUSTOMER VALUE
# ============================================================

st.sidebar.subheader("💰 Billing & Customer Value")

monthly_charge = st.sidebar.number_input(
    "Monthly Charge",
    min_value=0.0,
    value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

total_refunds = st.sidebar.number_input(
    "Total Refunds",
    min_value=0.0,
    value=0.0
)

total_extra_data_charges = st.sidebar.number_input(
    "Total Extra Data Charges",
    min_value=0.0,
    value=0.0
)

total_long_distance_charges = st.sidebar.number_input(
    "Total Long Distance Charges",
    min_value=0.0,
    value=0.0
)

total_revenue = st.sidebar.number_input(
    "Total Revenue",
    min_value=0.0,
    value=1000.0
)

satisfaction_score = st.sidebar.slider(
    "Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

cltv = st.sidebar.number_input(
    "CLTV",
    min_value=0.0,
    value=5000.0
)


# ============================================================
# LOCATION
# ============================================================

st.sidebar.subheader("📍 Location")

city = st.sidebar.text_input(
    "City",
    value="Los Angeles"
)

zip_code = st.sidebar.number_input(
    "Zip Code",
    min_value=0,
    max_value=99999,
    value=90001
)

lat_long = st.sidebar.text_input(
    "Lat Long",
    value="33.99, -118.18"
)

latitude = st.sidebar.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=34.0
)

longitude = st.sidebar.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=-118.2
)

population = st.sidebar.number_input(
    "Population",
    min_value=0,
    value=10000
)

# ============================================================
# PREDICTION
# ============================================================

st.sidebar.divider()

predict_button = st.sidebar.button(
    "🔮 Predict Churn",
    type="primary",
    use_container_width=True
)


if predict_button:

    # Convert Yes/No values to the format used by the dataset
    def yes_no(value):
        return "Yes" if value == "Yes" else "No"


    input_data = {
        "Gender": gender,
        "Age": age,
        "Under 30": yes_no(under_30),
        "Senior Citizen": yes_no(senior_citizen),
        "Married": yes_no(married),
        "Dependents": yes_no(dependents),
        "Number of Dependents": number_of_dependents,
        "Referred a Friend": yes_no(referred_a_friend),
        "Number of Referrals": number_of_referrals,
        "Tenure in Months": tenure_in_months,
        "Offer": offer,
        "Phone Service": yes_no(phone_service),
        "Avg Monthly Long Distance Charges":
            avg_monthly_long_distance_charges,
        "Multiple Lines": multiple_lines,
        "Internet Service": yes_no(internet_service),
        "Internet Type": internet_type,
        "Avg Monthly GB Download": avg_monthly_gb_download,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection Plan": device_protection_plan,
        "Premium Tech Support": premium_tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Streaming Music": streaming_music,
        "Unlimited Data": unlimited_data,
        "Contract": contract,
        "Paperless Billing": yes_no(paperless_billing),
        "Payment Method": payment_method,
        "Monthly Charge": monthly_charge,
        "Total Charges": total_charges,
        "Total Refunds": total_refunds,
        "Total Extra Data Charges": total_extra_data_charges,
        "Total Long Distance Charges": total_long_distance_charges,
        "Total Revenue": total_revenue,
        "Satisfaction Score": satisfaction_score,
        "CLTV": cltv,
        "City": city,
        "Zip Code": zip_code,
        "Lat Long": lat_long,
        "Latitude": latitude,
        "Longitude": longitude,
        "Population": population,
    }

    import pandas as pd

    input_df = pd.DataFrame([input_data])

    # Ensure exact feature order expected by the model
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    churn_probability = probabilities[1]

    st.session_state["prediction"] = prediction
    st.session_state["churn_probability"] = churn_probability


# ============================================================
# MAIN DASHBOARD
# ============================================================

# ============================================================
# PREDICTION RESULT
# ============================================================

if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]
    churn_probability = st.session_state["churn_probability"]

    st.subheader("🎯 Churn Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 1:
            st.error("🔴 CHURN")
        else:
            st.success("🟢 NO CHURN")

    with col2:
        st.metric(
            "Churn Probability",
            f"{churn_probability:.1%}"
        )

    with col3:
        st.metric(
            "Retention Probability",
            f"{1 - churn_probability:.1%}"
        )

    st.divider()
    
# ============================================================
# CHURN PROBABILITY VISUALIZATION
# ============================================================

if "prediction" in st.session_state:

    churn_probability = st.session_state["churn_probability"]

    st.subheader("📈 Churn Risk")

    st.progress(
        float(churn_probability),
        text=f"Churn Risk: {churn_probability:.1%}"
    )

    if churn_probability >= 0.70:
        st.warning(
            "⚠️ High churn risk — this customer may require "
            "retention attention."
        )
    elif churn_probability >= 0.40:
        st.info(
            "🟡 Moderate churn risk — monitor this customer."
        )
    else:
        st.success(
            "🟢 Low churn risk — customer shows relatively "
            "low predicted churn probability."
        )

    st.divider()
    
# ============================================================
# SHAP EXPLANATION
# ============================================================

if "prediction" in st.session_state:

    st.subheader("🔎 Explainable AI — SHAP")

    try:
        import numpy as np
        import shap

        SHAP_BACKGROUND_PATH = (
            BASE_DIR / "data" / "shap_background.csv"
        )

        # Load real customers
        shap_background = pd.read_csv(
            SHAP_BACKGROUND_PATH
        )

        # Keep only features used by the model
        shap_background = shap_background[
            model.feature_names_in_
        ].copy()

        # Use 50 real customers for faster dashboard response
        shap_background = shap_background.sample(
            n=min(50, len(shap_background)),
            random_state=42
        )

        # ----------------------------------------------------
        # Prediction wrapper
        # ----------------------------------------------------
        def predict_for_shap(data):

            if isinstance(data, np.ndarray):
                data = pd.DataFrame(
                    data,
                    columns=model.feature_names_in_
                )

            return model.predict_proba(data)[:, 1]

        # ----------------------------------------------------
        # Kernel SHAP
        # ----------------------------------------------------
        with st.spinner(
            "Calculating SHAP explanation..."
        ):

            explainer = shap.KernelExplainer(
                predict_for_shap,
                shap_background
            )

            shap_values = explainer.shap_values(
                input_df,
                nsamples=100
            )

        # ----------------------------------------------------
        # Extract values
        # ----------------------------------------------------
        values = np.asarray(shap_values)

        if values.ndim == 3:
            values = values[0, :, -1]

        elif values.ndim == 2:
            values = values[0]

        values = values.reshape(-1)

        # ----------------------------------------------------
        # SHAP results
        # ----------------------------------------------------
        shap_result = pd.DataFrame({
            "Feature": input_df.columns,
            "SHAP Value": values
        })

        shap_result["Absolute Impact"] = (
            shap_result["SHAP Value"].abs()
        )

        shap_result = shap_result.sort_values(
            "Absolute Impact",
            ascending=False
        )

        top_features = shap_result.head(10)

        # ----------------------------------------------------
        # Chart
        # ----------------------------------------------------
        st.write(
            "### Top factors influencing this prediction"
        )

        st.bar_chart(
            top_features.set_index("Feature")[
                "SHAP Value"
            ]
        )

        # ----------------------------------------------------
        # Explanation table
        # ----------------------------------------------------
        display_df = top_features[
            ["Feature", "SHAP Value"]
        ].copy()

        display_df["Effect"] = display_df[
            "SHAP Value"
        ].apply(
            lambda x:
                "Increases churn"
                if x > 0
                else "Decreases churn"
        )

        display_df["SHAP Value"] = display_df[
            "SHAP Value"
        ].round(4)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Positive SHAP values push the prediction "
            "toward churn. Negative values push it "
            "toward no churn."
        )

    except Exception as e:

        st.error(
            "SHAP explanation could not be generated."
        )

        st.code(str(e))

st.subheader("📋 Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Age", age)

with col2:
    st.metric("Dependents", number_of_dependents)

with col3:
    st.metric("Referrals", number_of_referrals)


st.divider()

st.subheader("🔍 Model Information")

col1, col2 = st.columns(2)

with col1:
    st.write("**Model:** Trained Churn Prediction Pipeline")

with col2:
    st.write("**Features:** 43")


st.divider()

st.caption(
    "Telecom Customer Churn Prediction • "
    "Machine Learning + Explainable AI + Responsible AI"
)

# ============================================================
# RESPONSIBLE AI — FAIRNESS AUDIT
# ============================================================

st.divider()

st.subheader("⚖️ Responsible AI — Fairness Audit")

st.write(
    "The following fairness metrics were calculated during "
    "the model evaluation using Fairlearn."
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Demographic Parity Difference",
        "0.0403"
    )

with col2:
    st.metric(
        "Equalized Odds Difference",
        "0.0332"
    )

st.caption(
    "Lower values indicate smaller measured differences "
    "between the evaluated groups. These are evaluation "
    "results for the audited test data and should not be "
    "interpreted as a universal fairness guarantee."
)

st.write("### Gender-wise Model Performance")

fairness_data = pd.DataFrame({
    "Group": ["Female", "Male"],
    "Accuracy": [0.969444, 0.965167],
    "Precision": [0.973822, 0.974194],
    "Recall": [0.916256, 0.883041],
    "F1 Score": [0.944162, 0.926380],
})

st.dataframe(
    fairness_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DATA DRIFT CHECK
# ============================================================

st.divider()

st.subheader("📊 Data Drift Monitoring")

st.write(
    "This section provides a basic monitoring check comparing "
    "the current prediction input with the SHAP background data."
)

drift_features = [
    "Age",
    "Tenure in Months",
    "Monthly Charge",
    "Total Charges",
    "Satisfaction Score",
    "CLTV"
]

drift_rows = []

for feature in drift_features:
    reference_mean = shap_background[feature].mean()
    current_value = input_df[feature].iloc[0]

    drift_rows.append({
        "Feature": feature,
        "Reference Mean": round(reference_mean, 2),
        "Current Value": round(float(current_value), 2),
        "Difference": round(
            float(current_value - reference_mean), 2
        )
    })

drift_df = pd.DataFrame(drift_rows)

st.dataframe(
    drift_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "This is an indicative input-vs-reference comparison, "
    "not a formal statistical drift test."
)
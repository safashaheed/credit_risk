import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# 1. PAGE CONFIGURATION AND FILE PATHS
# =========================================================

st.set_page_config(
    page_title="CreditWise | Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_risk_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "risk_threshold.pkl")

# Risk-category boundaries used in the notebook
LOW_RISK_LIMIT = 0.20
HIGH_RISK_LIMIT = 0.50


# =========================================================
# 2. LOAD THE SAVED MODEL AND THRESHOLD
# =========================================================

@st.cache_resource
def load_artifacts(model_path, threshold_path):
    trained_model = joblib.load(model_path)
    trained_threshold = float(joblib.load(threshold_path))
    return trained_model, trained_threshold


if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Place credit_risk_model.pkl inside "
        "the 'models' folder next to app.py."
    )
    st.stop()

if not os.path.exists(THRESHOLD_PATH):
    st.error(
        "Threshold file not found. Place risk_threshold.pkl inside "
        "the 'models' folder next to app.py."
    )
    st.stop()

try:
    model, CLASSIFICATION_THRESHOLD = load_artifacts(MODEL_PATH, THRESHOLD_PATH)
    preprocessor = model.named_steps["preprocessor"]
except Exception as error:
    st.error(f"Unable to load the saved model: {error}")
    st.stop()

# Read the exact feature order and feature groups from the fitted pipeline.
feature_order = list(model.feature_names_in_)
num_features = list(
    next(item for item in preprocessor.transformers_ if item[0] == "num")[2]
)
cat_features = list(
    next(item for item in preprocessor.transformers_ if item[0] == "cat")[2]
)

num_pipeline = preprocessor.named_transformers_["num"]
cat_pipeline = preprocessor.named_transformers_["cat"]
num_imputer = num_pipeline.named_steps["imputer"]
cat_imputer = cat_pipeline.named_steps["imputer"]

# Find the fitted OneHotEncoder without depending on a particular step name.
encoder = next(
    step for step in cat_pipeline.named_steps.values()
    if hasattr(step, "categories_")
)

numeric_defaults = dict(zip(num_features, num_imputer.statistics_))
categorical_defaults = dict(zip(cat_features, cat_imputer.statistics_))
categorical_options = {
    feature: [str(value) for value in encoder.categories_[index]]
    for index, feature in enumerate(cat_features)
}

# The notebook's engineered feature names.
engineered_features = [
    "EMPLOYMENT_YEARS",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_ANNUITY_RATIO"
]

expected_engineered = set(engineered_features)
if not expected_engineered.issubset(set(feature_order)):
    st.error(
        "The loaded model does not contain the expected engineered features. "
        "Please check that this is the latest model saved from your notebook."
    )
    st.stop()

# Applicant-entered fields. AGE is entered directly; the other four
# engineered features are calculated below using the notebook's formulas.
personal_features = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "AGE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS"
]
employment_features = ["NAME_INCOME_TYPE", "DAYS_EMPLOYED"]
financial_features = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE"
]
additional_features = [
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3"
]

input_features = (
    personal_features
    + employment_features
    + financial_features
    + additional_features
)

expected_model_features = set(input_features + engineered_features)
if set(feature_order) != expected_model_features:
    st.error(
        "The input fields in this app do not match the saved model's features. "
        "Make sure the model was saved from the latest notebook."
    )
    st.write("Features expected by this app:", sorted(expected_model_features))
    st.write("Features found in the model:", feature_order)
    st.stop()


# =========================================================
# 3. CUSTOM CSS – CREDITWISE DESIGN
# =========================================================

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fc;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}
header {
    background-color: transparent;
}
h1, h2, h3 {
    color: #17365d;
}
p, label {
    color: #344563;
}
.hero {
    background: linear-gradient(120deg, #142d4e, #245b91);
    padding: 30px;
    border-radius: 15px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 32px;
    margin-bottom: 8px;
}
.hero p {
    color: #e2ebf7;
    font-size: 16px;
    margin-bottom: 0;
}
.result-card {
    background: white;
    padding: 23px;
    border-radius: 13px;
    border: 1px solid #e1e8f0;
    box-shadow: 0 3px 12px rgba(20, 45, 78, 0.06);
    min-height: 135px;
}
.result-label {
    color: #65758b;
    font-size: 14px;
    font-weight: 600;
}
.result-value {
    color: #17365d;
    font-size: 27px;
    font-weight: 700;
    margin-top: 12px;
}
.risk-low {
    color: #16834a;
}
.risk-medium {
    color: #c18416;
}
.risk-high {
    color: #c0392b;
}
.stButton > button, .stFormSubmitButton > button {
    background-color: #173f6b;
    color: white;
    border-radius: 9px;
    border: none;
    min-height: 48px;
    font-weight: 600;
    width: 100%;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background-color: #245b91;
    color: white;
    border: none;
}
div[data-testid="stExpander"] {
    background-color: white;
    border-radius: 10px;
    border: 1px solid #e1e8f0;
}
div[data-testid="stForm"] {
    background-color: white;
    padding: 22px;
    border-radius: 13px;
    border: 1px solid #e1e8f0;
}
div[data-testid="stMetric"] {
    background-color: white;
    padding: 16px;
    border-radius: 10px;
}
.footer {
    text-align: center;
    color: #7a8798;
    font-size: 13px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🏦 CreditWise")
    st.caption("Machine Learning Credit Risk Assessment")
    st.divider()

    st.markdown("### About the Project")
    st.write(
        "An academic machine-learning application that estimates "
        "the likelihood of payment difficulties using applicant information."
    )

    st.divider()
    st.markdown("### Risk Categories")
    st.success("Low Risk: Below 20%")
    st.warning("Medium Risk: 20% to below 50%")
    st.error("High Risk: 50% or above")

    st.divider()
    st.caption(f"Binary prediction threshold: {CLASSIFICATION_THRESHOLD:.0%}")
    st.caption("Academic Machine Learning Project")


# =========================================================
# 5. HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🏦 Credit Risk Assessment Dashboard</h1>
    <p>
        Assess loan applicants using a trained Gradient Boosting
        machine-learning model.
    </p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Enter the applicant's information below to estimate payment-difficulty "
    "probability and assign a risk category."
)


# =========================================================
# 6. INPUT HELPER FUNCTIONS
# =========================================================

input_data = {}


def get_numeric_default(feature):
    """Use the training-set median stored in the fitted imputer."""
    value = numeric_defaults.get(feature, 0.0)
    if pd.isna(value):
        return 0.0
    return float(value)


def add_features(features):
    """Display the requested fields in two columns."""
    col1, col2 = st.columns(2)

    for index, feature in enumerate(features):
        current_col = col1 if index % 2 == 0 else col2
        label = feature.replace("_", " ").title()

        with current_col:
            if feature in cat_features:
                options = categorical_options[feature]
                default_value = str(categorical_defaults[feature])
                selected_index = (
                    options.index(default_value)
                    if default_value in options else 0
                )
                input_data[feature] = st.selectbox(
                    label,
                    options=options,
                    index=selected_index,
                    key=f"input_{feature}"
                )

            elif feature in num_features:
                default_value = get_numeric_default(feature)

                if feature in ["CNT_CHILDREN", "CNT_FAM_MEMBERS"]:
                    minimum = 0 if feature == "CNT_CHILDREN" else 1
                    input_data[feature] = st.number_input(
                        label,
                        min_value=minimum,
                        max_value=30,
                        value=int(np.clip(round(default_value), minimum, 30)),
                        step=1,
                        key=f"input_{feature}"
                    )

                elif feature == "AGE":
                    input_data[feature] = st.number_input(
                        label,
                        min_value=18.0,
                        max_value=100.0,
                        value=float(np.clip(default_value, 18.0, 100.0)),
                        step=1.0,
                        key=f"input_{feature}"
                    )

                elif feature in ["EXT_SOURCE_2", "EXT_SOURCE_3"]:
                    input_data[feature] = st.number_input(
                        label,
                        min_value=0.0,
                        max_value=1.0,
                        value=float(np.clip(default_value, 0.0, 1.0)),
                        step=0.01,
                        format="%.4f",
                        help="Enter a value between 0 and 1.",
                        key=f"input_{feature}"
                    )

                elif feature == "DAYS_EMPLOYED":
                    input_data[feature] = st.number_input(
                        label,
                        min_value=-200000,
                        max_value=365243,
                        value=int(np.clip(round(default_value), -200000, 365243)),
                        step=1,
                        help=(
                            "In this dataset, employment duration is represented "
                            "as a negative number of days. The special value "
                            "365243 is converted to missing, as in model training."
                        ),
                        key=f"input_{feature}"
                    )

                else:
                    # Positive amounts are required for the ratio calculations.
                    minimum = 1.0 if feature in [
                        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"
                    ] else 0.0
                    safe_default = max(minimum, default_value)
                    input_data[feature] = st.number_input(
                        label,
                        min_value=minimum,
                        value=float(safe_default),
                        step=1000.0,
                        format="%.2f",
                        key=f"input_{feature}"
                    )


# =========================================================
# 7. APPLICANT FORM
# =========================================================

st.markdown("## 👤 Applicant Information")
st.caption("Enter the applicant's details in the sections below.")

with st.form("applicant_form"):
    st.markdown("### 1. Personal Information")
    add_features(personal_features)

    st.divider()
    st.markdown("### 2. Employment Information")
    add_features(employment_features)

    st.divider()
    st.markdown("### 3. Financial Information")
    add_features(financial_features)

    st.divider()
    with st.expander("➕ Additional Applicant Information"):
        st.caption("Enter the remaining details used by the model.")
        add_features(additional_features)

    st.write("")
    submitted = st.form_submit_button(
        "🔍 Assess Credit Risk",
        use_container_width=True
    )


# =========================================================
# 8. FEATURE ENGINEERING AND PREDICTION
# =========================================================

if submitted:
    try:
        income = float(input_data["AMT_INCOME_TOTAL"])
        credit = float(input_data["AMT_CREDIT"])
        annuity = float(input_data["AMT_ANNUITY"])
        days_employed = int(input_data["DAYS_EMPLOYED"])

        if income <= 0 or credit <= 0 or annuity <= 0:
            st.error(
                "Income, credit amount and annuity must be greater than zero."
            )
            st.stop()

        # Match the feature engineering performed in the notebook.
        employment_years = (
            np.nan if days_employed == 365243
            else round(abs(days_employed) / 365, 1)
        )

        applicant_values = {
            **input_data,
            "AGE": round(float(input_data["AGE"]), 1),
            "EMPLOYMENT_YEARS": employment_years,
            "CREDIT_INCOME_RATIO": round(credit / income, 2),
            "ANNUITY_INCOME_RATIO": round(annuity / income, 3),
            "CREDIT_ANNUITY_RATIO": round(credit / annuity, 2)
        }

        # Keep exactly the feature order used during model training.
        applicant = pd.DataFrame(
            [[applicant_values[feature] for feature in feature_order]],
            columns=feature_order
        )

        # The saved pipeline performs imputation, encoding, scaling,
        # feature selection and prediction.
        probability = float(model.predict_proba(applicant)[0, 1])
        predicted_class = int(probability >= CLASSIFICATION_THRESHOLD)

        if probability < LOW_RISK_LIMIT:
            risk_category = "Low Risk"
            risk_class = "risk-low"
            risk_icon = "🟢"
        elif probability < HIGH_RISK_LIMIT:
            risk_category = "Medium Risk"
            risk_class = "risk-medium"
            risk_icon = "🟠"
        else:
            risk_category = "High Risk"
            risk_class = "risk-high"
            risk_icon = "🔴"

        # This is an illustrative inverse-probability score, not an official score.
        risk_score = round(100 * (1 - probability), 2)

        # =================================================
        # 9. RESULTS DASHBOARD
        # =================================================

        st.divider()
        st.markdown("## 📊 Credit Risk Assessment Results")
        st.caption(
            "Results generated by the trained Gradient Boosting model."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Payment Difficulty Probability</div>
                <div class="result-value">{probability * 100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Illustrative Risk Score</div>
                <div class="result-value">{risk_score}/100</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Risk Classification</div>
                <div class="result-value {risk_class}">
                    {risk_icon} {risk_category}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown("### Risk Probability Indicator")
        st.progress(float(np.clip(probability, 0.0, 1.0)))
        st.caption(
            "A higher probability indicates a higher model-estimated "
            "likelihood of recorded payment difficulties."
        )

        st.markdown("### Assessment Summary")
        if risk_category == "Low Risk":
            st.success(
                "The model places this applicant in the Low Risk category "
                "based on the information entered."
            )
        elif risk_category == "Medium Risk":
            st.warning(
                "The model places this applicant in the Medium Risk category. "
                "Further review may be appropriate."
            )
        else:
            st.error(
                "The model places this applicant in the High Risk category. "
                "Further review is recommended."
            )

        if predicted_class == 1:
            st.write(
                f"**Binary model prediction at the {CLASSIFICATION_THRESHOLD:.0%} "
                "threshold:** Potential payment difficulty (Class 1)."
            )
        else:
            st.write(
                f"**Binary model prediction at the {CLASSIFICATION_THRESHOLD:.0%} "
                "threshold:** No payment difficulty predicted (Class 0)."
            )

        st.info("""
        **Important Disclaimer**

        This is an academic machine-learning decision-support application.
        It estimates the likelihood of recorded payment difficulties using
        historical data.

        The illustrative risk score is not an official credit score.
        Predictions should not be used alone to approve or reject real
        loan applications.
        """)

    except Exception as error:
        st.error(f"Unable to generate assessment: {error}")


# =========================================================
# 10. FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    CreditWise | Credit Risk Assessment & Loan Default Prediction<br>
    Academic Machine Learning Project
</div>
""", unsafe_allow_html=True)

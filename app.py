import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ------------------------------------------------
# Page Config
# ------------------------------------------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

# Fix opacity / theme issues
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        opacity: 1 !important;
        filter: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# Load CSS
# ------------------------------------------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------------------------------------
# Load and train models
# ------------------------------------------------
@st.cache_data
def train_models():
    df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

    # Handle missing values
    df['ApplicantIncome'] = df['ApplicantIncome'].fillna(df['ApplicantIncome'].median())
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

    # Encode categorical
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    features = ['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Self_Employed']
    X = df[features]
    y = df['Loan_Status']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "Linear SVM": SVC(kernel="linear", C=0.1, probability=True),
        "Polynomial SVM": SVC(kernel="poly", degree=3, C=10, probability=True),
        "RBF SVM": SVC(kernel="rbf", gamma="scale", probability=True)
    }

    for model in models.values():
        model.fit(X_scaled, y)

    return scaler, models


scaler, models = train_models()

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
with st.sidebar:
    st.markdown("## 📋 Applicant Details")

    income = st.number_input("Applicant Income", min_value=0)
    loan = st.number_input("Loan Amount", min_value=0)

    credit = st.selectbox("Credit History", ["Yes", "No"])
    employment = st.selectbox("Employment Status", ["Not Self Employed", "Self Employed"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    st.markdown("---")
    st.markdown("## ⚙ Model Selection")

    kernel = st.radio(
        "Choose SVM Kernel",
        ["Linear SVM", "Polynomial SVM", "RBF SVM"]
    )

# ------------------------------------------------
# Main Layout
# ------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    st.markdown(
        """
        <div class="title-box">
            <h1>🏦 Smart Loan Approval System</h1>
            <p>
                This system uses Support Vector Machines (SVM) to predict loan approval.
                It handles non-linear decision boundaries commonly found in financial data.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("✅ Check Loan Eligibility", use_container_width=True):

        if income == 0 or loan == 0:
            st.warning("⚠ Please enter valid income and loan amount.")
            st.stop()

        credit_val = 1 if credit == "Yes" else 0
        emp_val = 1 if employment == "Self Employed" else 0

        input_data = np.array([[income, loan, credit_val, emp_val]])
        input_scaled = scaler.transform(input_data)

        model = models[kernel]
        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled).max() * 100

        st.markdown("---")

        # -------------------------------
        # Loan Decision
        # -------------------------------
        st.markdown(
            """
            <div class="loan-section">
                <h2>📌 Loan Decision</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        if prediction == 1:
            st.success("✅ Loan Approved")
            explanation = (
                "Based on the applicant’s income level and positive credit history, "
                "the model predicts strong repayment capability."
            )
        else:
            st.error("❌ Loan Rejected")
            explanation = (
                "Based on income level and credit history, "
                "the applicant shows higher repayment risk."
            )

        st.markdown(
    f"""
    <div>
        <span class="info-badge">Kernel Used: {kernel}</span>
        <span class="info-badge">Confidence Score: {confidence:.2f}%</span>
    </div>
    """,
    unsafe_allow_html=True
)

        # -------------------------------
        # Business Explanation
        # -------------------------------
        st.markdown(
            f"""
            <div class="loan-section">
                <h2>🧠 Business Explanation</h2>
                <p>{explanation}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "<div class='footer'>SVM-based FinTech System</div>",
        unsafe_allow_html=True
    )

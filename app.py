import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CreditWise Loan System", page_icon="🏦", layout="centered")

# ---------- Load the saved files (only once) ----------
@st.cache_resource
def load_files():
    model = joblib.load("loan_model.pkl")
    scaler = joblib.load("scaler.pkl")
    ohe = joblib.load("onehot_encoder.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, ohe, columns

model, scaler, ohe, columns = load_files()

ONEHOT_COLS = ["Employment_Status", "Marital_Status", "Loan_Purpose",
               "Property_Area", "Gender", "Employer_Category"]

# ---------- Page heading ----------
st.title("🏦 CreditWise Loan System")
st.write("SecureTrust Bank – enter the applicant's details and the ML model will "
         "predict whether the loan should be **Approved** or **Rejected**.")
st.divider()

# ---------- Input form ----------
st.subheader("1. Personal details")
c1, c2 = st.columns(2)
age = c1.number_input("Age", 18, 70, 35)
gender = c2.selectbox("Gender", ["Male", "Female"])
marital = c1.selectbox("Marital status", ["Married", "Single"])
dependents = c2.number_input("Number of dependents", 0, 10, 0)
education = c1.selectbox("Education level", ["Graduate", "Not Graduate"])
property_area = c2.selectbox("Property area", ["Urban", "Semiurban", "Rural"])

st.subheader("2. Job & income")
c1, c2 = st.columns(2)
employment = c1.selectbox("Employment status", ["Salaried", "Self-employed", "Contract", "Unemployed"])
employer = c2.selectbox("Employer category", ["Private", "Government", "MNC", "Business", "Unemployed"])
app_income = c1.number_input("Applicant monthly income (₹)", 0, 1_000_000, 10000, step=500)
coapp_income = c2.number_input("Co-applicant monthly income (₹)", 0, 1_000_000, 5000, step=500)

st.subheader("3. Credit & finance")
c1, c2 = st.columns(2)
credit_score = c1.number_input("Credit score", 300, 900, 650)
existing_loans = c2.number_input("Existing running loans", 0, 20, 0)
dti = c1.slider("Debt-to-Income (DTI) ratio", 0.0, 1.0, 0.30, 0.01)
savings = c2.number_input("Savings balance (₹)", 0, 10_000_000, 10000, step=500)
collateral = c1.number_input("Collateral value (₹)", 0, 10_000_000, 25000, step=1000)

st.subheader("4. Loan requested")
c1, c2 = st.columns(2)
loan_amount = c1.number_input("Loan amount (₹)", 0, 10_000_000, 20000, step=1000)
loan_term = c2.number_input("Loan term (months)", 6, 360, 36, step=6)
loan_purpose = c1.selectbox("Loan purpose", ["Personal", "Car", "Business", "Home", "Education"])

# ---------- Predict ----------
if st.button("Check loan eligibility", type="primary", use_container_width=True):

    # 1) put the answers in a one-row table (same column names as the CSV)
    row = pd.DataFrame([{
        "Applicant_Income": app_income,
        "Coapplicant_Income": coapp_income,
        "Employment_Status": employment,
        "Age": age,
        "Marital_Status": marital,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti,
        "Savings": savings,
        "Collateral_Value": collateral,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education,
        "Gender": gender,
        "Employer_Category": employer,
    }])

    # 2) same steps as the notebook
    row["Education_Level"] = row["Education_Level"].map({"Graduate": 0, "Not Graduate": 1})

    encoded = ohe.transform(row[ONEHOT_COLS])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ONEHOT_COLS), index=row.index)
    row = pd.concat([row.drop(columns=ONEHOT_COLS), encoded_df], axis=1)

    row["DTI_Ratio_sq"] = row["DTI_Ratio"] ** 2
    row["Credit_Score_sq"] = row["Credit_Score"] ** 2

    row = row[columns]                 # same columns, same order as training
    row_scaled = scaler.transform(row)

    # 3) predict
    prediction = model.predict(row_scaled)[0]
    prob_approved = model.predict_proba(row_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.success("✅ Loan APPROVED")
    else:
        st.error("❌ Loan REJECTED")
    st.metric("Chance of approval (model confidence)", f"{prob_approved * 100:.1f}%")
    st.progress(float(prob_approved))
    st.caption("This is a model prediction to support the loan officer. Final decision needs human verification.")

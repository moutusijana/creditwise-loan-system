# creditwise-loan-system

An ML-powered loan approval prediction system for SecureTrust Bank. It predicts whether a loan application should be Approved or Rejected, to support loan officers before final human verification.

Live app: https://creditwise-loan-systemapp.streamlit.app/

## Problem
Manual loan verification is slow and inconsistent. Good customers get rejected and high-risk customers get approved. This project uses machine learning to make decisions faster and more consistent.

## Dataset
1,000 loan applicants with 19 features: income, credit score, DTI ratio, savings, collateral, loan amount, employment details, and more. Target: Loan_Approved (Yes/No).

## What I did
- Filled missing values (mean for numbers, most frequent for categories)
- Explored the data with plots and a correlation heatmap
- Encoded categorical columns and scaled the features
- Added squared features for Credit Score and DTI Ratio
- Compared 3 models

## Results
| Model | Accuracy |
|---|---|
| Logistic Regression | 87.5% |
| Naive Bayes | 86.5% |
| KNN | 75.5% |

Logistic Regression was chosen as the final model.

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit

## Run locally
pip install -r requirements.txt
python -m streamlit run app.py

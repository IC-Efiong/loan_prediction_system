import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

#load the trained model
model_path = "best_loan_model_1.pkl"
model = joblib.load(model_path)

#Define streamlit UI
st.title("Loan Prediction App")
st.write("This app predicts the likelihood of a loan being approved or rejected based on various factors.")

#User input fields
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
Credit_History = st.selectbox("Credit History", [1, 0])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

#Compute Total Income
TotalIncome = ApplicantIncome + CoapplicantIncome

#Convert categorical values to numbers (Manual Label Encoding)
category_mappings = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"No": 0, "Yes": 1},
    "Education": {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"No": 0, "Yes": 1},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3}
}

#Apply encoding
encoded_data = [
    category_mappings["Gender"][Gender],
    category_mappings["Married"][Married],
    category_mappings["Dependents"][Dependents],
    category_mappings["Education"][Education],
    category_mappings["Self_Employed"][Self_Employed],
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    category_mappings["Property_Area"][Property_Area],
    TotalIncome
]

#convert to Numpy array
input_data = np.array([encoded_data])

#Standardize numerical values.
scaler = StandardScaler()
numerical_indices = [5, 6, 7, 8, 9, 11] #Indices of numerical column
input_data[:, numerical_indices] = scaler.fit_transform(input_data[:, numerical_indices])

#Predict on user input
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)[0]
    result = "Approved" if prediction == 1 else "Not Approved"
    st.success(f"Loan Status: {result}")
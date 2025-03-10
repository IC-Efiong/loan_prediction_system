# Loan Prediction System

## 📌 Project Overview
This project aims to develop a **Loan Prediction System** using **Machine Learning**. The goal is to automate the loan approval process by analyzing applicant details and predicting loan approval based on financial and demographic factors.

## 📊 Dataset Description
- **Dataset Used:** Loan Prediction Dataset
- **Records:** 614
- **Attributes:** 13 (including categorical and numerical variables)
- **Target Variable:** `Loan_Status` (Approved or Rejected)

## 🔄 Data Processing
### 1️⃣ Data Cleaning
- Handled **missing values** using mode/median.
- Treated **outliers** using the **Interquartile Range (IQR) method**.

### 2️⃣ Feature Engineering
- Encoded categorical variables using **Label Encoding**.
- Normalized numerical variables.
- Created a new feature: **TotalIncome = ApplicantIncome + CoapplicantIncome**.

### 3️⃣ Exploratory Data Analysis (EDA)
- Visualized **income distributions, loan amounts, credit history, and property area**.
- Analyzed **correlations and loan approval trends**.

## 🤖 Machine Learning Models
The following models were trained and evaluated:
1. **Logistic Regression** (Best Performing Model ✅)
2. **Decision Tree**
3. **Random Forest**
4. **Support Vector Machine (SVM)**
5. **K-Nearest Neighbors (KNN)**
6. **Gradient Boosting**
7. **XGBoost**

## 📈 Model Evaluation
- **Best Model:** Logistic Regression (Accuracy: **85.4%**)
- **Evaluation Metrics:**
  - Accuracy, Precision, Recall, F1-score
  - **Confusion Matrix & ROC-AUC Score** (AUC = **0.82** ✅)
  - **Feature Importance Analysis**

## ⚙️ How to Run the Project
1️⃣ **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2️⃣ **Run the Jupyter Notebook:**
```bash
jupyter notebook Loan_Prediction_System.ipynb
```

3️⃣ **Test the Model with a Single Input:**
```python
import joblib
import numpy as np

# Load Model
model = joblib.load("best_loan_model.pkl")

# Define Test Input
single_input = np.array([[1, 1, 0, 1, 0, 5000, 1500, 6500, 128, 360, 1, 2]])
prediction = model.predict(single_input)
print("Loan Status:", "Approved" if prediction[0] == 1 else "Rejected")
```

## 📜 Conclusion & Next Steps
- **Credit History is the strongest predictor of loan approval.**
- Further improvements:
  - Hyperparameter tuning for better accuracy.
  - Explore additional models (e.g., LightGBM).
  - Improve handling of imbalanced data.

🔹 **Author:** [IC-Efiong]  
🔹 **Date:** [10/03/2025]


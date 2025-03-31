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

# Loan Prediction App

This is a Streamlit-based web application that predicts loan approval based on user input. The model uses categorical encoding, feature scaling, and a machine learning model to make predictions.

## Features
- User-friendly interface for entering loan application details
- Automatic feature engineering (e.g., calculating Total Income)
- Label encoding for categorical variables
- Standardization of numerical features
- Machine learning-based loan approval prediction

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/loan-prediction.git
   cd loan-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Dependencies
- Python 3.x
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Usage
1. Launch the Streamlit app.
2. Fill in the required loan application details.
3. Click the "Predict Loan Approval" button.
4. The system will display whether the loan is "Approved" or "Not Approved".

## Model Training
The machine learning model used for prediction was trained separately. It includes:
- Label encoding for categorical variables
- Standardization of numerical features
- A trained classification model (e.g., Logistic Regression, Random Forest, etc.)

## Future Improvements
- Deploy the model using a cloud-based API.
- Improve prediction accuracy with additional features.
- Enhance UI with better visualizations.


## License
This project is licensed under the MIT License.



🔹 **Author:** [IC-Efiong]  
🔹 **Date:** [10/03/2025]


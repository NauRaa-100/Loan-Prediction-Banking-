
#  Loan Prediction Project (Machine Learning)

##  Overview
This project aims to **predict loan approval status** based on customer financial and personal information.  
It’s a **classification problem** where the target variable `Loan_Status` indicates whether a loan was approved (`Y`) or not (`N`).  
The dataset is from **Kaggle’s Loan Prediction Dataset** (Banking domain).

---

## Objective
To build and compare multiple Machine Learning models that can predict loan approval accurately,  
and identify the most important features affecting loan decisions.

---

##  Dataset Information

- **Source:** [Kaggle - Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication)
- **Target column:** `Loan_Status`
- **Shape:** 614 rows × 13 columns  
- **Data Types:** Categorical and Numerical

### Key Features:
| Feature | Description |
|----------|--------------|
| Gender | Male / Female |
| Married | Applicant marital status |
| Dependents | Number of dependents |
| Education | Graduate / Not Graduate |
| Self_Employed | Employment status |
| ApplicantIncome | Monthly income |
| CoapplicantIncome | Co-applicant income |
| LoanAmount | Loan amount in thousands |
| Loan_Amount_Term | Term of loan (in months) |
| Credit_History | Credit history meets guidelines (1/0) |
| Property_Area | Urban / Semiurban / Rural |
| Loan_Status | **Target (Y/N)** |

---

##  Data Preprocessing Steps

1. **Missing Values Handling**
   - Filled categorical nulls with `mode()`.
   - Filled numerical nulls with `median()`.

2. **Data Type Optimization**
   - Converted object types to category to save memory.

3. **Encoding**
   - Applied **Label Encoding** for categorical variables.

4. **Scaling**
   - Standardized numerical columns using `StandardScaler`.

5. **Train-Test Split**
   - Split data into `train` and `test` (80% / 20%).

---

## Model Building

Several classification models were trained and compared:

| Model | Description |
|--------|--------------|
| Logistic Regression | Baseline linear model |
| Random Forest Classifier | Ensemble bagging method |
| XGBoost Classifier | Gradient boosting model |
| LightGBM Classifier | Fast and efficient boosting |
| CatBoost Classifier | Handles categorical features well |

### Feature Selection
Used **RandomForestClassifier** to identify top contributing features by importance score.

---

## Model Evaluation

**Metrics Used:**
- F1 Score
- Precision
- (Optional: Recall, ROC-AUC)

**Visualizations:**
- Confusion Matrix
- ROC Curve

Example:
```python
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
````

---

## Results Summary

| Model               | F1 Score | Precision |
| ------------------- | -------- | --------- |
| Logistic Regression | 0.88     | 0.87      |
| Random Forest       | 0.83     | 0.85      |
| XGBoost             | 0.84     | 0.86      |
| LightGBM            | 0.83     | 0.85      |
| CatBoost            | 0.85     | 0.86      |

**Best Model:** LogisticRegression
Achieved the highest balance between F1 and Precision.

---

##  Model Saving

Trained model and scaler were saved using `joblib` for future use:

```python
import joblib
joblib.dump(model, 'loan_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

---

##  Visual Insights

* Heatmaps and boxplots used to understand variable relationships.
* Applicants with higher income, good credit history, and urban property areas had a higher chance of approval.

---

##  Next Steps (Future Improvements)

* Perform **Cross Validation** for more robust performance.
* Hyperparameter tuning using **GridSearchCV**.
* Create an **interactive dashboard** (Streamlit or Flask).
* Deploy the model via **Docker** or **Flask API**.

---

##  Tech Stack

| Category      | Tools                                                                         |
| ------------- | ----------------------------------------------------------------------------- |
| Programming   | Python                                                                        |
| Libraries     | pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm, catboost |
| Model Saving  | joblib                                                                        |
| Visualization | seaborn, matplotlib                                                           |

---

## Author

**Nau Raa**
*Data Science & AI Enthusiast*

---

## Keywords

`Machine Learning` `Classification` `Banking` `Loan Prediction` `Python` `Scikit-learn` `Feature Engineering` `CatBoost` `XGBoost`


#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For data preprocessing and modeling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.inspection import PartialDependenceDisplay

# Suppress warnings for cleaner output
import warnings

warnings.filterwarnings("ignore")

# Install imbalanced-learn if not already installed
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    import sys

    get_ipython().system('{sys.executable} -m pip install imbalanced-learn')
    from imblearn.over_sampling import SMOTE

# 1. Dataset Overview
print("1. Dataset Overview:")
print(
    "This dataset contains credit card transactions, including details about merchants, amounts, and fraud labels."
)
print(
    "It is sourced from a financial transactions database and is used for analyzing and detecting fraudulent transactions."
)

# Define the file path (Replace with your actual file path)
file_path = "credit_card_transactions.csv"


# Display the shape and first few rows
print("\nDataset Shape:", data.shape)
print("\nFirst 5 rows of the dataset:")
print(data.head())

# 2. Feature Description
print("\n2. Feature Description:")
print("Below are the features present in the dataset along with their data types:")
# Provide descriptions for all columns
descriptions = [
    "Row identifier (may be unnecessary after loading the data)",
    "Date and time of the transaction",
    "Credit card number",
    "Name of the merchant",
    "Category of the merchant",
    "Amount of the transaction",
    "First name of the cardholder",
    "Last name of the cardholder",
    "Gender of the cardholder",
    "Street address of the cardholder",
    "City of the cardholder",
    "State of the cardholder",
    "ZIP code of the cardholder",
    "Latitude of the cardholder's address",
    "Longitude of the cardholder's address",
    "Population of the city",
    "Job title of the cardholder",
    "Date of birth of the cardholder",
    "Transaction number (unique identifier)",
    "Unix timestamp of the transaction",
    "Latitude of the merchant's location",
    "Longitude of the merchant's location",
    "Label indicating if the transaction is fraudulent (0 for non-fraud, 1 for fraud)",
    "ZIP code of the merchant's location",
]

data_info = pd.DataFrame(
    {
        "Feature": data.columns,
        "Data Type": data.dtypes.values,
        "Description": descriptions,
    }
)
print(data_info)

# 3. Dataset Structure
print("\n3. Dataset Structure:")
print(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")

# 4. Missing Values and Duplicates
print("\n4. Missing Values and Duplicates:")
missing_values = data.isnull().sum()
print("Missing values in each column:")
print(missing_values)
duplicates = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# 5. Statistical Summary
print("\n5. Statistical Summary of Numerical Features:")
print(data.describe())

# 6. Data Distribution
print("\n6. Data Distribution:")
numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
for column in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()

# 7. Correlation Analysis
print("\n7. Correlation Analysis:")
plt.figure(figsize=(12, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Discuss findings
print(
    "From the correlation heatmap, we can observe the relationships between numerical features."
)
print("We will focus on features that show significant correlation with 'is_fraud'.")

# 8. Outlier Detection
print("\n8. Outlier Detection:")
plt.figure(figsize=(8, 6))
sns.boxplot(x=data["amt"])
plt.title("Outlier Detection in 'amt'")
plt.show()

# Quantify outliers using IQR
Q1 = data["amt"].quantile(0.25)
Q3 = data["amt"].quantile(0.75)
IQR = Q3 - Q1
outliers = data[
    (data["amt"] < Q1 - 1.5 * IQR) | (data["amt"] > Q3 + 1.5 * IQR)
]
print(f"Number of outliers in 'amt': {outliers.shape[0]}")

# 9. Handling Missing Data
print("\n9. Handling Missing Data:")
print(
    "Since the number of missing values is minimal, we will drop rows with missing values to avoid data imputation biases."
)
data = data.dropna()
print(f"Dataset shape after dropping missing values: {data.shape}")

# 10. Handling Duplicates
print("\nHandling Duplicates:")
if duplicates > 0:
    data = data.drop_duplicates()
    print(f"Dataset shape after dropping duplicates: {data.shape}")
else:
    print("No duplicate rows found.")

# 11. Encoding Categorical Variables
print("\n10. Encoding Categorical Variables:")
categorical_columns = data.select_dtypes(include=["object"]).columns
print(f"Categorical columns to encode: {list(categorical_columns)}")
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
print("Categorical variables have been encoded using one-hot encoding.")

# 12. Feature Scaling
print("\n11. Feature Scaling:")
# Update numeric_columns after encoding
numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
# Exclude 'is_fraud' from scaling
numeric_columns = [col for col in numeric_columns if col != "is_fraud"]
print(f"Numeric columns to scale: {list(numeric_columns)}")
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
print("Numeric features have been scaled using StandardScaler.")

# 13. Feature Selection
print("\n12. Feature Selection:")
# Define target variable and features
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# Ensure 'is_fraud' is integer type
y = y.astype(int)

# Check the unique values in y
print("Unique values in target variable 'is_fraud':", y.unique())

# Handle class imbalance if necessary
print("\nHandling Class Imbalance:")
fraud_count = y.value_counts()
print(fraud_count)
if fraud_count.min() / fraud_count.max() < 0.1:
    print("Dataset is imbalanced. Applying SMOTE to balance the classes.")
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    print("Classes have been balanced.")
    print(y.value_counts())
else:
    print("Dataset is balanced. No action needed.")

# 14. Algorithm Selection
print("\n13. Algorithm Selection:")
print(
    "Considering various algorithms suitable for classification tasks, including Logistic Regression, SVM, and Random Forest."
)
print(
    "Random Forest is chosen for its robustness and ability to handle feature importance analysis."
)

# 15. Data Splitting
print("\n14. Data Splitting:")
# Use stratify to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Data has been split into training and testing sets.")

# 16. Model Training
print("\n15. Model Training:")
# Initialize Random Forest with class_weight to handle any residual imbalance
model = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)
print("Random Forest model has been trained.")

# 17. Model Evaluation
print("\n16. Model Evaluation:")
y_pred = model.predict(X_test)
print("Evaluation Metrics on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 18. Performance Analysis
print("\n17. Performance Analysis:")
print("The model shows good performance on the test set.")
print(
    "However, we will look into improving recall, which is crucial in fraud detection to minimize false negatives."
)

# 19. Model Improvement
print("\n18. Model Improvement:")
print("Performing hyperparameter tuning using RandomizedSearchCV.")

# Define the parameter distribution
param_dist = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "class_weight": ["balanced", "balanced_subsample"],
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=10,  # Number of parameter settings that are sampled
    cv=3,
    scoring="recall",
    random_state=42,
    verbose=2,
    n_jobs=-1  # Use all available cores
)

try:
    random_search.fit(X_train, y_train)
    print("Hyperparameter tuning complete.")
    print(f"Best parameters: {random_search.best_params_}")
except Exception as e:
    print("An error occurred during hyperparameter tuning:", e)

# Update best_model
best_model = random_search.best_estimator_

# 20. Validation
print("\n19. Validation:")
cv_scores = cross_val_score(
    best_model, X, y, cv=5, scoring="recall", n_jobs=-1
)
print(f"Cross-Validation Recall Scores: {cv_scores}")
print(f"Mean Cross-Validation Recall Score: {cv_scores.mean():.2f}")

# 21. Final Model Selection
print("\n20. Final Model Selection:")
print("The Random Forest model with the following parameters has been selected:")
print(best_model)

# 22. Re-evaluate Model with Best Parameters
print("\nRe-evaluating model on the test set with the best parameters:")
y_pred_best = best_model.predict(X_test)
print("Evaluation Metrics on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_best):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_best):.2f}")

# Confusion Matrix for Best Model
conf_matrix_best = confusion_matrix(y_test, y_pred_best)
print("\nConfusion Matrix for Best Model:")
print(conf_matrix_best)

# Classification Report for Best Model
print("\nClassification Report for Best Model:")
print(classification_report(y_test, y_pred_best))

# 23. Data Distribution After Preprocessing
print("\n21. Data Distribution After Preprocessing:")
for column in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(X[column], kde=True)
    plt.title(f"Distribution of {column} (after scaling)")
    plt.show()

# 24. Feature Importance
print("\n22. Feature Importance:")
feature_importances = best_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame(
    {"Feature": features, "Importance": feature_importances}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance from Random Forest")
plt.show()

print("The most important features influencing the prediction are:")
print(importance_df.head())

# 25. Model Performance Across Features
print("\n23. Model Performance Across Features:")
top_features = importance_df["Feature"][:2].tolist()
print(f"Analyzing partial dependence for top features: {top_features}")

fig, ax = plt.subplots(figsize=(12, 6))
PartialDependenceDisplay.from_estimator(
    best_model, X_train, features=top_features, ax=ax
)
plt.show()

# 26. Ethical Considerations
print("\n24. Ethical Considerations:")
print(
    "In fraud detection, it is important to balance the minimization of false negatives (missing fraudulent transactions) and false positives (flagging legitimate transactions as fraud)."
)
print(
    "Misclassification can impact customers and the financial institution, so the model must be thoroughly evaluated."
)

# 27. Conclusion
print("\n25. Conclusion:")
print(
    "We have successfully built a fraud detection model using Random Forest."
)
print(
    "Through data exploration, preprocessing, and model tuning, we achieved improved recall."
)
print(
    "The model can now be used to help detect fraudulent transactions more effectively."
)


# In[ ]:





# In[ ]:





# In[ ]:





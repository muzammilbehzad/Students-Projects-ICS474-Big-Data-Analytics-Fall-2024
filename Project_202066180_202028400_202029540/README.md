
# Credit Card Fraud Detection Project

## Overview

This project focuses on detecting fraudulent credit card transactions using a dataset of 1.3 million credit card transaction records. The dataset includes features such as transaction amounts, timestamps, merchant information, and customer demographics. We use machine learning algorithms to classify transactions as either fraudulent or legitimate. The project employs a Voting Classifier with soft voting and a variety of algorithms to enhance fraud detection performance.

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Modeling](#modeling)
4. [Evaluation](#evaluation)
5. [Feature Importance](#feature-importance)
6. [Model Performance](#model-performance)
7. [Final Model Selection](#final-model-selection)
8. [Visualizations](#visualizations)

## Dataset Overview

The dataset used for this project is sourced from [Kaggle - Credit Card Transactions Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset). It contains **1,296,675 records** with **24 features**, capturing transaction details, merchant data, customer information, and location. The data is primarily used for **fraud detection** and **customer behavior analysis**.

### Key Features:
- **Transaction Details**: Includes the transaction amount (`amt`), timestamp, merchant (`merchant`), and location (`lat`, `long`).
- **Customer Information**: Features like `age`, `gender`, and `job`.
- **Fraud Indicator**: The target variable `is_fraud` indicates whether a transaction is fraudulent.

## Data Preprocessing

1. **Feature Scaling and Encoding**: 
   - Categorical variables are encoded using `LabelEncoder` and `One-Hot Encoding`.
   - Numerical features are scaled using `StandardScaler` to normalize values.
  
2. **Feature Selection**:
   - Features selected for the model include `amt`, `gender`, `category`, `age`, `job`, `lat`, `long`, and `network`.

## Modeling

1. **Algorithm Selection**:
   - We selected four algorithms for the binary classification task: 
     - **Decision Tree**: Excellent interpretability and handles non-linear patterns.
     - **Logistic Regression**: Efficient for binary classification.
     - **Random Forest**: Ensemble method that reduces overfitting.
     - **Naive Bayes**: Useful for high-dimensional datasets, but struggles with feature independence.

2. **Data Splitting**:
   - The data was split into **80% training** and **20% testing** sets using stratified sampling to preserve the class distribution of fraud and non-fraud transactions.

3. **Class Weight Adjustment**:
   - To address the class imbalance, models were trained with **class_weight='balanced'**.

## Evaluation

Model evaluation is performed using the **classification report** to assess precision, recall, F1-score, and accuracy. The evaluation is particularly focused on the **recall** for the fraud class (1), which is critical for minimizing false negatives in fraud detection.

### Key Results:
- **Random Forest** and **Decision Tree** outperformed other models in terms of **precision**, **recall**, and **F1-score**.
- **Logistic Regression** and **Naive Bayes** struggled, especially in identifying fraudulent transactions, due to high **false positives** and **low precision**.

## Feature Importance

We analyzed the **feature importance** using the Random Forest and Decision Tree models. The most significant features impacting the fraud detection model include:
- **amt**: Transaction amount (most important).
- **category**: Type of transaction category.
- **age**: Age of the customer.
- **gender**: Gender of the customer.

## Model Performance

The performance across various features and thresholds was assessed, showing that **Voting Classifier** performed consistently well across all thresholds for the feature `amt` (transaction amount).

## Final Model Selection

After evaluating various models, **Voting Classifier** which ensembles The decision tree, random forest, and logistic regression emerged as the best option as it yielded the highest score in all the metrics.



## Requirements

To run this project, you'll need:
- Python 3.x
- scikit-learn
- pandas
- matplotlib
- numpy

You can install the required packages using the following:

```bash
pip install -r requirements.txt
```

## Conclusion

This project demonstrates the application of multiple machine learning models for fraud detection in credit card transactions. Random Forest, due to its robustness and effectiveness in handling imbalanced data, was selected as the final model for real-world deployment.

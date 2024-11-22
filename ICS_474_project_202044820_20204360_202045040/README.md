# Credit Cart Transaction Project

This project analyzes a comprehensive credit card transaction dataset to uncover patterns to predict frauds in the transactions, and evaluate the performance using machine learning techniques.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Details](#dataset-details)
3. [Tools and Libraries](#tools-and-libraries)
4. [Tasks and Methods](#tasks-and-methods)
5. [Usage Instructions](#usage-instructions)
6. [Conclusion](#conclusion)

---

## Project Overview

This project focuses on analyzing a dataset of credit card transactions using Python for data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning modeling. The goal is to detect fraud, classify transaction types, and analyze customer segmentation.

---

## Dataset Details

- **Source**: SQLite database containing credit card transaction data [Dataset Link](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)


## Tools and Libraries

### Libraries:
- **Data Handling**: `pandas`, `numpy`, `sqlite3`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`
- **Modeling**: `RandomForestRegressor`, `LinearRegression`, `xgboost`

### Techniques:
- Missing data analysis and imputation
- Feature engineering and selection
- Machine learning modeling and evaluation
- Hyperparameter tuning

---

## Tasks and Methods

### 4. Missing Values and Duplicates
- **Method**: Calculated missing values and duplicates for each table. 
- **Output**: Summary of missing/duplicate entries and handling plan.

### 5. Statistical Summary
- **Method**: Generated descriptive statistics and medians for numerical features in all tables.
- **Output**: Insights into customer and transaction behavior.

### 6. Fraud Detection
- **Method**: Applied anomaly detection techniques (e.g., Isolation Forest).
- **Output**: Identified potential fraudulent transactions.

### 7. Correlation Analysis
- **Method**: Correlation matrix of relevant features using a heatmap.
- **Output**: Insights into feature relationships and dependencies.

### 8. Outlier Detection
- **Method**: Used IQR to identify outliers in goals and other features.
- **Output**: Outlier summary with boundaries.

### 9. Handling Missing Data
- **Method**: Used k-means clustering to group customers by behavior.
- **Output**: Segmentation for targeted marketing.

### 10. Transaction Classification
- **Method**: Trained supervised learning models like Random Forest and XGBoost.
- **Output**:  Classification accuracy and feature importance analysis.

### 11. Feature Scaling
- **Method**: Used `StandardScaler` for Z-score normalization of numerical features.
- **Output**: Scaled data saved to the database.

### 12. Feature Selection
- **Method**: Applied Random Forest for feature importance and selected top features.
- **Output**: Key features identified and saved.

### 13. Algorithm Selection
- **Method**: Compared Linear Regression, Random Forest, and Gradient Boosting using cross-validation.
- **Output**: Model performance summary.

### 14. Data Splitting
- **Method**: Performed 80/20 train-test split.
- **Output**: Prepared datasets for modeling.

### 15. Model Training
- **Method**: Trained models on the training set and evaluated using `R²`, `MAE`, and `RMSE`.
- **Output**: Model metrics and predictions.

### 16. Model Evaluation
- **Method**: Evaluated the best model on the test set using performance metrics.
- **Output**: Model evaluation results.

### 17. Performance Analysis
- **Method**: Analyzed predictions and performance metrics on the test set.
- **Output**: Detailed performance analysis.

### 18. Model Improvement
- **Method**: Gusiian Framwork evaluation tuning with `XGBoost`.
- **Output**: Improved model performance.

### 19. Validation
- **Method**: Applied cross-validation and compared results with tuned models.
- **Output**: Cross-validation scores and best hyperparameters.

### 20. Final Model Selection
- **Method**: Compared models and selected the best one based on performance.
- **Output**: Final model for predictions.

### 21. Data Distribution
- **Method**: Visualized histograms for all features.
- **Output**: Comprehensive distribution analysis.

### 22. Feature Importance
- **Method**: Plotted feature importance from Random Forest.
- **Output**: Ranking of most influential features.

### 23. Model Performance Across Features
- **Method**: Assessed performance using subsets of features.
- **Output**: Feature subset performance metrics.

---

## Usage Instructions

1. **Requirements**:
   - Ensure you have Python 3.x installed.
   - Install the required libraries using the following command:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```

2. **Run Jupyter Notebook**:
   - Open the provided `.ipynb` file in Jupyter Notebook.
   - Execute cells sequentially to replicate results.

3. **Database Connection**:
   - Ensure the `database.sqlite` file is in the same directory.

---

## Conclusion

This project delivers a comprehensive analysis of credit card transactions with actionable insights into fraud detection and customer segmentation. The workflow integrates robust data handling, EDA, and machine learning modeling, showcasing practical applications in financial analytics.

---

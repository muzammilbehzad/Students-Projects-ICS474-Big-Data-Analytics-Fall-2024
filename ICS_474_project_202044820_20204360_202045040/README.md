# Soccer Data Analytics Project

This project analyzes a comprehensive soccer dataset to uncover patterns, predict outcomes, and evaluate player/team performance using machine learning techniques.

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

This project explores a soccer dataset using Python for data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning modeling. The ultimate goal is to predict player ratings and match outcomes and understand feature importance in these predictions.

---

## Dataset Details

- **Source**: SQLite database containing soccer data [Dataset Link](https://www.kaggle.com/datasets/hugomathien/soccer)
- **Tables Used**:
  - `Country`: Information about countries
  - `League`: Details of soccer leagues
  - `Match`: Match-level data, including scores and betting odds
  - `Player`: Information about players
  - `Player_Attributes`: Player performance metrics
  - `Team`: Team-level data
  - `Team_Attributes`: Team performance metrics

---

## Tools and Libraries

### Libraries:
- **Data Handling**: `pandas`, `numpy`, `sqlite3`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`
- **Modeling**: `RandomForestRegressor`, `LinearRegression`

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
- **Output**: Insights into feature distributions.

### 6. Data Distribution Analysis
- **Method**: Visualized distributions of numerical features (e.g., goals, player ages, height).
- **Output**: Histograms and KDE plots.

### 7. Correlation Analysis
- **Method**: Correlation matrix of relevant features using a heatmap.
- **Output**: Insights into feature relationships and dependencies.

### 8. Outlier Detection
- **Method**: Used IQR to identify outliers in goals and other features.
- **Output**: Outlier summary with boundaries.

### 9. Handling Missing Data
- **Method**: Imputed missing values using mean/mode strategies.
- **Output**: Cleaned data saved to the database.

### 10. Encoding Categorical Variables
- **Method**: Applied one-hot and label encoding for categorical features.
- **Output**: Encoded data saved to the database.

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
- **Method**: Hyperparameter tuning with `RandomizedSearchCV` and `GridSearchCV`.
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
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. **Run Jupyter Notebook**:
   - Open the provided `.ipynb` file in Jupyter Notebook.
   - Execute cells sequentially to replicate results.

3. **Database Connection**:
   - Ensure the `database.sqlite` file is in the same directory.

---

## Conclusion

This project provides a detailed analysis of soccer data, including predictions, feature importance, and machine learning evaluations. By implementing data preprocessing, EDA, and model training, this project offers a comprehensive workflow for soccer analytics.

---

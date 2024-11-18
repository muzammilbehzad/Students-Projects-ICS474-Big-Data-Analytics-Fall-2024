ICS474 Project: Uber Trip Classification

Team Members

Nawaf Althunayyan (201820500)
Hamza Alhelal (201865160)

Table of Contents

Project Overview
Dataset
Data Preprocessing
Modeling
Results and Evaluation
Visualization
How to Run
Dependencies
Project Overview

This project focuses on classifying Uber trips as either "Business" or "Personal" using historical trip data. It aims to understand patterns in travel behavior, optimize expense categorization, and improve operational efficiency.

Dataset

Source: Uber anonymized trip logs.
Structure: 1,156 rows and 7 columns.
Key Features:
START_DATE, END_DATE: Trip start and end timestamps.
CATEGORY: Target variable indicating trip type.
START, STOP: Trip start and end locations.
MILES: Distance traveled.
PURPOSE: Reason for the trip (e.g., Meeting, Errand).
Data Issues
Missing Values: Found in PURPOSE, END_DATE, CATEGORY, START, and STOP.
Duplicates: One duplicated row identified.
Data Preprocessing

Missing Data Handling:
Critical columns: Dropped rows with missing values.
Non-critical columns: Imputed or left as-is.
Encoding:
Label encoding for CATEGORY.
One-hot encoding for START, STOP, PURPOSE.
Feature Scaling:
Standard scaling for MILES.
Feature Selection:
Used domain knowledge to select MILES, PURPOSE, START, and STOP.
Modeling

Algorithm: Logistic Regression for binary classification.
Data Splitting: 80% training and 20% testing.
Evaluation Metrics: Accuracy, Precision, Recall, F1-Score.
Results and Evaluation

Accuracy: 94%
Precision: 94%
Recall: 100%
F1-Score: 97%
The model effectively differentiates between Business and Personal trips.

Visualization

Data Distribution: Visualized trip frequencies.
Feature Importance: Highlighted significant predictors.
Outlier Detection: Box plots for MILES.
How to Run

Clone the repository.
Ensure all dependencies are installed.
Run the main script:
python main.py
Dependencies

Python 3.x
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib

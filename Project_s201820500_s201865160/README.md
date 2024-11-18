# ICS474 Project: Uber Trip Classification

## Team Members
- **Nawaf Althunayyan** (201820500)  
- **Hamza Alhelal** (201865160)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Modeling](#modeling)
5. [Results and Evaluation](#results-and-evaluation)
6. [Visualization](#visualization)
7. [How to Run](#how-to-run)
8. [Dependencies](#dependencies)

---

## Project Overview
This project focuses on classifying Uber trips as either "Business" or "Personal" using historical trip data. It aims to understand patterns in travel behavior, optimize expense categorization, and improve operational efficiency.

---

## Dataset
- **Source**: Uber anonymized trip logs.
- **Structure**: 1,156 rows and 7 columns.

### Key Features:
- **START_DATE, END_DATE**: Trip start and end timestamps.
- **CATEGORY**: Target variable indicating trip type.
- **START, STOP**: Trip start and end locations.
- **MILES**: Distance traveled.
- **PURPOSE**: Reason for the trip (e.g., Meeting, Errand).

### Data Issues:
- **Missing Values**: Found in `PURPOSE`, `END_DATE`, `CATEGORY`, `START`, and `STOP`.
- **Duplicates**: One duplicated row identified.

---

## Data Preprocessing
1. **Missing Data Handling**:
   - Critical columns: Dropped rows with missing values.
   - Non-critical columns: Imputed.

2. **Encoding**:
   - Label encoding for `CATEGORY`.
   - One-hot encoding for `START`, `STOP`, and `PURPOSE`.

3. **Feature Scaling**:
   - Standard scaling for `MILES`.

4. **Feature Selection**:
   - Selected `MILES`, `PURPOSE`, `START`, and `STOP` using domain knowledge.

---

## Modeling
- **Algorithm**: Logistic Regression for binary classification.
- **Data Splitting**: 80% training and 20% testing.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score.

---

## Results and Evaluation
- **Accuracy**: 94%
- **Precision**: 94%
- **Recall**: 100%
- **F1-Score**: 97%

The model effectively differentiates between Business and Personal trips.

---

## Visualization
1. **Data Distribution**: Visualized trip frequencies.
2. **Feature Importance**: Highlighted significant predictors.
3. **Outlier Detection**: Box plots for `MILES`.

---
## Dependencies

- **Python** 

### Libraries:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For machine learning algorithms and model evaluation.
- **Matplotlib**: For data visualization.

---
## How to Run

Clone the repository.  
Ensure all dependencies are installed.  
Run the main script:  
`python Project.py`


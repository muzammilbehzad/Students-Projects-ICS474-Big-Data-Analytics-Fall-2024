# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using machine learning techniques. We leverage a dataset containing credit card transaction records to build a predictive model that can identify fraudulent transactions, with a focus on improving recall to minimize false negatives.

## Table of Contents
- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Visualizations](#visualizations)
- [Ethical Considerations](#ethical-considerations)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
Fraud detection is critical in the financial industry to prevent monetary losses and protect customer trust. This project uses a dataset of credit card transactions to develop a machine learning model for detecting fraudulent transactions.

The model is built using the Random Forest algorithm, which provides robustness against overfitting and the ability to handle high-dimensional data. To address class imbalance (since fraudulent transactions are rare), we use SMOTE (Synthetic Minority Over-sampling Technique).

## Dataset Overview
- **Source**: Kaggle Credit Card Transactions Dataset.
- **Size**: ~1.3m transactions, 24 features.
- **Features**:
  - Transaction details: `trans_date_trans_time`, `cc_num`, `merchant`, `category`, `amt`
  - Cardholder details: `first`, `last`, `gender`, `street`, `city`, `state`, `zip`, `lat`, `long`, `city_pop`, `dob`
  - Merchant details: `merch_lat`, `merch_long`, `merch_zipcode`
  - Target variable: `is_fraud` (0 for non-fraud, 1 for fraud)

## Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/credit-card-fraud-detection.git
    ```
2. **Install dependencies**:
    Ensure you have Python 3.x installed, then install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the project**:
    Follow the instructions in the `instructions.txt` file within the `code` folder for running the model.

## Data Preprocessing
- **Handling missing data**: Dropped rows with missing values.
- **Encoding categorical variables**: Applied one-hot encoding for features like `merchant`, `category`, and `gender`.
- **Feature scaling**: Standardized numerical features using `StandardScaler`.
- **Handling class imbalance**: Used SMOTE to oversample fraudulent transactions in the training data.

## Modeling
- **Algorithm**: Random Forest Classifier
- **Data split**: 80% training, 20% testing (stratified to maintain class distribution).
- **Evaluation metrics**:
  - Accuracy
  - Precision
  - Recall (focus on improving recall to detect more fraudulent transactions)

### Hyperparameter Tuning
Used `RandomizedSearchCV` to tune key hyperparameters like `n_estimators`, `max_depth`, and `min_samples_split` to improve model performance.

## Visualizations
- Distribution of transaction amounts (`amt`)
- Geographic distribution of cardholders and merchants (`lat`, `long`)
- Feature importance from the Random Forest model
- Fraud detection results (`is_fraud`)

## Ethical Considerations
- **Impact of misclassification**:
  - False negatives can result in financial loss.
  - False positives may inconvenience customers.
- **Fairness**: Ensured the model does not discriminate based on sensitive attributes (e.g., gender, age).
- **Data privacy**: Handled sensitive customer information responsibly and ensured the data was anonymized.

## Conclusion
- **Achievements**: Built a fraud detection model with improved recall to better identify fraudulent transactions.
- **Future work**: Explore advanced models like Gradient Boosting and incorporate real-time data for continuous learning.

## References
- Breiman, L. (2001). Random Forests. Retrieved from [Springer](https://link.springer.com/article/10.1023/A:1010933404324)
- Chawla, N. V. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Retrieved from [JAIR](https://www.jair.org/index.php/jair/article/view/10302)
- Dal Pozzolo, A. B. (2018). Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy. Retrieved from [DOI](https://doi.org/10.1109/TNNLS.2017.2736643)
- Pedregosa, F. V. (2023). Scikit-learn: Machine Learning in Python. Retrieved from [Scikit-learn](https://scikit-learn.org/stable/)

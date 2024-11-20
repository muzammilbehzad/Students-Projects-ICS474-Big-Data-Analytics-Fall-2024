# Credit Card Transactions Analysis 💳
This project focuses on analyzing credit card transaction data to detect fraudulent activities. By leveraging machine learning techniques, the goal is to enhance fraud detection systems and provide actionable insights into transaction patterns.

---
## Project Objectives
The primary objective of this project is to develop a predictive model that accurately identifies fraudulent transactions within a dataset of credit card transactions. By analyzing key transaction-related features, this project aims to:

- Identify significant factors contributing to fraud detection.
- Build robust and interpretable models using machine learning techniques.
- Evaluate the effectiveness of feature importance and model performance metrics.

---
## About the Dataset
The dataset contains detailed records of credit card transactions, including numerical and categorical features. It provides a rich source of information for fraud detection analysis.

**Columns:**
- trans_date_trans_time: Timestamp of the transaction.
- cc_num: Credit card number (hashed or anonymized).
- merchant: Merchant or store where the transaction occurred.
- category: Type of transaction (e.g., grocery, entertainment).
- amt: Amount of the transaction.
- first: First name of the cardholder.
- last: Last name of the cardholder.
- gender: Gender of the cardholder.
- street: Address details of the cardholder.
- city: Address details of the cardholder.
- state: Address details of the cardholder.
- zip: Address details of the cardholder.
- lat: Geographical coordinates of the transaction.
- long: Geographical coordinates of the transaction.
- city_pop: Population of the city where the transaction occurred.
- job: Occupation of the cardholder.
- dob: Date of birth of the cardholder.
- trans_num: Unique transaction number.
- unix_time: Unix timestamp of the transaction.
- merch_lat: Geographical coordinates of the merchant.
- merch_long: Geographical coordinates of the merchant.
- is_fraud: Indicator of whether the transaction is fraudulent.
- merch_zipcode: Geographical coordinates of the merchant.

---
## Main Code Files and Their Purpose
### 1. MainCode.ipynb
A Jupyter Notebook containing all the code for data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.
Includes visualizations such as histograms, boxplots, and feature importance charts to understand the dataset and improve model interpretability.
### 2. ICS474_Report.pdf
A comprehensive report outlining the project's objectives, data exploration process, key analysis findings, applied methodologies, and conclusions derived from the dataset.

---
## Instructions for Running the Code
In the notebook, the dataset is automatically downloaded in the second cell using the following command:
path = kagglehub.dataset_download("priyamchoksi/credit-card-transactions-dataset")
This command downloads the dataset and assigns the directory path to the variable path.

To load the dataset into a DataFrame, the following line is used:
creditCardTransactions = pd.read_csv(os.path.join(path, "credit_card_transactions.csv"))

# ICS 474 Project: Credit Card Fraud Transaction Detection and Analysis Using Machine Learning


## Project Objectives

This project focuses on analyzing and detecting fraudulent transactions within the Credit Card Transactions dataset. The specific objectives are:
1.  **Data Exploration**:
    
    -   Understand the dataset structure, including features, data types, and distributions.
    -   Visualize data to uncover patterns and anomalies related to fraudulent and legitimate transactions.
2.  **Data Preprocessing**:
    
    -   Handle missing values to prepare the dataset for analysis.
    -   Apply techniques such as feature scaling and encoding for model compatibility.
3.  **Fraud Detection Modeling**:
    
    -   Implement machine learning model "Random Forest Classifier" to classify transactions as fraudulent or legitimate.
    
4.  **Model Evaluation**:
    
    -   Evaluate the models using metrics such as accuracy, precision, recall, and F1-score as implemented in the notebook.
    -   Compare results to identify the best-performing model.
5.  **Visualization**:
    
    -   Generate plots and graphs that illustrate key trends, feature importance, and model performance.

---

## Utilized Libraries

-   **Data Analysis and Visualization**:
    
    -   `pandas`: For data manipulation and analysis.
    -   `numpy`: For numerical operations.
    -   `matplotlib.pyplot`: For creating static visualizations.
    -   `seaborn`: For statistical data visualization.
-   **PySpark**:
    
    -   `pyspark.sql`: For handling structured data using SQL-like operations.
    -   `pyspark.sql.functions`: For transformations and SQL functions.
    -   `pyspark.sql.types`: For defining data types.
    -   `pyspark.ml`: For machine learning operations, including pipelines.
    -   `pyspark.ml.feature`: For feature engineering and transformations.
    -   `pyspark.ml.linalg`: For handling feature vectors.
    -   `pyspark.ml.classification`: For building classification models (e.g., Random Forest).
    -   `pyspark.ml.evaluation`: For evaluating model performance.
-   **Other**:
    
    -   `os`: For interacting with the operating system.
    -   `kagglehub`: For interacting with Kaggle datasets, specifically importing the dataset in our project.
---

## Main Code File: **ICS_474_Project.ipynb**

### **Purpose**

The notebook serves as the primary file for implementing the project tasks and objectives. It is structured to cover the following key areas:

#### **Part 1: Data Understanding and Exploration**

-   Loads and examines the dataset's structure and features.
-   Provides statistical summaries and visualizes data distributions.
-   Explores correlations and identifies outliers.

#### **Part 2: Data Preprocessing**

-   Handles missing values.
-   Encodes categorical features and Scales numerical data.
-  Selects features to be used in training.

#### **Part 3: Modeling**

-   Implements a **Random Forest Classifier** for fraud detection.
-   Trains, evaluates, and tunes the model to improve performance.

#### **Part 4: Visualization**

-   Visualizes data insights, feature importance, and model performance.

This file encapsulates the complete workflow from data ingestion to fraud detection modeling and evaluation, making it central to the project's deliverables.

---
### Instructions for Running the Code

#### **Prerequisites**

1.  **Environment**:
    
    -   Python 3.8+ is recommended.
    -   Install Jupyter Notebook to execute the code interactively.
2.  **Required Libraries**:
    
    -   Install the following Python libraries using pip:
        
        `$ pip install pandas numpy matplotlib seaborn pyspark` 
        

#### **Steps to Run**

1.  **Clone or Download**:
    
    -   Clone this repository or download the project files to your local machine.
2.  **Launch Jupyter Notebook**:
    
    -   Navigate to the directory containing the `ICS_474_Project.ipynb` file.
    -   Run the following command to start Jupyter Notebook:
 
        `jupyter notebook` 
        
3.  **Open the Notebook**:
    
    -   In the Jupyter interface, open `ICS_474_Project.ipynb`.
4.  **Execute the Notebook**:
    
    -   Run the cells in sequence by clicking `Shift + Enter` for each cell.
    -   Follow the comments and instructions in the notebook for each step.
5.  **View Results**:
    
    -   Review the outputs, visualizations, and model evaluation metrics displayed as you execute the cells.

#### **Dataset Access**

-   Ensure the Credit Card Transactions dataset is available in the specified path or update the dataset path in the notebook accordingly.
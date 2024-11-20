
# Uber Ride Analysis and Classification

This repository contains the final project for the **ICS474: Big Data Analytics** course at King Fahd University of Petroleum and Minerals. The project focuses on analyzing an Uber dataset to classify rides as either "Business" or "Personal" using machine learning models. The work includes data exploration, preprocessing, modeling, and visualization, presented in a Jupyter Notebook.

---

## Final Report & Presentation

For a detailed explanation of the methodology, results, and analysis, you can refer to the **Final Report**:  
[Final Report](ICS474_Project_FinalReport.pdf)

For a concise overview of the project methodology, results, and key insights, you can refer to the **Presentation Slides**:
[Presentation Slides](Presintation.pptx)

---

## Project Overview

The goal of this project is to extract meaningful insights from the Uber dataset and build predictive models to classify ride categories. The main tasks include:

1. **Data Understanding**: Explore the dataset for patterns, correlations, and outliers.
2. **Preprocessing**: Clean the data by handling missing values, encoding categorical variables, and scaling features.
3. **Modeling**: Build and evaluate classification models using Decision Tree and Random Forest algorithms.
4. **Visualization**: Use plots to illustrate data distributions, feature importance, and model performance.
5. **Performance Analysis**: Improve models through hyperparameter tuning and cross-validation.

---

## Dataset

The dataset used in this project is the **Uber dataset from Kaggle**, which includes details about ride times, categories, distances, and purposes. Key features include:

- `START_DATE`, `END_DATE`: Datetime of the ride.
- `CATEGORY`: Ride type ("Business" or "Personal").
- `MILES`: Distance covered.
- `PURPOSE`: Reason for the ride (e.g., "Meeting").

---

## Steps and Methodology

### 1. **Data Preprocessing**
   - Handled missing values by filling with appropriate labels.
   - Encoded categorical features using label encoding.
   - Scaled numerical features to mitigate the effect of outliers.
   - Reduced redundancy by replacing features like `START_DATE` with derived metrics such as `DURATION`.

### 2. **Modeling**
   - Used **Decision Tree** as a baseline model and **Random Forest** for improved accuracy.
   - Split the dataset into 70% training and 30% testing.
   - Optimized hyperparameters using GridSearchCV.

### 3. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1 Score (chosen as the primary metric due to class imbalance).

### 4. **Visualization**
   - Visualized data distributions, feature importance, and model performance using histograms, bar plots, and box plots.

---

## Results and Findings

- **Best Model**: Random Forest Classifier
  - Achieved higher F1 Score and better generalization than the Decision Tree.
- **Key Features**: `MILES` and `DURATION` were the most predictive features.
- **Limitations**:
  - Small dataset (1,155 rows) prone to overfitting.
  - Class imbalance skewed towards "Business" rides.
  - Missing values in `PURPOSE` limited its use in modeling.

---

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/muzammilbehzad/Students-Projects-ICS474-Big-Data-Analytics-Fall-2024.git
   ```
   
2. Navigate to the project folder:
   ```bash
   cd Students-Projects-ICS474-Big-Data-Analytics-Fall-2024/Project_s202045080_s202039000
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook FinalProject.ipynb
   ```

5. Follow the steps in the notebook to explore, preprocess, and model the data.

---

## Team Members

- **Ali Jaber** (202045080)
- **Abdullah Al Muheef** (202039000)

Supervised by **Dr. Muzammil Behzad**.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

# Used Cars Prices Prediection Project

This project aims to predict the selling_price of cars based on features such as year, km_driven, fuel type, etc., making it a regression problem. Since we are predicting a continuous variable, regression algorithms are particularly suitable

## Project Overview

This project aims to build a predictive model for estimating the selling prices of used cars, utilizing the "Car Details Dataset" by Akshay Dattatray Khare from Kaggle. The dataset contains detailed attributes for each car, such as model year, mileage, fuel type, and ownership history, which are all influential factors in determining resale prices. The goal is to analyze these features, clean and preprocess the data, and then apply machine learning techniques to develop a reliable model that accurately predicts car prices. 

### Libraries and Packages Used

- **pandas**: Data manipulation
- **pandas_datareader**: Financial data retrieval
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Enhanced data visualizations


### Installation

Ensure that you have **Python 3.8 or later** installed.

1. **Clone this repository** and navigate into the project directory.

2. **Install the required packages**:

   ```bash
   pip install seaborn
   pip install matplotlib
   pip install scikit-learn

   ```


3. **Run the notebook**:

   ```bash
   jupyter notebook
   ```

   Open the `.ipynb` file in the Jupyter Notebook environment and run the cells sequentially.

### Usage

The notebook is divided into several sections:

1. **Data Understanding and Exploration**: Loads data from the kagle website as csv file.
2. **Data Preprocessing**: Remove missing data .
3. **Model Building and Training**: Uses three algorithims Linear Regression,	Random Forest Regressor ,and Gradient Boosting Regressor .

---

### Troubleshooting

**Note : There are some issues with jupyterNoteBook if there any issue you can use the code in VS  
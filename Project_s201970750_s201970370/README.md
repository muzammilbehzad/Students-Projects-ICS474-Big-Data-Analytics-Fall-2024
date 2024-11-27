
# Uber Trip Data Analysis and Prediction

This project analyzes and models Uber trip data to predict trip-related metrics such as mileage and cost. It implements two machine learning models, **Linear Regression** and **Random Forest Regressor**, to predict trip mileage and estimates costs. The project includes data preprocessing, feature engineering, model training, evaluation, and visualizations.

## Project Overview

The goal of this project is to explore and analyze Uber trip data, including:

- **Mileage prediction**: Predict how far a trip will go based on various features.
- **Cost estimation**: Estimate the cost of a trip using a simple formula based on miles traveled.
- **Data insights**: Visualize patterns and correlations in the dataset.

The project uses the **UberDataset.csv** file, which contains trip data, including:

- Start and end dates (`START_DATE`, `END_DATE`)
- Miles traveled (`MILES`)
- Trip purpose (`PURPOSE`)
- Categorical columns like `CATEGORY`, `START`, and `STOP`.

## Requirements

To run this project, you'll need the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

You can install the required libraries by running:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Data

The dataset, `UberDataset.csv`, contains information on various Uber trips. The data preprocessing steps include:

- Filling missing values in the `PURPOSE` column with 'Unknown'.
- Converting date columns to datetime format for calculating trip durations.
- Removing rows with invalid or missing dates.
- One-hot encoding categorical variables for use in machine learning models.
- Splitting the data into training (80%) and testing (20%) sets.
- Standardizing features using `StandardScaler`.

## How to Run

1. Clone the repository or download the code.
2. Ensure `UberDataset.csv` is in the same directory as the script (`uber_analysis.py`).
3. Run the Python script using:

```bash
python uber_analysis.py
```

### The script outputs:
- **Data preprocessing logs**: Step-by-step data handling and transformations.
- **Model performance metrics**: Evaluation results for the machine learning models.
- **Visualizations**: Separate plots for data insights and model performance.

## Model Evaluation Results

### **Random Forest Regressor**:
- **MAE**: 4.91
- **MSE**: 195.73
- **RMSE**: 13.99
- **R² Score**: 0.74

## Visualizations

- **Correlation Heatmap**: Shows the correlations between numerical and one-hot encoded features. Helps identify strong relationships, such as between `MILES` and `Trip_Duration`.
- **Scatter Plot: Predicted Cost vs Actual Cost**: Visualizes the accuracy of the cost predictions. A diagonal line represents perfect predictions.
- **Histogram of Log Trip Duration**: Displays the distribution of trip durations (log-scaled) to highlight patterns in longer trips.
- **Boxplot of Miles**: Highlights outliers in the `MILES` variable.

## Cost Prediction

The project includes a cost estimation formula:

```
Cost = $0.5 per mile + $2 base fare
```

Seasonal rates and other features (e.g., traffic conditions) can be incorporated in future versions for more accurate cost predictions.

## Future Work

1. **Enhance Feature Engineering**:
   - Include additional features like time of day or traffic conditions.
   
2. **Hyperparameter Optimization**:
   - Use **Grid Search** or **Randomized Search** to improve model performance.

3. **Expand Dataset**:
   - Apply the model to multi-city Uber datasets for increased generalizability.

## Acknowledgments

This project uses open-source Python libraries and builds on foundational machine learning techniques. Special thanks to the dataset contributors for providing the raw data for analysis.

---

Feel free to open an issue or submit a pull request if you'd like to contribute or suggest improvements to the project.

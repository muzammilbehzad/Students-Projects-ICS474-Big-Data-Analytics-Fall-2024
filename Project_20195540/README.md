# Folder creation init commit
# Player Attributes Regression Analysis
## Project Objectives
The goal of this project is to predict a player's overall rating based on various attributes using a machine learning regression model. This project employs feature engineering, data visualization, and model training to identify and leverage the most critical factors influencing player ratings.

## Main Files
`ICS474-Project.ipynb`:

 - This is the main Jupyter Notebook containing:
    - Data preprocessing (loading, cleaning, and encoding).
    - Feature engineering and selection.
    - Model training and evaluation.
    - Visualizations for data analysis and feature importance.

`soccer.zip`:

- Contains the database file (database.sqlite) used as the raw data source for the project.

README.md:

- Documentation describing the project objectives, structure, and instructions.

# Instructions for Running the Notebook

1. Prerequisites
Python Environment: Ensure Python 3.8+ is installed.
Jupyter Notebook: Install Jupyter Notebook to run the .ipynb file.
Required Libraries: Install dependencies by running:
```bash
pip install -r requirements.txt
```
Required libraries include:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
2. Extract the Database
Unzip the soccer.zip file to extract the database.sqlite file.
Ensure the extracted file is in the same directory as the Jupyter Notebook.
3. Run the Notebook
- Open the ICS474-Project.ipynb file using Jupyter Notebook:
```bash
jupyter notebook ICS474-Project.ipynb
```

- Execute each cell in sequence to:
    - Load and preprocess the data.
    - Train the regression model.
    - Evaluate the model's performance.
    - Generate visualizations for data distribution and feature importance.
4. Results
- Model Performance:
    - Root Mean Squared Error (RMSE): ~0.92
    - R² Score: ~0.98
**Key Features**: The most influential features identified include:
- reactions
- potential
- ball_control
- standing_tackle
- gk_diving
## Visualizations
The notebook provides visual insights into:

1. Data Distribution:
Histograms and boxplots for numerical features to identify patterns and outliers.
2. Feature Importance:
A bar chart ranking features by their importance to the model.
3. Model Predictions:
Residual plots to assess prediction accuracy across data subsets.
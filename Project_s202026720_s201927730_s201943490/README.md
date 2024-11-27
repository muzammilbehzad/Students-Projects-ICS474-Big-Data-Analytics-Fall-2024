# Instructions:

1. After running the first cell, copy the data from the .cache printed output (copy path directory).
2. Ensure that sqllite3, scikit-learn, seaborn are installed, if not run (pip install sqllite3, scikit-learn, seaborn)
3. Run all.
---

# European Soccer Analytics Project

## Overview
This project is a comprehensive analysis of the European Soccer Database, sourced from Kaggle. The dataset contains data spanning matches, players, teams, and betting odds across 11 countries from 2008 to 2016. The primary goal is to explore the dataset, identify key insights, visualize the data effectively, and build predictive models for soccer analytics.

The project is divided into four main parts:
1. Data Understanding and Exploration
2. Data Preprocessing
3. Visualization
4. Modeling

---

## Dataset Description
- **Source**: Kaggle's European Soccer Database  
- **Contents**: Matches, player attributes, team attributes, and betting odds for European leagues  
- **Objective**: Understand soccer match dynamics, evaluate player/team performance, and explore betting odds  
- **Key Features**:  
  - **Player stats** (e.g., `overall_rating`, `potential`, `reactions`)  
  - **Team tactics and performance** (e.g., `buildUpPlaySpeed`, `chanceCreationPassing`)  
  - **Match details** (e.g., `goals scored`, `dates`, and `stages`)  

---

## Steps to Download and Prepare the Dataset
To reproduce this project, follow these steps:
1. Visit the Kaggle dataset page: *European Soccer Database*.
2. Download the dataset files as a single ZIP file.
3. Extract the contents into a directory called `data/`.
4. Verify that the following CSV files are present in the `data/` directory:
   - `Country.csv`
   - `League.csv`
   - `Match.csv`
   - `Player.csv`
   - `Player_Attributes.csv`
   - `Team.csv`
   - `Team_Attributes.csv`
5. Ensure that all notebooks and scripts appropriately reference the `data/` folder for reading the files.

---

## Structure of the Project

### 1. Data Understanding and Exploration
- **Feature Descriptions**:  
  Documented each table (e.g., `Match`, `Player`, `Team`) with feature types, descriptions, and relevance.
- **Missing Values**:  
  Identified columns with significant missing data (e.g., `attacking_work_rate` in `Player_Attributes`, betting odds in `Match`).
- **Outliers**:  
  Used histograms and statistical methods to detect and visualize outliers.
- **Key Insights**:  
  - Features like `potential`, `reactions`, and `ball_control` were highly correlated with player `overall_rating`.
  - Missing betting odds significantly impacted match predictions.

### 2. Data Preprocessing
- **Handling Missing Values**:  
  For numeric columns with minimal missing values, rows were dropped to avoid bias.
- **Encoding Categorical Variables**:  
  Categorical features like `preferred_foot` were one-hot encoded.
- **Feature Selection**:  
  Selected top 5 features based on mutual information and correlation: `potential`, `reactions`, `ball_control`, `short_passing`, and `standing_tackle`.

### 3. Visualization
- **Key Graphs**:  
  - Histograms for detecting outliers in numerical features.
  - Correlation heatmaps to evaluate relationships between variables.
  - Bar charts and rankings of top teams and players.
- **Findings**:  
  - Insights into team and player performance:  
    - Teams with the most wins and goals scored.  
    - Players with the highest ratings and contributions.

### 4. Modeling
- **Algorithm**: Random Forest Regressor  
- **Data Splitting**: 70-15-15 split for training, validation, and testing.  
- **Evaluation Metrics**:  
  - **R² Score**: Achieved 0.95, indicating high variance explanation.  
  - **Mean Squared Error (MSE)**: Achieved 2.63.  
- **Feature Importance**:  
  Calculated feature importance, highlighting key player attributes.

---

## Key Findings and Visualizations
1. **Data Exploration**:  
   - Detailed breakdown of features, missing values, and duplicates.
   - Data distributions visualized using histograms and box plots.
   - Correlation heatmaps highlighting relationships between features.
2. **Insights**:  
   - Most predictive player attributes: `potential`, `reactions`, `ball_control`.  
   - Tactical features like `buildUpPlayPassing` impact match outcomes.  
   - Teams like FC Barcelona and Real Madrid dominated in goals and wins.
3. **Performance Analysis**:  
   - The Random Forest model demonstrated strong predictive capabilities, balancing accuracy and interpretability.

---

## Instructions to Run
1. **Dataset Setup**:  
   Ensure all datasets (`Match`, `Player`, `Team`, etc.) are in the same directory as the notebooks.
2. **Notebooks**:  
   Each example and step is provided in separate cells for clarity and modularity.
3. **Dependencies**:  
   - Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`.  
   - Install via: `pip install -r requirements.txt`.

---

## Files Included
- **README.md**: This file.
- **main_report.pdf**: Comprehensive documentation of the project.
- **notebooks/**: Jupyter Notebooks for each part of the project.
  - `part1_exploration.ipynb`: Data exploration and visualization.
  - `part2_preprocessing.ipynb`: Preprocessing and feature engineering.
  - `part3_visualization.ipynb`: Advanced visualization and insights.
  - `part4_modeling.ipynb`: Model training, evaluation, and performance analysis.
- **data/**: Contains CSV files of the datasets (e.g., `Player.csv`, `Team.csv`).

---

## Contributors
- Turki Almutiri (201927730)  
- Ahmed Shewaikan (202026720)  
- Ali Al-Saleh (201943490)  

---

## References
- Kaggle Dataset: *European Soccer Database*  
- Betting Odds Documentation: *Football Data Notes*  

---

## Presentation
https://www.canva.com/design/DAGXcID0YHc/Fyon9Bn43QQ8TVtY3YlaLw/view?utm_content=DAGXcID0YHc&utm_campaign=designshare&utm_medium=link&utm_source=editor


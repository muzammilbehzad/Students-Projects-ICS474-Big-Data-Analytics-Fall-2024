# ASHRAE Energy Prediction Project


## Table of Contents


 - Project Overview
 - Dataset Description
 - Project Structure
 - Key Findings 
 - Future Improvements 
 - License 
 - Acknowledgments 

## Project Overview

This project focuses on predicting energy consumption for buildings using the ASHRAE Great Energy Predictor III dataset from Kaggle. The dataset includes building information, weather data, and energy usage readings, enabling the development of a predictive model to forecast energy usage across diverse building types and conditions.

## Download the PPT File

You can download the presentation file using the link below:

[Download the presentation](https://file.io/JoGzhzLDTziK)


## Dataset Description

The project utilizes the following CSV files from the competition dataset:

building_metadata.csv: Contains building-specific data (building_id, primary_use, square_feet, etc.).
weather_train.csv: Weather data related to the training set, with fields such as site_id, timestamp, air_temperature, and wind_speed.
train.csv: The main training dataset with meter readings over time for various buildings.
weather_test.csv and test.csv: Data for making predictions on unseen data.

## Project Structure

1. Data Preprocessing
Memory Optimization: Function to reduce DataFrame size for efficient processing.
Merging Datasets: Combining building, weather, and energy data into a unified training DataFrame.
Handling Missing Values: Strategies like forward filling, backward filling, and mean imputation.
2. Exploratory Data Analysis (EDA)
Descriptive Statistics: Summary statistics to understand key features.
Visualizations: Histograms and heatmaps to explore feature distributions and relationships.
Outlier Detection: Identification of outliers using statistical methods.
3. Feature Engineering
Categorical Encoding: One-hot encoding applied to categorical features (e.g., primary_use).
Timestamp Transformation: Extracting Month and Day from timestamps.
4. Modeling
Model Used: Basic linear regression implemented with scikit-learn.
Training and Validation: Data split using an 80/20 train-test split.
5. Model Evaluation
Feature Importance: Analysis of model coefficients to identify significant features.
Performance Metrics: Evaluation using metrics like Mean Squared Error (MSE).

## Key Findings

Data Characteristics: Initial data analysis highlighted missing values and potential inconsistencies.
Feature Importance: Key influential features included air_temperature, square_feet, and cloud_coverage.
Outlier Analysis: Detected anomalies in meter_reading that could affect predictions.

## Future Improvements

Algorithm Enhancements: Implement advanced algorithms like XGBoost or Random Forest.
Cross-Validation: Apply K-fold cross-validation for improved generalization.
Hyperparameter Tuning: Optimize models for better accuracy.
Feature Scaling: Investigate feature scaling for algorithms sensitive to magnitude differences.

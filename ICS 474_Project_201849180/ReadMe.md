# Uber Data Analysis and Prediction Project

This repository contains the work done for the **Uber Data Analysis and Prediction** project as part of the **ICS474 - Big Data Analytics** course.

## Project Objectives
1. **Analyze and visualize Uber trip data** to identify patterns and trends.
2. **Build a machine learning model** to classify trips into distance categories (Short, Medium, Long).

---

## Dataset Overview
- **Source**: Uber trip dataset with 1,156 records.
- **Columns**:
  - `START_DATE`: Start date and time of the trip.
  - `END_DATE`: End date and time of the trip.
  - `CATEGORY`: Type of trip (e.g., Business, Personal).
  - `START`: Start location of the trip.
  - `STOP`: End location of the trip.
  - `MILES`: Distance of the trip in miles.
  - `PURPOSE`: Purpose of the trip (e.g., Meeting, Meal, etc.).

---

## Project Workflow
1. **Data Understanding and Exploration**
   - Statistical summaries and data distributions.
   - Handling missing and duplicate values.
   - Visualizing trip patterns and purposes.

2. **Data Preprocessing**
   - Encoding categorical variables.
   - Feature scaling and normalization.
   - Feature selection and engineering.

3. **Modeling**
   - Algorithm selection: Random Forest Classifier.
   - Model training, evaluation, and cross-validation.

4. **Visualization**
   - Visualized key patterns and model results using histograms, bar charts, and feature importance plots.

---

## Setup and Installation

### Requirements
The following Python libraries are required:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install these packages using:
```bash
pip install -r requirements.txt


#  European Soccer Database

## Project Objectives
The purpose of this project is to explore the Soccer dataset from Kaggle and develop a predictive model to determine the likelihood of a team winning a match. This involves analyzing key features from team attributes, such as build-up play, chance creation, and defensive capabilities, alongside match-related data like goals scored and possession statistics. The project includes data cleaning, feature exploration, and the use of machine learning models for accurate predictions.

---

## Summary of the Dataset Used 

### 1-Dataset Overview

The dataset used in this project is sourced from Kaggle and focuses on soccer matches, teams, and players across several European leagues. It spans 11 seasons, from 2008 to 2016, providing rich data on player and team attributes, match results, and league information. This dataset is especially suited for machine learning and data analysis in the domain of sports analytics. It addresses the problem domain of predicting match outcomes based on team attributes and performance data. 

### `Match.csv from Match.zip`
- Contains data about 25,979 matches, including details like goals scored, team API IDs, and additional game events such as possession, corners, and fouls. 

### `Player.csv`
- Information on 11,060 players, including their names, height, weight, and birthdays. 

### `Player Attributes.csv from Player Attributes.zip `
- Contains 183,978 entries describing player abilities such as overall 
rating, potential, and specific attributes like crossing, dribbling, and finishing.

### `Team.csv`
- Includes 299 teams with their API IDs, names, and abbreviations.

### `Team Attributes.csv` 
- Contains 1,458 entries describing team characteristics like build-up 
play, chance creation, and defensive strategies. 

### `Country.csv` 
- Contains 11 entries representing countries with their respective IDs. 

### `League.csv` 
- Provides information on 11 leagues, including league and country IDs. 

---

## Main Code Files and Their Purposes


- [**`ICS474-project .ipynb`**](https://github.com/muzammilbehzad/Students-Projects-ICS474-Big-Data-Analytics-Fall-2024/blob/main/Project_s202025400_s202034520_s202170070/ICS474-project.ipynb): Contain all codes, figures, and results for the project.

- [**`Project Report .pdf`**](https://github.com/muzammilbehzad/Students-Projects-ICS474-Big-Data-Analytics-Fall-2024/blob/main/Project_s202025400_s202034520_s202170070/Project%20Report.pdf): This report outlines the procedures followed in the project, presents the findings and includes relevant figures to illustrate the results.
  
- [**`ICS474-Presntation.pptx`**](https://github.com/muzammilbehzad/Students-Projects-ICS474-Big-Data-Analytics-Fall-2024/blob/main/Project_s202025400_s202034520_s202170070/ICS474-Presntation.pptx): This will briefly show the presentation about our project.

- [**`final_data .csv`**](https://github.com/muzammilbehzad/Students-Projects-ICS474-Big-Data-Analytics-Fall-2024/blob/main/Project_s202025400_s202034520_s202170070/final_data.csv): contains all the necessary data for building the model. 
---

## Instructions for Running the Code

1. **Download all the CSV, zip file data, and the ipynb file to run the code** 
2. **Install the libraries used using the command in this README file** 
3. **Run Jupyter Notebook**:
   - Follow the sequence of notebook
   - Complete each cell to ensure the datasets are properly processed and models are trained.
---

## Required Installations

### Python Libraries
Ensure the following libraries are installed:
- `pandas`
- `numpy`
- `time`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

### Installation Command
Run the following to install dependencies:
```bash
pip install pandas numpy time matplotlib seaborn scikit-learn xgboost
```

---

## Acknowledgments
This project was developed as part of the ICS474: Big Data Analytics course project, Term 241.
- **Student 1**: Yousef Buali (ID: 202025400)
- **Student 2**: Mohammed Alnasser  (ID: 202034520)
- **Student 3**:  Ali Al-Haddad   (ID: 202170070)


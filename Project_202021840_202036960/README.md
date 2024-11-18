
# Energy Consumption Prediction in London

## Project Objectives
The purpose of this project is to explore the **Smart Meters in London** dataset and develop a predictive model for energy consumption in London for a given day. This involves cleaning the data, analyzing key features, and using machine learning models to make predictions.

---

## Summary of the Dataset Used 

### 1-Dataset Overview

The dataset contains a reorganized version of data sourced from the London Data Store, showcasing energy consumption measurements for 5,567 London households involved in the UK Power Networks' Low Carbon London initiative from November 2011 to February 2014. The data specifically relates to electricity consumption recorded by smart meters. 
The dataset helps analyze electricity consumption patterns at household level to identify peak usage times, optimize energy distribution, and reduce carbon emissions. 

### `Informations_households.csv`
- **ACORN Group**: Socio-economic classification of households.
- **Tariff Type**: Pricing plan for electricity.
- **Block File**: Specifies the file containing household electricity consumption data.

### `halfhourly_dataset.zip`
- **Smart Meter Readings**: Half-hourly electricity consumption readings for each household, capturing usage in 30-minute intervals.

### `daily_dataset.zip`
- Contains daily electricity usage statistics, including:
  - Number of measurements
  - Minimum, maximum, and mean consumption
  - Median consumption
  - Total daily consumption
  - Standard deviation of consumption

### `acorn_details.csv`
- Contains descriptive attributes of each ACORN group, with a comparative index against national averages.

### `weather_daily_darksky.csv` and `weather_hourly_darksky.csv`
- Provides daily and hourly weather data, including temperature, humidity, precipitation, and more, retrieved from the Dark Sky API.

---

## Main Code Files and Their Purposes




- [**`ICS474 Project Notebook- 202021840- 202036960 .ipynb`**](https://github.com/muzammilbehzad/Students-Projects-ICS474-Big-Data-Analytics-Fall-2024/tree/main/Project_202021840_202036960/Project.ipynb): Contain all codes, figures, results for the project
- [**`ICS474 Project Report- 202021840- 202036960.pdf`**](https://github.com/muzammilbehzad/Students-Projects-ICS474-Big-Data-Analytics-Fall-2024/tree/main/Project_202021840_202036960/ICS474%20Project%20Report-%20202021840-%20202036960.pdf): This is a report that has the procedure followed in this project with the found results and some figures
- [**`ICS474 Project Presentation- 202021840- 202036960.pptx`**](https://github.com/muzammilbehzad/Students-Projects-ICS474-Big-Data-Analytics-Fall-2024/tree/main/Project_202021840_202036960/ICS474%20Project%20Presentation-%20202021840-%20202036960.pptx): This is the presentation of the project that was conducted in class on Monday 11/18/2024
- [**`Acron analysis.ipynb`**](https://github.com/muzammilbehzad/Students-Projects-ICS474-Big-Data-Analytics-Fall-2024/tree/main/Project_202021840_202036960/Acron%20analysis.ipynb): This is an extra analysis we did for the Acron data. It is added since there are some intresting resaults that were found. (We asked Dr.Muzzamil before uploading this file in class on Monday 11/18/2024) 

---

## Instructions for Running the Code

1. **Download the data by running the first cell in the notebook** 
2. **Install the libraries used using the command in this README file** 
3. **Run Jupyter Notebook**:
   - Follow the sequence of notebook
   - Complete each cell in order to ensure the datasets are properly processed and models are trained.
---

## Required Installations

### Python Libraries
Ensure the following libraries are installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `jupyter`

### Installation Command
Run the following to install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

---

## Acknowledgments
This project was developed as part of the ICS474: Big Data Analytics course project, Term 241.
- **Student 1**: Ziyad Alshamrani (ID: 202021840)
- **Student 2**: Elyas Alhabub (ID: 202036960)

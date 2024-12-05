# Uber Data Analysis 🚗 🚕

Presentation link: https://drive.google.com/file/d/1eQNwGAfFisHY41kZV4wa67WrCBFTBQSO/view?usp=sharing

Uber is finding better ways to move and work. This project focuses on analyzing Uber trip data to gain insights into travel patterns, ride demand, and other trends.

## Project Objectives
The objective of this project is to develop a predictive model that estimates the distance traveled in Uber trips using a selection of three base models. By focusing on key trip-related features within the Uber dataset, this project aims to identify significant factors impacting trip distances, enhancing insights for trip planning and resource optimization. The analysis will emphasize accuracy and interpretability to support practical applications in the ride-sharing sector.

## About the Dataset
The dataset provides detailed information about various Uber rides, including:
- **Columns**:
  - **START_DATE**: The date and time when the trip started.
  - **END_DATE**: The date and time when the trip ended.
  - **CATEGORY**: Type of trip (e.g., Business, Personal).
  - **START**: Starting location of the trip.
  - **STOP**: Destination of the trip.
  - **MILES**: Distance covered in miles during the trip.
  - **PURPOSE**: Purpose of the trip (e.g., Meeting, Errand, Customer Visit).

This data allows us to perform in-depth analysis on travel patterns, trip categories, distance trends, and more.

## Main Code Files and Their Purpose
- **Uber_Data_Analysis.ipynb**: A Google Colab Notebook containing the code for the project, along with brief descriptions of each section in the code.
- **Project_Report.pdf**: A detailed report summarizing the project objectives, data exploration, analysis findings, methodologies, and conclusions drawn from the data.


## Instructions for Running the Code
In the notebook, the dataset is automatically downloaded in the second cell using the following command: <br>**```path = kagglehub.dataset_download("bhanupratapbiswas/uber-data-analysis")```**<br> This command downloads the dataset and assigns the directory path to the variable path.

To load the dataset into a DataFrame, the following line is used:<br>**```uberDataset = pd.read_csv(os.path.join(path, "UberDataset.csv"))```** <br>
Here, os.path.join correctly combines the directory path stored in path with the filename "UberDataset.csv", allowing the dataset to be read into the notebook seamlessly.

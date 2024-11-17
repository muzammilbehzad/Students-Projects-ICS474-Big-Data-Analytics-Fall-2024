# 3D Printer Error Detection: Nozzle Extrusion Classification

This project is focused on detecting 3D printing errors by classifying nozzle images to determine whether extrusion occurred or not. The **Attention-based Residual Convolutional Neural Network (ARCNN)** model is used for image classification. The goal is to predict potential 3D printing failures, such as extrusion issues, based on the nozzle images.

## Project Objectives

- **Classify Extrusion Occurrence**: Predict whether extrusion occurred based on images of the 3D printer nozzle.
- **Model Development**: Implement and train the **ARCNN (Attention-based Residual Convolutional Neural Network)** to classify nozzle images as either "extrusion occurred" or "no extrusion."
- **Data Preprocessing**: Clean and prepare the image dataset, including resizing, normalization, and augmentation if necessary.
- **Model Training & Evaluation**: Train the ARCNN model on the preprocessed dataset and evaluate its performance using metrics such as accuracy, precision, recall, and F1 score.
- **Prediction & Visualization**: Make predictions on new nozzle images and visualize model performance with metrics and confusion matrices.


## Dataset
The dataset can be found here: https://www.kaggle.com/competitions/early-detection-of-3d-printing-issues/
It consistes of around 12Gb of images from a nozzle camera with some having extrusion and others none.

## Main Code File

- **`ARCNN.ipynb`**: This Jupyter notebook contains all the necessary code for preprocessing, model training, evaluation, and prediction. It is structured as follows:

    1. **Data Preprocessing**: 
        - Load and clean the dataset
        - Resize and normalize the images
        - Split the data into training and test sets
    2. **Model Definition (ARCNN)**: 
        - Define the **ARCNN model**, an attention-based residual convolutional neural network designed for improved feature extraction and classification.
    3. **Model Training**: 
        - Train the ARCNN model on the preprocessed image data
        - Adjust hyperparameters and fit the model to the training data
        - Visualize model performance, including accuracy/loss curves for the training phase.
    4. **Model Evaluation**: 
        - Evaluate the trained model on the test set
        - Compute key metrics such as accuracy
    5. **Prediction**: 
        - Use the trained model to predict whether extrusion occurred or not in new nozzle images
  

## Instructions for Running the Code

### 1. Clone the Repository

```bash
git clone https://github.com/MysticalDawn/3D-Printer-Error-Detection-Final.git
cd 3D-Printer-Error-Detection-Final
```

### 2. Access the Folder
```bash
cd 3D-Printer-Error-Detection-Final
```

## The outputs are already shown in the Juypter Notebook. We decided not to include the dataset itself since it is quite large. 

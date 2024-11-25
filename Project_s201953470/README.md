# Dog Breed Classification with MobileNetV2

This project uses a **MobileNetV2** convolutional neural network to classify images of dogs into one of 120 breeds. The model is trained using TensorFlow and Keras and achieves high accuracy on the test set.

## Project Overview

This project includes data preprocessing, model building, and evaluation for classifying dog breeds based on images. The dataset consists of images from 120 different dog breeds. The notebook covers data loading, preprocessing, model training, and evaluation to achieve accurate predictions.

## Libraries and Packages Used

- **TensorFlow/Keras**: Machine learning framework for building and training the model.  
- **matplotlib**: Data visualization library for visualizing model performance and data distribution.  
- **numpy**: Numerical computing library used for handling image data and arrays.  

## Installation

1. Ensure you have **Python 3.8** or later installed.  

2. Clone this repository and navigate to the project directory:  
   ```bash
   git clone https://github.com/yourusername/dog-breed-classification.git
   cd dog-breed-classification
   ```

3. Install the required packages:  
   ```bash
   pip install tensorflow matplotlib numpy
   ```

4. Run the notebook:  
   ```bash
   jupyter notebook
   ```  
   Open the `.ipynb` file in Jupyter Notebook and run the cells sequentially.  

## Usage

The notebook is divided into several sections:

1. **Data Loading and Preprocessing**:  
   - Loads the dog image dataset.  
   - Scales the pixel values and encodes breed labels.  
   - Splits data into training and validation sets.  

2. **Exploratory Data Analysis (EDA)**:  
   - Visualizes the distribution of images across the different dog breeds.  
   - Displays sample images to understand dataset diversity.  

3. **Model Building and Training**:  
   - Uses the MobileNetV2 architecture to build a CNN model.  
   - Trains the model using the processed image data.  

4. **Model Evaluation**:  
   - Evaluates the model on the test set.  
   - Achieves a test accuracy of **93%** and a training accuracy of **99.89%**.
  
## Presentation Link
[Go to Presentation](https://github.com/muzammilbehzad/Students-Projects-ICS474-Big-Data-Analytics-Fall-2024/blob/main/Project_s201953470/ICS474%20-%20Project%20Presentation%20-%20Jawad%20Almuttawa%20-%20201953470.pptx)

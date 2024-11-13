
# Stock Market Prediction with LSTM - ICS 474 Project

This project uses an LSTM model to predict stock prices based on historical stock market data, leveraging TensorFlow and other data science libraries. The model was trained with **TensorFlow's Metal plugin** to take advantage of GPU acceleration on Apple Silicon. However, users without this setup can use standard **TensorFlow and Keras**, and the code will run without issue.

## Project Overview

This project includes data preprocessing, exploratory data analysis, and an LSTM model to predict stock closing prices. The notebook performs data visualization, data splitting, feature scaling, model training, and evaluation to generate accurate predictions on the test dataset.

### Libraries and Packages Used

- **TensorFlow/Keras**: Machine learning framework (using TensorFlow Metal for GPU acceleration, optional)
- **pandas**: Data manipulation
- **pandas_datareader**: Financial data retrieval
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Enhanced data visualizations
- **yfinance** (version 0.2.4): For fetching stock data (using this specific version to avoid conflicts with pandas_datareader)

### Installation

Ensure that you have **Python 3.8 or later** installed.

1. **Clone this repository** and navigate into the project directory.

2. **Install the required packages**:

   ```bash
   pip install tensorflow pandas pandas-datareader numpy matplotlib seaborn yfinance==0.2.4
   ```

   > **Note**: TensorFlow Metal is optional and only needed for Apple Silicon GPUs. Standard TensorFlow will work for other setups.

3. **Run the notebook**:

   ```bash
   jupyter notebook
   ```

   Open the `.ipynb` file in the Jupyter Notebook environment and run the cells sequentially.

### Usage

The notebook is divided into several sections:

1. **Data Loading and Preprocessing**: Loads stock market data from Yahoo Finance and performs scaling and splitting.
2. **Exploratory Data Analysis (EDA)**: Visualizes data distribution and trends.
3. **Model Building and Training**: Uses LSTM to train on historical data, predicting future prices.
4. **Predictions**: Computes and displays performance metrics.

---

### Troubleshooting

If you encounter issues with `pandas_datareader` compatibility with `yfinance`, ensure that you are using **yfinance version 0.2.4**.

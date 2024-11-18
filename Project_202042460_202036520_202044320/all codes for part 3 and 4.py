# note: i did this in vscode one by one, to see each output 
# in clear way see the word report file for part 2 and 3

############################################################################################################################################################

# part 2 question 9

import dask.dataframe as dd

# Load the CSV file with Dask
file_path = "C:/credit_card_transactions.csv"  # Use your updated file path
data = dd.read_csv(file_path)

# Calculate the percentage of missing values per column
total_count = len(data)
missing_data = data.isnull().sum() / total_count * 100

# Show missing data percentages
print("Percentage of missing values per column:")
print(missing_data.compute())

# Calculate the mode of merch_zipcode, excluding missing values
merch_zipcode_mode = data['merch_zipcode'].mode().compute()[0]

# Fill missing values in merch_zipcode with the mode
data = data.fillna({"merch_zipcode": merch_zipcode_mode})

# Verify that there are no more missing values in merch_zipcode
print("Percentage of missing values after imputation:")
print(data.isnull().sum().compute() / len(data) * 100)


############################################################################################################################################################

#part 2 question 10

import dask.dataframe as dd

# Load the CSV file with Dask (if not already loaded)
file_path = "C:/credit_card_transactions.csv"
data = dd.read_csv(file_path)

# Convert 'gender' and 'state' to categorical and use categorize() for Dask compatibility
data = data.categorize(columns=['gender', 'state'])

# One-hot encode 'gender' and 'state'
data = dd.get_dummies(data, columns=['gender', 'state'])

# Convert 'merchant', 'category', 'job' to categorical for label encoding
data = data.categorize(columns=['merchant', 'category', 'job'])

# Map categories to codes for 'merchant', 'category', 'job'
data['merchant_encoded'] = data['merchant'].cat.codes
data['category_encoded'] = data['category'].cat.codes
data['job_encoded'] = data['job'].cat.codes

# Drop original columns if needed after encoding
data = data.drop(['merchant', 'category', 'job'], axis=1)

# Print the first few rows to verify
print(data.head())


############################################################################################################################################################

#part 2 question 11

import dask.dataframe as dd
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file with Dask
file_path = "C:/credit_card_transactions.csv"
data = dd.read_csv(file_path)

# Define numerical columns based on inspection
numerical_columns = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']

# Convert numerical columns to Pandas for scaling
numerical_data_pd = data[numerical_columns].compute()  # Convert to Pandas for in-memory operation

# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the data with the scaler
scaled_values = scaler.fit_transform(numerical_data_pd)

# Create a Pandas DataFrame with scaled values and original column names
scaled_data_pd = pd.DataFrame(scaled_values, columns=numerical_columns)

# Convert back to Dask DataFrame if you need to continue with Dask operations
scaled_data_dd = dd.from_pandas(scaled_data_pd, npartitions=data.npartitions)

# Print the first few rows to verify the scaled data
print("Scaled data:")
print(scaled_data_dd.head())

############################################################################################################################################################

#part 2 question 12

import dask.dataframe as dd
import pandas as pd

# Load the CSV file with Dask
file_path = "C:/credit_card_transactions.csv"
data = dd.read_csv(file_path)

# Convert to Pandas to calculate correlations
data_pd = data.compute()

# Keep only numeric columns for correlation calculation
numeric_data = data_pd.select_dtypes(include=['float64', 'int64'])

# Calculate correlation with the target variable 'is_fraud'
correlation = numeric_data.corr()['is_fraud'].sort_values(ascending=False)

# Select features with significant correlation (e.g., absolute correlation > 0.1)
significant_features = correlation[abs(correlation) > 0.1].index.tolist()

# Drop the target from the list to keep only features
significant_features.remove('is_fraud')

# Filter data to include only significant features
selected_data = data[significant_features]

# Print the selected features
print("Selected features based on correlation with target:")
print(significant_features)

############################################################################################################################################################

#part 3 question 13

import dask.dataframe as dd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the CSV file with Dask
file_path = "C:/credit_card_transactions.csv"
data = dd.read_csv(file_path)

# Convert Dask DataFrame to Pandas for feature selection
data_pd = data.compute()

# Step 1: Feature Selection
# Keep only numeric columns for correlation calculation
numeric_data = data_pd.select_dtypes(include=['float64', 'int64'])

# Calculate correlation with the target variable 'is_fraud'
correlation = numeric_data.corr()['is_fraud'].sort_values(ascending=False)

# Select features with significant correlation (absolute correlation > 0.1)
significant_features = correlation[abs(correlation) > 0.1].index.tolist()
significant_features.remove('is_fraud')  # Exclude the target variable

# Step 2: Prepare Data for Model Training
# Define features (X) and target (y)
X = data_pd[significant_features]  # Use the selected significant features
y = data_pd['is_fraud']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Make Predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)

############################################################################################################################################################

#part 3 questions: 14 and 15 and 16

import dask.dataframe as dd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load the CSV file with Dask and convert to Pandas for processing
file_path = "C:/credit_card_transactions.csv"
data = dd.read_csv(file_path)
data_pd = data.compute()  # Convert to Pandas DataFrame

# Assuming significant_features is defined as per the previous feature selection step
significant_features = ['amt']  # Update with actual significant features list if needed
X = data_pd[significant_features]  # Features selected from the previous step
y = data_pd['is_fraud']  # Target variable

# Data Splitting
# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model Training
# Initialize and train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy and display classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

############################################################################################################################################################

#part 3 question 17

# Load the CSV file with Dask and convert to Pandas for processing
file_path = "C:/credit_card_transactions.csv"
data = dd.read_csv(file_path)
data_pd = data.compute()  # Convert to a Pandas DataFrame

# Assuming significant_features is defined as per the previous feature selection step
significant_features = ['amt']  # Update with actual significant features list if needed
X = data_pd[significant_features]  # Features selected from the previous step
y = data_pd['is_fraud']  # Target variable

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model Training with basic RandomForest parameters
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation on test set
y_pred = model.predict(X_test)

# Calculate accuracy and display classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

############################################################################################################################################################

#part 3 question 18

import dask.dataframe as dd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint

# Load the CSV file with Dask and convert to Pandas for processing
file_path = "C:/credit_card_transactions.csv"
data = dd.read_csv(file_path)
data_pd = data.compute()  # Convert to a Pandas DataFrame

# Assuming significant_features is defined as per the previous feature selection step
significant_features = ['amt']  # Update with actual significant features list if needed
X = data_pd[significant_features]  # Features selected from previous step
y = data_pd['is_fraud']  # Target variable

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the parameter grid for hyperparameter tuning
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
}

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    model, param_distributions=param_dist, n_iter=10, cv=3, scoring='f1', n_jobs=-1, random_state=42
)

# Fit the random search model
random_search.fit(X_train, y_train)

# Best model from hyperparameter tuning
best_model = random_search.best_estimator_

# Model Evaluation on test set
y_pred = best_model.predict(X_test)

# Calculate accuracy and display classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


############################################################################################################################################################

#part 3 question 19

import dask.dataframe as dd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the CSV file with Dask and convert to Pandas for processing
file_path = "C:/credit_card_transactions.csv"
data = dd.read_csv(file_path)
data_pd = data.compute()  # Convert to a Pandas DataFrame

# Assuming significant_features is defined as per the previous feature selection step
significant_features = ['amt']  # Update with actual significant features list if needed
X = data_pd[significant_features]  # Features selected from previous step
y = data_pd['is_fraud']  # Target variable

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)

# Perform k-fold cross-validation
k = 5  # Number of folds
cross_val_scores = cross_val_score(model, X, y, cv=k, scoring="accuracy")

# Print cross-validation scores and average accuracy
print(f"Cross-validation scores for {k} folds: {cross_val_scores}")
print(f"Average cross-validation accuracy: {np.mean(cross_val_scores)}")

# Training the model on the full training data for final evaluation
model.fit(X, y)
y_pred = model.predict(X)

# Final evaluation metrics
accuracy = accuracy_score(y, y_pred)
print(f"Final Model Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y, y_pred))


############################################################################################################################################################

#part 3 question 20

import dask.dataframe as dd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

# Load the CSV file with Dask and convert to Pandas for processing
file_path = "C:/credit_card_transactions.csv"
data = dd.read_csv(file_path)
data_pd = data.compute()  # Convert to a Pandas DataFrame

# Assuming significant_features is defined as per the previous feature selection step
significant_features = ['amt']  # Update with actual significant features list if needed
X = data_pd[significant_features]  # Features selected from previous step
y = data_pd['is_fraud']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
}

# Evaluate each model
for model_name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{model_name} - Cross-validation accuracy: {cv_scores.mean():.4f}")

    # Fit the model and evaluate on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} - Test Accuracy: {accuracy:.4f}")
    print(f"{model_name} - Classification Report:\n{classification_report(y_test, y_pred)}\n")



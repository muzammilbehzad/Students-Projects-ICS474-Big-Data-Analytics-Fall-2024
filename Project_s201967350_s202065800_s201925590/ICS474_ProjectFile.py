#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv('credit_card_transactions01.csv',delimiter = ',')
display(df.head())
print(df.head())
print("Dataset Info:")
print(df.info())
### Dataset Overview:
# The Credit Card Transactions Dataset is a comprehensive collection 
# of credit card transaction records designed for fraud detection analysis. 
# While it's a simulated dataset, it's structured to mirror real-world credit card transaction patterns and fraud scenarios.
###

# The dataset simulates real credit card transactions with both legitimate and fraudulent cases
# The dataset contains detailed transaction information including temporal, geographical, and demographic data
# The dataset's specifically designed for developing fraud detection models


# In[2]:


#### 1) the first thing we do is to remove columns that doesn't affect our model in fraud detection

## Columns to REMOVE: ###
# 1. Unnamed: 0 - This is likely just an index column with no predictive value
# 2. first - Personal names shouldn't influence fraud detection
# 3. last - Personal names shouldn't influence fraud detection
# 4. street - Specific street addresses are too granular and could lead to overfitting
# 5. city - The city_pop and lat/long provide better geographical indicators
# 6. state - Already represented by geographical coordinates
# 7. zip - Merchant zipcode provides sufficient location information
# 8. dob - Age might be relevant, but you can calculate it from dob if needed
# 9. trans_num - Transaction ID has no predictive value
# 10. unix_time - Already have trans_date_trans_time in a more usable format


# List of columns to remove
columns_to_drop = ['Unnamed: 0', 'first', 'last', 'street', 'city', 'state', 
                   'zip', 'dob', 'trans_num', 'unix_time']

# Drop the columns and create new dataframe
df_cleaned = df.drop(columns=columns_to_drop)

df_cleaned.head()


# In[3]:


# convert trans_date_trans_time to a proper value/format

# Before conversion (current format)
print("Before conversion:")
print(df_cleaned['trans_date_trans_time'].head())

# Convert to datetime
df_cleaned['trans_date_trans_time'] = pd.to_datetime(df_cleaned['trans_date_trans_time'])

# After conversion
print("\nAfter conversion:")
print(df_cleaned['trans_date_trans_time'].head())

# You can also see the data type has changed
print("\nData type:")
print(df_cleaned['trans_date_trans_time'].dtype)

# convert cc_num into a proper format
df_cleaned['cc_num'] = df_cleaned['cc_num'].astype('int64').astype(str)

df_cleaned.head()


# In[4]:


### Feature Description: ####

## 1.Numerical Features:
# amt: Transaction amount (Float)
# lat/long: Customer's geographical coordinates (Float)
# merch_lat/merch_long: Merchant's geographical coordinates (Float)
# city_pop: Population of the transaction city (Integer)

## 2.Categorical Features:
# merchant: Name of the business (String)
# category: Type of transaction (String)
# gender: Cardholder's gender (String)
# job: Cardholder's occupation (String)
# merch_zipcode: Merchant's postal code (String)

## 3.Temporal Features:
# trans_date_trans_time: Transaction timestamp (Datetime)

## 4.Identifier Features:
# cc_num: Credit card number (String)

## 5.Target Variable:
# is_fraud: Binary indicator (0 = legitimate, 1 = fraudulent)



# Display info about the dataset
print("Dataset Info:")
print(df_cleaned.info())

# Display feature types
print("\nFeature Types:")
print(df_cleaned.dtypes)

df_cleaned.describe()


# In[5]:


## checking & handeling for missing values

# Check for missing values
print("Missing Values in Each Column:")
print(df_cleaned.isnull().sum())

# Calculate percentage of missing values
print("\nPercentage of Missing Values:")
print((df_cleaned.isnull().sum() / len(df_cleaned)) * 100)


# In[6]:


# Handle merch_zipcode (convert to string and handle missing values)
df_cleaned['missing_zipcode_flag'] = df_cleaned['merch_zipcode'].isna().astype(int)
df_cleaned['merch_zipcode'] = df_cleaned['merch_zipcode'].fillna('Unknown')
df_cleaned['merch_zipcode'] = df_cleaned['merch_zipcode'].astype(str).replace('nan', 'Unknown')

# Verify the changes
print("Updated Data Types after fixing merch_zipcode:")
print(df_cleaned.dtypes)

# Verify binary nature of is_fraud
print("\nUnique values in is_fraud:")
print(df_cleaned['is_fraud'].unique())

# Verify missing_zipcode_flag
print("\nDistribution of missing_zipcode_flag:")
print(df_cleaned['missing_zipcode_flag'].value_counts())


# In[7]:


# Check for missing values
print("Missing Values in Each Column:")
print(df_cleaned.isnull().sum())

# Calculate percentage of missing values
print("\nPercentage of Missing Values:")
print((df_cleaned.isnull().sum() / len(df_cleaned)) * 100)


# In[8]:


print(df_cleaned.head())
df_cleaned.head()


# In[9]:


# now we need to check for duplicate: (same card, time and amount)

# 1. Check for completely duplicate rows (all columns identical)
complete_duplicates = df_cleaned.duplicated().sum()
print("Complete duplicate rows:", complete_duplicates)

# 2. Check for suspicious transaction duplicates
# Same card, same amount, same timestamp (potential fraud or error)
suspicious_duplicates = df_cleaned.duplicated(
    subset=['cc_num', 'amt', 'trans_date_trans_time'], 
    keep='first'
).sum()
print("\nSuspicious duplicates (same card, amount, and timestamp):", suspicious_duplicates)

# 3. Check for transactions per credit card
transactions_per_card = df_cleaned['cc_num'].value_counts()
print("\nTransactions per credit card:")
print("Min transactions:", transactions_per_card.min())
print("Max transactions:", transactions_per_card.max())
print("Mean transactions:", transactions_per_card.mean())

# 4. Check cards with unusually high number of transactions
print("\nCards with highest number of transactions:")
print(transactions_per_card.head())


# In[10]:


display(df_cleaned.head())
def perform_statistical_analysis(df_cleaned):
    """
    Perform comprehensive statistical analysis on the credit card transactions dataset
    """
    # Numerical columns for analysis
    numerical_cols = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
    
    # Basic statistics for numerical columns
    numerical_stats = df[numerical_cols].describe()
    
    # Transaction amount analysis
    amount_stats = {
        'Total Transaction Volume': df['amt'].sum(),
        'Average Transaction Amount': df['amt'].mean(),
        'Median Transaction Amount': df['amt'].median(),
        'Transaction Amount Std Dev': df['amt'].std(),
        'Max Transaction Amount': df['amt'].max(),
        'Min Transaction Amount': df['amt'].min()
    }
    
    # Categorical analysis
    categorical_stats = {
        'Transaction Categories': df['category'].value_counts(),
        'Gender Distribution': df['gender'].value_counts(),
        'Top 10 Jobs': df['job'].value_counts().head(10),
        'Fraud Distribution': df['is_fraud'].value_counts(normalize=True) * 100
    }
    
    # Geographical statistics
    geo_stats = {
        'Unique Merchant Locations': len(df[['merch_lat', 'merch_long']].drop_duplicates()),
        'Average City Population': df['city_pop'].mean(),
        'Median City Population': df['city_pop'].median(),
        'Max City Population': df['city_pop'].max(),
        'Min City Population': df['city_pop'].min()
    }
    
    return numerical_stats, amount_stats, categorical_stats, geo_stats

# Perform analysis
numerical_stats, amount_stats, categorical_stats, geo_stats = perform_statistical_analysis(df)

# Print results
print("\n=== Numerical Features Statistics ===")
print(numerical_stats)

print("\n=== Transaction Amount Analysis ===")
for key, value in amount_stats.items():
    print(f"{key}: {value:,.2f}")

print("\n=== Categorical Features Analysis ===")
print("\nTransaction Categories Distribution:")
print(categorical_stats['Transaction Categories'])
print("\nGender Distribution:")
print(categorical_stats['Gender Distribution'])
print("\nTop 10 Jobs:")
print(categorical_stats['Top 10 Jobs'])
print("\nFraud Distribution (%):")
print(categorical_stats['Fraud Distribution'])

print("\n=== Geographical Statistics ===")
for key, value in geo_stats.items():
    print(f"{key}: {value:,.2f}")
    


# In[11]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def analyze_distributions_and_correlations(df_cleaned):
    """
    Comprehensive analysis of distributions, correlations, and outliers
    """
    # Set style for better visualizations
    plt.style.use('seaborn')
    
    # 1. Distribution Analysis
    fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle('Distribution of Key Numerical Features', fontsize=16)
    
    # Transaction Amount Distribution
    sns.histplot(data=df_cleaned, x='amt', bins=50, ax=axes[0,0])
    axes[0,0].set_title('Transaction Amount Distribution')
    axes[0,0].set_xlabel('Amount')
    
    # City Population Distribution (log scale due to large range)
    sns.histplot(data=df_cleaned, x='city_pop', bins=50, ax=axes[0,1])
    axes[0,1].set_title('City Population Distribution')
    axes[0,1].set_xlabel('City Population')
    axes[0,1].set_xscale('log')
    
    # Box plots for numerical features
    sns.boxplot(data=df_cleaned[['amt', 'city_pop']], ax=axes[1,0])
    axes[1,0].set_title('Box Plots of Amount and City Population')
    axes[1,0].set_yscale('log')
    
    # Category distribution
    category_counts = df_cleaned['category'].value_counts()
    category_counts.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Transaction Categories Distribution')
    axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 2. Correlation Analysis
    fig2, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig2.suptitle('Correlation Analysis', fontsize=16)
    
    # Select numerical columns for correlation
    numerical_cols = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'is_fraud']
    correlation_matrix = df_cleaned[numerical_cols].corr()
    
    # Heatmap of correlations
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0])
    axes[0].set_title('Correlation Matrix')
    
    # Scatter plot of amount vs city_pop, colored by fraud
    sns.scatterplot(data=df_cleaned, x='amt', y='city_pop', hue='is_fraud', alpha=0.5, ax=axes[1])
    axes[1].set_title('Amount vs City Population (by Fraud Status)')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    # 3. Outlier Detection
    fig3, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig3.suptitle('Outlier Analysis', fontsize=16)
    
    # Calculate Z-scores for amount
    z_scores_amt = np.abs(stats.zscore(df_cleaned['amt']))
    outliers_amt = (z_scores_amt > 3).sum()
    
    # Plot amount distribution with outliers highlighted
    sns.histplot(data=df_cleaned, x='amt', bins=50, ax=axes[0])
    axes[0].axvline(df_cleaned['amt'].mean() + 3*df_cleaned['amt'].std(), color='r', linestyle='--')
    axes[0].axvline(df_cleaned['amt'].mean() - 3*df_cleaned['amt'].std(), color='r', linestyle='--')
    axes[0].set_title(f'Transaction Amount Distribution\n(Outliers: {outliers_amt})')
    
    # Box plot with outliers
    sns.boxplot(data=df_cleaned[['amt', 'city_pop']], ax=axes[1])
    axes[1].set_yscale('log')
    axes[1].set_title('Box Plots Showing Outliers')
    
    plt.tight_layout()
    
    return {
        'outliers_amt': outliers_amt,
        'correlation_matrix': correlation_matrix
    }

# Perform analysis
results = analyze_distributions_and_correlations(df_cleaned)

# Print key findings
print("\nKey Findings:")
print(f"1. Number of outliers in transaction amounts (|z-score| > 3): {results['outliers_amt']}")
print("\n2. Key correlations with fraud:")
for feature in ['amt', 'city_pop']:
    correlation = results['correlation_matrix'].loc['is_fraud', feature]
    print(f"   - {feature}: {correlation:.3f}")


# In[12]:


""" 
Data Distribution:

Transaction amounts show a right-skewed distribution, with most transactions being of lower value
City population has a highly skewed distribution with many small cities and few large ones
Transaction categories are relatively well-balanced, with gas_transport and grocery_pos being the most common


Correlation Analysis:

Most features show weak to moderate correlations with each other
Geographic features (lat/long) show expected correlations with their merchant counterparts
Transaction amount has a weak correlation with fraud status
City population shows minimal correlation with other features


Outlier Detection:

Transaction amounts have several outliers, particularly on the high end
City population data contains extreme outliers, representing major metropolitan areas
The outliers appear to be legitimate data points rather than errors, as they follow expected patterns

"""
print()


# In[13]:


display(df_cleaned)
print(df_cleaned) 


# In[14]:


print("Method 1 - Using dtypes:")
print(df_cleaned['trans_date_trans_time'].dtype)


# In[15]:


# First create all new temporal columns before dropping the original
df_cleaned['hour'] = df_cleaned['trans_date_trans_time'].dt.hour
df_cleaned['minute'] = df_cleaned['trans_date_trans_time'].dt.minute
df_cleaned['day_of_week'] = df_cleaned['trans_date_trans_time'].dt.dayofweek
df_cleaned['day_of_month'] = df_cleaned['trans_date_trans_time'].dt.day
df_cleaned['month'] = df_cleaned['trans_date_trans_time'].dt.month

# Drop the original trans_date_trans_time column
df_cleaned = df_cleaned.drop('trans_date_trans_time', axis=1)

# Let's look at the first few rows to verify the changes
print("First few rows of the transformed dataset:")
print(df_cleaned.head())

# Verify the ranges of all new temporal columns
print("\nValue ranges for new temporal features:")
for col in ['hour', 'minute', 'day_of_week', 'day_of_month', 'month']:
    print(f"\n{col}:")
    print(f"Range: {df_cleaned[col].min()} to {df_cleaned[col].max()}")
    print(f"Unique values: {sorted(df_cleaned[col].unique())}")

# Display the new column data types
print("\nData types of new temporal columns:")
print(df_cleaned[['hour', 'minute', 'day_of_week', 'day_of_month', 'month']].dtypes)


# In[16]:


# 1. Gender - Using Label Encoder since it's binary
from sklearn.preprocessing import LabelEncoder
le_gender = LabelEncoder()
df_cleaned['gender_encoded'] = le_gender.fit_transform(df_cleaned['gender'])

# 2. Category - Using One-Hot Encoding
category_dummies = pd.get_dummies(df_cleaned['category'], prefix='category')

# 3. Remove original columns and add encoded ones
df_cleaned = df_cleaned.drop(['gender', 'category'], axis=1)

# 4. Add category dummy columns to the main dataframe
df_cleaned = pd.concat([df_cleaned, category_dummies], axis=1)

# Display first few rows to verify changes
print("\nFirst few rows of transformed dataframe:")
print(df_cleaned.head())

# Verify the shape and new columns
print("\nDataframe shape:", df_cleaned.shape)
print("\nCategory columns added:", list(category_dummies.columns))


# In[17]:


# Check how many rows have Unknown zipcode
print("Number of rows with Unknown zipcode:", (df_cleaned['merch_zipcode'] == 'Unknown').sum())
print("Total rows before removal:", len(df_cleaned))

# Remove rows where merch_zipcode is Unknown
df_cleaned = df_cleaned[df_cleaned['merch_zipcode'] != 'Unknown']

# Convert merch_zipcode to numeric type since all values are now numbers
df_cleaned['merch_zipcode'] = pd.to_numeric(df_cleaned['merch_zipcode'])

print("\nTotal rows after removal:", len(df_cleaned))

# Verify no more Unknown values
print("\nUnique values in merch_zipcode (first 5):", df_cleaned['merch_zipcode'].unique()[:5])

# Display first few rows to verify changes
print("\nFirst few rows after cleaning:")
print(df_cleaned.head())


# In[18]:


# Remove cc_num column
df_cleaned = df_cleaned.drop('cc_num', axis=1)

# Verify the column is removed
print("Remaining columns:", list(df_cleaned.columns))
print("\nShape of dataframe after removing cc_num:", df_cleaned.shape)

# Display first few rows to verify changes
print("\nFirst few rows after removing cc_num:")
print(df_cleaned.head())


# 1. Remove job column
df_cleaned = df_cleaned.drop('job', axis=1)

# 2. Get top 50 most frequent merchants
top_50_merchants = df_cleaned['merchant'].value_counts().nlargest(50).index

# 3. Replace less frequent merchants with 'OTHER'
df_cleaned['merchant'] = df_cleaned['merchant'].apply(lambda x: x if x in top_50_merchants else 'OTHER')

# 4. Create dummy variables
merchant_dummies = pd.get_dummies(df_cleaned['merchant'], prefix='merchant')

# 5. Remove original merchant column and add encoded columns
df_cleaned = df_cleaned.drop('merchant', axis=1)
df_cleaned = pd.concat([df_cleaned, merchant_dummies], axis=1)

# Show information about the encoding
print("Number of merchant columns created:", len(merchant_dummies.columns))
print("\nMerchant columns:", list(merchant_dummies.columns))

# Show final dataset info
print("\nFinal shape of dataset:", df_cleaned.shape)


# Verify all columns are numeric
print("\nData types of all columns:")
print(df_cleaned.dtypes)


# In[19]:



display(df_cleaned)
print(df_cleaned) 


# In[20]:


# We are going to use tree-based classification models for predecting fraud accounts
'''
Problem Type Match


we have a binary classification problem (fraud vs. non-fraud)
Tree-based models like Decision Trees, Random Forests, or XGBoost are well-suited for classification tasks
The target variable is_fraud is clearly defined as a binary outcome


Advantages:
Handle Mixed Data Types: our dataset contains:

Numerical features (amt, lat, long, city_pop)
Categorical features (merchant, category, job, gender)
Tree models can naturally handle both without extensive preprocessing

'''


'''
To be more specific we are going to use Random Forest Classifier

Why:

Robust against overfitting through ensemble learning
Handles high-dimensional data well (we have 80 features)
Provides feature importance rankings
Good with imbalanced datasets when properly configured

'''

'''
The method of splitting we are going to choose is cross calidation
Method Choice: K-Fold Cross-Validation with Stratification

Using 5-fold stratified cross-validation
This means our data will be split into 5 equal parts, maintaining fraud/non-fraud proportions in each fold

Rationale for Choosing Cross-Validation:

More Robust Performance Estimation

Each data point will be used for both training and testing
Gets performance metrics from 5 different train-test combinations
Provides a more reliable estimate of model performance than a single train-test split


Better for Imbalanced Data

Our fraud detection dataset is imbalanced (few fraud cases)
Stratification ensures each fold maintains the same ratio of fraud/non-fraud cases
Reduces the risk of having folds with too few fraud cases


'''


# In[21]:


# Basic information about the dataset
print("Dataset Info:")
print(df_cleaned.info())

# Display first few rows
print("\nFirst few rows:")
print(df_cleaned.head())

# Check data types and number of columns
print("\nData types of columns:")
print(df_cleaned.dtypes)

# Get basic statistics
print("\nBasic statistics:")
print(df_cleaned.describe())

# Check unique values in each column
print("\nUnique values in each column:")
for column in df_cleaned.columns:
    print(f"\n{column}:", df_cleaned[column].nunique())

# Check class distribution (fraud vs non-fraud)
print("\nClass distribution (is_fraud):")
print(df_cleaned['is_fraud'].value_counts(normalize=True))


# In[22]:


'''
The method of splitting we are going to choose is cross calidation
Method Choice: K-Fold Cross-Validation with Stratification

Using 5-fold stratified cross-validation.
This means our data will be split into 5 equal parts, maintaining fraud/non-fraud proportions in each fold

Rationale for Choosing Cross-Validation:

More Robust Performance Estimation
Each data point will be used for both training and testing
Gets performance metrics from 5 different train-test combinations
Provides a more reliable estimate of model performance than a single train-test split


Better for Imbalanced Data

Our fraud detection dataset is imbalanced (few fraud cases)
Stratification ensures each fold maintains the same ratio of fraud/non-fraud cases
Reduces the risk of having folds with too few fraud cases


'''


from sklearn.model_selection import StratifiedKFold

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Prepare data
X = df_cleaned.drop('is_fraud', axis=1)
y = df_cleaned['is_fraud']

'''
Training Details:

Using Random Forest with parameters optimized for imbalanced fraud detection
class_weight='balanced' is crucial given your 0.37% fraud rate
Higher number of trees (200) to better capture rare fraud patterns
No max_depth restriction to allow model to learn complex patterns
'''

from sklearn.ensemble import RandomForestClassifier

# Initialize model with parameters for imbalanced data
rf_model = RandomForestClassifier(
    n_estimators=200,           # More trees for better performance
    max_depth=None,             # Allow full depth for complex fraud patterns
    min_samples_split=2,        # Default value
    min_samples_leaf=1,         # Default value
    class_weight='balanced',    # Critical for handling 0.37% fraud ratio
    random_state=42,
    n_jobs=-1                   # Use all CPU cores
)

# Train and evaluate
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    rf_model.fit(X_train, y_train)
    
    
'''
Model Evaluation Metrics Choice:

Precision: Measures false positives (important for reducing false fraud alerts)
Recall: Measures missed frauds (crucial - we want to catch most frauds)
F1-Score: Balances precision and recall
PR-AUC: Better than ROC-AUC for imbalanced data
'''

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(X, y, model, cv_folds):
    # Lists to store metrics for each fold
    precision_scores = []
    recall_scores = []
    f1_scores = []
    pr_aucs = []
    
    # Perform cross-validation and evaluation
    for fold, (train_idx, val_idx) in enumerate(cv_folds.split(X, y)):
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
        pr_aucs.append(auc(recall, precision))
        
        # Store fold metrics
        fold_report = classification_report(y_val, y_pred, output_dict=True)
        precision_scores.append(fold_report['1']['precision'])
        recall_scores.append(fold_report['1']['recall'])
        f1_scores.append(fold_report['1']['f1-score'])
        
        print(f"\nFold {fold+1} Results:")
        print(classification_report(y_val, y_pred))
    
    # Print average metrics
    print("\nAverage Metrics across folds:")
    print(f"Precision: {np.mean(precision_scores):.3f} (+/- {np.std(precision_scores):.3f})")
    print(f"Recall: {np.mean(recall_scores):.3f} (+/- {np.std(recall_scores):.3f})")
    print(f"F1-score: {np.mean(f1_scores):.3f} (+/- {np.std(f1_scores):.3f})")
    print(f"PR-AUC: {np.mean(pr_aucs):.3f} (+/- {np.std(pr_aucs):.3f})")

# Run evaluation
evaluate_model(X, y, rf_model, skf)


# In[23]:


'''
Performance Analysis of Your Model:

Precision (1.000 ± 0.000)

Perfect precision (1.0) means no false positives
When model predicts fraud, it's always correct
Very good for minimizing false fraud alerts


Recall (0.476 ± 0.244)

Model catches about 48% of actual fraud cases
High variation (±0.244) across folds
Looking at individual folds:

Best: Fold 2 (0.83 or 83% fraud detection)
Worst: Fold 3 (0.17 or 17% fraud detection)




F1-Score (0.608 ± 0.227)

Balanced metric between precision and recall
Moderate score due to lower recall
High variation matches recall variation


PR-AUC (0.965 ± 0.043)

Very good overall performance (close to 1.0)
Most stable metric across folds
Indicates good ranking of fraud probabilities



Key Observations:

Model is very conservative (high precision, lower recall)
Performance varies significantly between folds
Overall accuracy looks inflated due to imbalanced data

Suggestions for Improvement:

Could adjust class weights to improve recall
Consider using SMOTE or other sampling techniques
Might try different threshold for classification

'''


# In[24]:


get_ipython().system('pip install imbalanced-learn')
'''
Model Improvement Strategy

Problem Identification
Initial model had good precision (100%) but poor recall (47.6%)
High class imbalance (only 0.37% fraud cases)
Performance varied significantly between folds
Need to catch more fraud cases while maintaining precision

Solution Approach
Used SMOTE (Synthetic Minority Over-sampling Technique) to balance training data
Modified Random Forest parameters for better performance
Combined multiple techniques to handle imbalanced data
Kept validation data in original distribution for realistic evaluation

Key Improvements Made

Data Level (SMOTE):
Created synthetic fraud cases in training data only
Balanced the class distribution for better learning
Maintained original validation data to test real-world performance

Model Level (Random Forest):
Increased number of trees to 300 for better learning
Removed depth restrictions to capture complex fraud patterns
Added class weights to further handle imbalance
Utilized all CPU cores for efficient training

Results

Improved Metrics:
Recall increased from 47.6% to 81.9% (catching more fraud)
Precision remained high at 96.7% (few false alarms)
F1-score improved from 0.608 to 0.879 (better overall)
More consistent performance across folds

Feature Insights:
Transaction amount most important (26.6%)
City population second most important (11.1%)
Geographic features (lat/long) also significant

Why It Worked
SMOTE provided better examples of fraud patterns
More trees captured complex relationships
No depth restriction allowed detailed pattern learning
Combined approaches (SMOTE + class weights) handled imbalance effectively
Original validation data ensured realistic performance measurement
'''

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import classification_report

# 1. Initialize SMOTE
smote = SMOTE(random_state=42)

# 2. Initialize improved Random Forest with focused parameters
improved_rf = RandomForestClassifier(
    n_estimators=300,          # Increased number of trees
    max_depth=None,            # Allow full depth
    class_weight='balanced',   # Handle imbalance
    random_state=42,
    n_jobs=-1
)

# 3. Train and evaluate with SMOTE
precision_scores = []
recall_scores = []
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Apply SMOTE to training data only
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Train model on balanced data
    improved_rf.fit(X_train_balanced, y_train_balanced)
    
    # Predict on validation set (original distribution)
    y_pred = improved_rf.predict(X_val)
    
    # Print fold results
    print(f"\nFold {fold+1} Results:")
    print(classification_report(y_val, y_pred))
    
    # Store metrics
    fold_report = classification_report(y_val, y_pred, output_dict=True)
    precision_scores.append(fold_report['1']['precision'])
    recall_scores.append(fold_report['1']['recall'])
    f1_scores.append(fold_report['1']['f1-score'])

# Print average metrics
print("\nImproved Model - Average Metrics:")
print(f"Precision: {np.mean(precision_scores):.3f} (+/- {np.std(precision_scores):.3f})")
print(f"Recall: {np.mean(recall_scores):.3f} (+/- {np.std(recall_scores):.3f})")
print(f"F1-score: {np.mean(f1_scores):.3f} (+/- {np.std(f1_scores):.3f})")

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': improved_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))


# In[25]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Select numerical features
numerical_features = ['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long']

# Create subplot figure
fig, axes = plt.subplots(len(numerical_features), 2, figsize=(15, 20))
fig.suptitle('Distribution of Numerical Features', fontsize=16, y=1.02)

# Plot histograms and boxplots for each feature
for idx, feature in enumerate(numerical_features):
    # Histogram with KDE
    sns.histplot(data=df_cleaned, x=feature, kde=True, ax=axes[idx, 0])
    axes[idx, 0].set_title(f'Histogram of {feature}')
    axes[idx, 0].set_xlabel(feature)
    axes[idx, 0].set_ylabel('Count')
    
    # Boxplot
    sns.boxplot(data=df_cleaned, x=feature, ax=axes[idx, 1])
    axes[idx, 1].set_title(f'Boxplot of {feature}')
    axes[idx, 1].set_xlabel(feature)
    
    # Calculate and display outlier percentage
    Q1 = df_cleaned[feature].quantile(0.25)
    Q3 = df_cleaned[feature].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_cleaned[(df_cleaned[feature] < (Q1 - 1.5 * IQR)) | 
                         (df_cleaned[feature] > (Q3 + 1.5 * IQR))][feature].count()
    outlier_percentage = (outliers / len(df_cleaned)) * 100
    
    # Add outlier information to plot
    axes[idx, 1].text(0.02, 0.98, f'Outliers: {outliers} ({outlier_percentage:.2f}%)',
                     transform=axes[idx, 1].transAxes, 
                     verticalalignment='top')

plt.tight_layout()
plt.savefig('numerical_distributions.png', bbox_inches='tight', dpi=300)
plt.show()

# Print summary statistics
for feature in numerical_features:
    print(f"\nSummary statistics for {feature}:")
    print(df_cleaned[feature].describe())


# In[26]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Select categorical features
categorical_features = [
    'hour', 'day_of_week', 'day_of_month', 'month', 'gender_encoded',
    'category_entertainment', 'category_food_dining', 'category_gas_transport',
    'category_grocery_net', 'category_grocery_pos', 'category_health_fitness',
    'category_home', 'category_kids_pets', 'category_misc_net',
    'category_misc_pos', 'category_personal_care', 'category_shopping_net',
    'category_shopping_pos', 'category_travel'
]

# Create subplots for categorical features
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
fig.suptitle('Distribution of Categorical Features', fontsize=16, y=1.02)

# Flatten axes for easier iteration
axes = axes.flatten()

# Plot count plots for each feature
for idx, feature in enumerate(categorical_features):
    # Create countplot
    sns.countplot(data=df_cleaned, x=feature, ax=axes[idx])
    
    # Customize the plot
    axes[idx].set_title(f'Distribution of {feature}')
    axes[idx].tick_params(axis='x', rotation=45)
    
    # Add value counts as percentages
    total = len(df_cleaned[feature])
    for p in axes[idx].patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        axes[idx].annotate(percentage, 
                          (p.get_x() + p.get_width()/2., p.get_height()),
                          ha='center', va='bottom')
    
    # Print value counts
    print(f"\nValue counts for {feature}:")
    print(df_cleaned[feature].value_counts(normalize=True).multiply(100))

# Remove empty subplots if any
if len(categorical_features) < 20:
    for idx in range(len(categorical_features), 20):
        fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('categorical_distributions.png', bbox_inches='tight', dpi=300)
plt.show()

# Additional temporal pattern analysis
plt.figure(figsize=(15, 6))
sns.countplot(data=df_cleaned, x='hour', hue='is_fraud')
plt.title('Transaction Count by Hour (with Fraud Label)')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=0)
plt.legend(title='Is Fraud', labels=['Legitimate', 'Fraudulent'])
plt.tight_layout()
plt.savefig('hourly_fraud_distribution.png', bbox_inches='tight', dpi=300)
plt.show()

# Day of week analysis
plt.figure(figsize=(12, 6))
sns.countplot(data=df_cleaned, x='day_of_week', hue='is_fraud')
plt.title('Transaction Count by Day of Week (with Fraud Label)')
plt.xlabel('Day of Week (0=Sunday, 6=Saturday)')
plt.ylabel('Number of Transactions')
plt.legend(title='Is Fraud', labels=['Legitimate', 'Fraudulent'])
plt.tight_layout()
plt.savefig('daily_fraud_distribution.png', bbox_inches='tight', dpi=300)
plt.show()


# In[27]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import KBinsDiscretizer

def analyze_model_performance_across_features(model, X, y):
    """
    Analyze and visualize model performance across different feature ranges
    """
    # Get predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Create figure with subplots
    plt.figure(figsize=(20, 15))
    
    # 1. Performance across transaction amounts
    plt.subplot(2, 2, 1)
    analyze_numeric_feature_performance('amt', X, y, y_pred, y_pred_proba)
    
    # 2. Performance across hours
    plt.subplot(2, 2, 2)
    analyze_categorical_feature_performance('hour', X, y, y_pred)
    
    # 3. Performance across locations
    plt.subplot(2, 2, 3)
    analyze_geographic_performance(X, y, y_pred)
    
    # 4. Feature importance impact
    plt.subplot(2, 2, 4)
    plot_feature_importance_impact(model, X.columns)
    
    plt.tight_layout()
    plt.savefig('model_performance_analysis.png', bbox_inches='tight', dpi=300)
    plt.show()

def analyze_numeric_feature_performance(feature, X, y, y_pred, y_pred_proba):
    """
    Analyze model performance across different ranges of a numeric feature
    """
    # Create bins for the feature
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    bins = discretizer.fit_transform(X[[feature]]).flatten()
    
    # Calculate metrics for each bin
    bin_metrics = []
    for bin_idx in range(10):
        mask = (bins == bin_idx)
        if mask.sum() > 0:  # Check if bin has samples
            precision = precision_score(y[mask], y_pred[mask], zero_division=0)
            recall = recall_score(y[mask], y_pred[mask], zero_division=0)
            avg_proba = y_pred_proba[mask].mean()
            bin_metrics.append({
                'bin': bin_idx,
                'precision': precision,
                'recall': recall,
                'avg_proba': avg_proba,
                'count': mask.sum()
            })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(bin_metrics)
    
    # Plot metrics
    plt.plot(metrics_df['bin'], metrics_df['precision'], marker='o', label='Precision')
    plt.plot(metrics_df['bin'], metrics_df['recall'], marker='s', label='Recall')
    plt.plot(metrics_df['bin'], metrics_df['avg_proba'], marker='^', label='Avg Probability')
    
    plt.title(f'Model Performance Across {feature} Ranges')
    plt.xlabel(f'{feature} Bins (Low to High)')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

def analyze_categorical_feature_performance(feature, X, y, y_pred):
    """
    Analyze model performance across categories
    """
    # Calculate performance metrics for each category
    categories = sorted(X[feature].unique())
    precisions = []
    recalls = []
    counts = []
    
    for cat in categories:
        mask = (X[feature] == cat)
        if mask.sum() > 0:
            precisions.append(precision_score(y[mask], y_pred[mask], zero_division=0))
            recalls.append(recall_score(y[mask], y_pred[mask], zero_division=0))
            counts.append(mask.sum())
    
    # Plot metrics
    x = range(len(categories))
    plt.bar(x, precisions, alpha=0.5, label='Precision')
    plt.bar(x, recalls, alpha=0.5, label='Recall')
    
    plt.title(f'Model Performance Across {feature}')
    plt.xlabel(feature)
    plt.ylabel('Score')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True)

def analyze_geographic_performance(X, y, y_pred):
    """
    Analyze model performance across geographic regions
    """
    # Create geographic bins using lat/long
    lat_bins = pd.qcut(X['lat'], q=5, labels=['Very South', 'South', 'Central', 'North', 'Very North'])
    long_bins = pd.qcut(X['long'], q=5, labels=['Far West', 'West', 'Central', 'East', 'Far East'])
    
    # Calculate performance metrics for each region
    performance_data = []
    for lat_bin in lat_bins.unique():
        for long_bin in long_bins.unique():
            mask = (lat_bins == lat_bin) & (long_bins == long_bin)
            if mask.sum() > 0:
                precision = precision_score(y[mask], y_pred[mask], zero_division=0)
                recall = recall_score(y[mask], y_pred[mask], zero_division=0)
                performance_data.append({
                    'lat_bin': lat_bin,
                    'long_bin': long_bin,
                    'performance': (precision + recall) / 2
                })
    
    # Create heatmap
    perf_df = pd.pivot_table(
        pd.DataFrame(performance_data),
        values='performance',
        index='lat_bin',
        columns='long_bin'
    )
    
    sns.heatmap(perf_df, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Model Performance Across Geographic Regions')

def plot_feature_importance_impact(model, feature_names):
    """
    Plot feature importance and their impact on model predictions
    """
    # Get feature importance
    importance = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importance)[-10:]  # Top 10 features
    
    # Plot importance
    plt.barh(range(10), importance[indices])
    plt.yticks(range(10), feature_names[indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Most Important Features')

# Run the analysis
analyze_model_performance_across_features(improved_rf, X, y)

# Print additional insights
print("\nDetailed Performance Analysis:")
print("\n1. Transaction Amount Impact:")
amt_ranges = pd.qcut(X['amt'], q=5)
for amt_range in amt_ranges.unique():
    mask = (amt_ranges == amt_range)
    if mask.sum() > 0:
        precision = precision_score(y[mask], improved_rf.predict(X[mask]), zero_division=0)
        recall = recall_score(y[mask], improved_rf.predict(X[mask]), zero_division=0)
        print(f"\nAmount Range: {amt_range}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")

print("\n2. Hour of Day Impact:")
for hour in sorted(X['hour'].unique()):
    mask = (X['hour'] == hour)
    if mask.sum() > 0:
        precision = precision_score(y[mask], improved_rf.predict(X[mask]), zero_division=0)
        recall = recall_score(y[mask], improved_rf.predict(X[mask]), zero_division=0)
        print(f"\nHour: {hour}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")


# In[29]:


'''
Key Observations:

Transaction Amount Impact:


High Performance Ranges:

Very small transactions (1.01-7.71): Perfect performance (Precision=1.0, Recall=1.0)
Medium-low transactions (7.71-33.06): Perfect performance
Very large transactions (93.89-2753.89): Excellent performance (Precision=1.0, Recall=0.964)


Poor Performance Ranges:

Medium transactions (33.06-61.05 and 61.05-93.89): Zero performance
This suggests a significant blind spot in the model for medium-range transactions




Hour of Day Impact:


High Performance Hours:

Late night/Early morning (1-3 AM): Perfect detection
Mid-day hours (11 AM, 1 PM): Perfect detection
Afternoon/Evening (3-4 PM, 10-11 PM): Perfect or near-perfect detection


Poor Performance Hours:

Early morning (4-10 AM): Zero detection
Various hours throughout the day (12 PM, 2 PM, 5-7 PM, 8-9 PM): Zero detection




Overall Pattern Analysis:


The model shows extreme performance patterns (either perfect or zero)
Strong performance at transaction extremes (very low or very high amounts)
Better performance during non-business hours
Inconsistent performance during peak business hours

Recommendations:

Model Improvements:

Focus on improving detection in medium transaction ranges (33-94)
Enhance performance during business hours
Consider developing separate models for different time periods
Implement ensemble approaches for more consistent performance


Data Collection:

Gather more fraud examples in medium transaction ranges
Obtain additional features for transactions during poor-performance hours
Consider collecting more contextual data for business-hour transactions


Risk Management:

Implement additional verification for medium-range transactions
Enhanced monitoring during hours with poor model performance
Consider different thresholds for different time periods


Business Implementation:

Higher scrutiny for transactions in the 33-94 range
Additional verification steps during business hours
Different fraud detection strategies for different times of day



Limitations and Concerns:

The extreme performance pattern (0 or 1) suggests possible overfitting
Large gaps in performance across hours indicate potential instability
The model might be too specialized for extreme cases
Business hours performance needs significant improvement
'''
print()


# In[ ]:





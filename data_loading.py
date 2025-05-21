import pandas as pd

# Load the dataset
df = pd.read_csv('telco_customer_churn.csv')

# Display the first few rows
print(df.head())

# Check the number of rows and columns
print(f"Dataset shape: {df.shape}")

# Display column names and data types
print("\nColumn Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Get summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())
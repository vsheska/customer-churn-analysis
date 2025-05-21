import pandas as pd

# Load the dataset
df = pd.read_csv('telco_customer_churn.csv')

# Display the first few rows
print(df.head())

# Check the number of rows and columns
print(f"Dataset shape: {df.shape}")

# Find non-numeric values in 'TotalCharges'
non_numeric_values = df[~df['TotalCharges'].str.replace(" ", "").str.isnumeric()]['TotalCharges'].unique()

# Print unique non-numeric values
print("\nNon-numeric values in 'TotalCharges':")
print(non_numeric_values)

# Convert 'TotalCharges' to numeric, forcing errors='coerce' to handle any non-numeric values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Display column names and data types
print("\nColumn Info:")
print(df.info())

# Check if there are any missing values after conversion
print("\nMissing values after conversion:")
print(df.isnull().sum())

# Get summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())
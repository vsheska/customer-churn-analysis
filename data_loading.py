import pandas as pd

# Load the dataset
df = pd.read_csv('telco_customer_churn.csv')

# Display the first few rows
print(df.head())

# Check the number of rows and columns
print(f"Dataset shape: {df.shape}")

# Find rows where 'TotalCharges' is not numeric
non_numeric_values = df[pd.to_numeric(df['TotalCharges'], errors='coerce').isna()]['TotalCharges'].unique()

# Print unique non-numeric values
print("\nNon-numeric values in 'TotalCharges':")
print(non_numeric_values)

# Convert 'TotalCharges' to numeric, replacing spaces with NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')

# Check if there are missing values after conversion
print("\nMissing values in 'TotalCharges' after conversion:")
print(df['TotalCharges'].isnull().sum())

# Display column names and data types
print("\nColumn Info:")
print(df.info())

# Check if there are any missing values after conversion
print("\nMissing values after conversion:")
print(df.isnull().sum())

# Get summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())

# Option 1: Fill missing values in 'TotalCharges' with the median
df_filled = df.copy()
df_filled['TotalCharges'] = df_filled['TotalCharges'].fillna(df_filled['TotalCharges'].median())

# Option 2: Drop rows with missing 'TotalCharges'
df_dropped = df.copy()
df_dropped = df_dropped.dropna(subset=['TotalCharges'])

# Check missing values for both options
print("\nMissing values in 'TotalCharges' after imputation:")
print(df_filled['TotalCharges'].isnull().sum())

print("\nMissing values in 'TotalCharges' after dropping rows:")
print(df_dropped['TotalCharges'].isnull().sum())
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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

# Plot box plots for numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(2, len(numerical_columns) // 2 + 1, i)
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.savefig('plots/boxplots.png')  # Save the boxplots as a PNG file
plt.close()  # Close the plot to prevent it from displaying

# Plot histograms for numerical columns
plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(2, len(numerical_columns) // 2 + 1, i)
    df[col].hist(bins=20)
    plt.title(f"Histogram of {col}")
plt.tight_layout()
plt.savefig('plots/histograms.png')  # Save the histograms as a PNG file
plt.close()  # Close the plot to prevent it from displaying

# Use the cleaned dataset (either df_filled or df_dropped)
df_cleaned = df_filled.drop(columns=['customerID'])

# Identify categorical columns
categorical_columns = df_cleaned.select_dtypes(include=['object']).columns

# One-hot encoding of categorical variables
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_columns, drop_first=True)

# Check the resulting dataframe
print(df_encoded.head())
print(f"Number of columns after encoding: {df_encoded.shape[1]}")

# Define the target variable (e.g., 'Churn') and feature variables
X = df_encoded.drop(columns=['Churn'])  # Features (all columns except 'Churn')
y = df_encoded['Churn']  # Target variable (Churn)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the resulting datasets
print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Test target shape: {y_test.shape}")
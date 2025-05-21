import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def handle_missing_values(df, fill=True):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
    if fill:
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    else:
        df = df.dropna(subset=['TotalCharges'])
    return df

def drop_columns(df, columns):
    return df.drop(columns=columns)

def scale_data(df, scaler, fit=False):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if fit:
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    else:
        df[numerical_columns] = scaler.transform(df[numerical_columns])
    return df
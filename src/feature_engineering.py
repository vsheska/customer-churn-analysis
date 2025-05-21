import pandas as pd

def encode_categorical_columns(df):
    return pd.get_dummies(df, drop_first=True)

def convert_to_category(df, column):
    df[column] = df[column].astype('category')
    return df
import pandas as pd

def encode_categorical_columns(df):
    return pd.get_dummies(df, drop_first=True)

def convert_to_category(df, column):
    df[column] = df[column].astype('category')
    return df

def create_advanced_features(df):
    df_new = df.copy()

    # Customer Behavior Features
    df_new['TotalServices'] = calculate_total_services(df_new)
    df_new['HasPhoneAndInternet'] = ((df_new['PhoneService'] == 'Yes') &
                                     (df_new['InternetService'] != 'No')).astype(int)
    df_new['StreamingServices'] = create_streaming_score(df_new)
    return df_new

def calculate_total_services(df):
    """Calculate the total number of services per customer"""
    services = ['PhoneService', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']

    return df[services].apply(lambda x: (x != 'No').sum(), axis=1)

def create_streaming_score(df):
    """Create a score for streaming services"""
    streaming_services = ['StreamingTV', 'StreamingMovies']
    return df[streaming_services].apply(lambda x: (x == 'Yes').sum(), axis=1)

import pandas as pd
import numpy as np

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

    # Service Usage Features
    df_new['SecurityServices'] = create_security_score(df_new)

    # Financial Features
    df_new['AvgMonthlyCharges'] = df_new['TotalCharges'].astype(float) / df_new['tenure'].replace(0, 1)
    df_new['ChargePerService'] = df_new['MonthlyCharges'] / df_new['TotalServices'].replace(0, 1)
    df_new['ContractValue'] = calculate_contract_value(df_new)

    # Temporal Features
    df_new['CustomerAge'] = create_customer_age_categories(df_new)
    df_new['ContractRisk'] = create_contract_risk_score(df_new)

    # Interaction Features
    df_new['ServiceInteraction'] = create_service_interaction(df_new)
    df_new['FinancialRisk'] = create_financial_risk_score(df_new)

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

def create_security_score(df):
    """Create a score for security services"""
    security_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection']
    return df[security_services].apply(lambda x: (x == 'Yes').sum(), axis=1)

def calculate_contract_value(df):
    """Calculate total contract value"""
    # Convert tenure to months based on contract types
    contract_multiplier = {
        'Month-to-month': 1,
        'One year': 12,
        'Two year': 24
    }

    return df.apply(lambda x: x['MonthlyCharges'] *
                              contract_multiplier.get(x['Contract'], 1), axis=1)

def create_customer_age_categories(df):
    """Categorize customers based on tenure"""
    bins = [0, 12, 24, 48, float('inf')]
    labels = ['New', 'Developing', 'Mature', 'Loyal']
    return pd.cut(df['tenure'], bins=bins, labels=labels, right=False)

def create_contract_risk_score(df):
    """Create risk scores based on contract type and tenure"""
    contract_risk = {
        'Month-to-month': 3,
        'One year': 2,
        'Two year': 1
    }

    tenure_factor = np.where(df['tenure'] <= 12, 2,
                             np.where(df['tenure'] <= 24, 1.5, 1))

    return df['Contract'].map(contract_risk) * tenure_factor

def create_service_interaction(df):
    """Create interaction scores between different services"""
    return (df['TotalServices'] *
            (df['Contract'] != 'Month-to-month').astype(int) *
            (df['PaperlessBilling'] == 'Yes').astype(int))


def create_financial_risk_score(df):
    """Create financial risk scores based on payment method and charges"""
    payment_risk = {
        'Electronic check': 3,
        'Mailed check': 2,
        'Bank transfer (automatic)': 1,
        'Credit card (automatic)': 1
    }

    payment_factor = df['PaymentMethod'].map(payment_risk)
    charge_factor = pd.qcut(df['MonthlyCharges'], q=3, labels=[1, 2, 3])

    return payment_factor * charge_factor

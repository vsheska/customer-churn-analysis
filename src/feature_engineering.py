import pandas as pd
import numpy as np

def convert_to_category(df, column):
    df[column] = df[column].astype('category')
    return df

def create_advanced_features(df):
    df_new = df.copy()

    # Service Count (Ordinal)
    df_new['TotalServices'] = pd.Categorical(
        calculate_total_services(df_new),
        ordered=True
    )

    # Customer Profile Features (Categorical)
    df_new['CustomerAge'] = create_customer_age_categories(df_new)
    df_new['ServiceUsageProfile'] = create_service_usage_profile(df_new)
    df_new['ServiceLevel'] = create_service_level(df_new) # Requires TotalServices

    # Risk Assessment Features (Categorical)
    df_new['ContractRisk'] = create_contract_risk_score(df_new)
    df_new['FinancialRisk'] = create_financial_risk_score(df_new)

    # Financial Metrics (Numerical)
    df_new['AvgMonthlyCharges'] = df_new['TotalCharges'].astype(float) / df_new['tenure'].replace(0, 1)
    df_new['ContractValue'] = calculate_contract_value(df_new)

    return df_new

def encode_categorical_columns(df):
    """
    Enhanced encoding function that handles ordinal and nominal categories appropriately
    """
    df_encoded = df.copy()

    # Identify ordinal (ordered) categorical columns
    ordinal_columns = [
        'CustomerAge',
        'ServiceLevel',
        'ContractRisk',
        'FinancialRisk',
        'ServiceUsageProfile'
    ]

    # Identify nominal (unordered) categorical columns
    categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns
    nominal_columns = [col for col in categorical_columns if col not in ordinal_columns]

    # Encode ordinal columns while preserving order
    for col in ordinal_columns:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].cat.codes

    # One-hot encode nominal columns
    df_encoded = pd.get_dummies(df_encoded, columns=nominal_columns, drop_first=True)

    return df_encoded

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
    """
    Create categorical contract risk based on contract type and tenure
    Returns an ordered categorical variable
    """

    def categorize_contract_risk(row):
        contract = row['Contract']
        tenure = row['tenure']

        if contract == 'Month-to-month':
            if tenure <= 12:
                return 'Very High Risk'
            else:
                return 'High Risk'
        elif contract == 'One year':
            if tenure <= 12:
                return 'High Risk'
            elif tenure <= 24:
                return 'Medium Risk'
            else:
                return 'Low Risk'
        else:  # Two years
            if tenure <= 12:
                return 'Medium Risk'
            else:
                return 'Low Risk'

    risk_categories = df.apply(categorize_contract_risk, axis=1)

    return pd.Categorical(
        risk_categories,
        categories=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'],
        ordered=True
    )


def create_service_level(df):
    """
    Replace create_service_interaction with a categorical service level indicator
    """

    def categorize_service_level(row):
        total_services = row['TotalServices']
        is_long_term = row['Contract'] != 'Month-to-month'
        is_paperless = row['PaperlessBilling'] == 'Yes'

        if total_services >= 6 and is_long_term:
            return 'Premium'
        elif total_services >= 4 and (is_long_term or is_paperless):
            return 'Advanced'
        elif total_services >= 2:
            return 'Standard'
        else:
            return 'Basic'

    service_levels = df.apply(categorize_service_level, axis=1)

    return pd.Categorical(
        service_levels,
        categories=['Basic', 'Standard', 'Advanced', 'Premium'],
        ordered=True
    )


def create_service_usage_profile(df):
    """
    Create a categorical service usage profile combining different service types
    Replaces individual service scores with a more meaningful category
    """

    def categorize_usage(row):
        has_security = any(row[service] == 'Yes'
                           for service in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection'])
        has_streaming = any(row[service] == 'Yes'
                            for service in ['StreamingTV', 'StreamingMovies'])
        has_tech = row['TechSupport'] == 'Yes'

        if has_security and has_streaming and has_tech:
            return 'Full Suite User'
        elif (has_security and has_streaming) or (has_security and has_tech) or (has_streaming and has_tech):
            return 'Multi-Service User'
        elif has_security or has_streaming or has_tech:
            return 'Single-Service User'
        else:
            return 'Basic User'

    usage_profiles = df.apply(categorize_usage, axis=1)

    return pd.Categorical(
        usage_profiles,
        categories=['Basic User', 'Single-Service User', 'Multi-Service User', 'Full Suite User'],
        ordered=True
    )


def create_financial_risk_score(df):
    """Create categorical financial risk scores based on payment method and charges"""
    # Define payment method categories
    payment_risk = {
        'Electronic check': 'High',
        'Mailed check': 'Medium',
        'Bank transfer (automatic)': 'Low',
        'Credit card (automatic)': 'Low'
    }

    # Create charge level categories
    charge_levels = pd.qcut(df['MonthlyCharges'],
                            q=3,
                            labels=['Low', 'Medium', 'High'])

    # Create risk categories based on the combination of payment method and charge level
    def categorize_risk(row):
        payment = payment_risk[row['PaymentMethod']]
        charge = row['ChargeLevel']

        if payment == 'High' and charge == 'High':
            return 'Very High Risk'
        elif payment == 'High' or charge == 'High':
            return 'High Risk'
        elif payment == 'Medium' and charge == 'Medium':
            return 'Medium Risk'
        elif payment == 'Low' and charge == 'Low':
            return 'Low Risk'
        else:
            return 'Moderate Risk'

    # Add charge levels as an intermediate step
    df = df.copy()
    df['ChargeLevel'] = charge_levels

    # Create categorical risk scorse
    risk_categories = df.apply(categorize_risk, axis=1)

    # Convert to ordered categorical
    return pd.Categorical(risk_categories,
                          categories=['Low Risk', 'Moderate Risk', 'Medium Risk',
                                      'High Risk', 'Very High Risk'],
                          ordered=True)


from src.data_preprocessing import load_data, handle_missing_values, drop_columns, scale_data
from src.feature_engineering import encode_categorical_columns, convert_to_category
from src.model_training import train_logistic_regression, evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    # Step 1: Load data
    df = load_data('data/telco_customer_churn.csv')

    # Step 2: Preprocess data
    df = handle_missing_values(df)
    df = drop_columns(df, ['customerID'])

    # Step 3: Feature Engineering
    df = encode_categorical_columns(df)
    df = convert_to_category(df, 'SeniorCitizen')

    # Step 4: Train-Test Split
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Scale Data
    scaler = StandardScaler()
    X_train = scale_data(X_train, X_train.columns, scaler)
    X_test = scale_data(X_test, X_test.columns, scaler)

    # Step 6: Train Model
    model = train_logistic_regression(X_train, y_train)

    # Step 7: Evaluate Model
    accuracy, conf_matrix, class_report = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

if __name__ == "__main__":
    main()
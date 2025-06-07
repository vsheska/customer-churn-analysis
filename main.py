from src.data_preprocessing import load_data, handle_missing_values, drop_columns, scale_data
from src.feature_engineering import encode_categorical_columns, convert_to_category, create_advanced_features
from src.model_training import train_logistic_regression
from src.model_evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve, compute_auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.plots import plot_histogram, plot_boxplots, save_plot

def main():
    # Load data
    df = load_data('data/telco_customer_churn.csv')

    # Preprocess data
    df = handle_missing_values(df)
    df = drop_columns(df, ['customerID'])

    # Feature Engineering
    df = create_advanced_features(df)

    # Encode categorical columns
    df = convert_to_category(df, 'SeniorCitizen')
    df = encode_categorical_columns(df)

    # Generate Plots
    plot_boxplots(df)
    plot_histogram(df)

    # Train-Test Split
    df.rename(columns={'Churn_Yes': 'Churn'}, inplace=True)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale Data
    scaler = StandardScaler()
    X_train = scale_data(X_train, scaler, fit=True)
    X_test = scale_data(X_test, scaler, fit=False)

    # Train Model
    model = train_logistic_regression(X_train, y_train)

    # Evaluate Model
    y_pred, accuracy, conf_matrix, class_report = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, save_plot)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_proba, save_plot)
        auc_score = compute_auc(y_test, y_proba)
        print(f"AUC Score: {auc_score:.4f}")

if __name__ == "__main__":
    main()
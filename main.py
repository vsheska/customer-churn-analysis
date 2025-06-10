from src.data_preprocessing import load_data, handle_missing_values, drop_columns, scale_data
from src.feature_engineering import encode_categorical_columns, convert_to_category, create_advanced_features
from src.model_evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve, compute_auc, \
    plot_feature_importance, plot_cv_results
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.plots import plot_histogram, plot_boxplots, save_plot, plot_correlation_matrix, plot_churn_correlations
from src.model_selection import ModelSelector

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
    plot_correlation_matrix(df, threshold=0.1)  # Only show correlations >= 0.1
    plot_churn_correlations(df)  # Show correlations with Churn
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

    # Initialize model selector
    model_selector = ModelSelector(random_state=42)

    # Evaluate all models
    print("\nEvaluating different models...")
    model_results = model_selector.evaluate_models(X_train, y_train)
    print("\nModel Comparison Results:")
    print(model_results)

    # Find the best performing model based on F1 score
    best_model_name = model_results.sort_values('f1_mean', ascending=False).iloc[0]['model']
    print(f"\nBest performing model: {best_model_name}")

    # Tune the best model
    print(f"\nTuning {best_model_name}...")
    tuning_results = model_selector.tune_model(X_train, y_train, best_model_name)

    # Get the final tuned model
    final_model = tuning_results['best_model']
    print(f"\nBest parameters for {best_model_name}:")
    print(tuning_results['best_params'])

    if hasattr(final_model, 'feature_importances_'):
        plot_feature_importance(final_model, X_train.columns, save_plot)

    # Add cross-validation visualization
    from sklearn.model_selection import cross_validate
    cv_results = cross_validate(final_model, X_train, y_train,
                                cv=5, scoring='accuracy',
                                return_train_score=True)
    plot_cv_results(cv_results, save_plot)

    # Evaluate the final model
    print("\nFinal Model Evaluation:")
    y_pred, accuracy, conf_matrix, class_report = evaluate_model(final_model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, save_plot)

    # ROC curve and AUC score
    if hasattr(final_model, "predict_proba"):
        y_proba = final_model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_proba, save_plot)
        auc_score = compute_auc(y_test, y_proba)
        print(f"AUC Score: {auc_score:.4f}")


if __name__ == "__main__":
    main()

# Customer Churn Analysis
This project analyzes and predicts customer churn in a telecom dataset using various machine learning models.

## Project Overview
The project includes:
- Feature engineering and data preprocessing
- Model selection and evaluation of multiple algorithms:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - SVM
- Hyperparameter tuning using RandomizedSearchCV
- Performance visualization using confusion matrices and ROC curves

## Features
- Automated model selection pipeline
- Feature engineering including:
  - Contract value calculation
  - Average monthly charges
  - Numerical feature scaling
  - Categorical feature encoding
- Model evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score 
  - ROC AUC

## Project Structure
- `data/` : Contains the input dataset
  - 'telco_customer_churn.csv' : Telecom customer data
- `src/`
    - `data_preprocessing.py` : Data cleaning and imputation
    - `feature_engineering.py` : Composite feature creation
    - `model_selection.py` : Model comparison and hyperparameter tuning
    - `model_evaluation.py` : Performance metrics and visualization
- `outputs/plots/`: Generated visualizations
    - Feature distributions
    - Confusion matrix
    - ROC curve
    - Feature importance plots
    - Cross-validation results
    - Correlation matrices
    - Churn correlation analysis

## Current Results
Best performing model: Logistic Regression
- Accuracy: 81.90%
- ROC AUC Score: 0.8596

## Output Visualizations
The project generates several visualizations in the `outputs/plots/` directory:
- Feature distribution histograms
- Boxplots for numerical features
- Confusion matrix
- ROC curve
- Feature importance plots
- Cross-validation score distributions
- Correlation analysis:
  - Feature correlation matrix
  - Churn correlation analysis


## Setup
1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

## Run the Project
```bash
python main.py
```

## Run the Dashboard
```bash
streamlit run src/dashboard.py
```

## Future Improvements
- [x] Feature importance analysis
- [x] Cross-validation results visualization
- [x] Enhanced correlation analysis
- [ ] Model interpretability (SHAP/LIME)
- [ ] Advanced feature selection
- [ ] Model deployment pipeline
- [x] Interactive dashboard
- [ ] Automated model retraining
- [ ] Performance monitoring system
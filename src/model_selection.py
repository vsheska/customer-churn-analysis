# src/model_selection.py

import numpy as np
from sklearn.model_selection import (
    cross_validate,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    make_scorer
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pandas as pd


class ModelSelector:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'logistic': LogisticRegression(
                random_state=random_state,
                max_iter=1000),
            'random_forest': RandomForestClassifier(
                random_state=random_state),
            'gradient_boost': GradientBoostingClassifier(
                random_state=random_state),
            'xgboost': XGBClassifier(
                random_state=random_state),
            'svm': SVC(
                random_state=random_state,
                probability=True)
        }

        self.param_grids = {
            'logistic': {
                'C': np.logspace(-4, 4, 20),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [10000]
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 4, 5],
                'min_child_weight': [1, 3, 5]
            },
            'svm': {
                'C': np.logspace(-3, 3, 7),
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto'] + list(np.logspace(-3, 0, 4))
            }
        }

    def evaluate_models(self, X, y, cv=5):
        """
        Evaluate all models using cross-validation
        """
        results = []
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score),
            'roc_auc': make_scorer(roc_auc_score)
        }

        for name, model in self.models.items():
            cv_results = cross_validate(
                model, X, y,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
                scoring=scoring,
                return_train_score=True
            )

            # Calculate mean and std for each metric
            metrics = {}
            for metric in scoring.keys():
                test_scores = cv_results[f'test_{metric}']
                metrics[f'{metric}_mean'] = test_scores.mean()
                metrics[f'{metric}_std'] = test_scores.std()

            results.append({
                'model': name,
                **metrics
            })

        return pd.DataFrame(results)

    def tune_model(self, X, y, model_name, cv=5):
        """
        Perform hyperparameter tuning for a specific model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        param_grid = self.param_grids[model_name]

        # Define scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score),
            'roc_auc': make_scorer(roc_auc_score)
        }

        # Perform randomized search
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=20,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring=scoring,
            refit='f1',  # Optimize for F1 score
            random_state=self.random_state,
            n_jobs=-1
        )

        search.fit(X, y)

        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_model': search.best_estimator_
        }

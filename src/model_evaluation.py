import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    auc
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model and prints key metrics.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    return y_pred, accuracy, conf_matrix, class_report


def plot_confusion_matrix(y_true, y_pred, save_plot=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    if save_plot:
        save_plot(fig, 'confusion_matrix')
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, save_plot=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')

    if save_plot:
        save_plot(fig, 'roc_curve')
    plt.close(fig)


def compute_auc(y_true, y_proba):
    return roc_auc_score(y_true, y_proba)


def plot_feature_importance(model, feature_names, save_plot_func):
    """
    Plots feature importance for tree-based models.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)),
                   [feature_names[i] for i in indices],
                   rotation=45,
                   ha='right')
        plt.tight_layout()
        save_plot_func(plt.gcf(), 'feature_importance')
        plt.close()

def plot_cv_results(cv_results, save_plot_func):
    """
    Visualizes cross-validation scores across different models.
    """
    plt.figure(figsize=(10, 6))
    scores = cv_results['test_score']
    plt.boxplot(scores)
    plt.title('Cross-validation Scores Distribution')
    plt.ylabel('Score')
    save_plot_func(plt.gcf(), 'cv_scores')
    plt.close()
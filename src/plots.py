import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


def save_plot(fig, plot_name, folder='outputs/plots'):
    """
    Saves the plot to the specified folder with the given name.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    plot_path = os.path.join(folder, f'{plot_name}.png')
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


def plot_histogram(df, bins=20):
    """
    Generates a histogram for a specified column in the DataFrame.
    """
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

    for column in numerical_columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[column], bins=bins, kde=True, ax=ax)
        ax.set_title(f'{column} Distribution')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        save_plot(fig, f'histogram_{column}')
        plt.close(fig)


def plot_boxplots(df):
    """
    Generates boxplots for numerical columns in the DataFrame
    """
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

    fig, axes = plt.subplots(1, len(numerical_columns), figsize=(6 * len(numerical_columns), 6))

    # Ensure axes is iterable
    if len(numerical_columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, numerical_columns):
        sns.boxplot(data=df, x=col, ax=ax)
        ax.set_title(f'Boxplot of {col}')

    fig.tight_layout()
    save_plot(fig, 'boxplots')
    plt.close(fig)


def plot_correlation_matrix(df, threshold=0.1):
    """
    Generates a heatmap of the correlation matrix with improvements:
    - Shows only numerical columns
    - Filters weak correlations
    - Highlights correlations with target variable (Churn)

    Args:
        df: DataFrame containing the data
        threshold: Minimum absolute correlation value to display (default: 0.1)
    """
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numerical_cols].corr()

    # Create mask for weak correlations
    mask = np.abs(correlation_matrix) < threshold
    correlation_matrix_filtered = correlation_matrix.mask(mask, 0)

    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix_filtered,
                annot=True,
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                fmt='.2f')

    plt.title('Feature Correlation Matrix\n(correlations >= {})'.format(threshold))
    plt.tight_layout()
    save_plot(plt.gcf(), 'correlation_matrix')
    plt.close()


def plot_churn_correlations(df):
    """
    Plots correlations specifically with the target variable (Churn)
    """
    # Calculate correlations with Churn
    correlations = df.corr()['Churn_Yes'].sort_values(ascending=False)

    # Filter out weak correlations
    correlations = correlations[abs(correlations) >= 0.1]

    # Create the plot
    plt.figure(figsize=(10, 6))
    correlations.plot(kind='bar')
    plt.title('Feature Correlations with Churn')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(plt.gcf(), 'churn_correlations')
    plt.close()


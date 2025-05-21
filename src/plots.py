import matplotlib.pyplot as plt
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


def plot_correlation_matrix(df):
    """
    Generates a heatmap of the correlation matrix.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')

    save_plot(fig, 'correlation_matrix')
    plt.close(fig)
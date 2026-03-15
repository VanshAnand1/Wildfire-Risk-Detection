import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import data as wildfire_data
import model as wildfire_model


FEATURE_COLUMNS = [
    'temperature_c',
    'humidity_pct',
    'wind_speed_kmh',
    'rainfall_last_day_mm',
    'fire_last_year',
    'fire_last_5_years',
    'fire_ever',
]

def _load_preprocessed_dataframe(dataSize='large'):
    if dataSize == 'small':
        fires_df = wildfire_data.loadSmallData().copy()
    else:
        fires_df = wildfire_data.loadLargeData().copy()

    yes_no_map = {'Yes': 1, 'No': 0}
    binary_cols = ['fire_last_year', 'fire_last_5_years', 'fire_ever', 'fire_occurred']
    for col in binary_cols:
        fires_df[col] = fires_df[col].map(yes_no_map)

    return fires_df


def _train_and_get_test_data(dataSize='large'):
    X_train, X_test, y_train, y_test = wildfire_data.prepareData(dataSize)
    trained_model = wildfire_model.trainModel(X_train, y_train)
    return trained_model, X_test, y_test


def plot_target_balance(dataSize='large'):
    fires_df = _load_preprocessed_dataframe(dataSize)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = fires_df['fire_occurred'].value_counts().sort_index()
    labels = ['No Fire (0)', 'Fire (1)']
    ax.bar(labels, counts.values, color=['#4C78A8', '#F58518'])
    ax.set_title('Target Class Balance')
    ax.set_ylabel('Count')
    plt.tight_layout()
    return fig, ax


def plot_feature_distributions_by_class(dataSize='large'):
    fires_df = _load_preprocessed_dataframe(dataSize)
    features = ['temperature_c', 'humidity_pct', 'wind_speed_kmh', 'rainfall_last_day_mm']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.boxplot(data=fires_df, x='fire_occurred', y=feature, ax=axes[i], palette='Set2')
        axes[i].set_title(f'{feature} by Fire Outcome')
        axes[i].set_xlabel('Fire Occurred (0/1)')

    plt.tight_layout()
    return fig, axes


def plot_correlation_heatmap(dataSize='large'):
    fires_df = _load_preprocessed_dataframe(dataSize)
    numeric_df = fires_df[FEATURE_COLUMNS + ['fire_occurred']]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    return fig, ax


def plot_feature_importance(dataSize='large'):
    trained_model, _, _ = _train_and_get_test_data(dataSize)

    importance_df = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'importance': trained_model.feature_importances_,
    }).sort_values('importance', ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=importance_df, x='importance', y='feature', ax=ax, palette='viridis')
    ax.set_title('Random Forest Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    return fig, ax


def plot_confusion_matrix_and_report(dataSize='large'):
    trained_model, X_test, y_test = _train_and_get_test_data(dataSize)
    y_pred = trained_model.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title('Confusion Matrix')

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().iloc[:-1]
    sns.heatmap(report_df[['precision', 'recall', 'f1-score']], annot=True, fmt='.2f', cmap='YlGnBu', ax=axes[1])
    axes[1].set_title('Classification Report Metrics')

    plt.tight_layout()
    return fig, axes


def plot_predicted_risk_distribution(dataSize='large'):
    trained_model, X_test, _ = _train_and_get_test_data(dataSize)
    risk_scores = trained_model.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(risk_scores, bins=25, kde=True, ax=ax, color='#54A24B')
    ax.set_title('Predicted Wildfire Risk Score Distribution')
    ax.set_xlabel('Predicted Probability of Fire (Class 1)')
    ax.set_ylabel('Count')
    plt.tight_layout()
    return fig, ax


def plot_impact_visuals_when_fire_occurs(dataSize='large'):
    fires_df = _load_preprocessed_dataframe(dataSize)
    fire_df = fires_df[fires_df['fire_occurred'] == 1].copy()

    fire_df['trees_burned'] = pd.to_numeric(fire_df['trees_burned'], errors='coerce')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=fire_df['people_displaced'], ax=axes[0], color='#E45756')
    axes[0].set_title('People Displaced (Fire Cases Only)')
    axes[0].set_ylabel('People Displaced')

    sns.boxplot(y=fire_df['trees_burned'].dropna(), ax=axes[1], color='#72B7B2')
    axes[1].set_title('Trees Burned (Fire Cases Only)')
    axes[1].set_ylabel('Trees Burned')

    plt.tight_layout()
    return fig, axes


def visualizeData(dataSize='large'):
    """Convenience function to generate all available visuals."""
    plot_target_balance(dataSize)
    plot_feature_distributions_by_class(dataSize)
    plot_correlation_heatmap(dataSize)
    plot_feature_importance(dataSize)
    plot_confusion_matrix_and_report(dataSize)
    plot_predicted_risk_distribution(dataSize)
    plot_impact_visuals_when_fire_occurs(dataSize)
    plt.show()
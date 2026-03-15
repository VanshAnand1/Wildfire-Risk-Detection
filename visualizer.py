import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import data

def plotFeatureImportance(trainedModel):
    importance_df = pd.DataFrame({
        'feature': data.FEATURE_COLUMNS,
        'importance': trainedModel.feature_importances_,
    }).sort_values('importance', ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=importance_df, x='importance', y='feature', ax=ax, palette='viridis')
    ax.set_title('Random Forest Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    return fig, ax

def plotConfusionMatrix(trainedModel, X_test, y_test):
    y_pred = trainedModel.predict(X_test)

    fig, ax = plt.subplots(figsize=(5, 5))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix')
    return fig, ax

def plotPredictedRiskDistribution(trainedModel, X_test):
    risk_scores = trainedModel.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(risk_scores, bins=25, kde=True, ax=ax, color='#54A24B')
    ax.set_title('Predicted Wildfire Risk Score Distribution')
    ax.set_xlabel('Predicted Probability of Fire (Class 1)')
    ax.set_ylabel('Count')
    plt.tight_layout()
    return fig, ax

def visualizeData(trainedModel, X_test, y_test):
    plotFeatureImportance(trainedModel)
    plotConfusionMatrix(trainedModel, X_test, y_test)
    plotPredictedRiskDistribution(trainedModel, X_test)
    plt.show()
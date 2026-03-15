# Wildfire-Risk-Detection

## How to run

pip3 install -r requirements.txt

python3 main.py ['small', 'medium', 'large'] (pick one)

The accuracy will be printed in the terminal, and the figures will open automatically.

## Inspiration

Wildfires are increasing in frequency and impact, and early risk signals can help communities prepare sooner. This project was inspired by the idea that simple environmental and historical fire indicators can be used to estimate wildfire risks and prevent harm to communities.

## What it does

Wildfire Risk Detection uses sample historical wildfire-related data to predict whether a fire is likely to occur (Yes/No) for a given set of conditions. It:

- Loads and preprocesses wildfire data
- Converts categorical history fields into model-ready numeric values
- Trains a RandomForestClassifier
- Evaluates model performance with accuracy
- Visualizes model behaviours (feature importance, confusion matrix, risk distribution)

## How we built it

Data handling: pandas for loading and cleaning datasets
Modeling: scikit-learn Random Forest for classification
Pipeline structure: modular files for data prep, model training/evaluation, and visualization
Visualization: matplotlib + seaborn for interpretable analytics plots

## Accomplishments that we're proud of

Built an end-to-end ML workflow from raw CSV to prediction and insights.
Fixed real preprocessing issues that blocked model training.
Added multiple useful visualizations beyond plain accuracy for better model interpretability.
Kept the code modular and readable for future extension.

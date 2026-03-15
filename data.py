import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    'temperature_c',
    'humidity_pct',
    'wind_speed_kmh',
    'rainfall_last_day_mm',
    'fire_last_year',
    'fire_last_5_years',
    'fire_ever',
]

def loadSmallData():
    return pd.read_csv('wildfire_sample_data_small.csv')

def loadMediumData():
    return pd.read_csv('wildfire_sample_data_medium.csv')

def loadLargeData():
    return pd.read_csv('wildfire_sample_data_large.csv')

def prepareData(dataSize):
    if (dataSize == 'small'):
        fires_df = loadSmallData()
    elif (dataSize == 'medium'):
        fires_df = loadMediumData()
    else:
        fires_df = loadLargeData()

    # Convert categorical Yes/No columns to numeric 1/0
    yes_no_map = {'Yes': 1, 'No': 0}
    binary_cols = ['fire_last_year', 'fire_last_5_years', 'fire_ever', 'fire_occurred']
    for col in binary_cols:
        fires_df[col] = fires_df[col].map(yes_no_map)

    # Create features and labels
    X = fires_df[FEATURE_COLUMNS]
    y = fires_df['fire_occurred']  # 1 = fire, 0 = no fire

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test
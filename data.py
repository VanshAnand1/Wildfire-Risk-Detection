import pandas as pd
from sklearn.model_selection import train_test_split
from data import fires_df 

# Load wildfire dataset
fires_df = pd.read_csv('wildfire_sample_data.csv')

# Create features and labels

X = fires_df[['temperature', 'humidity',
              'wind_speed', 'rainfall']]
y = fires_df['fire_occurred']  # 1 = fire, 0 = no fire

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

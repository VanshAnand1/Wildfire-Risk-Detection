# Create features and labels
from sklearn.model_selection import train_test_split
X = fires_df[['temperature', 'humidity',
              'wind_speed', 'rainfall', 'vegetation_index']]
y = fires_df['fire_occurred']  # 1 = fire, 0 = no fire

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

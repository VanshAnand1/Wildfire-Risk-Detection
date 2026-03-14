import pandas as pd

# Load wildfire dataset
fires_df = pd.read_csv('australia_wildfires.csv')
print(fires_df.head())
print(fires_df.info())
print(fires_df.describe())

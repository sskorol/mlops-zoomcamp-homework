import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the parquet file
df = pd.read_parquet('yellow_tripdata_2023-01.parquet')

# Calculate duration and filter as before
df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

# Extract the target variable (duration)
y = df['duration'].values

# Select only the relevant columns - the location IDs
df_locations = df[['PULocationID', 'DOLocationID']]

# Convert the dataframe to a list of dictionaries
dict_list = df_locations.astype(str).to_dict(orient='records')

# Create and fit the DictVectorizer
dv = DictVectorizer(sparse=True)
X = dv.fit_transform(dict_list)

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions on the training data
y_pred = model.predict(X)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"RMSE on training data: {rmse:.4f}")